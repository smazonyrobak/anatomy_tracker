"""Measure the 3D search volume needed around frozen 019 pose proposals."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
out = root / 'runs/atlas_3d_capture_diagnostic_038'
side, beam, samples = 256, 8, 512
normal_radii_um = (1000, 2000, 4000, 6000, 8000)
tangent_radii_um = (1000, 2000, 4000, 6000)
torch.set_num_threads(4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'])
del checkpoint
out.mkdir(parents=True, exist_ok=False)
config = {'purpose': 'development-only search-volume diagnosis, not validation or benchmark',
          'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
          'source_sha256': sha(Path(__file__)), 'eligible_sections': len(records),
          'beam': beam, 'sampled_valid_points_per_section': samples,
          'normal_radii_um': normal_radii_um, 'tangent_radii_um': tangent_radii_um,
          'truth': 'synthetic dense observed tissue-to-CCF mapping',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = prior.topk(beam, -1).indices
        states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(states)
        ids = valid.flatten().nonzero().flatten()
        ids = ids[torch.linspace(0, ids.numel() - 1, min(samples, ids.numel()),
                                 device='cuda').long()]
        chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')), -1).float() / side
        chart = chart[None, None].expand(1, beam, -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                    (side - 1) / side - chart[..., 0], chart[..., 0])
        proposed = centre[..., None, :] + torch.einsum(
            'bkij,bkpj->bkpi', frame[..., :2] @ basis, chart - .5)
        residual = truth.reshape(-1, 3)[ids][None, None] - proposed
        normal = (residual * frame[..., None, :, 2]).sum(-1).abs()[0]
        tangent = (residual.square().sum(-1)[0] - normal.square()).clamp_min(0).sqrt()
        distance = residual.norm(dim=-1)[0].mean(-1)
        for arm, idx in (('top1', 0), ('oracle_best8', int(distance.argmin()))):
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
            row.update({'arm': arm, 'candidate_rank': idx, 'prior_branch': int(choice[0, idx]),
                        'sampled_valid_points': len(ids), 'mean_3d_error_um': float(distance[idx]),
                        'normal_abs_p50_um': float(normal[idx].median()),
                        'normal_abs_p90_um': float(torch.quantile(normal[idx], .9)),
                        'tangent_p50_um': float(tangent[idx].median()),
                        'tangent_p90_um': float(torch.quantile(tangent[idx], .9))})
            for n in normal_radii_um:
                row[f'normal_within_{n}_fraction'] = float((normal[idx] <= n).float().mean())
                for t in tangent_radii_um:
                    row[f'box_n{n}_t{t}_fraction'] = float(
                        ((normal[idx] <= n) & (tangent[idx] <= t)).float().mean())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')

summary = {'sections': len(records), 'synthetic_subjects': len({
    row['synthetic_subject_plan_id'] for row in records}), 'arms': {}}
for arm in ('top1', 'oracle_best8'):
    selected = [row for row in rows if row['arm'] == arm]
    groups = sorted({row['synthetic_subject_plan_id'] for row in selected})
    fields = ('mean_3d_error_um', 'normal_abs_p50_um', 'normal_abs_p90_um',
              'tangent_p50_um', 'tangent_p90_um') + tuple(
                  f'box_n{n}_t{t}_fraction' for n in normal_radii_um for t in tangent_radii_um)
    summary['arms'][arm] = {field: float(np.mean([np.mean([
        row[field] for row in selected if row['synthetic_subject_plan_id'] == group])
        for group in groups])) for field in fields}
    summary['arms'][arm]['sections_with_75pct_coverage_n4000_t4000'] = sum(
        row['box_n4000_t4000_fraction'] >= .75 for row in selected)
    summary['arms'][arm]['sections_with_75pct_coverage_n6000_t6000'] = sum(
        row['box_n6000_t6000_fraction'] >= .75 for row in selected)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows)}, indent=2))
print(json.dumps(summary), flush=True)
