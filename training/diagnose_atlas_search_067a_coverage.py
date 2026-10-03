"""Truth-only geometric coverage of the frozen 067 blind search family."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
blind = root / 'runs/atlas_image_search_067_pilot_001'
out = root / 'runs/atlas_search_067a_coverage_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = {row['section_id']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
blind_rows = list(map(json.loads, (blind / 'rows.jsonl').open()))
assert len(blind_rows) == 32
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
angles = torch.arange(720, device='cuda').float() * (2 * math.pi / 720)
co, si = angles.cos(), angles.sin()
rows = []

with torch.inference_mode():
    for frozen in blind_rows:
        record = records[frozen['section_id']]
        path = panel / record['file']
        assert sha(path) == record['sha256'] == frozen['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            tissue = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        assert choice.tolist() == [r['choice'] for r in frozen['candidate_results']]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        edge = frame[0, :, :, :2] @ basis[0]
        edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
        normal = frame[0, :, :, 2]
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')),
                         -1).float() / 256
        target = tissue.reshape(-1, 3)[ids]
        target_mean, xy_mean = target.mean(0), xy.mean(0)
        x = xy - xy_mean
        y = target - target_mean
        fitted = []
        for branch in range(14):
            u, v = edge[branch, :, 0], edge[branch, :, 1]
            b0 = x[:, 0, None] * u + x[:, 1, None] * v
            b90 = -x[:, 1, None] * u + x[:, 0, None] * v
            a = (y * b0).sum()
            b = (y * b90).sum()
            c = b0.square().sum()
            d = b90.square().sum()
            e = (b0 * b90).sum()
            dot = co * a + si * b
            norm = co.square() * c + si.square() * d + 2 * co * si * e
            quality = dot.clamp_min(0).square() / norm.clamp_min(1e-9)
            best = int(quality.argmax())
            scale = dot[best].clamp_min(0) / norm[best]
            pred = target_mean + scale * (co[best] * b0 + si[best] * b90)
            error = float((pred - target).norm(dim=-1).mean())
            delta = target_mean - centre[0, branch]
            dz = float(delta @ normal[branch])
            plane_delta = delta - dz * normal[branch]
            shift = torch.linalg.lstsq(edge[branch], plane_delta).solution
            mean_q = xy_mean - .5
            shift -= scale * torch.stack((co[best] * mean_q[0] - si[best] * mean_q[1],
                                          si[best] * mean_q[0] + co[best] * mean_q[1]))
            within = (abs(dz) <= 1750 and shift.abs().max() <= .875
                      and .52 <= float(scale) <= 1.8)
            fitted.append({'branch': branch, 'error_um': error,
                'normal_shift_um': dz, 'shift_xy': shift.tolist(),
                'scale': float(scale), 'angle_rad': float(angles[best]),
                'within_067_ranges': bool(within)})
        rows.append({key: frozen[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256', 'prior_selected_um',
            'blind_selected_um', 'blind_best14_um')})
        rows[-1].update({'truth_best14_family_um': min(r['error_um'] for r in fitted),
            'truth_best14_within_ranges_um': min((r['error_um'] for r in fitted
                if r['within_067_ranges']), default=None), 'fitted': fitted})

identities = sorted({r['synthetic_subject_plan_id'] for r in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
    'truth_only': True, 'public_benchmark_used': False, 'real_animal_validation': False}
for key in ('prior_selected_um', 'blind_selected_um', 'blind_best14_um',
            'truth_best14_family_um'):
    summary[key] = float(np.mean([np.mean([r[key] for r in rows
        if r['synthetic_subject_plan_id'] == identity]) for identity in identities]))
summary['sections_with_in_range_truth_fit'] = sum(r['truth_best14_within_ranges_um'] is not None for r in rows)
summary['mean_truth_best14_within_ranges_um'] = float(np.mean([
    r['truth_best14_within_ranges_um'] for r in rows
    if r['truth_best14_within_ranges_um'] is not None]))

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'blind_rows_sha256': sha(blind / 'rows.jsonl'),
    'source_sha256': sha(Path(__file__)),
    'fit': '720-angle truth-assisted physical SSE minimization in 067 candidate-edge similarity family',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'hashes_sha256': {name: sha(out / name)
        for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
