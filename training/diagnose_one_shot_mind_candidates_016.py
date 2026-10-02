"""Frozen 32-branch atlas-MIND ranking probe on synthetic development slices."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_image_information import (
    score_mind_candidates, score_support_penalized_mind_candidates)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

checkpoint_path = root / 'runs/one_shot_observed_coordinate_pose_015/joint_step_08000.pt'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/one_shot_mind_candidates_016'
source_side, render_side = 256, 96
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(records) == 256 and len({row['animal_id'] for row in records}) == 8
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
yy, xx = torch.meshgrid(torch.arange(render_side, device='cuda') / render_side,
                        torch.arange(render_side, device='cuda') / render_side, indexing='ij')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                 255 / source_side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        indices = valid.flatten().nonzero().flatten()
        chart = torch.stack((indices.remainder(source_side),
                             indices.div(source_side, rounding_mode='floor')), -1).float() / source_side
        target = reference.reshape(-1, 3)[indices]
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, model.modes, 2)
        physical = (points(states, flags, chart) - target).norm(dim=-1).mean(-1).flatten().cpu().numpy()
        images, supports = [], []
        for first in range(0, 32, 4):
            branches = torch.arange(first, first + 4, device='cuda')
            state = prediction['state'][0, branches // 2]
            reflected = (branches % 2).bool()
            centre, frame, basis = full_frame_state_to_components(state)
            sx = torch.where(reflected[:, None, None], 255 / source_side - xx, xx)
            chart = torch.stack((sx.expand(-1, render_side, -1),
                                 yy.expand(len(state), -1, -1)), -1)
            plane = normalized_raster_to_ccf(
                centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
            coordinates = (plane[:, None] + offsets[None, :, None, None, None]
                           * frame[:, None, None, None, :, 2])
            rendered = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.),
                weights[None].expand(len(state), -1))
            support = rendered[:, 1:2]
            images.append((rendered[:, :1] / support.clamp_min(1e-4))[:, 0].cpu().double().numpy())
            supports.append((support[:, 0] > .5).cpu().numpy())
        candidates = np.concatenate(images).astype(np.float64, copy=False)
        support = np.concatenate(supports)
        target_image = F.interpolate(image[:, :1], (render_side, render_side),
                                     mode='bilinear', align_corners=False)[0, 0].cpu().double().numpy()
        tissue_prob = F.interpolate(prediction['dense_coordinate'][:, 3:4].sigmoid(),
                                    (render_side, render_side), mode='bilinear',
                                    align_corners=False)[0, 0].cpu().numpy()
        predicted_mask = tissue_prob > .2
        if not predicted_mask.any():
            predicted_mask = tissue_prob >= np.quantile(tissue_prob, .99)
        true_mask = F.adaptive_max_pool2d(valid[None, None].float(),
                                          (render_side, render_side))[0, 0].cpu().numpy() > 0
        whole = np.ones((render_side, render_side), dtype=np.bool_)
        scores = {
            'whole': score_mind_candidates(target_image, candidates, whole, 125.)['scores'],
            'predicted': score_support_penalized_mind_candidates(
                target_image, candidates, predicted_mask, predicted_mask, support, 125.)['scores'],
            'true_mask_upper_bound': score_support_penalized_mind_candidates(
                target_image, candidates, true_mask, true_mask, support, 125.)['scores']}
        row = {**{key: record[key] for key in ('animal_id', 'specimen_id',
                                              'experiment_id', 'section_id', 'sha256')},
               'valid_fraction': float(valid.float().mean()),
               'predicted_mask_fraction': float(predicted_mask.mean()),
               'candidate_atlas_support_fraction': support.mean((1, 2)).tolist(),
               'prior_branch': int(prior.argmax()),
               'prior_rigid_um': float(physical[int(prior.argmax())]),
               'all32_oracle_rigid_um': float(physical.min()),
               'physical_error_um': physical.tolist()}
        for name, score in scores.items():
            chosen = int(np.argmax(score))
            correlation = spearmanr(score, -physical).statistic
            row[f'{name}_branch'] = chosen
            row[f'{name}_rigid_um'] = float(physical[chosen])
            row[f'{name}_score_vs_negative_error_spearman'] = (
                float(correlation) if np.isfinite(correlation) else None)
            row[f'{name}_maximum_score_ties'] = int(np.count_nonzero(score == score.max()))
            row[f'{name}_scores'] = score.tolist()
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'eligible_cases': len(rows)}), flush=True)
    stream.flush()

summary = {}
for name in ('prior', 'whole', 'predicted', 'true_mask_upper_bound', 'all32_oracle'):
    key = f'{name}_rigid_um'
    identities = sorted({row['animal_id'] for row in rows})
    summary[name] = {'plan_equal_mean_um': float(np.mean([
        np.mean([row[key] for row in rows if row['animal_id'] == animal])
        for animal in identities])),
        'under_5_percent_um': float(np.mean([row[key] for row in rows
                                            if row['valid_fraction'] < .05]))}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows),
    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'mind_source_sha256': hashlib.sha256(
        (Path(__file__).parent / 'arbitrary_plane_image_information.py').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows), 'summary': summary}), flush=True)
