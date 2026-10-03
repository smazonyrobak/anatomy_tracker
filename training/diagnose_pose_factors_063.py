"""Read-only factor audit of frozen 059 proposals on the 061 synthetic panel."""
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

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/pose_factors_063_development_audit'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [r for r in map(json.loads, (panel / 'records.jsonl').open()) if r['eligible']]
assert len(records) == 246 and len({r['animal_id'] for r in records}) == 8
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
rows = []

with torch.inference_mode():
    for record in records:
        path = panel / record['file']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            truth_reflection = int(arrays['reflection'])
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1,
            (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(state)
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth_state)
        edges = frame[..., :, :2] @ basis
        true_edges = true_frame[..., :, :2] @ true_basis
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                         -1).float()[None] / side
        chart = xy[:, None].expand(-1, len(choice[0]), -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                    255 / side - chart[..., 0], chart[..., 0])
        points = centre[..., None, :] + torch.einsum(
            'bkij,bkpj->bkpi', edges, chart - .5)
        physical = (points - target.reshape(-1, 3)[ids][None, None]).norm(dim=-1).mean(-1)[0]
        normal = torch.rad2deg(torch.acos((frame[0, :, :, 2] @
            true_frame[0, :, 2]).abs().clamp(max=1)))
        position = (centre[0] - true_centre[0]).norm(dim=-1) / 1000
        horizontal = F.normalize(edges[0, :, :, 0] *
            (1 - 2 * reflection[0, :, None].float()), dim=-1)
        vertical = F.normalize(edges[0, :, :, 1], dim=-1)
        true_horizontal = F.normalize(true_edges[0, :, 0] *
            (-1 if truth_reflection else 1), dim=-1)
        true_vertical = F.normalize(true_edges[0, :, 1], dim=-1)
        horizontal_angle = torch.rad2deg(torch.acos(
            (horizontal @ true_horizontal).clamp(-1, 1)))
        vertical_angle = torch.rad2deg(torch.acos(
            (vertical @ true_vertical).clamp(-1, 1)))
        frame_angle = .5 * (horizontal_angle + vertical_angle)
        selected = int(prior.gather(1, choice)[0].argmax())
        best = int(physical.argmin())
        normal_best = int(normal.argmin())
        center_best = int(position.argmin())
        frame_best = int(frame_angle.argmin())
        rows.append({key: record[key] for key in
            ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
             'appearance_mode', 'valid_pixels')})
        rows[-1].update({'visible_fraction': record['valid_pixels'] / (side * side),
            'selected_physical_um': float(physical[selected]),
            'best_physical_um': float(physical[best]),
            'normal_best_physical_um': float(physical[normal_best]),
            'center_best_physical_um': float(physical[center_best]),
            'frame_best_physical_um': float(physical[frame_best]),
            'selected_normal_deg': float(normal[selected]),
            'best_physical_normal_deg': float(normal[best]),
            'nearest_normal_deg': float(normal[normal_best]),
            'selected_center_mm': float(position[selected]),
            'best_physical_center_mm': float(position[best]),
            'nearest_center_mm': float(position[center_best]),
            'selected_frame_deg': float(frame_angle[selected]),
            'best_physical_frame_deg': float(frame_angle[best]),
            'nearest_frame_deg': float(frame_angle[frame_best]),
            'selected_reflection_correct': int(reflection[0, selected]) == truth_reflection,
            'best_physical_reflection_correct': int(reflection[0, best]) == truth_reflection,
            'normal_best_same_as_physical_best': normal_best == best,
            'center_best_same_as_physical_best': center_best == best,
            'frame_best_same_as_physical_best': frame_best == best})

summary = {'sections': len(rows), 'synthetic_identities': 8,
           'real_animal_validation': False, 'public_benchmark_used': False}
animals = sorted({r['animal_id'] for r in rows})
for key in ('selected_physical_um', 'best_physical_um',
            'normal_best_physical_um', 'center_best_physical_um',
            'frame_best_physical_um', 'selected_normal_deg',
            'best_physical_normal_deg', 'nearest_normal_deg',
            'selected_center_mm', 'best_physical_center_mm', 'nearest_center_mm',
            'selected_frame_deg', 'best_physical_frame_deg', 'nearest_frame_deg',
            'selected_reflection_correct', 'best_physical_reflection_correct',
            'normal_best_same_as_physical_best', 'center_best_same_as_physical_best',
            'frame_best_same_as_physical_best'):
    summary[key] = float(np.mean([np.mean([r[key] for r in rows
        if r['animal_id'] == animal]) for animal in animals]))
out.mkdir(parents=True, exist_ok=False)
with parent.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (panel / 'records.jsonl').open('rb') as stream:
    panel_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with Path(__file__).open('rb') as stream:
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'config.json').write_text(json.dumps({'parent_sha256': parent_sha,
    'panel_records_sha256': panel_sha, 'source_sha256': source_sha,
    'candidate_policy': 'old prior top8 + new prior top6, reflection expanded',
    'truth': 'observed-affine synthetic state and valid observed-pixel CCF map',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json'):
    with (out / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'completed.json').write_text(json.dumps({'hashes_sha256': hashes,
    'rows': len(rows), 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
