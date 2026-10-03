"""Blind fixed-panel readout of complete planes from the 073 correlation head."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_whole_correlation_073 import WholePlaneCorrelationHead

run = root / 'runs/whole_plane_correlation_073_pilot_002'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
control = root / 'runs/normal_fixed_066_development_audit/rows.jsonl'
real_control_path = root / 'runs/joint_pose_correction_072_development_eval/rows.jsonl'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/whole_plane_correlation_073_development_eval_001'
steps = (0, 250, 750, 1500, 2500)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


synthetic_records = [r for r in map(json.loads, (panel / 'records.jsonl').open()) if r['eligible']]
control_rows = {r['section_id']: r for r in map(json.loads, control.open())}
real_control = {r['section_id']: r for r in map(json.loads, real_control_path.open())
                if r['set'] == 'real_weak_allen' and r['step'] == 0}
real_records = [r for r in map(json.loads, (real / 'records.jsonl').open())
                if r['training_split'] == 'development']
assert len(synthetic_records) == 246 and len({r['animal_id'] for r in synthetic_records}) == 8
assert len(real_records) == 64 and len({r['animal_id'] for r in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt',
    map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
head = WholePlaneCorrelationHead().cuda().eval()
normal_shifts = torch.arange(-1500, 1501, 500, device='cuda').float()
rolls = torch.arange(12, device='cuda').float() * (math.pi / 6)
scales = torch.tensor([.65, 1., 1.5], device='cuda')
shifts = torch.arange(-13, 14, device='cuda').float() / 16
chart_axis = (torch.arange(64, device='cuda').float() + .5) / 16 - 2
by, bx = torch.meshgrid(chart_axis, chart_axis, indexing='ij')
chart = torch.stack((bx, by), -1)
psf = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
psf_weights = torch.tensor([1., 2., 2., 2., 1.], device='cuda') / 8
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def branches(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
    state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
    centre, frame, basis = full_frame_state_to_components(state)
    edge = frame[0, :, :, :2] @ basis[0]
    edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
    return prior, choice, centre[0], edge, frame[0, :, :, 2]


def atlas_planes(centre, edge, normal):
    base = centre + torch.einsum('ij,hwj->hwi', edge, chart)
    physical = base[None, None] + normal * (
        normal_shifts[:, None, None, None, None] + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    return rendered, rendered[:, 1:2].clamp(0, 1)


def decode(index):
    tx = index.remainder(27)
    index = index.div(27, rounding_mode='floor')
    ty = index.remainder(27)
    index = index.div(27, rounding_mode='floor')
    scale = index.remainder(3)
    index = index.div(3, rounding_mode='floor')
    angle = index.remainder(12)
    index = index.div(12, rounding_mode='floor')
    offset = index.remainder(7)
    branch = index.div(7, rounding_mode='floor')
    return branch, offset, angle, scale, ty, tx


def map_points(index, xy, centre, edge, normal):
    branch, zi, ai, si, tyi, txi = decode(index)
    q = xy[None] - .5
    co, sn = rolls[ai].cos()[:, None], rolls[ai].sin()[:, None]
    scale = scales[si][:, None]
    chart_xy = torch.stack((shifts[txi][:, None] + scale * (co * q[..., 0] - sn * q[..., 1]),
                            shifts[tyi][:, None] + scale * (sn * q[..., 0] + co * q[..., 1])), -1)
    return centre[branch, None] + normal[branch, None] * normal_shifts[zi, None, None] \
        + torch.einsum('nij,nkj->nki', edge[branch], chart_xy)


def candidate_peaks(logits, centre, edge, normal):
    values, positions = logits.flatten().topk(4096)
    geometry = map_points(positions, corners, centre, edge, normal).cpu().numpy()
    kept = []
    for row in range(len(positions)):
        if all(np.linalg.norm(geometry[row] - geometry[previous], axis=-1).mean() > 250
               for previous in kept):
            kept.append(row)
            if len(kept) == 8:
                break
    return positions[kept], values[kept]


def predict(image, xy):
    prediction = model.predict(image)
    prior, choice, centre, edge, normal = branches(prediction)
    query, mask_logits = head.query_features(prediction)
    scores = []
    for branch in range(14):
        rendered, support = atlas_planes(centre[branch], edge[branch], normal[branch])
        atlas_feature = model.atlas_encoder(rendered)
        scores.append(head.score(atlas_feature, support, query, mask_logits))
    logits = torch.cat(scores, 0)
    positions, values = candidate_peaks(logits, centre, edge, normal)
    mapped = map_points(positions, xy, centre, edge, normal)
    return prior, choice, positions, values, mapped, centre, edge, normal


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = run / f'head_step_{step:05d}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['step'] == step and not saved['calibrated']
        head.load_state_dict(saved['head'], strict=True)
        del saved
        for record in synthetic_records:
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                tissue = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            ids = valid.flatten().nonzero().flatten()
            ids = ids[torch.linspace(0, len(ids) - 1, 256, device='cuda').round().long()]
            xy = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')), -1).float() / side
            target = tissue.reshape(-1, 3)[ids]
            prior, choice, positions, values, mapped, centre, edge, normal = predict(image, xy)
            error = (mapped - target[None]).norm(dim=-1).mean(-1)
            baseline = control_rows[record['section_id']]
            best_branch = decode(positions)[0]
            row = {'set': 'synthetic', 'step': step,
                **{key: record[key] for key in ('animal_id', 'specimen_id',
                    'experiment_id', 'section_id', 'synthetic_subject_plan_id',
                    'sha256', 'appearance_mode')},
                'support_fraction': float(valid.float().mean()),
                'selected_rigid_um': float(error[0]), 'best8_rigid_um': float(error.min()),
                'prior_selected_rigid_um': baseline['selected_rigid_um'],
                'prior_best14_rigid_um': baseline['best14_rigid_um'],
                'selected_branch': int(best_branch[0]), 'selected_score': float(values[0]),
                'candidate_count': len(positions)}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            index = record['array_row_index']
            image = np.concatenate((real_images[index].astype(np.float32),
                                    np.zeros((4, 192, 192), np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            xy = corners
            affine = torch.as_tensor(real_affines[index], device='cuda', dtype=torch.float32)
            target = affine[:, 2] + 192 * xy[:, :1] * affine[:, 0] + 192 * xy[:, 1:] * affine[:, 1]
            prior, choice, positions, values, mapped, centre, edge, normal = predict(image, xy)
            error = (mapped - target[None]).norm(dim=-1).mean(-1)
            best_branch = decode(positions)[0]
            row = {'set': 'real_weak_allen', 'step': step,
                **{key: record[key] for key in ('animal_id', 'specimen_id',
                    'experiment_id', 'section_id')},
                'selected_five_um': float(error[0]), 'best8_five_um': float(error.min()),
                'prior_selected_five_um': real_control[record['section_id']]['prior_five_um'],
                'selected_branch': int(best_branch[0]), 'selected_score': float(values[0]),
                'candidate_count': len(positions)}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'checkpoint': step, 'synthetic_sections': len(synthetic_records),
                          'weak_real_sections': len(real_records)}), flush=True)

summary = []
for step in steps:
    for split in ('synthetic', 'real_weak_allen'):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        identities = sorted({row['animal_id'] for row in group})
        names = [key for key in group[0] if key.endswith('_um')]
        summary.append({'step': step, 'set': split, 'sections': len(group),
            'identities': len(identities), 'identity_equal_mean': {name: float(np.mean([
                np.mean([row[name] for row in group if row['animal_id'] == identity])
                for identity in identities])) for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'checkpoint_sha256': {str(step): sha(run / f'head_step_{step:05d}.pt') for step in steps},
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'control_rows_sha256': sha(control), 'real_control_rows_sha256': sha(real_control_path),
    'source_sha256': sha(Path(__file__)),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
