"""Frozen 32-section blind coarse/fine complete-plane pose comparison."""
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
from training.arbitrary_plane_correlation_fine_074 import refine_one
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_whole_correlation_073 import WholePlaneCorrelationHead

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
checkpoint = root / 'runs/whole_plane_correlation_073_pilot_002/head_step_02500.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
control = root / 'runs/normal_fixed_066_development_audit/rows.jsonl'
out = root / 'runs/correlation_fine_074_dev32'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
identities = sorted({row['synthetic_subject_plan_id'] for row in records})
modes = ('raw', 'exact_black', 'imperfect_brush')
selected = []
for identity_index, identity in enumerate(identities):
    group = [row for row in records if row['synthetic_subject_plan_id'] == identity]
    buckets = {mode: sorted((row for row in group if row['appearance_mode'] == mode),
                            key=lambda row: row['section_id']) for mode in modes}
    selected.extend(buckets[mode][len(buckets[mode]) // 2] for mode in modes)
    selected.append(buckets[modes[identity_index % 3]][-1])
assert len(records) == 246 and len(identities) == 8 and len(selected) == 32
assert len({row['section_id'] for row in selected}) == 32
controls = {row['section_id']: row for row in map(json.loads, control.open())}

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
head = WholePlaneCorrelationHead().cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2500 and not saved['calibrated']
head.load_state_dict(saved['head'], strict=True)
head.requires_grad_(False)
del saved

normal_shifts = torch.arange(-1500, 1501, 500, device='cuda').float()
local_normal = torch.arange(-250, 251, 125, device='cuda').float()
rolls = torch.arange(12, device='cuda').float() * (math.pi / 6)
scales = torch.tensor([.65, 1., 1.5], device='cuda')
shifts = torch.arange(-13, 14, device='cuda').float() / 16
axis = (torch.arange(64, device='cuda').float() + .5) / 16 - 2
by, bx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((bx, by), -1)
psf = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
psf_weights = torch.tensor([1., 2., 2., 2., 1.], device='cuda') / 8
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def atlas_planes(centre, edge, normal, dz):
    base = centre + torch.einsum('ij,hwj->hwi', edge, chart)
    physical = base[None, None] + normal * (dz[:, None, None, None, None]
                                             + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    return model.atlas_encoder(rendered), rendered[:, 1:2].clamp(0, 1)


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


def map_coarse(index, xy, centre, edge, normal):
    branch, zi, ai, si, tyi, txi = decode(index)
    q = xy[None] - .5
    co, sn = rolls[ai].cos()[:, None], rolls[ai].sin()[:, None]
    scale = scales[si][:, None]
    plane = torch.stack((shifts[txi][:, None] + scale * (co * q[..., 0] - sn * q[..., 1]),
                         shifts[tyi][:, None] + scale * (sn * q[..., 0] + co * q[..., 1])), -1)
    return centre[branch, None] + normal[branch, None] * normal_shifts[zi, None, None] \
        + torch.einsum('nij,nkj->nki', edge[branch], plane)


def map_fine(branch, dz, angle, scale, tx, ty, xy, centre, edge, normal):
    q = xy - .5
    plane = torch.stack((tx + scale * (math.cos(angle) * q[:, 0] - math.sin(angle) * q[:, 1]),
                         ty + scale * (math.sin(angle) * q[:, 0] + math.cos(angle) * q[:, 1])), -1)
    return centre[branch] + normal[branch] * dz + plane @ edge[branch].T


def top_distinct(logits, centre, edge, normal):
    _, positions = logits.flatten().topk(4096)
    corners_ccf = map_coarse(positions, corners, centre, edge, normal).cpu().numpy()
    kept = []
    for index in range(len(positions)):
        if all(np.linalg.norm(corners_ccf[index] - corners_ccf[previous], axis=-1).mean() > 250
               for previous in kept):
            kept.append(index)
            if len(kept) == 8:
                break
    assert len(kept) == 8
    return positions[kept]


rows = []
parity = None
with torch.inference_mode():
    for record in selected:
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
        prediction = model.predict(image)
        prior = prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = prior.flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        centre = centre[0]
        edge = frame[0, :, :, :2] @ basis[0]
        edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
        normal = frame[0, :, :, 2]
        query, mask_logits = head.query_features(prediction)
        logits = torch.cat([head.score(*atlas_planes(centre[b], edge[b], normal[b], normal_shifts),
                                       query, mask_logits) for b in range(14)], 0)
        positions = top_distinct(logits, centre, edge, normal)
        coarse_errors = (map_coarse(positions, xy, centre, edge, normal) - target[None]).norm(dim=-1).mean(-1)
        candidates = []
        for candidate_index, position in enumerate(positions):
            branch, zi, ai, si, tyi, txi = (int(v) for v in decode(position))
            dz = normal_shifts[zi] + local_normal
            atlas_features, support = atlas_planes(centre[branch], edge[branch], normal[branch], dz)
            refined, grid_scores = refine_one(head, query, mask_logits, atlas_features, support,
                rolls[ai], scales[si], shifts[txi], shifts[tyi], local_normal, return_scores=True)
            score, dzi, angle, scale, tx, ty = refined
            if parity is None:
                coarse_score = logits.flatten()[position]
                centre_score = grid_scores[2, 3, 2, 6, 6]
                parity = {'coarse_score': float(coarse_score), 'fine_zero_delta_score': float(centre_score),
                          'absolute_difference': float((coarse_score - centre_score).abs())}
                assert torch.isclose(coarse_score, centre_score, rtol=1e-4, atol=1e-3)
            fine_error = (map_fine(branch, dz[dzi], angle, scale, tx, ty,
                                   xy, centre, edge, normal) - target).norm(dim=-1).mean()
            candidates.append({'branch': branch, 'coarse_score': float(logits.flatten()[position]),
                'coarse_rigid_um': float(coarse_errors[candidate_index]),
                'fine_score': score, 'fine_rigid_um': float(fine_error),
                'coarse_normal_offset_um': float(normal_shifts[zi]),
                'fine_normal_offset_um': float(dz[dzi]),
                'coarse_roll_rad': float(rolls[ai]), 'fine_roll_rad': angle,
                'coarse_scale': float(scales[si]), 'fine_scale': scale,
                'coarse_tx': float(shifts[txi]), 'coarse_ty': float(shifts[tyi]),
                'fine_tx': tx, 'fine_ty': ty,
                'outside_coarse_shift_grid': max(abs(tx), abs(ty)) > 13 / 16 + 1e-6,
                'outside_protocol_shift_range': max(abs(tx), abs(ty)) > .8 + 1e-6})
        fine_selected = max(range(8), key=lambda index: candidates[index]['fine_score'])
        baseline = controls[record['section_id']]
        rows.append({'set': 'synthetic', **{key: record[key] for key in (
            'animal_id', 'specimen_id', 'experiment_id', 'section_id',
            'synthetic_subject_plan_id', 'appearance_mode', 'sha256')},
            'coarse_selected_rigid_um': float(coarse_errors[0]),
            'coarse_best8_rigid_um': float(coarse_errors.min()),
            'fine_selected_rigid_um': candidates[fine_selected]['fine_rigid_um'],
            'fine_best8_rigid_um': min(item['fine_rigid_um'] for item in candidates),
            'prior_selected_rigid_um': baseline['selected_rigid_um'],
            'prior_best14_rigid_um': baseline['best14_rigid_um'],
            'fine_selected_index': fine_selected, 'candidates': candidates})
        if len(rows) % 8 == 0:
            print(json.dumps({'event': 'sections_scored', 'sections': len(rows)}), flush=True)

summary = {'sections': len(rows), 'synthetic_identities': len(identities),
    'appearance_counts': {mode: sum(row['appearance_mode'] == mode for row in rows) for mode in modes},
    'identity_equal_mean_um': {}, 'parity': parity,
    'fine_candidates_outside_coarse_shift_grid': sum(candidate['outside_coarse_shift_grid']
        for row in rows for candidate in row['candidates']),
    'fine_candidates_outside_protocol_shift_range': sum(candidate['outside_protocol_shift_range']
        for row in rows for candidate in row['candidates']),
    'calibrated': False, 'public_benchmark_used': False, 'truth_in_candidate_selection': False}
for key in ('prior_selected_rigid_um', 'prior_best14_rigid_um',
            'coarse_selected_rigid_um', 'coarse_best8_rigid_um',
            'fine_selected_rigid_um', 'fine_best8_rigid_um'):
    summary['identity_equal_mean_um'][key] = float(np.mean([
        np.mean([row[key] for row in rows if row['synthetic_subject_plan_id'] == identity])
        for identity in identities]))

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({'parent_sha256': sha(parent),
    'head_sha256': sha(checkpoint), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'control_rows_sha256': sha(control), 'source_sha256': sha(Path(__file__)),
    'fine_source_sha256': sha(Path(refine_one.__code__.co_filename)),
    'section_ids': [row['section_id'] for row in selected],
    'normal_offsets_um': local_normal.tolist(), 'local_roll_step_degrees': 5,
    'local_roll_limits_degrees': [-15, 15],
    'scale_multipliers': [.8, .9, 1., 1.1, 1.2],
    'translation_step_chart': .025, 'translation_limits_from_coarse_chart': [-.15, .15],
    'coarse_grid_shift_max_chart': 13 / 16, 'protocol_shift_max_chart': .8,
    'fine_shift_unclamped': True, 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'hashes_sha256': {name: sha(out / name)
        for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(rows),
                  'identity_equal_mean_um': summary['identity_equal_mean_um'],
                  'parity': parity}), flush=True)
