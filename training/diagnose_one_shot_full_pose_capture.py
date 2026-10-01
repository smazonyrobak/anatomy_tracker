"""Frozen synthetic-dev atlas-fit capture and local gradient direction; no model weights."""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUT = ROOT / 'runs/one_shot_full_pose_capture_001'
SIDE = 192
torch.set_num_threads(4)
atlas = load_streaming_synthetic_v7(device='cuda')['atlas']
model = OneShotJointSliceModel()
axis = torch.arange(SIDE, device='cuda') / SIDE
y, x = torch.meshgrid(axis, axis, indexing='ij')
pixel = torch.stack((x, y), -1)
landmarks = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.], [95.5, 95.5]], device='cuda') / SIDE
units = torch.tensor([.05, .05, .05, 250., 250., 250., .05, .05, .05], device='cuda')


def atlas_cost(states, flags, image, offsets, weights, valid):
    centre, frame, basis = full_frame_state_to_components(states)
    chart = pixel[None].expand(len(states), -1, -1, -1).clone()
    chart[..., 0] = torch.where(flags[:, None, None].bool(), 191 / SIDE - chart[..., 0], chart[..., 0])
    plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    coordinates = plane[:, None] + offsets[None, :, None, None, None] * frame[:, None, None, None, :, 2]
    return model.atlas_fit_loss(image, {'coordinates': coordinates[None]}, atlas, weights, valid)[0]


rows, gradient_rows = [], []
for record in map(json.loads, (DEV / 'records.jsonl').read_text().splitlines()):
    with np.load(DEV / record['file']) as arrays:
        appearance = record['section_index'] % 3  # exactly one image per physical section
        if not arrays['eligible'][appearance]:
            continue
        image = torch.from_numpy(arrays['inputs'][appearance:appearance + 1].copy()).cuda()
        truth = torch.from_numpy(arrays['target_state'].copy()).cuda()
        flag = int(arrays['reflection'])
        offsets = torch.from_numpy(arrays['offsets_um'].copy()).cuda()
        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        valid = torch.from_numpy((arrays['visible_support'][appearance:appearance + 1] > .25).copy()).cuda()
    identity = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')}
    true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
    true_chart = landmarks.clone()
    if flag:
        true_chart[:, 0] = 191 / SIDE - true_chart[:, 0]
    true_points = true_centre + torch.einsum('ij,pj->pi', true_frame[:, :2] @ true_basis, true_chart - .5)

    candidates = [('truth', 0., torch.zeros(9, device='cuda'), flag)]
    for axis_index, name in enumerate(('AP', 'DV', 'ML')):
        local_axis = true_frame.T[:, axis_index]
        for magnitude in (100., 250., 500., 1000.):
            for sign in (-1., 1.):
                update = torch.zeros(9, device='cuda')
                update[3:6] = sign * magnitude * local_axis
                candidates.append((name + '_translation_um', sign * magnitude, update, flag))
    for axis_index, name in enumerate(('tilt_u', 'tilt_v', 'roll')):
        for degrees in (1., 2., 5., 10.):
            for sign in (-1., 1.):
                update = torch.zeros(9, device='cuda')
                update[axis_index] = math.radians(sign * degrees)
                candidates.append((name + '_deg', sign * degrees, update, flag))
    for percent in (-10., -5., -2., 2., 5., 10.):
        update = torch.zeros(9, device='cuda')
        update[6:8] = math.log1p(percent / 100)
        candidates.append(('isotropic_scale_percent', percent, update, flag))
    candidates.append(('reflection_flip', 1., torch.zeros(9, device='cuda'), 1 - flag))
    updates = torch.stack([entry[2] for entry in candidates])
    flags = torch.tensor([entry[3] for entry in candidates], device='cuda')
    with torch.no_grad():
        states = compose_full_frame_state(truth.expand(len(candidates), -1), updates)
        costs = atlas_cost(states, flags, image, offsets, weights, valid)
        centres, frames, bases = full_frame_state_to_components(states)
        charts = landmarks[None].expand(len(candidates), -1, -1).clone()
        charts[..., 0] = torch.where(flags[:, None].bool(), 191 / SIDE - charts[..., 0], charts[..., 0])
        points = centres[:, None] + torch.einsum('nij,npj->npi', frames[:, :, :2] @ bases, charts - .5)
        errors = (points - true_points).norm(dim=-1).mean(-1)
        angles = torch.rad2deg(torch.acos((frames[:, :, 2] * true_frame[:, 2]).sum(-1).abs().clamp(0, 1)))
    for index, (name, delta, _, candidate_flag) in enumerate(candidates):
        rows.append({**identity, 'appearance_mode': appearance, 'perturbation': name, 'signed_amount': delta,
                     'reflection': candidate_flag, 'valid_pixels': int(valid.sum()),
                     'fit_mismatch': float(costs[index]), 'truth_mismatch': float(costs[0]),
                     'five_point_error_um': float(errors[index]), 'normal_error_deg': float(angles[index])})

    near = [i for i, (name, amount, _, _) in enumerate(candidates)
            if (name.endswith('translation_um') and abs(amount) == 250
                or name.endswith('_deg') and abs(amount) == 2
                or name == 'isotropic_scale_percent' and abs(amount) == 2)]
    q = (updates[near] / units).detach().requires_grad_()
    near_states = compose_full_frame_state(truth.expand(len(near), -1), q * units)
    near_costs = atlas_cost(near_states, flags[near], image, offsets, weights, valid)
    gradient = torch.autograd.grad(near_costs.sum(), q)[0]
    direction = .1 * gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        stepped = compose_full_frame_state(truth.expand(2 * len(near), -1),
                                           torch.cat((q - direction, q + direction)).detach() * units)
        stepped_flags = flags[near].repeat(2)
        stepped_costs = atlas_cost(stepped, stepped_flags, image, offsets, weights, valid)
        stepped_centres, stepped_frames, stepped_bases = full_frame_state_to_components(stepped)
        stepped_charts = landmarks[None].expand(len(stepped), -1, -1).clone()
        stepped_charts[..., 0] = torch.where(stepped_flags[:, None].bool(),
                                            191 / SIDE - stepped_charts[..., 0], stepped_charts[..., 0])
        stepped_points = stepped_centres[:, None] + torch.einsum(
            'nij,npj->npi', stepped_frames[:, :, :2] @ stepped_bases, stepped_charts - .5)
        stepped_errors = (stepped_points - true_points).norm(dim=-1).mean(-1)
    for slot, index in enumerate(near):
        name, amount = candidates[index][:2]
        gradient_rows.append({**identity, 'appearance_mode': appearance, 'perturbation': name,
                              'signed_amount': amount, 'gradient_norm': float(gradient[slot].norm()),
                              'fit_before': float(near_costs[slot]),
                              'fit_after_negative_step': float(stepped_costs[slot]),
                              'fit_after_positive_step': float(stepped_costs[slot + len(near)]),
                              'five_point_before_um': float(errors[index]),
                              'five_point_after_negative_step_um': float(stepped_errors[slot]),
                              'five_point_after_positive_step_um': float(stepped_errors[slot + len(near)])})

OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
(OUT / 'gradient_rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in gradient_rows))
groups = sorted({(row['perturbation'], abs(row['signed_amount'])) for row in rows if row['perturbation'] != 'truth'})
capture = {}
for name, amount in groups:
    selected = [row for row in rows if row['perturbation'] == name and abs(row['signed_amount']) == amount]
    capture[f'{name}:{amount:g}'] = {
        'sections': len(selected),
        'truth_lower_fraction': float(np.mean([row['truth_mismatch'] < row['fit_mismatch'] for row in selected])),
        'mean_cost_gap': float(np.mean([row['fit_mismatch'] - row['truth_mismatch'] for row in selected])),
    }
gradient_summary = {}
for name in sorted({row['perturbation'] for row in gradient_rows}):
    selected = [row for row in gradient_rows if row['perturbation'] == name]
    gradient_summary[name] = {
        'perturbations': len(selected),
        'nonzero_gradient_fraction': float(np.mean([row['gradient_norm'] > 1e-8 for row in selected])),
        'negative_step_pose_improvement_fraction': float(np.mean([
            row['five_point_after_negative_step_um'] < row['five_point_before_um'] for row in selected])),
        'negative_step_fit_improvement_fraction': float(np.mean([
            row['fit_after_negative_step'] < row['fit_before'] for row in selected])),
        'positive_step_pose_improvement_fraction': float(np.mean([
            row['five_point_after_positive_step_um'] < row['five_point_before_um'] for row in selected])),
    }
summary = {'source': str(DEV), 'warp': 'zero', 'mask': 'known true valid tissue',
           'gradient_step': '0.1 unit L2 in [0.05 rad, 250 um, 0.05 log-scale/shear] tangent coordinates',
           'sections': len({(row['animal_id'], row['specimen_id'], row['experiment_id'], row['section_id'])
                            for row in rows}),
           'synthetic_donor_ids': sorted({row['animal_id'] for row in rows}),
           'capture': capture, 'gradient': gradient_summary}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
