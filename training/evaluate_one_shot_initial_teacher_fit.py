"""Held-out synthetic check of one-pass tissue mapping at the correct plane."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

RUN = ROOT / 'runs/one_shot_joint_initial_001'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
torch.set_num_threads(4)
atlas = load_streaming_synthetic_v7(device='cuda')['atlas']
records = [json.loads(row) for row in (DEV / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({row['animal_id'] for row in records}) == 4
rows = []
for step, filename in ((0, 'initial.pt'), (500, 'joint_step_00500.pt'),
                       (1500, 'joint_step_01500.pt'), (2500, 'latest.pt')):
    checkpoint = torch.load(RUN / filename, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    with torch.inference_mode():
        for record in records:
            with np.load(DEV / record['file']) as arrays:
                mode = record['section_index'] % 3
                image = torch.from_numpy(arrays['inputs'][mode:mode + 1].copy()).cuda()
                state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflected = torch.tensor([[int(arrays['reflection'])]], device='cuda')
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy((arrays['visible_support'][mode:mode + 1] > .25).copy()).cuda()
            prediction = model.predict(image)
            xy = state.new_tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.], [95.5, 95.5]]) / 192
            xy = xy[None].expand(2, -1, -1).clone()
            xy[1, :, 0] = 191 / 192 - xy[1, :, 0]
            truth_centre, truth_frame, truth_basis = full_frame_state_to_components(state)
            truth_points = truth_centre[:, None] + torch.einsum(
                'bij,pj->bpi', truth_frame[:, :, :2] @ truth_basis, xy[int(reflected[0, 0])] - .5)[0]
            candidate_centre, candidate_frame, candidate_basis = full_frame_state_to_components(prediction['state'][0])
            candidate_points = candidate_centre[:, None, None] + torch.einsum(
                'mij,rpj->mrpi', candidate_frame[:, :, :2] @ candidate_basis, xy - .5)
            candidate_error = (candidate_points - truth_points).norm(dim=-1).mean(-1)
            prior = prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)
            chosen = int(prior.flatten().argmax())
            pose = prediction['state'].clone()
            pose[:, 0] = state
            mapped = model.map({**prediction, 'state': pose}, offsets,
                               torch.zeros(1, 1, dtype=torch.long, device='cuda'), reflected, (192, 192))
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0]
            frame = full_frame_state_to_components(state)[1]
            local = mapped['local_displacement_um'][:, 0]
            zero = surface - torch.einsum('bij,bjhw->bhwi', frame, local)
            denom = valid.sum().clamp_min(1)
            error = (surface - target).norm(dim=-1)
            zero_error = (zero - target).norm(dim=-1)
            fit = model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0, 0]
            rows.append({'step': step, 'animal_id': record['animal_id'],
                         'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                         'section_id': record['section_id'], 'visible_pixels': int(valid.sum()),
                         'mapped_error_um': float((error * valid).sum() / denom),
                         'zero_warp_error_um': float((zero_error * valid).sum() / denom),
                         'atlas_fit_loss': float(fit),
                         'selected_mode': chosen // 2, 'selected_reflection': chosen % 2,
                         'truth_reflection': int(reflected[0, 0]),
                         'selected_pose_error_um': float(candidate_error.flatten()[chosen]),
                         'oracle_pose_error_um': float(candidate_error.min())})
    del model
with (RUN / 'teacher_fit_development.jsonl').open('w', encoding='utf8') as output:
    output.writelines(json.dumps(row) + '\n' for row in rows)
for step in (0, 500, 1500, 2500):
    selected = [row for row in rows if row['step'] == step and row['visible_pixels'] > 0]
    print(json.dumps({'step': step, 'held_out_sections': len(selected),
        'mapped_error_um': float(np.mean([row['mapped_error_um'] for row in selected])),
        'zero_warp_error_um': float(np.mean([row['zero_warp_error_um'] for row in selected])),
        'atlas_fit_loss': float(np.mean([row['atlas_fit_loss'] for row in selected])),
        'reflection_accuracy': float(np.mean([row['selected_reflection'] == row['truth_reflection'] for row in selected])),
        'selected_modes': [sum(row['selected_mode'] == mode for row in selected) for mode in range(8)]}), flush=True)
