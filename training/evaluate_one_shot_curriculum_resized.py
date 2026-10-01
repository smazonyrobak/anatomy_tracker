"""Frozen curriculum readout with development images resized as at training time."""
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

RUN = ROOT / 'runs/one_shot_joint_pose_curriculum_002'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'runs/one_shot_joint_pose_curriculum_002_resized_eval'
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 10000
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
synthetic = [json.loads(line) for line in (DEV / 'records.jsonl').read_text().splitlines()]
real = [row for row in map(json.loads, (REAL / 'records.jsonl').read_text().splitlines())
        if row['training_split'] == 'development']
assert len(synthetic) == len(real) == 64
assert len({row['animal_id'] for row in synthetic}) == 4
assert len({row['animal_id'] for row in real}) == 6
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz') as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
uv = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device='cuda') / 256


def five_points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = uv.expand(*state.shape[:-1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


rows = []
for step in (0, *range(1000, 10001, 1000)):
    checkpoint = torch.load(RUN / ('initial.pt' if step == 0 else f'joint_step_{step:05d}.pt'),
                            map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    with torch.inference_mode():
        for split, records in (('synthetic', synthetic), ('real_weak_allen', real)):
            for record in records:
                if split == 'synthetic':
                    with np.load(DEV / record['file']) as arrays:
                        appearance = record['section_index'] % 3
                        image = torch.from_numpy(arrays['inputs'][appearance:appearance + 1].copy()).cuda()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        reflected = torch.tensor([int(arrays['reflection'])], device='cuda')
                        target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                        valid = torch.from_numpy((arrays['visible_support'][appearance:appearance + 1] > .25).copy()).cuda()
                        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    reference = five_points(truth, reflected)[0]
                    normal = full_frame_state_to_components(truth)[1][0, :, 2]
                else:
                    index = record['array_row_index']
                    image = torch.from_numpy(np.concatenate((real_images[index].astype(np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32))))[None].cuda()
                    affine = torch.as_tensor(real_affines[index], device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + 192 * uv[:, :1] * affine[:, 0] + 192 * uv[:, 1:] * affine[:, 1]
                    normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
                prediction = model.predict(image)
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                candidate = five_points(states, torch.tensor([[0, 1]], device='cuda').expand(model.modes, -1))
                errors = (candidate - reference).norm(dim=-1).mean(-1)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                selected = int(prior.flatten().argmax())
                predicted_normal = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                angle = torch.rad2deg((predicted_normal * normal).sum(-1).abs().clamp(0, 1).acos())
                row = {'step': step, 'set': split,
                       **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                       'selected_error_um': float(errors.flatten()[selected]),
                       'oracle_error_um': float(errors.min()),
                       'selected_normal_angle_deg': float(angle[selected // 2]),
                       'selected_mode': selected // 2, 'selected_reflection': selected % 2}
                if split == 'synthetic' and bool(valid.any()):
                    state = prediction['state'].clone()
                    state[:, 0] = truth
                    mapped = model.map({**prediction, 'state': state}, offsets,
                                       torch.zeros((1, 1), dtype=torch.long, device='cuda'),
                                       reflected[:, None], (256, 256))
                    target = F.interpolate(target.permute(0, 3, 1, 2), (256, 256),
                                           mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    valid = F.interpolate(valid[:, None].float(), (256, 256), mode='nearest')[:, 0] > .5
                    surface = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0]
                    local = mapped['local_displacement_um'][:, 0]
                    zero = surface - torch.einsum('bij,bjhw->bhwi', full_frame_state_to_components(truth)[1], local)
                    row['teacher_map_error_um'] = float((surface - target).norm(dim=-1)[valid].mean())
                    row['zero_warp_error_um'] = float((zero - target).norm(dim=-1)[valid].mean())
                rows.append(row)
    print(json.dumps({'step': step, 'readout': 'resized_256', 'rows': len(rows)}), flush=True)
    del model, checkpoint

with (OUT / 'rows.jsonl').open('w') as output:
    output.writelines(json.dumps(row) + '\n' for row in rows)
summary = []
for step in (0, *range(1000, 10001, 1000)):
    for split in ('synthetic', 'real_weak_allen'):
        subset = [row for row in rows if row['step'] == step and row['set'] == split]
        summary.append({'step': step, 'set': split, 'animals': len({row['animal_id'] for row in subset}),
                        **{name: float(np.mean([np.mean([row[name] for row in subset
                             if row['animal_id'] == donor and name in row])
                             for donor in {row['animal_id'] for row in subset}]))
                           for name in ('selected_error_um', 'oracle_error_um', 'selected_normal_angle_deg')},
                        'teacher_map_error_um': float(np.mean([row['teacher_map_error_um'] for row in subset
                                                               if 'teacher_map_error_um' in row])) if split == 'synthetic' else None,
                        'zero_warp_error_um': float(np.mean([row['zero_warp_error_um'] for row in subset
                                                             if 'zero_warp_error_um' in row])) if split == 'synthetic' else None})
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
