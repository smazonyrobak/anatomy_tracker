"""Separate spatial-coordinate prediction failure from plane-fit weighting failure."""
import json
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

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
checkpoint = root / 'runs/one_shot_dense_coordinate_pose_013/joint_step_05000.pt'
out = root / 'runs/one_shot_dense_coordinate_pose_013_fit_diagnostic'
torch.set_num_threads(4)
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])


def points(state, chart):
    center, frame, basis = full_frame_state_to_components(state)
    return center[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, chart - .5)


def fits(prediction, oracle=None):
    field = prediction['dense_coordinate']
    fields = {'predicted': field,
              'uniform': torch.cat((field[:, :3], torch.zeros_like(field[:, 3:4])), 1)}
    if oracle is not None:
        fields['oracle_validity'] = torch.cat((field[:, :3],
            torch.logit(F.avg_pool2d(oracle.float(), 16).clamp(1e-7, 1 - 1e-7))), 1)
    return {name: model.dense_coordinate_plane({'dense_coordinate': value})
            for name, value in fields.items()}


rows = []
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
cells = torch.arange(16, device='cuda').float() * 16 + 7.5
yy, xx = torch.meshgrid(cells / 256, cells / 256, indexing='ij')
small_chart = torch.stack((xx, yy), -1).reshape(1, 256, 2)
with torch.inference_mode():
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None, None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
        prediction = model.predict(image)
        indices = valid.flatten().nonzero().flatten()
        chart = torch.stack((indices.remainder(256),
                             indices.div(256, rounding_mode='floor')), -1).float()[None] / 256
        target = reference.reshape(1, -1, 3)[:, indices]
        outcomes = {name: float((points(state, chart) - target).norm(dim=-1).mean())
                    for name, state in fits(prediction, valid).items()}
        rigid_chart = small_chart.clone()
        if reflection:
            rigid_chart[..., 0] = 255 / 256 - rigid_chart[..., 0]
        rigid_target = points(truth, rigid_chart)
        field = prediction['dense_coordinate'][:, :3] * model.center_scale[None, :, None, None]
        field = (field + model.center_origin[None, :, None, None]).flatten(2).transpose(1, 2)
        occupancy = F.avg_pool2d(valid.float(), 16).flatten()
        field_error = (field - rigid_target).norm(dim=-1).flatten()
        rows.append({'set': 'synthetic', 'animal_id': record['animal_id'],
                     'section_id': record['section_id'], 'tissue_fraction': float(valid.float().mean()),
                     'field_error_all_um': float(field_error.mean()),
                     'field_error_tissue_um': float((field_error * occupancy).sum() / occupancy.sum()),
                     **{f'{key}_rigid_um': value for key, value in outcomes.items()}})
    for record in real_records:
        i = record['array_row_index']
        image = np.concatenate((real_images[i].astype(np.float32),
                                np.zeros((4, 192, 192), dtype=np.float32)))[None]
        image = F.interpolate(torch.from_numpy(image).cuda(), (256, 256),
                              mode='bilinear', align_corners=False)
        affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
        prediction = model.predict(image)
        reference = (affine[:, 2][None, None] + 192 * small_chart[..., :1] * affine[:, 0]
                     + 192 * small_chart[..., 1:] * affine[:, 1])
        field = prediction['dense_coordinate'][:, :3] * model.center_scale[None, :, None, None]
        field = (field + model.center_origin[None, :, None, None]).flatten(2).transpose(1, 2)
        chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                              [127.5, 127.5]], device='cuda')[None] / 256
        five = (affine[:, 2][None, None] + 192 * chart[..., :1] * affine[:, 0]
                + 192 * chart[..., 1:] * affine[:, 1])
        outcomes = {name: float((points(state, chart) - five).norm(dim=-1).mean())
                    for name, state in fits(prediction).items()}
        rows.append({'set': 'real_weak_allen', 'animal_id': record['animal_id'],
                     'section_id': record['section_id'],
                     'field_error_all_um': float((field - reference).norm(dim=-1).mean()),
                     **{f'{key}_five_um': value for key, value in outcomes.items()}})

summary = {split: {'rows': sum(row['set'] == split for row in rows),
                   'identity_equal_mean': {key: float(np.mean([
                       np.mean([row[key] for row in rows if row['set'] == split
                                and row['animal_id'] == identity]) for identity in sorted({
                           row['animal_id'] for row in rows if row['set'] == split})]))
                       for key in (('field_error_all_um', 'field_error_tissue_um',
                                    'predicted_rigid_um', 'uniform_rigid_um', 'oracle_validity_rigid_um')
                                   if split == 'synthetic' else
                                   ('field_error_all_um', 'predicted_five_um', 'uniform_five_um'))}}
           for split in ('synthetic', 'real_weak_allen')}
out.mkdir(parents=True, exist_ok=False)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
