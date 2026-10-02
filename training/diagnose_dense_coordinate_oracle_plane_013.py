"""Check the best possible plane fit from exact pooled CCF coordinates on fixed DEV sections."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = Path('I:/AnatomyTracker/data/one_shot_fresh_synthetic_dev_panel_001')
model = OneShotJointSliceModel(modes=16, dense_coordinate=True).cuda().eval()
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
rows = []
with torch.inference_mode():
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None, None].copy()).cuda().float()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
        occupancy = F.avg_pool2d(valid, 16)
        pooled = F.avg_pool2d(centre.permute(0, 3, 1, 2) * valid, 16) / occupancy.clamp_min(1e-4)
        normalized = (pooled - model.center_origin[None, :, None, None]) / model.center_scale[None, :, None, None]
        field = torch.cat((normalized, torch.logit(occupancy.clamp(1e-7, 1 - 1e-7))), 1)
        fitted = model.dense_coordinate_plane({'dense_coordinate': field})
        indices = valid.flatten().nonzero().flatten()
        chart = torch.stack((indices.remainder(256), indices.div(256, rounding_mode='floor')), -1).float()[None] / 256
        target = centre.reshape(1, -1, 3)[:, indices]
        center, frame, basis = full_frame_state_to_components(fitted)
        estimate = center[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, chart - .5)
        center, frame, basis = full_frame_state_to_components(truth)
        source_chart = chart.clone()
        if reflection:
            source_chart[..., 0] = 255 / 256 - source_chart[..., 0]
        teacher = center[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, source_chart - .5)
        cells = torch.arange(16, device='cuda').float() * 16 + 7.5
        yy, xx = torch.meshgrid(cells / 256, cells / 256, indexing='ij')
        small_chart = torch.stack((xx, yy), -1).reshape(1, 256, 2)
        if reflection:
            small_chart[..., 0] = 255 / 256 - small_chart[..., 0]
        planar = center[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, small_chart - .5)
        planar = planar.reshape(1, 16, 16, 3).permute(0, 3, 1, 2)
        field = torch.cat(((planar - model.center_origin[None, :, None, None]) / model.center_scale[None, :, None, None],
                           torch.logit(occupancy.clamp(1e-7, 1 - 1e-7))), 1)
        plane_state = model.dense_coordinate_plane({'dense_coordinate': field})
        center, frame, basis = full_frame_state_to_components(plane_state)
        plane_estimate = center[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, chart - .5)
        rows.append((float(valid.mean()), float((estimate - target).norm(dim=-1).mean()),
                     float((teacher - target).norm(dim=-1).mean()),
                     float((plane_estimate - target).norm(dim=-1).mean())))

values = np.asarray(rows)
print(json.dumps({'eligible': len(rows), 'pooled_deformed_coordinate_plane_um': float(values[:, 1].mean()),
                  'synthetic_true_pose_rigid_um': float(values[:, 2].mean()),
                  'pooled_rigid_coordinate_plane_um': float(values[:, 3].mean()),
                  'by_support': [{'range': label, 'n': int(mask.sum()),
                                  'pooled_deformed_coordinate_plane_um': float(values[mask, 1].mean()),
                                  'true_pose_rigid_um': float(values[mask, 2].mean()),
                                  'pooled_rigid_coordinate_plane_um': float(values[mask, 3].mean())}
                                 for label, mask in [('<5%', values[:, 0] < .05),
                                                     ('5-15%', (values[:, 0] >= .05) & (values[:, 0] < .15)),
                                                     ('15-30%', (values[:, 0] >= .15) & (values[:, 0] < .30)),
                                                     ('>=30%', values[:, 0] >= .30)]]}))
