"""Does the exact synthetic CCF surface look more similar than proposed planes?"""
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
comparison = root / 'runs/one_shot_information_022/rows.jsonl'
rows = [json.loads(line) for line in comparison.open()]
records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open())
           if row['eligible']}
assert len(rows) == len(records) == 185
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
torch.set_num_threads(4)
results = []
with torch.inference_mode():
    for row in rows:
        with np.load(panel / records[row['sha256']]['file'], allow_pickle=False) as data:
            image = torch.from_numpy(data['inputs'][None, :1].copy()).cuda()
            valid = torch.from_numpy(data['valid_mask'][None, None].copy()).cuda().float()
            centre = torch.from_numpy(data['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(data['target_state'][None].copy()).cuda()
            offsets = torch.from_numpy(data['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(data['weights'][None].copy()).cuda()
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        scored = {}
        for size in (96, 256):
            tissue = F.interpolate(valid, (size, size), mode='area')
            image_small = F.interpolate(image, (size, size), mode='area')
            centre_small = F.interpolate(centre.permute(0, 3, 1, 2), (size, size),
                                         mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            coordinates = centre_small[:, None] + offsets[:, :, None, None, None] * normal[:, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights)
            support = rendered[:, 1:2].clamp(0, 1)
            target = rendered[:, :1] / support.clamp_min(1e-4)
            a = image_small - F.avg_pool2d(image_small, 9, 1, 4)
            b = target - F.avg_pool2d(target, 9, 1, 4)
            cross = F.avg_pool2d(a * b, 9, 1, 4)
            variance = F.avg_pool2d(a.square(), 9, 1, 4) * F.avg_pool2d(b.square(), 9, 1, 4)
            agreement = (cross / (variance + 1e-5).sqrt()).abs().clamp(0, 1)
            exact = (((1 - agreement) * tissue * support + tissue * (1 - support)).sum()
                     / tissue.sum())
            scored[str(size)] = {'exact': float(exact),
                                 'best_predicted': min(row['disagreement'][f'{size}_oracle']),
                                 'physical_best_predicted': row['disagreement'][f'{size}_oracle'][
                                     int(np.argmin(row['physical_mapped_um']))]}
        results.append(scored)
for size in ('96', '256'):
    exact = np.asarray([row[size]['exact'] for row in results])
    best = np.asarray([row[size]['best_predicted'] for row in results])
    physical = np.asarray([row[size]['physical_best_predicted'] for row in results])
    print(json.dumps({'size': int(size), 'exact_mean': float(exact.mean()),
                      'best_predicted_mean': float(best.mean()),
                      'physical_best_predicted_mean': float(physical.mean()),
                      'exact_beats_best_predicted_fraction': float((exact < best).mean()),
                      'exact_beats_physical_best_fraction': float((exact < physical).mean())}))
