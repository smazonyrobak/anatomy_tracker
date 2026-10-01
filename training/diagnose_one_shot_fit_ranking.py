"""Does atlas appearance favour the true plane over nearby wrong planes?"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUT = ROOT / 'runs/one_shot_fit_rank_001'
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
atlas = load_streaming_synthetic_v7(device='cuda')['atlas']
model = OneShotJointSliceModel().cuda().eval()
rows = []
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
shifts = torch.tensor([0., -1000., -500., -250., 250., 500., 1000.], device='cuda')

with torch.inference_mode():
    for record in map(json.loads, (DEV / 'records.jsonl').read_text().splitlines()):
        with np.load(DEV / record['file']) as arrays:
            mode = record['section_index'] % 3
            if not arrays['eligible'][mode]:
                continue
            image = torch.from_numpy(arrays['inputs'][mode:mode + 1].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda().expand(len(shifts), -1).clone()
            state[:, 0] += shifts
            reflected = bool(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            valid = torch.from_numpy((arrays['visible_support'][mode:mode + 1] > .25).copy()).cuda()
        centre, frame, basis = full_frame_state_to_components(state)
        chart = torch.stack((191 / 192 - x if reflected else x, y), -1)
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart[None])
        coordinates = plane[:, None] + offsets[None, :, None, None, None] * frame[:, None, None, None, :, 2]
        fit = model.atlas_fit_loss(image, {'coordinates': coordinates[None]}, atlas, weights, valid)[0].cpu().numpy()
        rows.append({'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                     'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                     'appearance_mode': mode, 'visible_pixels': int(valid.sum()),
                     'ap_shifts_um': shifts.cpu().tolist(), 'fit_mismatch': fit.tolist(),
                     'truth_rank': int(np.argsort(fit).tolist().index(0)) + 1})

with (OUT / 'rows.jsonl').open('w') as output:
    output.writelines(json.dumps(row) + '\n' for row in rows)
summary = {'sections': len(rows), 'donors': len({row['animal_id'] for row in rows}),
           'true_plane_lowest_fraction': float(np.mean([row['truth_rank'] == 1 for row in rows])),
           'true_plane_below_500um_fraction': float(np.mean([
               row['fit_mismatch'][0] < min(row['fit_mismatch'][2], row['fit_mismatch'][5]) for row in rows])),
           'mean_fit_by_ap_shift': np.mean([row['fit_mismatch'] for row in rows], axis=0).tolist()}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
