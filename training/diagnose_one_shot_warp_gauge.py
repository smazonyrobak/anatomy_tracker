"""Measure target warp excluded by the model's zero-affine displacement gauge."""
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
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

OUT = ROOT / 'runs/one_shot_warp_gauge_002'
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
rng = torch.Generator().manual_seed(2026100205)
axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
rows = []

with torch.inference_mode():
    for draw in range(64):
        subjects = torch.randint(len(context['subjects']), (4,), generator=rng).tolist()
        batch = sample_one_shot_stream(context, subjects, 2026100205000000 + draw, side=256)
        centre, frame, basis = full_frame_state_to_components(batch['state'])
        reflected_x = torch.where(batch['reflection'][:, None, None].bool(), 255 / 256 - x, x)
        chart = torch.stack((reflected_x.expand(-1, 256, -1), y.expand(4, -1, -1)), -1)
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        target = batch['centre']
        ideal = torch.einsum('bji,bhwj->bihw', frame, target - plane)
        projected, removed = project_surface_affine_out(ideal)
        attainable = plane + torch.einsum('bij,bjhw->bhwi', frame, projected)
        valid = batch['valid_mask']
        zero_error = (plane - target).norm(dim=-1)
        gauge_error = (attainable - target).norm(dim=-1)
        for index, record in enumerate(batch['provenance']):
            if not bool(batch['eligible'][index]):
                continue
            mask = valid[index]
            rows.append({'physical_section_id': record['physical_section_id'],
                         'virtual_subject_id': record['virtual_subject_id'],
                         'base_lineage': record['base_lineage'], 'seed_prefix': record['seed_prefix'],
                         'draw': draw, 'slot': index,
                         'warp_strength': record['one_shot']['warp_strength'],
                         'valid_pixels': int(mask.sum()),
                         'zero_warp_error_um': float(zero_error[index][mask].mean()),
                         'excluded_affine_error_um': float(gauge_error[index][mask].mean()),
                         'excluded_affine_coefficients_um': removed[index].cpu().tolist()})

with (OUT / 'rows.jsonl').open('w') as output:
    output.writelines(json.dumps(row) + '\n' for row in rows)
strength = np.asarray([row['warp_strength'] for row in rows])
zero = np.asarray([row['zero_warp_error_um'] for row in rows])
excluded = np.asarray([row['excluded_affine_error_um'] for row in rows])
summary = {'sections': len(rows), 'zero_warp_mean_um': float(zero.mean()),
           'excluded_affine_mean_um': float(excluded.mean()),
           'strength_quartiles': np.quantile(strength, [0, .25, .5, .75, 1]).tolist(),
           'excluded_affine_by_strength_quartile_um': [
               float(excluded[(strength >= a) & (strength <= b)].mean())
               for a, b in zip(np.quantile(strength, [0, .25, .5, .75]),
                               np.quantile(strength, [.25, .5, .75, 1]))]}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
