"""Six independently drawn sections from the GPU training stream, each paired only for display."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
OUT = ROOT / 'previews/one_shot_training_stream_20261001'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image, ImageDraw

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_ribbon_v6 import compose_curved_ribbon_coordinates
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

torch.set_num_threads(4)
context = load_streaming_synthetic_v7(device='cuda')
picked = {mode: [] for mode in ('raw', 'exact_black', 'imperfect_brush')}
rng = np.random.default_rng(20261001)
for draw in range(24):
    subjects = rng.integers(len(context['subjects']), size=4).tolist()
    batch = sample_one_shot_stream(context, subjects, 2026100100 + draw, side=256)
    for row, record in enumerate(batch['provenance']):
        mode = record['mode']
        if bool(batch['eligible'][row]) and len(picked[mode]) < 2:
            picked[mode].append({
                'image': batch['inputs'][row, 0].detach().cpu(),
                'state': batch['state'][row:row + 1].detach(),
                'reflection': bool(batch['reflection'][row]),
                'offsets': batch['offsets'][row:row + 1].detach(),
                'weights': batch['weights'][row:row + 1].detach(),
                'provenance': record,
            })
    if all(len(items) == 2 for items in picked.values()):
        break

samples = [item for mode in picked for item in picked[mode]]
assert len(samples) == 6
sheet = Image.new('RGB', (4 * 330, 3 * 350), 'white')
records = []
for index, sample in enumerate(samples):
    state = sample['state']
    zeros = state.new_zeros(1, 3, 256, 256)
    coordinates = compose_curved_ribbon_coordinates(state, zeros, zeros, sample['offsets'])['ccf_coordinates_ap_dv_ml_um']
    if sample['reflection']:
        coordinates = coordinates.flip(-2)
    rendered = render_finite_thickness_coordinate_grid(
        context['atlas'], coordinates, (0., 0., 0.), (25., 25., 25.), sample['weights']
    )
    atlas = (rendered[0, 0] / rendered[0, 1].clamp_min(1e-4)).clamp(0, 1).cpu().numpy()
    atlas[rendered[0, 1].cpu().numpy() < .05] = 0
    observed = sample['image'].numpy()
    Image.fromarray(np.uint8(np.rint(observed * 255)), 'L').save(OUT / f'observed_{index + 1:02d}.png')
    Image.fromarray(np.uint8(np.rint(atlas * 255)), 'L').save(OUT / f'atlas_{index + 1:02d}.png')
    x, y = (index % 2) * 660 + 10, (index // 2) * 350 + 10
    for column, values in enumerate((observed, atlas)):
        tile = Image.fromarray(np.uint8(np.rint(values * 255)), 'L').convert('RGB').resize((310, 310))
        sheet.paste(tile, (x + column * 330, y))
    events = [name for name, enabled in sample['provenance']['one_shot']['events'].items() if enabled]
    mode = sample['provenance']['mode']
    ImageDraw.Draw(sheet).text((x, y + 315), f'{index + 1}: {mode}; {", ".join(events) or "clean"}', fill='black')
    ImageDraw.Draw(sheet).text((x + 330, y + 315), 'matching atlas plane', fill='black')
    records.append({'observed': f'observed_{index + 1:02d}.png', 'atlas': f'atlas_{index + 1:02d}.png',
        **sample['provenance']})
sheet.save(OUT / 'contact_sheet.png')
(OUT / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf8')
print(json.dumps({'preview': str(OUT / 'contact_sheet.png'), 'sampled_sections': len(samples),
                  'modes': [record['mode'] for record in records]}), flush=True)
