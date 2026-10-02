"""Show frozen generator inputs beside their true finite-thickness atlas planes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    _decode_and_preprocess_allen_v6, TEMPLATE_RAW_SHA256_V6, ANNOTATION_RAW_SHA256_V6,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker')
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'reports/one_shot_generator_qc_20261002'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
atlas, _ = _decode_and_preprocess_allen_v6()
font = ImageFont.load_default(size=17)
small = ImageFont.load_default(size=14)
modes = ('raw', 'exact_black', 'imperfect_brush')
orientations = (('near coronal', .8, 1.01), ('intermediate', .4, .8), ('strongly oblique', 0, .4))
tile, gap, top = 192, 10, 62
cell_w, cell_h = tile * 2 + gap, tile + top + 12
canvas = Image.new('RGB', (3 * cell_w + 4 * 18, 3 * cell_h + 4 * 18), (238, 238, 238))
draw = ImageDraw.Draw(canvas)
axis = np.arange(256, dtype=np.float32) / 256
yy, xx = np.meshgrid(axis, axis, indexing='ij')
selected = []

for row, mode in enumerate(modes):
    for column, (orientation, lower, upper) in enumerate(orientations):
        candidates = [r for r in records if r['eligible'] and r['appearance_mode'] == mode
                      and lower <= abs(r['plane_normal_ap_dv_ml'][0]) < upper]
        midpoint = np.median([r['valid_pixels'] for r in candidates])
        record = min(candidates, key=lambda r: (abs(r['valid_pixels'] - midpoint), r['section_id']))
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            observed = np.clip(arrays['inputs'][0], 0, 1)
            state = torch.from_numpy(arrays['target_state']).float()
            centre, frame, basis = full_frame_state_to_components(state)
            reflected = bool(arrays['reflection'])
            s = 255 / 256 - xx if reflected else xx
            chart = np.stack((s, yy), -1)
            linear = (frame[:, :2] @ basis).numpy()
            plane = centre.numpy() + (chart - .5) @ linear.T
            normal = frame[:, 2].numpy()
            offsets = arrays['offsets_um']
            weights = arrays['weights'] / arrays['weights'].sum()
        intensity = np.zeros((256, 256), dtype=np.float32)
        support = np.zeros_like(intensity)
        for axial_offset, weight in zip(offsets, weights):
            query = np.moveaxis((plane + axial_offset * normal) / 25 - .5, -1, 0)
            intensity += weight * map_coordinates(atlas[0], query, order=1, mode='constant')
            support += weight * map_coordinates(atlas[1], query, order=1, mode='constant')
        atlas_plane = np.clip(intensity / np.maximum(support, 1e-4), 0, 1)
        left = Image.fromarray((observed * 255).astype(np.uint8)).resize((tile, tile))
        right = Image.fromarray((atlas_plane * 255).astype(np.uint8)).resize((tile, tile))
        x0 = 18 + column * (cell_w + 18)
        y0 = 18 + row * (cell_h + 18)
        draw.text((x0, y0), f'{mode.replace("_", " ")} | {orientation}', font=font, fill=(20, 20, 20))
        draw.text((x0, y0 + 25), f'{record["valid_pixels"] / 65536:.0%} visible tissue',
                  font=small, fill=(50, 50, 50))
        canvas.paste(left.convert('RGB'), (x0, y0 + top))
        canvas.paste(right.convert('RGB'), (x0 + tile + gap, y0 + top))
        draw.text((x0 + 3, y0 + top - 19), 'generator output', font=small, fill=(20, 20, 20))
        draw.text((x0 + tile + gap + 3, y0 + top - 19), 'true atlas plane',
                  font=small, fill=(20, 20, 20))
        selected.append({key: record[key] for key in ('section_id', 'animal_id', 'specimen_id',
                         'experiment_id', 'sha256', 'appearance_mode', 'valid_pixels',
                         'plane_normal_ap_dv_ml', 'thickness_um')})

out.mkdir(parents=True, exist_ok=True)
image = out / 'generator_and_atlas_grid.png'
canvas.save(image)
(out / 'selection.json').write_text(json.dumps({
    'selection': 'eligible median visible-pixel count per appearance by orientation stratum',
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'atlas_template_sha256': TEMPLATE_RAW_SHA256_V6,
    'atlas_annotation_sha256': ANNOTATION_RAW_SHA256_V6,
    'atlas_plane': 'true observed affine pose and reflection; nine-node finite-thickness slab',
    'not_model_predictions': True,
    'selected': selected,
    'grid_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
}, indent=2))
print(image)
