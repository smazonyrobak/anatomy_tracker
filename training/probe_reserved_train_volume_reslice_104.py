"""TRAIN-only visual feasibility probe: reslice one registered Allen donor stack."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

donor = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001/donor_111213883')
out = Path('I:/AnatomyTracker/runs/reserved_train_volume_reslice_104_probe')
records = [json.loads(line) for line in (donor / 'records.jsonl').open()]
volume = np.load(donor / 'images.npy', mmap_mode='r')[:, 0].astype(np.float32)
affines = np.asarray([row['model_pixel_to_ap_dv_ml_um'] for row in records])
assert volume.shape == (len(records), 192, 192)
assert all(row['training_split'] == 'train' for row in records)
assert np.all(np.diff([row['section_number'] for row in records]) == 1)
assert np.max(np.abs(affines[:, :, :2] - affines[0, :, :2])) < 1e-8
step = affines[1, :, 2] - affines[0, :, 2]
assert np.max(np.abs(np.diff(affines[:, :, 2], axis=0) - step)) < 1e-6
origin = affines[0, :, 2]
basis = np.column_stack((affines[0, :, 0], affines[0, :, 1], step))
center = origin + basis @ np.array([95.5, 95.5, (len(records) - 1) / 2])
axis = (np.arange(256) - 127.5) * 55
yy, xx = np.meshgrid(axis, axis, indexing='ij')
normal = np.array([.55, .55, .63])
normal /= np.linalg.norm(normal)
oblique_u = np.cross(normal, [0, 0, 1])
oblique_u /= np.linalg.norm(oblique_u)
oblique_v = np.cross(normal, oblique_u)
frames = (
    ('Native coronal', None, None),
    ('Horizontal', np.array([0., 0., 1.]), np.array([1., 0., 0.])),
    ('Sagittal', np.array([1., 0., 0.]), np.array([0., 1., 0.])),
    ('Oblique', oblique_u, oblique_v),
)
sheet = Image.new('RGB', (256 * 4, 286), 'white')
draw = ImageDraw.Draw(sheet)
for column, (title, u, v) in enumerate(frames):
    if u is None:
        image = np.asarray(volume[len(records) // 2])
        image = np.asarray(Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize((256, 256)))
    else:
        points = center[:, None, None] + u[:, None, None] * xx + v[:, None, None] * yy
        coordinates = np.linalg.solve(basis, (points - origin[:, None, None]).reshape(3, -1))
        image = map_coordinates(volume, coordinates[[2, 1, 0]], order=1,
                                mode='constant', cval=0).reshape(256, 256)
        image = np.uint8(np.clip(image, 0, 1) * 255)
    sheet.paste(Image.fromarray(image).convert('RGB'), (256 * column, 30))
    draw.text((256 * column + 8, 8), title, fill='black')
out.mkdir(parents=True, exist_ok=False)
sheet.save(out / 'reslices.png')
(out / 'receipt.json').write_text(json.dumps({'donor': donor.name,
    'train_sections': len(records), 'section_number_range': [records[0]['section_number'],
        records[-1]['section_number']], 'section_step_ccf_um': step.tolist(),
    'center_ccf_um': center.tolist(), 'fov_mm': 14.08,
    'scope': 'TRAIN-only visual feasibility; weak Allen affine lattice, not expert geometry or genuine oblique acquisition'}, indent=2))
print(json.dumps({'event': 'reslice_probe_complete', 'donor': donor.name,
    'train_sections': len(records), 'output': str(out / 'reslices.png')}))
