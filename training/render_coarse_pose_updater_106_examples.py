"""Actual v3 generator input beside true, prior, and 106-selected atlas planes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker')
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
evaluation = root / 'runs/coarse_pose_updater_106_dev_eval'
out = root / 'reports/coarse_pose_updater_106_examples_001'
assert not out.exists()
records = {row['section_id']: row for row in
    (json.loads(line) for line in (panel / 'records.jsonl').open())}
sections = [json.loads(line) for line in (evaluation / 'sections.jsonl').open()
    if line.strip()]
sections = [row for row in sections if row['step'] == 4000 and row['arm'] == 'atlas']
candidates = {(row['section_id'], row['beam_slot']): row for row in
    (json.loads(line) for line in (evaluation / 'candidates.jsonl').open())
    if row['step'] == 4000 and row['arm'] == 'atlas'}
rules = (('raw', 'AP', 25), ('imperfect_brush', 'DV', 25),
    ('exact_black', 'ML', 15))
chosen = []
for mode, family, minimum_angle in rules:
    group = [row for row in sections if row['appearance_mode'] == mode
        and row['angle_family'] == family
        and row['nearest_axis_angle_deg'] >= minimum_angle]
    median = np.median([row['valid_fraction'] for row in group])
    chosen.append(min(group, key=lambda row: (abs(row['valid_fraction'] - median),
        row['section_id'])))
group = [row for row in sections if row['appearance_mode'] == 'raw'
    and row['nearest_axis_angle_deg'] >= 40 and row not in chosen]
chosen.append(min(group, key=lambda row: (row['valid_fraction'], row['section_id'])))

atlas, _ = _decode_and_preprocess_allen_v6()
axis = np.arange(256, dtype=np.float32) / 256
yy, xx = np.meshgrid(axis, axis, indexing='ij')
tile, gap, margin, label = 184, 12, 16, 78
canvas = Image.new('RGB', (4 * tile + 3 * gap + 2 * margin,
    len(chosen) * (tile + label + margin) + margin), (235, 235, 235))
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=16)
small = ImageFont.load_default(size=13)
selection = []
for row_index, row in enumerate(chosen):
    record = records[row['section_id']]
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        observed = np.clip(arrays['inputs'][0], 0, 1)
        truth_state = arrays['target_state'].copy()
        truth_reflection = bool(arrays['reflection'])
        offsets = arrays['offsets_um'].copy()
        weights = arrays['weights'].copy()
    prior = candidates[(row['section_id'], row['prior_slot'])]
    selected = candidates[(row['section_id'], row['blind_selected_slot'])]
    images = []
    for state, reflected in ((truth_state, truth_reflection),
        (prior['input_state'], bool(prior['branch_id'] % 2)),
        (selected['iter2_state'], bool(selected['branch_id'] % 2))):
        centre, frame, basis = full_frame_state_to_components(
            torch.tensor(state, dtype=torch.float32))
        chart = np.stack((255 / 256 - xx if reflected else xx, yy), -1)
        plane = centre.numpy() + (chart - .5) @ (frame[:, :2] @ basis).numpy().T
        normal = frame[:, 2].numpy()
        intensity = np.zeros((256, 256), dtype=np.float32)
        support = np.zeros_like(intensity)
        for offset, weight in zip(offsets, weights):
            query = np.moveaxis((plane + offset * normal) / 25 - .5, -1, 0)
            intensity += weight * map_coordinates(atlas[0], query,
                order=1, mode='constant')
            support += weight * map_coordinates(atlas[1], query,
                order=1, mode='constant')
        images.append(np.clip(intensity / np.maximum(support, 1e-4), 0, 1))
    y = margin + row_index * (tile + label + margin)
    events = record['provenance']['one_shot_slide_artifacts_v3']['events']
    draw.text((margin, y), f"{row['angle_family']} | {row['nearest_axis_angle_deg']:.0f}° from nearest axis | "
        f"{row['appearance_mode']} | {row['valid_fraction']:.0%} valid", font=font,
        fill=(15, 15, 15))
    draw.text((margin, y + 24), f"prior {row['prior_input_um'] / 1000:.2f} mm; "
        f"106 selected {row['selected_iter2_um'] / 1000:.2f} mm | "
        f"events: {', '.join(key for key, value in events.items() if value) or 'none'}",
        font=small, fill=(40, 40, 40))
    for column, (image, title) in enumerate(zip(
        (images[0], observed, images[1], images[2]),
        ('true atlas plane', 'actual generated input', 'direct-model prior plane',
         '106 blind-selected plane'))):
        x = margin + column * (tile + gap)
        draw.text((x, y + 49), title, font=small, fill=(15, 15, 15))
        canvas.paste(Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
            .resize((tile, tile)), (x, y + label))
    selection.append({'section_id': row['section_id'], 'mode': row['appearance_mode'],
        'angle_family': row['angle_family'], 'angle_deg': row['nearest_axis_angle_deg'],
        'valid_fraction': row['valid_fraction'], 'events': events,
        'prior_error_um': row['prior_input_um'],
        'selected_error_um': row['selected_iter2_um'],
        'panel_file_sha256': record['sha256']})
out.mkdir(parents=True)
figure = out / 'actual_generator_atlas_and_106_predictions.png'
canvas.save(figure)
(out / 'selection.json').write_text(json.dumps({'rule':
    'median valid fraction for fixed appearance/family cells plus lowest-support raw >=40° case; no model-error selection',
    'evaluation_completed_sha256': hashlib.sha256(
        (evaluation / 'completed.json').read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256(
        (panel / 'records.jsonl').read_bytes()).hexdigest(),
    'figure_sha256': hashlib.sha256(figure.read_bytes()).hexdigest(),
    'rows': selection}, indent=2))
print(figure)
