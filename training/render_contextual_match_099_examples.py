"""One unbiased six-section grid: atlas truth, generator input, frozen 094 guess."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    _decode_and_preprocess_allen_v6, TEMPLATE_RAW_SHA256_V6,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker')
panel = root / 'data/contextual_match_099_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/contextual_match_099_fresh_development_eval'
out = root / 'reports/contextual_match_099_examples'
assert not out.exists()
assert json.loads((panel / 'completed.json').read_text())['physical_sections'] == 256
assert json.loads((evaluation / 'completed.json').read_text())['eligible_sections_per_step'] > 0
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
candidates = [json.loads(line) for line in (evaluation / 'candidates.jsonl').open()]
chosen_by_section = {}
for candidate in candidates:
    if candidate['step'] == 0 and (candidate['section_id'] not in chosen_by_section or
            candidate['parent_score'] > chosen_by_section[candidate['section_id']]['parent_score']):
        chosen_by_section[candidate['section_id']] = candidate
atlas, _ = _decode_and_preprocess_allen_v6()
families = ('AP', 'DV', 'ML')
selected = []
for family_index in range(3):
    eligible = [row for row in records if row['eligible'] and
                row['section_id'] in chosen_by_section and
                int(np.argmax(np.abs(row['plane_normal_ap_dv_ml']))) == family_index and
                np.degrees(np.arccos(max(np.abs(row['plane_normal_ap_dv_ml'])))) >= 25 and
                row['valid_pixels'] >= 65536 * .05]
    assert len(eligible) >= 2
    event_count = {row['section_id']: sum(row['provenance']['one_shot']['events'].values()) +
                   int(row['provenance']['appearance']['damage']) for row in eligible}
    low = min(eligible, key=lambda row: (event_count[row['section_id']],
        abs(row['valid_pixels'] / 65536 - .3), row['section_id']))
    high = max((row for row in eligible if row['section_id'] != low['section_id']),
        key=lambda row: (event_count[row['section_id']],
            np.degrees(np.arccos(max(np.abs(row['plane_normal_ap_dv_ml'])))),
            row['section_id']))
    selected.extend((low, high))

tile, gap, margin, label = 192, 12, 18, 80
canvas = Image.new('RGB', (3 * tile + 2 * gap + 2 * margin,
                           6 * (tile + label) + 7 * margin), (235, 235, 235))
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=17)
small = ImageFont.load_default(size=14)
axis = np.arange(256, dtype=np.float32) / 256
yy, xx = np.meshgrid(axis, axis, indexing='ij')
selection = []
for row_index, record in enumerate(selected):
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        observed = np.clip(arrays['inputs'][0], 0, 1)
        target_state = torch.from_numpy(arrays['target_state']).float()
        offsets = arrays['offsets_um']
        weights = arrays['weights'] / arrays['weights'].sum()
        true_reflection = bool(arrays['reflection'])
    guessed = chosen_by_section[record['section_id']]
    atlas_images = []
    for state, reflected in ((target_state, true_reflection),
                             (torch.tensor(guessed['fitted_state']),
                              bool(guessed['branch_id'] % 2))):
        centre, frame, basis = full_frame_state_to_components(state)
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
        atlas_images.append(np.clip(intensity / np.maximum(support, 1e-4), 0, 1))
    angle = np.degrees(np.arccos(max(np.abs(record['plane_normal_ap_dv_ml']))))
    y = margin + row_index * (tile + label + margin)
    draw.text((margin, y), f'{families[np.argmax(np.abs(record["plane_normal_ap_dv_ml"]))]} family | '
              f'{angle:.0f}° from nearest cardinal | {record["appearance_mode"]}',
              font=font, fill=(20, 20, 20))
    draw.text((margin, y + 25),
              f'{record["valid_pixels"] / 65536:.0%} valid tissue | '
              f'{sum(record["provenance"]["one_shot"]["events"].values())} artifact events | '
              f'094 selected {guessed["mapped96_error_um"] / 1000:.1f} mm mapped error',
              font=small, fill=(45, 45, 45))
    for column, (image, title) in enumerate(zip(
            (atlas_images[0], observed, atlas_images[1]),
            ('true atlas plane', 'actual generator input', '094 guessed atlas plane'))):
        x = margin + column * (tile + gap)
        draw.text((x, y + 52), title, font=small, fill=(20, 20, 20))
        canvas.paste(Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
                     .resize((tile, tile)), (x, y + label))
    selection.append({'section_id': record['section_id'], 'file_sha256': record['sha256'],
                      'nearest_family': families[np.argmax(np.abs(record['plane_normal_ap_dv_ml']))],
                      'angle_from_nearest_cardinal_deg': float(angle),
                      'valid_pixels': record['valid_pixels'],
                      'appearance_mode': record['appearance_mode'],
                      'events': record['provenance']['one_shot']['events'],
                      'selected_branch_id': guessed['branch_id'],
                      'selected_mapped96_error_um': guessed['mapped96_error_um']})
out.mkdir(parents=True)
image_path = out / 'actual_generator_true_and_guessed_planes.png'
canvas.save(image_path)
(out / 'selection.json').write_text(json.dumps({
    'selection': 'for each nearest-axis family, eligible >=25deg and >=5% valid pixels; minimum and maximum artifact-count independent sections, deterministic tie-breaks; no selection by model accuracy',
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'evaluation_candidates_sha256': hashlib.sha256((evaluation / 'candidates.jsonl').read_bytes()).hexdigest(),
    'atlas_template_sha256': TEMPLATE_RAW_SHA256_V6,
    'rows': selection,
    'grid_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
}, indent=2))
print(image_path)
