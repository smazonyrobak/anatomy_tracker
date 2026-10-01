"""Six distinct Allen-atlas planes for visual review before one-shot training."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
OUT = ROOT / 'previews/one_shot_slices_20261001_v4'
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / 'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import nrrd
import numpy as np
from PIL import Image, ImageDraw

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    ANNOTATION_PATH_V6, TEMPLATE_PATH_V6, TEMPLATE_RAW_SHA256_V6,
    _build_support_index_v6,
)
from training.arbitrary_plane_rendered_generator import (
    make_finite_arbitrary_plane_render_from_context, prepare_finite_render_context,
)
from training.arbitrary_plane_finite_slab_v4 import make_finite_slab_render_v4
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE, make_arbitrary_plane_synthetic_realization,
)

template = nrrd.read(str(TEMPLATE_PATH_V6), index_order='F')[0]
annotation = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')[0]
support = _build_support_index_v6(annotation)
context = prepare_finite_render_context(
    template, annotation, support, scalar_source_uri=str(TEMPLATE_PATH_V6),
    scalar_source_sha256=TEMPLATE_RAW_SHA256_V6,
    template_decoder='pynrrd 1.1.3', template_index_order='F',
    annotation_decoder='pynrrd 1.1.3', annotation_index_order='F',
)

cases = [
    ('AP-like', 0, .78, 0.00, 0.00),
    ('DV-like', 1, .78, 0.12, 0.02),
    ('ML-like', 2, .78, 0.27, 0.03),
    ('oblique A', -1, .72, 0.43, 0.03),
    ('oblique B', -1, .72, 0.62, 0.04),
    ('AP-like B', 0, .78, 0.82, 0.02),
]
sheet = Image.new('RGB', (4 * 330, 3 * 350), 'white')
records = []
used = set()
for index, (name, axis, threshold, background, texture) in enumerate(cases):
    for seed in range(10000 + index * 100, 10100 + index * 100):
        parent = make_finite_arbitrary_plane_render_from_context(
            context, 'train', seed, (256, 256), sample_index=index,
            margin_um=(250., 250.), minimum_brain_pixels=3000,
            animal_id=f'synthetic-preview-{index}', specimen_id=f'synthetic-preview-{index}',
            experiment_id=f'synthetic-preview-{index}',
        )
        normal = np.asarray(parent['geometry']['normal_rp2_ap_dv_ml'])
        matched = abs(normal[axis]) >= threshold if axis >= 0 else np.abs(normal).max() <= threshold
        if matched and parent['finite_plane_render_id'] not in used:
            break
    else:
        raise RuntimeError(f'No distinct {name} plane in the fixed 100-seed window')
    used.add(parent['finite_plane_render_id'])
    slab = make_finite_slab_render_v4(
        parent, context, nominal_cut_thickness_um=(35., 50., 70., 85., 60., 45.)[index]
    )['artifact']['slab_observation_v4']
    background_ranges = {key: [background, background] for key in
        ('dark-field-like', 'brightfield-glass-like', 'neutral-scanner-stress')}
    for root_seed in range(20000 + index * 100, 20030 + index * 100):
        tile_stitching = index == 4
        sample = make_arbitrary_plane_synthetic_realization(
            parent, support, slab_observation_v4=slab, root_seed=root_seed,
            outline_mode=ABSENT_OUTLINE,
            config_overrides={'g1': {'identity_probability': 0.0},
                'g2': {'background_base': [background, background],
                       'finite_background_base_by_family': background_ranges,
                       'background_field_std': [texture, texture],
                       'background_noise_std': [0.0, 0.01 if background else 0.0],
                       'finite_tile_stitch_probability': float(tile_stitching),
                       'finite_tile_gain_log_std': [0.08, 0.08],
                       'finite_tile_offset_std': [0.02, 0.02]},
                'g3': {'event_count_probabilities': [0.0, 0.6, 0.4],
                       'finite_transform_compression_probability': 0.6}},
        )
        desired_event = {3: 'mounting-bubble-ring', 4: 'tear-or-crack'}.get(index)
        if desired_event is None or desired_event in [event['type'] for event in sample['g3']['parameters']['events']]:
            break
    observed = sample['arrays']['model_input_image']
    assert observed.shape == (256, 256)
    assert not sample['outline']['parameters']['outline_available']
    Image.fromarray(np.uint8(np.rint(observed * 255)), 'L').save(OUT / f'slice_{index + 1:02d}.png')
    atlas = slab['observed_scalar_float32']
    atlas_mask = slab['slab_observable_support_mask']
    lower, upper = np.quantile(atlas[atlas_mask], [0.01, 0.99])
    atlas_display = np.where(atlas_mask, np.clip((atlas - lower) / (upper - lower), 0, 1), 0)
    Image.fromarray(np.uint8(np.rint(atlas_display * 255)), 'L').save(OUT / f'atlas_{index + 1:02d}.png')
    x, y = (index % 2) * 660 + 10, (index // 2) * 350 + 10
    for column, values in enumerate((observed, atlas_display)):
        tile = Image.fromarray(np.uint8(np.rint(values * 255)), 'L').convert('RGB').resize((310, 310))
        sheet.paste(tile, (x + column * 330, y))
    label = ImageDraw.Draw(sheet)
    label.text((x, y + 315), f'{index + 1}: observed {name}, background {background:.2f}', fill='black')
    label.text((x + 330, y + 315), 'matching clean atlas plane', fill='black')
    records.append({'image': f'slice_{index + 1:02d}.png', 'plane': name, 'seed': seed,
        'atlas_image': f'atlas_{index + 1:02d}.png',
        'appearance_seed': root_seed, 'section_thickness_um': (35., 50., 70., 85., 60., 45.)[index],
        'normal_ap_dv_ml': normal.tolist(), 'background_level': background,
        'damage_events': [event['type'] for event in sample['g3']['parameters']['events']],
        'tile_stitching': sample['g2']['parameters']['tile_stitching'],
        'finite_plane_render_id': parent['finite_plane_render_id'],
        'synthetic_realization_id': sample['synthetic_realization_id']})
    print(json.dumps({'sample': index + 1, 'plane': name, 'damage': records[-1]['damage_events']}), flush=True)

sheet.save(OUT / 'contact_sheet.png')
(OUT / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf8')
print(str(OUT / 'contact_sheet.png'), flush=True)
