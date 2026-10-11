"""Six independent TRAIN generator draws using Allen or KimLab LSFM intensity."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR'] = str(root / 'cache')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.kimlab_lsfm_appearance_232 import bind_lsfm_appearance_232, select_atlas_appearance_232
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.support_aware_framing_217 import sample_support_aware_framing_217


torch.set_num_threads(4)
context = bind_lsfm_appearance_232(load_streaming_synthetic_v7_64(device='cuda'))
rng = np.random.default_rng(20261011232)
items, attempts = [], []
with torch.no_grad():
    for base in rng.permutation(64):
        seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
        variant, source_contrast = select_atlas_appearance_232(context, seed)
        source = sample_panel_geometry_train_166(variant, [int(base)], seed, side=256)
        framed = sample_support_aware_framing_217(source, seed)
        sample = sample_one_shot_slide_artifacts_v4(variant, [int(base)], seed, side=256,
                                                     source_sample=framed)
        record = sample['provenance'][0]
        attempts.append({'base_index': int(base), 'section_seed': seed,
            'physical_section_id': record['physical_section_id'],
            'source_contrast': source_contrast, 'eligible': bool(sample['eligible'][0]),
            'visible_tissue_fraction': float(sample['visible'][0].mean())})
        if not bool(sample['eligible'][0]):
            continue
        center, frame, basis = full_frame_state_to_components(sample['state'])
        normal = frame[0, :, 2]
        angle = float(torch.rad2deg(torch.acos(normal.abs().max().clamp_max(1))))
        y, x = torch.meshgrid(torch.arange(256, device='cuda') / 256,
                              torch.arange(256, device='cuda') / 256, indexing='ij')
        sx = (255 / 256 - x) if bool(sample['reflection'][0]) else x
        chart = torch.stack((sx, y), -1)
        plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        slab = plane[:, None] + sample['offsets'][:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(variant['atlas'], slab,
            (0., 0., 0.), (25., 25., 25.), sample['weights'])
        atlas = (rendered[0, 0] / rendered[0, 1].clamp_min(1e-4)).cpu().numpy()
        atlas[rendered[0, 1].cpu().numpy() < .05] = 0
        items.append({'atlas': atlas, 'before': framed['inputs'][0, 0].cpu().numpy(),
            'after': sample['inputs'][0, 0].cpu().numpy(),
            'base_index': int(base), 'section_seed': seed,
            'physical_section_id': record['physical_section_id'],
            'source_contrast': source_contrast, 'atlas_binding': record['atlas_binding'],
            'normal_ap_dv_ml': normal.cpu().tolist(), 'nearest_cardinal_angle_deg': angle,
            'mode': record['mode'], 'artifact_events': record['one_shot_slide_artifacts_v3']['events'],
            'exposure': record['one_shot_slide_artifacts_v4']['exposure']})
        if len(items) == 6:
            break

assert len(items) == 6 and len({item['physical_section_id'] for item in items}) == 6
fig, axes = plt.subplots(6, 4, figsize=(12, 18))
for row, item in enumerate(items):
    for col, key in enumerate(('atlas', 'before', 'after', 'after')):
        image = item[key]
        upper = max(float(np.quantile(image, .995)), .01) if col == 3 else 1.
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=upper)
        axes[row, col].axis('off')
    axes[row, 0].text(.02, .98,
        f"{item['source_contrast']} | base {item['base_index']} | {item['nearest_cardinal_angle_deg']:.1f}°",
        transform=axes[row, 0].transAxes, va='top', color='white', fontsize=9,
        bbox={'facecolor': 'black', 'alpha': .7, 'pad': 3})
for col, title in enumerate(('atlas at fitted CCF plane', 'pre-slide-artifact image',
                              'final input, absolute [0,1]', 'final input, display stretch')):
    axes[0, col].set_title(title, fontsize=10)
fig.tight_layout()
out = root / 'previews/kimlab_lsfm_generator_independent_232.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
protocol = Path('docs/publication/KIMLAB_LSFM_APPEARANCE_PREVIEW_232_PROTOCOL_20261011.md')
scripts = ('kimlab_lsfm_appearance_232.py', 'render_kimlab_lsfm_generator_232.py',
           'arbitrary_plane_panel_geometry_train_166.py', 'support_aware_framing_217.py',
           'arbitrary_plane_one_shot_slide_artifacts_v3.py',
           'arbitrary_plane_one_shot_slide_artifacts_v4.py')
receipt = {'image': str(out), 'image_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'script_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in scripts},
    'lsfm_binding': context['lsfm_binding_232'],
    'allen_binding': context['provenance']['atlas_binding'],
    'attempts': attempts, 'accepted_distinct_train_bases': len(items),
    'rows': [{k: v for k, v in item.items() if k not in ('atlas', 'before', 'after')}
             for item in items],
    'limitations': 'Population-average registered source contrast; approximate TRAIN inverse-map targets; no physical arbitrary-angle validation or new donor identity.'}
out.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'image': str(out), 'accepted': len(items), 'attempts': len(attempts),
                  'sources': [item['source_contrast'] for item in items]}, indent=2))
