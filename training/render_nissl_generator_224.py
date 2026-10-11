"""Show independent arbitrary-plane v4 generator draws with either Allen intensity volume."""

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

from training.allen_nissl_appearance_224 import bind_nissl_appearance_224, select_atlas_appearance_224
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.support_aware_framing_217 import sample_support_aware_framing_217


torch.set_num_threads(4)
context = bind_nissl_appearance_224(load_streaming_synthetic_v7_64(device='cuda'))
rng = np.random.default_rng(20261011224)
items = []
attempts = 0
with torch.no_grad():
    while len(items) < 6:
        base = int(rng.integers(60))
        seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
        variant, modality = select_atlas_appearance_224(context, seed)
        source = sample_panel_geometry_train_166(variant, [base], seed, side=256)
        framed = sample_support_aware_framing_217(source, seed)
        sample = sample_one_shot_slide_artifacts_v4(variant, [base], seed, side=256,
                                                     source_sample=framed)
        attempts += 1
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
        before = framed['inputs'][0, 0].cpu().numpy()
        after = sample['inputs'][0, 0].cpu().numpy()
        record = sample['provenance'][0]
        items.append({'atlas': atlas, 'before': before, 'after': after,
            'seed': seed, 'base_index': base, 'modality': modality,
            'physical_section_id': record['physical_section_id'],
            'normal_ap_dv_ml': normal.cpu().tolist(), 'nearest_cardinal_angle_deg': angle,
            'visible_tissue_fraction': float(sample['visible'][0].mean()),
            'background_mode': record['mode'],
            'artifacts': record['one_shot_slide_artifacts_v3']['events']})

fig, axes = plt.subplots(len(items), 4, figsize=(12, 3 * len(items)))
for row, item in enumerate(items):
    for col, key in enumerate(('atlas', 'before', 'after', 'after')):
        image = item[key]
        upper = max(float(np.quantile(image, .995)), .01) if col == 3 else 1.
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=upper)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"{item['modality']}\n{item['nearest_cardinal_angle_deg']:.1f}°",
                           fontsize=9)
for col, title in enumerate(('atlas at labeled plane', 'pre-artifact synthetic',
                              'final absolute [0,1]', 'final display stretch')):
    axes[0, col].set_title(title, fontsize=10)
fig.tight_layout()
out = root / 'previews/nissl_generator_independent_224.png'
fig.savefig(out, dpi=150)
plt.close(fig)
receipt = {'image': str(out), 'image_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'attempts': attempts, 'accepted_distinct_physical_planes': len(items),
    'nissl_binding': context['nissl_binding_224'],
    'rows': [{k: v for k, v in item.items() if k not in ('atlas', 'before', 'after')}
             for item in items]}
out.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
