"""Four independently drawn physical planes from the actual v4 generator."""

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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import sample_streaming_synthetic_v7_64_appearance_v3

torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
rng = torch.Generator().manual_seed(20261010130)
chosen = {}
attempts = 0
with torch.no_grad():
    while len(chosen) < 4:
        virtual = int(torch.randint(len(context['subjects']), (1,), generator=rng))
        seed = 20261010130000000 + attempts
        source = sample_streaming_synthetic_v7_64_appearance_v3(context, [virtual], seed, side=256)
        sample = sample_one_shot_slide_artifacts_v4(context, [virtual], seed, side=256,
                                                     source_sample=source)
        attempts += 1
        if not bool(sample['eligible'][0]):
            continue
        normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        axis = int(normal.abs().argmax())
        angle = float(torch.rad2deg(torch.acos(normal.abs().max().clamp_max(1))))
        mode = sample['provenance'][0]['mode']
        low = sample['provenance'][0]['one_shot_slide_artifacts_v4']['low_exposure']
        target = ('coronal raw, low exposure' if axis == 0 and angle < 20 and mode == 'raw' and low else
                  'sagittal black, high exposure' if axis == 2 and angle < 20 and mode == 'exact_black' and not low else
                  'horizontal brush, low exposure' if axis == 1 and angle < 20 and mode == 'imperfect_brush' and low else
                  'oblique raw, high exposure' if angle > 40 and mode == 'raw' and not low else None)
        if target is None or target in chosen:
            continue
        center, frame, basis = full_frame_state_to_components(sample['state'])
        y, x = torch.meshgrid(torch.arange(256, device='cuda') / 256,
                              torch.arange(256, device='cuda') / 256, indexing='ij')
        sx = (255 / 256 - x) if bool(sample['reflection'][0]) else x
        chart = torch.stack((sx, y), -1)
        plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        coordinates = plane[:, None] + sample['offsets'][:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(context['atlas'], coordinates,
                                                            (0., 0., 0.), (25., 25., 25.), sample['weights'])
        atlas = (rendered[0, 0] / rendered[0, 1].clamp_min(1e-4)).cpu().numpy()
        atlas[rendered[0, 1].cpu().numpy() < .05] = 0
        pristine = (source['clean'][0] / source['support'][0].clamp_min(1e-4)).cpu().numpy()
        pristine[source['support'][0].cpu().numpy() < .05] = 0
        observed = sample['inputs'][0, 0].cpu().numpy()
        chosen[target] = {'atlas': atlas, 'pristine': pristine, 'observed': observed,
            'seed': seed, 'virtual_index': virtual, 'angle_deg': angle, 'mode': mode,
            'exposure': sample['provenance'][0]['one_shot_slide_artifacts_v4']['exposure'],
            'physical_section_id': sample['provenance'][0]['physical_section_id']}

labels = ('coronal raw, low exposure', 'sagittal black, high exposure',
          'horizontal brush, low exposure', 'oblique raw, high exposure')
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for row, label in enumerate(labels):
    item = chosen[label]
    images = (item['atlas'], item['pristine'], item['observed'], item['observed'])
    for col, image in enumerate(images):
        upper = max(float(np.quantile(image, .995)), .001) if col == 3 else 1.
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=upper)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"{label}\n{item['angle_deg']:.1f} deg", fontsize=9)
for col, title in enumerate(('atlas at fitted plane', 'pre-artifact virtual tissue',
                             'observed, absolute [0,1]', 'observed, display stretch')):
    axes[0, col].set_title(title, fontsize=10)
fig.tight_layout()
out = root / 'previews/actual_v4_independent_planes_20261010.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
receipt = {'attempts': attempts, 'image': str(out),
    'rows': [{key: value for key, value in chosen[label].items()
              if key not in ('atlas', 'pristine', 'observed')} for label in labels]}
(out.with_suffix('.json')).write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
