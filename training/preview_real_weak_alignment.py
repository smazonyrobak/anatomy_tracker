"""Compare three real TRAIN sections with their weak Allen atlas references."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_reserved_real_stream_v8 import sample_reserved_real_train

root = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001')
atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0])
side = 192
y, x = torch.meshgrid(torch.arange(side) / side, torch.arange(side) / side, indexing='ij')
fig, axes = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
for axis, donor_id in zip(axes, ('168180530', '606157233', '126161218')):
    directory = root / f'donor_{donor_id}'
    with np.load(directory / 'geometry.npz') as geometry:
        state = geometry['state'].copy()
        thickness = geometry['thickness_um'].copy()
    identities = [json.loads(line) for line in (directory / 'records.jsonl').open()]
    donor = {'images': directory / 'images.npy', 'state': state,
             'thickness_um': thickness, 'identities': identities}
    row = len(state) // 2
    sample = sample_reserved_real_train({'donors': [donor]}, 0, [row], device='cpu')
    center, frame, basis = full_frame_state_to_components(sample['state'])
    sx = (side - 1) / side - x if int(sample['reflection'][0]) else x
    chart = torch.stack((sx, y), -1)[None]
    plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                     basis[:, None, None], chart)
    offsets = torch.linspace(-.5, .5, 9)[None] * sample['thickness_um'][:, None]
    weights = torch.ones(1, 9)
    weights[:, [0, -1]] = .5
    coordinates = plane[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
    with torch.no_grad():
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights)[0].numpy()
    reference = rendered[0] / np.maximum(rendered[1], 1e-4)
    reference[rendered[1] < .05] = 0
    axis[0].imshow(sample['inputs'][0, 0], cmap='gray', vmin=0, vmax=1)
    axis[1].imshow(reference, cmap='gray', vmin=0, vmax=1)
    axis[0].set_title(f'Real TRAIN donor {donor_id}')
    axis[1].set_title('Weak Allen atlas plane')
    for panel in axis:
        panel.axis('off')
fig.savefig('I:/AnatomyTracker/runs/real_weak_atlas_alignment_preview.png', dpi=140)
