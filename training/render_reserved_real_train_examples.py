"""Show six seeded, distinct-donor images from the frozen real TRAIN pool."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001')
manifest = json.loads((root / 'completed.json').read_text())
rng = np.random.default_rng(20261002)
donors = rng.choice(sorted(manifest['donor_receipts']), 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
for axis, donor in zip(axes.flat, donors):
    images = np.load(root / f'donor_{donor}' / 'images.npy', mmap_mode='r')
    index = int(rng.integers(len(images)))
    axis.imshow(images[index, 0], cmap='gray', vmin=0, vmax=1)
    axis.set_title(f'Donor {donor} · section {index}')
    axis.axis('off')
fig.savefig('I:/AnatomyTracker/runs/reserved_real_train_examples.png', dpi=145)

sampled = np.random.default_rng(20261002).choice(
    sorted(manifest['donor_receipts']), 100, replace=False)
angles = []
for donor in sampled:
    with np.load(root / f'donor_{donor}' / 'geometry.npz') as geometry:
        state = torch.from_numpy(geometry['state'].astype(np.float32))
    normal_ap = full_frame_state_to_components(state)[1][:, 0, 2].abs().numpy()
    angles.extend(np.rad2deg(np.arccos(np.clip(normal_ap, 0, 1))))
angles = np.asarray(angles)
print(json.dumps({'sampled_donors': len(sampled), 'sections': len(angles),
                  'angle_to_coronal_normal_deg_quantiles':
                      np.percentile(angles, [0, 10, 50, 90, 99, 100]).round(2).tolist(),
                  'fraction_over_20deg': float(np.mean(angles > 20))}))
