"""Preview arbitrary cuts interpolated through one real TRAIN donor's serial images."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001/donor_168180530')
with np.load(root / 'geometry.npz') as geometry:
    state = geometry['state'].astype(np.float32)
images = np.load(root / 'images.npy', mmap_mode='r')[:, 0]
order = np.argsort(state[:, 0])
state = state[order]
volume = torch.from_numpy(np.asarray(images[order], dtype=np.float32))[None, None]
center, frame, basis = full_frame_state_to_components(torch.from_numpy(state))
center, frame, basis = center.numpy(), frame.numpy(), basis.numpy()
edge = frame[0, :, :2] @ basis[0]
step = (center[-1] - center[0]) / (len(state) - 1)
inverse = np.linalg.inv(np.column_stack((edge[:, 0], edge[:, 1], step)))
side = 192
y, x = np.meshgrid(np.arange(side) / side, np.arange(side) / side, indexing='ij')
midpoint = center[len(state) // 2]
u = .65 * edge[:, 0] + .8 * (center[-1] - center[0])
v = edge[:, 1]
planes = [('Original coronal section', None, None, None),
          ('Resampled original plane', midpoint, edge[:, 0], edge[:, 1]),
          ('Sagittal from real serial stack', np.array([6600., 4000., 5700.]),
           center[-1] - center[0], edge[:, 1]),
          ('Oblique from real serial stack', np.array([6600., 4000., 5700.]), u, v)]
fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
for axis, (title, point, width, height) in zip(axes.flat, planes):
    if point is None:
        image = volume[0, 0, len(state) // 2].numpy()
    else:
        world = point + (x[..., None] - .5) * width + (y[..., None] - .5) * height
        donor = (world - center[0]) @ inverse.T
        grid = np.stack(((donor[..., 0] + .5) * side / (side - 1) * 2 - 1,
                         (donor[..., 1] + .5) * side / (side - 1) * 2 - 1,
                         donor[..., 2] / (len(state) - 1) * 2 - 1), -1)
        image = F.grid_sample(volume, torch.from_numpy(grid[None, None].astype(np.float32)),
                              align_corners=True)[0, 0, 0].numpy()
        if title == 'Resampled original plane':
            print({'same_plane_mae': float(np.mean(np.abs(image - volume[0, 0, len(state) // 2].numpy())))})
    axis.imshow(image, cmap='gray', vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis('off')
fig.savefig('I:/AnatomyTracker/runs/real_serial_oblique_reslice_preview.png', dpi=140)
