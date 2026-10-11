"""Inspect one acquired DAPI specimen volume before using it for training."""

import json
import os
from pathlib import Path

os.environ['TEMP'] = os.environ['TMP'] = 'I:/AnatomyTracker/tmp'
os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

source = Path('I:/AnatomyTracker/data/g_node_dapi_196/BM03.nii')
volume = nib.load(source)
spacing = np.asarray(volume.header.get_zooms()[:3])
x, y, z = (size // 2 for size in volume.shape)
planes = (np.asarray(volume.dataobj[:, y, :]).T,
          np.asarray(volume.dataobj[x, :, :]).T,
          np.asarray(volume.dataobj[:, :, z]).T)
scales = ((spacing[0], spacing[2]), (spacing[1], spacing[2]),
          (spacing[0], spacing[1]))
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
metadata = []
for axis, image, scale, name in zip(axes, planes, scales,
                                     ('physical coronal', 'reconstructed sagittal',
                                      'reconstructed horizontal')):
    upper = float(np.percentile(image, 99.5))
    axis.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=upper,
                extent=(0, image.shape[1] * scale[0], 0, image.shape[0] * scale[1]),
                aspect='equal')
    axis.set_title(name)
    axis.set_xlabel('mm')
    metadata.append({'plane': name, 'shape': list(image.shape),
                     'nonzero_fraction': float(np.mean(image > 0)),
                     'p50_p90_p99_p995': np.percentile(image, [50, 90, 99, 99.5]).tolist()})
out = Path('I:/AnatomyTracker/previews/gnode_BM03_native_orthoplanes_196.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
plt.close(fig)
record = {'source': str(source), 'shape': volume.shape,
          'dtype': str(volume.get_data_dtype()), 'voxel_spacing_mm': spacing.tolist(),
          'affine': volume.affine.tolist(), 'planes': metadata,
          'caveat': 'Sagittal and horizontal are virtual reslices across 100-um acquired coronal spacing, not physically cut oblique sections.'}
out.with_suffix('.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
