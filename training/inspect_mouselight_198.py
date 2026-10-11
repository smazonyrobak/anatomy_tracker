"""Inspect one acquired 3D MouseLight specimen before any training use."""

import json
import os
from pathlib import Path

os.environ['TEMP'] = os.environ['TMP'] = 'I:/AnatomyTracker/tmp'
os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nrrd
import numpy as np

source = Path('I:/AnatomyTracker/data/janelia_mouselight_198/2017-01-15.nrrd')
volume, header = nrrd.read(source)
spacing = np.linalg.norm(header['space directions'], axis=1) / 1000
x, y, z = (size // 2 for size in volume.shape)
planes = (volume[:, :, z].T, volume[:, y, :].T, volume[x, :, :].T)
scales = ((spacing[0], spacing[1]), (spacing[0], spacing[2]),
          (spacing[1], spacing[2]))
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
metadata = []
for axis, image, scale, name in zip(axes, planes, scales,
                                     ('L–P plane', 'L–S plane', 'P–S plane')):
    upper = float(np.percentile(image, 99.5))
    axis.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=upper,
                extent=(0, image.shape[1] * scale[0], 0, image.shape[0] * scale[1]),
                aspect='equal')
    axis.set_title(name)
    axis.set_xlabel('mm')
    metadata.append({'plane': name, 'shape': list(image.shape),
                     'nonzero_fraction': float(np.mean(image > 0)),
                     'p50_p90_p99_p995': np.percentile(image, [50, 90, 99, 99.5]).tolist()})
out = Path('I:/AnatomyTracker/previews/mouselight_2017-01-15_native_orthoplanes_198.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
plt.close(fig)
record = {'source': str(source), 'shape': volume.shape, 'dtype': str(volume.dtype),
          'axis_spacing_mm': spacing.tolist(), 'space': header['space'],
          'space_origin': header['space origin'].tolist(), 'planes': metadata,
          'caveat': 'Native NRRD spacing is approximately 8 x 8 x 32 um, not isotropic; registered CCF transform was not loaded.'}
out.with_suffix('.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
