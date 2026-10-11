"""Check MouseLight-registered anatomy against Allen's AP/DV/ML array axes."""

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
from scipy.ndimage import map_coordinates

root = Path('I:/AnatomyTracker')
registered, header = nrrd.read(root / 'data/janelia_mouselight_198/Transformed_2017-01-15.nrrd')
atlas, _ = nrrd.read(root / 'data/Allen Brain Atlas 25um/average_template_25.nrrd')
axis0 = np.linspace(0, registered.shape[0] - 1, 256)
axis1 = np.linspace(0, registered.shape[1] - 1, 256)
y, x = np.meshgrid(axis1, axis0, indexing='xy')
slice_index = registered.shape[2] // 2
world = (header['space origin'][None, None] + x[..., None] * header['space directions'][0]
         + y[..., None] * header['space directions'][1]
         + slice_index * header['space directions'][2])
atlas_index = world[..., [2, 1, 0]] / 25
observed = map_coordinates(registered, (x, y, np.full_like(x, slice_index)),
                           order=1, mode='constant', cval=0)
reference = map_coordinates(atlas, tuple(atlas_index[..., k] for k in range(3)),
                            order=1, mode='constant', cval=0)
fig, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
for axis, image, title in zip(axes, (observed, reference),
                               ('MouseLight registered', 'Allen CCF at permuted world plane')):
    axis.imshow(image.T, cmap='gray', vmin=0, vmax=np.percentile(image, 99.5),
                origin='lower')
    axis.axis('off')
    axis.set_title(title)
out = root / 'previews/mouselight_ccf_axis_binding_200.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=145)
plt.close(fig)
record = {'registered_source': '2017-01-15', 'slice_index': slice_index,
          'atlas_source': 'Allen average_template_25.nrrd',
          'ccf_sample_inbounds_fraction': float(np.mean(np.all(
              (atlas_index >= 0) & (atlas_index < np.asarray(atlas.shape)), axis=-1))),
          'registered_nonzero_fraction': float(np.mean(observed > 0)),
          'atlas_nonzero_fraction': float(np.mean(reference > 0)),
          'basis': 'Janelia registered world xyz maps to Allen raw array AP/DV/ML by z/y/x permutation; Allen header LPS annotation is biologically misleading. This checks coordinate axes, not registration accuracy or oblique physical-section validity.'}
out.with_suffix('.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
