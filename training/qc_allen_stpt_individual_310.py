"""One-source, read-only coordinate and anatomy check for registered Allen STPT data."""
import os
from itertools import product
from pathlib import Path

os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import nrrd
import numpy as np
import matplotlib.pyplot as plt

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_PATH_V6


root = Path('I:/AnatomyTracker/data/allen_stpt_individual_qc_310')
source, source_header = nrrd.read(str(root / '472393704_red.nrrd'), index_order='F')
template, template_header = nrrd.read(str(root / 'average_template_25.nrrd'), index_order='F')
annotation, _ = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')
source_full = source.transpose(2, 1, 0)
source = source_full[::4, ::4, ::4].astype(np.float32)
template_full = template
template = template_full[::4, ::4, ::4].astype(np.float32)
brain = annotation[::4, ::4, ::4] != 0
print('source_shape', source_header['sizes'], 'template_shape', template_header['sizes'])
print('source_percentiles', np.percentile(source, (0, 25, 50, 75, 90, 99, 99.9)).tolist())
print('source_nonzero_inside_outside', float((source[brain] > 0).mean()),
      float((source[~brain] > 0).mean()))
for flips in product((False, True), repeat=3):
    candidate = source
    for axis, flip in enumerate(flips):
        if flip:
            candidate = np.flip(candidate, axis)
    src = np.log1p(candidate[brain])
    atlas = np.log1p(template[brain])
    correlation = float(np.corrcoef(src, atlas)[0, 1])
    print('flips', flips, 'log_intensity_correlation', correlation)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for col, index in enumerate((264, 160, 228)):
    for row, volume in enumerate((template_full, source_full)):
        section = (volume[index] if col == 0 else
                   volume[:, index] if col == 1 else volume[:, :, index])
        low, high = np.percentile(section, (5, 99.5))
        axes[row, col].imshow(section, cmap='gray', vmin=low, vmax=high)
        axes[row, col].set_axis_off()
        axes[row, col].set_title((('Pinned atlas', 'Individual STPT')[row]) +
                                 (' AP', ' DV', ' ML')[col])
fig.tight_layout()
fig.savefig('I:/AnatomyTracker/reports/allen_stpt_individual_qc_310.png', dpi=140)
plt.close(fig)
