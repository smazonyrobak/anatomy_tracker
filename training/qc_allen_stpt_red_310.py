"""One-volume QC of Allen's reference-aligned red-background STPT channel."""
import os
from itertools import product
from pathlib import Path
from zipfile import ZipFile

os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib.pyplot as plt
import nrrd
import numpy as np

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_PATH_V6, TEMPLATE_PATH_V6


archive = Path('I:/AnatomyTracker/data/allen_stpt_red_310_001/156198187_607436902.zip')
with ZipFile(archive) as source:
    header = source.read('resampled_red.mhd').decode()
    assert 'DimSize = 528 320 456' in header
    assert 'ElementSpacing = 25 25 25' in header
    assert 'Offset = 0 0 0' in header
    assert 'ElementType = MET_USHORT' in header
    red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
        (456, 320, 528)).transpose(2, 1, 0)
template, _ = nrrd.read(str(TEMPLATE_PATH_V6), index_order='F')
annotation, _ = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')
brain = annotation != 0
print('shape', red.shape, 'red_percentiles', np.percentile(red, (0, 25, 50, 75, 90, 99)).tolist())
print('red_inside_outside_median', float(np.median(red[brain])), float(np.median(red[~brain])))
small_red = np.log1p(red[::4, ::4, ::4].astype(np.float32))
small_atlas = np.log1p(template[::4, ::4, ::4].astype(np.float32))
small_brain = brain[::4, ::4, ::4]
for flips in product((False, True), repeat=3):
    candidate = small_red
    for axis, flip in enumerate(flips):
        if flip:
            candidate = np.flip(candidate, axis)
    print('flips', flips, 'brain_correlation',
          float(np.corrcoef(candidate[small_brain], small_atlas[small_brain])[0, 1]))

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for col, index in enumerate((264, 160, 228)):
    mask = (brain[index] if col == 0 else brain[:, index] if col == 1 else brain[:, :, index])
    for row, volume in enumerate((template, red)):
        section = (volume[index] if col == 0 else
                   volume[:, index] if col == 1 else volume[:, :, index])
        high = np.percentile(section[mask], 99.5)
        axes[row, col].imshow(section, cmap='gray', vmin=0, vmax=high)
        axes[row, col].set_axis_off()
        axes[row, col].set_title((('Pinned atlas', 'Individual red background')[row]) +
                                 (' AP', ' DV', ' ML')[col])
fig.tight_layout()
fig.savefig('I:/AnatomyTracker/reports/allen_stpt_red_310.png', dpi=140)
plt.close(fig)
