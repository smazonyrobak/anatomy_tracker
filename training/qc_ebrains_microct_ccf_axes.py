"""Bounded, remote three-plane CCF-axis QC for one CC-BY EBRAINS microCT brain."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6

# Neuroglancer atlas-layer transform: display [x,y,z]=[456-ML,DV,528-AP] at 25 um.
# The registered microCT image is displayed in the same physical frame at 20.8 um.
base = 'https://rgw.cscs.ch/ebrains:d-256fb2ff-93b2-4f3f-abbd-ccee0e52b020/'
volume = ts.open({'driver': 'neuroglancer_precomputed',
                  'kvstore': {'driver': 'http', 'base_url': base},
                  'scale_index': 5}, read=True).result()
atlas, _ = _decode_and_preprocess_allen_v6()
ap, dv, ml = 266, 160, 228
z = round((528 - ap) * 25 / 20.8)
y = round(dv * 25 / 20.8)
x = round((456 - ml) * 25 / 20.8)
microct = [np.asarray(volume[:, :, z].read().result()).squeeze().T[:, ::-1],
           np.asarray(volume[x, :, :].read().result()).squeeze()[:, ::-1],
           np.asarray(volume[:, y, :].read().result()).squeeze()[::-1, ::-1].T]
reference = [atlas[0, ap, :, :], atlas[0, :, :, ml].T, atlas[0, :, dv, :]]
figure, axes = plt.subplots(3, 2, figsize=(10, 13), layout='constrained')
for row, (name, ccf, ct) in enumerate(zip(('coronal', 'sagittal', 'horizontal'),
                                          reference, microct)):
    axes[row, 0].imshow(ccf, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[row, 1].imshow(ct, cmap='gray', vmin=9270, vmax=27039, interpolation='nearest')
    axes[row, 0].set_title(f'Allen CCF template · {name}')
    axes[row, 1].set_title(f'Humbel registered microCT · {name}')
    for axis in axes[row]:
        axis.axis('off')
figure.suptitle('Matched physical positions using the published Neuroglancer axis transform\n'
                 f'CCF AP/DV/ML={ap}/{dv}/{ml} at 25 µm; CT X/Y/Z={x}/{y}/{z} at 20.8 µm\n'
                 'Gross-axis QC only; not a measured voxelwise registration error')
output = ROOT / 'reports/ebrains_microct_ccf_three_plane_qc.png'
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=150)
plt.close(figure)
print({'figure': str(output), 'atlas_shape': atlas.shape,
       'microct_shape': tuple(volume.shape), 'ccf_ap_dv_ml': (ap, dv, ml),
       'microct_xyz': (x, y, z), 'ct_intensity_range': [
           (int(array.min()), int(array.max())) for array in microct]})
