"""Inspect one TRAIN animal's acquired sections against its authored CCF transform."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


root = Path('I:/AnatomyTracker')
source = root / 'data/allen_stpt_registered_volume_207_train_001'
report = root / 'reports/allen_stpt_native_207_train_001'
report.mkdir(parents=True, exist_ok=True)
metadata = json.loads((source / 'dataset_metadata.json').read_text())['msg'][0]
alignment = metadata['alignment3d']
inverse = np.array([alignment[f'tvr_{i:02d}'] for i in range(12)])
inverse_matrix, inverse_offset = inverse[:9].reshape(3, 3), inverse[9:]
field = np.memmap(source / 'dfmfld.raw', dtype='<f4', mode='r', shape=(456, 320, 528, 3))
registered = np.memmap(source / 'resampled_red.raw', dtype='<u2', mode='r',
                       shape=(528, 320, 456), order='F')
atlas, _ = nrrd.read(root / 'data/Allen Brain Atlas 25um/average_template_25.nrrd',
                     index_order='F')
fig, axes = plt.subplots(4, 3, figsize=(11, 12), layout='constrained')
rows = []

for row, section in enumerate((40, 60, 80, 100)):
    native = np.asarray(Image.open(source / f'native_section_{section}_ds5.jpg'))[..., 0]
    section_metadata = json.loads((source / f'native_section_{section}_metadata.json').read_text())['msg'][0]
    assert section_metadata['section_number'] == section
    assert section_metadata['alignment2d']['tsv_00'] == section_metadata['alignment2d']['tsv_03'] == .35
    xx, yy = np.meshgrid(np.linspace(0, native.shape[1] - 1, 320),
                         np.linspace(0, native.shape[0] - 1, 240))
    observed = map_coordinates(native, (yy, xx), order=1)
    native_points = np.column_stack((xx.ravel() * 11.2, yy.ravel() * 11.2,
                                     np.full(xx.size, section * 100.)))
    affine_ccf = native_points @ inverse_matrix.T + inverse_offset
    ccf = affine_ccf.copy()
    for _ in range(6):
        index = (ccf[:, 2] / 25, ccf[:, 1] / 25, ccf[:, 0] / 25)
        displacement = np.column_stack([map_coordinates(field[..., axis], index,
                                                         order=1, mode='nearest')
                                        for axis in range(3)])
        ccf = affine_ccf - displacement
    index = (ccf[:, 2] / 25, ccf[:, 1] / 25, ccf[:, 0] / 25)
    displacement = np.column_stack([map_coordinates(field[..., axis], index,
                                                     order=1, mode='nearest')
                                    for axis in range(3)])
    residual = np.linalg.norm(ccf + displacement - affine_ccf, axis=1)
    atlas_index = (ccf[:, 0] / 25, ccf[:, 1] / 25, ccf[:, 2] / 25)
    pulled_red = map_coordinates(registered, atlas_index, order=1, mode='constant').reshape(240, 320)
    pulled_atlas = map_coordinates(atlas, atlas_index, order=1, mode='constant').reshape(240, 320)
    support = pulled_atlas > 20
    rows.append({'section': section, 'image_id': section_metadata['id'],
                 'valid_pixels': int(support.sum()),
                 'registered_red_pearson': float(np.corrcoef(observed[support], pulled_red[support])[0, 1]),
                 'atlas_pearson': float(np.corrcoef(observed[support], pulled_atlas[support])[0, 1]),
                 'inverse_field_residual_p95_um': float(np.percentile(residual, 95))})
    for col, (image, title) in enumerate(((observed, 'Acquired red'),
                                          (pulled_red, 'Registered red → native'),
                                          (pulled_atlas, 'CCF template → native'))):
        high = np.percentile(image[support], 99)
        axes[row, col].imshow(np.clip(image / high, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'Section {section} · {title}')
        axes[row, col].axis('off')

fig.savefig(report / 'qc_grid.png', dpi=140)
(report / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
