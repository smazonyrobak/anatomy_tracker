"""Native-grid BM03/template DAPI preview; no registration or CCF assumption."""

import json
import os
from pathlib import Path

os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


root = Path('I:/AnatomyTracker')
source = root / 'data/g_node_dapi_cc0_individual_volumes_20261011'
out = root / 'reports/dapi_bm03_preflight_161'
out.mkdir(parents=True, exist_ok=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
report = {
    'scope': 'Native voxel-grid header and low-resolution intensity preflight only; no registration, resampling to a common space, training, or CCF alignment assumed.',
    'inputs': {},
}

for row, name in enumerate(('BM03.nii', 'dapi_template.nii.gz')):
    path = source / name
    image = nib.load(path)
    header = image.header
    stride = tuple(max(1, int(np.ceil(n / 128))) for n in image.shape)
    sample = np.asarray(image.dataobj[tuple(slice(None, None, s) for s in stride)], dtype=np.float32)
    finite = sample[np.isfinite(sample)]
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    report['inputs'][name] = {
        'path': str(path),
        'file_bytes': path.stat().st_size,
        'shape': list(image.shape),
        'storage_dtype': str(header.get_data_dtype()),
        'header_voxel_size': [float(x) for x in header.get_zooms()[:3]],
        'spatial_unit': header.get_xyzt_units()[0],
        'qform_code': int(qcode),
        'qform': None if qform is None else qform.tolist(),
        'sform_code': int(scode),
        'sform': None if sform is None else sform.tolist(),
        'selected_affine': image.affine.tolist(),
        'selected_affine_axis_step': np.linalg.norm(image.affine[:3, :3], axis=0).tolist(),
        'sample_stride': list(stride),
        'sample_shape': list(sample.shape),
        'sample_finite_fraction': float(np.isfinite(sample).mean()),
        'sample_nonzero_fraction': float((finite != 0).mean()),
        'sample_percentiles_0_1_25_50_75_99_100': np.percentile(
            finite, (0, 1, 25, 50, 75, 99, 100)).tolist(),
    }
    limit = float(np.percentile(finite, 99.5))
    sections = (sample[sample.shape[0] // 2].T,
                sample[:, sample.shape[1] // 2].T,
                sample[:, :, sample.shape[2] // 2].T)
    for axis, section in enumerate(sections):
        axes[row, axis].imshow(section, cmap='gray', origin='lower', vmin=0, vmax=limit,
                               interpolation='nearest', aspect='auto')
        axes[row, axis].set_title(f'{name}: voxel axis {axis} midpoint')
        axes[row, axis].set_xlabel('sampled voxel index (not aligned)')
        axes[row, axis].set_ylabel('sampled voxel index')

fig.suptitle('BM03 reconstructed DAPI and final DAPI template — separate native grids, not registered')
fig.savefig(out / 'native_orthogonal_montage.png', dpi=140)
plt.close(fig)
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
