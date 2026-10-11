"""TRAIN-only multi-AP DAPI specimen/average anatomy review; no pose labels."""

import gc
import hashlib
import json
import os
from pathlib import Path

os.environ['MPLCONFIGDIR'] = 'I:/AnatomyTracker/cache/matplotlib'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk


root = Path('I:/AnatomyTracker')
source = root / 'data/g_node_dapi_cc0_individual_volumes_20261011'
out = root / 'reports/dapi_train_landmark_grid_229_001'
protocol = Path(__file__).resolve().parents[1] / 'docs/publication/DAPI_TRAIN_LANDMARK_GRID_229_PROTOCOL_20261011.md'
fixed_path = source / 'dapi_template.nii.gz'
transform_paths = {
    'BM03': root / 'reports/dapi_bm03_antialiased_pilot_162/selected_fixed_to_bm03.tfm',
    **{animal: root / f'reports/dapi_train_multispecimen_affine_qc_183_001/{animal}/selected_fixed_to_specimen.tfm'
       for animal in ('BM04', 'BM06', 'BM10')},
}
out.mkdir(parents=True, exist_ok=False)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

fixed_nii = nib.load(fixed_path)
fixed_full = np.asarray(fixed_nii.dataobj, dtype=np.float32)
fixed_np = ndi.gaussian_filter(fixed_full, (1.8, 1.2, 1.8), mode='nearest', truncate=3.)[::4, ::3, ::4].copy()
del fixed_full
gc.collect()
fixed_spacing = np.linalg.norm(fixed_nii.affine[:3, :3], axis=0) * (4, 3, 4)
fixed = sitk.GetImageFromArray(fixed_np.transpose(2, 1, 0))
fixed.SetSpacing(fixed_spacing.tolist())
fixed.SetOrigin(fixed_nii.affine[:3, 3].tolist())
fixed.SetDirection(tuple((fixed_nii.affine[:3, :3] /
                          np.linalg.norm(fixed_nii.affine[:3, :3], axis=0)).ravel()))
ap_area = (fixed_np > 0.2).sum(axis=(0, 2))
ap_any = np.flatnonzero(ap_area > .05 * ap_area.max())
positions = [int(round(ap_any[0] + fraction * (ap_any[-1] - ap_any[0])))
             for fraction in (.2, .4, .6, .8)]
levels_mm = [float(fixed_nii.affine[1, 3] + index * 3 * fixed_nii.header.get_zooms()[1])
             for index in positions]
fixed_scale = np.percentile(fixed_np[fixed_np > 0], 99.5)
fixed_display = np.clip(fixed_np / fixed_scale, 0, 1)

fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
plain_fig, plain_axes = plt.subplots(5, 4, figsize=(16, 15), constrained_layout=True)
for col, ap_index in enumerate(positions):
    plain_axes[0, col].imshow(fixed_display[:, ap_index, :].T, origin='lower',
                              cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plain_axes[0, col].set_title(f'average, AP index {ap_index}')
    plain_axes[0, col].axis('off')
reports = {}
for row, (animal, transform_path) in enumerate(transform_paths.items()):
    moving_path = source / f'{animal}.nii'
    moving_nii = nib.load(moving_path)
    moving_full = np.asarray(moving_nii.dataobj, dtype=np.float32)
    moving_np = ndi.gaussian_filter(moving_full, (1.8, 0., 1.8), mode='nearest', truncate=3.)[::4, :, ::4].copy()
    del moving_full
    gc.collect()
    moving_spacing = np.asarray(moving_nii.header.get_zooms()[:3]) * (4, 1, 4)
    moving = sitk.GetImageFromArray(moving_np.transpose(2, 1, 0))
    moving.SetSpacing(moving_spacing.tolist())
    moving.SetOrigin((0., 0., 0.))
    moving.SetDirection(tuple(np.eye(3).ravel()))
    transform = sitk.ReadTransform(str(transform_path))
    warped = sitk.GetArrayFromImage(sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)).transpose(2, 1, 0)
    moving_scale = np.percentile(moving_np[moving_np > 0], 99.5)
    moving_display = np.clip(warped / moving_scale, 0, 1)
    for col, ap_index in enumerate(positions):
        overlay = np.stack((fixed_display[:, ap_index, :], moving_display[:, ap_index, :],
                            fixed_display[:, ap_index, :]), -1).transpose(1, 0, 2)
        axes[row, col].imshow(overlay, origin='lower', interpolation='nearest')
        axes[row, col].set_title(f'{animal}, average AP index {ap_index}, {levels_mm[col]:+.2f} mm')
        axes[row, col].axis('off')
        plain_axes[row + 1, col].imshow(moving_display[:, ap_index, :].T,
                                        origin='lower', cmap='gray', vmin=0, vmax=1,
                                        interpolation='nearest')
        plain_axes[row + 1, col].set_title(f'{animal}, same AP index {ap_index}')
        plain_axes[row + 1, col].axis('off')
    with moving_path.open('rb') as stream:
        moving_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with transform_path.open('rb') as stream:
        transform_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    reports[animal] = {
        'specimen_sha256': moving_hash,
        'transform_sha256': transform_hash,
        'native_shape_ijk': list(moving_nii.shape),
        'native_header_units': moving_nii.header.get_xyzt_units()[0],
        'native_spacing_mm_ijk_assumed': np.asarray(moving_nii.header.get_zooms()[:3]).tolist(),
        'moving_review_shape_ijk': list(moving_np.shape),
    }
    print(animal, 'resampled', flush=True)
    del moving, moving_nii, moving_np, warped, moving_display
    gc.collect()

fig.suptitle('DAPI population average magenta / individual TRAIN specimen green; same four AP positions')
image_path = out / 'train_anatomy_grid.png'
fig.savefig(image_path, dpi=150)
plt.close(fig)
plain_fig.suptitle('Separate grayscale DAPI average and TRAIN specimens; same four AP positions')
plain_path = out / 'train_anatomy_grayscale.png'
plain_fig.savefig(plain_path, dpi=150)
plt.close(plain_fig)
with fixed_path.open('rb') as stream:
    fixed_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'complete.json').write_text(json.dumps({
    'scope': 'TRAIN-only anatomical QC, not CCF pose-label approval',
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'average_sha256': fixed_hash,
    'average_review_shape_ijk': list(fixed_np.shape),
    'review_stride_ijk': [4, 3, 4],
    'ap_extent_rule': 'average >0.2 area exceeds 5% of maximum AP area',
    'ap_positions_average_review_index': positions,
    'ap_positions_average_world_mm': levels_mm,
    'specimens': reports,
    'grid_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
    'grayscale_sha256': hashlib.sha256(plain_path.read_bytes()).hexdigest(),
    'label_eligibility': 'none without independent anatomical landmark/axis review',
}, indent=2) + '\n', encoding='utf-8')
