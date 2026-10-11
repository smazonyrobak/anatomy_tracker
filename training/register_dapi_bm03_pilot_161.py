"""Low-resolution BM03-to-DAPI-average rigid/affine pilot, not a CCF mapping."""

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
import SimpleITK as sitk


root = Path('I:/AnatomyTracker')
source = root / 'data/g_node_dapi_cc0_individual_volumes_20261011'
out = root / 'reports/dapi_bm03_registration_pilot_161'
moving_path, fixed_path = source / 'BM03.nii', source / 'dapi_template.nii.gz'
out.mkdir(parents=True, exist_ok=True)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

hashes = {}
for path in (moving_path, fixed_path):
    with path.open('rb') as stream:
        hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()

moving_nii, fixed_nii = nib.load(moving_path), nib.load(fixed_path)
moving_stride, fixed_stride = (16, 1, 16), (16, 3, 16)
moving_np = np.asarray(moving_nii.dataobj[::16, ::1, ::16], dtype=np.float32)
fixed_np = np.asarray(fixed_nii.dataobj[::16, ::3, ::16], dtype=np.float32)
moving_spacing = np.asarray(moving_nii.header.get_zooms()[:3]) * moving_stride
fixed_spacing = np.linalg.norm(fixed_nii.affine[:3, :3], axis=0) * fixed_stride
moving = sitk.GetImageFromArray(moving_np.transpose(2, 1, 0))
fixed = sitk.GetImageFromArray(fixed_np.transpose(2, 1, 0))
moving.SetSpacing(moving_spacing.tolist())
fixed.SetSpacing(fixed_spacing.tolist())
fixed.SetOrigin(fixed_nii.affine[:3, 3].tolist())
fixed.SetDirection(tuple((fixed_nii.affine[:3, :3] /
                          np.linalg.norm(fixed_nii.affine[:3, :3], axis=0)).ravel()))
# BM03's identity sform and unknown spatial units are unusable. For this pilot,
# interpret its header pixdim as millimetres in positive NIfTI axes 0/1/2.
moving.SetOrigin((0., 0., 0.))
moving.SetDirection(tuple(np.eye(3).ravel()))

initial = sitk.CenteredTransformInitializer(
    fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
before = sitk.Euler3DTransform(initial)
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsCorrelation()
registration.SetMetricFixedMask(sitk.Cast(fixed > 0.2, sitk.sitkUInt8))
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.4, 20261011)
registration.SetInterpolator(sitk.sitkLinear)
registration.SetShrinkFactorsPerLevel([4, 2, 1])
registration.SetSmoothingSigmasPerLevel([0.4, 0.2, 0.0])
registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.5, minStep=1e-4, numberOfIterations=120,
    relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
registration.SetOptimizerScalesFromPhysicalShift()
registration.SetInitialTransform(initial, inPlace=True)
rigid = registration.Execute(fixed, moving)
rigid_metric = float(registration.GetMetricValue())
rigid_stop = registration.GetOptimizerStopConditionDescription()

fixed_array = sitk.GetArrayFromImage(fixed).transpose(2, 1, 0)
fixed_mask = fixed_array > 0.2
metrics, warped = {}, {}
for name, transform in (('before', before), ('rigid', rigid)):
    warped[name] = sitk.GetArrayFromImage(sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)).transpose(2, 1, 0)
    moving_mask = warped[name] > 0.2
    metrics[name] = {
        'tissue_dice_at_0_2': float(2 * (fixed_mask & moving_mask).sum() /
                                    (fixed_mask.sum() + moving_mask.sum())),
        'pearson_r_within_fixed_tissue': float(np.corrcoef(
            fixed_array[fixed_mask], warped[name][fixed_mask])[0, 1]),
    }

rigid_stable = (np.isfinite(rigid_metric)
                and metrics['rigid']['pearson_r_within_fixed_tissue'] >=
                metrics['before']['pearson_r_within_fixed_tissue'] + 0.005
                and metrics['rigid']['tissue_dice_at_0_2'] >=
                metrics['before']['tissue_dice_at_0_2'] - 0.02)
selected, selected_name = rigid, 'rigid'
affine_metric, affine_stop, affine_singular_values = None, None, None
if rigid_stable:
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.2, minStep=1e-4, numberOfIterations=100,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
    registration.SetInitialTransform(affine, inPlace=True)
    affine = registration.Execute(fixed, moving)
    affine_metric = float(registration.GetMetricValue())
    affine_stop = registration.GetOptimizerStopConditionDescription()
    affine_singular_values = np.linalg.svd(np.asarray(affine.GetMatrix()).reshape(3, 3))[1]
    warped['affine'] = sitk.GetArrayFromImage(sitk.Resample(
        moving, fixed, affine, sitk.sitkLinear, 0., sitk.sitkFloat32)).transpose(2, 1, 0)
    moving_mask = warped['affine'] > 0.2
    metrics['affine'] = {
        'tissue_dice_at_0_2': float(2 * (fixed_mask & moving_mask).sum() /
                                    (fixed_mask.sum() + moving_mask.sum())),
        'pearson_r_within_fixed_tissue': float(np.corrcoef(
            fixed_array[fixed_mask], warped['affine'][fixed_mask])[0, 1]),
    }
    if (np.isfinite(affine_metric) and np.linalg.det(np.asarray(affine.GetMatrix()).reshape(3, 3)) > 0
            and np.all((affine_singular_values > 0.8) & (affine_singular_values < 1.2))
            and metrics['affine']['pearson_r_within_fixed_tissue'] >=
            metrics['rigid']['pearson_r_within_fixed_tissue'] + 0.002
            and metrics['affine']['tissue_dice_at_0_2'] >=
            metrics['rigid']['tissue_dice_at_0_2'] - 0.01):
        selected, selected_name = affine, 'affine'

matrix = np.asarray(selected.GetMatrix()).reshape(3, 3)
center = np.asarray(selected.GetCenter())
offset = center + np.asarray(selected.GetTranslation()) - matrix @ center
fixed_to_bm03 = np.eye(4)
fixed_to_bm03[:3, :3], fixed_to_bm03[:3, 3] = matrix, offset
sitk.WriteTransform(selected, str(out / 'selected_fixed_to_bm03.tfm'))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
fixed_display = np.clip(fixed_array / np.percentile(fixed_array, 99.5), 0, 1)
for row, name in enumerate(('before', selected_name)):
    moving_display = np.clip(warped[name] / np.percentile(moving_np, 99.5), 0, 1)
    overlay = np.stack((fixed_display, moving_display, fixed_display), -1)
    sections = (overlay[overlay.shape[0] // 2].transpose(1, 0, 2),
                overlay[:, overlay.shape[1] // 2].transpose(1, 0, 2),
                overlay[:, :, overlay.shape[2] // 2].transpose(1, 0, 2))
    for axis, section in enumerate(sections):
        axes[row, axis].imshow(section, origin='lower', interpolation='nearest', aspect='auto')
        axes[row, axis].set_title(f'{name}: fixed voxel axis {axis} midpoint')
        axes[row, axis].set_xlabel('fixed low-resolution index')
        axes[row, axis].set_ylabel('fixed low-resolution index')
fig.suptitle('DAPI average magenta, BM03 green; before = geometry-centre initialization, after = selected pilot')
fig.savefig(out / 'before_after_orthogonal_overlays.png', dpi=140)
plt.close(fig)

report = {
    'scope': 'Low-resolution same-modality pilot only; not a CCF mapping or a training input.',
    'input_sha256': hashes,
    'moving': str(moving_path), 'fixed': str(fixed_path),
    'source_shapes_ijk': {'BM03': list(moving_nii.shape), 'DAPI_average': list(fixed_nii.shape)},
    'low_resolution_strides_ijk': {'BM03': list(moving_stride), 'DAPI_average': list(fixed_stride)},
    'low_resolution_shapes_ijk': {'BM03': list(moving_np.shape), 'DAPI_average': list(fixed_np.shape)},
    'low_resolution_spacing_mm_xyz': {'BM03': moving_spacing.tolist(), 'DAPI_average': fixed_spacing.tolist()},
    'downsampling_method': 'Direct NIfTI proxy slicing at the listed integer strides; no prefilter or antialiasing.',
    'physical_convention': 'SimpleITK xyz follows NIfTI array axes 0/1/2 (not asserted anatomical axes). BM03 origin=(0,0,0), positive diagonal spacing from header pixdim interpreted as mm despite unknown units; its identity sform is ignored in memory. DAPI average uses its NIfTI metric sform. No original is rewritten.',
    'before_convention': 'Geometry-centre initialized rigid transform, before intensity optimization; not identity voxel correspondence.',
    'mask_threshold': 0.2,
    'similarity': 'Pearson r within fixed tissue mask; tissue Dice after linear resampling and thresholding, both on low-resolution fixed grid.',
    'metrics': metrics,
    'rigid_optimizer_metric': rigid_metric, 'rigid_optimizer_stop': rigid_stop,
    'rigid_stable_for_affine': bool(rigid_stable),
    'affine_optimizer_metric': affine_metric, 'affine_optimizer_stop': affine_stop,
    'affine_singular_values': None if affine_singular_values is None else affine_singular_values.tolist(),
    'selected': selected_name,
    'selected_fixed_physical_to_bm03_physical_mm': fixed_to_bm03.tolist(),
    'selected_bm03_physical_to_fixed_physical_mm': np.linalg.inv(fixed_to_bm03).tolist(),
    'warning': 'The BM03 unit/origin convention is a pilot assumption, and these low-resolution metrics do not validate anatomical correspondence or CCF accuracy.',
}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
