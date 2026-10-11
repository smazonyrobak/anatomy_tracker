"""Antialiased BM03-to-DAPI-average refinement; no atlas-label inference."""

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
previous = root / 'reports/dapi_bm03_registration_pilot_161/report.json'
out = root / 'reports/dapi_bm03_antialiased_pilot_162'
moving_path, fixed_path = source / 'BM03.nii', source / 'dapi_template.nii.gz'
out.mkdir(parents=True, exist_ok=False)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

hashes = {}
for path in (moving_path, fixed_path):
    with path.open('rb') as stream:
        hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
prior_report = json.loads(previous.read_text())
assert hashes == prior_report['input_sha256']
prior_matrix = np.asarray(prior_report['selected_fixed_physical_to_bm03_physical_mm'])

low = {}
settings = ((moving_path, (8, 1, 8), (3.5, 0., 3.5)),
            (fixed_path, (8, 3, 8), (3.5, 1.2, 3.5)))
for path, stride, sigma in settings:
    image = nib.load(path)
    full = np.asarray(image.dataobj, dtype=np.float32)
    blurred = ndi.gaussian_filter(full, sigma=sigma, mode='nearest', truncate=3.)
    low[path.name] = blurred[::stride[0], ::stride[1], ::stride[2]].copy()
    del full, blurred
    gc.collect()

moving_nii, fixed_nii = nib.load(moving_path), nib.load(fixed_path)
moving_np, fixed_np = low[moving_path.name], low[fixed_path.name]
moving_stride, fixed_stride = settings[0][1], settings[1][1]
moving_spacing = np.asarray(moving_nii.header.get_zooms()[:3]) * moving_stride
fixed_spacing = np.linalg.norm(fixed_nii.affine[:3, :3], axis=0) * fixed_stride
moving = sitk.GetImageFromArray(moving_np.transpose(2, 1, 0))
fixed = sitk.GetImageFromArray(fixed_np.transpose(2, 1, 0))
moving.SetSpacing(moving_spacing.tolist())
fixed.SetSpacing(fixed_spacing.tolist())
moving.SetOrigin((0., 0., 0.))
fixed.SetOrigin(fixed_nii.affine[:3, 3].tolist())
moving.SetDirection(tuple(np.eye(3).ravel()))
fixed.SetDirection(tuple((fixed_nii.affine[:3, :3] /
                          np.linalg.norm(fixed_nii.affine[:3, :3], axis=0)).ravel()))

prior = sitk.AffineTransform(3)
prior.SetMatrix(prior_matrix[:3, :3].ravel().tolist())
prior.SetTranslation(prior_matrix[:3, 3].tolist())
candidate = sitk.AffineTransform(prior)
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsCorrelation()
registration.SetMetricFixedMask(sitk.Cast(fixed > 0.2, sitk.sitkUInt8))
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.08, 20261012)
registration.SetInterpolator(sitk.sitkLinear)
registration.SetShrinkFactorsPerLevel([2, 1])
registration.SetSmoothingSigmasPerLevel([0.15, 0.])
registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.1, minStep=5e-5, numberOfIterations=100,
    relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
registration.SetOptimizerScalesFromPhysicalShift()
registration.SetInitialTransform(candidate, inPlace=True)
fitted = registration.Execute(fixed, moving)
optimizer_metric = float(registration.GetMetricValue())
optimizer_stop = registration.GetOptimizerStopConditionDescription()

fixed_array = sitk.GetArrayFromImage(fixed).transpose(2, 1, 0)
fixed_mask = fixed_array > 0.2
interior = ndi.binary_erosion(fixed_mask, iterations=3)
metrics, warped = {}, {}
for name, transform in (('prior', prior), ('refined', fitted)):
    warped[name] = sitk.GetArrayFromImage(sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)).transpose(2, 1, 0)
    moving_mask = warped[name] > 0.2
    metrics[name] = {
        'tissue_dice_at_0_2': float(2 * (fixed_mask & moving_mask).sum() /
                                    (fixed_mask.sum() + moving_mask.sum())),
        'pearson_r_fixed_tissue': float(np.corrcoef(
            fixed_array[fixed_mask], warped[name][fixed_mask])[0, 1]),
        'pearson_r_eroded_fixed_tissue': float(np.corrcoef(
            fixed_array[interior], warped[name][interior])[0, 1]),
    }

singular_values = np.linalg.svd(np.asarray(fitted.GetMatrix()).reshape(3, 3))[1]
accept = (np.isfinite(optimizer_metric)
          and np.linalg.det(np.asarray(fitted.GetMatrix()).reshape(3, 3)) > 0
          and np.all((singular_values > 0.8) & (singular_values < 1.2))
          and metrics['refined']['pearson_r_eroded_fixed_tissue'] >=
          metrics['prior']['pearson_r_eroded_fixed_tissue'] + 0.005
          and metrics['refined']['tissue_dice_at_0_2'] >=
          metrics['prior']['tissue_dice_at_0_2'] - 0.005)
selected_name, selected = ('refined', fitted) if accept else ('prior', prior)
matrix = np.asarray(selected.GetMatrix()).reshape(3, 3)
center = np.asarray(selected.GetCenter())
offset = center + np.asarray(selected.GetTranslation()) - matrix @ center
fixed_to_bm03 = np.eye(4)
fixed_to_bm03[:3, :3], fixed_to_bm03[:3, 3] = matrix, offset
sitk.WriteTransform(selected, str(out / 'selected_fixed_to_bm03.tfm'))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
fixed_display = np.clip(fixed_array / np.percentile(fixed_array, 99.5), 0, 1)
for row, name in enumerate(('prior', selected_name)):
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
fig.suptitle('Antialiased DAPI average magenta, BM03 green; prior vs higher-resolution refinement')
fig.savefig(out / 'orthogonal_overlays.png', dpi=140)
plt.close(fig)

normals = ([0.55, 0.65, 0.52], [-0.40, 0.80, 0.45])
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
origin = np.asarray(fixed.GetOrigin())
spacing = np.asarray(fixed.GetSpacing())
centre = origin + (np.asarray(fixed_array.shape) - 1) * spacing / 2
axis = (np.arange(144) - 71.5) * 0.06
yy, xx = np.meshgrid(axis, axis, indexing='ij')
for row, direction in enumerate(normals):
    normal = np.asarray(direction) / np.linalg.norm(direction)
    u = np.array([1., 0., 0.]) - normal[0] * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    physical = centre + xx[..., None] * u + yy[..., None] * v
    ijk = ((physical - origin) / spacing).transpose(2, 0, 1)
    average = ndi.map_coordinates(fixed_array, ijk, order=1, mode='constant', cval=0.)
    specimen = ndi.map_coordinates(warped[selected_name], ijk, order=1, mode='constant', cval=0.)
    a = np.clip(average / np.percentile(fixed_array, 99.5), 0, 1)
    b = np.clip(specimen / np.percentile(moving_np, 99.5), 0, 1)
    for col, image in enumerate((a, b, np.stack((a, b, a), -1))):
        axes[row, col].imshow(image, cmap='gray' if col < 2 else None,
                              origin='lower', interpolation='nearest')
        axes[row, col].set_title(('DAPI average', 'BM03 reslice', 'overlay')[col])
        axes[row, col].axis('off')
fig.suptitle('Two synthetic oblique reslices of reconstructed volumes; not acquired oblique sections')
fig.savefig(out / 'oblique_reslice_qc.png', dpi=140)
plt.close(fig)

report = {
    'scope': 'TRAIN-specimen same-modality registration/QC only; no CCF labels, BM17/BM18 images, training, or anatomical validation.',
    'dataset_doi': 'https://doi.org/10.12751/g-node.16wrxa',
    'paper': 'https://doi.org/10.1038/s41597-020-0570-z',
    'source_coordinate_evidence': 'The source paper describes 100-um DAPI-stained coronal sections, 8.6-um in-plane processed pixels, and final-average interpolation to 33.3 um along the anterior-posterior stack axis. BM03 is an individual reconstructed volume; the final template is an iterative population average. Positive axis directions and BM03 world origin are not established by its identity sform.',
    'input_sha256': hashes,
    'prior_report': str(previous),
    'source_shapes_ijk': {'BM03': list(moving_nii.shape), 'DAPI_average': list(fixed_nii.shape)},
    'antialias': 'Full-volume Gaussian filter before integer subsampling; scipy.ndimage.gaussian_filter float32, mode nearest, truncate 3 sigma.',
    'sigma_original_voxels_ijk': {'BM03': list(settings[0][2]), 'DAPI_average': list(settings[1][2])},
    'strides_ijk': {'BM03': list(moving_stride), 'DAPI_average': list(fixed_stride)},
    'low_shapes_ijk': {'BM03': list(moving_np.shape), 'DAPI_average': list(fixed_np.shape)},
    'spacing_mm_xyz': {'BM03': moving_spacing.tolist(), 'DAPI_average': fixed_spacing.tolist()},
    'physical_convention': 'SimpleITK xyz = NIfTI axes 0/1/2; axis 1 is the coronal stack/AP axis, but positive directions are unverified. BM03 header pixdim is interpreted as millimetres with origin 0 and identity direction in memory; its identity sform is ignored. DAPI average uses its metric sform. Transform maps fixed physical xyz to BM03 physical xyz for resampling.',
    'metric': 'Correlation optimized on a random 8% fixed-brain sample; Pearson r and threshold-0.2 tissue Dice evaluated on the same antialiased fixed grid, not independent anatomical truth.',
    'metrics': metrics, 'optimizer_metric': optimizer_metric,
    'optimizer_stop': optimizer_stop, 'affine_singular_values': singular_values.tolist(),
    'refinement_accepted': bool(accept), 'selected': selected_name,
    'selected_fixed_physical_to_bm03_physical_mm': fixed_to_bm03.tolist(),
    'selected_bm03_physical_to_fixed_physical_mm': np.linalg.inv(fixed_to_bm03).tolist(),
    'oblique_qc': {'unit_normals_in_fixed_xyz': normals, 'pixel_size_mm': 0.06,
                   'size': 144, 'kind': 'synthetic reslices only'},
    'limitations': 'BM03 reconstructed from 100-um coronal sections, so oblique reslices interpolate between physical sections. A smooth affine and high Dice can hide local specimen/template differences; no held-out anatomical landmarks, physical oblique sections, or CCF truth were used.',
}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
