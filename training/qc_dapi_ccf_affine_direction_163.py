"""CPU-only coordinate-convention QC for the published DAPI-average/Allen affine."""

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
dapi_path = root / 'data/g_node_dapi_cc0_individual_volumes_20261011/dapi_template.nii.gz'
ccf_path = root / 'data/Allen Brain Atlas 25um/average_template_25.nrrd'
transform_path = root / 'data/g_node_dapi_cc0_individual_volumes_20261011/ccf_to_dapi_affine_transformation.txt'
out = root / 'reports/dapi_ccf_affine_direction_qc_163'
out.mkdir(parents=True, exist_ok=False)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

hashes = {}
for path in (dapi_path, ccf_path, transform_path):
    with path.open('rb') as stream:
        hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()

dapi_nifti = nib.load(dapi_path)
dapi_full = sitk.ReadImage(str(dapi_path), sitk.sitkFloat32)
ccf_full = sitk.ReadImage(str(ccf_path), sitk.sitkFloat32)
pull = sitk.ReadTransform(str(transform_path))
matrix = np.asarray(pull.GetMatrix()).reshape(3, 3)
translation = np.asarray(pull.GetTranslation())
ras_flip = np.diag([-1., -1., 1.])
ras_fed = sitk.AffineTransform(3)
ras_fed.SetMatrix((matrix @ ras_flip).ravel().tolist())
ras_fed.SetTranslation(translation.tolist())
inverse_fed = pull.GetInverse()

# Bin averages avoid picking arbitrary high-resolution voxels in the display/QC grid.
dapi = sitk.BinShrink(dapi_full, [12, 3, 9])
ccf = sitk.BinShrink(ccf_full, [4, 4, 4])
del dapi_full, ccf_full
dapi_array = sitk.GetArrayViewFromImage(dapi)
dapi_p99 = float(np.percentile(dapi_array[dapi_array > 0], 99.5))
ccf_array = sitk.GetArrayViewFromImage(ccf)
ccf_p99 = float(np.percentile(ccf_array[ccf_array > 0], 99.5))
dapi_mask = dapi_array > 0.1 * dapi_p99
ccf_ones = sitk.Image(ccf.GetSize(), sitk.sitkUInt8)
ccf_ones.CopyInformation(ccf)
ccf_ones += 1

warped = {}
metrics = {}
for name, transform in (('published_LPS_pull', pull),
                        ('RAS_fed_to_published_matrix', ras_fed),
                        ('inverse_fed_as_pull', inverse_fed)):
    warped[name] = sitk.GetArrayFromImage(sitk.Resample(
        ccf, dapi, transform, sitk.sitkLinear, 0., sitk.sitkFloat32))
    inside = sitk.GetArrayFromImage(sitk.Resample(
        ccf_ones, dapi, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
    ccf_mask = warped[name] > 0.1 * ccf_p99
    metrics[name] = {
        'DAPI_mask_voxels': int(dapi_mask.sum()),
        'sampled_CCF_grid_fraction_of_DAPI_mask': float(inside[dapi_mask].mean()),
        'CCF_signal_fraction_of_DAPI_mask': float(ccf_mask[dapi_mask].mean()),
        'coarse_threshold_mask_dice': float(2 * np.logical_and(dapi_mask, ccf_mask).sum() /
                                            (dapi_mask.sum() + ccf_mask.sum())),
    }

fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
dapi_display = np.clip(dapi_array / dapi_p99, 0, 1)
for row, name in enumerate(warped):
    ccf_display = np.clip(warped[name] / ccf_p99, 0, 1)
    overlay = np.stack((dapi_display, ccf_display, dapi_display), axis=-1)
    sections = (overlay[overlay.shape[0] // 2],
                overlay[:, overlay.shape[1] // 2],
                overlay[:, :, overlay.shape[2] // 2])
    for col, section in enumerate(sections):
        axes[row, col].imshow(section, origin='lower', interpolation='nearest', aspect='auto')
        axes[row, col].set_title(f'{name}: fixed index axis {2-col} midpoint')
        axes[row, col].set_xlabel('DAPI low-resolution index')
        axes[row, col].set_ylabel('DAPI low-resolution index')
fig.suptitle('DAPI average magenta; Allen average template green. Index-plane QC, not anatomical validation.')
fig.savefig(out / 'three_plane_direction_comparison.png', dpi=150)
plt.close(fig)

reader = sitk.ImageFileReader()
reader.SetFileName(str(dapi_path))
reader.ReadImageInformation()
dapi_size, dapi_spacing = reader.GetSize(), reader.GetSpacing()
dapi_origin, dapi_direction = reader.GetOrigin(), reader.GetDirection()
reader.SetFileName(str(ccf_path))
reader.ReadImageInformation()
ccf_size, ccf_spacing = reader.GetSize(), reader.GetSpacing()
ccf_origin, ccf_direction = reader.GetOrigin(), reader.GetDirection()
dapi_center_index = (np.asarray(dapi_size) - 1) / 2
ccf_center_index = (np.asarray(ccf_size) - 1) / 2
dapi_points = {'center': dapi_center_index.copy()}
for axis in range(3):
    for edge, index in (('min', 0.), ('max', float(dapi_size[axis] - 1))):
        point = dapi_center_index.copy()
        point[axis] = index
        dapi_points[f'axis_{axis}_{edge}_face_center'] = point

point_checks = {}
for name, ijk in dapi_points.items():
    dapi_lps_mm = np.asarray(dapi_origin) + np.asarray(dapi_direction).reshape(3, 3) @ (
        np.asarray(dapi_spacing) * ijk)
    ccf_lps_um = np.asarray(pull.TransformPoint(dapi_lps_mm.tolist()))
    ccf_index = (ccf_lps_um - ccf_origin) / ccf_spacing
    wrong_ras_index = (np.asarray(ras_fed.TransformPoint(dapi_lps_mm.tolist())) - ccf_origin) / ccf_spacing
    point_checks[name] = {
        'DAPI_original_index_xyz': ijk.tolist(),
        'DAPI_LPS_mm': dapi_lps_mm.tolist(),
        'DAPI_RAS_mm': (ras_flip @ dapi_lps_mm).tolist(),
        'published_CCF_LPS_um': ccf_lps_um.tolist(),
        'published_CCF_index_xyz': ccf_index.tolist(),
        'published_inside_CCF_grid': bool(np.all((ccf_index >= -0.5) &
                                                   (ccf_index <= np.asarray(ccf_size) - 0.5))),
        'RAS_fed_CCF_index_xyz': wrong_ras_index.tolist(),
        'inverse_roundtrip_error_mm': float(np.linalg.norm(
            np.asarray(inverse_fed.TransformPoint(ccf_lps_um.tolist())) - dapi_lps_mm)),
    }

ccf_center_lps_um = np.asarray(ccf_origin) + np.asarray(ccf_direction).reshape(3, 3) @ (
    np.asarray(ccf_spacing) * ccf_center_index)
dapi_from_ccf_center = np.asarray(inverse_fed.TransformPoint(ccf_center_lps_um.tolist()))
dapi_from_ccf_center_index = np.linalg.solve(np.asarray(dapi_direction).reshape(3, 3),
                                             dapi_from_ccf_center - dapi_origin) / dapi_spacing

report = {
    'scope': 'Published DAPI-average/Allen average intensity-coordinate QC only; no BM03/BM17/BM18 images, atlas labels, specimen-to-CCF mapping, or training.',
    'official_G_Node_README': 'https://gin.g-node.org/doi/B6_dapi_template/raw/10.12751/g-node.16wrxa/README.md',
    'official_README_command': 'antsApplyTransforms -i average_template_25.nrrd -d 3 -r dapi_template.nii.gz -o average_template_25_registered_to_dapi.nii.gz -t ccf_to_dapi_affine_transformation.txt',
    'SimpleITK_transform_convention': 'https://simpleitk.readthedocs.io/en/release/fundamentalConcepts.html',
    'interpretation': 'For DAPI output-grid points in SimpleITK LPS millimetres, the published ITK affine is a pull map into Allen input-grid LPS micrometre coordinates. The first two signs in the NIfTI RAS affine reverse on SimpleITK read. Matrix magnitude near 1000 includes mm-to-um scale; do not add another conversion.',
    'input_sha256': hashes,
    'DAPI_NIfTI_RAS_affine_mm': dapi_nifti.affine.tolist(),
    'DAPI_NIfTI_units': dapi_nifti.header.get_xyzt_units(),
    'DAPI_SimpleITK_LPS_header': {'size_xyz': dapi_size, 'spacing_mm_xyz': dapi_spacing,
                                  'origin_mm_xyz': dapi_origin, 'direction': dapi_direction},
    'Allen_SimpleITK_LPS_header': {'size_xyz': ccf_size, 'spacing_um_xyz': ccf_spacing,
                                   'origin_um_xyz': ccf_origin, 'direction': ccf_direction},
    'published_affine_matrix_DAPI_LPS_mm_to_CCF_LPS_um': matrix.tolist(),
    'published_affine_translation_CCF_um': translation.tolist(),
    'published_affine_fixed_center': pull.GetFixedParameters(),
    'downsampling': 'SimpleITK BinShrink block means; factors DAPI xyz=(12,3,9), Allen xyz=(4,4,4); resampling occurs on the shrunken DAPI physical grid.',
    'low_resolution': {'DAPI_size_xyz': dapi.GetSize(), 'DAPI_spacing_mm_xyz': dapi.GetSpacing(),
                       'Allen_size_xyz': ccf.GetSize(), 'Allen_spacing_um_xyz': ccf.GetSpacing()},
    'comparison': metrics,
    'point_checks': point_checks,
    'Allen_grid_center_check': {'CCF_center_index_xyz': ccf_center_index.tolist(),
                                'CCF_center_LPS_um': ccf_center_lps_um.tolist(),
                                'inverse_mapped_DAPI_LPS_mm': dapi_from_ccf_center.tolist(),
                                'inverse_mapped_DAPI_index_xyz': dapi_from_ccf_center_index.tolist()},
    'limitations': 'The README, headers, pull semantics, bounds, and overlays verify coordinate and resampling conventions only. Coarse intensity/mask agreement is not independent anatomical or landmark accuracy; the affine is not specimen-specific and does not justify native BM03-to-CCF labels.',
}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'comparison': metrics,
                  'DAPI_center_CCF_index': point_checks['center']['published_CCF_index_xyz'],
                  'CCF_center_DAPI_index': dapi_from_ccf_center_index.tolist()}, indent=2))
