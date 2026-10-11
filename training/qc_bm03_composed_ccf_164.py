"""CPU-only BM03/Allen composed-transform anatomy QC; no training-label export."""

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
atlas = root / 'data/Allen Brain Atlas 25um'
bm03_path = source / 'BM03.nii'
dapi_path = source / 'dapi_template.nii.gz'
ccf_path = atlas / 'average_template_25.nrrd'
annotation_path = atlas / 'annotation_25.nrrd'
specimen_transform_path = root / 'reports/dapi_bm03_antialiased_pilot_162/selected_fixed_to_bm03.tfm'
specimen_report_path = root / 'reports/dapi_bm03_antialiased_pilot_162/report.json'
published_transform_path = source / 'ccf_to_dapi_affine_transformation.txt'
convention_report_path = root / 'reports/dapi_ccf_affine_direction_qc_163/report.json'
out = root / 'reports/bm03_ccf_composed_qc_164'
out.mkdir(parents=True, exist_ok=False)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

paths = (bm03_path, dapi_path, ccf_path, annotation_path, specimen_transform_path,
         specimen_report_path, published_transform_path, convention_report_path)
hashes = {}
for path in paths:
    with path.open('rb') as stream:
        hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
specimen_report = json.loads(specimen_report_path.read_text())
convention_report = json.loads(convention_report_path.read_text())
assert hashes[str(bm03_path)] == specimen_report['input_sha256']['BM03.nii']
assert hashes[str(dapi_path)] == specimen_report['input_sha256']['dapi_template.nii.gz']
assert hashes[str(ccf_path)] == convention_report['input_sha256'][str(ccf_path)]
assert hashes[str(published_transform_path)] == convention_report['input_sha256'][str(published_transform_path)]
assert specimen_report['selected'] == 'refined'

bm03_nifti = nib.load(bm03_path)
dapi_nifti = nib.load(dapi_path)
assert np.allclose(bm03_nifti.affine, np.eye(4))
assert np.allclose(dapi_nifti.affine[:3, :3], np.diag(dapi_nifti.header.get_zooms()[:3]))
assert np.allclose(np.asarray(bm03_nifti.header.get_zooms()[:3]), [0.0086, 0.1, 0.0086])
fixed_to_bm03 = sitk.ReadTransform(str(specimen_transform_path))
dapi_lps_to_ccf = sitk.ReadTransform(str(published_transform_path))
assert np.allclose(fixed_to_bm03.GetFixedParameters(), 0)
assert np.allclose(dapi_lps_to_ccf.GetFixedParameters(), 0)
specimen_matrix = np.asarray(fixed_to_bm03.GetMatrix()).reshape(3, 3)
specimen_translation = np.asarray(fixed_to_bm03.GetTranslation())
saved_matrix = np.asarray(specimen_report['selected_fixed_physical_to_bm03_physical_mm'])
assert np.allclose(specimen_matrix, saved_matrix[:3, :3])
assert np.allclose(specimen_translation, saved_matrix[:3, 3])
ccf_matrix = np.asarray(dapi_lps_to_ccf.GetMatrix()).reshape(3, 3)
ccf_translation = np.asarray(dapi_lps_to_ccf.GetTranslation())
ras_to_lps = np.diag([-1., -1., 1.])
bm03_to_dapi_ras_matrix = np.linalg.inv(specimen_matrix)
bm03_to_dapi_ras_translation = -bm03_to_dapi_ras_matrix @ specimen_translation
composed_matrix = ccf_matrix @ ras_to_lps @ bm03_to_dapi_ras_matrix
composed_translation = ccf_matrix @ ras_to_lps @ bm03_to_dapi_ras_translation + ccf_translation
bm03_to_ccf = sitk.AffineTransform(3)
bm03_to_ccf.SetMatrix(composed_matrix.ravel().tolist())
bm03_to_ccf.SetTranslation(composed_translation.tolist())

bm03_full = sitk.ReadImage(str(bm03_path), sitk.sitkFloat32)
bm03_full.SetSpacing(tuple(float(x) for x in bm03_nifti.header.get_zooms()[:3]))
bm03_full.SetOrigin((0., 0., 0.))
bm03_full.SetDirection(tuple(np.eye(3).ravel()))
bm03 = sitk.BinShrink(bm03_full, [8, 1, 8])
del bm03_full
ccf = sitk.ReadImage(str(ccf_path), sitk.sitkFloat32)
annotation = sitk.ReadImage(str(annotation_path), sitk.sitkUInt32)
assert ccf.GetSize() == annotation.GetSize()
assert np.allclose(ccf.GetSpacing(), annotation.GetSpacing())
assert np.allclose(ccf.GetOrigin(), annotation.GetOrigin())
assert np.allclose(ccf.GetDirection(), annotation.GetDirection())
ccf_size, ccf_spacing, ccf_origin, ccf_direction = (
    ccf.GetSize(), ccf.GetSpacing(), ccf.GetOrigin(), ccf.GetDirection())
ccf_on_bm03 = sitk.Resample(ccf, bm03, bm03_to_ccf, sitk.sitkLinear, 0., sitk.sitkFloat32)
annotation_on_bm03 = sitk.Resample(annotation, bm03, bm03_to_ccf,
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)
del ccf, annotation
bm03_array = sitk.GetArrayFromImage(bm03)
ccf_array = sitk.GetArrayFromImage(ccf_on_bm03)
annotation_array = sitk.GetArrayFromImage(annotation_on_bm03)
bm03_p99 = float(np.percentile(bm03_array[bm03_array > 0], 99.5))
ccf_p99 = float(np.percentile(ccf_array[ccf_array > 0], 99.5))
bm03_mask = bm03_array > 0.2
ccf_mask = ccf_array > 0.1 * ccf_p99
annotation_mask = annotation_array > 0
spacing_zyx = np.asarray(bm03.GetSpacing())[::-1]
bm03_surface = bm03_mask & ~ndi.binary_erosion(bm03_mask)
annotation_surface = annotation_mask & ~ndi.binary_erosion(annotation_mask)
bm03_to_annotation_surface_mm = ndi.distance_transform_edt(~annotation_surface, sampling=spacing_zyx)
annotation_to_bm03_surface_mm = ndi.distance_transform_edt(~bm03_surface, sampling=spacing_zyx)

metrics = {
    'BM03_threshold': 0.2,
    'Allen_average_threshold': float(0.1 * ccf_p99),
    'BM03_tissue_voxels': int(bm03_mask.sum()),
    'Allen_average_tissue_voxels': int(ccf_mask.sum()),
    'Allen_annotation_voxels': int(annotation_mask.sum()),
    'BM03_vs_Allen_average_tissue_dice': float(2 * (bm03_mask & ccf_mask).sum() /
                                                (bm03_mask.sum() + ccf_mask.sum())),
    'BM03_vs_annotated_union_dice': float(2 * (bm03_mask & annotation_mask).sum() /
                                            (bm03_mask.sum() + annotation_mask.sum())),
    'annotated_fraction_of_BM03_tissue': float(annotation_mask[bm03_mask].mean()),
    'BM03_tissue_fraction_of_annotation': float(bm03_mask[annotation_mask].mean()),
    'BM03_surface_to_annotation_surface_mm_p50_p95': np.percentile(
        bm03_to_annotation_surface_mm[bm03_surface], [50, 95]).tolist(),
    'annotation_surface_to_BM03_surface_mm_p50_p95': np.percentile(
        annotation_to_bm03_surface_mm[annotation_surface], [50, 95]).tolist(),
}

bm03_display = np.clip(bm03_array / bm03_p99, 0, 1)
ccf_display = np.clip(ccf_array / ccf_p99, 0, 1)
intensity_overlay = np.stack((bm03_display, ccf_display, bm03_display), axis=-1)
planes = [('coronal (native axis 1)', 1), ('sagittal (native axis 0)', 2),
          ('horizontal (native axis 2)', 0)]
plane_metrics = {}
for kind in ('intensity', 'annotation'):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for row, (orientation, axis) in enumerate(planes):
        occupied = np.flatnonzero(np.any(bm03_mask, axis=tuple(a for a in range(3) if a != axis)))
        indices = np.rint(np.quantile(occupied, [0.25, 0.5, 0.75])).astype(int)
        for col, index in enumerate(indices):
            bm_slice = np.take(bm03_mask, index, axis=axis)
            label_slice = np.take(annotation_mask, index, axis=axis)
            plane_metrics[f'{orientation}: low index {index}'] = float(
                2 * (bm_slice & label_slice).sum() / (bm_slice.sum() + label_slice.sum()))
            if kind == 'intensity':
                section = np.take(intensity_overlay, index, axis=axis)
            else:
                gray = np.take(bm03_display, index, axis=axis)
                labels = np.take(annotation_array, index, axis=axis)
                edge = np.zeros(labels.shape, dtype=bool)
                edge[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0)
                edge[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0)
                section = np.stack((gray, gray, gray), axis=-1)
                section[edge] = [0, 1, 1]
            axes[row, col].imshow(section, origin='lower', interpolation='nearest', aspect='auto')
            axes[row, col].set_title(f'{orientation}, index {index}, union Dice {plane_metrics[f"{orientation}: low index {index}"]:.3f}')
            axes[row, col].set_xlabel('BM03 low-resolution grid index')
            axes[row, col].set_ylabel('BM03 low-resolution grid index')
    fig.suptitle('BM03 magenta / Allen average green' if kind == 'intensity'
                 else 'BM03 grayscale / Allen annotation boundaries cyan; QC only')
    fig.savefig(out / f'orthogonal_{kind}_overlays.png', dpi=145)
    plt.close(fig)

normals = ([0.37, 0.78, 0.50], [0.71, -0.39, 0.59])
bm03_centroid_zyx = np.argwhere(bm03_mask).mean(axis=0)
bm03_centroid_xyz_mm = np.asarray(bm03.GetOrigin()) + np.asarray(bm03.GetSpacing()) * bm03_centroid_zyx[::-1]
pixel_mm = 0.065
grid_mm = (np.arange(176) - 87.5) * pixel_mm
yy, xx = np.meshgrid(grid_mm, grid_mm, indexing='ij')
fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
oblique_metrics = {}
for row, direction in enumerate(normals):
    normal = np.asarray(direction) / np.linalg.norm(direction)
    u = np.array([1., 0., 0.]) - normal[0] * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    physical_xyz_mm = bm03_centroid_xyz_mm + xx[..., None] * u + yy[..., None] * v
    index_xyz = (physical_xyz_mm - bm03.GetOrigin()) / bm03.GetSpacing()
    coords_zyx = index_xyz[..., ::-1].transpose(2, 0, 1)
    sample_bm03 = ndi.map_coordinates(bm03_display, coords_zyx, order=1, mode='constant', cval=0.)
    sample_ccf = ndi.map_coordinates(ccf_display, coords_zyx, order=1, mode='constant', cval=0.)
    sample_labels = ndi.map_coordinates(annotation_array, coords_zyx, order=0, mode='constant', cval=0)
    sample_bm_mask = ndi.map_coordinates(bm03_mask.astype(np.uint8), coords_zyx,
                                         order=0, mode='constant', cval=0) > 0
    sample_label_mask = sample_labels > 0
    oblique_metrics[f'normal_{row}'] = float(2 * (sample_bm_mask & sample_label_mask).sum() /
                                              (sample_bm_mask.sum() + sample_label_mask.sum()))
    edge = np.zeros(sample_labels.shape, dtype=bool)
    edge[:, 1:] |= (sample_labels[:, 1:] != sample_labels[:, :-1]) & (sample_labels[:, 1:] > 0)
    edge[1:, :] |= (sample_labels[1:, :] != sample_labels[:-1, :]) & (sample_labels[1:, :] > 0)
    label_overlay = np.stack((sample_bm03, sample_bm03, sample_bm03), axis=-1)
    label_overlay[edge] = [0, 1, 1]
    images = (sample_bm03, sample_ccf,
              np.stack((sample_bm03, sample_ccf, sample_bm03), axis=-1), label_overlay)
    for col, section in enumerate(images):
        axes[row, col].imshow(section, cmap='gray' if col < 2 else None,
                              origin='lower', interpolation='nearest', extent=[grid_mm[0], grid_mm[-1], grid_mm[0], grid_mm[-1]])
        axes[row, col].set_title(('BM03', 'Allen average', 'intensity overlay', 'annotation boundaries')[col])
        axes[row, col].set_xlabel('synthetic plane mm')
        axes[row, col].set_ylabel('synthetic plane mm')
fig.suptitle('Two synthetic oblique reslices of BM03 and pulled Allen; not acquired oblique sections')
fig.savefig(out / 'oblique_qc.png', dpi=145)
plt.close(fig)

bm03_center_ijk = (np.asarray(bm03_nifti.shape) - 1) / 2
bm03_center_mm = np.asarray(bm03_nifti.header.get_zooms()[:3]) * bm03_center_ijk
dapi_center_ras_mm = bm03_to_dapi_ras_matrix @ bm03_center_mm + bm03_to_dapi_ras_translation
dapi_center_lps_mm = ras_to_lps @ dapi_center_ras_mm
ccf_center_um = ccf_matrix @ dapi_center_lps_mm + ccf_translation
ccf_index = (ccf_center_um - ccf_origin) / ccf_spacing
composition_residual_um = float(np.linalg.norm(
    np.asarray(bm03_to_ccf.TransformPoint(bm03_center_mm.tolist())) - ccf_center_um))

report = {
    'scope': 'BM03 native-grid intensity/annotation anatomy QC only. No BM17/BM18, model training, saved voxel label volume, or weak-label approval.',
    'input_sha256': hashes,
    'specimen_pilot_report': str(specimen_report_path),
    'published_direction_QC_report': str(convention_report_path),
    'coordinate_chain': 'BM03 synthetic metric xyz mm (header pixdim, origin 0, direction I) -> inverse of selected 162 DAPI-RAS-mm-to-BM03-mm affine -> flip RAS x/y signs to DAPI-LPS mm -> published ITK DAPI-LPS-mm-to-Allen-LPS-um pull -> Allen intensity/annotation input grid. Resample output grid is BM03; labels nearest-neighbor in memory only.',
    'BM03_source_header_units': bm03_nifti.header.get_xyzt_units(),
    'BM03_source_identity_sform_unusable': True,
    'BM03_low_grid': {'size_xyz': bm03.GetSize(), 'spacing_mm_xyz': bm03.GetSpacing(),
                      'origin_mm_xyz': bm03.GetOrigin(), 'direction': bm03.GetDirection(),
                      'downsampling': 'SimpleITK BinShrink block mean, factors xyz=(8,1,8)'},
    'Allen_grid': {'size_xyz': ccf_size, 'spacing_um_xyz': ccf_spacing,
                   'origin_um_xyz': ccf_origin, 'direction': ccf_direction},
    'composed_BM03_mm_to_CCF_um_matrix': composed_matrix.tolist(),
    'composed_BM03_mm_to_CCF_um_translation': composed_translation.tolist(),
    'point_check': {'BM03_center_original_index_xyz': bm03_center_ijk.tolist(),
                    'BM03_center_native_mm': bm03_center_mm.tolist(),
                    'DAPI_RAS_mm': dapi_center_ras_mm.tolist(),
                    'DAPI_LPS_mm': dapi_center_lps_mm.tolist(),
                    'CCF_LPS_um': ccf_center_um.tolist(),
                    'CCF_index_xyz': ccf_index.tolist(),
                    'explicit_vs_composed_residual_um': composition_residual_um},
    'gross_metrics': metrics,
    'orthogonal_annotation_union_dice': plane_metrics,
    'oblique_annotation_union_dice': oblique_metrics,
    'oblique_qc': {'normals_BM03_synthetic_xyz': normals, 'pixel_mm': pixel_mm,
                   'size': 176, 'center_BM03_native_mm': bm03_centroid_xyz_mm.tolist(),
                   'kind': 'synthetic reslices, not independent physical sections'},
    'limitations': 'BM03 lacks usable physical sform and unit metadata. Its metric identity frame is inherited from the 162 same-modality registration, not independent anatomical ground truth. Allen average is a different modality and annotation boundaries are registered-average priors; global Dice/surface distance can hide local region errors. No local landmarks, specimen-specific nonlinear registration, or acquired oblique sections were evaluated. Do not use the resampled annotation as training truth from this report.',
}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'gross_metrics': metrics,
                  'BM03_center_CCF_index': ccf_index.tolist(),
                  'composition_residual_um': composition_residual_um}, indent=2))
