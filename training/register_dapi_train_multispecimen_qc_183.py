"""QC-only antialiased DAPI-average registration of three TRAIN specimens."""

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
out = root / 'reports/dapi_train_multispecimen_affine_qc_183_001'
prior_dir = root / 'reports/dapi_bm03_antialiased_pilot_162'
manifest_path = source / 'source_manifest.json'
protocol_path = Path(__file__).resolve().parents[1] / 'docs/publication/DAPI_TRAIN_MULTISPECIMEN_AFFINE_QC_183_PROTOCOL_20261011.md'
fixed_path = source / 'dapi_template.nii.gz'
animals = ('BM04', 'BM06', 'BM10')
out.mkdir(parents=True, exist_ok=False)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
script_hash = sha(Path(__file__))
protocol_hash = sha(protocol_path)
manifest_hash = sha(manifest_path)
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
source_hashes = {name: manifest['files'][name]['sha256']
                 for name in ('BM03.nii', 'BM04.nii', 'BM06.nii', 'BM10.nii', fixed_path.name)}
prior_report = json.loads((prior_dir / 'report.json').read_text(encoding='utf-8'))
bm03_matrix = np.asarray(prior_report['selected_fixed_physical_to_bm03_physical_mm'])
fixed_hash = sha(fixed_path)
assert fixed_hash == source_hashes[fixed_path.name] == prior_report['input_sha256'][fixed_path.name]
assert prior_report['input_sha256']['BM03.nii'] == source_hashes['BM03.nii']

fixed_nii = nib.load(fixed_path)
fixed_full = np.asarray(fixed_nii.dataobj, dtype=np.float32)
fixed_np = ndi.gaussian_filter(fixed_full, (3.5, 1.2, 3.5), mode='nearest', truncate=3.)[::8, ::3, ::8].copy()
del fixed_full
gc.collect()
fixed_spacing = np.linalg.norm(fixed_nii.affine[:3, :3], axis=0) * (8, 3, 8)
fixed = sitk.GetImageFromArray(fixed_np.transpose(2, 1, 0))
fixed.SetSpacing(fixed_spacing.tolist())
fixed.SetOrigin(fixed_nii.affine[:3, 3].tolist())
fixed.SetDirection(tuple((fixed_nii.affine[:3, :3] / np.linalg.norm(fixed_nii.affine[:3, :3], axis=0)).ravel()))
fixed_mask = fixed_np > 0.2
interior = ndi.binary_erosion(fixed_mask, iterations=3)
fixed_display = np.clip(fixed_np / np.percentile(fixed_np, 99.5), 0, 1)
entries = {
    'BM03': {
        'status': 'existing_162_reference_only_not_recomputed',
        'report': str(prior_dir / 'report.json'),
        'report_sha256': sha(prior_dir / 'report.json'),
        'transform': str(prior_dir / 'selected_fixed_to_bm03.tfm'),
        'transform_sha256': sha(prior_dir / 'selected_fixed_to_bm03.tfm'),
        'orthogonal_overlay': str(prior_dir / 'orthogonal_overlays.png'),
        'orthogonal_overlay_sha256': sha(prior_dir / 'orthogonal_overlays.png'),
    }
}

for animal in animals:
    moving_path = source / f'{animal}.nii'
    moving_hash = sha(moving_path)
    assert moving_hash == source_hashes[moving_path.name]
    moving_nii = nib.load(moving_path)
    moving_full = np.asarray(moving_nii.dataobj, dtype=np.float32)
    moving_np = ndi.gaussian_filter(moving_full, (3.5, 0., 3.5), mode='nearest', truncate=3.)[::8, :, ::8].copy()
    del moving_full
    gc.collect()
    moving_spacing = np.asarray(moving_nii.header.get_zooms()[:3]) * (8, 1, 8)
    moving = sitk.GetImageFromArray(moving_np.transpose(2, 1, 0))
    moving.SetSpacing(moving_spacing.tolist())
    moving.SetOrigin((0., 0., 0.))
    moving.SetDirection(tuple(np.eye(3).ravel()))

    # BM03 is an intensity-registration seed, not a known anatomical orientation.
    seed_matrix = bm03_matrix.copy()
    seed_matrix[1, 3] += (moving_nii.shape[1] - 68) * moving_spacing[1] / 2
    seed = sitk.AffineTransform(3)
    seed.SetMatrix(seed_matrix[:3, :3].ravel().tolist())
    seed.SetTranslation(seed_matrix[:3, 3].tolist())
    fitted = sitk.AffineTransform(seed)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetMetricFixedMask(sitk.Cast(fixed > 0.2, sitk.sitkUInt8))
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.08, 20261011 + int(animal[2:]))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([0.15, 0.])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=5e-5, numberOfIterations=100,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(fitted, inPlace=True)
    fitted = registration.Execute(fixed, moving)

    warped, metrics = {}, {}
    for name, transform in (('bm03_seed', seed), ('refined', fitted)):
        warped[name] = sitk.GetArrayFromImage(sitk.Resample(
            moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)).transpose(2, 1, 0)
        moving_mask = warped[name] > 0.2
        metrics[name] = {
            'tissue_dice_at_0_2': float(2 * (fixed_mask & moving_mask).sum() /
                                         (fixed_mask.sum() + moving_mask.sum())),
            'pearson_r_eroded_fixed_tissue': float(np.corrcoef(
                fixed_np[interior], warped[name][interior])[0, 1]),
        }
    singular_values = np.linalg.svd(np.asarray(fitted.GetMatrix()).reshape(3, 3))[1]
    refined_for_qc = (np.isfinite(registration.GetMetricValue())
                      and np.linalg.det(np.asarray(fitted.GetMatrix()).reshape(3, 3)) > 0
                      and np.all((singular_values > 0.8) & (singular_values < 1.2))
                      and metrics['refined']['pearson_r_eroded_fixed_tissue'] >=
                      metrics['bm03_seed']['pearson_r_eroded_fixed_tissue'] + 0.005
                      and metrics['refined']['tissue_dice_at_0_2'] >=
                      metrics['bm03_seed']['tissue_dice_at_0_2'] - 0.005)
    selected_name, selected = ('refined', fitted) if refined_for_qc else ('bm03_seed', seed)
    matrix = np.asarray(selected.GetMatrix()).reshape(3, 3)
    center = np.asarray(selected.GetCenter())
    offset = center + np.asarray(selected.GetTranslation()) - matrix @ center
    pull = np.eye(4)
    pull[:3, :3], pull[:3, 3] = matrix, offset

    specimen_dir = out / animal
    specimen_dir.mkdir()
    transform_path = specimen_dir / 'selected_fixed_to_specimen.tfm'
    overlay_path = specimen_dir / 'orthogonal_overlays.png'
    report_path = specimen_dir / 'report.json'
    sitk.WriteTransform(selected, str(transform_path))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, name in enumerate(('bm03_seed', selected_name)):
        moving_display = np.clip(warped[name] / np.percentile(moving_np, 99.5), 0, 1)
        overlay = np.stack((fixed_display, moving_display, fixed_display), -1)
        sections = (overlay[overlay.shape[0] // 2].transpose(1, 0, 2),
                    overlay[:, overlay.shape[1] // 2].transpose(1, 0, 2),
                    overlay[:, :, overlay.shape[2] // 2].transpose(1, 0, 2))
        for axis, section in enumerate(sections):
            axes[row, axis].imshow(section, origin='lower', interpolation='nearest', aspect='auto')
            axes[row, axis].set_title(f'{name}: fixed voxel axis {axis} midpoint')
            axes[row, axis].axis('off')
    fig.suptitle(f'DAPI average magenta, {animal} green; central orthogonal reslices')
    fig.savefig(overlay_path, dpi=140)
    plt.close(fig)

    report = {
        'scope': 'TRAIN-specimen same-modality affine QC candidate only; no CCF labels, model examples, or validated local landmarks.',
        'animal_id': animal,
        'dataset_doi': 'https://doi.org/10.12751/g-node.16wrxa',
        'source_revision': '10.12751/g-node.16wrxa, locally hash-frozen files',
        'script_sha256': script_hash,
        'protocol_sha256': protocol_hash,
        'source_manifest_sha256': manifest_hash,
        'input_sha256': {moving_path.name: moving_hash, fixed_path.name: fixed_hash},
        'native_source_url': manifest['files'][moving_path.name]['immutable_source_url'],
        'bm03_seed_report': str(prior_dir / 'report.json'),
        'bm03_seed_report_sha256': entries['BM03']['report_sha256'],
        'source_shapes_ijk': {animal: list(moving_nii.shape), 'DAPI_average': list(fixed_nii.shape)},
        'native_selected_sform': moving_nii.affine.tolist(),
        'native_header_units': moving_nii.header.get_xyzt_units()[0],
        'antialias': 'Full-volume Gaussian filter before subsampling; scipy.ndimage.gaussian_filter float32, mode nearest, truncate 3 sigma.',
        'sigma_original_voxels_ijk': {animal: [3.5, 0., 3.5], 'DAPI_average': [3.5, 1.2, 3.5]},
        'strides_ijk': {animal: [8, 1, 8], 'DAPI_average': [8, 3, 8]},
        'low_shapes_ijk': {animal: list(moving_np.shape), 'DAPI_average': list(fixed_np.shape)},
        'spacing_mm_xyz': {animal: moving_spacing.tolist(), 'DAPI_average': fixed_spacing.tolist()},
        'physical_convention': 'SimpleITK xyz follows native NIfTI array axes 0/1/2. Individual header pixdim is assumed millimetres, native origin zero and direction identity in memory, despite identity sform and unknown units. Native positive directions, handedness, and AP sign are unverified. The DAPI average uses its metric sform. This pull maps DAPI-average fixed physical xyz to assumed native specimen physical xyz.',
        'initialization': 'BM03 selected 162 affine, AP translation adjusted only by half the difference in native stack extent; not an anatomical orientation prior.',
        'metric': 'Correlation optimized on random 8% fixed tissue, threshold 0.2. Dice and Pearson r measured on the same fixed grid, not independent anatomical validation.',
        'metrics': metrics,
        'optimizer_metric': float(registration.GetMetricValue()),
        'optimizer_stop': registration.GetOptimizerStopConditionDescription(),
        'affine_singular_values': singular_values.tolist(),
        'refinement_selected_for_qc': bool(refined_for_qc),
        'selected': selected_name,
        'selected_fixed_physical_to_specimen_physical_mm': pull.tolist(),
        'selected_specimen_physical_to_fixed_physical_mm': np.linalg.inv(pull).tolist(),
        'selected_transform_sha256': sha(transform_path),
        'orthogonal_overlay_sha256': sha(overlay_path),
        'label_eligibility': 'none: local landmarks, left/right/AP convention, and physical-section correspondence have not been validated',
        'limitations': 'The native volumes were reconstructed from 100-um coronal acquisitions. Intensity fit and gross tissue Dice can hide local anatomy errors. The average includes all 12 source animals. No DEV or sealed individual images were opened.',
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    entries[animal] = {
        'status': 'same_modality_affine_qc_candidate_not_label_approved',
        'report': str(report_path), 'report_sha256': sha(report_path),
        'transform': str(transform_path), 'transform_sha256': report['selected_transform_sha256'],
        'orthogonal_overlay': str(overlay_path),
        'orthogonal_overlay_sha256': report['orthogonal_overlay_sha256'],
    }
    print(animal, selected_name, metrics[selected_name], flush=True)
    del moving, moving_nii, moving_np, warped
    gc.collect()

(out / 'complete.json').write_text(json.dumps({
    'scope': 'QC-only TRAIN BM03/BM04/BM06/BM10; no DEV/secondary images, CCF labels, or model examples',
    'script_sha256': script_hash,
    'protocol_sha256': protocol_hash,
    'source_manifest_sha256': manifest_hash,
    'source_sha256': source_hashes,
    'animals': entries,
}, indent=2) + '\n', encoding='utf-8')
