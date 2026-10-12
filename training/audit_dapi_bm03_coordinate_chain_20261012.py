"""TRAIN-only BM03 coordinate-chain audit; reads no image pixels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


root = Path('I:/AnatomyTracker')
source = root / 'data/g_node_dapi_cc0_individual_volumes_20261011'
report162_path = root / 'reports/dapi_bm03_antialiased_pilot_162/report.json'
transform162_path = root / 'reports/dapi_bm03_antialiased_pilot_162/selected_fixed_to_bm03.tfm'
report163_path = root / 'reports/dapi_ccf_affine_direction_qc_163/report.json'
published_path = source / 'ccf_to_dapi_affine_transformation.txt'
report164_path = root / 'reports/bm03_ccf_composed_qc_164/report.json'
receipt229_path = root / 'reports/dapi_train_landmark_grid_229_001/complete.json'
out = root / 'reports/dapi_bm03_coordinate_chain_audit_20261012_001'

manifest = json.loads((source / 'source_manifest.json').read_text())
r162 = json.loads(report162_path.read_text())
r163 = json.loads(report163_path.read_text())
r164 = json.loads(report164_path.read_text())
r229 = json.loads(receipt229_path.read_text())
assert set(r229['specimens']) == {'BM03', 'BM04', 'BM06', 'BM10'}
assert r162['input_sha256']['BM03.nii'] == manifest['files']['BM03.nii']['sha256'] == r229['specimens']['BM03']['specimen_sha256']
assert r162['input_sha256']['dapi_template.nii.gz'] == manifest['files']['dapi_template.nii.gz']['sha256'] == r229['average_sha256']
assert hashlib.sha256(transform162_path.read_bytes()).hexdigest() == r229['specimens']['BM03']['transform_sha256'] == r164['input_sha256'][str(transform162_path)]
assert hashlib.sha256(report162_path.read_bytes()).hexdigest() == r164['input_sha256'][str(report162_path)]
assert hashlib.sha256(published_path.read_bytes()).hexdigest() == r163['input_sha256'][str(published_path)] == r164['input_sha256'][str(published_path)]
assert np.allclose(manifest['files']['BM03.nii']['selected_affine'], np.eye(4))
assert manifest['files']['BM03.nii']['spatial_unit'] == 'unknown'
assert np.allclose(r163['DAPI_NIfTI_RAS_affine_mm'], manifest['files']['dapi_template.nii.gz']['selected_affine'])

bm03_from_dapi = np.asarray(r162['selected_fixed_physical_to_bm03_physical_mm'])
dapi_lps_to_ccf = np.eye(4)
dapi_lps_to_ccf[:3, :3] = r163['published_affine_matrix_DAPI_LPS_mm_to_CCF_LPS_um']
dapi_lps_to_ccf[:3, 3] = r163['published_affine_translation_CCF_um']
ras_to_lps = np.diag([-1., -1., 1., 1.])
bm03_to_ccf = dapi_lps_to_ccf @ ras_to_lps @ np.linalg.inv(bm03_from_dapi)

saved_bm03 = sitk.ReadTransform(str(transform162_path))
saved_ccf = sitk.ReadTransform(str(published_path))
assert np.allclose(saved_bm03.GetFixedParameters(), 0)
assert np.allclose(saved_ccf.GetFixedParameters(), 0)
assert np.allclose(np.asarray(saved_bm03.GetMatrix()).reshape(3, 3), bm03_from_dapi[:3, :3])
assert np.allclose(saved_bm03.GetTranslation(), bm03_from_dapi[:3, 3])
assert np.allclose(np.asarray(saved_ccf.GetMatrix()).reshape(3, 3), dapi_lps_to_ccf[:3, :3])
assert np.allclose(saved_ccf.GetTranslation(), dapi_lps_to_ccf[:3, 3])
assert np.allclose(bm03_to_ccf[:3, :3], r164['composed_BM03_mm_to_CCF_um_matrix'])
assert np.allclose(bm03_to_ccf[:3, 3], r164['composed_BM03_mm_to_CCF_um_translation'])

size = np.asarray(r162['source_shapes_ijk']['BM03'])
spacing = np.asarray(manifest['files']['BM03.nii']['header_voxel_size'])
center = (size - 1) / 2
indices = {'center': center}
for axis in range(3):
    for edge, index in (('min', 0.), ('max', float(size[axis] - 1))):
        point = center.copy()
        point[axis] = index
        indices[f'axis_{axis}_{edge}_face_center'] = point

ccf_grid = r164['Allen_grid']
ccf_grid_matrix = np.asarray(ccf_grid['direction']).reshape(3, 3) @ np.diag(ccf_grid['spacing_um_xyz'])
max_roundtrip_mm = 0.
max_composition_difference_um = 0.
points = {}
for name, index in indices.items():
    bm03_mm = np.r_[index * spacing, 1.]
    dapi_ras_mm = np.linalg.inv(bm03_from_dapi) @ bm03_mm
    dapi_lps_mm = ras_to_lps @ dapi_ras_mm
    ccf_um = dapi_lps_to_ccf @ dapi_lps_mm
    direct_ccf_um = bm03_to_ccf @ bm03_mm
    bm03_return_mm = bm03_from_dapi @ ras_to_lps @ np.linalg.inv(dapi_lps_to_ccf) @ ccf_um
    max_roundtrip_mm = max(max_roundtrip_mm, float(np.linalg.norm(bm03_return_mm[:3] - bm03_mm[:3])))
    max_composition_difference_um = max(max_composition_difference_um, float(np.linalg.norm(direct_ccf_um[:3] - ccf_um[:3])))
    points[name] = {
        'native_index_ijk': index.tolist(),
        'Allen_index_xyz': np.linalg.solve(ccf_grid_matrix, ccf_um[:3] - ccf_grid['origin_um_xyz']).tolist(),
    }

center_164_delta = float(np.linalg.norm(np.asarray(points['center']['Allen_index_xyz']) - r164['point_check']['CCF_index_xyz']))
assert max_roundtrip_mm < 1e-8
assert max_composition_difference_um < 1e-6
assert center_164_delta < 1e-8

report = {
    'scope': 'Read-only input audit: BM03 and DAPI-average metadata, 162/163/164/229 reports and two text transforms. No image pixels, registration, CCF labels, or BM15-18 files read.',
    'input_reports': [str(report162_path), str(report163_path), str(report164_path), str(receipt229_path)],
    'coordinate_chain': 'BM03 index * assumed pixdim mm (origin 0, identity direction) -> inverse 162 fixed DAPI RAS-mm-to-BM03-mm pull -> RAS-to-LPS x/y sign flip -> published DAPI-LPS-mm-to-Allen-LPS-um pull. The published affine already includes mm-to-um scale.',
    'point_checks': points,
    'point_check_interpretation': 'Face-centre probes test transform arithmetic and axis direction only; edge points may lie outside Allen image support and are not anatomical landmarks.',
    'max_roundtrip_error_mm': max_roundtrip_mm,
    'max_explicit_vs_composed_error_um': max_composition_difference_um,
    'center_index_difference_vs_164': center_164_delta,
    'Allen_LPS_um_per_assumed_BM03_mm_axis_columns': bm03_to_ccf[:3, :3].tolist(),
    'source_orientation_evidence': {
        'BM03_header': 'Identity selected sform, unknown spatial unit; pixdim interpreted as mm only in the 162 constructed frame. It does not establish native world origin or anatomical signs.',
        'acquisition': 'The frozen 161/162 source contract reports physical 100-um coronal sections, consistent with native axis 1 as the stack/AP axis; original positive slice order is not documented here.',
        'DAPI_average': 'Metric RAS-mm NIfTI affine is recorded in 163 and the source manifest; SimpleITK reads this NIfTI in LPS, so the chain requires one RAS/LPS sign flip.',
        'positive_AP_sign': 'unresolved from available source/acquisition evidence',
        'native_left_right_sign': 'unresolved from available source/acquisition evidence',
        'native_origin': 'unresolved; zero is a registration convention, not a measured anatomical origin',
    },
    'landmark_status': '229 compares four preselected AP positions for BM03/BM04/BM06/BM10 but records visible ventricular, hippocampal/deep and edge offsets; no independent signed landmarks or local error bounds.',
    'decision': 'Internal transform arithmetic passes. BM03 is not yet admitted as weak CCF-coordinate pose/deformation or region-label training truth. Resolve acquisition AP/LR signs and predeclared bilateral multi-AP landmarks first; exclude uncertain regions. Synthetic obliques are interpolated from coronal sections, not acquired-oblique validation.',
    'holdout_boundary': 'No individual BM15-18 image file is accessed. The frozen manifest lists their metadata; the 229 receipt covers TRAIN BM03/BM04/BM06/BM10. BM15/BM16 remain DEV and BM17/BM18 sealed. The DAPI population average contains all 12 animals and is not a fully independent secondary reference holdout.',
}
out.mkdir(parents=True, exist_ok=True)
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ('max_roundtrip_error_mm', 'max_explicit_vs_composed_error_um', 'center_index_difference_vs_164', 'decision')}, indent=2))
