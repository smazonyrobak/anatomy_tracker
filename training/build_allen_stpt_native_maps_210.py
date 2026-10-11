"""Build TRAIN native-red voxel-centre -> CCF maps after all eight 208 receipts exist.

The Allen deformation field is CCF -> native. Its inverse here is the same
six-step fixed-point approximation used for the completed donor-14383 map 209.
Virtual oblique reslices of these coronal stacks are not acquired oblique data.
"""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from scipy.ndimage import map_coordinates


ROOT = Path('I:/AnatomyTracker')
ACQUIRED = ROOT / 'data/allen_stpt_native_208_train_001'
PILOT = ROOT / 'data/allen_stpt_registered_volume_207_train_001'
OLD_MAP = ROOT / 'data/allen_stpt_native_map_209_train_001'
OUT = ROOT / 'data/allen_stpt_native_maps_210_train_001'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == content, path
    else:
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(content)
        os.replace(temporary, path)


# Do not read the frozen selection or any donor data until every donor receipt exists.
receipt_paths = sorted(ACQUIRED.glob('donor_*/receipt.json'))
assert len(receipt_paths) == 8 and (ACQUIRED / 'completed.json').is_file()
completion = json.loads((ACQUIRED / 'completed.json').read_text())
selection_path = ACQUIRED / 'frozen_selection.json'
selection_sha = sha(selection_path)
assert completion['donors'] == 8 and completion['frozen_selection_sha256'] == selection_sha
assert completion['development_calibration_final_images_accessed'] == 0
selection = json.loads(selection_path.read_text())
assert len(selection['donors']) == 8 and selection['selected_before_image_download_or_performance_inspection']
assert {p.parent.name for p in receipt_paths} == {f"donor_{s['reservation']['animal_id']}" for s in selection['donors']}
assert set(completion['donor_ids']) == {s['reservation']['animal_id'] for s in selection['donors']}
assert set(completion['experiment_ids']) == {s['reservation']['experiment_id'] for s in selection['donors']}

donors = []
for item in selection['donors']:
    reservation, experiment = item['reservation'], item['experiment']
    donor_id, experiment_id = reservation['animal_id'], reservation['experiment_id']
    assert reservation['proposed_split'] == experiment['split'] == 'train'
    assert experiment['reference_space_id'] == 9 and experiment['plane_of_section_id'] == 1
    directory = ACQUIRED / f'donor_{donor_id}'
    receipt_path = directory / 'receipt.json'
    assert sha(receipt_path) == completion['donor_receipt_sha256'][receipt_path.relative_to(ACQUIRED).as_posix()]
    receipt = json.loads(receipt_path.read_text())
    assert receipt['selection_sha256'] == selection_sha
    assert (receipt['donor_id'], receipt['specimen_id'], receipt['experiment_id']) == (
        donor_id, reservation['specimen_id'], experiment_id)
    geometry_path = directory / 'native_stack_geometry.json'
    geometry = json.loads(geometry_path.read_text())
    assert (geometry['animal_id'], geometry['specimen_id'], geometry['experiment_id']) == (
        donor_id, reservation['specimen_id'], experiment_id)
    assert geometry['axis_order'] == 'zyx' and geometry['dtype'] == 'uint8'
    assert geometry['field_files'] == item['field_files']
    assert geometry['alignment3d_tvr'] == experiment['alignment3d_tvr']
    assert len(geometry['section_numbers']) == receipt['section_count'] == geometry['shape'][0]
    assert geometry['section_numbers'] == list(range(geometry['section_numbers'][0],
                                                       geometry['section_numbers'][-1] + 1))
    origin = np.asarray(geometry['native_origin_first_voxel_center_um_xyz'], dtype=np.float64)
    spacing = np.asarray(geometry['native_spacing_um_xyz'], dtype=np.float64)
    assert spacing[2] == 100 and np.all(spacing > 0)
    assert np.allclose(origin[:2], spacing[:2] / 2, rtol=0, atol=1e-9)
    assert origin[2] == geometry['section_numbers'][0] * 100
    stack_path = ACQUIRED / geometry['paths']['stack']
    header_path = directory / 'dfmfld.mhd'
    for path in (stack_path, geometry_path, header_path, ACQUIRED / geometry['paths']['sections']):
        assert sha(path) == receipt['file_sha256'][path.relative_to(ACQUIRED).as_posix()]
    assert geometry['field_header_sha256'] == sha(header_path)
    assert tuple(np.load(stack_path, mmap_mode='r').shape) == tuple(geometry['shape'])
    donors.append((item, receipt_path, receipt, geometry_path, geometry, stack_path, header_path))

OUT.mkdir(parents=True, exist_ok=True)
bindings = []
for item, receipt_path, receipt, geometry_path, geometry, stack_path, header_path in donors:
    donor_id = item['reservation']['animal_id']
    experiment_id = item['reservation']['experiment_id']
    directory = OUT / f'donor_{donor_id}'
    directory.mkdir(exist_ok=True)
    binding_path = directory / 'binding.json'
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        assert binding['acquisition_receipt_sha256'] == sha(receipt_path)
        assert binding['native_stack_sha256'] == sha(stack_path)
        assert binding['native_geometry_sha256'] == sha(geometry_path)
        assert binding['field_header_sha256'] == sha(header_path)
        assert sha(Path(binding['map_path'])) == binding['map_sha256']
        if donor_id != 14383:
            assert sha(Path(binding['field_raw_path'])) == binding['field_raw_sha256']
        bindings.append(binding)
        print(json.dumps({'donor_id': donor_id, 'reused_completed_map': True}), flush=True)
        continue

    header = dict(line.split('=', 1) for line in header_path.read_text().splitlines() if '=' in line)
    header = {key.strip(): value.strip() for key, value in header.items()}
    assert header['NDims'] == '3' and header['CompressedData'] == 'False'
    assert header['BinaryDataByteOrderMSB'] == 'False' and header['ElementType'] == 'MET_FLOAT'
    assert header['ElementNumberOfChannels'] == '3' and header['ElementDataFile'] == 'dfmfld.raw'
    dims = np.fromstring(header['DimSize'], dtype=np.int64, sep=' ')
    field_spacing = np.fromstring(header['ElementSpacing'], dtype=np.float64, sep=' ')
    field_origin = np.fromstring(header['Offset'], dtype=np.float64, sep=' ')
    field_direction = np.fromstring(header['TransformMatrix'], dtype=np.float64, sep=' ').reshape(3, 3)
    assert len(dims) == len(field_spacing) == len(field_origin) == 3
    assert np.all(dims > 0) and np.all(field_spacing > 0)
    inverse_direction = np.linalg.inv(field_direction)
    expected_bytes = int(np.prod(dims) * 3 * 4)

    if donor_id == 14383:
        old_binding_path = OLD_MAP / 'binding.json'
        old_binding = json.loads(old_binding_path.read_text())
        map_path = OLD_MAP / f'{experiment_id}_native_to_ccf_ap_dv_ml_um.npy'
        assert (old_binding['donor_id'], old_binding['experiment_id'], old_binding['specimen_id']) == (
            donor_id, experiment_id, item['reservation']['specimen_id'])
        assert old_binding['shape_zyxc'] == [*geometry['shape'], 3]
        assert old_binding['section_number_first'] == geometry['section_numbers'][0]
        assert old_binding['section_number_last'] == geometry['section_numbers'][-1]
        assert np.allclose(old_binding['native_xy_um_per_pixel'], geometry['native_spacing_um_xyz'][:2],
                           rtol=0, atol=1e-9)
        assert sha(map_path) == old_binding['mapping_sha256']
        assert sha(PILOT / 'dfmfld.mhd') == sha(header_path)
        pilot_metadata = json.loads((PILOT / 'dataset_metadata.json').read_text())['msg'][0]
        pilot_tvr = [pilot_metadata['alignment3d'][f'tvr_{i:02d}'] for i in range(12)]
        assert np.allclose(pilot_tvr, geometry['alignment3d_tvr'], rtol=0, atol=1e-9)
        field_raw_path = PILOT / 'dfmfld.raw'
        assert field_raw_path.stat().st_size == expected_bytes
        residual = {'p95_um_max': old_binding['inverse_residual_p95_um_max'],
                    'source': 'completed 209 binding; map reused without recomputation'}
        previous_binding_sha = sha(old_binding_path)
    else:
        field_raw_path = directory / 'dfmfld.raw'
        download_path = directory / 'field_download.json'
        field_record = item['field_files']['DeformationFieldVoxels']
        url = field_record['download_url']
        assert url == f"https://api.brain-map.org/api/v2/well_known_file_download/{field_record['id']}"
        if download_path.exists():
            download = json.loads(download_path.read_text())
            assert download['field_id'] == field_record['id'] and download['requested_url'] == url
            assert field_raw_path.stat().st_size == expected_bytes
            assert sha(field_raw_path) == download['sha256']
        else:
            temporary = directory / 'dfmfld.raw.part'
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                assert response.status_code == 200 and urlparse(response.url).hostname == 'api.brain-map.org'
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                response_url = response.url
            assert temporary.stat().st_size == expected_bytes
            os.replace(temporary, field_raw_path)
            download = {'field_id': field_record['id'], 'requested_url': url,
                        'response_url': response_url, 'bytes': expected_bytes,
                        'sha256': sha(field_raw_path)}
            write_json(download_path, download)
        print(json.dumps({'donor_id': donor_id, 'field_downloaded_bytes': expected_bytes}), flush=True)
        field = np.memmap(field_raw_path, dtype='<f4', mode='r', shape=(*dims[::-1], 3))
        n, height, width = geometry['shape']
        native_origin = np.asarray(geometry['native_origin_first_voxel_center_um_xyz'], dtype=np.float64)
        native_spacing = np.asarray(geometry['native_spacing_um_xyz'], dtype=np.float64)
        xx, yy = np.meshgrid(native_origin[0] + np.arange(width) * native_spacing[0],
                             native_origin[1] + np.arange(height) * native_spacing[1])
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        map_path = directory / f'{experiment_id}_native_to_ccf_ap_dv_ml_um.npy'
        temporary = directory / f'{experiment_id}_native_to_ccf_ap_dv_ml_um.part.npy'
        mapping = np.lib.format.open_memmap(temporary, mode='w+', dtype='<f4',
                                            shape=(n, height, width, 3))
        p95, maximum, outside_fraction = [], [], []
        tvr = np.asarray(geometry['alignment3d_tvr'], dtype=np.float64)
        for section_index, section_number in enumerate(geometry['section_numbers']):
            native = np.column_stack((xy, np.full(len(xy), native_origin[2] + section_index * native_spacing[2])))
            assert native[0, 2] == section_number * 100
            affine_ccf = native @ tvr[:9].reshape(3, 3).T + tvr[9:]
            ccf = affine_ccf.copy()
            for iteration in range(7):
                index_xyz = ((ccf - field_origin) @ inverse_direction.T) / field_spacing
                index = (index_xyz[:, 2], index_xyz[:, 1], index_xyz[:, 0])
                displacement = np.column_stack([map_coordinates(field[..., axis], index,
                    order=1, mode='nearest') for axis in range(3)])
                if iteration < 6:
                    ccf = affine_ccf - displacement
            error = np.linalg.norm(ccf + displacement - affine_ccf, axis=1)
            assert np.isfinite(ccf).all() and np.isfinite(error).all()
            inside = ((index_xyz >= 0) & (index_xyz <= dims - 1)).all(1)
            p95.append(float(np.percentile(error, 95)))
            maximum.append(float(error.max()))
            outside_fraction.append(float(1 - inside.mean()))
            mapping[section_index] = ccf.reshape(height, width, 3)
        mapping.flush()
        del mapping, field
        os.replace(temporary, map_path)
        residual = {'p95_um_by_section': p95, 'max_um_by_section': maximum,
                    'outside_field_fraction_by_section': outside_fraction,
                    'p95_um_max': max(p95), 'max_um': max(maximum),
                    'outside_field_fraction_max': max(outside_fraction),
                    'source': 'six fixed-point updates; residual ||ccf + authored_field(ccf) - tvr(native)||'}
        previous_binding_sha = None

    binding = {'donor_id': donor_id, 'specimen_id': item['reservation']['specimen_id'],
        'experiment_id': experiment_id, 'split': 'train',
        'selection_sha256': selection_sha, 'acquisition_completion_sha256': sha(ACQUIRED / 'completed.json'),
        'acquisition_receipt_sha256': sha(receipt_path),
        'native_stack_path': str(stack_path), 'native_stack_sha256': sha(stack_path),
        'native_geometry_path': str(geometry_path), 'native_geometry_sha256': sha(geometry_path),
        'section_numbers': geometry['section_numbers'], 'shape_zyxc': [*geometry['shape'], 3],
        'native_origin_first_voxel_center_um_xyz': geometry['native_origin_first_voxel_center_um_xyz'],
        'native_spacing_um_xyz': geometry['native_spacing_um_xyz'],
        'alignment3d_tvr': geometry['alignment3d_tvr'],
        'alignment3d_source_api_url': geometry['alignment3d_source_api_url'],
        'field_header_id': item['field_files']['DeformationFieldHeader']['id'],
        'field_header_requested_url': item['field_files']['DeformationFieldHeader']['download_url'],
        'field_header_path': str(header_path), 'field_header_sha256': sha(header_path),
        'field_voxels_id': item['field_files']['DeformationFieldVoxels']['id'],
        'field_voxels_requested_url': item['field_files']['DeformationFieldVoxels']['download_url'],
        'field_api_response_sha256': item['field_api_sha256'],
        'field_raw_path': str(field_raw_path), 'field_raw_sha256': sha(field_raw_path),
        'field_shape_zyxc': [*dims[::-1].tolist(), 3],
        'field_origin_um_xyz': field_origin.tolist(), 'field_spacing_um_xyz': field_spacing.tolist(),
        'field_direction_xyz': field_direction.tolist(),
        'map_path': str(map_path), 'map_sha256': sha(map_path),
        'map_axes_units': 'CCFv3 2017 reference-space-9 AP/DV/ML micrometres at native voxel centres',
        'inverse_residual': residual, 'reused_209_binding_sha256': previous_binding_sha,
        'weak_alignment_not_anatomical_truth': True,
        'virtual_obliques_not_physical_validation': True}
    write_json(binding_path, binding)
    bindings.append(binding)
    print(json.dumps({'donor_id': donor_id, 'inverse_residual_p95_um_max': residual['p95_um_max'],
                      'map_sha256': binding['map_sha256']}), flush=True)

completed = {'donors': len(bindings), 'donor_ids': [x['donor_id'] for x in bindings],
    'selection_sha256': selection_sha, 'acquisition_completion_sha256': sha(ACQUIRED / 'completed.json'),
    'source_sha256': sha(Path(__file__)),
    'binding_sha256': {f"donor_{x['donor_id']}/binding.json": sha(OUT / f"donor_{x['donor_id']}" / 'binding.json')
                       for x in bindings},
    'all_split': 'train', 'external_pretrained_weights_used': False,
    'development_calibration_final_used': False,
    'scope': 'weak Allen-authored native-to-CCF coordinate maps; no physical oblique validation'}
write_json(OUT / 'completed.json', completed)
print(json.dumps({'completed': str(OUT / 'completed.json'), 'donors': len(bindings)}), flush=True)
