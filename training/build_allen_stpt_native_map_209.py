"""Map the first frozen TRAIN STPT specimen's native 50-µm raster to Allen CCF."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


root = Path('I:/AnatomyTracker')
source = root / 'data/allen_stpt_registered_volume_207_train_001'
output = root / 'data/allen_stpt_native_map_209_train_001'
output.mkdir(parents=True, exist_ok=True)
dataset = json.loads((source / 'dataset_metadata.json').read_text())['msg'][0]
assert dataset['id'] == 100140756 and dataset['specimen']['donor_id'] == 14383
alignment = dataset['alignment3d']
tvr = np.array([alignment[f'tvr_{i:02d}'] for i in range(12)], dtype=np.float64)
field = np.memmap(source / 'dfmfld.raw', dtype='<f4', mode='r', shape=(456, 320, 528, 3))
image = json.loads((source / 'native_section_60_metadata.json').read_text())['msg'][0]
assert image['width'] == 39984 and image['height'] == 29984
first, last = 2, 141
width, height = 280, 210
spacing_x = (39984 // 32) * 11.2 / width
spacing_y = (29984 // 32) * 11.2 / height
xx, yy = np.meshgrid((np.arange(width) + .5) * spacing_x,
                     (np.arange(height) + .5) * spacing_y)
native_xy = np.column_stack((xx.ravel(), yy.ravel()))
mapping_path = output / '100140756_native_to_ccf_ap_dv_ml_um.npy'
mapping = np.lib.format.open_memmap(mapping_path, mode='w+', dtype='<f4',
                                    shape=(last - first + 1, height, width, 3))
residual_p95 = []

for section in range(first, last + 1):
    native = np.column_stack((native_xy, np.full(len(native_xy), section * 100.)))
    affine_ccf = native @ tvr[:9].reshape(3, 3).T + tvr[9:]
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
    residual_p95.append(float(np.percentile(np.linalg.norm(ccf + displacement - affine_ccf, axis=1), 95)))
    mapping[section - first] = ccf.reshape(height, width, 3)

mapping.flush()
with mapping_path.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(output / 'binding.json').write_text(json.dumps({
    'experiment_id': 100140756, 'specimen_id': 710918, 'donor_id': 14383,
    'section_number_first': first, 'section_number_last': last,
    'shape_zyxc': list(mapping.shape), 'native_xy_um_per_pixel': [spacing_x, spacing_y],
    'native_z_um': 'section_number * 100',
    'coordinate_units_and_axes': 'CCFv3 2017 reference-space-9 AP posterior, DV inferior, ML right; micrometres',
    'transform': 'native -> Allen tvr affine -> six fixed-point iterations in authored CCF-to-native displacement field',
    'inverse_residual_p95_um_max': max(residual_p95),
    'mapping_sha256': digest,
}, indent=2) + '\n')
print(json.dumps({'mapping_sha256': digest, 'max_inverse_residual_p95_um': max(residual_p95)}))
