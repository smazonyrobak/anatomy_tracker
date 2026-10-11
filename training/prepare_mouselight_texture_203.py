"""Resample registered MouseLight autofluorescence as appearance-only CCF texture."""

import hashlib
import json
import os
from pathlib import Path

os.environ['TEMP'] = os.environ['TMP'] = 'I:/AnatomyTracker/tmp'

import nrrd
import numpy as np
from scipy.ndimage import map_coordinates

root = Path('I:/AnatomyTracker')
source_id = os.environ.get('ANATOMY_MOUSELIGHT_SOURCE_ID', '2017-01-15')
source = root / f'data/janelia_mouselight_198/Transformed_{source_id}.nrrd'
volume, header = nrrd.read(source)
atlas, _ = nrrd.read(root / 'data/Allen Brain Atlas 25um/average_template_25.nrrd')
direction = np.asarray(header['space directions'])
inverse = np.linalg.inv(direction)
origin = np.asarray(header['space origin'])
ap, dv, ml = atlas.shape
dv_mm, ml_mm = np.meshgrid(np.arange(dv) * 25., np.arange(ml) * 25., indexing='ij')
texture = np.empty(atlas.shape, dtype=np.float16)
for a in range(ap):
    world = np.stack((ml_mm, dv_mm, np.full_like(ml_mm, a * 25.)), -1)
    coordinates = (world - origin) @ inverse
    texture[a] = map_coordinates(volume, tuple(coordinates[..., k] for k in range(3)),
                                  order=1, mode='constant', cval=0)
scale = float(np.percentile(texture[atlas > 0], 99))
texture = np.clip(texture.astype(np.float32) / scale, 0, 1).astype(np.float16)
texture[atlas == 0] = 0
out = root / f'data/janelia_mouselight_198/{source_id}_appearance_ccf25.npy'
np.save(out, texture)
record = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'output': str(out), 'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'shape_ap_dv_ml': [ap, dv, ml], 'normalization_inside_atlas_p99': scale,
    'registered_world_to_atlas_array_axes': 'z,y,x to AP,DV,ML; lateral sign unresolved',
    'allowed_use': 'appearance-only synthetic augmentation; any ML reflection must be explicit; not a biological laterality or site-level truth label',
    'source_license': 'CC BY 4.0, https://registry.opendata.aws/janelia-mouselight/',
    'source_id': source_id, 'specimen_id': source_id, 'experiment_id': None,
    'source_sha256_check_required_before_training': True}
out.with_suffix('.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
