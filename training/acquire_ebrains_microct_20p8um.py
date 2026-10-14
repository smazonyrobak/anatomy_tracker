"""Cache the independently imaged, CCF-registered CC-BY microCT at 20.8 um on I:."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import tensorstore as ts

source = 'https://rgw.cscs.ch/ebrains:d-256fb2ff-93b2-4f3f-abbd-ccee0e52b020/'
out = ROOT / 'data/ebrains_humbel_microct_20p8um'
out.mkdir(parents=True, exist_ok=True)
image_path = out / 'registered_xyz_uint16.npy'
assert not image_path.exists()
volume = ts.open({'driver': 'neuroglancer_precomputed',
                  'kvstore': {'driver': 'http', 'base_url': source},
                  'scale_index': 5}, read=True).result()
image = np.asarray(volume.read().result()).squeeze()
assert image.shape == (560, 384, 640) and image.dtype == np.uint16
temporary = out / 'registered_xyz_uint16.partial.npy'
np.save(temporary, image)
with temporary.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
temporary.replace(image_path)
(out / 'source.json').write_text(json.dumps({
    'doi': '10.25493/JWM7-2G1',
    'citation': 'Humbel et al., Advanced Science 12, 2416879 (2025)',
    'license': 'CC BY 4.0',
    'source': source,
    'neuroglancer_scale_key': '20800nm',
    'array_axes': ['display_x', 'display_y', 'display_z'],
    'voxel_size_um': [20.8, 20.8, 20.8],
    'voxel_offset': [0, 0, 0],
    'atlas_segmentation_to_display_25um': [[0, 0, -1, 456], [0, 1, 0, 0], [-1, 0, 0, 528]],
    'scope': 'one registered real 3-D mouse; virtual cuts are not physically cut histology or animal-level validation',
    'shape': list(image.shape), 'dtype': str(image.dtype), 'npy_sha256': digest,
}, indent=2) + '\n', encoding='utf8')
print({'path': str(image_path), 'shape': image.shape, 'sha256': digest})
