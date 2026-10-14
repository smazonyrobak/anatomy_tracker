"""Put registered microCT intensities on the 25-um CCF tensor grid for TRAIN-only pilots."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
from scipy.ndimage import map_coordinates

source = ROOT / 'data/ebrains_humbel_microct_20p8um'
raw = np.load(source / 'registered_xyz_uint16.npy', mmap_mode='r')
assert raw.shape == (560, 384, 640) and raw.dtype == np.uint16
target = source / 'registered_ccf_ap_dv_ml_float32.npy'
assert not target.exists()
temporary = source / 'registered_ccf_ap_dv_ml_float32.partial.npy'
out = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                shape=(528, 320, 456))
dv = (np.arange(320, dtype=np.float32)[None, :, None] + .5) * 25 / 20.8 - .5
ml = (456 - (np.arange(456, dtype=np.float32)[None, None, :] + .5)) * 25 / 20.8 - .5
for start in range(0, 528, 16):
    stop = min(start + 16, 528)
    ap = (528 - (np.arange(start, stop, dtype=np.float32)[:, None, None] + .5)
          ) * 25 / 20.8 - .5
    coords = np.broadcast_arrays(ml, dv, ap)
    values = map_coordinates(raw, coords, order=1, mode='constant', cval=0,
                             prefilter=False, output=np.float32)
    out[start:stop] = np.clip((values - 9270) / (27039 - 9270), 0, 1)
out.flush()
del out
with temporary.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
temporary.replace(target)
(source / 'ccf_resample.json').write_text(json.dumps({
    'source_npy_sha256': json.loads((source / 'source.json').read_text())['npy_sha256'],
    'target_npy_sha256': digest, 'target_shape': [528, 320, 456],
    'target_axes': ['AP', 'DV', 'ML'], 'target_voxel_um': 25,
    'coordinates': 'CT XYZ index=[(456-(ML+.5))*25/20.8-.5,(DV+.5)*25/20.8-.5,(528-(AP+.5))*25/20.8-.5]',
    'interpolation': 'trilinear; constant zero outside CT',
    'intensity': 'clip((uint16-9270)/(27039-9270),0,1), published viewer window',
    'scope': 'registered single CT animal as TRAIN-source diversity probe, not pose truth for physical histology',
}, indent=2) + '\n', encoding='utf8')
print({'path': str(target), 'shape': (528, 320, 456), 'sha256': digest})
