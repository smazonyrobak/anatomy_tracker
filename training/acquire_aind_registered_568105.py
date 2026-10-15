"""Acquire one licensed registered SmartSPIM volume for a source-shift diagnostic."""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import blosc2
import numpy as np
import requests

asset = 'SmartSPIM_568105_2024-03-06_10-19-06_stitched_2024-03-08_03-13-28'
channel = 'Ex_639_Em_667'
base = f'https://aind-open-data.s3.us-west-2.amazonaws.com/{asset}'
zarr = f'{base}/image_atlas_alignment/{channel}/OMEZarr/image.zarr/0'
out = ROOT / 'data/aind_smartspim_568105_registered'
out.mkdir(parents=True, exist_ok=True)
target = out / 'registered_ml_dv_ap_float32.npy'
temporary = out / 'registered_ml_dv_ap_float32.partial.npy'
assert not target.exists() and not temporary.exists()

description_response = requests.get(f'{base}/data_description.json', timeout=30)
description_response.raise_for_status()
description = description_response.json()
assert description['name'] == asset and description['license'] == 'CC-BY-4.0'
subject_response = requests.get(f'{base}/subject.json', timeout=30)
subject_response.raise_for_status()
assert str(subject_response.json()['subject_id']) == '568105'
array_response = requests.get(f'{zarr}/.zarray', timeout=30)
array_response.raise_for_status()
array = array_response.json()
assert array['shape'] == [1, 1, 456, 320, 528]
assert array['chunks'] == [1, 1, 64, 64, 64] and array['dtype'] == '<f4'
assert array['dimension_separator'] == '/'

chunks = [(z, y, x, f'{zarr}/0/0/{z}/{y}/{x}')
          for z in range(8) for y in range(5) for x in range(9)]
volume = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                    shape=(456, 320, 528))
digests = []
with ThreadPoolExecutor(max_workers=8) as pool:
    for (z, y, x, url), response in zip(chunks, pool.map(
            partial(requests.get, timeout=45), [row[3] for row in chunks])):
        response.raise_for_status()
        shape = (min(64, 456 - z * 64), min(64, 320 - y * 64),
                 min(64, 528 - x * 64))
        block = np.frombuffer(blosc2.decompress(response.content),
                              dtype='<f4').reshape((64, 64, 64))[
                                  :shape[0], :shape[1], :shape[2]]
        volume[z * 64:z * 64 + shape[0], y * 64:y * 64 + shape[1],
               x * 64:x * 64 + shape[2]] = block
        digests.append(hashlib.sha256(response.content).hexdigest())
volume.flush()
del volume
with temporary.open('rb') as stream:
    volume_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
temporary.replace(target)

receipt = {
    'asset': asset, 'subject_id': '568105', 'channel': channel,
    'source_grid': 'registered OME-Zarr level 0, [ML,DV,AP] array order, 25 um; signed CCF transform unverified',
    'data_license': description['license'],
    'description_url': f'{base}/data_description.json',
    'description_sha256': hashlib.sha256(description_response.content).hexdigest(),
    'subject_url': f'{base}/subject.json',
    'subject_sha256': hashlib.sha256(subject_response.content).hexdigest(),
    'zarray_url': f'{zarr}/.zarray',
    'zarray_sha256': hashlib.sha256(array_response.content).hexdigest(),
    'chunk_sha256': {f'{z}/{y}/{x}': digest for (z, y, x, _), digest in zip(chunks, digests)},
    'npy_sha256': volume_sha256,
    'scope': 'TRAIN-only source-shift diagnostic; not exact pose truth, physical histology, or validation',
}
(out / 'source.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
print({'path': str(target), 'shape': [456, 320, 528],
       'chunks': len(chunks), 'sha256': volume_sha256})
