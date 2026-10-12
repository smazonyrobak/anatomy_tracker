"""Save the exact verified Allen v6 float32 atlas once on I:."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

OUT = ROOT / 'data/allen_v6_float32_cache_248_001'
DOC = Path('docs/publication/ALLEN_V6_FLOAT32_CACHE_248_PROTOCOL_20261012.md')
atlas, annotation = allen._decode_and_preprocess_allen_v6()
assert atlas.shape == tuple(allen.ATLAS_FLOAT32_RECEIPT_V6['shape'])
assert atlas.dtype == np.float32
del annotation

OUT.mkdir(parents=True, exist_ok=False)
np.save(OUT / 'atlas_float32.npy', atlas, allow_pickle=False)
protocol = {
    'atlas_array_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'template_raw_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'annotation_raw_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'template_path': str(allen.TEMPLATE_PATH_V6),
    'annotation_path': str(allen.ANNOTATION_PATH_V6),
    'protocol_document_sha256': bundle._file_sha256(DOC),
    'source_sha256': {name: bundle._file_sha256(Path(__file__).parent / name) for name in
                      (Path(__file__).name, 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'scope': 'pinned reference atlas cache, not a model, learned feature or label',
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'atlas_file_sha256': bundle._file_sha256(OUT / 'atlas_float32.npy'),
    'protocol_sha256': bundle._file_sha256(OUT / 'protocol.json'),
    'atlas_array_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'allen_v6_float32_cache_frozen',
                  'file_bytes': (OUT / 'atlas_float32.npy').stat().st_size}), flush=True)
