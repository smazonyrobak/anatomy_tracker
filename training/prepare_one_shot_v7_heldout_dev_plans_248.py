"""Freeze eight deterministic v7 DEV deformation plans separately from atlas rendering."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

OUT = ROOT / 'data/one_shot_v7_heldout_dev_plans_248_001'
CONTEXT = ROOT / 'data/joint_v6_coherent_subject_plans_002/context.json'
DOC = Path('docs/publication/ONE_SHOT_V7_HELDOUT_DEV_PLANS_248_PROTOCOL_20261012.md')
SEED, FIRST, SUBJECTS = 2026101249, 2000, 8
CONTEXT_SHA = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
context_bytes = CONTEXT.read_bytes()
context = json.loads(context_bytes)
assert context['v2_context_sha256'] == CONTEXT_SHA
lower = np.asarray(context['full_ccf_lower_ap_dv_ml_um'], dtype=np.float64)
upper = np.asarray(context['full_ccf_upper_ap_dv_ml_um'], dtype=np.float64)
source = {name: bundle._file_sha256(Path(__file__).parent / name) for name in (
    Path(__file__).name, 'arbitrary_plane_subject_deformation_v2.py',
    'subject_deformed_slab_multiresolution_bundle_v2.py')}
protocol = {'split': 'development', 'root_seed': SEED,
            'animal_indices': list(range(FIRST, FIRST + SUBJECTS)),
            'context_sha256': hashlib.sha256(context_bytes).hexdigest(),
            'ccf_context_sha256': CONTEXT_SHA,
            'protocol_document_sha256': bundle._file_sha256(DOC),
            'source_sha256': source, 'numpy': np.__version__,
            'scope': 'synthetic DEV deformation plans from one Allen atlas; no acquired animals, CAL/FINAL or public benchmark'}
OUT.mkdir(parents=True, exist_ok=True)
protocol_path = OUT / 'protocol.json'
protocol_text = json.dumps(protocol, indent=2)
if protocol_path.exists():
    assert protocol_path.read_text(encoding='utf8') == protocol_text
else:
    protocol_path.write_text(protocol_text, encoding='utf8')

records = []
for index in protocol['animal_indices']:
    directory = OUT / f'subject_{index:08d}'
    done = directory / 'completed.json'
    if done.exists():
        row = json.loads(done.read_text(encoding='utf8'))
        assert row['animal_index'] == index
        assert all(bundle._file_sha256(directory / name) == digest
                   for name, digest in row['artifact_sha256'].items())
        records.append(row)
        continue
    assert not directory.exists()
    prefix = f'one-shot-v7-heldout-dev-248-subject-{index:08d}'
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=SEED, split='development', animal_index=index,
        animal_id=prefix, ccf_context_sha256=CONTEXT_SHA)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt['receipt_sha256'] == plan['receipt_sha256']
    directory.mkdir()
    files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    row = {'animal_index': index, 'animal_id': prefix,
           'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
           'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
           'subject_plan_receipt_sha256': plan['receipt_sha256'],
           'plan_files': files,
           'artifact_sha256': {name: bundle._file_sha256(directory / name)
                               for name in (*files.values(), 'subject_plan_receipt.json')}}
    done.write_text(json.dumps(row, indent=2), encoding='utf8')
    records.append(row)
    print(json.dumps({'event': 'subject_frozen', 'animal_index': index,
                      'count': len(records), 'total': SUBJECTS}), flush=True)
    del plan

assert len(records) == SUBJECTS
assert len({row['subject_deformation_plan_id'] for row in records}) == SUBJECTS
assert len({row['subject_deformation_realization_id'] for row in records}) == SUBJECTS
assert all(bundle._file_sha256(Path(__file__).parent / name) == digest
           for name, digest in source.items())
assert bundle._file_sha256(DOC) == protocol['protocol_document_sha256']
(OUT / 'completed.json').write_text(json.dumps({
    'subjects': records, 'protocol_sha256': bundle._file_sha256(protocol_path),
    'subject_completed_sha256': {str(row['animal_index']): bundle._file_sha256(
        OUT / f"subject_{row['animal_index']:08d}" / 'completed.json') for row in records},
    'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({'event': 'plans_frozen', 'subjects': SUBJECTS}), flush=True)
