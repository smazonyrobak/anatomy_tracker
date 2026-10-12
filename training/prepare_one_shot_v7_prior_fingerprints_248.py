"""Freeze local-deformation fingerprints for all pre-248 synthetic plans."""
import hashlib
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

from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

OUT = ROOT / 'data/one_shot_v7_prior_fingerprints_248_001'
DOC = Path('docs/publication/ONE_SHOT_V7_PRIOR_FINGERPRINTS_248_PROTOCOL_20261012.md')
OLD = ROOT / 'data/joint_v6_coherent_subject_plans_002'
TRAIN_MAPS = ROOT / 'data/joint_v7_independent_subject_maps_001'
PRIOR_DEV = tuple(ROOT / 'data' / name for name in (
    'one_shot_fresh_synthetic_dev_plans_001',
    'pose_feedback_037_fresh_synthetic_dev_plans_001',
    'pose_feedback_061_fresh_synthetic_dev_plans_001',
    'contextual_match_099_fresh_synthetic_dev_plans_001',
    'contextual_localization_102_fresh_dev_plans_001',
    'cross_candidate_match_103_fresh_dev_plans_001',
    'v3_pose_capture_confirmation_plans_001',
    'matcher_150_fresh_dev_plans'))

roots = (OLD, *PRIOR_DEV, TRAIN_MAPS)
input_sha = {root.name: bundle._file_sha256(root / 'completed.json') for root in roots}
fingerprints = []
for root in roots:
    completed = json.loads((root / 'completed.json').read_text())
    rows = completed['new_train_subjects'] if root == TRAIN_MAPS else completed['subjects']
    assert len(rows) == (56 if root == TRAIN_MAPS else 12 if root == OLD else 8)
    for ordinal, row in enumerate(rows):
        directory = root / row['directory']
        if root == TRAIN_MAPS:
            assert bundle._file_sha256(directory / 'completed.json') == row['completed_sha256']
        assert all(bundle._file_sha256(directory / name) == digest
                   for name, digest in row['artifact_sha256'].items())
        plan = bundle._read_raw_artifact(directory, row['plan_files'])
        assert plan['subject_deformation_plan_id'] == row['subject_deformation_plan_id']
        assert plan['subject_deformation_realization_id'] == row['subject_deformation_realization_id']
        assert plan['receipt_sha256'] == row['subject_plan_receipt_sha256']
        fields = (plan['state']['accepted_coarse_coefficients_um'],
                  plan['state']['accepted_fine_coefficients_um'])
        local_sha = hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest()
        if root == TRAIN_MAPS:
            assert local_sha == row['local_field_sha256'] and row['split'] == 'train'
        fingerprints.append({
            'source': root.name, 'source_ordinal': ordinal, 'directory': row['directory'],
            'subject_deformation_plan_id': row['subject_deformation_plan_id'],
            'subject_deformation_realization_id': row['subject_deformation_realization_id'],
            'subject_plan_receipt_sha256': row['subject_plan_receipt_sha256'],
            'local_field_sha256': local_sha,
        })
        del plan
    print(json.dumps({'source': root.name, 'rows': len(rows)}), flush=True)

assert len(fingerprints) == 132
for key in ('subject_deformation_plan_id', 'subject_deformation_realization_id',
            'subject_plan_receipt_sha256', 'local_field_sha256'):
    assert len({row[key] for row in fingerprints}) == len(fingerprints)
OUT.mkdir(parents=True, exist_ok=False)
protocol = {'sources': input_sha, 'subjects': len(fingerprints),
            'protocol_document_sha256': bundle._file_sha256(DOC),
            'source_sha256': {name: bundle._file_sha256(Path(__file__).parent / name) for name in
                              (Path(__file__).name, 'subject_deformed_slab_multiresolution_bundle_v2.py')},
            'scope': 'historical synthetic deformation identity audit; no acquired animals or model data'}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
(OUT / 'fingerprints.json').write_text(json.dumps(fingerprints, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'subjects': len(fingerprints), 'sources': input_sha,
    'protocol_sha256': bundle._file_sha256(OUT / 'protocol.json'),
    'fingerprints_sha256': bundle._file_sha256(OUT / 'fingerprints.json'),
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'prior_fingerprints_frozen', 'subjects': len(fingerprints)}), flush=True)
