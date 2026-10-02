"""Independent frozen-output and metric audit for global patch capture 030."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

bank = root / 'runs/atlas_global_patch_030'
result = root / 'runs/atlas_global_patch_030_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
with (bank / 'completed.json').open() as stream:
    bank_receipt = json.load(stream)
with (result / 'completed.json').open() as stream:
    result_receipt = json.load(stream)
for folder, receipt in ((bank, bank_receipt), (result, result_receipt)):
    for name, expected in receipt['files_sha256'].items():
        with (folder / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
with (result / 'config.json').open() as stream:
    config = json.load(stream)
with (result / 'summary.json').open() as stream:
    summary = json.load(stream)
assert config['bank_completed_sha256'] == hashlib.sha256((bank / 'completed.json').read_bytes()).hexdigest()
assert config['panel_records_sha256'] == hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest()
for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == expected
positions = np.load(bank / 'positions_um.npy', allow_pickle=False)
records = {record['sha256']: record for record in map(json.loads, (panel / 'records.jsonl').open())
           if record['eligible']}
rows = list(map(json.loads, (result / 'rows.jsonl').open()))
assert len(rows) == result_receipt['rows'] == summary['sections'] == len(records) == 185
assert len({row['sha256'] for row in rows}) == len(rows)

for row in rows:
    record = records[row['sha256']]
    assert all(row[key] == record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
               'section_id', 'synthetic_subject_plan_id', 'appearance_mode'))
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        xy = np.asarray(row['query_pixel_yx'])
        truth = arrays['target_centre_um'][xy[:, 0], xy[:, 1]]
        assert arrays['valid_mask'][xy[:, 0], xy[:, 1]].all()
    assert np.allclose(truth, row['true_ccf_um'], atol=1e-4)
    indices = np.asarray(row['top16_bank_index'])
    assert indices.shape == (32, 16) and np.all((indices >= 0) & (indices < bank_receipt['atlas_patches']))
    distances = np.linalg.norm(positions[indices // 512] - truth[:, None], axis=-1)
    assert np.allclose(distances, row['top16_distance_um'], atol=1e-3)
    selected = np.flatnonzero(distances.min(axis=1) < 500)
    separation = np.linalg.norm(truth[selected, None] - truth[None, selected], axis=-1)
    triple = any(separation[a, b] > 1000 and np.any((separation[a, b + 1:] > 1000) &
        (separation[b, b + 1:] > 1000)) for a in range(len(selected))
        for b in range(a + 1, len(selected)))
    assert triple == row['three_separated_matches']

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
point_recall = np.mean([np.mean([np.mean(np.min(row['top16_distance_um'], axis=1) < 500)
    for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
triples = np.mean([np.mean([row['three_separated_matches']
    for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
top1 = np.mean([np.mean([np.mean(np.asarray(row['top16_distance_um'])[:, 0] < 500)
    for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
assert abs(point_recall - summary['point_recall16_500um']) < 1e-12
assert abs(triples - summary['three_separated_section_fraction']) < 1e-12
assert abs(top1 - summary['point_recall1_500um']) < 1e-12
for mode, reported in summary['mode_point_recall16_500um'].items():
    observed = np.mean([np.mean([np.mean(np.min(row['top16_distance_um'], axis=1) < 500)
        for row in rows if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in rows)])
    assert abs(observed - reported) < 1e-12
gate = point_recall >= .10 and triples >= .20 and min(summary['mode_point_recall16_500um'].values()) >= .5 * point_recall
assert gate == summary['necessary_gate']
print(json.dumps({'verification': 'PASS', 'sections': len(rows), 'synthetic_identities': len(subjects),
                  'point_recall16_500um': float(point_recall),
                  'three_separated_section_fraction': float(triples), 'necessary_gate': bool(gate)}))
