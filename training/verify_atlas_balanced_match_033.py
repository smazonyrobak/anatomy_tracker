"""Independent frozen-hash, donor-provenance and physical-error replay for 033."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

run = root / 'runs/atlas_balanced_match_033'
result = root / 'runs/atlas_balanced_match_033_development_eval'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
with (run / 'completed.json').open() as stream:
    training_receipt = json.load(stream)
for name in ('draws', 'training', 'config'):
    path = run / f'{name}.jsonl' if name != 'config' else run / 'config.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == training_receipt[f'{name}_sha256']
with (run / 'config.json').open() as stream:
    training_config = json.load(stream)
assert training_config['parent_checkpoint_sha256'] == hashlib.sha256(
    (root / 'runs/atlas_learned_match_032/match_step_03000.pt').read_bytes()).hexdigest()
assert training_receipt['accepted_new_synthetic_sections'] == 12000
with (result / 'completed.json').open() as stream:
    receipt = json.load(stream)
for name, expected in receipt['files_sha256'].items():
    assert hashlib.sha256((result / name).read_bytes()).hexdigest() == expected
with (result / 'config.json').open() as stream:
    config = json.load(stream)
assert config['run_completed_sha256'] == hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest()
assert config['retrieval_completed_sha256'] == hashlib.sha256((retrieval / 'completed.json').read_bytes()).hexdigest()
for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == expected
for step, expected in config['checkpoint_sha256'].items():
    assert hashlib.sha256((run / f'match_step_{int(step):05d}.pt').read_bytes()).hexdigest() == expected

records = {record['sha256']: record for record in map(json.loads, (panel / 'records.jsonl').open())
           if record['eligible']}
retrieved = {row['sha256']: row for row in map(json.loads, (retrieval / 'rows.jsonl').open())}
rows = list(map(json.loads, (result / 'rows.jsonl').open()))
with (result / 'summary.json').open() as stream:
    summary = json.load(stream)
assert len(records) == len(retrieved) == 185 and len(rows) == receipt['rows'] == 740
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
for row in rows:
    record = records[row['sha256']]
    assert all(row[key] == record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
               'section_id', 'synthetic_subject_plan_id', 'appearance_mode'))
    source = retrieved[row['sha256']]
    choices = np.asarray(row['selected_match_rank'])
    grouped = np.asarray(row['grouped_selected_match_rank'])
    distances = np.asarray(source['top16_distance_um'])
    available = distances.min(axis=1) < 500
    correct = (choices < 16) & (distances[np.arange(32), choices.clip(max=15)] < 500)
    grouped_correct = (grouped < 16) & (distances[np.arange(32), grouped.clip(max=15)] < 500)
    assert row['available_count'] == int(available.sum())
    assert row['correct_available_count'] == int((correct & available).sum())
    assert row['grouped_correct_available_count'] == int((grouped_correct & available).sum())
    assert row['false_unavailable_count'] == int(((choices < 16) & ~available).sum())
    assert row['grouped_false_unavailable_count'] == int(((grouped < 16) & ~available).sum())
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        y, x = np.nonzero(arrays['valid_mask'])
        coefficient = np.asarray(row['affine_coefficient_normalized_xy_um'])
        mapped = coefficient[0] + (x[:, None] / 256 - .5) * coefficient[1] + (y[:, None] / 256 - .5) * coefficient[2]
        assert abs(np.linalg.norm(mapped - arrays['target_centre_um'][y, x], axis=-1).mean()
                   - row['rigid_mean_error_um']) < 1e-3

for checkpoint in summary['checkpoints']:
    selected = [row for row in rows if row['step'] == checkpoint['step']]
    for prefix in ('', 'grouped_'):
        recall = np.mean([sum(row[prefix + 'correct_available_count'] for row in selected
            if row['synthetic_subject_plan_id'] == subject) /
            sum(row['available_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
            for subject in subjects])
        false_rate = np.mean([sum(row[prefix + 'false_unavailable_count'] for row in selected
            if row['synthetic_subject_plan_id'] == subject) /
            sum(row['unavailable_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
            for subject in subjects])
        suffix = '_secondary' if prefix else ''
        assert abs(recall - checkpoint[prefix + 'available_match_recall500' + suffix]) < 1e-12
        assert abs(false_rate - checkpoint[prefix + 'unavailable_false_match_rate' + suffix]) < 1e-12
    error = np.mean([np.mean([row['rigid_mean_error_um'] for row in selected
        if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
    assert abs(error - checkpoint['weighted_fit_rigid_mean_um']) < 1e-9
print(json.dumps({'verification': 'PASS', 'rows': len(rows), 'synthetic_subjects': len(subjects),
                  'any_continuation_gate': summary['any_continuation_gate'],
                  'best_weighted_fit_rigid_mean_um': min(row['weighted_fit_rigid_mean_um']
                                                          for row in summary['checkpoints'][1:])}))
