"""Independent 032 raw-output and no-match-collapse audit."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

run = root / 'runs/atlas_learned_match_032'
result = root / 'runs/atlas_learned_match_032_development_eval'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
with (run / 'completed.json').open() as stream:
    training_receipt = json.load(stream)
for name in ('draws', 'training', 'config'):
    path = run / f'{name}.jsonl' if name != 'config' else run / 'config.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == training_receipt[f'{name}_sha256']
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
retrieved = {row['sha256']: row for row in map(json.loads, (retrieval / 'rows.jsonl').open())}
records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']}
rows = list(map(json.loads, (result / 'rows.jsonl').open()))
assert len(rows) == receipt['rows'] == 740 and len(retrieved) == len(records) == 185
with (result / 'summary.json').open() as stream:
    summary = json.load(stream)

diagnostics = []
for step in (0, 1000, 2000, 3000):
    selected = [row for row in rows if row['step'] == step]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in selected})
    recall = np.mean([sum(row['correct_available_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['available_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects])
    false_rate = np.mean([sum(row['false_unavailable_count'] for row in selected
        if row['synthetic_subject_plan_id'] == subject) /
        sum(row['unavailable_count'] for row in selected if row['synthetic_subject_plan_id'] == subject)
        for subject in subjects])
    error = np.mean([np.mean([row['rigid_mean_error_um'] for row in selected
        if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
    expected = next(checkpoint for checkpoint in summary['checkpoints'] if checkpoint['step'] == step)
    assert abs(recall - expected['available_match_recall500']) < 1e-12
    assert abs(false_rate - expected['unavailable_false_match_rate']) < 1e-12
    assert abs(error - expected['weighted_fit_rigid_mean_um']) < 1e-9
    dustbin, positive_probability, correct_rank = [], [], []
    for row in selected:
        source = retrieved[row['sha256']]
        logits = np.asarray(row['match_logits'])
        probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        dustbin.extend(probability[:, 16])
        distance = np.asarray(source['top16_distance_um'])
        available = distance.min(axis=-1) < 500
        nearest = distance.argmin(axis=-1)
        positive_probability.extend(probability[np.arange(32), nearest][available])
        correct_rank.extend((logits[available, :16] > logits[np.arange(32), nearest][available, None]).sum(-1) + 1)
        with np.load(panel / records[row['sha256']]['file'], allow_pickle=False) as arrays:
            y, x = np.nonzero(arrays['valid_mask'])
            coefficients = np.asarray(row['affine_coefficient_normalized_xy_um'])
            predicted = coefficients[0] + (x[:, None] / 256 - .5) * coefficients[1] + (y[:, None] / 256 - .5) * coefficients[2]
            assert abs(np.linalg.norm(predicted - arrays['target_centre_um'][y, x], axis=-1).mean()
                       - row['rigid_mean_error_um']) < 1e-3
    diagnostics.append({'step': step, 'dustbin_probability_median': float(np.median(dustbin)),
                        'available_correct_candidate_probability_median': float(np.median(positive_probability)),
                        'correct_candidate_rank_among_16_median': float(np.median(correct_rank)),
                        'available_match_recall500': float(recall),
                        'unavailable_false_match_rate': float(false_rate),
                        'weighted_fit_rigid_mean_um': float(error)})
print(json.dumps({'verification': 'PASS', 'sections': len(retrieved), 'synthetic_subjects': len(subjects),
                  'checkpoints': diagnostics, 'any_post_initial_gate': summary['any_post_initial_gate']}))
