"""Read-only provenance and arithmetic audit after the 155 fitter run exits."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


root = Path('I:/AnatomyTracker')
run = root / 'runs/joint_fitter_155_train151_001'
source = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
protocol = json.loads((run / 'protocol.json').read_text())
config = json.loads((run / 'config.json').read_text())
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
inner_draws = [json.loads(line) for line in (run / 'inner_draws.jsonl').open()]
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
scores = [json.loads(line) for line in (run / 'inner_scores.jsonl').open()]
for name in ('protocol', 'config', 'draws', 'inner_draws', 'training', 'inner_scores'):
    suffix = 'json' if name in ('protocol', 'config') else 'jsonl'
    assert sha(run / f'{name}.{suffix}') == done[f'{name}_sha256']
assert sha(root / 'runs/joint_pose_curriculum_154_train151_001' /
    f"joint_pose_step_{protocol['parent_step']:05d}.pt") == \
    protocol['parent_checkpoint_sha256'] == done['parent_checkpoint_sha256']
assert config['protocol'] == protocol and config['protocol_sha256'] == done['protocol_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(source / name) == expected == done['source_sha256'][name]
for step, expected in done['checkpoint_sha256'].items():
    assert sha(run / f'joint_fitter_step_{int(step):05d}.pt') == expected
assert len(training) == done['updates'] == done['accepted_physical_sections'] == 6000
assert len(draws) == done['attempted']
accepted = [row for row in draws if row['used']]
assert len(accepted) == len(training)
assert [row['update'] for row in accepted] == list(range(1, 6001))
assert [row['update'] for row in training] == list(range(1, 6001))
assert all(row['base_index'] < 60 and row['provenance']['split'] == 'train'
           for row in draws)
assert all(row['base_index'] in range(60, 64) and row['provenance']['split'] == 'train'
           for row in inner_draws)
assert len([row for row in inner_draws if row['used']]) == 48
assert all(row['physical_section_id'] == draw['provenance']['physical_section_id']
           for row, draw in zip(training, accepted))
assert len({row['physical_section_id'] for row in training}) == 6000
assert len(scores) == 3 and [row['step'] for row in scores] == [2000, 4000, 6000]
for item in scores:
    metrics, rows = item['metrics'], item['rows']
    assert len(rows) == 48 and {row['base_index'] for row in rows} == set(range(60, 64))
    for key, values in (
        ('exact_drift_mm', [row['corrected_mm'][0] for row in rows]),
        ('near_initial_mm', [row['initial_mm'][1] for row in rows]),
        ('near_corrected_mm', [row['corrected_mm'][1] for row in rows]),
        ('near_dense_mm', [row['dense_mm'][1] for row in rows]),
        ('exact_energy_below_wrong', [row['exact_energy_below_wrong'] for row in rows]),
    ):
        assert math.isclose(metrics[key], float(np.mean(values)), abs_tol=1e-8)
    assert math.isclose(metrics['near_gain_mm'],
        metrics['near_initial_mm'] - metrics['near_corrected_mm'], abs_tol=1e-8)
    score = metrics['exact_drift_mm'] + metrics['near_corrected_mm'] + \
        .5 * metrics['near_dense_mm'] + 1 - metrics['exact_energy_below_wrong']
    assert math.isclose(metrics['selection_score'], score, abs_tol=1e-8)
    assert metrics['feedback_gate_passed'] == (
        metrics['exact_drift_mm'] <= .25 and metrics['near_gain_mm'] >= .10
        and metrics['exact_energy_below_wrong'] >= .70)
eligible = [row for row in scores if row['metrics']['feedback_gate_passed']]
selected = min(eligible or scores, key=lambda row: row['metrics']['selection_score'])
assert done['selected_step'] == selected['step']
assert done['feedback_gate_passed'] == bool(eligible)
assert done['selected_inner_metrics'] == selected['metrics']
assert not any(done[key] or protocol[key] for key in
    ('calibrated', 'external_pretrained_weights_used', 'real_images_used',
     'dev_or_public_benchmark_used', 'final_animals_used'))
print(json.dumps({'verified': True, 'updates': len(training),
                  'selected_step': done['selected_step'],
                  'feedback_gate_passed': done['feedback_gate_passed'],
                  'selected_metrics': done['selected_inner_metrics']}, indent=2))
