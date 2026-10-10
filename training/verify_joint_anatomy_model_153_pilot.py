"""Read-only independent audit of the frozen 153 TRAIN-only pilot."""

import hashlib
import json
import math
from pathlib import Path


run = Path('I:/AnatomyTracker/runs/joint_anatomy_model_153_train151_002')
source = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lines(name):
    with (run / name).open() as stream:
        return [json.loads(line) for line in stream]


completed = json.loads((run / 'completed.json').read_text())
protocol = json.loads((run / 'protocol.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert config['protocol'] == protocol
for name in ('protocol.json', 'config.json', 'inner_draws.jsonl', 'draws.jsonl',
             'training.jsonl', 'inner_scores.jsonl'):
    assert digest(run / name) == completed[name.replace('.jsonl', '').replace('.json', '') + '_sha256']
for step, expected in completed['checkpoint_sha256'].items():
    assert digest(run / f'joint_step_{int(step):05d}.pt') == expected
for name, expected in completed['source_sha256'].items():
    assert digest(source / name) == expected == config['source_sha256'][name]
assert all(not completed[key] and not protocol[key] for key in
           ('calibrated', 'pretrained_weights_used', 'real_images_used',
            'dev_or_public_benchmark_used', 'final_animals_used'))

draws = lines('draws.jsonl')
inner_draws = lines('inner_draws.jsonl')
training = lines('training.jsonl')
scores = lines('inner_scores.jsonl')
used = [row for row in draws if row['used']]
inner_used = [row for row in inner_draws if row['used']]
assert len(draws) == completed['attempted'] == 2896
assert len(training) == len(used) == completed['accepted_physical_sections'] == 1800
assert [row['update'] for row in training] == list(range(1, 1801))
assert [row['update'] for row in used] == list(range(1, 1801))
assert [row['attempt'] for row in draws] == list(range(1, 2897))
assert len(inner_used) == 16
assert {(row['base_index'], row['slot']) for row in inner_used} == {
    (base, slot) for base in range(60, 64) for slot in range(4)}
assert {row['base_index'] for row in draws} <= set(range(60))
assert {row['base_index'] for row in inner_draws} <= set(range(60, 64))
assert all(row['provenance']['split'] == 'train' for row in draws + inner_draws)
assert all(row['appearance_version'] == ('v3' if row['update'] % 3 == 0 else 'v4')
           for row in used)
assert all(a['physical_section_id'] == b['provenance']['physical_section_id']
           and a['base_index'] == b['base_index'] and a['attempted'] == b['attempt']
           for a, b in zip(training, used))
train_animals = {row['provenance']['base_lineage']['animal_id'] for row in draws}
inner_animals = {row['provenance']['base_lineage']['animal_id'] for row in inner_draws}
assert train_animals.isdisjoint(inner_animals)
assert len(train_animals) == 60 and len(inner_animals) == 4

assert [row['step'] for row in scores] == [800, 1300, 1550, 1800]
for record in scores:
    rows, metrics = record['rows'], record['metrics']
    assert len(rows) == 16
    for key in ('blind_top2_has_near', 'direct_top1_mm',
                'selected_corrected_rigid_mm', 'selected_dense_mm'):
        assert math.isclose(sum(row[key] for row in rows) / 16, metrics[key], abs_tol=1e-8)
    assert math.isclose(sum(metrics[key] for key in
        ('direct_top1_mm', 'selected_corrected_rigid_mm', 'selected_dense_mm')),
        metrics['selection_score_mm'], abs_tol=1e-8)
gate = scores[1]['metrics']
assert math.isclose(gate['near_initial_mm'] - gate['near_corrected_mm'],
                    gate['near_gain_mm'], abs_tol=1e-8)
assert gate['feedback_gate_passed'] == (
    gate['near_gain_mm'] >= protocol['stage_c_requires_inner_gate']['near_rigid_gain_mm_at_least']
    and gate['exact_drift_mm'] <=
    protocol['stage_c_requires_inner_gate']['exact_rigid_drift_mm_at_most'])
assert completed['feedback_gate_passed'] == gate['feedback_gate_passed'] is False
assert all(row['stage'] == ('A' if row['update'] <= 800 else 'B') for row in training)
best = min(scores, key=lambda row: row['metrics']['selection_score_mm'])
assert completed['selected_step'] == best['step'] == 1300
assert math.isclose(completed['selected_inner_score_mm'],
                    best['metrics']['selection_score_mm'], abs_tol=1e-8)

print(json.dumps({
    'verified': True, 'run': str(run), 'attempted': len(draws), 'accepted': len(used),
    'train_animals': len(train_animals), 'inner_animals': len(inner_animals),
    'accepted_unique_physical_ids': len({row['physical_section_id'] for row in training}),
    'selected_step': best['step'], 'selected_metrics': best['metrics'],
    'feedback_gate_passed': gate['feedback_gate_passed'],
    'source_and_output_hashes_verified': True,
}, indent=2))
