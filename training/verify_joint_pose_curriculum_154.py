"""Read-only audit of the completed 154 TRAIN-only pose curriculum."""

import hashlib
import json
import math
from pathlib import Path


run = Path('I:/AnatomyTracker/runs/joint_pose_curriculum_154_train151_001')
source = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(name):
    with (run / name).open() as stream:
        return [json.loads(line) for line in stream]


done = json.loads((run / 'completed.json').read_text())
protocol = json.loads((run / 'protocol.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert config['protocol'] == protocol
assert done['updates'] == protocol['updates'] == 8000
assert protocol['batch_size'] == 4
assert protocol['gradient_bases'] == list(range(60))
assert protocol['inner_bases'] == list(range(60, 64))
assert sha(run / 'protocol.json') == config['protocol_sha256'] == done['protocol_sha256']
for name in ('config.json', 'inner_draws.jsonl', 'draws.jsonl',
             'training.jsonl', 'inner_scores.jsonl'):
    key = name.replace('.jsonl', '').replace('.json', '') + '_sha256'
    assert sha(run / name) == done[key]
for step, expected in done['checkpoint_sha256'].items():
    assert sha(run / f'joint_pose_step_{int(step):05d}.pt') == expected
for name, expected in config['source_sha256'].items():
    assert sha(source / name) == expected == done['source_sha256'][name]
assert sha(Path(protocol['parent_run']) / f"joint_step_{protocol['parent_step']:05d}.pt") == (
    protocol['parent_checkpoint_sha256'] == done['parent_checkpoint_sha256'])
assert all(not protocol[key] for key in ('calibrated', 'pretrained_external_weights_used',
    'real_images_used', 'dev_or_public_benchmark_used', 'final_animals_used'))
assert all(not done[key] for key in ('calibrated', 'real_images_used',
    'dev_or_public_benchmark_used', 'final_animals_used'))

draws = records('draws.jsonl')
inner_draws = records('inner_draws.jsonl')
training = records('training.jsonl')
scores = records('inner_scores.jsonl')
used = [row for row in draws if row['used']]
inner_used = [row for row in inner_draws if row['used']]
assert len(draws) == done['attempted']
assert len(used) == done['accepted_physical_sections'] == 32000
assert len(training) == 8000
assert [row['attempt'] for row in draws] == list(range(1, len(draws) + 1))
assert [row['update'] for row in training] == list(range(1, 8001))
assert all(row['accepted'] == 4 * row['update'] and row['batch_size'] == 4
           for row in training)
assert all(row['attempted'] >= row['accepted'] for row in training)
assert all(row['base_index'] in range(60) for row in draws)
assert all(row['base_index'] in range(60, 64) for row in inner_draws)
assert all(row['provenance']['split'] == 'train' for row in draws + inner_draws)
assert len(inner_used) == 64
assert {(row['base_index'], row['slot']) for row in inner_used} == {
    (base, slot) for base in range(60, 64) for slot in range(16)}
assert {row['provenance']['base_lineage']['animal_id'] for row in draws}.isdisjoint(
    {row['provenance']['base_lineage']['animal_id'] for row in inner_draws})
for row in draws:
    step = row['update']
    stage = 1 if step <= 2000 else 2 if step <= 4000 else 3
    appearance = ('v3' if stage == 1 or stage == 2 and step % 2 or
                  stage == 3 and step % 3 == 0 else 'v4')
    threshold = .12 if stage == 1 else .08 if stage == 2 else 0.
    assert row['stage'] == stage and row['appearance_version'] == appearance
    assert not row['used'] or row['valid_fraction'] >= threshold
for row in training:
    step = row['update']
    assert row['stage'] == (1 if step <= 2000 else 2 if step <= 4000 else 3)
assert [row['step'] for row in scores] == [0, 2000, 4000, 6000, 8000]
for record in scores:
    rows, metrics = record['rows'], record['metrics']
    assert len(rows) == metrics['n'] == 64
    assert {(row['base_index'], row['slot']) for row in rows} == {
        (base, slot) for base in range(60, 64) for slot in range(16)}
    for value, key in (('oracle_near', 'oracle_near_fraction'),
                       ('top16_near', 'top16_near_fraction'),
                       ('top2_near', 'top2_near_fraction'),
                       ('oracle_best_mm', 'mean_oracle_best_mm'),
                       ('direct_top1_mm', 'mean_direct_top1_mm')):
        assert math.isclose(sum(row[value] for row in rows) / 64,
                            metrics[key], abs_tol=1e-8)
    assert math.isclose(metrics['mean_oracle_best_mm'] +
        metrics['mean_direct_top1_mm'], metrics['selection_score_mm'], abs_tol=1e-8)
best = min(scores, key=lambda item: item['metrics']['selection_score_mm'])
assert done['selected_step'] == best['step']
assert math.isclose(done['selected_inner_score_mm'],
                    best['metrics']['selection_score_mm'], abs_tol=1e-8)

print(json.dumps({'verified': True, 'run': str(run), 'attempted': len(draws),
    'accepted': len(used), 'inner_sections': len(inner_used),
    'selected_step': best['step'],
    'scores': {str(row['step']): row['metrics'] for row in scores},
    'source_and_output_hashes_verified': True,
    'biologically_independent_validation': False}, indent=2))
