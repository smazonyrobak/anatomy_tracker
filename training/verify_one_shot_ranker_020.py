"""Read-only provenance and frozen-geometry audit for ranker probe 020."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

root = Path('I:/AnatomyTracker')
run = root / 'runs/one_shot_ranker_020'
report = root / 'runs/one_shot_ranker_020_development_eval'
config = json.loads((run / 'config.json').read_text())
receipt = json.loads((run / 'completed.json').read_text())
result = json.loads((report / 'completed.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert receipt['updates'] == config['updates'] == 6000
assert config['synthetic_per_batch'] == 2 and config['beam'] == 8
assert config['trainable'] == ['fitted_matcher'] and config['real_training_images'] == 0
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
for name in ('draws', 'training', 'config'):
    path = run / ('config.json' if name == 'config' else f'{name}.jsonl')
    assert sha(path) == receipt[f'{name}_sha256']
assert sha(config['parent']) == config['parent_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == expected
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == receipt['accepted_synthetic'] == 12000
assert {(row['step'], row['slot']) for row in used} == {
    (step, slot) for step in range(1, 6001) for slot in range(2)}
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001/records.jsonl'
development = {row['animal_id'] for row in map(json.loads, panel.open())}
assert not development & {row['animal_id'] for row in used}
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert [row['step'] for row in training] == list(range(1, 6001))
assert np.isfinite([row['loss'] for row in training]).all()
assert np.isfinite([row['gradient_norm'] for row in training]).all()
assert training[0]['gradient_norm'] > 0
parent = torch.load(config['parent'], map_location='cpu', weights_only=True)['model']
final = torch.load(run / 'joint_step_06000.pt', map_location='cpu', weights_only=True)['model']
assert all(torch.equal(value, final[name]) for name, value in parent.items()
           if not name.startswith('fitted_matcher.'))
assert any(not torch.equal(value, final[name]) for name, value in parent.items()
           if name.startswith('fitted_matcher.'))
assert not result['calibrated'] and not result['public_benchmark_used']
assert sha(Path(__file__).parent / 'evaluate_one_shot_ranker_020.py') == result['source_sha256']
assert sorted(map(int, result['checkpoint_sha256'])) == [0, 2000, 4000, 6000]
for step, expected in result['checkpoint_sha256'].items():
    assert sha(run / f'joint_step_{int(step):05d}.pt') == expected
real_records = root / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl'
assert sha(panel) == result['panel_records_sha256']
assert sha(real_records) == result['real_records_sha256']
assert sha(report / 'rows.jsonl') == result['rows_sha256']
assert sha(report / 'summary.json') == result['summary_sha256']
rows = [json.loads(line) for line in (report / 'rows.jsonl').open()]
summary = json.loads((report / 'summary.json').read_text())
assert len(rows) == result['rows'] == 4 * (185 + 64)
assert len(summary) == 8
for group in summary:
    selected = [row for row in rows if row['step'] == group['step'] and row['set'] == group['set']]
    animals = {row['animal_id'] for row in selected}
    assert len(selected) == group['rows'] and len(animals) == group['identities']
    for name, value in group['identity_equal_mean'].items():
        recomputed = np.mean([np.mean([row[name] for row in selected
                                       if row['animal_id'] == animal]) for animal in animals])
        assert np.isfinite(recomputed) and abs(recomputed - value) < 1e-6
print(json.dumps({'status': 'passed', 'train_batches': len(training),
                  'accepted_synthetic': len(used), 'development_rows': len(rows),
                  'summary_groups': len(summary)}))
