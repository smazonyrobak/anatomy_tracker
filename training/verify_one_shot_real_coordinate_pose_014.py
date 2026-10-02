"""Read-only provenance and raw-result audit for the frozen 014 pilot."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/one_shot_real_coordinate_pose_014'
config = json.loads((run / 'config.json').read_text())
receipt = json.loads((run / 'completed.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert receipt['updates'] == config['updates'] == 12000
for name in ('draws', 'training', 'config'):
    path = run / ('config.json' if name == 'config' else f'{name}.jsonl')
    assert sha(path) == receipt[f'{name}_sha256']
assert sha(config['parent']) == config['parent_sha256']
assert sha(run / 'real_schedule.npy') == config['real_schedule_sha256']
for path, expected in config['excluded_real_schedule_sha256'].items():
    assert sha(path) == expected
for path, expected in config['real_bindings'].items():
    assert sha(path) == expected
for name, expected in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == expected
schedule = np.load(run / 'real_schedule.npy')
prior = set().union(*(set(map(tuple, np.load(path)))
                      for path in config['excluded_real_schedule_sha256']))
assert schedule.shape == (24000, 2) and len(set(map(tuple, schedule))) == 24000
assert not set(map(tuple, schedule)) & prior
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
accepted_synthetic = [row for row in draws if row['slot'] < 2 and row['used']]
accepted_real = [row for row in draws if row['slot'] >= 2 and row['used']]
assert len(accepted_synthetic) == receipt['accepted_synthetic'] == 24000
assert len(accepted_real) == receipt['distinct_real_train'] == 24000
assert len({(r['animal_id'], r['section_id']) for r in accepted_real}) == 24000
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert [row['step'] for row in training] == list(range(1, 12001))

reports = []
for folder in ('one_shot_real_coordinate_pose_014_development_eval',
               'one_shot_real_coordinate_pose_014_mapped_eval'):
    report = root / 'runs' / folder
    result = json.loads((report / 'completed.json').read_text())
    assert not result['calibrated'] and not result['public_benchmark_used']
    assert sha(Path(__file__).parent / ('evaluate_one_shot_real_coordinate_pose_014_mapped.py'
        if 'mapped' in folder else 'evaluate_one_shot_real_coordinate_pose_014.py')) == result['source_sha256']
    for step, expected in result['checkpoint_sha256'].items():
        assert sha(run / f'joint_step_{int(step):05d}.pt') == expected
    assert sha(root / 'data/one_shot_fresh_synthetic_dev_panel_001/records.jsonl') == result['panel_records_sha256']
    if 'real_records_sha256' in result:
        assert sha(root / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl') == result['real_records_sha256']
    assert sha(report / 'rows.jsonl') == result['rows_sha256']
    assert sha(report / 'summary.json') == result['summary_sha256']
    rows = [json.loads(line) for line in (report / 'rows.jsonl').open()]
    summary = json.loads((report / 'summary.json').read_text())
    assert len(rows) == result['rows']
    for group in summary:
        selected = [row for row in rows if row['step'] == group['step']
                    and (row.get('set') == group['set'] if 'set' in group else True)]
        animals = {row['animal_id'] for row in selected}
        assert len(selected) == group['rows'] and len(animals) == group['identities']
        for name, value in group['identity_equal_mean'].items():
            recomputed = np.mean([np.mean([row[name] for row in selected
                                           if row['animal_id'] == animal]) for animal in animals])
            assert abs(recomputed - value) < 1e-6
    reports.append({'folder': folder, 'rows': len(rows), 'summary_groups': len(summary)})
print(json.dumps({'status': 'passed', 'train_batches': len(training),
                  'accepted_synthetic': len(accepted_synthetic),
                  'distinct_real_train': len(accepted_real), 'reports': reports}))
