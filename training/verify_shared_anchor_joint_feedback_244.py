"""Independent post-exit receipt, source, split and row audit for run 244."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/shared_anchor_joint_feedback_244_train_001'
readout = root / 'runs/shared_anchor_joint_feedback_244_readout_001'
source = Path('training')
protocol = Path('docs/publication/SHARED_ANCHOR_JOINT_FEEDBACK_244_PROTOCOL_20261012.md')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((run / 'config.json').read_text())
done = json.loads((run / 'completed.json').read_text())
assert config['initialization'].startswith('all model weights random') and 'parent' not in config
assert not config['calibrated'] and not config['public_benchmark_used']
assert not done['calibrated'] and not done['public_benchmark_used']
assert done['batches'] == config['batches'] == 6000
assert done['accepted_synthetic'] == 12000 and done['distinct_real_train'] == 6000
assert sha(protocol) == config['protocol_sha256']
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'real_schedule.npy') == config['real_schedule_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(source / name) == expected, name
for step, expected in done['checkpoint_sha256'].items():
    assert sha(run / f'joint_step_{int(step):05d}.pt') == expected
schedule = np.load(run / 'real_schedule.npy', allow_pickle=False)
assert schedule.shape == (6000, 2) and len({tuple(row) for row in schedule}) == 6000

training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert len(training) == 6000 and [row['batch'] for row in training] == list(range(1, 6001))
assert all(row['synthetic_presentations'] == 2 * row['batch']
           and row['real_presentations'] == row['batch'] for row in training)
assert all(all(math.isfinite(value) for key, value in row.items()
               if key not in ('batch', 'synthetic_presentations', 'real_presentations'))
           for row in training)
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
synthetic = [row for row in draws if row['slot'] < 2 and row['used']]
real = [row for row in draws if row['slot'] == 2 and row['used']]
assert len(synthetic) == 12000 and len(real) == 6000
assert len({row['physical_section_id'] for row in synthetic}) == len(synthetic)
assert len({row['section_id'] for row in real}) == len(real)
assert all(row['label_role'] == 'weak Allen affine' for row in real)
assert all('one_shot_slide_artifacts_v7' in row and
           not row['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair']
           for row in synthetic)

dev = [row for row in map(json.loads,
       (root / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl').open())
       if row['training_split'] == 'development']
assert not {row['animal_id'] for row in real} & {row['animal_id'] for row in dev}

report = json.loads((readout / 'completed.json').read_text())
assert report['steps'] == [0, 1000, 3000, 6000] and report['rows'] == 176
assert not report['calibrated'] and not report['public_benchmark_used']
assert report['checkpoint_sha256'] == done['checkpoint_sha256']
assert sha(readout / 'rows.jsonl') == report['rows_sha256']
assert sha(readout / 'summary.json') == report['summary_sha256']
for name, expected in report['source_sha256'].items():
    assert sha(source / name) == expected, name
rows = [json.loads(line) for line in (readout / 'rows.jsonl').open()]
assert len(rows) == 176
groups = defaultdict(list)
for row in rows:
    groups[(row['step'], row['set'], row['animal_id'])].append(row)
    assert all(math.isfinite(value) for key, value in row.items() if key.endswith('_um'))
for item in json.loads((readout / 'summary.json').read_text()):
    step, split = item['step'], item['set']
    donors = [group for (s, name, _), group in groups.items() if s == step and name == split]
    assert len(donors) == item['identities']
    assert sum(map(len, donors)) == item['rows']
    for metric, recorded in item['identity_equal_mean_um'].items():
        actual = np.mean([np.mean([row[metric] for row in group]) for group in donors])
        assert math.isclose(actual, recorded, abs_tol=1e-6)
print(json.dumps({'verified': True, 'training_batches': len(training),
                  'synthetic_draws': len(synthetic), 'real_train_sections': len(real),
                  'readout_rows': len(rows)}))
