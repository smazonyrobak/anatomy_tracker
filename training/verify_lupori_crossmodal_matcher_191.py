"""Read-only provenance and guard arithmetic audit for the 191 pilot."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/lupori_crossmodal_matcher_191_inner_AL1A_retry1'
guard = root / 'runs/lupori_crossmodal_matcher_191_guard_AL1A'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
draws = [json.loads(line) for line in (run / 'draws.jsonl').read_text().splitlines()]
training = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
assert done['updates'] == config['updates'] == len(draws) == len(training) == 600
assert not done['guard_animal_used_in_training']
assert done['gradient_animals'] == config['train_animals'] == ['CC2B', 'CC4B', 'CC4A']
assert done['gradient_sections'] == config['train_sections'] == 188
assert done['config_sha256'] == sha(run / 'config.json')
assert done['draws_sha256'] == sha(run / 'draws.jsonl')
assert done['training_sha256'] == sha(run / 'training.jsonl')
assert done['protocol_sha256'] == config['protocol_sha256'] == sha(source.parent /
    'docs/publication/LUPORI_CROSSMODAL_MATCHER_191_INNER_PROTOCOL_20261011.md')
assert config['source_sha256'] == done['source_sha256']
assert {name: sha(source / name) for name in done['source_sha256']} == done['source_sha256']
assert config['parent_checkpoint_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt')
assert config['parent_completion_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/completed.json')
assert {str(step): sha(run / f'matcher_step_{step:05d}.pt') for step in (0, 300, 600)
        } == done['checkpoint_sha256']
assert not done['calibrated'] and not done['public_benchmark_used']
assert not done['external_pretrained_weights_used']
assert [row['update'] for row in draws] == [row['update'] for row in training] == list(range(1, 601))
assert all(row['animal_id'] in config['train_animals'] for row in draws)
assert 'AL1A' not in {row['animal_id'] for row in draws}
assert len({(row['animal_id'], row['section_id']) for row in draws}) == 188
assert Counter(row['epoch'] for row in draws) == {0: 188, 1: 188, 2: 188, 3: 36}
assert all(len(row['branch_ids']) == len(set(row['branch_ids'])) ==
    len(row['five_point_mm']) == 4 for row in draws)
assert all(math.isfinite(row[arm]['loss']) and math.isfinite(row[arm]['gradient_norm'])
    for row in training for arm in ('full', 'support_only'))
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    records = {}
    for animal in config['train_animals']:
        visualign = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        for row in visualign['slices']:
            section = Path(row['filename']).name.removesuffix('-thumb.png')
            records[(animal, section)] = f'{animal}/{row["filename"]}'
    for animal, section in {(row['animal_id'], row['section_id']) for row in draws}:
        digest = hashlib.sha256(archive.read(records[(animal, section)])).hexdigest()
        assert all(row['source_thumbnail_sha256'] == digest for row in draws
                   if (row['animal_id'], row['section_id']) == (animal, section))

guard_done = json.loads((guard / 'completed.json').read_text())
guard_config = json.loads((guard / 'config.json').read_text())
guard_rows = [json.loads(line) for line in (guard / 'rows.jsonl').read_text().splitlines()]
summary = json.loads((guard / 'summary.json').read_text())
assert len(guard_rows) == summary['sections'] == guard_done['sections'] == 52
assert guard_config['source_sha256'] == guard_done['source_sha256'] == sha(
    source / 'evaluate_lupori_crossmodal_matcher_191.py')
assert guard_config['train_completion_sha256'] == guard_done['train_completion_sha256'] == sha(
    run / 'completed.json')
assert guard_config['checkpoint_sha256'] == done['checkpoint_sha256']['600']
assert {name: sha(guard / name) for name in guard_done['output_sha256']
        } == guard_done['output_sha256']
assert all(row['animal_id'] == 'AL1A' for row in guard_rows)
assert len({row['section_id'] for row in guard_rows}) == 52
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for row in guard_rows:
        assert hashlib.sha256(archive.read(row['source_thumbnail'])).hexdigest() == (
            row['source_thumbnail_sha256'])
        assert len(row['branch_ids']) == len(row['five_point_mm']) == 16
        assert len(set(row['branch_ids'])) == 16
        assert row['choice_full'] == int(np.argmax(row['score_full']))
        assert row['choice_support_only'] == int(np.argmax(row['score_support_only']))
        assert row['prior_selected_mm'] == row['five_point_mm'][0]
        assert row['oracle_16_mm'] == min(row['five_point_mm'])
        assert row['full_selected_mm'] == row['five_point_mm'][row['choice_full']]
        assert row['support_only_selected_mm'] == row['five_point_mm'][row['choice_support_only']]
assert summary['oracle_16_le_1p5'] == sum(row['oracle_16_mm'] <= 1.5 for row in guard_rows)
for name in ('prior', 'full', 'support_only'):
    values = np.array([row[f'{name}_selected_mm'] for row in guard_rows])
    assert summary['methods'][name] == {'mean_mm': float(values.mean()),
        'median_mm': float(np.median(values)), 'le_1p5': int((values <= 1.5).sum()),
        'gt_5': int((values > 5).sum())}
p, f, s = (summary['methods'][name] for name in ('prior', 'full', 'support_only'))
gate = p['mean_mm'] - f['mean_mm'] >= .15 and s['mean_mm'] - f['mean_mm'] >= .15 and (
    f['le_1p5'] >= p['le_1p5'] and f['gt_5'] <= p['gt_5'])
assert summary['predeclared_gate'] == {'passed': gate}
print(json.dumps({'verified': True, 'train_updates': len(training), 'guard_sections': len(guard_rows),
    'methods': summary['methods'], 'gate': gate, 'train_completion_sha256': sha(run / 'completed.json'),
    'guard_completion_sha256': sha(guard / 'completed.json')}))
