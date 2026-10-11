"""Read-only provenance and guard audit for real-field map adaptation."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/lupori_local_map_193_inner_AL1A'
guard = root / 'runs/lupori_local_map_193_guard_AL1A'


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
assert done['synthetic_presentations'] == 600
assert done['config_sha256'] == sha(run / 'config.json')
assert done['draws_sha256'] == sha(run / 'draws.jsonl')
assert done['training_sha256'] == sha(run / 'training.jsonl')
assert done['protocol_sha256'] == config['protocol_sha256'] == sha(source.parent /
    'docs/publication/LUPORI_LOCAL_MAP_ADAPTATION_193_PROTOCOL_20261011.md')
assert config['source_sha256'] == done['source_sha256']
assert {name: sha(source / name) for name in done['source_sha256']} == done['source_sha256']
assert config['parent_checkpoint_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt')
assert config['parent_completion_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/completed.json')
assert {str(step): sha(run / f'map_step_{step:05d}.pt') for step in (0, 300, 600)
        } == done['checkpoint_sha256']
assert not done['calibrated'] and not done['public_benchmark_used']
assert not done['external_pretrained_weights_used']
assert [row['update'] for row in draws] == [row['update'] for row in training] == list(range(1, 601))
assert all(row['real']['animal_id'] in config['train_animals'] for row in draws)
assert 'AL1A' not in {row['real']['animal_id'] for row in draws}
assert len({(row['real']['animal_id'], row['real']['section_id']) for row in draws}) == 188
assert len({row['synthetic']['physical_section_id'] for row in draws}) == 600
assert Counter(row['epoch'] for row in draws) == {0: 188, 1: 188, 2: 188, 3: 36}
assert all(math.isfinite(row[key]) for row in training for key in
    ('real_map_mm', 'synthetic_map_mm', 'loss', 'gradient_norm'))
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    metadata = {}
    for animal in config['train_animals']:
        visualign = json.loads(archive.read(f'{animal}/{animal}-visualign.json'))
        for row in visualign['slices']:
            section = Path(row['filename']).name.removesuffix('-thumb.png')
            metadata[(animal, section)] = f'{animal}/{row["filename"]}'
    for animal, section in {(row['real']['animal_id'], row['real']['section_id'])
                            for row in draws}:
        observed = [row['real'] for row in draws if
                    (row['real']['animal_id'], row['real']['section_id']) == (animal, section)]
        names = {'source_thumbnail_sha256': metadata[(animal, section)],
            'displacement_x_sha256': f'{animal}/dispFields/{section}-dispFieldX.csv',
            'displacement_y_sha256': f'{animal}/dispFields/{section}-dispFieldY.csv',
            'mask_sha256': f'{animal}/masks/{section}-mask.png'}
        for key, name in names.items():
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            assert all(row[key] == digest for row in observed)

guard_done = json.loads((guard / 'completed.json').read_text())
guard_config = json.loads((guard / 'config.json').read_text())
guard_rows = [json.loads(line) for line in (guard / 'rows.jsonl').read_text().splitlines()]
summary = json.loads((guard / 'summary.json').read_text())
assert len(guard_rows) == summary['sections'] == guard_done['sections'] == 52
assert guard_config['source_sha256'] == guard_done['source_sha256'] == sha(
    source / 'evaluate_lupori_local_map_193.py')
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
        section = row['section_id']
        for suffix, key in (('X', 'displacement_x_sha256'), ('Y', 'displacement_y_sha256')):
            assert hashlib.sha256(archive.read(f'AL1A/dispFields/{section}-dispField{suffix}.csv')
                ).hexdigest() == row[key]
        assert hashlib.sha256(archive.read(f'AL1A/masks/{section}-mask.png')
            ).hexdigest() == row['mask_sha256']
        assert row['tissue_sites_96'] > 0
        assert all(np.isfinite(row[key]) and row[key] >= 0 for key in (
            'zero_local_rms_mm', 'parent_local_rms_mm', 'adapted_local_rms_mm'))
for name in ('zero', 'parent', 'adapted'):
    values = np.array([row[f'{name}_local_rms_mm'] for row in guard_rows])
    assert summary['rms_mm'][name] == {'mean': float(values.mean()),
        'median': float(np.median(values)), 'p90': float(np.quantile(values, .9))}
zero, parent, adapted = (summary['rms_mm'][name]['mean']
    for name in ('zero', 'parent', 'adapted'))
worse = sum(row['adapted_local_rms_mm'] - row['zero_local_rms_mm'] > .1
            for row in guard_rows)
gate = adapted <= .9 * zero and adapted <= .9 * parent and worse <= 4
assert summary['predeclared_gate'] == {'passed': gate, 'worse_than_zero_by_gt_0p1mm': worse}
print(json.dumps({'verified': True, 'train_updates': len(training), 'guard_sections': len(guard_rows),
    'guard_rms_mm': summary['rms_mm'], 'gate': gate,
    'train_completion_sha256': sha(run / 'completed.json'),
    'guard_completion_sha256': sha(guard / 'completed.json')}))
