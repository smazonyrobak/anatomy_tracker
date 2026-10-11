"""Independent read-only identity and arithmetic audit of 192."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/lupori_acquired_local_map_192_train_001'
previous = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
earlier = [json.loads(line) for line in (previous / 'cases.jsonl').read_text().splitlines()]
assert len(rows) == done['sections'] == summary['sections'] == len(earlier) == 64
assert config['source_sha256'] == done['source_sha256'] == sha(
    source / 'assay_lupori_acquired_local_map_192.py')
assert config['protocol_sha256'] == sha(source.parent /
    'docs/publication/LUPORI_ACQUIRED_LOCAL_MAP_192_PROTOCOL_20261011.md')
assert config['checkpoint_sha256'] == done['checkpoint_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt')
assert config['previous_189_completion_sha256'] == sha(previous / 'completed.json')
assert config['previous_189_cases_sha256'] == sha(previous / 'cases.jsonl')
assert config['split_sha256'] == sha(source / 'lupori_animal_split_185.json')
assert {name: sha(run / name) for name in done['output_sha256']} == done['output_sha256']
assert not done['trained'] and not done['calibrated'] and not done['public_benchmark_used']
assert Counter(row['animal_id'] for row in rows) == {animal: 16 for animal in config['animals']}
assert len({(row['animal_id'], row['section_id']) for row in rows}) == 64
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for row, old in zip(rows, earlier):
        assert (row['animal_id'], row['section_id'], row['source_thumbnail']) == (
            old['animal_id'], old['section_id'], old['source_thumbnail'])
        assert hashlib.sha256(archive.read(row['source_thumbnail'])).hexdigest() == (
            row['source_thumbnail_sha256'])
        animal, section = row['animal_id'], row['section_id']
        for suffix, field in (('X', 'displacement_x_sha256'),
                              ('Y', 'displacement_y_sha256')):
            assert hashlib.sha256(archive.read(f'{animal}/dispFields/{section}-dispField{suffix}.csv')
                ).hexdigest() == row[field]
        assert hashlib.sha256(archive.read(f'{animal}/masks/{section}-mask.png')
            ).hexdigest() == row['mask_sha256']
        assert row['tissue_sites_96'] > 0
        assert all(np.isfinite(row[key]) and row[key] >= 0 for key in (
            'zero_local_rms_mm', 'model_local_rms_mm', 'predicted_local_rms_mm',
            'target_local_p95_mm'))
        assert np.isfinite(np.asarray(row['removed_affine_local_um'])).all()
animal = {}
for identity in config['animals']:
    group = [row for row in rows if row['animal_id'] == identity]
    animal[identity] = {'sections': len(group),
        'zero_mean_mm': float(np.mean([row['zero_local_rms_mm'] for row in group])),
        'model_mean_mm': float(np.mean([row['model_local_rms_mm'] for row in group]))}
assert summary['animal'] == animal
zero = float(np.mean([animal[a]['zero_mean_mm'] for a in config['animals']]))
model = float(np.mean([animal[a]['model_mean_mm'] for a in config['animals']]))
better = sum(animal[a]['model_mean_mm'] < animal[a]['zero_mean_mm']
             for a in config['animals'])
assert summary['zero_equal_animal_mean_mm'] == zero
assert summary['model_equal_animal_mean_mm'] == model
assert summary['predeclared_gate'] == {'passed': model <= .9 * zero and better >= 3,
    'animals_improved': better}
print(json.dumps({'verified': True, 'sections': len(rows), 'zero_mm': zero,
    'model_mm': model, 'gate': summary['predeclared_gate'],
    'completion_sha256': sha(run / 'completed.json')}))
