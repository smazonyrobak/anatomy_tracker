"""Independent frozen-output audit for the 240 TRAIN-inner descriptor pilot."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import torch

run = root / 'runs/learned_coherent_3d_match_240_train_001'
source = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(name):
    return [json.loads(line) for line in (run / name).read_text(encoding='utf-8').splitlines()]


receipt = json.loads((run / 'completed.json').read_text(encoding='utf-8'))
config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
manifest = json.loads((run / 'validation_manifest.json').read_text(encoding='utf-8'))
draws = rows('draws.jsonl')
training = rows('training.jsonl')
validations = rows('validation.jsonl')
assert receipt['training_exited'] and receipt['updates'] == len(training) in (256, 512, 768, 1024)
assert receipt['draw_attempts'] == len(draws)
assert all(not receipt[key] for key in ('reserved_animal_used',
    'external_pretrained_weights_used', 'public_benchmark_used', 'calibrated'))
assert all(not config[key] for key in ('reserved_animal_used',
    'external_pretrained_weights_used', 'public_benchmark_used', 'calibrated'))
assert sha(run / 'best.pt') == receipt['checkpoint_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(source / name) == expected, name
assert sha(source.parent / 'docs/publication/LEARNED_COHERENT_3D_MATCH_240_PROTOCOL_20261012.md') == config['protocol_sha256']
assert sha(root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt') == config['parent_sha256']
for name, expected in receipt['output_sha256'].items():
    assert sha(run / name) == expected, name
assert [row['step'] for row in training] == list(range(1, receipt['updates'] + 1))
used = [row for row in draws if row['used']]
assert len(used) == len(training)
assert [row['step'] for row in used] == [row['step'] for row in training]
assert all(0 <= row['base_index'] <= 47 and row['split'] == 'train' for row in used)
train_ids = [row['physical_section_id'] for row in used]
assert len(train_ids) == len(set(train_ids))
assert all(row['physical_section_id'] == used[index]['physical_section_id']
           for index, row in enumerate(training))
assert sum(row['appearance'] == 'v4' for row in training) == (len(training) + 1) // 2
assert sum(row['appearance'] == 'v7' for row in training) == len(training) // 2
assert len(manifest) == 32 and all(48 <= row['base'] <= 51 for row in manifest)
validation_ids = [row['physical_section_id'] for row in manifest]
assert len(validation_ids) == len(set(validation_ids)) and not set(train_ids) & set(validation_ids)
assert [row['step'] for row in validations] == list(range(256, receipt['updates'] + 1, 256))
best_loss, best_step = float('inf'), 0
for check in validations:
    assert len(check['rows']) == 32
    assert {row['physical_section_id'] for row in check['rows']} == set(validation_ids)
    improved = check['mean_loss'] < best_loss - .002
    assert check['improved'] == improved
    if improved:
        best_loss, best_step = check['mean_loss'], check['step']
    assert check['best_step'] == best_step
assert receipt['best_step'] == best_step
assert abs(receipt['best_validation_loss'] - best_loss) < 1e-10
checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == best_step and checkpoint['parent_sha256'] == config['parent_sha256']
assert checkpoint['config'] == config
print(json.dumps({'verified': True, 'updates': receipt['updates'],
    'unique_training_sections': len(train_ids), 'validation_sections': len(manifest),
    'best_step': best_step, 'best_validation_loss': best_loss,
    'checkpoint_sha256': receipt['checkpoint_sha256']}))
