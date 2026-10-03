"""Read-only frozen-row audit for per-candidate pose-feedback 036."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker/runs/pose_feedback_036_development_eval')
train = Path('I:/AnatomyTracker/runs/pose_feedback_036')
source = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


receipt = json.loads((root / 'completed.json').read_text())
training_receipt = json.loads((train / 'completed.json').read_text())
training_config = json.loads((train / 'config.json').read_text())
assert training_receipt['updates'] == 3000 and training_receipt['accepted_synthetic'] == 6000
for name, filename in (('config', 'config.json'), ('draws', 'draws.jsonl'),
                       ('training', 'training.jsonl')):
    assert sha(train / filename) == training_receipt[f'{name}_sha256']
assert len((train / 'training.jsonl').read_text().splitlines()) == 3000
draws = [json.loads(line) for line in (train / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 6000 and len({row['physical_section_id'] for row in used}) == 6000
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train' for row in used)
for name, digest in training_config['source_sha256'].items():
    assert sha(source / name) == digest
for name in ('config', 'summary'):
    assert sha(root / f'{name}.json') == receipt[f'{name}_sha256']
assert sha(root / 'rows.jsonl') == receipt['rows_sha256']
config = json.loads((root / 'config.json').read_text())
assert config['run_completed_sha256'] == sha(train / 'completed.json')
assert config['run_config_sha256'] == sha(train / 'config.json')
for step, digest in config['checkpoints_sha256'].items():
    assert sha(train / f'joint_step_{int(step):05d}.pt') == digest
rows = [json.loads(line) for line in (root / 'rows.jsonl').open()]
saved = json.loads((root / 'summary.json').read_text())
assert len(rows) == (185 + 64) * 4
assert len(saved['checkpoints']) == 4
fields = ('prior_top1_um', 'refined_top1_um', 'initial_best8_um',
          'refined_best8_um', 'refined_selected_um', 'mean_translation_update_um')
for result in saved['checkpoints']:
    step = result['step']
    assert step in (0, 1000, 2000, 3000)
    for subset, label, group, count, groups in (
            ('synthetic', 'synthetic', 'synthetic_subject_plan_id', 185, 8),
            ('real_weak_allen', 'real_weak', 'animal_id', 64, 6)):
        selected = [row for row in rows if row['step'] == step and row['set'] == subset]
        assert len(selected) == count
        identities = {row[group] for row in selected}
        assert len(identities) == groups
        assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'], row['section_id'])
                    for row in selected}) == count
        for field in fields:
            mean = np.mean([np.mean([row[field] for row in selected if row[group] == identity])
                            for identity in identities])
            assert math.isclose(result[label + '_' + field], mean, abs_tol=1e-5)
        assert all(np.isfinite(row[field]) and row[field] >= 0 for row in selected for field in fields)
    real = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    for donor, value in result['real_weak_by_donor_refined_top1_um'].items():
        assert math.isclose(value, np.mean([row['refined_top1_um'] for row in real
                                             if str(row['animal_id']) == donor]), abs_tol=1e-5)
base = saved['checkpoints'][0]
for result in saved['checkpoints']:
    regression = max(result['real_weak_by_donor_refined_top1_um'][donor] -
                     base['real_weak_by_donor_refined_top1_um'][donor]
                     for donor in base['real_weak_by_donor_refined_top1_um'])
    assert math.isclose(result['max_donor_refined_top1_regression_um'], regression, abs_tol=1e-5)
    gate = (result['step'] > 0 and
            result['synthetic_refined_top1_um'] <= base['synthetic_refined_top1_um'] - 250 and
            result['synthetic_refined_best8_um'] <= base['synthetic_refined_best8_um'] - 200 and
            regression <= 200)
    assert result['development_gate'] == gate
assert saved['any_development_gate'] == any(row['development_gate'] for row in saved['checkpoints'])
assert config['steps'] == [0, 1000, 2000, 3000]
print(json.dumps({'audit': 'PASS', 'rows': len(rows),
                  'gate': saved['any_development_gate']}))
