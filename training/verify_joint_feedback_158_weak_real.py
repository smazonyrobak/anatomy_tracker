"""Read-only hash, donor and aggregate audit for the completed weak-real guard."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
repo = Path(__file__).resolve().parents[1]
run = root / 'runs/joint_feedback_158_weak_real_guard_001'
train = root / 'runs/joint_feedback_158_train151_001'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
train_done = json.loads((train / 'completed.json').read_text())
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'summary.json') == done['summary_sha256']
assert sha(run / 'rows.jsonl') == done['rows_sha256']
assert sha(repo / 'training/evaluate_joint_feedback_158_weak_real.py') == config['evaluator_sha256']
assert sha(train / 'completed.json') == config['training_completion_sha256']
assert sha(coronal / 'completed.json') == config['coronal_completion_sha256']
assert sha(sagittal / 'summary.json') == config['sagittal_summary_sha256']
assert not any(done[key] or config[key] for key in
    ('calibrated', 'final_animals_used', 'public_benchmark_used'))

coronal_records = [r for line in (coronal / 'records.jsonl').open()
                   if (r := json.loads(line))['training_split'] == 'development']
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
assert len(coronal_records) == 64 and len({r['animal_id'] for r in coronal_records}) == 6
assert len(sagittal_records) == 158 and len({r['donor_id'] for r in sagittal_records}) == 8
checkpoints = [('baseline', 'feedback', 0)] + [
    (arm, arm, step) for arm in ('feedback', 'detached_control') for step in (500, 1000)]
assert len(rows) == done['rows'] == 222 * len(checkpoints)
for label, arm, step in checkpoints:
    key = label if label == 'baseline' else f'{arm}/{step}'
    path = train / arm / f'joint_feedback_step_{step:05d}.pt'
    assert sha(path) == config['checkpoint_sha256'][key] == \
        train_done['arms'][arm]['checkpoint_sha256'][str(step)]
    for domain, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
        group = [r for r in rows if r['checkpoint'] == key and r['domain'] == domain]
        assert [r['section_id'] for r in group] == [r['section_id'] for r in records]
        assert [r['specimen_id'] for r in group] == [r['specimen_id'] for r in records]
        assert [r['experiment_id'] for r in group] == [r['experiment_id'] for r in records]
        assert all(r['label_role'] == 'inherited weak Allen affine, not expert pose'
                   for r in group)
        donors = sorted({r['donor_id'] for r in group})
        result = summary[f'{key}/{domain}']
        assert result['sections'] == len(records) and result['donors'] == len(donors)
        for metric in ('top1_five_point_weak_mm', 'top16_best_five_point_weak_mm',
                       'top1_normal_weak_deg'):
            expected = np.mean([np.mean([r[metric] for r in group
                if r['donor_id'] == donor]) for donor in donors])
            assert math.isclose(result[metric], expected, abs_tol=1e-8)
print(json.dumps({'verified': True, 'real_weak_sections_per_checkpoint': 222,
                  'summary': summary}, indent=2))
