"""Read-only arithmetic and provenance audit of the completed 154 DEV comparison."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


root = Path('I:/AnatomyTracker')
run = root / 'runs/joint_pose_154_frozen_dev_eval_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
for name in ('config', 'rows', 'summary'):
    assert sha(run / f'{name}.{ "jsonl" if name == "rows" else "json"}') == done[f'{name}_sha256']
source = Path(__file__).resolve().parent
assert sha(source / 'evaluate_joint_pose_154_frozen_dev.py') == config['evaluator_sha256']
assert sha(source / 'joint_anatomy_model_153.py') == config['model_source_sha256']
assert sha(root / 'runs/joint_anatomy_model_153_train151_002/completed.json') == \
    config['train153_completion_sha256']
assert sha(root / 'runs/joint_pose_curriculum_154_train151_001/completed.json') == \
    config['train154_completion_sha256']
for key, path in (('153_parent', root / 'runs/joint_anatomy_model_153_train151_002/joint_step_01300.pt'),
                  ('154_selected', root / 'runs/joint_pose_curriculum_154_train151_001' /
                   f"joint_pose_step_{config['selected_154_step']:05d}.pt")):
    assert sha(path) == config['checkpoint_sha256'][key]
for cohort, directory in (('v3', 'joint_in_path_correspondence_128_dev_panel'),
                          ('v4', 'fresh_v4_pose_dev_panel_132')):
    assert sha(root / 'data' / directory / 'completed.json') == config['panel_receipt_sha256'][cohort]
assert done['case_rows'] == len(rows) == 982
assert len({(row['arm'], row['cohort'], row['section_id']) for row in rows}) == len(rows)
assert {(row['cohort'], row['section_id']) for row in rows if row['arm'] == '153_parent'} == \
       {(row['cohort'], row['section_id']) for row in rows if row['arm'] == '154_selected'}
for row in rows:
    error = np.asarray(row['error_mm_all160'])
    prior = np.asarray(row['log_prior_all160_uncalibrated'])
    assert error.shape == prior.shape == (160,) and np.isfinite(error).all() and np.isfinite(prior).all()
    order = np.argsort(-prior)
    for key, value in (('top1_mm', error[order[0]]),
                       ('top2_best_mm', error[order[:2]].min()),
                       ('top16_best_mm', error[order[:16]].min()),
                       ('oracle_best_mm', error.min())):
        assert math.isclose(row[key], value, abs_tol=1e-6)
    assert 0 <= row['top1_normal_deg'] <= 90 and 0 <= row['oracle_normal_deg'] <= 90
for arm in ('153_parent', '154_selected'):
    for cohort, expected in (('v3', 243), ('v4', 248)):
        group = [row for row in rows if row['arm'] == arm and row['cohort'] == cohort]
        result = summary[f'{arm}/{cohort}']
        plans = sorted({row['synthetic_plan_id'] for row in group})
        assert len(group) == result['n'] == expected and len(plans) == result['synthetic_plans'] == 8
        for metric in ('top1_mm', 'top2_best_mm', 'top16_best_mm', 'oracle_best_mm',
                       'top1_normal_deg', 'oracle_normal_deg'):
            mean = np.mean([np.mean([row[metric] for row in group
                                    if row['synthetic_plan_id'] == plan]) for plan in plans])
            assert math.isclose(mean, result[metric], abs_tol=1e-8)
            if metric.endswith('_mm'):
                near = np.mean([np.mean([row[metric] <= 1.5 for row in group
                                        if row['synthetic_plan_id'] == plan]) for plan in plans])
                assert math.isclose(near, result[metric.replace('_mm', '_within1p5')], abs_tol=1e-8)
assert not any(done[key] or config[key] for key in
    ('calibrated', 'public_benchmark_used', 'final_animals_used'))
print(json.dumps({'verified': True, 'rows': len(rows), 'summary': summary,
                  'biological_animal_validation': False}, indent=2))
