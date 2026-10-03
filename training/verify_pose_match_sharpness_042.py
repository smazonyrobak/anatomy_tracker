"""Independent read-only audit of the frozen 042 mechanism diagnostic."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/pose_match_sharpness_042_diagnostic'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
run = root / 'runs/pose_feedback_global_041_pilot'
evaluation = root / 'runs/pose_feedback_global_041_development_eval'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


receipt = json.loads((out / 'completed.json').read_text())
config = json.loads((out / 'config.json').read_text())
for field in ('config', 'rows', 'summary'):
    assert sha(out / (field + '.json' if field != 'rows' else 'rows.jsonl')) == (
        receipt[field + '_sha256'])
assert config['step'] == 1500 and config['beam'] == 8
assert config['source_sha256'] == sha(Path(__file__).with_name(
    'diagnose_pose_match_sharpness_042.py'))
assert config['parent_sha256'] == sha(root / 'runs/one_shot_exposure_019/joint_step_18000.pt')
assert config['head_sha256'] == sha(run / 'joint_step_01500.pt')
assert config['train_completed_sha256'] == sha(run / 'completed.json')
assert config['panel_records_sha256'] == sha(panel / 'records.jsonl')
assert config['real_records_sha256'] == sha(real / 'records.jsonl')
assert not config['calibrated'] and not config['public_benchmark_used']
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == receipt['rows'] == 241
assert summary['synthetic_eligible_sections'] == 177
assert summary['real_weak_sections'] == 64
assert summary['soft_fit_reproduction_max_um'] < 2
fields = [prefix + '_' + suffix + '_um' for prefix in
          ('prior', 'soft_raw', 'soft_gated', 'four_raw', 'four_gated',
           'hard_raw', 'hard_gated') for suffix in ('top1', 'best8')]
for subset, label, group, count, identities in (
        ('synthetic', 'synthetic', 'synthetic_subject_plan_id', 177, 8),
        ('real_weak_allen', 'real_weak', 'animal_id', 64, 6)):
    selected = [row for row in rows if row['set'] == subset]
    assert len(selected) == count and len({row[group] for row in selected}) == identities
    assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'],
                 row['section_id']) for row in selected}) == count
    for field in fields:
        mean = np.mean([np.mean([row[field] for row in selected if row[group] == identity])
                        for identity in {row[group] for row in selected}])
        assert math.isclose(summary[label][field], mean, abs_tol=1e-5)
        assert all(np.isfinite(row[field]) and row[field] >= 0 for row in selected)
        if field.endswith('best8_um'):
            assert all(row[field] <= row[field.replace('best8', 'top1')] + 1e-4
                       for row in selected)
synthetic = [row for row in rows if row['set'] == 'synthetic']
real_rows = [row for row in rows if row['set'] == 'real_weak_allen']
assert len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert {row['animal_id'] for row in real_rows}.isdisjoint(
    {row['animal_id'] for row in synthetic})
near = [row for row in synthetic if row['prior_best8_um'] <= 1000]
assert len(near) == summary['near_true_sections'] == 112
for field in fields:
    expected = np.mean([np.mean([row[field] for row in near
                                if row['synthetic_subject_plan_id'] == identity])
                        for identity in {row['synthetic_subject_plan_id'] for row in near}])
    assert math.isclose(summary['near_true'][field], expected, abs_tol=1e-5)
for appearance, values in summary['by_appearance'].items():
    selected = [row for row in synthetic if row['appearance_mode'] == appearance]
    assert selected
    for field in fields:
        assert math.isclose(values[field], np.mean([row[field] for row in selected]),
                            abs_tol=1e-5)
for name, saved in summary['worst_real_donor_regression_um'].items():
    worst = max(np.mean([row[name + '_top1_um'] - row['prior_top1_um']
                         for row in real_rows if row['animal_id'] == donor])
                for donor in {row['animal_id'] for row in real_rows})
    assert math.isclose(saved, worst, abs_tol=1e-5)
for name, decision in summary['mechanism_direction_promising'].items():
    expected = (summary['synthetic']['prior_best8_um'] -
                summary['synthetic'][name + '_best8_um'] >= 200 and
                summary['synthetic'][name + '_top1_um'] -
                summary['synthetic']['prior_top1_um'] <= 200)
    assert decision == expected
baseline = next(row for row in json.loads((evaluation / 'summary.json').read_text())[
    'checkpoints'] if row['step'] == 1500)
for subset, label in (('synthetic', 'synthetic'), ('real_weak_allen', 'real_weak')):
    for field in ('top1_um', 'best8_um'):
        for name, original in (('prior', 'prior'), ('soft_raw', 'raw'),
                               ('soft_gated', 'refined')):
            assert math.isclose(summary[label][name + '_' + field],
                baseline[label + '_' + original + '_' + field], abs_tol=.02)
print(json.dumps({'audit': 'PASS', 'rows': len(rows),
                  'max_soft_reproduction_um': summary['soft_fit_reproduction_max_um'],
                  'mechanism_direction_promising':
                      summary['mechanism_direction_promising']}))
