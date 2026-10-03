"""Independent read-only audit of frozen 041 training and development evaluation."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/pose_feedback_global_041_pilot'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
evaluation = root / 'runs/pose_feedback_global_041_development_eval'
source = Path(__file__).parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train_receipt = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert train_receipt['updates'] == train_config['updates'] == 2000
assert train_receipt['accepted_synthetic'] == 4000
assert train_receipt['unique_real_weak'] == 2000
assert train_config['synthetic_per_batch'] == 2
assert train_config['real_weak_per_batch'] == 1
assert not train_config['calibrated'] and not train_config['public_benchmark_used']
for field, file in (('config', 'config.json'), ('draws', 'draws.jsonl'),
                    ('training', 'training.jsonl')):
    assert sha(run / file) == train_receipt[field + '_sha256']
for name, digest in train_config['source_sha256'].items():
    assert sha(source / name) == digest
assert sha(Path(train_config['parent'])) == train_config['parent_sha256']
assert sha(Path(train_config['matcher_parent'])) == train_config['matcher_parent_sha256']
schedule = np.load(run / 'real_schedule.npy', allow_pickle=False)
assert schedule.shape == (2000, 2)
assert len({tuple(map(int, row)) for row in schedule}) == 2000
assert sha(run / 'real_schedule.npy') == train_config['real_schedule_sha256']
assert train_config['real_section_policy'] == (
    'without replacement within 041; reuse across experiments allowed')
train_rows = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert len(train_rows) == 2000
assert [row['step'] for row in train_rows] == list(range(1, 2001))
assert all(row['synthetic_presentations'] == row['step'] * 2 and
           row['real_weak_presentations'] == row['step'] for row in train_rows)
assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
           for row in train_rows)
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
accepted = [row for row in draws if row['used'] and row['slot'] < 2]
real_draws = [row for row in draws if row['slot'] == 2]
assert len(accepted) == 4000 and len(real_draws) == 2000
assert len({row['physical_section_id'] for row in accepted}) == 4000
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train'
           for row in accepted)
assert all(row['used'] and row['step'] == step for step, row in
           enumerate(real_draws, 1))

panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in panel_rows if row['eligible']]
assert len(panel_rows) == 256 and len(eligible) == 177
assert len({row['synthetic_subject_plan_id'] for row in panel_rows}) == 8
assert len({row['panel_physical_section_id'] for row in panel_rows}) == 256
assert {row['synthetic_animal_id'] for row in panel_rows}.isdisjoint(
    {row['base_lineage']['synthetic_animal_id'] for row in draws if row['slot'] < 2})
real_records = [json.loads(line) for line in (real / 'records.jsonl').open()
                if json.loads(line)['training_split'] == 'development']
assert len(real_records) == 64
assert len({row['animal_id'] for row in real_records}) == 6
assert {str(row['animal_id']) for row in real_records}.isdisjoint(
    {str(row['animal_id']) for row in real_draws})

receipt = json.loads((evaluation / 'completed.json').read_text())
config = json.loads((evaluation / 'config.json').read_text())
for field, file in (('config', 'config.json'), ('rows', 'rows.jsonl'),
                    ('summary', 'summary.json')):
    assert sha(evaluation / file) == receipt[field + '_sha256']
assert config['source_sha256'] == sha(source / 'evaluate_pose_feedback_global_041.py')
assert config['run_completed_sha256'] == sha(run / 'completed.json')
assert config['run_config_sha256'] == sha(run / 'config.json')
assert config['panel_records_sha256'] == sha(panel / 'records.jsonl')
assert config['real_records_sha256'] == sha(real / 'records.jsonl')
assert config['parent_sha256'] == sha(Path(train_config['parent']))
for step, digest in config['checkpoints_sha256'].items():
    assert sha(run / f'joint_step_{int(step):05d}.pt') == digest
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
saved = json.loads((evaluation / 'summary.json').read_text())
steps = (0, 500, 1000, 1500, 2000)
assert len(rows) == (177 + 64) * len(steps) == receipt['rows']
assert [entry['step'] for entry in saved['checkpoints']] == list(steps)
assert saved['synthetic_eligible_sections'] == 177
for entry in saved['checkpoints']:
    step = entry['step']
    for subset, label, group, count, groups in (
            ('synthetic', 'synthetic', 'synthetic_subject_plan_id', 177, 8),
            ('real_weak_allen', 'real_weak', 'animal_id', 64, 6)):
        selected = [row for row in rows if row['step'] == step and row['set'] == subset]
        assert len(selected) == count and len({row[group] for row in selected}) == groups
        assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'],
                     row['section_id']) for row in selected}) == count
        fields = ('prior_top1_um', 'prior_best4_um', 'prior_best8_um',
                  'raw_top1_um', 'raw_best8_um', 'refined_top1_um',
                  'refined_best4_um', 'refined_best8_um',
                  'refined_selected4_um', 'refined_selected_um',
                  'top1_correction_gate', 'selected_correction_gate')
        if subset == 'synthetic':
            fields += ('normal_angle_top1_deg', 'normal_angle_selected_deg',
                       'top1_match_recall_1500', 'prior_best8_match_recall_1500')
        for field in fields:
            mean = np.mean([np.mean([row[field] for row in selected if row[group] == identity])
                            for identity in {row[group] for row in selected}])
            assert math.isclose(entry[label + '_' + field], mean, abs_tol=1e-5)
            assert all(np.isfinite(row[field]) and row[field] >= 0 for row in selected)
    real_rows = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    for donor in entry['real_weak_by_donor_selected_um']:
        assert math.isclose(entry['real_weak_by_donor_selected_um'][donor],
            np.mean([row['refined_selected_um'] for row in real_rows
                     if str(row['animal_id']) == donor]), abs_tol=1e-5)
        assert math.isclose(entry['real_weak_by_donor_selected4_um'][donor],
            np.mean([row['refined_selected4_um'] for row in real_rows
                     if str(row['animal_id']) == donor]), abs_tol=1e-5)
        assert math.isclose(entry['real_weak_by_donor_prior_top1_um'][donor],
            np.mean([row['prior_top1_um'] for row in real_rows
                     if str(row['animal_id']) == donor]), abs_tol=1e-5)
    selected_gain = entry['synthetic_prior_top1_um'] - entry['synthetic_refined_selected_um']
    best8_gain = entry['synthetic_prior_best8_um'] - entry['synthetic_refined_best8_um']
    regression = max(entry['real_weak_by_donor_selected_um'][donor] -
        entry['real_weak_by_donor_prior_top1_um'][donor]
        for donor in entry['real_weak_by_donor_selected_um'])
    assert math.isclose(entry['synthetic_selected_gain_um'], selected_gain, abs_tol=1e-5)
    assert math.isclose(entry['synthetic_best8_gain_um'], best8_gain, abs_tol=1e-5)
    assert math.isclose(entry['max_donor_selected_regression_um'], regression, abs_tol=1e-5)
    assert entry['development_gate'] == (step > 0 and selected_gain >= 250 and
                                          best8_gain >= 200 and regression <= 200)
print(json.dumps({'audit': 'PASS', 'accepted_train_sections': len(accepted),
                  'unique_weak_real_train_sections': len(real_draws),
                  'panel_eligible': len(eligible), 'evaluation_rows': len(rows),
                  'gate': [row['development_gate'] for row in saved['checkpoints']]}))
