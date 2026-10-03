"""Independent read-only audit of frozen 039 TRAIN and development results."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/pose_feedback_3d_039_pilot_v2'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/pose_feedback_3d_039_pilot_development_eval'
source = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train_receipt = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert train_receipt['updates'] == 500 and train_receipt['accepted_synthetic'] == 1000
assert train_config['real_training_images'] == 0 and not train_config['public_benchmark_used']
for field, file in (('config', 'config.json'), ('draws', 'draws.jsonl'),
                    ('training', 'training.jsonl')):
    assert sha(run / file) == train_receipt[field + '_sha256']
for name, digest in train_config['source_sha256'].items():
    assert sha(source / name) == digest
assert sha(Path(train_config['parent'])) == train_config['parent_sha256']
train_rows = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert len(train_rows) == 500
assert [row['step'] for row in train_rows] == list(range(1, 501))
assert all(row['synthetic_presentations'] == row['step'] * 2 for row in train_rows)
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
accepted = [row for row in draws if row['used']]
assert len(accepted) == 1000
assert len({row['physical_section_id'] for row in accepted}) == 1000
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train'
           for row in accepted)

panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in panel_rows if row['eligible']]
assert len(panel_rows) == 256 and len(eligible) == 177
assert len({row['synthetic_subject_plan_id'] for row in panel_rows}) == 8
assert len({row['panel_physical_section_id'] for row in panel_rows}) == 256
assert len({row['subject_ouv_sha256'] for row in panel_rows}) == 256
assert {row['synthetic_animal_id'] for row in panel_rows}.isdisjoint(
    {row['base_lineage']['synthetic_animal_id'] for row in draws})

receipt = json.loads((evaluation / 'completed.json').read_text())
config = json.loads((evaluation / 'config.json').read_text())
for field, file in (('config', 'config.json'), ('rows', 'rows.jsonl'),
                    ('summary', 'summary.json')):
    assert sha(evaluation / file) == receipt[field + '_sha256']
assert config['source_sha256'] == sha(source / 'evaluate_pose_feedback_3d_039.py')
assert config['run_completed_sha256'] == sha(run / 'completed.json')
assert config['run_config_sha256'] == sha(run / 'config.json')
assert config['panel_records_sha256'] == sha(panel / 'records.jsonl')
for step, digest in config['checkpoints_sha256'].items():
    assert sha(run / f'joint_step_{int(step):05d}.pt') == digest
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
saved = json.loads((evaluation / 'summary.json').read_text())
assert len(rows) == (177 + 64) * 3 == receipt['rows']
assert len(saved['checkpoints']) == 3
assert saved['synthetic_eligible_sections'] == 177
for entry in saved['checkpoints']:
    step = entry['step']
    assert step in (0, 250, 500)
    for subset, label, group, count, groups in (
            ('synthetic', 'synthetic', 'synthetic_subject_plan_id', 177, 8),
            ('real_weak_allen', 'real_weak', 'animal_id', 64, 6)):
        selected = [row for row in rows if row['step'] == step and row['set'] == subset]
        assert len(selected) == count and len({row[group] for row in selected}) == groups
        assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'],
                     row['section_id']) for row in selected}) == count
        fields = ('prior_top1_um', 'prior_best8_um', 'refined_top1_um',
                  'refined_best8_um', 'refined_selected_um')
        if subset == 'synthetic':
            fields += ('normal_angle_top1_deg', 'normal_angle_selected_deg',
                       'top1_3d_eligible_match_fraction', 'top1_match_recall_1500',
                       'prior_best8_match_recall_1500')
        for field in fields:
            mean = np.mean([np.mean([row[field] for row in selected if row[group] == identity])
                            for identity in {row[group] for row in selected}])
            assert math.isclose(entry[label + '_' + field], mean, abs_tol=1e-5)
            assert all(np.isfinite(row[field]) and row[field] >= 0 for row in selected)
    real = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    for donor in entry['real_weak_by_donor_selected_um']:
        assert math.isclose(entry['real_weak_by_donor_selected_um'][donor],
            np.mean([row['refined_selected_um'] for row in real
                     if str(row['animal_id']) == donor]), abs_tol=1e-5)
        assert math.isclose(entry['real_weak_by_donor_prior_top1_um'][donor],
            np.mean([row['prior_top1_um'] for row in real
                     if str(row['animal_id']) == donor]), abs_tol=1e-5)
    assert math.isclose(entry['synthetic_selected_gain_um'],
        entry['synthetic_prior_top1_um'] - entry['synthetic_refined_selected_um'], abs_tol=1e-5)
    assert math.isclose(entry['synthetic_best8_gain_um'],
        entry['synthetic_prior_best8_um'] - entry['synthetic_refined_best8_um'], abs_tol=1e-5)
    regression = max(entry['real_weak_by_donor_selected_um'][donor] -
        entry['real_weak_by_donor_prior_top1_um'][donor]
        for donor in entry['real_weak_by_donor_selected_um'])
    assert math.isclose(entry['max_donor_selected_regression_um'], regression, abs_tol=1e-5)
    gate = (step > 0 and entry['synthetic_selected_gain_um'] >= 250 and
            entry['synthetic_best8_gain_um'] >= 200 and regression <= 200)
    assert entry['development_gate'] == gate
print(json.dumps({'audit': 'PASS', 'accepted_train_sections': len(accepted),
                  'panel_eligible': len(eligible), 'evaluation_rows': len(rows),
                  'gate': [row['development_gate'] for row in saved['checkpoints']]}))
