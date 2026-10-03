"""Independent read-only audit of frozen 037 training and development rows."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/pose_feedback_corr_037'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/pose_feedback_corr_037_development_eval'
source = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


training_receipt = json.loads((run / 'completed.json').read_text())
training_config = json.loads((run / 'config.json').read_text())
assert training_receipt['updates'] == 4000 and training_receipt['accepted_synthetic'] == 8000
assert training_config['real_training_images'] == 0 and not training_config['public_benchmark_used']
for name, filename in (('config', 'config.json'), ('draws', 'draws.jsonl'),
                       ('training', 'training.jsonl')):
    assert sha(run / filename) == training_receipt[f'{name}_sha256']
for name, digest in training_config['source_sha256'].items():
    assert sha(source / name) == digest
assert sha(Path(training_config['parent'])) == training_config['parent_sha256']
training_rows = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert len(training_rows) == 4000
assert [row['step'] for row in training_rows] == list(range(1, 4001))
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 8000
assert len({row['physical_section_id'] for row in used}) == 8000
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train' for row in used)
assert all(row['synthetic_presentations'] == row['step'] * 2 for row in training_rows)

panel_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(panel_records) == 256
panel_ids = {row['synthetic_subject_plan_id'] for row in panel_records}
assert len(panel_ids) == 8
assert all(sum(row['synthetic_subject_plan_id'] == identity for row in panel_records) == 32
           for identity in panel_ids)
assert len({row['panel_physical_section_id'] for row in panel_records}) == 256
assert len({row['subject_ouv_sha256'] for row in panel_records}) == 256
assert {row['synthetic_animal_id'] for row in panel_records}.isdisjoint(
    {row['base_lineage']['synthetic_animal_id'] for row in draws})
eligible = [row for row in panel_records if row['eligible']]

evaluation_receipt = json.loads((evaluation / 'completed.json').read_text())
evaluation_config = json.loads((evaluation / 'config.json').read_text())
for name in ('config', 'summary'):
    assert sha(evaluation / f'{name}.json') == evaluation_receipt[f'{name}_sha256']
assert sha(evaluation / 'rows.jsonl') == evaluation_receipt['rows_sha256']
assert evaluation_config['source_sha256'] == sha(source / 'evaluate_pose_feedback_corr_037.py')
assert evaluation_config['run_completed_sha256'] == sha(run / 'completed.json')
assert evaluation_config['run_config_sha256'] == sha(run / 'config.json')
assert evaluation_config['panel_records_sha256'] == sha(panel / 'records.jsonl')
assert evaluation_config['synthetic_panel_sections'] == 256
assert evaluation_config['synthetic_eligible_sections'] == len(eligible)
assert evaluation_config['synthetic_ineligible_sections'] == 256 - len(eligible)
for step, digest in evaluation_config['checkpoints_sha256'].items():
    assert sha(run / f'joint_step_{int(step):05d}.pt') == digest
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
saved = json.loads((evaluation / 'summary.json').read_text())
assert len(rows) == (len(eligible) + 64) * 2 * 4
assert len(saved['checkpoints']) == 8
assert saved['synthetic_eligible_sections'] == len(eligible)
fields = ('prior_top1_um', 'refined_top1_um', 'initial_best8_um',
          'refined_best8_um', 'refined_selected_um', 'mean_translation_update_um')
for result in saved['checkpoints']:
    step, arm = result['step'], result['arm']
    assert step in (0, 1000, 2000, 4000) and arm in ('atlas', 'control')
    for subset, label, group, count, groups in (
            ('synthetic', 'synthetic', 'synthetic_subject_plan_id', len(eligible), 8),
            ('real_weak_allen', 'real_weak', 'animal_id', 64, 6)):
        selected = [row for row in rows if row['step'] == step and row['arm'] == arm
                    and row['set'] == subset]
        assert len(selected) == count
        identities = {row[group] for row in selected}
        assert len(identities) == groups
        assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'], row['section_id'])
                    for row in selected}) == count
        for field in fields + (('normal_angle_top1_deg', 'normal_angle_selected_deg')
                               if subset == 'synthetic' else ()):
            mean = np.mean([np.mean([row[field] for row in selected if row[group] == identity])
                            for identity in identities])
            assert math.isclose(result[label + '_' + field], mean, abs_tol=1e-5)
            assert all(np.isfinite(row[field]) and row[field] >= 0 for row in selected)
    real = [row for row in rows if row['step'] == step and row['arm'] == arm
            and row['set'] == 'real_weak_allen']
    for metric in ('refined_top1', 'refined_selected'):
        for donor, value in result[f'real_weak_by_donor_{metric}_um'].items():
            assert math.isclose(value, np.mean([row[f'{metric}_um'] for row in real
                                                 if str(row['animal_id']) == donor]), abs_tol=1e-5)
base = next(row for row in saved['checkpoints'] if row['step'] == 0 and row['arm'] == 'atlas')
for result in saved['checkpoints']:
    control = next(row for row in saved['checkpoints'] if row['step'] == result['step']
                   and row['arm'] == 'control')
    regression = max(result['real_weak_by_donor_refined_selected_um'][donor] -
                     base['real_weak_by_donor_refined_selected_um'][donor]
                     for donor in base['real_weak_by_donor_refined_selected_um'])
    assert math.isclose(result['max_donor_refined_selected_regression_um'], regression,
                        abs_tol=1e-5)
    if result['arm'] == 'atlas':
        assert math.isclose(result['atlas_advantage_selected_um'],
            control['synthetic_refined_selected_um'] - result['synthetic_refined_selected_um'],
            abs_tol=1e-5)
    else:
        assert result['atlas_advantage_selected_um'] is None
    gate = (result['arm'] == 'atlas' and result['step'] > 0 and
            result['synthetic_refined_selected_um'] <= base['synthetic_refined_selected_um'] - 250 and
            result['synthetic_refined_best8_um'] <= base['synthetic_refined_best8_um'] - 200 and
            control['synthetic_refined_selected_um'] - result['synthetic_refined_selected_um'] >= 150 and
            regression <= 200)
    assert result['development_gate'] == gate
assert saved['any_development_gate'] == any(row['development_gate'] for row in saved['checkpoints'])
print(json.dumps({'audit': 'PASS', 'panel': len(panel_records), 'eligible': len(eligible),
                  'rows': len(rows), 'gate': saved['any_development_gate']}))
