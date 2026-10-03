"""Independent read-only receipt, lineage and row audit of 045."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
run = root / 'runs/pose_fine_patch_045_pilot'
out = root / 'runs/pose_fine_patch_045_development_eval'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
source = Path(__file__).parent
train_receipt = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert train_receipt['updates'] == train_config['updates'] == 500
assert train_receipt['synthetic_presentations'] == 1000
assert not train_receipt['calibrated'] and not train_receipt['public_benchmark_used']
for name in ('config', 'draws', 'training'):
    path = run / (name + ('.jsonl' if name != 'config' else '.json'))
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == train_receipt[name + '_sha256']
for name, digest in train_config['source_sha256'].items():
    with (source / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
for name, path in (('parent', root / 'runs/one_shot_exposure_019/joint_step_18000.pt'),
                   ('coarse', root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'),
                   ('fine_parent', root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == train_config[name + '_sha256']
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
assert len(training) == 500 and [row['step'] for row in training] == list(range(1, 501))
assert sum(row['used'] for row in draws) == 1000
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train'
           for row in draws)
assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
           for row in training)
panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in panel_rows if row['eligible']]
assert len(eligible) == 177
assert {row['synthetic_animal_id'] for row in panel_rows}.isdisjoint(
    {row['base_lineage']['synthetic_animal_id'] for row in draws})

receipt = json.loads((out / 'completed.json').read_text())
config = json.loads((out / 'config.json').read_text())
for name in ('config', 'rows', 'summary'):
    path = out / (name + ('.jsonl' if name == 'rows' else '.json'))
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt[name + '_sha256']
for name, path in (('source', source / 'evaluate_pose_fine_patch_045.py'),
                   ('patch_source', source / 'pose_fine_patch_045.py'),
                   ('panel_records', panel / 'records.jsonl'),
                   ('train_completed', run / 'completed.json'),
                   ('parent', root / 'runs/one_shot_exposure_019/joint_step_18000.pt'),
                   ('coarse', root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == config[name + '_sha256']
for step, digest in config['checkpoints_sha256'].items():
    with (run / f'patch_step_{int(step):05d}.pt').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
assert config['steps'] == [0, 500] and config['beam'] == 8 and config['shortlist'] == 32
assert not config['calibrated'] and not config['public_benchmark_used']
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == len(eligible) == receipt['rows'] == 177
assert summary['synthetic_eligible_sections'] == 177
assert summary['synthetic_subjects'] == 8
for row, record in zip(rows, eligible):
    for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
                'synthetic_subject_plan_id', 'appearance_mode', 'sha256'):
        assert row[key] == record[key]
    assert 0 < row['valid_query_cells'] <= 256
    assert len(row['query_pixel_yx']) == row['valid_query_cells']
    assert 0 <= row['prior_best8_um'] <= row['prior_top1_um'] + 1e-4
    assert row['prior_best8_index'] in range(8)
    for label in ('top1', 'best8'):
        assert (0 <= row[label + '_available_500um'] <=
                row[label + '_available_1500um'] <= row['valid_query_cells'])
        for step in (0, 500):
            values = np.asarray(row[f'{label}_step{step}_selected_um'])
            assert len(values) == row['valid_query_cells'] and np.isfinite(values).all()
            for threshold in (500, 1500):
                measured = np.mean(values <= threshold)
                assert math.isclose(measured,
                    row[f'{label}_step{step}_recall1_{threshold}um'], abs_tol=1e-8)
                assert int((values <= threshold).sum()) <= row[label + f'_available_{threshold}um']
identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
assert math.isclose(summary['mean_valid_query_cells'],
                    np.mean([row['valid_query_cells'] for row in rows]), abs_tol=1e-8)
for label in ('top1', 'best8'):
    for step in (0, 500):
        for threshold in (500, 1500):
            field = f'{label}_step{step}_recall1_{threshold}um'
            expected = np.mean([np.mean([row[field] for row in rows
                if row['synthetic_subject_plan_id'] == identity]) for identity in identities])
            assert math.isclose(expected, summary[label][f'step{step}_recall1_{threshold}um'],
                                abs_tol=1e-8)
    for appearance, values in summary[label]['by_appearance_1500um'].items():
        for step in (0, 500):
            means = [np.mean([row[f'{label}_step{step}_recall1_1500um'] for row in rows
                     if row['synthetic_subject_plan_id'] == identity and
                        row['appearance_mode'] == appearance])
                     for identity in identities if any(row['synthetic_subject_plan_id'] == identity and
                        row['appearance_mode'] == appearance for row in rows)]
            assert math.isclose(np.mean(means), values[str(step)], abs_tol=1e-8)
assert summary['correspondence_promising'] == (
    summary['best8']['step500_recall1_1500um'] >= .55 and
    summary['best8']['step500_recall1_1500um'] -
        summary['best8']['step0_recall1_1500um'] >= .15 and
    all(summary['best8']['by_appearance_1500um'][appearance]['500'] >=
        summary['best8']['by_appearance_1500um'][appearance]['0'] - .05
        for appearance in ('raw', 'exact_black', 'imperfect_brush')))
print(json.dumps({'audit': 'PASS', 'training_sections': 1000, 'dev_sections': 177,
                  'correspondence_promising': summary['correspondence_promising']}))
