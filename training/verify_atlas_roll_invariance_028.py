"""Independent lineage, baseline binding and DEV summary audit for 028."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

root = Path('I:/AnatomyTracker')
run = root / 'runs/atlas_roll_invariance_028'
evaluation = root / 'runs/atlas_roll_invariance_028_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((run / 'config.json').read_text())
completed = json.loads((run / 'completed.json').read_text())
assert config['updates'] == completed['updates'] == 3000
assert config['sections_per_batch'] == 2 and config['positive_points_per_section'] == 32
assert config['atlas_negatives_per_section'] == 32
assert completed['accepted_synthetic_sections'] == 6000 and completed['positive_pairs'] == 192000
assert config['random_initialization'] and not config['prior_model_weights_features_pseudolabels']
assert config['real_training_images'] == 0
assert not completed['calibrated'] and not completed['public_benchmark_used']
for name in ('config', 'draws', 'training'):
    path = run / ('config.json' if name == 'config' else name + '.jsonl')
    assert sha(path) == completed[name + '_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 6000
assert {(row['step'], row['slot']) for row in used} == {
    (step, slot) for step in range(1, 3001) for slot in range(2)}
development_animals = {row['animal_id'] for row in map(json.loads, (panel / 'records.jsonl').open())}
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train' for row in used)
assert not development_animals & {row['base_lineage']['animal_id'] for row in used}
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert [row['step'] for row in training] == list(range(1, 3001))
assert np.isfinite([row['loss'] for row in training]).all()
assert np.isfinite([row['gradient_norm'] for row in training]).all()
assert training[0]['gradient_norm'] > 0
checkpoints = [torch.load(run / f'patch_step_{step:05d}.pt', map_location='cpu',
                          weights_only=True) for step in (0, 1000, 2000, 3000)]
assert [checkpoint['step'] for checkpoint in checkpoints] == [0, 1000, 2000, 3000]
assert all(not checkpoint['calibrated'] and 'optimizer' in checkpoint and
           'subjects_rng' in checkpoint and 'rotation_rng' in checkpoint and
           'draw_seed' in checkpoint and 'torch_rng' in checkpoint and
           'cuda_rng' in checkpoint for checkpoint in checkpoints)
assert any(not torch.equal(value, checkpoints[-1]['model'][name])
           for name, value in checkpoints[0]['model'].items())
evaluation_config = json.loads((evaluation / 'config.json').read_text())
evaluation_completed = json.loads((evaluation / 'completed.json').read_text())
assert sha(run / 'completed.json') == evaluation_config['run_completed_sha256']
assert sha(run / 'config.json') == evaluation_config['run_config_sha256']
assert sha(panel / 'completed.json') == evaluation_config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == evaluation_config['panel_records_sha256']
prior_plane = root / 'runs/atlas_oriented_patch_025_development_eval/summary.json'
prior_angle = root / 'runs/atlas_angular_capture_027/summary.json'
assert sha(prior_plane) == evaluation_config['baseline_plane_summary_sha256']
assert sha(prior_angle) == evaluation_config['baseline_angle_summary_sha256']
assert evaluation_config['query_points_per_section'] == 32
assert evaluation_config['atlas_plane_candidates'] == 288
assert not evaluation_config['calibrated'] and not evaluation_config['public_benchmark_used']
for name, digest in evaluation_config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
for step, digest in evaluation_config['checkpoints_sha256'].items():
    assert sha(run / f'patch_step_{int(step):05d}.pt') == digest
for name in ('config', 'rows', 'summary'):
    assert sha(evaluation / (name + ('.jsonl' if name == 'rows' else '.json'))) == \
        evaluation_completed[name + '_sha256']
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
summary = json.loads((evaluation / 'summary.json').read_text())
assert len(rows) == evaluation_completed['rows'] == summary['rows'] == 740
assert sorted({row['step'] for row in rows}) == [0, 1000, 2000, 3000]
assert len({row['section_id'] for row in rows}) == 185
assert not summary['calibrated'] and not summary['public_benchmark_used']
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
assert len(subjects) == 8
condition_names = [condition[0] for condition in evaluation_config['conditions']]
for row in rows:
    assert set(row['angular_top1_500um']) == set(condition_names)
    assert 0 <= row['oracle_recall1_500um'] <= row['oracle_recall16_500um'] <= 1
    assert all(0 <= value <= 1 for value in row['angular_top1_500um'].values())
for group in summary['checkpoints']:
    selected = [row for row in rows if row['step'] == group['step']]
    assert len(selected) == group['sections'] == 185
    assert group['synthetic_subjects'] == 8
    for field in ('oracle_recall1_500um', 'oracle_recall16_500um'):
        mean = np.mean([np.mean([row[field] for row in selected
                                 if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
        assert abs(mean - group[field]) < 1e-9
    for mode, claimed in group['mode_oracle_recall1_500um'].items():
        subset = [row for row in selected if row['appearance_mode'] == mode]
        identities = sorted({row['synthetic_subject_plan_id'] for row in subset})
        value = np.mean([np.mean([row['oracle_recall1_500um'] for row in subset
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in identities])
        assert abs(value - claimed) < 1e-9
    for name in condition_names:
        value = np.mean([np.mean([row['angular_top1_500um'][name] for row in selected
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
        assert abs(value - group['angular_top1_500um'][name]) < 1e-9
    roll = np.mean([group['angular_top1_500um'][name] for name in ('roll_-30', 'roll_+30')])
    tilt = np.mean([group['angular_top1_500um'][name] for name in
                    ('tilt_u_-10', 'tilt_u_+10', 'tilt_v_-10', 'tilt_v_+10')])
    assert abs(roll - group['roll30_top1_500um']) < 1e-9
    assert abs(tilt - group['tilt10_top1_500um']) < 1e-9
    gate = bool(group['step'] > 0 and min(group['mode_oracle_recall1_500um'].values()) >= .8 and
        roll >= evaluation_config['baseline_025_roll30_top1_500um'] + .2 and
        tilt >= evaluation_config['baseline_025_tilt10_top1_500um'] - .1)
    assert gate == group['gate']
assert summary['matched_change_gate'] == any(group['gate'] for group in summary['checkpoints'][1:])
print(json.dumps({'status': 'passed', 'training_batches': len(training),
                  'synthetic_sections': len(used), 'development_rows': len(rows),
                  'matched_change_gate': summary['matched_change_gate']}))
