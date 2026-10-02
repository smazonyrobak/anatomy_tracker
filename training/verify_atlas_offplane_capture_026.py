"""Independent frozen-control and summary audit for off-plane capture 026."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import spearmanr

root = Path('I:/AnatomyTracker')
out = root / 'runs/atlas_offplane_capture_026'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
control = root / 'runs/one_shot_all_modes_023'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((out / 'config.json').read_text())
completed = json.loads((out / 'completed.json').read_text())
assert config['cases'] == completed['rows'] == 185
assert config['atlas_planes_per_case'] == 9 and config['patches_per_plane'] == 288
assert config['query_points_per_case'] == 32 and config['local_radius_px'] == 64
assert not config['weights_updated'] and not config['calibrated'] and not config['public_benchmark_used']
assert sha(panel / 'completed.json') == config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == config['panel_records_sha256']
assert sha(control / 'completed.json') == config['control_completed_sha256']
assert sha(control / 'rows.jsonl') == config['control_rows_sha256']
assert sha(root / 'runs/one_shot_exposure_019/joint_step_18000.pt') == config['pose_checkpoint_sha256']
assert sha(root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt') == config['patch_checkpoint_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
for name in ('config', 'rows', 'summary'):
    assert sha(out / (name + ('.jsonl' if name == 'rows' else '.json'))) == completed[name + '_sha256']
panels = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']}
controls = {row['sha256']: row for row in map(json.loads, (control / 'rows.jsonl').open())}
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == len(panels) == len(controls) == 185
assert len({row['section_id'] for row in rows}) == 185
for row in rows:
    reference = panels[row['sha256']]
    baseline = controls[row['sha256']]
    assert reference['animal_id'] == row['animal_id']
    assert reference['appearance_mode'] == row['appearance_mode']
    assert row['choice'] == baseline['top8']
    physical = np.asarray(baseline['rigid_um'])[row['choice']]
    score = np.asarray(row['scores9'])
    assert score.shape == (9,) and np.isfinite(score).all()
    assert np.allclose(physical, row['rigid8_um'], atol=1e-6)
    assert row['selected_index'] == int(score[1:].argmax())
    assert abs(row['selected_rigid_um'] - physical[row['selected_index']]) < 1e-6
    assert abs(row['prior_rigid_um'] - physical[0]) < 1e-6
    assert abs(row['best8_rigid_um'] - physical.min()) < 1e-6
    assert row['true_win'] == bool(score[0] > score[1:].max())
    assert abs(row['score_error_spearman'] - spearmanr(score[1:], -physical).statistic) < 1e-9
    assert len(row['true_top1_um']) == len(row['true_top16_um']) == 32
    assert np.all(np.asarray(row['true_top16_um']) <= np.asarray(row['true_top1_um']) + .1)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
assert len(subjects) == summary['synthetic_subjects'] == 8
assert len(rows) == summary['sections'] == 185
for field in ('true_win', 'score_error_spearman', 'selected_rigid_um', 'prior_rigid_um',
              'best8_rigid_um', 'true_local_matched_mean_um', 'selected_local_matched_mean_um'):
    mean = np.mean([np.mean([row[field] for row in rows if row['synthetic_subject_plan_id'] == subject])
                    for subject in subjects])
    assert abs(mean - summary[field]) < 1e-6
for field in ('true_top1_um', 'true_top16_um'):
    mean = np.mean([np.mean([np.mean(np.asarray(row[field]) < 500) for row in rows
                             if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
    assert abs(mean - summary[field.replace('_um', '_500um_recall')]) < 1e-9
for mode, claimed in summary['mode_true_win'].items():
    subset = [row for row in rows if row['appearance_mode'] == mode]
    identities = sorted({row['synthetic_subject_plan_id'] for row in subset})
    mean = np.mean([np.mean([row['true_win'] for row in subset
                             if row['synthetic_subject_plan_id'] == subject]) for subject in identities])
    assert abs(mean - claimed) < 1e-9
assert summary['promising_gate'] == (summary['true_win'] >= .75 and
    summary['score_error_spearman'] > 0 and
    summary['prior_rigid_um'] - summary['selected_rigid_um'] >= 250)
print(json.dumps({'status': 'passed', 'cases': len(rows),
                  'promising_gate': summary['promising_gate']}))
