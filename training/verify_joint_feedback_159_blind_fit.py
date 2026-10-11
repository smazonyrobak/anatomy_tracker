"""Independent read-only audit of the completed blind top-16 fit assay."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
repo = Path(__file__).resolve().parents[1]
run = root / 'runs/joint_feedback_159_blind_fit_dev_001'
train = root / 'runs/joint_feedback_158_train151_001'
panel = root / 'data/fresh_v4_pose_dev_panel_132'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
train_done = json.loads((train / 'completed.json').read_text())
panel_done = json.loads((panel / 'completed.json').read_text())
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'rows.jsonl') == done['rows_sha256']
assert sha(run / 'summary.json') == done['summary_sha256']
assert sha(repo / 'training/assay_joint_feedback_159_blind_fit.py') == config['source_sha256']
assert sha(train / 'completed.json') == config['training_completion_sha256']
assert sha(panel / 'completed.json') == config['panel_completion_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert len(rows) == done['rows'] == 192
assert not any(done[key] or config[key] for key in
    ('calibrated', 'final_animals_used', 'public_benchmark_used'))

records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
plans = sorted({r['synthetic_subject_plan_id'] for r in records})
selected = []
for plan in plans:
    eligible = [r for r in records if r['eligible'] and r['synthetic_subject_plan_id'] == plan]
    selected += sorted(eligible, key=lambda r:
        hashlib.sha256(r['section_id'].encode()).hexdigest())[:8]
assert len(plans) == 8 and config['section_ids'] == [r['section_id'] for r in selected]
checkpoints = [('baseline', 'feedback', 0),
               ('feedback_selected', 'feedback', train_done['arms']['feedback']['selected_step']),
               ('control_selected', 'detached_control',
                train_done['arms']['detached_control']['selected_step'])]
for label, arm, step in checkpoints:
    checkpoint = train / arm / f'joint_feedback_step_{step:05d}.pt'
    assert sha(checkpoint) == config['checkpoints'][label] == \
        train_done['arms'][arm]['checkpoint_sha256'][str(step)]
    group = [r for r in rows if r['checkpoint'] == label]
    assert [r['section_id'] for r in group] == config['section_ids']
    assert all(len(r['branch_ids']) == len(r['original_error_mm']) ==
               len(r['corrected_error_mm']) == len(r['fit_score_uncalibrated']) ==
               len(r['zero_intensity_score_uncalibrated']) == 16 for r in group)
    for row in group:
        original = np.asarray(row['original_error_mm'])
        corrected = np.asarray(row['corrected_error_mm'])
        score = np.asarray(row['fit_score_uncalibrated'])
        zero_score = np.asarray(row['zero_intensity_score_uncalibrated'])
        assert len(set(row['branch_ids'])) == 16
        assert math.isclose(row['direct_selected_mm'], original[0], abs_tol=1e-8)
        assert math.isclose(row['fit_selected_original_mm'], original[score.argmax()], abs_tol=1e-8)
        assert math.isclose(row['fit_selected_corrected_mm'], corrected[score.argmax()], abs_tol=1e-8)
        assert math.isclose(row['zero_intensity_selected_original_mm'],
                            original[zero_score.argmax()], abs_tol=1e-8)
        assert math.isclose(row['top16_oracle_mm'], original.min(), abs_tol=1e-8)
        assert row['direct_selected_near'] == (original[0] <= 1.5)
        assert row['fit_selected_near'] == (original[score.argmax()] <= 1.5)
        assert row['top16_oracle_near'] == (original.min() <= 1.5)
    result = summary[label]
    assert result['sections'] == 64 and result['synthetic_plans'] == 8
    for metric in ('direct_selected_mm', 'fit_selected_original_mm',
                   'fit_selected_corrected_mm', 'zero_intensity_selected_original_mm',
                   'top16_oracle_mm', 'direct_selected_near', 'fit_selected_near',
                   'top16_oracle_near'):
        expected = np.mean([np.mean([r[metric] for r in group if
            r['synthetic_plan_id'] == plan]) for plan in plans])
        assert math.isclose(result[metric], expected, abs_tol=1e-8)
    assert math.isclose(result['fit_selection_gain_mm'],
                        result['direct_selected_mm'] - result['fit_selected_original_mm'],
                        abs_tol=1e-8)
print(json.dumps({'verified': True, 'sections_per_checkpoint': 64,
                  'summary': summary}, indent=2))
