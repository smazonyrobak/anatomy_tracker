"""Independent receipt and arithmetic audit of the frozen 156 synthetic DEV assay."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
repo = Path(__file__).resolve().parents[1]
run = root / 'runs/joint_fitter_156_fresh_dev_assay_001'
panel = root / 'data/coarse_atlas_pose_150_fresh_dev_panel'
train = root / 'runs/joint_fitter_155_train151_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
panel_done = json.loads((panel / 'completed.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
records = {row['section_id']: row for row in
    (json.loads(line) for line in (panel / 'records.jsonl').open()) if row['eligible']}

for stem, extension in (('config', 'json'), ('summary', 'json'), ('rows', 'jsonl')):
    assert sha(run / f'{stem}.{extension}') == done[f'{stem}_sha256']
assert sha(repo / 'training/assay_joint_fitter_156_fresh_dev.py') == config['source_sha256']
for version, filename in (('155', 'joint_anatomy_model_153.py'),
                          ('156', 'joint_anatomy_model_156.py')):
    assert sha(repo / 'training' / filename) == config['model_source_sha256'][version]
assert sha(panel / 'completed.json') == config['panel_receipt_sha256']
assert sha(train / 'completed.json') == config['fitter_completion_sha256']
step = train_done['selected_step']
assert sha(train / f'joint_fitter_step_{step:05d}.pt') == config['checkpoint_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert len(records) == panel_done['eligible'] == 243
assert len(rows) == done['rows'] == 486
assert set(config['arms']) == {'155_full', '156_damped_full'}

case_sets = []
for arm in config['arms']:
    group = [row for row in rows if row['arm'] == arm]
    result = summary[arm]
    assert len(group) == result['n'] == 243
    assert len({row['synthetic_plan_id'] for row in group}) == result['synthetic_plans'] == 8
    assert len({row['section_id'] for row in group}) == len(group)
    case_sets.append({row['section_id'] for row in group})
    metrics = {'exact_drift_mm': np.mean([row['corrected_mm'][0] for row in group]),
        'near_initial_mm': np.mean([row['initial_mm'][1] for row in group]),
        'near_corrected_mm': np.mean([row['corrected_mm'][1] for row in group]),
        'near_dense_mm': np.mean([row['dense_mm'][1] for row in group]),
        'exact_energy_below_wrong': np.mean([row['exact_energy_below_wrong'] for row in group])}
    metrics['near_gain_mm'] = metrics['near_initial_mm'] - metrics['near_corrected_mm']
    for name, value in metrics.items():
        assert math.isclose(value, result[name], abs_tol=1e-8)
    assert result['feedback_gate_passed'] == (
        metrics['exact_drift_mm'] <= config['gate']['exact_drift_mm_at_most']
        and metrics['near_gain_mm'] >= config['gate']['near_gain_mm_at_least']
        and metrics['exact_energy_below_wrong'] >= config['gate']['exact_energy_below_wrong_at_least'])
    for row in group:
        record = records[row['section_id']]
        assert row['panel_file_sha256'] == record['sha256']
        assert row['synthetic_plan_id'] == record['synthetic_subject_plan_id']
        assert row['physical_section_id'] == record['panel_physical_section_id']
        assert len(row['initial_mm']) == len(row['corrected_mm']) == len(row['dense_mm']) == 3
        assert len(row['fit_energy_uncalibrated']) == 3
        assert math.isclose(row['exact_energy_below_wrong'], float(
            row['fit_energy_uncalibrated'][0] < row['fit_energy_uncalibrated'][2]))
assert case_sets[0] == case_sets[1] == set(records)
assert not any(done[key] or config[key] for key in
    ('calibrated', 'final_animals_used', 'public_benchmark_used'))
print(json.dumps({'verified': True, 'case_rows': len(rows), 'summary': summary}, indent=2))
