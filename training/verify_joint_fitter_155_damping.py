"""Read-only arithmetic audit of the frozen 155 DEV correction assay."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


root = Path('I:/AnatomyTracker')
run = root / 'runs/joint_fitter_155_damping_dev_assay_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
for name in ('config', 'rows', 'summary'):
    assert sha(run / f'{name}.{ "jsonl" if name == "rows" else "json"}') == done[f'{name}_sha256']
assert sha(Path(__file__).with_name('assay_joint_fitter_155_damping.py')) == config['source_sha256']
assert sha(Path(__file__).with_name('joint_anatomy_model_153.py')) == config['model_source_sha256']
assert sha(root / 'data/fresh_v4_pose_dev_panel_132/completed.json') == config['panel_receipt_sha256']
assert sha(root / 'runs/joint_pose_curriculum_154_train151_001/completed.json') == \
    config['parent_completion_sha256']
assert sha(root / 'runs/joint_fitter_155_train151_001/completed.json') == \
    config['fitter_completion_sha256']
assert len(rows) == done['rows'] == 192
case_sets = []
for arm in config['arms']:
    group = [row for row in rows if row['arm'] == arm]
    result = summary[arm]
    assert len(group) == result['n'] == 64
    assert len({row['synthetic_plan_id'] for row in group}) == result['synthetic_plans'] == 8
    case_sets.append({row['section_id'] for row in group})
    for gain in ('1.0', '0.8'):
        exact = np.mean([row['corrected_mm_by_gain'][gain][0] for row in group])
        near = np.mean([row['corrected_mm_by_gain'][gain][1] for row in group])
        baseline = np.mean([row['initial_mm'][1] for row in group])
        assert math.isclose(exact, result[f'exact_drift_mm_gain_{gain}'], abs_tol=1e-8)
        assert math.isclose(near, result[f'near_corrected_mm_gain_{gain}'], abs_tol=1e-8)
        assert math.isclose(baseline - near, result[f'near_gain_mm_gain_{gain}'], abs_tol=1e-8)
    rank = np.mean([row['exact_energy_below_wrong'] for row in group])
    assert math.isclose(rank, result['exact_energy_below_wrong'], abs_tol=1e-8)
assert case_sets[0] == case_sets[1] == case_sets[2]
for row in rows:
    assert len(row['initial_mm']) == len(row['fit_energy_uncalibrated']) == 3
    assert set(row['corrected_mm_by_gain']) == {'1.0', '0.8'}
    assert math.isclose(row['exact_energy_below_wrong'], float(
        row['fit_energy_uncalibrated'][0] < row['fit_energy_uncalibrated'][2]))
assert not any(done[key] or config[key] for key in
    ('calibrated', 'final_animals_used', 'public_benchmark_used'))
print(json.dumps({'verified': True, 'case_rows': len(rows), 'summary': summary}, indent=2))
