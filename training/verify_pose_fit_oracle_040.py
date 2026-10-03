"""Independent read-only audit of the frozen 040 geometric-oracle diagnosis."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
out = root / 'runs/pose_feedback_3d_oracle_fit_040'
source = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((out / 'config.json').read_text())
receipt = json.loads((out / 'completed.json').read_text())
for field, file in (('config', 'config.json'), ('rows', 'rows.jsonl'),
                    ('summary', 'summary.json')):
    assert sha(out / file) == receipt[field + '_sha256']
assert config['source_sha256'] == sha(source / 'diagnose_pose_fit_oracle_040.py')
assert config['panel_records_sha256'] == sha(panel / 'records.jsonl')
assert sha(root / 'runs/one_shot_exposure_019/joint_step_18000.pt') == config['parent_sha256']
panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in panel_rows if row['eligible']]
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(panel_rows) == 256 and len(eligible) == len(rows) == receipt['rows'] == 177
assert len({row['synthetic_subject_plan_id'] for row in rows}) == summary['synthetic_subjects'] == 8
assert {row['sha256'] for row in rows} == {row['sha256'] for row in eligible}
assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'],
             row['section_id']) for row in rows}) == 177
groups = {row['synthetic_subject_plan_id'] for row in rows}
for field in ('prior_top1_um', 'prior_best8_um', 'continuous_oracle_top1_um',
              'continuous_oracle_best8_um', 'discrete_oracle_top1_um',
              'discrete_oracle_best8_um', 'top1_in_range_valid_fraction',
              'best8_in_range_valid_fraction'):
    mean = np.mean([np.mean([row[field] for row in rows
                            if row['synthetic_subject_plan_id'] == group]) for group in groups])
    assert math.isclose(summary[field], mean, abs_tol=1e-5)
    assert all(np.isfinite(row[field]) and row[field] >= 0 for row in rows)
print(json.dumps({'audit': 'PASS', 'eligible_sections': len(rows),
                  'discrete_oracle_top1_um': summary['discrete_oracle_top1_um'],
                  'discrete_oracle_best8_um': summary['discrete_oracle_best8_um']}))
