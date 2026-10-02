"""Read-only integrity and raw-ranking audit for the frozen MIND probe."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import spearmanr

root = Path('I:/AnatomyTracker')
out = root / 'runs/one_shot_mind_candidates_016'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
receipt = json.loads((out / 'completed.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not receipt['calibrated'] and not receipt['public_benchmark_used']
assert sha(root / 'runs/one_shot_observed_coordinate_pose_015/joint_step_08000.pt') == receipt['checkpoint_sha256']
assert sha(panel / 'records.jsonl') == receipt['panel_records_sha256']
assert sha(Path(__file__).parent / 'diagnose_one_shot_mind_candidates_016.py') == receipt['source_sha256']
assert sha(Path(__file__).parent / 'arbitrary_plane_image_information.py') == receipt['mind_source_sha256']
assert sha(out / 'rows.jsonl') == receipt['rows_sha256']
assert sha(out / 'summary.json') == receipt['summary_sha256']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == receipt['rows'] == 185
for row in rows:
    physical = np.asarray(row['physical_error_um'])
    assert physical.shape == (32,)
    assert row['prior_rigid_um'] == physical[row['prior_branch']]
    assert row['all32_oracle_rigid_um'] == physical.min()
    for name in ('whole', 'predicted', 'true_mask_upper_bound'):
        scores = np.asarray(row[f'{name}_scores'])
        assert scores.shape == (32,) and np.isfinite(scores).all()
        assert row[f'{name}_branch'] == int(scores.argmax())
        assert row[f'{name}_rigid_um'] == physical[scores.argmax()]
        assert row[f'{name}_maximum_score_ties'] == np.count_nonzero(scores == scores.max())
        correlation = spearmanr(scores, -physical).statistic
        saved = row[f'{name}_score_vs_negative_error_spearman']
        assert (saved is None and not np.isfinite(correlation)) or abs(saved - correlation) < 1e-12
identities = sorted({row['animal_id'] for row in rows})
assert len(identities) == 8
for name, group in summary.items():
    key = f'{name}_rigid_um'
    equal = np.mean([np.mean([row[key] for row in rows if row['animal_id'] == animal])
                     for animal in identities])
    partial = np.mean([row[key] for row in rows if row['valid_fraction'] < .05])
    assert abs(equal - group['plan_equal_mean_um']) < 1e-6
    assert abs(partial - group['under_5_percent_um']) < 1e-6
print(json.dumps({'status': 'passed', 'rows': len(rows),
                  'candidate_errors_and_scores': len(rows) * 32,
                  'summary_means': len(summary) * 2}))
