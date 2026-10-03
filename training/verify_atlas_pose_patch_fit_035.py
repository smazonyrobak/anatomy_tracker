"""Independently recompute the frozen 035 directional gate from raw rows."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker/runs/atlas_pose_patch_fit_035')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


receipt = json.loads((root / 'completed.json').read_text())
for name in ('config', 'summary'):
    assert sha(root / f'{name}.json') == receipt[f'{name}_sha256']
assert sha(root / 'rows.jsonl') == receipt['rows_sha256']
config = json.loads((root / 'config.json').read_text())
rows = [json.loads(line) for line in (root / 'rows.jsonl').open()]
summary = json.loads((root / 'summary.json').read_text())
assert len(rows) == summary['sections'] == config['sections'] == 24
assert len({row['synthetic_subject_plan_id'] for row in rows}) == 8
assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'], row['section_id'])
            for row in rows}) == 24
prior = np.array([row['prior_top1_um'] for row in rows])
selected = np.array([row['selected_um'] for row in rows])
oracle = np.array([row['oracle8_um'] for row in rows])
above = np.array([row['exact_pose_score'] > row['median_candidate_score'] for row in rows])
better = np.array([row['gradient_probe']['prior_top1']['after_um'] <
                   row['gradient_probe']['prior_top1']['before_um'] for row in rows])
for key, value in (('mean_prior_top1_um', prior.mean()), ('mean_selected_um', selected.mean()),
                   ('mean_oracle8_um', oracle.mean()), ('exact_score_above_median_fraction', above.mean()),
                   ('selected_better_than_prior_fraction', (selected < prior).mean()),
                   ('prior_top1_gradient_positive_fraction', better.mean())):
    assert math.isclose(summary[key], value, abs_tol=1e-6)
for name in ('prior_top1', 'oracle8', 'perturbed_truth'):
    mean = np.mean([row['gradient_probe'][name]['gradient_cosine'] for row in rows])
    assert math.isclose(summary['gradient_cosine_mean'][name], mean, abs_tol=1e-6)
for row in rows:
    assert len(row['candidate_scores']) == len(row['candidate_branches']) == 8
    assert row['selected_branch'] == row['candidate_branches'][int(np.argmax(row['candidate_scores']))]
    assert all(math.isfinite(v) for v in (row['prior_top1_um'], row['selected_um'],
               row['oracle8_um'], row['exact_pose_score']))
gate = selected.mean() <= prior.mean() - 250 and above.mean() >= .7 and better.mean() >= .6
assert gate == summary['development_gate']
print(json.dumps({'audit': 'PASS', 'gate': bool(gate), 'sections': len(rows),
                  'mean_selected_um': float(selected.mean()),
                  'prior_top1_gradient_positive_fraction': float(better.mean())}))
