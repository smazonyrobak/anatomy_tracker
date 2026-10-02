"""Read-only spatial-ranker selection and score/error diagnostic on fixed DEV rows."""
import json
from pathlib import Path

import numpy as np

rows = [row for row in map(json.loads, Path(
    'I:/AnatomyTracker/runs/one_shot_correspondence_021_development_eval/rows.jsonl').open())
    if row['set'] == 'synthetic']
for step in (0, 500, 1000, 1500, 2000):
    group = [row for row in rows if row['step'] == step]
    old = np.asarray([row['old_mapped_um'] for row in group])
    new = np.asarray([row['new_mapped_um'] for row in group])
    score_correlation = np.asarray([np.corrcoef(row['new_score'],
                                               -np.asarray(row['candidate_error_um']))[0, 1]
                                    for row in group])
    residual_correlation = np.asarray([np.corrcoef(
        np.asarray(row['new_score']) - np.asarray(row['old_score']),
        -np.asarray(row['candidate_error_um']))[0, 1] for row in group])
    changed = old != new
    cutoff = np.quantile([row['valid_fraction'] for row in group], .25)
    low = np.asarray([row['valid_fraction'] <= cutoff for row in group])
    print(json.dumps({'step': step, 'changed_cases': int(changed.sum()),
                      'improved_among_changed': int(((new < old) & changed).sum()),
                      'worsened_among_changed': int(((new > old) & changed).sum()),
                      'mean_change_um': float((new - old).mean()),
                      'low_support_change_um': float((new[low] - old[low]).mean()),
                      'high_support_change_um': float((new[~low] - old[~low]).mean()),
                      'score_error_correlation': float(np.nanmean(score_correlation)),
                      'new_residual_error_correlation': float(np.nanmean(residual_correlation))}))
