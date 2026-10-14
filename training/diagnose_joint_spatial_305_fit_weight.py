"""Exploratory leave-one-synthetic-base-out fit-weight readout; no training."""
import hashlib
import json
from pathlib import Path

import numpy as np

run = Path('I:/AnatomyTracker/runs/joint_spatial_feedback_305_joint_004')
receipt = json.loads((run / 'completed.json').read_text(encoding='utf8'))
with (run / 'readouts.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['output_sha256']['readouts.jsonl']
entry = next(json.loads(line) for line in (run / 'readouts.jsonl').open(encoding='utf8')
             if json.loads(line)['step'] == 2000)
rows = entry['rows']
assert len(rows) == 64 and {row['base'] for row in rows} == set(range(56, 64))
weights = np.array([0., .125, .25, .5, 1., 2.])
scores = np.array([row['scores']['support_only'] for row in rows])
fit = np.array([np.array(row['scores']['full']) - row['scores']['support_only']
                for row in rows])
shuffled = np.array([np.array(row['scores']['shuffled']) - row['scores']['support_only']
                     for row in rows])
errors = np.array([row['mapped_error_mm'] for row in rows])
bases = np.array([row['base'] for row in rows])
choices = (scores[:, None] + weights[None, :, None] * fit[:, None]).argmax(-1)
selected = errors[np.arange(len(rows))[:, None], choices]
chosen_weights, heldout_errors, heldout_shuffled = [], [], []
for base in range(56, 64):
    train, test = bases != base, bases == base
    weight_index = int(selected[train].mean(0).argmin())
    chosen_weights.append(float(weights[weight_index]))
    heldout_errors.extend(selected[test, weight_index].tolist())
    shuffled_choice = (scores[test] + weights[weight_index] * shuffled[test]).argmax(-1)
    heldout_shuffled.extend(errors[test, shuffled_choice].tolist())
result = {'checkpoint_step': 2000, 'n': len(rows), 'eight_base_weight_choices': chosen_weights,
          'leave_base_out_full_mean_mm': float(np.mean(heldout_errors)),
          'same_weights_shuffled_mean_mm': float(np.mean(heldout_shuffled)),
          'direct_prior_mean_mm': float(np.mean([row['mapped_error_mm'][0] for row in rows])),
          'support_only_mean_mm': float(selected[:, 0].mean()),
          'full_fixed_weight_mean_mm': float(selected[:, 4].mean()),
          'in_sample_weight_means_mm': dict(zip(map(str, weights), selected.mean(0).tolist())),
          'note': 'TRAIN-inner panel and checkpoint already selected on this panel; diagnostic only'}
print(json.dumps(result, allow_nan=False))
