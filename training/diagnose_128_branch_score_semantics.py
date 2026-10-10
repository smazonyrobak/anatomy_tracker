"""Read-only diagnostic: give the learned matcher score to original actions too."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = [json.loads(line) for line in Path(
    'I:/AnatomyTracker/runs/joint_in_path_correspondence_128_dev_eval/synthetic_rows.jsonl'
).open()]
results = {}
for arm in ('control', 'treatment', 'geometry_only', 'atlas_intensity_zero',
            'source_swap'):
    by_plan = defaultdict(list)
    changed = 0
    for row in rows:
        reference = row['arms']['treatment'] if arm not in ('control', 'treatment') else row['arms'][arm]
        scored = row['treatment_ablations'][arm] if arm not in ('control', 'treatment') else reference
        branch = reference['beam_branch_ids']
        prior_slot = int(np.argmax(scored['input_scores']))
        match_slot = int(np.argmax(scored['corrected_scores']))
        changed += match_slot != prior_slot
        by_plan[row['synthetic_subject_plan_id']].append({
            'prior_original': reference['original_mapped_mm_by_branch_id'][branch[prior_slot]],
            'match_original': reference['original_mapped_mm_by_branch_id'][branch[match_slot]],
            'match_corrected': reference['corrected_mapped_mm_by_slot'][match_slot],
        })
    results[arm] = {
        'sections': sum(map(len, by_plan.values())),
        'changed_branch': changed,
        'plan_equal_mean_mm': {key: float(np.mean([
            np.mean([item[key] for item in plan]) for plan in by_plan.values()
        ])) for key in ('prior_original', 'match_original', 'match_corrected')},
        'plan_equal_capture_le_1p5': {key: float(np.mean([
            np.mean([item[key] <= 1.5 for item in plan]) for plan in by_plan.values()
        ])) for key in ('prior_original', 'match_original', 'match_corrected')},
    }
output = Path('I:/AnatomyTracker/runs/appearance_gap_129/branch_score_128.json')
output.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
