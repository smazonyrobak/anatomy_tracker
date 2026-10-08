"""Exploratory frozen-101 readout: user-supplied rough cut family, no new model."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = [json.loads(line) for line in Path(
    'I:/AnatomyTracker/runs/matcher_swap_101_development_diagnostic/candidates.jsonl'
).open()]
axis = {'coronal': 0, 'horizontal': 1, 'sagittal': 2}
groups = defaultdict(list)
for row in rows:
    state = np.asarray(row['fitted_state'])
    normal = np.cross(state[3:6], state[6:9])
    normal /= np.linalg.norm(normal)
    row['candidate_family'] = ('coronal', 'horizontal', 'sagittal')[np.abs(normal).argmax()]
    row['angle_to_supplied_family_deg'] = np.degrees(np.arccos(np.clip(
        abs(normal[axis[row['angle_family']]]), 0, 1)))
    groups[(row['section_id'], row['step'])].append(row)

results = defaultdict(lambda: defaultdict(list))
for (_, step), candidates in groups.items():
    assert len(candidates) == 14 and sum(r['score_selected'] for r in candidates) == 1
    selected = next(r for r in candidates if r['score_selected'])
    family = candidates[0]['angle_family']
    strict = [r for r in candidates if r['candidate_family'] == family]
    cone = [r for r in candidates if r['angle_to_supplied_family_deg'] <= 60]
    plan = candidates[0]['synthetic_subject_plan_id']
    result = {'unconstrained': selected['mapped96_error_um'],
        'family': max(strict, key=lambda r: r['score'])['mapped96_error_um'] if strict else None,
        'cone60': max(cone, key=lambda r: r['score'])['mapped96_error_um'] if cone else None,
        'best14': min(r['mapped96_error_um'] for r in candidates),
        'true_family_candidates': len(strict), 'cone60_candidates': len(cone),
        'selected_outside_family': selected['candidate_family'] != family}
    results[step][plan].append(result)

summary = {}
for step, plans in sorted(results.items()):
    summary[step] = {'sections': sum(map(len, plans.values())), 'plans': len(plans)}
    for key in ('unconstrained', 'family', 'cone60', 'best14'):
        available = {plan: [row for row in group if row[key] is not None]
                     for plan, group in plans.items()}
        summary[step][key] = {'available_sections': sum(
            len(group) for group in available.values()),
            'plan_equal_mean_mm': float(np.mean([np.mean([row[key] for row in group]) / 1000
                for group in available.values() if group])),
            'same_sections_unconstrained_mm': float(np.mean([np.mean([
                row['unconstrained'] for row in group]) / 1000
                for group in available.values() if group]))}
    summary[step]['selected_outside_supplied_family'] = sum(
        row['selected_outside_family'] for group in plans.values() for row in group)
print(json.dumps(summary, indent=2))
