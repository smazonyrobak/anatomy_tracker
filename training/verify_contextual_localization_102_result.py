"""Independent raw-row audit of the frozen 102A development decision."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker/runs/contextual_localization_102_fresh_development_eval')
completed = json.loads((root / 'completed.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
rows = [json.loads(line) for line in (root / 'candidates.jsonl').open()]
for name in ('config', 'candidates', 'sections', 'summary'):
    path = root / (name + ('.jsonl' if name in ('candidates', 'sections') else '.json'))
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == completed[name + '_sha256']
assert completed['eligible_sections_per_step'] == summary['eligible_sections_per_step'] == 241
assert completed['candidates_per_step'] == summary['candidates_per_step'] == 241 * 14
assert len(rows) == 241 * 14 * 3
assert summary['first_pair_step0_099_logit_difference'] == 0
assert summary['first_pair_step0_099_decoded_coordinate_difference_um'] == 0
for step in (0, 2000, 4000):
    group = [row for row in rows if row['step'] == step]
    assert len(group) == 241 * 14
    assert len({row['section_id'] for row in group}) == 241
    assert all(sum(row['section_id'] == section for row in group) == 14
               for section in {row['section_id'] for row in group})


def plan_equal(group, field):
    sites = defaultdict(list)
    for row in group:
        if row[field] is not None:
            sites[row['synthetic_subject_plan_id'], row['section_id']].append(row[field])
    subjects = defaultdict(list)
    for (plan, _), values in sites.items():
        subjects[plan].append(np.mean(values))
    return float(np.mean([np.mean(values) for values in subjects.values()]))


strata = [('all', lambda row: True), ('physical_best', lambda row: row['physical_best'])]
strata += [(f'appearance/{mode}/{kind}',
            lambda row, mode=mode, kind=kind: row['appearance_mode'] == mode and
            (kind == 'all' or row['physical_best']))
           for mode in ('raw', 'exact_black', 'imperfect_brush')
           for kind in ('all', 'physical_best')]
strata += [(f'angle_family/{family}/{kind}',
            lambda row, family=family, kind=kind: row['angle_family'] == family and
            (kind == 'all' or row['physical_best']))
           for family in ('AP', 'DV', 'ML') for kind in ('all', 'physical_best')]
values = {}
for step in (0, 2000, 4000):
    for name, predicate in strata:
        group = [row for row in rows if row['step'] == step and predicate(row)]
        for field in ('decoded_correct_fraction', 'null_correct_fraction'):
            value = plan_equal(group, field)
            expected = summary['steps'][str(step)][name]['plan_equal_mean'][field]
            assert abs(value - expected) < 1e-10, (step, name, field, value, expected)
            values[step, name, field] = value
best_delta = values[4000, 'physical_best', 'decoded_correct_fraction'] - values[
    4000, 'physical_best', 'null_correct_fraction']
all_delta = values[4000, 'all', 'decoded_correct_fraction'] - values[
    4000, 'all', 'null_correct_fraction']
plans = {row['synthetic_subject_plan_id'] for row in rows}
assert len(plans) == 8
improved = {}
for name in ('all', 'physical_best'):
    wins = 0
    for plan in plans:
        arm = [row for row in rows if row['step'] == 4000 and
               row['synthetic_subject_plan_id'] == plan and
               (name == 'all' or row['physical_best'])]
        baseline = [row for row in rows if row['step'] == 0 and
                    row['synthetic_subject_plan_id'] == plan and
                    (name == 'all' or row['physical_best'])]
        wins += plan_equal(arm, 'decoded_correct_fraction') > plan_equal(
            baseline, 'decoded_correct_fraction')
    improved[name] = wins
assert improved == {'all': 8, 'physical_best': 8}
assert abs(best_delta - summary['feedback_gates']['4000']['best_minus_null']) < 1e-10
assert abs(all_delta - summary['feedback_gates']['4000']['all_minus_null']) < 1e-10
assert best_delta < 0 and all_delta < 0 and not summary['primary_feedback_gate_passed']
print(json.dumps({'eligible_sections': 241, 'candidates_per_step': 3374,
    'step0_parity': True, 'best_decoded': values[4000, 'physical_best', 'decoded_correct_fraction'],
    'best_no_shift': values[4000, 'physical_best', 'null_correct_fraction'],
    'all_decoded': values[4000, 'all', 'decoded_correct_fraction'],
    'all_no_shift': values[4000, 'all', 'null_correct_fraction'],
    'improved_plans_vs_099': improved,
    'primary_gate_passed': False}))
