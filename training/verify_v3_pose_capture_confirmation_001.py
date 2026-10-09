"""Independent raw-row and frozen-binding audit for confirmation 001."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
repository = Path(__file__).resolve().parents[1]
run = root / 'runs/v3_pose_capture_confirmation_001_eval'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
train = root / 'runs/v3_one_pass_pose_capture_pilot_001'
selection = root / 'runs/v3_one_pass_pose_capture_pilot_001_development_eval'
config = json.loads((run / 'config.json').read_text())
receipt = json.loads((run / 'completed.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
sections = [json.loads(line) for line in (run / 'sections.jsonl').open()]
candidates = [json.loads(line) for line in (run / 'candidates.jsonl').open()]

for name in ('config', 'summary', 'sections', 'candidates'):
    path = run / (name + ('.jsonl' if name in ('sections', 'candidates') else '.json'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == receipt[f'{name}_sha256']
assert hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest() == \
    config['panel_completed_sha256'] == receipt['panel_completed_sha256']
assert hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest() == \
    config['panel_records_sha256'] == panel_receipt['records_sha256']
assert hashlib.sha256((train / 'completed.json').read_bytes()).hexdigest() == \
    config['train_completed_sha256'] == receipt['train_completed_sha256']
assert hashlib.sha256((selection / 'completed.json').read_bytes()).hexdigest() == \
    config['development_selection_completed_sha256'] == receipt['development_selection_completed_sha256']
assert all(hashlib.sha256((repository / 'training' / name).read_bytes()).hexdigest() == digest
           for name, digest in config['source_sha256'].items())
assert hashlib.sha256((repository / 'docs/publication/V3_POSE_CAPTURE_CONFIRMATION_001_PROTOCOL_20261009.md').read_bytes()).hexdigest() == \
    config['protocol_sha256'] == receipt['protocol_sha256']
for step, digest in config['checkpoint_sha256'].items():
    with (train / f'joint_step_{int(step):05d}.pt').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
assert receipt['checkpoint_sha256'] == config['checkpoint_sha256']
assert receipt['step0_parent_weight_equal'] and not any(receipt[key] for key in
    ('calibrated', 'public_benchmark_used', 'expert_real_truth_used', 'final_animals_used'))

eligible = {row['section_id']: row for row in records if row['eligible']}
assert len(records) == 256 and len(eligible) == panel_receipt['eligible']
assert len(sections) == 2 * len(eligible) == 2 * receipt['eligible_sections_per_step']
assert len(candidates) == 14 * len(sections) == 2 * receipt['candidates_per_step']
by_case = defaultdict(list)
for row in candidates:
    by_case[row['step'], row['section_id']].append(row)
assert set(by_case) == {(step, section_id) for step in (0, 2000) for section_id in eligible}
section_by_case = {(row['step'], row['section_id']): row for row in sections}
assert len(section_by_case) == len(sections) and set(section_by_case) == set(by_case)
for case, rows in by_case.items():
    section = section_by_case[case]
    assert len(rows) == 14 and sorted(row['beam_slot'] for row in rows) == list(range(14))
    assert len({row['branch_id'] for row in rows}) == 14
    assert len([row for row in rows if row['score_selected']]) == 1
    assert len([row for row in rows if row['physical_best']]) == 1
    selected = max(rows, key=lambda row: row['old094_score'])
    best = min(rows, key=lambda row: row['mapped96_error_um'])
    assert selected['score_selected'] and best['physical_best']
    assert section['score_selected_slot'] == selected['beam_slot']
    assert section['mapped_best_slot'] == best['beam_slot']
    for key, value in (('prefit_rigid_best14_um', min(row['prefit_rigid_error_um'] for row in rows)),
                       ('mapped_best14_um', best['mapped96_error_um']),
                       ('score_selected_mapped96_um', selected['mapped96_error_um']),
                       ('selection_regret_um', selected['mapped96_error_um'] - best['mapped96_error_um'])):
        assert np.isclose(section[key], value, rtol=1e-6, atol=1e-3)
    assert section['synthetic_subject_plan_id'] == eligible[case[1]]['synthetic_subject_plan_id']

plans = sorted({row['synthetic_subject_plan_id'] for row in eligible.values()})
assert len(plans) == 8
for step in (0, 2000):
    for name, subset in (('all', [row for row in sections if row['step'] == step]),
                         ('appearance/raw', [row for row in sections if row['step'] == step and row['appearance_mode'] == 'raw']),
                         ('appearance/exact_black', [row for row in sections if row['step'] == step and row['appearance_mode'] == 'exact_black']),
                         ('appearance/imperfect_brush', [row for row in sections if row['step'] == step and row['appearance_mode'] == 'imperfect_brush']),
                         *((f'angle_family/{axis}', [row for row in sections if row['step'] == step and row['angle_family'] == axis])
                           for axis in ('AP', 'DV', 'ML'))):
        stated = summary['steps'][str(step)]['synthetic'][name]
        assert stated['sections'] == len(subset)
        for metric in ('mapped_best14_um', 'score_selected_mapped96_um'):
            means = {plan: float(np.mean([row[metric] for row in subset
                                           if row['synthetic_subject_plan_id'] == plan]))
                     for plan in plans if any(row['synthetic_subject_plan_id'] == plan for row in subset)}
            assert set(means) == set(stated['by_plan'])
            assert all(np.isclose(value, stated['by_plan'][plan][metric], rtol=1e-8)
                       for plan, value in means.items())
            assert np.isclose(np.mean(list(means.values())), stated['plan_equal_mean'][metric], rtol=1e-8)

zero = summary['steps']['0']['synthetic']
current = summary['steps']['2000']['synthetic']
raw_gain = (zero['appearance/raw']['plan_equal_mean']['mapped_best14_um'] -
            current['appearance/raw']['plan_equal_mean']['mapped_best14_um'])
improved = {plan: current['appearance/raw']['by_plan'][plan]['mapped_best14_um'] <
            zero['appearance/raw']['by_plan'][plan]['mapped_best14_um'] for plan in plans}
nonregression = {name: current[name]['plan_equal_mean']['score_selected_mapped96_um'] <=
                 zero[name]['plan_equal_mean']['score_selected_mapped96_um'] + 200
                 for name in ('appearance/exact_black', 'appearance/imperfect_brush',
                              'angle_family/AP', 'angle_family/DV', 'angle_family/ML')}
conditions = {'raw_best14_gain_at_least_0.35_mm': raw_gain >= 350,
              'raw_best14_improves_in_at_least_six_plans': sum(improved.values()) >= 6,
              'overall_score_selected_nonregression':
                  current['all']['plan_equal_mean']['score_selected_mapped96_um'] <=
                  zero['all']['plan_equal_mean']['score_selected_mapped96_um'],
              'no_brush_or_axis_family_regression_over_0.20_mm': all(nonregression.values())}
gate = summary['confirmation_gate']
assert np.isclose(gate['raw_best14_gain_um'], raw_gain, rtol=1e-8)
assert gate['raw_plan_improves'] == improved
assert gate['brush_and_axis_family_nonregression'] == nonregression
assert gate['conditions'] == conditions and gate['passed'] == all(conditions.values())
assert gate['status'] == ('confirmed_synthetic_only' if gate['passed'] else 'failed')
print(json.dumps({'eligible': len(eligible), 'raw_best14_gain_mm': raw_gain / 1000,
                  'gate': gate['status'], 'conditions': conditions}))
