"""Independent read-only receipt and gate audit for the 104 DEV evaluation."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

root = Path('I:/AnatomyTracker')
train = root / 'runs/coherent_partial_atlas_fit_104'
evaluation = root / 'runs/coherent_partial_atlas_fit_104_dev_eval'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train_receipt = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
eval_receipt = json.loads((evaluation / 'completed.json').read_text())
eval_config = json.loads((evaluation / 'config.json').read_text())
summary = json.loads((evaluation / 'summary.json').read_text())
assert sha(train / 'config.json') == train_receipt['config_sha256']
assert sha(train / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train / 'training.jsonl') == train_receipt['training_sha256']
assert all(sha(root / 'agent_worktrees/joint_v6_integration/training' / name) == digest
           for name, digest in train_config['source_sha256'].items())
assert sha(evaluation / 'config.json') == eval_receipt['config_sha256']
assert sha(evaluation / 'candidates.jsonl') == eval_receipt['candidates_sha256']
assert sha(evaluation / 'sections.jsonl') == eval_receipt['sections_sha256']
assert sha(evaluation / 'summary.json') == eval_receipt['summary_sha256']
assert sha(train / 'completed.json') == eval_receipt['train_completed_sha256']
assert sha(root / 'data/cross_candidate_match_103_fresh_dev_panel_001/completed.json') == \
    eval_receipt['panel_completed_sha256']
assert all(sha(root / 'agent_worktrees/joint_v6_integration/training' / name) == digest
           for name, digest in eval_config['source_sha256'].items())
assert not any(eval_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used'))
steps = (0, 1000, 4000)
assert all(sha(train / f'fit_step_{step:05d}.pt') ==
           train_receipt['checkpoint_sha256'][str(step)] for step in steps)
candidates = [json.loads(line) for line in (evaluation / 'candidates.jsonl').open()]
sections = [json.loads(line) for line in (evaluation / 'sections.jsonl').open()]
assert len(candidates) == 64 * 14 * 3 and len(sections) == 64 * 3
assert len({row['section_id'] for row in sections}) == 64
assert len({row['animal_id'] for row in sections}) == 8
assert len({(row['section_id'], row['step']) for row in sections}) == 64 * 3
by_key = defaultdict(list)
for row in candidates:
    by_key[(row['section_id'], row['step'])].append(row)
assert all(sorted(row['beam_slot'] for row in rows) == list(range(14))
           for rows in by_key.values())
for section in sections:
    rows = sorted(by_key[(section['section_id'], section['step'])],
                  key=lambda row: row['beam_slot'])
    scores = np.array([row['new_score'] for row in rows])
    old = np.array([row['old_score'] for row in rows])
    support = np.array([row['support_score'] for row in rows])
    errors = np.array([row['old_map_error_um'] for row in rows])
    assert section['new_slot'] == int(scores.argmax())
    assert section['old_slot'] == int(old.argmax())
    assert section['support_slot'] == int(support.argmax())
    for slot, field in ((section['new_slot'], 'new_selected_old_map_error_um'),
                        (section['old_slot'], 'old_selected_old_map_error_um'),
                        (section['support_slot'], 'support_selected_old_map_error_um')):
        assert abs(section[field] - errors[slot]) < 1e-6
    assert abs(section['new_selected_new_map_error_um'] -
               rows[section['new_slot']]['new_map_error_um']) < 1e-6
    top = errors.argsort()[:3]
    assert abs(section['top3_new_map_error_um'] - np.mean(
        [rows[slot]['new_map_error_um'] for slot in top])) < 1e-6
    assert abs(section['top3_new_rigid_error_um'] - np.mean(
        [rows[slot]['new_rigid_error_um'] for slot in top])) < 1e-6
    assert abs(section['oracle_old_map_error_um'] - errors.min()) < 1e-6
    if section['rho_new_vs_old_error'] is not None:
        assert abs(section['rho_new_vs_old_error'] -
            spearmanr(scores, -errors).statistic) < 1e-9

fields = ('old_selected_old_map_error_um', 'support_selected_old_map_error_um',
    'new_selected_old_map_error_um', 'new_selected_new_map_error_um',
    'oracle_old_map_error_um', 'top3_new_map_error_um',
    'top3_new_rigid_error_um', 'top3_old_map_error_um')
for step in steps:
    rows = [row for row in sections if row['step'] == step]
    plans = sorted({row['synthetic_subject_plan_id'] for row in rows})
    assert len(plans) == 8 and all(sum(
        row['synthetic_subject_plan_id'] == plan for row in rows) == 8 for plan in plans)
    for field in fields:
        value = np.mean([np.mean([row[field] for row in rows
            if row['synthetic_subject_plan_id'] == plan]) for plan in plans]) / 1000
        assert abs(value - summary['steps'][str(step)]['all']['plan_equal_mm'][field]) < 1e-9
best_step = min(steps[1:], key=lambda step: (summary['steps'][str(step)]['all']
    ['plan_equal_mm']['new_selected_old_map_error_um'], step))
assert best_step == summary['best_development_step']
best = summary['steps'][str(best_step)]
mm = best['all']['plan_equal_mm']
rho = best['all']['mean_within_section_rho']
conditions = {
    'top3_map_beats_own_rigid_by_0.20_mm':
        mm['top3_new_rigid_error_um'] - mm['top3_new_map_error_um'] >= .2,
    'selected_old_map_beats_094_by_0.10_mm':
        mm['old_selected_old_map_error_um'] - mm['new_selected_old_map_error_um'] >= .1,
    'selected_old_map_beats_support_only_by_0.10_mm':
        mm['support_selected_old_map_error_um'] - mm['new_selected_old_map_error_um'] >= .1,
    'fit_ordering_exceeds_both_controls_by_0.05':
        rho['rho_new_vs_old_error'] >= max(rho['rho_old_vs_old_error'],
            rho['rho_support_vs_old_error']) + .05,
    'raw_brush_oblique_nonregression_0.20_mm': all(
        best[name]['sections'] > 0 and best[name]['plan_equal_mm']['new_selected_old_map_error_um'] <=
        best[name]['plan_equal_mm']['old_selected_old_map_error_um'] + .2
        for name in ('raw', 'brush', 'heavy_oblique'))}
assert conditions == summary['pilot_gate']['conditions']
assert all(conditions.values()) == summary['pilot_gate']['passed']
print(json.dumps({'verified': True, 'sections': 64,
    'best_development_step': best_step, 'pilot_gate': summary['pilot_gate']}, indent=2))
