"""Independent read-only audit of the frozen 105 synthetic DEV gate."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
pose = root / 'runs/v3_one_pass_pose_capture_pilot_001'
reference = root / 'runs/spatial_verifier_094_pilot'
parent = root / 'runs/coherent_partial_atlas_fit_104'
train = root / 'runs/direct_flow_partial_atlas_fit_105'
panel = root / 'data/cross_candidate_match_103_fresh_dev_panel_001'
evaluation = root / 'runs/direct_flow_partial_atlas_fit_105_dev_eval'
protocol = source.parent / 'docs/publication/DIRECT_FLOW_PARTIAL_ATLAS_FIT_105_PROTOCOL_20261009.md'
steps = (0, 1000, 4000)
fields = ('old_selected_old_map_error_um', 'support_selected_old_map_error_um',
    'new_selected_old_map_error_um', 'new_selected_new_map_error_um',
    'oracle_old_map_error_um', 'top3_new_map_error_um',
    'top3_new_rigid_error_um', 'top3_old_map_error_um')
oracle_fields = ('input_target_state_rigid_error_um',
    'fit_target_state_map_error_um', 'fit_target_state_rigid_error_um')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train_receipt = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
eval_receipt = json.loads((evaluation / 'completed.json').read_text())
eval_config = json.loads((evaluation / 'config.json').read_text())
summary = json.loads((evaluation / 'summary.json').read_text())
pose_receipt = json.loads((pose / 'completed.json').read_text())
reference_receipt = json.loads((reference / 'completed.json').read_text())
parent_receipt = json.loads((parent / 'completed.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert root.drive.upper() == source.drive.upper() == 'I:'
assert sha(protocol) == train_receipt['protocol_sha256'] == eval_receipt['protocol_sha256']
assert sha(pose / 'joint_step_02000.pt') == pose_receipt['checkpoint_sha256']['2000'] == train_config['pose_parent_sha256']
assert sha(reference / 'joint_step_20000.pt') == reference_receipt['checkpoint_sha256']['20000'] == eval_config['reference_094_checkpoint_sha256'] == eval_receipt['reference_094_checkpoint_sha256']
assert sha(parent / 'fit_step_01000.pt') == parent_receipt['checkpoint_sha256']['1000'] == train_config['fit_parent_sha256'] == train_receipt['fit_parent_sha256'] == eval_config['coherent_104_parent_checkpoint_sha256'] == eval_receipt['coherent_104_parent_checkpoint_sha256']
assert sha(parent / 'completed.json') == train_config['fit_completed_sha256']
assert sha(train / 'completed.json') == eval_config['train_completed_sha256'] == eval_receipt['train_completed_sha256']
assert sha(train / 'config.json') == train_receipt['config_sha256'] == eval_config['train_config_sha256']
assert sha(train / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train / 'training.jsonl') == train_receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert all(sha(train / f'fit_step_{step:05d}.pt') ==
    train_receipt['checkpoint_sha256'][str(step)] == eval_config['checkpoint_sha256'][str(step)]
    for step in steps)
assert sha(panel / 'completed.json') == eval_config['panel_completed_sha256'] == eval_receipt['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256'] == eval_config['panel_records_sha256']
for name, field in (('config.json', 'config_sha256'), ('candidates.jsonl', 'candidates_sha256'),
                    ('sections.jsonl', 'sections_sha256'), ('oracle_targets.jsonl', 'oracle_targets_sha256'),
                    ('summary.json', 'summary_sha256')):
    assert sha(evaluation / name) == eval_receipt[field]
assert all(sha(source / name) == digest for name, digest in eval_config['source_sha256'].items())
assert eval_config['oracle_target_state_candidate_excluded_from_selection']
assert eval_receipt['oracle_target_state_candidate_excluded_from_selection']
assert not any(train_receipt[key] or eval_receipt[key] for key in
    ('calibrated', 'public_benchmark_used', 'expert_real_truth_used'))
assert not train_receipt['external_pretrained_weights_used']

panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open() if line.strip()]
assert len(panel_rows) == panel_receipt['physical_sections'] == 256
eligible = [row for row in panel_rows if row['eligible']]
plans = sorted({row['synthetic_subject_plan_id'] for row in eligible})
assert len(plans) == 8
selected = [row for plan in plans for row in sorted(
    (row for row in eligible if row['synthetic_subject_plan_id'] == plan),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == 64 and len({row['section_id'] for row in selected}) == 64
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
selected_by_id = {row['section_id']: row for row in selected}
draws = [json.loads(line) for run in (pose, reference, parent, train)
    for line in (run / 'draws.jsonl').open() if line.strip()]
draws = [row for row in draws if row.get('used') and 'base_lineage' in row]
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in selected} & {row['base_lineage'][key] for row in draws}
assert not {row['panel_physical_section_id'] for row in selected} & {
    row['physical_section_id'] for row in draws}

candidates = [json.loads(line) for line in (evaluation / 'candidates.jsonl').open()]
sections = [json.loads(line) for line in (evaluation / 'sections.jsonl').open()]
oracles = [json.loads(line) for line in (evaluation / 'oracle_targets.jsonl').open()]
assert len(candidates) == 64 * 14 * 3 and len(sections) == len(oracles) == 64 * 3
assert eval_receipt['sections_per_step'] == 64 and eval_receipt['candidates_per_step'] == 64 * 14
assert {row['section_id'] for row in sections} == {row['section_id'] for row in selected}
assert len({row['animal_id'] for row in sections}) == 8
assert len({(row['section_id'], row['step']) for row in sections}) == 64 * 3
assert len({(row['section_id'], row['step']) for row in oracles}) == 64 * 3
oracle_by_key = {(row['section_id'], row['step']): row for row in oracles}
for row in sections:
    panel_row = selected_by_id[row['section_id']]
    oracle = oracle_by_key[(row['section_id'], row['step'])]
    for field in ('animal_id', 'specimen_id', 'experiment_id',
                  'synthetic_subject_plan_id', 'appearance_mode'):
        assert row[field] == oracle[field] == panel_row[field]
    assert row['panel_file_sha256'] == oracle['panel_file_sha256'] == panel_row['sha256']
assert all(row['diagnostic_only_not_selectable'] and 'beam_slot' not in row and
    'branch_id' not in row for row in oracles)
assert all('diagnostic_only_not_selectable' not in row and 'target_state' not in row
    for row in candidates)
by_key = defaultdict(list)
for row in candidates:
    by_key[(row['section_id'], row['step'])].append(row)
assert len(by_key) == 64 * 3
assert all(sorted(row['beam_slot'] for row in rows) == list(range(14))
    for rows in by_key.values())
assert all(len({row['branch_id'] for row in rows}) == 14
    for rows in by_key.values())
for section in sections:
    rows = sorted(by_key[(section['section_id'], section['step'])],
        key=lambda row: row['beam_slot'])
    scores = np.array([row['new_score'] for row in rows])
    old = np.array([row['old_score'] for row in rows])
    support = np.array([row['support_score'] for row in rows])
    errors = np.array([row['old_map_error_um'] for row in rows])
    assert [row['branch_id'] for row in rows] == section['beam_branch_ids']
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
    assert abs(section['top3_old_map_error_um'] - np.mean(errors[top])) < 1e-6
    assert abs(section['oracle_old_map_error_um'] - errors.min()) < 1e-6
    for name, values in (('rho_new_vs_old_error', scores),
                         ('rho_old_vs_old_error', old),
                         ('rho_support_vs_old_error', support)):
        actual = float(spearmanr(values, -errors).statistic) if np.std(values) > 1e-8 else None
        saved = section[name]
        assert (saved is None and actual is None) or abs(saved - actual) < 1e-9
    if section['step'] != 0:
        baseline = sorted(by_key[(section['section_id'], 0)],
            key=lambda row: row['beam_slot'])
        assert all(row['branch_id'] == base['branch_id'] and all(
            row[field] == base[field] for field in ('old_score', 'support_score',
                'old_map_error_um', 'mean_supported_bin_count'))
            for row, base in zip(rows, baseline))

for step in steps:
    step_rows = [row for row in sections if row['step'] == step]
    oracle_rows = [row for row in oracles if row['step'] == step]
    assert len(step_rows) == len(oracle_rows) == 64
    assert all(sum(row['synthetic_subject_plan_id'] == plan for row in step_rows) == 8
        for plan in plans)
    for field in fields:
        value = np.mean([np.mean([row[field] for row in step_rows
            if row['synthetic_subject_plan_id'] == plan]) for plan in plans]) / 1000
        assert abs(value - summary['steps'][str(step)]['all']['plan_equal_mm'][field]) < 1e-9
    for field in oracle_fields:
        value = np.mean([np.mean([row[field] for row in oracle_rows
            if row['synthetic_subject_plan_id'] == plan]) for plan in plans]) / 1000
        assert abs(value - summary['steps'][str(step)]['oracle_target_state_diagnostic_only']
            ['plan_equal_mm'][field]) < 1e-9
    for group, rows in (
        ('raw', [row for row in step_rows if row['appearance_mode'] == 'raw']),
        ('brush', [row for row in step_rows if row['appearance_mode'] == 'imperfect_brush']),
        ('black', [row for row in step_rows if row['appearance_mode'] == 'exact_black']),
        ('heavy_oblique', [row for row in step_rows if row['angle_bin'] == '45-54.7']),
        ('low_support', [row for row in step_rows if row['old_selected_mean_supported_bin_count'] < 96])):
        assert len(rows) == summary['steps'][str(step)][group]['sections']
        group_plans = sorted({row['synthetic_subject_plan_id'] for row in rows})
        for field in fields:
            if group_plans:
                value = np.mean([np.mean([row[field] for row in rows
                    if row['synthetic_subject_plan_id'] == plan]) for plan in group_plans]) / 1000
                assert abs(value - summary['steps'][str(step)][group]['plan_equal_mm'][field]) < 1e-9

best_step = min(steps[1:], key=lambda step: (np.mean([np.mean(
    [row['new_selected_old_map_error_um'] for row in sections
     if row['step'] == step and row['synthetic_subject_plan_id'] == plan])
    for plan in plans]), step))
assert best_step == summary['best_development_step']
best_rows = [row for row in sections if row['step'] == best_step]
best = summary['steps'][str(best_step)]
mm = best['all']['plan_equal_mm']
rho = best['all']['mean_within_section_rho']
for field in ('rho_new_vs_old_error', 'rho_old_vs_old_error', 'rho_support_vs_old_error'):
    value = np.mean([row[field] for row in best_rows if row[field] is not None])
    assert abs(value - rho[field]) < 1e-9
improved = sum(np.mean([row['new_selected_old_map_error_um'] for row in best_rows
    if row['synthetic_subject_plan_id'] == plan]) <
    np.mean([row['old_selected_old_map_error_um'] for row in best_rows
    if row['synthetic_subject_plan_id'] == plan]) for plan in plans)
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
        for name in ('raw', 'brush', 'heavy_oblique')),
    'at_least_six_of_eight_plans_improve_old_selection': improved >= 6}
assert improved == summary['plans_improved_over_old_selection']
assert conditions == summary['pilot_gate']['conditions']
assert all(conditions.values()) == summary['pilot_gate']['passed']
print(json.dumps({'verified': True, 'sections': 64,
    'candidate_rows': len(candidates), 'oracle_rows': len(oracles),
    'best_development_step': best_step, 'pilot_gate': summary['pilot_gate']}, indent=2))
