"""CPU-only independent audit of the completed, frozen 305 joint run."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import torch


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/joint_spatial_feedback_305_joint_004'
POSE = ROOT / 'runs/joint_spatial_feedback_305_pose_002'
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_PROTOCOL_20261012.md'
AMENDMENT = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_SCORE_STABILITY_AMENDMENT_20261014.md'
STEPS = (0, 1000, 2000, 4000)
ARMS = ('direct_prior', 'full', 'support_only', 'shuffled')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equal(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


assert (RUN / 'completed.json').is_file(), 'Joint run is not complete; do not audit it'
receipt = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
expected_files = {'config.json', 'training_draws.jsonl', 'training.jsonl',
    'readouts.jsonl', 'summary.json'} | {f'joint_step_{step:05d}.pt' for step in STEPS}
assert set(receipt['output_sha256']) == expected_files
assert {path.name for path in RUN.iterdir()} == expected_files | {'completed.json'}
assert all(sha(RUN / name) == digest for name, digest in receipt['output_sha256'].items())
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
summary = json.loads((RUN / 'summary.json').read_text(encoding='utf8'))
assert receipt['summary'] == summary and receipt['calibrated'] is False
assert config['candidate'] == summary['candidate'] == 305
assert config['stage'] == summary['stage'] == 'joint_feedback'
assert config['seed'] == 20261012306 and config['side'] == 192
assert config['optimizer_batches'] == summary['optimizer_batches_completed'] == 4000
assert config['independent_sections_per_batch'] == 2
assert config['blind_train_candidates'] == 8 and config['blind_readout_candidates'] == 16
assert config['readouts'] == list(STEPS)
assert config['gradient_base_indices'] == list(range(56))
assert config['inner_base_indices'] == list(range(56, 64))
assert config['derivative_cap'] is None
assert config['protocol_sha256'] == sha(PROTOCOL)
assert config['score_stability_amendment_sha256'] == sha(AMENDMENT)
assert all(sha(SOURCE / name) == digest for name, digest in config['source_sha256'].items())
assert config['no_external_weights_features_or_pseudolabels'] is True
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert summary['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is summary['calibrated'] is False

pose_receipt = json.loads((POSE / 'completed.json').read_text(encoding='utf8'))
pose_summary = json.loads((POSE / 'summary.json').read_text(encoding='utf8'))
pose_config = json.loads((POSE / 'config.json').read_text(encoding='utf8'))
assert pose_receipt['summary'] == pose_summary and pose_summary['joint_stage_capture_gate']
assert pose_config['no_external_weights_features_or_pseudolabels'] is True
assert pose_config['resume_step'] == 2000
assert all(sha(Path(pose_config['resume_parent']) / name) == digest
           for name, digest in pose_config['resume_parent_sha256'].items())
assert config['pose_receipt_sha256'] == sha(POSE / 'completed.json')
assert config['inner_panel_sha256'] == sha(POSE / 'inner_panel.pt') == \
    pose_receipt['output_sha256']['inner_panel.pt']
pose_checkpoint = POSE / f"pose_step_{pose_summary['selected_step']:05d}.pt"
assert config['pose_parent'] == str(pose_checkpoint)
assert config['pose_parent_sha256'] == sha(pose_checkpoint) == \
    pose_receipt['output_sha256'][pose_checkpoint.name]
assert config['pose_parent_model_source_sha256'] == \
    pose_config['source_sha256']['arbitrary_plane_one_shot_model.py']
assert pose_config['synthetic_provenance'] == config['synthetic_provenance']
panel = torch.load(POSE / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64
panel_ids = [item['physical_section_id'] for item in panel]
assert len(set(panel_ids)) == 64
assert [(item['base'], item['slot']) for item in panel] == [
    (base, slot) for base in range(56, 64) for slot in range(8)]

draw_ids = [[] for _ in range(4001)]
draw_bases = [[] for _ in range(4001)]
attempts = [0] * 4001
accepted = [0] * 4001
next_step, next_slot = 1, 0
for attempt, line in enumerate((RUN / 'training_draws.jsonl').open(encoding='utf8'), 1):
    row = json.loads(line)
    assert (row['step'], row['slot'], row['attempt']) == (next_step, next_slot, attempt)
    assert 0 <= row['base'] < 56 and 0 <= row['variant'] < 64
    assert row['seed'] == config['seed'] + 3_000_000_000 + attempt
    provenance = row['provenance']
    virtual = row['base'] * 64 + row['variant']
    assert provenance['virtual_index'] == virtual and provenance['split'] == 'train'
    assert provenance['physical_section_id'] == \
        f"stream-v7-seed-{row['seed']}-row-0-virtual-{virtual}"
    assert provenance['postartifact_gauge_v1']['code_sha256'] == \
        config['source_sha256']['arbitrary_plane_postartifact_gauge_v1.py']
    assert provenance['postartifact_gauge_v1']['pose_supervision_eligible'] == row['accepted']
    assert provenance['one_shot_slide_artifacts_v9']['code_sha256'] == \
        config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v9.py']
    attempts[next_step] += 1
    if row['accepted']:
        assert provenance['postartifact_gauge_v1']['fit_applied']
        accepted[next_step] += 1
        draw_ids[next_step].append(provenance['physical_section_id'])
        draw_bases[next_step].append(row['base'])
        next_slot += 1
        if next_slot == 2:
            next_step, next_slot = next_step + 1, 0
assert (next_step, next_slot) == (4001, 0)

attempt_count = accepted_count = 0
for step, line in enumerate((RUN / 'training.jsonl').open(encoding='utf8'), 1):
    row = json.loads(line)
    assert step <= 4000 and row['step'] == step and accepted[step] == 2
    attempt_count += attempts[step]
    accepted_count += accepted[step]
    assert row['attempted'] == attempt_count and row['accepted'] == accepted_count == 2 * step
    assert len(row['images']) == 2 and math.isfinite(row['gradient_norm'])
    assert [image['physical_section_id'] for image in row['images']] == draw_ids[step]
    assert [image['base'] for image in row['images']] == draw_bases[step]
    assert [image['slot'] for image in row['images']] == [0, 1]
    for image in row['images']:
        assert all(math.isfinite(image[name]) for name in (
            'direct_loss', 'q_loss', 'rank_loss', 'blind_selected_risk_mm',
            'near_map_error_mm', 'refined_rigid_error_mm', 'local_cost',
            'exact_map_error_mm', 'jitter_map_error_mm', 'exact_pose_drift_mm',
            'jitter_pose_error_mm', 'teacher_loss'))
        assert len(image['jitter_local_update']) == 9
        assert all(math.isfinite(value) for value in image['jitter_local_update'])
    if step == 1:
        assert math.isfinite(row['fit_only_score_to_pose_gradient_norm'])
        assert math.isfinite(row['fit_only_score_to_warp_head_gradient_norm'])
        assert row['fit_only_score_to_pose_gradient_norm'] > 0
        assert row['fit_only_score_to_warp_head_gradient_norm'] > 0
assert step == 4000 and summary['attempted'] == attempt_count
assert summary['accepted'] == accepted_count == 8000

readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open(encoding='utf8')]
assert [readout['step'] for readout in readouts] == list(STEPS)
facts = {}
for readout in readouts:
    step, rows, recorded = readout['step'], readout['rows'], readout['summary']
    assert len(rows) == 64 and [row['physical_section_id'] for row in rows] == panel_ids
    assert [(row['base'], row['slot']) for row in rows] == [
        (item['base'], item['slot']) for item in panel]
    for row, item in zip(rows, panel):
        assert all(row[name] == item[name] for name in (
            'obliquity_deg', 'visible_fraction', 'damage_fraction', 'exterior_style'))
        assert len(row['branch_ids']) == len(set(row['branch_ids'])) == 16
        assert all(0 <= branch < 160 for branch in row['branch_ids'])
        assert all(len(row[name]) == 16 for name in (
            'prior_log', 'rigid_error_mm', 'mapped_error_mm', 'normal_error_deg', 'support_mean'))
        assert set(row['scores']) == {'full', 'support_only', 'shuffled'}
        assert all(len(row['scores'][arm]) == 16 for arm in row['scores'])
        assert all(math.isfinite(value) for name in (
            'prior_log', 'rigid_error_mm', 'mapped_error_mm', 'normal_error_deg', 'support_mean')
            for value in row[name])
        assert all(math.isfinite(value) for arm in row['scores'] for value in row['scores'][arm])
        assert all(math.isfinite(row[name]) for name in (
            'exact_pose_drift_mm', 'exact_map_error_mm', 'zero_warp_error_mm',
            'zero_warp_residual_rms_mm'))
        assert row['chosen'] == {'direct_prior': 0, **{
            arm: max(range(16), key=lambda i: row['scores'][arm][i])
            for arm in row['scores']}}
        near = min(range(16), key=lambda i: row['rigid_error_mm'][i])
        far = [i for i in range(16) if row['rigid_error_mm'][near] <= 1.5
            and row['rigid_error_mm'][i] >= max(2., row['rigid_error_mm'][near] + 1.)
            and abs(row['support_mean'][i] - row['support_mean'][near]) <= .10]
        far = max(far, key=lambda i: row['prior_log'][i]) if far else None
        assert row['pair'] == (None if far is None else [near, far])
    means = {arm: statistics.fmean(row['mapped_error_mm'][row['chosen'][arm]]
        for row in rows) for arm in ARMS}
    normals = {arm: statistics.fmean(row['normal_error_deg'][row['chosen'][arm]]
        for row in rows) for arm in ARMS}
    assert all(equal(recorded['mean_selected_mapped_error_mm'][arm], value)
        for arm, value in means.items())
    assert all(equal(recorded['mean_selected_unsigned_normal_error_deg'][arm], value)
        for arm, value in normals.items())
    base_means = {str(base): {arm: statistics.fmean(
        row['mapped_error_mm'][row['chosen'][arm]] for row in rows if row['base'] == base)
        for arm in ARMS} for base in range(56, 64)}
    assert all(equal(recorded['per_base'][base][arm], value)
        for base, arm_values in base_means.items() for arm, value in arm_values.items())
    pairs = [row for row in rows if row['pair'] is not None]
    wins = {arm: sum((row['scores'][arm][row['pair'][0]] - row['prior_log'][row['pair'][0]])
        > (row['scores'][arm][row['pair'][1]] - row['prior_log'][row['pair'][1]])
        for row in pairs) for arm in ('full', 'support_only')}
    assert recorded['matched_pairs'] == len(pairs)
    assert recorded['matched_pair_evidence_wins'] == wins
    strong = sorted(rows, key=lambda row: row['zero_warp_residual_rms_mm'], reverse=True)[:16]
    strong_zero = statistics.fmean(row['zero_warp_error_mm'] for row in strong)
    strong_map = statistics.fmean(row['exact_map_error_mm'] for row in strong)
    drift = statistics.fmean(row['exact_pose_drift_mm'] for row in rows)
    assert equal(recorded['strong_warp_zero_mean_mm'], strong_zero)
    assert equal(recorded['strong_warp_exact_map_mean_mm'], strong_map)
    assert equal(recorded['exact_pose_drift_mean_mm'], drift)
    flags = {'gain_direct_0p40mm': means['direct_prior'] - means['full'] >= .40,
        'gain_support_0p25mm': means['support_only'] - means['full'] >= .25,
        'gain_shuffle_0p25mm': means['shuffled'] - means['full'] >= .25,
        'bases_6of8_both': sum(base_means[str(base)]['full'] < min(
            base_means[str(base)]['support_only'], base_means[str(base)]['shuffled'])
            for base in range(56, 64)) >= 6,
        'at_least_40_pairs': len(pairs) >= 40,
        'full_pairs_70pct': bool(pairs) and wins['full'] >= .70 * len(pairs),
        'pair_advantage_10points': bool(pairs) and wins['full'] - wins['support_only']
            >= .10 * len(pairs),
        'exact_pose_drift_0p25mm': drift <= .25,
        'strong_warp_gain_10pct': strong_map <= .9 * strong_zero}
    assert recorded['gate'] == {**flags, 'pass': all(flags.values())}
    strata = {'obliquity_ge45': [row for row in rows if row['obliquity_deg'] >= 45],
        'visible_lt0p1': [row for row in rows if row['visible_fraction'] < .1],
        'damage_ge0p02': [row for row in rows if row['damage_fraction'] >= .02]}
    strata.update({style: [row for row in rows if row['exterior_style'] == style]
        for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
    assert set(recorded['strata']) == set(strata)
    for name, group in strata.items():
        actual = recorded['strata'][name]
        assert actual['n'] == len(group)
        assert actual['full_mapped_mean_mm'] is None if not group else equal(
            actual['full_mapped_mean_mm'], statistics.fmean(
                row['mapped_error_mm'][row['chosen']['full']] for row in group))
        assert actual['full_unsigned_normal_mean_deg'] is None if not group else equal(
            actual['full_unsigned_normal_mean_deg'], statistics.fmean(
                row['normal_error_deg'][row['chosen']['full']] for row in group))
    assert summary['readouts'][str(step)] == recorded
    facts[str(step)] = {'mean_selected_mapped_error_mm': means,
        'matched_pairs': len(pairs), 'matched_pair_evidence_wins': wins,
        'exact_pose_drift_mean_mm': drift,
        'strong_warp_gain_fraction': (strong_zero - strong_map) / strong_zero,
        'gate': recorded['gate']}

selected = min(STEPS, key=lambda step: (
    not summary['readouts'][str(step)]['gate']['pass'],
    summary['readouts'][str(step)]['mean_selected_mapped_error_mm']['full']))
assert summary['selected_step'] == selected
assert summary['gate_pass'] == summary['readouts'][str(selected)]['gate']['pass']
assert summary['stop_reason'] == 'optimizer_batch_ceiling'
print(json.dumps({'verified': True, 'selected_step': selected,
    'gate_pass': summary['gate_pass'], 'attempted': attempt_count,
    'accepted': accepted_count, 'readouts': facts}, allow_nan=False))
