"""Reused synthetic DEV gate for atlas-conditioned recurrent rigid pose capture."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coarse_pose_updater_106 import CoarsePoseUpdater106
from training.global_atlas_contrast_090 import rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_POSE_UPDATER_106_PROTOCOL_20261009.md'
pose_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
fit_run = root / 'runs/spatial_verifier_094_pilot'
train_run = root / 'runs/coarse_pose_updater_106'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
out = root / 'runs/coarse_pose_updater_106_dev_eval'
pose_file = pose_run / 'joint_step_02000.pt'
fit_file = fit_run / 'joint_step_20000.pt'
steps = (0, 1000, 4000)
arms = ('atlas', 'support_only')
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mean_error(state, reflection, chart, truth, valid):
    error = (rigid_points_090(state, reflection, chart) - truth[None]).norm(dim=-1)
    return error[..., valid].mean(-1)[0]


def aggregate(rows, plan_ids):
    fields = ('best14_input_um', 'best14_iter1_um', 'best14_iter2_um',
        'prior_input_um', 'prior_iter1_um', 'prior_iter2_um',
        'selected_iter1_um', 'selected_iter2_um',
        'best_input_iter1_um', 'best_input_iter2_um',
        'mean14_input_um', 'mean14_iter1_um', 'mean14_iter2_um')
    by_plan = {}
    for plan in plan_ids:
        group = [row for row in rows if row['synthetic_subject_plan_id'] == plan]
        if group:
            by_plan[plan] = {'sections': len(group), **{field: float(np.mean(
                [row[field] for row in group]) / 1000) for field in fields}}
    rho = [row['score_error_rho'] for row in rows if row['score_error_rho'] is not None]
    return {'sections': len(rows), 'plans': len(by_plan), 'by_plan': by_plan,
        'plan_equal_mm': {field: float(np.mean([row[field] for row in by_plan.values()]))
            for field in fields} if by_plan else {},
        'mean_score_error_rho': float(np.mean(rho)) if rho else None}


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
pose_receipt = json.loads((pose_run / 'completed.json').read_text())
fit_receipt = json.loads((fit_run / 'completed.json').read_text())
train_receipt = json.loads((train_run / 'completed.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert sha(protocol) == train_config['protocol_sha256'] == train_receipt['protocol_sha256']
assert sha(pose_file) == pose_receipt['checkpoint_sha256']['2000'] == \
    train_config['pose_parent_sha256']
assert sha(fit_file) == fit_receipt['checkpoint_sha256']['20000'] == \
    train_config['fit_parent_sha256']
assert sha(train_run / 'config.json') == train_receipt['config_sha256']
assert sha(train_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train_run / 'training.jsonl') == train_receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in
    train_config['source_sha256'].items())
assert train_receipt['updates'] == train_config['updates'] == 4000
assert tuple(train_config['checkpoints']) == steps
assert not any(train_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
checkpoints = {step: train_run / f'pose_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == train_receipt['checkpoint_sha256'][str(step)]
    for step, path in checkpoints.items())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
assert sha(panel / 'protocol.json') == panel_receipt['protocol_sha256']
assert sha(root / 'data/v3_pose_capture_confirmation_plans_001/completed.json') == \
    panel_receipt['plan_completed_sha256'] == \
    panel_protocol['source_plan_completed_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(records) == panel_receipt['physical_sections'] == 256
assert len({row['section_id'] for row in records}) == 256
eligible = [row for row in records if row['eligible']]
assert len(eligible) == panel_receipt['eligible'] == 247
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plan_ids) == 8
selected = [row for plan in plan_ids for row in sorted((row for row in eligible
    if row['synthetic_subject_plan_id'] == plan),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == 64
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
train_draws = [json.loads(line) for path in (pose_run / 'draws.jsonl',
    fit_run / 'draws.jsonl', train_run / 'draws.jsonl') for line in path.open()]
train_draws = [row for row in train_draws if row.get('used') and 'base_lineage' in row]
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in selected} & \
        {row['base_lineage'][key] for row in train_draws}
assert not {row['panel_physical_section_id'] for row in selected} & \
    {row['physical_section_id'] for row in train_draws}

pose_checkpoint = torch.load(pose_file, map_location='cpu', weights_only=True)
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
pose.load_state_dict(pose_checkpoint['model'], strict=True)
del pose_checkpoint
models = {}
for step, path in checkpoints.items():
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == train_config
    assert checkpoint['calibrated'] is False
    assert set(checkpoint['models']) == set(arms)
    for arm in arms:
        model = CoarsePoseUpdater106(WholeSliceAtlasFeedback083(),
            SpatialJointFit094(WholeSliceAtlasFeedback083()).validity_head)
        model.load_state_dict(checkpoint['models'][arm], strict=True)
        models[(step, arm)] = model.cuda().eval().requires_grad_(False)
    del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
candidate_rows, section_rows, oracle_rows = [], [], []

with torch.inference_mode():
    for index, record in enumerate(selected, 1):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            target_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            target_reflection = torch.from_numpy(arrays['reflection'].reshape(1, 1).copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        chart = torch.stack((xx, yy), -1).reshape(-1, 2)
        valid16 = (F.interpolate(valid[:, None].float(), (16, 16),
            mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
        target = rigid_points_090(target_state[:, None], target_reflection,
            chart)[0, 0]
        prediction = pose.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior[:, :32].topk(8, -1).indices,
            prior[:, 32:].topk(6, -1).indices + 32), -1)
        initial = prediction['state'].gather(1,
            (beam // 2)[..., None].expand(-1, -1, 12))
        prior_slot = int(prior.gather(1, beam)[0].argmax())
        normal = np.abs(record['plane_normal_ap_dv_ml'])
        angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
        common = {'section_id': record['section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'appearance_mode': record['appearance_mode'],
            'angle_family': ('AP', 'DV', 'ML')[int(np.argmax(normal))],
            'nearest_axis_angle_deg': angle,
            'angle_bin': ('0-15' if angle < 15 else '15-30' if angle < 30
                else '30-45' if angle < 45 else '45-54.7'),
            'valid_fraction': record['valid_pixels'] / (side * side),
            'valid_coarse_sites': int(valid16.sum()),
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'valid16_mask_sha256': hashlib.sha256(
                valid16.cpu().numpy().astype('|u1').tobytes()).hexdigest()}
        baseline = mean_error(initial, beam % 2, chart, target, valid16)
        for step in steps:
            for arm in arms:
                model = models[(step, arm)]
                pair_rows = []
                for start in range(0, 14, 2):
                    branch = beam[:, start:start + 2]
                    reflected = branch % 2
                    result = model(prediction['feature'], initial[:, start:start + 2],
                        reflected, atlas, offsets, weights, source_shape=(side, side),
                        atlas_disabled=arm == 'support_only')
                    error = [mean_error(result['iteration_states'][:, iteration],
                        reflected, chart, target, valid16) for iteration in (1, 2)]
                    support = result['coarse_match_support'].amax(3).mean((-2, -1))[0]
                    tissue = F.adaptive_avg_pool2d(result['validity_logit'].sigmoid()[:, None],
                        (16, 16))[:, 0, None]
                    support_any = result['coarse_match_support'][:, 1].amax(2).clamp(0, 1)
                    reliability = result['coarse_reliability_logit'][:, 1].sigmoid()
                    q = (tissue * support_any * reliability).sum((-2, -1)) / \
                        tissue.sum((-2, -1)).clamp_min(1e-6)
                    score = prior.gather(1, branch) + 2 * q.clamp_min(1e-4).log()
                    for local in range(2):
                        slot = start + local
                        pair_rows.append({**common, 'step': step, 'arm': arm,
                            'beam_slot': slot, 'branch_id': int(branch[0, local]),
                            'prior_score': float(prior[0, branch[0, local]]),
                            'prior_selected': slot == prior_slot,
                            'blind_score': float(score[0, local]),
                            'reliability_coverage': float(q[0, local]),
                            'input_rigid_error_um': float(baseline[slot]),
                            'iter1_rigid_error_um': float(error[0][local]),
                            'iter2_rigid_error_um': float(error[1][local]),
                            'input_state': initial[0, slot].tolist(),
                            'iter1_state': result['iteration_states'][0, 1, local].tolist(),
                            'iter2_state': result['iteration_states'][0, 2, local].tolist(),
                            'iter1_atlas_support_fraction': float(support[0, local]),
                            'iter2_atlas_support_fraction': float(support[1, local])})
                assert len(pair_rows) == 14
                candidate_rows.extend(pair_rows)
                first = np.array([row['iter1_rigid_error_um'] for row in pair_rows])
                second = np.array([row['iter2_rigid_error_um'] for row in pair_rows])
                scores = np.array([row['blind_score'] for row in pair_rows])
                base = baseline.cpu().numpy()
                selected_slot = int(scores.argmax())
                best_input_slot = int(base.argmin())
                rho = (float(spearmanr(scores, -second).statistic)
                    if np.std(scores) > 1e-8 and np.std(second) > 1e-8 else None)
                section_rows.append({**common, 'step': step, 'arm': arm,
                    'beam_branch_ids': beam[0].tolist(), 'prior_slot': prior_slot,
                    'blind_selected_slot': selected_slot,
                    'score_error_rho': rho,
                    'best14_input_um': float(base.min()),
                    'best14_iter1_um': float(first.min()),
                    'best14_iter2_um': float(second.min()),
                    'prior_input_um': float(base[prior_slot]),
                    'prior_iter1_um': float(first[prior_slot]),
                    'prior_iter2_um': float(second[prior_slot]),
                    'selected_iter1_um': float(first[selected_slot]),
                    'selected_iter2_um': float(second[selected_slot]),
                    'best_input_iter1_um': float(first[best_input_slot]),
                    'best_input_iter2_um': float(second[best_input_slot]),
                    'mean14_input_um': float(base.mean()),
                    'mean14_iter1_um': float(first.mean()),
                    'mean14_iter2_um': float(second.mean()),
                    'best_input_slot': best_input_slot,
                    'best_iter1_slot': int(first.argmin()),
                    'best_iter2_slot': int(second.argmin())})
                oracle = model(prediction['feature'], target_state[:, None],
                    target_reflection, atlas, offsets, weights,
                    source_shape=(side, side), atlas_disabled=arm == 'support_only')
                oracle_rows.append({**common, 'step': step, 'arm': arm,
                    'diagnostic_only_not_selectable': True,
                    'input_target_state_rigid_error_um': float(mean_error(
                        target_state[:, None], target_reflection, chart, target, valid16)[0]),
                    'iter1_target_state_rigid_error_um': float(mean_error(
                        oracle['iteration_states'][:, 1], target_reflection,
                        chart, target, valid16)[0]),
                    'iter2_target_state_rigid_error_um': float(mean_error(
                        oracle['iteration_states'][:, 2], target_reflection,
                        chart, target, valid16)[0])})
        if index % 16 == 0:
            print(json.dumps({'eligible_sections_done': index,
                'eligible_sections_total': len(selected)}), flush=True)

assert len(candidate_rows) == 64 * len(steps) * len(arms) * 14
assert len(section_rows) == len(oracle_rows) == 64 * len(steps) * len(arms)
summary = {'panel_role': 'reused synthetic development panel, not untouched confirmation',
    'unit_note': 'candidate/section error keys ending _um are micrometres; plan_equal_mm and by_plan entries are millimetres',
    'sections_per_step_arm': 64, 'candidates_per_step_arm': 64 * 14,
    'ineligible_excluded': panel_receipt['ineligible'], 'steps': {}}
for step in steps:
    summary['steps'][str(step)] = {}
    for arm in arms:
        rows = [row for row in section_rows if row['step'] == step and row['arm'] == arm]
        groups = {'all': rows,
            'steep_ge_30': [row for row in rows if row['nearest_axis_angle_deg'] >= 30]}
        for mode in panel_protocol['modes']:
            groups[f'appearance/{mode}'] = [row for row in rows
                if row['appearance_mode'] == mode]
        for family in ('AP', 'DV', 'ML'):
            groups[f'angle_family/{family}'] = [row for row in rows
                if row['angle_family'] == family]
        for angle_bin in ('0-15', '15-30', '30-45', '45-54.7'):
            groups[f'angle_bin/{angle_bin}'] = [row for row in rows
                if row['angle_bin'] == angle_bin]
        group_summary = {name: aggregate(group, plan_ids) for name, group in groups.items()}
        oracle_group = [row for row in oracle_rows if row['step'] == step and row['arm'] == arm]
        oracle_by_plan = {plan: {field: float(np.mean([row[field] for row in oracle_group
            if row['synthetic_subject_plan_id'] == plan]) / 1000) for field in (
            'input_target_state_rigid_error_um', 'iter1_target_state_rigid_error_um',
            'iter2_target_state_rigid_error_um')} for plan in plan_ids}
        group_summary['oracle_target_state_diagnostic_only'] = {'sections': len(oracle_group),
            'by_plan_mm': oracle_by_plan, 'plan_equal_mm': {field: float(np.mean(
                [row[field] for row in oracle_by_plan.values()])) for field in (
                'input_target_state_rigid_error_um', 'iter1_target_state_rigid_error_um',
                'iter2_target_state_rigid_error_um')}}
        summary['steps'][str(step)][arm] = group_summary

best_step = min(steps, key=lambda step: (summary['steps'][str(step)]['atlas']['all']
    ['plan_equal_mm']['best14_iter2_um'], step))
atlas_arm = summary['steps'][str(best_step)]['atlas']
control_arm = summary['steps'][str(best_step)]['support_only']
atlas_all = atlas_arm['all']['plan_equal_mm']
control_all = control_arm['all']['plan_equal_mm']
paired_rho_rows = {arm: {row['section_id']: row for row in section_rows
    if row['step'] == best_step and row['arm'] == arm} for arm in arms}
paired_rho_gain = [paired_rho_rows['atlas'][section]['score_error_rho'] -
    paired_rho_rows['support_only'][section]['score_error_rho']
    for section in paired_rho_rows['atlas'] if
    paired_rho_rows['atlas'][section]['score_error_rho'] is not None and
    paired_rho_rows['support_only'][section]['score_error_rho'] is not None]
paired_rho_gain = float(np.mean(paired_rho_gain)) if paired_rho_gain else None
plan_improvement = {plan: atlas_arm['all']['by_plan'][plan]['best14_iter2_um'] <
    atlas_arm['all']['by_plan'][plan]['best14_input_um'] for plan in plan_ids}
strata = ('appearance/raw', 'appearance/imperfect_brush', 'steep_ge_30')
stratum_nonregression = {name: atlas_arm[name]['sections'] > 0 and
    atlas_arm[name]['plan_equal_mm']['selected_iter2_um'] <=
    atlas_arm[name]['plan_equal_mm']['prior_input_um'] + .2 for name in strata}
conditions = {
    'best14_corrected_gain_vs_input_at_least_0.50_mm':
        atlas_all['best14_input_um'] - atlas_all['best14_iter2_um'] >= .5,
    'same_best_input_atlas_gain_vs_support_only_at_least_0.20_mm':
        control_all['best_input_iter2_um'] - atlas_all['best_input_iter2_um'] >= .2,
    'blind_selected_gain_vs_prior_at_least_0.20_mm':
        atlas_all['prior_input_um'] - atlas_all['selected_iter2_um'] >= .2,
    'blind_selected_gain_vs_support_only_at_least_0.20_mm':
        control_all['selected_iter2_um'] - atlas_all['selected_iter2_um'] >= .2,
    'within_section_score_error_rho_gain_vs_support_only_at_least_0.05':
        paired_rho_gain is not None and paired_rho_gain >= .05,
    'best14_improves_in_at_least_six_of_eight_plans':
        sum(plan_improvement.values()) >= 6,
    'raw_brush_steep_blind_selection_nonregression_0.20_mm':
        all(stratum_nonregression.values()),
}
summary['best_development_step'] = best_step
summary['pilot_gate'] = {'conditions': conditions, 'passed': all(conditions.values()),
    'paired_score_error_rho_gain': paired_rho_gain,
    'plan_best14_improvement': plan_improvement,
    'stratum_blind_selection_nonregression': stratum_nonregression,
    'meaning': 'synthetic coordinate capture only; positive result needs fresh confirmation, fitting-feedback test, local map, and animal-level validation'}
config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_coarse_pose_updater_106.py', 'train_coarse_pose_updater_106.py',
        'coarse_pose_updater_106.py', 'arbitrary_plane_one_shot_model.py',
        'whole_slice_atlas_feedback_083.py', 'spatial_joint_fit_094.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'pose_parent_checkpoint_sha256': sha(pose_file),
    'fit_parent_checkpoint_sha256': sha(fit_file),
    'train_completed_sha256': sha(train_run / 'completed.json'),
    'train_config_sha256': sha(train_run / 'config.json'),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoints.items()},
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'record_rule': 'eight eligible sections per synthetic plan, smallest SHA256(section_id)',
    'physical_error': 'mean full-frame rigid CCF distance to the true cutting plane at the same valid 16x16 observed sites',
    'candidate_rule': 'frozen v3 pose step2000 natural prior top8 old plus top6 normal anchors; target state excluded',
    'blind_score': 'frozen pose prior + 2 log(validity-weighted atlas-support-any times reliability coverage, clamped at 1e-4)',
    'arms': {'atlas': 'correlation logits and atlas support',
        'support_only': 'correlation logits zeroed, atlas support retained'},
    'checkpoint_selection': 'lowest plan-equal atlas-arm best-of-14 iter2 error among 0/1000/4000, tie favors earlier step',
    'panel_role': 'reused identity-disjoint synthetic DEV, not fresh confirmation',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows),
                   ('oracle_targets.jsonl', oracle_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_step_arm': 64, 'candidates_per_step_arm': 64 * 14,
    'protocol_sha256': config['protocol_sha256'],
    'source_sha256': config['source_sha256'],
    'train_completed_sha256': config['train_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'oracle_targets_sha256': sha(out / 'oracle_targets.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False,
    'oracle_target_state_candidate_excluded_from_selection': True}, indent=2))
print(json.dumps({'event': 'complete', 'sections': 64,
    'best_development_step': best_step, 'pilot_gate': summary['pilot_gate']}), flush=True)
