"""Frozen blind synthetic DEV gate and separate target-state oracle diagnostic for direct flow."""
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
from training.direct_flow_partial_atlas_fit_105 import DirectFlowPartialAtlasFit105
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.global_atlas_contrast_090 import rigid_points_090
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/DIRECT_FLOW_PARTIAL_ATLAS_FIT_105_PROTOCOL_20261009.md'
pose_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
fit_run = root / 'runs/spatial_verifier_094_pilot'
coherent_run = root / 'runs/coherent_partial_atlas_fit_104'
train_run = root / 'runs/direct_flow_partial_atlas_fit_105'
panel = root / 'data/cross_candidate_match_103_fresh_dev_panel_001'
out = root / 'runs/direct_flow_partial_atlas_fit_105_dev_eval'
steps = (0, 1000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def at_chart(surface, chart):
    count = surface.shape[1]
    grid = ((chart + .5 / 256) * 2 - 1)[None].expand(
        count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
        padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
pose_receipt = json.loads((pose_run / 'completed.json').read_text())
fit_receipt = json.loads((fit_run / 'completed.json').read_text())
coherent_receipt = json.loads((coherent_run / 'completed.json').read_text())
train_receipt = json.loads((train_run / 'completed.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
pose_file = pose_run / 'joint_step_02000.pt'
fit_file = fit_run / 'joint_step_20000.pt'
coherent_file = coherent_run / 'fit_step_01000.pt'
assert sha(protocol) == train_config['protocol_sha256'] == train_receipt['protocol_sha256']
assert sha(pose_file) == pose_receipt['checkpoint_sha256']['2000'] == train_config['pose_parent_sha256']
assert sha(fit_file) == fit_receipt['checkpoint_sha256']['20000']
assert sha(coherent_file) == coherent_receipt['checkpoint_sha256']['1000'] == train_config['fit_parent_sha256'] == train_receipt['fit_parent_sha256']
assert sha(coherent_run / 'completed.json') == train_config['fit_completed_sha256']
assert train_receipt['updates'] == train_config['updates'] == 4000
assert sha(train_run / 'config.json') == train_receipt['config_sha256']
assert sha(train_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train_run / 'training.jsonl') == train_receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert not any(train_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
checkpoint_files = {step: train_run / f'fit_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == train_receipt['checkpoint_sha256'][str(step)]
           for step, path in checkpoint_files.items())
records = [json.loads(line) for line in (panel / 'records.jsonl').open() if line.strip()]
assert len(records) == panel_receipt['physical_sections'] == 256
assert len({row['section_id'] for row in records}) == 256
eligible = [row for row in records if row['eligible']]
assert len(eligible) == panel_receipt['eligible']
plans = sorted({row['synthetic_subject_plan_id'] for row in eligible})
assert len(plans) == 8
selected = [row for plan in plans for row in sorted((row for row in eligible
    if row['synthetic_subject_plan_id'] == plan), key=lambda row:
    hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == 64
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
train_draws = [json.loads(line) for path in (pose_run / 'draws.jsonl',
    fit_run / 'draws.jsonl', train_run / 'draws.jsonl')
    for line in path.open() if line.strip()]
train_draws = [row for row in train_draws if row.get('used') and 'base_lineage' in row]
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in selected} & \
        {row['base_lineage'][key] for row in train_draws}
assert not {row['panel_physical_section_id'] for row in selected} & \
    {row['physical_section_id'] for row in train_draws}

pose_checkpoint = torch.load(pose_file, map_location='cpu', weights_only=True)
fit_checkpoint = torch.load(fit_file, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(pose_checkpoint['model'], strict=True)
fitters = {}
for step, path in checkpoint_files.items():
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == train_config
    fitter = DirectFlowPartialAtlasFit105(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
    fitter.load_state_dict(checkpoint['spatial_fit'], strict=True)
    fitters[step] = fitter
reference = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
reference.load_state_dict(fit_checkpoint['spatial_fit'], strict=True)
del pose_checkpoint, fit_checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
axis = torch.arange(32, device='cuda')
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
heldout = ((yy // 2 + xx // 2) % 2) == 1
candidate_rows, section_rows, oracle_rows = [], [], []

with torch.inference_mode():
    for index, record in enumerate(selected, 1):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            target_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            target_reflection = torch.from_numpy(arrays['reflection'].reshape(1, 1).copy()).cuda().long()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        pixels = valid.flatten().nonzero().flatten()
        pixels = pixels[torch.linspace(0, len(pixels) - 1, 512,
            device='cuda').round().long()]
        chart = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        target = truth.reshape(-1, 3)[pixels]
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior[:, :32].topk(8, -1).indices,
            prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1,
            (beam // 2)[..., None].expand(-1, -1, 12))
        angle = math.degrees(math.acos(min(1.,
            float(np.max(np.abs(record['plane_normal_ap_dv_ml']))))))
        common = {'section_id': record['section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'appearance_mode': record['appearance_mode'],
            'angle_family': ('AP', 'DV', 'ML')[int(np.argmax(np.abs(
                record['plane_normal_ap_dv_ml'])))],
            'nearest_axis_angle_deg': angle,
            'angle_bin': ('0-15' if angle < 15 else '15-30' if angle < 30
                else '30-45' if angle < 45 else '45-54.7'),
            'valid_fraction': record['valid_pixels'] / 65536,
            'panel_file_sha256': record['sha256'],
            'sample_indices_sha256': hashlib.sha256(
                pixels.cpu().numpy().astype('<i4').tobytes()).hexdigest()}
        rows = {step: [] for step in steps}
        for start in range(0, 14, 2):
            pair = beam[:, start:start + 2]
            reflected = pair % 2
            pair_state = state[:, start:start + 2]
            baseline_fit = reference(prediction['feature'], pair_state, reflected,
                atlas, offsets, weights, source_shape=(256, 256))
            for step, fitter in fitters.items():
                fitted = fitter(prediction['feature'], pair_state, reflected,
                    atlas, offsets, weights, source_shape=(256, 256))
                new_error = (at_chart(fitted['coherent_match_ccf_um'], chart)
                    - target[None]).norm(dim=-1).mean(-1)
                rigid_error = (rigid_points_090(fitted['coherent_state'],
                    reflected, chart)[0] - target[None]).norm(dim=-1).mean(-1)
                score = prior.gather(1, pair) - fitted['coherent_energy'] / \
                    fitter.log_fit_temperature.clamp(math.log(.3), math.log(3.)).exp()
                for local in range(2):
                    row = {**common, 'step': step, 'beam_slot': start + local,
                        'branch_id': int(pair[0, local]),
                        'prior_score': float(prior[0, pair[0, local]]),
                        'new_score': float(score[0, local]),
                        'new_map_error_um': float(new_error[local]),
                        'new_rigid_error_um': float(rigid_error[local]),
                        'coherent_energy': float(fitted['coherent_energy'][0, local]),
                        'quality': float(fitted['coherent_quality'][0, local]),
                        'deformation_cost': float(fitted['coherent_deformation_cost'][0, local]),
                        'strain_cost': float(fitted['coherent_strain_cost'][0, local]),
                        'identity_probability_mean': float(fitted['coherent_identity_probability'][0, local].mean()),
                        'flow_magnitude_cells_mean': float(fitted['coherent_offset_cells_xyz'][0, local]
                            .square().sum(0).sqrt().mean())}
                    if step == 0:
                        selected_prediction = {**prediction,
                            'state': baseline_fit['state'][:, local:local + 1],
                            'log_mass': prediction['log_mass'].gather(1,
                                pair[:, local:local + 1] // 2),
                            'reflection_logit': prediction['reflection_logit'].gather(1,
                                pair[:, local:local + 1] // 2)}
                        mapped = model.map(selected_prediction, offsets,
                            torch.zeros((1, 1), dtype=torch.long, device='cuda'),
                            reflected[:, local:local + 1], (96, 96), atlas,
                            weights, feature_side=96, source_shape=(256, 256),
                            spatial_evidence=baseline_fit['spatial_evidence'][:, local:local + 1])
                        old_error = (at_chart(mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                            - target[None]).norm(dim=-1).mean()
                        local_warp = mapped['local_displacement_um'] / 1000
                        dx = local_warp[..., 1:] - local_warp[..., :-1]
                        dy = local_warp[..., 1:, :] - local_warp[..., :-1, :]
                        warp_cost = (local_warp.square().mean() + .1 *
                            (dx.square().mean() + dy.square().mean()))
                        temperature = reference.log_fit_temperature.clamp(
                            math.log(.5), math.log(5.)).exp()
                        old_score = (prior[0, pair[0, local]] -
                            baseline_fit['fit_energy'][0, local] / temperature - .05 * warp_cost)
                        centre_support = baseline_fit['fine_match_support'][0, local, 112] >= .5
                        tissue = baseline_fit['validity_logit'][0].sigmoid()
                        weight = tissue * (tissue >= .5) * heldout
                        support_score = (centre_support * weight).sum() / weight.sum().clamp_min(1)
                        supported_count = (baseline_fit['fine_match_support'][0, local] >= .5).sum(0)
                        row.update({'old_map_error_um': float(old_error),
                            'old_score': float(old_score),
                            'support_score': float(support_score),
                            'mean_supported_bin_count': float((supported_count * weight).sum() /
                                weight.sum().clamp_min(1))})
                    rows[step].append(row)
        initial_rigid_error = float((rigid_points_090(target_state[:, None],
            target_reflection, chart)[0, 0] - target).norm(dim=-1).mean())
        for step, fitter in fitters.items():
            oracle_fit = fitter(prediction['feature'], target_state[:, None],
                target_reflection, atlas, offsets, weights, source_shape=(256, 256))
            oracle_map_error = float((at_chart(oracle_fit['coherent_match_ccf_um'], chart)[0]
                - target).norm(dim=-1).mean())
            oracle_rigid_error = float((rigid_points_090(oracle_fit['coherent_state'],
                target_reflection, chart)[0, 0] - target).norm(dim=-1).mean())
            oracle_rows.append({**common, 'step': step,
                'diagnostic_only_not_selectable': True,
                'input_target_state_rigid_error_um': initial_rigid_error,
                'fit_target_state_map_error_um': oracle_map_error,
                'fit_target_state_rigid_error_um': oracle_rigid_error,
                'identity_probability_mean': float(oracle_fit['coherent_identity_probability'][0, 0].mean()),
                'flow_magnitude_cells_mean': float(oracle_fit['coherent_offset_cells_xyz'][0, 0]
                    .square().sum(0).sqrt().mean())})
        assert all(len(rows[step]) == 14 for step in steps)
        baseline = rows[0]
        old_error = np.array([row['old_map_error_um'] for row in baseline])
        old_score = np.array([row['old_score'] for row in baseline])
        support_score = np.array([row['support_score'] for row in baseline])
        old_slot, support_slot = int(old_score.argmax()), int(support_score.argmax())
        best_slots = old_error.argsort()[:3]
        for step in steps:
            current = rows[step]
            new_score = np.array([row['new_score'] for row in current])
            new_error = np.array([row['new_map_error_um'] for row in current])
            rigid_error = np.array([row['new_rigid_error_um'] for row in current])
            new_slot = int(new_score.argmax())
            for row in current:
                row.update({'old_map_error_um': baseline[row['beam_slot']]['old_map_error_um'],
                    'old_score': baseline[row['beam_slot']]['old_score'],
                    'support_score': baseline[row['beam_slot']]['support_score'],
                    'mean_supported_bin_count': baseline[row['beam_slot']]['mean_supported_bin_count']})
                candidate_rows.append(row)
            rho = (float(spearmanr(new_score, -old_error).statistic)
                if np.std(new_score) > 1e-8 and np.std(old_error) > 1e-8 else None)
            section_rows.append({**common, 'step': step,
                'beam_branch_ids': beam[0].tolist(),
                'old_slot': old_slot, 'support_slot': support_slot, 'new_slot': new_slot,
                'old_selected_old_map_error_um': float(old_error[old_slot]),
                'support_selected_old_map_error_um': float(old_error[support_slot]),
                'new_selected_old_map_error_um': float(old_error[new_slot]),
                'new_selected_new_map_error_um': float(new_error[new_slot]),
                'oracle_old_map_error_um': float(old_error.min()),
                'top3_new_map_error_um': float(new_error[best_slots].mean()),
                'top3_new_rigid_error_um': float(rigid_error[best_slots].mean()),
                'top3_old_map_error_um': float(old_error[best_slots].mean()),
                'rho_new_vs_old_error': rho,
                'rho_old_vs_old_error': (float(spearmanr(old_score, -old_error).statistic)
                    if np.std(old_score) > 1e-8 else None),
                'rho_support_vs_old_error': (float(spearmanr(support_score, -old_error).statistic)
                    if np.std(support_score) > 1e-8 else None),
                'old_selected_mean_supported_bin_count':
                    baseline[old_slot]['mean_supported_bin_count']})
        if index % 16 == 0:
            print(json.dumps({'eligible_sections_done': index,
                'eligible_sections_total': len(selected)}), flush=True)


def aggregate(group):
    by_plan = {}
    fields = ('old_selected_old_map_error_um', 'support_selected_old_map_error_um',
        'new_selected_old_map_error_um', 'new_selected_new_map_error_um',
        'oracle_old_map_error_um', 'top3_new_map_error_um',
        'top3_new_rigid_error_um', 'top3_old_map_error_um')
    for plan in plans:
        rows = [row for row in group if row['synthetic_subject_plan_id'] == plan]
        if rows:
            by_plan[plan] = {'sections': len(rows), **{name: float(np.mean(
                [row[name] for row in rows]) / 1000) for name in fields}}
    means = {name: float(np.mean([row[name] for row in by_plan.values()]))
        for name in fields} if by_plan else {}
    rhos = {name: float(np.mean([row[name] for row in group
        if row[name] is not None])) for name in ('rho_new_vs_old_error',
        'rho_old_vs_old_error', 'rho_support_vs_old_error')
        if any(row[name] is not None for row in group)}
    return {'sections': len(group), 'plans': len(by_plan),
        'plan_equal_mm': means, 'mean_within_section_rho': rhos, 'by_plan': by_plan}


summary = {'sections_per_step': len(selected), 'ineligible_excluded':
    panel_receipt['ineligible'], 'candidates_per_step': 14 * len(selected),
    'steps': {}}
for step in steps:
    group = [row for row in section_rows if row['step'] == step]
    groups = {'all': group, 'raw': [row for row in group if row['appearance_mode'] == 'raw'],
        'brush': [row for row in group if row['appearance_mode'] == 'imperfect_brush'],
        'black': [row for row in group if row['appearance_mode'] == 'exact_black'],
        'heavy_oblique': [row for row in group if row['angle_bin'] == '45-54.7'],
        'low_support': [row for row in group if row['old_selected_mean_supported_bin_count'] < 96]}
    summary['steps'][str(step)] = {name: aggregate(rows) for name, rows in groups.items()}
    oracle_group = [row for row in oracle_rows if row['step'] == step]
    oracle_by_plan = {plan: {field: float(np.mean([row[field] for row in oracle_group
        if row['synthetic_subject_plan_id'] == plan]) / 1000) for field in (
        'input_target_state_rigid_error_um', 'fit_target_state_map_error_um',
        'fit_target_state_rigid_error_um')} for plan in plans}
    summary['steps'][str(step)]['oracle_target_state_diagnostic_only'] = {
        'sections': len(oracle_group), 'by_plan_mm': oracle_by_plan,
        'plan_equal_mm': {field: float(np.mean([entry[field] for entry in oracle_by_plan.values()]))
            for field in ('input_target_state_rigid_error_um',
                'fit_target_state_map_error_um', 'fit_target_state_rigid_error_um')}}
best_step = min(steps[1:], key=lambda step: (summary['steps'][str(step)]['all']
    ['plan_equal_mm']['new_selected_old_map_error_um'], step))
best = summary['steps'][str(best_step)]
all_mm = best['all']['plan_equal_mm']
all_rho = best['all']['mean_within_section_rho']
improved_plans = sum(plan_row['new_selected_old_map_error_um'] <
    plan_row['old_selected_old_map_error_um'] for plan_row in best['all']['by_plan'].values())
conditions = {
    'top3_map_beats_own_rigid_by_0.20_mm':
        all_mm['top3_new_rigid_error_um'] - all_mm['top3_new_map_error_um'] >= .2,
    'selected_old_map_beats_094_by_0.10_mm':
        all_mm['old_selected_old_map_error_um'] - all_mm['new_selected_old_map_error_um'] >= .1,
    'selected_old_map_beats_support_only_by_0.10_mm':
        all_mm['support_selected_old_map_error_um'] - all_mm['new_selected_old_map_error_um'] >= .1,
    'fit_ordering_exceeds_both_controls_by_0.05':
        all_rho['rho_new_vs_old_error'] >= max(all_rho['rho_old_vs_old_error'],
            all_rho['rho_support_vs_old_error']) + .05,
    'raw_brush_oblique_nonregression_0.20_mm': all(
        best[name]['sections'] > 0 and best[name]['plan_equal_mm']['new_selected_old_map_error_um'] <=
        best[name]['plan_equal_mm']['old_selected_old_map_error_um'] + .2
        for name in ('raw', 'brush', 'heavy_oblique')),
    'at_least_six_of_eight_plans_improve_old_selection': improved_plans >= 6}
summary['best_development_step'] = best_step
summary['plans_improved_over_old_selection'] = improved_plans
summary['pilot_gate'] = {'conditions': conditions, 'passed': all(conditions.values()),
    'interpretation': 'reused synthetic DEV only; positive gate requires fresh confirmation and real-animal testing'}
config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_direct_flow_partial_atlas_fit_105.py', 'direct_flow_partial_atlas_fit_105.py',
        'spatial_joint_fit_094.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_allen_atlas_binding_v6.py')},
    'train_completed_sha256': sha(train_run / 'completed.json'),
    'train_config_sha256': sha(train_run / 'config.json'),
    'reference_094_checkpoint_sha256': sha(fit_file),
    'coherent_104_parent_checkpoint_sha256': sha(coherent_file),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoint_files.items()},
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'record_rule': 'eight eligible sections per synthetic plan, smallest SHA256(section_id)',
    'physical_error': 'mean CCF distance at 512 deterministic valid observed pixels',
    'selection_comparator': 'frozen 094 mapped96 error of selected candidate, common to all rankers',
    'checkpoint_selection': 'best trained step by plan-equal frozen 094 mapped96 error of newly selected blind candidate',
    'oracle_target_state_candidate_excluded_from_selection': True,
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows),
                   ('oracle_targets.jsonl', oracle_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_step': len(selected), 'candidates_per_step': 14 * len(selected),
    'protocol_sha256': sha(protocol), 'source_sha256': config['source_sha256'],
    'train_completed_sha256': config['train_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'reference_094_checkpoint_sha256': config['reference_094_checkpoint_sha256'],
    'coherent_104_parent_checkpoint_sha256': config['coherent_104_parent_checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'oracle_targets_sha256': sha(out / 'oracle_targets.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'oracle_target_state_candidate_excluded_from_selection': True}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(selected),
    'best_development_step': best_step, 'pilot_gate': summary['pilot_gate']}), flush=True)
