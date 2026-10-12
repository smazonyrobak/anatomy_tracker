"""Joint blind atlas-feedback continuation of the scratch-trained 305 model."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, UPDATES = 20261012306, 192, 4000
READOUTS = (0, 1000, 2000, 4000)
POSE_RUN = ROOT / 'runs/joint_spatial_feedback_305_pose_002'
RUN = ROOT / 'runs/joint_spatial_feedback_305_joint_001'
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_PROTOCOL_20261012.md'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


pose_receipt = json.loads((POSE_RUN / 'completed.json').read_text(encoding='utf8'))
pose_summary = json.loads((POSE_RUN / 'summary.json').read_text(encoding='utf8'))
pose_config = json.loads((POSE_RUN / 'config.json').read_text(encoding='utf8'))
assert pose_summary['joint_stage_capture_gate'] and pose_receipt['summary'] == pose_summary
assert pose_config['no_external_weights_features_or_pseudolabels']
assert pose_config['init'].startswith('randomly initialized in verified 305 pose run 001')
assert pose_config['resume_step'] == 2000
assert all(sha(Path(pose_config['resume_parent']) / name) == digest
    for name, digest in pose_config['resume_parent_sha256'].items())
assert pose_config['source_sha256']['arbitrary_plane_one_shot_model.py'] == sha(
    SOURCE / 'arbitrary_plane_one_shot_model.py')
assert pose_config['source_sha256']['arbitrary_plane_postartifact_gauge_v1.py'] == sha(
    SOURCE / 'arbitrary_plane_postartifact_gauge_v1.py')
assert pose_config['protocol_sha256'] == sha(PROTOCOL)
assert all(sha(POSE_RUN / name) == digest
    for name, digest in pose_receipt['output_sha256'].items())
POSE_CHECKPOINT = POSE_RUN / f"pose_step_{pose_summary['selected_step']:05d}.pt"
assert sha(POSE_CHECKPOINT) == pose_receipt['output_sha256'][POSE_CHECKPOINT.name]
assert pose_summary['selected_step'] in (0, 2000, 8000, 16000, 26000)
panel = torch.load(POSE_RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64 and len({row['physical_section_id'] for row in panel}) == 64
context = load_streaming_synthetic_v7_64('cuda')
assert pose_config['synthetic_provenance'] == context['provenance']
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
saved = torch.load(POSE_CHECKPOINT, map_location='cpu', weights_only=False)
assert saved['step'] == pose_summary['selected_step'] and not saved['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, vector_refinement=True, spatial_feedback=True).cuda()
model.load_state_dict(saved['model'], strict=True)
del saved
optimizer = torch.optim.AdamW([
    {'params': [parameter for name, parameter in model.named_parameters()
                if not name.startswith('pose_refiner.')], 'initial_lr': 1e-5},
    {'params': model.pose_refiner.parameters(), 'initial_lr': 1e-4},
], weight_decay=1e-4)
draw_rng = np.random.default_rng(SEED + 1)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
config = {'candidate': 305, 'stage': 'joint_feedback', 'seed': SEED,
    'side': SIDE, 'optimizer_batches': UPDATES, 'independent_sections_per_batch': 2,
    'blind_train_candidates': 8, 'blind_readout_candidates': 16,
    'first_map_side': 64, 'fitted_map_side': 96, 'readouts': READOUTS,
    'pose_parent': str(POSE_CHECKPOINT), 'pose_parent_sha256': sha(POSE_CHECKPOINT),
    'pose_receipt_sha256': sha(POSE_RUN / 'completed.json'),
    'inner_panel_sha256': sha(POSE_RUN / 'inner_panel.pt'),
    'protocol_sha256': sha(PROTOCOL), 'synthetic_provenance': context['provenance'],
    'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'first_map_gradient': 'stopped to bound memory; not fully end-to-end through first rendering',
    'second_map_score_gradient': 'enabled through refined state, atlas rendering, and local warp',
    'exact_and_jittered_pose_branches': 'training only; never candidates at inference',
    'same_candidate_controls': 'support-only and shuffled intensity change scoring, not the fitted candidate maps',
    'strong_warp_ordering': '16 highest truth-measured zero-warp rigid residual RMS on fixed panel points',
    'deployment_mapping': 'spatial-model inference uses the saved derivative_cap=None',
    'derivative_cap': None, 'per_image_candidate_accumulation': True,
    'blind_candidate_chunk': 2,
    'chunk_gradient': 'exact first-order rank and selected-risk gradients at unchanged same-batch weights',
    'initial_learning_rates': {'base': 1e-5, 'pose_refiner': 1e-4},
    'early_stop_repeated_regression_mm': .5,
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'train_joint_spatial_feedback_305.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_one_shot_inference.py',
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py')},
    'no_external_weights_features_or_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')


def surface_points(mapped, chart):
    batch, count, height, width = mapped['centre_surface_ccf_ap_dv_ml_um'].shape[:4]
    surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
    grid = ((chart + .5 / SIDE) * 2 - 1)[:, None].expand(-1, count, -1, -1)
    grid = grid.reshape(batch * count, 1, chart.shape[1], 2)
    return F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[:, :, 0].transpose(1, 2).reshape(
                             batch, count, chart.shape[1], 3)


def select_branch(prediction, branches):
    return {**prediction,
        'state': prediction['state'].gather(1, (branches // 2)[..., None].expand(-1, -1, 12)),
        'log_mass': prediction['log_mass'].gather(1, branches // 2),
        'reflection_logit': prediction['reflection_logit'].gather(1, branches // 2)}


attempted = accepted = 0
stop_reason = 'optimizer_batch_ceiling'
started = time.perf_counter()
readouts = {}
with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
        (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_file:
    for step in range(UPDATES + 1):
        if step:
            model.train()
            decay = .2 + .8 * (1 + math.cos(math.pi * step / UPDATES)) / 2
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * decay
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for slot in range(2):
                while True:
                    attempted += 1
                    base = int(draw_rng.integers(56))
                    variant = int(draw_rng.integers(VARIANTS))
                    subject = base * VARIANTS + variant
                    seed = SEED + 3_000_000_000 + attempted
                    with torch.no_grad():
                        sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE)
                    used = bool(sample['eligible'][0])
                    draws.write(json.dumps({'step': step, 'slot': slot, 'attempt': attempted,
                        'base': base, 'variant': variant, 'seed': seed, 'accepted': used,
                        'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                    if used:
                        accepted += 1
                        break
                    del sample
                inputs, true_state, true_reflection = (sample[key] for key in
                    ('inputs', 'state', 'reflection'))
                axial, masses = sample['offsets'], sample['weights']
                pixels = torch.multinomial(sample['valid_mask'].flatten(1).float(),
                                           128, replacement=True)
                chart = torch.stack((pixels.remainder(SIDE),
                    pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                truth = sample['centre'].flatten(1, 2).gather(
                    1, pixels[..., None].expand(-1, -1, 3))
                prediction = model.predict(inputs)
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                flags = reflection_options.expand(1, model.modes, 2)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                rigid = rigid_points_090(states, flags, chart[:, None, None], (SIDE, SIDE))
                five = rigid_points_090(states, flags, corners, (SIDE, SIDE))
                true_five = rigid_points_090(true_state, true_reflection,
                                              corners, (SIDE, SIDE))
                site_um = (rigid - truth[:, None, None]).norm(dim=-1).mean(-1)
                five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
                normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                true_normal = full_frame_state_to_components(true_state)[1][..., :, 2]
                normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
                physical = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
                nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
                near_mode = model.base_modes + nearest.indices
                near_cost = physical.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
                neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
                direct_loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                    + .5 * F.kl_div(prior, F.softmax(-physical.detach() / .7, -1),
                                      reduction='none').sum(-1)
                    + .5 * physical[:, :2 * model.base_modes].min(-1).values
                    + (neighbourhood * near_cost).sum(-1)).mean()
                q_loss = model.source_reliability_loss(prediction, sample['valid_mask'])
                branches = prior.detach().topk(8, -1).indices
                near_target = F.softmax(-site_um.flatten(1).detach().gather(1, branches) / 700, -1)
                (direct_loss / 2 + .05 * q_loss).backward()
                direct_value, q_value = float(direct_loss.detach()), float(q_loss.detach())
                del prediction, states, rigid, five, site_um, physical, direct_loss, q_loss

                # Frozen same-step values give exact first-order gradients of the global
                # rank and selected-risk objectives while each two-candidate graph is freed.
                with torch.no_grad():
                    coarse = model.predict(inputs)
                    score_parts, error_parts, rigid_parts = [], [], []
                    for start in range(0, 8, 2):
                        group = branches[:, start:start + 2]
                        selected = select_branch(coarse, group)
                        index = torch.arange(2, device='cuda')[None]
                        first = model.map(selected, axial, index, group % 2, (64, 64),
                            context['atlas'], masses, return_refinement_feature=True,
                            feature_side=64, source_shape=(SIDE, SIDE), max_derivative=None)
                        refined_state, _, _ = model.refine(first['refinement_feature'],
                                                           selected['state'])
                        refined = {**selected, 'state': refined_state}
                        fitted = model.map(refined, axial, index, group % 2, (96, 96),
                            context['atlas'], masses, feature_side=96,
                            source_shape=(SIDE, SIDE), max_derivative=None)
                        error_parts.append((surface_points(fitted, chart) - truth[:, None])
                            .norm(dim=-1).mean(-1) / 1000)
                        rigid_parts.append((rigid_points_090(refined_state, group % 2,
                            chart[:, None], (SIDE, SIDE)) - truth[:, None])
                            .norm(dim=-1).mean(-1) / 1000)
                        score_parts.append(model.score_fitted_candidates(inputs, selected, fitted,
                            context['atlas'], masses, evidence_arm='full'))
                    all_scores = torch.cat(score_parts, -1)
                    all_error = torch.cat(error_parts, -1)
                    all_rigid = torch.cat(rigid_parts, -1)
                    probabilities = F.softmax(all_scores, -1)
                    rank_target = F.softmax(-all_error / .7, -1)
                    expected = (probabilities * all_error).sum(-1, keepdim=True)
                    rank_coefficient = probabilities - rank_target
                    score_risk_coefficient = probabilities * (all_error - expected)
                    rank_value = float(F.kl_div(F.log_softmax(all_scores, -1), rank_target,
                        reduction='none').sum())
                    risk_value = float(expected.mean())
                    near_value = float((near_target * all_error).sum())
                    rigid_value = float((near_target * all_rigid).sum())
                local_value = 0.
                for start in range(0, 8, 2):
                    group = branches[:, start:start + 2]
                    index = torch.arange(2, device='cuda')[None]
                    prediction = model.predict(inputs)
                    selected = select_branch(prediction, group)
                    with torch.no_grad():
                        first = model.map(selected, axial, index, group % 2, (64, 64),
                            context['atlas'], masses, return_refinement_feature=True,
                            feature_side=64, source_shape=(SIDE, SIDE), max_derivative=None)
                    refined_state, _, update = model.refine(first['refinement_feature'],
                                                           selected['state'])
                    refined = {**selected, 'state': refined_state}
                    fitted = model.map(refined, axial, index, group % 2, (96, 96),
                        context['atlas'], masses, feature_side=96,
                        source_shape=(SIDE, SIDE), max_derivative=None)
                    error = (surface_points(fitted, chart) - truth[:, None]).norm(dim=-1).mean(-1) / 1000
                    scores = model.score_fitted_candidates(inputs, selected, fitted,
                                                            context['atlas'], masses, evidence_arm='full')
                    blind_rigid = (rigid_points_090(refined_state, group % 2, chart[:, None],
                        (SIDE, SIDE)) - truth[:, None]).norm(dim=-1).mean(-1) / 1000
                    source_q = F.interpolate(prediction['reliability_logit'], (96, 96),
                        mode='bilinear', align_corners=False).sigmoid().detach()
                    local = fitted['local_displacement_um'] / 1000
                    q_den = source_q.sum().clamp_min(1)
                    local_cost = (source_q * local.square().sum(2)).sum() / (q_den * 8)
                    local_cost += .5 * (source_q[..., 1:] * (local[..., 1:] -
                        local[..., :-1]).square().sum(2)).sum() / (q_den * 8)
                    local_cost += .5 * (source_q[..., 1:, :] * (local[..., 1:, :] -
                        local[..., :-1, :]).square().sum(2)).sum() / (q_den * 8)
                    local_value += float(local_cost.detach())
                    pose_update_cost = ((update[..., :3] / .9).square().sum() +
                        (update[..., 3:6] / 4000).square().sum()) / (8 * 3)
                    portion = slice(start, start + 2)
                    blind_loss = (.7 * (rank_coefficient[:, portion].detach() * scores).sum()
                        + (score_risk_coefficient[:, portion].detach() * scores).sum()
                        + (probabilities[:, portion].detach() * error).sum()
                        + .5 * (near_target[:, portion] * error).sum()
                        + .5 * (near_target[:, portion] * blind_rigid).sum()
                        + .05 * local_cost + .003 * pose_update_cost)
                    if step == 1 and slot == 0 and start == 0:
                        row_index = torch.arange(1, device='cuda')[:, None]
                        branch_prior = selected['log_mass'][row_index, index] + torch.where(
                            (group % 2).bool(),
                            F.logsigmoid(selected['reflection_logit'][row_index, index]),
                            F.logsigmoid(-selected['reflection_logit'][row_index, index]))
                        fit_only = model.score_spatial_candidates(inputs, selected, fitted,
                            context['atlas'], masses, evidence_arm='full',
                            exit_weight=0., warp_weight=0.)
                        evidence = (fit_only - branch_prior).sum()
                        pose_gradient, warp_gradient = torch.autograd.grad(evidence,
                            (refined_state, model.warp[-1].weight), retain_graph=True)
                        assert torch.isfinite(pose_gradient).all() and pose_gradient.norm() > 0
                        assert torch.isfinite(warp_gradient).all() and warp_gradient.norm() > 0
                        fit_pose_gradient = float(pose_gradient.norm())
                        fit_warp_gradient = float(warp_gradient.norm())
                    (blind_loss / 2).backward()
                    del prediction, selected, first, refined, fitted, scores, blind_loss
                metrics.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
                    'base': base, 'slot': slot, 'direct_loss': direct_value,
                    'q_loss': q_value, 'rank_loss': rank_value,
                    'blind_selected_risk_mm': risk_value,
                    'near_map_error_mm': near_value,
                    'refined_rigid_error_mm': rigid_value,
                    'local_cost': local_value})

                # A separate forward bounds the blind-candidate autograd graph before teacher branches.
                prediction = model.predict(inputs)
                jitter_scale = true_state.new_tensor([.12, .12, .12, 500., 500., 500.,
                                                       .025, .025, .025])
                jitter_update = torch.randn((1, 9), device='cuda') * jitter_scale
                jittered = compose_full_frame_state(true_state, jitter_update)
                seed_states = torch.cat((true_state[:, None], jittered[:, None]), 1)
                teacher_prediction = {**prediction, 'state': seed_states}
                teacher_index = torch.arange(2, device='cuda')[None]
                teacher_flags = true_reflection[:, None].expand(-1, 2)
                with torch.no_grad():
                    teacher_first = model.map(teacher_prediction, axial, teacher_index,
                        teacher_flags, (64, 64), context['atlas'], masses,
                        return_refinement_feature=True, feature_side=64,
                        source_shape=(SIDE, SIDE), max_derivative=None)
                corrected, _, _ = model.refine(teacher_first['refinement_feature'], seed_states)
                mapped_states = torch.cat((true_state[:, None], corrected[:, 1:2]), 1)
                mapped_prediction = {**teacher_prediction, 'state': mapped_states}
                teacher_map = model.map(mapped_prediction, axial, teacher_index,
                    teacher_flags, (96, 96), context['atlas'], masses,
                    feature_side=96, source_shape=(SIDE, SIDE), max_derivative=None)
                teacher_mm = (surface_points(teacher_map, chart) - truth[:, None]).norm(dim=-1).mean(-1) / 1000
                exact_drift_mm = (rigid_points_090(corrected[:, :1], true_reflection[:, None],
                    corners, (SIDE, SIDE)) - true_five[:, None]).norm(dim=-1).mean() / 1000
                jitter_pose_mm = (rigid_points_090(corrected[:, 1:2], true_reflection[:, None],
                    chart[:, None], (SIDE, SIDE)) - truth[:, None]).norm(dim=-1).mean() / 1000
                teacher_local = teacher_map['local_displacement_um'] / 1000
                teacher_q = prediction['reliability_logit'].sigmoid().detach()
                teacher_cost = (teacher_q * F.interpolate(
                    teacher_local.square().sum(2), teacher_q.shape[-2:], mode='area')).sum()
                teacher_cost /= (2 * teacher_q.sum().clamp_min(1))
                teacher_loss = (3 * teacher_mm[:, 0].mean() + teacher_mm[:, 1].mean()
                    + .8 * exact_drift_mm + .7 * jitter_pose_mm + .02 * teacher_cost)
                (teacher_loss / 2).backward()
                metrics[-1].update({'exact_map_error_mm': float(teacher_mm[:, 0].detach()),
                    'jitter_map_error_mm': float(teacher_mm[:, 1].detach()),
                    'exact_pose_drift_mm': float(exact_drift_mm.detach()),
                    'jitter_pose_error_mm': float(jitter_pose_mm.detach()),
                    'jitter_local_update': jitter_update[0].tolist(),
                    'teacher_loss': float(teacher_loss.detach())})
                del sample, prediction, teacher_first, teacher_map, teacher_loss
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            row = {'step': step, 'attempted': attempted, 'accepted': accepted,
                'images': metrics, 'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}
            if step == 1:
                row['fit_only_score_to_pose_gradient_norm'] = fit_pose_gradient
                row['fit_only_score_to_warp_head_gradient_norm'] = fit_warp_gradient
            training.write(json.dumps(row, allow_nan=False) + '\n')
            if step == 1 or step % 250 == 0:
                draws.flush(); training.flush()
                print(json.dumps({'event': 'joint_batch', 'step': step,
                    'selected_risk_mm': float(np.mean([m['blind_selected_risk_mm'] for m in metrics])),
                    'exact_map_mm': float(np.mean([m['exact_map_error_mm'] for m in metrics]))}),
                    flush=True)

        if step in READOUTS:
            draws.flush(); training.flush()
            model.eval()
            rows = []
            with torch.inference_mode():
                for item in panel:
                    inputs = item['inputs'][None].cuda()
                    chart = item['chart'][None].cuda()
                    truth = item['truth'][None].cuda()
                    axial = item['offsets'][None].cuda()
                    masses = item['weights'][None].cuda()
                    prediction = model.predict(inputs)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                    branches = prior.topk(16, -1).indices
                    branch_scores = {'full': [], 'support_only': [], 'shuffled': []}
                    rigid_error, mapped_error, normal_error, support_mean = [], [], [], []
                    true_normal = full_frame_state_to_components(
                        item['state'][None].cuda())[1][0, :, 2]
                    for start in range(0, 16, 2):
                        group = branches[:, start:start + 2]
                        selected = select_branch(prediction, group)
                        index = torch.arange(2, device='cuda')[None]
                        rigid = rigid_points_090(selected['state'], group % 2,
                            chart[:, None], (SIDE, SIDE))
                        rigid_error.extend(((rigid - truth[:, None]).norm(dim=-1).mean(-1)
                                            / 1000)[0].tolist())
                        first = model.map(selected, axial, index, group % 2, (64, 64),
                            context['atlas'], masses, return_refinement_feature=True,
                            feature_side=64, source_shape=(SIDE, SIDE), max_derivative=None)
                        refined_state, _, _ = model.refine(first['refinement_feature'],
                                                           selected['state'])
                        fitted_normal = full_frame_state_to_components(
                            refined_state)[1][0, :, :, 2]
                        normal_error.extend(torch.rad2deg(torch.acos((fitted_normal
                            @ true_normal).abs().clamp(0, 1))).tolist())
                        refined = {**selected, 'state': refined_state}
                        fitted = model.map(refined, axial, index, group % 2, (96, 96),
                            context['atlas'], masses, feature_side=96,
                            source_shape=(SIDE, SIDE), max_derivative=None)
                        mapped_error.extend(((surface_points(fitted, chart) - truth[:, None])
                            .norm(dim=-1).mean(-1) / 1000)[0].tolist())
                        rendered = render_finite_thickness_coordinate_grid(context['atlas'],
                            fitted['coordinates'][0], (0., 0., 0.), (25., 25., 25.),
                            masses.expand(2, -1))
                        support_mean.extend(rendered[:, 1].mean((1, 2)).tolist())
                        for arm in branch_scores:
                            branch_scores[arm].extend(model.score_fitted_candidates(
                                inputs, selected, fitted, context['atlas'], masses,
                                evidence_arm=arm)[0].tolist())
                    score_prior = prior[0, branches[0]].tolist()
                    chosen = {'direct_prior': 0, **{arm: int(np.argmax(values))
                        for arm, values in branch_scores.items()}}
                    teacher_prediction = {**prediction, 'state': item['state'][None, None].cuda()}
                    exact_index = torch.zeros((1, 1), device='cuda', dtype=torch.long)
                    exact_flag = item['reflection'][None, None].cuda()
                    exact_first = model.map(teacher_prediction, axial, exact_index, exact_flag,
                        (64, 64), context['atlas'], masses, return_refinement_feature=True,
                        feature_side=64, source_shape=(SIDE, SIDE), max_derivative=None)
                    corrected, _, _ = model.refine(exact_first['refinement_feature'],
                                                   teacher_prediction['state'])
                    exact_drift = (rigid_points_090(corrected, exact_flag, corners,
                        (SIDE, SIDE)) - rigid_points_090(teacher_prediction['state'], exact_flag,
                        corners, (SIDE, SIDE))).norm(dim=-1).mean().item() / 1000
                    exact_map = model.map(teacher_prediction, axial, exact_index, exact_flag,
                        (96, 96), context['atlas'], masses,
                        feature_side=96, source_shape=(SIDE, SIDE), max_derivative=None)
                    exact_error = ((surface_points(exact_map, chart) - truth[:, None])
                        .norm(dim=-1).mean().item() / 1000)
                    zero_residual = (rigid_points_090(teacher_prediction['state'], exact_flag,
                        chart[:, None], (SIDE, SIDE)) - truth[:, None]).norm(dim=-1)
                    zero_error = zero_residual.mean().item() / 1000
                    zero_rms = zero_residual.square().mean().sqrt().item() / 1000
                    near = int(np.argmin(rigid_error))
                    far = [i for i in range(16) if rigid_error[near] <= 1.5
                        and rigid_error[i] >= max(2., rigid_error[near] + 1.)
                        and abs(support_mean[i] - support_mean[near]) <= .10]
                    far = max(far, key=lambda i: score_prior[i]) if far else None
                    pair = None if far is None else [near, far]
                    row = {'physical_section_id': item['physical_section_id'],
                        'base': item['base'], 'slot': item['slot'],
                        'obliquity_deg': item['obliquity_deg'],
                        'visible_fraction': item['visible_fraction'],
                        'damage_fraction': item['damage_fraction'],
                        'exterior_style': item['exterior_style'],
                        'branch_ids': branches[0].tolist(), 'prior_log': score_prior,
                        'rigid_error_mm': rigid_error, 'mapped_error_mm': mapped_error,
                        'normal_error_deg': normal_error,
                        'support_mean': support_mean, 'scores': branch_scores,
                        'chosen': chosen, 'pair': pair,
                        'exact_pose_drift_mm': exact_drift,
                        'exact_map_error_mm': exact_error,
                        'zero_warp_error_mm': zero_error,
                        'zero_warp_residual_rms_mm': zero_rms}
                    rows.append(row)
            means = {name: float(np.mean([row['mapped_error_mm'][row['chosen'][name]]
                for row in rows])) for name in ('direct_prior', 'full', 'support_only', 'shuffled')}
            normal_means = {name: float(np.mean([row['normal_error_deg'][row['chosen'][name]]
                for row in rows])) for name in means}
            per_base = {str(base): {name: float(np.mean([
                row['mapped_error_mm'][row['chosen'][name]] for row in rows
                if row['base'] == base])) for name in means} for base in range(56, 64)}
            pairs = [row for row in rows if row['pair'] is not None]
            wins = {arm: sum((row['scores'][arm][row['pair'][0]] - row['prior_log'][row['pair'][0]])
                > (row['scores'][arm][row['pair'][1]] - row['prior_log'][row['pair'][1]])
                for row in pairs) for arm in ('full', 'support_only')}
            strong = sorted(rows, key=lambda row: row['zero_warp_residual_rms_mm'], reverse=True)[:16]
            strong_zero = float(np.mean([row['zero_warp_error_mm'] for row in strong]))
            strong_map = float(np.mean([row['exact_map_error_mm'] for row in strong]))
            gate = {'gain_direct_0p40mm': means['direct_prior'] - means['full'] >= .40,
                'gain_support_0p25mm': means['support_only'] - means['full'] >= .25,
                'gain_shuffle_0p25mm': means['shuffled'] - means['full'] >= .25,
                'bases_6of8_both': sum(per_base[str(base)]['full'] < min(
                    per_base[str(base)]['support_only'], per_base[str(base)]['shuffled'])
                    for base in range(56, 64)) >= 6,
                'at_least_40_pairs': len(pairs) >= 40,
                'full_pairs_70pct': bool(pairs) and wins['full'] >= .70 * len(pairs),
                'pair_advantage_10points': bool(pairs) and wins['full'] - wins['support_only']
                    >= .10 * len(pairs),
                'exact_pose_drift_0p25mm': float(np.mean([
                    row['exact_pose_drift_mm'] for row in rows])) <= .25,
                'strong_warp_gain_10pct': strong_map <= .9 * strong_zero}
            gate['pass'] = all(gate.values())
            strata = {'obliquity_ge45': [row for row in rows if row['obliquity_deg'] >= 45],
                'visible_lt0p1': [row for row in rows if row['visible_fraction'] < .1],
                'damage_ge0p02': [row for row in rows if row['damage_fraction'] >= .02]}
            strata.update({style: [row for row in rows if row['exterior_style'] == style]
                for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
            strata_metrics = {name: {'n': len(group),
                'full_mapped_mean_mm': (float(np.mean([r['mapped_error_mm'][r['chosen']['full']]
                    for r in group])) if group else None),
                'full_unsigned_normal_mean_deg': (float(np.mean([
                    r['normal_error_deg'][r['chosen']['full']] for r in group]))
                    if group else None)} for name, group in strata.items()}
            readouts[str(step)] = {'mean_selected_mapped_error_mm': means,
                'mean_selected_unsigned_normal_error_deg': normal_means,
                'per_base': per_base, 'matched_pairs': len(pairs),
                'matched_pair_evidence_wins': wins,
                'exact_pose_drift_mean_mm': float(np.mean([
                    row['exact_pose_drift_mm'] for row in rows])),
                'strong_warp_zero_mean_mm': strong_zero,
                'strong_warp_exact_map_mean_mm': strong_map,
                'strata': strata_metrics, 'gate': gate}
            readout_file.write(json.dumps({'step': step, 'summary': readouts[str(step)],
                'rows': rows}, allow_nan=False) + '\n')
            readout_file.flush()
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'config': config, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'draw_rng_state': draw_rng.bit_generator.state,
                'pose_parent_sha256': config['pose_parent_sha256'],
                'calibrated': False}, RUN / f'joint_step_{step:05d}.pt')
            print(json.dumps({'event': 'joint_readout', 'step': step,
                'summary': readouts[str(step)]}), flush=True)
            model.train()
            if step == 2000 and all(readouts[str(previous)][
                'mean_selected_mapped_error_mm']['full'] >= readouts['0'][
                'mean_selected_mapped_error_mm']['full'] + .5 for previous in (1000, 2000)):
                stop_reason = 'two_consecutive_material_inner_regressions'
                print(json.dumps({'event': 'early_stop', 'step': step,
                    'reason': stop_reason}), flush=True)
                break

selected_step = min((int(step) for step in readouts), key=lambda step: (
    not readouts[str(step)]['gate']['pass'],
    readouts[str(step)]['mean_selected_mapped_error_mm']['full']))
summary = {'candidate': 305, 'stage': 'joint_feedback', 'selected_step': selected_step,
    'readouts': readouts, 'attempted': attempted, 'accepted': accepted,
    'optimizer_batches_completed': step, 'stop_reason': stop_reason,
    'gate_pass': readouts[str(selected_step)]['gate']['pass'],
    'calibrated': False, 'no_dev_cal_final_or_public_benchmark': True}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
receipt = {'summary': summary, 'output_sha256': {path.name: sha(path) for path in RUN.iterdir()
    if path.name != 'completed.json'}, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
