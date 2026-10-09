"""Joint pose, finite-slab atlas comparison, and affine-free tissue-map training."""

import hashlib
import json
import math
import os
import sys
import time
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

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_from_components,
    full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.joint_pose_map_121 import (
    joint_fit_loss_121, joint_forward_121, joint_spatial_ce_121, joint_target_loss_121)
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_POSE_MAP_121_PROTOCOL_20261009.md'
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
run = root / 'runs/joint_pose_map_121'
seed, updates, side, sites, blind_count = 20261009121, 8000, 256, 128, 16
checkpoints = (0, 2000, 5000, 8000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def rigid_cost(states, reflection, target_state, target_reflection, chart):
    reference = rigid_points_090(target_state, target_reflection, chart)
    predicted = rigid_points_090(states, reflection, chart[:, None])
    fixed = rigid_points_090(target_state, target_reflection,
                             corners[None].expand(len(states), -1, -1))
    corners_pred = rigid_points_090(states, reflection,
                                    corners[None, None].expand(len(states), 1, -1, -1))
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
    penalty = 4 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    return (.75 * (predicted - reference[:, None]).norm(dim=-1).mean(-1)
            + .25 * (corners_pred - fixed[:, None]).norm(dim=-1).mean(-1)) / 1000 + penalty


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert sha(parent) == receipt['checkpoint_sha256']['1959']
assert sha(parent_run / 'config.json') == receipt['config_sha256']
assert not any(receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert sha(source / 'arbitrary_plane_one_shot_model.py') == \
    parent_config['source_sha256']['arbitrary_plane_one_shot_model.py']
context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
sag_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'model_input.npy') == sag_summary['output_sha256']['model_input.npy']
assert sha(sagittal_dir / 'geometry.jsonl') == sag_summary['output_sha256']['geometry.jsonl']
sag_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert all(row['split'] == 'train' for row in sag_records)
sag_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sag_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0],
                   side * affine[:, :, 1]), 1)
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
sag_normals = full_frame_state_to_components(sag_states)[1][..., :, 2]
sag_reflection = sag_normals.gather(1, sag_normals.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sag_reflection, 0] += (side - 1) / side * ouv[sag_reflection, 1]
ouv[sag_reflection, 1] *= -1
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()
sag_by_donor = {}
for index, record in enumerate(sag_records):
    sag_by_donor.setdefault(record['donor_id'], []).append(index)
sag_donors = sorted(sag_by_donor)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1959 and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
attach_global_plane_matcher(model, enabled=True)
spatial = WholeSliceAtlasFeedback083().cuda().train()
model.requires_grad_(False)
spatial.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
map_modules = (model.warp_shared, model.warp_condition,
               model.atlas_encoder, model.pair, model.warp)
for module in (*pose_modules, *shared_modules, *map_modules,
               model.global_plane_matcher_120, spatial.image, spatial.atlas):
    module.requires_grad_(True)
spatial.log_temperature.requires_grad_(True)
groups = [
    {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 1e-5},
    {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 2e-6},
    {'params': list(model.global_plane_matcher_120.parameters()), 'base_lr': 5e-5},
    {'params': [p for module in map_modules for p in module.parameters()], 'base_lr': 1e-5},
    {'params': [*spatial.image.parameters(), *spatial.atlas.parameters(),
                spatial.log_temperature], 'base_lr': 3e-5},
]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
parameters = [p for group in groups for p in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
source_files = ('train_joint_pose_map_121.py', 'joint_pose_map_121.py',
    'arbitrary_plane_one_shot_model.py', 'global_plane_matcher_120.py',
    'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
config = {'seed': seed, 'updates': updates, 'synthetic_per_update': 2,
    'synthetic_presentations': 2 * updates, 'side': side, 'valid_sites': sites,
    'blind_count': blind_count, 'blind_beam': '111 prior top 8 base + 6 anchor, plus 2 diverse unused anchors',
    'training_only_states': 'observed affine gauge exact and independent local perturbation',
    'checkpoints': checkpoints, 'real_start': 2001, 'real_every': 8,
    'real_role': 'TRAIN weak Allen affine pose retention only, coronal and sagittal',
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sag_summary['output_sha256'],
    'trainable': ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global',
        'global_plane_matcher_120', 'warp_shared', 'warp_condition', 'atlas_encoder',
        'pair', 'warp', '083.image', '083.atlas', '083.log_temperature'],
    'spatial_083_role': 'training-only auxiliary correspondence CE; not used by inference map or score',
    'frozen': ['legacy candidate/fitted scorers', 'legacy pose refiner',
        'fit quality head', 'uncalibrated uncertainty head'],
    'loss_weights': {'direct': 1., 'action_kl': .5, 'action_expected_cost': .25,
        'map': 1., 'selected_rigid': .5, 'fine_ce': .05, 'coarse_ce': .05,
        'warp': .03, 'exact_no_shift': .1, 'near_shift': .1,
        'fit_max': .05, 'real_weak_pose': .1},
    'fit_start': 2001, 'fit_ramp_end': 3000,
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'step': step, 'model': model.state_dict(), 'spatial': spatial.state_dict(),
        'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.get_state(),
        'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'fit_weight_cap': fit_weight_cap,
        'fit_audit_done': fit_audit_done, 'synthetic_draw_attempts': attempts,
        'fit_seen_count': fit_seen_count, 'fit_eligible_count': fit_eligible_count,
        'support_match_count': support_match_count,
        'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


fit_weight_cap, fit_audit_done = .05, False
attempts = 0
fit_seen_count, fit_eligible_count = 0, 0
support_match_count = 0
save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay
        optimizer.zero_grad(set_to_none=True)
        metrics, identities = [], []
        for slot in range(2):
            accepted = None
            while accepted is None:
                virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
                with torch.no_grad():
                    sampled = sample_one_shot_slide_artifacts_v3(context, virtual,
                        draw_seed, side=side)
                used = bool(sampled['eligible'][0])
                record = sampled['provenance'][0]
                attempts += 1
                draws.write(json.dumps({'update': step, 'slot': slot, 'draw_seed': draw_seed,
                    'draw_attempt': attempts, 'used': used, **record}) + '\n')
                draw_seed += 1
                if used:
                    accepted = {key: sampled[key] for key in
                        ('inputs', 'state', 'reflection', 'centre', 'valid_mask',
                         'offsets', 'weights')}
                    identities.append(record)
            inputs = accepted['inputs']
            prediction = model.predict(inputs)
            flags = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
            states160 = prediction['state'][:, :, None].expand(-1, -1, 2, -1).reshape(1, -1, 12)
            reflections160 = flags.reshape(1, -1)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                sites, replacement=True)
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            physical = rigid_cost(states160, reflections160, accepted['state'],
                accepted['reflection'], chart)
            target_mass = F.softmax(-physical.detach() / .7, -1)
            true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            direct = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1).mean()
                + .5 * F.kl_div(prior, target_mass, reduction='batchmean')
                + .5 * physical[:, :2 * model.base_modes].min(-1).values.mean()
                + (neighbourhood * near_cost).sum(-1).mean())

            with torch.no_grad():
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)
                normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                row = torch.arange(1, device='cuda')
                for _ in range(blind_count - 14):
                    chosen = normals[row[:, None], beam // 2]
                    similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                    diversity = -similarity
                    diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                    anchor = diversity.argmax(-1)
                    reflected = prior[:, 32:].reshape(1, 64, 2)[row, anchor].argmax(-1)
                    beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
                blind_mode, blind_reflection = beam // 2, beam % 2
            match16 = global_plane_match(model, prediction, blind_mode, blind_reflection,
                accepted['offsets'], accepted['weights'], context['atlas'],
                inputs.shape[-2:], side=24)
            original_cost = rigid_cost(match16['input_state'], blind_reflection,
                accepted['state'], accepted['reflection'], chart)
            corrected_cost = rigid_cost(match16['state'], blind_reflection,
                accepted['state'], accepted['reflection'], chart)
            action_cost = torch.stack((original_cost, corrected_cost), -1).flatten(1)
            action_scores = torch.stack((match16['input_score'], match16['score']), -1).flatten(1)
            action_target = F.softmax(-action_cost.detach() / .7, -1)
            action_kl = F.kl_div(F.log_softmax(action_scores, -1), action_target,
                reduction='batchmean')
            action_expected = (F.softmax(action_scores, -1) * action_cost.detach()).sum(-1).mean()
            with torch.no_grad():
                best = original_cost.argmin(-1)
                best_cost = original_cost.gather(1, best[:, None])
                wrong_eligible = original_cost >= best_cost + 1.
                wrong = original_cost.masked_fill(~wrong_eligible, float('inf')).argmin(-1)
                wrong_available = wrong_eligible.gather(1, wrong[:, None])[:, 0]
                chosen = beam.gather(1, torch.stack((best, wrong), -1))
                jitter_limits = accepted['state'].new_tensor(
                    (.2, .2, .2, 1500., 1500., 1500., .08, .08, .05))
                jitter = (2 * torch.rand(1, 9, generator=subject_rng).cuda() - 1) * jitter_limits
                near_state = compose_full_frame_state(accepted['state'], jitter)
            training_mode = torch.cat((chosen // 2, blind_mode[:, :1].expand(-1, 2)), 1)
            training_reflection = torch.cat((chosen % 2,
                accepted['reflection'][:, None].expand(-1, 2)), 1)
            row = torch.arange(1, device='cuda')[:, None]
            candidate_state = torch.cat((prediction['state'][row, chosen // 2],
                accepted['state'][:, None], near_state[:, None]), 1)
            joint = joint_forward_121(model, prediction, inputs, training_mode,
                training_reflection, context['atlas'], accepted['offsets'],
                accepted['weights'], candidate_state=candidate_state)
            mapped = joint['mapped']
            targets = joint_target_loss_121(mapped, accepted['valid_mask'],
                accepted['centre'], accepted['state'], accepted['reflection'], pixel)
            best_gate = torch.sigmoid((2.5 - best_cost.detach()) / .5)[:, 0]
            map_loss = (targets['map_loss'][:, 2:4].mean()
                + .5 * (best_gate * targets['map_loss'][:, 0]).mean())
            selected_rigid = (targets['rigid_loss'][:, 2:4].mean()
                + .5 * (best_gate * targets['rigid_loss'][:, 0]).mean())
            spatial_ce = joint_spatial_ce_121(spatial, prediction,
                joint['match']['state'][:, 2:4], training_reflection[:, 2:4],
                context['atlas'], accepted['offsets'], accepted['weights'],
                accepted['centre'], accepted['valid_mask'])
            local = mapped['local_displacement_um'][:, (0, 2, 3)] / 1000
            warp = (local.square().mean()
                + .3 * (local[..., 1:] - local[..., :-1]).square().mean()
                + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean())
            exact_gate = F.softplus(joint['match']['match_logit'][:, 2]).mean()
            near_gate = F.softplus(-joint['match']['match_logit'][:, 3]).mean()
            base_loss = (direct + .5 * action_kl + .25 * action_expected
                + map_loss + .5 * selected_rigid
                + .05 * (spatial_ce['fine_ce'] + spatial_ce['coarse_ce'])
                + .03 * warp + .1 * exact_gate + .1 * near_gate)
            fit_raw, fit_term, coverage_match = None, None, None
            fit_weight = 0.
            audit = None
            fit_eligible = False
            if step > 2000:
                selected_corrected_cost = rigid_cost(joint['match']['state'][:, :1],
                    training_reflection[:, :1], accepted['state'],
                    accepted['reflection'], chart)[:, 0]
                fit_eligible = bool((selected_corrected_cost.detach() <= 1.5)[0])
                fit_seen_count += 1
                fit_eligible_count += int(fit_eligible)
                if fit_eligible:
                    fit_raw = joint_fit_loss_121(inputs,
                        {'coordinates': mapped['coordinates'][:, :2]}, context['atlas'],
                        accepted['weights'], accepted['valid_mask'], side=64)
            if fit_raw is not None:
                coverage_match = (wrong_available &
                    ((fit_raw['atlas_coverage'][:, 0] -
                      fit_raw['atlas_coverage'][:, 1]).abs() <= .10))
                support_match_count += int(coverage_match[0])
                contrast = (F.relu(.10 + fit_raw['fit_loss'][:, 0]
                    - fit_raw['fit_loss'][:, 1].detach()) * coverage_match).mean()
                fit_term = (fit_raw['fit_loss'][:, 0]
                    + .5 * fit_raw['coverage_penalty'][:, 0]).mean() + contrast
                if not fit_audit_done and float(fit_raw['atlas_coverage'][:, 0].detach()) > .25:
                    audit_parameters = (joint['match']['state'], prediction['state'],
                        model.pose[-1].weight, model.anchor_pose[-1].weight,
                        model.anchor_global[-1].weight,
                        model.encoder[0][0].weight,
                        model.global_plane_matcher_120['head'][-1].weight)
                    gradients = torch.autograd.grad(fit_raw['fit_loss'][:, 0].mean(),
                        audit_parameters, retain_graph=True, allow_unused=True)
                    norms = [0. if gradient is None else float(gradient.norm())
                             for gradient in gradients]
                    assert all(math.isfinite(value) for value in norms)
                    assert norms[0] > 0 and norms[1] > 0 and max(norms[2:5]) > 0
                    assert norms[5] > 0 and norms[6] > 0
                    base_state_gradient = torch.autograd.grad(base_loss,
                        prediction['state'], retain_graph=True)[0]
                    base_state_norm = float(base_state_gradient.norm())
                    fit_weight_cap = min(.05, .2 * base_state_norm
                        / max(norms[1], 1e-12))
                    fit_audit_done = True
                    audit = {'fit_only_corrected_state_norm': norms[0],
                        'fit_only_direct_state_norm': norms[1],
                        'fit_only_pose_head_norm': max(norms[2:5]),
                        'fit_only_encoder_norm': norms[5],
                        'fit_only_matcher_norm': norms[6],
                        'base_direct_state_norm': base_state_norm,
                        'fit_to_direct_state_gradient_ratio': norms[1] / max(base_state_norm, 1e-12),
                        'capped_fit_to_direct_state_gradient_ratio':
                            fit_weight_cap * norms[1] / max(base_state_norm, 1e-12),
                        'fit_weight_cap': fit_weight_cap}
            if fit_audit_done and fit_raw is not None:
                fit_weight = fit_weight_cap * min(1., (step - 2000) / 1000)
            if audit is not None:
                audit['effective_weighted_fit_to_direct_state_gradient_ratio'] = (
                    fit_weight * audit['fit_to_direct_state_gradient_ratio'])
                print(json.dumps({'event': 'fit_gradient_audit', 'update': step,
                    'slot': slot, **audit}), flush=True)
            loss = base_loss + fit_weight * (fit_term if fit_term is not None else 0.)
            (loss / 2).backward()
            with torch.no_grad():
                selected_action = action_scores.argmax(-1)
                metrics.append({'direct': float(direct), 'action_kl': float(action_kl),
                    'action_expected_mm': float(action_expected),
                    'map_mm': float(map_loss), 'rigid_selected_mm': float(selected_rigid),
                    'spatial_fine_ce': float(spatial_ce['fine_ce']),
                    'spatial_coarse_ce': float(spatial_ce['coarse_ce']),
                    'spatial_fine_count': int(spatial_ce['fine_count']),
                    'spatial_coarse_count': int(spatial_ce['coarse_count']),
                    'warp': float(warp), 'fit': None if fit_raw is None else
                        float(fit_raw['fit_loss'][:, 0]),
                    'fit_coverage': None if fit_raw is None else
                        float(fit_raw['atlas_coverage'][:, 0]),
                    'support_matched_wrong': None if coverage_match is None else
                        bool(coverage_match[0]),
                    'fit_rigid_eligible': fit_eligible,
                    'fit_weight': fit_weight, 'fit_audit': audit,
                    'blind16_input_selected_mm': float(original_cost.gather(
                        1, match16['input_score'].argmax(-1)[:, None]).mean()),
                    'blind16_joint_selected_mm': float(action_cost.gather(
                        1, selected_action[:, None]).mean()),
                    'blind16_best_original_mm': float(original_cost.min(-1).values.mean()),
                    'blind16_best_corrected_mm': float(corrected_cost.min(-1).values.mean()),
                    'best_blind_index': int(best[0]), 'wrong_blind_index': int(wrong[0]),
                    'valid_site_indices_sha256': hashlib.sha256(
                        pixel.cpu().numpy().tobytes()).hexdigest(),
                    'jitter_local_updates': jitter[0].tolist(), 'loss': float(loss)})

        real_row = None
        if step > 2000 and step % 8 == 0:
            donor = int(torch.randint(len(coronal['donors']), (1,), generator=subject_rng))
            section = int(torch.randint(len(coronal['donors'][donor]['state']),
                (1,), generator=subject_rng))
            real_coronal = sample_reserved_real_train(coronal, donor, [section])
            coronal_inputs = F.interpolate(real_coronal['inputs'], (side, side),
                mode='bilinear', align_corners=False)
            sag_donor = sag_donors[int(torch.randint(len(sag_donors), (1,),
                generator=subject_rng))]
            sag_rows = sag_by_donor[sag_donor]
            sag_index = sag_rows[int(torch.randint(len(sag_rows), (1,), generator=subject_rng))]
            sag_inputs = torch.zeros(1, 5, side, side, device='cuda')
            sag_inputs[:, :1] = torch.from_numpy(np.asarray(
                sag_images[sag_index:sag_index + 1]).copy()).to('cuda')
            real_inputs = torch.cat((coronal_inputs, sag_inputs))
            real_state = torch.cat((real_coronal['state'],
                sag_states[sag_index:sag_index + 1].to('cuda')))
            real_reflection = torch.cat((real_coronal['reflection'],
                sag_reflection[sag_index:sag_index + 1].long().to('cuda')))
            real_prediction = model.predict(real_inputs)
            real_prior = (real_prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-real_prediction['reflection_logit']),
                F.logsigmoid(real_prediction['reflection_logit'])), -1)).flatten(1)
            real_flags = torch.arange(2, device='cuda')[None, None].expand(2, model.modes, 2)
            real_states = real_prediction['state'][:, :, None].expand(-1, -1, 2, -1).reshape(2, -1, 12)
            real_cost = rigid_cost(real_states, real_flags.reshape(2, -1),
                real_state, real_reflection, corners[None].expand(2, -1, -1))
            real_mass = F.softmax(-real_cost.detach() / .7, -1)
            real_loss = (-1.5 * torch.logsumexp(real_prior - real_cost / 1.5, -1)
                + .5 * F.kl_div(real_prior, real_mass, reduction='none').sum(-1)
                + .5 * real_cost.min(-1).values).mean()
            (.1 * real_loss).backward()
            real_row = {'coronal_identity': real_coronal['identities'][0],
                'sagittal_identity': {key: sag_records[sag_index][key] for key in
                    ('donor_id', 'specimen_id', 'experiment_id', 'section_id')},
                'weak_pose_loss': float(real_loss.detach()),
                'coronal_selected_weak_mm': float(real_cost[0,
                    real_prior[0].argmax()].detach()),
                'sagittal_selected_weak_mm': float(real_cost[1,
                    real_prior[1].argmax()].detach())}
            draws.write(json.dumps({'update': step, 'slot': 2, 'kind': 'coronal_weak_train',
                **real_row['coronal_identity'], 'used': True}) + '\n')
            draws.write(json.dumps({'update': step, 'slot': 3, 'kind': 'sagittal_weak_train',
                **real_row['sagittal_identity'], 'used': True}) + '\n')
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'synthetic_presentations': 2 * step,
            'physical_section_ids': [record['physical_section_id'] for record in identities],
            'synthetic': metrics, 'real': real_row, 'fit_weight_cap': fit_weight_cap,
            'fit_audit_done': fit_audit_done,
            'fit_eligible_fraction': fit_eligible_count / max(fit_seen_count, 1),
            'support_matched_wrong_count': support_match_count,
            'gradient_norm': float(gradient),
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 1000 == 0 or step in checkpoints:
            log.flush()
            draws.flush()
            print(json.dumps({'update': step, 'synthetic_presentations': 2 * step,
                'direct': sum(item['direct'] for item in metrics) / 2,
                'blind16_selected_mm': sum(item['blind16_joint_selected_mm']
                    for item in metrics) / 2,
                'map_mm': sum(item['map_mm'] for item in metrics) / 2,
                'fit_weight': metrics[-1]['fit_weight'],
                'fit_eligible_fraction': fit_eligible_count / max(fit_seen_count, 1),
                'gradient_norm': float(gradient),
                'seconds': time.perf_counter() - started}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

assert fit_audit_done
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': 2 * updates,
    'synthetic_draw_attempts': attempts,
    'weak_real_coronal_presentations': (updates - 2000) // 8,
    'weak_real_sagittal_presentations': (updates - 2000) // 8,
    'fit_seen_synthetic': fit_seen_count,
    'fit_eligible_synthetic': fit_eligible_count,
    'support_matched_wrong_synthetic': support_match_count,
    'fit_feedback_gradient_audited': fit_audit_done,
    'fit_weight_cap': fit_weight_cap,
    'parent_checkpoint_sha256': sha(parent),
    'protocol_sha256': sha(protocol), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'seconds': time.perf_counter() - started}), flush=True)
