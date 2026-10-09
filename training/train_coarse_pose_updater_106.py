"""Paired TRAIN-only atlas-correlation and support-only full-frame pose update."""
import copy
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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.coarse_pose_updater_106 import CoarsePoseUpdater106
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083, synthetic_match_targets

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_POSE_UPDATER_106_PROTOCOL_20261009.md'
pose_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
fit_run = root / 'runs/spatial_verifier_094_pilot'
pose_parent = pose_run / 'joint_step_02000.pt'
fit_parent = fit_run / 'joint_step_20000.pt'
run = root / 'runs/coarse_pose_updater_106'
seed, updates, side = 20261009106, 4000, 256
checkpoints = (0, 1000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
pose_receipt = json.loads((pose_run / 'completed.json').read_text())
fit_receipt = json.loads((fit_run / 'completed.json').read_text())
assert sha(pose_parent) == pose_receipt['checkpoint_sha256']['2000']
assert sha(fit_parent) == fit_receipt['checkpoint_sha256']['20000']
assert not any(pose_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert not any(fit_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
pose_checkpoint = torch.load(pose_parent, map_location='cpu', weights_only=True)
fit_checkpoint = torch.load(fit_parent, map_location='cpu', weights_only=True)
pose_model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
pose_model.load_state_dict(pose_checkpoint['model'], strict=True)
parent = SpatialJointFit094(WholeSliceAtlasFeedback083())
parent.load_state_dict(fit_checkpoint['spatial_fit'], strict=True)
atlas_model = CoarsePoseUpdater106(parent.matcher, parent.validity_head)
models = {'atlas': atlas_model.cuda().train(),
          'support_only': copy.deepcopy(atlas_model).cuda().train()}
assert all(torch.equal(left, right) for left, right in zip(
    models['atlas'].state_dict().values(), models['support_only'].state_dict().values()))
del pose_checkpoint, fit_checkpoint, parent, atlas_model

optimizers, parameters = {}, {}
for arm, model in models.items():
    model.requires_grad_(False)
    field = (model.field_input, model.field_gates, model.field_proposal,
        model.offset_head, model.reliability_head)
    for module in field:
        module.requires_grad_(True)
    model.validity_head.requires_grad_(True)
    groups = [{'params': list(module.parameters()), 'base_lr': 1e-4, 'from_step': 1}
              for module in field]
    groups += [{'params': list(model.validity_head.parameters()),
                'base_lr': 1e-5, 'from_step': 1},
               {'params': list(model.matcher.image.parameters()),
                'base_lr': 5e-6, 'from_step': 1001},
               {'params': list(model.matcher.atlas.parameters()),
                'base_lr': 5e-6, 'from_step': 1001},
               {'params': [model.matcher.log_temperature],
                'base_lr': 2e-6, 'from_step': 1001}]
    optimizers[arm] = torch.optim.AdamW(groups, weight_decay=1e-4)
    parameters[arm] = [parameter for group in groups for parameter in group['params']]

subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
flags = torch.tensor((0, 1), device='cuda')
source_files = ('train_coarse_pose_updater_106.py', 'coarse_pose_updater_106.py',
    'spatial_joint_fit_094.py', 'whole_slice_atlas_feedback_083.py',
    'whole_slice_atlas_feedback_081.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'global_atlas_contrast_090.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'checkpoints': checkpoints,
    'beam': {'old': 8, 'anchor': 6}, 'train_blind_candidates': 4,
    'teacher': 'one exact target state on odd updates; one jittered target state on even updates; TRAIN only',
    'jitter_limits': {'rotation_rad': .04, 'translation_um': 500,
        'basis_log': .025, 'shear': .015},
    'control': 'support_only zeros correlation logits but retains atlas support',
    'score': 'pose prior + 2*log(validity-weighted mean support_any*sigmoid(reliability))',
    'loss': '2 pose Huber + .5 observed-CCF site Huber + .1 supported coarse CE + .2 reliability BCE + .1 validity BCE + .25 blind rank + .02 offset penalty',
    'pose_parent': str(pose_parent), 'pose_parent_sha256': sha(pose_parent),
    'fit_parent': str(fit_parent), 'fit_parent_sha256': sha(fit_parent),
    'pose_completed_sha256': sha(pose_run / 'completed.json'),
    'fit_completed_sha256': sha(fit_run / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'models': {arm: model.state_dict() for arm, model in models.items()},
        'optimizers': {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'pose_step_{step:05d}.pt')


save(0)
attempts = 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        if step == 1001:
            for model in models.values():
                model.matcher.image.requires_grad_(True)
                model.matcher.atlas.requires_grad_(True)
                model.matcher.log_temperature.requires_grad_(True)
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            with torch.no_grad():
                sampled = sample_one_shot_slide_artifacts_v3(context, virtual, draw_seed, side=side)
            record = sampled['provenance'][0]
            used = bool(sampled['eligible'][0])
            attempts += 1
            draws.write(json.dumps({**record, 'update': step,
                'draw_attempt': attempts, 'used': used}) + '\n')
            draw_seed += 1
            if used:
                accepted = sampled
        with torch.no_grad():
            prediction = pose_model.predict(accepted['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags[None, None].expand(1, pose_model.modes, 2)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                128, replacement=True)
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            target_rigid = rigid_points_090(accepted['state'][:, None],
                accepted['reflection'].reshape(1, 1), chart)
            physical_mm = (rigid_points_090(states, reflections, chart) -
                target_rigid[:, :, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            beam_state = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            rendered = render_atlas_planes_090(context['atlas'], beam_state, beam % 2,
                accepted['offsets'], accepted['weights'], candidate_chunk=2)
            support = rendered[:, :, 1].mean((-2, -1))[0]
            error = physical_mm.gather(1, beam)[0]
            beam_prior = prior.gather(1, beam)[0]
            best = int(error.argmin())
            gap = error - error[best]
            remaining = [index for index in range(14) if index != best]
            options = [index for index in remaining if gap[index] >= .5]
            support_fallback = not options
            if not options:
                options = remaining
            support_choice = min(options, key=lambda index: float(abs(
                support[index] - support[best])))
            remaining.remove(support_choice)
            options = [index for index in remaining if gap[index] >= 1.]
            prior_fallback = not options
            if not options:
                options = remaining
            prior_choice = max(options, key=lambda index: float(beam_prior[index]))
            remaining.remove(prior_choice)
            fourth = remaining[int(torch.randint(len(remaining), (1,), generator=subject_rng))]
            slots = torch.tensor((best, support_choice, prior_choice, fourth), device='cuda')
            choice = beam[:, slots]
            chosen_error = error[slots]
            chosen_support = support[slots]
            jitter = torch.zeros((1, 9), device='cuda')
            if step % 2 == 0:
                limits = torch.tensor((.04, .04, .04, 500., 500., 500.,
                    .025, .025, .015), device='cuda')
                jitter = (2 * torch.rand((1, 9), generator=subject_rng) - 1).cuda() * limits
            teacher_state = compose_full_frame_state(accepted['state'], jitter)[:, None]
            candidate_state = torch.cat((prediction['state'].gather(1,
                (choice // 2)[..., None].expand(-1, -1, 12)), teacher_state), 1)
            reflected = torch.cat((choice % 2, accepted['reflection'].reshape(1, 1)), 1)
            cell = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
            yy, xx = torch.meshgrid(cell, cell, indexing='ij')
            cell_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
            pose_target = rigid_points_090(accepted['state'][:, None],
                accepted['reflection'].reshape(1, 1), cell_chart).reshape(1, 1, 16, 16, 3)
            map_target = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                (16, 16), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            valid_site = F.interpolate(accepted['valid_mask'][:, None].float(),
                (16, 16), mode='bilinear', align_corners=False)[:, 0] == 1

        metrics = {}
        for arm, updater in models.items():
            output = updater(prediction['feature'], candidate_state, reflected,
                context['atlas'], accepted['offsets'], accepted['weights'],
                source_shape=(side, side), atlas_disabled=(arm == 'support_only'))
            with torch.no_grad():
                labels = [synthetic_match_targets(accepted['centre'],
                    accepted['valid_mask'], output['iteration_states'][:, iteration],
                    reflected) for iteration in range(2)]
            truth_index = torch.stack([label['coarse_index'] for label in labels], 1)
            truth_mask = torch.stack([label['coarse_mask'] for label in labels], 1)
            supported = output['coarse_match_support'].gather(
                3, truth_index[:, :, :, None]).squeeze(3) >= .5
            surviving = valid_site[:, None, None].expand_as(truth_mask)
            matched = surviving & truth_mask & supported
            unmatched = surviving & ~matched
            candidate_weight = candidate_state.new_tensor((1., 1., 1., 1., 3.))[
                None, None, :, None, None]
            ce = F.cross_entropy(output['coarse_match_logits'].flatten(0, 2),
                truth_index.flatten(0, 2), reduction='none').reshape_as(matched)
            match_weight = matched * candidate_weight
            match_loss = (ce * match_weight).sum() / match_weight.sum().clamp_min(1)
            reliability_ce = F.binary_cross_entropy_with_logits(
                output['coarse_reliability_logit'], matched.float(), reduction='none')
            positive_weight = matched * candidate_weight
            negative_weight = unmatched * candidate_weight
            reliability_loss = .5 * (reliability_ce * positive_weight).sum() / \
                positive_weight.sum().clamp_min(1) + .5 * \
                (reliability_ce * negative_weight).sum() / negative_weight.sum().clamp_min(1)
            state_map = torch.stack([rigid_points_090(
                output['iteration_states'][:, iteration + 1], reflected, cell_chart).reshape(
                    1, 5, 16, 16, 3) for iteration in range(2)], 1)
            pose_site = F.smooth_l1_loss((state_map - pose_target[:, None]) / 1000,
                torch.zeros_like(state_map), reduction='none', beta=.5).sum(-1)
            mapped = output['coarse_mapped_ccf_um']
            map_site = F.smooth_l1_loss((mapped - map_target[:, None, None]) / 1000,
                torch.zeros_like(mapped), reduction='none', beta=.5).sum(-1)
            site_weight = (valid_site[:, None, None] * candidate_weight).expand(
                -1, 2, -1, -1, -1)
            pose_loss = (pose_site * site_weight).sum() / site_weight.sum().clamp_min(1)
            map_loss = (map_site * site_weight).sum() / site_weight.sum().clamp_min(1)
            validity_truth = F.interpolate(accepted['valid_mask'][:, None].float(),
                (32, 32), mode='bilinear', align_corners=False)[:, 0]
            validity_loss = F.binary_cross_entropy_with_logits(
                output['validity_logit'], validity_truth)
            flow = output['coarse_offsets_xy_cells_normal_um']
            normalized_flow = torch.cat((flow[:, :, :, :2] / 4,
                flow[:, :, :, 2:3] / 6000), 3)
            offset_loss = (normalized_flow.square().sum(3) * site_weight).sum() / \
                site_weight.sum().clamp_min(1)
            tissue = F.adaptive_avg_pool2d(
                output['validity_logit'].sigmoid()[:, None], (16, 16))[:, 0, None]
            support_any = output['coarse_match_support'][:, 1].amax(2).clamp(0, 1)
            reliability = output['coarse_reliability_logit'][:, 1].sigmoid()
            q = (tissue * support_any * reliability).sum((-2, -1)) / \
                tissue.sum((-2, -1)).clamp_min(1e-6)
            score = prior.gather(1, choice) + 2 * q[:, :4].clamp_min(1e-4).log()
            final_error = ((state_map[:, 1] - pose_target).norm(dim=-1)
                * valid_site[:, None]).sum((-2, -1)) / valid_site.sum().clamp_min(1)
            pairs = [F.softplus(.5 - (score[0, left] - score[0, right])
                * (final_error[0, right] - final_error[0, left]).detach().sign())
                for left in range(4) for right in range(left + 1, 4)
                if abs(float(final_error[0, right] - final_error[0, left])) >= 500]
            rank_loss = torch.stack(pairs).mean() if pairs else score.sum() * 0
            loss = (2 * pose_loss + .5 * map_loss + .1 * match_loss
                + .2 * reliability_loss + .1 * validity_loss
                + .25 * rank_loss + .02 * offset_loss)
            decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
            optimizer = optimizers[arm]
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * decay if step >= group['from_step'] else 0.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters[arm], 5., error_if_nonfinite=True)
            optimizer.step()
            metrics[arm] = {'pose_huber_mm': float(pose_loss.detach()),
                'map_huber_mm': float(map_loss.detach()),
                'coarse_ce': float(match_loss.detach()),
                'reliability_bce': float(reliability_loss.detach()),
                'validity_bce': float(validity_loss.detach()),
                'rank_loss': float(rank_loss.detach()),
                'offset_penalty': float(offset_loss.detach()),
                'total_loss': float(loss.detach()),
                'matched_sites': int(matched.sum()),
                'teacher_matched_sites': int(matched[:, :, 4].sum()),
                'input_error_mm': chosen_error.tolist(),
                'final_pose_error_mm': (final_error[0] / 1000).detach().tolist(),
                'coverage': q[0].detach().tolist(),
                'score': score[0].detach().tolist(),
                'gradient_norm': float(gradient)}
        row = {'update': step, 'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'candidate_error_mm': chosen_error.tolist(),
            'candidate_support': chosen_support.tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'teacher_exact': step % 2 == 1, 'teacher_jitter': jitter[0].tolist(),
            'arms': metrics, 'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1048576,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({'update': step,
                'atlas_pose_huber_mm': metrics['atlas']['pose_huber_mm'],
                'support_only_pose_huber_mm': metrics['support_only']['pose_huber_mm'],
                'atlas_rank_loss': metrics['atlas']['rank_loss'],
                'matched_sites': metrics['atlas']['matched_sites'],
                'peak_gpu_mb': row['peak_gpu_mb'], 'seconds': row['seconds']}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates, 'draw_attempts': attempts,
    'protocol_sha256': sha(protocol), 'pose_parent_sha256': sha(pose_parent),
    'fit_parent_sha256': sha(fit_parent), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'pose_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
