"""TRAIN-only spatial verification of model-generated arbitrary-plane candidates."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import (
    WholeSliceAtlasFeedback083, synthetic_match_targets,
)

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SPATIAL_VERIFIER_094_PROTOCOL_20261004.md'
parent_run = root / 'runs/spatial_joint_fit_092_pilot'
parent = parent_run / 'joint_step_06000.pt'
run = root / 'runs/spatial_verifier_094_pilot'
seed, batches, side, pixels = 2026100494, 20000, 256, 1024
checkpoints = (0, 2000, 8000, 20000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '68277e40c472678a180a88d738f220dc87cafaf061152b133ec9eaaec45d1a8a'
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 6000
assert sha(parent) == parent_receipt['checkpoint_sha256']['6000']
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
matcher = WholeSliceAtlasFeedback083().cuda().train()
matcher.load_state_dict({key.removeprefix('matcher.'): value for key, value in
    checkpoint['spatial_fit'].items() if key.startswith('matcher.')}, strict=True)
del checkpoint
fitter = SpatialJointFit094(matcher).cuda().train()
model.requires_grad_(False)
fitter.requires_grad_(False)
head_modules = (fitter.distribution_projection, fitter.image_projection,
    fitter.evidence_projection, fitter.spatial_input, fitter.spatial_blocks,
    fitter.correct_head, fitter.validity_head)
joint_model_modules = (model.encoder, model.lateral, model.pose, model.anchor_pose,
    model.anchor_global, model.warp_shared, model.warp_condition, model.warp,
    model.atlas_encoder, model.pair)
joint_matcher_modules = (fitter.matcher.image, fitter.matcher.atlas,
    fitter.matcher.spatial, fitter.matcher.mapper_feature)
for layer in head_modules:
    layer.requires_grad_(True)
groups = [
    {'params': list(head.parameters()), 'lr': 3e-5, 'base_lr': 3e-5}
    for head in head_modules
] + [
    {'params': list(layer.parameters()), 'lr': 0., 'base_lr': 3e-6}
    for layer in joint_model_modules
] + [
    {'params': list(layer.parameters()), 'lr': 0., 'base_lr': 5e-6}
    for layer in joint_matcher_modules
] + [
    {'params': [fitter.matcher.log_temperature, fitter.log_fit_temperature],
     'lr': 0., 'base_lr': 5e-6}
]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
trainable = [parameter for group in groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
attempts = 0
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')
local_index = torch.zeros((1, 1), dtype=torch.long, device='cuda')

config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 4,
    'candidate_rule': 'physical-best, support-nearest >=0.5mm worse, highest-prior >=1mm worse, remaining random; documented nonduplicate fallbacks',
    'score': 'log_prior - fit_energy / exp(clamp(log_fit_temperature,log(0.5),log(5))) - 0.05*warp_cost',
    'warmup': '2000 accepted sections; only new correct-match and image-validity heads',
    'loss': 'full-bank direct + best mapped96/warp + balanced coarse/fine correct-match BCE + image validity BCE + valid-bin CE + four-branch physical-error rank',
    'correct_match_tolerance_um': {'coarse': 2000, 'fine': 1500},
    'source_shape': (side, side), 'mapped_shape': (96, 96),
    'parent_092': str(parent), 'parent_092_sha256': sha(parent),
    'parent_092_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_092_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_spatial_verifier_094.py', 'spatial_joint_fit_094.py',
        'arbitrary_plane_one_shot_model.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py', 'global_atlas_contrast_090.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
        'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'synthetic_provenance': context['provenance'], 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'spatial_fit': fitter.state_dict(),
        'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.get_state(),
        'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step,
        'config': config, 'calibrated': False}, run / f'joint_step_{step:05d}.pt')


save(0)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
audit_nonzero = {'old_pose': 0, 'anchor_pose': 0, 'prior_logits': 0}
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        if step == 2001:
            for layer in joint_model_modules + joint_matcher_modules:
                layer.requires_grad_(True)
            fitter.matcher.log_temperature.requires_grad_(True)
            fitter.log_fit_temperature.requires_grad_(True)
        stage = 'heads' if step <= 2000 else 'joint'
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            record = sampled['provenance'][0]
            used = bool(sampled['eligible'][0])
            attempts += 1
            draws.write(json.dumps({**record, 'step': step, 'draw_attempt': attempts,
                                    'used': used}) + '\n')
            draw_seed += 1
            if used:
                accepted = {key: sampled[key] for key in (
                    'inputs', 'state', 'reflection', 'offsets', 'weights',
                    'centre', 'valid_mask')}

        optimizer.zero_grad(set_to_none=True)
        if stage == 'heads':
            with torch.no_grad():
                prediction = model.predict(accepted['inputs'])
        else:
            prediction = model.predict(accepted['inputs'])
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags[None, None].expand(1, model.modes, 2)
        pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                                  pixels, replacement=True)
        target = accepted['centre'].reshape(1, -1, 3).gather(
            1, pixel[..., None].expand(-1, -1, 3))
        chart = torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                            -1).float() / side
        reference = rigid_points_090(accepted['state'], accepted['reflection'], corners)
        five = (rigid_points_090(states, reflections, corners) - reference[:, None, None]
                ).norm(dim=-1).mean(-1)
        dense = (rigid_points_090(states, reflections, chart) - target[:, None, None]
                 ).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        physical_mm = (.75 * dense + .25 * five + normal_penalty[..., None]).flatten(1) / 1000
        rigid_mm = dense.flatten(1) / 1000
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = physical_mm.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
        target_mass = F.softmax(-physical_mm.detach() / .7, -1)
        direct = (-1.5 * torch.logsumexp(prior - physical_mm / 1.5, -1).mean()
                  + .5 * F.kl_div(prior, target_mass, reduction='batchmean')
                  + .5 * physical_mm[:, :2 * model.base_modes].min(-1).values.mean()
                  + (neighbourhood * near_distance).sum(-1).mean())

        with torch.no_grad():
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            initial = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
            atlas_pair = render_atlas_planes_090(context['atlas'], initial, beam % 2,
                accepted['offsets'], accepted['weights'], candidate_chunk=2)
            support = atlas_pair[:, :, 1].mean((-2, -1))[0]
            beam_error = rigid_mm.gather(1, beam)[0]
            beam_prior = prior.gather(1, beam)[0]
            best = int(beam_error.argmin())
            gap = beam_error - beam_error[best]
            remaining = [index for index in range(14) if index != best]
            close = [index for index in remaining if gap[index] >= .5]
            support_fallback = not close
            if not close:
                close = remaining
            support_choice = min(close, key=lambda index: float(abs(support[index] - support[best])))
            remaining.remove(support_choice)
            wrong = [index for index in remaining if gap[index] >= 1.]
            prior_fallback = not wrong
            if not wrong:
                wrong = remaining
            prior_choice = max(wrong, key=lambda index: float(beam_prior[index]))
            remaining.remove(prior_choice)
            fourth = remaining[int(torch.randint(len(remaining), (1,), generator=subject_rng))]
            slots = torch.tensor([best, support_choice, prior_choice, fourth], device='cuda')
            choice = beam[:, slots]
            chosen_rigid_mm = beam_error[slots]
            chosen_support = support[slots]

        reflected = choice % 2
        mode = choice // 2
        chosen_state = prediction['state'].gather(1, mode[..., None].expand(-1, -1, 12))
        fitted = fitter(prediction['feature'], chosen_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side))
        truth_coarse = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
            chosen_state, reflected)
        truth_fine = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
            fitted['state'], reflected)
        correspondence_loss = 0.
        correct_loss = 0.
        valid_counts, correct_counts = {}, {}
        for name, size, tolerance, truth in (
            ('coarse', 16, 2000, truth_coarse),
            ('fine', 32, 1500, truth_fine)):
            logits = fitted[f'{name}_match_logits']
            support_map = fitted[f'{name}_match_support']
            index = truth[f'{name}_index']
            support_at_truth = support_map.gather(2, index[:, :, None]).squeeze(2)
            valid = truth[f'{name}_mask'] & (support_at_truth >= .5)
            ce = F.cross_entropy(logits.flatten(0, 1), index.flatten(0, 1),
                                 reduction='none').reshape_as(valid)
            correspondence_loss = correspondence_loss + .5 * (ce * valid).sum() / valid.sum().clamp_min(1)
            truth_ccf = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                (size, size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            surviving = F.interpolate(accepted['valid_mask'][:, None].float(),
                (size, size), mode='bilinear', align_corners=False)[:, 0] == 1
            correct = (surviving[:, None] & (support_map.amax(2) >= .5) &
                ((fitted[f'{name}_match_ccf_um'].detach() - truth_ccf[:, None])
                 .norm(dim=-1) <= tolerance)).float()
            bce = F.binary_cross_entropy_with_logits(
                fitted[f'{name}_correct_logit'], correct, reduction='none')
            correct_loss = correct_loss + .25 * (
                (bce * correct).sum() / correct.sum().clamp_min(1)
                + (bce * (1 - correct)).sum() / (1 - correct).sum().clamp_min(1))
            valid_counts[name] = int(valid.sum())
            correct_counts[name] = int(correct.sum())
        valid_truth = F.interpolate(accepted['valid_mask'][:, None].float(),
            (32, 32), mode='bilinear', align_corners=False)[:, 0]
        validity_loss = F.binary_cross_entropy_with_logits(fitted['validity_logit'], valid_truth)
        temperature = fitter.log_fit_temperature.clamp(math.log(.5), math.log(5.)).exp()

        if stage == 'joint':
            selected = {**prediction, 'state': fitted['state'][:, :1],
                'log_mass': prediction['log_mass'].gather(1, mode[:, :1]),
                'reflection_logit': prediction['reflection_logit'].gather(1, mode[:, :1])}
            mapped = model.map(selected, accepted['offsets'], local_index, reflected[:, :1],
                (96, 96), context['atlas'], accepted['weights'], feature_side=96,
                source_shape=(side, side), spatial_evidence=fitted['spatial_evidence'][:, :1])
            grid = (torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                                -1).float() + .5) * (2 / side) - 1
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
            placed = F.grid_sample(surface, grid[:, None], padding_mode='border',
                                   align_corners=False).reshape(1, 3, pixels).transpose(1, 2)
            mapped_mm = (placed - target).norm(dim=-1).mean() / 1000
            local = mapped['local_displacement_um'] / 1000
            dx = local[..., 1:] - local[..., :-1]
            dy = local[..., 1:, :] - local[..., :-1, :]
            warp_cost = local.square().mean() + .1 * (dx.square().mean() + dy.square().mean())
            warp_penalty = .03 * local.square().mean() + .01 * (dx.square().mean() + dy.square().mean())
            scores = prior.gather(1, choice) - fitted['fit_energy'] / temperature - .05 * warp_cost
            pair_terms = []
            for left in range(4):
                for right in range(left + 1, 4):
                    delta = chosen_rigid_mm[right] - chosen_rigid_mm[left]
                    if abs(float(delta)) >= .5:
                        pair_terms.append(F.softplus(1 - delta.sign() * (scores[0, left] - scores[0, right])))
            rank_loss = torch.stack(pair_terms).mean() if pair_terms else scores.sum() * 0
            loss = direct + mapped_mm + warp_penalty + .1 * correspondence_loss \
                + .2 * correct_loss + .2 * validity_loss + .25 * rank_loss
        else:
            mapped_mm = fitted['fit_energy'].new_zeros(())
            warp_cost = fitted['fit_energy'].new_zeros(())
            warp_penalty = fitted['fit_energy'].new_zeros(())
            scores = prior.gather(1, choice) - fitted['fit_energy'] / temperature
            rank_loss = scores.sum() * 0
            loss = .1 * correct_loss + .1 * validity_loss

        audit = {}
        if stage == 'joint' and (step == 2001 or step % 200 == 0):
            pose_parameters = (model.pose[-1].weight, model.anchor_pose[-1].weight,
                               model.anchor_global[-1].weight)
            fit_grads = torch.autograd.grad(fitted['fit_energy'].sum(), pose_parameters,
                retain_graph=True, allow_unused=True)
            audit['fit_only_pose_grad_norms'] = {
                name: 0. if grad is None else float(grad.norm())
                for name, grad in zip(('old_pose', 'anchor_pose', 'anchor_global'), fit_grads)}
            prior_grads = torch.autograd.grad(rank_loss,
                (prediction['log_mass'], prediction['reflection_logit']),
                retain_graph=True, allow_unused=True)
            audit['rank_prior_logit_grad_norm'] = sum(
                0. if grad is None else float(grad.norm()) for grad in prior_grads)
            audit_nonzero['old_pose'] += audit['fit_only_pose_grad_norms']['old_pose'] > 0
            audit_nonzero['anchor_pose'] += (audit['fit_only_pose_grad_norms']['anchor_pose']
                                            + audit['fit_only_pose_grad_norms']['anchor_global']) > 0
            audit_nonzero['prior_logits'] += audit['rank_prior_logit_grad_norm'] > 0
        progress = (step / 2000) if stage == 'heads' else ((step - 2000) / 18000)
        factor = .2 + .8 * .5 * (1 + math.cos(math.pi * progress))
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = (group['base_lr'] if stage == 'joint' or index < len(head_modules) else 0.) * factor
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'stage': stage, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'rigid_error_mm': chosen_rigid_mm.tolist(), 'support': chosen_support.tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'mapped96_error_mm': float(mapped_mm.detach()),
            'score': scores[0].detach().tolist(),
            'fit_energy': fitted['fit_energy'][0].detach().tolist(),
            'quality': fitted['quality'][0].detach().tolist(),
            'coverage': fitted['coverage'][0].detach().tolist(),
            'pose_correction_cost': fitted['pose_correction_cost'][0].detach().tolist(),
            'warp_cost': float(warp_cost.detach()),
            'valid_bin_count': valid_counts, 'correct_match_count': correct_counts,
            'direct': float(direct.detach()), 'valid_bin_ce': float(correspondence_loss.detach()),
            'correct_match_bce': float(correct_loss.detach()),
            'image_validity_bce': float(validity_loss.detach()),
            'warp_penalty': float(warp_penalty.detach()),
            'pair_rank_loss': float(rank_loss.detach()), 'total_loss': float(loss.detach()),
            'fit_temperature': float(temperature.detach()), 'gradient_norm': float(gradient),
            'learning_rates': [group['lr'] for group in optimizer.param_groups],
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started, **audit}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 200 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'stage',
                'rigid_error_mm', 'mapped96_error_mm', 'valid_bin_ce',
                'correct_match_bce', 'image_validity_bce', 'pair_rank_loss',
                'gradient_norm', 'peak_gpu_mb', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'draw_attempts': attempts,
    'fit_gradient_audits_nonzero': audit_nonzero,
    'protocol_sha256': sha(protocol), 'parent_092_checkpoint_sha256': sha(parent),
    'source_sha256': config['source_sha256'], 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
