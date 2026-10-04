"""Joint TRAIN-only pose, spatial correspondence, fitted ranking, and local map."""
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
from training.spatial_joint_fit_092 import SpatialJointFit092
from training.whole_slice_atlas_feedback_083 import (
    WholeSliceAtlasFeedback083, synthetic_match_targets,
)

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SPATIAL_JOINT_FIT_092_PROTOCOL_20261004.md'
parent_run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
parent = parent_run / 'joint_step_06000.pt'
run = root / 'runs/spatial_joint_fit_092_pilot'
seed, batches, side, pixels = 2026100492, 6000, 256, 1024
checkpoints = (0, 2000, 6000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '2d6c8eccb552d42d7bce1e69210244157ad6dbf7994df7007f741df0c7066c67'
receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert receipt['batches'] == 6000 and not any(receipt[key] for key in (
    'calibrated', 'public_benchmark_used', 'real_labels_used', 'external_pretrained_weights_used'))
assert sha(parent) == receipt['checkpoint_sha256']['6000']
assert sha(parent_run / 'config.json') == receipt['config_sha256']
assert sha(parent_run / 'draws.jsonl') == receipt['draws_sha256']
assert sha(parent_run / 'training.jsonl') == receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
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
matcher.load_state_dict(checkpoint['feedback'], strict=True)
del checkpoint
fitter = SpatialJointFit092(matcher).cuda().train()
model.requires_grad_(False)
fitter.requires_grad_(False)
for layer in (model.pose[-1], model.anchor_pose[-1], model.anchor_global[-1],
              model.warp_shared, model.warp_condition, model.warp, model.pair,
              fitter.matcher.image, fitter.matcher.atlas, fitter.matcher.spatial,
              fitter.matcher.mapper_feature, fitter.reliability):
    layer.requires_grad_(True)
fitter.matcher.log_temperature.requires_grad_(True)
fitter.log_fit_temperature.requires_grad_(True)
groups = (
    (model.pose[-1].parameters(), 1e-5),
    (model.anchor_pose[-1].parameters(), 2e-5),
    (model.anchor_global[-1].parameters(), 2e-5),
    (model.warp_shared.parameters(), 1e-5),
    (model.warp_condition.parameters(), 1e-5),
    (model.warp.parameters(), 1e-5),
    (model.pair.parameters(), 1e-5),
    (fitter.matcher.image.parameters(), 1e-5),
    (fitter.matcher.atlas.parameters(), 1e-5),
    (fitter.matcher.spatial.parameters(), 2e-5),
    (fitter.matcher.mapper_feature.parameters(), 2e-5),
    (fitter.reliability.parameters(), 3e-5),
    ([fitter.matcher.log_temperature, fitter.log_fit_temperature], 1e-5),
)
optimizer = torch.optim.AdamW([{'params': list(parameters), 'lr': rate}
                               for parameters, rate in groups], weight_decay=1e-4)
rates = [rate for _, rate in groups]
trainable = [parameter for group in optimizer.param_groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
attempts = 0
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')
local_index = torch.arange(2, device='cuda')[None]

config = {'seed': seed, 'batches': batches, 'checkpoints': checkpoints,
    'candidate_beam': {'old': 8, 'anchor': 6}, 'train_candidates': 2,
    'pair': 'lowest rigid surviving-pixel CCF error vs support-nearest existing branch >=0.5 mm worse; farthest-error fallback',
    'score': 'log_prior - fit_energy / exp(clamp(log_fit_temperature,log(0.5),log(5))) - 0.05*warp_cost',
    'warp_cost': 'mean(local_mm^2) + 0.1*(mean(dx_local_mm^2)+mean(dy_local_mm^2))',
    'loss': '089 full-bank direct + mean mapped96_mm + 089 warp penalty + 0.1*mean(valid coarse/fine CE) + 0.1*mean(correct-match coarse/fine BCE) + 0.25*pair softplus margin 1',
    'reliability_target_um': {'coarse': 2000, 'fine': 1500},
    'reliability_target': 'predicted local-mode CCF within threshold of synthetic observed CCF, supported>=0.5, surviving tissue; all other sites negative',
    'spatial_map': '083 full-forward spatial_evidence at 092 fitted state; ignore 083 pose update; 089 map at 96 grid',
    'source_shape': (side, side), 'candidate_support_renderer': '090 finite-PSF renderer only, no global scorer',
    'parent_089': str(parent), 'parent_089_sha256': sha(parent),
    'parent_089_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_089_config_sha256': sha(parent_run / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'train_spatial_joint_fit_092.py', 'spatial_joint_fit_092.py',
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
            support = atlas_pair[:, :, 1].mean((-2, -1))
            beam_error = rigid_mm.gather(1, beam)
            best = beam_error.argmin(-1)
            gap = beam_error - beam_error.gather(1, best[:, None])
            eligible = gap >= .5
            support_gap = (support - support.gather(1, best[:, None])).abs()
            matched = support_gap.masked_fill(~eligible, float('inf')).argmin(-1)
            fallback = ~eligible.any(-1)
            farthest = gap.masked_fill(torch.arange(14, device='cuda')[None] == best[:, None],
                                       -float('inf')).argmax(-1)
            worse = torch.where(fallback, farthest, matched)
            slots = torch.stack((best, worse), -1)
            choice = beam.gather(1, slots)
            chosen_support = support.gather(1, slots)
            chosen_rigid_mm = beam_error.gather(1, slots)

        reflected = choice % 2
        mode = choice // 2
        chosen_state = prediction['state'].gather(1, mode[..., None].expand(-1, -1, 12))
        fitted = fitter(prediction['feature'], chosen_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side))
        evidence = fitter.matcher(prediction['feature'], fitted['state'], reflected,
            context['atlas'], accepted['offsets'], accepted['weights'],
            source_shape=(side, side))['spatial_evidence']
        selected = {**prediction, 'state': fitted['state'],
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}
        mapped = model.map(selected, accepted['offsets'], local_index, reflected,
            (96, 96), context['atlas'], accepted['weights'], feature_side=96,
            source_shape=(side, side), spatial_evidence=evidence)
        grid = (torch.stack((pixel.remainder(side), pixel.div(side, rounding_mode='floor')),
                            -1).float() + .5) * (2 / side) - 1
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        placed = F.grid_sample(surface, grid[:, None].expand(2, -1, -1, -1),
                               padding_mode='border', align_corners=False).reshape(
                                   2, 3, pixels).transpose(1, 2)
        mapped_mm = (placed - target[0, None]).norm(dim=-1).mean(-1) / 1000
        local = mapped['local_displacement_um'] / 1000
        dx = local[..., 1:] - local[..., :-1]
        dy = local[..., 1:, :] - local[..., :-1, :]
        warp_cost = local.square().mean((-3, -2, -1)) + .1 * (
            dx.square().mean((-3, -2, -1)) + dy.square().mean((-3, -2, -1)))
        warp_penalty = (.03 * local.square().mean() + .01 * dx.square().mean()
                        + .01 * dy.square().mean())
        temperature = fitter.log_fit_temperature.clamp(math.log(.5), math.log(5.)).exp()
        scores = prior.gather(1, choice) - fitted['fit_energy'] / temperature - .05 * warp_cost
        mapped_order = (mapped_mm[1] - mapped_mm[0]).detach().sign()
        rank_loss = F.softplus(1 - mapped_order * (scores[0, 0] - scores[0, 1]))

        losses, valid_counts, correct_counts = [], {}, {}
        coarse_targets = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
            chosen_state, reflected)
        fine_targets = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
            fitted['state'], reflected)
        for scale, size, tolerance, targets in (
            ('coarse', 16, 2000, coarse_targets), ('fine', 32, 1500, fine_targets)):
            logits = fitted[f'{scale}_match_logits']
            support_map = fitted[f'{scale}_match_support']
            index = targets[f'{scale}_index']
            support_at_truth = support_map.gather(2, index[:, :, None]).squeeze(2)
            valid = targets[f'{scale}_mask'] & (support_at_truth >= .5)
            ce = F.cross_entropy(logits.flatten(0, 1), index.flatten(0, 1),
                                 reduction='none').reshape_as(valid)
            match_ce = (ce * valid).sum() / valid.sum().clamp_min(1)
            truth_ccf = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
                (size, size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            peak = logits.masked_fill(support_map < .5, -1e4).argmax(2)
            peak_support = support_map.gather(2, peak[:, :, None]).squeeze(2) >= .5
            correct = (valid & peak_support &
                ((fitted[f'{scale}_match_ccf_um'].detach() - truth_ccf[:, None])
                 .norm(dim=-1) <= tolerance)).float()
            bce = F.binary_cross_entropy_with_logits(
                fitted[f'{scale}_reliability_logit'], correct, reduction='none')
            match_bce = .5 * ((bce * correct).sum() / correct.sum().clamp_min(1)
                + (bce * (1 - correct)).sum() / (1 - correct).sum().clamp_min(1))
            losses.append((match_ce, match_bce))
            valid_counts[scale] = int(valid.sum())
            correct_counts[scale] = int(correct.sum())
        ce_loss = .5 * (losses[0][0] + losses[1][0])
        reliability_loss = .5 * (losses[0][1] + losses[1][1])
        loss = direct + mapped_mm.mean() + warp_penalty + .1 * ce_loss \
               + .1 * reliability_loss + .25 * rank_loss

        audit = {}
        if step == 1 or step % 100 == 0:
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
        progress = step / batches
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * (.2 + .8 * .5 * (1 + math.cos(math.pi * progress)))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'draw_attempt': attempts,
            'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'support': chosen_support[0].tolist(), 'support_fallback': bool(fallback[0]),
            'rigid_error_mm': chosen_rigid_mm[0].tolist(),
            'mapped96_error_mm': mapped_mm.detach().tolist(),
            'score': scores[0].detach().tolist(),
            'fit_energy': fitted['fit_energy'][0].detach().tolist(),
            'warp_cost': warp_cost[0].detach().tolist(),
            'coarse_reliability_mean': float(fitted['coarse_reliability_logit'].sigmoid().mean().detach()),
            'fine_reliability_mean': float(fitted['fine_reliability_logit'].sigmoid().mean().detach()),
            'valid_bin_count': valid_counts, 'correct_match_count': correct_counts,
            'direct': float(direct.detach()), 'mapped_loss': float(mapped_mm.mean().detach()),
            'warp_penalty': float(warp_penalty.detach()), 'valid_bin_ce': float(ce_loss.detach()),
            'correct_match_bce': float(reliability_loss.detach()),
            'pair_rank_loss': float(rank_loss.detach()), 'total_loss': float(loss.detach()),
            'fit_temperature': float(temperature.detach()),
            'gradient_norm': float(gradient),
            'learning_rates': [group['lr'] for group in optimizer.param_groups],
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'seconds': time.perf_counter() - started, **audit}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 100 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in (
                'batch', 'rigid_error_mm', 'mapped96_error_mm', 'score',
                'valid_bin_ce', 'correct_match_bce', 'pair_rank_loss',
                'gradient_norm', 'peak_gpu_mb', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'draw_attempts': attempts,
    'fit_gradient_audits_nonzero': audit_nonzero,
    'protocol_sha256': sha(protocol), 'parent_089_checkpoint_sha256': sha(parent),
    'source_sha256': config['source_sha256'], 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
