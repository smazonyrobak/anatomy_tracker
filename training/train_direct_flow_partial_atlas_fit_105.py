"""TRAIN-only continuous-flow continuation of the standalone atlas fitter."""
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
from training.direct_flow_partial_atlas_fit_105 import DirectFlowPartialAtlasFit105
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083, synthetic_match_targets

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/DIRECT_FLOW_PARTIAL_ATLAS_FIT_105_PROTOCOL_20261009.md'
pose_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
parent_run = root / 'runs/coherent_partial_atlas_fit_104'
pose_parent = pose_run / 'joint_step_02000.pt'
fit_parent = parent_run / 'fit_step_01000.pt'
run = root / 'runs/direct_flow_partial_atlas_fit_105'
seed, updates, side = 20261009105, 4000, 256
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
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
assert sha(pose_parent) == pose_receipt['checkpoint_sha256']['2000']
assert sha(fit_parent) == parent_receipt['checkpoint_sha256']['1000']
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
pose_checkpoint = torch.load(pose_parent, map_location='cpu', weights_only=True)
parent_checkpoint = torch.load(fit_parent, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(pose_checkpoint['model'], strict=True)
fitter = DirectFlowPartialAtlasFit105(WholeSliceAtlasFeedback083()).cuda().train()
parent_state = parent_checkpoint['spatial_fit']
frozen_output = {key: parent_state[key] for key in
    ('field_output.weight', 'field_output.bias')}
missing, unexpected = fitter.load_state_dict({key: value for key, value in
    parent_state.items() if not key.startswith('field_output.')}, strict=False)
assert not unexpected and set(missing) == set(frozen_output)
with torch.no_grad():
    fitter.field_output.weight[:226].copy_(frozen_output['field_output.weight'][:226])
    fitter.field_output.bias[:226].copy_(frozen_output['field_output.bias'][:226])
del pose_checkpoint, parent_checkpoint, parent_state, frozen_output
fitter.requires_grad_(False)
field = (fitter.field_input, fitter.field_gates, fitter.field_proposal, fitter.field_output)
matcher = (fitter.matcher.image, fitter.matcher.atlas, fitter.matcher.log_temperature)
for module in field:
    module.requires_grad_(True)
fitter.validity_head.requires_grad_(True)
fitter.log_fit_temperature.requires_grad_(True)
groups = [{'params': list(module.parameters()), 'base_lr': 1e-4, 'from_step': 1}
          for module in field]
groups += [{'params': list(fitter.validity_head.parameters()),
            'base_lr': 1e-5, 'from_step': 1}]
groups += [{'params': list(module.parameters()), 'base_lr': 5e-6,
            'from_step': 1001} for module in matcher[:2]]
groups += [{'params': [matcher[2]], 'base_lr': 2e-6, 'from_step': 1001},
           {'params': [fitter.log_fit_temperature], 'base_lr': 2e-5, 'from_step': 1}]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
parameters = [parameter for group in groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
flags = torch.tensor((0, 1), device='cuda')
source_files = ('train_direct_flow_partial_atlas_fit_105.py',
    'direct_flow_partial_atlas_fit_105.py', 'spatial_joint_fit_094.py',
    'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py',
    'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'checkpoints': checkpoints,
    'beam': {'old': 8, 'anchor': 6}, 'train_blind_candidates': 4,
    'teacher': 'one TRAIN-only target-state candidate per independently drawn section; exact on odd updates, jittered on even updates',
    'jitter_limits': {'rotation_rad': .04, 'translation_um': 500,
        'basis_log': .025, 'shear': .015},
    'warmup': '1-1000 recurrent field and validity; 1001-4000 image/atlas descriptors too',
    'loss': '.5 balanced match/dustbin CE + map Huber + .05 identity BCE + .05 validity BCE + .02 fragment-aware smoothness + .5 blind rank',
    'pose_parent': str(pose_parent), 'pose_parent_sha256': sha(pose_parent),
    'fit_parent': str(fit_parent), 'fit_parent_sha256': sha(fit_parent),
    'pose_completed_sha256': sha(pose_run / 'completed.json'),
    'fit_completed_sha256': sha(parent_run / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'spatial_fit': fitter.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'fit_step_{step:05d}.pt')


save(0)
attempts = 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        if step == 1001:
            for module in matcher[:2]:
                module.requires_grad_(True)
            matcher[2].requires_grad_(True)
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
            prediction = model.predict(accepted['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags[None, None].expand(1, model.modes, 2)
            pixel = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                128, replacement=True)
            target = accepted['centre'].reshape(1, -1, 3).gather(
                1, pixel[..., None].expand(-1, -1, 3))
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            physical_mm = (rigid_points_090(states, reflections, chart) -
                target[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            beam_state = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
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
            support_choice = min(options, key=lambda index: float(abs(support[index] - support[best])))
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
        fitted = fitter(prediction['feature'], candidate_state, reflected,
            context['atlas'], accepted['offsets'], accepted['weights'], source_shape=(side, side))
        with torch.no_grad():
            truth = synthetic_match_targets(accepted['centre'], accepted['valid_mask'],
                fitted['state'], reflected)
        truth_index = truth['fine_index']
        supported = fitted['fine_match_support'].gather(
            2, truth_index[:, :, None]).squeeze(2) >= .5
        surviving = F.interpolate(accepted['valid_mask'][:, None].float(),
            (32, 32), mode='bilinear', align_corners=False)[:, 0] == 1
        matched = surviving[:, None] & truth['fine_mask'] & supported
        unmatched = surviving[:, None] & ~matched
        label = torch.where(matched, truth_index, 225).flatten(0, 1)
        ce = F.cross_entropy(fitted['coherent_match_logits'].flatten(0, 1),
            label, reduction='none').reshape(1, 5, 32, 32)
        match_loss = .5 * (ce * matched).sum() / matched.sum().clamp_min(1) + \
            .5 * (ce * unmatched).sum() / unmatched.sum().clamp_min(1)
        target_ccf = F.interpolate(accepted['centre'].permute(0, 3, 1, 2),
            (32, 32), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        map_mm = (fitted['coherent_match_ccf_um'] - target_ccf[:, None]) / 1000
        map_site = F.smooth_l1_loss(map_mm, torch.zeros_like(map_mm),
            reduction='none', beta=.25).sum(-1)
        map_weight = matched.float().clone()
        map_weight[:, 4] *= 3
        map_loss = (map_site * map_weight).sum() / map_weight.sum().clamp_min(1)
        axis = (torch.arange(32, device='cuda') + .5) / 32 - .5 / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        map_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
        rigid = rigid_points_090(fitted['state'], reflected, map_chart).reshape(1, 5, 32, 32, 3)
        no_shift = (target_ccf[:, None] - rigid).norm(dim=-1) < 200
        identity_site = F.binary_cross_entropy(
            fitted['coherent_identity_probability'].clamp(1e-5, 1 - 1e-5),
            no_shift.float(), reduction='none')
        identity_loss = (identity_site * matched).sum() / matched.sum().clamp_min(1)
        valid_truth = F.interpolate(accepted['valid_mask'][:, None].float(),
            (32, 32), mode='bilinear', align_corners=False)[:, 0]
        validity_loss = F.binary_cross_entropy_with_logits(
            fitted['validity_logit'], valid_truth)
        local_mm = fitted['coherent_local_displacement_um'] / 1000
        fragment = F.interpolate(accepted['fragment_displaced_mask'][:, None].float(),
            (32, 32), mode='nearest')[:, 0] > .5
        x_pair = (surviving[:, None, :, 1:] & surviving[:, None, :, :-1]
                  & (fragment[:, None, :, 1:] == fragment[:, None, :, :-1])).expand(-1, 5, -1, -1)
        y_pair = (surviving[:, None, 1:] & surviving[:, None, :-1]
                  & (fragment[:, None, 1:] == fragment[:, None, :-1])).expand(-1, 5, -1, -1)
        dx = (local_mm[:, :, :, 1:] - local_mm[:, :, :, :-1]).square().sum(-1)
        dy = (local_mm[:, :, 1:] - local_mm[:, :, :-1]).square().sum(-1)
        smooth_loss = .5 * ((dx * x_pair).sum() / x_pair.sum().clamp_min(1)
            + (dy * y_pair).sum() / y_pair.sum().clamp_min(1))
        temperature = fitter.log_fit_temperature.clamp(math.log(.3), math.log(3.)).exp()
        scores = prior.gather(1, choice) - fitted['coherent_energy'][:, :4] / temperature
        pairs = [F.softplus(.5 - (scores[0, left] - scores[0, right])
                 * (chosen_error[right] - chosen_error[left]).sign())
                 for left in range(4) for right in range(left + 1, 4)
                 if abs(float(chosen_error[right] - chosen_error[left])) >= .5]
        rank_loss = torch.stack(pairs).mean() if pairs else scores.sum() * 0
        loss = .5 * match_loss + map_loss + .05 * identity_loss + \
            .05 * validity_loss + .02 * smooth_loss + .5 * rank_loss
        progress = step / updates
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay if step >= group['from_step'] else 0.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'physical_section_id': record['physical_section_id'],
            'appearance_mode': record['mode'], 'candidate_ids': choice[0].tolist(),
            'candidate_error_mm': chosen_error.tolist(), 'support': chosen_support.tolist(),
            'support_fallback': support_fallback, 'prior_fallback': prior_fallback,
            'teacher_exact': step % 2 == 1, 'teacher_jitter': jitter[0].tolist(),
            'matched_sites': int(matched.sum()), 'teacher_matched_sites': int(matched[:, 4].sum()),
            'unmatched_surviving_sites': int(unmatched.sum()),
            'match_ce': float(match_loss.detach()), 'map_huber_mm': float(map_loss.detach()),
            'teacher_map_huber_mm': float((map_site[:, 4] * matched[:, 4]).sum().detach() /
                matched[:, 4].sum().clamp_min(1)),
            'identity_bce': float(identity_loss.detach()),
            'identity_positive_fraction': float((no_shift * matched).sum() / matched.sum().clamp_min(1)),
            'validity_bce': float(validity_loss.detach()),
            'fragment_aware_smoothness_mm2': float(smooth_loss.detach()),
            'rank_loss': float(rank_loss.detach()), 'total_loss': float(loss.detach()),
            'mean_abs_flow_cells': float(fitted['coherent_offset_cells_xyz'].abs().mean().detach()),
            'fit_energy': fitted['coherent_energy'][0].detach().tolist(),
            'scores': scores[0].detach().tolist(), 'temperature': float(temperature.detach()),
            'gradient_norm': float(gradient),
            'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1048576,
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('update', 'matched_sites',
                'teacher_matched_sites', 'match_ce', 'map_huber_mm', 'teacher_map_huber_mm',
                'mean_abs_flow_cells', 'rank_loss', 'total_loss', 'peak_gpu_mb', 'seconds')}), flush=True)
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
    'checkpoint_sha256': {str(step): sha(run / f'fit_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'draw_attempts': attempts, 'seconds': time.perf_counter() - started}), flush=True)
