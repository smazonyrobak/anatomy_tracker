"""Joint pose, atlas-feedback, and tissue-mapping development run on blind proposals."""
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
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.whole_slice_atlas_feedback_081 import WholeSliceAtlasFeedback081

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/whole_slice_joint_feedback_081_pilot'
seed, batches, warmup, side, candidates = 2026100481, 3000, 500, 256, 3
evaluation_steps = (0, 500, 1500, 3000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < batches:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == batches:
                break
assert len(set(schedule)) == batches
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, np.int32))
config = {
    'seed': seed, 'batches': batches, 'frozen_parent_warmup_batches': warmup,
    'synthetic_per_batch': 1, 'real_weak_per_batch': 1,
    'training_candidates': '3 from actual 8-old/6-anchor beam: prior-top old, prior-top anchor, physical-best existing branch',
    'evaluation_candidates': {'old': 8, 'anchor': 6},
    'shared_feedback_passes': 2, 'first_map_side': 64, 'final_map_side': 96,
    'fit_summary': ['warp_magnitude_mm', 'warp_roughness_mm', 'atlas_support', 'correspondence_reliability'],
    'real_label_role': real['label_role'], 'real_bindings': real['bindings'],
    'real_schedule_sha256': sha(run / 'real_schedule.npy'),
    'earlier_training_real_reuse_allowed': True,
    'synthetic_provenance': context['provenance'],
    'parent': str(parent), 'parent_sha256': sha(parent),
    'evaluation_steps': evaluation_steps,
    'development_gate': {'synthetic_selected_improvement_mm_at_least': .4,
                         'synthetic_best14_mm_at_most': .9,
                         'real_weak_donor_mean_regression_tolerance_mm': .1},
    'calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        'train_whole_slice_joint_feedback_081.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_full_frame_primitives.py')},
}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
frozen = torch.load(parent, map_location='cpu', weights_only=True)
assert frozen['step'] == 50000 and not frozen['calibrated']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda()
model.load_state_dict(frozen['model'], strict=True)
del frozen
model.requires_grad_(False)
model.eval()
feedback = WholeSliceAtlasFeedback081().cuda().train()
optimizer = torch.optim.AdamW((
    {'params': model.parameters(), 'lr': 1e-5},
    {'params': feedback.parameters(), 'lr': 1e-4}), weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def save(step):
    torch.save({'model': model.state_dict(), 'feedback': feedback.state_dict(),
        'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.get_state(),
        'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step,
        'config': config, 'calibrated': False}, run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        if step == warmup + 1:
            model.requires_grad_(True)
            model.train()
        accepted = None
        while accepted is None:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            used = bool(sampled['eligible'][0])
            draws.write(json.dumps({**sampled['provenance'][0], 'step': step,
                                     'slot': 0, 'used': used}) + '\n')
            if used:
                accepted = {key: sampled[key] for key in
                    ('inputs', 'state', 'reflection', 'offsets', 'weights',
                     'centre', 'valid_mask')}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
            'slot': 1, 'used': True, 'label_role': real['label_role']}) + '\n')
        real_input = F.interpolate(observation['inputs'], (side, side),
                                   mode='bilinear', align_corners=False)
        image = torch.cat((accepted['inputs'], real_input))
        real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
        real_weights = torch.ones_like(real_offsets)
        real_weights[:, [0, -1]] = .5
        real_weights /= real_weights.sum(-1, keepdim=True)
        offsets = torch.cat((accepted['offsets'], real_offsets))
        weights = torch.cat((accepted['weights'], real_weights))
        indices = torch.multinomial(accepted['valid_mask'].flatten(1).float(),
                                    96, replacement=True)
        target = accepted['centre'].reshape(1, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() / side
        weak_reference = points(observation['state'], observation['reflection'], corners)
        with torch.set_grad_enabled(step > warmup):
            prediction = model.predict(image)
            all_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.arange(2, device='cuda')[None, None].expand(2, model.modes, 2)
            synthetic_distance = (points(all_states[:1], flags[:1], chart[:, None, None])
                - target[:, None, None]).norm(dim=-1).mean(-1)
            real_distance = (points(all_states[1:], flags[1:], corners)
                - weak_reference[:, None, None]).norm(dim=-1).mean(-1)
            true_normal = full_frame_state_to_components(accepted['state'])[1][..., :, 2]
            estimated_normal = full_frame_state_to_components(prediction['state'][:1])[1][..., :, 2]
            synthetic_distance = synthetic_distance + 4000 * (
                1 - (estimated_normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))[..., None]
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            distance = torch.cat((synthetic_distance.flatten(1), real_distance.flatten(1)))
            old = prior[:, :32].topk(8, -1).indices
            anchor = prior[:, 32:].topk(6, -1).indices + 32
            beam = torch.cat((old, anchor), -1)
            best = distance.gather(1, beam).argmin(-1)
            selected_slot = torch.stack((torch.zeros_like(best),
                                         torch.full_like(best, 8), best), -1)
            choice = beam.gather(1, selected_slot)
            chosen_state = prediction['state'].gather(
                1, (choice // 2)[..., None].expand(-1, -1, 12))
            direct_synthetic = (synthetic_distance.flatten(1).topk(4, largest=False).values.mean() / 1000
                + .15 * F.kl_div(prior[:1],
                    F.softmax(-synthetic_distance.flatten(1).detach() / 700, -1),
                    reduction='batchmean'))
            direct_real = .15 * (real_distance.flatten(1).topk(4, largest=False).values.mean() / 1000
                + .15 * F.kl_div(prior[1:],
                    F.softmax(-real_distance.flatten(1).detach() / 700, -1),
                    reduction='batchmean'))
        optimizer.zero_grad(set_to_none=True)
        first = feedback(prediction['feature'], chosen_state, choice % 2,
                         context['atlas'], offsets, weights)
        selected = {key: value for key, value in prediction.items()}
        selected['state'] = first['state']
        selected['log_mass'] = prediction['log_mass'].gather(1, choice // 2)
        selected['reflection_logit'] = prediction['reflection_logit'].gather(1, choice // 2)
        local_index = torch.arange(candidates, device='cuda')[None].expand(2, -1)
        with torch.no_grad():
            initial_map = model.map(selected, offsets, local_index, choice % 2,
                (64, 64), context['atlas'], weights, return_refinement_feature=True,
                feature_side=64, source_shape=(side, side),
                spatial_evidence=first['spatial_evidence'])
            local = initial_map['local_displacement_um'] / 1000
            magnitude = local.square().sum(2).sqrt().mean((-2, -1))
            rough_x = (local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
            rough_y = (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1))
            support = initial_map['atlas_pair'][:, :, 1].mean((-2, -1))
            reliability = initial_map['correspondence_logit'].sigmoid().mean((-3, -2, -1))
            fit_summary = torch.stack((magnitude, rough_x + rough_y,
                                       support, reliability), -1)
        second = feedback(prediction['feature'], first['state'], choice % 2,
                          context['atlas'], offsets, weights, fit_summary=fit_summary)
        synthetic_prediction = {key: value[:1] if isinstance(value, torch.Tensor) else value
                                for key, value in selected.items()}
        synthetic_prediction['state'] = second['state'][:1]
        final_map = model.map(synthetic_prediction, offsets[:1], local_index[:1],
            choice[:1] % 2, (96, 96), context['atlas'], weights[:1],
            feature_side=96, source_shape=(side, side),
            spatial_evidence=second['spatial_evidence'][:1])
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        grid = grid[:, None].expand(-1, candidates, -1, -1).reshape(candidates, 1, 96, 2)
        surface = final_map['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, padding_mode='border',
                               align_corners=False).reshape(candidates, 3, 96).permute(0, 2, 1)
        mapped_error = (fitted - target[:, None]).norm(dim=-1).mean(-1).reshape(1, candidates)
        score = (model.score_fitted_candidates(accepted['inputs'], synthetic_prediction,
                                               final_map, context['atlas'], weights[:1])
                 + second['quality_logit'][:1])
        selection = score.softmax(-1)
        fit_expected = (selection * mapped_error).sum(-1).mean() / 1000
        fit_rank = F.kl_div(F.log_softmax(score / 2, -1),
            F.softmax(-mapped_error.detach() / 500, -1), reduction='batchmean')
        corrected_synthetic = (points(second['state'][:1], choice[:1] % 2,
            chart[:, None]) - target[:, None]).norm(dim=-1).mean(-1)
        corrected_real = (points(second['state'][1:], choice[1:] % 2,
            corners) - weak_reference[:, None]).norm(dim=-1).mean(-1)
        warp_mm = (final_map['local_displacement_um'] / 1000).square().mean((2, 3, 4))
        regularizer = (selection * warp_mm).sum(-1).mean()
        loss = (direct_synthetic + direct_real + .4 * corrected_synthetic.min(-1).values.mean() / 1000
                + .15 * corrected_real.min(-1).values.mean() / 1000
                + fit_expected + .3 * fit_rank + .03 * regularizer)
        if step == warmup + 1:
            fit_gradient = torch.autograd.grad(fit_expected, model.anchor_pose[-1].weight,
                                               retain_graph=True)[0].norm()
            assert torch.isfinite(fit_gradient) and fit_gradient > 0
        progress = max(0., (step - warmup) / (batches - warmup))
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[0]['lr'] = 0 if step <= warmup else 1e-5 * decay
        optimizer.param_groups[1]['lr'] = 1e-4 * decay
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (p for p in list(model.parameters()) + list(feedback.parameters()) if p.requires_grad),
            5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {'batch': step, 'synthetic_presentations': step,
                'real_weak_presentations': step, 'joint_training': step > warmup,
                'loss': float(loss.detach()), 'direct_synthetic': float(direct_synthetic.detach()),
                'direct_real_weak': float(direct_real.detach()),
                'synthetic_selected_mapped_um': float(mapped_error.gather(
                    1, score.argmax(-1, keepdim=True)).detach().mean()),
                'synthetic_best3_mapped_um': float(mapped_error.detach().min()),
                'synthetic_best3_corrected_pose_um': float(corrected_synthetic.detach().min()),
                'real_best3_corrected_pose_um': float(corrected_real.detach().min()),
                'fit_expected': float(fit_expected.detach()), 'fit_rank': float(fit_rank.detach()),
                'warp_regularizer': float(regularizer.detach()),
                'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            if step == warmup + 1:
                row['fit_to_parent_anchor_pose_gradient_norm'] = float(fit_gradient.detach())
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 500 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in evaluation_steps:
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'synthetic_presentations': batches,
    'real_weak_presentations': batches,
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in evaluation_steps},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
