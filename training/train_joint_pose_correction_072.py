"""Continue one joint model: corrected predicted poses, then gated fit-to-pose feedback."""
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

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/joint_pose_correction_072'
seed, batches, synthetic, side, beam, warmup = 2026100372, 10000, 2, 256, 8, 3000
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
    xy[..., 0] = torch.where(reflection[..., None].bool(), (side - 1) / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
used_real = {tuple(item) for item in np.load(root / 'runs/one_shot_anchor_quality_059/real_schedule.npy')}
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < batches:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == batches:
                break
assert len(set(schedule)) == batches and not set(schedule) & used_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
    'real_per_batch': 1, 'side': side, 'candidate_quota': {'old': 2, 'anchor': 6},
    'warmup_batches': warmup, 'parent': str(parent), 'parent_sha256': sha(parent),
    'real_schedule_sha256': sha(run / 'real_schedule.npy'),
    'excluded_059_real_schedule_sha256': sha(root / 'runs/one_shot_anchor_quality_059/real_schedule.npy'),
    'real_labels': real['label_role'], 'real_bindings': real['bindings'],
    'synthetic_provenance': context['provenance'],
    'training': 'direct pose + predicted-candidate correction/ranking; after warmup, near-correct fitted tissue and image agreement backpropagate to pose heads',
    'atlas_fit_weight_after_warmup': .02, 'near_correct_gate_um': 1000,
    'calibrated': False, 'biological_validation': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
        ('train_joint_pose_correction_072.py', 'arbitrary_plane_one_shot_model.py',
         'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
         'arbitrary_plane_full_frame_primitives.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
frozen = torch.load(parent, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(frozen['model'])
del frozen
new_refiner = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).pose_refiner
model.pose_refiner.load_state_dict(new_refiner.state_dict())
del new_refiner
model.requires_grad_(False)
modules = (model.encoder, model.pose, model.anchor_pose, model.anchor_global,
           model.lateral, model.warp_shared, model.atlas_encoder, model.pair,
           model.pose_refiner, model.warp_condition, model.warp)
rates = (8e-6, 1e-5, 3e-5, 3e-5, 1e-5, 1e-5, 2e-5, 2e-5, 1e-4, 1e-5, 1e-5)
for module in modules:
    module.requires_grad_(True)
model.warp_condition.requires_grad_(False)
model.warp.requires_grad_(False)
optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
    for module, rate in zip(modules, rates)], weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')
fit_to_pose_gradient = None


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        if step == warmup + 1:
            model.warp_condition.requires_grad_(True)
            model.warp.requires_grad_(True)
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'offsets', 'weights',
                         'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        observation['inputs'] = F.interpolate(observation['inputs'], (side, side),
                                              mode='bilinear', align_corners=False)
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
            'slot': synthetic, 'used': True, 'label_role': real['label_role']}) + '\n')
        image = torch.cat((batch['inputs'], observation['inputs']))
        truth = torch.cat((batch['state'], observation['state']))
        true_reflection = torch.cat((batch['reflection'], observation['reflection']))
        real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
        real_weights = torch.ones_like(real_offsets)
        real_weights[:, [0, -1]] = .5
        real_weights /= real_weights.sum(-1, keepdim=True)
        offsets = torch.cat((batch['offsets'], real_offsets))
        weights = torch.cat((batch['weights'], real_weights))
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 96, replacement=True)
        target = batch['centre'].reshape(synthetic, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() / side
        prediction = model.predict(image)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        branch_flags = flags[None, None].expand(len(image), model.modes, 2)
        reference = points(truth, true_reflection, corners)
        five = (points(states, branch_flags, corners) - reference[:, None, None]).norm(dim=-1).mean(-1)
        dense = (points(states[:synthetic], branch_flags[:synthetic], chart[:, None, None])
                 - target[:, None, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][:, :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        distance = torch.cat((.75 * dense + .25 * five[:synthetic]
                              + normal_penalty[:synthetic, :, None],
                              five[synthetic:] + .25 * normal_penalty[synthetic:, :, None]), 0)
        similarity = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs()
        nearest = similarity.topk(4, -1)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = distance.min(-1).values.gather(1, near_mode)
        position = (neighbourhood * near_distance).sum(-1) / 1000
        anchored = distance[:, model.base_modes:].min(-1).values
        anchor_target = F.softmax(-anchored.detach() / 700, -1)
        anchor_mass = prediction['log_mass'][:, model.base_modes:]
        anchor_mass = anchor_mass - anchor_mass.logsumexp(-1, keepdim=True)
        anchor_rank = -(anchor_target * anchor_mass).sum(-1)
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
        old = prior[:, :2 * model.base_modes].topk(2, -1).indices
        anchor = prior[:, 2 * model.base_modes:].topk(6, -1).indices + 2 * model.base_modes
        choice = torch.cat((old, anchor), -1)
        old_best = distance.flatten(1).gather(1, old).min(-1).values
        new_best = distance.flatten(1).gather(1, anchor).min(-1).values
        gate = prediction['log_mass'][:, model.base_modes:].logsumexp(-1).exp()
        gate_target = torch.sigmoid((old_best.detach() - new_best.detach()) / 500)
        direct = (position[:synthetic].mean() + position[synthetic:].mean()
                  + .25 * anchor_rank.mean()
                  + F.binary_cross_entropy(gate, gate_target))
        selected_state = prediction['state'].gather(
            1, (choice // 2)[..., None].expand(-1, -1, 12))
        selected = {**prediction, 'state': selected_state,
            'log_mass': prediction['log_mass'].gather(1, choice // 2),
            'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
        mode_index = torch.arange(beam, device='cuda')[None].expand(len(image), -1)
        first = model.map(selected, offsets, mode_index, choice % 2,
            (64, 64), context['atlas'], weights, return_refinement_feature=True,
            feature_side=64, source_shape=(side, side))
        refined_state, delta, update = model.refine(first['refinement_feature'], selected_state)
        before = torch.cat((dense.flatten(1), five[synthetic:].flatten(1)), 0).gather(1, choice)
        after_synthetic = (points(refined_state[:synthetic], choice[:synthetic] % 2,
                                  chart[:, None]) - target[:, None]).norm(dim=-1).mean(-1)
        after_real = (points(refined_state[synthetic:], choice[synthetic:] % 2, corners)
                      - reference[synthetic:, None]).norm(dim=-1).mean(-1)
        after = torch.cat((after_synthetic, after_real), 0)
        score = prior.gather(1, choice) + delta
        best_weight = F.softmax(-before.detach() / 700, -1)
        correction = (best_weight * after).sum(-1).mean() / 1000
        rank = F.kl_div(F.log_softmax(score, -1), F.softmax(-after.detach() / 700, -1),
                        reduction='none').sum(-1).mean()
        selected_error = (F.softmax(score, -1) * after).sum(-1).mean() / 1000
        loss = direct + correction + rank + .25 * selected_error
        mapped_error = fit_term = warp_penalty = None
        if step > warmup:
            best = after_synthetic.detach().argmin(-1)
            best_state = refined_state[:synthetic].gather(
                1, best[:, None, None].expand(-1, 1, 12))
            best_flag = (choice[:synthetic] % 2).gather(1, best[:, None])
            synthetic_prediction = {key: value[:synthetic] if isinstance(value, torch.Tensor)
                else value for key, value in prediction.items()}
            synthetic_prediction['state'] = best_state
            mapped = model.map(synthetic_prediction, batch['offsets'],
                torch.zeros((synthetic, 1), device='cuda', dtype=torch.long), best_flag,
                (96, 96), context['atlas'], batch['weights'],
                feature_side=96, source_shape=(side, side))
            grid = (torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
            fitted = F.grid_sample(
                mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0].permute(0, 3, 1, 2),
                grid[:, None], mode='bilinear', padding_mode='border', align_corners=False)
            fitted = fitted[:, :, 0].transpose(1, 2)
            mapped_error = (fitted - target).norm(dim=-1).mean(-1)
            local = mapped['local_displacement_um'][:, 0] / 1000
            warp_penalty = (local.square().mean()
                + .1 * (local[..., 1:] - local[..., :-1]).square().mean()
                + .1 * (local[..., 1:, :] - local[..., :-1, :]).square().mean())
            fit = model.atlas_fit_loss(batch['inputs'], mapped, context['atlas'],
                                       batch['weights'], batch['valid_mask'])[:, 0]
            reliable = (after_synthetic.detach().gather(1, best[:, None])[:, 0] < 1000).float()
            fit_term = (fit * reliable).sum() / reliable.sum().clamp_min(1)
            loss = loss + .5 * mapped_error.mean() / 1000 + .02 * fit_term + .01 * warp_penalty
            if fit_to_pose_gradient is None and bool(reliable.sum()):
                grads = torch.autograd.grad(fit_term,
                    (model.pose[-1].weight, model.anchor_pose[-1].weight),
                    retain_graph=True, allow_unused=True)
                fit_to_pose_gradient = float(sum(g.norm() for g in grads if g is not None))
                assert fit_to_pose_gradient > 0
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / batches))
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (parameter for group in optimizer.param_groups for parameter in group['params']),
            5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'batch': step, 'loss': float(loss.detach()), 'direct': float(direct.detach()),
                'correction': float(correction.detach()), 'rank': float(rank.detach()),
                'synthetic_best_corrected_um': float(after_synthetic.detach().min(-1).values.mean()),
                'synthetic_selected_corrected_um': float(after_synthetic.detach().gather(
                    1, score[:synthetic].detach().argmax(-1)[:, None]).mean()),
                'real_selected_weak_five_um': float(after_real.detach().gather(
                    1, score[synthetic:].detach().argmax(-1)[:, None]).mean()),
                'mapped_um': None if mapped_error is None else float(mapped_error.detach().mean()),
                'fit_loss': None if fit_term is None else float(fit_term.detach()),
                'fit_to_pose_gradient': fit_to_pose_gradient,
                'gradient_norm': float(gradient), 'elapsed_s': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            log.flush()
            draws.flush()
            if step == 1 or step % 1000 == 0:
                print(json.dumps(row), flush=True)
        if step in (warmup, 6000, batches):
            save(step)
print(json.dumps({'finished_batches': batches,
    'fit_to_pose_gradient': fit_to_pose_gradient}, flush=True))
