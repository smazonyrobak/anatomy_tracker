"""Matched on-policy image/atlas branch scoring within the native joint model."""
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

parent = root / 'runs/one_shot_candidate_energy_004/joint_step_08000.pt'
run = root / 'runs/one_shot_on_policy_atlas_rank_009'
seed, updates, synthetic, side = 2026100909, 8000, 2, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
schedule_paths = tuple(root / name for name in (
    'runs/one_shot_joint_mixed_real_003/real_schedule.npy',
    'runs/one_shot_fit_feedback_matched_001/new_real_schedule.npy',
    'runs/one_shot_joint_physical_pose_001/real_schedule.npy',
    'runs/one_shot_joint_physical_pose_002/real_schedule.npy',
    'runs/one_shot_pose_capture_003/real_schedule.npy',
    'runs/one_shot_candidate_energy_004/real_schedule.npy',
    'runs/one_shot_native_fit_feedback_005/real_schedule.npy',
    'runs/one_shot_vector_feedback_006/real_schedule.npy',
    'runs/one_shot_joint_feature_feedback_007/real_schedule.npy',
    'runs/one_shot_joint_native_fit_008/real_schedule.npy'))
used_real = {tuple(row) for path in schedule_paths for row in np.load(path)}
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < updates:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == updates:
                break
assert len(set(schedule)) == updates and not set(schedule) & used_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {
    'seed': seed, 'updates_per_arm': updates, 'synthetic_per_batch': synthetic,
    'real_per_batch': 1, 'side': side, 'branches': 32,
    'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
    'excluded_real_schedule_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in schedule_paths},
    'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
    'real_label_role': real['label_role'],
    'comparison': 'identical model, initialization, draws, losses and candidate states; atlas arm scorer has rendered atlas evidence; direct scorer receives zero atlas channels',
    'objective': 'all-32 on-policy listwise physical targets and detached score-weighted physical loss; predicted-branch mapped-tissue loss and gated native atlas fit',
    'fit_weight': 'direct-arm first gated pose-gradient ratio 0.2, upper bounded 0.02, no floor; same per-step coefficient in both arms',
    'uncertainty_calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ('train_one_shot_on_policy_atlas_rank_009.py',
                                   'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_one_shot_stream.py')},
}
(run / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


results = {}
shared_fit_weights = []
for arm in ('direct', 'atlas'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
    model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                   candidate_ranking=True).cuda().eval()
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    assert missing and all(key.startswith('candidate_matcher.') for key in missing) and not unexpected
    del checkpoint
    model.requires_grad_(False)
    modules = (model.encoder, model.pose, model.lateral, model.warp_shared,
               model.warp_condition, model.warp, model.atlas_encoder,
               model.pair, model.candidate_matcher)
    rates = (1e-5, 3e-5, 2e-5, 3e-5, 3e-5, 3e-5, 2e-5, 2e-5, 1e-4)
    for module in modules:
        module.requires_grad_(True)
    optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
                                   for module, rate in zip(modules, rates)], weight_decay=1e-4)
    subjects_rng = torch.Generator().manual_seed(seed)
    draw_seed = seed * 10000000
    fit_weight = None

    def save(step):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'step': step, 'arm': arm, 'fit_weight': fit_weight,
                    'config': config, 'calibrated': False},
                   directory / f'joint_step_{step:05d}.pt')

    save(0)
    started = time.perf_counter()
    with (directory / 'training.jsonl').open('w') as log, (directory / 'draws.jsonl').open('w') as draws:
        for step in range(1, updates + 1):
            accepted, pending = {}, list(range(synthetic))
            while pending:
                virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
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
                                    'slot': synthetic, 'used': True,
                                    'label_role': real['label_role']}) + '\n')
            image = torch.cat((batch['inputs'], observation['inputs']))
            truth = torch.cat((batch['state'], observation['state']))
            truth_reflection = torch.cat((batch['reflection'], observation['reflection']))
            real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
            real_weights = torch.ones_like(real_offsets)
            real_weights[:, [0, -1]] = .5
            real_weights /= real_weights.sum(-1, keepdim=True)
            offsets = torch.cat((batch['offsets'], real_offsets))
            weights = torch.cat((batch['weights'], real_weights))
            prediction = model.predict(image)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            branch_flags = flags[None, None].expand(len(image), model.modes, -1)
            reference = points(truth, truth_reflection, corners)
            five = (points(states, branch_flags, corners) - reference[:, None, None]).norm(dim=-1).mean(-1)
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            target = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, indices[..., None].expand(-1, -1, 3))
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            dense = (points(states[:synthetic], branch_flags[:synthetic], chart[:, None, None])
                     - target[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            true_normal = full_frame_state_to_components(truth)[1][:, :, 2]
            normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            distance = torch.cat((.75 * dense + .25 * five[:synthetic]
                                  + normal_penalty[:synthetic, :, None], five[synthetic:]), 0).flatten(1)
            physical = torch.cat((dense.flatten(1), five[synthetic:].flatten(1)))
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            flat_prior = prior.flatten(1)
            prior_rank = F.kl_div(flat_prior, (-distance.detach() / 1000).softmax(-1),
                                  reduction='none').sum(-1)
            direct_per_image = (distance.min(-1).values / 1000
                                - .75 * torch.logsumexp(flat_prior - distance / 1500, -1)
                                + .75 * prior_rank)
            scores = model.score_candidates(prediction, offsets, weights, context['atlas'],
                                             use_atlas=arm == 'atlas', side=64, chunk=4,
                                             image_shape=(side, side))
            rank = F.kl_div(F.log_softmax(scores, -1),
                            F.softmax(-physical.detach() / 1000, -1), reduction='none').sum(-1)
            score_weight = F.softmax(scores.detach(), -1)
            selected_pose = (physical * score_weight).sum(-1) / 1000
            per_image = direct_per_image + rank + .25 * selected_pose
            direct_loss = (per_image[:synthetic].sum()
                           + .5 * per_image[synthetic:].sum()) / (synthetic + .5)
            choice = torch.stack((scores[:synthetic].detach().argmax(-1),
                                  physical[:synthetic].detach().argmin(-1)), -1)
            synthetic_prediction = {key: value[:synthetic] if isinstance(value, torch.Tensor) else value
                                    for key, value in prediction.items()}
            mapped = model.map(synthetic_prediction, batch['offsets'], choice // 2, choice % 2,
                               (side, side), context['atlas'], batch['weights'])
            mapped_points = mapped['centre_surface_ccf_ap_dv_ml_um'].reshape(synthetic, 2, side * side, 3)
            mapped_points = mapped_points.gather(2, indices[:, None, :, None].expand(-1, 2, -1, 3))
            mapped_tissue = (mapped_points - target[:, None]).norm(dim=-1).mean(-1)
            mapping_loss = (.75 * mapped_tissue[:, 0] + .25 * mapped_tissue[:, 1]).mean() / 1000
            field = mapped['local_displacement_um']
            local_penalty = (.1 * field.square().mean().sqrt() / 1000
                             + .02 * ((field[..., 1:, :] - field[..., :-1, :]).abs().mean()
                                      + (field[..., 1:] - field[..., :-1]).abs().mean()) / 200)
            base_loss = direct_loss + mapping_loss + local_penalty
            rigid = physical[:synthetic].gather(1, choice)
            gate = (rigid.detach() < 2000).float()
            gate_count = int(gate.sum())
            fit_map = model.atlas_fit_loss(batch['inputs'], mapped, context['atlas'],
                                           batch['weights'], batch['valid_mask'])
            fit_term = (fit_map * gate).sum() / gate.sum().clamp_min(1)
            evidence = None
            if step == 1 or (arm == 'direct' and fit_weight is None and gate_count):
                evidence = {'step': step,
                    'score_to_global_pose_gradient': float(torch.autograd.grad(
                    rank.mean(), model.pose[-1].weight, retain_graph=True)[0].norm()),
                    'mapped_to_global_pose_gradient': float(torch.autograd.grad(
                    mapping_loss, model.pose[-1].weight, retain_graph=True)[0].norm())}
                base_gradient = torch.autograd.grad(base_loss, model.pose[-1].weight,
                                                    retain_graph=True)[0].norm()
                fit_gradient = torch.autograd.grad(fit_term, model.pose[-1].weight,
                                                   retain_graph=True)[0].norm()
                if arm == 'direct' and float(fit_gradient) > 0:
                    fit_weight = float((.2 * base_gradient / fit_gradient).clamp(max=.02))
                evidence.update({'fit_gate_branches': gate_count,
                                 'base_pose_gradient': float(base_gradient),
                                 'fit_pose_gradient': float(fit_gradient)})
            if arm == 'direct':
                shared_fit_weights.append(fit_weight)
            else:
                fit_weight = shared_fit_weights[step - 1]
            if evidence is not None:
                evidence['fit_weight'] = fit_weight
                (directory / 'fit_gradient_scaling.json').write_text(json.dumps(evidence, indent=2))
            loss = base_loss + (fit_weight or 0.) * fit_term
            decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / updates))
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate * decay
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                5., error_if_nonfinite=True)
            optimizer.step()
            selected = scores.detach().argmax(-1)
            row = {'arm': arm, 'step': step, 'presentations': step * len(image),
                   'synthetic_selected_tissue_um': float(dense.detach().flatten(1).gather(
                       1, selected[:synthetic, None]).mean()),
                   'synthetic_oracle_tissue_um': float(dense.detach().flatten(1).min(-1).values.mean()),
                   'selected_mapped_visible_tissue_um': float(mapped_tissue[:, 0].detach().mean()),
                   'oracle_branch_mapped_visible_tissue_um': float(mapped_tissue[:, 1].detach().mean()),
                   'real_weak_selected_five_um': float(five.detach().flatten(1).gather(
                       1, selected[synthetic:, None]).mean()),
                   'direct_loss': float(direct_loss.detach()), 'rank_loss': float(rank.detach().mean()),
                   'mapping_loss': float(mapping_loss.detach()),
                   'fit_term': float(fit_term.detach()),
                   'fit_gate_branches': gate_count, 'fit_weight': fit_weight,
                   'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 500 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
            if step % 2000 == 0:
                save(step)
    results[arm] = {'training_rows': updates,
                    'draws_sha256': hashlib.sha256((directory / 'draws.jsonl').read_bytes()).hexdigest(),
                    'fit_weight': fit_weight, 'seconds': time.perf_counter() - started}
assert results['direct']['draws_sha256'] == results['atlas']['draws_sha256']
(run / 'completed.json').write_text(json.dumps({
    'updates_per_arm': updates, 'accepted_synthetic_per_arm': updates * synthetic,
    'distinct_real_train_per_arm': updates,
    'matched_draws_sha256': results['direct']['draws_sha256'], 'arms': results,
    'fit_feedback_into_global_pose_and_warp': True, 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates_per_arm': updates,
                  'matched_draws_sha256': results['direct']['draws_sha256']}), flush=True)
