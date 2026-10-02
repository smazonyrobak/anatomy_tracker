"""Matched joint pose-and-mapper continuation with optional native atlas fit."""
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
run = root / 'runs/one_shot_joint_native_fit_008'
seed, updates, synthetic, side = 2026100808, 10000, 2, 256
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
    'runs/one_shot_joint_feature_feedback_007/real_schedule.npy'))
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
source_names = ('train_one_shot_joint_native_fit_008.py', 'arbitrary_plane_one_shot_model.py',
                'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
                'arbitrary_plane_reserved_real_stream_v8.py', 'arbitrary_plane_full_frame_primitives.py')
config = {
    'seed': seed, 'updates_per_arm': updates, 'synthetic_per_batch': synthetic,
    'real_per_batch': 1, 'side': side, 'modes': 16, 'reflection_branches': 2,
    'mapped_candidates': 'image-prior best and true-geometry best of 32 actual predicted branches; no supplied truth pose',
    'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
    'excluded_real_schedule_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in schedule_paths},
    'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
    'real_label_role': real['label_role'],
    'trainable_both_arms': 'encoder, pose, lateral, warp_shared, warp_condition, warp, atlas_encoder, pair',
    'frozen_both_arms': 'uncertainty and fit_quality_head; no pose_refiner',
    'shared_loss': 'all-32 synthetic visible-tissue/full-frame/normal and weak-real pose; predicted-branch mapped visible-tissue CCF and local-field regularization',
    'fit_only_loss': 'native differentiable atlas_fit_loss on synthetic valid tissue, rigid-geometry <2 mm detached gate; first-fit-batch pose-gradient scale',
    'fit_feedback_into_global_pose_and_warp': True, 'calibrated': False,
    'public_benchmark_used': False,
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in source_names},
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
for arm in ('control', 'fit'):
    directory = run / arm
    directory.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
    model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.requires_grad_(False)
    for module in (model.encoder, model.pose, model.lateral, model.warp_shared,
                   model.warp_condition, model.warp, model.atlas_encoder, model.pair):
        module.requires_grad_(True)
    rates = (1e-5, 3e-5, 2e-5, 3e-5, 3e-5, 3e-5, 2e-5, 2e-5)
    modules = (model.encoder, model.pose, model.lateral, model.warp_shared,
               model.warp_condition, model.warp, model.atlas_encoder, model.pair)
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
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            reference_normals = full_frame_state_to_components(truth)[1][:, :, 2]
            normal = 1 - (normals * reference_normals[:, None]).sum(-1).abs().clamp_max(1)
            distance = torch.cat((.75 * dense + .25 * five[:synthetic] + 4000 * normal[:synthetic, :, None],
                                  five[synthetic:]), 0).flatten(1)
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            flat_prior = prior.flatten(1)
            rank = F.kl_div(flat_prior, (-distance.detach() / 1000).softmax(-1),
                            reduction='none').sum(-1)
            direct_per_image = (distance.min(-1).values / 1000
                                - .75 * torch.logsumexp(flat_prior - distance / 1500, -1)
                                + .75 * rank)
            direct_loss = (direct_per_image[:synthetic].sum()
                           + .5 * direct_per_image[synthetic:].sum()) / (synthetic + .5)
            choice = torch.stack((flat_prior[:synthetic].detach().argmax(-1),
                                  distance[:synthetic].detach().argmin(-1)), -1)
            synthetic_prediction = {key: value[:synthetic] if isinstance(value, torch.Tensor) else value
                                    for key, value in prediction.items()}
            mapped = model.map(synthetic_prediction, batch['offsets'], choice // 2, choice % 2,
                               (side, side), context['atlas'], batch['weights'])
            mapped_points = mapped['centre_surface_ccf_ap_dv_ml_um'].reshape(synthetic, 2, side * side, 3)
            mapped_points = mapped_points.gather(2, indices[:, None, :, None].expand(-1, 2, -1, 3))
            mapped_tissue = (mapped_points - target[:, None]).norm(dim=-1).mean(-1)
            rigid_tissue = dense.flatten(1).gather(1, choice)
            branch_weight = F.softmax(-rigid_tissue.detach() / 1000, -1)
            mapping_loss = (mapped_tissue * branch_weight).sum(-1).mean() / 1000
            field = mapped['local_displacement_um']
            local_penalty = (.1 * field.square().mean().sqrt() / 1000
                             + .02 * ((field[..., 1:, :] - field[..., :-1, :]).abs().mean()
                                      + (field[..., 1:] - field[..., :-1]).abs().mean()) / 200)
            base_loss = direct_loss + mapping_loss + local_penalty
            fit_term = None
            gate_count = 0
            if arm == 'fit':
                fit_map = model.atlas_fit_loss(batch['inputs'], mapped, context['atlas'],
                                               batch['weights'], batch['valid_mask'])
                gate = (rigid_tissue.detach() < 2000).float()
                gate_count = int(gate.sum())
                fit_term = (fit_map * gate * branch_weight).sum() / (gate * branch_weight).sum().clamp_min(1)
            if step == 1:
                mapped_pose_gradient = torch.autograd.grad(mapping_loss, model.pose[-1].weight,
                                                           retain_graph=True)[0].norm()
                mapped_warp_gradient = torch.autograd.grad(mapping_loss, model.warp[-1].weight,
                                                           retain_graph=True)[0].norm()
                evidence = {'mapped_to_global_pose_gradient': float(mapped_pose_gradient),
                            'mapped_to_local_warp_gradient': float(mapped_warp_gradient)}
                if arm == 'fit':
                    base_pose_gradient = torch.autograd.grad(base_loss, model.pose[-1].weight,
                                                             retain_graph=True)[0].norm()
                    fit_pose_gradient = torch.autograd.grad(fit_term, model.pose[-1].weight,
                                                            retain_graph=True)[0].norm()
                    fit_warp_gradient = torch.autograd.grad(fit_term, model.warp[-1].weight,
                                                            retain_graph=True)[0].norm()
                    fit_weight = (float((.2 * base_pose_gradient / fit_pose_gradient).clamp(.01, .5))
                                  if float(fit_pose_gradient) > 0 else .1)
                    evidence.update({'fit_gate_branches': gate_count,
                                     'base_to_global_pose_gradient': float(base_pose_gradient),
                                     'fit_to_global_pose_gradient': float(fit_pose_gradient),
                                     'fit_to_local_warp_gradient': float(fit_warp_gradient),
                                     'fit_weight': fit_weight})
                (directory / 'first_batch_gradients.json').write_text(json.dumps(evidence, indent=2))
            loss = base_loss if fit_term is None else base_loss + fit_weight * fit_term
            decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / updates))
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate * decay
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                5., error_if_nonfinite=True)
            optimizer.step()
            selected = flat_prior.detach().argmax(-1)
            row = {'arm': arm, 'step': step, 'presentations': step * len(image),
                   'synthetic_selected_tissue_um': float(dense.detach().flatten(1).gather(
                       1, selected[:synthetic, None]).mean()),
                   'synthetic_oracle_tissue_um': float(dense.detach().flatten(1).min(-1).values.mean()),
                   'mapped_visible_tissue_um': float((mapping_loss.detach() * 1000)),
                   'real_weak_selected_five_um': float(five.detach().flatten(1).gather(
                       1, selected[synthetic:, None]).mean()),
                   'direct_loss': float(direct_loss.detach()),
                   'mapping_loss': float(mapping_loss.detach()),
                   'fit_term': None if fit_term is None else float(fit_term.detach()),
                   'fit_gate_branches': gate_count, 'fit_weight': fit_weight,
                   'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 1000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
            if step % 2000 == 0:
                save(step)
    results[arm] = {'training_rows': updates,
                    'draws_sha256': hashlib.sha256((directory / 'draws.jsonl').read_bytes()).hexdigest(),
                    'fit_weight': fit_weight, 'seconds': time.perf_counter() - started}
assert results['control']['draws_sha256'] == results['fit']['draws_sha256']
(run / 'completed.json').write_text(json.dumps({
    'updates_per_arm': updates, 'accepted_synthetic_per_arm': updates * synthetic,
    'distinct_real_train_per_arm': updates,
    'matched_draws_sha256': results['control']['draws_sha256'], 'arms': results,
    'fit_feedback_into_global_pose_and_warp': True, 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates_per_arm': updates,
                  'matched_draws_sha256': results['control']['draws_sha256']}), flush=True)
