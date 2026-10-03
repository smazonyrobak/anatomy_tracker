"""Continue the standalone joint model with projective-normal pose anchors and atlas-fit feedback."""
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

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
run = root / 'runs/one_shot_anchor_joint_058'
seed, batches, synthetic, side, beam = 2026100358, 12000, 2, 256, 8
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
used = {tuple(map(int, row)) for path in root.glob('runs/*/real_schedule.npy') for row in np.load(path)}
rng = np.random.default_rng(seed)
fresh, previous = [], []
for donor, records in enumerate(real['donors']):
    sections = rng.permutation(len(records['identities'])).tolist()
    fresh.append([section for section in sections if (donor, section) not in used])
    previous.append([section for section in sections if (donor, section) in used])
schedule = []
for pool in (fresh, previous):
    while len(schedule) < batches:
        progressed = False
        for donor in rng.permutation(len(pool)):
            if pool[donor]:
                schedule.append((int(donor), int(pool[donor].pop())))
                progressed = True
            if len(schedule) == batches:
                break
        if not progressed:
            break
assert len(set(schedule)) == batches
never_before_used = len(set(schedule) - used)

run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, np.int32))
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
          'real_per_batch': 1, 'normal_anchors': 64, 'legacy_modes': 16, 'beam': beam,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'unique_real_sections_within_run': batches,
          'never_before_used_real_sections': never_before_used,
          'previously_used_in_earlier_runs': batches - never_before_used,
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'training_only_true_normal_neighbourhood': True,
          'training_only_exact_pose_mapper': True,
          'fit_feedback': 'predicted pose -> finite-thickness atlas render -> fitted tissue map and score -> predicted pose',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_anchor_joint_058.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_full_frame_primitives.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
                               fit_quality=True, vector_refinement=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().train()
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
assert set(missing) == {'normal_anchor_frames', 'anchor_pose.0.weight', 'anchor_pose.0.bias',
                        'anchor_pose.2.weight', 'anchor_pose.2.bias'} and not unexpected
del checkpoint
model.requires_grad_(False)
modules = (model.encoder, model.pose, model.lateral, model.warp_shared,
           model.warp_condition, model.warp, model.atlas_encoder, model.pair,
           model.fitted_matcher, model.pose_refiner, model.anchor_pose)
rates = (3e-6, 3e-6, 3e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 2e-5, 2e-5, 8e-5)
for module in modules:
    module.requires_grad_(True)
optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
                               for module, rate in zip(modules, rates)], weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False}, run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                eligible = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': eligible}) + '\n')
                if eligible:
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
        branch_flags = flags[None, None].expand(len(image), model.modes, 2)
        reference = points(truth, truth_reflection, corners)
        five = (points(states, branch_flags, corners) - reference[:, None, None]).norm(dim=-1).mean(-1)
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 96, replacement=True)
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
                              + normal_penalty[:synthetic, :, None],
                              five[synthetic:] + .25 * normal_penalty[synthetic:, :, None]), 0)
        similarity = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs()
        nearest = similarity.topk(4, -1)
        weight = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        mode = model.base_modes + nearest.indices
        anchor_distance = distance.min(-1).values.gather(1, mode)
        per_image = (weight * anchor_distance).sum(-1) / 1000
        mass = -(weight * prediction['log_mass'].gather(1, mode)).sum(-1)
        reflected_target = distance.gather(1, mode[..., None].expand(-1, -1, 2)).argmin(-1)
        reflection_loss = (weight * F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'].gather(1, mode), reflected_target.float(),
            reduction='none')).sum(-1)
        direct = ((per_image[:synthetic] + .25 * mass[:synthetic]
                   + .1 * reflection_loss[:synthetic]).mean()
                  + .25 * (per_image[synthetic:] + .25 * mass[synthetic:]
                           + .1 * reflection_loss[synthetic:]).mean()
                  + .2 * distance[:synthetic].flatten(1).min(-1).values.mean() / 1000)
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
        old = prior[:synthetic, :2 * model.base_modes].topk(3, -1).indices
        new = prior[:synthetic, 2 * model.base_modes:].topk(3, -1).indices + 2 * model.base_modes
        target_reflection = distance[:synthetic].gather(
            1, mode[:synthetic, :2, None].expand(-1, -1, 2)).argmin(-1)
        guided = 2 * mode[:synthetic, :2] + target_reflection
        choice = torch.cat((old, new, guided), -1)
        source = {key: value[:synthetic] if isinstance(value, torch.Tensor) else value
                  for key, value in prediction.items()}
        selected_state = source['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        initial_state = torch.cat((selected_state, batch['state'][:, None]), 1)
        reflected = torch.cat((choice % 2, batch['reflection'][:, None]), 1)
        chosen_log_mass = source['log_mass'].gather(1, choice // 2)
        chosen_reflection_logit = source['reflection_logit'].gather(1, choice // 2)
        selected = {**source, 'state': initial_state,
                    'log_mass': torch.cat((chosen_log_mass, chosen_log_mass.new_full(
                        (synthetic, 1), -math.log(model.modes))), 1),
                    'reflection_logit': torch.cat((chosen_reflection_logit,
                                                   chosen_reflection_logit.new_zeros(synthetic, 1)), 1)}
        index = torch.arange(beam + 1, device='cuda')[None].expand(synthetic, -1)
        with torch.no_grad():
            first = model.map(selected, batch['offsets'], index, reflected,
                              (64, 64), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=64,
                              source_shape=(side, side))
        refined_state, score_delta, update = model.refine(first['refinement_feature'], initial_state)
        refined = {**selected, 'state': refined_state}
        mapped = model.map(refined, batch['offsets'], index, reflected,
                           (96, 96), context['atlas'], batch['weights'],
                           feature_side=96, source_shape=(side, side))
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        grid = grid[:, None].expand(-1, beam + 1, -1, -1).reshape(synthetic * (beam + 1), 1, 96, 2)
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                               align_corners=False).reshape(synthetic, beam + 1, 3, 96).permute(0, 1, 3, 2)
        mapped_error = (fitted - target[:, None]).norm(dim=-1).mean(-1)
        scores = model.score_fitted_candidates(batch['inputs'], refined, mapped,
                                               context['atlas'], batch['weights']) + score_delta
        branch_prior = prior[:synthetic].gather(1, choice)
        rank = F.kl_div(F.log_softmax(scores[:, :beam] - branch_prior, -1),
                        F.softmax(-mapped_error[:, :beam].detach() / 600, -1),
                        reduction='none').sum(-1).mean()
        expected = (F.softmax(scores[:, :beam], -1) * mapped_error[:, :beam]).sum(-1).mean() / 1000
        teacher = mapped_error[:, beam].mean() / 1000
        normal_refined = full_frame_state_to_components(refined_state[:, :beam])[1][..., :, 2]
        rigid = (points(refined_state[:, :beam], reflected[:, :beam], chart[:, None])
                 - target[:, None]).norm(dim=-1).mean(-1)
        rigid = rigid + 4000 * (1 - (normal_refined * true_normal[:synthetic, None]).sum(-1).abs())
        refine_loss = rigid.min(-1).values.mean() / 1000
        local = mapped['local_displacement_um']
        local_penalty = (.1 * local.square().mean().sqrt() / 1000
                         + .02 * ((local[..., 1:, :] - local[..., :-1, :]).abs().mean()
                                  + (local[..., 1:] - local[..., :-1]).abs().mean()) / 200)
        loss = direct + expected + .5 * rank + .5 * teacher + .5 * refine_loss + local_penalty
        if step == 1:
            feedback = torch.autograd.grad(expected + rank, model.anchor_pose[-1].weight,
                                           retain_graph=True)[0].norm()
            assert torch.isfinite(feedback) and feedback > 0
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / batches))
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 5., error_if_nonfinite=True)
        optimizer.step()
        selected_branch = scores[:, :beam].detach().argmax(-1)
        row = {'batch': step, 'synthetic_presentations': step * synthetic,
               'real_train_presentations': step,
               'anchor_neighbour_tissue_um': float(anchor_distance[:synthetic, 0].detach().mean()),
               'prior_selected_rigid_um': float(distance[:synthetic].flatten(1).gather(
                   1, prior[:synthetic].detach().argmax(-1)[:, None]).mean()),
               'train_selected_mapped_um': float(mapped_error[:, :beam].detach().gather(
                   1, selected_branch[:, None]).mean()),
               'train_best8_mapped_um': float(mapped_error[:, :beam].detach().min(-1).values.mean()),
               'train_true_pose_mapped_um': float(mapped_error[:, beam].detach().mean()),
               'direct': float(direct.detach()), 'fit_expected': float(expected.detach()),
               'fit_rank': float(rank.detach()), 'fit_teacher': float(teacher.detach()),
               'refine': float(refine_loss.detach()), 'local_penalty': float(local_penalty.detach()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        if step == 1:
            row['fit_to_anchor_pose_gradient_norm'] = float(feedback)
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 2000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches * synthetic,
    'unique_real_train_within_run': batches,
    'never_before_used_real_train': never_before_used,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
