"""Long same-model continuation: full-pose mass targets and protected old pose path."""
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
run = root / 'runs/one_shot_anchor_quality_059'
seed, batches, synthetic, side, beam = 2026100359, 50000, 2, 256, 8
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
          'real_per_batch': 1, 'normal_anchors': 64, 'legacy_modes': 16,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'real_labels': real['label_role'], 'real_bindings': real['bindings'],
          'synthetic_provenance': context['provenance'],
          'real_schedule_unique_within_run': True,
          'earlier_training_reuse_allowed': True,
          'old_pose_encoder_mapper_fitter_frozen': True,
          'normal_anchor_mass_target': 'full 3D physical pose cost, not plane-normal angle only',
          'training_only_true_normal_neighbourhood': True,
          'fit_feedback': 'predicted plane -> finite-thickness atlas render -> fitted physical loss -> normal-anchor pose head',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_anchor_quality_059.py',
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
assert set(missing) == {'normal_anchor_frames',
                        'anchor_pose.0.weight', 'anchor_pose.0.bias',
                        'anchor_pose.2.weight', 'anchor_pose.2.bias',
                        'anchor_global.0.weight', 'anchor_global.0.bias',
                        'anchor_global.2.weight', 'anchor_global.2.bias'} and not unexpected
del checkpoint
model.requires_grad_(False)
model.anchor_pose.requires_grad_(True)
model.anchor_global.requires_grad_(True)
optimizer = torch.optim.AdamW(({'params': model.anchor_pose.parameters(), 'lr': 8e-5},
                               {'params': model.anchor_global.parameters(), 'lr': 8e-5}),
                              weight_decay=1e-4)
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
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = distance.min(-1).values.gather(1, near_mode)
        position_loss = (neighbourhood * near_distance).sum(-1) / 1000
        anchored = distance[:, model.base_modes:].min(-1).values
        anchor_target = F.softmax(-anchored.detach() / 700, -1)
        anchor_log_mass = prediction['log_mass'][:, model.base_modes:]
        anchor_log_mass = anchor_log_mass - anchor_log_mass.logsumexp(-1, keepdim=True)
        anchor_rank = -(anchor_target * anchor_log_mass).sum(-1)
        prior = prediction['log_mass']
        gate = prior[:, model.base_modes:].logsumexp(-1).exp()
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        branch_prior = (prior[..., None] + reflection_log).flatten(1)
        old_choices = branch_prior[:, :2 * model.base_modes].topk(2, -1).indices
        new_choices = (branch_prior[:, 2 * model.base_modes:].topk(4, -1).indices
                       + 2 * model.base_modes)
        old_best = distance.flatten(1).gather(1, old_choices).min(-1).values
        new_best = distance.flatten(1).gather(1, new_choices).min(-1).values
        gate_target = torch.sigmoid((old_best.detach() - new_best.detach()) / 500)
        gate_loss = F.binary_cross_entropy(gate, gate_target, reduction='none')
        near_reflection = distance.gather(
            1, near_mode[..., None].expand(-1, -1, 2)).argmin(-1)
        reflection_loss = (neighbourhood * F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'].gather(1, near_mode), near_reflection.float(),
            reduction='none')).sum(-1)
        direct = (position_loss[:synthetic].mean() + position_loss[synthetic:].mean()
                  + .25 * anchor_rank.mean() + gate_loss.mean() + .1 * reflection_loss.mean())
        old, new = old_choices[:synthetic], new_choices[:synthetic]
        guided_reflection = distance[:synthetic].gather(
            1, near_mode[:synthetic, :2, None].expand(-1, -1, 2)).argmin(-1)
        guided = 2 * near_mode[:synthetic, :2] + guided_reflection
        choice = torch.cat((old, new, guided), -1)
        source = {key: value[:synthetic] if isinstance(value, torch.Tensor) else value
                  for key, value in prediction.items()}
        selected_state = source['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        selected = {**source, 'state': selected_state,
                    'log_mass': source['log_mass'].gather(1, choice // 2),
                    'reflection_logit': source['reflection_logit'].gather(1, choice // 2)}
        index = torch.arange(beam, device='cuda')[None].expand(synthetic, -1)
        with torch.no_grad():
            first = model.map(selected, batch['offsets'], index, choice % 2,
                              (64, 64), context['atlas'], batch['weights'],
                              return_refinement_feature=True, feature_side=64,
                              source_shape=(side, side))
        refined_state, score_delta, _ = model.refine(first['refinement_feature'], selected_state)
        refined = {**selected, 'state': refined_state}
        mapped = model.map(refined, batch['offsets'], index, choice % 2,
                           (96, 96), context['atlas'], batch['weights'],
                           feature_side=96, source_shape=(side, side))
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        grid = grid[:, None].expand(-1, beam, -1, -1).reshape(synthetic * beam, 1, 96, 2)
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                               align_corners=False).reshape(synthetic, beam, 3, 96).permute(0, 1, 3, 2)
        mapped_error = (fitted - target[:, None]).norm(dim=-1).mean(-1)
        score = model.score_fitted_candidates(batch['inputs'], refined, mapped,
                                              context['atlas'], batch['weights']) + score_delta
        predicted_anchor_error = mapped_error[:, 2:6]
        predicted_anchor_score = score[:, 2:6]
        fit_expected = (.5 * (F.softmax(score[:, :6], -1)
                              * mapped_error[:, :6]).sum(-1)
                        + .5 * (F.softmax(predicted_anchor_score, -1)
                                * predicted_anchor_error).sum(-1)).mean() / 1000
        fit_rank = F.kl_div(F.log_softmax(score[:, 2:8] - branch_prior[:synthetic].gather(
            1, choice)[:, 2:8], -1),
            F.softmax(-mapped_error[:, 2:8].detach() / 700, -1),
            reduction='none').sum(-1).mean()
        guided_fit = mapped_error[:, 6:8].mean() / 1000
        loss = direct + .5 * fit_expected + .5 * fit_rank + .25 * guided_fit
        if step == 1:
            gradient = torch.autograd.grad(fit_expected + fit_rank,
                                           model.anchor_pose[-1].weight,
                                           retain_graph=True)[0].norm()
            assert torch.isfinite(gradient) and gradient > 0
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / batches))
        for group in optimizer.param_groups:
            group['lr'] = 8e-5 * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 10., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'synthetic_presentations': step * synthetic,
               'real_train_presentations': step,
               'synthetic_near_anchor_um': float(near_distance[:synthetic, 0].detach().mean()),
               'real_near_anchor_um': float(near_distance[synthetic:, 0].detach().mean()),
               'synthetic_gate_target': float(gate_target[:synthetic].mean()),
               'real_gate_target': float(gate_target[synthetic:].mean()),
               'synthetic_gate_predicted': float(gate[:synthetic].detach().mean()),
               'real_gate_predicted': float(gate[synthetic:].detach().mean()),
               'synthetic_predicted_anchor_mapped_um': float(predicted_anchor_error.detach().mean()),
               'synthetic_guided_anchor_mapped_um': float(mapped_error[:, 6:8].detach().mean()),
               'direct': float(direct.detach()), 'fit_expected': float(fit_expected.detach()),
               'fit_rank': float(fit_rank.detach()), 'guided_fit': float(guided_fit.detach()),
               'gradient_norm': float(clipped), 'seconds': time.perf_counter() - started}
        if step == 1:
            row['fit_to_anchor_pose_gradient_norm'] = float(gradient)
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 10000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches * synthetic,
    'unique_real_train_within_run': batches,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
