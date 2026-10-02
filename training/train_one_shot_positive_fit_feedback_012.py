"""Joint pose/mapper training with an exact-plane positive in each synthetic fit set."""
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

parent = root / 'runs/one_shot_true_pose_mapper_011/joint_step_04000.pt'
run = root / 'runs/one_shot_positive_fit_feedback_012'
seed, updates, synthetic, side, beam, fit_side = 2026101200, 8000, 2, 256, 8, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
prior_schedules = sorted(root.glob('runs/*/real_schedule.npy'))
used_real = {tuple(map(int, row)) for path in prior_schedules for row in np.load(path)}
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
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic,
          'real_per_batch': 1, 'side': side, 'beam': beam, 'fit_side': fit_side,
          'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'excluded_real_schedule_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in prior_schedules},
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'fit_feedback': 'joint model maps its top-eight predicted poses and one exact synthetic pose; learned fit matching sees the exact pose as a positive while mapped physical loss differentiates through predicted pose; exact pose is absent from inference',
          'gradient_probe': 'step-one fitted matcher evidence, with direct mode/reflection prior removed, must differentiate to pose-head weights',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_positive_fit_feedback_012.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_reserved_real_stream_v8.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 4000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
modules = (model.encoder, model.pose, model.lateral, model.warp_shared,
           model.warp_condition, model.warp, model.atlas_encoder, model.pair,
           model.fitted_matcher)
rates = (1e-5, 2e-5, 1e-5, 2e-5, 2e-5, 2e-5, 1e-5, 2e-5, 1e-4)
for module in modules:
    module.requires_grad_(True)
optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
                               for module, rate in zip(modules, rates)], weight_decay=1e-4)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
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
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][:, :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        distance = torch.cat((.75 * dense + .25 * five[:synthetic]
                              + normal_penalty[:synthetic, :, None], five[synthetic:]), 0).flatten(1)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        prior_rank = F.kl_div(prior, (-distance.detach() / 1000).softmax(-1),
                              reduction='none').sum(-1)
        direct_per_image = (distance.min(-1).values / 1000
                            - .75 * torch.logsumexp(prior - distance / 1500, -1)
                            + .75 * prior_rank)
        direct_loss = (direct_per_image[:synthetic].sum()
                       + .5 * direct_per_image[synthetic:].sum()) / (synthetic + .5)
        choice = prior[:synthetic].detach().topk(beam, -1).indices
        selected_prediction = {key: value[:synthetic] if isinstance(value, torch.Tensor) else value
                               for key, value in prediction.items()}
        extended = {**selected_prediction,
                    'state': torch.cat((selected_prediction['state'], batch['state'][:, None]), 1),
                    'log_mass': torch.cat((selected_prediction['log_mass'],
                                           selected_prediction['log_mass'].new_full(
                                               (synthetic, 1), -math.log(model.modes))), 1),
                    'reflection_logit': torch.cat((selected_prediction['reflection_logit'],
                                                   selected_prediction['reflection_logit'].new_zeros(
                                                       synthetic, 1)), 1)}
        modes = torch.cat((choice // 2, torch.full((synthetic, 1), model.modes,
                                                   device='cuda', dtype=torch.long)), 1)
        reflected = torch.cat((choice % 2, batch['reflection'][:, None]), 1)
        mapped = model.map(extended, batch['offsets'], modes, reflected, (fit_side, fit_side),
                           context['atlas'], batch['weights'], feature_side=fit_side,
                           source_shape=(side, side))
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        grid = grid[:, None].expand(-1, beam + 1, -1, -1).reshape(synthetic * (beam + 1), 1, 128, 2)
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                               align_corners=False).reshape(synthetic, beam + 1, 3, 128).permute(0, 1, 3, 2)
        mapped_tissue = (fitted - target[:, None]).norm(dim=-1).mean(-1)
        scores = model.score_fitted_candidates(batch['inputs'], extended, mapped,
                                               context['atlas'], batch['weights'])
        row = torch.arange(synthetic, device='cuda')[:, None]
        branch_prior = (extended['log_mass'][row, modes] + torch.where(
            reflected.bool(), F.logsigmoid(extended['reflection_logit'][row, modes]),
            F.logsigmoid(-extended['reflection_logit'][row, modes])))
        fit_rank = F.kl_div(F.log_softmax(scores - branch_prior, -1),
                            F.softmax(-mapped_tissue.detach() / 500, -1),
                            reduction='none').sum(-1).mean()
        expected = (F.softmax(scores[:, :beam], -1) * mapped_tissue[:, :beam]).sum(-1).mean() / 1000
        mapping_loss = expected + .25 * mapped_tissue[:, :beam].min(-1).values.mean() / 1000
        teacher_loss = mapped_tissue[:, beam].mean() / 1000
        valid = F.interpolate(batch['valid_mask'][:, None].float(),
                              (fit_side, fit_side), mode='area')[:, None]
        reliability_loss = F.binary_cross_entropy_with_logits(
            mapped['correspondence_logit'], valid.expand(-1, beam + 1, -1, -1, -1))
        field = mapped['local_displacement_um']
        local_penalty = (.1 * field.square().mean().sqrt() / 1000
                         + .02 * ((field[..., 1:, :] - field[..., :-1, :]).abs().mean()
                                  + (field[..., 1:] - field[..., :-1]).abs().mean()) / 200)
        loss = (direct_loss + mapping_loss + 1.5 * fit_rank + teacher_loss
                + .1 * reliability_loss + local_penalty)
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / updates))
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1:
            probe_prediction = model.predict(batch['inputs'][:1])
            probe_mode = choice[:1, :1] // 2
            probe_reflection = choice[:1, :1] % 2
            probe_map = model.map(probe_prediction, batch['offsets'][:1],
                                  probe_mode, probe_reflection, (fit_side, fit_side),
                                  context['atlas'], batch['weights'][:1],
                                  feature_side=fit_side, source_shape=(side, side))
            probe_score = model.score_fitted_candidates(batch['inputs'][:1],
                                                        probe_prediction, probe_map,
                                                        context['atlas'], batch['weights'][:1])[0, 0]
            probe_branch = choice[0, 0]
            probe_log_mass = probe_prediction['log_mass'][0, probe_branch // 2]
            probe_log_reflection = torch.where(probe_branch.remainder(2).bool(),
                F.logsigmoid(probe_prediction['reflection_logit'][0, probe_branch // 2]),
                F.logsigmoid(-probe_prediction['reflection_logit'][0, probe_branch // 2]))
            feedback_gradient = torch.autograd.grad(
                probe_score - probe_log_mass - probe_log_reflection,
                model.pose[-1].weight)[0].norm()
            assert torch.isfinite(feedback_gradient) and feedback_gradient > 0
        selected = scores[:, :beam].detach().argmax(-1)
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'distinct_real_train_presentations': step,
               'synthetic_prior_tissue_um': float(dense.detach().flatten(1).gather(
                   1, choice[:, :1]).mean()),
               'synthetic_fitted_selected_um': float(mapped_tissue[:, :beam].detach().gather(
                   1, selected[:, None]).mean()),
               'synthetic_top8_oracle_um': float(mapped_tissue[:, :beam].detach().min(-1).values.mean()),
               'synthetic_teacher_mapped_um': float(mapped_tissue[:, beam].detach().mean()),
               'real_weak_prior_five_um': float(five[synthetic:].detach().flatten(1).gather(
                   1, prior[synthetic:].detach().argmax(-1)[:, None]).mean()),
               'direct_loss': float(direct_loss.detach()),
               'mapping_loss': float(mapping_loss.detach()),
               'fit_rank_loss': float(fit_rank.detach()),
               'teacher_loss': float(teacher_loss.detach()),
               'reliability_loss': float(reliability_loss.detach()),
               'local_penalty': float(local_penalty.detach()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        if step == 1:
            row['fit_evidence_to_pose_gradient_norm'] = float(feedback_gradient)
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates * synthetic,
    'distinct_real_train': updates,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
