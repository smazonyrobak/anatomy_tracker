"""Continue the standalone one-shot model with broader arbitrary-plane capture."""
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

parent = root / 'runs/one_shot_candidate_fit_energy_001/joint_step_01000.pt'
run = root / 'runs/one_shot_pose_capture_003'
seed, updates, side, synthetic = 2026100803, 40000, 256, 3
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda()
old_pose_weight = checkpoint['model'].pop('pose.2.weight')
old_pose_bias = checkpoint['model'].pop('pose.2.bias')
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
assert set(missing) == {'pose.2.weight', 'pose.2.bias'} and not unexpected
with torch.no_grad():
    model.pose[-1].weight[:8 * 21].copy_(old_pose_weight)
    model.pose[-1].bias[:8 * 21].copy_(old_pose_bias)
del checkpoint
model.fit_quality_head.requires_grad_(False)
base_rates = (2e-5, 1e-4, 3e-5)
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': base_rates[0]},
    {'params': model.pose.parameters(), 'lr': base_rates[1]},
    {'params': (parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad and not name.startswith(('encoder.', 'pose.'))),
     'lr': base_rates[2]},
], weight_decay=1e-4)

used_real = {tuple(row) for path in (
    root / 'runs/one_shot_joint_mixed_real_003/real_schedule.npy',
    root / 'runs/one_shot_fit_feedback_matched_001/new_real_schedule.npy',
    root / 'runs/one_shot_joint_physical_pose_002/real_schedule.npy')
    for row in np.load(path)}
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
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic, 'side': side,
          'modes': 16, 'parent': str(parent),
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'loss': 'best and ranked 32-branch tissue/full-frame geometry plus 4mm antipodal-normal anchor; true-pose local mapping; cosine learning rates',
          'fit_feedback_into_pose': False, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_pose_capture_003.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
landmarks = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
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
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
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
        draws.write(json.dumps({**observation['identities'][0], 'step': step, 'slot': synthetic,
                                'used': True, 'label_role': real['label_role']}) + '\n')
        inputs = torch.cat((batch['inputs'], observation['inputs']))
        truth = torch.cat((batch['state'], observation['state']))
        reflection = torch.cat((batch['reflection'], observation['reflection']))
        prediction = model.predict(inputs)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        branch_flags = flags[None, None].expand(len(inputs), model.modes, -1)
        reference = points(truth, reflection, landmarks)
        five = (points(states, branch_flags, landmarks) - reference[:, None, None]).norm(dim=-1).mean(-1)
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
        targets = batch['centre'].reshape(synthetic, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side), indices.div(side, rounding_mode='floor')), -1).float() / side
        dense = (points(states[:synthetic], branch_flags[:synthetic], chart[:, None, None])
                 - targets[:, None, None]).norm(dim=-1).mean(-1)
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        reference_normals = full_frame_state_to_components(truth)[1][:, :, 2]
        normal = 1 - (normals * reference_normals[:, None]).sum(-1).abs().clamp_max(1)
        distance = torch.cat((.75 * dense + .25 * five[:synthetic] + 4000 * normal[:synthetic, :, None],
                              five[synthetic:]), 0)
        prior = prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)
        flat_distance, flat_prior = distance.flatten(1), prior.flatten(1)
        direct = flat_distance.min(-1).values.mean() / 1000
        marginal = -torch.logsumexp(flat_prior - flat_distance / 1500, -1).mean()
        rank = F.kl_div(flat_prior, (-flat_distance.detach() / 1000).softmax(-1),
                        reduction='batchmean')
        mapping_prediction = {key: value[:synthetic] if torch.is_tensor(value) else value
                              for key, value in prediction.items()}
        true_state = mapping_prediction['state'].clone()
        true_state[:, 0] = batch['state']
        mapped = model.map({**mapping_prediction, 'state': true_state}, batch['offsets'],
                           torch.zeros((synthetic, 1), device='cuda', dtype=torch.long),
                           batch['reflection'][:, None], (side, side), context['atlas'], batch['weights'])
        error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
        valid = batch['valid_mask']
        mapping = (error * valid).sum() / valid.sum() / 500
        field = mapped['local_displacement_um'][:, 0]
        magnitude = field.square().mean().sqrt() / 1000
        smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                  + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
        loss = direct + .75 * marginal + .75 * rank + .5 * mapping + .1 * magnitude + .02 * smooth
        multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * step / updates))
        for group, rate in zip(optimizer.param_groups, base_rates):
            group['lr'] = rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        selected = flat_prior.detach().argmax(-1)
        selected_error = five.detach().flatten(1).gather(1, selected[:, None])
        selected_tissue = dense.detach().flatten(1).gather(1, selected[:synthetic, None])
        row = {'step': step, 'presentations': step * (synthetic + 1),
               'selected_synthetic_tissue_um': float(selected_tissue.mean()),
               'selected_synthetic_five_um': float(selected_error[:synthetic].mean()),
               'selected_real_weak_five_um': float(selected_error[-1]),
               'oracle_synthetic_five_um': float(five[:synthetic].detach().flatten(1).min(-1).values.mean()),
               'selected_synthetic_normal_deg': float(torch.rad2deg(torch.acos(
                   (normals[:synthetic].detach() * reference_normals[:synthetic, None]).sum(-1)
                   .abs().clamp(0, 1))).gather(1, (selected[:synthetic] // 2)[:, None]).mean()),
               'direct': float(direct.detach()), 'marginal': float(marginal.detach()),
               'rank': float(rank.detach()), 'true_pose_map_um': float(mapping.detach() * 500),
               'gradient': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 5000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'new_synthetic_accepted': updates * synthetic, 'new_distinct_real_train': updates,
    'seconds': time.perf_counter() - started, 'fit_feedback_into_pose': False,
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates}), flush=True)
