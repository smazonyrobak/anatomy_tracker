"""Teach the final model's match head to rank its current predicted planes."""
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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_pose_capture_003/joint_step_40000.pt'
run = root / 'runs/one_shot_candidate_energy_004'
seed, updates, side, synthetic = 2026100804, 10000, 256, 2
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 40000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
model.fit_quality_head.requires_grad_(True)
optimizer = torch.optim.AdamW(model.fit_quality_head.parameters(), lr=1e-4, weight_decay=1e-4)

used_real = {tuple(row) for path in (
    root / 'runs/one_shot_joint_mixed_real_003/real_schedule.npy',
    root / 'runs/one_shot_fit_feedback_matched_001/new_real_schedule.npy',
    root / 'runs/one_shot_joint_physical_pose_002/real_schedule.npy',
    root / 'runs/one_shot_pose_capture_003/real_schedule.npy')
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
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic, 'real_per_batch': 1,
          'side': side, 'modes': 16, 'candidates_per_image': 7, 'parent': str(parent),
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'candidates': 'true; near and wide tangent perturbations; predicted top-prior, best-geometry, second-best-geometry, random',
          'objective': 'log physical-distance regression and separate ranking of actual predicted candidates; real weak-label loss half weight',
          'pose_and_mapper_frozen': True, 'fit_feedback_into_pose': False,
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_candidate_energy_004.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
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
        with torch.no_grad():
            prediction = model.predict(image)
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            target = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, indices[..., None].expand(-1, -1, 3))
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            branch_flags = flags[None, None].expand(len(image), model.modes, -1)
            synthetic_error = (points(states[:synthetic], branch_flags[:synthetic],
                                      chart[:, None, None]) - target[:, None, None]).norm(dim=-1).mean(-1)
            real_reference = points(truth[synthetic:], truth_reflection[synthetic:], corners)
            real_error = (points(states[synthetic:], branch_flags[synthetic:], corners)
                          - real_reference[:, None, None]).norm(dim=-1).mean(-1)
            error = torch.cat((synthetic_error, real_error)).flatten(1)
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            best_two = error.topk(2, largest=False).indices
            chosen = torch.stack((prior.flatten(1).argmax(-1), best_two[:, 0],
                                  best_two[:, 1], torch.randint(32, (len(image),), device='cuda')), -1)
            candidate = prediction['state'][:, :7].clone()
            candidate[:, 0] = truth
            rotation = F.normalize(torch.randn(len(image), 2, 3, device='cuda'), dim=-1)
            translation = F.normalize(torch.randn(len(image), 2, 3, device='cuda'), dim=-1)
            update = torch.zeros(len(image), 2, 9, device='cuda')
            update[:, 0, :3] = rotation[:, 0] * torch.empty(len(image), 1, device='cuda').uniform_(2, 12) * math.pi / 180
            update[:, 1, :3] = rotation[:, 1] * torch.empty(len(image), 1, device='cuda').uniform_(12, 40) * math.pi / 180
            update[:, 0, 3:6] = translation[:, 0] * torch.empty(len(image), 1, device='cuda').uniform_(250, 1500)
            update[:, 1, 3:6] = translation[:, 1] * torch.empty(len(image), 1, device='cuda').uniform_(1500, 4000)
            update[:, :, 6:8] = torch.randn(len(image), 2, 2, device='cuda') * .04
            update[:, :, 8] = torch.randn(len(image), 2, device='cuda') * .025
            candidate[:, 1:3] = compose_full_frame_state(truth[:, None].expand(-1, 2, -1), update)
            for slot in range(4):
                candidate[:, 3 + slot] = prediction['state'][torch.arange(len(image), device='cuda'),
                                                             chosen[:, slot] // 2]
            reflection = torch.empty(len(image), 7, device='cuda', dtype=torch.long)
            reflection[:, :3] = truth_reflection[:, None]
            reflection[:, 2] = torch.where(torch.rand(len(image), device='cuda') < .5,
                                           1 - truth_reflection, truth_reflection)
            reflection[:, 3:] = chosen % 2
            synthetic_distance = (points(candidate[:synthetic], reflection[:synthetic], chart[:, None])
                                  - target[:, None]).norm(dim=-1).mean(-1)
            real_distance = (points(candidate[synthetic:], reflection[synthetic:], corners)
                             - real_reference[:, None]).norm(dim=-1).mean(-1)
            distance = torch.cat((synthetic_distance, real_distance))
            desired = torch.log1p(distance / 500)
        mapped = model.map({**prediction, 'state': candidate}, offsets,
                           torch.arange(7, device='cuda')[None].expand(len(image), -1),
                           reflection, (side, side), context['atlas'], weights)
        energy = mapped['fit_energy']
        regression = F.smooth_l1_loss(energy, desired, reduction='none').mean(-1)
        all_rank = F.kl_div(F.log_softmax(-energy, -1),
                            F.softmax(-distance / 1500, -1), reduction='none').sum(-1)
        predicted_rank = F.kl_div(F.log_softmax(-energy[:, 3:], -1),
                                  F.softmax(-distance[:, 3:] / 1000, -1),
                                  reduction='none').sum(-1)
        per_image = regression + .25 * all_rank + predicted_rank
        loss = (per_image[:synthetic].sum() + .5 * per_image[synthetic:].sum()) / (synthetic + .5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {'step': step, 'presentations': step * len(image),
               'synthetic_loss': float(per_image[:synthetic].detach().mean()),
               'real_weak_loss': float(per_image[synthetic:].detach().mean()),
               'predicted_rank': float(predicted_rank.detach().mean()),
               'selected_synthetic_tissue_um': float(distance[:synthetic].gather(
                   1, energy[:synthetic].detach().argmin(-1)[:, None]).mean()),
               'true_energy': float(energy[:, 0].detach().mean()),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 2000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': updates * synthetic, 'distinct_real_train': updates,
    'fit_feedback_into_pose': False, 'calibrated': False, 'public_benchmark_used': False,
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates}), flush=True)
