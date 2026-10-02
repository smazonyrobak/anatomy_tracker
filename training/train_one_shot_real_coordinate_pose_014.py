"""Adapt the spatial pose head using independent real weak-affine and arbitrary synthetic fields."""
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

parent = root / 'runs/one_shot_dense_coordinate_pose_013/joint_step_05000.pt'
run = root / 'runs/one_shot_real_coordinate_pose_014'
seed, updates, synthetic, real_per_batch, side = 2026101400, 12000, 2, 2, 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
prior_schedules = sorted([*root.glob('runs/*/real_schedule.npy'),
                          *root.glob('runs/*/new_real_schedule.npy')])
used_real = {tuple(map(int, row)) for path in prior_schedules for row in np.load(path)}
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < updates * real_per_batch:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == updates * real_per_batch:
                break
assert len(set(schedule)) == len(schedule) and not set(schedule) & used_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic,
          'real_per_batch': real_per_batch, 'side': side, 'parent': str(parent),
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'real_schedule_sha256': hashlib.sha256((run / 'real_schedule.npy').read_bytes()).hexdigest(),
          'excluded_real_schedule_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in prior_schedules},
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'trainable': 'dense_coordinate_head only; direct multimodal pose, encoder, atlas mapper and fit score frozen',
          'synthetic_coordinate_target': 'full-frame rigid CCF field, fivefold tissue emphasis',
          'real_coordinate_target': 'inherited weak Allen affine field over full model image',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_real_coordinate_pose_014.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_geometry.py',
                                         'arbitrary_plane_full_frame_primitives.py',
                                         'arbitrary_plane_joint_model_v7.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_reserved_real_stream_v8.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
cells = torch.arange(16, device='cuda').float() * 16 + 7.5
yy, xx = torch.meshgrid(cells / side, cells / side, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(256, 2)


def points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    observed = chart[None].expand(len(state), -1, -1).clone()
    observed[..., 0] = torch.where(reflection[:, None].bool(),
                                    255 / side - observed[..., 0], observed[..., 0])
    return centre[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, observed - .5)


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 5000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
model.dense_coordinate_head.requires_grad_(True)
optimizer = torch.optim.AdamW(model.dense_coordinate_head.parameters(), lr=5e-5,
                              weight_decay=1e-4)
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
                                      ('inputs', 'state', 'reflection', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        real_rows = []
        for slot in range(real_per_batch):
            donor, section = schedule[(step - 1) * real_per_batch + slot]
            observation = sample_reserved_real_train(real, donor, [section], device='cuda')
            real_rows.append(observation)
            draws.write(json.dumps({**observation['identities'][0], 'step': step,
                                    'slot': synthetic + slot, 'used': True,
                                    'label_role': real['label_role']}) + '\n')
        image = torch.cat((batch['inputs'], *(F.interpolate(row['inputs'], (side, side),
                           mode='bilinear', align_corners=False) for row in real_rows)))
        truth = torch.cat((batch['state'], *(row['state'] for row in real_rows)))
        reflection = torch.cat((batch['reflection'], *(row['reflection'] for row in real_rows)))
        target = points(truth, reflection).reshape(len(image), 16, 16, 3).permute(0, 3, 1, 2)
        target = (target - model.center_origin[None, :, None, None]) / model.center_scale[None, :, None, None]
        field = model.predict(image)['dense_coordinate']
        error = F.smooth_l1_loss(field[:, :3], target, beta=.05,
                                 reduction='none').sum(1, keepdim=True)
        occupancy = F.avg_pool2d(batch['valid_mask'][:, None].float(), 16)
        emphasis = .2 + .8 * occupancy
        synthetic_loss = ((error[:synthetic] * emphasis).flatten(1).sum(-1)
                          / emphasis.flatten(1).sum(-1)).mean()
        real_loss = error[synthetic:].mean()
        validity_loss = F.binary_cross_entropy_with_logits(field[:synthetic, 3:4], occupancy)
        loss = 3 * synthetic_loss + 1.5 * real_loss + validity_loss
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / updates))
        optimizer.param_groups[0]['lr'] = 5e-5 * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.dense_coordinate_head.parameters(),
                                                  5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'distinct_real_train_presentations': step * real_per_batch,
               'synthetic_coordinate_loss': float(synthetic_loss.detach()),
               'real_weak_coordinate_loss': float(real_loss.detach()),
               'synthetic_validity_loss': float(validity_loss.detach()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 2000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates * synthetic,
    'distinct_real_train': updates * real_per_batch,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
