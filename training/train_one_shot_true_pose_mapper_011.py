"""Teach the existing joint model local mapping at known synthetic planes."""
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

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_fitted_ranking_010/joint_step_12000.pt'
run = root / 'runs/one_shot_true_pose_mapper_011'
updates, synthetic, side, fit_side = 4000, 2, 256, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 12000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
modules = (model.lateral, model.warp_shared, model.warp_condition,
           model.warp, model.atlas_encoder, model.pair)
rates = (2e-5, 5e-5, 1e-4, 1e-4, 5e-5, 1e-4)
for module in modules:
    module.requires_grad_(True)
optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
                               for module, rate in zip(modules, rates)], weight_decay=1e-4)
subjects_rng = torch.Generator()
subjects_rng.set_state(checkpoint['subjects_rng'])
draw_seed = checkpoint['draw_seed']
torch.set_rng_state(checkpoint['torch_rng'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
run.mkdir(parents=True, exist_ok=False)
config = {'parent': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'updates': updates, 'synthetic_per_batch': synthetic, 'side': side,
          'fit_side': fit_side, 'synthetic_provenance': context['provenance'],
          'trainable': ['lateral', 'warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair'],
          'pose_and_fitted_matcher_frozen': True,
          'target': 'known-correct synthetic pose and reflection; observed visible tissue to CCF',
          'stage': 'mapper apprenticeship in deployment-target one-shot joint model; no inference truth input',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_one_shot_true_pose_mapper_011.py',
                                         'arbitrary_plane_one_shot_model.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
del checkpoint


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.clone()
    chart[..., 0] = torch.where(reflection[:, None].bool(),
                                (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, chart - .5)


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
                                      ('inputs', 'state', 'reflection', 'offsets',
                                       'weights', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        prediction = model.predict(batch['inputs'])
        states = prediction['state'].clone()
        states[:, 0] = batch['state']
        teacher = model.map({**prediction, 'state': states}, batch['offsets'],
                            torch.zeros((synthetic, 1), device='cuda', dtype=torch.long),
                            batch['reflection'][:, None], (fit_side, fit_side),
                            context['atlas'], batch['weights'], feature_side=fit_side,
                            source_shape=(side, side))
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
        target = batch['centre'].reshape(synthetic, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() / side
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        surface = teacher['centre_surface_ccf_ap_dv_ml_um'][:, 0].permute(0, 3, 1, 2)
        mapped = F.grid_sample(surface, grid[:, None], mode='bilinear',
                               padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)
        mapped_um = (mapped - target).norm(dim=-1).mean(-1)
        rigid_um = (points(batch['state'], batch['reflection'], chart) - target).norm(dim=-1).mean(-1)
        field = teacher['local_displacement_um']
        penalty = (.05 * field.square().mean().sqrt() / 1000
                   + .01 * ((field[..., 1:, :] - field[..., :-1, :]).abs().mean()
                            + (field[..., 1:] - field[..., :-1]).abs().mean()) / 200)
        loss = mapped_um.mean() / 1000 + penalty
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'teacher_rigid_um': float(rigid_um.detach().mean()),
               'teacher_mapped_um': float(mapped_um.detach().mean()),
               'penalty': float(penalty.detach()), 'gradient_norm': float(gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'checkpoint_sha256': hashlib.sha256(
        (run / f'joint_step_{updates:05d}.pt').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'public_benchmark_used': False, 'calibrated': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
