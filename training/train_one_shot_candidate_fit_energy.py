"""Train the final model's atlas-match energy on true, perturbed and predicted planes."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT = ROOT / 'runs/one_shot_joint_physical_pose_002/joint_step_08000.pt'
RUN = ROOT / 'runs/one_shot_candidate_fit_energy_001'
SEED, UPDATES, BATCH, SIDE = 2026100801, 4000, 3, 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
context = load_streaming_synthetic_v7_64(device='cuda')
checkpoint = torch.load(PARENT, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(atlas_conditioning=True, fit_quality=True).cuda().eval()
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
assert missing and all(name.startswith('fit_quality_head.') for name in missing) and not unexpected
del checkpoint
model.requires_grad_(False)
model.fit_quality_head.requires_grad_(True)
optimizer = torch.optim.AdamW(model.fit_quality_head.parameters(), lr=3e-4, weight_decay=1e-4)
RUN.mkdir(parents=True, exist_ok=False)
provenance = RUN / 'synthetic_source_provenance.json'
provenance.write_text(json.dumps(context['provenance'], indent=2))
config = {'seed': SEED, 'updates': UPDATES, 'batch': BATCH, 'candidates_per_section': 6,
          'parent': str(PARENT), 'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
          'synthetic_source_provenance_sha256': hashlib.sha256(provenance.read_bytes()).hexdigest(),
          'candidate_design': 'true; local and wide tangent perturbations; detached prior, geometry-best and random predicted branches',
          'target': 'log1p(visible-tissue CCF error / 500um), plus soft physical branch ranking',
          'role': 'random-head warmup in the same whole-model lineage; pose and local mapper frozen here',
          'fit_feedback_into_pose': False, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in ('train_one_shot_candidate_fit_energy.py', 'arbitrary_plane_one_shot_model.py',
                           'arbitrary_plane_one_shot_stream.py')}}
(RUN / 'config.json').write_text(json.dumps(config, indent=2))
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 10000000


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               RUN / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w') as log, (RUN / 'draws.jsonl').open('w') as draws:
    for step in range(1, UPDATES + 1):
        accepted, pending = {}, list(range(BATCH))
        while pending:
            subject_indices = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, subject_indices, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(BATCH)]) for key in accepted[0]}
        with torch.no_grad():
            prediction = model.predict(batch['inputs'])
            indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            chart = torch.stack((indices.remainder(SIDE), indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            target = batch['centre'].reshape(BATCH, -1, 3).gather(1, indices[..., None].expand(-1, -1, 3))
            original_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(BATCH, 8, -1)
            original_error = (points(original_states, flags, chart[:, None, None])
                              - target[:, None, None]).norm(dim=-1).mean(-1)
            prior = prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)
            chosen = torch.stack((prior.flatten(1).argmax(-1), original_error.flatten(1).argmin(-1),
                                  torch.randint(16, (BATCH,), device='cuda')), -1)
            states = prediction['state'].clone()
            states[:, 0] = batch['state']
            directions = F.normalize(torch.randn(BATCH, 2, 3, device='cuda'), dim=-1)
            translations = F.normalize(torch.randn(BATCH, 2, 3, device='cuda'), dim=-1)
            update = torch.zeros(BATCH, 2, 9, device='cuda')
            update[:, 0, :3] = directions[:, 0] * torch.empty(BATCH, 1, device='cuda').uniform_(2, 12) * math.pi / 180
            update[:, 1, :3] = directions[:, 1] * torch.empty(BATCH, 1, device='cuda').uniform_(12, 40) * math.pi / 180
            update[:, 0, 3:6] = translations[:, 0] * torch.empty(BATCH, 1, device='cuda').uniform_(250, 1500)
            update[:, 1, 3:6] = translations[:, 1] * torch.empty(BATCH, 1, device='cuda').uniform_(1500, 4000)
            update[:, :, 6:8] = torch.randn(BATCH, 2, 2, device='cuda') * .04
            update[:, :, 8] = torch.randn(BATCH, 2, device='cuda') * .025
            states[:, 1:3] = compose_full_frame_state(batch['state'][:, None].expand(-1, 2, -1), update)
            for slot in range(3):
                states[:, 3 + slot] = prediction['state'][torch.arange(BATCH, device='cuda'), chosen[:, slot] // 2]
            reflection = torch.empty(BATCH, 6, device='cuda', dtype=torch.long)
            reflection[:, :3] = batch['reflection'][:, None]
            reflection[:, 2] = torch.where(torch.rand(BATCH, device='cuda') < .5,
                                           1 - batch['reflection'], batch['reflection'])
            reflection[:, 3:] = chosen % 2
            distance = (points(states[:, :6], reflection, chart[:, None])
                        - target[:, None]).norm(dim=-1).mean(-1)
            desired = torch.log1p(distance / 500)
        mapped = model.map({**prediction, 'state': states}, batch['offsets'],
                           torch.arange(6, device='cuda')[None].expand(BATCH, -1),
                           reflection, (SIDE, SIDE), context['atlas'], batch['weights'])
        energy = mapped['fit_energy']
        regression = F.smooth_l1_loss(energy, desired)
        ranking = F.kl_div(F.log_softmax(-energy, -1),
                           F.softmax(-distance / 1500, -1), reduction='batchmean')
        loss = regression + .5 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {'step': step, 'presentations': step * BATCH,
               'regression': float(regression.detach()), 'ranking': float(ranking.detach()),
               'selected_tissue_um': float(distance.gather(1, energy.detach().argmin(-1)[:, None]).mean()),
               'oracle_tissue_um': float(distance.min(-1).values.mean()),
               'true_energy': float(energy[:, 0].detach().mean()),
               'negative_energy': float(energy[:, 1:].detach().mean()),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 200 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 1000 == 0:
            save(step)
(RUN / 'completed.json').write_text(json.dumps({'updates': UPDATES,
    'accepted_synthetic': UPDATES * BATCH, 'fit_feedback_into_pose': False,
    'calibrated': False, 'public_benchmark_used': False,
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'updates': UPDATES}), flush=True)
