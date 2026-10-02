"""Train scratch 2D histology / 3D atlas descriptors on fresh TRAIN sections."""
import hashlib
import json
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

from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_point_proposal_024 import AtlasPointProposal024, atlas_cubes, observed_patches

run = root / 'runs/atlas_point_proposal_024'
seed, updates, sections, points, side = 2026102400, 3000, 2, 32, 256
draw_seed = 20261024000000000
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = F.avg_pool3d(context['atlas'][None], 4, 4)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
model = AtlasPointProposal024().cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'updates': updates, 'sections_per_batch': sections,
          'points_per_section': points, 'side': side, 'draw_seed_start': draw_seed,
          'patch_side': 64, 'cube_side': 25, 'atlas_sampling_um': 100,
          'temperature': .1, 'near_negative_exclusion_um': 300,
          'atlas_cube_augmentation': 'independent signed axis permutations',
          'training_sections': 'fresh synthetic TRAIN sections, never cached image variants',
          'synthetic_provenance': context['provenance'], 'random_initialization': True,
          'prior_model_weights_features_pseudolabels': False,
          'real_training_images': 0, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_atlas_point_proposal_024.py',
                                         'atlas_point_proposal_024.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_streaming_synthetic_v7.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'point_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(sections))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1]
                                      for key in ('inputs', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(sections)])
                 for key in accepted[0]}
        rows, xy = [], []
        for row in range(sections):
            candidates = batch['valid_mask'][row].flatten().nonzero().flatten()
            chosen = candidates[torch.randperm(len(candidates), device='cuda')[:points]]
            rows.append(torch.full((points,), row, device='cuda', dtype=torch.long))
            xy.append(torch.stack((chosen // side, chosen % side), -1))
        rows, xy = torch.cat(rows), torch.cat(xy)
        coordinates = batch['centre'][rows, xy[:, 0], xy[:, 1]]
        patch = observed_patches(batch['inputs'], rows, xy.float())
        cubes = atlas_cubes(atlas, coordinates)
        cubes = torch.stack([torch.flip(cube.permute(0, *(axis + 1 for axis in
                            torch.randperm(3).tolist())),
                            [axis + 1 for axis in range(3) if torch.rand(()) < .5])
                             for cube in cubes])
        query, key = model(patch, cubes)
        logits = query @ key.T / .1
        near = torch.cdist(coordinates, coordinates) < 300
        near.fill_diagonal_(False)
        logits = logits.masked_fill(near, -1e4)
        target = torch.arange(sections * points, device='cuda')
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_sections': step * sections,
               'point_pairs': step * sections * points, 'loss': float(loss.detach()),
               'paired_top1': float((logits.detach().argmax(-1) == target).float().mean()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 1000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic_sections': updates * sections,
    'point_pairs': updates * sections * points,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
