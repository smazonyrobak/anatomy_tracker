"""Scratch cross-modal 2D patch training with independent atlas in-plane rotation."""
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

from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

run = root / 'runs/atlas_roll_invariance_028'
seed, updates, sections, points, side = 2026102800, 3000, 2, 32, 256
draw_seed = 20261028000000000
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
rotation_rng = torch.Generator(device='cuda').manual_seed(seed + 1)
model = AtlasOrientedPatch025().cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
config = {'seed': seed, 'updates': updates, 'sections_per_batch': sections,
          'positive_points_per_section': points, 'atlas_negatives_per_section': points,
          'side': side, 'patch_side': 64, 'draw_seed_start': draw_seed,
          'temperature': .1, 'near_negative_exclusion_um': 300,
          'only_change_from_025': 'independent uniform [-180,180) degree atlas-patch in-plane rotation',
          'atlas_positive': 'finite-thickness known CCF surface; training truth only',
          'training_sections': 'fresh independent random synthetic TRAIN sections',
          'synthetic_provenance': context['provenance'], 'random_initialization': True,
          'prior_model_weights_features_pseudolabels': False,
          'real_training_images': 0, 'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('train_atlas_roll_invariance_028.py',
                                         'atlas_oriented_patch_025.py',
                                         'arbitrary_plane_one_shot_stream.py',
                                         'arbitrary_plane_streaming_synthetic_v7_64.py',
                                         'arbitrary_plane_streaming_synthetic_v7.py',
                                         'arbitrary_plane_full_frame_primitives.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'rotation_rng': rotation_rng.get_state(),
                'draw_seed': draw_seed, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'patch_step_{step:05d}.pt')


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
                                      for key in ('inputs', 'centre', 'state', 'offsets', 'weights', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(sections)])
                 for key in accepted[0]}
        positive_rows, positive_xy, negative_rows, negative_xy = [], [], [], []
        for row in range(sections):
            candidates = batch['valid_mask'][row].flatten().nonzero().flatten()
            chosen = candidates[torch.randperm(len(candidates), device='cuda')[:2 * points]]
            positive_rows.append(torch.full((points,), row, device='cuda', dtype=torch.long))
            negative_rows.append(torch.full((points,), row, device='cuda', dtype=torch.long))
            positive_xy.append(torch.stack((chosen[:points] // side, chosen[:points] % side), -1))
            negative_xy.append(torch.stack((chosen[points:] // side, chosen[points:] % side), -1))
        positive_rows, negative_rows = torch.cat(positive_rows), torch.cat(negative_rows)
        positive_xy, negative_xy = torch.cat(positive_xy), torch.cat(negative_xy)
        atlas_rows = torch.cat((positive_rows, negative_rows))
        atlas_xy = torch.cat((positive_xy, negative_xy))
        coordinates = batch['centre'][atlas_rows, atlas_xy[:, 0], atlas_xy[:, 1]]
        with torch.no_grad():
            atlas_image = atlas_surface_image(context['atlas'], batch['centre'], batch['state'],
                                              batch['offsets'], batch['weights'])
            query = extract_patches(batch['inputs'], positive_rows, positive_xy.float())
            key = extract_patches(atlas_image, atlas_rows, atlas_xy.float())
            angle = (torch.rand(len(key), device='cuda', generator=rotation_rng) * 2 - 1) * math.pi
            cosine, sine = angle.cos(), angle.sin()
            theta = torch.stack((cosine, -sine, torch.zeros_like(angle),
                                 sine, cosine, torch.zeros_like(angle)), -1).reshape(-1, 2, 3)
            key = F.grid_sample(key, F.affine_grid(theta, key.shape, align_corners=True),
                                mode='bilinear', padding_mode='zeros', align_corners=True)
        image_descriptor, atlas_descriptor = model(query, key)
        logits = image_descriptor @ atlas_descriptor.T / .1
        near = torch.cdist(coordinates[:sections * points], coordinates) < 300
        target = torch.arange(sections * points, device='cuda')
        near[target, target] = False
        logits = logits.masked_fill(near, -1e4)
        loss = (F.cross_entropy(logits, target) +
                F.cross_entropy(logits[:, :sections * points].T, target)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_sections': step * sections,
               'positive_pairs': step * sections * points, 'loss': float(loss.detach()),
               'paired_top1_of_128': float((logits.detach().argmax(-1) == target).float().mean()),
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
    'positive_pairs': updates * sections * points,
    'draws_sha256': hashlib.sha256((run / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((run / 'training.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
