"""Continue scratch atlas-patch descriptors against whole-brain hard negatives."""
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

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

parent = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
bank_dir = root / 'runs/atlas_global_patch_030'
run = root / 'runs/global_hard_negative_052_pilot'
seed, batches, points, negatives = 2026105200, 5000, 16, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
context = load_streaming_synthetic_v7_64(device='cuda')
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
bank_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda() * 67
bank_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda() * 67
bank_n = torch.from_numpy(np.load(bank_dir / 'orientation_n.npy', allow_pickle=False)).cuda()
assert bank.shape == (2061824, 128) and positions.shape == (4027, 3)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model = AtlasOrientedPatch025().cuda().train()
model.load_state_dict(saved['model'])
miner = AtlasOrientedPatch025().cuda().eval()
miner.load_state_dict(saved['model'])
miner.requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
optimizer.load_state_dict(saved['optimizer'])
for group in optimizer.param_groups:
    group['lr'] = 5e-5
del saved
run.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
config = {'seed': seed, 'batches': batches, 'sections_per_batch': 1,
          'positive_points_per_section': points, 'hard_atlas_negatives_per_point': negatives,
          'miner_top_k': 512, 'negative_exclusion_um': 1000, 'temperature': .1,
          'atlas_patch_scale_um_per_pixel': 67, 'atlas_psf_um': [-50, 50, 9],
          'training': 'fresh independent arbitrary-plane TRAIN sections, one appearance each',
          'lineage': '025 complete model/optimizer continuation; frozen 025/030 miner; no external/pretrained features',
          'parent_sha256': sha(parent), 'bank_completed_sha256': sha(bank_dir / 'completed.json'),
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_global_hard_negative_052.py', 'atlas_oriented_patch_025.py')},
          'synthetic_provenance': context['provenance'],
          'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))
axis = torch.arange(64, device='cuda') - 31.5
dy, dx = torch.meshgrid(axis, axis, indexing='ij')
axial = torch.linspace(-50, 50, 9, device='cuda')
weights = torch.tensor([1] + [2] * 7 + [1], device='cuda', dtype=torch.float32) / 16


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'patch_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        while True:
            virtual = torch.randint(len(context['subjects']), (1,), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=256)
            draw_seed += 1
            used = bool(sampled['eligible'][0] and sampled['valid_mask'][0].sum() >= points)
            draws.write(json.dumps({**sampled['provenance'][0], 'batch': step, 'used': used}) + '\n')
            if used:
                break
        valid = sampled['valid_mask'][0].flatten().nonzero().flatten()
        chosen = valid[torch.randperm(len(valid), device='cuda')[:points]]
        query_yx = torch.stack((chosen // 256, chosen % 256), -1).float()
        rows = torch.zeros(points, device='cuda', dtype=torch.long)
        with torch.no_grad():
            query = extract_patches(sampled['inputs'], rows, query_yx)
            atlas_image = atlas_surface_image(context['atlas'], sampled['centre'],
                                              sampled['state'], sampled['offsets'], sampled['weights'])
            positive = extract_patches(atlas_image, rows, query_yx)
            truth = sampled['centre'].reshape(-1, 3)[chosen]
            frozen_query = F.normalize(miner.shared(miner.image_stem(query)), dim=-1)
            similarity = frozen_query.half() @ bank.T
            top_score, top_id = similarity.topk(512, -1)
            top_world = positions[top_id // 512]
            far = (top_world - truth[:, None]).norm(dim=-1) >= 1000
            assert bool((far.sum(-1) >= negatives).all())
            hard_rank = top_score.masked_fill(~far, -1e4).topk(negatives, -1).indices
            hard_id = top_id.gather(1, hard_rank).flatten()
            hard_position = positions[hard_id // 512]
            orientation = hard_id % 512
            negative_parts = []
            for first in range(0, len(hard_id), 64):
                end = min(first + 64, len(hard_id))
                point = hard_position[first:end]
                index = orientation[first:end]
                surface = (point[:, None, None] + dx[None, :, :, None] * bank_u[index, None, None]
                           + dy[None, :, :, None] * bank_v[index, None, None])
                coordinates = surface[:, None] + axial[None, :, None, None, None] * bank_n[index, None, None, None]
                rendered = render_finite_thickness_coordinate_grid(context['atlas'], coordinates,
                    (0., 0., 0.), (25., 25., 25.), weights[None].expand(len(index), -1))
                support = rendered[:, 1:2].clamp(0, 1)
                negative_parts.append(torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1))
            atlas_patches = torch.cat((positive, *negative_parts))
        image_descriptor, atlas_descriptor = model(query, atlas_patches)
        logits = image_descriptor @ atlas_descriptor.T / .1
        nearby = torch.cdist(truth, truth) < 300
        nearby.fill_diagonal_(False)
        logits[:, :points] = logits[:, :points].masked_fill(nearby, -1e4)
        target = torch.arange(points, device='cuda')
        loss = .8 * F.cross_entropy(logits, target) + .2 * F.cross_entropy(logits[:, :points].T, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'synthetic_presentations': step, 'positive_pairs': step * points,
               'loss': float(loss.detach()),
               'mined_top1': float((logits.detach().argmax(-1) == target).float().mean()),
               'gradient_norm': float(gradient),
               'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 1000 == 0:
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches, 'positive_pairs': batches * points,
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'config_sha256': sha(run / 'config.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
