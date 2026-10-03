"""Fit the image query encoder to the full frozen atlas key bank."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches

parent = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
bank_dir = root / 'runs/atlas_global_patch_030'
run = root / 'runs/frozen_bank_global_query_055_pilot'
seed, batches, points, nearby_positions, nearby_frames = 2026105500, 1500, 16, 4, 4
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
bank_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
bank_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
assert bank.shape == (2061824, 128) and positions.shape == (4027, 3)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model = AtlasOrientedPatch025().cuda().train()
model.load_state_dict(saved['model'])
teacher = AtlasOrientedPatch025().cuda().eval()
teacher.load_state_dict(saved['model'])
teacher.requires_grad_(False)
model.atlas_stem.requires_grad_(False)
optimizer = torch.optim.AdamW([*model.image_stem.parameters(), *model.shared.parameters()],
                              lr=2e-5, weight_decay=1e-4)
del saved
run.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
config = {'seed': seed, 'batches': batches, 'sections_per_batch': 1,
          'query_points_per_section': points, 'positive_bank_positions': nearby_positions,
          'positive_bank_frames': nearby_frames, 'positive_radius_um': 500,
          'temperature': .1, 'teacher_preservation_weight': .5,
          'atlas_bank_keys': len(bank), 'atlas_bank_trainable': False,
          'training': 'fresh independent arbitrary-plane TRAIN sections, one appearance each',
          'lineage': '025 scratch query weights; fixed 025/030 atlas key bank; no external weights',
          'parent_sha256': sha(parent), 'bank_completed_sha256': sha(bank_dir / 'completed.json'),
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_frozen_bank_global_query_055.py', 'atlas_oriented_patch_025.py')},
          'synthetic_provenance': context['provenance'],
          'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'query_step_{step:05d}.pt')


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
        query = extract_patches(sampled['inputs'],
                                torch.zeros(points, device='cuda', dtype=torch.long), query_yx)
        truth = sampled['centre'].reshape(-1, 3)[chosen]
        _, frame, basis = full_frame_state_to_components(sampled['state'])
        edges = frame[0, :, :2] @ basis[0]
        u = edges[:, 0][None].expand(points, -1).clone()
        v = edges[:, 1][None].expand(points, -1).clone()
        if bool(sampled['reflection'][0]):
            u = -u
        parity_flip = torch.cross(u, v, dim=-1)[:, 2] < 0
        u = F.normalize(torch.where(parity_flip[:, None], -u, u), dim=-1)
        v = F.normalize(v, dim=-1)
        query = torch.where(parity_flip[:, None, None, None], query.flip(-1), query)
        orientation = (u @ bank_u.T + v @ bank_v.T).topk(nearby_frames, -1).indices
        distance = torch.cdist(truth, positions)
        near_distance, near_position = distance.topk(nearby_positions, largest=False)
        near_mask = near_distance < 500
        near_mask[:, 0] = True
        positive_id = (near_position[:, :, None] * 512 + orientation[:, None]).reshape(points, -1)
        positive_mask = near_mask[:, :, None].expand(-1, -1, nearby_frames).reshape(points, -1)
        descriptor = F.normalize(model.shared(model.image_stem(query)), dim=-1)
        with torch.no_grad():
            teacher_descriptor = F.normalize(teacher.shared(teacher.image_stem(query)), dim=-1)
        logits = (descriptor.half() @ bank.T).float() / .1
        positive = logits.gather(1, positive_id).masked_fill(~positive_mask, -1e4)
        retrieval = (torch.logsumexp(logits, -1) - torch.logsumexp(positive, -1)).mean()
        preservation = (1 - (descriptor * teacher_descriptor).sum(-1)).mean()
        loss = retrieval + .5 * preservation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [*model.image_stem.parameters(), *model.shared.parameters()], 5., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            top = logits.topk(16, -1).indices
            top_distance = (positions[top // 512] - truth[:, None]).norm(dim=-1)
        row = {'batch': step, 'synthetic_presentations': step,
               'query_points': step * points, 'retrieval_loss': float(retrieval.detach()),
               'teacher_preservation': float(preservation.detach()),
               'train_point_recall1_500um': float((top_distance[:, 0] < 500).float().mean()),
               'train_point_recall16_500um': float((top_distance.min(-1).values < 500).float().mean()),
               'positive_position_distance_um': float(near_distance[:, 0].mean()),
               'gradient_norm': float(gradient),
               'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step in (500, 1500):
            save(step)
(run / 'completed.json').write_text(json.dumps({
    'batches': batches, 'accepted_synthetic': batches,
    'query_points': batches * points,
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'config_sha256': sha(run / 'config.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
