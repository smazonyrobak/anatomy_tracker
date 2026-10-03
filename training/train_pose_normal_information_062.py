"""Frozen-encoder antipodal-normal information probe on fresh arbitrary planes."""
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
from torch import nn

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/pose_normal_information_062_pilot'
seed, batches, batch_size, side = 2026100362, 5000, 4, 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000

context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
anchors = model.normal_anchor_frames[:, :, 2].detach()
head = nn.Sequential(nn.Linear(320 * 4 * 4, 512), nn.GELU(), nn.Linear(512, 64)).cuda()
optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
run.mkdir(parents=True, exist_ok=False)
with parent.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with Path(__file__).open('rb') as stream:
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
config = {'seed': seed, 'batches': batches, 'batch_size': batch_size,
          'parent': str(parent), 'parent_sha256': parent_sha,
          'source_sha256': source_sha, 'synthetic_provenance': context['provenance'],
          'trainable': '64-anchor antipodal-normal probe only',
          'frozen': '059 encoder, pose, mapping, atlas',
          'training_truth': 'nearest anchor to exact observed-affine synthetic normal',
          'calibrated': False, 'public_benchmark_used': False}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.save({'step': 0, 'head': head.state_dict(), 'config': config,
            'calibrated': False}, run / 'normal_step_00000.pt')
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted = []
        while len(accepted) < batch_size:
            virtual = torch.randint(len(context['subjects']),
                (batch_size - len(accepted),), generator=subjects_rng)
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step,
                                        'used': used}) + '\n')
                if used:
                    accepted.append(row)
            if accepted:
                break
        images = sampled['inputs'][accepted]
        truth_state = sampled['state'][accepted]
        while len(images) < batch_size:
            virtual = torch.randint(len(context['subjects']),
                (batch_size - len(images),), generator=subjects_rng)
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            extra = []
            for row, identity in enumerate(sampled['provenance']):
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step,
                                        'used': used}) + '\n')
                if used:
                    extra.append(row)
            if extra:
                images = torch.cat((images, sampled['inputs'][extra]))
                truth_state = torch.cat((truth_state, sampled['state'][extra]))
        with torch.no_grad():
            feature = images
            for layer in model.encoder:
                feature = layer(feature)
            feature = F.adaptive_avg_pool2d(feature, 4).flatten(1)
            normal = full_frame_state_to_components(truth_state)[1][..., :, 2]
            target = (normal @ anchors.T).abs().argmax(-1)
        logits = head(feature)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            angle = torch.rad2deg(torch.acos((normal * anchors[logits.argmax(-1)]).sum(-1)
                                           .abs().clamp(max=1))).mean()
        row = {'step': step, 'accepted_synthetic': step * batch_size,
               'normal_ce': float(loss.detach()), 'normal_map_angle_deg': float(angle),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step % 1000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            torch.save({'step': step, 'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(), 'config': config,
                        'calibrated': False}, run / f'normal_step_{step:05d}.pt')
with (run / 'config.json').open('rb') as stream:
    config_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (run / 'draws.jsonl').open('rb') as stream:
    draws_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (run / 'training.jsonl').open('rb') as stream:
    training_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(run / 'completed.json').write_text(json.dumps({'batches': batches,
    'accepted_synthetic': batches * batch_size, 'draws_sha256': draws_sha,
    'training_sha256': training_sha, 'config_sha256': config_sha,
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
