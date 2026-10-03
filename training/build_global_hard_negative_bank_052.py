"""Render one atlas-only orientation bank for the frozen 052 checkpoints."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025

train = root / 'runs/global_hard_negative_052_pilot'
geometry = root / 'runs/atlas_global_patch_030'
out = root / 'runs/global_hard_negative_052_atlas_bank'
steps = (1000, 3000, 5000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((train / 'completed.json').read_text())['batches'] == 5000
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
positions = torch.from_numpy(np.load(geometry / 'positions_um.npy', allow_pickle=False)).cuda()
u = torch.from_numpy(np.load(geometry / 'orientation_u.npy', allow_pickle=False)).cuda() * 67
v = torch.from_numpy(np.load(geometry / 'orientation_v.npy', allow_pickle=False)).cuda() * 67
n = torch.from_numpy(np.load(geometry / 'orientation_n.npy', allow_pickle=False)).cuda()
assert positions.shape == (4027, 3) and u.shape == (512, 3)
models = []
for step in steps:
    checkpoint = torch.load(train / f'patch_step_{step:05d}.pt', map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasOrientedPatch025().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    models.append(model)
del checkpoint
out.mkdir(parents=True, exist_ok=False)
config = {'steps': steps, 'positions': len(positions), 'orientations_per_position': len(u),
          'atlas_geometry_completed_sha256': sha(geometry / 'completed.json'),
          'positions_sha256': sha(geometry / 'positions_um.npy'),
          'orientation_sha256': {name: sha(geometry / name) for name in
                                 ('orientation_u.npy', 'orientation_v.npy', 'orientation_n.npy')},
          'checkpoint_sha256': {str(step): sha(train / f'patch_step_{step:05d}.pt') for step in steps},
          'training_completed_sha256': sha(train / 'completed.json'),
          'atlas_patch_side': 64, 'pixel_scale_um': 67, 'through_plane_offsets_um': [-50, 50, 9],
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('build_global_hard_negative_bank_052.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
features = {step: np.lib.format.open_memmap(out / f'features_step_{step:05d}.npy',
    mode='w+', dtype='float16', shape=(len(positions) * len(u), 128)) for step in steps}
offset = torch.arange(64, device='cuda') - 31.5
dy, dx = torch.meshgrid(offset, offset, indexing='ij')
axial = torch.linspace(-50, 50, 9, device='cuda')
weights = torch.tensor([1] + [2] * 7 + [1], device='cuda', dtype=torch.float32) / 16
started = time.perf_counter()
with torch.inference_mode():
    for first in range(0, len(positions) * len(u), 128):
        last = min(first + 128, len(positions) * len(u))
        row = torch.arange(first, last, device='cuda')
        point = positions[row // 512]
        orientation = row % 512
        surface = (point[:, None, None] + dx[None, :, :, None] * u[orientation, None, None]
                   + dy[None, :, :, None] * v[orientation, None, None])
        coordinates = surface[:, None] + axial[None, :, None, None, None] * n[orientation, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights[None].expand(len(row), -1))
        support = rendered[:, 1:2].clamp(0, 1)
        patch = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        for step, model in zip(steps, models):
            descriptor = F.normalize(model.shared(model.atlas_stem(patch)), dim=-1)
            features[step][first:last] = descriptor.cpu().numpy().astype('float16')
        if last % 131072 == 0 or last == len(positions) * len(u):
            for feature in features.values():
                feature.flush()
            print(json.dumps({'atlas_patches': last, 'total': len(positions) * len(u),
                              'seconds': time.perf_counter() - started}), flush=True)
for feature in features.values():
    feature.flush()
del features
(out / 'completed.json').write_text(json.dumps({
    'files_sha256': {name: sha(out / name) for name in
        ('config.json', *(f'features_step_{step:05d}.npy' for step in steps))},
    'atlas_patches': len(positions) * len(u),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'atlas_patches': len(positions) * len(u),
                  'seconds': time.perf_counter() - started}), flush=True)
