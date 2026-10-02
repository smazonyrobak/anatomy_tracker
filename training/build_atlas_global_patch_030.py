"""Frozen 025 atlas-only oriented-patch bank for unknown-plane retrieval."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025

checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/atlas_global_patch_030'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
support = F.avg_pool3d(atlas[1:2, None], 4, 4)[0, 0].cpu().numpy()
grid = np.argwhere(support[2::5, 2::5, 2::5] > .5)
positions = ((grid * 5 + 2.375) * 100).astype('float32')
assert len(positions) == 4027

index = np.arange(128)
z = 1 - (index + .5) / 64
phase = index * math.pi * (3 - math.sqrt(5))
normals = np.stack((np.sqrt(1 - z * z) * np.cos(phase),
                    np.sqrt(1 - z * z) * np.sin(phase), z), -1)
normals = normals[z >= 0]
reference = np.where(np.abs(normals[:, 2:3]) < .9,
                     np.array([0., 0., 1.]), np.array([0., 1., 0.]))
u = np.cross(reference, normals)
u /= np.linalg.norm(u, axis=-1, keepdims=True)
v = np.cross(normals, u)
roll = np.arange(8) * (2 * math.pi / 8)
bank_u = (np.cos(roll)[None, :, None] * u[:, None]
          + np.sin(roll)[None, :, None] * v[:, None]).reshape(-1, 3).astype('float32')
bank_v = (-np.sin(roll)[None, :, None] * u[:, None]
          + np.cos(roll)[None, :, None] * v[:, None]).reshape(-1, 3).astype('float32')
bank_n = np.repeat(normals, 8, axis=0).astype('float32')
assert len(bank_u) == 512

saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model = AtlasOrientedPatch025().cuda().eval()
model.load_state_dict(saved['model'], strict=True)
del saved

out.mkdir(parents=True, exist_ok=False)
np.save(out / 'positions_um.npy', positions)
np.save(out / 'orientation_u.npy', bank_u)
np.save(out / 'orientation_v.npy', bank_v)
np.save(out / 'orientation_n.npy', bank_n)
config = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'atlas_shape': list(atlas_array.shape),
          'position_count': len(positions), 'orientations_per_position': len(bank_u),
          'patch_side': 64, 'pixel_scale_um': 67.,
          'through_plane_offsets_um': np.linspace(-50, 50, 9).tolist(),
          'through_plane_weights': ([1] + [2] * 7 + [1]),
          'descriptor_dim': 128, 'descriptor_storage_dtype': 'float16',
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('build_atlas_global_patch_030.py', 'atlas_oriented_patch_025.py',
                                         'arbitrary_plane_full_frame_primitives.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
features = np.lib.format.open_memmap(out / 'features.npy', mode='w+', dtype='float16',
                                      shape=(len(positions) * len(bank_u), 128))
position_gpu = torch.from_numpy(positions).cuda()
u_gpu = torch.from_numpy(bank_u).cuda() * 67
v_gpu = torch.from_numpy(bank_v).cuda() * 67
n_gpu = torch.from_numpy(bank_n).cuda()
offset = torch.arange(64, device='cuda') - 31.5
dy, dx = torch.meshgrid(offset, offset, indexing='ij')
axial = torch.linspace(-50, 50, 9, device='cuda')
weights = torch.tensor([1] + [2] * 7 + [1], device='cuda', dtype=torch.float32) / 16

with torch.inference_mode():
    for first in range(0, len(features), 128):
        last = min(first + 128, len(features))
        row = torch.arange(first, last, device='cuda')
        point = position_gpu[row // 512]
        orientation = row % 512
        normal = n_gpu[orientation]
        surface = (point[:, None, None] + dx[None, :, :, None] * u_gpu[orientation, None, None]
                   + dy[None, :, :, None] * v_gpu[orientation, None, None])
        coordinates = surface[:, None] + axial[None, :, None, None, None] * normal[:, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights[None].expand(len(row), -1))
        support = rendered[:, 1:2].clamp(0, 1)
        patch = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        descriptor = F.normalize(model.shared(model.atlas_stem(patch)), dim=-1)
        features[first:last] = descriptor.cpu().numpy().astype('float16')
        if last % 131072 == 0 or last == len(features):
            features.flush()
            print(json.dumps({'atlas_patches': last, 'total': len(features),
                              'seconds': time.perf_counter() - started}), flush=True)
features.flush()
del features
result = {name: hashlib.file_digest((out / name).open('rb'), 'sha256').hexdigest()
          for name in ('config.json', 'positions_um.npy', 'orientation_u.npy',
                       'orientation_v.npy', 'orientation_n.npy', 'features.npy')}
(out / 'completed.json').write_text(json.dumps({'files_sha256': result,
    'atlas_patches': len(positions) * len(bank_u), 'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'atlas_patches': len(positions) * len(bank_u),
                  'seconds': time.perf_counter() - started}), flush=True)
