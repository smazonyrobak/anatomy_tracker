"""Read-only coverage ceiling for a coarse oriented atlas-patch search."""
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
atlas, _ = _decode_and_preprocess_allen_v6()
support = F.avg_pool3d(torch.from_numpy(atlas[1:2, None]).cuda(), 4, 4)[0, 0].cpu().numpy()
grid = np.argwhere(support[2::5, 2::5, 2::5] > .5)
points = (grid * 5 + 2.375) * 100
tree = cKDTree(points)

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
          + np.sin(roll)[None, :, None] * v[:, None]).reshape(-1, 3)
bank_v = (-np.sin(roll)[None, :, None] * u[:, None]
          + np.cos(roll)[None, :, None] * v[:, None]).reshape(-1, 3)
bank_n = np.repeat(normals, len(roll), axis=0)

distances, angles, scales = [], [], []
for record in records:
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        rng = np.random.default_rng(int(record['sha256'][:16], 16))
        selected = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
        xy = np.stack((selected // 256, selected % 256), -1)
        truth = arrays['target_centre_um'][xy[:, 0], xy[:, 1]]
        _, frame, basis = full_frame_state_to_components(torch.from_numpy(arrays['target_state'][None].copy()))
        edges = ((frame[:, :, :2] @ basis)[0] / 256).numpy()
        edge_u = edges[:, 0] * (-1 if bool(arrays['reflection']) else 1)
        edge_v = edges[:, 1]
    distances.extend(tree.query(truth)[0])
    scales.append(np.linalg.norm(edges, axis=0))
    target_u = edge_u / np.linalg.norm(edge_u)
    target_v = edge_v - target_u * np.dot(target_u, edge_v)
    target_v /= np.linalg.norm(target_v)
    target_n = np.cross(target_u, target_v)
    trace = (bank_u @ target_u + bank_v @ target_v + bank_n @ target_n)
    reflected_trace = (-bank_u @ target_u + bank_v @ target_v - bank_n @ target_n)
    angles.append(np.degrees(np.arccos(np.clip((max(trace.max(), reflected_trace.max()) - 1) / 2, -1, 1))))

distances = np.asarray(distances)
angles = np.asarray(angles)
scales = np.asarray(scales)
print(json.dumps({'sections': len(records), 'atlas_grid_points': len(points),
                  'orientations_per_point': len(bank_u), 'total_atlas_patches': len(points) * len(bank_u),
                  'nearest_point_within_250um': float(np.mean(distances < 250)),
                  'nearest_point_within_500um': float(np.mean(distances < 500)),
                  'nearest_point_within_1000um': float(np.mean(distances < 1000)),
                  'nearest_orientation_under_10deg': float(np.mean(angles < 10)),
                  'nearest_orientation_under_20deg': float(np.mean(angles < 20)),
                  'nearest_orientation_deg_median': float(np.median(angles)),
                  'pixel_scale_um_median_yx': np.median(scales, axis=0).tolist(),
                  'pixel_scale_um_q10_yx': np.quantile(scales, .1, axis=0).tolist(),
                  'pixel_scale_um_q90_yx': np.quantile(scales, .9, axis=0).tolist()}))
