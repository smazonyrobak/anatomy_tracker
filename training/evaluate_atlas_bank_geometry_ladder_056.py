"""Numerical atlas-patch similarity ladder; no images or models are produced."""
import hashlib
import json
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
bank_dir = root / 'runs/atlas_global_patch_030'
checkpoint_path = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/atlas_bank_geometry_ladder_056_development_eval'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
bank_receipt = json.loads((bank_dir / 'completed.json').read_text())
for name, expected in bank_receipt['files_sha256'].items():
    assert sha(bank_dir / name) == expected
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 3000 and not checkpoint['calibrated']
model = AtlasOrientedPatch025().cuda().eval()
model.load_state_dict(checkpoint['model'])
model.requires_grad_(False)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
bank = torch.from_numpy(np.load(bank_dir / 'features.npy', mmap_mode='r', allow_pickle=False)).cuda()
positions = torch.from_numpy(np.load(bank_dir / 'positions_um.npy', allow_pickle=False)).cuda()
bank_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
bank_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
bank_n = torch.from_numpy(np.load(bank_dir / 'orientation_n.npy', allow_pickle=False)).cuda()
assert bank.shape == (2061824, 128) and positions.shape == (4027, 3)
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185 and len({row['animal_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
config = {'checkpoint_sha256': sha(checkpoint_path),
          'bank_completed_sha256': sha(bank_dir / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'query_selection': 'same 32 seeded GT-valid pixels as 025/030/053; not inference',
          'atlas_stages': ['A_oracle_warped', 'B_rigid_true_geometry',
                           'C_true_point_bank_geometry', 'D_nearest_bank_key'],
          'orientation_neighbours': 4, 'bank_scale_um_per_pixel': 67,
          'bank_psf': [-50, 50, 9],
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_atlas_bank_geometry_ladder_056.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
offset = torch.arange(64, device='cuda', dtype=torch.float32) - 31.5
dy, dx = torch.meshgrid(offset, offset, indexing='ij')
axial = torch.linspace(-50, 50, 9, device='cuda')
fixed_weights = torch.tensor([1] + [2] * 7 + [1], device='cuda', dtype=torch.float32) / 16
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_yx = np.stack((chosen // 256, chosen % 256), -1)
            truth = torch.from_numpy(arrays['target_centre_um'][query_yx[:, 0], query_yx[:, 1]].copy()).cuda()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            warped = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        query_gpu = torch.from_numpy(query_yx.copy()).cuda().float()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long), query_gpu)
        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[0, :, :2] @ basis[0]
        u = edges[:, 0][None].expand(32, -1).clone()
        v = edges[:, 1][None].expand(32, -1).clone()
        if reflection:
            u = -u
        parity_flip = torch.cross(u, v, dim=-1)[:, 2] < 0
        u = torch.where(parity_flip[:, None], -u, u) / 256
        v = v / 256
        query = torch.where(parity_flip[:, None, None, None], query.flip(-1), query)
        descriptor = F.normalize(model.shared(model.image_stem(query)), dim=-1)

        oracle_image = atlas_surface_image(atlas, warped, state, offsets, weights)
        oracle_patch = extract_patches(oracle_image,
            torch.zeros(32, device='cuda', dtype=torch.long), query_gpu)
        oracle_patch = torch.where(parity_flip[:, None, None, None], oracle_patch.flip(-1), oracle_patch)
        oracle_key = F.normalize(model.shared(model.atlas_stem(oracle_patch)), dim=-1)
        score_a = (descriptor * oracle_key).sum(-1)

        exact_surface = (truth[:, None, None]
                         + dx[None, :, :, None] * u[:, None, None]
                         + dy[None, :, :, None] * v[:, None, None])
        normal = frame[0, :, 2]
        exact_coordinates = exact_surface[:, None] + offsets[:, :, None, None, None] * normal
        exact_render = render_finite_thickness_coordinate_grid(atlas, exact_coordinates,
            (0., 0., 0.), (25., 25., 25.), weights.expand(32, -1))
        exact_support = exact_render[:, 1:2].clamp(0, 1)
        exact_patch = torch.cat((exact_render[:, :1] / exact_support.clamp_min(1e-4),
                                 exact_support), 1)
        exact_key = F.normalize(model.shared(model.atlas_stem(exact_patch)), dim=-1)
        score_b = (descriptor * exact_key).sum(-1)

        unit_u = F.normalize(u, dim=-1)
        orthogonal_v = F.normalize(v - (v * unit_u).sum(-1, keepdim=True) * unit_u, dim=-1)
        orientation = (unit_u @ bank_u.T + F.normalize(v, dim=-1) @ bank_v.T).topk(4, -1).indices
        chosen_u = bank_u[orientation].reshape(-1, 3)
        chosen_v = bank_v[orientation].reshape(-1, 3)
        chosen_n = bank_n[orientation].reshape(-1, 3)
        centre_true = truth[:, None].expand(-1, 4, -1).reshape(-1, 3)
        bank_surface = (centre_true[:, None, None]
                        + 67 * dx[None, :, :, None] * chosen_u[:, None, None]
                        + 67 * dy[None, :, :, None] * chosen_v[:, None, None])
        bank_coordinates = bank_surface[:, None] + axial[None, :, None, None, None] * chosen_n[
            :, None, None, None]
        parts = []
        for first in range(0, len(bank_coordinates), 32):
            coords = bank_coordinates[first:first + 32]
            rendered = render_finite_thickness_coordinate_grid(atlas, coords,
                (0., 0., 0.), (25., 25., 25.), fixed_weights[None].expand(len(coords), -1))
            support = rendered[:, 1:2].clamp(0, 1)
            patch = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
            parts.append(F.normalize(model.shared(model.atlas_stem(patch)), dim=-1))
        bank_key_at_truth = torch.cat(parts).reshape(32, 4, -1)
        score_c = (descriptor[:, None] * bank_key_at_truth).sum(-1)

        near_distance, near_position = torch.cdist(truth, positions).min(-1)
        bank_id = near_position[:, None] * 512 + orientation
        score_d = (descriptor[:, None].half() * bank[bank_id]).sum(-1).float()
        full_score = descriptor.half() @ bank.T
        best_d = score_d.max(-1).values
        rank_d = (full_score > best_d[:, None]).sum(-1) + 1
        nearest_frame = torch.stack((chosen_u.reshape(32, 4, 3)[:, 0],
                                     chosen_v.reshape(32, 4, 3)[:, 0],
                                     chosen_n.reshape(32, 4, 3)[:, 0]), -1)
        exact_frame = torch.stack((unit_u, orthogonal_v,
                                   torch.cross(unit_u, orthogonal_v, dim=-1)), -1)
        trace = (nearest_frame * exact_frame).sum((-2, -1))
        angle = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1)))
        chart = torch.stack((query_gpu[:, 1] / 256, query_gpu[:, 0] / 256), -1)
        if reflection:
            chart[:, 0] = 255 / 256 - chart[:, 0]
        rigid_centre = normalized_raster_to_ccf(centre, frame, basis, chart[None])[0]
        warp_amplitude = (truth - rigid_centre).norm(dim=-1)
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                           'section_id', 'synthetic_subject_plan_id',
                                           'appearance_mode', 'sha256')}
        row.update(query_pixel_yx=query_yx.tolist(),
                   A_oracle_warped_cosine=score_a.cpu().tolist(),
                   B_rigid_true_geometry_cosine=score_b.cpu().tolist(),
                   C_true_point_bank_geometry_nearest_cosine=score_c[:, 0].cpu().tolist(),
                   C_true_point_bank_geometry_best4_cosine=score_c.max(-1).values.cpu().tolist(),
                   D_nearest_bank_key_nearest_cosine=score_d[:, 0].cpu().tolist(),
                   D_nearest_bank_key_best4_cosine=best_d.cpu().tolist(),
                   D_best4_global_rank=rank_d.cpu().tolist(),
                   global_top1_cosine=full_score.max(-1).values.float().cpu().tolist(),
                   nearest_bank_position_um=near_distance.cpu().tolist(),
                   nearest_bank_frame_degrees=angle.cpu().tolist(),
                   observed_pixel_scale_x_um=u.norm(dim=-1).cpu().tolist(),
                   observed_pixel_scale_y_um=v.norm(dim=-1).cpu().tolist(),
                   warped_to_rigid_centre_um=warp_amplitude.cpu().tolist())
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records)}), flush=True)

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
names = ('A_oracle_warped_cosine', 'B_rigid_true_geometry_cosine',
         'C_true_point_bank_geometry_nearest_cosine', 'C_true_point_bank_geometry_best4_cosine',
         'D_nearest_bank_key_nearest_cosine', 'D_nearest_bank_key_best4_cosine',
         'global_top1_cosine', 'nearest_bank_position_um', 'nearest_bank_frame_degrees',
         'observed_pixel_scale_x_um', 'observed_pixel_scale_y_um', 'warped_to_rigid_centre_um')
summary = {'sections': 185, 'synthetic_subjects': 8,
           'identity_equal_mean': {name: float(np.mean([np.mean([np.mean(row[name]) for row in rows
               if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])) for name in names},
           'D_best4_rank_le_16': float(np.mean([np.mean([np.mean(np.asarray(row['D_best4_global_rank']) <= 16)
               for row in rows if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])),
           'D_best4_rank_median_pooled': float(np.median(np.concatenate(
               [row['D_best4_global_rank'] for row in rows]))),
           'mode_D_best4_rank_le_16': {mode: float(np.mean([np.mean([
               np.mean(np.asarray(row['D_best4_global_rank']) <= 16) for row in rows
               if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
               for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                   row['appearance_mode'] == mode for row in rows)]))
               for mode in ('raw', 'exact_black', 'imperfect_brush')},
           'calibrated': False, 'public_benchmark_used': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(rows)}), flush=True)
