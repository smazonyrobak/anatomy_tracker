"""Matched numerical 2×2×2 atlas patch geometry factorial."""
import hashlib
import itertools
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
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
bank_dir = root / 'runs/atlas_global_patch_030'
checkpoint_path = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
ladder_dir = root / 'runs/atlas_bank_geometry_ladder_056_development_eval'
out = root / 'runs/atlas_key_geometry_factors_057_development_eval'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 3000 and not checkpoint['calibrated']
model = AtlasOrientedPatch025().cuda().eval()
model.load_state_dict(checkpoint['model'])
model.requires_grad_(False)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
bank_u = torch.from_numpy(np.load(bank_dir / 'orientation_u.npy', allow_pickle=False)).cuda()
bank_v = torch.from_numpy(np.load(bank_dir / 'orientation_v.npy', allow_pickle=False)).cuda()
bank_n = torch.from_numpy(np.load(bank_dir / 'orientation_n.npy', allow_pickle=False)).cuda()
ladder = json.loads((ladder_dir / 'summary.json').read_text())
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185 and len({row['animal_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
conditions = list(itertools.product(('exact', 'bank'), ('exact', 'fixed'), ('section', 'bank')))
names = ['_'.join(parts) for parts in conditions]
config = {'checkpoint_sha256': sha(checkpoint_path),
          'bank_completed_sha256': sha(bank_dir / 'completed.json'),
          'ladder_summary_sha256': sha(ladder_dir / 'summary.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'conditions': names, 'query_selection': 'same 32 GT-valid points as 056; not inference',
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_atlas_key_geometry_factors_057.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
offset = torch.arange(64, device='cuda', dtype=torch.float32) - 31.5
dy, dx = torch.meshgrid(offset, offset, indexing='ij')
fixed_offsets = torch.linspace(-50, 50, 9, device='cuda')
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
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            section_offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            section_weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        query_gpu = torch.from_numpy(query_yx.copy()).cuda().float()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long), query_gpu)
        _, frame, basis = full_frame_state_to_components(state)
        edges = frame[0, :, :2] @ basis[0]
        u = edges[:, 0][None].expand(32, -1).clone() / 256
        v = edges[:, 1][None].expand(32, -1).clone() / 256
        if reflection:
            u = -u
        parity_flip = torch.cross(u, v, dim=-1)[:, 2] < 0
        u = torch.where(parity_flip[:, None], -u, u)
        query = torch.where(parity_flip[:, None, None, None], query.flip(-1), query)
        descriptor = F.normalize(model.shared(model.image_stem(query)), dim=-1)
        length_u, length_v = u.norm(dim=-1), v.norm(dim=-1)
        unit_u, unit_v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
        normal_exact = F.normalize(torch.cross(unit_u, unit_v, dim=-1), dim=-1)
        orientation = (unit_u @ bank_u.T + unit_v @ bank_v.T).argmax(-1)
        scores = {}
        for (frame_kind, scale_kind, psf_kind), name in zip(conditions, names):
            base_u, base_v, normal = (unit_u, unit_v, normal_exact) if frame_kind == 'exact' else (
                bank_u[orientation], bank_v[orientation], bank_n[orientation])
            sx, sy = (length_u, length_v) if scale_kind == 'exact' else (
                length_u.new_full((32,), 67), length_v.new_full((32,), 67))
            surface = (truth[:, None, None]
                       + dx[None, :, :, None] * (sx[:, None] * base_u)[:, None, None]
                       + dy[None, :, :, None] * (sy[:, None] * base_v)[:, None, None])
            axial, weights = (section_offsets[0], section_weights.expand(32, -1)) \
                if psf_kind == 'section' else (fixed_offsets, fixed_weights[None].expand(32, -1))
            coordinates = surface[:, None] + axial[None, :, None, None, None] * normal[
                :, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
                (0., 0., 0.), (25., 25., 25.), weights)
            support = rendered[:, 1:2].clamp(0, 1)
            patch = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
            key = F.normalize(model.shared(model.atlas_stem(patch)), dim=-1)
            scores[name] = (descriptor * key).sum(-1).cpu().tolist()
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                           'section_id', 'synthetic_subject_plan_id',
                                           'appearance_mode', 'sha256')}
        row.update(query_pixel_yx=query_yx.tolist(), cosine=scores)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records)}), flush=True)

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
means = {name: float(np.mean([np.mean([np.mean(row['cosine'][name]) for row in rows
    if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])) for name in names}
assert abs(means['exact_exact_section'] - ladder['identity_equal_mean']['B_rigid_true_geometry_cosine']) < .003
assert abs(means['bank_fixed_bank'] - ladder['identity_equal_mean'][
    'C_true_point_bank_geometry_nearest_cosine']) < .003
summary = {'sections': 185, 'synthetic_subjects': 8,
           'identity_equal_mean_cosine': means,
           'one_factor_loss_from_exact': {
               'frame_including_shear': means['exact_exact_section'] - means['bank_exact_section'],
               'scale': means['exact_exact_section'] - means['exact_fixed_section'],
               'psf': means['exact_exact_section'] - means['exact_exact_bank']},
           'mode_exact_vs_bank_cosine': {mode: {name: float(np.mean([np.mean([
               np.mean(row['cosine'][name]) for row in rows
               if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
               for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                   row['appearance_mode'] == mode for row in rows)]))
               for name in ('exact_exact_section', 'bank_fixed_bank')}
               for mode in ('raw', 'exact_black', 'imperfect_brush')},
           'calibrated': False, 'public_benchmark_used': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(rows)}), flush=True)
