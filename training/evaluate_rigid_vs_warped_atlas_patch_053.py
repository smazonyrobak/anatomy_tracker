"""Matched numeric patch retrieval at oracle-warped versus rigid true atlas planes."""
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint_path = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/rigid_vs_warped_atlas_patch_053_development_eval'
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
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185 and len({row['animal_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
config = {'checkpoint_sha256': sha(checkpoint_path), 'panel_records_sha256': sha(panel / 'records.jsonl'),
          'query_selection': 'same 32 fixed seeded GT-valid points as 025; not inference',
          'atlas_candidates': '256 regular plus 32 exact query pixels; true known pose in both arms',
          'arms': ['oracle_warped_surface', 'rigid_true_plane'],
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_rigid_vs_warped_atlas_patch_053.py', 'atlas_oriented_patch_025.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = np.arange(8, 256, 16)
gy, gx = np.meshgrid(axis, axis, indexing='ij')
regular = np.stack((gy.flatten(), gx.flatten()), -1)
pixel = torch.arange(256, device='cuda', dtype=torch.float32) / 256
yy, xx = torch.meshgrid(pixel, pixel, indexing='ij')
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            valid = arrays['valid_mask'].reshape(-1)
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(valid), 32, replace=False)
            query_yx = np.stack((chosen // 256, chosen % 256), -1)
            atlas_yx = np.concatenate((regular, query_yx))
            truth = torch.from_numpy(arrays['target_centre_um'][query_yx[:, 0], query_yx[:, 1]].copy()).cuda()
            inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            warped = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        centre, frame, basis = full_frame_state_to_components(state)
        sx = 255 / 256 - xx if reflection else xx
        chart = torch.stack((sx, yy), -1)[None]
        rigid = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        query_gpu = torch.from_numpy(query_yx.copy()).cuda().float()
        atlas_gpu = torch.from_numpy(atlas_yx.copy()).cuda().float()
        query_patch = extract_patches(inputs, torch.zeros(32, device='cuda', dtype=torch.long), query_gpu)
        query_descriptor = F.normalize(model.shared(model.image_stem(query_patch)), dim=-1)
        displacement = (warped[0, query_yx[:, 0], query_yx[:, 1]] -
                        rigid[0, query_yx[:, 0], query_yx[:, 1]]).norm(dim=-1)
        for name, surface in (('oracle_warped_surface', warped), ('rigid_true_plane', rigid)):
            rendered = atlas_surface_image(atlas, surface, state, offsets, weights)
            patches = torch.cat([extract_patches(rendered,
                torch.zeros(min(64, len(atlas_yx) - first), device='cuda', dtype=torch.long),
                atlas_gpu[first:first + 64]) for first in range(0, len(atlas_yx), 64)])
            key = torch.cat([F.normalize(model.shared(model.atlas_stem(patches[first:first + 64])), dim=-1)
                             for first in range(0, len(atlas_yx), 64)])
            score = query_descriptor @ key.T
            top = score.topk(16, -1).indices
            atlas_ccf = surface[0, atlas_yx[:, 0], atlas_yx[:, 1]]
            top_distance = (atlas_ccf[top] - truth[:, None]).norm(dim=-1)
            exact_score = score[torch.arange(32, device='cuda'), torch.arange(32, device='cuda') + 256]
            exact_rank = (score > exact_score[:, None]).sum(-1) + 1
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                               'section_id', 'synthetic_subject_plan_id',
                                               'appearance_mode', 'sha256')}
            row.update(arm=name, query_pixel_yx=query_yx.tolist(),
                       top16_candidate_index=top.cpu().tolist(),
                       top1_distance_um=top_distance[:, 0].cpu().tolist(),
                       min_top16_distance_um=top_distance.min(-1).values.cpu().tolist(),
                       exact_candidate_rank=exact_rank.cpu().tolist(),
                       exact_candidate_cosine=exact_score.cpu().tolist(),
                       warped_to_rigid_displacement_um=displacement.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records)}), flush=True)

summary = {'sections': 185, 'synthetic_subjects': 8}
for arm in config['arms']:
    group = [row for row in rows if row['arm'] == arm]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in group})
    metrics = {}
    for field, metric in (('top1_distance_um', 'top1_500um'),
                          ('min_top16_distance_um', 'top16_500um'),
                          ('exact_candidate_rank', 'exact_candidate_top1')):
        threshold = 1 if field == 'exact_candidate_rank' else 500
        metrics[metric] = float(np.mean([np.mean([np.mean(np.asarray(row[field]) <= threshold)
            for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    for field, metric in (('exact_candidate_cosine', 'exact_candidate_cosine'),
                          ('warped_to_rigid_displacement_um', 'warped_to_rigid_displacement_um')):
        metrics[metric] = float(np.mean([np.mean([np.mean(row[field]) for row in group
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    metrics['mode_top1_500um'] = {mode: float(np.mean([
        np.mean([np.mean(np.asarray(row['top1_distance_um']) < 500) for row in group
                 if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in group)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summary[arm] = metrics
summary['rigid_top1_drop_percentage_points'] = 100 * (
    summary['oracle_warped_surface']['top1_500um'] - summary['rigid_true_plane']['top1_500um'])
summary['geometry_mismatch_major'] = summary['rigid_top1_drop_percentage_points'] >= 20
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'geometry_mismatch_major': summary['geometry_mismatch_major']}), flush=True)
