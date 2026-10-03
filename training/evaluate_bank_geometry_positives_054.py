"""Rigid-true-plane DEV gate for bank-geometry positive training."""
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

train = root / 'runs/bank_geometry_positives_054_pilot'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/bank_geometry_positives_054_rigid_local_eval'
steps = (0, 1000, 3000, 5000)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
receipt = json.loads((train / 'completed.json').read_text())
assert receipt['batches'] == 5000 and receipt['accepted_synthetic'] == 5000
models = []
for step in steps:
    checkpoint = torch.load(train / f'patch_step_{step:05d}.pt', map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasOrientedPatch025().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    models.append(model)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185 and len({row['animal_id'] for row in records}) == 8
out.mkdir(parents=True, exist_ok=False)
config = {'steps': steps, 'training_completed_sha256': sha(train / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'checkpoint_sha256': {str(step): sha(train / f'patch_step_{step:05d}.pt') for step in steps},
          'query_selection': 'same fixed seeded 32 GT-valid points as 025/053; not inference',
          'atlas_candidates': 'correct rigid true plane, 256 regular plus 32 exact query pixels',
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_bank_geometry_positives_054.py', 'atlas_oriented_patch_025.py')},
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
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_yx = np.stack((chosen // 256, chosen % 256), -1)
            atlas_yx = np.concatenate((regular, query_yx))
            truth = torch.from_numpy(arrays['target_centre_um'][query_yx[:, 0], query_yx[:, 1]].copy()).cuda()
            inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflection = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        centre, frame, basis = full_frame_state_to_components(state)
        chart = torch.stack((255 / 256 - xx if reflection else xx, yy), -1)[None]
        rigid = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        rendered = atlas_surface_image(atlas, rigid, state, offsets, weights)
        query_gpu = torch.from_numpy(query_yx.copy()).cuda().float()
        atlas_gpu = torch.from_numpy(atlas_yx.copy()).cuda().float()
        query_patch = extract_patches(inputs, torch.zeros(32, device='cuda', dtype=torch.long), query_gpu)
        patches = torch.cat([extract_patches(rendered,
            torch.zeros(min(64, len(atlas_yx) - first), device='cuda', dtype=torch.long),
            atlas_gpu[first:first + 64]) for first in range(0, len(atlas_yx), 64)])
        atlas_ccf = rigid[0, atlas_yx[:, 0], atlas_yx[:, 1]]
        for step, model in zip(steps, models):
            query_descriptor = F.normalize(model.shared(model.image_stem(query_patch)), dim=-1)
            key = torch.cat([F.normalize(model.shared(model.atlas_stem(patches[first:first + 64])), dim=-1)
                             for first in range(0, len(atlas_yx), 64)])
            score = query_descriptor @ key.T
            top = score.topk(16, -1).indices
            distance = (atlas_ccf[top] - truth[:, None]).norm(dim=-1)
            exact = score[torch.arange(32, device='cuda'), torch.arange(32, device='cuda') + 256]
            exact_rank = (score > exact[:, None]).sum(-1) + 1
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                               'section_id', 'synthetic_subject_plan_id',
                                               'appearance_mode', 'sha256')}
            row.update(step=step, query_pixel_yx=query_yx.tolist(),
                       top16_candidate_index=top.cpu().tolist(),
                       top1_distance_um=distance[:, 0].cpu().tolist(),
                       min_top16_distance_um=distance.min(-1).values.cpu().tolist(),
                       exact_candidate_rank=exact_rank.cpu().tolist(),
                       exact_candidate_cosine=exact.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(records)}), flush=True)

summary = {'sections': 185, 'synthetic_subjects': 8}
for step in steps:
    group = [row for row in rows if row['step'] == step]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in group})
    recall1 = float(np.mean([np.mean([np.mean(np.asarray(row['top1_distance_um']) < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    recall16 = float(np.mean([np.mean([np.mean(np.asarray(row['min_top16_distance_um']) < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    exact = float(np.mean([np.mean([np.mean(np.asarray(row['exact_candidate_rank']) == 1)
        for row in group if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    modes = {mode: float(np.mean([np.mean([np.mean(np.asarray(row['top1_distance_um']) < 500)
        for row in group if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in group)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summary[str(step)] = {'rigid_local_top1_500um': recall1, 'rigid_local_top16_500um': recall16,
                          'exact_candidate_top1': exact, 'mode_top1_500um': modes,
                          'small_bank_gate': bool(step > 0 and recall1 >= .85 and exact >= .70 and
                                                  min(modes.values()) >= .5 * recall1)}
summary['any_small_bank_gate'] = any(summary[str(step)]['small_bank_gate'] for step in steps[1:])
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'any_small_bank_gate': summary['any_small_bank_gate']}), flush=True)
