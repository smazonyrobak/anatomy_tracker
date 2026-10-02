"""Oracle-plane cross-modal patch diagnostic on disjoint synthetic DEV identities."""
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
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

run = root / 'runs/atlas_oriented_patch_025'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_oriented_patch_025_development_eval'
steps = (0, 1000, 2000, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 3000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [record for record in records if record['eligible']]
assert len(records) == 256 and len(eligible) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
out.mkdir(parents=True, exist_ok=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


models, checkpoints = [], {}
for step in steps:
    path = run / f'patch_step_{step:05d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = AtlasOrientedPatch025().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    models.append(model)
    checkpoints[str(step)] = sha(path)
del checkpoint
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'checkpoints_sha256': checkpoints,
          'query_points_per_section': 32,
          'atlas_bank': 'known finite-thickness atlas surface, regular 16x16 grid plus exact 32 query pixels',
          'atlas_candidates_per_section': 288,
          'query_selection': 'fixed seeded GT-valid diagnostic; not inference',
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_atlas_oriented_patch_025.py',
                             'atlas_oriented_patch_025.py',
                             'arbitrary_plane_full_frame_primitives.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = np.arange(8, 256, 16)
gy, gx = np.meshgrid(axis, axis, indexing='ij')
regular = np.stack((gy.flatten(), gx.flatten()), -1)
started = time.perf_counter()
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(eligible, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            valid = arrays['valid_mask'].reshape(-1)
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(valid), 32, replace=False)
            query_xy = np.stack((chosen // 256, chosen % 256), -1)
            atlas_xy = np.concatenate((regular, query_xy))
            truth = arrays['target_centre_um'][query_xy[:, 0], query_xy[:, 1]].copy()
            atlas_ccf = arrays['target_centre_um'][atlas_xy[:, 0], atlas_xy[:, 1]].copy()
            inputs = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        rendered = atlas_surface_image(atlas, centre, state, offsets, weights)
        query_patch = extract_patches(inputs, torch.zeros(32, device='cuda', dtype=torch.long),
                                      torch.from_numpy(query_xy.copy()).cuda().float())
        atlas_xy_gpu = torch.from_numpy(atlas_xy.copy()).cuda().float()
        atlas_patch = torch.cat([extract_patches(rendered,
            torch.zeros(min(64, len(atlas_xy) - start), device='cuda', dtype=torch.long),
            atlas_xy_gpu[start:start + 64]) for start in range(0, len(atlas_xy), 64)])
        atlas_ccf_gpu = torch.from_numpy(atlas_ccf).cuda()
        truth_gpu = torch.from_numpy(truth).cuda()
        for step, model in zip(steps, models):
            query_descriptor = F.normalize(model.shared(model.image_stem(query_patch)), dim=-1)
            keys = torch.cat([F.normalize(model.shared(model.atlas_stem(atlas_patch[start:start + 64])), dim=-1)
                              for start in range(0, len(atlas_xy), 64)])
            score = query_descriptor @ keys.T
            top = score.topk(16, -1).indices
            distance = (atlas_ccf_gpu[top] - truth_gpu[:, None]).norm(dim=-1).min(-1).values
            best = (atlas_ccf_gpu[top[:, 0]] - truth_gpu).norm(dim=-1)
            exact_rank = (score > score.gather(1,
                (torch.arange(32, device='cuda') + 256)[:, None])).sum(-1) + 1
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
            row.update(step=step, query_pixel_yx=query_xy.tolist(), true_ccf_um=truth.tolist(),
                       candidate_pixel_yx=atlas_xy.tolist(), candidate_ccf_um=atlas_ccf.tolist(),
                       top16_indices=top.cpu().tolist(), best_um=best.cpu().tolist(),
                       min_top16_um=distance.cpu().tolist(), exact_positive_rank=exact_rank.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
summary = []
for step in steps:
    selected = [row for row in rows if row['step'] == step]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in selected})
    result = {'step': step, 'sections': len(selected), 'synthetic_subjects': len(subjects)}
    for threshold in (250, 500, 1000):
        for field, name in (('best_um', 'recall1'), ('min_top16_um', 'recall16')):
            result[f'{name}_{threshold}um'] = float(np.mean([
                np.mean([np.mean(np.asarray(row[field]) < threshold) for row in selected
                         if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    result['exact_positive_top1'] = float(np.mean([
        np.mean([np.mean(np.asarray(row['exact_positive_rank']) == 1) for row in selected
                 if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    result['mode_recall16_500um'] = {mode: float(np.mean([
        np.mean([np.mean(np.asarray(row['min_top16_um']) < 500) for row in selected
                 if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in selected)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summary.append(result)
gate = any(row['recall1_500um'] >= .25 and row['recall16_500um'] >= .70 and
           min(row['mode_recall16_500um'].values()) >= .5 * row['recall16_500um']
           for row in summary[1:])
(out / 'summary.json').write_text(json.dumps({'rows': len(rows), 'checkpoints': summary,
    'necessary_gate': gate,
    'scope': 'truth-known atlas plane and appended exact positives; no unknown-plane localization claim',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'summaries': summary,
                  'seconds': time.perf_counter() - started}), flush=True)
