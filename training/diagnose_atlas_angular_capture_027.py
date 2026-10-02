"""Frozen angular tolerance of 025 descriptors at known CCF point locations."""
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
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/atlas_angular_capture_027'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
model = AtlasOrientedPatch025().cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 3000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotate(vector, axis, degrees):
    radians = math.radians(degrees)
    return (vector * math.cos(radians) + torch.cross(axis, vector, dim=-1) * math.sin(radians)
            + axis * (axis * vector).sum() * (1 - math.cos(radians)))


conditions = [('control', 'control', 0, 'none', 0)]
for degrees in (5, 10, 20, 40):
    for axis in ('u', 'v'):
        for sign in (-1, 1):
            conditions.append((f'tilt_{axis}_{sign * degrees:+d}', 'tilt', degrees, axis, sign))
for degrees in (15, 30, 60, 90):
    for sign in (-1, 1):
        conditions.append((f'roll_{sign * degrees:+d}', 'roll', degrees, 'normal', sign))
out.mkdir(parents=True, exist_ok=False)
config = {'checkpoint_sha256': sha(checkpoint),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('diagnose_atlas_angular_capture_027.py',
                             'atlas_oriented_patch_025.py',
                             'arbitrary_plane_full_frame_primitives.py')},
          'cases': len(records), 'points_per_case': 32, 'conditions': conditions,
          'center': 'oracle true CCF point location; diagnostic only',
          'finite_thickness': True, 'weights_updated': False,
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
offset = torch.arange(64, device='cuda') - 31.5
dy, dx = torch.meshgrid(offset, offset, indexing='ij')
started = time.perf_counter()
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            xy = np.stack((chosen // 256, chosen % 256), -1)
            truth = torch.from_numpy(arrays['target_centre_um'][xy[:, 0], xy[:, 1]].copy()).cuda()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflected = bool(arrays['reflection'])
            axial_offsets = torch.from_numpy(arrays['offsets_um'].copy()).cuda()
            axial_weights = torch.from_numpy(arrays['weights'].copy()).cuda()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                torch.from_numpy(xy.copy()).cuda().float())
        query_feature = F.normalize(model.shared(model.image_stem(query)), dim=-1)
        _, frame, basis = full_frame_state_to_components(state)
        edges = (frame[:, :, :2] @ basis)[0] / 256
        edge_u = edges[:, 0] * (-1 if reflected else 1)
        edge_v = edges[:, 1]
        normal = F.normalize(torch.cross(edge_u, edge_v, dim=-1), dim=-1)
        axis_u, axis_v = F.normalize(edge_u, dim=-1), F.normalize(edge_v, dim=-1)
        for name, family, degrees, axis, sign in conditions:
            rotation_axis = {'u': axis_u, 'v': axis_v, 'normal': normal}.get(axis, normal)
            tilted_u = rotate(edge_u, rotation_axis, sign * degrees)
            tilted_v = rotate(edge_v, rotation_axis, sign * degrees)
            tilted_n = rotate(normal, rotation_axis, sign * degrees)
            surface = (truth[:, None, None] + dx[None, :, :, None] * tilted_u
                       + dy[None, :, :, None] * tilted_v)
            coordinates = surface[:, None] + axial_offsets[None, :, None, None, None] * tilted_n
            rendered = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.),
                axial_weights[None].expand(32, -1))
            support = rendered[:, 1:2].clamp(0, 1)
            atlas_patch = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
            feature = F.normalize(model.shared(model.atlas_stem(atlas_patch)), dim=-1)
            similarity = query_feature @ feature.T
            predicted = similarity.argmax(-1)
            distance = (truth[predicted] - truth).norm(dim=-1)
            diagonal = similarity.diagonal()
            wrong = similarity.masked_fill(torch.eye(32, device='cuda', dtype=torch.bool), -1e4).max(-1).values
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
            row.update(condition=name, family=family, degrees=degrees, axis=axis, sign=sign,
                       top1_exact=float((predicted == torch.arange(32, device='cuda')).float().mean()),
                       top1_500um=float((distance < 500).float().mean()),
                       mean_margin=float((diagonal - wrong).mean()),
                       mean_correct_similarity=float(diagonal.mean()))
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summaries = []
for name, family, degrees, axis, sign in conditions:
    selected = [row for row in rows if row['condition'] == name]
    result = {'condition': name, 'family': family, 'degrees': degrees, 'axis': axis, 'sign': sign,
              'sections': len(selected)}
    for field in ('top1_exact', 'top1_500um', 'mean_margin', 'mean_correct_similarity'):
        result[field] = float(np.mean([np.mean([row[field] for row in selected
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    result['mode_top1_500um'] = {mode: float(np.mean([
        np.mean([row['top1_500um'] for row in selected
                 if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in selected)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    summaries.append(result)
chosen = [row for row in summaries if (row['family'], row['degrees']) in (('tilt', 10), ('roll', 15))]
gate = all(np.mean([row['top1_500um'] for row in chosen if row['family'] == family]) >= .4
           for family in ('tilt', 'roll'))
gate &= all(np.mean([row['mode_top1_500um'][mode] for row in chosen
                     if row['family'] == family]) >= .5 *
            np.mean([row['top1_500um'] for row in chosen if row['family'] == family])
            for mode in ('raw', 'exact_black', 'imperfect_brush') for family in ('tilt', 'roll'))
(out / 'summary.json').write_text(json.dumps({'sections': len(records), 'synthetic_subjects': len(subjects),
    'rows': len(rows), 'conditions': summaries, 'coarse_search_plausible_gate': bool(gate),
    'scope': 'oracle true CCF point center; not unknown-plane localization',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'coarse_search_plausible_gate': bool(gate),
                  'seconds': time.perf_counter() - started}), flush=True)
