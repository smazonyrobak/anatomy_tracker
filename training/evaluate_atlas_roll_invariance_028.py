"""Fixed DEV comparison of roll augmentation with frozen 025 descriptors."""
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
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

run = root / 'runs/atlas_roll_invariance_028'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_roll_invariance_028_development_eval'
prior_plane = root / 'runs/atlas_oriented_patch_025_development_eval/summary.json'
prior_angle = root / 'runs/atlas_angular_capture_027/summary.json'
steps = (0, 1000, 2000, 3000)
conditions = [('control', 'none', 0)] + [(f'tilt_{axis}_{sign * 10:+d}', axis, sign * 10)
    for axis in ('u', 'v') for sign in (-1, 1)] + [(f'roll_{sign * 30:+d}', 'normal', sign * 30)
    for sign in (-1, 1)]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 3000
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotate(vector, axis, degrees):
    radians = math.radians(degrees)
    return (vector * math.cos(radians) + torch.cross(axis, vector, dim=-1) * math.sin(radians)
            + axis * (axis * vector).sum() * (1 - math.cos(radians)))


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
prior_plane_summary = json.loads(prior_plane.read_text())
prior_angle_summary = json.loads(prior_angle.read_text())
baseline_roll30 = float(np.mean([row['top1_500um'] for row in prior_angle_summary['conditions']
                                 if row['family'] == 'roll' and row['degrees'] == 30]))
baseline_tilt10 = float(np.mean([row['top1_500um'] for row in prior_angle_summary['conditions']
                                 if row['family'] == 'tilt' and row['degrees'] == 10]))
baseline_plane = next(row for row in prior_plane_summary['checkpoints'] if row['step'] == 3000)
out.mkdir(parents=True, exist_ok=False)
config = {'checkpoints_sha256': checkpoints, 'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'baseline_plane_summary_sha256': sha(prior_plane),
          'baseline_angle_summary_sha256': sha(prior_angle),
          'baseline_025_plane_recall1_500um': baseline_plane['recall1_500um'],
          'baseline_025_roll30_top1_500um': baseline_roll30,
          'baseline_025_tilt10_top1_500um': baseline_tilt10,
          'conditions': conditions, 'query_points_per_section': 32,
          'atlas_plane_candidates': 288, 'oracle_truth_centers': True,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_atlas_roll_invariance_028.py',
                             'atlas_oriented_patch_025.py',
                             'arbitrary_plane_full_frame_primitives.py')},
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = np.arange(8, 256, 16)
gy, gx = np.meshgrid(axis, axis, indexing='ij')
regular = np.stack((gy.flatten(), gx.flatten()), -1)
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
            bank_xy = np.concatenate((regular, xy))
            truth = torch.from_numpy(arrays['target_centre_um'][xy[:, 0], xy[:, 1]].copy()).cuda()
            atlas_ccf = torch.from_numpy(arrays['target_centre_um'][bank_xy[:, 0],
                                                                  bank_xy[:, 1]].copy()).cuda()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflected = bool(arrays['reflection'])
            axial_offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            axial_weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        query = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                torch.from_numpy(xy.copy()).cuda().float())
        atlas_image = atlas_surface_image(atlas, centre, state, axial_offsets, axial_weights)
        bank_xy_gpu = torch.from_numpy(bank_xy.copy()).cuda().float()
        oracle_patches = torch.cat([extract_patches(atlas_image,
            torch.zeros(min(64, len(bank_xy) - start), device='cuda', dtype=torch.long),
            bank_xy_gpu[start:start + 64]) for start in range(0, len(bank_xy), 64)])
        _, frame, basis = full_frame_state_to_components(state)
        edges = (frame[:, :, :2] @ basis)[0] / 256
        edge_u = edges[:, 0] * (-1 if reflected else 1)
        edge_v = edges[:, 1]
        normal = F.normalize(torch.cross(edge_u, edge_v, dim=-1), dim=-1)
        axes = {'u': F.normalize(edge_u, dim=-1), 'v': F.normalize(edge_v, dim=-1),
                'normal': normal, 'none': normal}
        angular_patches = {}
        for name, axis_name, degrees in conditions:
            rotation_axis = axes[axis_name]
            tilted_u = rotate(edge_u, rotation_axis, degrees)
            tilted_v = rotate(edge_v, rotation_axis, degrees)
            tilted_n = rotate(normal, rotation_axis, degrees)
            surface = truth[:, None, None] + dx[None, :, :, None] * tilted_u + dy[None, :, :, None] * tilted_v
            coordinates = surface[:, None] + axial_offsets[:, :, None, None, None] * tilted_n
            rendered = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.), axial_weights.expand(32, -1))
            support = rendered[:, 1:2].clamp(0, 1)
            angular_patches[name] = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        for step, model in zip(steps, models):
            query_feature = F.normalize(model.shared(model.image_stem(query)), dim=-1)
            bank_feature = torch.cat([F.normalize(model.shared(model.atlas_stem(
                oracle_patches[start:start + 64])), dim=-1)
                for start in range(0, len(oracle_patches), 64)])
            top = (query_feature @ bank_feature.T).topk(16, -1).indices
            distance = (atlas_ccf[top] - truth[:, None]).norm(dim=-1)
            angular = {}
            for name, _, _ in conditions:
                feature = F.normalize(model.shared(model.atlas_stem(angular_patches[name])), dim=-1)
                choice = (query_feature @ feature.T).argmax(-1)
                angular[name] = float(((truth[choice] - truth).norm(dim=-1) < 500).float().mean())
            row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
            row.update(step=step, oracle_recall1_500um=float((distance[:, 0] < 500).float().mean()),
                       oracle_recall16_500um=float((distance.min(-1).values < 500).float().mean()),
                       angular_top1_500um=angular)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = []
for step in steps:
    selected = [row for row in rows if row['step'] == step]
    result = {'step': step, 'sections': len(selected), 'synthetic_subjects': len(subjects)}
    for field in ('oracle_recall1_500um', 'oracle_recall16_500um'):
        result[field] = float(np.mean([np.mean([row[field] for row in selected
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
    result['mode_oracle_recall1_500um'] = {mode: float(np.mean([
        np.mean([row['oracle_recall1_500um'] for row in selected
                 if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
        for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                      row['appearance_mode'] == mode for row in selected)]))
        for mode in ('raw', 'exact_black', 'imperfect_brush')}
    result['angular_top1_500um'] = {name: float(np.mean([
        np.mean([row['angular_top1_500um'][name] for row in selected
                 if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
        for name, _, _ in conditions}
    result['roll30_top1_500um'] = float(np.mean([result['angular_top1_500um'][name]
                                                for name in ('roll_-30', 'roll_+30')]))
    result['tilt10_top1_500um'] = float(np.mean([result['angular_top1_500um'][name]
                                                for name in ('tilt_u_-10', 'tilt_u_+10',
                                                             'tilt_v_-10', 'tilt_v_+10')]))
    result['gate'] = bool(step > 0 and min(result['mode_oracle_recall1_500um'].values()) >= .8 and
        result['roll30_top1_500um'] >= baseline_roll30 + .2 and
        result['tilt10_top1_500um'] >= baseline_tilt10 - .1)
    summary.append(result)
(out / 'summary.json').write_text(json.dumps({'rows': len(rows), 'checkpoints': summary,
    'matched_change_gate': any(row['gate'] for row in summary[1:]),
    'baseline_025': {'oracle_recall1_500um': baseline_plane['recall1_500um'],
                     'roll30_top1_500um': baseline_roll30, 'tilt10_top1_500um': baseline_tilt10},
    'scope': 'oracle-known atlas plane or CCF point; no global pose claim',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'summaries': summary,
                  'seconds': time.perf_counter() - started}), flush=True)
