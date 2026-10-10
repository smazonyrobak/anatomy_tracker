"""TRAIN-only oracle check of the 150 affine pose fit's fixed ridge."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
out = root / 'runs/coarse_atlas_pose_152_train_fit_ridge'
seed, side, per_base = 20261011152, 256, 32
bases = (60, 61, 62, 63)
penalties = {'fixed_150': (24., 12., 12.), 'center_only': (24., 0., 0.),
             'slope_only': (0., 12., 12.), 'none': (0., 0., 0.)}
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
torch.manual_seed(seed)
context = load_streaming_synthetic_v7_64('cuda')
assert all(context['bases'][i]['lineage']['split'] == 'train' for i in bases)
config = {'seed': seed, 'side': side, 'per_base': per_base, 'bases': bases,
          'penalties': penalties, 'source_sha256': {name: sha(source / name) for name in (
              'diagnose_train_pose_fit_ridge_152.py',
              'arbitrary_plane_panel_geometry_train_151.py',
              'arbitrary_plane_one_shot_slide_artifacts_v3.py',
              'arbitrary_plane_one_shot_slide_artifacts_v4.py',
              'coarse_atlas_pose_150.py')},
          'map_completion_sha256': context['bindings'][str(Path(
              'I:/AnatomyTracker/data/joint_v7_independent_subject_maps_001/completed.json'))],
          'candidate': 'one truth-perturbed 0.35-1.5mm state per independently drawn TRAIN section; both reflection representations',
          'fit': 'true rigid 24x24 atlas targets at oracle-valid image queries; four fixed penalties; native256 rigid mm',
          'selection': '32 eligible one-shot sections per TRAIN base; all attempts logged; no DEV or checkpoints read',
          'calibrated': False, 'public_benchmark_used': False,
          'expert_real_truth_used': False, 'final_animals_used': False,
          'pretrained_weights_used': False}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(config, indent=2))

axis24 = (torch.arange(24, device='cuda') + .5) / 24
qy, qx = torch.meshgrid(axis24, axis24, indexing='ij')
query = torch.stack((qx, qy), -1).reshape(-1, 2)
query_grid = (2 * query - 1).reshape(1, 24, 24, 2)
axis256 = torch.arange(side, device='cuda') / side
ny, nx = torch.meshgrid(axis256, axis256, indexing='ij')
native = torch.stack((nx, ny), -1)
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')
rows, attempts = [], []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for base in bases:
        accepted = 0
        attempt = 0
        while accepted < per_base:
            draw_seed = seed + base * 100000 + attempt
            original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
            sample = sample_one_shot_slide_artifacts_v4(
                context, [base], draw_seed + 1, side=side, source_sample=original)
            valid = sample['valid_mask'][0]
            valid_query = (F.grid_sample(valid[None, None].float(), query_grid,
                           mode='bilinear', align_corners=False).flatten() > .99)
            use = bool(sample['eligible'][0]) and int(valid_query.sum()) >= 4
            attempts.append({'base_index': base, 'draw_seed': draw_seed,
                             'used': use, 'valid_native_pixels': int(valid.sum()),
                             'valid_query_sites': int(valid_query.sum()),
                             'physical_section_id': sample['provenance'][0]['physical_section_id']})
            attempt += 1
            if not use:
                continue

            observed = native[valid]
            target = rigid_points_090(sample['state'], sample['reflection'], observed)[0]
            reference = target[None]
            while True:
                update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
                proposed = compose_full_frame_state(sample['state'].expand(64, -1), update)
                initial = (rigid_points_090(proposed, sample['reflection'].expand(64),
                                           observed) - reference).norm(dim=-1).mean(-1) / 1000
                possible = (initial >= .35) & (initial <= 1.5)
                if bool(possible.any()):
                    index = int((initial - .9).abs().masked_fill(~possible, 10).argmin())
                    candidate = proposed[index:index + 1].double()
                    break

            rigid_query = rigid_points_090(sample['state'].double(), sample['reflection'],
                                           (query - .5 / side).double())[0]
            valid_chart = observed.double()
            for reflected in (int(sample['reflection'][0]), 1 - int(sample['reflection'][0])):
                centre, frame, basis = full_frame_state_to_components(candidate)
                edges = frame[:, :, :2] @ basis
                chart = (query - .5 / side).double().clone()
                if reflected:
                    chart[:, 0] = (side - 1) / side - chart[:, 0]
                design = torch.cat((torch.ones_like(chart[:, :1]), chart - .5), -1)
                x = design[valid_query]
                y = rigid_query[valid_query]
                gram = x.T @ x
                singular = torch.linalg.svdvals(gram)
                rank = int((singular > singular[0] * 1e-8).sum())
                condition = float(singular[0] / singular[-1]) if rank == 3 else None
                prior = torch.stack((centre[0], edges[0, :, 0], edges[0, :, 1]))
                native_chart = valid_chart.clone()
                if reflected:
                    native_chart[:, 0] = (side - 1) / side - native_chart[:, 0]
                native_design = torch.cat((torch.ones_like(native_chart[:, :1]),
                                           native_chart - .5), -1)
                fit = {}
                for name, diagonal in penalties.items():
                    ridge = torch.diag(torch.tensor(diagonal, device='cuda', dtype=torch.float64))
                    root_ridge = ridge.sqrt()
                    fitted = torch.linalg.lstsq(
                        torch.cat((x, root_ridge)),
                        torch.cat((y, root_ridge @ prior))).solution
                    raw = native_design @ fitted
                    ouv = torch.stack((fitted[0] - .5 * (fitted[1] + fitted[2]),
                                       fitted[1], fitted[2]))[None]
                    state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
                    packed = rigid_points_090(state, torch.tensor([reflected], device='cuda'),
                                              observed.double())[0]
                    fit[name] = {'raw_mm': float((raw - target).norm(dim=-1).mean() / 1000),
                                 'repacked_mm': float((packed - target).norm(dim=-1).mean() / 1000)}
                row = {'base_index': base, 'draw_seed': draw_seed,
                       'physical_section_id': sample['provenance'][0]['physical_section_id'],
                       'subject_plan_receipt_sha256': sample['provenance'][0]['subject_plan_receipt_sha256'],
                       'animal_id': sample['provenance'][0]['base_lineage']['animal_id'],
                       'specimen_id': sample['provenance'][0]['base_lineage']['specimen_id'],
                       'experiment_id': sample['provenance'][0]['base_lineage']['experiment_id'],
                       'true_reflection': int(sample['reflection'][0]),
                       'candidate_reflection': reflected,
                       'reflection_match': reflected == int(sample['reflection'][0]),
                       'initial_rigid_mm': float((rigid_points_090(
                           candidate, torch.tensor([reflected], device='cuda'),
                           observed.double())[0] - target).norm(dim=-1).mean() / 1000),
                       'valid_native_pixels': int(valid.sum()),
                       'valid_query_sites': int(valid_query.sum()),
                       'design_rank': rank, 'design_condition': condition,
                       'fit_mm': fit}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            accepted += 1
        print(json.dumps({'event': 'base_complete', 'base': base,
                          'accepted': accepted, 'attempted': attempt}), flush=True)

(out / 'attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in attempts))
summary = {'sections': len(rows) // 2, 'rows': len(rows), 'attempts': len(attempts),
           'full_rank_rows': sum(row['design_rank'] == 3 for row in rows),
           'reflection_match': {str(flag): {name: {
               metric: float(np.mean([row['fit_mm'][name][metric] for row in rows
                    if row['reflection_match'] == flag]))
               for metric in ('raw_mm', 'repacked_mm')}
               for name in penalties} for flag in (False, True)}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'attempts_sha256': sha(out / 'attempts.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'source_sha256': config['source_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', **summary}), flush=True)
