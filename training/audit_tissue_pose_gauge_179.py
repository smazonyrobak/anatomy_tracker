"""TRAIN-only whole-canvas versus tissue-supported synthetic pose gauge audit."""

import hashlib
import json
import math
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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/TISSUE_POSE_GAUGE_179_PROTOCOL_20261011.md'
run = root / 'reports/tissue_pose_gauge_179_train_001'
assert run.drive.upper() == 'I:' and not run.exists() and torch.cuda.is_available()
config = {'seed': 202610179000, 'bases': list(range(64)), 'draws_per_base': 2,
    'split': 'TRAIN-only', 'support_threshold': .25, 'fit_weight': 'atlas support',
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ('audit_tissue_pose_gauge_179.py',
                     'arbitrary_plane_panel_geometry_train_166.py',
                     'arbitrary_plane_streaming_synthetic_v7_64.py',
                     'arbitrary_plane_full_frame_primitives.py',
                     'arbitrary_plane_geometry.py', 'global_atlas_contrast_090.py')},
    'development_used': False, 'final_animals_used': False,
    'real_expert_truth_used': False, 'public_benchmark_used': False}
run.mkdir(parents=True)
(run / 'config.json').write_text(json.dumps(config, indent=2))
context = load_streaming_synthetic_v7_64(device='cuda')
axis = torch.arange(256, device='cuda', dtype=torch.float32) / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / 256
rows = []
with torch.no_grad(), (run / 'rows.jsonl').open('w') as stream:
    for repetition in range(2):
        for base in range(64):
            draw_seed = config['seed'] + repetition * 100000 + base * 977
            sample = sample_panel_geometry_train_166(context, [base], draw_seed, 256)
            support = sample['support'][0].flatten().clamp(0, 1)
            mask = support > config['support_threshold']
            weight = support * mask
            reflected = bool(sample['reflection'][0])
            x = 255 / 256 - chart[:, 0] if reflected else chart[:, 0]
            design = torch.stack((torch.ones_like(x), x, chart[:, 1]), -1)
            coordinates = sample['centre'][0].reshape(-1, 3)
            gram = design.T @ (weight[:, None] * design)
            coefficients = torch.linalg.solve(gram, design.T @ (weight[:, None] * coordinates))
            tissue_state = full_frame_state_from_components(*physical_ouv_to_frame(
                coefficients[None]))
            old_points = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
            new_points = rigid_points_090(tissue_state, sample['reflection'], chart)[0]
            fit_points = design @ coefficients
            assert float((new_points - fit_points).abs().max()) < .05
            old_residual = (old_points - coordinates).norm(dim=-1)
            new_residual = (new_points - coordinates).norm(dim=-1)
            separation = (old_points - new_points).norm(dim=-1)
            old_five = rigid_points_090(sample['state'], sample['reflection'], five)[0]
            new_five = rigid_points_090(tissue_state, sample['reflection'], five)[0]
            old_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
            new_normal = full_frame_state_to_components(tissue_state)[1][0, :, 2]
            angle = math.degrees(math.acos(float((old_normal * new_normal).sum().abs().clamp(-1, 1))))
            physical_normal = np.asarray(sample['provenance'][0]['physical_unit_normal_ap_dv_ml'])
            nearest_cardinal = math.degrees(math.acos(float(np.abs(physical_normal).max())))
            record = {'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'base_index': base, 'draw_seed': draw_seed, 'repetition': repetition,
                'split': sample['provenance'][0]['split'],
                'eligible': bool(sample['eligible'][0]),
                'tissue_pixels': int(mask.sum()), 'tissue_fraction': float(mask.float().mean()),
                'nearest_cardinal_angle_deg': nearest_cardinal,
                'fit_condition_number': float(torch.linalg.cond(gram)),
                'whole_canvas_residual_tissue_mm': float((old_residual * weight).sum() / weight.sum() / 1000),
                'tissue_fit_residual_tissue_mm': float((new_residual * weight).sum() / weight.sum() / 1000),
                'pose_separation_tissue_mm': float((separation * weight).sum() / weight.sum() / 1000),
                'pose_separation_five_mm': float((old_five - new_five).norm(dim=-1).mean() / 1000),
                'normal_separation_deg': angle,
                'physical_unit_normal_ap_dv_ml': physical_normal.tolist(),
                'original_state12': sample['state'][0].tolist(),
                'tissue_fit_state12': tissue_state[0].tolist(),
                'generator_source_sha256': sample['provenance'][0]['sampler_code_sha256']}
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')

assert len(rows) == len({row['physical_section_id'] for row in rows}) == 128
threshold = float(np.quantile([row['tissue_fraction'] for row in rows], .25))
summary = {'sections': len(rows), 'sparse_tissue_fraction_q25': threshold,
    'cohorts': {}, 'interpretation_threshold': {'median_tissue_separation_mm': .25,
        'upper_quartile_tissue_separation_mm': .50}}
for name, subset in (('all', rows), ('sparse_q25', [row for row in rows
                     if row['tissue_fraction'] <= threshold]),
                     ('steep_ge_30deg', [row for row in rows
                     if row['nearest_cardinal_angle_deg'] >= 30])):
    summary['cohorts'][name] = {'sections': len(subset),
        'mean': {key: float(np.mean([row[key] for row in subset])) for key in (
            'tissue_fraction', 'whole_canvas_residual_tissue_mm',
            'tissue_fit_residual_tissue_mm', 'pose_separation_tissue_mm',
            'pose_separation_five_mm', 'normal_separation_deg', 'fit_condition_number')},
        'median_tissue_separation_mm': float(np.median([row['pose_separation_tissue_mm']
            for row in subset])),
        'q75_tissue_separation_mm': float(np.quantile([row['pose_separation_tissue_mm']
            for row in subset], .75)),
        'q90_tissue_separation_mm': float(np.quantile([row['pose_separation_tissue_mm']
            for row in subset], .90))}
summary['material_gauge_gap'] = (summary['cohorts']['all']['median_tissue_separation_mm'] >= .25
    or summary['cohorts']['all']['q75_tissue_separation_mm'] >= .50)
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(run / 'completed.json').write_text(json.dumps({
    'config_sha256': hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((run / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest(),
    'sections': len(rows), 'source_sha256': config['source_sha256'],
    'protocol_sha256': config['protocol_sha256'],
    'development_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'sections': len(rows),
                  'material_gauge_gap': summary['material_gauge_gap'],
                  'all': summary['cohorts']['all'],
                  'sparse': summary['cohorts']['sparse_q25'],
                  'steep': summary['cohorts']['steep_ge_30deg']}, allow_nan=False), flush=True)
