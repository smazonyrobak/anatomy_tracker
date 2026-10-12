"""No-training diagnosis of the frozen 289 local image/atlas cost volume."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_spatial_pair_map_v1 import SpatialPairJointModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.train_image_map_282 import SIDE, digest, mapped_sites, tensor_digest


SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/SPATIAL_PAIR_COST_290_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/spatial_pair_map_289_002'
RUN = ROOT / 'reports/spatial_pair_cost_290_001'
SOURCE_NAMES = ('audit_spatial_pair_cost_290.py', 'arbitrary_plane_spatial_pair_map_v1.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'train_image_map_282.py',
    'train_blind_fit_score_280.py')


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()
parent = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
assert parent['candidate'] == 289 and not parent['advance'] and parent['selected_step'] == 5000
assert digest(PARENT / 'completed.json') == \
    '5387c3514eeb91140f246a40240ec383d9c35f2ca917878eb195c7a1a87dd798'
assert digest(PARENT / 'inner_panel.pt') == parent['files_sha256']['inner_panel.pt']
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64 and len({item['physical_section_id'] for item in panel}) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf8'))
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
RUN.mkdir(parents=True, exist_ok=False)
config = {'candidate': 290, 'parent_289_completed_sha256': digest(PARENT / 'completed.json'),
    'parent_289_panel_sha256': digest(PARENT / 'inner_panel.pt'),
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'synthetic_provenance': context['provenance'], 'cases': len(panel),
    'cost_grid_side': 96, 'offset_radius_pixels': 8, 'local_normalization_window': 9,
    'correlation_window': 5, 'source_atlas_variance_min': 1e-5,
    'atlas_support_min': .9, 'true_offset_min_pixels': 1.,
    'peak_hit_radius_pixels': 2.,
    'source_role': 'synthetic TRAIN-inner fixed-cost information diagnostic; no training',
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

rows = []
with (RUN / 'sites.jsonl').open('w', encoding='utf8') as output, torch.no_grad():
    for case, item in enumerate(panel):
        assert tensor_digest(item['inputs']) == item['input_sha256']
        check = torch.cat((item['site_indices'][:, None].float(),
            item['centre'].reshape(-1, 3)[item['site_indices']]), -1)
        assert tensor_digest(check) == item['target_sha256']
        inputs = item['inputs'][None].cuda()
        state = item['state'][None].cuda()
        reflection = item['reflection'][None].cuda()
        offsets = item['offsets'][None].cuda()
        weights = item['weights'][None].cuda()
        center, frame, basis = full_frame_state_to_components(state)
        side = 96
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / SIDE
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / SIDE
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
        chart = torch.stack((sx.expand(1, -1, -1), yy[None]), -1)
        rigid = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                          basis[:, None, None], chart)
        coordinates = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            context['atlas'], coordinates, (0., 0., 0.), (25., 25., 25.), weights)
        support = rendered[:, 1:2]
        atlas = rendered[:, :1] / support.clamp_min(1e-4)
        source = F.interpolate(inputs, (side, side), mode='area')[:, :1]
        source_norm = SpatialPairJointModel.local_normalize(source)
        atlas_norm = SpatialPairJointModel.local_normalize(atlas)
        shifted = F.unfold(atlas_norm, 17, padding=8).reshape(1, 289, side, side)
        shifted_support = F.unfold(support, 17, padding=8).reshape(1, 289, side, side)
        cost = F.avg_pool2d(source_norm * shifted, 5, stride=1, padding=2).abs()
        cost *= shifted_support
        source_mean = F.avg_pool2d(source, 9, stride=1, padding=4)
        source_var = (F.avg_pool2d(source.square(), 9, stride=1, padding=4)
                      - source_mean.square()).clamp_min(0)
        atlas_mean = F.avg_pool2d(atlas, 9, stride=1, padding=4)
        atlas_var = (F.avg_pool2d(atlas.square(), 9, stride=1, padding=4)
                     - atlas_mean.square()).clamp_min(0)
        shifted_var = F.unfold(atlas_var, 17, padding=8).reshape(1, 289, side, side)
        for group, indices_cpu in (('intact', item['intact_indices']),
                                   ('fragment', item['fragment_indices'])):
            if not len(indices_cpu):
                continue
            indices = indices_cpu.cuda()
            xy = torch.stack((indices.remainder(SIDE),
                indices.div(SIDE, rounding_mode='floor')), -1).float()
            grid = ((xy + .5) * (2 / SIDE) - 1)[None, None]
            sampled_cost = F.grid_sample(cost, grid, align_corners=False,
                padding_mode='border')[0, :, 0].T
            sampled_support = F.grid_sample(shifted_support, grid, align_corners=False,
                padding_mode='border')[0, :, 0].T
            sampled_var = F.grid_sample(shifted_var, grid, align_corners=False,
                padding_mode='border')[0, :, 0].T
            source_variance = F.grid_sample(source_var, grid, align_corners=False,
                padding_mode='border')[0, 0, 0]
            target = item['centre'].reshape(-1, 3)[indices_cpu].cuda()
            rigid_points = mapped_sites(
                {'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
            delta = target - rigid_points
            local = delta @ frame[0]
            plane_offset = torch.linalg.solve(basis[0], local[:, :2].T).T * side
            true_xy = torch.stack((plane_offset[:, 0] * torch.where(reflection[0], -1., 1.),
                                   plane_offset[:, 1]), -1)
            rounded = true_xy.round().long().clamp(-8, 8)
            true_index = (rounded[:, 1] + 8) * 17 + rounded[:, 0] + 8
            peak_index = sampled_cost.argmax(-1)
            peak_xy = torch.stack((peak_index.remainder(17) - 8,
                peak_index.div(17, rounding_mode='floor') - 8), -1)
            peak_error = (peak_xy.float() - true_xy).norm(dim=-1)
            zero_error = true_xy.norm(dim=-1)
            selected_cost = sampled_cost.gather(1, true_index[:, None])[:, 0]
            zero_cost = sampled_cost[:, 144]
            peak_cost = sampled_cost.gather(1, peak_index[:, None])[:, 0]
            selected_support = sampled_support.gather(1, true_index[:, None])[:, 0]
            selected_var = sampled_var.gather(1, true_index[:, None])[:, 0]
            eligible = ((true_xy.abs() <= 8).all(-1) & (zero_error >= 1)
                & (source_variance >= 1e-5) & (selected_var >= 1e-5)
                & (sampled_support[:, 144] > .9) & (selected_support > .9))
            for j, index in enumerate(indices_cpu.tolist()):
                row = {'case': case, 'physical_section_id': item['physical_section_id'],
                    'base_index': item['base_index'], 'group': group, 'site_index': index,
                    'eligible': bool(eligible[j]), 'true_offset_xy_px96': true_xy[j].tolist(),
                    'peak_offset_xy_px96': peak_xy[j].tolist(),
                    'peak_error_px96': float(peak_error[j]), 'zero_error_px96': float(zero_error[j]),
                    'true_rounded_cost': float(selected_cost[j]), 'zero_cost': float(zero_cost[j]),
                    'mean_cost': float(sampled_cost[j].mean()), 'peak_cost': float(peak_cost[j]),
                    'source_variance': float(source_variance[j]),
                    'true_atlas_variance': float(selected_var[j]),
                    'zero_support': float(sampled_support[j, 144]),
                    'true_support': float(selected_support[j]),
                    'through_plane_residual_mm': float(local[j, 2].abs() / 1000)}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        if (case + 1) % 16 == 0:
            output.flush()
            print(json.dumps({'event': 'case_milestone', 'cases': case + 1}), flush=True)

summary = {}
for group in ('intact', 'fragment'):
    selected = [row for row in rows if row['group'] == group and row['eligible']]
    n = len(selected)
    summary[group] = {'sites_total': sum(row['group'] == group for row in rows),
        'sites_eligible': n,
        'peak_hit_within_2px_fraction': float(np.mean([row['peak_error_px96'] <= 2 for row in selected]))
                                        if n else None,
        'zero_hit_within_2px_fraction': float(np.mean([row['zero_error_px96'] <= 2 for row in selected]))
                                        if n else None,
        'peak_error_mean_px96': float(np.mean([row['peak_error_px96'] for row in selected]))
                                if n else None,
        'zero_error_mean_px96': float(np.mean([row['zero_error_px96'] for row in selected]))
                                if n else None,
        'true_minus_zero_cost_mean': float(np.mean([
            row['true_rounded_cost'] - row['zero_cost'] for row in selected])) if n else None,
        'true_above_zero_cost_fraction': float(np.mean([
            row['true_rounded_cost'] > row['zero_cost'] for row in selected])) if n else None,
        'true_minus_mean_cost_mean': float(np.mean([
            row['true_rounded_cost'] - row['mean_cost'] for row in selected])) if n else None,
        'through_plane_residual_mean_mm': float(np.mean([
            row['through_plane_residual_mm'] for row in selected])) if n else None}
intact, fragment = summary['intact'], summary['fragment']
gate = {'intact_sites_at_least_200': intact['sites_eligible'] >= 200,
    'fragment_sites_at_least_100': fragment['sites_eligible'] >= 100,
    'intact_peak_hit_at_least_0_60': intact['peak_hit_within_2px_fraction'] is not None
        and intact['peak_hit_within_2px_fraction'] >= .60,
    'fragment_peak_hit_at_least_0_50': fragment['peak_hit_within_2px_fraction'] is not None
        and fragment['peak_hit_within_2px_fraction'] >= .50,
    'intact_hit_advantage_at_least_0_10': intact['peak_hit_within_2px_fraction'] is not None
        and intact['peak_hit_within_2px_fraction'] - intact['zero_hit_within_2px_fraction'] >= .10,
    'fragment_hit_advantage_at_least_0_10': fragment['peak_hit_within_2px_fraction'] is not None
        and fragment['peak_hit_within_2px_fraction'] - fragment['zero_hit_within_2px_fraction'] >= .10}
completed = {'candidate': 290, 'parent': 289, 'cases': len(panel), 'summary': summary,
    'gate': gate, 'cost_sufficient': all(gate.values()),
    'elapsed_seconds': time.perf_counter() - started,
    'source_sha256': config['source_sha256'],
    'files_sha256': {name: digest(RUN / name) for name in ('config.json', 'sites.jsonl')},
    'source_role': config['source_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary, 'cost_sufficient':
                  completed['cost_sufficient']}), flush=True)
