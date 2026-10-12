"""No-training, prespecified candidate-mask control for the frozen 290 cost audit."""
import json
import os
import sys
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
PROTOCOL = SOURCE.parent / 'docs/publication/MASKED_PAIR_COST_291_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/spatial_pair_map_289_002'
AUDIT = ROOT / 'reports/spatial_pair_cost_290_002'
RUN = ROOT / 'reports/masked_pair_cost_291_001'
SOURCE_NAMES = ('audit_masked_pair_cost_291.py', 'arbitrary_plane_spatial_pair_map_v1.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'train_image_map_282.py',
    'train_blind_fit_score_280.py')

assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parent = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
previous = json.loads((AUDIT / 'completed.json').read_text(encoding='utf8'))
assert digest(PARENT / 'completed.json') == '5387c3514eeb91140f246a40240ec383d9c35f2ca917878eb195c7a1a87dd798'
assert digest(AUDIT / 'completed.json') == '1b5c593cfb6c2c41d52ffc1f067b9130f2eeae078d4831dae469bd66f5d986d2'
assert digest(PARENT / 'inner_panel.pt') == parent['files_sha256']['inner_panel.pt']
assert digest(AUDIT / 'config.json') == previous['files_sha256']['config.json']
for name, frozen_hash in previous['source_sha256'].items():
    assert digest(SOURCE / name) == frozen_hash
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64 and len({item['physical_section_id'] for item in panel}) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == json.loads(
    (PARENT / 'config.json').read_text(encoding='utf8'))['synthetic_provenance']
RUN.mkdir(parents=True, exist_ok=False)
config = {'candidate': 291, 'parent_289_receipt_sha256': digest(PARENT / 'completed.json'),
    'parent_290_receipt_sha256': digest(AUDIT / 'completed.json'),
    'panel_sha256': digest(PARENT / 'inner_panel.pt'), 'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'synthetic_provenance': context['provenance'], 'cases': len(panel),
    'side': 96, 'radius': 8, 'support_min': .9, 'variance_min': 1e-5,
    'prior_coefficients': [0., .002, .010],
    'source_role': 'synthetic TRAIN-inner fixed-cost control; no training',
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

variants = ('raw', 'masked', 'masked_prior_0002', 'masked_prior_0010')
dx = torch.arange(-8, 9, device='cuda', dtype=torch.float32)
dy, dx = torch.meshgrid(dx, dx, indexing='ij')
distance2 = (dx.square() + dy.square()).reshape(1, 289)
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
        y = (torch.arange(side, device='cuda', dtype=state.dtype) + .5) / side - .5 / SIDE
        x = (torch.arange(side, device='cuda', dtype=state.dtype) + .5) / side - .5 / SIDE
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
        cost = F.avg_pool2d(source_norm * shifted, 5, stride=1, padding=2).abs() * shifted_support
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
            rigid_points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
            local = (target - rigid_points) @ frame[0]
            plane_offset = torch.linalg.solve(basis[0], local[:, :2].T).T * side
            true_xy = torch.stack((plane_offset[:, 0] * torch.where(reflection[0].bool(), -1., 1.),
                                   plane_offset[:, 1]), -1)
            rounded = true_xy.round().long().clamp(-8, 8)
            true_index = (rounded[:, 1] + 8) * 17 + rounded[:, 0] + 8
            zero_error = true_xy.norm(dim=-1)
            allowed = (sampled_support > .9) & (sampled_var >= 1e-5)
            eligible = ((true_xy.abs() <= 8).all(-1) & (zero_error >= 1)
                & (source_variance >= 1e-5) & allowed[:, 144]
                & allowed.gather(1, true_index[:, None])[:, 0])
            scores = (sampled_cost,
                sampled_cost.masked_fill(~allowed, -1e6),
                (sampled_cost - .002 * distance2).masked_fill(~allowed, -1e6),
                (sampled_cost - .010 * distance2).masked_fill(~allowed, -1e6))
            peaks = [score.argmax(-1) for score in scores]
            raw_peak_disallowed = ~allowed.gather(1, peaks[0][:, None])[:, 0]
            for j, site_index in enumerate(indices_cpu.tolist()):
                if not bool(eligible[j]):
                    continue
                row = {'case': case, 'physical_section_id': item['physical_section_id'],
                    'base_index': item['base_index'], 'group': group, 'site_index': site_index,
                    'true_offset_xy_px96': true_xy[j].tolist(),
                    'zero_error_px96': float(zero_error[j]),
                    'raw_peak_disallowed': bool(raw_peak_disallowed[j])}
                for name, peak_index in zip(variants, peaks):
                    index = int(peak_index[j])
                    peak_xy = torch.tensor((index % 17 - 8, index // 17 - 8), device='cuda')
                    row[name + '_peak_offset_xy_px96'] = peak_xy.tolist()
                    row[name + '_peak_error_px96'] = float((peak_xy - true_xy[j]).norm())
                output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        if (case + 1) % 16 == 0:
            output.flush()
            print(json.dumps({'event': 'case_milestone', 'cases': case + 1}), flush=True)

summary = {}
for group in ('intact', 'fragment'):
    selected = [row for row in rows if row['group'] == group]
    reference = previous['summary'][group]
    assert len(selected) == reference['sites_eligible']
    assert abs(np.mean([row['raw_peak_error_px96'] for row in selected])
               - reference['peak_error_mean_px96']) < 1e-5
    group_result = {'sites_eligible': len(selected),
        'zero_hit_within_2px_fraction': float(np.mean([
            row['zero_error_px96'] <= 2 for row in selected])),
        'raw_peak_disallowed_fraction': float(np.mean([
            row['raw_peak_disallowed'] for row in selected]))}
    for name in variants:
        group_result[name] = {'peak_hit_within_2px_fraction': float(np.mean([
            row[name + '_peak_error_px96'] <= 2 for row in selected])),
            'peak_error_mean_px96': float(np.mean([
            row[name + '_peak_error_px96'] for row in selected]))}
    summary[group] = group_result
    assert abs(group_result['zero_hit_within_2px_fraction']
               - reference['zero_hit_within_2px_fraction']) < 1e-8
    assert abs(group_result['raw']['peak_hit_within_2px_fraction']
               - reference['peak_hit_within_2px_fraction']) < 1e-8
completed = {'candidate': 291, 'parent': 290, 'cases': len(panel), 'summary': summary,
    'source_sha256': config['source_sha256'],
    'files_sha256': {name: digest(RUN / name) for name in ('config.json', 'sites.jsonl')},
    'source_role': config['source_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
