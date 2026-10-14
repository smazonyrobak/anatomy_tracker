"""Exact-pose local-map retention after matched 305 fluorescence adaptation."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SIDE = 192
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/JOINT_SPATIAL_305_EXACT_MAP_RETENTION_PROTOCOL_20261014.md'
POSE = ROOT / 'runs/joint_spatial_feedback_305_pose_002'
ADAPTATION = ROOT / 'runs/joint_spatial_305_fluorescence_adaptation_001'
PANEL = POSE / 'inner_panel.pt'
RUN = ROOT / 'runs/joint_spatial_305_exact_map_retention_001'
CHECKPOINTS = {arm: ADAPTATION / arm / 'pose_step_00600.pt' for arm in ('control', 'adapt')}
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def surface_points(mapped, chart):
    batch, count = mapped['centre_surface_ccf_ap_dv_ml_um'].shape[:2]
    surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
    grid = ((chart + .5 / SIDE) * 2 - 1)[:, None].expand(-1, count, -1, -1)
    grid = grid.reshape(batch * count, 1, chart.shape[1], 2)
    return F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[:, :, 0].transpose(1, 2).reshape(
                             batch, count, chart.shape[1], 3)


pose_receipt = json.loads((POSE / 'completed.json').read_text(encoding='utf8'))
adapt_config = json.loads((ADAPTATION / 'config.json').read_text(encoding='utf8'))
assert adapt_config['no_dev_cal_final_or_public_benchmark']
assert adapt_config['optimized_modules'] == ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global']
assert sha(PANEL) == adapt_config['panel_sha256'] == pose_receipt['output_sha256']['inner_panel.pt']
checkpoint_hashes = {}
for arm, checkpoint in CHECKPOINTS.items():
    receipt = json.loads((checkpoint.parent / 'completed.json').read_text(encoding='utf8'))
    checkpoint_hashes[arm] = sha(checkpoint)
    assert receipt['arm'] == arm and receipt['updates'] == 600
    assert checkpoint_hashes[arm] == receipt['output_sha256'][checkpoint.name]

panel = torch.load(PANEL, map_location='cpu', weights_only=False)
assert len(panel) == 64 and len({row['physical_section_id'] for row in panel}) == 64
assert Counter(row['base'] for row in panel) == Counter({base: 8 for base in range(56, 64)})
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64('cuda')
pose_config = json.loads((POSE / 'config.json').read_text(encoding='utf8'))
assert pose_config['synthetic_provenance'] == context['provenance']

rows = []
with torch.inference_mode():
    for item in panel:
        chart = item['chart'][None].cuda()
        truth = item['truth'][None].cuda()
        state = item['state'][None, None].cuda()
        reflection = item['reflection'][None, None].cuda()
        residual = (rigid_points_090(state, reflection, chart[:, None],
                                     (SIDE, SIDE))[:, 0] - truth).norm(dim=-1)
        rows.append({'physical_section_id': item['physical_section_id'],
            'base': item['base'], 'slot': item['slot'],
            'obliquity_deg': item['obliquity_deg'],
            'visible_fraction': item['visible_fraction'],
            'damage_fraction': item['damage_fraction'],
            'exterior_style': item['exterior_style'],
            'zero_warp_mean_error_mm': float(residual.mean() / 1000),
            'zero_warp_residual_rms_mm': float(residual.square().mean().sqrt() / 1000)})

for arm, checkpoint in CHECKPOINTS.items():
    model, checkpoint_config = load_one_shot_checkpoint(checkpoint, 'cuda')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['step'] == 600 and saved['arm'] == arm and checkpoint_config == saved['config']
    with torch.inference_mode():
        for start in range(0, len(panel), 4):
            group = panel[start:start + 4]
            inputs = torch.stack([item['inputs'] for item in group]).cuda()
            chart = torch.stack([item['chart'] for item in group]).cuda()
            truth = torch.stack([item['truth'] for item in group]).cuda()
            state = torch.stack([item['state'] for item in group]).cuda()[:, None]
            reflection = torch.stack([item['reflection'] for item in group]).cuda()[:, None]
            offsets = torch.stack([item['offsets'] for item in group]).cuda()
            weights = torch.stack([item['weights'] for item in group]).cuda()
            prediction = model.predict(inputs)
            exact = {**prediction, 'state': state}
            index = torch.zeros((len(group), 1), device='cuda', dtype=torch.long)
            mapped = model.map(exact, offsets, index, reflection, (96, 96),
                context['atlas'], weights, feature_side=96,
                source_shape=(SIDE, SIDE), max_derivative=None)
            error = (surface_points(mapped, chart)[:, 0] - truth).norm(dim=-1).mean(-1) / 1000
            assert torch.isfinite(error).all()
            for offset, value in enumerate(error.tolist()):
                rows[start + offset][f'{arm}_exact_map_error_mm'] = value
    del model, saved
    torch.cuda.empty_cache()

strong = set(sorted(range(len(rows)), key=lambda index: rows[index][
    'zero_warp_residual_rms_mm'], reverse=True)[:16])
for index, row in enumerate(rows):
    row['strong_warp_top16'] = index in strong
    row['adapt_minus_control_exact_map_error_mm'] = (
        row['adapt_exact_map_error_mm'] - row['control_exact_map_error_mm'])

strong_rows = [row for row in rows if row['strong_warp_top16']]
zero_full = float(np.mean([row['zero_warp_mean_error_mm'] for row in rows]))
zero_strong = float(np.mean([row['zero_warp_mean_error_mm'] for row in strong_rows]))
arms = {}
for arm in ('control', 'adapt'):
    full = float(np.mean([row[f'{arm}_exact_map_error_mm'] for row in rows]))
    hard = float(np.mean([row[f'{arm}_exact_map_error_mm'] for row in strong_rows]))
    arms[arm] = {'full_exact_map_mean_mm': full,
        'full_gain_vs_zero_mm': zero_full - full,
        'strong_warp_exact_map_mean_mm': hard,
        'strong_warp_gain_vs_zero_mm': zero_strong - hard}

full_delta = arms['adapt']['full_exact_map_mean_mm'] - arms['control']['full_exact_map_mean_mm']
strong_delta = (arms['adapt']['strong_warp_exact_map_mean_mm']
                - arms['control']['strong_warp_exact_map_mean_mm'])
summary = {'cases': len(rows), 'strong_warp_cases': len(strong_rows),
    'zero_warp_full_mean_mm': zero_full, 'zero_warp_strong_mean_mm': zero_strong,
    'arms': arms, 'adapt_minus_control_full_mean_mm': full_delta,
    'adapt_minus_control_strong_mean_mm': strong_delta,
    'adapt_better_cases': sum(row['adapt_minus_control_exact_map_error_mm'] < 0 for row in rows),
    'adapt_worse_cases': sum(row['adapt_minus_control_exact_map_error_mm'] > 0 for row in rows),
    'adapt_equal_cases': sum(row['adapt_minus_control_exact_map_error_mm'] == 0 for row in rows),
    'feature_adaptation_harmed_full_panel_exact_map': full_delta > 0,
    'feature_adaptation_harmed_strong_warp_exact_map': strong_delta > 0,
    'interpretation': 'deterministic frozen TRAIN-inner retention diagnostic; not biological validation or calibration'}
source_hashes = {path.name: sha(path) for path in (SOURCE, PROTOCOL,
    SOURCE.with_name('arbitrary_plane_one_shot_inference.py'),
    SOURCE.with_name('arbitrary_plane_one_shot_model.py'),
    SOURCE.with_name('arbitrary_plane_streaming_synthetic_v7_64.py'),
    SOURCE.with_name('global_atlas_contrast_090.py'))}
config = {'protocol': str(PROTOCOL), 'source_commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=SOURCE.parent, text=True).strip(),
    'source_sha256': source_hashes, 'panel': str(PANEL), 'panel_sha256': sha(PANEL),
    'checkpoint_sha256': checkpoint_hashes,
    'mapping': {'pose': 'saved exact post-artifact state/reflection; no refinement',
        'map_shape': [96, 96], 'feature_side': 96, 'source_shape': [SIDE, SIDE],
        'max_derivative': None, 'target': '512 saved valid-pixel CCF AP/DV/ML samples'},
    'strong_warp': 'largest 16 zero-warp residual RMS; frozen panel order breaks ties',
    'harm_rule': 'adapt full-panel exact-map mean > control',
    'no_dev_cal_final_or_public_benchmark': True, 'model_updated': False}

RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
with (RUN / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
receipt = {'config_sha256': sha(RUN / 'config.json'), 'rows_sha256': sha(RUN / 'rows.jsonl'),
    'summary_sha256': sha(RUN / 'summary.json'), 'source_sha256': sha(SOURCE),
    'protocol_sha256': sha(PROTOCOL), 'panel_sha256': sha(PANEL),
    'checkpoint_sha256': checkpoint_hashes, 'completed': True}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'output': str(RUN), **summary}), flush=True)
