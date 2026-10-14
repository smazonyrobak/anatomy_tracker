"""One frozen, read-only decomposition of the 305 parent proposal misses."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

SOURCE = Path(__file__).resolve()
PARENT = ROOT / 'runs/joint_spatial_305_acquired_rehearsal_001/intervention/pose_local_00400.pt'
PANELS = ROOT / 'runs/fov_balance_v10_short_pose_001/panels.pt'
OUTPUT = ROOT / 'runs/proposal_geometry_307_diagnostic_001'
assert SOURCE.drive.upper() == PARENT.drive.upper() == PANELS.drive.upper() == OUTPUT.drive.upper() == 'I:'
assert not OUTPUT.exists()

torch.set_num_threads(4)
model, config = load_one_shot_checkpoint(PARENT, device='cuda')
assert config['candidate'] == 305 and model.modes == 80
panels = torch.load(PANELS, map_location='cpu', weights_only=True)
OUTPUT.mkdir(parents=True)
rows = []
with torch.inference_mode():
    for version, panel in panels.items():
        for case in panel:
            prediction = model.predict(case['inputs'][None].cuda())
            state = prediction['state']
            proposed = rigid_points_090(state[:, :, None].expand(-1, -1, 2, -1),
                torch.arange(2, device='cuda')[None, None].expand(1, 80, 2),
                case['chart'].cuda()[None, None, None], (192, 192))
            site_mm = (proposed - case['truth'].cuda()[None, None, None]).norm(
                dim=-1).mean(-1)[0] / 1000
            normal = full_frame_state_to_components(state)[1][0, :, :, 2]
            true_normal = full_frame_state_to_components(case['state'][None].cuda())[1][0, :, 2]
            angle = torch.rad2deg(torch.acos((normal @ true_normal).abs().clamp(0, 1)))
            close = angle <= 15
            best = int(site_mm.flatten().argmin())
            row = {**case['meta'], 'all160_best_site_mm': float(site_mm.min()),
                'all80_best_normal_deg': float(angle.min()),
                'best_site_normal_deg': float(angle[best // 2]),
                'normal_close_15deg': bool(close.any()),
                'normal_close_best_site_mm': float(site_mm[close].min()) if bool(close.any()) else None,
                'nearest_anchor_normal_deg': float(torch.rad2deg(torch.acos(
                    (model.normal_anchor_frames[:, :, 2] @ true_normal).abs().max().clamp(0, 1)))),
                'best_mode_family': 'free' if best // 2 < 16 else 'anchor'}
            rows.append(row)

summary = {}
for version in panels:
    group = [row for row in rows if row['version'] == version]
    misses = [row for row in group if row['all160_best_site_mm'] > 1.5]
    summary[version] = {'cases': len(group), 'all160_capture_le1p5mm': len(group) - len(misses),
        'misses': len(misses), 'misses_without_15deg_normal': sum(
            not row['normal_close_15deg'] for row in misses),
        'misses_with_15deg_normal': sum(row['normal_close_15deg'] for row in misses),
        'misses_sparse_lt15pct': sum(row['visible_fraction'] < .15 for row in misses),
        'misses_full_ge35pct': sum(row['visible_fraction'] >= .35 for row in misses),
        'misses_box_offset': sum(row['sampling_branch'] == 'uniform_box_offset' for row in misses),
        'misses_nearest_anchor_gt15deg': sum(row['nearest_anchor_normal_deg'] > 15 for row in misses)}
receipt = {'schema': 'proposal-geometry-307/v1', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    'panels_sha256': hashlib.sha256(PANELS.read_bytes()).hexdigest(),
    'thresholds': {'site_mm': 1.5, 'unsigned_normal_deg': 15}, 'summary': summary, 'rows': rows}
(OUTPUT / 'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps(summary), flush=True)
