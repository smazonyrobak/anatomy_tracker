"""Read-only decomposition of exact-pose synthetic residuals on frozen 309 v10 cuts."""
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090

panel = torch.load(root / 'runs/first_choice_exposure_309_001/panels.pt',
                   map_location='cpu', weights_only=True)['v10']
rows = []
with torch.inference_mode():
    for case in panel:
        state = case['state'][None, None, None].expand(1, 1, 2, -1)
        chart = case['chart'][None, None, None]
        reflection = torch.tensor([[[0, 1]]])
        rigid = rigid_points_090(state, reflection, chart, (192, 192))[0, 0]
        residual = case['truth'][None] - rigid
        branch = residual.norm(dim=-1).mean(-1).argmin()
        residual = residual[branch]
        frame = full_frame_state_to_components(case['state'][None])[1][0]
        local = residual @ frame
        in_plane = local[:, :2].norm(dim=-1) / 1000
        normal = local[:, 2].abs() / 1000
        total = residual.norm(dim=-1) / 1000
        rows.append((float(total.mean()), float(in_plane.mean()), float(normal.mean()),
                     float((normal.square().sum() / total.square().sum()).sqrt()),
                     math.degrees(math.acos(float(frame[:, 2].abs().max().clamp(0, 1))))))
values = np.asarray(rows)
print({'sections': len(rows), 'mean_exact_pose_total_mm': float(values[:, 0].mean()),
       'mean_in_plane_mm': float(values[:, 1].mean()),
       'mean_abs_normal_mm': float(values[:, 2].mean()),
       'median_normal_rms_fraction': float(np.median(values[:, 3])),
       'p90_normal_rms_fraction': float(np.percentile(values[:, 3], 90)),
       'steep_sections': int((values[:, 4] >= 45).sum()),
       'steep_mean_abs_normal_mm': float(values[values[:, 4] >= 45, 2].mean())})
