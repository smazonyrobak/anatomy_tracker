"""Read-only pose-factor audit of the frozen 309 TRAIN-inner panel."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/first_choice_exposure_309_001'
panels = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)
rows = []
with torch.inference_mode():
    for start in range(0, len(panels), 4):
        cases = panels[start:start + 4]
        inputs = torch.stack([case['inputs'] for case in cases]).cuda()
        chart = torch.stack([case['chart'] for case in cases]).cuda()
        truth = torch.stack([case['truth'] for case in cases]).cuda()
        true_state = torch.stack([case['state'] for case in cases]).cuda()
        prediction = model.predict(inputs)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflection = torch.arange(2, device='cuda')[None, None].expand(len(cases), model.modes, -1)
        mass = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        proposed = rigid_points_090(states, reflection, chart[:, None, None], (192, 192))
        error = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
        centres, frames, basis = full_frame_state_to_components(prediction['state'])
        true_centres, true_frames, true_basis = full_frame_state_to_components(true_state)
        angle = torch.rad2deg(torch.acos((frames[..., 2] * true_frames[:, None, :, 2]
            ).sum(-1).abs().clamp(0, 1)))
        centre_gap = (centres - true_centres[:, None]).norm(dim=-1) / 1000
        span = torch.linalg.svdvals(basis) / 1000
        true_span = torch.linalg.svdvals(true_basis) / 1000
        top16 = mass.topk(16, -1).indices
        first = mass.argmax(-1)
        oracle = top16.gather(1, error.gather(1, top16).argmin(-1, keepdim=True)).squeeze(1)
        for i in range(len(cases)):
            a, b = int(first[i]), int(oracle[i])
            rows.append({'first_mm': float(error[i, a]), 'best16_mm': float(error[i, b]),
                'first_angle_deg': float(angle[i, a // 2]), 'best16_angle_deg': float(angle[i, b // 2]),
                'first_centre_mm': float(centre_gap[i, a // 2]),
                'best16_centre_mm': float(centre_gap[i, b // 2]),
                'first_minor_fov_mm': float(span[i, a // 2, 1]),
                'best16_minor_fov_mm': float(span[i, b // 2, 1]),
                'true_minor_fov_mm': float(true_span[i, 1]),
                'first_anchor': a // 2 >= model.base_modes,
                'best16_anchor': b // 2 >= model.base_modes})

wrong = [r for r in rows if r['first_mm'] > 2 and r['best16_mm'] <= 1.5]
summary = {'n': len(rows), 'wrong_with_near16_count': len(wrong),
    'first_mean_mm': float(np.mean([r['first_mm'] for r in rows])),
    'best16_mean_mm': float(np.mean([r['best16_mm'] for r in rows])),
    'top16_within1p5': sum(r['best16_mm'] <= 1.5 for r in rows)}
for label, cohort in (('all', rows), ('wrong_with_near16', wrong)):
    summary[label] = {name: {'median': float(np.median([r[name] for r in cohort])),
        'p90': float(np.percentile([r[name] for r in cohort], 90))}
        for name in ('first_mm', 'best16_mm', 'first_angle_deg', 'best16_angle_deg',
                     'first_centre_mm', 'best16_centre_mm', 'first_minor_fov_mm',
                     'best16_minor_fov_mm', 'true_minor_fov_mm')}
    summary[label].update({'first_angle_over20': sum(r['first_angle_deg'] > 20 for r in cohort),
        'first_centre_over1mm': sum(r['first_centre_mm'] > 1 for r in cohort),
        'first_minor_under_half_truth': sum(r['first_minor_fov_mm'] < .5 * r['true_minor_fov_mm'] for r in cohort),
        'first_anchor': sum(r['first_anchor'] for r in cohort),
        'best16_anchor': sum(r['best16_anchor'] for r in cohort)})
print(summary)
