"""Read-only synthetic upper-bound test of optional rough acquisition hints."""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/first_choice_exposure_309_001'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
torch.manual_seed(20261014311)
assert Path(__file__).resolve().drive.upper() == RUN.drive.upper() == 'I:'
panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)['v10']
model, _ = load_one_shot_checkpoint(RUN / 'single/pose_local_20000.pt')
model.requires_grad_(False)
rows = []
with torch.inference_mode():
    for start in range(0, len(panels), 4):
        cases = panels[start:start + 4]
        inputs = torch.stack([case['inputs'] for case in cases]).cuda()
        chart = torch.stack([case['chart'] for case in cases]).cuda()
        truth = torch.stack([case['truth'] for case in cases]).cuda()
        true_state = torch.stack([case['state'] for case in cases]).cuda()
        predicted = model.predict(inputs)
        states = predicted['state'][:, :, None].expand(-1, -1, 2, -1)
        reflection = torch.arange(2, device='cuda')[None, None].expand(len(cases), model.modes, -1)
        mass = (predicted['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-predicted['reflection_logit']),
            F.logsigmoid(predicted['reflection_logit'])), -1)).flatten(1)
        rigid = rigid_points_090(states, reflection, chart[:, None, None], (192, 192))
        error = (rigid - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
        centres, frames, _ = full_frame_state_to_components(predicted['state'])
        true_centres, true_frames, _ = full_frame_state_to_components(true_state)
        true_normal = true_frames[:, :, 2]
        tangent = torch.randn_like(true_normal)
        tangent = F.normalize(tangent - (tangent * true_normal).sum(-1, keepdim=True) * true_normal,
                              dim=-1)
        hint_angle = torch.randn(len(cases), device='cuda') * math.radians(15)
        normal_hint = (hint_angle.cos()[:, None] * true_normal
                       + hint_angle.sin()[:, None] * tangent)
        ap_hint = true_centres[:, 0] + 750 * torch.randn(len(cases), device='cuda')
        ap_penalty = .5 * ((centres[:, :, 0] - ap_hint[:, None]) / 1800).square()
        angle = torch.acos((frames[:, :, :, 2] * normal_hint[:, None]).sum(-1).abs().clamp(0, 1))
        normal_penalty = .5 * (angle / math.radians(25)).square()
        wider_hint = ((2 * hint_angle).cos()[:, None] * true_normal
                      + (2 * hint_angle).sin()[:, None] * tangent)
        wider_angle = torch.acos((frames[:, :, :, 2] * wider_hint[:, None]).sum(-1).abs().clamp(0, 1))
        wider_penalty = .5 * (wider_angle / math.radians(25)).square()
        wrong_hint = torch.rand(len(cases), device='cuda') < .1
        wrong_angle = hint_angle + wrong_hint * math.radians(60)
        wrong_normal = (wrong_angle.cos()[:, None] * true_normal
                        + wrong_angle.sin()[:, None] * tangent)
        wrong_ap = ap_hint + wrong_hint * 3500
        wrong_normal_angle = torch.acos((frames[:, :, :, 2] * wrong_normal[:, None]
                                         ).sum(-1).abs().clamp(0, 1))
        wrong_penalty = (.5 * ((centres[:, :, 0] - wrong_ap[:, None]) / 1800).square()
                         + .5 * (wrong_normal_angle / math.radians(25)).square())
        true_ap_penalty = .5 * ((centres[:, :, 0] - true_centres[:, :1]) / 1800).square()
        true_angle = torch.acos((frames[:, :, :, 2] * true_normal[:, None]).sum(-1).abs().clamp(0, 1))
        true_normal_penalty = .5 * (true_angle / math.radians(25)).square()
        scores = {
            'blind': mass,
            'ap_hint_top16': mass - ap_penalty.repeat_interleave(2, 1),
            'normal_hint_top16': mass - normal_penalty.repeat_interleave(2, 1),
            'both_hints_top16': mass - (ap_penalty + normal_penalty).repeat_interleave(2, 1),
            'both_hints_all160': mass - (ap_penalty + normal_penalty).repeat_interleave(2, 1),
            'double_angle_noise_top16': mass - (ap_penalty + wider_penalty).repeat_interleave(2, 1),
            'ten_percent_wrong_top16': mass - wrong_penalty.repeat_interleave(2, 1),
            'ten_percent_wrong_robust_top16': mass + (.9 * (-wrong_penalty).exp() + .1
                                                      ).log().repeat_interleave(2, 1),
            'perfect_hints_top16': mass - (true_ap_penalty + true_normal_penalty).repeat_interleave(2, 1),
            'perfect_hints_all160': mass - (true_ap_penalty + true_normal_penalty).repeat_interleave(2, 1),
        }
        top16 = mass.topk(16, -1).indices
        for i, case in enumerate(cases):
            chosen = {}
            for arm, score in scores.items():
                ids = torch.arange(mass.shape[1], device='cuda') if arm.endswith('all160') else top16[i]
                chosen[arm] = int(ids[score[i, ids].argmax()])
            rows.append({'physical_section_id': case['meta']['physical_section_id'],
                'base': case['meta']['base'], 'visible_fraction': case['meta']['postartifact_visible_fraction'],
                'hint_ap_error_mm': float((ap_hint[i] - true_centres[i, 0]).abs() / 1000),
                'hint_normal_error_deg': float(hint_angle[i].abs() * 180 / math.pi),
                'wrong_hint': bool(wrong_hint[i]),
                'errors_mm': error[i].cpu().tolist(), 'chosen': chosen,
                'top16_oracle_mm': float(error[i, top16[i]].min()),
                'all160_oracle_mm': float(error[i].min())})

summary = {'n': len(rows), 'hint_ap_error_median_mm': float(np.median([
    row['hint_ap_error_mm'] for row in rows])),
    'hint_normal_error_median_deg': float(np.median([
        row['hint_normal_error_deg'] for row in rows])),
    'top16_oracle_mean_mm': float(np.mean([row['top16_oracle_mm'] for row in rows])),
    'all160_oracle_mean_mm': float(np.mean([row['all160_oracle_mm'] for row in rows]))}
for arm in rows[0]['chosen']:
    errors = np.array([row['errors_mm'][row['chosen'][arm]] for row in rows])
    summary[arm] = {'mean_mm': float(errors.mean()), 'median_mm': float(np.median(errors)),
        'p90_mm': float(np.percentile(errors, 90)), 'over5mm': int((errors > 5).sum()),
        'within1p5mm': int((errors <= 1.5).sum())}
summary['wrong_hint_cases'] = sum(row['wrong_hint'] for row in rows)
summary['wrong_hint_subset_mean_mm'] = {arm: float(np.mean([
    row['errors_mm'][row['chosen'][arm]] for row in rows if row['wrong_hint']]))
    for arm in ('blind', 'ten_percent_wrong_top16', 'ten_percent_wrong_robust_top16')}
print(json.dumps(summary, indent=2, allow_nan=False))
