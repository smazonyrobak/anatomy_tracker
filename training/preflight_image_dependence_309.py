"""Read-only within-base image-permutation null for frozen 309 TRAIN-inner cuts."""
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
cases = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)

rng = np.random.default_rng(309)
permutation = np.arange(len(cases))
for base in sorted({case['meta']['base'] for case in cases}):
    indices = np.array([i for i, case in enumerate(cases) if case['meta']['base'] == base])
    shuffled = rng.permutation(indices)
    permutation[shuffled] = np.roll(shuffled, 1)

rows = []
with torch.inference_mode():
    for start in range(0, len(cases), 4):
        batch = cases[start:start + 4]
        chart = torch.stack([case['chart'] for case in batch]).cuda()
        truth = torch.stack([case['truth'] for case in batch]).cuda()
        true_state = torch.stack([case['state'] for case in batch]).cuda()
        _, true_frames, _ = full_frame_state_to_components(true_state)
        for arm, source_indices in (('real', range(start, start + len(batch))),
                                    ('permuted', permutation[start:start + len(batch)])):
            inputs = torch.stack([cases[i]['inputs'] for i in source_indices]).cuda()
            prediction = model.predict(inputs)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            chosen = mass.argmax(-1)
            state = prediction['state'][torch.arange(len(batch), device='cuda'), chosen // 2]
            points = rigid_points_090(state, chosen % 2, chart, (192, 192))
            error = (points - truth).norm(dim=-1).mean(-1) / 1000
            _, frames, _ = full_frame_state_to_components(state)
            angle = torch.rad2deg(torch.acos((frames[:, :, 2] * true_frames[:, :, 2]
                ).sum(-1).abs().clamp(0, 1)))
            for i, case in enumerate(batch):
                rows.append((arm, case['meta']['base'], float(error[i]), float(angle[i])))

result = {}
for arm in ('real', 'permuted'):
    arm_rows = [row for row in rows if row[0] == arm]
    error = np.array([row[2] for row in arm_rows])
    angle = np.array([row[3] for row in arm_rows])
    result[arm] = {'n': len(arm_rows), 'mean_mm': float(error.mean()),
                   'median_angle_deg': float(np.median(angle)),
                   'over5mm': int((error > 5).sum())}
result['paired_mean_gain_mm'] = result['permuted']['mean_mm'] - result['real']['mean_mm']
result['paired_median_angle_gain_deg'] = (result['permuted']['median_angle_deg']
                                          - result['real']['median_angle_deg'])
result['base_mean_gain_mm'] = {base: float(np.mean([row[2] for row in rows
    if row[0] == 'permuted' and row[1] == base]) - np.mean([row[2] for row in rows
    if row[0] == 'real' and row[1] == base])) for base in range(56, 64)}
print(result)
