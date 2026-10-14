"""Frozen TRAIN-inner diagnostic: scramble tissue or exterior pixels, preserving the other."""
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

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/first_choice_exposure_309_001'
cases = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
context = load_fov_balance_v10('cuda')
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)
generator = torch.Generator().manual_seed(20261015)
rows = []

with torch.inference_mode():
    for start in range(0, len(cases), 4):
        batch = cases[start:start + 4]
        variants = {'intact': [], 'tissue_scrambled': [], 'exterior_scrambled': []}
        for case in batch:
            meta = case['meta']
            sample = sample_fov_balance_v10(context, [meta['base'] * VARIANTS + meta['variant']],
                                            meta['seed'], 192)
            assert torch.equal(sample['inputs'][0].cpu(), case['inputs'])
            tissue = (sample['visible'][0] > .05)
            if bool(case['inputs'][2, 0, 0]):
                tissue &= sample['brush_mask'][0]
            tissue = tissue.cpu().flatten()
            variants['intact'].append(case['inputs'])
            for arm, mask in (('tissue_scrambled', tissue), ('exterior_scrambled', ~tissue)):
                image = case['inputs'].clone()
                indices = torch.nonzero(mask).flatten()
                values = image[0].flatten()
                values[indices] = values[indices[torch.randperm(len(indices), generator=generator)]]
                variants[arm].append(image)
        chart = torch.stack([case['chart'] for case in batch]).cuda()
        truth = torch.stack([case['truth'] for case in batch]).cuda()
        true_state = torch.stack([case['state'] for case in batch]).cuda()
        true_normal = full_frame_state_to_components(true_state)[1][:, :, 2]
        for arm, images in variants.items():
            prediction = model.predict(torch.stack(images).cuda())
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            first = mass.argmax(-1)
            state = prediction['state'][torch.arange(len(batch), device='cuda'), first // 2]
            sites = rigid_points_090(state, first % 2, chart, (192, 192))
            error = (sites - truth).norm(dim=-1).mean(-1) / 1000
            normal = full_frame_state_to_components(state)[1][:, :, 2]
            angle = torch.rad2deg(torch.acos((normal * true_normal).sum(-1).abs().clamp(0, 1)))
            for i, case in enumerate(batch):
                rows.append((arm, case['meta']['base'], bool(case['inputs'][2, 0, 0]),
                             int(first[i]), float(error[i]), float(angle[i])))

for arm in variants:
    subset = [row for row in rows if row[0] == arm]
    error = np.array([row[4] for row in subset])
    angle = np.array([row[5] for row in subset])
    print(arm, {'n': len(subset), 'mean_mm': float(error.mean()),
                'median_normal_deg': float(np.median(angle)), 'over5mm': int((error > 5).sum()),
                'raw_mean_mm': float(np.mean([row[4] for row in subset if not row[2]])),
                'brush_mean_mm': float(np.mean([row[4] for row in subset if row[2]]))})
