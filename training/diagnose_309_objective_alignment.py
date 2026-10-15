"""Read-only comparison of the frozen 309 rank target with its reported site metric."""
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

run = root / 'runs/first_choice_exposure_309_001'
panel = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
draws = (json.loads(line) for line in (run / 'panel_draws.jsonl').open(encoding='utf8'))
reflection = {row['provenance']['physical_section_id']:
              int(row['provenance']['horizontal_reflection']) for row in draws
              if row['cohort'] == 'v10' and row['accepted']}
assert len(panel) == len(reflection) == 256
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt', device='cuda')
model.eval()
torch.set_num_threads(4)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / 192
flags = torch.arange(2, device='cuda')[None, None]
rows = []

with torch.inference_mode():
    for start in range(0, len(panel), 4):
        group = panel[start:start + 4]
        image = torch.stack([case['inputs'] for case in group]).cuda()
        chart = torch.stack([case['chart'] for case in group]).cuda()
        truth = torch.stack([case['truth'] for case in group]).cuda()
        state = torch.stack([case['state'] for case in group]).cuda()
        true_flag = torch.tensor([reflection[case['meta']['physical_section_id']]
                                  for case in group], device='cuda')[:, None, None]
        prediction = model.predict(image)
        proposed = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        proposed_flag = flags.expand(len(group), model.modes, 2)
        mass = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        tissue = (rigid_points_090(proposed, proposed_flag, chart[:, None, None],
                                  (192, 192)) - truth[:, None, None]).norm(dim=-1).mean(-1)
        true_five = rigid_points_090(state[:, None, None], true_flag, corners,
                                    (192, 192))
        five = (rigid_points_090(proposed, proposed_flag, corners, (192, 192))
                - true_five).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(state)[1][..., :, 2]
        normal_cost = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        physical = (.75 * tissue + .25 * five + normal_cost[..., None]).flatten(1) / 1000
        tissue = tissue.flatten(1) / 1000
        for j, case in enumerate(group):
            top = mass[j].topk(16).indices
            tissue_top = tissue[j, top]
            physical_top = physical[j, top]
            chosen = mass[j].argmax()
            rows.append({'id': case['meta']['physical_section_id'],
                         'first_mm': float(tissue[j, chosen]),
                         'top16_tissue_best_mm': float(tissue_top.min()),
                         'top16_physical_best_tissue_mm': float(tissue_top[physical_top.argmin()]),
                         'all_tissue_best_mm': float(tissue[j].min()),
                         'all_physical_best_tissue_mm': float(tissue[j, physical[j].argmin()]),
                         'top16_same_choice': bool(tissue_top.argmin() == physical_top.argmin())})

assert abs(np.mean([row['first_mm'] for row in rows]) - 2.469) < .005
assert abs(np.mean([row['top16_tissue_best_mm'] for row in rows]) - .931) < .005
for key in ('first_mm', 'top16_tissue_best_mm', 'top16_physical_best_tissue_mm',
            'all_tissue_best_mm', 'all_physical_best_tissue_mm'):
    print(key, float(np.mean([row[key] for row in rows])))
print('top16_same_choice', sum(row['top16_same_choice'] for row in rows), '/', len(rows))
print('top16_objective_mismatch_penalty_mm', float(np.mean([
    row['top16_physical_best_tissue_mm'] - row['top16_tissue_best_mm'] for row in rows])))
