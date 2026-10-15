"""Read-only confidence-versus-error audit on frozen TRAIN-inner sections."""
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

from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/first_choice_exposure_309_001'
panels = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)
assert model.spatial_feedback

for cohort in ('v10', 'v10_sparse', 'v10_steep'):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(panels[cohort]), 4):
            cases = panels[cohort][start:start + 4]
            image = torch.stack([case['inputs'] for case in cases]).cuda()
            chart = torch.stack([case['chart'] for case in cases]).cuda()
            truth = torch.stack([case['truth'] for case in cases]).cuda()
            prediction = model.predict(image)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflection = torch.arange(2, device='cuda')[None, None].expand(
                len(cases), model.modes, -1)
            mapped = rigid_points_090(state, reflection, chart[:, None, None], (192, 192))
            error = ((mapped - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten(1)
            selected = mass.argmax(-1)
            probability = mass.exp()
            entropy = -(probability * mass).sum(-1) / np.log(mass.shape[-1])
            margin = mass.topk(2, -1).values.diff(dim=-1).abs().squeeze(-1)
            reliability = prediction['reliability_logit'].sigmoid().mean((1, 2, 3))
            for i, case in enumerate(cases):
                rows.append((float(error[i, selected[i]]), float(probability[i].max()),
                             float(1 - entropy[i]), float(margin[i]),
                             case['meta']['postartifact_visible_fraction'],
                             float(reliability[i]), float(case['inputs'][2, 0, 0])))

    values = np.asarray(rows)
    if cohort == 'v10':
        assert abs(values[:, 0].mean() - 2.4693877) < 1e-4
    print(cohort, 'n=', len(values), 'mean_mm=', values[:, 0].mean(),
          'median_mm=', np.median(values[:, 0]), 'over5=', int((values[:, 0] > 5).sum()))
    for label, column in (('max_branch_mass', 1), ('inverse_entropy', 2),
                          ('top_two_log_margin', 3), ('oracle_visible_fraction', 4),
                          ('predicted_reliable_fraction', 5)):
        ranking = values[np.argsort(-values[:, column])]
        print(label, 'score_p10_p50_p90=', np.percentile(values[:, column], (10, 50, 90)),
              'accepted_mean_mm=', [round(float(ranking[:n, 0].mean()), 3)
                                   for n in (len(values) // 4, len(values) // 2,
                                             3 * len(values) // 4, len(values))],
              'accepted_over5=', [int((ranking[:n, 0] > 5).sum())
                                 for n in (len(values) // 4, len(values) // 2,
                                           3 * len(values) // 4, len(values))])
    for label, subset in (('raw_no_brush', values[values[:, 6] == 0]),
                          ('brush_available', values[values[:, 6] == 1])):
        selected = subset[np.argsort(-subset[:, 5])][:len(subset) // 2]
        print(label, 'n=', len(subset), 'mean_mm=', subset[:, 0].mean(),
              'over5=', int((subset[:, 0] > 5).sum()),
              'reliability_top_half_mean_mm=', selected[:, 0].mean(),
              'reliability_top_half_over5=', int((selected[:, 0] > 5).sum()))
