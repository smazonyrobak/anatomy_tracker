"""Frozen 309 pose-head equivariance and blind four-view medoid check."""
import json
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

SIDE = 192
PARENT = ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'
PANEL = ROOT / 'runs/red_source_diversity_319_001/panels.pt'
RUN = ROOT / 'runs/rotation_consensus_320_001'
assert not RUN.exists()
RUN.mkdir(parents=True)
(RUN / 'protocol.json').write_text(json.dumps({
    'checkpoint': str(PARENT), 'panel': str(PANEL),
    'scope': '128 frozen TRAIN-inner CCF virtual cuts, bases 56-63; no new training or real physical-oblique claim',
    'transform': 'four exact raster quarter-turns; compare predicted physical coordinates at identical original sites',
    'blind_selector': 'medoid of four first-choice physical maps by mean pairwise distance at 128 saved sites',
    'gate': 'mean first-choice gain >=0.35 mm and no increase in >5-mm cases; passage licenses a separate acquired-animal guard, not promotion'
}, indent=2) + '\n')

torch.set_num_threads(4)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    cases = torch.load(PANEL, map_location='cpu', weights_only=True)['ccf']
assert len(cases) == 128
model, _ = load_one_shot_checkpoint(PARENT, device='cuda')
model.eval()
flags = torch.arange(2, device='cuda')[None, None]
rows = []
with torch.inference_mode():
    for start in range(0, len(cases), 4):
        group = cases[start:start + 4]
        inputs = torch.stack([case['inputs'] for case in group]).cuda()
        chart = torch.stack([case['chart'] for case in group]).cuda()
        truth = torch.stack([case['truth'] for case in group]).cuda()
        x, y = chart.unbind(-1)
        c = (SIDE - 1) / SIDE
        charts = (chart, torch.stack((y, c - x), -1),
                  torch.stack((c - x, c - y), -1), torch.stack((c - y, x), -1))
        maps = []
        for k, rotated_chart in enumerate(charts):
            prediction = model.predict(torch.rot90(inputs, k, (-2, -1)))
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(group), model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            points = rigid_points_090(states, reflections,
                rotated_chart[:, None, None], (SIDE, SIDE)).flatten(1, 2)
            maps.append(points[torch.arange(len(group), device='cuda'), mass.argmax(-1)])
        maps = torch.stack(maps, 1)
        pairwise = (maps[:, :, None] - maps[:, None]).norm(dim=-1).mean(-1) / 1000
        medoid = pairwise.sum(-1).argmin(-1)
        errors = (maps - truth[:, None]).norm(dim=-1).mean(-1) / 1000
        for i, case in enumerate(group):
            rows.append({'base': case['base'], 'physical_section_id': case['physical_section_id'],
                'first_mm': float(errors[i, 0]), 'medoid_mm': float(errors[i, medoid[i]]),
                'four_view_oracle_mm': float(errors[i].min()),
                'medoid_view': int(medoid[i]),
                'mean_pairwise_mm': float(pairwise[i].triu(1).sum() / 6)})

first = np.array([row['first_mm'] for row in rows])
medoid = np.array([row['medoid_mm'] for row in rows])
decision = {'n': len(rows), 'first_mean_mm': float(first.mean()),
    'medoid_mean_mm': float(medoid.mean()), 'gain_mm': float(first.mean() - medoid.mean()),
    'first_gt5': int((first > 5).sum()), 'medoid_gt5': int((medoid > 5).sum()),
    'four_view_oracle_mean_mm': float(np.mean([row['four_view_oracle_mm'] for row in rows])),
    'median_pairwise_mm': float(np.median([row['mean_pairwise_mm'] for row in rows]))}
decision['pass'] = decision['gain_mm'] >= .35 and decision['medoid_gt5'] <= decision['first_gt5']
(RUN / 'rows.json').write_text(json.dumps(rows) + '\n')
(RUN / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
print(json.dumps(decision), flush=True)
