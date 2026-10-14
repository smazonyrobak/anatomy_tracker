"""Privileged Allen-label information check on frozen 309 TRAIN-inner candidates."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import nrrd
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import ANNOTATION_PATH_V6
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

RUN = ROOT / 'runs/first_choice_exposure_309_001'
MODEL = RUN / 'single/pose_local_20000.pt'
SIDE = 192
annotation, _ = nrrd.read(str(ANNOTATION_PATH_V6), index_order='F')
panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)['v10']
model, _ = load_one_shot_checkpoint(MODEL, device='cuda')
torch.set_num_threads(4)
offsets = np.array([(a, b, c) for a in range(-3, 4) for b in range(-3, 4)
                    for c in range(-3, 4) if a*a + b*b + c*c <= 9], dtype=np.int64)
shape = np.asarray(annotation.shape)
flags = torch.arange(2, device='cuda')[None, None]
rng = np.random.default_rng(20261015309)
rows = []

with torch.inference_mode():
    for start in range(0, len(panels), 4):
        group = panels[start:start + 4]
        image = torch.stack([case['inputs'] for case in group]).cuda()
        chart = torch.stack([case['chart'] for case in group]).cuda()
        truth = torch.stack([case['truth'] for case in group]).cuda()
        pred = model.predict(image)
        states = pred['state'][:, :, None].expand(-1, -1, 2, -1)
        reflection = flags.expand(len(group), model.modes, 2)
        mass = (pred['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-pred['reflection_logit']),
            F.logsigmoid(pred['reflection_logit'])), -1)).flatten(1)
        top = mass.topk(16, -1).indices
        points = rigid_points_090(states, reflection, chart[:, None, None],
                                  (SIDE, SIDE)).flatten(1, 2)
        errors = (points - truth[:, None]).norm(dim=-1).mean(-1) / 1000
        for j, case in enumerate(group):
            indices = top[j].cpu().numpy()
            candidate_points = points[j, top[j]].cpu().numpy()
            true_points = truth[j].cpu().numpy()
            true_voxels = np.rint(true_points / 25 - .5).astype(np.int64)
            true_inside = ((true_voxels >= 0) & (true_voxels < shape)).all(-1)
            true_clipped = np.clip(true_voxels, 0, shape - 1)
            labels = annotation[tuple(true_clipped.T)]
            valid = true_inside & (labels != 0)
            voxel = np.rint(candidate_points / 25 - .5).astype(np.int64)
            neighbouring = voxel[:, :, None, :] + offsets[None, None]
            inside = ((neighbouring >= 0) & (neighbouring < shape)).all(-1)
            clipped = np.clip(neighbouring, 0, shape - 1)
            candidates = annotation[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
            hits = ((candidates == labels[None, :, None]) & inside).any(-1)
            support = ((candidates != 0) & inside).any(-1)[:, valid].mean(-1)
            target = labels[valid]
            unique, counts = np.unique(target, return_counts=True)
            large = unique[counts >= 4]
            groups = [target == label for label in large]
            rare = ~np.isin(target, large)
            if rare.any():
                groups.append(rare)
            semantic = np.stack([hits[:, valid][:, mask].mean(-1) for mask in groups], -1).mean(-1)
            shuffled = labels.copy()
            shuffled[valid] = rng.permutation(target)
            shuffled_hits = ((candidates == shuffled[None, :, None]) & inside).any(-1)
            shuffled_score = np.stack([shuffled_hits[:, valid][:, mask].mean(-1)
                                       for mask in groups], -1).mean(-1)
            prior = mass[j, top[j]].cpu().numpy()
            choices = {'prior': 0,
                       'semantic': int(np.lexsort((-prior, -semantic))[0]),
                       'support': int(np.lexsort((-prior, -support))[0]),
                       'shuffled': int(np.lexsort((-prior, -shuffled_score))[0])}
            cost = errors[j, top[j]].cpu().numpy()
            rows.append({'physical_section_id': case['meta']['physical_section_id'],
                         'valid_label_sites': int(valid.sum()),
                         'distinct_labels': int(len(unique)),
                         'oracle_mm': float(cost.min()),
                         **{arm + '_mm': float(cost[index]) for arm, index in choices.items()},
                         **{arm + '_branch': int(indices[index]) for arm, index in choices.items()}})

for arm in ('prior', 'semantic', 'support', 'shuffled'):
    values = np.array([row[arm + '_mm'] for row in rows])
    print(json.dumps({'arm': arm, 'n': len(values), 'mean_mm': float(values.mean()),
                      'median_mm': float(np.median(values)),
                      'over_5mm': int((values > 5).sum())}))
print(json.dumps({'arm': 'oracle', 'n': len(rows),
                  'mean_mm': float(np.mean([row['oracle_mm'] for row in rows]))}))
print(json.dumps({'label_sites_min': min(row['valid_label_sites'] for row in rows),
                  'distinct_labels_median': float(np.median([row['distinct_labels'] for row in rows])),
                  'gate_semantic_gain_ge_0p35mm': bool(
                      np.mean([row['semantic_mm'] for row in rows])
                      <= np.mean([row['prior_mm'] for row in rows]) - .35),
                  'gate_no_tail_regression': bool(
                      sum(row['semantic_mm'] > 5 for row in rows)
                      <= sum(row['prior_mm'] > 5 for row in rows))}))
