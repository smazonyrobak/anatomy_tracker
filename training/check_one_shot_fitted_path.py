"""Single-case gradient check for the fitted candidate path."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']][:2]
with np.load(panel / records[0]['file'], allow_pickle=False) as first, np.load(
        panel / records[1]['file'], allow_pickle=False) as second:
    image = torch.from_numpy(np.stack((first['inputs'], second['inputs']))).cuda()
    offsets = torch.from_numpy(np.stack((first['offsets_um'], second['offsets_um']))).cuda()
    weights = torch.from_numpy(np.stack((first['weights'], second['weights']))).cuda()
checkpoint = torch.load(root / 'runs/one_shot_on_policy_atlas_rank_009/atlas/joint_step_08000.pt',
                        map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda()
model.load_state_dict(checkpoint['model'], strict=False)
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
prediction = model.predict(image)
prior = (prediction['log_mass'][..., None] + torch.stack((
    F.logsigmoid(-prediction['reflection_logit']),
    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
choice = prior.topk(8, -1).indices
mapped = model.map(prediction, offsets, choice // 2, choice % 2, (96, 96), atlas, weights,
                   feature_side=96, source_shape=(256, 256))
score = model.score_fitted_candidates(image, prediction, mapped, atlas, weights)
score.mean().backward()
print(json.dumps({'score_shape': list(score.shape),
                  'pose_gradient': float(model.pose[-1].weight.grad.norm()),
                  'fitted_head_gradient': float(model.fitted_matcher[-1].weight.grad.norm()),
                  'finite': bool(torch.isfinite(score).all())}), flush=True)
