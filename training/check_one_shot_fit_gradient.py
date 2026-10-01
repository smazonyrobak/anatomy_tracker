"""One focused check: atlas fitting alone must update the direct pose head."""
import math
import os
import sys

os.environ['TEMP'] = os.environ['TMP'] = 'I:/AnatomyTracker/tmp'
sys.dont_write_bytecode = True

import torch

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

torch.set_num_threads(4)
context = load_streaming_synthetic_v7(device='cuda')
for seed in range(202610011, 202610043):
    batch = sample_one_shot_stream(context, [0], seed, side=256)
    if batch['eligible'][0]:
        break
model = OneShotJointSliceModel().cuda()
truth = batch['state'][0]
with torch.no_grad():
    bias = model.pose[-1].bias.view(model.modes, 21)
    bias[0, :3] = (truth[:3] - model.center_origin) / model.center_scale
    bias[0, 3:9] = truth[3:9]
    bias[0, 9:11] = truth[9:11] - math.log(12000.)
    bias[0, 11] = truth[11]
output = model(batch['inputs'], batch['offsets'],
               mode_index=torch.zeros(1, dtype=torch.long, device='cuda'),
               reflection=batch['reflection'])
fit = model.atlas_fit_loss(batch['inputs'], output, context['atlas'],
                           batch['weights'], batch['valid_mask'])[0, 0]
gradient = torch.autograd.grad(fit, model.pose[-1].weight)[0]
print({'seed': seed, 'visible_mass': float(batch['visible'].sum()),
       'atlas_fit_loss': float(fit.detach()),
       'pose_head_fit_gradient_norm': float(gradient.norm()),
       'finite_gradient': bool(torch.isfinite(gradient).all())}, flush=True)
