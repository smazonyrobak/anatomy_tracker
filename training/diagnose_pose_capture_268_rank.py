"""Read-only branch-rank diagnosis on the frozen 268 TRAIN-inner panel."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = Path('I:/AnatomyTracker/runs/pose_capture_268_v8_train_inner_002')
panel = torch.load(run / 'inner_panel.pt', map_location='cpu', weights_only=False)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(run / 'pose_step_08000.pt', map_location='cpu',
                                 weights_only=False)['model'])
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / 192
flags = torch.arange(2, device='cuda')[None, None].expand(1, 80, 2)
counts = {f'image_top_{k}': 0 for k in (1, 2, 16, 32, 64, 160)}
counts.update(physical_top_1=0, physical_top_16=0, site_oracle=0)
with torch.inference_mode():
    for item in panel:
        state = item['state'][None].cuda()
        reflection = item['reflection'][None].cuda()
        chart = item['chart'].cuda()
        prediction = model.predict(item['inputs'][None].cuda())
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        candidates = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        truth = rigid_points_090(state, reflection, chart, (192, 192))
        proposed = rigid_points_090(candidates, flags, chart[None, None], (192, 192))
        site = ((proposed - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten()
        truth_five = rigid_points_090(state, reflection, corners, (192, 192))
        proposed_five = rigid_points_090(candidates, flags, corners, (192, 192))
        five = ((proposed_five - truth_five[:, None, None]).norm(dim=-1).mean(-1)
                / 1000).flatten()
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(state)[1][..., :, 2]
        normal = (4 * (1 - (normals * true_normal[:, None]).sum(-1).abs())
                  ).repeat_interleave(2).flatten()
        physical = .75 * site + .25 * five + normal
        site_hit = site <= 1.5
        image_order = prior.argsort(descending=True)
        physical_order = physical.argsort()
        for k in (1, 2, 16, 32, 64, 160):
            counts[f'image_top_{k}'] += int(site_hit[image_order[:k]].any())
        counts['physical_top_1'] += int(site_hit[physical_order[0]])
        counts['physical_top_16'] += int(site_hit[physical_order[:16]].any())
        counts['site_oracle'] += int(site_hit.any())
print(json.dumps({'panel_cases': len(panel), **counts}))
