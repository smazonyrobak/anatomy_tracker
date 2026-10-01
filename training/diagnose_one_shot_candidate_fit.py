"""Check whether the frozen fitter can rank the model's global pose candidates."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7

RUN = ROOT / 'runs/one_shot_joint_pose_curriculum_002'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUT = ROOT / 'runs/one_shot_candidate_fit_001'
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 10000
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
atlas = load_streaming_synthetic_v7(device='cuda')['atlas']
checkpoint = torch.load(RUN / 'joint_step_10000.pt', map_location='cpu', weights_only=True)
model = OneShotJointSliceModel().cuda().eval()
model.load_state_dict(checkpoint['model'])
chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device='cuda') / 256
rows = []

with torch.inference_mode():
    for record in map(json.loads, (DEV / 'records.jsonl').read_text().splitlines()):
        with np.load(DEV / record['file']) as arrays:
            appearance = record['section_index'] % 3
            if not arrays['eligible'][appearance]:
                continue
            image = torch.from_numpy(arrays['inputs'][appearance:appearance + 1].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            reflected = int(arrays['reflection'])
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            valid = torch.from_numpy((arrays['visible_support'][appearance:appearance + 1] > .25).copy()).cuda()
        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        valid = F.interpolate(valid[:, None].float(), (256, 256), mode='nearest')[:, 0] > .5
        prediction = model.predict(image)
        prior = prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)
        centre, frame, basis = full_frame_state_to_components(truth)
        reference_chart = chart.clone()
        if reflected:
            reference_chart[:, 0] = 255 / 256 - reference_chart[:, 0]
        reference = centre[:, None] + torch.einsum('bij,pj->bpi', frame[:, :, :2] @ basis, reference_chart - .5)
        candidate_state = prediction['state'][0, :, None].expand(-1, 2, -1)
        centre, frame, basis = full_frame_state_to_components(candidate_state)
        candidate_chart = chart.expand(model.modes, 2, -1, -1).clone()
        candidate_chart[:, 1, :, 0] = 255 / 256 - candidate_chart[:, 1, :, 0]
        candidate = centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
            frame[..., :, :2] @ basis, candidate_chart - .5)
        pose_error = (candidate - reference).norm(dim=-1).mean(-1)
        fit = torch.empty(model.modes, 2, device='cuda')
        for mode in range(model.modes):
            indices = torch.tensor([[mode, mode]], device='cuda')
            flags = torch.tensor([[0, 1]], device='cuda')
            mapped = model.map(prediction, offsets, indices, flags, (256, 256))
            fit[mode] = model.atlas_fit_loss(image, mapped, atlas, weights, valid)[0]
        prior_choice = int(prior.flatten().argmax())
        fit_choice = int(fit.flatten().argmin())
        rows.append({'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                     'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                     'appearance_mode': appearance, 'truth_reflection': reflected,
                     'prior_log_weight': prior.cpu().tolist(), 'fit_mismatch': fit.cpu().tolist(),
                     'pose_error_um': pose_error.cpu().tolist(),
                     'prior_error_um': float(pose_error.flatten()[prior_choice]),
                     'fit_selected_error_um': float(pose_error.flatten()[fit_choice]),
                     'oracle_error_um': float(pose_error.min()),
                     'fit_oracle_rank': int(np.argsort(fit.flatten().cpu().numpy()).tolist().index(
                         int(pose_error.flatten().argmin()))) + 1})

with (OUT / 'rows.jsonl').open('w') as output:
    output.writelines(json.dumps(row) + '\n' for row in rows)
summary = {'sections': len(rows), 'donors': len({row['animal_id'] for row in rows}),
           **{key: float(np.mean([row[key] for row in rows])) for key in
              ('prior_error_um', 'fit_selected_error_um', 'oracle_error_um', 'fit_oracle_rank')},
           'fit_better_than_prior_fraction': float(np.mean([
               row['fit_selected_error_um'] < row['prior_error_um'] for row in rows]))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
