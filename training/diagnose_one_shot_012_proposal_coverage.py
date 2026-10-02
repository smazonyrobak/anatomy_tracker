"""Physical oracle coverage after bounded expansion of actual predicted planes."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint_path = root / 'runs/one_shot_positive_fit_feedback_012/joint_step_06000.pt'
out = root / 'runs/one_shot_positive_fit_feedback_012_proposal_coverage'
side = 256
torch.set_num_threads(4)
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
updates = torch.zeros((35, 9), device='cuda')
for axis in range(3):
    for j, step in enumerate((500., -500., 2000., -2000.)):
        updates[1 + 4 * axis + j, 3 + axis] = step
    for j, step in enumerate((.12, -.12, .35, -.35)):
        updates[13 + 4 * axis + j, axis] = step
for axis in range(2):
    for j, step in enumerate((.05, -.05, .12, -.12)):
        updates[25 + 4 * axis + j, 6 + axis] = step
updates[33, 8], updates[34, 8] = .05, -.05

out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda().reshape(-1, 3)
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool().flatten()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        top = prior.topk(8).indices
        base = prediction['state'][0, :, None].expand(-1, 2, -1).reshape(32, 12)
        points = valid.nonzero().flatten()
        chart = torch.stack((points.remainder(side),
                             points.div(side, rounding_mode='floor')), -1).float() / side
        truth = target[points]
        errors = []
        for branch in range(32):
            states = compose_full_frame_state(base[branch].expand(len(updates), -1), updates)
            centre, frame, basis = full_frame_state_to_components(states)
            local_chart = chart.clone()
            if branch % 2:
                local_chart[:, 0] = (side - 1) / side - local_chart[:, 0]
            surface = centre[:, None] + torch.einsum(
                'kij,pj->kpi', frame[:, :, :2] @ basis, local_chart - .5)
            errors.append((surface - truth[None]).norm(dim=-1).mean(-1))
        error = torch.stack(errors)
        row = {'animal_id': record['animal_id'], 'section_id': record['section_id'],
               'sha256': record['sha256'], 'valid_pixels': record['valid_pixels'],
               'prior_rigid_um': float(error[int(prior.argmax()), 0]),
               'top8_base_oracle_um': float(error[top, 0].min()),
               'all32_base_oracle_um': float(error[:, 0].min()),
               'top8_expanded_oracle_um': float(error[top].min()),
               'all32_expanded_oracle_um': float(error.min()),
               'all32_best_branch': int(error.min(dim=1).values.argmin()),
               'all32_best_update': int(error.flatten().argmin().remainder(len(updates)))}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    stream.flush()

identities = sorted({row['animal_id'] for row in rows})
metrics = ('prior_rigid_um', 'top8_base_oracle_um', 'all32_base_oracle_um',
           'top8_expanded_oracle_um', 'all32_expanded_oracle_um')
summary = {'rows': len(rows), 'identities': len(identities),
           'updates_per_branch': len(updates),
           'candidate_branches': 32,
           'perturbations': 'base; ±0.5/2 mm local translations; ±0.12/0.35 rad local rotations; ±0.05/0.12 log-scale; ±0.05 shear',
           'identity_equal_mean_um': {metric: float(np.mean([
               np.mean([row[metric] for row in rows if row['animal_id'] == identity])
               for identity in identities])) for metric in metrics},
           'public_benchmark_used': False, 'calibrated': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'rows': len(rows), 'public_benchmark_used': False, 'calibrated': False}, indent=2))
print(json.dumps(summary), flush=True)
