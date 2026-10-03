"""Separate pose quality from branch probability after the completed 058 run."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_anchor_joint_058'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_anchor_joint_058_capture_diagnostic'
steps = (0, 2000, 6000, 12000)
side = 256
torch.set_num_threads(4)
assert json.loads((run / 'completed.json').read_text())['batches'] == 12000
synthetic_records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 185 and len(real_records) == 64
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz') as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
                               fit_quality=True, vector_refinement=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic_records:
            with np.load(panel / record['file']) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            selected = valid.flatten().nonzero().flatten()
            chosen = selected[torch.linspace(0, len(selected) - 1, 256).round().long()]
            chart = torch.stack((chosen.remainder(side),
                                 chosen.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[chosen]
            prediction = model.predict(image)
            state = prediction['state'][0].repeat_interleave(2, 0)
            reflected = torch.arange(2, device='cuda').repeat(model.modes)
            error = (points(state, reflected, chart) - target).norm(dim=-1).mean(-1)
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit'][0]),
                                          F.logsigmoid(prediction['reflection_logit'][0])), -1)
            prior = (prediction['log_mass'][0, :, None] + log_reflection).flatten()
            old, anchor = slice(0, 32), slice(32, 160)
            top_old = prior[old].topk(8).indices
            top_anchor = prior[anchor].topk(8).indices + 32
            top_all = prior.topk(8).indices
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id')},
                   'selected_old_um': float(error[prior[old].argmax()]),
                   'best_old_um': float(error[old].min()),
                   'selected_anchor_um': float(error[32 + prior[anchor].argmax()]),
                   'best_anchor_um': float(error[anchor].min()),
                   'best8_old_um': float(error[top_old].min()),
                   'best8_anchor_um': float(error[top_anchor].min()),
                   'best8_all_um': float(error[top_all].min()),
                   'top8_anchor_count': int((top_all >= 32).sum()),
                   'anchor_log_mass_max': float(prediction['log_mass'][0, 16:].max()),
                   'old_log_mass_max': float(prediction['log_mass'][0, :16].max())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            target = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            state = prediction['state'][0].repeat_interleave(2, 0)
            reflected = torch.arange(2, device='cuda').repeat(model.modes)
            error = (points(state, reflected, corners) - target).norm(dim=-1).mean(-1)
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit'][0]),
                                          F.logsigmoid(prediction['reflection_logit'][0])), -1)
            prior = (prediction['log_mass'][0, :, None] + log_reflection).flatten()
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id')},
                   'selected_old_um': float(error[prior[:32].argmax()]),
                   'best_old_um': float(error[:32].min()),
                   'selected_anchor_um': float(error[32 + prior[32:].argmax()]),
                   'best_anchor_um': float(error[32:].min())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': len(synthetic_records),
                          'real': len(real_records)}), flush=True)

summary = []
for step in steps:
    for split in ('synthetic', 'real_weak_allen'):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        names = [key for key in group[0] if key.endswith('_um') or key in
                 ('top8_anchor_count', 'anchor_log_mass_max', 'old_log_mass_max')]
        identities = sorted({row['animal_id'] for row in group})
        summary.append({'step': step, 'set': split, 'rows': len(group),
                        'identity_equal_mean': {name: float(np.mean([
                            np.mean([row[name] for row in group if row['animal_id'] == identity])
                            for identity in identities])) for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoints_sha256': {str(step): hashlib.sha256(
        (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
    'panel_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
