"""Truth-assisted quantized construction for the fixed 073 pose lattice."""
import hashlib
import json
import math
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

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/coupled_plane_lattice_073_geometry_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
angles = torch.arange(720, device='cuda').float() * (2 * math.pi / 720)
cosine, sine = angles.cos(), angles.sin()
normal_grid = torch.unique((torch.arange(-1500, 1501, 500, device='cuda')[:, None]
    + torch.arange(-250, 251, 125, device='cuda')[None]).flatten()).float()
scale_grid = torch.unique((torch.tensor([.65, 1., 1.5], device='cuda')[:, None]
    * torch.tensor([.8, .9, 1., 1.1, 1.2], device='cuda')[None]).flatten())
shift_step, shift_limit, roll_step = .025, .8, math.pi / 36
rows = []

with torch.inference_mode():
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            tissue = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        edge = frame[0, :, :, :2] @ basis[0]
        edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
        normal = frame[0, :, :, 2]
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')), -1).float() / 256
        target = tissue.reshape(-1, 3)[ids]
        target_mean, xy_mean = target.mean(0), xy.mean(0)
        x, y = xy - xy_mean, target - target_mean
        continuous, quantized, parameters = [], [], []
        for branch in range(14):
            horizontal, vertical = edge[branch, :, 0], edge[branch, :, 1]
            b0 = x[:, 0, None] * horizontal + x[:, 1, None] * vertical
            b90 = -x[:, 1, None] * horizontal + x[:, 0, None] * vertical
            a, b = (y * b0).sum(), (y * b90).sum()
            c, d, e = b0.square().sum(), b90.square().sum(), (b0 * b90).sum()
            dot = cosine * a + sine * b
            norm = cosine.square() * c + sine.square() * d + 2 * cosine * sine * e
            index = int((dot.clamp_min(0).square() / norm.clamp_min(1e-9)).argmax())
            scale = dot[index].clamp_min(0) / norm[index]
            fitted = target_mean + scale * (cosine[index] * b0 + sine[index] * b90)
            continuous.append(float((fitted - target).norm(dim=-1).mean()))
            delta = target_mean - centre[0, branch]
            dz = delta @ normal[branch]
            plane_delta = delta - dz * normal[branch]
            shift = torch.linalg.lstsq(edge[branch], plane_delta).solution
            mean_q = xy_mean - .5
            shift -= scale * torch.stack((cosine[index] * mean_q[0] - sine[index] * mean_q[1],
                                          sine[index] * mean_q[0] + cosine[index] * mean_q[1]))
            dzq = normal_grid[(normal_grid - dz).abs().argmin()]
            angleq = (angles[index] / roll_step).round() * roll_step
            scaleq = scale_grid[(scale_grid - scale).abs().argmin()]
            shiftq = (shift / shift_step).round().clamp(-shift_limit / shift_step,
                                                       shift_limit / shift_step) * shift_step
            q = xy - .5
            chart = torch.stack((shiftq[0] + scaleq * (angleq.cos() * q[:, 0] - angleq.sin() * q[:, 1]),
                                 shiftq[1] + scaleq * (angleq.sin() * q[:, 0] + angleq.cos() * q[:, 1])), -1)
            mapped = centre[0, branch] + normal[branch] * dzq + chart @ edge[branch].T
            quantized.append(float((mapped - target).norm(dim=-1).mean()))
            parameters.append({'normal_shift_um': float(dz), 'roll_deg': float(angles[index] * 180 / math.pi),
                'scale': float(scale), 'shift_xy': shift.tolist()})
        best = int(np.argmin(quantized))
        selected = int(prior[0, choice].argmax())
        rows.append({key: record[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')})
        rows[-1].update({'support_fraction': float(valid.float().mean()),
            'best14_continuous_um': min(continuous), 'best14_quantized_um': quantized[best],
            'prior_selected_quantized_um': quantized[selected], 'best_branch': best,
            'continuous_fit_for_best_quantized_branch': parameters[best]})
        if number % 50 == 0:
            print(json.dumps({'sections': number, 'of': len(records)}), flush=True)

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
    'truth_assisted_attainable_construction': True, 'real_animal_validation': False,
    'public_benchmark_used': False}
for key in ('best14_continuous_um', 'best14_quantized_um', 'prior_selected_quantized_um'):
    summary[key] = float(np.mean([np.mean([row[key] for row in rows
        if row['synthetic_subject_plan_id'] == identity]) for identity in identities]))
summary['quantized_gate_under_750_um'] = summary['best14_quantized_um'] <= 750

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({'parent_sha256': sha(parent),
    'panel_records_sha256': sha(panel / 'records.jsonl'), 'source_sha256': sha(Path(__file__)),
    'normal_grid_um': normal_grid.tolist(), 'scale_grid': scale_grid.tolist(),
    'roll_step_degrees': 5, 'shift_step_edge_units': shift_step,
    'shift_limit_edge_units': shift_limit, 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'hashes_sha256': {name: sha(out / name)
        for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
