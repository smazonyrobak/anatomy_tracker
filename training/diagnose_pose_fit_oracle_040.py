"""Isolate correspondence learning from geometric fitting on frozen synthetic DEV."""
import hashlib
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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
out = root / 'runs/pose_feedback_3d_oracle_fit_040'
side, beam = 256, 8
torch.set_num_threads(4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi',
        frame[..., :2] @ basis, chart - .5)


def fit(chart, target, weight, prior):
    ridge = torch.diag(prior.new_tensor((2., 1., 1.)))

    def solve(weight):
        lhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, weight, chart) + ridge
        rhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, weight, target) + ridge @ prior
        return torch.linalg.solve(lhs, rhs)

    coefficients = solve(weight)
    residual = (chart @ coefficients - target).norm(dim=-1)
    coefficients = solve(weight * (1500 / residual.clamp_min(1500)))
    centre, u, v = coefficients.unbind(-2)
    origin = centre - .5 * (u + v)
    return full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((origin, u, v), -2)))


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
out.mkdir(parents=True, exist_ok=False)
config = {'purpose': 'synthetic-development oracle geometric upper bound, not a deployable estimator',
          'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
          'source_sha256': sha(Path(__file__)), 'eligible_sections': len(records),
          'beam': beam, 'chart_side': 16, 'atlas_keys': '13x24x24 at 1mm depth and 2x plane field',
          'weights': 'exact visible tissue, no predicted confidence',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = (torch.arange(16, device='cuda') + .5) / 16
qy, qx = torch.meshgrid(axis, axis, indexing='ij')
atlas_axis = (torch.arange(24, device='cuda') + .5) / 12 - .5
ay, ax = torch.meshgrid(atlas_axis, atlas_axis, indexing='ij')
depth = torch.linspace(-6000., 6000., 13, device='cuda')
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior_logits = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = prior_logits.topk(beam, -1).indices
        states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(states)
        edges = frame[..., :, :2] @ basis
        qchart = torch.stack((qx, qy), -1).reshape(1, 1, 256, 2).expand(
            1, beam, -1, -1).clone()
        qchart[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / 256 - qchart[..., 0], qchart[..., 0])
        design = torch.cat((torch.ones_like(qchart[..., :1]), qchart - .5), -1)
        coefficients = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
        truth_grid = target[8::16, 8::16].reshape(1, 256, 3).expand(1, beam, -1, -1)
        valid_grid = valid[8::16, 8::16].reshape(1, 256).expand(1, beam, -1)
        continuous = fit(design, truth_grid, valid_grid.float(), coefficients)
        ax_grid = ax[None, None].expand(1, beam, -1, -1)
        ay_grid = ay[None, None].expand_as(ax_grid)
        key_chart = torch.stack((torch.where(reflection[..., None, None].bool(),
            255 / 256 - ax_grid, ax_grid), ay_grid), -1)
        key_base = centre[..., None, None, :] + torch.einsum(
            'bkij,bkhwj->bkhwi', edges, key_chart - .5)
        keys = (key_base[:, :, None] + depth[None, None, :, None, None, None] *
                frame[..., None, None, None, :, 2]).reshape(1, beam, -1, 3)
        key_distance = torch.cdist(truth_grid[0], keys[0])
        nearest_distance, nearest = key_distance.min(-1)
        nearest_world = keys[0].gather(1, nearest[..., None].expand(-1, -1, 3))[None]
        in_range = valid_grid & (nearest_distance[None] <= 1500)
        discrete = fit(design, nearest_world, in_range.float(), coefficients)
        ids = valid.flatten().nonzero().flatten()
        chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                            -1).float()[None] / side
        truth = target.reshape(-1, 3)[ids]
        before = (points(states, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
        continuous_error = (points(continuous, reflection, chart) - truth[None, None]
                            ).norm(dim=-1).mean(-1)[0]
        discrete_error = (points(discrete, reflection, chart) - truth[None, None]
                          ).norm(dim=-1).mean(-1)[0]
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
               'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update({'prior_top1_um': float(before[0]), 'prior_best8_um': float(before.min()),
                    'continuous_oracle_top1_um': float(continuous_error[0]),
                    'continuous_oracle_best8_um': float(continuous_error.min()),
                    'discrete_oracle_top1_um': float(discrete_error[0]),
                    'discrete_oracle_best8_um': float(discrete_error.min()),
                    'top1_in_range_valid_fraction': float(in_range[0, 0].sum() /
                        valid_grid[0, 0].sum().clamp_min(1)),
                    'best8_in_range_valid_fraction': float(in_range[0, int(before.argmin())].sum() /
                        valid_grid[0, int(before.argmin())].sum().clamp_min(1))})
        rows.append(row)
        stream.write(json.dumps(row) + '\n')


def equal_subject_mean(field):
    groups = sorted({row['synthetic_subject_plan_id'] for row in rows})
    return float(np.mean([np.mean([row[field] for row in rows
                                   if row['synthetic_subject_plan_id'] == group])
                          for group in groups]))


fields = ('prior_top1_um', 'prior_best8_um', 'continuous_oracle_top1_um',
          'continuous_oracle_best8_um', 'discrete_oracle_top1_um',
          'discrete_oracle_best8_um', 'top1_in_range_valid_fraction',
          'best8_in_range_valid_fraction')
summary = {'sections': len(rows), 'synthetic_subjects': 8,
           **{field: equal_subject_mean(field) for field in fields}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
