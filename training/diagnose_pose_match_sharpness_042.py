"""Frozen 041 soft, four-key, and hard-match pose-fit mechanism comparison."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.pose_feedback_global_041 import PoseFeedbackGlobal041

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
run = root / 'runs/pose_feedback_global_041_pilot'
out = root / 'runs/pose_match_sharpness_042_diagnostic'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
step, side, beam = 1500, 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
        frame[..., :, :2] @ basis, chart - .5)


def fit(matched, confidence, chart, prior, gate):
    ridge = torch.diag(matched.new_tensor((2., 1., 1.)))
    weight = confidence
    for iteration in range(2):
        lhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, weight, chart) + ridge
        rhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, weight, matched) + ridge @ prior
        fitted = torch.linalg.solve(lhs, rhs)
        if iteration == 0:
            residual = (chart @ fitted - matched).norm(dim=-1)
            weight = confidence * (1500 / residual.clamp_min(1500))
    raw_centre, raw_u, raw_v = fitted.unbind(-2)
    raw = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((raw_centre - .5 * (raw_u + raw_v), raw_u, raw_v), -2)))
    blended = prior + gate[..., None, None] * (fitted - prior)
    centre, u, v = blended.unbind(-2)
    gated = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((centre - .5 * (u + v), u, v), -2)))
    return raw, gated


completed = json.loads((run / 'completed.json').read_text())
assert completed['updates'] == 2000 and completed['accepted_synthetic'] == 4000
panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
synthetic = [row for row in panel_rows if row['eligible']]
real_rows = [json.loads(line) for line in (real / 'records.jsonl').open()
             if json.loads(line)['training_split'] == 'development']
assert len(synthetic) == 177 and len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert len(real_rows) == 64 and len({row['animal_id'] for row in real_rows}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
head = PoseFeedbackGlobal041().cuda().eval()
checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == step and not checkpoint['calibrated']
head.load_state_dict(checkpoint['head'])
del checkpoint
out.mkdir(parents=True, exist_ok=False)
config = {'step': step, 'beam': beam, 'parent_sha256': sha(parent),
          'head_sha256': sha(run / f'joint_step_{step:05d}.pt'),
          'train_completed_sha256': sha(run / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'source_sha256': sha(Path(__file__)),
          'strategies': ['soft_original', 'four_largest_logits', 'largest_logit'],
          'fitter': 'same 041 ridge and one 1.5-mm robust reweighting',
          'gate': 'frozen 041 gate, no recalibration',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = (torch.arange(16, device='cuda') + .5) / 16
y, x = torch.meshgrid(axis, axis, indexing='ij')
base_chart = torch.stack((x, y), -1).reshape(1, 1, 256, 2)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
rows = []
max_soft_reproduction_um = 0.
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for subset, records in (('synthetic', synthetic), ('real_weak_allen', real_rows)):
        for record in records:
            if subset == 'synthetic':
                path = panel / record['file']
                assert sha(path) == record['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                ids = valid.flatten().nonzero().flatten()
                chart_eval = torch.stack((ids.remainder(side),
                    ids.div(side, rounding_mode='floor')), -1).float()[None] / side
                reference = target.reshape(-1, 3)[ids]
            else:
                i = record['array_row_index']
                image = np.concatenate((real_images[i].astype(np.float32),
                                        np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                      mode='bilinear', align_corners=False)
                offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
                weights = torch.ones_like(offsets)
                weights[:, [0, -1]] = .5
                weights /= weights.sum(-1, keepdim=True)
                chart_eval = corners[None]
                affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                            + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            prior_score = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior_score.topk(beam, -1).indices
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            result = head(prediction, image, state, reflection, atlas, offsets, weights)
            qchart = base_chart.expand(1, beam, -1, -1).clone()
            qchart[..., 0] = torch.where(reflection[..., None].bool(),
                                         255 / 256 - qchart[..., 0], qchart[..., 0])
            chart = torch.cat((torch.ones_like(qchart[..., :1]), qchart - .5), -1)
            centre, frame, basis = full_frame_state_to_components(state)
            edges = frame[..., :, :2] @ basis
            prior = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
            soft_raw, soft_gated = fit(result['matched_world'], result['confidence'],
                                        chart, prior, result['correction_gate'])
            reproduction = (points(soft_raw, reflection, corners[None]) -
                points(result['raw_fitted_state'], reflection, corners[None])).abs().max()
            reproduction = torch.maximum(reproduction,
                (points(soft_gated, reflection, corners[None]) -
                 points(result['state'], reflection, corners[None])).abs().max())
            max_soft_reproduction_um = max(max_soft_reproduction_um, float(reproduction))
            top = result['logits'].topk(4, -1)
            keys = torch.gather(result['key_world'][:, :, None].expand(
                -1, -1, 256, -1, -1), 3,
                top.indices[..., None].expand(-1, -1, -1, -1, 3))
            matched = {'soft_raw': result['raw_fitted_state'],
                       'soft_gated': result['state']}
            for name, location in (
                    ('four', (keys * top.values.softmax(-1)[..., None]).sum(-2)),
                    ('hard', keys[..., 0, :])):
                matched[name + '_raw'], matched[name + '_gated'] = fit(
                    location, result['confidence'], chart, prior,
                    result['correction_gate'])
            errors = {'prior': (points(state, reflection, chart_eval) -
                                reference[None, None]).norm(dim=-1).mean(-1)[0]}
            errors.update({name: (points(pose, reflection, chart_eval) -
                reference[None, None]).norm(dim=-1).mean(-1)[0]
                for name, pose in matched.items()})
            row = {'set': subset,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id')},
                   **({'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                       'appearance_mode': record['appearance_mode'],
                       'sha256': record['sha256']} if subset == 'synthetic' else {}),
                   **{name + '_top1_um': float(error[0]) for name, error in errors.items()},
                   **{name + '_best8_um': float(error.min()) for name, error in errors.items()}}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'set': subset, 'rows': len(records)}), flush=True)

assert max_soft_reproduction_um < 2., max_soft_reproduction_um


def group_mean(records, field, group):
    return float(np.mean([np.mean([row[field] for row in records if row[group] == key])
                          for key in sorted({row[group] for row in records})]))


summary = {'soft_fit_reproduction_max_um': max_soft_reproduction_um,
           'synthetic_eligible_sections': len(synthetic), 'synthetic_subjects': 8,
           'real_weak_sections': len(real_rows), 'real_weak_donors': 6,
           'synthetic': {}, 'real_weak': {}}
fields = [prefix + '_' + suffix + '_um' for prefix in
          ('prior', 'soft_raw', 'soft_gated', 'four_raw', 'four_gated', 'hard_raw',
           'hard_gated') for suffix in ('top1', 'best8')]
for subset, label, group in (('synthetic', 'synthetic', 'synthetic_subject_plan_id'),
                             ('real_weak_allen', 'real_weak', 'animal_id')):
    selected = [row for row in rows if row['set'] == subset]
    summary[label] = {field: group_mean(selected, field, group) for field in fields}
    if label == 'synthetic':
        near = [row for row in selected if row['prior_best8_um'] <= 1000]
        summary['near_true_sections'] = len(near)
        summary['near_true'] = {field: group_mean(near, field, group)
                                for field in fields}
        summary['by_appearance'] = {appearance: {field: float(np.mean([
            row[field] for row in selected if row['appearance_mode'] == appearance]))
            for field in fields}
            for appearance in sorted({row['appearance_mode'] for row in selected})}
    else:
        summary['worst_real_donor_regression_um'] = {name: float(max(
            np.mean([row[name + '_top1_um'] - row['prior_top1_um']
                     for row in selected if row['animal_id'] == donor])
            for donor in {row['animal_id'] for row in selected}))
            for name in ('soft_gated', 'four_gated', 'hard_gated')}
summary['mechanism_direction_promising'] = {name: (
    summary['synthetic']['prior_best8_um'] - summary['synthetic'][name + '_best8_um'] >= 200
    and summary['synthetic'][name + '_top1_um'] -
        summary['synthetic']['prior_top1_um'] <= 200)
    for name in ('four_raw', 'four_gated', 'hard_raw', 'hard_gated')}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'mechanism_direction_promising':
                  summary['mechanism_direction_promising'],
                  'soft_reproduction_max_um': max_soft_reproduction_um}), flush=True)
