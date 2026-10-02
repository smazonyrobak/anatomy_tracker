"""Frozen same-checkpoint fitted-versus-prior readout on development data."""
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_fitted_ranking_010'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_fitted_ranking_010_development_eval'
checkpoint_path = run / 'joint_step_12000.pt'
side, beam, fit_side = 256, 8, 96
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 12000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(records) == 256 and len({row['animal_id'] for row in records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 12000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
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
    for record in records:
        row = {'set': 'synthetic', 'eligible': record['eligible'],
               **{key: record[key] for key in ('animal_id', 'specimen_id',
                                               'experiment_id', 'section_id',
                                               'synthetic_subject_plan_id', 'valid_pixels',
                                               'appearance_mode', 'sha256')}}
        if record['eligible']:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = torch.as_tensor(arrays['reflection'].copy(), device='cuda').long().reshape(1)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction = model.predict(image)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            top = prior.topk(beam).indices
            mapped_coarse = model.map(prediction, offsets, (top // 2)[None],
                                      (top % 2)[None], (fit_side, fit_side), atlas, weights,
                                      feature_side=fit_side, source_shape=(side, side))
            scores = model.score_fitted_candidates(image, prediction, mapped_coarse,
                                                   atlas, weights)[0]
            prior_choice = int(top[0])
            fitted_choice = int(top[scores.argmax()])
            indices = valid.flatten().nonzero().flatten()
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[indices]
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, 16, 2)
            geometry = (points(states, flags, chart) - target).norm(dim=-1).mean(-1).flatten()
            selected = torch.tensor([[prior_choice, fitted_choice]], device='cuda')
            mapped = model.map(prediction, offsets, selected // 2, selected % 2,
                               (side, side), atlas, weights)
            surfaces = mapped['centre_surface_ccf_ap_dv_ml_um'][0].reshape(2, side * side, 3)
            mapped_error = (surfaces[:, indices] - target[None]).norm(dim=-1).mean(-1)
            true_pose_unwarped = (points(truth, true_reflection, chart)[0] - target).norm(dim=-1).mean()
            truth_states = prediction['state'].clone()
            truth_states[:, 0] = truth
            true_map = model.map({**prediction, 'state': truth_states}, offsets,
                                 torch.zeros((1, 1), device='cuda', dtype=torch.long),
                                 true_reflection[:, None], (side, side), atlas, weights)
            true_surface = true_map['centre_surface_ccf_ap_dv_ml_um'][0, 0].reshape(side * side, 3)
            true_pose_mapped = (true_surface[indices] - target).norm(dim=-1).mean()
            true_normal = full_frame_state_to_components(truth)[1][0, :, 2]
            normals = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
            normal_error = torch.rad2deg(torch.acos((normals * true_normal).sum(-1)
                                                   .abs().clamp(0, 1)))
            row.update(prior_choice=prior_choice, fitted_choice=fitted_choice,
                       prior_tissue_um=float(geometry[prior_choice]),
                       fitted_tissue_um=float(geometry[fitted_choice]),
                       top8_oracle_tissue_um=float(geometry[top].min()),
                       all32_oracle_tissue_um=float(geometry.min()),
                       prior_mapped_tissue_um=float(mapped_error[0]),
                       fitted_mapped_tissue_um=float(mapped_error[1]),
                       true_pose_unwarped_um=float(true_pose_unwarped),
                       true_pose_mapped_um=float(true_pose_mapped),
                       fitted_normal_deg=float(normal_error[fitted_choice // 2]),
                       top8_branches=top.tolist(), top8_scores=scores.tolist(),
                       top8_tissue_um=geometry[top].tolist())
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    print(json.dumps({'synthetic_rows': len(rows)}), flush=True)
    for record in real_records:
        i = record['array_row_index']
        image = np.concatenate((real_images[i].astype(np.float32),
                                np.zeros((4, 192, 192), dtype=np.float32)))[None]
        image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                              mode='bilinear', align_corners=False)
        offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
        weights = torch.ones((1, 9), device='cuda')
        weights[:, [0, -1]] = .5
        weights /= weights.sum(-1, keepdim=True)
        affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
        reference = (affine[:, 2] + 192 * corners[:, :1] * affine[:, 0]
                     + 192 * corners[:, 1:] * affine[:, 1])
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        top = prior.topk(beam).indices
        mapped_coarse = model.map(prediction, offsets, (top // 2)[None], (top % 2)[None],
                                  (fit_side, fit_side), atlas, weights,
                                  feature_side=fit_side, source_shape=(side, side))
        scores = model.score_fitted_candidates(image, prediction, mapped_coarse, atlas, weights)[0]
        prior_choice = int(top[0])
        fitted_choice = int(top[scores.argmax()])
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, 16, 2)
        geometry = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1).flatten()
        row = {'set': 'real_weak_allen',
               **{key: record[key] for key in ('animal_id', 'specimen_id',
                                               'experiment_id', 'section_id')},
               'prior_choice': prior_choice, 'fitted_choice': fitted_choice,
               'prior_five_um': float(geometry[prior_choice]),
               'fitted_five_um': float(geometry[fitted_choice]),
               'top8_oracle_five_um': float(geometry[top].min()),
               'top8_branches': top.tolist(), 'top8_scores': scores.tolist(),
               'top8_five_um': geometry[top].tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    stream.flush()

summary = []
for split, names in (
    ('synthetic', ('prior_tissue_um', 'fitted_tissue_um', 'top8_oracle_tissue_um',
                   'all32_oracle_tissue_um', 'prior_mapped_tissue_um',
                   'fitted_mapped_tissue_um', 'true_pose_unwarped_um',
                   'true_pose_mapped_um', 'fitted_normal_deg')),
    ('real_weak_allen', ('prior_five_um', 'fitted_five_um', 'top8_oracle_five_um'))):
    group = [row for row in rows if row['set'] == split]
    scored = [row for row in group if split != 'synthetic' or row['eligible']]
    identities = sorted({row['animal_id'] for row in scored})
    report = {'set': split, 'rows': len(group), 'scored': len(scored),
              'identities': len(identities),
              'identity_equal_mean': {name: float(np.mean([np.mean([
                        row[name] for row in scored if row['animal_id'] == identity])
                        for identity in identities])) for name in names},
              'fitted_better_count': sum(row[names[1]] < row[names[0]] for row in scored),
              'fitted_worse_count': sum(row[names[1]] > row[names[0]] for row in scored),
              'prior_over_5mm': sum(row[names[0]] > 5000 for row in scored),
              'fitted_over_5mm': sum(row[names[1]] > 5000 for row in scored),
              'median_fitted_um': float(np.median([row[names[1]] for row in scored])),
              'p90_fitted_um': float(np.percentile([row[names[1]] for row in scored], 90))}
    if split == 'synthetic':
        report['true_pose_mapping'] = {
            'improved_count': sum(row['true_pose_mapped_um'] < row['true_pose_unwarped_um']
                                  for row in scored),
            'worse_count': sum(row['true_pose_mapped_um'] > row['true_pose_unwarped_um']
                               for row in scored),
        }
        report['mapped_comparison'] = {
            'fitted_better_count': sum(row['fitted_mapped_tissue_um'] < row['prior_mapped_tissue_um']
                                       for row in scored),
            'fitted_worse_count': sum(row['fitted_mapped_tissue_um'] > row['prior_mapped_tissue_um']
                                      for row in scored),
            'prior_over_5mm': sum(row['prior_mapped_tissue_um'] > 5000 for row in scored),
            'fitted_over_5mm': sum(row['fitted_mapped_tissue_um'] > 5000 for row in scored),
            'median_fitted_um': float(np.median([row['fitted_mapped_tissue_um'] for row in scored])),
            'p90_fitted_um': float(np.percentile([row['fitted_mapped_tissue_um'] for row in scored], 90)),
        }
        prior_regret = np.mean([np.mean([row['prior_tissue_um'] - row['top8_oracle_tissue_um']
                                        for row in scored if row['animal_id'] == identity])
                                for identity in identities])
        fitted_regret = np.mean([np.mean([row['fitted_tissue_um'] - row['top8_oracle_tissue_um']
                                         for row in scored if row['animal_id'] == identity])
                                 for identity in identities])
        report['top8_rigid_selection_regret_um'] = {'prior': float(prior_regret),
                                                    'fitted': float(fitted_regret)}
    summary.append(report)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
