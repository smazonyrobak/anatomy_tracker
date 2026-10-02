"""Does fitted evidence improve physical geometry near actual predicted poses?"""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint_path = root / 'runs/one_shot_positive_fit_feedback_012/joint_step_06000.pt'
out = root / 'runs/one_shot_positive_fit_feedback_012_local_direction_diag'
side, fit_side, beam = 256, 96, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
updates = torch.zeros((19, 9), device='cuda')
for axis in range(3):
    updates[1 + 2 * axis, 3 + axis] = 500
    updates[2 + 2 * axis, 3 + axis] = -500
    updates[7 + 2 * axis, axis] = .12
    updates[8 + 2 * axis, axis] = -.12
for axis in range(2):
    updates[13 + 2 * axis, 6 + axis] = .05
    updates[14 + 2 * axis, 6 + axis] = -.05
updates[17, 8], updates[18, 8] = .05, -.05

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
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        top = prior.topk(beam).indices
        mapped = model.map(prediction, offsets, (top // 2)[None], (top % 2)[None],
                           (fit_side, fit_side), atlas, weights, feature_side=fit_side,
                           source_shape=(side, side))
        scores = model.score_fitted_candidates(image, prediction, mapped, atlas, weights)[0]
        branch = int(top[scores.argmax()])
        states = compose_full_frame_state(
            prediction['state'][0, branch // 2].expand(len(updates), -1), updates)
        reflected = torch.full((1, len(updates)), branch % 2, device='cuda', dtype=torch.long)
        local_prediction = {**prediction, 'state': states[None],
                            'log_mass': image.new_zeros((1, len(updates))),
                            'reflection_logit': image.new_zeros((1, len(updates)))}
        mapped = model.map(local_prediction, offsets,
                           torch.arange(len(updates), device='cuda')[None], reflected,
                           (fit_side, fit_side), atlas, weights, feature_side=fit_side,
                           source_shape=(side, side))
        fit_score = model.score_fitted_candidates(
            image, local_prediction, mapped, atlas, weights)[0]
        indices = valid.nonzero().flatten()
        chart = torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() / side
        chart[:, 0] = torch.where(reflected[0, 0].bool(),
                                  (side - 1) / side - chart[:, 0], chart[:, 0])
        centre, frame, basis = full_frame_state_to_components(states)
        rigid = centre[:, None] + torch.einsum('kij,pj->kpi',
                                             frame[:, :, :2] @ basis, chart - .5)
        rigid_error = (rigid - target[indices][None]).norm(dim=-1).mean(-1)
        grid = (torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() + .5)
        grid = (grid * (2 / side) - 1)[None, None].expand(len(updates), -1, -1, -1)
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                               align_corners=False)[:, :, 0].transpose(1, 2)
        mapped_error = (fitted - target[indices][None]).norm(dim=-1).mean(-1)
        e = mapped_error.cpu().numpy()
        s = fit_score.cpu().numpy()
        rank_score = np.argsort(np.argsort(s))
        rank_truth = np.argsort(np.argsort(-e))
        row = {'animal_id': record['animal_id'], 'section_id': record['section_id'],
               'sha256': record['sha256'], 'valid_pixels': record['valid_pixels'],
               'branch': branch, 'base_mapped_um': float(e[0]),
               'score_choice_mapped_um': float(e[s.argmax()]),
               'oracle_mapped_um': float(e.min()),
               'base_rigid_um': float(rigid_error[0]),
               'score_choice_rigid_um': float(rigid_error[s.argmax()]),
               'score_choice_index': int(s.argmax()), 'oracle_index': int(e.argmin()),
               'score_error_spearman': float(np.corrcoef(rank_score, rank_truth)[0, 1]),
               'mapped_error_um': e.tolist(), 'fit_score': s.tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    stream.flush()

summary = {'rows': len(rows), 'perturbations': 'base; ±500 um local AP/DV/ML-frame translations; ±0.12 rad local rotations; ±0.05 log-scale in each plane axis; ±0.05 shear',
           'mean_base_mapped_um': float(np.mean([r['base_mapped_um'] for r in rows])),
           'mean_score_choice_mapped_um': float(np.mean([r['score_choice_mapped_um'] for r in rows])),
           'mean_oracle_mapped_um': float(np.mean([r['oracle_mapped_um'] for r in rows])),
           'score_improved': sum(r['score_choice_mapped_um'] < r['base_mapped_um'] for r in rows),
           'score_worsened': sum(r['score_choice_mapped_um'] > r['base_mapped_um'] for r in rows),
           'score_unchanged': sum(r['score_choice_mapped_um'] == r['base_mapped_um'] for r in rows),
           'mean_within_case_score_error_spearman': float(np.mean([r['score_error_spearman'] for r in rows])),
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
