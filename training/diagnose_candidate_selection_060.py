"""Frozen 059 old/new candidate-score diagnostic; no model updates."""
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

run = root / 'runs/one_shot_anchor_quality_059'
evaluation = root / 'runs/one_shot_anchor_quality_059_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_anchor_quality_059_selection_diagnostic'
side = 256
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['batches'] == 50000
assert json.loads((evaluation / 'completed.json').read_text())['rows'] == 996
assert not out.exists()

expected = {(row['set'], row['animal_id'], row['section_id']): row
            for row in map(json.loads, (evaluation / 'rows.jsonl').open()) if row['step'] == 50000}
synthetic_records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 185 and len(real_records) == 64 and len(expected) == 249
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as geometry:
    affines = geometry['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = geometry['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
                              fit_quality=True, vector_refinement=True,
                              candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(run / 'joint_step_50000.pt', map_location='cpu',
                                 weights_only=True)['model'], strict=True)


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def candidates(image, offsets, weights):
    prediction = model.predict(image)
    reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                  F.logsigmoid(prediction['reflection_logit'])), -1)
    prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
    choice = torch.cat((prior[:, :32].topk(2, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    selected = {**prediction, 'state': state,
                'log_mass': prediction['log_mass'].gather(1, choice // 2),
                'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
    index = torch.arange(8, device='cuda')[None]
    first = model.map(selected, offsets, index, choice % 2, (64, 64), atlas, weights,
                      return_refinement_feature=True, feature_side=64,
                      source_shape=(side, side))
    refined_state, delta, _ = model.refine(first['refinement_feature'], state)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, choice % 2, (96, 96), atlas, weights,
                       feature_side=96, source_shape=(side, side))
    score = model.score_fitted_candidates(image, refined, mapped, atlas, weights) + delta
    return choice, prior.gather(1, choice)[0], score[0], refined_state, mapped


out.mkdir(parents=True)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in synthetic_records:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        choice, prior, score, _, mapped = candidates(image, offsets, weights)
        indices = valid.flatten().nonzero().flatten()
        chosen = indices[torch.linspace(0, len(indices) - 1, 256, device='cuda').round().long()]
        target = reference.reshape(-1, 3)[chosen]
        grid = (torch.stack((chosen.remainder(side),
                             chosen.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
        grid = grid[None].expand(8, -1, -1).reshape(8, 1, 256, 2)
        surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
        fitted = F.grid_sample(surface, grid, padding_mode='border',
                               align_corners=False).squeeze(2).transpose(1, 2)
        errors = (fitted - target[None]).norm(dim=-1).mean(-1)
        split = 'synthetic'
        check_key = 'selected_mapped96_um'
    
        key = (split, record['animal_id'], record['section_id'])
        assert abs(float(errors[int(score.argmax())]) - expected[key][check_key]) < .01
        row = {'set': split, 'animal_id': record['animal_id'],
               'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
               'section_id': record['section_id'], 'sha256': record['sha256'],
               'branch': choice[0].tolist(), 'prior': prior.tolist(),
               'fitted_score': score.tolist(), 'truth_error_um': errors.tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    for record in real_records:
        i = record['array_row_index']
        image = np.concatenate((real_images[i].astype(np.float32),
                                np.zeros((4, 192, 192), np.float32)))[None]
        image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                              mode='bilinear', align_corners=False)
        offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
        weights = torch.ones_like(offsets)
        weights[:, [0, -1]] = .5
        weights /= weights.sum(-1, keepdim=True)
        affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
        reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
        choice, prior, score, state, _ = candidates(image, offsets, weights)
        errors = (points(state, choice % 2, corners) - reference).norm(dim=-1).mean(-1)[0]
        split = 'real_weak_allen'
        key = (split, record['animal_id'], record['section_id'])
        assert abs(float(errors[int(score.argmax())]) - expected[key]['selected_five_um']) < .01
        row = {'set': split, **{name: record[name] for name in
               ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
               'branch': choice[0].tolist(), 'prior': prior.tolist(),
               'fitted_score': score.tolist(), 'truth_error_um': errors.tolist()}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

summary = []
for split in ('synthetic', 'real_weak_allen'):
    group = [row for row in rows if row['set'] == split]
    identities = sorted({row['animal_id'] for row in group})
    for policy in ('fitted_score', 'prior', 'fitted_minus_prior', 'old_fitted',
                   'new_fitted', 'oracle_in_beam'):
        def error(row):
            score = np.asarray(row['fitted_score'])
            prior = np.asarray(row['prior'])
            truth = np.asarray(row['truth_error_um'])
            index = {'fitted_score': lambda: score.argmax(),
                     'prior': lambda: prior.argmax(),
                     'fitted_minus_prior': lambda: (score - prior).argmax(),
                     'old_fitted': lambda: score[:2].argmax(),
                     'new_fitted': lambda: score[2:].argmax() + 2,
                     'oracle_in_beam': lambda: truth.argmin()}[policy]()
            return float(truth[index]), int(index >= 2)
        summary.append({'set': split, 'policy': policy, 'rows': len(group),
                        'identities': len(identities),
                        'identity_equal_error_um': float(np.mean([
                            np.mean([error(row)[0] for row in group if row['animal_id'] == animal])
                            for animal in identities])),
                        'identity_equal_anchor_fraction': float(np.mean([
                            np.mean([error(row)[1] for row in group if row['animal_id'] == animal])
                            for animal in identities]))})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': hashlib.sha256(
        (run / 'joint_step_50000.pt').read_bytes()).hexdigest(),
    'evaluation_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows), 'summary': summary}), flush=True)
