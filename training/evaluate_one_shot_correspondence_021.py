"""Matched frozen-development comparison of 021 spatial and 019 pooled scores."""
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
from training.one_shot_candidate_correspondence_021 import OneShotCorrespondenceModel021

run = root / 'runs/one_shot_correspondence_021'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_correspondence_021_development_eval'
side, fit_side, beam = 256, 96, 8
steps = (0, 500, 1000, 1500, 2000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 2000
synthetic_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 256 and len({row['animal_id'] for row in synthetic_records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotCorrespondenceModel021().cuda().eval()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def candidates(image, offsets, weights):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
    choice = prior.topk(beam).indices
    state = prediction['state'][:, choice // 2]
    reflected = (choice % 2)[None]
    selected = {**prediction, 'state': state,
                'log_mass': prediction['log_mass'][:, choice // 2],
                'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
    index = torch.arange(beam, device='cuda')[None]
    first = model.map(selected, offsets, index, reflected, (64, 64), atlas, weights,
                      return_refinement_feature=True, feature_side=64,
                      source_shape=(side, side))
    refined_state, delta, _ = model.refine(first['refinement_feature'], state)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, reflected, (fit_side, fit_side),
                       atlas, weights, feature_side=fit_side, source_shape=(side, side))
    old = model.score_fitted_candidates(image, refined, mapped, atlas, weights) + delta
    new = model.correspondence_score_021(image, refined, mapped, atlas, weights, old)
    return choice, reflected, refined_state, mapped, old[0], new[0]


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        saved = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == step and not saved['calibrated']
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for record in synthetic_records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().float()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            choice, reflected, state, mapped, old, new = candidates(image, offsets, weights)
            target = F.interpolate(reference.permute(2, 0, 1)[None], (fit_side, fit_side),
                                   mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
            tissue = F.interpolate(valid[None, None], (fit_side, fit_side), mode='area')[0, 0]
            error = (mapped['centre_surface_ccf_ap_dv_ml_um'][0] - target[None]).norm(dim=-1)
            physical = (error * tissue).sum((-1, -2)) / tissue.sum()
            old_index, new_index, best_index = int(old.argmax()), int(new.argmax()), int(physical.argmin())
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id', 'sha256')},
                   'valid_fraction': float(valid.mean()),
                   'old_mapped_um': float(physical[old_index]),
                   'new_mapped_um': float(physical[new_index]),
                   'oracle_mapped_um': float(physical[best_index]),
                   'old_top1_oracle': old_index == best_index,
                   'new_top1_oracle': new_index == best_index,
                   'old_branch': int(choice[old_index]), 'new_branch': int(choice[new_index]),
                   'old_score': old.tolist(), 'new_score': new.tolist(),
                   'candidate_error_um': physical.tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
            weights = torch.ones_like(offsets)
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            choice, reflected, state, mapped, old, new = candidates(image, offsets, weights)
            physical = (points(state, reflected, corners) - reference).norm(dim=-1).mean(-1)[0]
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'old_five_um': float(physical[int(old.argmax())]),
                   'new_five_um': float(physical[int(new.argmax())])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': 185, 'real': 64}), flush=True)

summary = []
for step in steps:
    for split, names in (('synthetic', ('old_mapped_um', 'new_mapped_um', 'oracle_mapped_um',
                                      'old_top1_oracle', 'new_top1_oracle')),
                         ('real_weak_allen', ('old_five_um', 'new_five_um'))):
        group = [row for row in rows if row['set'] == split and row['step'] == step]
        identities = sorted({row['animal_id'] for row in group})
        summary.append({'step': step, 'set': split, 'rows': len(group),
                        'identities': len(identities), 'identity_equal_mean': {
                            name: float(np.mean([np.mean([row[name] for row in group
                                if row['animal_id'] == identity]) for identity in identities]))
                            for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows),
    'checkpoint_sha256': {str(step): hashlib.sha256(
        (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
