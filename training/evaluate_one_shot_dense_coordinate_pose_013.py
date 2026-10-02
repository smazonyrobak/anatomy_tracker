"""Matched synthetic and weak-real DEV readout for the spatial pose head."""
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

run = root / 'runs/one_shot_dense_coordinate_pose_013'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_dense_coordinate_pose_013_development_eval'
side, beam, fit_side = 256, 8, 96
steps = tuple(range(1000, 6001, 1000))
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 6000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(records) == 256 and len({row['animal_id'] for row in records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
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
        checkpoint_path = run / f'joint_step_{step:05d}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in records:
            row = {'set': 'synthetic', 'step': step, 'eligible': record['eligible'],
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id',
                                                   'valid_pixels', 'sha256')}}
            if record['eligible']:
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    true_reflection = int(arrays['reflection'])
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                prediction = model.predict(image)
                dense_state = model.dense_coordinate_plane(prediction)
                prior = (prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                top = prior.topk(beam).indices
                indices = valid.flatten().nonzero().flatten()
                chart = torch.stack((indices.remainder(side),
                                     indices.div(side, rounding_mode='floor')), -1).float() / side
                target = reference.reshape(-1, 3)[indices]
                all_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                all_flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, model.modes, 2)
                rigid = (points(all_states, all_flags, chart) - target).norm(dim=-1).mean(-1).flatten()
                dense_rigid = (points(dense_state, torch.zeros(1, device='cuda'), chart)[0]
                               - target).norm(dim=-1).mean()
                mapped_coarse = model.map(prediction, offsets, (top // 2)[None],
                                          (top % 2)[None], (fit_side, fit_side), atlas,
                                          weights, feature_side=fit_side,
                                          source_shape=(side, side))
                scores = model.score_fitted_candidates(image, prediction, mapped_coarse,
                                                       atlas, weights)[0]
                prior_choice = int(top[0])
                fitted_choice = int(top[scores.argmax()])
                extended = {**prediction, 'state': torch.cat((prediction['state'],
                                                              dense_state[:, None], truth[:, None]), 1)}
                chosen = torch.tensor([[prior_choice // 2, fitted_choice // 2, 16, 17]], device='cuda')
                flags = torch.tensor([[prior_choice % 2, fitted_choice % 2, 0, true_reflection]],
                                     device='cuda')
                mapped = model.map(extended, offsets, chosen, flags,
                                   (side, side), atlas, weights)
                surfaces = mapped['centre_surface_ccf_ap_dv_ml_um'][0].reshape(4, side * side, 3)
                mapped_error = (surfaces[:, indices] - target[None]).norm(dim=-1).mean(-1)
                row.update(prior_choice=prior_choice, fitted_choice=fitted_choice,
                           prior_rigid_um=float(rigid[prior_choice]),
                           fitted_rigid_um=float(rigid[fitted_choice]),
                           top8_oracle_rigid_um=float(rigid[top].min()),
                           all32_oracle_rigid_um=float(rigid.min()),
                           dense_rigid_um=float(dense_rigid),
                           prior_mapped_um=float(mapped_error[0]),
                           fitted_mapped_um=float(mapped_error[1]),
                           dense_mapped_um=float(mapped_error[2]),
                           true_mapped_um=float(mapped_error[3]),
                           predicted_tissue_fraction=float(prediction['dense_coordinate'][0, 3].sigmoid().mean()))
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        print(json.dumps({'step': step, 'synthetic_rows': len(records)}), flush=True)
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = (affine[:, 2] + 192 * corners[:, :1] * affine[:, 0]
                         + 192 * corners[:, 1:] * affine[:, 1])
            prediction = model.predict(image)
            dense_state = model.dense_coordinate_plane(prediction)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, model.modes, 2)
            five = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1).flatten()
            dense_five = (points(dense_state, torch.zeros(1, device='cuda'), corners)[0]
                          - reference).norm(dim=-1).mean()
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'prior_five_um': float(five[int(prior.argmax())]),
                   'dense_five_um': float(dense_five),
                   'predicted_tissue_fraction': float(prediction['dense_coordinate'][0, 3].sigmoid().mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        print(json.dumps({'step': step, 'real_rows': len(real_records)}), flush=True)
    stream.flush()

summary = []
for step in steps:
    for split, names in (('synthetic', ('prior_rigid_um', 'fitted_rigid_um',
                                      'top8_oracle_rigid_um', 'all32_oracle_rigid_um',
                                      'dense_rigid_um', 'prior_mapped_um',
                                      'fitted_mapped_um', 'dense_mapped_um', 'true_mapped_um')),
                         ('real_weak_allen', ('prior_five_um', 'dense_five_um'))):
        group = [row for row in rows if row['set'] == split and row['step'] == step
                 and (split != 'synthetic' or row['eligible'])]
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
