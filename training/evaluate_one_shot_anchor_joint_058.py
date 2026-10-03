"""Frozen 058 full-slice pose/mapper development readout; never run during training."""
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

run = root / 'runs/one_shot_anchor_joint_058'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_anchor_joint_058_development_eval'
steps = (0, 2000, 6000, 12000)
side, beam = 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['batches'] == 12000

synthetic_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 256 and len({row['animal_id'] for row in synthetic_records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz') as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
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


def candidates(image, offsets, weights):
    prediction = model.predict(image)
    reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                  F.logsigmoid(prediction['reflection_logit'])), -1)
    prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
    choice = prior.topk(beam, -1).indices
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    reflection = choice % 2
    selected = {**prediction, 'state': state,
                'log_mass': prediction['log_mass'].gather(1, choice // 2),
                'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
    index = torch.arange(beam, device='cuda')[None]
    first = model.map(selected, offsets, index, reflection, (64, 64), atlas, weights,
                      return_refinement_feature=True, feature_side=64,
                      source_shape=(side, side))
    refined_state, delta, _ = model.refine(first['refinement_feature'], state)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, reflection, (96, 96), atlas, weights,
                       feature_side=96, source_shape=(side, side))
    score = model.score_fitted_candidates(image, refined, mapped, atlas, weights) + delta
    return prediction, choice, state, reflection, refined_state, score


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic_records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file']) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, choice, state, reflection, refined, score = candidates(image, offsets, weights)
            indices = valid.flatten().nonzero().flatten()
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[indices]
            rigid = (points(state, reflection, chart) - target).norm(dim=-1).mean(-1)[0]
            refined_rigid = (points(refined, reflection, chart) - target).norm(dim=-1).mean(-1)[0]
            selected = int(score[0].argmax())
            mapped_state = torch.cat((state[:, :1], refined[:, selected:selected + 1], truth[:, None]), 1)
            mapped = model.map({**prediction, 'state': mapped_state}, offsets,
                               torch.tensor([[0, 1, 2]], device='cuda'),
                               torch.tensor([[int(reflection[0, 0]), int(reflection[0, selected]),
                                              true_reflection]], device='cuda'),
                               (side, side), atlas, weights)
            mapped_error = (mapped['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                            - target[None]).norm(dim=-1).mean(-1)
            estimated_normal = full_frame_state_to_components(state)[1][0, :, :, 2]
            true_normal = full_frame_state_to_components(truth)[1][0, :, 2]
            normal_angle = torch.rad2deg((estimated_normal * true_normal).sum(-1).abs()
                                         .clamp(0, 1).acos())
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id', 'sha256')},
                   'prior_mapped_um': float(mapped_error[0]),
                   'refined_mapped_um': float(mapped_error[1]),
                   'true_mapped_um': float(mapped_error[2]),
                   'prior_rigid_um': float(rigid[0]),
                   'best8_rigid_um': float(rigid.min()),
                   'refined_selected_rigid_um': float(refined_rigid[selected]),
                   'prior_normal_angle_deg': float(normal_angle[0]),
                   'best8_normal_angle_deg': float(normal_angle.min()),
                   'top_prior_anchor': bool(int(choice[0, 0] // 2) >= 16),
                   'selected_anchor': bool(int(choice[0, selected] // 2) >= 16),
                   'chosen_prior_branch': int(choice[0, 0]),
                   'chosen_refined_branch': int(choice[0, selected])}
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
            prediction, choice, state, reflection, refined, score = candidates(image, offsets, weights)
            prior_error = (points(state[:, :1], reflection[:, :1], corners)
                           - reference).norm(dim=-1).mean()
            selected = int(score[0].argmax())
            refined_error = (points(refined[:, selected:selected + 1],
                                    reflection[:, selected:selected + 1], corners)
                             - reference).norm(dim=-1).mean()
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id')},
                   'prior_five_um': float(prior_error), 'refined_five_um': float(refined_error)}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': 185, 'real': 64}), flush=True)

summary = []
for step in steps:
    for split, names in (('synthetic', ('prior_mapped_um', 'refined_mapped_um',
                                      'true_mapped_um', 'prior_rigid_um', 'best8_rigid_um',
                                      'refined_selected_rigid_um', 'prior_normal_angle_deg',
                                      'best8_normal_angle_deg', 'top_prior_anchor', 'selected_anchor')),
                         ('real_weak_allen', ('prior_five_um', 'refined_five_um'))):
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
