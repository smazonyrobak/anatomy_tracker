"""Fixed-panel, full-resolution endpoint for joint atlas-fit pose updates."""
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

run = root / 'runs/one_shot_joint_rerender_017'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_joint_rerender_017_development_eval'
side, first_side, fit_side, beam = 256, 64, 96, 8
steps = tuple(range(0, 16001, 2000))
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 16000
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
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               vector_refinement=True, candidate_ranking=True,
                               fitted_ranking=True).cuda().eval()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def refine_candidates(image, offsets, weights):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
    choice = prior.topk(beam).indices
    states = prediction['state'][:, choice // 2]
    reflected = (choice % 2)[None]
    selected = {**prediction, 'state': states,
                'log_mass': prediction['log_mass'][:, choice // 2],
                'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
    index = torch.arange(beam, device='cuda')[None]
    first = model.map(selected, offsets, index, reflected, (first_side, first_side),
                      atlas, weights, return_refinement_feature=True,
                      feature_side=first_side, source_shape=(side, side))
    refined_state, delta, _ = model.refine(first['refinement_feature'], states)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, reflected, (fit_side, fit_side),
                       atlas, weights, feature_side=fit_side, source_shape=(side, side))
    score = model.score_fitted_candidates(image, refined, mapped, atlas, weights)[0] + delta[0]
    return prediction, prior, choice, reflected, refined_state, score


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
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, prior, choice, reflected, refined_state, score = refine_candidates(
                image, offsets, weights)
            indices = valid.flatten().nonzero().flatten()
            chart = torch.stack((indices.remainder(side),
                                 indices.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[indices]
            initial_rigid = (points(prediction['state'][:, choice // 2], reflected, chart) - target)
            initial_rigid = initial_rigid.norm(dim=-1).mean(-1)[0]
            refined_rigid = (points(refined_state, reflected, chart) - target).norm(dim=-1).mean(-1)[0]
            selected_branch = int(score.argmax())
            chosen_states = torch.cat((prediction['state'][:, choice[:1] // 2],
                                       refined_state[:, selected_branch:selected_branch + 1],
                                       truth[:, None]), 1)
            mapped = model.map({**prediction, 'state': chosen_states}, offsets,
                               torch.tensor([[0, 1, 2]], device='cuda'),
                               torch.tensor([[int(choice[0] % 2),
                                              int(reflected[0, selected_branch]), true_reflection]], device='cuda'),
                               (side, side), atlas, weights)
            error = (mapped['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                     - target[None]).norm(dim=-1).mean(-1)
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id', 'sha256')},
                   'valid_fraction': float(valid.float().mean()),
                   'prior_mapped_um': float(error[0]),
                   'refined_mapped_um': float(error[1]),
                   'true_mapped_um': float(error[2]),
                   'prior_rigid_um': float(initial_rigid[0]),
                   'best8_initial_rigid_um': float(initial_rigid.min()),
                   'best8_refined_rigid_um': float(refined_rigid.min()),
                   'refined_selected_rigid_um': float(refined_rigid[selected_branch]),
                   'chosen_prior_branch': int(choice[0]),
                   'chosen_refined_branch': int(choice[selected_branch])}
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
            prediction, prior, choice, reflected, refined_state, score = refine_candidates(
                image, offsets, weights)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, model.modes, 2)
            direct = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1).flatten()
            refined = (points(refined_state, reflected, corners) - reference).norm(dim=-1).mean(-1)[0]
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'prior_five_um': float(direct[int(prior.argmax())]),
                   'refined_five_um': float(refined[int(score.argmax())])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': 185, 'real': 64}), flush=True)

summary = []
for step in steps:
    for split, names in (('synthetic', ('prior_mapped_um', 'refined_mapped_um',
                                      'true_mapped_um', 'prior_rigid_um',
                                      'best8_initial_rigid_um', 'best8_refined_rigid_um',
                                      'refined_selected_rigid_um')),
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
