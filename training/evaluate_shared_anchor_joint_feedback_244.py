"""Fixed synthetic-development and weak Allen donor readout; no calibration data."""
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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

training = root / 'runs/shared_anchor_joint_feedback_244_train_001'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/shared_anchor_joint_feedback_244_readout_001'
steps = (0, 1000, 3000, 6000)
side, first_side, fit_side, beam = 256, 64, 96, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


synthetic_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_all = [json.loads(line) for line in (real / 'records.jsonl').open()]
real_dev = [row for row in real_all if row['training_split'] == 'development']
synthetic = [row for identity in sorted({row['animal_id'] for row in synthetic_records})
             for row in [r for r in synthetic_records if r['animal_id'] == identity and r['eligible']][:4]]
weak_real = [row for identity in sorted({row['animal_id'] for row in real_dev})
             for row in [r for r in real_dev if r['animal_id'] == identity][:2]]
assert len(synthetic_records) == 256 and len(synthetic) == 32
assert len({row['animal_id'] for row in synthetic}) == 8
assert len(weak_real) == 12 and len({row['animal_id'] for row in weak_real}) == 6
assert not {row['animal_id'] for row in real_dev} & {
    row['animal_id'] for row in real_all if row['training_split'] == 'train'}
assert all(row['eligible'] and row['valid_pixels'] > 0 for row in synthetic)
panel_files = {row['file']: sha(panel / row['file']) for row in synthetic}
assert all(panel_files[row['file']] == row['sha256'] for row in synthetic)
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, shared_anchor_pose=True,
                              atlas_conditioning=True, vector_refinement=True,
                              fitted_ranking=True).cuda().eval()
corners = torch.tensor([[0., 0.], [side - 1., 0.], [0., side - 1.],
                        [side - 1., side - 1.], [(side - 1) / 2, (side - 1) / 2]],
                       device='cuda') / side


def infer(image, offsets, weights):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
    choice = prior.topk(beam).indices
    state = prediction['state'][:, choice // 2]
    reflection = (choice % 2)[None]
    selected = {**prediction, 'state': state,
                'log_mass': prediction['log_mass'][:, choice // 2],
                'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
    index = torch.arange(beam, device='cuda')[None]
    first = model.map(selected, offsets, index, reflection, (first_side, first_side),
                      atlas, weights, return_refinement_feature=True,
                      feature_side=first_side, source_shape=image.shape[-2:])
    refined_state, delta, _ = model.refine(first['refinement_feature'], state)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, reflection, (fit_side, fit_side),
                       atlas, weights, feature_side=fit_side, source_shape=image.shape[-2:])
    fitted = model.score_fitted_candidates(image, refined, mapped, atlas, weights)[0] + delta[0]
    return prediction, prior, choice, reflection, refined_state, fitted


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(training / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, prior, choice, reflection, refined_state, fitted = infer(image, offsets, weights)
            indices = valid.flatten().nonzero().flatten()
            chart = torch.stack((indices.remainder(side), indices.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[indices]
            prior_rigid = (points(prediction['state'][:, choice // 2], reflection, chart)
                           - target).norm(dim=-1).mean(-1)[0]
            refined_rigid = (points(refined_state, reflection, chart)
                             - target).norm(dim=-1).mean(-1)[0]
            selected_branch = int(fitted.argmax())
            selected_states = torch.cat((refined_state[:, selected_branch:selected_branch + 1], truth[:, None]), 1)
            selected_reflections = torch.tensor([[int(reflection[0, selected_branch]), true_reflection]], device='cuda')
            mapped = model.map({**prediction, 'state': selected_states}, offsets,
                               torch.tensor([[0, 1]], device='cuda'), selected_reflections,
                               (side, side), atlas, weights, source_shape=image.shape[-2:])
            mapped_error = (mapped['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                            - target[None]).norm(dim=-1).mean(-1)
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
                                                   'panel_physical_section_id', 'synthetic_subject_plan_id')},
                   'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
                   'valid_fraction': float(valid.float().mean()),
                   'prior_top1_rigid_um': float(prior_rigid[0]),
                   'prior_top8_best_rigid_um': float(prior_rigid.min()),
                   'refined_top8_best_rigid_um': float(refined_rigid.min()),
                   'fitted_selected_rigid_um': float(refined_rigid[selected_branch]),
                   'fitted_selected_mapped_um': float(mapped_error[0]),
                   'exact_pose_mapped_um': float(mapped_error[1]),
                   'prior_branch': int(choice[0]), 'fitted_branch': int(choice[selected_branch])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in weak_real:
            index = record['array_row_index']
            input_array = np.concatenate((real_images[index].astype(np.float32),
                                          np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(input_array).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[index])
            weights = torch.ones_like(offsets)
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction, prior, choice, reflection, refined_state, fitted = infer(image, offsets, weights)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, model.modes, 2)
            all_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            prior_five = (points(all_states, flags, corners) - reference).norm(dim=-1).mean(-1).flatten()
            refined_five = (points(refined_state, reflection, corners) - reference).norm(dim=-1).mean(-1)[0]
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                   'prior_five_um': float(prior_five[int(prior.argmax())]),
                   'refined_five_um': float(refined_five[int(fitted.argmax())]),
                   'prior_branch': int(prior.argmax()), 'fitted_branch': int(choice[int(fitted.argmax())])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': len(synthetic), 'real_weak': len(weak_real)}), flush=True)

summary = []
for step in steps:
    for split, metrics in (('synthetic', ('prior_top1_rigid_um', 'prior_top8_best_rigid_um',
                                         'refined_top8_best_rigid_um', 'fitted_selected_rigid_um',
                                         'fitted_selected_mapped_um', 'exact_pose_mapped_um')),
                           ('real_weak_allen', ('prior_five_um', 'refined_five_um'))):
        group = [row for row in rows if row['set'] == split and row['step'] == step]
        identities = sorted({row['animal_id'] for row in group})
        summary.append({'step': step, 'set': split, 'rows': len(group), 'identities': len(identities),
                        'identity_equal_mean_um': {metric: float(np.mean([
                            np.mean([row[metric] for row in group if row['animal_id'] == identity])
                            for identity in identities])) for metric in metrics},
                        'case_median_um': {metric: float(np.median([row[metric] for row in group]))
                                           for metric in metrics}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'steps': steps, 'rows': len(rows), 'synthetic_per_step': len(synthetic),
    'weak_real_per_step': len(weak_real),
    'checkpoint_sha256': {str(step): sha(training / f'joint_step_{step:05d}.pt') for step in steps},
    'panel_sha256': {name: sha(panel / name) for name in ('records.jsonl', 'protocol.json', 'completed.json')},
    'selected_panel_file_sha256': panel_files,
    'real_sha256': {name: sha(real / name) for name in ('records.jsonl', 'geometry.npz', 'images.npy')},
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'scope': 'TRAIN-inner synthetic development; donor-separated weak Allen development, not physical truth',
}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
