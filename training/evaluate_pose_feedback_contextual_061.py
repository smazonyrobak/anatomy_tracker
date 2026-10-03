"""Frozen 061 evaluation on new synthetic identities and separate weak-real donors."""
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
from training.pose_feedback_contextual_061 import PoseFeedbackContextual061

run = root / 'runs/pose_feedback_contextual_061_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/pose_feedback_contextual_061_development_eval'
parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
steps, side = (0, 1000, 3000, 6000), 256
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
                                255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi',
        frame[..., :, :2] @ basis, chart - .5)


def candidates(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)
    states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    return prior, choice, states, choice % 2


completed = json.loads((run / 'completed.json').read_text())
assert completed['batches'] == 6000 and completed['accepted_synthetic'] == 12000
assert completed['unique_real_weak'] == 6000
panel_records = list(map(json.loads, (panel / 'records.jsonl').open()))
synthetic = [row for row in panel_records if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(panel_records) == 256 and len(synthetic) > 0
assert len({row['synthetic_subject_plan_id'] for row in panel_records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
head = PoseFeedbackContextual061().cuda().eval()
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'), 'parent_sha256': sha(parent),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'checkpoints_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                                 for step in steps},
          'steps': steps, 'synthetic_eligible_sections': len(synthetic),
          'synthetic_subjects': 8, 'real_weak_sections': 64, 'real_weak_donors': 6,
          'source_sha256': sha(Path(__file__)), 'synthetic_truth':
          'observed tissue-to-CCF map from independent synthetic deformations',
          'real_reference': 'inherited weak Allen five-point affine; not expert oblique truth',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
rows = []

with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        head.load_state_dict(checkpoint['head'])
        del checkpoint
        for record in synthetic:
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction = model.predict(image)
            prior, choice, states, reflection = candidates(prediction)
            result = head(prediction, image, states, reflection, atlas, offsets, weights)
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            truth = target.reshape(-1, 3)[ids]
            before = (points(states, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            after = (points(result['state'], reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            scores = prior.gather(1, choice) + result['score_delta']
            prior_selected = int(prior.gather(1, choice)[0].argmax())
            selected = int(scores[0].argmax())
            best_prior = int(before.argmin())
            truth_grid = target[8::16, 8::16].reshape(256, 3)
            valid_grid = valid[8::16, 8::16].reshape(256)
            distances = torch.cdist(truth_grid[None].expand(14, -1, -1),
                                    result['key_world'][0])
            support = result['key_support'][0] > .1
            available500 = ((distances <= 500) & support[:, None]).any(-1) & valid_grid[None]
            available1500 = ((distances <= 1500) & support[:, None]).any(-1) & valid_grid[None]
            match_error = (result['matched_world'][0] - truth_grid[None]).norm(dim=-1)
            denominator = valid_grid.sum().clamp_min(1)
            mapped_prior = model.map(prediction, offsets,
                (choice[:, prior_selected:prior_selected + 1] // 2),
                reflection[:, prior_selected:prior_selected + 1], (side, side), atlas, weights)
            selected_prediction = {**prediction,
                'state': result['state'][:, selected:selected + 1]}
            mapped_selected = model.map(selected_prediction, offsets,
                torch.zeros((1, 1), device='cuda', dtype=torch.long),
                reflection[:, selected:selected + 1], (side, side), atlas, weights)
            mapped_prior_error = (mapped_prior['centre_surface_ccf_ap_dv_ml_um'][0, 0, valid]
                                  - target[valid]).norm(dim=-1).mean()
            mapped_selected_error = (mapped_selected['centre_surface_ccf_ap_dv_ml_um'][0, 0, valid]
                                     - target[valid]).norm(dim=-1).mean()
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id', 'synthetic_subject_plan_id',
                       'appearance_mode', 'sha256')},
                   'visible_fraction': float(valid.float().mean()),
                   'parent_selected_um': float(before[prior_selected]),
                   'parent_best14_um': float(before.min()),
                   'fitted_selected_um': float(after[selected]),
                   'fitted_best14_um': float(after.min()),
                   'parent_mapped_um': float(mapped_prior_error),
                   'fitted_selected_mapped_um': float(mapped_selected_error),
                   'prior_best_match_500': float(((match_error[best_prior] <= 500) &
                       valid_grid).sum() / denominator),
                   'prior_best_match_1500': float(((match_error[best_prior] <= 1500) &
                       valid_grid).sum() / denominator),
                   'prior_best_available_500': float(available500[best_prior].sum() / denominator),
                   'prior_best_available_1500': float(available1500[best_prior].sum() / denominator),
                   'selected_match_500': float(((match_error[selected] <= 500) &
                       valid_grid).sum() / denominator),
                   'selected_match_1500': float(((match_error[selected] <= 1500) &
                       valid_grid).sum() / denominator),
                   'prior_branch': int(choice[0, prior_selected]),
                   'fitted_branch': int(choice[0, selected])}
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
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                        + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            prior, choice, states, reflection = candidates(prediction)
            result = head(prediction, image, states, reflection, atlas, offsets, weights)
            before = (points(states, reflection, corners[None]) - reference[None, None]
                      ).norm(dim=-1).mean(-1)[0]
            after = (points(result['state'], reflection, corners[None]) - reference[None, None]
                     ).norm(dim=-1).mean(-1)[0]
            parent_index = int(prior.gather(1, choice)[0].argmax())
            selected = int((prior.gather(1, choice) + result['score_delta'])[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'parent_selected_um': float(before[parent_index]),
                   'parent_best14_um': float(before.min()),
                   'fitted_selected_um': float(after[selected]),
                   'fitted_best14_um': float(after.min()),
                   'prior_branch': int(choice[0, parent_index]),
                   'fitted_branch': int(choice[0, selected])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': len(synthetic),
                          'real_weak': len(real_records)}), flush=True)


def equal_group_mean(records, field, group):
    keys = sorted({row[group] for row in records})
    return float(np.mean([np.mean([row[field] for row in records if row[group] == key])
                          for key in keys]))


summary = []
for step in steps:
    syn = [row for row in rows if row['step'] == step and row['set'] == 'synthetic']
    acquired = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    entry = {'step': step, 'synthetic_sections': len(syn), 'real_weak_sections': len(acquired)}
    for field in ('parent_selected_um', 'parent_best14_um', 'fitted_selected_um',
                  'fitted_best14_um'):
        entry['synthetic_' + field] = equal_group_mean(syn, field, 'synthetic_subject_plan_id')
        entry['real_weak_' + field] = equal_group_mean(acquired, field, 'animal_id')
    for field in ('parent_mapped_um', 'fitted_selected_mapped_um',
                  'prior_best_match_500', 'prior_best_match_1500',
                  'prior_best_available_500', 'prior_best_available_1500',
                  'selected_match_500', 'selected_match_1500'):
        entry['synthetic_' + field] = equal_group_mean(syn, field, 'synthetic_subject_plan_id')
    entry['real_weak_by_donor'] = {animal: {
        field: float(np.mean([row[field] for row in acquired if row['animal_id'] == animal]))
        for field in ('parent_selected_um', 'fitted_selected_um')}
        for animal in sorted({row['animal_id'] for row in acquired})}
    entry['synthetic_by_appearance'] = {appearance: {
        field: float(np.mean([row[field] for row in syn if row['appearance_mode'] == appearance]))
        for field in ('parent_selected_um', 'fitted_selected_um',
                      'parent_mapped_um', 'fitted_selected_mapped_um')}
        for appearance in sorted({row['appearance_mode'] for row in syn})}
    entry['synthetic_selected_gain_um'] = (entry['synthetic_parent_selected_um'] -
                                           entry['synthetic_fitted_selected_um'])
    entry['synthetic_best14_gain_um'] = (entry['synthetic_parent_best14_um'] -
                                         entry['synthetic_fitted_best14_um'])
    entry['max_real_donor_regression_um'] = max(
        item['fitted_selected_um'] - item['parent_selected_um']
        for item in entry['real_weak_by_donor'].values())
    entry['development_gate'] = (step > 0 and entry['synthetic_selected_gain_um'] >= 250
        and entry['synthetic_best14_gain_um'] >= 200
        and entry['max_real_donor_regression_um'] <= 200)
    summary.append(entry)
(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'synthetic_panel_sections': len(panel_records), 'synthetic_eligible_sections': len(synthetic),
    'synthetic_subjects': 8, 'real_weak_sections': len(real_records)}, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'gate': [row['development_gate'] for row in summary],
                  'summary': str(out / 'summary.json')}), flush=True)
