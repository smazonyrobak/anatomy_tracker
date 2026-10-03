"""Frozen synthetic-identity and weak-real-donor development evaluation of 041."""
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
from training.pose_feedback_global_041 import PoseFeedbackGlobal041

run = root / 'runs/pose_feedback_global_041_pilot'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/pose_feedback_global_041_development_eval'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
steps, side, beam = (0, 500, 1000, 1500, 2000), 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


completed = json.loads((run / 'completed.json').read_text())
assert completed['updates'] == 2000 and completed['accepted_synthetic'] == 4000
assert completed['unique_real_weak'] == 2000
panel_records = list(map(json.loads, (panel / 'records.jsonl').open()))
synthetic = [row for row in panel_records if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(panel_records) == 256 and len(synthetic) == 177
assert len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
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
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'), 'parent_sha256': sha(parent),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'checkpoints_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                                 for step in steps},
          'steps': steps, 'synthetic_eligible_sections': len(synthetic),
          'synthetic_subjects': 8, 'real_weak_sections': 64, 'real_weak_donors': 6,
          'synthetic_truth': 'dense observed tissue-to-CCF mapping',
          'real_truth': 'inherited Allen five-point affine, not expert-oblique truth',
          'source_sha256': sha(Path(__file__)), 'calibrated': False,
          'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi',
        frame[..., :2] @ basis, chart - .5)


def candidates(image):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = prior.topk(beam, -1).indices
    states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    return prediction, prior, choice, states, choice % 2


rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        state = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert state['step'] == step and not state['calibrated']
        head.load_state_dict(state['head'])
        del state
        for record in synthetic:
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                true_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, prior, choice, states, reflection = candidates(image)
            result = head(prediction, image, states, reflection, atlas, offsets, weights)
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            truth = target.reshape(-1, 3)[ids]
            before = (points(states, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            raw = (points(result['raw_fitted_state'], reflection, chart) - truth[None, None]
                   ).norm(dim=-1).mean(-1)[0]
            after = (points(result['state'], reflection, chart) - truth[None, None]
                     ).norm(dim=-1).mean(-1)[0]
            true_normal = full_frame_state_to_components(true_state)[1][0, :, 2]
            normal = full_frame_state_to_components(result['state'])[1][0, :, :, 2]
            normal_angle = torch.rad2deg(torch.acos((normal * true_normal).sum(-1).abs().clamp(0, 1)))
            score = prior.gather(1, choice) + result['score_delta']
            picked = int(score[0].argmax())
            truth_grid = target[8::16, 8::16].reshape(256, 3)
            valid_grid = valid[8::16, 8::16].reshape(256)
            key_distance = torch.cdist(truth_grid[None].expand(beam, -1, -1),
                                       result['key_world'][0]).amin(-1)
            eligible_match = valid_grid[None] & (key_distance <= 1500)
            match_error = (result['matched_world'][0] - truth_grid[None]).norm(dim=-1)
            match_recall = (((match_error <= 1500) & eligible_match).sum(-1) /
                            eligible_match.sum(-1).clamp_min(1))
            initial_best = int(before.argmin())
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')},
                   'prior_top1_um': float(before[0]), 'prior_best8_um': float(before.min()),
                   'raw_top1_um': float(raw[0]), 'raw_best8_um': float(raw.min()),
                   'refined_top1_um': float(after[0]), 'refined_best8_um': float(after.min()),
                   'refined_selected_um': float(after[picked]),
                   'normal_angle_top1_deg': float(normal_angle[0]),
                   'normal_angle_selected_deg': float(normal_angle[picked]),
                   'top1_correction_gate': float(result['correction_gate'][0, 0]),
                   'selected_correction_gate': float(result['correction_gate'][0, picked]),
                   'top1_match_recall_1500': float(match_recall[0]),
                   'prior_best8_match_recall_1500': float(match_recall[initial_best]),
                   'prior_branch': int(choice[0, 0]), 'refined_branch': int(choice[0, picked])}
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
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                        + 192 * corners[:, 1:] * affine[:, 1]
            prediction, prior, choice, states, reflection = candidates(image)
            result = head(prediction, image, states, reflection, atlas, offsets, weights)
            before = (points(states, reflection, corners[None]) - reference[None, None]
                      ).norm(dim=-1).mean(-1)[0]
            raw = (points(result['raw_fitted_state'], reflection, corners[None]) - reference[None, None]
                   ).norm(dim=-1).mean(-1)[0]
            after = (points(result['state'], reflection, corners[None]) - reference[None, None]
                     ).norm(dim=-1).mean(-1)[0]
            score = prior.gather(1, choice) + result['score_delta']
            picked = int(score[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id')},
                   'prior_top1_um': float(before[0]), 'prior_best8_um': float(before.min()),
                   'raw_top1_um': float(raw[0]), 'raw_best8_um': float(raw.min()),
                   'refined_top1_um': float(after[0]), 'refined_best8_um': float(after.min()),
                   'refined_selected_um': float(after[picked]),
                   'top1_correction_gate': float(result['correction_gate'][0, 0]),
                   'selected_correction_gate': float(result['correction_gate'][0, picked]),
                   'prior_branch': int(choice[0, 0]), 'refined_branch': int(choice[0, picked])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': len(synthetic),
                          'real_weak': len(real_records)}), flush=True)


def equal_group_mean(records, field, group):
    keys = sorted({row[group] for row in records})
    return float(np.mean([np.mean([row[field] for row in records if row[group] == key])
                          for key in keys]))


summary = []
for step in steps:
    synthetic_rows = [row for row in rows if row['step'] == step and row['set'] == 'synthetic']
    real_rows = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    entry = {'step': step, 'synthetic_sections': len(synthetic_rows),
             'real_weak_sections': len(real_rows)}
    for field in ('prior_top1_um', 'prior_best8_um', 'raw_top1_um', 'raw_best8_um',
                  'refined_top1_um', 'refined_best8_um', 'refined_selected_um',
                  'top1_correction_gate', 'selected_correction_gate'):
        entry['synthetic_' + field] = equal_group_mean(synthetic_rows, field,
                                                       'synthetic_subject_plan_id')
        entry['real_weak_' + field] = equal_group_mean(real_rows, field, 'animal_id')
    for field in ('normal_angle_top1_deg', 'normal_angle_selected_deg',
                  'top1_match_recall_1500', 'prior_best8_match_recall_1500'):
        entry['synthetic_' + field] = equal_group_mean(synthetic_rows, field,
                                                       'synthetic_subject_plan_id')
    entry['real_weak_by_donor_selected_um'] = {animal: float(np.mean([
        row['refined_selected_um'] for row in real_rows if row['animal_id'] == animal]))
        for animal in sorted({row['animal_id'] for row in real_rows})}
    entry['real_weak_by_donor_prior_top1_um'] = {animal: float(np.mean([
        row['prior_top1_um'] for row in real_rows if row['animal_id'] == animal]))
        for animal in sorted({row['animal_id'] for row in real_rows})}
    entry['synthetic_by_appearance'] = {appearance: {field: float(np.mean([
        row[field] for row in synthetic_rows if row['appearance_mode'] == appearance]))
        for field in ('prior_top1_um', 'prior_best8_um', 'refined_top1_um',
                      'refined_best8_um', 'refined_selected_um')}
        for appearance in sorted({row['appearance_mode'] for row in synthetic_rows})}
    near = [row for row in synthetic_rows if row['prior_best8_um'] <= 1000]
    entry['near_true_best8_sections'] = len(near)
    entry['near_true_prior_best8_um'] = float(np.mean([row['prior_best8_um'] for row in near]))
    entry['near_true_refined_best8_um'] = float(np.mean([row['refined_best8_um'] for row in near]))
    summary.append(entry)
for entry in summary:
    entry['synthetic_selected_gain_um'] = (entry['synthetic_prior_top1_um'] -
                                          entry['synthetic_refined_selected_um'])
    entry['synthetic_best8_gain_um'] = (entry['synthetic_prior_best8_um'] -
                                       entry['synthetic_refined_best8_um'])
    entry['max_donor_selected_regression_um'] = max(
        entry['real_weak_by_donor_selected_um'][donor] -
        entry['real_weak_by_donor_prior_top1_um'][donor]
        for donor in entry['real_weak_by_donor_selected_um'])
    entry['development_gate'] = (entry['step'] > 0 and
        entry['synthetic_selected_gain_um'] >= 250 and
        entry['synthetic_best8_gain_um'] >= 200 and
        entry['max_donor_selected_regression_um'] <= 200)
(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'synthetic_panel_sections': len(panel_records), 'synthetic_eligible_sections': len(synthetic),
    'synthetic_subjects': 8, 'real_weak_sections': len(real_records)}, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'gate': [row['development_gate'] for row in summary],
                  'summary': str(out / 'summary.json')}), flush=True)
