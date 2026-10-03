"""Frozen identity-disjoint synthetic and donor-disjoint weak-real 037 comparison."""
import hashlib
import json
import math
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
from training.pose_feedback_corr_037 import PoseFeedbackCorrelation037

run = root / 'runs/pose_feedback_corr_037'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/pose_feedback_corr_037_development_eval'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
side, fit_side, beam = 256, 96, 8
steps = (0, 1000, 2000, 4000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


completed = json.loads((run / 'completed.json').read_text())
assert completed['updates'] == 4000
panel_records = list(map(json.loads, (panel / 'records.jsonl').open()))
synthetic = [row for row in panel_records if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(panel_records) == 256
panel_groups = {row['synthetic_subject_plan_id'] for row in panel_records}
assert len(panel_groups) == 8
assert all(sum(row['synthetic_subject_plan_id'] == identity for row in panel_records) == 32
           for identity in panel_groups)
assert {row['synthetic_subject_plan_id'] for row in synthetic} == panel_groups
assert len({row['panel_physical_section_id'] for row in panel_records}) == 256
assert len({row['subject_ouv_sha256'] for row in panel_records}) == 256
panel_animals = {row['synthetic_animal_id'] for row in panel_records}
assert len(panel_animals) == 8
train_animals = {row['base_lineage']['synthetic_animal_id']
                 for row in map(json.loads, (run / 'draws.jsonl').open())}
assert panel_animals.isdisjoint(train_animals)
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'])
del checkpoint
atlas_head = PoseFeedbackCorrelation037(use_atlas=True).cuda().eval()
control_head = PoseFeedbackCorrelation037(use_atlas=False).cuda().eval()
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'), 'parent_sha256': sha(parent),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'checkpoints_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in steps},
          'steps': steps, 'synthetic_panel_sections': len(panel_records),
          'synthetic_eligible_sections': len(synthetic),
          'synthetic_ineligible_sections': len(panel_records) - len(synthetic),
          'synthetic_eligible_by_subject': {identity: sum(
              row['synthetic_subject_plan_id'] == identity for row in synthetic)
              for identity in sorted(panel_groups)},
          'synthetic_subjects': 8,
          'real_weak_sections': 64, 'real_weak_donors': 6,
          'synthetic_label': 'dense observed tissue-to-CCF truth on new synthetic deformation identities',
          'real_label': 'inherited Allen five-point affine, not expert-oblique truth',
          'source_sha256': sha(Path(__file__)), 'calibrated': False,
          'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi', frame[..., :2] @ basis, chart - .5)


def candidates(image, offsets, weights):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = prior.topk(beam, -1).indices
    states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    reflection = choice % 2
    selected = {**prediction, 'state': states}
    branch = torch.arange(beam, device='cuda')[None]
    mapped = model.map(selected, offsets, branch, reflection, (fit_side, fit_side),
                       atlas, weights, return_refinement_feature=True,
                       feature_side=fit_side, source_shape=(side, side))
    atlas_feature = mapped['refinement_feature'][:, :, 64:128]
    return prediction, prior, choice, states, reflection, mapped, atlas_feature


rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        state = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert state['step'] == step and not state['calibrated']
        atlas_head.load_state_dict(state['atlas_head'])
        control_head.load_state_dict(state['control_head'])
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
            prediction, prior, choice, states, reflection, mapped, atlas_feature = candidates(
                image, offsets, weights)
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            truth = target.reshape(-1, 3)[ids]
            before = (points(states, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            true_normal = full_frame_state_to_components(true_state)[1][0, :, 2]
            for arm, head in (('atlas', atlas_head), ('control', control_head)):
                refined, delta, update = head(prediction, mapped, image, states,
                                              reflection, atlas_feature)
                after = (points(refined, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
                normal = full_frame_state_to_components(refined)[1][0, :, 2]
                normal_angle = torch.rad2deg(torch.acos((normal * true_normal).sum(-1).abs().clamp(0, 1)))
                score = prior.gather(1, choice) + delta
                picked = int(score[0].argmax())
                row = {'set': 'synthetic', 'step': step, 'arm': arm,
                       **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                           'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')},
                       'valid_fraction': float(valid.float().mean()),
                       'prior_top1_um': float(before[0]), 'refined_top1_um': float(after[0]),
                       'initial_best8_um': float(before.min()), 'refined_best8_um': float(after.min()),
                       'refined_selected_um': float(after[picked]),
                       'normal_angle_top1_deg': float(normal_angle[0]),
                       'normal_angle_selected_deg': float(normal_angle[picked]),
                       'prior_branch': int(choice[0, 0]), 'refined_branch': int(choice[0, picked]),
                       'mean_translation_update_um': float(update[0, :, 3:6].norm(dim=-1).mean())}
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
            prediction, prior, choice, states, reflection, mapped, atlas_feature = candidates(
                image, offsets, weights)
            before = (points(states, reflection, corners[None]) - reference[None, None]).norm(dim=-1).mean(-1)[0]
            for arm, head in (('atlas', atlas_head), ('control', control_head)):
                refined, delta, update = head(prediction, mapped, image, states,
                                              reflection, atlas_feature)
                after = (points(refined, reflection, corners[None]) - reference[None, None]).norm(dim=-1).mean(-1)[0]
                score = prior.gather(1, choice) + delta
                picked = int(score[0].argmax())
                row = {'set': 'real_weak_allen', 'step': step, 'arm': arm,
                       **{key: record[key] for key in ('animal_id', 'specimen_id',
                           'experiment_id', 'section_id')},
                       'prior_top1_um': float(before[0]), 'refined_top1_um': float(after[0]),
                       'initial_best8_um': float(before.min()), 'refined_best8_um': float(after.min()),
                       'refined_selected_um': float(after[picked]),
                       'mean_translation_update_um': float(update[0, :, 3:6].norm(dim=-1).mean())}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': len(synthetic),
                          'synthetic_ineligible': len(panel_records) - len(synthetic),
                          'real_weak': 64, 'arms': 2}), flush=True)


def equal_group_mean(records, field, group):
    keys = sorted({row[group] for row in records})
    return float(np.mean([np.mean([row[field] for row in records if row[group] == key]) for key in keys]))


summary = []
for step in steps:
    for arm in ('atlas', 'control'):
        synthetic_rows = [row for row in rows if row['step'] == step and row['arm'] == arm
                          and row['set'] == 'synthetic']
        real_rows = [row for row in rows if row['step'] == step and row['arm'] == arm
                     and row['set'] == 'real_weak_allen']
        entry = {'step': step, 'arm': arm, 'synthetic_sections': len(synthetic_rows),
                 'real_weak_sections': len(real_rows)}
        for field in ('prior_top1_um', 'refined_top1_um', 'initial_best8_um',
                      'refined_best8_um', 'refined_selected_um',
                      'mean_translation_update_um'):
            entry['synthetic_' + field] = equal_group_mean(
                synthetic_rows, field, 'synthetic_subject_plan_id')
            entry['real_weak_' + field] = equal_group_mean(real_rows, field, 'animal_id')
        for field in ('normal_angle_top1_deg', 'normal_angle_selected_deg'):
            entry['synthetic_' + field] = equal_group_mean(
                synthetic_rows, field, 'synthetic_subject_plan_id')
        entry['real_weak_by_donor_refined_top1_um'] = {animal: float(np.mean([
            row['refined_top1_um'] for row in real_rows if row['animal_id'] == animal]))
            for animal in sorted({row['animal_id'] for row in real_rows})}
        entry['real_weak_by_donor_refined_selected_um'] = {animal: float(np.mean([
            row['refined_selected_um'] for row in real_rows if row['animal_id'] == animal]))
            for animal in sorted({row['animal_id'] for row in real_rows})}
        summary.append(entry)
base = next(row for row in summary if row['step'] == 0 and row['arm'] == 'atlas')
for entry in summary:
    control = next(row for row in summary if row['step'] == entry['step'] and row['arm'] == 'control')
    entry['max_donor_refined_selected_regression_um'] = max(
        entry['real_weak_by_donor_refined_selected_um'][donor] -
        base['real_weak_by_donor_refined_selected_um'][donor]
        for donor in base['real_weak_by_donor_refined_selected_um'])
    entry['atlas_advantage_selected_um'] = (control['synthetic_refined_selected_um'] -
                                            entry['synthetic_refined_selected_um']) if entry['arm'] == 'atlas' else None
    entry['development_gate'] = (entry['arm'] == 'atlas' and entry['step'] > 0 and
        entry['synthetic_refined_selected_um'] <= base['synthetic_refined_selected_um'] - 250 and
        entry['synthetic_refined_best8_um'] <= base['synthetic_refined_best8_um'] - 200 and
        entry['atlas_advantage_selected_um'] >= 150 and
        entry['max_donor_refined_selected_regression_um'] <= 200)
(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'synthetic_panel_sections': len(panel_records),
    'synthetic_eligible_sections': len(synthetic),
    'synthetic_ineligible_sections': len(panel_records) - len(synthetic),
    'any_development_gate': any(row['development_gate'] for row in summary),
    'scope': 'new synthetic deformation identity DEV and weak-real donor DEV only; not qualification'}, indent=2))
(out / 'completed.json').write_text(json.dumps({name + '_sha256': sha(out / f'{name}.json')
    for name in ('config', 'summary')} | {'rows_sha256': sha(out / 'rows.jsonl')}, indent=2))
print(json.dumps({'event': 'complete', 'checkpoints': summary}), flush=True)
