"""Fixed synthetic and weak-real physical readout for per-candidate pose feedback."""
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

run = root / 'runs/pose_feedback_036'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/pose_feedback_036_development_eval'
side, fit_side, beam = 256, 64, 8
steps = (0, 1000, 2000, 3000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert json.loads((run / 'completed.json').read_text())['updates'] == 3000
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic) == 185 and len(real_records) == 64
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'checkpoints_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in steps},
          'steps': steps, 'synthetic_sections': 185, 'real_weak_sections': 64,
          'synthetic_label': 'dense observed tissue-to-CCF truth',
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
    first = model.map(selected, offsets, branch, reflection, (fit_side, fit_side),
                      atlas, weights, return_refinement_feature=True,
                      feature_side=fit_side, source_shape=(side, side))
    refined, score_delta, update = model.refine(first['refinement_feature'], states)
    score = prior.gather(1, choice) + score_delta
    return states, refined, reflection, choice, score, update


rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'])
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
            states, refined, reflection, choice, score, update = candidates(image, offsets, weights)
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            truth = target.reshape(-1, 3)[ids]
            before = (points(states, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            after = (points(refined, reflection, chart) - truth[None, None]).norm(dim=-1).mean(-1)[0]
            picked = int(score[0].argmax())
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                       'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')},
                   'valid_fraction': float(valid.float().mean()),
                   'prior_top1_um': float(before[0]), 'refined_top1_um': float(after[0]),
                   'initial_best8_um': float(before.min()), 'refined_best8_um': float(after.min()),
                   'refined_selected_um': float(after[picked]),
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
            states, refined, reflection, choice, score, update = candidates(image, offsets, weights)
            before = (points(states, reflection, corners[None]) - reference[None, None]).norm(dim=-1).mean(-1)[0]
            after = (points(refined, reflection, corners[None]) - reference[None, None]).norm(dim=-1).mean(-1)[0]
            picked = int(score[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id')},
                   'prior_top1_um': float(before[0]), 'refined_top1_um': float(after[0]),
                   'initial_best8_um': float(before.min()), 'refined_best8_um': float(after.min()),
                   'refined_selected_um': float(after[picked]),
                   'mean_translation_update_um': float(update[0, :, 3:6].norm(dim=-1).mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': 185, 'real_weak': 64}), flush=True)


def equal_group_mean(records, field, group):
    keys = sorted({row[group] for row in records})
    return float(np.mean([np.mean([row[field] for row in records if row[group] == key]) for key in keys]))


summary = []
for step in steps:
    synthetic_rows = [row for row in rows if row['step'] == step and row['set'] == 'synthetic']
    real_rows = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    entry = {'step': step, 'synthetic_sections': len(synthetic_rows), 'real_weak_sections': len(real_rows)}
    for field in ('prior_top1_um', 'refined_top1_um', 'initial_best8_um',
                  'refined_best8_um', 'refined_selected_um', 'mean_translation_update_um'):
        entry['synthetic_' + field] = equal_group_mean(synthetic_rows, field, 'synthetic_subject_plan_id')
        entry['real_weak_' + field] = equal_group_mean(real_rows, field, 'animal_id')
    entry['real_weak_by_donor_refined_top1_um'] = {animal: float(np.mean([
        row['refined_top1_um'] for row in real_rows if row['animal_id'] == animal]))
        for animal in sorted({row['animal_id'] for row in real_rows})}
    summary.append(entry)
base = summary[0]
for row in summary:
    regressions = [row['real_weak_by_donor_refined_top1_um'][donor] -
                   base['real_weak_by_donor_refined_top1_um'][donor]
                   for donor in base['real_weak_by_donor_refined_top1_um']]
    row['max_donor_refined_top1_regression_um'] = float(max(regressions))
    row['development_gate'] = (row['step'] > 0 and
        row['synthetic_refined_top1_um'] <= base['synthetic_refined_top1_um'] - 250 and
        row['synthetic_refined_best8_um'] <= base['synthetic_refined_best8_um'] - 200 and
        row['max_donor_refined_top1_regression_um'] <= 200)
(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'any_development_gate': any(row['development_gate'] for row in summary),
    'scope': 'historical synthetic DEV and weak-real donor DEV only; not qualification'}, indent=2))
(out / 'completed.json').write_text(json.dumps({name + '_sha256': sha(out / f'{name}.json')
    for name in ('config', 'summary')} | {'rows_sha256': sha(out / 'rows.jsonl')}, indent=2))
print(json.dumps({'event': 'complete', 'checkpoints': summary}), flush=True)
