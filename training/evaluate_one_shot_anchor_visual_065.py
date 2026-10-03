"""Frozen 065 development readout; run only after all 8000 training batches exit."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_anchor_visual_065_pilot'
parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_anchor_visual_065_development_eval'
steps, side = (0, 3000, 8000), 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pose_points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum(
        'bkij,bkpj->bkpi', frame[..., :, :2] @ basis, chart - .5)


def candidates(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)
    states = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    return prior.gather(1, choice), choice, states, choice % 2


def mean_by(records, field, group):
    identities = sorted({row[group] for row in records})
    return float(np.mean([np.mean([row[field] for row in records if row[group] == identity])
                          for identity in identities]))


completed = json.loads((run / 'completed.json').read_text())
assert completed['batches'] == 8000
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
acquired = [row for row in map(json.loads, (real / 'records.jsonl').open())
            if row['training_split'] == 'development']
assert len(synthetic) == 246 and len({r['synthetic_subject_plan_id'] for r in synthetic}) == 8
assert len(acquired) == 64 and len({r['animal_id'] for r in acquired}) == 6
for record in synthetic:
    assert sha(panel / record['file']) == record['sha256']

real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
out.mkdir(parents=True, exist_ok=False)
config = {'run_completed_sha256': sha(run / 'completed.json'),
          'run_config_sha256': sha(run / 'config.json'), 'parent_sha256': sha(parent),
          'model_source_sha256': sha(Path(__file__).parent / 'arbitrary_plane_one_shot_model.py'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'real_images_sha256': sha(real / 'images.npy'),
          'real_geometry_sha256': sha(real / 'geometry.npz'),
          'checkpoints_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                                 for step in steps},
          'steps': steps, 'parent_step': -1,
          'candidate_policy': 'old prior top8 + anchor prior top6, reflection expanded',
          'synthetic_truth': 'visible tissue-to-CCF map of eight synthetic development identities',
          'real_reference': 'inherited weak Allen five-point affine, not expert oblique truth',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': sha(Path(__file__))}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []

with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in (-1, *steps):
        if step == 0:
            del model
            model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
                atlas_conditioning=True, fit_quality=True, vector_refinement=True,
                candidate_ranking=True, fitted_ranking=True,
                anchor_specific_features=True).cuda().eval()
        checkpoint = torch.load(parent if step == -1 else
            run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert (step == -1 or checkpoint['step'] == step) and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth_reflection = int(arrays['reflection'])
            prior, choice, state, reflection = candidates(model.predict(image))
            ids = valid.flatten().nonzero().flatten()
            chart = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                -1).float()[None] / side
            error = (pose_points(state, reflection, chart) -
                     target.reshape(-1, 3)[ids][None, None]).norm(dim=-1).mean(-1)[0]
            selected, best = int(prior[0].argmax()), int(error.argmin())
            centre, frame, basis = full_frame_state_to_components(state)
            true_centre, true_frame, true_basis = full_frame_state_to_components(truth_state)
            edge = frame[..., :, :2] @ basis
            true_edge = true_frame[..., :, :2] @ true_basis
            normal = torch.rad2deg(torch.acos((frame[0, :, :, 2] @
                true_frame[0, :, 2]).abs().clamp(max=1)))
            centre_mm = (centre[0] - true_centre[0]).norm(dim=-1) / 1000
            horizontal = F.normalize(edge[0, :, :, 0] *
                (1 - 2 * reflection[0, :, None].float()), dim=-1)
            vertical = F.normalize(edge[0, :, :, 1], dim=-1)
            true_horizontal = F.normalize(true_edge[0, :, 0] *
                (-1 if truth_reflection else 1), dim=-1)
            true_vertical = F.normalize(true_edge[0, :, 1], dim=-1)
            frame_angle = .5 * torch.rad2deg(torch.acos(
                (horizontal @ true_horizontal).clamp(-1, 1))) + .5 * torch.rad2deg(
                torch.acos((vertical @ true_vertical).clamp(-1, 1)))
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id', 'synthetic_subject_plan_id',
                       'appearance_mode', 'sha256')},
                   'visible_fraction': float(valid.float().mean()),
                   'selected_rigid_um': float(error[selected]),
                   'best14_rigid_um': float(error[best]),
                   'old_best8_rigid_um': float(error[:8].min()),
                   'anchor_best6_rigid_um': float(error[8:].min()),
                   'selected_anchor': selected >= 8,
                   'best14_anchor': best >= 8,
                   'selected_normal_deg': float(normal[selected]),
                   'best14_normal_deg': float(normal[best]),
                   'selected_centre_mm': float(centre_mm[selected]),
                   'best14_centre_mm': float(centre_mm[best]),
                   'selected_frame_deg': float(frame_angle[selected]),
                   'best14_frame_deg': float(frame_angle[best]),
                   'selected_reflection_correct': int(reflection[0, selected]) == truth_reflection,
                   'best14_reflection_correct': int(reflection[0, best]) == truth_reflection}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in acquired:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                        + 192 * corners[:, 1:] * affine[:, 1]
            prior, choice, state, reflection = candidates(model.predict(image))
            error = (pose_points(state, reflection, corners[None]) -
                     reference[None, None]).norm(dim=-1).mean(-1)[0]
            selected = int(prior[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'selected_five_um': float(error[selected]),
                   'best14_five_um': float(error.min()),
                   'selected_anchor': selected >= 8,
                   'thickness_um': float(thickness[i])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic': len(synthetic),
                          'real_weak': len(acquired)}), flush=True)

summary = []
for step in (-1, *steps):
    syn = [row for row in rows if row['step'] == step and row['set'] == 'synthetic']
    weak = [row for row in rows if row['step'] == step and row['set'] == 'real_weak_allen']
    entry = {'step': step, 'source': '059_parent' if step == -1 else '065_checkpoint',
             'synthetic_sections': len(syn), 'synthetic_identities': 8,
             'real_weak_sections': len(weak), 'real_weak_donors': 6}
    for field in ('selected_rigid_um', 'best14_rigid_um', 'old_best8_rigid_um',
                  'anchor_best6_rigid_um', 'selected_anchor', 'best14_anchor',
                  'selected_normal_deg', 'best14_normal_deg',
                  'selected_centre_mm', 'best14_centre_mm',
                  'selected_frame_deg', 'best14_frame_deg',
                  'selected_reflection_correct', 'best14_reflection_correct'):
        entry['synthetic_' + field] = mean_by(syn, field, 'synthetic_subject_plan_id')
    for field in ('selected_five_um', 'best14_five_um', 'selected_anchor'):
        entry['real_weak_' + field] = mean_by(weak, field, 'animal_id')
    entry['synthetic_by_appearance'] = {mode: {
        'sections': len([row for row in syn if row['appearance_mode'] == mode]),
        'identities': len({row['synthetic_subject_plan_id'] for row in syn
                           if row['appearance_mode'] == mode}),
        **{field: mean_by([row for row in syn if row['appearance_mode'] == mode],
                          field, 'synthetic_subject_plan_id')
           for field in ('selected_rigid_um', 'best14_rigid_um', 'selected_anchor')}}
        for mode in sorted({row['appearance_mode'] for row in syn})}
    entry['real_weak_by_donor'] = {animal: float(np.mean([
        row['selected_five_um'] for row in weak if row['animal_id'] == animal]))
        for animal in sorted({row['animal_id'] for row in weak})}
    summary.append(entry)

baseline = summary[0]
for entry in summary:
    entry['max_real_donor_regression_um'] = max(
        error - baseline['real_weak_by_donor'][animal]
        for animal, error in entry['real_weak_by_donor'].items())
    entry['development_gate'] = (entry['step'] > 0 and
        entry['synthetic_best14_rigid_um'] <= 770 and
        entry['synthetic_selected_rigid_um'] <= 2400 and
        entry['max_real_donor_regression_um'] <= 200)

(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'gate_thresholds_um': {'synthetic_best14': 770, 'synthetic_selected': 2400,
                           'max_weak_real_donor_regression': 200},
    'real_reference_is_weak': True, 'real_animal_validation': False,
    'public_benchmark_used': False}, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'hashes_sha256': {name: sha(out / name) for name in
                      ('config.json', 'rows.jsonl', 'summary.json')},
    'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'gate': [row['development_gate'] for row in summary],
                  'summary': str(out / 'summary.json')}), flush=True)
