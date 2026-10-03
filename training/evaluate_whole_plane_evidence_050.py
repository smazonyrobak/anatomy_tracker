"""Frozen truth-blind branch-selection readout for whole-plane evidence."""
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
from training.whole_plane_evidence_050 import WholePlaneEvidence050

train = root / 'runs/whole_plane_evidence_050_pilot'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/whole_plane_evidence_050_development_eval'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
steps, beam = (0, 1000, 3000, 5000), 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((train / 'completed.json').read_text())['batches'] == 5000
synthetic = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
real_dev = [row for row in map(json.loads, (real / 'records.jsonl').open())
            if row['training_split'] == 'development']
assert len(synthetic) == 177 and len(real_dev) == 64
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                             vector_refinement=True, candidate_ranking=True,
                             fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
pose.requires_grad_(False)
models = []
for step in steps:
    checkpoint = torch.load(train / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and not checkpoint['calibrated']
    model = WholePlaneEvidence050().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    models.append(model)
del checkpoint
out.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
config = {'steps': steps, 'beam': beam, 'parent_sha256': sha(parent),
          'training_completed_sha256': sha(train / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'checkpoints_sha256': {str(step): sha(train / f'joint_step_{step:05d}.pt') for step in steps},
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_whole_plane_evidence_050.py', 'whole_plane_evidence_050.py')},
          'real_label_role': 'inherited weak Allen affine, not expert arbitrary-plane truth',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / 256


def physical_points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
        frame[..., :2] @ basis, chart - .5)


rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for subset, records in (('synthetic', synthetic), ('real_weak_allen', real_dev)):
        for record in records:
            if subset == 'synthetic':
                path = panel / record['file']
                assert sha(path) == record['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    truth_reflection = torch.from_numpy(arrays['reflection'][None].copy()).cuda()
                    target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                ids = valid.flatten().nonzero().flatten()
                chart = torch.stack((ids.remainder(256),
                    ids.div(256, rounding_mode='floor')), -1).float()[None] / 256
                reference = target.reshape(-1, 3)[ids]
            else:
                index = record['array_row_index']
                image = np.concatenate((real_images[index].astype(np.float32),
                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(image).cuda(), (256, 256),
                                      mode='bilinear', align_corners=False)
                offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[index])
                weights = torch.ones_like(offsets)
                weights[:, [0, -1]] = .5
                weights /= weights.sum(-1, keepdim=True)
                chart = corners[None]
                affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                            + 192 * corners[:, 1:] * affine[:, 1]
            prediction = pose.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            error = (physical_points(state, reflection, chart) - reference[None, None]).norm(dim=-1).mean(-1)
            if subset == 'synthetic':
                evaluated_state = torch.cat((state, truth[:, None]), 1)
                evaluated_reflection = torch.cat((reflection, truth_reflection[:, None]), 1)
            else:
                evaluated_state, evaluated_reflection = state, reflection
            for step, model in zip(steps, models):
                score, support = model(image, evaluated_state, evaluated_reflection,
                                       atlas, offsets, weights)
                selected = (prior.gather(1, choice) + score[:, :beam]).argmax(-1)
                hard = (error > 2000) & ((support[:, :beam] - support[:, beam:beam + 1]).abs() < .10) \
                       if subset == 'synthetic' else None
                row = {'set': subset, 'step': step,
                       **{key: record[key] for key in ('animal_id', 'specimen_id',
                           'experiment_id', 'section_id')},
                       'prior_rigid_um': float(error[0, 0]),
                       'best_eight_rigid_um': float(error.min()),
                       'selected_rigid_um': float(error.gather(1, selected[:, None]).mean()),
                       'selected_rank': int(selected[0]),
                       'choice': choice[0].cpu().tolist(),
                       'branch_error_um': error[0].cpu().tolist(),
                       'branch_prior': prior.gather(1, choice)[0].cpu().tolist(),
                       'branch_score': score[0, :beam].cpu().tolist()}
                if subset == 'synthetic':
                    row.update(synthetic_subject_plan_id=record['synthetic_subject_plan_id'],
                               appearance_mode=record['appearance_mode'], sha256=record['sha256'],
                               support_matched_hard=int(hard.sum()),
                               teacher_beats_hard=int(((score[:, beam:beam + 1] > score[:, :beam]) & hard).sum()))
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'set': subset, 'sections': len(records)}), flush=True)

summary = {'synthetic_eligible_sections': len(synthetic), 'synthetic_subjects': 8,
           'real_weak_sections': len(real_dev), 'real_weak_donors': 6,
           'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
for step in steps:
    summary[str(step)] = {}
    for subset, identity in (('synthetic', 'synthetic_subject_plan_id'),
                             ('real_weak_allen', 'animal_id')):
        selected = [row for row in rows if row['step'] == step and row['set'] == subset]
        identities = sorted({row[identity] for row in selected})
        stats = {field: float(np.mean([np.mean([row[field] for row in selected if row[identity] == name])
                     for name in identities])) for field in ('prior_rigid_um', 'best_eight_rigid_um',
                                                               'selected_rigid_um')}
        if subset == 'synthetic':
            stats['teacher_beats_support_matched_hard'] = float(np.mean([
                sum(row['teacher_beats_hard'] for row in selected if row[identity] == name) /
                max(1, sum(row['support_matched_hard'] for row in selected if row[identity] == name))
                for name in identities]))
            stats['appearance_discrimination'] = {mode: float(sum(row['teacher_beats_hard'] for row in selected
                if row['appearance_mode'] == mode) / max(1, sum(row['support_matched_hard'] for row in selected
                if row['appearance_mode'] == mode))) for mode in ('raw', 'exact_black', 'imperfect_brush')}
        else:
            stats['worst_donor_regression_um'] = float(max(np.mean([row['selected_rigid_um'] -
                row['prior_rigid_um'] for row in selected if row['animal_id'] == name]) for name in identities))
        summary[str(step)][subset] = stats
for step in steps[1:]:
    synthetic_stats = summary[str(step)]['synthetic']
    real_stats = summary[str(step)]['real_weak_allen']
    summary[str(step)]['useful_gate'] = bool(
        synthetic_stats['selected_rigid_um'] <= 2000 and
        synthetic_stats['prior_rigid_um'] - synthetic_stats['selected_rigid_um'] >= 400 and
        min(synthetic_stats['appearance_discrimination'].values()) >= .8 and
        real_stats['worst_donor_regression_um'] <= 200)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'gates': {str(s): summary[str(s)]['useful_gate']
                  for s in steps[1:]}}), flush=True)
