"""Truth-blind local pose expansion with frozen whole-plane evidence."""
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
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_plane_evidence_050 import WholePlaneEvidence050

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/local_pose_search_051_diagnostic'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
evidence = root / 'runs/whole_plane_evidence_050_pilot/joint_step_03000.pt'
beam, chunk = 8, 16
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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
model = WholePlaneEvidence050().cuda().eval()
model.load_state_dict(torch.load(evidence, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)

moves = [torch.zeros(9)]
for dimension in range(3):
    for magnitude in (.12, .24, -.12, -.24):
        update = torch.zeros(9)
        update[dimension] = magnitude
        moves.append(update)
for dimension in range(3, 6):
    for magnitude in (500., 1000., -500., -1000.):
        update = torch.zeros(9)
        update[dimension] = magnitude
        moves.append(update)
for dimension in (6, 7, 8):
    for magnitude in (.05, -.05):
        update = torch.zeros(9)
        update[dimension] = magnitude
        moves.append(update)
moves = torch.stack(moves).cuda()
scale = moves.new_tensor((.24, .24, .24, 1000, 1000, 1000, .05, .05, .05))
penalty = .25 * (moves / scale).square().sum(-1)
proposals = beam * len(moves)
out.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
config = {'beam': beam, 'moves_per_branch': len(moves), 'proposals': proposals,
          'moves_local_so3_translation_logsize_shear': moves.cpu().tolist(),
          'move_penalty': penalty.cpu().tolist(), 'score': '019 branch prior + frozen 050 whole-plane evidence - move penalty',
          'parent_sha256': sha(parent), 'evidence_sha256': sha(evidence),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('evaluate_local_pose_search_051.py', 'whole_plane_evidence_050.py')},
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
            original_error = (physical_points(state, reflection, chart) - reference[None, None]).norm(dim=-1).mean(-1)
            expanded = compose_full_frame_state(
                state[:, :, None].expand(-1, -1, len(moves), -1),
                moves[None, None].expand(1, beam, -1, -1)).reshape(1, proposals, 12)
            reflected = reflection[:, :, None].expand(-1, -1, len(moves)).reshape(1, proposals)
            scores = torch.cat([model(image, expanded[:, first:first + chunk],
                reflected[:, first:first + chunk], atlas, offsets, weights)[0]
                for first in range(0, proposals, chunk)], 1)
            inherited = prior.gather(1, choice)[:, :, None].expand(-1, -1, len(moves)).reshape(1, proposals)
            selection_score = inherited + scores - penalty.repeat(beam)[None]
            selected = int(selection_score.argmax(-1))
            error = (physical_points(expanded, reflected, chart) - reference[None, None]).norm(dim=-1).mean(-1)
            row = {'set': subset,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id')},
                   'original_top1_um': float(original_error[0, 0]),
                   'original_best8_um': float(original_error.min()),
                   'expanded_selected_um': float(error[0, selected]),
                   'expanded_best_um': float(error.min()),
                   'selected_index': selected,
                   'selected_parent': selected // len(moves),
                   'selected_move': selected % len(moves),
                   'selected_score': float(selection_score[0, selected]),
                   'all_errors_um': error[0].cpu().tolist(),
                   'all_scores': selection_score[0].cpu().tolist()}
            if subset == 'synthetic':
                row.update(synthetic_subject_plan_id=record['synthetic_subject_plan_id'],
                           appearance_mode=record['appearance_mode'], sha256=record['sha256'])
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'set': subset, 'sections': len(records)}), flush=True)

summary = {'synthetic_eligible_sections': len(synthetic), 'synthetic_subjects': 8,
           'real_weak_sections': len(real_dev), 'real_weak_donors': 6,
           'proposals_per_section': proposals, 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
for subset, identity in (('synthetic', 'synthetic_subject_plan_id'),
                         ('real_weak_allen', 'animal_id')):
    selected = [row for row in rows if row['set'] == subset]
    identities = sorted({row[identity] for row in selected})
    summary[subset] = {field: float(np.mean([np.mean([row[field] for row in selected
        if row[identity] == name]) for name in identities])) for field in
        ('original_top1_um', 'original_best8_um', 'expanded_selected_um', 'expanded_best_um')}
    if subset == 'real_weak_allen':
        summary[subset]['worst_donor_regression_um'] = float(max(np.mean([
            row['expanded_selected_um'] - row['original_top1_um'] for row in selected
            if row['animal_id'] == name]) for name in identities))
summary['necessary_gate'] = bool(
    summary['synthetic']['expanded_best_um'] <= 600 and
    summary['synthetic']['original_best8_um'] - summary['synthetic']['expanded_best_um'] >= 300 and
    summary['synthetic']['expanded_selected_um'] <= 2000 and
    summary['synthetic']['original_top1_um'] - summary['synthetic']['expanded_selected_um'] >= 400 and
    summary['real_weak_allen']['worst_donor_regression_um'] <= 200)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'necessary_gate': summary['necessary_gate']}), flush=True)
