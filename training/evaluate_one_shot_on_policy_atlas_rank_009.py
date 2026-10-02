"""Frozen new-identity panel and weak-real readout of on-policy candidate scoring."""
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

run = root / 'runs/one_shot_on_policy_atlas_rank_009'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_on_policy_atlas_rank_009_development_eval'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates_per_arm'] == 8000
assert json.loads((panel / 'completed.json').read_text())['physical_sections'] == 256
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(records) == 256 and len({row['animal_id'] for row in records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


out.mkdir(parents=True, exist_ok=False)
rows, checkpoint_hashes = [], {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm in ('parent', 'direct', 'atlas'):
        path = (root / 'runs/one_shot_candidate_energy_004/joint_step_08000.pt'
                if arm == 'parent' else run / arm / 'joint_step_08000.pt')
        checkpoint_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == 8000 and not checkpoint['calibrated']
        model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                       candidate_ranking=arm != 'parent').cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in records:
            row = {'arm': arm, 'set': 'synthetic', 'eligible': record['eligible'],
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id',
                                                   'synthetic_subject_plan_id', 'valid_pixels',
                                                   'appearance_mode', 'sha256')}}
            if record['eligible']:
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    true_reflection = int(arrays['reflection'])
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                prediction = model.predict(image)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                indices = valid.flatten().nonzero().flatten()
                chart = torch.stack((indices.remainder(side),
                                     indices.div(side, rounding_mode='floor')), -1).float() / side
                target = reference.reshape(-1, 3)[indices]
                states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
                flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, 16, 2)
                geometry = (points(states, flags, chart) - target).norm(dim=-1).mean(-1)[0]
                if arm == 'parent':
                    scores = torch.empty(32, device='cuda')
                    for mode in range(16):
                        mapped = model.map(prediction, offsets,
                                           torch.tensor([[mode, mode]], device='cuda'),
                                           torch.tensor([[0, 1]], device='cuda'),
                                           (side, side), atlas, weights)
                        scores[mode * 2:mode * 2 + 2] = -mapped['fit_energy'][0]
                else:
                    scores = model.score_candidates(prediction, offsets, weights, atlas,
                                                      use_atlas=arm == 'atlas', side=64,
                                                      chunk=4, image_shape=(side, side))[0]
                choice = int(scores.argmax())
                selected_map = model.map(prediction, offsets,
                                          torch.tensor([[choice // 2]], device='cuda'),
                                          torch.tensor([[choice % 2]], device='cuda'),
                                          (side, side), atlas, weights)
                surface = selected_map['centre_surface_ccf_ap_dv_ml_um'][0, 0].reshape(-1, 3)
                mapped_error = (surface[indices] - target).norm(dim=-1).mean()
                truth_states = prediction['state'].clone()
                truth_states[:, 0] = truth
                truth_map = model.map({**prediction, 'state': truth_states}, offsets,
                                       torch.zeros((1, 1), device='cuda', dtype=torch.long),
                                       torch.tensor([[true_reflection]], device='cuda'),
                                       (side, side), atlas, weights)
                true_map_error = (truth_map['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                                  .reshape(-1, 3)[indices] - target).norm(dim=-1).mean()
                true_normal = full_frame_state_to_components(truth)[1][0, :, 2]
                normals = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
                normal_error = torch.rad2deg(torch.acos((normals * true_normal).sum(-1)
                                                       .abs().clamp(0, 1)))
                row.update(selected_choice=choice, prior_choice=int(prior.flatten().argmax()),
                           selected_tissue_um=float(geometry.flatten()[choice]),
                           prior_tissue_um=float(geometry.flatten()[prior.flatten().argmax()]),
                           oracle_tissue_um=float(geometry.min()),
                           mapped_tissue_um=float(mapped_error),
                           true_pose_mapped_um=float(true_map_error),
                           selected_normal_deg=float(normal_error[choice // 2]),
                           branch_tissue_um=geometry.cpu().tolist(),
                           branch_score=scores.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
            weights = torch.ones((1, 9), device='cuda')
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            prior = prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            flags = torch.tensor([0, 1], device='cuda')[None, None].expand(1, 16, 2)
            geometry = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1)[0]
            if arm == 'parent':
                scores = torch.empty(32, device='cuda')
                for mode in range(16):
                    mapped = model.map(prediction, offsets,
                                       torch.tensor([[mode, mode]], device='cuda'),
                                       torch.tensor([[0, 1]], device='cuda'),
                                       (side, side), atlas, weights)
                    scores[mode * 2:mode * 2 + 2] = -mapped['fit_energy'][0]
            else:
                scores = model.score_candidates(prediction, offsets, weights, atlas,
                                                  use_atlas=arm == 'atlas', side=64,
                                                  chunk=4, image_shape=(side, side))[0]
            choice = int(scores.argmax())
            row = {'arm': arm, 'set': 'real_weak_allen',
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'selected_choice': choice, 'prior_choice': int(prior.flatten().argmax()),
                   'selected_five_um': float(geometry.flatten()[choice]),
                   'prior_five_um': float(geometry.flatten()[prior.flatten().argmax()]),
                   'oracle_five_um': float(geometry.min()),
                   'branch_five_um': geometry.cpu().tolist(),
                   'branch_score': scores.cpu().tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'arm': arm, 'rows': len(rows)}), flush=True)
        del model
summary = []
for arm in ('parent', 'direct', 'atlas'):
    for split, names in (
        ('synthetic', ('selected_tissue_um', 'prior_tissue_um', 'oracle_tissue_um',
                       'mapped_tissue_um', 'true_pose_mapped_um', 'selected_normal_deg')),
        ('real_weak_allen', ('selected_five_um', 'prior_five_um', 'oracle_five_um'))):
        group = [row for row in rows if row['arm'] == arm and row['set'] == split]
        scored = [row for row in group if split != 'synthetic' or row['eligible']]
        identities = sorted({row['animal_id'] for row in scored})
        summary.append({'arm': arm, 'set': split, 'rows': len(group), 'scored': len(scored),
                        'identities': len(identities),
                        'identity_equal_mean': {name: float(np.mean([np.mean([row[name] for row in scored
                            if row['animal_id'] == identity]) for identity in identities]))
                            for name in names},
                        'selection_regret_um': float(np.mean([
                            np.mean([row[names[0]] - row[names[2]] for row in scored
                                     if row['animal_id'] == identity]) for identity in identities])),
                        'median_selected_um': float(np.median([row[names[0]] for row in scored])),
                        'p90_selected_um': float(np.percentile([row[names[0]] for row in scored], 90)),
                        'over_5mm': sum(row[names[0]] > 5000 for row in scored)})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'panel_completed_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
