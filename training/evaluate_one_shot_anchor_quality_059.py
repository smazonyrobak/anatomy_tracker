"""Frozen 059 development readout; run only after training exits."""
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

run = root / 'runs/one_shot_anchor_quality_059'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_anchor_quality_059_development_eval'
steps = (0, 10000, 30000, 50000)
side, beam = 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['batches'] == 50000

synthetic_records = [row for row in map(json.loads, (panel / 'records.jsonl').open())
                     if row['eligible']]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 185 and len({row['animal_id'] for row in synthetic_records}) == 8
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
                               fit_quality=True, vector_refinement=True,
                               candidate_ranking=True, fitted_ranking=True).cuda().eval()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def candidates(image, offsets, weights):
    prediction = model.predict(image)
    reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                  F.logsigmoid(prediction['reflection_logit'])), -1)
    prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
    old = prior[:, :2 * model.base_modes].topk(2, -1).indices
    anchor = prior[:, 2 * model.base_modes:].topk(6, -1).indices + 2 * model.base_modes
    choice = torch.cat((old, anchor), -1)
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    selected = {**prediction, 'state': state,
                'log_mass': prediction['log_mass'].gather(1, choice // 2),
                'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
    index = torch.arange(beam, device='cuda')[None]
    first = model.map(selected, offsets, index, choice % 2, (64, 64), atlas, weights,
                      return_refinement_feature=True, feature_side=64,
                      source_shape=(side, side))
    refined_state, delta, _ = model.refine(first['refinement_feature'], state)
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, index, choice % 2, (96, 96), atlas, weights,
                       feature_side=96, source_shape=(side, side))
    score = model.score_fitted_candidates(image, refined, mapped, atlas, weights) + delta
    return prediction, prior, choice, refined_state, mapped, score


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic_records:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, prior, choice, refined, mapped, score = candidates(image, offsets, weights)
            indices = valid.flatten().nonzero().flatten()
            chosen = indices[torch.linspace(0, len(indices) - 1, 256, device='cuda').round().long()]
            chart = torch.stack((chosen.remainder(side),
                                 chosen.div(side, rounding_mode='floor')), -1).float() / side
            target = reference.reshape(-1, 3)[chosen]
            all_state = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            rigid = (points(all_state, flags, chart) - target).norm(dim=-1).mean(-1)[0]
            quota_rigid = rigid[choice[0]]
            refined_rigid = (points(refined, choice % 2, chart) - target).norm(dim=-1).mean(-1)[0]
            grid = (torch.stack((chosen.remainder(side),
                                 chosen.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
            grid = grid[None].expand(beam, -1, -1).reshape(beam, 1, 256, 2)
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
            fitted = F.grid_sample(surface, grid, padding_mode='border',
                                   align_corners=False).squeeze(2).transpose(1, 2)
            mapped96_error = (fitted - target[None]).norm(dim=-1).mean(-1)
            selected = int(score[0].argmax())
            mapped256 = model.map({**prediction, 'state': torch.cat((
                refined[:, selected:selected + 1], truth[:, None]), 1)}, offsets,
                torch.tensor([[0, 1]], device='cuda'),
                torch.tensor([[int(choice[0, selected] % 2), true_reflection]], device='cuda'),
                (side, side), atlas, weights)
            full_error = (mapped256['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                          - reference[valid][None]).norm(dim=-1).mean(-1)
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id', 'sha256')},
                   'selected_mapped_um': float(full_error[0]),
                   'true_mapped_um': float(full_error[1]),
                   'old_oracle_rigid_um': float(rigid[:32].min()),
                   'anchor_oracle_rigid_um': float(rigid[32:].min()),
                   'old_quota_rigid_um': float(quota_rigid[:2].min()),
                   'anchor_quota_rigid_um': float(quota_rigid[2:].min()),
                   'old_quota_refined_rigid_um': float(refined_rigid[:2].min()),
                   'anchor_quota_refined_rigid_um': float(refined_rigid[2:].min()),
                   'old_quota_mapped96_um': float(mapped96_error[:2].min()),
                   'anchor_quota_mapped96_um': float(mapped96_error[2:].min()),
                   'selected_mapped96_um': float(mapped96_error[selected]),
                   'prior_rigid_um': float(rigid[int(prior[0].argmax())]),
                   'selected_refined_rigid_um': float(refined_rigid[selected]),
                   'prior_anchor': bool(int(prior[0].argmax()) >= 32),
                   'fitted_anchor': bool(selected >= 2),
                   'prior_branch': int(prior[0].argmax()),
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
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction, prior, choice, refined, _, score = candidates(image, offsets, weights)
            all_state = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            five = (points(all_state, flags, corners) - reference).norm(dim=-1).mean(-1)[0]
            refined_five = (points(refined, choice % 2, corners) - reference).norm(dim=-1).mean(-1)[0]
            selected = int(score[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                  'experiment_id', 'section_id')},
                   'prior_five_um': float(five[int(prior[0].argmax())]),
                   'selected_five_um': float(refined_five[selected]),
                   'old_oracle_five_um': float(five[:32].min()),
                   'anchor_oracle_five_um': float(five[32:].min()),
                   'old_quota_five_um': float(five[choice[0, :2]].min()),
                   'anchor_quota_five_um': float(five[choice[0, 2:]].min()),
                   'prior_anchor': bool(int(prior[0].argmax()) >= 32),
                   'fitted_anchor': bool(selected >= 2),
                   'prior_branch': int(prior[0].argmax()),
                   'fitted_branch': int(choice[0, selected])}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': 185, 'real': 64}), flush=True)

summary = []
for step in steps:
    for split in ('synthetic', 'real_weak_allen'):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        names = [key for key in group[0] if key.endswith('_um') or key in ('prior_anchor', 'fitted_anchor')]
        identities = sorted({row['animal_id'] for row in group})
        summary.append({'step': step, 'set': split, 'rows': len(group),
                        'identities': len(identities), 'identity_equal_mean': {
                            name: float(np.mean([np.mean([row[name] for row in group
                                if row['animal_id'] == identity]) for identity in identities]))
                            for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows),
    'truth_scoring': '256 deterministic, evenly spaced valid tissue pixels per synthetic section',
    'inference_quota': {'old': 2, 'anchor': 6},
    'checkpoint_sha256': {str(step): hashlib.sha256(
        (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
