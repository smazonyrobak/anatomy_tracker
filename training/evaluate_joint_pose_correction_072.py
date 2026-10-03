"""Frozen synthetic and weak-real readout for the 072 joint continuation."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/joint_pose_correction_072'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/joint_pose_correction_072_development_eval'
steps, side, beam = (0, 3000, 6000, 10000), 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), (side - 1) / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


synthetic_records = [record for record in map(json.loads, (panel / 'records.jsonl').open())
                     if record['eligible']]
real_records = [record for record in map(json.loads, (real / 'records.jsonl').open())
                if record['training_split'] == 'development']
assert len(synthetic_records) == 185 and len({record['animal_id'] for record in synthetic_records}) == 8
assert len(real_records) == 64 and len({record['animal_id'] for record in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.requires_grad_(False)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def predict(image, offsets, weights):
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
    first = model.map(selected, offsets, torch.arange(beam, device='cuda')[None],
        choice % 2, (64, 64), atlas, weights, return_refinement_feature=True,
        feature_side=64, source_shape=(side, side))
    refined, delta, _ = model.refine(first['refinement_feature'], state)
    score = prior.gather(1, choice) + delta
    return prediction, prior, choice, refined, score


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = run / f'joint_step_{step:05d}.pt'
        frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert frozen['step'] == step and not frozen['calibrated']
        model.load_state_dict(frozen['model'], strict=True)
        del frozen
        for record in synthetic_records:
            path = panel / record['file']
            assert sha(path) == record['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                true_reflection = int(arrays['reflection'])
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction, prior, choice, refined, score = predict(image, offsets, weights)
            indices = valid.flatten().nonzero().flatten()
            chosen = indices[torch.linspace(0, len(indices) - 1, 256,
                                            device='cuda').round().long()]
            chart = torch.stack((chosen.remainder(side),
                                 chosen.div(side, rounding_mode='floor')), -1).float() / side
            reference = target.reshape(-1, 3)[chosen]
            all_state = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            prior_rigid = (points(all_state, flags, chart) - reference).norm(dim=-1).mean(-1)[0]
            corrected = (points(refined, choice % 2, chart) - reference).norm(dim=-1).mean(-1)[0]
            selected = int(score[0].argmax())
            image_prediction = {**prediction, 'state': torch.cat((
                refined[:, selected:selected + 1], truth[:, None]), 1)}
            mapped = model.map(image_prediction, offsets,
                torch.tensor([[0, 1]], device='cuda'),
                torch.tensor([[int(choice[0, selected] % 2), true_reflection]], device='cuda'),
                (128, 128), atlas, weights, feature_side=128, source_shape=(side, side))
            grid = (torch.stack((chosen.remainder(side),
                                 chosen.div(side, rounding_mode='floor')), -1).float() + .5) * (2 / side) - 1
            surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
            fitted = F.grid_sample(surface, grid[None, None].expand(2, -1, -1, -1),
                                   mode='bilinear', padding_mode='border', align_corners=False)
            mapped_error = (fitted[:, :, 0].transpose(1, 2) - reference[None]).norm(dim=-1).mean(-1)
            row = {'set': 'synthetic', 'step': step,
                **{key: record[key] for key in ('animal_id', 'specimen_id',
                    'experiment_id', 'section_id', 'sha256', 'appearance_mode')},
                'support_fraction': float(valid.float().mean()),
                'selected_mapped_um': float(mapped_error[0]),
                'true_mapped_um': float(mapped_error[1]),
                'selected_corrected_rigid_um': float(corrected[selected]),
                'best8_corrected_rigid_um': float(corrected.min()),
                'prior_rigid_um': float(prior_rigid[int(prior[0].argmax())]),
                'selected_branch': int(choice[0, selected]),
                'selected_anchor': bool(selected >= 2)}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            index = record['array_row_index']
            image = np.concatenate((real_images[index].astype(np.float32),
                                    np.zeros((4, 192, 192), np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[index])
            weights = torch.ones_like(offsets)
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction, prior, choice, refined, score = predict(image, offsets, weights)
            all_state = prediction['state'].repeat_interleave(2, 1)
            flags = torch.arange(2, device='cuda').repeat(model.modes)[None]
            prior_error = (points(all_state, flags, corners) - reference).norm(dim=-1).mean(-1)[0]
            corrected = (points(refined, choice % 2, corners) - reference).norm(dim=-1).mean(-1)[0]
            selected = int(score[0].argmax())
            row = {'set': 'real_weak_allen', 'step': step,
                **{key: record[key] for key in ('animal_id', 'specimen_id',
                    'experiment_id', 'section_id')},
                'prior_five_um': float(prior_error[int(prior[0].argmax())]),
                'selected_five_um': float(corrected[selected]),
                'best8_corrected_five_um': float(corrected.min()),
                'selected_branch': int(choice[0, selected]),
                'selected_anchor': bool(selected >= 2)}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'checkpoint': step, 'synthetic_sections': len(synthetic_records),
                          'weak_real_sections': len(real_records)}), flush=True)

summary = []
for step in steps:
    for split in ('synthetic', 'real_weak_allen'):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        identities = sorted({row['animal_id'] for row in group})
        names = [key for key in group[0] if key.endswith('_um') or key == 'selected_anchor']
        summary.append({'step': step, 'set': split, 'sections': len(group),
            'identities': len(identities), 'identity_equal_mean': {name: float(np.mean([
                np.mean([row[name] for row in group if row['animal_id'] == identity])
                for identity in identities])) for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt') for step in steps},
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'real_records_sha256': sha(real / 'records.jsonl'),
    'source_sha256': sha(Path(__file__)),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
