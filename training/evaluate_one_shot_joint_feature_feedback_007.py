"""Frozen readout of best-branch pose correction with learned atlas features."""
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
from training.arbitrary_plane_recurrent_model import local_correlation

run = root / 'runs/one_shot_joint_feature_feedback_007'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_joint_feature_feedback_007_development_eval'
steps = (0, 2000, 4000, 6000, 8000, 10000)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates_per_arm'] == 10000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(records) == 128 and sum(row['eligible'] for row in records) == 90
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


def prediction_and_update(model, image, offsets, weights, arm):
    prediction = model.predict(image)
    prior = prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)
    chosen = prior.flatten(1).topk(4, -1).indices
    modes, reflection = chosen // 2, chosen % 2
    state = prediction['state'][torch.arange(len(image), device='cuda')[:, None], modes]
    mapped = model.map(prediction, offsets, modes, reflection,
                       (side, side), atlas, weights, return_refinement_feature=True)
    feature = F.avg_pool2d(mapped['refinement_feature'].flatten(0, 1), 2).reshape(
        len(image), 4, 229, side // 4, side // 4)
    if arm == 'direct':
        feature = feature.clone()
        feature[:, :, 64:217] = 0
        feature[:, :, 219:] = 0
    else:
        source = feature[:, :, :64].flatten(0, 1)
        target_feature = F.avg_pool2d(model.atlas_encoder(mapped['atlas_pair'].flatten(0, 1)), 2)
        feature = torch.cat((source, target_feature, (source - target_feature).abs(),
                             local_correlation(source, target_feature, 2),
                             feature[:, :, 217:219].flatten(0, 1),
                             feature[:, :, 219:].flatten(0, 1)), 1).reshape_as(feature)
    updated, score, residual = model.refine(feature, state)
    return state, updated, score, reflection, residual


out.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm in ('direct', 'atlas'):
        for step in steps:
            path = run / arm / f'joint_step_{step:05d}.pt'
            checkpoint_hashes[f'{arm}:{step}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['step'] == step and checkpoint['arm'] == arm and not checkpoint['calibrated']
            model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                           vector_refinement=True).cuda().eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            for record in records:
                row = {'arm': arm, 'step': step, 'set': 'synthetic', 'eligible': record['eligible'],
                       **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                                       'section_id', 'panel_physical_section_id',
                                                       'appearance_mode', 'valid_pixels', 'sha256')}}
                if record['eligible']:
                    with np.load(panel / record['file'], allow_pickle=False) as arrays:
                        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                        reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        true_reflection = int(arrays['reflection'])
                        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                    original, updated, score, reflection, residual = prediction_and_update(
                        model, image, offsets, weights, arm)
                    indices = valid.flatten().nonzero().flatten()
                    chart = torch.stack((indices.remainder(side),
                                         indices.div(side, rounding_mode='floor')), -1).float() / side
                    target = reference.reshape(-1, 3)[indices]
                    before = (points(original, reflection, chart) - target).norm(dim=-1).mean(-1)[0]
                    after = (points(updated, reflection, chart) - target).norm(dim=-1).mean(-1)[0]
                    selected = int(score[0].argmax())
                    true_normal = full_frame_state_to_components(truth)[1][0, :, 2]
                    normals = full_frame_state_to_components(updated)[1][0, :, :, 2]
                    angle = torch.rad2deg(torch.acos((normals * true_normal).sum(-1).abs().clamp(0, 1)))
                    row.update(selected=selected, true_reflection=true_reflection,
                               original_top_tissue_um=float(before[0]),
                               original_top_four_oracle_um=float(before.min()),
                               updated_selected_tissue_um=float(after[selected]),
                               updated_oracle_tissue_um=float(after.min()),
                               updated_selected_normal_deg=float(angle[selected]),
                               updated_selected_reflection=int(reflection[0, selected]),
                               branch_before_um=before.cpu().tolist(),
                               branch_after_um=after.cpu().tolist(),
                               branch_score=score[0].cpu().tolist(),
                               branch_update=residual[0].cpu().tolist())
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
                original, updated, score, reflection, residual = prediction_and_update(
                    model, image, offsets, weights, arm)
                before = (points(original, reflection, corners) - reference).norm(dim=-1).mean(-1)[0]
                after = (points(updated, reflection, corners) - reference).norm(dim=-1).mean(-1)[0]
                selected = int(score[0].argmax())
                row = {'arm': arm, 'step': step, 'set': 'real_weak_allen',
                       **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                       'experiment_id', 'section_id')},
                       'selected': selected, 'original_top_five_point_um': float(before[0]),
                       'original_top_four_oracle_um': float(before.min()),
                       'updated_selected_five_point_um': float(after[selected]),
                       'updated_oracle_five_point_um': float(after.min()),
                       'branch_before_um': before.cpu().tolist(),
                       'branch_after_um': after.cpu().tolist(),
                       'branch_score': score[0].cpu().tolist()}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(json.dumps({'arm': arm, 'step': step, 'rows': len(rows)}), flush=True)
            del model
summary = []
for arm in ('direct', 'atlas'):
    for step in steps:
        for split, metric in (('synthetic', 'tissue_um'), ('real_weak_allen', 'five_point_um')):
            group = [row for row in rows if row['arm'] == arm and row['step'] == step and row['set'] == split]
            scored = [row for row in group if split != 'synthetic' or row['eligible']]
            identities = sorted({row['animal_id'] for row in scored})
            names = [f'updated_selected_{metric}', f'updated_oracle_{metric}']
            if split == 'synthetic':
                names += ['original_top_tissue_um', 'original_top_four_oracle_um',
                          'updated_selected_normal_deg']
            else:
                names += ['original_top_five_point_um', 'original_top_four_oracle_um']
            summary.append({'arm': arm, 'step': step, 'set': split,
                'rows': len(group), 'scored': len(scored), 'identities': len(identities),
                'identity_equal_mean': {name: float(np.mean([np.mean([row[name] for row in scored
                    if row['animal_id'] == identity]) for identity in identities])) for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows), 'steps': steps,
    'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
