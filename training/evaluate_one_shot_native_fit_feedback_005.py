"""Frozen matched readout of actual predicted-pose fitting feedback versus control."""
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

run = root / 'runs/one_shot_native_fit_feedback_005'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_native_fit_feedback_005_development_eval'
steps = (0, 4000, 8000)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates_per_arm'] == 8000
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
flags = torch.tensor([0, 1], device='cuda')[None].expand(16, -1)


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def scores(model, prediction, image, offsets, weights):
    energy = torch.empty(16, 2, device='cuda')
    for mode in range(16):
        mapped = model.map(prediction, offsets, torch.tensor([[mode, mode]], device='cuda'),
                           torch.tensor([[0, 1]], device='cuda'),
                           (side, side), atlas, weights)
        energy[mode] = mapped['fit_energy'][0]
    mass = prediction['log_mass'][0, :, None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'][0]),
        F.logsigmoid(prediction['reflection_logit'][0])), -1)
    return energy, mass


out.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm in ('control', 'fit'):
        for step in steps:
            path = run / arm / f'joint_step_{step:05d}.pt'
            checkpoint_hashes[f'{arm}:{step}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['step'] == step and checkpoint['arm'] == arm and not checkpoint['calibrated']
            model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
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
                    prediction = model.predict(image)
                    states = prediction['state'][0, :, None].expand(-1, 2, -1)
                    indices = valid.flatten().nonzero().flatten()
                    chart = torch.stack((indices.remainder(side),
                                         indices.div(side, rounding_mode='floor')), -1).float() / side
                    target = reference.reshape(-1, 3)[indices]
                    tissue = (points(states, flags, chart) - target).norm(dim=-1).mean(-1)
                    energy, mass = scores(model, prediction, image, offsets, weights)
                    prior_choice = int(mass.flatten().argmax())
                    energy_choice = int(energy.flatten().argmin())
                    truth_normal = full_frame_state_to_components(truth)[1][0, :, 2]
                    normals = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                    angles = torch.rad2deg(torch.acos((normals * truth_normal).sum(-1).abs().clamp(0, 1)))
                    row.update(prior_choice=prior_choice, energy_choice=energy_choice,
                               true_reflection=true_reflection,
                               prior_tissue_um=float(tissue.flatten()[prior_choice]),
                               energy_tissue_um=float(tissue.flatten()[energy_choice]),
                               oracle_tissue_um=float(tissue.min()),
                               prior_normal_deg=float(angles[prior_choice // 2]),
                               energy_normal_deg=float(angles[energy_choice // 2]),
                               prior_reflection=int(prior_choice % 2),
                               energy_reflection=int(energy_choice % 2),
                               branch_tissue_um=tissue.cpu().tolist(),
                               branch_log_mass=mass.cpu().tolist(),
                               branch_energy=energy.cpu().tolist())
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
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                error = (points(states, flags, corners) - reference).norm(dim=-1).mean(-1)
                energy, mass = scores(model, prediction, image, offsets, weights)
                prior_choice = int(mass.flatten().argmax())
                energy_choice = int(energy.flatten().argmin())
                row = {'arm': arm, 'step': step, 'set': 'real_weak_allen',
                       **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                       'experiment_id', 'section_id')},
                       'prior_choice': prior_choice, 'energy_choice': energy_choice,
                       'prior_five_point_um': float(error.flatten()[prior_choice]),
                       'energy_five_point_um': float(error.flatten()[energy_choice]),
                       'oracle_five_point_um': float(error.min()),
                       'branch_five_point_um': error.cpu().tolist(),
                       'branch_log_mass': mass.cpu().tolist(),
                       'branch_energy': energy.cpu().tolist()}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(json.dumps({'arm': arm, 'step': step, 'rows': len(rows)}), flush=True)
            del model
summary = []
for arm in ('control', 'fit'):
    for step in steps:
        for split, metric in (('synthetic', 'tissue_um'), ('real_weak_allen', 'five_point_um')):
            group = [row for row in rows if row['arm'] == arm and row['step'] == step and row['set'] == split]
            scored = [row for row in group if split != 'synthetic' or row['eligible']]
            identities = sorted({row['animal_id'] for row in scored})
            names = [f'prior_{metric}', f'energy_{metric}', f'oracle_{metric}']
            if split == 'synthetic':
                names += ['prior_normal_deg', 'energy_normal_deg']
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
