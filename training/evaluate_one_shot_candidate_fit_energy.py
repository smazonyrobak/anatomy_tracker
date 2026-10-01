"""Frozen synthetic and weak-real readout of the atlas-match energy warmup."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

RUN = ROOT / 'runs/one_shot_candidate_fit_energy_001'
PANEL = ROOT / 'data/one_shot_native256_synthetic_dev_001'
GEOMETRY = ROOT / 'runs/one_shot_joint_physical_pose_002_development_eval'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'runs/one_shot_candidate_fit_energy_001_development_eval'
STEPS = (0, 1000, 2000, 3000, 4000)
SIDE = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 4000
assert json.loads((GEOMETRY / 'completed.json').read_text())['rows'] == 960
synthetic = [row for row in map(json.loads, (GEOMETRY / 'rows.jsonl').read_text().splitlines())
             if row['step'] == 8000 and row['set'] == 'synthetic']
records = {row['panel_physical_section_id']: row for row in
           map(json.loads, (PANEL / 'records.jsonl').read_text().splitlines())}
assert len(synthetic) == len(records) == 128 and sum(row['eligible'] for row in synthetic) == 90
real = [row for row in map(json.loads, (REAL / 'records.jsonl').read_text().splitlines())
        if row['training_split'] == 'development']
assert len(real) == 64 and len({row['animal_id'] for row in real}) == 6
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / SIDE
OUT.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w') as stream:
    for step in STEPS:
        path = RUN / f'joint_step_{step:05d}.pt'
        checkpoint_hashes[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model = OneShotJointSliceModel(atlas_conditioning=True, fit_quality=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for truth in synthetic:
            record = records[truth['panel_physical_section_id']]
            row = {'step': step, 'set': 'synthetic', 'eligible': truth['eligible'],
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                                   'section_id', 'panel_physical_section_id',
                                                   'appearance_mode', 'valid_pixels', 'sha256')}}
            if truth['eligible']:
                with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                prediction = model.predict(image)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                assert torch.allclose(prior.cpu(), torch.tensor(truth['branch_log_mass']), atol=1e-4)
                energy = torch.empty(8, 2, device='cuda')
                for mode in range(8):
                    mapped = model.map(prediction, offsets,
                                       torch.tensor([[mode, mode]], device='cuda'),
                                       torch.tensor([[0, 1]], device='cuda'),
                                       (SIDE, SIDE), atlas, weights)
                    energy[mode] = mapped['fit_energy'][0]
                errors = np.asarray(truth['branch_tissue_um']).reshape(-1)
                scores = energy.flatten().cpu().numpy()
                prior_choice, energy_choice = int(prior.flatten().argmax()), int(energy.flatten().argmin())
                row.update(prior_choice=prior_choice, energy_choice=energy_choice,
                           prior_tissue_um=float(errors[prior_choice]),
                           energy_tissue_um=float(errors[energy_choice]),
                           oracle_tissue_um=float(errors.min()),
                           energy_error_rank_correlation=float(spearmanr(scores, errors).statistic),
                           branch_energy=energy.cpu().tolist(), branch_tissue_um=truth['branch_tissue_um'])
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (SIDE, SIDE),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
            weights = torch.ones((1, 9), device='cuda')
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            prediction = model.predict(image)
            prior = prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)
            centre, frame, basis = full_frame_state_to_components(prediction['state'][0])
            chart = corners[None, None].expand(8, 2, -1, -1).clone()
            chart[:, 1, :, 0] = 255 / SIDE - chart[:, 1, :, 0]
            location = centre[:, None, None] + torch.einsum('aij,abpj->abpi', frame[:, :, :2] @ basis, chart - .5)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            error = (location - reference).norm(dim=-1).mean(-1).flatten().cpu().numpy()
            energy = torch.empty(8, 2, device='cuda')
            for mode in range(8):
                mapped = model.map(prediction, offsets,
                                   torch.tensor([[mode, mode]], device='cuda'),
                                   torch.tensor([[0, 1]], device='cuda'),
                                   (SIDE, SIDE), atlas, weights)
                energy[mode] = mapped['fit_energy'][0]
            scores = energy.flatten().cpu().numpy()
            prior_choice, energy_choice = int(prior.flatten().argmax()), int(energy.flatten().argmin())
            row = {'step': step, 'set': 'real_weak_allen',
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                   'prior_choice': prior_choice, 'energy_choice': energy_choice,
                   'prior_five_point_um': float(error[prior_choice]),
                   'energy_five_point_um': float(error[energy_choice]),
                   'oracle_five_point_um': float(error.min()),
                   'energy_error_rank_correlation': float(spearmanr(scores, error).statistic),
                   'branch_energy': energy.cpu().tolist(), 'branch_five_point_um': error.reshape(8, 2).tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'evaluated_rows': len(rows)}), flush=True)
        del model
summary = []
for step in STEPS:
    for split, metric in (('synthetic', 'tissue_um'), ('real_weak_allen', 'five_point_um')):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        scored = [row for row in group if split != 'synthetic' or row['eligible']]
        identities = sorted({row['animal_id'] for row in scored})
        summary.append({'step': step, 'set': split, 'rows': len(group), 'scored': len(scored),
            'identities': len(identities), 'identity_equal_mean': {name: float(np.mean([
                np.mean([row[name] for row in scored if row['animal_id'] == identity]) for identity in identities]))
                for name in (f'prior_{metric}', f'energy_{metric}', f'oracle_{metric}',
                             'energy_error_rank_correlation')}})
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'steps': STEPS, 'rows': len(rows),
    'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest(),
    'geometry_rows_sha256': hashlib.sha256((GEOMETRY / 'rows.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
