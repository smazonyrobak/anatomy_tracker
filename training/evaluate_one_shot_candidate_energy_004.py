"""Frozen candidate-energy readout on independent synthetic and weak-real development sections."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_candidate_energy_004'
geometry = root / 'runs/one_shot_pose_capture_003_development_eval'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_candidate_energy_004_development_eval'
steps = (0, 2000, 4000, 6000, 8000, 10000)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 10000
assert json.loads((geometry / 'completed.json').read_text())['rows'] == 1728
geometry_rows = [row for row in map(json.loads, (geometry / 'rows.jsonl').open()) if row['step'] == 40000]
synthetic_truth = [row for row in geometry_rows if row['set'] == 'synthetic']
real_truth = [row for row in geometry_rows if row['set'] == 'real_weak_allen']
panel_records = [row for row in map(json.loads, (panel / 'records.jsonl').open())]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_truth) == len(panel_records) == 128
assert len(real_truth) == len(real_records) == 64
assert all(a['section_id'] == b['section_id'] for a, b in zip(real_truth, real_records))
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
out.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        path = run / f'joint_step_{step:05d}.pt'
        checkpoint_hashes[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for truth, record in zip(synthetic_truth, panel_records):
            assert truth['panel_physical_section_id'] == record['panel_physical_section_id']
            row = {'step': step, 'set': 'synthetic', 'eligible': truth['eligible'],
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                                   'section_id', 'panel_physical_section_id',
                                                   'appearance_mode', 'valid_pixels', 'sha256')}}
            if truth['eligible']:
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                prediction = model.predict(image)
                prior = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                assert torch.allclose(prior.cpu(), torch.tensor(truth['branch_log_mass']), atol=1e-4)
                energy = torch.empty(16, 2, device='cuda')
                for mode in range(16):
                    mapped = model.map(prediction, offsets,
                                       torch.tensor([[mode, mode]], device='cuda'),
                                       torch.tensor([[0, 1]], device='cuda'),
                                       (side, side), atlas, weights)
                    energy[mode] = mapped['fit_energy'][0]
                scores = energy.flatten().cpu().numpy()
                errors = np.asarray(truth['branch_tissue_um']).reshape(-1)
                prior_choice, energy_choice = int(prior.flatten().argmax()), int(scores.argmin())
                row.update(prior_choice=prior_choice, energy_choice=energy_choice,
                           prior_tissue_um=float(errors[prior_choice]),
                           energy_tissue_um=float(errors[energy_choice]),
                           oracle_tissue_um=float(errors.min()),
                           energy_error_rank_correlation=float(spearmanr(scores, errors).statistic),
                           branch_energy=energy.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for truth, record in zip(real_truth, real_records):
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[i])
            weights = torch.ones((1, 9), device='cuda')
            weights[:, [0, -1]] = .5
            weights /= weights.sum(-1, keepdim=True)
            prediction = model.predict(image)
            prior = prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)
            assert torch.allclose(prior.cpu(), torch.tensor(truth['branch_log_mass']), atol=1e-4)
            energy = torch.empty(16, 2, device='cuda')
            for mode in range(16):
                mapped = model.map(prediction, offsets,
                                   torch.tensor([[mode, mode]], device='cuda'),
                                   torch.tensor([[0, 1]], device='cuda'),
                                   (side, side), atlas, weights)
                energy[mode] = mapped['fit_energy'][0]
            scores = energy.flatten().cpu().numpy()
            errors = np.asarray(truth['branch_five_point_um']).reshape(-1)
            prior_choice, energy_choice = int(prior.flatten().argmax()), int(scores.argmin())
            row = {'step': step, 'set': 'real_weak_allen',
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                   'prior_choice': prior_choice, 'energy_choice': energy_choice,
                   'prior_five_point_um': float(errors[prior_choice]),
                   'energy_five_point_um': float(errors[energy_choice]),
                   'oracle_five_point_um': float(errors.min()),
                   'energy_error_rank_correlation': float(spearmanr(scores, errors).statistic),
                   'branch_energy': energy.cpu().tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'rows': len(rows)}), flush=True)
        del model
summary = []
for step in steps:
    for split, metric in (('synthetic', 'tissue_um'), ('real_weak_allen', 'five_point_um')):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        scored = [row for row in group if split != 'synthetic' or row['eligible']]
        identities = sorted({row['animal_id'] for row in scored})
        summary.append({'step': step, 'set': split, 'rows': len(group), 'scored': len(scored),
            'identities': len(identities), 'identity_equal_mean': {name: float(np.mean([
                np.mean([row[name] for row in scored if row['animal_id'] == identity]) for identity in identities]))
                for name in (f'prior_{metric}', f'energy_{metric}', f'oracle_{metric}',
                             'energy_error_rank_correlation')}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'steps': steps, 'rows': len(rows),
    'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'geometry_rows_sha256': hashlib.sha256((geometry / 'rows.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
