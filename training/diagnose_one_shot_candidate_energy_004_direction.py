"""Does learned atlas-match energy point toward truth from actual predicted planes?"""
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
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_candidate_energy_004'
evaluation = root / 'runs/one_shot_candidate_energy_004_development_eval'
geometry = root / 'runs/one_shot_pose_capture_003_development_eval'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
out = root / 'runs/one_shot_candidate_energy_004_direction'
checkpoint_path = run / 'joint_step_08000.pt'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 10000
assert json.loads((evaluation / 'completed.json').read_text())['rows'] == 1152
energy_rows = [row for row in map(json.loads, (evaluation / 'rows.jsonl').open())
               if row['step'] == 8000 and row['set'] == 'synthetic' and row['eligible']]
geometry_rows = {row['panel_physical_section_id']: row for row in
                 map(json.loads, (geometry / 'rows.jsonl').open())
                 if row['step'] == 40000 and row['set'] == 'synthetic' and row['eligible']}
records = {row['panel_physical_section_id']: row for row in
           map(json.loads, (panel / 'records.jsonl').open())}
assert len(energy_rows) == len(geometry_rows) == 90 and len(records) == 128
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
scale = torch.tensor([.06, .06, .06, 500., 500., 500., .05, .05, .05], device='cuda')


def physical_error(state, reflection, chart, reference):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.clone()
    xy[:, 0] = torch.where(reflection.bool(), 255 / side - xy[:, 0], xy[:, 0])
    points = centre[:, None] + torch.einsum('bij,bpj->bpi', frame[:, :, :2] @ basis, xy[None] - .5)
    return (points - reference[None]).norm(dim=-1).mean()


out.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'energy_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
            'geometry_rows_sha256': hashlib.sha256((geometry / 'rows.jsonl').read_bytes()).hexdigest(),
            'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': 90, 'starts': ('image_prior', 'learned_energy', 'geometry_oracle'),
            'tangent_scales': scale.cpu().tolist(), 'step_lengths': (.5, 1.),
            'criterion': 'gradient cosine to known physical error and paired down/up finite steps',
            'truth_mask': 'synthetic reference only for diagnostic scoring, never match-energy input',
            'public_benchmark_used': False}
(out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
rows = []
with (out / 'rows.jsonl').open('w') as stream:
    for index, result in enumerate(energy_rows):
        record = records[result['panel_physical_section_id']]
        truth = geometry_rows[result['panel_physical_section_id']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda().reshape(-1, 3)[valid.flatten()]
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        indices = valid.flatten().nonzero().flatten()
        chart = torch.stack((indices.remainder(side),
                             indices.div(side, rounding_mode='floor')), -1).float() / side
        with torch.no_grad():
            prediction = model.predict(image)
        choices = {'image_prior': result['prior_choice'],
                   'learned_energy': result['energy_choice'],
                   'geometry_oracle': int(np.asarray(truth['branch_tissue_um']).reshape(-1).argmin())}
        for start, choice in choices.items():
            state = prediction['state'][:, choice // 2].detach()
            reflection = torch.tensor([choice % 2], device='cuda')
            z = torch.zeros((1, 9), device='cuda', requires_grad=True)
            posed = compose_full_frame_state(state, z * scale)
            true_gradient = torch.autograd.grad(
                physical_error(posed, reflection, chart, reference), z)[0].detach()
            z = torch.zeros((1, 9), device='cuda', requires_grad=True)
            posed = compose_full_frame_state(state, z * scale)
            conditioned = {**prediction, 'state': posed[:, None]}
            mapped = model.map(conditioned, offsets, torch.zeros((1, 1), device='cuda', dtype=torch.long),
                               reflection[:, None], (side, side), atlas, weights)
            energy = mapped['fit_energy'][0, 0]
            energy_gradient = torch.autograd.grad(energy, z)[0].detach()
            cosine = float(F.cosine_similarity(energy_gradient, true_gradient).item())
            direction = energy_gradient / energy_gradient.norm().clamp_min(1e-10)
            row = {'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                   'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                   'panel_physical_section_id': record['panel_physical_section_id'],
                   'start': start, 'choice': choice, 'reflection': int(reflection),
                   'initial_physical_um': float(physical_error(state, reflection, chart, reference).detach()),
                   'initial_energy': float(energy.detach()),
                   'energy_gradient_norm': float(energy_gradient.norm()),
                   'physical_gradient_norm': float(true_gradient.norm()),
                   'gradient_cosine': cosine}
            with torch.no_grad():
                for length in (.5, 1.):
                    for sign, name in ((-1., 'down'), (1., 'up')):
                        moved = compose_full_frame_state(state, sign * length * direction * scale)
                        row[f'{name}_{length}_physical_um'] = float(
                            physical_error(moved, reflection, chart, reference))
                        moved_prediction = {**prediction, 'state': moved[:, None]}
                        moved_map = model.map(moved_prediction, offsets,
                                              torch.zeros((1, 1), device='cuda', dtype=torch.long),
                                              reflection[:, None], (side, side), atlas, weights)
                        row[f'{name}_{length}_energy'] = float(moved_map['fit_energy'][0, 0])
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        if (index + 1) % 15 == 0:
            stream.flush()
            print(json.dumps({'cases': index + 1, 'rows': len(rows)}), flush=True)
summary = []
for start in protocol['starts']:
    selected = [row for row in rows if row['start'] == start]
    animals = sorted({row['animal_id'] for row in selected})
    item = {'start': start, 'cases': len(selected), 'identities': len(animals),
            'positive_gradient_cosine': sum(row['gradient_cosine'] > 0 for row in selected),
            'identity_equal_mean_cosine': float(np.mean([np.mean([row['gradient_cosine'] for row in selected
                                        if row['animal_id'] == animal]) for animal in animals]))}
    for length in (.5, 1.):
        item[f'down_{length}_improves'] = sum(row[f'down_{length}_physical_um'] < row['initial_physical_um']
                                               for row in selected)
        item[f'up_{length}_improves'] = sum(row[f'up_{length}_physical_um'] < row['initial_physical_um']
                                             for row in selected)
        item[f'down_{length}_lowers_energy'] = sum(row[f'down_{length}_energy'] < row['initial_energy']
                                                    for row in selected)
    summary.append(item)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'checkpoint_sha256': protocol['checkpoint_sha256'],
    'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'source_sha256': protocol['source_sha256'], 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
