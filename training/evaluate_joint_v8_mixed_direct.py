"""Matched held-out synthetic and weak-real readout before/after mixed pose training."""
import hashlib
import json
import os
import subprocess
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel

PARENT = ROOT / 'runs/joint_v8_million_direct_tail_001'
MIXED = ROOT / 'runs/joint_v8_mixed_direct_001'
SYNTHETIC = ROOT / 'data/joint_v7_synthetic_dev192_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
RESERVATIONS = ROOT / 'data/joint_v7_reserved_train_metadata_001/proposed_donor_reservations.jsonl'
OUTPUT = ROOT / 'runs/joint_v8_mixed_direct_dev_001'
SIDE = 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def five_points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    uv = uv.expand(*state.shape[:-1], 5, 2).clone()
    uv[..., 0] = torch.where(torch.as_tensor(reflection, device=state.device)[..., None].bool(),
                             end - 1 - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv)


mixed_done = json.loads((MIXED / 'completed.json').read_text())
assert mixed_done['stage_updates'] == 20000 and mixed_done['total_updates'] == 181000
assert mixed_done['eligible_optimizer_synthetic_planes'] == 160000
assert mixed_done['real_training_image_exposures'] == 160000
syn_done = json.loads((SYNTHETIC / 'completed.json').read_text())
assert syn_done['physical_sections'] == 64 and syn_done['observations'] == 192
assert sha(SYNTHETIC / 'protocol.json') == syn_done['protocol_sha256']
assert sha(SYNTHETIC / 'records.jsonl') == syn_done['records_sha256']
synthetic_records = [json.loads(line) for line in (SYNTHETIC / 'records.jsonl').read_text().splitlines()]
assert len(synthetic_records) == 64 and len({row['animal_id'] for row in synthetic_records}) == 4
real_done = json.loads((REAL / 'completed.json').read_text())
assert real_done['development_images'] == 64 and real_done['development_donors'] == 6
for name, expected in real_done['output_sha256'].items():
    assert sha(REAL / name) == expected
real_records = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
real_dev = [row for row in real_records if row['training_split'] == 'development']
assert len(real_dev) == 64 and len({row['animal_id'] for row in real_dev}) == 6
reserved_train = {int(row['animal_id']) for row in
                  (json.loads(line) for line in RESERVATIONS.read_text().splitlines())
                  if row['proposed_split'] == 'train'}
assert not {row['animal_id'] for row in real_dev} & reserved_train
assert all(row['label_role'].startswith('weak Allen') for row in real_dev)
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz') as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    is_train = arrays['is_train'].copy()
assert all(not is_train[row['array_row_index']] for row in real_dev)

checkpoints = {'million': PARENT / 'joint_step_161000.pt',
               'mixed': MIXED / 'joint_step_181000.pt'}
assert sha(checkpoints['mixed']) == mixed_done['final_checkpoint_sha256']
repository = Path(__file__).resolve().parents[1]
configs = {name: torch.load(path, map_location='cpu', weights_only=True)['config']
           for name, path in checkpoints.items()}
assert configs['mixed']['parent_checkpoint_sha256'] == sha(checkpoints['million'])
assert not {row['animal_id'] for row in synthetic_records} & {
    row['lineage']['animal_id'] for row in configs['mixed']['synthetic_provenance']['base_subjects']}
for config in configs.values():
    for name, expected in config['source_sha256'].items():
        assert sha(repository / 'training' / name) == expected, name
OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoints': {name: {'path': str(path), 'sha256': sha(path)}
                            for name, path in checkpoints.items()},
            'mixed_completed_sha256': sha(MIXED / 'completed.json'),
            'synthetic_completed_sha256': sha(SYNTHETIC / 'completed.json'),
            'real_completed_sha256': sha(REAL / 'completed.json'),
            'reservations_sha256': sha(RESERVATIONS),
            'synthetic': '64 sections from four held-out synthetic maps, three input modes; exact physical target',
            'real': '64 images from six separate development donors; original weak Allen affine only',
            'selection': 'largest direct mode/reflection prior, unchanged before/after',
            'metrics': 'reflection-aware five full-canvas point error and antipodal plane-normal error',
            'aggregation': 'eligible observations within synthetic subject and background mode, or within real donor; groups equally weighted',
            'scope': 'internal development only; no expert anatomical validation, calibration or public benchmark',
            'evaluator_sha256': sha(Path(__file__)),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
pixels = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.], [95.5, 95.5]])
rows = []
for stage, path in checkpoints.items():
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = JointSliceFeedbackModel(modes=checkpoint['config']['modes']).cuda().eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    del checkpoint
    for index, record in enumerate(synthetic_records):
        source = SYNTHETIC / record['file']
        assert sha(source) == record['sha256']
        with np.load(source) as arrays:
            images = arrays['inputs'].copy()
            truth = torch.from_numpy(arrays['target_state'].copy())
            true_reflection = bool(arrays['reflection'])
            eligible = arrays['eligible'].copy()
        target = five_points(truth, true_reflection)
        true_normal = full_frame_state_to_components(truth)[1][..., 2]
        for appearance_index, appearance in enumerate(record['modes']):
            with torch.inference_mode():
                result = model.predict(torch.from_numpy(images[appearance_index:appearance_index + 1]).cuda())
            state = result['state'][0].cpu()
            reflection = torch.arange(2).expand(model.modes, 2)
            error = (five_points(state[:, None].expand(-1, 2, -1), reflection) - target).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(state)[1][..., 2]
            angle = torch.rad2deg((normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
            logits = result['reflection_logit'][0].cpu()
            prior = result['log_mass'][0].cpu()[:, None] + torch.stack((
                F.logsigmoid(-logits), F.logsigmoid(logits)), -1)
            chosen_mode, chosen_reflection = divmod(int(prior.flatten().argmax()), 2)
            save = OUTPUT / f'synthetic_{stage}_{index:02d}_{appearance}.npz'
            np.savez_compressed(save, state=state.numpy(), prior=prior.numpy(),
                                coordinate_error_um=error.numpy(), normal_angle_deg=angle.numpy(),
                                std=result['std'][0].cpu().numpy(),
                                concentration=result['concentration'][0].cpu().numpy())
            rows.append({'stage': stage, 'domain': 'synthetic', 'mode': appearance,
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                'eligible': bool(eligible[appearance_index]), 'prediction_file': save.name,
                'prediction_sha256': sha(save), 'selected_component': [chosen_mode, chosen_reflection],
                'selected_five_point_um': float(error[chosen_mode, chosen_reflection]),
                'oracle_five_point_um': float(error.min()),
                'selected_normal_deg': float(angle[chosen_mode])})
    for index, record in enumerate(real_dev):
        row_index = record['array_row_index']
        image = np.concatenate((real_images[row_index].astype(np.float32),
                                np.zeros((4, SIDE, SIDE), dtype=np.float32)))
        with torch.inference_mode():
            result = model.predict(torch.from_numpy(image[None]).cuda())
        state = result['state'][0].cpu()
        affine = torch.as_tensor(real_affines[row_index], dtype=state.dtype)
        target = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
        reflection = torch.arange(2).expand(model.modes, 2)
        error = (five_points(state[:, None].expand(-1, 2, -1), reflection) - target).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(state)[1][..., 2]
        true_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
        angle = torch.rad2deg((normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
        logits = result['reflection_logit'][0].cpu()
        prior = result['log_mass'][0].cpu()[:, None] + torch.stack((
            F.logsigmoid(-logits), F.logsigmoid(logits)), -1)
        chosen_mode, chosen_reflection = divmod(int(prior.flatten().argmax()), 2)
        save = OUTPUT / f'real_{stage}_{index:02d}.npz'
        np.savez_compressed(save, state=state.numpy(), prior=prior.numpy(),
                            coordinate_error_um=error.numpy(), normal_angle_deg=angle.numpy(),
                            std=result['std'][0].cpu().numpy(),
                            concentration=result['concentration'][0].cpu().numpy())
        rows.append({'stage': stage, 'domain': 'real_weak_affine', 'mode': 'raw',
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
            'eligible': True, 'prediction_file': save.name, 'prediction_sha256': sha(save),
            'selected_component': [chosen_mode, chosen_reflection],
            'selected_five_point_um': float(error[chosen_mode, chosen_reflection]),
            'oracle_five_point_um': float(error.min()),
            'selected_normal_deg': float(angle[chosen_mode])})
    print(json.dumps({'stage': stage, 'synthetic_observations': 192, 'real_observations': 64}), flush=True)
    del model
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
summary = {}
for stage in checkpoints:
    summary[stage] = {}
    for domain, modes in (('synthetic', synthetic_records[0]['modes']),
                          ('real_weak_affine', ['raw'])):
        for mode in modes:
            sample = [row for row in rows if row['stage'] == stage and row['domain'] == domain
                      and row['mode'] == mode and row['eligible']]
            groups = {}
            for animal in sorted({row['animal_id'] for row in sample}):
                subset = [row for row in sample if row['animal_id'] == animal]
                groups[str(animal)] = {'observations': len(subset), **{
                    key: float(np.mean([row[key] for row in subset])) for key in
                    ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
            summary[stage][f'{domain}/{mode}'] = {'groups': groups, 'group_equal': {
                key: float(np.mean([value[key] for value in groups.values()])) for key in
                ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
(OUTPUT / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'rows_sha256': sha(OUTPUT / 'rows.jsonl'), 'evaluation_sha256': sha(OUTPUT / 'evaluation.json'),
    'summary': summary, 'calibrated': False, 'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({stage: {domain: item['group_equal'] for domain, item in report.items()}
                  for stage, report in summary.items()}, indent=2), flush=True)
