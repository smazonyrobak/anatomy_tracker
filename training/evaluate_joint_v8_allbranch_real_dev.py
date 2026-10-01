"""Paired whole-model fit readout on donor-held-out real sections with weak Allen affines."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_inference_v8 import infer_joint_v8, load_joint_v8_checkpoint

REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
RESERVATIONS = ROOT / 'data/joint_v7_reserved_train_metadata_001/proposed_donor_reservations.jsonl'
MIXED = ROOT / 'runs/joint_v8_mixed_direct_001'
JOINT = ROOT / 'runs/joint_v8_allbranch_feedback_001'
OUTPUT = ROOT / 'runs/joint_v8_allbranch_real_dev_001'
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
joint_done = json.loads((JOINT / 'completed.json').read_text())
assert mixed_done['total_updates'] == 181000 and mixed_done['real_training_image_exposures'] == 160000
assert joint_done['total_updates'] == 211000 and joint_done['real_training_image_exposures'] == 120000
real_done = json.loads((REAL / 'completed.json').read_text())
assert real_done['development_images'] == 64 and real_done['development_donors'] == 6
for name, expected in real_done['output_sha256'].items():
    assert sha(REAL / name) == expected
records = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
dev = [row for row in records if row['training_split'] == 'development']
assert len(dev) == 64 and len({row['animal_id'] for row in dev}) == 6
assert all(row['label_role'].startswith('weak Allen') for row in dev)
reserved_train = {int(row['animal_id']) for row in
                  (json.loads(line) for line in RESERVATIONS.read_text().splitlines())
                  if row['proposed_split'] == 'train'}
assert not {row['animal_id'] for row in dev} & reserved_train
images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz') as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thicknesses = arrays['thickness_um'].copy()
    is_train = arrays['is_train'].copy()
assert all(not is_train[row['array_row_index']] for row in dev)
checkpoints = {'before_joint': MIXED / 'joint_step_181000.pt',
               'after_joint': JOINT / 'joint_step_211000.pt'}
assert sha(checkpoints['before_joint']) == mixed_done['final_checkpoint_sha256']
repository = Path(__file__).resolve().parents[1]
configs = {label: torch.load(path, map_location='cpu', weights_only=True)['config']
           for label, path in checkpoints.items()}
assert configs['after_joint']['parent_checkpoint_sha256'] == sha(checkpoints['before_joint'])
for config in configs.values():
    for name, expected in config['source_sha256'].items():
        assert sha(repository / 'training' / name) == expected, name
OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoints': {label: {'path': str(path), 'sha256': sha(path)}
                            for label, path in checkpoints.items()},
            'real_completed_sha256': sha(REAL / 'completed.json'),
            'reservations_sha256': sha(RESERVATIONS),
            'reference': '64 real sections from six separate development donors; upstream weak Allen affine, not blinded expert anatomical truth',
            'psf': 'section-specific recorded thickness, nine-node trapezoidal through-plane integration',
            'selection': 'same all-16 recurrent fitter and uncalibrated quality ranking for both checkpoints',
            'metrics': 'reflection-aware five-point physical error and antipodal plane-normal angle before/after fitting; failures retained',
            'aggregation': 'sections within donor, then six donors equally weighted',
            'scope': 'internal development only; no calibration, public benchmark, or deployment claim',
            'evaluator_sha256': sha(Path(__file__)),
            'inference_sha256': sha(repository / 'training/arbitrary_plane_joint_inference_v8.py'),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0][:1]).cuda()
pixels = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.], [95.5, 95.5]])
rows = []
for stage, path in checkpoints.items():
    model, config = load_joint_v8_checkpoint(path)
    assert config['resolution'] == [SIDE, SIDE] and model.modes == 8
    for index, record in enumerate(dev):
        row_index = record['array_row_index']
        image = np.concatenate((images[row_index].astype(np.float32),
                                np.zeros((4, SIDE, SIDE), dtype=np.float32)))
        input_tensor = torch.from_numpy(image[None]).cuda()
        offsets = (torch.linspace(-.5, .5, 9, device='cuda') * float(thicknesses[row_index]))[None]
        weights = torch.ones_like(offsets)
        weights[:, [0, -1]] = .5
        weights /= weights.sum(-1, keepdim=True)
        with torch.inference_mode():
            direct = model.predict(input_tensor)
            fitted = infer_joint_v8(model, input_tensor, atlas, offsets, weights)
        direct_state = direct['state'][0].cpu()
        affine = torch.as_tensor(affines[row_index], dtype=direct_state.dtype)
        target = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
        reference_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
        direct_reflection = torch.arange(2).expand(model.modes, 2)
        direct_error = (five_points(direct_state[:, None].expand(-1, 2, -1), direct_reflection)
                        - target).norm(dim=-1).mean(-1)
        fitted_error = (five_points(fitted['state'], direct_reflection) - target).norm(dim=-1).mean(-1)
        direct_normal = full_frame_state_to_components(direct_state)[1][..., 2]
        fitted_normal = full_frame_state_to_components(fitted['state'])[1][..., 2]
        direct_angle = torch.rad2deg((direct_normal * reference_normal).sum(-1).abs().clamp(0, 1).acos())
        fitted_angle = torch.rad2deg((fitted_normal * reference_normal).sum(-1).abs().clamp(0, 1).acos())
        prior_mode, prior_reflection = divmod(int(fitted['prior_log_weight'].flatten().argmax()), 2)
        selected_mode, selected_reflection = fitted['selected_component']
        save = OUTPUT / f'{stage}_{index:02d}.npz'
        np.savez_compressed(save, direct_state=direct_state.numpy(),
                            fitted_state=fitted['state'].numpy(), surface=fitted['surface'].numpy(),
                            direct_error_um=direct_error.numpy(), fitted_error_um=fitted_error.numpy(),
                            direct_normal_deg=direct_angle.numpy(), fitted_normal_deg=fitted_angle.numpy(),
                            prior=fitted['prior_log_weight'].numpy(),
                            fit_log_weight=fitted['component_log_weight'].numpy(),
                            mismatch=fitted['mismatch'].numpy(), difficulty=fitted['difficulty'].numpy(),
                            pose_sequence=fitted['pose_sequence'].numpy(),
                            joint_std=fitted['joint_std'].numpy(), joint_factor=fitted['joint_factor'].numpy())
        rows.append({'stage': stage, 'animal_id': record['animal_id'],
                     'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                     'section_id': record['section_id'], 'prediction_file': save.name,
                     'prediction_sha256': sha(save),
                     'prior_component': [prior_mode, prior_reflection],
                     'selected_component': [selected_mode, selected_reflection],
                     'direct_prior_five_point_um': float(direct_error[prior_mode, prior_reflection]),
                     'fitted_prior_five_point_um': float(fitted_error[prior_mode, prior_reflection]),
                     'fitted_selected_five_point_um': float(fitted_error[selected_mode, selected_reflection]),
                     'fitted_oracle_five_point_um': float(fitted_error.min()),
                     'direct_prior_normal_deg': float(direct_angle[prior_mode]),
                     'fitted_selected_normal_deg': float(fitted_angle[selected_mode, selected_reflection])})
    print(json.dumps({'stage': stage, 'development_sections': len(dev)}), flush=True)
    del model
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
summary = {}
metrics = ('direct_prior_five_point_um', 'fitted_prior_five_point_um',
           'fitted_selected_five_point_um', 'fitted_oracle_five_point_um',
           'direct_prior_normal_deg', 'fitted_selected_normal_deg')
for stage in checkpoints:
    donors = {}
    for donor in sorted({row['animal_id'] for row in dev}):
        sample = [row for row in rows if row['stage'] == stage and row['animal_id'] == donor]
        donors[str(donor)] = {'sections': len(sample), **{
            key: float(np.mean([row[key] for row in sample])) for key in metrics}}
    summary[stage] = {'donors': donors, 'donor_equal': {
        key: float(np.mean([item[key] for item in donors.values()])) for key in metrics}}
(OUTPUT / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'rows_sha256': sha(OUTPUT / 'rows.jsonl'), 'evaluation_sha256': sha(OUTPUT / 'evaluation.json'),
    'summary': summary, 'calibrated': False, 'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({stage: value['donor_equal'] for stage, value in summary.items()}, indent=2), flush=True)
