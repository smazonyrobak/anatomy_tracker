"""Matched held-out synthetic readout before and after all-branch feedback training."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_inference_v8 import infer_joint_v8, load_joint_v8_checkpoint

DIRECT = ROOT / 'runs/joint_v8_mixed_direct_001'
JOINT = ROOT / 'runs/joint_v8_allbranch_feedback_001'
DATA = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUTPUT = ROOT / 'runs/joint_v8_allbranch_feedback_dev192_001'
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


direct_done = json.loads((DIRECT / 'completed.json').read_text())
joint_done = json.loads((JOINT / 'completed.json').read_text())
assert direct_done['stage_updates'] == 20000 and direct_done['total_updates'] == 181000
assert direct_done['eligible_optimizer_synthetic_planes'] == 160000
assert direct_done['real_training_image_exposures'] == 160000
assert joint_done['stage_updates'] == 30000 and joint_done['total_updates'] == 211000
data_done = json.loads((DATA / 'completed.json').read_text())
assert data_done['physical_sections'] == 64 and data_done['observations'] == 192
assert sha(DATA / 'protocol.json') == data_done['protocol_sha256']
assert sha(DATA / 'records.jsonl') == data_done['records_sha256']
records = [json.loads(s) for s in (DATA / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({r['animal_id'] for r in records}) == 4
checkpoints = {'before_joint': DIRECT / 'joint_step_181000.pt',
               'after_joint': JOINT / 'joint_step_211000.pt'}
repository = Path(__file__).resolve().parents[1]
direct_config = torch.load(checkpoints['before_joint'], map_location='cpu', weights_only=True)['config']
joint_config = torch.load(checkpoints['after_joint'], map_location='cpu', weights_only=True)['config']
assert joint_config['parent_checkpoint_sha256'] == sha(checkpoints['before_joint'])
assert not {r['animal_id'] for r in records} & {
    r['lineage']['animal_id'] for r in joint_config['synthetic_provenance']['base_subjects']}
for config in (direct_config, joint_config):
    for name, expected in config['source_sha256'].items():
        assert sha(repository / 'training' / name) == expected, name
OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoints': {label: {'path': str(path), 'sha256': sha(path)}
                            for label, path in checkpoints.items()},
    'direct_completed_sha256': sha(DIRECT / 'completed.json'),
    'joint_completed_sha256': sha(JOINT / 'completed.json'),
    'development_completed_sha256': sha(DATA / 'completed.json'),
    'physical_sections': 64, 'observations_per_checkpoint': 192, 'synthetic_subjects': 4,
    'metrics': 'reflection-aware five-point physical pose error, antipodal normal angle, visible-tissue centre-surface error',
    'selection': 'fit quality + .2 direct log prior - .05 deformation difficulty; unchanged before and after training',
    'comparison': 'same sections, images, atlas, PSF, model and inference code; whole-model lineage before versus after joint feedback',
    'aggregation': 'eligible observations within synthetic subject and input mode, then four subjects equally weighted',
    'scope': 'synthetic development only; not biological-animal validation, calibration, public benchmark, or deployment',
    'evaluator_sha256': sha(Path(__file__)),
    'inference_sha256': sha(repository / 'training/arbitrary_plane_joint_inference_v8.py'),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0][:1]).cuda()
rows = []
started = time.perf_counter()
for label, path in checkpoints.items():
    model, config = load_joint_v8_checkpoint(path)
    assert config['resolution'] == [192, 192] and model.modes == 8
    for section_index, record in enumerate(records):
        source = DATA / record['file']
        assert sha(source) == record['sha256']
        with np.load(source) as arrays:
            images = arrays['inputs'].copy()
            truth = torch.from_numpy(arrays['target_state'].copy())
            target_surface = arrays['target_centre_um'].copy()
            visible = arrays['visible_support'].copy()
            offsets = torch.from_numpy(arrays['offsets_um'].copy())[None].cuda()
            weights = torch.from_numpy(arrays['weights'].copy())[None].cuda()
            target_reflection = bool(arrays['reflection'])
            eligible = arrays['eligible'].copy()
        target_points = five_points(truth, target_reflection)
        true_normal = full_frame_state_to_components(truth)[1][:, 2]
        for mode_index, mode in enumerate(record['modes']):
            image = torch.from_numpy(images[mode_index:mode_index + 1]).cuda()
            with torch.inference_mode():
                direct = model.predict(image)
                fitted = infer_joint_v8(model, image, atlas, offsets, weights)
            direct_state = direct['state'][0].cpu()
            prior = direct['log_mass'][0, :, None].cpu() + torch.stack((
                F.logsigmoid(-direct['reflection_logit'][0]),
                F.logsigmoid(direct['reflection_logit'][0])), -1).cpu()
            prior_mode, prior_reflection = divmod(int(prior.flatten().argmax()), 2)
            direct_error = torch.stack([(five_points(direct_state, r) - target_points).norm(dim=-1).mean(-1)
                                        for r in (0, 1)], 1)
            reflection = torch.arange(2).expand(model.modes, 2)
            fitted_error = (five_points(fitted['state'], reflection) - target_points).norm(dim=-1).mean(-1)
            mass = float(visible[mode_index].sum())
            surface_error = np.linalg.norm(fitted['surface'].numpy() - target_surface, axis=-1)
            surface_error = (surface_error * visible[mode_index]).sum((-2, -1)) / mass if mass else np.full((8, 2), np.nan)
            direct_normal = full_frame_state_to_components(direct_state)[1][..., :, 2]
            fitted_normal = full_frame_state_to_components(fitted['state'])[1][..., :, 2]
            direct_angle = torch.rad2deg((direct_normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
            fitted_angle = torch.rad2deg((fitted_normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
            chosen_mode, chosen_reflection = fitted['selected_component']
            save = OUTPUT / f'{label}_{section_index:02d}_{mode}.npz'
            np.savez_compressed(save, direct_state=direct_state.numpy(), direct_prior=prior.numpy(),
                direct_error_um=direct_error.numpy(), fitted_state=fitted['state'].numpy(),
                fitted_error_um=fitted_error.numpy(), fitted_surface_error_um=surface_error,
                pose_sequence=fitted['pose_sequence'].numpy(), pose_updates=fitted['pose_updates'].numpy(),
                mismatch=fitted['mismatch'].numpy(), difficulty=fitted['difficulty'].numpy(),
                fit_quality=fitted['fit_quality'].numpy(), component_log_weight=fitted['component_log_weight'].numpy(),
                selected_fitted_surface_um=fitted['surface'][chosen_mode, chosen_reflection].numpy())
            rows.append({'checkpoint': label, 'section_id': record['section_id'],
                'animal_id': record['animal_id'], 'subject_id': record['subject_id'],
                'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                'mode': mode, 'eligible': bool(eligible[mode_index]),
                'prediction_file': save.name, 'prediction_sha256': sha(save),
                'prior_component': [prior_mode, prior_reflection],
                'selected_component': [chosen_mode, chosen_reflection],
                'direct_prior_five_point_um': float(direct_error[prior_mode, prior_reflection]),
                'direct_oracle_five_point_um': float(direct_error.min()),
                'fitted_prior_five_point_um': float(fitted_error[prior_mode, prior_reflection]),
                'fitted_selected_five_point_um': float(fitted_error[chosen_mode, chosen_reflection]),
                'fitted_oracle_five_point_um': float(fitted_error.min()),
                'direct_prior_normal_deg': float(direct_angle[prior_mode]),
                'fitted_selected_normal_deg': float(fitted_angle[chosen_mode, chosen_reflection]),
                'fitted_selected_surface_um': float(surface_error[chosen_mode, chosen_reflection]) if mass else None,
                'fitted_oracle_surface_um': float(np.nanmin(surface_error)) if mass else None})
        print(json.dumps({'event': 'matched_dev192_section', 'checkpoint': label,
            'sections': section_index + 1, 'total_sections': len(records),
            'seconds': time.perf_counter() - started}), flush=True)
    del model
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
metrics = ('direct_prior_five_point_um', 'direct_oracle_five_point_um',
    'fitted_prior_five_point_um', 'fitted_selected_five_point_um', 'fitted_oracle_five_point_um',
    'direct_prior_normal_deg', 'fitted_selected_normal_deg',
    'fitted_selected_surface_um', 'fitted_oracle_surface_um')
summary = {}
for label in checkpoints:
    summary[label] = {}
    for mode in records[0]['modes']:
        subjects = {}
        for animal in sorted({row['animal_id'] for row in rows}):
            sample = [row for row in rows if row['checkpoint'] == label and row['animal_id'] == animal
                      and row['mode'] == mode and row['eligible']]
            assert sample
            subjects[animal] = {'eligible_observations': len(sample),
                **{name: float(np.mean([row[name] for row in sample])) for name in metrics}}
        summary[label][mode] = {'synthetic_subjects': subjects,
            'subject_equal': {name: float(np.mean([row[name] for row in subjects.values()])) for name in metrics}}
(OUTPUT / 'completed.json').write_text(json.dumps({'sections_per_checkpoint': len(records),
    'observations': len(rows), 'rows_sha256': sha(OUTPUT / 'rows.jsonl'),
    'evaluation_sha256': sha(OUTPUT / 'evaluation.json'), 'summary': summary,
    'seconds': time.perf_counter() - started, 'calibrated': False,
    'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({label: {mode: value['subject_equal'] for mode, value in modes.items()}
                  for label, modes in summary.items()}, indent=2), flush=True)
