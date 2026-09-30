"""Post-exit, equal-subject held-out 192px feedback diagnostic only."""
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

RUN = ROOT / 'runs/joint_v8_feedback_pilot_001'
DATA = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUTPUT = ROOT / 'runs/joint_v8_feedback_pilot_dev192_001'
SIDE = 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def five_points(state, reflection=None):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    uv = uv.expand(*state.shape[:-1], 5, 2).clone()
    if reflection is not None:
        uv[..., 0] = torch.where(torch.as_tensor(reflection, device=state.device)[..., None].bool(),
                                 end - 1 - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv)


training_done = json.loads((RUN / 'completed.json').read_text())
assert training_done['stage_updates'] == 6000 and training_done['total_updates'] == 36000
data_done = json.loads((DATA / 'completed.json').read_text())
assert data_done['physical_sections'] == 64 and data_done['observations'] == 192
assert sha(DATA / 'protocol.json') == data_done['protocol_sha256']
assert sha(DATA / 'records.jsonl') == data_done['records_sha256']
records = [json.loads(s) for s in (DATA / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({r['animal_id'] for r in records}) == 4
model, config = load_joint_v8_checkpoint(RUN / 'joint_step_36000.pt')
assert config['stage_updates'] == 6000 and config['resolution'] == [192, 192]
assert not {r['animal_id'] for r in records} & {r['lineage']['animal_id'] for r in config['synthetic_provenance']['base_subjects']}
repository = Path(__file__).resolve().parents[1]
for name, expected in config['source_sha256'].items():
    assert sha(repository / 'training' / name) == expected, name
OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(RUN / 'joint_step_36000.pt'),
    'checkpoint_sha256': sha(RUN / 'joint_step_36000.pt'),
    'training_completed_sha256': sha(RUN / 'completed.json'),
    'development_completed_sha256': sha(DATA / 'completed.json'),
    'physical_sections': 64, 'observations': 192, 'synthetic_subjects': 4,
    'metrics': 'reflection-aware five full-canvas physical points for direct and fitted pose; visible-support centre-surface error separately',
    'selected': 'trained fit-quality + .2 direct log prior - .05 deformation difficulty, fixed before this evaluation',
    'oracle': 'minimum reference error over 16 fitted branches; diagnostic, never a deployed selection rule',
    'aggregation': 'eligible observations averaged within synthetic subject and background mode, then four subjects equally weighted',
    'scope': 'synthetic development only; no biological animal, expert, calibration, GUI or public benchmark claim',
    'evaluator_sha256': sha(Path(__file__)),
    'inference_sha256': sha(repository / 'training/arbitrary_plane_joint_inference_v8.py'),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0][:1]).cuda()
rows = []
started = time.perf_counter()
for section_index, record in enumerate(records):
    path = DATA / record['file']
    assert sha(path) == record['sha256']
    with np.load(path) as arrays:
        images = arrays['inputs'].copy()
        truth = torch.from_numpy(arrays['target_state'].copy())
        target_surface = arrays['target_centre_um'].copy()
        visible = arrays['visible_support'].copy()
        offsets = torch.from_numpy(arrays['offsets_um'].copy())[None].cuda()
        weights = torch.from_numpy(arrays['weights'].copy())[None].cuda()
        target_reflection = bool(arrays['reflection'])
        eligible = arrays['eligible'].copy()
    truth_observed = five_points(truth, target_reflection)
    for mode_index, mode in enumerate(record['modes']):
        image = torch.from_numpy(images[mode_index:mode_index + 1]).cuda()
        with torch.inference_mode():
            direct = model.predict(image)
            fitted = infer_joint_v8(model, image, atlas, offsets, weights)
        direct_state = direct['state'][0].cpu()
        direct_prior = direct['log_mass'][0, :, None].cpu() + torch.stack((
            F.logsigmoid(-direct['reflection_logit'][0]),
            F.logsigmoid(direct['reflection_logit'][0])), -1).cpu()
        prior_mode, prior_reflection = divmod(int(direct_prior.flatten().argmax()), 2)
        direct_error = torch.stack([(five_points(direct_state, r) - truth_observed).norm(dim=-1).mean(-1)
                                    for r in (0, 1)], 1)
        reflection = torch.arange(2).expand(model.modes, 2)
        fitted_error = (five_points(fitted['state'], reflection) - truth_observed).norm(dim=-1).mean(-1)
        surface_error = np.linalg.norm(fitted['surface'].numpy() - target_surface, axis=-1)
        mass = float(visible[mode_index].sum())
        surface_error = (surface_error * visible[mode_index]).sum((-2, -1)) / mass if mass else np.full((8, 2), np.nan)
        chosen_mode, chosen_reflection = fitted['selected_component']
        direct_normal = full_frame_state_to_components(direct_state)[1][..., :, 2]
        fitted_normal = full_frame_state_to_components(fitted['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][:, 2]
        direct_angle = torch.rad2deg((direct_normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
        fitted_angle = torch.rad2deg((fitted_normal * true_normal).sum(-1).abs().clamp(0, 1).acos())
        save = OUTPUT / f'prediction_{section_index:02d}_{mode}.npz'
        np.savez_compressed(save, direct_state=direct_state.numpy(), direct_prior=direct_prior.numpy(),
            direct_error_um=direct_error.numpy(), fitted_state=fitted['state'].numpy(),
            fitted_error_um=fitted_error.numpy(), fitted_surface_error_um=surface_error,
            pose_sequence=fitted['pose_sequence'].numpy(), pose_updates=fitted['pose_updates'].numpy(),
            mismatch=fitted['mismatch'].numpy(), difficulty=fitted['difficulty'].numpy(),
            fit_quality=fitted['fit_quality'].numpy(), component_log_weight=fitted['component_log_weight'].numpy(),
            selected_fitted_surface_um=fitted['surface'][chosen_mode, chosen_reflection].numpy())
        rows.append({'section_id': record['section_id'], 'animal_id': record['animal_id'],
            'subject_id': record['subject_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'mode': mode,
            'eligible': bool(eligible[mode_index]), 'visible_mass': mass,
            'prediction_file': save.name, 'prediction_sha256': sha(save),
            'prior_component': [prior_mode, prior_reflection],
            'selected_component': [chosen_mode, chosen_reflection],
            'selection_changed': (prior_mode, prior_reflection) != (chosen_mode, chosen_reflection),
            'direct_prior_five_point_um': float(direct_error[prior_mode, prior_reflection]),
            'direct_best_branch_five_point_um': float(direct_error.min()),
            'fitted_prior_five_point_um': float(fitted_error[prior_mode, prior_reflection]),
            'fitted_selected_five_point_um': float(fitted_error[chosen_mode, chosen_reflection]),
            'fitted_oracle_five_point_um': float(fitted_error.min()),
            'direct_prior_normal_deg': float(direct_angle[prior_mode]),
            'fitted_selected_normal_deg': float(fitted_angle[chosen_mode, chosen_reflection]),
            'fitted_selected_surface_um': float(surface_error[chosen_mode, chosen_reflection]) if mass else None,
            'fitted_oracle_surface_um': float(np.nanmin(surface_error)) if mass else None,
            'selected_fit_quality': float(fitted['fit_quality'][chosen_mode, chosen_reflection]),
            'selected_mismatch': float(fitted['mismatch'][chosen_mode, chosen_reflection]),
            'selected_difficulty': float(fitted['difficulty'][chosen_mode, chosen_reflection])})
    print(json.dumps({'event': 'v8_dev192_section', 'sections': section_index + 1,
                      'total_sections': len(records), 'seconds': time.perf_counter() - started}), flush=True)
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
metrics = ('direct_prior_five_point_um', 'direct_best_branch_five_point_um',
    'fitted_prior_five_point_um', 'fitted_selected_five_point_um', 'fitted_oracle_five_point_um',
    'direct_prior_normal_deg', 'fitted_selected_normal_deg',
    'fitted_selected_surface_um', 'fitted_oracle_surface_um',
    'selected_fit_quality', 'selected_mismatch', 'selected_difficulty')
summary = {}
for mode in records[0]['modes']:
    subjects = {}
    for animal in sorted({r['animal_id'] for r in rows}):
        sample = [r for r in rows if r['animal_id'] == animal and r['mode'] == mode and r['eligible']]
        assert sample
        subjects[animal] = {'eligible_observations': len(sample),
            'selection_changed': sum(r['selection_changed'] for r in sample),
            **{name: float(np.mean([r[name] for r in sample])) for name in metrics}}
    summary[mode] = {'synthetic_subjects': subjects,
        'subject_equal': {name: float(np.mean([r[name] for r in subjects.values()])) for name in metrics},
        'eligible_selection_changed': sum(r['selection_changed'] for r in rows if r['mode'] == mode and r['eligible'])}
(OUTPUT / 'completed.json').write_text(json.dumps({'sections': len(records), 'observations': len(rows),
    'rows_sha256': sha(OUTPUT / 'rows.jsonl'), 'evaluation_sha256': sha(OUTPUT / 'evaluation.json'),
    'summary': summary, 'seconds': time.perf_counter() - started, 'calibrated': False,
    'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({mode: value['subject_equal'] for mode, value in summary.items()}, indent=2), flush=True)
