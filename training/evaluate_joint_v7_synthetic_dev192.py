"""Post-exit, resolution-matched synthetic development diagnostic only.

Four frozen synthetic development subjects are equal-weighted. Scores are not
biological animal validation, calibrated probabilities, or a public benchmark.
"""
import hashlib
import json
import os
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
from training.arbitrary_plane_joint_inference_v7 import infer_joint_v7, load_joint_v7_checkpoint

RUN = ROOT / 'runs/joint_v7_streaming_joint_002'
DATA = ROOT / 'data/joint_v7_synthetic_dev192_001'
CHECKPOINT = RUN / 'joint_step_30000.pt'
OUTPUT = ROOT / 'runs/joint_v7_synthetic_dev192_step30000_001'
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
        uv[..., 0] = torch.where(torch.as_tensor(reflection, device=state.device)[..., None].bool(), end - 1 - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv)


# An authoritative training completion is required before any checkpoint access.
training_done = json.loads((RUN / 'completed.json').read_text())
assert training_done['total_updates'] == 30000 and training_done['stage_updates'] == 24000
data_done = json.loads((DATA / 'completed.json').read_text())
assert data_done['physical_sections'] == 64 and data_done['observations'] == 192 and data_done['subjects'] == 4
assert sha(DATA / 'protocol.json') == data_done['protocol_sha256']
assert sha(DATA / 'records.jsonl') == data_done['records_sha256']
records = [json.loads(line) for line in (DATA / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({r['animal_id'] for r in records}) == 4
assert all(sum(r['animal_id'] == animal for r in records) == 16 for animal in {r['animal_id'] for r in records})
model, config = load_joint_v7_checkpoint(CHECKPOINT)
assert config['parent_step'] + config['stage_updates'] == 30000 and config['resolution'] == [192, 192]
assert not {r['animal_id'] for r in records} & {r['lineage']['animal_id'] for r in config['synthetic_provenance']['base_subjects']}
repository = Path(__file__).resolve().parents[1]
for name, expected in config['source_sha256'].items():
    assert sha(repository / 'training' / name) == expected, name

OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha(CHECKPOINT),
    'training_completed_sha256': sha(RUN / 'completed.json'), 'development_data': str(DATA),
    'development_completed_sha256': sha(DATA / 'completed.json'), 'data_protocol_sha256': data_done['protocol_sha256'],
    'metric': 'mean physical error at five full-canvas points, AP/DV/ML micrometres; direct canonical and reflection-aware observed versions',
    'map': 'largest direct mixture prior including learned reflection; native fitted selection uses prior minus mismatch minus .05 difficulty',
    'oracle': 'minimum reference error among candidate modes or fitted branches, diagnostic only; never used for inference or checkpoint selection',
    'fitted_metric': 'mean pixelwise Euclidean error of selected/other fitted observed-coordinate centre surfaces, weighted by known visible support',
    'aggregation': 'eligible observations averaged within each of four disjoint synthetic development subjects, then subjects equally weighted; report modes separately and all rows, including ineligible cases',
    'uncertainty': 'raw std/concentration and likelihood saved, uncalibrated; no confidence-coverage or region-probability claim',
    'scope': 'internal development diagnostic on synthetic subjects from one atlas; not biological animal validation, expert ground truth, or a public benchmark'}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
rows = []
started = time.perf_counter()
for section_index, record in enumerate(records):
    path = DATA / record['file']
    assert sha(path) == record['sha256']
    with np.load(path) as arrays:
        images = arrays['inputs'].copy()
        truth = torch.from_numpy(arrays['target_state'].copy())
        centre_truth = arrays['target_centre_um'].copy()
        visible = arrays['visible_support'].copy()
        offsets = torch.from_numpy(arrays['offsets_um'].copy())[None].cuda()
        weights = torch.from_numpy(arrays['weights'].copy())[None].cuda()
        target_reflection = bool(arrays['reflection'])
        eligible = arrays['eligible'].copy()
    truth_observed = five_points(truth, target_reflection)
    for mode_index, mode in enumerate(record['modes']):
        tick = time.perf_counter()
        image = torch.from_numpy(images[mode_index:mode_index + 1]).cuda()
        with torch.inference_mode():
            direct = model.predict(image)
            nll = -model.component_log_prob(direct, truth[None].cuda(),
                torch.tensor([target_reflection], device='cuda').long()).logsumexp(-1).item()
        state = direct['state'][0].cpu()
        direct_prior = direct['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-direct['reflection_logit'][0]), F.logsigmoid(direct['reflection_logit'][0])), -1)
        prior_mode, prior_reflection = divmod(int(direct_prior.flatten().argmax()), 2)
        canonical_error = (five_points(state) - five_points(truth)).norm(dim=-1).mean(-1)
        observed_error = torch.stack([(five_points(state, r) - truth_observed).norm(dim=-1).mean(-1) for r in (0, 1)], 1)
        centre, frame, _ = full_frame_state_to_components(state)
        true_centre, true_frame, _ = full_frame_state_to_components(truth)
        normal_error = torch.rad2deg((frame[..., :, 2] * true_frame[:, 2]).sum(-1).abs().clamp(0, 1).acos())
        centre_error = (centre - true_centre).norm(dim=-1)
        fitted = infer_joint_v7(model, image, atlas, offsets, weights)
        surface_error = np.linalg.norm(fitted['surface'].numpy() - centre_truth, axis=-1)
        mass = float(visible[mode_index].sum())
        fitted_error = (surface_error * visible[mode_index]).sum((-2, -1)) / mass if mass else np.full((8, 2), np.nan)
        selected_mode, selected_reflection = fitted['selected_component']
        fitted_state = fitted['state'][selected_mode, selected_reflection]
        fitted_centre, fitted_frame, _ = full_frame_state_to_components(fitted_state)
        fitted_normal_error = torch.rad2deg((fitted_frame[:, 2] * true_frame[:, 2]).sum().abs().clamp(0, 1).acos())
        fitted_centre_error = (fitted_centre - true_centre).norm()
        save = OUTPUT / f'prediction_{section_index:02d}_{mode}.npz'
        np.savez_compressed(save, direct_state=state.numpy(), direct_log_mass=direct['log_mass'][0].cpu().numpy(),
            direct_reflection_logit=direct['reflection_logit'][0].cpu().numpy(), direct_std=direct['std'][0].cpu().numpy(),
            direct_concentration=direct['concentration'][0].cpu().numpy(), canonical_five_point_error_um=canonical_error.numpy(),
            observed_five_point_error_um=observed_error.numpy(), normal_error_deg=normal_error.numpy(),
            centre_error_um=centre_error.numpy(), fitted_state=fitted['state'].numpy(),
            fitted_surface_error_um=fitted_error, mismatch=fitted['mismatch'].numpy(),
            difficulty=fitted['difficulty'].numpy(), component_log_weight=fitted['component_log_weight'].numpy(),
            selected_fitted_surface_um=fitted['surface'][selected_mode, selected_reflection].numpy())
        rows.append({'section_id': record['section_id'], 'parent_section_id': record['parent_section_id'],
            'animal_id': record['animal_id'], 'subject_id': record['subject_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'mode': mode, 'eligible': bool(eligible[mode_index]),
            'visible_mass': mass, 'prediction_file': save.name, 'prediction_sha256': sha(save),
            'direct_map_mode': prior_mode, 'direct_map_reflection': prior_reflection,
            'direct_map_canonical_five_point_um': float(canonical_error[prior_mode]),
            'direct_map_observed_five_point_um': float(observed_error[prior_mode, prior_reflection]),
            'direct_best_mode_canonical_five_point_um': float(canonical_error.min()),
            'direct_best_branch_observed_five_point_um': float(observed_error.min()),
            'direct_map_normal_deg': float(normal_error[prior_mode]), 'direct_map_centre_um': float(centre_error[prior_mode]),
            'direct_pose_nll': nll, 'fitted_selected_component': [selected_mode, selected_reflection],
            'fitted_selected_coordinate_um': float(fitted_error[selected_mode, selected_reflection]) if mass else None,
            'fitted_prior_component_coordinate_um': float(fitted_error[prior_mode, prior_reflection]) if mass else None,
            'fitted_oracle_coordinate_um': float(np.nanmin(fitted_error)) if mass else None,
            'fitted_selected_normal_deg': float(fitted_normal_error), 'fitted_selected_centre_um': float(fitted_centre_error),
            'fitted_mismatch': float(fitted['mismatch'][selected_mode, selected_reflection]),
            'fitted_difficulty': float(fitted['difficulty'][selected_mode, selected_reflection]),
            'seconds': time.perf_counter() - tick})
    print(json.dumps({'event': 'dev192_section_evaluated', 'sections': section_index + 1,
        'total_sections': len(records), 'seconds': time.perf_counter() - started}), flush=True)
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
metric_names = ('direct_map_canonical_five_point_um', 'direct_map_observed_five_point_um',
    'direct_best_mode_canonical_five_point_um', 'direct_best_branch_observed_five_point_um',
    'direct_map_normal_deg', 'direct_map_centre_um', 'direct_pose_nll',
    'fitted_selected_coordinate_um', 'fitted_prior_component_coordinate_um', 'fitted_oracle_coordinate_um',
    'fitted_selected_normal_deg', 'fitted_selected_centre_um', 'fitted_mismatch', 'fitted_difficulty')
summary = {}
for mode in record['modes']:
    animals = {}
    for animal in sorted({r['animal_id'] for r in rows}):
        sample = [r for r in rows if r['animal_id'] == animal and r['mode'] == mode and r['eligible']]
        assert sample
        animals[animal] = {'eligible_observations': len(sample),
            **{name: float(np.mean([r[name] for r in sample])) for name in metric_names}}
    summary[mode] = {'synthetic_subjects': animals, 'subject_equal':
        {name: float(np.mean([r[name] for r in animals.values()])) for name in metric_names}}
(OUTPUT / 'completed.json').write_text(json.dumps({'sections': len(records), 'observations': len(rows),
    'rows_sha256': sha(OUTPUT / 'rows.jsonl'), 'evaluation_sha256': sha(OUTPUT / 'evaluation.json'),
    'summary': summary, 'seconds': time.perf_counter() - started, 'calibrated': False,
    'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps(summary, indent=2), flush=True)
