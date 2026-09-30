"""One post-exit measurement of direct prediction and fitting on real run data."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import json
import time
import hashlib
import numpy as np
import torch

from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    compose_full_frame_state,
)
from training.arbitrary_plane_joint_inference_v7 import load_joint_v7_checkpoint
from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6

RUN = ROOT / 'runs/joint_v7_direct_joint_001'
DATA = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
OUT = ROOT / 'runs/joint_v7_direct_joint_001_learning_diagnosis'
assert json.loads((RUN / 'completed.json').read_text())['steps'] == 6000
OUT.mkdir(exist_ok=False)
torch.set_num_threads(4)
model, config = load_joint_v7_checkpoint(RUN / 'joint_step_06000.pt')
for filename, digest in config['source_sha256'].items():
    assert hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest() == digest
identities, images, states, reflections, selected_sections, gradient_sections = [], [], [], [], [], []
counts = {}
for record in json.loads((DATA / 'completed.json').read_text())['sections']:
    metadata = json.loads((DATA / record['artifacts']['metadata']).read_text())
    with np.load(DATA / record['artifacts']['arrays']) as a:
        fit = metadata['canonical_anatomy_plane_fit']['arrays']
        state = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.from_numpy(a[fit['physical_ouv_ap_dv_ml_um_float64']['__ndarray__']]).float()))
        reflection = int(a[metadata['reflection_xy']['__ndarray__']][0])
        for observation in metadata['observations']:
            if not observation['support_information_eligible']:
                continue
            image = torch.from_numpy(a[observation['image_outline_availability_float32']['__ndarray__']]).float()
            image = torch.cat((image, torch.zeros(2, *image.shape[-2:])))
            identities.append({**observation['lineage'], 'mode': observation['selected_mode']})
            images.append(image)
            states.append(state)
            reflections.append(reflection)
            subject = observation['lineage']['subject_id']
            if observation['lineage']['split'] == 'train' and observation['selected_mode'] == 'raw' and len(gradient_sections) < 8:
                gradient_sections.append((len(images) - 1,
                    a[metadata['target_psf_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']].astype(np.float32),
                    a[observation['visible_finite_support_float32']['__ndarray__']].astype(np.float32),
                    a[metadata['axial_offsets_um_float64']['__ndarray__']].astype(np.float32),
                    a[metadata['axial_weights_float64']['__ndarray__']].astype(np.float32)))
            if observation['lineage']['split'] == 'development' and observation['selected_mode'] == 'raw' and counts.get(subject, 0) < 8:
                counts[subject] = counts.get(subject, 0) + 1
                selected_sections.append((len(images) - 1,
                    a[metadata['target_psf_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']].astype(np.float32),
                    a[observation['visible_finite_support_float32']['__ndarray__']].astype(np.float32),
                    a[metadata['axial_offsets_um_float64']['__ndarray__']].astype(np.float32),
                    a[metadata['axial_weights_float64']['__ndarray__']].astype(np.float32)))
images, states, reflections = torch.stack(images), torch.stack(states), torch.tensor(reflections)
raw = {key: [] for key in ('state', 'std', 'concentration', 'log_mass', 'reflection_logit')}
with torch.inference_mode():
    for image in images.split(16):
        prediction = model.predict(image.cuda())
        for key in raw:
            raw[key].append(prediction[key].cpu())
raw = {key: torch.cat(values) for key, values in raw.items()}
frame = full_frame_state_to_components(raw['state'])[1]
truth_frame = full_frame_state_to_components(states)[1]
normal_error = torch.rad2deg((frame[..., :, 2] * truth_frame[:, None, :, 2]).sum(-1).abs().clamp(0, 1).acos())
rotation_error = torch.rad2deg((((frame * truth_frame[:, None]).sum((-2, -1)) - 1) / 2).clamp(-1, 1).acos())
center_error = (raw['state'][..., :3] - states[:, None, :3]).norm(dim=-1)
lp = model.component_log_prob({k: v.cuda() for k, v in raw.items()}, states.cuda(), reflections.cuda()).cpu()
map_index, density_index = raw['log_mass'].argmax(-1), lp.argmax(-1)
idx = torch.arange(len(images))
metrics = {
    'map_normal_deg': normal_error[idx, map_index], 'best_normal_deg': normal_error.min(-1).values,
    'density_normal_deg': normal_error[idx, density_index], 'map_rotation_deg': rotation_error[idx, map_index],
    'density_rotation_deg': rotation_error[idx, density_index], 'map_center_um': center_error[idx, map_index],
    'density_center_um': center_error[idx, density_index], 'nll': -lp.logsumexp(-1),
    'density_component_mass': raw['log_mass'][idx, density_index].exp(),
    'map_is_density_component': (map_index == density_index).float(),
    'map_reflection_correct': ((raw['reflection_logit'][idx, map_index] > 0) == reflections).float(),
}
summary = {}
for split in ('train', 'development'):
    group_metrics = {}
    for subject in sorted({r['subject_id'] for r in identities if r['split'] == split}):
        selected = torch.tensor([r['subject_id'] == subject for r in identities])
        group_metrics[subject] = {key: float(value[selected].mean()) for key, value in metrics.items()}
    summary[split] = {'groups': group_metrics, 'macro': {key: float(np.mean([r[key] for r in group_metrics.values()])) for key in metrics}}
np.savez_compressed(OUT / 'direct_predictions.npz', **{k: v.numpy() for k, v in raw.items()},
                    truth_state=states.numpy(), truth_reflection=reflections.numpy(), component_log_prob=lp.numpy(),
                    **{k: v.numpy() for k, v in metrics.items()})
(OUT / 'identities.json').write_text(json.dumps(identities, indent=2))
print(json.dumps({'direct': summary}), flush=True)

atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
gradient_ids = torch.tensor([s[0] for s in gradient_sections])
prediction = model.predict(images[gradient_ids].cuda())
truth, reflection = states[gradient_ids].cuda(), reflections[gradient_ids].cuda()
mode = model.component_log_prob(prediction, truth, reflection).detach().argmax(-1)
rr = torch.arange(len(mode), device='cuda')
state, std = prediction['state'][rr, mode], prediction['std'][rr, mode]
euclidean = (torch.cat((state[:, :3], state[:, 9:]), -1) - torch.cat((truth[:, :3], truth[:, 9:]), -1)) / model.euclidean_scale
normal_nll = .5 * (euclidean / std).square() + std.log()
rotation, target_rotation = full_frame_state_to_components(state)[1], full_frame_state_to_components(truth)[1]
kappa = prediction['concentration'][rr, mode]
theta = model.rotation_grid
log_z = ((2 / np.pi) * torch.trapezoid(torch.exp(2*kappa[:, None]*(theta.cos()-1))*(theta/2).sin().square(), theta)).log()
rotation_nll = (kappa * (3 - (rotation * target_rotation).sum((-2, -1))) + log_z).mean()
slab, support, z, w = [torch.from_numpy(np.stack([s[j] for s in gradient_sections])).cuda() for j in (1, 2, 3, 4)]
fitted = model.fit(prediction, atlas, z, w, mode[:, None], reflection[:, None])
distance = (((fitted['coordinates'] - slab)/500).square().sum(-1) + 1e-4).sqrt()
dense_loss = ((distance * support[:, None] * w[:, :, None, None]).sum((1, 2, 3)) / support.sum((-2, -1)).clamp_min(1)).mean()
gradient_report = {}
for name, loss in (('center_nll', normal_nll[:, :3].sum(-1).mean()),
                   ('span_shear_nll', normal_nll[:, 3:].sum(-1).mean()),
                   ('rotation_nll', rotation_nll), ('dense_coordinate_loss', dense_loss)):
    gradients = torch.autograd.grad(loss, (model.pose[-1].weight, model.encoder[0][0].weight), retain_graph=True)
    gradient_report[name] = {'value': float(loss.detach()), 'pose_head_norm': float(gradients[0].norm()),
                            'first_encoder_norm': float(gradients[1].norm())}
gradient_report['selected_kappa'] = kappa.detach().cpu().tolist()
gradient_report['selected_std'] = std.detach().cpu().tolist()
gradient_report['first_rotation_vector_norm'] = state[:, 3:6].norm(dim=-1).detach().cpu().tolist()
first_unit = torch.nn.functional.normalize(state[:, 3:6], dim=-1)
gradient_report['orthogonal_second_vector_norm'] = (state[:, 6:9] - (state[:, 6:9]*first_unit).sum(-1, keepdim=True)*first_unit).norm(dim=-1).detach().cpu().tolist()
(OUT / 'gradient_components.json').write_text(json.dumps(gradient_report, indent=2))
print(json.dumps({'gradient_components': gradient_report}), flush=True)
del prediction, fitted, normal_nll, rotation_nll, dense_loss, loss, gradients
fit_rows, fit_arrays = [], {}
with torch.inference_mode():
    for index, (row, slab, support, offsets, weights) in enumerate(selected_sections):
        prediction = model.predict(images[row:row + 1].cuda())
        truth = states[row:row + 1].cuda()
        z, w = torch.from_numpy(offsets[None]).cuda(), torch.from_numpy(weights[None]).cuda()
        q, target = torch.from_numpy(support).cuda(), torch.from_numpy(slab).cuda()
        delta = truth.new_tensor([[.15, 0., 0., 0., 0., 750., 0., 0., 0.]])
        wrong = compose_full_frame_state(truth, delta)
        report = {**identities[row], 'row_index': row}
        for condition, initial, steps in (('true_unwarped', truth, 0), ('true_fitted', truth, 3), ('wrong_fitted', wrong, 3)):
            result = model.fit(prediction, atlas, z, w, torch.zeros(1, 1, device='cuda', dtype=torch.long),
                               reflections[row:row + 1, None].cuda(), initial_state=initial[:, None], steps=steps)
            distance = (result['coordinates'][0] - target).norm(dim=-1)
            error = (distance * q[None] * w[0, :, None, None]).sum() / q.sum().clamp_min(1)
            report[condition] = {'coordinate_um': float(error), 'mismatch': float(result['mismatch'][0, 0]),
                                  'difficulty': float(result['difficulty'][0, 0])}
            fit_arrays[f'{index:02d}_{condition}_surface'] = result['geometry']['centre_surface_ccf_ap_dv_ml_um'][0].cpu().numpy()
        fit_rows.append(report)
np.savez_compressed(OUT / 'fit_surfaces.npz', **fit_arrays)
(OUT / 'fit_rows.json').write_text(json.dumps(fit_rows, indent=2))
fit_summary = {condition: {key: float(np.mean([r[condition][key] for r in fit_rows]))
                          for key in ('coordinate_um', 'mismatch', 'difficulty')}
               for condition in ('true_unwarped', 'true_fitted', 'wrong_fitted')}
fit_summary['fraction_true_fit_scores_better_than_wrong'] = float(np.mean([
    r['true_fitted']['mismatch'] + .05*r['true_fitted']['difficulty'] < r['wrong_fitted']['mismatch'] + .05*r['wrong_fitted']['difficulty'] for r in fit_rows]))
with (RUN / 'joint_step_06000.pt').open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'completed.json').write_text(json.dumps({'direct': summary, 'fitting': fit_summary, 'gradient_components': gradient_report,
    'fitting_sections': len(fit_rows), 'checkpoint_sha256': checkpoint_sha,
    'scope': 'one-atlas synthetic diagnostic; known-pose fitter check is not global prediction'}, indent=2))
print(json.dumps({'fitting': fit_summary}), flush=True)
