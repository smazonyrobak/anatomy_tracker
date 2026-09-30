"""Same-model initialization and connected fitting-feedback training on I:.

Initial 96px stage, not a final-resolution experiment or a public benchmark.
No checkpoint/feature/prediction from any previous model is loaded.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    compose_full_frame_state, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v7 import JointSliceModel, anatomical_cost
from training.arbitrary_plane_joint_uncertainty import local_rotation_log

RUN = ROOT / 'runs/joint_v7_direct_joint_001'
DATA = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
SEED, STEPS, WARMUP, BATCH = 2026093001, 6000, 2000, 8
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
RUN.mkdir(parents=True, exist_ok=False)
repository = Path(__file__).resolve().parents[1]
config = {'seed': SEED, 'steps': STEPS, 'warmup_steps': WARMUP, 'batch': BATCH,
          'initialization': 'all parameters random; no checkpoints loaded',
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
          'resolution': [96, 96], 'modes': 8, 'fitter_steps': 3,
          'scope': 'same deployment-target architecture, initial synthetic stage; no real generalization or calibration claim',
          'context': 'all absent in this initial stage; constraint-conditioned training still required',
          'data_completion_sha256': hashlib.sha256((DATA / 'completed.json').read_bytes()).hexdigest(),
          'source_sha256': {}, 'data_sha256': {}, 'optimizer': 'AdamW lr .0002 wd .0001 clip 5, FP32'}
for name in ('train_joint_v7', 'arbitrary_plane_joint_model_v7', 'arbitrary_plane_full_frame_primitives',
             'arbitrary_plane_geometry', 'arbitrary_plane_ribbon_v6', 'arbitrary_plane_recurrent_model',
             'arbitrary_plane_joint_uncertainty', 'arbitrary_plane_allen_atlas_binding_v6'):
    path = repository / 'training' / f'{name}.py'
    shutil.copyfile(path, RUN / path.name)
    config['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

images, identities, eligibility, section_indices = [], [], [], []
physical = {k: [] for k in ('state', 'centre', 'slab', 'offsets', 'weights', 'reflection', 'field')}
visible = []
completed = json.loads((DATA / 'completed.json').read_text())
print('Loading frozen synthetic sections and exact geometry; all weights will be initialized randomly', flush=True)
for si, record in enumerate(completed['sections']):
    metadata_path, arrays_path = (DATA / record['artifacts'][k] for k in ('metadata', 'arrays'))
    for path in (metadata_path, arrays_path):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == record['artifact_sha256'][path.relative_to(DATA).as_posix()]
        config['data_sha256'][str(path)] = digest
    metadata = json.loads(metadata_path.read_text())
    with np.load(arrays_path) as arrays:
        fit = metadata['canonical_anatomy_plane_fit']['arrays']
        ouv = torch.from_numpy(arrays[fit['physical_ouv_ap_dv_ml_um_float64']['__ndarray__']]).float()
        state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
        centre = torch.from_numpy(arrays[metadata['target_centre_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']]).float()
        slab = torch.from_numpy(arrays[metadata['target_psf_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']]).float()
        offsets = torch.from_numpy(arrays[metadata['axial_offsets_um_float64']['__ndarray__']]).float()
        weights = torch.from_numpy(arrays[metadata['axial_weights_float64']['__ndarray__']]).float()
        reflection = torch.tensor(int(arrays[metadata['reflection_xy']['__ndarray__']][0]))
        plane = torch.from_numpy(arrays[fit['fitted_coordinate_raster_ap_dv_ml_um_float64']['__ndarray__']]).float()
        canonical_centre, canonical_slab = (centre.flip(-2), slab.flip(-2)) if reflection else (centre, slab)
        director = torch.einsum('s,shwc->hwc', weights * offsets, canonical_slab - canonical_centre[None]) / (weights * offsets.square()).sum()
        frame = full_frame_state_to_components(state)[1]
        residual_local = torch.einsum('ji,hwj->ihw', frame, canonical_centre - plane)
        director_local = torch.einsum('ji,hwj->ihw', frame, director - frame[:, 2])
        field = torch.cat((residual_local / 200, director_local / .1))
        field = F.interpolate(field[None], (8, 8), mode='bilinear', align_corners=False)[0]
        for key, value in zip(physical, (state, centre, slab, offsets, weights, reflection, field)):
            physical[key].append(value)
        for observation in metadata['observations']:
            images.append(torch.from_numpy(arrays[observation['image_outline_availability_float32']['__ndarray__']]).float())
            visible.append(torch.from_numpy(arrays[observation['visible_finite_support_float32']['__ndarray__']]).float())
            eligibility.append(observation['support_information_eligible'])
            identities.append({**observation['lineage'], 'mode': observation['selected_mode'], 'physical_index': si})
            section_indices.append(si)
physical = {k: torch.stack(v) for k, v in physical.items()}
images = torch.stack(images)
images = torch.cat((images, torch.zeros(len(images), 2, *images.shape[-2:])), 1)
visible, section_indices = torch.stack(visible), torch.tensor(section_indices)
eligibility = torch.tensor(eligibility)
train = torch.tensor([r['split'] == 'train' for r in identities])
train_subjects = sorted({r['subject_id'] for r in identities if r['split'] == 'train'})
dev_subjects = {r['subject_id'] for r in identities if r['split'] == 'development'}
assert not set(train_subjects) & dev_subjects
choices = [torch.tensor([i for i, r in enumerate(identities) if r['subject_id'] == s and eligibility[i]]) for s in train_subjects]
generator = torch.Generator().manual_seed(SEED + 1)
subject_schedule = torch.randint(len(choices), (STEPS, BATCH), generator=generator)
schedule = torch.empty_like(subject_schedule)
for i, choice in enumerate(choices):
    selected = subject_schedule == i
    schedule[selected] = choice[torch.randint(len(choice), (int(selected.sum()),), generator=generator)]
perturbations = torch.randn(WARMUP, BATCH, 9, generator=generator) * torch.tensor([.035]*3 + [150.]*3 + [.02]*3)
torch.save({'observation_index': schedule, 'warmup_perturbations': perturbations}, RUN / 'schedule.pt')
(RUN / 'identities.json').write_text(json.dumps(identities, indent=2))
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2))
atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
model = JointSliceModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()), 'training_subjects': len(train_subjects),
                  'development_subjects': len(dev_subjects), 'training_observations': int(train.sum()),
                  'eligible_training_observations': int((train & eligibility).sum())}), flush=True)


def save_checkpoint(step):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': step, 'config': config, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'calibrated': False}, RUN / f'joint_step_{step:05d}.pt')


def evaluate(step):
    model.eval()
    rows = (~train).nonzero()[:, 0]
    states, masses, deviations, concentrations, reflections = [], [], [], [], []
    with torch.no_grad():
        for batch_rows in rows.split(16):
            prediction = model.predict(images[batch_rows].cuda())
            for output, key in zip((states, masses, deviations, concentrations, reflections),
                                   ('state', 'log_mass', 'std', 'concentration', 'reflection_logit')):
                output.append(prediction[key].cpu())
    state, mass = torch.cat(states), torch.cat(masses)
    selected = mass.argmax(-1)
    pred = state[torch.arange(len(rows)), selected]
    truth = physical['state'][section_indices[rows]]
    normal = full_frame_state_to_components(pred)[1][:, :, 2]
    truth_normal = full_frame_state_to_components(truth)[1][:, :, 2]
    angle = torch.rad2deg((normal * truth_normal).sum(-1).abs().clamp(0, 1).acos())
    distance = (pred[:, :3] - truth[:, :3]).norm(dim=-1)
    np.savez(RUN / f'development_pose_{step:05d}.npz', row_index=rows.numpy(), state=state.numpy(),
             log_mass=mass.numpy(), std=torch.cat(deviations).numpy(), concentration=torch.cat(concentrations).numpy(),
             reflection_logit=torch.cat(reflections).numpy(), normal_error_deg=angle.numpy(), center_error_um=distance.numpy(),
             eligible=eligibility[rows].numpy())
    grouped = {}
    for subject in sorted(dev_subjects):
        select = torch.tensor([identities[i]['subject_id'] == subject for i in rows]) & eligibility[rows]
        grouped[subject] = {'normal_deg': float(angle[select].mean()), 'center_um': float(distance[select].mean())}
    print(json.dumps({'stage': 'development_direct_pose', 'step': step, 'subject_metrics': grouped}), flush=True)
    model.train()


save_checkpoint(0)
evaluate(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace:
    for step, rows in enumerate(schedule, 1):
        ids = section_indices[rows]
        data = {k: v[ids].cuda() for k, v in physical.items()}
        q = visible[rows].cuda()
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict(images[rows].cuda())
        component_lp = model.component_log_prob(prediction, data['state'], data['reflection'])
        pose_nll = -component_lp.logsumexp(-1).mean()
        best = component_lp.detach().argmax(-1)[:, None]
        with torch.no_grad():
            true_image = render_finite_thickness_coordinate_grid(atlas, data['slab'], (0., 0., 0.), (25., 25., 25.), data['weights'])
        appearance_loss = anatomical_cost(prediction['appearance'], true_image).mean()
        warmup = step <= WARMUP
        initial = compose_full_frame_state(data['state'], perturbations[step - 1].cuda())[:, None] if warmup else None
        fitted = model.fit(prediction, atlas, data['offsets'], data['weights'], best,
                           data['reflection'][:, None], initial_state=initial)
        mass = q.sum((-2, -1)).clamp_min(1)
        error = (fitted['coordinates'] - data['slab']) / 500
        # Robust dense supervision anchors the anatomical metric and discourages
        # learned appearance/fitting shortcuts. The whole physical slab matters.
        dense_loss = ((error.square().sum(-1) + 1e-4).sqrt() * q[:, None] * data['weights'][:, :, None, None]).sum((1, 2, 3)) / mass
        feedback = fitted['mismatch'].mean() + .05 * fitted['difficulty'].mean()
        loss = pose_nll + appearance_loss + dense_loss.mean() + .1 * feedback
        # Local covariance is learned only inside a meaningful single-mode chart.
        current = fitted['state'].detach()
        c, r, a = full_frame_state_to_components(current)
        ct, rt, at = full_frame_state_to_components(data['state'])
        relative = r.transpose(-2, -1) @ rt
        angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
        local_rows = ((angle < .35) & ((ct - c).norm(dim=-1) < 1500)).nonzero()[:, 0]
        uncertainty_loss = loss.new_zeros(())
        if len(local_rows):
            b = torch.linalg.solve(a[local_rows], at[local_rows])
            delta = torch.cat((local_rotation_log(relative[local_rows]) / .1,
                               torch.einsum('bji,bj->bi', r[local_rows], ct[local_rows] - c[local_rows]) / 500,
                               b.diagonal(dim1=-2, dim2=-1).log() / .1,
                               (b[:, 0, 1] / b[:, 1, 1])[:, None] / .1), -1)
            g = fitted['geometry']
            field = torch.cat((g['residual_local_um'] / 200, g['director_delta_local'] / .1), 1)
            field = F.interpolate(field, (8, 8), mode='bilinear', align_corners=False)
            residual = torch.cat((delta, (data['field'][local_rows] - field[local_rows]).flatten(1)), -1).detach()
            distribution = torch.distributions.LowRankMultivariateNormal(
                torch.zeros_like(residual), fitted['joint_factor'][local_rows], fitted['joint_std'][local_rows].square())
            uncertainty_loss = -distribution.log_prob(residual).mean() / residual.shape[-1]
            loss = loss + .02 * uncertainty_loss
        if step == WARMUP + 1:
            feedback_gradient = torch.autograd.grad(feedback, model.pose[-1].weight, retain_graph=True)[0]
            gradient_norm = float(feedback_gradient.norm())
            assert np.isfinite(gradient_norm) and gradient_norm > 0
            print(json.dumps({'stage': 'fitting_feedback_reaches_coordinate_head', 'step': step,
                              'gradient_norm': gradient_norm}), flush=True)
        assert torch.isfinite(loss), f'nonfinite training loss at step {step}'
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'warmup': warmup, 'loss': float(loss.detach()),
               'pose_nll': float(pose_nll.detach()), 'appearance': float(appearance_loss.detach()),
               'dense_um': float(dense_loss.detach().mean() * 500), 'fit_cost': float(feedback.detach()),
               'uncertainty_nll_per_coordinate': float(uncertainty_loss.detach()),
               'gradient': float(gradient), 'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            trace.flush()
            print(json.dumps(row), flush=True)
        if step % 2000 == 0:
            save_checkpoint(step)
            evaluate(step)
(RUN / 'completed.json').write_text(json.dumps({'steps': STEPS, 'elapsed_seconds': time.perf_counter() - started,
                                               'calibrated': False, 'scope': config['scope']}, indent=2))
print('Training complete; artifacts are now available for inspection', flush=True)
