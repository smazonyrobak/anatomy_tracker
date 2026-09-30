"""Continue the same whole model with fitting-to-pose feedback on all 16 branches.

Synthetic TRAIN only. Run after the million-plane direct stage is frozen and read out.
The slab target is in observed raster order; fitted coordinates are reflected to
that order before physical supervision. This corrects the older joint pilot loss.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import (
    load_streaming_synthetic_v7_64, sample_streaming_synthetic_v7_64,
)

RUN = ROOT / 'runs/joint_v8_allbranch_feedback_001'
PARENT = ROOT / 'runs/joint_v8_million_direct_001'
SEED, STEPS, BATCH, SIDE = 2026100101, 30000, 2, 192
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def five_points(state):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    return centre[..., None, :] + torch.einsum('...ij,pj->...pi', frame[..., :2] @ basis, uv)


parent_done = json.loads((PARENT / 'completed.json').read_text())
assert parent_done['stage_updates'] == 125000
assert parent_done['unique_eligible_optimizer_synthetic_planes'] == 1000000
parent_path = PARENT / 'joint_step_161000.pt'
parent_sha = sha(parent_path)
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 161000 and not checkpoint['calibrated']
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
model = JointSliceFeedbackModel().cuda()
model.load_state_dict(checkpoint['model_state'], strict=True)
del checkpoint
model.uncertainty.requires_grad_(False)
pose_parameters = list(model.encoder.parameters()) + list(model.pose.parameters())
fit_parameters = [p for name, p in model.named_parameters()
                  if p.requires_grad and not name.startswith(('encoder.', 'pose.'))]
optimizer = torch.optim.AdamW([{'params': pose_parameters, 'lr': 1e-5},
                               {'params': fit_parameters, 'lr': 3e-5}], weight_decay=1e-4)
generator = torch.Generator().manual_seed(SEED)
base_indices = torch.randint(64, (STEPS,), generator=generator)
virtual_indices = base_indices[:, None] * 64 + torch.randint(64, (STEPS, BATCH), generator=generator)
pair_order = torch.stack([torch.randperm(16, generator=generator) for _ in range(STEPS)])
assert torch.equal(virtual_indices // 64, base_indices[:, None].expand_as(virtual_indices))

RUN.mkdir(parents=True, exist_ok=False)
torch.save({'base_index': base_indices, 'virtual_index': virtual_indices,
            'candidate_pair_order': pair_order}, RUN / 'schedule.pt')
(RUN / 'identities.json').write_text(json.dumps(context['subjects']), encoding='utf8')
source_names = sorted(set(context['provenance']['source_sha256']) | {
    'arbitrary_plane_joint_model_v8.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_joint_inference_v8.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'train_joint_v8_allbranch_feedback.py',
})
source_sha = {}
for name in source_names:
    path = repository / 'training' / name
    shutil.copyfile(path, RUN / name)
    source_sha[name] = sha(path)
config = {'seed': SEED, 'stage_updates': STEPS, 'batch': BATCH, 'resolution': [SIDE, SIDE],
    'modes': model.modes, 'fitter_steps': 4, 'candidate_branches_per_image': 16,
    'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': parent_sha,
    'parent_completed_sha256': sha(PARENT / 'completed.json'),
    'initialization': 'strict whole-model continuation of the standalone random-init v8 lineage',
    'sampling': '64 independent synthetic TRAIN maps; one base and two affine variants per update; fresh eligible arbitrary physical planes; all 16 mode/reflection fits per image in randomized pairs',
    'physical_target': 'observed PSF slab; reflect canonical predicted slab before comparison',
    'fit_to_pose': 'differentiable final and intermediate physical errors from the geometry-best mode with correct reflection; no detached predicted pose; all 16 scores trained on synthetic physical error',
    'quality_learning': 'all-branch physical-error score regression and randomized pair ranking; deformation penalty detached inside score to prevent wrong branches learning implausible deformation as a rejection signal',
    'anatomical_loss': 'atlas-like source appearance against finite-thickness atlas render on visible synthetic tissue only; damaged and missing pixels excluded',
    'negative_modes': 'quality/ranking only, no forced deformation to the true atlas location',
    'optimizer': 'AdamW; pose 1e-5 and fitting 3e-5 cosine to 20%; norm clip 5; FP32',
    'calibrated': False, 'constraints': 'absent, not trained',
    'scope': 'synthetic TRAIN joint feedback; not animal validation, calibration, public benchmark, or deployment',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name()},
    'source_sha256': source_sha, 'data_sha256': context['bindings'],
    'synthetic_provenance': context['provenance'],
    'schedule_sha256': sha(RUN / 'schedule.pt'),
    'identities_sha256': sha(RUN / 'identities.json')}
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf8')
atlas = context['atlas'][:1]


def draw(subjects, step, trace):
    pending = list(range(BATCH))
    accepted = {}
    attempt = attempted = 0
    while pending:
        seed = SEED * 10**18 + (step + attempt) * (step + attempt + 1) // 2 + attempt
        batch = sample_streaming_synthetic_v7_64(context, subjects[pending].tolist(), seed, side=SIDE)
        for row, record in enumerate(batch['provenance']):
            slot = pending[row]
            used = bool(batch['eligible'][row])
            trace.write(json.dumps({**record, 'stage_step': step, 'batch_slot': slot,
                                    'attempt': attempt, 'used': used}) + '\n')
            if used:
                accepted[slot] = {key: value[row:row + 1] for key, value in batch.items()
                                  if key in ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                             'slab', 'visible', 'clean', 'support', 'field')}
        attempted += len(pending)
        pending = [slot for row, slot in enumerate(pending) if not batch['eligible'][row]]
        attempt += 1
    sample = {key: torch.cat([accepted[i][key] for i in range(BATCH)]) for key in accepted[0]}
    sample['psf_pixel_y'], sample['psf_pixel_x'] = batch['psf_pixel_y'], batch['psf_pixel_x']
    return sample, attempted


def save_checkpoint(step, attempted):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': 161000 + step, 'stage_step': step, 'config': config,
                'eligible_optimizer_synthetic_planes': step * BATCH,
                'attempted_synthetic_planes': attempted,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'calibrated': False}, RUN / f'joint_step_{161000 + step:06d}.pt')


save_checkpoint(0, 0)
started = time.perf_counter()
attempted = 0
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace, \
     (RUN / 'synthetic_draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, STEPS + 1):
        sample, tries = draw(virtual_indices[step - 1], step, draws)
        attempted += tries
        image, truth, reflection = sample['inputs'], sample['state'], sample['reflection']
        offsets, weights = sample['offsets'], sample['weights']
        prediction = model.predict(image)
        direct_error = (five_points(prediction['state']) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
        best = direct_error.detach().argmin(-1)
        rows = torch.arange(BATCH, device='cuda')
        direct_loss = direct_error[rows, best].mean() / 500
        direct_loss = direct_loss + .1 * (-model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean())
        direct_loss = direct_loss + .2 * F.nll_loss(prediction['log_mass'], best)
        valid = (1 - sample['support'] + sample['visible']).clamp(0, 1)
        appearance_loss = ((prediction['appearance'][:, 0] - sample['clean']).abs() * valid).sum() / valid.sum().clamp_min(1)
        cosine = .2 + .8 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        for group, peak in zip(optimizer.param_groups, (1e-5, 3e-5)):
            group['lr'] = peak * cosine
        optimizer.zero_grad(set_to_none=True)
        (.3 * direct_loss + .05 * appearance_loss).backward(retain_graph=True)
        scores = image.new_empty(BATCH, 16)
        slab_errors = image.new_empty(BATCH, 16)
        fitted_errors = image.new_empty(BATCH, 16)
        fit_pose_gradient = None
        pair_loss_total = 0.
        py, px = sample['psf_pixel_y'], sample['psf_pixel_x']
        for pair in range(8):
            flat = pair_order[step - 1, 2 * pair:2 * pair + 2].cuda()
            mode = (flat // 2)[None].expand(BATCH, -1)
            flag = (flat % 2)[None].expand(BATCH, -1)
            fitted = model.fit(prediction, atlas, offsets, weights, mode, flag, steps=4)
            state = fitted['state'].reshape(BATCH, 2, 12)
            point_error = (five_points(state) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
            observed = torch.where(flag.reshape(-1, 1, 1, 1, 1).bool(),
                                   fitted['coordinates'].flip(-2), fitted['coordinates'])
            predicted_slab = observed[:, :, py[:, None], px[None, :], :]
            target_slab = sample['slab'][:, None].expand(-1, 2, -1, -1, -1, -1).reshape_as(predicted_slab)
            visible = sample['visible'][:, py[:, None], px[None, :]]
            q = visible[:, None].expand(-1, 2, -1, -1).reshape(2 * BATCH, len(py), len(px))
            w = weights[:, None].expand(-1, 2, -1).reshape(2 * BATCH, -1)
            slab_error = ((predicted_slab - target_slab).square().sum(-1) + 1e-4).sqrt()
            slab_error = ((slab_error * q[:, None] * w[:, :, None, None]).sum((1, 2, 3)) /
                          q.sum((1, 2)).clamp_min(1)).reshape(BATCH, 2)
            prior = prediction['log_mass'][rows[:, None], mode]
            logit = prediction['reflection_logit'][rows[:, None], mode]
            prior = prior + F.logsigmoid(torch.where(flag.bool(), logit, -logit))
            score = fitted['fit_quality'] + .2 * prior - .05 * fitted['difficulty'].detach()
            quality = F.smooth_l1_loss(score, -torch.log1p(slab_error.detach() / 500))
            margin = (slab_error[:, 0] - slab_error[:, 1]).detach().abs() > 250
            rank = F.cross_entropy(score[margin], slab_error.detach()[margin].argmin(-1)) if margin.any() else score.sum() * 0
            positive = (mode == best[:, None]) & (flag == reflection[:, None])
            physical = score.sum() * 0
            if positive.any():
                trajectory = fitted['pose_sequence'].reshape(BATCH, 2, 5, 12)[:, :, 1:-1]
                trajectory_error = (five_points(trajectory) - five_points(truth)[:, None, None]).norm(dim=-1).mean(-1)
                rendered = fitted['rendered'][:, 0].reshape(BATCH, 2, SIDE, SIDE)
                tissue = sample['visible'][:, None]
                appearance_fit = (((prediction['appearance'][:, 0, None] - rendered).square() * tissue)
                                  .sum((2, 3)) / tissue.sum((2, 3)).clamp_min(1))
                field = fitted['geometry']
                field = F.interpolate(torch.cat((field['residual_local_um'] / 200,
                                                 field['director_delta_local'] / .1), 1),
                                      (8, 8), mode='bilinear', align_corners=False).reshape(BATCH, 2, 6, 8, 8)
                field_error = F.smooth_l1_loss(field, sample['field'][:, None].expand_as(field), reduction='none').mean((2, 3, 4))
                physical = ((point_error / 500 + .5 * slab_error / 500 +
                             .08 * trajectory_error.mean(-1) / 500 + .05 * field_error +
                             .1 * appearance_fit +
                             .01 * fitted['difficulty']) * positive).sum() / BATCH
                if step == 1:
                    gradient = torch.autograd.grad(physical, model.pose[-1].weight,
                                                   retain_graph=True)[0]
                    fit_pose_gradient = float(gradient.norm().detach())
                    assert math.isfinite(fit_pose_gradient) and fit_pose_gradient > 0
            pair_loss = physical + (.25 * quality + .1 * rank) / 8
            pair_loss.backward(retain_graph=pair < 7)
            pair_loss_total += float(pair_loss.detach())
            scores[:, flat] = score.detach()
            slab_errors[:, flat] = slab_error.detach()
            fitted_errors[:, flat] = point_error.detach()
        gradient = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
        optimizer.step()
        selected = scores.argmax(-1)
        row = {'stage_step': step, 'total_step': 161000 + step,
            'base_index': int(base_indices[step - 1]),
            'virtual_indices': virtual_indices[step - 1].tolist(),
            'eligible_optimizer_planes': step * BATCH, 'attempted_planes': attempted,
            'loss': float((.3 * direct_loss + .05 * appearance_loss).detach()) + pair_loss_total,
            'direct_best_five_point_um': float(direct_error[rows, best].mean().detach()),
            'fitted_selected_five_point_um': float(fitted_errors[rows, selected].mean()),
            'fitted_oracle_five_point_um': float(fitted_errors.min(-1).values.mean()),
            'selected_slab_um': float(slab_errors[rows, selected].mean()),
            'physical_slab_selection': float((selected == slab_errors.argmin(-1)).float().mean()),
            'fit_only_pose_head_gradient_norm': fit_pose_gradient,
            'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 100 == 0:
            trace.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 5000 == 0:
            save_checkpoint(step, attempted)
for name, expected in source_sha.items():
    assert sha(repository / 'training' / name) == expected, name
(RUN / 'completed.json').write_text(json.dumps({'stage_updates': STEPS,
    'total_updates': 161000 + STEPS, 'eligible_optimizer_synthetic_planes': STEPS * BATCH,
    'attempted_synthetic_planes': attempted, 'seconds': time.perf_counter() - started,
    'calibrated': False, 'scope': config['scope']}, indent=2), encoding='utf8')
print('All-branch joint feedback stage complete; frozen audit and development evaluation required', flush=True)
