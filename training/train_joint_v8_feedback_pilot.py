"""Train actual atlas-render/fit/full-pose feedback on fresh TRAIN planes.

This is a bounded architecture diagnostic, not the final-scale or calibrated
model. The only parent is the whole v7 checkpoint from this random-init lineage.
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
from training.arbitrary_plane_streaming_synthetic_v7 import load_streaming_synthetic_v7, sample_streaming_synthetic_v7

RUN = ROOT / 'runs/joint_v8_feedback_pilot_001'
PARENT = ROOT / 'runs/joint_v7_streaming_joint_002'
SEED, STEPS, BATCH, SIDE = 2026093018, 6000, 4, 192
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def five_points(state, reflection=None):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    uv = uv.expand(*state.shape[:-1], 5, 2).clone()
    if reflection is not None:
        uv[..., 0] = torch.where(reflection[..., None].bool(), end - 1 - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv)


parent_done = json.loads((PARENT / 'completed.json').read_text())
assert parent_done['total_updates'] == 30000 and parent_done['unique_eligible_optimizer_synthetic_planes'] == 144000
context = load_streaming_synthetic_v7(device='cuda')
assert len(context['bases']) == 8 and len(context['subjects']) == 512
parent_path = PARENT / 'joint_step_30000.pt'
with parent_path.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 30000 and not checkpoint['calibrated']
model = JointSliceFeedbackModel().cuda()
missing, unexpected = model.load_state_dict(checkpoint['model_state'], strict=False)
new_names = tuple(name for name in model.state_dict() if name.startswith(
    ('pose_residual.', 'fit_quality.', 'pose_update_limits')))
assert set(missing) == set(new_names) and not unexpected
old = [p for name, p in model.named_parameters() if not name.startswith(('pose_residual.', 'fit_quality.'))]
new = list(model.pose_residual.parameters()) + list(model.fit_quality.parameters())
optimizer = torch.optim.AdamW([{'params': old, 'lr': 2e-5}, {'params': new, 'lr': 2e-4}], weight_decay=1e-4)
generator = torch.Generator().manual_seed(SEED)
schedule = torch.randint(512, (STEPS, BATCH), generator=generator)
other_mode_offset = torch.randint(1, 8, (STEPS, BATCH), generator=generator)
other_reflection_flip = torch.randint(0, 2, (STEPS, BATCH), generator=generator)
torch.manual_seed(SEED + 1)
torch.cuda.manual_seed_all(SEED + 1)
RUN.mkdir(parents=True, exist_ok=False)
source_names = ('arbitrary_plane_joint_model_v8.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_subject_torch_v6.py', 'arbitrary_plane_subject_sampling_v6.py',
    'train_joint_v8_feedback_pilot.py')
source_sha = {}
for name in source_names:
    path = repository / 'training' / name
    shutil.copyfile(path, RUN / name)
    source_sha[name] = hashlib.sha256(path.read_bytes()).hexdigest()
torch.save({'virtual_indices': schedule, 'other_mode_offset': other_mode_offset,
            'other_reflection_flip': other_reflection_flip}, RUN / 'schedule.pt')
config = {'seed': SEED, 'stage_updates': STEPS, 'batch': BATCH, 'resolution': [SIDE, SIDE],
    'modes': 8, 'fitter_steps': 4, 'parent_checkpoint': str(parent_path),
    'parent_checkpoint_sha256': parent_sha, 'parent_total_updates': 30000,
    'initialization': 'whole v7 randomly initialized lineage plus zero-initialized recurrent pose and quality heads; fresh AdamW',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name()},
    'sampling': 'four fresh physical arbitrary TRAIN planes/update; ineligible planes preserved and excluded, no re-draw; two fitted hypotheses/eligible plane, one geometry-best and one independently offset mode; second reflection flips with probability .5',
    'pose_only_first_iterations': 2,
    'quality_target': 'negative log1p(sparse physical 3D slab error /500um), two candidate branches; ranking uses .2 direct log prior and -.05 deformation difficulty',
    'loss': '.25 direct pose + final fitted best-branch five-point/500um + .12 intermediate fitted pose/500um + main sparse slab/500um + .25 competitor slab/500um + .25 quality Huber + .2 pairwise selection CE + .1 teacher field + .05 clean appearance',
    'uncertainty': 'parent covariance head retained but uncalibrated; no region probability claim',
    'constraints': 'all optional context/marks absent; no trained constraint support',
    'scope': 'synthetic TRAIN architecture pilot, not animal validation, public benchmark or deployment',
    'source_sha256': source_sha, 'data_sha256': context['bindings'],
    'synthetic_provenance': context['provenance'],
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.pt').read_bytes()).hexdigest(),
    'parameters': sum(p.numel() for p in model.parameters()), 'new_state_keys': new_names}
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf8')
atlas = context['atlas'][:1]
del checkpoint
started = time.perf_counter()
used = attempted = 0
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace, \
     (RUN / 'synthetic_draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, STEPS + 1):
        attempt = 0
        while True:
            sample = sample_streaming_synthetic_v7(context, schedule[step - 1].tolist(),
                                                   SEED * 100000 + step + attempt * STEPS, side=SIDE)
            eligible = sample['eligible'].nonzero()[:, 0]
            for row in sample['provenance']:
                draws.write(json.dumps({**row, 'stage_step': step, 'attempt': attempt,
                                        'used': row['eligible']}) + '\n')
            attempted += BATCH
            if len(eligible):
                break
            attempt += 1
        used += len(eligible)
        image, truth = sample['inputs'][eligible], sample['state'][eligible]
        reflection = sample['reflection'][eligible]
        offsets, weights = sample['offsets'][eligible], sample['weights'][eligible]
        prediction = model.predict(image)
        n = len(eligible)
        rows = torch.arange(n, device='cuda')
        direct_error = (five_points(prediction['state']) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
        best = direct_error.detach().argmin(-1)
        direct_loss = direct_error[rows, best].mean() / 500
        direct_loss = direct_loss + .05 * (-model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean())
        direct_loss = direct_loss + .2 * F.nll_loss(prediction['log_mass'], best)
        competitor = (best + other_mode_offset[step - 1, eligible.cpu()].cuda()) % model.modes
        candidate_mode = torch.stack((best, competitor), 1)
        competitor_reflection = reflection ^ other_reflection_flip[step - 1, eligible.cpu()].cuda()
        candidate_reflection = torch.stack((reflection, competitor_reflection), 1)
        teacher = rows == (step % n)
        initial = prediction['state'][rows[:, None], candidate_mode]
        initial = torch.where(teacher[:, None, None] & (torch.arange(2, device='cuda')[None, :, None] == 0),
                              truth[:, None], initial)
        fitted = model.fit(prediction, atlas, offsets, weights, candidate_mode, candidate_reflection,
                           steps=4, initial_state=initial)
        post_state = fitted['state'].reshape(n, 2, 12)
        post_error = (five_points(post_state[:, 0]) - five_points(truth)).norm(dim=-1).mean(-1)
        trajectory = fitted['pose_sequence'].reshape(n, 2, 5, 12)[:, 0, 1:-1]
        trajectory_error = (five_points(trajectory) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
        py, px = sample['psf_pixel_y'], sample['psf_pixel_x']
        source_slab = sample['slab'][eligible][:, None].expand(-1, 2, -1, -1, -1, -1).reshape(2 * n, len(weights[0]), len(py), len(px), 3)
        predicted_slab = fitted['coordinates'][:, :, py[:, None], px[None, :], :]
        visible = sample['visible'][eligible][:, py[:, None], px[None, :]]
        q = visible[:, None].expand(-1, 2, -1, -1).reshape(2 * n, len(py), len(px))
        w = weights[:, None].expand(-1, 2, -1).reshape(2 * n, -1)
        slab_error = ((predicted_slab - source_slab).square().sum(-1) + 1e-4).sqrt()
        slab_error = ((slab_error * q[:, None] * w[:, :, None, None]).sum((1, 2, 3)) /
                      q.sum((1, 2)).clamp_min(1)).reshape(n, 2)
        prior = prediction['log_mass'][rows[:, None], candidate_mode]
        logit = prediction['reflection_logit'][rows[:, None], candidate_mode]
        prior = prior + F.logsigmoid(torch.where(candidate_reflection.bool(), logit, -logit))
        score = fitted['fit_quality'] + .2 * prior - .05 * fitted['difficulty']
        quality_target = -torch.log1p(slab_error.detach() / 500)
        field = fitted['geometry']
        field = F.interpolate(torch.cat((field['residual_local_um'] / 200,
                                         field['director_delta_local'] / .1), 1), (8, 8),
                              mode='bilinear', align_corners=False).reshape(n, 2, 6, 8, 8)
        field_loss = F.smooth_l1_loss(field[teacher, 0], sample['field'][eligible][teacher])
        valid = (1 - sample['support'][eligible] + sample['visible'][eligible]).clamp(0, 1)
        appearance_loss = ((prediction['appearance'][:, 0] - sample['clean'][eligible]).abs() * valid).sum() / valid.sum().clamp_min(1)
        loss = (.25 * direct_loss + post_error.mean() / 500 + .12 * trajectory_error.mean() / 500 +
                slab_error[:, 0].mean() / 500 + .25 * slab_error[:, 1].mean() / 500 +
                .25 * F.smooth_l1_loss(fitted['fit_quality'], quality_target) +
                .2 * F.cross_entropy(score, slab_error.detach().argmin(-1)) +
                .1 * field_loss + .05 * appearance_loss)
        assert torch.isfinite(loss), f'nonfinite loss at step {step}'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'stage_step': step, 'total_step': 30000 + step, 'eligible_planes': used,
            'attempted_planes': attempted, 'loss': float(loss.detach()),
            'direct_best_five_point_um': float(direct_error[rows, best].mean().detach()),
            'fitted_best_five_point_um': float(post_error.mean().detach()),
            'fitted_main_slab_um': float(slab_error[:, 0].mean().detach()),
            'fitted_competitor_slab_um': float(slab_error[:, 1].mean().detach()),
            'quality_huber': float(F.smooth_l1_loss(fitted['fit_quality'], quality_target).detach()),
            'selection_correct': float((score.argmax(-1) == slab_error.argmin(-1)).float().mean().detach()),
            'mean_pose_update_um': float(fitted['pose_updates'][:, :, 3:6].norm(dim=-1).mean().detach()),
            'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 250 == 0:
            trace.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 2000 == 0:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': 30000 + step, 'stage_step': step, 'config': config, 'calibrated': False},
                RUN / f'joint_step_{30000 + step:05d}.pt')
for name, expected in source_sha.items():
    assert hashlib.sha256((repository / 'training' / name).read_bytes()).hexdigest() == expected
(RUN / 'completed.json').write_text(json.dumps({'stage_updates': STEPS, 'total_updates': 30000 + STEPS,
    'eligible_optimizer_synthetic_planes': used, 'attempted_synthetic_planes': attempted,
    'seconds': time.perf_counter() - started, 'calibrated': False, 'scope': config['scope']}, indent=2), encoding='utf8')
print('Recurrent full-pose feedback pilot complete; frozen audit required', flush=True)
