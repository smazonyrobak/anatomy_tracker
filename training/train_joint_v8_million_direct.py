"""Direct-pose continuation on one million fresh eligible TRAIN planes.

Run only after the 64 independent subject-map preparation is frozen. This is
not validation, calibration, fitting feedback, or a deployable checkpoint.
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
from training.arbitrary_plane_streaming_synthetic_v7 import VARIANTS
from training.arbitrary_plane_streaming_synthetic_v7_64 import (
    load_streaming_synthetic_v7_64, sample_streaming_synthetic_v7_64,
)

RUN = ROOT / 'runs/joint_v8_million_direct_001'
PARENT = ROOT / 'runs/joint_v8_feedback_pilot_001'
SEED, STEPS, BATCH, SIDE = 2026093031, 125000, 8, 192
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def five_points(state):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    return centre[..., None, :] + torch.einsum('...ij,pj->...pi', frame[..., :2] @ basis, uv)


parent_done = json.loads((PARENT / 'completed.json').read_text())
assert parent_done['total_updates'] == 36000 and parent_done['stage_updates'] == 6000
parent_path = PARENT / 'joint_step_36000.pt'
with parent_path.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 36000 and not checkpoint['calibrated']
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 64 * VARIANTS
assert VARIANTS == 64
assert all(r['base_index'] == i // VARIANTS and r['virtual_index'] == i
           for i, r in enumerate(context['subjects']))
model = JointSliceFeedbackModel().cuda()
model.load_state_dict(checkpoint['model_state'], strict=True)
del checkpoint
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(('encoder.', 'pose.')))
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=8e-5, weight_decay=1e-4)


def predict_direct(inputs):
    # The frozen image decoder is needed for joint fitting, not this pose-only stage.
    x = inputs
    for layer in model.encoder:
        x = layer(x)
    context = inputs.new_zeros(len(inputs), 12)
    z = model.pose(torch.cat((F.adaptive_avg_pool2d(x, 4).flatten(1), context), -1))
    z = z.reshape(-1, model.modes, 21)
    centre = model.center_origin + z[..., :3] * model.center_scale
    return {'state': torch.cat((centre, z[..., 3:9],
                                math.log(12000.) + z[..., 9:11].clamp(-2.5, 2.), z[..., 11:12]), -1),
            'std': (.005 + F.softplus(z[..., 12:18])).clamp_max(4.),
            'concentration': (.05 + F.softplus(z[..., 18])).clamp_max(500.),
            'log_mass': z[..., 19].log_softmax(-1), 'reflection_logit': z[..., 20]}

generator = torch.Generator().manual_seed(SEED)
base_indices = torch.randint(64, (STEPS,), generator=generator)
virtual_indices = base_indices[:, None] * VARIANTS + torch.randint(
    VARIANTS, (STEPS, BATCH), generator=generator)
assert torch.equal(virtual_indices // VARIANTS, base_indices[:, None].expand_as(virtual_indices))
RUN.mkdir(parents=True, exist_ok=False)
torch.save({'base_index': base_indices, 'virtual_index': virtual_indices}, RUN / 'schedule.pt')
(RUN / 'identities.json').write_text(json.dumps(context['subjects']), encoding='utf8')
source_names = sorted(set(context['provenance']['source_sha256']) | {
    'arbitrary_plane_joint_model_v8.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_joint_inference_v8.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'train_joint_v8_million_direct.py',
})
source_sha = {}
for name in source_names:
    path = repository / 'training' / name
    shutil.copyfile(path, RUN / name)
    source_sha[name] = hashlib.sha256(path.read_bytes()).hexdigest()
config = {
    'seed': SEED, 'stage_updates': STEPS, 'batch': BATCH, 'resolution': [SIDE, SIDE],
    'modes': model.modes, 'fitter_steps': 4,
    'eligible_optimizer_plane_target': STEPS * BATCH,
    'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': parent_sha,
    'parent_completed_sha256': hashlib.sha256((PARENT / 'completed.json').read_bytes()).hexdigest(),
    'parent_total_updates': 36000,
    'initialization': 'strict whole-model continuation of the random-init v8 lineage; fresh direct-only AdamW',
    'trainable_modules': ['encoder', 'pose'],
    'direct_forward': 'same encoder and pose head as model.predict; frozen image decoder skipped',
    'frozen_modules': ['lateral', 'image_decoder', 'atlas_encoder', 'pair', 'fitter',
                       'field', 'uncertainty', 'pose_residual', 'fit_quality'],
    'optimizer': 'AdamW, cosine learning rate 8e-5 to 1e-5, weight decay 1e-4, norm clip 5, FP32',
    'sampling': '64 independent TRAIN inverse-map subjects; one base and eight affine variants per batch; fresh arbitrary brain-intersecting physical planes; rejected slots redrawn for the same virtual identity; all attempts and selected modes recorded',
    'loss': 'geometry-best canonical five-point physical error /500um + .1 mixture pose NLL + .2 best-mode cross-entropy',
    'readout': 'fixed one eligible TRAIN plane per independent base, never optimizer; raw direct predictions at step 0 and each 10000; no DEV/test access',
    'recovery': 'whole model, optimizer, RNG and attempted-plane count saved every 10000 updates; deterministic draw seed and schedule',
    'scope': 'synthetic TRAIN direct-pose exposure, not joint fitting validation, calibration, benchmark, or deployment',
    'calibrated': False, 'constraints': 'absent, not trained',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name()},
    'source_sha256': source_sha, 'data_sha256': context['bindings'],
    'synthetic_provenance': context['provenance'],
    'schedule_sha256': hashlib.sha256((RUN / 'schedule.pt').read_bytes()).hexdigest(),
    'identities_sha256': hashlib.sha256((RUN / 'identities.json').read_bytes()).hexdigest(),
    'trainable_parameters': sum(p.numel() for p in trainable),
    'whole_model_parameters': sum(p.numel() for p in model.parameters()),
}
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf8')
torch.manual_seed(SEED + 1)
torch.cuda.manual_seed_all(SEED + 1)


def draw(subjects, draw_id, stage_step, role, trace):
    pending = list(range(len(subjects)))
    accepted = {}
    attempts = attempted_planes = 0
    while pending:
        seed = SEED * 10**18 + (draw_id + attempts) * (draw_id + attempts + 1) // 2 + attempts
        batch = sample_streaming_synthetic_v7_64(
            context, subjects[pending].tolist(), seed, side=SIDE)
        selected = batch['eligible'].nonzero()[:, 0].tolist()
        for row, record in enumerate(batch['provenance']):
            slot = pending[row]
            used = row in selected
            trace.write(json.dumps({**record, 'stage_step': stage_step, 'role': role,
                                    'batch_slot': slot, 'attempt': attempts, 'used': used}) + '\n')
            if used:
                accepted[slot] = {key: value[row:row + 1] for key, value in batch.items()
                                  if key in ('inputs', 'state', 'reflection')}
        attempted_planes += len(pending)
        pending = [slot for row, slot in enumerate(pending) if row not in selected]
        attempts += 1
    return {key: torch.cat([accepted[i][key] for i in range(len(subjects))])
            for key in ('inputs', 'state', 'reflection')}, attempted_planes


fixed = {'inputs': [], 'state': [], 'reflection': []}
fixed_virtual = torch.arange(64) * VARIANTS + VARIANTS // 2
with (RUN / 'fixed_train_draws.jsonl').open('w', encoding='utf8') as draws:
    for base in range(64):
        batch, _ = draw(fixed_virtual[base:base + 1], STEPS + base + 1,
                        0, 'fixed_train_readout_not_optimizer', draws)
        for key in fixed:
            fixed[key].append(batch[key].cpu())
fixed = {key: torch.cat(value) for key, value in fixed.items()}
torch.save({**fixed, 'virtual_index': fixed_virtual}, RUN / 'fixed_train_inputs.pt')


def readout(stage_step):
    model.eval()
    outputs = {key: [] for key in ('state', 'log_mass', 'std', 'concentration', 'reflection_logit')}
    with torch.no_grad():
        for start in range(0, 64, BATCH):
            prediction = predict_direct(fixed['inputs'][start:start + BATCH].cuda())
            for key in outputs:
                outputs[key].append(prediction[key].cpu())
    outputs = {key: torch.cat(value) for key, value in outputs.items()}
    distance = (five_points(outputs['state']) - five_points(fixed['state'])[:, None]).norm(dim=-1).mean(-1)
    mode = outputs['log_mass'].argmax(-1)
    rows = torch.arange(64)
    np.savez_compressed(RUN / f'train_only_direct_step_{stage_step:06d}.npz',
        **{key: value.numpy() for key, value in outputs.items()},
        target_state=fixed['state'].numpy(), target_reflection=fixed['reflection'].numpy(),
        virtual_index=fixed_virtual.numpy(), base_index=np.arange(64),
        map_mode=mode.numpy(), coordinate_error_um=distance.numpy())
    print(json.dumps({'event': 'fixed_train_direct_readout', 'stage_step': stage_step,
        'base_equal_map_coordinate_um': float(distance[rows, mode].mean()),
        'base_equal_oracle_coordinate_um': float(distance.min(-1).values.mean())}), flush=True)
    model.train()


def save_checkpoint(stage_step, attempted_planes):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': 36000 + stage_step, 'stage_step': stage_step, 'config': config,
                'eligible_optimizer_synthetic_planes': stage_step * BATCH,
                'attempted_synthetic_planes': attempted_planes,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'calibrated': False}, RUN / f'joint_step_{36000 + stage_step:06d}.pt')


save_checkpoint(0, 0)
readout(0)
started = time.perf_counter()
attempted = 0
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace, \
     (RUN / 'synthetic_draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, STEPS + 1):
        subjects = virtual_indices[step - 1]
        sample, tries = draw(subjects, step, step, 'optimizer', draws)
        attempted += tries
        for group in optimizer.param_groups:
            group['lr'] = 1e-5 + 7e-5 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        prediction = predict_direct(sample['inputs'])
        truth, reflection = sample['state'], sample['reflection']
        distance = (five_points(prediction['state']) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
        best = distance.detach().argmin(-1)
        rows = torch.arange(BATCH, device='cuda')
        physical_loss = distance[rows, best].mean() / 500
        pose_nll = -model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean()
        mode_ce = F.nll_loss(prediction['log_mass'], best)
        loss = physical_loss + .1 * pose_nll + .2 * mode_ce
        assert torch.isfinite(loss), f'nonfinite direct loss at stage step {step}'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'stage_step': step, 'total_step': 36000 + step,
            'base_index': int(base_indices[step - 1]), 'virtual_indices': subjects.tolist(),
            'eligible_optimizer_planes': step * BATCH, 'attempted_planes': attempted,
            'loss': float(loss.detach()), 'best_five_point_um': float(physical_loss.detach() * 500),
            'pose_nll': float(pose_nll.detach()), 'mode_ce': float(mode_ce.detach()),
            'gradient_norm': float(gradient), 'learning_rate': optimizer.param_groups[0]['lr'],
            'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
            trace.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 10000 == 0:
            save_checkpoint(step, attempted)
            readout(step)
for name, expected in source_sha.items():
    assert hashlib.sha256((repository / 'training' / name).read_bytes()).hexdigest() == expected
(RUN / 'completed.json').write_text(json.dumps({'stage_updates': STEPS,
    'total_updates': 36000 + STEPS, 'unique_eligible_optimizer_synthetic_planes': STEPS * BATCH,
    'attempted_synthetic_planes': attempted, 'seconds': time.perf_counter() - started,
    'calibrated': False, 'scope': config['scope']}, indent=2), encoding='utf8')
print('Million-plane direct stage complete; frozen audit and joint fine-tuning still required', flush=True)
