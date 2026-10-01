"""Restore real-image pose competence without losing arbitrary-plane candidates.

Continue the same randomly initialized whole model after the exact million-plane
checkpoint. Real TRAIN labels are weak Allen affines, never expert ground truth.
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
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train,
)
from training.arbitrary_plane_streaming_synthetic_v7_64 import (
    load_streaming_synthetic_v7_64, sample_streaming_synthetic_v7_64,
)

PARENT = ROOT / 'runs/joint_v8_million_direct_tail_001'
RUN = ROOT / 'runs/joint_v8_mixed_direct_001'
SEED, STEPS, BATCH, SIDE = 2026100111, 20000, 8, 192
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


def predict_direct(inputs):
    x = inputs
    for layer in model.encoder:
        x = layer(x)
    z = model.pose(torch.cat((F.adaptive_avg_pool2d(x, 4).flatten(1),
                              inputs.new_zeros(len(inputs), 12)), -1))
    z = z.reshape(-1, model.modes, 21)
    centre = model.center_origin + z[..., :3] * model.center_scale
    return {'state': torch.cat((centre, z[..., 3:9],
                                math.log(12000.) + z[..., 9:11].clamp(-2.5, 2.), z[..., 11:12]), -1),
            'std': (.005 + F.softplus(z[..., 12:18])).clamp_max(4.),
            'concentration': (.05 + F.softplus(z[..., 18])).clamp_max(500.),
            'log_mass': z[..., 19].log_softmax(-1), 'reflection_logit': z[..., 20]}


parent_done = json.loads((PARENT / 'completed.json').read_text())
assert parent_done['unique_eligible_optimizer_synthetic_planes'] == 1000000
assert parent_done['replayed_tail_updates'] == 5000 and parent_done['max_reference_loss_gap'] == 0
parent_path = PARENT / 'joint_step_161000.pt'
assert sha(parent_path) == parent_done['final_checkpoint_sha256']
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 161000 and not checkpoint['calibrated']
synthetic = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
assert len(synthetic['bases']) == 64 and len(synthetic['subjects']) == 4096
assert len(real['donors']) == 1885 and real['training_images'] == 263754
model = JointSliceFeedbackModel().cuda()
model.load_state_dict(checkpoint['model_state'], strict=True)
del checkpoint
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(('encoder.', 'pose.')))
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=4e-5, weight_decay=1e-4)

generator = torch.Generator().manual_seed(SEED)
base_indices = torch.randint(64, (STEPS,), generator=generator)
virtual_indices = base_indices[:, None] * 64 + torch.randint(64, (STEPS, BATCH), generator=generator)
real_donor_indices = torch.randint(len(real['donors']), (STEPS,), generator=generator)
real_row_indices = torch.stack([
    torch.randint(len(real['donors'][int(donor)]['state']), (BATCH,), generator=generator)
    for donor in real_donor_indices
])
assert torch.equal(virtual_indices // 64, base_indices[:, None].expand_as(virtual_indices))
RUN.mkdir(parents=True, exist_ok=False)
torch.save({'base_index': base_indices, 'virtual_index': virtual_indices,
            'real_donor_index': real_donor_indices, 'real_row_index': real_row_indices},
           RUN / 'schedule.pt')
(RUN / 'identities.json').write_text(json.dumps(synthetic['subjects']), encoding='utf8')
source_names = sorted(set(synthetic['provenance']['source_sha256']) | {
    'arbitrary_plane_joint_model_v8.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_reserved_real_stream_v8.py', 'train_joint_v8_mixed_direct.py',
})
source_sha = {}
for name in source_names:
    source = repository / 'training' / name
    shutil.copyfile(source, RUN / name)
    source_sha[name] = sha(source)
config = {'seed': SEED, 'stage_updates': STEPS, 'batch': BATCH, 'resolution': [SIDE, SIDE],
    'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completed_sha256': sha(PARENT / 'completed.json'),
    'initialization': 'strict whole-model continuation of standalone random-init v8 lineage',
    'trainable_modules': ['encoder', 'pose'],
    'synthetic': '64 independent TRAIN maps; fresh eligible arbitrary brain-intersecting planes',
    'real': '263754 TRAIN images from 1885 donors; one donor uniformly per update, eight sections within donor; weak Allen affine labels',
    'loss': 'synthetic best-of-eight physical five-point /500um + 0.1 mixture NLL + 1.0 best-mode CE; plus 0.5 times the same weak-real loss',
    'optimizer': 'fresh AdamW 4e-5 cosine to 1e-5, norm clip 5, FP32',
    'scope': 'joint synthetic and real TRAIN direct-pose retention; no fitting feedback, expert validation, calibration, benchmark or deployment',
    'calibrated': False, 'constraints': 'absent, not trained',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name()},
    'source_sha256': source_sha, 'synthetic_data_sha256': synthetic['bindings'],
    'real_data_sha256': real['bindings'], 'real_manifest_sha256': real['manifest_sha256'],
    'synthetic_provenance': synthetic['provenance'],
    'schedule_sha256': sha(RUN / 'schedule.pt'), 'identities_sha256': sha(RUN / 'identities.json')}
(RUN / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf8')


def draw(subjects, step, trace):
    pending = list(range(BATCH))
    accepted = {}
    attempts = attempted_planes = 0
    while pending:
        seed = SEED * 10**18 + (step + attempts) * (step + attempts + 1) // 2 + attempts
        batch = sample_streaming_synthetic_v7_64(synthetic, subjects[pending].tolist(), seed, side=SIDE)
        selected = batch['eligible'].nonzero()[:, 0].tolist()
        for row, record in enumerate(batch['provenance']):
            slot = pending[row]
            used = row in selected
            trace.write(json.dumps({**record, 'stage_step': step, 'batch_slot': slot,
                                    'attempt': attempts, 'used': used}) + '\n')
            if used:
                accepted[slot] = {key: batch[key][row:row + 1]
                                  for key in ('inputs', 'state', 'reflection')}
        attempted_planes += len(pending)
        pending = [slot for row, slot in enumerate(pending) if row not in selected]
        attempts += 1
    return {key: torch.cat([accepted[i][key] for i in range(BATCH)])
            for key in ('inputs', 'state', 'reflection')}, attempted_planes


def save_checkpoint(step, attempted):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': 161000 + step, 'stage_step': step, 'config': config,
                'eligible_optimizer_synthetic_planes': step * BATCH,
                'real_training_image_exposures': step * BATCH,
                'attempted_synthetic_planes': attempted,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'calibrated': False}, RUN / f'joint_step_{161000 + step:06d}.pt')


save_checkpoint(0, 0)
started, attempted = time.perf_counter(), 0
with (RUN / 'training.jsonl').open('w', encoding='utf8') as trace, \
     (RUN / 'synthetic_draws.jsonl').open('w', encoding='utf8') as synthetic_draws, \
     (RUN / 'real_draws.jsonl').open('w', encoding='utf8') as real_draws:
    for step in range(1, STEPS + 1):
        sample, tries = draw(virtual_indices[step - 1], step, synthetic_draws)
        attempted += tries
        donor = int(real_donor_indices[step - 1])
        real_sample = sample_reserved_real_train(real, donor, real_row_indices[step - 1].tolist())
        for identity in real_sample['identities']:
            real_draws.write(json.dumps({**identity, 'stage_step': step, 'donor_index': donor}) + '\n')
        optimizer.param_groups[0]['lr'] = 1e-5 + 3e-5 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for domain, batch in (('synthetic', sample), ('real_weak', real_sample)):
            prediction = predict_direct(batch['inputs'])
            truth, reflection = batch['state'], batch['reflection']
            distance = (five_points(prediction['state']) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
            best = distance.detach().argmin(-1)
            rows = torch.arange(BATCH, device='cuda')
            physical = distance[rows, best].mean() / 500
            pose_nll = -model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean()
            mode_ce = F.nll_loss(prediction['log_mass'], best)
            loss = physical + .1 * pose_nll + mode_ce
            assert torch.isfinite(loss), (step, domain)
            (loss if domain == 'synthetic' else .5 * loss).backward()
            metrics[domain] = {'loss': float(loss.detach()),
                'selected_five_point_um': float(distance[rows, prediction['log_mass'].argmax(-1)].mean().detach()),
                'oracle_five_point_um': float(distance[rows, best].mean().detach()),
                'mode_ce': float(mode_ce.detach())}
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'stage_step': step, 'total_step': 161000 + step,
               'base_index': int(base_indices[step - 1]),
               'virtual_indices': virtual_indices[step - 1].tolist(),
               'real_donor_index': donor, 'real_row_indices': real_row_indices[step - 1].tolist(),
               'eligible_optimizer_synthetic_planes': step * BATCH,
               'real_training_image_exposures': step * BATCH, 'attempted_synthetic_planes': attempted,
               'synthetic': metrics['synthetic'], 'real_weak': metrics['real_weak'],
               'gradient_norm': float(gradient), 'learning_rate': optimizer.param_groups[0]['lr'],
               'seconds': time.perf_counter() - started}
        trace.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
            trace.flush()
            synthetic_draws.flush()
            real_draws.flush()
            print(json.dumps(row), flush=True)
        if step % 5000 == 0:
            save_checkpoint(step, attempted)
for name, digest in source_sha.items():
    assert sha(repository / 'training' / name) == digest, name
(RUN / 'completed.json').write_text(json.dumps({
    'stage_updates': STEPS, 'total_updates': 161000 + STEPS,
    'eligible_optimizer_synthetic_planes': STEPS * BATCH,
    'real_training_image_exposures': STEPS * BATCH,
    'attempted_synthetic_planes': attempted,
    'final_checkpoint_sha256': sha(RUN / f'joint_step_{161000 + STEPS:06d}.pt'),
    'seconds': time.perf_counter() - started, 'calibrated': False,
    'scope': config['scope']}, indent=2), encoding='utf8')
print('Mixed direct-pose stage complete; independent synthetic and real development readouts required', flush=True)
