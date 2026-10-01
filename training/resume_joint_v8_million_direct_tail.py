"""Replay the unsaved final 5,000 direct-pose updates from the frozen 120k checkpoint."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import (
    load_streaming_synthetic_v7_64, sample_streaming_synthetic_v7_64,
)

SOURCE = ROOT / 'runs/joint_v8_million_direct_001'
OUTPUT = ROOT / 'runs/joint_v8_million_direct_tail_001'
SEED, START, END, BATCH, SIDE = 2026093031, 120000, 125000, 8, 192
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


original_done = json.loads((SOURCE / 'completed.json').read_text())
assert original_done['stage_updates'] == END
assert original_done['unique_eligible_optimizer_synthetic_planes'] == END * BATCH
parent_path = SOURCE / 'joint_step_156000.pt'
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 36000 + START and checkpoint['stage_step'] == START
assert checkpoint['eligible_optimizer_synthetic_planes'] == START * BATCH
assert not checkpoint['calibrated']
config = checkpoint['config']
assert config['seed'] == SEED and config['stage_updates'] == END
for name, digest in config['source_sha256'].items():
    assert sha(repository / 'training' / name) == digest, name
schedule = torch.load(SOURCE / 'schedule.pt', map_location='cpu', weights_only=True)
assert sha(SOURCE / 'schedule.pt') == config['schedule_sha256']
assert schedule['virtual_index'].shape == (END, BATCH)
reference = {}
with (SOURCE / 'training.jsonl').open(encoding='utf8') as stream:
    for line in stream:
        row = json.loads(line)
        if row['stage_step'] > START:
            reference[row['stage_step']] = row
assert len(reference) == END - START
context = load_streaming_synthetic_v7_64(device='cuda')
assert context['bindings'] == config['data_sha256']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
model = JointSliceFeedbackModel().cuda()
model.load_state_dict(checkpoint['model_state'], strict=True)
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(('encoder.', 'pose.')))
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=8e-5, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state'])
attempted = checkpoint['attempted_synthetic_planes']
torch.set_rng_state(checkpoint['torch_rng'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
del checkpoint


def predict_direct(inputs):
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


def draw(subjects, step, trace):
    pending = list(range(len(subjects)))
    accepted = {}
    attempts = attempted_planes = 0
    while pending:
        seed = SEED * 10**18 + (step + attempts) * (step + attempts + 1) // 2 + attempts
        batch = sample_streaming_synthetic_v7_64(context, subjects[pending].tolist(), seed, side=SIDE)
        selected = batch['eligible'].nonzero()[:, 0].tolist()
        for row, record in enumerate(batch['provenance']):
            slot = pending[row]
            used = row in selected
            trace.write(json.dumps({**record, 'stage_step': step, 'role': 'optimizer',
                                    'batch_slot': slot, 'attempt': attempts, 'used': used}) + '\n')
            if used:
                accepted[slot] = {key: batch[key][row:row + 1]
                                  for key in ('inputs', 'state', 'reflection')}
        attempted_planes += len(pending)
        pending = [slot for row, slot in enumerate(pending) if row not in selected]
        attempts += 1
    return {key: torch.cat([accepted[i][key] for i in range(len(subjects))])
            for key in ('inputs', 'state', 'reflection')}, attempted_planes


OUTPUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, OUTPUT / Path(__file__).name)
receipt = {'source_run': str(SOURCE), 'source_completed_sha256': sha(SOURCE / 'completed.json'),
           'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': sha(parent_path),
           'source_schedule_sha256': sha(SOURCE / 'schedule.pt'),
           'source_training_log_sha256': sha(SOURCE / 'training.jsonl'),
           'replay_script_sha256': sha(Path(__file__)), 'start_stage_step': START,
           'end_stage_step': END, 'initial_attempted_synthetic_planes': attempted,
           'comparison': 'same frozen per-step schedule, accepted-draw counts, loss and physical error',
           'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'experiment.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
max_loss_gap = max_error_gap = 0.
with (OUTPUT / 'training_tail.jsonl').open('w', encoding='utf8') as trace, \
     (OUTPUT / 'synthetic_draws_tail.jsonl').open('w', encoding='utf8') as draws:
    for step in range(START + 1, END + 1):
        subjects = schedule['virtual_index'][step - 1]
        sample, tries = draw(subjects, step, draws)
        attempted += tries
        for group in optimizer.param_groups:
            group['lr'] = 1e-5 + 7e-5 * .5 * (1 + math.cos(math.pi * (step - 1) / END))
        prediction = predict_direct(sample['inputs'])
        truth, reflection = sample['state'], sample['reflection']
        distance = (five_points(prediction['state']) - five_points(truth)[:, None]).norm(dim=-1).mean(-1)
        best = distance.detach().argmin(-1)
        rows = torch.arange(BATCH, device='cuda')
        physical_loss = distance[rows, best].mean() / 500
        pose_nll = -model.component_log_prob(prediction, truth, reflection).logsumexp(-1).mean()
        mode_ce = F.nll_loss(prediction['log_mass'], best)
        loss = physical_loss + .1 * pose_nll + .2 * mode_ce
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
        expected = reference[step]
        assert expected['virtual_indices'] == subjects.tolist()
        assert attempted == expected['attempted_planes'], step
        loss_gap = abs(float(loss.detach()) - expected['loss'])
        error_gap = abs(float(physical_loss.detach() * 500) - expected['best_five_point_um'])
        max_loss_gap = max(max_loss_gap, loss_gap)
        max_error_gap = max(max_error_gap, error_gap)
        assert loss_gap < .01 and error_gap < 5., (step, loss_gap, error_gap)
        trace.write(json.dumps({'stage_step': step, 'total_step': 36000 + step,
            'virtual_indices': subjects.tolist(), 'attempted_planes': attempted,
            'loss': float(loss.detach()), 'best_five_point_um': float(physical_loss.detach() * 500),
            'reference_loss_gap': loss_gap, 'reference_error_gap_um': error_gap}) + '\n')
        if step % 500 == 0:
            trace.flush()
            draws.flush()
            print(json.dumps({'stage_step': step, 'attempted_planes': attempted,
                              'max_loss_gap': max_loss_gap, 'max_error_gap_um': max_error_gap}), flush=True)
assert attempted == original_done['attempted_synthetic_planes']
torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'step': 36000 + END, 'stage_step': END, 'config': config,
            'eligible_optimizer_synthetic_planes': END * BATCH,
            'attempted_synthetic_planes': attempted, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'calibrated': False},
           OUTPUT / 'joint_step_161000.pt')
for name, digest in config['source_sha256'].items():
    assert sha(repository / 'training' / name) == digest, name
(OUTPUT / 'completed.json').write_text(json.dumps({
    'stage_updates': END, 'total_updates': 36000 + END,
    'unique_eligible_optimizer_synthetic_planes': END * BATCH,
    'attempted_synthetic_planes': attempted, 'replayed_tail_updates': END - START,
    'max_reference_loss_gap': max_loss_gap, 'max_reference_error_gap_um': max_error_gap,
    'final_checkpoint_sha256': sha(OUTPUT / 'joint_step_161000.pt'),
    'training_tail_sha256': sha(OUTPUT / 'training_tail.jsonl'),
    'synthetic_draws_tail_sha256': sha(OUTPUT / 'synthetic_draws_tail.jsonl'),
    'scope': 'deterministic recovery of unsaved final direct-pose weights; not validation or deployment',
    'calibrated': False}, indent=2), encoding='utf8')
print('Final million-plane checkpoint recovered; held-out development evaluation required', flush=True)
