"""Continue one-pass pose/warp training with fresh planes and unrepeated real TRAIN sections."""
import hashlib
import json
import math
import os
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
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT_RUN = ROOT / 'runs/one_shot_joint_pose_curriculum_002'
PARENT = PARENT_RUN / 'best_development.pt'
RUN = ROOT / 'runs/one_shot_joint_mixed_real_003'
SEED, UPDATES, SYNTHETIC_PER_BATCH, SIDE = 2026100203, 20000, 3, 256
assert json.loads((PARENT_RUN / 'completed.json').read_text())['updates'] == 10000
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
synthetic = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
assert real['training_images'] == 263754 and len(real['donors']) == 1885
parent = torch.load(PARENT, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel().cuda()
model.load_state_dict(parent['model'])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
optimizer.load_state_dict(parent['optimizer'])
for group in optimizer.param_groups:
    group['lr'] = 3e-5
del parent
RUN.mkdir(parents=True, exist_ok=False)


def points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    uv = state.new_tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]]) / SIDE
    uv = uv.expand(*state.shape[:-1], -1, -1).clone()
    uv[..., 0] = torch.where(reflection[..., None].bool(), 255 / SIDE - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv - .5)


rng = np.random.default_rng(SEED)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < UPDATES:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == UPDATES:
                break
schedule = np.asarray(schedule, dtype=np.int32)
np.save(RUN / 'real_schedule.npy', schedule)
config = {'seed': SEED, 'updates': UPDATES, 'synthetic_per_batch': SYNTHETIC_PER_BATCH,
    'distinct_real_train_sections': UPDATES, 'resolution': SIDE,
    'parent_checkpoint': str(PARENT), 'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    'initialization': 'whole-model continuation of random-start one-shot lineage',
    'synthetic_sampling': 'three new continuous arbitrary planes per batch, each with independently random appearance and damage',
    'real_sampling': 'one TRAIN section per batch, randomized donor rounds and within-donor order; no real section repeats',
    'real_label_role': real['label_role'], 'real_bindings': real['bindings'],
    'synthetic_provenance': synthetic['provenance'],
    'real_schedule_sha256': hashlib.sha256((RUN / 'real_schedule.npy').read_bytes()).hexdigest(),
    'fit_feedback': 'deferred pending useful direct plane accuracy; this stage trains pose and correct-plane local mapping',
    'calibrated': False, 'public_benchmark_used': False,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 10000000


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False}, RUN / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w', encoding='utf8') as log, \
     (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws:
    for step in range(1, UPDATES + 1):
        accepted, pending = {}, list(range(SYNTHETIC_PER_BATCH))
        while pending:
            virtual = torch.randint(len(synthetic['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sample = sample_one_shot_stream(synthetic, virtual, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sample['provenance']):
                slot = pending[row]
                used = bool(sample['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: value[row:row + 1] for key, value in sample.items()
                                      if key in ('inputs', 'state', 'reflection', 'offsets', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(SYNTHETIC_PER_BATCH)]) for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        observation['inputs'] = F.interpolate(observation['inputs'], (SIDE, SIDE), mode='bilinear', align_corners=False)
        draws.write(json.dumps({**observation['identities'][0], 'step': step, 'slot': SYNTHETIC_PER_BATCH,
                                'used': True, 'label_role': real['label_role']}) + '\n')
        inputs = torch.cat((batch['inputs'], observation['inputs']))
        truth = torch.cat((batch['state'], observation['state']))
        reflected = torch.cat((batch['reflection'], observation['reflection']))
        prediction = model.predict(inputs)
        candidate = points(prediction['state'], reflected[:, None].expand(-1, model.modes))
        distance = (candidate - points(truth, reflected)[:, None]).norm(dim=-1).mean(-1)
        best_mode = distance.detach().argmin(-1)
        direct = distance.gather(1, best_mode[:, None]).mean() / 1000
        nll = -model.component_log_prob(prediction, truth, reflected).logsumexp(-1).mean()
        rank = F.kl_div(prediction['log_mass'], (-distance.detach() / 2000).softmax(-1), reduction='batchmean')
        reflection_loss = F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'], reflected[:, None].float().expand_as(prediction['reflection_logit']))
        pose = prediction['state'][:SYNTHETIC_PER_BATCH].scatter(
            1, best_mode[:SYNTHETIC_PER_BATCH, None, None].expand(-1, 1, 12), batch['state'][:, None])
        mapping_prediction = {key: value[:SYNTHETIC_PER_BATCH] if torch.is_tensor(value) else value
                              for key, value in prediction.items()}
        mapping_prediction['state'] = pose
        mapped = model.map(mapping_prediction, batch['offsets'], best_mode[:SYNTHETIC_PER_BATCH, None],
                           batch['reflection'][:, None], (SIDE, SIDE))
        valid = batch['valid_mask'].float()
        error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
        mapping = (error * valid).sum() / valid.sum().clamp_min(1) / 500
        field = mapped['local_displacement_um'][:, 0]
        magnitude = field.square().mean().sqrt() / 1000
        smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                  + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
        mask = F.binary_cross_entropy_with_logits(mapped['correspondence_logit'][:, 0, 0], valid)
        loss = direct + .1 * nll + .5 * rank + .3 * reflection_loss \
            + mapping + .1 * magnitude + .02 * smooth + .1 * mask
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'presentations': step * 4, 'real_distinct': step,
            'direct_five_point_um': float(direct.detach() * 1000), 'real_weak_five_point_um': float(distance[-1, best_mode[-1]].detach()),
            'map_error_um': float(mapping.detach() * 500), 'nll': float(nll.detach()),
            'rank': float(rank.detach()), 'reflection_loss': float(reflection_loss.detach()),
            'loss': float(loss.detach()), 'gradient': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 200 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
        if step % 2000 == 0:
            save(step)
save(UPDATES)
(RUN / 'completed.json').write_text(json.dumps({'updates': UPDATES, 'presentations': UPDATES * 4,
    'distinct_real_train_sections': UPDATES, 'seconds': time.perf_counter() - started,
    'calibrated': False, 'public_benchmark_used': False}, indent=2), encoding='utf8')
