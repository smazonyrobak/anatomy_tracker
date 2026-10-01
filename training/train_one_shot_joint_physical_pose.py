"""Native-256 continuation: train one 8x2 pose posterior on observed tissue geometry."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

PARENT = ROOT / 'runs/one_shot_atlas_conditioned_warp_matched_001/atlas_conditioned/warp_step_04000.pt'
RUN = ROOT / 'runs/one_shot_joint_physical_pose_002'
SEED, UPDATES, SIDE, SYNTHETIC = 2026100702, 8000, 256, 3
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
assert real['training_images'] == 263754 and len(real['donors']) == 1885
checkpoint = torch.load(PARENT, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 4000 and not checkpoint['calibrated']
model = OneShotJointSliceModel(atlas_conditioning=True).cuda()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 2e-5},
    {'params': model.pose.parameters(), 'lr': 1e-4},
    {'params': (p for name, p in model.named_parameters()
                if not name.startswith(('encoder.', 'pose.'))), 'lr': 3e-5},
], weight_decay=1e-4)

used_real = {tuple(row) for path in (
    ROOT / 'runs/one_shot_joint_mixed_real_003/real_schedule.npy',
    ROOT / 'runs/one_shot_fit_feedback_matched_001/new_real_schedule.npy')
    for row in np.load(path)}
rng = np.random.default_rng(SEED)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < UPDATES:
    for donor in rng.permutation(len(remaining)):
        while remaining[donor] and (int(donor), int(remaining[donor][-1])) in used_real:
            remaining[donor].pop()
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == UPDATES:
                break
assert len(set(schedule)) == UPDATES and not set(schedule) & used_real
RUN.mkdir(parents=True, exist_ok=False)
np.save(RUN / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': SEED, 'updates': UPDATES, 'synthetic_per_batch': SYNTHETIC, 'side': SIDE,
    'parent': str(PARENT), 'parent_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    'real_prior_sections_reused': 0,
    'real_schedule_sha256': hashlib.sha256((RUN / 'real_schedule.npy').read_bytes()).hexdigest(),
    'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
    'real_label_role': real['label_role'],
    'geometry_objective': 'best and soft-ranked 16-branch physical distance; synthetic 75% visible tissue and 25% full frame; weak real full frame',
    'fit_feedback_into_pose': False,
    'mapping_objective': 'true-pose atlas-conditioned local fit retained as same-model curriculum',
    'calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('train_one_shot_joint_physical_pose.py', 'arbitrary_plane_one_shot_model.py',
                     'arbitrary_plane_one_shot_stream.py')}}
(RUN / 'config.json').write_text(json.dumps(config, indent=2))
landmarks = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                          [127.5, 127.5]], device='cuda') / SIDE
flags = torch.tensor([0, 1], device='cuda', dtype=torch.long)


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               RUN / f'joint_step_{step:05d}.pt')


subjects_rng = torch.Generator().manual_seed(SEED)
draw_seed = SEED * 10000000
save(0)
started = time.perf_counter()
with (RUN / 'training.jsonl').open('w') as log, (RUN / 'draws.jsonl').open('w') as draws:
    for step in range(1, UPDATES + 1):
        accepted, pending = {}, list(range(SYNTHETIC))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=SIDE)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                                      ('inputs', 'state', 'reflection', 'offsets', 'weights',
                                       'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(SYNTHETIC)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        observation['inputs'] = F.interpolate(observation['inputs'], (SIDE, SIDE),
                                               mode='bilinear', align_corners=False)
        draws.write(json.dumps({**observation['identities'][0], 'step': step, 'slot': SYNTHETIC,
                                'used': True, 'label_role': real['label_role']}) + '\n')
        inputs = torch.cat((batch['inputs'], observation['inputs']))
        truth = torch.cat((batch['state'], observation['state']))
        reflection = torch.cat((batch['reflection'], observation['reflection']))
        prediction = model.predict(inputs)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        branch_flags = flags[None, None].expand(len(inputs), model.modes, -1)
        reference = points(truth, reflection, landmarks)
        five = (points(states, branch_flags, landmarks) - reference[:, None, None]).norm(dim=-1).mean(-1)
        valid = batch['valid_mask']
        indices = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
        targets = batch['centre'].reshape(SYNTHETIC, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(SIDE), indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
        dense = (points(states[:SYNTHETIC], branch_flags[:SYNTHETIC], chart[:, None, None])
                 - targets[:, None, None]).norm(dim=-1).mean(-1)
        distance = torch.cat((.75 * dense + .25 * five[:SYNTHETIC], five[SYNTHETIC:]), 0)
        prior = prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)
        flat_distance, flat_prior = distance.flatten(1), prior.flatten(1)
        direct = flat_distance.min(-1).values.mean() / 1000
        marginal = -torch.logsumexp(flat_prior - flat_distance / 1500, -1).mean()
        rank = F.kl_div(flat_prior, (-flat_distance.detach() / 1500).softmax(-1),
                        reduction='batchmean')
        mapping_prediction = {key: value[:SYNTHETIC] if torch.is_tensor(value) else value
                              for key, value in prediction.items()}
        true_state = mapping_prediction['state'].clone()
        true_state[:, 0] = batch['state']
        mapped = model.map({**mapping_prediction, 'state': true_state}, batch['offsets'],
                           torch.zeros((SYNTHETIC, 1), device='cuda', dtype=torch.long),
                           batch['reflection'][:, None], (SIDE, SIDE),
                           context['atlas'], batch['weights'])
        error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - batch['centre']).norm(dim=-1)
        mapping = (error * valid).sum() / valid.sum() / 500
        field = mapped['local_displacement_um'][:, 0]
        magnitude = field.square().mean().sqrt() / 1000
        smooth = ((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                  + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()) / 200
        loss = direct + .25 * marginal + .5 * rank + .5 * mapping + .1 * magnitude + .02 * smooth
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        selected = flat_prior.detach().argmax(-1)
        selected_error = flat_distance.detach().gather(1, selected[:, None])
        row = {'step': step, 'presentations': step * 4, 'direct_um': float(direct.detach() * 1000),
               'selected_synthetic_um': float(selected_error[:SYNTHETIC].mean()),
               'selected_real_weak_um': float(selected_error[-1]),
               'oracle_synthetic_um': float(flat_distance[:SYNTHETIC].detach().min(-1).values.mean()),
               'marginal': float(marginal.detach()), 'rank': float(rank.detach()),
               'true_pose_map_um': float(mapping.detach() * 500), 'gradient': float(gradient),
               'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush(); draws.flush()
            print(json.dumps(row), flush=True)
        if step % 2000 == 0:
            save(step)
(RUN / 'completed.json').write_text(json.dumps({'updates': UPDATES, 'presentations': UPDATES * 4,
    'new_synthetic_accepted': UPDATES * SYNTHETIC, 'new_distinct_real_train': UPDATES,
    'seconds': time.perf_counter() - started, 'fit_feedback_into_pose': False,
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': UPDATES}), flush=True)
