"""Train global 2D/3D context and safe pose gating on fresh synthetic and real TRAIN."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.pose_feedback_global_041 import PoseFeedbackGlobal041

parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
matcher_parent = root / 'runs/pose_feedback_3d_039_pilot_v2/joint_step_00500.pt'
run = root / 'runs/pose_feedback_global_041_pilot'
seed, updates, synthetic, side, beam = 2026104100, 2000, 2, 256, 4
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < updates:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == updates:
                break
assert len(set(schedule)) == updates
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = {'seed': seed, 'updates': updates, 'synthetic_per_batch': synthetic,
          'real_weak_per_batch': 1, 'parent': str(parent), 'parent_sha256': sha(parent),
          'matcher_parent': str(matcher_parent), 'matcher_parent_sha256': sha(matcher_parent),
          'prior_beam_training': beam, 'real_schedule_sha256': sha(run / 'real_schedule.npy'),
          'real_section_policy': 'without replacement within 041; reuse across experiments allowed',
          'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
          'real_label_role': real['label_role'],
          'trainable': 'scratch-lineage 3D matcher plus new global image/atlas context and safe correction gate; 019 frozen',
          'real_role': 'weak-affine pose/gate supervision, no expert-oblique claim',
          'calibrated': False, 'public_benchmark_used': False,
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('train_pose_feedback_global_041.py', 'pose_feedback_global_041.py',
                             'pose_feedback_3d_039.py', 'arbitrary_plane_one_shot_model.py',
                             'arbitrary_plane_one_shot_stream.py',
                             'arbitrary_plane_full_frame_primitives.py',
                             'arbitrary_plane_streaming_synthetic_v7_64.py',
                             'arbitrary_plane_reserved_real_stream_v8.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 10000000
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
head = PoseFeedbackGlobal041().cuda().train()
old = torch.load(matcher_parent, map_location='cpu', weights_only=True)['head']
current = head.state_dict()
for name, value in old.items():
    mapped = ('query.local.' + name[len('query.'):] if name.startswith('query.') else
              'key_3d.local.' + name[len('key_3d.'):] if name.startswith('key_3d.') else name)
    current[mapped] = value
head.load_state_dict(current)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
perturb_scale = torch.tensor((.16, .16, .16, 1400., 1400., 1400., .10, .10, .08), device='cuda')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkpj->bkpi',
        frame[..., :2] @ basis, chart - .5)


def save(step):
    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                'subjects_rng': subjects_rng.get_state(), 'draw_seed': draw_seed,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'step': step, 'config': config, 'calibrated': False},
               run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),), generator=subjects_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
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
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
                                'slot': synthetic, 'used': True,
                                'label_role': real['label_role']}) + '\n')
        real_image = F.interpolate(observation['inputs'], (side, side),
                                   mode='bilinear', align_corners=False)
        real_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * observation['thickness_um'][:, None]
        real_weights = torch.ones_like(real_offsets)
        real_weights[:, [0, -1]] = .5
        real_weights /= real_weights.sum(-1, keepdim=True)
        images = torch.cat((batch['inputs'], real_image))
        offsets = torch.cat((batch['offsets'], real_offsets))
        weights = torch.cat((batch['weights'], real_weights))
        truth_state = torch.cat((batch['state'], observation['state']))
        truth_reflection = torch.cat((batch['reflection'], observation['reflection']))
        with torch.no_grad():
            prediction = model.predict(images)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior.topk(beam, -1).indices
            chosen = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            perturb = torch.randn((synthetic + 1, 2, 9), device='cuda') * perturb_scale
            teacher = compose_full_frame_state(truth_state[:, None].expand(-1, 3, -1),
                torch.cat((perturb, perturb.new_zeros(synthetic + 1, 1, 9)), 1))
            initial = torch.cat((chosen, teacher), 1)
            reflection = torch.cat((choice % 2, truth_reflection[:, None].expand(-1, 3)), 1)
        result = head(prediction, images, initial, reflection, context['atlas'], offsets, weights)
        truth_grid = batch['centre'][:, 8::16, 8::16].reshape(synthetic, 256, 3)
        valid_grid = batch['valid_mask'][:, 8::16, 8::16].reshape(synthetic, 256).bool()
        with torch.no_grad():
            distances = torch.cdist(truth_grid[:, None].expand(-1, beam + 3, -1, -1)
                .reshape(synthetic * (beam + 3), 256, 3),
                result['key_world'][:synthetic].reshape(synthetic * (beam + 3), -1, 3))
            nearest_distance, nearest = distances.min(-1)
            nearest = nearest.reshape(synthetic, beam + 3, 256)
            nearest_distance = nearest_distance.reshape(synthetic, beam + 3, 256)
            support = result['key_support'][:synthetic].gather(-1, nearest)
            in_range = valid_grid[:, None] & (nearest_distance <= 1500) & (support > .1)
            ids = torch.multinomial(batch['valid_mask'].flatten(1).float(), 128, replacement=True)
            truth128 = batch['centre'].reshape(synthetic, -1, 3).gather(
                1, ids[..., None].expand(-1, -1, 3))
            chart128 = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                                    -1).float() / side
        ce = F.cross_entropy(result['logits'][:synthetic].reshape(-1, result['logits'].shape[-1]),
                             nearest.flatten(), reduction='none').reshape_as(in_range)
        correspondence_loss = (ce * in_range).sum() / in_range.sum().clamp_min(1)
        expected_error = F.smooth_l1_loss(result['matched_world'][:synthetic] / 1000,
            truth_grid[:, None].expand(-1, beam + 3, -1, -1) / 1000,
            beta=.5, reduction='none').sum(-1)
        expected_loss = (expected_error * in_range).sum() / in_range.sum().clamp_min(1)
        synthetic_pose = (points(result['state'][:synthetic], reflection[:synthetic], chart128)
                          - truth128[:, None]).norm(dim=-1).mean(-1)
        raw_synthetic = (points(result['raw_fitted_state'][:synthetic],
            reflection[:synthetic], chart128) - truth128[:, None]).norm(dim=-1).mean(-1)
        parent_synthetic = (points(initial[:synthetic], reflection[:synthetic], chart128)
                            - truth128[:, None]).norm(dim=-1).mean(-1)
        synthetic_loss = F.smooth_l1_loss(synthetic_pose / 1000,
            torch.zeros_like(synthetic_pose), beta=1., reduction='none')
        synthetic_loss = synthetic_loss[:, :beam].mean() + .5 * synthetic_loss[:, beam:].mean()
        real_reference = points(truth_state[synthetic:, None],
                                truth_reflection[synthetic:, None], corners[None])[:, 0]
        real_pose = (points(result['state'][synthetic:], reflection[synthetic:], corners[None])
                     - real_reference[:, None]).norm(dim=-1).mean(-1)
        raw_real = (points(result['raw_fitted_state'][synthetic:],
            reflection[synthetic:], corners[None]) - real_reference[:, None]).norm(dim=-1).mean(-1)
        parent_real = (points(initial[synthetic:], reflection[synthetic:], corners[None])
                       - real_reference[:, None]).norm(dim=-1).mean(-1)
        real_loss = F.smooth_l1_loss(real_pose / 1000,
            torch.zeros_like(real_pose), beta=1., reduction='none')
        real_loss = real_loss[:, :beam].mean() + .5 * real_loss[:, beam:].mean()
        gate_target = torch.cat(((raw_synthetic < parent_synthetic - 50).float(),
                                 (raw_real < parent_real - 50).float()))
        gate_loss = F.binary_cross_entropy(result['correction_gate'], gate_target)
        visibility_loss = F.binary_cross_entropy_with_logits(
            result['visibility_logit'][:synthetic], valid_grid.float())
        score = prior.gather(1, choice) + result['score_delta'][:, :beam]
        physical = torch.cat((synthetic_pose[:, :beam], real_pose[:, :beam]))
        ranking_loss = F.kl_div(F.log_softmax(score, -1),
            F.softmax(-physical.detach() / 750, -1), reduction='none').sum(-1).mean()
        loss = synthetic_loss + .15 * correspondence_loss + .25 * expected_loss \
               + .1 * visibility_loss + .25 * ranking_loss + .25 * gate_loss + .25 * real_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'synthetic_presentations': step * synthetic,
               'real_weak_presentations': step, 'loss': float(loss.detach()),
               'synthetic_pose_loss': float(synthetic_loss.detach()),
               'correspondence_loss': float(correspondence_loss.detach()),
               'expected_loss': float(expected_loss.detach()),
               'real_weak_loss': float(real_loss.detach()),
               'gate_loss': float(gate_loss.detach()),
               'ranking_loss': float(ranking_loss.detach()),
               'synthetic_in_range_fraction': float(in_range.float().mean()),
               'synthetic_top1_um': float(synthetic_pose[:, 0].detach().mean()),
               'synthetic_best4_um': float(synthetic_pose[:, :beam].detach().min(-1).values.mean()),
               'real_weak_top1_um': float(real_pose[:, 0].detach().mean()),
               'synthetic_gate': float(result['correction_gate'][:synthetic].detach().mean()),
               'real_weak_gate': float(result['correction_gate'][synthetic:].detach().mean()),
               'gradient_norm': float(gradient), 'seconds': time.perf_counter() - started}
        log.write(json.dumps(row) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps(row), flush=True)
            save(step)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': updates * synthetic, 'unique_real_weak': updates,
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'), 'config_sha256': sha(run / 'config.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
