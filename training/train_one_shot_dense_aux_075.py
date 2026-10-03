"""Train the existing direct pose model with high-resolution tissue-coordinate supervision."""
import hashlib
import json
import math
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
run = root / 'runs/one_shot_dense_aux_075'
seed, batches, synthetic, side, field_side = 2026100375, 10000, 2, 256, 128
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < batches:
    for donor in rng.permutation(len(remaining)):
        if remaining[donor]:
            schedule.append((int(donor), int(remaining[donor].pop())))
            if len(schedule) == batches:
                break
assert len(set(schedule)) == batches
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config = {'seed': seed, 'batches': batches, 'synthetic_per_batch': synthetic,
    'real_per_batch': 1, 'side': side, 'field_side': field_side,
    'parent': str(parent), 'parent_sha256': sha(parent),
    'real_schedule_sha256': sha(run / 'real_schedule.npy'),
    'unique_real_within_run': True, 'random_reuse_from_prior_training_runs_allowed': True,
    'synthetic_provenance': context['provenance'], 'real_bindings': real['bindings'],
    'real_label_role': real['label_role'],
    'change': 'end-to-end high-resolution tissue-coordinate auxiliary on the existing direct probabilistic pose backbone',
    'synthetic_field_target': 'known surviving tissue-to-CCF map, not an injected inference candidate',
    'real_field_target': 'weak Allen affine, down-weighted; not expert ground truth',
    'mapper_and_fitter_frozen': True, 'fit_to_pose_feedback_in_this_stage': False,
    'calibrated': False, 'public_benchmark_used': False,
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
        ('train_one_shot_dense_aux_075.py', 'arbitrary_plane_one_shot_model.py',
         'arbitrary_plane_one_shot_stream.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
         'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_reserved_real_stream_v8.py')}}
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
saved = torch.load(parent, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True,
    dense_coordinate_pyramid=True).cuda().train()
missing, unexpected = model.load_state_dict(saved['model'], strict=False)
assert set(missing) == {f'dense_coordinate_pyramid_head.{name}'
    for name in model.dense_coordinate_pyramid_head.state_dict()} and not unexpected
del saved
model.requires_grad_(False)
modules = (model.encoder, model.pose, model.anchor_pose, model.anchor_global,
           model.lateral, model.warp_shared, model.dense_coordinate_pyramid_head)
rates = (2e-5, 3e-5, 5e-5, 5e-5, 2e-5, 2e-5, 1e-4)
for module in modules:
    module.requires_grad_(True)
optimizer = torch.optim.AdamW([{'params': module.parameters(), 'lr': rate}
    for module, rate in zip(modules, rates)], weight_decay=1e-4)
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
axis = (torch.arange(field_side, device='cuda').float() + .5) / field_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
field_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
flags = torch.tensor([0, 1], device='cuda')


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    observed = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    observed[..., 0] = torch.where(reflection[..., None].bool(),
                                    255 / side - observed[..., 0], observed[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, observed - .5)


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


save(0)
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, batches + 1):
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subject_rng).tolist()
            sampled = sample_one_shot_stream(context, virtual, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sampled['provenance']):
                slot = pending[row]
                eligible = bool(sampled['eligible'][row])
                draws.write(json.dumps({**identity, 'step': step, 'slot': slot,
                                        'used': eligible}) + '\n')
                if eligible:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in accepted[0]}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({**observation['identities'][0], 'step': step,
            'slot': synthetic, 'used': True, 'label_role': real['label_role']}) + '\n')
        real_image = F.interpolate(observation['inputs'], (side, side),
                                   mode='bilinear', align_corners=False)
        image = torch.cat((batch['inputs'], real_image))
        truth = torch.cat((batch['state'], observation['state']))
        true_reflection = torch.cat((batch['reflection'], observation['reflection']))
        prediction = model.predict(image)

        state = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        branch_flags = flags[None, None].expand(len(image), model.modes, 2)
        five_target = points(truth, true_reflection, corners)
        five = (points(state, branch_flags, corners) - five_target[:, None, None]).norm(dim=-1).mean(-1)
        indices = torch.multinomial(batch['valid_mask'].flatten(1).float(), 96, replacement=True)
        tissue = batch['centre'].reshape(synthetic, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side),
            indices.div(side, rounding_mode='floor')), -1).float() / side
        dense = (points(state[:synthetic], branch_flags[:synthetic], chart[:, None, None])
                 - tissue[:, None, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][:, :, 2]
        normal_penalty = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
        distance = torch.cat((.75 * dense + .25 * five[:synthetic]
                              + normal_penalty[:synthetic, :, None],
                              five[synthetic:] + .25 * normal_penalty[synthetic:, :, None]), 0)
        similarity = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs()
        nearest = similarity.topk(4, -1)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = distance.min(-1).values.gather(1, near_mode)
        old_best = distance[:, :model.base_modes].flatten(1).min(-1).values
        geometry = ((neighbourhood * near_distance).sum(-1) + .5 * old_best).mean() / 1000
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        rank_target = F.softmax(-distance.flatten(1).detach() / 700, -1)
        rank = -(rank_target * prior).sum(-1).mean()

        field = prediction['dense_coordinate']
        synthetic_target = F.interpolate(batch['centre'].permute(0, 3, 1, 2),
            (field_side, field_side), mode='bilinear', align_corners=False)
        real_target = points(observation['state'], observation['reflection'],
                             field_chart).reshape(1, field_side, field_side, 3).permute(0, 3, 1, 2)
        target = torch.cat((synthetic_target, real_target))
        target = (target - model.center_origin[None, :, None, None]) / model.center_scale[None, :, None, None]
        field_error = F.smooth_l1_loss(field[:, :3], target, beta=.05,
                                       reduction='none').sum(1, keepdim=True)
        occupancy = F.avg_pool2d(batch['valid_mask'][:, None].float(), 2)
        synthetic_field = ((field_error[:synthetic] * occupancy).flatten(1).sum(-1)
                           / occupancy.flatten(1).sum(-1).clamp_min(1)).mean()
        real_field = field_error[synthetic:].mean()
        validity = F.binary_cross_entropy_with_logits(field[:synthetic, 3:4], occupancy)
        loss = geometry + .25 * rank + 2 * synthetic_field + .5 * real_field + .1 * validity
        decay = .15 + .85 * .5 * (1 + math.cos(math.pi * step / batches))
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'batch': step, 'synthetic_presentations': step * synthetic,
                'distinct_real_train_presentations': step,
                'geometry': float(geometry.detach()), 'rank': float(rank.detach()),
                'synthetic_field': float(synthetic_field.detach()),
                'real_weak_field': float(real_field.detach()),
                'validity': float(validity.detach()), 'gradient_norm': float(clipped),
                'seconds': time.perf_counter() - started}
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 2000 == 0:
                log.flush()
                draws.flush()
                print(json.dumps(row), flush=True)
        if step in (2000, 5000, batches):
            save(step)
    log.flush()
    draws.flush()
(run / 'completed.json').write_text(json.dumps({'batches': batches,
    'accepted_synthetic': batches * synthetic, 'distinct_real_train': batches,
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'config_sha256': sha(run / 'config.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': batches,
                  'seconds': time.perf_counter() - started}), flush=True)
