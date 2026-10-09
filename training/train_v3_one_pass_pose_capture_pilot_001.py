"""TRAIN-only v3 adaptation of the frozen 094 one-pass pose-candidate generator."""
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
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_ONE_PASS_POSE_CAPTURE_PILOT_001_PROTOCOL_20261009.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent = parent_run / 'joint_step_20000.pt'
run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
seed, updates, synthetic, side, sites = 20261009111, 8000, 2, 256, 128
checkpoints = (0, 2000, 4000, 8000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent) == parent_receipt['checkpoint_sha256']['20000'] == \
    '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))

context = load_streaming_synthetic_v7_64(device='cuda')
real = load_reserved_real_train()
rng = np.random.default_rng(seed)
remaining = [rng.permutation(len(donor['identities'])).tolist() for donor in real['donors']]
schedule = []
while len(schedule) < updates:
    for donor in rng.permutation(len(remaining)):
        donor = int(donor)
        if not remaining[donor]:
            remaining[donor] = rng.permutation(len(real['donors'][donor]['identities'])).tolist()
        schedule.append((donor, int(remaining[donor].pop())))
        if len(schedule) == updates:
            break

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 20000 and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
teacher = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
teacher.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
heads = (model.pose, model.anchor_pose, model.anchor_global)
shared = (model.encoder, model.lateral)
for module in heads:
    module.requires_grad_(True)
groups = [{'params': list(module.parameters()), 'lr': 2e-5, 'base_lr': 2e-5}
          for module in heads] + [
    {'params': list(module.parameters()), 'lr': 0., 'base_lr': 2e-6}
    for module in shared]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
parameters = [parameter for group in groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
flags = torch.tensor([0, 1], device='cuda')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
source_files = (
    'train_v3_one_pass_pose_capture_pilot_001.py',
    'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py',
)
config = {'seed': seed, 'updates': updates, 'synthetic_per_update': synthetic,
    'real_per_update': 1, 'side': side, 'sampled_valid_pixels': sites,
    'checkpoints': checkpoints, 'parent_094': str(parent),
    'parent_094_sha256': sha(parent),
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'synthetic_label_role': 'source cutting plane plus observed surviving-pixel CCF coordinates',
    'real_label_role': real['label_role'], 'real_bindings': real['bindings'],
    'real_schedule': 'donor-uniform permuted cycles; shuffled within-donor sections',
    'real_loss_weight': .35, 'real_retention_margin_mm': .05,
    'real_retention_baseline': 'frozen 094 probability-weighted weak-affine five-point error',
    'trainable_head_updates': [1, 2000],
    'trainable_shared_updates': [2001, 8000],
    'frozen': '094 atlas matcher/fitter, warp/map and candidate-scoring heads',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'real_schedule.npy', np.asarray(schedule, dtype=np.int32))
config['real_schedule_sha256'] = sha(run / 'real_schedule.npy')
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


save(0)
attempts = 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        if step == 2001:
            for module in shared:
                module.requires_grad_(True)
        accepted, pending = {}, list(range(synthetic))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                                    generator=subject_rng).tolist()
            with torch.no_grad():
                sampled = sample_one_shot_slide_artifacts_v3(
                    context, virtual, draw_seed, side=side)
            for row, record in enumerate(sampled['provenance']):
                slot = pending[row]
                used = bool(sampled['eligible'][row])
                attempts += 1
                draws.write(json.dumps({'kind': 'synthetic_train', **record,
                    'update': step, 'slot': slot, 'draw_attempt': attempts, 'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
                    accepted[slot]['record'] = record
            pending = [slot for slot in pending if slot not in accepted]
            draw_seed += 1
        batch = {key: torch.cat([accepted[slot][key] for slot in range(synthetic)])
                 for key in ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
        donor, section = schedule[step - 1]
        observation = sample_reserved_real_train(real, donor, [section], device='cuda')
        draws.write(json.dumps({'kind': 'real_train', **observation['identities'][0],
            'update': step, 'slot': synthetic, 'used': True,
            'label_role': real['label_role']}) + '\n')
        real_image = F.interpolate(observation['inputs'], (side, side),
                                   mode='bilinear', align_corners=False)
        prediction = model.predict(torch.cat((batch['inputs'], real_image)))
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags[None, None].expand(len(states), model.modes, 2)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

        pixel = torch.multinomial(batch['valid_mask'].flatten(1).float(),
                                  sites, replacement=True)
        target = batch['centre'].reshape(synthetic, -1, 3).gather(
            1, pixel[..., None].expand(-1, -1, 3))
        chart = torch.stack((pixel.remainder(side),
                             pixel.div(side, rounding_mode='floor')), -1).float() / side
        source_five = rigid_points_090(batch['state'], batch['reflection'], corners)
        synthetic_five = (rigid_points_090(states[:synthetic],
            reflections[:synthetic], corners) - source_five[:, None, None]
            ).norm(dim=-1).mean(-1)
        synthetic_dense = (rigid_points_090(states[:synthetic],
            reflections[:synthetic], chart[:, None, None]) - target[:, None, None]
            ).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(batch['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal[:synthetic] * true_normal[:, None])
                                  .sum(-1).abs().clamp_max(1))
        synthetic_cost = (.75 * synthetic_dense + .25 * synthetic_five
                          + normal_penalty[..., None]).flatten(1) / 1000
        synthetic_target = F.softmax(-synthetic_cost.detach() / .7, -1)
        synthetic_kl = F.kl_div(prior[:synthetic], synthetic_target,
                                reduction='none').sum(-1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = synthetic_cost.reshape(synthetic, model.modes, 2).min(-1).values.gather(1, near_mode)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        synthetic_loss = (-1.5 * torch.logsumexp(prior[:synthetic] - synthetic_cost / 1.5, -1)
            + .5 * synthetic_kl
            + .5 * synthetic_cost[:, :2 * model.base_modes].min(-1).values
            + (neighbourhood * near_distance).sum(-1)).mean()

        real_five = rigid_points_090(observation['state'],
                                     observation['reflection'], corners)
        real_cost = (rigid_points_090(states[synthetic:],
            reflections[synthetic:], corners) - real_five[:, None, None]
            ).norm(dim=-1).mean(-1).flatten(1) / 1000
        real_expected = (F.softmax(prior[synthetic:] / .3, -1) * real_cost).sum(-1)
        with torch.no_grad():
            inherited = teacher.predict(real_image)
            inherited_states = inherited['state'][:, :, None].expand(-1, -1, 2, -1)
            inherited_prior = (inherited['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-inherited['reflection_logit']),
                F.logsigmoid(inherited['reflection_logit'])), -1)).flatten(1)
            inherited_cost = (rigid_points_090(inherited_states, reflections[synthetic:],
                corners) - real_five[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            inherited_expected = (F.softmax(inherited_prior / .3, -1) * inherited_cost).sum(-1)
        real_loss = F.relu(real_expected - inherited_expected - .05).mean()
        loss = synthetic_loss + .35 * real_loss

        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = group['base_lr'] * decay if index < len(heads) or step > 2000 else 0.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'stage': 'heads' if step <= 2000 else 'shared',
            'synthetic_section_ids': [accepted[slot]['record']['physical_section_id']
                                      for slot in range(synthetic)],
            'real_identity': observation['identities'][0], 'draw_attempts': attempts,
            'synthetic_source_five_mm': float(synthetic_five.detach().min(-1).values.mean() / 1000),
            'synthetic_observed_dense_mm': float(synthetic_dense.detach().min(-1).values.mean() / 1000),
            'synthetic_best160_cost_mm': float(synthetic_cost.detach().min(-1).values.mean()),
            'synthetic_prior_cost_mm': float(synthetic_cost.detach().gather(
                1, prior[:synthetic].detach().argmax(-1)[:, None]).mean()),
            'real_weak_prior_five_mm': float(real_cost.detach().gather(
                1, prior[synthetic:].detach().argmax(-1)[:, None]).mean()),
            'real_weak_weighted_five_mm': float(real_expected.detach().mean()),
            'real_weak_inherited_weighted_five_mm': float(inherited_expected.mean()),
            'synthetic_loss': float(synthetic_loss.detach()),
            'real_weak_loss': float(real_loss.detach()),
            'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 2000 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('update', 'stage',
                'synthetic_best160_cost_mm', 'synthetic_prior_cost_mm',
                'real_weak_prior_five_mm', 'total_loss', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates * synthetic,
    'synthetic_draw_attempts': attempts, 'real_train_presentations': updates,
    'real_label_role': real['label_role'], 'protocol_sha256': sha(protocol),
    'parent_094_checkpoint_sha256': sha(parent),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'real_schedule_sha256': sha(run / 'real_schedule.npy'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
                          for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'accepted_synthetic': updates * synthetic,
    'seconds': time.perf_counter() - started}), flush=True)
