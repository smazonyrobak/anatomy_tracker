"""Train the coherent anatomical field attached to the scratch 132 pose lineage."""

import hashlib
import json
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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import dense_local_targets_143, render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COHERENT_ANATOMY_FIELD_143_PROTOCOL_20261010.md'
parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
parent_done = root / 'runs/v4_pose_adaptation_132/completed.json'
run = root / 'runs/coherent_anatomy_field_143'
seed, side, radius, steps = 20261010143, 256, 6, 10000
arms = ('full', 'support_only')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert not run.exists() and root.drive == source.drive == run.drive == 'I:'
assert sha(parent) == json.loads(parent_done.read_text())['checkpoint_sha256']['2000']
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
fields = {name: CoherentAnatomyField143(radius=radius).cuda().train() for name in arms}
fields['support_only'].load_state_dict(fields['full'].state_dict())
optimizers = {name: torch.optim.AdamW(field.parameters(), lr=2e-4, weight_decay=1e-4)
              for name, field in fields.items()}
scalers = {name: torch.amp.GradScaler('cuda', init_scale=256., growth_interval=4001)
           for name in arms}
source_names = ('train_coherent_anatomy_field_143.py', 'coherent_anatomy_field_143.py',
    'coherent_anatomy_geometry_143.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'steps': steps, 'side': side, 'radius': radius,
    'slab_centres_um': [-1500, -1000, -500, 0, 500, 1000, 1500],
    'synthetic_per_step': {'v4': 2, 'v3': 1},
    'presentations': steps * 3, 'checkpoints': [0, 2000, 5000, 10000],
    'parent_checkpoint_sha256': sha(parent), 'parent_completion_sha256': sha(parent_done),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'training_stage': 'field supervised exact/near/wrong candidate curriculum; parent frozen until discriminative gate',
    'arms': arms, 'same_initial_weights': True,
    'loss_weights': {'reachable_ce': 1., 'valid_unreachable_ce': .3,
                     'background_ce': .05, 'offset_regression': .2,
                     'source_visibility': .1, 'support_matched_energy_rank': .5},
    'wrong_candidate': 'mean rigid target 2.2mm, drawn independently per section; support difference <=0.10 for energy ranking',
    'real_weak_train_used': False, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    for name in arms:
        path = run / f'{name}_step_{step:05d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save({'step': step, 'arm': name, 'field': fields[name].state_dict(),
            'optimizer': optimizers[name].state_dict(), 'scaler': scalers[name].state_dict(),
            'config': config, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
        os.replace(temporary, path)


grid_axis = (torch.arange(64, device='cuda') + .5) / 64
gy, gx = torch.meshgrid(grid_axis, grid_axis, indexing='ij')
chart = torch.stack((gx, gy), -1)
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')


def candidates(sample):
    valid = sample['valid_mask'][0, 2::4, 2::4]
    points = chart[valid]
    truth = rigid_points_090(sample['state'], sample['reflection'], points)[0]
    proposals = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
    near_states = compose_full_frame_state(sample['state'].expand(64, -1), proposals)
    near_error = (rigid_points_090(near_states,
        sample['reflection'].expand(64), points) - truth).norm(dim=-1).mean(-1) / 1000
    near_eligible = (near_error >= .35) & (near_error <= 1.5)
    near_choice = int((near_error - .9).abs().masked_fill(~near_eligible, 10).argmin()
                      if bool(near_eligible.any()) else (near_error - .9).abs().argmin())
    wrong_update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
    wrong_update[:, :3] *= 3
    wrong_update[:, 3:6] *= 3
    wrong_update[:, 6:] *= 2
    wrong_states = compose_full_frame_state(sample['state'].expand(64, -1), wrong_update)
    wrong_error = (rigid_points_090(wrong_states,
        sample['reflection'].expand(64), points) - truth).norm(dim=-1).mean(-1) / 1000
    wrong_eligible = (wrong_error >= 1.5) & (wrong_error <= 3.0)
    wrong_choice = int((wrong_error - 2.2).abs().masked_fill(~wrong_eligible, 10).argmin()
                       if bool(wrong_eligible.any()) else (wrong_error - 2.2).abs().argmin())
    states = torch.cat((sample['state'], near_states[near_choice:near_choice + 1],
                        wrong_states[wrong_choice:wrong_choice + 1]))
    return states, float(near_error[near_choice]), float(wrong_error[wrong_choice])


def field_loss(output, local, labels, reachable, valid, rank_allowed):
    logits = output['match_logits'].float()
    dustbin = logits.shape[1] - 1
    target = torch.where(reachable, labels, dustbin)
    cross_entropy = F.cross_entropy(logits, target, reduction='none')
    hit = cross_entropy[reachable].mean() if bool(reachable.any()) else cross_entropy.sum() * 0
    miss_mask = valid & ~reachable
    miss = cross_entropy[miss_mask].mean() if bool(miss_mask.any()) else cross_entropy.sum() * 0
    background = cross_entropy[~valid].mean() if bool((~valid).any()) else cross_entropy.sum() * 0
    estimate = output['offset_chart'].permute(0, 2, 3, 1).float()
    scaled = torch.cat(((estimate[..., :2] - local[..., :2]) * 64,
                        (estimate[..., 2:3] - local[..., 2:3]) / 500), -1)
    displacement = (F.smooth_l1_loss(scaled[reachable], torch.zeros_like(scaled[reachable]))
                    if bool(reachable.any()) else scaled.sum() * 0)
    visible = F.binary_cross_entropy(output['visibility'][:, 0].float(), valid.float())
    rank = F.softplus(5 * (.15 + output['energy'][0] - output['energy'][2])) / 5
    loss = hit + .3 * miss + .05 * background + .2 * displacement + .1 * visible
    if rank_allowed:
        loss = loss + .5 * rank
    return loss, hit, displacement, visible, rank


save(0)
started, accepted, attempted = time.perf_counter(), 0, 0
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as logs:
    for step in range(1, steps + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        totals = {name: {'loss': 0., 'hit_ce': 0., 'offset': 0., 'visibility': 0.,
                         'rank': 0., 'reachable': 0., 'valid': 0., 'ranked': 0}
                  for name in arms}
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v3
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                attempted += 1
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'step': step, 'slot': slot, 'appearance': appearance,
                    'attempt': attempted, 'used': used, **sample['provenance'][0]},
                    allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    break
            with torch.no_grad():
                prediction = model.predict(sample['inputs'])
                states, near_mm, wrong_mm = candidates(sample)
                reflection = sample['reflection'].expand(3)
                slabs, base, basis, normal = render_coherent_atlas_143(context['atlas'],
                    states, reflection, sample['offsets'].expand(3, -1),
                    sample['weights'].expand(3, -1))
                local, classes, reachable = dense_local_targets_143(
                    sample['centre'].expand(3, -1, -1, -1),
                    sample['valid_mask'].expand(3, -1, -1), base, basis, normal,
                    radius=radius)
                valid = sample['valid_mask'][:, 2::4, 2::4].expand(3, -1, -1)
                lattice = torch.stack((local[..., 0] * 64, local[..., 1] * 64,
                    local[..., 2] / 500), -1).round().long()
                yy, xx = torch.meshgrid(torch.arange(64, device='cuda'),
                                        torch.arange(64, device='cuda'), indexing='ij')
                depth = (lattice[..., 2] + 3).clamp(0, 6)
                row = (yy + lattice[..., 1]).clamp(0, 63)
                column = (xx + lattice[..., 0]).clamp(0, 63)
                key_support = slabs[:, 1][torch.arange(3, device='cuda')[:, None, None],
                                         depth, row, column] > .5
                reachable &= key_support
                source = sample['inputs'].expand(3, -1, -1, -1)
                feature = prediction['feature'].expand(3, -1, -1, -1)
                support_fraction = slabs[:, 1, 3].mean((1, 2))
                rank_allowed = bool((support_fraction[0] - support_fraction[2]).abs() <= .1)
            for name in arms:
                pair = slabs if name == 'full' else torch.cat((torch.zeros_like(slabs[:, :1]),
                                                               slabs[:, 1:2]), 1)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = fields[name](source, pair, 500., source_feature=feature)
                loss, hit, displacement, visible, rank = field_loss(
                    output, local, classes, reachable, valid, rank_allowed)
                scalers[name].scale(loss / 3).backward()
                row = totals[name]
                row['loss'] += float(loss.detach()) / 3
                row['hit_ce'] += float(hit.detach()) / 3
                row['offset'] += float(displacement.detach()) / 3
                row['visibility'] += float(visible.detach()) / 3
                row['rank'] += float(rank.detach()) / 3
                row['reachable'] += int(reachable.sum())
                row['valid'] += int(valid.sum())
                row['ranked'] += int(rank_allowed)
            del output, prediction, slabs, sample
        for name in arms:
            scalers[name].unscale_(optimizers[name])
            gradient = torch.nn.utils.clip_grad_norm_(fields[name].parameters(), 2.)
            assert torch.isfinite(gradient)
            scalers[name].step(optimizers[name])
            scalers[name].update()
        row = {'step': step, 'accepted': accepted, 'attempted': attempted,
               'elapsed_seconds': time.perf_counter() - started,
               'arms': totals, 'last_near_mm': near_mm, 'last_wrong_mm': wrong_mm}
        logs.write(json.dumps(row, allow_nan=False) + '\n')
        if step % 1000 == 0:
            logs.flush()
            draws.flush()
            print(json.dumps({'event': 'milestone', **row}), flush=True)
        if step in config['checkpoints']:
            save(step)
(run / 'completed.json').write_text(json.dumps({'steps': steps,
    'accepted_synthetic_physical_sections': accepted, 'attempted': attempted,
    'v4_presentations': 2 * steps, 'v3_presentations': steps,
    'config_sha256': sha(run / 'config.json'), 'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {name: {str(number): sha(run / f'{name}_step_{number:05d}.pt')
        for number in config['checkpoints']} for name in arms},
    'source_sha256': config['source_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print('143 coherent field stage complete; independent DEV readout required', flush=True)
