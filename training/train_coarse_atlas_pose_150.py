"""Scratch-train contextual source/atlas matches and independently supervised fit weights."""

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
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.coarse_atlas_pose_150 import CoarseAtlasPose150
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate(sample, lower, upper, target, wrong=False):
    points = chart64[sample['valid_mask'][0, 2::4, 2::4]]
    reference = rigid_points_090(sample['state'], sample['reflection'], points)
    while True:
        update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
        if wrong:
            update[:, :6] *= 3
            update[:, 6:] *= 2
        states = compose_full_frame_state(sample['state'].expand(64, -1), update)
        error = (rigid_points_090(states, sample['reflection'].expand(64), points)
                 - reference).norm(dim=-1).mean(-1) / 1000
        selected = (error >= lower) & (error <= upper)
        if bool(selected.any()):
            index = int((error - target).abs().masked_fill(~selected, 10).argmin())
            return states[index:index + 1], float(error[index])


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_150_PROTOCOL_20261010.md'
run = root / 'runs/coarse_atlas_pose_150'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent = parent_dir / 'joint_step_02000.pt'
seed, side, steps = 20261010150, 256, 6000
arms, checkpoints = ('full', 'support_only'), (0, 1000, 3000, 6000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive == source.drive == run.drive == 'I:' and not run.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['2000']
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
assert all(base['lineage']['split'] == 'train' for base in context['bases'])

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
matchers = {name: CoarseAtlasPose150().cuda().train() for name in arms}
matchers['support_only'].load_state_dict(matchers['full'].state_dict())
optimizers = {name: torch.optim.AdamW(matchers[name].parameters(), lr=2e-4,
    weight_decay=1e-4) for name in arms}
scalers = {name: torch.amp.GradScaler('cuda', init_scale=256., growth_interval=4001)
    for name in arms}
source_names = ('train_coarse_atlas_pose_150.py', 'coarse_atlas_pose_150.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py')
config = {'seed': seed, 'steps': steps, 'side': side, 'arms': arms,
    'checkpoints': checkpoints, 'synthetic_per_step': {'v4': 2, 'v3': 1},
    'gradient_train_bases': list(range(60)), 'inner_checkpoint_bases': list(range(60, 64)),
    'inner_cases_per_base': 4,
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'candidate_error_mm': {'exact': 0, 'near': [.35, 1.5], 'wrong': [1.5, 3.]},
    'soft_positive_sigma_um': 450, 'soft_positive_limit_um': 1200,
    'reachable_limit_um': 900, 'reliability_error_sigma_mm': .75,
    'loss_weights': {'soft_key_ce': 1., 'expected_ccf_huber': .5,
                     'corrected_plane_huber': 1., 'visibility_bce': .2,
                     'reliability_bce': .7},
    'checkpoint_selection': 'lowest full-arm inner-plan sum of exact and near rigid mm; same step for both arms',
    'pose_unfrozen': False, 'joint_feedback_trained': False,
    'real_weak_train_used': False, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))

limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')
axis64 = (torch.arange(64, device='cuda') + .5) / 64
cy64, cx64 = torch.meshgrid(axis64, axis64, indexing='ij')
chart64 = torch.stack((cx64, cy64), -1)
axis24 = (torch.arange(24, device='cuda') + .5) / 24
cy24, cx24 = torch.meshgrid(axis24, axis24, indexing='ij')
grid24 = (2 * torch.stack((cx24, cy24), -1) - 1)[None]
axis256 = torch.arange(side, device='cuda') / side
cy256, cx256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((cx256, cy256), -1)

# These 16 fixed cases use four entirely withheld synthetic TRAIN deformation bases.
inner = []
inner_records = []
inner_rng = np.random.default_rng(seed + 1)
for base in range(60, 64):
    for slot in range(4):
        attempt = 0
        while True:
            virtual = base * 64 + int(inner_rng.integers(64))
            draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
            with torch.no_grad():
                sample = sample_one_shot_slide_artifacts_v4(context, [virtual], draw_seed, side=side)
            attempt += 1
            used = bool(sample['eligible'][0]) and bool((F.grid_sample(
                sample['valid_mask'][:, None].float(), grid24,
                mode='bilinear', align_corners=False) > .99).any())
            if used:
                break
        with torch.no_grad():
            near_state, near_error = candidate(sample, .35, 1.5, .9)
        states = torch.cat((sample['state'], near_state))[None]
        reflections = sample['reflection'][:, None].expand(1, 2)
        inner.append((sample['inputs'], sample['state'], sample['reflection'],
                      sample['offsets'], sample['weights'], sample['valid_mask'],
                      states, reflections))
        inner_records.append({'base_index': base, 'slot': slot, 'virtual_index': virtual,
            'draw_seed': draw_seed, 'attempts': attempt, 'initial_near_mm': near_error,
            **sample['provenance'][0]})
(run / 'inner_records.jsonl').write_text(''.join(json.dumps(record, allow_nan=False) + '\n'
                                            for record in inner_records))
assert len(inner) == 16 and {row['base_index'] for row in inner_records} == set(range(60, 64))


def save(step):
    for name in arms:
        path = run / f'{name}_step_{step:05d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save({'step': step, 'arm': name, 'matcher': matchers[name].state_dict(),
            'optimizer': optimizers[name].state_dict(), 'scaler': scalers[name].state_dict(),
            'config': config, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
        os.replace(temporary, path)


def inner_score(name):
    matcher = matchers[name].eval()
    exact, near = [], []
    with torch.inference_mode():
        for image, state, reflection, offsets, weights, valid, states, reflections in inner:
            with torch.autocast('cuda', dtype=torch.float16):
                output = matcher(image, context['atlas'], states, reflections,
                                 offsets, weights, atlas_intensity=name == 'full')
            observed = chart256[valid[0]]
            target = rigid_points_090(state[:, None], reflection[:, None], observed)
            corrected = rigid_points_090(output['corrected_state'].float(), reflections, observed)
            errors = ((corrected - target).norm(dim=-1).mean(-1) / 1000)[0]
            exact.append(float(errors[0]))
            near.append(float(errors[1]))
    matcher.train()
    return {'exact_rigid_mm': float(np.mean(exact)),
            'near_rigid_mm': float(np.mean(near)),
            'score': float(np.mean(exact) + np.mean(near))}


save(0)
started, attempted, accepted_ids = time.perf_counter(), 0, set()
inner_scores = {}
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as logs, \
     (run / 'inner_scores.jsonl').open('w') as selections:
    inner_scores[0] = {name: inner_score(name) for name in arms}
    selections.write(json.dumps({'step': 0, 'arms': inner_scores[0]}, allow_nan=False) + '\n')
    selections.flush()
    print(json.dumps({'event': 'inner_checkpoint', 'step': 0,
                      'arms': inner_scores[0]}), flush=True)
    for step in range(1, steps + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        totals = {name: {'loss': 0., 'soft_key_ce': 0., 'expected_ccf': 0.,
                         'corrected_plane': 0., 'visibility': 0., 'reliability': 0.,
                         'mean_fit_weight': 0., 'reachable': 0, 'valid': 0}
                  for name in arms}
        candidate_errors = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(60 * 64))
                assert context['subjects'][virtual]['base_index'] < 60
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                attempted += 1
                record = sample['provenance'][0]
                used = bool(sample['eligible'][0]) and bool((F.grid_sample(
                    sample['valid_mask'][:, None].float(), grid24,
                    mode='bilinear', align_corners=False) > .99).any())
                draws.write(json.dumps({'step': step, 'slot': slot, 'appearance': appearance,
                    'attempt': attempted, 'used': used, **record}, allow_nan=False) + '\n')
                if used:
                    accepted_ids.add(record['physical_section_id'])
                    break
            with torch.no_grad():
                near_state, near_error = candidate(sample, .35, 1.5, .9)
                wrong_state, wrong_error = candidate(sample, 1.5, 3., 2.2, wrong=True)
                states = torch.cat((sample['state'], near_state, wrong_state))[None]
                reflections = sample['reflection'][:, None].expand(1, 3)
                candidate_errors.append((near_error, wrong_error))

            for name in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = matchers[name](sample['inputs'], context['atlas'], states,
                        reflections, sample['offsets'], sample['weights'],
                        atlas_intensity=name == 'full')
                with torch.no_grad():
                    query_grid = (2 * output['source_grid'] - 1).reshape(1, 24, 24, 2)
                    valid_weight = F.grid_sample(sample['valid_mask'][:, None].float(),
                        query_grid, mode='bilinear', align_corners=False).flatten(1)
                    truth = F.grid_sample(sample['centre'].permute(0, 3, 1, 2),
                        query_grid, mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
                    target_rigid = rigid_points_090(sample['state'], sample['reflection'],
                        output['source_grid'][0] - .5 / side)
                    valid_query = valid_weight > .99
                    distance = torch.cdist(truth.expand(3, -1, -1), output['key_ccf'][0].float())
                    distance.masked_fill_(~torch.isfinite(output['logits'][0, :, :, :-1]), torch.inf)
                    nearest = distance.amin(-1)
                    reachable = valid_query.expand(3, -1) & (nearest <= 900.)

                query_mask = valid_query.expand(3, -1)
                local_distance = distance[query_mask]
                logp = output['logits'][0][query_mask].float().log_softmax(-1)
                positive = torch.exp(-.5 * (local_distance / 450.).square())
                positive = torch.where(local_distance <= 1200., positive, 0.)
                mass = positive.sum(-1)
                soft_ce = -(torch.where(positive > 0, logp[:, :-1], 0.) * positive).sum(-1)
                soft_ce = torch.where(mass > 0, soft_ce / mass.clamp_min(1e-8), -logp[:, -1])
                ce_weight = torch.where(reachable[query_mask], 1., .3)
                key_ce = (soft_ce * ce_weight).sum() / ce_weight.sum()
                displacement = (output['expected_ccf'][0] - truth.expand(3, -1, -1)) / 1000
                expected = (F.smooth_l1_loss(displacement[reachable],
                    torch.zeros_like(displacement[reachable]), beta=.25)
                    if bool(reachable.any()) else displacement.sum() * 0)
                reliability_target = torch.exp(-.5 *
                    (displacement.detach().norm(dim=-1) / .75).square())
                reliability = F.binary_cross_entropy_with_logits(
                    output['reliability_logits'][0][query_mask].float(),
                    reliability_target[query_mask].float())
                mapped = rigid_points_090(output['corrected_state'], reflections,
                                          output['source_grid'][0] - .5 / side)
                rigid_mm = (mapped[0] - target_rigid.expand(3, -1, -1)).norm(dim=-1) / 1000
                corrected = F.smooth_l1_loss(rigid_mm[query_mask],
                    torch.zeros_like(rigid_mm[query_mask]), beta=.25)
                visibility = F.binary_cross_entropy_with_logits(
                    output['visibility_logits'][0].float(), valid_query[0].float())
                loss = key_ce + .5 * expected + corrected + .2 * visibility + .7 * reliability
                scalers[name].scale(loss / 3).backward()
                totals[name]['loss'] += float(loss.detach()) / 3
                totals[name]['soft_key_ce'] += float(key_ce.detach()) / 3
                totals[name]['expected_ccf'] += float(expected.detach()) / 3
                totals[name]['corrected_plane'] += float(rigid_mm[query_mask].detach().mean()) / 3
                totals[name]['visibility'] += float(visibility.detach()) / 3
                totals[name]['reliability'] += float(reliability.detach()) / 3
                totals[name]['mean_fit_weight'] += float(output['effective_fit_weight'].detach().mean()) / 3
                totals[name]['reachable'] += int(reachable.sum())
                totals[name]['valid'] += int(query_mask.sum())
                del output, loss, distance, logp, positive, soft_ce, displacement, mapped
            del sample
        for name in arms:
            scalers[name].unscale_(optimizers[name])
            gradient = torch.nn.utils.clip_grad_norm_(matchers[name].parameters(), 2.)
            assert torch.isfinite(gradient)
            scalers[name].step(optimizers[name])
            scalers[name].update()
        row = {'step': step, 'accepted': len(accepted_ids), 'attempted': attempted,
               'elapsed_seconds': time.perf_counter() - started, 'arms': totals,
               'candidate_initial_error_mm': candidate_errors}
        logs.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 1000 == 0:
            logs.flush()
            draws.flush()
            print(json.dumps({'event': 'milestone', **row}), flush=True)
        if step in checkpoints:
            save(step)
            inner_scores[step] = {name: inner_score(name) for name in arms}
            selections.write(json.dumps({'step': step, 'arms': inner_scores[step]}, allow_nan=False) + '\n')
            selections.flush()
            print(json.dumps({'event': 'inner_checkpoint', 'step': step,
                              'arms': inner_scores[step]}), flush=True)
best = min(checkpoints, key=lambda number: inner_scores[number]['full']['score'])
(run / 'completed.json').write_text(json.dumps({'steps': steps, 'selected_step': best,
    'accepted_synthetic_presentations': 3 * steps,
    'distinct_synthetic_physical_sections': len(accepted_ids), 'attempted': attempted,
    'v4_presentations': 2 * steps, 'v3_presentations': steps,
    'config_sha256': sha(run / 'config.json'), 'inner_records_sha256': sha(run / 'inner_records.jsonl'),
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'inner_scores_sha256': sha(run / 'inner_scores.jsonl'),
    'checkpoint_sha256': {name: {str(number): sha(run / f'{name}_step_{number:05d}.pt')
        for number in checkpoints} for name in arms},
    'source_sha256': config['source_sha256'], 'pose_unfrozen': False,
    'joint_feedback_trained': False, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'selected_step': best,
                  'presentations': 3 * steps}), flush=True)
