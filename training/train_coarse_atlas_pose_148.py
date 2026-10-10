"""Warm up coarse 3-D atlas correspondences on independent TRAIN sections."""

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
from training.coarse_atlas_pose_148 import CoarseAtlasPose148
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/COARSE_ATLAS_POSE_148_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/coarse_atlas_pose_148'
seed, side, steps = 20261010148, 256, 6000
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
matchers = {name: CoarseAtlasPose148().cuda().train() for name in arms}
matchers['support_only'].load_state_dict(matchers['full'].state_dict())
optimizers = {name: torch.optim.AdamW(matchers[name].parameters(), lr=2e-4,
    weight_decay=1e-4) for name in arms}
scalers = {name: torch.amp.GradScaler('cuda', init_scale=256., growth_interval=4001)
    for name in arms}
source_names = ('train_coarse_atlas_pose_148.py', 'coarse_atlas_pose_148.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'steps': steps, 'side': side, 'arms': arms,
    'checkpoints': checkpoints, 'synthetic_per_step': {'v4': 2, 'v3': 1},
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'split': '64 TRAIN deformation bases only; synthetic DEV, acquired DEV, calibration, final test excluded',
    'candidate_initial_error_mm': {'exact': 0, 'near_target': .9,
                                   'wrong_range': [1.5, 3.]},
    'key_reachable_um': 900,
    'loss_weights': {'key_ce': 1., 'expected_ccf_huber': .5,
                     'corrected_plane_huber': 1., 'image_only_visibility_bce': .2},
    'training_stage': 'independent coarse matcher warmup; frozen 132 supplies DEV beam only',
    'pose_unfreeze_after_development_gate': True,
    'pose_unfrozen_in_this_run': False, 'joint_feedback_trained': False,
    'same_initial_matcher_weights_and_draws': True,
    'real_weak_train_used': False, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    for name in arms:
        path = run / f'{name}_step_{step:05d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save({'step': step, 'arm': name, 'matcher': matchers[name].state_dict(),
            'optimizer': optimizers[name].state_dict(), 'scaler': scalers[name].state_dict(),
            'config': config, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
        os.replace(temporary, path)


save(0)
started, attempted, accepted_ids = time.perf_counter(), 0, set()
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')
axis24 = (torch.arange(24, device='cuda') + .5) / 24
q24y, q24x = torch.meshgrid(axis24, axis24, indexing='ij')
query_grid24 = 2 * torch.stack((q24x, q24y), -1)[None] - 1
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as logs:
    for step in range(1, steps + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        totals = {name: {'loss': 0., 'key_ce': 0., 'expected_ccf': 0.,
                         'corrected_plane': 0., 'visibility': 0., 'reachable': 0,
                         'valid': 0} for name in arms}
        candidate_errors = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                assert context['subjects'][virtual]['base_lineage']['split'] == 'train'
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                attempted += 1
                record = sample['provenance'][0]
                used = (bool(sample['eligible'][0])
                    and bool(sample['valid_mask'][0, 2::4, 2::4].any())
                    and bool((F.grid_sample(sample['valid_mask'][:, None].float(),
                        query_grid24, mode='bilinear', align_corners=False) > .99).any()))
                draws.write(json.dumps({'step': step, 'slot': slot, 'appearance': appearance,
                    'attempt': attempted, 'used': used, **record}, allow_nan=False) + '\n')
                if used:
                    accepted_ids.add(record['physical_section_id'])
                    break

            with torch.no_grad():
                chart = (torch.arange(64, device='cuda') + .5) / 64
                cy, cx = torch.meshgrid(chart, chart, indexing='ij')
                valid = sample['valid_mask'][0, 2::4, 2::4]
                points = torch.stack((cx, cy), -1)[valid]
                reference = rigid_points_090(sample['state'], sample['reflection'], points)
                while True:
                    near_update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
                    near_states = compose_full_frame_state(sample['state'].expand(64, -1), near_update)
                    near_error = (rigid_points_090(near_states,
                        sample['reflection'].expand(64), points) - reference).norm(dim=-1).mean(-1) / 1000
                    near_ok = (near_error >= .35) & (near_error <= 1.5)
                    if bool(near_ok.any()):
                        break
                near_choice = int((near_error - .9).abs().masked_fill(~near_ok, 10).argmin())
                while True:
                    wrong_update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
                    wrong_update[:, :6] *= 3
                    wrong_update[:, 6:] *= 2
                    wrong_states = compose_full_frame_state(sample['state'].expand(64, -1), wrong_update)
                    wrong_error = (rigid_points_090(wrong_states,
                        sample['reflection'].expand(64), points) - reference).norm(dim=-1).mean(-1) / 1000
                    wrong_ok = (wrong_error >= 1.5) & (wrong_error <= 3.)
                    if bool(wrong_ok.any()):
                        break
                wrong_choice = int((wrong_error - 2.2).abs().masked_fill(~wrong_ok, 10).argmin())
                states = torch.cat((sample['state'], near_states[near_choice:near_choice + 1],
                                    wrong_states[wrong_choice:wrong_choice + 1]))[None]
                reflection = sample['reflection'][:, None].expand(1, 3)
                candidate_errors.append((float(near_error[near_choice]),
                                         float(wrong_error[wrong_choice])))

            for name in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = matchers[name](sample['inputs'], context['atlas'], states,
                        reflection, sample['offsets'], sample['weights'],
                        atlas_intensity=name == 'full')
                if name == 'full':
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
                        nearest, labels = distance.min(-1)
                        reachable = valid_query.expand(3, -1) & (nearest <= 900.)
                        labels = torch.where(reachable, labels, output['key_ccf'].shape[2])
                        del distance
                query_mask = valid_query.expand(3, -1)
                ce = F.cross_entropy(output['logits'][0][query_mask].float(),
                                     labels[query_mask], reduction='none')
                ce_weight = torch.where(reachable[query_mask], 1., .3)
                key_ce = (ce * ce_weight).sum() / ce_weight.sum()
                displacement = (output['expected_ccf'][0] - truth.expand(3, -1, -1)) / 1000
                expected = (F.smooth_l1_loss(displacement[reachable],
                    torch.zeros_like(displacement[reachable]), beta=.25)
                    if bool(reachable.any()) else displacement.sum() * 0)
                mapped = rigid_points_090(output['corrected_state'], reflection,
                                          output['source_grid'][0] - .5 / side)
                rigid_mm = (mapped[0] - target_rigid.expand(3, -1, -1)).norm(dim=-1) / 1000
                corrected = F.smooth_l1_loss(rigid_mm[query_mask],
                    torch.zeros_like(rigid_mm[query_mask]), beta=.25)
                visibility = F.binary_cross_entropy_with_logits(
                    output['visibility_logits'][0].float(), valid_query[0].float())
                loss = key_ce + .5 * expected + corrected + .2 * visibility
                scalers[name].scale(loss / 3).backward()
                totals[name]['loss'] += float(loss.detach()) / 3
                totals[name]['key_ce'] += float(key_ce.detach()) / 3
                totals[name]['expected_ccf'] += float(expected.detach()) / 3
                totals[name]['corrected_plane'] += float(rigid_mm[query_mask].detach().mean()) / 3
                totals[name]['visibility'] += float(visibility.detach()) / 3
                totals[name]['reachable'] += int(reachable.sum())
                totals[name]['valid'] += int(query_mask.sum())
                del output, loss, ce, displacement, mapped
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
(run / 'completed.json').write_text(json.dumps({'steps': steps,
    'accepted_synthetic_presentations': 3 * steps,
    'distinct_synthetic_physical_sections': len(accepted_ids), 'attempted': attempted,
    'v4_presentations': 2 * steps, 'v3_presentations': steps,
    'config_sha256': sha(run / 'config.json'), 'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {name: {str(number): sha(run / f'{name}_step_{number:05d}.pt')
        for number in checkpoints} for name in arms},
    'source_sha256': config['source_sha256'], 'pose_unfrozen': False,
    'joint_feedback_trained': False, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}, indent=2))
print('148 coarse matcher warmup complete; frozen synthetic DEV gate required', flush=True)
