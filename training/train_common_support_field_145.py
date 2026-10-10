"""Matched 143 continuation with common-support rank and dense CCF contrast."""

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
protocol = source.parent / 'docs/publication/COMMON_SUPPORT_FIELD_145_PROTOCOL_20261010.md'
parent = root / 'runs/coherent_anatomy_field_143/full_step_10000.pt'
parent_done = root / 'runs/coherent_anatomy_field_143/completed.json'
pose_parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
run = root / 'runs/common_support_field_145'
seed, side, radius, steps = 20261010145, 256, 6, 2000
arms = ('treatment', 'control')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive == source.drive == run.drive == 'I:' and not run.exists()
assert sha(parent) == json.loads(parent_done.read_text())['checkpoint_sha256']['full']['10000']
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(pose_parent, map_location='cpu', weights_only=True)['model'])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
starting = torch.load(parent, map_location='cpu', weights_only=True)
fields = {name: CoherentAnatomyField143(radius=radius).cuda().train() for name in arms}
optimizers = {name: torch.optim.AdamW(fields[name].parameters(), lr=2e-4, weight_decay=1e-4)
              for name in arms}
scalers = {name: torch.amp.GradScaler('cuda', init_scale=256., growth_interval=4001)
           for name in arms}
for name in arms:
    fields[name].load_state_dict(starting['field'])
    optimizers[name].load_state_dict(starting['optimizer'])
    scalers[name].load_state_dict(starting['scaler'])
del starting
source_names = ('train_common_support_field_145.py', 'train_coherent_anatomy_field_143.py',
    'coherent_anatomy_field_143.py', 'coherent_anatomy_geometry_143.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'steps': steps, 'side': side, 'radius': radius,
    'slab_centres_um': [-1500, -1000, -500, 0, 500, 1000, 1500],
    'synthetic_per_step': {'v4': 2, 'v3': 1}, 'presentations': steps * 3,
    'checkpoints': [0, 500, 2000], 'arms': arms,
    'parent_checkpoint_sha256': sha(parent), 'parent_completion_sha256': sha(parent_done),
    'pose_parent_checkpoint_sha256': sha(pose_parent), 'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'same_initial_weights_optimizer_scaler': True,
    'loss_weights': {'reachable_ce': 1., 'valid_unreachable_ce': .3,
                     'background_ce': .05, 'offset_regression': .2,
                     'source_visibility': .1, 'common_support_energy_rank': .5,
                     'dense_ccf_infonce': .2},
    'candidate_rule': '64 near and 64 wrong proposals; near 0.35-1.5 mm, wrong 1-3 mm; choose maximum central pixelwise support IoU among eligible wrongs',
    'auxiliary_rule': 'exact/wrong all-depth binary support intersection; zero intensity outside; source 3x3 and common 3x3x3 erosion; at least 64 intact CCF sites and 0.05 common observed fraction',
    'rank_rule': 'softplus(5*(0.15+(exact_energy-wrong_energy)/common_observed_fraction))/5',
    'contrast_rule': 'up to 96 random CCF anchors; trilinear exact positive; same-grid wrong and other exact keys; mask negatives within 1 mm CCF; temperature 0.1',
    'real_weak_train_used': False, 'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'external_pretrained_weights_used': False}
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


axis = (torch.arange(64, device='cuda') + .5) / 64
gy, gx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((gx, gy), -1)
pixel_y, pixel_x = torch.meshgrid(torch.arange(64, device='cuda'),
                                  torch.arange(64, device='cuda'), indexing='ij')
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')


def candidates(sample):
    valid = sample['valid_mask'][0, 2::4, 2::4]
    points = chart[valid]
    truth = rigid_points_090(sample['state'], sample['reflection'], points)[0]
    updates = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
    near = compose_full_frame_state(sample['state'].expand(64, -1), updates)
    near_error = (rigid_points_090(near, sample['reflection'].expand(64), points) -
                  truth).norm(dim=-1).mean(-1) / 1000
    near_ok = (near_error >= .35) & (near_error <= 1.5)
    if not bool(near_ok.any()):
        return None
    near_choice = int((near_error - .9).abs().masked_fill(~near_ok, 100).argmin())
    updates = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
    updates[:, :3] *= 3
    updates[:, 3:6] *= 3
    updates[:, 6:] *= 2
    wrong = compose_full_frame_state(sample['state'].expand(64, -1), updates)
    wrong_error = (rigid_points_090(wrong, sample['reflection'].expand(64), points) -
                   truth).norm(dim=-1).mean(-1) / 1000
    eligible = ((wrong_error >= 1.) & (wrong_error <= 3.)).nonzero()[:, 0]
    if not len(eligible):
        return None
    exact_slab, _, _, _ = render_coherent_atlas_143(context['atlas'], sample['state'],
        sample['reflection'], sample['offsets'], sample['weights'])
    wrong_slabs, _, _, _ = render_coherent_atlas_143(context['atlas'], wrong[eligible],
        sample['reflection'].expand(len(eligible)), sample['offsets'].expand(len(eligible), -1),
        sample['weights'].expand(len(eligible), -1))
    exact_support = exact_slab[0, 1, 3] > .5
    wrong_support = wrong_slabs[:, 1, 3] > .5
    iou = ((wrong_support & exact_support).sum((1, 2)) /
           (wrong_support | exact_support).sum((1, 2)).clamp_min(1))
    choice = eligible[iou.argmax()]
    states = torch.cat((sample['state'], near[near_choice:near_choice + 1],
                        wrong[choice:choice + 1]))
    return states, float(near_error[near_choice]), float(wrong_error[choice]), float(iou.max())


def base_loss(output, local, labels, reachable, valid):
    logits = output['match_logits'].float()
    dustbin = logits.shape[1] - 1
    cross_entropy = F.cross_entropy(logits, torch.where(reachable, labels, dustbin),
                                    reduction='none')
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
    return hit + .3 * miss + .05 * background + .2 * displacement + .1 * visible, hit, displacement, visible


save(0)
started, accepted, attempted = time.perf_counter(), 0, 0
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as logs:
    for step in range(1, steps + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        totals = {name: {'loss': 0., 'hit_ce': 0., 'offset': 0., 'visibility': 0.,
                         'rank': 0., 'infonce': 0., 'reachable': 0., 'valid': 0.,
                         'ranked': 0., 'contrast_sites': 0., 'common_observed_fraction': 0.,
                         'wrong_central_support_iou': 0.} for name in arms}
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                    proposed = candidates(sample) if bool(sample['eligible'][0]) else None
                attempted += 1
                used = proposed is not None
                draws.write(json.dumps({'step': step, 'slot': slot, 'appearance': appearance,
                    'attempt': attempted, 'used': used,
                    'rejection': None if used else ('candidate_bank' if bool(sample['eligible'][0]) else 'section'),
                    'near_rigid_mm': proposed[1] if used else None,
                    'wrong_rigid_mm': proposed[2] if used else None,
                    'wrong_central_support_iou': proposed[3] if used else None,
                    **sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    break
            with torch.no_grad():
                prediction = model.predict(sample['inputs'])
                states, near_mm, wrong_mm, wrong_iou = proposed
                reflection = sample['reflection'].expand(3)
                slabs, base, basis, normal = render_coherent_atlas_143(context['atlas'],
                    states, reflection, sample['offsets'].expand(3, -1),
                    sample['weights'].expand(3, -1))
                local, classes, reachable = dense_local_targets_143(
                    sample['centre'].expand(3, -1, -1, -1),
                    sample['valid_mask'].expand(3, -1, -1), base, basis, normal, radius=radius)
                valid = sample['valid_mask'][:, 2::4, 2::4].expand(3, -1, -1)
                lattice = torch.stack((local[..., 0] * 64, local[..., 1] * 64,
                    local[..., 2] / 500), -1).round().long()
                depth = (lattice[..., 2] + 3).clamp(0, 6)
                row = (pixel_y + lattice[..., 1]).clamp(0, 63)
                column = (pixel_x + lattice[..., 0]).clamp(0, 63)
                key_support = slabs[:, 1][torch.arange(3, device='cuda')[:, None, None],
                                               depth, row, column] > .5
                reachable &= key_support
                feature = prediction['feature']
                common = ((slabs[0:1, 1:2] > .5) & (slabs[2:3, 1:2] > .5)).float()
                fraction = float(((common[0, 0, 3] > .5) & valid[0]).sum() /
                                 valid[0].sum().clamp_min(1))
                grid = torch.stack((2 * (pixel_x + local[0, ..., 0] * 64 + .5) / 64 - 1,
                                    2 * (pixel_y + local[0, ..., 1] * 64 + .5) / 64 - 1,
                                    2 * (3 + local[0, ..., 2] / 500 + .5) / 7 - 1), -1)
                eroded_source = F.avg_pool2d(valid[0][None, None].float(), 3, 1, 1)[0, 0] == 1
                eroded_common = F.avg_pool3d(common, 3, 1, 1)
                supported_truth = F.grid_sample(eroded_common, grid[None, None],
                    mode='nearest', align_corners=False)[0, 0, 0] == 1
                interior = reachable[0] & eroded_source & supported_truth
                interior_count = int(interior.sum())
                rank_allowed = interior_count >= 64 and fraction >= .05
                pair_aux = torch.cat((slabs[[0, 2], :1] * common,
                    common.expand(2, -1, -1, -1, -1)), 1) if rank_allowed else None
            for name in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = fields[name](sample['inputs'].expand(3, -1, -1, -1), slabs,
                                          500., source_feature=feature.expand(3, -1, -1, -1))
                loss, hit, displacement, visible = base_loss(output, local, classes, reachable, valid)
                scalers[name].scale(loss / 3).backward()
                item = totals[name]
                item['loss'] += float(loss.detach()) / 3
                item['hit_ce'] += float(hit.detach()) / 3
                item['offset'] += float(displacement.detach()) / 3
                item['visibility'] += float(visible.detach()) / 3
                item['reachable'] += int(reachable.sum())
                item['valid'] += int(valid.sum())
                item['common_observed_fraction'] += fraction / 3
                item['wrong_central_support_iou'] += wrong_iou / 3
                del output, loss, hit, displacement, visible
            if rank_allowed:
                field = fields['treatment']
                scaler = scalers['treatment']
                with torch.autocast('cuda', dtype=torch.float16):
                    output = field(sample['inputs'].expand(2, -1, -1, -1), pair_aux,
                                   500., source_feature=feature.expand(2, -1, -1, -1))
                rank = F.softplus(5 * (.15 +
                    (output['energy'][0].float() - output['energy'][1].float()) / fraction)) / 5
                scaler.scale(.5 * rank / 3).backward()
                del output
                locations = interior.nonzero()
                selected = locations[torch.randperm(len(locations), device='cuda')[:96]]
                y, x = selected[:, 0], selected[:, 1]
                n = len(selected)
                sample_grid = grid[y, x][None, None, None].expand(2, -1, -1, -1, -1)
                with torch.autocast('cuda', dtype=torch.float16):
                    image = field.source_fine(F.interpolate(sample['inputs'], (128, 128),
                        mode='bilinear', align_corners=False))
                    image = image + field.shared_source(F.interpolate(feature, (64, 64),
                        mode='bilinear', align_corners=False))
                    image = F.normalize(image + F.interpolate(field.source_context(image),
                        (64, 64), mode='bilinear', align_corners=False), dim=1)
                    atlas = F.normalize(field.atlas(pair_aux), dim=1)
                    sampled = F.grid_sample(atlas, sample_grid, mode='bilinear',
                                            align_corners=False)
                query = F.normalize(image[0, :, y, x].T.float(), dim=1)
                keys = F.normalize(sampled[:, :, 0, 0].permute(0, 2, 1).float(), dim=-1)
                truth = sample['centre'][0, 2::4, 2::4][y, x]
                delta = local[0, y, x]
                wrong_ccf = (base[2, y, x] + delta[:, 0, None] * basis[2, :, 0] +
                    delta[:, 1, None] * basis[2, :, 1] + delta[:, 2, None] * normal[2])
                near_exact = torch.cdist(truth, truth) < 1000
                near_exact.fill_diagonal_(False)
                near_wrong = torch.cdist(truth, wrong_ccf) < 1000
                logits = (query @ torch.cat((keys[0], keys[1])).T / .1).masked_fill(
                    torch.cat((near_exact, near_wrong), 1), -1e4)
                infonce = F.cross_entropy(logits, torch.arange(n, device='cuda'))
                scaler.scale(.2 * infonce / 3).backward()
                item = totals['treatment']
                item['loss'] += float((.5 * rank + .2 * infonce).detach()) / 3
                item['rank'] += float(rank.detach()) / 3
                item['infonce'] += float(infonce.detach()) / 3
                item['ranked'] += 1
                item['contrast_sites'] += n
                del rank, infonce, image, atlas, sampled, query, keys, logits
            del sample, prediction, slabs, pair_aux
        for name in arms:
            scalers[name].unscale_(optimizers[name])
            gradient = torch.nn.utils.clip_grad_norm_(fields[name].parameters(), 2.)
            assert torch.isfinite(gradient)
            scalers[name].step(optimizers[name])
            scalers[name].update()
        row = {'step': step, 'accepted': accepted, 'attempted': attempted,
               'elapsed_seconds': time.perf_counter() - started, 'arms': totals,
               'last_near_mm': near_mm, 'last_wrong_mm': wrong_mm}
        logs.write(json.dumps(row, allow_nan=False) + '\n')
        if step % 100 == 0:
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
    'source_sha256': config['source_sha256'], 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}, indent=2))
print('145 common-support field pilot complete; independent DEV gate required', flush=True)
