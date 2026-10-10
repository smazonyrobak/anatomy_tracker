"""Teacher-forced 3-D correspondence, then forced analytic full-plane fit."""

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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.atlas_spatial_fit_139 import AtlasSpatialFit139
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ATLAS_CORRESPONDENCE_CURRICULUM_140_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/atlas_correspondence_curriculum_140'
arms = ('full_intensity', 'support_only')
seed, side, updates = 20261010140, 256, 4000
valid_sites, invalid_sites = 128, 64
checkpoints = (0, 1000, 2000, 3000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_done.get(key, False) for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used', 'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
heads = {arms[0]: AtlasSpatialFit139().cuda().train()}
heads[arms[1]] = AtlasSpatialFit139().cuda().train()
heads[arms[1]].load_state_dict(heads[arms[0]].state_dict())
for head in heads.values():
    torch.nn.init.zeros_(head.gate.weight)
    torch.nn.init.constant_(head.gate.bias, 2.65)
    head.summary.requires_grad_(False)
    head.gate.requires_grad_(False)
    head.score.requires_grad_(False)
optimizers = {arm: torch.optim.AdamW(
    [parameter for parameter in heads[arm].parameters() if parameter.requires_grad],
    lr=2e-4, weight_decay=1e-4) for arm in arms}
scalers = {arm: torch.amp.GradScaler('cuda', init_scale=256., growth_interval=4001)
    for arm in arms}
source_files = ('train_atlas_correspondence_curriculum_140.py', 'atlas_spatial_fit_139.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side,
    'synthetic_per_update': {'v4': 2, 'v3': 1}, 'synthetic_presentations': 3 * updates,
    'valid_sites': valid_sites, 'invalid_visibility_sites': invalid_sites,
    'arms': arms, 'checkpoints': checkpoints, 'warmup_updates': 1000,
    'near_jitter_limits': [.07, .07, .07, 850., 850., 850., .04, .04, .03],
    'near_target_mm': .9, 'near_accepted_mm': [.35, 1.5],
    'amp_grad_scale': {'initial': 256., 'growth_interval': 4001},
    'loss_weights': {'coarse': 1., 'fine': .5, 'visibility': .2,
        'analytic_fit_after_warmup': .25},
    'gate': 'frozen zero weight, bias 2.65, tanh gain 0.990; score frozen zero',
    'truth_candidate_role': 'TRAIN teacher forcing only; frozen DEV evaluator uses blind 132 beam',
    'fit_path': 'fixed 16x16 queries to 3-D matches to two-pass affine O/U/V solve',
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'real_weak_train_used': False, 'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
for arm in arms:
    (run / arm).mkdir()
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    for arm in arms:
        path = run / arm / f'head_step_{step:05d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save({'step': step, 'arm': arm, 'head': heads[arm].state_dict(),
            'optimizer': optimizers[arm].state_dict(), 'scaler': scalers[arm].state_dict(),
            'config': config, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
        os.replace(temp, path)


def correspondence_loss(coarse_logits, fine_logits, coarse_world, coarse_support,
                        fine_world, fine_support, truth_world):
    candidates, queries = coarse_logits.shape[:2]
    truth = truth_world[None].expand(candidates, -1, -1).float()
    with torch.no_grad():
        distance = torch.cdist(truth, coarse_world.float())
        distance = distance.masked_fill(coarse_support[:, None] <= .5, torch.inf)
        nearest = distance.amin(-1)
        has_key = nearest <= 1500
        positive = distance <= torch.maximum(nearest[..., None] + 250, nearest.new_tensor(1000))
        fine_distance = (fine_world.float() - truth[:, :, None]).norm(dim=-1)
        fine_distance = fine_distance.masked_fill(fine_support <= .5, torch.inf)
        fine_nearest = fine_distance.amin(-1)
        fine_has_key = has_key & (fine_nearest <= 1500)
        fine_positive = fine_distance <= torch.maximum(
            fine_nearest[..., None] + 250, fine_nearest.new_tensor(750))
    coarse_log = torch.logsumexp(coarse_logits.float(), -1)
    coarse_positive = torch.logsumexp(coarse_logits[..., :-1].float().masked_fill(
        ~positive, -1e4), -1)
    coarse = (coarse_log - torch.where(has_key, coarse_positive,
                                      coarse_logits[..., -1].float())).mean()
    fine_log = torch.logsumexp(fine_logits.float(), -1)
    fine_positive_log = torch.logsumexp(fine_logits[..., :-1].float().masked_fill(
        ~fine_positive, -1e4), -1)
    fine_ce = fine_log - torch.where(fine_has_key, fine_positive_log,
                                     fine_logits[..., -1].float())
    fine_train = (~has_key) | fine_has_key
    fine = fine_ce.masked_select(fine_train).sum() / fine_train.sum().clamp_min(1)
    return coarse, fine, has_key.float().mean(), fine_has_key.float().sum() / has_key.sum().clamp_min(1)


save(0)
fixed_xy = heads[arms[0]].fixed_query_xy
fixed_pixel = (fixed_xy * side).long()
fixed_x, fixed_y = fixed_pixel[:, 0], fixed_pixel[:, 1]
limits = torch.tensor(config['near_jitter_limits'], device='cuda')
started, attempts = time.perf_counter(), 0
accepted_ids = set()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        totals = {arm: {'coarse': 0., 'fine': 0., 'visibility': 0., 'fit_mm': 0.,
                        'coarse_key_fraction': 0., 'fine_given_coarse_fraction': 0.,
                        'loss': 0.} for arm in arms}
        near_mm, jitter_batches, accepted_this_step = [], [], []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                attempts += 1
                record = sample['provenance'][0]
                eligible = bool(sample['eligible'][0])
                draws.write(json.dumps({'kind': 'synthetic_train', 'update': step,
                    'slot': slot, 'appearance_version': appearance,
                    'draw_attempt': attempts, 'used': eligible, **record}, allow_nan=False) + '\n')
                if eligible:
                    assert record['physical_section_id'] not in accepted_ids
                    accepted_ids.add(record['physical_section_id'])
                    accepted_this_step.append(record['physical_section_id'])
                    break

            valid = sample['valid_mask'][0]
            valid_index = torch.multinomial(valid.flatten().float(), valid_sites, replacement=True)
            invalid_index = torch.multinomial((~valid).flatten().float(), invalid_sites,
                                              replacement=True)
            x, y = valid_index.remainder(side), valid_index.div(side, rounding_mode='floor')
            chart = torch.stack((x, y), -1).float() / side
            aux_index = torch.cat((valid_index, invalid_index))
            aux_xy = torch.stack((aux_index.remainder(side),
                aux_index.div(side, rounding_mode='floor')), -1).float().add(.5).div(side)[None]
            truth_world = sample['centre'][0, y, x]
            fixed_valid = valid[fixed_y, fixed_x]
            fixed_truth = sample['centre'][0, fixed_y, fixed_x][fixed_valid]
            with torch.no_grad():
                prediction = model.predict(sample['inputs'])
                target = rigid_points_090(sample['state'], sample['reflection'], chart)
                for jitter_batch in range(1, 9):
                    jitter = (2 * torch.rand(32, 9, device='cuda') - 1) * limits
                    choices = compose_full_frame_state(sample['state'].expand(32, -1), jitter)
                    candidate_error = (rigid_points_090(choices,
                        sample['reflection'].expand(32), chart) - target).norm(dim=-1).mean(-1) / 1000
                    eligible_jitter = (candidate_error >= .35) & (candidate_error <= 1.5)
                    if bool(eligible_jitter.any()):
                        near_index = int((candidate_error - .9).abs().masked_fill(
                            ~eligible_jitter, torch.inf).argmin())
                        near_state = choices[near_index:near_index + 1]
                        near_update = jitter[near_index]
                        near_error = candidate_error[near_index]
                        break
                else:
                    near_update = torch.tensor([0., 0., 0., 900., 0., 0., 0., 0., 0.],
                                               device='cuda')
                    near_state = compose_full_frame_state(sample['state'], near_update[None])
                    near_error = (rigid_points_090(near_state, sample['reflection'], chart)
                                  - target).norm(dim=-1).mean() / 1000
                    jitter_batch = 9
                near_mm.append(float(near_error))
                jitter_batches.append(jitter_batch)
                states = torch.cat((sample['state'], near_state))[None]
                reflected = sample['reflection'][:, None].expand(-1, 2)
                draws.write(json.dumps({'kind': 'near_train_candidate', 'update': step,
                    'slot': slot, 'physical_section_id': record['physical_section_id'],
                    'jitter_batches': jitter_batch, 'near_initial_mm': float(near_error),
                    'near_local_update': near_update.tolist()}, allow_nan=False) + '\n')

            for arm in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = heads[arm](prediction, sample['inputs'], states, reflected,
                        context['atlas'], sample['offsets'], sample['weights'],
                        atlas_intensity_enabled=(arm == arms[0]), query_xy=aux_xy)
                coarse_key = output['coarse_key_world_um'][0]
                coarse_support = output['coarse_key_support'][0]
                if bool(fixed_valid.any()):
                    main_coarse, main_fine, main_key, main_fine_key = correspondence_loss(
                        output['coarse_logits'][0, :, fixed_valid],
                        output['fine_logits'][0, :, fixed_valid], coarse_key, coarse_support,
                        output['fine_key_world_um'][0, :, fixed_valid],
                        output['fine_key_support'][0, :, fixed_valid], fixed_truth)
                else:
                    main_coarse = main_fine = main_key = main_fine_key = states.new_zeros(())
                auxiliary = output['auxiliary']
                aux_coarse, aux_fine, aux_key, aux_fine_key = correspondence_loss(
                    auxiliary['coarse_logits'][0, :, :valid_sites],
                    auxiliary['fine_logits'][0, :, :valid_sites], coarse_key, coarse_support,
                    auxiliary['fine_key_world_um'][0, :, :valid_sites],
                    auxiliary['fine_key_support'][0, :, :valid_sites], truth_world)
                coarse, fine = .5 * (main_coarse + aux_coarse), .5 * (main_fine + aux_fine)
                visibility = .5 * (F.binary_cross_entropy_with_logits(
                    output['visibility_logit'][0].float(), fixed_valid.float()) +
                    F.binary_cross_entropy_with_logits(auxiliary['visibility_logit'][0].float(),
                        torch.cat((torch.ones(valid_sites, device='cuda'),
                                   torch.zeros(invalid_sites, device='cuda')))))
                corrected = rigid_points_090(output['corrected_state'][0], reflected[0], chart)
                fit_mm = (corrected - target).norm(dim=-1).mean(-1).mean() / 1000
                loss = coarse + .5 * fine + .2 * visibility + (.25 * fit_mm if step > 1000 else 0)
                scalers[arm].scale(loss / 3).backward()
                values = {'coarse': coarse, 'fine': fine, 'visibility': visibility,
                    'fit_mm': fit_mm, 'coarse_key_fraction': .5 * (main_key + aux_key),
                    'fine_given_coarse_fraction': .5 * (main_fine_key + aux_fine_key),
                    'loss': loss}
                for name, value in values.items():
                    totals[arm][name] += float(value.detach()) / 3

        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        gradients = {}
        for arm in arms:
            optimizer = optimizers[arm]
            optimizer.param_groups[0]['lr'] = 2e-4 * decay
            scalers[arm].unscale_(optimizer)
            gradients[arm] = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in heads[arm].parameters() if parameter.requires_grad],
                5., error_if_nonfinite=True))
            scalers[arm].step(optimizer)
            scalers[arm].update()
        log.write(json.dumps({'update': step, 'stage': 'correspondence' if step <= 1000 else 'fit',
            'synthetic_presentations': 3 * step, 'synthetic_section_ids': accepted_this_step,
            'near_initial_mm': near_mm, 'near_jitter_batches': jitter_batches,
            'arms': totals, 'gradient_norm': gradients,
            'seconds_since_process_start': time.perf_counter() - started}, allow_nan=False) + '\n')
        if step in checkpoints:
            log.flush()
            draws.flush()
            save(step)
            print(json.dumps({'event': 'train_milestone', 'update': step,
                'losses': {arm: totals[arm]['loss'] for arm in arms},
                'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 3)}),
                flush=True)

assert len(accepted_ids) == 3 * updates
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates, 'v4_presentations': 2 * updates,
    'v3_presentations': updates, 'synthetic_draw_attempts': attempts,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'parent_checkpoint_sha256': sha(parent_path), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'], 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {arm: {str(step): sha(run / arm / f'head_step_{step:05d}.pt')
        for step in checkpoints} for arm in arms}, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
