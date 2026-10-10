"""Matched blind-beam atlas-intensity versus support-only spatial-fit training."""

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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
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
protocol = source.parent / 'docs/publication/ATLAS_SPATIAL_FIT_139_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/atlas_spatial_fit_139'
arms = ('full_intensity', 'support_only')
seed, side, updates, valid_sites, invalid_sites = 20261010139, 256, 2000, 128, 64
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
optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    for arm, head in heads.items()}
scalers = {arm: torch.amp.GradScaler('cuda') for arm in arms}
source_files = ('train_atlas_spatial_fit_139.py', 'atlas_spatial_fit_139.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side, 'valid_sites': valid_sites,
    'invalid_visibility_sites': invalid_sites, 'synthetic_per_update': {'v4': 2, 'v3': 1},
    'arms': arms, 'checkpoints': (0, 500, 1000, 1500, updates),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'), 'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'beam': 'frozen prior top8 base + top6 anchors + two antipodal-normal diversity anchors',
    'branch_selection': 'near <=1.5mm, support-matched wrong >=2mm and <=0.05 support difference, top prior, normal-diverse; no truth candidate insertion',
    'support_matching': 'finite-PSF Allen support on the same 128 sampled valid observed sites',
    'correspondence': 'global coarse positive key bag within max(1mm, nearest+0.25mm), local fine positive bag, 1.5mm dustbin; invalid sites visibility only',
    'loss': 'valid-site fixed+aux coarse/fine 0.10 each, fixed+aux visibility 0.20, initially-near corrected rigid 1.0, corrected-cost listwise 1.0, near/wrong pair 0.5',
    'score': 'frozen mode/reflection log prior plus learned head score_delta; fixed-grid fit only',
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


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


def match_loss(logits, fine_logits, key_world, key_support, fine_world, fine_support,
               truth_world):
    candidates, queries = logits.shape[:2]
    world = truth_world[None].expand(candidates, -1, -1).float()
    with torch.no_grad():
        coarse_distance = torch.cdist(world, key_world.float())
        coarse_distance = coarse_distance.masked_fill(key_support[:, None] <= .5, float('inf'))
        nearest = coarse_distance.amin(-1)
        has_key = nearest <= 1500
        coarse_positive = coarse_distance <= torch.maximum(
            nearest[..., None] + 250, nearest.new_tensor(1000))
        fine_distance = (fine_world.float() - world[:, :, None]).norm(dim=-1)
        fine_distance = fine_distance.masked_fill(fine_support <= .5, float('inf'))
        fine_nearest = fine_distance.amin(-1)
        fine_has_key = has_key & (fine_nearest <= 1500)
        fine_positive = fine_distance <= torch.maximum(
            fine_nearest[..., None] + 250, fine_nearest.new_tensor(750))
    coarse_all = torch.logsumexp(logits.float(), -1)
    coarse_positive_log = torch.logsumexp(logits[..., :-1].float().masked_fill(
        ~coarse_positive, -1e4), -1)
    coarse = (coarse_all - torch.where(has_key, coarse_positive_log,
                                      logits[..., -1].float())).mean()
    fine_all = torch.logsumexp(fine_logits.float(), -1)
    fine_positive_log = torch.logsumexp(fine_logits[..., :-1].float().masked_fill(
        ~fine_positive, -1e4), -1)
    fine = fine_all - torch.where(fine_has_key, fine_positive_log,
                                  fine_logits[..., -1].float())
    fine_mask = (~has_key) | fine_has_key
    fine = fine.masked_select(fine_mask).sum() / fine_mask.sum().clamp_min(1)
    return coarse, fine


save(0)
fixed_xy = heads[arms[0]].fixed_query_xy
fixed_pixel = (fixed_xy * side).long()
fixed_x, fixed_y = fixed_pixel[:, 0], fixed_pixel[:, 1]
started = time.perf_counter()
attempts = 0
accepted_ids = set()
availability = {'near': 0, 'support_matched_wrong': 0}
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        arm_totals = {arm: {'pose': 0., 'rank': 0., 'pair': 0., 'coarse': 0.,
                            'fine': 0., 'visibility': 0., 'loss': 0.} for arm in arms}
        selected_counts = {'near': 0, 'support_matched_wrong': 0}
        accepted_this_step = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                eligible = bool(sample['eligible'][0])
                record = sample['provenance'][0]
                attempts += 1
                draws.write(json.dumps({'kind': 'synthetic_train', 'update': step, 'slot': slot,
                    'appearance_version': appearance, 'draw_attempt': attempts,
                    'used': eligible, **record}, allow_nan=False) + '\n')
                if eligible:
                    accepted_ids.add(record['physical_section_id'])
                    accepted_this_step.append(record['physical_section_id'])
                    break
            valid = sample['valid_mask'][0]
            valid_index = torch.multinomial(valid.flatten().float(), valid_sites, replacement=True)
            invalid_index = torch.multinomial((~valid).flatten().float(), invalid_sites,
                                              replacement=True)
            x = valid_index.remainder(side)
            y = valid_index.div(side, rounding_mode='floor')
            chart = torch.stack((x, y), -1).float() / side
            aux_index = torch.cat((valid_index, invalid_index))
            aux_xy = torch.stack((aux_index.remainder(side),
                aux_index.div(side, rounding_mode='floor')), -1).float().add(.5).div(side)[None]
            aux_truth = sample['centre'][0, y, x]
            fixed_valid = valid[fixed_y, fixed_x]
            fixed_truth = sample['centre'][0, fixed_y, fixed_x][fixed_valid]
            with torch.no_grad():
                prediction = model.predict(sample['inputs'])
                prior, beam = beam_for(prediction)
                states = prediction['state'][0, beam // 2]
                reflected = beam.remainder(2)
                target = rigid_points_090(sample['state'], sample['reflection'], chart)[0]
                rigid_cost = (rigid_points_090(states, reflected, chart) - target
                              ).norm(dim=-1).mean(-1) / 1000
                support_points = rigid_points_090(states, reflected, chart)
                normal = full_frame_state_to_components(states)[1][..., :, 2]
                slab = support_points[:, None, None] + sample['offsets'][0, None, :, None, None, None] * normal[:, None, None, None, :]
                atlas_support = render_finite_thickness_coordinate_grid(
                    context['atlas'][1:2], slab, (0., 0., 0.), (25., 25., 25.),
                    sample['weights'][0])[..., 0, 0, :].mean(-1)
                near_position = int(rigid_cost.argmin())
                near = near_position if float(rigid_cost[near_position]) <= 1.5 else None
                wrong = None
                if near is not None:
                    wrong_mask = (rigid_cost >= 2) & ((atlas_support - atlas_support[near]).abs() <= .05)
                    wrong_mask[near] = False
                    if bool(wrong_mask.any()):
                        wrong = int(prior[beam].masked_fill(~wrong_mask, -1e6).argmax())
                top = int((beam == prior.argmax()).nonzero()[0, 0])
                selected_positions = []
                for position in (near, wrong, top):
                    if position is not None and position not in selected_positions:
                        selected_positions.append(position)
                dissimilarity = 1 - (normal[:, None] * normal[selected_positions][None]
                    ).sum(-1).abs().amax(-1)
                dissimilarity[selected_positions] = -1
                diverse = int(dissimilarity.argmax())
                if diverse not in selected_positions:
                    selected_positions.append(diverse)
                chosen_ids = beam[selected_positions]
                chosen_states = prediction['state'][:, chosen_ids // 2]
                chosen_reflected = chosen_ids.remainder(2)[None]
                chosen_prior = prior[chosen_ids]
                near_chosen = selected_positions.index(near) if near is not None else None
                wrong_chosen = selected_positions.index(wrong) if wrong is not None else None
                draws.write(json.dumps({'kind': 'blind_beam_selection', 'update': step,
                    'slot': slot, 'physical_section_id': record['physical_section_id'],
                    'beam_ids': beam.tolist(), 'branch_rigid_error_mm': rigid_cost.tolist(),
                    'branch_atlas_support_fraction': atlas_support.tolist(),
                    'selected_ids': chosen_ids.tolist(), 'near_available': near is not None,
                    'support_matched_wrong_available': wrong is not None,
                    'near_selected_index': near_chosen,
                    'wrong_selected_index': wrong_chosen}, allow_nan=False) + '\n')
                selected_counts['near'] += near is not None
                selected_counts['support_matched_wrong'] += wrong is not None
                availability['near'] += near is not None
                availability['support_matched_wrong'] += wrong is not None

            for arm in arms:
                with torch.autocast('cuda', dtype=torch.float16):
                    output = heads[arm](prediction, sample['inputs'], chosen_states,
                        chosen_reflected, context['atlas'], sample['offsets'], sample['weights'],
                        atlas_intensity_enabled=(arm == arms[0]), query_xy=aux_xy)
                corrected = rigid_points_090(output['corrected_state'][0],
                    chosen_reflected[0], chart)
                corrected_cost = (corrected - target).norm(dim=-1).mean(-1) / 1000
                pose = corrected_cost[near_chosen] if near_chosen is not None else corrected_cost.new_zeros(())
                score = chosen_prior + output['score_delta'][0]
                rank_target = F.softmax(-corrected_cost.detach() / .7, -1)
                rank = -(rank_target * F.log_softmax(score, -1)).sum()
                pair = (F.softplus(1 - score[near_chosen] + score[wrong_chosen])
                    if wrong_chosen is not None else score.new_zeros(()))
                main = output
                aux = output['auxiliary']
                keys = output['coarse_key_world_um'][0]
                key_support = output['coarse_key_support'][0]
                if bool(fixed_valid.any()):
                    coarse_main, fine_main = match_loss(main['coarse_logits'][0, :, fixed_valid],
                        main['fine_logits'][0, :, fixed_valid], keys, key_support,
                        main['fine_key_world_um'][0, :, fixed_valid],
                        main['fine_key_support'][0, :, fixed_valid], fixed_truth)
                else:
                    coarse_main = fine_main = score.new_zeros(())
                coarse_aux, fine_aux = match_loss(aux['coarse_logits'][0, :, :valid_sites],
                    aux['fine_logits'][0, :, :valid_sites], keys, key_support,
                    aux['fine_key_world_um'][0, :, :valid_sites],
                    aux['fine_key_support'][0, :, :valid_sites], aux_truth)
                coarse = .5 * (coarse_main + coarse_aux)
                fine = .5 * (fine_main + fine_aux)
                visibility = .5 * (F.binary_cross_entropy_with_logits(
                    main['visibility_logit'][0].float(), fixed_valid.float()) +
                    F.binary_cross_entropy_with_logits(aux['visibility_logit'][0].float(),
                        torch.cat((torch.ones(valid_sites, device='cuda'),
                                   torch.zeros(invalid_sites, device='cuda')))))
                loss = pose + rank + .5 * pair + .10 * coarse + .10 * fine + .20 * visibility
                scalers[arm].scale(loss / 3).backward()
                values = {'pose': pose, 'rank': rank, 'pair': pair, 'coarse': coarse,
                          'fine': fine, 'visibility': visibility, 'loss': loss}
                for name, value in values.items():
                    arm_totals[arm][name] += float(value.detach()) / 3

        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        gradients = {}
        for arm in arms:
            optimizer = optimizers[arm]
            optimizer.param_groups[0]['lr'] = 2e-4 * decay
            scalers[arm].unscale_(optimizer)
            gradients[arm] = float(torch.nn.utils.clip_grad_norm_(
                heads[arm].parameters(), 5., error_if_nonfinite=True))
            scalers[arm].step(optimizer)
            scalers[arm].update()
        row = {'update': step, 'synthetic_presentations': 3 * step,
            'synthetic_section_ids': accepted_this_step,
            'near_available': selected_counts['near'],
            'support_matched_wrong_available': selected_counts['support_matched_wrong'],
            'arms': arm_totals, 'gradient_norm': gradients,
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step % 500 == 0:
            log.flush()
            draws.flush()
            save(step)
        if step == 1 or step % 500 == 0:
            if step == 1:
                log.flush()
                draws.flush()
            print(json.dumps({'event': 'train_milestone', 'update': step,
                'near_available_in_update': selected_counts['near'],
                'support_matched_wrong_in_update': selected_counts['support_matched_wrong'],
                'losses': {arm: arm_totals[arm]['loss'] for arm in arms},
                'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 3)
                    if step == 1 else None}), flush=True)

assert len(accepted_ids) == 3 * updates
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates, 'v4_presentations': 2 * updates,
    'v3_presentations': updates, 'synthetic_draw_attempts': attempts,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'branch_availability': availability,
    'parent_checkpoint_sha256': sha(parent_path), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'], 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'), 'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {arm: {str(step): sha(run / arm / f'head_step_{step:05d}.pt')
        for step in config['checkpoints']} for arm in arms}, 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
