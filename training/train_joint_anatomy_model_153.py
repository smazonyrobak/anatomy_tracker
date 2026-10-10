"""Bounded scratch TRAIN pilot for joint pose, atlas correspondence, and tissue map.

All checkpoint selection uses four held-out TRAIN deformation bases. This run
does not evaluate DEV data or qualify a model for deployment.
"""

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

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


seed, side, updates = 20261011153, 256, 1800
stage_a_end, stage_b_end = 800, 1300
checkpoints = (800, 1300, 1550, 1800)
source = Path(__file__).resolve().parent
run = root / 'runs/joint_anatomy_model_153_train151_002'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()

context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert all(base['lineage']['split'] == 'train' for base in context['bases'])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
model = JointAnatomyModel153().cuda().train()
pose_parameters = list(model.pose.parameters())
matcher_parameters = list(model.matcher.parameters())
field_parameters = list(model.field_body.parameters()) + list(model.field_out.parameters())
optimizer = torch.optim.AdamW([
    {'params': pose_parameters, 'lr': 1e-4},
    {'params': matcher_parameters, 'lr': 2e-4},
    {'params': field_parameters, 'lr': 0.},
], weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', init_scale=256., growth_interval=1000)

protocol = {'seed': seed, 'side': side, 'updates': updates,
    'stage_a_end': stage_a_end, 'stage_b_end': stage_b_end,
    'stage_c_requires_inner_gate': {'near_rigid_gain_mm_at_least': .1,
                                    'exact_rigid_drift_mm_at_most': .25},
    'checkpoints': checkpoints, 'gradient_bases': list(range(60)),
    'inner_bases': list(range(60, 64)), 'inner_sections_per_base': 4,
    'appearance_schedule': 'v4,v4,v3; one independently drawn physical plane per update',
    'candidate_errors_mm': {'near': [.35, 1.5], 'wrong': [1.5, 3.]},
    'minimum_24_grid_valid_queries_for_matcher': 4,
    'direct_pose_sites': 128, 'calibrated': False,
    'pretrained_weights_used': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False}
source_names = ('train_joint_anatomy_model_153.py', 'joint_anatomy_model_153.py',
    'coarse_atlas_pose_150.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'global_atlas_contrast_090.py')
config = {'protocol': protocol, 'source_sha256': {name: sha(source / name) for name in source_names},
          'synthetic_provenance': context['provenance'], 'torch': torch.__version__,
          'numpy': np.__version__}
run.mkdir(parents=True, exist_ok=False)
(run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
config['protocol_sha256'] = sha(run / 'protocol.json')
(run / 'config.json').write_text(json.dumps(config, indent=2))

axis24 = (torch.arange(24, device='cuda') + .5) / 24
y24, x24 = torch.meshgrid(axis24, axis24, indexing='ij')
grid24 = (2 * torch.stack((x24, y24), -1) - 1)[None]
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
reflection_options = torch.arange(2, device='cuda')[None, None]


def valid_chart(sample, sites=None):
    valid = sample['valid_mask'][0]
    pixels = valid.flatten().nonzero()[:, 0] if sites is None else torch.multinomial(
        valid.flatten().float(), sites, replacement=True)
    return torch.stack((pixels.remainder(side), pixels.div(side, rounding_mode='floor')), -1).float() / side


def direct_loss(sample, prediction, chart):
    states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
    reflections = reflection_options.expand(1, model.pose.modes, 2)
    prior = (prediction['log_mass'].float()[..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'].float()),
        F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
    true = rigid_points_090(sample['state'], sample['reflection'], chart)
    proposed = rigid_points_090(states, reflections, chart[None, None])
    true_five = rigid_points_090(sample['state'], sample['reflection'], corners)
    proposed_five = rigid_points_090(states, reflections, corners)
    tissue_um = (proposed - true[:, None, None]).norm(dim=-1).mean(-1)
    five_um = (proposed_five - true_five[:, None, None]).norm(dim=-1).mean(-1)
    normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
    true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
    normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
    target_mass = F.softmax(-physical.detach() / .7, -1)
    nearest = (true_normal @ model.pose.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
    near_mode = model.pose.base_modes + nearest.indices
    near_cost = physical.reshape(1, model.pose.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
            + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
            + .5 * physical[:, :2 * model.pose.base_modes].min(-1).values
            + (neighbourhood * near_cost).sum(-1)).mean()
    return loss, prior


def matcher_loss(sample, match):
    count = match['logits'].shape[1]
    with torch.no_grad():
        valid = (F.grid_sample(sample['valid_mask'][:, None].float(), grid24,
                               mode='bilinear', align_corners=False) > .99).flatten(1)
        truth = F.grid_sample(sample['centre'].permute(0, 3, 1, 2), grid24,
                              mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        query_mask = valid.expand(count, -1)
        distance = torch.cdist(truth.expand(count, -1, -1), match['key_ccf'][0].float())
        distance.masked_fill_(~torch.isfinite(match['logits'][0, :, :, :-1]), torch.inf)
        reachable = query_mask & (distance.amin(-1) <= 900.)
        local_distance = distance[query_mask]
        positive = torch.where(local_distance <= 1200.,
                               torch.exp(-.5 * (local_distance / 450.).square()), 0.)
        mass = positive.sum(-1)
        ce_weight = torch.where(reachable[query_mask], 1., .3)
    logp = match['logits'][0][query_mask].float().log_softmax(-1)
    soft_ce = -(torch.where(positive > 0, logp[:, :-1], 0.) * positive).sum(-1)
    soft_ce = torch.where(mass > 0, soft_ce / mass.clamp_min(1e-8), -logp[:, -1])
    key_ce = (soft_ce * ce_weight).sum() / ce_weight.sum()
    displacement = (match['expected_ccf'][0].float() - truth.expand(count, -1, -1)) / 1000
    expected = (F.smooth_l1_loss(displacement[reachable],
        torch.zeros_like(displacement[reachable]), beta=.25)
        if bool(reachable.any()) else displacement.sum() * 0)
    reliability_target = torch.exp(-.5 * (displacement.detach().norm(dim=-1) / .75).square())
    reliability = F.binary_cross_entropy_with_logits(
        match['reliability_logits'][0][query_mask].float(), reliability_target[query_mask])
    visibility = F.binary_cross_entropy_with_logits(
        match['visibility_logits'][0].float(), valid[0].float())
    reach_fraction = reachable.sum(-1).float() / valid.sum().clamp_min(1)
    return key_ce + .5 * expected + .2 * visibility + .7 * reliability, reach_fraction


def candidate(sample, chart, lower, upper, target, wrong=False):
    reference = rigid_points_090(sample['state'], sample['reflection'], chart)
    while True:
        update = (2 * torch.rand(64, 9, device='cuda') - 1) * limits
        if wrong:
            update[:, :6] *= 3
            update[:, 6:] *= 2
        states = compose_full_frame_state(sample['state'].expand(64, -1), update)
        error = (rigid_points_090(states, sample['reflection'].expand(64), chart)
                 - reference).norm(dim=-1).mean(-1) / 1000
        selected = (error >= lower) & (error <= upper)
        if bool(selected.any()):
            index = int((error - target).abs().masked_fill(~selected, 10).argmin())
            return states[index:index + 1], float(error[index])


def joint_loss(sample, output, chart, stage, reflections, near):
    matching, reach_fraction = matcher_loss(sample, output['matcher'])
    active = near & (reach_fraction >= .5) if stage == 'C' else near
    if stage == 'C' and not bool(active.any()):
        return None, 0., 0., 0., reach_fraction
    reference = rigid_points_090(sample['state'], sample['reflection'], chart)
    corrected = rigid_points_090(output['corrected_state'].float(),
                                 reflections, chart[None, None])
    rigid_mm = (corrected - reference[:, None]).norm(dim=-1) / 1000
    rigid_per = rigid_mm.mean(-1)
    rigid_points = F.smooth_l1_loss(rigid_mm, torch.zeros_like(rigid_mm), beta=.25,
                                    reduction='none').mean(-1)
    fit_active = active if stage == 'C' else (near | (reach_fraction >= .5))
    rigid = rigid_points[0, fit_active].mean()
    valid = sample['valid_mask'][0]
    dense_mm = (output['surface_ccf_um'][0, :, valid].float() - sample['centre'][0, valid]) / 1000
    dense_norm = dense_mm.norm(dim=-1).mean(-1)
    dense_points = F.smooth_l1_loss(dense_mm, torch.zeros_like(dense_mm), beta=.25,
                                    reduction='none').mean((-1, -2))
    dense = dense_points[active].mean()
    target = F.softmax(-(.5 * rigid_per.detach() + .5 * dense_norm.detach()) / .5, -1)
    logits = (-output['fit_energy_uncalibrated'] if stage == 'B'
              else output['score_uncalibrated']).float()
    ranking = (F.kl_div(F.log_softmax(logits, -1), target[None], reduction='batchmean')
               if stage == 'B' or bool(active.all()) else logits.sum() * 0)
    loss = (rigid + dense + .25 * ranking + .02 * output['fit_difficulty'][0, active].mean()
            + .05 * output['final_anatomy_fit'][0, active].mean()
            + (.5 * matching if stage == 'B' else 0.))
    return (loss, rigid_per[0, active].detach().mean(),
            dense_norm[active].detach().mean(), ranking.detach(), reach_fraction)


def save(step, feedback_enabled):
    path = run / f'joint_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(), 'numpy_rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'config': config, 'feedback_enabled': feedback_enabled, 'calibrated': False}, temporary)
    os.replace(temporary, path)


inner = []
inner_rng = np.random.default_rng(seed + 1)
with (run / 'inner_draws.jsonl').open('w') as inner_log:
    for base in range(60, 64):
        for slot in range(4):
            inner_appearance = 'v3' if slot == 3 else 'v4'
            inner_sampler = (sample_one_shot_slide_artifacts_v3 if inner_appearance == 'v3'
                             else sample_one_shot_slide_artifacts_v4)
            while True:
                draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
                    sample = inner_sampler(
                        context, [base], draw_seed, side, source_sample=original)
                valid_queries = int((F.grid_sample(sample['valid_mask'][:, None].float(), grid24,
                    mode='bilinear', align_corners=False) > .99).sum())
                used = bool(sample['eligible'][0]) and valid_queries >= 4
                inner_log.write(json.dumps({'base_index': base, 'slot': slot,
                    'draw_seed': draw_seed, 'appearance_version': inner_appearance,
                    'used': used, 'valid_queries': valid_queries,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
            chart = valid_chart(sample)
            with torch.no_grad():
                near, near_error = candidate(sample, chart, .35, 1.5, .9)
            inner.append({'sample': sample, 'near': near, 'near_initial_mm': near_error,
                          'chart': chart, 'base_index': base})
            del original


def inner_score(measure_gate=False):
    model.eval()
    rows = []
    with torch.inference_mode():
        for item in inner:
            sample, chart = item['sample'], item['chart']
            prediction = model.pose.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            branch = prior.topk(2, -1).indices
            modes, reflections = branch.div(2, rounding_mode='floor'), branch.remainder(2)
            output = model(sample['inputs'], context['atlas'], sample['offsets'],
                           sample['weights'], modes, reflections)
            choice = int(output['score_uncalibrated'][0].argmax())
            truth = rigid_points_090(sample['state'], sample['reflection'], chart)
            direct = rigid_points_090(prediction['state'][:, modes[0, 0]],
                reflections[:, 0], chart)
            proposed = rigid_points_090(prediction['state'][
                torch.arange(1, device='cuda')[:, None], modes], reflections, chart[None])
            corrected = rigid_points_090(output['corrected_state'][:, choice].float(),
                reflections[:, choice], chart)
            valid = sample['valid_mask'][0]
            dense = ((output['surface_ccf_um'][0, choice, valid].float()
                      - sample['centre'][0, valid]).norm(dim=-1).mean() / 1000)
            row = {'base_index': item['base_index'],
                   'blind_top2_has_near': float(bool((((proposed - truth[:, None]).norm(dim=-1)
                       .mean(-1) / 1000) <= 1.5).any())),
                   'direct_top1_mm': float((direct - truth).norm(dim=-1).mean() / 1000),
                   'selected_corrected_rigid_mm': float((corrected - truth).norm(dim=-1).mean() / 1000),
                   'selected_dense_mm': float(dense)}
            if measure_gate:
                states = torch.cat((sample['state'], item['near']))[None]
                known_reflection = sample['reflection'][:, None].expand(1, 2)
                fitted = model(sample['inputs'], context['atlas'], sample['offsets'],
                    sample['weights'], torch.zeros(1, 2, device='cuda', dtype=torch.long),
                    known_reflection, state_override=states)
                exact = rigid_points_090(fitted['corrected_state'][:, 0].float(),
                    sample['reflection'], chart)
                near = rigid_points_090(fitted['corrected_state'][:, 1].float(),
                    sample['reflection'], chart)
                row['exact_drift_mm'] = float((exact - truth).norm(dim=-1).mean() / 1000)
                row['near_initial_mm'] = item['near_initial_mm']
                row['near_corrected_mm'] = float((near - truth).norm(dim=-1).mean() / 1000)
            rows.append(row)
    model.train()
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in
        ('blind_top2_has_near', 'direct_top1_mm', 'selected_corrected_rigid_mm',
         'selected_dense_mm')}
    metrics['selection_score_mm'] = (metrics['direct_top1_mm']
        + metrics['selected_corrected_rigid_mm'] + metrics['selected_dense_mm'])
    if measure_gate:
        for key in ('exact_drift_mm', 'near_initial_mm', 'near_corrected_mm'):
            metrics[key] = float(np.mean([row[key] for row in rows]))
        metrics['near_gain_mm'] = metrics['near_initial_mm'] - metrics['near_corrected_mm']
        metrics['feedback_gate_passed'] = (metrics['near_gain_mm'] >= .1
            and metrics['exact_drift_mm'] <= .25)
    return metrics, rows


save(0, False)
started, attempts, accepted = time.perf_counter(), 0, 0
feedback_enabled = False
scores = {}
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training, \
        (run / 'inner_scores.jsonl').open('w') as selections:
    for step in range(1, updates + 1):
        stage = ('A' if step <= stage_a_end else
                 'C' if step > stage_b_end and feedback_enabled else 'B')
        optimizer.param_groups[0]['lr'] = {'A': 1e-4, 'B': 5e-5, 'C': 1e-5}[stage]
        optimizer.param_groups[1]['lr'] = {'A': 2e-4, 'B': 1e-4, 'C': 5e-5}[stage]
        optimizer.param_groups[2]['lr'] = {'A': 0., 'B': 2e-4, 'C': 1e-4}[stage]
        appearance = 'v3' if step % 3 == 0 else 'v4'
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        while True:
            base = int(rng.integers(60))
            draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            with torch.no_grad():
                original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
                sample = sampler(context, [base], draw_seed, side, source_sample=original)
            attempts += 1
            valid_queries = int((F.grid_sample(sample['valid_mask'][:, None].float(), grid24,
                mode='bilinear', align_corners=False) > .99).sum())
            used = bool(sample['eligible'][0]) and (stage == 'A' or valid_queries >= 4)
            draws.write(json.dumps({'update': step, 'stage': stage, 'appearance_version': appearance,
                'attempt': attempts, 'base_index': base, 'draw_seed': draw_seed,
                'used': used, 'valid_queries': valid_queries,
                'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
            if used:
                break
            del sample, original
        accepted += 1
        del original
        chart = valid_chart(sample, 128)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.pose.predict(sample['inputs'])
        direct, prior = direct_loss(sample, prediction, chart)
        on_policy = prior.detach().topk(2, -1).indices
        with torch.no_grad():
            selected_modes = on_policy.div(2, rounding_mode='floor')
            selected_reflections = on_policy.remainder(2)
            selected_states = prediction['state'][torch.arange(1, device='cuda')[:, None],
                                                  selected_modes].detach()
            true_rigid = rigid_points_090(sample['state'], sample['reflection'], chart)
            proposed_rigid = rigid_points_090(selected_states, selected_reflections,
                                               chart[None])
            on_policy_near = ((proposed_rigid - true_rigid[:, None]).norm(dim=-1)
                              .mean(-1)[0] / 1000 <= 1.5)
            blind_top2_near_count = int(on_policy_near.sum())
        scaler.scale(direct).backward()
        del prediction, prior

        anchor_value = 0.
        if valid_queries >= 4:
            with torch.autocast('cuda', dtype=torch.float16):
                match = model.matcher(sample['inputs'], context['atlas'],
                    sample['state'][:, None], sample['reflection'][:, None],
                    sample['offsets'], sample['weights'])
            anchor, _ = matcher_loss(sample, match)
            scaler.scale(anchor if stage == 'A' else .5 * anchor).backward()
            anchor_value = float(anchor.detach())
            del match, anchor

        joint_value = rigid_value = dense_value = ranking_value = 0.
        joint_reachable_count = 0
        if stage == 'B' or (stage == 'C' and blind_top2_near_count):
            if stage == 'B':
                with torch.no_grad():
                    near, _ = candidate(sample, chart, .35, 1.5, .9)
                    wrong, _ = candidate(sample, chart, 1.5, 3., 2.2, wrong=True)
                first = sample['state'] if step % 4 == 0 else near
                second = near if step % 4 == 2 else wrong
                states = torch.stack((first[0], second[0]), 0)[None]
                reflections = sample['reflection'][:, None].expand(1, 2).clone()
                if step % 2 == 0:
                    reflections[:, 1] = 1 - reflections[:, 1]
                modes = torch.zeros(1, 2, device='cuda', dtype=torch.long)
                with torch.no_grad():
                    proposed = rigid_points_090(states, reflections, chart[None])
                    near = ((proposed - true_rigid[:, None]).norm(dim=-1).mean(-1)[0]
                            / 1000 <= 1.5)
                model.pose.requires_grad_(False)
            else:
                branch = on_policy[0, on_policy_near][None]
                modes = branch.div(2, rounding_mode='floor')
                reflections = branch.remainder(2)
                states = None
                near = torch.ones(branch.shape[1], device='cuda', dtype=torch.bool)
            output = model(sample['inputs'], context['atlas'], sample['offsets'],
                sample['weights'], modes, reflections, state_override=states)
            joint, rigid_value, dense_value, ranking_value, reach_fraction = joint_loss(
                sample, output, chart, stage, reflections, near)
            joint_reachable_count = int((reach_fraction >= .5).sum())
            if joint is not None:
                scaler.scale(joint).backward()
                joint_value = float(joint.detach())
            if stage == 'B':
                model.pose.requires_grad_(True)
            del output, joint

        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        row = {'update': step, 'stage': stage, 'base_index': base,
            'physical_section_id': sample['provenance'][0]['physical_section_id'],
            'attempted': attempts, 'accepted': accepted, 'valid_queries': valid_queries,
            'blind_top2_near_count': blind_top2_near_count,
            'joint_reachable_count': joint_reachable_count,
            'direct_loss': float(direct.detach()), 'matcher_anchor_loss': anchor_value,
            'joint_loss': joint_value, 'corrected_rigid_mm': float(rigid_value),
            'dense_ccf_mm': float(dense_value), 'ranking_loss': float(ranking_value),
            'gradient_norm': float(gradient), 'elapsed_seconds': time.perf_counter() - started}
        training.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 100 == 0:
            training.flush()
            draws.flush()
            print(json.dumps({'event': 'train_milestone', **row}), flush=True)
        if step in checkpoints:
            save(step, feedback_enabled)
            metrics, inner_rows = inner_score(measure_gate=step == stage_b_end)
            if step == stage_b_end:
                feedback_enabled = metrics['feedback_gate_passed']
                save(step, feedback_enabled)
            scores[step] = metrics['selection_score_mm']
            selections.write(json.dumps({'step': step, 'metrics': metrics,
                'rows': inner_rows}, allow_nan=False) + '\n')
            selections.flush()
            print(json.dumps({'event': 'inner_checkpoint', 'step': step,
                              'metrics': metrics}), flush=True)
        del sample, direct

best = min(scores, key=scores.get)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'selected_step': best, 'selected_inner_score_mm': scores[best],
    'feedback_gate_passed': feedback_enabled, 'attempted': attempts,
    'accepted_physical_sections': accepted,
    'protocol_sha256': sha(run / 'protocol.json'),
    'config_sha256': sha(run / 'config.json'),
    'inner_draws_sha256': sha(run / 'inner_draws.jsonl'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'inner_scores_sha256': sha(run / 'inner_scores.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in (0, *checkpoints)},
    'source_sha256': config['source_sha256'], 'calibrated': False,
    'pretrained_weights_used': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'selected_step': best,
                  'feedback_gate_passed': feedback_enabled}), flush=True)
