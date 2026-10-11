"""Train the 154 checkpoint's atlas fitter before allowing fit-to-pose gradients.

Every TRAIN presentation is a new independently sampled physical plane. The
pose predictor is frozen; exact-plane preservation and near-plane correction
are tested on four disjoint held-out TRAIN deformation plans.
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

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


seed, side, updates = 20261011155, 256, 6000
checkpoints = (0, 2000, 4000, 6000)
source = Path(__file__).resolve().parent
parent_run = root / 'runs/joint_pose_curriculum_154_train151_001'
parent_done = json.loads((parent_run / 'completed.json').read_text())
parent_step = parent_done['selected_step']
parent_checkpoint = parent_run / f'joint_pose_step_{parent_step:05d}.pt'
run = root / 'runs/joint_fitter_155_train151_001'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256'][str(parent_step)]

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
perturb_rng = torch.Generator(device='cuda').manual_seed(seed + 2)
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert all(base['lineage']['split'] == 'train' for base in context['bases'])
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
model = JointAnatomyModel153().cuda()
model.load_state_dict(parent['model'], strict=True)
del parent
model.pose.requires_grad_(False)
model.train()
model.pose.eval()
optimizer = torch.optim.AdamW([
    {'params': model.matcher.parameters(), 'lr': 1e-4},
    {'params': list(model.field_body.parameters()) + list(model.field_out.parameters()),
     'lr': 1e-4},
], weight_decay=1e-4)

protocol = {'seed': seed, 'updates': updates, 'side': side,
    'parent_run': str(parent_run), 'parent_step': parent_step,
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'gradient_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'inner_cases_per_base': 12, 'checkpoints': checkpoints,
    'appearance_schedule': 'v4,v4,v3; one new geometry/appearance per accepted presentation',
    'candidate_schedule': 'exact/near, exact/wrong, near/wrong, exact/on-policy top16',
    'near_error_mm': [.35, 1.5], 'wrong_error_mm': [1.5, 3.],
    'inner_gate': {'exact_mean_drift_mm_at_most': .25,
                   'near_mean_gain_mm_at_least': .10,
                   'exact_energy_below_wrong_fraction_at_least': .70},
    'pose_frozen': True, 'calibrated': False,
    'external_pretrained_weights_used': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False}
source_names = ('train_joint_fitter_155.py', 'joint_anatomy_model_153.py',
    'coarse_atlas_pose_150.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_recurrent_model.py', 'global_atlas_contrast_090.py')
config = {'protocol': protocol, 'source_sha256': {name: sha(source / name)
    for name in source_names}, 'synthetic_provenance': context['provenance'],
    'torch': torch.__version__, 'numpy': np.__version__}
run.mkdir(parents=True, exist_ok=False)
(run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
config['protocol_sha256'] = sha(run / 'protocol.json')
(run / 'config.json').write_text(json.dumps(config, indent=2))

axis24 = (torch.arange(24, device='cuda') + .5) / 24
y24, x24 = torch.meshgrid(axis24, axis24, indexing='ij')
grid24 = (2 * torch.stack((x24, y24), -1) - 1)[None]
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')


def draw(base, draw_seed, appearance):
    sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
               else sample_one_shot_slide_artifacts_v4)
    with torch.no_grad():
        original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
        sample = sampler(context, [base], draw_seed, side, source_sample=original)
        valid_queries = int((F.grid_sample(sample['valid_mask'][:, None].float(), grid24,
            mode='bilinear', align_corners=False) > .99).sum())
    return sample, valid_queries


def candidate(sample, chart, lower, upper, target, generator, wrong=False):
    truth = rigid_points_090(sample['state'], sample['reflection'], chart)
    while True:
        delta = (2 * torch.rand(64, 9, device='cuda', generator=generator) - 1) * limits
        if wrong:
            delta[:, :6] *= 3
            delta[:, 6:] *= 2
        states = compose_full_frame_state(sample['state'].expand(64, -1), delta)
        errors = (rigid_points_090(states, sample['reflection'].expand(64), chart)
                  - truth).norm(dim=-1).mean(-1) / 1000
        accepted = (errors >= lower) & (errors <= upper)
        if bool(accepted.any()):
            index = int((errors - target).abs().masked_fill(~accepted, 10).argmin())
            return states[index:index + 1]


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
    return key_ce + .5 * expected + .2 * visibility + .7 * reliability


def physical(sample, output, reflections, chart):
    true = rigid_points_090(sample['state'], sample['reflection'], chart)
    proposed = rigid_points_090(output['candidate_state'].float(),
                                reflections, chart[None, None])
    corrected = rigid_points_090(output['corrected_state'].float(),
                                 reflections, chart[None, None])
    initial = (proposed - true[:, None]).norm(dim=-1).mean(-1)[0] / 1000
    rigid = (corrected - true[:, None]).norm(dim=-1).mean(-1)[0] / 1000
    valid = sample['valid_mask'][0]
    dense = ((output['surface_ccf_um'][0, :, valid].float()
              - sample['centre'][0, valid]).norm(dim=-1).mean(-1) / 1000)
    return initial, rigid, dense


inner = []
inner_rng = np.random.default_rng(seed + 1)
with (run / 'inner_draws.jsonl').open('w') as stream:
    for base in range(60, 64):
        for slot in range(12):
            appearance = 'v3' if slot % 3 == 0 else 'v4'
            while True:
                draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
                sample, valid_queries = draw(base, draw_seed, appearance)
                used = bool(sample['eligible'][0]) and valid_queries >= 4
                stream.write(json.dumps({'base_index': base, 'slot': slot,
                    'draw_seed': draw_seed, 'appearance_version': appearance,
                    'valid_queries': valid_queries, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    inner.append((base, draw_seed, appearance, seed + base * 100 + slot))
                    break
                del sample
            del sample


def inner_score():
    model.eval()
    rows = []
    with torch.inference_mode():
        for base, draw_seed, appearance, perturb_seed in inner:
            sample, _ = draw(base, draw_seed, appearance)
            pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(512, len(pixels)),
                device='cuda').long()]
            chart = torch.stack((pixels.remainder(side),
                pixels.div(side, rounding_mode='floor')), -1).float() / side
            generator = torch.Generator(device='cuda').manual_seed(perturb_seed)
            near = candidate(sample, chart, .35, 1.5, .9, generator)
            wrong = candidate(sample, chart, 1.5, 3., 2.2, generator, wrong=True)
            states = torch.stack((sample['state'][0], near[0], wrong[0]))[None]
            reflections = sample['reflection'][:, None].expand(1, 3)
            output = model(sample['inputs'], context['atlas'], sample['offsets'],
                           sample['weights'], torch.zeros(1, 3, device='cuda', dtype=torch.long),
                           reflections, state_override=states)
            initial, rigid, dense = physical(sample, output, reflections, chart)
            rows.append({'base_index': base,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'appearance_version': appearance, 'valid_fraction': float(sample['valid_mask'].float().mean()),
                'initial_mm': initial.tolist(), 'corrected_mm': rigid.tolist(),
                'dense_mm': dense.tolist(),
                'fit_energy_uncalibrated': output['fit_energy_uncalibrated'][0].tolist(),
                'exact_energy_below_wrong': float(output['fit_energy_uncalibrated'][0, 0]
                                                  < output['fit_energy_uncalibrated'][0, 2])})
            del sample, output
    model.train()
    model.pose.eval()
    metrics = {'exact_drift_mm': float(np.mean([r['corrected_mm'][0] for r in rows])),
        'near_initial_mm': float(np.mean([r['initial_mm'][1] for r in rows])),
        'near_corrected_mm': float(np.mean([r['corrected_mm'][1] for r in rows])),
        'near_dense_mm': float(np.mean([r['dense_mm'][1] for r in rows])),
        'exact_energy_below_wrong': float(np.mean([r['exact_energy_below_wrong'] for r in rows]))}
    metrics['near_gain_mm'] = metrics['near_initial_mm'] - metrics['near_corrected_mm']
    metrics['feedback_gate_passed'] = (metrics['exact_drift_mm'] <= .25
        and metrics['near_gain_mm'] >= .10 and metrics['exact_energy_below_wrong'] >= .70)
    metrics['selection_score'] = (metrics['exact_drift_mm'] + metrics['near_corrected_mm']
                                  + .5 * metrics['near_dense_mm']
                                  + (1 - metrics['exact_energy_below_wrong']))
    return metrics, rows


def save(step):
    path = run / f'joint_fitter_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config,
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'perturb_rng': perturb_rng.get_state(), 'calibrated': False}, temporary)
    os.replace(temporary, path)


scores = {}
save(0)
started, attempts = time.perf_counter(), 0
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training, \
        (run / 'inner_scores.jsonl').open('w') as selections:
    for step in range(1, updates + 1):
        appearance = 'v3' if step % 3 == 0 else 'v4'
        while True:
            base = int(rng.integers(60))
            draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            sample, valid_queries = draw(base, draw_seed, appearance)
            attempts += 1
            used = bool(sample['eligible'][0]) and valid_queries >= 4
            draws.write(json.dumps({'update': step, 'attempt': attempts,
                'base_index': base, 'draw_seed': draw_seed,
                'appearance_version': appearance, 'valid_queries': valid_queries,
                'used': used, 'provenance': sample['provenance'][0]},
                allow_nan=False) + '\n')
            if used:
                break
            del sample

        pixels = torch.multinomial(sample['valid_mask'][0].flatten().float(),
                                   256, replacement=True)
        chart = torch.stack((pixels.remainder(side),
            pixels.div(side, rounding_mode='floor')), -1).float() / side
        near = candidate(sample, chart, .35, 1.5, .9, perturb_rng)
        wrong = candidate(sample, chart, 1.5, 3., 2.2,
                          perturb_rng, wrong=True)
        if step % 4 == 0:
            with torch.no_grad():
                prediction = model.pose.predict(sample['inputs'])
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                branch = prior.topk(16, -1).indices[0]
                states16 = prediction['state'][0, branch // 2]
                flags16 = branch % 2
                true = rigid_points_090(sample['state'], sample['reflection'], chart)
                errors16 = ((rigid_points_090(states16, flags16, chart)
                             - true).norm(dim=-1).mean(-1) / 1000)
                wrong_index = int(errors16.argmax())
                second = states16[wrong_index:wrong_index + 1]
                second_reflection = flags16[wrong_index:wrong_index + 1]
            states = torch.stack((sample['state'][0], second[0]))[None]
            reflections = torch.stack((sample['reflection'][0], second_reflection[0]))[None]
        else:
            states = {1: torch.stack((sample['state'][0], near[0])),
                      2: torch.stack((sample['state'][0], wrong[0])),
                      3: torch.stack((near[0], wrong[0]))}[step % 4][None]
            reflections = sample['reflection'][:, None].expand(1, 2).clone()
            if step % 4 == 3 and step % 8 == 3:
                reflections[:, 1] = 1 - reflections[:, 1]

        output = model(sample['inputs'], context['atlas'], sample['offsets'],
            sample['weights'], torch.zeros(1, 2, device='cuda', dtype=torch.long),
            reflections, state_override=states)
        initial, rigid, dense = physical(sample, output, reflections, chart)
        positive = initial <= 1.5
        fit = F.smooth_l1_loss(rigid[positive], torch.zeros_like(rigid[positive]),
                              beta=.25)
        dense_fit = F.smooth_l1_loss(dense[positive], torch.zeros_like(dense[positive]),
                                    beta=.25)
        exact = initial < 1e-4
        if bool(exact.any()):
            fit = fit + F.smooth_l1_loss(rigid[exact], torch.zeros_like(rigid[exact]), beta=.25)
            dense_fit = dense_fit + F.smooth_l1_loss(dense[exact],
                                                     torch.zeros_like(dense[exact]), beta=.25)
        rank_target = F.softmax(-(.5 * rigid.detach() + .5 * dense.detach()) / .5, -1)
        ranking = F.kl_div(F.log_softmax(-output['fit_energy_uncalibrated'][0].float(), -1),
                           rank_target, reduction='sum')
        matching = matcher_loss(sample, output['matcher'])
        loss = (.5 * matching + fit + dense_fit + .25 * ranking
                + .02 * output['fit_difficulty'][0, positive].mean()
                + .05 * output['final_anatomy_fit'][0, positive].mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            list(model.matcher.parameters()) + list(model.field_body.parameters())
            + list(model.field_out.parameters()), 2., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'base_index': base,
            'physical_section_id': sample['provenance'][0]['physical_section_id'],
            'appearance_version': appearance, 'valid_queries': valid_queries,
            'attempted': attempts, 'fit_loss': float(loss.detach()),
            'matcher_loss': float(matching.detach()),
            'rigid_mm': rigid.detach().tolist(), 'dense_mm': dense.detach().tolist(),
            'initial_mm': initial.detach().tolist(), 'ranking_loss': float(ranking.detach()),
            'gradient_norm': float(gradient), 'elapsed_seconds': time.perf_counter() - started}
        training.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            draws.flush()
            training.flush()
            print(json.dumps({'event': 'train_milestone', **row}), flush=True)
        if step in checkpoints:
            save(step)
            metrics, inner_rows = inner_score()
            scores[step] = metrics
            selections.write(json.dumps({'step': step, 'metrics': metrics,
                'rows': inner_rows}, allow_nan=False) + '\n')
            selections.flush()
            print(json.dumps({'event': 'inner_checkpoint', 'step': step,
                              'metrics': metrics}), flush=True)
        del sample, output, loss

eligible = [step for step, row in scores.items() if row['feedback_gate_passed']]
selected = (min(eligible, key=lambda step: scores[step]['selection_score'])
            if eligible else min(scores, key=lambda step: scores[step]['selection_score']))
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_physical_sections': updates, 'attempted': attempts,
    'selected_step': selected, 'feedback_gate_passed': bool(eligible),
    'selected_inner_metrics': scores[selected],
    'parent_checkpoint_sha256': protocol['parent_checkpoint_sha256'],
    'protocol_sha256': sha(run / 'protocol.json'),
    'config_sha256': sha(run / 'config.json'),
    'inner_draws_sha256': sha(run / 'inner_draws.jsonl'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'inner_scores_sha256': sha(run / 'inner_scores.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_fitter_step_{step:05d}.pt')
        for step in checkpoints}, 'source_sha256': config['source_sha256'],
    'calibrated': False, 'external_pretrained_weights_used': False,
    'real_images_used': False, 'dev_or_public_benchmark_used': False,
    'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'selected_step': selected,
                  'feedback_gate_passed': bool(eligible),
                  'selected_inner_metrics': scores[selected]}), flush=True)
