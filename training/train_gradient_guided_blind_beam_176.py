"""Train a small blind-beam action ranker over explicit atlas-fit corrections."""

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
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.gradient_guided_plane_fit_176 import (
    FitEvidenceRanker176, GradientGuidedPlaneFitter176)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(len(beam), device=beam.device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(len(beam), 64, 2)[
            torch.arange(len(beam), device=beam.device), anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


def chart_for(valid, count):
    pixels = valid[0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(count, len(pixels)),
        device=pixels.device).round().long()]
    return torch.stack((pixels.remainder(256),
        pixels.div(256, rounding_mode='floor')), -1).float()[None] / 256


def physical_cost(states, reflection, sample, chart, corners):
    target = rigid_points_090(sample['state'], sample['reflection'], chart)
    predicted = rigid_points_090(states, reflection, chart[:, None])
    valid = (predicted - target[:, None]).norm(dim=-1).mean(-1) / 1000
    target_five = rigid_points_090(sample['state'], sample['reflection'], corners)
    predicted_five = rigid_points_090(states, reflection, corners)
    five = (predicted_five - target_five[:, None]).norm(dim=-1).mean(-1) / 1000
    return .75 * valid + .25 * five


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GRADIENT_GUIDED_BLIND_BEAM_176_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = Path(os.environ.get('AT176_RUN', str(root / 'runs/gradient_guided_blind_beam_176_train_001'))).resolve()
resume = os.environ.get('AT176_RESUME')
preflight = os.environ.get('AT176_PREFLIGHT', '0') == '1'
steps = int(os.environ.get('AT176_STEPS', '500'))
seed, panel_seed = 202610176, 202610177
assert source.drive.upper() == run.drive.upper() == root.drive.upper() == 'I:'
assert steps > 0 and torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
model.load_state_dict(parent['model'], strict=True)
del parent
fitter = GradientGuidedPlaneFitter176().cuda().eval()
ranker = FitEvidenceRanker176().cuda().train()
optimizer = torch.optim.AdamW(ranker.parameters(), lr=2e-4, weight_decay=1e-4)
parameters = list(ranker.parameters())
start_step = 0
resume_sha = None
if resume:
    resume_path = Path(resume).resolve()
    assert resume_path.drive.upper() == 'I:'
    resume_done = json.loads((resume_path.parent / 'completed.json').read_text())
    assert sha(resume_path) == resume_done['output_sha256'][resume_path.name]
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        previous = torch.load(resume_path, map_location='cpu', weights_only=True)
    start_step = previous['step']
    ranker.load_state_dict(previous['ranker'], strict=True)
    optimizer.load_state_dict(previous['optimizer'])
    resume_sha = sha(resume_path)
    assert steps > start_step
    del previous
rng = np.random.default_rng(seed + start_step)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / 256


def one_case(sample):
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        states = prediction['state'][:, ids // 2]
        reflection = (ids % 2)[None]
    fitted = fitter(sample['inputs'], states, reflection, sample['offsets'],
                    sample['weights'], atlas)
    evidence = {key: value.detach() for key, value in fitted.items()}
    logits = ranker(prior[:, ids].detach(), evidence)
    with torch.no_grad():
        chart = chart_for(sample['valid_mask'], 128)
        original_cost = physical_cost(states, reflection, sample, chart, corners)
        corrected_cost = physical_cost(fitted['state'].detach(), reflection, sample,
            chart, corners)
        action_costs = torch.stack((original_cost, corrected_cost), -1)
        action_costs[..., 1].masked_fill_(fitted['support_sites'] < 100, 1e4)
        target = F.softmax(-action_costs.flatten(1) / .7, -1)
        selected = int(logits.detach().flatten().argmax())
        prior_choice = int(prior[0, ids].argmax())
        action_cost = action_costs.flatten()
    loss = F.kl_div(F.log_softmax(logits.flatten(1), -1), target, reduction='batchmean')
    return loss, {'physical_section_id': sample['provenance'][0]['physical_section_id'],
        'appearance': 'v4' if 'one_shot_slide_artifacts_v4' in sample['provenance'][0] else 'v3',
        'rank_loss': float(loss.detach()), 'original_prior_mm': float(original_cost[0, prior_choice]),
        'selected_action_mm': float(action_cost[selected]),
        'best_available_mm': float(action_cost.min()),
        'selected_capture_1p5': int(action_cost[selected] <= 1.5),
        'original_capture_1p5': int(original_cost[0, prior_choice] <= 1.5),
        'selected_action': selected, 'selected_branch': int(ids[selected // 2]),
        'fit_sites_mean': float(fitted['support_sites'].float().mean()),
        'valid_corrected_actions': int((fitted['support_sites'] >= 100).sum()),
        'best_original_mm': float(original_cost.min()),
        'best_corrected_mm': float(action_costs[..., 1].min()),
        'mean_fit_gain': float((fitted['fit'] - fitted['original_fit']).mean())}


def inner_evaluate():
    ranker.eval()
    rows = []
    for base in range(64):
        trial = 0
        while True:
            draw_seed = panel_seed * 1000000 + 1000 * base + trial
            geometry = sample_panel_geometry_train_151 if base % 5 == 0 else sample_panel_geometry_train_166
            sampler = sample_one_shot_slide_artifacts_v4 if base % 2 else sample_one_shot_slide_artifacts_v3
            with torch.no_grad():
                original = geometry(context, [base], draw_seed, 256)
                sample = sampler(context, [base], draw_seed, 256, source_sample=original)
            if bool(sample['eligible'][0]):
                break
            trial += 1
        with torch.no_grad():
            prediction = model.predict(sample['inputs'])
            prior, beam = beam_for(prediction)
            ids = beam[0]
            states = prediction['state'][:, ids // 2]
            reflection = (ids % 2)[None]
        fitted = fitter(sample['inputs'], states, reflection, sample['offsets'],
                        sample['weights'], atlas)
        with torch.no_grad():
            logits = ranker(prior[:, ids], fitted)
            chart = chart_for(sample['valid_mask'], 128)
            original_cost = physical_cost(states, reflection, sample, chart, corners)
            corrected_cost = physical_cost(fitted['state'], reflection, sample, chart, corners)
            costs = torch.stack((original_cost, corrected_cost), -1)
            costs[..., 1].masked_fill_(fitted['support_sites'] < 100, 1e4)
            costs = costs.flatten()
            selected = int(logits.flatten().argmax())
            prior_choice = int(prior[0, ids].argmax())
            rows.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'selected_action_mm': float(costs[selected]),
                'original_prior_mm': float(original_cost[0, prior_choice]),
                'best_available_mm': float(costs.min()),
                'selected_capture_1p5': int(costs[selected] <= 1.5),
                'original_capture_1p5': int(original_cost[0, prior_choice] <= 1.5)})
    ranker.train()
    return {'sections': len(rows), 'selected_action_mm': float(np.mean([
        x['selected_action_mm'] for x in rows])), 'original_prior_mm': float(np.mean([
        x['original_prior_mm'] for x in rows])),
        'best_available_mm': float(np.mean([x['best_available_mm'] for x in rows])),
        'selected_capture_1p5': sum(x['selected_capture_1p5'] for x in rows),
        'original_capture_1p5': sum(x['original_capture_1p5'] for x in rows)}


if preflight:
    trial = 0
    while True:
        with torch.no_grad():
            original = sample_panel_geometry_train_166(context, [0], seed + trial, 256)
            sample = sample_one_shot_slide_artifacts_v4(context, [0], seed + trial, 256,
                source_sample=original)
        if bool(sample['eligible'][0]):
            break
        trial += 1
    loss, detail = one_case(sample)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    assert float(gradient) > 0 and all(p.grad is None for p in model.parameters())
    print(json.dumps({'preflight': True, 'loss': float(loss.detach()),
        'gradient_norm': float(gradient), 'detail': detail}, allow_nan=False), flush=True)
    raise SystemExit

source_files = ('train_gradient_guided_blind_beam_176.py', 'gradient_guided_plane_fit_176.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_panel_geometry_train_166.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_full_frame_primitives.py', 'global_atlas_contrast_090.py')
config = {'seed': seed + start_step, 'panel_seed': panel_seed,
    'start_update': start_step, 'end_update': steps,
    'resume_checkpoint_sha256': resume_sha,
    'synthetic_per_update': {'v4': 1, 'v3': 1}, 'accepted_bases': 64,
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'parent_checkpoint_sha256': sha(parent_path),
    'synthetic_provenance': context['provenance'],
    'beam': 'blind top8 base + top6 anchor + two diverse anchor reflections',
    'trainable': 'fresh small action ranker only; parent and direct map frozen',
    'loss': 'KL to geometry-soft 32-action physical pose target; truth only labels blind candidates',
    'fitter': 'two first-order bounded local gradient steps on smooth absolute image/atlas correlation',
    'calibrated': False, 'real_training_images': 0, 'expert_real_truth_used': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
assert not run.exists()
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))


def save(step):
    target = run / f'ranker_step_{step:05d}.pt'
    temp = target.with_suffix('.tmp')
    torch.save({'step': step, 'ranker': ranker.state_dict(),
        'optimizer': optimizer.state_dict(), 'config': config,
        'calibrated': False}, temp)
    os.replace(temp, target)


save(start_step)
started, attempt, bases = time.perf_counter(), 0, [0] * 64
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training, \
     (run / 'inner_scores.jsonl').open('w') as inner:
    for step in range(start_step + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        cases = []
        for appearance in ('v4', 'v3'):
            sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v3
            while True:
                attempt += 1
                base = int(rng.integers(64))
                draw_seed = int(rng.integers(0, 2**63 - 1))
                geometry = sample_panel_geometry_train_151 if rng.random() < .2 else sample_panel_geometry_train_166
                with torch.no_grad():
                    original = geometry(context, [base], draw_seed, 256)
                    sample = sampler(context, [base], draw_seed, 256, source_sample=original)
                eligible = bool(sample['eligible'][0])
                draws.write(json.dumps({'update': step, 'attempt': attempt,
                    'appearance': appearance, 'base_index': base, 'draw_seed': draw_seed,
                    'geometry': geometry.__name__, 'used': eligible,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if eligible:
                    bases[base] += 1
                    break
            loss, detail = one_case(sample)
            (loss / 2).backward()
            cases.append(detail)
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.param_groups[0]['lr'] = 2e-4 * (.2 + .8 * .5 *
            (1 + math.cos(math.pi * min(step, 2000) / 2000)))
        optimizer.step()
        training.write(json.dumps({'update': step, 'cases': cases,
            'gradient_norm': float(gradient), 'lr': optimizer.param_groups[0]['lr'],
            'seconds_since_start': time.perf_counter() - started}, allow_nan=False) + '\n')
        if step % 500 == 0 or step == steps:
            metrics = inner_evaluate()
            inner.write(json.dumps({'update': step, **metrics}, allow_nan=False) + '\n')
            save(step)
            print(json.dumps({'event': 'milestone', 'update': step, **metrics}), flush=True)
        if step % 50 == 0:
            draws.flush()
            training.flush()

assert sum(bases) == 2 * (steps - start_step)
(run / 'completed.json').write_text(json.dumps({'start_update': start_step,
    'end_update': steps,
    'accepted_synthetic': sum(bases), 'accepted_base_counts': bases,
    'attempts': attempt, 'config_sha256': sha(run / 'config.json'),
    'output_sha256': {p.name: sha(p) for p in run.iterdir()
        if p.is_file() and p.name != 'completed.json'},
    'calibrated': False, 'real_training_images': 0,
    'expert_real_truth_used': False, 'public_benchmark_used': False,
    'external_pretrained_weights_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'updates': steps,
    'seconds': time.perf_counter() - started}), flush=True)
