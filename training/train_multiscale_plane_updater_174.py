"""TRAIN-only first-stage correction of frozen 132 blind arbitrary-plane proposals.

Set AT174_PREFLIGHT=1 for one forward/backward without creating a run. Set
AT174_STEPS and AT174_RUN before a new run; interrupted runs resume from latest.pt.
"""

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
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.multiscale_plane_updater_174 import MultiscalePlaneUpdater174


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/MULTISCALE_POSE_UPDATER_174_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = Path(os.environ.get('AT174_RUN', str(root / 'runs/multiscale_plane_updater_174_train_001'))).resolve()
preflight = os.environ.get('AT174_PREFLIGHT', '0') == '1'
steps = int(os.environ.get('AT174_STEPS', '2000'))
seed, panel_seed, side, sites = 202610174, 202610175, 256, 128
assert source.drive.upper() == run.drive.upper() == root.drive.upper() == 'I:'
assert steps > 0 and torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert not any(parent_done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items()
           if name not in ('arbitrary_plane_one_shot_model.py',
                           'arbitrary_plane_one_shot_slide_artifacts_v3.py'))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
assert parent['config'] == parent_config
model.load_state_dict(parent['model'], strict=True)
del parent
updater = MultiscalePlaneUpdater174().cuda().train()
optimizer = torch.optim.AdamW(updater.parameters(), lr=1e-4, weight_decay=1e-4)
parameters = list(updater.parameters())
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
update_limits = torch.tensor((.55, .55, .55, 2200., 2200., 2200., .18, .18, .15), device='cuda')


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
    return torch.stack((pixels.remainder(side),
        pixels.div(side, rounding_mode='floor')), -1).float()[None] / side


def physical_cost(states, reflection, sample, chart):
    target = rigid_points_090(sample['state'], sample['reflection'], chart)
    predicted = rigid_points_090(states, reflection, chart[:, None])
    valid = (predicted - target[:, None]).norm(dim=-1).mean(-1) / 1000
    target_five = rigid_points_090(sample['state'], sample['reflection'], corners)
    predicted_five = rigid_points_090(states, reflection, corners)
    five = (predicted_five - target_five[:, None]).norm(dim=-1).mean(-1) / 1000
    return .75 * valid + .25 * five


def train_case(sample):
    # No truth, validity, or section identity enters the proposal beam.
    with torch.no_grad():
        prediction = model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        states = prediction['state'][0, ids // 2]
        flags = ids.remainder(2)
        chart = chart_for(sample['valid_mask'], sites)
        initial = physical_cost(states[None], flags[None], sample, chart)[0]
        closest = int(initial.argmin())
        top = int(prior[0, ids].argmax())
        wrong = (initial >= initial[closest] + .75).nonzero()[:, 0]
        hard = int(wrong[prior[0, ids[wrong]].argmax()]) if len(wrong) else int(initial.argmax())
        selected = list(dict.fromkeys((closest, top, hard)))
        for slot in prior[0, ids].argsort(descending=True):
            if len(selected) == 3:
                break
            if int(slot) not in selected:
                selected.append(int(slot))
        positions = [selected.index(slot) for slot in (closest, top, hard)]
        jitter_limits = sample['state'].new_tensor((.2, .2, .2, 1200., 1200., 1200., .08, .08, .06))
        jitter = (2 * torch.rand(1, 9, device='cuda') - 1) * jitter_limits
        teacher = compose_full_frame_state(sample['state'], jitter)
        candidates = torch.cat((states[selected][None], teacher[:, None]), 1)
        reflection = torch.cat((flags[selected][None], sample['reflection'][:, None]), 1)
        selected_prior = prior[0, ids[selected]]
        initial_selected = physical_cost(candidates, reflection, sample, chart)[0]

    output = updater(sample['inputs'], candidates, reflection, sample['offsets'],
                     sample['weights'], atlas)
    coarse = physical_cost(output['coarse_state'], reflection, sample, chart)[0]
    corrected = physical_cost(output['state'], reflection, sample, chart)[0]
    weight = .2 + .8 * (-(initial_selected.detach() / 4).square()).exp()
    weight[-1] = 1
    pose_loss = ((.4 * coarse + corrected) * weight).sum() / weight.sum()
    action_score = torch.stack((selected_prior, selected_prior + output['score'][0, :3]), -1).flatten()
    action_cost = torch.stack((initial_selected[:3], corrected[:3]), -1).flatten().detach()
    rank_loss = F.kl_div(F.log_softmax(action_score, -1),
                         F.softmax(-action_cost / .7, -1), reduction='sum')
    teacher_target = ((initial_selected[3] - corrected[3]).detach() / .4).sigmoid()
    teacher_score = F.binary_cross_entropy_with_logits(output['score'][0, 3], teacher_target)
    effort = (output['coarse_update'] / update_limits).square().mean() \
        + (output['fine_update'] / update_limits).square().mean()
    loss = pose_loss + .4 * rank_loss + .1 * teacher_score + .01 * effort
    return loss, {'pose_mm': float(pose_loss.detach()), 'rank': float(rank_loss.detach()),
        'teacher_score': float(teacher_score.detach()), 'effort': float(effort.detach()),
        'closest_initial_mm': float(initial[closest]), 'prior_initial_mm': float(initial[top]),
        'hard_initial_mm': float(initial[hard]),
        'closest_corrected_mm': float(corrected[positions[0]].detach()),
        'prior_corrected_mm': float(corrected[positions[1]].detach()),
        'hard_corrected_mm': float(corrected[positions[2]].detach()),
        'teacher_initial_mm': float(initial_selected[3]),
        'teacher_corrected_mm': float(corrected[3].detach()),
        'blind_branch_ids': ids.tolist(), 'selected_slots': selected}


def evaluate():
    updater.eval()
    errors, direct, oracle, captured, direct_captured = [], [], [], 0, 0
    with torch.inference_mode():
        for base in range(64):
            trial = 0
            while True:
                draw_seed = panel_seed * 1000000 + 1000 * base + trial
                geometry = sample_panel_geometry_train_151 if base % 5 == 0 else sample_panel_geometry_train_166
                original = geometry(context, [base], draw_seed, side=side)
                sampler = sample_one_shot_slide_artifacts_v4 if base % 2 else sample_one_shot_slide_artifacts_v3
                sample = sampler(context, [base], draw_seed, side=side, source_sample=original)
                if bool(sample['eligible'][0]):
                    break
                trial += 1
            prediction = model.predict(sample['inputs'])
            prior, beam = beam_for(prediction)
            ids = beam[0]
            states = prediction['state'][0, ids // 2]
            flags = ids.remainder(2)
            chart = chart_for(sample['valid_mask'], 256)
            original_cost = physical_cost(states[None], flags[None], sample, chart)[0]
            corrected_cost, match_score = [], []
            for start in range(0, 16, 2):
                out = updater(sample['inputs'], states[None, start:start + 2],
                              flags[None, start:start + 2], sample['offsets'],
                              sample['weights'], atlas)
                corrected_cost.append(physical_cost(out['state'], flags[None, start:start + 2],
                    sample, chart)[0])
                match_score.append(out['score'][0])
            corrected_cost = torch.cat(corrected_cost)
            match_score = torch.cat(match_score)
            action_score = torch.stack((prior[0, ids], prior[0, ids] + match_score), -1).flatten()
            action_cost = torch.stack((original_cost, corrected_cost), -1).flatten()
            choice = int(action_score.argmax())
            original_choice = int(prior[0, ids].argmax())
            errors.append(float(action_cost[choice]))
            direct.append(float(original_cost[original_choice]))
            oracle.append(float(original_cost.min()))
            captured += int(action_cost[choice] <= 1.5)
            direct_captured += int(original_cost[original_choice] <= 1.5)
    updater.train()
    return {'sections': 64, 'selected_action_mm': float(np.mean(errors)),
        'direct_prior_mm': float(np.mean(direct)), 'best_original16_mm': float(np.mean(oracle)),
        'selected_action_capture_1p5': captured, 'direct_prior_capture_1p5': direct_captured,
        'scope': 'fixed inner TRAIN sections; one per TRAIN base, not independent animals or DEV'}


if preflight:
    trial = 0
    while True:
        original = sample_panel_geometry_train_166(context, [0], seed * 1000000 + trial, side)
        sample = sample_one_shot_slide_artifacts_v4(context, [0], seed * 1000000 + trial,
            side, source_sample=original)
        if bool(sample['eligible'][0]):
            break
        trial += 1
    loss, detail = train_case(sample)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    assert float(gradient) > 0 and all(p.grad is None for p in model.parameters())
    print(json.dumps({'preflight': True, 'loss': float(loss.detach()),
        'gradient_norm': float(gradient), 'detail': detail}, allow_nan=False), flush=True)
    raise SystemExit

source_files = ('train_multiscale_plane_updater_174.py', 'multiscale_plane_updater_174.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py', 'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_panel_geometry_train_166.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'panel_seed': panel_seed, 'target_updates': steps, 'side': side,
    'synthetic_per_update': {'v4': 1, 'v3': 1}, 'accepted_train_bases': 64,
    'geometry': 'independent arbitrary planes; stochastic 80% support-intersecting 166 / 20% full-box 151 mixture',
    'draw_sampling': 'independent random base index, plane seed, geometry and appearance parameters per attempt; no paired recoloring or deliberate plane reuse',
    'beam': 'blind top 8 base + top 6 anchor + 2 normal-diverse anchors; truth enters only after beam',
    'training_candidates': 'closest predicted, prior top, prior-hard wrong, one near-truth jitter',
    'loss': '0.4 coarse + final five-point/valid-site rigid physical mm, corrected-error action KL, teacher no-shift score, update penalty',
    'inner_panel': '64 fixed independent physical TRAIN sections, one per TRAIN deformation base; best.pt is a TRAIN proxy only, not promoted',
    'protocol_sha256': sha(protocol),
    'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'parent_model_source_sha256': parent_config['source_sha256']['arbitrary_plane_one_shot_model.py'],
    'current_model_source_sha256': sha(source / 'arbitrary_plane_one_shot_model.py'),
    'parent_v3_sampler_source_sha256': parent_config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v3.py'],
    'current_v3_sampler_source_sha256': sha(source / 'arbitrary_plane_one_shot_slide_artifacts_v3.py'),
    'source_compatibility': 'frozen 132 model checkpoint loads strictly; current model adds optional default-off map derivative limit, and current v3 artifact sampler differs from 132 source; new draws use the current source hashes',
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'frozen': 'entire 132 pose, image, atlas, map and uncertainty model',
    'trainable': 'new random-init MultiscalePlaneUpdater174 only',
    'calibrated': False, 'real_training_images': 0, 'expert_real_truth_used': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
config_bytes = json.dumps(config, indent=2, allow_nan=False).encode()
attempt, best_metric, accepted_bases, start = 0, float('inf'), [0] * 64, 0


def save(path, step):
    target = run / path
    temp = target.with_suffix('.tmp')
    torch.save({'step': step, 'attempt': attempt, 'accepted_bases': accepted_bases,
        'best_metric': best_metric, 'updater': updater.state_dict(),
        'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'calibrated': False}, temp)
    os.replace(temp, target)


if run.exists():
    assert sha(run / 'config.json') == hashlib.sha256(config_bytes).hexdigest()
    resume_path = run / 'latest.pt' if (run / 'latest.pt').is_file() else run / 'updater_step_00000.pt'
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
    assert saved['config_sha256'] == sha(run / 'config.json') and saved['calibrated'] is False
    start, attempt, accepted_bases, best_metric = (saved['step'], saved['attempt'],
        saved['accepted_bases'], saved['best_metric'])
    updater.load_state_dict(saved['updater'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for name in ('draws.jsonl', 'training.jsonl', 'inner_scores.jsonl'):
        path = run / name
        if path.exists():
            rows = []
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record['update'] <= start:
                    rows.append(line)
            temp = path.with_suffix('.tmp')
            temp.write_text(''.join(line + '\n' for line in rows))
            os.replace(temp, path)
    assert start <= steps
else:
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_bytes(config_bytes)
    save('updater_step_00000.pt', 0)
    save('best.pt', 0)
    save('latest.pt', 0)

started = time.perf_counter()
with (run / 'draws.jsonl').open('a') as draws, (run / 'training.jsonl').open('a') as training, \
     (run / 'inner_scores.jsonl').open('a') as inner:
    for step in range(start + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        case_rows = []
        for slot in range(2):
            while True:
                attempt += 1
                base = int(rng.integers(64))
                draw_seed = int(rng.integers(0, 2**63 - 1))
                geometry = sample_panel_geometry_train_151 if rng.random() < .2 else sample_panel_geometry_train_166
                appearance = 'v4' if slot == 0 else 'v3'
                sampler = sample_one_shot_slide_artifacts_v4 if slot == 0 else sample_one_shot_slide_artifacts_v3
                with torch.no_grad():
                    original = geometry(context, [base], draw_seed, side)
                    sample = sampler(context, [base], draw_seed, side, source_sample=original)
                eligible = bool(sample['eligible'][0])
                draws.write(json.dumps({'update': step, 'slot': slot, 'attempt': attempt,
                    'base_index': base, 'draw_seed': draw_seed, 'geometry': geometry.__name__,
                    'appearance': appearance, 'used': eligible,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if eligible:
                    accepted_bases[base] += 1
                    break
            loss, detail = train_case(sample)
            (loss / 2).backward()
            case_rows.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
                              'base_index': base, 'appearance': appearance,
                              'loss': float(loss.detach()), **detail})
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.param_groups[0]['lr'] = 1e-4 * (.2 + .8 * .5 *
            (1 + math.cos(math.pi * min(step, 2000) / 2000)))
        optimizer.step()
        training.write(json.dumps({'update': step, 'cases': case_rows,
            'gradient_norm': float(gradient), 'lr': optimizer.param_groups[0]['lr'],
            'seconds_since_start': time.perf_counter() - started}, allow_nan=False) + '\n')
        if step % 500 == 0 or step == steps:
            metrics = evaluate()
            inner.write(json.dumps({'update': step, **metrics}, allow_nan=False) + '\n')
            if metrics['selected_action_mm'] < best_metric:
                best_metric = metrics['selected_action_mm']
                save('best.pt', step)
            save(f'updater_step_{step:05d}.pt', step)
            print(json.dumps({'update': step, **metrics, 'best_mm': best_metric}), flush=True)
        if step % 50 == 0 or step == steps:
            draws.flush()
            training.flush()
            inner.flush()
            save('latest.pt', step)

assert sum(accepted_bases) == 2 * steps and all(accepted_bases)
assert sha(parent_path) == config['parent_checkpoint_sha256']
receipt = {'updates': steps, 'accepted_synthetic': 2 * steps, 'draw_attempts': attempt,
    'accepted_base_counts': accepted_bases, 'best_inner_train_mm': best_metric,
    'config_sha256': sha(run / 'config.json'),
    'output_sha256': {path.name: sha(path) for path in run.iterdir()
        if path.is_file() and path.name != 'completed.json'},
    'protocol_sha256': sha(protocol),
    'parent_checkpoint_sha256': sha(parent_path),
    'calibrated': False, 'real_training_images': 0, 'expert_real_truth_used': False,
    'public_benchmark_used': False, 'external_pretrained_weights_used': False,
    'interpretation': 'TRAIN synthetic first-stage pose correction only; one atlas and 64 procedural deformation bases, no independent biological validation or fitted-tissue-to-pose feedback'}
temp = run / 'completed.tmp'
temp.write_text(json.dumps(receipt, indent=2, allow_nan=False))
os.replace(temp, run / 'completed.json')
print(json.dumps({'completed': True, 'updates': steps, 'best_inner_train_mm': best_metric}), flush=True)
