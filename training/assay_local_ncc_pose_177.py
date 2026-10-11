"""TRAIN-only 177 local-NCC pose mechanism assay; no blind selection claim."""

import hashlib
import json
import math
import os
import sys
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
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.gradient_guided_plane_fit_176 import GradientGuidedPlaneFitter176


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


def render(states, reflection, offsets, weights):
    count = len(states)
    plane = rigid_points_090(states, reflection.expand(count), chart, (256, 256))
    plane = plane.reshape(count, 96, 96, 3)
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    world = plane[:, None] + offsets.expand(count, -1)[:, :, None, None, None] * normal[:, None, None, None, :]
    sampled = render_finite_thickness_coordinate_grid(
        atlas, world, (0., 0., 0.), (25., 25., 25.), weights.expand(count, -1))
    support = sampled[:, 1:2].clamp(0, 1)
    return sampled[:, :1] / support.clamp_min(1e-4), support


def score(source, atlas_image, mask, window, permutation):
    if permutation is not None:
        atlas_image = atlas_image.flatten(1)[:, permutation].reshape_as(atlas_image)
    atlas_smooth = F.avg_pool2d(atlas_image, 3, 1, 1)
    sites = mask.sum()
    x_mean = (source * mask).sum() / sites
    y_mean = (atlas_smooth * mask).sum((-2, -1), keepdim=True) / sites
    x = (source - x_mean) * mask
    y = (atlas_smooth - y_mean) * mask
    covariance = (x * y).sum((-2, -1))
    variance = x.square().sum() * y.square().sum((-2, -1))
    global_fit = (covariance.square() / variance.clamp_min(1e-16) + 1e-4).sqrt()[:, 0]

    n = F.avg_pool2d(window, 9, 1, 4).clamp_min(1e-8)
    sx = F.avg_pool2d(window * source, 9, 1, 4)
    sy = F.avg_pool2d(window * atlas_smooth, 9, 1, 4)
    vx = (F.avg_pool2d(window * source.square(), 9, 1, 4) - sx.square() / n).clamp_min(0)
    vy = (F.avg_pool2d(window * atlas_smooth.square(), 9, 1, 4) - sy.square() / n).clamp_min(0)
    cov = F.avg_pool2d(window * source * atlas_smooth, 9, 1, 4) - sx * sy / n
    local = (cov.square() / (vx * vy).clamp_min(1e-12) + 1e-4).sqrt().clamp_max(1.00005)
    local_fit = (local * mask).sum((-2, -1))[:, 0] / sites
    return .3 * global_fit + .7 * local_fit, global_fit, local_fit


def optimize(state, reflection, source, offsets, weights, mask, window, permutation):
    sites = mask.sum()

    def measured(q):
        moved = compose_full_frame_state(state.expand(len(q), -1), q * scale)
        atlas_image, support = render(moved, reflection, offsets, weights)
        fit, global_fit, local_fit = score(source, atlas_image, mask, window, permutation)
        soft = (support * mask).sum((1, 2, 3)) / sites
        hard = ((support > .5).float() * mask).sum((1, 2, 3)) / sites
        energy = 1 - fit + .5 * (1 - soft) + .02 * q.square().sum(-1)
        return energy, fit, global_fit, local_fit, soft, hard

    q = state.new_zeros(1, 9)
    with torch.no_grad():
        initial = [float(value[0]) for value in measured(q)]
    chosen, norms = [], []
    for _ in range(3):
        q = q.detach().requires_grad_(True)
        energy = measured(q)[0]
        gradient = torch.autograd.grad(energy.sum(), q)[0].detach()
        norms.append(float(gradient.norm()))
        trial = q.detach().expand(3, -1) - lengths[:, None] * gradient / gradient.norm().clamp_min(1e-8)
        trial = trial / trial.norm(dim=-1, keepdim=True).clamp_min(1)
        options = torch.cat((q.detach(), trial))
        with torch.no_grad():
            choice = int(measured(options)[0].argmin())
        chosen.append((0., .1, .25, .5)[choice])
        q = options[choice:choice + 1]
    with torch.no_grad():
        final = [float(value[0]) for value in measured(q)]
        corrected = compose_full_frame_state(state, q * scale)[0]
    names = ('energy', 'fit', 'global_fit', 'local_fit', 'soft_retention', 'support_retention')
    assert float(q.norm()) <= 1.000001
    return corrected, {'initial': dict(zip(names, initial)), 'final': dict(zip(names, final)),
        'q': q[0].detach().tolist(), 'q_norm': float(q.norm()),
        'step_lengths': chosen, 'gradient_norms': norms, 'no_shift': bool(q.norm() == 0)}


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/LOCAL_NCC_TRUST_REGION_177_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/local_ncc_pose_177_train_001'
assert source.drive.upper() == out.drive.upper() == root.drive.upper() == 'I:'
assert not out.exists() and torch.cuda.is_available()
assert sha(parent_path) == json.loads((parent_dir / 'completed.json').read_text())['checkpoint_sha256']['2000']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed = 202610177
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
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
old_fitter = GradientGuidedPlaneFitter176().cuda().eval()
axis = (torch.arange(96, device='cuda') + .5) / 96 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
sites = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                      [127.5, 127.5]], device='cuda') / 256
scale = torch.tensor([.12, .12, .12, 900., 900., 900., .08, .08, .06], device='cuda')
lengths = torch.tensor([.1, .25, .5], device='cuda')
config = {'seed': seed, 'draws_per_appearance': 64, 'bases_per_appearance': 64,
    'sampling': 'one independent full-sphere TRAIN-166 draw per base and appearance; no resampling',
    'eligible_start_mm': [.5, 3.], 'minimum_mask_sites_96': 400,
    'mask': '7x7 erosion of initial candidate atlas support >0.5',
    'local_window': '9x9 NCC weighted by fixed initial atlas support',
    'score': '0.3 global + 0.7 local smooth absolute NCC after 3x3 source/atlas smoothing',
    'energy': '1-fit + 0.5*(1-soft-support-retention) + 0.02*||q||^2',
    'step_lengths': [.1, .25, .5], 'steps': 3, 'q_norm_bound': 1.,
    'permutation': 'cyclic one-third shift within initial >0.5 atlas support',
    'source_sha256': {name: sha(source.parent / name) for name in (
        source.name, 'arbitrary_plane_panel_geometry_train_166.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'gradient_guided_plane_fit_176.py')},
    'protocol_sha256': sha(protocol), 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'synthetic_provenance': context['provenance'],
    'diagnostic_truth_not_an_inference_input': True, 'calibrated': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

rows, draw_rows, physical_ids = [], [], set()
with (out / 'draws.jsonl').open('w') as draws, (out / 'rows.jsonl').open('w') as accepted:
    for appearance in ('v3', 'v4'):
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        for base in rng.permutation(64):
            base, draw_seed = int(base), int(rng.integers(0, 2**63 - 1))
            with torch.no_grad():
                geometry = sample_panel_geometry_train_166(context, [base], draw_seed, 256)
                sample = sampler(context, [base], draw_seed, 256, source_sample=geometry)
                truth, truth_flag = sample['state'][0], sample['reflection'][0]
                normal = full_frame_state_to_components(truth[None])[1][0, :, 2]
                angle = math.degrees(math.acos(float(normal.abs().max().clamp(-1, 1))))
            provenance = sample['provenance'][0]
            physical_id = provenance['physical_section_id']
            assert provenance['split'] == 'train' and physical_id not in physical_ids
            physical_ids.add(physical_id)
            digest = hashlib.sha256()
            for key in ('inputs', 'offsets', 'weights', 'state', 'reflection'):
                digest.update(sample[key].detach().cpu().contiguous().numpy().tobytes())
            row = {'draw_index': len(physical_ids) - 1, 'appearance': appearance,
                'base_index': base, 'draw_seed': draw_seed, 'physical_section_id': physical_id,
                'provenance': provenance, 'input_tensors_sha256': digest.hexdigest(),
                'model_inputs_sha256': hashlib.sha256(sample['inputs'].detach().cpu().contiguous()
                    .numpy().tobytes()).hexdigest(),
                'nearest_cardinal_angle_deg': angle, 'sampler_eligible': bool(sample['eligible'][0]),
                'eligible': False, 'exclusion_reason': None}
            slot = None
            if not row['sampler_eligible']:
                row['exclusion_reason'] = 'sampler_ineligible'
            else:
                with torch.no_grad():
                    prediction = model.predict(sample['inputs'])
                    prior, beam = beam_for(prediction)
                    ids = beam[0]
                    states = prediction['state'][0, ids // 2]
                    flags = ids.remainder(2)
                    reference = rigid_points_090(truth[None], truth_flag[None], sites)[0]
                    errors = (rigid_points_090(states, flags, sites) - reference).norm(dim=-1).mean(-1) / 1000
                    near = ((flags == truth_flag) & (errors >= .5) & (errors <= 3.)).nonzero()[:, 0]
                    near = near[errors[near].argsort()]
                    row['near_matching_reflection_count'] = len(near)
                    row['nearest_matching_reflection_mm'] = float(errors[near[0]]) if len(near) else None
                    row['near_support_sites96'] = []
                    for candidate in near:
                        candidate = int(candidate)
                        candidate_state = states[candidate:candidate + 1].detach()
                        candidate_flag = flags[candidate:candidate + 1]
                        _, initial_support = render(candidate_state, candidate_flag,
                                                    sample['offsets'], sample['weights'])
                        window = (initial_support > .5).float()
                        mask = (F.avg_pool2d(window, 7, 1, 3, count_include_pad=True)
                                >= 1 - 1e-6).float()
                        count = int(mask.sum())
                        row['near_support_sites96'].append({'branch_id': int(ids[candidate]), 'sites': count})
                        if count >= 400:
                            slot = int(candidate)
                            break
                if slot is None:
                    row['exclusion_reason'] = ('no_matching_reflection_candidate_0p5_to_3mm'
                                               if not len(near) else 'inadequate_candidate_atlas_support')
            if slot is not None:
                state = states[slot:slot + 1].detach()
                flag = flags[slot:slot + 1]
                source_image = F.avg_pool2d(F.interpolate(sample['inputs'][:, :1],
                                             (96, 96), mode='area'), 3, 1, 1)
                pixels = window.flatten().nonzero()[:, 0]
                permutation = torch.arange(96 * 96, device='cuda')
                permutation[pixels] = pixels.roll(max(1, len(pixels) // 3))
                old = old_fitter(sample['inputs'], state[:, None], flag[:, None],
                                 sample['offsets'], sample['weights'], atlas)
                full_state, full = optimize(state, flag, source_image, sample['offsets'],
                                             sample['weights'], mask, window, None)
                null_state, null = optimize(state, flag, source_image, sample['offsets'],
                                             sample['weights'], mask, window, permutation)
                with torch.no_grad():
                    candidate_states = torch.cat((state, old['state'][0],
                                                  full_state[None], null_state[None]))
                    physical = (rigid_points_090(candidate_states, flag.expand(4), sites)
                                - reference).norm(dim=-1).mean(-1) / 1000
                row.update({'eligible': True, 'branch_slot': slot, 'branch_id': int(ids[slot]),
                    'branch_prior_logit': float(prior[0, ids[slot]]), 'support_sites96': int(mask.sum()),
                    'five_point_mm': dict(zip(('starting', 'old176', 'full', 'null'),
                                              [float(value) for value in physical])),
                    'old176': {'fit_start': float(old['original_fit'][0, 0]),
                        'fit_final': float(old['fit'][0, 0]),
                        'support_sites48': int(old['support_sites'][0, 0]),
                        'update_eligible': bool(old['support_sites'][0, 0] >= 100),
                        'support_retention': float(old['support_retention'][0, 0]),
                        'gradient_norm': float(old['gradient_norm'][0, 0]),
                        'q_norm': float((old['local_update'][0, 0] / scale).norm()),
                        'local_update': old['local_update'][0, 0].detach().tolist()},
                    'full': full, 'null': null})
                rows.append(row)
                accepted.write(json.dumps(row, allow_nan=False) + '\n')
            draw_rows.append(row)
            draws.write(json.dumps(row, allow_nan=False) + '\n')
        accepted.flush()
        draws.flush()
        print(json.dumps({'event': 'appearance_done', 'appearance': appearance,
            'eligible': sum(row['appearance'] == appearance for row in rows)}), flush=True)


def summarise(part):
    errors = np.asarray([[row['five_point_mm'][key] for key in
        ('starting', 'old176', 'full', 'null')] for row in part])
    return {'eligible': len(part),
        'mean_mm': dict(zip(('starting', 'old176', 'full', 'null'),
            errors.mean(0).tolist())) if part else None,
        'mean_reduction_mm': {name: float(np.mean(errors[:, 0] - errors[:, index]))
            for index, name in enumerate(('old176', 'full', 'null'), 1)} if part else None,
        'fraction_physically_improved': {name: float(np.mean(errors[:, index] < errors[:, 0]))
            for index, name in enumerate(('old176', 'full', 'null'), 1)} if part else None,
        'full_minus_null_reduction_mm': float(np.mean(errors[:, 3] - errors[:, 2])) if part else None,
        'mean_fit': {'old176': {stage: float(np.mean([row['old176'][f'fit_{key}']
            for row in part])) for stage, key in (('initial', 'start'), ('final', 'final'))},
            **{name: {stage: float(np.mean([row[name][stage]['fit'] for row in part]))
            for stage in ('initial', 'final')} for name in ('full', 'null')}} if part else None,
        'mean_full_minus_null_final_fit': float(np.mean([
            row['full']['final']['fit'] - row['null']['final']['fit'] for row in part])) if part else None,
        'mean_support_retention': {name: float(np.mean([
            row[name]['final']['support_retention'] for row in part]))
            for name in ('full', 'null')} if part else None,
        'mean_old176_soft_support_retention': float(np.mean([
            row['old176']['support_retention'] for row in part])) if part else None,
        'old176_update_eligible_fraction': float(np.mean([
            row['old176']['update_eligible'] for row in part])) if part else None,
        'mean_q_norm': {name: float(np.mean([row[name]['q_norm'] for row in part]))
            for name in ('old176', 'full', 'null')} if part else None,
        'no_shift_fraction': {name: float(np.mean([row[name]['no_shift'] for row in part]))
            for name in ('full', 'null')} if part else None}


summary = {'draws': len(physical_ids), 'eligible': len(rows),
    'excluded': {reason: sum(row['exclusion_reason'] == reason for row in draw_rows) for reason in (
        'sampler_ineligible', 'no_matching_reflection_candidate_0p5_to_3mm',
        'inadequate_candidate_atlas_support')},
    'by_appearance': {}, 'by_angle_deg': {},
    'diagnostic_truth_chose_branch_only': True, 'blind_selection_claim': False}
for appearance in ('v3', 'v4'):
    part = [row for row in rows if row['appearance'] == appearance]
    result = summarise(part)
    result['draws'] = 64
    result['excluded'] = {reason: sum(row['appearance'] == appearance and
        row['exclusion_reason'] == reason for row in draw_rows) for reason in summary['excluded']}
    result['inconclusive'] = len(part) < 30
    result['gate'] = {'mean_reduction_ge_0p15_mm': bool(part and result['mean_reduction_mm']['full'] >= .15),
        'physically_improved_ge_0p65': bool(part and result['fraction_physically_improved']['full'] >= .65),
        'full_over_null_ge_0p10_mm': bool(part and result['full_minus_null_reduction_mm'] >= .10),
        'support_retention_ge_0p99': bool(part and result['mean_support_retention']['full'] >= .99)}
    result['passes'] = not result['inconclusive'] and all(result['gate'].values())
    summary['by_appearance'][appearance] = result
    summary['by_angle_deg'][appearance] = {}
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 55)):
        angle_rows = [row for row in draw_rows if row['appearance'] == appearance
            and lo <= row['nearest_cardinal_angle_deg'] < hi]
        summary['by_angle_deg'][appearance][f'[{lo},{hi})'] = {
            **summarise([row for row in angle_rows if row['eligible']]),
            'draws': len(angle_rows)}
summary['continuation_gate_passes'] = all(
    result['passes'] for result in summary['by_appearance'].values())
assert len(physical_ids) == 128 and all(sum(row['base_index'] == base and
    row['appearance'] == appearance for row in draw_rows) == 1
    for appearance in ('v3', 'v4') for base in range(64))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'output_sha256': {
    path.name: sha(path) for path in out.iterdir() if path.is_file() and path.name != 'completed.json'},
    'source_sha256': config['source_sha256'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'draws': len(physical_ids), 'eligible': len(rows),
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'draws': len(physical_ids), 'eligible': len(rows),
    'continuation_gate_passes': summary['continuation_gate_passes']}), flush=True)
