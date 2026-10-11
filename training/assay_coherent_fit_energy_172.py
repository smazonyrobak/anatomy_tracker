"""Frozen TRAIN-only coherent fit-energy diagnostic on blind arbitrary-plane beams."""

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

from training.anatomy_correspondence_feedback_168 import AnatomyCorrespondenceFeedback
from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/COHERENT_FIT_ENERGY_172_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
matcher_dir = root / 'runs/anatomy_hard_beam_171_train_001'
matcher_path = matcher_dir / 'matcher_step_01600.pt'
out = root / 'runs/coherent_fit_energy_172_train_001'
seed, side, grid_side, attempts = 202610111720, 256, 48, 240
device = 'cuda'
assert source.drive.upper() == out.drive.upper() == 'I:'
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_done = json.loads((parent_dir / 'completed.json').read_text())
matcher_done = json.loads((matcher_dir / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(matcher_path) == matcher_done['output_sha256'][matcher_path.name]
context = load_streaming_synthetic_v7_64(device=device)
atlas = context['atlas']
null_atlas = torch.cat((atlas[:1] * 0, atlas[1:2]))
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    saved_matcher = torch.load(matcher_path, map_location='cpu', weights_only=True)
assert parent['step'] == 2000 and parent['calibrated'] is False
assert saved_matcher['step'] == 1600 and saved_matcher['calibrated'] is False
model.load_state_dict(parent['model'], strict=True)
matcher = AnatomyCorrespondenceFeedback(side=grid_side, radius=3).cuda().eval().requires_grad_(False)
matcher.load_state_dict(saved_matcher['matcher'], strict=True)
del parent, saved_matcher

axis = (torch.arange(grid_side, device=device) + .5) / grid_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
grid_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
depths = torch.tensor([-500., -250., 0., 250., 500.], device=device)


def draw(index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    base = int(rng.integers(60, 64))
    geometry = sample_panel_geometry_train_151 if rng.random() < .2 else sample_panel_geometry_train_166
    stratum = 'unconditional' if index < 192 else 'orientation_conditioned_45_55'
    trials = 0
    while True:
        draw_seed = int(rng.integers(0, 2**31 - 1))
        trials += 1
        if index < 192:
            break
        normal = np.random.default_rng(np.random.SeedSequence([draw_seed, base, 0, 0])).normal(size=3)
        normal /= np.linalg.norm(normal)
        if math.degrees(math.acos(float(np.abs(normal).max()))) >= 45:
            break
    with torch.inference_mode():
        original = geometry(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(context, [base], draw_seed,
            side, source_sample=original)
    return sample, {'index': index, 'base_index': base, 'draw_seed': draw_seed,
        'geometry': geometry.__name__, 'panel_stratum': stratum,
        'orientation_seed_trials': trials, 'valid_pixels': int(sample['valid_mask'].sum()),
        'provenance': sample['provenance'][0]}


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device=device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


def render_rigid(states, reflection, offsets, weights, depth_bank=False):
    count = len(states)
    plane = rigid_points_090(states, reflection, grid_chart, (side, side)).reshape(
        count, grid_side, grid_side, 3)
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    if depth_bank:
        z = offsets.expand(count, -1)[:, None, :] + depths[None, :, None]
        coords = plane[:, None, None] + z[:, :, :, None, None, None] * normal[:, None, None, None, None, :]
        mass = weights.expand(count, -1)[:, None].expand(-1, len(depths), -1).flatten(0, 1)
        rendered = render_finite_thickness_coordinate_grid(atlas, coords.flatten(0, 1),
            (0., 0., 0.), (25., 25., 25.), mass).reshape(count, len(depths), 2, grid_side, grid_side)
        support = rendered[:, :, 1:2].clamp(0, 1)
        return rendered[:, :, :1] / support.clamp_min(1e-4), support
    coords = plane[:, None] + offsets.expand(count, -1)[:, :, None, None, None] * normal[:, None, None, None, :]
    rendered = render_finite_thickness_coordinate_grid(atlas, coords,
        (0., 0., 0.), (25., 25., 25.), weights.expand(count, -1))
    support = rendered[:, 1:2].clamp(0, 1)
    return rendered[:, :1] / support.clamp_min(1e-4), support


def chart_for(valid, count):
    pixels = valid[0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(count, len(pixels)),
        device=pixels.device).long()]
    return torch.stack((pixels.remainder(side),
        pixels.div(side, rounding_mode='floor')), -1).float() / side


def freeze_case(sample, record):
    item = dict(record)
    if not record['valid_pixels']:
        return item
    with torch.inference_mode():
        prediction = model.predict(sample['inputs'])
        prior, beam = beam_for(prediction)
        ids = beam[0]
        chosen = ids // 2
        flags = (ids % 2)[None]
        states = prediction['state'][:, chosen]
        native_score, native_state = [], []
        for start in range(0, 16, 4):
            fit = global_plane_match(model, prediction, chosen[None, start:start + 4],
                flags[:, start:start + 4], sample['offsets'], sample['weights'],
                atlas, (side, side), side=24)
            action_score = torch.stack((fit['input_score'][0], fit['score'][0]), -1)
            action = action_score.argmax(-1)
            native_score.append(action_score.max(-1).values)
            native_state.append(torch.where(action[:, None] == 0,
                fit['input_state'][0], fit['state'][0]))
        item.update({'inputs': sample['inputs'][0].cpu(),
            'offsets': sample['offsets'][0].cpu(), 'weights': sample['weights'][0].cpu(),
            'branch_ids': ids.cpu(), 'reflection': flags[0].cpu(),
            'initial_state': states[0].cpu(), 'prior_score': prior[0, ids].cpu(),
            'native_score': torch.cat(native_score).cpu(),
            'native_state': torch.cat(native_state).cpu(),
            'chart': chart_for(sample['valid_mask'], 1024).cpu(),
            'truth_state': sample['state'][0].cpu(),
            'truth_reflection': sample['reflection'][0].cpu(),
            'target_centre_48': F.interpolate(sample['centre'].permute(0, 3, 1, 2),
                (grid_side, grid_side), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu(),
            'target_valid_area_48': F.interpolate(sample['valid_mask'][:, None].float(),
                (grid_side, grid_side), mode='area')[0, 0].cpu()})
    return item


def mapped_terms(mapped, sampled, source_desc, state, reflection, offsets, zero_intensity=False):
    support = sampled[:, 1:2].clamp(0, 1)
    intensity = sampled[:, :1] / support.clamp_min(1e-4)
    if zero_intensity:
        intensity = torch.zeros_like(intensity)
    cosine = (source_desc.expand(len(state), -1, -1, -1)
              * matcher.encode_atlas(intensity)).sum(1).clamp(-1, 1)
    appearance = support[:, 0] * (1 - cosine) + (1 - support[:, 0])
    local = mapped['local_displacement_um'][0]
    displacement = local.square().sum(1) / 500.**2
    dx = F.pad((local[..., 1:] - local[..., :-1]) * grid_side, (0, 1, 0, 0))
    dx = torch.where(reflection[:, None, None, None].bool(), -dx, dx)
    dy = F.pad((local[..., 1:, :] - local[..., :-1, :]) * grid_side, (0, 0, 0, 1))
    chart_gradient = torch.stack((dx, dy), -1)
    basis = full_frame_state_to_components(state)[2]
    physical_gradient = torch.einsum('bchwp,bpq->bchwq', chart_gradient, torch.linalg.inv(basis))
    gradient = physical_gradient.square().sum((1, 4)) / .2**2
    canonical_local = torch.where(reflection[:, None, None, None].bool(),
        local.flip(-1), local)
    bound = ribbon_derivative_frobenius_bound(canonical_local, torch.zeros_like(local),
        basis, offsets.expand(len(state), -1))
    return appearance + .05 * displacement + .05 * gradient, {
        'appearance': appearance, 'support': support[:, 0],
        'displacement': displacement, 'gradient': gradient,
        'derivative_bound': bound,
        'removed_affine': mapped['removed_affine_local_um'][0],
        'surface': mapped['centre_surface_ccf_ap_dv_ml_um'][0]}


def energy_on(terms, mask):
    return (terms * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)


def standardize(x):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)


def evaluate_case(item):
    record = {key: item[key] for key in ('index', 'base_index', 'draw_seed',
        'geometry', 'valid_pixels', 'panel_stratum', 'orientation_seed_trials', 'provenance')}
    record['scored'] = 'inputs' in item
    if not record['scored']:
        return record
    image = item['inputs'][None].cuda()
    offsets = item['offsets'][None].cuda()
    weights = item['weights'][None].cuda()
    ids = item['branch_ids'].cuda()
    states = item['initial_state'].cuda()
    flags = item['reflection'].cuda()
    prediction = model.predict(image)
    prior, beam = beam_for(prediction)
    assert torch.equal(ids, beam[0])
    assert torch.allclose(item['prior_score'].cuda(), prior[0, ids], atol=1e-5)
    source_desc = matcher.encode_source(image)
    fields = {name: [] for name in ('map', 'rigid', 'score_null', 'full_null',
        'pre_mask', 'pre_eroded', 'mapped_support', 'full_null_support',
        'displacement', 'gradient', 'derivative_bound', 'removed_affine',
        'full_null_displacement', 'full_null_gradient', 'full_null_derivative_bound',
        'full_null_removed_affine', 'mapped_surface')}
    local_peak = []
    for start in range(0, 16, 2):
        stop = start + 2
        state, reflected = states[start:stop], flags[start:stop]
        mode = (ids[start:stop] // 2)[None]
        flip = (ids[start:stop] % 2)[None]
        rigid_intensity, rigid_support = render_rigid(state, reflected, offsets, weights)
        pre_mask = (rigid_support[:, 0] > .5).float()
        pre_eroded = (F.avg_pool2d(pre_mask[:, None], 7, 1, 3,
            count_include_pad=True)[:, 0] >= 1 - 1e-6).float()
        rigid_cosine = (source_desc.expand(len(state), -1, -1, -1)
                        * matcher.encode_atlas(rigid_intensity)).sum(1).clamp(-1, 1)
        rigid_term = rigid_support[:, 0] * (1 - rigid_cosine) + (1 - rigid_support[:, 0])
        actual = model.map(prediction, offsets, mode, flip,
            (grid_side, grid_side), atlas, weights,
            source_shape=(side, side))
        actual_render = render_finite_thickness_coordinate_grid(atlas,
            actual['coordinates'][0], (0., 0., 0.), (25., 25., 25.),
            weights.expand(len(state), -1))
        term, detail = mapped_terms(actual, actual_render, source_desc, state,
            reflected, offsets)
        null_cosine = (source_desc.expand(len(state), -1, -1, -1)
            * matcher.encode_atlas(torch.zeros_like(rigid_intensity))).sum(1).clamp(-1, 1)
        mapped_support = detail['support']
        score_null_term = (mapped_support * (1 - null_cosine) + (1 - mapped_support)
            + .05 * detail['displacement'] + .05 * detail['gradient'])
        full_null = model.map(prediction, offsets, mode, flip,
            (grid_side, grid_side), null_atlas, weights,
            source_shape=(side, side))
        null_render = render_finite_thickness_coordinate_grid(null_atlas,
            full_null['coordinates'][0], (0., 0., 0.), (25., 25., 25.),
            weights.expand(len(state), -1))
        null_term, null_detail = mapped_terms(full_null, null_render, source_desc,
            state, reflected, offsets, zero_intensity=True)
        initial_bank, initial_support = render_rigid(state, reflected,
            offsets, weights, depth_bank=True)
        first = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
            matcher.encode_atlas(initial_bank), initial_support, depths)
        corrected = compose_full_frame_state(state, first['update'])
        corrected_bank, corrected_support = render_rigid(corrected, reflected,
            offsets, weights, depth_bank=True)
        second = matcher.compare(source_desc.expand(len(state), -1, -1, -1),
            matcher.encode_atlas(corrected_bank), corrected_support, depths)
        effort = .04 * (first['update'] / matcher.update_limits).square().mean(-1)
        local_peak.append(second['score'] - effort)
        for key, value in (('map', term), ('rigid', rigid_term),
                ('score_null', score_null_term), ('full_null', null_term),
                ('pre_mask', pre_mask), ('pre_eroded', pre_eroded),
                ('mapped_support', detail['support']),
                ('full_null_support', null_detail['support']),
                ('displacement', detail['displacement']),
                ('gradient', detail['gradient']),
                ('derivative_bound', detail['derivative_bound']),
                ('removed_affine', detail['removed_affine']),
                ('full_null_displacement', null_detail['displacement']),
                ('full_null_gradient', null_detail['gradient']),
                ('full_null_derivative_bound', null_detail['derivative_bound']),
                ('full_null_removed_affine', null_detail['removed_affine']),
                ('mapped_surface', detail['surface'])):
            fields[key].append(value)
    fields = {key: torch.cat(value) for key, value in fields.items()}
    local_peak = torch.cat(local_peak)
    pre_mask, sites = fields['pre_mask'], fields['pre_eroded']
    missing = .2 * (1 - pre_mask.mean((-2, -1)))
    energies = {key: energy_on(fields[key], sites) + missing
                for key in ('map', 'rigid', 'score_null', 'full_null')}
    for key in energies:
        energies[key] = torch.where(sites.sum((-2, -1)) > 0,
            energies[key], energies[key].new_full((16,), 1e3))
    chosen = {'prior': int(item['prior_score'].argmax()),
        'support_only': int(pre_mask.mean((-2, -1)).argmax()),
        'native121': int(item['native_score'].argmax()),
        'local_peak171': int(local_peak.argmax())}
    for key, value in energies.items():
        chosen[key] = int(value.argmin())
        chosen['fused_' + key] = int((standardize(item['prior_score'].cuda())
            - standardize(value)).argmax())
    # All 16-way inference values and choices are now fixed; only now read synthetic truth.
    chart = item['chart'].cuda()
    truth = rigid_points_090(item['truth_state'][None].cuda(),
        item['truth_reflection'][None].cuda(), chart)[0]
    rigid_error = (rigid_points_090(states, flags, chart) - truth[None]
                   ).norm(dim=-1).mean(-1) / 1000
    native_error = (rigid_points_090(item['native_state'].cuda(), flags, chart)
                    - truth[None]).norm(dim=-1).mean(-1) / 1000
    target = item['target_centre_48'].cuda()
    valid_area = item['target_valid_area_48'].cuda()
    valid = valid_area >= .8
    fallback_valid = not bool(valid.any())
    if fallback_valid:
        valid = valid_area > 0
    map_error = (fields['mapped_surface'][:, valid] - target[valid][None]
                 ).norm(dim=-1).mean(-1) / 1000
    native_plane = rigid_points_090(item['native_state'].cuda(), flags,
        grid_chart, (side, side)).reshape(16, grid_side, grid_side, 3)
    native_tissue_error = (native_plane[:, valid] - target[valid][None]
        ).norm(dim=-1).mean(-1) / 1000
    pair = None
    best = int(rigid_error.argmin())
    if rigid_error[best] <= 1.5:
        area = pre_mask.mean((-2, -1))
        common = (pre_mask[best][None] * pre_mask).sum((-2, -1))
        dice = 2 * common / (pre_mask[best].sum() + pre_mask.sum((-2, -1))).clamp_min(1)
        eligible = ((rigid_error >= rigid_error[best] + 1.) & (dice >= .85)
                    & ((area - area[best]).abs() <= .08))
        wrongs = eligible.nonzero()[:, 0]
        if len(wrongs):
            wrong = int(wrongs[item['prior_score'].cuda()[wrongs].argmax()])
            common_sites = (F.avg_pool2d((pre_mask[best] * pre_mask[wrong])[None, None],
                7, 1, 3, count_include_pad=True)[0, 0] >= 1 - 1e-6).float()
            if int(common_sites.sum()) >= 64:
                pair = {'good_slot': best, 'wrong_slot': wrong,
                    'common_sites': int(common_sites.sum()), 'support_dice': float(dice[wrong]),
                    'support_area_gap': float((area[best] - area[wrong]).abs()),
                    'rigid_gap_mm': float(rigid_error[wrong] - rigid_error[best]),
                    'energy': {key: [float(energy_on(fields[key][slot:slot + 1],
                        common_sites[None])[0]) for slot in (best, wrong)]
                        for key in ('map', 'rigid', 'score_null', 'full_null')},
                    'local_peak171': [float(local_peak[slot]) for slot in (best, wrong)]}
    provenance = item['provenance']
    normal = np.abs(provenance['physical_unit_normal_ap_dv_ml'])
    record.update({'animal_id': provenance['base_lineage']['animal_id'],
        'specimen_id': provenance['base_lineage']['specimen_id'],
        'experiment_id': provenance['base_lineage']['experiment_id'],
        'physical_section_id': provenance['physical_section_id'],
        'appearance_mode': provenance['mode'],
        'nearest_cardinal_angle_deg': math.degrees(math.acos(min(1., float(np.max(normal))))),
        'valid_target_fraction': item['valid_pixels'] / (side * side),
        'target_valid_cells_48': int(valid.sum()),
        'target_valid_area_fallback': fallback_valid,
        'branch_ids': ids.cpu().tolist(), 'candidate_reflection': flags.cpu().tolist(),
        'candidate_initial_state': states.cpu().tolist(),
        'psf_offsets_um': offsets[0].cpu().tolist(), 'psf_weights': weights[0].cpu().tolist(),
        'candidate_pre_support_fraction': pre_mask.mean((-2, -1)).cpu().tolist(),
        'candidate_eroded_sites': sites.sum((-2, -1)).cpu().tolist(),
        'candidate_mapped_support_fraction': fields['mapped_support'].mean((-2, -1)).cpu().tolist(),
        'candidate_full_null_support_fraction': fields['full_null_support'].mean((-2, -1)).cpu().tolist(),
        'candidate_mean_displacement_cost': fields['displacement'].mean((-2, -1)).cpu().tolist(),
        'candidate_mean_gradient_cost': fields['gradient'].mean((-2, -1)).cpu().tolist(),
        'candidate_derivative_bound': fields['derivative_bound'].cpu().tolist(),
        'candidate_removed_affine_local_um': fields['removed_affine'].cpu().tolist(),
        'candidate_full_null_mean_displacement_cost': fields['full_null_displacement'].mean((-2, -1)).cpu().tolist(),
        'candidate_full_null_mean_gradient_cost': fields['full_null_gradient'].mean((-2, -1)).cpu().tolist(),
        'candidate_full_null_derivative_bound': fields['full_null_derivative_bound'].cpu().tolist(),
        'candidate_full_null_removed_affine_local_um': fields['full_null_removed_affine'].cpu().tolist(),
        'scores': {'prior': item['prior_score'].tolist(),
            'native121': item['native_score'].tolist(),
            'local_peak171': local_peak.cpu().tolist(),
            **{key: value.cpu().tolist() for key, value in energies.items()}},
        'chosen_slot': chosen, 'initial_rigid_mm': rigid_error.cpu().tolist(),
        'native_fitted_rigid_mm': native_error.cpu().tolist(),
        'mapped_tissue_mm': map_error.cpu().tolist(),
        'native_fitted_tissue_mm': native_tissue_error.cpu().tolist(),
        'matched_pair': pair})
    return record


def readout(rows):
    result = {'sections': len(rows)}
    for name in ('prior', 'support_only', 'native121', 'local_peak171', 'map', 'rigid',
            'score_null', 'full_null', 'fused_map', 'fused_rigid',
            'fused_score_null', 'fused_full_null'):
        error = np.asarray([row['initial_rigid_mm'][row['chosen_slot'][name]] for row in rows])
        result[name] = {'initial_mean_mm': float(error.mean()) if len(error) else None,
            'capture_le_1p5': float((error <= 1.5).mean()) if len(error) else None}
    oracle = np.asarray([min(row['initial_rigid_mm']) for row in rows])
    result['oracle16'] = {'initial_mean_mm': float(oracle.mean()) if len(oracle) else None,
        'capture_le_1p5': float((oracle <= 1.5).mean()) if len(oracle) else None}
    result['selected_tissue_mm'] = {name: float(np.mean([
        row['mapped_tissue_mm'][row['chosen_slot'][name]] for row in rows])) if rows else None
        for name in ('map', 'fused_map', 'fused_full_null')}
    result['selected_tissue_mm']['native121'] = float(np.mean([
        row['native_fitted_tissue_mm'][row['chosen_slot']['native121']]
        for row in rows])) if rows else None
    return result


if os.environ.get('JOINT172_MODE') == 'preflight':
    for preflight_index in range(999000, 999016):
        sample, record = draw(preflight_index)
        if record['valid_pixels']:
            break
    assert record['valid_pixels']
    frozen = freeze_case(sample, record)
    with torch.inference_mode():
        result = evaluate_case(frozen)
    print(json.dumps({'event': 'preflight', 'attempts': preflight_index - 999000 + 1,
        'scored': result['scored'],
        'chosen_slot': result.get('chosen_slot'),
        'matched_pair': result.get('matched_pair'),
        'source_sha256': sha(source)}, allow_nan=False), flush=True)
    sys.exit(0)

assert not out.exists()
out.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
panel = []
for index in range(attempts):
    sample, record = draw(index)
    panel.append(freeze_case(sample, record))
    if (index + 1) % 48 == 0:
        print(json.dumps({'event': 'panel_frozen', 'attempts': index + 1,
            'nonblank': sum('inputs' in row for row in panel)}), flush=True)
panel_path = out / 'panel.pt'
temp = panel_path.with_suffix('.tmp')
torch.save(panel, temp)
os.replace(temp, panel_path)
source_files = (source, source.parents[0] / 'anatomy_correspondence_feedback_168.py',
    source.parents[0] / 'train_anatomy_hard_beam_171.py',
    source.parents[0] / 'arbitrary_plane_one_shot_model.py',
    source.parents[0] / 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    source.parents[0] / 'arbitrary_plane_panel_geometry_train_151.py',
    source.parents[0] / 'arbitrary_plane_panel_geometry_train_166.py',
    source.parents[0] / 'arbitrary_plane_ribbon_v6.py',
    source.parents[0] / 'arbitrary_plane_streaming_synthetic_v7.py',
    source.parents[0] / 'arbitrary_plane_streaming_synthetic_v7_64.py',
    source.parents[0] / 'arbitrary_plane_full_frame_primitives.py',
    source.parents[0] / 'global_atlas_contrast_090.py',
    source.parents[0] / 'global_plane_matcher_120.py')
config = {'scope': 'TRAIN-only frozen-map coherent fit-energy assay',
    'seed': seed, 'attempts': attempts, 'grid_side': grid_side,
    'unconditional_attempts': 192, 'orientation_conditioned_attempts': 48,
    'panel_bases': [60, 61, 62, 63], 'blank_policy': 'retain all attempts',
    'beam_policy': 'frozen 132 blind16 with native 121 action, before reading truth',
    'energy_policy': 'mean S*(1-cosine)+(1-S)+.05*(u/500um)^2+.05*(physical gradient/.2)^2 on eroded pre-map support, plus .2 missing pre-map fraction',
    'pair_policy': 'truth-chosen best <=1.5mm vs highest-prior >=1mm-worse candidate; pre-map Dice >=.85, area gap <=.08, >=64 common eroded cells; same comparison sites',
    'fusion_policy': 'within-beam z(prior log mass)-z(energy); support-only picks largest pre-map atlas support fraction',
    'source_sha256': {path.name: sha(path) for path in source_files},
    'protocol_sha256': sha(protocol), 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'matcher_checkpoint_sha256': sha(matcher_path),
    'matcher_completion_sha256': sha(matcher_dir / 'completed.json'),
    'panel_sha256': sha(panel_path), 'synthetic_provenance': context['provenance'],
    'no_external_pretrained_weights': True, 'posterior_calibrated': False,
    'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for index, item in enumerate(panel):
        row = evaluate_case(item)
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if (index + 1) % 24 == 0:
            stream.flush()
            print(json.dumps({'event': 'panel_scored', 'attempts': index + 1,
                'nonblank': sum(row['scored'] for row in rows)}), flush=True)
scored = [row for row in rows if row['scored']]
pairs = [row['matched_pair'] for row in scored if row['matched_pair'] is not None]
unconditional = [row for row in scored if row['panel_stratum'] == 'unconditional']
unconditional_pairs = [row['matched_pair'] for row in unconditional if row['matched_pair'] is not None]
pair_report = {'unconditional_pairs': len(unconditional_pairs)}
for key in ('map', 'rigid', 'score_null', 'full_null', 'local_peak171'):
    decisions = [pair['energy'][key][0] < pair['energy'][key][1]
        if key != 'local_peak171' else pair[key][0] > pair[key][1]
        for pair in unconditional_pairs]
    pair_report[key] = {'correct': sum(decisions),
        'fraction': float(np.mean(decisions)) if decisions else None}
summary = {'attempts': attempts, 'nonblank': len(scored),
    'unconditional': readout(unconditional),
    'orientation_conditioned_45_55': readout([row for row in scored
        if row['panel_stratum'] != 'unconditional']),
    'sparse_valid_lt_0p10': readout([row for row in scored
        if row['valid_target_fraction'] < .1]),
    'matched_pairs': {'all': len(pairs), **pair_report},
    'pair_gate': {'eligible': len(unconditional_pairs) >= 40,
        'passed': len(unconditional_pairs) >= 40
        and pair_report['map']['fraction'] >= .75
        and pair_report['map']['fraction'] - pair_report['rigid']['fraction'] >= .10
        and pair_report['map']['fraction'] - pair_report['full_null']['fraction'] >= .10},
    'posterior_calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}
b = summary['unconditional']
m, p, n, z, s = (b[key] for key in ('fused_map', 'prior', 'native121',
    'fused_full_null', 'support_only'))
summary['blind_gate'] = {'passed': bool(summary['pair_gate']['passed']
    and p['initial_mean_mm'] - m['initial_mean_mm'] >= .25
    and n['initial_mean_mm'] - m['initial_mean_mm'] >= .25
    and m['capture_le_1p5'] - p['capture_le_1p5'] >= .10
    and m['capture_le_1p5'] - n['capture_le_1p5'] >= .10
    and b['selected_tissue_mm']['fused_map'] <= b['selected_tissue_mm']['native121']
    and z['initial_mean_mm'] - m['initial_mean_mm'] >= .15
    and m['capture_le_1p5'] - z['capture_le_1p5'] >= .05
    and s['initial_mean_mm'] - m['initial_mean_mm'] >= .15
    and m['capture_le_1p5'] - s['capture_le_1p5'] >= .05)}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
done = {'attempts': attempts, 'nonblank': len(scored),
    'elapsed_seconds': time.perf_counter() - started,
    'output_sha256': {path.name: sha(path) for path in out.iterdir()
                      if path.is_file() and path.name != 'completed.json'},
    'posterior_calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}
(out / 'completed.json').write_text(json.dumps(done, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'nonblank': len(scored),
    'pair_gate': summary['pair_gate'], 'blind_gate': summary['blind_gate']},
    allow_nan=False), flush=True)
