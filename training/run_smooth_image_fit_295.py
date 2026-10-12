"""Bounded TRAIN-inner exact-pose image-only smooth-control diagnostic."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_ribbon_v6 import (
    project_surface_affine_out, ribbon_derivative_frobenius_bound,
)
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64


SEED, SIDE, CONTROL, SITES, UPDATES = 20261012295, 192, 16, 512, 120
STEPS = (0, 40, 80, 120)
ARMS = ('privileged', 'observable')
LIMIT = .35
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/SMOOTH_IMAGE_FIT_295_PROTOCOL_20261012.md'
RUN = ROOT / 'reports/smooth_image_fit_295_001'
PRIOR = {
    '281': ('exact_map_capacity_281_001', ('inner_panel.pt',)),
    '282': ('image_map_282_002', ('inner_panel.pt', 'training_schedule.jsonl')),
    '286': ('pose_gauge_map_286_001', ('inner_panel.pt', 'training_schedule.jsonl')),
    '287': ('map_feature_287_001', ('inner_panel.pt', 'training.jsonl')),
    '289': ('spatial_pair_map_289_002', ('inner_panel.pt', 'training.jsonl')),
    '292': ('multiscale_pair_map_292_001', ('inner_panel.pt', 'training.jsonl')),
    '294': ('balanced_gated_pair_map_294_001', ('inner_panel.pt', 'training.jsonl')),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rigid_render(atlas, state, reflection, offsets, weights):
    centre, frame, basis = full_frame_state_to_components(state)
    axis = torch.arange(SIDE, device=state.device, dtype=state.dtype) / SIDE
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - xx, xx)
    chart = torch.stack((sx, yy.expand_as(sx)), -1)
    rigid = normalized_raster_to_ccf(
        centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    slab = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
    rendered = render_finite_thickness_coordinate_grid(
        atlas, slab, (0., 0., 0.), (25., 25., 25.), weights)
    return rendered, rigid, frame, basis


def constrained_field(logits, reflection, basis, offsets):
    controls = 1000 * logits.tanh()
    requested = F.interpolate(controls, (SIDE, SIDE), mode='bilinear', align_corners=False)
    requested = torch.cat((requested, torch.zeros_like(requested[:, :1])), 1)
    local, removed_affine = project_surface_affine_out(requested)
    canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
    before = ribbon_derivative_frobenius_bound(
        canonical, torch.zeros_like(canonical), basis, offsets)
    scale = (LIMIT / before.clamp_min(LIMIT)).clamp_max(1.)
    return controls, local * scale[:, None, None, None], removed_affine, before, scale


def image_objective(logits, image, atlas_render, mask, reflection, basis, offsets, base_grid):
    controls, local, removed_affine, before, scale = constrained_field(
        logits, reflection, basis, offsets)
    chart_delta = torch.einsum('bij,bjhw->bihw', torch.linalg.inv(basis), local[:, :2])
    sign_x = torch.where(reflection.bool(), -1., 1.)[:, None, None]
    grid = base_grid + torch.stack((2 * sign_x * chart_delta[:, 0],
                                   2 * chart_delta[:, 1]), -1)
    warped = F.grid_sample(atlas_render, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=False)
    support = warped[:, 1]
    observed = image[:, 0]
    predicted = warped[:, 0] / support.clamp_min(1e-4)
    count = mask.sum()
    observed_mean = (mask * observed).sum() / count
    predicted_mean = (mask * predicted).sum() / count
    observed_centered = observed - observed_mean
    predicted_centered = predicted - predicted_mean
    covariance = (mask * observed_centered * predicted_centered).sum()
    observed_power = (mask * observed_centered.square()).sum()
    predicted_power = (mask * predicted_centered.square()).sum()
    correlation = covariance / (observed_power * predicted_power).sqrt().clamp_min(1e-12)
    magnitude = (local[:, :2] / 1000).square().mean()
    control_mm = controls / 1000
    bend_x = (control_mm[..., 2:] - 2 * control_mm[..., 1:-1] + control_mm[..., :-2]).square().mean()
    bend_y = (control_mm[..., 2:, :] - 2 * control_mm[..., 1:-1, :] + control_mm[..., :-2, :]).square().mean()
    bending = (bend_x + bend_y) / 2
    support_deficit = (mask * (1 - support)).sum() / count
    loss = 1 - correlation.abs() + .01 * magnitude + .01 * bending + .1 * support_deficit
    diagnostics = {'correlation': correlation, 'magnitude': magnitude,
        'bending': bending, 'support_deficit': support_deficit,
        'support_exit_fraction': (mask * (support <= .9)).sum() / count,
        'raster_exit_fraction': (mask * (grid.abs().amax(-1) > 1)).sum() / count,
        'removed_affine': removed_affine, 'prelimit_derivative': before,
        'deformation_rescale': scale, 'local': local, 'controls': controls}
    return loss, diagnostics


def summary_subset(rows, arm):
    if not rows:
        return {'n': 0, 'rigid_mean_mm': None, 'fitted_mean_mm': None,
                'gain_mean_mm': None, 'image_objective_improved': 0}
    fits = [row['arms'][arm] for row in rows]
    return {'n': len(rows),
        'rigid_mean_mm': float(np.mean([row['rigid_site_error_mm'] for row in rows])),
        'fitted_mean_mm': float(np.mean([fit['fitted_site_error_mm'] for fit in fits])),
        'gain_mean_mm': float(np.mean([fit['gain_mm'] for fit in fits])),
        'image_objective_improved': sum(fit['selected_image_objective'] <
            fit['zero_image_objective'] for fit in fits)}


def main():
    assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    prior_ids, prior_hashes = set(), {}
    for candidate, (directory_name, names) in PRIOR.items():
        directory = ROOT / 'runs' / directory_name
        for name in names:
            path = directory / name
            prior_hashes[f'{candidate}/{name}'] = digest(path)
            if name.endswith('.pt'):
                prior_ids.update(item['physical_section_id'] for item in
                    torch.load(path, map_location='cpu', weights_only=False))
            else:
                prior_ids.update(json.loads(line)['physical_section_id'] for line in
                    path.open(encoding='utf8'))
    context = load_streaming_synthetic_v7_64(device='cuda')
    synthetic_provenance = context['provenance']
    source_names = (
        'run_smooth_image_fit_295.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_slide_artifacts_v7.py',
        'arbitrary_plane_one_shot_slide_artifacts_v6.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'arbitrary_plane_allen_atlas_binding_v6.py',
        'arbitrary_plane_geometry.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
    )
    source_hashes = {name: digest(SOURCE / name) for name in source_names}
    RUN.mkdir(parents=True, exist_ok=False)
    panel, attempted = [], 0
    rng = np.random.default_rng(SEED + 1)
    with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
        for base in range(56, 64):
            for slot in range(4):
                accepted = False
                for _ in range(1024):
                    attempted += 1
                    subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                    draw_seed = SEED + 3_000_000_000 + attempted
                    with torch.no_grad():
                        old = sample_one_shot_slide_artifacts_v9(
                            context, [subject_index], draw_seed, SIDE)
                    identity = old['provenance'][0]
                    section_id = identity['physical_section_id']
                    valid_count = int(old['valid_mask'][0].sum())
                    reason = ('ineligible' if not bool(old['eligible'][0]) else
                        'fewer_than_512_valid_sites' if valid_count < SITES else
                        'prior_or_duplicate_id' if section_id in prior_ids else 'accepted')
                    if reason == 'accepted':
                        with torch.no_grad():
                            sample = sample_postartifact_gauge_v1(
                                context, [subject_index], draw_seed, SIDE, source_sample=old)
                        identity = sample['provenance'][0]
                    draws.write(json.dumps({'base_index': base, 'slot': slot,
                        'attempt': attempted, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'physical_section_id': section_id,
                        'valid_count': valid_count, 'reason': reason,
                        'provenance': identity}, allow_nan=False) + '\n')
                    if reason != 'accepted':
                        del old
                        continue
                    assert identity['split'] == 'train'
                    assert identity['base_lineage']['animal_index'] == base
                    prior_ids.add(section_id)
                    valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
                    indices = valid[torch.linspace(0, len(valid) - 1, SITES,
                        device=valid.device).long()]
                    assert len(indices.unique()) == SITES
                    target = sample['centre'][0].reshape(-1, 3)[indices]
                    state = sample['state']
                    reflection = sample['reflection']
                    offsets = sample['offsets']
                    weights = sample['weights']
                    with torch.no_grad():
                        atlas_render, rigid, frame, _ = rigid_render(
                            context['atlas'], state, reflection, offsets, weights)
                    image = sample['inputs'][:, :1]
                    brush_available = bool(sample['inputs'][0, 2, 0, 0])
                    brush = sample['brush_mask'][0] if brush_available else None
                    rigid_support = atlas_render[0, 1] > .9
                    masks = {'privileged': sample['valid_mask'][0] & rigid_support,
                        'observable': rigid_support & brush if brush_available else rigid_support}
                    assert all(int(mask.sum()) > 2 for mask in masks.values())
                    normal = frame[0, :, 2]
                    artifacts = identity['one_shot_slide_artifacts_v3']
                    item = {'base_index': base, 'slot': slot,
                        'subject_index': subject_index, 'draw_seed': draw_seed,
                        'physical_section_id': section_id, 'provenance': identity,
                        'state': state.cpu(), 'reflection': reflection.cpu(),
                        'pre_artifact_state_sha256': tensor_digest(sample['pre_artifact_state']),
                        'post_artifact_state_sha256': tensor_digest(state),
                        'offsets': offsets.cpu(), 'weights': weights.cpu(),
                        'observed_image': image.cpu(), 'atlas_render': atlas_render.cpu(),
                        'rigid_surface_ccf_um': rigid.cpu(),
                        'fixed_masks': {name: mask.cpu() for name, mask in masks.items()},
                        'site_indices': indices.cpu(), 'site_target_ccf_um': target.cpu(),
                        'brush_available': brush_available,
                        'brush_pixel_count': int(brush.sum()) if brush_available else None,
                        'rigid_support_pixel_count': int(rigid_support.sum()),
                        'mask_pixel_count': {name: int(mask.sum()) for name, mask in masks.items()},
                        'valid_fraction': valid_count / (SIDE * SIDE),
                        'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                        'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
                        'event_flags': {name: bool(artifacts['events'][name]) for name in
                            ('fragment', 'fold', 'bubble', 'tile_seam')},
                        'tissue_placed': artifacts['tissue_placed'],
                        'artifact_pixel_counts': artifacts['pixel_counts'],
                        'observed_image_sha256': tensor_digest(image),
                        'full_coordinate_target_sha256': tensor_digest(sample['centre']),
                        'site_target_sha256': tensor_digest(target),
                        'rigid_atlas_render_sha256': tensor_digest(atlas_render),
                        'rigid_surface_sha256': tensor_digest(rigid),
                        'mask_sha256': {name: tensor_digest(mask) for name, mask in masks.items()},
                        'brush_sha256': tensor_digest(brush) if brush_available else None}
                    panel.append(item)
                    accepted = True
                    del old, sample
                    break
                assert accepted
    assert len(panel) == 32 and len({item['physical_section_id'] for item in panel}) == 32
    torch.save(panel, RUN / 'inner_panel.pt')
    del context
    torch.cuda.empty_cache()
    config = {'candidate': 295, 'protocol_sha256': digest(PROTOCOL),
        'source_sha256': source_hashes, 'prior_file_sha256': prior_hashes,
        'prior_physical_section_count': len(prior_ids) - len(panel),
        'synthetic_provenance': synthetic_provenance,
        'seed': SEED, 'attempted_draws': attempted, 'accepted_sections': len(panel),
        'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
        'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
        'train_base_indices': list(range(56, 64)), 'sections_per_base': 4,
        'minimum_distinct_valid_sites_for_inclusion': SITES,
        'source_shape': [SIDE, SIDE], 'control_shape': [CONTROL, CONTROL],
        'site_count_per_section': SITES, 'readout_steps': STEPS, 'arms': ARMS,
        'atlas_render': 'one fixed 192x192 finite-thickness intensity/support pair at robust postartifact gauge v1 state, reflection, sample offsets and weights',
        'control_rule': '1000*tanh(logits) um on 16x16 local in-plane controls; bilinear upsample align_corners=False to 192; through-plane zero; full-canvas affine projection; reflected canonical derivative bound; global rescale <=0.35',
        'warp_rule': 'bilinear grid_sample of fixed atlas pair; inverse in-plane basis converts constrained local um to chart delta; observed x sign reversed under reflection; align_corners=False and zero padding',
        'mask_rule': 'fixed privileged=synthetic valid and rigid support>0.9; fixed observable=rigid support>0.9 and optional user brush; without brush rigid support only',
        'objective': '1-abs(masked Pearson of observed versus warped intensity divided by warped support clamped at 1e-4)+0.01*mean((constrained in-plane dense field/1000)^2)+0.01*mean squared second differences of requested 16x16 controls/1000+0.1*mean(1-warped support) on fixed mask',
        'optimizer': 'per-section per-arm independent Adam lr=0.1, zero logits, 120 updates',
        'best_step_rule': 'minimum image objective among readouts 0,40,80,120; coordinate truth excluded',
        'map_derivative_limit': LIMIT,
        'coordinate_evaluation': '512 distinct prespecified valid source-grid sites, after each image-only readout only',
        'no_learned_model_or_checkpoint': True,
        'no_dev_cal_final_or_public_benchmark': True,
        'torch': torch.__version__, 'numpy': np.__version__}
    (RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')
    axis = (torch.arange(SIDE, device='cuda') + .5) * (2 / SIDE) - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    base_grid = torch.stack((xx, yy), -1)[None]
    sections, selected_fields = [], {}
    with (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
         (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readouts, \
         (RUN / 'sections.jsonl').open('w', encoding='utf8') as output:
        for item in panel:
            section_id = item['physical_section_id']
            state = item['state'].cuda()
            reflection = item['reflection'].cuda()
            offsets = item['offsets'].cuda()
            _, frame, basis = full_frame_state_to_components(state)
            image = item['observed_image'].cuda()
            atlas_render = item['atlas_render'].cuda()
            rigid = item['rigid_surface_ccf_um'].cuda()
            indices = item['site_indices'].cuda()
            target = item['site_target_ccf_um'].cuda()
            rigid_points = rigid.reshape(-1, 3)[indices]
            rigid_error = float((rigid_points - target).norm(dim=-1).mean() / 1000)
            row = {'base_index': item['base_index'], 'slot': item['slot'],
                'subject_index': item['subject_index'], 'draw_seed': item['draw_seed'],
                'physical_section_id': section_id,
                'donor_animal_id': item['provenance']['base_lineage']['animal_id'],
                'base_lineage': item['provenance']['base_lineage'],
                'virtual_subject_id': item['provenance']['virtual_subject_id'],
                'virtual_index': item['provenance']['virtual_index'],
                'true_postartifact_state': item['state'][0].tolist(),
                'true_reflection': int(item['reflection'][0]),
                'site_count': SITES, 'rigid_site_error_mm': rigid_error,
                'brush_available': item['brush_available'],
                'brush_pixel_count': item['brush_pixel_count'],
                'rigid_support_pixel_count': item['rigid_support_pixel_count'],
                'mask_pixel_count': item['mask_pixel_count'],
                'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'obliquity_deg': item['obliquity_deg'],
                'event_flags': item['event_flags'],
                'tissue_placed': item['tissue_placed'],
                'artifact_pixel_counts': item['artifact_pixel_counts'],
                'input_sha256': {name: item[name] for name in (
                    'pre_artifact_state_sha256', 'post_artifact_state_sha256',
                    'observed_image_sha256', 'full_coordinate_target_sha256',
                    'site_target_sha256', 'rigid_atlas_render_sha256',
                    'rigid_surface_sha256', 'mask_sha256', 'brush_sha256')},
                'arms': {}, 'provenance': item['provenance']}
            for arm in ARMS:
                mask = item['fixed_masks'][arm].cuda().float()[None]
                logits = torch.nn.Parameter(torch.zeros((1, 2, CONTROL, CONTROL), device='cuda'))
                optimizer = torch.optim.Adam([logits], lr=.1)
                best, best_logits, best_local, best_controls = None, None, None, None
                zero = None
                for step in range(UPDATES + 1):
                    if step in STEPS:
                        with torch.no_grad():
                            loss, d = image_objective(logits, image, atlas_render,
                                mask, reflection, basis, offsets, base_grid)
                            local = d['local']
                            mapped = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
                            mapped_points = mapped.reshape(-1, 3)[indices]
                            site_error = float((mapped_points - target).norm(dim=-1).mean() / 1000)
                            site_shift = (mapped_points - rigid_points).norm(dim=-1) / 1000
                            canonical = torch.where(reflection[:, None, None, None].bool(),
                                                    local.flip(-1), local)
                            after = ribbon_derivative_frobenius_bound(
                                canonical, torch.zeros_like(canonical), basis, offsets)
                            field_norm = local[:, :2].norm(dim=1) / 1000
                            result = {'physical_section_id': section_id,
                                'base_index': item['base_index'], 'slot': item['slot'],
                                'arm': arm, 'step': step,
                                'mask_pixel_count': item['mask_pixel_count'][arm],
                                'image_objective': float(loss),
                                'absolute_correlation': float(d['correlation'].abs()),
                                'signed_correlation': float(d['correlation']),
                                'magnitude_term': float(d['magnitude']),
                                'bending_term': float(d['bending']),
                                'warped_support_deficit': float(d['support_deficit']),
                                'warped_support_exit_fraction': float(d['support_exit_fraction']),
                                'raster_exit_fraction': float(d['raster_exit_fraction']),
                                'rigid_site_error_mm': rigid_error,
                                'fitted_site_error_mm': site_error,
                                'gain_mm': rigid_error - site_error,
                                'site_shift_mean_mm': float(site_shift.mean()),
                                'site_shift_rms_mm': float(site_shift.square().mean().sqrt()),
                                'field_displacement_mean_mm': float(field_norm.mean()),
                                'field_displacement_rms_mm': float(field_norm.square().mean().sqrt()),
                                'field_displacement_max_mm': float(field_norm.amax()),
                                'prelimit_derivative_bound': float(d['prelimit_derivative'][0]),
                                'deformation_rescale': float(d['deformation_rescale'][0]),
                                'postlimit_derivative_bound': float(after[0]),
                                'removed_affine_local_um_3_by_intercept_s_t':
                                    d['removed_affine'][0].tolist(),
                                'control_absmax_um': float(d['controls'].abs().amax()),
                                'logit_absmax': float(logits.abs().amax())}
                        assert np.isfinite(result['image_objective'])
                        assert result['postlimit_derivative_bound'] <= LIMIT + 1e-5
                        readouts.write(json.dumps(result, allow_nan=False) + '\n')
                        if step == 0:
                            zero = result
                        if best is None or result['image_objective'] < best['image_objective']:
                            best = result.copy()
                            best_logits = logits.detach().cpu().clone()
                            best_local = d['local'].detach().cpu().clone()
                            best_controls = d['controls'].detach().cpu().clone()
                    if step == UPDATES:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = image_objective(logits, image, atlas_render,
                        mask, reflection, basis, offsets, base_grid)
                    loss.backward()
                    objective_before = float(loss.detach())
                    gradient = float(logits.grad.norm())
                    optimizer.step()
                    training.write(json.dumps({'physical_section_id': section_id,
                        'arm': arm, 'step': step + 1,
                        'image_objective_before_update': objective_before,
                        'logit_gradient_norm': gradient,
                        'post_update_logit_absmax': float(logits.detach().abs().amax())},
                        allow_nan=False) + '\n')
                assert best is not None and zero is not None
                row['arms'][arm] = {'best_step': best['step'],
                    'zero_image_objective': zero['image_objective'],
                    'zero_absolute_correlation': zero['absolute_correlation'],
                    'selected_image_objective': best['image_objective'],
                    'selected_absolute_correlation': best['absolute_correlation'],
                    'selected_readout': best,
                    'fitted_site_error_mm': best['fitted_site_error_mm'],
                    'gain_mm': best['gain_mm'],
                    'selected_logits_sha256': tensor_digest(best_logits),
                    'selected_controls_sha256': tensor_digest(best_controls),
                    'selected_field_sha256': tensor_digest(best_local)}
                selected_fields[(section_id, arm)] = {'best_step': best['step'],
                    'logits': best_logits, 'requested_controls_um': best_controls,
                    'constrained_field_local_um': best_local}
                print(json.dumps({'event': 'section_fit', 'physical_section_id': section_id,
                    'arm': arm, 'best_step': best['step'],
                    'gain_mm': best['gain_mm'],
                    'image_objective': best['image_objective']}), flush=True)
            output.write(json.dumps(row, allow_nan=False) + '\n')
            sections.append(row)
            training.flush()
            readouts.flush()
            output.flush()
    torch.save(selected_fields, RUN / 'selected_fields.pt')
    summaries, gates = {}, {}
    for arm, threshold in (('privileged', .030), ('observable', .015)):
        overall = summary_subset(sections, arm)
        by_base = {str(base): summary_subset(
            [row for row in sections if row['base_index'] == base], arm)
            for base in range(56, 64)}
        summaries[arm] = {**overall, 'by_base': by_base,
            'positive_base_count': sum(value['gain_mean_mm'] > 0 for value in by_base.values()),
            'by_obliquity_30deg': {str(flag).lower(): summary_subset(
                [row for row in sections if (row['obliquity_deg'] >= 30) == flag], arm)
                for flag in (False, True)},
            'by_sparse_valid_under_15pct': {str(flag).lower(): summary_subset(
                [row for row in sections if (row['valid_fraction'] < .15) == flag], arm)
                for flag in (False, True)},
            'by_brush': {str(flag).lower(): summary_subset(
                [row for row in sections if row['brush_available'] == flag], arm)
                for flag in (False, True)},
            'by_artifact': {event: {str(flag).lower(): summary_subset(
                [row for row in sections if row['event_flags'][event] == flag], arm)
                for flag in (False, True)} for event in
                ('fragment', 'fold', 'bubble', 'tile_seam')},
            'max_postlimit_derivative_bound': max(
                row['arms'][arm]['selected_readout']['postlimit_derivative_bound']
                for row in sections),
            'selected_field_rms_mean_mm': float(np.mean([
                row['arms'][arm]['selected_readout']['field_displacement_rms_mm']
                for row in sections])),
            'selected_warped_support_exit_fraction_mean': float(np.mean([
                row['arms'][arm]['selected_readout']['warped_support_exit_fraction']
                for row in sections]))}
        gates[arm] = {'gain_threshold_mm': threshold,
            'mean_gain_at_least_threshold': overall['gain_mean_mm'] >= threshold,
            'at_least_six_positive_bases': summaries[arm]['positive_base_count'] >= 6,
            'at_least_24_improved_image_objectives': overall['image_objective_improved'] >= 24}
        gates[arm]['pass'] = all(value for key, value in gates[arm].items()
                                 if key != 'gain_threshold_mm')
    if gates['privileged']['pass'] and gates['observable']['pass']:
        interpretation = 'TRAIN-inner signal to pursue an image-loss/unrolled smooth-control model'
    elif gates['privileged']['pass']:
        interpretation = 'segmentation-information ceiling only; prioritize inference-valid tissue/quality weighting'
    else:
        interpretation = 'revisit the image objective and its local basin before scaling a learned model'
    output_names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
        'training.jsonl', 'readouts.jsonl', 'sections.jsonl', 'selected_fields.pt')
    completed = {'candidate': 295, 'eligible_sections': len(sections),
        'attempted_draws': attempted, 'summary': summaries, 'gate': gates,
        'gate_pass': gates['privileged']['pass'] and gates['observable']['pass'],
        'interpretation': interpretation,
        'result_role': 'bounded TRAIN-inner exact-pose image-only registration diagnostic; no deployable predictor',
        'protocol_sha256': config['protocol_sha256'],
        'source_sha256': source_hashes,
        'output_sha256': {name: digest(RUN / name) for name in output_names},
        'elapsed_seconds': time.perf_counter() - started,
        'dev_cal_final_or_public_benchmark_used': False}
    (RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'gate': gates,
        'elapsed_seconds': completed['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
