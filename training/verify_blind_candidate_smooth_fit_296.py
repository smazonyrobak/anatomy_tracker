"""Post-exit audit of 296. Run only after its process has exited.

From the worktree root: python -m training.verify_blind_candidate_smooth_fit_296
This script never writes to the run directory or imports either fit runner.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/blind_candidate_smooth_fit_296_001'
PROTOCOL = SOURCE.parent / 'docs/publication/BLIND_CANDIDATE_SMOOTH_FIT_296_PROTOCOL_20261012.md'
CHECKPOINT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
CHECKPOINT_SHA256 = 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
SEED, SIDE, SITES, CONTROL = 20261012296, 192, 512, 16
STEPS = (0, 40, 80, 120)
OUTPUT_NAMES = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
                'training.jsonl', 'readouts.jsonl', 'sections.jsonl', 'selected_fields.pt')
SOURCE_NAMES = (
    'run_blind_candidate_smooth_fit_296.py', 'run_smooth_image_fit_295.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py', 'arbitrary_plane_one_shot_stream.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'global_atlas_contrast_090.py')
PRIOR = {
    '281': ('runs/exact_map_capacity_281_001', ('inner_panel.pt',)),
    '282': ('runs/image_map_282_002', ('inner_panel.pt', 'training_schedule.jsonl')),
    '286': ('runs/pose_gauge_map_286_001', ('inner_panel.pt', 'training_schedule.jsonl')),
    '287': ('runs/map_feature_287_001', ('inner_panel.pt', 'training.jsonl')),
    '289': ('runs/spatial_pair_map_289_002', ('inner_panel.pt', 'training.jsonl')),
    '292': ('runs/multiscale_pair_map_292_001', ('inner_panel.pt', 'training.jsonl')),
    '294': ('runs/balanced_gated_pair_map_294_001', ('inner_panel.pt', 'training.jsonl')),
    '295': ('reports/smooth_image_fit_295_001', ('inner_panel.pt',)),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def records(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def same(actual, expected, label):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label
        for key, value in expected.items():
            same(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), label
        for index, value in enumerate(expected):
            same(actual[index], value, f'{label}[{index}]')
    elif isinstance(expected, int) and not isinstance(expected, bool):
        assert actual == expected, label
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-6), label
    else:
        assert actual == expected, label


def subset(rows):
    if not rows:
        return {'n': 0}
    first = np.array([r['selectors']['prior_first']['rigid_site_error_mm'] for r in rows])
    no_fit = np.array([r['selectors']['no_fit_image']['rigid_site_error_mm'] for r in rows])
    fitted = np.array([r['selectors']['fitted_image']['rigid_site_error_mm'] for r in rows])
    first_map = np.array([r['selectors']['prior_first']['fitted_site_error_mm'] for r in rows])
    fit_map = np.array([r['selectors']['fitted_image']['fitted_site_error_mm'] for r in rows])
    selected = [r['candidates'][r['selectors']['fitted_image']['candidate_index']] for r in rows]
    return {'n': len(rows),
        'prior_first_rigid_mean_mm': float(first.mean()),
        'no_fit_image_rigid_mean_mm': float(no_fit.mean()),
        'fitted_image_rigid_mean_mm': float(fitted.mean()),
        'fitted_image_rigid_gain_over_prior_mm': float((first-fitted).mean()),
        'fitted_image_rigid_gain_over_no_fit_mm': float((no_fit-fitted).mean()),
        'prior_first_fitted_map_mean_mm': float(first_map.mean()),
        'fitted_image_fitted_map_mean_mm': float(fit_map.mean()),
        'fitted_image_map_change_from_prior_mm': float((fit_map-first_map).mean()),
        'positive_rigid_gain_cases': int((first>fitted).sum()),
        'prior_first_capture_1p5mm': int((first<=1.5).sum()),
        'fitted_image_capture_1p5mm': int((fitted<=1.5).sum()),
        'mean_prior_first_support_fraction': float(np.mean([r['selectors']['prior_first']['support_fraction'] for r in rows])),
        'mean_fitted_image_support_fraction': float(np.mean([r['selectors']['fitted_image']['support_fraction'] for r in rows])),
        'mean_prior_first_normal_axis_error_deg': float(np.mean([r['candidates'][0]['normal_axis_error_deg'] for r in rows])),
        'mean_fitted_image_normal_axis_error_deg': float(np.mean([c['normal_axis_error_deg'] for c in selected])),
        'mean_prior_first_selected_image_objective': float(np.mean([r['selectors']['prior_first']['selected_image_objective'] for r in rows])),
        'mean_fitted_image_selected_image_objective': float(np.mean([c['selected_image_objective'] for c in selected])),
        'mean_fitted_image_field_rms_mm': float(np.mean([c['selected_readout']['field_displacement_rms_mm'] for c in selected])),
        'mean_fitted_image_warped_support_exit_fraction': float(np.mean([c['selected_readout']['warped_support_exit_fraction'] for c in selected]))}


def summaries_and_gate(rows):
    overall = subset(rows)
    by_base = {str(base): subset([r for r in rows if r['base_index'] == base]) for base in range(56, 64)}
    matched = [r['near_far_pair'] for r in rows if r['near_far_pair']['eligible']]
    rate = sum(p['near_preferred'] for p in matched) / len(matched) if matched else None
    summary = {**overall, 'by_base': by_base,
        'positive_base_count': sum(v['fitted_image_rigid_gain_over_prior_mm'] > 0 for v in by_base.values()),
        'matched_near_far_case_count': len(matched),
        'matched_near_far_win_count': sum(p['near_preferred'] for p in matched),
        'matched_near_far_win_rate': rate,
        'by_obliquity_30deg': {str(flag).lower(): subset([r for r in rows if (r['obliquity_deg'] >= 30) == flag]) for flag in (False, True)},
        'by_sparse_valid_under_15pct': {str(flag).lower(): subset([r for r in rows if (r['valid_fraction'] < .15) == flag]) for flag in (False, True)},
        'by_brush': {str(flag).lower(): subset([r for r in rows if r['brush_available'] == flag]) for flag in (False, True)},
        'by_artifact': {event: {str(flag).lower(): subset([r for r in rows if r['event_flags'][event] == flag]) for flag in (False, True)} for event in ('fragment', 'fold', 'bubble', 'tile_seam')},
        'by_prior_first_support_under_half': {str(flag).lower(): subset([r for r in rows if (r['selectors']['prior_first']['support_fraction'] < .5) == flag]) for flag in (False, True)},
        'best_step_distribution': {str(step): sum(c['best_step'] == step for r in rows for c in r['candidates']) for step in STEPS},
        'true_control_rigid_mean_mm': float(np.mean([r['true_control']['rigid_site_error_mm'] for r in rows])),
        'true_control_fitted_mean_mm': float(np.mean([r['true_control']['fitted_site_error_mm'] for r in rows])),
        'max_postlimit_derivative_bound': max(c['selected_readout']['postlimit_derivative_bound'] for r in rows for c in r['candidates'])}
    gate = {'rigid_gain_over_prior_at_least_0p25mm': overall['fitted_image_rigid_gain_over_prior_mm'] >= .25,
        'rigid_gain_over_no_fit_at_least_0p15mm': overall['fitted_image_rigid_gain_over_no_fit_mm'] >= .15,
        'at_least_six_positive_bases': summary['positive_base_count'] >= 6,
        'fitted_map_no_more_than_0p05mm_worse_than_prior': overall['fitted_image_map_change_from_prior_mm'] <= .05,
        'at_least_16_matched_near_far_cases': len(matched) >= 16,
        'matched_near_far_win_rate_at_least_0p70': rate >= .70 if rate is not None else False}
    gate['pass'] = all(gate.values())
    return summary, gate


def render(atlas, state, reflection, offsets, weights):
    from training.arbitrary_plane_full_frame_primitives import (
        full_frame_state_to_components, render_finite_thickness_coordinate_grid)
    from training.arbitrary_plane_geometry import normalized_raster_to_ccf
    centre, frame, basis = full_frame_state_to_components(state)
    axis = torch.arange(SIDE, device=state.device, dtype=state.dtype) / SIDE
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    sx = torch.where(reflection[:, None, None].bool(), (SIDE-1)/SIDE-xx, xx)
    chart = torch.stack((sx, yy.expand_as(sx)), -1)
    rigid = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
    slab = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
    image = render_finite_thickness_coordinate_grid(atlas, slab, (0., 0., 0.), (25., 25., 25.), weights)
    return image, rigid, frame, basis


def rescore(logits, image, atlas_render, mask, reflection, basis, offsets, rigid, frame, indices, target):
    from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out, ribbon_derivative_frobenius_bound
    controls = 1000 * logits.tanh()
    requested = F.interpolate(controls, (SIDE, SIDE), mode='bilinear', align_corners=False)
    requested = torch.cat((requested, torch.zeros_like(requested[:, :1])), 1)
    local, removed = project_surface_affine_out(requested)
    canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
    before = ribbon_derivative_frobenius_bound(canonical, torch.zeros_like(canonical), basis, offsets)
    scale = (.35 / before.clamp_min(.35)).clamp_max(1.)
    local = local * scale[:, None, None, None]
    delta = torch.einsum('bij,bjhw->bihw', torch.linalg.inv(basis), local[:, :2])
    axis = (torch.arange(SIDE, device=image.device) + .5) * (2 / SIDE) - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None] + torch.stack((
        2 * torch.where(reflection.bool(), -1., 1.)[:, None, None] * delta[:, 0],
        2 * delta[:, 1]), -1)
    warped = F.grid_sample(atlas_render, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    support = warped[:, 1]
    observed, predicted = image[:, 0], warped[:, 0] / support.clamp_min(1e-4)
    count = mask.sum()
    o = observed - (mask * observed).sum() / count
    p = predicted - (mask * predicted).sum() / count
    corr = (mask * o * p).sum() / ((mask * o.square()).sum() * (mask * p.square()).sum()).sqrt().clamp_min(1e-12)
    magnitude = (local[:, :2] / 1000).square().mean()
    c = controls / 1000
    bending = ((c[..., 2:] - 2*c[..., 1:-1] + c[..., :-2]).square().mean() +
               (c[..., 2:, :] - 2*c[..., 1:-1, :] + c[..., :-2, :]).square().mean()) / 2
    deficit = (mask * (1 - support)).sum() / count
    mapped = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
    points = rigid.reshape(-1, 3)[indices]
    moved = mapped.reshape(-1, 3)[indices]
    field_norm = local[:, :2].norm(dim=1) / 1000
    after = ribbon_derivative_frobenius_bound(
        torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local),
        torch.zeros_like(local), basis, offsets)
    metrics = {'image_objective': float(1-corr.abs()+.01*magnitude+.01*bending+.1*deficit),
        'absolute_correlation': float(corr.abs()), 'signed_correlation': float(corr),
        'magnitude_term': float(magnitude), 'bending_term': float(bending),
        'warped_support_deficit': float(deficit),
        'warped_support_exit_fraction': float((mask * (support <= .9)).sum() / count),
        'raster_exit_fraction': float((mask * (grid.abs().amax(-1) > 1)).sum() / count),
        'rigid_site_error_mm': float((points-target).norm(dim=-1).mean()/1000),
        'fitted_site_error_mm': float((moved-target).norm(dim=-1).mean()/1000),
        'site_shift_mean_mm': float((moved-points).norm(dim=-1).mean()/1000),
        'field_displacement_mean_mm': float(field_norm.mean()),
        'field_displacement_rms_mm': float(field_norm.square().mean().sqrt()),
        'field_displacement_max_mm': float(field_norm.amax()),
        'prelimit_derivative_bound': float(before[0]),
        'deformation_rescale': float(scale[0]),
        'postlimit_derivative_bound': float(after[0]),
        'removed_affine_local_um_3_by_intercept_s_t': removed[0].tolist(),
        'control_absmax_um': float(controls.abs().amax()),
        'logit_absmax': float(logits.abs().amax())}
    return metrics, controls, local


def main():
    assert RUN.drive.upper() == SOURCE.drive.upper() == 'I:'
    assert (RUN / 'completed.json').is_file(), '296 must finish before this verifier runs'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    completed = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
    config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
    assert completed['candidate'] == config['candidate'] == 296
    assert set(completed['output_sha256']) == set(OUTPUT_NAMES)
    for name in OUTPUT_NAMES:
        assert sha(RUN / name) == completed['output_sha256'][name], name
    assert sha(PROTOCOL) == completed['protocol_sha256'] == config['protocol_sha256']
    assert set(config['source_sha256']) == set(SOURCE_NAMES)
    assert completed['source_sha256'] == config['source_sha256']
    for name in SOURCE_NAMES:
        assert sha(SOURCE / name) == config['source_sha256'][name], name
    assert sha(CHECKPOINT) == CHECKPOINT_SHA256 == config['pose_checkpoint_sha256'] == completed['pose_checkpoint_sha256']
    assert config['seed'] == SEED and config['candidate_count'] == 16
    assert config['train_base_indices'] == list(range(56, 64)) and config['sections_per_base'] == 4
    assert config['site_count_per_section'] == config['minimum_distinct_valid_sites_for_inclusion'] == SITES
    assert config['source_shape'] == [SIDE, SIDE] and config['control_shape'] == [CONTROL, CONTROL]
    assert tuple(config['readout_steps']) == STEPS and config['map_derivative_limit'] == .35
    assert config['torch'] == torch.__version__ and config['numpy'] == np.__version__
    assert config['no_training_or_new_model'] and config['no_dev_cal_final_or_public_benchmark']
    assert completed['dev_cal_final_or_public_benchmark_used'] is False
    prior_ids, prior_hashes = set(), {}
    for candidate, (directory, names) in PRIOR.items():
        for name in names:
            path = ROOT / directory / name
            prior_hashes[f'{candidate}/{name}'] = sha(path)
            if name.endswith('.pt'):
                prior_ids.update(r['physical_section_id'] for r in torch.load(path, map_location='cpu', weights_only=False))
            else:
                prior_ids.update(r['physical_section_id'] for r in records(path))
    assert prior_hashes == config['prior_file_sha256']
    assert len(prior_ids) == config['prior_physical_section_count']
    synthetic = config['synthetic_provenance']
    assert synthetic['new_independent_train_bases'] == 56 and synthetic['all_train_bases'] == 64
    assert len(synthetic['base_subjects']) == 64 and len(synthetic['virtual_subjects']) == 64*64
    for path, expected in synthetic['input_sha256'].items():
        assert Path(path).drive.upper() == 'I:' and sha(Path(path)) == expected, path
    for name, expected in synthetic['source_sha256'].items():
        assert sha(SOURCE / name) == expected, name
    for kind in ('template', 'annotation'):
        binding = synthetic['atlas_binding']
        assert sha(Path(binding[f'{kind}_path'])) == binding[f'{kind}_sha256']

    draws = records(RUN / 'inner_draws.jsonl')
    panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
    rows = records(RUN / 'sections.jsonl')
    readouts = records(RUN / 'readouts.jsonl')
    training = records(RUN / 'training.jsonl')
    selected = torch.load(RUN / 'selected_fields.pt', map_location='cpu', weights_only=False)
    assert len(draws) == config['attempted_draws'] == completed['attempted_draws']
    assert len(panel) == len(rows) == config['accepted_sections'] == completed['eligible_sections'] == 32
    assert len(readouts) == 32*17*4 and len(training) == 32*17*120 and len(selected) == 32*17
    assert sha(RUN / 'inner_draws.jsonl') == config['inner_draws_sha256']
    assert sha(RUN / 'inner_panel.pt') == config['inner_panel_sha256']
    rng = np.random.default_rng(SEED+1)
    accepted, seen = [], set(prior_ids)
    for number, draw in enumerate(draws, 1):
        base, slot = 56 + len(accepted)//4, len(accepted)%4
        assert (draw['base_index'], draw['slot'], draw['attempt']) == (base, slot, number)
        subject = base*64 + int(rng.integers(64))
        assert draw['subject_index'] == subject and draw['draw_seed'] == SEED+3_000_000_000+number
        section_id = f"stream-v7-seed-{draw['draw_seed']}-row-0-virtual-{subject}"
        assert draw['physical_section_id'] == draw['provenance']['physical_section_id'] == section_id
        assert draw['provenance']['split'] == 'train'
        assert draw['provenance']['base_lineage']['animal_index'] == base
        if draw['reason'] == 'accepted':
            assert draw['valid_count'] >= SITES and section_id not in seen
            seen.add(section_id)
            accepted.append(draw)
        elif draw['reason'] == 'fewer_than_512_valid_sites':
            assert draw['valid_count'] < SITES
        elif draw['reason'] == 'prior_or_duplicate_id':
            assert section_id in seen
        else:
            assert draw['reason'] == 'ineligible'
    assert len(accepted) == 32 and not ({d['physical_section_id'] for d in accepted} & prior_ids)
    assert len({d['physical_section_id'] for d in accepted}) == 32

    grouped_readouts, grouped_training = {}, {}
    for readout in readouts:
        key = (readout['physical_section_id'], readout['candidate_index'])
        grouped_readouts.setdefault(key, []).append(readout)
        expected_objective = (1-readout['absolute_correlation']+.01*readout['magnitude_term']+
                              .01*readout['bending_term']+.1*readout['warped_support_deficit'])
        assert math.isclose(readout['image_objective'], expected_objective, abs_tol=2e-6)
        assert math.isclose(readout['absolute_correlation'], abs(readout['signed_correlation']), abs_tol=1e-6)
        assert readout['postlimit_derivative_bound'] <= .35001
    for update in training:
        grouped_training.setdefault((update['physical_section_id'], update['candidate_index']), []).append(update)
    assert set(grouped_readouts) == set(grouped_training) == {(r['physical_section_id'], j) for r in rows for j in range(17)}
    for key in grouped_readouts:
        assert [r['step'] for r in grouped_readouts[key]] == list(STEPS), key
        assert [r['step'] for r in grouped_training[key]] == list(range(1, 121)), key
        for step in STEPS[:-1]:
            same(grouped_training[key][step]['image_objective_before_update'],
                 grouped_readouts[key][STEPS.index(step)]['image_objective'], f'{key} training step {step}')

    for item, row, draw in zip(panel, rows, accepted):
        sid, base, slot = item['physical_section_id'], item['base_index'], item['slot']
        assert (sid, base, slot) == (row['physical_section_id'], row['base_index'], row['slot'])
        assert (sid, base, slot) == (draw['physical_section_id'], draw['base_index'], draw['slot'])
        assert (item['subject_index'], item['draw_seed']) == (row['subject_index'], row['draw_seed'])
        assert item['provenance'] == row['provenance'] == draw['provenance']
        assert item['provenance']['seed_prefix'] == [item['draw_seed'], item['subject_index'], 0]
        assert item['provenance']['virtual_index'] == item['subject_index']
        assert row['virtual_index'] == item['subject_index']
        virtual = synthetic['virtual_subjects'][item['subject_index']]
        assert virtual['base_index'] == base and virtual['virtual_subject_id'] == row['virtual_subject_id']
        assert virtual['base_lineage'] == row['base_lineage'] == synthetic['base_subjects'][base]['lineage']
        assert row['donor_animal_id'] == row['base_lineage']['animal_id']
        assert item['provenance']['one_shot_slide_artifacts_v9']['sampling'] == 'one independent physical plane per row; no appearance pairing'
        assert item['provenance']['one_shot_slide_artifacts_v9']['code_sha256'] == config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v9.py']
        assert item['provenance']['postartifact_gauge_v1']['source'] == 'caller-provided v9 sample; no new draw'
        assert item['provenance']['postartifact_gauge_v1']['code_sha256'] == config['source_sha256']['arbitrary_plane_postartifact_gauge_v1.py']
        assert item['provenance']['one_shot_slide_artifacts_v3']['code_sha256'] == config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v3.py']
        assert item['provenance']['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] == draw['valid_count']
        assert item['provenance']['one_shot_slide_artifacts_v3']['pixel_counts'] == item['artifact_pixel_counts']
        assert item['provenance']['one_shot_slide_artifacts_v3']['tissue_placed'] == item['tissue_placed']
        assert item['provenance']['one_shot_slide_artifacts_v9']['brush_available'] == item['brush_available']
        assert row['site_count'] == SITES and item['site_indices'].numel() == SITES
        assert len(item['site_indices'].unique()) == SITES and item['site_indices'].min() >= 0 and item['site_indices'].max() < SIDE*SIDE
        assert item['site_target_ccf_um'].shape == (SITES, 3)
        assert item['inputs'].shape == (1, 5, SIDE, SIDE)
        assert torch.equal(item['observed_image'], item['inputs'][:, :1])
        assert item['common_mask'].shape == (SIDE, SIDE) and item['common_mask'].dtype == torch.bool
        assert int(item['common_mask'].sum()) == item['mask_pixel_count'] == row['mask_pixel_count'] > 2
        assert item['candidate_states'].shape == (16, 12)
        assert item['offsets'].shape == item['weights'].shape == (1, 9)
        torch.testing.assert_close(item['weights'][0], torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=item['weights'].dtype)/16)
        torch.testing.assert_close(item['offsets'][0], torch.linspace(-.5, .5, 9)*item['provenance']['thickness_um'], atol=1e-5, rtol=1e-6)
        assert item['candidate_reflections'].shape == item['branch_ids'].shape == item['candidate_prior_log'].shape == (16,)
        assert len(item['branch_ids'].unique()) == 16 and bool(((item['branch_ids'] >= 0) & (item['branch_ids'] < 160)).all())
        assert torch.equal(item['candidate_reflections'].long(), item['branch_ids'] % 2)
        assert bool((item['candidate_prior_log'][:-1] >= item['candidate_prior_log'][1:]).all())
        assert bool(torch.isfinite(item['candidate_states']).all()) and bool(torch.isfinite(item['site_target_ccf_um']).all())
        for name, tensor in (('inputs', item['inputs']), ('observed_image', item['observed_image']),
                             ('true_postartifact_state', item['true_state']), ('site_target', item['site_target_ccf_um']),
                             ('candidate_states', item['candidate_states']), ('common_mask', item['common_mask'])):
            assert tensor_sha(tensor) == item[f'{name}_sha256'] == row['input_sha256'][f'{name}_sha256'], (sid, name)
        assert item['pre_artifact_state_sha256'] == row['input_sha256']['pre_artifact_state_sha256']
        assert item['full_coordinate_target_sha256'] == row['input_sha256']['full_coordinate_target_sha256']
        assert item['brush_sha256'] == row['input_sha256']['brush_sha256']
        assert item['true_state'][0].tolist() == item['provenance']['postartifact_gauge_v1']['post_artifact_state12']
        assert int(item['true_reflection'][0]) == int(item['provenance']['horizontal_reflection'])
        assert row['brush_available'] == item['brush_available'] == bool(item['inputs'][0, 2, 0, 0])
        assert row['brush_pixel_count'] == item['brush_pixel_count']
        assert row['valid_fraction'] == item['valid_fraction'] == draw['valid_count']/(SIDE*SIDE)
        assert row['event_flags'] == item['event_flags']
        assert all(item['event_flags'][key] == item['provenance']['one_shot_slide_artifacts_v3']['events'][key] for key in item['event_flags'])
        assert row['tissue_placed'] == item['tissue_placed']
        assert row['artifact_pixel_counts'] == item['artifact_pixel_counts']
        assert row['visible_fraction'] == item['visible_fraction']
        assert row['obliquity_deg'] == item['obliquity_deg']
        assert len(row['candidates']) == 16
        assert set(row['selectors']) == {'prior_first', 'no_fit_image', 'fitted_image', 'physically_best_diagnostic_only'}
        for index in range(17):
            c = row['candidates'][index] if index < 16 else row['true_control']
            label = str(index) if index < 16 else 'true_control'
            assert (c['candidate_index'], c['candidate_label']) == (index, label)
            key = (sid, index)
            rs = grouped_readouts[key]
            best = min(rs, key=lambda r: r['image_objective'])
            assert c['best_step'] == best['step'] == selected[(sid, label)]['best_step']
            same(c['selected_readout'], best, f'{sid}/{label} selected readout')
            same(c['zero_image_objective'], rs[0]['image_objective'], f'{sid}/{label} zero')
            same(c['selected_image_objective'], best['image_objective'], f'{sid}/{label} selected')
            same(c['fitted_site_error_mm'], best['fitted_site_error_mm'], f'{sid}/{label} fitted')
            same(c['rigid_site_error_mm'], best['rigid_site_error_mm'], f'{sid}/{label} rigid')
            fields = selected[(sid, label)]
            assert tensor_sha(fields['logits']) == c['selected_logits_sha256']
            assert tensor_sha(fields['requested_controls_um']) == c['selected_controls_sha256']
            assert tensor_sha(fields['constrained_field_local_um']) == c['selected_field_sha256']
            if index < 16:
                assert (c['rank'], c['branch_id'], c['reflection']) == (index, int(item['branch_ids'][index]), int(item['candidate_reflections'][index]))
                assert c['state'] == item['candidate_states'][index].tolist()
                same(c['prior_log_probability'], float(item['candidate_prior_log'][index]), f'{sid}/{index} prior')
                same(c['rigid_support_fraction_on_common_mask'], float(item['candidate_support_fraction'][index]), f'{sid}/{index} support')
                assert c['atlas_render_sha256'] == item['candidate_render_sha256'][index]
                assert c['rigid_surface_sha256'] == item['candidate_rigid_sha256'][index]
            else:
                assert c['state'] == item['true_state'][0].tolist()
                assert c['reflection'] == int(item['true_reflection'][0])
        candidates = row['candidates']
        indices = {'prior_first': 0,
            'no_fit_image': min(range(16), key=lambda j: candidates[j]['zero_image_objective']),
            'fitted_image': min(range(16), key=lambda j: candidates[j]['selected_image_objective']),
            'physically_best_diagnostic_only': min(range(16), key=lambda j: candidates[j]['rigid_site_error_mm'])}
        for name, index in indices.items():
            c, selector = candidates[index], row['selectors'][name]
            same(selector, {'candidate_index': index,
                'rigid_site_error_mm': c['rigid_site_error_mm'],
                'fitted_site_error_mm': c['fitted_site_error_mm'],
                'support_fraction': c['rigid_support_fraction_on_common_mask'],
                'zero_image_objective': c['zero_image_objective'],
                'selected_image_objective': c['selected_image_objective']}, f'{sid} {name}')
        near = indices['physically_best_diagnostic_only']
        far = next((j for j, c in enumerate(candidates) if c['rigid_site_error_mm'] >= 2 and
                    abs(c['rigid_support_fraction_on_common_mask']-candidates[near]['rigid_support_fraction_on_common_mask']) <= .10), None)
        if candidates[near]['rigid_site_error_mm'] > 1.5:
            far = None
        pair = {'eligible': far is not None, 'near_rank': near if far is not None else None,
            'far_rank': far, 'near_score': candidates[near]['selected_image_objective'] if far is not None else None,
            'far_score': candidates[far]['selected_image_objective'] if far is not None else None,
            'near_preferred': candidates[near]['selected_image_objective'] < candidates[far]['selected_image_objective'] if far is not None else None,
            'support_fraction_gap': abs(candidates[near]['rigid_support_fraction_on_common_mask']-candidates[far]['rigid_support_fraction_on_common_mask']) if far is not None else None}
        same(row['near_far_pair'], pair, f'{sid} near/far')

    summary, gate = summaries_and_gate(rows)
    same(completed['summary'], summary, 'summary')
    assert completed['gate'] == gate and completed['gate_pass'] == gate['pass']

    # Recompute blind top-16 branches from the frozen checkpoint, never using truth.
    from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
    assert torch.cuda.is_available(), 'Exact render-hash replay requires CUDA, as in the frozen run'
    device = 'cuda'
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.inference_mode():
        for item in panel:
            prediction = model.predict(item['inputs'].to(device))
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            order = prior.argsort(dim=-1, descending=True)[0, :16]
            assert torch.equal(order.cpu(), item['branch_ids'])
            torch.testing.assert_close(prediction['state'][0, order//2].cpu(), item['candidate_states'], atol=2e-4, rtol=1e-6)
            torch.testing.assert_close(prior[0, order].cpu(), item['candidate_prior_log'], atol=2e-5, rtol=1e-6)
    del model, checkpoint
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Independent atlas rendering and image-objective replay for every selected fit.
    from training import arbitrary_plane_allen_atlas_binding_v6 as allen
    atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
    atlas = torch.from_numpy(atlas_array).to(device)
    del atlas_array, annotation
    with torch.inference_mode():
        for item, row in zip(panel, rows):
            sid = item['physical_section_id']
            candidate_states = item['candidate_states'].to(device)
            candidate_reflections = item['candidate_reflections'].to(device)
            true_state = item['true_state'].to(device)
            true_reflection = item['true_reflection'].to(device)
            candidate_offsets = item['offsets'].to(device).expand(16, -1)
            candidate_weights = item['weights'].to(device).expand(16, -1)
            candidate_render, candidate_rigid, candidate_frames, candidate_bases = render(
                atlas, candidate_states, candidate_reflections, candidate_offsets, candidate_weights)
            true_render, true_rigid, true_frame, true_basis = render(
                atlas, true_state, true_reflection, item['offsets'].to(device), item['weights'].to(device))
            rendered = torch.cat((candidate_render, true_render))
            rigid = torch.cat((candidate_rigid, true_rigid))
            frames = torch.cat((candidate_frames, true_frame))
            bases = torch.cat((candidate_bases, true_basis))
            reflections = torch.cat((candidate_reflections, true_reflection))
            offsets = torch.cat((candidate_offsets, item['offsets'].to(device)))
            mask_bool = item['common_mask'].to(device)
            support = rendered[:16, 1] > .9
            union = support.any(0)
            assert bool((mask_bool <= union).all())
            if not item['brush_available']:
                assert torch.equal(mask_bool, union)
            else:
                assert int(mask_bool.sum()) <= item['brush_pixel_count']
            mask = mask_bool.float()[None]
            image = item['observed_image'].to(device)
            indices = item['site_indices'].to(device)
            target = item['site_target_ccf_um'].to(device)
            true_normal = frames[16, :, 2]
            for index in range(17):
                c = row['candidates'][index] if index < 16 else row['true_control']
                label = str(index) if index < 16 else 'true_control'
                if index < 16:
                    assert tensor_sha(rendered[index]) == item['candidate_render_sha256'][index], (sid, index, 'render')
                    assert tensor_sha(rigid[index]) == item['candidate_rigid_sha256'][index], (sid, index, 'rigid')
                    same(c['rigid_support_fraction_on_common_mask'], float((support[index] & mask_bool).sum()/mask_bool.sum()), f'{sid}/{index} support replay')
                    same(c['rigid_support_pixel_count'], int(support[index].sum()), f'{sid}/{index} support pixels')
                    angle = float(torch.rad2deg(torch.acos((frames[index, :, 2]*true_normal).sum().abs().clamp(0, 1))))
                    same(c['normal_axis_error_deg'], angle, f'{sid}/{index} normal')
                else:
                    assert tensor_sha(rendered[index:index+1]) == c['atlas_render_sha256']
                    assert tensor_sha(rigid[index:index+1]) == c['rigid_surface_sha256']
                    same(c['rigid_support_fraction_on_common_mask'], float(((rendered[index, 1]>.9)&mask_bool).sum()/mask_bool.sum()), f'{sid} true support')
                fields = selected[(sid, label)]
                state_slice = slice(index, index+1)
                arguments = (image, rendered[state_slice], mask, reflections[state_slice], bases[state_slice],
                             offsets[state_slice], rigid[state_slice], frames[state_slice], indices, target)
                zero, _, _ = rescore(torch.zeros_like(fields['logits']).to(device), *arguments)
                fitted, controls, local = rescore(fields['logits'].to(device), *arguments)
                same(zero['image_objective'], c['zero_image_objective'], f'{sid}/{label} zero replay')
                for name, value in fitted.items():
                    same(value, c['selected_readout'][name], f'{sid}/{label} {name} replay')
                assert tensor_sha(controls) == c['selected_controls_sha256']
                assert tensor_sha(local) == c['selected_field_sha256']
    print(json.dumps({'audit': 'PASS', 'sections': len(rows), 'fits_rescored': len(selected),
        'gate_pass': gate['pass'], 'gate': gate}, allow_nan=False))


if __name__ == '__main__':
    main()
