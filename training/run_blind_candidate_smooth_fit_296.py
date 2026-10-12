"""TRAIN-inner test of image-fit difficulty for blind top-16 atlas-plane choice."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.run_smooth_image_fit_295 import CONTROL, SIDE, SITES, STEPS, UPDATES, LIMIT, \
    digest, tensor_digest, rigid_render, image_objective


SEED = 20261012296
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/BLIND_CANDIDATE_SMOOTH_FIT_296_PROTOCOL_20261012.md'
CHECKPOINT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
EXPECTED_CHECKPOINT_SHA256 = 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
RUN = ROOT / 'reports/blind_candidate_smooth_fit_296_001'
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


def prior_ids_and_hashes():
    ids, hashes = set(), {}
    for candidate, (directory, names) in PRIOR.items():
        for name in names:
            path = ROOT / directory / name
            hashes[f'{candidate}/{name}'] = digest(path)
            if name.endswith('.pt'):
                ids.update(item['physical_section_id'] for item in
                           torch.load(path, map_location='cpu', weights_only=False))
            else:
                ids.update(json.loads(line)['physical_section_id'] for line in
                           path.open(encoding='utf8'))
    return ids, hashes


def blind_branches(model, inputs):
    with torch.inference_mode():
        prediction = model.predict(inputs)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        order = prior.argsort(dim=-1, descending=True)[:, :16]
    return prediction['state'][0, order[0] // 2], order[0] % 2, \
        prior[0, order[0]], order[0]


def make_panel(context, model, prior_ids):
    atlas = context['atlas']
    rng = np.random.default_rng(SEED + 1)
    panel, attempted = [], 0
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
                    state, reflection = sample['state'], sample['reflection']
                    offsets, weights = sample['offsets'], sample['weights']
                    inputs = sample['inputs']
                    brush_available = bool(inputs[0, 2, 0, 0])
                    brush = sample['brush_mask'][0] if brush_available else None
                    candidate_states, candidate_reflections, prior_log, branch_ids = \
                        blind_branches(model, inputs)
                    with torch.no_grad():
                        candidate_render, candidate_rigid, _, _ = rigid_render(
                            atlas, candidate_states, candidate_reflections,
                            offsets.expand(16, -1), weights.expand(16, -1))
                    candidate_support = candidate_render[:, 1] > .9
                    support_union = candidate_support.any(0)
                    mask = support_union & brush if brush_available else support_union
                    assert int(mask.sum()) > 2
                    support_fractions = ((candidate_support & mask).sum((1, 2)).float()
                                         / mask.sum())
                    true_normal = full_frame_state_to_components(state)[1][0, :, 2]
                    artifacts = identity['one_shot_slide_artifacts_v3']
                    item = {'base_index': base, 'slot': slot,
                        'subject_index': subject_index, 'draw_seed': draw_seed,
                        'physical_section_id': section_id, 'provenance': identity,
                        'inputs': inputs.cpu(), 'observed_image': inputs[:, :1].cpu(),
                        'true_state': state.cpu(), 'true_reflection': reflection.cpu(),
                        'offsets': offsets.cpu(), 'weights': weights.cpu(),
                        'candidate_states': candidate_states.cpu(),
                        'candidate_reflections': candidate_reflections.cpu(),
                        'candidate_prior_log': prior_log.cpu(), 'branch_ids': branch_ids.cpu(),
                        'common_mask': mask.cpu(), 'site_indices': indices.cpu(),
                        'site_target_ccf_um': target.cpu(),
                        'candidate_render_sha256': [tensor_digest(x) for x in candidate_render],
                        'candidate_rigid_sha256': [tensor_digest(x) for x in candidate_rigid],
                        'candidate_support_fraction': support_fractions.cpu(),
                        'candidate_support_pixel_count': candidate_support.sum((1, 2)).cpu(),
                        'brush_available': brush_available,
                        'brush_pixel_count': int(brush.sum()) if brush_available else None,
                        'mask_pixel_count': int(mask.sum()),
                        'valid_fraction': valid_count / (SIDE * SIDE),
                        'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                        'obliquity_deg': float(torch.rad2deg(torch.acos(
                            true_normal.abs().max().clamp(0, 1)))),
                        'event_flags': {name: bool(artifacts['events'][name]) for name in
                            ('fragment', 'fold', 'bubble', 'tile_seam')},
                        'tissue_placed': artifacts['tissue_placed'],
                        'artifact_pixel_counts': artifacts['pixel_counts'],
                        'pre_artifact_state_sha256': tensor_digest(sample['pre_artifact_state']),
                        'true_postartifact_state_sha256': tensor_digest(state),
                        'inputs_sha256': tensor_digest(inputs),
                        'observed_image_sha256': tensor_digest(inputs[:, :1]),
                        'full_coordinate_target_sha256': tensor_digest(sample['centre']),
                        'site_target_sha256': tensor_digest(target),
                        'candidate_states_sha256': tensor_digest(candidate_states),
                        'common_mask_sha256': tensor_digest(mask),
                        'brush_sha256': tensor_digest(brush) if brush_available else None}
                    panel.append(item)
                    accepted = True
                    del old, sample, candidate_render, candidate_rigid
                    break
                assert accepted
    assert len(panel) == 32 and len({x['physical_section_id'] for x in panel}) == 32
    torch.save(panel, RUN / 'inner_panel.pt')
    return panel, attempted


def fit_plane(item, index, state, reflection, rendered, rigid, mask, base_grid,
              readouts, training, selected_fields):
    offsets = item['offsets'].cuda()
    _, frame, basis = full_frame_state_to_components(state)
    image = item['observed_image'].cuda()
    indices = item['site_indices'].cuda()
    target = item['site_target_ccf_um'].cuda()
    rigid_points = rigid.reshape(-1, 3)[indices]
    rigid_error = float((rigid_points - target).norm(dim=-1).mean() / 1000)
    label = 'true_control' if index == 16 else str(index)
    logits = torch.nn.Parameter(torch.zeros((1, 2, CONTROL, CONTROL), device='cuda'))
    optimizer = torch.optim.Adam([logits], lr=.1)
    best, zero = None, None
    for step in range(UPDATES + 1):
        if step in STEPS:
            with torch.no_grad():
                loss, d = image_objective(logits, image, rendered,
                    mask, reflection, basis, offsets, base_grid)
                local = d['local']
                mapped = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
                mapped_points = mapped.reshape(-1, 3)[indices]
                site_error = float((mapped_points - target).norm(dim=-1).mean() / 1000)
                canonical = torch.where(reflection[:, None, None, None].bool(),
                                        local.flip(-1), local)
                after = ribbon_derivative_frobenius_bound(
                    canonical, torch.zeros_like(canonical), basis, offsets)
                field_norm = local[:, :2].norm(dim=1) / 1000
                result = {'physical_section_id': item['physical_section_id'],
                    'base_index': item['base_index'], 'slot': item['slot'],
                    'candidate_index': index, 'candidate_label': label,
                    'step': step, 'mask_pixel_count': item['mask_pixel_count'],
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
                    'site_shift_mean_mm': float((mapped_points-rigid_points).norm(dim=-1).mean()/1000),
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
                selected_fields[(item['physical_section_id'], label)] = {
                    'best_step': step, 'logits': logits.detach().cpu().clone(),
                    'requested_controls_um': d['controls'].detach().cpu().clone(),
                    'constrained_field_local_um': d['local'].detach().cpu().clone()}
        if step == UPDATES:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, _ = image_objective(logits, image, rendered,
            mask, reflection, basis, offsets, base_grid)
        loss.backward()
        gradient = float(logits.grad.norm())
        optimizer.step()
        training.write(json.dumps({'physical_section_id': item['physical_section_id'],
            'candidate_index': index, 'step': step + 1,
            'image_objective_before_update': float(loss.detach()),
            'logit_gradient_norm': gradient,
            'post_update_logit_absmax': float(logits.detach().abs().amax())},
            allow_nan=False) + '\n')
    chosen = selected_fields[(item['physical_section_id'], label)]
    return {'candidate_index': index, 'candidate_label': label,
        'rigid_site_error_mm': rigid_error,
        'zero_image_objective': zero['image_objective'],
        'selected_image_objective': best['image_objective'],
        'best_step': best['step'], 'fitted_site_error_mm': best['fitted_site_error_mm'],
        'selected_readout': best,
        'selected_logits_sha256': tensor_digest(chosen['logits']),
        'selected_controls_sha256': tensor_digest(chosen['requested_controls_um']),
        'selected_field_sha256': tensor_digest(chosen['constrained_field_local_um'])}


def summary(rows):
    if not rows:
        return {'n': 0}
    first = np.array([x['selectors']['prior_first']['rigid_site_error_mm'] for x in rows])
    no_fit = np.array([x['selectors']['no_fit_image']['rigid_site_error_mm'] for x in rows])
    fitted = np.array([x['selectors']['fitted_image']['rigid_site_error_mm'] for x in rows])
    mapped_first = np.array([x['selectors']['prior_first']['fitted_site_error_mm'] for x in rows])
    mapped_fitted = np.array([x['selectors']['fitted_image']['fitted_site_error_mm'] for x in rows])
    return {'n': len(rows), 'prior_first_rigid_mean_mm': float(first.mean()),
        'no_fit_image_rigid_mean_mm': float(no_fit.mean()),
        'fitted_image_rigid_mean_mm': float(fitted.mean()),
        'fitted_image_rigid_gain_over_prior_mm': float((first-fitted).mean()),
        'fitted_image_rigid_gain_over_no_fit_mm': float((no_fit-fitted).mean()),
        'prior_first_fitted_map_mean_mm': float(mapped_first.mean()),
        'fitted_image_fitted_map_mean_mm': float(mapped_fitted.mean()),
        'fitted_image_map_change_from_prior_mm': float((mapped_fitted-mapped_first).mean()),
        'positive_rigid_gain_cases': int((first>fitted).sum()),
        'prior_first_capture_1p5mm': int((first<=1.5).sum()),
        'fitted_image_capture_1p5mm': int((fitted<=1.5).sum()),
        'mean_prior_first_support_fraction': float(np.mean([
            x['selectors']['prior_first']['support_fraction'] for x in rows])),
        'mean_fitted_image_support_fraction': float(np.mean([
            x['selectors']['fitted_image']['support_fraction'] for x in rows])),
        'mean_prior_first_normal_axis_error_deg': float(np.mean([
            x['candidates'][0]['normal_axis_error_deg'] for x in rows])),
        'mean_fitted_image_normal_axis_error_deg': float(np.mean([
            x['candidates'][x['selectors']['fitted_image']['candidate_index']]
                ['normal_axis_error_deg'] for x in rows])),
        'mean_prior_first_selected_image_objective': float(np.mean([
            x['selectors']['prior_first']['selected_image_objective'] for x in rows])),
        'mean_fitted_image_selected_image_objective': float(np.mean([
            x['selectors']['fitted_image']['selected_image_objective'] for x in rows])),
        'mean_fitted_image_field_rms_mm': float(np.mean([
            x['candidates'][x['selectors']['fitted_image']['candidate_index']]
                ['selected_readout']['field_displacement_rms_mm'] for x in rows])),
        'mean_fitted_image_warped_support_exit_fraction': float(np.mean([
            x['candidates'][x['selectors']['fitted_image']['candidate_index']]
                ['selected_readout']['warped_support_exit_fraction'] for x in rows]))}


def main():
    assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    checkpoint_sha256 = digest(CHECKPOINT)
    assert checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    prior_ids, prior_hashes = prior_ids_and_hashes()
    source_names = (
        'run_blind_candidate_smooth_fit_296.py', 'run_smooth_image_fit_295.py',
        'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
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
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
    source_hashes = {name: digest(SOURCE / name) for name in source_names}
    protocol_sha256 = digest(PROTOCOL)
    context = load_streaming_synthetic_v7_64(device='cuda')
    synthetic_provenance = context['provenance']
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda()
    checkpoint = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    RUN.mkdir(parents=True, exist_ok=False)
    panel, attempted = make_panel(context, model, prior_ids)
    del model, checkpoint
    torch.cuda.empty_cache()
    config = {'candidate': 296, 'protocol_sha256': protocol_sha256,
        'source_sha256': source_hashes, 'pose_checkpoint_sha256': checkpoint_sha256,
        'prior_file_sha256': prior_hashes,
        'prior_physical_section_count': len(prior_ids) - len(panel),
        'synthetic_provenance': synthetic_provenance, 'seed': SEED,
        'attempted_draws': attempted, 'accepted_sections': len(panel),
        'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
        'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
        'train_base_indices': list(range(56, 64)), 'sections_per_base': 4,
        'minimum_distinct_valid_sites_for_inclusion': SITES,
        'source_shape': [SIDE, SIDE], 'control_shape': [CONTROL, CONTROL],
        'site_count_per_section': SITES, 'readout_steps': STEPS,
        'candidate_count': 16, 'unranked_true_control': True,
        'mask_rule': 'candidate-independent top-16 rigid-support union >0.9 intersect optional brush; no synthetic validity or true control',
        'render': 'sample finite-thickness offsets/weights; support-normalized intensity; true postartifact gauge v1',
        'control_objective_optimizer': 'unchanged training.run_smooth_image_fit_295 image_objective; independent zero logits per branch; Adam lr 0.1 120 updates',
        'best_step_rule': 'minimum image objective among readouts 0/40/80/120, without coordinate truth',
        'selection': 'lowest selected image objective among 16 blind candidates; no prior-score fusion',
        'pair_rule': 'nearest rigid-error near <=1.5mm versus highest-prior far >=2.0mm with common-mask support-fraction gap <=0.10',
        'coordinate_evaluation': '512 prespecified distinct valid source-grid CCF target sites, readout only',
        'map_derivative_limit': LIMIT, 'no_training_or_new_model': True,
        'no_dev_cal_final_or_public_benchmark': True,
        'torch': torch.__version__, 'numpy': np.__version__}
    (RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')
    axis = (torch.arange(SIDE, device='cuda') + .5) * (2 / SIDE) - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    base_grid = torch.stack((xx, yy), -1)[None]
    rows, selected_fields = [], {}
    with (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
         (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readouts, \
         (RUN / 'sections.jsonl').open('w', encoding='utf8') as output:
        for item in panel:
            states = item['candidate_states'].cuda()
            reflections = item['candidate_reflections'].cuda()
            offsets = item['offsets'].cuda()
            weights = item['weights'].cuda()
            mask = item['common_mask'].cuda().float()[None]
            with torch.no_grad():
                candidate_render, candidate_rigid, candidate_frames, _ = rigid_render(
                    context['atlas'], states, reflections,
                    offsets.expand(16, -1), weights.expand(16, -1))
            assert [tensor_digest(x) for x in candidate_render] == item['candidate_render_sha256']
            assert [tensor_digest(x) for x in candidate_rigid] == item['candidate_rigid_sha256']
            candidate_rows = []
            true_normal = full_frame_state_to_components(item['true_state'].cuda())[1][0, :, 2]
            for index in range(16):
                fitted = fit_plane(item, index, states[index:index+1],
                    reflections[index:index+1], candidate_render[index:index+1],
                    candidate_rigid[index:index+1], mask, base_grid,
                    readouts, training, selected_fields)
                normal = candidate_frames[index, :, 2]
                fitted.update({'rank': index, 'branch_id': int(item['branch_ids'][index]),
                    'state': states[index].tolist(), 'reflection': int(reflections[index]),
                    'prior_log_probability': float(item['candidate_prior_log'][index]),
                    'rigid_support_fraction_on_common_mask':
                        float(item['candidate_support_fraction'][index]),
                    'rigid_support_pixel_count': int(item['candidate_support_pixel_count'][index]),
                    'normal_axis_error_deg': float(torch.rad2deg(torch.acos(
                        (normal * true_normal).sum().abs().clamp(0, 1)))),
                    'atlas_render_sha256': item['candidate_render_sha256'][index],
                    'rigid_surface_sha256': item['candidate_rigid_sha256'][index]})
                candidate_rows.append(fitted)
            with torch.no_grad():
                true_render, true_rigid, _, _ = rigid_render(context['atlas'],
                    item['true_state'].cuda(), item['true_reflection'].cuda(), offsets, weights)
            true_control = fit_plane(item, 16, item['true_state'].cuda(),
                item['true_reflection'].cuda(), true_render, true_rigid, mask,
                base_grid, readouts, training, selected_fields)
            true_control.update({'state': item['true_state'][0].tolist(),
                'reflection': int(item['true_reflection'][0]),
                'rigid_support_fraction_on_common_mask': float(((true_render[0, 1]>.9)
                    & item['common_mask'].cuda()).sum() / mask.sum()),
                'atlas_render_sha256': tensor_digest(true_render),
                'rigid_surface_sha256': tensor_digest(true_rigid)})
            no_fit_index = min(range(16), key=lambda j: candidate_rows[j]['zero_image_objective'])
            fit_index = min(range(16), key=lambda j: candidate_rows[j]['selected_image_objective'])
            best_physical_index = min(range(16), key=lambda j: candidate_rows[j]['rigid_site_error_mm'])
            selections = {'prior_first': 0, 'no_fit_image': no_fit_index,
                'fitted_image': fit_index, 'physically_best_diagnostic_only': best_physical_index}
            selectors = {name: {'candidate_index': j,
                'rigid_site_error_mm': candidate_rows[j]['rigid_site_error_mm'],
                'fitted_site_error_mm': candidate_rows[j]['fitted_site_error_mm'],
                'support_fraction': candidate_rows[j]['rigid_support_fraction_on_common_mask'],
                'zero_image_objective': candidate_rows[j]['zero_image_objective'],
                'selected_image_objective': candidate_rows[j]['selected_image_objective']}
                for name, j in selections.items()}
            near = best_physical_index
            eligible_far = [j for j in range(16) if
                candidate_rows[j]['rigid_site_error_mm'] >= 2.0 and
                abs(candidate_rows[j]['rigid_support_fraction_on_common_mask'] -
                    candidate_rows[near]['rigid_support_fraction_on_common_mask']) <= .10]
            far = eligible_far[0] if candidate_rows[near]['rigid_site_error_mm'] <= 1.5 \
                and eligible_far else None
            pair = {'eligible': far is not None, 'near_rank': near if far is not None else None,
                'far_rank': far, 'near_score': candidate_rows[near]['selected_image_objective']
                    if far is not None else None,
                'far_score': candidate_rows[far]['selected_image_objective']
                    if far is not None else None,
                'near_preferred': candidate_rows[near]['selected_image_objective'] <
                    candidate_rows[far]['selected_image_objective'] if far is not None else None,
                'support_fraction_gap': abs(candidate_rows[near]['rigid_support_fraction_on_common_mask'] -
                    candidate_rows[far]['rigid_support_fraction_on_common_mask'])
                    if far is not None else None}
            row = {'base_index': item['base_index'], 'slot': item['slot'],
                'subject_index': item['subject_index'], 'draw_seed': item['draw_seed'],
                'physical_section_id': item['physical_section_id'],
                'donor_animal_id': item['provenance']['base_lineage']['animal_id'],
                'base_lineage': item['provenance']['base_lineage'],
                'virtual_subject_id': item['provenance']['virtual_subject_id'],
                'virtual_index': item['provenance']['virtual_index'],
                'provenance': item['provenance'], 'site_count': SITES,
                'mask_pixel_count': item['mask_pixel_count'],
                'common_mask_sha256': item['common_mask_sha256'],
                'brush_available': item['brush_available'],
                'brush_pixel_count': item['brush_pixel_count'],
                'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'obliquity_deg': item['obliquity_deg'],
                'event_flags': item['event_flags'],
                'tissue_placed': item['tissue_placed'],
                'artifact_pixel_counts': item['artifact_pixel_counts'],
                'input_sha256': {name: item[name] for name in (
                    'pre_artifact_state_sha256', 'true_postartifact_state_sha256',
                    'inputs_sha256', 'observed_image_sha256',
                    'full_coordinate_target_sha256', 'site_target_sha256',
                    'candidate_states_sha256', 'common_mask_sha256', 'brush_sha256')},
                'candidates': candidate_rows, 'true_control': true_control,
                'selectors': selectors, 'near_far_pair': pair}
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False) + '\n')
            training.flush(); readouts.flush(); output.flush()
            print(json.dumps({'event': 'section_complete',
                'physical_section_id': item['physical_section_id'],
                'base_index': item['base_index'], 'slot': item['slot'],
                'prior_first_mm': selectors['prior_first']['rigid_site_error_mm'],
                'fitted_image_mm': selectors['fitted_image']['rigid_site_error_mm']}), flush=True)
    torch.save(selected_fields, RUN / 'selected_fields.pt')
    overall = summary(rows)
    by_base = {str(base): summary([x for x in rows if x['base_index'] == base])
               for base in range(56, 64)}
    matched = [x['near_far_pair'] for x in rows if x['near_far_pair']['eligible']]
    matched_count = len(matched)
    matched_win_rate = (sum(x['near_preferred'] for x in matched) / matched_count
                        if matched_count else None)
    summaries = {**overall, 'by_base': by_base,
        'positive_base_count': sum(x['fitted_image_rigid_gain_over_prior_mm'] > 0
                                   for x in by_base.values()),
        'matched_near_far_case_count': matched_count,
        'matched_near_far_win_count': sum(x['near_preferred'] for x in matched),
        'matched_near_far_win_rate': matched_win_rate,
        'by_obliquity_30deg': {str(flag).lower(): summary([x for x in rows if
            (x['obliquity_deg'] >= 30) == flag]) for flag in (False, True)},
        'by_sparse_valid_under_15pct': {str(flag).lower(): summary([x for x in rows if
            (x['valid_fraction'] < .15) == flag]) for flag in (False, True)},
        'by_brush': {str(flag).lower(): summary([x for x in rows if
            x['brush_available'] == flag]) for flag in (False, True)},
        'by_artifact': {event: {str(flag).lower(): summary([x for x in rows if
            x['event_flags'][event] == flag]) for flag in (False, True)} for event in
            ('fragment', 'fold', 'bubble', 'tile_seam')},
        'by_prior_first_support_under_half': {str(flag).lower(): summary([x for x in rows if
            (x['selectors']['prior_first']['support_fraction'] < .5) == flag])
            for flag in (False, True)},
        'best_step_distribution': {str(step): sum(x['candidates'][j]['best_step'] == step
            for x in rows for j in range(16)) for step in STEPS},
        'true_control_rigid_mean_mm': float(np.mean([
            x['true_control']['rigid_site_error_mm'] for x in rows])),
        'true_control_fitted_mean_mm': float(np.mean([
            x['true_control']['fitted_site_error_mm'] for x in rows])),
        'max_postlimit_derivative_bound': max(
            x['candidates'][j]['selected_readout']['postlimit_derivative_bound']
            for x in rows for j in range(16))}
    gate = {'rigid_gain_over_prior_at_least_0p25mm':
            overall['fitted_image_rigid_gain_over_prior_mm'] >= .25,
        'rigid_gain_over_no_fit_at_least_0p15mm':
            overall['fitted_image_rigid_gain_over_no_fit_mm'] >= .15,
        'at_least_six_positive_bases': summaries['positive_base_count'] >= 6,
        'fitted_map_no_more_than_0p05mm_worse_than_prior':
            overall['fitted_image_map_change_from_prior_mm'] <= .05,
        'at_least_16_matched_near_far_cases': matched_count >= 16,
        'matched_near_far_win_rate_at_least_0p70':
            matched_win_rate >= .70 if matched_win_rate is not None else False}
    gate['pass'] = all(gate.values())
    output_names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
        'training.jsonl', 'readouts.jsonl', 'sections.jsonl', 'selected_fields.pt')
    completed = {'candidate': 296, 'eligible_sections': len(rows),
        'attempted_draws': attempted, 'summary': summaries, 'gate': gate,
        'gate_pass': gate['pass'], 'protocol_sha256': protocol_sha256,
        'source_sha256': source_hashes, 'pose_checkpoint_sha256': checkpoint_sha256,
        'output_sha256': {name: digest(RUN / name) for name in output_names},
        'elapsed_seconds': time.perf_counter() - started,
        'result_role': 'bounded TRAIN-inner blind candidate image-fit diagnostic; no deployable predictor',
        'dev_cal_final_or_public_benchmark_used': False}
    (RUN / 'completed.json').write_text(json.dumps(completed, indent=2,
        allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'gate': gate,
        'elapsed_seconds': completed['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
