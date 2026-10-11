"""Prespecified anchor-only DEV readout; no CAL, FINAL, or public benchmark."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

TRAIN = ROOT / 'runs/anchor_only_blind_feedback_247_train_001'
PANEL = ROOT / 'data/one_shot_fresh_synthetic_dev_panel_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'runs/anchor_only_blind_feedback_247_readout_001'
STEPS = (0, 1000, 3000, 6000)
SIDE, FIRST_SIDE, FIT_SIDE = 256, 64, 96
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (SIDE - 1) / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
                                               frame[..., :2] @ basis, chart - .5)


synthetic_records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
real_records = [json.loads(line) for line in (REAL / 'records.jsonl').open(encoding='utf8')]
real_dev = [row for row in real_records if row['training_split'] == 'development']
matched_synthetic = [row for identity in sorted({row['animal_id'] for row in synthetic_records})
                     for row in [r for r in synthetic_records if r['animal_id'] == identity and r['eligible']][:4]]
matched_real = [row for identity in sorted({row['animal_id'] for row in real_dev})
                for row in [r for r in real_dev if r['animal_id'] == identity][:2]]
full_synthetic = [row for row in synthetic_records if row['eligible'] and row['valid_pixels'] > 0]
assert len(synthetic_records) == 256 and len(matched_synthetic) == 32
assert len({row['animal_id'] for row in matched_synthetic}) == 8
assert len(matched_real) == 12 and len({row['animal_id'] for row in matched_real}) == 6
assert not {row['animal_id'] for row in real_dev} & {
    row['animal_id'] for row in real_records if row['training_split'] == 'train'}
assert all(row['eligible'] and row['valid_pixels'] > 0 for row in matched_synthetic)
panel_files = {row['file']: sha(PANEL / row['file']) for row in full_synthetic}
assert all(panel_files[row['file']] == row['sha256'] for row in full_synthetic)
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz', allow_pickle=False) as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    real_thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0]).cuda()
atlas_null = torch.stack((torch.roll(atlas[0], 80, 0), atlas[1]))
model = OneShotJointSliceModel(modes=0, normal_anchor_count=64,
                              shared_anchor_pose=True, atlas_conditioning=True,
                              vector_refinement=True, fitted_ranking=True).cuda().eval()
training_config = json.loads((TRAIN / 'config.json').read_text(encoding='utf8'))
assert training_config['source_sha256']['arbitrary_plane_one_shot_model.py'] == sha(
    Path(__file__).parent / 'arbitrary_plane_one_shot_model.py')
assert training_config['protocol_sha256'] == sha(
    Path('docs/publication/ANCHOR_ONLY_BLIND_FEEDBACK_247_PROTOCOL_20261012.md'))
corners = torch.tensor(((0., 0.), (SIDE - 1., 0.), (0., SIDE - 1.),
                        (SIDE - 1., SIDE - 1.), ((SIDE - 1.) / 2, (SIDE - 1.) / 2)),
                       device='cuda') / SIDE
full_chart = torch.stack(torch.meshgrid(torch.arange(SIDE, device='cuda'),
                                       torch.arange(SIDE, device='cuda'), indexing='ij'), -1)[..., [1, 0]].float() / SIDE


def fit(image, offsets, weights, prediction, mode_indices, reflections, atlas_image):
    selected = {**prediction, 'state': prediction['state'][:, mode_indices],
                'log_mass': prediction['log_mass'][:, mode_indices],
                'reflection_logit': prediction['reflection_logit'][:, mode_indices]}
    branch = torch.arange(8, device='cuda')[None]
    first = model.map(selected, offsets, branch, reflections, (FIRST_SIDE, FIRST_SIDE),
                      atlas_image, weights, return_refinement_feature=True,
                      feature_side=FIRST_SIDE, source_shape=image.shape[-2:])
    refined_state, _, _ = model.refine(first['refinement_feature'], selected['state'])
    refined = {**selected, 'state': refined_state}
    mapped = model.map(refined, offsets, branch, reflections, (FIT_SIDE, FIT_SIDE),
                       atlas_image, weights, feature_side=FIT_SIDE,
                       source_shape=image.shape[-2:])
    fitted = model.score_fitted_candidates(image, refined, mapped, atlas_image, weights)[0]
    return refined_state, fitted, mapped


def evaluate_synthetic(record, stage, step):
    with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        true_reflection = int(arrays['reflection'])
        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
    prediction = model.predict(image)
    modes = prediction['log_mass'][0].topk(4).indices
    mode_indices = modes.repeat_interleave(2)
    reflections = torch.tensor([[0, 1] * 4], device='cuda')
    prior = (prediction['log_mass'][0, mode_indices]
             + torch.where(reflections[0].bool(),
                           F.logsigmoid(prediction['reflection_logit'][0, mode_indices]),
                           F.logsigmoid(-prediction['reflection_logit'][0, mode_indices])))
    refined, fitted, mapped96 = fit(image, offsets, weights, prediction,
                                    mode_indices, reflections, atlas)
    null_refined, null_fitted, _ = fit(image, offsets, weights, prediction,
                                       mode_indices, reflections, atlas_null)
    indices = valid.flatten().nonzero().flatten()
    chart = full_chart.reshape(-1, 2)[indices]
    target = reference.reshape(-1, 3)[indices]
    initial_rigid = (points(prediction['state'][:, mode_indices], reflections, chart)
                     - target).norm(dim=-1).mean(-1)[0]
    rigid = (points(refined, reflections, chart) - target).norm(dim=-1).mean(-1)[0]
    null_rigid = (points(null_refined, reflections, chart) - target).norm(dim=-1).mean(-1)[0]
    selected = int(fitted.argmax())
    prior_selected = int(prior.argmax())
    null_selected = int(null_fitted.argmax())
    sample = ((torch.arange(FIT_SIDE, device='cuda') + .5) * SIDE / FIT_SIDE - .5).round().long()
    valid96 = valid[sample[:, None], sample[None, :]]
    target96 = reference[sample[:, None], sample[None, :]][valid96]
    mapped96_error = (mapped96['centre_surface_ccf_ap_dv_ml_um'][0, :, valid96]
                      - target96[None]).norm(dim=-1).mean(-1)
    selected_states = torch.cat((refined[:, selected:selected + 1], truth[:, None]), 1)
    selected_reflections = torch.tensor([[int(reflections[0, selected]), true_reflection]], device='cuda')
    mapped256 = model.map({**prediction, 'state': selected_states}, offsets,
                          torch.tensor([[0, 1]], device='cuda'), selected_reflections,
                          (SIDE, SIDE), atlas, weights, source_shape=image.shape[-2:])
    mapped256_error = (mapped256['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                       - target[None]).norm(dim=-1).mean(-1)
    zero_warp = (points(truth[:, None], selected_reflections[:, 1:], chart)
                 - target).norm(dim=-1).mean()
    true_axis = full_frame_state_to_components(truth)[1][0, :, 2]
    initial_axes = full_frame_state_to_components(prediction['state'][:, mode_indices])[1][0, :, :, 2]
    refined_axes = full_frame_state_to_components(refined)[1][0, :, :, 2]
    nearest_principal = torch.rad2deg(torch.acos(true_axis.abs().max().clamp_max(1)))
    initial_axis_error = torch.rad2deg(torch.acos(
        (initial_axes[prior_selected] * true_axis).sum().abs().clamp_max(1)))
    selected_axis_error = torch.rad2deg(torch.acos(
        (refined_axes[selected] * true_axis).sum().abs().clamp_max(1)))
    return {'stage': stage, 'set': 'synthetic', 'step': step,
            **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                            'section_id', 'panel_physical_section_id',
                                            'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
                                            'plan_receipt_sha256', 'subject_ouv_sha256',
                                            'geometry_seed_sequence', 'augmentation_seed')},
            'panel_file_sha256': record['sha256'],
            'appearance_mode': record['appearance_mode'],
            'artifacts': record['provenance']['one_shot']['events'],
            'valid_fraction': float(valid.float().mean()),
            'true_nearest_principal_axis_deg': float(nearest_principal),
            'prior_top1_normal_axis_error_deg': float(initial_axis_error),
            'fitted_selected_normal_axis_error_deg': float(selected_axis_error),
            'fitted_selected_normal_ap_dv_ml': refined_axes[selected].cpu().tolist(),
            'prior_top1_rigid_um': float(initial_rigid[prior_selected]),
            'prior_top8_best_rigid_um': float(initial_rigid.min()),
            'refined_top8_best_rigid_um': float(rigid.min()),
            'fitted_selected_rigid_um': float(rigid[selected]),
            'null_selected_rigid_um': float(null_rigid[null_selected]),
            'fitted_selected_mapped_fit96_um': float(mapped96_error[selected]),
            'best_of_eight_mapped_fit96_um': float(mapped96_error.min()),
            'fitted_selected_mapped_full256_um': float(mapped256_error[0]),
            'exact_pose_mapped_full256_um': float(mapped256_error[1]),
            'exact_pose_zero_warp_full256_um': float(zero_warp),
            'prior_top1_mode': int(mode_indices[prior_selected]),
            'fitted_selected_mode': int(mode_indices[selected]),
            'null_selected_mode': int(mode_indices[null_selected]),
            'prior_top1_reflection': int(reflections[0, prior_selected]),
            'fitted_selected_reflection': int(reflections[0, selected])}


def evaluate_real(record, stage, step):
    index = record['array_row_index']
    input_array = np.concatenate((real_images[index].astype(np.float32),
                                  np.zeros((4, 192, 192), dtype=np.float32)))[None]
    image = F.interpolate(torch.from_numpy(input_array).cuda(), (SIDE, SIDE),
                          mode='bilinear', align_corners=False)
    offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(real_thickness[index])
    weights = torch.ones_like(offsets)
    weights[:, [0, -1]] = .5
    weights /= weights.sum(-1, keepdim=True)
    affine = torch.as_tensor(real_affines[index], device='cuda', dtype=torch.float32)
    source_corners = (corners * SIDE + .5) * (192 / SIDE) - .5
    reference = (affine[:, 2] + source_corners[:, :1] * affine[:, 0]
                 + source_corners[:, 1:] * affine[:, 1])
    prediction = model.predict(image)
    modes = prediction['log_mass'][0].topk(4).indices
    mode_indices = modes.repeat_interleave(2)
    reflections = torch.tensor([[0, 1] * 4], device='cuda')
    prior = (prediction['log_mass'][0, mode_indices]
             + torch.where(reflections[0].bool(),
                           F.logsigmoid(prediction['reflection_logit'][0, mode_indices]),
                           F.logsigmoid(-prediction['reflection_logit'][0, mode_indices])))
    refined, fitted, _ = fit(image, offsets, weights, prediction, mode_indices, reflections, atlas)
    null_refined, null_fitted, _ = fit(image, offsets, weights, prediction,
                                       mode_indices, reflections, atlas_null)
    initial_five = (points(prediction['state'][:, mode_indices], reflections, corners)
                    - reference).norm(dim=-1).mean(-1)[0]
    refined_five = (points(refined, reflections, corners) - reference).norm(dim=-1).mean(-1)[0]
    null_five = (points(null_refined, reflections, corners) - reference).norm(dim=-1).mean(-1)[0]
    selected, null_selected = int(fitted.argmax()), int(null_fitted.argmax())
    return {'stage': stage, 'set': 'real_weak_allen', 'step': step,
            **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                            'section_id', 'section_number', 'array_row_index',
                                            'source_image_sha256', 'source_geometry_record_sha256')},
            'prior_five_um': float(initial_five[int(prior.argmax())]),
            'refined_five_um': float(refined_five[selected]),
            'best_of_eight_refined_five_um': float(refined_five.min()),
            'null_refined_five_um': float(null_five[null_selected]),
            'prior_top1_mode': int(mode_indices[int(prior.argmax())]),
            'fitted_selected_mode': int(mode_indices[selected]),
            'null_selected_mode': int(mode_indices[null_selected]),
            'fitted_selected_reflection': int(reflections[0, selected]),
            'label_role': record['label_role']}


def summarize(rows, stage, step, split):
    group = [row for row in rows if row['stage'] == stage and row['step'] == step and row['set'] == split]
    identities = sorted({row['animal_id'] for row in group})
    metrics = (('prior_top1_rigid_um', 'prior_top8_best_rigid_um', 'refined_top8_best_rigid_um',
                'fitted_selected_rigid_um', 'null_selected_rigid_um',
                'fitted_selected_mapped_fit96_um', 'best_of_eight_mapped_fit96_um',
                'fitted_selected_mapped_full256_um', 'exact_pose_mapped_full256_um',
                'exact_pose_zero_warp_full256_um', 'prior_top1_normal_axis_error_deg',
                'fitted_selected_normal_axis_error_deg') if split == 'synthetic' else
               ('prior_five_um', 'refined_five_um', 'best_of_eight_refined_five_um',
                'null_refined_five_um'))
    return {'stage': stage, 'step': step, 'set': split, 'rows': len(group),
            'identities': len(identities),
            'identity_equal_mean': {key: float(np.mean([
                np.mean([row[key] for row in group if row['animal_id'] == identity])
                for identity in identities])) for key in metrics},
            'case_median': {key: float(np.median([row[key] for row in group])) for key in metrics}}


OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for step in STEPS:
        checkpoint = torch.load(TRAIN / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in matched_synthetic:
            row = evaluate_synthetic(record, 'matched', step)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in matched_real:
            row = evaluate_real(record, 'matched', step)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'stage': 'matched', 'step': step, 'synthetic': len(matched_synthetic),
                          'real_weak': len(matched_real)}), flush=True)
    matched = [summarize(rows, 'matched', step, 'synthetic') for step in STEPS]
    chosen = min(STEPS, key=lambda step: next(item['identity_equal_mean']['fitted_selected_rigid_um']
                                             for item in matched if item['step'] == step))
    checkpoint = torch.load(TRAIN / f'joint_step_{chosen:05d}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    (OUT / 'chosen_checkpoint.json').write_text(json.dumps({
        'rule': 'lowest matched-32 synthetic identity-equal fitted-selected rigid error; ties earliest step',
        'step': chosen, 'checkpoint_sha256': sha(TRAIN / f'joint_step_{chosen:05d}.pt'),
        'matched_identity_equal_fitted_selected_rigid_um': {
            str(item['step']): item['identity_equal_mean']['fitted_selected_rigid_um'] for item in matched},
    }, indent=2), encoding='utf8')
    for record in full_synthetic:
        row = evaluate_synthetic(record, 'full', chosen)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    for record in real_dev:
        row = evaluate_real(record, 'full', chosen)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
    stream.flush()
    print(json.dumps({'stage': 'full', 'step': chosen, 'synthetic': len(full_synthetic),
                      'real_weak': len(real_dev)}), flush=True)

summary = [summarize(rows, 'matched', step, split) for step in STEPS
           for split in ('synthetic', 'real_weak_allen')]
summary += [summarize(rows, 'full', chosen, split)
            for split in ('synthetic', 'real_weak_allen')]
matched_s = next(item for item in summary if item['stage'] == 'matched'
                 and item['step'] == chosen and item['set'] == 'synthetic')['identity_equal_mean']
full_r = next(item for item in summary if item['stage'] == 'full'
              and item['step'] == chosen and item['set'] == 'real_weak_allen')['identity_equal_mean']
matched_rows = [row for row in rows if row['stage'] == 'matched'
                and row['step'] == chosen and row['set'] == 'synthetic']
full_rows = [row for row in rows if row['stage'] == 'full'
             and row['step'] == chosen and row['set'] == 'synthetic']
plans = sorted({row['animal_id'] for row in matched_rows})
plan_null_advantage = {identity: float(np.mean([
    row['null_selected_rigid_um'] - row['fitted_selected_rigid_um']
    for row in matched_rows if row['animal_id'] == identity])) for identity in plans}
angle_bins = ((0, 15), (15, 30), (30, 45), (45, 60))
mode_counts = {str(mode): sum(row['fitted_selected_mode'] == mode for row in full_rows)
               for mode in range(64)}
prior_mode_counts = {str(mode): sum(row['prior_top1_mode'] == mode for row in full_rows)
                     for mode in range(64)}
selected_normals = np.asarray([row['fitted_selected_normal_ap_dv_ml'] for row in full_rows])
projective_angles = np.degrees(np.arccos(np.clip(np.abs(selected_normals @ selected_normals.T), 0, 1)))
projective_medoid = int(projective_angles.mean(0).argmin())
projective_spread_75 = float(np.percentile(projective_angles[:, projective_medoid], 75))
selected_modes_used_at_least_five = sum(count >= 5 for count in mode_counts.values())
heavy = [row for row in full_rows if 45 <= row['true_nearest_principal_axis_deg'] < 55]
assert len(heavy) == 21
heavy_mean_rigid = float(np.mean([row['fitted_selected_rigid_um'] for row in heavy]))
heavy_median_axis = float(np.median([row['fitted_selected_normal_axis_error_deg'] for row in heavy]))
strata = {
    'appearance': {name: {'cases': sum(row['appearance_mode'] == name for row in full_rows),
                          'mean_fitted_selected_rigid_um': float(np.mean([
                              row['fitted_selected_rigid_um'] for row in full_rows
                              if row['appearance_mode'] == name]))}
                   for name in sorted({row['appearance_mode'] for row in full_rows})},
    'nearest_principal_axis_deg': {f'{lower}-{upper}': {
        'cases': len(group),
        'mean_fitted_selected_rigid_um': float(np.mean([row['fitted_selected_rigid_um'] for row in group]))
        if group else None,
        'mean_fitted_selected_normal_axis_error_deg': float(np.mean([
            row['fitted_selected_normal_axis_error_deg'] for row in group])) if group else None,
        'median_fitted_selected_normal_axis_error_deg': float(np.median([
            row['fitted_selected_normal_axis_error_deg'] for row in group])) if group else None}
        for lower, upper in angle_bins
        for group in [[row for row in full_rows
                       if lower <= row['true_nearest_principal_axis_deg'] < upper]]},
    'valid_fraction': {f'{lower:.1f}-{upper:.1f}': {
        'cases': len(group),
        'mean_fitted_selected_rigid_um': float(np.mean([row['fitted_selected_rigid_um'] for row in group]))
        if group else None}
        for lower, upper in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01))
        for group in [[row for row in full_rows if lower <= row['valid_fraction'] < upper]]},
}
decision = {
    'selected_checkpoint_step': chosen,
    'matched_synthetic_selected_rigid_improvement_vs_244_step6000_um':
        4037. - matched_s['fitted_selected_rigid_um'],
    'matched_synthetic_intact_minus_null_advantage_um':
        matched_s['null_selected_rigid_um'] - matched_s['fitted_selected_rigid_um'],
    'matched_synthetic_plan_null_advantage_um': plan_null_advantage,
    'positive_plan_null_advantages': sum(value > 0 for value in plan_null_advantage.values()),
    'full_real_weak_donor_equal_refined_five_um': full_r['refined_five_um'],
    'matched_exact_pose_map_improvement_fraction':
        1 - matched_s['exact_pose_mapped_full256_um']
        / matched_s['exact_pose_zero_warp_full256_um'],
    'full_synthetic_selected_modes_used_at_least_five_cases': selected_modes_used_at_least_five,
    'full_synthetic_selected_normal_projective_medoid_case_index': projective_medoid,
    'full_synthetic_selected_normal_projective_medoid_ap_dv_ml':
        selected_normals[projective_medoid].tolist(),
    'full_synthetic_selected_normal_75pct_medoid_angle_deg': projective_spread_75,
    'full_synthetic_heavy_45_55_cases': len(heavy),
    'full_synthetic_heavy_45_55_mean_selected_rigid_um': heavy_mean_rigid,
    'full_synthetic_heavy_45_55_median_selected_normal_axis_error_deg': heavy_median_axis,
    'gates': {
        'matched_synthetic_selected_rigid_0p4mm_better_than_244':
            matched_s['fitted_selected_rigid_um'] <= 3637.,
        'intact_beats_null_by_0p25mm':
            matched_s['null_selected_rigid_um'] - matched_s['fitted_selected_rigid_um'] >= 250.,
        'intact_beats_null_in_at_least_6_of_8_plans':
            sum(value > 0 for value in plan_null_advantage.values()) >= 6,
        'full_real_weak_donor_equal_at_most_3p5mm': full_r['refined_five_um'] <= 3500.,
        'exact_pose_map_at_least_10pct_better_than_zero_warp':
            matched_s['exact_pose_mapped_full256_um'] <= .9
            * matched_s['exact_pose_zero_warp_full256_um'],
        'at_least_four_selected_anchor_ids_used_five_times':
            selected_modes_used_at_least_five >= 4,
        'selected_physical_normal_75pct_projective_spread_gt25deg':
            projective_spread_75 > 25.,
        'heavy_45_55_mean_selected_rigid_at_most_3p5mm':
            heavy_mean_rigid <= 3500.,
        'heavy_45_55_median_selected_normal_error_at_most_20deg':
            heavy_median_axis <= 20.,
    },
}
decision['developmental_continuation_gate_passed'] = all(decision['gates'].values())
result = {'summary': summary, 'full_synthetic_strata': strata,
          'full_synthetic_fitted_selected_mode_counts': mode_counts,
          'full_synthetic_prior_top1_mode_counts': prior_mode_counts,
          'anchor_normals_ap_dv_ml': model.normal_anchor_frames[:, :, 2].cpu().tolist(),
          'decision': decision,
          'metric_note': 'Synthetic rigid and zero-warp physical errors use all valid 256-pixel sites. '
                         'Best-of-eight mapped errors use the 96-pixel scorer grid with nearest '
                         'source-target samples; selected and exact-pose maps are also measured '
                         'at the full 256-pixel grid. Null reruns map, pose update and fitted '
                         'ranking with rolled atlas intensity, fixed source encoder/prior beam.'}
(OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'steps': STEPS, 'chosen_step': chosen, 'rows': len(rows),
    'matched_synthetic_per_step': len(matched_synthetic),
    'matched_real_per_step': len(matched_real),
    'full_synthetic': len(full_synthetic), 'full_real_weak': len(real_dev),
    'checkpoint_sha256': {str(step): sha(TRAIN / f'joint_step_{step:05d}.pt') for step in STEPS},
    'training_config_sha256': sha(TRAIN / 'config.json'),
    'training_completed_sha256': sha(TRAIN / 'completed.json'),
    'protocol_sha256': sha(Path('docs/publication/ANCHOR_ONLY_BLIND_FEEDBACK_247_PROTOCOL_20261012.md')),
    'panel_sha256': {name: sha(PANEL / name) for name in ('records.jsonl', 'protocol.json', 'completed.json')},
    'eligible_panel_file_sha256': panel_files,
    'real_sha256': {name: sha(REAL / name) for name in ('records.jsonl', 'geometry.npz', 'images.npy')},
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'null_atlas': 'roll atlas intensity 80 voxels along AP; preserve support and fixed source prior beam',
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in (
        Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'rows_sha256': sha(OUT / 'rows.jsonl'),
    'chosen_checkpoint_sha256': sha(OUT / 'chosen_checkpoint.json'),
    'summary_sha256': sha(OUT / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'scope': 'independent synthetic DEV plans and donor-separated weak Allen DEV; not animal-ground-truth validation',
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'chosen_step': chosen, 'rows': len(rows)}), flush=True)
