"""Frozen 112-only DEV assay of atlas intensity as local pose evidence; no training.

The energy is 1 - absolute 9-pixel local NCC, averaged only on an identical
17-pixel-eroded interior mask for each compared set of candidate states.
Neither atlas support outside that fixed mask nor any learned score enters it.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interior(mask):
    return F.avg_pool2d(mask.float()[None, None], 17, 1, 8)[0, 0] > .999


def intensity_energy(source, target, mask):
    a = source - F.avg_pool2d(source, 9, 1, 4)
    b = target - F.avg_pool2d(target, 9, 1, 4)
    cross = F.avg_pool2d(a * b, 9, 1, 4)
    variance = F.avg_pool2d(a.square(), 9, 1, 4) * F.avg_pool2d(b.square(), 9, 1, 4)
    agreement = (cross / (variance + 1e-5).sqrt()).abs().clamp(0, 1)
    return (((1 - agreement) * mask[None, None]).sum((-2, -1)) / mask.sum()).flatten()


source = Path(__file__).resolve().parent
run = root / 'runs/v3_fitting_evidence_112_pilot'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
plans = root / 'data/v3_pose_capture_confirmation_plans_001'
out = root / 'runs/v3_interior_intensity_energy_118_dev'
side, map_side = 256, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
checkpoint_path = run / 'joint_step_00512.pt'
assert done['updates'] == config['updates'] == 512
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(checkpoint_path) == done['checkpoint_sha256']['512']
assert sha(config['parent_checkpoint']) == config['parent_checkpoint_sha256'] == done['parent_checkpoint_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(source / name) == digest, name
assert not any(config[key] or done[key] for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used'))

panel_done = json.loads((panel / 'completed.json').read_text())
assert panel_done['physical_sections'] == 256 and panel_done['eligible'] == 247
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(plans / 'completed.json') == panel_done['plan_completed_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plan_ids) == 8 and len(records) == len({row['section_id'] for row in records}) == 256
selected = [row for plan in plan_ids for row in sorted((row for row in records
    if row['synthetic_subject_plan_id'] == plan and row['eligible']),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == 64 and all(sha(panel / row['file']) == row['sha256'] for row in selected)
draws = [json.loads(line) for line in (run / 'draws.jsonl').open() if line.strip()]
used = [row for row in draws if row['used']]
assert len(used) == 1024 and all(row['base_lineage']['split'] == 'train' for row in used)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in records} & {row['base_lineage'][key] for row in used}
assert not {row['panel_physical_section_id'] for row in records} & {
    row['physical_section_id'] for row in used}

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 512 and checkpoint['config'] == config
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint

corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
scales = torch.tensor([.025, .025, .025, 180., 180., 120., .01, .01, .007], device='cuda')
wrong_updates = torch.zeros(18, 9, device='cuda')
for axis in range(3):
    for sign_index, sign in enumerate((-1., 1.)):
        for distance_index, distance in enumerate((1600., 2200.)):
            wrong_updates[axis * 4 + sign_index * 2 + distance_index, axis + 3] = sign * distance
        wrong_updates[12 + axis * 2 + sign_index, axis] = sign * .4
rows = []

for record in selected:
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
        offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
        weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
        valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
    assert int(valid.sum()) == record['valid_pixels']
    row = {key: record[key] for key in ('section_id', 'panel_physical_section_id',
        'animal_id', 'subject_id', 'specimen_id', 'experiment_id',
        'synthetic_animal_id', 'synthetic_subject_plan_id',
        'synthetic_subject_realization_id', 'plan_receipt_sha256',
        'section_index', 'appearance_mode')}
    row.update(panel_file=record['file'], panel_file_sha256=record['sha256'])
    seed = int(hashlib.sha256(record['section_id'].encode()).hexdigest()[:16], 16)
    u = torch.randn(1, 9, generator=torch.Generator().manual_seed(seed)).cuda() * scales
    row.update(perturbation_seed=seed, perturbation_update=u[0].tolist())
    with torch.no_grad():
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = prior.topk(14, -1).indices
        beam_state = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
        beam_reflection = beam % 2
        truth_five = rigid_points_090(truth, reflection, corners)
        beam_error = (rigid_points_090(beam_state, beam_reflection, corners)
            - truth_five[:, None]).norm(dim=-1).mean(-1) / 1000
        best_slot = beam_error.argmin(-1)
        best_error = float(beam_error[0, best_slot[0]])
        row.update(best14_branch=int(beam[0, best_slot[0]]), best14_five_point_mm=best_error,
            oracle_best14_near_eligible=best_error <= 1.5)

        pool = compose_full_frame_state(truth.expand(18, -1), wrong_updates)[None]
        pool_reflection = reflection[:, None].expand(-1, 18)
        rigid_error = (rigid_points_090(pool, pool_reflection, corners)
            - truth_five[:, None]).norm(dim=-1).mean(-1)[0] / 1000
        exact_support = render_atlas_planes_090(context['atlas'], truth[:, None],
            reflection[:, None], offsets, weights)[:, 0, 1] > .95
        pool_support = render_atlas_planes_090(context['atlas'], pool,
            pool_reflection, offsets, weights, candidate_chunk=2)[:, :, 1] > .95
        overlap = (pool_support & exact_support[:, None]).sum((-2, -1)).float()
        dice = 2 * overlap / (pool_support.sum((-2, -1))
            + exact_support.sum((-2, -1))[:, None]).clamp_min(1)
        area_gap = (pool_support.float().mean((-2, -1))
            - exact_support.float().mean((-2, -1))[:, None]).abs()
        eligible = (rigid_error[None] >= 1.5) & (dice >= .85) & (area_gap <= .08)
        row['rigid_support_matched_wrong_count'] = int(eligible.sum())
        if not bool(eligible.any()):
            row['status'] = 'no_spatial_support_matched_wrong'
            rows.append(row)
            continue
        slot = (dice - .01 * rigid_error[None]).masked_fill(~eligible, -1).argmax(-1)
        wrong = pool[:, slot[0]]
        row.update(wrong_pool_slot=int(slot[0]), wrong_five_point_mm=float(rigid_error[slot[0]]),
            rigid_support_dice=float(dice[0, slot[0]]),
            rigid_support_area_gap=float(area_gap[0, slot[0]]))

        states = torch.stack((truth, wrong,
            compose_full_frame_state(truth, .4 * u),
            compose_full_frame_state(truth, .5 * u),
            compose_full_frame_state(truth, .6 * u)), 1)
        reflections = reflection[:, None].expand(-1, 5)
        row['near_five_point_mm_t04_t05_t06'] = ((rigid_points_090(
            states[:, 2:], reflections[:, 2:], corners) - truth_five[:, None]
            ).norm(dim=-1).mean(-1)[0] / 1000).tolist()
        if best_error <= 1.5:
            states = torch.cat((states,
                beam_state.gather(1, best_slot[:, None, None].expand(-1, 1, 12))), 1)
            reflections = torch.cat((reflections,
                beam_reflection.gather(1, best_slot[:, None])), 1)
        chosen = {**prediction, 'state': states}
        mapped = model.map(chosen, offsets,
            torch.arange(states.shape[1], device='cuda')[None], reflections,
            (map_side, map_side), context['atlas'], weights,
            feature_side=map_side, source_shape=(side, side))
        rendered = render_finite_thickness_coordinate_grid(context['atlas'],
            mapped['coordinates'].flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
            weights.expand(states.shape[1], -1))
        support = rendered[:, 1:2].clamp(0, 1)
        target = rendered[:, :1] / support.clamp_min(1e-4)
        support_binary = support[:, 0] > .95
        mapped_dice = 2 * (support_binary[0] & support_binary[1]).sum().float() / (
            support_binary[0].sum() + support_binary[1].sum()).clamp_min(1)
        mapped_area_gap = (support_binary[0].float().mean()
            - support_binary[1].float().mean()).abs()
        row.update(mapped_support_dice=float(mapped_dice),
            mapped_support_area_gap=float(mapped_area_gap))
        if float(mapped_dice) < .85 or float(mapped_area_gap) > .08:
            row['status'] = 'mapped_support_unmatched'
            rows.append(row)
            continue

        donor = F.interpolate(valid[:, None].float(), (map_side, map_side),
            mode='area')[0, 0] > .999
        wrong_mask = interior(donor & support_binary[0] & support_binary[1])
        near_mask = interior(donor & support_binary[0] & support_binary[2]
            & support_binary[3] & support_binary[4])
        row.update(wrong_interior_pixels=int(wrong_mask.sum()),
            near_interior_pixels=int(near_mask.sum()), valid96_pixels=int(donor.sum()))
        if int(wrong_mask.sum()) < 256 or int(near_mask.sum()) < 256:
            row['status'] = 'insufficient_common_interior'
            rows.append(row)
            continue

        source_image = F.interpolate(image[:, :1], (map_side, map_side), mode='area')
        exact_wrong = intensity_energy(source_image, target[:2], wrong_mask)
        exact_near = intensity_energy(source_image, target[[0, 2, 3, 4]], near_mask)
        zero_wrong = intensity_energy(source_image, torch.zeros_like(target[:2]), wrong_mask)
        zero_near = intensity_energy(source_image, torch.zeros_like(target[[0, 2, 3, 4]]), near_mask)
        row.update(status='scored', intensity_exact_wrong_mask=float(exact_wrong[0]),
            intensity_wrong=float(exact_wrong[1]),
            wrong_minus_exact=float(exact_wrong[1] - exact_wrong[0]),
            intensity_exact_near_mask=float(exact_near[0]),
            intensity_toward_t04=float(exact_near[1]),
            intensity_near_t05=float(exact_near[2]),
            intensity_away_t06=float(exact_near[3]),
            near_minus_exact=float(exact_near[2] - exact_near[0]),
            toward_drop=float(exact_near[2] - exact_near[1]),
            away_rise=float(exact_near[3] - exact_near[2]),
            toward_one_sided_slope=float((exact_near[2] - exact_near[1]) / .1),
            away_one_sided_slope=float((exact_near[3] - exact_near[2]) / .1),
            zero_wrong_minus_exact=float(zero_wrong[1] - zero_wrong[0]),
            zero_toward_drop=float(zero_near[2] - zero_near[1]),
            zero_away_rise=float(zero_near[3] - zero_near[2]),
            zero_intensity_exact_wrong=zero_wrong.tolist(),
            zero_intensity_exact_toward_near_away=zero_near.tolist())
        if states.shape[1] == 6:
            predicted_mask = interior(donor & support_binary[5])
            row['predicted_near_interior_pixels'] = int(predicted_mask.sum())
            if int(predicted_mask.sum()) >= 256:
                predicted_energy = intensity_energy(source_image, target[[0, 5]], predicted_mask)
                row['predicted_near_minus_exact'] = float(predicted_energy[1] - predicted_energy[0])
                predicted_state = states[:, 5].detach()
                predicted_reflection = reflections[:, 5:6]
                with torch.enable_grad():
                    delta = torch.zeros(1, 9, device='cuda', requires_grad=True)
                    probe = {**prediction, 'state': compose_full_frame_state(
                        predicted_state, delta * scales)[:, None]}
                    probe_map = model.map(probe, offsets,
                        torch.zeros(1, 1, device='cuda', dtype=torch.long),
                        predicted_reflection, (map_side, map_side), context['atlas'],
                        weights, feature_side=map_side, source_shape=(side, side))
                    probe_render = render_finite_thickness_coordinate_grid(
                        context['atlas'], probe_map['coordinates'].flatten(0, 1),
                        (0., 0., 0.), (25., 25., 25.), weights)
                    probe_support = probe_render[:, 1:2].clamp_min(1e-4)
                    probe_target = probe_render[:, :1] / probe_support
                    probe_energy = intensity_energy(source_image, probe_target, predicted_mask)[0]
                    zero_energy = intensity_energy(source_image,
                        probe_target * 0, predicted_mask)[0]
                    gradient = torch.autograd.grad(probe_energy, delta, retain_graph=True)[0]
                    zero_gradient = torch.autograd.grad(zero_energy, delta)[0]
                gradient_norm = float(gradient.norm())
                row.update(predicted_near_gradient_norm=gradient_norm,
                    predicted_near_zero_intensity_gradient_norm=float(zero_gradient.norm()),
                    predicted_near_baseline_energy=float(probe_energy.detach()))
                dimensionless_step = -.25 * gradient.detach() / gradient.norm().clamp_min(1e-8)
                del probe, probe_map, probe_render, probe_support, probe_target
                del probe_energy, zero_energy
                del delta, gradient, zero_gradient
                if gradient_norm > 1e-8:
                    post_state = compose_full_frame_state(predicted_state,
                        dimensionless_step * scales)
                    post = {**prediction, 'state': post_state[:, None]}
                    post_map = model.map(post, offsets,
                        torch.zeros(1, 1, device='cuda', dtype=torch.long),
                        predicted_reflection, (map_side, map_side), context['atlas'],
                        weights, feature_side=map_side, source_shape=(side, side))
                    post_render = render_finite_thickness_coordinate_grid(
                        context['atlas'], post_map['coordinates'].flatten(0, 1),
                        (0., 0., 0.), (25., 25., 25.), weights)
                    post_support = post_render[:, 1:2].clamp(0, 1)
                    post_target = post_render[:, :1] / post_support.clamp_min(1e-4)
                    support_retained = float((post_support[0, 0][predicted_mask] > .95).float().mean())
                    row.update(predicted_near_step_update=dimensionless_step[0].tolist(),
                        predicted_near_step_support_retained=support_retained)
                    if support_retained >= .95:
                        before_energy = intensity_energy(source_image, target[5:6], predicted_mask)[0]
                        after_energy = intensity_energy(source_image, post_target, predicted_mask)[0]
                        target_ccf = F.interpolate(centre.permute(0, 3, 1, 2),
                            (map_side, map_side), mode='bilinear', align_corners=False
                            ).permute(0, 2, 3, 1)[0]
                        before_chart = (mapped['centre_surface_ccf_ap_dv_ml_um'][0, 5]
                            - target_ccf).norm(dim=-1)
                        after_chart = (post_map['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                            - target_ccf).norm(dim=-1)
                        before_mm = (before_chart * predicted_mask).sum() / predicted_mask.sum() / 1000
                        after_mm = (after_chart * predicted_mask).sum() / predicted_mask.sum() / 1000
                        after_rigid_mm = (rigid_points_090(post_state,
                            predicted_reflection[:, 0], corners) - truth_five).norm(dim=-1).mean() / 1000
                        row.update(predicted_near_step_energy_before=float(before_energy),
                            predicted_near_step_energy_after=float(after_energy),
                            predicted_near_step_energy_drop=float(before_energy - after_energy),
                            predicted_near_step_rigid_five_mm_before=best_error,
                            predicted_near_step_rigid_five_mm_after=float(after_rigid_mm),
                            predicted_near_step_rigid_five_mm_improvement=float(best_error - after_rigid_mm),
                            predicted_near_step_chart_mm_before=float(before_mm),
                            predicted_near_step_chart_mm_after=float(after_mm),
                            predicted_near_step_chart_mm_improvement=float(before_mm - after_mm))
    rows.append(row)

scored = [row for row in rows if row['status'] == 'scored']
stepped = [row for row in rows if 'predicted_near_step_chart_mm_improvement' in row]
metrics = ('wrong_minus_exact', 'near_minus_exact', 'toward_drop', 'away_rise',
    'toward_one_sided_slope', 'away_one_sided_slope',
    'zero_wrong_minus_exact', 'zero_toward_drop', 'zero_away_rise')
summary = {'sections': len(selected), 'scored': len(scored),
    'status_counts': {status: sum(row['status'] == status for row in rows)
        for status in sorted({row['status'] for row in rows})},
    'oracle_best14_near_count': sum(row['oracle_best14_near_eligible'] for row in rows),
    'predicted_near_scored_count': sum('predicted_near_minus_exact' in row for row in rows),
    'predicted_near_step_count': len(stepped),
    'predicted_near_step_support_retained_count': sum(
        row.get('predicted_near_step_support_retained', 0) >= .95 for row in rows),
    'predicted_near_step_mean_energy_drop': float(np.mean([
        row['predicted_near_step_energy_drop'] for row in stepped])) if stepped else None,
    'predicted_near_step_energy_improved_fraction': float(np.mean([
        row['predicted_near_step_energy_drop'] > 0 for row in stepped])) if stepped else None,
    'predicted_near_step_mean_chart_mm_improvement': float(np.mean([
        row['predicted_near_step_chart_mm_improvement'] for row in stepped])) if stepped else None,
    'predicted_near_step_chart_improved_fraction': float(np.mean([
        row['predicted_near_step_chart_mm_improvement'] > 0 for row in stepped])) if stepped else None,
    'predicted_near_step_mean_rigid_five_mm_improvement': float(np.mean([
        row['predicted_near_step_rigid_five_mm_improvement'] for row in stepped])) if stepped else None,
    'predicted_near_step_rigid_five_improved_fraction': float(np.mean([
        row['predicted_near_step_rigid_five_mm_improvement'] > 0 for row in stepped])) if stepped else None,
    'predicted_near_zero_intensity_gradient_max': max((
        row['predicted_near_zero_intensity_gradient_norm'] for row in rows
        if 'predicted_near_zero_intensity_gradient_norm' in row), default=None),
    'plan_equal': {key: float(np.mean([np.mean([row[key] for row in scored
        if row['synthetic_subject_plan_id'] == plan]) for plan in plan_ids
        if any(row['synthetic_subject_plan_id'] == plan for row in scored)]))
        for key in metrics} if scored else {},
    'fraction': {key: float(np.mean([row[key] > 0 for row in scored]))
        for key in metrics} if scored else {},
    'scope': 'one Allen atlas, identity-disjoint synthetic DEV; oracle valid pixels; no real-animal validation',
    'interpretation': 'Positive differences/slopes favour exact pose and a step toward truth. The probed top14 branch is oracle-selected by truth for this local assay, not deployable blind selection; no blind reranking or training was performed.'}
output_config = {'source_sha256': sha(__file__),
    'training_completion_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': sha(checkpoint_path),
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'plans_completion_sha256': sha(plans / 'completed.json'),
    'selected_section_ids_sha256': hashlib.sha256(json.dumps(
        [row['section_id'] for row in selected]).encode()).hexdigest(),
    'selection': '8 eligible sections per synthetic plan by SHA256(section_id)',
    'wrong': 'truth-relative 18-state translation/rotation pool; five-point >=1.5 mm, whole-field binary support Dice >=.85 and area gap <=.08 both rigid and mapped; skip unmatched',
    'near': 'section-ID-seeded update u; compare exact and t=.4,.5,.6, with fixed common interior; truth selects best of actual top14 for conditional local assay only if <=1.5 mm',
    'predicted_near_step': 'frozen 112 map, gradient of intensity-only energy at oracle-best-of-14 predicted branch on detached donor-valid and predicted-support interior; one -0.25 normalized dimensionless gradient step; truth does not choose step direction; assess rigid five-point and learned-map chart error on same fixed mask only when >=95% post-support retained',
    'mask': 'area-downsampled synthetic donor-valid plus compared mapped atlas support >.95; common 17x17 binary erosion; >=256 pixels',
    'energy': '1 - absolute local NCC; source/atlas 9x9 high pass, 9x9 correlation, epsilon 1e-5; no support penalty or learned scorer',
    'control': 'same mapped states and masks, but atlas target intensity set to zero only at energy readout',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(selected), 'scored': len(scored),
    'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'source_sha256': output_config['source_sha256'],
    'checkpoint_sha256': output_config['checkpoint_sha256'],
    'panel_completion_sha256': output_config['panel_completion_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
