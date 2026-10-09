"""Frozen TRAIN-only fitting pilot on 16 identity-disjoint synthetic DEV sections."""
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


source = Path(__file__).resolve().parent
run = root / 'runs/v3_fitting_evidence_112_pilot'
parent = root / 'runs/v3_mixed_real_pose_coronal_risk_111/joint_step_01959.pt'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
plans = root / 'data/v3_pose_capture_confirmation_plans_001'
out = root / 'runs/v3_fitting_evidence_112_dev_eval'
steps, side, map_side = (0, 128, 512), 256, 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert done['updates'] == config['updates'] == 512
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(parent) == config['parent_checkpoint_sha256'] == done['parent_checkpoint_sha256']
assert sha(parent.parent / 'completed.json') == config['parent_completion_sha256']
assert sha(source.parent / 'docs/publication/V3_FITTING_EVIDENCE_112_PROTOCOL_20261009.md') == config['protocol_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(source / name) == digest, name
assert not any(config[key] or done[key] for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used'))
checkpoints = {step: run / f'joint_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == done['checkpoint_sha256'][str(step)]
    for step, path in checkpoints.items())

panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_done['physical_sections'] == 256 and panel_done['eligible'] == 247
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(plans / 'completed.json') == panel_done['plan_completed_sha256'] == panel_protocol['source_plan_completed_sha256']
all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in all_records})
assert len(plan_ids) == 8 and len(all_records) == len({row['section_id'] for row in all_records}) == 256
selected = [row for plan_id in plan_ids for row in sorted((row for row in all_records
    if row['synthetic_subject_plan_id'] == plan_id and row['eligible']),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:2]]
assert len(selected) == 16 and all(sha(panel / row['file']) == row['sha256']
    for row in selected)
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 1024 and all(row['base_lineage']['split'] == 'train' for row in used)
assert len({row['physical_section_id'] for row in used}) == len(used)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in all_records} & {row['base_lineage'][key] for row in used}
assert not {row['panel_physical_section_id'] for row in all_records} & {
    row['physical_section_id'] for row in used}

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
parent_state = torch.load(parent, map_location='cpu', weights_only=True)['model']
initial_state = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
assert initial_state['step'] == 0 and initial_state['config'] == config
assert all(torch.equal(parent_state[key], value) for key, value in initial_state['model'].items())
del parent_state, initial_state

corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
axis = (torch.arange(map_side, device='cuda') + .5) / map_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
rows = []

for step in steps:
    checkpoint = torch.load(checkpoints[step], map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == config
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    for index, record in enumerate(selected):
        swapped_record = selected[(index + 2) % len(selected)]
        assert swapped_record['synthetic_subject_plan_id'] != record['synthetic_subject_plan_id']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
        with np.load(panel / swapped_record['file'], allow_pickle=False) as arrays:
            swapped_image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        with torch.no_grad():
            prediction = model.predict(image)
            swapped_prediction = model.predict(swapped_image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = prior.topk(16, -1).indices
            beam_states = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            beam_reflections = beam % 2
            truth_five = rigid_points_090(truth_state, truth_reflection, corners)
            error_mm = (rigid_points_090(beam_states, beam_reflections, corners)
                - truth_five[:, None]).norm(dim=-1).mean(-1) / 1000
            beam_pairs = render_atlas_planes_090(context['atlas'], beam_states,
                beam_reflections, offsets, weights, candidate_chunk=2)
            truth_pair = render_atlas_planes_090(context['atlas'], truth_state[:, None],
                truth_reflection[:, None], offsets, weights)
            support_gap = (beam_pairs[:, :, 1].mean((-2, -1))
                - truth_pair[:, :, 1].mean((-2, -1))).abs()
            wrong = error_mm >= 1.5
            matched = wrong & (support_gap <= .10)
            eligible = matched if bool(matched.any()) else wrong
            hard_slot = (prior.gather(1, beam) - 5 * support_gap).masked_fill(
                ~eligible, -1e6).argmax(-1)
            if not bool(eligible.any()):
                hard_slot = error_mm.argmax(-1)
            hard_state = beam_states[torch.arange(1, device='cuda'), hard_slot]
            hard_reflection = beam_reflections[torch.arange(1, device='cuda'), hard_slot]
            perturbation_seed = int(hashlib.sha256(record['section_id'].encode()).hexdigest()[:16], 16)
            generator = torch.Generator().manual_seed(perturbation_seed)
            u = (torch.randn(1, 9, generator=generator).cuda() * torch.tensor(
                [.025, .025, .025, 180., 180., 120., .01, .01, .007], device='cuda'))
            near_state = compose_full_frame_state(truth_state, .5 * u)
            far_state = compose_full_frame_state(truth_state, 1.5 * u)
            states = torch.stack((truth_state, hard_state, near_state, far_state), 1)
            reflections = torch.stack((truth_reflection, hard_reflection,
                truth_reflection, truth_reflection), 1)
            selected_prediction = {**prediction, 'state': states,
                'log_mass': torch.zeros(1, 4, device='cuda'),
                'reflection_logit': torch.zeros(1, 4, device='cuda')}
            mapped = model.map(selected_prediction, offsets,
                torch.arange(4, device='cuda')[None], reflections,
                (map_side, map_side), context['atlas'], weights,
                feature_side=map_side, source_shape=(side, side))
            scores = model.score_fitted_candidates(image, selected_prediction,
                mapped, context['atlas'], weights)[0]
            swapped_selected = {**swapped_prediction, 'state': truth_state[:, None],
                'log_mass': torch.zeros(1, 1, device='cuda'),
                'reflection_logit': torch.zeros(1, 1, device='cuda')}
            swapped_map = model.map(swapped_selected, offsets,
                torch.zeros(1, 1, device='cuda', dtype=torch.long),
                truth_reflection[:, None], (map_side, map_side), context['atlas'], weights,
                feature_side=map_side, source_shape=(side, side))
            swapped_score = model.score_fitted_candidates(swapped_image,
                swapped_selected, swapped_map, context['atlas'], weights)[0, 0]
            mask = F.interpolate(valid[:, None].float(), (map_side, map_side),
                mode='bilinear', align_corners=False)[:, 0] > .999
            target = F.interpolate(centre.permute(0, 3, 1, 2), (map_side, map_side),
                mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            mapped_error = ((mapped['centre_surface_ccf_ap_dv_ml_um'][:, :2]
                - target[:, None]).norm(dim=-1) * mask[:, None]).sum((-2, -1)) \
                / mask.sum().clamp_min(1) / 1000
            rigid = rigid_points_090(truth_state, truth_reflection, chart).reshape(
                1, map_side, map_side, 3)
            rigid_error = (((rigid - target).norm(dim=-1) * mask).sum()
                / mask.sum().clamp_min(1) / 1000)
            support = render_finite_thickness_coordinate_grid(context['atlas'],
                mapped['coordinates'].flatten(0, 1), (0., 0., 0.),
                (25., 25., 25.), weights.expand(4, -1))[:, 1].reshape(1, 4, map_side, map_side)
            oracle_mask = mask[:, None].float()
            silhouette_dice = (2 * (support * oracle_mask).sum((-2, -1)) /
                (support.sum((-2, -1)) + oracle_mask.sum((-2, -1))).clamp_min(1e-6))[0]

        t = torch.tensor(.5, device='cuda', requires_grad=True)
        probe_state = compose_full_frame_state(truth_state, t * u)
        probe = {**prediction, 'state': probe_state[:, None],
            'log_mass': torch.zeros(1, 1, device='cuda'),
            'reflection_logit': torch.zeros(1, 1, device='cuda')}
        probe_map = model.map(probe, offsets,
            torch.zeros(1, 1, device='cuda', dtype=torch.long),
            truth_reflection[:, None], (map_side, map_side), context['atlas'], weights,
            feature_side=map_side, source_shape=(side, side))
        probe_score = model.score_fitted_candidates(image, probe, probe_map,
            context['atlas'], weights)[0, 0]
        derivative = torch.autograd.grad(probe_score, t)[0]
        rows.append({'step': step, 'section_id': record['section_id'],
            'panel_physical_section_id': record['panel_physical_section_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'synthetic_animal_id': record['synthetic_animal_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'swapped_section_id': swapped_record['section_id'],
            'swapped_subject_plan_id': swapped_record['synthetic_subject_plan_id'],
            'perturbation_seed': perturbation_seed, 'perturbation_update': u[0].tolist(),
            'valid96_pixels': int(mask.sum()),
            'hard_branch': int(beam[0, hard_slot[0]]),
            'hard_rigid_five_point_mm': float(error_mm[0, hard_slot[0]]),
            'hard_support_fraction_gap': float(support_gap[0, hard_slot[0]]),
            'hard_support_matched': bool(matched[0, hard_slot[0]]),
            'rigid_exact_dense_mm': float(rigid_error),
            'mapped_exact_dense_mm': float(mapped_error[0, 0]),
            'mapped_hard_dense_mm': float(mapped_error[0, 1]),
            'fitted_score_exact_hard_near_far': scores.tolist(),
            'swapped_image_fitted_score': float(swapped_score),
            'mapped_oracle_silhouette_dice_exact_hard_near_far': silhouette_dice.tolist(),
            'fitted_exact_minus_hard': float(scores[0] - scores[1]),
            'silhouette_exact_minus_hard': float(silhouette_dice[0] - silhouette_dice[1]),
            'fitted_exact_minus_swapped': float(scores[0] - swapped_score),
            'fitted_near_minus_far': float(scores[2] - scores[3]),
            'silhouette_near_minus_far': float(silhouette_dice[2] - silhouette_dice[3]),
            'near_score_derivative_outward_per_u': float(derivative),
            'correct_ranks_over_hard': bool(scores[0] > scores[1]),
            'correct_ranks_over_swapped': bool(scores[0] > swapped_score),
            'near_ranks_over_far': bool(scores[2] > scores[3]),
            'near_gradient_points_toward_correct': bool(derivative < 0),
            'silhouette_ranks_correct_over_hard': bool(silhouette_dice[0] > silhouette_dice[1]),
            'silhouette_near_ranks_over_far': bool(silhouette_dice[2] > silhouette_dice[3])})
    print(json.dumps({'step': step, 'sections_done': len(selected)}), flush=True)

for index in range(len(selected)):
    same_section = [rows[step_index * len(selected) + index] for step_index in range(len(steps))]
    assert len({row['hard_branch'] for row in same_section}) == 1
    assert len({row['perturbation_seed'] for row in same_section}) == 1

summary = {'sections_per_step': len(selected), 'plans': len(plan_ids),
    'scope': 'identity-disjoint synthetic DEV from the same Allen atlas; not real-animal truth, calibration or public benchmark',
    'silhouette_control': 'oracle surviving-tissue valid mask versus mapped atlas support Dice; non-deployable and excludes image intensity',
    'checkpoints': {}}
for step in steps:
    subset = [row for row in rows if row['step'] == step]
    per_plan = {plan: [row for row in subset if row['synthetic_subject_plan_id'] == plan]
        for plan in plan_ids}
    metrics = ('rigid_exact_dense_mm', 'mapped_exact_dense_mm',
        'hard_rigid_five_point_mm', 'hard_support_fraction_gap',
        'fitted_exact_minus_hard', 'silhouette_exact_minus_hard',
        'fitted_exact_minus_swapped', 'fitted_near_minus_far',
        'silhouette_near_minus_far',
        'near_score_derivative_outward_per_u',
        'correct_ranks_over_hard', 'correct_ranks_over_swapped',
        'near_ranks_over_far', 'near_gradient_points_toward_correct',
        'silhouette_ranks_correct_over_hard', 'silhouette_near_ranks_over_far')
    summary['checkpoints'][str(step)] = {
        'plan_equal': {key: float(np.mean([np.mean([row[key] for row in group])
            for group in per_plan.values()])) for key in metrics},
        'by_plan': {plan: {key: float(np.mean([row[key] for row in group]))
            for key in metrics} for plan, group in per_plan.items()},
        'support_matched_hard_count': sum(row['hard_support_matched'] for row in subset),
        'mapped_exact_better_than_rigid_count': sum(row['mapped_exact_dense_mm'] <
            row['rigid_exact_dense_mm'] for row in subset)}

output_config = {'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': done['config_sha256'],
    'training_draws_sha256': done['draws_sha256'],
    'checkpoint_sha256': done['checkpoint_sha256'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': panel_done['protocol_sha256'],
    'panel_records_sha256': panel_done['records_sha256'],
    'plan_completed_sha256': panel_done['plan_completed_sha256'],
    'selected_section_ids_sha256': hashlib.sha256(json.dumps(
        [row['section_id'] for row in selected]).encode()).hexdigest(),
    'source_sha256': sha(source / 'evaluate_v3_fitting_evidence_112.py'),
    'steps': steps, 'map_side': map_side,
    'hard_candidate_rule': 'frozen-pose top16; >=1.5mm five-point wrong and <=0.10 atlas-support-fraction gap if available; else wrong/farthest',
    'candidate_priors': 'equal zero log mass and zero reflection logits',
    'perturbation_rule': 'independent section-ID-seeded u with fixed physical scales; near=0.5u, far=1.5u',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(output_config, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'sections_per_step': len(selected),
    'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'training_completed_sha256': output_config['training_completed_sha256'],
    'panel_completed_sha256': output_config['panel_completed_sha256'],
    'source_sha256': output_config['source_sha256'],
    'checkpoint_sha256': output_config['checkpoint_sha256'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections_per_step': len(selected),
    'summary': summary['checkpoints']}), flush=True)
