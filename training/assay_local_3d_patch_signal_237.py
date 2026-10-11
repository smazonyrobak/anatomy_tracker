"""Frozen TRAIN-only local 3-D patch signal on the exact 235 blind cases."""

import hashlib
import json
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
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.local_3d_patch_match_237 import local_3d_patch_match
from training.support_aware_framing_217 import sample_support_aware_framing_217


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def truth_at_chart(sample, chart):
    grid = (2 * (chart + .5 / 256) - 1).reshape(1, 1, -1, 2)
    target = F.grid_sample(sample['centre'].permute(0, 3, 1, 2), grid,
                           mode='bilinear', align_corners=False)[0, :, 0].T
    valid = F.grid_sample(sample['valid_mask'][:, None].float(), grid,
                          mode='nearest', align_corners=False)[0, 0, 0] > .5
    return target, valid


def fit_plane(state, reflection, match):
    chart = match['query_chart']
    canonical_x = torch.where(reflection.bool(), 255 / 256 - chart[:, 0], chart[:, 0])
    design = torch.stack((torch.ones_like(canonical_x), canonical_x - .5,
                          chart[:, 1] - .5), -1)
    weight = match['confidence'].clamp_min(0) * match['support'].float()
    if int((weight > 1e-6).sum()) < 12:
        return state, 0., 0
    weight = weight / weight.max().clamp_min(1e-6)
    for _ in range(3):
        gram = torch.einsum('ni,n,nj->ij', design, weight, design)
        right = torch.einsum('ni,n,nj->ij', design, weight, match['matched_world'])
        coefficients = torch.linalg.solve(gram + 1e-6 * torch.eye(3, device=state.device), right)
        residual = torch.linalg.vector_norm(design @ coefficients - match['matched_world'], dim=-1)
        weight = weight * (600 / residual.clamp_min(600)).detach()
    centre, frame, basis = full_frame_state_to_components(state)
    old = torch.stack((centre[0], (frame[0, :, :2] @ basis[0])[:, 0],
                       (frame[0, :, :2] @ basis[0])[:, 1]), 0)
    anchors = state.new_tensor(((1, -.5, -.5), (1, .5, -.5), (1, -.5, .5),
                                 (1, .5, .5), (1, 0, 0)))
    movement = torch.linalg.vector_norm(anchors @ (coefficients - old), dim=-1).mean()
    scale = (2500 / movement.clamp_min(2500)).clamp_max(1)
    coefficients = old + scale * (coefficients - old)
    origin = coefficients[0] - .5 * (coefficients[1] + coefficients[2])
    fitted = full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((origin, coefficients[1], coefficients[2]))[None]))
    return fitted, float(movement), int((weight > 1e-6).sum())


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

source = Path(__file__).resolve().parent
run = root / 'runs/local_3d_patch_signal_237_001'
protocol = source.parent / 'docs/publication/LOCAL_3D_PATCH_SIGNAL_237_PROTOCOL_20261011.md'
parent = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
previous = root / 'runs/blind_candidate_reachability_235_001/cases.jsonl'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert sha(parent) == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
assert sha(previous) == '1c081c4870b07af5d16cadfa1e43dafb62bbebacacc0b484a9732c03691f4e4c'
frozen = [json.loads(line) for line in previous.read_text().splitlines()]
assert len(frozen) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
del saved

rows = []
for case_index, record in enumerate(frozen):
    base, seed = record['base'], record['seed']
    sampler = sample_one_shot_slide_artifacts_v4 if record['appearance'] == 'v4' else sample_one_shot_slide_artifacts_v7
    with torch.inference_mode():
        geometry = sample_panel_geometry_train_166(context, [base], seed, side=256)
        framed = sample_support_aware_framing_217(geometry, seed)
        sample = sampler(context, [base], seed, side=256, source_sample=framed)
        assert bool(sample['eligible'][0])
        assert sample['provenance'][0]['physical_section_id'] == record['physical_section_id']
        assert hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest() == record['input_sha256']
        prediction = model.predict(sample['inputs'])
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        beam = prior.topk(16).indices.tolist()
        assert beam == [item['index'] for item in record['top16_candidates']]

    near = next((item for item in sorted(record['top16_candidates'],
        key=lambda value: value['tissue_mm']) if item['tissue_mm'] < 2), None)
    wrong = None if near is None else next((item for item in record['top16_candidates']
        if item['tissue_mm'] > 3 and item['reflection'] == near['reflection']
        and abs(item['in_support_fraction'] - near['in_support_fraction']) <= .05), None)
    roles = {'true': (sample['state'], sample['reflection'])}
    if near is not None:
        roles['near'] = (prediction['state'][:, near['index'] // 2],
                         torch.tensor([near['reflection']], device='cuda'))
    if wrong is not None:
        roles['wrong'] = (prediction['state'][:, wrong['index'] // 2],
                          torch.tensor([wrong['reflection']], device='cuda'))
    pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
                                   device='cuda').long()]
    sites = torch.stack((pixels.remainder(256),
        pixels.div(256, rounding_mode='floor')), -1).float() / 256
    truth_sites = sample['centre'][0].reshape(-1, 3)[pixels]
    row = {'case_index': case_index, 'appearance': record['appearance'],
           'base': base, 'seed': seed, 'physical_section_id': record['physical_section_id'],
           'input_sha256': record['input_sha256'], 'near_branch': None if near is None else near['index'],
           'wrong_branch': None if wrong is None else wrong['index'],
           'true_reflection': int(sample['reflection'][0]), 'results': {}}
    normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
    row['nearest_cardinal_angle_deg'] = float(torch.rad2deg(
        torch.acos(normal.abs().max().clamp(-1, 1))))
    row['visible_tissue_fraction'] = float((sample['visible'][0] > .25).float().mean())
    for role, (state, reflection) in roles.items():
        initial = rigid_points_090(state, reflection, sites)
        initial_error = float((initial[0] - truth_sites).norm(dim=-1).mean() / 1000)
        if role == 'near':
            assert abs(initial_error - near['tissue_mm']) < 1e-5
        if role == 'wrong':
            assert abs(initial_error - wrong['tissue_mm']) < 1e-5
        row['results'][role] = {'initial_mm': initial_error,
                                'reflection': int(reflection[0]), 'arms': {}}
        for arm in ('intact', 'null'):
            null_seed = None if arm == 'intact' else int.from_bytes(hashlib.sha256(
                f"237|{record['physical_section_id']}|{role}".encode()).digest()[:8], 'little') % (2**63)
            with torch.inference_mode():
                match = local_3d_patch_match(sample['inputs'][:, :1], state,
                    reflection, context['atlas'], sample['offsets'], sample['weights'],
                    null_seed=null_seed)
                target, valid = truth_at_chart(sample, match['query_chart'])
                supported = match['support'] & valid
                matched_error = (match['matched_world'] - target).norm(dim=-1) / 1000
                fitted, raw_movement, used = fit_plane(state, reflection, match)
                fitted_error = float((rigid_points_090(fitted, reflection, sites)[0]
                    - truth_sites).norm(dim=-1).mean() / 1000)
                row['results'][role]['arms'][arm] = {
                    'matched_valid_sites': int(supported.sum()),
                    'matched_mean_mm': float(matched_error[supported].mean()) if supported.any() else None,
                    'matched_le_0p5': float((matched_error[supported] <= .5).float().mean()) if supported.any() else None,
                    'matched_le_1p5': float((matched_error[supported] <= 1.5).float().mean()) if supported.any() else None,
                    'fitted_mm': fitted_error, 'raw_five_point_movement_mm': raw_movement / 1000,
                    'fit_sites': used, 'supported_query_sites': int(match['support'].sum()),
                    'score_mean_supported': float(match['score'][match['support']].mean())
                    if match['support'].any() else None,
                }
                if role in ('near', 'wrong'):
                    row['results'][role]['arms'][arm]['score'] = match['score'].cpu().tolist()
                    row['results'][role]['arms'][arm]['support'] = match['support'].cpu().tolist()
    if wrong is not None:
        row['pair'] = {}
        for arm in ('intact', 'null'):
            left, right = (row['results'][role]['arms'][arm] for role in ('near', 'wrong'))
            common = np.asarray(left['support']) & np.asarray(right['support'])
            row['pair'][arm] = {'common_sites': int(common.sum()),
                'near_minus_wrong_score': float(np.mean(np.asarray(left['score'])[common]
                    - np.asarray(right['score'])[common])) if common.any() else None}
    rows.append(row)
    if len(rows) % 16 == 0:
        print(json.dumps({'event': 'cases', 'count': len(rows)}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    near_rows = [r for r in group if r['near_branch'] is not None]
    pair_rows = [r for r in group if r.get('pair', {}).get('intact', {}).get('common_sites', 0) >= 16]
    summary[appearance] = {'cases': len(group), 'near_cases': len(near_rows),
        'paired_cases': len(pair_rows),
        'near_initial_mean_mm': float(np.mean([r['results']['near']['initial_mm'] for r in near_rows])),
        **{f'near_{arm}_fitted_mean_mm': float(np.mean([
            r['results']['near']['arms'][arm]['fitted_mm'] for r in near_rows]))
            for arm in ('intact', 'null')},
        **{f'true_{arm}_drift_mean_mm': float(np.mean([
            r['results']['true']['arms'][arm]['fitted_mm'] for r in group]))
            for arm in ('intact', 'null')},
        **{f'pair_{arm}_wins': sum(r['pair'][arm]['near_minus_wrong_score'] > 0
            for r in pair_rows) for arm in ('intact', 'null')},
        'near_reflection_correct': sum(r['results']['near']['reflection'] ==
            r['true_reflection'] for r in near_rows)}
summary['stage_a_pass'] = (summary['all']['near_initial_mean_mm']
    - summary['all']['near_intact_fitted_mean_mm'] >= .25
    and summary['all']['near_null_fitted_mean_mm']
    - summary['all']['near_intact_fitted_mean_mm'] >= .15
    and summary['all']['true_intact_drift_mean_mm'] <= .20
    and all(summary[a]['near_initial_mean_mm'] > summary[a]['near_intact_fitted_mean_mm']
        for a in ('v4', 'v7')))

run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
source_sha = {name: sha(source / name) for name in (
    'assay_local_3d_patch_signal_237.py', 'local_3d_patch_match_237.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'support_aware_framing_217.py', 'global_atlas_contrast_090.py')}
(run / 'completed.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'script_and_source_sha256': source_sha,
    'parent_checkpoint_sha256': sha(parent), 'frozen_235_rows_sha256': sha(previous),
    'output_sha256': {name: sha(run / name) for name in ('rows.jsonl', 'summary.json')},
    'cases': len(rows), 'training_updates': 0, 'reserved_animal_used': False,
    'public_benchmark_used': False, 'calibrated': False}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
