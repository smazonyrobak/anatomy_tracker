"""TRAIN-inner 221 diagnostic: ranked pose proposals versus affine and deformed truth."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
frozen = root / 'runs/pose_framing_221_inner_readout_001'
run = root / 'runs/pose_framing_221_affine_dense_diagnostic_001'
checkpoints = {
    '187_step_0800': root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt',
    '221_framed_step_0400': root / 'runs/pose_framing_221_framed_train_001/joint_step_00400.pt',
}
assert source.drive.upper() == root.drive.upper() == run.drive.upper() == 'I:'
assert not run.exists()
receipt = json.loads((frozen / 'completed.json').read_text())
for name, expected in receipt['source_sha256'].items():
    with (source / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
with (frozen / 'cases.jsonl').open('rb') as stream:
    frozen_cases_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
assert frozen_cases_sha256 == receipt['output_sha256']['cases.jsonl']
with (frozen / 'readout.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['output_sha256']['readout.jsonl']
fixed = [json.loads(line) for line in (frozen / 'cases.jsonl').read_text().splitlines()]
fixed = [row for row in fixed if row['panel'] == 'framed']
assert len(fixed) == 64 and len({row['physical_section_id'] for row in fixed}) == 64
checkpoint_sha256 = {}
for label, path in checkpoints.items():
    with path.open('rb') as stream:
        checkpoint_sha256[label] = hashlib.file_digest(stream, 'sha256').hexdigest()
    completed = json.loads((path.parent / 'completed.json').read_text())
    step = '800' if label.startswith('187') else '400'
    assert checkpoint_sha256[label] == completed['checkpoint_sha256'][step]
    assert receipt['input_sha256'][str(path)] == checkpoint_sha256[label]

context = load_streaming_synthetic_v7_64(device='cuda')
cases = []
for base in (60, 61, 62, 63):
    accepted = attempt = 0
    while accepted < 16:
        seed = 202610112210000 + 1000000 + 10000 * base + 1009 * attempt
        appearance = 'v4' if accepted % 2 == 0 else 'v3'
        sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                   else sample_one_shot_slide_artifacts_v3)
        with torch.no_grad():
            source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
            source_sample = sample_support_aware_framing_217(source_sample, seed)
            sample = sampler(context, [base], seed, side=256,
                             source_sample=source_sample)
        attempt += 1
        if not bool(sample['eligible'][0]):
            continue
        case = fixed[len(cases)]
        assert case['base'] == base and case['seed'] == seed and case['appearance'] == appearance
        replay = json.loads(json.dumps(sample['provenance'][0], allow_nan=False))
        if case['provenance'] != replay:
            print(json.dumps({'event': 'provenance_mismatch', 'case': len(cases),
                'changed_keys': [key for key in case['provenance']
                    if case['provenance'][key] != replay.get(key)],
                'frozen_section_id': case['physical_section_id'],
                'replay_section_id': replay['physical_section_id']}), flush=True)
            raise AssertionError('frozen case provenance changed')
        assert case['input_sha256'] == hashlib.sha256(
            sample['inputs'].cpu().numpy().tobytes()).hexdigest()
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1, min(128, len(pixels)),
            device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        cases.append((case, sample['inputs'].cpu(), sites.cpu(),
            sample['centre'][0].reshape(-1, 3)[pixels].cpu(),
            sample['state'].cpu(), sample['reflection'].cpu()))
        accepted += 1
assert len(cases) == 64
print(json.dumps({'event': 'frozen_cases_verified', 'count': len(cases)}), flush=True)

rows = []
for label, path in checkpoints.items():
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()
    attach_global_plane_matcher(model, enabled=True)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    for case, image, sites_cpu, dense_cpu, truth_state_cpu, truth_reflection_cpu in cases:
        with torch.inference_mode():
            sites, dense = sites_cpu.cuda(), dense_cpu.cuda()
            truth_state = truth_state_cpu.cuda()
            truth_reflection = truth_reflection_cpu.cuda()
            rigid_truth = rigid_points_090(truth_state, truth_reflection, sites)[0]
            prediction = model.predict(image.cuda())
            states = prediction['state'][0].repeat_interleave(2, 0)
            reflections = torch.arange(2, device='cuda').repeat(model.modes)
            prior = (prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
            proposed = rigid_points_090(states, reflections, sites)
            affine_error = (proposed - rigid_truth).norm(dim=-1).mean(-1) / 1000
            dense_error = (proposed - dense).norm(dim=-1).mean(-1) / 1000
            true_rigid_dense = (rigid_truth - dense).norm(dim=-1).mean() / 1000
            truth_normal = full_frame_state_to_components(truth_state)[1][0, :, 2]
            candidate_normals = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
            normal_angle = torch.rad2deg(torch.acos((candidate_normals * truth_normal
                ).sum(-1).abs().clamp_max(1)))
            ideal_angle = torch.rad2deg(torch.acos((model.normal_anchor_frames[:, :, 2]
                * truth_normal).sum(-1).abs().max().clamp_max(1)))
            top16 = prior.topk(16).indices
            first, best_affine, best_dense = int(prior.argmax()), int(affine_error.argmin()), int(dense_error.argmin())
            top16_list = top16.tolist()
            affine_list, dense_list, prior_list = (affine_error.tolist(), dense_error.tolist(), prior.tolist())
            anchor_start = 2 * model.base_modes
            row = {
                'checkpoint': label, 'checkpoint_sha256': checkpoint_sha256[label],
                'physical_section_id': case['physical_section_id'],
                'input_sha256': case['input_sha256'], 'base': case['base'],
                'seed': case['seed'], 'appearance': case['appearance'],
                'visible_fraction': case['visible_fraction'],
                'cardinal_angle_deg': case['cardinal_angle_deg'],
                'true_rigid_to_dense_mm': float(true_rigid_dense),
                'ideal_nearest_anchor_axis_deg': float(ideal_angle),
                'predicted_nearest_anchor_axis_deg': float(normal_angle[model.base_modes:].min()),
                'predicted_nearest_base_axis_deg': float(normal_angle[:model.base_modes].min()),
                'top1_candidate': first, 'top1_mode': first // 2,
                'top1_family': 'base' if first < anchor_start else 'anchor',
                'top1_affine_mm': affine_list[first], 'top1_dense_mm': dense_list[first],
                'top16_candidate_indices': top16_list,
                'top16_best_affine_mm': min(affine_list[i] for i in top16_list),
                'top16_best_dense_mm': min(dense_list[i] for i in top16_list),
                'top16_best_affine_base_mm': min((affine_list[i] for i in top16_list
                    if i < anchor_start), default=None),
                'top16_best_affine_anchor_mm': min((affine_list[i] for i in top16_list
                    if i >= anchor_start), default=None),
                'top16_best_dense_base_mm': min((dense_list[i] for i in top16_list
                    if i < anchor_start), default=None),
                'top16_best_dense_anchor_mm': min((dense_list[i] for i in top16_list
                    if i >= anchor_start), default=None),
                'best160_affine_candidate': best_affine,
                'best160_affine_family': 'base' if best_affine < anchor_start else 'anchor',
                'best160_affine_mm': affine_list[best_affine],
                'best160_dense_candidate': best_dense,
                'best160_dense_family': 'base' if best_dense < anchor_start else 'anchor',
                'best160_dense_mm': dense_list[best_dense],
                'best160_affine_base_mm': min(affine_list[:anchor_start]),
                'best160_affine_anchor_mm': min(affine_list[anchor_start:]),
                'best160_dense_base_mm': min(dense_list[:anchor_start]),
                'best160_dense_anchor_mm': min(dense_list[anchor_start:]),
                'affine_oracle_prior_rank': int((prior > prior[best_affine]).sum()) + 1,
                'dense_oracle_prior_rank': int((prior > prior[best_dense]).sum()) + 1,
                'anchor_probability_mass': float(prediction['log_mass'][0, model.base_modes:].exp().sum()),
                'affine_within_2mm_probability_mass': float(prior[affine_error < 2].exp().sum()),
                'dense_within_2mm_probability_mass': float(prior[dense_error < 2].exp().sum()),
                'candidate_log_prior': prior_list,
                'candidate_affine_mm': affine_list,
                'candidate_dense_mm': dense_list,
            }
        rows.append(row)
    selected = [row for row in rows if row['checkpoint'] == label]
    print(json.dumps({'checkpoint': label, 'n': len(selected),
        'top1_dense_median_mm': float(np.median([r['top1_dense_mm'] for r in selected])),
        'best160_dense_median_mm': float(np.median([r['best160_dense_mm'] for r in selected])),
        'true_rigid_to_dense_median_mm': float(np.median([
            r['true_rigid_to_dense_mm'] for r in selected]))}), flush=True)
    del model
    torch.cuda.empty_cache()

summary = {}
for label in checkpoints:
    selected = [row for row in rows if row['checkpoint'] == label]
    summary[label] = {
        'n': len(selected),
        'true_rigid_to_dense_median_mm': float(np.median([r['true_rigid_to_dense_mm'] for r in selected])),
        'top1_affine_median_mm': float(np.median([r['top1_affine_mm'] for r in selected])),
        'top1_dense_median_mm': float(np.median([r['top1_dense_mm'] for r in selected])),
        'top16_affine_median_mm': float(np.median([r['top16_best_affine_mm'] for r in selected])),
        'top16_dense_median_mm': float(np.median([r['top16_best_dense_mm'] for r in selected])),
        'best160_affine_median_mm': float(np.median([r['best160_affine_mm'] for r in selected])),
        'best160_dense_median_mm': float(np.median([r['best160_dense_mm'] for r in selected])),
        'top1_base_count': sum(r['top1_family'] == 'base' for r in selected),
        'affine_oracle_anchor_count': sum(r['best160_affine_family'] == 'anchor' for r in selected),
        'dense_oracle_anchor_count': sum(r['best160_dense_family'] == 'anchor' for r in selected),
        'anchor_probability_mass_median': float(np.median([r['anchor_probability_mass'] for r in selected])),
        'visible_fraction_median': float(np.median([r['visible_fraction'] for r in selected])),
    }
run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
with (source / 'diagnose_pose_framing_221.py').open('rb') as stream:
    source_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
output_sha256 = {}
for name in ('rows.jsonl', 'summary.json'):
    with (run / name).open('rb') as stream:
        output_sha256[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(run / 'completed.json').write_text(json.dumps({
    'case_count': len(cases), 'readout_count': len(rows),
    'source_sha256': source_sha256,
    'frozen_cases_sha256': frozen_cases_sha256,
    'checkpoint_sha256': checkpoint_sha256,
    'output_sha256': output_sha256,
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False,
    'calibrated': False}, indent=2))
print(json.dumps({'event': 'completed', 'case_count': len(cases),
    'readout_count': len(rows)}), flush=True)
