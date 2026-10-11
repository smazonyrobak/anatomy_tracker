"""Replay frozen 235 cases at their synthetic true plane, without a model."""

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

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090
from training.support_aware_framing_217 import sample_support_aware_framing_217


torch.set_num_threads(4)
source = Path(__file__).resolve().parent
run = root / 'runs/true_pose_deformation_floor_236_001'
frozen = root / 'runs/blind_candidate_reachability_235_001'
protocol = source.parent / 'docs/publication/TRUE_POSE_DEFORMATION_FLOOR_236_PROTOCOL_20261011.md'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
with (frozen / 'cases.jsonl').open('rb') as stream:
    frozen_rows_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert frozen_rows_hash == '1c081c4870b07af5d16cadfa1e43dafb62bbebacacc0b484a9732c03691f4e4c'
with (frozen / 'completed.json').open('rb') as stream:
    frozen_completion_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert frozen_completion_hash == '7e73cf7c024143048bd84fd6cff073c87a0d1ea4fe79dc37bc156bc213342454'
cases = [json.loads(line) for line in (frozen / 'cases.jsonl').open(encoding='utf-8')]
assert len(cases) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
rows = []

with torch.inference_mode():
    for frozen_case in cases:
        seed, base = frozen_case['seed'], frozen_case['base']
        sampler = (sample_one_shot_slide_artifacts_v4 if frozen_case['appearance'] == 'v4'
                   else sample_one_shot_slide_artifacts_v7)
        source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
        framed = sample_support_aware_framing_217(source_sample, seed)
        sample = sampler(context, [base], seed, side=256, source_sample=framed)
        input_hash = hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest()
        assert bool(sample['eligible'][0])
        assert sample['provenance'][0]['physical_section_id'] == frozen_case['physical_section_id']
        assert input_hash == frozen_case['input_sha256']
        pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixels = pixels[torch.linspace(0, len(pixels) - 1,
            min(128, len(pixels)), device='cuda').long()]
        sites = torch.stack((pixels.remainder(256),
            pixels.div(256, rounding_mode='floor')), -1).float() / 256
        truth = sample['centre'][0].reshape(-1, 3)[pixels]
        rigid = rigid_points_090(sample['state'], sample['reflection'], sites)[0]
        center, frame, basis = full_frame_state_to_components(sample['state'])
        through = torch.abs(torch.einsum('pc,c->p', truth - center[0], frame[0, :, 2])) / 1000
        shift = torch.linalg.vector_norm(torch.einsum('ij,pj->pi',
            torch.linalg.pinv(frame[0, :, :2] @ basis[0]), truth - rigid) * 256, dim=-1)
        reachable_fraction = float(((through <= .5) & (shift <= 32)).float().mean())
        near = [candidate for candidate in frozen_case['top16_candidates']
                if candidate['tissue_mm'] < 2]
        best = max(near, key=lambda candidate: candidate['reachable_site_fraction']) if near else None
        true_p75 = float(torch.quantile(through, .75))
        rows.append({'physical_section_id': frozen_case['physical_section_id'],
            'input_sha256': input_hash, 'base': base, 'appearance': frozen_case['appearance'],
            'seed': seed, 'true_rigid_tissue_mm': float((truth - rigid).norm(dim=-1).mean() / 1000),
            'true_through_plane_p75_mm': true_p75,
            'true_inplane_shift_p75_pixels': float(torch.quantile(shift, .75)),
            'true_reachable_site_fraction': reachable_fraction,
            'best_near_blind_p75_through_mm':
                best['through_plane_p75_mm'] if best else None,
            'best_near_blind_p75_excess_over_truth_mm':
                best['through_plane_p75_mm'] - true_p75 if best else None,
            'best_near_blind_reachable_site_fraction':
                best['reachable_site_fraction'] if best else None})

summary = {'n': len(rows),
    'true_rigid_tissue_median_mm': float(np.median([r['true_rigid_tissue_mm'] for r in rows])),
    'true_through_plane_p75_median_mm': float(np.median([
        r['true_through_plane_p75_mm'] for r in rows])),
    'true_inplane_shift_p75_median_pixels': float(np.median([
        r['true_inplane_shift_p75_pixels'] for r in rows])),
    'true_reachable_cases': sum(r['true_reachable_site_fraction'] >= .75 for r in rows),
    'near_candidate_cases': sum(r['best_near_blind_p75_through_mm'] is not None for r in rows),
    'near_candidate_paired_excess_p75_median_mm': float(np.median([
        r['best_near_blind_p75_excess_over_truth_mm'] for r in rows
        if r['best_near_blind_p75_excess_over_truth_mm'] is not None]))}
summary['interpretation'] = ('candidate_plane_error' if summary['true_through_plane_p75_median_mm'] <= .25
    else 'absolute_window_unfair' if summary['true_through_plane_p75_median_mm'] >= .75
    else 'intermediate')

run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
source_hashes = {}
for name in ('audit_true_pose_deformation_floor_236.py',
             'arbitrary_plane_one_shot_slide_artifacts_v4.py',
             'arbitrary_plane_one_shot_slide_artifacts_v7.py',
             'arbitrary_plane_panel_geometry_train_166.py',
             'arbitrary_plane_streaming_synthetic_v7_64.py',
             'arbitrary_plane_full_frame_primitives.py',
             'global_atlas_contrast_090.py', 'support_aware_framing_217.py'):
    with (source / name).open('rb') as stream:
        source_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
with protocol.open('rb') as stream:
    protocol_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
output_hashes = {}
for name in ('rows.jsonl', 'summary.json'):
    with (run / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(run / 'completed.json').write_text(json.dumps({
    'protocol_sha256': protocol_hash, 'source_sha256': source_hashes,
    'frozen_235_rows_sha256': frozen_rows_hash,
    'frozen_235_completion_sha256': frozen_completion_hash,
    'atlas_provenance': context['provenance'], 'n': len(rows),
    'output_sha256': output_hashes, 'training_updates': 0,
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False,
    'calibrated': False}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary}), flush=True)
