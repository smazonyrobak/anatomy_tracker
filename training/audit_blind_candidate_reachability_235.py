"""No-training reachability audit of the frozen 226 blind 160-plane proposals."""

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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
run = root / 'runs/blind_candidate_reachability_235_001'
training = root / 'runs/pose_appearance_exposure_226_train_001'
checkpoint = training / 'joint_step_01200.pt'
protocol = source.parent / 'docs/publication/BLIND_CANDIDATE_REACHABILITY_235_PROTOCOL_20261011.md'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
with checkpoint.open('rb') as stream:
    checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_hash == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
done = json.loads((training / 'completed.json').read_text())
assert checkpoint_hash == done['checkpoint_sha256']['1200']

context = load_streaming_synthetic_v7_64(device='cuda')
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(saved['model'], strict=True)
del saved

rng = np.random.default_rng(20261011235)
cases, draws = [], []
with torch.inference_mode():
    for base in (60, 61, 62, 63):
        for appearance, sampler in (('v4', sample_one_shot_slide_artifacts_v4),
                                    ('v7', sample_one_shot_slide_artifacts_v7)):
            accepted = 0
            while accepted < 8:
                seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                source_sample = sample_panel_geometry_train_166(context, [base], seed, side=256)
                framed = sample_support_aware_framing_217(source_sample, seed)
                sample = sampler(context, [base], seed, side=256, source_sample=framed)
                used = bool(sample['eligible'][0])
                physical_id = sample['provenance'][0]['physical_section_id']
                draws.append({'base': base, 'appearance': appearance, 'seed': seed,
                              'used': used, 'physical_section_id': physical_id})
                if not used:
                    continue
                accepted += 1
                pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
                pixels = pixels[torch.linspace(0, len(pixels) - 1,
                    min(128, len(pixels)), device='cuda').long()]
                sites = torch.stack((pixels.remainder(256),
                    pixels.div(256, rounding_mode='floor')), -1).float() / 256
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                prediction = model.predict(sample['inputs'])
                states = prediction['state'][0].repeat_interleave(2, 0)
                reflection = torch.arange(2, device='cuda').repeat(model.modes)
                prior = (prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                proposed = rigid_points_090(states, reflection, sites)
                error = (proposed - truth[None]).norm(dim=-1).mean(-1) / 1000
                center, frame, basis = full_frame_state_to_components(states)
                normal_residual = torch.abs(torch.einsum('kpc,kc->kp',
                    truth[None] - center[:, None], frame[:, :, 2])) / 1000
                edges = frame[:, :, :2] @ basis
                pixel_shift = torch.linalg.vector_norm(torch.einsum('kij,kpj->kpi',
                    torch.linalg.pinv(edges), truth[None] - proposed) * 256, dim=-1)
                reachable = ((normal_residual <= .5) & (pixel_shift <= 32)).float().mean(-1) >= .75
                indices = torch.round(proposed / 25).long()
                valid = ((indices >= 0) & (indices < torch.tensor(
                    context['atlas'].shape[1:], device='cuda'))).all(-1)
                clipped = torch.stack([indices[..., axis].clamp(0, size - 1)
                    for axis, size in enumerate(context['atlas'].shape[1:])], -1)
                support = (context['atlas'][1, clipped[..., 0], clipped[..., 1],
                    clipped[..., 2]] > .5) & valid
                support_fraction = support.float().mean(-1)
                top16 = prior.topk(16).indices
                near = error[top16] < 2
                wrong = error[top16] > 3
                same_reflection = reflection[top16, None] == reflection[None, top16]
                support_match = (support_fraction[top16, None] -
                    support_fraction[None, top16]).abs() <= .05
                pairs = near[:, None] & wrong[None, :] & same_reflection & support_match
                case = {'base': base, 'appearance': appearance, 'seed': seed,
                    'physical_section_id': physical_id,
                    'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
                    'visible_tissue_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'observed_tissue_sites': len(pixels),
                    'top1_tissue_mm': float(error[prior.argmax()]),
                    'oracle_all160_tissue_mm': float(error.min()),
                    'top16_near_2mm': bool(near.any()),
                    'top16_near_reachable': bool((near & reachable[top16]).any()),
                    'top16_support_matched_near_wrong_pairs': int(pairs.sum()),
                    'top16_candidates': [
                        {'index': int(index), 'reflection': int(reflection[index]),
                         'log_prior': float(prior[index]), 'tissue_mm': float(error[index]),
                         'in_support_fraction': float(support_fraction[index]),
                         'reachable_site_fraction': float(((normal_residual[index] <= .5) &
                            (pixel_shift[index] <= 32)).float().mean()),
                         'through_plane_p75_mm': float(torch.quantile(normal_residual[index], .75)),
                         'inplane_shift_p75_pixels': float(torch.quantile(pixel_shift[index], .75))}
                        for index in top16]}
                cases.append(case)
                print(json.dumps({'event': 'case', 'count': len(cases), 'appearance': appearance,
                    'top16_near': case['top16_near_2mm'],
                    'matched_pairs': case['top16_support_matched_near_wrong_pairs']}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [case for case in cases if case['appearance'] == appearance or appearance == 'all']
    summary[appearance] = {'n': len(group),
        'top1_median_mm': float(np.median([case['top1_tissue_mm'] for case in group])),
        'oracle_all160_median_mm': float(np.median([case['oracle_all160_tissue_mm'] for case in group])),
        'top16_near_count': sum(case['top16_near_2mm'] for case in group),
        'top16_near_reachable_count': sum(case['top16_near_reachable'] for case in group),
        'support_matched_pair_cases': sum(case['top16_support_matched_near_wrong_pairs'] > 0
            for case in group),
        'support_matched_pair_count': sum(case['top16_support_matched_near_wrong_pairs']
            for case in group)}
summary['stage_a_pass'] = (summary['all']['top16_near_count'] >= 32
    and summary['all']['support_matched_pair_cases'] >= 20)

run.mkdir(parents=True, exist_ok=False)
for name, records in (('draws.jsonl', draws), ('cases.jsonl', cases)):
    with (run / name).open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
with protocol.open('rb') as stream:
    protocol_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
with Path(__file__).open('rb') as stream:
    script_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
source_hashes = {}
for name in ('arbitrary_plane_one_shot_model.py',
             'arbitrary_plane_one_shot_slide_artifacts_v4.py',
             'arbitrary_plane_one_shot_slide_artifacts_v7.py',
             'arbitrary_plane_panel_geometry_train_166.py',
             'arbitrary_plane_streaming_synthetic_v7_64.py',
             'arbitrary_plane_full_frame_primitives.py',
             'global_atlas_contrast_090.py', 'global_plane_matcher_120.py',
             'support_aware_framing_217.py', 'arbitrary_plane_geometry.py'):
    with (source / name).open('rb') as stream:
        source_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
output_hashes = {}
for name in ('draws.jsonl', 'cases.jsonl', 'summary.json'):
    with (run / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(run / 'completed.json').write_text(json.dumps({
    'checkpoint_sha256': checkpoint_hash, 'protocol_sha256': protocol_hash,
    'script_sha256': script_hash, 'source_sha256': source_hashes,
    'source_provenance': context['provenance'],
    'draw_attempts': len(draws), 'accepted_cases': len(cases),
    'output_sha256': output_hashes,
    'reserved_dev_cal_final_used': False, 'public_benchmark_used': False,
    'training_updates': 0, 'calibrated': False}, indent=2), encoding='utf-8')
print(json.dumps({'event': 'completed', **summary}), flush=True)
