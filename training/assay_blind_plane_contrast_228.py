"""Frozen TRAIN-inner atlas-contrast signal on imperfect 226 blind candidates."""

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
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def erode(mask):
    return F.avg_pool2d(mask[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6


def correlation(source, target, mask):
    a, b = source[mask], target[mask]
    a, b = a - a.mean(), b - b.mean()
    return (a * b).mean().abs() / (a.square().mean() * b.square().mean()).sqrt().clamp_min(1e-8)


def support_pair(left, right):
    common = erode(left & right)
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    iou = intersection / union if union else 0.
    area_gap = abs(int(left.sum()) - int(right.sum())) / (96 * 96)
    cells = int(common.sum())
    return common, iou, area_gap, cells, iou >= .85 and area_gap <= .08 and cells >= 64


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/BLIND_PLANE_CONTRAST_228_PROTOCOL_20261011.md'
run = root / 'runs/blind_plane_contrast_228_train_001'
parent_run = root / 'runs/pose_appearance_exposure_226_train_001'
parent = parent_run / 'joint_step_01200.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['1200']
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent, map_location='cpu', weights_only=True)
assert saved['step'] == 1200 and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
offsets = torch.linspace(-25, 25, 9, device='cuda')
weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda').float() / 16
attempts, cases = [], []
with torch.inference_mode():
    for appearance_index, appearance in enumerate(('v4', 'v7')):
        sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v7
        for base in (60, 61, 62, 63):
            accepted = attempt = 0
            while accepted < 24:
                seed = 202610112280000 + 1000000 * appearance_index + 10000 * base + 1009 * attempt
                geometry = sample_panel_geometry_train_166(context, [base], seed, side=256)
                framed = sample_support_aware_framing_217(geometry, seed)
                sample = sampler(context, [base], seed, side=256, source_sample=framed)
                eligible = bool(sample['eligible'][0])
                provenance = sample['provenance'][0]
                attempts.append({'appearance': appearance, 'base': base, 'attempt': attempt,
                    'seed': seed, 'used': eligible, 'provenance': provenance})
                attempt += 1
                if not eligible:
                    continue
                accepted += 1
                prediction = model.predict(sample['inputs'])
                prior = (prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                beam = prior.topk(16).indices
                states = torch.cat((prediction['state'][0, beam // 2], sample['state']))
                reflection = torch.cat((beam % 2, sample['reflection'].long()))
                rendered = render_atlas_planes_090(context['atlas'], states[None],
                    reflection[None], offsets, weights, source_shape=(256, 256))[0]
                observed = F.avg_pool2d(F.interpolate(sample['inputs'][:, :1], (96, 96),
                    mode='area'), 3, 1, 1)[0, 0]
                atlas = F.avg_pool2d(rendered[:, :1], 3, 1, 1)[:, 0]
                support = rendered[:, 1] > .5
                shuffled = []
                for index in range(17):
                    null_seed = int.from_bytes(hashlib.sha256(
                        f"{provenance['physical_section_id']}|{index}".encode()).digest()[:8],
                        'little') % (2**63)
                    intensity = rendered[index, 0].flatten().clone()
                    inside = support[index].flatten().nonzero()[:, 0]
                    order = torch.randperm(len(inside), device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(null_seed))
                    intensity[inside] = intensity[inside][order]
                    shuffled.append(intensity.reshape(96, 96))
                null_atlas = F.avg_pool2d(torch.stack(shuffled)[:, None], 3, 1, 1)[:, 0]
                masks = [erode(s) for s in support]
                eligible_score = [int(mask.sum()) >= 64 for mask in masks[:16]]
                contrast = torch.stack([correlation(observed, atlas[index], masks[index])
                    if eligible_score[index] else observed.new_tensor(0.) for index in range(16)])
                null = torch.stack([correlation(observed, null_atlas[index], masks[index])
                    if eligible_score[index] else observed.new_tensor(0.) for index in range(16)])
                log_prior = prior[beam]
                if any(eligible_score):
                    allowed = torch.tensor(eligible_score, device='cuda')
                    contrast_only = int(contrast.masked_fill(~allowed, -1e9).argmax())
                    fused = int((log_prior + 2 * contrast).masked_fill(~allowed, -1e9).argmax())
                    fused4 = int((log_prior + 4 * contrast).masked_fill(~allowed, -1e9).argmax())
                    null_fused = int((log_prior + 2 * null).masked_fill(~allowed, -1e9).argmax())
                else:
                    contrast_only = fused = fused4 = null_fused = 0
                pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
                pixels = pixels[torch.linspace(0, len(pixels) - 1,
                    min(128, len(pixels)), device='cuda').long()]
                sites = torch.stack((pixels.remainder(256), pixels.div(256,
                    rounding_mode='floor')), -1).float() / 256
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                points = rigid_points_090(states, reflection, sites)
                error = (points - truth[None]).norm(dim=-1).mean(-1) / 1000
                target_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                normal = full_frame_state_to_components(states)[1][..., :, 2]
                axis_error = torch.rad2deg(torch.acos((normal * target_normal).sum(-1).abs().clamp_max(1)))
                near = int(error[:16].argmin())
                wrong = None
                pair = None
                if float(error[near]) < 2:
                    for index in range(16):
                        if float(error[index] - error[near]) < 1:
                            continue
                        common, iou, area_gap, cells, okay = support_pair(support[near], support[index])
                        if okay:
                            wrong = index
                            pair = {'near_index': near, 'wrong_index': index,
                                'near_error_mm': float(error[near]), 'wrong_error_mm': float(error[index]),
                                'near_prior': float(log_prior[near]), 'wrong_prior': float(log_prior[index]),
                                'support_iou': iou, 'support_area_fraction_gap': area_gap,
                                'common_eroded_cells': cells,
                                'intact_margin': float(correlation(observed, atlas[near], common)
                                    - correlation(observed, atlas[index], common)),
                                'null_margin': float(correlation(observed, null_atlas[near], common)
                                    - correlation(observed, null_atlas[index], common))}
                            break
                exact_pair = None
                if wrong is not None:
                    common, iou, area_gap, cells, okay = support_pair(support[16], support[wrong])
                    exact_pair = {'eligible': okay, 'true_gauge_error_mm': float(error[16]),
                        'support_iou': iou, 'support_area_fraction_gap': area_gap,
                        'common_eroded_cells': cells}
                    if okay:
                        exact_pair['intact_margin'] = float(correlation(observed, atlas[16], common)
                            - correlation(observed, atlas[wrong], common))
                        exact_pair['null_margin'] = float(correlation(observed, null_atlas[16], common)
                            - correlation(observed, null_atlas[wrong], common))
                cases.append({'appearance': appearance, 'base': base, 'seed': seed,
                    'physical_section_id': provenance['physical_section_id'],
                    'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
                    'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'provenance': provenance, 'beam_branches': beam.cpu().tolist(),
                    'beam_prior': log_prior.cpu().tolist(), 'beam_error_mm': error[:16].cpu().tolist(),
                    'beam_axis_deg': axis_error[:16].cpu().tolist(),
                    'beam_own_support_cells': [int(mask.sum()) for mask in masks[:16]],
                    'beam_contrast': contrast.cpu().tolist(), 'beam_null_contrast': null.cpu().tolist(),
                    'score_eligible_n': sum(eligible_score), 'prior_choice': 0,
                    'contrast_choice': contrast_only, 'fused_choice': fused,
                    'fused4_choice': fused4, 'null_fused_choice': null_fused,
                    'top16_oracle_error_mm': float(error[near]), 'near_wrong_pair': pair,
                    'true_gauge_vs_wrong': exact_pair})
                if len(cases) % 48 == 0:
                    print(json.dumps({'event': 'scored_cases', 'count': len(cases)}), flush=True)

assert len(cases) == 192 and len({row['physical_section_id'] for row in cases}) == 192
summary = {}
for appearance in ('v4', 'v7'):
    part = [row for row in cases if row['appearance'] == appearance]
    pairs = [row['near_wrong_pair'] for row in part if row['near_wrong_pair'] is not None]
    exact = [row['true_gauge_vs_wrong'] for row in part if row['true_gauge_vs_wrong']
             and row['true_gauge_vs_wrong']['eligible']]
    choice = {name: [row['beam_error_mm'][row[name + '_choice']] for row in part]
              for name in ('prior', 'contrast', 'fused', 'fused4', 'null_fused')}
    summary[appearance] = {'cases': len(part), 'attempts': sum(row['appearance'] == appearance for row in attempts),
        'no_candidate_contrast_support': sum(row['score_eligible_n'] == 0 for row in part),
        'top16_oracle_lt2mm': sum(row['top16_oracle_error_mm'] < 2 for row in part),
        'selected_median_tissue_mm': {name: float(np.median(values)) for name, values in choice.items()},
        'selected_lt2mm': {name: sum(value < 2 for value in values) for name, values in choice.items()},
        'selected_axis_median_deg': {name: float(np.median([
            row['beam_axis_deg'][row[name + '_choice']] for row in part])) for name in choice},
        'near_wrong_eligible': len(pairs),
        'near_wrong_intact_wins': sum(row['intact_margin'] > 0 for row in pairs),
        'near_wrong_null_wins': sum(row['null_margin'] > 0 for row in pairs),
        'near_wrong_prior_wins': sum(row['near_prior'] > row['wrong_prior'] for row in pairs),
        'true_gauge_wrong_eligible': len(exact),
        'true_gauge_wrong_intact_wins': sum(row['intact_margin'] > 0 for row in exact),
        'true_gauge_wrong_null_wins': sum(row['null_margin'] > 0 for row in exact)}
gate = (all(summary[a]['near_wrong_eligible'] >= 24 for a in ('v4', 'v7')) and
    summary['v7']['near_wrong_intact_wins'] / summary['v7']['near_wrong_eligible'] >= .75 and
    (summary['v7']['near_wrong_intact_wins'] - summary['v7']['near_wrong_null_wins']) /
    summary['v7']['near_wrong_eligible'] >= .10 and
    summary['v7']['selected_median_tissue_mm']['prior'] -
    summary['v7']['selected_median_tissue_mm']['fused'] >= .30 and
    summary['v4']['selected_median_tissue_mm']['fused'] -
    summary['v4']['selected_median_tissue_mm']['prior'] <= .20)
summary['gate'] = 'pass' if gate else 'inconclusive' if any(
    summary[a]['near_wrong_eligible'] < 24 for a in ('v4', 'v7')) else 'fail'
run.mkdir(parents=True, exist_ok=False)
for name, records in (('attempts.jsonl', attempts), ('cases.jsonl', cases)):
    with (run / name).open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
source_files = ('assay_blind_plane_contrast_228.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_panel_geometry_train_166.py', 'support_aware_framing_217.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py', 'global_atlas_contrast_090.py')
(run / 'completed.json').write_text(json.dumps({'accepted_cases': len(cases),
    'attempts': len(attempts), 'parent_checkpoint_sha256': sha(parent),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'output_sha256': {name: sha(run / name) for name in
        ('attempts.jsonl', 'cases.jsonl', 'summary.json')},
    'calibrated': False, 'reserved_dev_cal_final_used': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
