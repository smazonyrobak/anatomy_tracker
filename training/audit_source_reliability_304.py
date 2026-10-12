"""Fresh-section image-only reliability and atlas correlation diagnostic."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.oriented_slab_ranker_299 import render_bank_299


SEED, SIDE = 20261012304, 192
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/SOURCE_RELIABILITY_304_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
RUN = ROOT / 'reports/source_reliability_304_002'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(image, atlas, support, reliability):
    visible = (support > .95).float()
    weights = reliability[None] * visible
    mass = weights.sum((1, 2))
    total = reliability.sum()
    if float(total) == 0:
        zeros = mass.new_zeros(len(mass))
        return zeros, zeros, zeros
    coverage = mass / total
    effective = mass.square() / weights.square().sum((1, 2)).clamp_min(1e-8)
    centre_image = (weights * image[None]).sum((1, 2)) / mass.clamp_min(1e-8)
    centre_atlas = (weights * atlas).sum((1, 2)) / mass.clamp_min(1e-8)
    left = image[None] - centre_image[:, None, None]
    right = atlas - centre_atlas[:, None, None]
    numerator = (weights * left * right).sum((1, 2))
    denominator = ((weights * left.square()).sum((1, 2))
                   * (weights * right.square()).sum((1, 2))).sqrt().clamp_min(1e-8)
    correlation = torch.where(effective >= 32, (numerator / denominator).abs(), 0.)
    evidence = coverage * correlation - .5 * (1 - coverage)
    support_only = -.5 * (1 - coverage)
    return evidence, support_only, effective


assert sha(PARENT) == 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
context = load_streaming_synthetic_v7_64('cuda')
saved = torch.load(PARENT, map_location='cpu', weights_only=False)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
del saved
config = {'candidate': 304, 'seed': SEED, 'side': SIDE,
    'bases': [56, 63], 'sections_per_base': 16,
    'fixed_atlas_render_thickness_um': 50,
    'fixed_atlas_psf_weights': [1, 2, 2, 2, 2, 2, 2, 2, 1],
    'parent_checkpoint_sha256': sha(PARENT), 'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'audit_source_reliability_304.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_model.py',
        'oriented_slab_ranker_299.py')},
    'synthetic_provenance': context['provenance'],
    'fresh_sections_not_new_base_animals': True,
    'oracle_masks_used_for_error_only': True,
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rng = np.random.default_rng(SEED)
rows, attempted = [], 0
with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'cases.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for base in range(56, 64):
            for slot in range(16):
                while True:
                    attempted += 1
                    variant = int(rng.integers(VARIANTS))
                    seed = SEED + 4_000_000_000 + attempted
                    subject = base * VARIANTS + variant
                    original = sample_one_shot_slide_artifacts_v9(
                        context, [subject], seed, SIDE)
                    eligible = bool(original['eligible'][0])
                    identity = original['provenance'][0]
                    draws.write(json.dumps({'attempt': attempted, 'base': base,
                        'slot': slot, 'variant': variant, 'seed': seed,
                        'accepted': eligible, 'physical_section_id': identity['physical_section_id'],
                        'provenance': identity}, allow_nan=False) + '\n')
                    if eligible:
                        break
                    del original
                sample = sample_postartifact_gauge_v1(context, [subject], seed,
                    SIDE, source_sample=original)
                prediction = model.predict(sample['inputs'])
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                order = prior.argsort(descending=True)[:16]
                states = prediction['state'][0, order // 2]
                reflections = order % 2
                prior = prior[order]
                pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
                pixels = pixels[torch.linspace(0, len(pixels) - 1, min(256, len(pixels)),
                    device=pixels.device).long()]
                chart = torch.stack((pixels.remainder(SIDE), pixels.div(SIDE,
                    rounding_mode='floor')), -1).float() / SIDE
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                points = rigid_points_090(states[None], reflections[None],
                    chart[None], (SIDE, SIDE))[0]
                error = (points - truth[None]).norm(dim=-1).mean(-1) / 1000
                offsets = states.new_tensor(np.linspace(-25, 25, 9)).expand(16, -1)
                weights = states.new_tensor([1, 2, 2, 2, 2, 2, 2, 2, 1])[None].expand(16, -1)
                bank = render_bank_299(context['atlas'], states, reflections,
                    offsets, weights, depth_offsets=(0.,))[:, 0]
                support = bank[:, 1].clamp(0, 1)
                intensity = bank[:, 0] / support.clamp_min(1e-4)
                image = F.avg_pool2d(sample['inputs'][:, :1], 4)[0, 0]
                local_mean = F.avg_pool2d(image[None, None], 5, stride=1, padding=2,
                                          count_include_pad=False)[0, 0]
                mad = F.avg_pool2d((image - local_mean).abs()[None, None],
                                   5, stride=1, padding=2, count_include_pad=False)[0, 0]
                q50, q90 = torch.quantile(mad, mad.new_tensor([.5, .9]))
                reliability = ((mad - q50) / (q90 - q50 + 1e-6)).clamp(0, 1)
                full, support_only, effective = score(image, intensity, support,
                                                      reliability)
                uniform, _, _ = score(image, intensity, support,
                                      torch.ones_like(reliability))
                if float(reliability.sum()) == 0:
                    full.zero_(); support_only.zero_()
                evidence = {'full': full, 'support_only': support_only, 'uniform': uniform}
                selected = {name: int((prior + 4 * values).argmax())
                            for name, values in evidence.items()}
                central_support = support.mean((1, 2))
                near = int(error.argmin())
                allowed = ((error >= max(2., float(error[near]) + 1.))
                           & ((central_support - central_support[near]).abs() <= .10))
                pair = None
                if error[near] <= 1.5 and bool(allowed.any()):
                    pair = [near, int(torch.where(allowed, prior, -torch.inf).argmax())]
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                angle = float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1))))
                row = {'physical_section_id': identity['physical_section_id'],
                    'base': base, 'slot': slot, 'variant': variant, 'seed': seed,
                    'provenance': sample['provenance'][0],
                    'brush_available': bool(sample['inputs'][0, 2, 0, 0]),
                    'appearance_inverted': bool(identity['appearance']['invert']),
                    'exterior_style': identity['one_shot_slide_artifacts_v9']['exterior_style'],
                    'obliquity_deg': angle,
                    'reliability_mass': float(reliability.sum()),
                    'candidate_effective_weighted_count': effective.tolist(),
                    'candidate_support': central_support.tolist(),
                    'candidate_error_mm': error.tolist(),
                    'candidate_prior_log': prior.tolist(),
                    'candidate_branch_id': order.tolist(),
                    'near_far': pair,
                    'evidence': {name: values.tolist() for name, values in evidence.items()},
                    'selected': selected,
                    'selected_error_mm': {name: float(error[index])
                                          for name, index in selected.items()}}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                del original, sample, bank
            print(json.dumps({'event': 'base_complete', 'base': base,
                              'sections': len(rows), 'attempts': attempted}), flush=True)

assert len(rows) == 128 and len({row['physical_section_id'] for row in rows}) == 128
pairs = [row for row in rows if row['near_far'] is not None]
no_brush = [row for row in rows if not row['brush_available']]
no_brush_pairs = [row for row in pairs if not row['brush_available']]


def wins(group, name):
    return sum(row['evidence'][name][row['near_far'][0]] >
               row['evidence'][name][row['near_far'][1]] for row in group)


def mean_error(group, name):
    return float(np.mean([row['candidate_error_mm'][0] if name == 'prior'
                          else row['selected_error_mm'][name] for row in group]))


per_base = {str(base): {'prior': mean_error([row for row in rows if row['base'] == base], 'prior'),
    'full': mean_error([row for row in rows if row['base'] == base], 'full')}
    for base in range(56, 64)}
summary = {'n': len(rows), 'attempted': attempted,
    'matched_pairs': len(pairs), 'matched_pairs_no_brush': len(no_brush_pairs),
    'near_far_wins': {name: wins(pairs, name) for name in ('full', 'support_only', 'uniform')},
    'near_far_wins_no_brush': {name: wins(no_brush_pairs, name)
                               for name in ('full', 'support_only', 'uniform')},
    'mean_error_mm': {name: mean_error(rows, name)
                      for name in ('prior', 'full', 'support_only', 'uniform')},
    'mean_error_no_brush_mm': {name: mean_error(no_brush, name)
                               for name in ('prior', 'full', 'support_only', 'uniform')},
    'oracle_best_top16_mean_error_mm': float(np.mean(
        [min(row['candidate_error_mm']) for row in rows])),
    'per_base': per_base,
    'zero_reliability_cases': sum(row['reliability_mass'] == 0 for row in rows),
    'oblique_45plus': sum(row['obliquity_deg'] >= 45 for row in rows)}
groups = {'brush': [row for row in rows if row['brush_available']],
    'no_brush': no_brush,
    'inverted': [row for row in rows if row['appearance_inverted']],
    'not_inverted': [row for row in rows if not row['appearance_inverted']],
    'oblique_45plus': [row for row in rows if row['obliquity_deg'] >= 45]}
groups.update({style: [row for row in rows if row['exterior_style'] == style]
    for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
summary['strata'] = {name: {'n': len(group),
    'pairs': sum(row['near_far'] is not None for row in group),
    'full_pair_wins': wins([row for row in group if row['near_far'] is not None], 'full'),
    'prior_mean_error_mm': mean_error(group, 'prior'),
    'full_mean_error_mm': mean_error(group, 'full'),
    'support_only_mean_error_mm': mean_error(group, 'support_only')}
    for name, group in groups.items() if group}
summary['gate'] = {
    'pairs_60_and_no_brush_30': len(pairs) >= 60 and len(no_brush_pairs) >= 30,
    'full_pair_75pct': wins(pairs, 'full') >= .75 * len(pairs),
    'full_no_brush_pair_70pct': wins(no_brush_pairs, 'full') >= .70 * len(no_brush_pairs),
    'full_pair_advantage_10points': (wins(pairs, 'full') - wins(pairs, 'support_only'))
        >= .10 * len(pairs),
    'full_gain_prior_0p20mm': mean_error(rows, 'prior') - mean_error(rows, 'full') >= .20,
    'full_gain_support_0p15mm': mean_error(rows, 'support_only') - mean_error(rows, 'full') >= .15,
    'bases_6of8_nonworse': sum(item['full'] <= item['prior']
        for item in per_base.values()) >= 6,
    'no_brush_gain_prior_0p10mm': mean_error(no_brush, 'prior')
        - mean_error(no_brush, 'full') >= .10}
summary['gate']['pass'] = all(summary['gate'].values())
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
receipt = {'candidate': 304, 'summary': summary,
    'output_sha256': {name: sha(RUN / name) for name in
        ('config.json', 'draws.jsonl', 'cases.jsonl', 'summary.json')},
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
