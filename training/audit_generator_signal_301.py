"""Actual v9 image-to-atlas signal across native and ranker resolutions."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import sample_streaming_synthetic_v7_64_appearance_v3
from training.oriented_slab_ranker_299 import render_bank_299
from training.run_smooth_image_fit_295 import rigid_render


SEED, SIDE = 20261012301, 192
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/GENERATOR_SIGNAL_301_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
RUN = ROOT / 'reports/generator_signal_301_001'
FIGURE = SOURCE.parent / 'docs/publication/figures/generator_signal_301_examples.png'
assert SOURCE.drive.upper() == RUN.drive.upper() == FIGURE.drive.upper() == 'I:'
assert not RUN.exists() and not FIGURE.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def correlation(left, right, mask):
    a, b = left[mask], right[mask]
    if len(a) < 16:
        return None
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt().clamp_min(1e-8))


def median_abs(values):
    values = [abs(value) for value in values if value is not None]
    return float(np.median(values)) if values else None


assert sha(PARENT) == 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
context = load_streaming_synthetic_v7_64('cuda')
saved = torch.load(PARENT, map_location='cpu', weights_only=False)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
del saved
config = {'candidate': 301, 'seed': SEED, 'section_count': 32,
    'bases': [48, 55], 'slots_per_base': 4,
    'parent_checkpoint_sha256': sha(PARENT), 'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'audit_generator_signal_301.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'oriented_slab_ranker_299.py',
        'run_smooth_image_fit_295.py')},
    'synthetic_provenance': context['provenance'],
    'score': 'absolute masked Pearson at exact dense CCF sites and 192/96/48/24 scales',
    'finite_thickness_approximation': 'nine original offsets along fitted postartifact central normal; local observed director unavailable',
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rng = np.random.default_rng(SEED)
rows, pictures, seen, attempted = [], [], set(), 0
with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'cases.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for base in range(48, 56):
            for slot in range(4):
                while True:
                    attempted += 1
                    variant = int(rng.integers(VARIANTS))
                    seed = SEED + 2_000_000_000 + attempted
                    subject = base * VARIANTS + variant
                    original = sample_one_shot_slide_artifacts_v9(context, [subject], seed, SIDE)
                    identity = original['provenance'][0]
                    accepted = bool(original['eligible'][0]) and identity['physical_section_id'] not in seen
                    draws.write(json.dumps({'attempt': attempted, 'base': base, 'slot': slot,
                        'variant': variant, 'seed': seed, 'accepted': accepted,
                        'provenance': identity}, allow_nan=False) + '\n')
                    if accepted:
                        break
                    del original
                seen.add(identity['physical_section_id'])
                sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE,
                                                      source_sample=original)
                observed = sample['inputs'][:, :1]
                source = sample_streaming_synthetic_v7_64_appearance_v3(
                    context, [subject], seed, SIDE)
                source_atlas = (source['clean'] / source['support'].clamp_min(1e-4))[:, None]
                warped_source = F.grid_sample(source_atlas,
                    sample['observed_to_source_grid'], align_corners=True)
                normal = full_frame_state_to_components(sample['state'])[1][:, :, 2]
                world = (sample['centre'][:, None] + sample['offsets'][:, :, None, None, None]
                         * normal[:, None, None, None])
                exact = render_finite_thickness_coordinate_grid(context['atlas'], world,
                    (0., 0., 0.), (25., 25., 25.), sample['weights'])
                exact_intensity = exact[:, :1] / exact[:, 1:2].clamp_min(1e-4)
                true_plane, _, _, _ = rigid_render(context['atlas'], sample['state'],
                    sample['reflection'], sample['offsets'], sample['weights'])
                true_intensity = true_plane[:, :1] / true_plane[:, 1:2].clamp_min(1e-4)
                prediction = model.predict(sample['inputs'])
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                first = int(prior.argmax())
                guess, _, _, _ = rigid_render(context['atlas'],
                    prediction['state'][0, first // 2][None],
                    torch.tensor([first % 2], device='cuda'), sample['offsets'], sample['weights'])
                guess_intensity = guess[:, :1] / guess[:, 1:2].clamp_min(1e-4)
                ranker_bank = render_bank_299(context['atlas'],
                    torch.cat((sample['state'], prediction['state'][0, first // 2][None])),
                    torch.cat((sample['reflection'],
                        torch.tensor([first % 2], device='cuda'))),
                    sample['offsets'].repeat(2, 1), sample['weights'].repeat(2, 1),
                    depth_offsets=(0.,))[:, 0]
                ranker_intensity = (ranker_bank[:, :1]
                                    / ranker_bank[:, 1:2].clamp_min(1e-4))
                intact = (sample['valid_mask'] & ~sample['fragment_displaced_mask']
                          & (sample['visible'] > .95))[:, None]
                metrics = {}
                for down in (1, 2, 4, 8):
                    fraction = F.avg_pool2d(intact.float(), down)
                    mask = fraction[0, 0] > .75
                    image = (F.avg_pool2d(observed * intact, down)
                             / fraction.clamp_min(1e-4))[0, 0]
                    target = (F.avg_pool2d(exact_intensity * intact, down)
                              / fraction.clamp_min(1e-4))[0, 0]
                    rigid = (F.avg_pool2d(true_intensity * intact, down)
                             / fraction.clamp_min(1e-4))[0, 0]
                    warped = (F.avg_pool2d(warped_source * intact, down)
                              / fraction.clamp_min(1e-4))[0, 0]
                    metrics[str(SIDE // down)] = {'pixels': int(mask.sum()),
                        'warped_source_signed_correlation': correlation(image, warped, mask),
                        'exact_signed_correlation': correlation(image, target, mask),
                        'rigid_signed_correlation': correlation(image, rigid, mask)}
                ranker_image = F.avg_pool2d(observed, 4)[0, 0]
                ranker_mask = F.avg_pool2d(intact.float(), 4)[0, 0] > .999
                ranker_true = correlation(ranker_image, ranker_intensity[0, 0], ranker_mask)
                ranker_guess = correlation(ranker_image, ranker_intensity[1, 0], ranker_mask)
                valid = intact[0, 0]
                values = exact_intensity[0, 0][valid]
                shuffled = values[torch.randperm(len(values), device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(SEED + attempted))]
                null = correlation(observed[0, 0][valid], shuffled,
                    torch.ones_like(shuffled, dtype=torch.bool))
                angle = float(torch.rad2deg(torch.acos(normal[0].abs().max().clamp(0, 1))))
                row = {'physical_section_id': identity['physical_section_id'],
                    'base': base, 'slot': slot, 'variant': variant, 'seed': seed,
                    'brush_available': bool(sample['inputs'][0, 2, 0, 0]),
                    'appearance_inverted': bool(identity['appearance']['invert']),
                    'obliquity_deg': angle,
                    'valid_intact_pixels': int(valid.sum()),
                    'metrics': metrics, 'shuffled_null_signed_correlation_192': null,
                    'ranker_48': {'pixels': int(ranker_mask.sum()),
                        'true_signed_correlation': ranker_true,
                        'first_choice_signed_correlation': ranker_guess},
                    'provenance': sample['provenance'][0]}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                pictures.append((observed[0, 0].cpu(), true_intensity[0, 0].cpu(),
                                 guess_intensity[0, 0].cpu(), row))
                del original, sample, source
            print(json.dumps({'event': 'base_complete', 'base': base,
                              'sections': len(rows)}), flush=True)

summary = {'n': len(rows), 'attempted': attempted,
    'oblique_45plus': sum(row['obliquity_deg'] >= 45 for row in rows),
    'brush': sum(row['brush_available'] for row in rows),
    'scales': {scale: {
        'median_abs_warped_source_correlation': median_abs(
            row['metrics'][scale]['warped_source_signed_correlation'] for row in rows),
        'median_abs_exact_correlation': median_abs(
            row['metrics'][scale]['exact_signed_correlation'] for row in rows),
        'median_abs_rigid_correlation': median_abs(
            row['metrics'][scale]['rigid_signed_correlation'] for row in rows),
        'usable_cases': sum(row['metrics'][scale]['exact_signed_correlation'] is not None
            for row in rows),
        'mean_valid_pixels': float(np.mean([row['metrics'][scale]['pixels'] for row in rows]))}
        for scale in ('192', '96', '48', '24')},
    'ranker_48': {
        'median_abs_true_correlation': median_abs(
            row['ranker_48']['true_signed_correlation'] for row in rows),
        'median_abs_first_choice_correlation': median_abs(
            row['ranker_48']['first_choice_signed_correlation'] for row in rows),
        'usable_cases': sum(row['ranker_48']['true_signed_correlation'] is not None
            and row['ranker_48']['first_choice_signed_correlation'] is not None for row in rows),
        'mean_pure_tissue_blocks': float(np.mean([row['ranker_48']['pixels'] for row in rows]))},
    'median_abs_shuffled_null_192': median_abs(
        row['shuffled_null_signed_correlation_192'] for row in rows),
    'native_exact_sign_agreement': sum(row['metrics']['192']['exact_signed_correlation'] is not None
        and (row['metrics']['192']['exact_signed_correlation'] < 0) == row['appearance_inverted']
        for row in rows)}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

ordered = sorted(range(len(rows)), key=lambda i: rows[i]['obliquity_deg'])
selected = [ordered[2], ordered[7], ordered[14], ordered[21], ordered[27], ordered[-2]]
figure, axes = plt.subplots(6, 3, figsize=(9, 16), constrained_layout=True)
for row_index, index in enumerate(selected):
    observed, true, guess, metadata = pictures[index]
    for col, image in enumerate((true, observed, guess)):
        axes[row_index, col].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[row_index, col].axis('off')
    axes[row_index, 0].set_title(f"true fitted atlas plane · {metadata['obliquity_deg']:.0f}°")
    axes[row_index, 1].set_title('actual v9 synthetic')
    axes[row_index, 2].set_title('275 first-choice atlas')
FIGURE.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE, dpi=140)
plt.close(figure)
receipt = {'candidate': 301, 'summary': summary,
    'output_sha256': {name: sha(RUN / name) for name in
        ('config.json', 'draws.jsonl', 'cases.jsonl', 'summary.json')},
    'figure_sha256': sha(FIGURE), 'figure_path': str(FIGURE),
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
