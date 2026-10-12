"""Six actual v9 steep-oblique sections with true and frozen-guessed atlas planes."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.run_smooth_image_fit_295 import rigid_render


SEED = 20261012303
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/STEEP_OBLIQUE_303_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt'
RUN = ROOT / 'reports/steep_oblique_303_001'
FIGURE = SOURCE.parent / 'docs/publication/figures/steep_oblique_303_examples.png'
assert SOURCE.drive.upper() == RUN.drive.upper() == FIGURE.drive.upper() == 'I:'
assert not RUN.exists() and not FIGURE.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(PARENT) == 'b4689e1d0dcbf1e8d65a63a400962bf27f44b2e571c8361e29e0e51412b2b97f'
context = load_streaming_synthetic_v7_64('cuda')
saved = torch.load(PARENT, map_location='cpu', weights_only=False)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
del saved
config = {'candidate': 303, 'seed': SEED, 'bases': [48, 53],
    'minimum_postartifact_obliquity_deg': 45,
    'parent_checkpoint_sha256': sha(PARENT), 'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'audit_steep_oblique_303.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_one_shot_model.py',
        'run_smooth_image_fit_295.py')},
    'synthetic_provenance': context['provenance'],
    'angle_stratified_not_distributional': True,
    'external_weights_features_or_pseudolabels': False,
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
rng = np.random.default_rng(SEED)
rows, images, attempted = [], [], 0
with (RUN / 'draws.jsonl').open('w', encoding='utf8') as draws, \
        (RUN / 'cases.jsonl').open('w', encoding='utf8') as output:
    with torch.inference_mode():
        for base in range(48, 54):
            while True:
                attempted += 1
                variant = int(rng.integers(VARIANTS))
                seed = SEED + 3_000_000_000 + attempted
                subject = base * VARIANTS + variant
                original = sample_one_shot_slide_artifacts_v9(context, [subject], seed, 192)
                identity = original['provenance'][0]
                eligible = bool(original['eligible'][0])
                fitted_angle = None
                if eligible:
                    sample = sample_postartifact_gauge_v1(
                        context, [subject], seed, 192, source_sample=original)
                    normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                    fitted_angle = float(torch.rad2deg(torch.acos(
                        normal.abs().max().clamp(0, 1))))
                accepted = eligible and fitted_angle >= 45
                draws.write(json.dumps({'attempt': attempted, 'base': base, 'variant': variant,
                    'seed': seed, 'eligible': eligible, 'fitted_obliquity_deg': fitted_angle,
                    'accepted': accepted, 'physical_section_id': identity['physical_section_id'],
                    'provenance': identity}, allow_nan=False) + '\n')
                if accepted:
                    break
                del original
                if eligible:
                    del sample
            true_render, _, _, _ = rigid_render(context['atlas'], sample['state'],
                sample['reflection'], sample['offsets'], sample['weights'])
            prediction = model.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            first = int(prior.argmax())
            guess, rigid, _, _ = rigid_render(context['atlas'],
                prediction['state'][0, first // 2][None],
                torch.tensor([first % 2], device='cuda'),
                sample['offsets'], sample['weights'])
            intact = sample['valid_mask'][0] & ~sample['fragment_displaced_mask'][0]
            error_mm = float(torch.linalg.vector_norm(
                rigid[0][intact] - sample['centre'][0][intact], dim=-1).mean() / 1000)
            virtual = np.asarray(identity['virtual_unit_normal'])
            virtual_angle = float(np.degrees(np.arccos(np.abs(virtual).max())))
            row = {'physical_section_id': identity['physical_section_id'],
                'base': base, 'variant': variant, 'seed': seed,
                'postartifact_obliquity_deg': fitted_angle,
                'original_cut_obliquity_deg': virtual_angle,
                'first_choice_rigid_site_error_mm': error_mm,
                'brush_available': bool(sample['inputs'][0, 2, 0, 0]),
                'valid_intact_pixels': int(intact.sum()),
                'provenance': sample['provenance'][0]}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            images.append((true_render[0, 0] / true_render[0, 1].clamp_min(1e-4),
                sample['inputs'][0, 0], guess[0, 0] / guess[0, 1].clamp_min(1e-4)))
            print(json.dumps({'event': 'accepted_base', 'base': base,
                              'attempts': attempted, 'obliquity_deg': fitted_angle}), flush=True)
            del original, sample

summary = {'cases': len(rows), 'attempts': attempted,
    'postartifact_obliquity_range_deg': [min(row['postartifact_obliquity_deg'] for row in rows),
        max(row['postartifact_obliquity_deg'] for row in rows)],
    'original_cut_obliquity_range_deg': [min(row['original_cut_obliquity_deg'] for row in rows),
        max(row['original_cut_obliquity_deg'] for row in rows)],
    'mean_first_choice_rigid_site_error_mm': float(np.mean(
        [row['first_choice_rigid_site_error_mm'] for row in rows]))}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
figure, axes = plt.subplots(6, 3, figsize=(9, 16), constrained_layout=True)
for index, (row, triplet) in enumerate(zip(rows, images)):
    for column, picture in enumerate(triplet):
        axes[index, column].imshow(picture.cpu(), cmap='gray', vmin=0, vmax=1)
        axes[index, column].axis('off')
    axes[index, 0].set_title(f"true fitted atlas · {row['postartifact_obliquity_deg']:.0f}°")
    axes[index, 1].set_title('actual v9 synthetic')
    axes[index, 2].set_title(f"275 first choice · {row['first_choice_rigid_site_error_mm']:.1f} mm")
FIGURE.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE, dpi=140)
plt.close(figure)
receipt = {'candidate': 303, 'summary': summary,
    'output_sha256': {name: sha(RUN / name) for name in
        ('config.json', 'draws.jsonl', 'cases.jsonl', 'summary.json')},
    'figure_path': str(FIGURE), 'figure_sha256': sha(FIGURE),
    'dev_cal_final_or_public_benchmark_used': False, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
