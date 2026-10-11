"""Six fresh, independent sections: true plane, generated image, blind 188 pose."""

import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher

torch.set_num_threads(4)
variant = os.environ.get('ANATOMY_SYNTHETIC_VARIANT', 'v7')
sampler = sample_one_shot_slide_artifacts_v7 if variant != 'v4' else sample_one_shot_slide_artifacts_v4
rng = np.random.default_rng({'v4': 20261011194, 'v7': 20261011201,
                             'mouselight_v7': 20261011203}[variant])
context = load_streaming_synthetic_v7_64(device='cuda')
reference_atlas = context['atlas']
if variant == 'mouselight_v7':
    tissue = np.load(root / 'data/janelia_mouselight_198/2017-01-15_appearance_ccf25.npy')
    context = {**context, 'atlas': torch.stack((torch.from_numpy(tissue.astype(np.float32)).cuda(),
        reference_atlas[1])), 'appearance_source': 'MouseLight 2017-01-15 registered; lateral sign unresolved'}
samples = []
while len(samples) < 6:
    virtual = int(rng.integers(len(context['subjects'])))
    seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    with torch.inference_mode():
        sample = sampler(context, [virtual], seed, side=256)
    if bool(sample['eligible'][0]):
        samples.append(sample)

checkpoint = root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt'
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)

axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
figure, axes = plt.subplots(6, 3, figsize=(11, 18), constrained_layout=True)
rows = []
with torch.inference_mode():
    for i, sample in enumerate(samples):
        prediction = model.predict(sample['inputs'])
        score = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(score[0].argmax())
        guessed = prediction['state'][:, branch // 2]
        renders = []
        for state, reflection in ((sample['state'], sample['reflection']),
                                  (guessed, torch.tensor([branch % 2], device='cuda'))):
            plane = rigid_points_090(state, reflection, chart).reshape(1, 256, 256, 3)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            world = plane[:, None] + sample['offsets'][:, :, None, None, None] * normal[:, None, None, None]
            slab = render_finite_thickness_coordinate_grid(reference_atlas, world,
                (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            rendered = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            rendered[slab[1].cpu().numpy() < .05] = 0
            renders.append(rendered)
        valid = sample['valid_mask'][0].flatten()
        truth_points = rigid_points_090(sample['state'], sample['reflection'], chart[valid])[0]
        guessed_points = rigid_points_090(guessed,
            torch.tensor([branch % 2], device='cuda'), chart[valid])[0]
        error_mm = float((truth_points - guessed_points).norm(dim=-1).mean() / 1000)
        record = sample['provenance'][0]
        normal = np.asarray(record['virtual_unit_normal'])
        angle = float(np.degrees(np.arccos(np.clip(np.abs(normal).max(), 0, 1))))
        flags = [name for name, enabled in record['one_shot_slide_artifacts_v3']['events'].items() if enabled]
        images = (renders[0], sample['inputs'][0, 0].cpu().numpy(), renders[1])
        for j, image in enumerate(images):
            axes[i, j].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'{angle:.0f}° from cardinal\n{record["mode"]}\n{", ".join(flags) or "no added artifact"}', fontsize=9)
        axes[i, 2].set_xlabel(f'rigid error {error_mm:.2f} mm', fontsize=9)
        rows.append({'physical_section_id': record['physical_section_id'],
            'virtual_subject': record['virtual_subject_id'], 'angle_from_cardinal_deg': angle,
            'appearance_mode': record['mode'], 'artifact_flags': flags,
            'first_choice_branch': branch, 'rigid_error_mm': error_mm})

for axis, title in zip(axes[0], ('Atlas at true plane', 'Actual generated input',
                                  'Atlas at blind first-choice plane')):
    axis.set_title(title)
out = root / f'previews/current_{variant}_true_input_guessed_194.png'
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=135)
plt.close(figure)
(out.with_suffix('.json')).write_text(json.dumps({'checkpoint': str(checkpoint),
    'selection': f'first six eligible independent random {variant} draws, no error-based selection',
    'rows': rows}, indent=2))
print(json.dumps({'image': str(out), 'rows': rows}, indent=2))
