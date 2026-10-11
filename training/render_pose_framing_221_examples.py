"""Six fresh TRAIN generator sections: atlas, pre-artifact, final input, blind pose."""

import hashlib
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
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
checkpoint = root / 'runs/pose_framing_221_framed_train_001/joint_step_00400.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(saved['model'], strict=True)
del saved

rng = np.random.default_rng(20261011223)
samples, attempts = [], []
while len(samples) < 6:
    base = int(rng.integers(64))
    seed = int(rng.integers(2**63 - 1))
    appearance = 'v4' if rng.integers(2) else 'v3'
    sampler = sample_one_shot_slide_artifacts_v4 if appearance == 'v4' else sample_one_shot_slide_artifacts_v3
    with torch.inference_mode():
        source = sample_panel_geometry_train_166(context, [base], seed, side=256)
        source = sample_support_aware_framing_217(source, seed)
        sample = sampler(context, [base], seed, side=256, source_sample=source)
    attempts.append({'base': base, 'seed': seed, 'appearance': appearance,
                     'physical_section_id': sample['provenance'][0]['physical_section_id'],
                     'eligible': bool(sample['eligible'][0])})
    if bool(sample['eligible'][0]):
        samples.append((source, sample, base, seed, appearance))

axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
figure, axes = plt.subplots(6, 4, figsize=(13, 18), constrained_layout=True)
rows = []
with torch.inference_mode():
    for i, (source, sample, base, seed, appearance) in enumerate(samples):
        prediction = model.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        states = (sample['state'], prediction['state'][:, branch // 2])
        reflections = (sample['reflection'], torch.tensor([branch % 2], device='cuda'))
        renders = []
        for state, reflection in zip(states, reflections):
            plane = rigid_points_090(state, reflection, chart).reshape(1, 256, 256, 3)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            slab = plane[:, None] + sample['offsets'][:, :, None, None, None] * normal[:, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(context['atlas'], slab,
                (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            image = (rendered[0] / rendered[1].clamp_min(1e-4)).cpu().numpy()
            image[rendered[1].cpu().numpy() < .05] = 0
            renders.append(image)
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        sites = chart[valid]
        predicted = rigid_points_090(states[1], reflections[1], sites)[0]
        dense = sample['centre'][0].reshape(-1, 3)[valid]
        error_mm = float((predicted - dense).norm(dim=-1).mean() / 1000)
        truth_normal = full_frame_state_to_components(states[0])[1][0, :, 2]
        cardinal = float(torch.rad2deg(torch.acos(truth_normal.abs().max().clamp_max(1))))
        flags = [name for name, enabled in sample['provenance'][0][
            'one_shot_slide_artifacts_v3']['events'].items() if enabled]
        images = (renders[0], source['inputs'][0, 0].cpu().numpy(),
                  sample['inputs'][0, 0].cpu().numpy(), renders[1])
        for j, image in enumerate(images):
            axes[i, j].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'{cardinal:.0f}° from cardinal\n{appearance}, {sample["provenance"][0]["mode"]}\n'
                              f'{", ".join(flags) or "no extra artifact"}', fontsize=9)
        axes[i, 3].set_xlabel(f'blind error {error_mm:.2f} mm', fontsize=9)
        rows.append({'physical_section_id': sample['provenance'][0]['physical_section_id'],
            'base': base, 'seed': seed, 'appearance': appearance, 'mode': sample['provenance'][0]['mode'],
            'angle_from_cardinal_deg': cardinal, 'artifact_flags': flags,
            'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
            'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
            'first_choice_branch': branch, 'blind_dense_error_mm': error_mm})

for axis, title in zip(axes[0], ('Atlas at known plane', 'Before extra artifacts',
                                  'Final model input', 'Atlas at blind guess')):
    axis.set_title(title)
out = root / 'previews/pose_framing_221_fresh_generator_examples.png'
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=130)
plt.close(figure)
with checkpoint.open('rb') as stream:
    checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
with out.with_suffix('.json').open('w') as stream:
    json.dump({'checkpoint_sha256': checkpoint_sha256,
        'selection': 'first six eligible independent fresh TRAIN draws; no error-based selection',
        'attempts': attempts, 'rows': rows}, stream, indent=2)
print(json.dumps({'image': str(out), 'attempts': len(attempts), 'rows': rows}), flush=True)
