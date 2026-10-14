"""Four prechosen independent generator cuts: clean source, damaged input and atlas poses."""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_red_source_mix_311 import load_red_source_mix_311, sample_red_source_mix_311
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
context = load_red_source_mix_311('cuda')
model, _ = load_one_shot_checkpoint(ROOT / 'runs/first_choice_exposure_309_001/single/pose_local_20000.pt')
draws = ((3584, 20261015311), (3659, 20261015312),
         (3734, 20261015313), (3809, 20261015318))
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
fig, axes = plt.subplots(4, 4, figsize=(13, 13), layout='constrained')
rows = []

with torch.inference_mode():
    for row, (subject, seed) in enumerate(draws):
        sample = sample_red_source_mix_311(context, [subject], seed)
        prediction = model.predict(sample['inputs'])
        score = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        first = int(score.argmax())
        chosen = prediction['state'][0, first // 2]
        chosen_reflection = torch.tensor([first % 2], device='cuda')
        true = sample['state'][0]
        true_reflection = sample['reflection'][:1]
        true_normal = full_frame_state_to_components(true[None])[1][0, :, 2]
        chosen_normal = full_frame_state_to_components(chosen[None])[1][0, :, 2]
        angle = float(torch.rad2deg(torch.acos((true_normal * chosen_normal).sum().abs().clamp(0, 1))))
        views = []
        for state, reflection in ((true, true_reflection), (true, true_reflection),
                                  (chosen, chosen_reflection)):
            plane = rigid_points_090(state[None], reflection, chart, (192, 192)).reshape(1, 192, 192, 3)
            normal = full_frame_state_to_components(state[None])[1][0, :, 2]
            slab = plane[:, None] + sample['offsets'][:, :, None, None, None] * normal[None, None, None, None]
            if len(views) == 1:
                context['atlas'][0].copy_(context['source_intensities'][0])
            rendered = render_finite_thickness_coordinate_grid(
                context['atlas'], slab, (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            visual = (rendered[0] / rendered[1].clamp_min(1e-4)).clamp(0, 1)
            views.append(torch.where(rendered[1] < .05, 0, visual).cpu().numpy())
        valid = sample['valid_mask'][0].flatten()
        rigid = rigid_points_090(chosen[None], chosen_reflection, chart[valid], (192, 192))[0]
        truth = sample['centre'][0].reshape(-1, 3)[valid]
        error = float((rigid - truth).norm(dim=-1).mean() / 1000)
        input_image = sample['inputs'][0, 0].cpu().numpy()
        for column, image in enumerate((views[0], input_image, views[1], views[2])):
            axes[row, column].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row, column].axis('off')
        identity = sample['provenance'][0]['image_source']
        obliquity = math.degrees(math.acos(float(true_normal.abs().max().clamp(0, 1))))
        axes[row, 1].text(.02, .98, f"{identity['kind']}\n{obliquity:.0f}° from cardinal",
                          transform=axes[row, 1].transAxes, va='top', color='white', fontsize=8,
                          bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
        axes[row, 3].text(.02, .98, f'{error:.2f} mm · {angle:.0f}°',
                          transform=axes[row, 3].transAxes, va='top', color='white', fontsize=8,
                          bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
        rows.append({'subject': subject, 'seed': seed, 'source': identity,
                     'physical_section_id': sample['provenance'][0]['physical_section_id'],
                     'eligible': bool(sample['eligible'][0]), 'obliquity_deg': obliquity,
                     'first_rigid_survivor_error_mm': error, 'first_normal_error_deg': angle})

for ax, title in zip(axes[0], ('Clean source at truth', 'Actual generated input',
                              'Atlas at truth', 'Atlas at model first choice')):
    ax.set_title(title)
fig.suptitle('Four independent full-sphere TRAIN synthetic cuts; 309 pose-only checkpoint\n'
             'Clean views omit local warps, damage and acquired-image appearance')
output = ROOT / 'reports/red_source_mix_311_four_case_qc.png'
fig.savefig(output, dpi=150)
plt.close(fig)
(ROOT / 'reports/red_source_mix_311_four_case_qc.json').write_text(
    json.dumps(rows, indent=2) + '\n', encoding='utf8')
print(json.dumps({'figure': str(output), 'rows': rows}, indent=2))
