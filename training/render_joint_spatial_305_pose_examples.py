"""First six frozen TRAIN-inner sections: true atlas, actual input, direct pose guess."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
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
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

run = root / 'runs/joint_spatial_feedback_305_pose_002'
checkpoint = run / 'pose_step_26000.pt'
out = root / 'previews/joint_spatial_305_pose_26000_first6.png'
panel = torch.load(run / 'inner_panel.pt', map_location='cpu', weights_only=False)
receipt = json.loads((run / 'completed.json').read_text(encoding='utf8'))
assert receipt['output_sha256'][checkpoint.name] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
context = load_streaming_synthetic_v7_64('cuda')
model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
fig, axes = plt.subplots(6, 3, figsize=(9, 16), layout='constrained')
rows = []
with torch.inference_mode():
    for i, item in enumerate(panel[:6]):
        image = item['inputs'][None].cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        guessed_state = prediction['state'][:, branch // 2]
        guessed_reflection = torch.tensor([branch % 2], device='cuda')
        true_state = item['state'][None].cuda()
        true_reflection = item['reflection'][None].cuda()
        offsets = item['offsets'][None].cuda()
        weights = item['weights'][None].cuda()
        renders = []
        for state, reflection in ((true_state, true_reflection),
                                  (guessed_state, guessed_reflection)):
            centre, frame, basis = full_frame_state_to_components(state)
            chart = torch.stack((torch.where(reflection[:, None, None].bool(),
                191 / 192 - x, x).expand(1, -1, -1), y[None]), -1)
            plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                              basis[:, None, None], chart)
            slab = plane[:, None] + offsets[:, :, None, None, None] * frame[
                :, None, None, None, :, 2]
            rendered = render_finite_thickness_coordinate_grid(
                context['atlas'], slab, (0., 0., 0.), (25., 25., 25.), weights)[0]
            visual = (rendered[0] / rendered[1].clamp_min(1e-4)).cpu().numpy()
            visual[rendered[1].cpu().numpy() < .05] = 0
            renders.append(visual)
        chart = item['chart'][None].cuda()
        truth = item['truth'].cuda()
        guessed = rigid_points_090(guessed_state, guessed_reflection, chart, (192, 192))[0]
        error_mm = float((guessed - truth).norm(dim=-1).mean() / 1000)
        input_image = item['inputs'][0].numpy()
        p99 = max(float(np.quantile(input_image, .99)), .001)
        for j, visual in enumerate((renders[0], input_image / p99, renders[1])):
            axes[i, j].imshow(visual, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].text(.02, .98, f'{item["obliquity_deg"]:.0f}° oblique',
            transform=axes[i, 0].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
        axes[i, 1].text(.02, .98, f'{item["exterior_style"]}; p99={p99:.2f}',
            transform=axes[i, 1].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
        axes[i, 2].text(.02, .98, f'{error_mm:.2f} mm site error',
            transform=axes[i, 2].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
        rows.append({'physical_section_id': item['physical_section_id'],
            'obliquity_deg': item['obliquity_deg'],
            'exterior_style': item['exterior_style'],
            'damage_fraction': item['damage_fraction'],
            'visible_fraction': item['visible_fraction'],
            'input_p99': p99, 'direct_rigid_site_error_mm': error_mm})
for ax, title in zip(axes[0], ('Atlas at true plane', 'Actual generated input\n(display stretched)',
                                'Atlas at direct guessed plane')):
    ax.set_title(title)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
(out.with_suffix('.json')).write_text(json.dumps({
    'scope': 'first six frozen TRAIN-inner synthetic cases; no performance-based selection; pose stage only',
    'pose_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
    'panel_sha256': receipt['output_sha256']['inner_panel.pt'],
    'checkpoint_sha256': receipt['output_sha256'][checkpoint.name],
    'figure_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'rows': rows}, indent=2), encoding='utf8')
print(json.dumps({'image': str(out), 'rows': rows}))
