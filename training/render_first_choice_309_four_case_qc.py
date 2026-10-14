"""Four distinct frozen 309 TRAIN-inner cuts: generated image and rigid atlas planes."""
import hashlib
import json
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

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/first_choice_exposure_309_001'
panel_path = run / 'panels.pt'
checkpoint = run / 'single/pose_local_20000.pt'
receipt = json.loads((run / 'completed.json').read_text())
arm_receipt = json.loads((run / 'single/completed.json').read_text())
assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == receipt['output_sha256']['panels.pt']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == arm_receipt['output_sha256'][checkpoint.name]
panel = torch.load(panel_path, map_location='cpu', weights_only=True)['v10']
slots = (19, 5, 22, 176)  # Cardinal brush; moderate black; steep gray; sparse steep brush.
assert len({panel[i]['meta']['physical_section_id'] for i in slots}) == 4
assert all(panel[i]['meta']['base'] in range(56, 64) for i in slots)
context = load_fov_balance_v10('cuda')
model, _ = load_one_shot_checkpoint(checkpoint)
axis = torch.arange(192, device='cuda') / 192
y, x = torch.meshgrid(axis, axis, indexing='ij')
grid = torch.stack((x, y), -1).reshape(-1, 2)
figure, axes = plt.subplots(4, 4, figsize=(14, 14), layout='constrained')
rows = []

with torch.inference_mode():
    for row, slot in enumerate(slots):
        item = panel[slot]
        meta = item['meta']
        subject = meta['base'] * VARIANTS + meta['variant']
        sample = sample_fov_balance_v10(context, [subject], meta['seed'], 192)
        assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
        assert torch.equal(sample['inputs'][0].cpu(), item['inputs'])
        assert torch.equal(sample['state'][0].cpu(), item['state'])
        sites = (item['chart'] * 192).long()
        indices = sites[:, 1] * 192 + sites[:, 0]
        assert torch.equal(sample['centre'][0].reshape(-1, 3)[indices.cuda()].cpu(), item['truth'])
        prediction = model.predict(item['inputs'][None].cuda())
        score = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        branches = torch.arange(2 * model.modes, device='cuda')
        states = prediction['state'][0].repeat_interleave(2, 0)
        proposed = rigid_points_090(states, branches % 2, item['chart'].cuda(), (192, 192))
        error = (proposed - item['truth'].cuda()).norm(dim=-1).mean(-1) / 1000
        first = int(score.argmax())
        top16 = score.topk(16).indices
        oracle = int(top16[error[top16].argmin()])
        true_normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        chosen_normals = full_frame_state_to_components(states[[first, oracle]])[1][:, :, 2]
        angles = torch.rad2deg(torch.acos((chosen_normals * true_normal).sum(-1).abs().clamp(0, 1)))
        views = []
        for state, reflection in ((sample['state'][0], sample['reflection'][0]),
                                  (states[first], branches[first] % 2),
                                  (states[oracle], branches[oracle] % 2)):
            plane = rigid_points_090(state[None], reflection[None], grid, (192, 192)).reshape(1, 192, 192, 3)
            normal = full_frame_state_to_components(state[None])[1][0, :, 2]
            slab = plane[:, None] + sample['offsets'][:, :, None, None, None] * normal[None, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(
                context['atlas'], slab, (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            visual = (rendered[0] / rendered[1].clamp_min(1e-4)).clamp(0, 1)
            views.append(torch.where(rendered[1] < .05, 0, visual).cpu().numpy())
        style = sample['provenance'][0]['one_shot_slide_artifacts_v9']['exterior_style']
        brush = bool(item['inputs'][2, 0, 0])
        images = (item['inputs'][0].numpy(), *views)
        for column, image in enumerate(images):
            axes[row, column].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row, column].axis('off')
        axes[row, 0].text(.02, .98,
            f"slot {slot} · base {meta['base']}\n{meta['obliquity_deg']:.0f}° from cardinal · "
            f"{meta['postartifact_visible_fraction']:.0%} visible\n"
            f"{'brush' if brush else 'raw'} · {style}",
            transform=axes[row, 0].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
        axes[row, 2].text(.02, .98, f'{float(error[first]):.2f} mm · {float(angles[0]):.0f}°',
            transform=axes[row, 2].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
        axes[row, 3].text(.02, .98, f'{float(error[oracle]):.2f} mm · {float(angles[1]):.0f}°',
            transform=axes[row, 3].transAxes, va='top', color='white', fontsize=9,
            bbox={'facecolor': 'black', 'alpha': .7, 'edgecolor': 'none'})
        rows.append({'slot': slot, 'physical_section_id': meta['physical_section_id'],
            'base': meta['base'], 'brush': brush, 'exterior_style': style,
            'obliquity_deg': meta['obliquity_deg'],
            'postartifact_visible_fraction': meta['postartifact_visible_fraction'],
            'first_choice_site_error_mm': float(error[first]),
            'first_choice_normal_angle_deg': float(angles[0]),
            'top16_oracle_site_error_mm': float(error[oracle]),
            'top16_oracle_normal_angle_deg': float(angles[1])})

for axis, title in zip(axes[0], ('Actual generated input', 'Unwarped atlas at true plane',
                                 'Unwarped atlas at first choice',
                                 'Best top-16 atlas plane\n(oracle; unavailable at inference)')):
    axis.set_title(title)
figure.suptitle('309 v10 · four distinct frozen TRAIN-inner synthetic cuts · exact panel replay\n'
                 'Atlas renders use the recorded finite-thickness PSF; no local warp or artifact replay\n'
                 'Annotations: surviving-site rigid error (mm) · unsigned normal error (degrees)')
output = ROOT / 'reports/visual_qc_309_four_distinct_train_inner.png'
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=160)
plt.close(figure)
print(json.dumps({'figure': str(output), 'rows': rows}, indent=2))
