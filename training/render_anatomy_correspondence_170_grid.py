"""Illustrative atlas/input/selected-plane grid from frozen 170 TRAIN cases."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR'] = str(root / 'cache')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher

run = root / 'runs/anatomy_correspondence_170_blind_train_002'
parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
out = root / 'reports/anatomy_correspondence_170_visual_qc_001'
assert not out.exists()
receipt = json.loads((run / 'completed.json').read_text())
assert all(hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
           for name, digest in receipt['output_sha256'].items())
assert hashlib.sha256(parent.read_bytes()).hexdigest() == json.loads(
    (parent.parent / 'completed.json').read_text())['checkpoint_sha256']['2000']
rows = {row['index']: row for line in (run / 'rows.jsonl').open()
        if (row := json.loads(line))['step'] == 6000 and row['scored']}
indices = (0, 16, 32, 34, 40, 48)
assert all(index in rows for index in indices)

context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)

side = 256
axis = torch.arange(side, device='cuda') / side
y, x = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((x, y), -1).reshape(-1, 2)
figure, axes = plt.subplots(len(indices), 5, figsize=(15, 17), facecolor='white')
metadata = []
torch.set_num_threads(4)
with torch.inference_mode():
    for row_index, index in enumerate(indices):
        row = rows[index]
        geometry = sample_panel_geometry_train_151 if index % 5 == 0 else sample_panel_geometry_train_166
        original = geometry(context, [row['base_index']], row['draw_seed'], side)
        sample = sample_one_shot_slide_artifacts_v4(context, [row['base_index']],
            row['draw_seed'], side, source_sample=original)
        assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
        prediction = model.predict(sample['inputs'])
        prior_slot = row['chosen_slot']['prior']
        anatomy_slot = row['chosen_slot']['anatomy1']
        candidates = ((sample['state'], int(sample['reflection'][0])),
            (prediction['state'][:, row['branch_ids'][prior_slot] // 2],
             row['branch_ids'][prior_slot] % 2),
            (prediction['state'][:, row['branch_ids'][anatomy_slot] // 2],
             row['branch_ids'][anatomy_slot] % 2))
        atlas_images = []
        for state, reflection in candidates:
            plane = rigid_points_090(state, torch.tensor([reflection], device='cuda'),
                chart, (side, side)).reshape(1, side, side, 3)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            coords = (plane[:, None] + sample['offsets'][:, :, None, None, None]
                * normal[:, None, None, None, :])
            slab = render_finite_thickness_coordinate_grid(atlas, coords,
                (0., 0., 0.), (25., 25., 25.), sample['weights'])[0]
            intensity = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            intensity[slab[1].cpu().numpy() < .05] = 0
            atlas_images.append(intensity)
        observed = sample['inputs'][0, 0].cpu().numpy()
        stretch = max(float(np.quantile(observed, .995)), .001)
        views = (atlas_images[0], observed, observed, atlas_images[1], atlas_images[2])
        uppers = (1., 1., stretch, 1., 1.)
        for column, (view, upper) in enumerate(zip(views, uppers)):
            axes[row_index, column].imshow(view, cmap='gray', vmin=0, vmax=upper)
            axes[row_index, column].axis('off')
        prior_mm = row['initial_rigid_mm'][prior_slot]
        anatomy_mm = row['initial_rigid_mm'][anatomy_slot]
        axes[row_index, 0].annotate(
            f"#{index}: {row['appearance_mode']}, {row['nearest_cardinal_angle_deg']:.0f}°\n"
            f'prior {prior_mm:.2f} mm, anatomy {anatomy_mm:.2f} mm',
            xy=(-.03, .5), xycoords='axes fraction', ha='right', va='center',
            fontsize=9, clip_on=False)
        metadata.append({'index': index, 'physical_section_id': row['physical_section_id'],
            'animal_id': row['animal_id'], 'appearance_mode': row['appearance_mode'],
            'angle_deg': row['nearest_cardinal_angle_deg'],
            'prior_branch_id': row['branch_ids'][prior_slot],
            'anatomy_branch_id': row['branch_ids'][anatomy_slot],
            'prior_rigid_mm': prior_mm, 'anatomy_rigid_mm': anatomy_mm,
            'oracle16_rigid_mm': min(row['initial_rigid_mm'])})

for column, title in enumerate(('True atlas plane', 'Artifacted input [0,1]',
                                'Same input, display-stretched', 'Direct prior plane',
                                'Anatomy-score plane')):
    axes[0, column].set_title(title, fontsize=10)
figure.text(.5, .005, 'Six illustrative independent TRAIN draws from frozen blind panel 170. '
    'True atlas plane is the synthetic CCF gauge, not intact deformed tissue. '
    'Errors are rigid-gauge, not electrode errors.', ha='center', fontsize=9)
figure.tight_layout(rect=(0, .02, 1, 1))
out.mkdir(parents=True, exist_ok=False)
figure.savefig(out / 'atlas_input_prior_anatomy.png', dpi=140)
plt.close(figure)
(out / 'selection.json').write_text(json.dumps({'rule': 'fixed illustrative indices; '
    'successes and failures both shown, not a prevalence sample',
    'evaluation_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
    'rows': metadata}, indent=2))
print(json.dumps({'image': str(out / 'atlas_input_prior_anatomy.png'),
    'metadata': str(out / 'selection.json'), 'rows': metadata}, indent=2))
