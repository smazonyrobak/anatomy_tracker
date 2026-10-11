"""Four fixed TRAIN-native virtual planes; weak atlas reference and unchanged parent guess."""

import json
import hashlib
import os
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.sample_allen_stpt_native_211 import load_allen_stpt_native_211, sample_allen_stpt_native_211


output = root / 'reports/allen_stpt_native_211_train_001'
output.mkdir(parents=True, exist_ok=True)
donor_dir = root / 'data/allen_stpt_native_208_train_001/donor_14383'
map_path = root / 'data/allen_stpt_native_map_209_train_001/100140756_native_to_ccf_ap_dv_ml_um.npy'
parent = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt'
torch.set_num_threads(4)
with torch.no_grad():
    context = load_allen_stpt_native_211(donor_dir, map_path)
    source = sample_allen_stpt_native_211(context, [14383] * 4, 311, side=256)
    sample = sample_one_shot_slide_artifacts_v3({}, [14383] * 4, 311,
                                                 side=256, source_sample=source)
    atlas_array, _ = allen._decode_and_preprocess_allen_v6()
    atlas = torch.from_numpy(atlas_array).cuda()
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()
    attach_global_plane_matcher(model, enabled=True)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    prediction = model.predict(sample['inputs'])
    scores = prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)
    branch = scores.flatten(1).argmax(-1)
    guess = prediction['state'][torch.arange(4, device='cuda'), branch // 2]
    guessed_reflection = branch % 2
    axis = (torch.arange(128, device='cuda') + .5) / 128 - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    panels = []
    for state, reflection in ((source['state'], source['reflection']),
                              (guess, guessed_reflection)):
        plane = rigid_points_090(state, reflection, chart).reshape(4, 128, 128, 3)
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        slab = plane[:, None] + source['offsets'][:, :, None, None, None] * normal[:, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, slab, (0., 0., 0.), (25., 25., 25.), source['weights'])
        panels.append((rendered[:, 0] / rendered[:, 1].clamp_min(1e-4)).cpu().numpy())
    five = torch.tensor([[0., 0.], [255 / 256, 0.], [0., 255 / 256],
                         [255 / 256, 255 / 256], [127.5 / 256, 127.5 / 256]], device='cuda')
    error = (rigid_points_090(guess, guessed_reflection, five) -
             rigid_points_090(source['state'], source['reflection'], five)).norm(dim=-1).mean(-1) / 1000
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for row in range(4):
        for column, image in enumerate((panels[0][row], source['inputs'][row, 0].cpu().numpy(),
                                        sample['inputs'][row, 0].cpu().numpy(), panels[1][row])):
            axes[row, column].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[row, column].axis('off')
        axes[row, 0].set_ylabel(f'virtual plane {row + 1}', fontsize=11)
        axes[row, 3].set_title(f'parent error {error[row]:.2f} mm', fontsize=10)
    for column, title in enumerate(('Weak Allen atlas plane', 'Native virtual cut',
                                    'Cut after sampled artifacts', 'Unchanged parent guess')):
        axes[0, column].text(.5, 1.16, title, transform=axes[0, column].transAxes,
                             ha='center', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'four_planes_component_support.png', dpi=150)
    plt.close(fig)
    (output / 'readout_component_support.json').write_text(json.dumps({
        'donor_id': 14383, 'seed': 311, 'all_four_draws_shown_no_selection': True,
        'source_eligible': source['eligible'].tolist(), 'artifact_eligible': sample['eligible'].tolist(),
        'parent_five_point_error_mm_to_weak_Allen_reference': error.tolist(),
        'parent_checkpoint': str(parent), 'source_provenance': sample['provenance'],
        'sampler_sha256': hashlib.sha256(Path(sample_allen_stpt_native_211.__code__.co_filename).read_bytes()).hexdigest(),
        'note': 'Virtual angles interpolate 100um-spaced physical coronal images; this is not physical-oblique validation.'
    }, indent=2) + '\n')
    print(json.dumps({'output': str(output / 'four_planes_component_support.png'), 'parent_error_mm': error.tolist()}))
