"""One auditable TRAIN-generator grid against the frozen parent-132 first choice."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
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
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


source = Path(__file__).resolve().parent
parent = root / 'runs/v4_pose_adaptation_132'
checkpoint = parent / 'joint_step_02000.pt'
out = root / 'reports/train_synthetic_132_visual_grid_165'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed, side, max_attempts = 20261011165, 256, 256
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)

source_names = (
    'render_train_132_synthetic_grid_165.py', 'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py', 'global_atlas_contrast_090.py',
    'global_plane_matcher_120.py')
hashes = {}
for name in source_names:
    with (source / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
for name in ('config.json', 'completed.json', 'joint_step_02000.pt'):
    with (parent / name).open('rb') as stream:
        hashes[f'parent/{name}'] = hashlib.file_digest(stream, 'sha256').hexdigest()
config = json.loads((parent / 'config.json').read_text())
completed = json.loads((parent / 'completed.json').read_text())
assert hashes['parent/joint_step_02000.pt'] == completed['checkpoint_sha256']['2000']
assert hashes['parent/config.json'] == completed['config_sha256']
assert hashes['arbitrary_plane_one_shot_model.py'] == config['source_sha256'][
    'arbitrary_plane_one_shot_model.py']
assert hashes['arbitrary_plane_one_shot_slide_artifacts_v4.py'] == config['source_sha256'][
    'arbitrary_plane_one_shot_slide_artifacts_v4.py']

context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
atlas = context['atlas']
slots = ['sparse_tangential', 'moderate_steep_oblique',
         'substantial_1', 'substantial_2', 'substantial_3', 'substantial_4']
selected, attempts, used_bases = {}, [], set()

# Selection uses only the generated TRAIN section's tissue fraction and normal angle.
# The frozen model is not loaded until all six distinct-base physical sections are locked.
with torch.inference_mode():
    for attempt in range(max_attempts):
        base = int(rng.integers(60))
        draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
        clean_source = sample_panel_geometry_train_151(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(
            context, [base], draw_seed, side, source_sample=clean_source)
        provenance = sample['provenance'][0]
        assert provenance['split'] == provenance['base_lineage']['split'] == 'train'
        valid_fraction = float(sample['valid_mask'][0].float().mean())
        clean_tissue_fraction = float(clean_source['support'][0].gt(.5).float().mean())
        normal = np.asarray(provenance['physical_unit_normal_ap_dv_ml'])
        angle = float(np.degrees(np.arccos(np.clip(np.abs(normal).max(), 0, 1))))
        eligible = bool(sample['eligible'][0])
        slot = None
        if eligible and base not in used_bases:
            if ('sparse_tangential' not in selected and .025 <= valid_fraction < .08
                    and angle >= 30):
                slot = 'sparse_tangential'
            elif ('moderate_steep_oblique' not in selected and .08 <= valid_fraction < .20
                  and angle >= 35):
                slot = 'moderate_steep_oblique'
            elif valid_fraction >= .20:
                slot = next((s for s in slots[2:] if s not in selected), None)
        attempt_record = {
            'attempt': attempt, 'base_index': base, 'draw_seed': draw_seed,
            'physical_section_id': provenance['physical_section_id'],
            'inverse_map_sha256': provenance['inverse_map_sha256'],
            'eligible': eligible, 'valid_tissue_fraction': valid_fraction,
            'clean_support_fraction': clean_tissue_fraction,
            'nearest_cardinal_angle_deg': angle, 'selected_slot': slot,
            'skip_reason': None if slot else ('ineligible' if not eligible else
                          'base_already_selected' if base in used_bases else 'stratum_full_or_unmatched'),
        }
        attempts.append(attempt_record)
        if slot:
            selected[slot] = {'sample': sample, 'provenance': provenance,
                              'attempt_record': attempt_record}
            used_bases.add(base)
            print(json.dumps({'selected': slot, 'attempt': attempt,
                              'valid_fraction': valid_fraction, 'angle': angle}), flush=True)
        if len(selected) == 6:
            break
assert len(selected) == 6 and len(used_bases) == 6
assert len({a['physical_section_id'] for a in attempts}) == len(attempts)

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

axis = torch.arange(side, device='cuda') / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
panels, rows = [], []
with torch.inference_mode():
    for slot in slots:
        item = selected[slot]
        sample = item['sample']
        prediction = model.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior.argmax(-1)[0])
        predicted_state = prediction['state'][:, branch // 2]
        predicted_reflection = torch.tensor([branch % 2], device='cuda')
        valid = sample['valid_mask'][0]
        observed = chart[valid.flatten()]
        true_points = rigid_points_090(sample['state'], sample['reflection'], observed)[0]
        predicted_points = rigid_points_090(predicted_state, predicted_reflection, observed)[0]
        rigid_mm = float((predicted_points - true_points).norm(dim=-1).mean() / 1000)
        renders = []
        for state, reflection in ((sample['state'], sample['reflection']),
                                  (predicted_state, predicted_reflection)):
            plane = rigid_points_090(state, reflection, chart).reshape(1, side, side, 3)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            world = plane[:, None] + sample['offsets'][:, :, None, None, None] * normal[:, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(
                atlas, world, (0., 0., 0.), (25., 25., 25.), sample['weights'])
            renders.append(rendered[0, 0].clamp(0, 1).cpu().numpy())
        generated_input = sample['inputs'][0, 0].clamp(0, 1).cpu().numpy()
        panels.append((renders[0], generated_input, renders[1]))
        provenance = item['provenance']
        artifact = provenance['one_shot_slide_artifacts_v3']
        exposure = provenance['one_shot_slide_artifacts_v4']
        flags = [name for name, enabled in artifact['events'].items() if enabled]
        if provenance['appearance']['damage']:
            flags.append('damage')
        if exposure['low_exposure']:
            flags.append('low_exposure')
        rows.append({**item['attempt_record'], 'slot': slot,
            'direct_first_choice_branch_id_0_to_159': branch,
            'branch_prior_log_score_uncalibrated': float(prior[0, branch]),
            'true_rigid_error_mm': 0., 'predicted_rigid_error_mm': rigid_mm,
            'artifact_flags': flags, 'appearance_mode': provenance['mode'],
            'true_state': sample['state'][0].cpu().tolist(),
            'true_reflection': int(sample['reflection'][0]),
            'predicted_state': predicted_state[0].cpu().tolist(),
            'predicted_reflection': branch % 2,
            'provenance': provenance})
        print(json.dumps({'rendered': slot, 'rigid_mm': rigid_mm,
                          'branch': branch}), flush=True)

fig, axes = plt.subplots(6, 3, figsize=(14.5, 23.5))
fig.subplots_adjust(left=.235, right=.99, top=.972, bottom=.025, hspace=.29, wspace=.045)
for row, (slot, images, result) in enumerate(zip(slots, panels, rows)):
    for col, image in enumerate(images):
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    axes[row, 0].set_title('clean atlas @ truth rigid gauge\nerror 0.00 mm', fontsize=10)
    axes[row, 1].set_title('generated TRAIN input\nafter artifacts / background', fontsize=10)
    axes[row, 2].set_title(f'direct first-choice atlas\nrigid error {result["predicted_rigid_error_mm"]:.2f} mm', fontsize=10)
    flags = ', '.join(result['artifact_flags']) or 'none'
    fig.text(.015, .972 - (row + .5) * (.947 / 6),
             f'{row + 1}. {slot}\nattempt {result["attempt"]}; base {result["base_index"]}\n'
             f'valid tissue {result["valid_tissue_fraction"]:.1%}\n'
             f'nearest cardinal {result["nearest_cardinal_angle_deg"]:.1f}°\n'
             f'mode {result["appearance_mode"]}\nflags: {flags}',
             ha='left', va='center', fontsize=9)
fig.suptitle('Six independently drawn TRAIN physical sections; frozen parent 132 step 2000', fontsize=15)

manifest = {
    'scope': 'TRAIN synthetic only; six distinct independent deformation bases and physical section IDs. No DEV, holdout, real specimen, model training, error-based selection, or checkpoint modification.',
    'generator': 'v7_64 independent TRAIN bases + 151 full-box physical geometry + v3/v4 one-shot artifacts and exposure',
    'selection_locked_before_checkpoint_load': True,
    'seed': seed, 'side': side, 'max_attempts': max_attempts,
    'selection_rule': {'sparse_tangential': 'eligible, 2.5-8% valid pixels, nearest-cardinal angle >=30 degrees',
                       'moderate_steep_oblique': 'eligible, 8-20% valid pixels, angle >=35 degrees',
                       'substantial_1_to_4': 'eligible, >=20% valid pixels',
                       'all': 'First matching draw per still-open stratum; selected bases unique; no model inference before selection.'},
    'attempts': attempts, 'selected_rows': rows,
    'source_sha256': hashes,
    'atlas_binding': context['provenance']['atlas_binding'],
    'parent_checkpoint': str(checkpoint),
    'first_choice': 'argmax over 80 pose-mode log masses plus binary reflection logits (160 branches); no atlas reranking or truth selection',
    'rigid_error': 'Mean Euclidean displacement in CCF micrometres of true vs predicted rigid maps over generated valid native-256 pixels, divided by 1000 to mm; truth error is 0 by definition.',
    'clean_atlas_panel': 'Finite-thickness atlas render at the generator training-target fitted affine CCF gauge, with the sampled 9-point PSF and reflection. It is not the pristine nonrigid virtual specimen cut.',
    'predicted_atlas_panel': 'Same finite-thickness atlas render at the direct first-choice branch, same PSF and observed chart.',
    'display': 'All three panels grayscale linearly windowed [0,1], no per-panel normalization or contrast enhancement.',
}
out.mkdir(parents=True, exist_ok=False)
fig.savefig(out / 'six_train_sections_grid.png', dpi=145)
plt.close(fig)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps({'output': str(out), 'attempts': len(attempts),
                  'selected': [(r['slot'], r['attempt'], r['predicted_rigid_error_mm']) for r in rows]}),
      flush=True)
