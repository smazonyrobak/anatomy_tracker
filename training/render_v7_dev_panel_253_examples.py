"""Four actual frozen DEV sections: true atlas plane, generated input, fitted 247 guess."""
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

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, normalized_raster_to_ccf,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_v7_heldout_dev_panel_248_004'
cache = root / 'data/allen_v6_float32_cache_248_001'
checkpoint = root / 'runs/anchor_only_blind_feedback_247_train_001/joint_step_06000.pt'
readout = root / 'runs/one_shot_v7_heldout_dev_panel_248_readout_001'
out = root / 'previews/v7_dev_253_true_input_joint247_fit.png'
rows = [json.loads(line) for line in (panel / 'records.jsonl').read_text().splitlines()]
reference_rows = {row['physical_section_id']: row for row in
                  (json.loads(line) for line in (readout / 'rows.jsonl').read_text().splitlines())}
selection = [('raw', 0, 15, 'fold'), ('raw', 45, 60, 'fragment'),
             ('exact_black', 30, 45, 'bubble'), ('imperfect_brush', 30, 45, 'tile_seam')]
chosen, used_subjects = [], set()
for mode, low, high, event in selection:
    for row in rows:
        n = np.asarray(row['provenance']['virtual_unit_normal'])
        angle = np.degrees(np.arccos(np.clip(np.abs(n).max(), 0, 1)))
        if (row['eligible'] and row['appearance_mode'] == mode and low <= angle < high
                and row['provenance']['one_shot_slide_artifacts_v3']['events'][event]
                and row['animal_id'] not in used_subjects):
            chosen.append((row, angle, event))
            used_subjects.add(row['animal_id'])
            break
assert len(chosen) == 4

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(np.load(cache / 'atlas_float32.npy', allow_pickle=False)).cuda()
model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=True,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 6000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
axis = torch.arange(192, device='cuda') / 192
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
figure, axes = plt.subplots(4, 3, figsize=(10, 12), layout='constrained')
result = []
with torch.inference_mode():
    for i, (row, cut_angle, event) in enumerate(chosen):
        with np.load(panel / row['file'], allow_pickle=False) as a:
            image = torch.from_numpy(a['inputs'][None].copy()).cuda()
            true_state = torch.from_numpy(a['target_state'][None].copy()).cuda()
            true_reflection = torch.tensor([int(a['reflection'])], device='cuda')
            offsets = torch.from_numpy(a['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(a['weights'][None].copy()).cuda()
            valid = torch.from_numpy(a['valid_mask'].copy()).cuda()
            target = torch.from_numpy(a['target_centre_um'].copy()).cuda()
        prediction = model.predict(image)
        modes = prediction['log_mass'][0].topk(4).indices
        mode_indices = modes.repeat_interleave(2)
        reflections = torch.tensor([[0, 1] * 4], device='cuda')
        selected = {**prediction, 'state': prediction['state'][:, mode_indices],
                    'log_mass': prediction['log_mass'][:, mode_indices],
                    'reflection_logit': prediction['reflection_logit'][:, mode_indices]}
        branch = torch.arange(8, device='cuda')[None]
        first = model.map(selected, offsets, branch, reflections, (64, 64), atlas, weights,
                          return_refinement_feature=True, feature_side=64,
                          source_shape=image.shape[-2:])
        refined_state, _, _ = model.refine(first['refinement_feature'], selected['state'])
        refined = {**selected, 'state': refined_state}
        mapped = model.map(refined, offsets, branch, reflections, (96, 96), atlas, weights,
                           feature_side=96, source_shape=image.shape[-2:])
        fitted = model.score_fitted_candidates(image, refined, mapped, atlas, weights)[0]
        choice = int(fitted.argmax())
        guess_state = refined_state[:, choice]
        guess_reflection = reflections[:, choice]
        renders = []
        for state, reflection in ((true_state, true_reflection), (guess_state, guess_reflection)):
            centre, frame, basis = full_frame_state_to_components(state)
            sx = (191 / 192 - xx) if bool(reflection[0]) else xx
            chart = torch.stack((sx, yy), -1)[None]
            plane = normalized_raster_to_ccf(
                centre[:, None, None], frame[:, None, None], basis[:, None, None], chart)
            slab = render_finite_thickness_coordinate_grid(
                atlas, plane[:, None] + offsets[:, :, None, None, None] * frame[
                    :, None, None, None, :, 2], (0., 0., 0.), (25., 25., 25.), weights)[0]
            rendered = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            rendered[slab[1].cpu().numpy() < .05] = 0
            renders.append(rendered)
        centre, frame, basis = full_frame_state_to_components(guess_state)
        chart = torch.stack((xx[valid], yy[valid]), -1)
        if bool(guess_reflection[0]):
            chart[:, 0] = 191 / 192 - chart[:, 0]
        rigid = centre[:, None] + torch.einsum('bij,pj->bpi', frame[..., :2] @ basis, chart - .5)
        rigid_error_mm = float((rigid[0] - target[valid]).norm(dim=-1).mean() / 1000)
        true_axis = full_frame_state_to_components(true_state)[1][0, :, 2]
        guess_axis = frame[0, :, 2]
        axis_error = float(torch.rad2deg(torch.acos((true_axis * guess_axis).sum().abs().clamp_max(1))))
        true_centre, _, true_basis = full_frame_state_to_components(true_state)
        centre_error_mm = float((centre[0] - true_centre[0]).norm() / 1000)
        true_fov_mm = (torch.linalg.svdvals(true_basis[0]) / 1000).cpu().tolist()
        guess_fov_mm = (torch.linalg.svdvals(basis[0]) / 1000).cpu().tolist()
        reference = reference_rows[row['provenance']['physical_section_id']]
        assert np.isclose(axis_error, reference['fitted_selected_normal_axis_error_deg'], atol=1e-4)
        assert np.isclose(rigid_error_mm * 1000, reference['fitted_selected_rigid_um'], atol=.1)
        for j, visual in enumerate((renders[0], image[0, 0].cpu().numpy(), renders[1])):
            axes[i, j].imshow(visual, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 1].text(.02, .98, f'{cut_angle:.0f}° oblique · {row["appearance_mode"]} · {event}',
                        transform=axes[i, 1].transAxes, va='top', color='white', fontsize=9,
                        bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
        axes[i, 2].text(.02, .98, f'{axis_error:.0f}° axis error · {rigid_error_mm:.1f} mm coordinate error',
                        transform=axes[i, 2].transAxes, va='top', color='white', fontsize=9,
                        bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none'})
        result.append({'physical_section_id': row['provenance']['physical_section_id'],
                       'file_sha256': row['sha256'], 'mode': row['appearance_mode'],
                       'required_artifact': event, 'nearest_principal_angle_deg': cut_angle,
                       'normal_axis_error_deg': axis_error, 'rigid_error_mm': rigid_error_mm,
                       'centre_error_mm': centre_error_mm, 'true_fov_mm': true_fov_mm,
                       'guessed_fov_mm': guess_fov_mm})

for ax, title in zip(axes[0], ('Atlas at true plane', 'Actual generated input',
                               'Atlas at fitted model 247 plane')):
    ax.set_title(title, fontsize=11)
out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(out, dpi=150)
plt.close(figure)
(out.with_suffix('.json')).write_text(json.dumps({
    'selection': 'first eligible case per fixed mode/angle/artifact stratum, distinct synthetic subjects; no error-based selection',
    'panel_completed_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'readout_completed_sha256': hashlib.sha256((readout / 'completed.json').read_bytes()).hexdigest(),
    'atlas_cache_completed_sha256': hashlib.sha256((cache / 'completed.json').read_bytes()).hexdigest(),
    'rows': result}, indent=2))
print(json.dumps({'image': str(out), 'rows': result}))
