"""Six geometry/appearance-selected frozen v4 sections and direct 132 pose renders."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    _decode_and_preprocess_allen_v6, ATLAS_ORIGIN_AP_DV_ML_UM_V6,
    ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6)
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_plane_matcher_120 import attach_global_plane_matcher

panel = root / 'data/fresh_v4_pose_dev_panel_132'
run = root / 'runs/v4_pose_adaptation_132'
evaluation = root / 'runs/v4_pose_adaptation_132_dev_eval'
out = root / 'reports/v4_generator_grid_133'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
rows = [json.loads(line) for line in (evaluation / 'synthetic_rows.jsonl').open()
        if '"arm": "2000", "cohort": "v4"' in line]
evaluated = {row['section_id']: row for row in rows}
assert len(records) == 256 and len(evaluated) == 248
assert hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest() == json.loads(
    (panel / 'completed.json').read_text())['records_sha256']
assert hashlib.sha256((evaluation / 'synthetic_rows.jsonl').read_bytes()).hexdigest() == json.loads(
    (evaluation / 'completed.json').read_text())['output_sha256']['synthetic_rows.jsonl']

categories = (
    ('near-coronal, raw, low exposure', 0, 'raw', True, None),
    ('near-sagittal, black, non-low exposure', 2, 'exact_black', False, None),
    ('near-horizontal, raw, non-low exposure', 1, 'raw', False, None),
    ('oblique, brush, low, fragment', None, 'imperfect_brush', True, 'fragment'),
    ('oblique, raw, non-low, fold/seam', None, 'raw', False, 'fold_or_seam'),
    ('oblique, raw, low, bubble', None, 'raw', True, 'bubble'),
)
chosen = []
used_plans = set()
for label, axis, mode, low, event in categories:
    candidates = []
    for record in records:
        if not record['eligible'] or record['appearance_mode'] != mode:
            continue
        normal = np.abs(record['plane_normal_ap_dv_ml'])
        angle = float(np.degrees(np.arccos(normal.max())))
        if (int(normal.argmax()) != axis if axis is not None else angle < 35):
            continue
        if axis is not None and angle > 20:
            continue
        if record['provenance']['one_shot_slide_artifacts_v4']['low_exposure'] != low:
            continue
        events = record['provenance']['one_shot_slide_artifacts_v3']['events']
        if event == 'fold_or_seam' and not (events['fold'] or events['tile_seam']):
            continue
        if event not in (None, 'fold_or_seam') and not events[event]:
            continue
        candidates.append(record)
    candidates.sort(key=lambda record: hashlib.sha256(record['section_id'].encode()).hexdigest())
    record = next(record for record in candidates
                  if record['synthetic_subject_plan_id'] not in used_plans)
    used_plans.add(record['synthetic_subject_plan_id'])
    chosen.append((label, record))

checkpoint = run / 'joint_step_02000.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == json.loads(
    (run / 'completed.json').read_text())['checkpoint_sha256']['2000']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
atlas, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas).cuda()
axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
figure, axes = plt.subplots(6, 4, figsize=(12, 16), facecolor='white')
metadata = []
torch.set_num_threads(4)
with torch.inference_mode():
    for i, (label, record) in enumerate(chosen):
        file = panel / record['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256']
        with np.load(file, allow_pickle=False) as values:
            image = torch.from_numpy(values['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(values['target_state'][None].copy()).cuda()
            reflection = int(values['reflection'])
            offsets = torch.from_numpy(values['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(values['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        assert branch == evaluated[record['section_id']]['top1_branch_id']
        states = (truth, prediction['state'][:, branch // 2])
        rendered_images = []
        for state, flip in zip(states, (reflection, branch % 2)):
            center, frame, basis = full_frame_state_to_components(state)
            chart = torch.stack(((255 / 256 - x) if flip else x, y), -1)
            plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                             basis[:, None, None], chart)
            coords = plane[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
            slab = render_finite_thickness_coordinate_grid(atlas, coords,
                ATLAS_ORIGIN_AP_DV_ML_UM_V6, ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6, weights)[0]
            intensity = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            intensity[slab[1].cpu().numpy() < .05] = 0
            rendered_images.append(intensity)
        observed = image[0, 0].cpu().numpy()
        stretch = max(float(np.quantile(observed, .995)), .001)
        for j, (display, upper) in enumerate(zip(
                (rendered_images[0], observed, observed, rendered_images[1]),
                (1., 1., stretch, 1.))):
            axes[i, j].imshow(display, cmap='gray', vmin=0, vmax=upper)
            axes[i, j].axis('off')
        normal = np.abs(record['plane_normal_ap_dv_ml'])
        angle = float(np.degrees(np.arccos(normal.max())))
        exposure = record['provenance']['one_shot_slide_artifacts_v4']['exposure']
        axes[i, 0].annotate(f'{label}\n{angle:.0f}°; exposure {exposure:.2f}',
            xy=(-.03, .5), xycoords='axes fraction', ha='right', va='center',
            fontsize=9, clip_on=False)
        metadata.append({'label': label, 'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'angle_to_nearest_cardinal_deg': angle, 'exposure_multiplier': exposure,
            'appearance_mode': record['appearance_mode'],
            'events': record['provenance']['one_shot_slide_artifacts_v3']['events'],
            'target_reflection': reflection, 'predicted_branch_id': branch,
            'direct_top1_rigid_error_mm': evaluated[record['section_id']]['top1_rigid_mm']})

for j, title in enumerate(('target atlas gauge', 'actual input [0,1]',
                           'same input, display-stretched', 'direct top-1 atlas plane')):
    axes[0, j].set_title(title, fontsize=10)
figure.text(.5, .005, 'Atlas render uses frozen section thickness; target is an affine atlas gauge, not intact deformed tissue. '
    'The predicted image is direct top-1 pose, without local warping or calibration.',
    ha='center', fontsize=9)
figure.tight_layout(rect=(0, .02, 1, 1))
out.mkdir(parents=True, exist_ok=True)
figure.savefig(out / 'frozen_v4_target_input_prediction.png', dpi=150)
plt.close(figure)
(out / 'selection.json').write_text(json.dumps({
    'rule': 'preselected by orientation/mode/exposure/artifact only; SHA256(section_id) tie-break; distinct synthetic DEV plans; no accuracy-based choice',
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'evaluation_rows_sha256': hashlib.sha256((evaluation / 'synthetic_rows.jsonl').read_bytes()).hexdigest(),
    'rows': metadata}, indent=2))
print(json.dumps({'image': str(out / 'frozen_v4_target_input_prediction.png'),
                  'metadata': str(out / 'selection.json'), 'rows': metadata}, indent=2))
