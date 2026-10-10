"""Four median-case, frozen synthetic DEV plane-correction examples; run after 150 evaluation."""

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
from training.coarse_atlas_pose_150 import CoarseAtlasPose150
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


panel = root / 'data/coarse_atlas_pose_150_fresh_dev_panel'
evaluation = root / 'runs/coarse_atlas_pose_150_fresh_dev_eval'
training = root / 'runs/coarse_atlas_pose_150'
parent = root / 'runs/v4_pose_adaptation_132'
out = root / 'reports/coarse_atlas_pose_150_fresh_dev_examples'
assert all(path.drive.upper() == 'I:' for path in (panel, evaluation, training, parent, out))
assert not out.exists()

panel_done = json.loads((panel / 'completed.json').read_text())
eval_done = json.loads((evaluation / 'completed.json').read_text())
eval_config = json.loads((evaluation / 'config.json').read_text())
train_done = json.loads((training / 'completed.json').read_text())
parent_done = json.loads((parent / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(evaluation / 'rows.jsonl') == eval_done['rows_sha256']
assert sha(evaluation / 'config.json') == eval_done['config_sha256']
assert sha(evaluation / 'summary.json') == eval_done['summary_sha256']
assert eval_config['panel_completion_sha256'] == sha(panel / 'completed.json')
assert eval_config['train150_completion_sha256'] == sha(training / 'completed.json')
assert eval_config['selected_step'] == train_done['selected_step']

records = {row['section_id']: row for row in
           (json.loads(line) for line in (panel / 'records.jsonl').open())}
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
rows = [row for row in rows if row['arm'] == 'full' and row['role'] == 'blind_first']
assert len(records) == panel_done['physical_sections'] == 256
assert len(rows) == panel_done['eligible']
assert len({row['section_id'] for row in rows}) == len(rows)
assert all(records[row['section_id']]['eligible'] and
           records[row['section_id']]['sha256'] == row['panel_file_sha256'] for row in rows)

strata = (('raw, <45°', 'raw', False),
          ('exact black, <45°', 'exact_black', False),
          ('imperfect brush, <45°', 'imperfect_brush', False),
          ('steep oblique, ≥45°', None, True))
chosen = []
used_plans = set()
for label, mode, steep in strata:
    group = [row for row in rows if (mode is None or row['appearance_mode'] == mode)
             and (row['nearest_cardinal_angle_deg'] >= 45) == steep]
    median = float(np.median([row['after_rigid_mm'] for row in group]))
    ordered = sorted(group, key=lambda row: (
        abs(row['after_rigid_mm'] - median),
        hashlib.sha256(row['section_id'].encode()).hexdigest()))
    row = next(row for row in ordered if row['synthetic_subject_plan_id'] not in used_plans)
    used_plans.add(row['synthetic_subject_plan_id'])
    chosen.append((label, row, median, len(group)))

selected_step = train_done['selected_step']
checkpoint = training / f'full_step_{selected_step:05d}.pt'
parent_checkpoint = parent / 'joint_step_02000.pt'
assert sha(checkpoint) == train_done['checkpoint_sha256']['full'][str(selected_step)]
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256']['2000']
assert eval_config['parent_checkpoint_sha256'] == sha(parent_checkpoint)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
pose.load_state_dict(torch.load(parent_checkpoint, map_location='cpu',
                                weights_only=True)['model'], strict=True)
matcher = CoarseAtlasPose150().cuda().eval().requires_grad_(False)
matcher.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                   weights_only=True)['matcher'], strict=True)
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
figure, axes = plt.subplots(4, 4, figsize=(12, 12), facecolor='white')
metadata = []

with torch.inference_mode():
    for i, (label, row, median, count) in enumerate(chosen):
        record = records[row['section_id']]
        file = panel / record['file']
        assert sha(file) == record['sha256']
        with np.load(file, allow_pickle=False) as values:
            image = torch.from_numpy(values['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(values['target_state'][None].copy()).cuda()
            reflection = int(values['reflection'])
            offsets = torch.from_numpy(values['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(values['weights'][None].copy()).cuda()
        prediction = pose.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        assert branch == row['initial_branch_id']
        proposed = prediction['state'][:, branch // 2]
        proposed_reflection = torch.tensor([[branch % 2]], device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            result = matcher(image, atlas, proposed[:, None], proposed_reflection,
                             offsets, weights, atlas_intensity=True)
        corrected = result['corrected_state'][:, 0].float()
        images = []
        for state, flip in ((truth, reflection), (proposed, branch % 2),
                            (corrected, branch % 2)):
            center, frame, basis = full_frame_state_to_components(state)
            chart = torch.stack(((255 / 256 - x) if flip else x, y), -1)
            plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                             basis[:, None, None], chart)
            coordinates = (plane[:, None] + offsets[:, :, None, None, None] *
                           frame[:, None, None, None, :, 2])
            slab = render_finite_thickness_coordinate_grid(atlas, coordinates,
                ATLAS_ORIGIN_AP_DV_ML_UM_V6, ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6, weights)[0]
            intensity = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            intensity[slab[1].cpu().numpy() < .05] = 0
            images.append(intensity)
        for j, display in enumerate((images[0], image[0, 0].cpu().numpy(),
                                     images[1], images[2])):
            axes[i, j].imshow(display, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].annotate(f'{label}\n{row["nearest_cardinal_angle_deg"]:.0f}°',
            xy=(-.04, .5), xycoords='axes fraction', ha='right', va='center',
            fontsize=9, clip_on=False)
        metadata.append({'stratum': label, 'stratum_count': count,
            'stratum_median_after_rigid_mm': median,
            'section_id': row['section_id'],
            'physical_section_id': row['physical_section_id'],
            'synthetic_subject_plan_id': row['synthetic_subject_plan_id'],
            'plan_receipt_sha256': row['plan_receipt_sha256'],
            'animal_id': row['animal_id'], 'specimen_id': row['specimen_id'],
            'experiment_id': row['experiment_id'],
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'appearance_mode': row['appearance_mode'],
            'nearest_cardinal_angle_deg': row['nearest_cardinal_angle_deg'],
            'target_reflection': reflection, 'initial_branch_id': branch,
            'before_rigid_mm': row['before_rigid_mm'],
            'after_rigid_mm': row['after_rigid_mm']})

for j, title in enumerate(('true atlas plane', 'actual artifacted input',
                           '132 top-prior atlas plane', 'after 150 correction')):
    axes[0, j].set_title(title, fontsize=10)
figure.text(.5, .01, 'Median cases by corrected blind-first error in each declared stratum; distinct synthetic plans. '
    'Atlas panels are finite-thickness renders without local tissue warp. Synthetic DEV only.',
    ha='center', fontsize=8)
figure.tight_layout(rect=(0, .025, 1, 1))
out.mkdir(parents=True, exist_ok=False)
figure.savefig(out / 'target_input_proposed_corrected.png', dpi=150)
plt.close(figure)
(out / 'selection.json').write_text(json.dumps({
    'rule': 'For each declared stratum, rank by absolute deviation of full-arm blind-first after-error from stratum median, then SHA256(section_id); choose first not sharing an already selected synthetic plan. No manual examples or reranking.',
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'evaluation_completion_sha256': sha(evaluation / 'completed.json'),
    'evaluation_rows_sha256': sha(evaluation / 'rows.jsonl'),
    'train150_completion_sha256': sha(training / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'matcher_checkpoint_sha256': sha(checkpoint),
    'image_sha256': sha(out / 'target_input_proposed_corrected.png'),
    'scope': 'Synthetic development plans only; not biological all-angle validation, calibration, local-warp quality, or public benchmark.',
    'rows': metadata}, indent=2))
print(json.dumps({'image': str(out / 'target_input_proposed_corrected.png'),
                  'metadata': str(out / 'selection.json')}, indent=2))
