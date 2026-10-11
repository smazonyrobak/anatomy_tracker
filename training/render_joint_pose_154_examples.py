"""Fixed-case synthetic DEV examples: atlas truth, artifacted input, blind 154 pose."""

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
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


panel = root / 'data/fresh_v4_pose_dev_panel_132'
evaluation = root / 'runs/joint_pose_154_frozen_dev_eval_001'
training = root / 'runs/joint_pose_curriculum_154_train151_001'
out = root / 'reports/joint_pose_154_fresh_v4_examples'
assert all(path.drive.upper() == 'I:' for path in (panel, evaluation, training, out))
assert not out.exists()
panel_done = json.loads((panel / 'completed.json').read_text())
eval_done = json.loads((evaluation / 'completed.json').read_text())
eval_config = json.loads((evaluation / 'config.json').read_text())
train_done = json.loads((training / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(evaluation / 'rows.jsonl') == eval_done['rows_sha256']
assert eval_config['panel_receipt_sha256']['v4'] == sha(panel / 'completed.json')
selected = train_done['selected_step']
checkpoint = training / f'joint_pose_step_{selected:05d}.pt'
assert sha(checkpoint) == train_done['checkpoint_sha256'][str(selected)]
assert eval_config['checkpoint_sha256']['154_selected'] == sha(checkpoint)

records = {row['section_id']: row for row in
           (json.loads(line) for line in (panel / 'records.jsonl').open())}
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
rows = [row for row in rows if row['arm'] == '154_selected' and row['cohort'] == 'v4']
assert len(rows) == panel_done['eligible'] == 248

strata = (
    ('Raw, substantial tissue', lambda r: r['appearance_mode'] == 'raw'
        and r['valid_fraction'] >= .20),
    ('Raw, oblique cut', lambda r: r['appearance_mode'] == 'raw'
        and r['obliquity_deg'] >= 40 and r['valid_fraction'] >= .08),
    ('Clean black background', lambda r: r['appearance_mode'] == 'exact_black'
        and r['valid_fraction'] >= .08),
    ('Imperfect brush', lambda r: r['appearance_mode'] == 'imperfect_brush'
        and r['valid_fraction'] >= .08),
    ('Small surviving fragment', lambda r: r['valid_fraction'] < .08),
)
chosen, used_plans = [], set()
for label, condition in strata:
    candidates = sorted((r for r in rows if condition(r)),
        key=lambda r: hashlib.sha256(r['section_id'].encode()).hexdigest())
    case = next(r for r in candidates if r['synthetic_plan_id'] not in used_plans)
    used_plans.add(case['synthetic_plan_id'])
    chosen.append((label, case))

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = JointAnatomyModel153().cuda().eval().requires_grad_(False)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
axis = torch.arange(256, device='cuda') / 256
y, x = torch.meshgrid(axis, axis, indexing='ij')
figure, axes = plt.subplots(len(chosen), 3, figsize=(10, 15), facecolor='white')
metadata = []

with torch.inference_mode():
    for i, (label, row) in enumerate(chosen):
        record = records[row['section_id']]
        file = panel / record['file']
        assert sha(file) == record['sha256']
        with np.load(file, allow_pickle=False) as sample:
            image = torch.from_numpy(sample['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(sample['target_state'][None].copy()).cuda()
            target_reflection = int(sample['reflection'])
            offsets = torch.from_numpy(sample['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(sample['weights'][None].copy()).cuda()
        prediction = model.pose.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        assert branch == int(np.argmax(row['log_prior_all160_uncalibrated']))
        proposed = prediction['state'][:, branch // 2]
        displays = []
        for state, reflected in ((truth, target_reflection), (proposed, branch % 2)):
            centre, frame, basis = full_frame_state_to_components(state)
            chart = torch.stack(((255 / 256 - x) if reflected else x, y), -1)
            plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                             basis[:, None, None], chart)
            coordinates = (plane[:, None] + offsets[:, :, None, None, None]
                           * frame[:, None, None, None, :, 2])
            slab = render_finite_thickness_coordinate_grid(atlas, coordinates,
                ATLAS_ORIGIN_AP_DV_ML_UM_V6, ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6, weights)[0]
            intensity = (slab[0] / slab[1].clamp_min(1e-4)).cpu().numpy()
            intensity[slab[1].cpu().numpy() < .05] = 0
            displays.append(intensity)
        for j, display in enumerate((displays[0], image[0, 0].cpu().numpy(), displays[1])):
            axes[i, j].imshow(display, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].annotate(f'{label}\n{row["obliquity_deg"]:.0f}° from cardinal',
            xy=(-.04, .5), xycoords='axes fraction', ha='right', va='center',
            fontsize=9, clip_on=False)
        axes[i, 2].annotate(f'{row["top1_mm"]:.2f} mm error', xy=(.5, -.04),
            xycoords='axes fraction', ha='center', va='top', fontsize=9, clip_on=False)
        metadata.append({'stratum': label, 'section_id': row['section_id'],
            'physical_section_id': row['physical_section_id'],
            'synthetic_plan_id': row['synthetic_plan_id'],
            'appearance_mode': row['appearance_mode'],
            'obliquity_deg': row['obliquity_deg'], 'valid_fraction': row['valid_fraction'],
            'blind_branch_id': branch, 'blind_top1_mm': row['top1_mm'],
            'blind_top1_normal_deg': row['top1_normal_deg'],
            'panel_file': record['file'], 'panel_file_sha256': record['sha256']})

for j, title in enumerate(('Atlas at true plane', 'Actual artifacted input',
                           'Atlas at blind guessed plane')):
    axes[0, j].set_title(title, fontsize=11)
figure.text(.5, .01, 'Deterministic case selection ignores model errors. One Allen atlas and synthetic DEV plans; not physical-animal validation.',
            ha='center', fontsize=8)
figure.tight_layout(rect=(.17, .025, 1, 1))
out.mkdir(parents=True, exist_ok=False)
figure.savefig(out / 'true_input_guessed.png', dpi=150)
plt.close(figure)
(out / 'selection.json').write_text(json.dumps({
    'selection_rule': 'First SHA256(section_id) per declared stratum with distinct synthetic plan; no performance selection.',
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'evaluation_completion_sha256': sha(evaluation / 'completed.json'),
    'evaluation_rows_sha256': sha(evaluation / 'rows.jsonl'),
    'checkpoint_sha256': sha(checkpoint),
    'image_sha256': sha(out / 'true_input_guessed.png'),
    'scope': 'Synthetic DEV only; canonical atlas renders omit specimen warp and damage.',
    'rows': metadata}, indent=2))
print(json.dumps({'image': str(out / 'true_input_guessed.png'),
                  'metadata': str(out / 'selection.json')}, indent=2))
