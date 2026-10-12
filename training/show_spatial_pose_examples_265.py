"""Four predeclared fresh DEV examples from the failed spatial pose pilot."""
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

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.spatial_pose_candidate_265 import (
    SpatialPoseUpdater, candidate_bank, five_point_error, render_exact,
)

PANEL = ROOT / 'data/spatial_pose_fresh_dev_265_001'
READOUT = ROOT / 'runs/spatial_pose_updater_265_readout_001'
TRAIN = ROOT / 'runs/spatial_pose_updater_265_train_001'
PARENT = ROOT / 'runs/dense_physical_pose_260_train_001'
OUT = ROOT / 'reports/spatial_pose_265_examples'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

records = {row['section_id']: row for line in (PANEL / 'records.jsonl').open(encoding='utf8')
           if (row := json.loads(line))['eligible'] and row['valid_pixels'] > 0}
rows = [row for line in (READOUT / 'rows.jsonl').open(encoding='utf8')
        if (row := json.loads(line))['panel'] == 'fresh' and row['batch'] == 2000
        and row['arm'] == 'full']
rules = (
    ('near-principal raw', lambda r: r['appearance_mode'] == 'raw' and
     r['valid_fraction'] >= .2 and r['cut_nearest_principal_deg'] < 15),
    ('exact black', lambda r: r['appearance_mode'] == 'exact_black' and
     r['valid_fraction'] >= .25 and r['cut_nearest_principal_deg'] >= 20),
    ('imperfect brush', lambda r: r['appearance_mode'] == 'imperfect_brush' and
     r['valid_fraction'] >= .2 and r['cut_nearest_principal_deg'] >= 30),
    ('steep raw', lambda r: r['appearance_mode'] == 'raw' and
     r['valid_fraction'] >= .2 and r['cut_nearest_principal_deg'] >= 40),
)
chosen, used = [], set()
for label, rule in rules:
    row = next(row for row in rows if rule(row) and row['animal_id'] not in used)
    chosen.append((label, row))
    used.add(row['animal_id'])

context = load_streaming_synthetic_v7_64(device='cuda')
parent = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
    dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
parent.load_state_dict(torch.load(PARENT / 'joint_step_03000.pt',
    map_location='cpu', weights_only=True)['model'], strict=True)
anchor_model = OneShotJointSliceModel(modes=0, normal_anchor_count=64)
anchors = anchor_model.normal_anchor_frames.cuda()
del anchor_model
model = SpatialPoseUpdater().cuda().eval()
model.load_state_dict(torch.load(TRAIN / 'joint_step_02000.pt',
    map_location='cpu', weights_only=True)['full'], strict=True)

fig, axes = plt.subplots(4, 3, figsize=(10, 12), layout='constrained')
axes[0, 0].set_title('Actual generated input')
axes[0, 1].set_title('Correct atlas plane')
axes[0, 2].set_title('Model-selected atlas plane')
metadata = []
with torch.inference_mode():
    for i, (label, row) in enumerate(chosen):
        record = records[row['section_id']]
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            input_np = arrays['inputs'][0].copy()
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            true_flag = torch.from_numpy(np.asarray(arrays['reflection']).reshape(1).copy()).cuda()
        predicted = parent.dense_coordinate_plane(parent.predict(image), source_side=192)
        centre = full_frame_state_to_components(predicted)[0][0]
        bank, flags, _ = candidate_bank(centre, anchors)
        index = row['selected_candidate_index']
        state, flag = bank[index:index + 1], flags[index:index + 1]
        query = model.encode_image(image)
        for _ in range(2):
            _, update, _ = model(query, render_exact(context['atlas'], state, flag))
            state = compose_full_frame_state(state, update)
        assert abs(float(five_point_error(state, flag, truth, true_flag)[0]) -
                   row['selected_five_mm']) < .001
        rendered = [render_exact(context['atlas'], truth, true_flag)[0].cpu().numpy(),
                    render_exact(context['atlas'], state, flag)[0].cpu().numpy()]
        atlas_images = [np.where(pair[1] > .1, pair[0], 0) for pair in rendered]
        atlas_limit = max(np.quantile(np.concatenate([a.ravel() for a in atlas_images]), .995), .05)
        axes[i, 0].imshow(input_np, cmap='gray', vmin=0,
                          vmax=max(np.quantile(input_np, .995), .05), interpolation='nearest')
        for j, atlas_image in enumerate(atlas_images, start=1):
            axes[i, j].imshow(atlas_image, cmap='gray', vmin=0, vmax=atlas_limit,
                              interpolation='nearest')
        axes[i, 0].set_ylabel(f"{label}\n{row['cut_nearest_principal_deg']:.0f}° from principal",
                              fontsize=9)
        axes[i, 2].set_xlabel(f"{row['selected_rigid_mm']:.1f} mm tissue-mapping error",
                              fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        metadata.append({'label': label, 'section_id': row['section_id'],
            'physical_section_id': row['physical_section_id'],
            'cut_nearest_principal_deg': row['cut_nearest_principal_deg'],
            'selected_rigid_mm': row['selected_rigid_mm'],
            'parent260_rigid_mm': row['parent260_rigid_mm'],
            'selected_candidate_index': index, 'input_sha256': record['sha256']})

fig.suptitle('Failed 265 pilot · fresh synthetic sections · four distinct held-out plans', fontsize=12)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'actual_truth_prediction.png', dpi=150)
plt.close(fig)
(OUT / 'selection.json').write_text(json.dumps({'selection': metadata,
    'rule': 'First eligible row per displayed criterion, no repeated plan; selected by appearance and angle, not outcome',
    'rendering': 'Correct and predicted atlas are nominal 25-µm PSF, same per-row display contrast; generated input has independent per-row display contrast only',
    'checkpoint': '265 full atlas, 2000 training batches chosen on fresh DEV'}, indent=2),
    encoding='utf8')
