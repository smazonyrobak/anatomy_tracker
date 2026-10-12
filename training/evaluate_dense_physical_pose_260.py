"""Frozen v7 DEV readout of the physical-pose-supervised dense model."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
PARENT = ROOT / 'runs/dense_joint_geometry_258_readout_001'
TRAIN = ROOT / 'runs/dense_physical_pose_260_train_001'
OUT = ROOT / 'runs/dense_physical_pose_260_readout_001'
DOC = Path('docs/publication/DENSE_PHYSICAL_POSE_260_PROTOCOL_20261012.md')
STEPS, SIDE = (1000, 3000), 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_done = json.loads((PANEL / 'completed.json').read_text())
parent_done = json.loads((PARENT / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
config = json.loads((TRAIN / 'config.json').read_text())
assert hashlib.sha256(DOC.read_bytes()).hexdigest() == config['protocol_sha256']
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
assert hashlib.sha256((PARENT / 'rows.jsonl').read_bytes()).hexdigest() == parent_done['rows_sha256']
assert hashlib.sha256((TRAIN / 'config.json').read_bytes()).hexdigest() == train_done['config_sha256']
assert hashlib.sha256((TRAIN / 'training.jsonl').read_bytes()).hexdigest() == train_done['training_sha256']
assert hashlib.sha256((TRAIN / 'draws.jsonl').read_bytes()).hexdigest() == train_done['draws_sha256']
assert all(hashlib.sha256((TRAIN / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest()
           == train_done['checkpoint_sha256'][str(step)] for step in STEPS)
assert hashlib.sha256(Path('training/arbitrary_plane_one_shot_model.py').read_bytes()).hexdigest() == config['source_sha256']['arbitrary_plane_one_shot_model.py']

records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [record for record in records if record['eligible'] and record['valid_pixels'] > 0]
parent = {row['physical_section_id']: row for row in
          (json.loads(line) for line in (PARENT / 'rows.jsonl').open(encoding='utf8'))
          if row['batch'] == 5000}
assert len(eligible) == len(parent) == 227
assert {record['provenance']['physical_section_id'] for record in eligible} == set(parent)
assert len({record['animal_id'] for record in eligible}) == 8
assert all(hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
           for record in records)

model = OneShotJointSliceModel(modes=0, normal_anchor_count=0,
                              dense_coordinate_pyramid=True, atlas_conditioning=False).cuda().eval()
axis = torch.arange(SIDE, device='cuda') / SIDE
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for step in STEPS:
        saved = torch.load(TRAIN / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == step and not saved['calibrated']
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for record in eligible:
            identity = record['provenance']['physical_section_id']
            with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            prediction = model.predict(image)
            field = prediction['dense_coordinate']
            state = model.dense_coordinate_plane(prediction, source_side=SIDE)
            centre, frame, basis = full_frame_state_to_components(state)
            _, true_frame, true_basis = full_frame_state_to_components(truth)
            rigid = centre[:, None, None] + torch.einsum(
                'bij,hwj->bhwi', frame[:, :, :2] @ basis, chart - .5)
            rigid_mm = float((rigid - target)[valid].norm(dim=-1).mean() / 1000)
            axis_deg = float(torch.rad2deg(torch.acos(
                (frame[0, :, 2] @ true_frame[0, :, 2]).abs().clamp_max(1))))
            v96 = F.avg_pool2d(valid[:, None].float(), 2)[:, 0] > .999
            t96 = F.avg_pool2d(target.permute(0, 3, 1, 2), 2)
            coordinate = model.center_origin[None, :, None, None] + (
                field[:, :3] * model.center_scale[None, :, None, None])
            dense_mm = float((coordinate - t96).norm(dim=1)[v96].mean() / 1000)
            selected_valid = field[:, 3] > 0
            dice = float(2 * (selected_valid & v96).sum() /
                         (selected_valid.sum() + v96.sum()).clamp_min(1))
            predicted_minor = float(torch.linalg.svdvals(basis[0]).min() / 1000)
            true_minor = float(torch.linalg.svdvals(true_basis[0]).min() / 1000)
            assert parent[identity]['batch'] == 5000
            assert parent[identity]['panel_file_sha256'] == record['sha256']
            row = {'batch': step, 'physical_section_id': identity,
                   'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                   'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
                   'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
                   'cut_nearest_principal_deg': parent[identity]['cut_nearest_principal_deg'],
                   'valid_fraction': record['valid_pixels'] / SIDE**2,
                   'fitted_rigid_mm': rigid_mm, 'fitted_axis_deg': axis_deg,
                   'dense_coordinate_mm': dense_mm, 'validity_dice': dice,
                   'minor_fov_mm': predicted_minor, 'true_minor_fov_mm': true_minor,
                   'parent258_rigid_mm': parent[identity]['fitted_rigid_mm'],
                   'parent258_axis_deg': parent[identity]['fitted_axis_deg'],
                   'parent258_dense_mm': parent[identity]['dense_coordinate_mm'],
                   'parent258_minor_fov_mm': parent[identity]['minor_fov_mm']}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'batch': step, 'cases': len(eligible)}), flush=True)

metrics = ('fitted_rigid_mm', 'fitted_axis_deg', 'dense_coordinate_mm', 'validity_dice',
           'parent258_rigid_mm', 'parent258_axis_deg', 'parent258_dense_mm')
animals = sorted({row['animal_id'] for row in rows})
by_step = {}
for step in STEPS:
    selected = [row for row in rows if row['batch'] == step]
    subject_means = {}
    for animal in animals:
        group = [row for row in selected if row['animal_id'] == animal]
        subject_means[animal] = {key: float(np.mean([row[key] for row in group])) for key in metrics}
    equal = {key: float(np.mean([subject_means[animal][key] for animal in animals])) for key in metrics}
    appearance = {}
    for mode in sorted({row['appearance_mode'] for row in selected}):
        group = [row for row in selected if row['appearance_mode'] == mode]
        appearance[mode] = {'cases': len(group),
            'case_mean_fitted_mm': float(np.mean([row['fitted_rigid_mm'] for row in group])),
            'case_mean_dense_mm': float(np.mean([row['dense_coordinate_mm'] for row in group]))}
    cut_strata = {}
    for low, high in ((0, 15), (15, 30), (30, 45), (45, 60)):
        group = [row for row in selected if low <= row['cut_nearest_principal_deg'] < high]
        cut_strata[f'{low}_{high}'] = {'cases': len(group),
            'case_mean_fitted_mm': float(np.mean([row['fitted_rigid_mm'] for row in group])) if group else None}
    by_step[str(step)] = {'subject_equal_mean': equal, 'subject_means': subject_means,
        'appearance': appearance, 'cut_strata': cut_strata,
        'median_minor_fov_mm': float(np.median([row['minor_fov_mm'] for row in selected])),
        'true_median_minor_fov_mm': float(np.median([row['true_minor_fov_mm'] for row in selected])),
        'minor_fov_below_half_truth_cases': sum(row['minor_fov_mm'] < .5 * row['true_minor_fov_mm'] for row in selected),
        'plans_better_than_258': sum(subject_means[animal]['fitted_rigid_mm'] <
                                     subject_means[animal]['parent258_rigid_mm'] for animal in animals)}
best = min(STEPS, key=lambda step: by_step[str(step)]['subject_equal_mean']['fitted_rigid_mm'])
chosen = by_step[str(best)]
equal = chosen['subject_equal_mean']
gates = {'fitted_rigid': equal['fitted_rigid_mm'] <= 3.2,
         'dense_coordinate': equal['dense_coordinate_mm'] <= 3.25,
         'normal_axis': equal['fitted_axis_deg'] <= 49,
         'minor_fov': chosen['median_minor_fov_mm'] >= 6}
summary = {'status': 'synthetic DEV geometry only; no calibrated uncertainty or GUI readiness',
           'cases_per_checkpoint': len(eligible), 'synthetic_plans': len(animals),
           'checkpoints': by_step, 'best_batch': best, 'gates': gates,
           'continuation_gate': all(gates.values())}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'cases_per_checkpoint': len(eligible),
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'parent258_completed_sha256': hashlib.sha256((PARENT / 'completed.json').read_bytes()).hexdigest(),
    'train_completed_sha256': hashlib.sha256((TRAIN / 'completed.json').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(DOC.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((Path('training') / name).read_bytes()).hexdigest()
                      for name in (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                                   'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'best_batch': best,
                  'continuation_gate': summary['continuation_gate']}), flush=True)
