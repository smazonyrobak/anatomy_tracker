"""Compare frozen 047 checkpoints on the identity-disjoint synthetic DEV panel."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.joint_pose_fit_047 import joint_pose_fit
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
run = root / 'runs/joint_pose_fit_047_pilot'
out = root / 'runs/joint_pose_fit_047_development_eval'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
fine_path = root / 'runs/pose_fine_subgrid_046_pilot/joint_step_02000.pt'
steps = (0, 250, 500)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [record for record in records if record['eligible']]
assert len(eligible) == 177
assert json.loads((run / 'completed.json').read_text())['batches'] == 500
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(coarse_path, map_location='cpu', weights_only=True)['head'])
fine = PoseFineMatch046().cuda().eval()
fine.load_state_dict(torch.load(fine_path, map_location='cpu', weights_only=True)['model'])
models = {}
for step in steps:
    saved = torch.load(run / f'joint_step_{step:05d}.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == step and not saved['calibrated']
    model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                                  vector_refinement=True, candidate_ranking=True,
                                  fitted_ranking=True).cuda().eval()
    model.load_state_dict(saved['model'])
    models[step] = model

out.mkdir(parents=True, exist_ok=False)
config = {'steps': list(steps), 'panel_records_sha256': hashlib.sha256(
              (panel / 'records.jsonl').read_bytes()).hexdigest(),
          'train_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
          'checkpoints_sha256': {str(step): hashlib.sha256(
              (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
          'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('evaluate_joint_pose_fit_047.py', 'joint_pose_fit_047.py')},
          'scope': 'reused synthetic DEV identities; not unseen biological animals',
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
started = time.perf_counter()
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(eligible, 1):
        path = panel / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        ids = valid[8::16, 8::16].reshape(-1).nonzero().flatten()
        if not len(ids):
            continue
        truth = target[8::16, 8::16].reshape(256, 3)[ids]
        yx = torch.stack((8 + 16 * (ids // 16), 8 + 16 * (ids % 16)), -1)
        sample_grid = torch.stack((yx[:, 1], yx[:, 0]), -1).float()
        sample_grid = ((sample_grid + .5) * (2 / 256) - 1)[None, None].expand(4, 1, -1, -1)
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
               'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row['valid_query_cells'] = len(ids)
        for step, pose in models.items():
            prediction = pose.predict(image)
            fitted = joint_pose_fit(pose, coarse, fine, prediction, image, atlas, offsets, weights)
            mapped = fitted['mapped']['centre_surface_ccf_ap_dv_ml_um'][0]
            predicted = F.grid_sample(mapped.permute(0, 3, 1, 2), sample_grid,
                mode='bilinear', padding_mode='border', align_corners=False)
            predicted = predicted[:, :, 0].permute(0, 2, 1)
            mapped_error = (predicted - truth[None]).norm(dim=-1).mean(-1)
            state = fitted['fitted_state']
            centre, frame, basis = full_frame_state_to_components(state)
            chart = torch.stack((yx[:, 1] / 256, yx[:, 0] / 256), -1).float()
            chart = chart[None, None].expand(1, 4, -1, -1).clone()
            reflection = fitted['choice'] % 2
            chart[..., 0] = torch.where(reflection[..., None].bool(),
                                        255 / 256 - chart[..., 0], chart[..., 0])
            plane = centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
                frame[..., :, :2] @ basis, chart - .5)
            rigid_error = (plane[0] - truth[None]).norm(dim=-1).mean(-1)
            selected = int(fitted['score'][0].argmax())
            row[f'step{step}_selected_mapped_um'] = float(mapped_error[selected])
            row[f'step{step}_best4_mapped_um'] = float(mapped_error.min())
            row[f'step{step}_selected_rigid_um'] = float(rigid_error[selected])
            row[f'step{step}_best4_rigid_um'] = float(rigid_error.min())
            row[f'step{step}_prior_top1_fitted_mapped_um'] = float(mapped_error[0])
            row[f'step{step}_selected_branch'] = selected
            row[f'step{step}_match_residual_um'] = float(fitted['match_residual_um'][0, selected])
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number in (48, 96, 144, 177):
            stream.flush()
            print(json.dumps({'dev_sections': number,
                              'seconds': time.perf_counter() - started}), flush=True)

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'eligible_sections': len(rows), 'synthetic_subjects': len(identities),
           'seconds': time.perf_counter() - started,
           'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
for step in steps:
    for metric in ('selected_mapped_um', 'best4_mapped_um', 'selected_rigid_um',
                   'best4_rigid_um', 'prior_top1_fitted_mapped_um', 'match_residual_um'):
        field = f'step{step}_{metric}'
        summary[field] = float(np.mean([np.mean([row[field] for row in rows
            if row['synthetic_subject_plan_id'] == identity]) for identity in identities]))
    summary[f'step{step}_appearance_selected_mapped_um'] = {
        appearance: float(np.mean([np.mean([row[f'step{step}_selected_mapped_um']
            for row in rows if row['synthetic_subject_plan_id'] == identity and
            row['appearance_mode'] == appearance]) for identity in identities
            if any(row['synthetic_subject_plan_id'] == identity and
                   row['appearance_mode'] == appearance for row in rows)]))
        for appearance in ('raw', 'exact_black', 'imperfect_brush')}
summary['stage_promising'] = {str(step): bool(
    summary[f'step{step}_selected_mapped_um'] <= 1500 and
    summary[f'step{step}_selected_mapped_um'] <=
        .85 * summary['step0_selected_mapped_um'] and
    all(summary[f'step{step}_appearance_selected_mapped_um'][appearance] <=
        1.10 * summary['step0_appearance_selected_mapped_um'][appearance]
        for appearance in ('raw', 'exact_black', 'imperfect_brush')))
    for step in steps[1:]}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'eligible_sections': len(rows),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'stage_promising': summary['stage_promising'],
                  'seconds': time.perf_counter() - started}), flush=True)
