"""Frozen 025 patch matching on true rigid and 019 predicted atlas planes."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_physical_ouv
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025, atlas_surface_image, extract_patches

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
control = root / 'runs/one_shot_all_modes_023'
pose_path = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
patch_path = root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt'
out = root / 'runs/atlas_offplane_capture_026'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
controls = {row['sha256']: row for row in map(json.loads, (control / 'rows.jsonl').open())}
assert len(records) == len(controls) == 185
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
pose_saved = torch.load(pose_path, map_location='cpu', weights_only=True)
assert pose_saved['step'] == 18000 and not pose_saved['calibrated']
pose.load_state_dict(pose_saved['model'], strict=True)
patch = AtlasOrientedPatch025().cuda().eval()
patch_saved = torch.load(patch_path, map_location='cpu', weights_only=True)
assert patch_saved['step'] == 3000 and not patch_saved['calibrated']
patch.load_state_dict(patch_saved['model'], strict=True)
del pose_saved, patch_saved


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


out.mkdir(parents=True, exist_ok=False)
config = {'pose_checkpoint_sha256': sha(pose_path), 'patch_checkpoint_sha256': sha(patch_path),
          'panel_completed_sha256': sha(panel / 'completed.json'),
          'panel_records_sha256': sha(panel / 'records.jsonl'),
          'control_completed_sha256': sha(control / 'completed.json'),
          'control_rows_sha256': sha(control / 'rows.jsonl'),
          'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                            ('diagnose_atlas_offplane_capture_026.py',
                             'atlas_oriented_patch_025.py',
                             'arbitrary_plane_one_shot_model.py',
                             'arbitrary_plane_full_frame_primitives.py')},
          'cases': len(records), 'atlas_planes_per_case': 9, 'patches_per_plane': 288,
          'query_points_per_case': 32, 'local_radius_px': 64,
          'truth_usage': 'GT-valid query selection and error only; scores use no GT',
          'weights_updated': False, 'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
axis = np.arange(8, 256, 16)
gy, gx = np.meshgrid(axis, axis, indexing='ij')
regular = np.stack((gy.flatten(), gx.flatten()), -1)
y, x = torch.meshgrid(torch.arange(256, device='cuda') / 256,
                      torch.arange(256, device='cuda') / 256, indexing='ij')
started = time.perf_counter()
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            rng = np.random.default_rng(int(record['sha256'][:16], 16))
            chosen = rng.choice(np.flatnonzero(arrays['valid_mask'].reshape(-1)), 32, replace=False)
            query_xy = np.stack((chosen // 256, chosen % 256), -1)
            bank_xy = np.concatenate((regular, query_xy))
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][query_xy[:, 0],
                                                              query_xy[:, 1]].copy()).cuda()
            true_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            true_reflection = torch.from_numpy(arrays['reflection'][None].copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = pose.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        choice = prior.topk(8).indices
        control_row = controls[record['sha256']]
        assert choice.tolist() == control_row['top8']
        state = torch.cat((true_state, prediction['state'][:, choice // 2][0]), 0)
        reflection = torch.cat((true_reflection, choice % 2))
        ouv = full_frame_state_to_physical_ouv(state).reshape(9, 3, 3)
        chart_x = torch.where(reflection[:, None, None].bool(), 255 / 256 - x, x)
        centre = (ouv[:, 0, None, None] + chart_x[..., None] * ouv[:, 1, None, None]
                  + y[None, :, :, None] * ouv[:, 2, None, None])
        rendered = atlas_surface_image(atlas, centre, state,
                                       offsets.expand(9, -1), weights.expand(9, -1))
        query_xy_gpu = torch.from_numpy(query_xy.copy()).cuda().float()
        bank_xy_gpu = torch.from_numpy(bank_xy.copy()).cuda().float()
        query_patch = extract_patches(image, torch.zeros(32, device='cuda', dtype=torch.long),
                                      query_xy_gpu)
        query_feature = F.normalize(patch.shared(patch.image_stem(query_patch)), dim=-1)
        atlas_rows = torch.arange(9, device='cuda').repeat_interleave(288)
        atlas_xy = bank_xy_gpu.repeat(9, 1)
        features = []
        for start in range(0, len(atlas_rows), 64):
            cropped = extract_patches(rendered, atlas_rows[start:start + 64],
                                      atlas_xy[start:start + 64])
            features.append(F.normalize(patch.shared(patch.atlas_stem(cropped)), dim=-1))
        atlas_feature = torch.cat(features).reshape(9, 288, 128)
        similarity = torch.einsum('qc,bkc->bqk', query_feature, atlas_feature)
        locality = torch.cdist(query_xy_gpu[None], bank_xy_gpu[None])[0] <= 64
        local = similarity.masked_fill(~locality[None], -1e4)
        local_value, local_index = local.max(-1)
        score = local_value.mean(-1)
        selected = int(score[1:].argmax())
        candidate_ccf = centre[:, bank_xy_gpu[:, 0].long(), bank_xy_gpu[:, 1].long()]
        match_ccf = candidate_ccf.gather(1, local_index[..., None].expand(-1, -1, 3))
        match_distance = (match_ccf - truth[None]).norm(dim=-1).mean(-1)
        true_top = similarity[0].topk(16, -1).indices
        top_distance = (candidate_ccf[0, true_top] - truth[:, None]).norm(dim=-1)
        rigid = np.asarray(control_row['rigid_um'])[choice.cpu().numpy()]
        scores = score[1:].cpu().numpy()
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(choice=choice.tolist(), scores9=score.tolist(), rigid8_um=rigid.tolist(),
                   selected_index=selected, selected_rigid_um=float(rigid[selected]),
                   prior_rigid_um=float(rigid[0]), best8_rigid_um=float(rigid.min()),
                   true_win=bool(score[0] > score[1:].max()),
                   score_error_spearman=float(spearmanr(scores, -rigid).statistic),
                   true_top1_um=top_distance[:, 0].tolist(),
                   true_top16_um=top_distance.min(-1).values.tolist(),
                   true_local_matched_mean_um=float(match_distance[0]),
                   selected_local_matched_mean_um=float(match_distance[1 + selected]))
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 40 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'seconds': time.perf_counter() - started}), flush=True)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'sections': len(rows), 'synthetic_subjects': len(subjects)}
for field in ('true_win', 'score_error_spearman', 'selected_rigid_um', 'prior_rigid_um',
              'best8_rigid_um', 'true_local_matched_mean_um', 'selected_local_matched_mean_um'):
    summary[field] = float(np.mean([np.mean([row[field] for row in rows
        if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
for field in ('true_top1_um', 'true_top16_um'):
    summary[field.replace('_um', '_500um_recall')] = float(np.mean([
        np.mean([np.mean(np.asarray(row[field]) < 500) for row in rows
                 if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
summary['mode_true_win'] = {mode: float(np.mean([
    np.mean([row['true_win'] for row in rows if row['synthetic_subject_plan_id'] == subject
             and row['appearance_mode'] == mode]) for subject in subjects
    if any(row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode
           for row in rows)])) for mode in ('raw', 'exact_black', 'imperfect_brush')}
summary['promising_gate'] = (summary['true_win'] >= .75 and
    summary['score_error_spearman'] > 0 and
    summary['prior_rigid_um'] - summary['selected_rigid_um'] >= 250)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', 'summary': summary,
                  'seconds': time.perf_counter() - started}), flush=True)
