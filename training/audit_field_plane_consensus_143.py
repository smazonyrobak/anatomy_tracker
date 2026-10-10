"""Frozen 143 blind-beam correspondence consensus, with no truth in the fit."""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path('I:/AnatomyTracker')
out = root / 'runs/coherent_anatomy_field_143_plane_consensus_001'
os.environ['TEMP'] = os.environ['TMP'] = str(out / 'tmp')
os.environ['TORCH_HOME'] = str(out / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(out / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
pose_run = root / 'runs/v4_pose_adaptation_132'
pose_path = pose_run / 'joint_step_02000.pt'
pose_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
field_run = root / 'runs/coherent_anatomy_field_143'
field_eval = root / 'runs/coherent_anatomy_field_143_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('full', 'support_only')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

pose_done = json.loads((pose_run / 'completed.json').read_text())
pose_config = json.loads((pose_run / 'config.json').read_text())
pose_eval_done = json.loads((pose_eval / 'completed.json').read_text())
field_done = json.loads((field_run / 'completed.json').read_text())
field_config = json.loads((field_run / 'config.json').read_text())
field_eval_done = json.loads((field_eval / 'completed.json').read_text())
field_eval_config = json.loads((field_eval / 'config.json').read_text())
assert sha(pose_run / 'config.json') == pose_done['config_sha256']
assert sha(pose_path) == pose_done['checkpoint_sha256']['2000'] == field_config['parent_checkpoint_sha256']
assert sha(pose_path) == field_eval_config['parent_checkpoint_sha256']
assert sha(pose_run / 'completed.json') == field_config['parent_completion_sha256']
assert sha(pose_eval / 'completed.json') == field_eval_config['parent_evaluation_completion_sha256']
assert sha(pose_eval / 'synthetic_rows.jsonl') == pose_eval_done['output_sha256']['synthetic_rows.jsonl']
assert sha(source.parent / 'evaluate_v4_pose_adaptation_132.py') == pose_eval_done['evaluator_source_sha256']
assert sha(field_run / 'config.json') == field_done['config_sha256']
assert sha(field_run / 'draws.jsonl') == field_done['draws_sha256']
assert sha(field_run / 'training.jsonl') == field_done['training_sha256']
assert sha(field_run / 'completed.json') == field_eval_done['train_completion_sha256']
assert sha(field_eval / 'config.json') == field_eval_done['config_sha256']
assert sha(field_eval / 'rows.jsonl') == field_eval_done['rows_sha256']
assert sha(field_eval / 'summary.json') == field_eval_done['summary_sha256']
assert sha(source.parent / 'evaluate_coherent_anatomy_field_143.py') == field_eval_done['evaluator_sha256']
assert all(sha(source.parent / name) == digest for name, digest in field_config['source_sha256'].items())
assert all(sha(source.parent / name) == digest for name, digest in pose_config['source_sha256'].items())
assert field_config['radius'] == 6 and tuple(field_config['arms']) == arms
assert field_done['steps'] == field_config['steps'] == 10000
assert all(sha(field_run / f'{arm}_step_10000.pt') ==
           field_done['checkpoint_sha256'][arm]['10000'] for arm in arms)
assert not any(receipt.get(key, False) for receipt in
    (pose_done, pose_eval_done, field_done, field_eval_done) for key in
    ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
     'final_animals_used', 'external_pretrained_weights_used'))

pose_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (pose_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
near_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (field_eval / 'rows.jsonl').open())
    if row['step'] == 10000 and row['arm'] == 'full' and row['role'] == 'blind_near'}
records, panel_hashes = {}, {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    protocol = json.loads((panel / 'protocol.json').read_text())
    listed = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in protocol['source_sha256'].items())
    assert len(listed) == done['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in listed)
    assert all(row['provenance']['split'] == 'development' for row in listed)
    records[cohort] = [row for row in listed if row['eligible']]
    assert len(records[cohort]) == done['eligible'] == (248 if cohort == 'v4' else 243)
    panel_hashes[cohort] = sha(panel / 'completed.json')
assert len(pose_rows) == len(near_rows) == 491
assert {(cohort, row['section_id']) for cohort, panel in records.items()
        for row in panel} == set(pose_rows) == set(near_rows)
assert panel_hashes == field_eval_config['panel_receipt_sha256']
train_plans = {row['plan_receipt'] for row in field_config['synthetic_provenance']['base_subjects']}
dev_plans = {row['plan_receipt_sha256'] for panel in records.values() for row in panel}
assert len(train_plans) == 64 and len(dev_plans) == 8 and not train_plans & dev_plans

out.mkdir(parents=True, exist_ok=False)
for directory in ('tmp', 'cache/torch', 'cache/cuda'):
    (out / directory).mkdir(parents=True, exist_ok=True)

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(pose, enabled=True)
saved = torch.load(pose_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == pose_config
pose.load_state_dict(saved['model'], strict=True)
del saved
fields = {}
for arm in arms:
    saved = torch.load(field_run / f'{arm}_step_10000.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == 10000 and saved['arm'] == arm and saved['config'] == field_config
    fields[arm] = CoherentAnatomyField143(radius=6).cuda().eval().requires_grad_(False)
    fields[arm].load_state_dict(saved['field'], strict=True)
    del saved

axis64 = (np.arange(64) + .5) / 64
y64, x64 = np.meshgrid(axis64, axis64, indexing='ij')
chart_fit = np.stack((x64 - .5, y64 - .5, np.ones((64, 64))), -1).reshape(-1, 3)
axis256 = torch.arange(256, device='cuda') / 256
y256, x256 = torch.meshgrid(axis256, axis256, indexing='ij')
chart256 = torch.stack((x256, y256), -1)
middle = torch.tensor([[.5, .5]], device='cuda')

config = {'diagnostic_sha256': sha(source),
    'pose_completion_sha256': sha(pose_run / 'completed.json'),
    'pose_checkpoint_sha256': sha(pose_path),
    'pose_eval_completion_sha256': sha(pose_eval / 'completed.json'),
    'field_completion_sha256': sha(field_run / 'completed.json'),
    'field_eval_completion_sha256': sha(field_eval / 'completed.json'),
    'field_checkpoint_sha256': {arm: sha(field_run / f'{arm}_step_10000.pt') for arm in arms},
    'panel_receipt_sha256': panel_hashes,
    'cohorts': {cohort: len(panel) for cohort, panel in records.items()},
    'pose': 'frozen 132 truth-best branch of the original blind16; oracle diagnostic, not deployable selection',
    'correspondences': '143 continuous offset_chart converted by frozen 143 base, source basis, and normal to CCF micrometres; valid 64-grid sites with non-dustbin argmax',
    'fit': 'one 3D affine cutting plane in centered source chart coordinates; weighted least squares with visibility times maximum non-dustbin match probability, then 12 fixed Huber IRLS steps at 500 um; no synthetic truth or atlas score in fitting',
    'control': 'paired 143 support_only checkpoint with atlas intensity zeroed and identical atlas support, pose, source image, and fit rule; non-dustbin sites and confidence weights are arm-specific',
    'score': 'before and after mean Euclidean physical mapping error on every valid 256-grid pixel against synthetic target rigid state; normal angle and axial center distance use the same rigid truth only after fitting',
    'interpretation': 'a failed Huber IRLS fit does not exclude recovery by minority-consensus methods such as RANSAC',
    'split': '64 training deformation plans and eight disjoint synthetic DEV plans; no new training or benchmark',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        for record in records[cohort]:
            section_id = record['section_id']
            frozen = near_rows[cohort, section_id]
            parent = pose_rows[cohort, section_id]
            assert frozen['physical_section_id'] == parent['physical_section_id'] == record['panel_physical_section_id']
            assert parent['panel_file_sha256'] == record['sha256']
            branch = frozen['blind_near_branch_id']
            assert branch == parent['beam_branch_ids'][int(np.argmin(parent['beam_rigid_mm']))]
            assert abs(frozen['blind_near_rigid_mm'] - parent['beam_best_rigid_mm']) < 1e-4
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            assert int(valid.sum()) == record['valid_pixels']
            prediction = pose.predict(image)
            candidate_state = prediction['state'][:, branch // 2]
            candidate_reflection = torch.tensor([branch % 2], device='cuda')
            slabs, base, basis, normal = render_coherent_atlas_143(
                atlas, candidate_state, candidate_reflection, offsets, weights)
            section_fits = {}
            for arm in arms:
                atlas_input = slabs if arm == 'full' else torch.cat((
                    torch.zeros_like(slabs[:, :1]), slabs[:, 1:2]), 1)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = fields[arm](image, atlas_input, 500.,
                                         source_feature=prediction['feature'])
                delta = output['offset_chart'][0].permute(1, 2, 0).float()
                ccf = (base[0] + torch.einsum('ij,hwj->hwi', basis[0], delta[..., :2]) +
                       normal[0, None, None] * delta[..., 2:3]).cpu().numpy().reshape(-1, 3)
                matched = (output['match_logits'][0].argmax(0) !=
                           output['match_logits'].shape[1] - 1)
                selected = (valid[0, 2::4, 2::4] & matched).cpu().numpy().ravel()
                confidence = (output['visibility'][0, 0].float() *
                              output['confidence'][0, 0].float()).cpu().numpy().ravel()[selected]
                fitted = None
                residual = None
                if (selected.sum() >= 3 and confidence.sum() > 0 and
                        np.linalg.matrix_rank(chart_fit[selected]) == 3):
                    x, y = chart_fit[selected], ccf[selected]
                    w = confidence.copy()
                    for _ in range(12):
                        sw = np.sqrt(w)[:, None]
                        fitted = np.linalg.lstsq(x * sw, y * sw, rcond=None)[0]
                        residual = np.linalg.norm(x @ fitted - y, axis=1)
                        w = confidence * np.minimum(1., 500. / np.maximum(residual, 1.))
                    sw = np.sqrt(w)[:, None]
                    fitted = np.linalg.lstsq(x * sw, y * sw, rcond=None)[0]
                    residual = np.linalg.norm(x @ fitted - y, axis=1)
                section_fits[arm] = (fitted, int(selected.sum()),
                    float(np.median(residual)) if residual is not None else None,
                    float(np.sum(confidence * (residual <= 500)) / np.sum(confidence))
                    if residual is not None else None)

            observed = chart256[valid[0]].float()
            truth = rigid_points_090(state, reflection, observed)[0].cpu().numpy()
            prior = rigid_points_090(candidate_state, candidate_reflection, observed)[0].cpu().numpy()
            chart_score = np.column_stack((observed.cpu().numpy() - .5,
                                           np.ones(len(observed))))
            before_mm = float(np.linalg.norm(prior - truth, axis=1).mean() / 1000)
            assert abs(before_mm - parent['beam_best_rigid_mm']) < 1e-4
            true_middle = rigid_points_090(state, reflection, middle)[0, 0].cpu().numpy()
            prior_middle = rigid_points_090(candidate_state, candidate_reflection, middle)[0, 0].cpu().numpy()
            true_normal = full_frame_state_to_components(state)[1][0, :, 2].cpu().numpy()
            prior_normal = full_frame_state_to_components(candidate_state)[1][0, :, 2].cpu().numpy()
            before_angle = float(np.degrees(np.arccos(np.clip(abs(prior_normal @ true_normal), 0, 1))))
            before_axial = float(abs((prior_middle - true_middle) @ true_normal) / 1000)
            for arm in arms:
                fitted, sites, median, consensus = section_fits[arm]
                after_mm = float(np.linalg.norm(chart_score @ fitted - truth, axis=1).mean() / 1000) if fitted is not None else None
                fit_normal = np.cross(fitted[0], fitted[1]) if fitted is not None else None
                normal_length = np.linalg.norm(fit_normal) if fitted is not None else 0.
                after_angle = float(np.degrees(np.arccos(np.clip(
                    abs((fit_normal / normal_length) @ true_normal), 0, 1)))) if normal_length > 0 else None
                after_axial = float(abs((fitted[2] - true_middle) @ true_normal) / 1000) if fitted is not None else None
                row = {'cohort': cohort, 'section_id': section_id,
                    'physical_section_id': record['panel_physical_section_id'],
                    'subject_plan_id': record['synthetic_subject_plan_id'],
                    'panel_file_sha256': record['sha256'], 'arm': arm,
                    'blind_near_branch_id': branch,
                    'blind_near_available_1p5mm': frozen['blind_near_available_1p5mm'],
                    'matched_fit_sites': sites, 'fit_residual_median_um': median,
                    'weighted_consensus_le_500_fraction': consensus,
                    'plane_affine_um': fitted.tolist() if fitted is not None else None,
                    'before_rigid_mm': before_mm, 'after_rigid_mm': after_mm,
                    'gain_rigid_mm': before_mm - after_mm if after_mm is not None else None,
                    'before_normal_deg': before_angle, 'after_normal_deg': after_angle,
                    'before_axial_center_mm': before_axial,
                    'after_axial_center_mm': after_axial}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)

summary = {}
for cohort in panels:
    summary[cohort] = {}
    for arm in arms:
        summary[cohort][arm] = {}
        for scope in ('all', 'blind_near_le_1p5mm'):
            group = [row for row in rows if row['cohort'] == cohort and row['arm'] == arm and
                     (scope == 'all' or row['blind_near_available_1p5mm'])]
            fitted = [row for row in group if row['after_rigid_mm'] is not None]
            plans = defaultdict(list)
            for row in fitted:
                plans[row['subject_plan_id']].append(row)
            normals = [row['after_normal_deg'] for row in fitted
                       if row['after_normal_deg'] is not None]
            summary[cohort][arm][scope] = {
                'sections': len(group), 'fitted_sections': len(fitted), 'plans': len(plans),
                'before_rigid_mm_plan_equal': float(np.mean([np.mean([r['before_rigid_mm']
                    for r in plan]) for plan in plans.values()])) if plans else None,
                'after_rigid_mm_plan_equal': float(np.mean([np.mean([r['after_rigid_mm']
                    for r in plan]) for plan in plans.values()])) if plans else None,
                'gain_rigid_mm_plan_equal': float(np.mean([np.mean([r['gain_rigid_mm']
                    for r in plan]) for plan in plans.values()])) if plans else None,
                'improved_sections': sum(r['gain_rigid_mm'] > 0 for r in fitted),
                'after_le_500um_sections': sum(r['after_rigid_mm'] <= .5 for r in fitted),
                'before_normal_deg_median': float(np.median([r['before_normal_deg']
                    for r in fitted])) if fitted else None,
                'after_normal_deg_median': float(np.median(normals)) if normals else None,
                'before_axial_center_mm_median': float(np.median([r['before_axial_center_mm']
                    for r in fitted])) if fitted else None,
                'after_axial_center_mm_median': float(np.median([r['after_axial_center_mm']
                    for r in fitted])) if fitted else None}
assert len(rows) == 2 * 491
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'diagnostic_sha256': sha(source), 'calibrated': False,
    'public_benchmark_used': False, 'expert_real_truth_used': False,
    'final_animals_used': False}, indent=2))
