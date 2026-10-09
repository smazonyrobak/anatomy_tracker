"""Frozen internal DEV readout for joint pose and map training; no test claim."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match
from training.joint_pose_map_121 import joint_fit_loss_121

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_POSE_MAP_122_PROTOCOL_20261009.md'
run = root / 'runs/joint_pose_map_122'
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
assay118 = root / 'runs/v3_interior_intensity_energy_118_dev'
eval120 = root / 'runs/global_plane_matcher_120_dev_eval'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
reserved = root / 'data/joint_v7_reserved_train_images_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/joint_pose_map_122_dev_eval'
steps = (0, 500, 1000, 2000)
side = 256
map_side = 96
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate(rows, keys, unit):
    by_plan = {plan: [row for row in rows if row['synthetic_subject_plan_id'] == plan]
               for plan in plans}
    by_plan = {plan: group for plan, group in by_plan.items() if group}
    return {'sections': len(rows), 'plans': len(by_plan), 'unit': unit,
            'by_plan': {plan: {'sections': len(group), 'mean': {
                key: float(np.mean([row[key] for row in group])) for key in keys}}
                for plan, group in by_plan.items()},
            'mean': {key: float(np.mean([row[key] for row in rows])) if rows else None
                     for key in keys},
            'median': {key: float(np.median([row[key] for row in rows])) if rows else None
                       for key in keys},
            'p90': {key: float(np.quantile([row[key] for row in rows], .9)) if rows else None
                    for key in keys},
            'plan_equal': {key: float(np.mean([np.mean([row[key] for row in group])
                         for group in by_plan.values()])) if by_plan else None
                           for key in keys}}


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert tuple(config['checkpoints']) == steps and config['blind_count'] == 16
assert config['updates'] == done['updates'] == 2000
assert done['accepted_synthetic'] == config['synthetic_presentations'] == 4000
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(parent) == parent_done['checkpoint_sha256']['1959'] == config['parent_checkpoint_sha256']
assert sha(protocol) == config['protocol_sha256']
assert done['source_sha256'] == config['source_sha256']
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
checkpoints = {step: run / f'joint_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == done['checkpoint_sha256'][str(step)]
           for step, path in checkpoints.items())
assert not any(done.get(key, False) or config.get(key, False) for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used', 'final_animals_used'))

panel_done = json.loads((panel / 'completed.json').read_text())
assay_done = json.loads((assay118 / 'completed.json').read_text())
old120 = json.loads((eval120 / 'selection.json').read_text())
eval120_done = json.loads((eval120 / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(assay118 / 'rows.jsonl') == assay_done['rows_sha256']
assert sha(eval120 / 'selection.json') == eval120_done['selection_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
old118_ids = {json.loads(line)['section_id'] for line in (assay118 / 'rows.jsonl').open()}
old120_ids = set(old120['selected_section_ids'])
assert len(old118_ids) == len(old120_ids) == 64 and not old118_ids & old120_ids
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plans) == 8
selected = [row for plan in plans for row in sorted((row for row in records
    if row['eligible'] and row['synthetic_subject_plan_id'] == plan
    and row['section_id'] not in old118_ids | old120_ids),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == len({row['section_id'] for row in selected}) == 64
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in selected) == 8
           for plan in plans)
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
diagnostic_ids = {row['section_id'] for plan in plans for row in
                  [item for item in selected if item['synthetic_subject_plan_id'] == plan][:2]}

draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
synthetic_draws = [row for row in draws if 'physical_section_id' in row and row['used']]
assert synthetic_draws and all(row['base_lineage']['split'] == 'train' for row in synthetic_draws)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in selected} & {
        row['base_lineage'][key] for row in synthetic_draws}
assert not {row['panel_physical_section_id'] for row in selected} & {
    row['physical_section_id'] for row in synthetic_draws}

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert coronal_done['development_images'] == 64 and coronal_done['development_donors'] == 6
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_records = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
                   if row['training_split'] == 'development']
coronal_train = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
                 if row['training_split'] == 'train']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {row['animal_id'] for row in coronal_records} & {
    row['animal_id'] for row in coronal_train}
reserved_done = json.loads((reserved / 'completed.json').read_text())
assert reserved_done['training_role'] == 'train_only'
assert config['coronal_bindings'][str(reserved / 'completed.json')] == sha(reserved / 'completed.json')
assert not {str(row['animal_id']) for row in coronal_records} & set(reserved_done['donor_receipts'])
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    coronal_thickness_um = arrays['thickness_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sagittal_summary['roles']['weak_dev2_source_available'] == {'donors': 8, 'sections': 158}
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in
                          (sagittal_train / 'geometry.jsonl').open()]
assert sha(sagittal_train / 'summary.json') == config['sagittal_summary_sha256']
assert sha(sagittal_train / 'geometry.jsonl') == json.loads(
    (sagittal_train / 'summary.json').read_text())['output_sha256']['geometry.jsonl']
assert all(row['split'] == 'train' for row in sagittal_train_records)
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
atlas = context['atlas']
axis = torch.arange(side, device='cuda', dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device='cuda')
five_chart = five_pixels / side
unit_offsets = torch.linspace(-.5, .5, 9, device='cuda')[None]
real_weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device='cuda') / 16
near_delta = torch.tensor([[.10, -.08, .06, 500., -400., 350., .03, -.02, .02]],
                          device='cuda')
synthetic_rows, real_rows, fit_rows = [], [], []

with torch.inference_mode():
    for label, step, path in [('parent111', -1, parent)] + [
            ('joint122', step, checkpoints[step]) for step in steps]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == (1959 if step < 0 else step)
        if step >= 0:
            assert checkpoint['config'] == config and checkpoint['calibrated'] is False
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
            atlas_conditioning=True, fit_quality=True, vector_refinement=True,
            candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
        if step >= 0:
            attach_global_plane_matcher(model, enabled=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint

        for record in selected:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            assert int(valid.sum()) == record['valid_pixels']
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            for _ in range(2):
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity = -similarity
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            mode, reflection = beam // 2, beam % 2
            match = global_plane_match(model, prediction, mode, reflection,
                offsets, weights, atlas, (side, side), side=24)
            scores = torch.stack((match['input_score'], match['score']), -1)
            choice = int(scores[0, :14, 0].argmax()) * 2 if step < 0 else int(scores.flatten().argmax())
            slot, action = divmod(choice, 2)
            state = match['input_state'] if action == 0 else match['state']
            selected_state = state[:, slot:slot + 1]
            selected_reflection = reflection[:, slot:slot + 1]
            target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
            input_error = (rigid_points_090(match['input_state'], reflection, chart)[0]
                           - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
            corrected_error = (rigid_points_090(match['state'], reflection, chart)[0]
                               - target_rigid).norm(dim=-1)[:, valid[0].flatten()].mean(-1) / 1000
            oracle16 = torch.stack((input_error, corrected_error), -1)
            oracle_slot, oracle_action = divmod(int(oracle16.flatten().argmin()), 2)
            selected_rigid = float((input_error if action == 0 else corrected_error)[slot])
            selected_prediction = {**prediction, 'state': selected_state}
            mapped = model.map(selected_prediction, offsets, torch.zeros((1, 1),
                device='cuda', dtype=torch.long), selected_reflection,
                (map_side, map_side), atlas, weights, source_shape=(side, side))
            surface = F.interpolate(mapped['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                .permute(2, 0, 1)[None], (side, side), mode='bilinear', align_corners=False)
            mapped_error = float((surface[0].permute(1, 2, 0) - centre[0])
                                 .norm(dim=-1)[valid[0]].mean() / 1000)
            given_errors = {}
            for name, given in (('exact', truth),
                                ('near', compose_full_frame_state(truth, near_delta))):
                given_prediction = {**prediction, 'state': given[:, None]}
                given_map = model.map(given_prediction, offsets, torch.zeros((1, 1),
                    device='cuda', dtype=torch.long), truth_reflection[:, None],
                    (map_side, map_side), atlas, weights, source_shape=(side, side))
                given_surface = F.interpolate(given_map['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                    .permute(2, 0, 1)[None], (side, side), mode='bilinear', align_corners=False)
                given_errors[name] = float((given_surface[0].permute(1, 2, 0) - centre[0])
                                           .norm(dim=-1)[valid[0]].mean() / 1000)
            normal = np.abs(record['plane_normal_ap_dv_ml'])
            angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
            appearance = record['provenance']['appearance']
            artifact = record['provenance']['one_shot_slide_artifacts_v3']
            synthetic_rows.append({'model': label, 'step': step, 'section_id': record['section_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': record['appearance_mode'],
                'damage': bool(appearance['damage']),
                'artifact_any': any(artifact['events'].values()),
                'raw_exterior_code': int(artifact['parameters']['raw_exterior_code']),
                'nearest_axis': ('AP', 'DV', 'ML')[int(np.argmax(normal))],
                'angle_deg': angle, 'angle_bin': ('0-15' if angle < 15 else '15-30'
                    if angle < 30 else '30-45' if angle < 45 else '45-54.7'),
                'valid_pixels': record['valid_pixels'], 'beam_branch_ids': beam[0].tolist(),
                'selected_slot': slot, 'selected_action': 'original' if action == 0 else 'corrected',
                'oracle16_slot': oracle_slot,
                'oracle16_action': 'original' if oracle_action == 0 else 'corrected',
                'selected_rigid_gauge_mm': selected_rigid,
                'selected_warped_ccf_mm': mapped_error,
                'best14_original_rigid_gauge_mm': float(input_error[:14].min()),
                'best16_action_rigid_gauge_mm': float(oracle16.min()),
                'exact_given_warped_ccf_mm': given_errors['exact'],
                'near_given_warped_ccf_mm': given_errors['near'],
                'panel_file_sha256': record['sha256']})

            if step >= 0 and record['section_id'] in diagnostic_ids:
                all_prediction = {**prediction, 'state': match['state']}
                all_map = model.map(all_prediction, offsets, torch.arange(16,
                    device='cuda')[None], reflection, (map_side, map_side), atlas,
                    weights, source_shape=(side, side))
                fit = joint_fit_loss_121(image, all_map, atlas, weights, valid, side=64)
                warp_rms = (all_map['local_displacement_um'][0] / 1000).square().sum(1).mean((1, 2)).sqrt()
                best = int(corrected_error.argmin())
                coverage = fit['atlas_coverage'][0]
                eligible = (corrected_error > corrected_error[best] + 1) & (
                    (coverage - coverage[best]).abs() <= .10)
                wrong = int(torch.where(eligible, fit['fit_loss'][0],
                    torch.inf).argmin()) if bool(eligible.any()) else None
                fit_rows.append({'step': step, 'section_id': record['section_id'],
                    'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                    'best_slot': best, 'best_rigid_gauge_mm': float(corrected_error[best]),
                    'best_fit': float(fit['fit_loss'][0, best]),
                    'best_coverage': float(coverage[best]), 'wrong_slot': wrong,
                    'best_warp_rms_mm': float(warp_rms[best]),
                    'wrong_rigid_gauge_mm': float(corrected_error[wrong]) if wrong is not None else None,
                    'wrong_fit': float(fit['fit_loss'][0, wrong]) if wrong is not None else None,
                    'wrong_coverage': float(coverage[wrong]) if wrong is not None else None,
                    'wrong_warp_rms_mm': float(warp_rms[wrong]) if wrong is not None else None,
                    'wrong_better_fit': bool(fit['fit_loss'][0, wrong] < fit['fit_loss'][0, best])
                        if wrong is not None else None})

        for family, real_records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
            for record in real_records:
                index = record['array_row_index']
                if family == 'coronal':
                    native = np.concatenate((np.asarray(coronal_images[index], dtype=np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                        mode='bilinear', align_corners=False)
                    affine = torch.as_tensor(coronal_affines[index], device='cuda',
                                             dtype=torch.float32)
                    reference = affine[:, 2] + (192 / side) * (
                        five_pixels[:, :1] * affine[:, 0] + five_pixels[:, 1:] * affine[:, 1])
                    donor = record['animal_id']
                else:
                    native = np.concatenate((np.asarray(sagittal_images[index], dtype=np.float32),
                        np.zeros((4, side, side), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                             device='cuda', dtype=torch.float32)
                    reference = affine[:, 2] + (five_pixels[:, :1] * affine[:, 0]
                                                + five_pixels[:, 1:] * affine[:, 1])
                    donor = record['donor_id']
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                                  prior[:, 32:].topk(6, -1).indices + 32), -1)
                normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                for _ in range(2):
                    chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                    similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                    diversity = -similarity
                    diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                    anchor = diversity.argmax(-1)
                    reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                    beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
                mode, reflection = beam // 2, beam % 2
                real_offsets = unit_offsets * (float(coronal_thickness_um[index])
                    if family == 'coronal' else 62.5)
                match = global_plane_match(model, prediction, mode, reflection,
                    real_offsets, real_weights, atlas, (side, side), side=24)
                scores = torch.stack((match['input_score'], match['score']), -1)
                choice = int(scores[0, :14, 0].argmax()) * 2 if step < 0 else int(scores.flatten().argmax())
                slot, action = divmod(choice, 2)
                state = (match['input_state'] if action == 0 else match['state'])[:, slot]
                proposed = rigid_points_090(state, reflection[:, slot], five_chart)[0]
                weak_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                predicted_normal = full_frame_state_to_components(state)[1][0, :, 2]
                normal_deg = torch.rad2deg((predicted_normal @ weak_normal).abs().clamp(0, 1).acos())
                real_rows.append({'model': label, 'step': step, 'family': family,
                    'donor_id': donor, 'section_id': record['section_id'],
                    'selected_action': 'original' if action == 0 else 'corrected',
                    'selected_weak_five_point_mm': float((proposed - reference).norm(dim=-1).mean() / 1000),
                    'selected_weak_normal_deg': float(normal_deg),
                    'label_role': 'inherited weak Allen affine, not expert or biological truth'})
        del model
        torch.cuda.empty_cache()

assert len(synthetic_rows) == 64 * (len(steps) + 1)
assert len(real_rows) == (64 + 158) * (len(steps) + 1)
assert len(fit_rows) == 16 * len(steps)
metric_keys = ('selected_rigid_gauge_mm', 'selected_warped_ccf_mm',
    'best14_original_rigid_gauge_mm', 'best16_action_rigid_gauge_mm',
    'exact_given_warped_ccf_mm', 'near_given_warped_ccf_mm')
summary = {'scope': 'frozen internal synthetic DEV; not biological validation or final test',
    'frozen_panel_provenance_note': 'Panel records retain an old pose_gauge prose string; target_state is the pre-v3-artifact affine CCF fit to the virtual-subject warped field, not the pristine virtual cutting plane. Pixel arrays and hashes are unchanged.',
    'metric_contract': 'all valid original 256x256 pixels; rigid against observed affine gauge; '
        'mapped 96x96 surface bilinearly sampled to original pixels against warped CCF',
    'blind_selection': 'each checkpoint independently applies the same frozen 14-prior-plus-2-'
        'normal-diverse rule; maximum of original-prior or corrected-prior-plus-match-logit, '
        'no truth/valid mask/fit in selection',
    'synthetic': {}, 'real_weak_affine': {}, 'fit_shortcut_diagnostic': {}}
for label, step in [('parent111', -1)] + [('joint122', step) for step in steps]:
    rows = [row for row in synthetic_rows if row['step'] == step]
    groups = {'all_plane': rows,
        'angle/steep_ge_30': [row for row in rows if row['angle_deg'] >= 30]}
    for appearance in ('raw', 'exact_black', 'imperfect_brush'):
        groups[f'appearance/{appearance}'] = [row for row in rows
            if row['appearance_mode'] == appearance]
    for flag in (False, True):
        groups[f'damage/{flag}'] = [row for row in rows if row['damage'] == flag]
        groups[f'artifact_any/{flag}'] = [row for row in rows if row['artifact_any'] == flag]
    for code in (0, 1, 2):
        groups[f'raw_exterior_code/{code}'] = [row for row in rows
            if row['appearance_mode'] == 'raw' and row['raw_exterior_code'] == code]
    for axis_name in ('AP', 'DV', 'ML'):
        groups[f'nearest_axis/{axis_name}'] = [row for row in rows
            if row['nearest_axis'] == axis_name]
    for angle_bin in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'angle/{angle_bin}'] = [row for row in rows
            if row['angle_bin'] == angle_bin]
    summary['synthetic'][str(step)] = {'groups': {name: aggregate(group, metric_keys, 'mm')
        for name, group in groups.items()},
        'corrected_action_fraction': float(np.mean([row['selected_action'] == 'corrected'
            for row in rows]))}
    summary['real_weak_affine'][str(step)] = {}
    for family in ('coronal', 'sagittal'):
        group = [row for row in real_rows if row['step'] == step and row['family'] == family]
        donors = sorted({row['donor_id'] for row in group})
        keys = ('selected_weak_five_point_mm', 'selected_weak_normal_deg')
        by_donor = {str(donor): {'sections': len(rows_donor), **{key: float(np.mean(
            [row[key] for row in rows_donor])) for key in keys}}
            for donor in donors if (rows_donor := [row for row in group
                if row['donor_id'] == donor])}
        summary['real_weak_affine'][str(step)][family] = {
            'sections': len(group), 'donors': len(donors), 'by_donor': by_donor,
            'donor_equal': {key: float(np.mean([item[key] for item in by_donor.values()]))
                            for key in keys},
            'label_role': 'inherited weak Allen affine only; not expert truth'}

for step in steps:
    group = [row for row in fit_rows if row['step'] == step]
    matched = [row for row in group if row['wrong_slot'] is not None]
    summary['fit_shortcut_diagnostic'][str(step)] = {
        'sections': len(group), 'support_matched_pairs': len(matched),
        'skipped_no_support_matched_wrong': len(group) - len(matched),
        'best_fit_mean': float(np.mean([row['best_fit'] for row in matched])) if matched else None,
        'wrong_fit_mean': float(np.mean([row['wrong_fit'] for row in matched])) if matched else None,
        'best_coverage_mean': float(np.mean([row['best_coverage'] for row in matched])) if matched else None,
        'wrong_coverage_mean': float(np.mean([row['wrong_coverage'] for row in matched])) if matched else None,
        'best_warp_rms_mm_mean': float(np.mean([row['best_warp_rms_mm'] for row in matched])) if matched else None,
        'wrong_warp_rms_mm_mean': float(np.mean([row['wrong_warp_rms_mm'] for row in matched])) if matched else None,
        'wrong_better_fit_fraction': float(np.mean([row['wrong_better_fit'] for row in matched]))
            if matched else None,
        'diagnostic_only': True}

parent_by_section = {row['section_id']: row for row in synthetic_rows if row['step'] == -1}
for step in steps:
    rows = [row for row in synthetic_rows if row['step'] == step]
    paired = {key: np.array([row[key] - parent_by_section[row['section_id']][key]
        for row in rows]) for key in ('selected_rigid_gauge_mm', 'selected_warped_ccf_mm')}
    summary['synthetic'][str(step)]['paired_vs_parent111'] = {key: {
        'mean_delta_mm': float(values.mean()), 'median_delta_mm': float(np.median(values)),
        'p90_delta_mm': float(np.quantile(values, .9)),
        'fraction_improved': float(np.mean(values < 0)),
        'worsened_gt_0p2_mm': int(np.sum(values > .2))}
        for key, values in paired.items()}

parent_metric = summary['synthetic']['-1']['groups']['all_plane']['plan_equal']
initial_metric = summary['synthetic']['0']['groups']['all_plane']['plan_equal']
parent_real = summary['real_weak_affine']['-1']
initial_real = summary['real_weak_affine']['0']
summary['early_real_retention'] = {}
for step in (500, 1000):
    real = summary['real_weak_affine'][str(step)]
    conditions = {family: all(
        real[family]['donor_equal']['selected_weak_five_point_mm'] <=
        reference[family]['donor_equal']['selected_weak_five_point_mm'] + .20
        for reference in (parent_real, initial_real))
        for family in ('coronal', 'sagittal')}
    summary['early_real_retention'][str(step)] = {
        'conditions': conditions, 'passed': all(conditions.values()),
        'pre_fit_checkpoint': True, 'development_only': True}
summary['advancement_gates'] = {}
for step in (2000,):
    metric = summary['synthetic'][str(step)]['groups']['all_plane']['plan_equal']
    real = summary['real_weak_affine'][str(step)]
    gates = {
        'selected_rigid_gain_ge_0p30_vs_parent_and_step0': all(
            reference['selected_rigid_gauge_mm'] - metric['selected_rigid_gauge_mm'] >= .30
            for reference in (parent_metric, initial_metric)),
        'selected_mapped_gain_ge_0p20_vs_parent_and_step0': all(
            reference['selected_warped_ccf_mm'] - metric['selected_warped_ccf_mm'] >= .20
            for reference in (parent_metric, initial_metric)),
        'exact_near_map_retention_le_0p20_vs_parent_and_step0': all(
            metric[key] <= reference[key] + .20 for key in
            ('exact_given_warped_ccf_mm', 'near_given_warped_ccf_mm')
            for reference in (parent_metric, initial_metric)),
        'coronal_weak_nonregression_le_0p20': all(
            real['coronal']['donor_equal']['selected_weak_five_point_mm'] <=
            reference['coronal']['donor_equal']['selected_weak_five_point_mm'] + .20
            for reference in (parent_real, initial_real)),
        'sagittal_weak_nonregression_le_0p20': all(
            real['sagittal']['donor_equal']['selected_weak_five_point_mm'] <=
            reference['sagittal']['donor_equal']['selected_weak_five_point_mm'] + .20
            for reference in (parent_real, initial_real))}
    summary['advancement_gates'][str(step)] = {'conditions': gates,
        'passed': all(gates.values()), 'development_only': True}
passing = [step for step in (2000,) if summary['advancement_gates'][str(step)]['passed']]
summary['development_choice'] = min(passing, key=lambda step: (
    summary['synthetic'][str(step)]['groups']['all_plane']['plan_equal']
    ['selected_rigid_gauge_mm'], step)) if passing else None
summary['unmeasured_protocol_diagnostics'] = ['calibrated uncertainty',
    'expert-animal anatomical truth']

selection = {'rule': '8 eligible per frozen synthetic plan by SHA256(section_id) after excluding '
    'all 118 and 120 selected section IDs',
    'selected_section_ids': [row['section_id'] for row in selected],
    'fit_diagnostic_section_ids': sorted(diagnostic_ids),
    'assay118_rows_sha256': sha(assay118 / 'rows.jsonl'),
    'eval120_selection_sha256': sha(eval120 / 'selection.json')}
output_config = {'steps': steps, 'parent_checkpoint_sha256': sha(parent),
    'run_completion_sha256': sha(run / 'completed.json'),
    'run_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoints.items()},
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_train_geometry_sha256': sha(sagittal_train / 'geometry.jsonl'),
    'protocol_sha256': sha(protocol), 'source_sha256': sha(__file__),
    'map_side': map_side, 'near_delta': near_delta[0].tolist(),
    'real_psf': 'coronal per-section thickness_um; sagittal assumed 62.5 um '
        'because frozen sagittal DEV metadata has no section thickness',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
for name, data in (('selection.json', selection), ('config.json', output_config),
                   ('summary.json', summary)):
    (out / name).write_text(json.dumps(data, indent=2, allow_nan=False))
for name, rows in (('synthetic_rows.jsonl', synthetic_rows),
                   ('real_weak_rows.jsonl', real_rows), ('fit_diagnostic_rows.jsonl', fit_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'sections_per_model': 64,
    'real_sections_per_model': 222, 'fit_sections_per_joint_checkpoint': 16,
    'output_sha256': {name: sha(out / name) for name in ('selection.json', 'config.json',
        'summary.json', 'synthetic_rows.jsonl', 'real_weak_rows.jsonl',
        'fit_diagnostic_rows.jsonl')},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'completed': True, 'development_choice': summary['development_choice'],
                  'advancement_gates': summary['advancement_gates']}, allow_nan=False), flush=True)
