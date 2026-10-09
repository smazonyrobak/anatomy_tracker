"""Matched direct-pose DEV readout for the coronal-risk 111 checkpoint."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def synthetic_aggregate(rows, plans):
    metrics = ('best14_rigid_error_um', 'prior_selected_rigid_error_um')
    by_plan = {plan: {'sections': len(group), **{key: float(np.mean(
        [row[key] for row in group])) for key in metrics}}
        for plan in plans if (group := [row for row in rows
            if row['synthetic_subject_plan_id'] == plan])}
    return {'sections': len(rows), 'plans_present': len(by_plan), 'by_plan': by_plan,
        'plan_equal_um': {key: float(np.mean([item[key] for item in by_plan.values()]))
            for key in metrics} if by_plan else {}}


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_MIXED_REAL_POSE_CORONAL_RISK_111_PROTOCOL_20261009.md'
parent_run = root / 'runs/v3_mixed_real_pose_continuation_108'
parent_file = parent_run / 'joint_step_00653.pt'
run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
baseline_eval = root / 'runs/v3_mixed_real_pose_continuation_108_dev_eval'
heads_eval = root / 'runs/v3_mixed_real_pose_heads_only_109_dev_eval'
retention_eval = root / 'runs/v3_mixed_real_pose_coronal_retention_110_dev_eval'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
plans = root / 'data/v3_pose_capture_confirmation_plans_001'
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
reserved = root / 'data/joint_v7_reserved_train_images_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
out = root / 'runs/v3_mixed_real_pose_coronal_risk_111_dev_eval'
steps, side = (1959,), 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
assert (run / 'completed.json').is_file(), '111 training must complete before evaluation'
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert done['updates'] == config['updates'] == 1959
assert done['initial_step'] == config['initial_step'] == 653
assert done['completed_updates'] == config['completed_updates'] == 1959 - 653
assert done['accepted_synthetic'] == 2 * (1959 - 653)
assert done['real_coronal_presentations'] == done['real_sagittal_presentations'] == 1959 - 653
assert done['shared_image_features_unchanged'] is False
assert done['shared_image_features_trainable'] is True
assert done['retention_coefficient'] == config['retention_coefficient'] == 2.0
assert done['risk_coefficient'] == config['risk_coefficient'] == 2.0
assert done['risk_hinge_mm'] == config['risk_hinge_mm'] == 1.0
assert config['retention_scope'] == 'coronal TRAIN only; teacher frozen from 108 step653'
assert done['teacher_checkpoint_sha256'] == sha(parent_file)
assert 653 in config['checkpoints'] and 1959 in config['checkpoints']
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert done['draws_sha256'] == config['expected_stage2_draws_sha256']
assert sha(run / 'training.jsonl') == done['training_sha256']
assert sha(protocol) == config['protocol_sha256'] == done['protocol_sha256']
assert done['schedule_sha256'] == config['schedule_sha256']
assert all(sha(run / name) == digest for name, digest in config['schedule_sha256'].items())
assert done['source_sha256'] == config['source_sha256']
for name, digest in config['source_sha256'].items():
    path = Path(name)
    assert sha(path if path.is_absolute() else source / path) == digest, name
assert not any(done.get(key, False) or config.get(key, False) for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used', 'final_animals_used'))
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert sha(parent_run / 'completed.json') == config['parent_completion_sha256']
assert sha(parent_run / 'config.json') == config['parent_config_sha256']
assert sha(parent_run / 'draws.jsonl') == config['parent_draws_sha256']
assert done['schedule_sha256'] == parent_done['schedule_sha256']
assert sha(parent_file) == parent_done['checkpoint_sha256']['653'] == \
    config['parent_checkpoint_sha256'] == done['parent_checkpoint_sha256']
checkpoints = {step: run / f'joint_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == done['checkpoint_sha256'][str(step)]
    for step, path in checkpoints.items())
assert sha(run / 'joint_step_00653.pt') == done['checkpoint_sha256']['653']
baseline_done = json.loads((baseline_eval / 'completed.json').read_text())
for name in ('config.json', 'sections.jsonl', 'real_rows.jsonl', 'summary.json'):
    assert sha(baseline_eval / name) == baseline_done[name.split('.')[0] + '_sha256']
baseline_config = json.loads((baseline_eval / 'config.json').read_text())
baseline_sections = [json.loads(line) for line in (baseline_eval / 'sections.jsonl').open()]
baseline_real = [json.loads(line) for line in (baseline_eval / 'real_rows.jsonl').open()]
assert len(baseline_sections) == 3 * 64 and len(baseline_real) == 3 * (64 + 158)
heads_done = json.loads((heads_eval / 'completed.json').read_text())
for name in ('config.json', 'sections.jsonl', 'real_rows.jsonl', 'summary.json'):
    assert sha(heads_eval / name) == heads_done[name.split('.')[0] + '_sha256']
heads_config = json.loads((heads_eval / 'config.json').read_text())
heads_sections = [json.loads(line) for line in (heads_eval / 'sections.jsonl').open()]
heads_real = [json.loads(line) for line in (heads_eval / 'real_rows.jsonl').open()]
assert len(heads_sections) == 64 and len(heads_real) == 64 + 158
assert heads_config['baseline_108_eval_completion_sha256'] == sha(baseline_eval / 'completed.json')
retention_done = json.loads((retention_eval / 'completed.json').read_text())
for name in ('config.json', 'sections.jsonl', 'real_rows.jsonl', 'summary.json'):
    assert sha(retention_eval / name) == retention_done[name.split('.')[0] + '_sha256']
retention_sections = [json.loads(line) for line in (retention_eval / 'sections.jsonl').open()]
retention_real = [json.loads(line) for line in (retention_eval / 'real_rows.jsonl').open()]
assert len(retention_sections) == 64 and len(retention_real) == 64 + 158
retention_config = json.loads((retention_eval / 'config.json').read_text())
assert retention_config['baseline_108_eval_completion_sha256'] == sha(baseline_eval / 'completed.json')
assert retention_config['baseline_109_eval_completion_sha256'] == sha(heads_eval / 'completed.json')

panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_done['physical_sections'] == 256 and panel_done['eligible'] == 247
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(plans / 'completed.json') == panel_done['plan_completed_sha256'] == \
    panel_protocol['source_plan_completed_sha256']
all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(all_records) == len({row['section_id'] for row in all_records}) == 256
eligible = [row for row in all_records if row['eligible']]
assert len(eligible) == 247
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in all_records})
assert len(plan_ids) == 8
selected = [row for plan in plan_ids for row in sorted((row for row in eligible
    if row['synthetic_subject_plan_id'] == plan),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == 64 and all(sha(panel / row['file']) == row['sha256']
    for row in selected)
assert hashlib.sha256(json.dumps([row['section_id'] for row in selected]).encode()
    ).hexdigest() == baseline_config['selected_section_ids_sha256']
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
synthetic_draws = [row for row in draws
    if row['kind'] == 'synthetic_train' and row['used']]
assert len(synthetic_draws) == 2 * (1959 - 653)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in all_records} & \
        {row['base_lineage'][key] for row in synthetic_draws}
assert not {row['panel_physical_section_id'] for row in all_records} & \
    {row['physical_section_id'] for row in synthetic_draws}

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert coronal_done['development_images'] == 64 and coronal_done['development_donors'] == 6
for name in ('images.npy', 'geometry.npz', 'records.jsonl'):
    assert sha(coronal / name) == coronal_done['output_sha256'][name]
coronal_records = [row for row in map(json.loads, (coronal / 'records.jsonl').open())
    if row['training_split'] == 'development']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
coronal_draws = [row for row in draws if row['kind'] == 'coronal_weak_train']
assert len(coronal_draws) == 1959 - 653
for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id'):
    assert not {row[key] for row in coronal_records} & \
        {row[key] for row in coronal_draws}
reserved_done = json.loads((reserved / 'completed.json').read_text())
assert reserved_done['training_role'] == 'train_only'
assert config['coronal_bindings'][str(reserved / 'completed.json')] == \
    sha(reserved / 'completed.json')
assert not {str(row['animal_id']) for row in coronal_records} & \
    set(reserved_done['donor_receipts'])
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sagittal_summary['roles']['weak_dev2_source_available'] == \
    {'donors': 8, 'sections': 158}
for name, digest in sagittal_summary['output_sha256'].items():
    assert sha(sagittal / name) == digest
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
dev2_donors = {10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354}
assert len(sagittal_records) == 158 and \
    {row['donor_id'] for row in sagittal_records} == dev2_donors
assert len({row['section_id'] for row in sagittal_records}) == 158
assert all(row['split'] == 'weak_dev2_source_available' for row in sagittal_records)
assert [row['array_row_index'] for row in sagittal_records] == list(range(158))
sagittal_draws = [row for row in draws if row['kind'] == 'sagittal_weak_train']
assert len(sagittal_draws) == 1959 - 653
for key in ('donor_id', 'specimen_id', 'experiment_id', 'section_id'):
    assert not {row[key] for row in sagittal_records} & \
        {row[key] for row in sagittal_draws}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
assert sagittal_images.shape == (158, 1, side, side)
train_sagittal_summary = json.loads((sagittal_train / 'summary.json').read_text())
assert sha(sagittal_train / 'summary.json') == config['sagittal_summary_sha256']
assert train_sagittal_summary['output_sha256'] == config['sagittal_output_sha256']
assert sha(sagittal_train / 'geometry.jsonl') == \
    train_sagittal_summary['output_sha256']['geometry.jsonl']
train_sagittal_records = [json.loads(line) for line in
    (sagittal_train / 'geometry.jsonl').open()]
assert len(train_sagittal_records) == 653 and all(
    row['split'] == 'train' for row in train_sagittal_records)
assert not {row['donor_id'] for row in train_sagittal_records} & dev2_donors
assert {row['section_id'] for row in sagittal_draws} == \
    {row['section_id'] for row in train_sagittal_records}

parent_state = torch.load(parent_file, map_location='cpu', weights_only=True)
initial = torch.load(run / 'joint_step_00653.pt', map_location='cpu', weights_only=True)
assert initial['step'] == 653 and initial['config'] == config and initial['calibrated'] is False
assert set(initial['model']) == set(parent_state['model'])
assert all(torch.equal(initial['model'][key], value)
    for key, value in parent_state['model'].items())
del parent_state, initial

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
axis = (torch.arange(16, device='cuda') + .5) / 16 - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(-1, 2)
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda')
five_chart = pixels / side
candidate_rows, section_rows, real_rows = [], [], []

with torch.inference_mode():
    for step in steps:
        checkpoint = torch.load(checkpoints[step], map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and checkpoint['config'] == config
        assert checkpoint['calibrated'] is False
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for index, record in enumerate(selected, 1):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                target_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                target_reflection = torch.from_numpy(
                    arrays['reflection'].reshape(1, 1).copy()).cuda().long()
            assert int(valid.sum()) == record['valid_pixels']
            valid16 = (F.interpolate(valid[:, None].float(), (16, 16),
                mode='bilinear', align_corners=False)[0, 0] == 1).flatten()
            assert int(valid16.sum()) > 0
            target = rigid_points_090(target_state[:, None], target_reflection,
                chart16)[0, 0]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            assert len(set(beam[0].tolist())) == 14
            state = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            error = (rigid_points_090(state, beam % 2, chart16) -
                target[None, None]).norm(dim=-1)[..., valid16].mean(-1)[0]
            prior_selected_branch = int(prior[0].argmax())
            prior_selected_slot = int((beam[0] == prior_selected_branch).nonzero()[0, 0])
            normal = np.abs(record['plane_normal_ap_dv_ml'])
            angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
            common = {'step': step, 'section_id': record['section_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'panel_physical_section_id': record['panel_physical_section_id'],
                'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
                'appearance_mode': record['appearance_mode'],
                'nearest_axis': ('AP', 'DV', 'ML')[int(np.argmax(normal))],
                'nearest_axis_angle_deg': angle,
                'angle_bin': ('0-15' if angle < 15 else '15-30' if angle < 30
                    else '30-45' if angle < 45 else '45-54.7'),
                'valid_coarse_sites': int(valid16.sum()),
                'valid16_mask_sha256': hashlib.sha256(
                    valid16.cpu().numpy().astype('|u1').tobytes()).hexdigest()}
            for slot in range(14):
                candidate_rows.append({**common, 'beam_slot': slot,
                    'branch_id': int(beam[0, slot]),
                    'branch_family': 'old' if slot < 8 else 'anchor',
                    'prior_log_mass': float(prior[0, beam[0, slot]]),
                    'rigid_error_um': float(error[slot]),
                    'state': state[0, slot].tolist()})
            section_rows.append({**common, 'beam_branch_ids': beam[0].tolist(),
                'best14_rigid_error_um': float(error.min()),
                'best14_slot': int(error.argmin()),
                'prior_selected_rigid_error_um': float(error[prior_selected_slot]),
                'prior_selected_branch': prior_selected_branch,
                'prior_selected_slot': prior_selected_slot})
            if index % 16 == 0:
                print(json.dumps({'step': step, 'synthetic_sections_done': index}), flush=True)

        for domain, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
            for record in records:
                row_index = record['array_row_index']
                if domain == 'coronal':
                    native = np.concatenate((np.asarray(coronal_images[row_index], dtype=np.float32),
                        np.zeros((4, 192, 192), dtype=np.float32)))[None]
                    image = F.interpolate(torch.from_numpy(native).cuda(), (side, side),
                        mode='bilinear', align_corners=False)
                    affine = torch.as_tensor(coronal_affines[row_index],
                        device='cuda', dtype=torch.float32)
                    reference = (affine[:, 2] + (192 / side) * pixels[:, :1] * affine[:, 0]
                        + (192 / side) * pixels[:, 1:] * affine[:, 1])
                    donor = record['animal_id']
                else:
                    native = np.concatenate((np.asarray(sagittal_images[row_index], dtype=np.float32),
                        np.zeros((4, side, side), dtype=np.float32)))[None]
                    image = torch.from_numpy(native).cuda()
                    affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                        device='cuda', dtype=torch.float32)
                    reference = (affine[:, 2] + pixels[:, :1] * affine[:, 0]
                        + pixels[:, 1:] * affine[:, 1])
                    donor = record['donor_id']
                prediction = model.predict(image)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                branch = int(prior[0].argmax())
                proposed = rigid_points_090(prediction['state'][:, branch // 2],
                    torch.tensor([branch % 2], device='cuda'), five_chart)[0]
                weak_normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                predicted_normal = full_frame_state_to_components(
                    prediction['state'][:, branch // 2])[1][0, :, 2]
                normal_deg = torch.rad2deg((predicted_normal @ weak_normal).abs().clamp(0, 1).acos())
                real_rows.append({'set': domain, 'step': step, 'donor_id': donor,
                    'specimen_id': record['specimen_id'],
                    'experiment_id': record['experiment_id'],
                    'section_id': record['section_id'], 'array_row_index': row_index,
                    'prior_selected_branch': branch,
                    'prior_selected_five_point_error_um': float(
                        (proposed - reference).norm(dim=-1).mean()),
                    'prior_selected_normal_error_deg': float(normal_deg),
                    'label_role': 'inherited weak Allen affine, not expert pose'})
            print(json.dumps({'step': step, 'set': domain,
                'weak_sections_done': len(records)}), flush=True)

assert len(candidate_rows) == 64 * 14
assert len(section_rows) == 64
assert len(real_rows) == 64 + 158

summary = {'role': 'reused synthetic development panel and weak real DEV guardrails',
    'metric': 'direct rigid cutting-plane CCF distance on fully valid 16x16 sites; '
        'real five-point/normal discrepancy to inherited weak Allen affines',
    'not_claimed': 'no fitted mapping, ranking-score, expert-truth, confirmation, or public-benchmark claim',
    'synthetic_sections_per_step': 64, 'synthetic_candidates_per_step': 64 * 14,
    'ineligible_panel_sections_excluded': panel_done['ineligible'],
    'real_coronal_weak_sections_per_step': 64,
    'real_sagittal_dev2_source_available_sections_per_step': 158,
    'real_sagittal_dev2_original_sections': 159,
    'real_sagittal_source_unavailable_section_ids': [101345593],
    'step653_parent_weight_equal': True, 'steps': {}}
for step in steps:
    synthetic_rows = [row for row in section_rows if row['step'] == step]
    groups = {'all': synthetic_rows,
        'steep_ge_30': [row for row in synthetic_rows
            if row['nearest_axis_angle_deg'] >= 30]}
    for mode in ('raw', 'exact_black', 'imperfect_brush'):
        groups[f'appearance/{mode}'] = [row for row in synthetic_rows
            if row['appearance_mode'] == mode]
    for axis_name in ('AP', 'DV', 'ML'):
        groups[f'nearest_axis/{axis_name}'] = [row for row in synthetic_rows
            if row['nearest_axis'] == axis_name]
    for angle_bin in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'angle_bin/{angle_bin}'] = [row for row in synthetic_rows
            if row['angle_bin'] == angle_bin]
    real_summary = {}
    for domain in ('coronal', 'sagittal'):
        rows = [row for row in real_rows if row['step'] == step and row['set'] == domain]
        donor_ids = sorted({row['donor_id'] for row in rows})
        by_donor = {str(donor): {'sections': len(group),
            'prior_selected_five_point_error_um': float(np.mean([
                row['prior_selected_five_point_error_um'] for row in group])),
            'prior_selected_normal_error_deg': float(np.mean([
                row['prior_selected_normal_error_deg'] for row in group]))}
            for donor in donor_ids if (group := [row for row in rows
                if row['donor_id'] == donor])}
        real_summary[domain] = {'sections': len(rows), 'donors': len(by_donor),
            'by_donor': by_donor, 'donor_equal': {key: float(np.mean([
                item[key] for item in by_donor.values()])) for key in (
                    'prior_selected_five_point_error_um',
                    'prior_selected_normal_error_deg')}}
    summary['steps'][str(step)] = {'synthetic': {name: synthetic_aggregate(group, plan_ids)
        for name, group in groups.items()}, 'real_weak': real_summary}
summary['paired_delta_111_minus_baseline'] = {}
for baseline_name, baseline_step, reference_sections, reference_real in (
        ('108_step_00653', 653, baseline_sections, baseline_real),
        ('108_step_01959', 1959, baseline_sections, baseline_real),
        ('109_step_01959', 1959, heads_sections, heads_real),
        ('110_step_01959', 1959, retention_sections, retention_real)):
    old_sections = {row['section_id']: row for row in reference_sections
        if row['step'] == baseline_step}
    assert len(old_sections) == 64 and all(
        row['panel_file_sha256'] == old_sections[row['section_id']]['panel_file_sha256']
        and row['valid16_mask_sha256'] == old_sections[row['section_id']]['valid16_mask_sha256']
        for row in section_rows)
    synthetic_delta = [{**row, **{key: row[key] - old_sections[row['section_id']][key]
        for key in ('best14_rigid_error_um', 'prior_selected_rigid_error_um')}}
        for row in section_rows]
    old_real = {(row['set'], row['section_id']): row for row in reference_real
        if row['step'] == baseline_step}
    assert len(old_real) == 64 + 158 and all(
        all(row[key] == old_real[row['set'], row['section_id']][key]
            for key in ('donor_id', 'specimen_id', 'experiment_id', 'array_row_index'))
        for row in real_rows)
    real_delta = {}
    for domain in ('coronal', 'sagittal'):
        rows = [row for row in real_rows if row['set'] == domain]
        by_donor = {str(donor): {'sections': len(group), **{key: float(np.mean([
            row[key] - old_real[domain, row['section_id']][key] for row in group]))
            for key in ('prior_selected_five_point_error_um',
                'prior_selected_normal_error_deg')}}
            for donor in sorted({row['donor_id'] for row in rows})
            if (group := [row for row in rows if row['donor_id'] == donor])}
        real_delta[domain] = {'by_donor': by_donor,
            'donor_equal': {key: float(np.mean([item[key]
                for item in by_donor.values()])) for key in (
                'prior_selected_five_point_error_um',
                'prior_selected_normal_error_deg')}}
    summary['paired_delta_111_minus_baseline'][baseline_name] = {
        'synthetic': synthetic_aggregate(synthetic_delta, plan_ids),
        'real_weak': real_delta}

eval_config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_v3_mixed_real_pose_coronal_risk_111.py', 'arbitrary_plane_one_shot_model.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py')},
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_file),
    'train_completion_sha256': sha(run / 'completed.json'),
    'train_config_sha256': sha(run / 'config.json'),
    'train_draws_sha256': sha(run / 'draws.jsonl'),
    'train_training_sha256': sha(run / 'training.jsonl'),
    'train_schedule_sha256': config['schedule_sha256'],
    'teacher_checkpoint_sha256': done['teacher_checkpoint_sha256'],
    'retention_coefficient': config['retention_coefficient'],
    'retention_scope': config['retention_scope'],
    'risk_coefficient': config['risk_coefficient'],
    'risk_hinge_mm': config['risk_hinge_mm'],
    'checkpoint_sha256': done['checkpoint_sha256'],
    'baseline_108_eval_completion_sha256': sha(baseline_eval / 'completed.json'),
    'baseline_108_eval_sections_sha256': baseline_done['sections_sha256'],
    'baseline_108_eval_real_rows_sha256': baseline_done['real_rows_sha256'],
    'baseline_109_eval_completion_sha256': sha(heads_eval / 'completed.json'),
    'baseline_109_eval_sections_sha256': heads_done['sections_sha256'],
    'baseline_109_eval_real_rows_sha256': heads_done['real_rows_sha256'],
    'baseline_110_eval_completion_sha256': sha(retention_eval / 'completed.json'),
    'baseline_110_eval_sections_sha256': retention_done['sections_sha256'],
    'baseline_110_eval_real_rows_sha256': retention_done['real_rows_sha256'],
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'panel_plans_completion_sha256': sha(plans / 'completed.json'),
    'selected_section_ids_sha256': hashlib.sha256(json.dumps(
        [row['section_id'] for row in selected]).encode()).hexdigest(),
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'coronal_output_sha256': coronal_done['output_sha256'],
    'reserved_train_completion_sha256': sha(reserved / 'completed.json'),
    'sagittal_dev2_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_dev2_output_sha256': sagittal_summary['output_sha256'],
    'sagittal_train_summary_sha256': sha(sagittal_train / 'summary.json'),
    'sagittal_train_geometry_sha256': sha(sagittal_train / 'geometry.jsonl'),
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(eval_config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows),
                   ('sections.jsonl', section_rows), ('real_rows.jsonl', real_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'synthetic_sections_per_step': 64, 'synthetic_candidates_per_step': 64 * 14,
    'real_coronal_weak_sections_per_step': 64,
    'real_sagittal_dev2_source_available_sections_per_step': 158,
    'step653_parent_weight_equal': True,
    'shared_image_features_trainable': True,
    'retention_coefficient': eval_config['retention_coefficient'],
    'risk_coefficient': eval_config['risk_coefficient'],
    'risk_hinge_mm': eval_config['risk_hinge_mm'],
    'teacher_checkpoint_sha256': eval_config['teacher_checkpoint_sha256'],
    'source_sha256': eval_config['source_sha256'],
    'parent_checkpoint_sha256': eval_config['parent_checkpoint_sha256'],
    'train_completion_sha256': eval_config['train_completion_sha256'],
    'checkpoint_sha256': eval_config['checkpoint_sha256'],
    'baseline_108_eval_completion_sha256': eval_config['baseline_108_eval_completion_sha256'],
    'baseline_109_eval_completion_sha256': eval_config['baseline_109_eval_completion_sha256'],
    'baseline_110_eval_completion_sha256': eval_config['baseline_110_eval_completion_sha256'],
    'panel_completion_sha256': eval_config['panel_completion_sha256'],
    'coronal_completion_sha256': eval_config['coronal_completion_sha256'],
    'sagittal_dev2_summary_sha256': eval_config['sagittal_dev2_summary_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'real_rows_sha256': sha(out / 'real_rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False},
    indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'steps': steps,
    'synthetic_sections_per_step': 64, 'coronal_weak_per_step': 64,
    'sagittal_weak_dev2_per_step': 158}), flush=True)
