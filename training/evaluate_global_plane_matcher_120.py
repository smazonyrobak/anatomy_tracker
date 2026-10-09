"""Frozen, disjoint synthetic DEV evaluation of the 120 head-only pilot."""

import copy
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher, global_plane_match

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_PLANE_MATCHER_120_DEV_GATE_PROTOCOL_20261009.md'
train_run = root / 'runs/global_plane_matcher_120_pilot'
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
plans = root / 'data/v3_pose_capture_confirmation_plans_001'
assay118 = root / 'runs/v3_interior_intensity_energy_118_dev'
out = root / 'runs/global_plane_matcher_120_dev_eval'
steps, arms, side = (0, 256, 512), ('atlas', 'support_only'), 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
done = json.loads((train_run / 'completed.json').read_text())
config = json.loads((train_run / 'config.json').read_text())
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert done['updates'] == config['updates'] == 512
assert done['accepted_synthetic'] == config['batch'] * config['updates'] == 1024
assert tuple(config['checkpoints']) == steps and config['blind_count'] == 16
assert sha(train_run / 'config.json') == done['config_sha256']
assert sha(train_run / 'draws.jsonl') == done['draws_sha256']
assert sha(train_run / 'training.jsonl') == done['training_sha256']
assert sha(parent) == config['parent_checkpoint_sha256'] == \
    done['parent_checkpoint_sha256'] == parent_done['checkpoint_sha256']['1959']
assert sha(parent_run / 'completed.json') == config['parent_completion_sha256']
assert sha(parent_run / 'config.json') == config['parent_config_sha256']
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
assert not any(config[key] or done[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
checkpoints = {step: train_run / f'head_step_{step:05d}.pt' for step in steps}
assert all(sha(path) == done['checkpoint_sha256'][str(step)]
    for step, path in checkpoints.items())
draws = [json.loads(line) for line in (train_run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 1024 and all(row['base_lineage']['split'] == 'train' for row in used)

panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_done['physical_sections'] == 256 and panel_done['eligible'] == 247
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(plans / 'completed.json') == panel_done['plan_completed_sha256'] == \
    panel_protocol['source_plan_completed_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(records) == len({row['section_id'] for row in records}) == 256
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plan_ids) == 8

assay_done = json.loads((assay118 / 'completed.json').read_text())
assay_config = json.loads((assay118 / 'config.json').read_text())
assert assay_done['sections'] == 64
assert sha(assay118 / 'config.json') == assay_done['config_sha256']
assert sha(assay118 / 'rows.jsonl') == assay_done['rows_sha256']
assert sha(source / 'diagnose_v3_interior_intensity_energy_118.py') == \
    assay_done['source_sha256']
assert sha(panel / 'completed.json') == assay_done['panel_completion_sha256'] == \
    assay_config['panel_completion_sha256']
assert sha(panel / 'records.jsonl') == assay_config['panel_records_sha256']
old_rows = [json.loads(line) for line in (assay118 / 'rows.jsonl').open()]
old_ids = [row['section_id'] for row in old_rows]
assert len(old_ids) == len(set(old_ids)) == 64
assert hashlib.sha256(json.dumps(old_ids).encode()).hexdigest() == \
    assay_config['selected_section_ids_sha256']
selected = [row for plan in plan_ids for row in sorted((row for row in records
    if row['synthetic_subject_plan_id'] == plan and row['eligible']
    and row['section_id'] not in old_ids),
    key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())[:8]]
assert len(selected) == len({row['section_id'] for row in selected}) == 64
assert not {row['section_id'] for row in selected} & set(old_ids)
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in selected) == 8
    for plan in plan_ids)
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in selected} & {row['base_lineage'][key] for row in used}
assert not {row['panel_physical_section_id'] for row in selected} & \
    {row['physical_section_id'] for row in used}

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
parent_checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert parent_checkpoint['step'] == 1959 and parent_checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(parent_checkpoint['model'], strict=True)
del parent_checkpoint
attach_global_plane_matcher(model, enabled=True)
heads = {}
for step, path in checkpoints.items():
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == config
    assert checkpoint['calibrated'] is False and set(checkpoint['heads']) == set(arms)
    if step == 0:
        assert all(torch.equal(checkpoint['heads']['atlas'][key], value)
            for key, value in checkpoint['heads']['support_only'].items())
    for arm in arms:
        head = copy.deepcopy(model.global_plane_matcher_120)
        head.load_state_dict(checkpoint['heads'][arm], strict=True)
        heads[(step, arm)] = head.eval().requires_grad_(False)
    del checkpoint

five_chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side
candidate_rows, section_rows = [], []

with torch.inference_mode():
    for record in selected:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(
                arrays['reflection'].reshape(1).copy()).cuda().long()
            centre = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        valid_pixels = valid[0].flatten().nonzero()[:, 0].cpu().numpy()
        assert len(valid_pixels) > 0
        site_rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(record['section_id'].encode()).digest()[:8], 'little'))
        site_indices = torch.as_tensor(site_rng.choice(valid_pixels, size=256,
            replace=len(valid_pixels) < 256), device='cuda')
        chart_sites = torch.stack((site_indices.remainder(side),
            site_indices.div(side, rounding_mode='floor')), -1).float() / side
        target_rigid = rigid_points_090(truth, truth_reflection, chart_sites)[0]
        target_five = rigid_points_090(truth, truth_reflection, five_chart)[0]
        target_mapped = centre.reshape(-1, 3)[site_indices]
        atlas_support = float(render_atlas_planes_090(context['atlas'], truth[:, None],
            truth_reflection[:, None], offsets, weights, candidate_chunk=1)[0, 0, 1].mean())
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
        mode_index, reflection = beam // 2, beam % 2
        prior_beam = prior.gather(1, beam)[0]
        assert torch.equal(prior_beam[:14].max(), prior[0].max())
        state = prediction['state'].gather(1,
            mode_index[..., None].expand(-1, -1, 12))
        normal = np.abs(record['plane_normal_ap_dv_ml'])
        angle = math.degrees(math.acos(min(1., float(np.max(normal)))))
        common = {'section_id': record['section_id'],
            'panel_physical_section_id': record['panel_physical_section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'synthetic_subject_realization_id': record['synthetic_subject_realization_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'appearance_mode': record['appearance_mode'],
            'nearest_axis': ('AP', 'DV', 'ML')[int(np.argmax(normal))],
            'nearest_axis_angle_deg': angle,
            'angle_bin': ('0-15' if angle < 15 else '15-30' if angle < 30
                else '30-45' if angle < 45 else '45-54.7'),
            'atlas_support_fraction': atlas_support,
            'support_bin': ('<0.25' if atlas_support < .25 else '0.25-0.50'
                if atlas_support < .5 else '>=0.50'),
            'valid_fraction': record['valid_pixels'] / (side * side),
            'valid_sampled_sites': 256,
            'valid_site_indices_sha256': hashlib.sha256(
                site_indices.cpu().numpy().tobytes()).hexdigest(),
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'plan_receipt_sha256': record['plan_receipt_sha256']}

        for step in steps:
            for arm in arms:
                output = global_plane_match(model, prediction, mode_index, reflection,
                    offsets, weights, context['atlas'], (side, side), side=24,
                    support_only=arm == 'support_only', matcher=heads[(step, arm)])
                assert torch.equal(output['input_score'][0], prior_beam)
                input_rigid = (rigid_points_090(output['input_state'], reflection,
                    chart_sites) - target_rigid[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                corrected_rigid = (rigid_points_090(output['state'], reflection,
                    chart_sites) - target_rigid[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                input_mapped = (rigid_points_090(output['input_state'], reflection,
                    chart_sites) - target_mapped[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                corrected_mapped = (rigid_points_090(output['state'], reflection,
                    chart_sites) - target_mapped[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                input_five = (rigid_points_090(output['input_state'], reflection,
                    five_chart) - target_five[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                corrected_five = (rigid_points_090(output['state'], reflection,
                    five_chart) - target_five[None, None]).norm(dim=-1).mean(-1)[0] / 1000
                prior_slot = int(output['input_score'][0, :14].argmax())
                input_slot = int(output['input_score'][0].argmax())
                corrected_slot = int(output['score'][0].argmax())
                action_score = torch.stack((output['input_score'], output['score']), -1)
                joint = int(action_score.flatten(1)[0].argmax())
                joint_slot, joint_action = divmod(joint, 2)
                rigid_pair = torch.stack((input_rigid, corrected_rigid), -1)
                mapped_pair = torch.stack((input_mapped, corrected_mapped), -1)
                five_pair = torch.stack((input_five, corrected_five), -1)
                best_rigid_slot = int(corrected_rigid.argmin())
                best_five_slot = int(corrected_five.argmin())
                near = float(input_five[:14].min()) <= 1.5
                section_rows.append({**common, 'step': step, 'arm': arm,
                    'beam_branch_ids': beam[0].tolist(),
                    'original14_prior_slot': prior_slot, 'input16_slot': input_slot,
                    'corrected16_slot': corrected_slot, 'joint16_slot': joint_slot,
                    'joint16_action': 'corrected' if joint_action else 'no_shift',
                    'near_best14_five_le_1p5_mm': near,
                    'best14_input_rigid_mm': float(input_rigid[:14].min()),
                    'best14_input_five_mm': float(input_five[:14].min()),
                    'parent111_prior14_rigid_mm': float(input_rigid[prior_slot]),
                    'parent111_prior14_mapped_mm': float(input_mapped[prior_slot]),
                    'parent111_prior14_five_mm': float(input_five[prior_slot]),
                    'input16_rigid_mm': float(input_rigid[input_slot]),
                    'input16_mapped_mm': float(input_mapped[input_slot]),
                    'input16_five_mm': float(input_five[input_slot]),
                    'corrected16_rigid_mm': float(corrected_rigid[corrected_slot]),
                    'corrected16_mapped_mm': float(corrected_mapped[corrected_slot]),
                    'corrected16_five_mm': float(corrected_five[corrected_slot]),
                    'joint16_rigid_mm': float(rigid_pair[joint_slot, joint_action]),
                    'joint16_mapped_mm': float(mapped_pair[joint_slot, joint_action]),
                    'joint16_five_mm': float(five_pair[joint_slot, joint_action]),
                    'best_corrected16_rigid_mm': float(corrected_rigid[best_rigid_slot]),
                    'best_rigid_slot_corrected16_five_mm': float(corrected_five[best_rigid_slot]),
                    'best_corrected16_five_mm': float(corrected_five[best_five_slot]),
                    'best_five_slot_corrected16_rigid_mm': float(corrected_rigid[best_five_slot]),
                    'best_corrected16_mapped_mm': float(corrected_mapped.min()),
                    'delta_joint_vs_parent_rigid_mm': float(
                        rigid_pair[joint_slot, joint_action] - input_rigid[prior_slot]),
                    'delta_joint_vs_parent_five_mm': float(
                        five_pair[joint_slot, joint_action] - input_five[prior_slot]),
                    'near_rigid_nonregressed_0p2_mm': bool(
                        rigid_pair[joint_slot, joint_action] <= input_rigid[prior_slot] + .2),
                    'near_five_nonregressed_0p2_mm': bool(
                        five_pair[joint_slot, joint_action] <= input_five[prior_slot] + .2)})
                for slot in range(16):
                    candidate_rows.append({**common, 'step': step, 'arm': arm,
                        'beam_slot': slot, 'branch_id': int(beam[0, slot]),
                        'original14_member': slot < 14,
                        'input_score': float(output['input_score'][0, slot]),
                        'corrected_score': float(output['score'][0, slot]),
                        'match_logit': float(output['match_logit'][0, slot]),
                        'input_state': output['input_state'][0, slot].tolist(),
                        'corrected_state': output['state'][0, slot].tolist(),
                        'input_rigid_mm': float(input_rigid[slot]),
                        'corrected_rigid_mm': float(corrected_rigid[slot]),
                        'input_mapped_mm': float(input_mapped[slot]),
                        'corrected_mapped_mm': float(corrected_mapped[slot]),
                        'input_five_mm': float(input_five[slot]),
                        'corrected_five_mm': float(corrected_five[slot])})

assert len(section_rows) == 64 * len(steps) * len(arms)
assert len(candidate_rows) == 16 * len(section_rows)
fields = ('parent111_prior14_rigid_mm', 'parent111_prior14_mapped_mm',
    'parent111_prior14_five_mm', 'input16_rigid_mm', 'input16_mapped_mm',
    'input16_five_mm', 'corrected16_rigid_mm', 'corrected16_mapped_mm',
    'corrected16_five_mm', 'joint16_rigid_mm', 'joint16_mapped_mm',
    'joint16_five_mm', 'best14_input_rigid_mm', 'best14_input_five_mm',
    'best_corrected16_rigid_mm', 'best_rigid_slot_corrected16_five_mm',
    'best_corrected16_five_mm', 'best_five_slot_corrected16_rigid_mm',
    'best_corrected16_mapped_mm', 'delta_joint_vs_parent_rigid_mm',
    'delta_joint_vs_parent_five_mm')


def aggregate(rows, metric_fields=fields):
    by_plan = {plan: {'sections': len(group), **{key: float(np.mean(
        [row[key] for row in group])) for key in metric_fields}}
        for plan in plan_ids if (group := [row for row in rows
            if row['synthetic_subject_plan_id'] == plan])}
    return {'sections': len(rows), 'represented_plans': len(by_plan), 'by_plan': by_plan,
        'plan_equal_mm': {key: float(np.mean([group[key] for group in by_plan.values()]))
            for key in metric_fields} if by_plan else {}}


summary = {'role': 'predeclared 64-case synthetic DEV, disjoint from 118; not confirmation',
    'primary': 'rigid full-frame CCF error against target_state at 256 frozen valid observed pixels',
    'secondary': 'rigid-only observed-pixel CCF error against target_centre_um at the same sites; separately labeled',
    'five_point': 'rigid CCF error at the fixed four corners and centre',
    'near_case': 'original14 minimum five-point rigid error <=1.5 mm, diagnostic stratum only',
    'nonregression': 'joint blind selected error <= parent111 prior-selected error +0.2 mm',
    'steps': {}}
for step in steps:
    summary['steps'][str(step)] = {}
    groups = {'all': lambda row: True,
        'near_best14_five_le_1p5_mm': lambda row: row['near_best14_five_le_1p5_mm'],
        'appearance/raw': lambda row: row['appearance_mode'] == 'raw',
        'appearance/imperfect_brush': lambda row: row['appearance_mode'] == 'imperfect_brush',
        'appearance/exact_black': lambda row: row['appearance_mode'] == 'exact_black',
        'angle/steep_ge_30': lambda row: row['nearest_axis_angle_deg'] >= 30}
    for angle_bin in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'angle_bin/{angle_bin}'] = lambda row, value=angle_bin: row['angle_bin'] == value
    for support_bin in ('<0.25', '0.25-0.50', '>=0.50'):
        groups[f'atlas_support/{support_bin}'] = \
            lambda row, value=support_bin: row['support_bin'] == value
    for arm in arms:
        rows = [row for row in section_rows if row['step'] == step and row['arm'] == arm]
        summary['steps'][str(step)][arm] = {name: aggregate(
            [row for row in rows if predicate(row)]) for name, predicate in groups.items()}
        near_rows = [row for row in rows if row['near_best14_five_le_1p5_mm']]
        summary['steps'][str(step)][arm]['near_nonregression'] = {
            'sections': len(near_rows),
            'rigid_within_0p2_fraction': float(np.mean([
                row['near_rigid_nonregressed_0p2_mm'] for row in near_rows])) if near_rows else None,
            'five_within_0p2_fraction': float(np.mean([
                row['near_five_nonregressed_0p2_mm'] for row in near_rows])) if near_rows else None,
            'rigid_plan_equal_nonregressed': (
                summary['steps'][str(step)][arm]['near_best14_five_le_1p5_mm']
                ['plan_equal_mm']['delta_joint_vs_parent_rigid_mm'] <= .2) if near_rows else None,
            'five_plan_equal_nonregressed': (
                summary['steps'][str(step)][arm]['near_best14_five_le_1p5_mm']
                ['plan_equal_mm']['delta_joint_vs_parent_five_mm'] <= .2) if near_rows else None}
    paired = []
    atlas_rows = {row['section_id']: row for row in section_rows
        if row['step'] == step and row['arm'] == 'atlas'}
    control_rows = {row['section_id']: row for row in section_rows
        if row['step'] == step and row['arm'] == 'support_only'}
    assert set(atlas_rows) == set(control_rows) == {row['section_id'] for row in selected}
    for section_id, atlas_row in atlas_rows.items():
        control = control_rows[section_id]
        assert atlas_row['beam_branch_ids'] == control['beam_branch_ids']
        paired.append({**atlas_row,
            'atlas_minus_support_joint_rigid_mm': atlas_row['joint16_rigid_mm']
                - control['joint16_rigid_mm'],
            'atlas_minus_support_joint_five_mm': atlas_row['joint16_five_mm']
                - control['joint16_five_mm'],
            'atlas_minus_support_best_corrected_rigid_mm':
                atlas_row['best_corrected16_rigid_mm']
                - control['best_corrected16_rigid_mm']})
    paired_fields = ('atlas_minus_support_joint_rigid_mm',
        'atlas_minus_support_joint_five_mm',
        'atlas_minus_support_best_corrected_rigid_mm')
    summary['steps'][str(step)]['paired_atlas_minus_support_only'] = {
        name: aggregate([row for row in paired if predicate(row)], paired_fields)
        for name, predicate in groups.items()}

best_step = min(steps, key=lambda step: (
    summary['steps'][str(step)]['atlas']['all']['plan_equal_mm']['joint16_rigid_mm'], step))
atlas_groups = summary['steps'][str(best_step)]['atlas']
control_groups = summary['steps'][str(best_step)]['support_only']
atlas_all = atlas_groups['all']['plan_equal_mm']
control_all = control_groups['all']['plan_equal_mm']
oracle_plan_gain = {plan: group['best_corrected16_rigid_mm'] <
    group['best14_input_rigid_mm'] for plan, group in atlas_groups['all']['by_plan'].items()}
near = atlas_groups['near_best14_five_le_1p5_mm']
strata = ('appearance/raw', 'appearance/imperfect_brush', 'angle/steep_ge_30',
    'atlas_support/<0.25', 'atlas_support/0.25-0.50', 'atlas_support/>=0.50')
stratum_nonregression = {name: (atlas_groups[name]['plan_equal_mm']
    ['delta_joint_vs_parent_rigid_mm'] <= .2) if atlas_groups[name]['sections'] >= 8
    else None for name in strata}
conditions = {
    'atlas_joint_gain_vs_parent_ge_0p20_mm':
        atlas_all['parent111_prior14_rigid_mm'] - atlas_all['joint16_rigid_mm'] >= .2,
    'atlas_joint_gain_vs_support_only_ge_0p20_mm':
        control_all['joint16_rigid_mm'] - atlas_all['joint16_rigid_mm'] >= .2,
    'atlas_best_corrected_gain_vs_best14_input_ge_0p50_mm':
        atlas_all['best14_input_rigid_mm'] - atlas_all['best_corrected16_rigid_mm'] >= .5,
    'atlas_best_corrected_improves_ge_6_of_8_plans': sum(oracle_plan_gain.values()) >= 6,
    'near_rigid_and_five_nonregression_0p20_mm': near['sections'] > 0 and
        near['plan_equal_mm']['delta_joint_vs_parent_rigid_mm'] <= .2 and
        near['plan_equal_mm']['delta_joint_vs_parent_five_mm'] <= .2,
    'raw_brush_steep_and_populated_support_nonregression_0p20_mm':
        all(value is not False for value in stratum_nonregression.values()) and
        all(stratum_nonregression[name] is not None for name in strata[:3]),
}
summary['best_development_step'] = best_step
summary['pilot_gate'] = {'conditions': conditions, 'passed': all(conditions.values()),
    'oracle_plan_gain': oracle_plan_gain,
    'stratum_nonregression': stratum_nonregression,
    'interpretation': 'Synthetic DEV feasibility only; no deployment or confirmation claim.'}

selection = {'rule': '8 eligible sections per plan by ascending SHA256(section_id) after excluding all 64 assay-118 IDs',
    'assay118_excluded_section_ids_sha256': assay_config['selected_section_ids_sha256'],
    'selected_section_ids': [row['section_id'] for row in selected],
    'selected_section_ids_sha256': hashlib.sha256(json.dumps(
        [row['section_id'] for row in selected]).encode()).hexdigest(),
    'panel_files': [{'section_id': row['section_id'], 'file': row['file'],
        'sha256': row['sha256']} for row in selected]}
eval_config = {'source_sha256': {name: sha(source / name) for name in (
    'evaluate_global_plane_matcher_120.py', 'train_global_plane_matcher_120.py',
    'global_plane_matcher_120.py', 'arbitrary_plane_one_shot_model.py',
    'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_streaming_synthetic_v7_64.py')},
    'protocol_sha256': sha(protocol),
    'train_completion_sha256': sha(train_run / 'completed.json'),
    'train_config_sha256': sha(train_run / 'config.json'),
    'train_draws_sha256': sha(train_run / 'draws.jsonl'),
    'train_training_sha256': sha(train_run / 'training.jsonl'),
    'parent_checkpoint_sha256': sha(parent),
    'checkpoint_sha256': {str(step): sha(path) for step, path in checkpoints.items()},
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'plans_completion_sha256': sha(plans / 'completed.json'),
    'assay118_completion_sha256': sha(assay118 / 'completed.json'),
    'assay118_rows_sha256': sha(assay118 / 'rows.jsonl'),
    'selected_section_ids_sha256': selection['selected_section_ids_sha256'],
    'selection': selection['rule'],
    'beam': 'exact 120 trainer original 111 top8 base/top6 anchor plus 2 normal-diverse unused anchors',
    'support_stratum': 'truth-plane 96x96 finite-thickness atlas support fraction, bins <.25/.25-.50/>=.50; diagnostic only',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(eval_config, indent=2, allow_nan=False))
(out / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_step_arm': 64, 'candidates_per_step_arm': 64 * 16,
    'protocol_sha256': eval_config['protocol_sha256'],
    'source_sha256': eval_config['source_sha256'],
    'train_completion_sha256': eval_config['train_completion_sha256'],
    'panel_completion_sha256': eval_config['panel_completion_sha256'],
    'assay118_completion_sha256': eval_config['assay118_completion_sha256'],
    'checkpoint_sha256': eval_config['checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'selection_sha256': sha(out / 'selection.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False,
    'oracle_corrected16_diagnostic_only': True}, indent=2))
print(json.dumps({'event': 'complete', 'sections': 64,
    'summary': {step: summary['steps'][str(step)]['atlas']['all']['plan_equal_mm']
                for step in steps}}, allow_nan=False), flush=True)
