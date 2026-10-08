"""Frozen 103 cross-candidate evidence on a fresh natural 094 beam."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.contextual_matcher_099 import WholeSliceAtlasFeedback099
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/CROSS_CANDIDATE_MATCH_103_PROTOCOL_20261009.md'
dev_protocol = source.parent / 'docs/publication/CROSS_CANDIDATE_MATCH_103_DEV_PROTOCOL_20261009.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent_file = parent_run / 'joint_step_20000.pt'
baseline_run = root / 'runs/contextual_match_099_pilot'
baseline_file = baseline_run / 'contextual_step_04000.pt'
train_run = root / 'runs/cross_candidate_match_103_pilot'
plans = root / 'data/cross_candidate_match_103_fresh_dev_plans_001'
panel = root / 'data/cross_candidate_match_103_fresh_dev_panel_001'
out = root / 'runs/cross_candidate_match_103_fresh_development_eval'
steps = (0, 1000, 2000, 3000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def avg(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def at_chart(surface, chart):
    count = surface.shape[1]
    grid = ((chart + .5 / 256) * 2 - 1)[None].expand(
        count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
        padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent_file) == parent_receipt['checkpoint_sha256']['20000'] == \
    '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())

baseline_receipt = json.loads((baseline_run / 'completed.json').read_text())
baseline_config = json.loads((baseline_run / 'config.json').read_text())
assert baseline_receipt['batches'] == 4000
assert sha(baseline_file) == baseline_receipt['checkpoint_sha256']['4000'] == \
    'c4a85f1141109460eb2257ae77bb170bf8a650cd41a1d4d1dc550e84b913ec61'
assert sha(baseline_run / 'config.json') == baseline_receipt['config_sha256']
assert sha(baseline_run / 'draws.jsonl') == baseline_receipt['draws_sha256']
assert sha(baseline_run / 'training.jsonl') == baseline_receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in baseline_config['source_sha256'].items())
assert baseline_config['parent_094_sha256'] == sha(parent_file)

train_receipt = json.loads((train_run / 'completed.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
assert train_receipt['batches'] == train_receipt['accepted_synthetic'] == \
    train_receipt['distinct_physical_sections'] == train_config['batches'] == 4000
assert tuple(train_config['checkpoints']) == steps
assert sha(protocol) == train_config['protocol_sha256'] == train_receipt['protocol_sha256']
assert train_config['parent_094_sha256'] == train_receipt['parent_094_checkpoint_sha256'] == sha(parent_file)
assert train_config['parent_099_sha256'] == train_receipt['parent_099_checkpoint_sha256'] == sha(baseline_file)
assert train_config['source_sha256'] == train_receipt['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert sha(train_run / 'config.json') == train_receipt['config_sha256']
assert sha(train_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train_run / 'training.jsonl') == train_receipt['training_sha256']
assert not any(receipt[key] for receipt in (parent_receipt, baseline_receipt, train_receipt)
    for key in ('calibrated', 'public_benchmark_used', 'real_labels_used',
                'external_pretrained_weights_used'))
checkpoint_files = {step: train_run / f'contextual_step_{step:05d}.pt' for step in steps}
assert all(sha(checkpoint_files[step]) == train_receipt['checkpoint_sha256'][str(step)]
           for step in steps)

plan_receipt = json.loads((plans / 'completed.json').read_text())
plan_protocol = json.loads((plans / 'protocol.json').read_text())
assert plan_receipt['protocol'] == plan_protocol
assert plan_protocol['source_file_sha256']['docs/publication/CROSS_CANDIDATE_MATCH_103_DEV_PROTOCOL_20261009.md'] == sha(dev_protocol)
assert len(plan_receipt['subjects']) == 8 and plan_receipt['section_count'] == 0
assert all(sha(plans / name) == digest for name, digest in
           plan_receipt['artifact_sha256'].items())
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_receipt['physical_sections'] == 256
assert panel_protocol['synthetic_subjects'] == 8 and panel_protocol['planes_per_subject'] == 32
assert panel_protocol['side'] == 256
assert panel_protocol['dev_protocol_sha256'] == sha(dev_protocol)
assert panel_protocol['train_lineage'] == plan_protocol['train_lineage']
assert panel_receipt['plan_completed_sha256'] == panel_protocol['source_plan_completed_sha256'] == \
    sha(plans / 'completed.json')
assert panel_receipt['protocol_sha256'] == sha(panel / 'protocol.json')
assert panel_receipt['records_sha256'] == sha(panel / 'records.jsonl')
assert all(sha(panel / 'source' / name) == digest for name, digest in
           panel_protocol['source_sha256'].items())
assert all(sha(root / 'data' / name / 'records.jsonl') == digest
           for name, digest in panel_protocol['prior_panel_records_sha256'].items())
prior_records = [row for name in panel_protocol['prior_panel_records_sha256']
    for row in map(json.loads, (root / 'data' / name / 'records.jsonl').open())]
assert len(prior_records) > 0
all_records = list(map(json.loads, (panel / 'records.jsonl').open()))
assert len(all_records) == 256 and len({row['section_id'] for row in all_records}) == 256
assert len({row['panel_physical_section_id'] for row in all_records}) == 256
assert len({row['subject_ouv_sha256'] for row in all_records}) == 256
assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
plan_rows = {row['subject_deformation_plan_id']: row for row in plan_receipt['subjects']}
assert len(plan_rows) == 8
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in all_records) == 32
           for plan in plan_rows)
assert all(row['plan_receipt_sha256'] ==
           plan_rows[row['synthetic_subject_plan_id']]['subject_plan_receipt_sha256']
           for row in all_records)
old_draws = [row for row in map(json.loads, (baseline_run / 'draws.jsonl').open()) if row['used']]
new_draws = [row for row in map(json.loads, (train_run / 'draws.jsonl').open()) if row['used']]
assert len(old_draws) == len(new_draws) == 4000
assert len({row['physical_section_id'] for row in new_draws}) == 4000
assert not {row['physical_section_id'] for row in new_draws} & {
    row['physical_section_id'] for row in old_draws}
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in all_records} & \
        {row['base_lineage'][key] for row in old_draws + new_draws}
for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id'):
    assert not {row[key] for row in all_records} & {row[key] for row in prior_records}
assert not {row['plan_receipt_sha256'] for row in all_records} & \
    {row['plan_receipt_sha256'] for row in prior_records}
assert not {row['panel_physical_section_id'] for row in all_records} & \
    {row['physical_section_id'] for row in old_draws + new_draws}
assert sum(row['eligible'] for row in all_records) == panel_receipt['eligible']
assert sum(not row['eligible'] for row in all_records) == panel_receipt['ineligible']
assert {mode: sum(row['appearance_mode'] == mode for row in all_records)
    for mode in panel_protocol['modes']} == panel_receipt['modes']
records = [row for row in all_records if row['eligible']]
plan_ids = sorted(plan_rows)
assert all(any(row['synthetic_subject_plan_id'] == plan for row in records)
           for plan in plan_ids)
assert all(any(row['appearance_mode'] == mode for row in records)
           for mode in panel_protocol['modes'])
assert {int(np.argmax(np.abs(row['plane_normal_ap_dv_ml']))) for row in records} == {0, 1, 2}

saved = torch.load(parent_file, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and saved['calibrated'] is False
baseline_state = torch.load(baseline_file, map_location='cpu', weights_only=True)
assert baseline_state['step'] == 4000 and baseline_state['calibrated'] is False
assert baseline_state['config'] == baseline_config
states = {}
for step in steps:
    checkpoint = torch.load(checkpoint_files[step], map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == train_config
    assert checkpoint['calibrated'] is False
    states[step] = checkpoint['matcher']
assert set(states[0]) == set(baseline_state['matcher'])
assert all(torch.equal(states[0][key], value)
           for key, value in baseline_state['matcher'].items())
assert all(torch.equal(states[step][key], baseline_state['matcher'][key])
    for step in steps for key in states[step] if '.context.' not in key)
baseline_matcher_state = baseline_state['matcher']
del baseline_state

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
teacher = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
teacher.load_state_dict(saved['spatial_fit'], strict=True)
students = {}
for step in steps:
    student = WholeSliceAtlasFeedback099().cuda().eval().requires_grad_(False)
    student.load_state_dict(states[step], strict=True)
    students[step] = student
baseline_student = WholeSliceAtlasFeedback099().cuda().eval().requires_grad_(False)
baseline_student.load_state_dict(baseline_matcher_state, strict=True)
del saved

candidate_rows, section_rows = [], []
step0_max_logit_difference = 0.
step0_max_support_difference = 0.
with torch.inference_mode():
    for record_index, record in enumerate(records, 1):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            assert bool(arrays['eligible']) and int(valid.sum()) == record['valid_pixels']
        chosen = valid.flatten().nonzero().flatten()
        chosen = chosen[torch.linspace(0, len(chosen) - 1, 1024,
            device='cuda').round().long()]
        chosen_hash = hashlib.sha256(chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
        chart = torch.stack((chosen.remainder(256),
            chosen.div(256, rounding_mode='floor')), -1).float() / 256
        reference = truth.reshape(-1, 3)[chosen]
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = torch.cat((prior[:, :32].topk(8, -1).indices,
            prior[:, 32:].topk(6, -1).indices + 32), -1)
        assert len(set(branch[0].tolist())) == 14
        initial = prediction['state'].gather(1,
            (branch // 2)[..., None].expand(-1, -1, 12))
        normal = np.abs(record['plane_normal_ap_dv_ml'])
        family = ('AP', 'DV', 'ML')[int(np.argmax(normal))]
        angle_deg = math.degrees(math.acos(min(1., float(np.max(normal)))))
        angle_bin = ('0-15' if angle_deg < 15 else '15-30' if angle_deg < 30
                     else '30-45' if angle_deg < 45 else '45-54.7')
        common = {'section_id': record['section_id'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'appearance_mode': record['appearance_mode'],
            'angle_family': family, 'nearest_axis_angle_deg': angle_deg,
            'angle_bin': angle_bin, 'valid_fraction': record['valid_pixels'] / 65536,
            'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
            'sample_indices_sha256': chosen_hash}
        errors, old_scores = [], []
        by_step = {step: [] for step in steps}
        for start in range(0, 14, 2):
            pair = branch[:, start:start + 2]
            reflected = pair % 2
            state = initial[:, start:start + 2]
            fitted = teacher(prediction['feature'], state, reflected, atlas,
                offsets, weights, source_shape=(256, 256))
            selected_prediction = {**prediction, 'state': fitted['state'],
                'log_mass': prediction['log_mass'].gather(1, pair // 2),
                'reflection_logit': prediction['reflection_logit'].gather(1, pair // 2)}
            mapped = model.map(selected_prediction, offsets,
                torch.arange(2, device='cuda')[None], reflected, (96, 96),
                atlas, weights, feature_side=96, source_shape=(256, 256),
                spatial_evidence=fitted['spatial_evidence'])
            pair_errors = (at_chart(mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                - reference[None]).norm(dim=-1).mean(-1).tolist()
            errors.extend(pair_errors)
            local_warp_mm = mapped['local_displacement_um'] / 1000
            dx = local_warp_mm[..., 1:] - local_warp_mm[..., :-1]
            dy = local_warp_mm[..., 1:, :] - local_warp_mm[..., :-1, :]
            warp_cost = (local_warp_mm.square().mean((-3, -2, -1)) + .1 *
                (dx.square().mean((-3, -2, -1)) +
                 dy.square().mean((-3, -2, -1))))
            fit_temperature = teacher.log_fit_temperature.clamp(
                math.log(.5), math.log(5.)).exp()
            pair_old_scores = (prior.gather(1, pair) -
                fitted['fit_energy'] / fit_temperature - .05 * warp_cost)
            old_scores.extend(pair_old_scores[0].tolist())
            validity = fitted['validity_logit'].sigmoid()[:, None]
            predicted_tissue = validity >= .5
            tissue_weight = validity * predicted_tissue
            block_weight = tissue_weight.reshape(1, 1, 8, 4, 8, 4).sum((3, 5))
            occupied = block_weight > 0
            occupied_blocks = int(occupied.sum())
            tissue_sites = int(predicted_tissue.sum())
            for step in steps:
                student = students[step]
                match = student(prediction['feature'], fitted['state'], reflected,
                    atlas, offsets, weights, source_shape=(256, 256), match_only=True)
                logits = match['fine_match_logits']
                support = match['fine_match_support']
                if record_index == 1 and start == 0 and step == 0:
                    baseline = baseline_student(prediction['feature'], fitted['state'],
                        reflected, atlas, offsets, weights, source_shape=(256, 256),
                        match_only=True)
                    step0_max_logit_difference = float((
                        logits - baseline['fine_match_logits']).abs().max())
                    step0_max_support_difference = float((
                        support - baseline['fine_match_support']).abs().max())
                    assert step0_max_logit_difference < 1e-5
                    assert step0_max_support_difference == 0
                supported = support >= .5
                support_count = supported.sum(2)
                temperature = student.log_temperature.exp().clamp(2, 20)
                cosine = (logits + 4 * (1 - support)) / temperature
                evidence = (torch.logsumexp(
                    (20 * cosine).masked_fill(~supported, -1e4), 2)
                    - support_count.clamp_min(1).log()) / 20
                evidence = torch.where(support_count > 0, evidence,
                    evidence.new_full((), -1.))
                block_sum = (evidence * tissue_weight).reshape(
                    1, 2, 8, 4, 8, 4).sum((3, 5))
                block_mean = block_sum / block_weight.clamp_min(1e-6)
                evidence_score = ((block_mean * occupied).sum((2, 3)) /
                    occupied.sum((2, 3)).clamp_min(1))[0]
                for local in range(2):
                    present = support_count[0, local] > 0
                    by_step[step].append({**common, 'step': step,
                        'beam_slot': start + local, 'branch_id': int(pair[0, local]),
                        'initial_state': state[0, local].tolist(),
                        'fitted_state': fitted['state'][0, local].tolist(),
                        'mapped96_error_um': pair_errors[local],
                        'old094_score': float(pair_old_scores[0, local]),
                        'prior_score': float(prior[0, pair[0, local]]),
                        'fit_energy': float(fitted['fit_energy'][0, local]),
                        'warp_cost': float(warp_cost[0, local]),
                        'evidence_score': float(evidence_score[local]),
                        'predicted_tissue_sites': tissue_sites,
                        'populated_blocks': occupied_blocks,
                        'atlas_available_tissue_fraction': (
                            float((present & predicted_tissue[0, 0]).sum() / tissue_sites)
                            if tissue_sites else None),
                        'mean_supported_bin_count': (
                            float(support_count[0, local][predicted_tissue[0, 0]].float().mean())
                            if tissue_sites else None)})
        assert len(errors) == len(old_scores) == 14
        old_slot = int(np.argmax(old_scores))
        oracle_slot = int(np.argmin(errors))
        for step in steps:
            rows = by_step[step]
            assert len(rows) == 14 and [row['beam_slot'] for row in rows] == list(range(14))
            selected_slot = (old_slot if rows[0]['populated_blocks'] == 0 else
                int(np.argmax([row['evidence_score'] for row in rows])))
            for row in rows:
                row['old094_selected'] = row['beam_slot'] == old_slot
                row['evidence_selected'] = row['beam_slot'] == selected_slot
                row['physical_best'] = row['beam_slot'] == oracle_slot
                candidate_rows.append(row)
            selected_row = rows[selected_slot]
            support_count = selected_row['mean_supported_bin_count']
            support_stratum = ('none' if support_count is None else
                '<32' if support_count < 32 else '32-96' if support_count < 96 else '>=96')
            old_support_count = rows[old_slot]['mean_supported_bin_count']
            old_support_stratum = ('none' if old_support_count is None else
                '<32' if old_support_count < 32 else '32-96' if old_support_count < 96 else '>=96')
            section_rows.append({**common, 'step': step,
                'beam_branch_ids': branch[0].tolist(),
                'old094_selected_slot': old_slot,
                'evidence_selected_slot': selected_slot,
                'physical_best_slot': oracle_slot,
                'old094_selected_error_um': errors[old_slot],
                'evidence_selected_error_um': errors[selected_slot],
                'physical_best_error_um': errors[oracle_slot],
                'selection_regret_um': errors[selected_slot] - errors[oracle_slot],
                'selected_evidence_score': selected_row['evidence_score'],
                'selected_atlas_available_tissue_fraction':
                    selected_row['atlas_available_tissue_fraction'],
                'selected_mean_supported_bin_count': support_count,
                'selected_support_count_stratum': support_stratum,
                'old094_support_count_stratum': old_support_stratum,
                'predicted_tissue_sites': selected_row['predicted_tissue_sites'],
                'populated_blocks': selected_row['populated_blocks'],
                'fell_back_to_old094': selected_row['populated_blocks'] == 0})
        if record_index % 32 == 0:
            print(json.dumps({'eligible_sections_done': record_index,
                'eligible_sections_total': len(records)}), flush=True)

assert len(candidate_rows) == len(records) * 14 * len(steps)
assert len(section_rows) == len(records) * len(steps)
paired = {(row['section_id'], row['beam_slot'], row['step']): row for row in candidate_rows}
for record in records:
    for slot in range(14):
        zero = paired[(record['section_id'], slot, 0)]
        for step in steps[1:]:
            current = paired[(record['section_id'], slot, step)]
            for key in ('branch_id', 'initial_state', 'fitted_state',
                        'mapped96_error_um', 'old094_score',
                        'predicted_tissue_sites', 'populated_blocks',
                        'atlas_available_tissue_fraction',
                        'mean_supported_bin_count'):
                assert zero[key] == current[key]

def aggregate(group):
    by_plan = {}
    for plan in plan_ids:
        rows = [row for row in group if row['synthetic_subject_plan_id'] == plan]
        by_plan[plan] = {'sections': len(rows),
            **{key: avg([row[key] for row in rows]) for key in (
                'old094_selected_error_um', 'evidence_selected_error_um',
                'physical_best_error_um', 'selection_regret_um',
                'selected_atlas_available_tissue_fraction',
                'selected_mean_supported_bin_count', 'populated_blocks')}}
    metrics = tuple(key for key in next(iter(by_plan.values())) if key != 'sections')
    return {'sections': len(group), 'by_plan': by_plan,
        'plan_equal_mean': {key: avg([row[key] for row in by_plan.values()])
                            for key in metrics}}

summary = {'eligible_sections_per_step': len(records),
    'ineligible_sections_excluded': panel_receipt['ineligible'],
    'candidates_per_step': 14 * len(records),
    'step0_max_099_logit_difference': step0_max_logit_difference,
    'step0_max_099_support_difference': step0_max_support_difference,
    'steps': {}}
for step in steps:
    rows = [row for row in section_rows if row['step'] == step]
    groups = {'all': rows}
    for mode in panel_protocol['modes']:
        groups[f'appearance/{mode}'] = [row for row in rows
            if row['appearance_mode'] == mode]
    for family in ('AP', 'DV', 'ML'):
        groups[f'angle_family/{family}'] = [row for row in rows
            if row['angle_family'] == family]
    for bin_name in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'angle_bin/{bin_name}'] = [row for row in rows
            if row['angle_bin'] == bin_name]
    for stratum in ('none', '<32', '32-96', '>=96'):
        groups[f'support_count/{stratum}'] = [row for row in rows
            if row['old094_support_count_stratum'] == stratum]
    summary['steps'][str(step)] = {name: aggregate(group)
        for name, group in groups.items()}

trained_steps = steps[1:]
best_step = min(trained_steps, key=lambda step: (
    summary['steps'][str(step)]['all']['plan_equal_mean']['evidence_selected_error_um'], step))
best = summary['steps'][str(best_step)]
zero = summary['steps']['0']
old_error = best['all']['plan_equal_mean']['old094_selected_error_um']
zero_error = zero['all']['plan_equal_mean']['evidence_selected_error_um']
best_error = best['all']['plan_equal_mean']['evidence_selected_error_um']
plan_conditions = {plan: (
    best['all']['by_plan'][plan]['evidence_selected_error_um'] <
        best['all']['by_plan'][plan]['old094_selected_error_um']
    and best['all']['by_plan'][plan]['evidence_selected_error_um'] <
        zero['all']['by_plan'][plan]['evidence_selected_error_um'])
    for plan in plan_ids}
regression_groups = ([f'appearance/{mode}' for mode in panel_protocol['modes']]
    + [f'angle_family/{family}' for family in ('AP', 'DV', 'ML')])
group_conditions = {name: (
    best[name]['plan_equal_mean']['evidence_selected_error_um'] <=
        best[name]['plan_equal_mean']['old094_selected_error_um'] + 200
    and best[name]['plan_equal_mean']['evidence_selected_error_um'] <=
        zero[name]['plan_equal_mean']['evidence_selected_error_um'] + 200)
    for name in regression_groups}
lower_support_improvement = {name: (
    best[name]['sections'] > 0 and
    best[name]['plan_equal_mean']['evidence_selected_error_um'] <
        best[name]['plan_equal_mean']['old094_selected_error_um'] and
    best[name]['plan_equal_mean']['evidence_selected_error_um'] <
        zero[name]['plan_equal_mean']['evidence_selected_error_um'])
    for name in ('support_count/<32', 'support_count/32-96')}
lower_support_available = any(best[name]['sections'] > 0 for name in lower_support_improvement)
conditions = {'improves_old094_by_0.20_mm': old_error - best_error >= 200,
    'improves_step0_by_0.20_mm': zero_error - best_error >= 200,
    'improves_both_in_at_least_six_plans': sum(plan_conditions.values()) >= 6,
    'no_appearance_or_angle_family_regression_over_0.20_mm':
        all(group_conditions.values()),
    'gain_not_confined_to_high_support_count': any(lower_support_improvement.values())}
summary['best_development_step'] = best_step
summary['best_development_selected_error_um'] = best_error
summary['pilot_gate'] = {'conditions': conditions, 'passed': all(conditions.values()),
    'status': ('inconclusive' if not lower_support_available else
               'passed' if all(conditions.values()) else 'failed'),
    'plan_improves_both': plan_conditions, 'group_nonregression': group_conditions,
    'lower_support_stratum_improves_both': lower_support_improvement,
    'old094_error_um': old_error, 'step0_evidence_error_um': zero_error}
summary['checkpoint_selection_note'] = 'best of 1000/2000/3000/4000 on this same DEV panel; development-selected, not unbiased confirmation'

config = {'protocol_sha256': sha(protocol), 'dev_protocol_sha256': sha(dev_protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_cross_candidate_match_103.py',
        'train_cross_candidate_match_103.py',
        'prepare_cross_candidate_match_103_fresh_dev_plans.py',
        'prepare_cross_candidate_match_103_fresh_dev_panel.py',
        'contextual_matcher_099.py', 'spatial_joint_fit_094.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_checkpoint_sha256': sha(parent_file),
    'baseline_099_completed_sha256': sha(baseline_run / 'completed.json'),
    'baseline_099_checkpoint_sha256': sha(baseline_file),
    'train_103_completed_sha256': sha(train_run / 'completed.json'),
    'train_103_config_sha256': sha(train_run / 'config.json'),
    'train_103_draws_sha256': sha(train_run / 'draws.jsonl'),
    'matcher_checkpoint_sha256': {str(step): sha(checkpoint_files[step])
                                   for step in steps},
    'plans_completed_sha256': sha(plans / 'completed.json'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'fixed_geometry': '094 step20000 predictor; natural prior top8 old + top6 anchor; frozen 094 teacher fits and maps all 14',
    'candidate_score': 'support-corrected cosine; logmeanexp(20*cosine over fine bins); no support=-1; predicted validity>=0.5; validity-weighted 4x4 block means; equal occupied-block mean',
    'checkpoint_selection': 'best trained checkpoint by plan-equal evidence-selected mapped96 error on this DEV panel; development-selected',
    'physical_error': 'mean CCF distance at 1024 deterministic valid observed pixels sampled from frozen mapped96',
    'angle_bins': 'nearest-axis 0-15, 15-30, 30-45, 45-54.7 degrees',
    'support_count_strata': 'fixed by old094-selected candidate mean supported fine-bin count at predicted tissue sites: none, <32, [32,96), >=96; selected candidate count also logged',
    'pilot_gate': 'best trained step must improve plan-equal selected mapped error by >=200um vs old094 and step0, improve both in >=6/8 plans, have no appearance or nearest-axis family regression >200um vs either, and improve both in at least one fixed lower-support stratum',
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'eligible_sections_per_step': len(records),
    'candidates_per_step': len(records) * 14,
    'protocol_sha256': config['protocol_sha256'],
    'parent_094_checkpoint_sha256': config['parent_094_checkpoint_sha256'],
    'baseline_099_checkpoint_sha256': config['baseline_099_checkpoint_sha256'],
    'train_103_completed_sha256': config['train_103_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'matcher_checkpoint_sha256': config['matcher_checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'eligible_sections': len(records),
    'best_development_step': best_step, 'pilot_gate': summary['pilot_gate']}), flush=True)
