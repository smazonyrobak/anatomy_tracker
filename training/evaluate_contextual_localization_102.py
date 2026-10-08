"""Held-out 102A fine-match localization on the frozen 094 beam and fitted planes."""
import hashlib
import json
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.contextual_matcher_099 import WholeSliceAtlasFeedback099
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/CONTEXTUAL_LOCALIZATION_102_PROTOCOL_20261008.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent_file = parent_run / 'joint_step_20000.pt'
baseline_run = root / 'runs/contextual_match_099_pilot'
baseline_file = baseline_run / 'contextual_step_04000.pt'
train_run = root / 'runs/contextual_localization_102_pilot'
plans = root / 'data/contextual_localization_102_fresh_dev_plans_001'
panel = root / 'data/contextual_localization_102_fresh_dev_panel_001'
out = root / 'runs/contextual_localization_102_fresh_development_eval'
steps = (0, 2000, 4000)
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
checkpoint_files = {step: train_run / f'localization_step_{step:05d}.pt' for step in steps}
assert all(sha(checkpoint_files[step]) == train_receipt['checkpoint_sha256'][str(step)]
           for step in steps)

plan_receipt = json.loads((plans / 'completed.json').read_text())
plan_protocol = json.loads((plans / 'protocol.json').read_text())
assert plan_receipt['protocol'] == plan_protocol
assert len(plan_receipt['subjects']) == 8 and plan_receipt['section_count'] == 0
assert all(sha(plans / name) == digest for name, digest in
           plan_receipt['artifact_sha256'].items())
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_receipt['physical_sections'] == 256
assert panel_protocol['synthetic_subjects'] == 8 and panel_protocol['planes_per_subject'] == 32
assert panel_protocol['side'] == 256
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
step0_max_logit_diff = 0.
step0_max_coordinate_diff_um = 0.
with torch.inference_mode():
    for index, record in enumerate(records, 1):
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
        normal = np.abs(np.asarray(record['plane_normal_ap_dv_ml'], dtype=float))
        family = ('AP', 'DV', 'ML')[int(normal.argmax())]
        angle = float(np.degrees(np.arccos(np.clip(normal.max(), 0, 1))))
        angle_bin = ('0-15', '15-30', '30-45', '45-54.7')[min(int(angle // 15), 3)]
        visible_fraction = record['valid_pixels'] / 256 ** 2
        visibility_bin = ('<0.1' if visible_fraction < .1 else
                          '0.1-0.3' if visible_fraction < .3 else '>=0.3')
        common = {key: record[key] for key in ('section_id', 'animal_id',
            'specimen_id', 'experiment_id', 'synthetic_animal_id',
            'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
            'panel_physical_section_id', 'appearance_mode', 'section_index')}
        common.update({'source_file': record['file'], 'source_sha256': record['sha256'],
            'sample_indices_sha256': chosen_hash, 'angle_family': family,
            'angle_from_family_axis_deg': angle, 'angle_bin_deg': angle_bin,
            'visible_fraction': visible_fraction, 'visibility_bin': visibility_bin})
        per_step = {step: [] for step in steps}
        mapped_errors = []
        for start in range(0, 14, 2):
            pair = branch[:, start:start + 2]
            reflected = pair % 2
            state = initial[:, start:start + 2]
            fitted = teacher(prediction['feature'], state, reflected,
                atlas, offsets, weights, source_shape=(256, 256))
            mapped_prediction = {**prediction, 'state': fitted['state'],
                'log_mass': prediction['log_mass'].gather(1, pair // 2),
                'reflection_logit': prediction['reflection_logit'].gather(1, pair // 2)}
            mapped = model.map(mapped_prediction, offsets,
                torch.arange(2, device='cuda')[None], reflected, (96, 96),
                atlas, weights, feature_side=96, source_shape=(256, 256),
                spatial_evidence=fitted['spatial_evidence'])
            mapped_errors.extend((at_chart(mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                - reference[None]).norm(dim=-1).mean(-1).tolist())

            centre, frame, basis = full_frame_state_to_components(fitted['state'])
            edges = frame[..., :, :2] @ basis
            axis = (torch.arange(32, device='cuda') + .5) / 32 - .5 / 256
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            chart_x = torch.where(reflected.bool()[..., None, None],
                255 / 256 - xx, xx)
            site_chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
            plane_ccf = centre[..., None, None, :] + torch.einsum(
                'bkij,bkhwj->bkhwi', edges, site_chart - .5)
            true_sites = F.interpolate(truth.permute(0, 3, 1, 2),
                (32, 32), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            visible = F.interpolate(valid[:, None].float(), (32, 32),
                mode='bilinear', align_corners=False)[:, 0] == 1
            classes = torch.arange(225, device='cuda')
            depth = classes // 25 - 4
            dy = classes // 5 % 5 - 2
            dx = classes % 5 - 2
            lateral = torch.stack((torch.where(reflected.bool()[..., None], -dx, dx),
                dy.expand(1, 2, -1)), -1).float() / 32
            shift = torch.einsum('bkij,bkcj->bkci', edges, lateral)
            bin_ccf = (plane_ccf[:, :, None] + shift[:, :, :, None, None] +
                1500 * depth[None, None, :, None, None, None] *
                frame[..., :, 2][:, :, None, None, None, :])
            distance = (bin_ccf - true_sites[:, None, None]).norm(dim=-1)

            for step in steps:
                student = students[step]
                match = student(prediction['feature'], fitted['state'], reflected,
                    atlas, offsets, weights, source_shape=(256, 256), match_only=True)
                logits = match['fine_match_logits']
                support = match['fine_match_support']
                assert torch.equal(support, fitted['fine_match_support'])
                verified, _, _, available = teacher._verify('fine', logits, support,
                    prediction['feature'], state.new_zeros(1, 2, 64, 32, 32),
                    fitted['state'], reflected, (256, 256))
                if step == 0 and index == 1 and start == 0:
                    baseline_match = baseline_student(prediction['feature'], fitted['state'],
                        reflected, atlas, offsets, weights, source_shape=(256, 256),
                        match_only=True)
                    baseline_logits = baseline_match['fine_match_logits']
                    assert torch.equal(support, baseline_match['fine_match_support'])
                    baseline_verified, _, _, _ = teacher._verify('fine', baseline_logits,
                        support, prediction['feature'], state.new_zeros(1, 2, 64, 32, 32),
                        fitted['state'], reflected, (256, 256))
                    step0_max_logit_diff = max(step0_max_logit_diff,
                        float((logits - baseline_logits).abs().max()))
                    step0_max_coordinate_diff_um = max(step0_max_coordinate_diff_um,
                        float((verified - baseline_verified).abs().max()))
                    assert step0_max_logit_diff == 0
                    assert step0_max_coordinate_diff_um == 0
                supported = support >= .5
                mask = visible[:, None] & supported.any(2)
                assert torch.equal(available >= .5, supported.any(2))
                temperature = student.log_temperature.exp().clamp(2, 20)
                cosine = (logits + 4 * (1 - support)) / temperature
                peak = cosine.masked_fill(~supported, -1e4).argmax(2)
                raw_correct = distance.gather(2, peak[:, :, None]).squeeze(2) <= 1500
                decoded_error = (verified - true_sites[:, None]).norm(dim=-1)
                null_error = (plane_ccf - true_sites[:, None]).norm(dim=-1)
                for local in range(2):
                    slot = start + local
                    valid_sites = mask[0, local]
                    count = int(valid_sites.sum())
                    per_step[step].append({**common, 'step': step,
                        'beam_slot': slot, 'branch_id': int(pair[0, local]),
                        'initial_state': state[0, local].tolist(),
                        'fitted_state': fitted['state'][0, local].tolist(),
                        'mapped96_error_um': mapped_errors[slot],
                        'visible_sites': int(visible[0].sum()),
                        'atlas_available_visible_sites': count,
                        'atlas_support_fraction': float(supported.any(2)[0, local].float().mean()),
                        'raw_top1_fraction': (float(raw_correct[0, local][valid_sites].float().mean())
                                              if count else None),
                        'decoded_correct_fraction': (float((decoded_error[0, local][valid_sites]
                            <= 1500).float().mean()) if count else None),
                        'decoded_error_um': (float(decoded_error[0, local][valid_sites].mean())
                                             if count else None),
                        'null_correct_fraction': (float((null_error[0, local][valid_sites]
                            <= 1500).float().mean()) if count else None),
                        'null_error_um': (float(null_error[0, local][valid_sites].mean())
                                          if count else None)})
        assert len(mapped_errors) == 14
        best = int(np.argmin(mapped_errors))
        for step in steps:
            rows = per_step[step]
            assert len(rows) == 14 and [row['beam_slot'] for row in rows] == list(range(14))
            for row in rows:
                row['physical_best'] = row['beam_slot'] == best
                candidate_rows.append(row)
            section_rows.append({**common, 'step': step,
                'beam_branch_ids': branch[0].tolist(), 'physical_best_beam_slot': best,
                'best_mapped96_error_um': mapped_errors[best],
                **{f'all_{key}': avg([row[key] for row in rows]) for key in (
                    'raw_top1_fraction', 'decoded_correct_fraction', 'decoded_error_um',
                    'null_correct_fraction', 'null_error_um')},
                **{f'best_{key}': rows[best][key] for key in (
                    'raw_top1_fraction', 'decoded_correct_fraction', 'decoded_error_um',
                    'null_correct_fraction', 'null_error_um')}})
        if index % 32 == 0:
            print(json.dumps({'eligible_sections_done': index,
                              'eligible_sections_total': len(records)}), flush=True)

assert len(candidate_rows) == len(records) * 14 * len(steps)
assert len(section_rows) == len(records) * len(steps)
paired = {(row['section_id'], row['beam_slot'], row['step']): row for row in candidate_rows}
for record in records:
    for slot in range(14):
        zero = paired[(record['section_id'], slot, 0)]
        for step in steps[1:]:
            current = paired[(record['section_id'], slot, step)]
            for key in ('branch_id', 'initial_state', 'fitted_state', 'mapped96_error_um',
                        'atlas_support_fraction', 'null_correct_fraction', 'null_error_um'):
                assert zero[key] == current[key]

metrics = ('raw_top1_fraction', 'decoded_correct_fraction', 'decoded_error_um',
           'null_correct_fraction', 'null_error_um')
summary = {'eligible_sections_per_step': len(records),
    'ineligible_sections_excluded': panel_receipt['ineligible'],
    'candidates_per_step': 14 * len(records),
    'first_pair_step0_099_logit_difference': step0_max_logit_diff,
    'first_pair_step0_099_decoded_coordinate_difference_um': step0_max_coordinate_diff_um,
    'steps': {}}
for step in steps:
    selected = [row for row in candidate_rows if row['step'] == step]
    groups = {'all': selected,
        'physical_best': [row for row in selected if row['physical_best']]}
    for mode in panel_protocol['modes']:
        groups[f'appearance/{mode}/all'] = [row for row in groups['all']
            if row['appearance_mode'] == mode]
        groups[f'appearance/{mode}/physical_best'] = [row for row in groups['physical_best']
            if row['appearance_mode'] == mode]
    for family in ('AP', 'DV', 'ML'):
        for kind in ('all', 'physical_best'):
            groups[f'angle_family/{family}/{kind}'] = [row for row in groups[kind]
                if row['angle_family'] == family]
    for bin_name in ('0-15', '15-30', '30-45', '45-54.7'):
        for kind in ('all', 'physical_best'):
            groups[f'angle_bin/{bin_name}/{kind}'] = [row for row in groups[kind]
                if row['angle_bin_deg'] == bin_name]
    for bin_name in ('<0.1', '0.1-0.3', '>=0.3'):
        for kind in ('all', 'physical_best'):
            groups[f'visibility/{bin_name}/{kind}'] = [row for row in groups[kind]
                if row['visibility_bin'] == bin_name]
    summary['steps'][str(step)] = {}
    for name, group in groups.items():
        by_plan = {}
        for plan in plan_ids:
            plan_rows = [row for row in group if row['synthetic_subject_plan_id'] == plan]
            sections = sorted({row['section_id'] for row in plan_rows})
            by_plan[plan] = {'sections': len(sections), 'candidates': len(plan_rows),
                **{key: avg([avg([row[key] for row in plan_rows
                    if row['section_id'] == section]) for section in sections])
                   for key in metrics}}
        summary['steps'][str(step)][name] = {'sections': len({row['section_id'] for row in group}),
            'candidates': len(group), 'by_plan': by_plan,
            'plan_equal_mean': {key: avg([row[key] for row in by_plan.values()])
                                for key in metrics}}

baseline = summary['steps']['0']
summary['feedback_gates'] = {}
for step in steps[1:]:
    arm = summary['steps'][str(step)]
    all_delta = (arm['all']['plan_equal_mean']['decoded_correct_fraction'] -
                 arm['all']['plan_equal_mean']['null_correct_fraction'])
    best_delta = (arm['physical_best']['plan_equal_mean']['decoded_correct_fraction'] -
                  arm['physical_best']['plan_equal_mean']['null_correct_fraction'])
    plan_deltas = {plan: {kind: (arm[kind]['by_plan'][plan]['decoded_correct_fraction'] -
        baseline[kind]['by_plan'][plan]['decoded_correct_fraction'])
        for kind in ('all', 'physical_best')} for plan in plan_ids}
    regressions = {name: {kind: (arm[f'{name}/{kind}']['plan_equal_mean']['decoded_correct_fraction'] -
        baseline[f'{name}/{kind}']['plan_equal_mean']['decoded_correct_fraction'])
        for kind in ('all', 'physical_best')}
        for name in ([f'appearance/{mode}' for mode in panel_protocol['modes']] +
                     [f'angle_family/{family}' for family in ('AP', 'DV', 'ML')])}
    conditions = {
        'best_beats_unshifted_by_0.03': best_delta >= .03,
        'all_beats_unshifted_by_0.02': all_delta >= .02,
        'six_of_eight_plans_improve_best': sum(row['physical_best'] > 0
            for row in plan_deltas.values()) >= 6,
        'six_of_eight_plans_improve_all': sum(row['all'] > 0
            for row in plan_deltas.values()) >= 6,
        'no_family_or_appearance_drop_worse_than_0.03': all(value >= -.03
            for row in regressions.values() for value in row.values())}
    summary['feedback_gates'][str(step)] = {'best_minus_null': best_delta,
        'all_minus_null': all_delta, 'per_plan_vs_099': plan_deltas,
        'family_appearance_vs_099': regressions, 'conditions': conditions,
        'passed': all(conditions.values())}
summary['primary_checkpoint'] = 4000
summary['primary_feedback_gate_passed'] = summary['feedback_gates']['4000']['passed']

config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_contextual_localization_102.py', 'train_contextual_localization_102.py',
        'evaluate_contextual_match_099.py', 'contextual_matcher_099.py',
        'spatial_joint_fit_094.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_checkpoint_sha256': sha(parent_file),
    'baseline_099_completed_sha256': sha(baseline_run / 'completed.json'),
    'baseline_099_checkpoint_sha256': sha(baseline_file),
    'train_102_completed_sha256': sha(train_run / 'completed.json'),
    'train_102_config_sha256': sha(train_run / 'config.json'),
    'train_102_draws_sha256': sha(train_run / 'draws.jsonl'),
    'matcher_checkpoint_sha256': {str(step): sha(checkpoint_files[step]) for step in steps},
    'plans_completed_sha256': sha(plans / 'completed.json'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'fixed_geometry': '094 step20000 predictor; prior top eight branch IDs 0:31 plus top six IDs 32:63; 094 teacher fits all fourteen once; match on these fitted states',
    'physical_best': 'minimum frozen094 mapped96 error over 1024 deterministic surviving observed pixels; oracle readout only',
    'match_readout': 'fine 32x32 grid, 225 bins, support>=0.5, visible observed site, exact 094 argmax-neighborhood local softmax decoder',
    'raw_top1': 'argmax normalized descriptor cosine over supported bins; fraction within 1.5mm',
    'unshifted_null': 'frozen094 fitted candidate plane coordinate at each visible atlas-available site, without matcher displacement; same denominator',
    'denominator': 'per-candidate mean over visible atlas-available sites; candidate mean within section; section mean within plan; equal mean over nonempty plans',
    'angle_family': 'nearest AP, DV or ML normal axis; angle bins 0-15, 15-30, 30-45, 45-54.7 degrees',
    'visibility_bins': 'valid observed pixels / 65536: <0.1, [0.1,0.3), >=0.3',
    'feedback_gate': '4000 is primary; 2000 is learning-curve diagnostic. Each checkpoint separately: decoded correct fraction minus null >=0.03 physical-best and >=0.02 all; positive vs 099 in >=6/8 plans for both; no appearance or nearest-axis family drop below -0.03 for either',
    'checkpoint_selection': 'none; 4000 is the prespecified primary gate, and a pass at only 2000 is exploratory pending fresh confirmation',
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
    'train_102_completed_sha256': config['train_102_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'matcher_checkpoint_sha256': config['matcher_checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'eligible_sections': len(records),
    'feedback_gates': summary['feedback_gates']}), flush=True)
