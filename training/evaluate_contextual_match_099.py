"""Fresh-panel 099 descriptor readout on fixed 094 beam and fitted geometry."""
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
protocol = source.parent / 'docs/publication/CONTEXTUAL_MATCH_099_PROTOCOL_20261008.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent_checkpoint = parent_run / 'joint_step_20000.pt'
train_run = root / 'runs/contextual_match_099_pilot'
plans_dir = root / 'data/contextual_match_099_fresh_synthetic_dev_plans_001'
panel = root / 'data/contextual_match_099_fresh_synthetic_dev_panel_001'
out = root / 'runs/contextual_match_099_fresh_development_eval'
steps = (0, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def at_chart(surface, chart):
    count = surface.shape[1]
    grid = ((chart + .5 / 256) * 2 - 1)[None].expand(
        count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
        padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


assert root.drive.upper() == source.drive.upper() == 'I:'
assert not out.exists()
assert sha(protocol) == '0aa387fd8db0517ebe10271c75da8610ccdbe84f79c07f9ed07a0f9dba52d974'
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent_checkpoint) == parent_receipt['checkpoint_sha256']['20000'] == \
    '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert sha(parent_run / 'draws.jsonl') == parent_receipt['draws_sha256']
assert sha(parent_run / 'training.jsonl') == parent_receipt['training_sha256']
assert all(sha(source / name) == digest for name, digest in
           parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))

train_receipt = json.loads((train_run / 'completed.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
assert train_receipt['batches'] == train_receipt['accepted_synthetic'] == \
    train_receipt['distinct_physical_sections'] == train_config['batches'] == 4000
assert tuple(train_config['checkpoints']) == steps
assert train_config['protocol_sha256'] == train_receipt['protocol_sha256'] == sha(protocol)
assert train_config['parent_094_sha256'] == \
    train_receipt['parent_094_checkpoint_sha256'] == sha(parent_checkpoint)
assert train_config['parent_094_completed_sha256'] == sha(parent_run / 'completed.json')
assert train_config['source_sha256'] == train_receipt['source_sha256']
assert all(sha(source / name) == digest for name, digest in
           train_config['source_sha256'].items())
assert sha(train_run / 'config.json') == train_receipt['config_sha256']
assert sha(train_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train_run / 'training.jsonl') == train_receipt['training_sha256']
assert not any(train_receipt[key] or train_config[key] for key in
    ('calibrated', 'public_benchmark_used', 'real_labels_used',
     'external_pretrained_weights_used'))
checkpoints = {step: train_run / f'contextual_step_{step:05d}.pt' for step in steps}
assert all(sha(checkpoints[step]) == train_receipt['checkpoint_sha256'][str(step)]
           for step in steps)
train_draws = [row for row in map(json.loads, (train_run / 'draws.jsonl').open())
               if row['used']]
assert len(train_draws) == 4000
assert len({row['physical_section_id'] for row in train_draws}) == 4000
assert all(row['split'] == 'train' for row in train_draws)

plans_receipt = json.loads((plans_dir / 'completed.json').read_text())
plans_protocol = json.loads((plans_dir / 'protocol.json').read_text())
assert plans_receipt['protocol'] == plans_protocol
assert plans_receipt['section_count'] == 0 and len(plans_receipt['subjects']) == 8
assert plans_protocol['subject_count'] == 8
assert all(sha(plans_dir / name) == digest for name, digest in
           plans_receipt['artifact_sha256'].items())
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_receipt['physical_sections'] == panel_protocol['physical_sections'] == 256
assert panel_receipt['synthetic_subjects'] == panel_protocol['synthetic_subjects'] == 8
assert panel_protocol['planes_per_subject'] == 32 and panel_protocol['side'] == 256
assert panel_receipt['protocol_sha256'] == sha(panel / 'protocol.json')
assert panel_receipt['records_sha256'] == sha(panel / 'records.jsonl')
assert panel_protocol['source_plan_completed_sha256'] == \
    panel_receipt['plan_completed_sha256'] == sha(plans_dir / 'completed.json')
assert panel_protocol['train_lineage'] == plans_protocol['train_lineage']
assert all(sha(panel / 'source' / name) == digest for name, digest in
           panel_protocol['source_sha256'].items())
assert all(sha(root / 'data' / name / 'records.jsonl') == digest for name, digest in
           panel_protocol['prior_panel_records_sha256'].items())
prior_records = [row for name in panel_protocol['prior_panel_records_sha256']
                 for row in map(json.loads,
                     (root / 'data' / name / 'records.jsonl').open())]
all_records = list(map(json.loads, (panel / 'records.jsonl').open()))
assert len(all_records) == 256
assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
assert len({row['section_id'] for row in all_records}) == 256
assert len({row['panel_physical_section_id'] for row in all_records}) == 256
assert len({row['subject_ouv_sha256'] for row in all_records}) == 256
plans = {row['subject_deformation_plan_id']: row for row in plans_receipt['subjects']}
assert len(plans) == 8
assert all(sum(row['synthetic_subject_plan_id'] == plan for row in all_records) == 32
           for plan in plans)
assert all(row['plan_receipt_sha256'] ==
           plans[row['synthetic_subject_plan_id']]['subject_plan_receipt_sha256']
           for row in all_records)
assert all(row['animal_id'] == plans[row['synthetic_subject_plan_id']]['animal_id'] and
           row['specimen_id'] == plans[row['synthetic_subject_plan_id']]['specimen_id'] and
           row['experiment_id'] == plans[row['synthetic_subject_plan_id']]['experiment_id']
           for row in all_records)
for key in ('animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id'):
    assert not {row[key] for row in all_records} & \
        {row['base_lineage'][key] for row in train_draws}
for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id'):
    assert not {row[key] for row in all_records} & {row[key] for row in prior_records}
assert not {row['plan_receipt_sha256'] for row in all_records} & \
    {row['plan_receipt_sha256'] for row in prior_records}
assert not {row['synthetic_subject_plan_id'] for row in all_records} & \
    {row['base_lineage']['subject_deformation_plan_id'] for row in train_draws}
assert not {row['panel_physical_section_id'] for row in all_records} & \
    {row['physical_section_id'] for row in train_draws}
assert sum(row['eligible'] for row in all_records) == panel_receipt['eligible']
assert sum(not row['eligible'] for row in all_records) == panel_receipt['ineligible']
assert {mode: sum(row['appearance_mode'] == mode for row in all_records)
        for mode in panel_protocol['modes']} == panel_receipt['modes']
records = [row for row in all_records if row['eligible']]
plan_ids = sorted(plans)
assert all(any(row['synthetic_subject_plan_id'] == plan for row in records)
           for plan in plan_ids)
assert all(any(row['appearance_mode'] == mode for row in records)
           for mode in panel_protocol['modes'])

saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and saved['calibrated'] is False
parent_matcher = {key.removeprefix('matcher.'): value for key, value in
                  saved['spatial_fit'].items() if key.startswith('matcher.')}
matcher_states = {}
for step in steps:
    checkpoint = torch.load(checkpoints[step], map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['calibrated'] is False
    assert checkpoint['config'] == train_config
    matcher_states[step] = checkpoint['matcher']
assert set(matcher_states[0]) == set(matcher_states[4000])
for key, value in parent_matcher.items():
    wrapped = (key.replace('image.', 'image.base.', 1) if key.startswith('image.') else
        key.replace('atlas.', 'atlas.base.', 1) if key.startswith('atlas.') else key)
    assert torch.equal(matcher_states[0][wrapped], value)
    assert torch.equal(matcher_states[4000][wrapped], value)
assert all(torch.equal(matcher_states[0][key], matcher_states[4000][key])
           for key in matcher_states[0] if '.context.' not in key)

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
teacher = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
teacher.load_state_dict(saved['spatial_fit'], strict=True)
student = WholeSliceAtlasFeedback099().cuda().eval().requires_grad_(False)
assert set(student.state_dict()) == set(matcher_states[0])
del saved

candidate_rows, section_rows = [], []
max_step0_logit_difference = 0.
max_step0_verified_coordinate_difference_um = 0.
max_fixed_support_difference = 0.
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
        chosen_sha256 = hashlib.sha256(
            chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
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
        section_candidates = {step: [] for step in steps}
        mapped_errors = []
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
            mapped_errors.extend((at_chart(
                mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                - reference[None]).norm(dim=-1).mean(-1).tolist())

            geometry = {}
            for name, grid, radius, positive_um, negative_um in (
                ('fine', 32, 2, 1500, 2000), ('coarse', 16, 4, 2000, 3000)):
                basis_state = fitted['state'] if name == 'fine' else state
                centre, frame, basis = full_frame_state_to_components(basis_state)
                edges = frame[..., :, :2] @ basis
                visible = F.interpolate(valid[:, None].float(), (grid, grid),
                    mode='bilinear', align_corners=False)[:, 0] == 1
                true_sites = F.interpolate(truth.permute(0, 3, 1, 2),
                    (grid, grid), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / 256
                yy, xx = torch.meshgrid(axis, axis, indexing='ij')
                chart_x = torch.where(reflected.bool()[..., None, None],
                    255 / 256 - xx, xx)
                site_chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
                centre_ccf = centre[..., None, None, :] + torch.einsum(
                    'bkij,bkhwj->bkhwi', edges, site_chart - .5)
                width = 2 * radius + 1
                index = torch.arange(9 * width * width, device='cuda')
                depth = index // (width * width) - 4
                dy = index // width % width - radius
                dx = index % width - radius
                lateral = torch.stack((torch.where(
                    reflected.bool()[..., None], -dx, dx),
                    dy.expand(1, 2, -1)), -1).to(state.dtype) / grid
                shift = torch.einsum('bkij,bkcj->bkci', edges, lateral)
                bin_ccf = (centre_ccf[:, :, None] +
                    shift[:, :, :, None, None] +
                    1500 * depth[None, None, :, None, None, None] *
                    frame[..., :, 2][:, :, None, None, None, :])
                distance = (bin_ccf - true_sites[:, None, None]).norm(dim=-1)
                geometry[name] = (visible, true_sites, centre_ccf, distance,
                                  positive_um, negative_um)

            for step in steps:
                student.load_state_dict(matcher_states[step], strict=True)
                temperature = student.log_temperature.exp().clamp(2, 20)
                readout = {}
                for name in ('fine', 'coarse'):
                    basis_state = fitted['state'] if name == 'fine' else state
                    matched = student(prediction['feature'], basis_state, reflected,
                        atlas, offsets, weights, source_shape=(256, 256), match_only=True)
                    logits = matched[f'{name}_match_logits']
                    support = matched[f'{name}_match_support']
                    fixed_logits = fitted[f'{name}_match_logits']
                    fixed_support = fitted[f'{name}_match_support']
                    max_fixed_support_difference = max(max_fixed_support_difference,
                        float((support - fixed_support).abs().max()))
                    assert torch.equal(support, fixed_support)
                    if step == 0:
                        max_step0_logit_difference = max(max_step0_logit_difference,
                            float((logits - fixed_logits).abs().max()))
                        assert max_step0_logit_difference < 1e-5
                    visible, true_sites, centre_ccf, distance, positive_um, negative_um = \
                        geometry[name]
                    supported = support >= .5
                    available = supported.any(2)
                    mask = visible[:, None] & available
                    positive = supported & (distance <= positive_um)
                    negative = supported & (distance >= negative_um)
                    oracle = positive.any(2)
                    eligible = mask & oracle & negative.any(2)
                    cosine = (logits + 4 * (1 - support)) / temperature
                    raw_peak = cosine.masked_fill(~supported, -1e4).argmax(2)
                    raw_correct = distance.gather(2, raw_peak[:, :, None]).squeeze(2) <= positive_um
                    best_positive = cosine.masked_fill(~positive, -torch.inf).amax(2)
                    hard_negative = cosine.masked_fill(~negative, -torch.inf).amax(2)
                    readout[name] = {'visible': visible, 'available': available,
                        'mask': mask, 'oracle': oracle, 'eligible': eligible,
                        'raw_correct': raw_correct,
                        'hard_margin': best_positive - hard_negative,
                        'support_fraction': available.float().mean((-2, -1))}
                    if name == 'fine':
                        verified_ccf, _, _, verified_available = teacher._verify(
                            'fine', logits, support, prediction['feature'],
                            state.new_zeros(1, 2, 64, 32, 32), fitted['state'],
                            reflected, (256, 256))
                        assert torch.equal(available, verified_available >= .5)
                        if step == 0:
                            max_step0_verified_coordinate_difference_um = max(
                                max_step0_verified_coordinate_difference_um,
                                float((verified_ccf - fitted['fine_match_ccf_um']).abs().max()))
                            assert max_step0_verified_coordinate_difference_um < 1e-3
                        readout[name]['verified_correct'] = \
                            (verified_ccf - true_sites[:, None]).norm(dim=-1) <= positive_um
                        readout[name]['centre_correct'] = \
                            (centre_ccf - true_sites[:, None]).norm(dim=-1) <= positive_um

                for local in range(2):
                    row = {'step': step, 'section_id': record['section_id'],
                        'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                        'appearance_mode': record['appearance_mode'],
                        'source_file': record['file'], 'source_sha256': record['sha256'],
                        'sample_indices_sha256': chosen_sha256,
                        'beam_slot': start + local, 'branch_id': int(pair[0, local]),
                        'prior_score': float(prior[0, pair[0, local]]),
                        'initial_state': state[0, local].tolist(),
                        'fitted_state': fitted['state'][0, local].tolist(),
                        'mapped96_error_um': mapped_errors[start + local]}
                    for name in ('fine', 'coarse'):
                        result = readout[name]
                        mask = result['mask'][0, local]
                        eligible = result['eligible'][0, local]
                        count = int(mask.sum())
                        eligible_count = int(eligible.sum())
                        row[f'{name}_visible_sites'] = int(result['visible'][0].sum())
                        row[f'{name}_atlas_available_visible_sites'] = count
                        row[f'{name}_atlas_support_fraction'] = float(
                            result['support_fraction'][0, local])
                        row[f'{name}_oracle_fraction'] = (
                            int((result['oracle'][0, local] & mask).sum()) / count
                            if count else None)
                        row[f'{name}_raw_top1_fraction'] = (
                            int((result['raw_correct'][0, local] & mask).sum()) / count
                            if count else None)
                        row[f'{name}_hard_margin_cosine'] = (
                            float(result['hard_margin'][0, local][eligible].mean())
                            if eligible_count else None)
                        row[f'{name}_contrastive_eligible_sites'] = eligible_count
                        if name == 'fine':
                            row['fine_verified_fraction'] = (
                                int((result['verified_correct'][0, local] & mask).sum()) / count
                                if count else None)
                            row['fine_centre_fraction'] = (
                                int((result['centre_correct'][0, local] & mask).sum()) / count
                                if count else None)
                    fraction = row['fine_atlas_support_fraction']
                    row['atlas_support_stratum'] = (
                        'low_<0.5' if fraction < .5 else
                        'middle_0.5_to_0.7' if fraction < .7 else 'high_>=0.7')
                    section_candidates[step].append(row)
        best_slot = int(np.argmin(mapped_errors))
        assert len(mapped_errors) == 14
        for step in steps:
            rows = section_candidates[step]
            assert len(rows) == 14
            assert [row['beam_slot'] for row in rows] == list(range(14))
            for row in rows:
                row['best_physical_candidate'] = row['beam_slot'] == best_slot
                candidate_rows.append(row)
            section_rows.append({'step': step, 'section_id': record['section_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': record['appearance_mode'],
                'source_sha256': record['sha256'],
                'sample_indices_sha256': chosen_sha256,
                'beam_branch_ids': branch[0].tolist(),
                'best_physical_beam_slot': best_slot,
                'best_physical_mapped96_error_um': mapped_errors[best_slot],
                'mean_fine_raw_top1_fraction': mean(
                    [row['fine_raw_top1_fraction'] for row in rows]),
                'best_fine_raw_top1_fraction': rows[best_slot]['fine_raw_top1_fraction'],
                'mean_fine_verified_fraction': mean(
                    [row['fine_verified_fraction'] for row in rows]),
                'best_fine_verified_fraction': rows[best_slot]['fine_verified_fraction'],
                'mean_fine_centre_fraction': mean(
                    [row['fine_centre_fraction'] for row in rows]),
                'best_fine_centre_fraction': rows[best_slot]['fine_centre_fraction']})
        if record_index % 32 == 0:
            print(json.dumps({'eligible_sections_done': record_index,
                              'eligible_sections_total': len(records)}), flush=True)

assert len(candidate_rows) == 2 * len(records) * 14
assert len(section_rows) == 2 * len(records)
by_key = {(row['section_id'], row['beam_slot'], row['step']): row
          for row in candidate_rows}
for section in (row['section_id'] for row in records):
    for slot in range(14):
        zero = by_key[(section, slot, 0)]
        final = by_key[(section, slot, 4000)]
        for key in ('branch_id', 'initial_state', 'fitted_state', 'mapped96_error_um',
                    'best_physical_candidate', 'fine_atlas_support_fraction',
                    'coarse_atlas_support_fraction', 'fine_oracle_fraction',
                    'coarse_oracle_fraction', 'fine_centre_fraction'):
            assert zero[key] == final[key]

metrics = ('fine_oracle_fraction', 'fine_raw_top1_fraction',
    'fine_verified_fraction', 'fine_centre_fraction', 'fine_hard_margin_cosine',
    'coarse_oracle_fraction', 'coarse_raw_top1_fraction',
    'coarse_hard_margin_cosine')
summary = {'eligible_sections_per_step': len(records),
    'ineligible_sections_excluded': panel_receipt['ineligible'],
    'candidates_per_step': len(records) * 14,
    'max_step0_094_logit_difference': max_step0_logit_difference,
    'max_step0_094_verified_coordinate_difference_um':
        max_step0_verified_coordinate_difference_um,
    'max_fixed_support_difference': max_fixed_support_difference, 'steps': {}}
for step in steps:
    selected = [row for row in candidate_rows if row['step'] == step]
    groups = {'all': selected,
        'best_physical': [row for row in selected if row['best_physical_candidate']]}
    for mode in panel_protocol['modes']:
        for kind in ('all', 'best_physical'):
            groups[f'appearance/{mode}/{kind}'] = [row for row in groups[kind]
                if row['appearance_mode'] == mode]
    for stratum in ('low_<0.5', 'middle_0.5_to_0.7', 'high_>=0.7'):
        for kind in ('all', 'best_physical'):
            groups[f'support/{stratum}/{kind}'] = [row for row in groups[kind]
                if row['atlas_support_stratum'] == stratum]
    summary['steps'][str(step)] = {}
    for name, group in groups.items():
        by_plan = {}
        for plan in plan_ids:
            plan_rows = [row for row in group
                         if row['synthetic_subject_plan_id'] == plan]
            section_ids = sorted({row['section_id'] for row in plan_rows})
            by_section = {section: {key: mean([row[key] for row in plan_rows
                if row['section_id'] == section]) for key in metrics}
                for section in section_ids}
            by_plan[plan] = {'sections': len(section_ids),
                'candidates': len(plan_rows),
                **{key: mean([row[key] for row in by_section.values()])
                   for key in metrics}}
        summary['steps'][str(step)][name] = {'candidates': len(group),
            'by_plan': by_plan,
            'plan_equal_mean': {key: mean([row[key] for row in by_plan.values()])
                                for key in metrics}}
    assert summary['steps'][str(step)]['best_physical']['candidates'] == len(records)

zero = summary['steps']['0']
final = summary['steps']['4000']
all_delta = (final['all']['plan_equal_mean']['fine_raw_top1_fraction'] -
             zero['all']['plan_equal_mean']['fine_raw_top1_fraction'])
best_delta = (final['best_physical']['plan_equal_mean']['fine_raw_top1_fraction'] -
              zero['best_physical']['plan_equal_mean']['fine_raw_top1_fraction'])
plan_deltas = {plan: (final['all']['by_plan'][plan]['fine_raw_top1_fraction'] -
                       zero['all']['by_plan'][plan]['fine_raw_top1_fraction'])
               for plan in plan_ids}
appearance_deltas = {mode: (
    final[f'appearance/{mode}/all']['plan_equal_mean']['fine_raw_top1_fraction'] -
    zero[f'appearance/{mode}/all']['plan_equal_mean']['fine_raw_top1_fraction'])
    for mode in panel_protocol['modes']}
summary['scale_up_signal'] = {
    'all_candidates_plan_equal_raw_top1_delta': all_delta,
    'best_physical_plan_equal_raw_top1_delta': best_delta,
    'all_candidates_raw_top1_delta_by_plan': plan_deltas,
    'positive_plan_count': sum(value > 0 for value in plan_deltas.values()),
    'all_candidates_raw_top1_delta_by_appearance': appearance_deltas,
    'all_candidates_gain_at_least_0.05': all_delta >= .05,
    'best_physical_gain_at_least_0.10': best_delta >= .10,
    'positive_all_candidate_change_in_at_least_six_plans':
        sum(value > 0 for value in plan_deltas.values()) >= 6,
    'no_appearance_drop_more_than_0.05':
        all(value >= -.05 for value in appearance_deltas.values())}
summary['scale_up_signal']['passed'] = all(summary['scale_up_signal'][key] for key in (
    'all_candidates_gain_at_least_0.05',
    'best_physical_gain_at_least_0.10',
    'positive_all_candidate_change_in_at_least_six_plans',
    'no_appearance_drop_more_than_0.05'))

config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_contextual_match_099.py', 'contextual_matcher_099.py',
        'train_contextual_match_099.py', 'evaluate_spatial_verifier_094.py',
        'spatial_joint_fit_094.py', 'whole_slice_atlas_feedback_083.py',
        'whole_slice_atlas_feedback_081.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_checkpoint_sha256': sha(parent_checkpoint),
    'train_completed_sha256': sha(train_run / 'completed.json'),
    'train_config_sha256': sha(train_run / 'config.json'),
    'train_draws_sha256': sha(train_run / 'draws.jsonl'),
    'descriptor_checkpoint_sha256': {str(step): sha(checkpoints[step])
                                     for step in steps},
    'plans_completed_sha256': sha(plans_dir / 'completed.json'),
    'plans_protocol_sha256': sha(plans_dir / 'protocol.json'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'fixed_geometry': '094 parent step20000 predictor; prior top 8 of branch IDs 0:31 plus top 6 of 32:63; parent 094 teacher fits every pair once; coarse 099 uses initial state and fine uses frozen fitted state at both steps',
    'physical_best': 'minimum parent094 mapped96 point error on 1024 deterministic linearly spaced valid pixels, as in evaluate_spatial_verifier_094.py; first beam slot on tie',
    'candidate_geometry_frozen_in': 'candidates.jsonl: beam ID, initial state, fitted state, parent mapped96 error, support, and source/sample hashes at both descriptor steps',
    'synthetic_truth': 'panel target_centre_um at bilinearly sampled visible cell centres; readout only, never passed to predictor, teacher fit, matcher or mapper',
    'atlas_available': 'any match bin support >= 0.5',
    'denominator': 'visible sites with atlas available support, per candidate; candidate mean within section, section mean within plan, equal mean across nonempty plans',
    'fine_positive_um': 1500, 'fine_negative_um': 2000,
    'coarse_positive_um': 2000, 'coarse_negative_um': 3000,
    'fine_grid': 32, 'coarse_grid': 16,
    'raw_top1': 'argmax normalized descriptor cosine among supported bins; operational atlas-support penalty removed',
    'operational': 'frozen 094 _verify local-softmax coordinate on fine logits and fitted states',
    'centre_null': 'unshifted fitted-state plane centre at each fine site, same atlas-available denominator',
    'hard_margin': 'best supported positive minus hardest supported negative normalized cosine at visible available candidate-sites with both',
    'support_strata': '[0,0.5), [0.5,0.7), [0.7,1] on full fine-grid atlas-available fraction',
    'scale_up_signal': 'plan-equal fine raw top1 change: all >=0.05 and parent-mapped-physical-best >=0.10; positive all-candidate delta in >=6/8 plans; all-candidate appearance delta >=-0.05 in every mode',
    'eligible_sections': len(records), 'ineligible_sections_excluded': panel_receipt['ineligible'],
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False, 'external_pretrained_weights_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows),
                   ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'eligible_sections_per_step': len(records),
    'candidates_per_step': len(records) * 14,
    'protocol_sha256': config['protocol_sha256'],
    'parent_094_checkpoint_sha256': config['parent_094_checkpoint_sha256'],
    'train_completed_sha256': config['train_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'descriptor_checkpoint_sha256': config['descriptor_checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json')}, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'eligible_sections': len(records),
    'scale_up_signal': summary['scale_up_signal']}), flush=True)
