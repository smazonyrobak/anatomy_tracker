"""Independent 097 descriptor readout on the frozen complementary 096 DEV panel."""
import hashlib
import json
import os
import sys
from collections import defaultdict
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
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/CONTRASTIVE_MATCH_097_PROTOCOL_20261005.md'
parent = root / 'runs/spatial_verifier_094_pilot'
prior = root / 'runs/spatial_verifier_094_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
ranking = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
diagnostic = root / 'runs/spatial_match_signal_093_diagnostic'
null_run = root / 'runs/geometry_null_096_diagnostic'
training = root / 'runs/contrastive_match_097_pilot'
out = root / 'runs/contrastive_match_097_development_eval'
parent_checkpoint = parent / 'joint_step_20000.pt'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


assert sha(protocol) == '0a39656bce38a3364d7f1a2aab03b7e9a9d1077a271765da3f0bf926bba493fe'
assert sha(null_run / 'completed.json') == \
    '7e834daa513a5d55a415e64ec8bfc2a3de8129134d148429d7c2f41b52d8606c'
null_receipt = json.loads((null_run / 'completed.json').read_text())
for name, key in (('config.json', 'config_sha256'),
                  ('candidates.jsonl', 'candidates_sha256'),
                  ('sections.jsonl', 'sections_sha256'),
                  ('summary.json', 'summary_sha256')):
    assert sha(null_run / name) == null_receipt[key]
null_config = json.loads((null_run / 'config.json').read_text())
for name, expected in null_config['source_sha256'].items():
    assert sha(source / name) == expected
assert sha(parent_checkpoint) == null_config['parent_checkpoint_sha256']
assert sha(parent / 'completed.json') == null_config['parent_completed_sha256']
assert sha(prior / 'completed.json') == null_config['prior_eval_completed_sha256']
assert sha(prior / 'candidates.jsonl') == null_config['prior_eval_candidates_sha256']
assert sha(panel / 'completed.json') == null_config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == null_config['panel_records_sha256']
assert sha(ranking / 'candidates.jsonl') == null_config['ranking_candidates_sha256']
assert sha(diagnostic / 'config.json') == null_config['diagnostic_config_sha256']

training_receipt = json.loads((training / 'completed.json').read_text())
assert sha(training / 'config.json') == training_receipt['config_sha256']
assert sha(training / 'draws.jsonl') == training_receipt['draws_sha256']
assert sha(training / 'training.jsonl') == training_receipt['training_sha256']
training_config = json.loads((training / 'config.json').read_text())
assert training_config['protocol_sha256'] == sha(protocol)
assert training_config['source_sha256'] == training_receipt['source_sha256']
assert all(sha(source / name) == digest for name, digest in
           training_config['source_sha256'].items())
assert training_receipt['batches'] == training_receipt['accepted_synthetic'] == \
    training_receipt['distinct_physical_sections'] == 4000
assert training_receipt['parent_094_checkpoint_sha256'] == sha(parent_checkpoint)
steps = (0, 4000)
checkpoints = {step: training / f'descriptor_step_{step:05d}.pt' for step in steps}
for step in steps:
    assert sha(checkpoints[step]) == training_receipt['checkpoint_sha256'][str(step)]

inspected = set(json.loads((diagnostic / 'config.json').read_text())['section_ids'])
all_records = [row for row in map(json.loads, (panel / 'records.jsonl').open())
               if row['eligible']]
records = [row for row in all_records if row['section_id'] not in inspected]
assert len(all_records) == 246 and len(inspected) == 64 and len(records) == 182
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8

frozen = defaultdict(list)
for row in map(json.loads, (prior / 'candidates.jsonl').open()):
    if row['step'] == 20000 and row['in_fixed_beam'] and row['independent_head_readout']:
        frozen[row['section_id']].append(row)
ranked = defaultdict(dict)
for row in map(json.loads, (ranking / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row['branch_id']
null_rows = {(row['section_id'], row['beam_slot']): row for row in
             map(json.loads, (null_run / 'candidates.jsonl').open())}
assert set(frozen) == {row['section_id'] for row in records}
assert len(null_rows) == 182 * 14
for record in records:
    section = record['section_id']
    frozen[section].sort(key=lambda row: row['beam_slot'])
    assert len(frozen[section]) == 14
    assert [row['beam_slot'] for row in frozen[section]] == list(range(14))
    assert [row['branch_id'] for row in frozen[section]] == [
        ranked[section][slot] for slot in range(14)]
    assert all(row['source_sha256'] == record['sha256'] and row['step'] == 20000
               for row in frozen[section])
    assert all(null_rows[(section, slot)]['branch_id'] == old['branch_id'] and
               null_rows[(section, slot)]['source_sha256'] == record['sha256']
               for slot, old in enumerate(frozen[section]))

saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and not saved['calibrated']
parent_matcher = {key.removeprefix('matcher.'): value for key, value in
                  saved['spatial_fit'].items() if key.startswith('matcher.')}
matcher_states = {}
for step in steps:
    checkpoint = torch.load(checkpoints[step], map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['calibrated'] is False
    assert isinstance(checkpoint['config'], dict)
    assert set(checkpoint['matcher']) == set(parent_matcher)
    for key, value in checkpoint['matcher'].items():
        if step == 0 or not key.startswith(('image.', 'atlas.')):
            assert torch.equal(value, parent_matcher[key]), (step, key)
    matcher_states[step] = checkpoint['matcher']
assert any(not torch.equal(matcher_states[4000][key], parent_matcher[key])
           for key in parent_matcher if key.startswith('image.'))
assert any(not torch.equal(matcher_states[4000][key], parent_matcher[key])
           for key in parent_matcher if key.startswith('atlas.'))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model'], strict=True)
spatial = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
spatial.load_state_dict(saved['spatial_fit'], strict=True)
del saved

axis = (torch.arange(32, device='cuda') + .5) / 32 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
bin_index = torch.arange(225, device='cuda')
bin_depth = bin_index // 25 - 4
bin_dy = bin_index // 5 % 5 - 2
bin_dx = bin_index % 5 - 2
candidate_rows, section_rows = [], []
max_step0_difference = 0.

with torch.inference_mode():
    for record in records:
        section = record['section_id']
        rows = frozen[section]
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        visible = F.interpolate(valid[:, None].float(), (32, 32),
            mode='bilinear', align_corners=False)[:, 0] == 1
        truth32 = F.interpolate(truth.permute(0, 3, 1, 2), (32, 32),
            mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        denominator = visible.sum((-2, -1))[:, None].clamp_min(1)
        prediction = model.predict(image)
        branch = torch.tensor([[row['branch_id'] for row in rows]], device='cuda')
        initial = prediction['state'].gather(1,
            (branch // 2)[..., None].expand(-1, -1, 12))
        for slot, old in enumerate(rows):
            assert max(abs(a - b) for a, b in zip(
                initial[0, slot].tolist(), old['initial_state'])) < 1e-3

        for step in steps:
            spatial.matcher.load_state_dict(matcher_states[step], strict=True)
            temperature = spatial.matcher.log_temperature.exp().clamp(2, 20)
            section_candidates = []
            best_positive = torch.full((32, 32), -torch.inf, device='cuda')
            hard_negative = torch.full((32, 32), -torch.inf, device='cuda')
            positive_sum = torch.zeros((32, 32), device='cuda')
            negative_sum = torch.zeros((32, 32), device='cuda')
            positive_count = torch.zeros((32, 32), device='cuda')
            negative_count = torch.zeros((32, 32), device='cuda')

            for start in range(0, 14, 2):
                pair = branch[:, start:start + 2]
                state = torch.tensor([old['fitted_state'] for old in rows[start:start + 2]],
                    device='cuda', dtype=initial.dtype)[None]
                matched = spatial.matcher(prediction['feature'], state, pair % 2,
                    atlas, offsets, weights, source_shape=(256, 256), match_only=True)
                match_ccf, _, _, available = spatial._verify('fine',
                    matched['fine_match_logits'], matched['fine_match_support'],
                    prediction['feature'], state.new_zeros(1, 2, 64, 32, 32),
                    state, pair % 2, (256, 256))
                available = available >= .5
                centre, frame, basis = full_frame_state_to_components(state)
                edges = frame[..., :, :2] @ basis
                chart_x = torch.where((pair % 2).bool()[..., None, None],
                    255 / 256 - xx, xx)
                chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
                centre_ccf = centre[..., None, None, :] + torch.einsum(
                    'bkij,bkhwj->bkhwi', edges, chart - .5)
                match_correct = visible[:, None] & available & (
                    (match_ccf - truth32[:, None]).norm(dim=-1) <= 1500)
                centre_correct = visible[:, None] & available & (
                    (centre_ccf - truth32[:, None]).norm(dim=-1) <= 1500)

                displacement = torch.stack((torch.where(
                    (pair % 2).bool()[..., None], -bin_dx, bin_dx),
                    bin_dy.expand(1, 2, -1)), -1).to(state.dtype) / 32
                bin_shift = torch.einsum('bkij,bkcj->bkci', edges, displacement)
                bin_shift = bin_shift + 1500 * bin_depth[None, None, :, None] * (
                    frame[..., :, 2][:, :, None, :])
                bin_ccf = centre_ccf[:, :, None] + bin_shift[:, :, :, None, None]
                distance = (bin_ccf - truth32[:, None, None]).norm(dim=-1)
                support = matched['fine_match_support']
                supported = support >= .5
                positive = supported & (distance <= 1500) & visible[:, None, None]
                negative = supported & (distance >= 2000) & visible[:, None, None]
                similarity = (matched['fine_match_logits'] + 4 * (1 - support)) / temperature
                positive_max = similarity.masked_fill(~positive, -torch.inf).amax(2)
                negative_max = similarity.masked_fill(~negative, -torch.inf).amax(2)
                best_positive = torch.maximum(best_positive, positive_max.amax(1)[0])
                hard_negative = torch.maximum(hard_negative, negative_max.amax(1)[0])
                positive_sum += similarity.masked_fill(~positive, 0).sum((1, 2))[0]
                negative_sum += similarity.masked_fill(~negative, 0).sum((1, 2))[0]
                positive_count += positive.sum((1, 2))[0]
                negative_count += negative.sum((1, 2))[0]

                for local, old in enumerate(rows[start:start + 2]):
                    slot = start + local
                    old_null = null_rows[(section, slot)]
                    match_fraction = float(match_correct[0, local].sum() / denominator[0, 0])
                    centre_fraction = float(centre_correct[0, local].sum() / denominator[0, 0])
                    available_count = int((available[0, local] & visible[0]).sum())
                    assert available_count == old_null['atlas_available_visible_sites']
                    assert int(denominator[0, 0]) == old_null['visible_sites']
                    assert abs(centre_fraction - old_null['null_fraction']) < 1e-5
                    if step == 0:
                        error = abs(match_fraction - old_null['actual_fraction'])
                        max_step0_difference = max(max_step0_difference, error)
                        assert error < 1e-4
                    recovered = int((match_correct[0, local] &
                        ~centre_correct[0, local]).sum())
                    corrupted = int((centre_correct[0, local] &
                        ~match_correct[0, local]).sum())
                    centre_wrong = available_count - int(centre_correct[0, local].sum())
                    centre_right = int(centre_correct[0, local].sum())
                    both = positive[0, local].any(0) & negative[0, local].any(0)
                    mean_positive = (similarity[0, local].masked_fill(
                        ~positive[0, local], 0).sum(0) /
                        positive[0, local].sum(0).clamp_min(1))
                    mean_negative = (similarity[0, local].masked_fill(
                        ~negative[0, local], 0).sum(0) /
                        negative[0, local].sum(0).clamp_min(1))
                    support_fraction = old['atlas_support_fraction']
                    support_stratum = ('low_<0.5' if support_fraction < .5 else
                        'middle_0.5_to_0.7' if support_fraction < .7 else 'high_>=0.7')
                    candidate = {'step': step, 'section_id': section,
                        'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                        'appearance_mode': record['appearance_mode'],
                        'source_sha256': record['sha256'], 'beam_slot': slot,
                        'branch_id': old['branch_id'],
                        'mapped96_error_um': old_null['mapped96_error_um'],
                        'visible_sites': int(denominator[0, 0]),
                        'atlas_available_visible_sites': available_count,
                        'atlas_support_fraction': support_fraction,
                        'atlas_support_stratum': support_stratum,
                        'matched_fraction': match_fraction,
                        'centre_fraction': centre_fraction,
                        'matched_minus_centre': match_fraction - centre_fraction,
                        'centre_wrong_recoveries': recovered,
                        'centre_correct_corruptions': corrupted,
                        'recovery_fraction': recovered / int(denominator[0, 0]),
                        'corruption_fraction': corrupted / int(denominator[0, 0]),
                        'recovery_given_centre_wrong': recovered / centre_wrong
                            if centre_wrong else None,
                        'corruption_given_centre_correct': corrupted / centre_right
                            if centre_right else None,
                        'positive_negative_sites': int(both.sum()),
                        'best_positive_minus_hard_negative_similarity':
                            float((positive_max[0, local] - negative_max[0, local])[
                                both].mean()) if bool(both.any()) else None,
                        'mean_positive_minus_mean_negative_similarity':
                            float((mean_positive - mean_negative)[both].mean())
                            if bool(both.any()) else None}
                    candidate_rows.append(candidate)
                    section_candidates.append(candidate)

            both = visible[0] & (positive_count > 0) & (negative_count > 0)
            errors = np.array([row['mapped96_error_um'] for row in section_candidates])
            best = section_candidates[int(errors.argmin())]
            section_rows.append({'step': step, 'section_id': section,
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': record['appearance_mode'],
                'mean_matched_minus_centre': mean(section_candidates, 'matched_minus_centre'),
                'mean_matched_fraction': mean(section_candidates, 'matched_fraction'),
                'mean_centre_fraction': mean(section_candidates, 'centre_fraction'),
                'best_matched_fraction': best['matched_fraction'],
                'best_centre_fraction': best['centre_fraction'],
                'mean_recovery_fraction': mean(section_candidates, 'recovery_fraction'),
                'mean_corruption_fraction': mean(section_candidates, 'corruption_fraction'),
                'centre_wrong_recoveries': sum(row['centre_wrong_recoveries']
                                               for row in section_candidates),
                'centre_correct_corruptions': sum(row['centre_correct_corruptions']
                                                  for row in section_candidates),
                'all_bank_positive_negative_sites': int(both.sum()),
                'all_bank_best_positive_minus_hard_negative_similarity':
                    float((best_positive - hard_negative)[both].mean())
                    if bool(both.any()) else None,
                'all_bank_mean_positive_minus_mean_negative_similarity':
                    float((positive_sum / positive_count.clamp_min(1) -
                           negative_sum / negative_count.clamp_min(1))[both].mean())
                    if bool(both.any()) else None})

section_metrics = ('mean_matched_minus_centre', 'mean_matched_fraction',
    'mean_centre_fraction', 'best_matched_fraction', 'best_centre_fraction',
    'mean_recovery_fraction', 'mean_corruption_fraction',
    'all_bank_best_positive_minus_hard_negative_similarity',
    'all_bank_mean_positive_minus_mean_negative_similarity')
candidate_metrics = ('matched_minus_centre', 'matched_fraction', 'centre_fraction',
    'recovery_fraction', 'corruption_fraction', 'recovery_given_centre_wrong',
    'corruption_given_centre_correct',
    'best_positive_minus_hard_negative_similarity',
    'mean_positive_minus_mean_negative_similarity')
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
summary = {'sections': 182, 'candidates_per_step': 182 * 14,
    'max_step0_096_matched_fraction_difference': max_step0_difference, 'steps': {}}
for step in steps:
    selected = [row for row in section_rows if row['step'] == step]
    candidates = [row for row in candidate_rows if row['step'] == step]
    by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan
        for row in selected), **{key: mean([row for row in selected
        if row['synthetic_subject_plan_id'] == plan], key) for key in section_metrics}}
        for plan in plans}
    plan_equal = {key: mean(list(by_plan.values()), key) for key in section_metrics}
    appearances = {}
    for appearance in ('raw', 'exact_black', 'imperfect_brush'):
        sub = [row for row in selected if row['appearance_mode'] == appearance]
        per_plan = {plan: {key: mean([row for row in sub
            if row['synthetic_subject_plan_id'] == plan], key) for key in section_metrics}
            for plan in plans}
        appearances[appearance] = {'sections': len(sub),
            'plan_equal_mean': {key: mean(list(per_plan.values()), key)
                                for key in section_metrics}}
    support_strata = {}
    for stratum in ('low_<0.5', 'middle_0.5_to_0.7', 'high_>=0.7'):
        sub = [row for row in candidates if row['atlas_support_stratum'] == stratum]
        per_plan = {plan: {key: mean([row for row in sub
            if row['synthetic_subject_plan_id'] == plan], key) for key in candidate_metrics}
            for plan in plans}
        support_strata[stratum] = {'candidates': len(sub),
            'plan_equal_mean': {key: mean(list(per_plan.values()), key)
                                for key in candidate_metrics}}
    summary['steps'][str(step)] = {'by_plan': by_plan, 'plan_equal_mean': plan_equal,
        'appearance_strata': appearances, 'atlas_support_strata': support_strata}

old_summary = json.loads((null_run / 'summary.json').read_text())
assert abs(summary['steps']['0']['plan_equal_mean']['best_matched_fraction'] -
           old_summary['plan_equal_mean']['best_actual_fraction']) < 1e-5
assert abs(summary['steps']['0']['plan_equal_mean']['best_centre_fraction'] -
           old_summary['plan_equal_mean']['best_null_fraction']) < 1e-5
assert abs(summary['steps']['4000']['plan_equal_mean']['best_centre_fraction'] -
           old_summary['plan_equal_mean']['best_null_fraction']) < 1e-5
assert len(candidate_rows) == 2 * 182 * 14 and len(section_rows) == 2 * 182
final = summary['steps']['4000']
summary['focused_gate'] = {
    'overall_positive': final['plan_equal_mean']['mean_matched_minus_centre'] > 0,
    'positive_plans': sum(row['mean_matched_minus_centre'] > 0
                          for row in final['by_plan'].values()),
    'best_matched_at_least_centre': final['plan_equal_mean']['best_matched_fraction'] >=
        old_summary['plan_equal_mean']['best_null_fraction']}
summary['focused_gate']['passed'] = (summary['focused_gate']['overall_positive'] and
    summary['focused_gate']['positive_plans'] >= 6 and
    summary['focused_gate']['best_matched_at_least_centre'])

config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in
        ('evaluate_contrastive_match_097.py', 'diagnose_geometry_null_096.py',
         'spatial_joint_fit_094.py', 'whole_slice_atlas_feedback_083.py',
         'whole_slice_atlas_feedback_081.py',
         'arbitrary_plane_full_frame_primitives.py',
         'arbitrary_plane_one_shot_model.py',
         'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'parent_completed_sha256': sha(parent / 'completed.json'),
    'training_completed_sha256': sha(training / 'completed.json'),
    'training_config_sha256': sha(training / 'config.json'),
    'descriptor_checkpoint_sha256': {str(step): sha(checkpoints[step]) for step in steps},
    'geometry_null_completed_sha256': sha(null_run / 'completed.json'),
    'geometry_null_candidates_sha256': sha(null_run / 'candidates.jsonl'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'fixed_ranking_candidates_sha256': sha(ranking / 'candidates.jsonl'),
    'section_count': 182, 'fixed_branches_per_section': 14,
    'visible_denominator': 'bilinear valid_mask at 32x32 equals 1',
    'atlas_available': 'any fine match support >= 0.5',
    'correct_tolerance_um': 1500,
    'centre_null': '094 fitted-state plane coordinate with zero dx/dy/depth displacement',
    'similarity': '(fine logits + 4*(1-support))/fixed temperature; normalized descriptor cosine',
    'positive': 'supported bin at most 1500 um from exact observed-site CCF',
    'negative': 'supported bin at least 2000 um from exact observed-site CCF',
    'atlas_support_strata': '[0,.5), [.5,.7), [.7,1] on 094 full-grid support fraction'}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows), ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_step': 182, 'candidates_per_step': 182 * 14,
    'protocol_sha256': sha(protocol),
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'training_completed_sha256': sha(training / 'completed.json'),
    'descriptor_checkpoint_sha256': config['descriptor_checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json')}, indent=2))
print(json.dumps({'event': 'complete', 'focused_gate': summary['focused_gate'],
    'plan_equal_mean': final['plan_equal_mean']}, allow_nan=False))
