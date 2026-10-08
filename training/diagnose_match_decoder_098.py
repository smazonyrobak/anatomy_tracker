"""Read-only match-decoder diagnosis on the frozen 096/097 synthetic DEV sections."""
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
diagnostic_protocol = source.parent / 'docs/publication/MATCH_DECODER_098_PROTOCOL_20261008.md'
parent = root / 'runs/spatial_verifier_094_pilot'
prior = root / 'runs/spatial_verifier_094_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
ranking = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
diagnostic = root / 'runs/spatial_match_signal_093_diagnostic'
null_run = root / 'runs/geometry_null_096_diagnostic'
train_run = root / 'runs/contrastive_match_097_pilot'
eval_run = root / 'runs/contrastive_match_097_development_eval'
out = root / 'runs/match_decoder_098_diagnostic'
assert not out.exists()
parent_checkpoint = parent / 'joint_step_20000.pt'
steps = (0, 4000)
checkpoints = {step: train_run / f'descriptor_step_{step:05d}.pt' for step in steps}
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(protocol) == '0a39656bce38a3364d7f1a2aab03b7e9a9d1077a271765da3f0bf926bba493fe'
assert sha(diagnostic_protocol) == '0ed9d571b252d59c9c919eff6b1c7307b1801cd2afec2d7af7643de5c1ca773f'
assert sha(null_run / 'completed.json') == '7e834daa513a5d55a415e64ec8bfc2a3de8129134d148429d7c2f41b52d8606c'
assert sha(train_run / 'completed.json') == '2dea4576c67f1e5cbe407d991c807a385a793cedfe067bfd5237f0d42e4156a4'
assert sha(eval_run / 'completed.json') == 'cc4826209f84343f1c150b5e489bdf6d5dd5584935a34ce61f212f1991199420'
null_receipt = json.loads((null_run / 'completed.json').read_text())
train_receipt = json.loads((train_run / 'completed.json').read_text())
eval_receipt = json.loads((eval_run / 'completed.json').read_text())
for run, receipt, names in (
    (null_run, null_receipt, ('config', 'candidates', 'sections', 'summary')),
    (train_run, train_receipt, ('config', 'draws', 'training')),
    (eval_run, eval_receipt, ('config', 'candidates', 'sections', 'summary')),
):
    for name in names:
        suffix = '.json' if name in ('config', 'summary') else '.jsonl'
        assert sha(run / (name + suffix)) == receipt[name + '_sha256']
null_config = json.loads((null_run / 'config.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
eval_config = json.loads((eval_run / 'config.json').read_text())
for config in (null_config, train_config, eval_config):
    assert all(sha(source / name) == digest for name, digest in
               config['source_sha256'].items())
assert sha(parent_checkpoint) == null_config['parent_checkpoint_sha256'] == \
    train_receipt['parent_094_checkpoint_sha256'] == eval_receipt['parent_checkpoint_sha256']
assert sha(parent / 'completed.json') == null_config['parent_completed_sha256'] == \
    eval_config['parent_completed_sha256']
assert sha(prior / 'completed.json') == null_config['prior_eval_completed_sha256']
assert sha(prior / 'candidates.jsonl') == null_config['prior_eval_candidates_sha256']
assert sha(panel / 'completed.json') == null_config['panel_completed_sha256'] == \
    eval_config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == null_config['panel_records_sha256'] == \
    eval_config['panel_records_sha256']
assert sha(ranking / 'candidates.jsonl') == null_config['ranking_candidates_sha256'] == \
    eval_config['fixed_ranking_candidates_sha256']
assert sha(diagnostic / 'config.json') == null_config['diagnostic_config_sha256']
assert sha(null_run / 'completed.json') == eval_config['geometry_null_completed_sha256']
assert sha(null_run / 'candidates.jsonl') == eval_config['geometry_null_candidates_sha256']
assert sha(train_run / 'completed.json') == eval_receipt['training_completed_sha256'] == \
    eval_config['training_completed_sha256']
assert sha(train_run / 'config.json') == eval_config['training_config_sha256']
assert train_config['protocol_sha256'] == eval_config['protocol_sha256'] == sha(protocol)
assert train_config['source_sha256'] == train_receipt['source_sha256']
assert train_receipt['batches'] == train_receipt['accepted_synthetic'] == \
    train_receipt['distinct_physical_sections'] == 4000
assert eval_receipt['sections_per_step'] == 182
assert eval_receipt['candidates_per_step'] == 182 * 14
for step in steps:
    digest = sha(checkpoints[step])
    assert digest == train_receipt['checkpoint_sha256'][str(step)] == \
        eval_receipt['descriptor_checkpoint_sha256'][str(step)] == \
        eval_config['descriptor_checkpoint_sha256'][str(step)]

inspected = set(json.loads((diagnostic / 'config.json').read_text())['section_ids'])
all_records = [row for row in map(json.loads, (panel / 'records.jsonl').open())
               if row['eligible']]
records = [row for row in all_records if row['section_id'] not in inspected]
assert len(all_records) == 246 and len(inspected) == 64 and len(records) == 182
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plans) == 8

frozen = defaultdict(list)
for row in map(json.loads, (prior / 'candidates.jsonl').open()):
    if row['step'] == 20000 and row['in_fixed_beam'] and row['independent_head_readout']:
        frozen[row['section_id']].append(row)
ranked = defaultdict(dict)
for row in map(json.loads, (ranking / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row['branch_id']
null_rows = {(row['section_id'], row['beam_slot']): row for row in
             map(json.loads, (null_run / 'candidates.jsonl').open())}
eval_rows = {(row['step'], row['section_id'], row['beam_slot']): row for row in
             map(json.loads, (eval_run / 'candidates.jsonl').open())}
assert set(frozen) == {row['section_id'] for row in records}
assert len(null_rows) == 182 * 14 and len(eval_rows) == 2 * 182 * 14
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
candidate_rows = []
max_096_difference = 0.
max_097_difference = 0.

with torch.inference_mode():
    for record_index, record in enumerate(records, 1):
        section = record['section_id']
        rows = frozen[section]
        best_slot = int(np.argmin([null_rows[(section, slot)]['mapped96_error_um']
                                   for slot in range(14)]))
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
        visible_count = int(visible.sum())
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
            for start in range(0, 14, 2):
                pair = branch[:, start:start + 2]
                state = torch.tensor([old['fitted_state'] for old in rows[start:start + 2]],
                    device='cuda', dtype=initial.dtype)[None]
                matched = spatial.matcher(prediction['feature'], state, pair % 2,
                    atlas, offsets, weights, source_shape=(256, 256), match_only=True)
                logits = matched['fine_match_logits']
                support = matched['fine_match_support']
                supported = support >= .5
                available = supported.any(2)
                match_ccf, _, _, verify_available = spatial._verify('fine', logits,
                    support, prediction['feature'], state.new_zeros(1, 2, 64, 32, 32),
                    state, pair % 2, (256, 256))
                assert torch.equal(available, verify_available >= .5)
                centre, frame, basis = full_frame_state_to_components(state)
                edges = frame[..., :, :2] @ basis
                chart_x = torch.where((pair % 2).bool()[..., None, None],
                    255 / 256 - xx, xx)
                chart = torch.stack((chart_x, yy.expand_as(chart_x)), -1)
                centre_ccf = centre[..., None, None, :] + torch.einsum(
                    'bkij,bkhwj->bkhwi', edges, chart - .5)
                displacement = torch.stack((torch.where(
                    (pair % 2).bool()[..., None], -bin_dx, bin_dx),
                    bin_dy.expand(1, 2, -1)), -1).to(state.dtype) / 32
                bin_shift = torch.einsum('bkij,bkcj->bkci', edges, displacement)
                bin_shift = bin_shift + 1500 * bin_depth[None, None, :, None] * (
                    frame[..., :, 2][:, :, None, :])
                bin_ccf = centre_ccf[:, :, None] + bin_shift[:, :, :, None, None]
                distance = (bin_ccf - truth32[:, None, None]).norm(dim=-1)
                positive = supported & (distance <= 1500)
                negative = supported & (distance >= 2000)
                raw_cosine = (logits + 4 * (1 - support)) / temperature
                raw_peak = raw_cosine.masked_fill(~supported, -1e4).argmax(2)
                penalized_peak = logits.masked_fill(~supported, -1e4).argmax(2)
                raw_correct = distance.gather(2, raw_peak[:, :, None]).squeeze(2) <= 1500
                penalized_correct = distance.gather(
                    2, penalized_peak[:, :, None]).squeeze(2) <= 1500
                verified_correct = (match_ccf - truth32[:, None]).norm(dim=-1) <= 1500
                centre_correct = (centre_ccf - truth32[:, None]).norm(dim=-1) <= 1500
                oracle = positive.any(2)
                contrastive = oracle & negative.any(2)
                positive_max = raw_cosine.masked_fill(~positive, -torch.inf).amax(2)
                negative_max = raw_cosine.masked_fill(~negative, -torch.inf).amax(2)

                for local, old in enumerate(rows[start:start + 2]):
                    slot = start + local
                    old_null = null_rows[(section, slot)]
                    old_eval = eval_rows[(step, section, slot)]
                    mask = visible[0] & available[0, local]
                    eligible = mask & contrastive[0, local]
                    count = int(mask.sum())
                    eligible_count = int(eligible.sum())
                    verified_count = int((verified_correct[0, local] & mask).sum())
                    centre_count = int((centre_correct[0, local] & mask).sum())
                    assert visible_count == old_null['visible_sites'] == old_eval['visible_sites']
                    assert count == old_null['atlas_available_visible_sites'] == \
                        old_eval['atlas_available_visible_sites']
                    assert old['branch_id'] == old_eval['branch_id']
                    assert old_null['mapped96_error_um'] == old_eval['mapped96_error_um']
                    assert abs(float(available[0, local].float().mean()) -
                               old['atlas_support_fraction']) < 1e-5
                    difference = max(abs(verified_count / visible_count - old_eval['matched_fraction']),
                        abs(centre_count / visible_count - old_eval['centre_fraction']))
                    max_097_difference = max(max_097_difference, difference)
                    assert difference < 1e-5
                    if step == 0:
                        difference = max(abs(verified_count / visible_count - old_null['actual_fraction']),
                            abs(centre_count / visible_count - old_null['null_fraction']))
                        max_096_difference = max(max_096_difference, difference)
                        assert difference < 1e-4
                    support_fraction = old['atlas_support_fraction']
                    stratum = ('low_<0.5' if support_fraction < .5 else
                        'middle_0.5_to_0.7' if support_fraction < .7 else 'high_>=0.7')
                    candidate_rows.append({'step': step, 'section_id': section,
                        'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                        'appearance_mode': record['appearance_mode'],
                        'source_sha256': record['sha256'], 'beam_slot': slot,
                        'branch_id': old['branch_id'],
                        'mapped96_error_um': old_null['mapped96_error_um'],
                        'best_physical_candidate': slot == best_slot,
                        'atlas_support_fraction': support_fraction,
                        'atlas_support_stratum': stratum,
                        'visible_sites': visible_count,
                        'atlas_available_visible_sites': count,
                        'supported_positive_sites': int((oracle[0, local] & mask).sum()),
                        'contrastive_eligible_sites': eligible_count,
                        'atlas_available_fraction': count / visible_count,
                        'oracle_ceiling_fraction': float((oracle[0, local] & mask).sum() / count)
                            if count else None,
                        'raw_cosine_argmax_fraction': float((raw_correct[0, local] & mask).sum() / count)
                            if count else None,
                        'penalized_argmax_fraction': float((penalized_correct[0, local] & mask).sum() / count)
                            if count else None,
                        'verified_fraction': verified_count / count if count else None,
                        'centre_fraction': centre_count / count if count else None,
                        'raw_top1_positive_given_contrastive_eligible':
                            float((raw_correct[0, local] & eligible).sum() / eligible_count)
                            if eligible_count else None,
                        'penalized_top1_positive_given_contrastive_eligible':
                            float((penalized_correct[0, local] & eligible).sum() / eligible_count)
                            if eligible_count else None,
                        'best_positive_minus_hard_negative_cosine':
                            float((positive_max[0, local] - negative_max[0, local])[
                                eligible].mean()) if eligible_count else None})
        if record_index % 64 == 0:
            print(json.dumps({'sections_done': record_index}), flush=True)

assert len(candidate_rows) == 2 * 182 * 14
metrics = ('atlas_available_fraction', 'oracle_ceiling_fraction',
    'raw_cosine_argmax_fraction', 'penalized_argmax_fraction', 'verified_fraction',
    'centre_fraction', 'raw_top1_positive_given_contrastive_eligible',
    'penalized_top1_positive_given_contrastive_eligible',
    'best_positive_minus_hard_negative_cosine')
summary = {'sections_per_step': 182, 'candidates_per_step': 182 * 14,
    'max_step0_096_visible_denominator_difference': max_096_difference,
    'max_097_visible_denominator_difference': max_097_difference, 'steps': {}}
for step in steps:
    selected = [row for row in candidate_rows if row['step'] == step]
    groups = {'all': selected,
        'best_physical': [row for row in selected if row['best_physical_candidate']]}
    for appearance in ('raw', 'exact_black', 'imperfect_brush'):
        for key in ('all', 'best_physical'):
            groups[f'appearance/{appearance}/{key}'] = [row for row in groups[key]
                if row['appearance_mode'] == appearance]
    for stratum in ('low_<0.5', 'middle_0.5_to_0.7', 'high_>=0.7'):
        for key in ('all', 'best_physical'):
            groups[f'support/{stratum}/{key}'] = [row for row in groups[key]
                if row['atlas_support_stratum'] == stratum]
    summary['steps'][str(step)] = {}
    for name, group in groups.items():
        by_plan = {}
        for plan in plans:
            sub = [row for row in group if row['synthetic_subject_plan_id'] == plan]
            sections = sorted({row['section_id'] for row in sub})
            section_means = {section: {key:
                float(np.mean([row[key] for row in sub if row['section_id'] == section
                               and row[key] is not None]))
                if any(row['section_id'] == section and row[key] is not None
                       for row in sub) else None for key in metrics}
                for section in sections}
            by_plan[plan] = {'sections': len(sections), 'candidates': len(sub),
                **{key: float(np.mean([row[key] for row in section_means.values()
                    if row[key] is not None])) if any(row[key] is not None
                    for row in section_means.values()) else None for key in metrics}}
        summary['steps'][str(step)][name] = {'candidates': len(group),
            'by_plan': by_plan,
            'plan_equal_mean': {key: float(np.mean([row[key] for row in by_plan.values()
                if row[key] is not None])) if any(row[key] is not None
                for row in by_plan.values()) else None for key in metrics}}
    assert summary['steps'][str(step)]['best_physical']['candidates'] == 182

config = {'diagnostic_protocol_sha256': sha(diagnostic_protocol),
    'source_sha256': {name: sha(source / name) for name in
    ('diagnose_match_decoder_098.py', 'evaluate_contrastive_match_097.py',
     'diagnose_geometry_null_096.py', 'spatial_joint_fit_094.py',
     'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
     'arbitrary_plane_full_frame_primitives.py',
     'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_allen_atlas_binding_v6.py')},
    'protocol_sha256': sha(protocol),
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'descriptor_checkpoint_sha256': {str(step): sha(checkpoints[step]) for step in steps},
    'training_completed_sha256': sha(train_run / 'completed.json'),
    'evaluation_completed_sha256': sha(eval_run / 'completed.json'),
    'geometry_null_completed_sha256': sha(null_run / 'completed.json'),
    'prior_candidates_sha256': sha(prior / 'candidates.jsonl'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_candidates_sha256': sha(ranking / 'candidates.jsonl'),
    'visible_sites': 'bilinear valid_mask at 32x32 equals 1',
    'denominator': 'visible sites with any fine bin support >= 0.5, per candidate',
    'correct_tolerance_um': 1500,
    'oracle': 'any supported bin <= 1500 um from synthetic true tissue CCF',
    'raw_cosine_argmax': 'argmax of normalized descriptor cosine among supported bins',
    'penalized_argmax': 'argmax of operational fine logits among supported bins',
    'verified': 'operational _verify argmax plus local softmax coordinate',
    'centre': '094 fitted-state plane coordinate, zero displacement',
    'contrastive_eligible': 'visible and atlas-available, with supported positive <= 1500 um and hard negative >= 2000 um',
    'best_physical': 'minimum frozen mapped96_error_um, first beam slot on tie',
    'plan_equal_mean': 'candidate mean within section, section mean within plan, equal mean across nonempty plans'}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
with (out / 'candidates.jsonl').open('w') as stream:
    for row in candidate_rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_step': 182, 'candidates_per_step': 182 * 14,
    'diagnostic_protocol_sha256': config['diagnostic_protocol_sha256'],
    'source_sha256': config['source_sha256']['diagnose_match_decoder_098.py'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'descriptor_checkpoint_sha256': config['descriptor_checkpoint_sha256'],
    'training_completed_sha256': config['training_completed_sha256'],
    'evaluation_completed_sha256': config['evaluation_completed_sha256'],
    'geometry_null_completed_sha256': config['geometry_null_completed_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'summary_sha256': sha(out / 'summary.json')}, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete',
    'step4000': {key: summary['steps']['4000'][key]['plan_equal_mean']
                 for key in ('all', 'best_physical')}}, allow_nan=False))
