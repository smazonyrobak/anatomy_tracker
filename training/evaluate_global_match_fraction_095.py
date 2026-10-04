"""Independent fixed-branch DEV readout for the frozen-parent 095 quality head."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_match_fraction_095 import GlobalMatchFraction095
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/GLOBAL_MATCH_FRACTION_095_PROTOCOL_20261004.md'
train_run = root / 'runs/global_match_fraction_095_pilot'
parent = root / 'runs/spatial_verifier_094_pilot'
parent_eval = root / 'runs/spatial_verifier_094_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
ranking = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
diagnostic = root / 'runs/spatial_match_signal_093_diagnostic'
baseline = root / 'runs/one_shot_joint_atlas_feedback_089_development_eval'
out = root / 'runs/global_match_fraction_095_development_eval'
steps = (0, 4000, 12000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rho(x, y):
    value = spearmanr(x, y).statistic
    return float(value) if math.isfinite(value) else 0.


assert sha(protocol) == '701f768a156f95618ca5543e8865e0e26f8628c93c258d50e2bbc9da2ffe0a6c'
config = json.loads((train_run / 'config.json').read_text())
finished = json.loads((train_run / 'completed.json').read_text())
assert config['batches'] == finished['batches'] == finished['accepted_synthetic'] == 12000
assert tuple(config['checkpoints']) == steps
assert config['protocol_sha256'] == finished['protocol_sha256'] == sha(protocol)
assert not any(config[key] or finished[key] for key in ('calibrated',
    'public_benchmark_used', 'real_labels_used', 'external_pretrained_weights_used'))
assert all(sha(train_run / file) == finished[key] for file, key in (
    ('config.json', 'config_sha256'), ('draws.jsonl', 'draws_sha256'),
    ('training.jsonl', 'training_sha256')))
assert all(sha(source / file) == digest for file, digest in config['source_sha256'].items())
assert all(sha(train_run / f'fraction_step_{step:05d}.pt') ==
           finished['checkpoint_sha256'][str(step)] for step in steps)
parent_receipt = json.loads((parent / 'completed.json').read_text())
assert sha(parent / 'joint_step_20000.pt') == parent_receipt['checkpoint_sha256']['20000'] \
    == config['parent_094_sha256'] \
    == '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent / 'completed.json') == config['parent_094_completed_sha256']
assert sha(parent / 'config.json') == config['parent_094_config_sha256']
parent_eval_receipt = json.loads((parent_eval / 'completed.json').read_text())
assert sha(parent_eval / 'completed.json') == '24c27b5f93e9fbc08e9321ea77d6d246b4372381363c5d1be7057a8d4e793b28'
assert sha(parent_eval / 'candidates.jsonl') == parent_eval_receipt['candidates_sha256']
assert sha(parent_eval / 'sections.jsonl') == parent_eval_receipt['sections_sha256']
assert sha(parent_eval / 'summary.json') == parent_eval_receipt['summary_sha256']
prior_summary = json.loads((parent_eval / 'summary.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256']
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8
rank_receipt = json.loads((ranking / 'completed.json').read_text())
assert sha(ranking / 'candidates.jsonl') == rank_receipt['candidates_sha256']
ranked = {row['section_id']: [None] * 14 for row in records}
for row in map(json.loads, (ranking / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row['branch_id']
assert all(len(set(beam)) == 14 for beam in ranked.values())
diagnostic_receipt = json.loads((diagnostic / 'completed.json').read_text())
assert sha(diagnostic / 'config.json') == diagnostic_receipt['config_sha256']
assert sha(diagnostic / 'candidates.jsonl') == diagnostic_receipt['candidates_sha256']
inspected = set(json.loads((diagnostic / 'config.json').read_text())['section_ids'])
independent = {row['section_id'] for row in records} - inspected
assert len(inspected) == 64 and len(independent) == 182
fixed_parent = {}
for row in map(json.loads, (parent_eval / 'candidates.jsonl').open()):
    if row['step'] == 20000:
        fixed_parent[(row['section_id'], row['branch_id'])] = row
baseline_receipt = json.loads((baseline / 'completed.json').read_text())
assert sha(baseline / 'summary.json') == baseline_receipt['summary_sha256']
baseline_summary = json.loads((baseline / 'summary.json').read_text())['6000']

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'training_completed_sha256': sha(train_run / 'completed.json'),
    'training_config_sha256': sha(train_run / 'config.json'),
    'checkpoint_sha256': finished['checkpoint_sha256'],
    'parent_checkpoint_sha256': sha(parent / 'joint_step_20000.pt'),
    'parent_completed_sha256': sha(parent / 'completed.json'),
    'parent_eval_completed_sha256': sha(parent_eval / 'completed.json'),
    'parent_eval_candidates_sha256': sha(parent_eval / 'candidates.jsonl'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_candidates_sha256': sha(ranking / 'candidates.jsonl'),
    'independent_section_ids': sorted(independent), 'steps': steps,
    'fixed_beam': 'exact 086 14 branches and 094 step20000 mapping',
    'score': '094 direct log prior + log(sigmoid(095 quality_logit))',
    'fraction_target': 'visible surviving fine sites with supported soft local match within 1500um / visible surviving fine sites',
    'aggregation': 'section mean within plan, then equal mean of eight plans',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
parent_state = torch.load(parent / 'joint_step_20000.pt', map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(parent_state['model'], strict=True)
spatial = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
spatial.load_state_dict(parent_state['spatial_fit'], strict=True)
del parent_state
heads = {}
for step in steps:
    saved = torch.load(train_run / f'fraction_step_{step:05d}.pt',
                       map_location='cpu', weights_only=True)
    assert saved['step'] == step and saved['config'] == config and not saved['calibrated']
    head = GlobalMatchFraction095().cuda().eval().requires_grad_(False)
    head.load_state_dict(saved['fraction_head'], strict=True)
    heads[step] = head

candidate_rows, section_rows = [], []
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidates_file, \
        (out / 'sections.jsonl').open('w') as sections_file:
    for record in records:
        section = record['section_id']
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        natural = torch.cat((prior[:, :32].topk(8, -1).indices,
            prior[:, 32:].topk(6, -1).indices + 32), -1)[0].tolist()
        fixed = ranked[section]
        union = list(dict.fromkeys(fixed + natural))
        beam = torch.tensor([union], device='cuda', dtype=torch.long)
        initial = prediction['state'].gather(1,
            (beam // 2)[..., None].expand(-1, -1, 12))
        visible = F.interpolate(valid[:, None].float(), (32, 32),
            mode='bilinear', align_corners=False)[:, 0] == 1
        truth32 = F.interpolate(truth.permute(0, 3, 1, 2), (32, 32),
            mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        predicted = {step: [] for step in steps}
        actual = []
        for start in range(0, len(union), 2):
            branch = beam[:, start:start + 2]
            fitted = spatial(prediction['feature'], initial[:, start:start + 2],
                branch % 2, atlas, offsets, weights, source_shape=(256, 256))
            available = (fitted['fine_match_support'] >= .5).any(2)
            correct = visible[:, None] & available & (
                (fitted['fine_match_ccf_um'] - truth32[:, None]).norm(dim=-1) <= 1500)
            actual.extend((correct.sum((-2, -1)) / visible.sum((-2, -1))[
                :, None].clamp_min(1)).flatten().cpu().tolist())
            for step, head in heads.items():
                quality_logit = head(fitted)['quality_logit']
                predicted[step].extend(quality_logit.sigmoid().flatten().cpu().tolist())
            for local_slot, candidate in enumerate(union[start:start + 2]):
                previous = fixed_parent[(section, candidate)]
                assert previous['source_sha256'] == record['sha256']
                assert max(abs(a - b) for a, b in zip(
                    previous['initial_state'], initial[0, start + local_slot].tolist())) < 1e-3
                assert max(abs(a - b) for a, b in zip(
                    previous['fitted_state'], fitted['state'][0, local_slot].tolist())) < 1e-2
                assert abs(previous['prior_score'] - float(prior[0, candidate])) < 1e-4
        physical_error = np.array([fixed_parent[(section, branch)]['mapped96_error_um']
                                   for branch in union])
        actual = np.asarray(actual)
        fixed_positions = np.arange(14)
        natural_positions = np.array([union.index(branch) for branch in natural])
        assert abs(physical_error[fixed_positions].min() - min(
            fixed_parent[(section, branch)]['mapped96_error_um'] for branch in fixed)) < 1e-5
        for step in steps:
            fraction = np.asarray(predicted[step])
            score = prior[0, beam[0]].cpu().numpy() + np.log(np.maximum(fraction, 1e-4))
            selected = int(fixed_positions[score[fixed_positions].argmax()])
            natural_selected = int(natural_positions[score[natural_positions].argmax()])
            shared = {key: record[key] for key in ('section_id', 'animal_id',
                'synthetic_animal_id', 'specimen_id', 'experiment_id',
                'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
                'panel_physical_section_id', 'appearance_mode')}
            shared.update({'step': step, 'source_sha256': record['sha256'],
                'independent_fraction_readout': section in independent})
            for slot, branch in enumerate(union):
                row = {**shared, 'branch_id': branch, 'beam_slot': slot,
                    'in_fixed_beam': slot < 14, 'in_natural_beam': branch in natural,
                    'mapped96_error_um': float(physical_error[slot]),
                    'direct_prior': float(prior[0, branch]),
                    'quality_fraction': float(fraction[slot]),
                    'actual_correct_fraction': float(actual[slot]),
                    'score': float(score[slot]), 'fixed_selected': slot == selected,
                    'natural_selected': slot == natural_selected}
                candidates_file.write(json.dumps(row, allow_nan=False) + '\n')
                candidate_rows.append(row)
            row = {**shared, 'fixed_beam_branch_ids': fixed,
                'natural_beam_branch_ids': natural,
                'fixed_selected_branch': union[selected],
                'natural_selected_branch': union[natural_selected],
                'fixed_selected_mapped96_um': float(physical_error[selected]),
                'natural_selected_mapped96_um': float(physical_error[natural_selected]),
                'fixed_best14_mapped96_um': float(physical_error[fixed_positions].min()),
                'fixed_score_error_spearman': rho(score[fixed_positions],
                                                   -physical_error[fixed_positions]),
                'fixed_fraction_error_spearman': rho(fraction[fixed_positions],
                                                      -physical_error[fixed_positions]),
                'fixed_fraction_actual_spearman': rho(fraction[fixed_positions],
                                                       actual[fixed_positions]),
                'fixed_fraction_mae': float(np.abs(
                    fraction[fixed_positions] - actual[fixed_positions]).mean()),
                'natural_score_error_spearman': rho(score[natural_positions],
                                                     -physical_error[natural_positions])}
            sections_file.write(json.dumps(row, allow_nan=False) + '\n')
            section_rows.append(row)
    candidates_file.flush()
    sections_file.flush()

metrics = ('fixed_selected_mapped96_um', 'natural_selected_mapped96_um',
    'fixed_best14_mapped96_um', 'fixed_score_error_spearman',
    'fixed_fraction_error_spearman', 'fixed_fraction_actual_spearman',
    'fixed_fraction_mae', 'natural_score_error_spearman')
summary = {}
for step in steps:
    rows = [row for row in section_rows if row['step'] == step]
    groups = {'all': rows, 'independent_182': [row for row in rows
        if row['independent_fraction_readout']]}
    groups.update({f'appearance_{mode}': [row for row in rows
        if row['appearance_mode'] == mode]
        for mode in ('exact_black', 'raw', 'imperfect_brush')})
    summary[str(step)] = {}
    for name, group in groups.items():
        plans = sorted({row['synthetic_subject_plan_id'] for row in group})
        by_plan = {plan: {metric: float(np.mean([row[metric] for row in group
            if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}
            for plan in plans}
        summary[str(step)][name] = {'sections': len(group), 'plans': len(plans),
            'by_plan': by_plan,
            'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric]
                for plan in plans])) for metric in metrics}}
    assert summary[str(step)]['all']['sections'] == 246
    assert summary[str(step)]['independent_182']['sections'] == 182
    assert summary[str(step)]['all']['plans'] == 8
best_step = min(steps, key=lambda step: summary[str(step)]['all']['plan_equal_mean'][
    'fixed_selected_mapped96_um'])
selected = summary[str(best_step)]['all']['plan_equal_mean']
independent_head = summary[str(best_step)]['independent_182']['plan_equal_mean']
appearance = {mode: summary[str(best_step)][f'appearance_{mode}']['plan_equal_mean']
    for mode in ('exact_black', 'raw', 'imperfect_brush')}
summary['selected_development_checkpoint'] = {'step': best_step,
    'rule': 'lowest plan-equal fixed-beam selected mapped96 of 0/4000/12000'}
summary['selected_gate'] = {
    'fraction_error_rho_ge_0_30': independent_head['fixed_fraction_error_spearman'] >= .30,
    'selected_mapped96_le_2192um': selected['fixed_selected_mapped96_um'] <= 2192.,
    'fraction_mae_le_0_15': independent_head['fixed_fraction_mae'] <= .15,
    'intact_parent_best14': abs(selected['fixed_best14_mapped96_um'] -
        prior_summary['20000']['groups']['all']['plan_equal_mean'][
            'fixed_best14_mapped96_um']) < 1e-3,
    **{f'{mode}_within_150um_of_089': appearance[mode][
        'fixed_selected_mapped96_um'] <= baseline_summary['groups'][
            f'appearance_{mode}']['plan_equal_mean']['fitted_selected_mapped96_um'] + 150.
       for mode in appearance}}
summary['selected_gate']['all_pass'] = all(summary['selected_gate'].values())
assert len(section_rows) == 246 * len(steps)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(section_rows), 'candidates': len(candidate_rows),
    'config_sha256': sha(out / 'config.json'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'selected_step': best_step,
                  'selected_gate': summary['selected_gate']}), flush=True)
