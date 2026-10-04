"""Independent frozen-branch synthetic DEV readout for the 094 spatial verifier."""
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
from scipy.stats import rankdata, spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SPATIAL_VERIFIER_094_PROTOCOL_20261004.md'
run = root / 'runs/spatial_verifier_094_pilot'
parent = root / 'runs/spatial_joint_fit_092_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
ranking = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
diagnostic = root / 'runs/spatial_match_signal_093_diagnostic'
baseline_run = root / 'runs/one_shot_joint_atlas_feedback_089_development_eval'
out = root / 'runs/spatial_verifier_094_development_eval'
steps = (0, 2000, 8000, 20000)
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def at_chart(surface, chart):
    count = surface.shape[1]
    grid = ((chart + .5 / side) * 2 - 1)[None].expand(
        count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
        padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


def selected_prediction(prediction, branch, state):
    mode = branch // 2
    return {**prediction, 'state': state,
        'log_mass': prediction['log_mass'].gather(1, mode),
        'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def correlation(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if math.isfinite(value) else 0.


def auroc(confidence, correct):
    positive = np.concatenate(correct).astype(bool)
    ranks = rankdata(np.concatenate(confidence))
    count = int(positive.sum())
    return float((ranks[positive].sum() - count * (count + 1) / 2)
                 / (count * (len(positive) - count)))


assert sha(protocol) == '68277e40c472678a180a88d738f220dc87cafaf061152b133ec9eaaec45d1a8a'
config = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert finished['batches'] == finished['accepted_synthetic'] == config['batches'] == 20000
assert tuple(config['checkpoints']) == steps
assert not any(finished[key] or config[key] for key in (
    'calibrated', 'public_benchmark_used', 'real_labels_used',
    'external_pretrained_weights_used'))
assert sha(run / 'config.json') == finished['config_sha256']
assert sha(run / 'draws.jsonl') == finished['draws_sha256']
assert sha(run / 'training.jsonl') == finished['training_sha256']
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
assert all(sha(run / f'joint_step_{step:05d}.pt') ==
           finished['checkpoint_sha256'][str(step)] for step in steps)
assert config['protocol_sha256'] == finished['protocol_sha256'] == sha(protocol)

parent_receipt = json.loads((parent / 'completed.json').read_text())
assert sha(parent / 'joint_step_06000.pt') == parent_receipt['checkpoint_sha256']['6000'] \
    == '4ed54ad8c19a242d81d6778edd79796bcab3a56f21e720b88f6985fbd11cb4d7'
assert sha(parent / 'completed.json') == config['parent_092_completed_sha256']
assert sha(parent / 'config.json') == config['parent_092_config_sha256']
assert sha(parent / 'joint_step_06000.pt') == config['parent_092_sha256']

panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'completed.json') == '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256'] \
    == '481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open())
           if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8
rank_receipt = json.loads((ranking / 'completed.json').read_text())
assert sha(ranking / 'completed.json') == '5eb6dc84b615f3ce01ee9ac84ce124057a42a6dda39dbc2b999c87c1e05b3711'
assert sha(ranking / 'candidates.jsonl') == rank_receipt['candidates_sha256']
ranked = {row['section_id']: [None] * 14 for row in records}
for row in map(json.loads, (ranking / 'candidates.jsonl').open()):
    ranked[row['section_id']][row['beam_slot']] = row
assert all(len({item['branch_id'] for item in group}) == 14
           for group in ranked.values())

diagnostic_receipt = json.loads((diagnostic / 'completed.json').read_text())
assert sha(diagnostic / 'config.json') == diagnostic_receipt['config_sha256']
assert sha(diagnostic / 'candidates.jsonl') == diagnostic_receipt['candidates_sha256']
assert sha(diagnostic / 'summary.json') == diagnostic_receipt['summary_sha256']
inspected_ids = set(json.loads((diagnostic / 'config.json').read_text())['section_ids'])
independent_ids = {row['section_id'] for row in records} - inspected_ids
assert len(inspected_ids) == 64 and len(independent_ids) == 182
assert len({row['synthetic_subject_plan_id'] for row in records
            if row['section_id'] in independent_ids}) == 8

baseline_receipt = json.loads((baseline_run / 'completed.json').read_text())
assert sha(baseline_run / 'summary.json') == baseline_receipt['summary_sha256']
baseline = json.loads((baseline_run / 'summary.json').read_text())['6000']

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': finished['checkpoint_sha256'],
    'parent_completed_sha256': sha(parent / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent / 'joint_step_06000.pt'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_completed_sha256': sha(ranking / 'completed.json'),
    'ranking_candidates_sha256': sha(ranking / 'candidates.jsonl'),
    'diagnostic_completed_sha256': sha(diagnostic / 'completed.json'),
    'diagnostic_config_sha256': sha(diagnostic / 'config.json'),
    'inspected_093_section_ids': sorted(inspected_ids),
    'independent_094_section_ids': sorted(independent_ids),
    'baseline_completed_sha256': sha(baseline_run / 'completed.json'),
    'baseline_summary_sha256': sha(baseline_run / 'summary.json'),
    'steps': steps, 'fixed_beam': 'exact 086 14 branch IDs and slots',
    'natural_beam': 'current prior top 8 old and top 6 anchor branches; evaluated on union',
    'pixels': 'same 1024 deterministic surviving pixels as 086/089',
    'score': 'prior - fit_energy / clipped_exp(log_fit_temperature) - .05 * warp_cost',
    'site_correct': 'visible fine cell, peak supported, predicted CCF within 1500 um of truth',
    'predicted_inlier_fraction': 'sum(sigmoid(correct_logit) * visible * available) / visible count',
    'site_head_readout': '182 sections not inspected in 093; truth is readout only',
    'aggregation': 'section mean within plan, equal mean of eight represented plans',
    'numpy': np.__version__, 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()


def new_model():
    return OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()


model = new_model().requires_grad_(False)
spatial = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
sections = []
candidate_count = 0
site_auc = {}
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidate_stream, \
        (out / 'sections.jsonl').open('w') as section_stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and checkpoint['config'] == config
        assert not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        spatial.load_state_dict(checkpoint['spatial_fit'], strict=True)
        del checkpoint
        temperature = spatial.log_fit_temperature.clamp(
            math.log(.5), math.log(5.)).exp()
        confidence, correct = [], []
        for record in records:
            frozen = ranked[record['section_id']]
            if step == 0:
                assert sha(panel / record['file']) == record['sha256']
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            indices = valid.flatten().nonzero().flatten()
            chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                            device='cuda').round().long()]
            sample_hash = hashlib.sha256(chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
            assert sample_hash == frozen[0]['sample_indices_sha256']
            chart = torch.stack((chosen.remainder(side),
                chosen.div(side, rounding_mode='floor')), -1).float() / side
            reference = target.reshape(-1, 3)[chosen]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            updated = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)[0].tolist()
            fixed = [row['branch_id'] for row in frozen]
            union = list(dict.fromkeys(fixed + updated))
            beam = torch.tensor([union], device='cuda', dtype=torch.long)
            initial = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            states, evidences, surfaces = [], [], []
            scores, energies, warp_costs, supports = [], [], [], []
            inliers, qualities, coverages, correction_costs = [], [], [], []
            actual_correct = []
            for start in range(0, len(union), 2):
                branch = beam[:, start:start + 2]
                reflected = branch % 2
                state = initial[:, start:start + 2]
                fitted = spatial(prediction['feature'], state, reflected,
                    atlas, offsets, weights, source_shape=(side, side))
                evidence = fitted['spatial_evidence']
                selected = selected_prediction(prediction, branch, fitted['state'])
                mapped = model.map(selected, offsets,
                    torch.arange(branch.shape[1], device='cuda')[None], reflected,
                    (96, 96), atlas, weights, feature_side=96,
                    source_shape=(side, side), spatial_evidence=evidence)
                local = mapped['local_displacement_um'] / 1000
                dx, dy = local[..., 1:] - local[..., :-1], \
                         local[..., 1:, :] - local[..., :-1, :]
                warp = local.square().mean((-3, -2, -1)) + .1 * (
                    dx.square().mean((-3, -2, -1)) + dy.square().mean((-3, -2, -1)))
                score = prior.gather(1, branch) - fitted['fit_energy'] / temperature - .05 * warp
                states.append(fitted['state'])
                evidences.append(evidence)
                surfaces.append(mapped['centre_surface_ccf_ap_dv_ml_um'])
                scores.append(score)
                energies.append(fitted['fit_energy'])
                warp_costs.append(warp)
                supports.append((fitted['fine_match_support'] >= .5).any(2).float().mean((-2, -1)))
                qualities.append(fitted['quality'])
                coverages.append(fitted['coverage'])
                correction_costs.append(fitted['pose_correction_cost'])
                visible = F.interpolate(valid[:, None].float(), (32, 32),
                    mode='bilinear', align_corners=False)[:, 0] == 1
                available = (fitted['fine_match_support'] >= .5).any(2)
                predicted = fitted['fine_correct_logit'].sigmoid()
                inliers.append((predicted * visible[:, None] * available).sum((-2, -1))
                    / visible.sum((-2, -1))[:, None].clamp_min(1))
                if record['section_id'] in independent_ids:
                    truth = F.interpolate(target.permute(0, 3, 1, 2), (32, 32),
                        mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    peak = fitted['fine_match_logits'].masked_fill(
                        fitted['fine_match_support'] < .5, -1e4).argmax(2)
                    supported = fitted['fine_match_support'].gather(
                        2, peak[:, :, None]).squeeze(2) >= .5
                    positive = visible[:, None] & supported & (
                        (fitted['fine_match_ccf_um'] - truth[:, None]).norm(dim=-1) <= 1500)
                    actual_correct.append(positive.sum((-2, -1))
                        / visible.sum((-2, -1))[:, None].clamp_min(1))
                    if start < 14:
                        confidence.append(predicted[visible[:, None].expand_as(predicted)]
                            .cpu().numpy())
                        correct.append(positive[visible[:, None].expand_as(positive)]
                            .cpu().numpy())
            state_all = torch.cat(states, 1)
            evidence_all = torch.cat(evidences, 1)
            score = torch.cat(scores, 1)[0].cpu().numpy()
            energy = torch.cat(energies, 1)[0].cpu().numpy()
            warp_cost = torch.cat(warp_costs, 1)[0].cpu().numpy()
            support = torch.cat(supports, 1)[0].cpu().numpy()
            predicted_inlier = torch.cat(inliers, 1)[0].cpu().numpy()
            quality = torch.cat(qualities, 1)[0].cpu().numpy()
            coverage = torch.cat(coverages, 1)[0].cpu().numpy()
            correction_cost = torch.cat(correction_costs, 1)[0].cpu().numpy()
            actual_inlier = (torch.cat(actual_correct, 1)[0].cpu().numpy()
                             if actual_correct else None)
            mapped_error = (at_chart(torch.cat(surfaces, 1), chart)
                            - reference[None]).norm(dim=-1).mean(-1).cpu().numpy()
            fixed_positions = np.arange(14)
            natural_positions = np.array([union.index(branch) for branch in updated])
            fixed_selected = int(fixed_positions[score[fixed_positions].argmax()])
            natural_selected = int(natural_positions[score[natural_positions].argmax()])
            fixed_best = int(fixed_positions[mapped_error[fixed_positions].argmin()])
            natural_best = int(natural_positions[mapped_error[natural_positions].argmin()])
            branch = beam[:, fixed_selected:fixed_selected + 1]
            mapped256 = model.map(selected_prediction(prediction, branch,
                state_all[:, fixed_selected:fixed_selected + 1]), offsets,
                torch.zeros((1, 1), device='cuda', dtype=torch.long), branch % 2,
                (256, 256), atlas, weights, feature_side=96,
                source_shape=(side, side),
                spatial_evidence=evidence_all[:, fixed_selected:fixed_selected + 1])
            selected256_error = (at_chart(
                mapped256['centre_surface_ccf_ap_dv_ml_um'], chart)
                - reference[None]).norm(dim=-1).mean().item()
            shared = {key: record[key] for key in ('section_id', 'animal_id',
                'synthetic_animal_id', 'specimen_id', 'experiment_id',
                'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
                'panel_physical_section_id', 'appearance_mode')}
            shared.update({'step': step, 'source_file': record['file'],
                'source_sha256': record['sha256'],
                'sample_indices_sha256': sample_hash,
                'valid_pixels': record['valid_pixels'],
                'independent_head_readout': record['section_id'] in independent_ids})
            for slot, candidate in enumerate(union):
                candidate_stream.write(json.dumps({**shared,
                    'branch_id': candidate, 'beam_slot': slot,
                    'in_fixed_beam': slot < 14, 'in_natural_beam': candidate in updated,
                    'mode_index': candidate // 2, 'reflection': candidate % 2,
                    'initial_state': initial[0, slot].tolist(),
                    'fitted_state': state_all[0, slot].tolist(),
                    'mapped96_error_um': float(mapped_error[slot]),
                    'score': float(score[slot]),
                    'prior_score': float(prior[0, candidate]),
                    'fit_energy': float(energy[slot]),
                    'warp_cost': float(warp_cost[slot]),
                    'atlas_support_fraction': float(support[slot]),
                    'predicted_inlier_fraction': float(predicted_inlier[slot]),
                    'actual_correct_fraction': (None if actual_inlier is None
                        else float(actual_inlier[slot])),
                    'quality': float(quality[slot]),
                    'coverage': float(coverage[slot]),
                    'pose_correction_cost': float(correction_cost[slot]),
                    'fixed_selected': slot == fixed_selected,
                    'fixed_best': slot == fixed_best,
                    'natural_selected': slot == natural_selected,
                    'natural_best': slot == natural_best}, allow_nan=False) + '\n')
                candidate_count += 1
            row = {**shared, 'fixed_beam_branch_ids': fixed,
                'natural_beam_branch_ids': updated,
                'natural_beam_overlap_count': len(set(fixed) & set(updated)),
                'fixed_selected_branch': union[fixed_selected],
                'natural_selected_branch': union[natural_selected],
                'fixed_best_branch': union[fixed_best],
                'natural_best_branch': union[natural_best],
                'fixed_selected_mapped96_um': float(mapped_error[fixed_selected]),
                'fixed_selected_mapped256_um': selected256_error,
                'fixed_best14_mapped96_um': float(mapped_error[fixed_best]),
                'fixed_selected_regret_um': float(
                    mapped_error[fixed_selected] - mapped_error[fixed_best]),
                'natural_selected_mapped96_um': float(mapped_error[natural_selected]),
                'natural_best14_mapped96_um': float(mapped_error[natural_best]),
                'fixed_score_error_spearman': correlation(
                    score[fixed_positions], -mapped_error[fixed_positions]),
                'fixed_score_support_spearman': correlation(
                    score[fixed_positions], support[fixed_positions]),
                'natural_score_error_spearman': correlation(
                    score[natural_positions], -mapped_error[natural_positions]),
                'independent_predicted_inlier_error_spearman': (
                    correlation(predicted_inlier[fixed_positions],
                                -mapped_error[fixed_positions])
                    if actual_inlier is not None else None),
                'independent_actual_inlier_error_spearman': (
                    correlation(actual_inlier[fixed_positions],
                                -mapped_error[fixed_positions])
                    if actual_inlier is not None else None),
                'fixed_selected_fit_energy': float(energy[fixed_selected]),
                'fixed_selected_support': float(support[fixed_selected]),
                'fixed_selected_predicted_inlier': float(predicted_inlier[fixed_selected]),
                'fixed_selected_quality': float(quality[fixed_selected]),
                'fixed_selected_coverage': float(coverage[fixed_selected]),
                'fixed_selected_pose_correction_cost': float(
                    correction_cost[fixed_selected]),
                'fixed_selected_warp_cost': float(warp_cost[fixed_selected])}
            for name, positions in (('fixed', fixed_positions), ('natural', natural_positions)):
                order = positions[np.argsort(-score[positions], kind='stable')]
                best = mapped_error[positions].min()
                for k in (1, 3, 5, 14):
                    row[f'{name}_top{k}_regret_um'] = float(
                        mapped_error[order[:k]].min() - best)
            sections.append(row)
            section_stream.write(json.dumps(row, allow_nan=False) + '\n')
        site_auc[str(step)] = {'auroc': auroc(confidence, correct),
            'sites': int(sum(len(x) for x in correct)),
            'correct_sites': int(sum(np.count_nonzero(x) for x in correct))}
        candidate_stream.flush()
        section_stream.flush()
        print(json.dumps({'checkpoint': step, 'sections': len(records),
            'independent_site_auroc': site_auc[str(step)]['auroc']}), flush=True)

metrics = ('fixed_selected_mapped96_um', 'fixed_selected_mapped256_um',
    'fixed_best14_mapped96_um', 'fixed_selected_regret_um',
    'natural_selected_mapped96_um', 'natural_best14_mapped96_um',
    'fixed_score_error_spearman', 'fixed_score_support_spearman',
    'natural_score_error_spearman', 'natural_beam_overlap_count',
    'fixed_selected_fit_energy', 'fixed_selected_support',
    'fixed_selected_predicted_inlier', 'fixed_selected_quality',
    'fixed_selected_coverage', 'fixed_selected_pose_correction_cost',
    'fixed_selected_warp_cost',
    *(f'{name}_top{k}_regret_um' for name in ('fixed', 'natural')
      for k in (1, 3, 5, 14)))
summary = {}
for step in steps:
    group = [row for row in sections if row['step'] == step]
    subsets = {'all': group}
    subsets.update({f'appearance_{mode}': [row for row in group
        if row['appearance_mode'] == mode]
        for mode in ('exact_black', 'raw', 'imperfect_brush')})
    summary[str(step)] = {'groups': {}, 'independent_sites': site_auc[str(step)]}
    for name, subset in subsets.items():
        plans = sorted({row['synthetic_subject_plan_id'] for row in subset})
        by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan
            for row in subset), **{metric: float(np.mean([
            row[metric] for row in subset if row['synthetic_subject_plan_id'] == plan]))
            for metric in metrics}} for plan in plans}
        summary[str(step)]['groups'][name] = {
            'sections': len(subset), 'plans': len(plans), 'by_plan': by_plan,
            'plan_equal_mean': {metric: float(np.mean([
                by_plan[plan][metric] for plan in plans])) for metric in metrics}}
    independent = [row for row in group if row['independent_head_readout']]
    plans = sorted({row['synthetic_subject_plan_id'] for row in independent})
    summary[str(step)]['independent_candidates'] = {
        'sections': len(independent), 'plans': len(plans),
        'by_plan': {plan: {metric: float(np.mean([row[metric] for row in independent
            if row['synthetic_subject_plan_id'] == plan]))
            for metric in ('independent_predicted_inlier_error_spearman',
                           'independent_actual_inlier_error_spearman')}
            for plan in plans},
        'plan_equal_mean': {metric: float(np.mean([
            np.mean([row[metric] for row in independent
                if row['synthetic_subject_plan_id'] == plan]) for plan in plans]))
            for metric in ('independent_predicted_inlier_error_spearman',
                           'independent_actual_inlier_error_spearman')}}
    assert summary[str(step)]['groups']['all']['plans'] == 8
    assert summary[str(step)]['independent_candidates']['sections'] == 182

# Independent fit-only backward pass at the last checkpoint, outside inference mode.
checkpoint = torch.load(run / 'joint_step_20000.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
spatial.load_state_dict(checkpoint['spatial_fit'], strict=True)
del checkpoint
model.requires_grad_(True)
first = records[0]
with np.load(panel / first['file'], allow_pickle=False) as arrays:
    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
branches = [next(row['branch_id'] for row in ranked[first['section_id']]
                 if row['branch_id'] < 32),
            next(row['branch_id'] for row in ranked[first['section_id']]
                 if row['branch_id'] >= 32)]
with torch.enable_grad():
    prediction = model.predict(image)
    branch = torch.tensor([branches], device='cuda', dtype=torch.long)
    state = prediction['state'].gather(1,
        (branch // 2)[..., None].expand(-1, -1, 12))
    fitted = spatial(prediction['feature'], state, branch % 2,
        atlas, offsets, weights, source_shape=(side, side))
    pose_parameters = (model.pose[-1].weight, model.anchor_pose[-1].weight,
                       model.anchor_global[-1].weight)
    gradients = torch.autograd.grad(fitted['fit_energy'].sum(), pose_parameters,
        allow_unused=True)
    grad_norms = {key: (0. if grad is None else float(grad.norm()))
                  for key, grad in zip(('old_pose', 'anchor_pose', 'anchor_global'), gradients)}
summary['independent_fit_gradient_audit'] = {
    'section_id': first['section_id'], 'branch_ids': branches,
    'fit_only_pose_grad_norms': grad_norms}

final = summary['20000']['groups']['all']['plan_equal_mean']
head = summary['20000']['independent_candidates']['plan_equal_mean']
appearance = {mode: summary['20000']['groups'][f'appearance_{mode}']['plan_equal_mean']
              for mode in ('exact_black', 'raw', 'imperfect_brush')}
baseline_all = baseline['groups']['all']['plan_equal_mean']
summary['final_gate'] = {
    'selected_mapped96_le_2192um': final['fixed_selected_mapped96_um'] <= 2192.,
    'score_error_spearman_ge_0_30': final['fixed_score_error_spearman'] >= .30,
    'best14_mapped96_le_1011um': final['fixed_best14_mapped96_um'] <= 1011.,
    'independent_site_correct_auroc_ge_0_70': site_auc['20000']['auroc'] >= .70,
    'independent_predicted_inlier_error_spearman_ge_0_30':
        head['independent_predicted_inlier_error_spearman'] >= .30,
    **{f'{mode}_within_150um_of_089':
       appearance[mode]['fixed_selected_mapped96_um'] <=
       baseline['groups'][f'appearance_{mode}']['plan_equal_mean'][
           'fitted_selected_mapped96_um'] + 150.
       for mode in appearance},
    'independent_fit_gradient_path_nonzero':
        grad_norms['old_pose'] > 0 and
        grad_norms['anchor_pose'] + grad_norms['anchor_global'] > 0}
summary['final_gate']['all_pass'] = all(summary['final_gate'].values())
summary['delta_from_089_final_um'] = {
    'fixed_selected_mapped96': final['fixed_selected_mapped96_um']
        - baseline_all['fitted_selected_mapped96_um'],
    'fixed_best14_mapped96': final['fixed_best14_mapped96_um']
        - baseline_all['best14_mapped96_um']}
best_dev_step = min(steps, key=lambda step: summary[str(step)]['groups']['all'][
    'plan_equal_mean']['fixed_selected_mapped96_um'])
summary['selected_development_checkpoint'] = {
    'step': best_dev_step,
    'rule': 'minimum plan-equal fixed-beam selected mapped96 error among the four frozen checkpoints',
    'selected_mapped96_um': summary[str(best_dev_step)]['groups']['all'][
        'plan_equal_mean']['fixed_selected_mapped96_um'],
    'final_step20k_gate_passed': summary['final_gate']['all_pass']}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
assert len(sections) == 246 * len(steps)
(out / 'completed.json').write_text(json.dumps({
    'sections': len(sections), 'candidates': candidate_count,
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(sections),
    'candidates': candidate_count, 'gate': summary['final_gate']}), flush=True)
