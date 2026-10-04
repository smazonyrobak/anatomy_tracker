"""Read-only candidate-order diagnostic for the frozen 085 step-1000 selector."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

run = root / 'runs/allbeam_fitted_ranker_085_pilot'
frozen_eval = root / 'runs/allbeam_fitted_ranker_085_development_eval'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ALLBEAM_CANDIDATE_RANK_DIAGNOSTIC_086_PROTOCOL_20261004.md'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


def candidate_prediction(prediction, choice, state):
    mode = choice // 2
    return {**prediction, 'state': state,
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def infer(image, offsets, weights, model, head, atlas):
    prediction = model.predict(image)
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                        prior[:, 32:].topk(6, -1).indices + 32), -1)
    state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
    scores, surfaces, learned, qualities = [], [], [], []
    captured = []
    hook = model.fitted_matcher.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    for start in range(0, 14, 2):
        c = choice[:, start:start + 2]
        reflection = c % 2
        first = head(prediction['feature'], state[:, start:start + 2], reflection,
                     atlas, offsets, weights)
        selected = candidate_prediction(prediction, c, first['state'])
        index = torch.arange(2, device='cuda')[None]
        mapped64 = model.map(selected, offsets, index, reflection, (64, 64), atlas, weights,
                             return_refinement_feature=True, feature_side=64,
                             source_shape=(side, side), spatial_evidence=first['spatial_evidence'])
        second = head(prediction['feature'], first['state'], reflection, atlas, offsets,
                      weights, fit_summary=fit_summary(mapped64))
        selected['state'] = second['state']
        mapped96 = model.map(selected, offsets, index, reflection, (96, 96), atlas, weights,
                             feature_side=96, source_shape=(side, side),
                             spatial_evidence=second['spatial_evidence'])
        fitted = model.score_fitted_candidates(image, selected, mapped96, atlas, weights)
        assert len(captured) == start // 2 + 1
        score = fitted + second['quality_logit']
        scores.append(score)
        surfaces.append(mapped96['centre_surface_ccf_ap_dv_ml_um'])
        learned.append(captured[-1].reshape_as(fitted))
        qualities.append(second['quality_logit'])
    hook.remove()
    return (choice, torch.cat(scores, 1), torch.cat(surfaces, 1),
            prior.gather(1, choice), torch.cat(learned, 1), torch.cat(qualities, 1))


def field_at_chart(surface, chart):
    count = surface.shape[1]
    grid = (chart + .5 / side) * 2 - 1
    grid = grid[None].expand(count, -1, -1).reshape(count, 1, len(chart), 2)
    field = surface[0].permute(0, 3, 1, 2)
    return F.grid_sample(field, grid, padding_mode='border',
                         align_corners=False).squeeze(2).transpose(1, 2)


assert hashlib.sha256(protocol.read_bytes().replace(b'\r\n', b'\n')).hexdigest() \
       == 'b77002ba69df4e24ed9e2370d031838ad3bfeaa22ffdd4f242b2a04d7a47190c'
assert sha(run / 'completed.json') == '707bbe5ff24f506a0da7b69e21799d607d91d7e44085c55f71437d4cf8ec0a90'
assert sha(panel / 'completed.json') == '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
assert sha(frozen_eval / 'completed.json') == '0130c6198c758e6ab274c01525e340feb2a549956568751a50f0ec507f82a345'
assert sha(source / 'evaluate_allbeam_fitted_ranker_085.py') == '9d5520ab55a5b0e1af718b087ae77dc5b8a23b3ac60287fc66be53bee4c1a114'

training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert finished['batches'] == training['batches'] == 1000
assert not finished['calibrated'] and not finished['public_benchmark_used']
assert sha(run / 'config.json') == finished['config_sha256']
assert sha(run / 'draws.jsonl') == finished['draws_sha256']
assert sha(run / 'training.jsonl') == finished['training_sha256']
assert sha(run / 'ranker_step_01000.pt') == finished['checkpoint_sha256']['1000'] \
       == '67ba266f8583a626ee63bae3202afaeb04602ad10cfc1395aae41873dc48b9ce'
assert sha(training['parent']) == training['parent_sha256']
assert all(sha(source / name) == digest for name, digest in training['source_sha256'].items())
assert all(sha(source / name) == digest
           for name, digest in training['synthetic_provenance']['source_sha256'].items())
assert sha(root / 'runs/dense_atlas_correspondence_083_pilot/completed.json') \
       == training['pilot_083_completed_sha256']
assert sha(root / 'runs/dense_atlas_correspondence_083_pilot/match_step_06000.pt') \
       == training['pilot_083_checkpoint_6000_sha256']

panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256'] \
       == '481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8

frozen_receipt = json.loads((frozen_eval / 'completed.json').read_text())
frozen_config = json.loads((frozen_eval / 'config.json').read_text())
assert sha(frozen_eval / 'config.json') == frozen_receipt['config_sha256']
assert sha(frozen_eval / 'rows.jsonl') == frozen_receipt['rows_sha256'] \
       == 'cc44ce1bd7e6959f39c59aacb314754b59f09c76b0a7cdaa7d2e131691491abd'
assert sha(frozen_eval / 'summary.json') == frozen_receipt['summary_sha256']
assert frozen_config['evaluator_sha256'] == sha(source / 'evaluate_allbeam_fitted_ranker_085.py')
assert frozen_config['training_completed_sha256'] == sha(run / 'completed.json')
assert frozen_config['synthetic_records_sha256'] == sha(panel / 'records.jsonl')
assert frozen_config['checkpoint_sha256']['1000'] == sha(run / 'ranker_step_01000.pt')
frozen = {row['section_id']: row for row in map(json.loads, (frozen_eval / 'rows.jsonl').open())
          if row['set'] == 'synthetic' and row['step'] == 1000}
assert len(frozen) == len(records) and set(frozen) == {row['section_id'] for row in records}
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in records], .25))

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': sha(run / 'ranker_step_01000.pt'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'frozen_evaluator_source_sha256': sha(source / 'evaluate_allbeam_fitted_ranker_085.py'),
    'frozen_evaluation_completed_sha256': sha(frozen_eval / 'completed.json'),
    'frozen_evaluation_rows_sha256': sha(frozen_eval / 'rows.jsonl'),
    'model_source_sha256': sha(source / 'arbitrary_plane_one_shot_model.py'),
    'feedback_source_sha256': sha(source / 'whole_slice_atlas_feedback_083.py'),
    'sections': len(records), 'synthetic_plans': 8, 'candidate_count': 14,
    'support_cutoff_visible_fraction': support_cutoff,
    'numpy': np.__version__, 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False
}, indent=2))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
head = WholeSliceAtlasFeedback083().cuda().eval()
model.requires_grad_(False)
head.requires_grad_(False)
checkpoint = torch.load(run / 'ranker_step_01000.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1000 and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
head.load_state_dict(checkpoint['feedback'], strict=True)
del checkpoint

sections = []
max_parity_error_um = 0.
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidate_stream, \
        (out / 'sections.jsonl').open('w') as section_stream:
    for record in records:
        previous = frozen[record['section_id']]
        assert sha(panel / record['file']) == record['sha256'] == previous['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        indices = valid.flatten().nonzero().flatten()
        chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                        device='cuda').round().long()]
        chart = torch.stack((chosen.remainder(side),
            chosen.div(side, rounding_mode='floor')), -1).float() / side
        reference = target.reshape(-1, 3)[chosen]
        choice, score, surface, prior, learned, quality = infer(
            image, offsets, weights, model, head, atlas)
        error = (field_at_chart(surface, chart) - reference[None]).norm(dim=-1).mean(-1)
        branches = choice[0].cpu().numpy()
        fitted_scores = score[0].cpu().numpy()
        prior_scores = prior[0].cpu().numpy()
        learned_scores = learned[0].cpu().numpy()
        quality_scores = quality[0].cpu().numpy()
        errors = error.cpu().numpy()
        fitted_order = np.argsort(-fitted_scores, kind='stable')
        prior_order = np.argsort(-prior_scores, kind='stable')
        truth_order = np.argsort(errors, kind='stable')
        fitted_rank = np.empty(14, dtype=int)
        prior_rank = np.empty(14, dtype=int)
        truth_rank = np.empty(14, dtype=int)
        fitted_rank[fitted_order] = np.arange(1, 15)
        prior_rank[prior_order] = np.arange(1, 15)
        truth_rank[truth_order] = np.arange(1, 15)
        selected = int(score[0].argmax())
        prior_selected = int(prior[0].argmax())
        best = int(error.argmin())
        parity = max(abs(float(error[selected]) - previous['model_selected_mapped96_um']),
                     abs(float(error[best]) - previous['model_best14_mapped96_um']))
        assert int(branches[selected]) == previous['model_selected_branch'] and parity <= 1.
        max_parity_error_um = max(max_parity_error_um, parity)
        visible_fraction = record['valid_pixels'] / side**2
        low_support = visible_fraction <= support_cutoff
        assert len(indices) == record['valid_pixels']
        assert low_support == previous['low_support']
        assert visible_fraction == previous['visible_fraction']
        shared = {key: record[key] for key in ('section_id', 'animal_id', 'synthetic_animal_id',
            'specimen_id', 'experiment_id', 'synthetic_subject_plan_id',
            'panel_physical_section_id', 'appearance_mode')}
        shared.update({'source_file': record['file'], 'source_sha256': record['sha256'],
                       'sample_indices_sha256': hashlib.sha256(
                           chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest(),
                       'valid_pixels': record['valid_pixels'], 'visible_fraction': visible_fraction,
                       'low_support': low_support})
        for slot in range(14):
            row = {**shared, 'beam_slot': slot, 'branch_id': int(branches[slot]),
                   'family': 'old' if branches[slot] < 32 else 'anchor',
                   'mode_index': int(branches[slot] // 2), 'reflection': int(branches[slot] % 2),
                   'prior_score': float(prior_scores[slot]),
                   'fitted_learned_score': float(learned_scores[slot]),
                   'quality_score': float(quality_scores[slot]),
                   'warp_penalty_score': float(fitted_scores[slot] - prior_scores[slot]
                                               - learned_scores[slot] - quality_scores[slot]),
                   'fitted_score': float(fitted_scores[slot]),
                   'mapped96_error_um': float(errors[slot]),
                   'prior_rank': int(prior_rank[slot]), 'fitted_rank': int(fitted_rank[slot]),
                   'physical_error_rank': int(truth_rank[slot]),
                   'prior_selected': slot == prior_selected,
                   'fitted_selected': slot == selected, 'truth_best': slot == best,
                   'parent_selected': int(branches[slot]) == previous['parent_selected_branch']}
            candidate_stream.write(json.dumps(row, allow_nan=False) + '\n')
        section = {**shared, 'fitted_selected_branch': int(branches[selected]),
                   'prior_selected_branch': int(branches[prior_selected]),
                   'truth_best_branch': int(branches[best]),
                   'parent_selected_branch': previous['parent_selected_branch'],
                   'selection_transition': ('anchor' if previous['parent_selected_branch'] >= 32 else 'old')
                                           + '_to_' + ('anchor' if branches[selected] >= 32 else 'old'),
                   'selection_changed_from_parent': int(branches[selected]) != previous['parent_selected_branch'],
                   'fitted_selected_mapped96_um': float(errors[selected]),
                   'prior_selected_mapped96_um': float(errors[prior_selected]),
                   'truth_best14_mapped96_um': float(errors[best]),
                   'truth_best_fitted_rank': int(fitted_rank[best]),
                   'truth_best_prior_rank': int(prior_rank[best]),
                   'parent_selected_mapped256_um': previous['parent_selected_mapped256_um'],
                   'model_selected_mapped256_um': previous['model_selected_mapped256_um'],
                   'model_minus_parent_mapped256_um': previous['model_selected_mapped256_um']
                                                     - previous['parent_selected_mapped256_um']}
        for name, values, order in (('fitted', fitted_scores, fitted_order),
                                    ('prior', prior_scores, prior_order),
                                    ('learned', learned_scores, np.argsort(-learned_scores, kind='stable'))):
            section[name + '_score_error_spearman'] = float(spearmanr(values, -errors).statistic)
            section[name + '_score_error_kendall'] = float(kendalltau(values, -errors).statistic)
            if name != 'learned':
                for k in (1, 3, 5, 14):
                    section[f'{name}_top{k}_regret_um'] = float(errors[order[:k]].min() - errors[best])
        assert all(np.isfinite(value) for key, value in section.items()
                   if key.endswith(('_spearman', '_kendall')))
        sections.append(section)
        section_stream.write(json.dumps(section, allow_nan=False) + '\n')

metrics = ('fitted_selected_mapped96_um', 'prior_selected_mapped96_um',
           'truth_best14_mapped96_um', 'model_minus_parent_mapped256_um',
           'fitted_score_error_spearman', 'fitted_score_error_kendall',
           'prior_score_error_spearman', 'prior_score_error_kendall',
           'learned_score_error_spearman', 'learned_score_error_kendall',
           *(f'{name}_top{k}_regret_um' for name in ('fitted', 'prior') for k in (1, 3, 5, 14)))
groups = {'all': sections}
groups.update({f'appearance_{mode}': [row for row in sections if row['appearance_mode'] == mode]
               for mode in ('exact_black', 'raw', 'imperfect_brush')})
groups.update({f'low_support_{value}': [row for row in sections if row['low_support'] == value]
               for value in (True, False)})
groups.update({f'changed_from_parent_{value}': [row for row in sections
                if row['selection_changed_from_parent'] == value] for value in (True, False)})
groups.update({f'bank_{transition}': [row for row in sections
                if row['selection_transition'] == transition]
               for transition in ('old_to_old', 'old_to_anchor', 'anchor_to_old', 'anchor_to_anchor')
               if any(row['selection_transition'] == transition for row in sections)})
summary = {'groups': {}, 'max_frozen_mapped96_parity_error_um': max_parity_error_um,
           'changed_from_parent_sections': sum(row['selection_changed_from_parent'] for row in sections),
           'model_256_better_sections': sum(row['model_minus_parent_mapped256_um'] < 0 for row in sections),
           'model_256_worse_sections': sum(row['model_minus_parent_mapped256_um'] > 0 for row in sections)}
for name, group in groups.items():
    plans = sorted({row['synthetic_subject_plan_id'] for row in group})
    by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan for row in group),
                      **{metric: float(np.mean([row[metric] for row in group
                          if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}}
               for plan in plans}
    summary['groups'][name] = {'sections': len(group), 'plans': len(plans), 'by_plan': by_plan,
        'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric] for plan in plans]))
                            for metric in metrics}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(sections), 'candidates': len(sections) * 14,
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False
}, indent=2))
