"""Frozen 086-beam synthetic-DEV readout of the four 089 joint-feedback checkpoints."""
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
os.environ['TRITON_CACHE_DIR'] = str(root / 'cache/triton')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/JOINT_ATLAS_FEEDBACK_089_PROTOCOL_20261004.md'
run = root / 'runs/one_shot_joint_atlas_feedback_089_pilot'
parent_run = root / 'runs/allbeam_fitted_ranker_085_pilot'
parent = parent_run / 'ranker_step_01000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
rank_run = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
frozen_eval = root / 'runs/allbeam_fitted_ranker_085_development_eval'
out = root / 'runs/one_shot_joint_atlas_feedback_089_development_eval'
steps, side = (0, 1000, 3000, 6000), 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


def candidate_prediction(prediction, branch, state):
    mode = branch // 2
    return {**prediction, 'state': state,
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def field_at_chart(surface, chart):
    count = surface.shape[1]
    grid = (chart + .5 / side) * 2 - 1
    grid = grid[None].expand(count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
                         padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


def ranks(values, descending=False):
    order = np.argsort(-values if descending else values, kind='stable')
    result = np.empty(len(values), dtype=int)
    result[order] = np.arange(1, len(values) + 1)
    return order, result


def assert_finite(value):
    if isinstance(value, dict):
        for nested in value.values():
            assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_finite(nested)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        assert math.isfinite(value)


assert sha(protocol) == 'f580ce63afa04c05ca5deb5512934c68ab8264603df8f0cb477e6e72bb3acb96'
training = json.loads((run / 'config.json').read_text())
finished = json.loads((run / 'completed.json').read_text())
assert finished['batches'] == finished['accepted_synthetic'] == training['batches'] == 6000
assert tuple(training['checkpoints']) == steps
assert training['accepted_synthetic_per_batch'] == 1
assert training['candidate_beam'] == {'old': 8, 'anchor': 6}
for receipt in (training, finished):
    assert not any(receipt[key] for key in ('calibrated', 'public_benchmark_used',
                                            'real_labels_used', 'external_pretrained_weights_used'))
assert sha(run / 'config.json') == finished['config_sha256']
assert sha(run / 'draws.jsonl') == finished['draws_sha256']
assert sha(run / 'training.jsonl') == finished['training_sha256']
assert set(finished['checkpoint_sha256']) == {str(step) for step in steps}
assert all(sha(run / f'joint_step_{step:05d}.pt') == finished['checkpoint_sha256'][str(step)]
           for step in steps)
assert Path(training['parent_085']) == parent
assert sha(parent) == training['parent_085_sha256'] == finished['parent_085_sha256'] \
       == '67ba266f8583a626ee63bae3202afaeb04602ad10cfc1395aae41873dc48b9ce'
assert sha(parent_run / 'config.json') == training['parent_085_config_sha256']
assert sha(parent_run / 'completed.json') == training['parent_085_completed_sha256'] \
       == '707bbe5ff24f506a0da7b69e21799d607d91d7e44085c55f71437d4cf8ec0a90'
assert all(sha(source / name) == digest for name, digest in training['source_sha256'].items())
assert all(sha(source / name) == digest for name, digest in
           training['synthetic_provenance']['source_sha256'].items())
assert training['synthetic_provenance']['all_train_bases'] == 64
assert sha(root / 'runs/dense_atlas_correspondence_083_pilot/match_step_06000.pt') \
       == training['parent_083_checkpoint_sha256']

attempts, accepted, attempts_by_step = 0, 0, {0: 0}
physical_ids = set()
virtual_subjects = training['synthetic_provenance']['virtual_subjects']
for draw in map(json.loads, (run / 'draws.jsonl').open()):
    assert draw['step'] == accepted + 1
    assert draw['seed_prefix'] == [training['seed'] * 1000000 + attempts,
                                   draw['virtual_index'], 0]
    assert 0 <= draw['virtual_index'] < len(virtual_subjects)
    assert draw['virtual_subject_id'] == virtual_subjects[draw['virtual_index']]['virtual_subject_id']
    assert draw['base_lineage']['split'] == draw['split'] == 'train'
    assert draw['shape_h_w'] == [side, side]
    assert draw['one_shot']['seed_branch'] == 19
    assert draw['physical_section_id'] not in physical_ids
    physical_ids.add(draw['physical_section_id'])
    attempts += 1
    if draw['used']:
        accepted += 1
        if accepted in steps:
            attempts_by_step[accepted] = attempts
assert accepted == 6000 and len(physical_ids) == attempts

logs = list(map(json.loads, (run / 'training.jsonl').open()))
assert [row['batch'] for row in logs] == [1, *range(100, 6001, 100)]
fit_weights = []
for row in logs:
    assert row['synthetic_presentations'] == row['batch']
    assert_finite(row)
    assert row['gradient_norm'] >= 0 and row['fit_weight'] >= 0
    assert row['fit_weight'] <= row['fit_weight_cap'] + 1e-12
    assert set(row['base_gradient_norms']) == set(row['fit_gradient_norms']) \
           == {'old_pose', 'anchor_pose', 'anchor_global', 'warp', 'head_pose', 'head_mapper'}
    fit_weights.append(row['fit_weight'])
assert all(right <= left + 1e-12 for left, right in zip(fit_weights, fit_weights[1:]))
logged_weight = {row['batch']: row['fit_weight'] for row in logs}

assert sha(panel / 'completed.json') == '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
panel_receipt = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'protocol.json') == panel_receipt['protocol_sha256']
assert sha(root / 'data/pose_feedback_061_fresh_synthetic_dev_plans_001/completed.json') \
       == panel_receipt['plan_completed_sha256']
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256'] \
       == '481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a'
all_records = list(map(json.loads, (panel / 'records.jsonl').open()))
assert len(all_records) == panel_receipt['physical_sections'] == 256
assert sum(row['eligible'] for row in all_records) == panel_receipt['eligible'] == 246
assert len({row['section_id'] for row in all_records}) == 256
assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
records = [row for row in all_records if row['eligible']]
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in records], .25))

assert sha(rank_run / 'completed.json') == '5eb6dc84b615f3ce01ee9ac84ce124057a42a6dda39dbc2b999c87c1e05b3711'
rank_receipt = json.loads((rank_run / 'completed.json').read_text())
rank_config = json.loads((rank_run / 'config.json').read_text())
assert rank_receipt['sections'] == 246 and rank_receipt['candidates'] == 246 * 14
for name in ('config', 'candidates', 'sections', 'summary'):
    suffix = 'jsonl' if name in ('candidates', 'sections') else 'json'
    assert sha(rank_run / f'{name}.{suffix}') == rank_receipt[f'{name}_sha256']
assert rank_config['panel_completed_sha256'] == sha(panel / 'completed.json')
assert rank_config['panel_records_sha256'] == sha(panel / 'records.jsonl')
assert rank_config['checkpoint_sha256'] == sha(parent)
assert rank_config['script_sha256'] == sha(source / 'diagnose_allbeam_candidate_ranking_086.py')
assert rank_config['model_source_sha256'] == sha(source / 'arbitrary_plane_one_shot_model.py')
assert rank_config['feedback_source_sha256'] == sha(source / 'whole_slice_atlas_feedback_083.py')
assert rank_config['support_cutoff_visible_fraction'] == support_cutoff
assert sha(frozen_eval / 'completed.json') == rank_config['frozen_evaluation_completed_sha256'] \
       == '0130c6198c758e6ab274c01525e340feb2a549956568751a50f0ec507f82a345'
frozen_receipt = json.loads((frozen_eval / 'completed.json').read_text())
assert sha(frozen_eval / 'rows.jsonl') == frozen_receipt['rows_sha256'] \
       == rank_config['frozen_evaluation_rows_sha256']
assert sha(frozen_eval / 'config.json') == frozen_receipt['config_sha256']
assert sha(frozen_eval / 'summary.json') == frozen_receipt['summary_sha256']
ranked = {record['section_id']: [None] * 14 for record in records}
for row in map(json.loads, (rank_run / 'candidates.jsonl').open()):
    slot = row['beam_slot']
    assert row['section_id'] in ranked and 0 <= slot < 14
    assert ranked[row['section_id']][slot] is None
    ranked[row['section_id']][slot] = row
rank_sections = {row['section_id']: row for row in
                 map(json.loads, (rank_run / 'sections.jsonl').open())}
assert len(rank_sections) == len(ranked) == 246
for record in records:
    old = ranked[record['section_id']]
    assert all(row is not None for row in old)
    branches = [row['branch_id'] for row in old]
    assert len(set(branches)) == 14
    assert all(branch < 32 for branch in branches[:8])
    assert all(branch >= 32 for branch in branches[8:])
    assert all(row['source_sha256'] == record['sha256'] for row in old)
    assert all(row['sample_indices_sha256'] == old[0]['sample_indices_sha256'] for row in old)
rank_summary = json.loads((rank_run / 'summary.json').read_text())

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': finished['checkpoint_sha256'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_086_completed_sha256': sha(rank_run / 'completed.json'),
    'ranking_086_candidates_sha256': sha(rank_run / 'candidates.jsonl'),
    'ranking_086_sections_sha256': sha(rank_run / 'sections.jsonl'),
    'frozen_085_evaluation_completed_sha256': sha(frozen_eval / 'completed.json'),
    'frozen_085_evaluation_rows_sha256': sha(frozen_eval / 'rows.jsonl'),
    'steps': steps, 'sections_per_step': 246, 'candidate_count': 14,
    'beam': 'exact 086 branch IDs and slots per section; updated model-prior beam measured only for overlap',
    'pixels': 'same 1024 deterministic valid flattened indices as 086',
    'mapping': '083 pass 1, 64-grid map and fit summary, 083 pass 2, 96-grid map',
    'pose_metric': 'five fixed points versus frozen target_state plus 1024-pixel rigid CCF error',
    'mapped_metric': 'mean 3-D CCF error at the same 1024 surviving pixels, micrometres',
    'aggregation': 'section mean within synthetic subject plan, then equal mean of represented plans',
    'support_cutoff_visible_fraction': support_cutoff,
    'numpy': np.__version__, 'torch': torch.__version__,
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
head = WholeSliceAtlasFeedback083().cuda().eval()
model.requires_grad_(False)
head.requires_grad_(False)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
sections = []
candidate_count = 0
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidate_stream, \
        (out / 'sections.jsonl').open('w') as section_stream:
    for step in steps:
        checkpoint = torch.load(run / f'joint_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        assert json.loads(json.dumps(checkpoint['config'])) == training
        assert checkpoint['draw_seed'] == training['seed'] * 1000000 + attempts_by_step[step]
        assert checkpoint['fit_weight'] == (None if step == 0 else logged_weight[step])
        model.load_state_dict(checkpoint['model'], strict=True)
        head.load_state_dict(checkpoint['feedback'], strict=True)
        del checkpoint
        for record in records:
            frozen = ranked[record['section_id']]
            previous = rank_sections[record['section_id']]
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                target_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                target_reflection = torch.as_tensor(arrays['reflection'].copy(), device='cuda').reshape(1)
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            indices = valid.flatten().nonzero().flatten()
            assert len(indices) == record['valid_pixels']
            chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                            device='cuda').round().long()]
            sample_hash = hashlib.sha256(chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
            assert sample_hash == frozen[0]['sample_indices_sha256']
            chart = torch.stack((chosen.remainder(side),
                chosen.div(side, rounding_mode='floor')), -1).float() / side
            reference = target.reshape(-1, 3)[chosen]
            true_five = points(target_state, target_reflection, corners)
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            updated_beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                                      prior[:, 32:].topk(6, -1).indices + 32), -1)[0].tolist()
            branches = [row['branch_id'] for row in frozen]
            beam = torch.tensor([branches], device='cuda', dtype=torch.long)
            initial = prediction['state'].gather(1, (beam // 2)[..., None].expand(-1, -1, 12))
            initial_rigid = (points(initial, beam % 2, chart) - reference).norm(dim=-1).mean(-1)[0]
            initial_five = (points(initial, beam % 2, corners) - true_five[:, None]).norm(dim=-1).mean(-1)[0]
            first_states, final_states, scores, learned, qualities = [], [], [], [], []
            surfaces64, surfaces96, warp64, warp96 = [], [], [], []
            captured = []
            hook = model.fitted_matcher.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output))
            for start in range(0, 14, 2):
                branch = beam[:, start:start + 2]
                reflection = branch % 2
                first = head(prediction['feature'], initial[:, start:start + 2], reflection,
                             atlas, offsets, weights)
                selected = candidate_prediction(prediction, branch, first['state'])
                index = torch.arange(2, device='cuda')[None]
                mapped64 = model.map(selected, offsets, index, reflection, (64, 64),
                                     atlas, weights, return_refinement_feature=True,
                                     feature_side=64, source_shape=(side, side),
                                     spatial_evidence=first['spatial_evidence'])
                second = head(prediction['feature'], first['state'], reflection,
                              atlas, offsets, weights, fit_summary=fit_summary(mapped64))
                selected['state'] = second['state']
                mapped96 = model.map(selected, offsets, index, reflection, (96, 96),
                                     atlas, weights, feature_side=96,
                                     source_shape=(side, side),
                                     spatial_evidence=second['spatial_evidence'])
                fitted = model.score_fitted_candidates(image, selected, mapped96,
                                                       atlas, weights)
                assert len(captured) == start // 2 + 1
                first_states.append(first['state'])
                final_states.append(second['state'])
                scores.append(fitted + second['quality_logit'])
                learned.append(captured[-1].reshape_as(fitted))
                qualities.append(second['quality_logit'])
                surfaces64.append(mapped64['centre_surface_ccf_ap_dv_ml_um'])
                surfaces96.append(mapped96['centre_surface_ccf_ap_dv_ml_um'])
                warp64.append(mapped64['local_displacement_um'])
                warp96.append(mapped96['local_displacement_um'])
            hook.remove()
            first_state = torch.cat(first_states, 1)
            final_state = torch.cat(final_states, 1)
            score = torch.cat(scores, 1)[0].cpu().numpy()
            learned_score = torch.cat(learned, 1)[0].cpu().numpy()
            quality_score = torch.cat(qualities, 1)[0].cpu().numpy()
            prior_score = prior.gather(1, beam)[0].cpu().numpy()
            first_rigid = (points(first_state, beam % 2, chart) - reference).norm(dim=-1).mean(-1)[0]
            final_rigid = (points(final_state, beam % 2, chart) - reference).norm(dim=-1).mean(-1)[0]
            first_five = (points(first_state, beam % 2, corners) - true_five[:, None]).norm(dim=-1).mean(-1)[0]
            final_five = (points(final_state, beam % 2, corners) - true_five[:, None]).norm(dim=-1).mean(-1)[0]
            mapped64_error = (field_at_chart(torch.cat(surfaces64, 1), chart)
                              - reference[None]).norm(dim=-1).mean(-1).cpu().numpy()
            mapped96_error = (field_at_chart(torch.cat(surfaces96, 1), chart)
                              - reference[None]).norm(dim=-1).mean(-1).cpu().numpy()
            warp64_magnitude = torch.cat(warp64, 1).square().sum(2).sqrt().mean((-2, -1))[0].cpu().numpy()
            warp96_magnitude = torch.cat(warp96, 1).square().sum(2).sqrt().mean((-2, -1))[0].cpu().numpy()
            initial_rigid = initial_rigid.cpu().numpy()
            first_rigid = first_rigid.cpu().numpy()
            final_rigid = final_rigid.cpu().numpy()
            initial_five = initial_five.cpu().numpy()
            first_five = first_five.cpu().numpy()
            final_five = final_five.cpu().numpy()
            fitted_order, fitted_rank = ranks(score, descending=True)
            prior_order, prior_rank = ranks(prior_score, descending=True)
            _, physical_rank = ranks(mapped96_error)
            selected = int(fitted_order[0])
            prior_selected = int(prior_order[0])
            best = int(mapped96_error.argmin())
            if step == 0:
                assert branches == updated_beam
                assert np.allclose(prior_score, [row['prior_score'] for row in frozen], rtol=1e-5, atol=1e-4)
                assert np.allclose(score, [row['fitted_score'] for row in frozen], rtol=1e-5, atol=1e-4)
                assert np.allclose(mapped96_error, [row['mapped96_error_um'] for row in frozen],
                                   rtol=1e-5, atol=1.)
                assert branches[selected] == previous['fitted_selected_branch']
                assert branches[best] == previous['truth_best_branch']
                assert abs(float(mapped96_error[selected]) - previous['fitted_selected_mapped96_um']) <= 1.
                assert abs(float(mapped96_error[best]) - previous['truth_best14_mapped96_um']) <= 1.
            shared = {key: record[key] for key in ('section_id', 'animal_id',
                'synthetic_animal_id', 'specimen_id', 'experiment_id',
                'synthetic_subject_plan_id', 'synthetic_subject_realization_id',
                'panel_physical_section_id', 'appearance_mode')}
            shared.update({'step': step, 'source_file': record['file'],
                           'source_sha256': record['sha256'],
                           'sample_indices_sha256': sample_hash,
                           'valid_pixels': record['valid_pixels'],
                           'visible_fraction': record['valid_pixels'] / side**2,
                           'low_support': record['valid_pixels'] / side**2 <= support_cutoff})
            updated_set = set(updated_beam)
            for slot, branch in enumerate(branches):
                row = {**shared, 'beam_slot': slot, 'branch_id': branch,
                       'family': 'old' if branch < 32 else 'anchor',
                       'mode_index': branch // 2, 'reflection': branch % 2,
                       'in_updated_prior_beam': branch in updated_set,
                       'initial_state': initial[0, slot].tolist(),
                       'first_pass_state': first_state[0, slot].tolist(),
                       'corrected_state': final_state[0, slot].tolist(),
                       'initial_rigid_error_um': float(initial_rigid[slot]),
                       'first_pass_rigid_error_um': float(first_rigid[slot]),
                       'corrected_rigid_error_um': float(final_rigid[slot]),
                       'initial_five_point_pose_error_um': float(initial_five[slot]),
                       'first_pass_five_point_pose_error_um': float(first_five[slot]),
                       'corrected_five_point_pose_error_um': float(final_five[slot]),
                       'first_pass_mapped64_error_um': float(mapped64_error[slot]),
                       'mapped96_error_um': float(mapped96_error[slot]),
                       'first_pass_warp64_magnitude_um': float(warp64_magnitude[slot]),
                       'warp96_magnitude_um': float(warp96_magnitude[slot]),
                       'prior_score': float(prior_score[slot]),
                       'fitted_learned_score': float(learned_score[slot]),
                       'quality_score': float(quality_score[slot]),
                       'warp_penalty_score': float(score[slot] - prior_score[slot]
                                                   - learned_score[slot] - quality_score[slot]),
                       'fitted_score': float(score[slot]),
                       'prior_rank': int(prior_rank[slot]),
                       'fitted_rank': int(fitted_rank[slot]),
                       'physical_error_rank': int(physical_rank[slot]),
                       'prior_selected': slot == prior_selected,
                       'fitted_selected': slot == selected,
                       'truth_best': slot == best}
                candidate_stream.write(json.dumps(row, allow_nan=False) + '\n')
                candidate_count += 1
            section = {**shared,
                'fitted_selected_branch': branches[selected],
                'prior_selected_branch': branches[prior_selected],
                'best14_branch': branches[best],
                'updated_prior_beam_branch_ids': updated_beam,
                'updated_prior_beam_overlap_count': len(updated_set.intersection(branches)),
                'fitted_selected_initial_rigid_um': float(initial_rigid[selected]),
                'fitted_selected_first_pass_rigid_um': float(first_rigid[selected]),
                'fitted_selected_corrected_rigid_um': float(final_rigid[selected]),
                'fitted_selected_initial_five_point_pose_um': float(initial_five[selected]),
                'fitted_selected_first_pass_five_point_pose_um': float(first_five[selected]),
                'fitted_selected_corrected_five_point_pose_um': float(final_five[selected]),
                'fitted_selected_first_pass_mapped64_um': float(mapped64_error[selected]),
                'fitted_selected_first_pass_warp64_magnitude_um': float(warp64_magnitude[selected]),
                'fitted_selected_mapped96_um': float(mapped96_error[selected]),
                'fitted_selected_warp96_magnitude_um': float(warp96_magnitude[selected]),
                'prior_selected_mapped96_um': float(mapped96_error[prior_selected]),
                'best14_initial_rigid_um': float(initial_rigid.min()),
                'best14_corrected_rigid_um': float(final_rigid.min()),
                'best14_mapped96_um': float(mapped96_error[best]),
                'best14_warp96_magnitude_um': float(warp96_magnitude[best]),
                'selected_regret_um': float(mapped96_error[selected] - mapped96_error[best]),
                'truth_best_fitted_rank': int(fitted_rank[best]),
                'truth_best_prior_rank': int(prior_rank[best]),
                'fitted_score_error_spearman': float(spearmanr(score, -mapped96_error).statistic),
                'fitted_score_error_kendall': float(kendalltau(score, -mapped96_error).statistic),
                'prior_score_error_spearman': float(spearmanr(prior_score, -mapped96_error).statistic)}
            for name, order in (('fitted', fitted_order), ('prior', prior_order)):
                for k in (1, 3, 5, 14):
                    section[f'{name}_top{k}_regret_um'] = float(
                        mapped96_error[order[:k]].min() - mapped96_error[best])
            sections.append(section)
            section_stream.write(json.dumps(section, allow_nan=False) + '\n')
        candidate_stream.flush()
        section_stream.flush()
        print(json.dumps({'checkpoint': step, 'sections': len(records)}), flush=True)

metrics = ('fitted_selected_initial_rigid_um', 'fitted_selected_first_pass_rigid_um',
           'fitted_selected_corrected_rigid_um',
           'fitted_selected_initial_five_point_pose_um',
           'fitted_selected_first_pass_five_point_pose_um',
           'fitted_selected_corrected_five_point_pose_um',
           'fitted_selected_first_pass_mapped64_um',
           'fitted_selected_first_pass_warp64_magnitude_um',
           'fitted_selected_mapped96_um', 'fitted_selected_warp96_magnitude_um',
           'prior_selected_mapped96_um', 'best14_initial_rigid_um',
           'best14_corrected_rigid_um', 'best14_mapped96_um',
           'best14_warp96_magnitude_um', 'selected_regret_um',
           'truth_best_fitted_rank', 'truth_best_prior_rank',
           'fitted_score_error_spearman', 'fitted_score_error_kendall',
           'prior_score_error_spearman', 'updated_prior_beam_overlap_count',
           *(f'{name}_top{k}_regret_um' for name in ('fitted', 'prior')
             for k in (1, 3, 5, 14)))
summary = {}
for step in steps:
    group = [row for row in sections if row['step'] == step]
    groups = {'all': group}
    groups.update({f'appearance_{appearance}': [row for row in group
        if row['appearance_mode'] == appearance]
        for appearance in ('exact_black', 'raw', 'imperfect_brush')})
    summary[str(step)] = {'groups': {}}
    for name, subset in groups.items():
        plans = sorted({row['synthetic_subject_plan_id'] for row in subset})
        assert plans
        by_plan = {plan: {'sections': sum(row['synthetic_subject_plan_id'] == plan for row in subset),
            **{metric: float(np.mean([row[metric] for row in subset
                if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}}
            for plan in plans}
        summary[str(step)]['groups'][name] = {'sections': len(subset), 'plans': len(plans),
            'by_plan': by_plan,
            'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric] for plan in plans]))
                                for metric in metrics}}
    assert summary[str(step)]['groups']['all']['plans'] == 8

baseline = rank_summary['groups']['all']['plan_equal_mean']
at_zero = summary['0']['groups']['all']['plan_equal_mean']
assert abs(at_zero['fitted_selected_mapped96_um'] - baseline['fitted_selected_mapped96_um']) <= 1.
assert abs(at_zero['best14_mapped96_um'] - baseline['truth_best14_mapped96_um']) <= 1.
for step in steps:
    result = summary[str(step)]
    current = result['groups']['all']['plan_equal_mean']
    result['delta_from_086_um'] = {
        'fitted_selected_mapped96': current['fitted_selected_mapped96_um']
                                     - baseline['fitted_selected_mapped96_um'],
        'best14_mapped96': current['best14_mapped96_um']
                            - baseline['truth_best14_mapped96_um']}
    result['appearance_delta_from_086_um'] = {
        appearance: result['groups'][f'appearance_{appearance}']['plan_equal_mean'][
            'fitted_selected_mapped96_um']
        - rank_summary['groups'][f'appearance_{appearance}']['plan_equal_mean'][
            'fitted_selected_mapped96_um']
        for appearance in ('exact_black', 'raw', 'imperfect_brush')}
    result['prespecified_physical_gate'] = {
        'selected_gain_ge_300um': result['delta_from_086_um']['fitted_selected_mapped96'] <= -300.,
        'best14_not_worse': result['delta_from_086_um']['best14_mapped96'] <= 0.}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
assert len(sections) == len(records) * len(steps) == 984
assert candidate_count == len(sections) * 14
(out / 'completed.json').write_text(json.dumps({
    'sections': len(sections), 'candidates': candidate_count,
    'training_draw_attempts': attempts, 'accepted_synthetic': accepted,
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
