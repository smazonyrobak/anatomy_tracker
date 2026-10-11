"""Sealed TRAIN-inner 64-section readout of learned coherent 3-D matching."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.local_3d_descriptor_240 import (
    Local3DDescriptor240, coherent_score_240, query_chart_240,
    refine_candidate_240, render_candidate_bank_240, sample_candidate_features_240,
)
from training.support_aware_framing_217 import sample_support_aware_framing_217

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = Path(__file__).resolve().parent
run = root / 'runs/learned_coherent_3d_match_240_readout_001'
training_run = root / 'runs/learned_coherent_3d_match_240_train_001'
protocol = source.parent / 'docs/publication/LEARNED_COHERENT_3D_MATCH_240_PROTOCOL_20261012.md'
parent = root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt'
checkpoint = training_run / 'best.pt'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mean(values):
    return float(np.mean(values)) if values else None


# No readout image is drawn, and no checkpoint is loaded, until training has exited
# and its saved checkpoint and code binding agree with the training receipt.
receipt = json.loads((training_run / 'completed.json').read_text(encoding='utf-8'))
config = json.loads((training_run / 'config.json').read_text(encoding='utf-8'))
assert receipt['training_exited'] and receipt['best_step'] > 0
assert sha(checkpoint) == receipt['checkpoint_sha256']
assert sha(training_run / 'config.json') == receipt['output_sha256']['config.json']
assert sha(protocol) == config['protocol_sha256']
assert sha(parent) == config['parent_sha256'] == 'dd76016764b1929a883f800f435e45e81fc503d3ee6bb76d66674e7c8e3e3fbc'
assert all(sha(source / name) == digest for name, digest in config['source_sha256'].items())
saved_descriptor = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved_descriptor['step'] == receipt['best_step']
assert saved_descriptor['parent_sha256'] == config['parent_sha256']

context = load_streaming_synthetic_v7_64(device='cuda')
saved_parent = torch.load(parent, map_location='cpu', weights_only=True)
proposer = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(proposer, enabled=True)
proposer.load_state_dict(saved_parent['model'], strict=True)
descriptor = Local3DDescriptor240().cuda().eval().requires_grad_(False)
descriptor.load_state_dict(saved_descriptor['descriptor'], strict=True)
del saved_parent, saved_descriptor


def correspondence_mm(source_grid, atlas_grid, support, state, reflection, truth, valid):
    query = source_grid[:, :, ::2, ::2].flatten(2).transpose(1, 2)[0]
    keys = atlas_grid.flatten(2)[0]
    similarity = (query @ keys).masked_fill(support.flatten()[None] <= .5, -1e4)
    if not bool((support > .5).any()):
        return None
    index = similarity.argmax(-1)
    depth = index.div(44 * 44, rounding_mode='floor')
    y = index.remainder(44 * 44).div(44, rounding_mode='floor')
    x = index.remainder(44)
    axis = (torch.arange(44, device='cuda', dtype=state.dtype) - 5.5) / 32 - .5 / 256
    chart = torch.stack((axis[x], axis[y]), -1)
    normal = full_frame_state_to_components(state)[1][0, :, 2]
    estimated = rigid_points_090(state, reflection, chart)[0] + (depth - 8)[:, None] * 500 * normal
    error = (estimated - truth[0]).norm(dim=-1) / 1000
    return float(error[valid[0]].mean()) if bool(valid.any()) else None


def score_candidate(source_grid, state, reflection, null_seed=None, truth=None, valid=None):
    with torch.no_grad():
        bank, support = render_candidate_bank_240(context['atlas'], state, reflection,
            sample['offsets'], sample['weights'], null_seed=null_seed)
        atlas_grid = descriptor.encode_atlas(bank)
        initial, count = coherent_score_240(source_grid, atlas_grid, support,
            state, reflection, state)
        reach = None
        if truth is not None:
            _, available = sample_candidate_features_240(atlas_grid, support,
                state, reflection, truth)
            reach = (available > .5) & valid
        return atlas_grid, support, float(initial[0]), int(count[0]), reach


def common_site_score(source_grid, atlas_grid, support, state, reflection, common):
    world = rigid_points_090(state, reflection, query_chart)
    keys, _ = sample_candidate_features_240(atlas_grid, support, state, reflection, world)
    query = source_grid[:, :, ::2, ::2].flatten(2).transpose(1, 2)
    return float((query * keys).sum(-1).masked_fill(~common, -1.)
                 .topk(64, dim=-1).values.mean())


rows = []
draws = []
rng = np.random.default_rng(20261012242)
query_chart = query_chart_240('cuda', torch.float32)
query_yx = ((query_chart[:, [1, 0]] * 256).round().long()).clamp(0, 255)
query_pixels = query_yx[:, 0] * 256 + query_yx[:, 1]
run.mkdir(parents=True, exist_ok=False)
with (run / 'rows.jsonl').open('w', encoding='utf-8') as row_file, \
        (run / 'draws.jsonl').open('w', encoding='utf-8') as draw_file:
    for base in (52, 53, 54, 55):
        for appearance, sampler in (('v4', sample_one_shot_slide_artifacts_v4),
                                    ('v7', sample_one_shot_slide_artifacts_v7)):
            accepted = 0
            while accepted < 8:
                seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    framed = sample_support_aware_framing_217(
                        sample_panel_geometry_train_166(context, [base], seed, side=256), seed)
                    sample = sampler(context, [base], seed, side=256, source_sample=framed)
                eligible = bool(sample['eligible'][0])
                draw = {'base': base, 'appearance': appearance, 'seed': seed,
                        'used': eligible, **sample['provenance'][0]}
                draws.append(draw)
                draw_file.write(json.dumps(draw, allow_nan=False) + '\n')
                if not eligible:
                    continue
                accepted += 1
                physical_id = sample['provenance'][0]['physical_section_id']
                pixels = sample['valid_mask'][0].flatten().nonzero().flatten()
                pixels = pixels[torch.linspace(0, len(pixels) - 1,
                    min(128, len(pixels)), device='cuda').long()]
                sites = torch.stack((pixels.remainder(256),
                    pixels.div(256, rounding_mode='floor')), -1).float() / 256
                truth = sample['centre'][0].reshape(-1, 3)[pixels]
                query_truth = sample['centre'][0].reshape(-1, 3)[query_pixels][None]
                query_valid = sample['valid_mask'][0].flatten()[query_pixels][None]
                with torch.no_grad():
                    prediction = proposer.predict(sample['inputs'])
                    states = prediction['state'][0].repeat_interleave(2, 0)
                    reflections = torch.arange(2, device='cuda').repeat(proposer.modes)
                    prior = (prediction['log_mass'][0, :, None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit'][0]),
                        F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                    ordered = prior.argsort(descending=True)
                    initial_error = (rigid_points_090(states, reflections, sites)
                        - truth[None]).norm(dim=-1).mean(-1) / 1000
                    source_grid = descriptor.encode_source(sample['inputs'][:, :1])
                    interior = F.avg_pool2d((sample['visible'][:, None] > .5).float(),
                        kernel_size=7, stride=1, padding=3) > .999
                    swapped = sample['inputs'][:, :1].clone()
                    values = swapped[interior].clone()
                    swapped[interior] = values.roll(len(values) // 2)
                    swapped_grid = descriptor.encode_source(swapped)

                near_choices = ordered[:16][initial_error[ordered[:16]] < 2]
                near = int(near_choices[initial_error[near_choices].argmin()]) if len(near_choices) else None
                candidate_rows = []
                union_reach = torch.zeros_like(query_valid)
                for rank, branch in enumerate(ordered, 1):
                    index = int(branch)
                    state = states[index:index + 1]
                    reflection = reflections[index:index + 1]
                    atlas_grid, support, score, count, reach = score_candidate(
                        source_grid, state, reflection, truth=query_truth, valid=query_valid)
                    union_reach |= reach
                    fit, diagnostics = refine_candidate_240(source_grid, atlas_grid,
                        support, state, reflection)
                    fitted_error = float((rigid_points_090(fit, reflection, sites)[0]
                        - truth).norm(dim=-1).mean() / 1000)
                    candidate_rows.append({'branch': index, 'prior_rank': rank,
                        'reflection': int(reflection[0]), 'prior_logit': float(prior[index]),
                        'initial_mm': float(initial_error[index]), 'fitted_mm': fitted_error,
                        'score': score, 'supported_sites': count,
                        'selected_score': diagnostics['final_score'] if diagnostics['accepted'] else score,
                        'true_query_sites_reached': int(reach.sum()), **diagnostics})
                    del atlas_grid, support, fit
                selected = max(candidate_rows, key=lambda item: item['selected_score'])
                direct = candidate_rows[0]
                best_initial = min(item['initial_mm'] for item in candidate_rows)
                best_fitted = min(item['fitted_mm'] for item in candidate_rows)
                row = {'case_index': len(rows), 'base': base, 'appearance': appearance,
                    'seed': seed, 'physical_section_id': physical_id,
                    'provenance': sample['provenance'][0],
                    'input_sha256': hashlib.sha256(sample['inputs'].cpu().numpy().tobytes()).hexdigest(),
                    'true_reflection': int(sample['reflection'][0]),
                    'visible_tissue_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'valid_query_sites': int(query_valid.sum()),
                    'true_query_sites_inside_any_bank': int(union_reach.sum()),
                    'true_query_sites_outside_every_bank': int((query_valid & ~union_reach).sum()),
                    'best_initial_mm': best_initial, 'best_fitted_mm': best_fitted,
                    'direct_branch': direct['branch'], 'direct_mm': direct['initial_mm'],
                    'direct_reflection': direct['reflection'],
                    'blind_selected_branch': selected['branch'],
                    'blind_selected_mm': selected['fitted_mm'],
                    'blind_selected_reflection': selected['reflection'],
                    'near_branch': near, 'near': None, 'ranking': None,
                    'true_gauge': None, 'candidates': candidate_rows}

                if near is not None:
                    state, reflection = states[near:near + 1], reflections[near:near + 1]
                    null_seed = int.from_bytes(hashlib.sha256(
                        f'240|atlas-null|{physical_id}|{near}'.encode()).digest()[:8], 'little') % (2**63)
                    null_grid, null_support, null_score, null_count, _ = score_candidate(
                        source_grid, state, reflection, null_seed=null_seed)
                    null_fit, null_diagnostics = refine_candidate_240(source_grid,
                        null_grid, null_support, state, reflection)
                    null_error = float((rigid_points_090(null_fit, reflection, sites)[0]
                        - truth).norm(dim=-1).mean() / 1000)
                    near_result = candidate_rows[next(i for i, item in enumerate(candidate_rows)
                        if item['branch'] == near)]
                    row['near'] = {'initial_mm': near_result['initial_mm'],
                        'intact_fitted_mm': near_result['fitted_mm'],
                        'null_fitted_mm': null_error,
                        'reflection': int(reflection[0]),
                        'intact_correspondence_mm': None,
                        'null_score': null_score, 'null_supported_sites': null_count,
                        'null_diagnostics': null_diagnostics}
                    del null_grid, null_support, null_fit
                    near_grid, near_support, _, _, _ = score_candidate(
                        source_grid, state, reflection)
                    row['near']['intact_correspondence_mm'] = correspondence_mm(
                        source_grid, near_grid, near_support, state, reflection,
                        query_truth, query_valid)
                    del near_grid, near_support

                    near_count = near_result['supported_sites']
                    wrong = next((item for item in candidate_rows[:16]
                        if item['branch'] != near
                        and item['initial_mm'] >= near_result['initial_mm'] + .5
                        and min(near_count, item['supported_sites']) >= 32
                        and abs(near_count - item['supported_sites']) <= 32), None)
                    if wrong is not None and near_count >= 32:
                        paired = {}
                        for role, index in (('near', near), ('wrong', wrong['branch'])):
                            state, reflection = states[index:index + 1], reflections[index:index + 1]
                            bank_grid, bank_support, _, _, _ = score_candidate(
                                source_grid, state, reflection)
                            world = rigid_points_090(state, reflection, query_chart)
                            _, available = sample_candidate_features_240(bank_grid,
                                bank_support, state, reflection, world)
                            null_seed = int.from_bytes(hashlib.sha256(
                                f'240|atlas-null|{physical_id}|{index}'.encode()).digest()[:8], 'little') % (2**63)
                            null_grid, null_support, _, _, _ = score_candidate(
                                source_grid, state, reflection, null_seed=null_seed)
                            paired[role] = (state, reflection, bank_grid, bank_support,
                                null_grid, null_support, available > .5)
                        common = paired['near'][-1] & paired['wrong'][-1]
                        if int(common.sum()) >= 32:
                            ranking = {'near_branch': near, 'wrong_branch': wrong['branch'],
                                'near_mm': near_result['initial_mm'],
                                'wrong_mm': wrong['initial_mm'],
                                'near_supported_sites': near_count,
                                'wrong_supported_sites': wrong['supported_sites'],
                                'common_supported_sites': int(common.sum())}
                            for role in ('near', 'wrong'):
                                state, reflection, bank_grid, bank_support, null_grid, null_support, _ = paired[role]
                                for arm, source_feature, atlas_feature, atlas_support in (
                                    ('intact', source_grid, bank_grid, bank_support),
                                    ('swap', swapped_grid, bank_grid, bank_support),
                                    ('null', source_grid, null_grid, null_support)):
                                    ranking[f'{arm}_{role}_score'] = common_site_score(
                                        source_feature, atlas_feature, atlas_support,
                                        state, reflection, common)
                            row['ranking'] = ranking
                        del paired

                state, reflection = sample['state'], sample['reflection']
                atlas_grid, support, true_score, true_count, _ = score_candidate(
                    source_grid, state, reflection)
                true_correspondence = correspondence_mm(source_grid, atlas_grid,
                    support, state, reflection, query_truth, query_valid)
                true_fit, diagnostics = refine_candidate_240(source_grid, atlas_grid,
                    support, state, reflection)
                row['true_gauge'] = {'initial_mm': float((rigid_points_090(state,
                    reflection, sites)[0] - truth).norm(dim=-1).mean() / 1000),
                    'fitted_mm': float((rigid_points_090(true_fit, reflection, sites)[0]
                    - truth).norm(dim=-1).mean() / 1000),
                    'correspondence_mm': true_correspondence,
                    'score': true_score, 'supported_sites': true_count,
                    **diagnostics}
                rows.append(row)
                row_file.write(json.dumps(row, allow_nan=False) + '\n')
                row_file.flush(); draw_file.flush()
                if len(rows) % 8 == 0:
                    print(json.dumps({'event': 'cases', 'count': len(rows)}), flush=True)

summary = {}
for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    near_group = [r for r in group if r['near'] is not None]
    pairs = [r['ranking'] for r in group if r['ranking'] is not None]
    summary[appearance] = {'cases': len(group), 'near_top16_cases': len(near_group),
        'matched_support_pairs': len(pairs),
        'near_initial_mean_mm': mean([r['near']['initial_mm'] for r in near_group]),
        'near_intact_fitted_mean_mm': mean([r['near']['intact_fitted_mm'] for r in near_group]),
        'near_null_fitted_mean_mm': mean([r['near']['null_fitted_mm'] for r in near_group]),
        'near_correspondence_mean_mm': mean([r['near']['intact_correspondence_mm']
            for r in near_group if r['near']['intact_correspondence_mm'] is not None]),
        'true_initial_mean_mm': mean([r['true_gauge']['initial_mm'] for r in group]),
        'true_fitted_mean_mm': mean([r['true_gauge']['fitted_mm'] for r in group]),
        'true_correspondence_mean_mm': mean([r['true_gauge']['correspondence_mm']
            for r in group if r['true_gauge']['correspondence_mm'] is not None]),
        'direct_mean_mm': mean([r['direct_mm'] for r in group]),
        'blind_selected_mean_mm': mean([r['blind_selected_mm'] for r in group]),
        'initial_all160_coverage_lt2mm': sum(r['best_initial_mm'] < 2 for r in group),
        'corrected_all160_coverage_lt2mm': sum(r['best_fitted_mm'] < 2 for r in group),
        'no_valid_query_cases': sum(r['valid_query_sites'] == 0 for r in group),
        'outside_every_bank_cases': sum(r['valid_query_sites'] > 0
            and r['true_query_sites_inside_any_bank'] == 0 for r in group),
        'mean_true_query_sites_outside_every_bank_fraction': mean([
            r['true_query_sites_outside_every_bank'] / r['valid_query_sites']
            for r in group if r['valid_query_sites'] > 0]),
        'near_reflection_correct': sum(r['near']['reflection'] == r['true_reflection']
            for r in near_group),
        'direct_reflection_correct': sum(r['direct_reflection'] == r['true_reflection']
            for r in group),
        'blind_selected_reflection_correct': sum(
            r['blind_selected_reflection'] == r['true_reflection'] for r in group),
        'ranking_intact_win_fraction': mean([float(p['intact_near_score'] > p['intact_wrong_score'])
            for p in pairs]),
        'ranking_null_win_fraction': mean([float(p['null_near_score'] > p['null_wrong_score'])
            for p in pairs]),
        'ranking_swap_win_fraction': mean([float(p['swap_near_score'] > p['swap_wrong_score'])
            for p in pairs])}

result = summary['all']
summary['stage_b_pass'] = bool(result['matched_support_pairs'] and result['near_top16_cases']
    and result['ranking_intact_win_fraction'] >= .75
    and result['ranking_intact_win_fraction'] - result['ranking_null_win_fraction'] >= .15
    and result['ranking_intact_win_fraction'] - result['ranking_swap_win_fraction'] >= .15
    and result['near_initial_mean_mm'] - result['near_intact_fitted_mean_mm'] >= .25
    and result['near_null_fitted_mean_mm'] - result['near_intact_fitted_mean_mm'] >= .15
    and result['true_fitted_mean_mm'] <= .20
    and result['direct_mean_mm'] - result['blind_selected_mean_mm'] >= .30
    and result['corrected_all160_coverage_lt2mm'] >= 60)
assert len(rows) == 64 and sum(r['appearance'] == 'v4' for r in rows) == 32
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
names = ('arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_panel_geometry_train_166.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py',
    'local_3d_descriptor_240.py', 'support_aware_framing_217.py',
    'arbitrary_plane_geometry.py')
(run / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'draw_attempts': len(draws), 'training_updates': 0,
    'train_best_step': receipt['best_step'],
    'train_best_validation_loss': receipt['best_validation_loss'],
    'external_pretrained_weights_used': False, 'reserved_animal_used': False,
    'public_benchmark_used': False, 'calibrated': False,
    'parent_sha256': sha(parent), 'descriptor_checkpoint_sha256': sha(checkpoint),
    'training_receipt_sha256': sha(training_run / 'completed.json'),
    'training_config_sha256': sha(training_run / 'config.json'),
    'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in names},
    'synthetic_provenance': context['provenance'],
    'output_sha256': {name: sha(run / name) for name in
        ('draws.jsonl', 'rows.jsonl', 'summary.json')},
}, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'event': 'completed', 'cases': len(rows),
    'stage_b_pass': summary['stage_b_pass'], **summary['all']}), flush=True)
