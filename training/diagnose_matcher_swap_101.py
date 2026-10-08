"""Matched 094 full-fit diagnostic with only the 099 matcher checkpoint swapped."""
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.contextual_matcher_099 import WholeSliceAtlasFeedback099
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent_file = parent_run / 'joint_step_20000.pt'
matcher_run = root / 'runs/contextual_match_099_pilot'
panel = root / 'data/contextual_match_099_fresh_synthetic_dev_panel_001'
out = root / 'runs/matcher_swap_101_development_diagnostic'
protocol = source.parent / 'docs/publication/MATCHER_SWAP_101_PROTOCOL_20261008.md'
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


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent_file) == parent_receipt['checkpoint_sha256']['20000'] == \
    '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert all(sha(source / name) == digest for name, digest in
           parent_config['source_sha256'].items())
train_receipt = json.loads((matcher_run / 'completed.json').read_text())
train_config = json.loads((matcher_run / 'config.json').read_text())
assert train_receipt['batches'] == train_receipt['accepted_synthetic'] == 4000
assert train_config['protocol_sha256'] == train_receipt['protocol_sha256'] == sha(
    source.parent / 'docs/publication/CONTEXTUAL_MATCH_099_PROTOCOL_20261008.md')
assert train_receipt['source_sha256'] == train_config['source_sha256']
assert all(sha(source / name) == digest for name, digest in
           train_config['source_sha256'].items())
assert sha(matcher_run / 'config.json') == train_receipt['config_sha256']
assert sha(matcher_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(matcher_run / 'training.jsonl') == train_receipt['training_sha256']
matcher_files = {step: matcher_run / f'contextual_step_{step:05d}.pt'
                 for step in steps}
assert all(sha(matcher_files[step]) == train_receipt['checkpoint_sha256'][str(step)]
           for step in steps)
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_receipt['physical_sections'] == 256
assert panel_receipt['protocol_sha256'] == sha(panel / 'protocol.json')
assert panel_receipt['records_sha256'] == sha(panel / 'records.jsonl')
assert panel_protocol['synthetic_subjects'] == 8
assert all(sha(panel / 'source' / name) == digest for name, digest in
           panel_protocol['source_sha256'].items())
all_records = list(map(json.loads, (panel / 'records.jsonl').open()))
assert len(all_records) == 256
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in all_records})
assert len(plan_ids) == 8
records = [row for plan in plan_ids for row in sorted(
    (item for item in all_records if item['synthetic_subject_plan_id'] == plan
     and item['eligible']), key=lambda item: item['section_index'])[:8]]
assert len(records) == 64 and all(sum(row['synthetic_subject_plan_id'] == plan
    for row in records) == 8 for plan in plan_ids)
assert len({row['section_id'] for row in records}) == 64
assert all(sha(panel / row['file']) == row['sha256'] for row in records)
assert not any(train_receipt[key] or train_config[key] for key in
    ('calibrated', 'public_benchmark_used', 'real_labels_used',
     'external_pretrained_weights_used'))

saved = torch.load(parent_file, map_location='cpu', weights_only=True)
assert saved['step'] == 20000 and not saved['calibrated']
matcher_states = {}
for step in steps:
    checkpoint = torch.load(matcher_files[step], map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['config'] == train_config
    assert not checkpoint['calibrated']
    matcher_states[step] = checkpoint['matcher']
base_matcher = {key.removeprefix('matcher.'): value for key, value in
    saved['spatial_fit'].items() if key.startswith('matcher.')}
fit_head = {key: value for key, value in saved['spatial_fit'].items()
            if not key.startswith('matcher.')}
for key, value in base_matcher.items():
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
spatial = SpatialJointFit094(WholeSliceAtlasFeedback099()).cuda().eval().requires_grad_(False)
parent_spatial = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
parent_spatial.load_state_dict(saved['spatial_fit'], strict=True)
del saved

config = {'source_sha256': {name: sha(source / name) for name in (
    'diagnose_matcher_swap_101.py', 'evaluate_spatial_verifier_094.py',
    'evaluate_contextual_match_099.py', 'spatial_joint_fit_094.py',
    'contextual_matcher_099.py', 'whole_slice_atlas_feedback_083.py',
    'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_allen_atlas_binding_v6.py')},
    'protocol_sha256': sha(protocol),
    'parent_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_checkpoint_sha256': sha(parent_file),
    'matcher_training_completed_sha256': sha(matcher_run / 'completed.json'),
    'matcher_checkpoint_sha256': {str(step): sha(matcher_files[step])
                                  for step in steps},
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'selection': 'first eight eligible section indices per each of eight independent synthetic DEV plans; frozen before either matcher arm is evaluated',
    'section_ids': [row['section_id'] for row in records],
    'source_files_sha256': {row['file']: row['sha256'] for row in records},
    'fixed_model': '094 step20000 predictor, encoder, mapper, fitted-scoring head and Atlas volume',
    'only_swapped_component': '099 whole-slice matcher step0 versus step4000',
    'beam': 'same predictor prior top eight old branch IDs 0:31 and top six anchor branch IDs 32:63, all fourteen processed at both arms',
    'score': 'prior - fit_energy / clipped_exp(log_fit_temperature) - 0.05 * warp_cost',
    'mapped96': 'mean CCF distance over 1024 deterministic valid source pixels; same at_chart sampling as 094 and 099 evaluators',
    'normal_error_deg': 'unsigned plane-normal angle acos(abs(dot(predicted, truth)))',
    'angle_family': 'nearest cardinal normal axis, coronal/AP; horizontal/DV; sagittal/ML; steep means >=30 degrees from nearest cardinal',
    'scope': 'matched synthetic development diagnostic, not model selection, calibration, biological validation or public benchmark',
    'calibrated': False, 'public_benchmark_used': False,
    'real_labels_used': False},
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

candidate_rows, section_rows = [], []
parity = {}
with torch.inference_mode():
    for step in steps:
        spatial.load_state_dict({**fit_head, **{f'matcher.{key}': value
            for key, value in matcher_states[step].items()}}, strict=True)
        temperature = spatial.log_fit_temperature.clamp(
            math.log(.5), math.log(5.)).exp()
        for record_index, record in enumerate(records):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            chosen = valid.flatten().nonzero().flatten()
            chosen = chosen[torch.linspace(0, len(chosen) - 1, 1024,
                device='cuda').round().long()]
            sample_hash = hashlib.sha256(
                chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
            chart = torch.stack((chosen.remainder(256),
                chosen.div(256, rounding_mode='floor')), -1).float() / 256
            reference = truth.reshape(-1, 3)[chosen]
            truth_centre, truth_frame, _ = full_frame_state_to_components(
                truth_state[:, None])
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            initial = prediction['state'].gather(1,
                (beam // 2)[..., None].expand(-1, -1, 12))
            normal = np.abs(np.asarray(record['plane_normal_ap_dv_ml'], dtype=float))
            family = ('coronal', 'horizontal', 'sagittal')[int(normal.argmax())]
            angle = float(np.rad2deg(np.arccos(normal.max().clip(0, 1))))
            common = {key: record[key] for key in (
                'section_id', 'animal_id', 'specimen_id', 'experiment_id',
                'synthetic_animal_id', 'synthetic_subject_plan_id',
                'synthetic_subject_realization_id', 'panel_physical_section_id',
                'appearance_mode', 'section_index')}
            common.update({'step': step, 'source_file': record['file'],
                'source_sha256': record['sha256'],
                'sample_indices_sha256': sample_hash,
                'angle_family': family, 'angle_from_family_axis_deg': angle,
                'obliquity': 'steep_>=30deg' if angle >= 30 else 'near_<30deg'})
            rows = []
            for start in range(0, 14, 2):
                branch = beam[:, start:start + 2]
                reflected = branch % 2
                state = initial[:, start:start + 2]
                fitted = spatial(prediction['feature'], state, reflected,
                    atlas, offsets, weights, source_shape=(256, 256))
                selected = {**prediction, 'state': fitted['state'],
                    'log_mass': prediction['log_mass'].gather(1, branch // 2),
                    'reflection_logit': prediction['reflection_logit'].gather(
                        1, branch // 2)}
                mapped = model.map(selected, offsets,
                    torch.arange(2, device='cuda')[None], reflected, (96, 96),
                    atlas, weights, feature_side=96, source_shape=(256, 256),
                    spatial_evidence=fitted['spatial_evidence'])
                local = mapped['local_displacement_um'] / 1000
                dx = local[..., 1:] - local[..., :-1]
                dy = local[..., 1:, :] - local[..., :-1, :]
                warp = local.square().mean((-3, -2, -1)) + .1 * (
                    dx.square().mean((-3, -2, -1)) +
                    dy.square().mean((-3, -2, -1)))
                score = prior.gather(1, branch) - fitted['fit_energy'] / temperature - .05 * warp
                error = (at_chart(mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                         - reference[None]).norm(dim=-1).mean(-1)
                centre, frame, _ = full_frame_state_to_components(fitted['state'])
                centre_error = (centre - truth_centre).norm(dim=-1)
                normal_error = torch.rad2deg(torch.acos((frame[..., :, 2] *
                    truth_frame[..., :, 2]).sum(-1).abs().clamp(0, 1)))
                support = (fitted['fine_match_support'] >= .5).any(2).float().mean((-2, -1))
                if step == 0 and record_index == 0 and start == 0:
                    expected = parent_spatial(prediction['feature'], state,
                        reflected, atlas, offsets, weights, source_shape=(256, 256))
                    for key in ('state', 'fit_energy', 'coarse_match_logits',
                                'coarse_match_support', 'fine_match_logits',
                                'fine_match_support', 'spatial_evidence',
                                'quality', 'coverage', 'pose_correction_cost'):
                        parity[key] = float((fitted[key] - expected[key]).abs().max())
                    expected_selected = {**selected, 'state': expected['state']}
                    expected_mapped = model.map(expected_selected, offsets,
                        torch.arange(2, device='cuda')[None], reflected, (96, 96),
                        atlas, weights, feature_side=96, source_shape=(256, 256),
                        spatial_evidence=expected['spatial_evidence'])
                    parity['mapped_ccf_um'] = float((mapped[
                        'centre_surface_ccf_ap_dv_ml_um'] - expected_mapped[
                        'centre_surface_ccf_ap_dv_ml_um']).abs().max())
                    expected_local = expected_mapped['local_displacement_um'] / 1000
                    expected_dx = expected_local[..., 1:] - expected_local[..., :-1]
                    expected_dy = expected_local[..., 1:, :] - expected_local[..., :-1, :]
                    expected_warp = expected_local.square().mean((-3, -2, -1)) + .1 * (
                        expected_dx.square().mean((-3, -2, -1)) +
                        expected_dy.square().mean((-3, -2, -1)))
                    expected_score = (prior.gather(1, branch) -
                        expected['fit_energy'] / temperature - .05 * expected_warp)
                    parity['selection_score'] = float((score - expected_score).abs().max())
                    assert max(parity.values()) < 1e-3
                for local_slot in range(2):
                    slot = start + local_slot
                    rows.append({**common, 'beam_slot': slot,
                        'branch_id': int(branch[0, local_slot]),
                        'initial_state': state[0, local_slot].tolist(),
                        'fitted_state': fitted['state'][0, local_slot].tolist(),
                        'prior_score': float(prior[0, branch[0, local_slot]]),
                        'score': float(score[0, local_slot]),
                        'mapped96_error_um': float(error[local_slot]),
                        'normal_error_deg': float(normal_error[0, local_slot]),
                        'centre_error_um': float(centre_error[0, local_slot]),
                        'fit_energy': float(fitted['fit_energy'][0, local_slot]),
                        'warp_cost': float(warp[0, local_slot]),
                        'fine_support_fraction': float(support[0, local_slot]),
                        'quality': float(fitted['quality'][0, local_slot]),
                        'coverage': float(fitted['coverage'][0, local_slot]),
                        'pose_correction_cost': float(fitted[
                            'pose_correction_cost'][0, local_slot])})
            assert len(rows) == 14 and [row['beam_slot'] for row in rows] == list(range(14))
            selected_slot = max(range(14), key=lambda slot: rows[slot]['score'])
            best_slot = min(range(14), key=lambda slot: rows[slot]['mapped96_error_um'])
            for row in rows:
                row['score_selected'] = row['beam_slot'] == selected_slot
                row['physical_best'] = row['beam_slot'] == best_slot
            candidate_rows.extend(rows)
            section_rows.append({**common, 'beam_branch_ids': beam[0].tolist(),
                'selected_slot': selected_slot, 'selected_branch_id': rows[selected_slot]['branch_id'],
                'best_slot': best_slot, 'best_branch_id': rows[best_slot]['branch_id'],
                **{f'selected_{key}': rows[selected_slot][key] for key in (
                    'mapped96_error_um', 'normal_error_deg', 'centre_error_um',
                    'fit_energy', 'warp_cost', 'fine_support_fraction', 'quality',
                    'coverage', 'pose_correction_cost')},
                'best_mapped96_error_um': rows[best_slot]['mapped96_error_um'],
                'selected_regret_um': rows[selected_slot]['mapped96_error_um'] -
                    rows[best_slot]['mapped96_error_um']})
            if (record_index + 1) % 16 == 0:
                print(json.dumps({'step': step, 'sections': record_index + 1}), flush=True)

assert len(section_rows) == 128 and len(candidate_rows) == 1792
paired = {(row['section_id'], row['step']): row for row in section_rows}
candidate_pairs = {(row['section_id'], row['beam_slot'], row['step']): row
                   for row in candidate_rows}
for record in records:
    zero = paired[(record['section_id'], 0)]
    final = paired[(record['section_id'], 4000)]
    assert zero['beam_branch_ids'] == final['beam_branch_ids']
    assert zero['source_sha256'] == final['source_sha256']
    assert zero['sample_indices_sha256'] == final['sample_indices_sha256']
    for slot in range(14):
        initial = candidate_pairs[(record['section_id'], slot, 0)]
        updated = candidate_pairs[(record['section_id'], slot, 4000)]
        assert initial['initial_state'] == updated['initial_state']
        assert initial['prior_score'] == updated['prior_score']
metrics = ('selected_mapped96_error_um', 'best_mapped96_error_um',
    'selected_regret_um', 'selected_normal_error_deg', 'selected_centre_error_um',
    'selected_fit_energy', 'selected_warp_cost', 'selected_fine_support_fraction',
    'selected_quality', 'selected_coverage', 'selected_pose_correction_cost')
summary = {'sections_per_arm': 64, 'candidates_per_arm': 896,
    'first_section_step0_094_parity_max_abs': parity, 'arms': {}}
for step in steps:
    arm = [row for row in section_rows if row['step'] == step]
    groups = {'all': arm}
    groups.update({f'plan/{plan}': [row for row in arm
        if row['synthetic_subject_plan_id'] == plan] for plan in plan_ids})
    for family in ('coronal', 'horizontal', 'sagittal'):
        groups[f'angle_family/{family}'] = [row for row in arm
            if row['angle_family'] == family]
        for obliquity in ('near_<30deg', 'steep_>=30deg'):
            groups[f'angle_family/{family}/{obliquity}'] = [row for row in arm
                if row['angle_family'] == family and row['obliquity'] == obliquity]
    summary['arms'][str(step)] = {name: {'sections': len(group),
        **{key: float(np.mean([row[key] for row in group])) for key in metrics}}
        for name, group in groups.items() if group}
summary['final_minus_initial'] = {name: {
    key: summary['arms']['4000'][name][key] - summary['arms']['0'][name][key]
    for key in metrics} for name in summary['arms']['0']}
for name, rows in (('candidates.jsonl', candidate_rows),
                   ('sections.jsonl', section_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections_per_arm': 64, 'candidates_per_arm': 896,
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections_per_arm': 64,
    'first_section_parity': parity,
    'mapped96_delta_um': summary['final_minus_initial']['all'][
        'selected_mapped96_error_um']}), flush=True)
