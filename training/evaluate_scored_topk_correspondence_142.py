"""Frozen blind-16 DEV comparison of the four scored correspondence heads."""

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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_spatial_fit_142 import AtlasSpatialFit142
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/SCORED_TOPK_CORRESPONDENCE_142_PROTOCOL_20261010.md'
run = root / 'runs/scored_topk_correspondence_142'
out = root / 'runs/scored_topk_correspondence_142_dev_eval'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
prior_train = root / 'runs/atlas_correspondence_curriculum_140'
prior_eval = root / 'runs/atlas_correspondence_curriculum_140_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('top1_full_intensity', 'top4_full_intensity',
        'top1_support_only', 'top4_support_only')
conditions = ('standard', 'atlas_intensity_off', 'source_intensity_flat')
roles = ('blind_best_original', 'prior_selected', 'blind_best_corrected')
angle_bins = ('all', '<15', '15-30', '30-45', '>=45')
device, side, step = 'cuda', 256, 1200
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == protocol.drive.upper() == run.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_eval_done = json.loads((parent_eval / 'completed.json').read_text())
prior_train_done = json.loads((prior_train / 'completed.json').read_text())
prior_train_config = json.loads((prior_train / 'config.json').read_text())
prior_eval_done = json.loads((prior_eval / 'completed.json').read_text())
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(parent_dir / f'joint_step_{int(key):05d}.pt') == digest
           for key, digest in parent_done['checkpoint_sha256'].items())
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert sha(source.parent / 'docs/publication/V4_POSE_ADAPTATION_132_PROTOCOL_20261010.md') == parent_config['protocol_sha256']
assert all(sha(parent_eval / name) == digest
           for name, digest in parent_eval_done['output_sha256'].items())
assert sha(source / 'evaluate_v4_pose_adaptation_132.py') == parent_eval_done['evaluator_source_sha256']
assert parent_eval_done['checkpoint_sha256']['2000'] == sha(parent_path)
assert sha(prior_train / 'config.json') == prior_train_done['config_sha256']
assert sha(prior_train / 'draws.jsonl') == prior_train_done['draws_sha256']
assert sha(prior_train / 'training.jsonl') == prior_train_done['training_sha256']
assert all(sha(source / name) == digest for name, digest in prior_train_config['source_sha256'].items())
assert sha(source.parent / 'docs/publication/ATLAS_CORRESPONDENCE_CURRICULUM_140_PROTOCOL_20261010.md') == prior_train_config['protocol_sha256']
assert all(sha(prior_train / arm / f'head_step_{int(key):05d}.pt') == digest
           for arm, files in prior_train_done['checkpoint_sha256'].items()
           for key, digest in files.items())
assert sha(prior_eval / 'config.json') == prior_eval_done['config_sha256']
assert sha(prior_eval / 'rows.jsonl') == prior_eval_done['rows_sha256']
assert sha(prior_eval / 'summary.json') == prior_eval_done['summary_sha256']
assert sha(source / 'evaluate_atlas_correspondence_curriculum_140.py') == prior_eval_done['evaluator_sha256']
assert prior_train_done['parent_checkpoint_sha256'] == sha(parent_path)
assert prior_eval_done['train_completion_sha256'] == sha(prior_train / 'completed.json')
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(run / 'draws.jsonl') == train_done['draws_sha256']
assert sha(run / 'training.jsonl') == train_done['training_sha256']
assert train_config['source_sha256'] == train_done['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert train_config['protocol_sha256'] == train_done['protocol_sha256'] == sha(protocol)
assert train_config['parent_checkpoint_sha256'] == train_done['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['parent_completion_sha256'] == sha(parent_dir / 'completed.json')
assert train_config['parent_config_sha256'] == sha(parent_dir / 'config.json')
assert train_done['updates'] == step and train_done['synthetic_presentations'] == 3600
assert train_done['unique_synthetic_physical_sections'] == 3600
assert train_done['v4_presentations'] == 2400 and train_done['v3_presentations'] == 1200
assert tuple(train_config['arms']) == arms
assert train_config['checkpoints'] == [0, 600, step]
assert train_config['arm_neighborhoods'] == {arm: 4 if arm.startswith('top4') else 1 for arm in arms}
assert train_config['arm_atlas_intensity_enabled'] == {arm: arm.endswith('full_intensity') for arm in arms}
assert not any(done.get(key, False) for done in
    (parent_done, parent_eval_done, prior_train_done, prior_eval_done, train_done)
    for key in ('calibrated', 'public_benchmark_used', 'expert_real_truth_used',
                'final_animals_used', 'external_pretrained_weights_used'))
checkpoints = {arm: {int(key): run / arm / f'head_step_{int(key):05d}.pt'
                     for key in train_done['checkpoint_sha256'][arm]} for arm in arms}
assert all(sha(path) == train_done['checkpoint_sha256'][arm][str(number)]
           for arm, files in checkpoints.items() for number, path in files.items())
assert all(number in checkpoints[arm] for arm in arms for number in (0, 600, step))

parent_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
prior_rows = {}
for line in (prior_eval / 'rows.jsonl').open():
    row = json.loads(line)
    if row['step'] != 4000 or row['role'] not in ('blind_best_original', 'prior_selected'):
        continue
    key = row['cohort'], row['section_id'], row['role']
    if row['arm'] == 'full_intensity':
        assert key not in prior_rows
        prior_rows[key] = row
    else:
        reference = prior_rows[key]
        assert row['branch_id'] == reference['branch_id']
        assert row['original_rigid_mm'] == reference['original_rigid_mm']
assert len(parent_rows) == 491 and len(prior_rows) == 2 * 491

panel_records, panel_done = {}, {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    frozen = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in frozen['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == done['eligible']
    assert len({row['panel_physical_section_id'] for row in records}) == 256
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
    panel_done[cohort] = done
    assert all((cohort, row['section_id']) in parent_rows for row in panel_records[cohort])
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243
assert panel_done['v4']['plan_completed_sha256'] == panel_done['v3']['plan_completed_sha256']
assert not {row['panel_physical_section_id'] for row in panel_records['v4']} & {
    row['panel_physical_section_id'] for row in panel_records['v3']}

atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
del atlas_array, annotation
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved
heads = {}
for arm in arms:
    saved = torch.load(checkpoints[arm][step], map_location='cpu', weights_only=True)
    assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
    head = AtlasSpatialFit142(neighborhoods=4 if arm.startswith('top4') else 1)
    head.load_state_dict(saved['head'], strict=True)
    assert head.gate.weight.abs().max() == 0 and abs(float(head.gate.bias[0]) - 2.65) < 1e-5
    assert head.score.weight.abs().max() == 0 and head.score.bias.abs().max() == 0
    heads[arm] = head.to(device).eval().requires_grad_(False)
    del saved
initial = torch.load(checkpoints[arms[0]][0], map_location='cpu', weights_only=True)['head']
for arm in arms[1:]:
    comparison = torch.load(checkpoints[arm][0], map_location='cpu', weights_only=True)['head']
    assert all(torch.equal(initial[name], comparison[name]) for name in initial)
del initial, comparison

axis = torch.arange(side, device=device) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
fixed = heads[arms[0]].fixed_query_xy
assert all(torch.equal(fixed, head.fixed_query_xy) for head in heads.values())
fixed_x = (fixed[:, 0] * side).long()
fixed_y = (fixed[:, 1] * side).long()


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device=device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


def match_counts(output, truth, valid, slot):
    truth = truth[valid].float()
    counts = {'valid_fixed_sites': len(truth), 'global_supported_le_500_um': 0,
              'local_supported_le_500_um': 0, 'scored_fine_le_500_um': 0,
              'fine_dustbin': 0}
    if len(truth) == 0:
        return counts
    keys = output['coarse_key_world_um'][0, slot].float()
    support = output['coarse_key_support'][0, slot] > .5
    global_distance = torch.cdist(truth[None], keys[None])[0]
    global_distance[:, ~support] = torch.inf
    global_500 = global_distance.amin(-1) <= 500
    fine_logits = output['fine_logits'][0, slot, valid].float()
    fine_world = output['fine_key_world_um'][0, slot, valid].float()
    fine_support = output['fine_key_support'][0, slot, valid] > .5
    local_size = fine_world.shape[1]
    local_distance = (fine_world - truth[:, None]).norm(dim=-1)
    local_distance = local_distance.masked_fill(~fine_support, torch.inf)
    fine_index = fine_logits.argmax(-1)
    selected = fine_index.clamp_max(local_size - 1)
    scored_distance = local_distance[torch.arange(len(truth), device=device), selected]
    scored = global_500 & (fine_index < local_size) & (scored_distance <= 500)
    counts.update({'global_supported_le_500_um': int(global_500.sum()),
                   'local_supported_le_500_um': int((global_500 &
                       (local_distance.amin(-1) <= 500)).sum()),
                   'scored_fine_le_500_um': int(scored.sum()),
                   'fine_dustbin': int((fine_index == local_size).sum())})
    assert counts['scored_fine_le_500_um'] <= counts['local_supported_le_500_um'] <= counts['global_supported_le_500_um']
    return counts


def summarize(group):
    plans = defaultdict(list)
    for row in group:
        plans[row['subject_plan_id']].append(row)
    total_global = sum(row['global_supported_le_500_um'] for row in group)
    total_local = sum(row['local_supported_le_500_um'] for row in group)
    total_scored = sum(row['scored_fine_le_500_um'] for row in group)
    count = len(group)
    return {'sections': count, 'plans': len(plans),
        'valid_fixed_sites': sum(row['valid_fixed_sites'] for row in group),
        'global_supported_le_500_um': total_global,
        'local_supported_le_500_um': total_local,
        'scored_fine_le_500_um': total_scored,
        'fine_dustbin': sum(row['fine_dustbin'] for row in group),
        'local_ceiling_over_global500_percent': 100 * total_local / total_global if total_global else None,
        'scored_over_global500_percent': 100 * total_scored / total_global if total_global else None,
        'scored_over_local500_percent': 100 * total_scored / total_local if total_local else None,
        'original_mm_plan_equal': float(np.mean([np.mean([r['original_rigid_mm'] for r in p])
            for p in plans.values()])) if plans else None,
        'fitted_mm_plan_equal': float(np.mean([np.mean([r['fitted_rigid_mm'] for r in p])
            for p in plans.values()])) if plans else None,
        'original_within_0p5_percent': 100 * sum(r['original_rigid_mm'] <= .5 for r in group) / count if count else None,
        'fitted_within_0p5_percent': 100 * sum(r['fitted_rigid_mm'] <= .5 for r in group) / count if count else None,
        'whole_beam_original_retained_1p5_percent': 100 * sum(r['beam_best_original_mm'] <= 1.5 for r in group) / count if count else None,
        'whole_beam_fitted_retained_1p5_percent': (100 * sum(r['beam_best_fitted_mm'] <= 1.5
            for r in group) / count if count and all(r['beam_best_fitted_mm'] is not None
            for r in group) else None)}


out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'evaluator_sha256': sha(__file__),
          'parent_checkpoint_sha256': sha(parent_path),
          'parent_evaluation_completion_sha256': sha(parent_eval / 'completed.json'),
          'prior_training_completion_sha256': sha(prior_train / 'completed.json'),
          'prior_evaluation_completion_sha256': sha(prior_eval / 'completed.json'),
          'train_completion_sha256': sha(run / 'completed.json'),
          'panel_receipt_sha256': {cohort: sha(panel / 'completed.json')
                                   for cohort, panel in panels.items()},
          'arms': arms, 'step': step, 'conditions': conditions,
          'blind_branch': 'original frozen 132 blind16 IDs and candidate poses; unchanged for all arms and ablations',
          'truth_best_roles': 'blind_best_original and blind_best_corrected use DEV truth as oracle diagnostics, never model selection',
          'primary': 'scored fine argmax <=500um per globally supported <=500um fixed valid site, conditional on original truth-best beam error <=1.5mm',
          'support': 'fractional atlas support >0.5 for both global denominator and selected fine key',
          'source_flattening': 'input channel 0 replaced by its full-image mean; brush channels unchanged; parent features recomputed; original candidate pose retained',
          'ablations': 'full-intensity top4 original branch only; distribution shifts, not causal proof',
          'v3_material_retention_loss_threshold_percentage_points': 5.,
          'calibrated': False, 'public_benchmark_used': False,
          'expert_real_truth_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for cohort, panel in panels.items():
        for index, record in enumerate(panel_records[cohort]):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                target_state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).to(device).bool()
                dense = torch.from_numpy(arrays['target_centre_um'].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
            assert int(valid.sum()) == record['valid_pixels']
            prediction = model.predict(image)
            canonical_states = full_frame_state_from_components(
                *full_frame_state_to_components(prediction['state']))
            prior, ids = beam_for(prediction)
            observed = chart[valid]
            target = rigid_points_090(target_state, reflection, observed)[0]
            original = rigid_points_090(prediction['state'][:, ids // 2],
                                        (ids % 2)[None], observed)[0]
            canonical_points = rigid_points_090(canonical_states[:, ids // 2],
                                                (ids % 2)[None], observed)[0]
            assert (canonical_points - original).abs().amax() < .1
            errors = (original - target).norm(dim=-1).mean(-1) / 1000
            near_slot = int(errors.argmin())
            selected_slot = int(prior[ids].argmax())
            frozen = parent_rows[(cohort, record['section_id'])]
            assert ids.tolist() == frozen['beam_branch_ids']
            assert int(prior.argmax()) == frozen['top1_branch_id']
            assert all(abs(float(error) - previous) < 1e-4
                       for error, previous in zip(errors, frozen['beam_rigid_mm']))
            assert abs(float(errors[near_slot]) - frozen['beam_best_rigid_mm']) < 1e-4
            assert abs(float(errors[selected_slot]) - frozen['top1_rigid_mm']) < 1e-4
            for role, slot in (('blind_best_original', near_slot),
                               ('prior_selected', selected_slot)):
                previous = prior_rows[(cohort, record['section_id'], role)]
                assert int(ids[slot]) == previous['branch_id']
                assert abs(float(errors[slot]) - previous['original_rigid_mm']) < 1e-4
            truth_fixed = dense[fixed_y, fixed_x]
            valid_fixed = valid[fixed_y, fixed_x]
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            angle_bin = '<15' if angle < 15 else '15-30' if angle < 30 else '30-45' if angle < 45 else '>=45'
            provenance = record['provenance']
            artifacts = provenance['one_shot_slide_artifacts_v3']
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'appearance_mode': record['appearance_mode'],
                'nearest_cardinal_angle_deg': angle,
                'nearest_cardinal_angle_bin_deg': angle_bin,
                'observed_valid_tissue_fraction': record['valid_pixels'] / (side * side),
                'raw_exterior_code': (int(artifacts['parameters']['raw_exterior_code'])
                    if record['appearance_mode'] == 'raw' else None),
                'exposure': provenance.get('one_shot_slide_artifacts_v4', {}).get('exposure'),
                'artifact_events': artifacts['events'],
                'beam_branch_ids': ids.tolist(),
                'beam_best_original_mm': float(errors[near_slot]),
                'blind_best_available_1p5mm': bool(errors[near_slot] <= 1.5)}
            for arm in arms:
                head = heads[arm]
                outputs, corrected_parts = [], []
                for start in range(0, 16, 4):
                    part_ids = ids[start:start + 4]
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = head(prediction, image,
                            canonical_states[:, part_ids // 2], (part_ids % 2)[None],
                            atlas, offsets, weights,
                            atlas_intensity_enabled=arm.endswith('full_intensity'))
                    outputs.append(output)
                    corrected_parts.append(output['corrected_state'])
                corrected = torch.cat(corrected_parts, 1)
                fitted = rigid_points_090(corrected, (ids % 2)[None], observed)[0]
                fitted_error = (fitted - target).norm(dim=-1).mean(-1) / 1000
                corrected_slot = int(fitted_error.argmin())
                match_by_slot = {}
                for slot in {near_slot, selected_slot, corrected_slot}:
                    match_by_slot[slot] = match_counts(outputs[slot // 4],
                        truth_fixed, valid_fixed, slot % 4)
                for role, slot in (('blind_best_original', near_slot),
                                   ('prior_selected', selected_slot),
                                   ('blind_best_corrected', corrected_slot)):
                    row = {**common, 'arm': arm, 'condition': 'standard',
                        'role': role, 'branch_id': int(ids[slot]),
                        'original_rigid_mm': float(errors[slot]),
                        'fitted_rigid_mm': float(fitted_error[slot]),
                        'beam_best_fitted_mm': float(fitted_error[corrected_slot]),
                        **match_by_slot[slot]}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
                del outputs, corrected_parts, corrected
                if arm != 'top4_full_intensity':
                    continue
                branch = int(ids[near_slot])
                candidate = canonical_states[:, branch // 2:branch // 2 + 1]
                candidate_reflection = (ids[near_slot:near_slot + 1] % 2)[None]
                flat_image = image.clone()
                flat_image[:, 0] = image[:, 0].mean()
                flat_prediction = model.predict(flat_image)
                for condition in conditions[1:]:
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = head(
                            prediction if condition == 'atlas_intensity_off' else flat_prediction,
                            image if condition == 'atlas_intensity_off' else flat_image,
                            candidate, candidate_reflection, atlas, offsets, weights,
                            atlas_intensity_enabled=(condition == 'source_intensity_flat'))
                    fitted_ablated = rigid_points_090(output['corrected_state'],
                        candidate_reflection, observed)[0, 0]
                    fitted_ablated_error = float((fitted_ablated - target).norm(dim=-1).mean() / 1000)
                    row = {**common, 'arm': arm, 'condition': condition,
                        'role': 'blind_best_original', 'branch_id': int(ids[near_slot]),
                        'original_rigid_mm': float(errors[near_slot]),
                        'fitted_rigid_mm': fitted_ablated_error,
                        'beam_best_fitted_mm': None,
                        **match_counts(output, truth_fixed, valid_fixed, 0)}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
            if (index + 1) % 64 == 0:
                stream.flush()
                print(json.dumps({'event': 'development_milestone', 'cohort': cohort,
                                  'sections': index + 1}), flush=True)

summary = {'metrics': {}, 'diagnostic_warning':
    'Truth-best original and best-corrected branches use DEV truth only as oracle diagnostics. '
    'The two content ablations are distribution shifts, not causal proof.'}
for cohort in panels:
    summary['metrics'][cohort] = {}
    for arm in arms:
        summary['metrics'][cohort][arm] = {}
        for condition in (conditions if arm == 'top4_full_intensity' else ('standard',)):
            summary['metrics'][cohort][arm][condition] = {}
            for angle_bin in angle_bins:
                subset = [row for row in rows if row['cohort'] == cohort and row['arm'] == arm
                          and row['condition'] == condition and
                          (angle_bin == 'all' or row['nearest_cardinal_angle_bin_deg'] == angle_bin)]
                summary['metrics'][cohort][arm][condition][angle_bin] = {}
                for role in (roles if condition == 'standard' else ('blind_best_original',)):
                    group = [row for row in subset if row['role'] == role]
                    summary['metrics'][cohort][arm][condition][angle_bin][role] = {
                        'all_sections': summarize(group),
                        'near_le_1p5mm': summarize([row for row in group
                            if row['blind_best_available_1p5mm']])}


def metric(cohort, arm, role, scope='near_le_1p5mm', condition='standard'):
    return summary['metrics'][cohort][arm][condition]['all'][role][scope]


summary['matched_arm_comparisons'] = {}
for cohort in panels:
    matched = {arm: metric(cohort, arm, 'blind_best_original') for arm in arms}
    assert len({item['sections'] for item in matched.values()}) == 1
    assert len({item['global_supported_le_500_um'] for item in matched.values()}) == 1
    assert matched[arms[0]]['global_supported_le_500_um'] > 0
    rates = {arm: item['scored_over_global500_percent'] for arm, item in matched.items()}
    summary['matched_arm_comparisons'][cohort] = {
        'original_near_sections': matched[arms[0]]['sections'],
        'global_supported_le_500_um': matched[arms[0]]['global_supported_le_500_um'],
        'scored_over_global500_percent': rates,
        'full_top4_minus_full_top1_percentage_points':
            rates['top4_full_intensity'] - rates['top1_full_intensity'],
        'support_top4_minus_support_top1_percentage_points':
            rates['top4_support_only'] - rates['top1_support_only'],
        'full_top4_minus_support_top4_percentage_points':
            rates['top4_full_intensity'] - rates['top4_support_only'],
        'full_top1_minus_support_top1_percentage_points':
            rates['top1_full_intensity'] - rates['top1_support_only']}


near_full4 = metric('v4', 'top4_full_intensity', 'blind_best_original')
near_full1 = metric('v4', 'top1_full_intensity', 'blind_best_original')
near_support4 = metric('v4', 'top4_support_only', 'blind_best_original')
assert near_full4['sections'] == near_full1['sections'] == near_support4['sections'] == 179
assert near_full4['global_supported_le_500_um'] == near_full1['global_supported_le_500_um'] == near_support4['global_supported_le_500_um']
scored_full4 = near_full4['scored_over_global500_percent']
scored_full1 = near_full1['scored_over_global500_percent']
scored_support4 = near_support4['scored_over_global500_percent']
capture_gain = near_full4['fitted_within_0p5_percent'] - near_full4['original_within_0p5_percent']
plan_gain = near_full4['original_mm_plan_equal'] - near_full4['fitted_mm_plan_equal']
selected_v4 = metric('v4', 'top4_full_intensity', 'prior_selected', 'all_sections')
selected_v3 = metric('v3', 'top4_full_intensity', 'prior_selected', 'all_sections')
selected_regression = any(m['fitted_mm_plan_equal'] > m['original_mm_plan_equal'] or
                          m['fitted_within_0p5_percent'] < m['original_within_0p5_percent']
                          for m in (selected_v4, selected_v3))
v3_beam = metric('v3', 'top4_full_intensity', 'blind_best_original', 'all_sections')
v3_retention_loss = (v3_beam['whole_beam_original_retained_1p5_percent'] -
                     v3_beam['whole_beam_fitted_retained_1p5_percent'])
gate = {'cohort': 'v4', 'arm': 'top4_full_intensity',
    'global_supported_le_500_um': near_full4['global_supported_le_500_um'],
    'scored_full_top4_minus_full_top1_percentage_points': scored_full4 - scored_full1,
    'scored_full_top4_minus_support_top4_percentage_points': scored_full4 - scored_support4,
    'conditional_fit_capture_gain_percentage_points': capture_gain,
    'plan_equal_fit_error_reduction_mm': plan_gain,
    'selected_branch_regression_v4_or_v3': selected_regression,
    'v3_whole_beam_1p5_retention_loss_percentage_points': v3_retention_loss,
    'thresholds': {'scored_gain_vs_full_top1_percentage_points': 5.,
                   'scored_gain_vs_support_top4_percentage_points': 5.,
                   'conditional_fit_capture_gain_percentage_points': 10.,
                   'plan_equal_fit_error_reduction_mm': .20,
                   'v3_material_retention_loss_percentage_points': 5.}}
gate['pass'] = (gate['global_supported_le_500_um'] > 0 and
    gate['scored_full_top4_minus_full_top1_percentage_points'] >= 5 and
    gate['scored_full_top4_minus_support_top4_percentage_points'] >= 5 and
    capture_gain >= 10 and plan_gain >= .20 and not selected_regression and
    v3_retention_loss <= 5)
summary['decision'] = gate
summary['distribution_shift_diagnostics'] = {}
for cohort in panels:
    standard = metric(cohort, 'top4_full_intensity', 'blind_best_original')
    summary['distribution_shift_diagnostics'][cohort] = {}
    for condition in conditions[1:]:
        ablated = metric(cohort, 'top4_full_intensity', 'blind_best_original', condition=condition)
        assert ablated['sections'] == standard['sections']
        assert ablated['global_supported_le_500_um'] == standard['global_supported_le_500_um']
        summary['distribution_shift_diagnostics'][cohort][condition] = {
            'scored_fine_hit_change_percentage_points':
                ablated['scored_over_global500_percent'] - standard['scored_over_global500_percent'],
            'conditional_fit_capture_change_percentage_points':
                ablated['fitted_within_0p5_percent'] - standard['fitted_within_0p5_percent'],
            'plan_equal_fit_error_change_mm':
                ablated['fitted_mm_plan_equal'] - standard['fitted_mm_plan_equal']}
assert len(rows) == 491 * (4 * 3 + 2)
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'protocol_sha256': sha(protocol), 'train_completion_sha256': sha(run / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows), 'gate_pass': gate['pass']}), flush=True)
