"""Blind-beam and clearly marked oracle-pose readout for correspondence curriculum 140."""

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
from training.atlas_spatial_fit_139 import AtlasSpatialFit139
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
run = root / 'runs/atlas_correspondence_curriculum_140'
out = root / 'runs/atlas_correspondence_curriculum_140_dev_eval'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
arms = ('full_intensity', 'support_only')
steps = (0, 4000)
device = 'cuda'
side = 256
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert source.drive.upper() == run.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
parent_eval_done = json.loads((parent_eval / 'completed.json').read_text())
assert all(sha(parent_eval / name) == digest
           for name, digest in parent_eval_done['output_sha256'].items())
parent_rows = {(row['cohort'], row['section_id']): row for row in
    (json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open())
    if row['arm'] == '2000'}
train_done = json.loads((run / 'completed.json').read_text())
train_config = json.loads((run / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(run / 'config.json') == train_done['config_sha256']
assert sha(run / 'draws.jsonl') == train_done['draws_sha256']
assert sha(run / 'training.jsonl') == train_done['training_sha256']
assert train_config['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['source_sha256'] == train_done['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert train_done['unique_synthetic_physical_sections'] == 12000
assert not any(train_done.get(key, False) for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'final_animals_used', 'external_pretrained_weights_used'))
checkpoints = {arm: {step: run / arm / f'head_step_{step:05d}.pt' for step in steps}
               for arm in arms}
assert all(sha(path) == train_done['checkpoint_sha256'][arm][str(step)]
           for arm, files in checkpoints.items() for step, path in files.items())

panel_records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert len(records) == done['physical_sections'] == 256
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    assert all(row['provenance']['split'] == 'development' for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243

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
    for step in steps:
        saved = torch.load(checkpoints[arm][step], map_location='cpu', weights_only=True)
        assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
        head = AtlasSpatialFit139().to(device).eval().requires_grad_(False)
        head.load_state_dict(saved['head'], strict=True)
        assert head.gate.weight.abs().max() == 0 and abs(float(head.gate.bias[0]) - 2.65) < 1e-5
        assert head.score.weight.abs().max() == 0 and head.score.bias.abs().max() == 0
        heads[(arm, step)] = head
        del saved

axis = torch.arange(side, device=device) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
fixed = heads[(arms[0], 0)].fixed_query_xy
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
    if len(truth) == 0:
        return {'accessible': 0, 'coarse_hit': 0, 'coarse_top4_hit': 0, 'fine_hit': 0}
    keys = output['coarse_key_world_um'][0, slot].float()
    support = output['coarse_key_support'][0, slot] > .5
    distance = torch.cdist(truth[None], keys[None])[0]
    distance[:, ~support] = float('inf')
    accessible = distance.amin(-1) <= 1000
    coarse = output['coarse_logits'][0, slot, valid].float()
    top = coarse[:, :-1].topk(4, -1).indices
    top_distance = distance.gather(1, top)
    matched = coarse.argmax(-1) < keys.shape[0]
    coarse_hit = accessible & matched & (top_distance[:, 0] <= 1000)
    coarse_top4 = accessible & (top_distance.amin(-1) <= 1000)
    fine = output['fine_logits'][0, slot, valid].float()
    fine_index = fine.argmax(-1)
    fine_world = output['fine_key_world_um'][0, slot, valid].float()
    fine_pos = fine_world[torch.arange(len(truth), device=device), fine_index.clamp_max(74)]
    fine_hit = accessible & (fine_index < 75) & ((fine_pos - truth).norm(dim=-1) <= 500)
    return {'accessible': int(accessible.sum()), 'coarse_hit': int(coarse_hit.sum()),
            'coarse_top4_hit': int(coarse_top4.sum()), 'fine_hit': int(fine_hit.sum())}


out.mkdir(parents=True, exist_ok=False)
config = {'parent_checkpoint_sha256': sha(parent_path),
          'train_completion_sha256': sha(run / 'completed.json'),
          'train_config_sha256': sha(run / 'config.json'),
          'evaluator_sha256': sha(__file__),
          'panel_receipt_sha256': {cohort: sha(panel / 'completed.json')
                                   for cohort, panel in panels.items()},
          'arms': arms, 'steps': steps,
          'blind_branch': 'all 16 original frozen-parent beam branches fitted in fixed four-branch chunks',
          'truth_best_roles': 'diagnostic oracle within the unchanged blind16, never selection',
          'oracle_branch': 'exact target physical pose injected for correspondence diagnostic only, never selection',
          'matched_key': 'within 1mm of observed valid dense CCF coordinate on 16x16 fixed query grid',
          'descriptive_strata': 'angle, observed tissue fraction, input mode, raw exterior, exposure, and artifact events; not additional decision gates',
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
            prediction = model.predict(image)
            canonical_states = full_frame_state_from_components(
                *full_frame_state_to_components(prediction['state']))
            prior, ids = beam_for(prediction)
            observed = chart[valid]
            target = rigid_points_090(target_state, reflection, observed)[0]
            original = rigid_points_090(prediction['state'][:, ids // 2], (ids % 2)[None], observed)[0]
            canonical_points = rigid_points_090(canonical_states[:, ids // 2],
                (ids % 2)[None], observed)[0]
            assert (canonical_points - original).abs().amax() < .1
            errors = (original - target).norm(dim=-1).mean(-1) / 1000
            near_slot = int(errors.argmin())
            selected_slot = int(prior[ids].argmax())
            frozen = parent_rows[(cohort, record['section_id'])]
            assert ids.tolist() == frozen['beam_branch_ids']
            assert int(prior.argmax()) == frozen['top1_branch_id']
            assert abs(float(errors[near_slot]) - frozen['beam_best_rigid_mm']) < 1e-4
            assert abs(float(errors[selected_slot]) - frozen['top1_rigid_mm']) < 1e-4
            truth_fixed = dense[fixed_y, fixed_x]
            valid_fixed = valid[fixed_y, fixed_x]
            provenance = record['provenance']
            artifacts = provenance['one_shot_slide_artifacts_v3']
            angle = float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0., 1.))))
            assert int(valid.sum()) == record['valid_pixels']
            for arm in arms:
                for step in steps:
                    corrected_parts = []
                    match_by_slot = {}
                    for start in range(0, 16, 4):
                        part_ids = ids[start:start + 4]
                        with torch.autocast('cuda', dtype=torch.float16):
                            output = heads[(arm, step)](prediction, image,
                                canonical_states[:, part_ids // 2], (part_ids % 2)[None],
                                atlas, offsets, weights,
                                atlas_intensity_enabled=(arm == arms[0]))
                        corrected_parts.append(output['corrected_state'])
                        for slot in (near_slot, selected_slot):
                            if start <= slot < start + 4:
                                match_by_slot[slot] = match_counts(
                                    output, truth_fixed, valid_fixed, slot - start)
                    corrected = torch.cat(corrected_parts, 1)
                    fitted = rigid_points_090(corrected, (ids % 2)[None], observed)[0]
                    fitted_error = (fitted - target).norm(dim=-1).mean(-1) / 1000
                    corrected_best_slot = int(fitted_error.argmin())
                    with torch.autocast('cuda', dtype=torch.float16):
                        exact_output = heads[(arm, step)](prediction, image,
                            target_state[:, None], reflection[:, None], atlas,
                            offsets, weights, atlas_intensity_enabled=(arm == arms[0]))
                    exact_fit = rigid_points_090(exact_output['corrected_state'],
                        reflection[:, None], observed)[0, 0]
                    roles = (('blind_best_original', near_slot),
                             ('prior_selected', selected_slot),
                             ('blind_best_corrected', corrected_best_slot),
                             ('oracle_exact', None))
                    for role, slot in roles:
                        counts = (match_counts(exact_output, truth_fixed, valid_fixed, 0)
                            if slot is None else match_by_slot.get(slot,
                                {'accessible': 0, 'coarse_hit': 0,
                                 'coarse_top4_hit': 0, 'fine_hit': 0}))
                        row = {'cohort': cohort, 'section_id': record['section_id'],
                            'physical_section_id': record['panel_physical_section_id'],
                            'subject_plan_id': record['synthetic_subject_plan_id'],
                            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                            'experiment_id': record['experiment_id'],
                            'appearance_mode': record['appearance_mode'],
                            'nearest_cardinal_angle_deg': angle,
                            'observed_valid_tissue_fraction': record['valid_pixels'] / (side * side),
                            'raw_exterior_code': (int(artifacts['parameters']['raw_exterior_code'])
                                if record['appearance_mode'] == 'raw' else None),
                            'exposure': provenance.get('one_shot_slide_artifacts_v4', {}).get('exposure'),
                            'artifact_events': artifacts['events'],
                            'arm': arm, 'step': step, 'role': role,
                            'branch_id': int(ids[slot]) if slot is not None else None,
                            'blind_best_available_1p5mm': bool(errors[near_slot] <= 1.5),
                            'original_rigid_mm': float(errors[slot]) if slot is not None else 0.,
                            'fitted_rigid_mm': float(fitted_error[slot]) if slot is not None
                                else float((exact_fit - target).norm(dim=-1).mean() / 1000),
                            'valid_fixed_sites': int(valid_fixed.sum()), **counts}
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        rows.append(row)
            if (index + 1) % 64 == 0:
                stream.flush()
                print(json.dumps({'event': 'development_milestone', 'cohort': cohort,
                    'sections': index + 1}), flush=True)

summary = {}
for cohort in panels:
    for arm in arms:
        for step in steps:
            subset = [r for r in rows if r['cohort'] == cohort and r['arm'] == arm and r['step'] == step]
            block = {}
            for role in ('blind_best_original', 'blind_best_corrected',
                         'prior_selected', 'oracle_exact'):
                group = [r for r in subset if r['role'] == role]
                if role in ('blind_best_original', 'blind_best_corrected'):
                    group = [r for r in group if r['blind_best_available_1p5mm']]
                plans = defaultdict(list)
                for row in group:
                    plans[row['subject_plan_id']].append(row)
                block[role] = {'sections': len(group), 'plans': len(plans),
                    'original_mm_plan_equal': float(np.mean([np.mean([r['original_rigid_mm'] for r in p])
                        for p in plans.values()])),
                    'fitted_mm_plan_equal': float(np.mean([np.mean([r['fitted_rigid_mm'] for r in p])
                        for p in plans.values()])),
                    'original_within_0p5': sum(r['original_rigid_mm'] <= .5 for r in group),
                    'fitted_within_0p5': sum(r['fitted_rigid_mm'] <= .5 for r in group),
                    'accessible_fixed_sites': sum(r['accessible'] for r in group),
                    'coarse_hit': sum(r['coarse_hit'] for r in group),
                    'coarse_top4_hit': sum(r['coarse_top4_hit'] for r in group),
                    'fine_hit': sum(r['fine_hit'] for r in group)}
            all_original = [r for r in subset if r['role'] == 'blind_best_original']
            all_corrected = [r for r in subset if r['role'] == 'blind_best_corrected']
            block['all_blind'] = {'sections': len(all_original),
                'original_within_0p5': sum(r['original_rigid_mm'] <= .5 for r in all_original),
                'fitted_within_0p5': sum(r['fitted_rigid_mm'] <= .5 for r in all_corrected),
                'original_within_1p5': sum(r['original_rigid_mm'] <= 1.5 for r in all_original),
                'fitted_within_1p5': sum(r['fitted_rigid_mm'] <= 1.5 for r in all_corrected)}
            summary[f'{cohort}:{arm}:{step}'] = block
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'evaluator_sha256': sha(__file__),
    'train_completion_sha256': sha(run / 'completed.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'rows': len(rows)}), flush=True)
