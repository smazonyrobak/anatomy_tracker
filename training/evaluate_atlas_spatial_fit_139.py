"""Frozen blind-beam DEV readout for the 139 spatial atlas-fit heads."""

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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_spatial_fit_139 import AtlasSpatialFit139
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/ATLAS_SPATIAL_FIT_139_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
parent_eval = root / 'runs/v4_pose_adaptation_132_dev_eval'
train = root / 'runs/atlas_spatial_fit_139'
out = root / 'runs/atlas_spatial_fit_139_dev_eval'
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
arms = ('full_intensity', 'support_only')
side = 256
device = 'cuda'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_eval_done = json.loads((parent_eval / 'completed.json').read_text())
frozen_rows = [json.loads(line) for line in (parent_eval / 'synthetic_rows.jsonl').open()]
frozen = {(row['cohort'], row['section_id']): row for row in frozen_rows if row['arm'] == '2000'}
frozen_real = {(row['family'], row['section_id']): row for row in
    (json.loads(line) for line in (parent_eval / 'real_weak_rows.jsonl').open())
    if row['arm'] == '2000'}
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert all(sha(parent_eval / name) == digest for name, digest in parent_eval_done['output_sha256'].items())
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256']
assert train_config['parent_checkpoint_sha256'] == train_done['parent_checkpoint_sha256'] == sha(parent_path)
assert train_config['protocol_sha256'] == train_done['protocol_sha256'] == sha(protocol)
assert train_config['source_sha256'] == train_done['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert not any(train_done.get(key, False) for key in ('calibrated', 'expert_real_truth_used',
    'final_animals_used', 'public_benchmark_used', 'external_pretrained_weights_used'))
assert train_done['updates'] == 2000 and train_done['synthetic_presentations'] == 6000
draws = [json.loads(line) for line in (train / 'draws.jsonl').open()]
presentations = [row for row in draws if row['kind'] == 'synthetic_train' and row['used']]
selections = [row for row in draws if row['kind'] == 'blind_beam_selection']
assert len(presentations) == len(selections) == 6000
assert len({row['physical_section_id'] for row in presentations}) == 6000
assert len({(row['update'], row['slot']) for row in presentations}) == 6000
assert all(len(row['beam_ids']) == 16 and len(row['selected_ids']) <= 4 and
    set(row['selected_ids']) <= set(row['beam_ids']) for row in selections)
training_log = [json.loads(line) for line in (train / 'training.jsonl').open()]
assert len(training_log) == 2000
checkpoints = {arm: {str(step): train / arm / f'head_step_{step:05d}.pt'
                     for step in (0, 2000)} for arm in arms}
assert all(sha(path) == train_done['checkpoint_sha256'][arm][step]
           for arm, paths in checkpoints.items() for step, path in paths.items())

panel_records = {}
for cohort, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    panel_protocol = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / name) == digest
               for name, digest in panel_protocol['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == done['eligible']
    assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
assert len(panel_records['v4']) == 248 and len(panel_records['v3']) == 243

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_records = [row for row in coronal_all if row['training_split'] == 'development']
assert len(coronal_records) == 64 and len({row['animal_id'] for row in coronal_records}) == 6
assert not {row['animal_id'] for row in coronal_records} & {
    row['animal_id'] for row in coronal_all if row['training_split'] == 'train'}
coronal_images = np.load(coronal / 'images.npy', mmap_mode='r')
with np.load(coronal / 'geometry.npz', allow_pickle=False) as arrays:
    coronal_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert all(sha(sagittal / name) == digest for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_records) == 158 and len({row['donor_id'] for row in sagittal_records}) == 8
assert not {row['donor_id'] for row in sagittal_records} & {
    row['donor_id'] for row in sagittal_train_records}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')

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
zero_heads = {}
for arm in arms:
    initial = torch.load(checkpoints[arm]['0'], map_location='cpu', weights_only=True)
    terminal = torch.load(checkpoints[arm]['2000'], map_location='cpu', weights_only=True)
    assert initial['step'] == 0 and terminal['step'] == 2000
    assert initial['arm'] == terminal['arm'] == arm
    assert initial['config'] == terminal['config'] == train_config
    zero_heads[arm] = AtlasSpatialFit139().to(device).eval().requires_grad_(False)
    heads[arm] = AtlasSpatialFit139().to(device).eval().requires_grad_(False)
    zero_heads[arm].load_state_dict(initial['head'], strict=True)
    heads[arm].load_state_dict(terminal['head'], strict=True)
    del initial, terminal
assert all(torch.equal(a, b) for a, b in zip(
    zero_heads[arms[0]].state_dict().values(), zero_heads[arms[1]].state_dict().values()))

axis = torch.arange(side, device=device, dtype=torch.float32) / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
five_pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                            [127.5, 127.5]], device=device)
five_chart = five_pixels / side
real_weights = torch.tensor([[1., 2., 2., 2., 2., 2., 2., 2., 1.]], device=device) / 16
real_unit_offsets = torch.linspace(-.5, .5, 9, device=device)[None]


def beam_for(prediction):
    states = prediction['state']
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(states)[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device=device)[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior[0], beam[0]


def head_on_beam(head, prediction, image, states, ids, offsets, weights,
                 intensity=True, atlas_volume=None):
    corrected, scores, residual, support = [], [], [], []
    volume = atlas if atlas_volume is None else atlas_volume
    for part in ids.split(4):
        output = head(prediction, image, states[:, part // 2], (part % 2)[None],
                      volume, offsets, weights, atlas_intensity_enabled=intensity)
        corrected.append(output['corrected_state'])
        scores.append(output['score_delta'])
        residual.append(output['fit_residual_um'])
        support.append(output['coarse_key_support'])
    return (torch.cat(corrected, 1), torch.cat(scores, 1)[0],
            torch.cat(residual, 1)[0], torch.cat(support, 1)[0])


def rigid_errors(states, ids, observed_chart, target):
    predicted = rigid_points_090(states, (ids % 2)[None], observed_chart)[0]
    return ((predicted - target).norm(dim=-1).mean(-1) / 1000).tolist()


def score_and_fit(head, prediction, image, states, ids, prior, offsets, weights,
                  observed_chart=None, target=None, intensity=True, atlas_volume=None):
    corrected, score_delta, residual, support = head_on_beam(head, prediction, image,
        states, ids, offsets, weights, intensity=intensity, atlas_volume=atlas_volume)
    scores = {'fit_only': score_delta.tolist(),
              'prior_plus_fit': (prior[ids] + score_delta).tolist()}
    chosen = {name: int(np.argmax(values)) for name, values in scores.items()}
    result = {'score_delta': score_delta.tolist(), 'fit_residual_um': residual.tolist(),
              'scores': scores, 'selected_slot': chosen}
    if observed_chart is not None:
        costs = rigid_errors(corrected, ids, observed_chart, target)
        result['corrected_rigid_mm'] = costs
        result['corrected_beam_best_mm'] = min(costs)
        result['selected_rigid_mm'] = {name: costs[slot] for name, slot in chosen.items()}
    return result, corrected


def exact_support_fraction(states, ids, observed_chart, offsets, weights):
    sites = observed_chart[torch.linspace(0, len(observed_chart) - 1,
        min(256, len(observed_chart)), device=device).long()]
    branches = states[:, ids // 2]
    surface = rigid_points_090(branches, (ids % 2)[None], sites)[0]
    normal = full_frame_state_to_components(branches)[1][0, :, :, 2]
    coordinates = surface[:, None, :, None] + offsets[0, None, :, None, None, None] * (
        normal[:, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
        (0., 0., 0.), (25., 25., 25.), weights.expand(16, -1))
    return rendered[:, 1].mean((1, 2)).tolist()


v4_swap = {}
for plan in {row['synthetic_subject_plan_id'] for row in panel_records['v4']}:
    plan_rows = sorted((row for row in panel_records['v4']
        if row['synthetic_subject_plan_id'] == plan), key=lambda row: str(row['section_id']))
    for mode in {row['appearance_mode'] for row in plan_rows}:
        group = [row for row in plan_rows if row['appearance_mode'] == mode]
        for index, row in enumerate(group):
            donor = (group[(index + 1) % len(group)] if len(group) > 1 else
                     plan_rows[(plan_rows.index(row) + 1) % len(plan_rows)])
            v4_swap[row['section_id']] = donor
assert len(v4_swap) == 248 and all(key != donor['section_id']
    for key, donor in v4_swap.items())

synthetic_rows, real_rows, pair_rows = [], [], []
zero_checked = False
with torch.inference_mode():
    for cohort, panel in panels.items():
        for index, record in enumerate(panel_records[cohort]):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
                valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).to(device).bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
            assert int(valid.sum()) == record['valid_pixels']
            observed_chart = chart[valid]
            target = rigid_points_090(truth, reflection, observed_chart)[0]
            prediction = model.predict(image)
            prior, ids = beam_for(prediction)
            assert int(prior.argmax()) in ids
            states = prediction['state']
            original = rigid_errors(states[:, ids // 2], ids, observed_chart, target)
            parent_slot = int(prior[ids].argmax())
            frozen_row = frozen[(cohort, record['section_id'])]
            assert ids.tolist() == frozen_row['beam_branch_ids']
            assert int(prior.argmax()) == frozen_row['top1_branch_id']
            assert abs(min(original) - frozen_row['beam_best_rigid_mm']) < 1e-4
            assert abs(original[parent_slot] - frozen_row['top1_rigid_mm']) < 1e-4
            if not zero_checked:
                for arm in arms:
                    zero, zero_score, _, _ = head_on_beam(zero_heads[arm], prediction,
                        image, states, ids[:4], offsets, weights,
                        intensity=(arm == 'full_intensity'))
                    # Packing and unpacking the physical frame has sub-micrometre
                    # float32 cancellation even with the zero-initialized blend.
                    assert torch.allclose(zero, states[:, ids[:4] // 2], rtol=0, atol=.01)
                    assert torch.allclose(zero_score, torch.zeros_like(zero_score), rtol=0, atol=1e-6)
                zero_heads.clear()
                zero_checked = True
            common = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': record['appearance_mode'], 'panel_file_sha256': record['sha256'],
                'nearest_cardinal_angle_deg': float(np.degrees(np.arccos(np.clip(
                    np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
                'valid_pixels': int(valid.sum()), 'valid_fraction': float(valid.float().mean()),
                'exposure_multiplier': (record['provenance']['one_shot_slide_artifacts_v4']['exposure']
                    if cohort == 'v4' else 1.), 'eligible': True,
                'beam_branch_ids': ids.tolist(), 'beam_prior_logit': prior[ids].tolist(),
                'original_rigid_mm': original, 'original_beam_best_mm': min(original),
                'parent_direct_slot': parent_slot, 'parent_direct_rigid_mm': original[parent_slot]}
            row = {**common, 'arms': {}}
            for arm in arms:
                result, corrected = score_and_fit(heads[arm], prediction, image,
                    states, ids, prior, offsets, weights, observed_chart, target,
                    intensity=(arm == 'full_intensity'))
                row['arms'][arm] = result
            near_slot = int(np.argmin(original))
            support = exact_support_fraction(states, ids, observed_chart, offsets, weights)
            row['beam_atlas_support_fraction_on_valid_sites'] = support
            wrong = [slot for slot in range(16) if original[slot] >= 2 and
                     abs(support[slot] - support[near_slot]) <= .05]
            pair = {'cohort': cohort, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'near_slot': near_slot, 'near_original_mm': original[near_slot],
                'support_fraction_near': support[near_slot],
                'support_matched_wrong_count': len(wrong),
                'status': 'no_near_candidate'}
            if original[near_slot] <= 1.5:
                pair['status'] = 'no_support_matched_wrong'
                if wrong:
                    wrong_slot = min(wrong, key=lambda slot: original[slot])
                    pair.update(status='scored', wrong_slot=wrong_slot,
                        wrong_original_mm=original[wrong_slot],
                        support_fraction_wrong=support[wrong_slot],
                        corrected_physical_margin_mm={arm: (
                            row['arms'][arm]['corrected_rigid_mm'][wrong_slot] -
                            row['arms'][arm]['corrected_rigid_mm'][near_slot]) for arm in arms},
                        score_margin={arm: {mode: (
                            row['arms'][arm]['scores'][mode][near_slot] -
                            row['arms'][arm]['scores'][mode][wrong_slot])
                            for mode in ('fit_only', 'prior_plus_fit')} for arm in arms})
            pair_rows.append(pair)
            synthetic_rows.append(row)
            if (index + 1) % 64 == 0:
                print(json.dumps({'event': 'synthetic_milestone', 'cohort': cohort,
                    'eligible_sections': index + 1}), flush=True)
        print(json.dumps({'event': 'synthetic_completed', 'cohort': cohort,
                          'eligible_sections': len(panel_records[cohort])}), flush=True)

    for family, records in (('coronal', coronal_records), ('sagittal', sagittal_records)):
        for index, record in enumerate(records):
            array_index = record['array_row_index']
            if family == 'coronal':
                native = np.concatenate((np.asarray(coronal_images[array_index], dtype=np.float32),
                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(native).to(device), (side, side),
                    mode='bilinear', align_corners=False)
                affine = torch.as_tensor(coronal_affines[array_index], device=device,
                                         dtype=torch.float32)
                old_pixels = (five_pixels + .5) * (192 / side) - .5
                reference = affine[:, 2] + (old_pixels[:, :1] * affine[:, 0] +
                                             old_pixels[:, 1:] * affine[:, 1])
                donor = record['animal_id']
            else:
                native = np.concatenate((np.asarray(sagittal_images[array_index], dtype=np.float32),
                    np.zeros((4, side, side), dtype=np.float32)))[None]
                image = torch.from_numpy(native).to(device)
                affine = torch.as_tensor(record['model_pixel_to_ccf_ref9_ap_dv_ml_um'],
                                         device=device, dtype=torch.float32)
                reference = affine[:, 2] + (five_pixels[:, :1] * affine[:, 0] +
                                             five_pixels[:, 1:] * affine[:, 1])
                donor = record['donor_id']
            offsets = real_unit_offsets * 100.
            prediction = model.predict(image)
            prior, ids = beam_for(prediction)
            states = prediction['state']
            original = rigid_errors(states[:, ids // 2], ids, five_chart, reference)
            parent_slot = int(prior[ids].argmax())
            frozen_reference = frozen_real[(family, record['section_id'])]
            assert int(ids[parent_slot]) == frozen_reference['top1_branch_id']
            assert abs(original[parent_slot] - frozen_reference['direct_top1_weak_five_point_mm']) < 1e-4
            row = {'family': family, 'donor_id': donor, 'section_id': record['section_id'],
                'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                'beam_branch_ids': ids.tolist(), 'parent_direct_slot': parent_slot,
                'parent_direct_weak_five_point_mm': original[parent_slot],
                'parent_best16_weak_five_point_mm': min(original),
                'label_role': 'inherited weak Allen affine, not expert truth', 'arms': {}}
            for arm in arms:
                corrected, score_delta, residual, _ = head_on_beam(heads[arm], prediction,
                    image, states, ids, offsets, real_weights,
                    intensity=(arm == 'full_intensity'))
                costs = rigid_errors(corrected, ids, five_chart, reference)
                scores = {'fit_only': score_delta.tolist(),
                          'prior_plus_fit': (prior[ids] + score_delta).tolist()}
                chosen = {mode: int(np.argmax(values)) for mode, values in scores.items()}
                row['arms'][arm] = {'corrected_weak_five_point_mm': costs,
                    'score_delta': score_delta.tolist(), 'fit_residual_um': residual.tolist(),
                    'selected_slot': chosen,
                    'selected_weak_five_point_mm': {mode: costs[slot]
                        for mode, slot in chosen.items()}}
            real_rows.append(row)
            if (index + 1) % 64 == 0:
                print(json.dumps({'event': 'weak_real_milestone', 'family': family,
                    'sections': index + 1}), flush=True)
        print(json.dumps({'event': 'weak_real_completed', 'family': family,
                          'sections': len(records)}), flush=True)

scored_pairs = [row for row in pair_rows if row['cohort'] == 'v4' and row['status'] == 'scored']
if scored_pairs:
    generator = torch.Generator(device='cpu').manual_seed(139)
    atlas_shuffled = atlas.clone()
    foreground = atlas[1] > .5
    foreground_values = atlas[0][foreground]
    permutation = torch.randperm(foreground_values.numel(), generator=generator).to(device)
    atlas_shuffled[0][foreground] = foreground_values[permutation]
    del foreground_values, permutation, foreground
    v4_record = {row['section_id']: row for row in panel_records['v4']}
    with torch.inference_mode():
        for index, pair in enumerate(scored_pairs):
            record = v4_record[pair['section_id']]
            donor = v4_swap[pair['section_id']]
            with np.load(panels['v4'] / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
                weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
                valid = torch.from_numpy(arrays['valid_mask'].flatten().copy()).to(device).bool()
                centre_surface = torch.from_numpy(arrays['target_centre_um'].copy()).to(device)
                truth = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).to(device).long()
            with np.load(panels['v4'] / donor['file'], allow_pickle=False) as arrays:
                donor_image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
            prediction = model.predict(image)
            donor_prediction = model.predict(donor_image)
            prior, ids = beam_for(prediction)
            assert ids.tolist() == next(row['beam_branch_ids'] for row in synthetic_rows
                if row['cohort'] == 'v4' and row['section_id'] == pair['section_id'])
            observed_chart = chart[valid]
            target = rigid_points_090(truth, reflection, observed_chart)[0]
            near, wrong = pair['near_slot'], pair['wrong_slot']
            pair_ids = ids[torch.tensor([near, wrong], device=device)]
            matched = heads['full_intensity'](prediction, image, prediction['state'][:, pair_ids // 2],
                (pair_ids % 2)[None], atlas, offsets, weights)
            pixel_index = torch.arange(16, device=device) * 16 + 8
            y_index, x_index = torch.meshgrid(pixel_index, pixel_index, indexing='ij')
            fixed_valid = valid.reshape(side, side)[y_index, x_index].flatten()
            fixed_surface = centre_surface[y_index, x_index].reshape(256, 3)
            pair['dustbin'] = {}
            for slot, role in enumerate(('near', 'wrong')):
                key_support = matched['coarse_key_support'][0, slot] > .5
                key_world = matched['coarse_key_world_um'][0, slot, key_support]
                actual = ~fixed_valid.clone()
                if key_world.numel():
                    actual[fixed_valid] = (torch.cdist(fixed_surface[fixed_valid][None],
                        key_world[None])[0].amin(-1) > 1500)
                for scale in ('coarse', 'fine'):
                    predicted = matched[f'{scale}_logits'][0, slot].argmax(-1) == (
                        7168 if scale == 'coarse' else 75)
                    pair['dustbin'][f'{role}_{scale}'] = {
                        'tp': int((actual & predicted).sum()),
                        'fp': int((~actual & predicted).sum()),
                        'fn': int((actual & ~predicted).sum()),
                        'tn': int((~actual & ~predicted).sum())}
            del matched
            for name, source_prediction, source_image, atlas_volume in (
                    ('source_swap', donor_prediction, donor_image, atlas),
                    ('within_support_atlas_shuffle', prediction, image, atlas_shuffled)):
                corrected, score_delta, _, _ = head_on_beam(heads['full_intensity'],
                    source_prediction, source_image, prediction['state'], ids, offsets,
                    weights, atlas_volume=atlas_volume)
                costs = rigid_errors(corrected, ids, observed_chart, target)
                score = prior[ids] + score_delta
                pair[name] = {'donor_section_id': donor['section_id'] if name == 'source_swap' else None,
                    'donor_file_sha256': donor['sha256'] if name == 'source_swap' else None,
                    'prior_plus_fit_margin': float(score[near] - score[wrong]),
                    'corrected_physical_margin_mm': costs[wrong] - costs[near]}
            if (index + 1) % 64 == 0:
                print(json.dumps({'event': 'conditional_ablation_milestone',
                    'scored_pairs': index + 1}), flush=True)
    del atlas_shuffled
print(json.dumps({'event': 'conditional_ablations_completed',
                  'scored_v4_pairs': len(scored_pairs)}), flush=True)

assert len(synthetic_rows) == 248 + 243
assert len(real_rows) == 64 + 158
assert len(pair_rows) == len(synthetic_rows)
v4_rows = [row for row in synthetic_rows if row['cohort'] == 'v4']
v3_rows = [row for row in synthetic_rows if row['cohort'] == 'v3']
assert sum(row['original_beam_best_mm'] <= .5 for row in v4_rows) == 25
assert sum(row['parent_direct_rigid_mm'] <= 1.5 for row in v4_rows) == 73
assert abs(float(np.mean([row['parent_direct_rigid_mm'] for row in v4_rows])) - 3.282) < .002


def synthetic_summary(rows, arm, mode):
    if not rows:
        return {'sections': 0, 'blind16_best_mean_mm': None,
            'blind16_best_le_0p5': None, 'blind16_best_le_1p5': None,
            'selected_mean_mm': None, 'selected_le_0p5': None,
            'selected_le_1p5': None}
    if arm == 'parent':
        best = [row['original_beam_best_mm'] for row in rows]
        selected = [row['parent_direct_rigid_mm'] for row in rows]
    else:
        best = [row['arms'][arm]['corrected_beam_best_mm'] for row in rows]
        selected = [row['arms'][arm]['selected_rigid_mm'][mode] for row in rows]
    return {'sections': len(rows), 'blind16_best_mean_mm': float(np.mean(best)),
        'blind16_best_le_0p5': float(np.mean(np.asarray(best) <= .5)),
        'blind16_best_le_1p5': float(np.mean(np.asarray(best) <= 1.5)),
        'selected_mean_mm': float(np.mean(selected)),
        'selected_le_0p5': float(np.mean(np.asarray(selected) <= .5)),
        'selected_le_1p5': float(np.mean(np.asarray(selected) <= 1.5))}


synthetic = {}
for cohort, rows in (('v4', v4_rows), ('v3', v3_rows)):
    cohort_result = {'parent_direct': synthetic_summary(rows, 'parent', 'direct'),
                     'arms': {}}
    for arm in arms:
        cohort_result['arms'][arm] = {}
        for mode in ('fit_only', 'prior_plus_fit'):
            cohort_result['arms'][arm][mode] = {'overall': synthetic_summary(rows, arm, mode),
                'by_plan': {str(plan): synthetic_summary([row for row in rows
                    if row['subject_plan_id'] == plan], arm, mode)
                    for plan in sorted({row['subject_plan_id'] for row in rows})},
                'by_mode': {appearance: synthetic_summary([row for row in rows
                    if row['appearance_mode'] == appearance], arm, mode)
                    for appearance in ('raw', 'exact_black', 'imperfect_brush')},
                'by_angle_deg': {f'[{low},{high})': synthetic_summary([row for row in rows
                    if low <= row['nearest_cardinal_angle_deg'] < high], arm, mode)
                    for low, high in ((0, 15), (15, 30), (30, 45), (45, 55))},
                'by_valid_fraction': {f'[{low},{high})': synthetic_summary([row for row in rows
                    if low <= row['valid_fraction'] < high], arm, mode)
                    for low, high in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))},
                'by_exposure': {f'[{low},{high})': synthetic_summary([row for row in rows
                    if low <= row['exposure_multiplier'] < high], arm, mode)
                    for low, high in ((0, .15), (.15, .3), (.3, .6), (.6, 1.21))}}
    synthetic[cohort] = cohort_result

conversions = {}
for cohort, rows in (('v4', v4_rows), ('v3', v3_rows)):
    at_risk = [row for row in rows if .5 < row['original_beam_best_mm'] <= 1.5]
    conversions[cohort] = {'parent_0p5_to_1p5_count': len(at_risk),
        'arms': {arm: {'converted_to_le_0p5_count': sum(
            row['arms'][arm]['corrected_beam_best_mm'] <= .5 for row in at_risk),
            'conditional_fraction': float(np.mean([
                row['arms'][arm]['corrected_beam_best_mm'] <= .5 for row in at_risk]))
                if at_risk else None} for arm in arms}}

real = {}
for family in ('coronal', 'sagittal'):
    family_rows = [row for row in real_rows if row['family'] == family]
    donors = sorted({row['donor_id'] for row in family_rows})
    real[family] = {'sections': len(family_rows), 'donors': len(donors),
        'by_donor': {str(donor): {'sections': len([row for row in family_rows
            if row['donor_id'] == donor]),
            'parent_selected_mean_mm': float(np.mean([row['parent_direct_weak_five_point_mm']
                for row in family_rows if row['donor_id'] == donor])),
            'arm_selected_mean_mm': {arm: float(np.mean([
                row['arms'][arm]['selected_weak_five_point_mm']['prior_plus_fit']
                for row in family_rows if row['donor_id'] == donor])) for arm in arms}}
            for donor in donors}}
    real[family]['donor_equal_parent_mean_mm'] = float(np.mean([
        value['parent_selected_mean_mm'] for value in real[family]['by_donor'].values()]))
    real[family]['donor_equal_arm_mean_mm'] = {arm: float(np.mean([
        value['arm_selected_mean_mm'][arm] for value in real[family]['by_donor'].values()]))
        for arm in arms}

pair_summary = {'status_counts': {status: sum(row['status'] == status for row in pair_rows)
    for status in ('scored', 'no_near_candidate', 'no_support_matched_wrong')},
    'v4_scored_pairs': len(scored_pairs)}
if scored_pairs:
    pair_summary['v4_corrected_near_physically_better_fraction'] = {arm: float(np.mean([
        row['corrected_physical_margin_mm'][arm] > 0 for row in scored_pairs])) for arm in arms}
    pair_summary['v4_prior_plus_fit_near_win_fraction'] = {arm: float(np.mean([
        row['score_margin'][arm]['prior_plus_fit'] > 0 for row in scored_pairs])) for arm in arms}
    pair_summary['v4_fit_only_near_win_fraction'] = {arm: float(np.mean([
        row['score_margin'][arm]['fit_only'] > 0 for row in scored_pairs])) for arm in arms}
    pair_summary['v4_ablation_near_win_fraction'] = {name: float(np.mean([
        row[name]['prior_plus_fit_margin'] > 0 for row in scored_pairs]))
        for name in ('source_swap', 'within_support_atlas_shuffle')}
    pair_summary['v4_dustbin'] = {}
    for role in ('near', 'wrong'):
        for scale in ('coarse', 'fine'):
            counts = {name: sum(row['dustbin'][f'{role}_{scale}'][name]
                for row in scored_pairs) for name in ('tp', 'fp', 'fn', 'tn')}
            pair_summary['v4_dustbin'][f'{role}_{scale}'] = {**counts,
                'precision': counts['tp'] / (counts['tp'] + counts['fp'])
                    if counts['tp'] + counts['fp'] else None,
                'recall': counts['tp'] / (counts['tp'] + counts['fn'])
                    if counts['tp'] + counts['fn'] else None}

v4_parent = synthetic['v4']['parent_direct']
v4_full = synthetic['v4']['arms']['full_intensity']['prior_plus_fit']['overall']
v4_support = synthetic['v4']['arms']['support_only']['prior_plus_fit']['overall']
v3_parent = synthetic['v3']['parent_direct']
v3_full = synthetic['v3']['arms']['full_intensity']['prior_plus_fit']['overall']
plans = sorted({row['subject_plan_id'] for row in v4_rows})
plan_improved = sum(synthetic_summary([row for row in v4_rows
    if row['subject_plan_id'] == plan], 'full_intensity', 'prior_plus_fit')['selected_le_1p5']
    > synthetic_summary([row for row in v4_rows
    if row['subject_plan_id'] == plan], 'parent', 'direct')['selected_le_1p5']
    for plan in plans)
pair_evidence = bool(scored_pairs and
    pair_summary['v4_prior_plus_fit_near_win_fraction']['full_intensity'] > .5 and
    pair_summary['v4_prior_plus_fit_near_win_fraction']['full_intensity'] >
        pair_summary['v4_prior_plus_fit_near_win_fraction']['support_only'] and
    pair_summary['v4_corrected_near_physically_better_fraction']['full_intensity'] > .5)
ablation_collapse = bool(scored_pairs and all(
    pair_summary['v4_ablation_near_win_fraction'][name] <= max(.5,
        pair_summary['v4_prior_plus_fit_near_win_fraction']['support_only'])
    for name in ('source_swap', 'within_support_atlas_shuffle')))
conditions = {
    'v4_blind16_le_0p5_gain_vs_parent_ge_0p15':
        v4_full['blind16_best_le_0p5'] - v4_parent['blind16_best_le_0p5'] >= .15,
    'v4_selected_le_1p5_gain_vs_parent_ge_0p10':
        v4_full['selected_le_1p5'] - v4_parent['selected_le_1p5'] >= .10,
    'v4_selected_mean_gain_vs_parent_ge_0p30_mm':
        v4_parent['selected_mean_mm'] - v4_full['selected_mean_mm'] >= .30,
    'v4_selected_le_1p5_gain_vs_support_ge_0p10':
        v4_full['selected_le_1p5'] - v4_support['selected_le_1p5'] >= .10,
    'v4_selected_mean_gain_vs_support_ge_0p20_mm':
        v4_support['selected_mean_mm'] - v4_full['selected_mean_mm'] >= .20,
    'v4_selected_near_improves_ge_6_of_8_plans': plan_improved >= 6,
    'v3_blind16_le_1p5_drop_vs_parent_le_0p05':
        v3_full['blind16_best_le_1p5'] >= v3_parent['blind16_best_le_1p5'] - .05,
    'all_weak_real_donors_selected_regression_vs_parent_le_0p20_mm': all(
        donor['arm_selected_mean_mm']['full_intensity'] <=
        donor['parent_selected_mean_mm'] + .20 for family in real.values()
        for donor in family['by_donor'].values()),
    'support_matched_pair_evidence_positive_on_physical_pose': pair_evidence,
    'pair_advantage_collapses_under_both_conditional_ablations': ablation_collapse}
summary = {'scope': 'frozen 132 blind16; terminal matched 139 heads; reused-plan synthetic DEV and donor-disjoint inherited weak-affine DEV only',
    'metric': 'mean 3D rigid displacement on observed-valid native256 tissue pixels; weak real five-point inherited-affine discrepancy in mm',
    'oracle_role': 'truth-best blind16 and matched near/wrong pairs are diagnostic only, never inference',
    'primary_selection': 'frozen reflected pose prior plus spatial-fit score residual',
    'conditional_ablation_definition': 'non-self within-plan/style source-image+feature swap; global atlas intensity permuted among supported voxels with support fixed',
    'pair_collapse_operationalization': 'each ablation near-win fraction no higher than max(0.5, support-only near-win fraction)',
    'synthetic': synthetic, 'conditional_near_to_precise': conversions,
    'real_weak_affine': real, 'support_matched_pairs': pair_summary,
    'endpoint': {'v4_plan_selected_near_improvement_count': plan_improved,
        'conditions': conditions, 'mechanism_gate_passed': all(conditions.values()),
        'promotion_permitted': False, 'development_only': True},
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
config = {'evaluator_source_sha256': sha(__file__), 'head_source_sha256': sha(source / 'atlas_spatial_fit_139.py'),
    'protocol_sha256': sha(protocol), 'train_completion_sha256': sha(train / 'completed.json'),
    'train_config_sha256': sha(train / 'config.json'),
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_evaluation_completion_sha256': sha(parent_eval / 'completed.json'),
    'checkpoint_sha256': {arm: {step: sha(path) for step, path in paths.items()}
        for arm, paths in checkpoints.items()},
    'panel_completion_sha256': {cohort: sha(panel / 'completed.json')
        for cohort, panel in panels.items()},
    'panel_protocol_sha256': {cohort: sha(panel / 'protocol.json')
        for cohort, panel in panels.items()},
    'panel_records_sha256': {cohort: sha(panel / 'records.jsonl')
        for cohort, panel in panels.items()},
    'coronal_completion_sha256': sha(coronal / 'completed.json'),
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'allen_atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'real_psf': 'neutral 9-point boxcar across 100um, identical for coronal/sagittal weak-reference DEV; no synthetic truth PSF',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
for name, rows in (('synthetic_rows.jsonl', synthetic_rows),
                   ('real_weak_rows.jsonl', real_rows),
                   ('support_matched_pair_rows.jsonl', pair_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'eligible_v4_sections': len(v4_rows), 'eligible_v3_sections': len(v3_rows),
    'real_coronal_sections': len(coronal_records),
    'real_sagittal_sections': len(sagittal_records),
    'output_sha256': {name: sha(out / name) for name in
        ('config.json', 'summary.json', 'synthetic_rows.jsonl',
         'real_weak_rows.jsonl', 'support_matched_pair_rows.jsonl')},
    'evaluator_source_sha256': config['evaluator_source_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'mechanism_gate_passed': summary['endpoint']['mechanism_gate_passed'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'endpoint': summary['endpoint']},
                 allow_nan=False), flush=True)
