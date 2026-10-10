"""Frozen 132 DEV assay: does pointwise anatomy agreement rank fitted planes?"""

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
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def beam_for(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                      prior[:, 32:].topk(6, -1).indices + 32), -1)
    normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
    for _ in range(2):
        chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
        diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
        diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
        anchor = diversity.argmax(-1)
        reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
        beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
    return prior, beam


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/FIXED_MAP_ENERGY_133_PROTOCOL_20261010.md'
panel = root / 'data/fresh_v4_pose_dev_panel_132'
train = root / 'runs/v4_pose_adaptation_132'
checkpoint = train / 'joint_step_02000.pt'
out = root / 'runs/fixed_map_energy_133'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
panel_done = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256']
assert sha(checkpoint) == train_done['checkpoint_sha256']['2000']
assert sha(source.parent / 'docs/publication/V4_POSE_ADAPTATION_132_PROTOCOL_20261010.md') == train_config['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert sha(panel / 'protocol.json') == panel_done['protocol_sha256']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
assert all(sha(panel / 'source' / name) == digest for name, digest in panel_protocol['source_sha256'].items())
assert len(records) == panel_done['physical_sections'] == 256
assert sum(row['eligible'] for row in records) == panel_done['eligible'] == 248
assert sum(not row['eligible'] for row in records) == panel_done['ineligible'] == 8
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
assert len({row['panel_physical_section_id'] for row in records}) == 256
assert all(row['provenance']['split'] == 'development' for row in records)
assert all(sha(panel / row['file']) == row['sha256'] for row in records)
assert not any(train_done.get(key, False) for key in ('calibrated', 'expert_real_truth_used',
    'final_animals_used', 'public_benchmark_used', 'external_pretrained_weights_used'))

atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == train_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

axis = torch.arange(256, device='cuda', dtype=torch.float32) / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
arms = ('prior', 'full', 'atlas_zero', 'atlas_shuffle', 'source_shuffle')
rows = []
with torch.inference_mode():
    for index, record in enumerate(records):
        row = {'section_id': record['section_id'],
            'physical_section_id': record['panel_physical_section_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'subject_plan_id': record['synthetic_subject_plan_id'],
            'appearance_mode': record['appearance_mode'],
            'panel_file_sha256': record['sha256'], 'eligible': record['eligible']}
        if not record['eligible']:
            row['status'] = 'frozen_ineligible_unscored'
            rows.append(row)
            continue

        # No target state, valid mask or tissue coordinates are read until scoring is complete.
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior, beam = beam_for(prediction)
        ids = beam[0]
        source_feature = F.adaptive_avg_pool2d(prediction['feature'], (64, 64))
        gray = F.adaptive_avg_pool2d(image[:, :1], (64, 64))
        gray_mean = F.avg_pool2d(gray, 5, 1, 2)
        texture = (F.avg_pool2d(gray.square(), 5, 1, 2) - gray_mean.square()).clamp_min(0).sqrt()
        reliability = .2 + .8 * (texture / .12).clamp(0, 1)
        reliability_mass = reliability.sum()
        reliability96 = F.interpolate(reliability, (96, 96), mode='bilinear', align_corners=False)
        shuffled_source_index = torch.randperm(64 * 64,
            generator=torch.Generator().manual_seed(133000 + index)).cuda()
        shuffled_source = source_feature.flatten(2)[:, :, shuffled_source_index].reshape(1, 64, 64, 64)
        energy, components, surfaces, supports = {arm: [] for arm in arms[1:]}, [], [], []

        for part in ids.split(2):
            count = len(part)
            mapped = model.map(prediction, offsets, (part // 2)[None], (part % 2)[None],
                (96, 96), atlas, weights, source_shape=(256, 256))
            surfaces.append(mapped['centre_surface_ccf_ap_dv_ml_um'][0])
            coordinates = mapped['coordinates'][0].permute(0, 1, 4, 2, 3)
            coordinates = F.interpolate(coordinates.reshape(count, -1, 96, 96),
                (64, 64), mode='bilinear', align_corners=False)
            coordinates = coordinates.reshape(count, offsets.shape[-1], 3, 64, 64).permute(0, 1, 3, 4, 2)
            rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
                (0., 0., 0.), (25., 25., 25.), weights.expand(count, -1))
            support = rendered[:, 1:2].clamp(0, 1)
            supports.append(support)
            intensity = rendered[:, :1] / support.clamp_min(1e-4)
            pair = torch.cat((intensity, support), 1)
            target = model.atlas_encoder(pair)
            zero_target = model.atlas_encoder(torch.cat((torch.zeros_like(intensity), support), 1))
            scrambled = intensity.clone()
            for slot in range(count):
                pixels = torch.where((support[slot, 0] > .5).flatten())[0]
                order = torch.randperm(len(pixels), generator=torch.Generator().manual_seed(
                    133000000 + index * 160 + int(part[slot]))).cuda()
                flat = scrambled[slot, 0].flatten()
                flat[pixels] = intensity[slot, 0].flatten()[pixels[order]]
            scrambled_target = model.atlas_encoder(torch.cat((scrambled, support), 1))

            local = mapped['local_displacement_um'][0]
            _, _, basis = full_frame_state_to_components(mapped['state'][0])
            xstep = basis[..., :, 0].norm(dim=-1) / 96
            ystep = basis[..., :, 1].norm(dim=-1) / 96
            magnitude = local.norm(dim=1) / 1000
            displacement = ((magnitude * reliability96[0, 0]).sum((-2, -1))
                / reliability96.sum())
            xweight = .5 * (reliability96[0, 0, :, 1:] + reliability96[0, 0, :, :-1])
            yweight = .5 * (reliability96[0, 0, 1:, :] + reliability96[0, 0, :-1, :])
            xstrain = (local[..., 1:] - local[..., :-1]).norm(dim=1) / xstep[:, None, None]
            ystrain = (local[..., 1:, :] - local[..., :-1, :]).norm(dim=1) / ystep[:, None, None]
            strain = ((xstrain * xweight).sum((-2, -1)) + (ystrain * yweight).sum((-2, -1))) / (
                xweight.sum() + yweight.sum())
            coverage = (reliability * (1 - support)).sum((1, 2, 3)) / reliability_mass
            part_components = {'coverage': coverage, 'displacement_mm': displacement,
                'strain': strain}
            for arm, image_descriptor, atlas_descriptor in (
                ('full', source_feature, target), ('atlas_zero', source_feature, zero_target),
                ('atlas_shuffle', source_feature, scrambled_target),
                ('source_shuffle', shuffled_source, target)):
                cosine = (F.normalize(image_descriptor, dim=1) *
                    F.normalize(atlas_descriptor, dim=1)).sum(1, keepdim=True)
                appearance = (reliability * support * ((1 - cosine) / 2).clamp(0, 1)).sum(
                    (1, 2, 3)) / reliability_mass
                energy[arm].append(appearance + coverage + .25 * displacement + .25 * strain)
                part_components[arm + '_appearance'] = appearance
            components.append(part_components)

        energy = {arm: torch.cat(values) for arm, values in energy.items()}
        component_values = {key: torch.cat([part[key] for part in components]).tolist()
            for key in components[0]}
        prior_values = prior[0, ids]
        scores = {'prior': prior_values,
            **{arm: prior_values - 3 * value for arm, value in energy.items()}}
        chosen = {arm: int(values.argmax()) for arm, values in scores.items()}
        support = torch.cat(supports)
        surfaces = torch.cat(surfaces)

        # Frozen synthetic labels enter here, only for error and conditional-pair assessment.
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            true_surface = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
        proposed = rigid_points_090(prediction['state'][:, ids // 2], (ids % 2)[None], chart)[0]
        rigid_mm = ((proposed - target_rigid).norm(dim=-1)[:, valid.flatten()].mean(-1) / 1000).tolist()
        upsampled = F.interpolate(surfaces.permute(0, 3, 1, 2), (256, 256),
            mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        mapped_mm = ((upsampled - true_surface).norm(dim=-1)[:, valid].mean(-1) / 1000).tolist()

        best = int(np.argmin(mapped_mm))
        mask = support[:, 0] > .5
        overlap = (mask & mask[best]).sum((-2, -1)).float()
        dice = 2 * overlap / (mask.sum((-2, -1)) + mask[best].sum()).clamp_min(1)
        area_gap = (mask.float().mean((-2, -1)) - mask[best].float().mean()).abs()
        wrong = ((torch.as_tensor(mapped_mm, device='cuda') > mapped_mm[best] + 1.)
            & (dice >= .85) & (area_gap <= .08) & (overlap >= 32))
        pair = {'best_slot': best, 'best_mapped_mm': mapped_mm[best],
            'near_candidate': bool(mapped_mm[best] <= 1.5),
            'support_matched_wrong_count': int(wrong.sum()), 'status': 'no_near_candidate'}
        if pair['near_candidate']:
            pair['status'] = 'no_support_matched_wrong'
            if bool(wrong.any()):
                wrong_slot = int(torch.as_tensor(mapped_mm, device='cuda')
                    .masked_fill(~wrong, torch.inf).argmin())
                pair.update(status='scored', wrong_slot=wrong_slot,
                    wrong_mapped_mm=mapped_mm[wrong_slot],
                    support_dice=float(dice[wrong_slot]),
                    support_area_gap=float(area_gap[wrong_slot]),
                    common_supported_pixels=int(overlap[wrong_slot]),
                    margin={arm: float(value[best] - value[wrong_slot])
                        for arm, value in scores.items()})

        row.update(status='scored', valid_pixels=int(valid.sum()),
            nearest_cardinal_angle_deg=float(np.degrees(np.arccos(np.clip(
                np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
            beam_branch_ids=ids.tolist(), beam_prior_logit=prior_values.tolist(),
            beam_mapped_mm=mapped_mm, beam_rigid_mm=rigid_mm,
            beam_best_mapped_mm=min(mapped_mm), beam_best_rigid_mm=min(rigid_mm),
            beam_support_fraction=support.mean((1, 2, 3)).tolist(),
            energy_components=component_values,
            beam_energy={arm: values.tolist() for arm, values in energy.items()},
            beam_score={arm: values.tolist() for arm, values in scores.items()},
            selected={arm: {'slot': slot, 'branch_id': int(ids[slot]),
                'mapped_mm': mapped_mm[slot], 'rigid_mm': rigid_mm[slot],
                'mapped_le_1p5': mapped_mm[slot] <= 1.5}
                for arm, slot in chosen.items()}, pair=pair)
        rows.append(row)
        if (index + 1) % 32 == 0:
            print(json.dumps({'event': 'panel_milestone', 'records': index + 1,
                'eligible_scored': sum(item['status'] == 'scored' for item in rows)}), flush=True)

assert len(rows) == 256 and sum(row['status'] == 'scored' for row in rows) == 248
scored = [row for row in rows if row['status'] == 'scored']
plans = sorted({row['subject_plan_id'] for row in scored})
summary = {'sections': 256, 'eligible_scored': 248, 'ineligible_unscored': 8,
    'synthetic_plans': len(plans), 'arms': {},
    'conditional_pairs': sum(row['pair']['status'] == 'scored' for row in scored)}
for arm in arms:
    selected = [row['selected'][arm] for row in scored]
    pair_rows = [row for row in scored if row['pair']['status'] == 'scored']
    summary['arms'][arm] = {
        'section_equal_mapped_mm': float(np.mean([item['mapped_mm'] for item in selected])),
        'section_equal_rigid_mm': float(np.mean([item['rigid_mm'] for item in selected])),
        'section_equal_mapped_le_1p5': float(np.mean([item['mapped_le_1p5'] for item in selected])),
        'plan_equal_mapped_mm': float(np.mean([np.mean([row['selected'][arm]['mapped_mm']
            for row in scored if row['subject_plan_id'] == plan]) for plan in plans])),
        'plan_equal_rigid_mm': float(np.mean([np.mean([row['selected'][arm]['rigid_mm']
            for row in scored if row['subject_plan_id'] == plan]) for plan in plans])),
        'plan_equal_mapped_le_1p5': float(np.mean([np.mean([
            row['selected'][arm]['mapped_le_1p5'] for row in scored
            if row['subject_plan_id'] == plan]) for plan in plans])),
        'changed_from_prior': sum(row['selected'][arm]['slot'] != row['selected']['prior']['slot']
            for row in scored),
        'pair_wins': sum(row['pair']['margin'][arm] > 0 for row in pair_rows),
        'pair_win_fraction': (float(np.mean([row['pair']['margin'][arm] > 0
            for row in pair_rows])) if pair_rows else None),
        'by_plan': {plan: {'sections': sum(row['subject_plan_id'] == plan for row in scored),
            'selected_mapped_mm': float(np.mean([row['selected'][arm]['mapped_mm']
                for row in scored if row['subject_plan_id'] == plan])),
            'selected_rigid_mm': float(np.mean([row['selected'][arm]['rigid_mm']
                for row in scored if row['subject_plan_id'] == plan])),
            'selected_mapped_le_1p5': float(np.mean([row['selected'][arm]['mapped_le_1p5']
                for row in scored if row['subject_plan_id'] == plan]))} for plan in plans}}
summary['beam_bound'] = {
    'section_equal_best_mapped_mm': float(np.mean([row['beam_best_mapped_mm'] for row in scored])),
    'section_equal_best_rigid_mm': float(np.mean([row['beam_best_rigid_mm'] for row in scored])),
    'section_equal_any_mapped_le_1p5': float(np.mean([
        row['beam_best_mapped_mm'] <= 1.5 for row in scored]))}
full, prior_arm, zero = (summary['arms'][name] for name in ('full', 'prior', 'atlas_zero'))
summary['signal_gate'] = {
    'full_plan_mapped_gain_ge_0p15_vs_prior_and_zero': all(
        arm['plan_equal_mapped_mm'] - full['plan_equal_mapped_mm'] >= .15
        for arm in (prior_arm, zero)),
    'full_plan_rigid_gain_ge_0p15_vs_prior_and_zero': all(
        arm['plan_equal_rigid_mm'] - full['plan_equal_rigid_mm'] >= .15
        for arm in (prior_arm, zero)),
    'full_plan_near_gain_ge_0p05_vs_prior_and_zero': all(
        full['plan_equal_mapped_le_1p5'] - arm['plan_equal_mapped_le_1p5'] >= .05
        for arm in (prior_arm, zero)),
    'full_pair_win_ge_0p60_and_plus_0p10_vs_zero': bool(
        summary['conditional_pairs'] >= 64 and full['pair_win_fraction'] is not None
        and full['pair_win_fraction'] >= .60
        and full['pair_win_fraction'] - zero['pair_win_fraction'] >= .10)}
summary['signal_gate']['passed'] = all(summary['signal_gate'].values())

config = {'protocol_sha256': sha(protocol), 'evaluator_source_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in (
        'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_geometry.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
        'global_plane_matcher_120.py')},
    'checkpoint_sha256': sha(checkpoint), 'training_completion_sha256': sha(train / 'completed.json'),
    'panel_completion_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'allen_atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'allen_template_raw_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'allen_annotation_raw_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'beam': '132 exact blind16', 'map_side': 96, 'score_side': 64,
    'energy': 'appearance+coverage+0.25*displacement_mm+0.25*strain',
    'selection': 'direct_log_prior-3*energy',
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'config_sha256': sha(out / 'config.json'), 'protocol_sha256': config['protocol_sha256'],
    'evaluator_source_sha256': config['evaluator_source_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'panel_completion_sha256': config['panel_completion_sha256'],
    'sections': 256, 'eligible_scored': 248, 'ineligible_unscored': 8,
    'signal_gate_passed': summary['signal_gate']['passed'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'signal_gate': summary['signal_gate'],
    'full': summary['arms']['full'], 'prior': summary['arms']['prior'],
    'atlas_zero': summary['arms']['atlas_zero']}), flush=True)
