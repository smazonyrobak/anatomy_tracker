"""Independent 134 descriptor assay on frozen synthetic DEV sections."""

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
import torch.nn as nn
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


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/BEAM_FIT_DESCRIPTOR_134_EVAL_PROTOCOL_20261010.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_checkpoint = parent_dir / 'joint_step_02000.pt'
train = root / 'runs/beam_fit_descriptor_134'
out = root / 'runs/beam_fit_descriptor_134_eval'
panels = {'fresh_v4_132': root / 'data/fresh_v4_pose_dev_panel_132',
          'older_v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
steps = (0, 1000, 2000, 3000, 4000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert sha(parent_dir / 'draws.jsonl') == parent_done['draws_sha256']
assert sha(parent_dir / 'training.jsonl') == parent_done['training_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert sha(train / 'config.json') == train_done['config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256']
assert sha(parent_checkpoint) == train_config['parent_checkpoint_sha256']
assert train_done['parent_checkpoint_sha256'] == train_config['parent_checkpoint_sha256']
assert train_done['protocol_sha256'] == train_config['protocol_sha256']
assert train_done['source_sha256'] == train_config['source_sha256']
assert sha(source.parent / 'docs/publication/BEAM_FIT_DESCRIPTOR_134_PROTOCOL_20261010.md') == train_config['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert train_done['updates'] == train_done['accepted_presentations'] == 4000
assert not any(train_done.get(key, False) for key in ('calibrated', 'expert_real_truth_used',
    'final_animals_used', 'public_benchmark_used', 'external_pretrained_weights_used'))

panel_records = {}
panel_completions = {}
for name, panel in panels.items():
    done = json.loads((panel / 'completed.json').read_text())
    panel_protocol = json.loads((panel / 'protocol.json').read_text())
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert sha(panel / 'protocol.json') == done['protocol_sha256']
    assert sha(panel / 'records.jsonl') == done['records_sha256']
    assert all(sha(panel / 'source' / file) == digest
        for file, digest in panel_protocol['source_sha256'].items())
    assert len(records) == done['physical_sections'] == 256
    assert sum(row['eligible'] for row in records) == done['eligible']
    assert len({row['panel_physical_section_id'] for row in records}) == 256
    assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[name], panel_completions[name] = records, done
assert panel_completions['fresh_v4_132']['eligible'] == 248
assert panel_completions['older_v3']['eligible'] == 243

atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
assert train_config['allen_atlas_float32_sha256'] == allen.ATLAS_FLOAT32_RECEIPT_V6['sha256']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

heads = {}
for step in steps:
    checkpoint = train / f'descriptor_step_{step:05d}.pt'
    assert sha(checkpoint) == train_done['checkpoint_sha256'][str(step)]
    pair = nn.ModuleDict({arm: nn.ModuleDict({
        'source': nn.Sequential(nn.Conv2d(65, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=4, dilation=4)),
        'atlas': nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=4, dilation=4))})
        for arm in ('full', 'support_only')}).cuda().eval().requires_grad_(False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['step'] == step and saved['config'] == train_config
    assert saved['parent_checkpoint_sha256'] == train_config['parent_checkpoint_sha256']
    assert saved['calibrated'] is False
    pair.load_state_dict(saved['heads'], strict=True)
    heads[step] = pair
    del saved

axis = torch.arange(256, device='cuda', dtype=torch.float32) / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
score_keys = ['direct'] + [f'step_{step:05d}_{arm}_{mode}' for step in steps
    for arm in ('full', 'support_only') for mode in ('fit_only', 'prior_plus_fit')]
rows = []
with torch.inference_mode():
    for panel_name, panel in panels.items():
        for index, record in enumerate(panel_records[panel_name]):
            appearance = record['appearance_mode']
            exposure = (record['provenance']['one_shot_slide_artifacts_v4']['low_exposure']
                if panel_name == 'fresh_v4_132' else None)
            row = {'panel': panel_name, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'subject_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': appearance, 'low_exposure': exposure,
                'panel_file_sha256': record['sha256'], 'eligible': record['eligible']}
            if not record['eligible']:
                row['status'] = 'frozen_ineligible_unscored'
                rows.append(row)
                continue

            # Labels are not loaded until all descriptor and direct scores exist.
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            prediction = model.predict(image)
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
            ids = beam[0]
            prior_values = prior[0, ids]
            feature = F.adaptive_avg_pool2d(prediction['feature'], (64, 64))
            gray = F.adaptive_avg_pool2d(image[:, :1], (64, 64))
            gray_mean = F.avg_pool2d(gray, 5, 1, 2)
            texture = (F.avg_pool2d(gray.square(), 5, 1, 2) - gray_mean.square()).clamp_min(0).sqrt()
            reliability = .2 + .8 * (texture / .12).clamp(0, 1)
            reliability_mass = reliability.sum()
            reliability96 = F.interpolate(reliability, (96, 96),
                mode='bilinear', align_corners=False)
            atlas_pairs, surfaces, fixed_costs = [], [], []
            for part in ids.split(2):
                count = len(part)
                mapped = model.map(prediction, offsets, (part // 2)[None],
                    (part % 2)[None], (96, 96), atlas, weights, source_shape=(256, 256))
                surfaces.append(mapped['centre_surface_ccf_ap_dv_ml_um'][0])
                coordinates = mapped['coordinates'][0].permute(0, 1, 4, 2, 3)
                coordinates = F.interpolate(coordinates.reshape(count, -1, 96, 96),
                    (64, 64), mode='bilinear', align_corners=False)
                coordinates = coordinates.reshape(count, offsets.shape[-1], 3,
                    64, 64).permute(0, 1, 3, 4, 2)
                rendered = render_finite_thickness_coordinate_grid(atlas, coordinates,
                    (0., 0., 0.), (25., 25., 25.), weights.expand(count, -1))
                support = rendered[:, 1:2].clamp(0, 1)
                atlas_pairs.append(torch.cat((rendered[:, :1] / support.clamp_min(1e-4),
                    support), 1))
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
                strain = ((xstrain * xweight).sum((-2, -1)) +
                    (ystrain * yweight).sum((-2, -1))) / (xweight.sum() + yweight.sum())
                coverage = (reliability * (1 - support)).sum((1, 2, 3)) / reliability_mass
                fixed_costs.append(coverage + .25 * displacement + .25 * strain)
            atlas_pair = torch.cat(atlas_pairs)
            support = atlas_pair[:, 1:2]
            fixed_cost = torch.cat(fixed_costs)
            surfaces = torch.cat(surfaces)
            source_input = torch.cat((feature, gray), 1)
            scores = {'direct': prior_values}
            energy_values = {}
            for step in steps:
                for arm in ('full', 'support_only'):
                    target_input = (atlas_pair if arm == 'full' else torch.cat((
                        torch.zeros_like(atlas_pair[:, :1]), support), 1))
                    source_descriptor = F.normalize(heads[step][arm]['source'](source_input), dim=1)
                    atlas_descriptor = F.normalize(heads[step][arm]['atlas'](target_input), dim=1)
                    cosine = (source_descriptor * atlas_descriptor).sum(1, keepdim=True)
                    appearance_cost = (reliability * support * ((1 - cosine) / 2).clamp(0, 1)
                        ).sum((1, 2, 3)) / reliability_mass
                    energy = appearance_cost + fixed_cost
                    prefix = f'step_{step:05d}_{arm}'
                    energy_values[prefix] = energy.tolist()
                    scores[prefix + '_fit_only'] = -3 * energy
                    scores[prefix + '_prior_plus_fit'] = prior_values - 3 * energy
            chosen = {key: int(values.argmax()) for key, values in scores.items()}

            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                true_surface = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            assert int(valid.sum()) == record['valid_pixels']
            target_rigid = rigid_points_090(truth, truth_reflection, chart)[0]
            proposed = rigid_points_090(prediction['state'][:, ids // 2],
                (ids % 2)[None], chart)[0]
            rigid_mm = ((proposed - target_rigid).norm(dim=-1)[:, valid.flatten()]
                .mean(-1) / 1000).tolist()
            upsampled = F.interpolate(surfaces.permute(0, 3, 1, 2), (256, 256),
                mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            mapped_mm = ((upsampled - true_surface).norm(dim=-1)[:, valid]
                .mean(-1) / 1000).tolist()
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
                        margin={key: float(value[best] - value[wrong_slot])
                            for key, value in scores.items()})
            row.update(status='scored', valid_pixels=int(valid.sum()),
                nearest_cardinal_angle_deg=float(np.degrees(np.arccos(np.clip(
                    np.max(np.abs(record['plane_normal_ap_dv_ml'])), 0, 1)))),
                beam_branch_ids=ids.tolist(), beam_prior_logit=prior_values.tolist(),
                beam_mapped_mm=mapped_mm, beam_rigid_mm=rigid_mm,
                beam_best_mapped_mm=min(mapped_mm), beam_best_rigid_mm=min(rigid_mm),
                beam_support_fraction=support.mean((1, 2, 3)).tolist(),
                beam_fixed_cost=fixed_cost.tolist(), beam_energy=energy_values,
                beam_score={key: value.tolist() for key, value in scores.items()},
                selected={key: {'slot': slot, 'branch_id': int(ids[slot]),
                    'mapped_mm': mapped_mm[slot], 'rigid_mm': rigid_mm[slot],
                    'mapped_le_1p5': mapped_mm[slot] <= 1.5}
                    for key, slot in chosen.items()}, pair=pair)
            rows.append(row)
            if (index + 1) % 32 == 0:
                print(json.dumps({'event': 'panel_milestone', 'panel': panel_name,
                    'records': index + 1,
                    'eligible_scored': sum(item['panel'] == panel_name and item['status'] == 'scored'
                        for item in rows)}), flush=True)

assert len(rows) == 512
summary = {'panels': {}, 'primary_checkpoint': 4000,
    'primary_score': 'prior_plus_fit', 'intermediate_checkpoints_descriptive_only': True}
for panel_name, records in panel_records.items():
    panel_rows = [row for row in rows if row['panel'] == panel_name]
    scored = [row for row in panel_rows if row['status'] == 'scored']
    pair_rows = [row for row in scored if row['pair']['status'] == 'scored']
    plans = sorted({row['subject_plan_id'] for row in scored})
    modes = sorted({row['appearance_mode'] for row in scored})
    exposure_groups = ([('low', True), ('nonlow', False)]
        if panel_name == 'fresh_v4_132' else [])
    panel_summary = {'sections': 256, 'eligible_scored': len(scored),
        'ineligible_unscored': 256 - len(scored), 'synthetic_plans': len(plans),
        'conditional_pairs': len(pair_rows),
        'pair_status': {status: sum(row['pair']['status'] == status for row in scored)
            for status in ('scored', 'no_near_candidate', 'no_support_matched_wrong')},
        'beam_bound': {
            'section_equal_best_mapped_mm': float(np.mean([row['beam_best_mapped_mm'] for row in scored])),
            'section_equal_best_rigid_mm': float(np.mean([row['beam_best_rigid_mm'] for row in scored])),
            'section_equal_any_mapped_le_1p5': float(np.mean([
                row['beam_best_mapped_mm'] <= 1.5 for row in scored]))}, 'scores': {}}
    for key in score_keys:
        selected = [row['selected'][key] for row in scored]
        by_plan = {plan: {'sections': sum(row['subject_plan_id'] == plan for row in scored),
            'selected_mapped_mm': float(np.mean([row['selected'][key]['mapped_mm']
                for row in scored if row['subject_plan_id'] == plan])),
            'selected_rigid_mm': float(np.mean([row['selected'][key]['rigid_mm']
                for row in scored if row['subject_plan_id'] == plan])),
            'selected_mapped_le_1p5': float(np.mean([row['selected'][key]['mapped_le_1p5']
                for row in scored if row['subject_plan_id'] == plan]))} for plan in plans}
        by_mode = {mode: {'sections': sum(row['appearance_mode'] == mode for row in scored),
            'selected_mapped_mm': float(np.mean([row['selected'][key]['mapped_mm']
                for row in scored if row['appearance_mode'] == mode])),
            'selected_mapped_le_1p5': float(np.mean([row['selected'][key]['mapped_le_1p5']
                for row in scored if row['appearance_mode'] == mode]))} for mode in modes}
        by_exposure = {name: {'sections': sum(row['low_exposure'] == state for row in scored),
            'selected_mapped_mm': float(np.mean([row['selected'][key]['mapped_mm']
                for row in scored if row['low_exposure'] == state])),
            'selected_mapped_le_1p5': float(np.mean([row['selected'][key]['mapped_le_1p5']
                for row in scored if row['low_exposure'] == state]))}
            for name, state in exposure_groups}
        panel_summary['scores'][key] = {
            'section_equal_mapped_mm': float(np.mean([item['mapped_mm'] for item in selected])),
            'section_equal_rigid_mm': float(np.mean([item['rigid_mm'] for item in selected])),
            'section_equal_mapped_le_1p5': float(np.mean([item['mapped_le_1p5'] for item in selected])),
            'plan_equal_mapped_mm': float(np.mean([item['selected_mapped_mm'] for item in by_plan.values()])),
            'plan_equal_rigid_mm': float(np.mean([item['selected_rigid_mm'] for item in by_plan.values()])),
            'plan_equal_mapped_le_1p5': float(np.mean([
                item['selected_mapped_le_1p5'] for item in by_plan.values()])),
            'changed_from_direct': sum(row['selected'][key]['slot'] != row['selected']['direct']['slot']
                for row in scored),
            'pair_wins': sum(row['pair']['margin'][key] > 0 for row in pair_rows),
            'pair_win_fraction': (float(np.mean([row['pair']['margin'][key] > 0
                for row in pair_rows])) if pair_rows else None),
            'by_plan': by_plan, 'by_mode': by_mode, 'by_exposure': by_exposure}
    summary['panels'][panel_name] = panel_summary

v4 = summary['panels']['fresh_v4_132']
v3 = summary['panels']['older_v3']
full_key = 'step_04000_full_prior_plus_fit'
support_key = 'step_04000_support_only_prior_plus_fit'
full, support, direct = (v4['scores'][key] for key in (full_key, support_key, 'direct'))
signal = {
    'full_mapped_gain_ge_0p15_vs_support_and_direct': all(
        arm['plan_equal_mapped_mm'] - full['plan_equal_mapped_mm'] >= .15
        for arm in (support, direct)),
    'full_near_gain_ge_0p05_vs_support_and_direct': all(
        full['plan_equal_mapped_le_1p5'] - arm['plan_equal_mapped_le_1p5'] >= .05
        for arm in (support, direct)),
    'conditional_pairs_ge_64': v4['conditional_pairs'] >= 64,
    'full_pair_wins_ge_0p60_and_plus_0p10_vs_support': bool(
        full['pair_win_fraction'] is not None and support['pair_win_fraction'] is not None
        and full['pair_win_fraction'] >= .60
        and full['pair_win_fraction'] - support['pair_win_fraction'] >= .10)}
signal['passed'] = all(signal.values())
v3_full, v3_direct = (v3['scores'][key] for key in (full_key, 'direct'))
retention = {
    'mapped_no_worse_than_direct_plus_0p15': (
        v3_full['plan_equal_mapped_mm'] <= v3_direct['plan_equal_mapped_mm'] + .15),
    'near_no_lower_than_direct_minus_0p05': (
        v3_full['plan_equal_mapped_le_1p5'] >= v3_direct['plan_equal_mapped_le_1p5'] - .05)}
retention['passed'] = all(retention.values())
summary['primary_v4_signal_gate'] = signal
summary['v3_retention_guard'] = retention
summary['pose_feedback_experiment_justified'] = signal['passed'] and retention['passed']

config = {'protocol_sha256': sha(protocol), 'evaluator_source_sha256': sha(__file__),
    'source_sha256': {name: sha(source / name) for name in (
        'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_geometry.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py',
        'global_plane_matcher_120.py')},
    'train_completion_sha256': sha(train / 'completed.json'),
    'train_config_sha256': sha(train / 'config.json'),
    'checkpoint_sha256': {str(step): sha(train / f'descriptor_step_{step:05d}.pt')
        for step in steps},
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'panel_completion_sha256': {name: sha(panel / 'completed.json') for name, panel in panels.items()},
    'panel_protocol_sha256': {name: sha(panel / 'protocol.json') for name, panel in panels.items()},
    'panel_records_sha256': {name: sha(panel / 'records.jsonl') for name, panel in panels.items()},
    'allen_atlas_float32_sha256': allen.ATLAS_FLOAT32_RECEIPT_V6['sha256'],
    'allen_template_raw_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'allen_annotation_raw_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'beam': '132 exact blind16', 'map_side': 96, 'descriptor_side': 64,
    'primary_checkpoint': 4000, 'primary_score': 'prior_plus_fit',
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
    'train_completion_sha256': config['train_completion_sha256'],
    'panel_completion_sha256': config['panel_completion_sha256'],
    'sections': 512, 'eligible_scored': sum(row['status'] == 'scored' for row in rows),
    'primary_v4_signal_passed': signal['passed'],
    'v3_retention_passed': retention['passed'],
    'pose_feedback_experiment_justified': summary['pose_feedback_experiment_justified'],
    'calibrated': False, 'expert_real_truth_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'primary_v4_signal': signal,
    'v3_retention': retention,
    'pose_feedback_experiment_justified': summary['pose_feedback_experiment_justified'],
    'v4_full': full, 'v4_support_only': support, 'v4_direct': direct}), flush=True)
