"""Frozen 103 DEV capture readout for the v3 continuation of one-pass 094."""
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

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    ATLAS_FLOAT32_RECEIPT_V6, _decode_and_preprocess_allen_v6,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.spatial_joint_fit_094 import SpatialJointFit094
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_ONE_PASS_POSE_CAPTURE_PILOT_001_PROTOCOL_20261009.md'
parent_run = root / 'runs/spatial_verifier_094_pilot'
parent_file = parent_run / 'joint_step_20000.pt'
train_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
panel = root / 'data/cross_candidate_match_103_fresh_dev_panel_001'
plans = root / 'data/cross_candidate_match_103_fresh_dev_plans_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
reserved_train = root / 'data/joint_v7_reserved_train_images_192_001/completed.json'
out = root / 'runs/v3_one_pass_pose_capture_pilot_001_development_eval'
steps = (0, 2000, 4000, 8000)
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
    grid = ((chart + .5 / side) * 2 - 1)[None].expand(count, -1, -1)
    grid = grid.reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
        padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


def aggregate(rows):
    metrics = ('prefit_rigid_best14_um', 'mapped_best14_um',
        'score_selected_mapped96_um', 'selection_regret_um',
        'selected_mean_supported_bin_count', 'selected_atlas_available_tissue_fraction')
    by_plan = {}
    for plan in plan_ids:
        subset = [row for row in rows if row['synthetic_subject_plan_id'] == plan]
        if subset:
            by_plan[plan] = {'sections': len(subset), **{
                key: float(np.mean([row[key] for row in subset if row[key] is not None]))
                if any(row[key] is not None for row in subset) else None
                for key in metrics}}
    return {'sections': len(rows), 'plans_present': len(by_plan), 'by_plan': by_plan,
        'plan_equal_mean': {key: float(np.mean([
            value[key] for value in by_plan.values() if value[key] is not None]))
            if any(value[key] is not None for value in by_plan.values()) else None
            for key in metrics}}


assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
parent_config = json.loads((parent_run / 'config.json').read_text())
assert parent_receipt['batches'] == 20000
assert sha(parent_file) == parent_receipt['checkpoint_sha256']['20000'] == \
    '81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815'
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert all(sha(source / name) == digest for name, digest in parent_config['source_sha256'].items())
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'real_labels_used', 'external_pretrained_weights_used'))
train_receipt = json.loads((train_run / 'completed.json').read_text())
train_config = json.loads((train_run / 'config.json').read_text())
assert train_receipt['updates'] == 8000 and train_receipt['accepted_synthetic'] == 16000
assert train_receipt['real_train_presentations'] == 8000
assert tuple(train_config['checkpoints']) == steps
assert sha(protocol) == train_config['protocol_sha256'] == train_receipt['protocol_sha256']
assert sha(parent_file) == train_config['parent_094_sha256'] == \
    train_receipt['parent_094_checkpoint_sha256']
assert train_config['source_sha256'] == train_receipt['source_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert sha(train_run / 'config.json') == train_receipt['config_sha256']
assert sha(train_run / 'training.jsonl') == train_receipt['training_sha256']
assert sha(train_run / 'draws.jsonl') == train_receipt['draws_sha256']
assert sha(train_run / 'real_schedule.npy') == train_receipt['real_schedule_sha256']
assert np.load(train_run / 'real_schedule.npy', mmap_mode='r').shape == (8000, 2)
assert not any(train_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
checkpoint_files = {step: train_run / f'joint_step_{step:05d}.pt' for step in steps}
assert all(sha(checkpoint_files[step]) == train_receipt['checkpoint_sha256'][str(step)]
           for step in steps)
draws = [json.loads(line) for line in (train_run / 'draws.jsonl').open()]
assert sum(row['kind'] == 'synthetic_train' for row in draws) == \
    train_receipt['synthetic_draw_attempts']
synthetic_draws = [row for row in draws if row['kind'] == 'synthetic_train' and row['used']]
real_draws = [row for row in draws if row['kind'] == 'real_train']
assert len(synthetic_draws) == len({row['physical_section_id'] for row in synthetic_draws}) == 16000
assert len(real_draws) == 8000

plan_receipt = json.loads((plans / 'completed.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
panel_protocol = json.loads((panel / 'protocol.json').read_text())
assert panel_receipt['physical_sections'] == 256 and panel_receipt['eligible'] == 237
assert panel_receipt['plan_completed_sha256'] == panel_protocol['source_plan_completed_sha256'] == \
    sha(plans / 'completed.json')
assert panel_receipt['protocol_sha256'] == sha(panel / 'protocol.json')
assert panel_receipt['records_sha256'] == sha(panel / 'records.jsonl')
assert panel_protocol['atlas_normalized_receipt'] == ATLAS_FLOAT32_RECEIPT_V6
assert all(sha(plans / name) == digest for name, digest in
           plan_receipt['artifact_sha256'].items())
assert all(sha(panel / 'source' / name) == digest for name, digest in
           panel_protocol['source_sha256'].items())
all_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(all_records) == 256
assert len({row['section_id'] for row in all_records}) == 256
assert all(sha(panel / row['file']) == row['sha256'] for row in all_records)
records = [row for row in all_records if row['eligible']]
assert len(records) == 237 and len({row['section_id'] for row in records}) == 237
plan_ids = sorted({row['synthetic_subject_plan_id'] for row in all_records})
assert len(plan_ids) == 8 and all(any(row['synthetic_subject_plan_id'] == plan
                                 for row in records) for plan in plan_ids)
assert not {row['panel_physical_section_id'] for row in all_records} & \
    {row['physical_section_id'] for row in synthetic_draws}
for key in ('animal_id', 'specimen_id', 'experiment_id'):
    assert not {row[key] for row in all_records} & \
        {row['base_lineage'][key] for row in synthetic_draws}

real_receipt = json.loads((real / 'completed.json').read_text())
assert real_receipt['development_images'] == 64 and real_receipt['development_donors'] == 6
assert real_receipt['animal_specimen_experiment_section_image_disjoint']
assert all(sha(real / name) == digest for name, digest in
           real_receipt['output_sha256'].items())
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(real_records) == 64 and len({row['animal_id'] for row in real_records}) == 6
reserved_receipt = json.loads(reserved_train.read_text())
assert reserved_receipt['training_role'] == 'train_only'
assert not {str(row['animal_id']) for row in real_records} & \
    set(reserved_receipt['donor_receipts'])
assert train_config['real_bindings'][str(reserved_train)] == sha(reserved_train)

parent_state = torch.load(parent_file, map_location='cpu', weights_only=True)
assert parent_state['step'] == 20000 and parent_state['calibrated'] is False
states = {}
for step in steps:
    saved = torch.load(checkpoint_files[step], map_location='cpu', weights_only=True)
    assert saved['step'] == step and saved['config'] == train_config
    assert saved['calibrated'] is False
    states[step] = saved['model']
assert set(states[0]) == set(parent_state['model'])
assert all(torch.equal(states[0][key], value)
           for key, value in parent_state['model'].items())
step0_parent_weight_equal = True

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
fitter = SpatialJointFit094(WholeSliceAtlasFeedback083()).cuda().eval().requires_grad_(False)
fitter.load_state_dict(parent_state['spatial_fit'], strict=True)
del parent_state
flags = torch.tensor([0, 1], device='cuda')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

candidate_rows, section_rows, real_rows = [], [], []
with torch.inference_mode():
    for step in steps:
        model.load_state_dict(states[step], strict=True)
        for index, record in enumerate(records, 1):
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                truth = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                assert bool(arrays['eligible']) and int(valid.sum()) == record['valid_pixels']
            chosen = valid.flatten().nonzero().flatten()
            chosen = chosen[torch.linspace(0, len(chosen) - 1, 1024,
                device='cuda').round().long()]
            chosen_hash = hashlib.sha256(chosen.cpu().numpy().astype('<i4').tobytes()).hexdigest()
            chart = torch.stack((chosen.remainder(side),
                chosen.div(side, rounding_mode='floor')), -1).float() / side
            reference = truth.reshape(-1, 3)[chosen]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            branch = torch.cat((prior[:, :32].topk(8, -1).indices,
                prior[:, 32:].topk(6, -1).indices + 32), -1)
            assert len(set(branch[0].tolist())) == 14
            initial = prediction['state'].gather(1,
                (branch // 2)[..., None].expand(-1, -1, 12))
            normal = np.abs(record['plane_normal_ap_dv_ml'])
            family = ('AP', 'DV', 'ML')[int(np.argmax(normal))]
            angle_deg = math.degrees(math.acos(min(1., float(np.max(normal)))))
            angle_bin = ('0-15' if angle_deg < 15 else '15-30' if angle_deg < 30
                         else '30-45' if angle_deg < 45 else '45-54.7')
            common = {'step': step, 'section_id': record['section_id'],
                'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
                'experiment_id': record['experiment_id'],
                'appearance_mode': record['appearance_mode'],
                'angle_family': family, 'nearest_axis_angle_deg': angle_deg,
                'angle_bin': angle_bin, 'valid_fraction': record['valid_pixels'] / (side * side),
                'panel_file': record['file'], 'panel_file_sha256': record['sha256'],
                'sample_indices_sha256': chosen_hash}
            section_candidates = []
            for start in range(0, 14, 2):
                pair = branch[:, start:start + 2]
                reflected = pair % 2
                state = initial[:, start:start + 2]
                prefit_error = (rigid_points_090(state, reflected, chart)
                    - reference[None]).norm(dim=-1).mean(-1)[0]
                fitted = fitter(prediction['feature'], state, reflected, atlas,
                    offsets, weights, source_shape=(side, side))
                mapped_prediction = {**prediction, 'state': fitted['state'],
                    'log_mass': prediction['log_mass'].gather(1, pair // 2),
                    'reflection_logit': prediction['reflection_logit'].gather(1, pair // 2)}
                mapped = model.map(mapped_prediction, offsets,
                    torch.arange(2, device='cuda')[None], reflected, (96, 96),
                    atlas, weights, feature_side=96, source_shape=(side, side),
                    spatial_evidence=fitted['spatial_evidence'])
                mapped_error = (at_chart(mapped['centre_surface_ccf_ap_dv_ml_um'], chart)
                    - reference[None]).norm(dim=-1).mean(-1)
                local_warp = mapped['local_displacement_um'] / 1000
                dx = local_warp[..., 1:] - local_warp[..., :-1]
                dy = local_warp[..., 1:, :] - local_warp[..., :-1, :]
                warp_cost = (local_warp.square().mean((-3, -2, -1)) + .1 *
                    (dx.square().mean((-3, -2, -1)) + dy.square().mean((-3, -2, -1))))
                fit_temperature = fitter.log_fit_temperature.clamp(
                    math.log(.5), math.log(5.)).exp()
                score = prior.gather(1, pair) - fitted['fit_energy'] / fit_temperature \
                    - .05 * warp_cost
                predicted_tissue = fitted['validity_logit'].sigmoid() >= .5
                supported = fitted['fine_match_support'] >= .5
                support_count = supported.sum(2)
                for local in range(2):
                    tissue = predicted_tissue[0]
                    site_count = int(tissue.sum())
                    available = support_count[0, local] > 0
                    section_candidates.append({**common, 'beam_slot': start + local,
                        'branch_id': int(pair[0, local]),
                        'initial_state': state[0, local].tolist(),
                        'fitted_state': fitted['state'][0, local].tolist(),
                        'prefit_rigid_error_um': float(prefit_error[local]),
                        'mapped96_error_um': float(mapped_error[local]),
                        'old094_score': float(score[0, local]),
                        'prior_score': float(prior[0, pair[0, local]]),
                        'fit_energy': float(fitted['fit_energy'][0, local]),
                        'warp_cost': float(warp_cost[0, local]),
                        'predicted_tissue_sites': site_count,
                        'atlas_available_tissue_fraction': (
                            float((available & tissue).sum() / site_count) if site_count else None),
                        'mean_supported_bin_count': (
                            float(support_count[0, local][tissue].float().mean())
                            if site_count else None)})
            assert len(section_candidates) == 14
            best = min(range(14), key=lambda slot: section_candidates[slot]['mapped96_error_um'])
            prefit_best = min(row['prefit_rigid_error_um'] for row in section_candidates)
            selected = max(range(14), key=lambda slot: section_candidates[slot]['old094_score'])
            for row in section_candidates:
                row['score_selected'] = row['beam_slot'] == selected
                row['physical_best'] = row['beam_slot'] == best
                candidate_rows.append(row)
            selected_row = section_candidates[selected]
            support_count = selected_row['mean_supported_bin_count']
            support_stratum = ('none' if support_count is None else '<32' if support_count < 32
                else '32-96' if support_count < 96 else '>=96')
            section_rows.append({**common,
                'beam_branch_ids': branch[0].tolist(),
                'prefit_rigid_best14_um': prefit_best,
                'mapped_best14_um': section_candidates[best]['mapped96_error_um'],
                'score_selected_mapped96_um': selected_row['mapped96_error_um'],
                'selection_regret_um': (selected_row['mapped96_error_um']
                                        - section_candidates[best]['mapped96_error_um']),
                'score_selected_slot': selected, 'mapped_best_slot': best,
                'selected_mean_supported_bin_count': support_count,
                'selected_support_stratum': support_stratum,
                'selected_atlas_available_tissue_fraction':
                    selected_row['atlas_available_tissue_fraction'],
                'predicted_tissue_sites': selected_row['predicted_tissue_sites']})
            if index % 32 == 0:
                print(json.dumps({'step': step, 'eligible_sections_done': index,
                    'eligible_sections_total': len(records)}), flush=True)

        for record in real_records:
            row_index = record['array_row_index']
            image = np.concatenate((real_images[row_index].astype(np.float32),
                np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(real_affines[row_index], device='cuda', dtype=torch.float32)
            reference = (affine[:, 2] + 192 * corners[:, :1] * affine[:, 0]
                         + 192 * corners[:, 1:] * affine[:, 1])
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            branch_id = int(prior.argmax())
            proposed = rigid_points_090(prediction['state'][:, branch_id // 2],
                torch.tensor([branch_id % 2], device='cuda'), corners)[0]
            real_rows.append({'step': step, 'set': 'weak_allen_development',
                **{key: record[key] for key in ('animal_id', 'specimen_id',
                    'experiment_id', 'section_id', 'array_row_index')},
                'branch_id': branch_id,
                'prior_selected_five_point_error_um': float((proposed - reference).norm(dim=-1).mean()),
                'label_role': 'inherited weak Allen affine, not expert pose'})

assert len(section_rows) == len(records) * len(steps)
assert len(candidate_rows) == len(records) * len(steps) * 14
assert len(real_rows) == len(real_records) * len(steps)
sample_bindings = {}
for row in section_rows:
    previous = sample_bindings.setdefault(row['section_id'], row['sample_indices_sha256'])
    assert previous == row['sample_indices_sha256']
baseline_support = {row['section_id']: row['selected_support_stratum']
                    for row in section_rows if row['step'] == 0}
for row in section_rows:
    row['step0_support_stratum'] = baseline_support[row['section_id']]
for row in candidate_rows:
    row['step0_support_stratum'] = baseline_support[row['section_id']]

summary = {'eligible_sections_per_step': len(records),
    'ineligible_sections_excluded': panel_receipt['ineligible'],
    'candidates_per_step': len(records) * 14,
    'real_weak_development_sections_per_step': len(real_records),
    'real_weak_development_donors': 6,
    'step0_parent_weight_equal': step0_parent_weight_equal,
    'steps': {}}
for step in steps:
    subset = [row for row in section_rows if row['step'] == step]
    groups = {'all': subset}
    for mode in panel_protocol['modes']:
        groups[f'appearance/{mode}'] = [row for row in subset
            if row['appearance_mode'] == mode]
    for family in ('AP', 'DV', 'ML'):
        groups[f'angle_family/{family}'] = [row for row in subset
            if row['angle_family'] == family]
    for angle in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'angle_bin/{angle}'] = [row for row in subset
            if row['angle_bin'] == angle]
    for stratum in ('none', '<32', '32-96', '>=96'):
        groups[f'step0_support_count/{stratum}'] = [row for row in subset
            if row['step0_support_stratum'] == stratum]
    donor_mean = {donor: float(np.mean([
        row['prior_selected_five_point_error_um'] for row in real_rows
        if row['step'] == step and str(row['animal_id']) == donor]))
        for donor in sorted({str(row['animal_id']) for row in real_records})}
    summary['steps'][str(step)] = {'synthetic': {name: aggregate(group)
        for name, group in groups.items()},
        'real_weak_five_point_by_donor_um': donor_mean,
        'real_weak_five_point_donor_equal_um': float(np.mean(list(donor_mean.values())))}

zero = summary['steps']['0']
step_gates = {}
for step in steps[1:]:
    current = summary['steps'][str(step)]
    raw0 = zero['synthetic']['appearance/raw']
    raw = current['synthetic']['appearance/raw']
    plan_improves = {plan: (plan in raw['by_plan'] and plan in raw0['by_plan'] and
        raw['by_plan'][plan]['mapped_best14_um'] < raw0['by_plan'][plan]['mapped_best14_um'])
        for plan in plan_ids}
    modes_and_families = ([f'appearance/{mode}' for mode in ('exact_black', 'imperfect_brush')]
        + [f'angle_family/{family}' for family in ('AP', 'DV', 'ML')])
    nonregression = {name: (
        current['synthetic'][name]['plan_equal_mean']['score_selected_mapped96_um'] <=
        zero['synthetic'][name]['plan_equal_mean']['score_selected_mapped96_um'] + 200)
        for name in modes_and_families}
    real_nonregression = {donor: (error <=
        zero['real_weak_five_point_by_donor_um'][donor] + 200)
        for donor, error in current['real_weak_five_point_by_donor_um'].items()}
    conditions = {'raw_best14_gain_at_least_0.35_mm': (
        raw0['plan_equal_mean']['mapped_best14_um'] -
        raw['plan_equal_mean']['mapped_best14_um'] >= 350),
        'raw_best14_improves_in_at_least_six_plans': sum(plan_improves.values()) >= 6,
        'overall_score_selected_nonregression': (
            current['synthetic']['all']['plan_equal_mean']['score_selected_mapped96_um'] <=
            zero['synthetic']['all']['plan_equal_mean']['score_selected_mapped96_um']),
        'no_brush_or_axis_family_regression_over_0.20_mm': all(nonregression.values()),
        'no_weak_real_donor_regression_over_0.20_mm': all(real_nonregression.values())}
    step_gates[str(step)] = {'passed': all(conditions.values()), 'conditions': conditions,
        'raw_plan_improves': plan_improves,
        'brush_and_axis_family_nonregression': nonregression,
        'weak_real_donor_nonregression': real_nonregression,
        'raw_best14_gain_um': (raw0['plan_equal_mean']['mapped_best14_um']
                               - raw['plan_equal_mean']['mapped_best14_um'])}
passing = [step for step in steps[1:] if step_gates[str(step)]['passed']]
best_step = min(passing if passing else steps[1:], key=lambda step: (
    summary['steps'][str(step)]['synthetic']['appearance/raw']['plan_equal_mean']['mapped_best14_um'],
    step))
summary['pilot_gate'] = {'passed': bool(passing),
    'status': 'passed_development_only' if passing else 'failed',
    'passing_trained_steps': passing, 'best_development_step': best_step,
    'step_conditions': step_gates,
    'checkpoint_selection_note': 'chosen on the same 103 DEV panel; not an unbiased confirmation'}

config = {'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in (
        'evaluate_v3_one_pass_pose_capture_pilot_001.py',
        'train_v3_one_pass_pose_capture_pilot_001.py',
        'arbitrary_plane_one_shot_model.py', 'spatial_joint_fit_094.py',
        'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
        'global_atlas_contrast_090.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_allen_atlas_binding_v6.py')},
    'parent_094_completed_sha256': sha(parent_run / 'completed.json'),
    'parent_094_checkpoint_sha256': sha(parent_file),
    'train_completed_sha256': sha(train_run / 'completed.json'),
    'train_config_sha256': sha(train_run / 'config.json'),
    'train_draws_sha256': sha(train_run / 'draws.jsonl'),
    'train_schedule_sha256': sha(train_run / 'real_schedule.npy'),
    'checkpoint_sha256': {str(step): sha(checkpoint_files[step]) for step in steps},
    'plans_completed_sha256': sha(plans / 'completed.json'),
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_protocol_sha256': sha(panel / 'protocol.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'real_development_completed_sha256': sha(real / 'completed.json'),
    'real_development_outputs_sha256': real_receipt['output_sha256'],
    'reserved_real_train_completed_sha256': sha(reserved_train),
    'synthetic_candidate_rule': 'each checkpoint natural prior top8 old + top6 anchor',
    'fitter_mapper_score': 'frozen 094 fitter; checkpoint model frozen warp/map; 094 prior - energy/temperature - 0.05 warp_cost',
    'physical_metric': 'mean CCF distance at same 1024 deterministic surviving observed pixels; mapped96 bilinear readout',
    'real_metric': 'prior-selected five-point CCF discrepancy from inherited weak Allen affine on six development donors',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
for name, rows in (('candidates.jsonl', candidate_rows),
                   ('sections.jsonl', section_rows), ('real_rows.jsonl', real_rows)):
    with (out / name).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'eligible_sections_per_step': len(records),
    'candidates_per_step': len(records) * 14,
    'real_weak_development_sections_per_step': len(real_records),
    'step0_parent_weight_equal': step0_parent_weight_equal,
    'protocol_sha256': config['protocol_sha256'],
    'source_sha256': config['source_sha256'],
    'parent_094_checkpoint_sha256': config['parent_094_checkpoint_sha256'],
    'train_completed_sha256': config['train_completed_sha256'],
    'panel_completed_sha256': config['panel_completed_sha256'],
    'real_development_completed_sha256': config['real_development_completed_sha256'],
    'checkpoint_sha256': config['checkpoint_sha256'],
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'real_rows_sha256': sha(out / 'real_rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'final_animals_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'complete', 'eligible_sections': len(records),
    'pilot_gate': summary['pilot_gate']}), flush=True)
