"""Preregistered TRAIN-inner blind-fit mechanism pilot; no checkpoint promotion."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, UPDATES = 20261012280, 192, 1500
STEPS = (0, 500, 1500)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/BLIND_FIT_SCORE_280_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
RUN = ROOT / 'runs/blind_fit_score_280_pilot_001'
ARMS = ('full_atlas', 'support_only')
MAP_DERIVATIVE_LIMIT = .35
FIT_SIDE, TEACHER_SIDE, SUPPORT_SIDE = 24, 64, 24
TRAIN_SITES, READOUT_SITES = 128, 512
TRAIN_CANDIDATES, FIT_CHUNK = 4, 2
POSE_MODULES = ('encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global')
POSE_BUFFERS = ('normal_anchor_frames', 'center_origin', 'center_scale',
                'euclidean_scale', 'rotation_grid')
TRAIN_MODULES = ('warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair', 'fitted_matcher')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def blind_prior(prediction):
    prior = (prediction['log_mass'][..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit']),
        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
    return prior, prior.topk(16, -1).indices


def sites_from_sample(sample, count):
    valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
    indices = valid[torch.linspace(0, len(valid) - 1, min(count, len(valid)),
                                   device=valid.device).long()]
    chart = torch.stack((indices.remainder(SIDE),
        indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
    target = sample['centre'][0].reshape(-1, 3)[indices]
    return indices, chart, target


def mapped_sites(mapped, indices):
    batch, count, height, width = mapped['centre_surface_ccf_ap_dv_ml_um'].shape[:4]
    xy = torch.stack((indices.remainder(SIDE),
        indices.div(SIDE, rounding_mode='floor')), -1).float()
    grid = ((xy + .5) * (2 / SIDE) - 1)[None, None].expand(batch, count, -1, -1)
    grid = grid.reshape(batch * count, 1, -1, 2)
    surface = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
    points = F.grid_sample(surface, grid, mode='bilinear', padding_mode='border',
                           align_corners=False)[:, :, 0].transpose(1, 2)
    return points.reshape(batch, count, -1, 3)


def candidate_errors(prediction, order, chart, target, true_state, true_reflection):
    states = prediction['state'][0, order[0] // 2]
    points = rigid_points_090(states, order[0] % 2, chart, (SIDE, SIDE))
    true_plane = rigid_points_090(true_state[0], true_reflection[0], chart, (SIDE, SIDE))
    rigid = (points - true_plane[None]).norm(dim=-1).mean(-1) / 1000
    physical = (points - target[None]).norm(dim=-1).mean(-1) / 1000
    return rigid, physical


def supports_24(atlas, prediction, order, offsets, weights):
    axis = (torch.arange(SUPPORT_SIDE, device=atlas.device) + .5) / SUPPORT_SIDE - .5 / SIDE
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    result = []
    for first in range(0, 16, FIT_CHUNK):
        branch = order[0, first:first + FIT_CHUNK]
        states = prediction['state'][0, branch // 2]
        normal = full_frame_state_to_components(states)[1][..., :, 2]
        plane = rigid_points_090(states, branch % 2, chart, (SIDE, SIDE))
        plane = plane.reshape(len(branch), SUPPORT_SIDE, SUPPORT_SIDE, 3)
        grid = plane[:, None] + offsets[0, None, :, None, None, None] * normal[:, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(atlas[1:2], grid, (0., 0., 0.),
            (25., 25., 25.), weights.expand(len(branch), -1))
        result.append(rendered[:, 0] > .5)
    return torch.cat(result)


def nearest_support_pair(errors, masks):
    near = int(errors.argmin())
    common = (masks & masks[near]).sum((1, 2))
    area = masks.sum((1, 2))
    dice = 2 * common.float() / (area + area[near]).clamp_min(1)
    gap = (area - area[near]).abs().float() / (SUPPORT_SIDE * SUPPORT_SIDE)
    match = (errors >= errors[near] + 1) & (dice >= .85) & (gap <= .08) & (common >= 32)
    wrong = int(torch.where(match, errors, torch.inf).argmin()) if bool(match.any()) else None
    return near, wrong, dice, gap, common


def train_subset(errors, near, wrong):
    chosen = [0]
    if near not in chosen:
        chosen.append(near)
    if wrong is not None and wrong not in chosen:
        chosen.append(wrong)
    for threshold in (1., .5):
        extra = next((index for index in errors.argsort().tolist()
            if index not in chosen and errors[index] >= errors[near] + threshold), None)
        if extra is not None:
            chosen.append(extra)
            break
    for index in errors.argsort().tolist():
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == TRAIN_CANDIDATES:
            break
    return chosen[:TRAIN_CANDIDATES]


def fit_candidates(model, prediction, inputs, order, offsets, weights, atlas, indices=None,
                   target=None, chosen=None):
    chosen = list(range(16)) if chosen is None else chosen
    scores, details = [], []
    for first in range(0, len(chosen), FIT_CHUNK):
        selected = order[:, chosen[first:first + FIT_CHUNK]]
        modes, reflections = selected // 2, selected % 2
        mapped = model.map(prediction, offsets, modes, reflections,
            (FIT_SIDE, FIT_SIDE), atlas, weights, feature_side=FIT_SIDE,
            source_shape=(SIDE, SIDE), max_derivative=MAP_DERIVATIVE_LIMIT)
        scores.append(model.score_fitted_candidates(inputs, prediction, mapped, atlas, weights))
        if indices is not None:
            error = (mapped_sites(mapped, indices) - target[None, None]).norm(dim=-1).mean(-1) / 1000
            frame = full_frame_state_to_components(mapped['state'].flatten(0, 1))[1]
            displacement_ccf = torch.einsum('bij,bjhw->bhwi',
                frame, mapped['local_displacement_um'].flatten(0, 1))
            displacement_ccf = displacement_ccf.reshape(1, selected.shape[1], FIT_SIDE, FIT_SIDE, 3)
            no_warp_surface = mapped['centre_surface_ccf_ap_dv_ml_um'] - displacement_ccf
            no_warp_error = (mapped_sites({'centre_surface_ccf_ap_dv_ml_um': no_warp_surface},
                indices) - target[None, None]).norm(dim=-1).mean(-1) / 1000
            local = mapped['local_displacement_um'] / 1000
            norm = local.square().sum(2).mean((2, 3)).sqrt()
            rough = ((local[..., 1:] - local[..., :-1]).square().sum(2).mean((2, 3))
                + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).mean((2, 3))).sqrt()
            actual_derivative = mapped['prelimit_derivative_bound'] * mapped['deformation_rescale']
            for j in range(len(chosen[first:first + FIT_CHUNK])):
                details.append({'mapped_mm': float(error[0, j]),
                    'same_grid_no_warp_mm': float(no_warp_error[0, j]),
                    'displacement_rms_mm': float(norm[0, j]),
                    'roughness_mm_per_cell': float(rough[0, j]),
                    'prelimit_derivative': float(mapped['prelimit_derivative_bound'][0, j]),
                    'actual_derivative': float(actual_derivative[0, j])})
    return torch.cat(scores, 1), details


def exact_map(model, prediction, inputs, state, reflection, offsets, weights, atlas,
              indices, target):
    teacher = model.map({**prediction, 'state': state[:, None]}, offsets,
        torch.zeros((1, 1), device=inputs.device, dtype=torch.long), reflection[:, None],
        (TEACHER_SIDE, TEACHER_SIDE), atlas, weights, feature_side=TEACHER_SIDE,
        source_shape=(SIDE, SIDE), max_derivative=MAP_DERIVATIVE_LIMIT)
    frame = full_frame_state_to_components(state)[1]
    displacement_ccf = torch.einsum('bij,bjhw->bhwi',
        frame, teacher['local_displacement_um'][:, 0])[:, None]
    no_warp_surface = teacher['centre_surface_ccf_ap_dv_ml_um'] - displacement_ccf
    rigid = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': no_warp_surface}, indices)[0, 0]
    no_warp = (rigid - target).norm(dim=-1).mean() / 1000
    mapped = mapped_sites(teacher, indices)[0, 0]
    error = (mapped - target).norm(dim=-1).mean() / 1000
    local = teacher['local_displacement_um'][0, 0]
    rms = local.square().sum(0).mean().sqrt() / 1000
    shift = (mapped - rigid).norm(dim=-1) / 1000
    oracle = (target - rigid).norm(dim=-1) / 1000
    actual_derivative = (teacher['prelimit_derivative_bound'] * teacher['deformation_rescale'])[0, 0]
    return (no_warp, error, rms, actual_derivative, teacher,
            shift.mean(), shift.square().mean().sqrt(),
            oracle.mean(), oracle.square().mean().sqrt())


def load_acquired():
    receipt = json.loads((TRANSFER / 'completed.json').read_text())
    assert receipt['candidate'] == 272
    assert digest(TRANSFER / 'cases.jsonl') == receipt['cases_sha256']
    assert digest(TRANSFER / 'config.json') == receipt['config_sha256']
    transfer_config = json.loads((TRANSFER / 'config.json').read_text())
    cases = [json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()]
    sources = {
        'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                    ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                    'model_pixel_to_ap_dv_ml_um'),
        'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                     ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                     'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
    }
    acquired, source_data = [], {}
    for domain, (records_path, images_path, affine_key) in sources.items():
        assert digest(records_path) == transfer_config['datasets'][domain]['records_sha256']
        assert digest(images_path) == transfer_config['datasets'][domain]['images_sha256']
        source_data[domain] = {'records_sha256': digest(records_path),
                               'images_sha256': digest(images_path)}
        records = {row['array_row_index']: row for row in
                   (json.loads(line) for line in records_path.open())}
        images = np.load(images_path, mmap_mode='r')
        for case in cases:
            if case['domain'] != domain:
                continue
            record = records[case['array_row_index']]
            assert record['section_id'] == case['section_id']
            assert record.get('source_image_sha256', record.get('image_sha256')) == case['source_image_sha256']
            assert record['training_split' if domain == 'coronal' else 'split'] == 'train'
            image = torch.from_numpy(np.asarray(images[case['array_row_index'], 0],
                                                dtype=np.float32).copy())[None, None]
            if domain == 'sagittal':
                image = F.interpolate(image, (SIDE, SIDE), mode='area')
            matrix = np.asarray(record[affine_key], dtype=np.float64)
            weak_normal = np.cross(matrix[:, 0], matrix[:, 1])
            weak_normal /= np.linalg.norm(weak_normal)
            axis = int(np.abs(weak_normal).argmax())
            assert case['weak_axis'] == ('AP', 'DV', 'ML')[axis]
            acquired.append({'case': case, 'image': image, 'normal': weak_normal, 'axis': axis})
    assert len(acquired) == 98
    assert sum(row['case']['domain'] == 'coronal' for row in acquired) == 58
    assert sum(row['case']['domain'] == 'sagittal' for row in acquired) == 40
    assert len({(row['case']['domain'], row['case']['donor_id']) for row in acquired}) == 98
    return acquired, source_data, receipt


def readout(model, atlas, panel, acquired, arm, step, output):
    rows, real_rows = [], []
    model.eval()
    with torch.inference_mode():
        for item in panel:
            inputs = item['inputs'][None].cuda()
            state = item['state'][None].cuda()
            reflection = item['reflection'][None].cuda()
            offsets = item['offsets'][None].cuda()
            weights = item['weights'][None].cuda()
            indices = item['site_indices'].cuda()
            target = item['site_target'].cuda()
            chart = torch.stack((indices.remainder(SIDE),
                indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            prediction = model.predict(inputs)
            prior, order = blind_prior(prediction)
            rigid_errors, physical_errors = candidate_errors(
                prediction, order, chart, target, state, reflection)
            masks = supports_24(atlas, prediction, order, offsets, weights)
            near, wrong, dice, area_gap, common = nearest_support_pair(physical_errors, masks)
            scores, details = fit_candidates(model, prediction, inputs, order,
                offsets, weights, atlas, indices, target)
            scores = scores[0]
            (exact_no_warp, exact_mapped, exact_rms, exact_derivative, _,
             exact_shift_mean, exact_shift_rms, oracle_shift_mean, oracle_shift_rms) = exact_map(
                model, prediction, inputs, state, reflection, offsets, weights,
                atlas, indices, target)
            chosen = int(scores.argmax())
            near_half = [j for j in range(16) if physical_errors[j] <= .5]
            branch_rows = []
            for j in range(16):
                branch = int(order[0, j])
                normal = full_frame_state_to_components(
                    prediction['state'][0, branch // 2])[1][:, 2]
                obliquity = float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1))))
                branch_rows.append({'rank': j, 'branch': branch, 'mode': branch // 2,
                    'reflection': branch % 2, 'prior': float(prior[0, branch]),
                    'fit_score': float(scores[j]),
                    'original_rigid_plane_mm': float(rigid_errors[j]),
                    'original_surviving_tissue_mm': float(physical_errors[j]),
                    'support_fraction': float(masks[j].float().mean()),
                    'obliquity_deg': obliquity, **details[j]})
            row = {'arm': arm, 'step': step, 'panel': 'new_v9',
                'physical_section_id': item['physical_section_id'],
                'base_index': item['base_index'], 'slot': item['slot'],
                'provenance': item['provenance'],
                'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'true_obliquity_deg': item['obliquity_deg'],
                'direct_top1_original_rigid_plane_mm': float(rigid_errors[0]),
                'direct_top1_surviving_tissue_mm': float(physical_errors[0]),
                'rank_selected_index': chosen,
                'rank_selected_original_rigid_plane_mm': float(rigid_errors[chosen]),
                'rank_selected_original_surviving_tissue_mm': float(physical_errors[chosen]),
                'rank_selected_mapped_mm': details[chosen]['mapped_mm'],
                'best_rigid_plane_index_diagnostic_only': int(rigid_errors.argmin()),
                'best_rigid_plane_mm_diagnostic_only': float(rigid_errors.min()),
                'physical_best_index_diagnostic_only': int(physical_errors.argmin()),
                'physical_best_surviving_tissue_mm_diagnostic_only': float(physical_errors.min()),
                'near_index': near, 'support_matched_wrong_index': wrong,
                'near_eligible': bool(physical_errors[near] <= 1.5),
                'support_matched_pair_eligible': bool(physical_errors[near] <= 1.5 and wrong is not None),
                'support_matched_near_win': bool(scores[near] > scores[wrong])
                    if physical_errors[near] <= 1.5 and wrong is not None else None,
                'support_matched_dice': float(dice[wrong]) if wrong is not None else None,
                'support_matched_area_gap': float(area_gap[wrong]) if wrong is not None else None,
                'support_matched_common_cells': int(common[wrong]) if wrong is not None else None,
                'near_half_mm_branch_count': len(near_half),
                'near_half_mm_section_mean_map_gain': float(np.mean([
                    details[j]['same_grid_no_warp_mm'] - details[j]['mapped_mm']
                    for j in near_half])) if near_half else None,
                'exact_no_warp_mm': float(exact_no_warp),
                'exact_mapped_mm': float(exact_mapped),
                'exact_displacement_rms_mm': float(exact_rms),
                'exact_surviving_displacement_mean_mm': float(exact_shift_mean),
                'exact_surviving_displacement_rms_mm': float(exact_shift_rms),
                'oracle_surviving_displacement_mean_mm': float(oracle_shift_mean),
                'oracle_surviving_displacement_rms_mm': float(oracle_shift_rms),
                'exact_actual_derivative': float(exact_derivative),
                'branches': branch_rows}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            del inputs, prediction
        # Acquired images have no measured thickness. The same nominal 62.5-um
        # trapezoid PSF is used in both arms; only weak family/normal labels are read.
        offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')[None]
        weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda')[None] / 16
        for item in acquired:
            inputs = torch.cat((item['image'].cuda(),
                torch.zeros((1, 4, SIDE, SIDE), device='cuda')), 1)
            prediction = model.predict(inputs)
            prior, order = blind_prior(prediction)
            scores, _ = fit_candidates(model, prediction, inputs, order,
                                       offsets, weights, atlas)
            selected = int(order[0, int(scores[0].argmax())])
            direct = int(order[0, 0])
            weak = torch.as_tensor(item['normal'], device='cuda')
            def orientation(branch):
                normal = full_frame_state_to_components(
                    prediction['state'][0, branch // 2])[1][:, 2]
                return (int(normal.abs().argmax()),
                    float(torch.rad2deg(torch.acos((normal * weak).sum().abs().clamp(0, 1)))))
            axis, angle = orientation(selected)
            direct_axis, direct_angle = orientation(direct)
            case = item['case']
            row = {'arm': arm, 'step': step, 'panel': 'acquired_train_orientation',
                'domain': case['domain'], 'donor_id': case['donor_id'],
                'specimen_id': case['specimen_id'], 'experiment_id': case['experiment_id'],
                'section_id': case['section_id'], 'array_row_index': case['array_row_index'],
                'source_image_sha256': case['source_image_sha256'],
                'weak_axis': ('AP', 'DV', 'ML')[item['axis']],
                'selected_branch': selected, 'selected_axis': ('AP', 'DV', 'ML')[axis],
                'first_axis_correct': axis == item['axis'],
                'normal_gap_to_weak_affine_deg': angle,
                'frozen_direct_branch': direct,
                'frozen_direct_axis': ('AP', 'DV', 'ML')[direct_axis],
                'frozen_direct_normal_gap_deg': direct_angle}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            real_rows.append(row)
    model.train()
    by_base = {base: [row for row in rows if row['base_index'] == base] for base in range(56, 64)}
    summary = {'n': len(rows), 'top16_hits_rigid_plane': sum(row['best_rigid_plane_mm_diagnostic_only'] <= 1.5
                   for row in rows),
        'direct_top1_mean_rigid_plane_mm': float(np.mean([
            row['direct_top1_original_rigid_plane_mm'] for row in rows])),
        'selected_original_mean_rigid_plane_mm': float(np.mean([
            row['rank_selected_original_rigid_plane_mm'] for row in rows])),
        'direct_top1_mean_surviving_tissue_mm': float(np.mean([
            row['direct_top1_surviving_tissue_mm'] for row in rows])),
        'selected_original_mean_surviving_tissue_mm': float(np.mean([
            row['rank_selected_original_surviving_tissue_mm'] for row in rows])),
        'selected_mapped_mean_mm': float(np.mean([row['rank_selected_mapped_mm'] for row in rows])),
        'best_rigid_plane_mean_mm_diagnostic_only': float(np.mean([
            row['best_rigid_plane_mm_diagnostic_only'] for row in rows])),
        'physical_best_mean_surviving_tissue_mm_diagnostic_only': float(np.mean([
            row['physical_best_surviving_tissue_mm_diagnostic_only'] for row in rows])),
        'exact_no_warp_mean_mm': float(np.mean([row['exact_no_warp_mm'] for row in rows])),
        'exact_mapped_mean_mm': float(np.mean([row['exact_mapped_mm'] for row in rows])),
        'exact_displacement_rms_mean_mm': float(np.mean([row['exact_displacement_rms_mm'] for row in rows])),
        'exact_displacement_rms_median_mm': float(np.median([row['exact_displacement_rms_mm'] for row in rows])),
        'exact_surviving_displacement_mean_mm': float(np.mean([
            row['exact_surviving_displacement_mean_mm'] for row in rows])),
        'exact_surviving_displacement_rms_mm': float(np.mean([
            row['exact_surviving_displacement_rms_mm'] for row in rows])),
        'oracle_surviving_displacement_mean_mm': float(np.mean([
            row['oracle_surviving_displacement_mean_mm'] for row in rows])),
        'oracle_surviving_displacement_rms_mm': float(np.mean([
            row['oracle_surviving_displacement_rms_mm'] for row in rows])),
        'max_actual_derivative': max(row['exact_actual_derivative'] for row in rows),
        'max_blind_actual_derivative': max(branch['actual_derivative'] for row in rows
            for branch in row['branches']),
        'near_half_mm_n_sections': sum(row['near_half_mm_branch_count'] > 0 for row in rows),
        'near_half_mm_n_branches': sum(row['near_half_mm_branch_count'] for row in rows),
        'near_half_mm_mean_map_gain': float(np.mean([
            row['near_half_mm_section_mean_map_gain']
            for row in rows if row['near_half_mm_branch_count'] > 0]))
            if any(row['near_half_mm_branch_count'] > 0 for row in rows) else None,
        'support_matched_sections': sum(row['support_matched_pair_eligible'] for row in rows),
        'support_matched_near_wins': sum(row['support_matched_near_win'] is True for row in rows),
        'base_mean_gain_mm': {str(base): float(np.mean([
            row['direct_top1_original_rigid_plane_mm'] - row['rank_selected_original_rigid_plane_mm']
            for row in by_base[base]])) for base in by_base},
        'coronal': {}, 'sagittal': {}}
    for domain in ('coronal', 'sagittal'):
        subset = [row for row in real_rows if row['domain'] == domain]
        summary[domain] = {'n': len(subset),
            'first_axis_correct': sum(row['first_axis_correct'] for row in subset),
            'median_weak_normal_angle_deg': float(np.median([
                row['normal_gap_to_weak_affine_deg'] for row in subset])),
            'frozen_direct_first_axis_correct': sum(row['frozen_direct_axis'] == row['weak_axis']
                for row in subset),
            'frozen_direct_median_weak_normal_angle_deg': float(np.median([
                row['frozen_direct_normal_gap_deg'] for row in subset]))}
    return summary


def main():
    assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receipt = json.loads((PARENT / 'completed.json').read_text())
    assert receipt['candidate'] == 275 and receipt['selected_step'] == 8000
    checkpoint_path = PARENT / 'pose_step_08000.pt'
    assert digest(checkpoint_path) == receipt['checkpoint_sha256']['8000']
    assert digest(PARENT / 'config.json') == receipt['config_sha256']
    parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert parent['step'] == 8000 and parent['config']['candidate'] == 275
    pose_weights = {name: value for name, value in parent['model'].items()
                    if name.split('.')[0] in POSE_MODULES}
    pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
    del parent
    context = load_streaming_synthetic_v7_64(device='cuda')
    support_only_atlas = context['atlas'].clone()
    support_only_atlas[0].zero_()
    acquired, acquired_sources, transfer_receipt = load_acquired()
    source_names = ('train_blind_fit_score_280.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_joint_model_v7.py',
        'arbitrary_plane_recurrent_model.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_one_shot_slide_artifacts_v7.py',
        'arbitrary_plane_one_shot_slide_artifacts_v6.py',
        'arbitrary_plane_one_shot_slide_artifacts_v4.py',
        'arbitrary_plane_one_shot_slide_artifacts_v3.py',
        'arbitrary_plane_one_shot_stream.py',
        'arbitrary_plane_streaming_synthetic_v7_64.py',
        'arbitrary_plane_streaming_synthetic_v7.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
        'arbitrary_plane_allen_atlas_binding_v6.py',
        'arbitrary_plane_geometry.py',
        'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
    source_hashes = {name: digest(SOURCE / name) for name in source_names}
    RUN.mkdir(parents=True, exist_ok=False)
    panel = []
    panel_rng = np.random.default_rng(SEED + 1)
    panel_attempts = 0
    with (RUN / 'inner_draws.jsonl').open('w') as log:
        for base in range(56, 64):
            for slot in range(8):
                while True:
                    panel_attempts += 1
                    subject_index = base * VARIANTS + int(panel_rng.integers(VARIANTS))
                    draw_seed = SEED + 1_000_000_000 + panel_attempts
                    with torch.no_grad():
                        sample = sample_one_shot_slide_artifacts_v9(context, [subject_index],
                                                                      draw_seed, SIDE)
                    used = bool(sample['eligible'][0])
                    identity = sample['provenance'][0]
                    log.write(json.dumps({'base_index': base, 'slot': slot,
                        'subject_index': subject_index, 'draw_seed': draw_seed,
                        'used': used, 'provenance': identity}, allow_nan=False) + '\n')
                    if used:
                        break
                    del sample
                section_id = identity['physical_section_id']
                assert identity['split'] == 'train'
                indices, _, target = sites_from_sample(sample, READOUT_SITES)
                normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
                panel.append({'inputs': sample['inputs'][0].cpu(),
                    'state': sample['state'][0].cpu(),
                    'reflection': sample['reflection'][0].cpu(),
                    'offsets': sample['offsets'][0].cpu(),
                    'weights': sample['weights'][0].cpu(),
                    'site_indices': indices.cpu(), 'site_target': target.cpu(),
                    'base_index': base, 'slot': slot, 'physical_section_id': section_id,
                    'valid_fraction': float(sample['valid_mask'][0].float().mean()),
                    'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
                    'provenance': identity})
                del sample
    assert len(panel) == 64
    torch.save(panel, RUN / 'inner_panel.pt')
    config = {'candidate': 280, 'protocol_sha256': digest(PROTOCOL),
        'source_sha256': source_hashes, 'seed': SEED, 'parent_candidate': 275,
        'parent_checkpoint_sha256': digest(checkpoint_path),
        'parent_pose_buffers_loaded': POSE_BUFFERS,
        'parent_completed_sha256': digest(PARENT / 'completed.json'),
        'parent_config_sha256': receipt['config_sha256'],
        'parent_draws_sha256': digest(PARENT / 'draws.jsonl'),
        'parent_inner_draws_sha256': digest(PARENT / 'inner_draws.jsonl'),
        'transfer_cases_sha256': transfer_receipt['cases_sha256'],
        'transfer_config_sha256': transfer_receipt['config_sha256'],
        'acquired_source_data': acquired_sources,
        'synthetic_provenance': context['provenance'],
        'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
        'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
        'panel_attempts': panel_attempts, 'arms': ARMS, 'updates_per_arm': UPDATES,
        'readout_steps': STEPS, 'train_base_indices': list(range(56)),
        'inner_base_indices': list(range(56, 64)),
        'blind_candidates': 16, 'train_subset_size': TRAIN_CANDIDATES,
        'fit_side': FIT_SIDE, 'teacher_side': TEACHER_SIDE,
        'support_side': SUPPORT_SIDE, 'support_threshold': .5,
        'support_pair': 'nearest physical candidate <=1.5 mm versus minimum-error alternative >=1 mm worse, Dice >=.85, area gap <=.08, common cells >=32',
        'ranking_pairs': 'all selected blind pairs >=.5 mm apart in surviving-tissue CCF error; no injected candidates',
        'first_choice_gate_metric': 'mean geometric rigid-plane distance to true state at identical surviving chart sites; 275-comparable, mm',
        'local_map_metric': 'mean CCF error against surviving-tissue correspondence at identical chart sites; mm',
        'map_derivative_limit': MAP_DERIVATIVE_LIMIT,
        'optimizer': 'fresh matched AdamW, lr=1e-4, weight_decay=1e-4, clip=2',
        'loss': 'blind pairwise softplus + exact-plane map mm + .05 reliability BCE + .005 map magnitude/roughness',
        'acquired_psf': 'fixed nominal 62.5-um 9-offset trapezoid for both arms; thickness unavailable; weak orientation only',
        'map_noncollapse_gate': 'same-cell >=0.05-mm error gain implies >=0.05-mm mean displacement by reverse triangle inequality',
        'checkpoint_selection': 'none; only 1500 eligible for gate',
        'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False,
        'torch': torch.__version__, 'numpy': np.__version__}
    (RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    schedule, readouts, counters = [], {}, {}
    started = time.perf_counter()
    with (RUN / 'training_draws.jsonl').open('w') as draws, \
         (RUN / 'training.jsonl').open('w') as training, \
         (RUN / 'readouts.jsonl').open('w') as output:
        for arm in ARMS:
            torch.manual_seed(SEED + 2)
            torch.cuda.manual_seed_all(SEED + 2)
            model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
                atlas_conditioning=True, fitted_ranking=True).cuda()
            for name in POSE_MODULES:
                module = getattr(model, name)
                prefix = name + '.'
                module.load_state_dict({key[len(prefix):]: value for key, value in pose_weights.items()
                                        if key.startswith(prefix)}, strict=True)
            with torch.no_grad():
                for name in POSE_BUFFERS:
                    getattr(model, name).copy_(pose_buffers[name])
                    assert torch.equal(getattr(model, name).cpu(), pose_buffers[name])
            model.requires_grad_(False)
            for name in TRAIN_MODULES:
                getattr(model, name).requires_grad_(True)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
            atlas = context['atlas'] if arm == 'full_atlas' else support_only_atlas
            train_rng = np.random.default_rng(SEED + 3)
            attempted = accepted = 0
            readouts[arm] = {}
            for step in range(UPDATES + 1):
                if step:
                    if arm == 'full_atlas':
                        while True:
                            attempted += 1
                            base = int(train_rng.integers(56))
                            subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
                            draw_seed = SEED + 2_000_000_000 + attempted
                            with torch.no_grad():
                                sample = sample_one_shot_slide_artifacts_v9(context,
                                    [subject_index], draw_seed, SIDE)
                            used = bool(sample['eligible'][0])
                            identity = sample['provenance'][0]
                            draws.write(json.dumps({'step': step, 'attempt': attempted,
                                'base_index': base, 'subject_index': subject_index,
                                'draw_seed': draw_seed, 'used': used,
                                'provenance': identity}, allow_nan=False) + '\n')
                            if used:
                                section_id = identity['physical_section_id']
                                assert identity['split'] == 'train'
                                input_hash = hashlib.sha256(sample['inputs'][0].cpu().numpy().tobytes()).hexdigest()
                                schedule.append((subject_index, draw_seed, section_id, input_hash))
                                break
                            del sample
                    else:
                        subject_index, draw_seed, section_id, input_hash = schedule[step - 1]
                        with torch.no_grad():
                            sample = sample_one_shot_slide_artifacts_v9(context,
                                [subject_index], draw_seed, SIDE)
                        assert bool(sample['eligible'][0])
                        assert sample['provenance'][0]['physical_section_id'] == section_id
                        assert hashlib.sha256(sample['inputs'][0].cpu().numpy().tobytes()).hexdigest() == input_hash
                        attempted, accepted = step, step
                    accepted += arm == 'full_atlas'
                    inputs = sample['inputs']
                    state = sample['state']
                    reflection = sample['reflection']
                    offsets = sample['offsets']
                    weights = sample['weights']
                    indices, chart, target = sites_from_sample(sample, TRAIN_SITES)
                    prediction = model.predict(inputs)
                    _, order = blind_prior(prediction)
                    with torch.no_grad():
                        rigid_errors, physical_errors = candidate_errors(
                            prediction, order, chart, target, state, reflection)
                        masks = supports_24(context['atlas'], prediction, order, offsets, weights)
                        near, wrong, dice, area_gap, common = nearest_support_pair(physical_errors, masks)
                        chosen = train_subset(physical_errors, near, wrong)
                    optimizer.zero_grad(set_to_none=True)
                    scores, _ = fit_candidates(model, prediction, inputs, order,
                        offsets, weights, atlas, chosen=chosen)
                    pair_losses = []
                    for i in range(len(chosen)):
                        for j in range(i + 1, len(chosen)):
                            delta = float(physical_errors[chosen[j]] - physical_errors[chosen[i]])
                            if abs(delta) >= .5:
                                sign = 1 if delta > 0 else -1
                                pair_losses.append(F.softplus(-sign * (scores[0, i] - scores[0, j])))
                    rank_loss = torch.stack(pair_losses).mean() if pair_losses else scores.sum() * 0
                    (no_warp, map_error, map_rms, derivative, teacher,
                     shift_mean, shift_rms, oracle_mean, oracle_rms) = exact_map(
                        model, prediction, inputs, state, reflection, offsets, weights,
                        atlas, indices, target)
                    valid_small = F.interpolate(sample['valid_mask'][:, None].float(),
                        (TEACHER_SIDE, TEACHER_SIDE), mode='area')
                    reliability = F.binary_cross_entropy_with_logits(
                        teacher['correspondence_logit'][:, 0], valid_small)
                    local = teacher['local_displacement_um'] / 1000
                    roughness = ((local[..., 1:] - local[..., :-1]).abs().mean()
                        + (local[..., 1:, :] - local[..., :-1, :]).abs().mean())
                    loss = rank_loss + map_error + .05 * reliability + .005 * (map_rms + roughness)
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
                    optimizer.step()
                    training.write(json.dumps({'arm': arm, 'step': step,
                        'physical_section_id': section_id,
                        'blind_top1_rigid_plane_mm': float(rigid_errors[0]),
                        'blind_top1_surviving_tissue_mm': float(physical_errors[0]),
                        'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                        'blind_best_surviving_tissue_mm': float(physical_errors[near]),
                        'near_within_1_5mm': bool(physical_errors[near] <= 1.5),
                        'near_wrong_available': bool((physical_errors >= physical_errors[near] + 1).any()),
                        'near_wrong_supplied': bool(physical_errors[near] <= 1.5 and any(
                            physical_errors[index] >= physical_errors[near] + 1 for index in chosen)),
                        'support_matched_wrong_available': wrong is not None,
                        'support_matched_near_wrong_update': bool(
                            physical_errors[near] <= 1.5 and wrong is not None and wrong in chosen),
                        'support_matched_wrong_in_subset': wrong in chosen if wrong is not None else False,
                        'support_matched_dice': float(dice[wrong]) if wrong is not None else None,
                        'support_matched_area_gap': float(area_gap[wrong]) if wrong is not None else None,
                        'support_matched_common_cells': int(common[wrong]) if wrong is not None else None,
                        'selected_blind_indices': chosen, 'rank_pairs': len(pair_losses),
                        'rank_loss': float(rank_loss.detach()),
                        'exact_no_warp_mm': float(no_warp.detach()),
                        'exact_mapped_mm': float(map_error.detach()),
                        'exact_displacement_rms_mm': float(map_rms.detach()),
                        'exact_surviving_displacement_mean_mm': float(shift_mean.detach()),
                        'exact_surviving_displacement_rms_mm': float(shift_rms.detach()),
                        'oracle_surviving_displacement_mean_mm': float(oracle_mean.detach()),
                        'oracle_surviving_displacement_rms_mm': float(oracle_rms.detach()),
                        'exact_actual_derivative': float(derivative.detach()),
                        'reliability': float(reliability.detach()),
                        'loss': float(loss.detach()), 'gradient_norm': float(gradient),
                        'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
                    if step == 1 or step % 100 == 0:
                        training.flush()
                        draws.flush()
                        print(json.dumps({'event': 'train_milestone', 'arm': arm,
                            'step': step, 'accepted': accepted,
                            'loss': float(loss.detach())}), flush=True)
                    del sample, inputs, prediction, teacher, loss

                if step in STEPS:
                    draws.flush()
                    training.flush()
                    readouts[arm][step] = readout(model, atlas, panel, acquired,
                                                   arm, step, output)
                    output.flush()
                    checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save({'candidate': 280, 'arm': arm, 'step': step,
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': config, 'attempted': attempted, 'accepted': accepted,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'calibrated': False}, temporary)
                    os.replace(temporary, checkpoint)
                    print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                        'summary': readouts[arm][step]}), flush=True)
            counters[arm] = {'attempted': attempted, 'accepted': accepted}
            assert accepted == UPDATES
            del model, optimizer
            torch.cuda.empty_cache()

    full, control = readouts['full_atlas'][UPDATES], readouts['support_only'][UPDATES]
    full_rate = full['support_matched_near_wins'] / max(1, full['support_matched_sections'])
    control_rate = control['support_matched_near_wins'] / max(1, control['support_matched_sections'])
    assert full['support_matched_sections'] == control['support_matched_sections']
    assert full['direct_top1_mean_rigid_plane_mm'] == control['direct_top1_mean_rigid_plane_mm']
    gate = {
        'full_original_gain_vs_frozen_direct_at_least_0_25mm':
            full['direct_top1_mean_rigid_plane_mm'] - full['selected_original_mean_rigid_plane_mm'] >= .25,
        'full_original_gain_vs_trained_support_at_least_0_15mm':
            control['selected_original_mean_rigid_plane_mm'] - full['selected_original_mean_rigid_plane_mm'] >= .15,
        'at_least_six_bases_nonnegative_direct_gain':
            sum(gain >= 0 for gain in full['base_mean_gain_mm'].values()) >= 6,
        'at_least_24_distinct_support_matched_sections': full['support_matched_sections'] >= 24,
        'full_support_matched_near_win_at_least_70pct': full_rate >= .70,
        'full_support_matched_near_win_gain_at_least_10pp': full_rate - control_rate >= .10,
        'exact_map_mean_gain_at_least_0_05mm':
            full['exact_no_warp_mean_mm'] - full['exact_mapped_mean_mm'] >= .05,
        'all_maps_derivative_at_most_0_35': max(full['max_actual_derivative'],
            full['max_blind_actual_derivative']) <= MAP_DERIVATIVE_LIMIT + 1e-5,
        'near_half_mm_map_degradation_at_most_0_05mm':
            full['near_half_mm_mean_map_gain'] is not None
            and full['near_half_mm_mean_map_gain'] >= -.05,
        'coronal_family_at_least_54_of_58': full['coronal']['first_axis_correct'] >= 54,
        'sagittal_family_at_least_18_of_40': full['sagittal']['first_axis_correct'] >= 18,
        'sagittal_median_weak_angle_rise_at_most_5deg':
            full['sagittal']['median_weak_normal_angle_deg']
            <= full['sagittal']['frozen_direct_median_weak_normal_angle_deg'] + 5,
    }
    training_rows = [json.loads(line) for line in (RUN / 'training.jsonl').open()]
    update_coverage = {arm: {
        'updates': UPDATES,
        'with_rank_pairs': sum(row['rank_pairs'] > 0 for row in training_rows if row['arm'] == arm),
        'with_near_versus_wrong': sum(row['near_wrong_supplied']
            for row in training_rows if row['arm'] == arm),
        'with_near_within_1_5mm': sum(row['near_within_1_5mm'] for row in training_rows if row['arm'] == arm),
        'with_support_matched_near_wrong': sum(row['support_matched_near_wrong_update']
            for row in training_rows if row['arm'] == arm),
        'sparse_under_15pct': sum(row['visible_fraction'] < .15 for row in training_rows
            if row['arm'] == arm),
        'sparse_with_rank_pairs': sum(row['visible_fraction'] < .15 and row['rank_pairs'] > 0
            for row in training_rows if row['arm'] == arm),
        'sparse_with_near_versus_wrong': sum(row['visible_fraction'] < .15
            and row['near_wrong_supplied']
            for row in training_rows if row['arm'] == arm),
        'sparse_with_support_matched_near_wrong': sum(row['visible_fraction'] < .15
            and row['support_matched_near_wrong_update']
            for row in training_rows if row['arm'] == arm),
    } for arm in ARMS}
    for coverage in update_coverage.values():
        for key in ('with_rank_pairs', 'with_near_versus_wrong',
                    'with_support_matched_near_wrong'):
            coverage[key + '_fraction'] = coverage[key] / UPDATES
        for key in ('sparse_with_rank_pairs', 'sparse_with_near_versus_wrong',
                    'sparse_with_support_matched_near_wrong'):
            coverage[key + '_fraction_of_sparse'] = coverage[key] / max(1, coverage['sparse_under_15pct'])
    output_hashes = {name: digest(RUN / name) for name in
        ('config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'training_draws.jsonl',
         'training.jsonl', 'readouts.jsonl')}
    checkpoint_hashes = {f'{arm}_{step}': digest(RUN / f'{arm}_step_{step:05d}.pt')
        for arm in ARMS for step in STEPS}
    completed = {'candidate': 280, 'steps': STEPS, 'counters': counters,
        'readouts': readouts, 'update_coverage': update_coverage,
        'support_matched_full_rate': full_rate, 'support_matched_control_rate': control_rate,
        'gate': gate, 'gate_numeric_pass': all(gate.values()),
        'gate_pass': all(gate.values()),
        'gate_status': 'positive TRAIN-inner mechanism signal only' if all(gate.values())
            else 'failed bounded TRAIN-inner mechanism gate; do not scale',
        'result_role': 'TRAIN-inner bounded mechanism pilot only; no checkpoint promotion',
        'protocol_sha256': config['protocol_sha256'], 'source_sha256': source_hashes,
        'output_sha256': output_hashes, 'checkpoint_sha256': checkpoint_hashes,
        'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
    (RUN / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf8')
    print(json.dumps({'event': 'complete', 'gate': gate,
        'gate_numeric_pass': completed['gate_numeric_pass'],
        'gate_status': completed['gate_status']}), flush=True)


if __name__ == '__main__':
    main()
