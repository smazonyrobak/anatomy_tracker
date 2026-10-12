"""Fresh TRAIN-inner balanced correspondence and gated-map mechanism test."""
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
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_multiscale_pair_map_v1 import MultiScalePairJointModel
from training.arbitrary_plane_multiscale_pair_map_v2 import GatedMultiScalePairJointModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_spatial_pair_map_v1 import SpatialPairJointModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.train_image_map_282 import (POSE_BUFFERS, POSE_MODULES, SIDE, digest,
                                          mapped_sites, sites_from_sample, tensor_digest)


SEED, UPDATES, SITES = 20261012294, 3000, 512
STEPS = (0, 600, 1800, 3000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/BALANCED_GATED_PAIR_MAP_294_PROTOCOL_20261012.md'
POSE_PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
CONTROL_289 = ROOT / 'runs/spatial_pair_map_289_002'
PARENT_292 = ROOT / 'runs/multiscale_pair_map_292_001'
PREVIOUS_287 = ROOT / 'runs/map_feature_287_001'
PREVIOUS_286 = ROOT / 'runs/pose_gauge_map_286_001'
RUN = ROOT / 'runs/balanced_gated_pair_map_294_001'
SOURCE_NAMES = ('train_balanced_gated_pair_map_294.py',
    'arbitrary_plane_multiscale_pair_map_v2.py', 'arbitrary_plane_multiscale_pair_map_v1.py',
    'arbitrary_plane_spatial_pair_map_v1.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py', 'arbitrary_plane_postartifact_gauge_v1.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'train_image_map_282.py',
    'train_blind_fit_score_280.py', 'global_atlas_contrast_090.py')


def pair(context, subject_index, draw_seed):
    old = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
    new = sample_postartifact_gauge_v1(context, [subject_index], draw_seed, SIDE,
                                       source_sample=old)
    return old, new


def offset_loss(mapped, target, valid, fragment, state, reflection):
    displacement = target - mapped['rigid_surface_ccf_ap_dv_ml_um']
    _, frame, basis = full_frame_state_to_components(state)
    local = torch.einsum('bij,bhwj->bhwi', frame.transpose(1, 2), displacement)
    canonical = torch.linalg.solve(basis, local[..., :2].reshape(len(state), -1, 2)
                                   .transpose(1, 2)).transpose(1, 2).reshape_as(local[..., :2])
    sign = torch.where(reflection.bool(), -1., 1.)[:, None, None]
    canonical = torch.stack((canonical[..., 0] * sign, canonical[..., 1]), -1)
    losses, coverage = [], {}
    for side, radius in ((48, 6), (96, 4), (192, 3)):
        level = mapped['levels'][side]
        if side == SIDE:
            desired = canonical.permute(0, 3, 1, 2) * side
            pure = valid[:, None]
            fragment_level = fragment
        else:
            # Nearest target is one physical tissue point, never a mean across a tear.
            desired = F.interpolate(canonical.permute(0, 3, 1, 2), (side, side),
                                    mode='nearest') * side
            fraction = F.avg_pool2d(valid[:, None].float(), SIDE // side)
            fragment_fraction = F.avg_pool2d(fragment[:, None].float(), SIDE // side)
            pure = (fraction > .999) & ((fragment_fraction < .001) |
                                       (fragment_fraction > .999))
            fragment_level = fragment_fraction[:, 0] > .999
        residual = (desired - level['base_flow'].detach()).round().long()
        inside = ((residual[:, 0].abs() <= radius) & (residual[:, 1].abs() <= radius)
                  & pure[:, 0])
        candidate = ((residual[:, 1] + radius) * (2 * radius + 1) +
                     residual[:, 0] + radius).clamp(0, (2 * radius + 1) ** 2 - 1)
        supported = level['support'].gather(1, candidate[:, None]).squeeze(1) > .9
        eligible = inside & supported
        strata = {}
        scale_losses = []
        logits = level['logits'].permute(0, 2, 3, 1)
        for name, mask in (('intact', ~fragment_level), ('fragment', fragment_level)):
            selected = eligible & mask
            count = int(selected.sum())
            if count:
                ce = F.cross_entropy(logits[selected], candidate[selected])
                scale_losses.append(ce)
                ce_sum = float(ce.detach()) * count
            else:
                ce_sum = 0.
            strata[name] = {'eligible_pixels': count, 'cross_entropy_sum': ce_sum}
        if scale_losses:
            losses.append(torch.stack(scale_losses).mean())
        coverage[str(side)] = {'pure_pixels': int(pure.sum()),
            'in_window_pixels': int(inside.sum()), 'supported_pixels': int(eligible.sum()),
            'strata': strata,
            'balanced_cross_entropy': float(torch.stack(scale_losses).mean().detach())
                if scale_losses else None}
    return (torch.stack(losses).mean() if losses else
            mapped['levels'][192]['logits'].sum() * 0), coverage


def smoothness_loss(local, valid, fragment, image):
    field = local / 1000
    same_x = valid[..., 1:] & valid[..., :-1] & (fragment[..., 1:] == fragment[..., :-1])
    same_y = valid[..., 1:, :] & valid[..., :-1, :] & (fragment[..., 1:, :] == fragment[..., :-1, :])
    edge_x = torch.exp(-6 * (image[..., 1:] - image[..., :-1]).abs()) * same_x[:, None]
    edge_y = torch.exp(-6 * (image[..., 1:, :] - image[..., :-1, :]).abs()) * same_y[:, None]
    return (((field[..., 1:] - field[..., :-1]).abs() * edge_x).sum() /
            (3 * edge_x.sum()).clamp_min(1) +
            ((field[..., 1:, :] - field[..., :-1, :]).abs() * edge_y).sum() /
            (3 * edge_y.sum()).clamp_min(1)) / 2


def gate_loss(mapped, target, valid, fragment):
    raw_move = (mapped['raw_centre_surface_ccf_ap_dv_ml_um'][:, 0] -
                mapped['rigid_surface_ccf_ap_dv_ml_um']).detach()
    truth_move = (target - mapped['rigid_surface_ccf_ap_dv_ml_um']).detach()
    optimal = ((truth_move * raw_move).sum(-1) /
               raw_move.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
    predicted = mapped['move_gate'][:, 0]
    losses = [F.mse_loss(predicted[mask], optimal[mask])
              for mask in (valid & ~fragment, valid & fragment) if bool(mask.any())]
    return torch.stack(losses).mean()


def group_error(mapped, item, key, atlas):
    indices_cpu = item[key]
    if not len(indices_cpu):
        return None
    indices = indices_cpu.cuda()
    target = item['centre'].reshape(-1, 3)[indices_cpu].cuda()
    points = mapped_sites(mapped, indices)[0, 0]
    rigid = mapped_sites({'centre_surface_ccf_ap_dv_ml_um':
        mapped['rigid_surface_ccf_ap_dv_ml_um'][:, None]}, indices)[0, 0]
    no_warp = float((rigid - target).norm(dim=-1).mean() / 1000)
    fitted = float((points - target).norm(dim=-1).mean() / 1000)
    support = render_finite_thickness_coordinate_grid(
        atlas[1:2], points[None, None, None], (0., 0., 0.),
        (25., 25., 25.), points.new_ones(1, 1))[0, 0, 0]
    result = {'sites': len(indices), 'no_warp_mm': no_warp,
              'mapped_mm': fitted, 'gain_mm': no_warp - fitted,
              'mapped_support_mean': float(support.mean()),
              'mapped_support_below_half': int((support < .5).sum())}
    if 'levels' in mapped:
        peak = mapped['levels'][192]['logits'].softmax(1).amax(1).flatten()[indices]
        result['fine_posterior_peak_mean'] = float(peak.mean())
    if 'move_gate' in mapped:
        result['gate_mean'] = float(mapped['move_gate'].flatten()[indices].mean())
    return result


def summary(rows):
    groups = {}
    for name in ('all', 'intact', 'fragment'):
        found = [row[name] for row in rows if row[name] is not None]
        count = sum(row['sites'] for row in found)
        groups[name] = {'sites': count,
            'mapped_support_below_half': sum(row['mapped_support_below_half'] for row in found)}
        for metric in ('no_warp_mm', 'mapped_mm', 'gain_mm', 'mapped_support_mean',
                       'fine_posterior_peak_mean', 'gate_mean'):
            if metric in rows[0]['all']:
                groups[name][metric] = float(sum(row[metric] * row['sites'] for row in found) /
                                              count) if count else None
    bases = {str(base): float(np.mean([row['all']['gain_mm'] for row in rows
        if row['base_index'] == base])) for base in range(56, 64)}
    report = {'sections': len(rows), 'groups': groups, 'gain_by_base_mm': bases,
        'positive_gain_bases': sum(gain > 0 for gain in bases.values()),
        'pose_first_choice_mean_mm': float(np.mean([row['pose_first_choice_mm'] for row in rows])),
        'pose_first_choice_sha256': tensor_digest(torch.tensor(
            [row['pose_first_choice_mm'] for row in rows], dtype=torch.float64)),
        'field_absmax_mm': max(row['field_absmax_mm'] for row in rows),
        'dense_endpoint_mean_mm': float(np.mean([row['dense_endpoint_mm'] for row in rows]))
            if rows[0]['dense_endpoint_mm'] is not None else None,
        'true_inplane_over_1_5mm_valid_fraction': float(np.mean([
            row['true_inplane_over_1_5mm_valid_fraction'] for row in rows]))
            if rows[0]['true_inplane_over_1_5mm_valid_fraction'] is not None else None,
        'true_normal_over_250um_valid_fraction': float(np.mean([
            row['true_normal_over_250um_valid_fraction'] for row in rows]))
            if rows[0]['true_normal_over_250um_valid_fraction'] is not None else None,
        'true_inplane_over_1_5mm_fragment_fraction': float(np.mean([
            row['true_inplane_over_1_5mm_fragment_fraction'] for row in rows
            if row['true_inplane_over_1_5mm_fragment_fraction'] is not None]))
            if any(row['true_inplane_over_1_5mm_fragment_fraction'] is not None for row in rows)
            else None,
        'true_normal_over_250um_fragment_fraction': float(np.mean([
            row['true_normal_over_250um_fragment_fraction'] for row in rows
            if row['true_normal_over_250um_fragment_fraction'] is not None]))
            if any(row['true_normal_over_250um_fragment_fraction'] is not None for row in rows)
            else None}
    if 'offset_coverage' in rows[0]:
        levels, balanced = {}, []
        for side in ('48', '96', '192'):
            level = {'pure_pixels': sum(row['offset_coverage'][side]['pure_pixels'] for row in rows),
                'in_window_pixels': sum(row['offset_coverage'][side]['in_window_pixels'] for row in rows),
                'supported_pixels': sum(row['offset_coverage'][side]['supported_pixels'] for row in rows)}
            means = []
            for name in ('intact', 'fragment'):
                count = sum(row['offset_coverage'][side]['strata'][name]['eligible_pixels']
                            for row in rows)
                ce_sum = sum(row['offset_coverage'][side]['strata'][name]['cross_entropy_sum']
                             for row in rows)
                level[name] = {'eligible_pixels': count,
                    'cross_entropy': ce_sum / count if count else None}
                if count:
                    means.append(ce_sum / count)
            level['balanced_cross_entropy'] = float(np.mean(means)) if means else None
            if means:
                balanced.append(level['balanced_cross_entropy'])
            levels[side] = level
        report['offset_classification'] = {
            'levels': levels, 'balanced_cross_entropy': float(np.mean(balanced)) if balanced else None}
    return report


def readout(model, atlas, panel, model_id, step, output):
    rows = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for item in panel:
            inputs = item['inputs'][None].cuda()
            state = item['state'][None].cuda()
            mapped = model.map_at_pose(inputs, state, item['reflection'][None].cuda(),
                item['offsets'][None].cuda(), item['weights'][None].cuda(), atlas)
            pose = model.pose_model.predict(inputs)
            log_reflection = torch.stack((F.logsigmoid(-pose['reflection_logit']),
                                         F.logsigmoid(pose['reflection_logit'])), -1)
            choice = (pose['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)
            chosen_state = pose['state'][0, choice[0] // 2]
            reflection = choice[0] % 2
            indices = item['site_indices'].cuda()
            chart = torch.stack((indices.remainder(SIDE),
                indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            pose_points = rigid_points_090(chosen_state, reflection, chart, (SIDE, SIDE))
            pose_target = item['centre'].reshape(-1, 3)[item['site_indices']].cuda()
            valid = item['valid_mask'][None].cuda()
            fragment = valid & item['fragment_mask'][None].cuda()
            target = item['centre'][None].cuda()
            dense_endpoint, inplane_valid, normal_valid = None, None, None
            inplane_fragment, normal_fragment = None, None
            if mapped['rigid_surface_ccf_ap_dv_ml_um'].shape[1] == SIDE:
                dense = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - target).norm(dim=-1) / 1000
                _, frame, _ = full_frame_state_to_components(state)
                true_local = torch.einsum('bij,bhwj->bhwi', frame.transpose(1, 2),
                    target - mapped['rigid_surface_ccf_ap_dv_ml_um'])
                inplane_over = true_local[..., :2].norm(dim=-1) > 1500
                normal_over = true_local[..., 2].abs() > 250
                dense_endpoint = float(dense[valid].mean())
                inplane_valid = float(inplane_over[valid].float().mean())
                normal_valid = float(normal_over[valid].float().mean())
                if bool(fragment.any()):
                    inplane_fragment = float(inplane_over[fragment].float().mean())
                    normal_fragment = float(normal_over[fragment].float().mean())
            row = {'model_id': model_id, 'step': step,
                'physical_section_id': item['physical_section_id'],
                'base_index': item['base_index'], 'event_flags': item['event_flags'],
                'all': group_error(mapped, item, 'site_indices', atlas),
                'intact': group_error(mapped, item, 'intact_indices', atlas),
                'fragment': group_error(mapped, item, 'fragment_indices', atlas),
                'pose_first_choice_mm': float((pose_points - pose_target).norm(dim=-1).mean() / 1000),
                'field_absmax_mm': float(mapped['local_displacement_um'].abs().amax() / 1000),
                'dense_endpoint_mm': dense_endpoint,
                'true_inplane_over_1_5mm_valid_fraction': inplane_valid,
                'true_normal_over_250um_valid_fraction': normal_valid,
                'true_inplane_over_1_5mm_fragment_fraction': inplane_fragment,
                'true_normal_over_250um_fragment_fraction': normal_fragment}
            if 'levels' in mapped:
                _, row['offset_coverage'] = offset_loss(mapped, target, valid, fragment,
                                                         state, item['reflection'][None].cuda())
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    output.flush()
    model.train(was_training)
    return summary(rows)


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

pose_receipt = json.loads((POSE_PARENT / 'completed.json').read_text(encoding='utf8'))
pose_path = POSE_PARENT / 'pose_step_08000.pt'
assert pose_receipt['selected_step'] == 8000
assert digest(pose_path) == pose_receipt['checkpoint_sha256']['8000']
parent_292_receipt = json.loads((PARENT_292 / 'completed.json').read_text(encoding='utf8'))
assert digest(PARENT_292 / 'completed.json') == \
    '5823e81dd08ff3d1eb316a30662c451057a365814a0f5b962070a49e4d008306'
assert parent_292_receipt['selected_step'] == 2400
parent_292_path = PARENT_292 / 'joint_multiscale_pair_step_02400.pt'
assert digest(parent_292_path) == parent_292_receipt['checkpoint_sha256']['2400']
control_289_receipt = json.loads((CONTROL_289 / 'completed.json').read_text(encoding='utf8'))
assert digest(CONTROL_289 / 'completed.json') == \
    '5387c3514eeb91140f246a40240ec383d9c35f2ca917878eb195c7a1a87dd798'
assert control_289_receipt['selected_step'] == 5000
control_289_path = CONTROL_289 / 'joint_spatial_pair_step_05000.pt'
assert digest(control_289_path) == control_289_receipt['checkpoint_sha256']['5000']
for directory, expected in ((PREVIOUS_287,
        '239d6d4f008029e331234bec0bcc088f3d5ab71c5fa5d856ddbd1e0120a6f63d'),
        (PREVIOUS_286, '08b88ad1906303b33947dd55ce7e6d730883266f28682a8904b56b01a27551eb')):
    assert digest(directory / 'completed.json') == expected
for directory, receipt, files in ((PARENT_292, parent_292_receipt,
        ('config.json', 'training.jsonl', 'inner_panel.pt')),
        (CONTROL_289, control_289_receipt, ('training.jsonl', 'inner_panel.pt'))):
    for name in files:
        assert digest(directory / name) == receipt['files_sha256'][name]
prior_ids = set()
for directory in (PARENT_292, CONTROL_289, PREVIOUS_287):
    prior_ids.update(json.loads(line)['physical_section_id'] for line in
                     (directory / 'training.jsonl').open(encoding='utf8'))
for directory in (PARENT_292, CONTROL_289, PREVIOUS_287, PREVIOUS_286):
    prior_ids.update(item['physical_section_id'] for item in
        torch.load(directory / 'inner_panel.pt', map_location='cpu', weights_only=False))
prior_ids.update(json.loads(line)['physical_section_id'] for line in
                 (PREVIOUS_286 / 'training_schedule.jsonl').open(encoding='utf8'))
parent_292_config = json.loads((PARENT_292 / 'config.json').read_text(encoding='utf8'))
assert parent_292_config['parent_275_checkpoint_sha256'] == digest(pose_path)
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == parent_292_config['synthetic_provenance']
RUN.mkdir(parents=True, exist_ok=False)

panel, panel_attempts = [], 0
panel_attempts_with_fragment, panel_accepted_with_fragment = 0, 0
rng = np.random.default_rng(SEED + 1)
with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                panel_attempts += 1
                subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = SEED + 1_000_000_000 + panel_attempts
                with torch.no_grad():
                    old, new = pair(context, subject_index, draw_seed)
                identity = old['provenance'][0]
                physical_id = identity['physical_section_id']
                fragment_pixels = int((old['valid_mask'] & old['fragment_displaced_mask']).sum())
                panel_attempts_with_fragment += fragment_pixels > 0
                used = (bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES
                        and physical_id not in prior_ids)
                draws.write(json.dumps({'base_index': base, 'slot': slot,
                    'attempt': panel_attempts, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'physical_section_id': physical_id,
                    'fragment_pixels': fragment_pixels, 'used': used}, allow_nan=False) + '\n')
                if used:
                    break
            assert identity['split'] == 'train'
            prior_ids.add(physical_id)
            panel_accepted_with_fragment += fragment_pixels > 0
            indices, _, site_target = sites_from_sample(old, SITES)
            fragment = (old['valid_mask'][0] & old['fragment_displaced_mask'][0]).flatten().nonzero()[:, 0]
            intact = (old['valid_mask'][0] & ~old['fragment_displaced_mask'][0]).flatten().nonzero()[:, 0]
            if len(fragment):
                fragment = fragment[torch.linspace(0, len(fragment) - 1,
                    min(128, len(fragment)), device=fragment.device).long()]
            intact = intact[torch.linspace(0, len(intact) - 1,
                min(512, len(intact)), device=intact.device).long()]
            panel.append({'inputs': old['inputs'][0].cpu(), 'state': new['state'][0].cpu(),
                'reflection': old['reflection'][0].cpu(), 'offsets': old['offsets'][0].cpu(),
                'weights': old['weights'][0].cpu(), 'centre': old['centre'][0].cpu(),
                'valid_mask': old['valid_mask'][0].cpu(),
                'fragment_mask': old['fragment_displaced_mask'][0].cpu(),
                'site_indices': indices.cpu(), 'fragment_indices': fragment.cpu(),
                'intact_indices': intact.cpu(), 'physical_section_id': physical_id,
                'base_index': base, 'slot': slot, 'subject_index': subject_index,
                'draw_seed': draw_seed,
                'animal_id': identity['base_lineage']['animal_id'],
                'specimen_id': identity['base_lineage']['specimen_id'],
                'experiment_id': identity['base_lineage']['experiment_id'],
                'input_sha256': tensor_digest(old['inputs']),
                'target_sha256': tensor_digest(old['centre']),
                'site_target_sha256': tensor_digest(torch.cat((indices[:, None].float(),
                                                               site_target), -1)),
                'postartifact_state_sha256': tensor_digest(new['state']),
                'event_flags': identity['one_shot_slide_artifacts_v3']['events'],
                'provenance': new['provenance'][0]})
assert len(panel) == 64
torch.save(panel, RUN / 'inner_panel.pt')
print(json.dumps({'event': 'panel_ready', 'sections': len(panel),
                  'attempts': panel_attempts}), flush=True)

config = {'candidate': 294, 'seed': SEED,
    'parent_275_completed_sha256': digest(POSE_PARENT / 'completed.json'),
    'parent_275_checkpoint_sha256': digest(pose_path),
    'parent_292_completed_sha256': digest(PARENT_292 / 'completed.json'),
    'parent_292_selected_step': parent_292_receipt['selected_step'],
    'parent_292_checkpoint_sha256': digest(parent_292_path),
    'parent_292_training_sha256': digest(PARENT_292 / 'training.jsonl'),
    'parent_292_panel_sha256': digest(PARENT_292 / 'inner_panel.pt'),
    'control_289_completed_sha256': digest(CONTROL_289 / 'completed.json'),
    'control_289_checkpoint_sha256': digest(control_289_path),
    'control_289_training_sha256': digest(CONTROL_289 / 'training.jsonl'),
    'control_289_panel_sha256': digest(CONTROL_289 / 'inner_panel.pt'),
    'previous_287_completed_sha256': digest(PREVIOUS_287 / 'completed.json'),
    'previous_287_training_sha256': digest(PREVIOUS_287 / 'training.jsonl'),
    'previous_287_panel_sha256': digest(PREVIOUS_287 / 'inner_panel.pt'),
    'previous_286_completed_sha256': digest(PREVIOUS_286 / 'completed.json'),
    'previous_286_schedule_sha256': digest(PREVIOUS_286 / 'training_schedule.jsonl'),
    'previous_286_panel_sha256': digest(PREVIOUS_286 / 'inner_panel.pt'),
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'synthetic_provenance': context['provenance'],
    'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'panel_attempts_with_fragment': panel_attempts_with_fragment,
    'panel_accepted_with_fragment': panel_accepted_with_fragment,
    'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'updates': UPDATES, 'readout_steps': STEPS, 'sites_per_section': SITES,
    'batch_size': 1, 'map_grid_side': 192,
    'cost_volume_radii': {'48': 6, '96': 4, '192': 3},
    'preloaded_matcher': '292 selected own scratch-lineage checkpoint',
    'balanced_strata': 'equal intact/fragment means when both are eligible; single available otherwise',
    'stage_1_updates': 600, 'stage_1_loss': 'balanced three-scale exact offset CE only',
    'training_draw_rule': 'every even accepted update requires at least one displaced-fragment pixel; odd updates accept any eligible independent physical section',
    'fragment_required_updates': 1500,
    'stage_2_updates': 2400, 'stage_2_loss_weights': {
        'dense_endpoint_mm': 1., 'fragment_endpoint_mm': .5, 'balanced_offset_ce': .1,
        'within_piece_smoothness_mm': .01, 'validity_bce': .005,
        'balanced_move_gate_mse': .05},
    'gate_target': 'clipped [0,1] scalar projection of truth onto detached raw matcher move',
    'learning_rate': .0002, 'late_learning_rate': .00008,
    'late_after_update': 1800, 'weight_decay': 1e-4,
    'pose_trainable': False, 'gate_trainable_after_update': 600,
    'no_external_pretrained_weights_features_pseudolabels': True,
    'dev_cal_final_or_public_benchmark_used': False,
    'result_role': 'TRAIN-inner exact-pose map mechanism only; no blind/GUI claim',
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

parent_model = torch.load(parent_292_path, map_location='cpu', weights_only=False)['model']
pose_model = torch.load(pose_path, map_location='cpu', weights_only=False)['model']
for name, value in parent_model.items():
    if name.startswith('pose_model.'):
        key = name[len('pose_model.'):]
        if key.split('.')[0] in POSE_MODULES or key in POSE_BUFFERS:
            assert tensor_digest(value) == tensor_digest(pose_model[key])
model = GatedMultiScalePairJointModel().cuda()
missing, unexpected = model.load_state_dict(parent_model, strict=False)
assert set(missing) == {'move_gate_head.0.weight', 'move_gate_head.0.bias',
                        'move_gate_head.2.weight', 'move_gate_head.2.bias'}
assert not unexpected
del parent_model, pose_model
model.pose_model.requires_grad_(False)
model.move_gate_head.requires_grad_(False)
pose_hashes = {name: tensor_digest(value) for name, value in model.pose_model.state_dict().items()
               if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS}
parameters = [parameter for name, parameter in model.named_parameters()
              if not name.startswith('pose_model.')]
optimizer = torch.optim.AdamW(parameters, lr=.0002, weight_decay=1e-4)

with (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_stream:
    control_289 = SpatialPairJointModel().cuda().eval()
    control_289.load_state_dict(torch.load(control_289_path, map_location='cpu',
                                          weights_only=False)['model'], strict=True)
    control_289_report = readout(control_289, context['atlas'], panel, 289, 5000, readout_stream)
    del control_289
    torch.cuda.empty_cache()
    control_292 = MultiScalePairJointModel().cuda().eval()
    control_292.load_state_dict(torch.load(parent_292_path, map_location='cpu',
                                          weights_only=False)['model'], strict=True)
    control_292_report = readout(control_292, context['atlas'], panel, 292, 2400, readout_stream)
    del control_292
    torch.cuda.empty_cache()

    readouts, checkpoint_hashes, best_step, best_error = {}, {}, None, float('inf')
    accepted, training_attempts = 0, 0
    training_attempts_with_fragment = 0
    eligible_attempts_with_fragment = 0
    accepted_training_with_fragment = 0
    with (RUN / 'training.jsonl').open('w', encoding='utf8') as training_stream:
        for step in range(UPDATES + 1):
            if step in STEPS:
                report = readout(model, context['atlas'], panel, 294, step, readout_stream)
                readouts[str(step)] = report
                current_pose_hashes = {name: tensor_digest(value)
                    for name, value in model.pose_model.state_dict().items()
                    if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS}
                assert current_pose_hashes == pose_hashes
                checkpoint_path = RUN / f'joint_balanced_gated_pair_step_{step:05d}.pt'
                temporary = checkpoint_path.with_suffix('.tmp')
                torch.save({'candidate': 294, 'step': step,
                    'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    'config_sha256': digest(RUN / 'config.json'), 'calibrated': False}, temporary)
                os.replace(temporary, checkpoint_path)
                checkpoint_hashes[str(step)] = digest(checkpoint_path)
                mapped_error = report['groups']['all']['mapped_mm']
                if step > 0 and mapped_error < best_error:
                    best_step, best_error = step, mapped_error
                print(json.dumps({'event': 'readout', 'step': step,
                    'map_gain_mm': report['groups']['all']['gain_mm'],
                    'fragment_gain_mm': report['groups']['fragment']['gain_mm'],
                    'balanced_offset_ce': report['offset_classification']['balanced_cross_entropy']}),
                    flush=True)
            if step == UPDATES:
                break
            if step == 600:
                model.move_gate_head.requires_grad_(True)
            while True:
                training_attempts += 1
                base = int(rng.integers(56))
                subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = SEED + 2_000_000_000 + training_attempts
                with torch.no_grad():
                    old, new = pair(context, subject_index, draw_seed)
                identity = old['provenance'][0]
                fragment_pixels = int((old['valid_mask'] & old['fragment_displaced_mask']).sum())
                training_attempts_with_fragment += fragment_pixels > 0
                eligible = (bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES
                            and identity['physical_section_id'] not in prior_ids)
                eligible_attempts_with_fragment += eligible and fragment_pixels > 0
                if (eligible and ((step + 1) % 2 or fragment_pixels > 0)):
                    break
            assert identity['split'] == 'train'
            prior_ids.add(identity['physical_section_id'])
            accepted_training_with_fragment += fragment_pixels > 0
            indices, _, site_target = sites_from_sample(old, SITES)
            optimizer.zero_grad(set_to_none=True)
            mapped = model.map_at_pose(old['inputs'], new['state'], old['reflection'],
                                       old['offsets'], old['weights'], context['atlas'])
            valid = old['valid_mask']
            fragment = valid & old['fragment_displaced_mask']
            offset_ce, coverage = offset_loss(mapped, old['centre'], valid, fragment,
                                               new['state'], old['reflection'])
            dense = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] -
                     old['centre']).norm(dim=-1) / 1000
            endpoint = dense[valid].mean()
            fragment_endpoint = (dense[fragment].mean() if bool(fragment.any()) else
                                 endpoint.new_zeros(()))
            smooth = smoothness_loss(mapped['local_displacement_um'], valid, fragment,
                                     old['inputs'][:, :1])
            confidence = F.binary_cross_entropy_with_logits(
                mapped['correspondence_logit'], valid[:, None].float())
            move_gate = gate_loss(mapped, old['centre'], valid, fragment)
            loss = (offset_ce if step < 600 else
                    endpoint + .5 * fragment_endpoint + .1 * offset_ce + .01 * smooth +
                    .005 * confidence + .05 * move_gate)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            if step + 1 == 1800:
                for group in optimizer.param_groups:
                    group['lr'] = .00008
            accepted += 1
            training_stream.write(json.dumps({'step': step + 1,
                'stage': 'balanced_offset' if step < 600 else 'joint_map_gate',
                'fragment_required': (step + 1) % 2 == 0,
                'fragment_pixels': fragment_pixels,
                'physical_section_id': identity['physical_section_id'],
                'subject_index': subject_index, 'draw_seed': draw_seed,
                'base_index': subject_index // VARIANTS,
                'animal_id': identity['base_lineage']['animal_id'],
                'specimen_id': identity['base_lineage']['specimen_id'],
                'experiment_id': identity['base_lineage']['experiment_id'],
                'input_sha256': tensor_digest(old['inputs']),
                'target_sha256': tensor_digest(old['centre']),
                'site_target_sha256': tensor_digest(torch.cat((indices[:, None].float(),
                                                               site_target), -1)),
                'postartifact_state_sha256': tensor_digest(new['state']),
                'loss': float(loss.detach()), 'endpoint_mm': float(endpoint.detach()),
                'fragment_endpoint_mm': float(fragment_endpoint.detach())
                    if bool(fragment.any()) else None,
                'balanced_offset_cross_entropy': float(offset_ce.detach()),
                'offset_coverage': coverage, 'smoothness_mm': float(smooth.detach()),
                'validity_bce': float(confidence.detach()),
                'move_gate_mse': float(move_gate.detach()),
                'gate_mean_valid': float(mapped['move_gate'][:, 0][valid].mean().detach()),
                'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if (step + 1) % 600 == 0:
                training_stream.flush()
                print(json.dumps({'event': 'train_milestone', 'step': step + 1,
                    'stage': 'balanced_offset' if step < 600 else 'joint_map_gate',
                    'balanced_offset_ce': float(offset_ce.detach()),
                    'endpoint_mm': float(endpoint.detach())}), flush=True)

selected = readouts[str(best_step)]
gate = {'whole_panel_gain_at_least_0_030mm': selected['groups']['all']['gain_mm'] >= .030,
    'fragment_gain_at_least_0_070mm': selected['groups']['fragment']['gain_mm'] is not None
        and selected['groups']['fragment']['gain_mm'] >= .070,
    'fragment_sites_at_least_100': selected['groups']['fragment']['sites'] >= 100,
    'advantage_over_289_at_least_0_015mm':
        selected['groups']['all']['gain_mm'] - control_289_report['groups']['all']['gain_mm'] >= .015,
    'at_least_six_positive_bases': selected['positive_gain_bases'] >= 6,
    'intact_nonnegative': selected['groups']['intact']['gain_mm'] >= 0,
    'pose_hash_unchanged': current_pose_hashes == pose_hashes,
    'pose_first_choice_identical': all(
        report['pose_first_choice_sha256'] == readouts['0']['pose_first_choice_sha256']
        for report in readouts.values()),
    'balanced_panel_offset_ce_below_zero':
        selected['offset_classification']['balanced_cross_entropy'] <
        readouts['0']['offset_classification']['balanced_cross_entropy']}
files = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 294, 'selected_step': best_step,
    'accepted_training_sections': accepted, 'training_attempts': training_attempts,
    'training_attempts_with_fragment': training_attempts_with_fragment,
    'eligible_attempts_with_fragment': eligible_attempts_with_fragment,
    'accepted_training_with_fragment': accepted_training_with_fragment,
    'panel_sections': len(panel), 'panel_attempts': panel_attempts,
    'panel_attempts_with_fragment': panel_attempts_with_fragment,
    'panel_accepted_with_fragment': panel_accepted_with_fragment,
    'frozen_289_control': control_289_report, 'frozen_292_control': control_292_report,
    'readouts': readouts, 'gate': gate, 'advance': all(gate.values()),
    'pose_sha256': pose_hashes,
    'parent_sha256': {name: value for name, value in config.items()
                      if name.startswith(('parent_', 'control_', 'previous_'))
                      and name.endswith('_sha256')},
    'protocol_sha256': config['protocol_sha256'], 'source_sha256': config['source_sha256'],
    'checkpoint_sha256': checkpoint_hashes,
    'files_sha256': {name: digest(RUN / name) for name in files},
    'elapsed_seconds': time.perf_counter() - started,
    'result_role': config['result_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'selected_step': best_step,
    'gain_mm': selected['groups']['all']['gain_mm'], 'advance': completed['advance']}), flush=True)
