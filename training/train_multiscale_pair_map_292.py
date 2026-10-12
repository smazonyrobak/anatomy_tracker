"""Frozen TRAIN-inner exact-pose test of learned multiscale atlas correspondence."""
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
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_spatial_pair_map_v1 import SpatialPairJointModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.train_image_map_282 import (POSE_BUFFERS, POSE_MODULES, SIDE, digest,
                                          mapped_sites, sites_from_sample, tensor_digest)


SEED, UPDATES, SITES = 20261012292, 2400, 512
STEPS = (0, 400, 1200, 2400)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/MULTISCALE_PAIR_MAP_292_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
CONTROL = ROOT / 'runs/spatial_pair_map_289_002'
PREVIOUS = ROOT / 'runs/map_feature_287_001'
GAUGE = ROOT / 'runs/pose_gauge_map_286_001'
RUN = ROOT / 'runs/multiscale_pair_map_292_001'
SOURCE_NAMES = ('train_multiscale_pair_map_292.py', 'arbitrary_plane_multiscale_pair_map_v1.py',
    'arbitrary_plane_spatial_pair_map_v1.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py', 'arbitrary_plane_postartifact_gauge_v1.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py', 'train_image_map_282.py',
    'train_blind_fit_score_280.py', 'global_atlas_contrast_090.py')


def pair(context, subject_index, draw_seed):
    old = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
    new = sample_postartifact_gauge_v1(context, [subject_index], draw_seed, SIDE,
                                       source_sample=old)
    return old, new


def group_error(mapped, item, key, atlas):
    indices = item[key]
    if not len(indices):
        return None
    indices_cuda = indices.cuda()
    target = item['centre'].reshape(-1, 3)[indices].cuda()
    mapped_points = mapped_sites(mapped, indices_cuda)[0, 0]
    rigid_points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um':
        mapped['rigid_surface_ccf_ap_dv_ml_um'][:, None]}, indices_cuda)[0, 0]
    no_warp = float((rigid_points - target).norm(dim=-1).mean() / 1000)
    fitted = float((mapped_points - target).norm(dim=-1).mean() / 1000)
    mapped_support = render_finite_thickness_coordinate_grid(
        atlas[1:2], mapped_points[None, None, None], (0., 0., 0.),
        (25., 25., 25.), mapped_points.new_ones(1, 1))[0, 0, 0]
    return {'sites': len(indices), 'no_warp_mm': no_warp,
            'mapped_mm': fitted, 'gain_mm': no_warp - fitted,
            'mapped_support_below_half': int((mapped_support < .5).sum())}


def summary(rows):
    groups = {}
    for name in ('all', 'fragment', 'intact'):
        found = [row[name] for row in rows if row[name] is not None]
        count = sum(row['sites'] for row in found)
        groups[name] = {'sites': count}
        groups[name]['mapped_support_below_half'] = sum(
            row['mapped_support_below_half'] for row in found)
        for metric in ('no_warp_mm', 'mapped_mm', 'gain_mm'):
            groups[name][metric] = (float(sum(row[metric] * row['sites'] for row in found) / count)
                                    if count else None)
    bases = {str(base): float(np.mean([row['all']['gain_mm'] for row in rows
        if row['base_index'] == base])) for base in range(56, 64)}
    return {'sections': len(rows), 'groups': groups, 'gain_by_base_mm': bases,
            'positive_gain_bases': sum(gain > 0 for gain in bases.values()),
            'pose_first_choice_mean_mm': float(np.mean([row['pose_first_choice_mm'] for row in rows])),
            'field_absmax_mm': max(row['field_absmax_mm'] for row in rows),
            'dense_endpoint_mean_mm': float(np.mean([row['dense_endpoint_mm'] for row in rows])),
            'true_inplane_over_1_5mm_valid_fraction': float(np.mean([
                row['true_inplane_over_1_5mm_valid_fraction'] for row in rows])),
            'true_inplane_over_1_5mm_fragment_fraction': float(np.mean([
                row['true_inplane_over_1_5mm_fragment_fraction'] for row in rows
                if row['true_inplane_over_1_5mm_fragment_fraction'] is not None]))
                if any(row['true_inplane_over_1_5mm_fragment_fraction'] is not None for row in rows)
                else None,
            'true_normal_over_250um_valid_fraction': float(np.mean([
                row['true_normal_over_250um_valid_fraction'] for row in rows])),
            'valid_confidence_mean': float(np.mean([row['valid_confidence_mean'] for row in rows])),
            'invalid_confidence_mean': float(np.mean([row['invalid_confidence_mean'] for row in rows]))}


def readout(model, atlas, panel, step, output):
    rows = []
    model.eval()
    with torch.inference_mode():
        for item in panel:
            inputs = item['inputs'][None].cuda()
            mapped = model.map_at_pose(inputs, item['state'][None].cuda(),
                item['reflection'][None].cuda(), item['offsets'][None].cuda(),
                item['weights'][None].cuda(), atlas)
            pose = model.pose_model.predict(inputs)
            log_reflection = torch.stack((F.logsigmoid(-pose['reflection_logit']),
                                         F.logsigmoid(pose['reflection_logit'])), -1)
            choice = (pose['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)
            state = pose['state'][0, choice[0] // 2]
            reflection = choice[0] % 2
            indices = item['site_indices'].cuda()
            chart = torch.stack((indices.remainder(SIDE),
                indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            pose_points = rigid_points_090(state, reflection, chart, (SIDE, SIDE))
            pose_target = item['centre'].reshape(-1, 3)[item['site_indices']].cuda()
            valid = item['valid_mask'][None, None].cuda().float()
            fragment = item['fragment_indices']
            confidence = mapped['correspondence_logit'].sigmoid()
            dense_error = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] -
                           item['centre'][None].cuda()).norm(dim=-1) / 1000
            _, frame, _ = full_frame_state_to_components(item['state'][None].cuda())
            true_local = torch.einsum('bij,bhwj->bhwi', frame.transpose(1, 2),
                item['centre'][None].cuda() - mapped['rigid_surface_ccf_ap_dv_ml_um'])
            inplane_over = true_local[..., :2].norm(dim=-1) > 1500
            normal_over = true_local[..., 2].abs() > 250
            valid_flat = item['valid_mask'].cuda()
            frag_over = (float(inplane_over.flatten()[fragment.cuda()].float().mean())
                         if len(fragment) else None)
            row = {'step': step, 'physical_section_id': item['physical_section_id'],
                'base_index': item['base_index'], 'event_flags': item['event_flags'],
                'all': group_error(mapped, item, 'site_indices', atlas),
                'fragment': group_error(mapped, item, 'fragment_indices', atlas),
                'intact': group_error(mapped, item, 'intact_indices', atlas),
                'pose_first_choice_mm': float((pose_points - pose_target).norm(dim=-1).mean() / 1000),
                'field_absmax_mm': float(mapped['local_displacement_um'].abs().amax() / 1000),
                'dense_endpoint_mm': float((dense_error * valid[:, 0]).sum() / valid.sum()),
                'true_inplane_over_1_5mm_valid_fraction':
                    float(inplane_over[0][valid_flat].float().mean()),
                'true_inplane_over_1_5mm_fragment_fraction': frag_over,
                'true_normal_over_250um_valid_fraction':
                    float(normal_over[0][valid_flat].float().mean()),
                'valid_confidence_mean': float((confidence * valid).sum() / valid.sum()),
                'invalid_confidence_mean': float((confidence * (1 - valid)).sum() /
                                                 (1 - valid).sum())}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    output.flush()
    model.train()
    return summary(rows)


def offset_loss(mapped, target, valid, fragment, state, reflection):
    rigid = mapped['rigid_surface_ccf_ap_dv_ml_um']
    displacement = target - rigid
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
        else:
            # Sample one actual tissue point, never average CCF targets across a tear.
            desired = F.interpolate(canonical.permute(0, 3, 1, 2), (side, side),
                                    mode='nearest') * side
            fraction = F.avg_pool2d(valid[:, None].float(), SIDE // side)
            fragment_fraction = F.avg_pool2d(fragment[:, None].float(), SIDE // side)
            pure = ((fraction > .999) & ((fragment_fraction < .001) |
                                         (fragment_fraction > .999)))
        residual = (desired - level['base_flow'].detach()).round().long()
        inside = ((residual[:, 0].abs() <= radius) & (residual[:, 1].abs() <= radius)
                  & pure[:, 0])
        candidate = ((residual[:, 1] + radius) * (2 * radius + 1) +
                     residual[:, 0] + radius).clamp(0, (2 * radius + 1) ** 2 - 1)
        supported = level['support'].gather(1, candidate[:, None]).squeeze(1) > .9
        eligible = inside & supported
        coverage[str(side)] = {'pure_pixels': int(pure.sum()),
            'in_window_pixels': int(inside.sum()), 'supported_pixels': int(eligible.sum())}
        if bool(eligible.any()):
            losses.append(F.cross_entropy(level['logits'].permute(0, 2, 3, 1)[eligible],
                                          candidate[eligible]))
    return (torch.stack(losses).mean() if losses else target.new_zeros(())), coverage


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


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()
parent_receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
parent_path = PARENT / 'pose_step_08000.pt'
assert parent_receipt['selected_step'] == 8000
assert digest(parent_path) == parent_receipt['checkpoint_sha256']['8000']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
pose_weights = {name: value for name, value in parent['model'].items()
                if name.split('.')[0] in POSE_MODULES}
pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
del parent
control_receipt = json.loads((CONTROL / 'completed.json').read_text(encoding='utf8'))
assert control_receipt['selected_step'] == 5000
assert digest(CONTROL / 'completed.json') == \
    '5387c3514eeb91140f246a40240ec383d9c35f2ca917878eb195c7a1a87dd798'
assert digest(PREVIOUS / 'completed.json') == \
    '239d6d4f008029e331234bec0bcc088f3d5ab71c5fa5d856ddbd1e0120a6f63d'
assert digest(GAUGE / 'completed.json') == \
    '08b88ad1906303b33947dd55ce7e6d730883266f28682a8904b56b01a27551eb'
control_path = CONTROL / 'joint_spatial_pair_step_05000.pt'
assert digest(control_path) == control_receipt['checkpoint_sha256']['5000']
for name in ('training.jsonl', 'inner_panel.pt'):
    assert digest(CONTROL / name) == control_receipt['files_sha256'][name]
prior_ids = {json.loads(line)['physical_section_id']
             for line in (CONTROL / 'training.jsonl').open(encoding='utf8')}
prior_ids.update(json.loads(line)['physical_section_id'] for line in
                 (PREVIOUS / 'training.jsonl').open(encoding='utf8'))
for directory in (CONTROL, PREVIOUS, GAUGE):
    prior_ids.update(item['physical_section_id'] for item in
        torch.load(directory / 'inner_panel.pt', map_location='cpu', weights_only=False))
prior_ids.update(json.loads(line)['physical_section_id'] for line in
                 (GAUGE / 'training_schedule.jsonl').open(encoding='utf8'))
context = load_streaming_synthetic_v7_64(device='cuda')
RUN.mkdir(parents=True, exist_ok=False)

panel, attempts = [], 0
rng = np.random.default_rng(SEED + 1)
with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                attempts += 1
                subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = SEED + 1_000_000_000 + attempts
                with torch.no_grad():
                    old, new = pair(context, subject_index, draw_seed)
                physical_id = old['provenance'][0]['physical_section_id']
                used = (bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES
                        and physical_id not in prior_ids)
                draws.write(json.dumps({'base_index': base, 'slot': slot, 'attempt': attempts,
                    'subject_index': subject_index, 'draw_seed': draw_seed,
                    'physical_section_id': physical_id, 'used': used}, allow_nan=False) + '\n')
                if used:
                    break
            assert old['provenance'][0]['split'] == 'train'
            prior_ids.add(physical_id)
            indices, _, target = sites_from_sample(old, SITES)
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
                'valid_mask': old['valid_mask'][0].cpu(), 'site_indices': indices.cpu(),
                'fragment_indices': fragment.cpu(), 'intact_indices': intact.cpu(),
                'physical_section_id': physical_id, 'base_index': base, 'slot': slot,
                'subject_index': subject_index, 'draw_seed': draw_seed,
                'input_sha256': tensor_digest(old['inputs']),
                'target_sha256': tensor_digest(torch.cat((indices[:, None].float(), target), -1)),
                'event_flags': old['provenance'][0]['one_shot_slide_artifacts_v3']['events'],
                'provenance': new['provenance'][0]})
assert len(panel) == 64
torch.save(panel, RUN / 'inner_panel.pt')
print(json.dumps({'event': 'panel_ready', 'sections': len(panel), 'attempts': attempts}), flush=True)

config = {'candidate': 292, 'seed': SEED, 'parent_275_checkpoint_sha256': digest(parent_path),
    'parent_289_completed_sha256': digest(CONTROL / 'completed.json'),
    'parent_289_checkpoint_sha256': digest(control_path),
    'parent_289_training_sha256': digest(CONTROL / 'training.jsonl'),
    'parent_287_completed_sha256': digest(PREVIOUS / 'completed.json'),
    'parent_287_training_sha256': digest(PREVIOUS / 'training.jsonl'),
    'parent_286_completed_sha256': digest(GAUGE / 'completed.json'),
    'parent_286_schedule_sha256': digest(GAUGE / 'training_schedule.jsonl'),
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'synthetic_provenance': context['provenance'], 'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'updates': UPDATES, 'readout_steps': STEPS, 'sites_per_section': SITES,
    'map_grid_side': 192, 'cost_volume_radii': {'48': 6, '96': 4, '192': 3},
    'batch_size': 1, 'learning_rate': .0002, 'late_learning_rate': .00008,
    'late_start_step': 1200, 'endpoint_loss': 'dense valid Euclidean observed-pixel-to-CCF mm',
    'fragment_extra_weight': .5, 'offset_ce_weight': .004,
    'smoothness_weight': .01, 'confidence_weight': .005,
    'pose_trainable': False, 'no_external_pretrained_weights_features_pseudolabels': True,
    'dev_cal_final_or_public_benchmark_used': False,
    'result_role': 'TRAIN-inner exact-pose map mechanism only; no blind/GUI claim',
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

model = MultiScalePairJointModel().cuda()
for name in POSE_MODULES:
    prefix = name + '.'
    getattr(model.pose_model, name).load_state_dict({key[len(prefix):]: value
        for key, value in pose_weights.items() if key.startswith(prefix)}, strict=True)
with torch.no_grad():
    for name in POSE_BUFFERS:
        getattr(model.pose_model, name).copy_(pose_buffers[name])
model.pose_model.requires_grad_(False)
pose_hashes = {name: tensor_digest(value) for name, value in model.pose_model.state_dict().items()
               if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS}
parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=.0002, weight_decay=1e-4)

control_model = SpatialPairJointModel().cuda()
control_model.load_state_dict(torch.load(control_path, map_location='cpu', weights_only=False)['model'],
                              strict=True)
control_model.eval()
control_rows = []
with torch.inference_mode():
    for item in panel:
        mapped = control_model.map_at_pose(item['inputs'][None].cuda(), item['state'][None].cuda(),
            item['reflection'][None].cuda(), item['offsets'][None].cuda(),
            item['weights'][None].cuda(), context['atlas'])
        control_rows.append({'base_index': item['base_index'],
            'all': group_error(mapped, item, 'site_indices', context['atlas']),
            'fragment': group_error(mapped, item, 'fragment_indices', context['atlas']),
            'intact': group_error(mapped, item, 'intact_indices', context['atlas'])})
control_groups = {}
for name in ('all', 'fragment', 'intact'):
    found = [row[name] for row in control_rows if row[name] is not None]
    count = sum(row['sites'] for row in found)
    control_groups[name] = {'sites': count, 'gain_mm': float(sum(
        row['gain_mm'] * row['sites'] for row in found) / count) if count else None,
        'mapped_support_below_half': sum(row['mapped_support_below_half'] for row in found)}
del control_model
torch.cuda.empty_cache()
print(json.dumps({'event': 'frozen_control', 'groups': control_groups}), flush=True)

readouts, checkpoint_hashes, best_step, best_error = {}, {}, None, float('inf')
accepted, attempts_train = 0, 0
with (RUN / 'readouts.jsonl').open('w', encoding='utf8') as readout_stream, \
     (RUN / 'training.jsonl').open('w', encoding='utf8') as training_stream:
    for step in range(UPDATES + 1):
        if step in STEPS:
            report = readout(model, context['atlas'], panel, step, readout_stream)
            readouts[str(step)] = report
            current_hashes = {name: tensor_digest(value)
                for name, value in model.pose_model.state_dict().items()
                if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS}
            assert current_hashes == pose_hashes
            path = RUN / f'joint_multiscale_pair_step_{step:05d}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save({'candidate': 292, 'step': step,
                'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                'config_sha256': digest(RUN / 'config.json'), 'calibrated': False}, temporary)
            os.replace(temporary, path)
            checkpoint_hashes[str(step)] = digest(path)
            error = report['groups']['all']['mapped_mm']
            if step > 0 and error < best_error:
                best_step, best_error = step, error
            print(json.dumps({'event': 'readout', 'step': step,
                'map_gain_mm': report['groups']['all']['gain_mm'],
                'fragment_gain_mm': report['groups']['fragment']['gain_mm'],
                'dense_endpoint_mm': report['dense_endpoint_mean_mm'],
                'pose_first_choice_mm': report['pose_first_choice_mean_mm']}), flush=True)
        if step == UPDATES:
            break
        while True:
            attempts_train += 1
            base = int(rng.integers(56))
            subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
            draw_seed = SEED + 2_000_000_000 + attempts_train
            with torch.no_grad():
                old, new = pair(context, subject_index, draw_seed)
            identity = old['provenance'][0]
            if (bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES
                    and identity['physical_section_id'] not in prior_ids):
                break
        assert identity['split'] == 'train'
        prior_ids.add(identity['physical_section_id'])
        indices, _, target = sites_from_sample(old, SITES)
        optimizer.zero_grad(set_to_none=True)
        mapped = model.map_at_pose(old['inputs'], new['state'], old['reflection'],
                                   old['offsets'], old['weights'], context['atlas'])
        dense = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - old['centre']).norm(dim=-1) / 1000
        valid = old['valid_mask']
        fragment = valid & old['fragment_displaced_mask']
        endpoint = (dense * valid).sum() / valid.sum()
        fragment_endpoint = ((dense * fragment).sum() / fragment.sum()
                             if bool(fragment.any()) else endpoint.new_zeros(()))
        offset_ce, coverage = offset_loss(mapped, old['centre'], valid, fragment,
                                           new['state'], old['reflection'])
        smooth = smoothness_loss(mapped['local_displacement_um'], valid, fragment,
                                 old['inputs'][:, :1])
        confidence = F.binary_cross_entropy_with_logits(
            mapped['correspondence_logit'], valid[:, None].float())
        loss = endpoint + .5 * fragment_endpoint + .004 * offset_ce + .01 * smooth + .005 * confidence
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
        optimizer.step()
        if step + 1 == 1200:
            for group in optimizer.param_groups:
                group['lr'] = .00008
        accepted += 1
        training_stream.write(json.dumps({'step': step + 1,
            'physical_section_id': identity['physical_section_id'],
            'subject_index': subject_index, 'draw_seed': draw_seed,
            'base_index': subject_index // VARIANTS,
            'animal_id': identity['base_lineage']['animal_id'],
            'specimen_id': identity['base_lineage']['specimen_id'],
            'experiment_id': identity['base_lineage']['experiment_id'],
            'input_sha256': tensor_digest(old['inputs']),
            'target_sha256': tensor_digest(torch.cat((indices[:, None].float(), target), -1)),
            'postartifact_state_sha256': tensor_digest(new['state']),
            'endpoint_mm': float(endpoint.detach()),
            'fragment_endpoint_mm': float(fragment_endpoint.detach()) if bool(fragment.any()) else None,
            'offset_cross_entropy': float(offset_ce.detach()),
            'offset_coverage': coverage, 'confidence_loss': float(confidence.detach()),
            'smoothness_mm': float(smooth.detach()),
            'gradient_norm': float(gradient),
            'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
        if (step + 1) % 400 == 0:
            training_stream.flush()
            print(json.dumps({'event': 'train_milestone', 'step': step + 1,
                'endpoint_mm': float(endpoint.detach()),
                'fragment_endpoint_mm': float(fragment_endpoint.detach()) if bool(fragment.any()) else None,
                'offset_coverage': coverage}), flush=True)

selected = readouts[str(best_step)]
gate = {'gain_at_least_0_030mm': selected['groups']['all']['gain_mm'] >= .030,
    'fragment_gain_at_least_0_070mm': selected['groups']['fragment']['gain_mm'] is not None
        and selected['groups']['fragment']['gain_mm'] >= .070,
    'fragment_sites_at_least_100': selected['groups']['fragment']['sites'] >= 100,
    'advantage_over_289_at_least_0_015mm':
        selected['groups']['all']['gain_mm'] - control_groups['all']['gain_mm'] >= .015,
    'at_least_six_positive_bases': selected['positive_gain_bases'] >= 6,
    'intact_nonnegative': selected['groups']['intact']['gain_mm'] >= 0,
    'pose_hash_unchanged': current_hashes == pose_hashes,
    'pose_first_choice_identical': selected['pose_first_choice_mean_mm'] ==
        readouts['0']['pose_first_choice_mean_mm'],
    'field_within_1_5mm': selected['field_absmax_mm'] <= 1.5 + 1e-6}
files = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 292, 'selected_step': best_step,
    'accepted_training_sections': accepted, 'new_training_attempts': attempts_train,
    'panel_sections': len(panel), 'panel_attempts': attempts,
    'frozen_289_control_groups': control_groups, 'readouts': readouts,
    'gate': gate, 'advance': all(gate.values()), 'pose_sha256': pose_hashes,
    'source_sha256': config['source_sha256'], 'checkpoint_sha256': checkpoint_hashes,
    'files_sha256': {name: digest(RUN / name) for name in files},
    'elapsed_seconds': time.perf_counter() - started,
    'result_role': config['result_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'selected_step': best_step,
    'gain_mm': selected['groups']['all']['gain_mm'], 'advance': completed['advance']}), flush=True)
