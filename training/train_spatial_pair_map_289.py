"""Train a spatial atlas-pair map without changing the scratch-lineage pose predictor."""
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

from training.arbitrary_plane_spatial_pair_map_v1 import SpatialPairJointModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.train_image_map_282 import (MAP_MODULES, POSE_BUFFERS, POSE_MODULES,
    SIDE, digest, exact_map, make_model, mapped_sites, rigid_surface,
    sites_from_sample, tensor_digest)


SEED, UPDATES, SITES = 20261012289, 5000, 512
STEPS = (0, 1000, 3000, 5000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/SPATIAL_PAIR_MAP_289_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
CONTROL = ROOT / 'runs/pose_gauge_map_286_001'
PREVIOUS = ROOT / 'runs/map_feature_287_001'
RUN = ROOT / 'runs/spatial_pair_map_289_001'
SOURCE_NAMES = ('train_spatial_pair_map_289.py', 'arbitrary_plane_spatial_pair_map_v1.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'train_image_map_282.py', 'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py')


def pair(context, subject_index, draw_seed):
    old = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
    new = sample_postartifact_gauge_v1(context, [subject_index], draw_seed, SIDE,
                                        source_sample=old)
    return old, new


def readout(model, atlas, panel, step, stream):
    rows = []
    model.eval()
    with torch.inference_mode():
        for item in panel:
            inputs = item['inputs'][None].cuda()
            state = item['state'][None].cuda()
            reflection = item['reflection'][None].cuda()
            offsets = item['offsets'][None].cuda()
            weights = item['weights'][None].cuda()
            mapped = model.map_at_pose(inputs, state, reflection, offsets, weights, atlas)
            rigid = mapped['rigid_surface_ccf_ap_dv_ml_um']
            groups = {'all': item['site_indices'], 'fragment': item['fragment_indices'],
                      'intact': item['intact_indices']}
            errors = {}
            for name, indices_cpu in groups.items():
                if not len(indices_cpu):
                    continue
                indices = indices_cpu.cuda()
                target = item['centre'].reshape(-1, 3)[indices_cpu].cuda()
                points = mapped_sites(mapped, indices)[0, 0]
                rigid_points = mapped_sites(
                    {'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
                errors[name] = {'sites': len(indices),
                    'no_warp_mm': float((rigid_points - target).norm(dim=-1).mean() / 1000),
                    'mapped_mm': float((points - target).norm(dim=-1).mean() / 1000)}
                errors[name]['gain_mm'] = errors[name]['no_warp_mm'] - errors[name]['mapped_mm']
            valid = F.interpolate(item['valid_mask'][None, None].cuda().float(),
                                  (96, 96), mode='area')
            confidence = mapped['correspondence_logit'].sigmoid()
            predicted_pose = model.pose_model.predict(inputs)
            log_reflection = torch.stack((F.logsigmoid(-predicted_pose['reflection_logit']),
                                          F.logsigmoid(predicted_pose['reflection_logit'])), -1)
            chosen = (predicted_pose['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)
            chosen_state = predicted_pose['state'][0, chosen[0] // 2][None]
            chosen_reflection = (chosen % 2).to(reflection)
            chosen_surface, _, _ = rigid_surface(chosen_state, chosen_reflection)
            pose_points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um':
                                        chosen_surface[:, None]}, item['site_indices'].cuda())[0, 0]
            pose_target = item['centre'].reshape(-1, 3)[item['site_indices']].cuda()
            row = {'step': step, 'physical_section_id': item['physical_section_id'],
                'base_index': item['base_index'], 'event_flags': item['event_flags'],
                'all': errors['all'], 'fragment': errors.get('fragment'),
                'intact': errors.get('intact'),
                'pose_first_choice_mm': float((pose_points - pose_target).norm(dim=-1).mean() / 1000),
                'valid_confidence_mean': float((confidence * valid).sum() / valid.sum()),
                'invalid_confidence_mean': float((confidence * (1 - valid)).sum() / (1 - valid).sum()),
                'field_absmax_mm': float(mapped['local_displacement_um'].abs().amax() / 1000)}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    stream.flush()
    model.train()
    group_means = {}
    for group in ('all', 'fragment', 'intact'):
        values = [row[group] for row in rows if row[group] is not None]
        count = sum(value['sites'] for value in values)
        group_means[group] = {'sites': count, 'gain_mean_mm':
            float(sum(value['gain_mm'] * value['sites'] for value in values) / count) if count else None,
            'mapped_mean_mm': float(sum(value['mapped_mm'] * value['sites'] for value in values) / count)
                              if count else None,
            'no_warp_mean_mm': float(sum(value['no_warp_mm'] * value['sites'] for value in values) / count)
                                if count else None}
    bases = {str(base): float(np.mean([row['all']['gain_mm'] for row in rows
                                     if row['base_index'] == base])) for base in range(56, 64)}
    return {'sections': len(rows), 'groups': group_means, 'gain_by_base_mm': bases,
        'positive_gain_bases': sum(value > 0 for value in bases.values()),
        'pose_first_choice_mean_mm': float(np.mean([row['pose_first_choice_mm'] for row in rows])),
        'valid_confidence_mean': float(np.mean([row['valid_confidence_mean'] for row in rows])),
        'invalid_confidence_mean': float(np.mean([row['invalid_confidence_mean'] for row in rows])),
        'field_absmax_mm': max(row['field_absmax_mm'] for row in rows)}


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
assert digest(CONTROL / 'completed.json') == \
    '08b88ad1906303b33947dd55ce7e6d730883266f28682a8904b56b01a27551eb'
assert digest(CONTROL / 'training_schedule.jsonl') == \
    control_receipt['files_sha256']['training_schedule.jsonl']
control_path = CONTROL / 'postartifact_gauge_v1_step_03000.pt'
assert digest(control_path) == control_receipt['checkpoint_sha256']['postartifact_gauge_v1']['3000']
previous_receipt = json.loads((PREVIOUS / 'completed.json').read_text(encoding='utf8'))
assert digest(PREVIOUS / 'completed.json') == \
    '239d6d4f008029e331234bec0bcc088f3d5ab71c5fa5d856ddbd1e0120a6f63d'
assert digest(PREVIOUS / 'inner_panel.pt') == previous_receipt['files_sha256']['inner_panel.pt']
previous_panel = torch.load(PREVIOUS / 'inner_panel.pt', map_location='cpu', weights_only=False)
schedule = [json.loads(line) for line in (CONTROL / 'training_schedule.jsonl').open(encoding='utf8')]
assert len(schedule) == 3000
prior_ids = {entry['physical_section_id'] for entry in schedule}
prior_ids.update(item['physical_section_id'] for item in previous_panel)
context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == \
    json.loads((CONTROL / 'config.json').read_text())['synthetic_provenance']
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
                used = bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES
                draws.write(json.dumps({'base_index': base, 'slot': slot, 'attempt': attempts,
                    'subject_index': subject_index, 'draw_seed': draw_seed,
                    'physical_section_id': physical_id, 'used': used}, allow_nan=False) + '\n')
                if used:
                    break
            assert old['provenance'][0]['split'] == 'train' and physical_id not in prior_ids
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

config = {'candidate': 289, 'seed': SEED, 'parent_275_checkpoint_sha256': digest(parent_path),
    'parent_286_completed_sha256': digest(CONTROL / 'completed.json'),
    'parent_286_schedule_sha256': digest(CONTROL / 'training_schedule.jsonl'),
    'parent_286_control_checkpoint_sha256': digest(control_path),
    'parent_287_completed_sha256': digest(PREVIOUS / 'completed.json'),
    'parent_287_panel_sha256': digest(PREVIOUS / 'inner_panel.pt'),
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'synthetic_provenance': context['provenance'], 'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'updates': UPDATES, 'steps': STEPS, 'sites_per_section': SITES,
    'map_grid_side': 96, 'cost_volume_offsets': 17 * 17,
    'learning_rate': .0002, 'late_learning_rate': .00008, 'late_start_step': 3000,
    'coordinate_loss': 'mean Euclidean observed-pixel-to-CCF error in mm',
    'confidence_loss_weight': .01, 'within_piece_edge_smoothness_weight': .02,
    'pose_trainable': False, 'no_external_pretrained_weights_features_pseudolabels': True,
    'dev_cal_final_or_public_benchmark_used': False,
    'result_role': 'TRAIN-inner exact-pose map mechanism stage; no blind/GUl claim',
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

model = SpatialPairJointModel().cuda()
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
control_model = make_model(pose_weights, pose_buffers, 16)
control_weights = torch.load(control_path, map_location='cpu', weights_only=False)['mapper']
for name in MAP_MODULES:
    prefix = name + '.'
    getattr(control_model, name).load_state_dict({key[len(prefix):]: value
        for key, value in control_weights.items() if key.startswith(prefix)}, strict=True)
control_model.eval()
control_rows = []
with torch.inference_mode():
    for item in panel:
        indices = item['site_indices'].cuda()
        target = item['centre'].reshape(-1, 3)[item['site_indices']].cuda()
        mapped, points = exact_map(control_model, item['inputs'][None].cuda(),
            item['state'][None].cuda(), item['reflection'][None].cuda(),
            item['offsets'][None].cuda(), item['weights'][None].cuda(), context['atlas'], indices)
        rigid = mapped_sites({'centre_surface_ccf_ap_dv_ml_um':
            rigid_surface(item['state'][None].cuda(), item['reflection'][None].cuda())[0][:, None]},
            indices)[0, 0]
        control_rows.append(float(((rigid - target).norm(dim=-1).mean()
            - (points - target).norm(dim=-1).mean()) / 1000))
control_gain = float(np.mean(control_rows))
del control_model, control_weights
torch.cuda.empty_cache()
print(json.dumps({'event': 'frozen_control', 'gain_mm': control_gain}), flush=True)

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
            path = RUN / f'joint_spatial_pair_step_{step:05d}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save({'candidate': 289, 'step': step,
                'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                'config_sha256': digest(RUN / 'config.json'), 'calibrated': False}, temporary)
            os.replace(temporary, path)
            checkpoint_hashes[str(step)] = digest(path)
            error = report['groups']['all']['mapped_mean_mm']
            if step > 0 and error < best_error:
                best_step, best_error = step, error
            print(json.dumps({'event': 'readout', 'step': step, 'map_gain_mm':
                report['groups']['all']['gain_mean_mm'], 'mapped_mm': error,
                'fragment_gain_mm': report['groups']['fragment']['gain_mean_mm'],
                'pose_first_choice_mm': report['pose_first_choice_mean_mm']}), flush=True)
        if step == UPDATES:
            break
        if step < len(schedule):
            entry = schedule[step]
            subject_index, draw_seed = entry['subject_index'], entry['draw_seed']
            with torch.no_grad():
                old, new = pair(context, subject_index, draw_seed)
            assert old['provenance'][0]['physical_section_id'] == entry['physical_section_id']
            indices, _, target = sites_from_sample(old, SITES)
            assert tensor_digest(old['inputs']) == entry['input_sha256']
            assert tensor_digest(torch.cat((indices[:, None].float(), target), -1)) == entry['target_sha256']
            assert tensor_digest(new['state']) == entry['postartifact_state_sha256']
        else:
            while True:
                attempts_train += 1
                base = int(rng.integers(56))
                subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = SEED + 2_000_000_000 + attempts_train
                with torch.no_grad():
                    old, new = pair(context, subject_index, draw_seed)
                if bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= SITES:
                    break
            indices, _, target = sites_from_sample(old, SITES)
        identity = old['provenance'][0]
        assert identity['split'] == 'train'
        assert step < 3000 or identity['physical_section_id'] not in prior_ids
        prior_ids.add(identity['physical_section_id'])
        optimizer.zero_grad(set_to_none=True)
        mapped = model.map_at_pose(old['inputs'], new['state'], old['reflection'],
                                   old['offsets'], old['weights'], context['atlas'])
        points = mapped_sites(mapped, indices)[0, 0]
        coordinate_loss = (points - target).norm(dim=-1).mean() / 1000
        fragment_indices = (old['valid_mask'][0] & old['fragment_displaced_mask'][0]).flatten().nonzero()[:, 0]
        if len(fragment_indices):
            fragment_indices = fragment_indices[torch.linspace(0, len(fragment_indices) - 1,
                min(128, len(fragment_indices)), device=fragment_indices.device).long()]
            fragment_points = mapped_sites(mapped, fragment_indices)[0, 0]
            fragment_target = old['centre'][0].reshape(-1, 3)[fragment_indices]
            fragment_loss = (fragment_points - fragment_target).norm(dim=-1).mean() / 1000
        else:
            fragment_loss = coordinate_loss.new_zeros(())
        valid = F.interpolate(old['valid_mask'][:, None].float(), (96, 96), mode='area')
        confidence_loss = F.binary_cross_entropy_with_logits(mapped['correspondence_logit'], valid)
        local = mapped['local_displacement_um'] / 1000
        tissue = F.interpolate(old['valid_mask'][:, None].float(), (96, 96), mode='nearest') > .5
        fragment = F.interpolate(old['fragment_displaced_mask'][:, None].float(),
                                  (96, 96), mode='nearest') > .5
        image = F.interpolate(old['inputs'][:, :1], (96, 96), mode='area')
        same_x = tissue[..., 1:] & tissue[..., :-1] & (fragment[..., 1:] == fragment[..., :-1])
        same_y = tissue[..., 1:, :] & tissue[..., :-1, :] & (fragment[..., 1:, :] == fragment[..., :-1, :])
        edge_x = torch.exp(-6 * (image[..., 1:] - image[..., :-1]).abs()) * same_x
        edge_y = torch.exp(-6 * (image[..., 1:, :] - image[..., :-1, :]).abs()) * same_y
        smoothness = (((local[..., 1:] - local[..., :-1]).abs() * edge_x).sum()
            / (3 * edge_x.sum()).clamp_min(1) +
            ((local[..., 1:, :] - local[..., :-1, :]).abs() * edge_y).sum()
            / (3 * edge_y.sum()).clamp_min(1)) / 2
        loss = coordinate_loss + .25 * fragment_loss + .01 * confidence_loss + .02 * smoothness
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
        optimizer.step()
        if step + 1 == 3000:
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
            'coordinate_loss_mm': float(coordinate_loss.detach()),
            'fragment_loss_mm': float(fragment_loss.detach()) if len(fragment_indices) else None,
            'confidence_loss': float(confidence_loss.detach()),
            'smoothness_mm': float(smoothness.detach()),
            'gradient_norm': float(gradient),
            'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
        if (step + 1) % 1000 == 0:
            training_stream.flush()
            print(json.dumps({'event': 'train_milestone', 'step': step + 1,
                'coordinate_loss_mm': float(coordinate_loss.detach())}), flush=True)

selected = readouts[str(best_step)]
gate = {'gain_at_least_0_02mm': selected['groups']['all']['gain_mean_mm'] >= .02,
    'advantage_over_control_at_least_0_01mm':
        selected['groups']['all']['gain_mean_mm'] - control_gain >= .01,
    'at_least_six_positive_bases': selected['positive_gain_bases'] >= 6,
    'fragment_sites_at_least_20': selected['groups']['fragment']['sites'] >= 20,
    'positive_fragment_gain': selected['groups']['fragment']['gain_mean_mm'] is not None
        and selected['groups']['fragment']['gain_mean_mm'] > 0,
    'intact_nonnegative': selected['groups']['intact']['gain_mean_mm'] >= 0,
    'pose_hash_unchanged': current_hashes == pose_hashes,
    'pose_first_choice_identical': selected['pose_first_choice_mean_mm'] ==
        readouts['0']['pose_first_choice_mean_mm'],
    'field_within_1_5mm': selected['field_absmax_mm'] <= 1.5 + 1e-6}
files = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
         'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 289, 'selected_step': best_step, 'accepted_training_sections': accepted,
    'new_training_attempts': attempts_train, 'panel_sections': len(panel),
    'panel_attempts': attempts, 'frozen_286_control_gain_mm': control_gain,
    'readouts': readouts, 'gate': gate, 'advance': all(gate.values()),
    'pose_sha256': pose_hashes, 'source_sha256': config['source_sha256'],
    'checkpoint_sha256': checkpoint_hashes,
    'files_sha256': {name: digest(RUN / name) for name in files},
    'elapsed_seconds': time.perf_counter() - started,
    'result_role': config['result_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'selected_step': best_step,
    'gain_mm': selected['groups']['all']['gain_mean_mm'], 'advance': completed['advance']}), flush=True)
