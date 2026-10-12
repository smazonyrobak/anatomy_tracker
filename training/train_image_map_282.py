"""TRAIN-inner matched image-to-exact-plane-map mechanism test; no checkpoint selection."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.run_exact_map_capacity_281 import rigid_surface
from training.train_blind_fit_score_280 import mapped_sites, sites_from_sample


SEED, SIDE, GRID, UPDATES = 20261012282, 192, 64, 1500
STEPS = (0, 500, 1500)
TRAIN_SITES, READOUT_SITES, LIMIT = 128, 512, .35
ARMS = {'smooth16_full': (16, 'full'), 'free64_full': (64, 'full'),
        'smooth16_support': (16, 'support'), 'smooth16_zero': (16, 'zero')}
POSE_MODULES = ('encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global')
POSE_BUFFERS = ('normal_anchor_frames', 'center_origin', 'center_scale',
                'euclidean_scale', 'rotation_grid')
MAP_MODULES = ('warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair')
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/IMAGE_MAP_282_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
RUN = ROOT / 'runs/image_map_282_002'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def mapper_digest(model):
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.split('.')[0] in MAP_MODULES:
            sha.update(name.encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def make_model(pose_weights, pose_buffers, control_side):
    torch.manual_seed(SEED + 2)
    torch.cuda.manual_seed_all(SEED + 2)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, warp_control_side=control_side).cuda()
    for name in POSE_MODULES:
        prefix = name + '.'
        getattr(model, name).load_state_dict({key[len(prefix):]: value
            for key, value in pose_weights.items() if key.startswith(prefix)}, strict=True)
    with torch.no_grad():
        for name in POSE_BUFFERS:
            getattr(model, name).copy_(pose_buffers[name])
    model.requires_grad_(False)
    for name in MAP_MODULES:
        getattr(model, name).requires_grad_(True)
    return model


def exact_map(model, inputs, state, reflection, offsets, weights, atlas, indices):
    prediction = model.predict(inputs)
    prediction['state'] = state[:, None]
    mapped = model.map(prediction, offsets,
        torch.zeros((len(inputs), 1), device=inputs.device, dtype=torch.long),
        reflection[:, None], (GRID, GRID), atlas, weights,
        feature_side=GRID, source_shape=(SIDE, SIDE), max_derivative=LIMIT)
    points = mapped_sites(mapped, indices)[0, 0]
    return mapped, points


def subgroup(rows):
    if not rows:
        return {'n': 0, 'same_grid_no_warp_mean_mm': None,
            'mapped_mean_mm': None, 'gain_mean_mm': None}
    return {'n': len(rows),
        'same_grid_no_warp_mean_mm': float(np.mean([r['same_grid_no_warp_mm'] for r in rows])),
        'mapped_mean_mm': float(np.mean([r['exact_mapped_mm'] for r in rows])),
        'gain_mean_mm': float(np.mean([r['gain_mm'] for r in rows])),
        'field_displacement_mean_mm': float(np.mean([r['field_displacement_mean_mm'] for r in rows])),
        'field_displacement_rms_mm': float(np.mean([r['field_displacement_rms_mm'] for r in rows])),
        'surviving_site_displacement_mean_mm': float(np.mean([
            r['surviving_site_displacement_mean_mm'] for r in rows])),
        'surviving_site_displacement_rms_mm': float(np.mean([
            r['surviving_site_displacement_rms_mm'] for r in rows])),
        'max_postlimit_derivative_bound': max(r['postlimit_derivative_bound'] for r in rows)}


def readout(model, atlas, panel, arm, step, output):
    rows = []
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
            rigid_points = item['rigid_points'].cuda()
            mapped, points = exact_map(model, inputs, state, reflection,
                                        offsets, weights, atlas, indices)
            local = mapped['local_displacement_um'][:, 0]
            basis = full_frame_state_to_components(state)[2]
            canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
            derivative = ribbon_derivative_frobenius_bound(
                canonical, torch.zeros_like(canonical), basis, offsets)
            shift = (points - rigid_points).norm(dim=-1) / 1000
            oracle = (target - rigid_points).norm(dim=-1) / 1000
            magnitude = local.norm(dim=1) / 1000
            error = float((points - target).norm(dim=-1).mean() / 1000)
            artifact = item['provenance']['one_shot_slide_artifacts_v3']
            row = {'arm': arm, 'step': step, 'panel': 'new_v9_train_inner',
                'base_index': item['base_index'], 'slot': item['slot'],
                'subject_index': item['subject_index'], 'draw_seed': item['draw_seed'],
                'physical_section_id': item['physical_section_id'],
                'true_state': item['state'].tolist(), 'true_reflection': int(item['reflection']),
                'site_count': len(indices), 'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'sparse_support_under_15pct_visible': item['visible_fraction'] < .15,
                'obliquity_deg': item['obliquity_deg'],
                'event_flags': item['event_flags'],
                'tissue_placed': artifact['tissue_placed'],
                'artifact_pixel_counts': artifact['pixel_counts'],
                'same_grid_no_warp_mm': item['same_grid_no_warp_mm'],
                'exact_mapped_mm': error, 'gain_mm': item['same_grid_no_warp_mm'] - error,
                'field_displacement_mean_mm': float(magnitude.mean()),
                'field_displacement_rms_mm': float(magnitude.square().mean().sqrt()),
                'surviving_site_displacement_mean_mm': float(shift.mean()),
                'surviving_site_displacement_rms_mm': float(shift.square().mean().sqrt()),
                'oracle_surviving_site_displacement_mean_mm': float(oracle.mean()),
                'oracle_surviving_site_displacement_rms_mm': float(oracle.square().mean().sqrt()),
                'prelimit_derivative_bound': float(mapped['prelimit_derivative_bound'][0, 0]),
                'deformation_rescale': float(mapped['deformation_rescale'][0, 0]),
                'postlimit_derivative_bound': float(derivative[0]),
                'provenance': item['provenance']}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    model.train()
    summary = {**subgroup(rows),
        'by_base': {str(base): subgroup([r for r in rows if r['base_index'] == base])
                    for base in range(56, 64)},
        'by_sparse_support': {str(flag).lower(): subgroup([r for r in rows
            if r['sparse_support_under_15pct_visible'] == flag]) for flag in (False, True)},
        'by_event': {event: {str(flag).lower(): subgroup([r for r in rows
            if r['event_flags'][event] == flag]) for flag in (False, True)}
            for event in EVENTS}}
    summary['nonnegative_gain_bases'] = sum(
        summary['by_base'][str(base)]['gain_mean_mm'] >= 0 for base in range(56, 64))
    return summary


def main():
    assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
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
    source_names = ('train_image_map_282.py', 'run_exact_map_capacity_281.py',
        'train_blind_fit_score_280.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
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
        'arbitrary_plane_geometry.py', 'arbitrary_plane_full_frame_primitives.py',
        'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
    source_hashes = {name: digest(SOURCE / name) for name in source_names}
    RUN.mkdir(parents=True, exist_ok=False)

    panel, panel_attempts, panel_ids = [], 0, set()
    panel_rng = np.random.default_rng(SEED + 1)
    with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
        for base in range(56, 64):
            for slot in range(8):
                while True:
                    panel_attempts += 1
                    subject_index = base * VARIANTS + int(panel_rng.integers(VARIANTS))
                    draw_seed = SEED + 1_000_000_000 + panel_attempts
                    with torch.no_grad():
                        sample = sample_one_shot_slide_artifacts_v9(
                            context, [subject_index], draw_seed, SIDE)
                    identity = sample['provenance'][0]
                    valid_pixels = int(sample['valid_mask'][0].count_nonzero())
                    used = bool(sample['eligible'][0]) and valid_pixels >= READOUT_SITES
                    draws.write(json.dumps({'base_index': base, 'slot': slot,
                        'attempt': panel_attempts, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'used': used, 'valid_pixels': valid_pixels,
                        'provenance': identity}, allow_nan=False) + '\n')
                    if used:
                        break
                    del sample
                assert identity['split'] == 'train'
                lineage = context['bases'][base]['lineage']
                assert all(identity['base_lineage'][key] == lineage[key] for key in
                           ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
                assert identity['physical_section_id'] not in panel_ids
                panel_ids.add(identity['physical_section_id'])
                indices, _, target = sites_from_sample(sample, READOUT_SITES)
                assert len(indices) == READOUT_SITES
                rigid, _, _ = rigid_surface(sample['state'], sample['reflection'])
                rigid_points = mapped_sites(
                    {'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
                normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
                artifact = identity['one_shot_slide_artifacts_v3']
                panel.append({'inputs': sample['inputs'][0].cpu(),
                    'state': sample['state'][0].cpu(),
                    'reflection': sample['reflection'][0].cpu(),
                    'offsets': sample['offsets'][0].cpu(),
                    'weights': sample['weights'][0].cpu(),
                    'site_indices': indices.cpu(), 'site_target': target.cpu(),
                    'rigid_points': rigid_points.cpu(),
                    'same_grid_no_warp_mm': float((rigid_points - target).norm(dim=-1).mean() / 1000),
                    'base_index': base, 'slot': slot, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'physical_section_id': identity['physical_section_id'],
                    'valid_fraction': float(sample['valid_mask'][0].float().mean()),
                    'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                    'obliquity_deg': float(torch.rad2deg(torch.acos(normal.abs().max().clamp(0, 1)))),
                    'event_flags': {name: bool(artifact['events'][name]) for name in EVENTS},
                    'provenance': identity})
                del sample
    assert len(panel) == len(panel_ids) == 64
    torch.save(panel, RUN / 'inner_panel.pt')
    print(json.dumps({'event': 'panel_ready', 'sections': len(panel),
                      'attempts': panel_attempts}), flush=True)

    config = {'candidate': 282, 'protocol_sha256': digest(PROTOCOL),
        'source_sha256': source_hashes, 'seed': SEED, 'parent_candidate': 275,
        'parent_checkpoint_sha256': digest(checkpoint_path),
        'parent_completed_sha256': digest(PARENT / 'completed.json'),
        'parent_config_sha256': receipt['config_sha256'],
        'parent_pose_modules': POSE_MODULES, 'parent_pose_buffers': POSE_BUFFERS,
        'synthetic_provenance': context['provenance'],
        'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
        'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
        'panel_attempts': panel_attempts, 'arms': ARMS,
        'updates_per_arm': UPDATES, 'readout_steps': STEPS,
        'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
        'sections_per_inner_base': 8, 'source_shape': [SIDE, SIDE],
        'map_grid_shape': [GRID, GRID], 'train_surviving_sites': TRAIN_SITES,
        'readout_surviving_sites': READOUT_SITES,
        'map_derivative_limit': LIMIT,
        'objective': 'mean exact-plane surviving-site Euclidean CCF error in mm; no rank, reliability, or deformation penalty',
        'optimizer': 'fresh matched AdamW, lr=1e-4, weight_decay=1e-4, clip=2',
        'feature_lineage': 'frozen 275 step-8000 scratch pose encoder/lateral; true synthetic state/reflection override pose outputs',
        'atlas_rendering': 'true-plane intensity/support in A/B; true support only in C; both channels zero in D',
        'checkpoint_selection': 'none; fixed 0,500,1500 readouts; only 1500 eligible for gate',
        'result_role': 'TRAIN-inner mechanism diagnostic, not release model or real-animal location validation',
        'dev_cal_final_or_public_benchmark_used': False,
        'torch': torch.__version__, 'numpy': np.__version__}
    (RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

    schedule, readouts, counters, initial_hashes = [], {}, {}, {}
    atlas_masked = None
    with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, \
         (RUN / 'training_schedule.jsonl').open('w', encoding='utf8') as accepted_draws, \
         (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
         (RUN / 'readouts.jsonl').open('w', encoding='utf8') as output:
        for arm, (control_side, atlas_mode) in ARMS.items():
            model = make_model(pose_weights, pose_buffers, control_side)
            initial_hashes[arm] = mapper_digest(model)
            assert initial_hashes[arm] == initial_hashes['smooth16_full']
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
            if atlas_mode != 'full' and atlas_masked is None:
                atlas_masked = context['atlas'].clone()
                atlas_masked[0].zero_()
            if atlas_mode == 'zero':
                atlas_masked[1].zero_()
            atlas = context['atlas'] if atlas_mode == 'full' else atlas_masked
            train_rng = np.random.default_rng(SEED + 3)
            attempted = 0
            readouts[arm] = {}
            for step in range(UPDATES + 1):
                if step:
                    if arm == 'smooth16_full':
                        while True:
                            attempted += 1
                            base = int(train_rng.integers(56))
                            subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
                            draw_seed = SEED + 2_000_000_000 + attempted
                            with torch.no_grad():
                                sample = sample_one_shot_slide_artifacts_v9(
                                    context, [subject_index], draw_seed, SIDE)
                            identity = sample['provenance'][0]
                            used = bool(sample['eligible'][0])
                            draws.write(json.dumps({'step': step, 'attempt': attempted,
                                'base_index': base, 'subject_index': subject_index,
                                'draw_seed': draw_seed, 'used': used,
                                'provenance': identity}, allow_nan=False) + '\n')
                            if used:
                                break
                            del sample
                    else:
                        entry = schedule[step - 1]
                        subject_index, draw_seed = entry['subject_index'], entry['draw_seed']
                        with torch.no_grad():
                            sample = sample_one_shot_slide_artifacts_v9(
                                context, [subject_index], draw_seed, SIDE)
                        identity = sample['provenance'][0]
                        assert bool(sample['eligible'][0])
                        assert identity['physical_section_id'] == entry['physical_section_id']
                    assert identity['split'] == 'train'
                    lineage = context['bases'][subject_index // VARIANTS]['lineage']
                    assert all(identity['base_lineage'][key] == lineage[key] for key in
                               ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
                    indices, _, target = sites_from_sample(sample, TRAIN_SITES)
                    assert len(indices) == TRAIN_SITES
                    input_sha = tensor_digest(sample['inputs'])
                    target_sha = tensor_digest(torch.cat((indices[:, None].float(), target), -1))
                    state_sha = tensor_digest(torch.cat((sample['state'],
                        sample['reflection'][:, None].float(), sample['offsets'], sample['weights']), -1))
                    if arm == 'smooth16_full':
                        entry = {'step': step, 'base_index': subject_index // VARIANTS,
                            'subject_index': subject_index, 'draw_seed': draw_seed,
                            'physical_section_id': identity['physical_section_id'],
                            'input_sha256': input_sha, 'site_target_sha256': target_sha,
                            'state_psf_sha256': state_sha}
                        schedule.append(entry)
                        accepted_draws.write(json.dumps(entry, allow_nan=False) + '\n')
                    else:
                        assert (input_sha, target_sha, state_sha) == (entry['input_sha256'],
                            entry['site_target_sha256'], entry['state_psf_sha256'])
                    optimizer.zero_grad(set_to_none=True)
                    mapped, points = exact_map(model, sample['inputs'], sample['state'],
                        sample['reflection'], sample['offsets'], sample['weights'],
                        atlas, indices)
                    loss = (points - target).norm(dim=-1).mean() / 1000
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
                    optimizer.step()
                    training.write(json.dumps({'arm': arm, 'step': step,
                        'physical_section_id': identity['physical_section_id'],
                        'input_sha256': input_sha, 'site_target_sha256': target_sha,
                        'loss_mm_before_update': float(loss.detach()),
                        'gradient_norm': float(gradient),
                        'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
                    if step % 500 == 0:
                        draws.flush()
                        accepted_draws.flush()
                        training.flush()
                        print(json.dumps({'event': 'train_milestone', 'arm': arm,
                            'step': step, 'loss_mm': float(loss.detach())}), flush=True)
                    del sample, mapped, points, loss

                if step in STEPS:
                    draws.flush()
                    accepted_draws.flush()
                    training.flush()
                    readouts[arm][step] = readout(model, atlas, panel, arm, step, output)
                    output.flush()
                    checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save({'candidate': 282, 'arm': arm, 'step': step,
                        'mapper': {name: value.cpu() for name, value in model.state_dict().items()
                            if name.split('.')[0] in MAP_MODULES},
                        'optimizer': optimizer.state_dict(), 'config_sha256': digest(RUN / 'config.json'),
                        'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
                        'initial_mapper_sha256': initial_hashes[arm],
                        'calibrated': False}, temporary)
                    os.replace(temporary, checkpoint)
                    summary = readouts[arm][step]
                    print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                        'gain_mean_mm': summary['gain_mean_mm'],
                        'nonnegative_gain_bases': summary['nonnegative_gain_bases'],
                        'max_derivative': summary['max_postlimit_derivative_bound']}), flush=True)
            counters[arm] = {'attempted_draws_in_first_arm': attempted if arm == 'smooth16_full' else None,
                'accepted_sections': UPDATES, 'schedule_replayed': arm != 'smooth16_full'}
            del model, optimizer
            torch.cuda.empty_cache()

    assert len(schedule) == UPDATES
    assert len({entry['physical_section_id'] for entry in schedule}) == UPDATES
    a, b, c, d = (readouts[name][UPDATES] for name in ARMS)
    gate = {'smooth_architecture_mean_gain_at_least_0_05mm': a['gain_mean_mm'] >= .05,
        'smooth_architecture_at_least_six_nonnegative_bases': a['nonnegative_gain_bases'] >= 6,
        'smooth_architecture_derivative_at_most_0_35':
            a['max_postlimit_derivative_bound'] <= LIMIT + 1e-5,
        'directional_smooth_advantage_over_free_at_least_0_03mm':
            a['gain_mean_mm'] - b['gain_mean_mm'] >= .03,
        'atlas_intensity_value_at_least_0_03mm': a['gain_mean_mm'] - c['gain_mean_mm'] >= .03,
        'atlas_support_value_at_least_0_03mm': c['gain_mean_mm'] - d['gain_mean_mm'] >= .03,
        'all_arms_derivative_at_most_0_35': all(
            readouts[arm][step]['max_postlimit_derivative_bound'] <= LIMIT + 1e-5
            for arm in ARMS for step in STEPS)}
    output_names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
        'training_draws.jsonl', 'training_schedule.jsonl', 'training.jsonl', 'readouts.jsonl')
    completed = {'candidate': 282, 'seed': SEED, 'steps': STEPS,
        'panel_sections': len(panel), 'panel_attempts': panel_attempts,
        'training_schedule_sections': len(schedule), 'counters': counters,
        'initial_mapper_sha256': initial_hashes, 'readouts': readouts,
        'step_1500_gain_differences_mm': {'A_minus_B': a['gain_mean_mm'] - b['gain_mean_mm'],
            'A_minus_C': a['gain_mean_mm'] - c['gain_mean_mm'],
            'C_minus_D': c['gain_mean_mm'] - d['gain_mean_mm'],
            'A_minus_D': a['gain_mean_mm'] - d['gain_mean_mm']},
        'gate': gate,
        'smooth_architecture_learns': all(gate[key] for key in (
            'smooth_architecture_mean_gain_at_least_0_05mm',
            'smooth_architecture_at_least_six_nonnegative_bases',
            'smooth_architecture_derivative_at_most_0_35')),
        'result_role': config['result_role'],
        'protocol_sha256': config['protocol_sha256'], 'source_sha256': source_hashes,
        'input_sha256': context['provenance']['input_sha256'],
        'output_sha256': {name: digest(RUN / name) for name in output_names},
        'checkpoint_sha256': {f'{arm}_{step}': digest(
            RUN / f'{arm}_step_{step:05d}.pt') for arm in ARMS for step in STEPS},
        'elapsed_seconds': time.perf_counter() - started,
        'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
    (RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'gate': gate,
        'smooth_architecture_learns': completed['smooth_architecture_learns']}), flush=True)


if __name__ == '__main__':
    main()
