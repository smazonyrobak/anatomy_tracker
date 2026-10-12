"""Matched TRAIN-only sparse-versus-dense exact-plane image-map supervision pilot."""
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
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.train_image_map_282 import (LIMIT, MAP_MODULES, PARENT, POSE_BUFFERS,
    POSE_MODULES, SEED as SEED282, SIDE, digest, exact_map, make_model, mapped_sites,
    mapper_digest, readout, rigid_surface, sites_from_sample, tensor_digest)


SEED, UPDATES = 20261012284, 3000
STEPS = (0, 750, 1500, 3000)
ARMS = {'sparse128': 128, 'dense2048': 2048}
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/DENSE_MAP_SUPERVISION_284_PROTOCOL_20261012.md'
RUN = ROOT / 'runs/dense_map_supervision_284_001'
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

parent_receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
parent_path = PARENT / 'pose_step_08000.pt'
assert digest(parent_path) == parent_receipt['checkpoint_sha256']['8000']
assert digest(PARENT / 'config.json') == parent_receipt['config_sha256']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
assert parent['step'] == 8000 and parent['config']['candidate'] == 275
pose_weights = {name: value for name, value in parent['model'].items()
                if name.split('.')[0] in POSE_MODULES}
pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
del parent
context = load_streaming_synthetic_v7_64(device='cuda')
source_names = ('train_dense_map_284.py', 'train_image_map_282.py',
    'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_joint_model_v7.py',
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
    'arbitrary_plane_allen_atlas_binding_v6.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'global_atlas_contrast_090.py')
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
                valid_pixels = int(sample['valid_mask'][0].sum())
                used = bool(sample['eligible'][0]) and valid_pixels >= 512
                draws.write(json.dumps({'base_index': base, 'slot': slot,
                    'attempt': panel_attempts, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'used': used, 'valid_pixels': valid_pixels,
                    'provenance': identity}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            lineage = context['bases'][base]['lineage']
            assert identity['split'] == 'train'
            assert all(identity['base_lineage'][key] == lineage[key] for key in
                       ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
            assert identity['physical_section_id'] not in panel_ids
            panel_ids.add(identity['physical_section_id'])
            indices, _, target = sites_from_sample(sample, 512)
            assert len(indices) == 512
            with torch.inference_mode():
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
print(json.dumps({'event': 'panel_ready', 'sections': 64,
                  'attempts': panel_attempts}), flush=True)

config = {'candidate': 284, 'seed': SEED, 'protocol_sha256': digest(PROTOCOL),
    'mapper_initialization_seed': SEED282 + 2,
    'source_sha256': source_hashes, 'parent_candidate': 275,
    'parent_completed_sha256': digest(PARENT / 'completed.json'),
    'parent_checkpoint_sha256': digest(parent_path),
    'synthetic_provenance': context['provenance'],
    'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
    'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'panel_attempts': panel_attempts, 'arms': ARMS,
    'updates_per_arm': UPDATES, 'readout_steps': STEPS,
    'train_base_indices': list(range(56)),
    'inner_base_indices': list(range(56, 64)),
    'sections_per_inner_base': 8, 'source_shape': [SIDE, SIDE],
    'map_grid_shape': [64, 64], 'readout_sites': 512,
    'map_derivative_limit': LIMIT,
    'optimizer': 'matched AdamW lr=1e-4 weight_decay=1e-4 clip=2 batch=1',
    'objective': 'exact-plane mean Euclidean surviving-site CCF error in mm',
    'feature_lineage': 'frozen 275 step-8000 scratch pose encoder/lateral; true state/reflection overrides pose',
    'checkpoint_selection': 'none; fixed 0,750,1500,3000 readouts; only 3000 eligible for gate',
    'result_role': 'TRAIN-inner supervision-density diagnostic, not a release model or acquired location validation',
    'dev_cal_final_or_public_benchmark_used': False,
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

schedule, readouts, initial_hashes, train_counts = [], {}, {}, {}
training_attempts = 0
with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, \
     (RUN / 'training_schedule.jsonl').open('w', encoding='utf8') as accepted_draws, \
     (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
     (RUN / 'readouts.jsonl').open('w', encoding='utf8') as output:
    for arm, site_budget in ARMS.items():
        model = make_model(pose_weights, pose_buffers, 16)
        initial_hashes[arm] = mapper_digest(model)
        assert initial_hashes[arm] == initial_hashes['sparse128']
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
        train_rng = np.random.default_rng(SEED + 3)
        attempted = 0
        readouts[arm], train_counts[arm] = {}, []
        for step in range(UPDATES + 1):
            if step in STEPS:
                readouts[arm][step] = readout(model, context['atlas'], panel, arm, step, output)
                output.flush()
                checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'candidate': 284, 'arm': arm, 'step': step,
                    'mapper': {name: value.cpu() for name, value in model.state_dict().items()
                        if name.split('.')[0] in MAP_MODULES},
                    'optimizer': optimizer.state_dict(),
                    'config_sha256': digest(RUN / 'config.json'),
                    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
                    'initial_mapper_sha256': initial_hashes[arm],
                    'calibrated': False}, temporary)
                os.replace(temporary, checkpoint)
                report = readouts[arm][step]
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                    'gain_mean_mm': report['gain_mean_mm'],
                    'nonnegative_gain_bases': report['nonnegative_gain_bases'],
                    'max_derivative': report['max_postlimit_derivative_bound']}), flush=True)
            if step == UPDATES:
                break
            update = step + 1
            if arm == 'sparse128':
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
                    draws.write(json.dumps({'step': update, 'attempt': attempted,
                        'base_index': base, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'used': used,
                        'provenance': identity}, allow_nan=False) + '\n')
                    if used:
                        break
                    del sample
            else:
                entry = schedule[step]
                subject_index, draw_seed = entry['subject_index'], entry['draw_seed']
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v9(
                        context, [subject_index], draw_seed, SIDE)
                identity = sample['provenance'][0]
                assert bool(sample['eligible'][0])
                assert identity['physical_section_id'] == entry['physical_section_id']
            base = subject_index // VARIANTS
            lineage = context['bases'][base]['lineage']
            assert identity['split'] == 'train'
            assert all(identity['base_lineage'][key] == lineage[key] for key in
                       ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
            sparse_indices, _, sparse_target = sites_from_sample(sample, 128)
            dense_indices, _, dense_target = sites_from_sample(sample, 2048)
            assert len(sparse_indices) == 128 and len(dense_indices) >= 128
            input_sha = tensor_digest(sample['inputs'])
            sparse_sha = tensor_digest(torch.cat((sparse_indices[:, None].float(), sparse_target), -1))
            dense_sha = tensor_digest(torch.cat((dense_indices[:, None].float(), dense_target), -1))
            state_sha = tensor_digest(torch.cat((sample['state'],
                sample['reflection'][:, None].float(), sample['offsets'], sample['weights']), -1))
            if arm == 'sparse128':
                entry = {'step': update, 'base_index': base, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'physical_section_id': identity['physical_section_id'],
                    'input_sha256': input_sha, 'sparse_target_sha256': sparse_sha,
                    'dense_target_sha256': dense_sha, 'state_psf_sha256': state_sha,
                    'dense_site_count': len(dense_indices)}
                schedule.append(entry)
                accepted_draws.write(json.dumps(entry, allow_nan=False) + '\n')
            else:
                assert (input_sha, sparse_sha, dense_sha, state_sha, len(dense_indices)) == (
                    entry['input_sha256'], entry['sparse_target_sha256'],
                    entry['dense_target_sha256'], entry['state_psf_sha256'], entry['dense_site_count'])
            indices, target = ((sparse_indices, sparse_target) if arm == 'sparse128'
                               else (dense_indices, dense_target))
            optimizer.zero_grad(set_to_none=True)
            mapped, points = exact_map(model, sample['inputs'], sample['state'],
                sample['reflection'], sample['offsets'], sample['weights'], context['atlas'], indices)
            loss = (points - target).norm(dim=-1).mean() / 1000
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            train_counts[arm].append(len(indices))
            training.write(json.dumps({'arm': arm, 'step': update,
                'physical_section_id': identity['physical_section_id'],
                'input_sha256': input_sha, 'site_target_sha256': sparse_sha if arm == 'sparse128' else dense_sha,
                'site_count': len(indices), 'loss_mm_before_update': float(loss.detach()),
                'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if update % 750 == 0:
                draws.flush()
                accepted_draws.flush()
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    'step': update, 'loss_mm': float(loss.detach()),
                    'site_count': len(indices)}), flush=True)
            del sample, mapped, points, loss
        del model, optimizer
        torch.cuda.empty_cache()
        if arm == 'sparse128':
            training_attempts = attempted

assert len(schedule) == len({entry['physical_section_id'] for entry in schedule}) == UPDATES
s, d = (readouts[arm][UPDATES] for arm in ARMS)
gate = {'dense_gain_at_least_0_02mm': d['gain_mean_mm'] >= .02,
    'dense_at_least_six_nonnegative_bases': d['nonnegative_gain_bases'] >= 6,
    'dense_derivative_at_most_0_35': d['max_postlimit_derivative_bound'] <= LIMIT + 1e-5,
    'dense_advantage_over_sparse_at_least_0_01mm': d['gain_mean_mm'] - s['gain_mean_mm'] >= .01,
    'all_arms_derivative_at_most_0_35': all(
        readouts[arm][step]['max_postlimit_derivative_bound'] <= LIMIT + 1e-5
        for arm in ARMS for step in STEPS)}
output_names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
    'training_draws.jsonl', 'training_schedule.jsonl', 'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 284, 'seed': SEED, 'steps': STEPS,
    'panel_sections': len(panel), 'panel_attempts': panel_attempts,
    'training_schedule_sections': len(schedule),
    'training_attempts': training_attempts, 'initial_mapper_sha256': initial_hashes,
    'train_site_count': {arm: {'min': min(values), 'mean': float(np.mean(values)),
        'max': max(values)} for arm, values in train_counts.items()},
    'readouts': readouts, 'step_3000_dense_minus_sparse_gain_mm': d['gain_mean_mm'] - s['gain_mean_mm'],
    'gate': gate, 'dense_supervision_signal': all(gate[key] for key in (
        'dense_gain_at_least_0_02mm', 'dense_at_least_six_nonnegative_bases',
        'dense_derivative_at_most_0_35', 'dense_advantage_over_sparse_at_least_0_01mm')),
    'result_role': config['result_role'], 'protocol_sha256': config['protocol_sha256'],
    'source_sha256': source_hashes, 'input_sha256': context['provenance']['input_sha256'],
    'output_sha256': {name: digest(RUN / name) for name in output_names},
    'checkpoint_sha256': {f'{arm}_{step}': digest(RUN / f'{arm}_step_{step:05d}.pt')
        for arm in ARMS for step in STEPS},
    'elapsed_seconds': time.perf_counter() - started,
    'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'complete', 'gate': gate,
    'dense_supervision_signal': completed['dense_supervision_signal']}), flush=True)
