"""Matched TRAIN-only pre/post-artifact pose-gauge mapper comparison."""
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
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.train_image_map_282 import (LIMIT, MAP_MODULES, PARENT, POSE_BUFFERS,
    POSE_MODULES, SEED as SEED282, SIDE, digest, exact_map, make_model,
    mapped_sites, mapper_digest, rigid_surface, sites_from_sample, tensor_digest)


SEED, UPDATES = 20261012286, 3000
STEPS = (0, 750, 1500, 3000)
ARMS = ('preartifact_v9', 'postartifact_gauge_v1')
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/POSE_GAUGE_MAP_286_PROTOCOL_20261012.md'
RUN = ROOT / 'runs/pose_gauge_map_286_001'
SOURCE_NAMES = ('train_pose_gauge_map_286.py', 'train_image_map_282.py',
    'train_blind_fit_score_280.py', 'run_exact_map_capacity_281.py',
    'arbitrary_plane_postartifact_gauge_v1.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_stream.py',
    'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_recurrent_model.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_allen_atlas_binding_v6.py', 'global_atlas_contrast_090.py')


def paired_sample(context, subject_index, draw_seed):
    old = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
    new = sample_postartifact_gauge_v1(context, [subject_index], draw_seed, SIDE,
                                        source_sample=old)
    for key in ('inputs', 'centre', 'valid_mask', 'reflection', 'offsets', 'weights',
                'fragment_displaced_mask'):
        assert torch.equal(old[key], new[key]), key
    assert old['provenance'][0]['physical_section_id'] == new['provenance'][0]['physical_section_id']
    assert torch.equal(old['state'], new['pre_artifact_state'])
    return old, new


def evaluate(model, atlas, panel, arm, step, stream):
    rows = []
    model.eval()
    with torch.inference_mode():
        for item in panel:
            state = item['states'][arm][None].cuda()
            reflection = item['reflection'][None].cuda()
            offsets = item['offsets'][None].cuda()
            weights = item['weights'][None].cuda()
            indices = item['site_indices'].cuda()
            target = item['site_target'].cuda()
            inputs = item['inputs'][None].cuda()
            rigid, _, basis = rigid_surface(state, reflection)
            rigid_points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
            mapped, points = exact_map(model, inputs, state, reflection,
                                        offsets, weights, atlas, indices)
            local = mapped['local_displacement_um'][:, 0]
            canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
            derivative = ribbon_derivative_frobenius_bound(
                canonical, torch.zeros_like(canonical), basis, offsets)
            no_warp = float((rigid_points - target).norm(dim=-1).mean() / 1000)
            error = float((points - target).norm(dim=-1).mean() / 1000)
            shift = float((points - rigid_points).norm(dim=-1).mean() / 1000)
            row = {'arm': arm, 'step': step, 'base_index': item['base_index'],
                'slot': item['slot'], 'subject_index': item['subject_index'],
                'draw_seed': item['draw_seed'],
                'physical_section_id': item['physical_section_id'],
                'input_sha256': item['input_sha256'], 'target_sha256': item['target_sha256'],
                'no_warp_mm': no_warp, 'mapped_mm': error, 'map_gain_mm': no_warp - error,
                'surviving_site_shift_mm': shift,
                'prelimit_derivative': float(mapped['prelimit_derivative_bound'][0, 0]),
                'deformation_rescale': float(mapped['deformation_rescale'][0, 0]),
                'postlimit_derivative': float(derivative[0]),
                'valid_fraction': item['valid_fraction'],
                'visible_fraction': item['visible_fraction'],
                'event_flags': item['event_flags'],
                'pose_center_delta_mm': item['pose_center_delta_mm'],
                'pose_normal_delta_deg': item['pose_normal_delta_deg'],
                'provenance': item['provenance']}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    model.train()
    base_means = {str(base): float(np.mean([row['map_gain_mm'] for row in rows
                                            if row['base_index'] == base])) for base in range(56, 64)}
    return {'sections': len(rows),
        'no_warp_mean_mm': float(np.mean([row['no_warp_mm'] for row in rows])),
        'mapped_mean_mm': float(np.mean([row['mapped_mm'] for row in rows])),
        'map_gain_mean_mm': float(np.mean([row['map_gain_mm'] for row in rows])),
        'surviving_site_shift_mean_mm': float(np.mean([row['surviving_site_shift_mm'] for row in rows])),
        'max_postlimit_derivative': max(row['postlimit_derivative'] for row in rows),
        'nonnegative_gain_bases': sum(value >= 0 for value in base_means.values()),
        'gain_by_base_mm': base_means,
        'pose_center_delta_mean_mm': float(np.mean([row['pose_center_delta_mm'] for row in rows])),
        'pose_normal_delta_mean_deg': float(np.mean([row['pose_normal_delta_deg'] for row in rows]))}


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()
parent_receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
assert digest(PARENT / 'config.json') == parent_receipt['config_sha256']
parent_path = PARENT / 'pose_step_08000.pt'
assert digest(parent_path) == parent_receipt['checkpoint_sha256']['8000']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
assert parent['step'] == 8000 and parent['config']['candidate'] == 275
pose_weights = {name: value for name, value in parent['model'].items()
                if name.split('.')[0] in POSE_MODULES}
pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
del parent
context = load_streaming_synthetic_v7_64(device='cuda')
source_hashes = {name: digest(SOURCE / name) for name in SOURCE_NAMES}
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
                    old, new = paired_sample(context, subject_index, draw_seed)
                identity = old['provenance'][0]
                valid_pixels = int(old['valid_mask'][0].sum())
                used = bool(old['eligible'][0]) and valid_pixels >= 512
                draws.write(json.dumps({'attempt': panel_attempts, 'base_index': base,
                    'slot': slot, 'subject_index': subject_index, 'draw_seed': draw_seed,
                    'used': used, 'valid_pixels': valid_pixels,
                    'physical_section_id': identity['physical_section_id']}, allow_nan=False) + '\n')
                if used:
                    break
                del old, new
            assert identity['split'] == 'train'
            assert identity['physical_section_id'] not in panel_ids
            panel_ids.add(identity['physical_section_id'])
            indices, _, target = sites_from_sample(old, 512)
            assert len(indices) == 512
            old_center, old_frame, _ = full_frame_state_to_components(old['state'])
            new_center, new_frame, _ = full_frame_state_to_components(new['state'])
            angle = torch.rad2deg(torch.acos((old_frame[:, :, 2] *
                new_frame[:, :, 2]).sum(-1).abs().clamp(0, 1)))
            panel.append({'inputs': old['inputs'][0].cpu(),
                'states': {ARMS[0]: old['state'][0].cpu(), ARMS[1]: new['state'][0].cpu()},
                'reflection': old['reflection'][0].cpu(),
                'offsets': old['offsets'][0].cpu(), 'weights': old['weights'][0].cpu(),
                'site_indices': indices.cpu(), 'site_target': target.cpu(),
                'input_sha256': tensor_digest(old['inputs']),
                'target_sha256': tensor_digest(torch.cat((indices[:, None].float(), target), -1)),
                'base_index': base, 'slot': slot, 'subject_index': subject_index,
                'draw_seed': draw_seed, 'physical_section_id': identity['physical_section_id'],
                'valid_fraction': float(old['valid_mask'][0].float().mean()),
                'visible_fraction': float((old['visible'][0] > .25).float().mean()),
                'event_flags': {name: bool(identity['one_shot_slide_artifacts_v3']['events'][name])
                                for name in ('fragment', 'fold', 'bubble', 'tile_seam')},
                'pose_center_delta_mm': float((new_center - old_center).norm(dim=-1)[0] / 1000),
                'pose_normal_delta_deg': float(angle[0]),
                'provenance': new['provenance'][0]})
            del old, new
assert len(panel) == len(panel_ids) == 64
torch.save(panel, RUN / 'inner_panel.pt')
print(json.dumps({'event': 'panel_ready', 'sections': 64, 'attempts': panel_attempts}), flush=True)

config = {'candidate': 286, 'seed': SEED, 'protocol_sha256': digest(PROTOCOL),
    'mapper_initialization_seed': SEED282 + 2,
    'source_sha256': source_hashes, 'parent_candidate': 275,
    'parent_completed_sha256': digest(PARENT / 'completed.json'),
    'parent_config_sha256': parent_receipt['config_sha256'],
    'parent_checkpoint_sha256': digest(parent_path),
    'synthetic_provenance': context['provenance'],
    'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
    'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'panel_attempts': panel_attempts, 'arms': ARMS,
    'updates_per_arm': UPDATES, 'readout_steps': STEPS,
    'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'sections_per_inner_base': 8, 'readout_sites': 512,
    'map_grid_shape': [64, 64], 'map_control_side': 16,
    'map_derivative_limit': LIMIT,
    'optimizer': 'matched AdamW lr=1e-4 weight_decay=1e-4 clip=2 batch=1',
    'objective': 'exact-plane mean Euclidean surviving-site CCF error in mm',
    'only_arm_difference': 'pre-artifact v9 versus post-artifact intact-tissue affine state',
    'feature_lineage': 'frozen 275 step-8000 scratch pose encoder/lateral; true state/reflection override',
    'result_role': 'TRAIN-inner conditional-map pose-gauge diagnostic, not blind pose or release model',
    'dev_cal_final_or_public_benchmark_used': False,
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

schedule, readouts, initial_hashes, training_attempts = [], {}, {}, 0
with (RUN / 'training_draws.jsonl').open('w', encoding='utf8') as draws, \
     (RUN / 'training_schedule.jsonl').open('w', encoding='utf8') as accepted_draws, \
     (RUN / 'training.jsonl').open('w', encoding='utf8') as training, \
     (RUN / 'readouts.jsonl').open('w', encoding='utf8') as output:
    for arm in ARMS:
        model = make_model(pose_weights, pose_buffers, 16)
        initial_hashes[arm] = mapper_digest(model)
        assert initial_hashes[arm] == initial_hashes[ARMS[0]]
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
        train_rng = np.random.default_rng(SEED + 3)
        attempts = 0
        readouts[arm] = {}
        for step in range(UPDATES + 1):
            if step in STEPS:
                readouts[arm][step] = evaluate(model, context['atlas'], panel, arm, step, output)
                output.flush()
                checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'candidate': 286, 'arm': arm, 'step': step,
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
                    'map_gain_mean_mm': report['map_gain_mean_mm'],
                    'mapped_mean_mm': report['mapped_mean_mm'],
                    'nonnegative_gain_bases': report['nonnegative_gain_bases']}), flush=True)
            if step == UPDATES:
                break
            update = step + 1
            if arm == ARMS[0]:
                while True:
                    attempts += 1
                    base = int(train_rng.integers(56))
                    subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
                    draw_seed = SEED + 2_000_000_000 + attempts
                    with torch.no_grad():
                        old, new = paired_sample(context, subject_index, draw_seed)
                    identity = old['provenance'][0]
                    valid_pixels = int(old['valid_mask'][0].sum())
                    used = bool(old['eligible'][0]) and valid_pixels >= 512
                    draws.write(json.dumps({'step': update, 'attempt': attempts,
                        'base_index': base, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'used': used, 'valid_pixels': valid_pixels,
                        'physical_section_id': identity['physical_section_id']}, allow_nan=False) + '\n')
                    if used:
                        break
                    del old, new
            else:
                entry = schedule[step]
                base, subject_index, draw_seed = (entry[key] for key in
                    ('base_index', 'subject_index', 'draw_seed'))
                with torch.no_grad():
                    old, new = paired_sample(context, subject_index, draw_seed)
                identity = old['provenance'][0]
                assert bool(old['eligible'][0]) and int(old['valid_mask'][0].sum()) >= 512
                assert identity['physical_section_id'] == entry['physical_section_id']
            assert identity['split'] == 'train'
            lineage = context['bases'][base]['lineage']
            assert all(identity['base_lineage'][key] == lineage[key] for key in
                       ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
            indices, _, target = sites_from_sample(old, 512)
            assert len(indices) == 512
            input_sha = tensor_digest(old['inputs'])
            target_sha = tensor_digest(torch.cat((indices[:, None].float(), target), -1))
            old_state_sha = tensor_digest(old['state'])
            new_state_sha = tensor_digest(new['state'])
            if arm == ARMS[0]:
                entry = {'step': update, 'base_index': base,
                    'subject_index': subject_index, 'draw_seed': draw_seed,
                    'physical_section_id': identity['physical_section_id'],
                    'input_sha256': input_sha, 'target_sha256': target_sha,
                    'preartifact_state_sha256': old_state_sha,
                    'postartifact_state_sha256': new_state_sha}
                schedule.append(entry)
                accepted_draws.write(json.dumps(entry, allow_nan=False) + '\n')
            else:
                assert (input_sha, target_sha, old_state_sha, new_state_sha) == (
                    entry['input_sha256'], entry['target_sha256'],
                    entry['preartifact_state_sha256'], entry['postartifact_state_sha256'])
            state = old['state'] if arm == ARMS[0] else new['state']
            optimizer.zero_grad(set_to_none=True)
            mapped, points = exact_map(model, old['inputs'], state, old['reflection'],
                old['offsets'], old['weights'], context['atlas'], indices)
            loss = (points - target).norm(dim=-1).mean() / 1000
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            training.write(json.dumps({'arm': arm, 'step': update,
                'physical_section_id': identity['physical_section_id'],
                'input_sha256': input_sha, 'target_sha256': target_sha,
                'state_sha256': old_state_sha if arm == ARMS[0] else new_state_sha,
                'site_count': len(indices), 'loss_mm_before_update': float(loss.detach()),
                'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if update % 750 == 0:
                draws.flush()
                accepted_draws.flush()
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    'step': update, 'loss_mm': float(loss.detach())}), flush=True)
            del old, new, mapped, points, loss
        if arm == ARMS[0]:
            training_attempts = attempts
        del model, optimizer
        torch.cuda.empty_cache()

assert len(schedule) == len({row['physical_section_id'] for row in schedule}) == UPDATES
a, b = (readouts[arm][UPDATES] for arm in ARMS)
gate = {'postartifact_map_gain_at_least_0_02mm': b['map_gain_mean_mm'] >= .02,
    'postartifact_map_gain_advantage_at_least_0_01mm':
        b['map_gain_mean_mm'] - a['map_gain_mean_mm'] >= .01,
    'postartifact_at_least_six_nonnegative_bases': b['nonnegative_gain_bases'] >= 6,
    'all_arms_derivative_at_most_0_35': all(
        readouts[arm][step]['max_postlimit_derivative'] <= LIMIT + 1e-5
        for arm in ARMS for step in STEPS)}
names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
    'training_draws.jsonl', 'training_schedule.jsonl', 'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 286, 'seed': SEED, 'arms': ARMS, 'steps': STEPS,
    'panel_sections': len(panel), 'panel_attempts': panel_attempts,
    'training_attempts': training_attempts,
    'training_sections_per_arm': UPDATES,
    'unique_training_physical_section_ids': len({row['physical_section_id'] for row in schedule}),
    'initial_mapper_sha256': initial_hashes,
    'readouts': readouts, 'gate': gate, 'gate_pass': all(gate.values()),
    'elapsed_seconds': time.perf_counter() - started,
    'source_sha256': source_hashes,
    'files_sha256': {name: digest(RUN / name) for name in names},
    'checkpoint_sha256': {arm: {str(step): digest(RUN / f'{arm}_step_{step:05d}.pt')
        for step in STEPS} for arm in ARMS},
    'result_role': config['result_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'gate_pass': completed['gate_pass'],
    'preartifact_gain_mm': a['map_gain_mean_mm'],
    'postartifact_gain_mm': b['map_gain_mean_mm'],
    'preartifact_final_mm': a['mapped_mean_mm'],
    'postartifact_final_mm': b['mapped_mean_mm']}), flush=True)
