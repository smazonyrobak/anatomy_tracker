"""Read-only, outcome-blind seen/same-base/new-base map diagnostic for frozen 282."""
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

from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.run_exact_map_capacity_281 import rigid_surface
from training.train_blind_fit_score_280 import mapped_sites, sites_from_sample
from training.train_image_map_282 import (LIMIT, MAP_MODULES, PARENT, SIDE,
    digest, exact_map, make_model, tensor_digest)


SEED = 20261012283
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parent.parent / 'docs/publication/IMAGE_MAP_TRAIN_CHECK_283_PROTOCOL_20261012.md'
PARENT_282 = ROOT / 'runs/image_map_282_002'
RUN = ROOT / 'runs/image_map_train_check_283_002'


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()

prior = json.loads((PARENT_282 / 'completed.json').read_text(encoding='utf8'))
config_282 = json.loads((PARENT_282 / 'config.json').read_text(encoding='utf8'))
assert prior['candidate'] == 282 and prior['gate']['smooth_architecture_mean_gain_at_least_0_05mm'] is False
assert digest(PARENT_282 / 'completed.json') != ''
checkpoint_path = PARENT_282 / 'smooth16_full_step_01500.pt'
assert digest(checkpoint_path) == prior['checkpoint_sha256']['smooth16_full_1500']
parent_path = PARENT / 'pose_step_08000.pt'
assert digest(parent_path) == config_282['parent_checkpoint_sha256']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
pose_names = set(config_282['parent_pose_modules'])
pose_weights = {key: value for key, value in parent['model'].items()
                if key.split('.')[0] in pose_names}
pose_buffers = {name: parent['model'][name] for name in config_282['parent_pose_buffers']}
del parent
model = make_model(pose_weights, pose_buffers, 16)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert (checkpoint['candidate'], checkpoint['arm'], checkpoint['step']) == (282, 'smooth16_full', 1500)
for name in MAP_MODULES:
    prefix = name + '.'
    getattr(model, name).load_state_dict({key[len(prefix):]: value
        for key, value in checkpoint['mapper'].items() if key.startswith(prefix)}, strict=True)
model.eval()
del checkpoint

schedule = [json.loads(line) for line in (PARENT_282 / 'training_schedule.jsonl').read_text(encoding='utf8').splitlines()]
draws = [json.loads(line) for line in (PARENT_282 / 'training_draws.jsonl').read_text(encoding='utf8').splitlines()]
assert len(schedule) == 1500
accepted = {row['provenance']['physical_section_id']: row for row in draws if row['used']}
assert len(accepted) == len(schedule)
valid_schedule = [entry for entry in schedule if
    accepted[entry['physical_section_id']]['provenance']['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] >= 512]
last_by_base = {base: next(entry for entry in reversed(valid_schedule) if entry['base_index'] == base)
                for base in range(56)}
selected_ids = {entry['physical_section_id'] for entry in last_by_base.values()}
remaining = [entry for entry in valid_schedule if entry['physical_section_id'] not in selected_ids]
extra = [remaining[int(index)] for index in np.linspace(0, len(remaining) - 1, 8).round()]
selected = sorted([*last_by_base.values(), *extra], key=lambda entry: entry['step'])
assert len(selected) == len({entry['physical_section_id'] for entry in selected}) == 64

context = load_streaming_synthetic_v7_64(device='cuda')
assert context['provenance'] == config_282['synthetic_provenance']
RUN.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(SEED)
rows = []
attempts = 0
with (RUN / 'fresh_draws.jsonl').open('w', encoding='utf8') as fresh_draws, \
     (RUN / 'rows.jsonl').open('w', encoding='utf8') as output:
    for entry in selected:
        base = entry['base_index']
        with torch.inference_mode():
            seen = sample_one_shot_slide_artifacts_v9(
                context, [entry['subject_index']], entry['draw_seed'], SIDE)
        identity = seen['provenance'][0]
        assert identity['physical_section_id'] == entry['physical_section_id']
        assert json.loads(json.dumps(identity)) == accepted[entry['physical_section_id']]['provenance']
        assert identity['split'] == 'train'
        lineage = context['bases'][base]['lineage']
        assert all(identity['base_lineage'][key] == lineage[key] for key in
                   ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
        assert bool(seen['eligible'][0]) and int(seen['valid_mask'][0].sum()) >= 512
        indices128, _, target128 = sites_from_sample(seen, 128)
        assert tensor_digest(seen['inputs']) == entry['input_sha256']
        assert tensor_digest(torch.cat((indices128[:, None].float(), target128), -1)) == entry['site_target_sha256']
        assert tensor_digest(torch.cat((seen['state'], seen['reflection'][:, None].float(),
            seen['offsets'], seen['weights']), -1)) == entry['state_psf_sha256']
        for tier in ('seen_training_section', 'new_section_same_base'):
            if tier == 'seen_training_section':
                sample = seen
                subject_index, draw_seed = entry['subject_index'], entry['draw_seed']
            else:
                while True:
                    attempts += 1
                    subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                    draw_seed = SEED + 4_000_000_000 + attempts
                    with torch.inference_mode():
                        sample = sample_one_shot_slide_artifacts_v9(
                            context, [subject_index], draw_seed, SIDE)
                    used = bool(sample['eligible'][0]) and int(sample['valid_mask'][0].sum()) >= 512
                    fresh_draws.write(json.dumps({'attempt': attempts, 'base_index': base,
                        'subject_index': subject_index, 'draw_seed': draw_seed,
                        'used': used, 'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                    if used:
                        break
                    del sample
            identity = sample['provenance'][0]
            assert identity['split'] == 'train'
            assert all(identity['base_lineage'][key] == lineage[key] for key in
                       ('subject_id', 'animal_id', 'specimen_id', 'experiment_id'))
            indices, _, target = sites_from_sample(sample, 512)
            assert len(indices) == 512
            with torch.inference_mode():
                rigid, _, _ = rigid_surface(sample['state'], sample['reflection'])
                rigid_points = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
                mapped, points = exact_map(model, sample['inputs'], sample['state'], sample['reflection'],
                    sample['offsets'], sample['weights'], context['atlas'], indices)
                baseline = float((rigid_points - target).norm(dim=-1).mean() / 1000)
                error = float((points - target).norm(dim=-1).mean() / 1000)
                shift = float((points - rigid_points).norm(dim=-1).mean() / 1000)
                derivative = float(mapped['prelimit_derivative_bound'][0, 0] *
                                   mapped['deformation_rescale'][0, 0])
            row = {'tier': tier, 'base_index': base, 'training_step': entry['step'],
                'subject_index': subject_index, 'draw_seed': draw_seed,
                'physical_section_id': identity['physical_section_id'],
                'site_count': len(indices), 'same_grid_no_warp_mm': baseline,
                'mapped_mm': error, 'gain_mm': baseline - error,
                'surviving_site_shift_mm': shift, 'postlimit_derivative_bound': derivative,
                'provenance': identity}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if tier != 'seen_training_section':
                del sample
        del seen

summary = {}
for tier in ('seen_training_section', 'new_section_same_base'):
    group = [row for row in rows if row['tier'] == tier]
    base_gains = {str(base): float(np.mean([row['gain_mm'] for row in group
        if row['base_index'] == base])) for base in range(56)}
    summary[tier] = {'n': len(group), 'bases': len(base_gains),
        'same_grid_no_warp_mean_mm': float(np.mean([row['same_grid_no_warp_mm'] for row in group])),
        'mapped_mean_mm': float(np.mean([row['mapped_mm'] for row in group])),
        'gain_mean_mm': float(np.mean([row['gain_mm'] for row in group])),
        'nonnegative_sections': sum(row['gain_mm'] >= 0 for row in group),
        'nonnegative_bases': sum(gain >= 0 for gain in base_gains.values()),
        'site_shift_mean_mm': float(np.mean([row['surviving_site_shift_mm'] for row in group])),
        'max_postlimit_derivative_bound': max(row['postlimit_derivative_bound'] for row in group),
        'by_base_gain_mm': base_gains}

inner = prior['readouts']['smooth16_full']['1500']
summary['new_base_train_inner_282'] = {'n': inner['n'], 'bases': len(inner['by_base']),
    'same_grid_no_warp_mean_mm': inner['same_grid_no_warp_mean_mm'],
    'mapped_mean_mm': inner['mapped_mean_mm'], 'gain_mean_mm': inner['gain_mean_mm'],
    'nonnegative_bases': inner['nonnegative_gain_bases']}
assert len(rows) == 128 and all(row['postlimit_derivative_bound'] <= LIMIT + 1e-5 for row in rows)
receipt = {'candidate': 283, 'result_role': 'read-only synthetic TRAIN mechanism diagnostic',
    'seed': SEED, 'source_sha256': digest(SOURCE), 'protocol_sha256': digest(PROTOCOL),
    'parent_282_completed_sha256': digest(PARENT_282 / 'completed.json'),
    'parent_282_checkpoint_sha256': digest(checkpoint_path),
    'selected_training_steps': [entry['step'] for entry in selected],
    'selected_training_ids': [entry['physical_section_id'] for entry in selected],
    'fresh_attempts': attempts, 'summary': summary,
    'output_sha256': {name: digest(RUN / name) for name in ('fresh_draws.jsonl', 'rows.jsonl')},
    'elapsed_seconds': time.perf_counter() - started,
    'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'complete', 'fresh_attempts': attempts,
    'gain_mm': {name: value['gain_mean_mm'] for name, value in summary.items()}}, allow_nan=False), flush=True)
