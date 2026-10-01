"""Fresh-draw synthetic fitter check; these draws are not independent animals."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_stream import sample_one_shot_stream
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/one_shot_joint_mixed_real_003'
OUT = ROOT / 'runs/one_shot_fresh_fitter_001'
SIDE, WANTED, SEED = 256, 64, 20261002090000000
STEPS = (0, 8000, 20000)
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 20000
torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
models, checkpoint_sha256 = {}, {}
for step in STEPS:
    path = RUN / f'joint_step_{step:05d}.pt'
    with path.open('rb') as stream:
        checkpoint_sha256[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step
    model = OneShotJointSliceModel().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    models[step] = model
    del checkpoint

axis = torch.arange(SIDE, device='cuda') / SIDE
y, x = torch.meshgrid(axis, axis, indexing='ij')
subjects_rng = torch.Generator().manual_seed(SEED)
rows, attempt = [], 0
with torch.inference_mode():
    while len(rows) < WANTED:
        subjects = torch.randint(len(context['subjects']), (4,), generator=subjects_rng).tolist()
        batch = sample_one_shot_stream(context, subjects, SEED + attempt, side=SIDE)
        accepted = torch.nonzero(batch['eligible']).flatten()[:WANTED - len(rows)]
        attempt += 1
        if len(accepted) == 0:
            continue
        inputs = batch['inputs'][accepted]
        truth = batch['state'][accepted]
        reflection = batch['reflection'][accepted]
        target = batch['centre'][accepted]
        valid = batch['valid_mask'][accepted]
        offsets = batch['offsets'][accepted]
        centre, frame, basis = full_frame_state_to_components(truth)
        reflected_x = torch.where(reflection[:, None, None].bool(), (SIDE - 1) / SIDE - x, x)
        chart = torch.stack((reflected_x.expand(-1, SIDE, -1), y.expand(len(accepted), -1, -1)), -1)
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
        zero_error = (plane - target).norm(dim=-1)
        errors = {}
        for step, model in models.items():
            prediction = model.predict(inputs)
            state = prediction['state'].clone()
            state[:, 0] = truth
            mapped = model.map({**prediction, 'state': state}, offsets,
                               torch.zeros(len(accepted), 1, device='cuda', dtype=torch.long),
                               reflection[:, None], (SIDE, SIDE))
            errors[step] = (mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0] - target).norm(dim=-1)
        for position, index in enumerate(accepted.tolist()):
            mask = valid[position]
            record = batch['provenance'][index]
            row = {'draw': len(rows), 'attempt': attempt - 1, 'slot': index,
                   'subject_index': subjects[index], 'physical_section_id': record['physical_section_id'],
                   'virtual_subject_id': record['virtual_subject_id'], 'base_lineage': record['base_lineage'],
                   'appearance_mode': record['mode'], 'background_mean': record['appearance']['background_mean'],
                   'warp_strength': record['one_shot']['warp_strength'],
                   'damage_events': record['one_shot']['events'], 'valid_pixels': int(mask.sum()),
                   'provenance': record,
                   'zero': {'mean_um': float(zero_error[position][mask].mean()),
                            'median_um': float(zero_error[position][mask].median())}}
            for step in STEPS:
                error = errors[step][position][mask]
                row[f'step_{step}'] = {'mean_um': float(error.mean()), 'median_um': float(error.median())}
            rows.append(row)

OUT.mkdir(parents=True, exist_ok=False)
with (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    stream.writelines(json.dumps(row) + '\n' for row in rows)
zero = np.array([row['zero']['mean_um'] for row in rows])
summary = {'label': 'fresh-draw synthetic diagnostic; shared training deformation bases, not independent animals',
    'training_run': str(RUN), 'checkpoint_sha256': checkpoint_sha256, 'draw_seed_start': SEED,
    'draw_attempts': attempt, 'eligible_draws': len(rows), 'side': SIDE,
    'source_map_bases': len(context['bases']),
    'stream_source_sha256': hashlib.sha256(Path(sample_one_shot_stream.__code__.co_filename).read_bytes()).hexdigest(),
    'zero': {'mean_of_draw_means_um': float(zero.mean()), 'median_of_draw_means_um': float(np.median(zero)),
             'median_of_draw_pixel_medians_um': float(np.median([row['zero']['median_um'] for row in rows]))},
    'steps': {}, 'warp_strength_tertiles': {}, 'appearance_modes': {}}
for step in STEPS:
    key = f'step_{step}'
    values = np.array([row[key]['mean_um'] for row in rows])
    summary['steps'][key] = {'mean_of_draw_means_um': float(values.mean()),
        'median_of_draw_means_um': float(np.median(values)),
        'median_of_draw_pixel_medians_um': float(np.median([row[key]['median_um'] for row in rows])),
        'mean_reduction_vs_zero_um': float((zero - values).mean()),
        'median_reduction_vs_zero_um': float(np.median(zero - values)),
        'improved_draw_fraction': float(np.mean(values < zero))}
strength = np.array([row['warp_strength'] for row in rows])
last = np.array([row['step_20000']['mean_um'] for row in rows])
edges = np.quantile(strength, [0, 1 / 3, 2 / 3, 1])
for index in range(3):
    selected = (strength >= edges[index]) & (strength < edges[index + 1] if index < 2 else strength <= edges[index + 1])
    summary['warp_strength_tertiles'][str(index + 1)] = {'draws': int(selected.sum()),
        'strength_range': [float(edges[index]), float(edges[index + 1])],
        'zero_mean_um': float(zero[selected].mean()), 'step_20000_mean_um': float(last[selected].mean()),
        'improved_draw_fraction': float(np.mean(last[selected] < zero[selected]))}
for mode in sorted({row['appearance_mode'] for row in rows}):
    selected = np.array([row['appearance_mode'] == mode for row in rows])
    summary['appearance_modes'][mode] = {'draws': int(selected.sum()),
        'zero_mean_um': float(zero[selected].mean()), 'step_20000_mean_um': float(last[selected].mean()),
        'improved_draw_fraction': float(np.mean(last[selected] < zero[selected]))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
