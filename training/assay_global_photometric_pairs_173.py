"""Frozen 172 TRAIN-pair image/atlas correlation diagnostic."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def correlation(x, y, mask):
    count = mask.sum()
    if count < 16:
        return None
    a = (x * mask).sum() / count
    b = (y * mask).sum() / count
    u, v = (x - a) * mask, (y - b) * mask
    return float((u * v).sum() / ((u.square().sum() * v.square().sum()).sqrt().clamp_min(1e-8)))


source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/GLOBAL_PHOTOMETRIC_PAIR_173_PROTOCOL_20261011.md'
parent = root / 'runs/coherent_fit_energy_172_train_001'
out = root / 'runs/global_photometric_pair_173_train_001'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
done = json.loads((parent / 'completed.json').read_text())
assert sha(parent / 'panel.pt') == done['output_sha256']['panel.pt']
assert sha(parent / 'rows.jsonl') == done['output_sha256']['rows.jsonl']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
# The hash-verified local panel contains NumPy provenance scalars, not just tensors.
panel = torch.load(parent / 'panel.pt', map_location='cpu', weights_only=False)
records = [json.loads(line) for line in (parent / 'rows.jsonl').open()]
assert len(panel) == len(records) == 240
axis = (torch.arange(48, device='cuda') + .5) / 48 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)

out.mkdir(parents=True, exist_ok=False)
config = {'scope': 'in-panel TRAIN-only mechanism check; no new model or validation',
    'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'parent_completion_sha256': sha(parent / 'completed.json'),
    'parent_panel_sha256': sha(parent / 'panel.pt'),
    'parent_rows_sha256': sha(parent / 'rows.jsonl'),
    'synthetic_provenance': context['provenance'],
    'oracle_visibility_only_for_diagnostic': True,
    'no_external_pretrained_weights': True, 'posterior_calibrated': False,
    'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

results = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for item, record in zip(panel, records):
        assert item['index'] == record['index']
        pair = record['matched_pair']
        if record['panel_stratum'] != 'unconditional' or pair is None:
            continue
        slots = [pair['good_slot'], pair['wrong_slot']]
        state = item['initial_state'][slots].cuda()
        reflected = item['reflection'][slots].cuda()
        plane = rigid_points_090(state, reflected, chart, (256, 256)).reshape(2, 48, 48, 3)
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        offsets = item['offsets'][None].cuda().expand(2, -1)
        weights = item['weights'][None].cuda().expand(2, -1)
        coords = plane[:, None] + offsets[:, :, None, None, None] * normal[:, None, None, None, :]
        rendered = render_finite_thickness_coordinate_grid(atlas, coords,
            (0., 0., 0.), (25., 25., 25.), weights)
        support = rendered[:, 1:2].clamp(0, 1)
        image_atlas = rendered[:, :1] / support.clamp_min(1e-4)
        shared = ((support[0, 0] > .5) & (support[1, 0] > .5)).float()
        common = (F.avg_pool2d(shared[None, None], 7, 1, 3,
            count_include_pad=True)[0, 0] >= 1 - 1e-6).float()
        assert int(common.sum()) == pair['common_sites']
        observed = F.interpolate(item['inputs'][None, :1].cuda(), (48, 48), mode='area')
        oracle = common * (item['target_valid_area_48'].cuda() >= .8)
        scores = {}
        for width in (1, 3, 7):
            x = F.avg_pool2d(observed, width, 1, width // 2)[0, 0]
            y = F.avg_pool2d(image_atlas, width, 1, width // 2)[:, 0]
            scores[f'box_{width}'] = [correlation(x, y[slot], common) for slot in (0, 1)]
            if width == 3:
                scores['box_3_oracle_visible'] = [correlation(x, y[slot], oracle) for slot in (0, 1)]
        row = {'index': item['index'], 'physical_section_id': record['physical_section_id'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'appearance_mode': record['appearance_mode'],
            'nearest_cardinal_angle_deg': record['nearest_cardinal_angle_deg'],
            'good_slot': slots[0], 'wrong_slot': slots[1],
            'good_rigid_mm': record['initial_rigid_mm'][slots[0]],
            'wrong_rigid_mm': record['initial_rigid_mm'][slots[1]],
            'shared_sites': int(common.sum()), 'oracle_visible_sites': int(oracle.sum()),
            'ncc': scores}
        results.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(results) % 48 == 0:
            stream.flush()
            print(json.dumps({'event': 'pairs', 'scored': len(results)}), flush=True)

summary = {'pairs': len(results), 'methods': {},
    'oracle_visibility_only_for_diagnostic': True,
    'posterior_calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}
for name in ('box_1', 'box_3', 'box_7', 'box_3_oracle_visible'):
    rows = [r for r in results if all(x is not None for x in r['ncc'][name])]
    summary['methods'][name] = {'eligible': len(rows),
        'correct': sum(r['ncc'][name][0] > r['ncc'][name][1] for r in rows),
        'mean_good_minus_wrong': (sum(r['ncc'][name][0] - r['ncc'][name][1]
            for r in rows) / len(rows)) if rows else None}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
receipt = {'output_sha256': {p.name: sha(p) for p in out.iterdir()
    if p.is_file() and p.name != 'completed.json'},
    'pairs': len(results), 'posterior_calibrated': False,
    'final_animals_used': False, 'public_benchmark_used': False}
(out / 'completed.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
print(json.dumps({'event': 'completed', 'summary': summary}, allow_nan=False), flush=True)
