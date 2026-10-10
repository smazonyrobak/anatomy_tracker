"""Compare independent v3/v4 raw sections with acquired sagittal-ish images."""

import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64

version = os.environ.get('APPEARANCE_AUDIT_VERSION', 'v4')
sampler = sample_one_shot_slide_artifacts_v3 if version == 'v3' else sample_one_shot_slide_artifacts_v4
data = root / 'data/allen_sagittal_ish_weak_inputs_001_20261008'
real = np.load(data / 'model_input.npy', mmap_mode='r')[:, 0]
records = [json.loads(line) for line in (data / 'geometry.jsonl').open()]
real_measure = np.array([[
    np.mean(image), np.median(image), np.quantile(image, .95),
    np.mean(image <= .004), np.std(image),
    np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))
] for image in real])

torch.set_num_threads(4)
context = load_streaming_synthetic_v7_64(device='cuda')
rng = torch.Generator().manual_seed(20261010129 + (1000000 if version == 'v4' else 0))
synthetic_measure = []
attempts = 0
with torch.no_grad():
    while len(synthetic_measure) < 40:
        virtual = int(torch.randint(len(context['subjects']), (1,), generator=rng))
        sample = sampler(context, [virtual],
            20261010129000000 + (1000000 if version == 'v4' else 0) + attempts, side=256)
        attempts += 1
        normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        if not bool(sample['eligible'][0]) or sample['provenance'][0]['mode'] != 'raw' \
                or float(normal[2].abs()) < np.cos(np.deg2rad(20)):
            continue
        image = sample['inputs'][0, 0].cpu().numpy()
        synthetic_measure.append([
            np.mean(image), np.median(image), np.quantile(image, .95),
            np.mean(image <= .004), np.std(image),
            np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))
        ])

names = ('mean', 'median', 'p95', 'fraction_le_004', 'std', 'mean_abs_xy_gradient')
result = {'version': version, 'synthetic_attempts': attempts,
    'synthetic_near_ml_raw_sections': len(synthetic_measure),
    'real_split_donors': {}, 'statistics': {}}
for split in ('train', 'weak_dev'):
    donors = sorted({row['donor_id'] for row in records if row['split'] == split})
    result['real_split_donors'][split] = len(donors)
    donor_values = np.array([np.median(real_measure[[i for i, row in enumerate(records)
        if row['split'] == split and row['donor_id'] == donor]], axis=0) for donor in donors])
    result['statistics'][split] = {name: np.quantile(donor_values[:, j], [.1, .5, .9]).tolist()
        for j, name in enumerate(names)}
synthetic_measure = np.array(synthetic_measure)
result['statistics']['synthetic'] = {name: np.quantile(synthetic_measure[:, j], [.1, .5, .9]).tolist()
    for j, name in enumerate(names)}
print(json.dumps(result, indent=2))
(root / 'runs/appearance_gap_129').mkdir(parents=True, exist_ok=True)
(root / f'runs/appearance_gap_129/{version}.json').write_text(json.dumps(result, indent=2))
