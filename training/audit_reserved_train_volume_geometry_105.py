"""Bounded TRAIN-donor audit before using real stacks for virtual oblique cuts."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001')
out = Path('I:/AnatomyTracker/runs/reserved_train_volume_geometry_105_audit')
donors = sorted(root.glob('donor_*'))
chosen = np.random.default_rng(20261008105).choice(len(donors), 64, replace=False)
rows = []
for index in sorted(chosen):
    donor = donors[int(index)]
    record_path = donor / 'records.jsonl'
    records = [json.loads(line) for line in record_path.open()]
    affines = np.asarray([row['model_pixel_to_ap_dv_ml_um'] for row in records])
    translation_steps = np.diff(affines[:, :, 2], axis=0)
    median_step = np.median(translation_steps, axis=0)
    sections = np.asarray([row['section_number'] for row in records])
    image_shape = list(np.load(donor / 'images.npy', mmap_mode='r').shape)
    rows.append({'donor': donor.name, 'sections': len(records), 'image_shape': image_shape,
        'train_only': all(row['training_split'] == 'train' for row in records),
        'section_number_min_max': [int(sections.min()), int(sections.max())],
        'section_number_gaps': int(np.sum(np.diff(sections) != 1)),
        'inplane_basis_max_variation_um': float(np.max(np.abs(
            affines[:, :, :2] - affines[0, :, :2]))),
        'section_step_max_residual_um': float(np.max(np.linalg.norm(
            translation_steps - median_step, axis=1))),
        'median_section_step_um': median_step.tolist(),
        'physical_step_um': float(np.linalg.norm(median_step)),
        'records_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest()})
assert all(row['train_only'] for row in rows)
regular = [row for row in rows if row['section_number_gaps'] == 0
    and row['inplane_basis_max_variation_um'] < 1e-5
    and row['section_step_max_residual_um'] < 1e-4
    and row['image_shape'] == [row['sections'], 1, 192, 192]]
summary = {'sampled_donors': len(rows), 'reserved_train_donors': len(donors),
    'affinely_regular_complete_stacks': len(regular),
    'sections_min_median_max': [int(np.min([row['sections'] for row in rows])),
        float(np.median([row['sections'] for row in rows])),
        int(np.max([row['sections'] for row in rows]))],
    'physical_step_um_min_median_max': [float(np.min([row['physical_step_um'] for row in rows])),
        float(np.median([row['physical_step_um'] for row in rows])),
        float(np.max([row['physical_step_um'] for row in rows]))],
    'selection_seed': 20261008105,
    'scope': '64 preselected reserved TRAIN donors only; weak affine regularity, not expert anatomical correctness'}
out.mkdir(parents=True, exist_ok=False)
(out / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
