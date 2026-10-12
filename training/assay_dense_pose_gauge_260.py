"""Measure the full-canvas pose versus a visible-tissue fit on frozen v7 DEV."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/dense_physical_pose_260_gauge_preflight_002'
panel_done = json.loads((PANEL / 'completed.json').read_text())
assert hashlib.sha256((PANEL / 'records.jsonl').read_bytes()).hexdigest() == panel_done['records_sha256']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
eligible = [record for record in records if record['eligible'] and record['valid_pixels'] > 0]
assert len(eligible) == 227 and len({record['animal_id'] for record in eligible}) == 8
axis = (np.arange(96) + .5) / 96 - .5 / 192
yy, xx = np.meshgrid(axis, axis, indexing='ij')
design = np.stack((np.ones_like(xx), xx, yy), -1).reshape(-1, 3)

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        assert hashlib.sha256((PANEL / record['file']).read_bytes()).hexdigest() == record['sha256']
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            target = arrays['target_centre_um'].copy()
            valid = arrays['valid_mask'].copy().astype(bool)
            truth = torch.from_numpy(arrays['target_state'][None].copy())
        pooled = target.reshape(96, 2, 96, 2, 3).mean((1, 3)).reshape(-1, 3)
        support = valid.reshape(96, 2, 96, 2).all((1, 3)).reshape(-1)
        fitted = np.linalg.lstsq(design[support], pooled[support], rcond=None)[0]
        true_centre, true_frame, true_basis = full_frame_state_to_components(truth)
        original_edges = (true_frame[0, :, :2] @ true_basis[0]).numpy()
        original_normal = true_frame[0, :, 2].numpy()
        reflected = record['provenance']['horizontal_reflection']
        original_u = -original_edges[:, 0] if reflected else original_edges[:, 0]
        original_v = original_edges[:, 1]
        original_centre = true_centre[0].numpy() - original_edges[:, 0] / 192 if reflected else true_centre[0].numpy()
        predicted_centre = fitted[0] + .5 * (fitted[1] + fitted[2])
        predicted_edges = fitted[1:].T
        predicted_normal = np.cross(fitted[1], fitted[2])
        predicted_normal /= np.linalg.norm(predicted_normal)
        axis_deg = float(np.rad2deg(np.arccos(np.clip(abs(predicted_normal @ original_normal), 0, 1))))
        original_field = (original_centre + (design[:, 1, None] - .5) * original_u
                          + (design[:, 2, None] - .5) * original_v)
        fitted_field = design @ fitted
        row = {'physical_section_id': record['provenance']['physical_section_id'],
               'animal_id': record['animal_id'], 'panel_file_sha256': record['sha256'],
               'valid_pooled_pixels': int(support.sum()),
               'centre_discrepancy_mm': float(np.linalg.norm(predicted_centre - original_centre) / 1000),
               'axis_discrepancy_deg': axis_deg,
               'edge_u_discrepancy_mm': float(np.linalg.norm(fitted[1] - original_u) / 1000),
               'edge_v_discrepancy_mm': float(np.linalg.norm(fitted[2] - original_v) / 1000),
               'fitted_minor_fov_mm': float(np.linalg.svd(predicted_edges, compute_uv=False).min() / 1000),
               'true_minor_fov_mm': float(np.linalg.svd(original_edges, compute_uv=False).min() / 1000),
               'original_valid_plane_mm': float(np.linalg.norm(original_field[support] - pooled[support], axis=-1).mean() / 1000),
               'best_valid_plane_mm': float(np.linalg.norm(fitted_field[support] - pooled[support], axis=-1).mean() / 1000)}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

keys = ('centre_discrepancy_mm', 'axis_discrepancy_deg', 'edge_u_discrepancy_mm',
        'edge_v_discrepancy_mm', 'fitted_minor_fov_mm', 'true_minor_fov_mm',
        'original_valid_plane_mm', 'best_valid_plane_mm')
animals = sorted({row['animal_id'] for row in rows})
subject_means = {animal: {key: float(np.mean([row[key] for row in rows if row['animal_id'] == animal]))
                          for key in keys} for animal in animals}
summary = {'cases': len(rows), 'synthetic_plans': len(animals),
           'subject_equal_mean': {key: float(np.mean([subject_means[animal][key] for animal in animals]))
                                  for key in keys},
           'case_median': {key: float(np.median([row[key] for row in rows])) for key in keys},
           'centre_over_1mm_cases': sum(row['centre_discrepancy_mm'] > 1 for row in rows),
           'axis_over_20deg_cases': sum(row['axis_discrepancy_deg'] > 20 for row in rows),
           'subject_means': subject_means}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'panel_completed_sha256': hashlib.sha256((PANEL / 'completed.json').read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'public_benchmark_used': False, 'calibrated': False}, indent=2), encoding='utf8')
print(json.dumps({key: value for key, value in summary.items() if key != 'subject_means'}), flush=True)
