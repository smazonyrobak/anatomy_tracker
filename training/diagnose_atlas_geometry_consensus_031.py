"""Truth-blind selection of an affine plane from frozen 030 patch matches."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

bank = root / 'runs/atlas_global_patch_030'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_geometry_consensus_031'
started = time.perf_counter()

with (bank / 'completed.json').open() as stream:
    bank_receipt = json.load(stream)
with (retrieval / 'completed.json').open() as stream:
    retrieval_receipt = json.load(stream)
assert len(bank_receipt['files_sha256']) == 6 and retrieval_receipt['rows'] == 185
for folder, receipt in ((bank, bank_receipt), (retrieval, retrieval_receipt)):
    for name, expected in receipt['files_sha256'].items():
        with (folder / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
positions = np.load(bank / 'positions_um.npy', allow_pickle=False)
bank_u = np.load(bank / 'orientation_u.npy', allow_pickle=False)
bank_v = np.load(bank / 'orientation_v.npy', allow_pickle=False)
bank_n = np.load(bank / 'orientation_n.npy', allow_pickle=False)
records = {record['sha256']: record for record in map(json.loads, (panel / 'records.jsonl').open())
           if record['eligible']}
retrieved_rows = list(map(json.loads, (retrieval / 'rows.jsonl').open()))
assert len(retrieved_rows) == len(records) == 185

out.mkdir(parents=True, exist_ok=False)
config = {'bank_completed_sha256': hashlib.sha256((bank / 'completed.json').read_bytes()).hexdigest(),
          'retrieval_completed_sha256': hashlib.sha256((retrieval / 'completed.json').read_bytes()).hexdigest(),
          'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
          'seed_position_tolerance_um': 1500, 'seed_frame_tolerance_deg': 35,
          'quality': '1 - 0.5*position_fraction - 0.5*rotation_trace_fraction within both tolerances',
          'affine_ridge_diagonal': [.01, 100, 100], 'fallback': 'seed plane when fewer than three supports',
          'query_selection': 'frozen 030 GT-valid; truth used only after fit for development evaluation',
          'gate': {'mean_rigid_error_um_below': 1500, 'fraction_below_750um_at_least': .30},
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
threshold_trace = 1 + 2 * math.cos(math.radians(35))
with (out / 'rows.jsonl').open('w') as stream:
    for number, retrieved in enumerate(retrieved_rows, 1):
        query = np.asarray(retrieved['query_pixel_yx'], dtype='float64')
        indices = np.asarray(retrieved['top16_bank_index'], dtype='int64')
        parity = np.asarray(retrieved['top16_parity'], dtype='int64')
        location = positions[indices // 512].astype('float64')
        orientation = indices % 512
        sign = 1 - 2 * parity
        axis_u = bank_u[orientation] * sign[..., None]
        axis_v = bank_v[orientation]
        normal = bank_n[orientation] * sign[..., None]
        seed_location = location.reshape(-1, 3)
        seed_u = axis_u.reshape(-1, 3)
        seed_v = axis_v.reshape(-1, 3)
        seed_n = normal.reshape(-1, 3)
        seed_pixel = np.repeat(query, 16, axis=0)
        x_difference = query[None, :, 1] - seed_pixel[:, None, 1]
        y_difference = query[None, :, 0] - seed_pixel[:, None, 0]
        projected = (seed_location[:, None] + x_difference[..., None] * (67 * seed_u[:, None])
                     + y_difference[..., None] * (67 * seed_v[:, None]))
        distance = np.linalg.norm(projected[:, :, None] - location[None], axis=-1)
        trace = (seed_u @ axis_u.reshape(-1, 3).T + seed_v @ axis_v.reshape(-1, 3).T
                 + seed_n @ normal.reshape(-1, 3).T).reshape(512, 32, 16)
        supported = (distance < 1500) & (trace > threshold_trace)
        quality = np.where(supported, 1 - .5 * distance / 1500
                           - .5 * (3 - trace) / (3 - threshold_trace), 0)
        best_per_query = quality.max(axis=-1)
        seed = int(best_per_query.sum(axis=-1).argmax())
        choice = quality[seed].argmax(axis=-1)
        weight = best_per_query[seed]
        keep = weight > 0
        target = location[np.arange(32), choice]
        anchor = seed_pixel[seed]
        prior = np.stack((seed_location[seed], 67 * seed_u[seed], 67 * seed_v[seed]))
        design = np.stack((np.ones(32), query[:, 1] - anchor[1], query[:, 0] - anchor[0]), -1)
        if keep.sum() >= 3:
            ridge = np.diag([.01, 100, 100])
            coefficient = np.linalg.solve(design[keep].T @ (weight[keep, None] * design[keep]) + ridge,
                                          design[keep].T @ (weight[keep, None] * target[keep]) + ridge @ prior)
        else:
            coefficient = prior
        affine = np.stack((coefficient[0] - anchor[1] * coefficient[1] - anchor[0] * coefficient[2],
                           coefficient[1], coefficient[2]))

        record = records[retrieved['sha256']]
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            y, x = np.nonzero(arrays['valid_mask'])
            mapped = affine[0] + x[:, None] * affine[1] + y[:, None] * affine[2]
            error_um = float(np.linalg.norm(mapped - arrays['target_centre_um'][y, x], axis=-1).mean())
        row = {key: retrieved[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(selected_seed_query=int(seed // 16), selected_seed_rank=int(seed % 16),
                   selected_seed_bank_index=int(indices.reshape(-1)[seed]),
                   affine_origin_x_y_um=affine.tolist(), support_count=int(keep.sum()),
                   support_query_indices=np.flatnonzero(keep).tolist(),
                   support_bank_indices=indices[np.arange(32), choice][keep].tolist(),
                   consensus_score=float(best_per_query[seed].sum()), rigid_mean_error_um=error_um)
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 50 == 0:
            stream.flush()
            print(json.dumps({'dev_sections': number, 'total': len(retrieved_rows),
                              'seconds': time.perf_counter() - started}), flush=True)

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
mean_error = float(np.mean([np.mean([row['rigid_mean_error_um'] for row in rows
    if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
fraction_750 = float(np.mean([np.mean([row['rigid_mean_error_um'] < 750 for row in rows
    if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
fraction_1000 = float(np.mean([np.mean([row['rigid_mean_error_um'] < 1000 for row in rows
    if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
mode_error = {mode: float(np.mean([np.mean([row['rigid_mean_error_um'] for row in rows
    if row['synthetic_subject_plan_id'] == subject and row['appearance_mode'] == mode])
    for subject in subjects if any(row['synthetic_subject_plan_id'] == subject and
                                  row['appearance_mode'] == mode for row in rows)]))
    for mode in ('raw', 'exact_black', 'imperfect_brush')}
summary = {'sections': len(rows), 'synthetic_subjects': len(subjects),
           'identity_equal_mean_rigid_error_um': mean_error, 'fraction_below_750um': fraction_750,
           'fraction_below_1000um': fraction_1000,
           'support_count_median': float(np.median([row['support_count'] for row in rows])),
           'mode_identity_equal_mean_rigid_error_um': mode_error,
           'necessary_gate': mean_error < 1500 and fraction_750 >= .30,
           'scope': 'oracle tissue-pixel selection; truth-blind match/plane fit; synthetic DEV only',
           'calibrated': False, 'public_benchmark_used': False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'files_sha256': {
    name: hashlib.sha256((out / name).read_bytes()).hexdigest()
    for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'rows': len(rows), 'seconds': time.perf_counter() - started}, indent=2))
print(json.dumps({'event': 'complete', **summary, 'seconds': time.perf_counter() - started}), flush=True)
