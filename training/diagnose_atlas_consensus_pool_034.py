"""Read-only capture by ranked geometry seeds before expensive atlas fitting."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

bank = root / 'runs/atlas_global_patch_030'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
out = root / 'runs/atlas_consensus_pool_034'
positions = np.load(bank / 'positions_um.npy', allow_pickle=False)
bank_u = np.load(bank / 'orientation_u.npy', allow_pickle=False)
bank_v = np.load(bank / 'orientation_v.npy', allow_pickle=False)
bank_n = np.load(bank / 'orientation_n.npy', allow_pickle=False)
source = list(map(json.loads, (retrieval / 'rows.jsonl').open()))
assert len(source) == 185
threshold_trace = 1 + 2 * math.cos(math.radians(35))
sizes = (1, 8, 16, 32, 64, 128, 512)
out.mkdir(parents=True, exist_ok=False)
config = {'retrieval_completed_sha256': hashlib.sha256((retrieval / 'completed.json').read_bytes()).hexdigest(),
          'bank_completed_sha256': hashlib.sha256((bank / 'completed.json').read_bytes()).hexdigest(),
          'candidate_ranking': 'unchanged 031 geometry-only quality, truth-blind',
          'pool_sizes': sizes, 'evaluation': 'minimum 32-query rigid coordinate error after ranking only',
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with (out / 'rows.jsonl').open('w') as stream:
    for retrieved in source:
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
        score = quality.max(axis=-1).sum(-1)
        ranked = np.argsort(-score, kind='stable')
        truth = np.asarray(retrieved['true_ccf_um'])
        error = np.linalg.norm(projected - truth[None], axis=-1).mean(-1)
        row = {key: retrieved[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                   'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')}
        row.update(best_query_mean_error_um_by_pool={str(size): float(error[ranked[:size]].min())
                                                      for size in sizes},
                   ranked_top32_seed_indices=ranked[:32].tolist(),
                   ranked_top32_scores=score[ranked[:32]].tolist())
        rows.append(row)
        stream.write(json.dumps(row) + '\n')

subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'sections': len(rows), 'synthetic_subjects': len(subjects), 'pools': []}
for size in sizes:
    key = str(size)
    summary['pools'].append({'size': size,
        'identity_equal_best_query_mean_error_um': float(np.mean([np.mean([
            row['best_query_mean_error_um_by_pool'][key] for row in rows
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])),
        'identity_equal_fraction_under_750um': float(np.mean([np.mean([
            row['best_query_mean_error_um_by_pool'][key] < 750 for row in rows
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])),
        'identity_equal_fraction_under_1000um': float(np.mean([np.mean([
            row['best_query_mean_error_um_by_pool'][key] < 1000 for row in rows
            if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))})
summary.update(scope='oracle choice within ranked pool; no plane selected from image',
               calibrated=False, public_benchmark_used=False)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'files_sha256': {
    name: hashlib.sha256((out / name).read_bytes()).hexdigest()
    for name in ('config.json', 'rows.jsonl', 'summary.json')}, 'rows': len(rows)}, indent=2))
print(json.dumps({'event': 'complete', **summary}))
