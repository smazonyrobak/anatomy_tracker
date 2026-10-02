"""Independent 031 audit: separate candidate coverage from consensus selection."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
sys.dont_write_bytecode = True

import numpy as np

bank = root / 'runs/atlas_global_patch_030'
retrieval = root / 'runs/atlas_global_patch_030_development_eval'
consensus = root / 'runs/atlas_geometry_consensus_031'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/atlas_geometry_consensus_031_audit'
with (consensus / 'completed.json').open() as stream:
    receipt = json.load(stream)
for name, expected in receipt['files_sha256'].items():
    with (consensus / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
with (consensus / 'config.json').open() as stream:
    config = json.load(stream)
assert hashlib.sha256((retrieval / 'completed.json').read_bytes()).hexdigest() == config['retrieval_completed_sha256']
assert hashlib.sha256((Path(__file__).parent / 'diagnose_atlas_geometry_consensus_031.py').read_bytes()).hexdigest() == config['source_sha256']
positions = np.load(bank / 'positions_um.npy', allow_pickle=False)
bank_u = np.load(bank / 'orientation_u.npy', allow_pickle=False)
bank_v = np.load(bank / 'orientation_v.npy', allow_pickle=False)
records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']}
retrieved = {row['sha256']: row for row in map(json.loads, (retrieval / 'rows.jsonl').open())}
selected = list(map(json.loads, (consensus / 'rows.jsonl').open()))
assert len(records) == len(retrieved) == len(selected) == 185

rows = []
for row in selected:
    source = retrieved[row['sha256']]
    record = records[row['sha256']]
    query = np.asarray(source['query_pixel_yx'])
    true_query = np.asarray(source['true_ccf_um'])
    candidates = np.asarray(source['top16_bank_index'])
    locations = positions[candidates // 512]
    distance = np.linalg.norm(locations - true_query[:, None], axis=-1)
    nearest = distance.argmin(axis=1)
    accepted = distance[np.arange(32), nearest] < 500
    fit = np.stack((np.ones(32), query[:, 1], query[:, 0]), -1)
    oracle_affine = np.linalg.lstsq(fit[accepted], locations[np.arange(32), nearest][accepted], rcond=None)[0] if accepted.sum() >= 3 else None
    chosen = np.asarray(row['affine_origin_x_y_um'])
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        y, x = np.nonzero(arrays['valid_mask'])
        truth = arrays['target_centre_um'][y, x]
    predicted = chosen[0] + x[:, None] * chosen[1] + y[:, None] * chosen[2]
    selected_error = float(np.linalg.norm(predicted - truth, axis=-1).mean())
    assert abs(selected_error - row['rigid_mean_error_um']) < 1e-3
    oracle_error = None
    if oracle_affine is not None:
        predicted = oracle_affine[0] + x[:, None] * oracle_affine[1] + y[:, None] * oracle_affine[2]
        oracle_error = float(np.linalg.norm(predicted - truth, axis=-1).mean())
    supported = np.asarray(row['support_query_indices'])
    support_locations = positions[np.asarray(row['support_bank_indices']) // 512]
    supported_correct = float(np.mean(np.linalg.norm(support_locations - true_query[supported], axis=-1) < 500))
    seed_pixel = np.repeat(query, 16, axis=0)
    flat_candidates = candidates.reshape(-1)
    sign = 1 - 2 * np.asarray(source['top16_parity']).reshape(-1)
    seeds = positions[flat_candidates // 512]
    seed_u = bank_u[flat_candidates % 512] * sign[:, None]
    seed_v = bank_v[flat_candidates % 512]
    seed_estimate = (seeds[:, None] + (query[None, :, 1] - seed_pixel[:, None, 1])[..., None]
                     * (67 * seed_u[:, None]) + (query[None, :, 0] - seed_pixel[:, None, 0])[..., None]
                     * (67 * seed_v[:, None]))
    best_seed_point_error = float(np.linalg.norm(seed_estimate - true_query[None], axis=-1).mean(axis=1).min())
    rows.append({'sha256': row['sha256'], 'synthetic_subject_plan_id': row['synthetic_subject_plan_id'],
                 'appearance_mode': row['appearance_mode'], 'selected_error_um': selected_error,
                 'oracle_fit_error_um': oracle_error, 'oracle_fit_available': oracle_affine is not None,
                 'oracle_match_count': int(accepted.sum()), 'selected_support_correct_fraction': supported_correct,
                 'best_seed_query_error_um': best_seed_point_error})

with (consensus / 'summary.json').open() as stream:
    original = json.load(stream)
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
selected_mean = float(np.mean([np.mean([row['selected_error_um'] for row in rows
    if row['synthetic_subject_plan_id'] == subject]) for subject in subjects]))
assert abs(selected_mean - original['identity_equal_mean_rigid_error_um']) < 1e-3
summary = {'verification': 'PASS', 'sections': len(rows), 'synthetic_subjects': len(subjects),
           'selected_identity_equal_mean_error_um': selected_mean,
           'oracle_fit_available_fraction': float(np.mean([row['oracle_fit_available'] for row in rows])),
           'oracle_fit_mean_error_um_where_available': float(np.mean([row['oracle_fit_error_um'] for row in rows
                                                               if row['oracle_fit_available']])),
           'best_seed_query_mean_error_um': float(np.mean([row['best_seed_query_error_um'] for row in rows])),
           'selected_support_correct_fraction_mean': float(np.mean([row['selected_support_correct_fraction'] for row in rows])),
           'scope': 'oracle fit uses synthetic ground truth to diagnose availability, never for selection',
           'calibrated': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'files_sha256': {
    name: hashlib.sha256((out / name).read_bytes()).hexdigest()
    for name in ('rows.jsonl', 'summary.json')},
    'consensus_completed_sha256': hashlib.sha256((consensus / 'completed.json').read_bytes()).hexdigest()}, indent=2))
print(json.dumps(summary))
