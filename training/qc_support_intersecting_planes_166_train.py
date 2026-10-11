"""Bounded TRAIN-only tissue-coverage QC for support-intersecting planes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


root = Path('I:/AnatomyTracker')
out = root / 'reports/support_intersecting_plane_qc_166_train_001'
out.mkdir(parents=True, exist_ok=True)
old_run = root / 'runs/joint_feedback_162_pose_gradient_001'
old_receipt = json.loads((old_run / 'completed.json').read_text())
old_draws_bytes = (old_run / 'draws.jsonl').read_bytes()
assert old_receipt['status'] == 'complete'
assert hashlib.sha256(old_draws_bytes).hexdigest() == old_receipt['output_sha256']['draws.jsonl']
old_rows = [json.loads(line) for line in old_draws_bytes.splitlines()]
old_fraction = np.array([row['valid_fraction'] for row in old_rows if row['split'] == 'train'])
assert len(old_fraction) == 2000

context = load_streaming_synthetic_v7_64('cuda')
rng = np.random.default_rng(202610111666)
schedule = [(int(rng.integers(60)), int(rng.integers(0, 2**63 - 1))) for _ in range(128)]
assert len({seed for _, seed in schedule}) == len(schedule)
rows = []
for index, (base_index, seed) in enumerate(schedule):
    source = sample_panel_geometry_train_166(context, [base_index], seed, 256)
    sample = sample_one_shot_slide_artifacts_v4(context, [base_index], seed, 256,
                                                 source_sample=source)
    p = source['provenance'][0]
    point = np.asarray(p['support_point_physical_ap_dv_ml_um'])
    origin, edge_u, edge_v = np.asarray(p['physical_ouv_um'])
    u = float(np.dot(point - origin, edge_u) / np.dot(edge_u, edge_u))
    v = float(np.dot(point - origin, edge_v) / np.dot(edge_v, edge_v))
    plane_residual_um = float(np.dot(point - origin, np.asarray(p['physical_unit_normal_ap_dv_ml'])))
    rows.append({'index': index, 'base_index': base_index, 'draw_seed': seed,
                 'physical_section_id': p['physical_section_id'],
                 'animal_id': p['base_lineage']['animal_id'],
                 'support_point_index': p['support_point_index'],
                 'support_point_physical_ap_dv_ml_um': point.tolist(),
                 'support_points_sha256': p['support_points_sha256'],
                 'physical_ouv_um': p['physical_ouv_um'],
                 'normal_ap_dv_ml': p['physical_unit_normal_ap_dv_ml'],
                 'support_point_image_uv': [u, v],
                 'support_point_plane_residual_um': plane_residual_um,
                 'pre_artifact_support_fraction': float((source['support'][0] > .05).float().mean()),
                 'pre_artifact_visible_fraction': float((source['visible'][0] > .05).float().mean()),
                 'valid_fraction': float(sample['valid_mask'][0].float().mean()),
                 'valid_pixels': int(sample['valid_mask'][0].sum()),
                 'eligible': bool(sample['eligible'][0]),
                 'appearance_mode': p['mode'],
                 'artifact_events': sample['provenance'][0]['one_shot_slide_artifacts_v3']['events'],
                 'exposure': sample['provenance'][0]['one_shot_slide_artifacts_v4']['exposure']})
    del source, sample

fraction = np.array([row['valid_fraction'] for row in rows])
new_summary = {'n': len(rows), 'zero_valid_n': int((fraction == 0).sum()),
               'under2pct_n': int((fraction < .02).sum()),
               'under8pct_n': int((fraction < .08).sum()),
               'eligible_n': sum(row['eligible'] for row in rows),
               'valid_fraction_quantiles_0_10_25_50_75_90_100':
                   np.quantile(fraction, [0, .1, .25, .5, .75, .9, 1]).tolist(),
               'mean_valid_fraction': float(fraction.mean()),
               'support_point_outside_image_n': sum(not (0 <= row['support_point_image_uv'][0] < 1
                                                       and 0 <= row['support_point_image_uv'][1] < 1)
                                                    for row in rows),
               'max_abs_support_point_plane_residual_um':
                   max(abs(row['support_point_plane_residual_um']) for row in rows),
               'distinct_physical_section_ids': len({row['physical_section_id'] for row in rows}),
               'distinct_plane_origins': len({tuple(row['physical_ouv_um'][0]) for row in rows})}
old_summary = {'n': len(old_fraction), 'zero_valid_n': int((old_fraction == 0).sum()),
               'under2pct_n': int((old_fraction < .02).sum()),
               'under8pct_n': int((old_fraction < .08).sum()),
               'valid_fraction_quantiles_0_10_25_50_75_90_100':
                   np.quantile(old_fraction, [0, .1, .25, .5, .75, .9, 1]).tolist(),
               'mean_valid_fraction': float(old_fraction.mean())}
summary = {'new_166_train_128': new_summary, 'frozen_162_train_2000': old_summary,
           'frozen_162_completed_sha256': hashlib.sha256((old_run / 'completed.json').read_bytes()).hexdigest(),
           'frozen_162_draws_sha256': hashlib.sha256(old_draws_bytes).hexdigest(),
           'seed': 202610111666, 'base_indices': '0–59 TRAIN only', 'side': 256,
           'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                             for name in ('arbitrary_plane_panel_geometry_train_166.py',
                                          'arbitrary_plane_one_shot_slide_artifacts_v3.py',
                                          'arbitrary_plane_one_shot_slide_artifacts_v4.py',
                                          Path(__file__).name)},
           'scope': 'synthetic TRAIN geometry/coverage QC only; not model accuracy or real physical sections'}
(out / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps({'output': str(out), **new_summary, 'frozen_162': old_summary}, allow_nan=False))
