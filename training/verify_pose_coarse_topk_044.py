"""Independent read-only audit of the frozen 044 coarse-retrieval diagnosis."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/pose_coarse_topk_044_diagnostic'
panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
run = root / 'runs/pose_feedback_global_041_pilot'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
source = Path(__file__).with_name('diagnose_pose_coarse_topk_044.py')
protocol = Path(__file__).parents[1] / 'docs/publication/POSE_COARSE_TOPK_044_PROTOCOL_20261003.md'
receipt = json.loads((out / 'completed.json').read_text())
config = json.loads((out / 'config.json').read_text())
for name in ('config', 'rows', 'summary'):
    path = out / (name + ('.jsonl' if name == 'rows' else '.json'))
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt[name + '_sha256']
for name, path in (('source', source), ('protocol', protocol), ('parent', parent),
                   ('head', run / 'joint_step_01500.pt'),
                   ('panel_records', panel / 'records.jsonl'),
                   ('train_completed', run / 'completed.json')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == config[name + '_sha256']
assert (config['step'], config['beam'], config['truth_threshold_um'],
        config['key_support_threshold']) == (1500, 8, 1500, .1)
assert not config['calibrated'] and not config['public_benchmark_used']
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
eligible = [row for row in panel_rows if row['eligible']]
assert len(rows) == len(eligible) == receipt['rows'] == 177
assert summary['synthetic_eligible_sections'] == 177
assert len({row['synthetic_subject_plan_id'] for row in rows}) == summary['synthetic_subjects'] == 8
assert len({(row['animal_id'], row['specimen_id'], row['experiment_id'], row['section_id'])
            for row in rows}) == 177
for row, record in zip(rows, eligible):
    for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
                'synthetic_subject_plan_id', 'appearance_mode', 'sha256'):
        assert row[key] == record[key]
    assert 0 < row['valid_cells'] <= 256
    assert row['prior_best8_index'] in range(8)
    assert 0 <= row['prior_best8_um'] <= row['prior_top1_um'] + 1e-4
    for label in ('top1', 'best8'):
        available = row[label + '_available_cells']
        assert 0 <= available <= row['valid_cells']
        hits = [row[f'{label}_top{k}_hits'] for k in (8, 32, 64)]
        assert 0 <= hits[0] <= hits[1] <= hits[2] <= available
        assert row[label + '_top64_quadrants'] in range(5)
        assert (hits[2] == 0) == (row[label + '_top64_quadrants'] == 0)
identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
assert math.isclose(summary['mean_valid_cells'],
                    np.mean([row['valid_cells'] for row in rows]), abs_tol=1e-6)
for label in ('top1', 'best8'):
    assert summary[label]['sections_with_available_cells'] == sum(
        row[label + '_available_cells'] > 0 for row in rows)
    for name in ('availability', 'top8_absolute', 'top32_absolute', 'top64_absolute',
                 'top8_conditional', 'top32_conditional', 'top64_conditional',
                 'top64_three_quadrant_fraction'):
        means = []
        for identity in identities:
            cases = [row for row in rows if row['synthetic_subject_plan_id'] == identity]
            if name == 'availability':
                values = [row[label + '_available_cells'] / row['valid_cells'] for row in cases]
            elif name == 'top64_three_quadrant_fraction':
                values = [float(row[label + '_top64_quadrants'] >= 3) for row in cases]
            else:
                k, denominator = name.split('_')
                denominator_field = ('valid_cells' if denominator == 'absolute'
                                     else label + '_available_cells')
                values = [row[f'{label}_{k}_hits'] / row[denominator_field]
                          for row in cases if row[denominator_field] > 0]
            if values:
                means.append(np.mean(values))
        assert math.isclose(summary[label][name], np.mean(means), abs_tol=1e-8)
        assert 0 <= summary[label][name] <= 1
    for denominator in ('absolute', 'conditional'):
        assert (summary[label]['top8_' + denominator] <=
                summary[label]['top32_' + denominator] <=
                summary[label]['top64_' + denominator])
assert summary['fine_reranker_justified'] == (
    summary['best8']['top64_conditional'] >= .8 and
    summary['top1']['top64_conditional'] >= .6 and
    summary['best8']['top64_three_quadrant_fraction'] >= .8)
print(json.dumps({'audit': 'PASS', 'sections': len(rows),
                  'fine_reranker_justified': summary['fine_reranker_justified']}))
