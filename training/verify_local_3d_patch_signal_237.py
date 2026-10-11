"""Independent post-exit audit of the frozen 237 mechanism assay."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/local_3d_patch_signal_237_001'
receipt = json.loads((run / 'completed.json').read_text())
assert receipt['cases'] == 64 and receipt['training_updates'] == 0
assert not receipt['reserved_animal_used'] and not receipt['public_benchmark_used']
assert not receipt['calibrated']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sha(source.parent / 'docs/publication/LOCAL_3D_PATCH_SIGNAL_237_PROTOCOL_20261011.md') == receipt['protocol_sha256']
for name, digest in receipt['script_and_source_sha256'].items():
    assert sha(source / name) == digest, name
for name, digest in receipt['output_sha256'].items():
    assert sha(run / name) == digest, name
assert sha(root / 'runs/pose_appearance_exposure_226_train_001/joint_step_01200.pt') == receipt['parent_checkpoint_sha256']
assert sha(root / 'runs/blind_candidate_reachability_235_001/cases.jsonl') == receipt['frozen_235_rows_sha256']

rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
previous = [json.loads(line) for line in (root / 'runs/blind_candidate_reachability_235_001/cases.jsonl').read_text().splitlines()]
summary = json.loads((run / 'summary.json').read_text())
assert len(rows) == len(previous) == 64
assert len({r['physical_section_id'] for r in rows}) == 64
assert [r['case_index'] for r in rows] == list(range(64))
for row, prior in zip(rows, previous):
    assert all(row[key] == prior[key] for key in ('base', 'seed', 'appearance',
        'physical_section_id', 'input_sha256'))
    near = next((c for c in sorted(prior['top16_candidates'], key=lambda c: c['tissue_mm'])
        if c['tissue_mm'] < 2), None)
    assert row['near_branch'] == (None if near is None else near['index'])
    if near is not None:
        assert abs(row['results']['near']['initial_mm'] - near['tissue_mm']) < 1e-5
    for role, detail in row['results'].items():
        assert detail['initial_mm'] >= 0
        for arm, result in detail['arms'].items():
            assert arm in ('intact', 'null') and result['fitted_mm'] >= 0
            assert result['fit_sites'] <= 256 and result['supported_query_sites'] <= 256
            assert np.isfinite([result['fitted_mm'], result['raw_five_point_movement_mm']]).all()
            if role in ('near', 'wrong'):
                assert len(result['score']) == len(result['support']) == 256
                assert np.isfinite(result['score']).all()
    if row['wrong_branch'] is not None:
        assert row['near_branch'] is not None and 'pair' in row
        for arm in ('intact', 'null'):
            left = row['results']['near']['arms'][arm]
            right = row['results']['wrong']['arms'][arm]
            common = np.asarray(left['support']) & np.asarray(right['support'])
            assert row['pair'][arm]['common_sites'] == int(common.sum())
            expected = float(np.mean(np.asarray(left['score'])[common] - np.asarray(right['score'])[common])) if common.any() else None
            assert (expected is None and row['pair'][arm]['near_minus_wrong_score'] is None) or abs(expected - row['pair'][arm]['near_minus_wrong_score']) < 1e-9

for appearance in ('v4', 'v7', 'all'):
    group = [r for r in rows if appearance == 'all' or r['appearance'] == appearance]
    near = [r for r in group if r['near_branch'] is not None]
    paired = [r for r in group if r.get('pair', {}).get('intact', {}).get('common_sites', 0) >= 16]
    found = summary[appearance]
    assert found['cases'] == len(group) and found['near_cases'] == len(near)
    assert found['paired_cases'] == len(paired)
    metrics = {'near_initial_mean_mm': np.mean([r['results']['near']['initial_mm'] for r in near])}
    for arm in ('intact', 'null'):
        metrics[f'near_{arm}_fitted_mean_mm'] = np.mean([r['results']['near']['arms'][arm]['fitted_mm'] for r in near])
        metrics[f'true_{arm}_drift_mean_mm'] = np.mean([r['results']['true']['arms'][arm]['fitted_mm'] for r in group])
        assert found[f'pair_{arm}_wins'] == sum(r['pair'][arm]['near_minus_wrong_score'] > 0 for r in paired)
    for key, value in metrics.items():
        assert abs(found[key] - value) < 1e-9, (appearance, key)
    assert found['near_reflection_correct'] == sum(r['results']['near']['reflection'] == r['true_reflection'] for r in near)

all_result = summary['all']
gate = (all_result['near_initial_mean_mm'] - all_result['near_intact_fitted_mean_mm'] >= .25
    and all_result['near_null_fitted_mean_mm'] - all_result['near_intact_fitted_mean_mm'] >= .15
    and all_result['true_intact_drift_mean_mm'] <= .20
    and all(summary[a]['near_initial_mean_mm'] > summary[a]['near_intact_fitted_mean_mm'] for a in ('v4', 'v7')))
assert summary['stage_a_pass'] == gate is False
for role in ('near', 'true'):
    for arm in ('intact', 'null'):
        group = [r['results'][role]['arms'][arm] for r in rows if role in r['results']]
        print(role, arm, 'n', len(group), 'matched_le_0p5', np.mean([r['matched_le_0p5'] for r in group if r['matched_le_0p5'] is not None]),
              'matched_le_1p5', np.mean([r['matched_le_1p5'] for r in group if r['matched_le_1p5'] is not None]))
print('AUDIT_PASS', json.dumps(summary['all'], sort_keys=True))
