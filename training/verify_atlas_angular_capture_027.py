"""Independent frozen lineage and grouped angular-capture summary audit."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/atlas_angular_capture_027'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((out / 'config.json').read_text())
completed = json.loads((out / 'completed.json').read_text())
assert config['cases'] == 185 and config['points_per_case'] == 32
assert config['finite_thickness'] and not config['weights_updated']
assert not config['calibrated'] and not config['public_benchmark_used']
assert sha(root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt') == config['checkpoint_sha256']
assert sha(panel / 'completed.json') == config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == config['panel_records_sha256']
for name, digest in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
for name in ('config', 'rows', 'summary'):
    assert sha(out / (name + ('.jsonl' if name == 'rows' else '.json'))) == completed[name + '_sha256']
records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']}
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
conditions = {row[0]: row for row in config['conditions']}
assert len(conditions) == 25
assert len(rows) == completed['rows'] == summary['rows'] == 185 * 25
assert len({row['section_id'] for row in rows}) == summary['sections'] == 185
assert len({row['synthetic_subject_plan_id'] for row in rows}) == summary['synthetic_subjects'] == 8
assert not summary['calibrated'] and not summary['public_benchmark_used']
assert {(row['sha256'], row['condition']) for row in rows} == {
    (digest, condition) for digest in records for condition in conditions}
for row in rows:
    reference = records[row['sha256']]
    assert row['animal_id'] == reference['animal_id']
    assert row['appearance_mode'] == reference['appearance_mode']
    assert (row['condition'], row['family'], row['degrees'], row['axis'], row['sign']) == tuple(
        conditions[row['condition']])
    assert 0 <= row['top1_exact'] <= row['top1_500um'] <= 1
    assert np.isfinite(row['mean_margin']) and np.isfinite(row['mean_correct_similarity'])
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
assert len(summary['conditions']) == len(conditions)
for group in summary['conditions']:
    selected = [row for row in rows if row['condition'] == group['condition']]
    assert len(selected) == group['sections'] == 185
    for field in ('top1_exact', 'top1_500um', 'mean_margin', 'mean_correct_similarity'):
        value = np.mean([np.mean([row[field] for row in selected
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
        assert abs(value - group[field]) < 1e-9
    for mode, claimed in group['mode_top1_500um'].items():
        subset = [row for row in selected if row['appearance_mode'] == mode]
        mode_subjects = sorted({row['synthetic_subject_plan_id'] for row in subset})
        value = np.mean([np.mean([row['top1_500um'] for row in subset
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in mode_subjects])
        assert abs(value - claimed) < 1e-9
chosen = [row for row in summary['conditions']
          if (row['family'], row['degrees']) in (('tilt', 10), ('roll', 15))]
gate = all(np.mean([row['top1_500um'] for row in chosen if row['family'] == family]) >= .4
           for family in ('tilt', 'roll'))
gate &= all(np.mean([row['mode_top1_500um'][mode] for row in chosen
                     if row['family'] == family]) >= .5 *
            np.mean([row['top1_500um'] for row in chosen if row['family'] == family])
            for mode in ('raw', 'exact_black', 'imperfect_brush') for family in ('tilt', 'roll'))
assert bool(gate) == summary['coarse_search_plausible_gate']
print(json.dumps({'status': 'passed', 'cases': len(records), 'rows': len(rows),
                  'coarse_search_plausible_gate': bool(gate)}))
