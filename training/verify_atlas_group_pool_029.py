"""Independent baseline binding and grouped pooled-descriptor result audit."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/atlas_group_pool_029'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((out / 'config.json').read_text())
completed = json.loads((out / 'completed.json').read_text())
assert config['cases'] == completed['rows'] == 185
assert config['query_points_per_case'] == 32 and config['atlas_candidates_per_case'] == 288
assert config['groups_degrees'] == [90, 30] and config['atlas_half_shift_degrees'] == [45, 15]
assert not config['weights_updated'] and not config['calibrated'] and not config['public_benchmark_used']
assert sha(root / 'runs/atlas_oriented_patch_025/patch_step_03000.pt') == config['checkpoint_sha256']
assert sha(panel / 'completed.json') == config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == config['panel_records_sha256']
prior = root / 'runs/atlas_oriented_patch_025_development_eval/summary.json'
assert sha(prior) == config['baseline_summary_sha256']
baseline = next(row for row in json.loads(prior.read_text())['checkpoints'] if row['step'] == 3000)
assert abs(baseline['recall1_500um'] - config['baseline_recall1_500um']) < 1e-9
assert abs(baseline['recall16_500um'] - config['baseline_recall16_500um']) < 1e-9
for name, digest in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == digest
for name in ('config', 'rows', 'summary'):
    assert sha(out / (name + ('.jsonl' if name == 'rows' else '.json'))) == completed[name + '_sha256']
panel_records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open())
                 if row['eligible']}
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == len(panel_records) == summary['cases'] == 185
assert len({row['synthetic_subject_plan_id'] for row in rows}) == summary['synthetic_subjects'] == 8
assert not summary['calibrated'] and not summary['public_benchmark_used']
conditions = ('group4', 'group4_halfshift', 'group12', 'group12_halfshift')
subjects = sorted({row['synthetic_subject_plan_id'] for row in rows})
for row in rows:
    reference = panel_records[row['sha256']]
    assert row['animal_id'] == reference['animal_id']
    assert row['appearance_mode'] == reference['appearance_mode']
    assert set(row['conditions']) == set(conditions)
    for condition in conditions:
        score = row['conditions'][condition]
        assert 0 <= score['recall1_500um'] <= score['recall16_500um'] <= 1
for condition in conditions:
    group = summary['conditions'][condition]
    for field in ('recall1_500um', 'recall16_500um'):
        value = np.mean([np.mean([row['conditions'][condition][field] for row in rows
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in subjects])
        assert abs(value - group[field]) < 1e-9
    for mode, claimed in group['mode_recall1_500um'].items():
        subset = [row for row in rows if row['appearance_mode'] == mode]
        identities = sorted({row['synthetic_subject_plan_id'] for row in subset})
        value = np.mean([np.mean([row['conditions'][condition]['recall1_500um'] for row in subset
                                  if row['synthetic_subject_plan_id'] == subject]) for subject in identities])
        assert abs(value - claimed) < 1e-9
groups = summary['conditions']
gate = (groups['group12']['recall1_500um'] >= .8 and
        groups['group12']['recall1_500um'] - groups['group12_halfshift']['recall1_500um'] <= .1 and
        all(min(groups[name]['mode_recall1_500um'].values()) >= .5 * groups[name]['recall1_500um']
            for name in ('group12', 'group12_halfshift')))
assert gate == summary['pooled_bank_gate']
print(json.dumps({'status': 'passed', 'cases': len(rows), 'pooled_bank_gate': gate}))
