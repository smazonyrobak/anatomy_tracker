"""Read-only hash and rigid-capture summary audit for diagnostic 023."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/one_shot_all_modes_023'
receipt = json.loads((out / 'completed.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert receipt['rows'] == 185 and not receipt['calibrated'] and not receipt['public_benchmark_used']
assert sha(root / 'runs/one_shot_exposure_019/joint_step_18000.pt') == receipt['checkpoint_sha256']
assert sha(root / 'data/one_shot_fresh_synthetic_dev_panel_001/records.jsonl') == receipt['panel_records_sha256']
assert sha(Path(__file__).parent / 'diagnose_one_shot_all_modes_023.py') == receipt['source_sha256']
assert sha(out / 'rows.jsonl') == receipt['rows_sha256']
assert sha(out / 'summary.json') == receipt['summary_sha256']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == summary['cases'] == 185 and len({row['sha256'] for row in rows}) == 185
animals = sorted({row['animal_id'] for row in rows})
assert len(animals) == summary['identities'] == 8
for row in rows:
    assert len(row['top8']) == 8 and len(row['rigid_um']) == len(row['prior']) == 32
    assert abs(row['best8_um'] - min(row['rigid_um'][branch] for branch in row['top8'])) < 1e-4
    assert abs(row['best32_um'] - min(row['rigid_um'])) < 1e-4
for name in ('best8_um', 'best32_um'):
    values = np.asarray([row[name] for row in rows])
    group = summary[name]
    assert abs(group['case_mean'] - values.mean()) < 1e-6
    assert abs(group['identity_equal_mean'] - np.mean([
        np.mean([row[name] for row in rows if row['animal_id'] == animal])
        for animal in animals])) < 1e-6
    for threshold, label in ((250, 'below_250_um'), (500, 'below_500_um'), (1000, 'below_1000_um')):
        assert abs(group[label] - (values < threshold).mean()) < 1e-8
print(json.dumps({'status': 'passed', 'cases': len(rows), 'branches': 32}))
