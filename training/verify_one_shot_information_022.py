"""Read-only provenance and summary audit for 022 image-information diagnostic."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

root = Path('I:/AnatomyTracker')
out = root / 'runs/one_shot_information_022'
receipt = json.loads((out / 'completed.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert receipt['rows'] == 185 and not receipt['calibrated'] and not receipt['public_benchmark_used']
assert sha(root / 'runs/one_shot_exposure_019/joint_step_18000.pt') == receipt['checkpoint_sha256']
assert sha(root / 'data/one_shot_fresh_synthetic_dev_panel_001/records.jsonl') == receipt['panel_records_sha256']
assert sha(root / 'runs/one_shot_exposure_019_candidate_rank_diagnostic/rows.jsonl') == receipt['control_rows_sha256']
assert sha(Path(__file__).parent / 'diagnose_one_shot_information_022.py') == receipt['source_sha256']
assert sha(out / 'rows.jsonl') == receipt['rows_sha256']
assert sha(out / 'summary.json') == receipt['summary_sha256']
rows = [json.loads(line) for line in (out / 'rows.jsonl').open()]
summary = json.loads((out / 'summary.json').read_text())
assert len(rows) == 185 and len({row['sha256'] for row in rows}) == 185
assert len({row['animal_id'] for row in rows}) == 8
assert len(summary) == 6
for row in rows:
    assert len(row['choice']) == len(row['physical_mapped_um']) == len(row['fitted_score']) == 8
    assert set(row['disagreement']) == {'96_oracle', '96_predicted', '256_oracle', '256_predicted'}
    assert all(len(values) == 8 and np.isfinite(values).all()
               for values in row['disagreement'].values())
animals = sorted({row['animal_id'] for row in rows})
for group in summary:
    name = group['method']
    if name == 'fitted':
        chosen = [int(np.argmax(row['fitted_score'])) for row in rows]
    elif name == 'best8':
        chosen = [int(np.argmin(row['physical_mapped_um'])) for row in rows]
    else:
        chosen = [int(np.argmin(row['disagreement'][name])) for row in rows]
    errors = [row['physical_mapped_um'][index] for row, index in zip(rows, chosen)]
    identity_equal = np.mean([np.mean([error for row, error in zip(rows, errors)
                                       if row['animal_id'] == animal]) for animal in animals])
    best = [int(np.argmin(row['physical_mapped_um'])) for row in rows]
    assert abs(group['identity_equal_mapped_um'] - identity_equal) < 1e-6
    assert abs(group['case_mean_mapped_um'] - np.mean(errors)) < 1e-6
    assert abs(group['top1_oracle_fraction'] - np.mean(np.asarray(chosen) == best)) < 1e-8
print(json.dumps({'status': 'passed', 'cases': len(rows), 'methods': len(summary)}))
