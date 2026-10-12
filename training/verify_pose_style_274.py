"""Independent read-only recount of the frozen 274 TRAIN-inner comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/pose_style_274_train_inner_001'
SOURCE = Path(__file__).resolve().parent
receipt = json.loads((RUN / 'completed.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
assert receipt['candidate'] == config['candidate'] == 274
for name in ('config', 'draws', 'training', 'readouts'):
    extension = 'json' if name == 'config' else 'jsonl'
    with (RUN / f'{name}.{extension}').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt[f'{name}_sha256']
for key, digest in receipt['checkpoint_sha256'].items():
    arm, step = key.rsplit('_', 1)
    with (RUN / f'{arm}_step_{int(step):05d}.pt').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
for name, digest in config['source_sha256'].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((SOURCE.parent / 'docs/publication/FOURIER_STYLE_274_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
with (ROOT / 'runs/pose_capture_270_v9_continuation_001/pose_step_08000.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == config['parent_checkpoint_sha256']

draws = [json.loads(line) for line in (RUN / 'draws.jsonl').open()]
training = [json.loads(line) for line in (RUN / 'training.jsonl').open()]
readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open()]
assert len(training) == 4000 and len(readouts) == 24
for arm in ('control_v9', 'fourier_v10'):
    subset = [row for row in draws if row['arm'] == arm]
    trained = [row for row in training if row['arm'] == arm]
    assert len(trained) == 2000 and [row['step'] for row in trained] == list(range(1, 2001))
    assert sum(row['used'] for row in subset) == receipt['counters'][arm]['accepted'] == 8000
    assert len(subset) == receipt['counters'][arm]['attempted']
    assert [row['attempt'] for row in subset] == list(range(1, len(subset) + 1))
    assert all(0 <= row['base_index'] < 56 for row in subset)
    if arm == 'fourier_v10':
        for row in subset:
            style = row['provenance']['one_shot_slide_artifacts_v10']
            if style['fourier_style_applied']:
                domain = style['style_domain']
                donor = style['style_source']['donor_id']
                assert hashlib.sha256(f'{domain}|{donor}'.encode()).digest()[0] % 2 == 0
    for step in (0, 1000, 2000):
        for panel in ('fresh_v9', 'old_v8', 'coronal', 'sagittal'):
            row, = [row for row in readouts if (row['arm'], row['step'], row['panel']) ==
                    (arm, step, panel)]
            records, metrics = row['rows'], row['metrics']
            assert len(records) == metrics['n'] == (64 if panel in ('fresh_v9', 'old_v8') else
                                                    28 if panel == 'coronal' else 22)
            frozen = receipt['readouts'][arm][str(step)][panel]
            assert frozen == metrics
            if panel in ('fresh_v9', 'old_v8'):
                assert metrics['top16_hits'] == sum(item['top16_hit'] for item in records)
                assert abs(metrics['mean_top1_mm'] - np.mean([
                    item['top1_mm'] for item in records])) < 1e-9
            else:
                assert metrics['first_axis_correct'] == sum(
                    item['first_axis_correct'] for item in records)
                assert abs(metrics['median_weak_normal_angle_deg'] - np.median([
                    item['normal_gap_to_weak_affine_deg'] for item in records])) < 1e-9
                assert all(hashlib.sha256(f'{panel}|{item["donor_id"]}'.encode()).digest()[0]
                           % 2 == 1 for item in records)

control = receipt['readouts']['control_v9']['2000']
style = receipt['readouts']['fourier_v10']['2000']
gate = {'sagittal_gain_at_least_8':
            style['sagittal']['first_axis_correct'] - control['sagittal']['first_axis_correct'] >= 8,
        'coronal_loss_at_most_2':
            control['coronal']['first_axis_correct'] - style['coronal']['first_axis_correct'] <= 2,
        'fresh_top16_loss_at_most_4':
            control['fresh_v9']['top16_hits'] - style['fresh_v9']['top16_hits'] <= 4,
        'old_top16_at_least_40': style['old_v8']['top16_hits'] >= 40}
assert gate == receipt['gate'] and all(gate.values()) == receipt['gate_pass']
print(json.dumps({'verified': True, 'draws': len(draws), 'training_batches': len(training),
                  'readout_blocks': len(readouts), 'gate': gate, 'gate_pass': all(gate.values())}))
