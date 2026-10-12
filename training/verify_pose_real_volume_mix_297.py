"""Independent raw-result and provenance audit for the frozen 297 pilot."""
import hashlib
import json
from pathlib import Path
from statistics import fmean, median


ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/pose_real_volume_mix_297_train_inner_001'
receipt = json.loads((RUN / 'completed.json').read_text())
config = json.loads((RUN / 'config.json').read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert receipt['candidate'] == config['candidate'] == 297
assert receipt['calibrated'] is False and receipt['dev_cal_final_or_public_benchmark_used'] is False
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert sha256(SOURCE.parent / 'docs/publication/POSE_REAL_VOLUME_MIX_297_PROTOCOL_20261012.md') == config['protocol_sha256']
assert all(sha256(SOURCE / name) == digest for name, digest in config['source_sha256'].items())
assert sha256(ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt') == config['parent_checkpoint_sha256'] == receipt['parent_checkpoint_sha256']
for binding in ('reserved_source_sha256', 'acquired_source_sha256',
                'geometry_audit_sha256'):
    assert all(sha256(Path(path)) == digest for path, digest in config[binding].items())
for name, path in {'frozen_270_v9': ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
                   'frozen_268_v8': ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt'}.items():
    assert sha256(path) == config['old_panel_sha256'][name]
assert all(sha256(RUN / name) == digest for name, digest in receipt['output_sha256'].items())
for key, digest in receipt['checkpoint_sha256'].items():
    arm, step = key.rsplit('_', 1)
    assert sha256(RUN / f'{arm}_step_{int(step):05d}.pt') == digest

train = set(map(str, config['virtual_train_animal_ids']))
held = set(map(str, config['virtual_inner_readout_animal_ids']))
acquired = set(map(str, config['acquired_readout_donor_ids']))
assert (len(train), len(held)) == (53, 8)
assert not train & held and not train & acquired and not held & acquired
assert set(map(str, config['excluded_audit_animal_ids'])) == {'16207', '14855', '15975'}

draws = [json.loads(line) for line in (RUN / 'draws.jsonl').open()]
synthetic = [row for row in draws if row['role'] == 'matched_synthetic']
virtual = [row for row in draws if row['role'] == 'mixed_weak_virtual_train']
assert len(synthetic) == receipt['attempted_synthetic_draws'] == 2211
assert sum(row['used'] for row in synthetic) == receipt['accepted_synthetic_presentations_per_arm'] == 2000
assert len(virtual) == receipt['attempted_virtual_draws'] == 502
assert sum(row['used'] for row in virtual) == receipt['updates_per_arm'] == 500
assert all(str(row['provenance']['animal_id']) in train for row in virtual)
assert {row['step'] for row in draws} == set(range(1, 501))
for name, count, allowed in (('inner_draws.jsonl', 64, None),
                             ('virtual_inner_draws.jsonl', 24, held)):
    rows = [json.loads(line) for line in (RUN / name).open()]
    assert sum(row['used'] for row in rows) == count
    if allowed is not None:
        assert all(str(row['provenance']['animal_id']) in allowed for row in rows)

readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open()]
assert len(readouts) == 36
indexed = {(row['step'], row['arm'], row['panel']): row for row in readouts}
assert len(indexed) == 36
assert {key[0] for key in indexed} == {0, 250, 500}
for row in readouts:
    metrics, cases = row['metrics'], row['rows']
    assert metrics['n'] == len(cases)
    if row['panel'] in ('new_v9', 'frozen_270_v9', 'frozen_268_v8', 'virtual_weak'):
        assert abs(metrics['mean_top1_mm'] - fmean(case['top1_mm'] for case in cases)) < 1e-9
        assert abs(metrics['mean_normal_axis_error_deg'] - fmean(
            case['normal_axis_error_deg'] for case in cases)) < 1e-9
        assert metrics['top16_hits'] == sum(case['top16_hit'] for case in cases)
    else:
        assert metrics['first_axis_correct'] == sum(case['first_axis_correct'] for case in cases)
        assert abs(metrics['median_weak_normal_angle_deg'] - median(
            case['normal_gap_to_weak_affine_deg'] for case in cases)) < 1e-9
    assert metrics == receipt['readouts'][str(row['step'])][row['arm']][row['panel']]
for panel in ('new_v9', 'frozen_270_v9', 'frozen_268_v8', 'virtual_weak', 'coronal', 'sagittal'):
    assert indexed[0, 'control', panel]['rows'] == indexed[0, 'mixed', panel]['rows']

control = {panel: indexed[500, 'control', panel] for panel in
           ('new_v9', 'frozen_270_v9', 'frozen_268_v8', 'virtual_weak', 'coronal', 'sagittal')}
mixed = {panel: indexed[500, 'mixed', panel] for panel in control}
base_gains = {str(base): fmean(a['top1_mm'] - b['top1_mm'] for a, b in zip(
    control['new_v9']['rows'], mixed['new_v9']['rows']) if a['base_index'] == base)
    for base in range(56, 64)}
donor_gains = {donor: fmean(a['top1_mm'] - b['top1_mm'] for a, b in zip(
    control['virtual_weak']['rows'], mixed['virtual_weak']['rows'])
    if str(a['donor_id']) == donor) for donor in held}
assert all(abs(value - receipt['base_gains_mm'][key]) < 1e-9 for key, value in base_gains.items())
assert all(abs(value - receipt['virtual_donor_gains_mm'][key]) < 1e-9 for key, value in donor_gains.items())
c = {name: row['metrics'] for name, row in control.items()}
m = {name: row['metrics'] for name, row in mixed.items()}
gate = {
    'new_top1_gain_at_least_0p15mm': c['new_v9']['mean_top1_mm'] - m['new_v9']['mean_top1_mm'] >= .15,
    'at_least_six_positive_bases': sum(value > 0 for value in base_gains.values()) >= 6,
    'new_top16_loses_at_most_two': m['new_v9']['top16_hits'] >= c['new_v9']['top16_hits'] - 2,
    'frozen_270_top16_loses_at_most_two': m['frozen_270_v9']['top16_hits'] >= c['frozen_270_v9']['top16_hits'] - 2,
    'frozen_268_top16_loses_at_most_two': m['frozen_268_v8']['top16_hits'] >= c['frozen_268_v8']['top16_hits'] - 2,
    'virtual_weak_top1_gain_at_least_0p15mm': c['virtual_weak']['mean_top1_mm'] - m['virtual_weak']['mean_top1_mm'] >= .15,
    'at_least_six_virtual_readout_donors_positive': sum(value > 0 for value in donor_gains.values()) >= 6,
    'sagittal_weak_family_gain_at_least_four': m['sagittal']['first_axis_correct'] >= c['sagittal']['first_axis_correct'] + 4,
    'coronal_weak_family_loses_at_most_one': m['coronal']['first_axis_correct'] >= c['coronal']['first_axis_correct'] - 1,
}
assert gate == receipt['gate'] and receipt['gate_pass'] == all(gate.values()) is False
print(json.dumps({'audit_pass': True, 'gate_pass': False, 'cases':
    {'synthetic': 64, 'virtual_weak': 24, 'acquired': 50},
    'new_top1_gain_mm': c['new_v9']['mean_top1_mm'] - m['new_v9']['mean_top1_mm'],
    'synthetic_bases_improved': sum(value > 0 for value in base_gains.values()),
    'virtual_weak_gain_mm': c['virtual_weak']['mean_top1_mm'] - m['virtual_weak']['mean_top1_mm'],
    'virtual_donors_improved': sum(value > 0 for value in donor_gains.values())}))
