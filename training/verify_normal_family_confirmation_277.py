"""Read-only frozen output and paired-donor audit for 277."""
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path('I:/AnatomyTracker')
run = root / 'runs/normal_family_confirmation_277_train_inner_001'
config = json.loads((run / 'config.json').read_text())
completed = json.loads((run / 'completed.json').read_text())
assert config['candidate'] == completed['candidate'] == 277
assert config['fitted_parameters'] == completed['fitted_parameters'] == 0
assert config['calibrated'] is completed['calibrated'] is False
assert completed['dev_cal_final_or_public_benchmark_used'] is False
for name in ('config', 'rows', 'summary'):
    path = run / f'{name}.{"jsonl" if name == "rows" else "json"}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == completed[f'{name}_sha256']
source = Path(__file__).resolve().parent
assert hashlib.sha256((source / 'confirm_normal_family_marginal_277.py').read_bytes()).hexdigest() == config['script_sha256']
assert hashlib.sha256((source.parent / 'docs/publication/NORMAL_FAMILY_CONFIRMATION_277_PROTOCOL_20261012.md').read_bytes()).hexdigest() == config['protocol_sha256']
for name, path in (('rule_completed', root / 'runs/normal_family_marginal_276_train_inner_001/completed.json'),
                   ('transfer_cases', root / 'runs/acquired_train_transfer_272_001/cases.jsonl')):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config[f'{name}_sha256']
for path, digest in config['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
parent = json.loads((root / 'runs/pose_capture_275_v9_continuation_001/completed.json').read_text())
assert config['parent_selected_step'] == parent['selected_step']
checkpoint = root / f'runs/pose_capture_275_v9_continuation_001/pose_step_{parent["selected_step"]:05d}.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config['parent_checkpoint_sha256']
assert config['parent_checkpoint_sha256'] == parent['checkpoint_sha256'][str(parent['selected_step'])]

rows = [json.loads(line) for line in (run / 'rows.jsonl').open()]
summary = json.loads((run / 'summary.json').read_text())
assert summary['cases'] == {'coronal': 30, 'sagittal': 18} and len(rows) == 96
case_inventory = {(case['domain'], case['donor_id']): case for case in
                  (json.loads(line) for line in
                   (root / 'runs/acquired_train_transfer_272_001/cases.jsonl').open())}
for index in range(0, len(rows), 2):
    raw, marginal = rows[index:index + 2]
    assert (raw['domain'], raw['donor_id']) == (marginal['domain'], marginal['donor_id'])
    assert raw['arm'] == 'top_one' and marginal['arm'] == 'family_marginal'
    assert raw['section_id'] == marginal['section_id']
    assert raw['source_image_sha256'] == marginal['source_image_sha256']
    assert np.allclose(raw['family_mass_ap_dv_ml'], marginal['family_mass_ap_dv_ml'])
    assert abs(sum(raw['family_mass_ap_dv_ml']) - 1) < 1e-5
    assert marginal['selected_axis'] == int(np.argmax(marginal['family_mass_ap_dv_ml']))
    case = case_inventory[raw['domain'], raw['donor_id']]
    assert hashlib.sha256(f'{raw["domain"]}|{raw["donor_id"]}'.encode()).digest()[0] % 2 == 0
    for row in (raw, marginal):
        assert all(row[key] == case[key] for key in
                   ('animal_id', 'specimen_id', 'experiment_id', 'section_id', 'source_image_sha256'))
        assert row['axis_correct'] == (row['selected_axis'] == row['true_axis'])
assert len({(row['domain'], row['donor_id']) for row in rows}) == 48
for arm in ('top_one', 'family_marginal'):
    for domain, count in summary['cases'].items():
        chosen = [row for row in rows if row['arm'] == arm and row['domain'] == domain]
        saved = summary['arms'][arm][domain]
        assert len(chosen) == saved['n'] == count
        assert sum(row['axis_correct'] for row in chosen) == saved['axis_correct']
        assert abs(np.median([row['normal_angle_deg'] for row in chosen]) -
                   saved['median_normal_angle_deg']) < 1e-9
raw, marginal = summary['arms']['top_one'], summary['arms']['family_marginal']
gate = {'sagittal_gain_at_least_4': marginal['sagittal']['axis_correct'] - raw['sagittal']['axis_correct'] >= 4,
        'coronal_loss_at_most_1': raw['coronal']['axis_correct'] - marginal['coronal']['axis_correct'] <= 1,
        'sagittal_median_angle_reduction_at_least_5deg':
            raw['sagittal']['median_normal_angle_deg'] - marginal['sagittal']['median_normal_angle_deg'] >= 5}
assert gate == summary['gate'] and all(gate.values()) == summary['gate_pass']
print(json.dumps({'verified': True, 'donors': 48, 'rows': 96,
                  'sagittal_raw': raw['sagittal']['axis_correct'],
                  'sagittal_marginal': marginal['sagittal']['axis_correct'],
                  'gate_pass': summary['gate_pass']}))
