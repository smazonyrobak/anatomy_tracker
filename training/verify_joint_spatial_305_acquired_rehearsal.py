"""Independent post-exit audit of the frozen 305 acquired-axis rehearsal."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import torch

root = Path('I:/AnatomyTracker')
run = root / 'runs/joint_spatial_305_acquired_rehearsal_001'
assert (run / 'completed.json').is_file(), 'Runner has not completed'
source = Path(__file__).resolve().with_name('train_joint_spatial_305_acquired_rehearsal.py')
protocol = source.parents[1] / 'docs/publication/JOINT_SPATIAL_305_ACQUIRED_REHEARSAL_PROTOCOL_20261014.md'
config = json.loads((run / 'config.json').read_text(encoding='utf8'))
summary = json.loads((run / 'summary.json').read_text(encoding='utf8'))
receipt = json.loads((run / 'completed.json').read_text(encoding='utf8'))
assert receipt['summary'] == summary
for name, expected in receipt['output_sha256'].items():
    with (run / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
for name, path in (('source_sha256', source), ('protocol_sha256', protocol),
                   ('split_sha256', source.with_name('lupori_animal_split_185.json')),
                   ('parent_checkpoint_sha256', root / 'runs/joint_spatial_305_fluorescence_adaptation_001/adapt/pose_step_00600.pt'),
                   ('panel_sha256', root / 'runs/joint_spatial_feedback_305_pose_002/inner_panel.pt'),
                   ('prior_cases_sha256', root / 'runs/joint_spatial_feedback_305_acquired_train_001/cases.jsonl')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == config[name], name
assert config['candidate'] == 305 and config['local_optimizer_batches'] == 400
assert config['readouts'] == [0, 200, 400]
assert config['matched_arms'] == {'control_old_loss_weight': 0., 'intervention_old_loss_weight': .3}
assert config['old_heldout_donors_have_no_gradients'] is True
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is summary['calibrated'] is False
heldout = {domain: set(config['datasets'][domain]['fully_heldout_donors'])
           for domain in ('coronal', 'sagittal')}
assert all(len(ids) == 20 for ids in heldout.values())
assert config['datasets']['coronal']['train_rows'] == 1280
assert config['datasets']['sagittal']['train_rows'] == 653

readouts = {}
for arm in ('control', 'intervention'):
    folder = run / arm
    arm_receipt = json.loads((folder / 'completed.json').read_text(encoding='utf8'))
    assert arm_receipt['arm'] == arm and arm_receipt['updates'] == 400
    for name, expected in arm_receipt['output_sha256'].items():
        with (folder / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, (arm, name)
    draws = [json.loads(line) for line in (folder / 'draws.jsonl').open(encoding='utf8')]
    accepted = [row for row in draws if row['kind'] == 'synthetic' and row['accepted']]
    old = [row for row in draws if row['kind'] == 'old']
    lupori = [row for row in draws if row['kind'] == 'lupori']
    sites = [row for row in draws if row['kind'] == 'site_indices']
    assert len(accepted) == 800 and len(old) == len(lupori) == len(sites) == 400
    assert len({row['provenance']['physical_section_id'] for row in accepted}) == 800
    assert all(row['animal_id'] in ('CC2B', 'CC4B', 'CC4A') for row in lupori)
    assert all(row['donor_id'] not in heldout[row['domain']] for row in old)
    assert all(row['domain'] == ('sagittal' if row['step'] % 2 else 'coronal') for row in old)
    assert [row['step'] for row in old] == list(range(1, 401))
    train = [json.loads(line) for line in (folder / 'training.jsonl').open(encoding='utf8')]
    assert len(train) == 400
    for step, row in enumerate(train, 1):
        assert row['step'] == step and row['synthetic_presentations'] == 2 * step
        assert row['lupori_presentations'] == row['old_presentations'] == step
        assert all(math.isfinite(row[key]) for key in (
            'synthetic_loss', 'lupori_weak_loss', 'old_weak_loss', 'loss', 'gradient_norm'))
        expected = row['synthetic_loss'] + row['lupori_weak_loss'] + (
            .3 * row['old_weak_loss'] if arm == 'intervention' else 0)
        assert math.isclose(row['loss'], expected, rel_tol=1e-5, abs_tol=1e-5)
    entries = [json.loads(line) for line in (folder / 'readouts.jsonl').open(encoding='utf8')]
    assert [entry['step'] for entry in entries] == [0, 200, 400]
    for entry in entries:
        rows, metrics = entry['rows'], entry['metrics']
        assert len(rows) == 52 + 64 + 98
        assert sum(row['cohort'] == 'guard' for row in rows) == 52
        assert sum(row['cohort'] == 'synthetic' for row in rows) == 64
        for cohort in ('guard', 'synthetic'):
            group = [row for row in rows if row['cohort'] == cohort]
            assert math.isclose(metrics[cohort]['mean_first_gap_mm'], statistics.fmean(
                row['first_five_or_site_gap_mm'] for row in group), abs_tol=1e-10)
            assert metrics[cohort]['top16_capture_1p5mm'] == sum(
                row['top16_oracle_gap_mm'] <= 1.5 for row in group)
        for domain, count in (('coronal', 58), ('sagittal', 40)):
            group = [row for row in rows if row['cohort'] == 'old_' + domain]
            assert len(group) == count and len({row['donor_id'] for row in group}) == count
            for panel in ('heldout', 'all'):
                subset = [row for row in group if panel == 'all' or row['fully_heldout']]
                value = metrics[f'old_{domain}_{panel}']
                assert len(subset) == value['donors'] == (20 if panel == 'heldout' else count)
                assert value['first_axis_correct'] == sum(row['first_axis_correct'] for row in subset)
                assert value['top16_contains_weak_axis'] == sum(
                    row['top16_contains_weak_axis'] for row in subset)
                assert all((row['donor_id'] in heldout[domain]) == row['fully_heldout'] for row in group)
    readouts[arm] = {str(entry['step']): entry['metrics'] for entry in entries}
    assert arm_receipt['readouts'] == readouts[arm]

with (run / 'control/draws.jsonl').open('rb') as stream:
    matched = hashlib.file_digest(stream, 'sha256').hexdigest()
with (run / 'intervention/draws.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == matched
assert summary['matched_draws_sha256'] == matched and summary['readouts'] == readouts
assert readouts['control']['0'] == readouts['intervention']['0']
c, a = readouts['control']['400'], readouts['intervention']['400']
gate = (a['old_sagittal_heldout']['first_axis_correct'] >= c['old_sagittal_heldout']['first_axis_correct'] + 3
        and a['old_sagittal_heldout']['first_axis_correct'] >= 15
        and a['guard']['mean_first_gap_mm'] <= c['guard']['mean_first_gap_mm'] + .2
        and a['synthetic']['mean_first_gap_mm'] <= c['synthetic']['mean_first_gap_mm'] + .2
        and a['synthetic']['top16_capture_1p5mm'] >= c['synthetic']['top16_capture_1p5mm'] - 5
        and a['old_coronal_heldout']['first_axis_correct'] >= c['old_coronal_heldout']['first_axis_correct'] - 1)
assert summary['step_400_gate_pass'] == gate
start_c = torch.load(run / 'control/pose_local_00000.pt', map_location='cpu', weights_only=False)
start_a = torch.load(run / 'intervention/pose_local_00000.pt', map_location='cpu', weights_only=False)
assert start_c['local_step'] == start_a['local_step'] == 0
assert all(torch.equal(value, start_a['model'][key]) for key, value in start_c['model'].items())
assert start_c['optimizer']['param_groups'] == start_a['optimizer']['param_groups']
assert start_c['optimizer']['state'].keys() == start_a['optimizer']['state'].keys()
for key, state in start_c['optimizer']['state'].items():
    other = start_a['optimizer']['state'][key]
    assert state.keys() == other.keys()
    assert all(torch.equal(value, other[name]) if torch.is_tensor(value)
               else value == other[name] for name, value in state.items())
print(json.dumps({'verified': True, 'gate_pass': gate,
                  'control_step_400': c, 'intervention_step_400': a}, allow_nan=False))
