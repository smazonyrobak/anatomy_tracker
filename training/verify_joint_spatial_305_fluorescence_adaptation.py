"""Independent CPU-only audit, to run after 305 fluorescence adaptation exits."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import torch


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/joint_spatial_305_fluorescence_adaptation_001'
SOURCE = Path(__file__).resolve().with_name('train_joint_spatial_305_fluorescence_adaptation.py')
PROTOCOL = SOURCE.parents[1] / 'docs/publication/JOINT_SPATIAL_305_FLUORESCENCE_ADAPTATION_PROTOCOL_20261014.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PARENT = ROOT / 'runs/joint_spatial_feedback_305_pose_002/pose_step_26000.pt'
PANEL = PARENT.parent / 'inner_panel.pt'
STEPS = (0, 200, 400, 600)
TRAIN = ('CC2B', 'CC4B', 'CC4A')
GUARD = 'AL1A'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


# The trainer writes this only after both arms and its final summary are frozen.
assert (RUN / 'completed.json').is_file(), 'Training is active or incomplete; do not audit'
receipt = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
summary = json.loads((RUN / 'summary.json').read_text(encoding='utf8'))
assert receipt['summary'] == summary
assert set(receipt['output_sha256']) == {
    'config.json', 'summary.json', 'control/completed.json', 'adapt/completed.json'}
assert all(digest(RUN / name) == value for name, value in receipt['output_sha256'].items())
assert {path.name for path in RUN.iterdir()} == {
    'config.json', 'summary.json', 'completed.json', 'control', 'adapt'}
assert config['candidate'] == 305 and config['seed'] == 20261014305
assert config['optimizer_batches'] == 600 and config['readouts'] == list(STEPS)
assert config['synthetic_per_batch'] == 2 and config['real_per_batch'] == 1
assert config['synthetic_train_bases'] == list(range(56))
assert config['synthetic_inner_bases'] == list(range(56, 64))
assert tuple(config['gradient_animals']) == TRAIN and config['inner_guard_animal'] == GUARD
assert config['gradient_real_sections'] == 188 and config['guard_sections'] == 52
assert config['matched_arms'] == {
    'control_real_loss_weight': 0., 'adapt_real_loss_weight': 1.}
assert config['no_external_weights_features_or_pseudolabels'] is True
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert summary['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is summary['calibrated'] is False
assert config['source_sha256'] == digest(SOURCE)
assert config['protocol_sha256'] == digest(PROTOCOL)
assert config['split_sha256'] == digest(SPLIT)
assert config['parent_checkpoint_sha256'] == digest(PARENT)
assert config['parent_completion_sha256'] == digest(PARENT.parent / 'completed.json')
assert config['panel_sha256'] == digest(PANEL)
parent_receipt = json.loads((PARENT.parent / 'completed.json').read_text(encoding='utf8'))
assert parent_receipt['summary']['selected_step'] == 26000
assert parent_receipt['output_sha256'][PARENT.name] == config['parent_checkpoint_sha256']
assert parent_receipt['output_sha256'][PANEL.name] == config['panel_sha256']
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert tuple(split['train']) == (GUARD, *TRAIN)
assert config['source_doi'] == split['source_doi']
assert config['archive_bytes'] == split['archive_bytes'] == ARCHIVE.stat().st_size
assert config['archive_md5_from_verified_source'] == split['archive_md5']
# Source 185 already authenticated the 48.7-GB archive; section hashes bind this run.
panel = torch.load(PANEL, map_location='cpu', weights_only=False)
assert len(panel) == 64 and {row['base'] for row in panel} == set(range(56, 64))
panel_ids = [row['physical_section_id'] for row in panel]
assert len(set(panel_ids)) == 64

arm_receipts = {}
readouts = {}
final_rows = {}
guard_ids = None
for arm in ('control', 'adapt'):
    folder = RUN / arm
    arm_receipt = json.loads((folder / 'completed.json').read_text(encoding='utf8'))
    expected = {'draws.jsonl', 'training.jsonl', 'readouts.jsonl'} | {
        f'pose_step_{step:05d}.pt' for step in STEPS}
    assert arm_receipt['arm'] == arm and arm_receipt['updates'] == 600
    assert arm_receipt['calibrated'] is False
    assert set(arm_receipt['output_sha256']) == expected
    assert {path.name for path in folder.iterdir()} == expected | {'completed.json'}
    assert all(digest(folder / name) == value
               for name, value in arm_receipt['output_sha256'].items())
    arm_receipts[arm] = arm_receipt

    if arm == 'control':
        ids = [[] for _ in range(601)]
        bases = [[] for _ in range(601)]
        epochs = [None] * 601
        next_step, next_slot, attempts = 1, 0, 0
        for line in (folder / 'draws.jsonl').open(encoding='utf8'):
            row = json.loads(line)
            assert row['step'] == next_step
            if row['kind'] == 'synthetic':
                attempts += 1
                assert row['slot'] == next_slot and row['attempt'] == attempts
                assert 0 <= row['base'] < 56 and 0 <= row['variant'] < 64
                provenance = row['provenance']
                virtual = row['base'] * 64 + row['variant']
                assert provenance['split'] == 'train' and provenance['virtual_index'] == virtual
                assert provenance['physical_section_id'] == \
                    f"stream-v7-seed-{row['seed']}-row-0-virtual-{virtual}"
                assert provenance['postartifact_gauge_v1']['pose_supervision_eligible'] == \
                    row['accepted']
                if row['accepted']:
                    ids[next_step].append(provenance['physical_section_id'])
                    bases[next_step].append(row['base'])
                    next_slot += 1
            else:
                assert row['kind'] == 'real' and next_slot == 2 and row['slot'] == 2
                assert row['animal_id'] in TRAIN and row['animal_id'] != GUARD
                assert row['specimen_id'] == row['animal_id']
                assert row['archive_member'].startswith(row['animal_id'] + '/')
                assert len(row['source_image_sha256']) == 64
                epochs[next_step] = row['epoch']
                next_step, next_slot = next_step + 1, 0
        assert (next_step, next_slot) == (601, 0)
        assert all(len(ids[step]) == 2 for step in range(1, 601))

    training_rows = [json.loads(line) for line in (folder / 'training.jsonl').open(encoding='utf8')]
    assert len(training_rows) == 600
    for step, row in enumerate(training_rows, 1):
        assert row['step'] == step
        assert row['synthetic_presentations'] == 2 * step
        assert row['real_presentations'] == step
        assert all(math.isfinite(row[name]) for name in (
            'synthetic_loss', 'real_weak_loss', 'loss', 'gradient_norm', 'elapsed_seconds'))
        expected_loss = row['synthetic_loss'] + (
            row['real_weak_loss'] if arm == 'adapt' else 0)
        assert math.isclose(row['loss'], expected_loss, rel_tol=1e-5, abs_tol=1e-5)
        if arm == 'control':
            assert row['real_epoch'] == epochs[step]
        else:
            control = control_training[step - 1]
            assert row['real_epoch'] == control['real_epoch']
    if arm == 'control':
        control_training = training_rows

    entries = [json.loads(line) for line in (folder / 'readouts.jsonl').open(encoding='utf8')]
    assert [entry['step'] for entry in entries] == list(STEPS)
    for entry in entries:
        rows, recorded = entry['rows'], entry['metrics']
        assert len(rows) == 116
        assert [row['cohort'] for row in rows] == ['guard'] * 52 + ['synthetic'] * 64
        guard, synthetic = rows[:52], rows[52:]
        current_guard_ids = [(row['animal_id'], row['source_section_id']) for row in guard]
        assert len(set(current_guard_ids)) == 52
        assert all(row['animal_id'] == row['specimen_id'] == GUARD and
                   row['archive_member'].startswith(GUARD + '/') for row in guard)
        if guard_ids is None:
            guard_ids = current_guard_ids
        assert current_guard_ids == guard_ids
        assert [row['physical_section_id'] for row in synthetic] == panel_ids
        assert [row['base'] for row in synthetic] == [item['base'] for item in panel]
        for row, item in zip(synthetic, panel):
            assert all(row[name] == item[name] for name in (
                'obliquity_deg', 'visible_fraction', 'exterior_style'))
        for cohort, group in (('guard', guard), ('synthetic', synthetic)):
            for row in group:
                assert all(math.isfinite(row[name]) and row[name] >= 0 for name in (
                    'first_five_or_site_gap_mm', 'top16_oracle_gap_mm',
                    'first_unsigned_normal_gap_deg'))
            actual = recorded[cohort]
            assert actual['n'] == len(group)
            assert close(actual['mean_first_gap_mm'], statistics.fmean(
                row['first_five_or_site_gap_mm'] for row in group))
            assert close(actual['mean_top16_oracle_gap_mm'], statistics.fmean(
                row['top16_oracle_gap_mm'] for row in group))
            assert actual['top16_capture_1p5mm'] == sum(
                row['top16_oracle_gap_mm'] <= 1.5 for row in group)
            assert close(actual['median_first_unsigned_normal_gap_deg'], statistics.median(
                row['first_unsigned_normal_gap_deg'] for row in group))
        assert arm_receipt['readouts'][str(entry['step'])] == recorded
    readouts[arm] = {str(entry['step']): entry['metrics'] for entry in entries}
    final_rows[arm] = entries[-1]['rows']

assert digest(RUN / 'control/draws.jsonl') == digest(RUN / 'adapt/draws.jsonl')
assert summary['matched_draws_sha256'] == digest(RUN / 'control/draws.jsonl')
assert summary['readouts'] == readouts
c, a = readouts['control']['600'], readouts['adapt']['600']
gain = c['guard']['mean_first_gap_mm'] - a['guard']['mean_first_gap_mm']
relative = gain / c['guard']['mean_first_gap_mm']
gate = (gain >= 1.0 and relative >= .25
    and a['synthetic']['mean_first_gap_mm'] <= c['synthetic']['mean_first_gap_mm'] + .2
    and a['synthetic']['top16_capture_1p5mm'] >= c['synthetic']['top16_capture_1p5mm'] - 5)
assert close(summary['guard_absolute_gain_mm'], gain)
assert close(summary['guard_relative_gain'], relative)
assert summary['step_600_gate_pass'] == gate

control_start = torch.load(RUN / 'control/pose_step_00000.pt',
    map_location='cpu', weights_only=False)
adapt_start = torch.load(RUN / 'adapt/pose_step_00000.pt',
    map_location='cpu', weights_only=False)
assert control_start['step'] == adapt_start['step'] == 0
assert control_start['arm'] == 'control' and adapt_start['arm'] == 'adapt'
assert json.loads(json.dumps(control_start['config'])) == \
    json.loads(json.dumps(adapt_start['config'])) == config
assert control_start['calibrated'] is adapt_start['calibrated'] is False
assert control_start['model'].keys() == adapt_start['model'].keys()
assert all(torch.equal(value, adapt_start['model'][name])
    for name, value in control_start['model'].items())
strata = {}
for name, keep in (
        ('obliquity_ge45', lambda row: row['obliquity_deg'] >= 45),
        ('visible_lt0p1', lambda row: row['visible_fraction'] < .1),
        ('gray_background', lambda row: row['exterior_style'] == 'gray')):
    strata[name] = {}
    for arm in ('control', 'adapt'):
        group = [row for row in final_rows[arm] if row['cohort'] == 'synthetic' and keep(row)]
        strata[name][arm] = {'n': len(group),
            'mean_first_gap_mm': statistics.fmean(row['first_five_or_site_gap_mm'] for row in group),
            'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5 for row in group)}
guard_tail = {}
for arm in ('control', 'adapt'):
    values = sorted(row['first_five_or_site_gap_mm'] for row in final_rows[arm]
                    if row['cohort'] == 'guard')
    guard_tail[arm] = {'median_mm': statistics.median(values),
        'p90_mm': values[math.ceil(.9 * len(values)) - 1],
        'first_le_1p5mm': sum(value <= 1.5 for value in values),
        'over_5mm': sum(value > 5 for value in values)}
print(json.dumps({'verified': True, 'matched_draws': True,
    'identical_step_zero_weights': True, 'step_600_gate_pass': gate,
    'guard_absolute_gain_mm': gain, 'guard_relative_gain': relative,
    'strata': strata, 'guard_tail': guard_tail}, allow_nan=False))
