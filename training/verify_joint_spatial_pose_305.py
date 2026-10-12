"""Independent CPU-only audit of the completed 305 pose continuation."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090


ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/joint_spatial_feedback_305_pose_002'
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/joint_spatial_feedback_305_pose_001'
STEPS = (0, 2000, 8000, 16000, 26000)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# Completion is written last by the trainer. Never inspect a partially written run.
assert (RUN / 'completed.json').is_file(), 'Pose run has not completed; do not audit it yet'
receipt = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
expected = {'config.json', 'inner_panel.pt', 'panel_draws.jsonl',
    'training_draws.jsonl', 'training.jsonl', 'readouts.jsonl', 'summary.json'}
expected |= {f'pose_step_{step:05d}.pt' for step in STEPS}
assert set(receipt['output_sha256']) == expected
assert {path.name for path in RUN.iterdir()} == expected | {'completed.json'}
assert all(sha(RUN / name) == digest for name, digest in receipt['output_sha256'].items())
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
summary = json.loads((RUN / 'summary.json').read_text(encoding='utf8'))
assert receipt['summary'] == summary and receipt['calibrated'] is False
assert config['candidate'] == summary['candidate'] == 305
assert config['stage'] == summary['stage'] == 'scratch_pose'
assert config['seed'] == 20261012305 and config['side'] == 192
assert config['batch_size'] == 4 and config['optimizer_batches'] == 26000
assert config['readouts'] == list(STEPS)
assert config['gradient_base_indices'] == list(range(56))
assert config['inner_base_indices'] == list(range(56, 64))
assert config['model']['spatial_feedback'] is True
assert config['no_external_weights_features_or_pseudolabels'] is True
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert summary['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is summary['calibrated'] is False
assert config['protocol_sha256'] == sha(PROTOCOL)
assert all(sha(SOURCE / name) == digest for name, digest in config['source_sha256'].items())
assert config['resume_parent'] == str(PARENT) and config['resume_step'] == 2000
assert all(sha(PARENT / name) == digest for name, digest in
           config['resume_parent_sha256'].items())
parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf8'))
assert parent_config['protocol_sha256'] == config['protocol_sha256']
assert parent_config['source_sha256']['arbitrary_plane_one_shot_model.py'] == \
       config['source_sha256']['arbitrary_plane_one_shot_model.py']
for name in ('panel_draws.jsonl', 'inner_panel.pt', 'pose_step_00000.pt',
             'pose_step_02000.pt'):
    assert sha(RUN / name) == config['resume_parent_sha256'][name]
for name in ('training_draws.jsonl', 'training.jsonl', 'readouts.jsonl'):
    with (PARENT / name).open(encoding='utf8') as previous, \
            (RUN / name).open(encoding='utf8') as current:
        for line in previous:
            if json.loads(line)['step'] > 2000:
                break
            assert line == current.readline()

panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64
slots = [(base, slot) for base in range(56, 64) for slot in range(8)]
assert [(row['base'], row['slot']) for row in panel] == slots
panel_ids = [row['physical_section_id'] for row in panel]
assert len(set(panel_ids)) == 64
normals = full_frame_state_to_components(torch.stack([row['state'] for row in panel]))[1][:, :, 2]
axes = [int(torch.argmax(normal.abs())) for normal in normals]
axis_names = ('AP/coronal', 'DV/horizontal', 'ML/sagittal')
fitted_rigid_errors = [float((rigid_points_090(row['state'][None],
    row['reflection'][None], row['chart'][None], (192, 192))[0] - row['truth'])
    .norm(dim=-1).mean() / 1000) for row in panel]
baseline_quantiles = torch.quantile(torch.tensor(fitted_rigid_errors),
    torch.tensor([0., .25, .5, .75, .9, .95, 1.])).tolist()
assert all(row['provenance']['physical_section_id'] == row['physical_section_id']
           for row in panel)
assert all(row['provenance']['postartifact_gauge_v1']['fit_applied'] for row in panel)
assert all(0 <= row['visible_fraction'] <= 1 and 0 <= row['damage_fraction'] <= 1
           for row in panel)

panel_accepted, slot_index = [], 0
for index, line in enumerate((RUN / 'panel_draws.jsonl').open(encoding='utf8'), 1):
    row = json.loads(line)
    assert slot_index < 64 and (row['base'], row['slot']) == slots[slot_index]
    assert 0 <= row['variant'] < 64 and row['seed'] == config['seed'] + 1_000_000_000 + index
    provenance = row['provenance']
    virtual = row['base'] * 64 + row['variant']
    assert provenance['virtual_index'] == virtual and provenance['split'] == 'train'
    assert provenance['physical_section_id'] == \
        f"stream-v7-seed-{row['seed']}-row-0-virtual-{virtual}"
    assert provenance['postartifact_gauge_v1']['code_sha256'] == \
           parent_config['source_sha256']['arbitrary_plane_postartifact_gauge_v1.py']
    assert provenance['one_shot_slide_artifacts_v9']['code_sha256'] == \
           config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v9.py']
    if row['accepted']:
        assert provenance['postartifact_gauge_v1']['fit_applied']
        panel_accepted.append(provenance['physical_section_id'])
        slot_index += 1
assert slot_index == 64 and panel_accepted == panel_ids

draw_counts = [0] * 26001
accept_counts = [0] * 26001
previous_step = 1
for attempt, line in enumerate((RUN / 'training_draws.jsonl').open(encoding='utf8'), 1):
    row = json.loads(line)
    step = row['step']
    assert previous_step <= step <= 26000 and row['attempt'] == attempt
    previous_step = step
    assert 0 <= row['base'] < 56 and 0 <= row['variant'] < 64
    assert row['seed'] == config['seed'] + 2_000_000_000 + attempt
    provenance = row['provenance']
    virtual = row['base'] * 64 + row['variant']
    assert provenance['virtual_index'] == virtual and provenance['split'] == 'train'
    assert provenance['physical_section_id'] == \
        f"stream-v7-seed-{row['seed']}-row-0-virtual-{virtual}"
    gauge = provenance['postartifact_gauge_v1']
    expected_gauge = (parent_config if step <= 2000 else config)[
        'source_sha256']['arbitrary_plane_postartifact_gauge_v1.py']
    assert gauge['code_sha256'] == expected_gauge
    assert provenance['one_shot_slide_artifacts_v9']['code_sha256'] == \
           config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v9.py']
    if step > 2000:
        assert gauge['pose_supervision_eligible'] == row['accepted']
    if row['accepted']:
        assert gauge['fit_applied']
        accept_counts[step] += 1
    draw_counts[step] += 1

training_rows = [json.loads(line) for line in (RUN / 'training.jsonl').open(encoding='utf8')]
assert len(training_rows) == 26000
attempted = accepted = 0
for step, row in enumerate(training_rows, 1):
    assert row['step'] == step and accept_counts[step] == 4
    attempted += draw_counts[step]
    accepted += accept_counts[step]
    assert row['attempted'] == attempted and row['accepted'] == accepted == 4 * step
    assert all(math.isfinite(row[name]) for name in
               ('direct_loss', 'reliability_loss', 'loss', 'gradient_norm'))
    assert (row['teacher_error_mm'] is not None) == (step % 4 == 0)
    if step % 4 == 0:
        assert math.isfinite(row['teacher_error_mm'])
assert summary['attempted'] == attempted and summary['accepted'] == accepted

readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open(encoding='utf8')]
assert [row['step'] for row in readouts] == list(STEPS)
diagnostics = {}
for readout in readouts:
    step, rows, recorded = readout['step'], readout['rows'], readout['summary']
    assert len(rows) == 64 and [row['physical_section_id'] for row in rows] == panel_ids
    assert [(row['base'], row['slot']) for row in rows] == slots
    assert all(len(row['branch_ids']) == len(row['rigid_site_error_mm']) == 16
        and len(set(row['branch_ids'])) == 16
        and all(0 <= branch < 32 for branch in row['branch_ids'])
        and all(math.isfinite(error) and error >= 0 for error in row['rigid_site_error_mm'])
        and math.isfinite(row['source_reliability_loss']) for row in rows)
    for row, original in zip(rows, panel):
        assert all(row[key] == original[key] for key in
            ('obliquity_deg', 'visible_fraction', 'damage_fraction', 'exterior_style'))
    oracle = [min(row['rigid_site_error_mm']) for row in rows]
    first = [row['rigid_site_error_mm'][0] for row in rows]
    assert recorded['capture_1p5mm'] == sum(error <= 1.5 for error in oracle)
    assert math.isclose(recorded['oracle_top16_mean_error_mm'], statistics.fmean(oracle),
                        rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(recorded['first_choice_mean_error_mm'], statistics.fmean(first),
                        rel_tol=1e-12, abs_tol=1e-12)
    strata = {'obliquity_ge45': [row for row in rows if row['obliquity_deg'] >= 45],
        'visible_lt0p1': [row for row in rows if row['visible_fraction'] < .1],
        'damage_ge0p02': [row for row in rows if row['damage_fraction'] >= .02]}
    strata.update({style: [row for row in rows if row['exterior_style'] == style]
                   for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
    assert set(recorded['strata']) == set(strata)
    for name, group in strata.items():
        actual = recorded['strata'][name]
        assert actual['n'] == len(group)
        assert actual['capture_1p5mm'] == sum(
            min(row['rigid_site_error_mm']) <= 1.5 for row in group)
        for field, values in (('oracle_top16_mean_error_mm', [
            min(row['rigid_site_error_mm']) for row in group]),
            ('first_choice_mean_error_mm', [row['rigid_site_error_mm'][0] for row in group])):
            assert (actual[field] is None if not group else math.isclose(
                actual[field], statistics.fmean(values), rel_tol=1e-12, abs_tol=1e-12))
    assert summary['readouts'][str(step)] == recorded
    diagnostics[str(step)] = {'top8_capture_1p5mm': sum(
        min(row['rigid_site_error_mm'][:8]) <= 1.5 for row in rows),
        'top16_capture_1p5mm': recorded['capture_1p5mm'],
        'axis_dominant': {axis_names[axis]: {'n': sum(a == axis for a in axes),
            'top8_capture_1p5mm': sum(axes[i] == axis and
                min(rows[i]['rigid_site_error_mm'][:8]) <= 1.5 for i in range(64)),
            'top16_capture_1p5mm': sum(axes[i] == axis and
                oracle[i] <= 1.5 for i in range(64)),
            'oracle_top16_mean_error_mm': statistics.fmean(
                oracle[i] for i in range(64) if axes[i] == axis)}
            for axis in range(3) if axis in axes}}

selected = max(STEPS, key=lambda step: (
    int(summary['readouts'][str(step)]['capture_1p5mm'] >= 45 and
        summary['readouts'][str(step)]['oracle_top16_mean_error_mm'] <= 1.5),
    summary['readouts'][str(step)]['capture_1p5mm'],
    -summary['readouts'][str(step)]['oracle_top16_mean_error_mm'],
    -summary['readouts'][str(step)]['first_choice_mean_error_mm']))
assert summary['selected_step'] == selected
assert summary['joint_stage_capture_gate'] == (
    summary['readouts'][str(selected)]['capture_1p5mm'] >= 45 and
    summary['readouts'][str(selected)]['oracle_top16_mean_error_mm'] <= 1.5)
print(json.dumps({'verified': True, 'selected_step': selected,
    'joint_stage_capture_gate': summary['joint_stage_capture_gate'],
    'attempted': attempted, 'accepted': accepted,
    'true_fitted_state_rigid_baseline': {
        'mean_error_mm': statistics.fmean(fitted_rigid_errors),
        'quantiles_0_25_50_75_90_95_100_mm': baseline_quantiles,
        'cases_gt1p5mm': [{'physical_section_id': panel[i]['physical_section_id'],
            'base': panel[i]['base'], 'slot': panel[i]['slot'],
            'error_mm': error, 'visible_fraction': panel[i]['visible_fraction'],
            'damage_fraction': panel[i]['damage_fraction']}
            for i, error in enumerate(fitted_rigid_errors) if error > 1.5]},
    'readout_diagnostics': diagnostics}, allow_nan=False))
