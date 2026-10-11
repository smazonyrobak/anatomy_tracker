"""CPU-only independent provenance and arithmetic audit of the frozen 160 readout."""

import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root = Path('I:/AnatomyTracker')
train = root / 'runs/mixed_pose_recovery_160_train_001'
out = root / 'runs/mixed_pose_recovery_160_dev_eval_001'
source = Path(__file__).resolve().parent
panels = {'v4': root / 'data/coarse_atlas_pose_150_fresh_dev_panel',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
coronal = root / 'data/joint_v7_allen_fullcanvas_192_001'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008'
sagittal_train = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
arms = ('0', '326', '653')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((out / 'completed.json').read_text())
config = json.loads((out / 'config.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
synthetic = [json.loads(line) for line in (out / 'synthetic_rows.jsonl').open()]
mapping = [json.loads(line) for line in (out / 'map_rows.jsonl').open()]
real = [json.loads(line) for line in (out / 'real_weak_rows.jsonl').open()]
train_done = json.loads((train / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
draws = [json.loads(line) for line in (train / 'draws.jsonl').open()]
curve = [json.loads(line) for line in (train / 'training.jsonl').open()]
assert train_done['updates'] == len(curve) == 653
assert [row['update'] for row in curve] == list(range(1, 654))
assert all(row['synthetic_presentations'] == 3 * row['update'] for row in curve)
assert all(math.isfinite(row[key]) for row in curve for key in (
    'synthetic_loss', 'real_weak_loss', 'loss', 'gradient_norm'))
synthetic_draws = [row for row in draws if row['kind'] == 'synthetic_train']
used = [row for row in synthetic_draws if row['used']]
coronal_draws = [row for row in draws if row['kind'] == 'coronal_weak_train']
sagittal_draws = [row for row in draws if row['kind'] == 'sagittal_weak_train']
assert len(synthetic_draws) == train_done['synthetic_draw_attempts'] == 2471
assert len(used) == train_done['synthetic_presentations'] == 1959
assert len({row['physical_section_id'] for row in used}) == \
    train_done['unique_synthetic_physical_sections'] == 1959
assert len({tuple(row['seed_prefix']) for row in synthetic_draws}) == len(synthetic_draws)
assert [row['draw_attempt'] for row in synthetic_draws] == list(range(1, 2472))
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train'
           for row in synthetic_draws)
assert len(coronal_draws) == len(sagittal_draws) == 653
assert len({row['animal_id'] for row in coronal_draws}) == \
    train_done['distinct_coronal_train_donors'] == 653
assert len({row['section_id'] for row in sagittal_draws}) == \
    train_done['distinct_sagittal_train_sections'] == 653
for step, row in enumerate(curve, 1):
    selected = [item for item in used if item['update'] == step]
    assert [item['slot'] for item in selected] == [0, 1, 2]
    assert [item['physical_section_id'] for item in selected] == row['synthetic_section_ids']
    assert [item['geometry'] for item in selected] == ['v7', 'v7', 'full_box']
    assert [item['appearance_version'] for item in selected] == ['v3', 'v4', 'v4']
    assert coronal_draws[step - 1]['update'] == sagittal_draws[step - 1]['update'] == step
    assert all(coronal_draws[step - 1][key] == row['coronal_identity'][key]
               for key in row['coronal_identity'])
    assert all(sagittal_draws[step - 1][key] == row['sagittal_identity'][key]
               for key in row['sagittal_identity'])
assert sha(source / 'evaluate_mixed_pose_recovery_160.py') == \
    done['evaluator_sha256'] == config['evaluator_sha256']
assert sha(train / 'completed.json') == done['training_completion_sha256'] == \
    config['training_completion_sha256']
assert sha(train / 'config.json') == train_done['config_sha256'] == config['training_config_sha256']
assert sha(train / 'draws.jsonl') == train_done['draws_sha256'] == config['training_draws_sha256']
assert sha(train / 'training.jsonl') == train_done['training_sha256'] == config['training_curve_sha256']
assert sha(source.parent / 'docs/publication/MIXED_POSE_RECOVERY_160_PROTOCOL_20261011.md') == \
    train_done['protocol_sha256'] == train_config['protocol_sha256'] == config['protocol_sha256']
assert all(sha(source / name) == digest for name, digest in train_config['source_sha256'].items())
assert all(sha(out / name) == digest for name, digest in done['output_sha256'].items())
assert all(sha(train / f'joint_step_{int(arm):05d}.pt') ==
           train_done['checkpoint_sha256'][arm] == config['checkpoint_sha256'][arm]
           for arm in arms)
assert all(not row.get(key, False) for row in (done, config, summary, train_done, train_config)
           for key in ('calibrated', 'public_benchmark_used', 'final_animals_used',
                       'expert_real_truth_used', 'external_pretrained_weights_used'))
assert (len(synthetic), len(mapping), len(real)) == \
    (done['synthetic_rows'], done['map_rows'], done['real_weak_rows'])

records = {}
for cohort, folder in panels.items():
    receipt = json.loads((folder / 'completed.json').read_text())
    panel = [json.loads(line) for line in (folder / 'records.jsonl').open()]
    assert sha(folder / 'completed.json') == config['panel_completion_sha256'][cohort]
    assert sha(folder / 'protocol.json') == receipt['protocol_sha256']
    assert sha(folder / 'records.jsonl') == receipt['records_sha256']
    assert len(panel) == receipt['physical_sections'] == 256
    assert sum(r['eligible'] for r in panel) == receipt['eligible']
    assert config['panel_sections'][cohort] == {'recorded': 256,
        'eligible': receipt['eligible'], 'ineligible': 256 - receipt['eligible']}
    assert len({r['synthetic_subject_plan_id'] for r in panel}) == 8
    assert all(r['provenance']['split'] == 'development' for r in panel)
    assert all(sha(folder / r['file']) == r['sha256'] for r in panel)
    records[cohort] = {r['section_id']: r for r in panel if r['eligible']}
assert not set(records['v4']) & set(records['v3'])
assert len(records['v4']) + len(records['v3']) == len(synthetic) // 3
expected_synthetic = {(arm, cohort, section) for arm in arms for cohort in panels
                      for section in records[cohort]}
assert {(r['arm'], r['cohort'], r['section_id']) for r in synthetic} == expected_synthetic
assert len(expected_synthetic) == len(synthetic)

map_ids = set(config['map_subset_v4_section_ids'])
assert len(map_ids) == 64 and map_ids <= set(records['v4'])
for plan in {r['synthetic_subject_plan_id'] for r in records['v4'].values()}:
    ordered = sorted((r for r in records['v4'].values()
                      if r['synthetic_subject_plan_id'] == plan), key=lambda r: r['section_id'])
    expected = {ordered[int(i)]['section_id'] for i in
                [j * (len(ordered) - 1) / 7 for j in range(8)]}
    # np.linspace's integer truncation is equivalent for these eight positions.
    assert map_ids & {r['section_id'] for r in ordered} == expected
assert len(mapping) == 3 * 64 * 2
assert {(r['arm'], r['role'], r['section_id']) for r in mapping} == {
    (arm, role, section) for arm in arms for role in ('exact', 'baseline_near')
    for section in map_ids}

for row in synthetic:
    record = records[row['cohort']][row['section_id']]
    assert row['physical_section_id'] == record['panel_physical_section_id']
    assert row['synthetic_subject_plan_id'] == record['synthetic_subject_plan_id']
    assert row['animal_id'] == record['animal_id']
    assert row['specimen_id'] == record['specimen_id']
    assert row['experiment_id'] == record['experiment_id']
    assert row['panel_file_sha256'] == record['sha256']
    assert row['appearance_mode'] == record['appearance_mode']
    assert row['valid_pixels'] == record['valid_pixels']
    assert math.isclose(row['valid_fraction'], record['valid_pixels'] / 256**2, abs_tol=1e-6)
    normal = record['plane_normal_ap_dv_ml']
    angle = math.degrees(math.acos(min(1, max(abs(v) for v in normal))))
    assert math.isclose(row['nearest_cardinal_angle_deg'], angle, abs_tol=1e-6)
    ids, errors = row['beam_branch_ids'], row['beam_rigid_mm']
    assert len(ids) == len(errors) == 16 and len(set(ids)) == 16
    assert all(0 <= x < 160 for x in ids) and all(math.isfinite(x) and x >= 0 for x in errors)
    assert row['top1_branch_id'] in ids
    assert math.isclose(row['top1_rigid_mm'], errors[ids.index(row['top1_branch_id'])], abs_tol=1e-6)
    assert math.isclose(row['top16_truth_best_rigid_mm'], min(errors), abs_tol=1e-6)
for row in mapping:
    record = records['v4'][row['section_id']]
    assert row['synthetic_subject_plan_id'] == record['synthetic_subject_plan_id']
    assert row['animal_id'] == record['animal_id']
    assert row['specimen_id'] == record['specimen_id']
    assert row['experiment_id'] == record['experiment_id']
    assert row['panel_file_sha256'] == record['sha256']
    assert 0 < row['valid_sites'] <= 4096
    assert math.isfinite(row['baseline_near_rigid_mm']) and row['baseline_near_rigid_mm'] >= 0
    assert math.isfinite(row['mapped_valid_mean_mm']) and row['mapped_valid_mean_mm'] >= 0
for section in map_ids:
    fixed = [r for r in mapping if r['section_id'] == section]
    assert len({r['baseline_near_rigid_mm'] for r in fixed}) == 1
    assert len({r['valid_sites'] for r in fixed}) == 1
    baseline_pose = next(r for r in synthetic if r['arm'] == '0' and
                         r['cohort'] == 'v4' and r['section_id'] == section)
    assert math.isclose(fixed[0]['baseline_near_rigid_mm'],
                        baseline_pose['top16_truth_best_rigid_mm'], abs_tol=1e-6)

coronal_done = json.loads((coronal / 'completed.json').read_text())
assert sha(coronal / 'completed.json') == config['coronal_completion_sha256']
assert all(sha(coronal / name) == coronal_done['output_sha256'][name]
           for name in ('images.npy', 'geometry.npz', 'records.jsonl'))
coronal_all = [json.loads(line) for line in (coronal / 'records.jsonl').open()]
coronal_dev = [r for r in coronal_all if r['training_split'] == 'development']
assert len(coronal_dev) == 64 and len({r['animal_id'] for r in coronal_dev}) == 6
assert not {r['animal_id'] for r in coronal_dev} & {
    r['animal_id'] for r in coronal_all if r['training_split'] == 'train'}
sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sha(sagittal / 'summary.json') == config['sagittal_summary_sha256']
assert all(sha(sagittal / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_dev = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
sagittal_train_records = [json.loads(line) for line in (sagittal_train / 'geometry.jsonl').open()]
assert len(sagittal_dev) == 158 and len({r['donor_id'] for r in sagittal_dev}) == 8
assert all(r['split'] == 'weak_dev2_source_available' for r in sagittal_dev)
assert not {r['donor_id'] for r in sagittal_dev} & {
    r['donor_id'] for r in sagittal_train_records}
real_records = {'coronal': {r['section_id']: r for r in coronal_dev},
                'sagittal': {r['section_id']: r for r in sagittal_dev}}
assert len(real) == 3 * (64 + 158)
assert {(r['arm'], r['family'], r['section_id']) for r in real} == {
    (arm, family, section) for arm in arms for family in real_records
    for section in real_records[family]}
for row in real:
    record = real_records[row['family']][row['section_id']]
    assert row['donor_id'] == record['animal_id' if row['family'] == 'coronal' else 'donor_id']
    assert row['specimen_id'] == record['specimen_id']
    assert row['experiment_id'] == record['experiment_id']
    assert row['array_row_index'] == record['array_row_index']
    assert row['label_role'] == 'inherited weak Allen affine, not expert truth'
    ids, errors = row['beam_branch_ids'], row['beam_weak_five_point_mm']
    assert len(ids) == len(errors) == 16 and len(set(ids)) == 16
    assert all(0 <= x < 160 for x in ids) and all(math.isfinite(x) and x >= 0 for x in errors)
    assert row['top1_branch_id'] in ids
    assert math.isclose(row['top1_weak_five_point_mm'],
                        errors[ids.index(row['top1_branch_id'])], abs_tol=1e-6)
    assert math.isclose(row['top16_truth_best_weak_five_point_mm'], min(errors), abs_tol=1e-6)
    assert math.isfinite(row['top1_weak_normal_deg']) and 0 <= row['top1_weak_normal_deg'] <= 90

for arm in arms:
    for cohort in panels:
        group = [r for r in synthetic if r['arm'] == arm and r['cohort'] == cohort]
        plans = {r['synthetic_subject_plan_id'] for r in group}
        saved = summary['synthetic'][f'{arm}/{cohort}']
        top1 = sum(sum(r['top1_rigid_mm'] for r in group if
            r['synthetic_subject_plan_id'] == plan) /
            sum(r['synthetic_subject_plan_id'] == plan for r in group) for plan in plans) / len(plans)
        near = sum(sum(r['top16_truth_best_rigid_mm'] <= 1.5 for r in group if
            r['synthetic_subject_plan_id'] == plan) /
            sum(r['synthetic_subject_plan_id'] == plan for r in group) for plan in plans) / len(plans)
        assert saved['sections'] == len(group) and saved['plans'] == len(plans) == 8
        assert math.isclose(saved['top1_plan_equal_mm'], top1, abs_tol=1e-6)
        assert math.isclose(saved['top16_near_plan_equal'], near, abs_tol=1e-6)
        assert set(saved['by_plan']) == plans
        for plan in plans:
            part = [r for r in group if r['synthetic_subject_plan_id'] == plan]
            item = saved['by_plan'][plan]
            assert item['sections'] == len(part)
            assert math.isclose(item['top1_mean_mm'],
                sum(r['top1_rigid_mm'] for r in part) / len(part), abs_tol=1e-6)
            assert math.isclose(item['top16_near'],
                sum(r['top16_truth_best_rigid_mm'] <= 1.5 for r in part) / len(part),
                abs_tol=1e-6)
        for name, field, bins in (
            ('by_mode', 'appearance_mode', ('raw', 'exact_black', 'imperfect_brush')),
            ('by_angle', 'nearest_cardinal_angle_deg', ((0, 15), (15, 30), (30, 45), (45, 55))),
            ('by_tissue_fraction', 'valid_fraction', ((0, .15), (.15, .30), (.30, .60), (.60, 1.01)))):
            for bin_value in bins:
                part = [r for r in group if (r[field] == bin_value if isinstance(bin_value, str)
                    else bin_value[0] <= r[field] < bin_value[1])]
                item = saved[name][str(bin_value)]
                assert item['sections'] == len(part)
                if part:
                    assert math.isclose(item['top1_mean_mm'],
                        sum(r['top1_rigid_mm'] for r in part) / len(part), abs_tol=1e-6)
                    assert math.isclose(item['top16_near'],
                        sum(r['top16_truth_best_rigid_mm'] <= 1.5 for r in part) / len(part),
                        abs_tol=1e-6)
                else:
                    assert item['top1_mean_mm'] is item['top16_near'] is None
    for role in ('exact', 'baseline_near'):
        part = [r for r in mapping if r['arm'] == arm and r['role'] == role and
                (role == 'exact' or r['baseline_near_rigid_mm'] <= 1.5)]
        plans = {r['synthetic_subject_plan_id'] for r in part}
        saved = summary['map'][f'{arm}/{role}']
        assert saved['sections'] == len(part) and saved['plans'] == len(plans)
        value = sum(sum(r['mapped_valid_mean_mm'] for r in part if
            r['synthetic_subject_plan_id'] == plan) /
            sum(r['synthetic_subject_plan_id'] == plan for r in part) for plan in plans) / len(plans) if plans else None
        assert (saved['plan_equal_valid_mean_mm'] is None and value is None) or math.isclose(
            saved['plan_equal_valid_mean_mm'], value, abs_tol=1e-6)
    for family in ('coronal', 'sagittal'):
        part = [r for r in real if r['arm'] == arm and r['family'] == family]
        donors = {r['donor_id'] for r in part}
        saved = summary['weak_real'][f'{arm}/{family}']
        assert saved['sections'] == len(part) and saved['donors'] == len(donors)
        by_donor = {str(donor): sum(r['top1_weak_five_point_mm'] for r in part
                    if r['donor_id'] == donor) / sum(r['donor_id'] == donor for r in part)
                    for donor in donors}
        assert set(saved['by_donor_top1_mm']) == set(by_donor)
        assert all(math.isclose(saved['by_donor_top1_mm'][donor], value, abs_tol=1e-6)
                   for donor, value in by_donor.items())
        assert math.isclose(saved['donor_equal_top1_mm'],
                            sum(by_donor.values()) / len(by_donor), abs_tol=1e-6)
        for reported, field in (('donor_equal_top16_truth_best_mm', 'top16_truth_best_weak_five_point_mm'),
                                ('donor_equal_top1_normal_deg', 'top1_weak_normal_deg')):
            value = sum(sum(r[field] for r in part if r['donor_id'] == donor) /
                        sum(r['donor_id'] == donor for r in part) for donor in donors) / len(donors)
            assert math.isclose(saved[reported], value, abs_tol=1e-6)

baseline, final = summary['synthetic']['0/v4'], summary['synthetic']['653/v4']
conditions = {'v4_plan_equal_top1_gain_ge_0p25_mm':
              baseline['top1_plan_equal_mm'] - final['top1_plan_equal_mm'] >= .25,
              'v4_plan_equal_top16_near_drop_le_0p05':
              final['top16_near_plan_equal'] >= baseline['top16_near_plan_equal'] - .05,
              'both_weak_real_families_regression_le_0p20_mm': all(
                  summary['weak_real'][f'653/{family}']['donor_equal_top1_mm'] <=
                  summary['weak_real'][f'0/{family}']['donor_equal_top1_mm'] + .20
                  for family in ('coronal', 'sagittal')),
              'each_weak_real_donor_regression_le_0p50_mm': all(
                  summary['weak_real'][f'653/{family}']['by_donor_top1_mm'][donor] <= value + .50
                  for family in ('coronal', 'sagittal')
                  for donor, value in summary['weak_real'][f'0/{family}']['by_donor_top1_mm'].items()),
              'supplied_exact_and_near_map_regression_le_0p10_mm': all(
                  summary['map'][f'653/{role}']['plan_equal_valid_mean_mm'] <=
                  summary['map'][f'0/{role}']['plan_equal_valid_mean_mm'] + .10
                  for role in ('exact', 'baseline_near'))}
assert summary['development_gate']['conditions'] == conditions
assert summary['development_gate']['passed'] == all(conditions.values())
print(json.dumps({'event': 'independent_verifier_passed',
                  'synthetic_rows': len(synthetic), 'map_rows': len(mapping),
                  'real_weak_rows': len(real), 'development_gate': summary['development_gate']}))
