"""Read-only, post-exit verification of the frozen TRAIN-only 308 run."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/real_corpus_diversity_308_001'
SOURCE = Path(__file__).with_name('train_real_corpus_diversity_308.py')
PROTOCOL = SOURCE.parents[1] / 'docs/publication/REAL_CORPUS_DIVERSITY_308_TRAIN_PROTOCOL_20261014.md'
SPLIT = ROOT / 'data/coronal_train_split_308_001/split.json'
STEPS = (0, 1000, 2000, 4000)
COHORT_COUNTS = {'v9': 128, 'v10': 128, 'v10_sparse': 32,
                 'lupori': 52, 'coronal_new': 100, 'sagittal': 20}
SAGITTAL_HELDOUT = {10164, 10177, 10180, 10184, 10236, 10237, 10239, 10253,
                    10255, 10301, 10304, 10308, 10336, 10349, 10363, 10371,
                    10407, 10412, 10413, 10417}
EXPECTED_SHA = {
    'source': 'dd30b09bf3200f2b63fb7861a1d6a000b694f525096e77c4df4029bd7ce2e26d',
    'protocol': '90bb144a64c8f356dd6ffd6b5b0f43571fa4e53261927ddc7d33e8dda87b66d4',
    'split': 'b0e5ce4060c6192c2fc166494459c5134beda3caa55488d01a6581c374821bbf',
    'parent': '56d10febb1a6e6592adf18b9b4c7fe1fe0beae2a9179d9ddaffb78eefb82e692',
}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(SOURCE) == EXPECTED_SHA['source']
assert sha(PROTOCOL) == EXPECTED_SHA['protocol']
assert sha(SPLIT) == EXPECTED_SHA['split']
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['counts'] == {'source_donors': 1885, 'old_excluded': 58,
                           'new_inner': 100, 'gradient_donors': 1727}
gradient = set(split['gradient_donors'])
old = set(split['prior_pool_excluded_from_308_gradients'])
inner = {row['animal_partition_key'] for row in split['new_inner_readout']}
assert len(gradient) == 1727 and len(old) == 58 and len(inner) == 100
assert not (gradient & old or gradient & inner or old & inner)
control_donors = sorted(gradient, key=lambda key: hashlib.sha256(
    f'coronal-308-control-20261014:{key}'.encode()).hexdigest())[:20]

config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
completion = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
assert config['candidate'] == completion['candidate'] == 308
assert config['seed'] == 20261014308 and config['updates_per_arm'] == 4000
assert tuple(config['readout_steps']) == STEPS
assert config['control_donors'] == control_donors
assert config['gradient_donors'] == 1727 and config['coronal_presentations_per_arm'] == 2000
assert config['weak_allen_loss_weight'] == .3
assert config['source_sha256'] == EXPECTED_SHA['source']
assert config['protocol_sha256'] == EXPECTED_SHA['protocol']
assert config['split_sha256'] == EXPECTED_SHA['split']
assert config['parent_sha256'] == EXPECTED_SHA['parent']
assert config['corpus_source_manifest_sha256'] == split['source_manifest_sha256']
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is False and completion['calibrated'] is False
assert set(completion['output_sha256']) == {
    'config.json', 'schedule.jsonl', 'panel_draws.jsonl', 'panels.pt'}
for name, digest in completion['output_sha256'].items():
    assert sha(RUN / name) == digest
assert config['panels_sha256'] == sha(RUN / 'panels.pt')
assert config['panel_draws_sha256'] == sha(RUN / 'panel_draws.jsonl')
assert config['schedule_sha256'] == sha(RUN / 'schedule.jsonl')

panel_draws = [json.loads(line) for line in (RUN / 'panel_draws.jsonl').open(encoding='utf8')]
accepted = [row for row in panel_draws if row['accepted']]
assert Counter(row['cohort'] for row in accepted) == Counter({
    'v9': 128, 'v10': 128, 'v10_sparse': 32})
assert len({row['seed'] for row in panel_draws}) == len(panel_draws)
assert len({row['provenance']['physical_section_id'] for row in accepted}) == 288
assert all(row['provenance']['split'] == 'train' for row in panel_draws)
assert all(56 <= row['base'] < 64 for row in panel_draws)
assert all({row['slot'] for row in accepted if row['cohort'] == cohort} ==
           set(range(count)) for cohort, count in (('v9', 128), ('v10', 128), ('v10_sparse', 32)))
# The runner logged valid-mask fraction here, not an independent tissue-visibility measure.
assert all(row['visible_fraction'] < .15 for row in accepted
           if row['cohort'] == 'v10_sparse')
panel_ids = {cohort: {row['provenance']['physical_section_id'] for row in accepted
                      if row['cohort'] == cohort}
             for cohort in ('v9', 'v10', 'v10_sparse')}

schedule = [json.loads(line) for line in (RUN / 'schedule.jsonl').open(encoding='utf8')]
assert len(schedule) == 4000
assert [row['step'] for row in schedule] == list(range(1, 4001))
coronal = {'control': [], 'intervention': []}
for row in schedule:
    step = row['step']
    assert len(row['synthetic_bases']) == 2
    assert all(0 <= base < 56 for base in row['synthetic_bases'])
    assert row['lupori']['animal_id'] in ('CC2B', 'CC4B', 'CC4A')
    assert row['allen_domain'] == ('sagittal' if step % 2 else 'coronal')
    if step % 2:
        assert row['sagittal']['donor_id'] not in SAGITTAL_HELDOUT
        assert row['sagittal']['image_sha256']
    else:
        for arm in coronal:
            sample = row['coronal'][arm]
            key = sample['animal_partition_key']
            assert key in gradient and key not in old and key not in inner
            assert 0 <= sample['donor_array_row'] < split['donor_shards'][key]['image_count']
            assert sample['source_image_sha256']
            coronal[arm].append(key)
assert len(coronal['control']) == len(coronal['intervention']) == 2000
assert set(coronal['control']) == set(control_donors)
assert set(Counter(coronal['control']).values()) == {100}
assert set(coronal['intervention'][:1727]) == gradient
assert len(set(coronal['intervention'][1727:])) == 273

matched = {}
readouts = {}
readout_ids = {}
expected_inner = {(row['animal_partition_key'], row['section_id'],
                   row['source_image_sha256']) for row in split['new_inner_readout']}
for arm in ('control', 'intervention'):
    directory = RUN / arm
    receipt = json.loads((directory / 'completed.json').read_text(encoding='utf8'))
    assert receipt['arm'] == arm and receipt['updates'] == 4000
    assert receipt['calibrated'] is False
    assert set(receipt['output_sha256']) == {'draws.jsonl', 'training.jsonl',
        'readouts.jsonl', *(f'pose_local_{step:05d}.pt' for step in STEPS)}
    assert completion['arm_completion_sha256'][arm] == sha(directory / 'completed.json')
    for name, digest in receipt['output_sha256'].items():
        assert sha(directory / name) == digest
    training = [json.loads(line) for line in (directory / 'training.jsonl').open(encoding='utf8')]
    assert len(training) == 4000
    assert [row['step'] for row in training] == list(range(1, 4001))
    assert all(row['allen_domain'] == ('sagittal' if row['step'] % 2 else 'coronal')
               for row in training)

    draws = [json.loads(line) for line in (directory / 'draws.jsonl').open(encoding='utf8')]
    kinds = Counter(row['kind'] for row in draws)
    assert kinds['lupori'] == kinds['allen'] == kinds['site_indices'] == 4000
    synthetic_draws = [row for row in draws if row['kind'] == 'synthetic']
    assert len({row['seed'] for row in synthetic_draws}) == len(synthetic_draws)
    assert not ({row['seed'] for row in synthetic_draws} &
                {row['seed'] for row in panel_draws})
    assert len({row['provenance']['physical_section_id'] for row in synthetic_draws}) == len(synthetic_draws)
    assert not ({row['provenance']['physical_section_id'] for row in synthetic_draws} &
                {row['provenance']['physical_section_id'] for row in panel_draws})
    accepted_synthetic = Counter()
    for row in draws:
        step = row['step']
        assert 1 <= step <= 4000
        scheduled = schedule[step - 1]
        if row['kind'] == 'synthetic':
            slot = row['slot']
            assert slot in (0, 1)
            assert row['version'] == ('v9' if slot == 0 else 'v10')
            assert row['base'] == scheduled['synthetic_bases'][slot]
            assert row['provenance']['split'] == 'train'
            if row['accepted']:
                accepted_synthetic[step, slot] += 1
        elif row['kind'] == 'lupori':
            assert row['slot'] == 2
            assert all(row[key] == value for key, value in scheduled['lupori'].items())
        elif row['kind'] == 'allen':
            assert row['slot'] == 3 and row['domain'] == scheduled['allen_domain']
            if row['domain'] == 'coronal':
                source = scheduled['coronal'][arm]
                assert row['animal_partition_key'] == source['animal_partition_key']
                assert row['array_row_index'] == source['donor_array_row']
            else:
                source = scheduled['sagittal']
                assert row['donor_id'] == source['donor_id']
                assert row['array_row_index'] == source['array_row_index']
                assert row['source_image_sha256'] == source['image_sha256']
            assert all(row[key] == source[key] for key in (
                'specimen_id', 'experiment_id', 'section_id', 'source_image_sha256')
                if key in source)
        else:
            assert row['kind'] == 'site_indices'
            assert len(row['synthetic_sha256']) == len(row['lupori_sha256']) == 64
    assert sum(accepted_synthetic.values()) == 8000
    assert len(accepted_synthetic) == 8000
    assert set(accepted_synthetic.values()) == {1}
    matched[arm] = [row for row in draws if row['kind'] != 'allen'
                    or row['domain'] == 'sagittal']

    records = [json.loads(line) for line in (directory / 'readouts.jsonl').open(encoding='utf8')]
    assert len(records) == 4 and tuple(row['step'] for row in records) == STEPS
    readouts[arm] = {}
    fixed_ids = None
    for record in records:
        rows = record['rows']
        assert Counter(row['cohort'] for row in rows) == Counter(COHORT_COUNTS)
        ids = {cohort: [row for row in rows if row['cohort'] == cohort]
               for cohort in COHORT_COUNTS}
        assert {(row['animal_partition_key'], row['section_id'],
                 row['source_image_sha256']) for row in ids['coronal_new']} == expected_inner
        assert {row['donor_id'] for row in ids['sagittal']} == SAGITTAL_HELDOUT
        assert all(row['animal_id'] == 'AL1A' for row in ids['lupori'])
        assert len({row['source_section_id'] for row in ids['lupori']}) == 52
        for cohort in panel_ids:
            assert {row['physical_section_id'] for row in ids[cohort]} == panel_ids[cohort]
        identity = {
            'synthetic': {cohort: sorted(row['physical_section_id'] for row in ids[cohort])
                          for cohort in panel_ids},
            'lupori': sorted((row['animal_id'], row['source_section_id'],
                              row['source_image_sha256']) for row in ids['lupori']),
            'coronal_new': sorted((row['animal_partition_key'], row['section_id'],
                                   row['source_image_sha256']) for row in ids['coronal_new']),
            'sagittal': sorted((row['donor_id'], row['section_id'],
                                row['source_image_sha256']) for row in ids['sagittal']),
        }
        if fixed_ids is None:
            fixed_ids = identity
        assert identity == fixed_ids
        metrics = {}
        for cohort in ('v9', 'v10', 'v10_sparse'):
            group = ids[cohort]
            metrics[cohort] = {'n': len(group),
                'mean_first_gap_mm': float(np.mean([
                    row['first_rigid_site_gap_mm'] for row in group])),
                'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5
                                            for row in group)}
        group = ids['lupori']
        metrics['lupori'] = {'n': len(group),
            'mean_first_gap_mm': float(np.mean([
                row['first_five_site_gap_mm'] for row in group]))}
        for cohort in ('coronal_new', 'sagittal'):
            group = ids[cohort]
            metrics[cohort] = {'n': len(group),
                'median_weak_centre_gap_mm': float(np.median([
                    row['first_weak_centre_gap_mm'] for row in group])),
                'first_axis_correct': sum(row['first_axis_correct'] for row in group)}
        assert metrics == record['metrics'] == receipt['readouts'][str(record['step'])]
        readouts[arm][str(record['step'])] = metrics
    readout_ids[arm] = fixed_ids

assert matched['control'] == matched['intervention']
assert readout_ids['control'] == readout_ids['intervention']
matched_sha = hashlib.sha256(json.dumps(
    matched['control'], sort_keys=True, allow_nan=False).encode()).hexdigest()
assert completion['matched_noncoronal_draw_sha256'] == matched_sha
assert readouts['control']['0'] == readouts['intervention']['0']
control = readouts['control']['4000']
broad = readouts['intervention']['4000']
clauses = {
    'new_coronal_median_centre_improvement_ge0p15mm':
        broad['coronal_new']['median_weak_centre_gap_mm'] <=
        control['coronal_new']['median_weak_centre_gap_mm'] - .15,
    'new_coronal_first_axis_loss_le2':
        broad['coronal_new']['first_axis_correct'] >=
        control['coronal_new']['first_axis_correct'] - 2,
    'v9_mean_first_gap_regression_le0p15mm':
        broad['v9']['mean_first_gap_mm'] <= control['v9']['mean_first_gap_mm'] + .15,
    'v10_mean_first_gap_regression_le0p15mm':
        broad['v10']['mean_first_gap_mm'] <= control['v10']['mean_first_gap_mm'] + .15,
    'v9_top16_capture_loss_le4':
        broad['v9']['top16_capture_1p5mm'] >= control['v9']['top16_capture_1p5mm'] - 4,
    'v10_top16_capture_loss_le4':
        broad['v10']['top16_capture_1p5mm'] >= control['v10']['top16_capture_1p5mm'] - 4,
    'v10_sparse_top16_capture_loss_le2':
        broad['v10_sparse']['top16_capture_1p5mm'] >=
        control['v10_sparse']['top16_capture_1p5mm'] - 2,
    'lupori_first_gap_regression_le0p20mm':
        broad['lupori']['mean_first_gap_mm'] <= control['lupori']['mean_first_gap_mm'] + .20,
    'sagittal_first_axis_loss_le1':
        broad['sagittal']['first_axis_correct'] >= control['sagittal']['first_axis_correct'] - 1,
}
assert completion['clauses'] == clauses
assert completion['control_step_4000'] == control
assert completion['broad_step_4000'] == broad
assert completion['gate_pass'] is all(clauses.values())
print(json.dumps({'verified': True, 'gate_pass': all(clauses.values()),
                  'clauses': clauses, 'matched_noncoronal_draw_sha256': matched_sha}))
