"""Read-only post-exit audit of the frozen 309 first-choice exposure run."""
import copy
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
import torch

from training.arbitrary_plane_fov_balance_v10 import (
    CODE_SHA256 as V10_SHA, FOV_SCALE, sample_fov_balance_v10)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64

RUN = ROOT / 'runs/first_choice_exposure_309_001'
SOURCE = Path(__file__).with_name('train_first_choice_exposure_309.py')
PROTOCOL = SOURCE.parents[1] / 'docs/publication/FIRST_CHOICE_EXPOSURE_309_PROTOCOL_20261014.md'
SPLIT = ROOT / 'data/coronal_train_split_308_001/split.json'
PARENT = ROOT / 'runs/real_corpus_diversity_308_001/control/pose_local_04000.pt'
STEPS = (0, 5000, 10000, 20000)
COUNTS = {'v9': 256, 'v10': 256, 'v10_sparse': 64, 'v10_steep': 64,
          'lupori': 52, 'coronal_new': 100, 'sagittal': 20}
SAGITTAL_HELDOUT = {10164, 10177, 10180, 10184, 10236, 10237, 10239, 10253,
                    10255, 10301, 10304, 10308, 10336, 10349, 10363, 10371,
                    10407, 10412, 10413, 10417}
EXPECTED = {'source': 'c6d623926c2bdfa5793f335e339252a16431159f92e5d88ca46c96b2fc9f3e0f',
            'protocol': 'dd0451ec74565022240988bee80089924ca3d406aebae97870a3e91c22855489',
            'split': 'b0e5ce4060c6192c2fc166494459c5134beda3caa55488d01a6581c374821bbf',
            'parent': '33d7440e6fa2a75567dcc65675a466314af0fb2a35b981654fa0b4b3b1428f11'}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(SOURCE) == EXPECTED['source']
assert sha(PROTOCOL) == EXPECTED['protocol']
assert sha(SPLIT) == EXPECTED['split']
assert sha(PARENT) == EXPECTED['parent']
split = json.loads(SPLIT.read_text(encoding='utf8'))
gradient = set(split['gradient_donors'])
old = set(split['prior_pool_excluded_from_308_gradients'])
inner_rows = split['new_inner_readout']
inner = {row['animal_partition_key'] for row in inner_rows}
assert len(gradient) == 1727 and len(old) == 58 and len(inner) == 100
assert not (gradient & old or gradient & inner or old & inner)
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
completion = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
assert config['candidate'] == completion['candidate'] == 309
assert config['seed'] == 20261014309 and config['optimizer_batches'] == 20000
assert tuple(config['readout_steps']) == STEPS
assert config['batch_slots'] == ['v9', 'v10', 'v10', 'cyclic_lupori_coronal_sagittal']
assert config['gradient_donors'] == 1727 and config['weak_allen_loss_weight'] == .3
assert config['optimized_modules'] == ['encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global']
assert {key: config[key + '_sha256'] for key in ('source', 'protocol', 'split', 'parent')} == EXPECTED
assert config['corpus_source_manifest_sha256'] == split['source_manifest_sha256']
assert config['no_dev_cal_final_or_public_benchmark'] is True
assert config['calibrated'] is False and completion['calibrated'] is False
assert config['v10_generator_sha256'] == sha(SOURCE.with_name(
    'arbitrary_plane_fov_balance_v10.py')) == V10_SHA
for name, source in (('model_source_sha256', 'arbitrary_plane_one_shot_model.py'),
                     ('inference_source_sha256', 'arbitrary_plane_one_shot_inference.py'),
                     ('postartifact_gauge_source_sha256', 'arbitrary_plane_postartifact_gauge_v1.py')):
    assert config[name] == sha(SOURCE.with_name(source))
assert config['lupori_split_sha256'] == sha(SOURCE.with_name('lupori_animal_split_185.json'))
assert config['sagittal_manifest_sha256'] == sha(ROOT /
    'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl')
parent_arm = PARENT.parent / 'completed.json'
parent_run = PARENT.parents[1] / 'completed.json'
assert config['parent_arm_completion_sha256'] == sha(parent_arm)
assert config['parent_run_completion_sha256'] == sha(parent_run)
assert json.loads(parent_arm.read_text(encoding='utf8'))['output_sha256'][PARENT.name] == EXPECTED['parent']
assert json.loads(parent_run.read_text(encoding='utf8'))['arm_completion_sha256']['control'] == sha(parent_arm)
assert set(completion['output_sha256']) == {
    'config.json', 'schedule.jsonl', 'panel_draws.jsonl', 'panels.pt'}
for name, digest in completion['output_sha256'].items():
    assert sha(RUN / name) == digest
assert config['panels_sha256'] == sha(RUN / 'panels.pt')
assert config['panel_draws_sha256'] == sha(RUN / 'panel_draws.jsonl')
assert config['schedule_sha256'] == sha(RUN / 'schedule.jsonl')

panel_draws = [json.loads(line) for line in (RUN / 'panel_draws.jsonl').open(encoding='utf8')]
accepted = [row for row in panel_draws if row['accepted']]
assert Counter(row['cohort'] for row in accepted) == Counter({key: COUNTS[key] for key in
    ('v9', 'v10', 'v10_sparse', 'v10_steep')})
assert len({row['seed'] for row in panel_draws}) == len(panel_draws)
assert len({row['provenance']['physical_section_id'] for row in accepted}) == 640
assert all(row['provenance']['split'] == 'train' and 56 <= row['base'] < 64
           for row in panel_draws)
for cohort in ('v9', 'v10', 'v10_sparse', 'v10_steep'):
    assert {row['slot'] for row in accepted if row['cohort'] == cohort} == set(range(COUNTS[cohort]))
assert all(row['postartifact_visible_fraction'] < .15 for row in accepted
           if row['cohort'] == 'v10_sparse')
assert all(row['obliquity_deg'] >= 45 for row in accepted if row['cohort'] == 'v10_steep')
panel_ids = {cohort: {row['provenance']['physical_section_id'] for row in accepted
                      if row['cohort'] == cohort}
             for cohort in ('v9', 'v10', 'v10_sparse', 'v10_steep')}

schedule = [json.loads(line) for line in (RUN / 'schedule.jsonl').open(encoding='utf8')]
assert len(schedule) == 20000 and [row['step'] for row in schedule] == list(range(1, 20001))
assert Counter(row['acquired_domain'] for row in schedule) == Counter(
    {'lupori': 6667, 'coronal': 6667, 'sagittal': 6666})
coronal, sagittal, lupori = [], [], []
for row in schedule:
    step, domain = row['step'], row['acquired_domain']
    assert domain == ('lupori', 'coronal', 'sagittal')[(step - 1) % 3]
    assert len(row['synthetic_bases']) == 3 and all(0 <= base < 56 for base in row['synthetic_bases'])
    if domain == 'lupori':
        sample = row['lupori']
        assert sample['animal_id'] in ('CC2B', 'CC4B', 'CC4A') and sample['source_image_sha256']
        lupori.append(sample['animal_id'])
    elif domain == 'coronal':
        sample = row['coronal']
        key = sample['animal_partition_key']
        assert key in gradient and key not in inner and key not in old
        assert 0 <= sample['donor_array_row'] < split['donor_shards'][key]['image_count']
        assert sample['source_image_sha256']
        coronal.append(key)
    else:
        sample = row['sagittal']
        assert sample['donor_id'] not in SAGITTAL_HELDOUT and sample['image_sha256']
        sagittal.append(sample['donor_id'])
for start in range(0, len(coronal), 1727):
    cohort = coronal[start:start + 1727]
    assert len(cohort) == len(set(cohort))
    if len(cohort) == 1727:
        assert set(cohort) == gradient
for start in range(0, len(lupori), 3):
    if len(lupori[start:start + 3]) == 3:
        assert set(lupori[start:start + 3]) == {'CC2B', 'CC4B', 'CC4A'}
sagittal_train = set(sagittal)
assert len(sagittal_train) == 20 and not (sagittal_train & SAGITTAL_HELDOUT)
for start in range(0, len(sagittal), 20):
    cohort = sagittal[start:start + 20]
    assert len(cohort) == len(set(cohort))
    if len(cohort) == 20:
        assert set(cohort) == sagittal_train
assert config['acquired_presentations'] == {'lupori': 6667, 'coronal': 6667, 'sagittal': 6666}

directory = RUN / 'single'
receipt = json.loads((directory / 'completed.json').read_text(encoding='utf8'))
assert receipt['arm'] == 'single' and receipt['updates'] == 20000
assert receipt['calibrated'] is False and completion['training_completion_sha256'] == sha(
    directory / 'completed.json')
assert set(receipt['output_sha256']) == {'draws.jsonl', 'training.jsonl', 'readouts.jsonl',
    *(f'pose_local_{step:05d}.pt' for step in STEPS)}
for name, digest in receipt['output_sha256'].items():
    assert sha(directory / name) == digest
with (directory / 'training.jsonl').open(encoding='utf8') as file:
    for step, line in enumerate(file, 1):
        row = json.loads(line)
        assert step <= 20000 and row['step'] == step
        domain = schedule[step - 1]['acquired_domain']
        assert row['acquired_domain'] == domain
        assert row['acquired_loss_weight'] == (1. if domain == 'lupori' else .3)
        assert all(math.isfinite(row[key]) for key in (
            'synthetic_loss', 'acquired_weak_loss', 'loss', 'gradient_norm'))
    assert step == 20000

panel_seed_set = {row['seed'] for row in panel_draws}
panel_id_set = {row['provenance']['physical_section_id'] for row in panel_draws}
draw_seeds, draw_ids = set(), set()
accepted_slots, acquired_count, site_count = Counter(), Counter(), Counter()
with (directory / 'draws.jsonl').open(encoding='utf8') as file:
    for line in file:
        row = json.loads(line)
        step = row['step']
        assert 1 <= step <= 20000
        scheduled = schedule[step - 1]
        if row['kind'] == 'synthetic':
            assert row['slot'] in (0, 1, 2)
            assert row['version'] == ('v9' if row['slot'] == 0 else 'v10')
            assert row['base'] == scheduled['synthetic_bases'][row['slot']]
            assert row['provenance']['split'] == 'train'
            assert row['seed'] not in draw_seeds and row['seed'] not in panel_seed_set
            physical_id = row['provenance']['physical_section_id']
            assert physical_id not in draw_ids and physical_id not in panel_id_set
            draw_seeds.add(row['seed'])
            draw_ids.add(physical_id)
            if row['accepted']:
                accepted_slots[step, row['slot']] += 1
        elif row['kind'] == 'site_indices':
            assert len(row['synthetic_sha256']) == 64
            assert ('lupori_sha256' in row) == (scheduled['acquired_domain'] == 'lupori')
            if 'lupori_sha256' in row:
                assert len(row['lupori_sha256']) == 64
            site_count[step] += 1
        else:
            domain = scheduled['acquired_domain']
            assert row['slot'] == 3
            assert row['kind'] == ('lupori' if domain == 'lupori' else 'allen')
            if domain == 'lupori':
                assert all(row[key] == value for key, value in scheduled['lupori'].items())
            else:
                sample = scheduled[domain]
                assert row['specimen_id'] == sample['specimen_id']
                assert row['experiment_id'] == sample['experiment_id']
                assert row['section_id'] == sample['section_id']
                assert row['array_row_index'] == sample[
                    'donor_array_row' if domain == 'coronal' else 'array_row_index']
                assert row['source_image_sha256'] == sample[
                    'source_image_sha256' if domain == 'coronal' else 'image_sha256']
            acquired_count[step] += 1
assert len(accepted_slots) == 60000 and set(accepted_slots.values()) == {1}
assert len(acquired_count) == len(site_count) == 20000
assert set(acquired_count.values()) == set(site_count.values()) == {1}

records = [json.loads(line) for line in (directory / 'readouts.jsonl').open(encoding='utf8')]
assert len(records) == 4 and tuple(row['step'] for row in records) == STEPS
expected_inner = {(row['animal_partition_key'], row['section_id'],
                   row['source_image_sha256']) for row in inner_rows}
readouts, fixed_ids = {}, None
for record in records:
    rows = record['rows']
    assert Counter(row['cohort'] for row in rows) == Counter(COUNTS)
    groups = {cohort: [row for row in rows if row['cohort'] == cohort] for cohort in COUNTS}
    assert {(row['animal_partition_key'], row['section_id'], row['source_image_sha256'])
            for row in groups['coronal_new']} == expected_inner
    assert {row['donor_id'] for row in groups['sagittal']} == SAGITTAL_HELDOUT
    assert all(row['animal_id'] == 'AL1A' for row in groups['lupori'])
    assert len({row['source_section_id'] for row in groups['lupori']}) == 52
    for cohort in panel_ids:
        assert {row['physical_section_id'] for row in groups[cohort]} == panel_ids[cohort]
    identities = {
        'synthetic': {cohort: sorted(row['physical_section_id'] for row in groups[cohort])
                      for cohort in panel_ids},
        'lupori': sorted((row['animal_id'], row['source_section_id'], row['source_image_sha256'])
                         for row in groups['lupori']),
        'coronal': sorted((row['animal_partition_key'], row['section_id'], row['source_image_sha256'])
                          for row in groups['coronal_new']),
        'sagittal': sorted((row['donor_id'], row['section_id'], row['source_image_sha256'])
                           for row in groups['sagittal'])}
    if fixed_ids is None:
        fixed_ids = identities
    assert identities == fixed_ids
    metrics = {}
    for cohort in ('v9', 'v10', 'v10_sparse', 'v10_steep'):
        group = groups[cohort]
        metrics[cohort] = {'n': len(group),
            'mean_first_gap_mm': float(np.mean([row['first_rigid_site_gap_mm'] for row in group])),
            'top16_capture_1p5mm': sum(row['top16_oracle_gap_mm'] <= 1.5 for row in group)}
    group = groups['lupori']
    metrics['lupori'] = {'n': len(group),
        'mean_first_gap_mm': float(np.mean([row['first_five_site_gap_mm'] for row in group])),
        'median_weak_centre_gap_mm': float(np.median([
            row['first_weak_centre_gap_mm'] for row in group])),
        'first_axis_correct': sum(row['first_axis_correct'] for row in group)}
    for cohort in ('coronal_new', 'sagittal'):
        group = groups[cohort]
        metrics[cohort] = {'n': len(group),
            'median_weak_centre_gap_mm': float(np.median([
                row['first_weak_centre_gap_mm'] for row in group])),
            'first_axis_correct': sum(row['first_axis_correct'] for row in group)}
    assert metrics == record['metrics'] == receipt['readouts'][str(record['step'])]
    readouts[str(record['step'])] = metrics

baseline = readouts['0']
clauses_by_step = {}
for step in STEPS:
    result = readouts[str(step)]
    clauses_by_step[str(step)] = {
        'v10_first_mean_improves_ge0p35mm':
            result['v10']['mean_first_gap_mm'] <= baseline['v10']['mean_first_gap_mm'] - .35,
        'v9_first_mean_regression_le0p15mm':
            result['v9']['mean_first_gap_mm'] <= baseline['v9']['mean_first_gap_mm'] + .15,
        'v10_top16_capture_loss_le4':
            result['v10']['top16_capture_1p5mm'] >= baseline['v10']['top16_capture_1p5mm'] - 4,
        'v10_sparse_first_mean_regression_le0p20mm':
            result['v10_sparse']['mean_first_gap_mm'] <= baseline['v10_sparse']['mean_first_gap_mm'] + .20,
        'v10_steep_first_mean_regression_le0p20mm':
            result['v10_steep']['mean_first_gap_mm'] <= baseline['v10_steep']['mean_first_gap_mm'] + .20,
        'lupori_first_mean_regression_le0p20mm':
            result['lupori']['mean_first_gap_mm'] <= baseline['lupori']['mean_first_gap_mm'] + .20,
        'coronal_median_weak_centre_regression_le0p20mm':
            result['coronal_new']['median_weak_centre_gap_mm'] <=
            baseline['coronal_new']['median_weak_centre_gap_mm'] + .20,
        'sagittal_first_axis_loss_le1':
            result['sagittal']['first_axis_correct'] >= baseline['sagittal']['first_axis_correct'] - 1,
    }
guarded = [step for step in STEPS if all(value for key, value in
    clauses_by_step[str(step)].items() if key != 'v10_first_mean_improves_ge0p35mm')]
selected = min(guarded, key=lambda step: (readouts[str(step)]['v10']['mean_first_gap_mm'], step))
assert completion['clauses_by_step'] == clauses_by_step
assert completion['selected_step'] == selected
assert completion['selected_checkpoint'] == f'single/pose_local_{selected:05d}.pt'
assert completion['readouts'] == readouts
assert completion['gate_pass'] == all(clauses_by_step['20000'].values())

with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(PARENT, map_location='cpu', weights_only=True)
    initial = torch.load(directory / 'pose_local_00000.pt', map_location='cpu', weights_only=True)
    final = torch.load(directory / 'pose_local_20000.pt', map_location='cpu', weights_only=True)
assert initial['step'] == 5000 and initial['local_step'] == 0
assert final['step'] == 25000 and final['local_step'] == 20000
assert initial['arm'] == final['arm'] == 'single'
assert initial['calibrated'] is final['calibrated'] is False
assert parent['model'].keys() == initial['model'].keys() == final['model'].keys()
assert all(torch.equal(value, initial['model'][name]) for name, value in parent['model'].items())
trainable = ('encoder.', 'lateral.', 'pose.', 'anchor_pose.', 'anchor_global.')
assert all(torch.equal(value, final['model'][name]) for name, value in parent['model'].items()
           if not name.startswith(trainable))
assert any(not torch.equal(value, final['model'][name]) for name, value in parent['model'].items()
           if name.startswith(trainable))
assert parent['optimizer']['state'].keys() == initial['optimizer']['state'].keys()
for parameter, state in parent['optimizer']['state'].items():
    assert state.keys() == initial['optimizer']['state'][parameter].keys()
    for name, value in state.items():
        restored = initial['optimizer']['state'][parameter][name]
        assert torch.equal(value, restored) if torch.is_tensor(value) else value == restored
for index, (saved, loaded, trained) in enumerate(zip(
        parent['optimizer']['param_groups'], initial['optimizer']['param_groups'],
        final['optimizer']['param_groups'])):
    assert saved['params'] == loaded['params'] == trained['params']
    assert saved['base_lr'] == loaded['base_lr'] == trained['base_lr'] == (2e-5, 5e-6)[index]
    assert loaded['lr'] == trained['lr'] == .2 * (2e-5, 5e-6)[index]
del parent, initial, final

torch.set_num_threads(4)
v9 = load_streaming_synthetic_v7_64('cuda')
assert config['v9_generator_input_sha256'] == v9['provenance']['input_sha256']
assert FOV_SCALE == .72
v10 = {**v9, 'subjects': copy.deepcopy(v9['subjects']),
       'nuisance': v9['nuisance'].copy(), 'provenance': copy.deepcopy(v9['provenance'])}
v10['provenance']['virtual_subjects'] = v10['subjects']
v10['provenance']['fov_balance_v10'] = {'code_sha256': V10_SHA, 'fov_scale': FOV_SCALE}
for subject in v10['subjects']:
    bounds = np.asarray(subject['bounds_um'])
    middle = bounds.mean(0)
    subject['bounds_um'] = (middle + FOV_SCALE * (bounds - middle)).tolist()
v10['nuisance'][:, 2:4] *= FOV_SCALE
with torch.inference_mode():
    for row in accepted:
        if row['cohort'] not in ('v10_sparse', 'v10_steep'):
            continue
        sample = sample_fov_balance_v10(v10,
            [row['base'] * VARIANTS + row['variant']], row['seed'], 192)
        visible = float(((sample['visible'][0] > .05) &
            ((sample['inputs'][0, 2] < .5) | sample['brush_mask'][0])).float().mean())
        normal = full_frame_state_to_components(sample['state'])[1][0, :, 2]
        obliquity = math.degrees(math.acos(float(normal.abs().max().clamp(0, 1))))
        assert bool(sample['eligible'][0])
        assert sample['provenance'][0]['physical_section_id'] == row['provenance']['physical_section_id']
        assert abs(visible - row['postartifact_visible_fraction']) < 1e-6
        assert abs(obliquity - row['obliquity_deg']) < 1e-5
        assert row['cohort'] != 'v10_sparse' or visible < .15
        assert row['cohort'] != 'v10_steep' or obliquity >= 45
print(json.dumps({'verified': True, 'gate_pass': completion['gate_pass'],
                  'selected_step': selected, 'clauses': clauses_by_step['20000']}))
