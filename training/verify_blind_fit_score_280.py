"""Read-only independent audit of 280; run only after its runner exits."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/blind_fit_score_280_pilot_001'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
PROTOCOL = SOURCE.parent / 'docs/publication/BLIND_FIT_SCORE_280_PROTOCOL_20261012.md'
COMMIT = '1e273a2f40ad0641d0a99a72bdbcd4e2f6f5de65'
RUNNER_SHA256 = '8173b0b082c3565e69c686cfdc381f84f6e6bea487a36029db4f4d8c5f1f2e90'
PROTOCOL_SHA256 = 'beec09db6389ac320d075d05a330d3d48717de58da90484c6bfbb85d72c64f33'
SEED, SIDE, VARIANTS, UPDATES = 20261012280, 192, 64, 1500
ARMS, STEPS = ('full_atlas', 'support_only'), (0, 500, 1500)
POSE_MODULES = ('encoder', 'lateral', 'pose', 'anchor_pose', 'anchor_global')
POSE_BUFFERS = ('normal_anchor_frames', 'center_origin', 'center_scale',
                'euclidean_scale', 'rotation_grid')
TRAIN_MODULES = ('warp_shared', 'warp_condition', 'warp', 'atlas_encoder', 'pair', 'fitted_matcher')
SOURCE_NAMES = (
    'train_blind_fit_score_280.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_stream.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_full_frame_primitives.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonl(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def committed_sha256(path):
    content = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=SOURCE.parent)
    return hashlib.sha256(content).hexdigest()


def identity_check(identity, subject_index, draw_seed, used, synthetic, stage_hashes):
    base = subject_index // VARIANTS
    subject = synthetic['virtual_subjects'][subject_index]
    assert identity['seed_prefix'] == [draw_seed, subject_index, 0]
    assert identity['physical_section_id'] == f'stream-v7-seed-{draw_seed}-row-0-virtual-{subject_index}'
    assert identity['virtual_index'] == subject_index
    assert identity['virtual_subject_id'] == subject['virtual_subject_id']
    assert identity['base_lineage'] == synthetic['base_subjects'][base]['lineage']
    assert identity['split'] == subject['base_lineage']['split'] == 'train'
    assert identity['shape_h_w'] == [SIDE, SIDE]
    assert identity['eligible'] == (identity['visible_support_mass'] >= SIDE * SIDE / 144)
    assert used == (identity['eligible'] and
                    identity['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] >= SIDE * SIDE / 144)
    for version in (3, 4, 6, 7, 9):
        assert identity[f'one_shot_slide_artifacts_v{version}']['code_sha256'] == stage_hashes[version]
    assert identity['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair'] is False
    assert identity['one_shot_slide_artifacts_v9']['seed_branch'] == 34
    assert identity['one_shot_slide_artifacts_v9']['sampling'] == 'one independent physical plane per row; no appearance pairing'
    return identity['physical_section_id']


def summary(rows, acquired):
    by_base = {base: [row for row in rows if row['base_index'] == base] for base in range(56, 64)}
    result = {'n': len(rows),
        'top16_hits_rigid_plane': sum(row['best_rigid_plane_mm_diagnostic_only'] <= 1.5 for row in rows),
        'direct_top1_mean_rigid_plane_mm': float(np.mean([row['direct_top1_original_rigid_plane_mm'] for row in rows])),
        'selected_original_mean_rigid_plane_mm': float(np.mean([row['rank_selected_original_rigid_plane_mm'] for row in rows])),
        'direct_top1_mean_surviving_tissue_mm': float(np.mean([row['direct_top1_surviving_tissue_mm'] for row in rows])),
        'selected_original_mean_surviving_tissue_mm': float(np.mean([row['rank_selected_original_surviving_tissue_mm'] for row in rows])),
        'selected_mapped_mean_mm': float(np.mean([row['rank_selected_mapped_mm'] for row in rows])),
        'best_rigid_plane_mean_mm_diagnostic_only': float(np.mean([row['best_rigid_plane_mm_diagnostic_only'] for row in rows])),
        'physical_best_mean_surviving_tissue_mm_diagnostic_only': float(np.mean([row['physical_best_surviving_tissue_mm_diagnostic_only'] for row in rows])),
        'exact_no_warp_mean_mm': float(np.mean([row['exact_no_warp_mm'] for row in rows])),
        'exact_mapped_mean_mm': float(np.mean([row['exact_mapped_mm'] for row in rows])),
        'exact_displacement_rms_mean_mm': float(np.mean([row['exact_displacement_rms_mm'] for row in rows])),
        'exact_displacement_rms_median_mm': float(np.median([row['exact_displacement_rms_mm'] for row in rows])),
        'exact_surviving_displacement_mean_mm': float(np.mean([row['exact_surviving_displacement_mean_mm'] for row in rows])),
        'exact_surviving_displacement_rms_mm': float(np.mean([row['exact_surviving_displacement_rms_mm'] for row in rows])),
        'oracle_surviving_displacement_mean_mm': float(np.mean([row['oracle_surviving_displacement_mean_mm'] for row in rows])),
        'oracle_surviving_displacement_rms_mm': float(np.mean([row['oracle_surviving_displacement_rms_mm'] for row in rows])),
        'max_actual_derivative': max(row['exact_actual_derivative'] for row in rows),
        'max_blind_actual_derivative': max(branch['actual_derivative'] for row in rows for branch in row['branches']),
        'near_half_mm_n_sections': sum(row['near_half_mm_branch_count'] > 0 for row in rows),
        'near_half_mm_n_branches': sum(row['near_half_mm_branch_count'] for row in rows),
        'near_half_mm_mean_map_gain': float(np.mean([row['near_half_mm_section_mean_map_gain']
            for row in rows if row['near_half_mm_branch_count'] > 0]))
            if any(row['near_half_mm_branch_count'] > 0 for row in rows) else None,
        'support_matched_sections': sum(row['support_matched_pair_eligible'] for row in rows),
        'support_matched_near_wins': sum(row['support_matched_near_win'] is True for row in rows),
        'base_mean_gain_mm': {str(base): float(np.mean([
            row['direct_top1_original_rigid_plane_mm'] - row['rank_selected_original_rigid_plane_mm']
            for row in by_base[base]])) for base in by_base},
        'coronal': {}, 'sagittal': {}}
    for domain in ('coronal', 'sagittal'):
        subset = [row for row in acquired if row['domain'] == domain]
        result[domain] = {'n': len(subset),
            'first_axis_correct': sum(row['first_axis_correct'] for row in subset),
            'median_weak_normal_angle_deg': float(np.median([
                row['normal_gap_to_weak_affine_deg'] for row in subset])),
            'frozen_direct_first_axis_correct': sum(row['frozen_direct_axis'] == row['weak_axis'] for row in subset),
            'frozen_direct_median_weak_normal_angle_deg': float(np.median([
                row['frozen_direct_normal_gap_deg'] for row in subset]))}
    return result


assert sha256(SOURCE / SOURCE_NAMES[0]) == committed_sha256('training/' + SOURCE_NAMES[0]) == RUNNER_SHA256
assert sha256(PROTOCOL) == committed_sha256('docs/publication/' + PROTOCOL.name) == PROTOCOL_SHA256
receipt = json.loads((RUN / 'completed.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
parent_receipt = json.loads((PARENT / 'completed.json').read_text())
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
transfer_config = json.loads((TRANSFER / 'config.json').read_text())
assert receipt['candidate'] == config['candidate'] == 280
assert receipt['steps'] == config['readout_steps'] == list(STEPS)
assert config['seed'] == SEED and config['arms'] == list(ARMS)
assert config['updates_per_arm'] == UPDATES and config['checkpoint_selection'] == 'none; only 1500 eligible for gate'
assert config['train_base_indices'] == list(range(56)) and config['inner_base_indices'] == list(range(56, 64))
assert config['blind_candidates'] == 16 and config['train_subset_size'] == 4
assert (config['fit_side'], config['teacher_side'], config['support_side']) == (24, 64, 24)
assert config['map_derivative_limit'] == .35 and config['support_threshold'] == .5
assert config['parent_pose_buffers_loaded'] == list(POSE_BUFFERS)
assert config['no_dev_cal_final_or_public_benchmark'] is True and config['calibrated'] is False
assert receipt['protocol_sha256'] == config['protocol_sha256'] == PROTOCOL_SHA256
assert set(config['source_sha256']) == set(SOURCE_NAMES) and receipt['source_sha256'] == config['source_sha256']
for name in SOURCE_NAMES:
    expected = committed_sha256('training/' + name)
    assert sha256(SOURCE / name) == config['source_sha256'][name] == expected

assert parent_receipt['candidate'] == config['parent_candidate'] == 275
assert parent_receipt['selected_step'] == 8000
assert config['parent_completed_sha256'] == sha256(PARENT / 'completed.json')
assert config['parent_config_sha256'] == parent_receipt['config_sha256'] == sha256(PARENT / 'config.json')
assert config['parent_checkpoint_sha256'] == parent_receipt['checkpoint_sha256']['8000'] == sha256(PARENT / 'pose_step_08000.pt')
for name, key in (('draws.jsonl', 'parent_draws_sha256'), ('inner_draws.jsonl', 'parent_inner_draws_sha256')):
    assert config[key] == parent_receipt['draws_sha256' if name == 'draws.jsonl' else 'inner_draws_sha256'] == sha256(PARENT / name)
parent_checkpoint = torch.load(PARENT / 'pose_step_08000.pt', map_location='cpu', weights_only=False)
assert parent_checkpoint['step'] == 8000 and parent_checkpoint['config']['candidate'] == 275
parent_model = parent_checkpoint['model']
del parent_checkpoint

assert transfer_receipt['candidate'] == 272
assert config['transfer_cases_sha256'] == transfer_receipt['cases_sha256'] == sha256(TRANSFER / 'cases.jsonl')
assert config['transfer_config_sha256'] == transfer_receipt['config_sha256'] == sha256(TRANSFER / 'config.json')
cases = jsonl(TRANSFER / 'cases.jsonl')
sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'animal_id', 'training_split', 'model_pixel_to_ap_dv_ml_um', 58),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'donor_id', 'split', 'model_pixel_to_ccf_ref9_ap_dv_ml_um', 40),
}
assert [case['domain'] for case in cases] == ['coronal'] * 58 + ['sagittal'] * 40
assert set(config['acquired_source_data']) == set(sources)
for domain, (records_path, images_path, donor_key, split_key, affine_key, count) in sources.items():
    hashes = {'records_sha256': sha256(records_path), 'images_sha256': sha256(images_path)}
    assert hashes == config['acquired_source_data'][domain]
    assert hashes == {key: transfer_config['datasets'][domain][key] for key in hashes}
    records = jsonl(records_path)
    eligible = [record for record in records if record[split_key] == 'train']
    donors = sorted({record[donor_key] for record in eligible})
    expected = [sorted((record for record in eligible if record[donor_key] == donor),
                       key=lambda record: (record['section_number'], record['section_id']))[
                       (sum(record[donor_key] == donor for record in eligible) - 1) // 2]
                for donor in donors]
    subset = [case for case in cases if case['domain'] == domain]
    assert len(donors) == len(subset) == count == transfer_config['datasets'][domain]['donors']
    for case, record in zip(subset, expected):
        assert case['donor_id'] == record[donor_key]
        for key in ('specimen_id', 'experiment_id', 'section_id', 'array_row_index'):
            assert case[key] == record[key]
        assert case['source_image_sha256'] == record.get('source_image_sha256', record.get('image_sha256'))
        matrix = np.asarray(record[affine_key], dtype=np.float64)
        normal = np.cross(matrix[:, 0], matrix[:, 1])
        assert case['weak_axis'] == ('AP', 'DV', 'ML')[int(np.abs(normal).argmax())]

synthetic = config['synthetic_provenance']
assert synthetic['new_independent_train_bases'] == 56 and synthetic['all_train_bases'] == 64
assert len(synthetic['base_subjects']) == 64 and len(synthetic['virtual_subjects']) == 64 * VARIANTS
assert synthetic['calibrated'] is False
for name, expected in synthetic['source_sha256'].items():
    assert Path(name).name == name
    assert sha256(SOURCE / name) == expected == committed_sha256('training/' + name)
for path, expected in synthetic['input_sha256'].items():
    bound = Path(path).resolve()
    assert bound.is_relative_to(ROOT / 'data')
    assert sha256(bound) == expected
for key in ('template', 'annotation'):
    bound = Path(synthetic['atlas_binding'][key + '_path']).resolve()
    assert bound.is_relative_to(ROOT / 'data')
    assert sha256(bound) == synthetic['atlas_binding'][key + '_sha256']
for index, subject in enumerate(synthetic['virtual_subjects']):
    assert subject['virtual_index'] == index
    assert (subject['base_index'], subject['variant']) == divmod(index, VARIANTS)
    assert subject['base_lineage'] == synthetic['base_subjects'][index // VARIANTS]['lineage']
    assert subject['base_lineage']['split'] == 'train'

assert config['inner_panel_sha256'] == sha256(RUN / 'inner_panel.pt')
assert config['inner_draws_sha256'] == sha256(RUN / 'inner_draws.jsonl')
assert set(receipt['output_sha256']) == {
    'config.json', 'inner_draws.jsonl', 'inner_panel.pt', 'training_draws.jsonl',
    'training.jsonl', 'readouts.jsonl'}
for name, expected in receipt['output_sha256'].items():
    assert sha256(RUN / name) == expected
assert set(receipt['checkpoint_sha256']) == {f'{arm}_{step}' for arm in ARMS for step in STEPS}
for arm in ARMS:
    for step in STEPS:
        assert sha256(RUN / f'{arm}_step_{step:05d}.pt') == receipt['checkpoint_sha256'][f'{arm}_{step}']

stage_hashes = {version: config['source_sha256'][f'arbitrary_plane_one_shot_slide_artifacts_v{version}.py']
                for version in (3, 4, 6, 7, 9)}
parent_ids = {row['provenance']['physical_section_id'] for name in ('draws.jsonl', 'inner_draws.jsonl')
              for row in jsonl(PARENT / name)}
inner_draws = jsonl(RUN / 'inner_draws.jsonl')
panel_rng = np.random.default_rng(SEED + 1)
accepted_panel, all_ids = [], set()
for attempt, row in enumerate(inner_draws, 1):
    base, slot = 56 + len(accepted_panel) // 8, len(accepted_panel) % 8
    subject_index = base * VARIANTS + int(panel_rng.integers(VARIANTS))
    draw_seed = SEED + 1_000_000_000 + attempt
    assert (row['base_index'], row['slot'], row['subject_index'], row['draw_seed']) == (
        base, slot, subject_index, draw_seed)
    section_id = identity_check(row['provenance'], subject_index, draw_seed, row['used'], synthetic, stage_hashes)
    assert section_id not in all_ids and section_id not in parent_ids
    all_ids.add(section_id)
    if row['used']:
        accepted_panel.append(row)
assert len(accepted_panel) == 64 and config['panel_attempts'] == len(inner_draws)
panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64
for item, draw in zip(panel, accepted_panel):
    assert item['base_index'] == draw['base_index'] and item['slot'] == draw['slot']
    assert item['physical_section_id'] == draw['provenance']['physical_section_id']
    assert json.loads(json.dumps(item['provenance'])) == draw['provenance']
    assert np.isclose(item['valid_fraction'],
        draw['provenance']['one_shot_slide_artifacts_v3']['pixel_counts']['valid'] / SIDE**2,
        rtol=0, atol=5e-8)
    assert 0 <= item['visible_fraction'] <= 1 and 0 <= item['obliquity_deg'] <= 90
    assert item['inputs'].shape == (5, SIDE, SIDE) and item['state'].shape == (12,)
    assert item['offsets'].shape == item['weights'].shape == (9,)
    indices = item['site_indices']
    valid_count = draw['provenance']['one_shot_slide_artifacts_v3']['pixel_counts']['valid']
    assert indices.ndim == 1 and len(indices) == min(512, valid_count) == len(indices.unique())
    assert bool(((indices >= 0) & (indices < SIDE**2)).all())
    assert item['site_target'].shape == (len(indices), 3) and bool(torch.isfinite(item['site_target']).all())

draws = jsonl(RUN / 'training_draws.jsonl')
training = jsonl(RUN / 'training.jsonl')
readouts = jsonl(RUN / 'readouts.jsonl')
assert len(training) == 2 * UPDATES and len(readouts) == 2 * len(STEPS) * (64 + 98)
train_rng = np.random.default_rng(SEED + 3)
schedule, attempts_at_step = [], {}
for attempt, row in enumerate(draws, 1):
    base = int(train_rng.integers(56))
    subject_index = base * VARIANTS + int(train_rng.integers(VARIANTS))
    draw_seed = SEED + 2_000_000_000 + attempt
    assert (row['step'], row['attempt'], row['base_index'], row['subject_index'], row['draw_seed']) == (
        len(schedule) + 1, attempt, base, subject_index, draw_seed)
    section_id = identity_check(row['provenance'], subject_index, draw_seed, row['used'], synthetic, stage_hashes)
    assert section_id not in all_ids and section_id not in parent_ids
    all_ids.add(section_id)
    if row['used']:
        schedule.append(section_id)
        attempts_at_step[len(schedule)] = attempt
assert len(schedule) == UPDATES
assert receipt['counters'] == {'full_atlas': {'attempted': len(draws), 'accepted': UPDATES},
                               'support_only': {'attempted': UPDATES, 'accepted': UPDATES}}
for arm_index, arm in enumerate(ARMS):
    block = training[arm_index * UPDATES:(arm_index + 1) * UPDATES]
    assert [(row['arm'], row['step'], row['physical_section_id']) for row in block] == [
        (arm, step, section_id) for step, section_id in enumerate(schedule, 1)]
    for row in block:
        chosen = row['selected_blind_indices']
        assert len(chosen) == 4 and len(set(chosen)) == 4 and chosen[0] == 0
        assert all(0 <= index < 16 for index in chosen)
        assert 0 <= row['rank_pairs'] <= 6
        assert row['near_wrong_supplied'] <= row['near_wrong_available']
        assert row['support_matched_near_wrong_update'] <= row['support_matched_wrong_available']
        assert row['support_matched_near_wrong_update'] <= row['support_matched_wrong_in_subset']
        assert row['near_within_1_5mm'] == (row['blind_best_surviving_tissue_mm'] <= 1.5)
        assert all(math.isfinite(row[key]) for key in ('loss', 'rank_loss', 'gradient_norm',
            'exact_no_warp_mm', 'exact_mapped_mm', 'exact_surviving_displacement_mean_mm',
            'exact_surviving_displacement_rms_mm', 'exact_actual_derivative'))
        assert 0 <= row['visible_fraction'] <= 1 and 0 <= row['rank_loss'] <= row['loss']
        assert 0 <= row['blind_best_surviving_tissue_mm'] <= row['blind_top1_surviving_tissue_mm']
        assert 0 <= row['exact_no_warp_mm'] and 0 <= row['exact_mapped_mm']
        assert 0 <= row['gradient_norm'] and 0 <= row['exact_actual_derivative'] <= .35 + 1e-5
        assert row['exact_no_warp_mm'] - row['exact_mapped_mm'] <= row['exact_surviving_displacement_mean_mm'] + 1e-4
        assert row['exact_surviving_displacement_mean_mm'] <= row['exact_surviving_displacement_rms_mm'] + 1e-4

initial, initial_optimizer, trained = {}, {}, {}
pose_keys = {name for name in parent_model
             if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS}
for arm in ARMS:
    for step in STEPS:
        checkpoint = torch.load(RUN / f'{arm}_step_{step:05d}.pt', map_location='cpu', weights_only=False)
        assert checkpoint['candidate'] == 280 and checkpoint['arm'] == arm and checkpoint['step'] == step
        assert json.loads(json.dumps(checkpoint['config'])) == config and checkpoint['calibrated'] is False
        assert checkpoint['accepted'] == step
        assert checkpoint['attempted'] == (attempts_at_step[step] if arm == 'full_atlas' and step else step)
        state = checkpoint['model']
        assert {name for name in state if name.split('.')[0] in POSE_MODULES or name in POSE_BUFFERS} == pose_keys
        for name in pose_keys:
            assert torch.equal(state[name], parent_model[name])
        assert bool(checkpoint['optimizer']['state']) == (step > 0)
        assert len(checkpoint['optimizer']['param_groups']) == 1
        assert checkpoint['optimizer']['param_groups'][0]['lr'] == 1e-4
        assert checkpoint['optimizer']['param_groups'][0]['weight_decay'] == 1e-4
        if step == 0:
            initial[arm] = state
            initial_optimizer[arm] = checkpoint['optimizer']
        if step == UPDATES:
            trained[arm] = any(not torch.equal(initial[arm][name], state[name])
                               for name in state if name.split('.')[0] in TRAIN_MODULES)
        del checkpoint
initial_full, initial_control = initial['full_atlas'], initial['support_only']
assert initial_full.keys() == initial_control.keys()
assert all(torch.equal(initial_full[name], initial_control[name]) for name in initial_full)
assert initial_optimizer['full_atlas'] == initial_optimizer['support_only']
assert all(trained.values())

summaries, frozen_pose, frozen_acquired = {}, {}, {}
offset = 0
for arm in ARMS:
    summaries[arm] = {}
    for step in STEPS:
        rows = readouts[offset:offset + 64]
        real_rows = readouts[offset + 64:offset + 162]
        offset += 162
        assert len(rows) == 64 and len(real_rows) == 98
        assert all(row['arm'] == arm and row['step'] == step and row['panel'] == 'new_v9' for row in rows)
        assert all(row['arm'] == arm and row['step'] == step and row['panel'] == 'acquired_train_orientation'
                   for row in real_rows)
        for row, item in zip(rows, panel):
            for key in ('physical_section_id', 'base_index', 'slot', 'valid_fraction', 'visible_fraction'):
                assert row[key] == item[key]
            assert row['provenance'] == json.loads(json.dumps(item['provenance']))
            assert row['true_obliquity_deg'] == item['obliquity_deg']
            branches = row['branches']
            assert len(branches) == 16
            assert [branch['rank'] for branch in branches] == list(range(16))
            assert len({branch['branch'] for branch in branches}) == 16
            assert all(branch['mode'] == branch['branch'] // 2 and
                       branch['reflection'] == branch['branch'] % 2 and 0 <= branch['branch'] < 160
                       for branch in branches)
            assert all(branches[i]['prior'] >= branches[i + 1]['prior'] for i in range(15))
            for branch in branches:
                assert all(math.isfinite(branch[key]) for key in ('prior', 'fit_score',
                    'original_rigid_plane_mm', 'original_surviving_tissue_mm', 'mapped_mm',
                    'same_grid_no_warp_mm', 'displacement_rms_mm', 'roughness_mm_per_cell',
                    'prelimit_derivative', 'actual_derivative', 'support_fraction', 'obliquity_deg'))
                assert all(branch[key] >= 0 for key in ('original_rigid_plane_mm',
                    'original_surviving_tissue_mm', 'mapped_mm', 'same_grid_no_warp_mm',
                    'displacement_rms_mm', 'roughness_mm_per_cell', 'actual_derivative'))
                assert 0 <= branch['support_fraction'] <= 1 and 0 <= branch['obliquity_deg'] <= 90
                assert branch['actual_derivative'] <= .35 + 1e-5
            rigid = [branch['original_rigid_plane_mm'] for branch in branches]
            physical = [branch['original_surviving_tissue_mm'] for branch in branches]
            scores = [branch['fit_score'] for branch in branches]
            selected, nearest = int(np.argmax(scores)), int(np.argmin(physical))
            assert row['rank_selected_index'] == selected and row['near_index'] == nearest
            assert row['direct_top1_original_rigid_plane_mm'] == rigid[0]
            assert row['direct_top1_surviving_tissue_mm'] == physical[0]
            assert row['rank_selected_original_rigid_plane_mm'] == rigid[selected]
            assert row['rank_selected_original_surviving_tissue_mm'] == physical[selected]
            assert row['rank_selected_mapped_mm'] == branches[selected]['mapped_mm']
            assert row['best_rigid_plane_index_diagnostic_only'] == int(np.argmin(rigid))
            assert row['best_rigid_plane_mm_diagnostic_only'] == min(rigid)
            assert row['physical_best_index_diagnostic_only'] == nearest
            assert row['physical_best_surviving_tissue_mm_diagnostic_only'] == min(physical)
            assert row['near_eligible'] == (physical[nearest] <= 1.5)
            wrong = row['support_matched_wrong_index']
            if wrong is not None:
                assert 0 <= wrong < 16 and wrong != nearest
                assert physical[wrong] + 1e-6 >= physical[nearest] + 1
                assert row['support_matched_dice'] >= .85
                assert row['support_matched_area_gap'] <= .08
                assert row['support_matched_common_cells'] >= 32
            else:
                assert all(row[key] is None for key in ('support_matched_dice',
                    'support_matched_area_gap', 'support_matched_common_cells'))
            eligible = physical[nearest] <= 1.5 and wrong is not None
            assert row['support_matched_pair_eligible'] == eligible
            assert (row['support_matched_near_win'] == (scores[nearest] > scores[wrong])
                    if eligible else row['support_matched_near_win'] is None)
            close = [branch for branch in branches if branch['original_surviving_tissue_mm'] <= .5]
            assert row['near_half_mm_branch_count'] == len(close)
            assert row['near_half_mm_section_mean_map_gain'] == (
                float(np.mean([branch['same_grid_no_warp_mm'] - branch['mapped_mm'] for branch in close]))
                if close else None)
            for key in ('exact_no_warp_mm', 'exact_mapped_mm', 'exact_displacement_rms_mm',
                        'exact_surviving_displacement_mean_mm', 'exact_surviving_displacement_rms_mm',
                        'oracle_surviving_displacement_mean_mm', 'oracle_surviving_displacement_rms_mm',
                        'exact_actual_derivative'):
                assert math.isfinite(row[key]) and row[key] >= 0
            assert row['exact_no_warp_mm'] - row['exact_mapped_mm'] <= row['exact_surviving_displacement_mean_mm'] + 1e-4
            assert row['exact_surviving_displacement_mean_mm'] <= row['exact_surviving_displacement_rms_mm'] + 1e-4
            assert row['oracle_surviving_displacement_mean_mm'] <= row['oracle_surviving_displacement_rms_mm'] + 1e-4
            assert row['exact_actual_derivative'] <= .35 + 1e-5
            pose_signature = [(branch['branch'], branch['prior'], branch['original_rigid_plane_mm'],
                               branch['original_surviving_tissue_mm'], branch['support_fraction'],
                               branch['obliquity_deg']) for branch in branches]
            pose_signature += [(row['near_index'], row['support_matched_wrong_index'],
                                row['support_matched_dice'], row['support_matched_area_gap'],
                                row['support_matched_common_cells'])]
            section_id = row['physical_section_id']
            if section_id in frozen_pose:
                assert pose_signature == frozen_pose[section_id]
            else:
                frozen_pose[section_id] = pose_signature
        for row, case in zip(real_rows, cases):
            for key in ('domain', 'donor_id', 'specimen_id', 'experiment_id',
                        'section_id', 'array_row_index', 'source_image_sha256', 'weak_axis'):
                assert row[key] == case[key]
            assert 0 <= row['selected_branch'] < 160 and 0 <= row['frozen_direct_branch'] < 160
            assert row['selected_axis'] in ('AP', 'DV', 'ML') and row['frozen_direct_axis'] in ('AP', 'DV', 'ML')
            assert row['first_axis_correct'] == (row['selected_axis'] == row['weak_axis'])
            assert 0 <= row['normal_gap_to_weak_affine_deg'] <= 90.001
            assert 0 <= row['frozen_direct_normal_gap_deg'] <= 90.001
            signature = (row['frozen_direct_branch'], row['frozen_direct_axis'],
                         row['frozen_direct_normal_gap_deg'])
            key = (row['domain'], row['donor_id'])
            if key in frozen_acquired:
                assert signature == frozen_acquired[key]
            else:
                frozen_acquired[key] = signature
        result = summary(rows, real_rows)
        assert result == receipt['readouts'][arm][str(step)]
        summaries[arm][step] = result
assert offset == len(readouts) and len(frozen_pose) == 64 and len(frozen_acquired) == 98

coverage = {}
for arm in ARMS:
    block = [row for row in training if row['arm'] == arm]
    c = {'updates': UPDATES,
        'with_rank_pairs': sum(row['rank_pairs'] > 0 for row in block),
        'with_near_versus_wrong': sum(row['near_wrong_supplied'] for row in block),
        'with_near_within_1_5mm': sum(row['near_within_1_5mm'] for row in block),
        'with_support_matched_near_wrong': sum(row['support_matched_near_wrong_update'] for row in block),
        'sparse_under_15pct': sum(row['visible_fraction'] < .15 for row in block),
        'sparse_with_rank_pairs': sum(row['visible_fraction'] < .15 and row['rank_pairs'] > 0 for row in block),
        'sparse_with_near_versus_wrong': sum(row['visible_fraction'] < .15 and row['near_wrong_supplied'] for row in block),
        'sparse_with_support_matched_near_wrong': sum(row['visible_fraction'] < .15 and
            row['support_matched_near_wrong_update'] for row in block)}
    for key in ('with_rank_pairs', 'with_near_versus_wrong', 'with_support_matched_near_wrong'):
        c[key + '_fraction'] = c[key] / UPDATES
    for key in ('sparse_with_rank_pairs', 'sparse_with_near_versus_wrong',
                'sparse_with_support_matched_near_wrong'):
        c[key + '_fraction_of_sparse'] = c[key] / max(1, c['sparse_under_15pct'])
    coverage[arm] = c
assert coverage == receipt['update_coverage']

full, control = summaries['full_atlas'][UPDATES], summaries['support_only'][UPDATES]
assert full['support_matched_sections'] == control['support_matched_sections']
assert full['direct_top1_mean_rigid_plane_mm'] == control['direct_top1_mean_rigid_plane_mm']
full_rate = full['support_matched_near_wins'] / max(1, full['support_matched_sections'])
control_rate = control['support_matched_near_wins'] / max(1, control['support_matched_sections'])
gate = {
    'full_original_gain_vs_frozen_direct_at_least_0_25mm':
        full['direct_top1_mean_rigid_plane_mm'] - full['selected_original_mean_rigid_plane_mm'] >= .25,
    'full_original_gain_vs_trained_support_at_least_0_15mm':
        control['selected_original_mean_rigid_plane_mm'] - full['selected_original_mean_rigid_plane_mm'] >= .15,
    'at_least_six_bases_nonnegative_direct_gain':
        sum(gain >= 0 for gain in full['base_mean_gain_mm'].values()) >= 6,
    'at_least_24_distinct_support_matched_sections': full['support_matched_sections'] >= 24,
    'full_support_matched_near_win_at_least_70pct': full_rate >= .70,
    'full_support_matched_near_win_gain_at_least_10pp': full_rate - control_rate >= .10,
    'exact_map_mean_gain_at_least_0_05mm':
        full['exact_no_warp_mean_mm'] - full['exact_mapped_mean_mm'] >= .05,
    'all_maps_derivative_at_most_0_35': max(full['max_actual_derivative'],
        full['max_blind_actual_derivative']) <= .35 + 1e-5,
    'near_half_mm_map_degradation_at_most_0_05mm':
        full['near_half_mm_mean_map_gain'] is not None and full['near_half_mm_mean_map_gain'] >= -.05,
    'coronal_family_at_least_54_of_58': full['coronal']['first_axis_correct'] >= 54,
    'sagittal_family_at_least_18_of_40': full['sagittal']['first_axis_correct'] >= 18,
    'sagittal_median_weak_angle_rise_at_most_5deg':
        full['sagittal']['median_weak_normal_angle_deg']
        <= full['sagittal']['frozen_direct_median_weak_normal_angle_deg'] + 5,
}
assert receipt['support_matched_full_rate'] == full_rate
assert receipt['support_matched_control_rate'] == control_rate
assert receipt['gate'] == gate
assert receipt['gate_numeric_pass'] == receipt['gate_pass'] == all(gate.values())
assert receipt['gate_status'] == ('positive TRAIN-inner mechanism signal only' if all(gate.values())
                                  else 'failed bounded TRAIN-inner mechanism gate; do not scale')
assert receipt['result_role'] == 'TRAIN-inner bounded mechanism pilot only; no checkpoint promotion'
assert receipt['calibrated'] is False and receipt['dev_cal_final_or_public_benchmark_used'] is False
print(json.dumps({'verified': True, 'gate': gate, 'gate_pass': all(gate.values()),
                  'panel_sections': len(panel), 'paired_updates_per_arm': UPDATES,
                  'readout_rows': len(readouts), 'support_matched_sections': full['support_matched_sections'],
                  'not_replayable_from_saved_rows': [
                      'minimum-error support-matched alternative: other 24x24 masks were not saved',
                      'byte-identical paired inputs: control-arm input hashes were checked in runner but not saved',
                      'branch score/map values: model inference was not rerendered by this verifier']}))
