"""Read-only audit of 279; execute only after the training runner has exited."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    ANNOTATION_PATH_V6, ANNOTATION_RAW_SHA256_V6,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/anatomy_contrast_279_pilot_001'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
OLD = ROOT / 'runs/pose_capture_270_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
PREVIEW = ROOT / 'previews/anatomy_contrast_278_actual_generator_001'
CENSUS = ROOT / 'runs/anatomy_contrast_278_texture_census_001'
PROTOCOL = SOURCE.parent / 'docs/publication/ANATOMY_CONTRAST_279_PILOT_PROTOCOL_20261012.md'
ARMS = {'control_v9': 2026101227901, 'anatomy_v11': 2026201227901}
STEPS = (0, 500, 1000)
PANELS = ('new_275_v9', 'frozen_270_v9', 'coronal', 'sagittal')
SOURCE_NAMES = (
    'train_pose_anatomy_contrast_279.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v11.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_full_frame_primitives.py',
    'global_atlas_contrast_090.py', 'arbitrary_plane_allen_atlas_binding_v6.py',
)

receipt = json.loads((RUN / 'completed.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
parent_receipt = json.loads((PARENT / 'completed.json').read_text())
old_receipt = json.loads((OLD / 'completed.json').read_text())
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
transfer_config = json.loads((TRANSFER / 'config.json').read_text())
preview_receipt = json.loads((PREVIEW / 'completed.json').read_text())
preview_selection = json.loads((PREVIEW / 'selection.json').read_text())
census_receipt = json.loads((CENSUS / 'completed.json').read_text())
assert receipt['candidate'] == config['candidate'] == 279
assert receipt['steps'] == config['readout_batches'] == list(STEPS)
assert config['arms'] == ARMS and config['optimizer_batches_per_arm'] == 1000
assert set(receipt['counters']) == set(receipt['readouts']) == set(ARMS)
assert all(set(receipt['readouts'][arm]) == {'0', '500', '1000'} for arm in ARMS)
assert all(set(receipt['readouts'][arm][str(step)]) == set(PANELS)
           for arm in ARMS for step in STEPS)
assert config['batch_size'] == 4 and config['learning_rate'] == 3e-5
assert config['gradient_base_indices'] == list(range(56))
assert config['inner_base_indices'] == list(range(56, 64))
assert config['parent_selected_step'] == 8000
assert config['gate_checkpoint'] == 1000 and config['checkpoint_selection'] == 'none'
assert not config['weak_affines_or_acquired_images_used_for_gradients']
assert config['no_prior_external_weights_features_pseudolabels']
assert config['no_dev_cal_final_or_public_benchmark']
assert not receipt['weak_affines_or_acquired_images_used_for_gradients']
assert not receipt['calibrated'] and not receipt['dev_cal_final_or_public_benchmark_used']
assert receipt['result_role'] == 'bounded TRAIN-inner mechanism pilot; pass permits confirmation only, not GUI promotion'

assert config['protocol_sha256'] == receipt['protocol_sha256'] == sha256(PROTOCOL)
assert set(config['source_sha256']) == set(SOURCE_NAMES)
assert receipt['source_sha256'] == config['source_sha256']
for name in SOURCE_NAMES:
    assert sha256(SOURCE / name) == config['source_sha256'][name]
assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
assert old_receipt['candidate'] == 270 and transfer_receipt['candidate'] == 272
assert config['parent_completed_sha256'] == sha256(PARENT / 'completed.json')
assert config['parent_config_sha256'] == parent_receipt['config_sha256'] == sha256(PARENT / 'config.json')
assert (config['parent_checkpoint_sha256'] == receipt['parent_checkpoint_sha256']
        == parent_receipt['checkpoint_sha256']['8000'] == sha256(PARENT / 'pose_step_08000.pt'))
assert config['new_275_panel_sha256'] == parent_receipt['inner_panel_sha256'] == sha256(PARENT / 'inner_panel.pt')
assert config['frozen_270_panel_sha256'] == old_receipt['inner_panel_sha256'] == sha256(OLD / 'inner_panel.pt')
assert config['transfer_cases_sha256'] == transfer_receipt['cases_sha256'] == sha256(TRANSFER / 'cases.jsonl')
assert transfer_receipt['config_sha256'] == sha256(TRANSFER / 'config.json')
assert config['preview_278_selection_sha256'] == preview_receipt['selection_sha256'] == sha256(PREVIEW / 'selection.json')
assert config['preview_278_image_sha256'] == preview_receipt['image_sha256'] == sha256(PREVIEW / 'atlas_source_input.png')
assert config['census_278_rows_sha256'] == census_receipt['rows_sha256'] == sha256(CENSUS / 'rows.jsonl')
assert config['census_278_summary_sha256'] == census_receipt['summary_sha256'] == sha256(CENSUS / 'summary.json')
assert (preview_selection['generator_sha256'] == census_receipt['generator_sha256']
        == config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v11.py'])
assert census_receipt['preview_selection_sha256'] == preview_receipt['selection_sha256']
assert config['annotation_path'] == str(ANNOTATION_PATH_V6)
assert config['annotation_raw_sha256'] == ANNOTATION_RAW_SHA256_V6 == sha256(ANNOTATION_PATH_V6)

assert set(receipt['output_sha256']) == {'config.json', 'draws.jsonl', 'training.jsonl', 'readouts.jsonl'}
for name, digest in receipt['output_sha256'].items():
    assert sha256(RUN / name) == digest
assert set(receipt['checkpoint_sha256']) == {f'{arm}_{step}' for arm in ARMS for step in STEPS[1:]}
for key, digest in receipt['checkpoint_sha256'].items():
    arm, step = key.rsplit('_', 1)
    assert sha256(RUN / f'{arm}_step_{int(step):05d}.pt') == digest

synthetic = config['synthetic_provenance']
assert synthetic['new_independent_train_bases'] == 56 and synthetic['all_train_bases'] == 64
assert len(synthetic['base_subjects']) == 64
assert len(synthetic['virtual_subjects']) == 64 * VARIANTS
assert synthetic['atlas_binding']['annotation_path'] == config['annotation_path']
assert synthetic['atlas_binding']['annotation_sha256'] == config['annotation_raw_sha256']
for name, digest in synthetic['source_sha256'].items():
    assert sha256(SOURCE / name) == digest
assert synthetic['source_sha256']['arbitrary_plane_streaming_synthetic_v7_64.py'] == config['source_sha256']['arbitrary_plane_streaming_synthetic_v7_64.py']
for path, digest in synthetic['input_sha256'].items():
    assert sha256(Path(path)) == digest
for index, subject in enumerate(synthetic['virtual_subjects']):
    assert subject['virtual_index'] == index
    assert subject['base_index'] == index // VARIANTS and subject['variant'] == index % VARIANTS
    assert subject['base_lineage'] == synthetic['base_subjects'][index // VARIANTS]['lineage']
    assert subject['base_lineage']['split'] == 'train'

sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'animal_id', 'training_split', 'model_pixel_to_ap_dv_ml_um', 58),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'donor_id', 'split', 'model_pixel_to_ccf_ref9_ap_dv_ml_um', 40),
}
assert set(config['source_data']) == set(sources)
transfer_cases = [json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()]
assert len(transfer_cases) == 98
cases_by_domain, source_records, source_images = {}, {}, {}
for domain, (records_path, images_path, donor_key, split_key, affine_key, count) in sources.items():
    assert config['source_data'][domain] == {
        'records_sha256': sha256(records_path), 'images_sha256': sha256(images_path)}
    assert config['source_data'][domain] == {
        'records_sha256': transfer_config['datasets'][domain]['records_sha256'],
        'images_sha256': transfer_config['datasets'][domain]['images_sha256']}
    records = [json.loads(line) for line in records_path.open()]
    source_records[domain] = {record['array_row_index']: record for record in records}
    source_images[domain] = images_path
    eligible = [record for record in records if record[split_key] == 'train']
    donors = sorted({record[donor_key] for record in eligible})
    selected = []
    for donor in donors:
        cohort = sorted((record for record in eligible if record[donor_key] == donor),
                        key=lambda record: (record['section_number'], record['section_id']))
        selected.append(cohort[(len(cohort) - 1) // 2])
    cases = [case for case in transfer_cases if case['domain'] == domain]
    assert len(donors) == len(cases) == count == transfer_config['datasets'][domain]['donors']
    assert len({case['donor_id'] for case in cases}) == count
    for case, record in zip(cases, selected):
        assert case['donor_id'] == record[donor_key]
        for key in ('specimen_id', 'experiment_id', 'section_id', 'array_row_index'):
            assert case[key] == record[key]
        assert case['source_image_sha256'] == record.get('source_image_sha256', record.get('image_sha256'))
        normal = np.cross(np.asarray(record[affine_key])[:, 0], np.asarray(record[affine_key])[:, 1])
        assert case['weak_axis'] == ('AP', 'DV', 'ML')[int(np.abs(normal).argmax())]
        assert case['weak_axis'] == ('AP' if domain == 'coronal' else 'ML')
    cases_by_domain[domain] = cases
assert [case['domain'] for case in transfer_cases] == ['coronal'] * 58 + ['sagittal'] * 40

panels = {
    'new_275_v9': torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False),
    'frozen_270_v9': torch.load(OLD / 'inner_panel.pt', map_location='cpu', weights_only=False),
}
panel_ids = set()
for panel in panels.values():
    assert len(panel) == 64
    assert [item['base_index'] for item in panel] == [base for base in range(56, 64) for _ in range(8)]
    for item in panel:
        assert item['provenance']['physical_section_id'] == item['physical_section_id']
        assert item['provenance']['virtual_index'] // VARIANTS == item['base_index']
        assert item['physical_section_id'] not in panel_ids
        panel_ids.add(item['physical_section_id'])

draws = [json.loads(line) for line in (RUN / 'draws.jsonl').open()]
training = [json.loads(line) for line in (RUN / 'training.jsonl').open()]
readouts = [json.loads(line) for line in (RUN / 'readouts.jsonl').open()]
assert len(training) == 2 * 1000 and len(readouts) == 2 * len(STEPS) * len(PANELS)
assert [(row['arm'], row['step'], row['panel']) for row in readouts] == [
    (arm, step, panel) for arm in ARMS for step in STEPS for panel in PANELS]
assert [row['arm'] for row in training] == [arm for arm in ARMS for _ in range(1000)]
assert [row['arm'] for row in draws] == [arm for arm in ARMS
                                       for _ in range(receipt['counters'][arm]['attempted'])]
stage_hashes = {f'one_shot_slide_artifacts_v{version}': sha256(
    SOURCE / f'arbitrary_plane_one_shot_slide_artifacts_v{version}.py')
    for version in (3, 4, 6, 7)}
draw_ids = set()
for arm, seed in ARMS.items():
    arm_draws = [row for row in draws if row['arm'] == arm]
    arm_training = [row for row in training if row['arm'] == arm]
    rng = np.random.default_rng(seed)
    accepted, attempted_at_step, rng_at_step = 0, {}, {}
    for attempt, row in enumerate(arm_draws, 1):
        base, variant = int(rng.integers(56)), int(rng.integers(VARIANTS))
        subject_index, draw_seed = base * VARIANTS + variant, seed + 2_000_000_000 + attempt
        provenance = row['provenance']
        assert row['attempt'] == attempt and row['step'] == accepted // 4 + 1
        assert row['base_index'] == base and row['variant'] == variant
        assert row['subject_index'] == subject_index and row['draw_seed'] == draw_seed
        assert row['used'] == provenance['eligible'] == (provenance['visible_support_mass'] >= 192 * 192 / 144)
        assert provenance['seed_prefix'] == [draw_seed, subject_index, 0]
        assert provenance['virtual_index'] == subject_index
        assert provenance['virtual_subject_id'] == synthetic['virtual_subjects'][subject_index]['virtual_subject_id']
        assert provenance['base_lineage'] == synthetic['base_subjects'][base]['lineage']
        for stage, digest in stage_hashes.items():
            assert provenance[stage]['code_sha256'] == digest
        assert provenance['one_shot_slide_artifacts_v6']['source_plane_reused_for_appearance_pair'] is False
        physical_id = f'stream-v7-seed-{draw_seed}-row-0-virtual-{subject_index}'
        assert provenance['physical_section_id'] == physical_id
        assert physical_id not in draw_ids and physical_id not in panel_ids
        draw_ids.add(physical_id)
        if arm == 'control_v9':
            assert 'one_shot_slide_artifacts_v11' not in provenance
            exterior = provenance['one_shot_slide_artifacts_v9']
            assert exterior['code_sha256'] == config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v9.py']
            assert exterior['seed_branch'] == 34
            assert exterior['exterior_style'] in ('exact_black', 'near_black', 'gray', 'exact_black_brush')
        else:
            assert 'one_shot_slide_artifacts_v9' not in provenance
            anatomy = provenance['one_shot_slide_artifacts_v11']
            assert anatomy['code_sha256'] == config['source_sha256']['arbitrary_plane_one_shot_slide_artifacts_v11.py']
            assert anatomy['annotation_raw_sha256'] == config['annotation_raw_sha256']
            assert anatomy['seed_branch'] == 37 and anatomy['exterior']['seed_branch'] == 34
            assert anatomy['exterior']['style'] in ('exact_black', 'near_black', 'gray', 'exact_black_brush')
            assert .25 <= anatomy['contrast_strength'] <= .8
            assert .012 <= anatomy['texture_strength'] <= .075
            contrast_rng = np.random.default_rng(np.random.SeedSequence([draw_seed, subject_index, 0, 37]))
            region_codes = contrast_rng.normal(0, .65, anatomy['region_code_count']).astype(np.float32)
            assert hashlib.sha256(region_codes.tobytes()).hexdigest() == anatomy['region_code_sha256']
            assert float(contrast_rng.uniform(.25, .8)) == anatomy['contrast_strength']
            assert float(contrast_rng.uniform(.012, .075)) == anatomy['texture_strength']
            assert int(contrast_rng.integers(0, 2**63)) == anatomy['texture_seed']
        if row['used']:
            accepted += 1
            if accepted % 4 == 0:
                step = accepted // 4
                attempted_at_step[step] = attempt
                if step in STEPS[1:]:
                    rng_at_step[step] = copy.deepcopy(rng.bit_generator.state)
    assert len(arm_draws) == receipt['counters'][arm]['attempted']
    assert accepted == receipt['counters'][arm]['accepted'] == 4000
    assert [row['step'] for row in arm_training] == list(range(1, 1001))
    for row in arm_training:
        step = row['step']
        assert row['attempted'] == attempted_at_step[step] and row['accepted'] == step * 4
        assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
    for step in STEPS[1:]:
        checkpoint = torch.load(RUN / f'{arm}_step_{step:05d}.pt', map_location='cpu', weights_only=False)
        assert checkpoint['arm'] == arm and checkpoint['step'] == step
        assert checkpoint['config'] == config and not checkpoint['calibrated']
        assert checkpoint['attempted'] == attempted_at_step[step] and checkpoint['accepted'] == 4 * step
        assert checkpoint['subject_rng'] == rng_at_step[step]
        del checkpoint

for block in readouts:
    arm, step, panel_name = block['arm'], block['step'], block['panel']
    rows, metrics = block['rows'], block['metrics']
    assert metrics == receipt['readouts'][arm][str(step)][panel_name]
    if panel_name in panels:
        panel = panels[panel_name]
        assert len(rows) == metrics['n'] == 64
        for row, item in zip(rows, panel):
            assert row['physical_section_id'] == item['physical_section_id']
            assert row['base_index'] == item['base_index']
            assert row['visible_fraction'] == item['visible_fraction']
            assert isinstance(row['top16_hit'], bool)
            assert math.isfinite(row['top1_mm']) and row['top1_mm'] >= 0
            assert math.isfinite(row['normal_axis_error_deg']) and 0 <= row['normal_axis_error_deg'] <= 90.001
        assert metrics['top16_hits'] == sum(row['top16_hit'] for row in rows)
        assert metrics['mean_top1_mm'] == float(np.mean([row['top1_mm'] for row in rows]))
        assert metrics['mean_normal_axis_error_deg'] == float(np.mean([
            row['normal_axis_error_deg'] for row in rows]))
    else:
        cases = cases_by_domain[panel_name]
        assert len(rows) == metrics['total']['n'] == len(cases)
        for row, case in zip(rows, cases):
            for key in ('donor_id', 'specimen_id', 'experiment_id', 'section_id',
                        'array_row_index', 'source_image_sha256', 'weak_axis'):
                assert row[key] == case[key]
            parity = 'even' if hashlib.sha256(f'{panel_name}|{case["donor_id"]}'.encode()).digest()[0] % 2 == 0 else 'odd'
            assert row['donor_parity'] == parity
            assert row['selected_axis'] in ('AP', 'DV', 'ML')
            assert row['first_axis_correct'] == (row['selected_axis'] == row['weak_axis'])
            assert math.isfinite(row['normal_gap_to_weak_affine_deg'])
            assert 0 <= row['normal_gap_to_weak_affine_deg'] <= 90.001
        for parity in ('total', 'even', 'odd'):
            subset = rows if parity == 'total' else [row for row in rows if row['donor_parity'] == parity]
            assert metrics[parity] == {'n': len(subset),
                'first_axis_correct': sum(row['first_axis_correct'] for row in subset),
                'median_weak_normal_angle_deg': float(np.median([
                    row['normal_gap_to_weak_affine_deg'] for row in subset]))}
        assert (metrics['even']['n'], metrics['odd']['n']) == (
            (30, 28) if panel_name == 'coronal' else (18, 22))

assert receipt['readouts']['control_v9']['0'] == receipt['readouts']['anatomy_v11']['0']
control = receipt['readouts']['control_v9']['1000']
anatomy = receipt['readouts']['anatomy_v11']['1000']
deltas = {
    'new_275_mean_top1_drop_mm': control['new_275_v9']['mean_top1_mm'] - anatomy['new_275_v9']['mean_top1_mm'],
    'new_275_top16_loss': control['new_275_v9']['top16_hits'] - anatomy['new_275_v9']['top16_hits'],
    'frozen_270_mean_top1_worsening_mm': anatomy['frozen_270_v9']['mean_top1_mm'] - control['frozen_270_v9']['mean_top1_mm'],
    'sagittal_total_family_gain': anatomy['sagittal']['total']['first_axis_correct'] - control['sagittal']['total']['first_axis_correct'],
    'sagittal_odd_family_gain': anatomy['sagittal']['odd']['first_axis_correct'] - control['sagittal']['odd']['first_axis_correct'],
    'sagittal_even_family_gain': anatomy['sagittal']['even']['first_axis_correct'] - control['sagittal']['even']['first_axis_correct'],
    'sagittal_median_weak_angle_drop_deg': control['sagittal']['total']['median_weak_normal_angle_deg'] - anatomy['sagittal']['total']['median_weak_normal_angle_deg'],
    'coronal_total_family_loss': control['coronal']['total']['first_axis_correct'] - anatomy['coronal']['total']['first_axis_correct'],
}
gate = {
    'new_275_mean_top1_drop_at_least_0_20mm': deltas['new_275_mean_top1_drop_mm'] >= .20,
    'new_275_top16_loss_at_most_2': deltas['new_275_top16_loss'] <= 2,
    'frozen_270_mean_top1_worsening_at_most_0_10mm': deltas['frozen_270_mean_top1_worsening_mm'] <= .10,
    'sagittal_total_family_gain_at_least_6': deltas['sagittal_total_family_gain'] >= 6,
    'sagittal_odd_family_gain_at_least_2': deltas['sagittal_odd_family_gain'] >= 2,
    'sagittal_even_family_gain_at_least_2': deltas['sagittal_even_family_gain'] >= 2,
    'sagittal_median_weak_angle_drop_at_least_5deg': deltas['sagittal_median_weak_angle_drop_deg'] >= 5,
    'coronal_total_family_loss_at_most_2': deltas['coronal_total_family_loss'] <= 2,
}
assert deltas == receipt['gate_deltas'] and gate == receipt['gate']
assert all(gate.values()) == receipt['gate_pass']

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_checkpoint = torch.load(PARENT / 'pose_step_08000.pt', map_location='cpu', weights_only=False)
assert parent_checkpoint['step'] == 8000 and parent_checkpoint['config']['candidate'] == 275
assert parent_checkpoint['config'] == json.loads((PARENT / 'config.json').read_text())
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).to(device).eval()
replayed = 0
with torch.inference_mode():
    for arm in ARMS:
        for step in STEPS:
            checkpoint = (parent_checkpoint if step == 0 else torch.load(
                RUN / f'{arm}_step_{step:05d}.pt', map_location='cpu', weights_only=False))
            model.load_state_dict(checkpoint['model'])
            for panel_name, panel in panels.items():
                saved = next(block['rows'] for block in readouts if (
                    block['arm'], block['step'], block['panel']) == (arm, step, panel_name))
                for index in (0, 32):
                    item = panel[index]
                    prediction = model.predict(item['inputs'][None].to(device))
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                    order = prior.argsort(descending=True)
                    chart = item['chart'].to(device)
                    truth = rigid_points_090(item['state'][None].to(device),
                        item['reflection'][None].to(device), chart, (192, 192))
                    points = rigid_points_090(prediction['state'][0, order[:16] // 2][None],
                        (order[:16] % 2)[None], chart[None], (192, 192))
                    errors = ((points - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
                    true_normal = full_frame_state_to_components(item['state'].to(device))[1][:, 2]
                    chosen_normal = full_frame_state_to_components(
                        prediction['state'][0, order[0] // 2])[1][:, 2]
                    angle = float(torch.rad2deg(torch.acos((true_normal * chosen_normal)
                                                 .sum().abs().clamp(0, 1))))
                    assert bool((errors <= 1.5).any()) == saved[index]['top16_hit']
                    assert abs(float(errors[0]) - saved[index]['top1_mm']) < 1e-4
                    assert abs(angle - saved[index]['normal_axis_error_deg']) < 1e-3
                    replayed += 1
            for domain in ('coronal', 'sagittal'):
                cases = cases_by_domain[domain]
                saved = next(block['rows'] for block in readouts if (
                    block['arm'], block['step'], block['panel']) == (arm, step, domain))
                images = np.load(source_images[domain], mmap_mode='r')
                affine_key = sources[domain][4]
                for index in (0, len(cases) // 2):
                    case = cases[index]
                    image = torch.from_numpy(np.asarray(images[case['array_row_index'], 0],
                                                  dtype=np.float32).copy())[None, None].to(device)
                    if domain == 'sagittal':
                        image = F.interpolate(image, (192, 192), mode='area')
                    network_input = torch.cat((image, torch.zeros(1, 4, 192, 192, device=device)), 1)
                    prediction = model.predict(network_input)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                    first = int(prior.argmax()) // 2
                    normal = full_frame_state_to_components(prediction['state'][0, first])[1][:, 2]
                    matrix = np.asarray(source_records[domain][case['array_row_index']][affine_key], dtype=np.float64)
                    weak_normal = np.cross(matrix[:, 0], matrix[:, 1])
                    weak_normal /= np.linalg.norm(weak_normal)
                    angle = float(torch.rad2deg(torch.acos((normal * torch.as_tensor(
                        weak_normal, device=device)).sum().abs().clamp(0, 1))))
                    selected_axis = ('AP', 'DV', 'ML')[int(normal.abs().argmax())]
                    assert selected_axis == saved[index]['selected_axis']
                    assert abs(angle - saved[index]['normal_gap_to_weak_affine_deg']) < 1e-3
                    replayed += 1
                del images
            if step:
                del checkpoint

print(json.dumps({'verified': True, 'attempted_draws': len(draws),
                  'accepted_per_arm': 4000, 'optimizer_batches_per_arm': 1000,
                  'readout_blocks': len(readouts), 'replayed_predictions': replayed,
                  'gate': gate, 'gate_pass': all(gate.values())}))
