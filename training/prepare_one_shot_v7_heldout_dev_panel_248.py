"""CPU-only, independent v7 synthetic DEV panel; run after the 247 pilot."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_acquisition_v2 as acquisition
from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle
from training.arbitrary_plane_one_shot_slide_artifacts_v7 import sample_one_shot_slide_artifacts_v7
from training.arbitrary_plane_streaming_synthetic_v7 import SUPPORT_SEED, VARIANTS, VIRTUAL_SEED
from training.arbitrary_plane_subject_sampling_v6 import subject_support_bounds_um
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6

OLD = ROOT / 'data/joint_v6_coherent_subject_plans_002'
TRAIN_MAPS = ROOT / 'data/joint_v7_independent_subject_maps_001'
TRAIN_GEOMETRY = ROOT / 'data/joint_v7_training_data_001'
PRIOR_DEV = tuple(ROOT / 'data' / name for name in (
    'one_shot_fresh_synthetic_dev_plans_001',
    'pose_feedback_037_fresh_synthetic_dev_plans_001',
    'pose_feedback_061_fresh_synthetic_dev_plans_001',
    'contextual_match_099_fresh_synthetic_dev_plans_001',
    'contextual_localization_102_fresh_dev_plans_001',
    'cross_candidate_match_103_fresh_dev_plans_001',
    'v3_pose_capture_confirmation_plans_001',
    'matcher_150_fresh_dev_plans'))
OUT = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_001'
PROTOCOL_DOC = Path('docs/publication/ONE_SHOT_V7_HELDOUT_DEV_PANEL_248_PROTOCOL_20261012.md')
SEED, SIDE, SUBJECTS, PLANES = 2026101248, 192, 8, 32
PLAN_SEED, FIRST_INDEX = 2026101249, 2000
CONTEXT_SHA = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
OLD_SHA = '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
assert not torch.cuda.is_available()

old_bytes = (OLD / 'completed.json').read_bytes()
assert hashlib.sha256(old_bytes).hexdigest() == OLD_SHA
old = json.loads(old_bytes)
train_maps_path = TRAIN_MAPS / 'completed.json'
train_maps = json.loads(train_maps_path.read_text())
assert train_maps['parent_plan_completed_sha256'] == OLD_SHA
assert train_maps['total_independent_train_deformations'] == 64
assert len(train_maps['new_train_subjects']) == 56
prior = [(root, json.loads((root / 'completed.json').read_text())) for root in PRIOR_DEV]
assert all(len(index['subjects']) == 8 for _, index in prior)
context_bytes = (OLD / 'context.json').read_bytes()
assert hashlib.sha256(context_bytes).hexdigest() == old['artifact_sha256']['context.json']
parent_context = json.loads(context_bytes)
assert old['context_sha256'] == parent_context['v2_context_sha256'] == CONTEXT_SHA
lower = np.asarray(parent_context['full_ccf_lower_ap_dv_ml_um'], dtype=np.float64)
upper = np.asarray(parent_context['full_ccf_upper_ap_dv_ml_um'], dtype=np.float64)

seen_ids, seen_realizations, seen_receipts, seen_local = set(), set(), set(), set()
for root, rows in [(OLD, old['subjects'])] + [(root, index['subjects']) for root, index in prior]:
    for row in rows:
        for name, expected in row['artifact_sha256'].items():
            assert bundle._file_sha256(root / row['directory'] / name) == expected
        seen_ids.add(row['subject_deformation_plan_id'])
        seen_realizations.add(row['subject_deformation_realization_id'])
        seen_receipts.add(row['subject_plan_receipt_sha256'])
        previous = bundle._read_raw_artifact(root / row['directory'], row['plan_files'])
        assert previous['subject_deformation_plan_id'] == row['subject_deformation_plan_id']
        assert previous['subject_deformation_realization_id'] == row['subject_deformation_realization_id']
        assert previous['receipt_sha256'] == row['subject_plan_receipt_sha256']
        fields = (previous['state']['accepted_coarse_coefficients_um'],
                  previous['state']['accepted_fine_coefficients_um'])
        seen_local.add(hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest())
for row in train_maps['new_train_subjects']:
    assert row['split'] == 'train'
    directory = TRAIN_MAPS / row['directory']
    assert bundle._file_sha256(directory / 'completed.json') == row['completed_sha256']
    for name, expected in row['artifact_sha256'].items():
        assert bundle._file_sha256(directory / name) == expected
    previous = bundle._read_raw_artifact(directory, row['plan_files'])
    assert previous['subject_deformation_plan_id'] == row['subject_deformation_plan_id']
    assert previous['subject_deformation_realization_id'] == row['subject_deformation_realization_id']
    assert previous['receipt_sha256'] == row['subject_plan_receipt_sha256']
    fields = (previous['state']['accepted_coarse_coefficients_um'],
              previous['state']['accepted_fine_coefficients_um'])
    assert hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest() == row['local_field_sha256']
    seen_ids.add(row['subject_deformation_plan_id'])
    seen_realizations.add(row['subject_deformation_realization_id'])
    seen_receipts.add(row['subject_plan_receipt_sha256'])
    seen_local.add(row['local_field_sha256'])
assert len(seen_ids) == 12 + 8 * len(PRIOR_DEV) + 56
assert len(seen_realizations) == len(seen_receipts) == len(seen_local) == len(seen_ids)

geometry_path = TRAIN_GEOMETRY / 'train_acquisition_geometry.jsonl'
geometry_bytes = geometry_path.read_bytes()
map_protocol = json.loads((TRAIN_GEOMETRY / 'protocol.json').read_text())
assert hashlib.sha256(geometry_bytes).hexdigest() == map_protocol['geometry_source']['sha256']
geometry = [json.loads(line) for line in geometry_bytes.splitlines()]
assert all(row['split'] == 'development_train' for row in geometry)
donors = sorted({row['animal_id'] for row in geometry})
donor_rows = [np.array([i for i, row in enumerate(geometry) if row['animal_id'] == donor]) for donor in donors]
affine = np.array([row['model_pixel_to_ap_dv_ml_um'] for row in geometry])
u, v = affine[:, :, 0] * 96, affine[:, :, 1] * 96
lu = np.linalg.norm(u, axis=1)
u /= lu[:, None]
along = (u * v).sum(1)
v -= along[:, None] * u
lv = np.linalg.norm(v, axis=1)
v /= lv[:, None]
atlas_centre = np.array(allen.ATLAS_SHAPE_AP_DV_ML_V6) * 25. / 2
image_centre = affine[:, :, 2] + 48 * (affine[:, :, 0] + affine[:, :, 1])
nuisance = np.column_stack(((u * (image_centre - atlas_centre)).sum(1),
    (v * (image_centre - atlas_centre)).sum(1), lu, lv, along / lv))

source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) | {
    Path(__file__).name, 'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_streaming_synthetic_v7.py', 'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py', 'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py', 'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_subject_sampling_v6.py', 'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_full_frame_primitives.py', 'subject_deformed_slab_multiresolution_bundle_v2.py'})
protocol = {
    'split': 'development', 'seed': SEED, 'plan_seed': PLAN_SEED,
    'subject_indices': list(range(FIRST_INDEX, FIRST_INDEX + SUBJECTS)),
    'subjects': SUBJECTS, 'planes_per_subject': PLANES, 'side': SIDE,
    'selection': 'all independently drawn physical planes retained; one independent v7 appearance/artifact draw each; no paired repeats or eligibility retries',
    'sampling': 'unchanged v7 appearance-v3 and one-shot artifact-v3/v4/v6/v7 functions; 64 derived affine variants per new 3D base; train-hardcoded provenance split relabeled development',
    'map': 'accepted FP64 inverse RK4 baked at 100um on CPU; 50um if fixed 4096-query sampled max exceeds 5um; same FP32 interpolation as TRAIN',
    'eligibility': 'source visible support and observed valid pixels each >= side^2/144; ineligible rows retained',
    'old_plan_completed_sha256': OLD_SHA,
    'train_map_completed_sha256': bundle._file_sha256(train_maps_path),
    'prior_dev_plan_completed_sha256': {root.name: bundle._file_sha256(root / 'completed.json')
                                        for root in PRIOR_DEV},
    'parent_context_sha256': hashlib.sha256(context_bytes).hexdigest(),
    'train_geometry_sha256': hashlib.sha256(geometry_bytes).hexdigest(),
    'train_map_protocol_sha256': bundle._file_sha256(TRAIN_GEOMETRY / 'protocol.json'),
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'protocol_document_sha256': bundle._file_sha256(PROTOCOL_DOC),
    'source_sha256': {name: bundle._file_sha256(repository / 'training' / name) for name in source_names},
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
    'scope': 'synthetic DEV from one Allen atlas and TRAIN-only acquisition framing; not biological validation, CAL, FINAL, or public benchmark',
}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
del annotation
support = np.flatnonzero(atlas_array[1] > 0)
selected = np.random.default_rng(SUPPORT_SEED).choice(support, 4096, replace=False)
support_ccf = (np.array(np.unravel_index(selected, atlas_array.shape[1:])).T.astype(np.float64) + .5) * 25.
support_sha = hashlib.sha256(support_ccf.tobytes()).hexdigest()
atlas = torch.from_numpy(atlas_array)
del atlas_array, support

tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
               'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um',
               'global_scale', 'frozen_center_um')
records, subject_records = [], []
for ordinal, index in enumerate(protocol['subject_indices']):
    prefix = f'one-shot-v7-heldout-dev-248-subject-{index:08d}'
    lineage = {'split': 'development', 'animal_index': index, 'subject_id': prefix,
               'animal_id': prefix, 'specimen_id': f'{prefix}-specimen',
               'experiment_id': f'{prefix}-experiment'}
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=PLAN_SEED, split='development', animal_index=index,
        animal_id=lineage['animal_id'], ccf_context_sha256=CONTEXT_SHA)
    fields = (plan['state']['accepted_coarse_coefficients_um'], plan['state']['accepted_fine_coefficients_um'])
    local_sha = hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest()
    assert plan['subject_deformation_plan_id'] not in seen_ids
    assert plan['subject_deformation_realization_id'] not in seen_realizations
    assert plan['receipt_sha256'] not in seen_receipts and local_sha not in seen_local
    seen_ids.add(plan['subject_deformation_plan_id'])
    seen_realizations.add(plan['subject_deformation_realization_id'])
    seen_receipts.add(plan['receipt_sha256'])
    seen_local.add(local_sha)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt['receipt_sha256'] == plan['receipt_sha256']
    directory = OUT / f'subject_{index:08d}'
    directory.mkdir()
    plan_files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    parameters = [torch.as_tensor(plan['state'][key], dtype=torch.float64) for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])
    map_centre, map_scale = plan['state']['frozen_center_um'], plan['state']['global_scale']
    forward_support = map_accepted_subject_points_torch_v6(torch.from_numpy(support_ccf), *parameters,
        inverse=False, steps=steps).numpy()
    mapped_centre = map_accepted_subject_points_torch_v6(torch.as_tensor(atlas_centre[None], dtype=torch.float64),
        *parameters, inverse=False, steps=steps).numpy()[0]
    support_bounds = []
    for level in ('coarse', 'fine'):
        coefficients = plan['state'][f'accepted_{level}_coefficients_um']
        nonzero = np.array(np.nonzero(np.any(coefficients != 0, axis=-1)))
        origin, spacing = plan['state'][f'{level}_origin_um'], plan['state'][f'{level}_spacing_um']
        support_bounds.append(np.stack((origin + (nonzero.min(1) - 2) * spacing,
                                        origin + (nonzero.max(1) + 2) * spacing)))
    bounds = np.stack((np.min(np.array(support_bounds)[:, 0], axis=0) - 100,
                       np.max(np.array(support_bounds)[:, 1], axis=0) + 100))
    check_rng = np.random.default_rng(np.random.SeedSequence([PLAN_SEED, index, 3]))
    check_unscaled = np.concatenate((check_rng.uniform(lower, upper, (2048, 3)),
                                     check_rng.uniform(bounds[0], bounds[1], (2048, 3))))
    check_subject = map_centre + map_scale * (check_unscaled - map_centre)
    check_exact = map_accepted_subject_points_torch_v6(torch.from_numpy(check_subject), *parameters,
        inverse=True, steps=steps).numpy()
    checks = []
    for spacing in (100., 50.):
        axes = [np.arange(bounds[0, k], bounds[1, k] + spacing, spacing) for k in range(3)]
        grid_lower, grid_upper = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
        shape = tuple(len(a) for a in axes)
        baked = np.empty((*shape, 3), dtype=np.float32)
        for start in range(0, shape[0], 4):
            points = np.stack(np.meshgrid(axes[0][start:start + 4], axes[1], axes[2], indexing='ij'), -1)
            subject_points = map_centre + map_scale * (points - map_centre)
            exact = map_accepted_subject_points_torch_v6(torch.from_numpy(subject_points), *parameters,
                inverse=True, steps=steps).numpy()
            baked[start:start + 4] = exact - points
        displacement = torch.from_numpy(np.moveaxis(baked, -1, 0).copy())
        grid = torch.as_tensor((check_unscaled - grid_lower) / (grid_upper - grid_lower) * 2 - 1,
                               dtype=torch.float32).flip(-1).reshape(1, 1, 1, -1, 3)
        delta = F.grid_sample(displacement[None], grid, mode='bilinear', padding_mode='zeros',
                              align_corners=True)[0, :, 0, 0].T.numpy()
        check_approximate = check_unscaled + delta
        error = np.linalg.norm(check_approximate - check_exact, axis=1)
        check = {'spacing_um': spacing, 'shape': shape, 'max_error_um': float(error.max()),
                 'p99_error_um': float(np.quantile(error, .99)), 'mean_error_um': float(error.mean()),
                 'passed': bool(error.max() <= 5.)}
        checks.append(check)
        if check['passed']:
            break
    assert check['passed']
    np.savez_compressed(directory / 'inverse_map.npz', displacement_um=baked,
        lower_unscaled_um=grid_lower, upper_unscaled_um=grid_upper,
        global_centre_um=map_centre, global_scale=map_scale,
        check_subject_um=check_subject, check_exact_ccf_um=check_exact,
        check_approximate_ccf_um=check_approximate)
    lineage.update({'synthetic_animal_id': plan['synthetic_animal_id'],
        'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
        'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
        'subject_plan_receipt_sha256': plan['receipt_sha256'], 'root_seed': PLAN_SEED,
        'ccf_context_sha256': CONTEXT_SHA})
    subject_record = {'lineage': lineage, 'local_field_sha256': local_sha,
        'plan_files': plan_files, 'plan_artifact_sha256': {name: bundle._file_sha256(directory / name)
            for name in (*plan_files.values(), 'subject_plan_receipt.json')},
        'inverse_map_sha256': bundle._file_sha256(directory / 'inverse_map.npz'),
        'mapping_checks': checks, 'forward_support_points_sha256': hashlib.sha256(forward_support.tobytes()).hexdigest(),
        'support_ccf_points_sha256': support_sha, 'mapped_atlas_centre_um': mapped_centre.tolist()}
    (directory / 'subject.json').write_text(json.dumps(subject_record, indent=2), encoding='utf8')
    subject_records.append({'file': str((directory / 'subject.json').relative_to(OUT)).replace('\\', '/'),
        'sha256': bundle._file_sha256(directory / 'subject.json'), 'plan_receipt_sha256': plan['receipt_sha256'],
        'local_field_sha256': local_sha, 'inverse_map_sha256': subject_record['inverse_map_sha256']})
    base = {'lineage': lineage, 'displacement': displacement,
        'lower': torch.as_tensor(grid_lower, dtype=torch.float32),
        'upper': torch.as_tensor(grid_upper, dtype=torch.float32),
        'map_centre': torch.as_tensor(map_centre, dtype=torch.float32),
        'map_scale': torch.as_tensor(map_scale, dtype=torch.float32),
        'bounds': subject_support_bounds_um(plan), 'anchor': mapped_centre,
        'support_points': forward_support}
    variants = []
    for variant in range(VARIANTS):
        virtual_rng = np.random.default_rng(np.random.SeedSequence([VIRTUAL_SEED, index, variant]))
        scale = np.exp(virtual_rng.uniform(-.1, .1, 3))
        variants.append({'virtual_index': index * VARIANTS + variant, 'base_index': 0, 'variant': variant,
            'virtual_subject_id': f"{lineage['subject_id']}-virtual-affine-v7-{variant:03d}",
            'base_lineage': lineage, 'anchor_um': mapped_centre.tolist(),
            'scale_ap_dv_ml': scale.tolist(),
            'bounds_um': (mapped_centre + scale * (base['bounds'] - mapped_centre)).tolist()})
    sampling_context = {'atlas': atlas, 'bases': [base], 'subjects': variants,
        'geometry': geometry, 'donor_rows': donor_rows, 'nuisance': nuisance}
    for section in range(PLANES):
        selection_rng = np.random.default_rng(np.random.SeedSequence([SEED, index, section, 4]))
        variant = int(selection_rng.integers(VARIANTS))
        draw_seed = SEED * 100000 + ordinal * 1000 + section
        sample = sample_one_shot_slide_artifacts_v7(sampling_context, [variant], draw_seed, side=SIDE)
        source = sample['provenance'][0]
        assert source['split'] == 'train' and source['base_lineage']['split'] == 'development'
        source['split'] = 'development'
        source['subject_plan_receipt_sha256'] = plan['receipt_sha256']
        source['inverse_map_sha256'] = subject_record['inverse_map_sha256']
        valid_pixels = int(sample['valid_mask'][0].sum())
        assert bool(sample['eligible'][0]) == (
            source['visible_support_mass'] >= SIDE * SIDE / 144 and valid_pixels >= SIDE * SIDE / 144)
        filename = f'subject_{index:08d}_section_{section:02d}.npz'
        np.savez_compressed(OUT / filename, inputs=sample['inputs'][0].numpy(),
            target_state=sample['state'][0].numpy(), target_centre_um=sample['centre'][0].numpy(),
            valid_mask=sample['valid_mask'][0].numpy(), reflection=sample['reflection'][0].numpy(),
            offsets_um=sample['offsets'][0].numpy(), weights=sample['weights'][0].numpy(),
            visible_support=sample['visible'][0].numpy(),
            source_visible_support_mass=np.float32(source['visible_support_mass']),
            eligible=sample['eligible'][0].numpy())
        record = {'file': filename, 'sha256': bundle._file_sha256(OUT / filename),
            'animal_id': lineage['animal_id'], 'subject_id': lineage['subject_id'],
            'specimen_id': lineage['specimen_id'], 'experiment_id': lineage['experiment_id'],
            'section_id': f"{lineage['subject_id']}-section-{section:08d}",
            'section_index': section, 'split': 'development',
            'synthetic_subject_plan_id': plan['subject_deformation_plan_id'],
            'synthetic_subject_realization_id': plan['subject_deformation_realization_id'],
            'plan_receipt_sha256': plan['receipt_sha256'], 'inverse_map_sha256': subject_record['inverse_map_sha256'],
            'variant': variant, 'draw_seed': draw_seed, 'eligible': bool(sample['eligible'][0]),
            'valid_pixels': valid_pixels, 'source_visible_support_mass': source['visible_support_mass'],
            'appearance_mode': source['mode'], 'provenance': source}
        records.append(record)
        print(json.dumps({'section': len(records), 'animal_index': index,
            'eligible': record['eligible'], 'valid_pixels': record['valid_pixels']}), flush=True)
    del plan, parameters, baked, displacement, base, sampling_context, sample, variants, forward_support

(OUT / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf8')
completed = {'physical_sections': len(records), 'synthetic_subjects': SUBJECTS,
    'eligible': sum(row['eligible'] for row in records),
    'ineligible': sum(not row['eligible'] for row in records),
    'modes': {mode: sum(row['appearance_mode'] == mode for row in records)
              for mode in ('raw', 'exact_black', 'imperfect_brush')},
    'subjects': subject_records, 'protocol_sha256': bundle._file_sha256(OUT / 'protocol.json'),
    'records_sha256': bundle._file_sha256(OUT / 'records.jsonl'),
    'scope': protocol['scope']}
assert len(records) == SUBJECTS * PLANES
assert len({row['provenance']['physical_section_id'] for row in records}) == len(records)
assert all(bundle._file_sha256(repository / 'training' / name) == digest
           for name, digest in protocol['source_sha256'].items())
assert bundle._file_sha256(PROTOCOL_DOC) == protocol['protocol_document_sha256']
assert bundle._file_sha256(train_maps_path) == protocol['train_map_completed_sha256']
assert bundle._file_sha256(geometry_path) == protocol['train_geometry_sha256']
(OUT / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf8')
print(json.dumps({'event': 'panel_frozen', 'subjects': SUBJECTS,
    'physical_sections': len(records), 'eligible': completed['eligible']}), flush=True)
