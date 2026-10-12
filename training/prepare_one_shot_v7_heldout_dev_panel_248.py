"""Independent v7 synthetic DEV panel; run after the 247 pilot."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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
FINGERPRINTS = ROOT / 'data/one_shot_v7_prior_fingerprints_248_001'
CACHE = ROOT / 'data/allen_v6_float32_cache_248_001'
PLANS = ROOT / 'data/one_shot_v7_heldout_dev_plans_248_001'
PRIOR_DEV = tuple(ROOT / 'data' / name for name in (
    'one_shot_fresh_synthetic_dev_plans_001',
    'pose_feedback_037_fresh_synthetic_dev_plans_001',
    'pose_feedback_061_fresh_synthetic_dev_plans_001',
    'contextual_match_099_fresh_synthetic_dev_plans_001',
    'contextual_localization_102_fresh_dev_plans_001',
    'cross_candidate_match_103_fresh_dev_plans_001',
    'v3_pose_capture_confirmation_plans_001',
    'matcher_150_fresh_dev_plans'))
OUT = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
PROTOCOL_DOC = Path('docs/publication/ONE_SHOT_V7_HELDOUT_DEV_PANEL_248_PROTOCOL_20261012.md')
SEED, SIDE, SUBJECTS, PLANES = 2026101248, 192, 8, 32
PLAN_SEED, FIRST_INDEX = 2026101249, 2000
MAP_BATCH = 1024
CONTEXT_SHA = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
OLD_SHA = '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
assert torch.cuda.is_available()

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

fingerprint_done = json.loads((FINGERPRINTS / 'completed.json').read_text())
fingerprint_protocol = json.loads((FINGERPRINTS / 'protocol.json').read_text())
assert bundle._file_sha256(FINGERPRINTS / 'protocol.json') == fingerprint_done['protocol_sha256']
assert bundle._file_sha256(FINGERPRINTS / 'fingerprints.json') == fingerprint_done['fingerprints_sha256']
assert fingerprint_protocol['sources'] == fingerprint_done['sources']
assert bundle._file_sha256(repository / 'docs/publication/ONE_SHOT_V7_PRIOR_FINGERPRINTS_248_PROTOCOL_20261012.md') == fingerprint_protocol['protocol_document_sha256']
assert all(bundle._file_sha256(repository / 'training' / name) == digest
           for name, digest in fingerprint_protocol['source_sha256'].items())
fingerprints = json.loads((FINGERPRINTS / 'fingerprints.json').read_text())
assert len(fingerprints) == fingerprint_done['subjects'] == 132
historical = [(OLD, old['subjects'])] + [(root, index['subjects']) for root, index in prior]
historical.append((TRAIN_MAPS, train_maps['new_train_subjects']))
assert fingerprint_done['sources'] == {root.name: bundle._file_sha256(root / 'completed.json')
                                       for root, _ in historical}
fingerprint_by_key = {(row['source'], row['source_ordinal']): row for row in fingerprints}
assert len(fingerprint_by_key) == len(fingerprints)
for root, rows in historical:
    for ordinal, row in enumerate(rows):
        prior_fingerprint = fingerprint_by_key[(root.name, ordinal)]
        assert prior_fingerprint['directory'] == row['directory']
        for key in ('subject_deformation_plan_id', 'subject_deformation_realization_id',
                    'subject_plan_receipt_sha256'):
            assert prior_fingerprint[key] == row[key]
        assert all(bundle._file_sha256(root / row['directory'] / name) == digest
                   for name, digest in row['artifact_sha256'].items())
for ordinal, row in enumerate(train_maps['new_train_subjects']):
    assert row['split'] == 'train'
    directory = TRAIN_MAPS / row['directory']
    assert bundle._file_sha256(directory / 'completed.json') == row['completed_sha256']
    assert fingerprint_by_key[(TRAIN_MAPS.name, ordinal)]['local_field_sha256'] == row['local_field_sha256']
seen_ids = {row['subject_deformation_plan_id'] for row in fingerprints}
seen_realizations = {row['subject_deformation_realization_id'] for row in fingerprints}
seen_receipts = {row['subject_plan_receipt_sha256'] for row in fingerprints}
seen_local = {row['local_field_sha256'] for row in fingerprints}
assert len(seen_ids) == 12 + 8 * len(PRIOR_DEV) + 56
assert len(seen_realizations) == len(seen_receipts) == len(seen_local) == len(seen_ids)

cache_done = json.loads((CACHE / 'completed.json').read_text())
cache_protocol = json.loads((CACHE / 'protocol.json').read_text())
assert bundle._file_sha256(CACHE / 'protocol.json') == cache_done['protocol_sha256']
assert bundle._file_sha256(CACHE / 'atlas_float32.npy') == cache_done['atlas_file_sha256']
assert cache_done['atlas_array_receipt'] == cache_protocol['atlas_array_receipt'] == allen.ATLAS_FLOAT32_RECEIPT_V6
assert cache_protocol['template_raw_sha256'] == allen.TEMPLATE_RAW_SHA256_V6
assert cache_protocol['annotation_raw_sha256'] == allen.ANNOTATION_RAW_SHA256_V6
assert bundle._file_sha256(repository / 'docs/publication/ALLEN_V6_FLOAT32_CACHE_248_PROTOCOL_20261012.md') == cache_protocol['protocol_document_sha256']
assert all(bundle._file_sha256(repository / 'training' / name) == digest
           for name, digest in cache_protocol['source_sha256'].items())

plans_completed_path = PLANS / 'completed.json'
assert plans_completed_path.is_file(), 'independent 248 plans must be frozen first'
plans_completed = json.loads(plans_completed_path.read_text(encoding='utf8'))
assert bundle._file_sha256(PLANS / 'protocol.json') == plans_completed['protocol_sha256']
plans_protocol = json.loads((PLANS / 'protocol.json').read_text(encoding='utf8'))
assert plans_protocol['root_seed'] == PLAN_SEED
assert plans_protocol['animal_indices'] == list(range(FIRST_INDEX, FIRST_INDEX + SUBJECTS))
assert all(bundle._file_sha256(repository / 'training' / name) == digest
           for name, digest in plans_protocol['source_sha256'].items())
plan_rows = {row['animal_index']: row for row in plans_completed['subjects']}
assert len(plan_rows) == SUBJECTS
assert all(bundle._file_sha256(PLANS / f'subject_{index:08d}' / 'completed.json') == digest
           for index, digest in ((int(i), h) for i, h in plans_completed['subject_completed_sha256'].items()))

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
    'map': 'accepted FP64 inverse RK4 baked at 100um on CUDA; 50um if fixed 4096-query sampled max exceeds 5um; same GPU FP32 interpolation as TRAIN',
    'map_point_batch_size': MAP_BATCH,
    'eligibility': 'source visible support and observed valid pixels each >= side^2/144; ineligible rows retained',
    'old_plan_completed_sha256': OLD_SHA,
    'train_map_completed_sha256': bundle._file_sha256(train_maps_path),
    'prior_fingerprints_completed_sha256': bundle._file_sha256(FINGERPRINTS / 'completed.json'),
    'allen_float32_cache_completed_sha256': bundle._file_sha256(CACHE / 'completed.json'),
    'prior_dev_plan_completed_sha256': {root.name: bundle._file_sha256(root / 'completed.json')
                                        for root in PRIOR_DEV},
    'parent_context_sha256': hashlib.sha256(context_bytes).hexdigest(),
    'train_geometry_sha256': hashlib.sha256(geometry_bytes).hexdigest(),
    'train_map_protocol_sha256': bundle._file_sha256(TRAIN_GEOMETRY / 'protocol.json'),
    'independent_dev_plans_completed_sha256': bundle._file_sha256(plans_completed_path),
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'atlas_annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'atlas_normalized_receipt': allen.ATLAS_FLOAT32_RECEIPT_V6,
    'protocol_document_sha256': bundle._file_sha256(PROTOCOL_DOC),
    'source_sha256': {name: bundle._file_sha256(repository / 'training' / name) for name in source_names},
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
    'scope': 'synthetic DEV from one Allen atlas and TRAIN-only acquisition framing; not biological validation, CAL, FINAL, or public benchmark',
}
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / 'completed.json').exists(), 'panel is already frozen'
protocol_path = OUT / 'protocol.json'
protocol_text = json.dumps(protocol, indent=2)
if protocol_path.exists():
    assert protocol_path.read_text(encoding='utf8') == protocol_text
else:
    assert not any(OUT.iterdir())
    protocol_path.write_text(protocol_text, encoding='utf8')
atlas_array = np.load(CACHE / 'atlas_float32.npy', allow_pickle=False)
assert atlas_array.shape == tuple(allen.ATLAS_FLOAT32_RECEIPT_V6['shape']) and atlas_array.dtype == np.float32
support = np.flatnonzero(atlas_array[1] > 0)
selected = np.random.default_rng(SUPPORT_SEED).choice(support, 4096, replace=False)
support_ccf = (np.array(np.unravel_index(selected, atlas_array.shape[1:])).T.astype(np.float64) + .5) * 25.
support_sha = hashlib.sha256(support_ccf.tobytes()).hexdigest()
atlas = torch.from_numpy(atlas_array).cuda()
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
    stored = plan_rows[index]
    plan_directory = PLANS / f'subject_{index:08d}'
    assert stored['animal_id'] == lineage['animal_id']
    assert all(bundle._file_sha256(plan_directory / name) == digest
               for name, digest in stored['artifact_sha256'].items())
    plan = bundle._read_raw_artifact(plan_directory, stored['plan_files'])
    assert plan['receipt_sha256'] == stored['subject_plan_receipt_sha256']
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
    done = directory / 'completed.json'
    if done.exists():
        saved = json.loads(done.read_text(encoding='utf8'))
        assert saved['animal_index'] == index
        assert saved['plan_receipt_sha256'] == plan['receipt_sha256']
        assert saved['local_field_sha256'] == local_sha
        assert len(saved['records']) == PLANES
        assert saved['subject_index_record']['sha256'] == saved['subject_json_sha256']
        assert all(bundle._file_sha256(OUT / row['file']) == row['sha256']
                   for row in saved['records'])
        assert bundle._file_sha256(directory / 'subject.json') == saved['subject_json_sha256']
        assert bundle._file_sha256(directory / 'inverse_map.npz') == saved['inverse_map_sha256']
        records.extend(saved['records'])
        subject_records.append(saved['subject_index_record'])
        print(json.dumps({'event': 'subject_reused', 'animal_index': index,
                          'sections': len(saved['records'])}), flush=True)
        del plan
        continue
    assert not directory.exists()
    directory.mkdir()
    plan_files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    parameters = [torch.as_tensor(plan['state'][key], device='cuda', dtype=torch.float64) for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])
    map_centre, map_scale = plan['state']['frozen_center_um'], plan['state']['global_scale']
    forward_support = map_accepted_subject_points_torch_v6(torch.as_tensor(support_ccf, device='cuda'), *parameters,
        inverse=False, steps=steps, batch_size=MAP_BATCH).cpu().numpy()
    mapped_centre = map_accepted_subject_points_torch_v6(torch.as_tensor(atlas_centre[None], device='cuda', dtype=torch.float64),
        *parameters, inverse=False, steps=steps, batch_size=MAP_BATCH).cpu().numpy()[0]
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
    check_exact = map_accepted_subject_points_torch_v6(torch.as_tensor(check_subject, device='cuda'), *parameters,
        inverse=True, steps=steps, batch_size=MAP_BATCH).cpu().numpy()
    checks = []
    for spacing in (100., 50.):
        axes = [np.arange(bounds[0, k], bounds[1, k] + spacing, spacing) for k in range(3)]
        grid_lower, grid_upper = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
        shape = tuple(len(a) for a in axes)
        baked = np.empty((*shape, 3), dtype=np.float32)
        for start in range(0, shape[0], 4):
            points = np.stack(np.meshgrid(axes[0][start:start + 4], axes[1], axes[2], indexing='ij'), -1)
            subject_points = map_centre + map_scale * (points - map_centre)
            exact = map_accepted_subject_points_torch_v6(torch.as_tensor(subject_points, device='cuda'), *parameters,
                inverse=True, steps=steps, batch_size=MAP_BATCH).cpu().numpy()
            baked[start:start + 4] = exact - points
        displacement = torch.from_numpy(np.moveaxis(baked, -1, 0).copy()).cuda()
        grid = torch.as_tensor((check_unscaled - grid_lower) / (grid_upper - grid_lower) * 2 - 1,
                               device='cuda', dtype=torch.float32).flip(-1).reshape(1, 1, 1, -1, 3)
        delta = F.grid_sample(displacement[None], grid, mode='bilinear', padding_mode='zeros',
                              align_corners=True)[0, :, 0, 0].T.cpu().numpy()
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
        'lower': torch.as_tensor(grid_lower, device='cuda', dtype=torch.float32),
        'upper': torch.as_tensor(grid_upper, device='cuda', dtype=torch.float32),
        'map_centre': torch.as_tensor(map_centre, device='cuda', dtype=torch.float32),
        'map_scale': torch.as_tensor(map_scale, device='cuda', dtype=torch.float32),
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
    first_record = len(records)
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
        np.savez_compressed(OUT / filename, inputs=sample['inputs'][0].cpu().numpy(),
            target_state=sample['state'][0].cpu().numpy(), target_centre_um=sample['centre'][0].cpu().numpy(),
            valid_mask=sample['valid_mask'][0].cpu().numpy(), reflection=sample['reflection'][0].cpu().numpy(),
            offsets_um=sample['offsets'][0].cpu().numpy(), weights=sample['weights'][0].cpu().numpy(),
            visible_support=sample['visible'][0].cpu().numpy(),
            source_visible_support_mass=np.float32(source['visible_support_mass']),
            eligible=sample['eligible'][0].cpu().numpy())
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
    (directory / 'completed.json').write_text(json.dumps({
        'animal_index': index, 'plan_receipt_sha256': plan['receipt_sha256'],
        'local_field_sha256': local_sha, 'subject_index_record': subject_records[-1],
        'subject_json_sha256': subject_records[-1]['sha256'],
        'inverse_map_sha256': subject_record['inverse_map_sha256'],
        'records': records[first_record:],
    }, indent=2), encoding='utf8')
    print(json.dumps({'event': 'subject_frozen', 'animal_index': index,
                      'sections': len(records) - first_record}), flush=True)
    del plan, parameters, baked, displacement, base, sampling_context, sample, variants, forward_support

(OUT / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf8')
completed = {'physical_sections': len(records), 'synthetic_subjects': SUBJECTS,
    'eligible': sum(row['eligible'] for row in records),
    'ineligible': sum(not row['eligible'] for row in records),
    'modes': {mode: sum(row['appearance_mode'] == mode for row in records)
              for mode in ('raw', 'exact_black', 'imperfect_brush')},
    'subjects': subject_records,
    'subject_completed_sha256': {str(index): bundle._file_sha256(
        OUT / f'subject_{index:08d}' / 'completed.json') for index in protocol['subject_indices']},
    'protocol_sha256': bundle._file_sha256(OUT / 'protocol.json'),
    'records_sha256': bundle._file_sha256(OUT / 'records.jsonl'),
    'scope': protocol['scope']}
assert len(records) == SUBJECTS * PLANES
assert len({row['provenance']['physical_section_id'] for row in records}) == len(records)
assert all(bundle._file_sha256(repository / 'training' / name) == digest
           for name, digest in protocol['source_sha256'].items())
assert bundle._file_sha256(PROTOCOL_DOC) == protocol['protocol_document_sha256']
assert bundle._file_sha256(train_maps_path) == protocol['train_map_completed_sha256']
assert bundle._file_sha256(FINGERPRINTS / 'completed.json') == protocol['prior_fingerprints_completed_sha256']
assert bundle._file_sha256(CACHE / 'completed.json') == protocol['allen_float32_cache_completed_sha256']
assert bundle._file_sha256(geometry_path) == protocol['train_geometry_sha256']
(OUT / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf8')
print(json.dumps({'event': 'panel_frozen', 'subjects': SUBJECTS,
    'physical_sections': len(records), 'eligible': completed['eligible']}), flush=True)
