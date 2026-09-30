"""Extend the accepted TRAIN synthetic cohort to 64 distinct 3D deformations.

Run after GPU training releases the device. Only subject plans and inverse maps
are stored: fresh 192px arbitrary-plane sections remain generated on demand.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_acquisition_v2 as acquisition
from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6


def map_baked_inverse(points, displacement, lower, upper, centre, scale):
    unscaled = centre + (np.asarray(points) - centre) / scale
    grid = torch.as_tensor((unscaled - lower) / (upper - lower) * 2 - 1,
                           device=displacement.device, dtype=torch.float32)
    delta = F.grid_sample(displacement[None], grid.flip(-1).reshape(1, 1, 1, -1, 3),
                          mode='bilinear', padding_mode='zeros', align_corners=True)
    return unscaled + delta[0, :, 0, 0].T.reshape(*unscaled.shape).cpu().numpy()


OLD = ROOT / 'data/joint_v6_coherent_subject_plans_002'
OUTPUT = ROOT / 'data/joint_v7_independent_subject_maps_001'
OLD_SHA = '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
CONTEXT_SHA = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
SEED, INDEX_RANGE = 2026092911, range(8, 64)
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)

old_bytes = (OLD / 'completed.json').read_bytes()
assert hashlib.sha256(old_bytes).hexdigest() == OLD_SHA
old = json.loads(old_bytes)
assert old['context_sha256'] == CONTEXT_SHA
old_train = sorted((r for r in old['subjects'] if r['split'] == 'train'), key=lambda r: r['animal_index'])
old_dev = [r for r in old['subjects'] if r['split'] == 'development']
assert [r['animal_index'] for r in old_train] == list(range(8)) and len(old_dev) == 4
context_bytes = (OLD / 'context.json').read_bytes()
assert hashlib.sha256(context_bytes).hexdigest() == old['artifact_sha256']['context.json']
context = json.loads(context_bytes)
lower = np.asarray(context['full_ccf_lower_ap_dv_ml_um'], dtype=np.float64)
upper = np.asarray(context['full_ccf_upper_ap_dv_ml_um'], dtype=np.float64)
centre = (lower + upper) / 2

source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) | {
    Path(__file__).name, 'arbitrary_plane_subject_torch_v6.py',
    'subject_deformed_slab_multiresolution_bundle_v2.py',
})
source_paths = [repository / 'training' / name for name in source_names]
source_hashes = {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source_paths}
old_local_hashes = set()
for prior in old_train:
    directory = OLD / prior['directory']
    plan = bundle._read_raw_artifact(directory, prior['plan_files'])
    fields = (plan['state']['accepted_coarse_coefficients_um'], plan['state']['accepted_fine_coefficients_um'])
    old_local_hashes.add(hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest())
assert len(old_local_hashes) == 8

protocol = {
    'purpose': '56 further independent accepted local 3D TRAIN deformations; union with old 8 yields 64',
    'root_seed': SEED, 'split': 'train', 'new_animal_indices': list(INDEX_RANGE),
    'parent_plan_root': str(OLD), 'parent_completed_sha256': OLD_SHA,
    'context_sha256': CONTEXT_SHA, 'old_development_subjects_excluded': len(old_dev),
    'no_real_donor_or_benchmark_data': True, 'no_archived_sections': True,
    'sampler': 'unchanged accepted v2 deformation sampler, first passing candidate, distinct index seed',
    'map': 'exact accepted FP64 inverse RK4 baked at100um; use50um only if4096 fixed held queries exceed5um',
    'map_check': '2048 full CCF-box and2048 entire spline-support points; sampled, not supremum, max <=5um',
    'map_coordinates': 'unscaled subject AP/DV/ML displacement plus analytic inverse global scale; no clamping',
    'input_resolution_px': 192, 'future_rendering': 'on-demand through existing finite-thickness streaming sampler',
    'scope': 'synthetic geometric variants of one Allen atlas, not independent biological animals',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'source_sha256': source_hashes, 'python': sys.version, 'numpy': np.__version__,
    'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
}
OUTPUT.mkdir(parents=True, exist_ok=True)
if (OUTPUT / 'protocol.json').exists():
    frozen_protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    assert frozen_protocol['parent_completed_sha256'] == OLD_SHA
    assert frozen_protocol['source_sha256'] == source_hashes
else:
    (OUTPUT / 'source').mkdir()
    (OUTPUT / 'parent_plan_completed.json').write_bytes(old_bytes)
    (OUTPUT / 'parent_context.json').write_bytes(context_bytes)
    for path in source_paths:
        archived = OUTPUT / 'source' / path.relative_to(repository)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archived)
    (OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
protocol_sha = hashlib.sha256((OUTPUT / 'protocol.json').read_bytes()).hexdigest()
tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
               'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um',
               'global_scale', 'frozen_center_um')
seen_local_hashes = set(old_local_hashes)
records = []
started = time.perf_counter()
for index in INDEX_RANGE:
    tick = time.perf_counter()
    directory = OUTPUT / 'train' / f'subject_{index:08d}'
    if (directory / 'completed.json').exists():
        record = json.loads((directory / 'completed.json').read_text())
        assert record['subject_sha256'] == bundle._file_sha256(directory / 'subject.json')
        assert record['inverse_map_sha256'] == bundle._file_sha256(directory / 'inverse_map.npz')
        assert record['local_field_sha256'] not in seen_local_hashes
        seen_local_hashes.add(record['local_field_sha256'])
        records.append({**record, 'completed_sha256': bundle._file_sha256(directory / 'completed.json')})
        continue
    prefix = 'joint-v6-coherent-cohort-002-train'
    identity = {'split': 'train', 'animal_index': index,
                'subject_id': f'{prefix}-subject-{index:08d}',
                'animal_id': f'{prefix}-animal-{index:08d}',
                'specimen_id': f'{prefix}-specimen-{index:08d}',
                'experiment_id': f'{prefix}-experiment-{index:08d}'}
    print(json.dumps({'event': 'subject_started', 'index': index, 'planned_total': 64}), flush=True)
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=SEED, split='train', animal_index=index,
        animal_id=identity['animal_id'], ccf_context_sha256=CONTEXT_SHA)
    fields = (plan['state']['accepted_coarse_coefficients_um'], plan['state']['accepted_fine_coefficients_um'])
    local_hash = hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in fields)).hexdigest()
    assert local_hash not in seen_local_hashes and any(np.any(a != 0) for a in fields)
    seen_local_hashes.add(local_hash)
    directory.mkdir(parents=True, exist_ok=True)
    files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt['receipt_sha256'] == plan['receipt_sha256']
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    lineage = {**identity, 'synthetic_animal_id': plan['synthetic_animal_id'],
               'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
               'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
               'subject_plan_receipt_sha256': plan['receipt_sha256'], 'root_seed': SEED,
               'ccf_context_sha256': CONTEXT_SHA}
    (directory / 'lineage.json').write_text(json.dumps(lineage, indent=2), encoding='utf8')
    parameters = [torch.as_tensor(plan['state'][key], device='cuda', dtype=torch.float64) for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])
    exact = lambda points: map_accepted_subject_points_torch_v6(
        torch.as_tensor(points, device='cuda', dtype=torch.float64), *parameters,
        inverse=True, steps=steps, batch_size=8192).cpu().numpy()
    support_bounds = []
    for level in ('coarse', 'fine'):
        coefficients = plan['state'][f'accepted_{level}_coefficients_um']
        nonzero = np.array(np.nonzero(np.any(coefficients != 0, axis=-1)))
        origin, spacing = plan['state'][f'{level}_origin_um'], plan['state'][f'{level}_spacing_um']
        support_bounds.append(np.stack((origin + (nonzero.min(1) - 2) * spacing,
                                       origin + (nonzero.max(1) + 2) * spacing)))
    bounds = np.stack((np.min(np.array(support_bounds)[:, 0], axis=0) - 100,
                       np.max(np.array(support_bounds)[:, 1], axis=0) + 100))
    map_centre, map_scale = plan['state']['frozen_center_um'], plan['state']['global_scale']
    check_rng = np.random.default_rng(np.random.SeedSequence([SEED, index, 3]))
    check_unscaled = np.concatenate((check_rng.uniform(lower, upper, (2048, 3)),
                                    check_rng.uniform(bounds[0], bounds[1], (2048, 3))))
    check_subject = map_centre + map_scale * (check_unscaled - map_centre)
    check_exact = exact(check_subject)
    mapping_checks = []
    for grid_spacing in (100., 50.):
        bake_started = time.perf_counter()
        axes = [np.arange(bounds[0, k], bounds[1, k] + grid_spacing, grid_spacing) for k in range(3)]
        grid_lower, grid_upper = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
        grid_shape = tuple(len(a) for a in axes)
        baked = np.empty((*grid_shape, 3), dtype=np.float32)
        print(json.dumps({'event': 'baking_inverse_subject_map', 'index': index,
                          'spacing_um': grid_spacing, 'shape': grid_shape}), flush=True)
        for start in range(0, grid_shape[0], 8):
            points = np.stack(np.meshgrid(axes[0][start:start + 8], axes[1], axes[2], indexing='ij'), -1)
            baked[start:start + 8] = exact(map_centre + map_scale * (points - map_centre)) - points
        displacement = torch.from_numpy(np.moveaxis(baked, -1, 0).copy()).cuda()
        approximate = map_baked_inverse(check_subject, displacement, grid_lower, grid_upper, map_centre, map_scale)
        errors = np.linalg.norm(approximate - check_exact, axis=1)
        check = {'spacing_um': grid_spacing, 'shape': grid_shape, 'max_error_um': float(errors.max()),
                 'p99_error_um': float(np.quantile(errors, .99)), 'mean_error_um': float(errors.mean()),
                 'seconds': time.perf_counter() - bake_started, 'passed': bool(errors.max() <= 5.)}
        mapping_checks.append(check)
        print(json.dumps({'event': 'inverse_map_accuracy', 'index': index, **check}), flush=True)
        if check['passed']:
            break
        del baked, displacement
    assert check['passed'], '50um inverse-map interpolation exceeds the fixed5um sampled-error limit'
    np.savez_compressed(directory / 'inverse_map.npz', displacement_um=baked,
                        lower_unscaled_um=grid_lower, upper_unscaled_um=grid_upper,
                        global_centre_um=map_centre, global_scale=map_scale,
                        check_subject_um=check_subject, check_exact_ccf_um=check_exact,
                        check_approximate_ccf_um=approximate)
    plan_record = {**lineage, 'directory': directory.relative_to(OUTPUT).as_posix(),
                   'plan_files': files, 'accepted_amplitude_um': plan['realization']['accepted_amplitude_um'],
                   'accepted_candidate_index': plan['realization']['accepted_candidate_index'],
                   'global_scale_ap_dv_ml': map_scale.tolist(), 'local_field_sha256': local_hash,
                   'artifact_sha256': {name: bundle._file_sha256(directory / name)
                                       for name in (*files.values(), 'subject_plan_receipt.json', 'lineage.json')}}
    (directory / 'subject.json').write_text(json.dumps({'lineage': lineage, 'parent_plan': plan_record,
        'flow_steps': steps, 'mapped_atlas_centre_um': deformation._ccf_to_subject_points_from_verified_plan_v2(
            centre[None], plan)[0].tolist(), 'protocol_sha256': protocol_sha,
        'inverse_map_sha256': bundle._file_sha256(directory / 'inverse_map.npz'),
        'mapping_checks': mapping_checks}, indent=2), encoding='utf8')
    record = {**plan_record, 'subject_sha256': bundle._file_sha256(directory / 'subject.json'),
              'inverse_map_sha256': bundle._file_sha256(directory / 'inverse_map.npz'),
              'generation_seconds': time.perf_counter() - tick, 'section_count': 0}
    (directory / 'completed.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    records.append({**record, 'completed_sha256': bundle._file_sha256(directory / 'completed.json')})
    print(json.dumps({'event': 'subject_frozen', 'index': index,
                      'generation_seconds': record['generation_seconds'],
                      'max_mapping_error_um': check['max_error_um']}), flush=True)
    del plan, parameters, displacement, baked
for path in source_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hashes[path.relative_to(repository).as_posix()]
completed = {'protocol_sha256': protocol_sha, 'parent_plan_completed_sha256': OLD_SHA,
             'new_train_subjects': records, 'existing_train_subjects': 8,
             'total_independent_train_deformations': 64, 'held_out_synthetic_dev_subjects': len(old_dev),
             'new_sections': 0, 'seconds': time.perf_counter() - started,
             'synthetic_only': True, 'not_biological_animals': True}
(OUTPUT / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf8')
print(json.dumps({'event': 'independent_subject_maps_frozen',
                  'new_train_subjects': 56, 'total_independent_train_deformations': 64,
                  'seconds': completed['seconds']}), flush=True)
