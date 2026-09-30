"""4,096 new 192px TRAIN planes; bounded map approximation, exact sparse targets.

Run only after the current training/evaluation releases the GPU. Each completed
512-section subject shard is immutable and can be consumed independently.
No learned models, catalogue, development data or automatic segmentation.
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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_coherent_subject_v6 import make_coherent_subject_section_v6
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_subject_deformation_v2 import (
    subject_deformation_plan_receipt_v2, _ccf_to_subject_points_from_verified_plan_v2,
)
from training.arbitrary_plane_subject_sampling_v6 import sample_subject_planes
from training.arbitrary_plane_subject_torch_v6 import map_accepted_subject_points_torch_v6
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact


def map_baked_inverse(points, displacement, lower, upper, centre, scale):
    """Interpolate inverse-flow displacement; preserve analytic global scaling.

    The caller's grid encloses the complete support of both spline velocities.
    Outside it the velocity and inverse displacement are identically zero, so
    zero padding is valid; neither absolute coordinates nor queries are clamped.
    """
    unscaled = centre + (np.asarray(points) - centre) / scale
    grid = torch.as_tensor((unscaled - lower) / (upper - lower) * 2 - 1, device=displacement.device, dtype=torch.float32)
    delta = F.grid_sample(displacement[None], grid.flip(-1).reshape(1, 1, 1, -1, 3),
                          mode='bilinear', padding_mode='zeros', align_corners=True)
    return unscaled + delta[0, :, 0, 0].T.reshape(*unscaled.shape).cpu().numpy()


PLAN_ROOT = ROOT / 'data/joint_v6_coherent_subject_plans_002'
OUTPUT = ROOT / 'data/joint_v7_training_data_001'
REAL_GEOMETRY = ROOT / 'data/allen_real_training_inputs_20260929/image_geometry.jsonl'
PLAN_SHA = '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
GEOMETRY_SHA = '9a718b2fb2791584e1d225123facbe1c02c96cfa701a9e6f3fc68f17a565d218'
GPU_RECEIPT = ROOT / 'runs/joint_v6_subject_torch_gpu_check_001/completed.json'
GPU_RECEIPT_SHA = 'a5b71a0483c027499059160399ab742e914b3c9163145d0b4a895a3dc9646546'
SEED, SHAPE, PER_SUBJECT = 2026093007, (192, 192), 512
MODES = ('raw', 'exact_black', 'imperfect_brush')
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)

plan_bytes = (PLAN_ROOT / 'completed.json').read_bytes()
geometry_bytes = REAL_GEOMETRY.read_bytes()
assert hashlib.sha256(plan_bytes).hexdigest() == PLAN_SHA
assert hashlib.sha256(geometry_bytes).hexdigest() == GEOMETRY_SHA
assert hashlib.sha256(GPU_RECEIPT.read_bytes()).hexdigest() == GPU_RECEIPT_SHA
cohort = json.loads(plan_bytes)
subjects = sorted((r for r in cohort['subjects'] if r['split'] == 'train'), key=lambda r: r['animal_index'])
assert len(subjects) == 8 and {r['animal_index'] for r in subjects} == set(range(8))
geometry = [json.loads(line) for line in geometry_bytes.decode('utf8').splitlines()]
assert len(geometry) == 256 and all(r['split'] == 'development_train' for r in geometry)
donors = sorted({r['animal_id'] for r in geometry})
donor_rows = [np.array([i for i, r in enumerate(geometry) if r['animal_id'] == donor]) for donor in donors]
assert len(donors) == 58
atlas_centre = np.asarray(allen.ATLAS_SHAPE_AP_DV_ML_V6) * 25. / 2.
affine = np.asarray([r['model_pixel_to_ap_dv_ml_um'] for r in geometry])
u, v = affine[:, :, 0] * 96, affine[:, :, 1] * 96
span_u = np.linalg.norm(u, axis=1)
u = u / span_u[:, None]
along = (v * u).sum(1)
v = v - along[:, None] * u
span_v = np.linalg.norm(v, axis=1)
v = v / span_v[:, None]
source_centres = affine[:, :, 2] + 48 * (affine[:, :, 0] + affine[:, :, 1])
nuisance = np.column_stack(((u * (source_centres - atlas_centre)).sum(1),
                           (v * (source_centres - atlas_centre)).sum(1), span_u, span_v, along / span_v))

OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'source').mkdir()
(OUTPUT / 'parent_plan_completed.json').write_bytes(plan_bytes)
(OUTPUT / 'train_acquisition_geometry.jsonl').write_bytes(geometry_bytes)
shutil.copyfile(GPU_RECEIPT, OUTPUT / 'gpu_coordinate_equivalence_receipt.json')
source_names = sorted(set(allen.DETERMINISTIC_SOURCE_FILES_V6) | {
    f'training/{name}.py' for name in (
        Path(__file__).stem, 'arbitrary_plane_coherent_subject_v6', 'arbitrary_plane_subject_torch_v6',
        'arbitrary_plane_subject_sampling_v6', 'arbitrary_plane_subject_deformation_v2',
        'arbitrary_plane_subject_section_v2', 'arbitrary_plane_full_frame_primitives',
        'subject_deformed_slab_multiresolution_bundle_v2',
    )})
source_hashes = {}
for name in source_names:
    target = OUTPUT / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, target)
    source_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
assert source_hashes['training/arbitrary_plane_subject_torch_v6.py'] == json.loads(GPU_RECEIPT.read_text())['mapper_source_sha256']
pixel_y = np.unique(np.r_[np.arange(0, SHAPE[0], 4), SHAPE[0] - 1]).astype(np.int64)
pixel_x = np.unique(np.r_[np.arange(0, SHAPE[1], 4), SHAPE[1] - 1]).astype(np.int64)
weights = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float64) / 16.
protocol = {
    'seed': SEED, 'split': 'train', 'subjects': 8, 'sections_per_subject': PER_SUBJECT,
    'shape_h_w': SHAPE, 'modes': MODES, 'plan_root': str(PLAN_ROOT), 'plan_completion_sha256': PLAN_SHA,
    'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()},
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'source_sha256': source_hashes, 'context_sha256': cohort['context_sha256'],
    'atlas': {'template_sha256': allen.TEMPLATE_RAW_SHA256_V6, 'annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
              'normalization': allen.ATLAS_FLOAT32_RECEIPT_V6, 'axes': ['AP', 'DV', 'ML'], 'units': 'um'},
    'geometry_source': {'sha256': GEOMETRY_SHA, 'donors': len(donors), 'role': 'TRAIN acquisition framing only, not synthetic animal ancestry'},
    'sampling': 'uniform RP2 normal and roll, conservative subject-box slab offsets; all draws kept, including empty/marginal views',
    'framing': 'half conservative full-box framing; half donor-uniform TRAIN tangent-shift/span/shear tuples, with log-scale SD .12, tangent-shift SD 300um and shear SD .02; physical rerender, not image crop',
    'psf': '9 explicit samples, 25-100um thickness, globally normalized trapezoid weights; bounded interpolated inverse map for full render, exact inverse map for every saved sparse slab target',
    'mapping': 'authenticated accepted coefficients, exact plan RK4 steps, CUDA FP64 inverse at baked grid and sparse supervision;100um lattice first,50um only if4096 fixed held queries exceed5um norm error; fail if50um also exceeds5um',
    'baked_domain': 'complete nonzero coarse/fine spline coefficient support plus100um, in inverse-global-scaled coordinates; inverse displacement is exactly zero outside this domain, analytic inverse global scale is never clamped',
    'approximation_measurement': 'same4096 held points per subject:2048 inside fullCCF box plus2048 whole velocity support; observed error is a numerical sample check, not a formal supremum bound; exact sparse errors on every rendered section also recorded',
    'coordinates': 'O + x/W U + y/H V, finite horizontal reflection once, physical AP/DV/ML um; no target coordinate clamping',
    'target_precision': 'full center and canonicalOUV/field8 use the interpolated inverse map; sparsePSF including center slab uses exact accepted FP64 inverse; both coordinate arrays storedFP32, OUV remainsFP64; interpolation and cast errors explicitly distinct',
    'sparse_psf': 'target_psf_um[:,pixel_y[:,None],pixel_x[None,:],:] extracted at exact observed-image pixels; last interval is three pixels, never treat this as a uniformly spaced rescaled image',
    'canonical_field_8': 'first3 local residual/200um, last3 local director residual/.1; canonical (unreflected) full-resolution field bilinearly reduced to8x8, not a replacement for exact coordinate truth',
    'input_recipe': 'stack(shared_pre_brush_image * brush_keep_mask[mode], brush_outline[mode], constant outline_available[mode], zeros, zeros); marks/constraints absent',
    'visible_weight_recipe': 'finite_support * tissue_retained * brush_keep_mask[mode]; do not infer support from image brightness',
    'appearance': 'synthetic conditional-tissue gamma/inversion/gain, smooth illumination, background gradient/texture, pixel noise; optional known missing-tissue ellipse; this is not real histology or a validated stain simulator',
    'brush': 'paired raw/allkeep, exact remaining synthetic tissue on black, imperfect dilated/eroded brush; one-pixel four-neighbor inner outline; no automatic segmentation prerequisite',
    'clean_target': 'clean_atlas_intensity is the unaugmented finite-PSF atlas render, including finite support; missing-tissue/brush pixels excluded using visible weights',
    'information_eligibility': 'visible finite support >= H*W/144; metadata only, no geometry rejection/resampling',
    'shards': 'train/subject_N/completed.json is written only after512 immutable per-section NPZs and sections.jsonl finish; consumers must pin that completion and file hashes',
    'limitations': 'eight synthetic maps of one atlas, not independent biological animals; no DEV additions, real pixels, learned targets, checkpoints, calibration or benchmark claims',
}
(OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
protocol_sha = hashlib.sha256((OUTPUT / 'protocol.json').read_bytes()).hexdigest()
print('Loading pinned atlas; generating only new TRAIN planes on accepted subject maps', flush=True)
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array)
del annotation
identity_yx = np.stack(np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing='ij')).astype(np.float64)
yy, xx = torch.meshgrid(torch.linspace(-1, 1, SHAPE[0]), torch.linspace(-1, 1, SHAPE[1]), indexing='ij')
tensor_keys = ('accepted_coarse_coefficients_um', 'coarse_origin_um', 'coarse_spacing_um',
               'accepted_fine_coefficients_um', 'fine_origin_um', 'fine_spacing_um', 'global_scale', 'frozen_center_um')
started = time.perf_counter()
shards = []
for subject in subjects:
    directory = PLAN_ROOT / subject['directory']
    for name, expected in subject['artifact_sha256'].items():
        with (directory / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    plan = _read_raw_artifact(directory, subject['plan_files'])
    assert subject_deformation_plan_receipt_v2(plan)['receipt_sha256'] == subject['subject_plan_receipt_sha256']
    assert plan['provenance']['split'] == 'train' and plan['provenance']['animal_id'] == subject['animal_id']
    for name, expected in plan['source_sha256'].items():
        content = (repository / 'training' / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        assert hashlib.sha256(content).hexdigest() == expected
    parameters_gpu = [torch.tensor(plan['state'][key], dtype=torch.float64, device='cuda') for key in tensor_keys]
    steps = int(plan['resolved_config']['flow']['steps'])
    exact_mapper = lambda points: map_accepted_subject_points_torch_v6(torch.from_numpy(points).to(device='cuda', dtype=torch.float64), *parameters_gpu, inverse=True, steps=steps, batch_size=8192).cpu().numpy()
    subject_centre = _ccf_to_subject_points_from_verified_plan_v2(atlas_centre[None], plan)[0]
    output = OUTPUT / 'train' / f"subject_{subject['animal_index']:08d}"
    output.mkdir(parents=True)
    lineage = {key: subject[key] for key in ('subject_id', 'animal_id', 'specimen_id', 'experiment_id', 'split', 'synthetic_animal_id')}
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
    check_rng = np.random.default_rng(np.random.SeedSequence([SEED, subject['animal_index'], 3]))
    check_unscaled = np.concatenate((check_rng.uniform(plan['state']['full_ccf_lower_um'], plan['state']['full_ccf_upper_um'], (2048, 3)),
                                    check_rng.uniform(bounds[0], bounds[1], (2048, 3))))
    check_subject = map_centre + map_scale * (check_unscaled - map_centre)
    check_exact = exact_mapper(check_subject)
    mapping_checks = []
    for grid_spacing in (100., 50.):
        bake_started = time.perf_counter()
        axes = [np.arange(bounds[0, k], bounds[1, k] + grid_spacing, grid_spacing) for k in range(3)]
        lower, upper = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
        grid_shape = tuple(len(a) for a in axes)
        baked = np.empty((*grid_shape, 3), dtype=np.float32)
        print(json.dumps({'event': 'baking_inverse_subject_map', 'subject': subject['animal_index'],
            'spacing_um': grid_spacing, 'shape': grid_shape, 'points': int(np.prod(grid_shape))}), flush=True)
        for start in range(0, grid_shape[0], 8):
            points = np.stack(np.meshgrid(axes[0][start:start + 8], axes[1], axes[2], indexing='ij'), -1)
            baked[start:start + 8] = exact_mapper(map_centre + map_scale * (points - map_centre)) - points
        displacement = torch.from_numpy(np.moveaxis(baked, -1, 0).copy()).cuda()
        mapper = lambda points: map_baked_inverse(points, displacement, lower, upper, map_centre, map_scale)
        check_approximate = mapper(check_subject)
        errors = np.linalg.norm(check_approximate - check_exact, axis=1)
        check = {'spacing_um': grid_spacing, 'shape': grid_shape, 'max_error_um': float(errors.max()),
            'p99_error_um': float(np.quantile(errors, .99)), 'mean_error_um': float(errors.mean()),
            'seconds': time.perf_counter() - bake_started, 'passed': bool(errors.max() <= 5.)}
        mapping_checks.append(check)
        print(json.dumps({'event': 'inverse_map_accuracy', 'subject': subject['animal_index'], **check}), flush=True)
        if check['passed']:
            break
        del displacement, baked
    assert check['passed'], '50um inverse-map interpolation exceeds the fixed5um sampled-error limit'
    np.savez_compressed(output / 'inverse_map.npz', displacement_um=baked, lower_unscaled_um=lower,
        upper_unscaled_um=upper, global_centre_um=map_centre, global_scale=map_scale,
        check_subject_um=check_subject, check_exact_ccf_um=check_exact, check_approximate_ccf_um=check_approximate)
    with (output / 'inverse_map.npz').open('rb') as stream:
        inverse_map_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    (output / 'subject.json').write_text(json.dumps({'lineage': lineage, 'parent_plan': subject,
        'flow_steps': steps, 'mapped_atlas_centre_um': subject_centre.tolist(), 'protocol_sha256': protocol_sha,
        'inverse_map_sha256': inverse_map_sha, 'mapping_checks': mapping_checks}, indent=2), encoding='utf8')
    section_bytes, eligible_counts, shard_started = 0, np.zeros(3, dtype=np.int64), time.perf_counter()
    with (output / 'sections.jsonl').open('w', encoding='utf8') as index_file:
        for index in range(PER_SUBJECT):
            tick = time.perf_counter()
            prefix = [SEED, subject['animal_index'], index]
            rng = np.random.default_rng(np.random.SeedSequence([*prefix, 0]))
            appearance_rng = np.random.default_rng(np.random.SeedSequence([*prefix, 1]))
            noise_seed = int(np.random.SeedSequence([*prefix, 2]).generate_state(1, dtype=np.uint64)[0])
            noise_rng = torch.Generator().manual_seed(noise_seed)
            offsets = np.linspace(-.5, .5, 9) * rng.uniform(25, 100)
            reflection = (bool(rng.integers(2)), False)
            draw = sample_subject_planes(plan, rng, 1, SHAPE, offsets)
            ouv = draw['physical_ouv_ap_dv_ml_um'][0].copy()
            framing = {'kind': 'conservative_subject_box'}
            if rng.random() < .5:
                source_row = int(rng.choice(donor_rows[int(rng.integers(len(donors)))]))
                shift_u, shift_v, length_u, length_v, shear = nuisance[source_row]
                shift_u, shift_v = np.array([shift_u, shift_v]) + rng.normal(0, 300, 2)
                length_u, length_v = np.array([length_u, length_v]) * np.exp(rng.normal(0, .12, 2))
                shear += rng.normal(0, .02)
                normal = draw['normal_ap_dv_ml'][0]
                u_axis, v_axis = ouv[1] / np.linalg.norm(ouv[1]), ouv[2] / np.linalg.norm(ouv[2])
                centre = subject_centre + normal * np.dot(normal, ouv[0] - subject_centre) + shift_u * u_axis + shift_v * v_axis
                edge_u, edge_v = length_u * u_axis, length_v * (v_axis + shear * u_axis)
                ouv = np.stack((centre - .5 * (edge_u + edge_v), edge_u, edge_v))
                framing = {'kind': 'TRAIN_acquisition_tuple', 'source_geometry_row': source_row,
                    'source_identity': {key: geometry[source_row][key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id', 'split')},
                    'realized_shift_u_v_length_u_v_shear': [shift_u, shift_v, length_u, length_v, shear]}
            section_id = f"joint-v7-training-001-subject-{subject['animal_index']:08d}-section-{index:08d}"
            section = make_coherent_subject_section_v6(plan, ouv, identity_yx, reflection, offsets, weights,
                {**lineage, 'section_id': section_id}, {'protocol_sha256': protocol_sha}, atlas,
                allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
                subject_to_ccf_mapper=mapper)
            rendered = section['raw_rendered_channels']
            support = rendered[1].clamp(0, 1)
            parameters = {'gain': float(appearance_rng.uniform(.6, 1.4)),
                'gamma': float(np.exp(appearance_rng.uniform(np.log(.6), np.log(1.6)))),
                'invert_tissue': bool(appearance_rng.integers(2)), 'noise_std': float(appearance_rng.uniform(.005, .05)),
                'background_mean': float(appearance_rng.uniform(0, .8)),
                'background_slope_yx': appearance_rng.uniform(-.15, .15, 2).tolist(),
                'illumination_amplitude': float(appearance_rng.uniform(0, .15)),
                'background_texture_amplitude': float(appearance_rng.uniform(0, .06)),
                'mask_dilate': bool(appearance_rng.integers(2)), 'mask_radius_px': int(appearance_rng.integers(1, 7)),
                'damage': bool(appearance_rng.random() < .2), 'noise_seed_uint64': noise_seed}
            illumination = F.interpolate(torch.randn(1, 1, 5, 5, generator=noise_rng), SHAPE, mode='bilinear', align_corners=False)[0, 0]
            background_texture = F.interpolate(torch.randn(1, 1, 16, 16, generator=noise_rng), SHAPE, mode='bilinear', align_corners=False)[0, 0]
            tissue_intensity = (rendered[0] / support.clamp_min(1e-6)).clamp(0, 1).pow(parameters['gamma'])
            if parameters['invert_tissue']:
                tissue_intensity = 1 - tissue_intensity
            tissue_intensity = (tissue_intensity * parameters['gain'] * (1 + parameters['illumination_amplitude'] * illumination)).clamp(0, 1)
            retained = torch.ones(SHAPE, dtype=torch.bool)
            if parameters['damage']:
                cy, cx = appearance_rng.uniform(-.7, .7, 2)
                ry, rx = appearance_rng.uniform(.05, .25, 2)
                retained = ((yy - cy) / ry).square() + ((xx - cx) / rx).square() >= 1
                parameters['missing_tissue_ellipse_cy_cx_ry_rx'] = [cy, cx, ry, rx]
            visible_support = support * retained
            slope = parameters['background_slope_yx']
            background = parameters['background_mean'] + slope[0] * yy + slope[1] * xx + parameters['background_texture_amplitude'] * background_texture
            before_brush = (tissue_intensity * visible_support + background * (1 - visible_support)
                + parameters['noise_std'] * torch.randn(SHAPE, generator=noise_rng)).clamp(0, 1)
            exact = (support > 0) & retained
            radius = parameters['mask_radius_px']
            if parameters['mask_dilate']:
                imperfect = F.max_pool2d(exact[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0
            else:
                imperfect = -F.max_pool2d(-F.pad(exact[None, None].float(), (radius,) * 4), 2 * radius + 1, 1)[0, 0] > 0
            masks = torch.stack((torch.ones_like(exact), exact, imperfect))
            eroded = masks.clone()
            eroded[:, 1:] &= masks[:, :-1]
            eroded[:, :-1] &= masks[:, 1:]
            eroded[:, :, 1:] &= masks[:, :, :-1]
            eroded[:, :, :-1] &= masks[:, :, 1:]
            eroded[:, [0, -1], :] = False
            eroded[:, :, [0, -1]] = False
            outlines = masks & ~eroded
            outlines[0] = False
            visible_mass = (visible_support[None] * masks).sum((1, 2)).numpy()
            eligible = visible_mass >= np.prod(SHAPE) / 144.
            eligible_counts += eligible
            centre = section['target_centre_ccf_coordinates_ap_dv_ml_um_float64']
            slab = section['target_psf_ccf_coordinates_ap_dv_ml_um_float64']
            canonical_slab = slab[:, :, ::-1].copy() if reflection[0] else slab
            canonical_centre = section['canonical_anatomy_centre_ccf_ap_dv_ml_um_float64']
            fit = section['canonical_anatomy_plane_fit']['arrays']
            canonical_ouv = fit['physical_ouv_ap_dv_ml_um_float64']
            frame = physical_ouv_to_frame(torch.from_numpy(canonical_ouv))[1]
            director = np.einsum('s,shwc->hwc', weights * offsets, canonical_slab - canonical_centre[None]) / (weights * offsets**2).sum()
            residual = torch.from_numpy(canonical_centre - fit['fitted_coordinate_raster_ap_dv_ml_um_float64'])
            local_r = torch.einsum('ji,hwj->ihw', frame, residual) / 200.
            local_d = torch.einsum('ji,hwj->ihw', frame, torch.from_numpy(director) - frame[:, 2]) / .1
            field_8 = F.interpolate(torch.cat((local_r, local_d))[None], (8, 8), mode='bilinear', align_corners=False)[0].float().numpy()
            sparse_approximate = slab[:, pixel_y[:, None], pixel_x[None, :], :]
            sparse_subject = section['subject_psf_coordinates_ap_dv_ml_um_float64'][:, pixel_y[:, None], pixel_x[None, :], :]
            sparse = exact_mapper(sparse_subject)
            sparse_error = np.linalg.norm(sparse_approximate - sparse, axis=-1)
            assert sparse_error.max() <= 5., 'section sparse queries exceed the fixed5um inverse-map approximation limit'
            centre_f32, sparse_f32 = centre.astype(np.float32), sparse.astype(np.float32)
            arrays_path = output / f'section_{index:08d}.npz'
            np.savez_compressed(arrays_path, shared_pre_brush_image=before_brush.numpy(),
                clean_atlas_intensity=rendered[0].numpy(), finite_support=support.numpy(),
                tissue_retained=retained.numpy(), brush_keep_mask=masks.numpy(), brush_outline=outlines.numpy(),
                outline_available=np.array([False, True, True]), eligible=eligible,
                target_centre_um=centre_f32, target_psf_um=sparse_f32, psf_pixel_y=pixel_y, psf_pixel_x=pixel_x,
                canonical_ouv_um=canonical_ouv, subject_ouv_um=ouv, canonical_field_8=field_8,
                offsets_um=offsets.astype(np.float32), weights=weights.astype(np.float32),
                horizontal_reflection=np.array(reflection[0]))
            with arrays_path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            size = arrays_path.stat().st_size
            section_bytes += size
            record = {'section_index': index, 'section_id': section_id, 'file': arrays_path.name, 'sha256': digest,
                'seed_prefix': prefix, 'seed_branches': 'geometry0,appearance1,TorchCPU-noise2',
                'framing': framing, 'appearance': parameters, 'thickness_um': float(offsets[-1] - offsets[0]),
                'horizontal_reflection': reflection[0], 'visible_support_mass': visible_mass.tolist(),
                'eligible_by_mode': eligible.tolist(), 'canonical_fit': section['canonical_anatomy_plane_fit']['diagnostics'],
                'render_mapping_exact_sparse_error_max_um': float(sparse_error.max()),
                'render_mapping_exact_sparse_error_p99_um': float(np.quantile(sparse_error, .99)),
                'coordinate_fp32_cast_max_abs_um': float(max(np.abs(centre - centre_f32).max(), np.abs(sparse - sparse_f32).max())),
                'bytes': size, 'seconds': time.perf_counter() - tick}
            index_file.write(json.dumps(record) + '\n')
            if index == 0 or (index + 1) % 64 == 0:
                index_file.flush()
                print(json.dumps({'event': 'TRAIN_sections_generated', 'subject': subject['animal_index'],
                    'sections': index + 1, 'planned': PER_SUBJECT, 'last_seconds': record['seconds'],
                    'subject_elapsed_seconds': time.perf_counter() - shard_started, 'bytes': section_bytes,
                    'eligible_by_mode': eligible_counts.tolist(), 'first_section_not_benchmark': index == 0}), flush=True)
    shard = {'subject_id': subject['subject_id'], 'animal_id': subject['animal_id'], 'split': 'train',
        'sections': PER_SUBJECT, 'observations': PER_SUBJECT * 3, 'protocol_sha256': protocol_sha,
        'subject_sha256': hashlib.sha256((output / 'subject.json').read_bytes()).hexdigest(),
        'sections_sha256': hashlib.sha256((output / 'sections.jsonl').read_bytes()).hexdigest(),
        'bytes': section_bytes, 'eligible_by_mode': eligible_counts.tolist(), 'seconds': time.perf_counter() - shard_started}
    (output / 'completed.json').write_text(json.dumps(shard, indent=2), encoding='utf8')
    shards.append({'directory': output.relative_to(OUTPUT).as_posix(),
        'completed_sha256': hashlib.sha256((output / 'completed.json').read_bytes()).hexdigest(), **shard})
    print(json.dumps({'event': 'TRAIN_subject_shard_frozen', **shard}), flush=True)
    del plan, parameters_gpu, displacement, baked
for name, expected in source_hashes.items():
    assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected
completed = {'protocol_sha256': protocol_sha, 'shards': shards, 'section_count': sum(r['sections'] for r in shards),
    'observation_count': sum(r['observations'] for r in shards), 'bytes': sum(r['bytes'] for r in shards),
    'seconds': time.perf_counter() - started, 'synthetic_only': True, 'new_development_sections': 0}
(OUTPUT / 'completed.json').write_text(json.dumps(completed, indent=2), encoding='utf8')
print(json.dumps({'event': 'expanded_TRAIN_data_frozen', **completed}), flush=True)
