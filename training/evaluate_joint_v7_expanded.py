"""Post-exit native evaluation of the whole 30,000-update v7 model.

Same synthetic DEV96 cases as the initial native evaluation, plus the new
full-canvas real DEV192 inputs. Real affine errors are weak-label agreement,
not expert anatomical landmark error. No benchmark or calibration claim.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import time

import numpy as np
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_inference_v7 import load_joint_v7_checkpoint, infer_joint_v7

PARENT = ROOT / 'runs/joint_v7_expanded_joint_001'
OUTPUT = ROOT / 'runs/joint_v7_expanded_joint_001_native_development'
SYNTHETIC = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
CHECKPOINT = PARENT / 'joint_step_30000.pt'
completion = json.loads((PARENT / 'completed.json').read_text())
assert completion['stage_updates'] == 24000 and completion['total_updates'] == 30000
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUTPUT.mkdir(parents=True, exist_ok=False)
bindings = {}


def bind(path, expected=None):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    bindings[str(path)] = digest


for path in (CHECKPOINT, PARENT / 'completed.json', PARENT / 'experiment.json', PARENT / 'identities.json', SYNTHETIC / 'completed.json',
             REAL / 'completed.json', Path(__file__), Path(__file__).with_name('arbitrary_plane_joint_inference_v7.py')):
    bind(path)
model, config = load_joint_v7_checkpoint(CHECKPOINT)
assert config['parent_step'] + config['stage_updates'] == 30000 and config['modes'] == 8
assert bindings[str(PARENT / 'identities.json')] == config['identities_sha256']
training_ids = json.loads((PARENT / 'identities.json').read_text())
for filename, digest in config['source_sha256'].items():
    if not filename.startswith('train_joint_v7'):
        bind(Path(__file__).parent / filename, digest)
real_completed = json.loads((REAL / 'completed.json').read_text())
for name, digest in real_completed['output_sha256'].items():
    bind(REAL / name, digest)
real_records = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
real_train = [r for r in real_records if r['training_split'] == 'train']
real_development = [r for r in real_records if r['training_split'] == 'development']
for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id', 'source_image_sha256'):
    assert not {r[key] for r in real_train} & {r[key] for r in real_development}
assert len(real_train) == 1280 and len({r['animal_id'] for r in real_train}) == 58
assert len(real_development) == 64 and len({r['animal_id'] for r in real_development}) == 6
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
assert real_images.shape == (1344, 1, 192, 192)
with np.load(REAL / 'geometry.npz') as a:
    real_affines = a['model_pixel_to_ap_dv_ml_um'].copy()
    real_thickness = a['thickness_um'].copy()
    real_is_train = a['is_train'].copy()

samples = []
records = json.loads((SYNTHETIC / 'completed.json').read_text())['sections']
for record in records:
    if record['lineage']['split'] != 'development':
        continue
    for name, digest in record['artifact_sha256'].items():
        bind(SYNTHETIC / name, digest)
    metadata = json.loads((SYNTHETIC / record['artifacts']['metadata']).read_text())
    with np.load(SYNTHETIC / record['artifacts']['arrays']) as a:
        truth = a[metadata['target_centre_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']].astype(np.float32)
        normal = a[metadata['canonical_anatomy_plane_fit']['arrays']['fitted_plane_unit_normal_ap_dv_ml_float64']['__ndarray__']].astype(np.float32)
        z = a[metadata['axial_offsets_um_float64']['__ndarray__']].astype(np.float32)
        w = a[metadata['axial_weights_float64']['__ndarray__']].astype(np.float32)
        for observation in metadata['observations']:
            image = a[observation['image_outline_availability_float32']['__ndarray__']]
            image = np.concatenate((image, np.zeros((2, *image.shape[-2:]), dtype=np.float32)))
            samples.append((image, truth, normal, a[observation['visible_finite_support_float32']['__ndarray__']], z, w,
                {**observation['lineage'], 'mode': observation['selected_mode'], 'eligible': observation['support_information_eligible'],
                 'domain': 'synthetic_96', 'group': observation['lineage']['subject_id'], 'shape_h_w': [96, 96],
                 'reference': 'exact observed curved centre-surface coordinates with visible finite support',
                 'source_metadata': str(SYNTHETIC / record['artifacts']['metadata']),
                 'source_arrays': str(SYNTHETIC / record['artifacts']['arrays'])}))
assert len(samples) == 384 and len({s[-1]['group'] for s in samples}) == 4
assert not {s[-1]['animal_id'] for s in samples} & {r['animal_id'] for r in training_ids['synthetic_sections']}
yy, xx = np.mgrid[:192, :192]
for row in real_development:
    index = row['array_row_index']
    assert not real_is_train[index]
    affine = real_affines[index]
    truth = affine[:, 2] + xx[..., None] * affine[:, 0] + yy[..., None] * affine[:, 1]
    normal = np.cross(affine[:, 0], affine[:, 1])
    normal /= np.linalg.norm(normal)
    image = np.concatenate((real_images[index].astype(np.float32), np.zeros((4, 192, 192), dtype=np.float32)))
    z = np.linspace(-.5, .5, 9, dtype=np.float32) * real_thickness[index]
    w = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float32) / 16
    samples.append((image, truth, normal, np.ones((192, 192), dtype=np.float32), z, w,
        {**row, 'mode': 'raw', 'eligible': True, 'domain': 'real_weak_affine_192', 'group': str(row['animal_id']),
         'shape_h_w': [192, 192], 'reference': 'original observed full-canvas weak Allen affine; not expert landmarks'}))
protocol = {'checkpoint': str(CHECKPOINT), 'input_sha256': bindings, 'samples': len(samples), 'modes': 8, 'reflections': 2,
    'selection': 'native prior minus anatomical mismatch minus .05 deformation difficulty; no reference-assisted selection',
    'normal_error': 'degrees acos(abs(dot(predicted canonical plane normal, reference canonical/observed plane normal)))',
    'coordinate_error': 'mean physical centre-surface error in um, weighted by known visible finite support for synthetic and uniformly over the full canvas for weak real affines',
    'unfitted_diagnostic': 'same selected component with zero deformation; prior_fitted_coordinate uses the fitted prior-selected branch, not its unwarped plane',
    'oracle': 'minimum reference coordinate error among all16 fitted branches; diagnostic upper bound only, not available inference',
    'aggregation': 'eligible observations averaged within synthetic subject or real donor, then groups equally weighted; domains/resolutions never pooled',
    'censoring': 'all predictions retained; synthetic eligibility fixed by preparation; zero-visible-mass coordinate scores are undefined, not zero error',
    'real_comparison': 'new192px GUI-photometry full-canvas inputs differ from the initial96px real inputs; do not interpret their cross-run metric difference as a paired model improvement',
    'uncertainty': 'all native covariance parameters and energy weights retained, explicitly uncalibrated',
    'scope': 'internal development only; no public benchmark, expert anatomical landmark metric, real deformation ground truth or confidence-coverage claim'}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
shutil.copyfile(__file__, OUTPUT / Path(__file__).name)
atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
started, rows = time.perf_counter(), []
for index, (image, truth, normal, support, z, w, identity) in enumerate(samples):
    tick = time.perf_counter()
    result = infer_joint_v7(model, torch.from_numpy(image[None]).cuda(), atlas,
                            torch.from_numpy(z[None]).cuda(), torch.from_numpy(w[None]).cuda())
    state = result['state']
    centre, frame, basis = full_frame_state_to_components(state)
    normal_error = torch.rad2deg((frame[..., :, 2] * torch.as_tensor(normal)).sum(-1).abs().clamp(0, 1).acos()).numpy()
    height, width = image.shape[-2:]
    gy, gx = torch.meshgrid(torch.arange(height) / height, torch.arange(width) / width, indexing='ij')
    uv = torch.stack((gx, gy), -1) - .5
    plane = centre[..., None, None, :] + torch.einsum('...ij,hwj->...hwi', frame[..., :2] @ basis, uv)
    plane[:, 1] = plane[:, 1].flip(-2)
    mass = float(support.sum())
    if mass:
        coordinate_error = (np.linalg.norm(result['surface'].numpy() - truth, axis=-1) * support).sum((-2, -1)) / mass
        unfitted_error = (np.linalg.norm(plane.numpy() - truth, axis=-1) * support).sum((-2, -1)) / mass
    else:
        coordinate_error = unfitted_error = np.full((8, 2), np.nan)
    selected = result['selected_component']
    prior = np.unravel_index(int(result['prior_log_weight'].argmax()), result['prior_log_weight'].shape)
    path = OUTPUT / f'prediction_{index:04d}.npz'
    np.savez_compressed(path, **{key: value.numpy() for key, value in result.items() if torch.is_tensor(value)},
        normal_error_deg=normal_error, coordinate_error_um=coordinate_error, unfitted_coordinate_error_um=unfitted_error)
    with path.open('rb') as stream:
        output_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    rows.append({**identity, 'prediction_file': path.name, 'prediction_sha256': output_sha,
        'selected_component': list(selected), 'prior_component': [int(v) for v in prior], 'visible_mass': mass,
        'normal_deg': float(normal_error[selected]), 'coordinate_um': float(coordinate_error[selected]) if mass else None,
        'prior_normal_deg': float(normal_error[prior]), 'prior_fitted_coordinate_um': float(coordinate_error[prior]) if mass else None,
        'unfitted_selected_coordinate_um': float(unfitted_error[selected]) if mass else None,
        'oracle_component_coordinate_um': float(coordinate_error.min()) if mass else None,
        'fit_cost': float(result['mismatch'][selected]), 'difficulty': float(result['difficulty'][selected]),
        'seconds': time.perf_counter() - tick})
    if (index + 1) % 48 == 0 or index + 1 == len(samples):
        print(json.dumps({'evaluated': index + 1, 'total': len(samples), 'seconds': time.perf_counter() - started}), flush=True)
(OUTPUT / 'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
summary = {}
keys = ('normal_deg', 'coordinate_um', 'prior_normal_deg', 'prior_fitted_coordinate_um',
        'unfitted_selected_coordinate_um', 'oracle_component_coordinate_um')
for domain in ('synthetic_96', 'real_weak_affine_192'):
    groups = {}
    for group in sorted({r['group'] for r in rows if r['domain'] == domain}):
        selected_rows = [r for r in rows if r['domain'] == domain and r['group'] == group and r['eligible']]
        assert selected_rows and all(r['visible_mass'] > 0 for r in selected_rows)
        groups[group] = {'observations': len(selected_rows), **{key: float(np.mean([r[key] for r in selected_rows])) for key in keys}}
    summary[domain] = {'groups': groups, 'group_macro': {key: float(np.mean([r[key] for r in groups.values()])) for key in keys}}
(OUTPUT / 'completed.json').write_text(json.dumps({'summary': summary, 'input_sha256': bindings,
    'evaluation_sha256': hashlib.sha256((OUTPUT / 'evaluation.json').read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((OUTPUT / 'rows.json').read_bytes()).hexdigest(),
    'rows': len(rows), 'seconds': time.perf_counter() - started, 'calibrated': False, 'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps(summary, indent=2), flush=True)
