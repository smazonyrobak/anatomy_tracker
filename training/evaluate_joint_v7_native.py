"""Post-exit internal evaluation: predicted poses AND their actual anatomical fit.

Real Allen affine references are weak labels, not blinded expert ground truth.
Never run this against a live training output tree.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import hashlib
import json
import time
import numpy as np
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_inference_v7 import load_joint_v7_checkpoint, infer_joint_v7

PARENT = ROOT / 'runs/joint_v7_direct_joint_001'
OUTPUT = ROOT / 'runs/joint_v7_direct_joint_001_native_development_002'
DATA = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
REAL = ROOT / 'runs/joint_v6_imagekey_retrieval_001_allen_raw'
TRAIN_REAL = ROOT / 'data/allen_real_training_expansion_20260929/union_training_index.jsonl'
CHECKPOINT = PARENT / 'joint_step_06000.pt'
assert json.loads((PARENT / 'completed.json').read_text())['steps'] == 6000
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUTPUT.mkdir(parents=True, exist_ok=False)
inputs = [CHECKPOINT, PARENT / 'completed.json', DATA / 'completed.json', REAL / 'image_geometry.jsonl',
          REAL / 'raw_model_input.npy', TRAIN_REAL, Path(__file__)]
bindings = {}
for path in inputs:
    with path.open('rb') as stream:
        bindings[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
model, config = load_joint_v7_checkpoint(CHECKPOINT)
for filename, digest in config['source_sha256'].items():
    assert hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest() == digest
real_records = [json.loads(line) for line in (REAL / 'image_geometry.jsonl').read_text().splitlines()]
real_train = [json.loads(line) for line in TRAIN_REAL.read_text().splitlines()]
for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id'):
    assert not {r[key] for r in real_records} & {r[key] for r in real_train}
assert len(real_records) == 64 and len({r['animal_id'] for r in real_records}) == 6
assert len(real_train) == 1280 and len({r['animal_id'] for r in real_train}) == 58
real_images = np.load(REAL / 'raw_model_input.npy')
atlas_array, annotation = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array[:1]).cuda()
del atlas_array, annotation
samples = []
records = json.loads((DATA / 'completed.json').read_text())['sections']
for record in records:
    if record['lineage']['split'] != 'development':
        continue
    metadata = json.loads((DATA / record['artifacts']['metadata']).read_text())
    with np.load(DATA / record['artifacts']['arrays']) as arrays:
        truth = arrays[metadata['target_centre_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']].astype(np.float32)
        normal = arrays[metadata['canonical_anatomy_plane_fit']['arrays']['fitted_plane_unit_normal_ap_dv_ml_float64']['__ndarray__']].astype(np.float32)
        z = arrays[metadata['axial_offsets_um_float64']['__ndarray__']].astype(np.float32)
        w = arrays[metadata['axial_weights_float64']['__ndarray__']].astype(np.float32)
        for row in metadata['observations']:
            image = arrays[row['image_outline_availability_float32']['__ndarray__']]
            image = np.concatenate((image, np.zeros((2, *image.shape[-2:]), dtype=np.float32)))
            samples.append((image, truth, normal, arrays[row['visible_finite_support_float32']['__ndarray__']], z, w,
                            {**row['lineage'], 'mode': row['selected_mode'], 'eligible': row['support_information_eligible'],
                             'domain': 'synthetic', 'group': row['lineage']['subject_id']}))
yy, xx = np.mgrid[:96, :96]
for image, row in zip(real_images, real_records):
    affine = np.asarray(row['model_pixel_to_ap_dv_ml_um'])
    truth = affine[:, 2] + xx[..., None] * affine[:, 0] + yy[..., None] * affine[:, 1]
    channels = np.concatenate((image, np.zeros((4, 96, 96), dtype=np.float32)))
    z = np.linspace(-.5, .5, 9, dtype=np.float32) * row['section_thickness_um']
    w = np.array([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float32) / 16
    samples.append((channels, truth, np.asarray(row['truth_normal_ap_dv_ml']),
                    np.ones((96, 96), dtype=np.float32), z, w,
                    {**row, 'mode': 'raw', 'eligible': True, 'domain': 'real_weak_affine', 'group': str(row['animal_id'])}))

started, rows = time.perf_counter(), []
for index, (image, truth, normal, support, z, w, identity) in enumerate(samples):
    tick = time.perf_counter()
    result = infer_joint_v7(model, torch.from_numpy(image[None]).cuda(), atlas,
                            torch.from_numpy(z[None]).cuda(), torch.from_numpy(w[None]).cuda())
    state = result['state']
    _, frame, _ = full_frame_state_to_components(state)
    normal_error = torch.rad2deg((frame[..., :, 2] * torch.as_tensor(normal)).sum(-1).abs().clamp(0, 1).acos()).numpy()
    distance = np.linalg.norm(result['surface'].numpy() - truth, axis=-1)
    coordinate_error = (distance * support).sum((-2, -1)) / max(float(support.sum()), 1.)
    selected = result['selected_component']
    prior_selected = np.unravel_index(int(result['prior_log_weight'].argmax()), result['prior_log_weight'].shape)
    raw = {key: value.numpy() for key, value in result.items() if torch.is_tensor(value)}
    # Preserve exact native output, including correlated uncertainty parameters.
    np.savez_compressed(OUTPUT / f'prediction_{index:04d}.npz', **raw,
                        normal_error_deg=normal_error, coordinate_error_um=coordinate_error)
    rows.append({**identity, 'prediction_file': f'prediction_{index:04d}.npz',
                 'selected_component': list(selected), 'prior_component': [int(v) for v in prior_selected],
                 'normal_deg': float(normal_error[selected]), 'coordinate_um': float(coordinate_error[selected]),
                 'prior_normal_deg': float(normal_error[prior_selected]),
                 'oracle_component_coordinate_um': float(coordinate_error.min()),
                 'fit_cost': float(result['mismatch'][selected]), 'difficulty': float(result['difficulty'][selected]),
                 'seconds': time.perf_counter() - tick})
    if (index + 1) % 48 == 0:
        print(json.dumps({'evaluated': index + 1, 'total': len(samples), 'seconds': time.perf_counter() - started}), flush=True)
(OUTPUT / 'rows.json').write_text(json.dumps(rows, indent=2))
summary = {}
for domain in ('synthetic', 'real_weak_affine'):
    group_metrics = {}
    for group in sorted({r['group'] for r in rows if r['domain'] == domain}):
        selected_rows = [r for r in rows if r['domain'] == domain and r['group'] == group and r['eligible']]
        group_metrics[group] = {key: float(np.mean([r[key] for r in selected_rows]))
                               for key in ('normal_deg', 'coordinate_um', 'prior_normal_deg', 'oracle_component_coordinate_um')}
    summary[domain] = {'groups': group_metrics,
                       'group_macro': {key: float(np.mean([r[key] for r in group_metrics.values()]))
                                       for key in next(iter(group_metrics.values()))}}
(OUTPUT / 'completed.json').write_text(json.dumps({'summary': summary, 'input_sha256': bindings,
    'rows': len(rows), 'seconds': time.perf_counter() - started, 'calibrated': False,
    'scope': 'internal development only; real coordinates are weak full-canvas affines, not anatomical landmark error'}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
