"""Full downloaded Allen images with GUI photometry and weak affine labels.

No prior-model pixels, weights, features, crops, inferred masks or pseudolabels.
These donor-disjoint TRAIN/DEV labels are not independent expert ground truth.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import time

import cv2
import numpy as np
import torch

from source.proprietary_trajectory_tool import as_gray, normalize_u8, downsample_for_display
from source.joint_slice_input import prepare_joint_slice_input
from training.arbitrary_plane_geometry import physical_ouv_to_frame, normalized_raster_to_ccf
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components

TRAIN_INDEX = ROOT / 'data/allen_real_training_expansion_20260929/union_training_index.jsonl'
DEV_INDEX = ROOT / 'runs/joint_v6_imagekey_retrieval_001_allen_raw/image_geometry.jsonl'
COHORT = ROOT / 'data/allen_real_development_20260928'
IMAGE_INDEX = COHORT / 'images/images.jsonl'
OUTPUT = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
SIDE = 192
cv2.setNumThreads(4)
torch.set_num_threads(4)
train = [json.loads(line) for line in TRAIN_INDEX.read_text().splitlines()]
dev = [json.loads(line) for line in DEV_INDEX.read_text().splitlines()]
download_records = {r['section_id']: r for r in map(json.loads, IMAGE_INDEX.read_text().splitlines())}
assert len(train) == 1280 and len(dev) == 64
assert len({r['animal_id'] for r in train}) == 58 and len({r['animal_id'] for r in dev}) == 6
for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id', 'actual_image_sha256'):
    assert not {r[key] for r in train} & {r[key] for r in dev}, key
assert len({r['section_id'] for r in train + dev}) == len(train + dev)
assert all(r['split'] == 'development_train' for r in train)
assert all(r['split'] == 'development_validation' for r in dev)
OUTPUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, OUTPUT / 'preparation_source.py')
repository = Path(__file__).resolve().parents[1]
source_files = [Path(__file__), repository / 'source/proprietary_trajectory_tool.py',
                repository / 'source/joint_slice_input.py', repository / 'training/arbitrary_plane_geometry.py',
                repository / 'training/arbitrary_plane_full_frame_primitives.py', TRAIN_INDEX, DEV_INDEX,
                IMAGE_INDEX, COHORT / 'images/manifest.json']
source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
images = np.empty((len(train + dev), 1, SIDE, SIDE), dtype=np.float16)
states, affines, model_to_download, thicknesses, records = [], [], [], [], []
max_quantization_error, max_coordinate_error = 0., 0.
pixels = np.array([[0., 0.], [SIDE - 1., 0.], [0., SIDE - 1.],
                   [SIDE - 1., SIDE - 1.], [(SIDE - 1.) / 2, (SIDE - 1.) / 2]])
started = time.perf_counter()
print('Preparing full-canvas real inputs from downloaded JPEGs; CPU only', flush=True)
for i, record in enumerate(train + dev):
    is_train = i < len(train)
    if is_train:
        path = Path(record['source_image_path'])
    else:
        downloaded = download_records[record['section_id']]
        assert downloaded['sha256'] == record['actual_image_sha256']
        assert downloaded['split'] == 'development_validation'
        path = COHORT / 'images' / downloaded['relative_path']
    image_bytes = path.read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == record['actual_image_sha256']
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    display, display_scale = downsample_for_display(as_gray(decoded))
    assert display_scale == 1., 'This frozen cohort is below the GUI 1800-pixel display limit'
    weight_image = normalize_u8(display)
    prepared = prepare_joint_slice_input(weight_image.astype(np.float32) / 255., np.eye(3),
                                         weight_image.shape, (SIDE, SIDE))
    images[i] = prepared['channels'][:1].astype(np.float16)
    max_quantization_error = max(max_quantization_error,
                                 float(np.abs(images[i].astype(np.float32) - prepared['channels'][:1]).max()))
    download_from_model = np.linalg.inv(prepared['raw_to_model_xy'])
    # Inverse resize uses downloaded pixel = (model pixel + .5) * scale - .5.
    ccf_from_download = (np.asarray(record['model_pixel_to_ap_dv_ml_um'])
                         @ np.linalg.inv(np.asarray(record['model_pixel_to_downloaded_pixel'])))
    affine = ccf_from_download @ download_from_model
    # The observed affine itself defines a proper frame and positive basis.
    # Reflection=0 is a chart convention, not a guessed biological laterality.
    ouv = np.concatenate((affine[:, 2], SIDE * affine[:, 0], SIDE * affine[:, 1]))
    state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(ouv))).float()
    reconstructed = normalized_raster_to_ccf(*full_frame_state_to_components(state),
                                            torch.from_numpy(pixels / SIDE).float()).numpy()
    expected = np.column_stack((pixels, np.ones(len(pixels)))) @ affine.T
    error = float(np.linalg.norm(reconstructed - expected, axis=-1).max())
    assert error < .02, 'Affine/state pixel-centre mapping disagreement'
    max_coordinate_error = max(max_coordinate_error, error)
    states.append(state.numpy())
    affines.append(affine)
    model_to_download.append(download_from_model)
    thicknesses.append(record['section_thickness_um'])
    identity = {key: record[key] for key in ('animal_id', 'animal_id_namespace', 'animal_partition_key',
                                            'specimen_id', 'experiment_id', 'section_id',
                                            'section_id_namespace', 'section_number', 'split')}
    records.append({**identity, 'array_row_index': i, 'training_split': 'train' if is_train else 'development',
                    'source_image_path': str(path), 'source_image_sha256': record['actual_image_sha256'],
                    'source_requested_url': record['requested_url'],
                    'source_geometry_index': str(TRAIN_INDEX if is_train else DEV_INDEX),
                    'source_geometry_row_index': i if is_train else i - len(train),
                    'source_geometry_record_sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                    'source_section_record_sha256': record['source_section_record_sha256'],
                    'source_experiment_record_sha256': record['source_experiment_record_sha256'],
                    'downloaded_shape_h_w': list(weight_image.shape), 'model_shape_h_w': [SIDE, SIDE],
                    'gui_weight_image_sha256': hashlib.sha256(weight_image.tobytes()).hexdigest(),
                    'model_pixel_to_downloaded_pixel': download_from_model.tolist(),
                    'model_pixel_to_ap_dv_ml_um': affine.tolist(),
                    'section_thickness_um': record['section_thickness_um'], 'horizontal_reflection': False,
                    'outline_available': False, 'mark_available': False,
                    'label_role': 'weak Allen registered affine, not blinded expert alignment; no native deformation reference',
                    'learned_source_dependency': 'none'})
    if (i + 1) % 256 == 0 or i + 1 == len(images):
        print(json.dumps({'prepared': i + 1, 'total': len(images),
                          'seconds': time.perf_counter() - started}), flush=True)

np.save(OUTPUT / 'images.npy', images)
np.savez(OUTPUT / 'geometry.npz', state=np.stack(states),
         model_pixel_to_ap_dv_ml_um=np.stack(affines), model_pixel_to_downloaded_pixel=np.stack(model_to_download),
         reflection=np.zeros(len(images), dtype=np.int64), thickness_um=np.asarray(thicknesses, dtype=np.float32),
         is_train=np.arange(len(images)) < len(train))
(OUTPUT / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf8')
output_hashes = {}
for name in ('images.npy', 'geometry.npz', 'records.jsonl', 'preparation_source.py'):
    with (OUTPUT / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
summary = {'source_sha256': source_hashes, 'output_sha256': output_hashes,
           'training_images': len(train), 'training_donors': 58, 'development_images': len(dev), 'development_donors': 6,
           'animal_specimen_experiment_section_image_disjoint': True, 'shape': list(images.shape), 'dtype': str(images.dtype),
           'photometry': 'exact GUI as_gray channel arithmetic mean; normalize_u8 whole-image .2/99.8 percentiles, uint8 truncation; /255 then INTER_AREA full-canvas resize',
           'display_curve': 'not applied', 'crop': 'none', 'automatic_segmentation': 'none',
           'optional_channels': 'append four zero channels: outline, outline availability, marks, mark availability',
           'geometry': 'original weak downloaded-image affine composed with pixel-centre resize; state12 proper frame, horizontal reflection zero chart',
           'label_scope': 'Allen upstream affine labels are weak; no independently verified anatomy, deformation, calibration or final-test reference',
           'upstream_image_role': 'source manifests designated these images appearance/domain supervision; weak affine training use must remain explicitly qualified',
           'max_float16_pixel_error': max_quantization_error, 'max_state_affine_coordinate_error_um': max_coordinate_error,
           'calibrated': False, 'seconds': time.perf_counter() - started}
(OUTPUT / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
