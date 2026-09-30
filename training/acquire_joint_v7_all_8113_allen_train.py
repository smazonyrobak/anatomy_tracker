"""All frozen 58-donor TRAIN sections; full GUI canvas, unchanged 64-row DEV.

New I: output only. Reuse original1280 images/pixels/geometry; download6833
missing images from already-frozen metadata. No benchmark, new donor, mask,
legacy model, learned features or pseudolabels. CPU preparation only.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ.update(TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'), CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True

import hashlib
import inspect
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from source.proprietary_trajectory_tool import as_gray, normalize_u8, downsample_for_display
from source.joint_slice_input import prepare_joint_slice_input
from training.acquire_allen_real_histology_images import deterministic_section_selection, _download_record
from training.allen_real_histology_metadata import _canonical_bytes
from training.arbitrary_plane_geometry import physical_ouv_to_frame, normalized_raster_to_ccf
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components

REPO = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'data/allen_real_development_20260928/metadata'
OLD_INDEX = ROOT / 'data/allen_real_training_expansion_20260929/union_training_index.jsonl'
PARENT = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'data/joint_v7_allen_fullcanvas_8113_192_001'
SIDE, TRAIN_COUNT, DEV_COUNT = 192, 8113, 64
cv2.setNumThreads(1)
torch.set_num_threads(1)
started = time.perf_counter()
assert hashlib.sha256((COHORT / 'sections.jsonl').read_bytes()).hexdigest() == 'f209370dc548e36b4a887524041b5d5c92ec6c2b91f770b5b64d8a543e4b7a9c'
assert hashlib.sha256((COHORT / 'experiments.jsonl').read_bytes()).hexdigest() == '67b19f91610169371e4a824b7e1095e3ab2e1b9a5720a7c04a0cfb99afbf401b'
sections = [json.loads(line) for line in (COHORT / 'sections.jsonl').read_text().splitlines()]
experiments = {r['experiment_id']: r for r in map(json.loads, (COHORT / 'experiments.jsonl').read_text().splitlines())}
old = [json.loads(line) for line in OLD_INDEX.read_text().splitlines()]
old_ids = {r['section_id'] for r in old}
selected = deterministic_section_selection(sections, {'development_train': TRAIN_COUNT})
missing = [r for r in selected if r['section_id'] not in old_ids]
assert len(old) == len(old_ids) == 1280 and len(missing) == 6833
assert {r['animal_id'] for r in selected} == {r['animal_id'] for r in old}
assert len({r['animal_id'] for r in selected}) == 58
assert {r['animal_id'] for r in selected}.isdisjoint({14452, 15219, 15336, 15439, 15447, 15935})
parent_summary = json.loads((PARENT / 'completed.json').read_text())
for name, expected in parent_summary['output_sha256'].items():
    with (PARENT / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
parent_records = [json.loads(line) for line in (PARENT / 'records.jsonl').read_text().splitlines()]
assert len(parent_records) == 1344 and [r['section_id'] for r in parent_records[:1280]] == [r['section_id'] for r in old]
assert all(r['training_split'] == 'development' for r in parent_records[1280:])
OUT.mkdir(parents=True, exist_ok=False)
for source, name in ((Path(__file__), 'acquisition_preparation_source.py'),
                     (OLD_INDEX, 'original_1280_union_training_index.jsonl'),
                     (PARENT / 'completed.json', 'parent_fullcanvas_completed.json'),
                     (REPO / 'training/acquire_allen_real_histology_images.py', 'download_source.py'),
                     (REPO / 'source/joint_slice_input.py', 'joint_slice_input_source.py')):
    shutil.copyfile(source, OUT / name)
(OUT / 'gui_photometry_source.py').write_text('import cv2\nimport numpy as np\n\n' + '\n'.join(
    inspect.getsource(function) for function in (as_gray, normalize_u8, downsample_for_display)), encoding='utf8')
(OUT / 'planned_additional_sections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in missing), encoding='utf8')
source_paths = [Path(__file__), OLD_INDEX, COHORT / 'sections.jsonl', COHORT / 'experiments.jsonl',
                PARENT / 'completed.json', REPO / 'source/joint_slice_input.py',
                REPO / 'training/arbitrary_plane_geometry.py', REPO / 'training/arbitrary_plane_full_frame_primitives.py']
source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
images = np.lib.format.open_memmap(OUT / 'images.npy', mode='w+', dtype=np.float16,
                                  shape=(TRAIN_COUNT + DEV_COUNT, 1, SIDE, SIDE))
parent_images = np.load(PARENT / 'images.npy', mmap_mode='r')
images[:1280] = parent_images[:1280]
images[TRAIN_COUNT:] = parent_images[1280:]
states, affines, model_to_download, thicknesses = {}, {}, {}, {}
with np.load(PARENT / 'geometry.npz', allow_pickle=False) as geometry:
    for target, key in ((states, 'state'), (affines, 'model_pixel_to_ap_dv_ml_um'),
                        (model_to_download, 'model_pixel_to_downloaded_pixel'), (thicknesses, 'thickness_um')):
        for i, value in enumerate(geometry[key]):
            target[i if i < 1280 else TRAIN_COUNT + i - 1280] = value.copy()
records = {i if i < 1280 else TRAIN_COUNT + i - 1280:
           {**record, 'array_row_index': i if i < 1280 else TRAIN_COUNT + i - 1280,
            'parent_prepared_record_path': str(PARENT / 'records.jsonl'), 'parent_prepared_array_row_index': i}
           for i, record in enumerate(parent_records)}
pixels = np.array([[0., 0.], [SIDE - 1., 0.], [0., SIDE - 1.], [SIDE - 1., SIDE - 1.], [(SIDE - 1.) / 2, (SIDE - 1.) / 2]])
local = threading.local()


def download(item):
    rank, section = item
    if not hasattr(local, 'session'):
        local.session = requests.Session()
        local.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))))
    try:
        record = _download_record(local.session.get, section, OUT, rank)
        record['retrieved_at_utc'] = datetime.now(timezone.utc).isoformat()
        return rank, record, None
    except Exception as error:
        return rank, None, {'section_id': section['section_id'], 'animal_id': section['animal_id'],
                           'requested_url': section['image_download_url'], 'error': str(error)}


downloaded, failed, downloaded_bytes, max_coordinate_error, max_quantization_error = 0, 0, 0, 0., 0.
print(json.dumps({'started': True, 'new_train_downloads': len(missing), 'reused_train': 1280,
                  'unchanged_development': DEV_COUNT, 'download_workers': 8, 'output': str(OUT)}), flush=True)
with (OUT / 'downloaded_additional_images.jsonl').open('w', encoding='utf8') as manifest, \
     (OUT / 'failures.jsonl').open('w', encoding='utf8') as failures, ThreadPoolExecutor(max_workers=8) as pool:
    for rank, downloaded_record, error in pool.map(download, enumerate(missing)):
        if error is not None:
            failed += 1
            failures.write(json.dumps(error) + '\n'); failures.flush()
            print(json.dumps({'download_error': error, 'failed_count': failed}), flush=True)
            continue
        section = missing[rank]
        experiment = experiments[section['experiment_id']]
        manifest.write(json.dumps(downloaded_record) + '\n'); manifest.flush()
        path = OUT / downloaded_record['relative_path']
        image_bytes = path.read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == downloaded_record['sha256']
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        display, display_scale = downsample_for_display(as_gray(decoded))
        assert display_scale == 1., 'New frozen-cohort image exceeds GUI display-size contract'
        weight_image = normalize_u8(display)
        prepared = prepare_joint_slice_input(weight_image.astype(np.float32) / 255., np.eye(3),
                                             weight_image.shape, (SIDE, SIDE))
        i = 1280 + rank
        images[i] = prepared['channels'][:1].astype(np.float16)
        max_quantization_error = max(max_quantization_error, float(np.abs(images[i].astype(np.float32) - prepared['channels'][:1]).max()))
        download_from_model = np.linalg.inv(prepared['raw_to_model_xy'])
        pyramid_scale = 2 ** int(parse_qs(urlparse(downloaded_record['requested_url']).query)['downsample'][0])
        assert pyramid_scale == 32
        full_from_download = np.array([[pyramid_scale, 0., (pyramid_scale - 1.) / 2],
                                      [0., pyramid_scale, (pyramid_scale - 1.) / 2], [0., 0., 1.]])
        tsv, tvr = np.asarray(section['alignment2d_tsv']), np.asarray(experiment['alignment3d_tvr'])
        volume_from_full = np.array([[tsv[0], tsv[1], tsv[4]], [tsv[2], tsv[3], tsv[5]],
                                     [0., 0., section['section_number'] * experiment['section_thickness_um']]])
        ccf_from_full = tvr[:9].reshape(3, 3) @ volume_from_full
        ccf_from_full[:, 2] += tvr[9:] + 12.5
        affine = ccf_from_full @ full_from_download @ download_from_model
        ouv = np.concatenate((affine[:, 2], SIDE * affine[:, 0], SIDE * affine[:, 1]))
        state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(ouv))).float()
        reconstructed = normalized_raster_to_ccf(*full_frame_state_to_components(state), torch.from_numpy(pixels / SIDE).float()).numpy()
        expected = np.column_stack((pixels, np.ones(len(pixels)))) @ affine.T
        error_um = float(np.linalg.norm(reconstructed - expected, axis=-1).max())
        assert error_um < .02, 'Affine/state pixel-centre mapping disagreement'
        max_coordinate_error = max(max_coordinate_error, error_um)
        states[i], affines[i], model_to_download[i], thicknesses[i] = state.numpy(), affine, download_from_model, experiment['section_thickness_um']
        identity = {key: section[key] for key in ('animal_id', 'animal_id_namespace', 'animal_partition_key', 'specimen_id',
                    'experiment_id', 'section_id', 'section_id_namespace', 'section_number', 'split')}
        records[i] = {**identity, 'array_row_index': i, 'training_split': 'train',
            'source_image_path': str(path), 'source_image_sha256': downloaded_record['sha256'],
            'source_requested_url': downloaded_record['requested_url'],
            'source_section_record_sha256': downloaded_record['source_section_record_sha256'],
            'source_experiment_record_sha256': hashlib.sha256(_canonical_bytes(experiment)).hexdigest(),
            'source_experiment_response_sha256': downloaded_record['source_experiment_response_sha256'],
            'source_metadata_index': str(COHORT / 'sections.jsonl'), 'source_geometry_role': 'original Allen TSV/TVR composition',
            'alignment2d_tsv': tsv.tolist(), 'alignment3d_tvr': tvr.tolist(), 'api_to_model_shift_um': [12.5] * 3,
            'downloaded_shape_h_w': list(weight_image.shape), 'model_shape_h_w': [SIDE, SIDE],
            'gui_weight_image_sha256': hashlib.sha256(weight_image.tobytes()).hexdigest(),
            'model_pixel_to_downloaded_pixel': download_from_model.tolist(), 'model_pixel_to_ap_dv_ml_um': affine.tolist(),
            'section_thickness_um': experiment['section_thickness_um'], 'horizontal_reflection': False,
            'outline_available': False, 'mark_available': False,
            'label_role': 'weak Allen registered affine, not blinded expert alignment; no native deformation reference',
            'learned_source_dependency': 'none'}
        downloaded += 1
        downloaded_bytes += downloaded_record['bytes']
        if downloaded % 512 == 0 or downloaded + failed == len(missing):
            images.flush()
            print(json.dumps({'additional_downloaded_and_prepared': downloaded, 'planned': len(missing),
                              'failed_count': failed, 'downloaded_bytes': downloaded_bytes,
                              'seconds': time.perf_counter() - started}), flush=True)
assert not failed, f'{failed} planned downloads failed; incomplete output is not a training dataset'
assert len(records) == TRAIN_COUNT + DEV_COUNT
np.testing.assert_array_equal(images[:1280], parent_images[:1280])
np.testing.assert_array_equal(images[TRAIN_COUNT:], parent_images[1280:])
images.flush()
del images, parent_images
np.savez(OUT / 'geometry.npz', state=np.stack([states[i] for i in range(len(records))]),
         model_pixel_to_ap_dv_ml_um=np.stack([affines[i] for i in range(len(records))]),
         model_pixel_to_downloaded_pixel=np.stack([model_to_download[i] for i in range(len(records))]),
         reflection=np.zeros(len(records), dtype=np.int64), thickness_um=np.array([thicknesses[i] for i in range(len(records))], dtype=np.float32),
         is_train=np.arange(len(records)) < TRAIN_COUNT)
(OUT / 'records.jsonl').write_text(''.join(json.dumps(records[i]) + '\n' for i in range(len(records))), encoding='utf8')
output_hashes = {}
for name in ('images.npy', 'geometry.npz', 'records.jsonl', 'planned_additional_sections.jsonl',
             'downloaded_additional_images.jsonl', 'failures.jsonl', 'acquisition_preparation_source.py',
             'original_1280_union_training_index.jsonl', 'parent_fullcanvas_completed.json',
             'download_source.py', 'joint_slice_input_source.py', 'gui_photometry_source.py'):
    with (OUT / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
summary = {'training_images': TRAIN_COUNT, 'training_donors': 58, 'development_images': DEV_COUNT,
    'development_donors': 6, 'downloaded_new_images': downloaded, 'downloaded_new_image_bytes': downloaded_bytes,
    'reused_train_images': 1280, 'original1280_and_development64_pixels_exact': True,
    'source_sha256': source_hashes, 'output_sha256': output_hashes,
    'photometry': parent_summary['photometry'], 'display_curve': 'not applied', 'crop': 'none', 'automatic_segmentation': 'none',
    'geometry': 'original weak Allen TSV/TVR affine plus unchanged12.5um API-to-model shift and pixel-centre resize',
    'label_scope': parent_summary['label_scope'], 'shape': [TRAIN_COUNT + DEV_COUNT, 1, SIDE, SIDE], 'dtype': 'float16',
    'max_float16_pixel_error': max_quantization_error, 'max_state_affine_coordinate_error_um': max_coordinate_error,
    'new_donors': 0, 'development_or_benchmark_downloads': 0, 'probabilities_calibrated': False,
    'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started}
(OUT / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
