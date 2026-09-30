"""Download and prepare only the frozen Allen TRAIN donor reservation on I:.

Run after joint_v7_reserved_train_metadata_001 is complete. Donor shards are
restartable; no development, calibration, final-test, or benchmark images are
requested. Allen affine labels are weak registration references, not truth.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ.update(TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'),
                  CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from source.proprietary_trajectory_tool import as_gray, normalize_u8, downsample_for_display
from source.joint_slice_input import prepare_joint_slice_input
from training.acquire_allen_real_histology_images import _download_record, _image_relative_path
from training.allen_real_histology_metadata import _canonical_bytes
from training.arbitrary_plane_geometry import physical_ouv_to_frame, normalized_raster_to_ccf
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components

REPO = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / 'data/allen_expansion_metadata_plan_20260930/proposed_donor_reservations.jsonl'
METADATA = ROOT / 'data/joint_v7_reserved_train_metadata_001'
OLD_TRAIN = ROOT / 'data/joint_v7_allen_fullcanvas_8113_192_001'
OUT = ROOT / 'data/joint_v7_reserved_train_images_192_001'
SIDE = 192
cv2.setNumThreads(1)
torch.set_num_threads(1)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
frozen = json.loads((METADATA / 'completed.json').read_text())
for name in ('experiments.jsonl', 'sections.jsonl', 'receipts.jsonl'):
    assert sha(METADATA / name) == frozen['files_sha256'][name], name
reservation_hash = sha(RESERVATION)
assert reservation_hash == frozen['files_sha256']['proposed_donor_reservations.jsonl']
for name, digest in frozen['raw_response_sha256'].items():
    assert sha(METADATA / name) == digest, name
reserved = [json.loads(line) for line in RESERVATION.read_text().splitlines()]
train_experiments = {(r['animal_id'], r['experiment_id']) for r in reserved if r['proposed_split'] == 'train'}
train_donors = {animal for animal, _ in train_experiments}
assert len(train_donors) == 1885
assert not train_donors.intersection(r['animal_id'] for r in reserved if r['proposed_split'] != 'train')
experiments = {}
for line in (METADATA / 'experiments.jsonl').open():
    row = json.loads(line)
    assert row['split'] == 'train' and (row['animal_id'], row['experiment_id']) in train_experiments
    experiments[row['experiment_id']] = row
assert len(experiments) == frozen['training_experiments']
old_summary = json.loads((OLD_TRAIN / 'completed.json').read_text())
assert old_summary['training_images'] == 8113
old_records = {}
with (OLD_TRAIN / 'records.jsonl').open() as stream:
    for line in islice(stream, old_summary['training_images']):
        old = json.loads(line)
        assert old['training_split'] == 'train' and old['animal_id'] in train_donors
        old_records[old['section_id']] = old
assert len(old_records) == old_summary['training_images']
sections, excluded = {}, []
for line in (METADATA / 'sections.jsonl').open():
    row = json.loads(line)
    assert row['split'] == 'train' and (row['animal_id'], row['experiment_id']) in train_experiments
    if not row['eligible_for_appearance_training']:
        continue
    experiment = experiments[row['experiment_id']]
    tsv, tvr = row['alignment2d_tsv'], experiment['alignment3d_tvr']
    thickness = experiment['section_thickness_um']
    valid = tsv is not None and tvr is not None and thickness is not None
    if valid:
        tsv, tvr = np.asarray(tsv, dtype=np.float64), np.asarray(tvr, dtype=np.float64)
        valid = (tsv.shape == (6,) and tvr.shape == (12,) and np.isfinite(tsv).all()
                 and np.isfinite(tvr).all() and np.isfinite(thickness) and thickness > 0
                 and abs(np.linalg.det(tsv[:4].reshape(2, 2))) > 1e-8
                 and abs(np.linalg.det(tvr[:9].reshape(3, 3))) > 1e-8)
    if valid:
        sections.setdefault(row['animal_id'], []).append(row)
    else:
        excluded.append({'animal_id': row['animal_id'], 'experiment_id': row['experiment_id'],
                         'section_id': row['section_id'], 'reason': 'nonfinite_or_degenerate_Allen_TSV_TVR'})
assert sections and set(sections).issubset(train_donors)
assert sum(map(len, sections.values())) + len(excluded) == frozen['metadata_eligible_sections']
OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(__file__), OUT / 'acquisition_preparation_source.py')
shutil.copyfile(RESERVATION, OUT / 'frozen_donor_reservation.jsonl')
shutil.copyfile(METADATA / 'completed.json', OUT / 'frozen_train_metadata_completed.json')
(OUT / 'excluded_geometry_sections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in excluded), encoding='utf8')
source_hashes = {str(path): sha(path) for path in (Path(__file__), RESERVATION,
    METADATA / 'completed.json', METADATA / 'experiments.jsonl', METADATA / 'sections.jsonl',
    METADATA / 'receipts.jsonl', REPO / 'source/joint_slice_input.py',
    OLD_TRAIN / 'completed.json',
    REPO / 'source/proprietary_trajectory_tool.py',
    REPO / 'training/acquire_allen_real_histology_images.py',
    REPO / 'training/arbitrary_plane_geometry.py',
    REPO / 'training/arbitrary_plane_full_frame_primitives.py')}
anchors = np.array([[0., 0.], [SIDE - 1., 0.], [0., SIDE - 1.],
                    [SIDE - 1., SIDE - 1.], [(SIDE - 1.) / 2, (SIDE - 1.) / 2]])
started = time.perf_counter()
local = threading.local()
print(json.dumps({'start': True, 'train_donors': len(sections),
                  'train_candidates': sum(map(len, sections.values())), 'output': str(OUT)}), flush=True)
reused_total = 0


def download(item):
    rank, row = item
    old = old_records.get(row['section_id'])
    if old is not None and old['source_requested_url'] == row['image_download_url']:
        assert old['animal_id'] == row['animal_id'] and old['experiment_id'] == row['experiment_id']
        source = Path(old['source_image_path'])
        assert source.is_relative_to(ROOT) and sha(source) == old['source_image_sha256']
        relative = _image_relative_path(row)
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == old['source_image_sha256']
        else:
            shutil.copyfile(source, target)
            assert sha(target) == old['source_image_sha256']
        return rank, {'animal_id': row['animal_id'], 'section_id': row['section_id'],
            'relative_path': relative, 'sha256': old['source_image_sha256'],
            'requested_url': row['image_download_url'], 'response_url': old.get('source_response_url'),
            'content_type': old.get('source_content_type'), 'etag': old.get('source_etag'),
            'last_modified': old.get('source_last_modified'),
            'source_section_record_sha256': hashlib.sha256(_canonical_bytes(row)).hexdigest(),
            'source_experiment_response_sha256': row['source_experiment_response_sha256'],
            'reused_from_completed_train': str(source),
            'reused_from_completed_train_row': old['array_row_index']}
    if not hasattr(local, 'session'):
        local.session = requests.Session()
        local.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))))
    for attempt in range(4):
        try:
            return rank, _download_record(local.session.get, row, OUT, rank)
        except (requests.RequestException, OSError) as error:
            if attempt == 3:
                raise RuntimeError(f"Allen TRAIN section {row['section_id']}: {error}") from error
            time.sleep(2 ** attempt)


for donor_index, animal in enumerate(sorted(sections)):
    donor = OUT / f'donor_{animal}'
    donor.mkdir(exist_ok=True)
    rows = sorted(sections[animal], key=lambda r: (r['experiment_id'], r['section_number'], r['section_id']))
    done = donor / 'completed.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        assert receipt['animal_id'] == animal and receipt['reservation_sha256'] == reservation_hash
        assert receipt['source_sha256'] == source_hashes and receipt['section_ids'] == [r['section_id'] for r in rows]
        for name, digest in receipt['output_sha256'].items():
            assert sha(donor / name) == digest
        for name, digest in receipt['raw_image_sha256'].items():
            assert sha(OUT / name) == digest
        reused_total += receipt['reused_train_images']
        continue
    images = np.lib.format.open_memmap(donor / 'images.npy', mode='w+', dtype=np.float16,
                                       shape=(len(rows), 1, SIDE, SIDE))
    states, affines, model_to_download, thicknesses, records = [], [], [], [], []
    reused_donor = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, downloaded in pool.map(download, enumerate(rows)):
            row = rows[i]
            experiment = experiments[row['experiment_id']]
            reused_donor += 'reused_from_completed_train' in downloaded
            assert downloaded['animal_id'] == animal and downloaded['section_id'] == row['section_id']
            image_path = OUT / downloaded['relative_path']
            image_bytes = image_path.read_bytes()
            assert hashlib.sha256(image_bytes).hexdigest() == downloaded['sha256']
            decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            display, display_scale = downsample_for_display(as_gray(decoded))
            assert display_scale == 1.
            weight_image = normalize_u8(display)
            prepared = prepare_joint_slice_input(weight_image.astype(np.float32) / 255., np.eye(3),
                                                 weight_image.shape, (SIDE, SIDE))
            images[i] = prepared['channels'][:1].astype(np.float16)
            download_from_model = np.linalg.inv(prepared['raw_to_model_xy'])
            pyramid_scale = 2 ** int(parse_qs(urlparse(downloaded['requested_url']).query)['downsample'][0])
            assert pyramid_scale == 32
            full_from_download = np.array([[pyramid_scale, 0., (pyramid_scale - 1.) / 2],
                                           [0., pyramid_scale, (pyramid_scale - 1.) / 2], [0., 0., 1.]])
            tsv = np.asarray(row['alignment2d_tsv'], dtype=np.float64)
            tvr = np.asarray(experiment['alignment3d_tvr'], dtype=np.float64)
            thickness = float(experiment['section_thickness_um'])
            assert np.isfinite(tsv).all() and np.isfinite(tvr).all() and np.isfinite(thickness) and thickness > 0
            tsv_linear = np.array([[tsv[0], tsv[1]], [tsv[2], tsv[3]]])
            rotation = tvr[:9].reshape(3, 3)
            assert abs(np.linalg.det(tsv_linear)) > 1e-8 and abs(np.linalg.det(rotation)) > 1e-8
            volume_from_full = np.array([[tsv[0], tsv[1], tsv[4]], [tsv[2], tsv[3], tsv[5]],
                                         [0., 0., row['section_number'] * thickness]])
            affine = rotation @ volume_from_full @ full_from_download @ download_from_model
            affine[:, 2] += tvr[9:] + 12.5
            assert np.isfinite(affine).all() and np.linalg.norm(np.cross(affine[:, 0], affine[:, 1])) > 1e-8
            expected = np.column_stack((anchors, np.ones(len(anchors)))) @ affine.T
            volume = np.linalg.solve(rotation, (expected - tvr[9:] - 12.5).T).T
            full = np.linalg.solve(tsv_linear, (volume[:, :2] - tsv[[4, 5]]).T).T
            download_xy = np.column_stack((full, np.ones(len(full)))) @ np.linalg.inv(full_from_download).T
            recovered = download_xy @ prepared['raw_to_model_xy'].T
            assert np.max(np.linalg.norm(recovered[:, :2] - anchors, axis=1)) < .02
            assert np.max(abs(volume[:, 2] - row['section_number'] * thickness)) < .02
            ouv = np.concatenate((affine[:, 2], SIDE * affine[:, 0], SIDE * affine[:, 1]))
            state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(ouv))).float()
            reconstructed = normalized_raster_to_ccf(*full_frame_state_to_components(state),
                                                      torch.from_numpy(anchors / SIDE).float()).numpy()
            assert np.max(np.linalg.norm(reconstructed - expected, axis=1)) < .02
            states.append(state.numpy())
            affines.append(affine)
            model_to_download.append(download_from_model)
            thicknesses.append(thickness)
            records.append({
                'animal_id': animal, 'animal_id_namespace': row['animal_id_namespace'],
                'animal_partition_key': row['animal_partition_key'], 'specimen_id': row['specimen_id'],
                'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
                'section_id_namespace': row['section_id_namespace'], 'section_number': row['section_number'],
                'training_split': 'train', 'donor_array_row': i, 'source_image_path': str(image_path),
                'source_image_sha256': downloaded['sha256'], 'source_requested_url': downloaded['requested_url'],
                'source_response_url': downloaded['response_url'], 'source_content_type': downloaded['content_type'],
                'source_etag': downloaded['etag'], 'source_last_modified': downloaded['last_modified'],
                'image_acquisition_mode': 'copied_verified_earlier_train' if 'reused_from_completed_train' in downloaded else 'new_official_download',
                'reused_from_completed_train': downloaded.get('reused_from_completed_train'),
                'reused_from_completed_train_row': downloaded.get('reused_from_completed_train_row'),
                'source_section_record_sha256': downloaded['source_section_record_sha256'],
                'source_experiment_record_sha256': hashlib.sha256(_canonical_bytes(experiment)).hexdigest(),
                'source_experiment_response_sha256': downloaded['source_experiment_response_sha256'],
                'source_metadata_index': str(METADATA / 'sections.jsonl'),
                'alignment2d_tsv': tsv.tolist(), 'alignment3d_tvr': tvr.tolist(),
                'model_pixel_to_downloaded_pixel': download_from_model.tolist(),
                'model_pixel_to_ap_dv_ml_um': affine.tolist(), 'section_thickness_um': thickness,
                'model_shape_h_w': [SIDE, SIDE], 'downloaded_shape_h_w': list(weight_image.shape),
                'gui_weight_image_sha256': hashlib.sha256(weight_image.tobytes()).hexdigest(),
                'label_role': 'weak Allen registered affine, not blinded expert alignment; no deformation truth',
                'automatic_segmentation': 'none', 'learned_source_dependency': 'none'})
    images.flush()
    del images
    np.savez(donor / 'geometry.npz', state=np.stack(states),
             model_pixel_to_ap_dv_ml_um=np.stack(affines),
             model_pixel_to_downloaded_pixel=np.stack(model_to_download),
             thickness_um=np.array(thicknesses, dtype=np.float32))
    (donor / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf8')
    receipt = {'animal_id': animal, 'reservation_sha256': reservation_hash,
        'section_ids': [r['section_id'] for r in rows], 'source_sha256': source_hashes,
        'output_sha256': {name: sha(donor / name) for name in ('images.npy', 'geometry.npz', 'records.jsonl')},
        'raw_image_sha256': {str(Path(r['source_image_path']).relative_to(OUT)).replace('\\', '/'): r['source_image_sha256']
                             for r in records},
        'image_count': len(rows), 'reused_train_images': reused_donor,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    done.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    reused_total += reused_donor
    if (donor_index + 1) % 50 == 0 or donor_index + 1 == len(sections):
        print(json.dumps({'donors_completed': donor_index + 1, 'donors_total': len(sections),
                          'seconds': time.perf_counter() - started}), flush=True)

manifest = {'training_role': 'train_only', 'training_donors': len(sections),
    'training_images': sum(map(len, sections.values())), 'reservation_sha256': reservation_hash,
    'reused_verified_train_images': reused_total,
    'new_official_downloads': sum(map(len, sections.values())) - reused_total,
    'excluded_invalid_geometry_sections': len(excluded),
    'source_sha256': source_hashes, 'donor_receipts': {str(animal): sha(OUT / f'donor_{animal}/completed.json')
                                                        for animal in sorted(sections)},
    'photometry': 'GUI as_gray -> downsample_for_display -> normalize_u8 -> prepare_joint_slice_input',
    'geometry': 'weak Allen TSV/TVR affine with 12.5um API-to-model shift; five-anchor inverse check',
    'shape_per_image': [1, SIDE, SIDE], 'dtype': 'float16',
    'development_calibration_final_or_benchmark_downloads': 0,
    'pretrained_weights_features_or_pseudolabels': 'none',
    'earlier_train_source': str(OLD_TRAIN),
    'completed_at_utc': datetime.now(timezone.utc).isoformat()}
(OUT / 'completed.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print(json.dumps({'complete': True, 'training_donors': manifest['training_donors'],
                  'training_images': manifest['training_images'],
                  'excluded_invalid_geometry_sections': len(excluded),
                  'seconds': time.perf_counter() - started}), flush=True)
