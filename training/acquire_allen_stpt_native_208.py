"""Freeze eight TRAIN STPT donors, then acquire native red stacks on a second run."""
import os
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TMPDIR'] = str(ROOT / 'tmp')

import hashlib
import io
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RESERVATION = ROOT / 'data/joint_v7_reserved_train_images_192_001/frozen_donor_reservation.jsonl'
METADATA = ROOT / 'data/joint_v7_reserved_train_metadata_001'
PILOT = ROOT / 'data/allen_stpt_registered_volume_207_train_001'
OUT = ROOT / 'data/allen_stpt_native_208_train_001'
SELECTION = OUT / 'frozen_selection.json'
API = 'https://api.brain-map.org'
N_DONORS = 8
local = threading.local()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == content, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_bytes(content)
    os.replace(temporary, path)


def session():
    if not hasattr(local, 'session'):
        local.session = requests.Session()
        local.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=('GET', 'HEAD'))))
    return local.session


def get(url, **kwargs):
    assert urlparse(url).hostname == 'api.brain-map.org' and url.startswith(API + '/')
    response = session().get(url, timeout=120, **kwargs)
    response.raise_for_status()
    assert urlparse(response.url).hostname == 'api.brain-map.org'
    return response


def freeze_selection():
    reservations = [json.loads(line) for line in RESERVATION.open()]
    train = {r['experiment_id']: r for r in reservations if r['proposed_split'] == 'train'}
    assert len({r['animal_id'] for r in train.values()}) == len(train)
    assert not {r['animal_id'] for r in train.values()} & {
        r['animal_id'] for r in reservations if r['proposed_split'] != 'train'}
    experiments = {e['experiment_id']: e for e in map(json.loads, (METADATA / 'experiments.jsonl').open())}
    candidates = [r for r in train.values() if r['experiment_id'] in experiments
                  and experiments[r['experiment_id']]['split'] == 'train'
                  and experiments[r['experiment_id']]['red_channel'] == 'background fluorescence'
                  and experiments[r['experiment_id']]['reference_space_id'] == 9
                  and experiments[r['experiment_id']]['plane_of_section_id'] == 1
                  and experiments[r['experiment_id']]['section_thickness_um'] == 100
                  and experiments[r['experiment_id']]['alignment3d_tvr'] is not None]
    candidates.sort(key=lambda r: (r['animal_id'], r['experiment_id']))
    counts = {r['experiment_id']: [0, []] for r in candidates}
    for row in map(json.loads, (METADATA / 'sections.jsonl').open()):
        stat = counts.get(row['experiment_id'])
        if stat is None:
            continue
        stat[0] += 1
        if (row['split'] == 'train' and row['eligible_for_appearance_training']
                and row['alignment2d_tsv'] is not None and row['image_download_url']
                and parse_qs(urlparse(row['image_download_url']).query).get('downsample') == ['5']):
            stat[1].append(row['section_number'])
    selected = []
    for r in candidates:
        experiment_id = r['experiment_id']
        total, numbers = counts[experiment_id]
        if not numbers or len(numbers) != total or len(set(numbers)) != total:
            continue
        if max(numbers) - min(numbers) + 1 != total:
            continue
        url = API + '/api/v2/data/WellKnownFile/query.json'
        response = get(url, params={
            'criteria': f"[attachable_id$eq{experiment_id}],well_known_file_type[name$in'DeformationFieldHeader','DeformationFieldVoxels']",
            'include': 'well_known_file_type', 'num_rows': 'all'})
        payload = response.json()
        assert payload['success']
        field = {x['well_known_file_type']['name']: x for x in payload['msg']}
        if not {'DeformationFieldHeader', 'DeformationFieldVoxels'} <= field.keys():
            continue
        selected.append({'reservation': r, 'experiment': experiments[experiment_id],
                         'field_api_url': response.url,
                         'field_api_sha256': hashlib.sha256(response.content).hexdigest(),
                         'field_files': {name: {'id': field[name]['id'],
                             'download_url': API + '/api/v2/well_known_file_download/' + str(field[name]['id']),
                             'path': field[name].get('path'), 'file_size': field[name].get('file_size')}
                             for name in ('DeformationFieldHeader', 'DeformationFieldVoxels')}})
        write_json(OUT / 'field_api' / f'{experiment_id}.json', payload)
        if len(selected) == N_DONORS:
            break
    assert len(selected) == N_DONORS
    selected_ids = {s['reservation']['experiment_id'] for s in selected}
    sections = {experiment_id: [] for experiment_id in selected_ids}
    for row in map(json.loads, (METADATA / 'sections.jsonl').open()):
        if row['experiment_id'] in sections:
            sections[row['experiment_id']].append(row)
    for item in selected:
        experiment_id = item['reservation']['experiment_id']
        item['sections'] = sorted(sections[experiment_id], key=lambda x: x['section_number'])
        assert len(item['sections']) == counts[experiment_id][0]
        assert all(row['animal_id'] == item['reservation']['animal_id']
                   and row['specimen_id'] == item['reservation']['specimen_id']
                   and row['eligible_for_appearance_training'] for row in item['sections'])
    write_json(SELECTION, {
        'selection_rule': 'smallest Allen Donor.id among frozen TRAIN Product5 coronal 100um red-background experiments with complete contiguous ds5 SectionImage metadata, Alignment2d/3d and both deformation WellKnownFile types; one experiment per donor',
        'reservation_sha256': sha(RESERVATION),
        'metadata_sha256': {name: sha(METADATA / name) for name in ('experiments.jsonl', 'sections.jsonl', 'completed.json')},
        'selected_before_image_download_or_performance_inspection': True,
        'selected_at_utc': datetime.now(timezone.utc).isoformat(),
        'donors': selected})
    print(json.dumps({'frozen': [(x['reservation']['animal_id'], x['reservation']['experiment_id'])
                                  for x in selected], 'selection': str(SELECTION)}), flush=True)


def download_section(row, donor):
    number, section_id = row['section_number'], row['section_id']
    path = donor / 'original_ds5' / f'section_{number:03d}_id_{section_id}.jpg'
    reused = None
    if not path.exists() and row['animal_id'] == 14383 and number in (40, 60, 80, 100):
        pilot_image = PILOT / f'native_section_{number}_ds5.jpg'
        pilot_metadata = json.loads((PILOT / f'native_section_{number}_metadata.json').read_text())['msg'][0]
        assert pilot_metadata['id'] == section_id
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pilot_image, path)
        reused = str(pilot_image)
    if not path.exists():
        response = get(row['image_download_url'])
        content = response.content
        with Image.open(io.BytesIO(content)) as image:
            assert image.format == 'JPEG' and image.mode == 'RGB'
            dimensions = image.size
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.jpg.part')
        temporary.write_bytes(content)
        os.replace(temporary, path)
        response_url = response.url
        content_type = response.headers.get('Content-Type')
    else:
        response_url = None
        content_type = None
    with Image.open(path) as image:
        assert image.format == 'JPEG' and image.mode == 'RGB'
        dimensions = image.size
    assert dimensions == (row['width_full_resolution_px'] // 32,
                          row['height_full_resolution_px'] // 32)
    return {'section_number': number, 'section_id': section_id,
            'requested_url': row['image_download_url'], 'response_url': response_url,
            'content_type': content_type, 'relative_path': path.relative_to(OUT).as_posix(),
            'sha256': sha(path), 'bytes': path.stat().st_size,
            'dimensions_xy_px': list(dimensions), 'reused_verified_pilot': reused,
            'alignment2d_tsv': row['alignment2d_tsv'],
            'source_section_api_url': row['source_section_api_url'],
            'source_experiment_response_sha256': row['source_experiment_response_sha256']}


def acquire():
    frozen = json.loads(SELECTION.read_text())
    assert frozen['reservation_sha256'] == sha(RESERVATION)
    assert frozen['metadata_sha256'] == {name: sha(METADATA / name)
                                         for name in ('experiments.jsonl', 'sections.jsonl', 'completed.json')}
    for item in frozen['donors']:
        r, e, rows = item['reservation'], item['experiment'], item['sections']
        donor = OUT / f"donor_{r['animal_id']}"
        donor.mkdir(exist_ok=True)
        receipt_path = donor / 'receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert all(sha(OUT / p) == digest for p, digest in receipt['file_sha256'].items())
            print(json.dumps({'donor': r['animal_id'], 'complete': True, 'reused': True}), flush=True)
            continue
        header = get(item['field_files']['DeformationFieldHeader']['download_url'])
        assert len(header.content) < 100000 and b'ElementDataFile' in header.content
        header_path = donor / 'dfmfld.mhd'
        if header_path.exists():
            assert header_path.read_bytes() == header.content
        else:
            header_path.write_bytes(header.content)
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(lambda row: download_section(row, donor), rows))
        widths = {x['dimensions_xy_px'][0] for x in records}
        heights = {x['dimensions_xy_px'][1] for x in records}
        assert len(widths) == len(heights) == 1
        width, height = widths.pop(), heights.pop()
        raw_xy_spacing = 32 * rows[0]['resolution_um_per_px']
        target_width = round(width * raw_xy_spacing / 50)
        target_height = round(height * raw_xy_spacing / 50)
        spacing_x = width * raw_xy_spacing / target_width
        spacing_y = height * raw_xy_spacing / target_height
        stack_path = donor / 'native_red_50um_xy_100um_z.npy'
        temporary = donor / 'native_red_50um_xy_100um_z.part.npy'
        stack = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                          shape=(len(records), target_height, target_width))
        for index, record in enumerate(records):
            with Image.open(OUT / record['relative_path']) as image:
                stack[index] = np.asarray(image.getchannel('R').resize(
                    (target_width, target_height), Image.Resampling.BOX))
        stack.flush()
        del stack
        os.replace(temporary, stack_path)
        write_json(donor / 'sections.json', records)
        write_json(donor / 'native_stack_geometry.json', {
            'axis_order': 'zyx', 'dtype': 'uint8', 'shape': [len(records), target_height, target_width],
            'paths': {'stack': stack_path.relative_to(OUT).as_posix(),
                      'sections': (donor / 'sections.json').relative_to(OUT).as_posix(),
                      'original_ds5_directory': (donor / 'original_ds5').relative_to(OUT).as_posix()},
            'physical_voxel_center_convention': 'x=(i+0.5)*spacing_x; y=(j+0.5)*spacing_y; z=section_number*100 micrometers; no isotropic-z claim',
            'native_origin_first_voxel_center_um_xyz': [spacing_x / 2, spacing_y / 2, rows[0]['section_number'] * 100],
            'native_spacing_um_xyz': [spacing_x, spacing_y, 100.0],
            'original_section_number_start_end': [rows[0]['section_number'], rows[-1]['section_number']],
            'section_numbers': [x['section_number'] for x in records],
            'original_ds5_spacing_um_xy': [raw_xy_spacing, raw_xy_spacing],
            'resampling': 'PIL BOX of JPEG red channel only; no interpolated z sections',
            'animal_id': r['animal_id'], 'specimen_id': r['specimen_id'], 'experiment_id': r['experiment_id'],
            'section_thickness_um': e['section_thickness_um'],
            'alignment3d_tvr': e['alignment3d_tvr'],
            'alignment3d_source_api_url': e['source_api_url'],
            'field_files': item['field_files'],
            'field_header_sha256': sha(header_path),
            'registration_scope': 'Allen upstream weak CCF alignment; virtual obliques are not physically acquired'})
        files = [header_path, stack_path, donor / 'sections.json', donor / 'native_stack_geometry.json']
        files += [OUT / record['relative_path'] for record in records]
        write_json(receipt_path, {
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'donor_id': r['animal_id'], 'specimen_id': r['specimen_id'], 'experiment_id': r['experiment_id'],
            'selection_sha256': sha(SELECTION), 'section_count': len(records),
            'rights': {'source': 'Allen Mouse Brain Connectivity Atlas',
                       'terms_url': 'https://alleninstitute.org/legal/terms-of-use',
                       'citation_url': 'https://alleninstitute.org/legal/citation-policy',
                       'license_spdx': None, 'redistribution_cleared': False},
            'field_header_requested_url': item['field_files']['DeformationFieldHeader']['download_url'],
            'field_header_response_url': header.url,
            'file_sha256': {path.relative_to(OUT).as_posix(): sha(path) for path in files}})
        print(json.dumps({'donor': r['animal_id'], 'experiment': r['experiment_id'],
                          'sections': len(records), 'shape_zyx': [len(records), target_height, target_width],
                          'complete': True}), flush=True)
    completed = OUT / 'completed.json'
    if not completed.exists():
        receipts = [OUT / f"donor_{item['reservation']['animal_id']}" / 'receipt.json'
                    for item in frozen['donors']]
        donor_rows = [json.loads(path.read_text()) for path in receipts]
        write_json(completed, {
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'frozen_selection_path': str(SELECTION), 'frozen_selection_sha256': sha(SELECTION),
            'acquisition_source_path': str(Path(__file__).resolve()),
            'acquisition_source_sha256': sha(Path(__file__).resolve()),
            'donor_ids': [row['donor_id'] for row in donor_rows],
            'experiment_ids': [row['experiment_id'] for row in donor_rows],
            'donors': len(donor_rows), 'original_ds5_jpegs': sum(row['section_count'] for row in donor_rows),
            'original_jpeg_bytes': sum(sum((OUT / path).stat().st_size
                                       for path in row['file_sha256'] if path.endswith('.jpg'))
                                       for row in donor_rows),
            'stack_dtype': 'uint8', 'stack_axis_order': 'zyx',
            'stack_xy_resolution': 'approximately 50 micrometers; exact per-donor spacing in native_stack_geometry.json',
            'native_z_spacing_um': 100.0,
            'field_headers_acquired': len(donor_rows), 'field_voxel_files_acquired': 0,
            'development_calibration_final_images_accessed': 0,
            'scope': 'physically acquired coronal STPT sections; Allen CCF alignment is weak; virtual obliques are not physically imaged',
            'donor_receipt_sha256': {path.relative_to(OUT).as_posix(): sha(path) for path in receipts}})
    print(json.dumps({'all_eight_donors_complete': True, 'output': str(OUT)}), flush=True)


OUT.mkdir(parents=True, exist_ok=True)
if SELECTION.exists():
    acquire()
else:
    freeze_selection()
