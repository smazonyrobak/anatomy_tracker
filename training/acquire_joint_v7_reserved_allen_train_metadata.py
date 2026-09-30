"""Freeze official Allen geometry metadata for reserved TRAIN donors only."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import hashlib
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from training.allen_real_histology_metadata import _experiment_record, _section_records, _official_url

PLAN = ROOT / 'data/allen_expansion_metadata_plan_20260930'
OUT = ROOT / 'data/joint_v7_reserved_train_metadata_001'
plan_path = PLAN / 'proposed_donor_reservations.jsonl'
plan = [json.loads(line) for line in plan_path.read_text().splitlines()]
train = sorted((r for r in plan if r['proposed_split'] == 'train'), key=lambda r: r['experiment_id'])
summary = json.loads((PLAN / 'candidate_count_summary.json').read_text())
assert len(train) == summary['donors_by_split']['train'] == 1885
assert len({r['animal_id'] for r in train}) == len(train)
assert not {r['animal_id'] for r in train} & set(summary['benchmark_donors_excluded'])
assert all(r['product_id'] == 5 for r in train)
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'raw').mkdir()
shutil.copyfile(__file__, OUT / 'acquisition_source.py')
shutil.copyfile(plan_path, OUT / plan_path.name)
shutil.copyfile(PLAN / 'candidate_count_summary.json', OUT / 'candidate_count_summary.json')
local = threading.local()


def fetch(reservation):
    if not hasattr(local, 'session'):
        local.session = requests.Session()
        local.session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))))
    experiment_id = reservation['experiment_id']
    url = f'https://api.brain-map.org/api/v2/data/SectionDataSet/{experiment_id}.json'
    response = local.session.get(url, params={'include':
        'specimen(donor),products,alignment3d,equalization,section_images(alignment2d)'}, timeout=90)
    response.raise_for_status()
    _official_url(response.url)
    content = bytes(response.content)
    payload = response.json()
    assert payload['success'] and len(payload['msg']) == 1
    dataset = payload['msg'][0]
    assert dataset['id'] == experiment_id
    assert dataset['specimen_id'] == reservation['specimen_id']
    assert dataset['specimen']['donor']['id'] == reservation['animal_id']
    assert not dataset['failed'] and dataset['reference_space_id'] == 9 and dataset['plane_of_section_id'] == 1
    assert 5 in {r['id'] for r in dataset['products']}
    receipt = {'requested_url': url, 'response_url': response.url,
        'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'content_type': response.headers.get('Content-Type'), 'etag': response.headers.get('ETag')}
    return reservation, content, dataset, receipt


started = time.perf_counter()
total_sections = eligible_sections = 0
with (OUT / 'experiments.jsonl').open('w', encoding='utf8') as experiments, \
     (OUT / 'sections.jsonl').open('w', encoding='utf8') as sections, \
     (OUT / 'receipts.jsonl').open('w', encoding='utf8') as receipts, \
     ThreadPoolExecutor(max_workers=6) as pool:
    for index, (reservation, content, dataset, receipt) in enumerate(pool.map(fetch, train), 1):
        raw = OUT / 'raw' / f"experiment_{reservation['experiment_id']}.json"
        raw.write_bytes(content)
        receipt['relative_path'] = raw.relative_to(OUT).as_posix()
        experiment = _experiment_record(dataset, receipt)
        experiment['split'] = 'train'
        experiment['reserved_split_source'] = str(plan_path)
        experiment['reserved_split'] = reservation['proposed_split']
        experiment['training_role'] = 'weak_real_train_geometry_and_appearance'
        if experiment['alignment3d_tvr'] is not None:
            assert all(math.isfinite(v) for v in experiment['alignment3d_tvr'])
        rows = _section_records(dataset, experiment, receipt)
        total_sections += len(rows)
        eligible_sections += sum(r['eligible_for_appearance_training'] for r in rows)
        experiments.write(json.dumps(experiment) + '\n')
        sections.writelines(json.dumps(r) + '\n' for r in rows)
        receipts.write(json.dumps(receipt) + '\n')
        if index % 100 == 0 or index == len(train):
            experiments.flush(); sections.flush(); receipts.flush()
            print(json.dumps({'train_experiments': index, 'planned': len(train),
                'sections': total_sections, 'eligible_sections': eligible_sections,
                'seconds': time.perf_counter() - started}), flush=True)

files = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in OUT.iterdir() if p.is_file() and p.name != 'completed.json'}
raw_receipts = [json.loads(line) for line in (OUT / 'receipts.jsonl').read_text().splitlines()]
assert len(raw_receipts) == len(train)
assert all(hashlib.sha256((OUT / r['relative_path']).read_bytes()).hexdigest() == r['sha256'] for r in raw_receipts)
(OUT / 'completed.json').write_text(json.dumps({'source': 'official Allen RMA API Product5',
    'training_donors': len(train), 'training_experiments': len(train),
    'sections': total_sections, 'metadata_eligible_sections': eligible_sections,
    'planned_candidate_sections': summary['section_counts_by_split']['train'],
    'development_calibration_final_or_benchmark_images_or_alignments_accessed': 0,
    'alignment_scope': 'weak upstream Allen metadata; not blinded expert ground truth',
    'files_sha256': files, 'raw_response_sha256': {r['relative_path']: r['sha256'] for r in raw_receipts},
    'seconds': time.perf_counter() - started, 'completed_at_utc': datetime.now(timezone.utc).isoformat()},
    indent=2), encoding='utf8')
print('Reserved TRAIN-only metadata frozen', flush=True)
