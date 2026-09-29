"""1024 additional fixed TRAIN sections, unchanged raw-red conversion, linked1280 union."""
READY_AFTER_SOURCE_REVIEW = True
assert READY_AFTER_SOURCE_REVIEW, "Root reviews, commits and launches this fixed acquisition"

import os
import sys
from pathlib import Path
ROOT = Path(r'I:\AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT/'tmp')
sys.dont_write_bytecode = True

import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from training.acquire_allen_real_histology_images import deterministic_section_selection, _download_record
from training.allen_real_histology_metadata import _canonical_bytes

COHORT = ROOT/'data/allen_real_development_20260928'
OLD = ROOT/'data/allen_real_training_inputs_20260929'
OUTPUT = ROOT/'data/allen_real_training_expansion_20260929'
REPOSITORY = Path(__file__).resolve().parents[1]
SIDE, FIELD_UM = 96, 12000.
API_TO_MODEL_SHIFT_UM = np.full(3, 12.5)
PLANNED_IDS_SHA256 = '4d1d731c9d668112bad0e140f3c5bfe5fbd5476f8c1862f2040dcd8c62e753ea'
pins = {
    COHORT/'metadata/sections.jsonl': 'f209370dc548e36b4a887524041b5d5c92ec6c2b91f770b5b64d8a543e4b7a9c',
    COHORT/'metadata/experiments.jsonl': '67b19f91610169371e4a824b7e1095e3ab2e1b9a5720a7c04a0cfb99afbf401b',
    COHORT/'metadata/receipt.json': '34d5fc1e8b75398b0d8393059327402dbe1a02d3e15235293c3c5054a49244f7',
    COHORT/'images/receipt.json': '653be9c2357c1461a67e2d997ebd210c622931aceefa3dcd257d44146d4171fa',
    OLD/'summary.json': '4f8484bbd2ab5719971d9862f3f5ea87ea3d1c9de76468a936af67bcc65da9fd',
    OLD/'image_geometry.jsonl': '9a718b2fb2791584e1d225123facbe1c02c96cfa701a9e6f3fc68f17a565d218',
    OLD/'raw_model_input.npy': 'b4c0bc908ca4b61a4000bbaa02b140e440c3b4da42069b6b793e435c64f38379',
    REPOSITORY/'training/acquire_allen_real_histology_images.py': '1a3d935c3e829391f4e6d7b6f56e0d099b0065a8769fd98fb3e9bc9201e139e4',
    REPOSITORY/'training/prepare_joint_v6_allen_training_inputs.py': 'f17022e971c9054b80e4bd1d61b2994495e425948b8003147d378ad86070db33',
}
for path, digest in pins.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
old_records = [json.loads(line) for line in (OLD/'image_geometry.jsonl').read_text().splitlines()]
old_summary = json.loads((OLD/'summary.json').read_text())
train_sections = [r for r in map(json.loads, (COHORT/'metadata/sections.jsonl').read_text().splitlines()) if r['split'] == 'development_train']
experiments = {r['experiment_id']: r for r in map(json.loads, (COHORT/'metadata/experiments.jsonl').read_text().splitlines()) if r['split'] == 'development_train'}
old_ids = {r['section_id'] for r in old_records}
selected = [r for r in deterministic_section_selection(train_sections, {'development_train':1280}) if r['section_id'] not in old_ids]
assert len(old_records) == len(old_ids) == 256 and len(selected) == 1024
assert hashlib.sha256(json.dumps([r['section_id'] for r in selected], separators=(',', ':')).encode()).hexdigest() == PLANNED_IDS_SHA256
assert {r['animal_id'] for r in selected} == {r['animal_id'] for r in old_records} and len({r['animal_id'] for r in selected}) == 58
assert {r['animal_id'] for r in selected}.isdisjoint({14452,15219,15336,15439,15447,15935})
assert all(r['split'] == 'development_train' for r in selected+old_records)
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT/'acquisition_source.py').write_bytes(Path(__file__).read_bytes())
(OUTPUT/'download_source.py').write_bytes((REPOSITORY/'training/acquire_allen_real_histology_images.py').read_bytes())
(OUTPUT/'conversion_reference_source.py').write_bytes((REPOSITORY/'training/prepare_joint_v6_allen_training_inputs.py').read_bytes())
(OUTPUT/'planned_sections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected), encoding='utf-8')
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=(500,502,503,504), allowed_methods=('GET',))))
records = []
with (OUTPUT/'images.jsonl').open('w', encoding='utf-8') as manifest, (OUTPUT/'failures.jsonl').open('w', encoding='utf-8') as failures:
    for i, section in enumerate(selected):
        try:
            record = _download_record(session.get, section, OUTPUT, i)
        except Exception as error:
            failures.write(json.dumps({'section_id':section['section_id'], 'animal_id':section['animal_id'], 'experiment_id':section['experiment_id'], 'requested_url':section['image_download_url'], 'error_type':type(error).__name__, 'error':str(error), 'time_utc':datetime.now(timezone.utc).isoformat()})+'\n')
            failures.flush()
            raise
        record['source_experiment_record_sha256'] = hashlib.sha256(_canonical_bytes(experiments[record['experiment_id']])).hexdigest()
        record['retrieved_at_utc'] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        manifest.write(json.dumps(record)+'\n'); manifest.flush()
        if (i+1) % 64 == 0:
            print(json.dumps({'downloaded_additional_train':i+1, 'planned':1024, 'image_bytes':sum(r['bytes'] for r in records)}), flush=True)

cv2.setNumThreads(4)
images = np.empty((1024,1,SIDE,SIDE), dtype=np.float32)
pixel_x, pixel_y = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
pixel_grid = np.stack((pixel_x.ravel(), pixel_y.ravel(), np.ones(SIDE*SIDE)))
provenance = []
for i, (record, section) in enumerate(zip(records, selected)):
    experiment = experiments[record['experiment_id']]
    image_bytes = (OUTPUT/record['relative_path']).read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == record['sha256']
    red = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'), dtype=np.float32)[...,0]/255.
    native_height, native_width = red.shape
    pyramid_scale = 2**int(parse_qs(urlparse(record['requested_url']).query)['downsample'][0])
    assert pyramid_scale == 32
    resolution = section['resolution_um_per_px']
    step = FIELD_UM/(SIDE*resolution)
    center_x, center_y = (section['width_full_resolution_px']-1)/2, (section['height_full_resolution_px']-1)/2
    full_from_model = np.array([[step,0,center_x-FIELD_UM/(2*resolution)], [0,step,center_y-FIELD_UM/(2*resolution)], [0,0,1]])
    native_from_full = np.array([[1/pyramid_scale,0,.5/pyramid_scale-.5], [0,1/pyramid_scale,.5/pyramid_scale-.5], [0,0,1]])
    native_from_model = native_from_full@full_from_model
    native_grid = (native_from_model@pixel_grid)[:2].reshape(2,SIDE,SIDE).astype(np.float32)
    sigma = .5*np.sqrt(max((step/pyramid_scale)**2-1,0))
    padding = float(np.median(np.concatenate((red[0],red[-1],red[:,0],red[:,-1]))))
    filtered = cv2.GaussianBlur(red,(0,0),sigma,borderType=cv2.BORDER_REPLICATE) if sigma > 0 else red
    images[i,0] = cv2.remap(filtered,native_grid[0],native_grid[1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=padding)
    tsv = np.asarray(section['alignment2d_tsv'],dtype=np.float64)
    tvr = np.asarray(experiment['alignment3d_tvr'],dtype=np.float64)
    volume_from_full = np.array([[tsv[0],tsv[1],tsv[4]], [tsv[2],tsv[3],tsv[5]], [0,0,section['section_number']*experiment['section_thickness_um']]])
    pir_from_full = tvr[:9].reshape(3,3)@volume_from_full
    pir_from_full[:,2] += tvr[9:]
    physical_from_model = pir_from_full@full_from_model
    physical_from_model[:,2] += API_TO_MODEL_SHIFT_UM
    normal = np.cross(physical_from_model[:,0],physical_from_model[:,1]); normal /= np.linalg.norm(normal)
    native_support = (native_grid[0]>=0)&(native_grid[0]<=native_width-1)&(native_grid[1]>=0)&(native_grid[1]<=native_height-1)
    provenance.append({**record, 'array_row_index':i, 'actual_image_sha256':record['sha256'], 'full_resolution_shape_h_w':[section['height_full_resolution_px'],section['width_full_resolution_px']], 'resolution_um_per_full_pixel':resolution, 'pyramid_scale':pyramid_scale, 'model_pixel_to_full_pixel':full_from_model.tolist(), 'model_pixel_to_downloaded_pixel':native_from_model.tolist(), 'model_pixel_to_ap_dv_ml_um':physical_from_model.tolist(), 'alignment2d_tsv':tsv.tolist(), 'alignment3d_tvr':tvr.tolist(), 'section_thickness_um':experiment['section_thickness_um'], 'antialias_sigma_downloaded_px':float(sigma), 'padding_red_value':padding, 'native_support_fraction':float(native_support.mean()), 'upstream_affine_center_ap_dv_ml_um':(physical_from_model@np.array([48.,48.,1.])).tolist(), 'upstream_affine_normal_ap_dv_ml':normal.tolist(), 'outline_available':False})
    if (i+1) % 128 == 0: print(json.dumps({'additional_train_inputs_prepared':i+1}),flush=True)
np.save(OUTPUT/'raw_model_input.npy',images)
(OUTPUT/'image_geometry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance),encoding='utf-8')
union = []
for directory, image_root, rows in ((OLD,COHORT/'images',old_records),(OUTPUT,OUTPUT,provenance)):
    for record in rows:
        union.append({**record, 'union_row_index':len(union), 'source_array_path':str(directory/'raw_model_input.npy'), 'source_array_row_index':record['array_row_index'], 'source_geometry_path':str(directory/'image_geometry.jsonl'), 'source_image_path':str(image_root/record['relative_path'])})
assert len(union) == len({r['section_id'] for r in union}) == 1280
(OUTPUT/'union_training_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in union),encoding='utf-8')
flat = images[:,0].reshape(1024,-1).astype(np.float64)
std, dynamic, quantiles = flat.std(-1), np.ptp(flat,axis=-1), np.quantile(flat,[.01,.05,.5,.95,.99],axis=-1).T
animals = np.array([r['animal_id'] for r in records])
np.savez(OUTPUT/'input_statistics.npz', animal_id=animals, specimen_id=[r['specimen_id'] for r in records], experiment_id=[r['experiment_id'] for r in records], section_id=[r['section_id'] for r in records], image_std=std, image_dynamic_range=dynamic, image_q01_q05_q50_q95_q99=quantiles)
by_donor = {str(d):{'additional_sections':int((animals==d).sum()), 'image_std_mean':float(std[animals==d].mean()), 'mean_image_q01_q05_q50_q95_q99':quantiles[animals==d].mean(0).tolist()} for d in np.unique(animals)}
output_hashes = {name:hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in ('acquisition_source.py','download_source.py','conversion_reference_source.py','planned_sections.jsonl','images.jsonl','failures.jsonl','raw_model_input.npy','image_geometry.jsonl','union_training_index.jsonl','input_statistics.npz')}
summary = {'completed_at_utc':datetime.now(timezone.utc).isoformat(), 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip(), 'input_sha256':{str(p):h for p,h in pins.items()}, 'planned_ordered_section_ids_sha256':PLANNED_IDS_SHA256, 'selection':'unchanged donor-round-robin first1280 TRAIN metadata rows minus original256; fixed1024 additional, zero retries/redraw of scientific selection', 'additional_section_count':1024, 'union_section_count':1280, 'donor_count':58, 'additional_per_donor_distribution':dict(Counter(Counter(animals.tolist()).values())), 'downloaded_image_bytes':sum(r['bytes'] for r in records), 'failed_download_count':0, 'metadata_refetched':False, 'development_or_benchmark_images_accessed':False, 'preprocessing':old_summary['preprocessing'], 'model_pixel_coordinate_contract':old_summary['model_pixel_coordinate_contract'], 'api_to_model_shift_ap_dv_ml_um':API_TO_MODEL_SHIFT_UM.tolist(), 'outline_available':False, 'array_contract':'float32[1024,1,96,96]; union index links existing256 and new1024 arrays without byte copying', 'terms':old_summary['terms'], 'by_donor':by_donor, 'scope':'TRAIN-only appearance and weak upstream-affine pairing; same58 biological donors, not1024 new animals; no independent arbitrary-plane, dense deformation, calibrated uncertainty or final-benchmark truth', 'historical_exposure':old_summary['historical_exposure'], 'output_sha256':output_hashes, 'image_byte_sha256':{r['relative_path']:r['sha256'] for r in records}}
(OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'completed':True,'additional_sections':1024,'union_sections':1280,'donors':58,'downloaded_image_bytes':summary['downloaded_image_bytes'],'summary_sha256':hashlib.sha256((OUTPUT/'summary.json').read_bytes()).hexdigest()}),flush=True)
