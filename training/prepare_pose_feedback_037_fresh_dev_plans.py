"""Freeze eight new synthetic DEV deformation identities; generate no sections."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import numpy as np
import torch

from training import arbitrary_plane_acquisition_v2 as acquisition
from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

OUT = ROOT / 'data/pose_feedback_037_fresh_synthetic_dev_plans_001'
PRIOR = (ROOT / 'data/joint_v6_coherent_subject_plans_002',
         ROOT / 'data/one_shot_fresh_synthetic_dev_plans_001')
SEED = 2026100371
CONTEXT_SHA256 = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


prior = [json.loads((directory / 'completed.json').read_text()) for directory in PRIOR]
prior_ids = {row['subject_deformation_plan_id'] for index in prior for row in index['subjects']}
prior_animals = {row['animal_id'] for index in prior for row in index['subjects']}
source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) |
                      {Path(__file__).name, 'subject_deformed_slab_multiresolution_bundle_v2.py'})
source_paths = [repository / 'training' / name for name in source_names]
source_paths.append(repository / 'publication/arbitrary_plane_acquisition_hardening_preflight.yaml')
roster = []
for ordinal in range(8):
    index = 3700 + ordinal
    prefix = f'pose-feedback-037-fresh-dev-synthetic-subject-{index:08d}'
    roster.append({'split': 'development', 'animal_index': index,
                   'subject_id': prefix, 'animal_id': prefix,
                   'specimen_id': f'{prefix}-specimen',
                   'experiment_id': f'{prefix}-experiment',
                   'planned_future_planes': 32})
assert not {row['animal_id'] for row in roster} & prior_animals
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'source').mkdir()
source_hashes = {}
for path in source_paths:
    relative = path.relative_to(repository)
    archived = OUT / 'source' / relative
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, archived)
    source_hashes[relative.as_posix()] = sha(archived)
protocol = {
    'root_seed': SEED, 'subject_roster': roster, 'subject_count': len(roster),
    'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                               text=True).strip(),
    'source_file_sha256': source_hashes,
    'source_hash_kind': 'exact archived file bytes; plan receipts also bind LF-normalized sampler sources',
    'expected_context_sha256': CONTEXT_SHA256,
    'prior_plan_completed_sha256': {directory.name: sha(directory / 'completed.json')
                                    for directory in PRIOR},
    'plan_parameters': 'sample_animal_subject_deformation_plan_v2 defaults; standard nonidentity stratum',
    'sampling': 'eight independent PCG64DXSM-derived DEV deformation identities; no prior plan reuse',
    'scope': 'synthetic deformation identities from one pinned Allen atlas; not biological animals, external validation or expert truth',
    'split_policy': 'development-only; no descendants enter training; checkpoint selection may use this panel',
    'future_sections': '32 independent arbitrary physical planes and appearances per identity; not made by this script',
    'section_count': 0, 'numpy': np.__version__, 'torch': str(torch.__version__),
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
started = time.perf_counter()
context, atlas_inputs = bundle.load_pinned_allen_context(ROOT / 'data/Allen Brain Atlas 25um')
assert context['v2_context_sha256'] == CONTEXT_SHA256
lower = np.zeros(3, dtype=np.float64)
upper = np.asarray(context['opaque_v1_context']['scalar_tensor'].shape, dtype=np.float64) * 25.
(OUT / 'context.json').write_text(json.dumps({
    'v2_context_sha256': context['v2_context_sha256'],
    'receipt': acquisition._json_value(context['receipt']),
    'atlas_inputs': atlas_inputs,
    'full_ccf_lower_ap_dv_ml_um': lower.tolist(),
    'full_ccf_upper_ap_dv_ml_um': upper.tolist(),
    'voxel_face_origin_ap_dv_ml_um': lower.tolist(),
    'voxel_size_ap_dv_ml_um': [25., 25., 25.]}, indent=2))
del context
records = []
for ordinal, identity in enumerate(roster):
    tick = time.perf_counter()
    print(json.dumps({'event': 'subject_started', 'ordinal': ordinal + 1,
                      'total': len(roster), **identity}), flush=True)
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=SEED, split='development',
        animal_index=identity['animal_index'], animal_id=identity['animal_id'],
        ccf_context_sha256=CONTEXT_SHA256)
    assert plan['subject_deformation_plan_id'] not in prior_ids
    directory = OUT / f"subject_{identity['animal_index']:08d}"
    directory.mkdir()
    files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt['receipt_sha256'] == plan['receipt_sha256']
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2))
    lineage = {**identity, 'synthetic_animal_id': plan['synthetic_animal_id'],
               'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
               'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
               'subject_plan_receipt_sha256': plan['receipt_sha256'],
               'root_seed': SEED, 'ccf_context_sha256': CONTEXT_SHA256}
    (directory / 'lineage.json').write_text(json.dumps(lineage, indent=2))
    record = {**lineage, 'directory': directory.relative_to(OUT).as_posix(),
              'plan_files': files,
              'accepted_amplitude_um': plan['realization']['accepted_amplitude_um'],
              'accepted_candidate_index': plan['realization']['accepted_candidate_index'],
              'candidate_audits': acquisition._json_value(plan['realization']['candidate_audits']),
              'global_scale_ap_dv_ml': plan['state']['global_scale'].tolist(),
              'generation_seconds': time.perf_counter() - tick,
              'artifact_sha256': {name: sha(directory / name) for name in
                                  (*files.values(), 'subject_plan_receipt.json', 'lineage.json')}}
    records.append(record)
    (directory / 'completed.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'event': 'subject_frozen', 'ordinal': ordinal + 1,
                      'total': len(roster), 'subject_id': identity['subject_id'],
                      'seconds': record['generation_seconds']}), flush=True)
    del plan
assert len({row['subject_deformation_plan_id'] for row in records}) == len(roster)
assert len({row['synthetic_animal_id'] for row in records}) == len(roster)
for path in source_paths:
    assert sha(path) == source_hashes[path.relative_to(repository).as_posix()]
completed = {'protocol': protocol, 'context_sha256': CONTEXT_SHA256,
             'subjects': records, 'section_count': 0,
             'total_seconds': time.perf_counter() - started,
             'artifact_sha256': {path.relative_to(OUT).as_posix(): sha(path)
                                 for path in sorted(OUT.rglob('*')) if path.is_file()}}
(OUT / 'completed.json').write_text(json.dumps(completed, indent=2))
print(json.dumps({'event': 'plans_frozen', 'subjects': len(records),
                  'total_seconds': completed['total_seconds']}), flush=True)
