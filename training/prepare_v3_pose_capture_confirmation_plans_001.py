"""Freeze eight new v3 pose-capture confirmation identities; generate no sections."""
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
from training.arbitrary_plane_streaming_synthetic_v7 import VARIANTS

OUT = ROOT / 'data/v3_pose_capture_confirmation_plans_001'
PRIOR = (ROOT / 'data/joint_v6_coherent_subject_plans_002',
         ROOT / 'data/one_shot_fresh_synthetic_dev_plans_001',
         ROOT / 'data/pose_feedback_037_fresh_synthetic_dev_plans_001',
         ROOT / 'data/pose_feedback_061_fresh_synthetic_dev_plans_001',
         ROOT / 'data/contextual_match_099_fresh_synthetic_dev_plans_001',
         ROOT / 'data/contextual_localization_102_fresh_dev_plans_001',
         ROOT / 'data/cross_candidate_match_103_fresh_dev_plans_001')
TRAIN_MAPS = ROOT / 'data/joint_v7_independent_subject_maps_001'
TRAIN_ORIGINAL_MAPS = ROOT / 'data/joint_v7_training_data_001'
SEED = 202610094101
CONTEXT_SHA256 = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
repository = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


prior = [json.loads((directory / 'completed.json').read_text()) for directory in PRIOR]
original_map_index = json.loads((TRAIN_ORIGINAL_MAPS / 'completed.json').read_text())
original_map_protocol = json.loads((TRAIN_ORIGINAL_MAPS / 'protocol.json').read_text())
train_index = json.loads((TRAIN_MAPS / 'completed.json').read_text())
train_maps = train_index['new_train_subjects']
train_bases = [row for row in prior[0]['subjects'] if row['split'] == 'train'] + train_maps
assert len(train_bases) == train_index['total_independent_train_deformations'] == 64
assert len(train_maps) == 56 and VARIANTS == 64
assert train_index['parent_plan_completed_sha256'] == sha(PRIOR[0] / 'completed.json')
assert original_map_index['section_count'] == 4096 and len(original_map_index['shards']) == 8
assert original_map_index['protocol_sha256'] == sha(TRAIN_ORIGINAL_MAPS / 'protocol.json')
assert original_map_protocol['plan_completion_sha256'] == sha(PRIOR[0] / 'completed.json')
assert [row['animal_index'] for row in train_bases] == list(range(64))
for row, shard in zip(train_bases[:8], original_map_index['shards']):
    subject_path = TRAIN_ORIGINAL_MAPS / shard['directory'] / 'subject.json'
    assert sha(subject_path) == shard['subject_sha256']
    assert json.loads(subject_path.read_text())['lineage']['subject_id'] == row['subject_id']
for row in train_maps:
    subject_path = TRAIN_MAPS / row['directory'] / 'subject.json'
    assert sha(subject_path) == row['subject_sha256']
    assert json.loads(subject_path.read_text())['lineage']['subject_id'] == row['subject_id']
virtual_subject_ids = [f"{row['subject_id']}-virtual-affine-v7-{variant:03d}"
                       for row in train_bases for variant in range(VARIANTS)]
assert len(set(virtual_subject_ids)) == 4096
identity_keys = ('animal_id', 'subject_id', 'specimen_id', 'experiment_id',
                 'synthetic_animal_id', 'subject_deformation_plan_id',
                 'subject_deformation_realization_id', 'subject_plan_receipt_sha256')
reserved = [row for index in prior for row in index['subjects']] + train_maps
reserved_ids = {key: {row[key] for row in reserved} for key in identity_keys}
source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) |
                      {Path(__file__).name, 'subject_deformed_slab_multiresolution_bundle_v2.py',
                       'arbitrary_plane_streaming_synthetic_v7.py',
                       'arbitrary_plane_streaming_synthetic_v7_64.py'})
source_paths = [repository / 'training' / name for name in source_names]
source_paths.append(repository / 'publication/arbitrary_plane_acquisition_hardening_preflight.yaml')
source_paths.append(repository / 'docs/publication/V3_ONE_PASS_POSE_CAPTURE_PILOT_001_PROTOCOL_20261009.md')
source_paths.append(repository / 'docs/publication/V3_POSE_CAPTURE_CONFIRMATION_001_PROTOCOL_20261009.md')
roster = []
for ordinal in range(8):
    index = 10400 + ordinal
    prefix = f'v3-pose-capture-confirmation-synthetic-subject-{index:08d}'
    roster.append({'split': 'development', 'animal_index': index,
                   'subject_id': prefix, 'animal_id': prefix,
                   'specimen_id': f'{prefix}-specimen',
                   'experiment_id': f'{prefix}-experiment',
                   'planned_future_planes': 32})
assert all(not {row[key] for row in roster} & reserved_ids[key]
           for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id'))
assert not {row['subject_id'] for row in roster} & set(virtual_subject_ids)
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
    'confirmation_protocol_sha256': source_hashes['docs/publication/V3_POSE_CAPTURE_CONFIRMATION_001_PROTOCOL_20261009.md'],
    'expected_context_sha256': CONTEXT_SHA256,
    'prior_plan_completed_sha256': {directory.name: sha(directory / 'completed.json')
                                    for directory in PRIOR},
    'train_maps_completed_sha256': sha(TRAIN_MAPS / 'completed.json'),
    'train_original_maps_completed_sha256': sha(TRAIN_ORIGINAL_MAPS / 'completed.json'),
    'train_lineage': {'independent_base_deformations': len(train_bases),
                      'virtual_variants_per_base': VARIANTS,
                      'virtual_subjects': len(train_bases) * VARIANTS,
                      'virtual_subject_ids_sha256': hashlib.sha256(
                          '\n'.join(virtual_subject_ids).encode()).hexdigest(),
                      'virtual_subjects_are_affine_descendants_of_train_bases': True},
    'plan_parameters': 'sample_animal_subject_deformation_plan_v2 defaults; standard nonidentity stratum',
    'sampling': 'eight independent PCG64DXSM-derived confirmation deformation identities; no TRAIN or prior DEV plan reuse',
    'scope': 'synthetic deformation identities from one pinned Allen atlas; not biological animals, external validation or expert truth',
    'split_policy': 'confirmation-only; no descendants enter training; compare preselected pilot step 2000 to step 0 without checkpoint selection',
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
    assert plan['subject_deformation_plan_id'] not in reserved_ids['subject_deformation_plan_id']
    assert plan['subject_deformation_realization_id'] not in reserved_ids['subject_deformation_realization_id']
    assert plan['synthetic_animal_id'] not in reserved_ids['synthetic_animal_id']
    assert plan['receipt_sha256'] not in reserved_ids['subject_plan_receipt_sha256']
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
assert all(len({row[key] for row in records}) == len(roster) and
           not {row[key] for row in records} & reserved_ids[key] for key in identity_keys)
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
