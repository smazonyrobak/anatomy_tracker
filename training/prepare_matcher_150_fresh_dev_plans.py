"""Freeze eight independent synthetic DEV deformation plans for matcher 150."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training import arbitrary_plane_acquisition_v2 as acquisition
from training import arbitrary_plane_subject_deformation_v2 as deformation
from training import subject_deformed_slab_multiresolution_bundle_v2 as bundle

repo = Path(__file__).resolve().parents[1]
out = root / 'data/matcher_150_fresh_dev_plans'
protocol_file = repo / 'docs/publication/MATCHER_150_FRESH_SYNTHETIC_DEV_PROTOCOL_20261010.md'
seed = 2026101015001
context_sha256 = 'c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8'
prior_dirs = tuple(root / 'data' / name for name in (
    'joint_v6_coherent_subject_plans_002',
    'one_shot_fresh_synthetic_dev_plans_001',
    'pose_feedback_037_fresh_synthetic_dev_plans_001',
    'pose_feedback_061_fresh_synthetic_dev_plans_001',
    'contextual_match_099_fresh_synthetic_dev_plans_001',
    'contextual_localization_102_fresh_dev_plans_001',
    'cross_candidate_match_103_fresh_dev_plans_001',
    'v3_pose_capture_confirmation_plans_001'))
train_maps_dir = root / 'data/joint_v7_independent_subject_maps_001'
train_original_maps_dir = root / 'data/joint_v7_training_data_001'
torch.set_num_threads(4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert repo.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
prior = [json.loads((directory / 'completed.json').read_text()) for directory in prior_dirs]
train_map_index = json.loads((train_maps_dir / 'completed.json').read_text())
train_bases = ([row for row in prior[0]['subjects'] if row['split'] == 'train'] +
               train_map_index['new_train_subjects'])
assert len(train_bases) == 64
assert [row['animal_index'] for row in train_bases] == list(range(64))
virtual_ids = {f"{row['subject_id']}-virtual-affine-v7-{variant:03d}"
               for row in train_bases for variant in range(64)}
assert len(virtual_ids) == 4096
identity_keys = ('animal_id', 'subject_id', 'specimen_id', 'experiment_id',
                 'synthetic_animal_id', 'subject_deformation_plan_id',
                 'subject_deformation_realization_id', 'subject_plan_receipt_sha256')
reserved = [row for cohort in prior for row in cohort['subjects']] + train_map_index['new_train_subjects']
reserved_ids = {key: {row[key] for row in reserved} for key in identity_keys}
roster = []
for index in range(15000, 15008):
    prefix = f'matcher-150-synthetic-dev-subject-{index:08d}'
    roster.append({'split': 'development', 'animal_index': index,
                   'subject_id': prefix, 'animal_id': prefix,
                   'specimen_id': f'{prefix}-specimen',
                   'experiment_id': f'{prefix}-experiment',
                   'planned_planes': 32})
assert all(not {row[key] for row in roster} & reserved_ids[key]
           for key in ('animal_id', 'subject_id', 'specimen_id', 'experiment_id'))
assert not {row['subject_id'] for row in roster} & virtual_ids

source_names = sorted(set(acquisition._source_hashes()) | set(deformation._source_hashes()) |
                      {Path(__file__).name, 'subject_deformed_slab_multiresolution_bundle_v2.py'})
source_paths = [repo / 'training' / name for name in source_names] + [protocol_file]
out.mkdir(parents=True, exist_ok=False)
source_hashes = {}
for path in source_paths:
    relative = path.relative_to(repo)
    archived = out / 'source' / relative
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, archived)
    source_hashes[relative.as_posix()] = sha(archived)
protocol = {'root_seed': seed, 'subject_roster': roster,
            'plan_sampling': 'eight independent nonidentity synthetic deformation plans; no TRAIN or prior DEV plan reused',
            'atlas_context_sha256': context_sha256,
            'prior_plan_completed_sha256': {directory.name: sha(directory / 'completed.json')
                                            for directory in prior_dirs},
            'train_maps_completed_sha256': sha(train_maps_dir / 'completed.json'),
            'train_original_maps_completed_sha256': sha(train_original_maps_dir / 'completed.json'),
            'train_base_deformation_count': len(train_bases),
            'train_virtual_subject_count': len(virtual_ids),
            'train_virtual_subject_ids_sha256': hashlib.sha256('\n'.join(sorted(virtual_ids)).encode()).hexdigest(),
            'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                                       text=True).strip(),
            'source_file_sha256': source_hashes,
            'scope': 'synthetic deformation identities on one Allen atlas; not biological animals or expert truth',
            'section_count': 0, 'numpy': np.__version__, 'torch': str(torch.__version__)}
(out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

context, atlas_inputs = bundle.load_pinned_allen_context(root / 'data/Allen Brain Atlas 25um')
assert context['v2_context_sha256'] == context_sha256
lower = np.zeros(3, dtype=np.float64)
upper = np.asarray(context['opaque_v1_context']['scalar_tensor'].shape, dtype=np.float64) * 25.
(out / 'context.json').write_text(json.dumps({
    'v2_context_sha256': context_sha256, 'receipt': acquisition._json_value(context['receipt']),
    'atlas_inputs': atlas_inputs, 'full_ccf_lower_ap_dv_ml_um': lower.tolist(),
    'full_ccf_upper_ap_dv_ml_um': upper.tolist(),
    'voxel_face_origin_ap_dv_ml_um': lower.tolist(), 'voxel_size_ap_dv_ml_um': [25., 25., 25.]}, indent=2))
del context
records = []
for identity in roster:
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower, upper, root_seed=seed, split='development',
        animal_index=identity['animal_index'], animal_id=identity['animal_id'],
        ccf_context_sha256=context_sha256)
    receipt = deformation.subject_deformation_plan_receipt_v2(plan)
    assert receipt['receipt_sha256'] == plan['receipt_sha256']
    for key, value in (('synthetic_animal_id', plan['synthetic_animal_id']),
                       ('subject_deformation_plan_id', plan['subject_deformation_plan_id']),
                       ('subject_deformation_realization_id', plan['subject_deformation_realization_id']),
                       ('subject_plan_receipt_sha256', plan['receipt_sha256'])):
        assert value not in reserved_ids[key]
    directory = out / f"subject_{identity['animal_index']:08d}"
    directory.mkdir()
    files = bundle._write_raw_artifact(directory, 'subject_plan', plan)
    (directory / 'subject_plan_receipt.json').write_text(json.dumps(receipt, indent=2))
    lineage = {**identity, 'synthetic_animal_id': plan['synthetic_animal_id'],
               'subject_deformation_plan_id': plan['subject_deformation_plan_id'],
               'subject_deformation_realization_id': plan['subject_deformation_realization_id'],
               'subject_plan_receipt_sha256': plan['receipt_sha256'],
               'root_seed': seed, 'ccf_context_sha256': context_sha256}
    (directory / 'lineage.json').write_text(json.dumps(lineage, indent=2))
    record = {**lineage, 'directory': directory.relative_to(out).as_posix(),
              'plan_files': files,
              'accepted_amplitude_um': plan['realization']['accepted_amplitude_um'],
              'accepted_candidate_index': plan['realization']['accepted_candidate_index'],
              'candidate_audits': acquisition._json_value(plan['realization']['candidate_audits']),
              'global_scale_ap_dv_ml': plan['state']['global_scale'].tolist(),
              'artifact_sha256': {name: sha(directory / name) for name in
                                  (*files.values(), 'subject_plan_receipt.json', 'lineage.json')}}
    records.append(record)
    (directory / 'completed.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'event': 'subject_frozen', 'subject_id': identity['subject_id'],
                      'ordinal': len(records), 'total': len(roster)}), flush=True)
    del plan
assert all(len({row[key] for row in records}) == len(roster) for key in identity_keys)
assert all(sha(path) == source_hashes[path.relative_to(repo).as_posix()] for path in source_paths)
completed = {'protocol': protocol, 'context_sha256': context_sha256,
             'subjects': records, 'section_count': 0,
             'artifact_sha256': {path.relative_to(out).as_posix(): sha(path)
                                 for path in sorted(out.rglob('*')) if path.is_file()}}
(out / 'completed.json').write_text(json.dumps(completed, indent=2))
print(json.dumps({'event': 'plans_frozen', 'subjects': len(records)}), flush=True)
