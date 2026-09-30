"""CPU-only, TRAIN-only shared physical probes; no section/model/live-output I/O."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
sys.dont_write_bytecode = True

import hashlib
import json
import time

import nrrd
import numpy as np
from threadpoolctl import threadpool_limits

from training.arbitrary_plane_probe_observations_v7 import (
    CONTEXT_DIVISORS, CONTEXT_FIELDS, sample_subject_probe, sample_probe_metadata,
)
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact


threadpool_limits(limits=2)
PLAN_ROOT = ROOT / 'data/joint_v6_coherent_subject_plans_002'
CHECK_ROOT = ROOT / 'data/joint_v7_probe_observation_cpu_check_002'
ATLAS = ROOT / 'data/Allen Brain Atlas 25um'
OUTPUT = ROOT / 'data/joint_v7_shared_probe_bank_train_001'
SEED, PER_SUBJECT = 2026093021, 64
VARIANTS = (('full_consistent_0', 0., 0.), ('full_consistent_1', 0., 0.),
            ('partly_missing', .5, 0.), ('occasionally_contradictory', .2, .15),
            ('all_contradictory', 0., 1.), ('all_missing', 1., 0.))
repository = Path(__file__).resolve().parents[1]
started = time.perf_counter()
plan_bytes = (PLAN_ROOT / 'completed.json').read_bytes()
plan_sha = hashlib.sha256(plan_bytes).hexdigest()
assert plan_sha == '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
cohort = json.loads(plan_bytes)
subjects = sorted((r for r in cohort['subjects'] if r['split'] == 'train'), key=lambda r: r['animal_index'])
assert len(subjects) == 8 and {r['animal_index'] for r in subjects} == set(range(8))
annotation_sha = hashlib.sha256((ATLAS / 'annotation_25.nrrd').read_bytes()).hexdigest()
assert annotation_sha == 'c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42'
ontology_bytes = (CHECK_ROOT / 'allen_structure_graph_1.json').read_bytes()
ontology_sha = hashlib.sha256(ontology_bytes).hexdigest()
assert ontology_sha == 'b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386'
pending = [(node, False) for node in json.loads(ontology_bytes)['msg']]
cortical_ids = set()
while pending:
    node, is_cortical = pending.pop()
    is_cortical = is_cortical or node['id'] == 315
    if is_cortical:
        cortical_ids.add(node['id'])
    pending.extend((child, is_cortical) for child in node['children'])
annotation = nrrd.read(str(ATLAS / 'annotation_25.nrrd'), index_order='F')[0]
top = (annotation != 0).argmax(axis=1)
ap, ml = np.indices(top.shape)
ap, ml = np.argwhere(np.isin(annotation[ap, top, ml], list(cortical_ids))).T
candidates = np.column_stack((ap, top[ap, ml], ml))
sources = {name: (repository / 'training' / name).read_bytes() for name in
           ('arbitrary_plane_probe_observations_v7.py', Path(__file__).name)}
protocol = {
    'prepared_not_trained': True, 'seed': SEED, 'split': 'train', 'synthetic_subjects': 8,
    'probes_per_subject': PER_SUBJECT, 'metadata_variants': VARIANTS,
    'plan_completion_sha256': plan_sha, 'atlas_annotation_sha256': annotation_sha,
    'ontology_sha256': ontology_sha, 'ontology_source': 'https://api.brain-map.org/api/v2/structure_graph_download/1.json',
    'source_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    'cortical_descendants': sorted(cortical_ids), 'dorsal_cortical_columns': len(candidates),
    'dorsal_cortical_voxels_sha256': hashlib.sha256(candidates.astype('<i8').tobytes()).hexdigest(),
    'cpu_thread_limit': 2, 'runtime': {'python': sys.version, 'numpy': np.__version__, 'pynrrd': nrrd.__version__},
    'physical_geometry': 'voxel-boundary origin0; dorsal face at25*[AP+.5,first_tissue_DV,ML+.5]; GUI legacy bregma index converted with+.5 voxel',
    'bregma_ccf_ap_dv_ml_um': [5412.5, 344.5, 5751.5],
    'track_sampling': 'one independently seeded subject-space straight track per animal/probe; elevation45-90deg, hidden azimuth uniform; requested depth1500-6000um; stop before first sampled exit, minimum500um; only physical-entry retries, never section-based',
    'metadata_semantics': 'subject stereotaxic hard AP/ML disk, hard elevation interval without azimuth, hard maximum path length; perturbed nominal entry remains inside correct bounds unless corruption is labelled; hard bounds are not measurement-noise SDs',
    'context_fields': CONTEXT_FIELDS, 'context_divisors': CONTEXT_DIVISORS.tolist(),
    'context_ap_anchor': None, 'section_ap_bounds_available': False,
    'array_layout': 'one NPZ/subject; track_offsets[i]:track_offsets[i+1] select probe i from concatenated float64 track arrays; context arrays [probe,variant,12]; JSONL has exact raw metadata/IDs/seeds',
    'limitations': 'eight synthetic maps of one atlas, not512 biological animals; alternative probes not512 simultaneous implants; no image tracks/section clicks/AP bounds generated; not trainer or GUI integration; no validation or calibration claims',
}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'source').mkdir()
for name, content in sources.items():
    (OUTPUT / 'source' / name).write_bytes(content)
(OUTPUT / 'parent_plan_completed.json').write_bytes(plan_bytes)
(OUTPUT / 'allen_structure_graph_1.json').write_bytes(ontology_bytes)
(OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
protocol_sha = hashlib.sha256((OUTPUT / 'protocol.json').read_bytes()).hexdigest()
completed = []
print(json.dumps({'event': 'TRAIN_shared_probe_bank_start', 'subjects': len(subjects), 'probes': len(subjects) * PER_SUBJECT,
                  'cpu_threads': 2, 'dorsal_cortical_columns': len(candidates)}), flush=True)
for subject in subjects:
    tick = time.perf_counter()
    directory = PLAN_ROOT / subject['directory']
    for name, expected in subject['artifact_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    plan = _read_raw_artifact(directory, subject['plan_files'])
    for name, expected in plan['source_sha256'].items():
        content = (repository / 'training' / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        assert hashlib.sha256(content).hexdigest() == expected
    lineage = {k: subject[k] for k in ('subject_id', 'animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id', 'split')}
    bindings = {'subject_plan_receipt_sha256': subject['subject_plan_receipt_sha256'],
                'plan_completion_sha256': plan_sha, 'annotation_raw_sha256': annotation_sha,
                'ontology_sha256': ontology_sha, 'protocol_sha256': protocol_sha}
    records, depths, subject_tracks, ccf_tracks, offsets, contexts, raw_contexts, corruptions = [], [], [], [], [0], [], [], []
    for index in range(PER_SUBJECT):
        probe = sample_subject_probe(plan, annotation, cortical_ids, lineage, f'probe-{index:03d}', SEED,
            np.array(protocol['bregma_ccf_ap_dv_ml_um']), bindings, dorsal_cortical_voxels=candidates)
        metadata = [sample_probe_metadata(probe, SEED, name, missing_probability=missing,
                    contradiction_probability=contradictory) for name, missing, contradictory in VARIANTS]
        # JSON keeps physical metadata; dense curve arrays use one compact ragged NPZ.
        record = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in probe.items()
                  if k not in ('track_depth_um', 'track_subject_ap_dv_ml_um', 'track_ccf_ap_dv_ml_um')}
        record['probe_index'] = index
        record['metadata_variants'] = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()
                                       if k not in ('lineage', 'context_fields', 'context_divisors', 'bregma_subject_ap_dv_ml_um', 'metadata_frame')}
                                      for row in metadata]
        records.append(record)
        depths.append(probe['track_depth_um'])
        subject_tracks.append(probe['track_subject_ap_dv_ml_um'])
        ccf_tracks.append(probe['track_ccf_ap_dv_ml_um'])
        offsets.append(offsets[-1] + len(depths[-1]))
        contexts.append(np.stack([row['context_float32'] for row in metadata]))
        raw_contexts.append(np.stack([row['context_raw_units'] for row in metadata]))
        corruptions.append(np.stack([row['training_only_contradictory_entry_angle_depth'] for row in metadata]))
    stem = f"subject_{subject['animal_index']:08d}"
    np.savez_compressed(OUTPUT / f'{stem}.npz', track_offsets=np.array(offsets, dtype=np.int64),
        track_depth_um=np.concatenate(depths), track_subject_ap_dv_ml_um=np.concatenate(subject_tracks),
        track_ccf_ap_dv_ml_um=np.concatenate(ccf_tracks), context_float32=np.stack(contexts),
        context_raw_units=np.stack(raw_contexts), metadata_contradictory=np.stack(corruptions))
    (OUTPUT / f'{stem}.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf8')
    result = {'animal_id': subject['animal_id'], 'subject_id': subject['subject_id'], 'probes': PER_SUBJECT,
        'metadata_variants': PER_SUBJECT * len(VARIANTS), 'track_points': offsets[-1],
        'subject_plan_receipt_sha256': subject['subject_plan_receipt_sha256'],
        'files': {f'{stem}{suffix}': hashlib.sha256((OUTPUT / f'{stem}{suffix}').read_bytes()).hexdigest() for suffix in ('.npz', '.jsonl')},
        'bytes': sum((OUTPUT / f'{stem}{suffix}').stat().st_size for suffix in ('.npz', '.jsonl')),
        'physical_retry_count': sum(row['physical_draw_attempt'] for row in records),
        'depth_um_range': [min(row['depth_um'] for row in records), max(row['depth_um'] for row in records)],
        'seconds': time.perf_counter() - tick}
    completed.append(result)
    print(json.dumps({'event': 'TRAIN_shared_probes_subject_frozen', **result}), flush=True)
for name, content in sources.items():
    assert (repository / 'training' / name).read_bytes() == content
summary = {'prepared_not_trained': True, 'split': 'train', 'protocol_sha256': protocol_sha,
    'subjects': completed, 'probes': sum(row['probes'] for row in completed),
    'metadata_variants': sum(row['metadata_variants'] for row in completed),
    'bytes': sum(row['bytes'] for row in completed), 'seconds': time.perf_counter() - started}
(OUTPUT / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({'event': 'TRAIN_shared_probe_bank_frozen', **{k: v for k, v in summary.items() if k != 'subjects'}}), flush=True)
