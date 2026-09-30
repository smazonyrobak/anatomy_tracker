"""One CPU-only probe on one frozen TRAIN animal; no training or live data I/O."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True

import hashlib
import json
import time

import nrrd
import numpy as np
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_probe_observations_v7 import sample_subject_probe, sample_probe_metadata, observe_probe_section
from training.arbitrary_plane_subject_deformation_v2 import _ccf_to_subject_points_from_verified_plan_v2
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact, _write_raw_artifact


PLAN_ROOT = ROOT / 'data/joint_v6_coherent_subject_plans_002'
SECTIONS = ROOT / 'data/joint_v6_coherent_subject_cohort_sections_002'
OUTPUT = ROOT / 'data/joint_v7_probe_observation_cpu_check_002'
ATLAS = ROOT / 'data/Allen Brain Atlas 25um'
SEED = 2026093017
started = time.perf_counter()
plan_bytes = (PLAN_ROOT / 'completed.json').read_bytes()
assert hashlib.sha256(plan_bytes).hexdigest() == '83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89'
subject = next(r for r in json.loads(plan_bytes)['subjects'] if r['split'] == 'train' and r['animal_index'] == 0)
for name, expected in subject['artifact_sha256'].items():
    assert hashlib.sha256((PLAN_ROOT / subject['directory'] / name).read_bytes()).hexdigest() == expected
plan = _read_raw_artifact(PLAN_ROOT / subject['directory'], subject['plan_files'])
repository = Path(__file__).resolve().parents[1]
for name, expected in plan['source_sha256'].items():
    content = (repository / 'training' / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    assert hashlib.sha256(content).hexdigest() == expected
annotation_hash = hashlib.sha256((ATLAS / 'annotation_25.nrrd').read_bytes()).hexdigest()
assert annotation_hash == 'c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42'
annotation = nrrd.read(str(ATLAS / 'annotation_25.nrrd'), index_order='F')[0]
ontology_bytes = (ROOT / 'tmp/allen_structure_graph_1_probe_check_20260930.json').read_bytes()
ontology_hash = hashlib.sha256(ontology_bytes).hexdigest()
assert ontology_hash == 'b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386'
pending = [(node, False) for node in json.loads(ontology_bytes)['msg']]
cortical_ids = set()
while pending:
    node, is_cortical = pending.pop()
    is_cortical = is_cortical or node['id'] == 315
    if is_cortical:
        cortical_ids.add(node['id'])
    pending.extend((child, is_cortical) for child in node['children'])
lineage = {k: subject[k] for k in ('subject_id', 'animal_id', 'specimen_id', 'experiment_id', 'synthetic_animal_id', 'split')}
sources = {'subject_plan_receipt_sha256': subject['subject_plan_receipt_sha256'],
           'plan_completion_sha256': hashlib.sha256(plan_bytes).hexdigest(),
           'annotation_raw_sha256': annotation_hash,
           'ontology_sha256': ontology_hash, 'cortical_descendants_including315': len(cortical_ids),
           'ontology_url': 'https://api.brain-map.org/api/v2/structure_graph_download/1.json'}
probe = sample_subject_probe(plan, annotation, cortical_ids, lineage, 'probe-000', SEED,
                             np.array([5400., 332., 5739.]) + 12.5, sources)
ap, dv, ml = probe['entry_dorsal_atlas_voxel_ap_dv_ml']
assert annotation[ap, dv, ml] in cortical_ids and not annotation[ap, :dv, ml].any()
np.testing.assert_array_equal(probe['entry_ccf_ap_dv_ml_um'], np.array([ap + .5, dv, ml + .5]) * 25.)
track_voxels = np.floor(probe['track_ccf_ap_dv_ml_um'][1:] / 25.).astype(int)
assert (annotation[tuple(track_voxels.T)] > 0).all()
del annotation
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'allen_structure_graph_1.json').write_bytes(ontology_bytes)
probe_files = _write_raw_artifact(OUTPUT, 'probe', probe)
metadata = [sample_probe_metadata(probe, SEED, 'ordinary'),
            sample_probe_metadata(probe, SEED, 'all-consistent', missing_probability=0., contradiction_probability=0.),
            sample_probe_metadata(probe, SEED, 'all-contradictory', missing_probability=0., contradiction_probability=1.),
            sample_probe_metadata(probe, SEED, 'all-missing', missing_probability=1., contradiction_probability=0.)]
_write_raw_artifact(OUTPUT, 'metadata', {'variants': metadata})
assert all(row['context_raw_units'][2] == 0 for row in metadata)
assert metadata[-1]['context_float32'].sum() == 0
assert metadata[2]['training_only_contradictory_entry_angle_depth'].all()
ordinary, consistent, contradictory, missing = metadata
truth_entry = probe['entry_subject_stereotaxic_ap_dv_ml_um'][[0, 2]]
for row, expected in ((consistent, True), (contradictory, False)):
    r = row['context_raw_units']
    assert bool(np.linalg.norm(truth_entry - r[3:5]) <= r[5]) == expected
    assert bool(r[7] <= probe['elevation_deg'] <= r[8]) == expected
    assert bool(probe['depth_um'] <= r[10]) == expected
rows = json.loads((SECTIONS / 'train/subject_00000000/completed.json').read_text())['sections']
checks, observations = [], []
for row in rows:
    raw = (SECTIONS / row['artifacts']['metadata']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['artifact_sha256'][row['artifacts']['metadata']]
    description = json.loads(raw)
    with np.load(SECTIONS / row['artifacts']['arrays'], allow_pickle=False) as arrays:
        ouv = arrays[description['subject_ouv_ap_dv_ml_um_float64']['__ndarray__']]
        reflection = arrays[description['reflection_xy']['__ndarray__']]
        pullback = arrays[description['section_processing_observed_pullback_yx_px_float64']['__ndarray__']]
        centre_ccf = arrays[description['target_centre_ccf_coordinates_ap_dv_ml_um_float64']['__ndarray__']]
        for presentation in description['observations']:
            visibility = arrays[presentation['visible_finite_support_float32']['__ndarray__']]
            obs = observe_probe_section(probe, plan, ouv, reflection, row['thickness_um'], visibility,
                description['lineage'], presentation['lineage']['observation_id'], SEED,
                observed_pullback_yx_px=pullback, source_identifiers=row['artifact_sha256'])
            xy = obs['candidate_pixel_xy']
            canonical = xy * (1 - 2 * reflection.astype(int)) + reflection * (visibility.shape[::-1] - np.ones(2))
            normal = np.cross(ouv[1], ouv[2]); normal /= np.linalg.norm(normal)
            recovered = (ouv[0] + canonical[:, :1] / visibility.shape[1] * ouv[1]
                         + canonical[:, 1:] / visibility.shape[0] * ouv[2]
                         + obs['candidate_through_plane_offset_um'][:, None] * normal)
            geometry_error = float(np.linalg.norm(recovered - obs['candidate_track_subject_ap_dv_ml_um'], axis=1).max(initial=0.))
            interpolated = np.column_stack([map_coordinates(centre_ccf[..., k], xy[:, ::-1].T, order=1) for k in range(3)]) if len(xy) else np.empty((0, 3))
            interpolation_error = float(np.linalg.norm(interpolated - obs['candidate_projected_centre_ccf_ap_dv_ml_um'], axis=1).max(initial=0.))
            assert geometry_error < 1e-6
            assert np.abs(obs['candidate_through_plane_offset_um']).max(initial=0.) <= row['thickness_um'] / 2 + 1e-7
            assert not (obs['candidate_selected'] & (obs['candidate_visibility_weight'] <= .05)).any()
            checks.append({'section_id': row['lineage']['section_id'], 'mode': presentation['selected_mode'],
                           'horizontal_reflection': bool(reflection[0]), 'geometric_intersection': obs['geometric_intersection'],
                           'marks': len(obs['mark_pixel_xy']), 'candidates': len(xy), 'geometry_error_um': geometry_error,
                           'frozen_raster_bilinear_error_um': interpolation_error})
            observations.append(obs)
    # Paired views use identical candidate locations/noise/dropout, before mask visibility.
    for obs in observations[-2:]:
        np.testing.assert_array_equal(obs['candidate_noisy_pixel_xy'], observations[-3]['candidate_noisy_pixel_xy'])
_write_raw_artifact(OUTPUT, 'observations', {'observations': observations})
forward = _ccf_to_subject_points_from_verified_plan_v2(probe['track_ccf_ap_dv_ml_um'], plan)
cycle_error = np.linalg.norm(forward - probe['track_subject_ap_dv_ml_um'], axis=1).max()
ccf = probe['track_ccf_ap_dv_ml_um']
chord = ccf[0] + (probe['track_depth_um'] / probe['depth_um'])[:, None] * (ccf[-1] - ccf[0])
summary = {
    'prepared_not_trained': True, 'split': 'train', 'synthetic_animals': 1, 'probes': 1,
    'section_count': len(rows), 'presentation_count': len(checks), 'source_identifiers': sources,
    'seed': SEED, 'probe_draw_attempt': probe['physical_draw_attempt'], 'depth_um': probe['depth_um'],
    'elevation_deg': probe['elevation_deg'], 'geometric_hit_sections': sum(r['geometric_intersection'] for r in checks[::3]),
    'selected_mark_count': sum(r['marks'] for r in checks),
    'max_reflection_and_plane_reconstruction_error_um': max(r['geometry_error_um'] for r in checks),
    'max_frozen_raster_bilinear_error_um': max(r['frozen_raster_bilinear_error_um'] for r in checks),
    'inverse_forward_cycle_max_um': float(cycle_error), 'ccf_chord_departure_max_um': float(np.linalg.norm(ccf - chord, axis=1).max()),
    'source_sha256': {name: hashlib.sha256((repository / 'training' / name).read_bytes()).hexdigest()
                      for name in ('arbitrary_plane_probe_observations_v7.py', Path(__file__).name)},
    'supersedes_invalid_check': 'joint_v7_probe_observation_cpu_check_001 used centre-origin entry/occupancy instead of project voxel-boundary-origin geometry; do not use that output',
    'seconds': time.perf_counter() - started, 'checks': checks,
    'limitations': 'one fixed TRAIN probe; voxel-face dorsal surface and sampled brain exit; identity section processing only; synthetic optional clicks not fluorescence; subject-frame bounds not a GUI adapter; no calibration or performance claim',
}
(OUTPUT / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in summary.items() if k != 'checks'}), flush=True)
