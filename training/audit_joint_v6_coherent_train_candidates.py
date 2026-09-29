"""One post-exit CPU score/geometry reconstruction; no model imports or rendering."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/joint_v6_coherent_train_candidates_001'
OUT = ROOT / 'runs/joint_v6_coherent_train_candidates_001_independent_audit'
COMPLETION = '5f075e6ce34db1527c0c172757c6995a59fc8d7d42a833fbe25bf59942838ba2'
TOL = 2e-5  # CPU/GPU FP32 score/rank-roundoff envelope, not probability calibration.
started = time.perf_counter()
torch.set_num_threads(4)
hashes = {}


def sha(path):
    path = Path(path)
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    hashes[str(path)] = value
    return value


assert sha(RUN / 'completed.json') == COMPLETION
done = json.loads((RUN / 'completed.json').read_text())
cfg = done['experiment']
assert cfg == json.loads((RUN / 'experiment.json').read_text())
for name, expected in done['output_sha256'].items():
    assert sha(RUN / name) == expected
for name, expected in cfg['source']['file_sha256'].items():
    assert sha(RUN / 'source' / name) == expected
for name in ('checkpoint', 'gallery', 'catalogue'):
    assert sha(cfg[name]) == cfg[name + '_sha256'] == done['input_sha256'][cfg[name]]
parent = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
assert json.dumps(parent['experiment']['model_kwargs'], sort_keys=True) == json.dumps(cfg['model_kwargs'], sort_keys=True)
del parent
cat = torch.load(cfg['catalogue'], map_location='cpu', weights_only=False)
assert cat['receipt_sha256'] == cfg['catalogue_receipt_sha256']
assert done['source_unchanged'] and cfg['optimizer_steps'] == cfg['native_refinement_calls'] == cfg['gallery_rebuilds'] == 0
saved = dict(np.load(RUN / 'candidates.npz', allow_pickle=False))
side = dict(np.load(RUN / 'supervision_sidecar.npz', allow_pickle=False))
readout = dict(np.load(RUN / 'descriptive_readouts.npz', allow_pickle=False))
ids = json.loads((RUN / 'observation_identities.json').read_text())
sections = json.loads((RUN / 'section_provenance.json').read_text())
assert len(ids) == 1920 and len(sections) == 640
assert len({r['observation_id'] for r in ids}) == 1920 and {r['split'] for r in ids} == {'train'}
assert len({r['subject_id'] for r in ids}) == 8
expected_sections = []
for corpus, directory, count in [('base', 'joint_v6_coherent_subject_cohort_sections_002', 512), ('acquisition', 'joint_v6_coherent_acquisition_views_001', 128)]:
    directory = ROOT / 'data' / directory
    assert sha(directory / 'completed.json') == done['input_sha256'][str(directory / 'completed.json')]
    records = [r for r in json.loads((directory / 'completed.json').read_text())['sections'] if r['lineage']['split'] == 'train']
    assert len(records) == count
    expected_sections.extend((corpus, directory, r) for r in records)
for i, ((corpus, directory, record), section) in enumerate(zip(expected_sections, sections)):
    assert section['corpus'] == corpus and section['record'] == record and section['source_directory'] == str(directory)
    for path, digest in record['artifact_sha256'].items():
        assert done['input_sha256'][str(directory / path)] == digest  # Original cache authenticated bytes; no second raw-data scan.
    for j, mode in enumerate(('raw', 'exact_black', 'imperfect_brush')):
        row = ids[3*i+j]
        assert all(row[k] == v for k, v in record['lineage'].items())
        assert row['corpus'] == corpus and row['selected_mode'] == mode and row['section_index_in_cache'] == i
        assert row['arrays_path'] == str(directory / record['artifacts']['arrays'])
        assert len(row['input_channels_sha256']) == 64 and row['input_array_key']
        assert all(row[k] == v for k, v in record['by_mode'][mode].items())
        assert row['support_information_eligible'] != row['support_censored']
        if mode == 'raw': assert not row['outline_available']
assert np.array_equal(saved['observation_section_index'], np.repeat(np.arange(640), 3))
assert not side['reflection_xy'][:, 1].any()
assert np.array_equal(side['reflection_xy'][:, 0], [r['record']['horizontal_reflection'] for r in sections])
assert np.allclose(side['axial_weights'].sum(-1), 1, rtol=0, atol=1e-12)
assert np.isfinite(side['axial_offsets_um']).all() and (side['axial_weights'] >= 0).all()
print(json.dumps({'authenticated_outputs_sources_and_parent': True, 'TRAIN_observations': 1920, 'sections': 640, 'synthetic_subjects': 8}), flush=True)

query = saved['query_descriptor'].astype(np.float32)
bank = np.load(cfg['gallery'], allow_pickle=False).astype(np.float32)
assert query.shape == (1920, 256) and bank.shape == (98304, 2, 256)
assert np.isfinite(query).all() and np.isfinite(bank).all()
query /= np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), np.float32(1e-12))
bank /= np.maximum(np.linalg.norm(bank, axis=-1, keepdims=True), np.float32(1e-12))
mass = cat['tensors']['cell_log_mass'][0].numpy().astype(np.float32)
rprior = cat['tensors']['representation_log_weight'][0].numpy().astype(np.float32)
top = saved['top128_cell_index']
assert top.shape == (1920, 128) and ((top >= 0) & (top < 98304)).all()
assert (np.diff(np.sort(top, axis=1), axis=1) != 0).all()
errors = {k: 0. for k in ('cell_lp', 'conditional_R_lp', 'conditional_R_probability', 'full_component_lp', 'reconstructed_retained_mass', 'full_roundoff_normalizer', 'rank_interval_score_violation')}
different_order_rows = different_membership_rows = 0
for start in range(0, 1920, 16):
    end = start + 16
    score = (query[start:end] @ bank.reshape(-1, 256).T).reshape(16, 98304, 2) / np.float32(.1) + rprior[None]
    cell = np.logaddexp(score[..., 0], score[..., 1]) + mass[None]
    lp = cell - logsumexp(cell, axis=1, keepdims=True)
    assert np.isfinite(lp).all()
    selected = top[start:end]
    slp = np.take_along_axis(lp, selected, axis=1)
    sr = score[np.arange(16)[:, None], selected]
    rlp = sr - logsumexp(sr, axis=-1, keepdims=True)
    rprob = np.exp(rlp)
    comp = slp[..., None] + rlp
    z64 = logsumexp(lp.astype(np.float64), axis=1)
    retained = np.exp(logsumexp(slp.astype(np.float64), axis=1) - z64)
    for name, actual, key in [('cell_lp', slp, 'top128_cell_log_probability'), ('conditional_R_lp', rlp, 'top128_conditional_representation_log_probability'), ('conditional_R_probability', rprob, 'top128_conditional_representation_probability'), ('full_component_lp', comp, 'top128_full_catalogue_component_log_probability'), ('reconstructed_retained_mass', retained, 'retained_coarse_mass'), ('full_roundoff_normalizer', z64, 'full_cell_lp_roundoff_log_normalizer')]:
        errors[name] = max(errors[name], float(np.max(np.abs(actual - saved[key][start:end]))))
    order = np.argsort(-lp, axis=1, kind='stable')
    reference = np.take_along_axis(lp, order[:, :128], axis=1)
    # At each saved rank, require the independently reconstructed score to equal
    # that rank's order statistic within roundoff; this covers boundary swaps too.
    errors['rank_interval_score_violation'] = max(errors['rank_interval_score_violation'], float(np.max(np.abs(slp-reference))))
    different_order_rows += int(np.any(order[:, :128] != selected, axis=1).sum())
    different_membership_rows += int(np.any(np.sort(order[:, :128], axis=1) != np.sort(selected, axis=1), axis=1).sum())
    if end % 256 == 0: print(json.dumps({'CPU_full_gallery_reconstructed': end, 'max_cell_lp_error': errors['cell_lp']}), flush=True)
assert max(errors.values()) < TOL
tlp = saved['top128_cell_log_probability'].astype(np.float64)
rlp = saved['top128_conditional_representation_log_probability'].astype(np.float64)
assert np.max(np.abs(logsumexp(rlp, axis=-1))) < 2e-6
assert np.max(np.abs(saved['top128_conditional_representation_probability'].sum(-1)-1)) < 2e-6
assert np.max(np.abs(saved['top128_full_catalogue_component_log_probability'] - (tlp[..., None] + rlp))) < 2e-6
retained = np.exp(logsumexp(tlp, axis=1) - saved['full_cell_lp_roundoff_log_normalizer'])
omitted = -np.expm1(logsumexp(tlp, axis=1) - saved['full_cell_lp_roundoff_log_normalizer'])
assert np.max(np.abs(retained - saved['retained_coarse_mass'])) < 1e-12
assert np.max(np.abs(omitted - saved['omitted_coarse_mass'])) < 1e-12
assert np.max(np.abs(retained+omitted-1)) < 1e-12

# Independent NumPy Gram-Schmidt state decoding and explicit four pixel corners.
state = np.asarray(cat['arrays']['cell_states_float64'], dtype=np.float64)
u = state[:, 3:6] / np.linalg.norm(state[:, 3:6], axis=1, keepdims=True)
v = state[:, 6:9] - np.sum(state[:, 6:9]*u, axis=1, keepdims=True)*u
v /= np.linalg.norm(v, axis=1, keepdims=True)
U = np.exp(state[:, 9, None])*u
V = np.exp(state[:, 10, None])*(v+state[:, 11, None]*u)
O = state[:, :3] - .5*(U+V)
n = np.cross(U, V); n /= np.linalg.norm(n, axis=1, keepdims=True)
assert np.max(np.abs(n-cat['arrays']['cell_normal_ap_dv_ml_float64'])) < 1e-12
origin = np.asarray(cat['support_geometry']['support_origin_ap_dv_ml_um'])
truth = side['canonical_anatomy_fitted_ouv_ap_dv_ml_um'][saved['observation_section_index']]
observed = side['already_observed_total_map_fitted_ouv_ap_dv_ml_um'][saved['observation_section_index']]
nt = np.cross(truth[:, 1], truth[:, 2]); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
dot = np.sum(n[top]*nt[:, None], axis=-1)
angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n[top], nt[:, None]), axis=-1), np.abs(dot)))
offset = np.abs(np.sum((truth[:, 0]-origin)*nt, axis=-1)[:, None] - np.where(dot < 0, -1., 1.)*np.sum((O-origin)*n, axis=-1)[top])
near = (angle <= 10) & (offset <= 500)
st = np.array([[0, 0], [95/96, 0], [0, 95/96], [95/96, 95/96]])
corners = O[:, None] + st[None, :, :1]*U[:, None] + st[None, :, 1:]*V[:, None]
truth_corners = observed[:, None, 0] + st[None, :, :1]*observed[:, None, 1] + st[None, :, 1:]*observed[:, None, 2]
branch_corners = np.stack((corners[top], corners[top][:, :, [1, 0, 3, 2]]), axis=2)
rms = np.sqrt(np.sum((branch_corners-truth_corners[:, None, None])**2, axis=-1).mean(-1))
pred_r = saved['top128_conditional_representation_probability'][:, 0].argmax(-1)
best = rms.reshape(1920, -1).argmin(-1)
assert np.array_equal(pred_r, readout['predicted_R_of_top_cell'])
assert np.array_equal(best//2, readout['oracle_best_frame_rank_zero_based']) and np.array_equal(best%2, readout['oracle_best_frame_R'])
metrics = {'top1_plane_angle_deg': angle[:, 0], 'top1_normal_offset_error_um': offset[:, 0], 'physical_plane_capture_at32': near[:, :32].any(1), 'physical_plane_capture_at128': near.any(1), 'top1_predicted_R_finite_four_corner_rms_um': rms[np.arange(1920), 0, pred_r], 'oracle_best_top128x2_finite_four_corner_rms_um': rms.reshape(1920, -1)[np.arange(1920), best], 'retained_coarse_mass': retained, 'omitted_coarse_mass': omitted}
geometry_errors = {k: float(np.max(np.abs(v.astype(float)-readout[k]))) for k, v in metrics.items()}
assert max(geometry_errors.values()) < 1e-7
eligible = np.array([r['support_information_eligible'] for r in ids]); subject = np.array([r['subject_id'] for r in ids])
corpus = np.array([r['corpus'] for r in ids]); mode = np.array([r['selected_mode'] for r in ids])
summaries = {}; aggregate_error = 0.
for label, original in done['summaries'].items():
    if label == 'all': mask = np.ones(1920, dtype=bool)
    else:
        a, b, c = label.split(':')
        mask = (corpus == a) & (mode == b) & (True if c == 'all' else eligible if c == 'eligible' else ~eligible)
    groups = [mask & (subject == s) for s in np.unique(subject[mask])]
    assert int(mask.sum()) == original['observations'] and len(groups) == original['subjects']
    assert int((mask & eligible).sum()) == original['eligible'] and int((mask & ~eligible).sum()) == original['censored']
    macro = {k: float(np.mean([v[g].mean() for g in groups])) for k, v in metrics.items()}
    aggregate_error = max(aggregate_error, max(abs(macro[k]-original['subject_macro'][k]) for k in macro))
    for k, v in original['row_p50_p90_p95'].items():
        aggregate_error = max(aggregate_error, float(np.max(np.abs(np.quantile(metrics[k][mask], [.5, .9, .95])-v))))
    summaries[label] = {'observations': int(mask.sum()), 'eligible': int((mask & eligible).sum()), 'subject_macro': macro}
assert aggregate_error < 1e-7
OUT.mkdir(exist_ok=False)
result = {'integrity_pass': True, 'completion_sha256': COMPLETION, 'auditor_sha256': sha(__file__), 'input_output_source_sha256': hashes, 'counts': {'observations': 1920, 'sections': 640, 'base_sections': 512, 'acquisition_sections': 128, 'synthetic_subjects': 8, 'eligible_observations': int(eligible.sum()), 'censored_observations': int((~eligible).sum())}, 'cpu_reconstruction_max_abs_errors': errors, 'score_tolerance': TOL, 'cpu_GPU_different_top128_order_rows': different_order_rows, 'cpu_GPU_different_top128_membership_rows': different_membership_rows, 'geometry_max_abs_errors': geometry_errors, 'aggregate_max_abs_error': aggregate_error, 'summaries': summaries, 'scope': 'All1920 TRAIN rows, complete98304x2 score reconstruction from saved descriptors and matching whole-model gallery; CPU NumPy/SciPy only scoring/geometry, torch CPU deserialize only. Stable saved GPU rankings remain authoritative within documented FP32 tolerance. No encoder/rerender/native replay, no fresh underlying raw-artifact hashes; existing authenticated corpus records and cached input byte receipts checked. Frame geometry is a fitted-frame proxy, not curved dense truth. Eight synthetic subjects, no biological validation, calibration or advancement gate.', 'elapsed_seconds': time.perf_counter()-started}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'integrity_pass': True, 'counts': result['counts'], 'score_errors': errors, 'different_order_rows': different_order_rows, 'different_membership_rows': different_membership_rows, 'geometry_errors': geometry_errors, 'aggregate_error': aggregate_error, 'audit_sha256': sha(OUT / 'audit.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
