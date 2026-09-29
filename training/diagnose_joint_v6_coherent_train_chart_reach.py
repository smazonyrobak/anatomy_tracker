import hashlib
import json
from pathlib import Path

import numpy as np

source = Path('I:/AnatomyTracker/runs/joint_v6_coherent_train_candidates_001')
output = Path('I:/AnatomyTracker/runs/joint_v6_coherent_train_chart_reach_001')
identities = json.loads((source / 'observation_identities.json').read_text())
ouv = np.load(source / 'supervision_sidecar.npz')['canonical_anatomy_fitted_ouv_ap_dv_ml_um']
u, v = ouv[:, 1], ouv[:, 2]
span_u = np.linalg.norm(u, axis=1)
unit_u = u / span_u[:, None]
span_v = np.linalg.norm(v - unit_u * (v * unit_u).sum(1)[:, None], axis=1)
log_span = np.log(np.stack((span_u, span_v), axis=1) / 12000.)
outside = np.abs(log_span) > .36
raw_rows = [row for row in identities if row['selected_mode'] == 'raw']
assert len(raw_rows) == len(ouv) == 640
assert [row['section_index_in_cache'] for row in raw_rows] == list(range(640))
corpus = np.array([row['corpus'] for row in raw_rows])
eligibility = {mode: np.array([row['support_information_eligible'] for row in identities if row['selected_mode'] == mode]) for mode in ('raw', 'exact_black', 'imperfect_brush')}
summaries = {}
for name in ('base', 'acquisition'):
    summaries[name] = {}
    for mode, eligible in {'all_physical_sections': np.ones(640, dtype=bool), **eligibility}.items():
        keep = (corpus == name) & eligible
        summaries[name][mode] = {
            'section_count': int(keep.sum()),
            'either_qr_log_span_outside_count': int(outside[keep].any(1).sum()),
            'either_qr_log_span_outside_fraction': float(outside[keep].any(1).mean()),
            'per_axis_outside_counts_u_v': outside[keep].sum(0).tolist(),
            'above_positive_bound_count': int((log_span[keep] > .36).any(1).sum()),
            'below_negative_bound_count': int((log_span[keep] < -.36).any(1).sum()),
            'qr_span_um_quantiles_min_median_max': np.quantile(np.stack((span_u, span_v), axis=1)[keep], [0, .5, 1], axis=0).tolist(),
            'qr_log_span_quantiles_min_median_max': np.quantile(log_span[keep], [0, .5, 1], axis=0).tolist(),
        }
input_paths = [source / name for name in ('completed.json', 'supervision_sidecar.npz', 'observation_identities.json')]
input_paths += [source / 'source/training/cache_joint_v6_coherent_train_candidates.py']
input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
output.mkdir(parents=True, exist_ok=False)
np.savez(output / 'per_section.npz', section_index=np.arange(640), corpus=corpus, qr_span_um=np.stack((span_u, span_v), axis=1), qr_log_span=log_span, outside_per_axis=outside, **{mode + '_eligible': eligible for mode, eligible in eligibility.items()})
result = {
    'scope': 'Read-only frozen TRAIN geometry diagnostic; no target-array reread, model load, inference, renderer, training or benchmark. Existing8 synthetic subjects, not biological replication.',
    'definition': 'Positive QR spans: norm(U), norm(V - Uhat dot(Uhat,V)); logs relative to the shared12000um catalogue chart. Necessary-only span reach: either absolute log-span exceeds3*.12=.36. This does not test centre, rotation, shear, reflection, deformation, retrieval or full capture; in-bound is not success.',
    'eligibility': 'Each mode uses its own existing support_information_eligible boolean. raw counts are physical-section denominators, not threefold repeated observations; all_physical_sections includes censored rows.',
    'updates': 3,
    'absolute_per_update_log_span_bound': .12,
    'absolute_cumulative_log_span_bound': .36,
    'initial_qr_span_um': [12000., 12000.],
    'reachable_qr_span_interval_um': (12000. * np.exp(np.array([-.36, .36]))).tolist(),
    'source_sha256': input_hashes,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'per_section_sha256': hashlib.sha256((output / 'per_section.npz').read_bytes()).hexdigest(),
    'summaries': summaries,
}
(output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'summaries': summaries, 'result_sha256': hashlib.sha256((output / 'result.json').read_bytes()).hexdigest()}, indent=2))
