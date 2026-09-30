"""One terminal-only closeout of the two completed TRAIN experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker/runs')
result = {}
for name in ('joint_v6_train_pose_jacobian_readout_001', 'joint_v6_coherent_canonical_adaptation_001'):
    folder = root / name
    completion = json.loads((folder / 'completed.json').read_text())
    mismatches = []
    for file, expected in completion['output_sha256'].items():
        with (folder / file).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            mismatches.append(file)
    result[name] = {'completion_sha256': hashlib.sha256((folder / 'completed.json').read_bytes()).hexdigest(),
                    'output_mismatches': mismatches, 'verified_files': len(completion['output_sha256'])}

folder = root / 'joint_v6_train_pose_jacobian_readout_001'
arrays = [np.load(folder / f'section_{i:02d}.npz') for i in range(8)]
result[folder.name]['paired_trajectories'] = 144
result[folder.name]['arms'] = ['learned_capped', 'damped_feature_solve']
for key in ('normal_error_deg', 'centre_error_um', 'slab_error_um', 'half_squared_feature_cost'):
    values = np.stack([a[key] for a in arrays])
    result[folder.name][key] = values.mean(axis=(0, 2)).tolist()
result[folder.name]['nonfinite_solve_count'] = int(sum(a['nonfinite_solve'].sum() for a in arrays))
result[folder.name]['scope'] = 'TRAIN only; 8 synthetic subjects; fixed oracle fields; off-policy single-axis starts, not global inference'

folder = root / 'joint_v6_coherent_canonical_adaptation_001'
completion = json.loads((folder / 'completed.json').read_text())
result[folder.name]['gates'] = completion['gates']
result[folder.name]['TRAIN_effectiveness_gate_passed'] = completion['TRAIN_effectiveness_gate_passed']
for step in ('step_0000', 'step_1920'):
    result[folder.name][step] = {}
    for domain, identity_name, group_key in (
            ('coherent', 'coherent_TRAIN_identities.json', 'subject_id'),
            ('synthetic', 'synthetic_TRAIN_identities.json', 'animal_id'),
            ('real', 'real_TRAIN_identities.json', 'animal_id')):
        identities = json.loads((folder / identity_name).read_text())
        raw = np.load(folder / step / f'{domain}_rows.npz')
        groups = np.array([str(r[group_key]) for r in identities])
        eligible = raw['original_eligible'].astype(bool)
        metrics = {key: float(np.mean([raw[key][(groups == g) & eligible].mean()
                   for g in np.unique(groups[eligible])]))
                   for key in ('plane_angle_deg', 'normal_offset_error_um', 'capture32', 'capture128')}
        result[folder.name][step][domain] = metrics

output = root / 'joint_v6_final_diagnostics_closeout_001.json'
with output.open('x', encoding='utf8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
