"""Frozen C4 synthetic cutting-normal information ceiling; no training."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090

ROOT = Path('I:/AnatomyTracker')
WORK = ROOT / 'agent_worktrees/joint_v6_integration'
RUN = ROOT / 'runs/c4_joint_321_002'
OUT = ROOT / 'runs/c4_cut_normal_323_001'
PROTOCOL = WORK / 'docs/publication/C4_CUT_NORMAL_323_PROTOCOL_20261015.md'
SEED = 20261015323
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not OUT.exists()
checkpoint = torch.load(RUN / 'joint_12000.pt', map_location='cpu', weights_only=True)
model = C4JointSliceModel(side=192).eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)
frozen = json.loads((RUN / 'readouts.jsonl').read_text().splitlines()[-1])
reference = {(row['cohort'], row['physical_section_id']): row for row in frozen['rows']
             if 'physical_section_id' in row}
draws = [json.loads(line) for line in (RUN / 'panel_draws.jsonl').open()]
provenance = {row['provenance']['physical_section_id']: row['provenance']
              for row in draws if row['accepted']}
assert checkpoint['batch'] == frozen['batch'] == 12000
assert [len(panels[key]) for key in ('inner', 'steep', 'sparse')] == [128, 32, 32]
assert len(provenance) == 192

rng = np.random.default_rng(SEED)
rows = []
for cohort in ('inner', 'steep', 'sparse'):
    for case in panels[cohort]:
        meta = case['meta']
        truth_state = torch.tensor(provenance[meta['physical_section_id']]
            ['postartifact_gauge_v1']['post_artifact_state12']).float()[None]
        truth_normal = full_frame_state_to_components(truth_state)[1][0, :, 2].numpy()
        truth_normal *= np.sign(truth_normal[np.abs(truth_normal).argmax()])
        axis = np.eye(3)[np.abs(truth_normal).argmin()]
        tangent_a = np.cross(truth_normal, axis)
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(truth_normal, tangent_a)
        phase = float(rng.uniform(-math.pi, math.pi))
        direction = math.cos(phase) * tangent_a + math.sin(phase) * tangent_b
        coarse_error_deg = float(rng.uniform(0, 20))
        reports = {'perfect': (truth_normal, 30.),
            'coarse': (math.cos(math.radians(coarse_error_deg)) * truth_normal +
                       math.sin(math.radians(coarse_error_deg)) * direction, 35.),
            'incorrect': (math.cos(math.radians(55)) * truth_normal +
                          math.sin(math.radians(55)) * direction, 35.)}
        with torch.inference_mode():
            prediction = model.predict(case['input'][None])
            scores = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)[0]
            branches = scores.topk(16).indices
            modes, flips = branches // 2, branches % 2
            states = prediction['state'][0, modes]
            normals = full_frame_state_to_components(states)[1][:, :, 2].numpy()
            chart = model.canonical_chart(case['chart'][None].expand(16, -1, -1),
                                          modes // model.normal_modes, 192)
            sites = rigid_points_090(states, flips, chart, (192, 192)).numpy()
        errors = np.linalg.norm(sites - case['truth'].numpy()[None], axis=-1).mean(-1) / 1000
        assert abs(float(errors[0]) - reference[(cohort, meta['physical_section_id'])]
                   ['first_rigid_gap_mm']) < .001
        arm = {}
        for name, (reported, angle) in reports.items():
            angle_cos = np.abs(normals @ reported)
            feasible = np.flatnonzero(angle_cos >= math.cos(math.radians(angle)))
            chosen = int(feasible[0]) if len(feasible) else None
            arm[name] = {'reported_error_deg': 0. if name == 'perfect' else
                         coarse_error_deg if name == 'coarse' else 55.,
                         'true_inside_cone': bool(abs(truth_normal @ reported) >=
                                                  math.cos(math.radians(angle))),
                         'feasible_count': int(len(feasible)), 'rank': chosen,
                         'selected_gap_mm': None if chosen is None else float(errors[chosen])}
        rows.append({'cohort': cohort, 'base': meta['base'],
            'physical_section_id': meta['physical_section_id'],
            'first_gap_mm': float(errors[0]), 'oracle_top16_gap_mm': float(errors.min()),
            'arms': arm})

summary = {}
gates = {}
for cohort in ('inner', 'steep', 'sparse'):
    group = [row for row in rows if row['cohort'] == cohort]
    result = {'n': len(group), 'unconstrained_mean_mm': float(np.mean(
        [row['first_gap_mm'] for row in group])),
        'top16_oracle_mean_mm': float(np.mean([row['oracle_top16_gap_mm'] for row in group]))}
    for name in ('perfect', 'coarse', 'incorrect'):
        usable = [row for row in group if row['arms'][name]['rank'] is not None]
        gain = float(np.mean([row['first_gap_mm'] - row['arms'][name]['selected_gap_mm']
                              for row in usable])) if usable else None
        result[name] = {'feasible': len(usable), 'fraction_feasible': len(usable) / len(group),
            'true_inside_cone': sum(row['arms'][name]['true_inside_cone'] for row in group),
            'paired_unconstrained_mean_mm': float(np.mean([row['first_gap_mm']
                                            for row in usable])) if usable else None,
            'selected_mean_mm': float(np.mean([row['arms'][name]['selected_gap_mm']
                                       for row in usable])) if usable else None,
            'paired_gain_mm': gain,
            'paired_unconstrained_over_5mm': sum(row['first_gap_mm'] > 5 for row in usable),
            'selected_over_5mm': sum(row['arms'][name]['selected_gap_mm'] > 5
                                      for row in usable)}
    common = [row for row in group if row['arms']['perfect']['rank'] is not None and
              row['arms']['coarse']['rank'] is not None]
    perfect_common_gain = float(np.mean([row['first_gap_mm'] -
        row['arms']['perfect']['selected_gap_mm'] for row in common])) if common else None
    coarse_common_gain = float(np.mean([row['first_gap_mm'] -
        row['arms']['coarse']['selected_gap_mm'] for row in common])) if common else None
    result['common_feasible'] = {'n': len(common), 'perfect_gain_mm': perfect_common_gain,
                                 'coarse_gain_mm': coarse_common_gain}
    result['base_gains_mm'] = {str(base): float(np.mean([row['first_gap_mm'] -
        row['arms']['coarse']['selected_gap_mm'] for row in group if row['base'] == base
        and row['arms']['coarse']['rank'] is not None])) for base in range(56, 64)
        if any(row['base'] == base and row['arms']['coarse']['rank'] is not None for row in group)}
    summary[cohort] = result
    gates[cohort + '_perfect_feasible'] = result['perfect']['fraction_feasible'] >= (
        .85 if cohort == 'sparse' else .95)
    gates[cohort + '_perfect_gain'] = (result['perfect']['paired_gain_mm'] is not None and
        result['perfect']['paired_gain_mm'] >= (.40 if cohort == 'inner' else .25))
    gates[cohort + '_coarse_feasible'] = result['coarse']['fraction_feasible'] >= (
        .80 if cohort == 'sparse' else .90)
    gates[cohort + '_coarse_tail'] = result['coarse']['selected_over_5mm'] <= \
        result['coarse']['paired_unconstrained_over_5mm']
    if cohort != 'sparse':
        gates[cohort + '_coarse_gain_retained'] = (perfect_common_gain is not None and
            perfect_common_gain > 0 and coarse_common_gain >= .60 * perfect_common_gain)

OUT.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(Path(__file__)),
    'checkpoint_sha256': sha(RUN / 'joint_12000.pt'), 'panel_sha256': sha(RUN / 'panels.pt'),
    'readout_sha256': sha(RUN / 'readouts.jsonl'),
    'draws_sha256': sha(RUN / 'panel_draws.jsonl'), 'device': 'cpu', 'seed': SEED,
    'scope': 'TRAIN synthetic information ceiling only; no acquired ground truth'}
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
result = {'cohorts': summary, 'gate_clauses': gates, 'gate_pass': all(gates.values()),
          'model_promoted': False, 'calibrated': False}
(OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
(OUT / 'completed.json').write_text(json.dumps({'config_sha256': sha(OUT / 'config.json'),
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'n': len(rows), 'gate_pass': result['gate_pass']}, indent=2) + '\n')
print(json.dumps(result, allow_nan=False), flush=True)
