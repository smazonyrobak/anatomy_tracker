"""Exploratory frozen sparse-cut check: apply a perfect 30° normal cone before C4 top-k."""
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
RUN = ROOT / 'runs/c4_joint_321_002'
PRIOR = ROOT / 'runs/c4_cut_normal_323_001'
OUT = ROOT / 'runs/c4_full_bank_normal_324_001'
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not OUT.exists()
checkpoint = torch.load(RUN / 'joint_12000.pt', map_location='cpu', weights_only=True)
model = C4JointSliceModel(side=192).eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
cases = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)['sparse']
prior = {row['physical_section_id']: row for row in
         map(json.loads, (PRIOR / 'rows.jsonl').read_text().splitlines())}
draws = [json.loads(line) for line in (RUN / 'panel_draws.jsonl').open()]
provenance = {row['provenance']['physical_section_id']: row['provenance']
              for row in draws if row['accepted']}
assert len(cases) == 32 and checkpoint['batch'] == 12000

rows = []
for case in cases:
    meta = case['meta']
    target = torch.tensor(provenance[meta['physical_section_id']]
        ['postartifact_gauge_v1']['post_artifact_state12']).float()[None]
    true_normal = full_frame_state_to_components(target)[1][0, :, 2]
    with torch.inference_mode():
        prediction = model.predict(case['input'][None])
        scores = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)[0]
        branches = torch.arange(len(scores))
        modes, flips = branches // 2, branches % 2
        states = prediction['state'][0, modes]
        normals = full_frame_state_to_components(states)[1][:, :, 2]
        feasible = (normals @ true_normal).abs() >= math.cos(math.radians(30))
        chart = model.canonical_chart(case['chart'][None].expand(len(scores), -1, -1),
                                      modes // model.normal_modes, 192)
        sites = rigid_points_090(states, flips, chart, (192, 192))
        error = (sites - case['truth'][None]).norm(dim=-1).mean(-1) / 1000
    first = int(scores.argmax())
    selected = int(scores.masked_fill(~feasible, -torch.inf).argmax()) if feasible.any() else None
    old = prior[meta['physical_section_id']]
    assert abs(float(error[first]) - old['first_gap_mm']) < .001
    rows.append({'physical_section_id': meta['physical_section_id'], 'base': meta['base'],
        'top16_feasible': bool(old['arms']['perfect']['feasible_count']),
        'full_feasible_count': int(feasible.sum()),
        'first_gap_mm': float(error[first]),
        'full_selected_gap_mm': None if selected is None else float(error[selected]),
        'full_selected_rank_by_prior': None if selected is None else
            int((scores > scores[selected]).sum()),
        'full_feasible_oracle_mm': None if selected is None else float(error[feasible].min())})

rescued = [row for row in rows if not row['top16_feasible'] and row['full_feasible_count']]
usable = [row for row in rows if row['full_feasible_count']]
summary = {'n': len(rows), 'top16_infeasible': sum(not row['top16_feasible'] for row in rows),
    'rescued_by_full_bank': len(rescued), 'full_bank_feasible': len(usable),
    'all_case_unconstrained_mean_mm': float(np.mean([row['first_gap_mm'] for row in rows])),
    'feasible_paired_unconstrained_mean_mm': float(np.mean([row['first_gap_mm'] for row in usable]))
        if usable else None,
    'full_bank_selected_mean_mm': float(np.mean([row['full_selected_gap_mm'] for row in usable]))
        if usable else None,
    'full_bank_selected_over_5mm': sum(row['full_selected_gap_mm'] > 5 for row in usable),
    'full_bank_feasible_oracle_mean_mm': float(np.mean([row['full_feasible_oracle_mm']
                                                      for row in usable])) if usable else None,
    'exploratory_gate': bool(len(rescued) >= 8 and len(usable) == 32 and
        np.mean([row['full_selected_gap_mm'] for row in usable]) < 3.0),
    'model_promoted': False, 'calibrated': False}
OUT.mkdir(parents=True, exist_ok=False)
config = {'script_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(RUN / 'joint_12000.pt'),
    'panel_sha256': sha(RUN / 'panels.pt'), 'draws_sha256': sha(RUN / 'panel_draws.jsonl'),
    'prior_rows_sha256': sha(PRIOR / 'rows.jsonl'), 'cone_deg': 30,
    'scope': 'exploratory TRAIN sparse fixed-panel mechanism; perfect normal is privileged'}
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
(OUT / 'completed.json').write_text(json.dumps({'config_sha256': sha(OUT / 'config.json'),
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'n': len(rows)}, indent=2) + '\n')
print(json.dumps(summary, allow_nan=False), flush=True)
