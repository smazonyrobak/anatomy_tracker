"""CPU-only, frozen C4 internal-boundary information diagnostic."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift, sobel

from training.arbitrary_plane_c4_joint_model import C4JointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090

ROOT = Path('I:/AnatomyTracker')
WORK = ROOT / 'agent_worktrees/joint_v6_integration'
RUN = ROOT / 'runs/c4_joint_321_002'
OUT = ROOT / 'runs/c4_internal_boundary_322_002'
PROTOCOL = WORK / 'docs/publication/C4_INTERNAL_BOUNDARY_322_PROTOCOL_20261015.md'
ANNOTATION = ROOT / 'data/Allen Brain Atlas 25um/annotation_25.nrrd'
ONTOLOGY = ROOT / 'data/joint_v7_probe_observation_cpu_check_001/allen_structure_graph_1.json'
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boundary_score(boundary, distance, valid):
    count = int(boundary.sum())
    return max(float((shift(boundary, (dy, dx), order=0, mode='constant', cval=0)
                      .astype(bool) & valid & (distance <= 2)).sum() / count)
               for dy in range(-2, 3) for dx in range(-2, 3))


checkpoint = torch.load(RUN / 'joint_12000.pt', map_location='cpu', weights_only=True)
model = C4JointSliceModel(side=192).eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
panels = torch.load(RUN / 'panels.pt', map_location='cpu', weights_only=True)
cases = panels['inner']
frozen = json.loads((RUN / 'readouts.jsonl').read_text().splitlines()[-1])
reference = [row for row in frozen['rows'] if row['cohort'] == 'inner']
comparator = json.loads((RUN / 'frozen_309_comparator.json').read_text())
assert len(cases) == len(reference) == 128 and checkpoint['batch'] == frozen['batch'] == 12000

roots = [315, 698, 1089, 703, 477, 803, 549, 1097, 313, 771, 354, 512, 1009, 73]
tree = json.loads(ONTOLOGY.read_text())['msg'][0]
classes_by_id = {0: 0}
stack = [(tree, 0)]
while stack:
    node, parent = stack.pop()
    cls = roots.index(node['id']) + 1 if node['id'] in roots else parent
    classes_by_id[node['id']] = cls
    stack.extend((child, cls) for child in node['children'])
annotation = nrrd.read(str(ANNOTATION), index_order='F')[0]
ids = np.unique(annotation)
classes = np.array([classes_by_id.get(int(k), 0) for k in ids], np.uint8)

axis = (4 * np.arange(48) + 1.5) / 192
yy, xx = np.meshgrid(axis, axis, indexing='ij')
grid = torch.from_numpy(np.stack((xx, yy), -1).reshape(1, -1, 2)).float()
rows = []
for i, case in enumerate(cases):
    with torch.inference_mode():
        prediction = model.predict(case['input'][None])
        logits = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)[0]
        branches = logits.topk(16).indices
        modes, flips = branches // 2, branches % 2
        states = prediction['state'][0, modes]
        coarse = model.canonical_chart(grid.expand(16, -1, -1), modes // 64, 192)
        sites = model.canonical_chart(case['chart'][None].expand(16, -1, -1),
                                      modes // 64, 192)
        atlas_points = rigid_points_090(states, flips, coarse, (192, 192)).numpy()
        site_points = rigid_points_090(states, flips, sites, (192, 192)).numpy()
    errors = np.linalg.norm(site_points - case['truth'].numpy()[None], axis=-1).mean(-1) / 1000
    assert abs(float(errors[0]) - reference[i]['first_rigid_gap_mm']) < 0.001
    voxel = np.floor(atlas_points.reshape(-1, 3) / 25).astype(np.int64)
    inside = ((voxel >= 0) & (voxel < np.array(annotation.shape))).all(1)
    raw = np.zeros(len(voxel), np.int64)
    raw[inside] = annotation[tuple(voxel[inside].T)]
    labels = classes[np.searchsorted(ids, raw)].reshape(16, 48, 48)
    valid = case['valid'].numpy().reshape(48, 4, 48, 4).mean((1, 3)) >= .9
    image = case['input'][0].numpy().reshape(48, 4, 48, 4).mean((1, 3))
    image = gaussian_filter(image, 1)
    gradient = np.hypot(sobel(image, axis=0), sobel(image, axis=1))
    source_edge = valid & (gradient >= np.quantile(gradient[valid], .8))
    distance = distance_transform_edt(~source_edge)
    candidates = []
    for rank, label in enumerate(labels):
        support = label > 0
        boundary = np.zeros((48, 48), bool)
        horizontal = (label[:, :-1] != label[:, 1:]) & support[:, :-1] & support[:, 1:]
        vertical = (label[:-1] != label[1:]) & support[:-1] & support[1:]
        boundary[:, :-1] |= horizontal
        boundary[:, 1:] |= horizontal
        boundary[:-1] |= vertical
        boundary[1:] |= vertical
        boundary &= valid
        count = int(boundary.sum())
        available = count >= 24 and int(source_edge.sum()) >= 24
        evidence = null = None
        if available:
            evidence = boundary_score(boundary, distance, valid)
            positions = np.flatnonzero(support & valid)
            shuffled = np.zeros(48 * 48, bool)
            rng = np.random.default_rng(20261015322 + i * 16 + rank)
            shuffled[rng.choice(positions, count, replace=False)] = True
            null = boundary_score(shuffled.reshape(48, 48), distance, valid)
        overlap = int((support & valid).sum())
        dice = 2 * overlap / (int(support.sum()) + int(valid.sum()))
        candidates.append({'branch': int(branches[rank]), 'prior': float(logits[branches[rank]]),
            'rigid_gap_mm': float(errors[rank]), 'atlas_support_fraction': float(support.mean()),
            'internal_boundary_cells': count, 'boundary_evidence': evidence,
            'support_dice': dice, 'shuffled_boundary_evidence': null})
    available = [j for j, item in enumerate(candidates) if item['boundary_evidence'] is not None]
    rows.append({'case': i, 'base': case['meta']['base'],
        'physical_section_id': case['meta']['physical_section_id'],
        'source_edge_cells': int(source_edge.sum()), 'available_count': len(available),
        'oracle_rank': int(errors.argmin()), 'candidates': candidates})
    if i == 15:
        enough = sum(row['available_count'] >= 4 for row in rows)
        oracle_available = sum(row['candidates'][row['oracle_rank']]['boundary_evidence']
                               is not None for row in rows)
        observable = enough >= 12 and oracle_available >= 8
        if not observable:
            break

OUT.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(Path(__file__)),
    'checkpoint_sha256': sha(RUN / 'joint_12000.pt'), 'panel_sha256': sha(RUN / 'panels.pt'),
    'readouts_sha256': sha(RUN / 'readouts.jsonl'),
    'comparator_sha256': sha(RUN / 'frozen_309_comparator.json'),
    'annotation_sha256': sha(ANNOTATION), 'ontology_sha256': sha(ONTOLOGY),
    'device': 'cpu', 'n_scored': len(rows)}
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
preflight = {'cases_with_four_available': sum(row['available_count'] >= 4 for row in rows[:16]),
    'cases_with_oracle_available': sum(row['candidates'][row['oracle_rank']]['boundary_evidence']
                                       is not None for row in rows[:16])}
if len(rows) == 16:
    summary = {'status': 'stopped_unobservable', 'preflight': preflight}
else:
    decisions = []
    pairs = []
    for row in rows:
        candidates = row['candidates']
        available = [j for j, item in enumerate(candidates) if item['boundary_evidence'] is not None]
        choices = {'prior': 0}
        for key, field in [('boundary', 'boundary_evidence'), ('support', 'support_dice'),
                           ('shuffled', 'shuffled_boundary_evidence')]:
            choices[key] = max(available, key=lambda j: candidates[j][field]) if available else 0
        decisions.append({'case': row['case'], 'base': row['base'],
                          **{key + '_gap_mm': candidates[j]['rigid_gap_mm']
                             for key, j in choices.items()}})
        if available:
            near = min(available, key=lambda j: candidates[j]['rigid_gap_mm'])
            anchor = candidates[near]
            if anchor['rigid_gap_mm'] <= 1.5:
                far = next((j for j in available if candidates[j]['rigid_gap_mm'] >= 2
                            and candidates[j]['rigid_gap_mm'] >= anchor['rigid_gap_mm'] + 1
                            and abs(candidates[j]['atlas_support_fraction'] -
                                    anchor['atlas_support_fraction']) <= .10), None)
                if far is not None:
                    pairs.append({key: int(anchor[field] > candidates[far][field])
                                  for key, field in [('boundary', 'boundary_evidence'),
                                      ('support', 'support_dice'),
                                      ('shuffled', 'shuffled_boundary_evidence')]})
    means = {key: float(np.mean([item[key + '_gap_mm'] for item in decisions]))
             for key in ('prior', 'boundary', 'support', 'shuffled')}
    tails = {key: sum(item[key + '_gap_mm'] > 5 for item in decisions)
             for key in ('prior', 'boundary', 'support', 'shuffled')}
    wins = {key: float(np.mean([item[key] for item in pairs])) if pairs else None
            for key in ('boundary', 'support', 'shuffled')}
    base_nonworse = int(sum(np.mean([item['boundary_gap_mm'] for item in decisions if item['base'] == base])
                        <= np.mean([item['prior_gap_mm'] for item in decisions if item['base'] == base])
                        for base in range(56, 64)))
    old = comparator['metrics']['inner']['mean_first_rigid_gap_mm']
    gates = {'at_least_40_pairs': len(pairs) >= 40,
        'boundary_pair_win_70pct': wins['boundary'] is not None and wins['boundary'] >= .70,
        'boundary_pair_advantage_10pp': wins['boundary'] is not None and
            wins['boundary'] >= max(wins['support'], wins['shuffled']) + .10,
        'gain_vs_321_0p20mm': means['boundary'] <= means['prior'] - .20,
        'gain_vs_309_0p15mm': means['boundary'] <= old - .15,
        'gain_vs_nulls_0p15mm': means['boundary'] <= min(means['support'], means['shuffled']) - .15,
        'tail_no_worse_than_309': tails['boundary'] <= comparator['metrics']['inner']['first_rigid_over_5mm'],
        'six_bases_nonworse': base_nonworse >= 6}
    summary = {'status': 'completed', 'preflight': preflight, 'mean_rigid_gap_mm': means,
        'first_over_5mm': tails, 'frozen309_mean_mm': old,
        'frozen309_over_5mm': comparator['metrics']['inner']['first_rigid_over_5mm'],
        'top16_oracle_mean_mm': float(np.mean([min(c['rigid_gap_mm'] for c in row['candidates'])
                                              for row in rows])),
        'support_matched_pairs': len(pairs), 'near_far_wins': wins,
        'base_nonworse': base_nonworse, 'gate_clauses': gates, 'gate_pass': all(gates.values())}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
(OUT / 'completed.json').write_text(json.dumps({'config_sha256': sha(OUT / 'config.json'),
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'status': summary['status']}, indent=2) + '\n')
print(json.dumps(summary, allow_nan=False), flush=True)
