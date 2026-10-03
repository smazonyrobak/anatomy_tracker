"""Matched finite-thickness atlas-score discrimination on frozen 067 transforms."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
blind = root / 'runs/atlas_image_search_067_pilot_001'
coverage = root / 'runs/atlas_search_067a_coverage_audit'
out = root / 'runs/structural_score_068_development_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = {row['section_id']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
blind_rows = {row['section_id']: row for row in map(json.loads, (blind / 'rows.jsonl').open())}
fitted_rows = list(map(json.loads, (coverage / 'rows.jsonl').open()))
assert len(blind_rows) == len(fitted_rows) == 32
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
axis = (torch.arange(48, device='cuda').float() + .5) / 48 - .5
qy, qx = torch.meshgrid(axis, axis, indexing='ij')
query_chart = torch.stack((qx, qy), -1)
psf = torch.tensor([-50, -25, 0, 25, 50], device='cuda').float()
psf_weights = torch.tensor([1, 2, 2, 2, 1], device='cuda').float() / 8
offsets = ((2, 0), (-2, 0), (0, 2), (0, -2),
           (2, 2), (2, -2), (-2, 2), (-2, -2))


def render(centre, edge, normal, data):
    angle = torch.tensor(data['angle_rad'], device='cuda')
    scale = data['scale']
    co, si = angle.cos(), angle.sin()
    x, y = query_chart.unbind(-1)
    chart = torch.stack((data['shift_xy'][0] + scale * (co * x - si * y),
                         data['shift_xy'][1] + scale * (si * x + co * y)), -1)
    physical = centre + torch.einsum('ij,hwj->hwi', edge, chart)
    physical = physical[None, None] + normal * (
        data['normal_shift_um'] + psf[None, :, None, None, None])
    raw = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    support = raw[:, 1:2].clamp(0, 1)
    return raw[:, :1] / support.clamp_min(1e-4) * support, support


def mind(image):
    padded = F.pad(image, (2, 2, 2, 2), mode='replicate')
    distance = []
    for dy, dx in offsets:
        shifted = padded[:, :, 2 + dy:50 + dy, 2 + dx:50 + dx]
        distance.append(F.avg_pool2d((image - shifted).square(), 3, 1, 1))
    distance = torch.cat(distance, 1)
    variance = distance.mean(1, keepdim=True).clamp_min(1e-4)
    return torch.exp(-distance / variance)


def metrics(query, atlas_image, support, query_mind):
    weight = support.flatten().clamp(0, 1)
    qhp = query - F.avg_pool2d(query, 7, 1, 3)
    ahp = atlas_image - F.avg_pool2d(atlas_image, 7, 1, 3)
    q, a = qhp.flatten(), ahp.flatten()
    mass = weight.sum().clamp_min(1e-5)
    qm, am = (weight * q).sum() / mass, (weight * a).sum() / mass
    qq, aa = q - qm, a - am
    ncc = (weight * qq * aa).sum().abs() / ((weight * qq.square()).sum() *
        (weight * aa.square()).sum()).clamp_min(1e-9).sqrt()
    ncc *= (mass / (48 * 48 * .2)).clamp(max=1)

    smooth = lambda x: F.avg_pool2d(x, 7, 1, 3)
    local_mass = smooth(support).clamp_min(1e-5)
    qmean = smooth(support * query) / local_mass
    amean = smooth(support * atlas_image) / local_mass
    qvar = (smooth(support * query.square()) / local_mass - qmean.square()).clamp_min(0)
    avar = (smooth(support * atlas_image.square()) / local_mass - amean.square()).clamp_min(0)
    cov = smooth(support * query * atlas_image) / local_mass - qmean * amean
    local = cov / (qvar * avar + 1e-7).sqrt()
    valid = ((local_mass > .35) & (qvar > 1e-4) & (avar > 1e-4)).float()
    lncc = (local * valid).sum().abs() / valid.sum().clamp_min(1)
    lncc *= (valid.mean() / .15).clamp(max=1)

    atlas_mind = mind(atlas_image)
    interior = (F.avg_pool2d(support, 5, 1, 2) > .8).float()
    mind_score = (-((query_mind - atlas_mind).square() * interior).sum() /
        (interior.sum() * len(offsets)).clamp_min(1))
    return {'ncc': float(ncc), 'lncc': float(lncc), 'mind': float(mind_score)}


rows = []
with torch.inference_mode():
    for frozen in fitted_rows:
        record = records[frozen['section_id']]
        blind_row = blind_rows[frozen['section_id']]
        path = panel / record['file']
        assert sha(path) == record['sha256'] == frozen['sha256'] == blind_row['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        assert choice.tolist() == [r['choice'] for r in blind_row['candidate_results']]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        edge = frame[0, :, :, :2] @ basis[0]
        edge[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
        normal = frame[0, :, :, 2]
        query = F.interpolate(image[:, :1], (48, 48), mode='area')
        query_mind = mind(query)
        candidate_scores = []
        for branch in range(14):
            oracle = frozen['fitted'][branch]
            wrong = blind_row['candidate_results'][branch]
            good_image, good_support = render(centre[0, branch], edge[branch], normal[branch], oracle)
            wrong_image, wrong_support = render(centre[0, branch], edge[branch], normal[branch], wrong)
            candidate_scores.append({'branch': branch, 'oracle_error_um': oracle['error_um'],
                'blind_error_um': wrong['physical_um'],
                'oracle': metrics(query, good_image, good_support, query_mind),
                'blind': metrics(query, wrong_image, wrong_support, query_mind)})
        best_geom = min(candidate_scores, key=lambda c: c['oracle_error_um'])
        rows.append({key: frozen[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'best_geometry_branch': best_geom['branch'],
            'best_geometry_um': best_geom['oracle_error_um'],
            'scores': candidate_scores})

summary = {'sections': len(rows), 'synthetic_identities': len({r['synthetic_subject_plan_id'] for r in rows}),
    'truth_only_score_diagnostic': True, 'real_animal_validation': False,
    'public_benchmark_used': False}
for name in ('ncc', 'lncc', 'mind'):
    within = cross = 0
    for row in rows:
        best = row['scores'][row['best_geometry_branch']]
        good = best['oracle'][name]
        within += good > best['blind'][name]
        cross += good > max(candidate['blind'][name] for candidate in row['scores'])
    summary[name] = {'best_geometry_outscores_blind_same_branch': within,
        'best_geometry_outscores_any_blind': cross,
        'by_appearance': {mode: {'sections': sum(r['appearance_mode'] == mode for r in rows),
            'cross_branch_wins': sum(r['scores'][r['best_geometry_branch']]['oracle'][name] >
                max(c['blind'][name] for c in r['scores']) for r in rows
                if r['appearance_mode'] == mode)}
            for mode in ('exact_black', 'imperfect_brush', 'raw')}}
summary['advance_signal'] = any(summary[name]['best_geometry_outscores_any_blind'] >= 16
    and summary[name]['by_appearance']['raw']['cross_branch_wins'] >= 4
    for name in ('lncc', 'mind'))

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'blind_rows_sha256': sha(blind / 'rows.jsonl'),
    'truth_fitted_rows_sha256': sha(coverage / 'rows.jsonl'),
    'source_sha256': sha(Path(__file__)),
    'scores': ['direct-render high-pass NCC', 'local NCC', '8-neighbour MIND-style'],
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows), 'hashes_sha256': {name: sha(out / name)
        for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
