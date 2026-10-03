"""Blind MIND-style atlas-image search on the frozen 067 development subset."""
import hashlib
import json
import math
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
reference = root / 'runs/normal_fixed_066_development_audit/rows.jsonl'
subset = root / 'runs/atlas_image_search_067_pilot_001/config.json'
out = root / 'runs/mind_image_search_069_pilot_001'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


all_records = {row['section_id']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
records = [all_records[key] for key in json.loads(subset.read_text())['selected_section_ids']]
assert len(records) == 32
control = {row['section_id']: row for row in map(json.loads, reference.open())}
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)

side = 160
axis = torch.linspace(-1.5, 2.5, side, device='cuda') - .5
by, bx = torch.meshgrid(axis, axis, indexing='ij')
broad_chart = torch.stack((bx, by), -1)
psf = torch.tensor([-50, -25, 0, 25, 50], device='cuda').float()
psf_weights = torch.tensor([1, 2, 2, 2, 1], device='cuda').float() / 8
normal_shifts = torch.arange(-1500, 1501, 500, device='cuda').float()
ca, cs, cx, cy = torch.meshgrid(
    torch.arange(12, device='cuda').float() * (2 * math.pi / 12),
    torch.tensor([.65, 1., 1.5], device='cuda'),
    torch.linspace(-.75, .75, 7, device='cuda'),
    torch.linspace(-.75, .75, 7, device='cuda'), indexing='ij')
coarse = torch.stack((ca.flatten(), cs.flatten(), cx.flatten(), cy.flatten()), -1)
fa, fs, fx, fy = torch.meshgrid(
    torch.linspace(-math.pi / 12, math.pi / 12, 5, device='cuda'),
    torch.tensor([.8, 1., 1.2], device='cuda'),
    torch.tensor([-.125, 0, .125], device='cuda'),
    torch.tensor([-.125, 0, .125], device='cuda'), indexing='ij')
fine = torch.stack((fa.flatten(), fs.flatten(), fx.flatten(), fy.flatten()), -1)
neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1))


def mind(image, step):
    padded = F.pad(image, (step, step, step, step), mode='replicate')
    size = image.shape[-1]
    distances = []
    for dy, dx in neighbours:
        shifted = padded[:, :, step + dy * step:step + dy * step + size,
                         step + dx * step:step + dx * step + size]
        distances.append(F.avg_pool2d((image - shifted).square(), 3, 1, 1))
    distances = torch.cat(distances, 1)
    return torch.exp(-distances / distances.mean(1, keepdim=True).clamp_min(1e-4))


def mind_score(sampled, query_desc, size):
    atlas_desc = mind(sampled[:, :1], 1 if size == 24 else 2)
    support = sampled[:, 1:2].clamp(0, 1)
    interior = (F.avg_pool2d(support, 3 if size == 24 else 5,
                              1, 1 if size == 24 else 2) > .8).float()
    mass = interior.sum((1, 2, 3))
    distance = (((atlas_desc - query_desc).square() * interior).sum((1, 2, 3)) /
        (mass * 8).clamp_min(1))
    fraction = mass / (size * size)
    score = -distance - .05 * ((.10 - fraction) / .10).clamp(min=0)
    return torch.where(mass >= 5, score, score.new_full(score.shape, -1.))


def atlas_planes(centre, edge, normal, shifts):
    base = centre + torch.einsum('ij,hwj->hwi', edge, broad_chart)
    physical = base[None, None] + normal * (
        shifts[:, None, None, None, None] + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    support = rendered[:, 1:2].clamp(0, 1)
    return torch.cat((rendered[:, :1] / support.clamp_min(1e-4) * support,
                      support), 1)


def search_scores(plane, query_desc, params, size):
    axis = (torch.arange(size, device='cuda').float() + .5) / size - .5
    qy, qx = torch.meshgrid(axis, axis, indexing='ij')
    angle, scale, tx, ty = params.T
    co, si = angle.cos(), angle.sin()
    gx = tx[:, None, None] + scale[:, None, None] * (
        co[:, None, None] * qx - si[:, None, None] * qy)
    gy = ty[:, None, None] + scale[:, None, None] * (
        si[:, None, None] * qx + co[:, None, None] * qy)
    grid = torch.stack((gx, gy), -1) / 2
    sampled = F.grid_sample(plane.expand(len(params), -1, -1, -1), grid,
        mode='bilinear', padding_mode='zeros', align_corners=True)
    return mind_score(sampled, query_desc, size)


def direct_score(centre, edge, normal, param, dz, query_desc):
    axis = (torch.arange(48, device='cuda').float() + .5) / 48 - .5
    qy, qx = torch.meshgrid(axis, axis, indexing='ij')
    angle, scale, tx, ty = param
    co, si = angle.cos(), angle.sin()
    chart = torch.stack((tx + scale * (co * qx - si * qy),
                         ty + scale * (si * qx + co * qy)), -1)
    physical = centre + torch.einsum('ij,hwj->hwi', edge, chart)
    physical = physical[None, None] + normal * (dz + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    support = rendered[:, 1:2].clamp(0, 1)
    sampled = torch.cat((rendered[:, :1] / support.clamp_min(1e-4) * support,
                         support), 1)
    return float(mind_score(sampled, query_desc, 48)[0])


def physical_error(centre, edge, normal, param, dz, xy, target):
    angle, scale, tx, ty = param
    q = xy - .5
    co, si = angle.cos(), angle.sin()
    chart = torch.stack((tx + scale * (co * q[:, 0] - si * q[:, 1]),
                         ty + scale * (si * q[:, 0] + co * q[:, 1])), -1)
    mapped = centre + normal * dz + chart @ edge.T
    return float((mapped - target).norm(dim=-1).mean())


rows = []
with torch.inference_mode():
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centres, frames, bases = full_frame_state_to_components(state)
        edges = frames[0, :, :, :2] @ bases[0]
        edges[:, :, 0] *= torch.where(choice % 2 == 1, -1., 1.)[:, None]
        normals = frames[0, :, :, 2]
        q24 = F.interpolate(image[:, :1], (24, 24), mode='area')
        q48 = F.interpolate(image[:, :1], (48, 48), mode='area')
        query24, query48 = mind(q24, 1), mind(q48, 2)
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')),
                         -1).float() / 256
        target = truth.reshape(-1, 3)[ids]
        candidate_results = []
        for branch in range(14):
            centre, edge, normal = centres[0, branch], edges[branch], normals[branch]
            rendered = atlas_planes(centre, edge, normal, normal_shifts)
            proposals = []
            for axial in range(len(normal_shifts)):
                scores = search_scores(rendered[axial:axial + 1], query24, coarse, 24)
                best = int(scores.argmax())
                proposals.append((float(scores[best]), float(normal_shifts[axial]), coarse[best]))
            proposals.sort(key=lambda p: p[0], reverse=True)
            finalists = []
            for _, base_dz, base_param in proposals[:3]:
                params = torch.stack((base_param[0] + fine[:, 0],
                    base_param[1] * fine[:, 1], base_param[2] + fine[:, 2],
                    base_param[3] + fine[:, 3]), -1)
                for delta in (-250., 0., 250.):
                    dz = base_dz + delta
                    plane = atlas_planes(centre, edge, normal,
                        torch.tensor([dz], device='cuda'))
                    scores = search_scores(plane, query48, params, 48)
                    best = int(scores.argmax())
                    finalists.append((float(scores[best]), dz, params[best]))
            finalists.sort(key=lambda p: p[0], reverse=True)
            direct = [(direct_score(centre, edge, normal, param, dz, query48), dz, param)
                      for _, dz, param in finalists[:3]]
            original = torch.tensor([0., 1., 0., 0.], device='cuda')
            direct.append((direct_score(centre, edge, normal, original, 0., query48),
                           0., original))
            score, dz, param = max(direct, key=lambda p: p[0])
            candidate_results.append({'branch': branch, 'choice': int(choice[branch]),
                'score': score, 'normal_shift_um': dz,
                'angle_rad': float(param[0]), 'scale': float(param[1]),
                'shift_xy': [float(param[2]), float(param[3])],
                'physical_um': physical_error(centre, edge, normal, param, dz, xy, target)})
        selected = max(candidate_results, key=lambda r: r['score'])
        baseline = control[record['section_id']]
        rows.append({key: record[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'prior_selected_um': baseline['selected_rigid_um'],
            'prior_best14_um': baseline['best14_rigid_um'],
            'blind_selected_um': selected['physical_um'],
            'blind_best14_um': min(r['physical_um'] for r in candidate_results),
            'selected_branch': selected['branch'], 'candidate_results': candidate_results})
        if number % 8 == 0:
            print(json.dumps({'cases': number, 'total': len(records)}), flush=True)

identities = sorted({r['synthetic_subject_plan_id'] for r in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
           'uses_synthetic_truth_to_select': False, 'real_animal_validation': False,
           'public_benchmark_used': False}
for key in ('prior_selected_um', 'prior_best14_um', 'blind_selected_um', 'blind_best14_um'):
    summary[key] = float(np.mean([np.mean([r[key] for r in rows
        if r['synthetic_subject_plan_id'] == identity]) for identity in identities]))
for mode in ('exact_black', 'imperfect_brush', 'raw'):
    subset_rows = [r for r in rows if r['appearance_mode'] == mode]
    summary[mode] = {'sections': len(subset_rows),
        'prior_selected_um': float(np.mean([r['prior_selected_um'] for r in subset_rows])),
        'blind_selected_um': float(np.mean([r['blind_selected_um'] for r in subset_rows]))}
summary['advance_signal'] = (summary['prior_selected_um'] - summary['blind_selected_um'] >= 300
    and summary['blind_best14_um'] <= 1000
    and summary['raw']['blind_selected_um'] < 4000)

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'reference_rows_sha256': sha(reference), 'subset_sha256': sha(subset),
    'source_sha256': sha(Path(__file__)),
    'selected_section_ids': [r['section_id'] for r in records],
    'coarse_angles': 12, 'coarse_scales': [.65, 1., 1.5],
    'coarse_shifts_chart': [-.75, .75, 7],
    'normal_shifts_um': [-1500, 1500, 500],
    'psf_offsets_um': psf.tolist(), 'psf_weights': psf_weights.tolist(),
    'score': '8-neighbour MIND-style SSD over atlas support with empty-support penalty',
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
