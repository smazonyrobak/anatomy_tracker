"""Score truth-fitted atlas transforms with the frozen 067 similarity objective."""
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
out = root / 'runs/atlas_score_067b_development_audit'
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
axis = torch.linspace(-1.5, 2.5, 160, device='cuda') - .5
by, bx = torch.meshgrid(axis, axis, indexing='ij')
broad_chart = torch.stack((bx, by), -1)
psf = torch.tensor([-50, -25, 0, 25, 50], device='cuda').float()
psf_weights = torch.tensor([1, 2, 2, 2, 1], device='cuda').float() / 8
query_axis = (torch.arange(48, device='cuda').float() + .5) / 48 - .5
qy, qx = torch.meshgrid(query_axis, query_axis, indexing='ij')


def score(centre, edge, normal, param, dz, query):
    physical = centre + torch.einsum('ij,hwj->hwi', edge, broad_chart)
    physical = physical[None, None] + normal * (dz + psf[None, :, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, physical, (0., 0., 0.), (25., 25., 25.), psf_weights)
    support = rendered[:, 1:2].clamp(0, 1)
    image = rendered[:, :1] / support.clamp_min(1e-4) * support
    feature = image - F.avg_pool2d(image, 7, 1, 3)
    angle, scale, tx, ty = param
    co, si = angle.cos(), angle.sin()
    grid = torch.stack((tx + scale * (co * qx - si * qy),
                        ty + scale * (si * qx + co * qy)), -1)[None] / 2
    sampled = F.grid_sample(torch.cat((feature, support), 1), grid,
        mode='bilinear', padding_mode='zeros', align_corners=True)
    a = sampled[0, 0].flatten()
    w = sampled[0, 1].flatten().clamp(0, 1)
    q = query.flatten()
    mass = w.sum().clamp_min(1e-5)
    am, qm = (w * a).sum() / mass, (w * q).sum() / mass
    aa, qq = a - am, q - qm
    numerator = (w * aa * qq).sum().abs()
    denominator = ((w * aa.square()).sum() *
                   (w * qq.square()).sum()).clamp_min(1e-9).sqrt()
    return float(numerator / denominator * (mass / (48 * 48 * .2)).clamp(max=1))


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
        query = query - F.avg_pool2d(query, 7, 1, 3)
        candidates = []
        for branch in range(14):
            oracle = frozen['fitted'][branch]
            param = torch.tensor([oracle['angle_rad'], oracle['scale'],
                *oracle['shift_xy']], device='cuda')
            oracle_score = score(centre[0, branch], edge[branch], normal[branch],
                param, oracle['normal_shift_um'], query)
            candidates.append({'branch': branch, 'oracle_error_um': oracle['error_um'],
                'oracle_score': oracle_score,
                'blind_error_um': blind_row['candidate_results'][branch]['physical_um'],
                'blind_score': blind_row['candidate_results'][branch]['score']})
        best_geom = min(candidates, key=lambda c: c['oracle_error_um'])
        rows.append({key: frozen[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'best_geometry_branch': best_geom['branch'],
            'best_geometry_um': best_geom['oracle_error_um'],
            'best_geometry_oracle_score': best_geom['oracle_score'],
            'best_geometry_blind_score': best_geom['blind_score'],
            'best_geometry_blind_um': best_geom['blind_error_um'],
            'highest_oracle_score': max(c['oracle_score'] for c in candidates),
            'highest_blind_score': max(c['blind_score'] for c in candidates),
            'candidates': candidates})

summary = {'sections': len(rows), 'synthetic_identities': len({r['synthetic_subject_plan_id'] for r in rows}),
    'truth_only_score_diagnostic': True, 'real_animal_validation': False,
    'public_benchmark_used': False,
    'best_geometry_oracle_score_mean': float(np.mean([r['best_geometry_oracle_score'] for r in rows])),
    'best_geometry_blind_score_mean': float(np.mean([r['best_geometry_blind_score'] for r in rows])),
    'highest_blind_score_mean': float(np.mean([r['highest_blind_score'] for r in rows])),
    'best_geometry_oracle_outscores_blind_same_branch': sum(
        r['best_geometry_oracle_score'] > r['best_geometry_blind_score'] for r in rows),
    'best_geometry_oracle_outscores_any_blind': sum(
        r['best_geometry_oracle_score'] > r['highest_blind_score'] for r in rows)}

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'blind_rows_sha256': sha(blind / 'rows.jsonl'),
    'truth_fitted_rows_sha256': sha(coverage / 'rows.jsonl'),
    'source_sha256': sha(Path(__file__)),
    'score': 'frozen 067 fine-resolution high-pass support-weighted absolute NCC',
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
