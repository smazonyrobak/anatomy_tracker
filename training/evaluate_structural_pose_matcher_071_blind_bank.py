"""Frozen 071 score on the 069 blind bank; 067a geometry is a truth-only diagnostic."""
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
    full_frame_state_from_components, full_frame_state_to_components,
    render_finite_thickness_plane)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_structural_matcher_071 import StructuralPoseMatcher

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
checkpoint = root / 'runs/structural_pose_matcher_071_phase1/batch_03000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
blind = root / 'runs/mind_image_search_069_pilot_001'
coverage = root / 'runs/atlas_search_067a_coverage_audit'
out = root / 'runs/structural_pose_matcher_071_blind_bank'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate_state(centre, canonical_edge, normal, reflected, data):
    flip = canonical_edge.new_tensor([[-1., 0.], [0., 1.]]) if reflected else torch.eye(2, device='cuda')
    observed_edge = canonical_edge @ flip
    angle = canonical_edge.new_tensor(data['angle_rad'])
    co, si = angle.cos(), angle.sin()
    rotation = torch.stack((torch.stack((co, -si)), torch.stack((si, co))))
    transformed_edge = observed_edge @ (data['scale'] * rotation)
    new_canonical_edge = transformed_edge @ flip
    centre = centre + normal * data['normal_shift_um'] + observed_edge @ canonical_edge.new_tensor(data['shift_xy'])
    reflected_offset = centre.new_tensor([-1 / 256, 0.]) if reflected else centre.new_zeros(2)
    centre -= new_canonical_edge @ reflected_offset
    u, v = new_canonical_edge.unbind(-1)
    origin = centre - .5 * (u + v)
    return full_frame_state_from_components(*physical_ouv_to_frame(torch.stack((origin, u, v))))


records = {row['section_id']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
blind_rows = {row['section_id']: row for row in map(json.loads, (blind / 'rows.jsonl').open())}
coverage_rows = list(map(json.loads, (coverage / 'rows.jsonl').open()))
assert len(blind_rows) == len(coverage_rows) == 32
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
pose.requires_grad_(False)
matcher = StructuralPoseMatcher().cuda().eval()
frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert frozen['batch'] == 3000 and not frozen['calibrated']
matcher.load_state_dict(frozen['model'])
matcher.requires_grad_(False)
psf = torch.tensor([-50, -25, 0, 25, 50], device='cuda').float()
psf_weights = torch.tensor([1, 2, 2, 2, 1], device='cuda').float() / 8


def render(states, flags):
    slab = render_finite_thickness_plane(atlas, states, (64, 64),
        (0., 0., 0.), (25., 25., 25.), psf, psf_weights)
    slab = torch.where(flags[:, None, None, None].bool(), slab.flip(-1), slab)
    support = slab[:, 1:2].clamp(0, 1)
    return torch.cat((slab[:, :1] / support.clamp_min(1e-4) * support,
                      support), 1)[None]


rows = []
with torch.inference_mode():
    for frozen_row in coverage_rows:
        record = records[frozen_row['section_id']]
        blind_row = blind_rows[frozen_row['section_id']]
        path = panel / record['file']
        assert sha(path) == record['sha256'] == frozen_row['sha256'] == blind_row['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        prediction = pose.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        assert choice.tolist() == [item['choice'] for item in blind_row['candidate_results']]
        states = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centres, frames, bases = full_frame_state_to_components(states)
        edges = frames[0, :, :, :2] @ bases[0]
        normals = frames[0, :, :, 2]
        good, wrong = [], []
        for branch in range(14):
            reflected = bool(int(choice[branch]) % 2)
            good.append(candidate_state(centres[0, branch], edges[branch], normals[branch],
                reflected, frozen_row['fitted'][branch]))
            wrong.append(candidate_state(centres[0, branch], edges[branch], normals[branch],
                reflected, blind_row['candidate_results'][branch]))
        query = F.interpolate(image, (64, 64), mode='area')
        flags = choice % 2
        good_score = matcher(query, render(torch.stack(good), flags))[0]
        wrong_score = matcher(query, render(torch.stack(wrong), flags))[0]
        oracle_errors = [item['error_um'] for item in frozen_row['fitted']]
        best_geometry = int(np.argmin(oracle_errors))
        selected = int(wrong_score.argmax())
        rows.append({key: frozen_row[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'best_geometry_branch': best_geometry,
            'best_geometry_um': oracle_errors[best_geometry],
            'best_geometry_score': float(good_score[best_geometry]),
            'max_blind_score': float(wrong_score.max()),
            'selected_blind_branch': selected,
            'selected_blind_um': blind_row['candidate_results'][selected]['physical_um'],
            'blind_best14_um': min(item['physical_um'] for item in blind_row['candidate_results']),
            'prior_selected_um': blind_row['prior_selected_um'],
            'mind_selected_um': blind_row['blind_selected_um'],
            'good_scores': good_score.tolist(), 'blind_scores': wrong_score.tolist()})

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})


def identity_equal(key, subset):
    ids = {row['synthetic_subject_plan_id'] for row in subset}
    return float(np.mean([np.mean([row[key] for row in subset
        if row['synthetic_subject_plan_id'] == identity]) for identity in ids]))


summary = {'sections': len(rows), 'synthetic_identities': len(identities),
    'truth_only_score_diagnostic': True, 'biological_validation': False,
    'public_benchmark_used': False,
    'best_geometry_outscores_all_blind': sum(row['best_geometry_score'] > row['max_blind_score'] for row in rows),
    'under_1mm_selected': sum(row['selected_blind_um'] < 1000 for row in rows)}
for name in ('selected_blind_um', 'blind_best14_um', 'prior_selected_um', 'mind_selected_um'):
    summary[name] = identity_equal(name, rows)
for mode in ('exact_black', 'imperfect_brush', 'raw'):
    subset = [row for row in rows if row['appearance_mode'] == mode]
    summary[mode] = {'sections': len(subset),
        'selected_blind_um': identity_equal('selected_blind_um', subset),
        'prior_selected_um': identity_equal('prior_selected_um', subset),
        'cross_branch_wins': sum(row['best_geometry_score'] > row['max_blind_score'] for row in subset)}
summary['advance_signal'] = (summary['selected_blind_um'] < summary['prior_selected_um']
    and summary['raw']['selected_blind_um'] <= summary['raw']['prior_selected_um'])

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({'matcher_checkpoint_sha256': sha(checkpoint),
    'parent_pose_checkpoint_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'blind_rows_sha256': sha(blind / 'rows.jsonl'),
    'truth_fitted_rows_sha256': sha(coverage / 'rows.jsonl'),
    'source_sha256': sha(Path(__file__)), 'fixed_psf_um': psf.tolist(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'hashes_sha256': {name: sha(out / name) for name in
        ('config.json', 'rows.jsonl', 'summary.json')},
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
