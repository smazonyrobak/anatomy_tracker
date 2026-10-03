"""Fixed-pair 059 atlas-matcher discrimination on 069 candidate poses."""
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
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
blind = root / 'runs/mind_image_search_069_pilot_001'
coverage = root / 'runs/atlas_search_067a_coverage_audit'
out = root / 'runs/frozen_atlas_matcher_070_development_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = {row['section_id']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
blind_rows = {row['section_id']: row for row in map(json.loads, (blind / 'rows.jsonl').open())}
coverage_rows = list(map(json.loads, (coverage / 'rows.jsonl').open()))
assert len(blind_rows) == len(coverage_rows) == 32
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
psf = torch.tensor([-50, -25, 0, 25, 50], device='cuda').float()
weights = torch.tensor([1, 2, 2, 2, 1], device='cuda').float() / 8


def candidate_state(centre, canonical_edge, normal, reflected, data):
    flip = canonical_edge.new_tensor([[-1., 0.], [0., 1.]]) if reflected else torch.eye(2, device='cuda')
    observed_edge = canonical_edge @ flip
    angle = torch.tensor(data['angle_rad'], device='cuda')
    co, si = angle.cos(), angle.sin()
    rotate = torch.stack((torch.stack((co, -si)), torch.stack((si, co))))
    transformed_edge = observed_edge @ (data['scale'] * rotate)
    new_canonical_edge = transformed_edge @ flip
    shift = torch.tensor(data['shift_xy'], device='cuda')
    physical_centre = centre + normal * data['normal_shift_um'] + observed_edge @ shift
    reflection_offset = centre.new_tensor([-1 / 256, 0.]) if reflected else centre.new_zeros(2)
    new_centre = physical_centre - new_canonical_edge @ reflection_offset
    u, v = new_canonical_edge.unbind(-1)
    origin = new_centre - .5 * (u + v)
    return full_frame_state_from_components(*physical_ouv_to_frame(torch.stack((origin, u, v))))


rows = []
with torch.inference_mode():
    for frozen in coverage_rows:
        record = records[frozen['section_id']]
        blind_row = blind_rows[frozen['section_id']]
        path = panel / record['file']
        assert sha(path) == record['sha256'] == frozen['sha256'] == blind_row['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
        model.modes = 80
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)[0]
        assert choice.tolist() == [r['choice'] for r in blind_row['candidate_results']]
        state = prediction['state'].gather(1, (choice[None] // 2)[..., None].expand(-1, -1, 12))
        centres, frames, bases = full_frame_state_to_components(state)
        edges = frames[0, :, :, :2] @ bases[0]
        normals = frames[0, :, :, 2]
        good_states, bad_states = [], []
        for branch in range(14):
            reflected = bool(int(choice[branch]) % 2)
            good_states.append(candidate_state(centres[0, branch], edges[branch],
                normals[branch], reflected, frozen['fitted'][branch]))
            bad_states.append(candidate_state(centres[0, branch], edges[branch],
                normals[branch], reflected, blind_row['candidate_results'][branch]))
        model.modes = 14
        neutral = {'feature': prediction['feature'],
            'log_mass': torch.zeros((1, 14), device='cuda'),
            'reflection_logit': torch.zeros((1, 14), device='cuda')}
        good_prediction = {**neutral, 'state': torch.stack(good_states)[None]}
        bad_prediction = {**neutral, 'state': torch.stack(bad_states)[None]}
        good_score = model.score_candidates(good_prediction, psf, weights, atlas,
            side=64, chunk=4)[0, choice % 2 + 2 * torch.arange(14, device='cuda')]
        bad_score = model.score_candidates(bad_prediction, psf, weights, atlas,
            side=64, chunk=4)[0, choice % 2 + 2 * torch.arange(14, device='cuda')]
        model.modes = 80
        oracle_errors = [r['error_um'] for r in frozen['fitted']]
        best = int(np.argmin(oracle_errors))
        selected = int(bad_score.argmax())
        rows.append({key: frozen[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'best_geometry_branch': best,
            'best_geometry_um': oracle_errors[best],
            'best_geometry_matcher_score': float(good_score[best]),
            'best_geometry_blind_matcher_score': float(bad_score[best]),
            'highest_blind_matcher_score': float(bad_score.max()),
            'matcher_selected_blind_um': blind_row['candidate_results'][selected]['physical_um'],
            'prior_selected_um': blind_row['prior_selected_um'],
            'scores_good': good_score.tolist(), 'scores_blind': bad_score.tolist()})

identities = sorted({r['synthetic_subject_plan_id'] for r in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
    'truth_only_score_diagnostic': True, 'real_animal_validation': False,
    'public_benchmark_used': False,
    'best_geometry_outscores_blind_same_branch': sum(
        r['best_geometry_matcher_score'] > r['best_geometry_blind_matcher_score'] for r in rows),
    'best_geometry_outscores_any_blind': sum(
        r['best_geometry_matcher_score'] > r['highest_blind_matcher_score'] for r in rows)}
for key in ('prior_selected_um', 'matcher_selected_blind_um'):
    summary[key] = float(np.mean([np.mean([r[key] for r in rows
        if r['synthetic_subject_plan_id'] == identity]) for identity in identities]))
for mode in ('exact_black', 'imperfect_brush', 'raw'):
    subset = [r for r in rows if r['appearance_mode'] == mode]
    summary[mode] = {'sections': len(subset), 'cross_branch_wins': sum(
        r['best_geometry_matcher_score'] > r['highest_blind_matcher_score'] for r in subset)}
summary['advance_signal'] = (summary['best_geometry_outscores_any_blind'] >= 16
    and summary['raw']['cross_branch_wins'] >= 4)

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'blind_rows_sha256': sha(blind / 'rows.jsonl'),
    'truth_fitted_rows_sha256': sha(coverage / 'rows.jsonl'),
    'source_sha256': sha(Path(__file__)),
    'scorer': 'frozen 059 score_candidates with zero log_mass/reflection logits',
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
