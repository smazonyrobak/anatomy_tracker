"""Truth-only bound for fitting an in-plane frame while holding proposed normals fixed."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
reference = root / 'runs/pose_factors_063_development_audit/rows.jsonl'
out = root / 'runs/normal_fixed_066_development_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
prior_rows = {row['section_id']: row for row in map(json.loads, reference.open())}
assert len(records) == len(prior_rows) == 246
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
rows = []

with torch.inference_mode():
    for record in records:
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        normal = frame[0, :, :, 2]
        edge = frame[..., :, :2] @ basis
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(256), ids.div(256, rounding_mode='floor')),
                         -1).float() / 256
        tissue = target.reshape(-1, 3)[ids]
        chart = xy[None, None].expand(1, 14, -1, -1).clone()
        chart[..., 0] = torch.where((choice % 2)[..., None].bool(),
                                    255 / 256 - chart[..., 0], chart[..., 0])
        predicted = centre[..., None, :] + torch.einsum(
            'bkij,bkpj->bkpi', edge, chart - .5)
        rigid = (predicted[0] - tissue[None]).norm(dim=-1).mean(-1)
        assert abs(float(rigid.min()) - prior_rows[record['section_id']]['best_physical_um']) < .01
        selected = int(prior.gather(1, choice)[0].argmax())
        assert abs(float(rigid[selected]) - prior_rows[record['section_id']]['selected_physical_um']) < .01
        design = torch.cat((torch.ones((len(xy), 1), device='cuda'), xy - xy.mean(0)), -1)
        coefficient = torch.linalg.solve(design.T @ design, design.T @ tissue)
        fitted_edge = coefficient[None, 1:] - torch.einsum(
            'kij,kj->ki', coefficient[None, 1:].expand(14, -1, -1), normal
        )[:, :, None] * normal[:, None]
        fitted = coefficient[0][None, None] + torch.einsum(
            'pi,kij->kpj', design[:, 1:], fitted_edge)
        fit_error = (fitted - tissue[None]).norm(dim=-1).mean(-1)
        normal_bound = ((tissue - tissue.mean(0)) @ normal.T).abs().mean(0)
        rows.append({key: record[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id',
            'appearance_mode', 'sha256')})
        rows[-1].update({'visible_fraction': float(valid.float().mean()),
            'selected_rigid_um': float(rigid[selected]),
            'best14_rigid_um': float(rigid.min()),
            'selected_normal_fixed_um': float(fit_error[selected]),
            'best14_normal_fixed_um': float(fit_error.min()),
            'best14_normal_only_lower_bound_um': float(normal_bound.min()),
            'best_normal_fixed_branch': int(fit_error.argmin())})

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
           'fit_uses_synthetic_truth': True, 'real_animal_validation': False,
           'public_benchmark_used': False}
for key in ('selected_rigid_um', 'best14_rigid_um', 'selected_normal_fixed_um',
            'best14_normal_fixed_um', 'best14_normal_only_lower_bound_um'):
    summary[key] = float(np.mean([np.mean([row[key] for row in rows
        if row['synthetic_subject_plan_id'] == identity]) for identity in identities]))
summary['in_plane_search_geometric_headroom'] = summary['best14_normal_fixed_um'] <= 750
for mode in sorted({row['appearance_mode'] for row in rows}):
    subset = [row for row in rows if row['appearance_mode'] == mode]
    summary[mode] = {'sections': len(subset), 'best14_normal_fixed_um': float(np.mean([
        np.mean([row['best14_normal_fixed_um'] for row in subset
                 if row['synthetic_subject_plan_id'] == identity])
        for identity in identities]))}

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'reference_rows_sha256': sha(reference), 'source_sha256': sha(Path(__file__)),
    'candidate_policy': '059 old top8 plus anchor top6 with reflection-expanded prior',
    'normal_only_bound': 'mean absolute centred tissue displacement on proposed normal',
    'normal_fixed_fit': 'truth-assisted least-squares affine with both edges perpendicular to normal',
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
