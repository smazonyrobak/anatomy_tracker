"""Truth-only similarity-fit ceiling for normal-fixed atlas matching."""
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
reference = root / 'runs/normal_fixed_066_development_audit/rows.jsonl'
out = root / 'runs/normal_fixed_similarity_066b_development_audit'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
control = {row['section_id']: row for row in map(json.loads, reference.open())}
assert len(records) == len(control) == 246
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
            tissue = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        _, frame, _ = full_frame_state_to_components(state)
        xy_indices = valid.flatten().nonzero().flatten()
        xy = torch.stack((xy_indices.remainder(256), xy_indices.div(256, rounding_mode='floor')),
                         -1).float() / 256
        x = xy - xy.mean(0)
        target = tissue.reshape(-1, 3)[xy_indices]
        target_mean = target.mean(0)
        y = (target - target_mean) @ frame[0, :, :, :2]
        cross = torch.einsum('ni,knj->kij', x, y)
        u, singular, vh = torch.linalg.svd(cross)
        rotation = u @ vh
        scale = singular.sum(-1) / x.square().sum()
        fitted_y = torch.einsum('ni,kij->knj', x, rotation) * scale[:, None, None]
        fitted = target_mean + torch.einsum('knj,kij->kni', fitted_y, frame[0, :, :, :2])
        error = (fitted - target[None]).norm(dim=-1).mean(-1)
        old = control[record['section_id']]
        assert float(error.min()) + .01 >= old['best14_normal_only_lower_bound_um']
        selected = int(prior.gather(1, choice)[0].argmax())
        rows.append({key: record[key] for key in ('animal_id', 'specimen_id',
            'experiment_id', 'section_id', 'synthetic_subject_plan_id', 'appearance_mode', 'sha256')})
        rows[-1].update({'selected_similarity_um': float(error[selected]),
            'best14_similarity_um': float(error.min()),
            'best14_affine_um': old['best14_normal_fixed_um'],
            'best14_normal_only_lower_bound_um': old['best14_normal_only_lower_bound_um']})

identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(identities),
           'truth_fit_unavailable_at_inference': True, 'real_animal_validation': False,
           'public_benchmark_used': False}
for key in ('selected_similarity_um', 'best14_similarity_um', 'best14_affine_um',
            'best14_normal_only_lower_bound_um'):
    summary[key] = float(np.mean([np.mean([row[key] for row in rows
        if row['synthetic_subject_plan_id'] == identity]) for identity in identities]))
summary['similarity_search_geometric_headroom'] = summary['best14_similarity_um'] <= 750
for mode in sorted({row['appearance_mode'] for row in rows}):
    subset = [row for row in rows if row['appearance_mode'] == mode]
    summary[mode] = {'sections': len(subset), 'best14_similarity_um': float(np.mean([
        np.mean([row['best14_similarity_um'] for row in subset
                 if row['synthetic_subject_plan_id'] == identity])
        for identity in identities]))}

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'parent_sha256': sha(parent), 'panel_records_sha256': sha(panel / 'records.jsonl'),
    'reference_rows_sha256': sha(reference), 'source_sha256': sha(Path(__file__)),
    'fit': 'truth-assisted 2D isotropic scale, rotation/reflection and translation; normal fixed',
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
