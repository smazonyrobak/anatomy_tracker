"""Truth-only upper bound for recombining actual proposal centres and frames."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/pose_recombination_064_development_audit'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [r for r in map(json.loads, (panel / 'records.jsonl').open()) if r['eligible']]
assert len(records) == 246 and len({r['animal_id'] for r in records}) == 8
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
rows = []

with torch.inference_mode():
    for record in records:
        path = panel / record['file']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
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
        state = prediction['state'].gather(1,
            (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        ids = valid.flatten().nonzero().flatten()
        xy = torch.stack((ids.remainder(side), ids.div(side, rounding_mode='floor')),
                         -1).float()[None] / side
        chart = xy[:, None].expand(-1, 14, -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                    255 / side - chart[..., 0], chart[..., 0])
        points = centre[..., None, :] + torch.einsum(
            'bkij,bkpj->bkpi', edges, chart - .5)
        truth = target.reshape(-1, 3)[ids]
        original = (points[0] - truth[None]).norm(dim=-1).mean(-1)
        midpoint = centre[0] - edges[0, :, :, 0] * reflection[0, :, None] / side
        shape = points[0] - midpoint[:, None]
        recombined = midpoint[:, None, None] + shape[None]
        error = (recombined - truth[None, None]).norm(dim=-1).mean(-1)
        best_flat = int(error.flatten().argmin())
        selected = int(prior.gather(1, choice)[0].argmax())
        rows.append({key: record[key] for key in
            ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
             'appearance_mode', 'valid_pixels')})
        rows[-1].update({'visible_fraction': record['valid_pixels'] / (side * side),
            'selected_original_um': float(original[selected]),
            'best_original_um': float(original.min()),
            'best_recombined_um': float(error.flatten()[best_flat]),
            'recombined_centre_slot': best_flat // 14,
            'recombined_frame_slot': best_flat % 14,
            'best_recombined_is_original': best_flat // 14 == best_flat % 14})

animals = sorted({r['animal_id'] for r in rows})
summary = {'sections': len(rows), 'synthetic_identities': len(animals),
           'real_animal_validation': False, 'public_benchmark_used': False}
for key in ('selected_original_um', 'best_original_um', 'best_recombined_um',
            'best_recombined_is_original'):
    summary[key] = float(np.mean([np.mean([r[key] for r in rows
        if r['animal_id'] == animal]) for animal in animals]))
summary['oracle_recombination_gain_um'] = (
    summary['best_original_um'] - summary['best_recombined_um'])
summary['factorization_gate'] = summary['oracle_recombination_gain_um'] >= 250
for mode in ('raw', 'exact_black', 'imperfect_brush'):
    selected = [r for r in rows if r['appearance_mode'] == mode]
    summary[mode] = {'sections': len(selected),
                     'original_um': float(np.mean([r['best_original_um'] for r in selected])),
                     'recombined_um': float(np.mean([r['best_recombined_um'] for r in selected]))}
out.mkdir(parents=True, exist_ok=False)
with parent.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (panel / 'records.jsonl').open('rb') as stream:
    panel_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with Path(__file__).open('rb') as stream:
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'config.json').write_text(json.dumps({'parent_sha256': parent_sha,
    'panel_records_sha256': panel_sha, 'source_sha256': source_sha,
    'candidate_policy': 'old prior top8 + new prior top6, reflection expanded',
    'centre_frame_combinations': 196, 'target': 'all valid observed-pixel CCF coordinates',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json'):
    with (out / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'completed.json').write_text(json.dumps({'hashes_sha256': hashes,
    'rows': len(rows), 'public_benchmark_used': False}, indent=2))
print(json.dumps(summary), flush=True)
