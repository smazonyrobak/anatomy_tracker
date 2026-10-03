"""Frozen normal-information readout on existing synthetic development identities."""
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
from torch import nn

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/pose_normal_information_062_pilot'
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
out = root / 'runs/pose_normal_information_062_development_eval'
parent = root / 'runs/one_shot_anchor_quality_059/joint_step_50000.pt'
steps = (0, 1000, 3000, 5000)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['accepted_synthetic'] == 20000
records = [r for r in map(json.loads, (panel / 'records.jsonl').open()) if r['eligible']]
assert len(records) == 246 and len({r['animal_id'] for r in records}) == 8
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
model.requires_grad_(False)
head = nn.Sequential(nn.Linear(320 * 4 * 4, 512), nn.GELU(), nn.Linear(512, 64)).cuda().eval()
anchors = model.normal_anchor_frames[:, :, 2]
features, normals, base = [], [], []
with torch.inference_mode():
    for record in records:
        path = panel / record['file']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth_state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        normal = full_frame_state_to_components(truth_state)[1][0, :, 2]
        prediction = model.predict(image)
        predicted_normal = full_frame_state_to_components(prediction['state'])[1][0, :, :, 2]
        parent_angle = torch.rad2deg(torch.acos((predicted_normal @ normal).abs().clamp(max=1)))
        prior = prediction['log_mass'][0]
        parent_selected = parent_angle[prior.argmax()]
        parent_best8 = parent_angle[prior.topk(8).indices].min()
        feature = image
        for layer in model.encoder:
            feature = layer(feature)
        features.append(F.adaptive_avg_pool2d(feature, 4).flatten(1))
        normals.append(normal)
        base.append((float(parent_selected), float(parent_best8)))
    features = torch.cat(features)
    normals = torch.stack(normals)
    anchor_angles = torch.rad2deg(torch.acos((normals @ anchors.T).abs().clamp(max=1)))

out.mkdir(parents=True, exist_ok=False)
with parent.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (panel / 'records.jsonl').open('rb') as stream:
    panel_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with Path(__file__).open('rb') as stream:
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (run / 'completed.json').open('rb') as stream:
    run_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
checkpoint_hashes = {}
for step in steps:
    with (run / f'normal_step_{step:05d}.pt').open('rb') as stream:
        checkpoint_hashes[str(step)] = hashlib.file_digest(stream, 'sha256').hexdigest()
config = {'run_completed_sha256': run_sha, 'parent_sha256': parent_sha,
          'panel_records_sha256': panel_sha, 'checkpoints_sha256': checkpoint_hashes,
          'source_sha256': source_sha, 'steps': steps,
          'eligible_sections': len(records), 'synthetic_identities': 8,
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows, summary = [], []
with torch.inference_mode():
    for step in steps:
        checkpoint = torch.load(run / f'normal_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        head.load_state_dict(checkpoint['head'])
        logits = head(features)
        selected = logits.argmax(-1)
        top8 = logits.topk(8, -1).indices
        map_angle = anchor_angles.gather(1, selected[:, None])[:, 0]
        top8_angle = anchor_angles.gather(1, top8).min(-1).values
        for index, record in enumerate(records):
            rows.append({'step': step, 'animal_id': record['animal_id'],
                         'specimen_id': record['specimen_id'],
                         'experiment_id': record['experiment_id'],
                         'section_id': record['section_id'],
                         'appearance_mode': record['appearance_mode'],
                         'visible_fraction': record['valid_pixels'] / 65536,
                         'parent_map_angle_deg': base[index][0],
                         'parent_top8_angle_deg': base[index][1],
                         'probe_map_angle_deg': float(map_angle[index]),
                         'probe_top8_angle_deg': float(top8_angle[index])})
        current = rows[-len(records):]
        animals = sorted({r['animal_id'] for r in current})
        means = {}
        for key in ('parent_map_angle_deg', 'parent_top8_angle_deg',
                    'probe_map_angle_deg', 'probe_top8_angle_deg'):
            means[key] = float(np.mean([np.mean([r[key] for r in current
                if r['animal_id'] == animal]) for animal in animals]))
        for key in ('parent_map_angle_deg', 'parent_top8_angle_deg',
                    'probe_map_angle_deg', 'probe_top8_angle_deg'):
            means[key.replace('_angle_deg', '_within15')] = float(np.mean([
                np.mean([r[key] <= 15 for r in current if r['animal_id'] == animal])
                for animal in animals]))
        means['step'] = step
        means['appearance'] = {mode: {
            'parent_map_angle_deg': float(np.mean([r['parent_map_angle_deg']
                for r in current if r['appearance_mode'] == mode])),
            'probe_map_angle_deg': float(np.mean([r['probe_map_angle_deg']
                for r in current if r['appearance_mode'] == mode]))}
            for mode in ('raw', 'exact_black', 'imperfect_brush')}
        means['visibility'] = {name: float(np.mean([r['probe_map_angle_deg']
            for r in current if lo <= r['visible_fraction'] < hi]))
            for name, lo, hi in (('under10', 0, .1), ('10to25', .1, .25),
                                 ('over25', .25, 1))}
        means['normal_information_gate'] = (
            means['parent_map_angle_deg'] - means['probe_map_angle_deg'] >= 10
            and means['probe_top8_within15'] - means['parent_top8_within15'] >= .15
            and all(means['appearance'][mode]['probe_map_angle_deg'] <=
                means['appearance'][mode]['parent_map_angle_deg']
                for mode in ('raw', 'exact_black', 'imperfect_brush')))
        summary.append(means)
        print(json.dumps(means), flush=True)
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps({'checkpoints': summary,
    'eligible_sections': len(records), 'synthetic_identities': len(animals)}, indent=2))
hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json'):
    with (out / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'completed.json').write_text(json.dumps({'hashes_sha256': hashes,
    'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
