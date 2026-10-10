"""Exact-versus-blind-near pose control for the frozen 142 matcher."""

import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_spatial_fit_142 import AtlasSpatialFit142
from training.global_plane_matcher_120 import attach_global_plane_matcher


device = 'cuda'
panel = root / 'data/fresh_v4_pose_dev_panel_132'
run = root / 'runs/scored_topk_correspondence_142'
parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
rows = [json.loads(line) for line in
        (root / 'runs/scored_topk_correspondence_142_dev_eval/rows.jsonl').open()]
near = {row['section_id']: row for row in rows if row['cohort'] == 'v4'
        and row['arm'] == 'top4_full_intensity' and row['condition'] == 'standard'
        and row['role'] == 'blind_best_original' and row['original_rigid_mm'] <= 1.5}
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
chosen = {}
for record in records:
    if record['section_id'] in near:
        chosen.setdefault(record['synthetic_subject_plan_id'], record)
assert len(chosen) == 8

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
heads = {}
for arm in ('top4_full_intensity', 'top4_support_only'):
    head = AtlasSpatialFit142(neighborhoods=4).to(device).eval().requires_grad_(False)
    head.load_state_dict(torch.load(run / arm / 'head_step_01200.pt',
                                    map_location='cpu', weights_only=True)['head'])
    heads[arm] = head
atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).to(device)
fixed = heads['top4_full_intensity'].fixed_query_xy
fixed_x, fixed_y = (fixed[:, 0] * 256).long(), (fixed[:, 1] * 256).long()


def counts(output, truth, valid):
    truth = truth[valid].float()
    global_world = output['coarse_key_world_um'][0, 0].float()
    global_support = output['coarse_key_support'][0, 0] > .5
    global_distance = torch.cdist(truth[None], global_world[None])[0]
    global_distance[:, ~global_support] = torch.inf
    global_hit = global_distance.amin(-1) <= 500
    logits = output['fine_logits'][0, 0, valid].float()
    world = output['fine_key_world_um'][0, 0, valid].float()
    support = output['fine_key_support'][0, 0, valid] > .5
    distance = (world - truth[:, None]).norm(dim=-1).masked_fill(~support, torch.inf)
    choice = logits.argmax(-1)
    scored = (choice < world.shape[1]) & (distance[
        torch.arange(len(choice), device=device), choice.clamp_max(world.shape[1] - 1)] <= 500)
    return {'valid': len(truth), 'global': int(global_hit.sum()),
            'local': int((global_hit & (distance.amin(-1) <= 500)).sum()),
            'scored': int((global_hit & scored).sum()),
            'dustbin': int((choice == world.shape[1]).sum())}


result = []
with torch.inference_mode():
    for record in chosen.values():
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).to(device)
            state = torch.from_numpy(arrays['target_state'][None].copy()).to(device)
            reflection = torch.from_numpy(arrays['reflection'].reshape(1, 1).copy()).to(device).long()
            truth = torch.from_numpy(arrays['target_centre_um'].copy()).to(device)[fixed_y, fixed_x]
            valid = torch.from_numpy(arrays['valid_mask'].copy()).to(device).bool()[fixed_y, fixed_x]
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).to(device)
            weights = torch.from_numpy(arrays['weights'][None].copy()).to(device)
        prediction = model.predict(image)
        branch = near[record['section_id']]['branch_id']
        candidates = {'exact': (state[:, None], reflection),
                      'blind_near': (prediction['state'][:, branch // 2:branch // 2 + 1],
                                     reflection.new_full((1, 1), branch % 2))}
        for arm, head in heads.items():
            for role, (candidate, reflected) in candidates.items():
                with torch.autocast('cuda', dtype=torch.float16):
                    output = head(prediction, image, candidate, reflected, atlas,
                                  offsets, weights, atlas_intensity_enabled=arm.endswith('full_intensity'))
                result.append({'section_id': record['section_id'],
                    'plan_id': record['synthetic_subject_plan_id'], 'arm': arm,
                    'role': role, 'blind_near_rigid_mm': near[record['section_id']]['original_rigid_mm'],
                    **counts(output, truth, valid)})
for arm in heads:
    for role in ('exact', 'blind_near'):
        group = [row for row in result if row['arm'] == arm and row['role'] == role]
        total = {name: sum(row[name] for row in group)
                 for name in ('valid', 'global', 'local', 'scored', 'dustbin')}
        print(json.dumps({'arm': arm, 'role': role, 'sections': len(group), **total,
            'scored_global_percent': 100 * total['scored'] / total['global'],
            'scored_local_percent': 100 * total['scored'] / total['local']}), flush=True)
print(json.dumps({'rows': result}), flush=True)
