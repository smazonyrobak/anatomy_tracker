"""Frozen 128 source-by-atlas interaction on matched synthetic DEV section pairs."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
assert root.drive.upper() == source.drive.upper() == 'I:'
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.in_path_global_matcher_128 import global_plane_match_128


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


checkpoint_path = root / 'runs/joint_in_path_correspondence_128_treatment/joint_step_06000.pt'
run_done = json.loads((checkpoint_path.parent / 'completed.json').read_text())
panel = root / 'data/joint_in_path_correspondence_128_dev_panel'
panel_done = json.loads((panel / 'completed.json').read_text())
assert sha(checkpoint_path) == run_done['checkpoint_sha256']['6000']
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').read_text().splitlines()]
records = sorted((row for row in records if row['eligible']),
                 key=lambda row: row['section_id'])

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
del context
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 6000 and checkpoint['calibrated'] is False
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint

offsets = torch.linspace(-31.25, 31.25, 9, device='cuda')
weights = torch.tensor((1., 2., 2., 2., 2., 2., 2., 2., 1.), device='cuda') / 16
axis = (torch.arange(24, device='cuda') + .5) / 24 - .5 / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
cases = []
with torch.inference_mode():
    for record in records:
        assert sha(panel / record['file']) == record['sha256']
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'][None, None].copy()).cuda().float()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda().float()
            reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
        prediction = model.predict(image)
        match = global_plane_match_128(model, prediction,
            torch.zeros((1, 1), device='cuda', dtype=torch.long), reflection[:, None],
            offsets, weights, atlas, (256, 256), side=24,
            candidate_state=state[:, None], return_support=True)
        intact = F.adaptive_avg_pool2d(valid, (24, 24))[0, 0] >= .95
        cases.append({'record': record, 'state': state[0].cpu(),
            'reflection': int(reflection[0]),
            'feature': F.adaptive_avg_pool2d(prediction['feature'], (24, 24))[0].cpu(),
            'log_mass': prediction['log_mass'][0].cpu(),
            'reflection_logit': prediction['reflection_logit'][0].cpu(),
            'support': match['atlas_support'][0, 0].cpu().numpy(),
            'intact': intact.cpu().numpy(),
            'points': rigid_points_090(state, reflection, chart)[0]
                .reshape(24, 24, 3).cpu().numpy()})

available = set(range(len(cases)))
pairs = []
for i, a in enumerate(cases):
    if i not in available:
        continue
    options = []
    a_support = a['support'] > .5
    for j in sorted(available - {i}):
        b = cases[j]
        if a['record']['appearance_mode'] != b['record']['appearance_mode']:
            continue
        b_support = b['support'] > .5
        overlap = np.count_nonzero(a_support & b_support)
        dice = float(2 * overlap / max(
            np.count_nonzero(a_support) + np.count_nonzero(b_support), 1))
        area_gap = float(abs(a_support.mean() - b_support.mean()))
        common = (a['intact'] & b['intact']
                  & (a['support'] >= .8) & (b['support'] >= .8))
        common_count = int(common.sum())
        if dice < .85 or area_gap > .08 or common_count < 32:
            continue
        separation = float(np.linalg.norm(a['points'] - b['points'], axis=-1)[common].mean()
                           / 1000)
        if separation <= 1.5:
            continue
        different_plan = a['record']['synthetic_subject_plan_id'] != b['record'][
            'synthetic_subject_plan_id']
        options.append((int(different_plan), dice, common_count, -j,
                        j, area_gap, separation))
    if options:
        _, dice, common_count, _, j, area_gap, separation = max(options)
        pairs.append((i, j, dice, area_gap, common_count, separation))
        available.remove(i)
        available.remove(j)

rows = []
with torch.inference_mode():
    for i, j, dice, area_gap, common_count, separation in pairs:
        a, b = cases[i], cases[j]
        prediction = {'feature': torch.stack((a['feature'], b['feature'])).cuda(),
            'log_mass': a['log_mass'][None].expand(2, -1).cuda(),
            'reflection_logit': a['reflection_logit'][None].expand(2, -1).cuda()}
        states = torch.stack((a['state'], b['state']))[None].expand(2, -1, -1).cuda()
        reflection = torch.tensor((a['reflection'], b['reflection']),
            device='cuda')[None].expand(2, -1)
        mode = torch.zeros((2, 2), device='cuda', dtype=torch.long)
        full = global_plane_match_128(model, prediction, mode, reflection,
            offsets, weights, atlas, (256, 256), side=24, candidate_state=states)
        zero = global_plane_match_128(model, prediction, mode, reflection,
            offsets, weights, atlas, (256, 256), side=24, candidate_state=states,
            ablation='atlas_intensity_zero')
        assert torch.equal(full['input_score'][0], full['input_score'][1])
        full = full['match_logit'].cpu().numpy()
        zero = zero['match_logit'].cpu().numpy()
        full_interaction = float(full[0, 0] + full[1, 1] - full[0, 1] - full[1, 0])
        zero_interaction = float(zero[0, 0] + zero[1, 1] - zero[0, 1] - zero[1, 0])
        row = {'section_a': a['record']['section_id'],
            'section_b': b['record']['section_id'],
            'plan_a': a['record']['synthetic_subject_plan_id'],
            'plan_b': b['record']['synthetic_subject_plan_id'],
            'appearance_mode': a['record']['appearance_mode'],
            'support_dice': dice, 'support_area_gap': area_gap,
            'common_intact_supported_cells': common_count,
            'mean_rigid_separation_mm': separation,
            'full_match_logit_2x2': full.tolist(),
            'atlas_zero_match_logit_2x2': zero.tolist(),
            'matched_minus_source_swapped': full_interaction,
            'atlas_zero_matched_minus_source_swapped': zero_interaction,
            'atlas_intensity_increment': full_interaction - zero_interaction}
        rows.append(row)

full_values = np.array([row['matched_minus_source_swapped'] for row in rows])
zero_values = np.array([row['atlas_zero_matched_minus_source_swapped'] for row in rows])
increment = full_values - zero_values
summary = {'event': 'summary',
    'scope': 'oracle-state conditional DEV diagnostic; match_logit is correction propensity, so interaction sign is not a localization score; not independent animals',
    'checkpoint_sha256': run_done['checkpoint_sha256']['6000'],
    'panel_records_sha256': panel_done['records_sha256'],
    'eligible_sections': len(cases), 'disjoint_support_matched_pairs': len(rows),
    'full_interaction_mean': float(full_values.mean()) if len(rows) else None,
    'full_absolute_interaction_mean': float(np.abs(full_values).mean()) if len(rows) else None,
    'atlas_zero_interaction_mean': float(zero_values.mean()) if len(rows) else None,
    'atlas_zero_absolute_interaction_mean': float(np.abs(zero_values).mean())
        if len(rows) else None,
    'atlas_intensity_increment_mean': float(increment.mean()) if len(rows) else None,
    'absolute_interaction_gain_fraction_ge_0p01': float((
        np.abs(full_values) >= np.abs(zero_values) + .01).mean())
        if len(rows) else None}
output = root / 'runs/appearance_gap_129/crossed_image_atlas_128.json'
output.write_text(json.dumps({'summary': summary, 'pairs': rows}, indent=2, allow_nan=False))
print(json.dumps(summary, allow_nan=False), flush=True)
