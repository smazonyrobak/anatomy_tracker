"""TRAIN-only availability of support-matched pairs in the frozen 132 blind beam."""

import hashlib
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
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher

source = Path(__file__).resolve().parent
parent_dir = root / 'runs/v4_pose_adaptation_132'
checkpoint = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/joint_feedback_161_train_preflight_001'
seed, draws, side = 20261011161, 128, 256
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)

source_names = (
    'audit_joint_feedback_161_train_pairs.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
hashes = {}
for name in source_names:
    with (source / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
with checkpoint.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (parent_dir / 'config.json').open('rb') as stream:
    parent_config_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
with (parent_dir / 'completed.json').open('rb') as stream:
    parent_done_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
parent_config = json.loads((parent_dir / 'config.json').read_text())
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert checkpoint_sha == parent_done['checkpoint_sha256']['2000']
assert hashes['arbitrary_plane_one_shot_model.py'] == parent_config['source_sha256'][
    'arbitrary_plane_one_shot_model.py']
assert hashes['global_plane_matcher_120.py'] == parent_config['source_sha256'][
    'global_plane_matcher_120.py']

context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
atlas = context['atlas']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config and saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
del saved

axis = torch.arange(side, device='cuda') / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart256 = torch.stack((xx, yy), -1)
axis24 = (torch.arange(24, device='cuda') + .5) / 24 - .5 / side
yy24, xx24 = torch.meshgrid(axis24, axis24, indexing='ij')
chart24 = torch.stack((xx24, yy24), -1).reshape(-1, 2)
config = {'seed': seed, 'draws': draws, 'parent': '132 step 2000 only',
    'checkpoint_sha256': checkpoint_sha, 'parent_config_sha256': parent_config_sha,
    'parent_completion_sha256': parent_done_sha, 'source_sha256': hashes,
    'synthetic_provenance': context['provenance'],
    'sampling': '128 independent v4 TRAIN physical sections, one attempt each; base indices 0..59',
    'blind16': 'top 8 free + top 6 anchors + 2 antipodal-normal diversity anchors',
    'rigid_cost': 'mean CCF displacement over all valid native-256 pixels, mm',
    'pair_rule': 'near <=1.5 mm; wrong >=near+1 mm; original 24-grid support >0.5, Dice >=0.85, area gap <=0.08, common cells >=32',
    'atlas_support': 'nine-sample finite-thickness render with each section PSF offsets/weights',
    'no_model_training': True, 'development_or_final_data_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for attempt in range(draws):
        base = int(rng.integers(60))
        draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
        original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
        sample = sample_one_shot_slide_artifacts_v4(
            context, [base], draw_seed, side, source_sample=original)
        provenance = sample['provenance'][0]
        assert provenance['split'] == 'train' and provenance['base_lineage']['split'] == 'train'
        valid = sample['valid_mask'][0]
        normal = np.asarray(provenance['physical_unit_normal_ap_dv_ml'])
        angle = float(np.degrees(np.arccos(np.clip(np.abs(normal).max(), 0, 1))))
        row = {'attempt': attempt, 'base_index': base, 'draw_seed': draw_seed,
            'physical_section_id': provenance['physical_section_id'],
            'base_lineage': provenance['base_lineage'],
            'appearance': provenance['mode'],
            'eligible': bool(sample['eligible'][0]),
            'valid_pixels': int(valid.sum()), 'valid_fraction': float(valid.float().mean()),
            'nearest_cardinal_angle_deg': angle, 'provenance': provenance,
            'beam_branch_ids': [], 'beam_rigid_mm': [], 'near_slots': [],
            'matched_pairs': []}
        if row['eligible']:
            prediction = model.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                              prior[:, 32:].topk(6, -1).indices + 32), -1)
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            for _ in range(2):
                chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
                diversity = -(normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                anchor = diversity.argmax(-1)
                reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
                beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
            ids = beam[0]
            state = prediction['state'][:, ids // 2]
            reflection = (ids % 2)[None]
            observed = chart256[valid]
            truth = rigid_points_090(sample['state'], sample['reflection'], observed)[0]
            proposed = rigid_points_090(state, reflection, observed)[0]
            cost = (proposed - truth).norm(dim=-1).mean(-1) / 1000

            plane = rigid_points_090(state, reflection, chart24).reshape(16, 24, 24, 3)
            frame = full_frame_state_to_components(state)[1][0]
            world = plane[:, None] + sample['offsets'][0][None, :, None, None, None] * \
                frame[:, None, None, None, :, 2]
            weights = sample['weights'].expand(16, -1)
            rendered = render_finite_thickness_coordinate_grid(
                atlas, world, (0., 0., 0.), (25., 25., 25.), weights)
            support = (rendered[:, 1] > .5).flatten(1).float()
            area = support.sum(-1)
            overlap = support @ support.T
            dice = 2 * overlap / (area[:, None] + area[None, :]).clamp_min(1)
            area_gap = (area[:, None] - area[None, :]).abs() / (24 * 24)
            near = cost <= 1.5
            pair = near[:, None] & (cost[None, :] >= cost[:, None] + 1.) & \
                (dice >= .85) & (area_gap <= .08) & (overlap >= 32)
            slots = pair.nonzero(as_tuple=False).tolist()
            row['beam_branch_ids'] = ids.tolist()
            row['beam_rigid_mm'] = cost.tolist()
            row['near_slots'] = near.nonzero(as_tuple=False).flatten().tolist()
            row['matched_pairs'] = [{'near_slot': i, 'wrong_slot': j,
                'support_dice': float(dice[i, j]), 'support_area_gap': float(area_gap[i, j]),
                'common_support_cells': int(overlap[i, j])} for i, j in slots]
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if (attempt + 1) % 16 == 0:
            stream.flush()
            print(json.dumps({'draws': attempt + 1,
                'eligible': sum(r['eligible'] for r in rows),
                'pair_sections': sum(bool(r['matched_pairs']) for r in rows)}), flush=True)

tissue_bins = {'<8%': (0, .08), '8-20%': (.08, .20), '>=20%': (.20, 1.01)}
angle_bins = {'0-15deg': (0, 15), '15-30deg': (15, 30),
              '30-45deg': (30, 45), '45-55deg': (45, 55)}
summary = {'draws': len(rows), 'eligible': sum(r['eligible'] for r in rows),
    'near_sections': sum(bool(r['near_slots']) for r in rows),
    'pair_sections': sum(bool(r['matched_pairs']) for r in rows),
    'matched_pairs': sum(len(r['matched_pairs']) for r in rows)}
for key, bins, field in (('by_tissue_fraction', tissue_bins, 'valid_fraction'),
                         ('by_nearest_cardinal_angle', angle_bins, 'nearest_cardinal_angle_deg')):
    summary[key] = {}
    for name, (lower, upper) in bins.items():
        group = [r for r in rows if lower <= r[field] < upper]
        summary[key][name] = {'draws': len(group),
            'eligible': sum(r['eligible'] for r in group),
            'near_sections': sum(bool(r['near_slots']) for r in group),
            'pair_sections': sum(bool(r['matched_pairs']) for r in group),
            'matched_pairs': sum(len(r['matched_pairs']) for r in group)}
assert len({r['physical_section_id'] for r in rows}) == draws
assert sum(v['draws'] for v in summary['by_tissue_fraction'].values()) == draws
assert sum(v['draws'] for v in summary['by_nearest_cardinal_angle'].values()) == draws
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
output_hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json'):
    with (out / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'completed.json').write_text(json.dumps({
    'status': 'complete', 'output_sha256': output_hashes,
    'source_sha256': hashes['audit_joint_feedback_161_train_pairs.py'],
    'checkpoint_sha256': checkpoint_sha, 'summary': summary}, indent=2, allow_nan=False))
print(json.dumps({'run': str(out), **summary}), flush=True)
