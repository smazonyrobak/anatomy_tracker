"""TRAIN-only feasibility of a 2-D finite-slab attention target; no optimization."""

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
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
parent_dir = root / 'runs/joint_pose_map_122'
parent_path = parent_dir / 'joint_step_02000.pt'
out = root / 'runs/joint_in_path_ce_128_train_mask_jitter_audit'
seed, draw_seed, accepted_target = 2026101012801, 2026101012802000000, 100
side, grid_side = 256, 24
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == 'I:' and not out.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
context = load_streaming_synthetic_v7_64(device='cuda')
atlas = context['atlas']
checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
subject_rng = torch.Generator().manual_seed(seed)
jitter_rng = torch.Generator().manual_seed(seed + 1)
axis = (torch.arange(grid_side, device='cuda') + .5) / grid_side - .5 / side
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def mask_counts(section, state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    sx = torch.where(reflection[:, None, None].bool(), (side - 1) / side - xx, xx)
    chart = torch.stack((sx.expand(len(state), -1, -1),
                         yy.expand(len(state), -1, -1)), -1)
    plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                     basis[:, None, None], chart)
    offsets, weights = section['offsets'], section['weights']
    slab = plane[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
    rendered = render_finite_thickness_coordinate_grid(
        atlas, slab, (0., 0., 0.), (25., 25., 25.), weights)
    support = rendered[:, 1:2]
    target = F.interpolate(section['centre'].permute(0, 3, 1, 2),
                           (grid_side, grid_side), mode='bilinear',
                           align_corners=False).permute(0, 2, 3, 1)
    intact = F.adaptive_avg_pool2d(section['valid_mask'][:, None].float(),
                                   (grid_side, grid_side))[:, 0] >= .95
    delta = target - centre[:, None, None]
    local = torch.einsum('bhwi,bij->bhwj', delta, frame[:, :, :2])
    raster = .5 + torch.linalg.solve(basis[:, None, None], local[..., None]).squeeze(-1)
    key_x = (torch.where(reflection[:, None, None].bool(),
                         (side - 1) / side - raster[..., 0], raster[..., 0])
             + .5 / side) * grid_side - .5
    key_y = (raster[..., 1] + .5 / side) * grid_side - .5
    in_grid = ((key_x >= 0) & (key_x < grid_side - 1)
               & (key_y >= 0) & (key_y < grid_side - 1))
    location = torch.stack((2 * key_x / (grid_side - 1) - 1,
                            2 * key_y / (grid_side - 1) - 1), -1)
    sampled_support = F.grid_sample(support, location, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)[:, 0]
    in_support = sampled_support >= .8
    normal = torch.einsum('bhwi,bi->bhw', delta, frame[:, :, 2]).abs()
    in_psf = normal <= offsets.abs().amax(-1)[:, None, None] + 12.5
    return {'intact': int(intact.sum()),
        'intact_in_grid': int((intact & in_grid).sum()),
        'intact_in_grid_support': int((intact & in_grid & in_support).sum()),
        'final': int((intact & in_grid & in_support & in_psf).sum()),
        'psf_halfwidth_um': float(offsets.abs().amax()),
        'median_normal_um_intact_in_grid_support': float(normal[intact & in_grid & in_support]
            .median()) if bool((intact & in_grid & in_support).any()) else None}


rows, attempts = [], 0
with torch.inference_mode():
    while len(rows) < accepted_target:
        virtual = int(torch.randint(len(context['subjects']), (1,), generator=subject_rng))
        section = sample_one_shot_slide_artifacts_v3(context, [virtual], draw_seed, side=side)
        attempts += 1
        draw_seed += 1
        if not bool(section['eligible'][0]):
            continue
        prediction = model.predict(section['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                          prior[:, 32:].topk(6, -1).indices + 32), -1)
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        for _ in range(2):
            chosen = normals[torch.arange(1, device='cuda')[:, None], beam // 2]
            similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
            diversity = -similarity
            diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
            anchor = diversity.argmax(-1)
            reflected = prior[:, 32:].reshape(1, 64, 2)[0, anchor].argmax(-1)
            beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
        pixel = torch.multinomial(section['valid_mask'].flatten(1).float(),
                                  128, replacement=True)
        chart = torch.stack((pixel.remainder(side),
            pixel.div(side, rounding_mode='floor')), -1).float() / side
        states = prediction['state'][0, beam[0] // 2]
        flags = (beam % 2)[0]
        reference = rigid_points_090(section['state'], section['reflection'], chart)[0]
        proposed = rigid_points_090(states[None], flags[None], chart)[0]
        fixed = rigid_points_090(section['state'], section['reflection'], corners[None])[0]
        corners_pred = rigid_points_090(states[None], flags[None], corners[None, None])[0]
        true_normal = full_frame_state_to_components(section['state'])[1][0, :, 2]
        chosen_normal = full_frame_state_to_components(states)[1][:, :, 2]
        physical = (.75 * (proposed - reference).norm(dim=-1).mean(-1)
                    + .25 * (corners_pred - fixed).norm(dim=-1).mean(-1)) / 1000
        physical += 4 * (1 - (chosen_normal * true_normal).sum(-1).abs().clamp_max(1))
        best = int(physical.argmin())
        jitter = torch.zeros(1, 9, device='cuda')
        random = 2 * torch.rand(3, generator=jitter_rng).cuda() - 1
        jitter[0, 2] = .05 * random[0]
        jitter[0, 3:5] = 750 * random[1:]
        jitter_state = compose_full_frame_state(section['state'], jitter)
        exact = mask_counts(section, section['state'], section['reflection'])
        jittered = mask_counts(section, jitter_state, section['reflection'])
        predicted = (mask_counts(section, states[best:best + 1], flags[best:best + 1])
                     if float(physical[best]) <= 1.5 else None)
        rows.append({'draw_seed': draw_seed - 1, 'virtual_subject_index': virtual,
            'physical_section_id': section['provenance'][0]['physical_section_id'],
            'appearance_mode': section['provenance'][0]['mode'],
            'valid_fraction': float(section['valid_mask'].float().mean()),
            'exact': exact, 'jittered': jittered,
            'jitter_local_update': jitter[0].tolist(),
            'predicted_near_rigid_cost_mm': float(physical[best]),
            'predicted': predicted})
        if len(rows) % 25 == 0:
            print(json.dumps({'event': 'accepted', 'count': len(rows),
                              'attempts': attempts}), flush=True)

summary = {'accepted': len(rows), 'attempts': attempts,
    'exact_sections_with_any': sum(row['exact']['final'] > 0 for row in rows),
    'exact_total_intact': sum(row['exact']['intact'] for row in rows),
    'exact_total_in_grid': sum(row['exact']['intact_in_grid'] for row in rows),
    'exact_total_in_support': sum(row['exact']['intact_in_grid_support'] for row in rows),
    'exact_total_final': sum(row['exact']['final'] for row in rows),
    'jittered_sections_with_any': sum(row['jittered']['final'] > 0 for row in rows),
    'jittered_total_intact': sum(row['jittered']['intact'] for row in rows),
    'jittered_total_in_grid': sum(row['jittered']['intact_in_grid'] for row in rows),
    'jittered_total_in_support': sum(row['jittered']['intact_in_grid_support'] for row in rows),
    'jittered_total_final': sum(row['jittered']['final'] for row in rows),
    'predicted_near_sections': sum(row['predicted'] is not None for row in rows),
    'predicted_sections_with_any': sum(row['predicted'] is not None
        and row['predicted']['final'] > 0 for row in rows),
    'predicted_total_final': sum(row['predicted']['final'] for row in rows
        if row['predicted'] is not None),
    'mean_exact_final_per_section': float(np.mean([row['exact']['final'] for row in rows])),
    'median_exact_final_per_section': float(np.median([row['exact']['final'] for row in rows])),
    'mean_jittered_final_per_section': float(np.mean([row['jittered']['final'] for row in rows])),
    'median_jittered_final_per_section': float(np.median([row['jittered']['final'] for row in rows])),
    'by_mode': {mode: {'sections': sum(row['appearance_mode'] == mode for row in rows),
        'exact_with_any': sum(row['appearance_mode'] == mode and row['exact']['final'] > 0
            for row in rows),
        'exact_total_final': sum(row['exact']['final'] for row in rows
            if row['appearance_mode'] == mode),
        'jittered_with_any': sum(row['appearance_mode'] == mode
            and row['jittered']['final'] > 0 for row in rows),
        'jittered_total_final': sum(row['jittered']['final'] for row in rows
            if row['appearance_mode'] == mode)}
        for mode in ('raw', 'exact_black', 'imperfect_brush')},
    'role': 'TRAIN-only physical-label feasibility; no optimization or DEV access'}
config = {'seed': seed, 'first_draw_seed': 2026101012802000000,
    'accepted_target': accepted_target, 'source_sha256': sha(source / Path(__file__).name),
    'parent_checkpoint_sha256': sha(parent_path),
    'synthetic_provenance': context['provenance'],
    'mask': '24-grid: adaptive-valid>=.95, four in-grid keys, bilinear support>=.8, normal residual<=sample PSF halfwidth+12.5um',
    'jitter': 'uniform independent local in-plane x/y shifts +/-750um and roll +/-0.05rad; zero normal shift'}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'output_sha256': {name: sha(out / name)
    for name in ('config.json', 'rows.jsonl', 'summary.json')},
    'source_sha256': config['source_sha256'],
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'accepted': len(rows), 'attempts': attempts,
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False}, indent=2))
print(json.dumps({'event': 'completed', **summary}), flush=True)
