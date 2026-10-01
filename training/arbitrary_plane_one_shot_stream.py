"""One image and dense observed-pixel-to-CCF supervision per sampled physical plane.

The source atlas/animal map is the v7 streaming generator. The added in-plane
sampling grid is shared by image, optional brush, visibility and CCF targets.
"""
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_streaming_synthetic_v7 import sample_streaming_synthetic_v7
from training.arbitrary_plane_streaming_synthetic_v7_64 import sample_streaming_synthetic_v7_64


def sample_one_shot_stream(context, subject_indices, seed, side=192):
    sampler = (sample_streaming_synthetic_v7_64 if 'displacement_cpu' in context['bases'][0]
               else sample_streaming_synthetic_v7)
    source = sampler(context, subject_indices, seed, side)
    device = source['inputs'].device
    coefficients, parameters, records = [[], [], []], [], []
    for row, (subject_index, original) in enumerate(zip(subject_indices, source['provenance'])):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 19]))
        strength = float(rng.uniform(.55, 1.3))
        for level, size in enumerate((4, 8, 16)):
            coefficients[level].append(rng.standard_normal((2, size, size)).astype('float32')
                * strength * side / 192 * (2.8, 1.4, .55)[level])
        event = {'tear': bool(rng.random() < .22), 'missing': bool(rng.random() < .16),
            'fold': bool(rng.random() < .20), 'bubble': bool(rng.random() < .15),
            'tile_seam': bool(rng.random() < .25)}
        p = {'tear_centre_yx': rng.uniform(-.35, .35, 2).tolist(), 'tear_angle': float(rng.uniform(0, np.pi)),
            'tear_half_length': float(rng.uniform(.2, .45)), 'tear_half_width_px': float(rng.uniform(.8, 2.4)),
            'tear_curvature_px': float(rng.uniform(-2.5, 2.5)),
            'fragment_shift_px': float(rng.choice((-1, 1)) * rng.uniform(1.5, 6.0)),
            'missing_centre_yx': rng.uniform(-.5, .5, 2).tolist(), 'missing_radius_yx': rng.uniform(.05, .16, 2).tolist(),
            'fold_centre_yx': rng.uniform(-.4, .4, 2).tolist(), 'fold_angle': float(rng.uniform(0, np.pi)),
            'fold_width_px': float(rng.uniform(1.5, 4)), 'fold_half_length': float(rng.uniform(.2, .48)),
            'fold_gain': float(rng.uniform(.12, .35)),
            'bubble_centre_yx': rng.uniform(-.4, .4, 2).tolist(), 'bubble_radius': float(rng.uniform(.08, .23)),
            'bubble_ring_px': float(rng.uniform(1.5, 3.5)),
            'seam_vertical': bool(rng.integers(2)), 'seam_position': float(rng.uniform(-.45, .45)),
            'seam_gain': float(rng.uniform(.85, 1.15)), 'seam_offset': float(rng.uniform(-.045, .045))}
        parameters.append((event, p))
        records.append({**original, 'one_shot': {'seed_branch': 19, 'warp_strength': strength,
            'warp_grid_sizes': [4, 8, 16], 'events': event, 'parameters': p,
            'dense_target': 'observed pixel to CCF AP/DV/ML um; invalid where tissue is missing or obscured'}})

    displacement = sum(F.interpolate(torch.from_numpy(np.stack(level)).to(device), (side, side),
        mode='bicubic', align_corners=True) for level in coefficients)
    displacement -= displacement.mean((2, 3), keepdim=True)
    axis = torch.linspace(-1, 1, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None] + displacement.permute(0, 2, 3, 1) * (2 / (side - 1))
    values = torch.tensor([[
        *p['tear_centre_yx'], p['tear_angle'], p['tear_half_length'], p['tear_half_width_px'], p['tear_curvature_px'], p['fragment_shift_px'],
        *p['missing_centre_yx'], *p['missing_radius_yx'],
        *p['fold_centre_yx'], p['fold_angle'], p['fold_width_px'], p['fold_half_length'], p['fold_gain'],
        *p['bubble_centre_yx'], p['bubble_radius'], p['bubble_ring_px'],
        p['seam_vertical'], p['seam_position'], p['seam_gain'], p['seam_offset'],
        record['appearance']['background_mean'], *record['appearance']['background_slope_yx']]
        for (_, p), record in zip(parameters, records)], device=device, dtype=torch.float32)
    (ty, tx, ta, tl, tw, tc, shift, my, mx, ry, rx, fy, fx, fa, fw, fl, fg,
        by, bx, br, bw, sv, sp, sg, so, bm, bsy, bsx) = [column[:, None, None] for column in values.unbind(1)]
    tear_on, missing_on, fold_on, bubble_on, seam_on = [column[:, None, None]
        for column in torch.tensor([[event[key] for key in ('tear', 'missing', 'fold', 'bubble', 'tile_seam')]
            for event, _ in parameters], device=device).unbind(1)]
    raw = torch.tensor([record['mode'] == 'raw' for record in records], device=device)[:, None, None]
    across = -(xx - tx) * torch.sin(ta) + (yy - ty) * torch.cos(ta)
    along = (xx - tx) * torch.cos(ta) + (yy - ty) * torch.sin(ta)
    fragment = tear_on.float() * (across > 0).float() * torch.sigmoid((tl - along.abs()) * side / 8)
    grid = grid + torch.stack((-torch.sin(ta) * shift * fragment,
                               torch.cos(ta) * shift * fragment), -1) * (2 / (side - 1))
    inside = grid.abs().amax(-1) <= 1
    fields = torch.cat((source['inputs'], source['centre'].permute(0, 3, 1, 2),
        source['visible'][:, None], source['support'][:, None], source['masks'][:, None].float()), 1)
    warped = F.grid_sample(fields, grid, mode='bilinear', padding_mode='border', align_corners=True)
    inputs, centre = warped[:, :5].clone(), warped[:, 5:8].permute(0, 2, 3, 1)
    visible, support, brush = warped[:, 8], warped[:, 9], warped[:, 10] > .5
    valid = inside & (visible > .25) & (support > .25)
    tear = tear_on & ((across + tc * 2 / side * torch.sin(along * 8)).abs() < tw * 2 / side) & (along.abs() < tl)
    missing = missing_on & (((yy - my) / ry).square() + ((xx - mx) / rx).square() < 1)
    removed = tear | missing
    transverse = -(xx - fx) * torch.sin(fa) + (yy - fy) * torch.cos(fa)
    longitudinal = (xx - fx) * torch.cos(fa) + (yy - fy) * torch.sin(fa)
    ridge = torch.exp(-.5 * (transverse * side / (2 * fw)).square() - .5 * (longitudinal / fl).pow(8))
    ridge *= fold_on * (support > .15)
    radius = ((yy - by).square() + (xx - bx).square()).sqrt()
    ring = torch.exp(-.5 * ((radius - br) * side / (2 * bw)).square()) * bubble_on
    image = (inputs[:, 0] + fg * ridge) * (1 - .25 * ring) + .28 * ring
    image = torch.where(seam_on & (torch.where(sv > .5, xx, yy) > sp), image * sg + so, image)
    image = torch.where(removed & raw, bm + bsy * yy + bsx * xx, image)
    brush &= ~(removed & ~raw)
    inputs[:, 0] = torch.where(raw, image, image * brush).clamp(0, 1)
    valid &= ~removed & (ridge < .8) & (ring < .6)
    eligible = source['eligible'] & (valid.sum((1, 2)) >= side * side / 144)
    available = ~raw[:, 0, 0]
    eroded = brush.clone()
    eroded[:, 1:] &= brush[:, :-1]
    eroded[:, :-1] &= brush[:, 1:]
    eroded[:, :, 1:] &= brush[:, :, :-1]
    eroded[:, :, :-1] &= brush[:, :, 1:]
    inputs[:, 1] = ((brush & ~eroded) & available[:, None, None]).float()
    inputs[:, 2] = available[:, None, None].float()
    return {'inputs': inputs, 'state': source['state'], 'reflection': source['reflection'],
        'offsets': source['offsets'], 'weights': source['weights'], 'centre': centre,
        'valid_mask': valid, 'visible': visible * ~removed, 'brush_mask': brush, 'eligible': eligible,
        'observed_to_source_grid': grid, 'provenance': records}
