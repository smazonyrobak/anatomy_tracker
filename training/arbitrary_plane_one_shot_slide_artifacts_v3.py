"""TRAIN-only v3 appearance mix on independently sampled arbitrary physical planes.

The source state is the pre-artifact affine CCF gauge fitted to the virtual
subject's warped section. V3 adds no artifact-selected affine refit. Dense CCF
coordinates follow the observed pixels, including displaced valid fragments.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_streaming_synthetic_v7_appearance_v3 import (
    MODE_PROBS, sample_streaming_synthetic_v7_appearance_v3,
    sample_streaming_synthetic_v7_64_appearance_v3,
)


VERSION = 'one_shot_slide_artifacts_v3'
EVENT_PROBS = {'fragment': .22, 'fold': .20, 'bubble': .15, 'tile_seam': .25}
RAW_EXTERIOR_PROBS = (.8, .1, .1)


def sample_one_shot_slide_artifacts_v3(context, subject_indices, seed, side=192, source_sample=None):
    if source_sample is None:
        sampler = (sample_streaming_synthetic_v7_64_appearance_v3 if 'displacement_cpu' in context['bases'][0]
                   else sample_streaming_synthetic_v7_appearance_v3)
        source = sampler(context, subject_indices, seed, side)
    else:
        source = source_sample
    device = source['inputs'].device
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    coefficients, parameters, records = [[], [], []], [], []
    for row, (subject_index, original) in enumerate(zip(subject_indices, source['provenance'])):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject_index), row, 28]))
        strength = float(1.2 * rng.random() ** 2)
        for level, size in enumerate((4, 8, 16)):
            coefficients[level].append((rng.standard_normal((2, size, size)) * strength
                * side / 192 * (2.4, 1.2, .45)[level]).astype('float32'))
        events = {name: bool(rng.random() < probability) for name, probability in EVENT_PROBS.items()}
        angle = float(rng.uniform(0, np.pi))
        fragment_distance = float(rng.uniform(5, 13) * side / 192)
        p = {'fragment_x': float(rng.uniform(-.35, .35)), 'fragment_y': float(rng.uniform(-.35, .35)),
             'fragment_angle': angle, 'fragment_length': float(rng.uniform(.15, .32)),
             'fragment_width': float(rng.uniform(.065, .14)), 'fragment_wave': float(rng.uniform(-.012, .012)),
             'fragment_dx_px': float(fragment_distance * -np.sin(angle) + rng.uniform(-2, 2)),
             'fragment_dy_px': float(fragment_distance * np.cos(angle) + rng.uniform(-2, 2)),
             'fold_x': float(rng.uniform(-.45, .45)), 'fold_y': float(rng.uniform(-.45, .45)),
             'fold_angle': float(rng.uniform(0, np.pi)), 'fold_width_px': float(rng.uniform(3, 8) * side / 192),
             'fold_half_length': float(rng.uniform(.18, .48)), 'fold_slide_px': float(rng.uniform(1, 4) * side / 192),
             'bubble_x': float(rng.uniform(-.5, .5)), 'bubble_y': float(rng.uniform(-.5, .5)),
             'bubble_radius': float(rng.uniform(.11, .25)),
             'bubble_lens_px': float(rng.uniform(2, 6) * side / 192),
             'bubble_blur': float(rng.uniform(.5, .85)),
             'seam_vertical': float(rng.integers(2)), 'seam_position': float(rng.uniform(-.5, .5)),
             'seam_dx_px': float(rng.choice((-1, 1)) * rng.uniform(2, 7) * side / 192),
             'seam_dy_px': float(rng.uniform(-3, 3) * side / 192),
             'seam_gain': float(rng.uniform(.78, 1.2)), 'seam_offset': float(rng.uniform(-.07, .07)),
             'seam_defocus': float(rng.uniform(.3, .75)),
             'raw_exterior_code': float(rng.choice((0, 1, 2), p=RAW_EXTERIOR_PROBS))}
        parameters.append((events, p))
        records.append({**original, VERSION: {'seed_branch': 28, 'code_sha256': code_sha256,
            'mode_probabilities_raw_black_brush': MODE_PROBS,
            'event_probabilities': EVENT_PROBS,
            'raw_exterior_probabilities_black_near_gray': RAW_EXTERIOR_PROBS,
            'calibration': 'TRAIN appearance mixture, not measured physical-slide prevalence',
            'warp_strength': strength, 'warp_grid_sizes': [4, 8, 16], 'events': events, 'parameters': p,
            'pose_gauge': 'unchanged source affine CCF gauge fitted before v3 artifacts; not the pristine virtual-subject cutting plane and not fitted from artifact-selected pixels',
            'dense_target': 'observed pixel to CCF AP/DV/ML um; displaced single-layer fragment valid; overlap, fold, blur, background and out-of-frame invalid',
            'source_sampling': 'zero padding plus explicit background fill; no copied edge tissue',
            'raw_exterior_codes': {'0': 'exact black', '1': 'near-black without added noise',
                                   '2': 'inherited gray/confusing background; no brush in all raw cases'}}})

    q = {key: torch.tensor([p[key] for _, p in parameters], device=device, dtype=torch.float32)[:, None, None]
         for key in parameters[0][1]}
    event = {key: torch.tensor([e[key] for e, _ in parameters], device=device)[:, None, None]
             for key in parameters[0][0]}
    warp = sum(F.interpolate(torch.from_numpy(np.stack(level)).to(device), (side, side),
        mode='bicubic', align_corners=True) for level in coefficients)
    warp -= warp.mean((2, 3), keepdim=True)
    axis = torch.linspace(-1, 1, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    pixel = 2 / (side - 1)
    grid = torch.stack((xx + warp[:, 0] * pixel, yy + warp[:, 1] * pixel), -1)

    bx, by = xx - q['bubble_x'], yy - q['bubble_y']
    bubble_r = torch.sqrt((bx / q['bubble_radius']).square() + (by / q['bubble_radius']).square())
    lens = (1 - bubble_r.square()).clamp_min(0).square() * event['bubble']
    grid = grid + torch.stack((bx / q['bubble_radius'] * lens * q['bubble_lens_px'] * pixel,
                               by / q['bubble_radius'] * lens * q['bubble_lens_px'] * pixel), -1)
    seam_side = event['tile_seam'] & torch.where(q['seam_vertical'] > .5,
        xx > q['seam_position'], yy > q['seam_position'])
    grid = grid + torch.stack((seam_side * q['seam_dx_px'] * pixel,
                               seam_side * q['seam_dy_px'] * pixel), -1)

    fragment_grid = grid - torch.stack((q['fragment_dx_px'] * pixel,
                                        q['fragment_dy_px'] * pixel), -1)
    pair_grid = torch.stack((grid, fragment_grid), 1)
    dx, dy = pair_grid[..., 0] - q['fragment_x'][:, None], pair_grid[..., 1] - q['fragment_y'][:, None]
    fu = dx * q['fragment_angle'][:, None].cos() + dy * q['fragment_angle'][:, None].sin()
    fv = -dx * q['fragment_angle'][:, None].sin() + dy * q['fragment_angle'][:, None].cos()
    fragment_source = (fu / q['fragment_length'][:, None]).square() + (
        (fv - q['fragment_wave'][:, None] * torch.sin(8 * fu / q['fragment_length'][:, None]))
        / q['fragment_width'][:, None]).square() < 1

    fields = torch.cat((source['inputs'][:, :1], source['centre'].permute(0, 3, 1, 2),
        source['visible'][:, None], source['support'][:, None], source['masks'][:, None].float()), 1)
    base = F.grid_sample(fields, grid, padding_mode='zeros', align_corners=True)
    piece = F.grid_sample(fields, fragment_grid, padding_mode='zeros', align_corners=True)
    ones = torch.ones_like(source['support'][:, None])
    base_coverage = F.grid_sample(ones, grid, padding_mode='zeros', align_corners=True)[:, 0]
    piece_coverage = F.grid_sample(ones, fragment_grid, padding_mode='zeros', align_corners=True)[:, 0]
    raw = torch.tensor([r['mode'] == 'raw' for r in records], device=device)[:, None, None]
    slope = torch.tensor([r['appearance']['background_slope_yx'] for r in records], device=device)
    mean = torch.tensor([r['appearance']['background_mean'] for r in records], device=device)[:, None, None]
    background = (mean + slope[:, 0, None, None] * yy + slope[:, 1, None, None] * xx).clamp(0, 1)

    hole = event['fragment'] & fragment_source[:, 0] & (base[:, 4] > .25)
    displaced = event['fragment'] & fragment_source[:, 1] & (piece[:, 4] > .25) & (piece_coverage > .999)
    fragment_overlap = displaced & (base[:, 4] > .25) & ~hole
    image = torch.where(hole, background, base[:, 0] + (1 - base_coverage) * background)
    image = torch.where(displaced, torch.where(fragment_overlap,
        .7 * piece[:, 0] + .3 * image, piece[:, 0]), image)
    centre = torch.where(displaced[..., None], piece[:, 1:4].permute(0, 2, 3, 1),
                         base[:, 1:4].permute(0, 2, 3, 1))
    visible = torch.where(displaced, piece[:, 4], torch.where(hole, 0, base[:, 4]))
    brush = torch.where(displaced, piece[:, 6] > .5, (base[:, 6] > .5) & ~hole)
    valid = torch.where(displaced,
        (piece_coverage > .999) & (piece[:, 4] > .25) & (piece[:, 5] > .25) & ~fragment_overlap,
        (base_coverage > .999) & (base[:, 4] > .25) & (base[:, 5] > .25) & ~hole)

    transverse = -(xx - q['fold_x']) * q['fold_angle'].sin() + (yy - q['fold_y']) * q['fold_angle'].cos()
    longitudinal = (xx - q['fold_x']) * q['fold_angle'].cos() + (yy - q['fold_y']) * q['fold_angle'].sin()
    fold_width = q['fold_width_px'] * pixel
    fold_band = event['fold'] & (transverse.abs() < fold_width) & (longitudinal.abs() < q['fold_half_length'])
    fold_grid = grid + torch.stack((
        -q['fold_angle'].sin() * (-2 * transverse + q['fold_slide_px'] * pixel),
         q['fold_angle'].cos() * (-2 * transverse + q['fold_slide_px'] * pixel)), -1)
    fold = F.grid_sample(fields, fold_grid, padding_mode='zeros', align_corners=True)
    fold_coverage = F.grid_sample(ones, fold_grid, padding_mode='zeros', align_corners=True)[:, 0]
    fold_occlusion = fold_band & (fold[:, 4] > .25) & (fold_coverage > .999)
    fold_overlap = fold_occlusion & (visible > .25)
    fold_alpha = fold_occlusion * (.35 + .4 * (1 - transverse.abs() / fold_width).clamp_min(0))
    image = image * (1 - fold_alpha) + fold[:, 0] * fold_alpha
    crease = torch.exp(-.5 * (transverse / (.18 * fold_width)).square()) * fold_occlusion
    image *= 1 - .22 * crease
    visible = torch.where(fold_occlusion, torch.maximum(visible, fold[:, 4]), visible)
    valid &= ~fold_occlusion

    bubble_blur = event['bubble'] * (1 - bubble_r).clamp_min(0).square() * q['bubble_blur']
    blurred = F.avg_pool2d(F.pad(image[:, None], (3, 3, 3, 3), mode='reflect'), 7, 1)[:, 0]
    image = image * (1 - bubble_blur) + blurred * bubble_blur
    bubble_invalid = bubble_blur > .55
    valid &= ~bubble_invalid
    defocused = F.avg_pool2d(F.pad(image[:, None], (2, 2, 2, 2), mode='reflect'), 5, 1)[:, 0]
    image = torch.where(seam_side, (image * (1 - q['seam_defocus'])
        + defocused * q['seam_defocus']) * q['seam_gain'] + q['seam_offset'], image)
    exterior = raw & (visible <= .05)
    image = torch.where(exterior & (q['raw_exterior_code'] == 0), 0, image)
    image = torch.where(exterior & (q['raw_exterior_code'] == 1),
        (.006 + .012 * background).clamp(0, .02), image)

    brush = torch.where(raw, torch.ones_like(brush), brush)
    inputs = source['inputs'].clone()
    inputs[:, 0] = torch.where(raw, image, image * brush).clamp(0, 1)
    eroded = brush.clone()
    eroded[:, 1:] &= brush[:, :-1]
    eroded[:, :-1] &= brush[:, 1:]
    eroded[:, :, 1:] &= brush[:, :, :-1]
    eroded[:, :, :-1] &= brush[:, :, 1:]
    eroded[:, [0, -1], :] = False
    eroded[:, :, [0, -1]] = False
    inputs[:, 1] = ((brush & ~eroded) & ~raw).float()
    inputs[:, 2] = (~raw).float()
    centre = torch.where(valid[..., None], centre, 0)
    eligible = source['eligible'] & (valid.sum((1, 2)) >= side * side / 144)
    for row, record in enumerate(records):
        record[VERSION]['pixel_counts'] = {
            'valid': int(valid[row].sum()), 'out_of_frame': int((base_coverage[row] < .999).sum()),
            'fragment_displaced': int(displaced[row].sum()), 'fragment_overlap_invalid': int(fragment_overlap[row].sum()),
            'fold_occlusion_invalid': int(fold_occlusion[row].sum()), 'fold_overlap_invalid': int(fold_overlap[row].sum()),
            'bubble_defocus_invalid': int(bubble_invalid[row].sum())}
    return {'inputs': inputs, 'state': source['state'], 'reflection': source['reflection'],
        'offsets': source['offsets'], 'weights': source['weights'], 'centre': centre,
        'valid_mask': valid, 'visible': visible, 'brush_mask': brush, 'eligible': eligible,
        'observed_to_source_grid': torch.where(displaced[..., None], fragment_grid, grid),
        'fragment_displaced_mask': displaced, 'fragment_overlap_mask': fragment_overlap,
        'fold_occlusion_mask': fold_occlusion, 'fold_overlap_mask': fold_overlap,
        'bubble_invalid_mask': bubble_invalid, 'provenance': records}
