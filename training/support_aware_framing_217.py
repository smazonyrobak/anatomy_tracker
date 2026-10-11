"""Optional TRAIN-only crop of one fresh 166 or 211 section before v3/v4 artifacts."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_physical_ouv,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_support_aware_framing_217(source_sample, seed):
    """Return one reframed source_sample for sample_one_shot_slide_artifacts_v3/v4.

    One call transforms one independent physical draw; it does not make paired
    versions. The canonical state is composed before the unchanged observed
    horizontal-reflection flag, using pixel coordinates x/W and y/H.
    """
    source = source_sample
    assert source['inputs'].shape[0] == len(source['provenance']) == 1
    original = source['provenance'][0]
    assert original.get('split', original.get('reserved_split')) == 'train'
    plane_id = original['physical_section_id']
    plane_key = int.from_bytes(hashlib.sha256(plane_id.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), plane_key, 217]))
    _, _, height, width = source['inputs'].shape
    midpoint_x, midpoint_y = (width - 1) / 2, (height - 1) / 2

    branch = 'full_box'
    anchor_y, anchor_x = midpoint_y, midpoint_x
    if rng.random() < .5:
        pixels = torch.nonzero(source['visible'][0] > .25).cpu().numpy()
        if len(pixels):
            anchor_y, anchor_x = map(int, pixels[rng.integers(len(pixels))])
            branch = 'visible_support_pixel'
    scale = float(np.exp(rng.uniform(np.log(.6), np.log(1.25))))
    translation_x, translation_y = rng.uniform(-.12, .12, 2) * (width, height)
    centre_x, centre_y = anchor_x + translation_x, anchor_y + translation_y
    x = centre_x + scale * (torch.arange(width, device=source['inputs'].device,
                                           dtype=source['inputs'].dtype) - midpoint_x)
    y = centre_y + scale * (torch.arange(height, device=source['inputs'].device,
                                          dtype=source['inputs'].dtype) - midpoint_y)
    yy, xx = torch.meshgrid(2 * y / (height - 1) - 1, 2 * x / (width - 1) - 1, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None]

    inputs = F.grid_sample(source['inputs'], grid, align_corners=True)
    support = F.grid_sample(source['support'][:, None], grid, align_corners=True)[:, 0]
    visible = F.grid_sample(source['visible'][:, None], grid, align_corners=True)[:, 0]
    centre = F.grid_sample(source['centre'].permute(0, 3, 1, 2), grid,
                           align_corners=True).permute(0, 2, 3, 1)
    masks = F.grid_sample(source['masks'][:, None].float(), grid, mode='nearest',
                          align_corners=True)[:, 0] > .5
    inside = F.grid_sample(torch.ones_like(source['support'][:, None]), grid,
                           align_corners=True)[:, 0] > .999
    masks &= inside
    support = support * inside
    visible = visible * inside * masks
    centre = centre * inside[..., None]
    available = source['inputs'][:, 2, 0, 0] > .5
    raw = ~available
    masks = torch.where(raw[:, None, None], torch.ones_like(masks), masks)
    background = float(original['appearance']['background_mean'])
    inputs[:, 0] = torch.where(raw[:, None, None] & ~inside, background, inputs[:, 0])

    eroded = masks.clone()
    eroded[:, 1:] &= masks[:, :-1]
    eroded[:, :-1] &= masks[:, 1:]
    eroded[:, :, 1:] &= masks[:, :, :-1]
    eroded[:, :, :-1] &= masks[:, :, 1:]
    eroded[:, [0, -1], :] = False
    eroded[:, :, [0, -1]] = False
    inputs[:, 0] *= masks
    inputs[:, 1] = ((masks & ~eroded) & available[:, None, None]).float()
    inputs[:, 2] = available[:, None, None].float()

    ouv = full_frame_state_to_physical_ouv(source['state']).reshape(1, 3, 3)
    reflected = bool(source['reflection'][0])
    # For a reflected source, old canonical x = W-1-old observed x.
    # Thus old canonical x at new canonical x=0 is W-1-centre_x-scale*(W-1)/2.
    canonical_x0 = (width - 1 - centre_x if reflected else centre_x) - scale * midpoint_x
    y0 = centre_y - scale * midpoint_y
    new_ouv = torch.stack((ouv[:, 0] + canonical_x0 / width * ouv[:, 1]
                           + y0 / height * ouv[:, 2],
                           scale * ouv[:, 1], scale * ouv[:, 2]), 1)
    state = full_frame_state_from_components(*physical_ouv_to_frame(new_ouv))
    eligible = visible.sum((1, 2)) >= height * width / 144
    mass = float(visible[0].sum())
    record = {**original, 'eligible': bool(eligible[0]), 'visible_support_mass': mass}
    record['support_aware_framing_217'] = {
        'code_sha256': CODE_SHA256, 'seed': int(seed), 'source_physical_section_id': plane_id,
        'source_eligible': bool(source['eligible'][0]),
        'source_visible_support_mass': float(source['visible'][0].sum()),
        'branch': branch, 'anchor_observed_xy_px': [float(anchor_x), float(anchor_y)],
        'translation_xy_px': [float(translation_x), float(translation_y)],
        'source_centre_observed_xy_px': [float(centre_x), float(centre_y)],
        'source_pixels_per_output_pixel': scale, 'align_corners': True,
        'horizontal_reflection_unchanged': reflected,
    }
    return {**source, 'inputs': inputs, 'state': state, 'centre': centre,
            'support': support, 'visible': visible, 'masks': masks,
            'eligible': eligible, 'provenance': [record]}
