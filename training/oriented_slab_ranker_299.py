"""Candidate-aligned, shared-residual anatomical evidence for blind pose beams."""

import math
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf


SIDE = 48
DEPTH_UM = (-500., -250., 0., 250., 500.)
SHIFTS = tuple((y, x) for y in range(-2, 3) for x in range(-2, 3))


def make_ranker_299():
    def encoder(channels):
        return nn.Sequential(
            nn.Conv2d(channels, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1),
        )
    return nn.ModuleDict({
        'source': encoder(5), 'atlas': encoder(2),
        'reliability': nn.Conv2d(32, 1, 1),
    })


def render_bank_299(atlas, states, reflections, offsets, weights, depth_offsets=DEPTH_UM):
    """Render on the centre of each 4x4 block of a native 192-pixel input."""
    count = len(states)
    centre, frame, basis = full_frame_state_to_components(states)
    axis = (torch.arange(SIDE, device=states.device, dtype=states.dtype) * 4 + 1.5) / 192
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    reflected_x = torch.where(reflections[:, None, None].bool(), 191 / 192 - xx, xx)
    chart = torch.stack((reflected_x, yy.expand_as(reflected_x)), -1)
    plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                     basis[:, None, None], chart)
    depth = states.new_tensor(depth_offsets)
    through = depth[None, :, None] + offsets[:, None, :]
    world = (plane[:, None, None] + through[..., None, None, None]
             * frame[:, None, None, None, None, :, 2])
    values = render_finite_thickness_coordinate_grid(
        atlas, world.reshape(count * len(depth), len(offsets[0]), SIDE, SIDE, 3),
        (0., 0., 0.), (25., 25., 25.), weights.repeat_interleave(len(depth), 0))
    return values.reshape(count, len(depth), 2, SIDE, SIDE)


def score_bank_299(ranker, inputs, bank, support_only=False):
    """One source-weighted score per plane; a single XY/depth residual serves every site."""
    count, levels = bank.shape[:2]
    source = F.avg_pool2d(inputs, 4)
    query = ranker['source'](source)
    reliability = ranker['reliability'](query)
    query = F.normalize(query, dim=1)[0]
    support = bank[:, :, 1:2].clamp(0, 1)
    intensity = bank[:, :, :1] / support.clamp_min(1e-4)
    if support_only:
        intensity = torch.zeros_like(intensity)
    atlas_input = torch.cat((intensity, support), 2).flatten(0, 1)
    key = F.normalize(ranker['atlas'](atlas_input), dim=1)
    key = key.reshape(count, levels, 32, SIDE * SIDE)
    support = support.reshape(count, levels, SIDE * SIDE)
    site_y, site_x = torch.meshgrid(torch.arange(3, SIDE, 6, device=inputs.device),
                                  torch.arange(3, SIDE, 6, device=inputs.device), indexing='ij')
    site_y, site_x = site_y.flatten(), site_x.flatten()
    source_site = query[:, site_y, site_x]
    confidence = reliability.sigmoid()[0, 0, site_y, site_x].clamp_min(.05)
    shifts = torch.tensor(SHIFTS, device=inputs.device)
    shifted = ((site_y[None] + shifts[:, :1]) * SIDE
               + site_x[None] + shifts[:, 1:]).long()
    atlas_site = key[:, :, :, shifted]
    similarity = (atlas_site * source_site[None, None, :, None, :]).sum(2)
    atlas_support = support[:, :, shifted]
    weighted = ((similarity * atlas_support - .5 * (1 - atlas_support))
                * confidence[None, None, None]).sum(-1) / confidence.sum()
    offset_cost = .012 * shifts.square().sum(-1)[None, None] + \
        .02 * torch.arange(levels, device=inputs.device).sub(levels // 2).square()[None, :, None]
    hypotheses = weighted - offset_cost
    evidence = (torch.logsumexp(12 * hypotheses.flatten(1), 1)
                - math.log(hypotheses.shape[1] * hypotheses.shape[2])) / 12
    return evidence, reliability
