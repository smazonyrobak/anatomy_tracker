"""Whole-plane image/CCF contrast with a finite-thickness arbitrary-plane renderer."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)


def rigid_points_090(state, reflection, chart, source_shape=(256, 256)):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(),
                             (source_shape[1] - 1) / source_shape[1] - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def render_atlas_planes_090(atlas, states, reflection, offsets, weights,
                            candidate_chunk=2, source_shape=(256, 256)):
    """Return (B,K,2,96,96) intensity/support pairs at the exact acquisition PSF."""
    batch, candidates = states.shape[:2]
    side = 96
    y = (torch.arange(side, device=states.device, dtype=states.dtype) + .5) / side - .5 / source_shape[0]
    x = (torch.arange(side, device=states.device, dtype=states.dtype) + .5) / side - .5 / source_shape[1]
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    offsets = torch.as_tensor(offsets, device=states.device, dtype=states.dtype)
    weights = torch.as_tensor(weights, device=states.device, dtype=states.dtype)
    if offsets.ndim == 1:
        offsets = offsets[None].expand(batch, -1)
    if weights.ndim == 1:
        weights = weights[None].expand(batch, -1)
    pairs = []
    for start in range(0, candidates, candidate_chunk):
        stop = min(start + candidate_chunk, candidates)
        count = stop - start
        state = states[:, start:stop].reshape(-1, 12)
        reflected = reflection[:, start:stop].reshape(-1)
        plane = rigid_points_090(state, reflected, chart, source_shape).reshape(-1, side, side, 3)
        frame = full_frame_state_to_components(state)[1]
        z = offsets[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        w = weights[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        world = plane[:, None] + z[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world, (0., 0., 0.), (25., 25., 25.), w)
        support = rendered[:, 1:2].clamp(0, 1)
        pairs.append(torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
                     .reshape(batch, count, 2, side, side))
    return torch.cat(pairs, 1)


class GlobalAtlasContrast090(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(5, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GELU())
        self.atlas = nn.Sequential(
            nn.Conv2d(2, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GELU())
        cells = 96 * (1 + 4 + 16)
        self.image_project = nn.Linear(cells, 128)
        self.atlas_project = nn.Linear(cells, 128)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.)))

    def forward(self, image, atlas_pair, candidate_chunk=3):
        query = self.image(F.interpolate(image, (96, 96), mode='area'))
        query = F.normalize(self.image_project(torch.cat([
            F.adaptive_avg_pool2d(query, (side, side)).flatten(1) for side in (1, 2, 4)
        ], 1)), dim=1)
        batch, candidates = atlas_pair.shape[:2]
        scores = []
        temperature = self.log_temperature.clamp(math.log(2.), math.log(20.)).exp()
        for start in range(0, candidates, candidate_chunk):
            count = min(candidate_chunk, candidates - start)
            key = self.atlas(atlas_pair[:, start:start + count].reshape(-1, 2, 96, 96))
            key = F.normalize(self.atlas_project(torch.cat([
                F.adaptive_avg_pool2d(key, (side, side)).flatten(1) for side in (1, 2, 4)
            ], 1)), dim=1).reshape(batch, count, -1)
            scores.append(temperature * (query[:, None] * key).sum(-1))
        return torch.cat(scores, 1)
