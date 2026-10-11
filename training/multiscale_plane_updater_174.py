"""Shared coarse-to-fine atlas-image full-frame updater for arbitrary sections.

The same cell sees a candidate finite-thickness render at 32 and 64 pixels.
It has no absolute pose input: the candidate state only defines the render and
the frame in which the nine residual coordinates are applied.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_recurrent_model import local_correlation
from training.global_atlas_contrast_090 import rigid_points_090


class MultiscalePlaneUpdater174(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(
            nn.Conv2d(8, 48, 5, padding=2), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.atlas = nn.Sequential(
            nn.Conv2d(6, 48, 5, padding=2), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.local = nn.Sequential(
            nn.Conv2d(172, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.query = nn.Conv2d(48, 48, 1)
        self.key = nn.Conv2d(48, 48, 1)
        self.head = nn.Sequential(
            nn.Conv2d(163, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.GELU(),
            nn.Linear(128, 10))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def render_bank(states, reflection, offsets, weights, atlas, image_shape, side):
        batch, count = states.shape[:2]
        bank = states.new_zeros(3, 9)
        bank[1, 5], bank[2, 5] = -750., 750.
        flat_candidates = states.flatten(0, 1)
        state = compose_full_frame_state(
            flat_candidates[:, None].expand(-1, 3, -1),
            bank[None].expand(len(flat_candidates), -1, -1))
        flat = state.flatten(0, 1)
        reflected = reflection.flatten().repeat_interleave(3)
        y = (torch.arange(side, device=states.device, dtype=states.dtype) + .5) / side
        x = (torch.arange(side, device=states.device, dtype=states.dtype) + .5) / side
        y -= .5 / image_shape[0]
        x -= .5 / image_shape[1]
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        chart = torch.stack((xx, yy), -1).reshape(-1, 2)
        plane = rigid_points_090(flat, reflected, chart, image_shape).reshape(-1, side, side, 3)
        normal = full_frame_state_to_components(flat)[1][..., :, 2]
        axial = torch.as_tensor(offsets, device=states.device, dtype=states.dtype)
        mass = torch.as_tensor(weights, device=states.device, dtype=states.dtype)
        if axial.ndim == 1:
            axial = axial[None].expand(batch, -1)
        if mass.ndim == 1:
            mass = mass[None].expand(batch, -1)
        axial = axial[:, None, None].expand(-1, count, 3, -1).reshape(len(flat), -1)
        mass = mass[:, None, None].expand(-1, count, 3, -1).reshape(len(flat), -1)
        coordinates = plane[:, None] + axial[:, :, None, None, None] * normal[:, None, None, None, :]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), mass)
        support = rendered[:, 1:2].clamp(0, 1)
        intensity = rendered[:, :1] / support.clamp_min(1e-4)
        tissue_mean = (intensity * support).sum((2, 3), keepdim=True) / support.sum((2, 3), keepdim=True).clamp_min(1)
        tissue_std = ((intensity - tissue_mean).square() * support).sum((2, 3), keepdim=True)
        tissue_std = (tissue_std / support.sum((2, 3), keepdim=True).clamp_min(1) + .01).sqrt()
        normalized = ((intensity - tissue_mean) / tissue_std).clamp(-4, 4) * support
        return torch.cat((normalized, support), 1).reshape(batch * count, 6, side, side)

    def step(self, inputs, states, reflection, offsets, weights, atlas, side):
        batch, count = states.shape[:2]
        source = F.interpolate(inputs, (side, side), mode='area')
        image = source[:, :1]
        standardized = (image - image.mean((2, 3), keepdim=True)) / (
            image.var((2, 3), keepdim=True, unbiased=False) + .01).sqrt()
        high = standardized - F.avg_pool2d(standardized, 7, 1, 3)
        source = torch.cat((source, standardized, high, high.abs()), 1)
        source = source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        rendered = self.render_bank(states, reflection, offsets, weights, atlas, inputs.shape[-2:], side)
        src, atl = self.source(source), self.atlas(rendered)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, side, device=states.device),
                                torch.linspace(-1, 1, side, device=states.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(batch * count, -1, -1, -1)
        local = self.local(torch.cat((src, atl, (src - atl).abs(),
                                      local_correlation(src, atl, 2),
                                      rendered[:, 1:2], xy), 1))
        query = F.adaptive_avg_pool2d(self.query(src), 16).flatten(2).transpose(1, 2)
        key = F.adaptive_avg_pool2d(self.key(atl), 16).flatten(2).transpose(1, 2)
        attention = (query @ key.transpose(1, 2) / math.sqrt(48)).softmax(-1)
        matched = attention @ key
        axis = torch.linspace(-1, 1, 16, device=states.device)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        position = torch.stack((gx, gy), -1).reshape(1, 256, 2).expand(batch * count, -1, -1)
        displacement = attention @ position - position
        evidence = torch.cat((F.adaptive_avg_pool2d(local, 16),
                              query.transpose(1, 2).reshape(batch * count, 48, 16, 16),
                              matched.transpose(1, 2).reshape(batch * count, 48, 16, 16),
                              displacement.transpose(1, 2).reshape(batch * count, 2, 16, 16),
                              attention.amax(-1).reshape(batch * count, 1, 16, 16)), 1)
        raw = self.head(evidence).reshape(batch, count, 10)
        limits = states.new_tensor((.55, .55, .55, 2200., 2200., 2200., .18, .18, .15))
        update = raw[..., :9].tanh() * limits
        return compose_full_frame_state(states, update), raw[..., 9], update

    def forward(self, inputs, states, reflection, offsets, weights, atlas):
        first, first_score, first_update = self.step(
            inputs, states, reflection, offsets, weights, atlas, 32)
        second, second_score, second_update = self.step(
            inputs, first, reflection, offsets, weights, atlas, 64)
        return {'input_state': states, 'coarse_state': first, 'state': second,
                'score': first_score + second_score,
                'coarse_update': first_update, 'fine_update': second_update}
