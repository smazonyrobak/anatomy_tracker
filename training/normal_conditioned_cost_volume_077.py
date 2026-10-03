"""Shared two-pass normal-conditioned atlas/image pose updater for 059 proposals."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)


class NormalConditionedCostVolume077(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(69, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.atlas = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv3d(7, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.location = nn.Conv3d(32, 1, 1)
        self.vector = nn.Sequential(nn.Linear(104, 64), nn.GELU(), nn.Linear(64, 7))
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

    def update_once(self, query, state, reflection, atlas, offsets, weights):
        count = len(state)
        centre, frame, basis = full_frame_state_to_components(state)
        axis = (torch.arange(24, device=state.device, dtype=state.dtype) + .5) / 12 - .5
        ay, ax = torch.meshgrid(axis, axis, indexing='ij')
        chart = torch.stack((torch.where(reflection[:, None, None].bool(),
            255 / 256 - ax, ax), ay.expand(count, -1, -1)), -1)
        base = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi',
            frame[..., :, :2] @ basis, chart - .5)
        depth = torch.linspace(-6000., 6000., 13, device=state.device, dtype=state.dtype)
        coordinates = base[:, None, None] + (
            depth[None, :, None] + offsets[:, None]
        )[..., None, None, None] * frame[:, None, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(atlas,
            coordinates.reshape(count * 13, offsets.shape[1], 24, 24, 3),
            (0., 0., 0.), (25., 25., 25.),
            weights[:, None].expand(-1, 13, -1).reshape(count * 13, -1))
        rendered = rendered.reshape(count, 13, 2, 24, 24).permute(0, 2, 1, 3, 4)
        support = rendered[:, 1:2]
        atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        key = F.normalize(self.atlas(atlas_pair), dim=1)
        query = F.normalize(query, dim=1)
        correlation = torch.stack([F.conv2d(key[i].permute(1, 0, 2, 3),
            query[i].reshape(4, 8, 12, 12), groups=4).permute(1, 0, 2, 3)
            for i in range(count)]) / 144
        position_axis = torch.linspace(-1, 1, 13, device=state.device, dtype=state.dtype)
        zz, yy, xx = torch.meshgrid(position_axis, position_axis, position_axis, indexing='ij')
        position = torch.stack((xx, yy, zz))[None].expand(count, -1, -1, -1, -1)
        context = self.context(torch.cat((correlation, position), 1))
        probability = self.location(context).flatten(2).softmax(-1)
        evidence = torch.cat(((context.flatten(2) * probability).sum(-1),
            (position.flatten(2) * probability).sum(-1),
            query.mean((-2, -1)), key.mean((-3, -2, -1)),
            frame[..., :, 2], state[:, 9:11] - math.log(12000.)), -1)
        raw = self.vector(evidence).tanh()
        tx, ty, normal, tilt_x, tilt_y, roll, scale = raw.unbind(-1)
        zero = torch.zeros_like(tx)
        update = torch.stack((tilt_x * (math.pi / 12), tilt_y * (math.pi / 12),
            roll * (math.pi / 12),
            tx * 3000, ty * 3000, normal * 3000,
            scale * math.log(1.2), scale * math.log(1.2), zero), -1)
        return compose_full_frame_state(state, update)

    def forward(self, prediction, inputs, state, reflection, atlas, offsets, weights):
        batch, candidates = state.shape[:2]
        source = torch.cat((F.adaptive_avg_pool2d(prediction['feature'], (12, 12)),
                            F.adaptive_avg_pool2d(inputs, (12, 12))), 1)
        query = self.query(source)[:, None].expand(-1, candidates, -1, -1, -1)
        query = query.reshape(batch * candidates, 32, 12, 12)
        current = state.reshape(batch * candidates, 12)
        flags = reflection.reshape(-1)
        slab_offsets = offsets[:, None].expand(-1, candidates, -1).flatten(0, 1)
        slab_weights = weights[:, None].expand(-1, candidates, -1).flatten(0, 1)
        first = self.update_once(query, current, flags, atlas, slab_offsets, slab_weights)
        second = self.update_once(query, first, flags, atlas, slab_offsets, slab_weights)
        return first.reshape(batch, candidates, 12), second.reshape(batch, candidates, 12)
