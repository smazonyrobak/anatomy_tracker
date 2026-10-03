"""Trainable whole-slice pose feedback from a local 2D-to-3D atlas cost volume."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)


class WholeSliceAtlasFeedback081(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
        )
        self.atlas = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.GroupNorm(8, 16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(51, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.mapper_feature = nn.Conv2d(32, 64, 1)
        self.summary = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(1024, 128), nn.GELU(),
        )
        self.pose = nn.Linear(142, 9)
        self.quality = nn.Linear(142, 1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.)))
        for layer in (self.mapper_feature, self.pose, self.quality):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, image_feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256), fit_summary=None):
        """Fit summary: warp magnitude/roughness in mm, support, correspondence reliability."""
        batch, candidates = state.shape[:2]
        count, side, margin, depths = batch * candidates, 32, 8, 9
        current = state.reshape(count, 12)
        reflected = reflection.reshape(count).bool()
        centre, frame, basis = full_frame_state_to_components(current)
        edges = frame[..., :, :2] @ basis
        axis = (torch.arange(side + 2 * margin, device=state.device,
                             dtype=state.dtype) + .5 - margin) / side
        yy, xx = torch.meshgrid(axis - .5 / source_shape[0],
                                axis - .5 / source_shape[1], indexing='ij')
        chart = torch.stack((
            torch.where(reflected[:, None, None],
                        (source_shape[1] - 1) / source_shape[1] - xx, xx),
            yy.expand(count, -1, -1)), -1)
        base = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi', edges, chart - .5)
        z = torch.linspace(-6000., 6000., depths, device=state.device, dtype=state.dtype)
        offsets = torch.as_tensor(offsets, device=state.device, dtype=state.dtype)
        weights = torch.as_tensor(weights, device=state.device, dtype=state.dtype)
        if offsets.ndim == 1:
            offsets = offsets[None].expand(batch, -1)
        if weights.ndim == 1:
            weights = weights[None].expand(batch, -1)
        offsets = offsets[:, None].expand(-1, candidates, -1).reshape(count, -1)
        weights = weights[:, None].expand(-1, candidates, -1).reshape(count, -1)
        world = base[:, None, None] + (z[None, :, None] + offsets[:, None, :])[
            ..., None, None, None] * frame[:, None, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world.reshape(count * depths, offsets.shape[-1],
                                 side + 2 * margin, side + 2 * margin, 3),
            (0., 0., 0.), (25., 25., 25.),
            weights[:, None].expand(-1, depths, -1).reshape(count * depths, -1))
        slab = rendered.reshape(count, depths, 2, side + 2 * margin,
                                side + 2 * margin).permute(0, 2, 1, 3, 4)
        support = slab[:, 1:2].clamp(0, 1)
        atlas_pair = torch.cat((slab[:, :1] / support.clamp_min(1e-4), support), 1)
        key = F.normalize(self.atlas(atlas_pair), dim=1)
        query = F.normalize(self.image(F.adaptive_avg_pool2d(
            image_feature, (side, side))), dim=1)
        query = query[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            count, 16, side, side)

        temperature = self.log_temperature.exp().clamp(2, 20)
        coarse_query = F.normalize(F.avg_pool2d(query, 2), dim=1)
        coarse_key = F.normalize(F.avg_pool3d(key, (1, 2, 2)), dim=1)
        coarse_support = F.avg_pool3d(support, (1, 2, 2))
        match_maps = []
        for q, k, tissue, radius, origin in (
            (query, key, support, 2, margin),
            (coarse_query, coarse_key, coarse_support, 4, margin // 2),
        ):
            width = q.shape[-1]
            correlation, visible = [], []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = origin + dy, origin + dx
                    patch = k[:, :, :, y:y + width, x:x + width]
                    correlation.append((q[:, :, None] * patch).sum(1))
                    visible.append(tissue[:, 0, :, y:y + width, x:x + width])
            visible = torch.stack(visible, 2)
            scores = temperature * torch.stack(correlation, 2) - 4 * (1 - visible)
            probability = scores.flatten(1, 2).softmax(1)
            displacement = torch.linspace(-1, 1, 2 * radius + 1,
                                          device=state.device, dtype=state.dtype)
            dy, dx = torch.meshgrid(displacement, displacement, indexing='ij')
            zz = torch.linspace(-1, 1, depths, device=state.device, dtype=state.dtype)
            positions = torch.stack((
                dx.flatten()[None].expand(depths, -1),
                dy.flatten()[None].expand(depths, -1),
                zz[:, None].expand(-1, (2 * radius + 1) ** 2)), 0)
            moment = (probability[:, None] * positions.reshape(
                1, 3, -1, 1, 1)).sum(2)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(
                1, keepdim=True) / math.log(depths * (2 * radius + 1) ** 2)
            confidence = scores.flatten(1, 2).amax(1, keepdim=True) / temperature
            matched_support = (probability * visible.flatten(1, 2)).sum(1, keepdim=True)
            match_maps.append(F.interpolate(torch.cat((moment, entropy, confidence,
                matched_support), 1), (side, side), mode='bilinear', align_corners=False))
        image_axis = (torch.arange(side, device=state.device,
                                   dtype=state.dtype) + .5) / side - .5
        py, px = torch.meshgrid(image_axis, image_axis, indexing='ij')
        position = torch.stack((px, py))[None].expand(count, -1, -1, -1)
        reflection_map = reflected.to(state.dtype)[:, None, None, None].expand(
            -1, 1, side, side)
        if fit_summary is None:
            fit_summary = state.new_zeros(batch, candidates, 4)
        fit_summary = fit_summary.reshape(count, 4)
        fit_map = fit_summary[:, :, None, None].expand(-1, -1, side, side)
        spatial = self.spatial(torch.cat((
            query, key[:, :, depths // 2, margin:-margin, margin:-margin],
            *match_maps, position,
            reflection_map, fit_map), 1))
        descriptor = torch.cat((
            self.summary(spatial), centre / 10000, frame[..., :, 2],
            current[:, 9:11] - math.log(12000.), current[:, 11:12],
            reflected.to(state.dtype)[:, None], fit_summary), -1)
        limits = current.new_tensor((.35, .35, .35, 3000., 3000., 3000.,
                                     math.log(1.25), math.log(1.25), .2))
        update = self.pose(descriptor).tanh() * limits
        with torch.autocast(device_type=state.device.type, enabled=False):
            corrected = compose_full_frame_state(current, update.to(current.dtype))
        return {
            'state': corrected.reshape(batch, candidates, 12),
            'update': update.reshape(batch, candidates, 9),
            'quality_logit': self.quality(descriptor).reshape(batch, candidates),
            'spatial_evidence': self.mapper_feature(spatial).reshape(
                batch, candidates, 64, side, side),
        }
