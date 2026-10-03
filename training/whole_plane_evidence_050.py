"""Spatial whole-plane evidence for the scratch-lineage arbitrary-plane model."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_joint_model_v7 import ResidualBlock
from training.arbitrary_plane_recurrent_model import local_correlation
from training.atlas_oriented_patch_025 import AtlasOrientedPatch025


class WholePlaneEvidence050(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = AtlasOrientedPatch025()
        self.image_reduce = nn.Conv2d(256, 64, 1)
        self.atlas_reduce = nn.Conv2d(256, 64, 1)
        self.score = nn.Sequential(
            nn.Conv2d(222, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            ResidualBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            ResidualBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(2048, 128), nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, image, state, reflection, atlas, offsets, weights):
        batch, candidates = state.shape[:2]
        side = 128
        image = F.interpolate(image, (side, side), mode='area')
        image_map = self.patch.shared[:12](self.patch.image_stem(image))
        image_map = self.image_reduce(image_map)
        y, x = torch.meshgrid((torch.arange(side, device=image.device, dtype=image.dtype) + .5) / side,
                              (torch.arange(side, device=image.device, dtype=image.dtype) + .5) / side,
                              indexing='ij')
        centre, frame, basis = full_frame_state_to_components(state)
        chart = torch.stack((torch.where(reflection[..., None, None].bool(), 1 - x, x),
                             y.expand(batch, candidates, -1, -1)), -1)
        surface = centre[..., None, None, :] + torch.einsum(
            'bkij,bkhwj->bkhwi', frame[..., :, :2] @ basis, chart - .5)
        axial = torch.as_tensor(offsets, device=image.device, dtype=image.dtype)
        masses = torch.as_tensor(weights, device=image.device, dtype=image.dtype)
        if axial.ndim == 1:
            axial = axial[None].expand(batch, -1)
        if masses.ndim == 1:
            masses = masses[None].expand(batch, -1)
        world = surface[:, :, None] + axial[:, None, :, None, None, None] * frame[..., :, 2][:, :, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world.flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
            masses[:, None].expand(-1, candidates, -1).flatten(0, 1))
        support = rendered[:, 1:2].clamp(0, 1)
        plane = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        atlas_map = self.patch.shared[:12](self.patch.atlas_stem(plane))
        atlas_map = self.atlas_reduce(atlas_map)
        image_map = image_map[:, None].expand(-1, candidates, -1, -1, -1).flatten(0, 1)
        source = image[:, None].expand(-1, candidates, -1, -1, -1).flatten(0, 1)
        source = F.adaptive_avg_pool2d(source[:, :1], image_map.shape[-2:])
        plane = F.adaptive_avg_pool2d(plane, image_map.shape[-2:])
        correlation = local_correlation(image_map, atlas_map, 2)
        xy = F.interpolate(torch.stack((x, y), 0)[None], image_map.shape[-2:], mode='bilinear',
                           align_corners=False).expand(batch * candidates, -1, -1, -1)
        evidence = torch.cat((image_map, atlas_map, (image_map - atlas_map).abs(),
                              correlation, source, plane, xy), 1)
        return self.score(evidence).reshape(batch, candidates), support.mean((1, 2, 3)).reshape(batch, candidates)
