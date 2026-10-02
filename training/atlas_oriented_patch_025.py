"""Scratch cross-contrast 2D descriptors for observed and oriented atlas patches."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)


class AtlasOrientedPatch025(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_stem = nn.Sequential(nn.Conv2d(5, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU())
        self.atlas_stem = nn.Sequential(nn.Conv2d(2, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU())
        self.shared = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128))

    def forward(self, image_patches, atlas_patches):
        return (F.normalize(self.shared(self.image_stem(image_patches)), dim=-1),
                F.normalize(self.shared(self.atlas_stem(atlas_patches)), dim=-1))


def atlas_surface_image(atlas, centre, state, offsets, weights):
    normal = full_frame_state_to_components(state)[1][..., 2]
    coordinates = centre[:, None] + offsets[:, :, None, None, None] * normal[:, None, None, None]
    rendered = render_finite_thickness_coordinate_grid(
        atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights)
    support = rendered[:, 1:2].clamp(0, 1)
    return torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)


def extract_patches(image, rows, pixel_yx, side=64):
    offset = torch.arange(side, device=image.device, dtype=image.dtype) - (side - 1) / 2
    yy, xx = torch.meshgrid(offset, offset, indexing='ij')
    y = pixel_yx[:, 0, None, None] + yy
    x = pixel_yx[:, 1, None, None] + xx
    grid = torch.stack((x * (2 / (image.shape[-1] - 1)) - 1,
                        y * (2 / (image.shape[-2] - 1)) - 1), -1)
    return F.grid_sample(image[rows], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
