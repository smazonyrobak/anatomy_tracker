"""Scratch-trained 2D tissue to 3D atlas point descriptors for global pose proposals."""
import torch
import torch.nn.functional as F
from torch import nn


class AtlasPointProposal024(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(5, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 64))
        self.atlas = nn.Sequential(
            nn.Conv3d(2, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv3d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, 64))

    def forward(self, image_patches, atlas_cubes):
        return F.normalize(self.image(image_patches), dim=-1), F.normalize(self.atlas(atlas_cubes), dim=-1)


def observed_patches(inputs, rows, pixel_yx, side=64):
    offset = torch.arange(side, device=inputs.device, dtype=inputs.dtype) - (side - 1) / 2
    yy, xx = torch.meshgrid(offset, offset, indexing='ij')
    y = pixel_yx[:, 0, None, None] + yy
    x = pixel_yx[:, 1, None, None] + xx
    grid = torch.stack((x * (2 / (inputs.shape[-1] - 1)) - 1,
                        y * (2 / (inputs.shape[-2] - 1)) - 1), -1)
    return F.grid_sample(inputs[rows], grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def atlas_cubes(atlas_100um, ccf_ap_dv_ml_um, side=25):
    offset = torch.arange(side, device=ccf_ap_dv_ml_um.device, dtype=ccf_ap_dv_ml_um.dtype) - (side - 1) / 2
    ap, dv, ml = torch.meshgrid(offset, offset, offset, indexing='ij')
    center = ccf_ap_dv_ml_um / 100 - .375
    depth, height, width = atlas_100um.shape[-3:]
    grid = torch.stack(((center[:, 2, None, None, None] + ml) * (2 / (width - 1)) - 1,
                        (center[:, 1, None, None, None] + dv) * (2 / (height - 1)) - 1,
                        (center[:, 0, None, None, None] + ap) * (2 / (depth - 1)) - 1), -1)
    return F.grid_sample(atlas_100um.expand(len(center), -1, -1, -1, -1), grid,
                         mode='bilinear', padding_mode='zeros', align_corners=True)
