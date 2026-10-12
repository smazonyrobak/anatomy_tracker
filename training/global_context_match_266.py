"""Whole-slice 2-D to whole-atlas 3-D correspondence, trained from scratch."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class GlobalContextMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(5, 48, 5, stride=2, padding=2), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 72, 3, stride=2, padding=1), nn.GroupNorm(8, 72), nn.GELU(),
            nn.Conv2d(72, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.position = nn.Linear(2, 96)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=96, nhead=4, dim_feedforward=192, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True), num_layers=2)
        self.query = nn.Linear(96, 96)
        self.matchable = nn.Linear(96, 1)
        self.atlas0 = nn.Sequential(
            nn.Conv3d(5, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv3d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.atlas1 = nn.Sequential(
            nn.Conv3d(48, 72, 3, padding=1), nn.GroupNorm(8, 72), nn.GELU(),
            nn.Conv3d(72, 72, 3, padding=1), nn.GroupNorm(8, 72), nn.GELU())
        self.atlas2 = nn.Sequential(
            nn.Conv3d(72, 72, 3, padding=1), nn.GroupNorm(8, 72), nn.GELU())
        self.atlas_out = nn.Sequential(
            nn.Conv3d(48 + 72 + 72, 96, 1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv3d(96, 96, 3, padding=1))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.07)))

    def encode_image(self, image):
        feature = self.image(image)
        axis = (torch.arange(16, device=image.device, dtype=feature.dtype) * 12 + 6) / 8
        y_sample, x_sample = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((2 * x_sample / (feature.shape[-1] - 1) - 1,
                            2 * y_sample / (feature.shape[-2] - 1) - 1), -1)
        feature = F.grid_sample(feature, grid[None].expand(len(image), -1, -1, -1),
                                mode='bilinear', padding_mode='border', align_corners=True)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 16, device=image.device),
                              torch.linspace(-1, 1, 16, device=image.device), indexing='ij')
        position = torch.stack((x, y), -1).reshape(1, 256, 2)
        tokens = feature.flatten(2).transpose(1, 2) + self.position(position)
        tokens = self.context(tokens)
        return F.normalize(self.query(tokens), dim=-1), self.matchable(tokens)[..., 0]

    def encode_atlas(self, atlas):
        a0 = self.atlas0(atlas[None])
        a1 = self.atlas1(F.avg_pool3d(a0, 2, ceil_mode=True))
        a2 = self.atlas2(F.avg_pool3d(a1, 2, ceil_mode=True))
        feature = self.atlas_out(torch.cat((a0,
            F.interpolate(a1, size=a0.shape[-3:], mode='trilinear', align_corners=False),
            F.interpolate(a2, size=a0.shape[-3:], mode='trilinear', align_corners=False)), 1))
        coarse = F.avg_pool3d(feature, 2, ceil_mode=True)
        return F.normalize(feature[0].flatten(1).T, dim=-1), \
            F.normalize(coarse[0].flatten(1).T, dim=-1)

    def similarity(self, query, atlas):
        return query @ atlas.T / self.log_temperature.exp().clamp(.03, .2)


def atlas_input_and_masks(atlas):
    pooled = F.avg_pool3d(atlas[None], 8)[0]
    support = F.max_pool3d(atlas[1:2][None], 8)[0, 0] > .05
    z, y, x = torch.meshgrid(*(torch.linspace(-1, 1, n, device=atlas.device)
        for n in pooled.shape[-3:]), indexing='ij')
    source = torch.cat((pooled[:1] / pooled[1:2].clamp_min(.01), pooled[1:2],
                        torch.stack((z, y, x))), 0)
    coarse_support = F.max_pool3d(support[None, None].float(), 2,
                                  ceil_mode=True)[0, 0] > 0
    return source, support.flatten(), coarse_support.flatten()
