"""Candidate-conditioned patch matching with continuous local 3D correction."""
import torch
import torch.nn.functional as F
from torch import nn

from training.atlas_oriented_patch_025 import AtlasOrientedPatch025


class PoseFineMatch046(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = AtlasOrientedPatch025()
        self.pair = nn.Sequential(nn.Linear(512 + 25 + 4, 128), nn.GELU(),
                                  nn.Linear(128, 4))
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(self, image_patches, atlas_patches, relative_local, coarse_logit):
        lists, keys = relative_local.shape[:2]
        query_map = self.patch.shared[:-3](self.patch.image_stem(image_patches))
        atlas_map = torch.cat([self.patch.shared[:-3](self.patch.atlas_stem(chunk))
                               for chunk in atlas_patches.split(64)])
        query = F.normalize(self.patch.shared[-3:](query_map), dim=-1)
        atlas = F.normalize(self.patch.shared[-3:](atlas_map), dim=-1).reshape(lists, keys, 128)
        query_map = query_map[:, None].expand(-1, keys, -1, -1, -1).reshape_as(atlas_map)
        shifted = F.unfold(atlas_map, kernel_size=5, padding=2).reshape(
            lists * keys, 256, 25, 64)
        correlation = (query_map.flatten(2)[:, :, None] * shifted).mean((1, 3))
        query = query[:, None].expand(-1, keys, -1)
        coarse = ((coarse_logit - coarse_logit.mean(-1, keepdim=True)) /
                  coarse_logit.std(-1, keepdim=True).clamp_min(1.))
        features = torch.cat((query, atlas, (query - atlas).abs(), query * atlas,
                              correlation.reshape(lists, keys, 25),
                              relative_local / 6000, coarse[..., None]), -1)
        residual = self.pair(features)
        score = (query * atlas).sum(-1) / .1 + residual[..., 0]
        offset_local = 1500 * residual[..., 1:].tanh()
        return score, offset_local
