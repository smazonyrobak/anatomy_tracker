"""Scratch-trained image/atlas agreement score for full-pose candidates."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_joint_model_v7 import ResidualBlock
from training.arbitrary_plane_recurrent_model import local_correlation


class StructuralPoseMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(5, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32), nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(), ResidualBlock(32))
        self.atlas = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32), nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(), ResidualBlock(32))
        self.score = nn.Sequential(
            nn.Conv2d(124, 64, 3, padding=1), nn.GroupNorm(8, 64),
            nn.GELU(), ResidualBlock(64),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96),
            nn.GELU(), ResidualBlock(96),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128),
            nn.GELU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1))

    def forward(self, image, rendered):
        batch, candidates = rendered.shape[:2]
        query = self.image(image)
        target = self.atlas(rendered.flatten(0, 1))
        query = query[:, None].expand(-1, candidates, -1, -1, -1).flatten(0, 1)
        support = F.interpolate(rendered.flatten(0, 1)[:, 1:2],
                                size=query.shape[-2:], mode='area')
        y, x = torch.meshgrid(torch.linspace(-1, 1, query.shape[-2], device=query.device),
                              torch.linspace(-1, 1, query.shape[-1], device=query.device),
                              indexing='ij')
        position = torch.stack((x, y))[None].expand(len(query), -1, -1, -1)
        evidence = torch.cat((query, target, (query - target).abs(),
                              local_correlation(query, target, 2), support, position), 1)
        return self.score(evidence).reshape(batch, candidates)
