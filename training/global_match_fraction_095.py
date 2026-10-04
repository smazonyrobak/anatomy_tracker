"""Whole-field readout of the frozen 094 candidate correspondence evidence."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class GlobalMatchFraction095(nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution = nn.Conv2d(450, 32, 1)
        self.evidence = nn.Conv2d(64, 16, 1)
        self.local = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.whole = nn.Sequential(
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.readout = nn.Sequential(nn.Linear(148, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.constant_(self.readout[-1].bias, -1.)

    def forward(self, fitted, feature=None):
        logits = fitted['fine_match_logits']
        support = fitted['fine_match_support']
        batch, candidates, _, side, _ = logits.shape
        available = support.amax(2)
        probability = logits.masked_fill(support < .5, -1e4).softmax(2)
        peak = probability.amax(2)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(2) / math.log(225)
        validity = fitted['validity_logit'].sigmoid()[:, None].expand(-1, candidates, -1, -1)
        axis = (torch.arange(side, device=logits.device, dtype=logits.dtype) + .5) / side - .5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        position = torch.stack((xx, yy), 0)[None].expand(batch * candidates, -1, -1, -1)
        ccf = fitted['fine_match_ccf_um']
        dx = torch.cat((ccf[:, :, :, 1:] - ccf[:, :, :, :-1],
                        torch.zeros_like(ccf[:, :, :, :1])), 3) / 1000
        dy = torch.cat((ccf[:, :, 1:] - ccf[:, :, :-1],
                        torch.zeros_like(ccf[:, :, :1])), 2) / 1000
        geometry = torch.cat((ccf / 15000, dx, dy), -1).permute(0, 1, 4, 2, 3).flatten(0, 1)
        distribution = torch.cat((logits.clamp(-40, 40) / 20, support), 2).flatten(0, 1)
        summaries = torch.stack((peak, entropy, available, validity,
                                 fitted['fine_correct_logit'].sigmoid()), 2).flatten(0, 1)
        local = self.local(torch.cat((self.distribution(distribution),
            self.evidence(fitted['spatial_evidence'].flatten(0, 1)),
            summaries, position, geometry), 1))
        tissue = (validity * available).flatten(0, 1)[:, None]
        pooled = (local * tissue).sum((-2, -1)) / tissue.sum((-2, -1)).clamp_min(1e-6)
        whole = self.whole(local).mean((-2, -1))
        scalars = torch.stack((fitted['quality'], fitted['coverage'],
            fitted['pose_correction_cost'].clamp(0, 10),
            (validity * available).mean((-2, -1))), -1).flatten(0, 1)
        quality_logit = self.readout(torch.cat((whole, pooled, scalars), -1))
        return {'quality_logit': quality_logit.reshape(batch, candidates)}
