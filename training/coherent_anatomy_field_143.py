"""Spatially coherent source-to-atlas matches around one rendered candidate plane.

Input source is [B, 5, H, W]. Atlas slabs are [B, 2, 7, 64, 64],
with intensity then tissue support, rendered at seven uniformly spaced
through-plane positions centred on the candidate. The in-plane chart of the
atlas slabs must span the same [0, 1] chart as the source image. Outputs use
chart fractions for u/v and micrometres for the through-plane displacement.
"""

import torch
import torch.nn.functional as F
from torch import nn


class CoherentAnatomyField143(nn.Module):
    def __init__(self, radius=3):
        super().__init__()
        self.radius = radius
        self.source_fine = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2), nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.source_context = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.shared_source = nn.Conv2d(64, 32, 1)
        nn.init.zeros_(self.shared_source.weight)
        nn.init.zeros_(self.shared_source.bias)
        self.atlas = nn.Sequential(
            nn.Conv3d(2, 24, (3, 5, 5), padding=(1, 2, 2)),
            nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv3d(24, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        locations = 7 * (2 * radius + 1) ** 2
        self.cost_stem = nn.Sequential(
            nn.Conv2d(2 * locations + 32, 64, 1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.cost_context = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.match_logits = nn.Conv2d(64, locations + 1, 1)
        self.subvoxel = nn.Conv2d(64, 3, 1)
        self.visibility = nn.Conv2d(32, 1, 1)
        self.to_warp = nn.Conv2d(64, 64, 1)
        nn.init.zeros_(self.to_warp.weight)
        nn.init.zeros_(self.to_warp.bias)
        dy, dx = torch.meshgrid(torch.arange(-radius, radius + 1),
                                torch.arange(-radius, radius + 1), indexing='ij')
        self.register_buffer('local_xy', torch.stack((dx, dy), -1).reshape(-1, 2).float(),
                             persistent=False)
        self.register_buffer('local_depth', torch.arange(-3, 4).float(), persistent=False)

    def forward(self, source, atlas_slabs, slab_spacing_um, source_feature=None):
        """Return offset_chart [B,3,64,64], visibility, confidence, energy [B].

        offset_chart channels are delta-u, delta-v, and delta-normal in um.
        Visibility and confidence are uncalibrated [B,1,64,64] probabilities.
        Energy is an uncalibrated scalar per candidate; train it against
        support-matched wrong planes before using it for pose selection.
        """
        batch = source.shape[0]
        image = self.source_fine(F.interpolate(source, size=(128, 128),
                                               mode='bilinear', align_corners=False))
        if source_feature is not None:
            image = image + self.shared_source(F.interpolate(source_feature, (64, 64),
                mode='bilinear', align_corners=False))
        image = F.normalize(image + F.interpolate(self.source_context(image),
                            size=(64, 64), mode='bilinear', align_corners=False), dim=1)
        visibility = self.visibility(image).sigmoid()
        atlas = F.normalize(self.atlas(atlas_slabs), dim=1)
        width = 2 * self.radius + 1
        patches = F.unfold(atlas.permute(0, 2, 1, 3, 4).reshape(batch * 7, 32, 64, 64),
                           kernel_size=width, padding=self.radius)
        patches = patches.view(batch, 7, 32, width * width, 64, 64)
        similarity = (patches * image[:, None, :, None]).sum(2)
        supported = F.unfold(atlas_slabs[:, 1].reshape(batch * 7, 1, 64, 64),
                             kernel_size=width, padding=self.radius)
        supported = supported.view(batch, 7, width * width, 64, 64) > .5
        cost = torch.cat((similarity.flatten(1, 2), supported.flatten(1, 2).float(),
                          image), 1)
        context = self.cost_stem(cost)
        context = context + self.cost_context(context)
        logits = self.match_logits(context)
        logits = torch.cat((logits[:, :-1].masked_fill(~supported.flatten(1, 2), -1e4),
                            logits[:, -1:]), 1)
        probability = logits.softmax(1)
        matched = probability[:, :-1].view(batch, 7, width * width, 64, 64)
        matched_mass = matched.sum((1, 2)).clamp_min(1e-6)
        mean_xy = torch.einsum('bdlhw,lc->bchw', matched, self.local_xy) / matched_mass[:, None]
        mean_depth = torch.einsum('bdlhw,d->bhw', matched, self.local_depth) / matched_mass
        fine = .49 * self.subvoxel(context).tanh()
        offset_chart = torch.cat(((mean_xy + fine[:, :2]) / 64,
                                  ((mean_depth + fine[:, 2]) * slab_spacing_um)[:, None]), 1)
        offset_chart = offset_chart * matched_mass[:, None]
        confidence = matched.amax((1, 2))[:, None]
        mismatch = (matched * (1 - similarity).clamp_min(0)).sum((1, 2))
        mismatch = mismatch + probability[:, -1]
        offset_pixels = torch.cat((offset_chart[:, :2] * 64,
                                   offset_chart[:, 2:3] / slab_spacing_um), 1)
        smoothness = (offset_pixels[:, :, 1:] - offset_pixels[:, :, :-1]).square().mean(
            (1, 2, 3)) + (offset_pixels[:, :, :, 1:] - offset_pixels[:, :, :, :-1]).square(
            ).mean((1, 2, 3))
        deformation = offset_pixels.square().mean((1, 2, 3))
        tissue_weight = visibility[:, 0].detach()
        energy = ((mismatch * tissue_weight).sum((1, 2)) /
                  tissue_weight.sum((1, 2)).clamp_min(1)) + .02 * deformation + .05 * smoothness
        return {'offset_chart': offset_chart, 'visibility': visibility,
                'confidence': confidence, 'energy': energy,
                'match_logits': logits, 'spatial_evidence': self.to_warp(context)}
