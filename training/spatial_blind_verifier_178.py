"""Shared spatial image/atlas verifier for the frozen 132 blind beam."""

import torch
from torch import nn
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.global_atlas_contrast_090 import render_atlas_planes_090


def permute_within_support_178(pairs, keys):
    """Scramble atlas intensity at every render without changing its support."""
    batch, branches = pairs.shape[:2]
    intensity = pairs[:, :, 0].flatten(0, 1).flatten(1)
    support = pairs[:, :, 1].flatten(0, 1).flatten(1)
    keys = keys.reshape(batch * branches, -1)
    permuted = intensity.clone()
    for row in range(len(intensity)):
        pixels = (support[row] > .01).nonzero()[:, 0]
        order = keys[row, pixels].argsort()
        permuted[row, pixels] = intensity[row, pixels[order]]
    return torch.stack((permuted.reshape(batch, branches, 96, 96), pairs[:, :, 1]), 2)


class SpatialBlindVerifier178(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(8, 32, 3, padding=1)
        self.res96 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(32, 32, 3, padding=1))
        self.down48 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.res48 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(64, 64, 3, padding=1))
        self.down24 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.res24 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(96, 96, 3, padding=1))
        self.head = nn.Sequential(nn.Linear((64 + 96) * 4 * 4, 128), nn.GELU(),
                                  nn.Linear(128, 10))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_prior_weight = nn.Parameter(torch.tensor(-1.5))
        self.register_buffer('update_scale', torch.tensor(
            [.12, .12, .12, 900., 900., 900., .08, .08, .06]))

    def forward(self, inputs, pairs):
        batch, branches = pairs.shape[:2]
        image = F.interpolate(inputs[:, :1], (96, 96), mode='area')[:, None].expand(
            -1, branches, -1, -1, -1)
        atlas = pairs[:, :, :1]
        support = pairs[:, :, 1:2]
        mask = (support > .5).to(image.dtype)
        count = mask.sum((-2, -1), keepdim=True).clamp_min(1)
        image_mean = (image * mask).sum((-2, -1), keepdim=True) / count
        atlas_mean = (atlas * mask).sum((-2, -1), keepdim=True) / count
        image_std = (((image - image_mean).square() * mask).sum((-2, -1), keepdim=True)
                     / count + 1e-4).sqrt()
        atlas_std = (((atlas - atlas_mean).square() * mask).sum((-2, -1), keepdim=True)
                     / count + 1e-4).sqrt()
        observed = ((image - image_mean) / image_std).clamp(-4, 4) * mask
        rendered = ((atlas - atlas_mean) / atlas_std).clamp(-4, 4) * mask
        local_mean_x = F.avg_pool2d(observed.flatten(0, 1), 9, 1, 4)
        local_mean_y = F.avg_pool2d(rendered.flatten(0, 1), 9, 1, 4)
        local_cov = F.avg_pool2d((observed * rendered).flatten(0, 1), 9, 1, 4)
        local_cov = local_cov - local_mean_x * local_mean_y
        local_var_x = (F.avg_pool2d(observed.square().flatten(0, 1), 9, 1, 4)
                       - local_mean_x.square()).clamp_min(0)
        local_var_y = (F.avg_pool2d(rendered.square().flatten(0, 1), 9, 1, 4)
                       - local_mean_y.square()).clamp_min(0)
        local_corr = (local_cov / (local_var_x * local_var_y + 1e-4).sqrt()
                      ).clamp(-1, 1).reshape(batch, branches, 1, 96, 96) * mask
        brush = F.interpolate(inputs[:, 1:3], (96, 96), mode='area')[:, None].expand(
            -1, branches, -1, -1, -1)
        pair = torch.cat((observed, rendered, support, local_corr,
                          (observed - rendered).abs(), (observed + rendered).abs(),
                          brush), 2).flatten(0, 1)
        x = F.gelu(self.stem(pair))
        x = F.gelu(x + self.res96(x))
        fine = F.gelu(self.down48(x))
        fine = F.gelu(fine + self.res48(fine))
        coarse = F.gelu(self.down24(fine))
        coarse = F.gelu(coarse + self.res24(coarse))
        features = torch.cat((F.adaptive_avg_pool2d(fine, 4).flatten(1),
                              F.adaptive_avg_pool2d(coarse, 4).flatten(1)), 1)
        raw = self.head(features).reshape(batch, branches, 10)
        q = raw[..., :9].tanh()
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.)
        return q, raw[..., 9]

    def actions(self, inputs, states, reflection, offsets, weights, atlas, prior, keys=None):
        with torch.no_grad():
            original_pairs = render_atlas_planes_090(
                atlas, states, reflection, offsets, weights, candidate_chunk=4)
        original_view = (permute_within_support_178(original_pairs, keys)
                         if keys is not None else original_pairs)
        original = [self(inputs, original_view[:, start:start + 4])
                    for start in range(0, states.shape[1], 4)]
        q = torch.cat([part[0] for part in original], 1)
        original_visual = torch.cat([part[1] for part in original], 1)
        corrected = compose_full_frame_state(states, q * self.update_scale)
        corrected_pairs = render_atlas_planes_090(
            atlas, corrected, reflection, offsets, weights, candidate_chunk=4)
        corrected_view = (permute_within_support_178(corrected_pairs, keys)
                          if keys is not None else corrected_pairs)
        corrected_visual = torch.cat([self(inputs, corrected_view[:, start:start + 4])[1]
                                      for start in range(0, states.shape[1], 4)], 1)
        prior_weight = F.softplus(self.log_prior_weight)
        base = prior_weight * (prior - prior.max(-1, keepdim=True).values)
        logits = torch.stack((base + original_visual, base + corrected_visual), -1)
        return {'state': corrected, 'logits': logits, 'q': q,
                'original_support': original_pairs[:, :, 1],
                'corrected_support': corrected_pairs[:, :, 1],
                'prior_weight': prior_weight}
