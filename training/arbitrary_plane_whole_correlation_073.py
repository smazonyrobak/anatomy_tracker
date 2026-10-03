"""Learned whole-plane matching over roll, scale, and every in-plane shift."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class WholePlaneCorrelationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 16, 1),
        )
        self.atlas_projection = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 16, 1),
        )
        self.mask_head = nn.Conv2d(64, 1, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

        angles = torch.arange(12).repeat_interleave(3) * (math.pi / 6)
        scales = torch.tensor([0.65, 1.0, 1.5]).repeat(12)
        theta = torch.zeros(36, 2, 3)
        theta[:, 0, 0] = 4 * angles.cos() / scales
        theta[:, 0, 1] = 4 * angles.sin() / scales
        theta[:, 1, 0] = -4 * angles.sin() / scales
        theta[:, 1, 1] = 4 * angles.cos() / scales
        self.register_buffer('sampling_grids', F.affine_grid(theta, (36, 1, 64, 64), align_corners=False))
        self.register_buffer('shift_indices', torch.arange(-13, 14).remainder(128))

    def query_features(self, parent_prediction):
        source = F.adaptive_avg_pool2d(parent_prediction['feature'], 16)
        return self.query_projection(source), self.mask_head(source)

    def score(self, atlas_features, atlas_support, query_features, mask_logits):
        """Return [candidate, roll, scale, shift_y, shift_x] logits.

        Positive shift means the query template moves down/right on the atlas chart.
        Atlas charts span [-2, 2], so one shift bin is 0.0625 chart units.
        """
        query = F.grid_sample(query_features.expand(36, -1, -1, -1),
                              self.sampling_grids, align_corners=False)
        mask = F.grid_sample(mask_logits.sigmoid().expand(36, -1, -1, -1),
                             self.sampling_grids, align_corners=False).float()
        query = F.normalize(query.float(), dim=1) * mask
        atlas = F.normalize(self.atlas_projection(atlas_features).float(), dim=1)
        support = atlas_support.float()

        query_fft = torch.fft.rfft2(query, s=(128, 128))
        atlas_fft = torch.fft.rfft2(atlas * support, s=(128, 128))
        support_fft = torch.fft.rfft2(support, s=(128, 128))
        mask_fft = torch.fft.rfft2(mask, s=(128, 128))
        numerator = torch.fft.irfft2(torch.einsum('ochw,nchw->nohw', query_fft.conj(), atlas_fft),
                                     s=(128, 128))
        overlap = torch.fft.irfft2(mask_fft.conj()[None] * support_fft[:, None], s=(128, 128)).squeeze(2)
        shifts = self.shift_indices
        numerator = numerator.index_select(-2, shifts).index_select(-1, shifts)
        overlap = overlap.index_select(-2, shifts).index_select(-1, shifts).clamp_min(0)
        mass = mask.sum((-2, -1)).view(1, 36, 1, 1)
        fraction = (overlap / (mass + 1e-4)).clamp(1e-4, 1)
        scores = numerator / (overlap + 1e-4) + 0.3 * fraction.log()
        return (scores * self.logit_scale.exp()).reshape(-1, 12, 3, 27, 27)
