"""094 matcher with frozen local projections and trainable residual context."""
import torch
import torch.nn.functional as F
from torch import nn

from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083


class ImageContext099(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.context = nn.ModuleDict({
            'near': nn.Sequential(nn.Conv2d(80, 32, 3, padding=1),
                                  nn.GroupNorm(8, 32), nn.GELU()),
            'mid': nn.Sequential(nn.Conv2d(32, 48, 3, padding=1),
                                 nn.GroupNorm(8, 48), nn.GELU()),
            'far': nn.Sequential(nn.Conv2d(48, 64, 3, padding=1),
                                 nn.GroupNorm(8, 64), nn.GELU()),
            'up_mid': nn.Conv2d(64, 48, 3, padding=1),
            'up_near': nn.Conv2d(48, 32, 3, padding=1),
            'out': nn.Conv2d(32, 16, 1),
        })
        nn.init.zeros_(self.context['out'].weight)
        nn.init.zeros_(self.context['out'].bias)

    def forward(self, x):
        base = self.base(x)
        near = self.context['near'](torch.cat((
            F.avg_pool2d(x, 2), F.avg_pool2d(base, 2)), 1))
        mid = self.context['mid'](F.avg_pool2d(near, 2))
        far = self.context['far'](F.avg_pool2d(mid, 2))
        mid = F.gelu(mid + self.context['up_mid'](F.interpolate(
            far, mid.shape[-2:], mode='bilinear', align_corners=False)))
        near = F.gelu(near + self.context['up_near'](F.interpolate(
            mid, near.shape[-2:], mode='bilinear', align_corners=False)))
        residual = self.context['out'](F.interpolate(
            near, base.shape[-2:], mode='bilinear', align_corners=False))
        return base + residual


class AtlasContext099(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.context = nn.ModuleDict({
            'near': nn.Sequential(nn.Conv3d(18, 32, 3, padding=1),
                                  nn.GroupNorm(8, 32), nn.GELU()),
            'mid': nn.Sequential(nn.Conv3d(32, 48, 3, padding=1),
                                 nn.GroupNorm(8, 48), nn.GELU()),
            'far': nn.Sequential(nn.Conv3d(48, 64, 3, padding=1),
                                 nn.GroupNorm(8, 64), nn.GELU()),
            'up_mid': nn.Conv3d(64, 48, 3, padding=1),
            'up_near': nn.Conv3d(48, 32, 3, padding=1),
            'out': nn.Conv3d(32, 16, 1),
        })
        nn.init.zeros_(self.context['out'].weight)
        nn.init.zeros_(self.context['out'].bias)

    def forward(self, x):
        base = self.base(x)
        near = self.context['near'](torch.cat((
            F.avg_pool3d(x, (1, 2, 2)),
            F.avg_pool3d(base, (1, 2, 2))), 1))
        mid = self.context['mid'](F.avg_pool3d(near, (1, 2, 2)))
        far = self.context['far'](F.avg_pool3d(mid, (1, 2, 2)))
        mid = F.gelu(mid + self.context['up_mid'](F.interpolate(
            far, mid.shape[-3:], mode='trilinear', align_corners=False)))
        near = F.gelu(near + self.context['up_near'](F.interpolate(
            mid, near.shape[-3:], mode='trilinear', align_corners=False)))
        residual = self.context['out'](F.interpolate(
            near, base.shape[-3:], mode='trilinear', align_corners=False))
        return base + residual


class WholeSliceAtlasFeedback099(WholeSliceAtlasFeedback083):
    def __init__(self):
        super().__init__()
        self.image = ImageContext099(self.image)
        self.atlas = AtlasContext099(self.atlas)
