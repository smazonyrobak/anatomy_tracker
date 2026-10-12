"""One-checkpoint frozen-pose and trainable spatial atlas-pair map."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


class SpatialPairJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_model = OneShotJointSliceModel(modes=16, normal_anchor_count=64)
        self.high = nn.Sequential(
            nn.Conv2d(299, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.low = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(160, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def local_normalize(image):
        mean = F.avg_pool2d(image, 9, stride=1, padding=4)
        variance = (F.avg_pool2d(image.square(), 9, stride=1, padding=4)
                    - mean.square()).clamp_min(1e-4)
        return ((image - mean) * variance.rsqrt()).clamp(-3, 3)

    def map_at_pose(self, inputs, state, reflection, offsets, weights, atlas):
        batch, _, source_height, source_width = inputs.shape
        side = 96
        center, frame, basis = full_frame_state_to_components(state)
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / source_height
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / source_width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        sx = torch.where(reflection[:, None, None].bool(),
                         (source_width - 1) / source_width - xx, xx)
        chart = torch.stack((sx.expand(batch, -1, -1), yy.expand(batch, -1, -1)), -1)
        rigid = normalized_raster_to_ccf(
            center[:, None, None], frame[:, None, None], basis[:, None, None], chart)
        coordinates = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        with torch.no_grad():
            rendered = render_finite_thickness_coordinate_grid(
                atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights)
            support = rendered[:, 1:2]
            target = rendered[:, :1] / support.clamp_min(1e-4)
            source = F.interpolate(inputs, (side, side), mode='area')
            source_norm = self.local_normalize(source[:, :1])
            target_norm = self.local_normalize(target)
            shifted = F.unfold(target_norm, 17, padding=8).reshape(batch, 289, side, side)
            shifted_support = F.unfold(support, 17, padding=8).reshape(batch, 289, side, side)
            cost = F.avg_pool2d(source_norm * shifted, 5, stride=1, padding=2).abs()
            cost = cost * shifted_support
            position = torch.stack((xx, yy), 0)[None].expand(batch, -1, -1, -1)
            evidence = torch.cat((cost, source, source_norm, target_norm, support, position), 1)
        high = self.high(evidence)
        low = self.low(high)
        raw = self.fuse(torch.cat((high, F.interpolate(
            low, size=high.shape[-2:], mode='bilinear', align_corners=False)), 1))
        local = 1500 * raw[:, :3].tanh()
        surface = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
        return {'centre_surface_ccf_ap_dv_ml_um': surface[:, None],
                'rigid_surface_ccf_ap_dv_ml_um': rigid,
                'local_displacement_um': local,
                'correspondence_logit': raw[:, 3:4],
                'atlas_support': support}
