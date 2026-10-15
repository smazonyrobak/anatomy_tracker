"""C4-consistent one-pass plane mixture and tissue-to-atlas field, from scratch."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    frame_to_rotation_6d, full_frame_state_to_components, so3_exp_map,
)
from training.global_atlas_contrast_090 import rigid_points_090


def supported_affine_residual(field, support_logit):
    batch, _, height, width = field.shape
    y = (torch.arange(height, device=field.device, dtype=field.dtype)
         - (height - 1) / 2) / height
    x = (torch.arange(width, device=field.device, dtype=field.dtype)
         - (width - 1) / 2) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    design = torch.stack((torch.ones_like(xx), xx, yy), -1).reshape(-1, 3)
    weights = support_logit.sigmoid().detach().flatten(1).clamp_min(.01)
    gram = torch.einsum('bp,pi,pj->bij', weights, design, design)
    rhs = torch.einsum('bp,pi,bcp->bic', weights, design, field.flatten(2))
    ridge = torch.eye(3, device=field.device, dtype=field.dtype)[None] * (
        1e-4 * weights.sum(-1)[:, None, None])
    coefficients = torch.linalg.solve(gram + ridge, rhs)
    trend = torch.einsum('pi,bic->bcp', design, coefficients).reshape_as(field)
    return field - trend, coefficients.transpose(-2, -1)


class C4Lift(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, channels, 5, 5))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, image):
        weights = torch.stack([torch.rot90(self.weight, k, (-2, -1))
                               for k in range(4)], 1).flatten(0, 1)
        bias = self.bias.repeat_interleave(4)
        out = F.conv2d(image, weights, bias, padding=2)
        return out.reshape(len(image), -1, 4, *out.shape[-2:])


class C4Conv(nn.Module):
    def __init__(self, channels, width, kernel=3):
        super().__init__()
        self.spatial = nn.Parameter(torch.empty(channels, 1, kernel, kernel))
        self.mix = nn.Parameter(torch.empty(width, channels, 4))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.kaiming_normal_(self.spatial)
        nn.init.normal_(self.mix, std=1 / math.sqrt(4 * channels))
        self.padding = kernel // 2
        self.register_buffer('relative', (
            torch.arange(4)[None, :] - torch.arange(4)[:, None]) % 4)

    def forward(self, group):
        batch, channels, _, height, width = group.shape
        spatial = torch.stack([torch.rot90(self.spatial, k, (-2, -1))
                               for k in range(4)], 1).flatten(0, 1)
        out = F.conv2d(group.reshape(batch, channels * 4, height, width),
                       spatial, groups=channels * 4, padding=self.padding)
        mix = self.mix[:, :, self.relative].permute(0, 2, 1, 3).reshape(
            self.mix.shape[0] * 4, channels * 4, 1, 1)
        out = F.conv2d(out, mix, self.bias.repeat_interleave(4))
        return out.reshape(batch, self.mix.shape[0], 4, height, width)


class C4Block(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.first = C4Conv(channels, width)
        self.second = C4Conv(width, width)
        self.skip = C4Conv(channels, width, 1) if channels != width else None
        self.norm1 = nn.GroupNorm(8, 4 * width)
        self.norm2 = nn.GroupNorm(8, 4 * width)

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.first(x)
        out = F.gelu(self.norm1(out.flatten(1, 2)).reshape_as(out))
        out = self.second(out)
        out = self.norm2(out.flatten(1, 2)).reshape_as(out)
        return F.gelu(out + residual)


class C4JointSliceModel(nn.Module):
    def __init__(self, normal_modes=64, side=192):
        super().__init__()
        self.normal_modes, self.side = normal_modes, side
        self.lift = C4Lift(5, 64)
        self.lift_norm = nn.GroupNorm(8, 64 * 4)
        self.blocks = nn.ModuleList((C4Block(64, 64), C4Block(64, 128),
                                     C4Block(128, 192), C4Block(192, 256)))
        self.deep_blocks = nn.ModuleList((C4Block(192, 192), C4Block(256, 256)))
        self.coarse_to_fine = C4Conv(256, 64, 1)
        self.pose = nn.Sequential(nn.Linear(256 * 4 * 4 + 12, 768), nn.GELU(),
                                  nn.Linear(768, normal_modes * 18))
        self.warp_condition = nn.Sequential(nn.Linear(13, 128), nn.GELU(),
                                            nn.Linear(128, 128))
        self.warp = nn.Sequential(
            nn.Conv2d(69, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 4, 1),
        )
        self.register_buffer('center_origin', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('center_scale', torch.tensor([6600., 4000., 5700.]))
        generator = torch.Generator().manual_seed(20261015)
        candidates = F.normalize(torch.randn(4096, 3, generator=generator), dim=-1)
        selected = [0]
        distance = 1 - (candidates @ candidates[0]).abs()
        for _ in range(1, normal_modes):
            selected.append(int(distance.argmax()))
            distance = torch.minimum(distance,
                                     1 - (candidates @ candidates[selected[-1]]).abs())
        normals = candidates[selected]
        normals *= torch.where(normals.gather(1, normals.abs().argmax(-1, keepdim=True))
                               < 0, -1, 1)
        axis = F.one_hot(normals.abs().argmin(-1), 3).to(normals.dtype)
        horizontal = F.normalize(axis - (axis * normals).sum(-1, keepdim=True)
                                 * normals, dim=-1)
        vertical = torch.linalg.cross(normals, horizontal)
        self.register_buffer('normal_frames', torch.stack((horizontal, vertical, normals), -1))
        nn.init.normal_(self.pose[-1].weight, std=.001)
        nn.init.zeros_(self.pose[-1].bias)
        nn.init.normal_(self.warp[-1].weight, std=1e-4)
        nn.init.zeros_(self.warp[-1].bias)

    @staticmethod
    def canonical_chart(chart, group, side):
        x, y = chart.unbind(-1)
        c = (side - 1) / side
        k = group[..., None]
        sx = torch.where(k == 0, x, torch.where(k == 1, c - y,
                         torch.where(k == 2, c - x, y)))
        sy = torch.where(k == 0, y, torch.where(k == 1, x,
                         torch.where(k == 2, c - y, c - x)))
        return torch.stack((sx, sy), -1)

    def predict(self, image, context=None):
        if context is None:
            context = image.new_zeros(len(image), 12)
        x = self.lift(image)
        x = F.gelu(self.lift_norm(x.flatten(1, 2)).reshape_as(x))
        pyramid = []
        for stage, block in enumerate(self.blocks):
            x = F.avg_pool2d(x.flatten(1, 2), 2).reshape(
                len(x), x.shape[1], 4, x.shape[-2] // 2, x.shape[-1] // 2)
            x = block(x)
            if stage >= 2:
                x = self.deep_blocks[stage - 2](x)
            pyramid.append(x)
        fine = pyramid[0]
        coarse = self.coarse_to_fine(pyramid[-1])
        coarse = F.interpolate(coarse.flatten(1, 2), size=fine.shape[-2:],
                               mode='bilinear', align_corners=False).reshape_as(fine)
        fine = fine + coarse
        canonical_fine = torch.stack([torch.rot90(fine[:, :, k], -k, (-2, -1))
                                      for k in range(4)], 1)
        raw_half = F.avg_pool2d(image, 2)
        canonical_raw = torch.stack([torch.rot90(raw_half, -k, (-2, -1))
                                     for k in range(4)], 1)
        canonical_coarse = torch.stack([torch.rot90(x[:, :, k], -k, (-2, -1))
                                        for k in range(4)], 1)
        pooled = F.adaptive_avg_pool2d(canonical_coarse.flatten(0, 1), 4).flatten(1)
        z = self.pose(torch.cat((pooled, context[:, None].expand(-1, 4, -1)
                                .flatten(0, 1)), -1)).reshape(
                                    len(image), 4, self.normal_modes, 18)
        tilt = torch.cat((.65 * z[..., 3:5].tanh(),
                          torch.zeros_like(z[..., 5:6])), -1)
        roll = torch.cat((torch.zeros_like(z[..., 3:5]),
                          math.pi * z[..., 5:6].tanh()), -1)
        frame = self.normal_frames[None, None] @ so3_exp_map(tilt) @ so3_exp_map(roll)
        state = torch.cat((self.center_origin + z[..., :3] * self.center_scale,
                           frame_to_rotation_6d(frame),
                           math.log(12000.) + z[..., 6:8].clamp(-2.5, 2.),
                           z[..., 8:9]), -1).flatten(1, 2)
        return {
            'state': state,
            'log_mass': z[..., 16].flatten(1, 2).log_softmax(-1),
            'reflection_logit': z[..., 17].flatten(1, 2),
            'std': (.005 + F.softplus(z[..., 9:15])).flatten(1, 2),
            'concentration': (.05 + F.softplus(z[..., 15])).flatten(1, 2),
            'canonical_feature': canonical_fine,
            'canonical_raw': canonical_raw,
            'calibrated': False,
        }

    def map_sites(self, prediction, chart, mode_index, reflection, state=None):
        batch, count = mode_index.shape
        rows = torch.arange(batch, device=mode_index.device)[:, None]
        if state is None:
            state = prediction['state'][rows, mode_index]
        group = mode_index // self.normal_modes
        canonical = self.canonical_chart(chart[:, None].expand(-1, count, -1, -1),
                                         group, self.side)
        feature = prediction['canonical_feature'][rows, group].flatten(0, 1)
        condition = torch.cat(((state[..., :3] - self.center_origin)
                               / self.center_scale, state[..., 3:9],
                               state[..., 9:11] - math.log(12000.),
                               state[..., 11:12], reflection[..., None].to(state.dtype)), -1)
        scale, bias = self.warp_condition(condition.flatten(0, 1)).chunk(2, -1)
        feature = feature * (1 + .25 * scale.tanh()[..., None, None]) + bias[..., None, None]
        feature = torch.cat((feature, prediction['canonical_raw'][rows, group].flatten(0, 1)), 1)
        raw = F.interpolate(self.warp(feature), (self.side, self.side),
                            mode='bilinear', align_corners=False)
        local, removed_affine = supported_affine_residual(
            1000 * raw[:, :3].tanh(), raw[:, 3:4])
        grid = (2 * canonical.flatten(0, 1) + 1 / self.side - 1)[:, None]
        displacement = F.grid_sample(local, grid, align_corners=False,
                                     padding_mode='zeros').squeeze(2).transpose(1, 2)
        rigid = rigid_points_090(state.flatten(0, 1), reflection.flatten(),
                                 canonical.flatten(0, 1),
                                 (self.side, self.side))
        frame = full_frame_state_to_components(state.flatten(0, 1))[1]
        mapped = rigid + torch.einsum('bij,bnj->bni', frame, displacement)
        group_flat = group.flatten()
        row_flat = torch.arange(len(group_flat), device=group.device)
        observed_local = torch.stack([torch.rot90(local, k, (-2, -1))
                                      for k in range(4)], 1)[row_flat, group_flat]
        observed_logit = torch.stack([torch.rot90(raw[:, 3:4], k, (-2, -1))
                                      for k in range(4)], 1)[row_flat, group_flat]
        return {
            'coordinates': mapped.reshape(batch, count, -1, 3),
            'rigid_coordinates': rigid.reshape(batch, count, -1, 3),
            'local_displacement_um': observed_local.reshape(batch, count, 3, self.side, self.side),
            'removed_affine_local_um': removed_affine.reshape(batch, count, 3, 3),
            'correspondence_logit': observed_logit.reshape(batch, count, 1, self.side, self.side),
            'state': state,
            'mode_index': mode_index,
            'reflection': reflection,
            'calibrated': False,
        }

    def forward(self, image, chart, context=None, candidates=4):
        prediction = self.predict(image, context)
        reflection_mass = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                       F.logsigmoid(prediction['reflection_logit'])), -1)
        scores = (prediction['log_mass'][..., None] + reflection_mass).flatten(1)
        branches = scores.topk(candidates, -1).indices
        mapped = self.map_sites(prediction, chart, branches // 2, branches % 2)
        mapped['prediction'] = prediction
        mapped['log_mass'] = scores.gather(1, branches)
        return mapped
