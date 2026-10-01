"""Randomly initialized one-pass arbitrary-plane pose and tissue mapping.

The atlas is used for the differentiable training loss and display, not as an
input to the warp head. Mixture scores and uncertainty outputs are uncalibrated.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_joint_model_v7 import ResidualBlock
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out


class OneShotJointSliceModel(nn.Module):
    def __init__(self, modes=8, uncertainty_rank=4):
        super().__init__()
        self.modes, self.uncertainty_rank = modes, uncertainty_rank
        widths = (64, 128, 256, 320)
        self.encoder = nn.ModuleList()
        channels = 5
        for width in widths:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels, width, 5, stride=2, padding=2),
                nn.GroupNorm(8, width), nn.GELU(),
                ResidualBlock(width), ResidualBlock(width),
            ))
            channels = width
        self.pose = nn.Sequential(
            nn.Linear(widths[-1] * 4 * 4 + 12, 768), nn.GELU(),
            nn.Linear(768, modes * 21),
        )
        self.lateral = nn.ModuleList(nn.Conv2d(width, 64, 1) for width in widths)
        self.warp_shared = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 64, 1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.warp_condition = nn.Sequential(
            nn.Linear(13, 256), nn.GELU(), nn.Linear(256, 128),
        )
        self.warp = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.GELU(),
            nn.Conv2d(64, 64, 1), nn.GELU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )
        latent_size = 9 + 3 * 8 * 8
        self.uncertainty = nn.Linear(widths[-1] + 13, latent_size * (1 + uncertainty_rank))
        self.register_buffer('center_origin', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('center_scale', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('euclidean_scale', torch.tensor([6600., 4000., 5700., 1., 1., 1.]))
        self.register_buffer('rotation_grid', torch.linspace(0, math.pi, 257))
        nn.init.normal_(self.pose[-1].weight, std=.001)
        nn.init.zeros_(self.pose[-1].bias)
        nn.init.normal_(self.warp[-1].weight, std=1e-4)
        nn.init.zeros_(self.warp[-1].bias)
        nn.init.normal_(self.uncertainty.weight, std=.001)
        nn.init.zeros_(self.uncertainty.bias)
        with torch.no_grad():
            rotations, _ = torch.linalg.qr(torch.randn(modes, 3, 3))
            rotations[:, :, 2] *= torch.linalg.det(rotations)[:, None]
            self.pose[-1].bias.view(modes, 21)[:, 3:9].copy_(
                rotations[:, :, :2].transpose(1, 2).reshape(modes, 6)
            )

    def predict(self, inputs, context=None):
        if context is None:
            context = inputs.new_zeros(len(inputs), 12)
        pyramid, x = [], inputs
        for layer in self.encoder:
            x = layer(x)
            pyramid.append(x)
        z = self.pose(torch.cat((F.adaptive_avg_pool2d(x, 4).flatten(1), context), -1))
        z = z.reshape(-1, self.modes, 21)
        center = self.center_origin + z[..., :3] * self.center_scale
        state = torch.cat((center, z[..., 3:9],
                           math.log(12000.) + z[..., 9:11].clamp(-2.5, 2.),
                           z[..., 11:12]), -1)
        feature = self.lateral[-1](pyramid[-1])
        for level in (2, 1, 0):
            feature = self.lateral[level](pyramid[level]) + F.interpolate(
                feature, size=pyramid[level].shape[-2:], mode='bilinear', align_corners=False
            )
        return {
            'state': state,
            'std': (.005 + F.softplus(z[..., 12:18])).clamp_max(4.),
            'concentration': (.05 + F.softplus(z[..., 18])).clamp_max(500.),
            'log_mass': z[..., 19].log_softmax(-1),
            'reflection_logit': z[..., 20],
            'feature': self.warp_shared(feature),
            'global_feature': x.mean((-2, -1)),
            'calibrated': False,
        }

    def component_log_prob(self, prediction, target_state, reflection):
        state = prediction['state']
        euclidean = torch.cat((state[..., :3], state[..., 9:]), -1) / self.euclidean_scale
        truth = torch.cat((target_state[..., :3], target_state[..., 9:]), -1) / self.euclidean_scale
        std = prediction['std']
        normal_lp = (-.5 * ((euclidean - truth[:, None]) / std).square()
                     - std.log() - .5 * math.log(2 * math.pi)).sum(-1)
        frame = full_frame_state_to_components(state)[1]
        target_frame = full_frame_state_to_components(target_state)[1]
        trace = (frame * target_frame[:, None]).sum((-2, -1)).clamp(-1., 3.)
        kappa = prediction['concentration']
        angles = self.rotation_grid
        integrand = torch.exp(2 * kappa[..., None] * (angles.cos() - 1)) * (angles / 2).sin().square()
        log_z = ((2 / math.pi) * torch.trapezoid(integrand, angles, dim=-1)).log()
        rotation_lp = kappa * (trace - 3) - log_z
        reflection_lp = -F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'], reflection[:, None].expand_as(kappa).float(), reduction='none'
        )
        return prediction['log_mass'] + normal_lp + rotation_lp + reflection_lp

    def map(self, prediction, offsets, mode_index, reflection, image_shape):
        batch, count = mode_index.shape
        row = torch.arange(batch, device=mode_index.device)[:, None]
        state = prediction['state'][row, mode_index].reshape(-1, 12)
        reflected = reflection.reshape(-1).bool()
        condition = torch.cat((
            (state[:, :3] - self.center_origin) / self.center_scale,
            state[:, 3:9], state[:, 9:11] - math.log(12000.),
            state[:, 11:12], reflected[:, None].to(state.dtype),
        ), -1)
        feature = prediction['feature'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        scale, bias = self.warp_condition(condition).chunk(2, -1)
        feature = feature * (1 + .25 * scale.tanh()[..., None, None]) + bias[..., None, None]
        raw = F.interpolate(self.warp(feature), image_shape, mode='bilinear', align_corners=False)
        requested_local = 1000 * raw[:, :3].tanh()
        local, removed_affine = project_surface_affine_out(requested_local)
        center, frame, basis = full_frame_state_to_components(state)
        height, width = image_shape
        y, x = torch.meshgrid(
            torch.arange(height, device=state.device, dtype=state.dtype) / height,
            torch.arange(width, device=state.device, dtype=state.dtype) / width,
            indexing='ij',
        )
        s = torch.where(reflected[:, None, None], (width - 1) / width - x, x)
        st = torch.stack((s.expand(-1, height, -1), y.expand(len(state), -1, -1)), -1)
        plane = normalized_raster_to_ccf(
            center[:, None, None], frame[:, None, None], basis[:, None, None], st
        )
        surface = plane + torch.einsum('bij,bjhw->bhwi', frame, local)
        z = torch.as_tensor(offsets, device=state.device, dtype=state.dtype)
        if z.ndim == 1:
            z = z[None].expand(batch, -1)
        z = z[:, None].expand(-1, count, -1).flatten(0, 1)
        coordinates = surface[:, None] + z[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        global_feature = prediction['global_feature'][:, None].expand(-1, count, -1).flatten(0, 1)
        covariance = self.uncertainty(torch.cat((global_feature, condition), -1)).reshape(
            len(state), 9 + 3 * 8 * 8, 1 + self.uncertainty_rank
        )
        return {
            'state': state.reshape(batch, count, 12),
            'mode_index': mode_index,
            'reflection': reflection,
            'centre_surface_ccf_ap_dv_ml_um': surface.reshape(batch, count, height, width, 3),
            'coordinates': coordinates.reshape(batch, count, z.shape[-1], height, width, 3),
            'local_displacement_um': local.reshape(batch, count, 3, height, width),
            'removed_affine_local_um': removed_affine.reshape(batch, count, 3, 3),
            'correspondence_logit': raw[:, 3:4].reshape(batch, count, 1, height, width),
            'joint_std': (.001 + F.softplus(covariance[..., 0])).reshape(batch, count, -1),
            'joint_factor': covariance[..., 1:].reshape(batch, count, -1, self.uncertainty_rank),
            'calibrated': False,
        }

    def forward(self, inputs, offsets, context=None, mode_index=None, reflection=None, candidates=4):
        prediction = self.predict(inputs, context)
        if mode_index is None:
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            scores = (prediction['log_mass'][..., None] + log_reflection).flatten(1)
            selected = scores.topk(min(candidates, scores.shape[1]), -1).indices
            mode_index, reflection = selected // 2, selected % 2
        if mode_index.ndim == 1:
            mode_index, reflection = mode_index[:, None], reflection[:, None]
        mapped = self.map(prediction, offsets, mode_index, reflection, inputs.shape[-2:])
        mapped['prediction'] = prediction
        return mapped

    def atlas_fit_loss(self, inputs, mapped, atlas, weights, valid_mask, side=96):
        batch, count, samples, height, width = mapped['coordinates'].shape[:5]
        coordinates = mapped['coordinates'].flatten(0, 1).permute(0, 1, 4, 2, 3)
        coordinates = F.interpolate(coordinates.reshape(batch * count, samples * 3, height, width),
                                    (side, side), mode='bilinear', align_corners=False)
        coordinates = coordinates.reshape(batch * count, samples, 3, side, side).permute(0, 1, 3, 4, 2)
        weights = torch.as_tensor(weights, device=inputs.device, dtype=inputs.dtype)
        if weights.ndim == 1:
            weights = weights[None].expand(batch, -1)
        weights = weights[:, None].expand(-1, count, -1).flatten(0, 1)
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), weights
        )
        target = rendered[:, :1] / rendered[:, 1:2].clamp_min(1e-4)
        source = F.interpolate(inputs[:, :1], (side, side), mode='area')
        source = source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        valid = F.interpolate(valid_mask[:, None].float(), (side, side), mode='area')
        valid = valid[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        support = rendered[:, 1:2].clamp(0, 1)
        a = source - F.avg_pool2d(source, 9, 1, 4)
        b = target - F.avg_pool2d(target, 9, 1, 4)
        cross = F.avg_pool2d(a * b, 9, 1, 4)
        variance = F.avg_pool2d(a.square(), 9, 1, 4) * F.avg_pool2d(b.square(), 9, 1, 4)
        agreement = (cross / (variance + 1e-5).sqrt()).abs().clamp(0, 1)
        mismatch = ((1 - agreement) * valid * support + valid * (1 - support))
        return mismatch.sum((1, 2, 3)).div(valid.sum((1, 2, 3)).clamp_min(1)).reshape(batch, count)
