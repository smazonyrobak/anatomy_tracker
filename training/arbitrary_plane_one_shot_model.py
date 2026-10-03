"""Randomly initialized one-pass arbitrary-plane pose and tissue mapping.

The optional atlas-conditioned warp compares the slice with a rendered candidate
plane; the original image-only path remains the default. Mixture scores and
uncertainty outputs are uncalibrated.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_from_components, full_frame_state_to_components,
    frame_to_rotation_6d, render_finite_thickness_coordinate_grid, so3_exp_map,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf, physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v7 import ResidualBlock
from training.arbitrary_plane_recurrent_model import local_correlation
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out


class OneShotJointSliceModel(nn.Module):
    def __init__(self, modes=8, uncertainty_rank=4, atlas_conditioning=False, fit_quality=False,
                 vector_refinement=False, candidate_ranking=False, fitted_ranking=False,
                 dense_coordinate=False, normal_anchor_count=0, anchor_specific_features=False):
        super().__init__()
        self.base_modes, self.normal_anchor_count = modes, normal_anchor_count
        self.anchor_specific_features = anchor_specific_features
        self.modes, self.uncertainty_rank = modes + normal_anchor_count, uncertainty_rank
        self.atlas_conditioning = atlas_conditioning
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
        if normal_anchor_count:
            generator = torch.Generator().manual_seed(20261003)
            candidates = F.normalize(torch.randn(4096, 3, generator=generator), dim=-1)
            selected = [0]
            distance = 1 - (candidates @ candidates[0]).abs()
            for _ in range(1, normal_anchor_count):
                selected.append(int(distance.argmax()))
                distance = torch.minimum(distance, 1 - (candidates @ candidates[selected[-1]]).abs())
            normals = candidates[selected]
            normals *= torch.where(normals.gather(1, normals.abs().argmax(-1, keepdim=True)) < 0, -1, 1)
            axis = F.one_hot(normals.abs().argmin(-1), 3).to(normals.dtype)
            horizontal = F.normalize(axis - (axis * normals).sum(-1, keepdim=True) * normals, dim=-1)
            vertical = torch.linalg.cross(normals, horizontal)
            self.register_buffer('normal_anchor_frames', torch.stack((horizontal, vertical, normals), -1))
            self.anchor_pose = nn.Sequential(
                nn.Linear(64 * 8 * 8 + 12, 768), nn.GELU(),
                nn.Linear(768, normal_anchor_count * 18),
            )
            self.anchor_global = nn.Sequential(
                nn.Linear(widths[-1] * 4 * 4 + 12, 384), nn.GELU(),
                nn.Linear(384, normal_anchor_count * 18),
            )
            nn.init.normal_(self.anchor_pose[-1].weight, std=.001)
            nn.init.zeros_(self.anchor_pose[-1].bias)
            nn.init.zeros_(self.anchor_global[-1].weight)
            nn.init.zeros_(self.anchor_global[-1].bias)
            with torch.no_grad():
                self.anchor_pose[-1].bias.view(normal_anchor_count, 18)[:, 16] = -2
        if dense_coordinate:
            self.dense_coordinate_head = nn.Sequential(
                nn.Conv2d(2 * widths[-1] + 2, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                ResidualBlock(128), nn.Conv2d(128, 4, 1),
            )
        self.lateral = nn.ModuleList(nn.Conv2d(width, 64, 1) for width in widths)
        if anchor_specific_features:
            assert normal_anchor_count
            self.anchor_encoder = copy.deepcopy(self.encoder)
            self.anchor_lateral = copy.deepcopy(self.lateral)
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
        if atlas_conditioning:
            self.atlas_encoder = nn.Sequential(
                nn.Conv2d(2, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
                ResidualBlock(64),
            )
            self.pair = nn.Sequential(
                nn.Conv2d(64 * 3 + 25 + 2, 64, 3, padding=1),
                nn.GroupNorm(8, 64), nn.GELU(), ResidualBlock(64),
                nn.Conv2d(64, 64, 1),
            )
            nn.init.zeros_(self.pair[-1].weight)
            nn.init.zeros_(self.pair[-1].bias)
            if fit_quality:
                self.fit_quality_head = nn.Sequential(
                    nn.Conv2d(64 * 3 + 25 + 2 + 4, 64, 3, padding=1),
                    nn.GroupNorm(8, 64), nn.GELU(),
                    nn.Conv2d(64, 32, 3, stride=2, padding=1),
                    nn.GroupNorm(8, 32), nn.GELU(),
                    nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 4 * 4, 1),
                )
            if candidate_ranking:
                self.candidate_matcher = nn.Sequential(
                    nn.Conv2d(64 * 3 + 25 + 2 + 1 + 13, 64, 3, padding=1),
                    nn.GroupNorm(8, 64), nn.GELU(), ResidualBlock(64),
                    nn.Conv2d(64, 32, 3, stride=2, padding=1),
                    nn.GroupNorm(8, 32), nn.GELU(),
                    nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 4 * 4, 1),
                )
            if fitted_ranking:
                self.fitted_matcher = nn.Sequential(
                    nn.Conv2d(241, 64, 3, padding=1),
                    nn.GroupNorm(8, 64), nn.GELU(), ResidualBlock(64),
                    nn.Conv2d(64, 32, 3, stride=2, padding=1),
                    nn.GroupNorm(8, 32), nn.GELU(),
                    nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 4 * 4, 1),
                )
            if vector_refinement:
                self.pose_refiner = nn.Sequential(
                    nn.Conv2d(229, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
                    ResidualBlock(64),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                    ResidualBlock(128), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 10),
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
        if hasattr(self, 'pose_refiner'):
            nn.init.zeros_(self.pose_refiner[-1].weight)
            nn.init.zeros_(self.pose_refiner[-1].bias)
        if hasattr(self, 'candidate_matcher'):
            nn.init.zeros_(self.candidate_matcher[-1].weight)
            nn.init.zeros_(self.candidate_matcher[-1].bias)
        if hasattr(self, 'fitted_matcher'):
            nn.init.zeros_(self.fitted_matcher[-1].weight)
            nn.init.zeros_(self.fitted_matcher[-1].bias)
        if hasattr(self, 'dense_coordinate_head'):
            nn.init.normal_(self.dense_coordinate_head[-1].weight, std=.001)
            nn.init.zeros_(self.dense_coordinate_head[-1].bias)
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
        z = z.reshape(-1, self.base_modes, 21)
        center = self.center_origin + z[..., :3] * self.center_scale
        state = torch.cat((center, z[..., 3:9],
                           math.log(12000.) + z[..., 9:11].clamp(-2.5, 2.),
                           z[..., 11:12]), -1)
        feature = self.lateral[-1](pyramid[-1])
        for level in (2, 1, 0):
            feature = self.lateral[level](pyramid[level]) + F.interpolate(
                feature, size=pyramid[level].shape[-2:], mode='bilinear', align_corners=False
            )
        if self.normal_anchor_count:
            if self.anchor_specific_features:
                anchor_pyramid, anchor_x = [], inputs
                for layer in self.anchor_encoder:
                    anchor_x = layer(anchor_x)
                    anchor_pyramid.append(anchor_x)
                anchor_feature = self.anchor_lateral[-1](anchor_pyramid[-1])
                for level in (2, 1, 0):
                    anchor_feature = self.anchor_lateral[level](anchor_pyramid[level]) + F.interpolate(
                        anchor_feature, size=anchor_pyramid[level].shape[-2:],
                        mode='bilinear', align_corners=False)
            else:
                anchor_feature, anchor_x = feature, x
            anchors = (self.anchor_pose(torch.cat((
                F.adaptive_avg_pool2d(anchor_feature, 8).flatten(1), context), -1))
                + self.anchor_global(torch.cat((
                    F.adaptive_avg_pool2d(anchor_x, 4).flatten(1), context), -1))).reshape(
                        -1, self.normal_anchor_count, 18)
            tilt = torch.cat((.65 * anchors[..., 3:5].tanh(),
                              torch.zeros_like(anchors[..., 5:6])), -1)
            roll = torch.cat((torch.zeros_like(anchors[..., 3:5]),
                              math.pi * anchors[..., 5:6].tanh()), -1)
            frame = (self.normal_anchor_frames[None] @ so3_exp_map(tilt) @ so3_exp_map(roll))
            anchor_state = torch.cat((
                self.center_origin + anchors[..., :3] * self.center_scale,
                frame_to_rotation_6d(frame),
                math.log(12000.) + anchors[..., 6:8].clamp(-2.5, 2.),
                anchors[..., 8:9]), -1)
            state = torch.cat((state, anchor_state), 1)
            std = torch.cat(((.005 + F.softplus(z[..., 12:18])).clamp_max(4.),
                             (.005 + F.softplus(anchors[..., 9:15])).clamp_max(4.)), 1)
            concentration = torch.cat(((.05 + F.softplus(z[..., 18])).clamp_max(500.),
                                       (.05 + F.softplus(anchors[..., 15])).clamp_max(500.)), 1)
            log_mass = torch.cat((z[..., 19], anchors[..., 16]), 1).log_softmax(-1)
            reflection_logit = torch.cat((z[..., 20], anchors[..., 17]), 1)
        else:
            std = (.005 + F.softplus(z[..., 12:18])).clamp_max(4.)
            concentration = (.05 + F.softplus(z[..., 18])).clamp_max(500.)
            log_mass = z[..., 19].log_softmax(-1)
            reflection_logit = z[..., 20]
        result = {
            'state': state,
            'std': std,
            'concentration': concentration,
            'log_mass': log_mass,
            'reflection_logit': reflection_logit,
            'feature': self.warp_shared(feature),
            'global_feature': x.mean((-2, -1)),
            'calibrated': False,
        }
        if hasattr(self, 'dense_coordinate_head'):
            yy, xx = torch.meshgrid(
                (torch.arange(x.shape[-2], device=x.device, dtype=x.dtype) + .5) / x.shape[-2]
                - .5 / inputs.shape[-2],
                (torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) + .5) / x.shape[-1]
                - .5 / inputs.shape[-1], indexing='ij')
            position = torch.stack((xx, yy), 0)[None].expand(len(x), -1, -1, -1)
            result['dense_coordinate'] = self.dense_coordinate_head(
                torch.cat((x, x.mean((-2, -1), keepdim=True).expand_as(x), position), 1))
        return result

    def dense_coordinate_plane(self, prediction, source_side=256):
        field = prediction['dense_coordinate']
        batch, _, height, width = field.shape
        y, x = torch.meshgrid((torch.arange(height, device=field.device) + .5) / height - .5 / source_side,
                              (torch.arange(width, device=field.device) + .5) / width - .5 / source_side,
                              indexing='ij')
        design = torch.stack((torch.ones_like(x), x, y), -1).reshape(-1, 3)
        weight = field[:, 3].sigmoid().flatten(1).clamp_min(1e-6)
        coordinates = (self.center_origin[None, :, None, None]
                       + field[:, :3] * self.center_scale[None, :, None, None])
        coordinates = coordinates.flatten(2).transpose(1, 2)
        gram = torch.einsum('ni,bn,nj->bij', design, weight, design)
        right = torch.einsum('ni,bn,bnj->bij', design, weight, coordinates)
        coefficients = torch.linalg.solve(
            gram + 1e-6 * torch.eye(3, device=field.device, dtype=field.dtype)[None], right)
        return full_frame_state_from_components(*physical_ouv_to_frame(coefficients))

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

    def score_candidates(self, prediction, offsets, weights, atlas, use_atlas=True,
                         side=64, chunk=4, image_shape=(256, 256)):
        batch = prediction['state'].shape[0]
        device = prediction['state'].device
        source = F.adaptive_avg_pool2d(prediction['feature'], (side, side))
        y, x = torch.meshgrid(torch.arange(side, device=device) / side,
                              torch.arange(side, device=device) / side, indexing='ij')
        xy = torch.stack((x, y))[None]
        z = torch.as_tensor(offsets, device=device, dtype=source.dtype)
        w = torch.as_tensor(weights, device=device, dtype=source.dtype)
        if z.ndim == 1:
            z = z[None].expand(batch, -1)
        if w.ndim == 1:
            w = w[None].expand(batch, -1)
        def score_chunk(branches, all_states, image_source):
            count = len(branches)
            state = all_states[:, branches // 2].reshape(-1, 12)
            reflected = (branches % 2)[None].expand(batch, -1).reshape(-1).bool()
            condition = torch.cat(((state[:, :3] - self.center_origin) / self.center_scale,
                                   state[:, 3:9], state[:, 9:11] - math.log(12000.),
                                   state[:, 11:12], reflected[:, None].to(state.dtype)), -1)
            if use_atlas:
                center, frame, basis = full_frame_state_to_components(state)
                sx = torch.where(reflected[:, None, None],
                                 (image_shape[1] - 1) / image_shape[1] - x, x)
                chart = torch.stack((sx.expand(-1, side, -1),
                                     y.expand(len(state), -1, -1)), -1)
                plane = normalized_raster_to_ccf(
                    center[:, None, None], frame[:, None, None], basis[:, None, None], chart)
                axial = z[:, None].expand(-1, count, -1).flatten(0, 1)
                masses = w[:, None].expand(-1, count, -1).flatten(0, 1)
                coordinates = plane[:, None] + axial[:, :, None, None, None] * frame[:, None, None, None, :, 2]
                rendered = render_finite_thickness_coordinate_grid(
                    atlas, coordinates, (0., 0., 0.), (25., 25., 25.), masses)
                support = rendered[:, 1:2]
                atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
            else:
                support = image_source.new_zeros(batch * count, 1, side, side)
                atlas_pair = image_source.new_zeros(batch * count, 2, side, side)
            target = self.atlas_encoder(atlas_pair)
            image = image_source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
            evidence = torch.cat((image, target, (image - target).abs(),
                                  local_correlation(image, target, 2),
                                  xy.expand(batch * count, -1, -1, -1), support,
                                  condition[..., None, None].expand(-1, -1, side, side)), 1)
            return self.candidate_matcher(evidence).reshape(batch, count)

        scores = []
        for first in range(0, self.modes * 2, chunk):
            branches = torch.arange(first, min(first + chunk, self.modes * 2), device=device)
            if torch.is_grad_enabled():
                scores.append(checkpoint(score_chunk, branches, prediction['state'], source,
                                         use_reentrant=False))
            else:
                scores.append(score_chunk(branches, prediction['state'], source))
        match = torch.cat(scores, -1)
        reflection_log = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                      F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = (prediction['log_mass'][..., None] + reflection_log).flatten(1)
        return prior + match

    def map(self, prediction, offsets, mode_index, reflection, image_shape, atlas=None, weights=None,
            return_refinement_feature=False, feature_side=None, source_shape=None):
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
        if feature_side is not None:
            feature = F.adaptive_avg_pool2d(feature, (feature_side, feature_side))
        scale, bias = self.warp_condition(condition).chunk(2, -1)
        feature = feature * (1 + .25 * scale.tanh()[..., None, None]) + bias[..., None, None]
        height, width = image_shape
        if self.atlas_conditioning:
            if atlas is None or weights is None:
                raise ValueError('atlas-conditioned warp requires the atlas and axial PSF weights')
            center, frame, basis = full_frame_state_to_components(state)
            fh, fw = feature.shape[-2:]
            if source_shape is None:
                fy = torch.arange(fh, device=state.device, dtype=state.dtype) / fh
                fx = torch.arange(fw, device=state.device, dtype=state.dtype) / fw
            else:
                fy = (torch.arange(fh, device=state.device, dtype=state.dtype) + .5) / fh - .5 / source_shape[0]
                fx = (torch.arange(fw, device=state.device, dtype=state.dtype) + .5) / fw - .5 / source_shape[1]
            yy, xx = torch.meshgrid(fy, fx, indexing='ij')
            source_width = width if source_shape is None else source_shape[1]
            sx = torch.where(reflected[:, None, None], (source_width - 1) / source_width - xx, xx)
            chart = torch.stack((sx.expand(-1, fh, -1), yy.expand(len(state), -1, -1)), -1)
            plane = normalized_raster_to_ccf(
                center[:, None, None], frame[:, None, None], basis[:, None, None], chart
            )
            z = torch.as_tensor(offsets, device=state.device, dtype=state.dtype)
            w = torch.as_tensor(weights, device=state.device, dtype=state.dtype)
            if z.ndim == 1:
                z = z[None].expand(batch, -1)
            if w.ndim == 1:
                w = w[None].expand(batch, -1)
            z = z[:, None].expand(-1, count, -1).flatten(0, 1)
            w = w[:, None].expand(-1, count, -1).flatten(0, 1)
            rigid = plane[:, None] + z[:, :, None, None, None] * frame[:, None, None, None, :, 2]
            rendered = render_finite_thickness_coordinate_grid(
                atlas, rigid, (0., 0., 0.), (25., 25., 25.), w
            )
            support = rendered[:, 1:2]
            atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
            target = self.atlas_encoder(atlas_pair)
            xy = torch.stack((xx, yy))[None].expand(len(state), -1, -1, -1)
            evidence = torch.cat((feature, target, (feature - target).abs(),
                                  local_correlation(feature, target, 2), xy), 1)
            feature = feature + self.pair(evidence)
        raw = F.interpolate(self.warp(feature), image_shape, mode='bilinear', align_corners=False)
        if hasattr(self, 'fit_quality_head'):
            raw_small = F.interpolate(raw, evidence.shape[-2:], mode='bilinear', align_corners=False)
            match = torch.cat((evidence, raw_small), 1)
            fit_energy = F.softplus(self.fit_quality_head(match).squeeze(-1))
        requested_local = 1000 * raw[:, :3].tanh()
        local, removed_affine = project_surface_affine_out(requested_local)
        center, frame, basis = full_frame_state_to_components(state)
        if source_shape is None:
            oy = torch.arange(height, device=state.device, dtype=state.dtype) / height
            ox = torch.arange(width, device=state.device, dtype=state.dtype) / width
        else:
            oy = (torch.arange(height, device=state.device, dtype=state.dtype) + .5) / height - .5 / source_shape[0]
            ox = (torch.arange(width, device=state.device, dtype=state.dtype) + .5) / width - .5 / source_shape[1]
        y, x = torch.meshgrid(oy, ox, indexing='ij')
        source_width = width if source_shape is None else source_shape[1]
        s = torch.where(reflected[:, None, None], (source_width - 1) / source_width - x, x)
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
        output = {
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
        if hasattr(self, 'fit_quality_head'):
            output['fit_energy'] = fit_energy.reshape(batch, count)
        if return_refinement_feature:
            local_small = F.interpolate(local, evidence.shape[-2:], mode='bilinear',
                                        align_corners=False) / 1000
            dx = F.pad((local_small[..., 1:] - local_small[..., :-1]).square().sum(1, keepdim=True),
                       (0, 1, 0, 0))
            dy = F.pad((local_small[..., 1:, :] - local_small[..., :-1, :]).square().sum(1, keepdim=True),
                       (0, 0, 0, 1))
            cost = torch.cat((local_small.square().sum(1, keepdim=True).sqrt(),
                              (dx + dy + 1e-8).sqrt()), 1)
            output['refinement_feature'] = torch.cat((evidence, raw_small, local_small,
                                                      support, cost), 1).reshape(
                batch, count, 229, *evidence.shape[-2:])
            output['atlas_pair'] = atlas_pair.reshape(batch, count, 2, *atlas_pair.shape[-2:])
        return output

    def score_fitted_candidates(self, inputs, prediction, mapped, atlas, weights):
        batch, count, samples, side = mapped['coordinates'].shape[:4]
        coordinates = mapped['coordinates'].flatten(0, 1)
        masses = torch.as_tensor(weights, device=inputs.device, dtype=inputs.dtype)
        if masses.ndim == 1:
            masses = masses[None].expand(batch, -1)
        masses = masses[:, None].expand(-1, count, -1).flatten(0, 1)
        rendered = render_finite_thickness_coordinate_grid(
            atlas, coordinates, (0., 0., 0.), (25., 25., 25.), masses)
        support = rendered[:, 1:2].clamp(0, 1)
        atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        target = self.atlas_encoder(atlas_pair)
        image = F.adaptive_avg_pool2d(prediction['feature'], (side, side))[:, None]
        image = image.expand(-1, count, -1, -1, -1).flatten(0, 1)
        source = F.interpolate(inputs[:, :1], (side, side), mode='area')[:, None]
        source = source.expand(-1, count, -1, -1, -1).flatten(0, 1)
        local = mapped['local_displacement_um'].flatten(0, 1) / 1000
        magnitude = local.square().sum(1, keepdim=True).sqrt()
        dx = F.pad((local[..., 1:] - local[..., :-1]).square().sum(1, keepdim=True),
                   (0, 1, 0, 0))
        dy = F.pad((local[..., 1:, :] - local[..., :-1, :]).square().sum(1, keepdim=True),
                   (0, 0, 0, 1))
        roughness = (dx + dy + 1e-8).sqrt()
        reliability = mapped['correspondence_logit'].flatten(0, 1).sigmoid()
        state = mapped['state'].flatten(0, 1)
        reflected = mapped['reflection'].flatten(0, 1).to(state.dtype)
        condition = torch.cat(((state[:, :3] - self.center_origin) / self.center_scale,
                               state[:, 3:9], state[:, 9:11] - math.log(12000.),
                               state[:, 11:12], reflected[:, None]), -1)
        yy, xx = torch.meshgrid(torch.arange(side, device=state.device) / side,
                                torch.arange(side, device=state.device) / side, indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(batch * count, -1, -1, -1)
        evidence = torch.cat((image, target, (image - target).abs(),
                              local_correlation(image, target, 2), xy, source,
                              atlas_pair[:, :1], support, local, magnitude, roughness,
                              reliability, condition[..., None, None].expand(-1, -1, side, side)), 1)
        learned = self.fitted_matcher(evidence).reshape(batch, count)
        penalty = (magnitude.square() + .1 * roughness.square()).mean((1, 2, 3)).reshape(batch, count)
        row = torch.arange(batch, device=state.device)[:, None]
        mode = mapped['mode_index']
        log_reflection = torch.where(mapped['reflection'].bool(),
                                     F.logsigmoid(prediction['reflection_logit'][row, mode]),
                                     F.logsigmoid(-prediction['reflection_logit'][row, mode]))
        prior = prediction['log_mass'][row, mode] + log_reflection
        return prior + learned - .1 * penalty

    def refine(self, feature, state):
        batch, count = state.shape[:2]
        raw = self.pose_refiner(feature.flatten(0, 1)).reshape(batch, count, 10)
        limits = raw.new_tensor((.9, .9, .9, 4000., 4000., 4000., .25, .25, .2))
        update = raw[..., :9].tanh() * limits
        return compose_full_frame_state(state, update), raw[..., 9], update

    def forward(self, inputs, offsets, context=None, mode_index=None, reflection=None, candidates=4,
                atlas=None, weights=None):
        prediction = self.predict(inputs, context)
        if mode_index is None:
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            scores = (prediction['log_mass'][..., None] + log_reflection).flatten(1)
            if self.normal_anchor_count and candidates > 1:
                old_count = min(2 * self.base_modes, max(1, candidates // 4))
                old = scores[:, :2 * self.base_modes].topk(old_count, -1).indices
                new = scores[:, 2 * self.base_modes:].topk(
                    min(candidates - old_count, 2 * self.normal_anchor_count), -1
                ).indices + 2 * self.base_modes
                selected = torch.cat((old, new), -1)
                selected = selected.gather(1, scores.gather(1, selected).argsort(-1, descending=True))
            else:
                selected = scores.topk(min(candidates, scores.shape[1]), -1).indices
            mode_index, reflection = selected // 2, selected % 2
        if mode_index.ndim == 1:
            mode_index, reflection = mode_index[:, None], reflection[:, None]
        mapped = self.map(prediction, offsets, mode_index, reflection, inputs.shape[-2:], atlas, weights)
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
