"""Blind-candidate 2D/3D atlas matches, dustbin, and a single robust plane fit."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v7 import ResidualBlock


class AtlasSpatialFit139(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(69, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32),
        )
        self.atlas = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
        )
        self.visibility = nn.Conv2d(32, 1, 1)
        self.coarse_dustbin = nn.Linear(64, 1)
        self.fine_dustbin = nn.Linear(64, 1)
        self.summary = nn.Sequential(nn.Linear(9, 32), nn.GELU())
        self.gate = nn.Linear(32, 1)
        self.score = nn.Linear(32, 1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(12.)))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        axis = (torch.arange(16) + .5) / 16
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('fixed_query_xy', torch.stack((xx, yy), -1).reshape(256, 2),
                             persistent=False)
        zz, dy, dx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-2, 3),
                                    torch.arange(-2, 3), indexing='ij')
        self.register_buffer('local_offsets', torch.stack((zz, dy, dx), -1).reshape(75, 3),
                             persistent=False)

    def _match(self, image_coarse, image_fine, visibility, atlas_key, key_support,
               key_world, key_summary, query_xy):
        batch, queries = query_xy.shape[:2]
        count, channels, depths, side, _ = atlas_key.shape
        candidates = count // batch
        grid = query_xy[:, :, None] * 2 - 1
        coarse_query = F.grid_sample(image_coarse, grid, align_corners=False)[
            ..., 0].transpose(1, 2)
        fine_query = F.grid_sample(image_fine, grid, align_corners=False)[
            ..., 0].transpose(1, 2)
        visible_logit = F.grid_sample(visibility, grid, align_corners=False)[:, 0, :, 0]
        coarse_query = F.normalize(coarse_query, dim=-1)[:, None].expand(
            -1, candidates, -1, -1).reshape(count, queries, channels)
        fine_query = F.normalize(fine_query, dim=-1)[:, None].expand(
            -1, candidates, -1, -1).reshape(count, queries, channels)
        key = F.normalize(atlas_key, dim=1).permute(0, 2, 3, 4, 1).reshape(
            count, depths * side * side, channels)
        support = key_support.reshape(count, -1)
        world = key_world.reshape(count, -1, 3)
        context = key_summary[:, None].expand(-1, queries, -1)
        temperature = self.log_temperature.exp().clamp(5, 30)
        coarse_match = temperature * torch.einsum('nqc,nlc->nql', coarse_query, key)
        coarse_match = coarse_match - 8 * (1 - support[:, None])
        coarse_unmatched = self.coarse_dustbin(torch.cat((coarse_query, context), -1))
        coarse_logits = torch.cat((coarse_match, coarse_unmatched + math.log(key.shape[1])), -1)

        best = coarse_match.argmax(-1)
        depth = best.div(side * side, rounding_mode='floor')
        row = best.div(side, rounding_mode='floor').remainder(side)
        column = best.remainder(side)
        index = torch.stack((depth, row, column), -1)[:, :, None] + self.local_offsets
        index = torch.stack((index[..., 0].clamp(0, depths - 1),
                             index[..., 1].clamp(0, side - 1),
                             index[..., 2].clamp(0, side - 1)), -1)
        index = index[..., 0] * side * side + index[..., 1] * side + index[..., 2]
        local_key = torch.gather(key[:, None].expand(-1, queries, -1, -1), 2,
                                 index[..., None].expand(-1, -1, -1, channels))
        local_support = torch.gather(support[:, None].expand(-1, queries, -1), 2, index)
        local_world = torch.gather(world[:, None].expand(-1, queries, -1, -1), 2,
                                   index[..., None].expand(-1, -1, -1, 3))
        fine_match = temperature * (fine_query[:, :, None] * local_key).sum(-1)
        fine_match = fine_match - 8 * (1 - local_support)
        fine_unmatched = self.fine_dustbin(torch.cat((fine_query, context), -1))
        fine_logits = torch.cat((fine_match, fine_unmatched + math.log(75)), -1)
        return {
            'query_xy': query_xy,
            'visibility_logit': visible_logit,
            'coarse_logits': coarse_logits.reshape(batch, candidates, queries, -1),
            'fine_logits': fine_logits.reshape(batch, candidates, queries, 76),
            'fine_key_world_um': local_world.reshape(batch, candidates, queries, 75, 3),
            'fine_key_support': local_support.reshape(batch, candidates, queries, 75),
        }

    def forward(self, prediction, inputs, candidate_states, candidate_reflections,
                atlas, offsets, weights, atlas_intensity_enabled=True, query_xy=None):
        """Optional query_xy gives auxiliary logits only; fit/score use fixed queries."""
        batch, candidates = candidate_states.shape[:2]
        assert candidates <= 4
        count, side, depths = batch * candidates, 32, 7
        source = torch.cat((F.adaptive_avg_pool2d(prediction['feature'], (side, side)),
                            F.adaptive_avg_pool2d(inputs, (side, side))), 1)
        image = self.image(source)
        image = image + F.interpolate(F.adaptive_avg_pool2d(image, (4, 4)),
                                       size=(side, side), mode='bilinear', align_corners=False)
        image_coarse = F.avg_pool2d(image, 2)
        visibility = self.visibility(image)

        centre, frame, basis = full_frame_state_to_components(candidate_states.reshape(count, 12))
        edges = frame[:, :, :2] @ basis
        axis = (torch.arange(side, device=inputs.device, dtype=inputs.dtype) + .5) * 1.5 / side - .25
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        reflected = candidate_reflections.reshape(count).bool()
        chart = torch.stack((torch.where(reflected[:, None, None],
                             (inputs.shape[-1] - 1) / inputs.shape[-1] - xx, xx),
                             yy.expand(count, -1, -1)), -1)
        base = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi', edges, chart - .5)
        axial = torch.linspace(-3000., 3000., depths, device=inputs.device, dtype=inputs.dtype)
        section_offsets = offsets[:, None].expand(-1, candidates, -1).reshape(count, -1)
        section_weights = weights[:, None].expand(-1, candidates, -1).reshape(count, -1)
        world = base[:, None, None] + (axial[None, :, None] + section_offsets[:, None, :])[
            ..., None, None, None] * frame[:, None, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world.reshape(count * depths, section_offsets.shape[-1], side, side, 3),
            (0., 0., 0.), (25., 25., 25.),
            section_weights[:, None].expand(-1, depths, -1).reshape(count * depths, -1))
        rendered = rendered.reshape(count, depths, 2, side, side).permute(0, 2, 1, 3, 4)
        support = rendered[:, 1:2].clamp(0, 1)
        intensity = rendered[:, :1] / support.clamp_min(1e-4)
        if not atlas_intensity_enabled:
            intensity = torch.zeros_like(intensity)
        atlas_key = self.atlas(torch.cat((intensity, support), 1))
        atlas_key = atlas_key + F.interpolate(F.adaptive_avg_pool3d(
            atlas_key, (depths, 4, 4)), size=(depths, side, side),
            mode='trilinear', align_corners=False)
        atlas_summary = (atlas_key * support).flatten(2).sum(-1) / support.flatten(2).sum(
            -1).clamp_min(1)
        key_world = base[:, None] + axial[None, :, None, None, None] * frame[:, None, None, None, :, 2]
        key_chart = chart[:, None].expand(-1, depths, -1, -1, -1)
        fixed = self.fixed_query_xy[None].expand(batch, -1, -1)
        main = self._match(image_coarse, image, visibility, atlas_key, support,
                           key_world, atlas_summary, fixed)
        result = {
            **main,
            'coarse_key_world_um': key_world.reshape(batch, candidates, -1, 3),
            'coarse_key_support': support.reshape(batch, candidates, -1),
            'coarse_key_chart_xy': key_chart.reshape(batch, candidates, -1, 2),
        }

        probability = main['fine_logits'].flatten(0, 1).float().softmax(-1)
        matched_probability = probability[..., :75]
        matched_mass = matched_probability.sum(-1).clamp_min(1e-6)
        conditional = matched_probability / matched_mass[..., None]
        matched_world = (conditional[..., None] * main['fine_key_world_um'].flatten(0, 1)
                         ).sum(-2).float()
        match_entropy = -(conditional * conditional.clamp_min(1e-8).log()).sum(-1) / math.log(75)
        visible = main['visibility_logit'][:, None].expand(-1, candidates, -1).reshape(
            count, -1).float().sigmoid()
        confidence = visible * matched_mass * (.1 + .9 * (1 - match_entropy))
        fixed_branch = fixed[:, None].expand(-1, candidates, -1, -1).reshape(count, 256, 2)
        chart_x = torch.where(reflected[:, None],
                              (inputs.shape[-1] - 1) / inputs.shape[-1] - fixed_branch[..., 0],
                              fixed_branch[..., 0])
        fit_chart = torch.stack((chart_x, fixed_branch[..., 1]), -1)
        design = torch.cat((torch.ones_like(fit_chart[..., :1]), fit_chart - .5), -1).float()
        prior = torch.stack((centre, edges[:, :, 0], edges[:, :, 1]), -2).float()
        ridge = torch.diag(candidate_states.new_tensor((2., 1., 1.))).float()
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            def solve(site_weights):
                lhs = torch.einsum('nqi,nq,nqj->nij', design, site_weights, design) + ridge
                rhs = torch.einsum('nqi,nq,nqj->nij', design, site_weights, matched_world) + ridge @ prior
                return torch.linalg.solve(lhs, rhs)

            fitted = solve(confidence)
            residual = (design @ fitted - matched_world).norm(dim=-1)
            robust = (1500 / residual.clamp_min(1500)).detach()
            fitted = solve(confidence * robust)
            residual = (design @ fitted - matched_world).norm(dim=-1)
            centre_fit, u_fit, v_fit = fitted.unbind(-2)
            prior_ouv = torch.stack((centre - .5 * (edges[:, :, 0] + edges[:, :, 1]),
                                     edges[:, :, 0], edges[:, :, 1]), -2).float()
            fitted_ouv = torch.stack((centre_fit - .5 * (u_fit + v_fit), u_fit, v_fit), -2)
            delta = fitted_ouv - prior_ouv
            support_match = (conditional * main['fine_key_support'].flatten(0, 1)).sum(-1)
            coarse_dustbin = main['coarse_logits'].flatten(0, 1).float().softmax(-1)[..., -1]
            metrics = torch.stack((matched_mass.mean(-1), conditional.amax(-1).mean(-1),
                match_entropy.mean(-1), visible.mean(-1), support_match.mean(-1),
                (confidence * residual).sum(-1) / confidence.sum(-1).clamp_min(1e-6) / 1000,
                delta[:, 0].norm(dim=-1) / 1000,
                delta[:, 1:].norm(dim=-1).mean(-1) / 1000,
                coarse_dustbin.mean(-1)), -1)
            summary = self.summary(metrics)
            blend = torch.tanh(self.gate(summary))[..., None] * delta.clamp(-3000, 3000)
            corrected_ouv = prior_ouv + blend
            corrected = full_frame_state_from_components(*physical_ouv_to_frame(corrected_ouv))
            baseline = full_frame_state_from_components(*physical_ouv_to_frame(prior_ouv))
            result['corrected_state'] = (candidate_states.reshape(count, 12).float()
                                         + corrected - baseline).reshape(batch, candidates, 12)
            result['score_delta'] = (5 * torch.tanh(self.score(summary)[:, 0] / 5)).reshape(
                batch, candidates)
            result['fit_residual_um'] = ((confidence * residual).sum(-1) /
                                         confidence.sum(-1).clamp_min(1e-6)).reshape(batch, candidates)

        if query_xy is not None:
            result['auxiliary'] = self._match(image_coarse, image, visibility, atlas_key,
                                               support, key_world, atlas_summary, query_xy)
        return result
