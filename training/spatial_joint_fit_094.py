"""Candidate-conditioned 2-D/3-D correspondence verification and bounded plane fit.

The matcher and image encoder are trained from scratch in this model lineage. The
synthetic tissue mask and correspondence truth are supervision only; forward needs
neither a mask nor a reference plane. Match class order is the 083 (depth, dy, dx)
order, with finite-thickness atlas rendering inside that matcher.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


class SpatialJointFit094(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.distribution_projection = nn.ModuleDict({
            'coarse': nn.Conv2d(2 * 729, 24, 1),
            'fine': nn.Conv2d(2 * 225, 24, 1),
        })
        self.image_projection = nn.Conv2d(64, 24, 1)
        self.evidence_projection = nn.Conv2d(64, 24, 1)
        self.spatial_input = nn.Sequential(
            nn.Conv2d(83, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.spatial_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                          nn.GroupNorm(8, 64), nn.GELU(),
                          nn.Conv2d(64, 64, 3, padding=1),
                          nn.GroupNorm(8, 64)) for _ in range(2)
        ])
        self.correct_head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.correct_head.weight)
        nn.init.constant_(self.correct_head.bias, -1.5)
        self.validity_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.validity_head[-1].weight)
        nn.init.zeros_(self.validity_head[-1].bias)
        self.log_fit_temperature = nn.Parameter(torch.tensor(math.log(2.)))

    def _verify(self, name, match_logits, match_support, image_feature,
                spatial_evidence, state, reflection, source_shape):
        batch, candidates, classes, side, _ = match_logits.shape
        radius = 4 if name == 'coarse' else 2
        width = 2 * radius + 1
        index = torch.arange(classes, device=state.device)
        depth = index // (width * width)
        dy = index // width % width - radius
        dx = index % width - radius
        available = match_support.amax(2).clamp(0, 1)
        masked = match_logits.masked_fill(match_support < .5, -1e4)
        probability = masked.softmax(2)
        peak = probability.argmax(2)
        local_region = (
            ((depth[None, None, :, None, None] - depth[peak][:, :, None]).abs() <= 1)
            & ((dy[None, None, :, None, None] - dy[peak][:, :, None]).abs() <= 1)
            & ((dx[None, None, :, None, None] - dx[peak][:, :, None]).abs() <= 1))
        local = masked.masked_fill(~local_region, -1e4).softmax(2)
        mx = (local * dx[None, None, :, None, None]).sum(2)
        my = (local * dy[None, None, :, None, None]).sum(2)
        md = (local * depth[None, None, :, None, None]).sum(2)
        mode_probability = probability.gather(2, peak[:, :, None]).squeeze(2)
        mode_support = match_support.gather(2, peak[:, :, None]).squeeze(2)
        mode_logit = match_logits.gather(2, peak[:, :, None]).squeeze(2)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(2) / math.log(classes)
        near_centre = ((depth - 4).abs() <= 1) & (dy.abs() <= 1) & (dx.abs() <= 1)
        centre_probability = (probability * near_centre[None, None, :, None, None]).sum(2)

        height, width_px = source_shape
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / height
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / width_px
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        reflected = reflection.bool()[..., None, None]
        chart = torch.stack((torch.where(reflected, (width_px - 1) / width_px - xx, xx),
                             yy.expand(batch, candidates, -1, -1)), -1)
        shift = torch.stack((torch.where(reflected, -mx, mx) / side, my / side), -1)
        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        match_ccf = (centre[..., None, None, :]
                     + torch.einsum('bkij,bkhwj->bkhwi', edges, chart - .5 + shift)
                     + 1500 * (md - 4)[..., None] * frame[..., None, None, :, 2])

        summary = torch.stack((mode_probability, entropy, mode_support,
            match_support.mean(2), mode_logit / 20, mx / radius, my / radius,
            (md - 4) / 4, centre_probability), 2).flatten(0, 1)
        position = torch.stack((xx - .5, yy - .5), 0)[None].expand(
            batch * candidates, -1, -1, -1)
        distribution = torch.cat((match_logits.clamp(-40, 40) / 20,
                                  match_support), 2).flatten(0, 1)
        image = F.adaptive_avg_pool2d(image_feature, (side, side))[:, None].expand(
            -1, candidates, -1, -1, -1).flatten(0, 1)
        evidence = F.adaptive_avg_pool2d(spatial_evidence.flatten(0, 1), (side, side))
        features = torch.cat((self.distribution_projection[name](distribution),
            self.image_projection(image), self.evidence_projection(evidence),
            summary, position), 1)
        features = self.spatial_input(features)
        for block in self.spatial_blocks:
            features = F.gelu(features + block(features))
        correct_logit = self.correct_head(features).reshape(
            batch, candidates, side, side)
        return match_ccf, chart, correct_logit, available

    def forward(self, feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256)):
        batch, candidates = state.shape[:2]
        validity_logit = self.validity_head(F.adaptive_avg_pool2d(
            feature, (32, 32)))[:, 0]
        initial = self.matcher(feature, state, reflection, atlas, offsets, weights,
            source_shape=source_shape, return_match_logits=True)
        coarse_match, chart, coarse_correct, coarse_available = self._verify(
            'coarse', initial['coarse_match_logits'], initial['coarse_match_support'],
            feature, initial['spatial_evidence'], state, reflection, source_shape)

        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        prior = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
        design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
        axis = torch.arange(16, device=state.device)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        fit_sites = ((yy + xx) % 2 == 0)[None, None]
        coarse_validity = F.adaptive_avg_pool2d(validity_logit.sigmoid()[:, None],
                                                (16, 16))[:, 0, None]
        fit_weight = (coarse_correct.sigmoid() * coarse_validity
                      * coarse_available * fit_sites)
        ridge = torch.diag(state.new_tensor((8., 1., 1.)))
        robust = torch.ones_like(fit_weight)
        for iteration in range(2):
            weight = fit_weight * robust
            lhs = torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, design) + ridge
            rhs = torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, coarse_match) + ridge @ prior
            fitted = torch.linalg.solve(lhs, rhs)
            if iteration == 0:
                residual = (torch.einsum('bkhwi,bkij->bkhwj', design, fitted)
                            - coarse_match).norm(dim=-1)
                robust = (1000 / residual.clamp_min(1000)).detach()

        delta = fitted - prior
        centre_delta = delta[..., :1, :]
        centre_delta = centre_delta * (1500 / (1500 + centre_delta.norm(dim=-1, keepdim=True)))
        edge_delta = delta[..., 1:, :]
        edge_limit = .1 * prior[..., 1:, :].norm(dim=-1, keepdim=True)
        edge_delta = edge_delta * (edge_limit / (edge_limit + edge_delta.norm(dim=-1, keepdim=True)))
        fitted = prior + torch.cat((centre_delta, edge_delta), -2)
        fit_centre, u, v = fitted.unbind(-2)
        fitted_state = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((fit_centre - .5 * (u + v), u, v), -2)))
        correction_cost = (centre_delta.square().sum(-1).squeeze(-1) / 1500 ** 2
            + (edge_delta.square().sum(-1) / edge_limit.squeeze(-1).square()).mean(-1))

        final = self.matcher(feature, fitted_state, reflection, atlas, offsets, weights,
            source_shape=source_shape, return_match_logits=True)
        fine_match, _, fine_correct, fine_available = self._verify(
            'fine', final['fine_match_logits'], final['fine_match_support'],
            feature, final['spatial_evidence'], fitted_state, reflection, source_shape)
        axis = torch.arange(32, device=state.device)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        heldout = (((yy // 2 + xx // 2) % 2) == 1)[None, None].expand(
            batch, 1, -1, -1)
        tissue = validity_logit.sigmoid()[:, None] * heldout
        conditional = tissue * fine_available
        quality = (conditional * fine_correct.sigmoid()).sum((-2, -1)) / conditional.sum(
            (-2, -1)).clamp_min(1e-6)
        coverage = conditional.sum((-2, -1)) / tissue.sum((-2, -1)).clamp_min(1e-6)
        energy = (-quality.clamp_min(1e-4).log()
                  - .15 * coverage.clamp_min(.05).log()
                  + .05 * correction_cost)
        return {
            'state': fitted_state, 'fit_energy': energy,
            'coarse_match_logits': initial['coarse_match_logits'],
            'coarse_match_support': initial['coarse_match_support'],
            'coarse_match_ccf_um': coarse_match,
            'coarse_correct_logit': coarse_correct,
            'coarse_fit_weight': fit_weight,
            'fine_match_logits': final['fine_match_logits'],
            'fine_match_support': final['fine_match_support'],
            'fine_match_ccf_um': fine_match,
            'fine_correct_logit': fine_correct,
            'fine_heldout_mask': heldout,
            'validity_logit': validity_logit,
            'spatial_evidence': final['spatial_evidence'],
            'quality': quality, 'coverage': coverage,
            'pose_correction_cost': correction_cost,
        }
