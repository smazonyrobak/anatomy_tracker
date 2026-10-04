"""Spatial correspondence fit for the 089-initialized 083 matcher.

``forward(feature, state, reflection, atlas, offsets, weights, source_shape)``
returns B,K tensors: ``state`` (12), lower-is-better ``fit_energy`` (scalar), coarse/fine
``*_match_logits`` and ``*_match_support`` (729x16² / 225x32²),
``*_reliability_logit`` (16² / 32²), and ``*_match_ccf_um`` (H,W,3), plus
``coarse_fit_weight`` (16²) and B,1 ``fine_heldout_mask`` (32²).
Reliability should be supervised as *correct predicted match* against synthetic
CCF coordinates, including background, missing tissue, out-of-window and wrong
in-window negatives. No synthetic truth mask or mandatory brush enters inference.
The top-mode index, support masks, checkerboard split, and IRLS reweight are
discrete/detached; local-mode probabilities, ridge solve, bounded state, fine
rerender, and fit energy remain differentiable to matcher, reliability, and
input pose. Reflections stay discrete and use 083's finite-thickness renderer.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


class SpatialJointFit092(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.reliability = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.constant_(self.reliability[-1].bias, -2.)
        self.log_fit_temperature = nn.Parameter(torch.tensor(math.log(2.)))

    def _matches(self, logits, support, state, reflection, source_shape):
        batch, candidates, classes, side, _ = logits.shape
        radius = 4 if side == 16 else 2
        width = 2 * radius + 1
        index = torch.arange(classes, device=state.device)
        depth = index // (width * width)
        dy = index // width % width - radius
        dx = index % width - radius
        available = (support >= .5).any(2)
        masked = logits.masked_fill(support < .5, -1e4)
        probability = masked.softmax(2)
        peak = probability.argmax(2)
        near = ((depth[None, None, :, None, None] - depth[peak][:, :, None]).abs() <= 1) \
             & ((dy[None, None, :, None, None] - dy[peak][:, :, None]).abs() <= 1) \
             & ((dx[None, None, :, None, None] - dx[peak][:, :, None]).abs() <= 1)
        local = masked.masked_fill(~near, -1e4).softmax(2)
        mx = (local * dx[None, None, :, None, None]).sum(2)
        my = (local * dy[None, None, :, None, None]).sum(2)
        md = (local * depth[None, None, :, None, None]).sum(2)
        peak_probability = probability.gather(2, peak[:, :, None]).squeeze(2)
        peak_support = support.gather(2, peak[:, :, None]).squeeze(2)
        peak_logit = masked.gather(2, peak[:, :, None]).squeeze(2)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(2) / math.log(classes)
        features = torch.stack((peak_probability, entropy, peak_support,
            support.mean(2), torch.where(available, peak_logit / 20, 0),
            mx / radius, my / radius, (md - 4) / 4), 2)
        reliability = self.reliability(features.flatten(0, 1)).reshape(
            batch, candidates, side, side)

        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        height, width = source_shape
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / height
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        reflected = reflection.bool()[..., None, None]
        chart = torch.stack((torch.where(reflected, (width - 1) / width - xx, xx),
            yy.expand(batch, candidates, -1, -1)), -1)
        shift = torch.stack((torch.where(reflected, -mx, mx) / side, my / side), -1)
        world = (centre[..., None, None, :]
                 + torch.einsum('bkij,bkhwj->bkhwi', edges, chart - .5 + shift)
                 + 1500 * (md - 4)[..., None] * frame[..., None, None, :, 2])
        return world, chart, reliability, available, masked

    def forward(self, feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256)):
        coarse = self.matcher(feature, state, reflection, atlas, offsets, weights,
            source_shape=source_shape, match_only=True)
        coarse_logits = coarse['coarse_match_logits']
        coarse_support = coarse['coarse_match_support']
        match, chart, coarse_reliability, available, _ = self._matches(
            coarse_logits, coarse_support, state, reflection, source_shape)

        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        prior = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
        design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
        axis = torch.arange(16, device=state.device)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        fit_sites = ((yy + xx) % 2 == 0)[None, None]
        site_weight = coarse_reliability.sigmoid() * available * fit_sites
        ridge = torch.diag(state.new_tensor((4., .5, .5)))

        def solve(weight):
            lhs = torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, design) + ridge
            rhs = torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, match) + ridge @ prior
            return torch.linalg.solve(lhs, rhs)

        fitted = solve(site_weight)
        residual = (torch.einsum('bkhwi,bkij->bkhwj', design, fitted) - match).norm(dim=-1)
        robust = (1500 / residual.clamp_min(1500)).clamp_max(1).detach()
        fitted = solve(site_weight * robust)
        delta = fitted - prior
        centre_delta = delta[..., :1, :]
        centre_delta = centre_delta * (3000 / (3000 + centre_delta.norm(dim=-1, keepdim=True)))
        edge_delta = delta[..., 1:, :]
        edge_limit = .2 * torch.linalg.svdvals(edges)[..., -1, None, None]
        edge_delta = edge_delta * (edge_limit / (edge_limit
                                   + edge_delta.norm(dim=-1, keepdim=True)))
        fitted = prior + torch.cat((centre_delta, edge_delta), -2)
        fit_centre, u, v = fitted.unbind(-2)
        fitted_state = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((fit_centre - .5 * (u + v), u, v), -2)))

        fine = self.matcher(feature, fitted_state, reflection, atlas, offsets, weights,
            source_shape=source_shape, match_only=True)
        fine_logits = fine['fine_match_logits']
        fine_support = fine['fine_match_support']
        fine_match, _, fine_reliability, fine_available, masked = self._matches(
            fine_logits, fine_support, fitted_state, reflection, source_shape)
        classes = fine_logits.shape[2]
        index = torch.arange(classes, device=state.device)
        depth = index // 25
        dy = index // 5 % 5 - 2
        dx = index % 5 - 2
        _, fit_frame, fit_basis = full_frame_state_to_components(fitted_state)
        fit_edges = fit_frame[..., :, :2] @ fit_basis
        sign = torch.where(reflection.bool(), -1., 1.)[..., None]
        xy = torch.stack((sign * dx / 32, dy[None, None].expand_as(sign * dx) / 32), -1)
        offset = (torch.einsum('bkij,bkcj->bkci', fit_edges, xy)
                  + 1500 * (depth - 4)[None, None, :, None] * fit_frame[..., None, :, 2])
        near_weight = (-offset.square().sum(-1) / (2 * 1500 ** 2)).exp()
        near_probability = torch.einsum('bkchw,bkc->bkhw', masked.softmax(2), near_weight)
        supported_near = fine_reliability.sigmoid() * fine_available * near_probability
        axis = torch.arange(32, device=state.device)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        heldout = (((yy // 2 + xx // 2) % 2) == 1)[None, None].expand(
            state.shape[0], 1, -1, -1)
        energy = -(supported_near.clamp_min(1e-6).log() * heldout).sum((-2, -1)) \
                 / heldout.sum((-2, -1)).clamp_min(1)
        return {'state': fitted_state, 'fit_energy': energy,
            'coarse_match_logits': coarse_logits, 'coarse_match_support': coarse_support,
            'coarse_reliability_logit': coarse_reliability,
            'coarse_match_ccf_um': match, 'coarse_fit_weight': site_weight,
            'fine_match_logits': fine_logits, 'fine_match_support': fine_support,
            'fine_reliability_logit': fine_reliability,
            'fine_match_ccf_um': fine_match, 'fine_heldout_mask': heldout}
