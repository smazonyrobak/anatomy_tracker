"""Shared two-step atlas-conditioned full-frame pose correction."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


class CoarsePoseUpdater106(nn.Module):
    def __init__(self, matcher, validity_head):
        super().__init__()
        self.matcher = matcher
        self.validity_head = validity_head
        self.field_input = nn.Sequential(
            nn.Conv2d(2 * 729 + 64 + 4, 64, 1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.field_gates = nn.Conv2d(128, 128, 3, padding=1)
        self.field_proposal = nn.Conv2d(128, 64, 3, padding=1)
        self.offset_head = nn.Conv2d(64, 3, 1)
        self.reliability_head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)

    def forward(self, feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256), atlas_disabled=False):
        batch, candidates = state.shape[:2]
        side = 16
        validity_logit = self.validity_head(
            F.adaptive_avg_pool2d(feature, (32, 32)))[:, 0]
        validity = F.adaptive_avg_pool2d(
            validity_logit.sigmoid()[:, None], (side, side))[:, 0]
        image = F.adaptive_avg_pool2d(feature, (side, side))[:, None].expand(
            -1, candidates, -1, -1, -1).flatten(0, 1)
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side \
            - .5 / source_shape[0]
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side \
            - .5 / source_shape[1]
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        reflected = reflection.bool()[..., None, None]
        chart = torch.stack((torch.where(reflected,
            (source_shape[1] - 1) / source_shape[1] - xx, xx),
            yy.expand(batch, candidates, -1, -1)), -1)
        position = torch.stack((xx - .5, yy - .5))[None].expand(
            batch * candidates, -1, -1, -1)
        reflection_map = reflection.flatten().to(feature.dtype)[:, None, None, None].expand(
            -1, 1, side, side)
        validity_map = validity[:, None].expand(-1, candidates, -1, -1).flatten(
            0, 1)[:, None]
        axis = torch.arange(side, device=state.device)
        iy, ix = torch.meshgrid(axis, axis, indexing='ij')
        fit_sites = ((iy + ix) % 2 == 0)[None, None]
        ridge = torch.diag(state.new_tensor((8., 1., 1.)))
        hidden = image.new_zeros(batch * candidates, 64, side, side)
        states = [state]
        mapped_sites, reliability_logits, supports, predicted_offsets, match_logits = \
            [], [], [], [], []
        for _ in range(2):
            current = states[-1]
            matched = self.matcher(feature, current, reflection, atlas, offsets, weights,
                source_shape=source_shape, match_only=True)
            logits = matched['coarse_match_logits']
            support = matched['coarse_match_support']
            if atlas_disabled:
                logits = torch.zeros_like(logits)
            field = self.field_input(torch.cat((logits.flatten(0, 1).clamp(-40, 40) / 20,
                support.flatten(0, 1), image, validity_map, position, reflection_map), 1))
            update, reset = self.field_gates(torch.cat((field, hidden), 1)).sigmoid().chunk(2, 1)
            proposal = self.field_proposal(torch.cat((field, reset * hidden), 1)).tanh()
            hidden = (1 - update) * hidden + update * proposal
            raw = self.offset_head(hidden).reshape(batch, candidates, 3, side, side)
            offset = torch.cat((4 * raw[:, :, :2].tanh(),
                6000 * raw[:, :, 2:3].tanh()), 2)
            reliability = self.reliability_head(hidden).reshape(
                batch, candidates, side, side)

            centre, frame, basis = full_frame_state_to_components(current)
            edges = frame[..., :, :2] @ basis
            prior = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
            shift = torch.stack((torch.where(reflected, -offset[:, :, 0],
                offset[:, :, 0]) / side, offset[:, :, 1] / side), -1)
            mapped = (centre[..., None, None, :]
                + torch.einsum('bkij,bkhwj->bkhwi', edges, chart - .5 + shift)
                + offset[:, :, 2, ..., None] * frame[..., None, None, :, 2])
            design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
            fit_weight = (validity[:, None] * support.amax(2).clamp(0, 1)
                * reliability.sigmoid() * fit_sites)
            robust = torch.ones_like(fit_weight)
            for fit_iteration in range(2):
                weight = fit_weight * robust
                lhs = torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, design) + ridge
                rhs = (torch.einsum('bkhwi,bkhw,bkhwj->bkij', design, weight, mapped)
                    + ridge @ prior)
                fitted = torch.linalg.solve(lhs, rhs)
                if fit_iteration == 0:
                    residual = (torch.einsum('bkhwi,bkij->bkhwj', design, fitted)
                        - mapped).norm(dim=-1)
                    robust = (1000 / residual.clamp_min(1000)).detach()
            delta = fitted - prior
            centre_delta = delta[..., 0, :]
            centre_delta = centre_delta * (3000 / (3000 + centre_delta.norm(
                dim=-1, keepdim=True)))
            edge_delta = delta[..., 1:, :]
            edge_limit = .5 * prior[..., 1:, :].norm(dim=-1, keepdim=True)
            edge_delta = edge_delta * edge_limit / (edge_limit + edge_delta.norm(
                dim=-1, keepdim=True))
            fitted = prior + torch.cat((centre_delta[..., None, :], edge_delta), -2)
            fit_centre, u, v = fitted.unbind(-2)
            next_state = full_frame_state_from_components(*physical_ouv_to_frame(
                torch.stack((fit_centre - .5 * (u + v), u, v), -2)))
            states.append(next_state)
            mapped_sites.append(mapped)
            reliability_logits.append(reliability)
            supports.append(support)
            predicted_offsets.append(offset)
            match_logits.append(logits)
        return {
            'iteration_states': torch.stack(states, 1),
            'state': states[-1],
            'coarse_mapped_ccf_um': torch.stack(mapped_sites, 1),
            'coarse_reliability_logit': torch.stack(reliability_logits, 1),
            'coarse_match_support': torch.stack(supports, 1),
            'coarse_offsets_xy_cells_normal_um': torch.stack(predicted_offsets, 1),
            'coarse_match_logits': torch.stack(match_logits, 1),
            'validity_logit': validity_logit,
        }
