"""Candidate-conditioned 3-D atlas correspondence and full-plane correction."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


class CoarseAtlasPose148(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        # Support is only a key-validity mask; it never enters the feature score.
        self.atlas = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
        )
        self.visibility = nn.Conv2d(32, 1, 1)
        self.dustbin = nn.Linear(32, 1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, source, atlas, state, reflection, offsets, weights,
                atlas_intensity=True):
        batch, candidates = state.shape[:2]
        count, side, depths = batch * candidates, 24, 9
        height, width = source.shape[-2:]
        image = self.source(F.adaptive_avg_pool2d(source, (side, side)))
        query = F.normalize(image.flatten(2).transpose(1, 2), dim=-1)
        visibility_logits = self.visibility(image).flatten(1)
        source_visibility = visibility_logits.sigmoid()

        centre, frame, basis = full_frame_state_to_components(state.reshape(count, 12))
        edges = frame[:, :, :2] @ basis
        normal = frame[:, :, 2]
        axis = (torch.arange(side, device=source.device, dtype=source.dtype) + .5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        source_grid = torch.stack((xx, yy), -1).reshape(1, side * side, 2).expand(batch, -1, -1)
        source_chart = source_grid[:, None].expand(-1, candidates, -1, -1).reshape(count, -1, 2)
        source_chart = source_chart - source_chart.new_tensor((.5 / width, .5 / height))
        source_chart = torch.stack((torch.where(reflection.reshape(count, 1).bool(),
            (width - 1) / width - source_chart[..., 0], source_chart[..., 0]),
            source_chart[..., 1]), -1)
        query_ccf = centre[:, None] + torch.einsum('nij,nqj->nqi', edges, source_chart - .5)

        key_axis = axis * 1.5 - .25
        ky, kx = torch.meshgrid(key_axis, key_axis, indexing='ij')
        key_chart = torch.stack((kx - .5 / width, ky - .5 / height), -1)[None].expand(count, -1, -1, -1)
        key_chart = torch.stack((torch.where(reflection.reshape(count, 1, 1).bool(),
            (width - 1) / width - key_chart[..., 0], key_chart[..., 0]),
            key_chart[..., 1]), -1)
        base = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi', edges, key_chart - .5)
        axial = torch.linspace(-3000., 3000., depths, device=source.device, dtype=source.dtype)
        key_ccf = base[:, None] + axial[None, :, None, None, None] * normal[:, None, None, None]
        section_offsets = offsets[:, None].expand(-1, candidates, -1).reshape(count, -1)
        section_weights = weights[:, None].expand(-1, candidates, -1).reshape(count, -1)
        world = base[:, None, None] + (
            axial[None, :, None] + section_offsets[:, None, :]
        )[:, :, :, None, None, None] * normal[:, None, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world.reshape(count * depths, section_offsets.shape[-1], side, side, 3),
            (0., 0., 0.), (25., 25., 25.),
            section_weights[:, None].expand(-1, depths, -1).reshape(count * depths, -1),
        ).reshape(count, depths, 2, side, side).permute(0, 2, 1, 3, 4)
        support = rendered[:, 1:2].clamp(0, 1)
        intensity = rendered[:, :1] / support.clamp_min(1e-4)
        if not atlas_intensity:
            intensity = torch.zeros_like(intensity)
        key = F.normalize(self.atlas(intensity).flatten(2).transpose(1, 2), dim=-1)
        key_ccf = key_ccf.reshape(count, depths * side * side, 3)
        key_support = support.reshape(count, depths * side * side)

        query = query[:, None].expand(-1, candidates, -1, -1).reshape(count, side * side, 32)
        valid = ((torch.cdist(query_ccf, key_ccf) <= 3000.) &
                 (key_support[:, None] > .5))
        match_logits = (self.log_temperature.exp() * query @ key.transpose(-2, -1))
        match_logits = match_logits.masked_fill(~valid, -torch.inf)
        dustbin = self.dustbin(query) + valid.sum(-1, keepdim=True).clamp_min(1).log()
        logits = torch.cat((match_logits, dustbin), -1)
        probability_all = logits.float().softmax(-1)
        probability = probability_all[..., :-1]
        match_mass = probability.sum(-1)
        expected_ccf = ((probability / match_mass.clamp_min(1e-6)[..., None])
                        @ key_ccf.float())
        confidence = source_visibility[:, None].expand(-1, candidates, -1).reshape(
            count, side * side) * match_mass

        with torch.autocast(device_type=source.device.type, enabled=False):
            chart = source_chart.float()
            design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
            prior = torch.stack((centre.float(), edges[:, :, 0].float(),
                                 edges[:, :, 1].float()), -2)
            ridge = torch.diag(state.new_tensor((24., 12., 12.))).float()
            lhs = torch.einsum('nqi,nq,nqj->nij', design, confidence.float(), design) + ridge
            rhs = (torch.einsum('nqi,nq,nqj->nij', design, confidence.float(), expected_ccf)
                   + ridge @ prior)
            fitted = torch.linalg.solve(lhs, rhs)
            fitted_ouv = torch.stack((fitted[:, 0] - .5 * (fitted[:, 1] + fitted[:, 2]),
                                      fitted[:, 1], fitted[:, 2]), -2)
            prior_ouv = torch.stack((prior[:, 0] - .5 * (prior[:, 1] + prior[:, 2]),
                                     prior[:, 1], prior[:, 2]), -2)
            corrected = full_frame_state_from_components(*physical_ouv_to_frame(fitted_ouv))
            baseline = full_frame_state_from_components(*physical_ouv_to_frame(prior_ouv))
            corrected_state = state.reshape(count, 12).float() + corrected - baseline

        return {
            'corrected_state': corrected_state.reshape(batch, candidates, 12),
            'expected_ccf': expected_ccf.reshape(batch, candidates, side * side, 3),
            'logits': logits.reshape(batch, candidates, side * side, -1),
            'key_ccf': key_ccf.reshape(batch, candidates, -1, 3),
            'key_support': key_support.reshape(batch, candidates, -1),
            'match_mass': match_mass.reshape(batch, candidates, side * side),
            'dustbin_probability': probability_all[..., -1].reshape(batch, candidates, side * side),
            'confidence': confidence.reshape(batch, candidates, side * side),
            'source_visibility': source_visibility,
            'visibility_logits': visibility_logits,
            'source_grid': source_grid,
            'source_chart': source_chart.reshape(batch, candidates, side * side, 2),
        }
