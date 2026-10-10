"""Contextual 2-D/3-D correspondence and reliability-weighted plane correction."""

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


class CoarseAtlasPose150(nn.Module):
    def __init__(self, detach_fit_weights=True):
        super().__init__()
        self.detach_fit_weights = detach_fit_weights
        self.source = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2), nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.source_context = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.source_low = nn.Conv2d(5, 48, 1)
        self.source_grid = nn.Sequential(
            nn.Conv2d(128, 48, 1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2), nn.GroupNorm(8, 48), nn.GELU(),
        )
        # Atlas support gates keys, but is not a descriptor input.
        self.atlas = nn.Sequential(
            nn.Conv3d(1, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv3d(24, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.atlas_context = nn.Sequential(
            nn.Conv3d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv3d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.visibility = nn.Conv2d(48, 1, 1)
        self.dustbin = nn.Linear(48, 1)
        self.reliability = nn.Sequential(
            nn.Linear(147, 48), nn.GELU(), nn.Linear(48, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, source, atlas, state, reflection, offsets, weights,
                atlas_intensity=True):
        batch, candidates = state.shape[:2]
        count, side, depths = batch * candidates, 24, 9
        height, width = source.shape[-2:]
        source_mid = self.source(source)
        image = self.source_grid(torch.cat((
            F.adaptive_avg_pool2d(source_mid, (side, side)),
            F.adaptive_avg_pool2d(self.source_context(source_mid), (side, side)),
            self.source_low(F.adaptive_avg_pool2d(source, (side, side))),
        ), dim=1))
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
        atlas_local = self.atlas(intensity)
        atlas_coarse = F.avg_pool3d(atlas_local, 3, stride=2, padding=1)
        atlas_coarse = F.interpolate(self.atlas_context(atlas_coarse),
            size=atlas_local.shape[-3:], mode='trilinear', align_corners=False)
        key = F.normalize((atlas_local + atlas_coarse).flatten(2).transpose(1, 2), dim=-1)
        key_ccf = key_ccf.reshape(count, depths * side * side, 3)
        key_support = support.reshape(count, depths * side * side)

        query = query[:, None].expand(-1, candidates, -1, -1).reshape(count, side * side, 48)
        with torch.autocast(device_type=source.device.type, enabled=False):
            valid = ((torch.cdist(query_ccf.float() / 1000.,
                                  key_ccf.float() / 1000.) <= 3.) &
                     (key_support[:, None] > .5))
        match_logits = self.log_temperature.exp() * query @ key.transpose(-2, -1)
        match_logits = match_logits.masked_fill(~valid, -torch.inf)
        dustbin = self.dustbin(query) + valid.sum(-1, keepdim=True).clamp_min(1).log()
        logits = torch.cat((match_logits, dustbin), -1)
        probability_all = logits.float().softmax(-1)
        probability = probability_all[..., :-1]
        match_mass = probability.sum(-1)
        normalized = probability / match_mass.clamp_min(1e-6)[..., None]
        with torch.autocast(device_type=source.device.type, enabled=False):
            expected_ccf = normalized @ key_ccf.float()
            expected_ccf = torch.where(valid.any(-1)[..., None], expected_ccf, query_ccf.float())

        best_key = normalized.argmax(-1)
        matched = key.gather(1, best_key[..., None].expand(-1, -1, 48))
        peak = normalized.gather(-1, best_key[..., None])
        entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        entropy = entropy / valid.sum(-1, keepdim=True).clamp_min(2).float().log()
        reliability_logits = self.reliability(torch.cat((
            query, matched, query * matched, peak, entropy, match_mass[..., None],
        ), -1)).squeeze(-1)
        reliability_probability = reliability_logits.sigmoid() * valid.any(-1)
        raw_confidence = source_visibility[:, None].expand(-1, candidates, -1).reshape(
            count, side * side) * reliability_probability
        mean_confidence = raw_confidence.mean(-1, keepdim=True)
        trust = (mean_confidence / (mean_confidence + .05)).clamp(max=.8)
        confidence = (raw_confidence / mean_confidence.clamp_min(.05)).clamp(max=3.) * trust
        fit_weight = confidence.detach() if self.detach_fit_weights else confidence

        with torch.autocast(device_type=source.device.type, enabled=False):
            chart = source_chart.float()
            design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
            prior = torch.stack((centre.float(), edges[:, :, 0].float(),
                                 edges[:, :, 1].float()), -2)
            ridge = torch.diag(state.new_tensor((24., 12., 12.))).float()
            lhs = torch.einsum('nqi,nq,nqj->nij', design, fit_weight.float(), design) + ridge
            rhs = (torch.einsum('nqi,nq,nqj->nij', design, fit_weight.float(), expected_ccf)
                   + ridge @ prior)
            fitted = torch.linalg.solve(lhs, rhs)
            fitted_ouv = torch.stack((fitted[:, 0] - .5 * (fitted[:, 1] + fitted[:, 2]),
                                      fitted[:, 1], fitted[:, 2]), -2)
            corrected_state = full_frame_state_from_components(*physical_ouv_to_frame(fitted_ouv))

        return {
            'corrected_state': corrected_state.reshape(batch, candidates, 12),
            'expected_ccf': expected_ccf.reshape(batch, candidates, side * side, 3),
            'logits': logits.reshape(batch, candidates, side * side, -1),
            'key_ccf': key_ccf.reshape(batch, candidates, -1, 3),
            'key_support': key_support.reshape(batch, candidates, -1),
            'match_mass': match_mass.reshape(batch, candidates, side * side),
            'dustbin_probability': probability_all[..., -1].reshape(batch, candidates, side * side),
            'confidence': confidence.reshape(batch, candidates, side * side),
            'effective_fit_weight': confidence.reshape(batch, candidates, side * side),
            'raw_confidence': raw_confidence.reshape(batch, candidates, side * side),
            'reliability_logits': reliability_logits.reshape(batch, candidates, side * side),
            'reliability_probability': reliability_probability.reshape(batch, candidates, side * side),
            'source_visibility': source_visibility,
            'visibility_logits': visibility_logits,
            'source_grid': source_grid,
            'source_chart': source_chart.reshape(batch, candidates, side * side, 2),
        }
