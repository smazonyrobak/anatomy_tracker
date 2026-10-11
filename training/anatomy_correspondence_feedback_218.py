"""Opt-in, image-only coarse anatomy correspondence and rigid-plane feedback.

``forward(inputs, atlas_intensity, atlas_support, atlas_ccf_um, state,
reflection, source_weight=None, source_chart_st=None, original_width=None,
source_encoded=None, return_logits=False)`` takes N candidate branches at once.
``inputs`` is [N,5,H,W]; only channel 0 enters learned features. The caller
supplies finite-thickness atlas intensity and support [N,D,1,24,24] and the
CCF coordinates of those rendered slab centres [N,D,24,24,3] in AP/DV/ML um.
Render D=9 slabs at ``depth_offsets_um`` (-4 to +4 mm) on the *same original
image chart* sampled by the 24x24 source grid. ``state`` is [N,12] in the
full-frame convention; ``reflection`` is [N] and means horizontal source
reflection, not a change to the proper frame. ``source_chart_st`` optionally
supplies observed (unreflected) original-image normalized (s,t) coordinates
[N,24,24,2], necessary for a cropped source view; without it, the full input
canvas is used. With a crop, pass its original full-image width as
``original_width``. Canonical fit s is ``(W-1)/W - observed_s`` if reflected.
``source_weight`` is optional observed-tissue/reliability [N,1,H,W] or
[N,1,24,24]; otherwise a source-image-only soft reliability head is used.

Outputs include ``anatomy_logit`` [N], explicit ``no_match_logit`` [N,24,24],
``match_probability`` and ``fit_weight`` [N,24,24], soft
``expected_atlas_chart_st`` [N,24,24,2] in the *observed original* chart,
``expected_ccf_um`` [N,24,24,3], ``fitted_state`` and ``unchanged_state``
[N,12]. The latter is always an available no-update branch. Set
``return_logits=True`` for [N,576,D*576] correspondence logits and their
validity mask, for supervised correspondence loss. ``encode_source`` may be
called once and passed as ``source_encoded`` across candidate chunks.

All 160 branches should be processed in small chunks (N<=4 on an 11-GB GPU).
The 3-mm in-plane search and CCF coordinates are used only for a hard
geometric admissibility mask and the affine fit: learned descriptors and the
null score receive image intensity alone, never pose, CCF coordinates, prior,
atlas support, outline, or brush channels. Candidate evidence is normalized
by the *original source-weight mass*, not the source-atlas overlap; no atlas
support therefore scores as a mismatch, not a good fit.
"""

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame


class AnatomyCorrespondenceFeedback218(nn.Module):
    def __init__(self, side=24, channels=24, search_um=3000., temperature=.12):
        super().__init__()
        self.side, self.search_um, self.temperature = side, search_um, temperature
        self.source_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, channels, 1),
        )
        self.atlas_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, channels, 1),
        )
        self.reliability = nn.Conv2d(channels, 1, 1)
        self.no_match = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        nn.init.zeros_(self.no_match.weight)
        nn.init.constant_(self.no_match.bias, .25)
        self.register_buffer('depth_offsets_um', torch.arange(-4000., 4001., 1000.))

    def encode_source(self, inputs):
        image = F.interpolate(inputs[:, :1], (self.side, self.side), mode='area')
        feature = self.source_encoder(image)
        descriptor = F.normalize(feature, dim=1, eps=1e-6)
        return descriptor, .05 + .95 * self.reliability(feature).sigmoid(), self.no_match(feature)

    def encode_atlas(self, atlas_intensity):
        n, depth = atlas_intensity.shape[:2]
        image = atlas_intensity.flatten(0, 1)
        descriptor = F.normalize(self.atlas_encoder(image), dim=1, eps=1e-6)
        return descriptor.reshape(n, depth, *descriptor.shape[1:])

    def forward(self, inputs, atlas_intensity, atlas_support, atlas_ccf_um, state,
                reflection, source_weight=None, source_chart_st=None,
                original_width=None, source_encoded=None, return_logits=False):
        n, depth = atlas_intensity.shape[:2]
        side, sites = self.side, self.side * self.side
        source, inferred_weight, null = (self.encode_source(inputs) if source_encoded is None
                                          else source_encoded)
        atlas = self.encode_atlas(atlas_intensity)
        source = source.float().flatten(2)
        atlas = atlas.float().flatten(3)
        similarity = torch.einsum('ncq,ndck->ndqk', source, atlas)
        support = atlas_support.float().reshape(n, depth, sites).clamp(0, 1)

        if source_chart_st is None:
            height, width = inputs.shape[-2:]
            sy = (torch.arange(side, device=inputs.device, dtype=torch.float32) + .5) / side - .5 / height
            sx = (torch.arange(side, device=inputs.device, dtype=torch.float32) + .5) / side - .5 / width
            yy, xx = torch.meshgrid(sy, sx, indexing='ij')
            chart = torch.stack((xx, yy), -1)[None].expand(n, -1, -1, -1)
        else:
            chart = source_chart_st.float()
            width = original_width
        observed_chart = chart.reshape(n, sites, 2)
        canonical_s = torch.where(reflection.bool()[:, None],
                                  (width - 1) / width - observed_chart[..., 0],
                                  observed_chart[..., 0])
        design = torch.stack((torch.ones_like(canonical_s), canonical_s - .5,
                              observed_chart[..., 1] - .5), -1)
        centre, frame, basis = full_frame_state_to_components(state.float())
        edges = frame[..., :2] @ basis
        base = torch.stack((centre, edges[..., 0], edges[..., 1]), 1)
        plane = design @ base
        local_plane = torch.einsum('nqi,nij->nqj', plane - centre[:, None], frame[..., :2])
        keys = atlas_ccf_um.float().reshape(n, depth, sites, 3)
        local_keys = torch.einsum('ndki,nij->ndkj', keys - centre[:, None, None], frame[..., :2])
        distance2 = (local_plane.square().sum(-1)[:, None, :, None]
                     + local_keys.square().sum(-1)[:, :, None, :]
                     - 2 * torch.einsum('nqi,ndki->ndqk', local_plane, local_keys)).clamp_min(0)
        valid = (distance2 <= self.search_um ** 2) & (support[:, :, None, :] > .01)
        support_key = support[:, :, None, :] * valid
        available_mass = support_key.sum((1, 3))
        mean_similarity = (similarity * support_key).sum((1, 3)) / available_mass.clamp_min(1)
        key_logits = ((similarity - mean_similarity[:, None, :, None]) / self.temperature
                      + support[:, :, None, :].clamp_min(1e-6).log()
                      - available_mass[:, None, :, None].clamp_min(1).log())
        key_logits = key_logits.masked_fill(~valid, -1e4)
        flat_logits = key_logits.permute(0, 2, 1, 3).reshape(n, sites, depth * sites)
        evidence = torch.logsumexp(flat_logits, -1)
        evidence = torch.where(available_mass > 0, evidence, evidence.new_full(evidence.shape, -4.))
        null = null.float().reshape(n, sites)
        match_probability = torch.sigmoid(evidence - null) * (available_mass > 0)
        conditional = flat_logits.softmax(-1)
        expected_ccf = conditional @ keys.reshape(n, depth * sites, 3)
        key_chart = observed_chart[:, None].expand(n, depth, sites, 2).reshape(n, depth * sites, 2)
        expected_chart = conditional @ key_chart
        expected_ccf = torch.where((available_mass > 0)[..., None], expected_ccf, plane)
        expected_chart = torch.where((available_mass > 0)[..., None], expected_chart, observed_chart)

        if source_weight is None:
            weight = inferred_weight.float()
        else:
            weight = F.interpolate(source_weight.float(), (side, side), mode='area')
        weight = weight.reshape(n, sites).clamp(0, 1)
        anatomy_logit = (weight * (evidence - null)).sum(-1) / weight.sum(-1).clamp_min(1)
        entropy = -(conditional * conditional.clamp_min(1e-8).log()).sum(-1)
        key_count = valid.sum((1, 3)).clamp_min(2)
        concentration = (1 - entropy / key_count.log()).clamp(0, 1)
        fit_weight = weight * match_probability * concentration

        ridge = torch.diag(state.new_tensor((1., 4., 4.)).float())[None]
        gram = torch.einsum('nqi,nq,nqj->nij', design, fit_weight, design) + ridge
        residual = expected_ccf - plane
        rhs = torch.einsum('nqi,nq,nqj->nij', design, fit_weight, residual)
        delta = torch.linalg.solve(gram, rhs)
        centre_limit = delta.new_full((n, 1), 3000.)
        edge_limit = (base[:, 1:].norm(dim=-1) * .35).clamp_max(4000.)
        limits = torch.cat((centre_limit, edge_limit), 1)
        delta = delta * (limits / delta.norm(dim=-1).clamp_min(1)).clamp_max(1)[..., None]
        fitted = base + delta
        origin = fitted[:, 0] - .5 * (fitted[:, 1] + fitted[:, 2])
        fitted_state = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((origin, fitted[:, 1], fitted[:, 2]), 1))).to(state.dtype)
        result = {
            'anatomy_logit': anatomy_logit,
            'no_match_logit': null.reshape(n, side, side),
            'match_probability': match_probability.reshape(n, side, side),
            'source_weight': weight.reshape(n, side, side),
            'fit_weight': fit_weight.reshape(n, side, side),
            'source_chart_st': observed_chart.reshape(n, side, side, 2),
            'expected_atlas_chart_st': expected_chart.reshape(n, side, side, 2),
            'expected_ccf_um': expected_ccf.reshape(n, side, side, 3),
            'support_reach_fraction': (weight * (available_mass > 0)).sum(-1)
                                      / weight.sum(-1).clamp_min(1),
            'fitted_state': fitted_state,
            'unchanged_state': state,
        }
        if return_logits:
            result['correspondence_logits'] = flat_logits
            result['correspondence_valid'] = valid.permute(0, 2, 1, 3).reshape(n, sites, depth * sites)
        return result
