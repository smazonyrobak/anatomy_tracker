"""Candidate-conditioned tissue/atlas correspondence for the 019 joint model.

The parent predicts the plane and local tissue map. This head compares the
observed slice with each *fitted* finite-thickness atlas candidate pixelwise;
its evidence changes the candidate posterior, not the local deformation.
Scores are not calibrated probabilities.
"""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_recurrent_model import local_correlation


class OneShotCorrespondenceModel021(OneShotJointSliceModel):
    def __init__(self):
        super().__init__(modes=16, atlas_conditioning=True, fit_quality=True,
                         vector_refinement=True, candidate_ranking=True, fitted_ranking=True)
        self.image_descriptor_021 = nn.Sequential(
            nn.Conv2d(69, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 1))
        self.atlas_descriptor_021 = nn.Sequential(
            nn.Conv2d(66, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 1))
        self.match_field_021 = nn.Sequential(
            nn.Conv2d(25 + 6, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.match_field_021[-1].weight)
        nn.init.zeros_(self.match_field_021[-1].bias)

    def correspondence_score_021(self, inputs, prediction, mapped, atlas, weights,
                                 old_score, return_fields=False):
        batch, count, samples, side = mapped['coordinates'].shape[:4]
        masses = torch.as_tensor(weights, device=inputs.device, dtype=inputs.dtype)
        if masses.ndim == 1:
            masses = masses[None].expand(batch, -1)
        masses = masses[:, None].expand(-1, count, -1).flatten(0, 1)
        rendered = render_finite_thickness_coordinate_grid(
            atlas, mapped['coordinates'].flatten(0, 1),
            (0., 0., 0.), (25., 25., 25.), masses)
        support = rendered[:, 1:2].clamp(0, 1)
        atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        atlas_feature = self.atlas_encoder(atlas_pair)
        source_feature = F.adaptive_avg_pool2d(prediction['feature'], (side, side))
        source_feature = source_feature[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        source_input = F.interpolate(inputs, (side, side), mode='area')
        source_input = source_input[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        image_descriptor = self.image_descriptor_021(torch.cat((source_feature, source_input), 1))
        atlas_descriptor = self.atlas_descriptor_021(torch.cat((atlas_feature, atlas_pair), 1))
        correlation = local_correlation(image_descriptor, atlas_descriptor, 2)
        reliability = mapped['correspondence_logit'].flatten(0, 1).sigmoid().detach()
        displacement = mapped['local_displacement_um'].flatten(0, 1).norm(dim=1, keepdim=True) / 1000
        peak = correlation.max(1, keepdim=True).values
        evidence = torch.cat((correlation, peak, source_input[:, :1], atlas_pair,
                              reliability, displacement), 1)
        pixel_logit = self.match_field_021(evidence)
        residual = (pixel_logit * reliability).sum((1, 2, 3)) / reliability.sum((1, 2, 3)).clamp_min(1)
        score = old_score + residual.reshape(batch, count)
        if return_fields:
            return score, pixel_logit.reshape(batch, count, side, side), peak.reshape(batch, count, side, side)
        return score
