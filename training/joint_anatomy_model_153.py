"""Scratch-initialized arbitrary-plane pose, atlas matching, and constrained tissue map."""

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_ribbon_v6 import compose_curved_ribbon_coordinates
from training.coarse_atlas_pose_150 import CoarseAtlasPose150


class JointAnatomyModel153(nn.Module):
    """Fit only supplied mode/reflection branches; probabilities remain uncalibrated."""

    def __init__(self):
        super().__init__()
        self.pose = OneShotJointSliceModel(modes=16, normal_anchor_count=64)
        self.matcher = CoarseAtlasPose150(detach_fit_weights=True)
        self.field_body = nn.Sequential(
            nn.Conv2d(69, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.field_out = nn.Conv2d(64, 6, 1)
        nn.init.zeros_(self.field_out.weight)
        nn.init.zeros_(self.field_out.bias)

    def forward(self, inputs, atlas, offsets, weights, mode_index, reflection,
                context=None, state_override=None):
        prediction = self.pose.predict(inputs, context)
        batch, candidates = mode_index.shape
        rows = torch.arange(batch, device=inputs.device)[:, None]
        state = prediction['state'][rows, mode_index] if state_override is None else state_override
        match = self.matcher(inputs, atlas, state, reflection, offsets, weights)

        chart = match['source_chart'].reshape(-1, 24 * 24, 2).float()
        design = torch.cat((torch.ones_like(chart[..., :1]), chart - .5), -1)
        target = match['expected_ccf'].reshape(-1, 24 * 24, 3).float()
        confidence = match['confidence'].reshape(-1, 24 * 24).float().detach()
        centre, frame, basis = full_frame_state_to_components(state.reshape(-1, 12).float())
        edges = frame[:, :, :2] @ basis
        prior = torch.stack((centre, edges[:, :, 0], edges[:, :, 1]), 1)
        identity = torch.eye(3, device=inputs.device, dtype=torch.float32)[None]

        def solve(point_weight):
            gram = torch.einsum('nqi,nq,nqj->nij', design, point_weight, design)
            right = torch.einsum('nqi,nq,nqj->nij', design, point_weight, target)
            # The atlas fit is unbiased when spatial evidence is sufficient.
            ridge = (1. - torch.linalg.eigvalsh(gram).amin(-1).detach()).clamp_min(0.) + 1e-4
            regularizer = ridge[:, None, None] * identity
            return torch.linalg.solve(gram + regularizer, right + regularizer @ prior)

        initial = solve(confidence)
        initial_residual_mm = (design @ initial - target).norm(dim=-1) / 1000.
        robust = (.75 / initial_residual_mm.clamp_min(.75)).detach()
        fitted = solve(confidence * robust)
        confidence_mass = confidence.sum(-1)
        trust = (((confidence * robust).sum(-1) / confidence_mass.clamp_min(1e-6))
                 * confidence_mass / (confidence_mass + 10.)).detach()
        delta = fitted - prior
        centre_step = delta[:, 0].norm(dim=-1)
        edge_fraction = torch.stack((delta[:, 1].norm(dim=-1) / prior[:, 1].norm(dim=-1),
                                     delta[:, 2].norm(dim=-1) / prior[:, 2].norm(dim=-1)), -1).amax(-1)
        correction_limit = torch.minimum((2000. / centre_step.clamp_min(1.)),
                                          (.25 / edge_fraction.clamp_min(1e-6))).clamp(max=1.)
        fitted = prior + (trust * correction_limit)[:, None, None] * delta
        ouv = torch.stack((fitted[:, 0] - .5 * (fitted[:, 1] + fitted[:, 2]),
                           fitted[:, 1], fitted[:, 2]), 1)
        corrected = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
        residual = target - design @ fitted
        corrected_frame = full_frame_state_to_components(corrected)[1]
        residual_local = torch.einsum('nij,nqj->nqi', corrected_frame.transpose(-1, -2), residual)
        residual_local = (residual_local / 1000.).transpose(1, 2).reshape(-1, 3, 24, 24)

        feature = prediction['feature'][:, None].expand(-1, candidates, -1, -1, -1)
        feature = F.adaptive_avg_pool2d(feature.flatten(0, 1), (64, 64))
        evidence = torch.cat((feature,
            F.interpolate(residual_local.to(feature.dtype), (64, 64), mode='bilinear', align_corners=False),
            F.interpolate(confidence.reshape(-1, 1, 24, 24).to(feature.dtype),
                          (64, 64), mode='bilinear', align_corners=False),
            F.interpolate(match['match_mass'].reshape(-1, 1, 24, 24).to(feature.dtype),
                          (64, 64), mode='bilinear', align_corners=False)), 1)
        hidden = self.field_body(evidence)
        raw = F.interpolate(self.field_out(hidden), inputs.shape[-2:],
                            mode='bilinear', align_corners=False)
        local = torch.cat((1000. * raw[:, :3].tanh(), .2 * raw[:, 3:].tanh()), 1).float()
        reflected = reflection.reshape(-1).bool()
        local = torch.where(reflected[:, None, None, None], local.flip(-1), local)
        axial = offsets[:, None].expand(-1, candidates, -1).reshape(-1, offsets.shape[-1])
        geometry = compose_curved_ribbon_coordinates(corrected, local[:, :3], local[:, 3:], axial)
        surface = geometry['centre_surface_ccf_ap_dv_ml_um']
        coordinates = geometry['ccf_coordinates_ap_dv_ml_um']
        surface = torch.where(reflected[:, None, None, None], surface.flip(-2), surface)
        coordinates = torch.where(reflected[:, None, None, None, None],
                                  coordinates.flip(-2), coordinates)

        difficulty = (geometry['residual_local_um'] / 200.).square().mean((1, 2, 3))
        difficulty = difficulty + (geometry['director_delta_local'] / .1).square().mean((1, 2, 3))
        difficulty = difficulty + (geometry['prelimit_derivative_frobenius_bound'] / .35).square()
        residual_mm = ((residual.norm(dim=-1) / 1000.) * confidence).sum(-1) / confidence_mass.clamp_min(1e-6)
        residual_mm = torch.where(confidence_mass >= 1., residual_mm, 3. * torch.ones_like(residual_mm))
        query_grid = (2 * match['source_grid'] - 1).reshape(batch, 24, 24, 2)
        query_grid = query_grid[:, None].expand(-1, candidates, -1, -1, -1).reshape(-1, 24, 24, 2)
        final_query = F.grid_sample(surface.permute(0, 3, 1, 2), query_grid,
                                    mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        atlas_distance = torch.cdist(final_query.float(), match['key_ccf'].reshape(-1, 9 * 24 * 24, 3).float())
        log_probability = match['logits'].reshape(-1, 24 * 24, 9 * 24 * 24 + 1).float().log_softmax(-1)
        agreement_terms = log_probability[..., :-1] - .5 * (atlas_distance / 400.).square()
        agreement = -torch.logsumexp(torch.cat((agreement_terms,
            agreement_terms.new_full((*agreement_terms.shape[:2], 1), -9.)), -1), -1)
        visibility = match['source_visibility'][:, None].expand(-1, candidates, -1).reshape(-1, 24 * 24).detach()
        coverage = visibility.mean(-1)
        final_anatomy_fit = ((agreement * visibility).sum(-1) / visibility.sum(-1).clamp_min(1.)
                             - coverage.clamp_min(1e-4).log())
        correction_cost = (centre_step / 2000.).square() + (edge_fraction / .25).square()
        fit_energy = final_anatomy_fit + .1 * residual_mm + .05 * difficulty + .05 * correction_cost
        reflection_log_mass = torch.where(reflection.bool(),
            F.logsigmoid(prediction['reflection_logit'][rows, mode_index]),
            F.logsigmoid(-prediction['reflection_logit'][rows, mode_index]))
        direct_log_mass = prediction['log_mass'][rows, mode_index] + reflection_log_mass
        return {
            'direct_prediction': prediction, 'candidate_state': state,
            'corrected_state': corrected.reshape(batch, candidates, 12),
            'surface_ccf_um': surface.reshape(batch, candidates, *inputs.shape[-2:], 3),
            'coordinates_ccf_um': coordinates.reshape(batch, candidates, axial.shape[-1],
                                                      *inputs.shape[-2:], 3),
            'fit_difficulty': difficulty.reshape(batch, candidates),
            'fit_residual_mm': residual_mm.reshape(batch, candidates),
            'final_anatomy_fit': final_anatomy_fit.reshape(batch, candidates),
            'fit_correction_cost': correction_cost.reshape(batch, candidates),
            'fit_energy_uncalibrated': fit_energy.reshape(batch, candidates),
            'direct_log_mass': direct_log_mass,
            'score_uncalibrated': direct_log_mass - fit_energy.reshape(batch, candidates),
            'matcher': match, 'geometry': geometry, 'calibrated': False,
        }
