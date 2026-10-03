"""Candidate-level evidence available without synthetic truth or manual masks."""
import math

import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components


def candidate_evidence(prediction, fitted):
    mapped = fitted['mapped']
    probability = fitted['match_probability']
    local = mapped['local_displacement_um'] / 1000
    mode = fitted['choice'] // 2
    first = prediction['state'].gather(1, mode[..., None].expand(-1, -1, 12))
    start_centre, start_frame, _ = full_frame_state_to_components(first)
    fit_centre, fit_frame, _ = full_frame_state_to_components(fitted['fitted_state'])
    return torch.stack((
        fitted['prior'],
        fitted['score'] - fitted['prior'],
        fitted['match_residual_um'] / 1000,
        probability.amax(-1).mean(-1),
        -(probability * probability.clamp_min(1e-9).log()).sum(-1).mean(-1) / math.log(
            probability.shape[-1]),
        mapped['fit_energy'],
        local.square().sum(2).sqrt().mean((-2, -1)),
        ((local[..., 1:, :] - local[..., :-1, :]).square().mean((2, 3, 4)) +
         (local[..., 1:] - local[..., :-1]).square().mean((2, 3, 4))).sqrt(),
        mapped['correspondence_logit'].sigmoid().mean((2, 3, 4)),
        (fit_centre - start_centre).norm(dim=-1) / 1000,
        (1 - (fit_frame[..., :, 2] * start_frame[..., :, 2]).sum(-1).clamp(-1, 1)),
    ), -1)
