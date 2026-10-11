"""Candidate-aligned, anatomy-only 2-D/3-D descriptors for arbitrary-plane matching."""

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.global_atlas_contrast_090 import rigid_points_090


class Local3DDescriptor240(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.atlas = nn.Sequential(
            nn.Conv3d(1, 32, (1, 5, 5), padding=(0, 2, 2)), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1)),
        )

    def encode_source(self, raw_image):
        return F.normalize(self.source(raw_image), dim=1)

    def encode_atlas(self, intensity):
        return F.normalize(self.atlas(intensity), dim=1)

    def forward(self, raw_image, intensity):
        return self.encode_source(raw_image), self.encode_atlas(intensity)


def render_candidate_bank_240(atlas, state, reflection, offsets, weights, null_seed=None):
    axis = (torch.arange(44, device=state.device, dtype=state.dtype) - 5.5) / 32 - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    plane = rigid_points_090(state, reflection, chart)[0].reshape(44, 44, 3)
    normal = full_frame_state_to_components(state)[1][0, :, 2]
    depth = torch.linspace(-4000, 4000, 17, device=state.device, dtype=state.dtype)
    axial = depth[:, None] + torch.as_tensor(offsets, device=state.device, dtype=state.dtype).reshape(1, -1)
    world = plane[None, None] + axial[:, :, None, None, None] * normal
    rendered = render_finite_thickness_coordinate_grid(
        atlas, world, (0., 0., 0.), (25., 25., 25.),
        torch.as_tensor(weights, device=state.device, dtype=state.dtype).reshape(1, -1).expand(17, -1),
    )
    support = rendered[:, 1].clamp(0, 1)
    intensity = rendered[:, 0] / support.clamp_min(1e-4)
    if null_seed is not None:
        intensity = intensity.clone()
        generator = torch.Generator(device=intensity.device).manual_seed(int(null_seed))
        for index in range(17):
            pixels = (support[index] > .5).flatten().nonzero().flatten()
            values = intensity[index].flatten()
            values[pixels] = values[pixels][torch.randperm(len(pixels), device=intensity.device,
                                                          generator=generator)]
    return intensity[None, None], support[None, None]


def sample_candidate_features_240(features, support, candidate_state, reflection, world_points):
    centre, frame, basis = full_frame_state_to_components(candidate_state)
    local = (world_points - centre[:, None]) @ frame
    chart = torch.linalg.solve(basis, local[..., :2].transpose(1, 2)).transpose(1, 2) + .5
    reflected = torch.as_tensor(reflection, device=features.device).reshape(-1, 1).bool()
    x = torch.where(reflected, 255 / 256 - chart[..., 0], chart[..., 0])
    ix = 32 * (x + .5 / 256) + 5.5
    iy = 32 * (chart[..., 1] + .5 / 256) + 5.5
    iz = local[..., 2] / 500 + 8
    grid = torch.stack((2 * ix / 43 - 1, 2 * iy / 43 - 1, iz / 8 - 1), -1)
    grid = grid[:, None, None]
    descriptor = F.grid_sample(features, grid, align_corners=True).squeeze(2).squeeze(2).transpose(1, 2)
    available = F.grid_sample(support, grid, align_corners=True).flatten(1)
    return F.normalize(descriptor, dim=-1), available


def query_chart_240(device, dtype):
    axis = (torch.arange(0, 32, 2, device=device, dtype=dtype) + .5) / 32 - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(256, 2)


def _score_world_240(source_features, atlas_features, support, candidate_state,
                     reflection, world):
    key, available = sample_candidate_features_240(
        atlas_features, support, candidate_state, reflection, world)
    query = source_features[:, :, ::2, ::2].flatten(2).transpose(1, 2)
    cosine = (query * key).sum(-1)
    supported = available > .5
    score = cosine.masked_fill(~supported, -1.).topk(64, dim=-1).values.mean(-1)
    return score, supported.sum(-1)


def coherent_score_240(source_features, atlas_features, support, candidate_state,
                       reflection, hypothesis_state):
    chart = query_chart_240(source_features.device, source_features.dtype)
    world = rigid_points_090(hypothesis_state, reflection, chart)
    return _score_world_240(source_features, atlas_features, support,
                            candidate_state, reflection, world)


def refine_candidate_240(source_features, atlas_features, support, candidate_state,
                         reflection, steps=24, learning_rate=.12):
    with torch.inference_mode(False), torch.enable_grad():
        source = source_features.detach().clone()
        atlas = atlas_features.detach().clone()
        availability = support.detach().clone()
        start = candidate_state.detach().clone()
        centre, frame, basis = full_frame_state_to_components(start)
        edges = frame[:, :, :2] @ basis
        original = torch.stack((centre[0], edges[0, :, 0], edges[0, :, 1]))
        chart = query_chart_240(source.device, source.dtype)
        canonical_x = torch.where(torch.as_tensor(reflection, device=source.device).bool(),
                                  255 / 256 - chart[:, 0], chart[:, 0])
        design = torch.stack((torch.ones_like(canonical_x), canonical_x - .5,
                              chart[:, 1] - .5), -1)
        anchors = start.new_tensor(((1, -.5, -.5), (1, .5, -.5),
                                    (1, -.5, .5), (1, .5, .5), (1, 0, 0)))
        initial_score, initial_count = _score_world_240(
            source, atlas, availability, start, reflection, (design @ original)[None])
        delta = torch.zeros_like(original, requires_grad=True)
        best_delta = delta.detach().clone()
        best_score = initial_score.detach()
        optimizer = torch.optim.Adam((delta,), lr=learning_rate)
        for _ in range(steps):
            coefficients = original + 1000 * delta
            score, _ = _score_world_240(
                source, atlas, availability, start, reflection, (design @ coefficients)[None])
            if bool((score > best_score).all()):
                best_score = score.detach()
                best_delta = delta.detach().clone()
            gradient, = torch.autograd.grad(-score.sum(), delta)
            optimizer.zero_grad(set_to_none=True)
            delta.grad = gradient
            optimizer.step()
            with torch.no_grad():
                movement = torch.linalg.vector_norm(anchors @ delta * 1000, dim=-1).mean()
                delta.mul_(2499 / movement.clamp_min(2499))
        final_trial = original + 1000 * delta
        trial_score, _ = _score_world_240(
            source, atlas, availability, start, reflection, (design @ final_trial)[None])
        if bool((trial_score > best_score).all()):
            best_delta = delta.detach().clone()
        delta = best_delta
        coefficients = original + 1000 * delta
        final_score, final_count = _score_world_240(
            source, atlas, availability, start, reflection, (design @ coefficients)[None])
        movement = torch.linalg.vector_norm(anchors @ (coefficients - original), dim=-1).mean()
        edge_u, edge_v = coefficients[1], coefficients[2]
        sine = torch.linalg.vector_norm(torch.cross(edge_u, edge_v, dim=-1)) / (
            torch.linalg.vector_norm(edge_u) * torch.linalg.vector_norm(edge_v)).clamp_min(1)
        valid_frame = bool((sine > 1e-3).detach()
                           and (torch.linalg.vector_norm(edge_u) > 1).detach()
                           and (torch.linalg.vector_norm(edge_v) > 1).detach())
        accepted = bool((final_score >= initial_score + .02).all()
                        and (movement <= 2500).all() and valid_frame)
        origin = coefficients[0] - .5 * (edge_u + edge_v)
        fitted = (full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((origin, edge_u, edge_v))[None])) if accepted else start)
        diagnostics = {
            "accepted": accepted,
            "initial_score": float(initial_score[0].detach()),
            "final_score": float(final_score[0].detach()),
            "initial_supported": int(initial_count[0]),
            "final_supported": int(final_count[0]),
            "five_anchor_movement_um": float(movement.detach()),
            "valid_frame": valid_frame,
        }
        return fitted.detach(), diagnostics
