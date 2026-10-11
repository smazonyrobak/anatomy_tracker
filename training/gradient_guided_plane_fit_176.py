"""Bounded, first-order rigid atlas fitting for blind arbitrary-plane candidates."""

import torch
from torch import nn
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.global_atlas_contrast_090 import rigid_points_090


def _render(atlas, state, reflection, chart, offsets, weights):
    count = len(state)
    plane = rigid_points_090(state, reflection, chart, (256, 256)).reshape(count, 48, 48, 3)
    normal = full_frame_state_to_components(state)[1][..., :, 2]
    world = plane[:, None] + offsets[:, :, None, None, None] * normal[:, None, None, None, :]
    sampled = render_finite_thickness_coordinate_grid(
        atlas, world, (0., 0., 0.), (25., 25., 25.), weights)
    support = sampled[:, 1:2].clamp(0, 1)
    return sampled[:, :1] / support.clamp_min(1e-4), support


def _score(source_smooth, atlas_image, mask):
    atlas_smooth = F.avg_pool2d(atlas_image, 3, 1, 1)
    count = mask.sum((-2, -1), keepdim=True).clamp_min(1)
    source_mean = (source_smooth * mask).sum((-2, -1), keepdim=True) / count
    atlas_mean = (atlas_smooth * mask).sum((-2, -1), keepdim=True) / count
    source_centered = (source_smooth - source_mean) * mask
    atlas_centered = (atlas_smooth - atlas_mean) * mask
    covariance = (source_centered * atlas_centered).sum((-2, -1))
    variance = source_centered.square().sum((-2, -1)) * atlas_centered.square().sum((-2, -1))
    correlation = covariance / variance.clamp_min(1e-16).sqrt()
    return (correlation.square() + 1e-4).sqrt()[:, 0]


def _permutation(mask):
    index = torch.arange(48 * 48, device=mask.device)[None].expand(len(mask), -1).clone()
    for row in range(len(mask)):
        pixels = mask[row, 0].flatten().nonzero()[:, 0]
        if len(pixels):
            index[row, pixels] = pixels.roll(max(1, len(pixels) // 3))
    return index


class GradientGuidedPlaneFitter176(nn.Module):
    """Fit B,K,12 states with two bounded 9D steps; never selects candidates."""

    def __init__(self, candidate_chunk=4):
        super().__init__()
        self.candidate_chunk = candidate_chunk

    def forward(self, inputs, states, reflection, offsets, weights, atlas, intensity_null=False):
        batch, candidates = states.shape[:2]
        axis = ((torch.arange(48, device=states.device, dtype=states.dtype) + .5) / 48
                - .5 / 256)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        chart = torch.stack((xx, yy), -1).reshape(-1, 2)
        source = F.avg_pool2d(F.interpolate(inputs[:, :1], (48, 48), mode='area'), 3, 1, 1)
        offsets = torch.as_tensor(offsets, device=states.device, dtype=states.dtype)
        weights = torch.as_tensor(weights, device=states.device, dtype=states.dtype)
        if offsets.ndim == 1:
            offsets = offsets[None].expand(batch, -1)
        if weights.ndim == 1:
            weights = weights[None].expand(batch, -1)
        scale = states.new_tensor([.12, .12, .12, 900., 900., 900., .08, .08, .06])
        output = {key: [] for key in ('original_state', 'state', 'original_fit', 'fit',
                  'gradient_norm', 'support_sites', 'support_retention', 'local_update')}

        with torch.enable_grad():
            for start in range(0, candidates, self.candidate_chunk):
                stop = min(start + self.candidate_chunk, candidates)
                width = stop - start
                state = states[:, start:stop].reshape(-1, 12)
                reflected = reflection[:, start:stop].reshape(-1)
                z = offsets[:, None].expand(-1, width, -1).reshape(batch * width, -1)
                w = weights[:, None].expand(-1, width, -1).reshape(batch * width, -1)
                image = source[:, None].expand(-1, width, -1, -1, -1).reshape(-1, 1, 48, 48)
                original_atlas, original_support = _render(atlas, state, reflected, chart, z, w)
                mask = (F.avg_pool2d((original_support.detach() > .5).to(state.dtype),
                                     7, 1, 3, count_include_pad=True) >= 1 - 1e-6).to(state.dtype)
                sites = mask.sum((1, 2, 3))
                permutation = _permutation(mask) if intensity_null else None

                def measured(atlas_image):
                    if permutation is not None:
                        atlas_image = atlas_image.flatten(1).gather(1, permutation).reshape(-1, 1, 48, 48)
                    return _score(image, atlas_image, mask)

                original_fit = measured(original_atlas)
                q = state.new_zeros(len(state), 9)
                norms = []
                for _ in range(2):
                    q = q.detach().requires_grad_(True)
                    trial = compose_full_frame_state(state, q * scale)
                    atlas_image, support = _render(atlas, trial, reflected, chart, z, w)
                    fit = measured(atlas_image)
                    retention = (support * mask).sum((1, 2, 3)) / sites.clamp_min(1)
                    energy = 1 - fit + .5 * (1 - retention) + .02 * q.square().sum(-1)
                    gradient = torch.autograd.grad(energy.sum(), q, create_graph=False)[0].detach()
                    norm = gradient.norm(dim=-1)
                    norms.append(norm)
                    q = q.detach() - .5 * gradient / norm.clamp_min(1e-8)[:, None] * (sites >= 100)[:, None]

                corrected = compose_full_frame_state(state, q.detach() * scale)
                atlas_image, support = _render(atlas, corrected, reflected, chart, z, w)
                output['original_state'].append(state.reshape(batch, width, 12))
                output['state'].append(corrected.reshape(batch, width, 12))
                output['original_fit'].append(original_fit.reshape(batch, width))
                output['fit'].append(measured(atlas_image).reshape(batch, width))
                output['gradient_norm'].append(torch.stack(norms).mean(0).reshape(batch, width))
                output['support_sites'].append(sites.reshape(batch, width))
                output['support_retention'].append(
                    ((support * mask).sum((1, 2, 3)) / sites.clamp_min(1)).reshape(batch, width))
                output['local_update'].append((q.detach() * scale).reshape(batch, width, 9))

        return {key: torch.cat(value, 1) for key, value in output.items()}


class FitEvidenceRanker176(nn.Module):
    """Return B,K,2 no-shift/corrected logits from fit evidence plus blind prior."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prior, fit_output):
        original = fit_output['original_fit']
        corrected = fit_output['fit']
        evidence = torch.stack((original, corrected, corrected - original,
                                fit_output['gradient_norm'].log1p(),
                                fit_output['support_retention']), -1)
        logits = prior[..., None] + self.net(evidence)
        valid = fit_output['support_sites'] >= 100
        return torch.stack((torch.where(valid, logits[..., 0], prior),
                            logits[..., 1].masked_fill(~valid, -1e4)), -1)
