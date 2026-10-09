"""Optional global atlas-conditioned correction for one-shot plane candidates."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf


def attach_global_plane_matcher(model, enabled=False):
    if enabled:
        model.global_plane_matcher_120 = nn.ModuleDict({
            'source': nn.Sequential(
                nn.Conv2d(69, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 64, 1)),
            'atlas': nn.Sequential(
                nn.Conv2d(7, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 64, 1)),
            'head': nn.Sequential(
                nn.Conv2d(201, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(1024, 128), nn.GELU(),
                nn.Linear(128, 10)),
        }).to(model.center_origin)
        nn.init.zeros_(model.global_plane_matcher_120['head'][-1].weight)
        nn.init.zeros_(model.global_plane_matcher_120['head'][-1].bias)
    return model


def global_plane_match(model, prediction, mode_index, reflection, offsets, weights, atlas,
                       image_shape, side=24, support_only=False, candidate_state=None,
                       matcher=None):
    batch, count = mode_index.shape
    row = torch.arange(batch, device=mode_index.device)[:, None]
    state = (prediction['state'][row, mode_index] if candidate_state is None
             else candidate_state)
    reflection_log = torch.where(reflection.bool(),
                                 F.logsigmoid(prediction['reflection_logit'][row, mode_index]),
                                 F.logsigmoid(-prediction['reflection_logit'][row, mode_index]))
    prior = prediction['log_mass'][row, mode_index] + reflection_log
    update = state.new_zeros(batch, count, 9)
    match_logit = prior.new_zeros(batch, count)
    if matcher is None and not hasattr(model, 'global_plane_matcher_120'):
        return {'input_state': state, 'input_score': prior, 'state': state,
                'score': prior, 'match_logit': match_logit, 'update': update}

    flat_state = state.flatten(0, 1)
    reflected = reflection.flatten().bool()
    center, frame, basis = full_frame_state_to_components(flat_state)
    y, x = torch.meshgrid((torch.arange(side, device=state.device, dtype=state.dtype) + .5)
                          / side - .5 / image_shape[0],
                          (torch.arange(side, device=state.device, dtype=state.dtype) + .5)
                          / side - .5 / image_shape[1],
                          indexing='ij')
    sx = torch.where(reflected[:, None, None],
                     (image_shape[1] - 1) / image_shape[1] - x, x)
    chart = torch.stack((sx.expand(-1, side, -1),
                         y.expand(len(flat_state), -1, -1)), -1)
    plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None],
                                     basis[:, None, None], chart)
    position = ((plane - model.center_origin) / model.center_scale).permute(0, 3, 1, 2)
    xy = torch.stack((x, y))[None].expand(len(flat_state), -1, -1, -1)

    axial = torch.as_tensor(offsets, device=state.device, dtype=state.dtype)
    mass = torch.as_tensor(weights, device=state.device, dtype=state.dtype)
    if axial.ndim == 1:
        axial = axial[None].expand(batch, -1)
    if mass.ndim == 1:
        mass = mass[None].expand(batch, -1)
    axial = axial[:, None].expand(-1, count, -1).flatten(0, 1)
    mass = mass[:, None].expand(-1, count, -1).flatten(0, 1)
    coordinates = plane[:, None] + axial[:, :, None, None, None] * frame[:, None, None, None, :, 2]
    rendered = render_finite_thickness_coordinate_grid(
        atlas, coordinates, (0., 0., 0.), (25., 25., 25.), mass)
    support = rendered[:, 1:2]
    intensity = rendered[:, :1] / support.clamp_min(1e-4)
    if support_only:
        intensity = torch.zeros_like(intensity)
    atlas_pair = torch.cat((intensity, support), 1)

    source = F.adaptive_avg_pool2d(prediction['feature'], (side, side))
    source = source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
    matcher = model.global_plane_matcher_120 if matcher is None else matcher
    query = matcher['source'](torch.cat((source, xy, position), 1)).flatten(2).transpose(1, 2)
    key = matcher['atlas'](torch.cat((atlas_pair, xy, position), 1)).flatten(2).transpose(1, 2)
    attention = (query @ key.transpose(1, 2) / math.sqrt(query.shape[-1])).softmax(-1)
    matched = attention @ key
    physical = position.flatten(2).transpose(1, 2)
    displacement = attention @ physical - physical
    evidence = torch.cat((query, matched, query - matched, displacement,
                          attention.amax(-1, keepdim=True),
                          xy.flatten(2).transpose(1, 2), physical), -1)
    raw = matcher['head'](evidence.transpose(1, 2).reshape(batch * count, 201, side, side))
    raw = raw.reshape(batch, count, 10)
    limits = state.new_tensor((.9, .9, .9, 4000., 4000., 4000., .25, .25, .2))
    update = raw[..., :9].tanh() * limits
    match_logit = raw[..., 9]
    return {'input_state': state, 'input_score': prior,
            'state': compose_full_frame_state(state, update), 'score': prior + match_logit,
            'match_logit': match_logit, 'update': update}
