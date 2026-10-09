"""Support-matched local atlas evidence for the frozen-122 blind beam."""

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf


def make_atlas_anatomy_ranker_124():
    return nn.ModuleDict({
        'image': nn.Sequential(
            nn.Conv2d(65, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1)),
        'atlas': nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.GroupNorm(8, 16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1)),
    })


def atlas_anatomy_rank_124(ranker, prediction, inputs, state, reflection, atlas,
                           *, mapped_coordinates=None, local_displacement_um=None,
                           support_only=False, side=32, min_common=32):
    """Positive pair margin means the row plane is easier to fit than the column.

    Only input pixels, predicted features, and rendered atlas values enter the
    score. The fixed 62.5-um PSF is identical for TRAIN and deployment.
    """
    batch, candidates = state.shape[:2]
    radius, depth_offsets = 2, (-500., 0., 500.)
    flat = state.flatten(0, 1)
    reflected = reflection.flatten().bool()
    centre, frame, basis = full_frame_state_to_components(flat)
    extent = side + 2 * radius
    axis = (torch.arange(extent, device=state.device, dtype=state.dtype) + .5 - radius) / side
    yy, xx = torch.meshgrid(axis - .5 / inputs.shape[-2],
                            axis - .5 / inputs.shape[-1], indexing='ij')
    chart = torch.stack((torch.where(reflected[:, None, None],
        (inputs.shape[-1] - 1) / inputs.shape[-1] - xx, xx),
        yy.expand(len(flat), -1, -1)), -1)
    if mapped_coordinates is None:
        plane = normalized_raster_to_ccf(centre[:, None, None], frame[:, None, None],
                                         basis[:, None, None], chart)
    else:
        surface = mapped_coordinates[:, :, mapped_coordinates.shape[2] // 2].flatten(0, 1)
        surface = F.interpolate(surface.permute(0, 3, 1, 2), (side, side),
                                mode='bilinear', align_corners=False)
        plane = F.pad(surface, (radius, radius, radius, radius),
                      mode='replicate').permute(0, 2, 3, 1)
    depth = state.new_tensor(depth_offsets)
    psf = torch.linspace(-31.25, 31.25, 9, device=state.device, dtype=state.dtype)
    mass = state.new_tensor((1, 2, 2, 2, 2, 2, 2, 2, 1)) / 16
    world = (plane[:, None, None] + (depth[None, :, None] + psf[None, None, :])[
        ..., None, None, None] * frame[:, None, None, None, None, :, 2])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, world.reshape(-1, 9, extent, extent, 3),
        (0., 0., 0.), (25., 25., 25.), mass)
    rendered = rendered.reshape(len(flat), 3, 2, extent, extent).permute(0, 2, 1, 3, 4)
    support = rendered[:, 1:2].clamp(0, 1)
    intensity = rendered[:, :1] / support.clamp_min(1e-4)
    if support_only:
        intensity = torch.zeros_like(intensity)
    key = F.normalize(ranker['atlas'](torch.cat((intensity, support), 1)), dim=1)
    source = torch.cat((F.adaptive_avg_pool2d(prediction['feature'], (side, side)),
                        F.adaptive_avg_pool2d(inputs[:, :1], (side, side))), 1)
    query = F.normalize(ranker['image'](source), dim=1)
    query = query[:, None].expand(-1, candidates, -1, -1, -1).flatten(0, 1)

    matches, visibility, offsets = [], [], []
    for z in range(3):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = radius + dy, radius + dx
                matches.append((query * key[:, :, z, y:y + side, x:x + side]).sum(1))
                visibility.append(support[:, 0, z, y:y + side, x:x + side])
                offsets.append((dx, dy, z - 1))
    offsets = state.new_tensor(offsets)
    displacement_cost = (.6 * offsets[:, :2].square().sum(-1)
                         + .8 * offsets[:, 2].square())
    logits = (8 * torch.stack(matches, 1) - displacement_cost[None, :, None, None]
              + torch.stack(visibility, 1).clamp_min(1e-4).log())
    probability = logits.softmax(1)
    movement = (probability[:, :, None] * offsets[None, :, :, None, None]).sum(1)
    movement_cost = (probability * displacement_cost[None, :, None, None]).sum(1)
    dx = F.pad((movement[..., 1:] - movement[..., :-1]).square().sum(1), (0, 1))
    dy = F.pad((movement[..., 1:, :] - movement[..., :-1, :]).square().sum(1), (0, 0, 0, 1))
    local_cost = (-torch.logsumexp(logits, 1) + .10 * movement_cost
                  + .10 * (dx + dy))
    if local_displacement_um is not None:
        local = F.interpolate(local_displacement_um.flatten(0, 1), (side, side),
                              mode='bilinear', align_corners=False) / 1000
        edges = frame[:, :, :2] @ basis
        spacing_x = edges[:, :, 0].norm(dim=-1).clamp_min(1)[:, None, None] / side / 1000
        spacing_y = edges[:, :, 1].norm(dim=-1).clamp_min(1)[:, None, None] / side / 1000
        strain_x = F.pad((local[..., 1:] - local[..., :-1]).square().sum(1)
                         / spacing_x.square(), (0, 1))
        strain_y = F.pad((local[..., 1:, :] - local[..., :-1, :]).square().sum(1)
                         / spacing_y.square(), (0, 0, 0, 1))
        local_cost = (local_cost + .10 * local.square().sum(1).clamp_max(9)
                      + .10 * (strain_x + strain_y).clamp_max(9))

    centre_support = support[:, 0, 1, radius:-radius, radius:-radius] > .95
    interior = F.avg_pool2d(centre_support[:, None].to(state.dtype), 5, 1, 2)[:, 0] > .999
    local_cost = local_cost.reshape(batch, candidates, side, side)
    centre_support = centre_support.reshape(batch, candidates, side, side)
    interior = interior.reshape(batch, candidates, side, side)
    common = interior[:, :, None] & interior[:, None, :]
    count = common.sum((-2, -1))
    valid = count >= min_common
    eye = torch.eye(candidates, device=state.device, dtype=torch.bool)[None]
    valid = valid & ~eye
    difference = (local_cost[:, None] - local_cost[:, :, None]).clamp(-2, 2)
    margin = (difference * common).sum((-2, -1)) / count.clamp_min(1)
    margin = torch.where(valid, margin, torch.zeros_like(margin))
    evidence = 2 * torch.tanh(margin).sum(-1) / valid.sum(-1).clamp_min(1)
    return {'evidence': evidence, 'pair_margin': margin,
            'pair_common_count': count, 'pair_valid': valid,
            'local_cost': local_cost, 'centre_support': centre_support,
            'interior_support': interior}
