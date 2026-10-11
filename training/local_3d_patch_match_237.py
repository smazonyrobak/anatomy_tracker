"""Local, untrained 2-D/3-D atlas patch matches around one candidate plane."""

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.global_atlas_contrast_090 import rigid_points_090


def local_3d_patch_match(source, state, reflection, atlas, offsets, weights, null_seed=None):
    axis = (torch.arange(44, device=source.device, dtype=source.dtype) - 5.5) / 32 - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    atlas_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    plane = rigid_points_090(state, reflection, atlas_chart)[0].reshape(44, 44, 3)
    normal = full_frame_state_to_components(state)[1][0, :, 2]
    depth = torch.linspace(-2500, 2500, 11, device=source.device, dtype=source.dtype)
    axial = depth[:, None] + offsets[0, None]
    world = plane[None, None] + axial[:, :, None, None, None] * normal
    rendered = render_finite_thickness_coordinate_grid(
        atlas, world, (0., 0., 0.), (25., 25., 25.), weights.expand(11, -1))
    support = rendered[:, 1].clamp(0, 1)
    intensity = rendered[:, 0] / support.clamp_min(1e-4)
    if null_seed is not None:
        intensity = intensity.clone()
        generator = torch.Generator(device=source.device).manual_seed(int(null_seed))
        for index in range(11):
            pixels = (support[index] > .5).flatten().nonzero().flatten()
            order = torch.randperm(len(pixels), device=source.device, generator=generator)
            values = intensity[index].flatten()
            values[pixels] = values[pixels][order]

    small = F.interpolate(source, (32, 32), mode='area')
    source_patches = F.unfold(F.pad(small, (2, 2, 2, 2), mode='replicate'), 5)
    qy, qx = torch.meshgrid(torch.arange(0, 32, 2, device=source.device),
                            torch.arange(0, 32, 2, device=source.device), indexing='ij')
    qy, qx = qy.flatten(), qx.flatten()
    query_patches = source_patches[0, :, qy * 32 + qx].T
    texture = query_patches.std(-1, unbiased=False)
    query_patches = F.normalize(query_patches - query_patches.mean(-1, keepdim=True),
                                dim=-1, eps=1e-6)
    atlas_patches = F.unfold(F.pad(intensity[:, None], (2, 2, 2, 2),
                                   mode='replicate'), 5)
    atlas_patches = F.normalize(atlas_patches - atlas_patches.mean(1, keepdim=True),
                                dim=1, eps=1e-6)
    dy, dx = torch.meshgrid(torch.arange(-6, 7, device=source.device),
                            torch.arange(-6, 7, device=source.device), indexing='ij')
    key_y = qy[:, None] + 6 + dy.flatten()[None]
    key_x = qx[:, None] + 6 + dx.flatten()[None]
    key_index = key_y * 44 + key_x
    key_patches = atlas_patches[:, :, key_index].permute(2, 0, 3, 1)
    correlation = torch.einsum('qp,qdkp->qdk', query_patches, key_patches).abs()
    available = (support > .5).flatten(1)[:, key_index].permute(1, 0, 2)
    local_scores = correlation.flatten(1).masked_fill(~available.flatten(1), -torch.inf)
    matched = local_scores.argmax(-1)
    supported = available.flatten(1).any(-1)
    best = local_scores.gather(1, matched[:, None])[:, 0]
    best = torch.where(supported, best, torch.zeros_like(best))
    median = local_scores.masked_fill(~available.flatten(1), torch.nan).nanmedian(-1).values
    median = torch.nan_to_num(median, nan=0.)
    confidence = (best - median).clamp_min(0) * best * (texture / .1).clamp(0, 1)
    confidence = torch.where(supported, confidence, torch.zeros_like(confidence))

    depth_index, local_index = matched.div(169, rounding_mode='floor'), matched.remainder(169)
    pixel_index = key_index[torch.arange(256, device=source.device), local_index]
    matched_world = plane.reshape(-1, 3)[pixel_index] + depth[depth_index, None] * normal
    query_chart = torch.stack(((qx + .5) / 32 - .5 / 256,
                               (qy + .5) / 32 - .5 / 256), -1)
    query_world = rigid_points_090(state, reflection, query_chart)[0]
    matched_world = torch.where(supported[:, None], matched_world, query_world)
    return {'matched_world': matched_world, 'query_chart': query_chart,
            'query_world': query_world, 'score': best, 'confidence': confidence,
            'support': supported}
