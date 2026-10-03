"""Cross-modal patch fit at an arbitrary full-frame atlas pose."""
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.atlas_oriented_patch_025 import extract_patches


def atlas_patches(atlas, state, reflection, offsets, weights, pixel_yx, source_side=256, patch_side=64):
    count, patches = state.shape[0], pixel_yx.shape[0]
    centre, frame, basis = full_frame_state_to_components(state)
    radius = torch.arange(patch_side, device=state.device, dtype=state.dtype) - (patch_side - 1) / 2
    yy, xx = torch.meshgrid(radius, radius, indexing='ij')
    y = pixel_yx[:, 0, None, None] + yy
    x = pixel_yx[:, 1, None, None] + xx
    chart = torch.stack((x, y), -1)[None].expand(count, -1, -1, -1, -1).clone() / source_side
    chart[..., 0] = torch.where(reflection[:, None, None, None].bool(),
                                (source_side - 1) / source_side - chart[..., 0], chart[..., 0])
    plane = normalized_raster_to_ccf(centre[:, None, None, None], frame[:, None, None, None],
                                     basis[:, None, None, None], chart)
    axial = plane[:, :, None] + offsets[:, None, :, None, None, None] * frame[:, None, None, None, None, :, 2]
    rendered = render_finite_thickness_coordinate_grid(
        atlas, axial.flatten(0, 1), (0., 0., 0.), (25., 25., 25.),
        weights[:, None].expand(-1, patches, -1).flatten(0, 1))
    support = rendered[:, 1:2].clamp(0, 1)
    return torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)


def patch_fit(model, image, atlas, state, reflection, offsets, weights, pixel_yx, source_side=256):
    source = extract_patches(image, torch.zeros(len(pixel_yx), device=image.device, dtype=torch.long),
                             pixel_yx)
    query = F.normalize(model.shared(model.image_stem(source)), dim=-1)
    rendered = atlas_patches(atlas, state, reflection, offsets, weights, pixel_yx, source_side)
    target = F.normalize(model.shared(model.atlas_stem(rendered)), dim=-1)
    similarity = (query[None] * target.reshape(len(state), len(pixel_yx), -1)).sum(-1)
    return similarity.mean(-1), similarity
