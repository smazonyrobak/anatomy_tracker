"""Atlas slabs and dense CCF targets on the same 64-grid source chart."""

import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)


def render_coherent_atlas_143(atlas, state, reflection, offsets, weights, slab_step_um=500.):
    """Return atlas [B,2,7,64,64], base CCF grid, source basis, and normal.

    Source grid (i,j) is physical 256-pixel index (4i+2,4j+2). The returned
    basis columns are CCF micrometres per unit source-grid x/y fraction;
    reflection changes the sign of its first column.
    """
    batch, side = state.shape[0], 64
    centre, frame, basis = full_frame_state_to_components(state)
    edges = frame[:, :, :2] @ basis
    normal = frame[:, :, 2]
    axis = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    chart_x = torch.where(reflection[:, None, None].bool(), 255 / 256 - xx, xx)
    chart = torch.stack((chart_x, yy.expand(batch, -1, -1)), -1)
    base = centre[:, None, None] + torch.einsum('bij,bhwj->bhwi', edges, chart - .5)
    source_basis = torch.stack((torch.where(reflection[:, None].bool(),
        -edges[:, :, 0], edges[:, :, 0]), edges[:, :, 1]), -1)
    slab_offsets = torch.arange(-3, 4, device=state.device, dtype=state.dtype) * slab_step_um
    world = (base[:, None, None] +
        (slab_offsets[None, :, None] + offsets[:, None, :])[:, :, :, None, None, None]
        * normal[:, None, None, None, None])
    rendered = render_finite_thickness_coordinate_grid(
        atlas, world.reshape(batch * 7, offsets.shape[1], side, side, 3),
        (0., 0., 0.), (25., 25., 25.),
        weights[:, None].expand(-1, 7, -1).reshape(batch * 7, -1))
    rendered = rendered.reshape(batch, 7, 2, side, side).permute(0, 2, 1, 3, 4)
    support = rendered[:, 1:2].clamp(0, 1)
    pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
    return pair, base, source_basis, normal


def dense_local_targets_143(target_centre_um, valid_mask, base, source_basis, normal,
                            radius=3, slab_step_um=500.):
    """Return local (du,dv,dnormal_um), lattice class, and reachable mask.

    Class order is z-major over seven depths, then y and x over the in-plane
    offsets [-radius,+radius]. Unreachable classes are -1.
    """
    truth = target_centre_um[:, 2::4, 2::4]
    valid = valid_mask[:, 2::4, 2::4].bool()
    transform = torch.cat((source_basis, normal[..., None]), -1)
    local = torch.linalg.solve(transform[:, None, None], (truth - base)[..., None])[..., 0]
    lattice = torch.stack((local[..., 0] * 64, local[..., 1] * 64,
                           local[..., 2] / slab_step_um), -1).round().long()
    gx = torch.arange(64, device=base.device)[None, None, :]
    gy = torch.arange(64, device=base.device)[None, :, None]
    reachable = (valid & (lattice[..., :2].abs() <= radius).all(-1) &
                 (lattice[..., 2].abs() <= 3) &
                 ((gx + lattice[..., 0]) >= 0) & ((gx + lattice[..., 0]) < 64) &
                 ((gy + lattice[..., 1]) >= 0) & ((gy + lattice[..., 1]) < 64))
    width = 2 * radius + 1
    classes = ((lattice[..., 2] + 3) * width + lattice[..., 1] + radius) * width + lattice[..., 0] + radius
    return local, torch.where(reachable, classes, -1), reachable
