"""Lift the coherent field's single local map to the observed image grid."""

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components


def coherent_surface_144(state, reflection, offset_chart, image_shape):
    """CCF AP/DV/ML micrometres at each observed pixel, without a second warp."""
    height, width = image_shape
    field_height, field_width = offset_chart.shape[-2:]
    y = torch.arange(height, device=state.device, dtype=state.dtype) / height
    x = torch.arange(width, device=state.device, dtype=state.dtype) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sample = torch.stack((2 * (xx * field_width - .5) / (field_width - 1) - 1,
                          2 * (yy * field_height - .5) / (field_height - 1) - 1), -1)
    local = F.grid_sample(offset_chart, sample[None].expand(len(state), -1, -1, -1),
                          mode='bilinear', padding_mode='border', align_corners=True)
    local = local.permute(0, 2, 3, 1)
    centre, frame, basis = full_frame_state_to_components(state)
    edges = frame[:, :, :2] @ basis
    sx = torch.where(reflection[:, None, None].bool(), (width - 1) / width - xx, xx)
    chart = torch.stack((sx.expand(len(state), -1, -1), yy.expand(len(state), -1, -1)), -1)
    base = centre[:, None, None] + torch.einsum('bij,bhwj->bhwi', edges, chart - .5)
    source_basis = torch.stack((torch.where(reflection[:, None].bool(),
        -edges[:, :, 0], edges[:, :, 0]), edges[:, :, 1]), -1)
    return (base + torch.einsum('bij,bhwj->bhwi', source_basis, local[..., :2]) +
            frame[:, None, None, :, 2] * local[..., 2:3])
