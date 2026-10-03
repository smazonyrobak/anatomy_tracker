"""Finite-thickness atlas patches at shortlisted 3D correspondence keys."""
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.atlas_oriented_patch_025 import atlas_surface_image


def atlas_key_patches(atlas, key_world, state, reflection, offsets, weights):
    _, frame, basis = full_frame_state_to_components(state)
    edges = frame[..., :, :2] @ basis
    horizontal = torch.where(reflection[:, None].bool(), -edges[..., 0], edges[..., 0])
    vertical = edges[..., 1]
    axis = torch.arange(64, device=key_world.device, dtype=key_world.dtype) - 31.5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    coordinates = (key_world[:, None, None] +
                   xx[None, :, :, None] * horizontal[:, None, None] / 256 +
                   yy[None, :, :, None] * vertical[:, None, None] / 256)
    return atlas_surface_image(atlas, coordinates, state, offsets, weights)
