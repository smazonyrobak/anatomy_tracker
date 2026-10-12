"""Spatial atlas comparison and bounded recurrent pose correction for pilot 265."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    render_finite_thickness_plane,
)
from training.arbitrary_plane_recurrent_model import local_correlation


BANK = 64 * 3 * 2 * 8 * 2
SQUARE_UM = 32000.
OFFSETS = (-2000., 0., 2000.)
PSF_OFFSETS = (-12.5, 0., 12.5)
PSF_WEIGHTS = (.25, .5, .25)


def candidate_bank(centre, anchor_frames):
    """One image-conditioned centre, fixed antipodal frames, no truth dependence."""
    device, dtype = centre.device, centre.dtype
    indices = torch.arange(BANK, device=device)
    base = indices // 32
    assignment = (indices // 16) % 2
    roll = (indices // 2) % 8
    flags = indices % 2
    base_frames = anchor_frames[base // 3].to(dtype)
    angle = roll * (2 * math.pi / 8)
    cosine, sine = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    rotation = torch.stack((
        torch.stack((cosine, -sine, zero), -1),
        torch.stack((sine, cosine, zero), -1),
        torch.stack((zero, zero, one), -1)), -2)
    frames = base_frames @ rotation
    spans = torch.where(assignment[:, None].bool(),
                        centre.new_tensor((24000., 16000.)),
                        centre.new_tensor((16000., 24000.)))
    basis = torch.diag_embed(spans)
    offsets = centre.new_tensor(OFFSETS)[base % 3]
    centres = centre[None] + base_frames[:, :, 2] * offsets[:, None]
    states = full_frame_state_from_components(centres, frames, basis)
    return states, flags, base


def render_coarse(atlas, centre, anchor_frames, indices, *, support_only=False):
    """Render unique square 3-D slabs, then resample exact raster geometry."""
    states, flags, base = candidate_bank(centre, anchor_frames)
    chosen_base, inverse = torch.unique(base[indices], sorted=True, return_inverse=True)
    frame = anchor_frames[chosen_base // 3].to(centre.dtype)
    offset = centre.new_tensor(OFFSETS)[chosen_base % 3]
    center = centre[None] + frame[:, :, 2] * offset[:, None]
    square = torch.diag_embed(centre.new_full((len(chosen_base), 2), SQUARE_UM))
    base_states = full_frame_state_from_components(center, frame, square)
    slab = render_finite_thickness_plane(atlas, base_states, (96, 96),
        (0., 0., 0.), (25., 25., 25.),
        centre.new_tensor(PSF_OFFSETS), centre.new_tensor(PSF_WEIGHTS))
    if support_only:
        slab = torch.cat((torch.zeros_like(slab[:, :1]), slab[:, 1:2]), 1)
    selected = states[indices]
    _, selected_frame, selected_basis = full_frame_state_to_components(selected)
    edge = selected_frame[:, :, :2] @ selected_basis
    base_frame = frame[inverse]
    local_edge = base_frame[:, :, :2].transpose(1, 2) @ edge
    axis = (torch.arange(48, device=centre.device, dtype=centre.dtype) * 4 + 1.5) / 192
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    x = torch.where(flags[indices, None, None].bool(), 191 / 192 - xx, xx)
    chart = torch.stack((x, yy.expand_as(x)), -1) - .5
    local = torch.einsum('nij,nhwj->nhwi', local_edge, chart)
    square_st = .5 + local / SQUARE_UM
    grid = square_st * (2 * 96 / 95) - 1
    rendered = F.grid_sample(slab[inverse], grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
    support = rendered[:, 1:2].clamp(0, 1)
    return torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)


def render_exact(atlas, states, flags, *, support_only=False):
    centre, frame, basis = full_frame_state_to_components(states)
    edges = frame[:, :, :2] @ basis
    # The generator samples x,y=i/192. Area-downsampling to 96 has centres
    # (2i+0.5)/192; this shift preserves the original reflection convention.
    aligned = full_frame_state_from_components(
        centre + (edges[:, :, 0] + edges[:, :, 1]) * (.5 / 192), frame, basis)
    slab = render_finite_thickness_plane(atlas, aligned, (96, 96),
        (0., 0., 0.), (25., 25., 25.),
        states.new_tensor(PSF_OFFSETS), states.new_tensor(PSF_WEIGHTS))
    if support_only:
        slab = torch.cat((torch.zeros_like(slab[:, :1]), slab[:, 1:2]), 1)
    slab = torch.where(flags[:, None, None, None].bool(), slab.flip(-1), slab)
    support = slab[:, 1:2].clamp(0, 1)
    return torch.cat((slab[:, :1] / support.clamp_min(1e-4), support), 1)


def bounded_update(raw):
    tilt = raw[:, :2].tanh()
    tilt = tilt / tilt.norm(dim=-1, keepdim=True).clamp_min(1) * math.radians(7.5)
    roll = raw[:, 2:3].tanh() * math.radians(22.5)
    shift = raw[:, 3:6].tanh()
    shift = shift / shift.norm(dim=-1, keepdim=True).clamp_min(1) * 1500.
    span = raw[:, 6:8].tanh() * .35
    shear = raw[:, 8:9].tanh() * .15
    return torch.cat((tilt, roll, shift, span, shear), -1)


def five_point_error(states, flags, truth, true_flag, side=192):
    chart = states.new_tensor(((0., 0.), (1., 0.), (0., 1.), (1., 1.), (.5, .5)))

    def points(state, reflect):
        centre, frame, basis = full_frame_state_to_components(state)
        pixels = chart.expand(len(state), -1, -1).clone()
        pixels[:, :, 0] = torch.where(reflect[:, None].bool(),
            (side - 1) / side - pixels[:, :, 0], pixels[:, :, 0])
        return centre[:, None] + torch.einsum('nij,nkj->nki',
                                              frame[:, :, :2] @ basis, pixels - .5)

    expected = points(truth.expand(len(states), -1), true_flag.expand(len(states)))
    return (points(states, flags) - expected).norm(dim=-1).mean(-1) / 1000


class SpatialPoseUpdater(nn.Module):
    """Shared spatial pair representation for score and physical pose residual."""
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(5, 24, 5, stride=2, padding=2),
            nn.GroupNorm(6, 24), nn.GELU(), nn.Conv2d(24, 24, 3, padding=1),
            nn.GroupNorm(6, 24), nn.GELU())
        self.atlas = nn.Sequential(nn.Conv2d(2, 24, 5, stride=2, padding=2),
            nn.GroupNorm(6, 24), nn.GELU(), nn.Conv2d(24, 24, 3, padding=1),
            nn.GroupNorm(6, 24), nn.GELU())
        self.pair = nn.Sequential(nn.Conv2d(24 * 3 + 25 + 3, 48, 3, padding=1),
            nn.GroupNorm(8, 48), nn.GELU(), nn.Conv2d(48, 48, 3, padding=1),
            nn.GroupNorm(8, 48), nn.GELU())
        self.output = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                    nn.Linear(48 * 4 * 4, 10))
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)

    def encode_image(self, image):
        return self.image(F.interpolate(image, (48, 48), mode='area'))

    def forward(self, query, rendered):
        atlas = self.atlas(F.interpolate(rendered, (48, 48), mode='area'))
        query = query.expand(len(atlas), -1, -1, -1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, atlas.shape[-2], device=atlas.device),
                              torch.linspace(-1, 1, atlas.shape[-1], device=atlas.device),
                              indexing='ij')
        position = torch.stack((x, y))[None].expand(len(atlas), -1, -1, -1)
        support = F.interpolate(rendered[:, 1:2], atlas.shape[-2:], mode='area')
        spatial = self.pair(torch.cat((query, atlas, (query - atlas).abs(),
            local_correlation(query, atlas, 2), support, position), 1))
        output = self.output(spatial)
        return output[:, 0], bounded_update(output[:, 1:]), spatial
