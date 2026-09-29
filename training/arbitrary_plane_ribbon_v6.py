"""Pose-separated curved finite slabs; canonical raster, physical micrometres.

Pure differentiable geometry, not a trained predictor. The base frame is proper;
apply any observed-raster reflection after constructing its coordinate grid.
"""

import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf


def project_surface_affine_out(residual_local_um):
    """Remove uniform full-canvas 1,s,t components from B,3,H,W residuals.

    Coefficients B,3,3 are intercept/slope-s/slope-t in the centered pixel chart.
    No mask-dependent pose gauge or constraint on the director is introduced.
    """
    height, width = residual_local_um.shape[-2:]
    x = (torch.arange(width, device=residual_local_um.device, dtype=residual_local_um.dtype) - (width - 1) / 2) / width
    y = (torch.arange(height, device=residual_local_um.device, dtype=residual_local_um.dtype) - (height - 1) / 2) / height
    mean = residual_local_um.mean((-2, -1))
    slope_x = (residual_local_um * x).mean((-2, -1)) / x.square().mean()
    slope_y = (residual_local_um * y[:, None]).mean((-2, -1)) / y.square().mean()
    residual = residual_local_um - mean[..., None, None] - slope_x[..., None, None] * x - slope_y[..., None, None] * y[:, None]
    return residual, torch.stack((mean, slope_x, slope_y), dim=-1)


def ribbon_derivative_frobenius_bound(residual_local_um, director_delta_local, inplane_basis_um, axial_offsets_um):
    """Maximum vertex derivative norm for a bilinear-in-raster, linear-z ribbon.

    Derivatives are with respect to physical coordinates of the undeformed
    plane/slab, not pixels. The physical base Jacobian is blockdiag(basis,1).
    Evaluate every raster-cell corner at both extrema of the finite z interval,
    including zero. This bounds the multiaffine derivative throughout each cell;
    it is not a centred-difference approximation or a tissue-only sample.
    """
    height, width = residual_local_um.shape[-2:]
    r, d = residual_local_um, director_delta_local
    rs = (r[..., 1:] - r[..., :-1]) * width
    rt = (r[..., 1:, :] - r[..., :-1, :]) * height
    ds = (d[..., 1:] - d[..., :-1]) * width
    dt = (d[..., 1:, :] - d[..., :-1, :]) * height
    rs = torch.stack((rs[..., :-1, :], rs[..., :-1, :], rs[..., 1:, :], rs[..., 1:, :]), dim=1)
    rt = torch.stack((rt[..., :-1], rt[..., 1:], rt[..., :-1], rt[..., 1:]), dim=1)
    ds = torch.stack((ds[..., :-1, :], ds[..., :-1, :], ds[..., 1:, :], ds[..., 1:, :]), dim=1)
    dt = torch.stack((dt[..., :-1], dt[..., 1:], dt[..., :-1], dt[..., 1:]), dim=1)
    corner_d = torch.stack((d[..., :-1, :-1], d[..., :-1, 1:], d[..., 1:, :-1], d[..., 1:, 1:]), dim=1)
    offsets = axial_offsets_um.expand(r.shape[0], -1)
    z_min, z_max = offsets.amin(-1).clamp_max(0), offsets.amax(-1).clamp_min(0)
    inverse = torch.linalg.inv(inplane_basis_um)
    squared_bounds = []
    for z in (z_min, z_max):
        derivative_s = rs + z[:, None, None, None, None] * ds
        derivative_t = rt + z[:, None, None, None, None] * dt
        derivative_u = derivative_s * inverse[:, 0, 0, None, None, None, None] + derivative_t * inverse[:, 1, 0, None, None, None, None]
        derivative_v = derivative_s * inverse[:, 0, 1, None, None, None, None] + derivative_t * inverse[:, 1, 1, None, None, None, None]
        squared_bounds.append((derivative_u.square() + derivative_v.square() + corner_d.square()).sum(dim=2).flatten(1).amax(-1))
    # A zero field stays exactly zero and has a finite derivative through sqrt.
    squared = torch.maximum(*squared_bounds)
    return torch.where(squared > 0, squared.clamp_min(torch.finfo(squared.dtype).tiny).sqrt(), torch.zeros_like(squared))


def compose_curved_ribbon_coordinates(state, residual_local_um, director_delta_local, axial_offsets_um, maximum_derivative_norm=.35):
    """Return a canonical B,S,H,W,3 CCF grid and its constrained deformation.

    State B,12 is the existing full finite-plane state. Both fields are B,3,H,W
    in the state's local proper frame: residual is um; director delta is um/um.
    X(s,t,z) = P_pose(s,t) + R*r(s,t) + z*(n + R*d(s,t)).
    Global rescaling of both affine-free r and d limits the full physical
    displacement derivative, preserving their coupled finite-slab geometry.

    For the continuous piecewise-bilinear fields, a norm bound <1 makes this
    mapping injective on its convex finite canonical domain in exact arithmetic.
    It is not a certificate outside the raster or for a subsequently composed
    section-processing map. Floating-point topology still requires measurement.
    No automatic segmentation, reflected anatomy or sampled-image warp is used.
    """
    assert 0 < maximum_derivative_norm < 1
    residual, removed_affine = project_surface_affine_out(residual_local_um.to(state))
    director_delta = director_delta_local.to(state)
    offsets = torch.as_tensor(axial_offsets_um, device=state.device, dtype=state.dtype).expand(state.shape[0], -1)
    center, frame, basis = full_frame_state_to_components(state)
    before = ribbon_derivative_frobenius_bound(residual, director_delta, basis, offsets)
    scale = (maximum_derivative_norm / before.clamp_min(maximum_derivative_norm)).clamp_max(1.)
    residual = residual * scale[:, None, None, None]
    director_delta = director_delta * scale[:, None, None, None]
    height, width = residual.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=state.device, dtype=state.dtype) / height,
                          torch.arange(width, device=state.device, dtype=state.dtype) / width, indexing="ij")
    st = torch.stack((x, y), dim=-1)[None].expand(state.shape[0], -1, -1, -1)
    plane = normalized_raster_to_ccf(center[:, None, None], frame[:, None, None], basis[:, None, None], st)
    surface = plane + torch.einsum("bij,bjhw->bhwi", frame, residual)
    director = frame[:, None, None, :, 2] + torch.einsum("bij,bjhw->bhwi", frame, director_delta)
    coordinates = surface[:, None] + offsets[:, :, None, None, None] * director[:, None]
    return {"ccf_coordinates_ap_dv_ml_um": coordinates, "centre_surface_ccf_ap_dv_ml_um": surface,
            "director_ccf": director, "residual_local_um": residual, "director_delta_local": director_delta,
            "removed_surface_affine_coefficients_local_um": removed_affine,
            "prelimit_derivative_frobenius_bound": before, "deformation_rescale": scale,
            "postlimit_derivative_frobenius_bound": before * scale}
