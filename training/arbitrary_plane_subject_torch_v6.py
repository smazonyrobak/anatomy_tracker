"""FP64 Torch coordinate mapping for an externally authenticated accepted plan.

Target preparation only. No plan sampling, receipt verification, learned state,
dense displacement cache or implicit CPU/device conversion occurs here.
"""

import torch


def cubic_bspline_velocity_torch_v6(points_um, coefficients_um, lattice_origin_um, lattice_spacing_um):
    """Zero-extended cardinal cubic spline, physical AP/DV/ML -> velocity um.

    Inputs are resident FP64 tensors on one device; coefficients have shape
    [AP_knots,DV_knots,ML_knots,3], origin/spacing have shape [3]. No half-voxel
    shift or boundary-weight renormalization applies to the spline lattice.
    The vectorized 64-term reduction is not promised bitwise equal to NumPy.
    """
    assert all(value.dtype == torch.float64 for value in (points_um, coefficients_um, lattice_origin_um, lattice_spacing_um))
    coordinate = (points_um.reshape(-1, 3) - lattice_origin_um) / lattice_spacing_um
    knot = torch.floor(coordinate).to(torch.int64)
    t = coordinate - knot
    weights = torch.stack((
        (1.0 - t) ** 3 / 6.0,
        (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
        (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
        t**3 / 6.0,
    ), dim=-1)
    offsets = torch.arange(4, device=points_um.device)
    neighbors = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=-1).reshape(64, 3)
    base = knot - 1
    index_ap = base[:, 0:1] + neighbors[None, :, 0]
    index_dv = base[:, 1:2] + neighbors[None, :, 1]
    index_ml = base[:, 2:3] + neighbors[None, :, 2]
    valid = ((index_ap >= 0) & (index_ap < coefficients_um.shape[0])
             & (index_dv >= 0) & (index_dv < coefficients_um.shape[1])
             & (index_ml >= 0) & (index_ml < coefficients_um.shape[2]))
    values = coefficients_um[
        index_ap.clamp(0, coefficients_um.shape[0] - 1),
        index_dv.clamp(0, coefficients_um.shape[1] - 1),
        index_ml.clamp(0, coefficients_um.shape[2] - 1),
    ]
    weight = (valid.to(points_um.dtype) * weights[:, 0, neighbors[:, 0]]
              * weights[:, 1, neighbors[:, 1]] * weights[:, 2, neighbors[:, 2]])
    return (weight[..., None] * values).sum(dim=1).reshape(points_um.shape)


def map_accepted_subject_points_torch_v6(
    points_um,
    accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um,
    accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um,
    global_scale, frozen_center_um,
    *, inverse=True, steps=8, batch_size=8192, identity=False,
):
    """Map [...,3] physical points with the accepted plan's fixed-step RK4.

    Caller converts authenticated plan state to resident FP64 tensors ONCE and
    supplies its flow step count and identity-stratum flag. No coefficient
    weighting/projection is repeated: accepted coarse and fine fields sum.
    inverse=True: subject -> inverse diagonal scale -> exp(-v) -> CCF.
    inverse=False: CCF -> exp(+v) -> diagonal scale -> subject.
    Scaling is about frozen_center_um; the finite-step flows are the original
    numerical inverse pair, not algebraically exact inverses. No Jacobian or
    autograd contract is supplied. GPU equivalence remains unverified.
    """
    assert all(value.dtype == torch.float64 for value in (
        points_um, accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um,
        accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um, global_scale, frozen_center_um,
    ))
    assert isinstance(steps, int) and steps > 0 and isinstance(batch_size, int) and batch_size > 0
    with torch.no_grad():
        if identity:
            return points_um.clone()
        flat = points_um.reshape(-1, 3)
        mapped = torch.empty_like(flat)
        h, sign = 1.0 / steps, -1.0 if inverse else 1.0
        for start in range(0, len(flat), batch_size):
            x = flat[start:start + batch_size].clone()
            if inverse:
                x = frozen_center_um + (x - frozen_center_um) / global_scale
            for _ in range(steps):
                k1 = sign * (
                    cubic_bspline_velocity_torch_v6(x, accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um)
                    + cubic_bspline_velocity_torch_v6(x, accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um)
                )
                x2 = x + 0.5 * h * k1
                k2 = sign * (
                    cubic_bspline_velocity_torch_v6(x2, accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um)
                    + cubic_bspline_velocity_torch_v6(x2, accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um)
                )
                x3 = x + 0.5 * h * k2
                k3 = sign * (
                    cubic_bspline_velocity_torch_v6(x3, accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um)
                    + cubic_bspline_velocity_torch_v6(x3, accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um)
                )
                x4 = x + h * k3
                k4 = sign * (
                    cubic_bspline_velocity_torch_v6(x4, accepted_coarse_coefficients_um, coarse_origin_um, coarse_spacing_um)
                    + cubic_bspline_velocity_torch_v6(x4, accepted_fine_coefficients_um, fine_origin_um, fine_spacing_um)
                )
                x = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if not inverse:
                x = frozen_center_um + (x - frozen_center_um) * global_scale
            mapped[start:start + batch_size] = x
        return mapped.reshape(points_um.shape)
