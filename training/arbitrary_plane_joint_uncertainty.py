"""Local joint frame/SVF uncertainty; raw distributions, never calibrated claims.

The Gaussian coordinates are [local SO(3) rotation3, local translation3,
log-basis-diagonal2, shear1, smooth-velocity coefficients26]. Scale these units
explicitly when training a head. Global plane and raster-reflection modes remain
discrete: sample a cell AND its representation before drawing these residuals.
"""

import math

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state,
    full_frame_state_to_components,
)


def affine_free_velocity_basis(output_shape_h_w, *, device=None, dtype=torch.float32):
    """Build once and retain 26 smooth fields, shape (26,2,H,W), in the canvas gauge.

    Bilinearly lift the 32 coordinates of a 4x4 vector grid, remove each vector
    component's constant/x/y fields, and orthonormalize. Each basis field has
    mean squared vector magnitude one, so coefficient units are pixels.
    """
    height, width = output_shape_h_w
    if min(height, width) < 4:
        raise ValueError("the 4x4 velocity basis requires both raster dimensions >=4")
    control = torch.eye(32, dtype=torch.float64).reshape(32, 2, 4, 4)
    fields = F.interpolate(control, (height, width), mode="bilinear", align_corners=True)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, dtype=torch.float64),
        torch.linspace(-1, 1, width, dtype=torch.float64), indexing="ij",
    )
    affine = torch.stack((torch.ones_like(y), y, x)).flatten(1).T
    affine = torch.block_diag(affine, affine)
    affine, _ = torch.linalg.qr(affine, mode="reduced")
    matrix = fields.flatten(1).T
    matrix = matrix - affine @ (affine.T @ matrix)
    vectors, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    vectors = vectors[:, :26]
    peak = vectors.abs().argmax(dim=0)
    vectors = vectors * vectors[peak, torch.arange(26)].sign()
    return (vectors.T.reshape(26, 2, height, width) * math.sqrt(height * width)).to(
        device=device, dtype=dtype,
    )


def velocity_residual_coefficients(velocity_residual_yx, basis):
    """Project a fully observed residual; report omitted mean squared vector error.

    This is not an estimator from a partially observed tissue mask. Use only
    identifiable synthetic fields in the same uniform-canvas gauge. Residual
    energy outside this basis is NOT represented by the sampled uncertainty.
    """
    pixels = basis.shape[-2] * basis.shape[-1]
    coefficients = torch.einsum("...cyx,qcyx->...q", velocity_residual_yx, basis) / pixels
    reconstruction = torch.einsum("...q,qcyx->...cyx", coefficients, basis)
    omitted_energy = (velocity_residual_yx - reconstruction).square().sum(-3).mean((-2, -1))
    return coefficients, omitted_energy


def local_rotation_log(rotation):
    """SO(3) logarithm in the open angle<pi chart; reject its ambiguous cut locus."""
    r00, r11, r22 = rotation[..., 0, 0], rotation[..., 1, 1], rotation[..., 2, 2]
    qabs = torch.stack((
        1 + r00 + r11 + r22, 1 + r00 - r11 - r22,
        1 - r00 + r11 - r22, 1 - r00 - r11 + r22,
    ), dim=-1).clamp_min(torch.finfo(rotation.dtype).eps ** 2).sqrt()
    dx = rotation[..., 2, 1] - rotation[..., 1, 2]
    dy = rotation[..., 0, 2] - rotation[..., 2, 0]
    dz = rotation[..., 1, 0] - rotation[..., 0, 1]
    xy = rotation[..., 0, 1] + rotation[..., 1, 0]
    xz = rotation[..., 0, 2] + rotation[..., 2, 0]
    yz = rotation[..., 1, 2] + rotation[..., 2, 1]
    candidates = torch.stack((
        torch.stack((qabs[..., 0].square(), dx, dy, dz), dim=-1),
        torch.stack((dx, qabs[..., 1].square(), xy, xz), dim=-1),
        torch.stack((dy, xy, qabs[..., 2].square(), yz), dim=-1),
        torch.stack((dz, xz, yz, qabs[..., 3].square()), dim=-1),
    ), dim=-2) / (2 * qabs.clamp_min(0.1)[..., :, None])
    index = qabs.argmax(dim=-1)
    quaternion = torch.gather(
        candidates, -2, index[..., None, None].expand(*index.shape, 1, 4),
    ).squeeze(-2)
    quaternion = quaternion * torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    norm = torch.linalg.vector_norm(quaternion[..., 1:], dim=-1)
    angle = 2 * torch.atan2(norm, quaternion[..., 0])
    if bool((angle >= math.pi - 8 * torch.finfo(rotation.dtype).eps).any()):
        raise ValueError("local rotation residual is ambiguous at angle pi; retain distinct modes")
    scale = torch.where(norm > 1e-8, angle / norm.clamp_min(1e-12), 2 + norm.square() / 3)
    return quaternion[..., 1:] * scale[..., None]


def full_frame_residual(reference_state, target_state):
    """Inverse of compose_full_frame_state, without identifying raster reflections.

    Returned coordinates are radians3, micrometres3, log-scale2 and shear1.
    A local Gaussian in this chart is not a global distribution over SO(3).
    """
    center, frame, basis = full_frame_state_to_components(reference_state)
    target_center, target_frame, target_basis = full_frame_state_to_components(target_state)
    rotation = local_rotation_log(frame.transpose(-1, -2) @ target_frame)
    translation = (frame.transpose(-1, -2) @ (target_center - center)[..., None]).squeeze(-1)
    delta_basis = torch.linalg.solve(basis, target_basis)
    log_diagonal = torch.diagonal(delta_basis, dim1=-2, dim2=-1).log()
    shear = delta_basis[..., 0, 1] / delta_basis[..., 1, 1]
    return torch.cat((rotation, translation, log_diagonal, shear[..., None]), dim=-1)


def joint_lowrank_parameters(raw, rank=4, minimum_scale=1e-4):
    """Decode one 35*(rank+1) head; positive diagonal SD and shared loadings."""
    dimension = 35
    scale = F.softplus(raw[..., :dimension].float()) + minimum_scale
    factor = raw[..., dimension:].float().reshape(*raw.shape[:-1], dimension, rank)
    return scale, factor


def lowrank_gaussian_nll(residual, diagonal_scale, factor, observed=None):
    """Exact Gaussian NLL/marginal for diag(scale**2)+factor@factor.T.

    ``observed`` selects dimensions independently for each sample. Missing
    targets may contain NaN: they are excluded, never trained as zero targets.
    All-missing samples return zero and must be excluded from loss averaging.
    Compute in FP32 (or FP64 for FP64 inputs), outside mixed precision.
    """
    dtype = torch.float64 if residual.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=residual.device.type, enabled=False):
        residual = residual.to(dtype)
        scale = diagonal_scale.to(dtype)
        factor = factor.to(dtype)
        observed = torch.ones_like(residual, dtype=torch.bool) if observed is None else observed.bool()
        whitened = torch.where(observed, residual, 0) / scale
        loading = torch.where(observed[..., None], factor, 0) / scale[..., None]
        rank = factor.shape[-1]
        gram = torch.eye(rank, device=factor.device, dtype=dtype) + loading.transpose(-1, -2) @ loading
        cholesky = torch.linalg.cholesky(gram)
        latent = torch.cholesky_solve(loading.transpose(-1, -2) @ whitened[..., None], cholesky)
        # Equivalent to Woodbury's quadratic, without subtracting large equal terms.
        remainder = whitened - (loading @ latent).squeeze(-1)
        quadratic = remainder.square().sum(-1) + latent.square().sum((-2, -1))
        logdet = 2 * torch.where(observed, scale.log(), 0).sum(-1)
        logdet = logdet + 2 * torch.diagonal(cholesky, dim1=-2, dim2=-1).log().sum(-1)
        return 0.5 * (quadratic + logdet + observed.sum(-1).to(dtype) * math.log(2 * math.pi))


def sample_joint_frame_velocity(
    mean_frame_state, diagonal_scale, factor, basis, sample_count,
    *, coordinate_scale=None, generator=None,
):
    """Draw correlated frame and smooth SVF residuals, with sample axis first.

    Call AFTER sampling the discrete cell and raster representation. Inputs must
    describe that same component. ``coordinate_scale`` reverses training-unit
    normalization. Returned velocity is a perturbation to add before existing
    SVF integration; no clipping, topology rejection or calibration is hidden.
    Samples outside the local angle<pi chart are flagged, not silently redrawn.
    """
    shape = (sample_count, *diagonal_scale.shape)
    independent = torch.randn(shape, device=factor.device, dtype=factor.dtype, generator=generator)
    latent = torch.randn(
        (sample_count, *factor.shape[:-2], factor.shape[-1]),
        device=factor.device, dtype=factor.dtype, generator=generator,
    )
    residual = independent * diagonal_scale + (factor @ latent[..., None]).squeeze(-1)
    if coordinate_scale is not None:
        residual = residual * torch.as_tensor(coordinate_scale, device=residual.device, dtype=residual.dtype)
    frame_residual = residual[..., :9]
    mean = mean_frame_state.to(residual).expand(sample_count, *mean_frame_state.shape)
    return {
        "frame_state": compose_full_frame_state(mean, frame_residual),
        "frame_residual": frame_residual,
        "velocity_perturbation_yx_px": torch.einsum("...q,qcyx->...cyx", residual[..., 9:], basis.to(residual)),
        "joint_residual": residual,
        "shared_latent": latent,
        "inside_local_rotation_chart": torch.linalg.vector_norm(frame_residual[..., :3], dim=-1) < math.pi,
        "probabilities_calibrated": False,
    }


def joint_uncertainty_nll(
    distribution, target_frame_state, target_velocity_yx_px, basis, observed,
    *, detach_mean=True,
):
    """Exact local 35D objective for explicit (B,K,R,35) observed coordinates.

    Targets must use the selected component's canonical frame and source-raster
    velocity gauge. Mark ONLY truth-compatible cell/representation components
    observed; do not label every reflection or distant plane with one target.
    Missing frame/velocity targets can be None. A partially observed frame still
    needs a valid complete target frame; missing entries of ``observed`` are not
    interpreted as zero residuals. All-missing components are excluded by caller.

    SVFs must be identifiable, fully observed synthetic fields in the same
    uniform-canvas gauge. Dense-censored/pose-only rows never supervise SVF here.
    Return out-of-basis energy rather than pretending 26 coefficients cover every
    deformation. Stop-gradient means protect the deterministic loss from this
    likelihood; gradients through the covariance's shared context remain live.
    """
    mean_frame = distribution["mean_frame_state"].float()
    mean_velocity = distribution["mean_stationary_velocity_yx_px"].float()
    if detach_mean:
        mean_frame, mean_velocity = mean_frame.detach(), mean_velocity.detach()
    observed = observed.bool().clone()
    observed[..., 9:] &= distribution["velocity_supervision_eligible"][:, None, None, None]
    if target_frame_state is None:
        observed[..., :9] = False
        target_frame_state = mean_frame
    if target_velocity_yx_px is None:
        observed[..., 9:] = False
        target_velocity_yx_px = mean_velocity
    with torch.autocast(device_type=mean_frame.device.type, enabled=False):
        frame_target = torch.where(
            observed[..., :9].any(-1)[..., None], target_frame_state.float(), mean_frame
        )
        frame_residual = full_frame_residual(mean_frame, frame_target)
        velocity_observed = observed[..., 9:].any(-1)
        velocity_residual = torch.where(
            velocity_observed[..., None, None, None],
            target_velocity_yx_px.float() - mean_velocity, 0,
        )
        coefficients, omitted_energy = velocity_residual_coefficients(
            velocity_residual, basis.to(mean_velocity)
        )
        residual = torch.cat((frame_residual, coefficients), dim=-1)
        coordinate_scale = distribution["coordinate_scale"].to(residual)
        normalized_nll = lowrank_gaussian_nll(
            residual / coordinate_scale, distribution["diagonal_scale"],
            distribution["factor"], observed,
        )
        native_nll = normalized_nll + torch.where(observed, coordinate_scale.log(), 0).sum(-1)
    return {
        "normalized_coordinate_nll": normalized_nll,
        "native_coordinate_nll": native_nll,
        "observed_coordinate_count": observed.sum(-1),
        "joint_residual": residual.masked_fill(~observed, torch.nan),
        "omitted_velocity_energy_px2": omitted_energy.masked_fill(~velocity_observed, torch.nan),
        "mean_detached": detach_mean,
        "probabilities_calibrated": False,
    }


def sample_joint_uncertainty(distribution, basis, sample_count, *, generator=None):
    """Sample cell (including unresolved tail), representation, then local 35D.

    Outputs use leading (draw,B) dimensions. One representation is retained for
    all marks in a section: apply its exact affine AFTER the sampled SVF warp.
    Tail draws carry cell/representation -1 and NaN geometry, never a fabricated
    refined mode. Teacher-forced selection is training-only and cannot be drawn
    as an inference posterior. This returns SVF states before integration; no
    topology clipping/rejection or calibration is hidden in sampling.
    """
    if bool(distribution["teacher_forced_mask"].any()):
        raise ValueError("teacher-forced selected cells do not define an inference posterior")
    cell_probability = distribution["conditional_cell_log_probability"].exp()
    cells = cell_probability.shape[1]
    cell_mass = torch.cat((
        cell_probability * distribution["retained_probability"][:, None],
        distribution["omitted_probability"][:, None],
    ), dim=-1)
    selected_cell = torch.multinomial(cell_mass, sample_count, replacement=True, generator=generator).T
    unresolved = selected_cell == cells
    cell = selected_cell.clamp_max(cells - 1)
    row = torch.arange(cell.shape[1], device=cell.device)[None].expand_as(cell)
    representation_probability = distribution["representation_log_conditional_within_cell"][row, cell].exp()
    representation = torch.multinomial(
        representation_probability.reshape(-1, representation_probability.shape[-1]),
        1, generator=generator,
    ).reshape_as(cell)
    component = (row, cell, representation)
    drawn = sample_joint_frame_velocity(
        distribution["mean_frame_state"][component],
        distribution["diagonal_scale"][component], distribution["factor"][component],
        basis, 1, coordinate_scale=distribution["coordinate_scale"], generator=generator,
    )
    output = {key: value.squeeze(0) for key, value in drawn.items() if isinstance(value, torch.Tensor)}
    output["stationary_velocity_yx_px"] = (
        distribution["mean_stationary_velocity_yx_px"][component]
        + output["velocity_perturbation_yx_px"]
    )
    output["representation_to_canonical_raster_affine"] = distribution["representation_to_canonical_raster_affine"][component]
    for key, value in output.items():
        mask = unresolved.reshape(*unresolved.shape, *((1,) * (value.ndim - 2)))
        output[key] = value.masked_fill(mask, False if value.dtype == torch.bool else torch.nan)
    output.update({
        "selected_cell_index": cell.masked_fill(unresolved, -1),
        "cell_id": distribution["cell_id"][row, cell].masked_fill(unresolved, -1),
        "representation_index": representation.masked_fill(unresolved, -1),
        "unresolved_tail": unresolved,
        "velocity_supervision_eligible": distribution["velocity_supervision_eligible"][row] & ~unresolved,
        "probabilities_calibrated": False,
    })
    return output
