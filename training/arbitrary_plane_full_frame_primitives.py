"""Pure-Torch full-frame composition and finite-thickness atlas rendering."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training.arbitrary_plane_geometry import (
    frame_to_physical_ouv,
    frame_to_rotation_6d,
    inplane_basis_to_parameters,
    normalized_raster_to_ccf,
    physical_um_to_allen_index_points,
    rotation_6d_to_frame,
)


FULL_FRAME_STATE_SIZE = 12
FULL_FRAME_UPDATE_SIZE = 9
FINITE_PSF_FAMILIES = ("boxcar", "gaussian")


def _finite_floating(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")
    return value


def _positive_upper_triangular_basis(
    log_diagonal: torch.Tensor,
    shear: torch.Tensor,
) -> torch.Tensor:
    diagonal = torch.exp(log_diagonal)
    zero = torch.zeros_like(shear)
    basis = torch.stack(
        (
            torch.stack((diagonal[..., 0], shear * diagonal[..., 1]), dim=-1),
            torch.stack((zero, diagonal[..., 1]), dim=-1),
        ),
        dim=-2,
    )
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("in-plane basis became nonfinite")
    return basis


def so3_exp_map(local_rotation_rad: torch.Tensor) -> torch.Tensor:
    """Exponentiate local axis-angle vectors to proper rotation matrices."""
    rotation = _finite_floating(local_rotation_rad, "local SO(3) rotation")
    if rotation.shape[-1:] != (3,):
        raise ValueError("local SO(3) rotation must end in three coordinates")
    x, y, z = rotation.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    angle = torch.linalg.vector_norm(rotation, dim=-1)
    first = torch.sinc(angle / torch.pi)
    second = 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    return (
        identity.expand(rotation.shape[:-1] + (3, 3))
        + first[..., None, None] * skew
        + second[..., None, None] * (skew @ skew)
    )


def full_frame_state_from_components(
    center_ap_dv_ml_um: torch.Tensor,
    frame_ap_dv_ml: torch.Tensor,
    inplane_basis_um: torch.Tensor,
) -> torch.Tensor:
    """Pack centre, proper frame and positive upper-triangular basis into 12 values."""
    center = torch.as_tensor(center_ap_dv_ml_um)
    frame = torch.as_tensor(frame_ap_dv_ml)
    basis = torch.as_tensor(inplane_basis_um)
    frame_to_physical_ouv(center, frame, basis)
    rotation_6d = frame_to_rotation_6d(frame)
    log_diagonal, shear = inplane_basis_to_parameters(basis)
    return torch.cat((center, rotation_6d, log_diagonal, shear[..., None]), dim=-1)


def full_frame_state_to_components(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a physical state ``[..., centre3, rotation6d6, logdiag2, shear1]``."""
    state = _finite_floating(state, "full-frame state")
    if state.shape[-1:] != (FULL_FRAME_STATE_SIZE,):
        raise ValueError(f"full-frame state must end in {FULL_FRAME_STATE_SIZE} values")
    center = state[..., :3]
    frame = rotation_6d_to_frame(state[..., 3:9])
    basis = _positive_upper_triangular_basis(state[..., 9:11], state[..., 11])
    return center, frame, basis


def full_frame_state_to_physical_ouv(state: torch.Tensor) -> torch.Tensor:
    """Decode a state to flattened physical AP/DV/ML O/U/V vectors."""
    return frame_to_physical_ouv(*full_frame_state_to_components(state))


def compose_full_frame_state(state: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Right-compose one local update without mixing raster reflection into pose.

    Updates are ``[..., local_so3_rad3, pre-update-local translation_um3,
    delta_log_basis_diagonal2, delta_shear1]``. The exact rule is
    ``R'=R exp(w)``, ``c'=c+R t`` and ``A'=A DeltaA``.
    """
    state = _finite_floating(state, "full-frame state")
    update = _finite_floating(update, "full-frame update")
    if state.shape[-1:] != (FULL_FRAME_STATE_SIZE,):
        raise ValueError(f"full-frame state must end in {FULL_FRAME_STATE_SIZE} values")
    if update.shape[-1:] != (FULL_FRAME_UPDATE_SIZE,):
        raise ValueError(f"full-frame update must end in {FULL_FRAME_UPDATE_SIZE} values")
    if state.shape[:-1] != update.shape[:-1]:
        raise ValueError("full-frame state and update leading dimensions must match")
    if state.device != update.device or state.dtype != update.dtype:
        raise ValueError("full-frame state and update must share one device and dtype")

    center, frame, basis = full_frame_state_to_components(state)
    delta_frame = so3_exp_map(update[..., :3])
    local_translation = update[..., 3:6]
    delta_basis = _positive_upper_triangular_basis(
        update[..., 6:8], update[..., 8]
    )
    composed_center = center + (frame @ local_translation[..., None]).squeeze(-1)
    composed_frame = frame @ delta_frame
    composed_basis = basis @ delta_basis
    return full_frame_state_from_components(
        composed_center, composed_frame, composed_basis
    )


def finite_psf_axial_sample_count(
    thickness_um: float,
    axial_step_um_max: float,
) -> int:
    """Return the odd sample count used by the frozen symmetric slab schedule."""
    thickness = float(thickness_um)
    step_max = float(axial_step_um_max)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("physical thickness must be finite and positive")
    if not math.isfinite(step_max) or step_max <= 0.0:
        raise ValueError("axial step maximum must be finite and positive")
    return 2 * int(math.ceil(thickness / (2.0 * step_max))) + 1


def normalized_finite_psf_kernel(
    thickness_um: torch.Tensor,
    axial_sample_count: int,
    family: str,
    *,
    gaussian_sigma_um: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric physical offsets and positive unit-mass PSF weights.

    Gaussian support is exactly ``[-thickness/2, thickness/2]``. Its explicit
    sigma controls density inside that support; values below floating-point
    resolution relative to the support saturate at ``eps * thickness``.
    Trapezoidal endpoint masses are applied before normalization for both PSF
    families.
    """
    thickness = _finite_floating(thickness_um, "physical thickness")
    if bool((thickness <= 0.0).any()):
        raise ValueError("physical thickness must be positive")
    if (
        not isinstance(axial_sample_count, int)
        or isinstance(axial_sample_count, bool)
        or axial_sample_count < 3
        or axial_sample_count % 2 == 0
    ):
        raise ValueError("axial sample count must be an odd integer of at least three")
    if family not in FINITE_PSF_FAMILIES:
        raise ValueError(f"PSF family must be one of {FINITE_PSF_FAMILIES}")

    unit_offsets = torch.linspace(
        -0.5,
        0.5,
        axial_sample_count,
        device=thickness.device,
        dtype=thickness.dtype,
    )
    offsets = thickness[..., None] * unit_offsets
    quadrature_mass = torch.ones_like(unit_offsets)
    quadrature_mass[[0, -1]] = 0.5
    if family == "boxcar":
        unnormalized = torch.ones_like(offsets) * quadrature_mass
        if gaussian_sigma_um is not None:
            raise ValueError("boxcar PSF does not accept a Gaussian sigma")
    else:
        if gaussian_sigma_um is None:
            raise ValueError("Gaussian PSF requires an explicit sigma in micrometres")
        sigma = _finite_floating(gaussian_sigma_um, "Gaussian sigma")
        if sigma.shape != thickness.shape or sigma.device != thickness.device or sigma.dtype != thickness.dtype:
            raise ValueError("Gaussian sigma must match thickness shape, device and dtype")
        if bool((sigma <= 0.0).any()):
            raise ValueError("Gaussian sigma must be positive")
        safe_sigma = torch.maximum(
            sigma, thickness * torch.finfo(thickness.dtype).eps
        )
        log_unnormalized = (
            -0.5 * (offsets / safe_sigma[..., None]).square()
            + torch.log(quadrature_mass)
        )
        log_unnormalized = log_unnormalized - log_unnormalized.amax(
            dim=-1, keepdim=True
        )
        log_floor = math.log(torch.finfo(thickness.dtype).tiny)
        unnormalized = torch.exp(log_unnormalized.clamp_min(log_floor))
    if not bool(torch.isfinite(unnormalized).all()) or bool((unnormalized <= 0.0).any()):
        raise ValueError("PSF weights became nonpositive or nonfinite")
    mass = unnormalized.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(mass).all()) or bool((mass <= 0.0).any()):
        raise ValueError("finite PSF weights must have finite positive mass")
    weights = unnormalized / mass
    return offsets, weights


def render_finite_thickness_plane(
    volume_c_ap_dv_ml: torch.Tensor,
    state: torch.Tensor,
    output_shape_h_w: tuple[int, int],
    origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    axial_offsets_um: torch.Tensor,
    axial_weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiably render a unit-mass finite PSF along the physical normal."""
    volume = _finite_floating(volume_c_ap_dv_ml, "atlas feature volume")
    if volume.ndim != 4 or any(size < 2 for size in volume.shape[-3:]):
        raise ValueError("atlas feature volume must have shape (C,AP,DV,ML) with spatial sizes >=2")
    state = torch.as_tensor(state, device=volume.device, dtype=volume.dtype)
    state = _finite_floating(state, "full-frame state")
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2 or state.shape[-1] != FULL_FRAME_STATE_SIZE:
        raise ValueError("renderer state must have shape (12,) or (B,12)")
    center, frame, basis = full_frame_state_to_components(state)
    batch = state.shape[0]

    offsets = _finite_floating(
        torch.as_tensor(axial_offsets_um, device=volume.device, dtype=volume.dtype),
        "axial offsets",
    )
    weights = _finite_floating(
        torch.as_tensor(axial_weights, device=volume.device, dtype=volume.dtype),
        "axial weights",
    )
    if offsets.ndim == 1:
        offsets = offsets[None].expand(batch, -1)
    if weights.ndim == 1:
        weights = weights[None].expand(batch, -1)
    if offsets.ndim != 2 or weights.shape != offsets.shape or offsets.shape[0] != batch:
        raise ValueError("axial offsets and weights must have shape (S,) or aligned (B,S)")
    if offsets.shape[1] < 1 or bool((weights <= 0.0).any()):
        raise ValueError("axial sampling requires at least one strictly positive weight")
    weight_mass = weights.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(weight_mass).all()) or bool((weight_mass <= 0.0).any()):
        raise ValueError("weights must have finite positive mass")
    normalized_weights = weights / weight_mass

    if (
        len(output_shape_h_w) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in output_shape_h_w)
    ):
        raise ValueError("output height and width must be positive integers")
    height, width = output_shape_h_w
    s = torch.arange(width, device=volume.device, dtype=volume.dtype) / width
    t = torch.arange(height, device=volume.device, dtype=volume.dtype) / height
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    st = torch.stack((ss, tt), dim=-1)[None].expand(batch, -1, -1, -1)
    base_points_um = normalized_raster_to_ccf(
        center[:, None, None], frame[:, None, None], basis[:, None, None], st
    )
    normal = frame[..., :, 2]
    points_um = (
        base_points_um[:, None]
        + offsets[:, :, None, None, None] * normal[:, None, None, None]
    )
    points = physical_um_to_allen_index_points(
        points_um, origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um
    )
    depth, native_height, native_width = volume.shape[-3:]
    grid = torch.stack(
        (
            points[..., 2] / (native_width - 1) * 2.0 - 1.0,
            points[..., 1] / (native_height - 1) * 2.0 - 1.0,
            points[..., 0] / (depth - 1) * 2.0 - 1.0,
        ),
        dim=-1,
    ).reshape(batch * offsets.shape[1], 1, height, width, 3)
    sampled = F.grid_sample(
        volume[None].expand(batch * offsets.shape[1], -1, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, 0].reshape(batch, offsets.shape[1], volume.shape[0], height, width)
    return (sampled * normalized_weights[:, :, None, None, None]).sum(dim=1)
