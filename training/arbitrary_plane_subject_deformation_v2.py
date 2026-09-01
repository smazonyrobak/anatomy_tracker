"""Standalone full-CCF animal subject-deformation primitives for v2 synthetic data."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from training.arbitrary_plane_acquisition_v2 import (
    _array_receipt,
    _freeze_value,
    _json_value,
    _nonnegative_integer,
    _payload_sha256,
    _root_seed_hex,
)


SUBJECT_DEFORMATION_V2_SCHEMA = "anatomy-tracker.arbitrary-plane-subject-deformation/v2"
SUBJECT_DEFORMATION_V2_ALGORITHM = (
    "positive-diagonal-full-ccf-scale-after-coarse-fine-cubic-bspline-svf-rk4/v2"
)
SUBJECT_DEFORMATION_V2_COORDINATES = "right-handed CCF AP-DV-ML physical micrometres"
SUBJECT_DEFORMATION_V2_RNG_DOMAIN = "anatomy-tracker.subject-deformation-rng/v2"
SUBJECT_DEFORMATION_V2_SOURCE_CANONICALIZATION = "CRLF and CR normalized to LF before SHA-256"
SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
SUBJECT_DEFORMATION_V2_RK4_ORIENTATION_CERTIFICATE = (
    "for q=global gradient Frobenius bound / RK4 steps, require "
    "q + q^2/2 + q^3/6 + q^4/24 < 1; then every forward and inverse "
    "RK4 step-map Jacobian lies in the positive-determinant unit ball about I"
)
_SOURCE_ROOT = Path(__file__).parent


def _normalized_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        name: _normalized_text_sha256(_SOURCE_ROOT / name)
        for name in (
            "arbitrary_plane_subject_deformation_v2.py",
            "arbitrary_plane_acquisition_v2.py",
        )
    }


def _domain_id(domain: str, payload: object) -> str:
    return _payload_sha256({"id_domain": domain, "payload": payload})


def derive_subject_deformation_seed_v2(
    root_seed: int | str,
    split: str,
    animal_index: int,
    scope: str,
    stage: str,
    field: str,
    attempt: int = 0,
) -> int:
    """Derive a PCG64DXSM seed without using mutable animal provenance labels."""
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    attempt = _nonnegative_integer(attempt, "attempt")
    if (
        split not in {"train", "development"}
        or not isinstance(scope, str)
        or not scope
        or not isinstance(stage, str)
        or not stage
        or not isinstance(field, str)
        or not field
    ):
        raise ValueError("subject deformation RNG hierarchy is invalid")
    components = (
        SUBJECT_DEFORMATION_V2_RNG_DOMAIN,
        SUBJECT_DEFORMATION_V2_SCHEMA,
        split,
        _root_seed_hex(root_seed),
        str(animal_index),
        scope,
        stage,
        field,
        str(attempt),
    )
    encoded = b"".join(
        len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8")
        for value in components
    )
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-SUBJ-V2").digest(), "big"
    )


def cubic_bspline_basis(
    fractional_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cardinal cubic B-spline weights and knot-coordinate derivatives."""
    t = np.asarray(fractional_coordinate)
    weights = np.stack(
        (
            (1.0 - t) ** 3 / 6.0,
            (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
            (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
            t**3 / 6.0,
        ),
        -1,
    )
    derivatives = np.stack(
        (
            -0.5 * (1.0 - t) ** 2,
            1.5 * t**2 - 2.0 * t,
            -1.5 * t**2 + t + 0.5,
            0.5 * t**2,
        ),
        -1,
    )
    return weights, derivatives


def cubic_bspline_velocity(
    points_um: np.ndarray,
    coefficients_um: np.ndarray,
    lattice_origin_um: np.ndarray,
    lattice_spacing_um: float | np.ndarray,
    *,
    return_gradient: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate a zero-extended tensor-product cubic B-spline velocity field."""
    points = np.asarray(points_um)
    coefficients = np.asarray(coefficients_um)
    origin = np.asarray(lattice_origin_um)
    spacing = np.broadcast_to(np.asarray(lattice_spacing_um), (3,))
    if (
        points.shape[-1:] != (3,)
        or coefficients.ndim != 4
        or coefficients.shape[-1] != 3
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("points, coefficients, lattice origin, or spacing are invalid")
    dtype = np.result_type(points.dtype, coefficients.dtype, origin.dtype, spacing.dtype, np.float32)
    flat = np.asarray(points, dtype=dtype).reshape(-1, 3)
    origin = np.asarray(origin, dtype=dtype)
    spacing = np.asarray(spacing, dtype=dtype)
    coordinate = (flat - origin) / spacing
    knot = np.floor(coordinate).astype(np.int64)
    weights, derivatives = cubic_bspline_basis(coordinate - knot)
    base = knot - 1
    velocity = np.zeros((len(flat), 3), dtype=dtype)
    gradient = np.zeros((len(flat), 3, 3), dtype=dtype) if return_gradient else None
    shape = np.asarray(coefficients.shape[:3], dtype=np.int64)

    for i in range(4):
        index_i = base[:, 0] + i
        valid_i = (index_i >= 0) & (index_i < shape[0])
        clipped_i = np.clip(index_i, 0, shape[0] - 1)
        for j in range(4):
            index_j = base[:, 1] + j
            valid_j = (index_j >= 0) & (index_j < shape[1])
            clipped_j = np.clip(index_j, 0, shape[1] - 1)
            for k in range(4):
                index_k = base[:, 2] + k
                valid = valid_i & valid_j & (index_k >= 0) & (index_k < shape[2])
                clipped_k = np.clip(index_k, 0, shape[2] - 1)
                values = coefficients[clipped_i, clipped_j, clipped_k].astype(dtype, copy=False)
                valid_float = valid.astype(dtype)
                common = valid_float * weights[:, 0, i] * weights[:, 1, j] * weights[:, 2, k]
                velocity += common[:, None] * values
                if return_gradient:
                    gradient[:, :, 0] += (
                        valid_float
                        * derivatives[:, 0, i]
                        * weights[:, 1, j]
                        * weights[:, 2, k]
                        / spacing[0]
                    )[:, None] * values
                    gradient[:, :, 1] += (
                        valid_float
                        * weights[:, 0, i]
                        * derivatives[:, 1, j]
                        * weights[:, 2, k]
                        / spacing[1]
                    )[:, None] * values
                    gradient[:, :, 2] += (
                        valid_float
                        * weights[:, 0, i]
                        * weights[:, 1, j]
                        * derivatives[:, 2, k]
                        / spacing[2]
                    )[:, None] * values

    velocity = velocity.reshape(points.shape)
    if not return_gradient:
        return velocity
    return velocity, gradient.reshape(points.shape[:-1] + (3, 3))


def integrate_stationary_velocity(
    points_um: np.ndarray,
    velocity_field: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    *,
    direction: int = 1,
    steps: int = 8,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Integrate exp(+/-v) pointwise with fixed-step classical RK4."""
    points = np.asarray(points_um)
    if points.shape[-1:] != (3,) or direction not in {-1, 1} or int(steps) != steps or steps < 1:
        raise ValueError("points, direction, or fixed RK4 step count are invalid")
    flat = points.reshape(-1, 3)
    size = len(flat) if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be positive")
    mapped = np.empty_like(flat, dtype=np.result_type(points.dtype, np.float32))
    jacobians = np.empty((len(flat), 3, 3), dtype=mapped.dtype) if return_jacobian else None
    h = mapped.dtype.type(1.0 / int(steps))
    sign = mapped.dtype.type(direction)

    for start in range(0, len(flat), size):
        stop = min(start + size, len(flat))
        x = np.asarray(flat[start:stop], dtype=mapped.dtype).copy()
        if return_jacobian:
            jacobian = np.broadcast_to(np.eye(3, dtype=mapped.dtype), (len(x), 3, 3)).copy()
        for _ in range(int(steps)):
            if return_jacobian:
                v1, g1 = velocity_field(x, return_gradient=True)
                k1x = sign * np.asarray(v1)
                k1j = sign * np.matmul(np.asarray(g1), jacobian)
                x2 = x + 0.5 * h * k1x
                j2 = jacobian + 0.5 * h * k1j
                v2, g2 = velocity_field(x2, return_gradient=True)
                k2x = sign * np.asarray(v2)
                k2j = sign * np.matmul(np.asarray(g2), j2)
                x3 = x + 0.5 * h * k2x
                j3 = jacobian + 0.5 * h * k2j
                v3, g3 = velocity_field(x3, return_gradient=True)
                k3x = sign * np.asarray(v3)
                k3j = sign * np.matmul(np.asarray(g3), j3)
                x4 = x + h * k3x
                j4 = jacobian + h * k3j
                v4, g4 = velocity_field(x4, return_gradient=True)
                k4x = sign * np.asarray(v4)
                k4j = sign * np.matmul(np.asarray(g4), j4)
                x += h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                jacobian += h * (k1j + 2.0 * k2j + 2.0 * k3j + k4j) / 6.0
            else:
                k1 = sign * np.asarray(velocity_field(x, return_gradient=False))
                k2 = sign * np.asarray(velocity_field(x + 0.5 * h * k1, return_gradient=False))
                k3 = sign * np.asarray(velocity_field(x + 0.5 * h * k2, return_gradient=False))
                k4 = sign * np.asarray(velocity_field(x + h * k3, return_gradient=False))
                x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        mapped[start:stop] = x
        if return_jacobian:
            jacobians[start:stop] = jacobian

    mapped = mapped.reshape(points.shape)
    if not return_jacobian:
        return mapped
    return mapped, jacobians.reshape(points.shape[:-1] + (3, 3))


def bspline_stationary_velocity_flow(
    points_um: np.ndarray,
    coefficients_um: np.ndarray,
    lattice_origin_um: np.ndarray,
    lattice_spacing_um: float | np.ndarray,
    *,
    direction: int = 1,
    steps: int = 8,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate exp(+/-v) for one cubic B-spline stationary velocity field."""

    def field(query: np.ndarray, *, return_gradient: bool = False):
        return cubic_bspline_velocity(
            query,
            coefficients_um,
            lattice_origin_um,
            lattice_spacing_um,
            return_gradient=return_gradient,
        )

    return integrate_stationary_velocity(
        points_um,
        field,
        direction=direction,
        steps=steps,
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )


def apply_positive_diagonal_scale(
    points_um: np.ndarray,
    scale_ap_dv_ml: np.ndarray,
    center_um: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Apply a positive diagonal scale about the frozen full-CCF centre."""
    points = np.asarray(points_um)
    scale = np.asarray(scale_ap_dv_ml)
    center = np.asarray(center_um)
    if (
        points.shape[-1:] != (3,)
        or scale.shape != (3,)
        or center.shape != (3,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("global subject scale must be finite, positive, and diagonal")
    if inverse:
        return center + (points - center) / scale
    return center + (points - center) * scale


def compose_diagonal_scale_svf_forward(
    points_um: np.ndarray,
    velocity_field: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    scale_ap_dv_ml: np.ndarray,
    center_um: np.ndarray,
    *,
    steps: int = 8,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points_um = np.asarray(points_um, dtype="<f8", order="C")
    scale_ap_dv_ml = np.asarray(scale_ap_dv_ml, dtype="<f8", order="C")
    center_um = np.asarray(center_um, dtype="<f8", order="C")
    flowed = integrate_stationary_velocity(
        points_um,
        velocity_field,
        steps=steps,
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )
    local_points, local_jacobian = flowed if return_jacobian else (flowed, None)
    mapped = apply_positive_diagonal_scale(local_points, scale_ap_dv_ml, center_um)
    if not return_jacobian:
        return mapped
    return mapped, np.einsum("ij,...jk->...ik", np.diag(scale_ap_dv_ml), local_jacobian)


def compose_diagonal_scale_svf_inverse(
    points_um: np.ndarray,
    velocity_field: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    scale_ap_dv_ml: np.ndarray,
    center_um: np.ndarray,
    *,
    steps: int = 8,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points_um = np.asarray(points_um, dtype="<f8", order="C")
    scale_ap_dv_ml = np.asarray(scale_ap_dv_ml, dtype="<f8", order="C")
    center_um = np.asarray(center_um, dtype="<f8", order="C")
    local_points = apply_positive_diagonal_scale(
        points_um, scale_ap_dv_ml, center_um, inverse=True
    )
    flowed = integrate_stationary_velocity(
        local_points,
        velocity_field,
        direction=-1,
        steps=steps,
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )
    mapped, local_jacobian = flowed if return_jacobian else (flowed, None)
    if not return_jacobian:
        return mapped
    return mapped, np.einsum(
        "...ij,jk->...ik", local_jacobian, np.diag(1.0 / np.asarray(scale_ap_dv_ml))
    )


def _fixed_full_ccf_grid(
    lower_um: np.ndarray, upper_um: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    return np.stack(
        np.meshgrid(
            *[
                np.linspace(lower_um[axis], upper_um[axis], int(shape[axis]), dtype=np.float64)
                for axis in range(3)
            ],
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)


def _full_ccf_grid_geometry(
    lower_um: np.ndarray,
    upper_um: np.ndarray,
    maximum_spacing_um: np.ndarray,
) -> tuple[tuple[int, int, int], np.ndarray]:
    extent = np.asarray(upper_um, dtype="<f8") - np.asarray(lower_um, dtype="<f8")
    maximum_spacing = np.broadcast_to(
        np.asarray(maximum_spacing_um, dtype="<f8"), (3,)
    )
    if np.any(maximum_spacing <= 0.0):
        raise ValueError("fixed full-CCF grid spacing must be positive")
    segments = np.maximum(1, np.ceil(extent / maximum_spacing).astype(np.int64))
    shape = tuple((segments + 1).tolist())
    return shape, np.asarray(extent / segments, dtype="<f8", order="C")


def _fixed_gaussian_smooth(values: np.ndarray, sigma_knots: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * float(sigma_knots)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / float(sigma_knots)) ** 2)
    kernel /= kernel.sum()
    smoothed = np.asarray(values, dtype=np.float64)
    for axis in range(3):
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, radius, mode="reflect"), kernel, mode="valid"),
            axis,
            smoothed,
        )
    return smoothed


def _boundary_taper(shape: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for size in shape:
        distance = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float64)
        q = np.clip((distance - 1.0) / 3.0, 0.0, 1.0)
        axes.append(q**3 * (q * (q * 6.0 - 15.0) + 10.0))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def _lattice_geometry(
    lower_um: np.ndarray,
    upper_um: np.ndarray,
    spacing_um: float | np.ndarray,
    padding_um: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    spacing = np.broadcast_to(np.asarray(spacing_um, dtype=np.float64), (3,)).copy()
    if np.any(spacing <= 0.0) or float(padding_um) < 4.0 * float(spacing.max()):
        raise ValueError("each SVF requires positive spacing and at least four-knot padding")
    origin = lower_um - float(padding_um)
    shape = tuple(
        (
            np.ceil((upper_um - lower_um + 2.0 * float(padding_um)) / spacing).astype(int)
            + 1
        ).tolist()
    )
    return origin, spacing, shape


def _affine_mode_coefficients(
    shape: tuple[int, int, int],
    origin_um: np.ndarray,
    spacing_um: np.ndarray,
    center_um: np.ndarray,
    taper: np.ndarray,
) -> np.ndarray:
    coordinates = np.stack(
        np.meshgrid(
            *[
                origin_um[axis] + spacing_um[axis] * np.arange(shape[axis])
                for axis in range(3)
            ],
            indexing="ij",
        ),
        -1,
    )
    relative = coordinates - center_um
    modes = np.zeros((12,) + shape + (3,), dtype=np.float64)
    for output_axis in range(3):
        modes[output_axis, ..., output_axis] = 1.0
        for input_axis in range(3):
            modes[3 + 3 * output_axis + input_axis, ..., output_axis] = relative[..., input_axis]
    return modes * taper[None, ..., None]


def _remove_full_affine_fit(
    raw_coefficients: np.ndarray,
    projection_grid_um: np.ndarray,
    mode_coefficients: np.ndarray,
    origin_um: np.ndarray,
    spacing_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    observed = cubic_bspline_velocity(
        projection_grid_um, raw_coefficients, origin_um, spacing_um
    )
    design = np.stack(
        [
            cubic_bspline_velocity(
                projection_grid_um, mode, origin_um, spacing_um
            ).reshape(-1)
            for mode in mode_coefficients
        ],
        1,
    )
    fitted = np.linalg.lstsq(design, observed.reshape(-1), rcond=None)[0]
    projected = raw_coefficients - np.einsum(
        "m,m...c->...c", fitted, mode_coefficients
    )
    residual = cubic_bspline_velocity(
        projection_grid_um, projected, origin_um, spacing_um
    )
    residual_fit = np.linalg.lstsq(design, residual.reshape(-1), rcond=None)[0]
    return projected, fitted, float(np.max(np.abs(residual_fit)))


def _physical_affine_coefficients(
    points_um: np.ndarray,
    velocity_um: np.ndarray,
    center_um: np.ndarray,
    half_extent_um: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        (
            np.ones(len(points_um), dtype=np.float64),
            (np.asarray(points_um, dtype=np.float64) - center_um) / half_extent_um,
        )
    )
    return np.linalg.lstsq(
        design, np.asarray(velocity_um, dtype=np.float64), rcond=None
    )[0].T.reshape(-1)


def _remove_post_float32_complete_affine_fit(
    projection_grid_um: np.ndarray,
    center_um: np.ndarray,
    half_extent_um: np.ndarray,
    coarse_coefficients: np.ndarray,
    coarse_modes: np.ndarray,
    coarse_mode_affine_response: np.ndarray,
    coarse_origin_um: np.ndarray,
    coarse_spacing_um: np.ndarray,
    fine_coefficients: np.ndarray,
    fine_origin_um: np.ndarray,
    fine_spacing_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    corrected = np.asarray(coarse_coefficients, dtype=np.float64)
    correction = np.zeros(12, dtype=np.float64)
    residual = np.zeros(12, dtype=np.float64)
    for _ in range(3):
        combined = cubic_bspline_velocity(
            projection_grid_um, corrected, coarse_origin_um, coarse_spacing_um
        ) + cubic_bspline_velocity(
            projection_grid_um,
            fine_coefficients,
            fine_origin_um,
            fine_spacing_um,
        )
        residual = _physical_affine_coefficients(
            projection_grid_um, combined, center_um, half_extent_um
        )
        delta = np.linalg.solve(coarse_mode_affine_response, residual)
        correction += delta
        corrected = np.asarray(
            corrected - np.einsum("m,m...c->...c", delta, coarse_modes),
            dtype="<f4",
            order="C",
        )
    combined = cubic_bspline_velocity(
        projection_grid_um, corrected, coarse_origin_um, coarse_spacing_um
    ) + cubic_bspline_velocity(
        projection_grid_um,
        fine_coefficients,
        fine_origin_um,
        fine_spacing_um,
    )
    residual = _physical_affine_coefficients(
        projection_grid_um, combined, center_um, half_extent_um
    )
    return corrected, correction, float(np.max(np.abs(residual)))


def _combined_field(
    coarse_coefficients_um: np.ndarray,
    coarse_origin_um: np.ndarray,
    coarse_spacing_um: np.ndarray,
    fine_coefficients_um: np.ndarray,
    fine_origin_um: np.ndarray,
    fine_spacing_um: np.ndarray,
):
    def field(query: np.ndarray, *, return_gradient: bool = False):
        coarse = cubic_bspline_velocity(
            query,
            coarse_coefficients_um,
            coarse_origin_um,
            coarse_spacing_um,
            return_gradient=return_gradient,
        )
        fine = cubic_bspline_velocity(
            query,
            fine_coefficients_um,
            fine_origin_um,
            fine_spacing_um,
            return_gradient=return_gradient,
        )
        if not return_gradient:
            return coarse + fine
        return coarse[0] + fine[0], coarse[1] + fine[1]

    return field


def _combined_cubic_bspline_bounds(
    coarse_coefficients_um: np.ndarray,
    coarse_spacing_um: np.ndarray,
    fine_coefficients_um: np.ndarray,
    fine_spacing_um: np.ndarray,
) -> dict[str, object]:
    component_speed = np.zeros(3, dtype="<f8")
    component_derivative = np.zeros((3, 3), dtype="<f8")
    for coefficients, spacing in (
        (coarse_coefficients_um, coarse_spacing_um),
        (fine_coefficients_um, fine_spacing_um),
    ):
        coefficients = np.asarray(coefficients, dtype="<f8")
        spacing = np.broadcast_to(np.asarray(spacing, dtype="<f8"), (3,))
        component_speed += np.max(np.abs(coefficients), axis=(0, 1, 2))
        for input_axis in range(3):
            padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
            padding[input_axis] = (1, 1)
            adjacent_difference = np.diff(
                np.pad(coefficients, padding, mode="constant"), axis=input_axis
            ) / spacing[input_axis]
            component_derivative[:, input_axis] += np.max(
                np.abs(adjacent_difference), axis=(0, 1, 2)
            )
    return {
        "component_speed_abs_bound_um": np.asarray(
            component_speed, dtype="<f8", order="C"
        ),
        "speed_l2_bound_um": float(np.linalg.norm(component_speed)),
        "component_derivative_abs_bound": np.asarray(
            component_derivative, dtype="<f8", order="C"
        ),
        "component_derivative_abs_max": float(component_derivative.max()),
        "gradient_frobenius_bound": float(np.linalg.norm(component_derivative)),
        "divergence_abs_bound": float(np.trace(component_derivative)),
    }


def _rk4_step_map_orientation_certificate(
    gradient_frobenius_bound: float, steps: int
) -> dict[str, float | bool]:
    """Certify every classical-RK4 step map stays in the positive-det ball about I."""
    q = float(gradient_frobenius_bound) / int(steps)
    perturbation = q + q**2 / 2.0 + q**3 / 6.0 + q**4 / 24.0
    return {
        "rk4_step_gradient_product_bound": q,
        "rk4_step_map_jacobian_perturbation_bound": perturbation,
        "rk4_step_map_orientation_margin": 1.0 - perturbation,
        "rk4_step_map_orientation_certified": bool(perturbation < 1.0),
    }


def _halo_bounds(
    origin_um: np.ndarray, spacing_um: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    return origin_um + 2.0 * spacing_um, origin_um + (np.asarray(shape) - 3.0) * spacing_um


def _candidate_audit(
    audit_grid_um: np.ndarray,
    projection_grid_um: np.ndarray,
    half_extent_um: np.ndarray,
    coarse_unit_coefficients: np.ndarray,
    coarse_modes: np.ndarray,
    coarse_mode_affine_response: np.ndarray,
    coarse_origin_um: np.ndarray,
    coarse_spacing_um: np.ndarray,
    fine_unit_coefficients: np.ndarray,
    fine_origin_um: np.ndarray,
    fine_spacing_um: np.ndarray,
    amplitude_um: float,
    global_scale: np.ndarray,
    center_um: np.ndarray,
    steps: int,
    gate_limits: dict[str, float],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    audit_grid_um = np.asarray(audit_grid_um, dtype="<f8", order="C")
    coarse = np.asarray(coarse_unit_coefficients * float(amplitude_um), dtype="<f4", order="C")
    fine = np.asarray(fine_unit_coefficients * float(amplitude_um), dtype="<f4", order="C")
    coarse, candidate_affine_correction, candidate_affine_residual = (
        _remove_post_float32_complete_affine_fit(
            projection_grid_um,
            center_um,
            half_extent_um,
            coarse,
            coarse_modes,
            coarse_mode_affine_response,
            coarse_origin_um,
            coarse_spacing_um,
            fine,
            fine_origin_um,
            fine_spacing_um,
        )
    )
    field_bounds = _combined_cubic_bspline_bounds(
        coarse, coarse_spacing_um, fine, fine_spacing_um
    )
    orientation_certificate = _rk4_step_map_orientation_certificate(
        field_bounds["gradient_frobenius_bound"], steps
    )
    field = _combined_field(
        coarse,
        coarse_origin_um,
        coarse_spacing_um,
        fine,
        fine_origin_um,
        fine_spacing_um,
    )
    local_forward, local_forward_jacobian = integrate_stationary_velocity(
        audit_grid_um, field, steps=steps, return_jacobian=True
    )
    subject_forward = apply_positive_diagonal_scale(local_forward, global_scale, center_um)
    forward_inverse_input = apply_positive_diagonal_scale(
        subject_forward, global_scale, center_um, inverse=True
    )
    forward_cycle = integrate_stationary_velocity(
        forward_inverse_input, field, direction=-1, steps=steps
    )
    inverse_local_input = apply_positive_diagonal_scale(
        audit_grid_um, global_scale, center_um, inverse=True
    )
    inverse_ccf, local_inverse_jacobian = integrate_stationary_velocity(
        inverse_local_input, field, direction=-1, steps=steps, return_jacobian=True
    )
    inverse_cycle_local_forward = integrate_stationary_velocity(
        inverse_ccf, field, steps=steps
    )
    inverse_cycle = apply_positive_diagonal_scale(
        inverse_cycle_local_forward, global_scale, center_um
    )
    forward_jacobian = np.einsum(
        "ij,...jk->...ik", np.diag(global_scale), local_forward_jacobian
    )
    inverse_jacobian = np.einsum(
        "...ij,jk->...ik", local_inverse_jacobian, np.diag(1.0 / global_scale)
    )
    local_forward_det = np.linalg.det(local_forward_jacobian)
    local_inverse_det = np.linalg.det(local_inverse_jacobian)
    forward_det = np.linalg.det(forward_jacobian)
    inverse_det = np.linalg.det(inverse_jacobian)
    local_displacement = float(
        max(
            np.linalg.norm(local_forward - audit_grid_um, axis=1).max(),
            np.linalg.norm(inverse_ccf - inverse_local_input, axis=1).max(),
        )
    )
    coarse_lower, coarse_upper = _halo_bounds(
        coarse_origin_um, coarse_spacing_um, coarse.shape[:3]
    )
    fine_lower, fine_upper = _halo_bounds(fine_origin_um, fine_spacing_um, fine.shape[:3])
    halo_lower = np.maximum(coarse_lower, fine_lower)
    halo_upper = np.minimum(coarse_upper, fine_upper)
    integration_start_points = np.concatenate(
        (
            audit_grid_um,
            forward_inverse_input,
            inverse_local_input,
            inverse_ccf,
        ),
        axis=0,
    )
    start_halo_margin = float(
        min(
            np.min(integration_start_points - halo_lower),
            np.min(halo_upper - integration_start_points),
        )
    )
    halo_margin = start_halo_margin - field_bounds["speed_l2_bound_um"]
    forward_cycle_error = forward_cycle - audit_grid_um
    inverse_cycle_error = inverse_cycle - audit_grid_um
    audit_state = {
        "accepted_audit_forward_subject_um": np.asarray(
            subject_forward, dtype="<f8", order="C"
        ),
        "accepted_audit_inverse_ccf_um": np.asarray(
            inverse_ccf, dtype="<f8", order="C"
        ),
        "accepted_audit_forward_jacobian": np.asarray(
            forward_jacobian, dtype="<f8", order="C"
        ),
        "accepted_audit_inverse_jacobian": np.asarray(
            inverse_jacobian, dtype="<f8", order="C"
        ),
        "accepted_audit_forward_jacobian_det": np.asarray(
            forward_det, dtype="<f8", order="C"
        ),
        "accepted_audit_inverse_jacobian_det": np.asarray(
            inverse_det, dtype="<f8", order="C"
        ),
        "accepted_audit_forward_then_inverse_cycle_error_um": np.asarray(
            forward_cycle_error, dtype="<f8", order="C"
        ),
        "accepted_audit_inverse_then_forward_cycle_error_um": np.asarray(
            inverse_cycle_error, dtype="<f8", order="C"
        ),
        "accepted_candidate_affine_correction": np.asarray(
            candidate_affine_correction, dtype="<f8", order="C"
        ),
        "accepted_component_speed_abs_bound_um": np.asarray(
            field_bounds["component_speed_abs_bound_um"], dtype="<f8", order="C"
        ),
        "accepted_component_derivative_abs_bound": np.asarray(
            field_bounds["component_derivative_abs_bound"], dtype="<f8", order="C"
        ),
    }
    values = {
        "local_forward_jacobian_det_min": float(local_forward_det.min()),
        "local_forward_jacobian_det_max": float(local_forward_det.max()),
        "local_inverse_jacobian_det_min": float(local_inverse_det.min()),
        "local_inverse_jacobian_det_max": float(local_inverse_det.max()),
        "composed_jacobian_det_min": float(
            min(forward_det.min(), inverse_det.min())
        ),
        "forward_then_inverse_cycle_max_um": float(
            np.linalg.norm(forward_cycle_error, axis=1).max()
        ),
        "inverse_then_forward_cycle_max_um": float(
            np.linalg.norm(inverse_cycle_error, axis=1).max()
        ),
        "max_local_displacement_um": local_displacement,
        "component_derivative_abs_max": field_bounds[
            "component_derivative_abs_max"
        ],
        "gradient_frobenius_bound": field_bounds["gradient_frobenius_bound"],
        "divergence_abs_bound": field_bounds["divergence_abs_bound"],
        "speed_l2_bound_um": field_bounds["speed_l2_bound_um"],
        "rk4_step_gradient_product_bound": orientation_certificate[
            "rk4_step_gradient_product_bound"
        ],
        "rk4_step_map_jacobian_perturbation_bound": orientation_certificate[
            "rk4_step_map_jacobian_perturbation_bound"
        ],
        "rk4_step_map_orientation_margin": orientation_certificate[
            "rk4_step_map_orientation_margin"
        ],
        "physical_affine_residual_max_um": candidate_affine_residual,
        "minimum_interpolation_start_halo_um": start_halo_margin,
        "minimum_interpolation_halo_um": halo_margin,
    }
    finite = all(np.isfinite(value) for value in values.values())
    failed = []
    if not finite:
        failed.append("finite")
    if values["local_forward_jacobian_det_min"] < gate_limits["local_jacobian_det_min"]:
        failed.append("local_forward_jacobian_det_min")
    if values["local_forward_jacobian_det_max"] > gate_limits["local_jacobian_det_max"]:
        failed.append("local_forward_jacobian_det_max")
    if values["local_inverse_jacobian_det_min"] < gate_limits["local_jacobian_det_min"]:
        failed.append("local_inverse_jacobian_det_min")
    if values["local_inverse_jacobian_det_max"] > gate_limits["local_jacobian_det_max"]:
        failed.append("local_inverse_jacobian_det_max")
    if values["composed_jacobian_det_min"] < gate_limits["composed_jacobian_det_floor"]:
        failed.append("composed_jacobian_det_floor")
    if values["forward_then_inverse_cycle_max_um"] > gate_limits["cycle_max_um"]:
        failed.append("forward_then_inverse_cycle")
    if values["inverse_then_forward_cycle_max_um"] > gate_limits["cycle_max_um"]:
        failed.append("inverse_then_forward_cycle")
    if values["max_local_displacement_um"] > gate_limits["max_local_displacement_um"]:
        failed.append("max_local_displacement")
    if values["component_derivative_abs_max"] > gate_limits[
        "component_derivative_abs_max"
    ]:
        failed.append("component_derivative_abs_max")
    if values["gradient_frobenius_bound"] > gate_limits[
        "gradient_frobenius_bound"
    ]:
        failed.append("gradient_frobenius_bound")
    if values["divergence_abs_bound"] > gate_limits["divergence_abs_bound"]:
        failed.append("divergence_abs_bound")
    if values["speed_l2_bound_um"] > gate_limits["speed_l2_bound_um"]:
        failed.append("speed_l2_bound_um")
    if values["rk4_step_map_jacobian_perturbation_bound"] >= gate_limits[
        "rk4_step_map_jacobian_perturbation_bound"
    ]:
        failed.append("rk4_step_map_orientation_certificate")
    if values["physical_affine_residual_max_um"] > gate_limits[
        "physical_affine_residual_max_um"
    ]:
        failed.append("physical_affine_residual_max_um")
    if values["minimum_interpolation_halo_um"] < gate_limits["minimum_halo_um"]:
        failed.append("interpolation_halo")
    payload = {
        "amplitude_um": float(amplitude_um),
        "gate_values": values,
        "failed_gates": failed,
        "accepted": not failed,
        "coarse_coefficients_receipt": _array_receipt(coarse),
        "fine_coefficients_receipt": _array_receipt(fine),
        "candidate_affine_correction_receipt": _array_receipt(
            candidate_affine_correction
        ),
        "field_bounds": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in field_bounds.items()
        },
        "candidate_state_receipts": {
            name: _array_receipt(array) for name, array in audit_state.items()
        },
    }
    return {
        **payload,
        "candidate_id": _domain_id(
            "anatomy-tracker.subject-deformation-candidate/v2", payload
        ),
    }, coarse, fine, audit_state


def sample_animal_subject_deformation_plan_v2(
    full_ccf_lower_um: np.ndarray,
    full_ccf_upper_um: np.ndarray,
    *,
    root_seed: int | str,
    split: str,
    animal_index: int,
    animal_id: str | int,
    ccf_context_sha256: str,
    deformation_stratum: str = "standard",
    coarse_spacing_um: float | np.ndarray = 1000.0,
    fine_spacing_um: float | np.ndarray = 500.0,
    coarse_padding_um: float = 4000.0,
    fine_padding_um: float = 2000.0,
    smoothing_sigma_knots: float = 1.0,
    coarse_weight: float = 0.75,
    fine_weight: float = 0.25,
    a0_um: float = 125.0,
    global_log_scale_half_range: float = 0.03,
    integration_steps: int = 8,
    local_jacobian_det_min: float = 0.50,
    local_jacobian_det_max: float = 2.00,
    composed_jacobian_det_floor: float = 0.25,
    cycle_max_um: float = 2.5,
    max_local_displacement_um: float = 750.0,
    component_derivative_abs_max: float = 1.0,
    gradient_frobenius_bound_max: float = 2.0,
    divergence_abs_bound_max: float = 1.5,
    speed_l2_bound_um_max: float = 750.0,
    minimum_halo_um: float = 100.0,
    post_float32_affine_residual_max: float = 1.0e-6,
) -> Mapping[str, object]:
    """Create one deterministic full-CCF plan and its first passing realization."""
    lower = np.asarray(full_ccf_lower_um, dtype="<f8", order="C")
    upper = np.asarray(full_ccf_upper_um, dtype="<f8", order="C")
    root_seed = f"0x{_root_seed_hex(root_seed)}"
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    ccf_context_sha256 = str(ccf_context_sha256)
    numeric_parameters = np.asarray(
        [
            smoothing_sigma_knots,
            coarse_weight,
            fine_weight,
            a0_um,
            global_log_scale_half_range,
            local_jacobian_det_min,
            local_jacobian_det_max,
            composed_jacobian_det_floor,
            cycle_max_um,
            max_local_displacement_um,
            component_derivative_abs_max,
            gradient_frobenius_bound_max,
            divergence_abs_bound_max,
            speed_l2_bound_um_max,
            minimum_halo_um,
            post_float32_affine_residual_max,
        ],
        dtype=np.float64,
    )
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper <= lower)
        or split not in {"train", "development"}
        or animal_index < 0
        or str(animal_id) == ""
        or len(ccf_context_sha256) != 64
        or any(character not in "0123456789abcdef" for character in ccf_context_sha256)
        or deformation_stratum not in {"identity", "standard"}
        or not np.isfinite(numeric_parameters).all()
        or float(smoothing_sigma_knots) <= 0.0
        or min(float(coarse_weight), float(fine_weight)) <= 0.0
        or not np.isclose(float(coarse_weight) + float(fine_weight), 1.0)
        or float(a0_um) <= 0.0
        or float(global_log_scale_half_range) < 0.0
        or int(integration_steps) != integration_steps
        or integration_steps < 1
        or float(local_jacobian_det_min) <= 0.0
        or float(local_jacobian_det_max) < float(local_jacobian_det_min)
        or float(composed_jacobian_det_floor) <= 0.0
        or min(
            float(component_derivative_abs_max),
            float(gradient_frobenius_bound_max),
            float(divergence_abs_bound_max),
            float(speed_l2_bound_um_max),
        )
        <= 0.0
        or min(
            float(cycle_max_um),
            float(max_local_displacement_um),
            float(minimum_halo_um),
        )
        < 0.0
        or float(post_float32_affine_residual_max) <= 0.0
    ):
        raise ValueError("full-CCF subject deformation inputs are invalid")
    center = np.asarray((lower + upper) / 2.0, dtype="<f8", order="C")
    coarse_origin, coarse_spacing, coarse_shape = _lattice_geometry(
        lower, upper, coarse_spacing_um, coarse_padding_um
    )
    fine_origin, fine_spacing, fine_shape = _lattice_geometry(
        lower, upper, fine_spacing_um, fine_padding_um
    )
    grid_maximum_spacing = np.asarray(fine_spacing / 2.0, dtype="<f8", order="C")
    projection_grid_shape, projection_grid_spacing = _full_ccf_grid_geometry(
        lower, upper, grid_maximum_spacing
    )
    audit_grid_shape, audit_grid_spacing = _full_ccf_grid_geometry(
        lower, upper, grid_maximum_spacing
    )
    projection_grid = _fixed_full_ccf_grid(
        lower, upper, projection_grid_shape
    )
    audit_grid = _fixed_full_ccf_grid(lower, upper, audit_grid_shape)
    source_coordinates = (
        "subject-deformation",
        "animal-realization",
    )
    source_specs = {
        "global_scale_x": (*source_coordinates, "global-positive-diagonal-scale-x", 0),
        "global_scale_y": (*source_coordinates, "global-positive-diagonal-scale-y", 0),
        "global_scale_z": (*source_coordinates, "global-positive-diagonal-scale-z", 0),
        "coarse_svf": (*source_coordinates, "coarse-cubic-bspline-svf", 0),
        "fine_svf": (*source_coordinates, "fine-cubic-bspline-svf", 0),
    }
    rng_sources = {}
    for name, (scope, stage, field, attempt) in source_specs.items():
        seed = derive_subject_deformation_seed_v2(
            root_seed, split, animal_index, scope, stage, field, attempt
        )
        rng_sources[name] = {
            "scope": scope,
            "stage": stage,
            "field": field,
            "attempt": attempt,
            "seed_uint64": f"0x{seed:016x}",
            "generator": "NumPy PCG64DXSM",
        }

    coarse_modes = _affine_mode_coefficients(
        coarse_shape,
        coarse_origin,
        coarse_spacing,
        center,
        _boundary_taper(coarse_shape),
    )
    fine_modes = _affine_mode_coefficients(
        fine_shape,
        fine_origin,
        fine_spacing,
        center,
        _boundary_taper(fine_shape),
    )
    half_extent = np.asarray((upper - lower) / 2.0, dtype="<f8", order="C")

    if deformation_stratum == "identity":
        coarse_mode_affine_response = np.eye(12, dtype="<f8")
        raw_coarse = np.zeros(coarse_shape + (3,), dtype="<f4")
        raw_fine = np.zeros(fine_shape + (3,), dtype="<f4")
        projected_coarse = np.zeros_like(raw_coarse)
        projected_fine = np.zeros_like(raw_fine)
        coarse_affine_fit = np.zeros(12, dtype=np.float64)
        fine_affine_fit = np.zeros(12, dtype=np.float64)
        coarse_projection_residual = 0.0
        fine_projection_residual = 0.0
        post_float32_affine_correction = np.zeros(12, dtype="<f8")
        post_float32_affine_residual = 0.0
        global_log_scale = np.zeros(3, dtype="<f8")
        global_scale = np.ones(3, dtype="<f8")
        effective_a0_um = 0.0
        combined_unit_rms = 0.0
    else:
        coarse_mode_affine_response = np.stack(
            [
                _physical_affine_coefficients(
                    projection_grid,
                    cubic_bspline_velocity(
                        projection_grid, mode, coarse_origin, coarse_spacing
                    ),
                    center,
                    half_extent,
                )
                for mode in coarse_modes
            ],
            1,
        )
        coarse_rng = np.random.Generator(
            np.random.PCG64DXSM(int(rng_sources["coarse_svf"]["seed_uint64"], 16))
        )
        fine_rng = np.random.Generator(
            np.random.PCG64DXSM(int(rng_sources["fine_svf"]["seed_uint64"], 16))
        )
        raw_coarse = np.asarray(
            _fixed_gaussian_smooth(
                coarse_rng.standard_normal(coarse_shape + (3,), dtype=np.float64),
                smoothing_sigma_knots,
            )
            * _boundary_taper(coarse_shape)[..., None],
            dtype="<f4",
            order="C",
        )
        raw_fine = np.asarray(
            _fixed_gaussian_smooth(
                fine_rng.standard_normal(fine_shape + (3,), dtype=np.float64),
                smoothing_sigma_knots,
            )
            * _boundary_taper(fine_shape)[..., None],
            dtype="<f4",
            order="C",
        )
        projected_coarse, coarse_affine_fit, coarse_projection_residual = (
            _remove_full_affine_fit(
                raw_coarse,
                projection_grid,
                coarse_modes,
                coarse_origin,
                coarse_spacing,
            )
        )
        projected_fine, fine_affine_fit, fine_projection_residual = _remove_full_affine_fit(
            raw_fine,
            projection_grid,
            fine_modes,
            fine_origin,
            fine_spacing,
        )
        coarse_velocity = cubic_bspline_velocity(
            projection_grid, projected_coarse, coarse_origin, coarse_spacing
        )
        fine_velocity = cubic_bspline_velocity(
            projection_grid, projected_fine, fine_origin, fine_spacing
        )
        coarse_rms = float(np.sqrt(np.mean(np.sum(coarse_velocity**2, axis=1))))
        fine_rms = float(np.sqrt(np.mean(np.sum(fine_velocity**2, axis=1))))
        projected_coarse = float(coarse_weight) * projected_coarse / coarse_rms
        projected_fine = float(fine_weight) * projected_fine / fine_rms
        combined_velocity = cubic_bspline_velocity(
            projection_grid, projected_coarse, coarse_origin, coarse_spacing
        ) + cubic_bspline_velocity(
            projection_grid, projected_fine, fine_origin, fine_spacing
        )
        combined_unit_rms = float(
            np.sqrt(np.mean(np.sum(combined_velocity**2, axis=1)))
        )
        projected_coarse = np.asarray(
            projected_coarse / combined_unit_rms, dtype="<f4", order="C"
        )
        projected_fine = np.asarray(
            projected_fine / combined_unit_rms, dtype="<f4", order="C"
        )
        projected_coarse, post_float32_affine_correction, post_float32_affine_residual = (
            _remove_post_float32_complete_affine_fit(
                projection_grid,
                center,
                half_extent,
                projected_coarse,
                coarse_modes,
                coarse_mode_affine_response,
                coarse_origin,
                coarse_spacing,
                projected_fine,
                fine_origin,
                fine_spacing,
            )
        )
        if post_float32_affine_residual > float(post_float32_affine_residual_max):
            raise ValueError("post-float32 complete-affine projection gate failed")
        global_log_scale = np.asarray(
            [
                np.random.Generator(
                    np.random.PCG64DXSM(
                        int(rng_sources[f"global_scale_{axis}"]["seed_uint64"], 16)
                    )
                ).uniform(
                    -float(global_log_scale_half_range),
                    float(global_log_scale_half_range),
                )
                for axis in "xyz"
            ],
            dtype="<f8",
            order="C",
        )
        global_scale = np.asarray(np.exp(global_log_scale), dtype="<f8", order="C")
        effective_a0_um = float(a0_um)

    gate_limits = {
        "local_jacobian_det_min": float(local_jacobian_det_min),
        "local_jacobian_det_max": float(local_jacobian_det_max),
        "composed_jacobian_det_floor": float(composed_jacobian_det_floor),
        "cycle_max_um": float(cycle_max_um),
        "max_local_displacement_um": float(max_local_displacement_um),
        "component_derivative_abs_max": float(component_derivative_abs_max),
        "gradient_frobenius_bound": float(gradient_frobenius_bound_max),
        "divergence_abs_bound": float(divergence_abs_bound_max),
        "speed_l2_bound_um": float(speed_l2_bound_um_max),
        "rk4_step_map_jacobian_perturbation_bound": 1.0,
        "physical_affine_residual_max_um": float(
            post_float32_affine_residual_max
        ),
        "minimum_halo_um": float(minimum_halo_um),
    }
    candidate_schedule = [
        float(effective_a0_um * factor)
        for factor in SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS
    ]
    candidate_audits = []
    accepted_index = None
    accepted_coarse = None
    accepted_fine = None
    accepted_audit_state = None
    for index, amplitude in enumerate(candidate_schedule):
        audit, candidate_coarse, candidate_fine, candidate_audit_state = _candidate_audit(
            audit_grid,
            projection_grid,
            half_extent,
            projected_coarse,
            coarse_modes,
            coarse_mode_affine_response,
            coarse_origin,
            coarse_spacing,
            projected_fine,
            fine_origin,
            fine_spacing,
            amplitude,
            global_scale,
            center,
            int(integration_steps),
            gate_limits,
        )
        audit = {"candidate_index": index, "amplitude_factor": SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS[index], **audit}
        candidate_audits.append(audit)
        if audit["accepted"]:
            accepted_index = index
            accepted_coarse = candidate_coarse
            accepted_fine = candidate_fine
            accepted_audit_state = candidate_audit_state
            break
    if accepted_index is None or accepted_audit_state is None:
        raise ValueError("no deterministic subject deformation realization candidate passed")

    source_sha256 = _source_hashes()
    resolved_config = {
        "schema_version": SUBJECT_DEFORMATION_V2_SCHEMA,
        "algorithm": SUBJECT_DEFORMATION_V2_ALGORITHM,
        "coordinate_contract": SUBJECT_DEFORMATION_V2_COORDINATES,
        "deformation_stratum": deformation_stratum,
        "fixed_grid_rule": (
            "per-axis ceil(full-CCF extent / (fine lattice spacing / 2)) segments"
        ),
        "grid_maximum_spacing_um": grid_maximum_spacing.tolist(),
        "projection_grid_shape": list(projection_grid_shape),
        "projection_grid_spacing_um": projection_grid_spacing.tolist(),
        "audit_grid_shape": list(audit_grid_shape),
        "audit_grid_spacing_um": audit_grid_spacing.tolist(),
        "coarse_spacing_um": coarse_spacing.tolist(),
        "fine_spacing_um": fine_spacing.tolist(),
        "coarse_padding_um": float(coarse_padding_um),
        "fine_padding_um": float(fine_padding_um),
        "smoothing_sigma_knots": float(smoothing_sigma_knots),
        "boundary_taper": "outer two knots zero; quintic smootherstep to full at distance four",
        "affine_projection": "complete 12-DOF translation plus unconstrained 3x3 linear fit",
        "candidate_affine_projection": "repeated after candidate multiplication and float32 cast",
        "post_float32_affine_residual_max": float(
            post_float32_affine_residual_max
        ),
        "coarse_weight": float(coarse_weight),
        "fine_weight": float(fine_weight),
        "a0_um": float(a0_um),
        "effective_a0_um": effective_a0_um,
        "candidate_factors": list(SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS),
        "global_log_scale_half_range": float(global_log_scale_half_range),
        "global_operator": "positive diagonal scale about frozen full-CCF centre; no rigid pose",
        "flow": {"integrator": "fixed-step classical RK4", "steps": int(integration_steps)},
        "coefficient_bounds": (
            "tensor-product B-spline convex-hull speed bounds and adjacent-coefficient "
            "derivative bounds, summed across coarse and fine lattices"
        ),
        "path_halo_bound": (
            "minimum RK4 integration-start halo minus the global coefficient speed bound"
        ),
        "numerical_orientation_certificate": (
            SUBJECT_DEFORMATION_V2_RK4_ORIENTATION_CERTIFICATE
        ),
        "gate_limits": gate_limits,
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
    }
    provenance = {
        "animal_id": animal_id,
        "animal_index": animal_index,
        "split": split,
        "root_seed_uint64": f"0x{_root_seed_hex(root_seed)}",
        "ccf_context_sha256": ccf_context_sha256,
    }
    coordinate_domain = {
        "full_ccf_lower_um": lower.tolist(),
        "full_ccf_upper_um": upper.tolist(),
        "frozen_center_um": center.tolist(),
        "full_ccf_lower_receipt": _array_receipt(lower),
        "full_ccf_upper_receipt": _array_receipt(upper),
        "frozen_center_receipt": _array_receipt(center),
    }
    fixed_grids = {
        "projection": {
            "source": "Cartesian linspace over frozen full-CCF closed bounds only at no more than fine_spacing/2",
            "shape": list(projection_grid_shape),
            "spacing_um": projection_grid_spacing.tolist(),
            "maximum_spacing_um": grid_maximum_spacing.tolist(),
            "spacing_receipt": _array_receipt(projection_grid_spacing),
            "array_receipt": _array_receipt(projection_grid),
        },
        "audit": {
            "source": "Cartesian linspace over frozen full-CCF closed bounds only at no more than fine_spacing/2",
            "shape": list(audit_grid_shape),
            "spacing_um": audit_grid_spacing.tolist(),
            "maximum_spacing_um": grid_maximum_spacing.tolist(),
            "spacing_receipt": _array_receipt(audit_grid_spacing),
            "array_receipt": _array_receipt(audit_grid),
        },
    }
    raw_svf = {
        "coarse": {
            "origin_um": coarse_origin.tolist(),
            "spacing_um": coarse_spacing.tolist(),
            "shape": list(coarse_shape),
            "raw_coefficients_receipt": _array_receipt(raw_coarse),
            "rng_source": "coarse_svf",
        },
        "fine": {
            "origin_um": fine_origin.tolist(),
            "spacing_um": fine_spacing.tolist(),
            "shape": list(fine_shape),
            "raw_coefficients_receipt": _array_receipt(raw_fine),
            "rng_source": "fine_svf",
        },
    }
    plan_payload = {
        "schema_version": SUBJECT_DEFORMATION_V2_SCHEMA,
        "algorithm": SUBJECT_DEFORMATION_V2_ALGORITHM,
        "resolved_config": resolved_config,
        "provenance": provenance,
        "coordinate_domain": coordinate_domain,
        "rng_sources": rng_sources,
        "fixed_grids": fixed_grids,
        "raw_svf": raw_svf,
        "source_sha256": source_sha256,
        "source_sha256_canonicalization": SUBJECT_DEFORMATION_V2_SOURCE_CANONICALIZATION,
    }
    plan_id = _domain_id(
        "anatomy-tracker.subject-deformation-plan/v2", plan_payload
    )
    global_scale_metadata = {
        "operator": "diag(scale) about frozen full-CCF centre",
        "determinant": float(np.prod(global_scale)),
        "log_scale_receipt": _array_receipt(global_log_scale),
        "scale_receipt": _array_receipt(global_scale),
        "rng_sources": [
            "global_scale_x",
            "global_scale_y",
            "global_scale_z",
        ],
    }
    local_svf = {
        "coarse": {
            "origin_um": coarse_origin.tolist(),
            "spacing_um": coarse_spacing.tolist(),
            "shape": list(coarse_shape),
            "projected_unit_coefficients_receipt": _array_receipt(projected_coarse),
            "accepted_coefficients_receipt": _array_receipt(accepted_coarse),
            "removed_affine_coefficients_receipt": _array_receipt(coarse_affine_fit),
            "raw_source": "raw_svf.coarse",
        },
        "fine": {
            "origin_um": fine_origin.tolist(),
            "spacing_um": fine_spacing.tolist(),
            "shape": list(fine_shape),
            "projected_unit_coefficients_receipt": _array_receipt(projected_fine),
            "accepted_coefficients_receipt": _array_receipt(accepted_fine),
            "removed_affine_coefficients_receipt": _array_receipt(fine_affine_fit),
            "raw_source": "raw_svf.fine",
        },
        "projection": {
            "mode_count": 12,
            "source_grid": "fixed_grids.projection",
            "coarse_post_projection_max_abs_affine_coefficient": coarse_projection_residual,
            "fine_post_projection_max_abs_affine_coefficient": fine_projection_residual,
            "pre_unit_normalization_combined_rms": combined_unit_rms,
            "post_float32_complete_affine_correction_receipt": _array_receipt(
                post_float32_affine_correction
            ),
            "post_float32_complete_affine_residual_max": post_float32_affine_residual,
        },
    }
    realization = {
        "candidate_schedule_um": candidate_schedule,
        "candidate_audits": candidate_audits,
        "accepted_candidate_index": accepted_index,
        "accepted_amplitude_um": candidate_schedule[accepted_index],
        "accepted_candidate_id": candidate_audits[accepted_index]["candidate_id"],
        "accepted_candidate_affine_correction_receipt": candidate_audits[
            accepted_index
        ]["candidate_affine_correction_receipt"],
        "accepted_candidate_physical_affine_residual_max_um": candidate_audits[
            accepted_index
        ]["gate_values"]["physical_affine_residual_max_um"],
        "accepted_candidate_field_bounds": candidate_audits[accepted_index][
            "field_bounds"
        ],
        "gate_limits": gate_limits,
        "no_redraw": True,
    }
    accepted_audit = {
        "source_grid": "fixed_grids.audit",
        "evaluation": (
            "forward at fixed CCF audit points; inverse at inverse-scaled fixed "
            "subject audit points"
        ),
        "state_receipts": {
            name: _array_receipt(array)
            for name, array in accepted_audit_state.items()
        },
    }
    realization_payload = {
        "subject_deformation_plan_id": plan_id,
        "global_scale": global_scale_metadata,
        "local_svf": local_svf,
        "realization": realization,
        "accepted_audit": accepted_audit,
    }
    realization_id = _domain_id(
        "anatomy-tracker.subject-deformation-realization/v2", realization_payload
    )
    synthetic_animal_id = _domain_id(
        "anatomy-tracker.synthetic-animal/v2",
        {
            "subject_deformation_plan_id": plan_id,
            "subject_deformation_realization_id": realization_id,
            "animal_provenance": provenance,
        },
    )
    state = {
        "full_ccf_lower_um": lower,
        "full_ccf_upper_um": upper,
        "frozen_center_um": center,
        "projection_grid_um": projection_grid,
        "audit_grid_um": audit_grid,
        "grid_maximum_spacing_um": grid_maximum_spacing,
        "projection_grid_spacing_um": projection_grid_spacing,
        "audit_grid_spacing_um": audit_grid_spacing,
        "global_log_scale": global_log_scale,
        "global_scale": global_scale,
        "coarse_origin_um": coarse_origin,
        "coarse_spacing_um": coarse_spacing,
        "fine_origin_um": fine_origin,
        "fine_spacing_um": fine_spacing,
        "raw_coarse_coefficients": raw_coarse,
        "raw_fine_coefficients": raw_fine,
        "coarse_removed_affine_coefficients": coarse_affine_fit,
        "fine_removed_affine_coefficients": fine_affine_fit,
        "projected_coarse_unit_coefficients": projected_coarse,
        "projected_fine_unit_coefficients": projected_fine,
        "accepted_coarse_coefficients_um": accepted_coarse,
        "accepted_fine_coefficients_um": accepted_fine,
        "post_float32_complete_affine_correction": post_float32_affine_correction,
        **accepted_audit_state,
    }
    artifact = {
        **plan_payload,
        "subject_deformation_plan_id": plan_id,
        "subject_deformation_realization_id": realization_id,
        "synthetic_animal_id": synthetic_animal_id,
        "global_scale": global_scale_metadata,
        "local_svf": local_svf,
        "realization": realization,
        "accepted_audit": accepted_audit,
        "state": state,
    }
    artifact["receipt_sha256"] = _payload_sha256(_receipt_payload(artifact))
    return _freeze_value(artifact)


_TOP_LEVEL_KEYS = {
    "schema_version",
    "algorithm",
    "resolved_config",
    "provenance",
    "coordinate_domain",
    "rng_sources",
    "fixed_grids",
    "raw_svf",
    "source_sha256",
    "source_sha256_canonicalization",
    "subject_deformation_plan_id",
    "subject_deformation_realization_id",
    "synthetic_animal_id",
    "global_scale",
    "local_svf",
    "realization",
    "accepted_audit",
    "state",
    "receipt_sha256",
}
_STATE_KEYS = {
    "full_ccf_lower_um",
    "full_ccf_upper_um",
    "frozen_center_um",
    "projection_grid_um",
    "audit_grid_um",
    "grid_maximum_spacing_um",
    "projection_grid_spacing_um",
    "audit_grid_spacing_um",
    "global_log_scale",
    "global_scale",
    "coarse_origin_um",
    "coarse_spacing_um",
    "fine_origin_um",
    "fine_spacing_um",
    "raw_coarse_coefficients",
    "raw_fine_coefficients",
    "coarse_removed_affine_coefficients",
    "fine_removed_affine_coefficients",
    "projected_coarse_unit_coefficients",
    "projected_fine_unit_coefficients",
    "accepted_coarse_coefficients_um",
    "accepted_fine_coefficients_um",
    "post_float32_complete_affine_correction",
    "accepted_audit_forward_subject_um",
    "accepted_audit_inverse_ccf_um",
    "accepted_audit_forward_jacobian",
    "accepted_audit_inverse_jacobian",
    "accepted_audit_forward_jacobian_det",
    "accepted_audit_inverse_jacobian_det",
    "accepted_audit_forward_then_inverse_cycle_error_um",
    "accepted_audit_inverse_then_forward_cycle_error_um",
    "accepted_candidate_affine_correction",
    "accepted_component_speed_abs_bound_um",
    "accepted_component_derivative_abs_bound",
}
_ACCEPTED_AUDIT_STATE_KEYS = {
    "accepted_audit_forward_subject_um",
    "accepted_audit_inverse_ccf_um",
    "accepted_audit_forward_jacobian",
    "accepted_audit_inverse_jacobian",
    "accepted_audit_forward_jacobian_det",
    "accepted_audit_inverse_jacobian_det",
    "accepted_audit_forward_then_inverse_cycle_error_um",
    "accepted_audit_inverse_then_forward_cycle_error_um",
    "accepted_candidate_affine_correction",
    "accepted_component_speed_abs_bound_um",
    "accepted_component_derivative_abs_bound",
}


def _live_state_receipts(plan: Mapping[str, object]) -> dict[str, object]:
    return {
        name: _array_receipt(array)
        for name, array in plan["state"].items()
    }


def _plan_id_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return _json_value(
        {
            key: plan[key]
            for key in (
                "schema_version",
                "algorithm",
                "resolved_config",
                "provenance",
                "coordinate_domain",
                "rng_sources",
                "fixed_grids",
                "raw_svf",
                "source_sha256",
                "source_sha256_canonicalization",
            )
        }
    )


def _realization_id_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return _json_value(
        {
            "subject_deformation_plan_id": plan["subject_deformation_plan_id"],
            "global_scale": plan["global_scale"],
            "local_svf": plan["local_svf"],
            "realization": plan["realization"],
            "accepted_audit": plan["accepted_audit"],
        }
    )


def _receipt_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return {
        "subject_deformation_plan_id": plan["subject_deformation_plan_id"],
        "subject_deformation_realization_id": plan[
            "subject_deformation_realization_id"
        ],
        "synthetic_animal_id": plan["synthetic_animal_id"],
        "plan_payload": _plan_id_payload(plan),
        "realization_payload": _realization_id_payload(plan),
        "live_state_receipts": _live_state_receipts(plan),
    }


def subject_deformation_plan_receipt_v2(plan: Mapping[str, object]) -> dict[str, object]:
    payload = _receipt_payload(plan)
    return {**payload, "receipt_sha256": _payload_sha256(payload)}


def verify_subject_deformation_plan_v2(
    plan: Mapping[str, object],
    *,
    expected_ccf_context_sha256: str,
    expected_full_ccf_lower_um: np.ndarray,
    expected_full_ccf_upper_um: np.ndarray,
) -> None:
    """Verify exact structure, source binding, live receipts, and deterministic replay."""
    if set(plan) != _TOP_LEVEL_KEYS or set(plan.get("state", {})) != _STATE_KEYS:
        raise ValueError("subject deformation plan has unauthenticated structure")
    expected_lower = np.asarray(expected_full_ccf_lower_um, dtype="<f8", order="C")
    expected_upper = np.asarray(expected_full_ccf_upper_um, dtype="<f8", order="C")
    if (
        expected_lower.shape != (3,)
        or expected_upper.shape != (3,)
        or str(expected_ccf_context_sha256) != plan["provenance"]["ccf_context_sha256"]
        or not np.array_equal(expected_lower, plan["state"]["full_ccf_lower_um"])
        or not np.array_equal(expected_upper, plan["state"]["full_ccf_upper_um"])
        or _array_receipt(expected_lower)
        != _array_receipt(plan["state"]["full_ccf_lower_um"])
        or _array_receipt(expected_upper)
        != _array_receipt(plan["state"]["full_ccf_upper_um"])
    ):
        raise ValueError("authoritative CCF context or full-CCF bounds do not match")
    rng_keys = {
        "global_scale_x",
        "global_scale_y",
        "global_scale_z",
        "coarse_svf",
        "fine_svf",
    }
    scale_keys = {
        "operator",
        "determinant",
        "log_scale_receipt",
        "scale_receipt",
        "rng_sources",
    }
    raw_level_keys = {
        "origin_um",
        "spacing_um",
        "shape",
        "raw_coefficients_receipt",
        "rng_source",
    }
    realized_level_keys = {
        "origin_um",
        "spacing_um",
        "shape",
        "projected_unit_coefficients_receipt",
        "accepted_coefficients_receipt",
        "removed_affine_coefficients_receipt",
        "raw_source",
    }
    projection_keys = {
        "mode_count",
        "source_grid",
        "coarse_post_projection_max_abs_affine_coefficient",
        "fine_post_projection_max_abs_affine_coefficient",
        "pre_unit_normalization_combined_rms",
        "post_float32_complete_affine_correction_receipt",
        "post_float32_complete_affine_residual_max",
    }
    realization_keys = {
        "candidate_schedule_um",
        "candidate_audits",
        "accepted_candidate_index",
        "accepted_amplitude_um",
        "accepted_candidate_id",
        "accepted_candidate_affine_correction_receipt",
        "accepted_candidate_physical_affine_residual_max_um",
        "accepted_candidate_field_bounds",
        "gate_limits",
        "no_redraw",
    }
    candidate_keys = {
        "candidate_index",
        "amplitude_factor",
        "amplitude_um",
        "gate_values",
        "failed_gates",
        "accepted",
        "coarse_coefficients_receipt",
        "fine_coefficients_receipt",
        "candidate_affine_correction_receipt",
        "field_bounds",
        "candidate_state_receipts",
        "candidate_id",
    }
    field_bound_keys = {
        "component_speed_abs_bound_um",
        "speed_l2_bound_um",
        "component_derivative_abs_bound",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
    }
    gate_value_keys = {
        "local_forward_jacobian_det_min",
        "local_forward_jacobian_det_max",
        "local_inverse_jacobian_det_min",
        "local_inverse_jacobian_det_max",
        "composed_jacobian_det_min",
        "forward_then_inverse_cycle_max_um",
        "inverse_then_forward_cycle_max_um",
        "max_local_displacement_um",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "rk4_step_gradient_product_bound",
        "rk4_step_map_jacobian_perturbation_bound",
        "rk4_step_map_orientation_margin",
        "physical_affine_residual_max_um",
        "minimum_interpolation_start_halo_um",
        "minimum_interpolation_halo_um",
    }
    gate_limit_keys = {
        "local_jacobian_det_min",
        "local_jacobian_det_max",
        "composed_jacobian_det_floor",
        "cycle_max_um",
        "max_local_displacement_um",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "rk4_step_map_jacobian_perturbation_bound",
        "physical_affine_residual_max_um",
        "minimum_halo_um",
    }
    config_keys = {
        "schema_version",
        "algorithm",
        "coordinate_contract",
        "deformation_stratum",
        "fixed_grid_rule",
        "grid_maximum_spacing_um",
        "projection_grid_shape",
        "projection_grid_spacing_um",
        "audit_grid_shape",
        "audit_grid_spacing_um",
        "coarse_spacing_um",
        "fine_spacing_um",
        "coarse_padding_um",
        "fine_padding_um",
        "smoothing_sigma_knots",
        "boundary_taper",
        "affine_projection",
        "candidate_affine_projection",
        "post_float32_affine_residual_max",
        "coarse_weight",
        "fine_weight",
        "a0_um",
        "effective_a0_um",
        "candidate_factors",
        "global_log_scale_half_range",
        "global_operator",
        "flow",
        "coefficient_bounds",
        "path_halo_bound",
        "numerical_orientation_certificate",
        "gate_limits",
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
    }
    if (
        set(plan) != _TOP_LEVEL_KEYS
        or set(plan.get("state", {})) != _STATE_KEYS
        or set(plan["resolved_config"]) != config_keys
        or set(plan["resolved_config"]["flow"]) != {"integrator", "steps"}
        or set(plan["resolved_config"]["gate_limits"]) != gate_limit_keys
        or set(plan["provenance"])
        != {"animal_id", "animal_index", "split", "root_seed_uint64", "ccf_context_sha256"}
        or set(plan["coordinate_domain"])
        != {
            "full_ccf_lower_um",
            "full_ccf_upper_um",
            "frozen_center_um",
            "full_ccf_lower_receipt",
            "full_ccf_upper_receipt",
            "frozen_center_receipt",
        }
        or set(plan["source_sha256"])
        != {"arbitrary_plane_subject_deformation_v2.py", "arbitrary_plane_acquisition_v2.py"}
        or set(plan["rng_sources"]) != rng_keys
        or any(
            set(plan["rng_sources"][name])
            != {"scope", "stage", "field", "attempt", "seed_uint64", "generator"}
            for name in rng_keys
        )
        or set(plan["fixed_grids"]) != {"projection", "audit"}
        or any(
            set(plan["fixed_grids"][name])
            != {
                "source",
                "shape",
                "spacing_um",
                "maximum_spacing_um",
                "spacing_receipt",
                "array_receipt",
            }
            for name in ("projection", "audit")
        )
        or set(plan["raw_svf"]) != {"coarse", "fine"}
        or any(set(plan["raw_svf"][name]) != raw_level_keys for name in ("coarse", "fine"))
        or set(plan["local_svf"]) != {"coarse", "fine", "projection"}
        or any(
            set(plan["local_svf"][name]) != realized_level_keys
            for name in ("coarse", "fine")
        )
        or set(plan["local_svf"]["projection"]) != projection_keys
        or set(plan["global_scale"]) != scale_keys
        or tuple(plan["global_scale"]["rng_sources"])
        != ("global_scale_x", "global_scale_y", "global_scale_z")
        or set(plan["realization"]) != realization_keys
        or set(plan["realization"]["gate_limits"]) != gate_limit_keys
        or set(plan["accepted_audit"])
        != {"source_grid", "evaluation", "state_receipts"}
        or set(plan["accepted_audit"]["state_receipts"])
        != _ACCEPTED_AUDIT_STATE_KEYS
    ):
        raise ValueError("subject deformation plan has unauthenticated structure")

    state = plan["state"]
    live = _live_state_receipts(plan)
    coarse = plan["local_svf"]["coarse"]
    fine = plan["local_svf"]["fine"]
    raw_coarse = plan["raw_svf"]["coarse"]
    raw_fine = plan["raw_svf"]["fine"]
    realization = plan["realization"]
    audits = realization["candidate_audits"]
    accepted_index = realization["accepted_candidate_index"]
    expected_schedule = [
        float(plan["resolved_config"]["effective_a0_um"] * factor)
        for factor in SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS
    ]
    if (
        accepted_index is None
        or not isinstance(accepted_index, int)
        or accepted_index < 0
        or accepted_index >= len(audits)
        or len(audits) != accepted_index + 1
        or list(realization["candidate_schedule_um"]) != expected_schedule
        or any(set(audit) != candidate_keys for audit in audits)
        or any(set(audit["gate_values"]) != gate_value_keys for audit in audits)
        or any(set(audit["field_bounds"]) != field_bound_keys for audit in audits)
        or any(
            set(audit["candidate_state_receipts"]) != _ACCEPTED_AUDIT_STATE_KEYS
            for audit in audits
        )
        or any(audit["candidate_index"] != index for index, audit in enumerate(audits))
        or any(
            audit["amplitude_factor"] != SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS[index]
            or audit["amplitude_um"] != expected_schedule[index]
            for index, audit in enumerate(audits)
        )
        or any(audit["accepted"] for audit in audits[:accepted_index])
        or not audits[accepted_index]["accepted"]
    ):
        raise ValueError("subject deformation candidate audit structure does not match")
    for audit in audits:
        candidate_payload = {
            key: audit[key]
            for key in (
                "amplitude_um",
                "gate_values",
                "failed_gates",
                "accepted",
                "coarse_coefficients_receipt",
                "fine_coefficients_receipt",
                "candidate_affine_correction_receipt",
                "field_bounds",
                "candidate_state_receipts",
            )
        }
        if audit["candidate_id"] != _domain_id(
            "anatomy-tracker.subject-deformation-candidate/v2",
            _json_value(candidate_payload),
        ):
            raise ValueError("subject deformation candidate ID does not match")
        certificate = _rk4_step_map_orientation_certificate(
            audit["field_bounds"]["gradient_frobenius_bound"],
            plan["resolved_config"]["flow"]["steps"],
        )
        certificate_failed = "rk4_step_map_orientation_certificate" in tuple(
            audit["failed_gates"]
        )
        if (
            audit["gate_values"]["rk4_step_gradient_product_bound"]
            != certificate["rk4_step_gradient_product_bound"]
            or audit["gate_values"][
                "rk4_step_map_jacobian_perturbation_bound"
            ]
            != certificate["rk4_step_map_jacobian_perturbation_bound"]
            or audit["gate_values"]["rk4_step_map_orientation_margin"]
            != certificate["rk4_step_map_orientation_margin"]
            or certificate_failed
            != (not certificate["rk4_step_map_orientation_certified"])
        ):
            raise ValueError("RK4 step-map orientation certificate does not match")

    accepted_audit = audits[accepted_index]
    recomputed_bounds = _combined_cubic_bspline_bounds(
        state["accepted_coarse_coefficients_um"],
        state["coarse_spacing_um"],
        state["accepted_fine_coefficients_um"],
        state["fine_spacing_um"],
    )
    accepted_velocity = cubic_bspline_velocity(
        state["projection_grid_um"],
        state["accepted_coarse_coefficients_um"],
        state["coarse_origin_um"],
        state["coarse_spacing_um"],
    ) + cubic_bspline_velocity(
        state["projection_grid_um"],
        state["accepted_fine_coefficients_um"],
        state["fine_origin_um"],
        state["fine_spacing_um"],
    )
    recomputed_affine_residual = float(
        np.max(
            np.abs(
                _physical_affine_coefficients(
                    state["projection_grid_um"],
                    accepted_velocity,
                    state["frozen_center_um"],
                    (
                        state["full_ccf_upper_um"]
                        - state["full_ccf_lower_um"]
                    )
                    / 2.0,
                )
            )
        )
    )
    if (
        not np.array_equal(
            accepted_audit["field_bounds"]["component_speed_abs_bound_um"],
            recomputed_bounds["component_speed_abs_bound_um"],
        )
        or not np.array_equal(
            accepted_audit["field_bounds"]["component_derivative_abs_bound"],
            recomputed_bounds["component_derivative_abs_bound"],
        )
        or any(
            accepted_audit["field_bounds"][name] != recomputed_bounds[name]
            or accepted_audit["gate_values"][name] != recomputed_bounds[name]
            for name in (
                "speed_l2_bound_um",
                "component_derivative_abs_max",
                "gradient_frobenius_bound",
                "divergence_abs_bound",
            )
        )
        or not np.isclose(
            accepted_audit["gate_values"]["physical_affine_residual_max_um"],
            recomputed_affine_residual,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError("accepted candidate coefficient bounds or affine audit changed")

    expected_plan_id = _domain_id(
        "anatomy-tracker.subject-deformation-plan/v2", _plan_id_payload(plan)
    )
    expected_realization_id = _domain_id(
        "anatomy-tracker.subject-deformation-realization/v2",
        _realization_id_payload(plan),
    )
    expected_animal_id = _domain_id(
        "anatomy-tracker.synthetic-animal/v2",
        {
            "subject_deformation_plan_id": expected_plan_id,
            "subject_deformation_realization_id": expected_realization_id,
            "animal_provenance": _json_value(plan["provenance"]),
        },
    )
    accepted_audit_receipts = {
        name: live[name] for name in _ACCEPTED_AUDIT_STATE_KEYS
    }
    if (
        plan["schema_version"] != SUBJECT_DEFORMATION_V2_SCHEMA
        or plan["algorithm"] != SUBJECT_DEFORMATION_V2_ALGORITHM
        or _json_value(plan["source_sha256"]) != _source_hashes()
        or plan["source_sha256_canonicalization"]
        != SUBJECT_DEFORMATION_V2_SOURCE_CANONICALIZATION
        or plan["subject_deformation_plan_id"] != expected_plan_id
        or plan["subject_deformation_realization_id"] != expected_realization_id
        or plan["synthetic_animal_id"] != expected_animal_id
        or plan["receipt_sha256"] != _payload_sha256(_receipt_payload(plan))
        or _json_value(plan["coordinate_domain"]["full_ccf_lower_receipt"])
        != live["full_ccf_lower_um"]
        or _json_value(plan["coordinate_domain"]["full_ccf_upper_receipt"])
        != live["full_ccf_upper_um"]
        or _json_value(plan["coordinate_domain"]["frozen_center_receipt"])
        != live["frozen_center_um"]
        or not np.array_equal(
            plan["coordinate_domain"]["full_ccf_lower_um"], state["full_ccf_lower_um"]
        )
        or not np.array_equal(
            plan["coordinate_domain"]["full_ccf_upper_um"], state["full_ccf_upper_um"]
        )
        or not np.array_equal(
            plan["coordinate_domain"]["frozen_center_um"], state["frozen_center_um"]
        )
        or _json_value(plan["fixed_grids"]["projection"]["array_receipt"])
        != live["projection_grid_um"]
        or _json_value(plan["fixed_grids"]["audit"]["array_receipt"])
        != live["audit_grid_um"]
        or _json_value(plan["fixed_grids"]["projection"]["spacing_receipt"])
        != live["projection_grid_spacing_um"]
        or _json_value(plan["fixed_grids"]["audit"]["spacing_receipt"])
        != live["audit_grid_spacing_um"]
        or _json_value(raw_coarse["raw_coefficients_receipt"])
        != live["raw_coarse_coefficients"]
        or _json_value(raw_fine["raw_coefficients_receipt"])
        != live["raw_fine_coefficients"]
        or not np.array_equal(raw_coarse["origin_um"], state["coarse_origin_um"])
        or not np.array_equal(raw_coarse["spacing_um"], state["coarse_spacing_um"])
        or not np.array_equal(raw_fine["origin_um"], state["fine_origin_um"])
        or not np.array_equal(raw_fine["spacing_um"], state["fine_spacing_um"])
        or not np.array_equal(coarse["origin_um"], state["coarse_origin_um"])
        or not np.array_equal(coarse["spacing_um"], state["coarse_spacing_um"])
        or not np.array_equal(fine["origin_um"], state["fine_origin_um"])
        or not np.array_equal(fine["spacing_um"], state["fine_spacing_um"])
        or _json_value(plan["global_scale"]["log_scale_receipt"])
        != live["global_log_scale"]
        or _json_value(plan["global_scale"]["scale_receipt"])
        != live["global_scale"]
        or _json_value(coarse["projected_unit_coefficients_receipt"])
        != live["projected_coarse_unit_coefficients"]
        or _json_value(fine["projected_unit_coefficients_receipt"])
        != live["projected_fine_unit_coefficients"]
        or _json_value(coarse["accepted_coefficients_receipt"])
        != live["accepted_coarse_coefficients_um"]
        or _json_value(fine["accepted_coefficients_receipt"])
        != live["accepted_fine_coefficients_um"]
        or _json_value(coarse["removed_affine_coefficients_receipt"])
        != live["coarse_removed_affine_coefficients"]
        or _json_value(fine["removed_affine_coefficients_receipt"])
        != live["fine_removed_affine_coefficients"]
        or _json_value(
            plan["local_svf"]["projection"][
                "post_float32_complete_affine_correction_receipt"
            ]
        )
        != live["post_float32_complete_affine_correction"]
        or _json_value(plan["accepted_audit"]["state_receipts"])
        != accepted_audit_receipts
        or _json_value(audits[accepted_index]["candidate_state_receipts"])
        != accepted_audit_receipts
        or _json_value(audits[accepted_index]["coarse_coefficients_receipt"])
        != live["accepted_coarse_coefficients_um"]
        or _json_value(audits[accepted_index]["fine_coefficients_receipt"])
        != live["accepted_fine_coefficients_um"]
        or _json_value(
            audits[accepted_index]["candidate_affine_correction_receipt"]
        )
        != live["accepted_candidate_affine_correction"]
        or _json_value(
            realization["accepted_candidate_affine_correction_receipt"]
        )
        != live["accepted_candidate_affine_correction"]
        or realization["accepted_candidate_physical_affine_residual_max_um"]
        != audits[accepted_index]["gate_values"][
            "physical_affine_residual_max_um"
        ]
        or _json_value(realization["accepted_candidate_field_bounds"])
        != _json_value(audits[accepted_index]["field_bounds"])
        or not np.array_equal(
            audits[accepted_index]["field_bounds"][
                "component_speed_abs_bound_um"
            ],
            state["accepted_component_speed_abs_bound_um"],
        )
        or not np.array_equal(
            audits[accepted_index]["field_bounds"][
                "component_derivative_abs_bound"
            ],
            state["accepted_component_derivative_abs_bound"],
        )
        or audits[accepted_index]["candidate_id"]
        != realization["accepted_candidate_id"]
        or realization["accepted_amplitude_um"] != expected_schedule[accepted_index]
        or not np.isclose(
            accepted_audit["gate_values"]["minimum_interpolation_halo_um"],
            accepted_audit["gate_values"][
                "minimum_interpolation_start_halo_um"
            ]
            - accepted_audit["field_bounds"]["speed_l2_bound_um"],
            rtol=0.0,
            atol=1.0e-12,
        )
        or plan["accepted_audit"]["source_grid"] != "fixed_grids.audit"
        or plan["resolved_config"]["flow"]["integrator"]
        != "fixed-step classical RK4"
        or plan["resolved_config"]["numerical_orientation_certificate"]
        != SUBJECT_DEFORMATION_V2_RK4_ORIENTATION_CERTIFICATE
        or realization["gate_limits"] != plan["resolved_config"]["gate_limits"]
        or realization["gate_limits"][
            "rk4_step_map_jacobian_perturbation_bound"
        ]
        != 1.0
        or tuple(plan["resolved_config"]["candidate_factors"])
        != SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS
        or any(
            plan["resolved_config"][name]
            for name in (
                "learned_checkpoint_dependencies",
                "previous_model_dependencies",
                "pretrained_feature_dependencies",
            )
        )
        or plan["local_svf"]["projection"][
            "post_float32_complete_affine_residual_max"
        ]
        > plan["resolved_config"]["post_float32_affine_residual_max"]
        or not np.all(np.asarray(state["global_scale"]) > 0.0)
        or not np.isclose(
            plan["global_scale"]["determinant"],
            np.prod(state["global_scale"]),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("subject deformation plan receipt does not match")

    expected_grid_shape, expected_grid_spacing = _full_ccf_grid_geometry(
        np.asarray(state["full_ccf_lower_um"]),
        np.asarray(state["full_ccf_upper_um"]),
        np.asarray(state["fine_spacing_um"]) / 2.0,
    )
    expected_projection = _fixed_full_ccf_grid(
        np.asarray(state["full_ccf_lower_um"]),
        np.asarray(state["full_ccf_upper_um"]),
        expected_grid_shape,
    )
    expected_audit = _fixed_full_ccf_grid(
        np.asarray(state["full_ccf_lower_um"]),
        np.asarray(state["full_ccf_upper_um"]),
        expected_grid_shape,
    )
    if (
        not np.array_equal(expected_projection, state["projection_grid_um"])
        or not np.array_equal(expected_audit, state["audit_grid_um"])
        or tuple(plan["resolved_config"]["projection_grid_shape"])
        != expected_grid_shape
        or tuple(plan["resolved_config"]["audit_grid_shape"])
        != expected_grid_shape
        or tuple(plan["fixed_grids"]["projection"]["shape"])
        != expected_grid_shape
        or tuple(plan["fixed_grids"]["audit"]["shape"])
        != expected_grid_shape
        or not np.array_equal(
            expected_grid_spacing, state["projection_grid_spacing_um"]
        )
        or not np.array_equal(expected_grid_spacing, state["audit_grid_spacing_um"])
        or not np.array_equal(
            np.asarray(state["fine_spacing_um"]) / 2.0,
            state["grid_maximum_spacing_um"],
        )
        or not np.array_equal(
            plan["resolved_config"]["projection_grid_spacing_um"],
            expected_grid_spacing,
        )
        or not np.array_equal(
            plan["resolved_config"]["audit_grid_spacing_um"],
            expected_grid_spacing,
        )
        or not np.array_equal(
            plan["resolved_config"]["grid_maximum_spacing_um"],
            state["grid_maximum_spacing_um"],
        )
        or not np.array_equal(
            plan["fixed_grids"]["projection"]["spacing_um"],
            expected_grid_spacing,
        )
        or not np.array_equal(
            plan["fixed_grids"]["audit"]["spacing_um"], expected_grid_spacing
        )
        or not np.array_equal(
            plan["fixed_grids"]["projection"]["maximum_spacing_um"],
            state["grid_maximum_spacing_um"],
        )
        or not np.array_equal(
            plan["fixed_grids"]["audit"]["maximum_spacing_um"],
            state["grid_maximum_spacing_um"],
        )
    ):
        raise ValueError("fixed full-CCF projection or audit grid changed")

    replayed = replay_animal_subject_deformation_plan_v2(plan)
    if subject_deformation_plan_receipt_v2(replayed) != subject_deformation_plan_receipt_v2(
        plan
    ):
        raise ValueError("deterministic subject deformation replay does not match")


def replay_animal_subject_deformation_plan_v2(
    plan: Mapping[str, object],
) -> Mapping[str, object]:
    config = plan["resolved_config"]
    provenance = plan["provenance"]
    domain = plan["coordinate_domain"]
    gates = config["gate_limits"]
    return sample_animal_subject_deformation_plan_v2(
        np.asarray(domain["full_ccf_lower_um"]),
        np.asarray(domain["full_ccf_upper_um"]),
        root_seed=provenance["root_seed_uint64"],
        split=provenance["split"],
        animal_index=provenance["animal_index"],
        animal_id=provenance["animal_id"],
        ccf_context_sha256=provenance["ccf_context_sha256"],
        deformation_stratum=config["deformation_stratum"],
        coarse_spacing_um=np.asarray(config["coarse_spacing_um"]),
        fine_spacing_um=np.asarray(config["fine_spacing_um"]),
        coarse_padding_um=config["coarse_padding_um"],
        fine_padding_um=config["fine_padding_um"],
        smoothing_sigma_knots=config["smoothing_sigma_knots"],
        coarse_weight=config["coarse_weight"],
        fine_weight=config["fine_weight"],
        a0_um=config["a0_um"],
        global_log_scale_half_range=config["global_log_scale_half_range"],
        integration_steps=config["flow"]["steps"],
        local_jacobian_det_min=gates["local_jacobian_det_min"],
        local_jacobian_det_max=gates["local_jacobian_det_max"],
        composed_jacobian_det_floor=gates["composed_jacobian_det_floor"],
        cycle_max_um=gates["cycle_max_um"],
        max_local_displacement_um=gates["max_local_displacement_um"],
        component_derivative_abs_max=gates["component_derivative_abs_max"],
        gradient_frobenius_bound_max=gates["gradient_frobenius_bound"],
        divergence_abs_bound_max=gates["divergence_abs_bound"],
        speed_l2_bound_um_max=gates["speed_l2_bound_um"],
        minimum_halo_um=gates["minimum_halo_um"],
        post_float32_affine_residual_max=gates[
            "physical_affine_residual_max_um"
        ],
    )


def _accepted_field(plan: Mapping[str, object]):
    realization = plan.get("realization", {})
    index = realization.get("accepted_candidate_index")
    audits = realization.get("candidate_audits", ())
    if (
        index is None
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= len(audits)
        or not audits[index]["accepted"]
        or not plan.get("subject_deformation_realization_id")
    ):
        raise ValueError("an accepted subject deformation realization is required")
    state = plan["state"]
    return _combined_field(
        state["accepted_coarse_coefficients_um"],
        state["coarse_origin_um"],
        state["coarse_spacing_um"],
        state["accepted_fine_coefficients_um"],
        state["fine_origin_um"],
        state["fine_spacing_um"],
    )


def ccf_to_subject_points_v2(
    points_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Map arbitrary CCF points using only an accepted animal realization."""
    field = _accepted_field(plan)
    points = np.asarray(points_um)
    if plan["resolved_config"]["deformation_stratum"] == "identity":
        mapped = np.array(points, copy=True, order="K")
        if not return_jacobian:
            return mapped
        jacobian = np.broadcast_to(
            np.eye(3, dtype=np.result_type(points.dtype, np.float32)),
            points.shape[:-1] + (3, 3),
        ).copy()
        return mapped, jacobian
    points = np.asarray(points, dtype="<f8", order="C")
    state = plan["state"]
    return compose_diagonal_scale_svf_forward(
        points,
        field,
        state["global_scale"],
        state["frozen_center_um"],
        steps=plan["resolved_config"]["flow"]["steps"],
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )


def subject_to_ccf_points_v2(
    points_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Map arbitrary subject plane/slab points using only an accepted realization."""
    field = _accepted_field(plan)
    points = np.asarray(points_um)
    if plan["resolved_config"]["deformation_stratum"] == "identity":
        mapped = np.array(points, copy=True, order="K")
        if not return_jacobian:
            return mapped
        jacobian = np.broadcast_to(
            np.eye(3, dtype=np.result_type(points.dtype, np.float32)),
            points.shape[:-1] + (3, 3),
        ).copy()
        return mapped, jacobian
    points = np.asarray(points, dtype="<f8", order="C")
    state = plan["state"]
    return compose_diagonal_scale_svf_inverse(
        points,
        field,
        state["global_scale"],
        state["frozen_center_um"],
        steps=plan["resolved_config"]["flow"]["steps"],
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )
