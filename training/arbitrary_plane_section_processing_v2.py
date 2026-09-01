"""Standalone section-specific 2-D processing deformation and pullback rendering."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from training.arbitrary_plane_acquisition_v2 import (
    V2_SOURCE_SHA256_CANONICALIZATION,
    _array_receipt,
    _freeze_value,
    _json_value,
    _nonnegative_integer,
    _normalized_text_sha256,
    _payload_sha256,
    _root_seed_hex,
)


SECTION_PROCESSING_V2_SCHEMA = "anatomy-tracker.section-processing-deformation/v2"
SECTION_PROCESSING_V2_ALGORITHM = (
    "physical-yx-affine-free-coarse-fine-cubic-bspline-svf-rk4/v2"
)
SECTION_PROCESSING_RENDER_V2_SCHEMA = "anatomy-tracker.section-processing-render/v2"
SECTION_PROCESSING_RENDER_V2_ALGORITHM = "inverse-svf-pullback-explicit-mixed-interpolation/v2"
SECTION_PROCESSING_V2_RNG_DOMAIN = "anatomy-tracker.section-processing-rng/v2"
SECTION_PROCESSING_V2_CANDIDATE_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
SECTION_PROCESSING_V2_ORIENTATION_CERTIFICATE = (
    "for q=global gradient Frobenius bound / RK4 steps, require "
    "q + q^2/2 + q^3/6 + q^4/24 < 1"
)
SECTION_PROCESSING_V2_PIXEL_METRIC_RELATIVE_TOLERANCE = 1.0e-4
_SOURCE_ROOT = Path(__file__).parent


def _source_hashes() -> dict[str, str]:
    return {
        name: _normalized_text_sha256(_SOURCE_ROOT / name)
        for name in (
            "arbitrary_plane_section_processing_v2.py",
            "arbitrary_plane_acquisition_v2.py",
        )
    }


def _render_source_hashes() -> dict[str, str]:
    return {
        **_source_hashes(),
        "arbitrary_plane_subject_slab_v2.py": _normalized_text_sha256(
            _SOURCE_ROOT / "arbitrary_plane_subject_slab_v2.py"
        ),
    }


def _domain_id(domain: str, payload: object) -> str:
    return _payload_sha256({"id_domain": domain, "payload": payload})


def derive_section_processing_seed_v2(
    root_seed: int | str,
    split: str,
    animal_index: int,
    section_index: int,
    stage: str,
    field: str,
    attempt: int = 0,
) -> int:
    """Derive one section RNG seed; provenance labels are deliberately absent."""
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    section_index = _nonnegative_integer(section_index, "section_index")
    attempt = _nonnegative_integer(attempt, "attempt")
    if (
        not isinstance(split, str)
        or not split
        or not isinstance(stage, str)
        or not stage
        or not isinstance(field, str)
        or not field
    ):
        raise ValueError("section-processing RNG hierarchy is invalid")
    components = (
        SECTION_PROCESSING_V2_RNG_DOMAIN,
        SECTION_PROCESSING_V2_SCHEMA,
        split,
        _root_seed_hex(root_seed),
        str(animal_index),
        str(section_index),
        stage,
        field,
        str(attempt),
    )
    encoded = b"".join(
        len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8")
        for value in components
    )
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-SECT-V2").digest(), "big"
    )


def cubic_bspline_basis_2d_v2(
    fractional_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def cubic_bspline_velocity_2d_v2(
    points_yx_um: np.ndarray,
    coefficients_yx_um: np.ndarray,
    lattice_origin_yx_um: np.ndarray,
    lattice_spacing_yx_um: np.ndarray,
    *,
    return_gradient: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_yx_um)
    coefficients = np.asarray(coefficients_yx_um)
    origin = np.asarray(lattice_origin_yx_um)
    spacing = np.broadcast_to(np.asarray(lattice_spacing_yx_um), (2,))
    if (
        points.shape[-1:] != (2,)
        or coefficients.ndim != 3
        or coefficients.shape[-1] != 2
        or origin.shape != (2,)
        or np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("2-D points, coefficients, origin, or spacing are invalid")
    dtype = np.result_type(points.dtype, coefficients.dtype, origin.dtype, spacing.dtype, np.float32)
    flat = np.asarray(points, dtype=dtype).reshape(-1, 2)
    origin = np.asarray(origin, dtype=dtype)
    spacing = np.asarray(spacing, dtype=dtype)
    coordinate = (flat - origin) / spacing
    knot = np.floor(coordinate).astype(np.int64)
    weights, derivatives = cubic_bspline_basis_2d_v2(coordinate - knot)
    base = knot - 1
    velocity = np.zeros((len(flat), 2), dtype=dtype)
    gradient = np.zeros((len(flat), 2, 2), dtype=dtype) if return_gradient else None
    shape = np.asarray(coefficients.shape[:2], dtype=np.int64)
    for i in range(4):
        iy = base[:, 0] + i
        valid_y = (iy >= 0) & (iy < shape[0])
        cy = np.clip(iy, 0, shape[0] - 1)
        for j in range(4):
            ix = base[:, 1] + j
            valid = valid_y & (ix >= 0) & (ix < shape[1])
            cx = np.clip(ix, 0, shape[1] - 1)
            values = coefficients[cy, cx].astype(dtype, copy=False)
            valid_float = valid.astype(dtype)
            common = valid_float * weights[:, 0, i] * weights[:, 1, j]
            velocity += common[:, None] * values
            if return_gradient:
                gradient[:, :, 0] += (
                    valid_float * derivatives[:, 0, i] * weights[:, 1, j] / spacing[0]
                )[:, None] * values
                gradient[:, :, 1] += (
                    valid_float * weights[:, 0, i] * derivatives[:, 1, j] / spacing[1]
                )[:, None] * values
    velocity = velocity.reshape(points.shape)
    if not return_gradient:
        return velocity
    return velocity, gradient.reshape(points.shape[:-1] + (2, 2))


def integrate_stationary_velocity_2d_v2(
    points_yx_um: np.ndarray,
    velocity_field: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    *,
    direction: int = 1,
    steps: int = 8,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_yx_um)
    if points.shape[-1:] != (2,) or direction not in {-1, 1} or int(steps) != steps or steps < 1:
        raise ValueError("2-D points, direction, or RK4 steps are invalid")
    flat = points.reshape(-1, 2)
    size = len(flat) if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be positive")
    mapped = np.empty_like(flat, dtype=np.result_type(points.dtype, np.float32))
    jacobians = np.empty((len(flat), 2, 2), dtype=mapped.dtype) if return_jacobian else None
    h = mapped.dtype.type(1.0 / int(steps))
    sign = mapped.dtype.type(direction)
    for start in range(0, len(flat), size):
        stop = min(start + size, len(flat))
        x = np.asarray(flat[start:stop], dtype=mapped.dtype).copy()
        if return_jacobian:
            jacobian = np.broadcast_to(np.eye(2, dtype=mapped.dtype), (len(x), 2, 2)).copy()
        for _ in range(int(steps)):
            if return_jacobian:
                v1, g1 = velocity_field(x, return_gradient=True)
                k1x = sign * np.asarray(v1)
                k1j = sign * np.matmul(np.asarray(g1), jacobian)
                x2, j2 = x + 0.5 * h * k1x, jacobian + 0.5 * h * k1j
                v2, g2 = velocity_field(x2, return_gradient=True)
                k2x, k2j = sign * np.asarray(v2), sign * np.matmul(np.asarray(g2), j2)
                x3, j3 = x + 0.5 * h * k2x, jacobian + 0.5 * h * k2j
                v3, g3 = velocity_field(x3, return_gradient=True)
                k3x, k3j = sign * np.asarray(v3), sign * np.matmul(np.asarray(g3), j3)
                x4, j4 = x + h * k3x, jacobian + h * k3j
                v4, g4 = velocity_field(x4, return_gradient=True)
                k4x, k4j = sign * np.asarray(v4), sign * np.matmul(np.asarray(g4), j4)
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
    return mapped, jacobians.reshape(points.shape[:-1] + (2, 2))


def _fixed_grid(lower: np.ndarray, upper: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.stack(
        np.meshgrid(
            *[np.linspace(lower[a], upper[a], shape[a], dtype=np.float64) for a in range(2)],
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 2)


def _grid_geometry(
    lower: np.ndarray, upper: np.ndarray, maximum_spacing: np.ndarray
) -> tuple[tuple[int, int], np.ndarray]:
    extent = upper - lower
    segments = np.maximum(1, np.ceil(extent / maximum_spacing).astype(np.int64))
    return tuple((segments + 1).tolist()), np.asarray(extent / segments, dtype="<f8")


def _lattice_geometry(
    lower: np.ndarray,
    upper: np.ndarray,
    spacing_yx_um: np.ndarray,
    padding_um: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    spacing = np.broadcast_to(np.asarray(spacing_yx_um, dtype=np.float64), (2,)).copy()
    if (
        np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.isfinite(float(padding_um))
        or float(padding_um) < 4.0 * float(spacing.max())
    ):
        raise ValueError("2-D SVF lattices require positive spacing and four-knot padding")
    origin = lower - float(padding_um)
    shape = tuple(
        (
            np.ceil((upper - lower + 2.0 * float(padding_um)) / spacing).astype(np.int64)
            + 1
        ).tolist()
    )
    return origin, spacing, shape


def _smooth(values: np.ndarray, sigma_knots: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * float(sigma_knots)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / float(sigma_knots)) ** 2)
    kernel /= kernel.sum()
    smoothed = np.asarray(values, dtype=np.float64)
    for axis in range(2):
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, radius, mode="reflect"), kernel, mode="valid"),
            axis,
            smoothed,
        )
    return smoothed


def _taper(shape: tuple[int, int]) -> np.ndarray:
    axes = []
    for size in shape:
        distance = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float64)
        q = np.clip((distance - 1.0) / 3.0, 0.0, 1.0)
        axes.append(q**3 * (q * (q * 6.0 - 15.0) + 10.0))
    return axes[0][:, None] * axes[1][None, :]


def _affine_modes(
    shape: tuple[int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    center: np.ndarray,
    taper: np.ndarray,
) -> np.ndarray:
    coordinates = np.stack(
        np.meshgrid(
            *[origin[a] + spacing[a] * np.arange(shape[a]) for a in range(2)],
            indexing="ij",
        ),
        -1,
    )
    relative = coordinates - center
    modes = np.zeros((6,) + shape + (2,), dtype=np.float64)
    for output_axis in range(2):
        modes[output_axis, ..., output_axis] = 1.0
        for input_axis in range(2):
            modes[2 + 2 * output_axis + input_axis, ..., output_axis] = relative[..., input_axis]
    return modes * taper[None, ..., None]


def _physical_affine(points: np.ndarray, velocity: np.ndarray, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(points)), (points - center) / half_extent))
    return np.linalg.lstsq(design, velocity, rcond=None)[0].T.reshape(-1)


def _remove_affine(
    coefficients: np.ndarray,
    projection_grid: np.ndarray,
    modes: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    observed = cubic_bspline_velocity_2d_v2(projection_grid, coefficients, origin, spacing)
    design = np.stack(
        [cubic_bspline_velocity_2d_v2(projection_grid, mode, origin, spacing).reshape(-1) for mode in modes],
        1,
    )
    fitted = np.linalg.lstsq(design, observed.reshape(-1), rcond=None)[0]
    projected = coefficients - np.einsum("m,m...c->...c", fitted, modes)
    residual = cubic_bspline_velocity_2d_v2(projection_grid, projected, origin, spacing)
    residual_fit = np.linalg.lstsq(design, residual.reshape(-1), rcond=None)[0]
    return projected, fitted, float(np.max(np.abs(residual_fit)))


def _post_float32_affine_correction(
    projection_grid: np.ndarray,
    center: np.ndarray,
    half_extent: np.ndarray,
    coarse: np.ndarray,
    coarse_modes: np.ndarray,
    coarse_response: np.ndarray,
    coarse_origin: np.ndarray,
    coarse_spacing: np.ndarray,
    fine: np.ndarray,
    fine_origin: np.ndarray,
    fine_spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    corrected = np.asarray(coarse, dtype=np.float64)
    correction = np.zeros(6, dtype=np.float64)
    for _ in range(3):
        velocity = cubic_bspline_velocity_2d_v2(
            projection_grid, corrected, coarse_origin, coarse_spacing
        ) + cubic_bspline_velocity_2d_v2(projection_grid, fine, fine_origin, fine_spacing)
        residual = _physical_affine(projection_grid, velocity, center, half_extent)
        delta = np.linalg.solve(coarse_response, residual)
        correction += delta
        corrected = np.asarray(
            corrected - np.einsum("m,m...c->...c", delta, coarse_modes),
            dtype="<f4",
            order="C",
        )
    velocity = cubic_bspline_velocity_2d_v2(
        projection_grid, corrected, coarse_origin, coarse_spacing
    ) + cubic_bspline_velocity_2d_v2(projection_grid, fine, fine_origin, fine_spacing)
    residual = _physical_affine(projection_grid, velocity, center, half_extent)
    return corrected, correction, float(np.max(np.abs(residual)))


def _combined_field(state: Mapping[str, object], coarse: np.ndarray, fine: np.ndarray):
    def field(query: np.ndarray, *, return_gradient: bool = False):
        first = cubic_bspline_velocity_2d_v2(
            query,
            coarse,
            state["coarse_origin_yx_um"],
            state["coarse_spacing_yx_um"],
            return_gradient=return_gradient,
        )
        second = cubic_bspline_velocity_2d_v2(
            query,
            fine,
            state["fine_origin_yx_um"],
            state["fine_spacing_yx_um"],
            return_gradient=return_gradient,
        )
        if not return_gradient:
            return first + second
        return first[0] + second[0], first[1] + second[1]

    return field


def _coefficient_bounds(
    coarse: np.ndarray,
    coarse_spacing: np.ndarray,
    fine: np.ndarray,
    fine_spacing: np.ndarray,
) -> dict[str, object]:
    component_speed = np.zeros(2, dtype="<f8")
    component_derivative = np.zeros((2, 2), dtype="<f8")
    for coefficients, spacing in ((coarse, coarse_spacing), (fine, fine_spacing)):
        coefficients = np.asarray(coefficients, dtype="<f8")
        spacing = np.asarray(spacing, dtype="<f8")
        component_speed += np.max(np.abs(coefficients), axis=(0, 1))
        for input_axis in range(2):
            padding = [(0, 0), (0, 0), (0, 0)]
            padding[input_axis] = (1, 1)
            difference = np.diff(np.pad(coefficients, padding), axis=input_axis) / spacing[input_axis]
            component_derivative[:, input_axis] += np.max(np.abs(difference), axis=(0, 1))
    return {
        "component_speed_abs_bound_yx_um": component_speed,
        "speed_l2_bound_um": float(np.linalg.norm(component_speed)),
        "component_derivative_abs_bound": component_derivative,
        "component_derivative_abs_max": float(component_derivative.max()),
        "gradient_frobenius_bound": float(np.linalg.norm(component_derivative)),
        "divergence_abs_bound": float(np.trace(component_derivative)),
    }


def rk4_step_orientation_certificate_2d_v2(
    gradient_frobenius_bound: float, steps: int
) -> dict[str, float | bool]:
    q = float(gradient_frobenius_bound) / int(steps)
    perturbation = q + q**2 / 2.0 + q**3 / 6.0 + q**4 / 24.0
    return {
        "rk4_step_gradient_product_bound": q,
        "rk4_step_jacobian_perturbation_bound": perturbation,
        "rk4_step_orientation_margin": 1.0 - perturbation,
        "rk4_step_orientation_certified": bool(perturbation < 1.0),
    }


def _halo_bounds(origin: np.ndarray, spacing: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return origin + 2.0 * spacing, origin + (np.asarray(shape) - 3.0) * spacing


def _candidate_audit(
    state: dict[str, object],
    coarse_unit: np.ndarray,
    fine_unit: np.ndarray,
    coarse_modes: np.ndarray,
    coarse_response: np.ndarray,
    amplitude_um: float,
    steps: int,
    limits: dict[str, float],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    projection = state["projection_grid_yx_um"]
    audit_grid = state["audit_grid_yx_um"]
    coarse = np.asarray(coarse_unit * amplitude_um, dtype="<f4", order="C")
    fine = np.asarray(fine_unit * amplitude_um, dtype="<f4", order="C")
    coarse, correction, affine_residual = _post_float32_affine_correction(
        projection,
        state["domain_center_yx_um"],
        state["domain_half_extent_yx_um"],
        coarse,
        coarse_modes,
        coarse_response,
        state["coarse_origin_yx_um"],
        state["coarse_spacing_yx_um"],
        fine,
        state["fine_origin_yx_um"],
        state["fine_spacing_yx_um"],
    )
    bounds = _coefficient_bounds(
        coarse, state["coarse_spacing_yx_um"], fine, state["fine_spacing_yx_um"]
    )
    certificate = rk4_step_orientation_certificate_2d_v2(
        bounds["gradient_frobenius_bound"], steps
    )
    field = _combined_field(state, coarse, fine)
    forward, forward_jacobian = integrate_stationary_velocity_2d_v2(
        audit_grid, field, steps=steps, return_jacobian=True
    )
    inverse, inverse_jacobian = integrate_stationary_velocity_2d_v2(
        audit_grid, field, direction=-1, steps=steps, return_jacobian=True
    )
    forward_cycle = integrate_stationary_velocity_2d_v2(
        forward, field, direction=-1, steps=steps
    )
    inverse_cycle = integrate_stationary_velocity_2d_v2(inverse, field, steps=steps)
    forward_det = np.linalg.det(forward_jacobian)
    inverse_det = np.linalg.det(inverse_jacobian)
    coarse_lower, coarse_upper = _halo_bounds(
        state["coarse_origin_yx_um"], state["coarse_spacing_yx_um"], coarse.shape[:2]
    )
    fine_lower, fine_upper = _halo_bounds(
        state["fine_origin_yx_um"], state["fine_spacing_yx_um"], fine.shape[:2]
    )
    halo_lower, halo_upper = np.maximum(coarse_lower, fine_lower), np.minimum(coarse_upper, fine_upper)
    starts = np.concatenate((audit_grid, forward, inverse), axis=0)
    start_halo = float(min(np.min(starts - halo_lower), np.min(halo_upper - starts)))
    path_halo = start_halo - bounds["speed_l2_bound_um"]
    forward_error = forward_cycle - audit_grid
    inverse_error = inverse_cycle - audit_grid
    values = {
        "forward_jacobian_det_min": float(forward_det.min()),
        "forward_jacobian_det_max": float(forward_det.max()),
        "inverse_jacobian_det_min": float(inverse_det.min()),
        "inverse_jacobian_det_max": float(inverse_det.max()),
        "forward_inverse_cycle_max_um": float(np.linalg.norm(forward_error, axis=1).max()),
        "inverse_forward_cycle_max_um": float(np.linalg.norm(inverse_error, axis=1).max()),
        "maximum_displacement_um": float(
            max(
                np.linalg.norm(forward - audit_grid, axis=1).max(),
                np.linalg.norm(inverse - audit_grid, axis=1).max(),
            )
        ),
        "component_derivative_abs_max": bounds["component_derivative_abs_max"],
        "gradient_frobenius_bound": bounds["gradient_frobenius_bound"],
        "divergence_abs_bound": bounds["divergence_abs_bound"],
        "speed_l2_bound_um": bounds["speed_l2_bound_um"],
        "rk4_step_gradient_product_bound": certificate["rk4_step_gradient_product_bound"],
        "rk4_step_jacobian_perturbation_bound": certificate[
            "rk4_step_jacobian_perturbation_bound"
        ],
        "rk4_step_orientation_margin": certificate["rk4_step_orientation_margin"],
        "physical_affine_residual_max_um": affine_residual,
        "minimum_integration_start_halo_um": start_halo,
        "minimum_continuous_path_halo_um": path_halo,
    }
    failed = []
    if not all(np.isfinite(value) for value in values.values()):
        failed.append("finite")
    if values["forward_jacobian_det_min"] < limits["jacobian_det_min"]:
        failed.append("forward_jacobian_det_min")
    if values["forward_jacobian_det_max"] > limits["jacobian_det_max"]:
        failed.append("forward_jacobian_det_max")
    if values["inverse_jacobian_det_min"] < limits["jacobian_det_min"]:
        failed.append("inverse_jacobian_det_min")
    if values["inverse_jacobian_det_max"] > limits["jacobian_det_max"]:
        failed.append("inverse_jacobian_det_max")
    if max(values["forward_inverse_cycle_max_um"], values["inverse_forward_cycle_max_um"]) > limits["cycle_max_um"]:
        failed.append("cycle")
    if values["maximum_displacement_um"] > limits["maximum_displacement_um"]:
        failed.append("maximum_displacement")
    for name in (
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "physical_affine_residual_max_um",
    ):
        if values[name] > limits[name]:
            failed.append(name)
    if values["rk4_step_jacobian_perturbation_bound"] >= 1.0:
        failed.append("rk4_step_orientation_certificate")
    if values["minimum_continuous_path_halo_um"] < limits["minimum_halo_um"]:
        failed.append("continuous_path_halo")
    audit_state = {
        "accepted_audit_forward_yx_um": np.asarray(forward, dtype="<f8", order="C"),
        "accepted_audit_inverse_yx_um": np.asarray(inverse, dtype="<f8", order="C"),
        "accepted_audit_forward_jacobian": np.asarray(forward_jacobian, dtype="<f8", order="C"),
        "accepted_audit_inverse_jacobian": np.asarray(inverse_jacobian, dtype="<f8", order="C"),
        "accepted_audit_forward_cycle_error_yx_um": np.asarray(forward_error, dtype="<f8", order="C"),
        "accepted_audit_inverse_cycle_error_yx_um": np.asarray(inverse_error, dtype="<f8", order="C"),
        "accepted_candidate_affine_correction": np.asarray(correction, dtype="<f8", order="C"),
    }
    payload = {
        "amplitude_um": float(amplitude_um),
        "gate_values": values,
        "failed_gates": failed,
        "accepted": not failed,
        "coarse_coefficients_receipt": _array_receipt(coarse),
        "fine_coefficients_receipt": _array_receipt(fine),
        "affine_correction_receipt": _array_receipt(correction),
        "coefficient_bounds": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in bounds.items()
        },
        "audit_state_receipts": {name: _array_receipt(value) for name, value in audit_state.items()},
    }
    return (
        {**payload, "candidate_id": _domain_id("anatomy-tracker.section-processing-candidate/v2", payload)},
        coarse,
        fine,
        audit_state,
    )


def sample_section_processing_plan_v2(
    image_shape_h_w: tuple[int, int],
    pixel_pitch_y_x_um: tuple[float, float],
    *,
    root_seed: int | str,
    split: str,
    animal_index: int,
    section_index: int,
    animal_id: str | int,
    section_id: str | int,
    deformation_mode: str = "standard",
    coarse_spacing_yx_um: float | tuple[float, float] = 800.0,
    fine_spacing_yx_um: float | tuple[float, float] = 300.0,
    coarse_padding_um: float = 3200.0,
    fine_padding_um: float = 1200.0,
    smoothing_sigma_knots: float = 1.0,
    coarse_weight: float = 0.7,
    fine_weight: float = 0.3,
    a0_um: float = 80.0,
    integration_steps: int = 8,
    jacobian_det_min: float = 0.50,
    jacobian_det_max: float = 2.00,
    cycle_max_um: float = 1.0,
    maximum_displacement_um: float = 400.0,
    component_derivative_abs_max: float = 0.75,
    gradient_frobenius_bound_max: float = 1.0,
    divergence_abs_bound_max: float = 0.75,
    speed_l2_bound_um_max: float = 400.0,
    minimum_halo_um: float = 25.0,
    affine_residual_max_um: float = 1.0e-6,
) -> Mapping[str, object]:
    """Sample one deterministic section-scoped processing plan and first passing amplitude."""
    shape = np.asarray(image_shape_h_w, dtype=np.int64)
    pitch = np.asarray(pixel_pitch_y_x_um, dtype="<f8")
    root_seed = f"0x{_root_seed_hex(root_seed)}"
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    section_index = _nonnegative_integer(section_index, "section_index")
    numeric = np.asarray(
        [
            smoothing_sigma_knots,
            coarse_weight,
            fine_weight,
            a0_um,
            jacobian_det_min,
            jacobian_det_max,
            cycle_max_um,
            maximum_displacement_um,
            component_derivative_abs_max,
            gradient_frobenius_bound_max,
            divergence_abs_bound_max,
            speed_l2_bound_um_max,
            minimum_halo_um,
            affine_residual_max_um,
        ],
        dtype=np.float64,
    )
    if (
        shape.shape != (2,)
        or np.any(shape < 2)
        or pitch.shape != (2,)
        or np.any(~np.isfinite(pitch))
        or np.any(pitch <= 0.0)
        or not isinstance(split, str)
        or not split
        or animal_index < 0
        or section_index < 0
        or str(animal_id) == ""
        or str(section_id) == ""
        or deformation_mode not in {"identity", "standard"}
        or not np.isfinite(numeric).all()
        or smoothing_sigma_knots <= 0.0
        or min(coarse_weight, fine_weight) <= 0.0
        or not np.isclose(coarse_weight + fine_weight, 1.0)
        or a0_um <= 0.0
        or int(integration_steps) != integration_steps
        or integration_steps < 1
        or jacobian_det_min <= 0.0
        or jacobian_det_max < jacobian_det_min
        or min(
            cycle_max_um,
            maximum_displacement_um,
            minimum_halo_um,
        )
        < 0.0
        or min(
            component_derivative_abs_max,
            gradient_frobenius_bound_max,
            divergence_abs_bound_max,
            speed_l2_bound_um_max,
            affine_residual_max_um,
        )
        <= 0.0
    ):
        raise ValueError("section-processing inputs are invalid")
    lower = np.zeros(2, dtype="<f8")
    upper = np.asarray(shape * pitch, dtype="<f8")
    center = np.asarray((lower + upper) / 2.0, dtype="<f8")
    half_extent = np.asarray((upper - lower) / 2.0, dtype="<f8")
    coarse_origin, coarse_spacing, coarse_shape = _lattice_geometry(
        lower, upper, np.broadcast_to(np.asarray(coarse_spacing_yx_um), (2,)), coarse_padding_um
    )
    fine_origin, fine_spacing, fine_shape = _lattice_geometry(
        lower, upper, np.broadcast_to(np.asarray(fine_spacing_yx_um), (2,)), fine_padding_um
    )
    maximum_grid_spacing = np.asarray(fine_spacing / 2.0, dtype="<f8")
    grid_shape, grid_spacing = _grid_geometry(lower, upper, maximum_grid_spacing)
    projection_grid = _fixed_grid(lower, upper, grid_shape)
    audit_grid = _fixed_grid(lower, upper, grid_shape)
    coarse_modes = _affine_modes(
        coarse_shape, coarse_origin, coarse_spacing, center, _taper(coarse_shape)
    )
    fine_modes = _affine_modes(
        fine_shape, fine_origin, fine_spacing, center, _taper(fine_shape)
    )
    rng_sources = {}
    for name, field in (
        ("coarse_svf", "coarse-cubic-bspline-svf"),
        ("fine_svf", "fine-cubic-bspline-svf"),
    ):
        seed = derive_section_processing_seed_v2(
            root_seed,
            split,
            animal_index,
            section_index,
            "section-realization",
            field,
        )
        rng_sources[name] = {
            "stage": "section-realization",
            "field": field,
            "attempt": 0,
            "seed_uint64": f"0x{seed:016x}",
            "generator": "NumPy PCG64DXSM",
        }
    coarse_response = np.stack(
        [
            _physical_affine(
                projection_grid,
                cubic_bspline_velocity_2d_v2(
                    projection_grid, mode, coarse_origin, coarse_spacing
                ),
                center,
                half_extent,
            )
            for mode in coarse_modes
        ],
        1,
    )
    if deformation_mode == "identity":
        raw_coarse = np.zeros(coarse_shape + (2,), dtype="<f4")
        raw_fine = np.zeros(fine_shape + (2,), dtype="<f4")
        projected_coarse = np.zeros_like(raw_coarse)
        projected_fine = np.zeros_like(raw_fine)
        coarse_fit = np.zeros(6, dtype="<f8")
        fine_fit = np.zeros(6, dtype="<f8")
        coarse_projection_residual = 0.0
        fine_projection_residual = 0.0
        unit_correction = np.zeros(6, dtype="<f8")
        unit_affine_residual = 0.0
        combined_unit_rms = 0.0
        effective_a0 = 0.0
    else:
        coarse_rng = np.random.Generator(
            np.random.PCG64DXSM(int(rng_sources["coarse_svf"]["seed_uint64"], 16))
        )
        fine_rng = np.random.Generator(
            np.random.PCG64DXSM(int(rng_sources["fine_svf"]["seed_uint64"], 16))
        )
        raw_coarse = np.asarray(
            _smooth(coarse_rng.standard_normal(coarse_shape + (2,)), smoothing_sigma_knots)
            * _taper(coarse_shape)[..., None],
            dtype="<f4",
            order="C",
        )
        raw_fine = np.asarray(
            _smooth(fine_rng.standard_normal(fine_shape + (2,)), smoothing_sigma_knots)
            * _taper(fine_shape)[..., None],
            dtype="<f4",
            order="C",
        )
        projected_coarse, coarse_fit, coarse_projection_residual = _remove_affine(
            raw_coarse, projection_grid, coarse_modes, coarse_origin, coarse_spacing
        )
        projected_fine, fine_fit, fine_projection_residual = _remove_affine(
            raw_fine, projection_grid, fine_modes, fine_origin, fine_spacing
        )
        coarse_velocity = cubic_bspline_velocity_2d_v2(
            projection_grid, projected_coarse, coarse_origin, coarse_spacing
        )
        fine_velocity = cubic_bspline_velocity_2d_v2(
            projection_grid, projected_fine, fine_origin, fine_spacing
        )
        coarse_rms = float(np.sqrt(np.mean(np.sum(coarse_velocity**2, axis=1))))
        fine_rms = float(np.sqrt(np.mean(np.sum(fine_velocity**2, axis=1))))
        projected_coarse = coarse_weight * projected_coarse / coarse_rms
        projected_fine = fine_weight * projected_fine / fine_rms
        combined_velocity = cubic_bspline_velocity_2d_v2(
            projection_grid, projected_coarse, coarse_origin, coarse_spacing
        ) + cubic_bspline_velocity_2d_v2(
            projection_grid, projected_fine, fine_origin, fine_spacing
        )
        combined_unit_rms = float(np.sqrt(np.mean(np.sum(combined_velocity**2, axis=1))))
        projected_coarse = np.asarray(projected_coarse / combined_unit_rms, dtype="<f4")
        projected_fine = np.asarray(projected_fine / combined_unit_rms, dtype="<f4")
        projected_coarse, unit_correction, unit_affine_residual = _post_float32_affine_correction(
            projection_grid,
            center,
            half_extent,
            projected_coarse,
            coarse_modes,
            coarse_response,
            coarse_origin,
            coarse_spacing,
            projected_fine,
            fine_origin,
            fine_spacing,
        )
        if unit_affine_residual > affine_residual_max_um:
            raise ValueError("section-processing post-float32 affine projection failed")
        effective_a0 = float(a0_um)
    state = {
        "domain_lower_face_yx_um": lower,
        "domain_upper_face_yx_um": upper,
        "domain_center_yx_um": center,
        "domain_half_extent_yx_um": half_extent,
        "projection_grid_yx_um": projection_grid,
        "audit_grid_yx_um": audit_grid,
        "grid_maximum_spacing_yx_um": maximum_grid_spacing,
        "grid_spacing_yx_um": grid_spacing,
        "coarse_origin_yx_um": np.asarray(coarse_origin, dtype="<f8"),
        "coarse_spacing_yx_um": np.asarray(coarse_spacing, dtype="<f8"),
        "fine_origin_yx_um": np.asarray(fine_origin, dtype="<f8"),
        "fine_spacing_yx_um": np.asarray(fine_spacing, dtype="<f8"),
        "raw_coarse_coefficients": raw_coarse,
        "raw_fine_coefficients": raw_fine,
        "coarse_removed_affine_coefficients": np.asarray(coarse_fit, dtype="<f8"),
        "fine_removed_affine_coefficients": np.asarray(fine_fit, dtype="<f8"),
        "projected_coarse_unit_coefficients": projected_coarse,
        "projected_fine_unit_coefficients": projected_fine,
        "unit_post_float32_affine_correction": np.asarray(unit_correction, dtype="<f8"),
    }
    limits = {
        "jacobian_det_min": float(jacobian_det_min),
        "jacobian_det_max": float(jacobian_det_max),
        "cycle_max_um": float(cycle_max_um),
        "maximum_displacement_um": float(maximum_displacement_um),
        "component_derivative_abs_max": float(component_derivative_abs_max),
        "gradient_frobenius_bound": float(gradient_frobenius_bound_max),
        "divergence_abs_bound": float(divergence_abs_bound_max),
        "speed_l2_bound_um": float(speed_l2_bound_um_max),
        "physical_affine_residual_max_um": float(affine_residual_max_um),
        "minimum_halo_um": float(minimum_halo_um),
        "rk4_step_jacobian_perturbation_bound": 1.0,
    }
    schedule = [effective_a0 * factor for factor in SECTION_PROCESSING_V2_CANDIDATE_FACTORS]
    audits = []
    accepted_index = None
    accepted_coarse = accepted_fine = accepted_state = None
    for index, amplitude in enumerate(schedule):
        audit, candidate_coarse, candidate_fine, candidate_state = _candidate_audit(
            state,
            projected_coarse,
            projected_fine,
            coarse_modes,
            coarse_response,
            amplitude,
            int(integration_steps),
            limits,
        )
        audit = {
            "candidate_index": index,
            "amplitude_factor": SECTION_PROCESSING_V2_CANDIDATE_FACTORS[index],
            **audit,
        }
        audits.append(audit)
        if audit["accepted"]:
            accepted_index = index
            accepted_coarse, accepted_fine, accepted_state = (
                candidate_coarse,
                candidate_fine,
                candidate_state,
            )
            break
    if accepted_index is None:
        raise ValueError("no deterministic section-processing amplitude candidate passed")
    state.update(
        {
            "accepted_coarse_coefficients_yx_um": accepted_coarse,
            "accepted_fine_coefficients_yx_um": accepted_fine,
            **accepted_state,
        }
    )
    source_hashes = _source_hashes()
    resolved_config = {
        "schema_version": SECTION_PROCESSING_V2_SCHEMA,
        "algorithm": SECTION_PROCESSING_V2_ALGORITHM,
        "coordinate_contract": "orthogonal section image Y-X physical micrometres",
        "deformation_mode": deformation_mode,
        "image_shape_h_w": shape.tolist(),
        "pixel_pitch_y_x_um": pitch.tolist(),
        "pixel_center_convention": "(index_yx + 0.5) * pixel_pitch_y_x_um",
        "closed_face_domain_yx_um": [lower.tolist(), upper.tolist()],
        "coarse_spacing_yx_um": coarse_spacing.tolist(),
        "fine_spacing_yx_um": fine_spacing.tolist(),
        "coarse_padding_um": float(coarse_padding_um),
        "fine_padding_um": float(fine_padding_um),
        "smoothing_sigma_knots": float(smoothing_sigma_knots),
        "coarse_weight": float(coarse_weight),
        "fine_weight": float(fine_weight),
        "a0_um": float(a0_um),
        "effective_a0_um": float(effective_a0),
        "candidate_factors": list(SECTION_PROCESSING_V2_CANDIDATE_FACTORS),
        "affine_projection": "complete six-DOF 2-D translation plus unconstrained 2x2 linear fit",
        "grid_rule": "closed full pixel-face domain at no more than fine lattice spacing / 2",
        "grid_shape": list(grid_shape),
        "grid_spacing_yx_um": grid_spacing.tolist(),
        "integration": {"method": "fixed-step classical RK4", "steps": int(integration_steps)},
        "orientation_certificate": SECTION_PROCESSING_V2_ORIENTATION_CERTIFICATE,
        "gate_limits": limits,
        "pose_label_policy": "upstream plane pose and anatomy labels are immutable; processing warp is separate",
        "section_id_policy": (
            "processing-stage provenance identifier bound to this plan and render, "
            "excluded from RNG, and not claimed as upstream subject-slab provenance"
        ),
        "learned_checkpoint_dependencies": [],
        "pretrained_feature_dependencies": [],
        "previous_model_dependencies": [],
    }
    provenance = {
        "split": split,
        "root_seed_uint64": f"0x{_root_seed_hex(root_seed)}",
        "animal_index": animal_index,
        "section_index": section_index,
        "animal_id": animal_id,
        "section_id": section_id,
    }
    raw_svf = {
        "coarse_receipt": _array_receipt(raw_coarse),
        "fine_receipt": _array_receipt(raw_fine),
    }
    plan_payload = {
        "schema_version": SECTION_PROCESSING_V2_SCHEMA,
        "algorithm": SECTION_PROCESSING_V2_ALGORITHM,
        "resolved_config": resolved_config,
        "provenance": provenance,
        "rng_sources": rng_sources,
        "raw_svf": raw_svf,
        "source_sha256": source_hashes,
        "source_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
    }
    plan_id = _domain_id("anatomy-tracker.section-processing-plan/v2", plan_payload)
    local_svf = {
        "projected_coarse_unit_receipt": _array_receipt(projected_coarse),
        "projected_fine_unit_receipt": _array_receipt(projected_fine),
        "accepted_coarse_receipt": _array_receipt(accepted_coarse),
        "accepted_fine_receipt": _array_receipt(accepted_fine),
        "coarse_removed_affine_receipt": _array_receipt(coarse_fit),
        "fine_removed_affine_receipt": _array_receipt(fine_fit),
        "unit_affine_correction_receipt": _array_receipt(unit_correction),
        "unit_affine_residual_max_um": float(unit_affine_residual),
        "pre_unit_normalization_combined_rms": float(combined_unit_rms),
    }
    realization = {
        "candidate_schedule_um": schedule,
        "candidate_audits": audits,
        "accepted_candidate_index": accepted_index,
        "accepted_candidate_id": audits[accepted_index]["candidate_id"],
        "accepted_amplitude_um": schedule[accepted_index],
        "no_redraw": True,
    }
    accepted_audit = {
        "source_grid": "fixed closed full-domain audit grid",
        "state_receipts": {name: _array_receipt(value) for name, value in accepted_state.items()},
    }
    realization_payload = {
        "section_processing_plan_id": plan_id,
        "local_svf": local_svf,
        "realization": realization,
        "accepted_audit": accepted_audit,
    }
    realization_id = _domain_id(
        "anatomy-tracker.section-processing-realization/v2", realization_payload
    )
    synthetic_section_id = _domain_id(
        "anatomy-tracker.synthetic-section-processing/v2",
        {
            "section_processing_plan_id": plan_id,
            "section_processing_realization_id": realization_id,
            "provenance": provenance,
        },
    )
    artifact = {
        **plan_payload,
        "section_processing_plan_id": plan_id,
        "section_processing_realization_id": realization_id,
        "synthetic_section_processing_id": synthetic_section_id,
        "local_svf": local_svf,
        "realization": realization,
        "accepted_audit": accepted_audit,
        "state": state,
    }
    artifact["receipt_sha256"] = _payload_sha256(_plan_receipt_payload(artifact))
    return _freeze_value(artifact)


_PLAN_TOP_LEVEL_KEYS = {
    "schema_version",
    "algorithm",
    "resolved_config",
    "provenance",
    "rng_sources",
    "raw_svf",
    "source_sha256",
    "source_sha256_canonicalization",
    "section_processing_plan_id",
    "section_processing_realization_id",
    "synthetic_section_processing_id",
    "local_svf",
    "realization",
    "accepted_audit",
    "state",
    "receipt_sha256",
}
_PLAN_STATE_KEYS = {
    "domain_lower_face_yx_um",
    "domain_upper_face_yx_um",
    "domain_center_yx_um",
    "domain_half_extent_yx_um",
    "projection_grid_yx_um",
    "audit_grid_yx_um",
    "grid_maximum_spacing_yx_um",
    "grid_spacing_yx_um",
    "coarse_origin_yx_um",
    "coarse_spacing_yx_um",
    "fine_origin_yx_um",
    "fine_spacing_yx_um",
    "raw_coarse_coefficients",
    "raw_fine_coefficients",
    "coarse_removed_affine_coefficients",
    "fine_removed_affine_coefficients",
    "projected_coarse_unit_coefficients",
    "projected_fine_unit_coefficients",
    "unit_post_float32_affine_correction",
    "accepted_coarse_coefficients_yx_um",
    "accepted_fine_coefficients_yx_um",
    "accepted_audit_forward_yx_um",
    "accepted_audit_inverse_yx_um",
    "accepted_audit_forward_jacobian",
    "accepted_audit_inverse_jacobian",
    "accepted_audit_forward_cycle_error_yx_um",
    "accepted_audit_inverse_cycle_error_yx_um",
    "accepted_candidate_affine_correction",
}
_ACCEPTED_AUDIT_STATE_KEYS = {
    "accepted_audit_forward_yx_um",
    "accepted_audit_inverse_yx_um",
    "accepted_audit_forward_jacobian",
    "accepted_audit_inverse_jacobian",
    "accepted_audit_forward_cycle_error_yx_um",
    "accepted_audit_inverse_cycle_error_yx_um",
    "accepted_candidate_affine_correction",
}


def _live_plan_state_receipts(plan: Mapping[str, object]) -> dict[str, object]:
    return {name: _array_receipt(value) for name, value in plan["state"].items()}


def _plan_id_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return _json_value(
        {
            key: plan[key]
            for key in (
                "schema_version",
                "algorithm",
                "resolved_config",
                "provenance",
                "rng_sources",
                "raw_svf",
                "source_sha256",
                "source_sha256_canonicalization",
            )
        }
    )


def _realization_id_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return _json_value(
        {
            "section_processing_plan_id": plan["section_processing_plan_id"],
            "local_svf": plan["local_svf"],
            "realization": plan["realization"],
            "accepted_audit": plan["accepted_audit"],
        }
    )


def _plan_receipt_payload(plan: Mapping[str, object]) -> dict[str, object]:
    return {
        "section_processing_plan_id": plan["section_processing_plan_id"],
        "section_processing_realization_id": plan[
            "section_processing_realization_id"
        ],
        "synthetic_section_processing_id": plan["synthetic_section_processing_id"],
        "plan_payload": _plan_id_payload(plan),
        "realization_payload": _realization_id_payload(plan),
        "live_state_receipts": _live_plan_state_receipts(plan),
    }


def section_processing_plan_receipt_v2(
    plan: Mapping[str, object],
) -> dict[str, object]:
    payload = _plan_receipt_payload(plan)
    return {**payload, "receipt_sha256": _payload_sha256(payload)}


def replay_section_processing_plan_v2(
    plan: Mapping[str, object],
) -> Mapping[str, object]:
    config = plan["resolved_config"]
    provenance = plan["provenance"]
    gates = config["gate_limits"]
    return sample_section_processing_plan_v2(
        tuple(config["image_shape_h_w"]),
        tuple(config["pixel_pitch_y_x_um"]),
        root_seed=provenance["root_seed_uint64"],
        split=provenance["split"],
        animal_index=provenance["animal_index"],
        section_index=provenance["section_index"],
        animal_id=provenance["animal_id"],
        section_id=provenance["section_id"],
        deformation_mode=config["deformation_mode"],
        coarse_spacing_yx_um=tuple(config["coarse_spacing_yx_um"]),
        fine_spacing_yx_um=tuple(config["fine_spacing_yx_um"]),
        coarse_padding_um=config["coarse_padding_um"],
        fine_padding_um=config["fine_padding_um"],
        smoothing_sigma_knots=config["smoothing_sigma_knots"],
        coarse_weight=config["coarse_weight"],
        fine_weight=config["fine_weight"],
        a0_um=config["a0_um"],
        integration_steps=config["integration"]["steps"],
        jacobian_det_min=gates["jacobian_det_min"],
        jacobian_det_max=gates["jacobian_det_max"],
        cycle_max_um=gates["cycle_max_um"],
        maximum_displacement_um=gates["maximum_displacement_um"],
        component_derivative_abs_max=gates["component_derivative_abs_max"],
        gradient_frobenius_bound_max=gates["gradient_frobenius_bound"],
        divergence_abs_bound_max=gates["divergence_abs_bound"],
        speed_l2_bound_um_max=gates["speed_l2_bound_um"],
        minimum_halo_um=gates["minimum_halo_um"],
        affine_residual_max_um=gates["physical_affine_residual_max_um"],
    )


def verify_section_processing_plan_v2(
    plan: Mapping[str, object],
    *,
    expected_image_shape_h_w: tuple[int, int],
    expected_pixel_pitch_y_x_um: tuple[float, float],
    expected_split: str,
    expected_animal_index: int,
    expected_section_index: int,
    expected_animal_id: str | int,
    expected_section_id: str | int,
) -> None:
    """Authenticate an accepted plan against authoritative section geometry and IDs."""
    config_keys = {
        "schema_version",
        "algorithm",
        "coordinate_contract",
        "deformation_mode",
        "image_shape_h_w",
        "pixel_pitch_y_x_um",
        "pixel_center_convention",
        "closed_face_domain_yx_um",
        "coarse_spacing_yx_um",
        "fine_spacing_yx_um",
        "coarse_padding_um",
        "fine_padding_um",
        "smoothing_sigma_knots",
        "coarse_weight",
        "fine_weight",
        "a0_um",
        "effective_a0_um",
        "candidate_factors",
        "affine_projection",
        "grid_rule",
        "grid_shape",
        "grid_spacing_yx_um",
        "integration",
        "orientation_certificate",
        "gate_limits",
        "pose_label_policy",
        "section_id_policy",
        "learned_checkpoint_dependencies",
        "pretrained_feature_dependencies",
        "previous_model_dependencies",
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
        "affine_correction_receipt",
        "coefficient_bounds",
        "audit_state_receipts",
        "candidate_id",
    }
    gate_value_keys = {
        "forward_jacobian_det_min",
        "forward_jacobian_det_max",
        "inverse_jacobian_det_min",
        "inverse_jacobian_det_max",
        "forward_inverse_cycle_max_um",
        "inverse_forward_cycle_max_um",
        "maximum_displacement_um",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "rk4_step_gradient_product_bound",
        "rk4_step_jacobian_perturbation_bound",
        "rk4_step_orientation_margin",
        "physical_affine_residual_max_um",
        "minimum_integration_start_halo_um",
        "minimum_continuous_path_halo_um",
    }
    bound_keys = {
        "component_speed_abs_bound_yx_um",
        "speed_l2_bound_um",
        "component_derivative_abs_bound",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
    }
    gate_limit_keys = {
        "jacobian_det_min",
        "jacobian_det_max",
        "cycle_max_um",
        "maximum_displacement_um",
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "physical_affine_residual_max_um",
        "minimum_halo_um",
        "rk4_step_jacobian_perturbation_bound",
    }
    if (
        set(plan) != _PLAN_TOP_LEVEL_KEYS
        or set(plan.get("state", {})) != _PLAN_STATE_KEYS
        or set(plan.get("resolved_config", {})) != config_keys
        or set(plan["resolved_config"].get("integration", {})) != {"method", "steps"}
        or set(plan["resolved_config"].get("gate_limits", {})) != gate_limit_keys
        or set(plan.get("provenance", {}))
        != {
            "split",
            "root_seed_uint64",
            "animal_index",
            "section_index",
            "animal_id",
            "section_id",
        }
        or set(plan.get("rng_sources", {})) != {"coarse_svf", "fine_svf"}
        or any(
            set(value) != {"stage", "field", "attempt", "seed_uint64", "generator"}
            for value in plan["rng_sources"].values()
        )
        or set(plan.get("raw_svf", {})) != {"coarse_receipt", "fine_receipt"}
        or set(plan.get("local_svf", {}))
        != {
            "projected_coarse_unit_receipt",
            "projected_fine_unit_receipt",
            "accepted_coarse_receipt",
            "accepted_fine_receipt",
            "coarse_removed_affine_receipt",
            "fine_removed_affine_receipt",
            "unit_affine_correction_receipt",
            "unit_affine_residual_max_um",
            "pre_unit_normalization_combined_rms",
        }
        or set(plan.get("realization", {}))
        != {
            "candidate_schedule_um",
            "candidate_audits",
            "accepted_candidate_index",
            "accepted_candidate_id",
            "accepted_amplitude_um",
            "no_redraw",
        }
        or set(plan.get("accepted_audit", {})) != {"source_grid", "state_receipts"}
        or set(plan["accepted_audit"].get("state_receipts", {}))
        != _ACCEPTED_AUDIT_STATE_KEYS
    ):
        raise ValueError("section-processing plan has unauthenticated structure")

    config = plan["resolved_config"]
    provenance = plan["provenance"]
    state = plan["state"]
    expected_animal_index = _nonnegative_integer(
        expected_animal_index, "expected_animal_index"
    )
    expected_section_index = _nonnegative_integer(
        expected_section_index, "expected_section_index"
    )
    shape = tuple(int(value) for value in expected_image_shape_h_w)
    pitch = np.asarray(expected_pixel_pitch_y_x_um, dtype="<f8", order="C")
    if (
        tuple(config["image_shape_h_w"]) != shape
        or pitch.shape != (2,)
        or not np.array_equal(pitch, np.asarray(config["pixel_pitch_y_x_um"]))
        or provenance["split"] != expected_split
        or provenance["animal_index"] != expected_animal_index
        or provenance["section_index"] != expected_section_index
        or provenance["animal_id"] != expected_animal_id
        or provenance["section_id"] != expected_section_id
    ):
        raise ValueError("authoritative section geometry or provenance does not match")

    live = _live_plan_state_receipts(plan)
    expected_plan_id = _domain_id(
        "anatomy-tracker.section-processing-plan/v2", _plan_id_payload(plan)
    )
    expected_realization_id = _domain_id(
        "anatomy-tracker.section-processing-realization/v2",
        _realization_id_payload(plan),
    )
    expected_synthetic_id = _domain_id(
        "anatomy-tracker.synthetic-section-processing/v2",
        {
            "section_processing_plan_id": expected_plan_id,
            "section_processing_realization_id": expected_realization_id,
            "provenance": _json_value(provenance),
        },
    )
    receipt_pairs = {
        "raw_coarse_coefficients": plan["raw_svf"]["coarse_receipt"],
        "raw_fine_coefficients": plan["raw_svf"]["fine_receipt"],
        "projected_coarse_unit_coefficients": plan["local_svf"][
            "projected_coarse_unit_receipt"
        ],
        "projected_fine_unit_coefficients": plan["local_svf"][
            "projected_fine_unit_receipt"
        ],
        "accepted_coarse_coefficients_yx_um": plan["local_svf"][
            "accepted_coarse_receipt"
        ],
        "accepted_fine_coefficients_yx_um": plan["local_svf"][
            "accepted_fine_receipt"
        ],
        "coarse_removed_affine_coefficients": plan["local_svf"][
            "coarse_removed_affine_receipt"
        ],
        "fine_removed_affine_coefficients": plan["local_svf"][
            "fine_removed_affine_receipt"
        ],
        "unit_post_float32_affine_correction": plan["local_svf"][
            "unit_affine_correction_receipt"
        ],
    }
    if (
        plan["schema_version"] != SECTION_PROCESSING_V2_SCHEMA
        or plan["algorithm"] != SECTION_PROCESSING_V2_ALGORITHM
        or _json_value(plan["source_sha256"]) != _source_hashes()
        or plan["source_sha256_canonicalization"] != V2_SOURCE_SHA256_CANONICALIZATION
        or plan["section_processing_plan_id"] != expected_plan_id
        or plan["section_processing_realization_id"] != expected_realization_id
        or plan["synthetic_section_processing_id"] != expected_synthetic_id
        or plan["receipt_sha256"] != _payload_sha256(_plan_receipt_payload(plan))
        or any(_json_value(receipt) != live[name] for name, receipt in receipt_pairs.items())
        or _json_value(plan["accepted_audit"]["state_receipts"])
        != {name: live[name] for name in _ACCEPTED_AUDIT_STATE_KEYS}
        or tuple(config["candidate_factors"]) != SECTION_PROCESSING_V2_CANDIDATE_FACTORS
        or config["orientation_certificate"] != SECTION_PROCESSING_V2_ORIENTATION_CERTIFICATE
        or config["integration"]["method"] != "fixed-step classical RK4"
        or config["gate_limits"]["rk4_step_jacobian_perturbation_bound"] != 1.0
        or any(
            config[name]
            for name in (
                "learned_checkpoint_dependencies",
                "pretrained_feature_dependencies",
                "previous_model_dependencies",
            )
        )
        or plan["realization"]["no_redraw"] is not True
    ):
        raise ValueError("section-processing source, identifier, or live receipt changed")

    lower = np.zeros(2, dtype="<f8")
    upper = np.asarray(shape, dtype="<f8") * pitch
    grid_shape, grid_spacing = _grid_geometry(
        lower, upper, np.asarray(state["fine_spacing_yx_um"]) / 2.0
    )
    expected_grid = _fixed_grid(lower, upper, grid_shape)
    if (
        not np.array_equal(state["domain_lower_face_yx_um"], lower)
        or not np.array_equal(state["domain_upper_face_yx_um"], upper)
        or not np.array_equal(state["domain_center_yx_um"], (lower + upper) / 2.0)
        or not np.array_equal(state["domain_half_extent_yx_um"], (upper - lower) / 2.0)
        or not np.array_equal(state["projection_grid_yx_um"], expected_grid)
        or not np.array_equal(state["audit_grid_yx_um"], expected_grid)
        or not np.array_equal(
            state["grid_maximum_spacing_yx_um"],
            np.asarray(state["fine_spacing_yx_um"]) / 2.0,
        )
        or not np.array_equal(state["grid_spacing_yx_um"], grid_spacing)
        or tuple(config["grid_shape"]) != grid_shape
        or not np.array_equal(config["grid_spacing_yx_um"], grid_spacing)
        or not np.array_equal(config["closed_face_domain_yx_um"], [lower, upper])
    ):
        raise ValueError("half-fine-spacing closed-face grids changed")

    realization = plan["realization"]
    audits = tuple(realization["candidate_audits"])
    accepted_index = realization["accepted_candidate_index"]
    schedule = [
        float(config["effective_a0_um"] * factor)
        for factor in SECTION_PROCESSING_V2_CANDIDATE_FACTORS
    ]
    if (
        not isinstance(accepted_index, int)
        or accepted_index < 0
        or accepted_index >= len(audits)
        or len(audits) != accepted_index + 1
        or list(realization["candidate_schedule_um"]) != schedule
        or any(set(audit) != candidate_keys for audit in audits)
        or any(set(audit["gate_values"]) != gate_value_keys for audit in audits)
        or any(set(audit["coefficient_bounds"]) != bound_keys for audit in audits)
        or any(set(audit["audit_state_receipts"]) != _ACCEPTED_AUDIT_STATE_KEYS for audit in audits)
        or any(audit["accepted"] for audit in audits[:accepted_index])
        or not audits[accepted_index]["accepted"]
    ):
        raise ValueError("deterministic candidate fallback structure changed")
    for index, audit in enumerate(audits):
        payload = {
            key: audit[key]
            for key in (
                "amplitude_um",
                "gate_values",
                "failed_gates",
                "accepted",
                "coarse_coefficients_receipt",
                "fine_coefficients_receipt",
                "affine_correction_receipt",
                "coefficient_bounds",
                "audit_state_receipts",
            )
        }
        certificate = rk4_step_orientation_certificate_2d_v2(
            audit["coefficient_bounds"]["gradient_frobenius_bound"],
            config["integration"]["steps"],
        )
        if (
            audit["candidate_index"] != index
            or audit["amplitude_factor"] != SECTION_PROCESSING_V2_CANDIDATE_FACTORS[index]
            or audit["amplitude_um"] != schedule[index]
            or audit["candidate_id"]
            != _domain_id(
                "anatomy-tracker.section-processing-candidate/v2", _json_value(payload)
            )
            or audit["gate_values"]["rk4_step_gradient_product_bound"]
            != certificate["rk4_step_gradient_product_bound"]
            or audit["gate_values"]["rk4_step_jacobian_perturbation_bound"]
            != certificate["rk4_step_jacobian_perturbation_bound"]
            or audit["gate_values"]["rk4_step_orientation_margin"]
            != certificate["rk4_step_orientation_margin"]
            or ("rk4_step_orientation_certificate" in tuple(audit["failed_gates"]))
            != (not certificate["rk4_step_orientation_certified"])
        ):
            raise ValueError("candidate identity or RK4 orientation certificate changed")

    accepted = audits[accepted_index]
    bounds = _coefficient_bounds(
        state["accepted_coarse_coefficients_yx_um"],
        state["coarse_spacing_yx_um"],
        state["accepted_fine_coefficients_yx_um"],
        state["fine_spacing_yx_um"],
    )
    accepted_velocity = _combined_field(
        state,
        state["accepted_coarse_coefficients_yx_um"],
        state["accepted_fine_coefficients_yx_um"],
    )(state["projection_grid_yx_um"], return_gradient=False)
    affine_residual = float(
        np.max(
            np.abs(
                _physical_affine(
                    state["projection_grid_yx_um"],
                    accepted_velocity,
                    state["domain_center_yx_um"],
                    state["domain_half_extent_yx_um"],
                )
            )
        )
    )
    if (
        _json_value(accepted["coefficient_bounds"])
        != _json_value(bounds)
        or not np.isclose(
            accepted["gate_values"]["physical_affine_residual_max_um"],
            affine_residual,
            rtol=0.0,
            atol=1.0e-15,
        )
        or accepted["gate_values"]["minimum_continuous_path_halo_um"]
        != accepted["gate_values"]["minimum_integration_start_halo_um"]
        - accepted["gate_values"]["speed_l2_bound_um"]
        or realization["accepted_candidate_id"] != accepted["candidate_id"]
        or realization["accepted_amplitude_um"] != schedule[accepted_index]
        or _json_value(accepted["coarse_coefficients_receipt"])
        != live["accepted_coarse_coefficients_yx_um"]
        or _json_value(accepted["fine_coefficients_receipt"])
        != live["accepted_fine_coefficients_yx_um"]
        or _json_value(accepted["affine_correction_receipt"])
        != live["accepted_candidate_affine_correction"]
        or _json_value(accepted["audit_state_receipts"])
        != {name: live[name] for name in _ACCEPTED_AUDIT_STATE_KEYS}
    ):
        raise ValueError("accepted section-processing realization changed")

    replayed = replay_section_processing_plan_v2(plan)
    if section_processing_plan_receipt_v2(replayed) != section_processing_plan_receipt_v2(
        plan
    ):
        raise ValueError("deterministic section-processing replay does not match")


def _accepted_field(plan: Mapping[str, object]):
    realization = plan.get("realization", {})
    index = realization.get("accepted_candidate_index")
    audits = realization.get("candidate_audits", ())
    if (
        not isinstance(index, (int, np.integer))
        or index < 0
        or index >= len(audits)
        or not audits[index]["accepted"]
        or not plan.get("section_processing_realization_id")
    ):
        raise ValueError("an accepted section-processing realization is required")
    state = plan["state"]
    return _combined_field(
        state,
        state["accepted_coarse_coefficients_yx_um"],
        state["accepted_fine_coefficients_yx_um"],
    )


def subject_to_processed_points_yx_um_v2(
    points_yx_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Push subject-section points through exp(+v) into processed coordinates."""
    field = _accepted_field(plan)
    points = np.asarray(points_yx_um)
    if plan["resolved_config"]["deformation_mode"] == "identity":
        mapped = np.array(points, copy=True, order="K")
        if not return_jacobian:
            return mapped
        jacobian = np.broadcast_to(
            np.eye(2, dtype=np.result_type(points.dtype, np.float32)),
            points.shape[:-1] + (2, 2),
        ).copy()
        return mapped, jacobian
    return integrate_stationary_velocity_2d_v2(
        np.asarray(points, dtype="<f8", order="C"),
        field,
        steps=plan["resolved_config"]["integration"]["steps"],
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )


def processed_to_subject_points_yx_um_v2(
    points_yx_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    return_jacobian: bool = False,
    batch_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Pull processed pixels back through exp(-v) to subject-section coordinates."""
    field = _accepted_field(plan)
    points = np.asarray(points_yx_um)
    if plan["resolved_config"]["deformation_mode"] == "identity":
        mapped = np.array(points, copy=True, order="K")
        if not return_jacobian:
            return mapped
        jacobian = np.broadcast_to(
            np.eye(2, dtype=np.result_type(points.dtype, np.float32)),
            points.shape[:-1] + (2, 2),
        ).copy()
        return mapped, jacobian
    return integrate_stationary_velocity_2d_v2(
        np.asarray(points, dtype="<f8", order="C"),
        field,
        direction=-1,
        steps=plan["resolved_config"]["integration"]["steps"],
        return_jacobian=return_jacobian,
        batch_size=batch_size,
    )


_SECTION_RASTER_KEYS = {
    "scalar",
    "centre_plane_annotation",
    "centre_plane_support_mask",
    "slab_brain_occupancy",
    "slab_observable_support_mask",
    "slab_modal_annotation",
    "slab_label_purity",
    "centre_label_support_weight",
    "dense_correspondence_weight",
    "dense_correspondence_abstention_mask",
}
_CONTINUOUS_RASTER_KEYS = {
    "scalar",
    "slab_brain_occupancy",
    "slab_label_purity",
    "centre_label_support_weight",
    "dense_correspondence_weight",
}
_NEAREST_RASTER_KEYS = {
    "centre_plane_annotation",
    "centre_plane_support_mask",
    "slab_observable_support_mask",
    "slab_modal_annotation",
}


def _flatten_section_raster(raster: Mapping[str, object]) -> dict[str, np.ndarray]:
    supervision = raster.get("slab_supervision_weight_or_abstention", {})
    if (
        set(raster)
        != {
            "scalar",
            "centre_plane_annotation",
            "centre_plane_support_mask",
            "slab_brain_occupancy",
            "slab_observable_support_mask",
            "slab_modal_annotation",
            "slab_label_purity",
            "centre_label_support_weight",
            "slab_supervision_weight_or_abstention",
        }
        or set(supervision) != {"dense_correspondence_weight", "abstention_mask"}
    ):
        raise ValueError("source section raster has unauthenticated structure")
    return {
        "scalar": np.asarray(raster["scalar"]),
        "centre_plane_annotation": np.asarray(raster["centre_plane_annotation"]),
        "centre_plane_support_mask": np.asarray(raster["centre_plane_support_mask"]),
        "slab_brain_occupancy": np.asarray(raster["slab_brain_occupancy"]),
        "slab_observable_support_mask": np.asarray(
            raster["slab_observable_support_mask"]
        ),
        "slab_modal_annotation": np.asarray(raster["slab_modal_annotation"]),
        "slab_label_purity": np.asarray(raster["slab_label_purity"]),
        "centre_label_support_weight": np.asarray(
            raster["centre_label_support_weight"]
        ),
        "dense_correspondence_weight": np.asarray(
            supervision["dense_correspondence_weight"]
        ),
        "dense_correspondence_abstention_mask": np.asarray(
            supervision["abstention_mask"]
        ),
    }


def _nest_section_raster(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        **{
            name: arrays[name]
            for name in _SECTION_RASTER_KEYS
            if not name.startswith("dense_")
        },
        "slab_supervision_weight_or_abstention": {
            "dense_correspondence_weight": arrays["dense_correspondence_weight"],
            "abstention_mask": arrays[
                "dense_correspondence_abstention_mask"
            ],
        },
    }


def _byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _bilinear_indices_weights(
    source_index_yx: np.ndarray, shape_h_w: tuple[int, int]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    coordinates = np.asarray(source_index_yx, dtype=np.float64)
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lower
    results = []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        iy, ix = lower[..., 0] + dy, lower[..., 1] + dx
        weight = (
            (fraction[..., 0] if dy else 1.0 - fraction[..., 0])
            * (fraction[..., 1] if dx else 1.0 - fraction[..., 1])
        )
        inside = (
            (iy >= 0)
            & (iy < shape_h_w[0])
            & (ix >= 0)
            & (ix < shape_h_w[1])
        )
        results.append(
            (
                np.clip(iy, 0, shape_h_w[0] - 1),
                np.clip(ix, 0, shape_h_w[1] - 1),
                weight,
                inside,
            )
        )
    return tuple(results)


def _bilinear_sample_zero(
    source: np.ndarray, source_index_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source)
    shape = tuple(source.shape[:2])
    extra = (1,) * (source.ndim - 2)
    result = np.zeros(
        source_index_yx.shape[:-1] + source.shape[2:],
        dtype=np.result_type(source.dtype, np.float64),
    )
    all_positive_neighbors_inside = np.ones(source_index_yx.shape[:-1], dtype=bool)
    for iy, ix, weight, inside in _bilinear_indices_weights(source_index_yx, shape):
        values = np.where(
            inside.reshape(inside.shape + extra), source[iy, ix], np.asarray(0, dtype=source.dtype)
        )
        result += values * weight.reshape(weight.shape + extra)
        all_positive_neighbors_inside &= (weight <= 0.0) | inside
    return np.asarray(result, dtype=source.dtype), all_positive_neighbors_inside


def _nearest_sample_zero(
    source: np.ndarray, source_index_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source)
    nearest = np.floor(np.asarray(source_index_yx, dtype=np.float64) + 0.5).astype(
        np.int64
    )
    inside = (
        (nearest[..., 0] >= 0)
        & (nearest[..., 0] < source.shape[0])
        & (nearest[..., 1] >= 0)
        & (nearest[..., 1] < source.shape[1])
    )
    iy = np.clip(nearest[..., 0], 0, source.shape[0] - 1)
    ix = np.clip(nearest[..., 1], 0, source.shape[1] - 1)
    result = np.array(source[iy, ix], copy=True, order="C")
    result[~inside] = np.asarray(0, dtype=source.dtype)
    return result, inside


def _strict_valid_bilinear_coordinates(
    source_coordinates: np.ndarray,
    source_valid: np.ndarray,
    source_index_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_coordinates = np.asarray(source_coordinates)
    source_valid = np.asarray(source_valid, dtype=bool)
    result = np.zeros(source_index_yx.shape[:-1] + (3,), dtype=np.float64)
    valid = np.ones(source_index_yx.shape[:-1], dtype=bool)
    for iy, ix, weight, inside in _bilinear_indices_weights(
        source_index_yx, tuple(source_valid.shape)
    ):
        neighbor_valid = inside & source_valid[iy, ix]
        positive = weight > 0.0
        valid &= (~positive) | neighbor_valid
        values = np.where(neighbor_valid[..., None], source_coordinates[iy, ix], 0.0)
        result += values * weight[..., None]
    result[~valid] = np.nan
    return result, valid


def _processed_pixel_geometry(
    plan: Mapping[str, object], batch_size: int | None
) -> dict[str, np.ndarray]:
    shape = tuple(plan["resolved_config"]["image_shape_h_w"])
    pitch = np.asarray(plan["resolved_config"]["pixel_pitch_y_x_um"], dtype="<f8")
    centres = np.stack(
        np.meshgrid(
            (np.arange(shape[0], dtype=np.float64) + 0.5) * pitch[0],
            (np.arange(shape[1], dtype=np.float64) + 0.5) * pitch[1],
            indexing="ij",
        ),
        -1,
    )
    subject_points = processed_to_subject_points_yx_um_v2(
        centres, plan, batch_size=batch_size
    )
    source_index = np.asarray(subject_points, dtype=np.float64) / pitch - 0.5
    return {
        "processed_pixel_centres_yx_um": np.asarray(centres, dtype="<f8", order="C"),
        "subject_pullback_points_yx_um": np.asarray(
            subject_points, dtype="<f8", order="C"
        ),
        "source_index_yx": np.asarray(source_index, dtype="<f8", order="C"),
    }


def _source_input_reference(
    arrays: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
    source_stage_receipt: Mapping[str, object],
    pose_anatomy_reference: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "source_stage_receipt": _json_value(source_stage_receipt),
        "source_raster_array_receipts": {
            name: _array_receipt(arrays[name]) for name in sorted(_SECTION_RASTER_KEYS)
        },
        "source_mapped_ccf_coordinate_receipt": _array_receipt(coordinates),
        "source_image_shape_h_w": list(coordinates.shape[:2]),
        "source_pixel_pitch_y_x_um": list(
            plan["resolved_config"]["pixel_pitch_y_x_um"]
        ),
        "pose_anatomy_reference": _json_value(pose_anatomy_reference),
    }
    return {**payload, "source_input_receipt_sha256": _payload_sha256(payload)}


def _plan_reference(plan: Mapping[str, object]) -> dict[str, object]:
    plan_receipt = section_processing_plan_receipt_v2(plan)
    return {
        "section_processing_plan_id": plan["section_processing_plan_id"],
        "section_processing_realization_id": plan[
            "section_processing_realization_id"
        ],
        "synthetic_section_processing_id": plan["synthetic_section_processing_id"],
        "section_processing_plan_receipt": _json_value(plan_receipt),
        "section_processing_realization_receipt_sha256": _payload_sha256(
            _realization_id_payload(plan)
        ),
    }


def _render_identity_payload(render: Mapping[str, object]) -> dict[str, object]:
    return _json_value(
        {
            key: render[key]
            for key in (
                "schema_version",
                "algorithm",
                "implementation_source_sha256",
                "implementation_source_sha256_canonicalization",
                "source_input_reference",
                "plan_reference",
                "mapping_contract",
                "interpolation_semantics",
                "pose_anatomy_policy",
                "identity_reference_path",
                "state_receipts",
                "raster_array_receipts",
                "mapped_ccf_coordinate_receipt",
            )
        }
    )


def section_processing_render_receipt_v2(
    render: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "section_processing_render_id": render["section_processing_render_id"],
        "render_identity_payload": _render_identity_payload(render),
    }
    return {**payload, "receipt_sha256": _payload_sha256(payload)}


def _make_section_processing_render_from_arrays_v2(
    source_raster: Mapping[str, object],
    source_mapped_ccf_coordinates_ap_dv_ml_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    source_stage_receipt: Mapping[str, object],
    pose_anatomy_reference: Mapping[str, object],
    batch_size: int | None = None,
) -> Mapping[str, object]:
    """Apply the accepted 2-D processing warp after the subject-slab reduction."""
    arrays = _flatten_section_raster(source_raster)
    coordinates = np.asarray(source_mapped_ccf_coordinates_ap_dv_ml_um)
    shape = tuple(plan["resolved_config"]["image_shape_h_w"])
    if (
        set(arrays) != _SECTION_RASTER_KEYS
        or any(array.shape != shape for array in arrays.values())
        or coordinates.shape != shape + (3,)
        or not isinstance(source_stage_receipt, Mapping)
        or not source_stage_receipt
        or not isinstance(pose_anatomy_reference, Mapping)
        or not pose_anatomy_reference
    ):
        raise ValueError("section-processing source arrays or references are invalid")

    geometry = _processed_pixel_geometry(plan, batch_size)
    source_index = geometry["source_index_yx"]
    identity = plan["resolved_config"]["deformation_mode"] == "identity"
    if identity:
        output = {
            name: np.array(array, copy=True, order="C") for name, array in arrays.items()
        }
        mapped_coordinates = np.array(coordinates, copy=True, order="C")
        dense_valid = (
            np.isfinite(coordinates).all(axis=-1)
            & (arrays["dense_correspondence_weight"] > 0.0)
            & ~arrays["dense_correspondence_abstention_mask"].astype(bool)
        )
        bilinear_domain_valid = np.ones(shape, dtype=bool)
        nearest_domain_valid = np.ones(shape, dtype=bool)
    else:
        output = {}
        bilinear_domain_valid = np.ones(shape, dtype=bool)
        for name in _CONTINUOUS_RASTER_KEYS:
            output[name], current_valid = _bilinear_sample_zero(
                arrays[name], source_index
            )
            bilinear_domain_valid &= current_valid
        nearest_domain_valid = np.ones(shape, dtype=bool)
        for name in _NEAREST_RASTER_KEYS:
            output[name], current_valid = _nearest_sample_zero(
                arrays[name], source_index
            )
            nearest_domain_valid &= current_valid
        source_dense_valid = (
            np.isfinite(coordinates).all(axis=-1)
            & (arrays["dense_correspondence_weight"] > 0.0)
            & ~arrays["dense_correspondence_abstention_mask"].astype(bool)
        )
        mapped_coordinates, dense_valid = _strict_valid_bilinear_coordinates(
            coordinates, source_dense_valid, source_index
        )
        output["dense_correspondence_weight"][~dense_valid] = np.asarray(
            0, dtype=output["dense_correspondence_weight"].dtype
        )
        output["dense_correspondence_abstention_mask"] = ~dense_valid

    geometry.update(
        {
            "bilinear_domain_valid_mask": np.asarray(
                bilinear_domain_valid, dtype=bool, order="C"
            ),
            "nearest_domain_valid_mask": np.asarray(
                nearest_domain_valid, dtype=bool, order="C"
            ),
            "dense_coordinate_valid_mask": np.asarray(
                dense_valid, dtype=bool, order="C"
            ),
        }
    )
    interpolation = {
        "scalar": "bilinear in source centre-index coordinates; zero outside",
        "continuous_supervision": [
            "bilinear in source centre-index coordinates; zero outside",
            *sorted(_CONTINUOUS_RASTER_KEYS - {"scalar"}),
        ],
        "categorical_labels": [
            "nearest centre, half ties toward increasing index; label zero outside",
            "centre_plane_annotation",
            "slab_modal_annotation",
        ],
        "support_masks": [
            "nearest centre, half ties toward increasing index; false outside",
            "centre_plane_support_mask",
            "slab_observable_support_mask",
        ],
        "mapped_ccf_coordinates": (
            "strict-valid bilinear: every positive-weight neighbor must be in bounds, "
            "finite, have positive dense weight, and not abstain; otherwise NaN"
        ),
        "dense_correspondence_weight": (
            "bilinear then forced to zero wherever mapped CCF coordinates are invalid"
        ),
        "dense_correspondence_abstention_mask": (
            "logical not of strict mapped-CCF coordinate validity"
        ),
    }
    source_reference = _source_input_reference(
        arrays,
        coordinates,
        source_stage_receipt,
        pose_anatomy_reference,
        plan,
    )
    mapped_coordinates = np.asarray(mapped_coordinates, dtype=coordinates.dtype, order="C")
    render = {
        "schema_version": SECTION_PROCESSING_RENDER_V2_SCHEMA,
        "algorithm": SECTION_PROCESSING_RENDER_V2_ALGORITHM,
        "implementation_source_sha256": _render_source_hashes(),
        "implementation_source_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
        "source_input_reference": source_reference,
        "plan_reference": _plan_reference(plan),
        "mapping_contract": {
            "forward": "subject section Y-X to processed Y-X is exp(+v)",
            "render_pullback": "processed pixel centre to subject Y-X is exp(-v)",
            "physical_to_source_index": "subject_yx_um / pixel_pitch_y_x_um - 0.5",
        },
        "interpolation_semantics": interpolation,
        "pose_anatomy_policy": {
            "policy": "preserve upstream pose/anatomy reference unchanged",
            "processing_warp_is_separate_from_plane_pose": True,
            "pose_anatomy_reference": _json_value(pose_anatomy_reference),
        },
        "identity_reference_path": identity,
        "state": geometry,
        "state_receipts": {name: _array_receipt(value) for name, value in geometry.items()},
        "raster": _nest_section_raster(output),
        "raster_array_receipts": {
            name: _array_receipt(output[name]) for name in sorted(_SECTION_RASTER_KEYS)
        },
        "mapped_ccf_physical_coordinates_ap_dv_ml_um": mapped_coordinates,
        "mapped_ccf_coordinate_receipt": _array_receipt(mapped_coordinates),
    }
    render["section_processing_render_id"] = _domain_id(
        "anatomy-tracker.section-processing-render/v2",
        _render_identity_payload(render),
    )
    render["receipt_sha256"] = _payload_sha256(
        section_processing_render_receipt_v2(render)
    )
    return _freeze_value(render)


def _replay_section_processing_render_from_arrays_v2(
    render: Mapping[str, object],
    source_raster: Mapping[str, object],
    source_mapped_ccf_coordinates_ap_dv_ml_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    source_stage_receipt: Mapping[str, object],
    pose_anatomy_reference: Mapping[str, object],
    batch_size: int | None = None,
) -> Mapping[str, object]:
    return _make_section_processing_render_from_arrays_v2(
        source_raster,
        source_mapped_ccf_coordinates_ap_dv_ml_um,
        plan,
        source_stage_receipt=source_stage_receipt,
        pose_anatomy_reference=pose_anatomy_reference,
        batch_size=batch_size,
    )


def _verify_section_processing_render_from_arrays_v2(
    render: Mapping[str, object],
    source_raster: Mapping[str, object],
    source_mapped_ccf_coordinates_ap_dv_ml_um: np.ndarray,
    plan: Mapping[str, object],
    *,
    source_stage_receipt: Mapping[str, object],
    pose_anatomy_reference: Mapping[str, object],
) -> None:
    arrays = _flatten_section_raster(source_raster)
    coordinates = np.asarray(source_mapped_ccf_coordinates_ap_dv_ml_um)
    config = plan["resolved_config"]
    provenance = plan["provenance"]
    verify_section_processing_plan_v2(
        plan,
        expected_image_shape_h_w=tuple(config["image_shape_h_w"]),
        expected_pixel_pitch_y_x_um=tuple(config["pixel_pitch_y_x_um"]),
        expected_split=provenance["split"],
        expected_animal_index=provenance["animal_index"],
        expected_section_index=provenance["section_index"],
        expected_animal_id=provenance["animal_id"],
        expected_section_id=provenance["section_id"],
    )
    state_keys = {
        "processed_pixel_centres_yx_um",
        "subject_pullback_points_yx_um",
        "source_index_yx",
        "bilinear_domain_valid_mask",
        "nearest_domain_valid_mask",
        "dense_coordinate_valid_mask",
    }
    if (
        set(render)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "source_input_reference",
            "plan_reference",
            "mapping_contract",
            "interpolation_semantics",
            "pose_anatomy_policy",
            "identity_reference_path",
            "state",
            "state_receipts",
            "raster",
            "raster_array_receipts",
            "mapped_ccf_physical_coordinates_ap_dv_ml_um",
            "mapped_ccf_coordinate_receipt",
            "section_processing_render_id",
            "receipt_sha256",
        }
        or set(render.get("state", {})) != state_keys
        or set(render.get("state_receipts", {})) != state_keys
        or set(render.get("raster_array_receipts", {})) != _SECTION_RASTER_KEYS
        or set(render.get("mapping_contract", {}))
        != {"forward", "render_pullback", "physical_to_source_index"}
        or set(render.get("pose_anatomy_policy", {}))
        != {
            "policy",
            "processing_warp_is_separate_from_plane_pose",
            "pose_anatomy_reference",
        }
    ):
        raise ValueError("section-processing render has unauthenticated structure")
    output = _flatten_section_raster(render["raster"])
    source_reference = _source_input_reference(
        arrays,
        coordinates,
        source_stage_receipt,
        pose_anatomy_reference,
        plan,
    )
    expected_plan_reference = _plan_reference(plan)
    mapped = render["mapped_ccf_physical_coordinates_ap_dv_ml_um"]
    live_state = {
        name: _array_receipt(value) for name, value in render["state"].items()
    }
    live_output = {
        name: _array_receipt(output[name]) for name in sorted(_SECTION_RASTER_KEYS)
    }
    expected_render_id = _domain_id(
        "anatomy-tracker.section-processing-render/v2",
        _render_identity_payload(render),
    )
    dense_valid = np.asarray(render["state"]["dense_coordinate_valid_mask"], dtype=bool)
    if (
        render["schema_version"] != SECTION_PROCESSING_RENDER_V2_SCHEMA
        or render["algorithm"] != SECTION_PROCESSING_RENDER_V2_ALGORITHM
        or _json_value(render["implementation_source_sha256"])
        != _render_source_hashes()
        or render["implementation_source_sha256_canonicalization"]
        != V2_SOURCE_SHA256_CANONICALIZATION
        or _json_value(render["source_input_reference"])
        != _json_value(source_reference)
        or _json_value(render["plan_reference"])
        != _json_value(expected_plan_reference)
        or _json_value(render["pose_anatomy_policy"]["pose_anatomy_reference"])
        != _json_value(pose_anatomy_reference)
        or render["pose_anatomy_policy"][
            "processing_warp_is_separate_from_plane_pose"
        ]
        is not True
        or render["identity_reference_path"]
        != (config["deformation_mode"] == "identity")
        or _json_value(render["state_receipts"]) != live_state
        or _json_value(render["raster_array_receipts"]) != live_output
        or _json_value(render["mapped_ccf_coordinate_receipt"])
        != _array_receipt(mapped)
        or render["section_processing_render_id"] != expected_render_id
        or render["receipt_sha256"]
        != _payload_sha256(section_processing_render_receipt_v2(render))
        or not np.array_equal(
            render["state"]["source_index_yx"],
            np.asarray(render["state"]["subject_pullback_points_yx_um"])
            / np.asarray(config["pixel_pitch_y_x_um"])
            - 0.5,
        )
        or (
            not render["identity_reference_path"]
            and not np.array_equal(np.isfinite(mapped).all(axis=-1), dense_valid)
        )
        or not np.array_equal(
            output["dense_correspondence_abstention_mask"].astype(bool), ~dense_valid
        )
        or np.any(output["dense_correspondence_weight"][~dense_valid] != 0)
    ):
        raise ValueError("section-processing render source, plan, or live receipt changed")
    if render["identity_reference_path"] and (
        any(not _byte_equal(output[name], arrays[name]) for name in _SECTION_RASTER_KEYS)
        or not _byte_equal(mapped, coordinates)
    ):
        raise ValueError("identity processing render is not byte-identical")
    replayed = _replay_section_processing_render_from_arrays_v2(
        render,
        source_raster,
        coordinates,
        plan,
        source_stage_receipt=source_stage_receipt,
        pose_anatomy_reference=pose_anatomy_reference,
    )
    if section_processing_render_receipt_v2(replayed) != section_processing_render_receipt_v2(
        render
    ):
        raise ValueError("deterministic section-processing render replay does not match")
    replay_arrays = _flatten_section_raster(replayed["raster"])
    if (
        any(not _byte_equal(output[name], replay_arrays[name]) for name in _SECTION_RASTER_KEYS)
        or not _byte_equal(
            mapped,
            replayed["mapped_ccf_physical_coordinates_ap_dv_ml_um"],
        )
        or any(
            not _byte_equal(render["state"][name], replayed["state"][name])
            for name in state_keys
        )
    ):
        raise ValueError("deterministic section-processing render arrays do not match")


def _orthogonal_section_pixel_metric(
    subject_physical_coordinates_ap_dv_ml_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        subject_physical_coordinates_ap_dv_ml_um, dtype=np.float64
    )
    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] != 3
        or min(coordinates.shape[:2]) < 2
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("subject-section physical coordinate raster is invalid")
    y_vectors = coordinates[1:, :, :] - coordinates[:-1, :, :]
    x_vectors = coordinates[:, 1:, :] - coordinates[:, :-1, :]
    y_steps = np.linalg.norm(y_vectors, axis=-1)
    x_steps = np.linalg.norm(x_vectors, axis=-1)
    pitch = np.asarray([y_steps.mean(), x_steps.mean()], dtype=np.float64)
    y_reference = y_vectors.mean(axis=(0, 1))
    x_reference = x_vectors.mean(axis=(0, 1))
    tolerance = SECTION_PROCESSING_V2_PIXEL_METRIC_RELATIVE_TOLERANCE
    y_residual = float(np.linalg.norm(y_vectors - y_reference, axis=-1).max())
    x_residual = float(np.linalg.norm(x_vectors - x_reference, axis=-1).max())
    parallelogram = (
        coordinates[1:, 1:, :]
        - coordinates[1:, :-1, :]
        - coordinates[:-1, 1:, :]
        + coordinates[:-1, :-1, :]
    )
    parallelogram_residual = float(np.linalg.norm(parallelogram, axis=-1).max())
    reference_norm_product = float(
        np.linalg.norm(y_reference) * np.linalg.norm(x_reference)
    )
    orthogonality = (
        np.inf
        if reference_norm_product <= 0.0
        else abs(float(y_reference @ x_reference)) / reference_norm_product
    )
    if (
        np.any(~np.isfinite(pitch))
        or np.any(pitch <= 0.0)
        or not np.isfinite(orthogonality)
        or y_residual > 1.0e-6 + tolerance * pitch[0]
        or x_residual > 1.0e-6 + tolerance * pitch[1]
        or parallelogram_residual > 1.0e-6 + tolerance * float(pitch.max())
        or orthogonality > tolerance
    ):
        raise ValueError(
            "subject-section physical pixel metric is not a constant orthogonal parallelogram grid"
        )
    return pitch, y_steps, x_steps


def _subject_slab_inputs(
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    np.ndarray,
    Mapping[str, object],
    Mapping[str, object],
]:
    from training.arbitrary_plane_subject_slab_v2 import (
        subject_slab_render_receipt_v2,
    )

    coordinate = subject_slab_render["coordinate_map"]
    centre_index = int(coordinate["kernel"]["centre_index"])
    mapped = np.asarray(
        coordinate["arrays"][
            "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
        ][centre_index]
    )
    subject_physical = np.asarray(
        coordinate["arrays"][
            "subject_physical_coordinates_ap_dv_ml_um_float64"
        ][centre_index]
    )
    arrays = _flatten_section_raster(subject_slab_render["raster"])
    pitch, y_steps, x_steps = _orthogonal_section_pixel_metric(subject_physical)
    source_receipt = subject_slab_render_receipt_v2(subject_slab_render)
    pose_reference = {
        "source_subject_coordinate_map_id": subject_slab_render[
            "subject_coordinate_map_id"
        ],
        "context_reference": _json_value(subject_slab_render["context_reference"]),
        "precursor_reference": _json_value(subject_slab_render["precursor_reference"]),
        "deformation_reference": _json_value(coordinate["deformation_reference"]),
        "atlas_domain": _json_value(coordinate["atlas_domain"]),
        "kernel": _json_value(coordinate["kernel"]),
        "centre_plane_fit": _json_value(coordinate["centre_plane_fit"]),
    }
    coordinate_arrays = coordinate["arrays"]
    coordinate_receipts = coordinate["array_receipts"]
    if (
        tuple(mapped.shape[:2])
        != tuple(plan["resolved_config"]["image_shape_h_w"])
        or subject_physical.shape != mapped.shape
        or not np.allclose(
            pitch,
            plan["resolved_config"]["pixel_pitch_y_x_um"],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        or not np.allclose(
            y_steps,
            pitch[0],
            rtol=SECTION_PROCESSING_V2_PIXEL_METRIC_RELATIVE_TOLERANCE,
            atol=1.0e-6,
        )
        or not np.allclose(
            x_steps,
            pitch[1],
            rtol=SECTION_PROCESSING_V2_PIXEL_METRIC_RELATIVE_TOLERANCE,
            atol=1.0e-6,
        )
        or subject_slab_render["receipt_sha256"] != _payload_sha256(source_receipt)
        or _json_value(subject_slab_render["raster_array_receipts"])
        != {name: _array_receipt(arrays[name]) for name in arrays}
        or _json_value(
            coordinate_receipts[
                "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
            ]
        )
        != _array_receipt(
            coordinate_arrays[
                "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
            ]
        )
        or _json_value(
            coordinate_receipts[
                "subject_physical_coordinates_ap_dv_ml_um_float64"
            ]
        )
        != _array_receipt(
            coordinate_arrays[
                "subject_physical_coordinates_ap_dv_ml_um_float64"
            ]
        )
    ):
        raise ValueError("subject-slab source receipt or physical pixel metric changed")
    return subject_slab_render["raster"], mapped, source_receipt, pose_reference


def _verify_subject_slab_processing_lineage(
    subject_slab_render: Mapping[str, object], plan: Mapping[str, object]
) -> None:
    precursor_reference = subject_slab_render["precursor_reference"]
    provenance = plan["provenance"]
    verify_section_processing_plan_v2(
        plan,
        expected_image_shape_h_w=tuple(plan["resolved_config"]["image_shape_h_w"]),
        expected_pixel_pitch_y_x_um=tuple(
            plan["resolved_config"]["pixel_pitch_y_x_um"]
        ),
        expected_split=precursor_reference["split"],
        expected_animal_index=precursor_reference["animal_index"],
        expected_section_index=precursor_reference["plane_sample_index"],
        expected_animal_id=precursor_reference["animal_id"],
        expected_section_id=provenance["section_id"],
    )


def _make_section_processing_render_with_mapper_v2(
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> Mapping[str, object]:
    from training.arbitrary_plane_subject_slab_v2 import (
        _verify_subject_slab_render_with_mapper_v2,
    )

    _verify_subject_slab_render_with_mapper_v2(
        subject_slab_render,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    raster, mapped, source_receipt, pose_reference = _subject_slab_inputs(
        subject_slab_render, plan
    )
    _verify_subject_slab_processing_lineage(subject_slab_render, plan)
    return _make_section_processing_render_from_arrays_v2(
        raster,
        mapped,
        plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
        batch_size=batch_size,
    )


def make_section_processing_render_v2(
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
) -> Mapping[str, object]:
    """Verify the upstream subject slab, then create its processing render."""
    return _make_section_processing_render_with_mapper_v2(
        subject_slab_render,
        plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )


def _replay_section_processing_render_with_mapper_v2(
    render: Mapping[str, object],
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> Mapping[str, object]:
    return _make_section_processing_render_with_mapper_v2(
        subject_slab_render,
        plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def replay_section_processing_render_v2(
    render: Mapping[str, object],
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
) -> Mapping[str, object]:
    return _replay_section_processing_render_with_mapper_v2(
        render,
        subject_slab_render,
        plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )


def _verify_section_processing_render_with_mapper_v2(
    render: Mapping[str, object],
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> None:
    from training.arbitrary_plane_subject_slab_v2 import (
        _verify_subject_slab_render_with_mapper_v2,
    )

    _verify_subject_slab_render_with_mapper_v2(
        subject_slab_render,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    raster, mapped, source_receipt, pose_reference = _subject_slab_inputs(
        subject_slab_render, plan
    )
    _verify_subject_slab_processing_lineage(subject_slab_render, plan)
    _verify_section_processing_render_from_arrays_v2(
        render,
        raster,
        mapped,
        plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )


def verify_section_processing_render_v2(
    render: Mapping[str, object],
    subject_slab_render: Mapping[str, object],
    plan: Mapping[str, object],
    prepared_context: Mapping[str, object],
    precursor: Mapping[str, object],
    *,
    subject_plan: Mapping[str, object] | None,
    batch_size: int | None = None,
) -> None:
    """Verify the full authenticated subject-slab -> processing-render chain."""
    _verify_section_processing_render_with_mapper_v2(
        render,
        subject_slab_render,
        plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )


make_section_processing_render_from_subject_slab_v2 = make_section_processing_render_v2
verify_section_processing_render_from_subject_slab_v2 = verify_section_processing_render_v2
