"""Standalone 2-D operators for arbitrary-plane synthetic sections.

Maps are float32 absolute pixel-centre coordinates in ``(x, y)`` order.  A map
``A_to_B[:, y, x]`` is the point in B occupied by point ``(x, y)`` in A.  The
same array is a pullback only when its name is ``output_to_input``: sampling an
input with that map produces an image on the output grid.  All interpolation
uses the ``align_corners=True`` pixel convention.

This module contains no learned model, checkpoint, or historical generator
dependency.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


ARBITRARY_PLANE_SYNTHETIC_OPS_VERSION = "arbitrary-plane-synthetic-ops-g1-v1"


def identity_pixel_map(shape: tuple[int, int]) -> np.ndarray:
    """Return a ``(2,H,W)`` float32 identity map ordered as x, y."""
    height, width = (int(value) for value in shape)
    y, x = np.mgrid[:height, :width]
    return np.stack((x, y)).astype(np.float32)


def _bilinear_indices(
    output_to_input: np.ndarray,
    input_shape: tuple[int, int],
    padding_mode: str,
) -> tuple[np.ndarray, ...]:
    pixel_map = np.asarray(output_to_input, dtype=np.float32)
    if pixel_map.ndim != 3 or pixel_map.shape[0] != 2:
        raise ValueError("pixel map must have shape (2,H,W)")
    if not np.isfinite(pixel_map).all():
        raise ValueError("pixel map must be finite")
    height, width = input_shape
    if padding_mode not in {"zeros", "border"}:
        raise ValueError("padding_mode must be zeros or border")
    x = pixel_map[0]
    y = pixel_map[1]
    if padding_mode == "border":
        x = np.clip(x, 0.0, width - 1.0)
        y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    return x0, x1, y0, y1, wx, wy


def bilinear_sample_scalar(
    image: np.ndarray,
    output_to_input: np.ndarray,
    *,
    padding_mode: str = "zeros",
) -> np.ndarray:
    """Pull back one ``H_in x W_in`` scalar image onto the map's output grid."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("scalar image must have shape (H,W)")
    height, width = values.shape
    x0, x1, y0, y1, wx, wy = _bilinear_indices(
        output_to_input, (height, width), padding_mode
    )

    def corner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sampled = values[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
        if padding_mode == "zeros":
            sampled = sampled * ((x >= 0) & (x < width) & (y >= 0) & (y < height))
        return sampled

    result = (
        corner(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + corner(x1, y0) * wx * (1.0 - wy)
        + corner(x0, y1) * (1.0 - wx) * wy
        + corner(x1, y1) * wx * wy
    )
    return result.astype(np.float32, copy=False)


def bilinear_sample_field(
    field_xy: np.ndarray,
    output_to_input: np.ndarray,
    *,
    padding_mode: str = "border",
) -> np.ndarray:
    """Pull back a ``(2,H_in,W_in)`` x,y field onto the map's output grid."""
    field = np.asarray(field_xy, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError("field must have shape (2,H,W)")
    return np.stack(
        [
            bilinear_sample_scalar(field[axis], output_to_input, padding_mode=padding_mode)
            for axis in range(2)
        ]
    ).astype(np.float32, copy=False)


def nearest_sample_labels(
    labels: np.ndarray,
    output_to_input: np.ndarray,
    *,
    outside_label: int = 0,
) -> np.ndarray:
    """Nearest-centre pullback that never casts integer labels through float."""
    values = np.asarray(labels)
    pixel_map = np.asarray(output_to_input, dtype=np.float32)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be one integer array with shape (H,W)")
    if pixel_map.ndim != 3 or pixel_map.shape[0] != 2 or not np.isfinite(pixel_map).all():
        raise ValueError("pixel map must be finite with shape (2,H,W)")
    height, width = values.shape
    x = np.floor(pixel_map[0] + 0.5).astype(np.int64)
    y = np.floor(pixel_map[1] + 0.5).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled = values[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    return np.where(valid, sampled, np.asarray(outside_label, dtype=values.dtype))


def compose_pixel_maps(
    output_to_middle: np.ndarray,
    middle_to_input: np.ndarray,
) -> np.ndarray:
    """Return ``middle_to_input(output_to_middle(x))`` using border extension."""
    return bilinear_sample_field(
        middle_to_input, output_to_middle, padding_mode="border"
    )


def physical_velocity_to_pixel(
    velocity_uv_um: np.ndarray,
    pixel_basis_uv_um: np.ndarray,
) -> np.ndarray:
    """Convert an in-plane physical SVF to x,y pixel displacement.

    ``pixel_basis_uv_um`` is 2x2.  Its columns are the physical ``(u,v)``
    displacement, in micrometres, produced by one +x or +y raster pixel.
    """
    velocity = np.asarray(velocity_uv_um, dtype=np.float64)
    basis = np.asarray(pixel_basis_uv_um, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError("physical velocity must have shape (2,H,W)")
    if basis.shape != (2, 2) or not np.isfinite(basis).all() or np.linalg.det(basis) <= 0.0:
        raise ValueError("pixel basis must be one finite positive-orientation 2x2 matrix")
    converted = np.linalg.solve(basis, velocity.reshape(2, -1))
    return converted.reshape(velocity.shape).astype(np.float32)


def remove_tissue_affine_component(
    velocity_xy: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    """Subtract the least-squares affine velocity fitted over tissue pixels."""
    velocity = np.asarray(velocity_xy, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if velocity.ndim != 3 or velocity.shape[0] != 2 or tissue.shape != velocity.shape[1:]:
        raise ValueError("velocity and tissue mask must have shapes (2,H,W) and (H,W)")
    if np.count_nonzero(tissue) < 3:
        raise ValueError("affine removal needs at least three tissue pixels")
    height, width = tissue.shape
    identity = identity_pixel_map((height, width))
    x = identity[0] * (2.0 / max(width - 1, 1)) - 1.0
    y = identity[1] * (2.0 / max(height - 1, 1)) - 1.0
    design = np.stack((np.ones_like(x), x, y), axis=-1)
    coefficients = np.linalg.lstsq(
        design[tissue].astype(np.float64),
        velocity[:, tissue].T.astype(np.float64),
        rcond=None,
    )[0]
    affine = np.einsum("hwk,kc->chw", design, coefficients)
    return (velocity - affine).astype(np.float32)


def sample_multiscale_physical_velocity(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    correlation_lengths_px: tuple[float, ...],
    rms_amplitudes_um: tuple[float, ...],
) -> np.ndarray:
    """Sample a deterministic multiscale smooth ``(u,v)`` SVF in micrometres."""
    height, width = (int(value) for value in shape)
    scales = np.asarray(correlation_lengths_px, dtype=np.float64)
    amplitudes = np.asarray(rms_amplitudes_um, dtype=np.float64)
    if (
        height < 2
        or width < 2
        or scales.ndim != 1
        or len(scales) == 0
        or scales.shape != amplitudes.shape
        or not np.isfinite(scales).all()
        or not np.isfinite(amplitudes).all()
        or np.any(scales <= 0.0)
        or np.any(amplitudes < 0.0)
    ):
        raise ValueError("shape, positive correlation lengths, and nonnegative RMS amplitudes are required")
    velocity = np.zeros((2, height, width), dtype=np.float64)
    for scale, amplitude in zip(scales, amplitudes):
        noise = rng.standard_normal((2, height, width))
        smooth = gaussian_filter(noise, sigma=(0.0, float(scale), float(scale)), mode="reflect")
        smooth -= smooth.mean(axis=(1, 2), keepdims=True)
        rms = np.sqrt(np.mean(np.sum(smooth * smooth, axis=0)))
        if amplitude > 0.0:
            velocity += smooth * (float(amplitude) / max(float(rms), np.finfo(np.float64).eps))
    return velocity.astype(np.float32)


def _integration_steps(velocity_xy: np.ndarray, max_scaled_displacement_px: float) -> int:
    maximum = float(np.sqrt(np.sum(np.asarray(velocity_xy, dtype=np.float64) ** 2, axis=0)).max())
    if maximum == 0.0:
        return 0
    return max(0, int(math.ceil(math.log2(maximum / max_scaled_displacement_px))))


def _scaling_and_squaring_steps(velocity_xy: np.ndarray, steps: int) -> np.ndarray:
    velocity = np.asarray(velocity_xy, dtype=np.float32)
    identity = identity_pixel_map(velocity.shape[1:])
    displacement = velocity / float(2**steps)
    for _ in range(steps):
        displacement = displacement + bilinear_sample_field(
            displacement, identity + displacement, padding_mode="border"
        )
    return (identity + displacement).astype(np.float32)


def scaling_and_squaring(
    velocity_xy: np.ndarray,
    *,
    max_scaled_displacement_px: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Exponentiate an SVF after choosing enough squarings for a <=0.5 px seed."""
    velocity = np.asarray(velocity_xy, dtype=np.float32)
    if velocity.ndim != 3 or velocity.shape[0] != 2 or not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite with shape (2,H,W)")
    if (
        not np.isfinite(max_scaled_displacement_px)
        or max_scaled_displacement_px <= 0.0
        or max_scaled_displacement_px > 0.5
    ):
        raise ValueError("max_scaled_displacement_px must be in (0,0.5]")
    steps = _integration_steps(velocity, float(max_scaled_displacement_px))
    return _scaling_and_squaring_steps(velocity, steps), steps


def integrate_stationary_velocity(
    velocity_xy: np.ndarray,
    *,
    max_scaled_displacement_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return point maps ``exp(v)``, ``exp(-v)`` and their shared step count."""
    velocity = np.asarray(velocity_xy, dtype=np.float32)
    if velocity.ndim != 3 or velocity.shape[0] != 2 or not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite with shape (2,H,W)")
    if (
        not np.isfinite(max_scaled_displacement_px)
        or max_scaled_displacement_px <= 0.0
        or max_scaled_displacement_px > 0.5
    ):
        raise ValueError("max_scaled_displacement_px must be in (0,0.5]")
    steps = _integration_steps(velocity, float(max_scaled_displacement_px))
    return (
        _scaling_and_squaring_steps(velocity, steps),
        _scaling_and_squaring_steps(-velocity, steps),
        steps,
    )


def _apply_similarity(
    points_xy: np.ndarray,
    angle_rad: float,
    scale: float,
    translation_xy: np.ndarray,
    center_xy: np.ndarray,
) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64).reshape(2, -1)
    transformed = float(scale) * rotation @ (points - center_xy[:, None])
    transformed += center_xy[:, None] + translation_xy[:, None]
    return transformed.reshape(np.asarray(points_xy).shape).astype(np.float32)


def similarity_maps(
    shape: tuple[int, int],
    *,
    angle_rad: float,
    scale: float,
    translation_xy: tuple[float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and inverse positive-scale, no-reflection similarity maps."""
    height, width = (int(value) for value in shape)
    translation = np.asarray(translation_xy, dtype=np.float64)
    if (
        height < 2
        or width < 2
        or translation.shape != (2,)
        or not np.isfinite(translation).all()
        or not np.isfinite(angle_rad)
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("similarity needs an H,W > 1, finite parameters, and positive scale")
    identity = identity_pixel_map((height, width))
    center = np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    forward = _apply_similarity(identity, angle_rad, scale, translation, center)
    inverse = _apply_similarity(
        identity,
        -float(angle_rad),
        1.0 / float(scale),
        -(np.asarray(((math.cos(angle_rad), math.sin(angle_rad)),
                     (-math.sin(angle_rad), math.cos(angle_rad)))) @ translation) / float(scale),
        center,
    )
    return forward, inverse


def fixed_source_maps(
    velocity_uv_um: np.ndarray,
    pixel_basis_uv_um: np.ndarray,
    *,
    angle_rad: float,
    scale: float,
    translation_xy: tuple[float, float] | np.ndarray,
    max_scaled_displacement_px: float = 0.5,
) -> dict[str, np.ndarray | int]:
    """Construct exact synthetic point maps for a fixed atlas and source image.

    The local SVF acts in fixed coordinates.  ``fixed_to_source_map`` is
    ``similarity o exp(v)``.  ``source_to_fixed_map`` is
    ``exp(-v) o inverse(similarity)`` and is the pullback used to synthesize the
    source image from the fixed raster.
    """
    velocity_xy = physical_velocity_to_pixel(velocity_uv_um, pixel_basis_uv_um)
    local_forward, local_inverse, steps = integrate_stationary_velocity(
        velocity_xy, max_scaled_displacement_px=max_scaled_displacement_px
    )
    similarity_forward, similarity_inverse = similarity_maps(
        velocity_xy.shape[1:],
        angle_rad=angle_rad,
        scale=scale,
        translation_xy=translation_xy,
    )
    height, width = velocity_xy.shape[1:]
    center = np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    fixed_to_source = _apply_similarity(
        local_forward,
        angle_rad,
        scale,
        np.asarray(translation_xy, dtype=np.float64),
        center,
    )
    identity = identity_pixel_map((height, width))
    source_to_fixed = similarity_inverse + bilinear_sample_field(
        local_inverse - identity,
        similarity_inverse,
        padding_mode="border",
    )
    return {
        "velocity_xy_px": velocity_xy,
        "local_fixed_to_fixed_map": local_forward,
        "local_fixed_inverse_map": local_inverse,
        "similarity_fixed_to_source_map": similarity_forward,
        "similarity_source_to_fixed_map": similarity_inverse,
        "fixed_to_source_map": fixed_to_source.astype(np.float32),
        "source_to_fixed_map": source_to_fixed.astype(np.float32),
        "integration_steps": steps,
    }


def jacobian_determinant(pixel_map: np.ndarray) -> np.ndarray:
    """Return one forward-difference determinant for each raster cell."""
    values = np.asarray(pixel_map, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("pixel map must have shape (2,H,W)")
    derivative_x = values[:, :-1, 1:] - values[:, :-1, :-1]
    derivative_y = values[:, 1:, :-1] - values[:, :-1, :-1]
    return (
        derivative_x[0] * derivative_y[1] - derivative_x[1] * derivative_y[0]
    ).astype(np.float32)


def _inside_map(pixel_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        (pixel_map[0] >= 0.0)
        & (pixel_map[0] <= width - 1.0)
        & (pixel_map[1] >= 0.0)
        & (pixel_map[1] <= height - 1.0)
    )


def forward_inverse_cycle_metrics(
    fixed_to_source_map: np.ndarray,
    source_to_fixed_map: np.ndarray,
    *,
    fixed_valid_mask: np.ndarray | None = None,
    source_valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Measure both map cycles wherever the intermediate point stays in-frame."""
    forward = np.asarray(fixed_to_source_map, dtype=np.float32)
    inverse = np.asarray(source_to_fixed_map, dtype=np.float32)
    if forward.shape != inverse.shape or forward.ndim != 3 or forward.shape[0] != 2:
        raise ValueError("forward and inverse maps must share shape (2,H,W)")
    shape = forward.shape[1:]
    identity = identity_pixel_map(shape)
    fixed_valid = _inside_map(forward, shape)
    source_valid = _inside_map(inverse, shape)
    if fixed_valid_mask is not None:
        fixed_valid &= np.asarray(fixed_valid_mask, dtype=bool)
    if source_valid_mask is not None:
        source_valid &= np.asarray(source_valid_mask, dtype=bool)
    fixed_error = np.sqrt(
        np.sum((compose_pixel_maps(forward, inverse) - identity) ** 2, axis=0)
    )
    source_error = np.sqrt(
        np.sum((compose_pixel_maps(inverse, forward) - identity) ** 2, axis=0)
    )

    def summary(name: str, error: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
        selected = error[valid]
        if not len(selected):
            raise ValueError(f"{name} cycle has no valid pixels")
        return {
            f"{name}_cycle_valid_pixel_count": int(len(selected)),
            f"{name}_cycle_mean_px": float(selected.mean(dtype=np.float64)),
            f"{name}_cycle_rms_px": float(
                np.sqrt(np.mean(selected.astype(np.float64) ** 2))
            ),
            f"{name}_cycle_p95_px": float(np.quantile(selected, 0.95)),
            f"{name}_cycle_q99_px": float(np.quantile(selected, 0.99)),
            f"{name}_cycle_max_px": float(selected.max()),
        }

    return {
        **summary("fixed", fixed_error, fixed_valid),
        **summary("source", source_error, source_valid),
    }


def topology_acceptance_metrics(
    fixed_to_source_map: np.ndarray,
    source_to_fixed_map: np.ndarray,
    *,
    minimum_jacobian: float = 0.20,
    maximum_jacobian: float = 5.0,
    maximum_cycle_rms_px: float = 0.05,
    maximum_cycle_q99_px: float = 0.25,
    maximum_cycle_max_px: float = 0.50,
    fixed_cell_mask: np.ndarray | None = None,
    source_cell_mask: np.ndarray | None = None,
    fixed_valid_mask: np.ndarray | None = None,
    source_valid_mask: np.ndarray | None = None,
) -> dict[str, float | int | bool | None]:
    """Return deterministic two-way topology/cycle metrics and one acceptance bit."""
    forward_jacobian = jacobian_determinant(fixed_to_source_map)
    inverse_jacobian = jacobian_determinant(source_to_fixed_map)
    forward_cells = np.ones_like(forward_jacobian, dtype=bool)
    inverse_cells = np.ones_like(inverse_jacobian, dtype=bool)
    if fixed_cell_mask is not None:
        forward_cells &= np.asarray(fixed_cell_mask, dtype=bool)
    if source_cell_mask is not None:
        inverse_cells &= np.asarray(source_cell_mask, dtype=bool)
    if not forward_cells.any() or not inverse_cells.any():
        raise ValueError("topology metrics need at least one cell in each direction")
    selected_forward = forward_jacobian[forward_cells]
    selected_inverse = inverse_jacobian[inverse_cells]
    if not all(
        np.isfinite(value)
        for value in (
            minimum_jacobian,
            maximum_jacobian,
            maximum_cycle_rms_px,
            maximum_cycle_q99_px,
            maximum_cycle_max_px,
        )
    ) or not (
        0.0 < minimum_jacobian <= maximum_jacobian
        and maximum_cycle_rms_px >= 0.0
        and maximum_cycle_q99_px >= 0.0
        and maximum_cycle_max_px >= 0.0
    ):
        raise ValueError("topology and cycle gates must be finite, ordered, and nonnegative")

    def jacobian_summary(name: str, values: np.ndarray) -> dict[str, float | int | None]:
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            return {
                f"{name}_jacobian_min": None,
                f"{name}_jacobian_q01": None,
                f"{name}_jacobian_median": None,
                f"{name}_jacobian_q99": None,
                f"{name}_jacobian_max": None,
                f"{name}_nonpositive_jacobian_count": 0,
                f"{name}_nonfinite_jacobian_count": int(values.size),
            }
        return {
            f"{name}_jacobian_min": float(finite_values.min()),
            f"{name}_jacobian_q01": float(np.quantile(finite_values, 0.01)),
            f"{name}_jacobian_median": float(np.median(finite_values)),
            f"{name}_jacobian_q99": float(np.quantile(finite_values, 0.99)),
            f"{name}_jacobian_max": float(finite_values.max()),
            f"{name}_nonpositive_jacobian_count": int(
                np.count_nonzero(finite_values <= 0.0)
            ),
            f"{name}_nonfinite_jacobian_count": int(
                values.size - finite_values.size
            ),
        }

    forward_summary = jacobian_summary("forward", selected_forward)
    inverse_summary = jacobian_summary("inverse", selected_inverse)
    maps_finite = bool(
        np.isfinite(fixed_to_source_map).all()
        and np.isfinite(source_to_fixed_map).all()
    )
    cycles = (
        forward_inverse_cycle_metrics(
            fixed_to_source_map,
            source_to_fixed_map,
            fixed_valid_mask=fixed_valid_mask,
            source_valid_mask=source_valid_mask,
        )
        if maps_finite
        else {
            f"{domain}_cycle_{metric}": (0 if metric == "valid_pixel_count" else None)
            for domain in ("fixed", "source")
            for metric in (
                "valid_pixel_count",
                "mean_px",
                "rms_px",
                "p95_px",
                "q99_px",
                "max_px",
            )
        }
    )
    finite = bool(
        maps_finite
        and forward_summary["forward_nonfinite_jacobian_count"] == 0
        and inverse_summary["inverse_nonfinite_jacobian_count"] == 0
        and all(
            np.isfinite(value)
            for key, value in cycles.items()
            if key.endswith("_px")
        )
    )
    metrics: dict[str, float | int | bool | None] = {
        "finite": finite,
        "forward_cell_count": int(selected_forward.size),
        "inverse_cell_count": int(selected_inverse.size),
        **forward_summary,
        **inverse_summary,
        **cycles,
    }
    metrics["jacobian_min_passed"] = bool(
        finite
        and metrics["forward_jacobian_min"] >= minimum_jacobian
        and metrics["inverse_jacobian_min"] >= minimum_jacobian
    )
    metrics["jacobian_max_passed"] = bool(
        finite
        and metrics["forward_jacobian_max"] <= maximum_jacobian
        and metrics["inverse_jacobian_max"] <= maximum_jacobian
    )
    metrics["cycle_rms_passed"] = bool(
        finite
        and metrics["fixed_cycle_rms_px"] <= maximum_cycle_rms_px
        and metrics["source_cycle_rms_px"] <= maximum_cycle_rms_px
    )
    metrics["cycle_q99_passed"] = bool(
        finite
        and metrics["fixed_cycle_q99_px"] <= maximum_cycle_q99_px
        and metrics["source_cycle_q99_px"] <= maximum_cycle_q99_px
    )
    metrics["cycle_max_passed"] = bool(
        finite
        and metrics["fixed_cycle_max_px"] <= maximum_cycle_max_px
        and metrics["source_cycle_max_px"] <= maximum_cycle_max_px
    )
    metrics["accepted"] = bool(
        metrics["jacobian_min_passed"]
        and metrics["jacobian_max_passed"]
        and metrics["cycle_rms_passed"]
        and metrics["cycle_q99_passed"]
        and metrics["cycle_max_passed"]
    )
    return metrics


__all__ = [
    "ARBITRARY_PLANE_SYNTHETIC_OPS_VERSION",
    "bilinear_sample_field",
    "bilinear_sample_scalar",
    "compose_pixel_maps",
    "fixed_source_maps",
    "forward_inverse_cycle_metrics",
    "identity_pixel_map",
    "integrate_stationary_velocity",
    "jacobian_determinant",
    "nearest_sample_labels",
    "physical_velocity_to_pixel",
    "remove_tissue_affine_component",
    "sample_multiscale_physical_velocity",
    "scaling_and_squaring",
    "similarity_maps",
    "topology_acceptance_metrics",
]
