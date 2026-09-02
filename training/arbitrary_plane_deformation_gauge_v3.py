"""Uniform-canvas affine gauge for v3 pose/deformation targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import linalg, ndimage

import training.arbitrary_plane_acquisition_v2 as acquisition


DEFORMATION_GAUGE_V3_SCHEMA = "anatomy-tracker.deformation-pose-gauge/v3"
DEFORMATION_GAUGE_V3_ALGORITHM = (
    "uniform-canvas-affine-svf-projection-and-pose-recomposition/v3"
)
MAXIMUM_VALID_RECOMPOSITION_ERROR_PX = 0.05
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_deformation_gauge_v3.py",
    "arbitrary_plane_deformation_primitives.py",
)


def _source_hashes():
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def deformation_pose_gauge_receipt_v3(artifact):
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "implementation_source_sha256": artifact[
            "implementation_source_sha256"
        ],
        "projection_weighting": artifact["projection_weighting"],
        "integration_steps": artifact["integration_steps"],
        "maximum_valid_recomposition_error_px": artifact[
            "maximum_valid_recomposition_error_px"
        ],
        "input_array_receipts": artifact["input_array_receipts"],
        "array_receipts": artifact["array_receipts"],
        "diagnostics": artifact["diagnostics"],
        "deformation_pose_gauge_id": artifact["deformation_pose_gauge_id"],
    }


def deformation_pose_gauge_summary_v3(artifact):
    arrays = artifact["arrays"]
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "implementation_source_sha256": artifact[
            "implementation_source_sha256"
        ],
        "projection_weighting": artifact["projection_weighting"],
        "integration_steps": artifact["integration_steps"],
        "maximum_valid_recomposition_error_px": artifact[
            "maximum_valid_recomposition_error_px"
        ],
        "input_array_receipts": artifact["input_array_receipts"],
        "removed_affine_coefficients_yx_float64": arrays[
            "removed_affine_coefficients_yx_float64"
        ].tolist(),
        "postprojection_affine_coefficients_yx_float64": arrays[
            "postprojection_affine_coefficients_yx_float64"
        ].tolist(),
        "removed_affine_flow_xy_float64": arrays[
            "removed_affine_flow_xy_float64"
        ].tolist(),
        "pose_adjusted_effective_quicknii_ouv_float64": arrays[
            "pose_adjusted_effective_quicknii_ouv_float64"
        ].tolist(),
        "array_receipts": artifact["array_receipts"],
        "diagnostics": artifact["diagnostics"],
        "deformation_pose_gauge_id": artifact["deformation_pose_gauge_id"],
        "receipt_sha256": artifact["receipt_sha256"],
    }


def deformation_pose_gauge_reference_v3(artifact):
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "projection_weighting": artifact["projection_weighting"],
        "deformation_pose_gauge_id": artifact["deformation_pose_gauge_id"],
        "receipt_sha256": artifact["receipt_sha256"],
    }


def uniform_canvas_affine_projection_yx_v3(velocity_yx):
    velocity = np.asarray(velocity_yx, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape H,W,2 in y-x order")
    height, width = velocity.shape[:2]
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    basis = np.stack((np.ones_like(y), y, x))
    gram = np.einsum("phw,qhw->pq", basis, basis) / (height * width)
    right = np.einsum("hwc,phw->cp", velocity, basis) / (height * width)
    coefficients = np.linalg.solve(gram, right.T).T
    fitted = np.einsum("cp,phw->hwc", coefficients, basis)
    residual = velocity - fitted
    post_right = np.einsum("hwc,phw->cp", residual, basis) / (height * width)
    post = np.linalg.solve(gram, post_right.T).T
    return (
        np.ascontiguousarray(residual),
        np.ascontiguousarray(coefficients),
        np.ascontiguousarray(post),
    )


def affine_velocity_flow_xy_v3(coefficients_yx, shape_h_w):
    coefficients = np.asarray(coefficients_yx, dtype=np.float64)
    if coefficients.shape != (2, 3):
        raise ValueError("affine velocity coefficients must have shape 2,3")
    height, width = shape_h_w
    cy, ay_y, ay_x = coefficients[0]
    cx, ax_y, ax_x = coefficients[1]
    generator = np.array(
        [
            [2.0 * ax_x / (width - 1), 2.0 * ax_y / (height - 1), cx - ax_y - ax_x],
            [2.0 * ay_x / (width - 1), 2.0 * ay_y / (height - 1), cy - ay_y - ay_x],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    flow = linalg.expm(generator)
    flow[2] = (0.0, 0.0, 1.0)
    return np.ascontiguousarray(flow)


def integrate_stationary_velocity_yx_v3(velocity_yx, steps=7):
    velocity = np.asarray(velocity_yx, dtype=np.float64)
    height, width = velocity.shape[:2]
    y, x = np.indices((height, width), dtype=np.float64)
    displacement = np.moveaxis(velocity, -1, 0) / float(2**steps)
    for _ in range(steps):
        coordinates = (y + displacement[0], x + displacement[1])
        sampled = np.stack(
            (
                ndimage.map_coordinates(
                    displacement[0], coordinates, order=1, mode="nearest", prefilter=False
                ),
                ndimage.map_coordinates(
                    displacement[1], coordinates, order=1, mode="nearest", prefilter=False
                ),
            )
        )
        displacement += sampled
    return np.ascontiguousarray(
        np.stack((y + displacement[0], x + displacement[1]), axis=-1)
    )


def apply_canvas_affine_to_map_yx_v3(map_yx, affine_xy):
    mapping = np.asarray(map_yx, dtype=np.float64)
    affine = np.asarray(affine_xy, dtype=np.float64)
    xy = mapping[..., ::-1]
    transformed = xy @ affine[:2, :2].T + affine[:2, 2]
    return np.ascontiguousarray(transformed[..., ::-1])


def compose_quicknii_ouv_with_canvas_affine_v3(
    quicknii_ouv, affine_xy, shape_h_w
):
    ouv = np.asarray(quicknii_ouv, dtype=np.float64).reshape(3, 3)
    affine = np.asarray(affine_xy, dtype=np.float64)
    height, width = shape_h_w

    def map_to_quicknii(point_xy):
        transformed = affine @ np.asarray((*point_xy, 1.0), dtype=np.float64)
        return ouv[0] + transformed[0] / width * ouv[1] + transformed[1] / height * ouv[2]

    origin = map_to_quicknii((0.0, 0.0))
    edge_u = map_to_quicknii((float(width), 0.0)) - origin
    edge_v = map_to_quicknii((0.0, float(height))) - origin
    return np.ascontiguousarray(np.stack((origin, edge_u, edge_v)))


def gauge_fix_canvas_deformation_v3(
    stationary_velocity_yx,
    original_pullback_map_yx,
    effective_quicknii_ouv,
    valid_mask,
    *,
    integration_steps=7,
    maximum_valid_recomposition_error_px=MAXIMUM_VALID_RECOMPOSITION_ERROR_PX,
):
    velocity = np.asarray(stationary_velocity_yx, dtype=np.float64)
    original_map = np.asarray(original_pullback_map_yx, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if original_map.shape != velocity.shape or valid.shape != velocity.shape[:2]:
        raise ValueError("velocity, pullback map, and validity shapes disagree")
    residual, removed, post = uniform_canvas_affine_projection_yx_v3(velocity)
    affine_flow = affine_velocity_flow_xy_v3(removed, velocity.shape[:2])
    residual_map = integrate_stationary_velocity_yx_v3(
        residual, steps=integration_steps
    )
    recomposed = apply_canvas_affine_to_map_yx_v3(residual_map, affine_flow)
    adjusted_ouv = compose_quicknii_ouv_with_canvas_affine_v3(
        effective_quicknii_ouv, affine_flow, velocity.shape[:2]
    )
    error = np.linalg.norm(recomposed - original_map, axis=-1)
    finite_valid = valid & np.isfinite(error)
    _, after_float32, _ = uniform_canvas_affine_projection_yx_v3(
        residual.astype(np.float32)
    )
    valid_error_mean = float(error[finite_valid].mean()) if finite_valid.any() else 0.0
    valid_error_max = float(error[finite_valid].max()) if finite_valid.any() else 0.0
    maximum_valid_recomposition_error_px = float(
        maximum_valid_recomposition_error_px
    )
    if (
        not np.isfinite(maximum_valid_recomposition_error_px)
        or maximum_valid_recomposition_error_px <= 0.0
        or not np.isfinite(valid_error_max)
        or valid_error_max > maximum_valid_recomposition_error_px
    ):
        raise ValueError(
            "affine-gauge pose/deformation recomposition exceeds the production bound"
        )
    diagnostics = {
        "uniform_canvas_affine_coefficient_max_abs": float(np.abs(removed).max()),
        "postprojection_affine_coefficient_max_abs": float(np.abs(post).max()),
        "postprojection_after_float32_coefficient_max_abs": float(
            np.abs(after_float32).max()
        ),
        "valid_recomposition_error_mean_px": valid_error_mean,
        "valid_recomposition_error_max_px": valid_error_max,
    }
    arrays = {
        "removed_affine_coefficients_yx_float64": np.ascontiguousarray(removed),
        "postprojection_affine_coefficients_yx_float64": np.ascontiguousarray(post),
        "removed_affine_flow_xy_float64": np.ascontiguousarray(affine_flow),
        "pose_adjusted_effective_quicknii_ouv_float64": np.ascontiguousarray(
            adjusted_ouv
        ),
        "affine_free_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            residual
        ),
        "affine_free_pullback_map_yx_px_float64": np.ascontiguousarray(
            residual_map
        ),
        "pose_then_deformation_recomposed_pullback_map_yx_px_float64": (
            np.ascontiguousarray(recomposed)
        ),
    }
    artifact = {
        "schema_version": DEFORMATION_GAUGE_V3_SCHEMA,
        "algorithm": DEFORMATION_GAUGE_V3_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "projection_weighting": "fixed uniform full canvas, matching decoder gauge",
        "integration_steps": int(integration_steps),
        "maximum_valid_recomposition_error_px": (
            maximum_valid_recomposition_error_px
        ),
        "input_array_receipts": {
            "stationary_velocity_yx_px_float64": acquisition._array_receipt(
                velocity
            ),
            "original_pullback_map_yx_px_float64": acquisition._array_receipt(
                original_map
            ),
            "effective_quicknii_ouv_float64": acquisition._array_receipt(
                np.asarray(effective_quicknii_ouv, dtype=np.float64)
            ),
            "valid_mask": acquisition._array_receipt(valid),
        },
        "arrays": arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in arrays.items()
        },
        "diagnostics": diagnostics,
    }
    artifact["deformation_pose_gauge_id"] = acquisition._payload_sha256(
        {
            "domain": DEFORMATION_GAUGE_V3_SCHEMA,
            "implementation_source_sha256": artifact[
                "implementation_source_sha256"
            ],
            "projection_weighting": artifact["projection_weighting"],
            "integration_steps": artifact["integration_steps"],
            "maximum_valid_recomposition_error_px": artifact[
                "maximum_valid_recomposition_error_px"
            ],
            "input_array_receipts": artifact["input_array_receipts"],
            "array_receipts": artifact["array_receipts"],
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        deformation_pose_gauge_receipt_v3(artifact)
    )
    return artifact
