import numpy as np
import pytest

import training.arbitrary_plane_deformation_gauge_v3 as gauge


def test_asymmetric_canvas_affine_factor_recomposes_pose_and_residual_map():
    height, width = 37, 61
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    raw_residual = np.stack(
        (0.35 * np.sin(2.0 * x) * np.cos(y), 0.25 * np.sin(1.5 * y) * np.cos(x)),
        axis=-1,
    )
    residual, _, _ = gauge.uniform_canvas_affine_projection_yx_v3(raw_residual)
    removed = np.array(
        [[0.8, 0.025, -0.018], [-1.1, 0.014, -0.021]], dtype=np.float64
    )
    basis = np.stack((np.ones_like(y), y, x))
    affine_velocity = np.einsum("cp,phw->hwc", removed, basis)
    velocity = residual + affine_velocity
    affine_flow = gauge.affine_velocity_flow_xy_v3(removed, (height, width))
    residual_map = gauge.integrate_stationary_velocity_yx_v3(residual)
    original_map = gauge.apply_canvas_affine_to_map_yx_v3(
        residual_map, affine_flow
    )
    ouv = np.array(
        [[220.0, 310.0, 140.0], [410.0, -25.0, 30.0], [18.0, 290.0, -340.0]],
        dtype=np.float64,
    )
    result = gauge.gauge_fix_canvas_deformation_v3(
        velocity, original_map, ouv, np.ones((height, width), dtype=bool)
    )
    arrays = result["arrays"]
    assert np.allclose(
        arrays["removed_affine_coefficients_yx_float64"], removed, atol=1e-13
    )
    assert (
        np.abs(arrays["postprojection_affine_coefficients_yx_float64"]).max()
        < 1e-14
    )
    assert np.allclose(
        arrays["affine_free_stationary_velocity_yx_px_float64"], residual, atol=1e-13
    )
    assert np.allclose(
        arrays["pose_then_deformation_recomposed_pullback_map_yx_px_float64"],
        original_map,
        atol=2e-12,
    )

    adjusted = arrays["pose_adjusted_effective_quicknii_ouv_float64"]
    points_xy = np.array([[0.0, 0.0], [13.2, 7.5], [60.0, 36.0]])
    transformed = (
        points_xy @ affine_flow[:2, :2].T + affine_flow[:2, 2]
    )
    original_ccf = (
        ouv[0]
        + transformed[:, :1] / width * ouv[1]
        + transformed[:, 1:] / height * ouv[2]
    )
    adjusted_ccf = (
        adjusted[0]
        + points_xy[:, :1] / width * adjusted[1]
        + points_xy[:, 1:] / height * adjusted[2]
    )
    assert np.allclose(adjusted_ccf, original_ccf, atol=2e-12)
    assert result["receipt_sha256"] == gauge.acquisition._payload_sha256(
        gauge.deformation_pose_gauge_receipt_v3(result)
    )


def test_asymmetric_crop_of_parent_affine_free_field_is_reprojected_on_canvas():
    parent_height, parent_width = 81, 97
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, parent_height),
        np.linspace(-1.0, 1.0, parent_width),
        indexing="ij",
    )
    raw = np.stack(
        (
            0.7 * np.sin(2.1 * x) * np.cos(1.3 * y) + 0.15 * np.sin(3.0 * y),
            0.55 * np.sin(1.7 * y) * np.cos(1.2 * x) + 0.1 * np.sin(2.0 * x),
        ),
        axis=-1,
    )
    parent, _, _ = gauge.uniform_canvas_affine_projection_yx_v3(raw)
    cropped = np.ascontiguousarray(parent[7:70, 18:76])
    original_map = gauge.integrate_stationary_velocity_yx_v3(cropped)
    ouv = np.array(
        [[220.0, 310.0, 140.0], [410.0, -25.0, 30.0], [18.0, 290.0, -340.0]],
        dtype=np.float64,
    )
    result = gauge.gauge_fix_canvas_deformation_v3(
        cropped, original_map, ouv, np.ones(cropped.shape[:2], dtype=bool)
    )
    arrays = result["arrays"]
    assert (
        np.abs(arrays["removed_affine_coefficients_yx_float64"]).max() > 0.1
    )
    assert (
        np.abs(arrays["postprojection_affine_coefficients_yx_float64"]).max()
        < 1e-14
    )
    assert result["diagnostics"]["valid_recomposition_error_max_px"] < 0.006


def test_noncommuting_recomposition_above_bound_is_rejected_and_bound_is_receipted():
    height, width = 81, 97
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    raw = np.stack(
        (
            0.7 * np.sin(2.1 * x) * np.cos(1.3 * y) + 0.15 * np.sin(3.0 * y),
            0.55 * np.sin(1.7 * y) * np.cos(1.2 * x) + 0.1 * np.sin(2.0 * x),
        ),
        axis=-1,
    )
    parent, _, _ = gauge.uniform_canvas_affine_projection_yx_v3(raw)
    cropped = np.ascontiguousarray(4.0 * parent[7:70, 18:76])
    original_map = gauge.integrate_stationary_velocity_yx_v3(cropped)
    ouv = np.array(
        [[220.0, 310.0, 140.0], [410.0, -25.0, 30.0], [18.0, 290.0, -340.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="production bound"):
        gauge.gauge_fix_canvas_deformation_v3(
            cropped,
            original_map,
            ouv,
            np.ones(cropped.shape[:2], dtype=bool),
        )
    relaxed = gauge.gauge_fix_canvas_deformation_v3(
        cropped,
        original_map,
        ouv,
        np.ones(cropped.shape[:2], dtype=bool),
        maximum_valid_recomposition_error_px=0.1,
    )
    assert relaxed["maximum_valid_recomposition_error_px"] == 0.1
    tampered = {**relaxed, "maximum_valid_recomposition_error_px": 0.2}
    assert tampered["receipt_sha256"] != gauge.acquisition._payload_sha256(
        gauge.deformation_pose_gauge_receipt_v3(tampered)
    )
