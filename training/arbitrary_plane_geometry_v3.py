"""Stable float32 geometry seam for the arbitrary-plane v3 generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition


GEOMETRY_V3_SCHEMA = "anatomy-tracker.global-reference-plane-geometry/v3"
GEOMETRY_V3_ALGORITHM = "canonical-renderer-matmul-float32-raster/v3"


def stable_global_reference_plane_geometry_v3(
    normal_ap_dv_ml: np.ndarray,
    signed_offset_um_about_support_origin: float,
    roll_rad: float,
    support_index: dict[str, object],
    parent_shape_h_w: tuple[int, int] = (256, 256),
) -> dict[str, object]:
    """Construct a fixed FOV using one canonical float32 renderer expression.

    The frozen v2 geometry compared two different float32 evaluation orders.  At
    Allen-scale coordinates they can differ by one or two ULPs even though they
    describe the same O/U/V plane.  V3 makes the renderer matmul raster canonical
    and keeps the expanded O/U/V expression as a receipt-bound diagnostic only.
    """
    acquisition.verify_annotation_support_index(support_index)
    normal, signed_offset, _ = acquisition.canonicalize_plane(
        normal_ap_dv_ml, signed_offset_um_about_support_origin
    )
    fov = acquisition.global_reference_support_geometry(
        support_index, parent_shape_h_w
    )
    height, width = fov["parent_shape_h_w"]
    support_origin = np.asarray(
        fov["support_origin_ap_dv_ml_um"], dtype=np.float64
    )
    intervals = acquisition.shifted_component_interval_union(normal, support_index)
    interval_array = np.asarray(intervals["support_origin_interval_union_um"])
    if not np.any(
        (signed_offset >= interval_array[:, 0])
        & (signed_offset <= interval_array[:, 1])
    ):
        raise ValueError("plane does not intersect the support-origin interval union")
    projection_offset = signed_offset - float(
        intervals["projection_to_support_origin_shift_um"]
    )
    membership = acquisition._json_value(
        acquisition.plane_interval_membership_certificate(
            normal, projection_offset, support_index
        )
    )
    if not membership["intersects"]:
        raise ValueError(
            "shifted plane failed the authenticated support membership certificate"
        )

    frame = acquisition.physical_plane_frame(normal, roll_rad)
    u, v = frame[:, 0], frame[:, 1]
    diameter = float(fov["diameter_um"])
    plane_center = support_origin + signed_offset * normal
    origin_physical = plane_center - 0.5 * diameter * u - 0.5 * diameter * v
    edge_u_physical = diameter * width / (width - 1.0) * u
    edge_v_physical = diameter * height / (height - 1.0) * v
    atlas_origin = tuple(support_index["origin_um"])
    spacing = tuple(support_index["voxel_size_um"])
    origin_index = acquisition.physical_um_to_allen_index_points(
        torch.as_tensor(origin_physical), atlas_origin, spacing
    ).numpy()
    edge_u_index = acquisition.physical_um_to_allen_index_vectors(
        torch.as_tensor(edge_u_physical), spacing
    ).numpy()
    edge_v_index = acquisition.physical_um_to_allen_index_vectors(
        torch.as_tensor(edge_v_physical), spacing
    ).numpy()
    atlas_shape = tuple(int(value) for value in support_index["annotation_shape"])
    quicknii_ouv = np.concatenate(
        (
            acquisition.allen_to_quicknii_points(
                torch.as_tensor(origin_index), atlas_shape
            ).numpy(),
            acquisition.allen_to_quicknii_vectors(
                torch.as_tensor(edge_u_index)
            ).numpy(),
            acquisition.allen_to_quicknii_vectors(
                torch.as_tensor(edge_v_index)
            ).numpy(),
        )
    )
    center64, frame64, basis64 = acquisition.quicknii_ouv_to_frame(
        torch.as_tensor(quicknii_ouv, dtype=torch.float64), atlas_shape
    )
    effective_center = center64.to(torch.float32)
    effective_frame = frame64.to(torch.float32)
    effective_basis = basis64.to(torch.float32)
    renderer_geometry = {
        "output_shape_h_w": [height, width],
        "renderer_center_ap_dv_ml": effective_center.tolist(),
        "renderer_frame_ap_dv_ml": effective_frame.tolist(),
        "renderer_inplane_basis": effective_basis.tolist(),
    }
    effective = acquisition.effective_renderer_sampling_arrays(
        renderer_geometry,
        atlas_shape,
        origin_ap_dv_ml_um=atlas_origin,
        voxel_size_ap_dv_ml_um=spacing,
    )
    if not all(
        acquisition._array_receipt(values.numpy())
        == acquisition._array_receipt(
            np.asarray(renderer_geometry[name], dtype=np.float32)
        )
        for name, values in (
            ("renderer_center_ap_dv_ml", effective_center),
            ("renderer_frame_ap_dv_ml", effective_frame),
            ("renderer_inplane_basis", effective_basis),
        )
    ):
        raise ValueError("v3 effective renderer state changed during serialization")

    s32 = torch.arange(width, dtype=torch.float32) / width
    t32 = torch.arange(height, dtype=torch.float32) / height
    tt32, ss32 = torch.meshgrid(t32, s32, indexing="ij")
    st32 = torch.stack((ss32, tt32), -1)
    edges32 = effective_frame[:, :2] @ effective_basis
    canonical_points = effective_center + torch.matmul(
        edges32, (st32 - 0.5).unsqueeze(-1)
    ).squeeze(-1)
    canonical_grid = torch.stack(
        (
            canonical_points[..., 2] / (atlas_shape[2] - 1) * 2 - 1,
            canonical_points[..., 1] / (atlas_shape[1] - 1) * 2 - 1,
            canonical_points[..., 0] / (atlas_shape[0] - 1) * 2 - 1,
        ),
        -1,
    )
    canonical_valid = torch.ones((height, width), dtype=torch.bool)
    rounded = torch.round(canonical_points).to(torch.int64)
    for axis, size in enumerate(atlas_shape):
        canonical_valid &= (rounded[..., axis] >= 0) & (rounded[..., axis] < size)
    canonical_receipts = {
        "coordinate_raster_allen_index_float32": acquisition._array_receipt(
            canonical_points.numpy()
        ),
        "normalized_interpolation_grid_xyz_float32": acquisition._array_receipt(
            canonical_grid.numpy()
        ),
        "valid_atlas_label_sampling_mask": acquisition._array_receipt(
            canonical_valid.numpy()
        ),
    }
    if any(
        canonical_receipts[name] != acquisition._array_receipt(effective[name])
        for name in canonical_receipts
    ):
        raise ValueError("v3 canonical raster is not byte-identical to the renderer")

    effective_origin = effective_center - 0.5 * edges32.sum(dim=1)
    effective_allen_ouv = torch.cat(
        (effective_origin, edges32[:, 0], edges32[:, 1])
    )
    effective_quicknii_ouv = torch.cat(
        (
            acquisition.allen_to_quicknii_points(
                effective_allen_ouv[:3], atlas_shape
            ),
            acquisition.allen_to_quicknii_vectors(effective_allen_ouv[3:6]),
            acquisition.allen_to_quicknii_vectors(effective_allen_ouv[6:9]),
        )
    )
    effective_allen_numpy = effective_allen_ouv.numpy().astype(np.float64)
    effective_physical_ouv = np.concatenate(
        (
            np.asarray(atlas_origin)
            + (effective_allen_numpy[:3] + 0.5) * np.asarray(spacing),
            effective_allen_numpy[3:6] * np.asarray(spacing),
            effective_allen_numpy[6:9] * np.asarray(spacing),
        )
    )
    effective_ouv = {
        "allen_index_ouv_ap_dv_ml_float32": effective_allen_ouv.numpy(),
        "quicknii_ouv_ml_ap_dv_float32": effective_quicknii_ouv.numpy(),
        "physical_ouv_ap_dv_ml_um_from_float32_state": effective_physical_ouv,
    }
    if any(
        acquisition._array_receipt(values)
        != acquisition._array_receipt(effective[name])
        for name, values in effective_ouv.items()
    ):
        raise ValueError("v3 effective O/U/V serialization changed renderer state")

    legacy_expanded_points = (
        effective_origin
        + ss32[..., None] * edges32[:, 0]
        + tt32[..., None] * edges32[:, 1]
    )
    legacy_expanded_grid = torch.stack(
        (
            legacy_expanded_points[..., 2] / (atlas_shape[2] - 1) * 2 - 1,
            legacy_expanded_points[..., 1] / (atlas_shape[1] - 1) * 2 - 1,
            legacy_expanded_points[..., 0] / (atlas_shape[0] - 1) * 2 - 1,
        ),
        -1,
    )
    legacy_max_index = float(
        torch.max(torch.abs(legacy_expanded_points - canonical_points))
    )
    legacy_max_normalized = float(
        torch.max(torch.abs(legacy_expanded_grid - canonical_grid))
    )

    x = np.arange(width, dtype=np.float64) / width
    y = np.arange(height, dtype=np.float64) / height
    design_physical_grid = (
        origin_physical[None, None]
        + x[None, :, None] * edge_u_physical[None, None]
        + y[:, None, None] * edge_v_physical[None, None]
    )
    design_index_grid = (
        origin_index[None, None]
        + x[None, :, None] * edge_u_index[None, None]
        + y[:, None, None] * edge_v_index[None, None]
    )
    reconstructed_physical = np.asarray(atlas_origin) + (
        design_index_grid + 0.5
    ) * np.asarray(spacing)
    lower = np.asarray(fov["closed_face_lower_ap_dv_ml_um"])
    upper = np.asarray(fov["closed_face_upper_ap_dv_ml_um"])
    corners = np.stack(
        np.meshgrid(
            *[(lower[axis], upper[axis]) for axis in range(3)], indexing="ij"
        ),
        -1,
    ).reshape(-1, 3)
    projected_u = (corners - plane_center) @ u
    projected_v = (corners - plane_center) @ v
    clearance = min(
        float(projected_u.min() + diameter / 2.0),
        float(diameter / 2.0 - projected_u.max()),
        float(projected_v.min() + diameter / 2.0),
        float(diameter / 2.0 - projected_v.max()),
    )
    arrays = {
        "design_physical_coordinate_raster_ap_dv_ml_um_float64": design_physical_grid,
        "design_allen_index_coordinate_raster_float64": design_index_grid,
        **effective,
        "independent_ouv_parameterized_coordinate_raster_float32": canonical_points.numpy(),
        "independent_ouv_parameterized_normalized_grid_float32": canonical_grid.numpy(),
    }
    array_receipts = {
        name: acquisition._array_receipt(value) for name, value in arrays.items()
    }
    diagnostics = {
        "normal_norm_error": abs(float(np.linalg.norm(normal)) - 1.0),
        "frame_orthogonality_max_abs": float(
            np.max(np.abs(frame.T @ frame - np.eye(3)))
        ),
        "frame_determinant_error": abs(float(np.linalg.det(frame)) - 1.0),
        "physical_index_roundtrip_max_abs_um": float(
            np.max(np.abs(reconstructed_physical - design_physical_grid))
        ),
        "plane_residual_max_abs_um": float(
            np.max(
                np.abs(
                    (design_physical_grid - support_origin) @ normal
                    - signed_offset
                )
            )
        ),
        "support_corner_minimum_inplane_clearance_um": clearance,
        "canonical_effective_grid_byte_equal": True,
        "canonical_effective_ouv_byte_equal": True,
        "legacy_expanded_ouv_max_abs_index_diagnostic_only": legacy_max_index,
        "legacy_expanded_ouv_max_abs_normalized_diagnostic_only": legacy_max_normalized,
    }
    if (
        diagnostics["normal_norm_error"] > 1e-12
        or diagnostics["frame_orthogonality_max_abs"] > 1e-12
        or diagnostics["frame_determinant_error"] > 1e-12
        or diagnostics["physical_index_roundtrip_max_abs_um"] > 1e-9
        or diagnostics["plane_residual_max_abs_um"] > 1e-9
        or clearance < float(fov["margin_um"]) - 1e-9
    ):
        raise ValueError("v3 global-reference geometry failed its numerical gates")

    geometry_contract = {
        "schema_version": GEOMETRY_V3_SCHEMA,
        "algorithm": GEOMETRY_V3_ALGORITHM,
        "implementation_source_sha256": acquisition._normalized_text_sha256(
            Path(__file__)
        ),
        "canonical_float32_coordinate_raster": (
            "renderer center + matmul(renderer edges, [s-0.5,t-0.5])"
        ),
        "expanded_ouv_expression_role": "receipt-bound diagnostic only",
        "learned_dependencies": [],
    }
    physical_ouv = np.concatenate(
        (origin_physical, edge_u_physical, edge_v_physical)
    )
    allen_ouv = np.concatenate((origin_index, edge_u_index, edge_v_index))
    payload = {
        **renderer_geometry,
        "geometry_contract_v3": geometry_contract,
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "signed_offset_um_about_support_origin": signed_offset,
        "roll_rad": float(roll_rad),
        "frame_ap_dv_ml_physical": frame.tolist(),
        "plane_center_ap_dv_ml_um": plane_center.tolist(),
        "physical_ouv_ap_dv_ml_um": physical_ouv.tolist(),
        "allen_index_ouv_ap_dv_ml": allen_ouv.tolist(),
        "quicknii_ouv_ml_ap_dv": quicknii_ouv.tolist(),
        "pose_state": {
            "center_ap_dv_ml_um": plane_center.tolist(),
            "proper_frame_ap_dv_ml": frame.tolist(),
            "positive_inplane_basis_um": [
                [float(np.linalg.norm(edge_u_physical)), 0.0],
                [0.0, float(np.linalg.norm(edge_v_physical))],
            ],
        },
        "global_reference_fov": fov,
        "shifted_intervals": intervals,
        "projection_origin_membership_certificate": membership,
        "array_receipts": array_receipts,
        "diagnostics": diagnostics,
        "raster_endpoint_semantics": {
            "pixel_mapping": "P(x,y)=O+(x/W)U+(y/H)V",
            "first_sample_ap_dv_ml_um": design_physical_grid[0, 0].tolist(),
            "last_sample_ap_dv_ml_um": design_physical_grid[-1, -1].tolist(),
            "u_edge_factor": width / (width - 1.0),
            "v_edge_factor": height / (height - 1.0),
        },
    }
    return {
        **payload,
        "global_reference_grid_id": acquisition._payload_sha256(payload),
    }
