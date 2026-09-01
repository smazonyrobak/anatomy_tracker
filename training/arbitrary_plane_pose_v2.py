"""Deterministic finite-plane truth derived from a final v2 realization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_realization_v2 as realization_v2
from training.arbitrary_plane_geometry import (
    allen_to_quicknii_points,
    allen_to_quicknii_vectors,
    crop_quicknii_ouv,
    frame_to_physical_ouv,
    frame_to_rotation_6d,
    horizontal_flip_quicknii_ouv,
    inplane_basis_to_parameters,
    physical_ouv_to_frame,
    physical_um_to_allen_index_points,
    physical_um_to_allen_index_vectors,
    positive_inplane_basis,
    rotation_6d_to_frame,
    vertical_flip_quicknii_ouv,
)
from training.arbitrary_plane_manifest import canonicalize_plane


FINITE_PLANE_POSE_TRUTH_V2_SCHEMA = "anatomy-tracker.finite-plane-pose-truth/v2"
FINITE_PLANE_POSE_TRUTH_V2_ALGORITHM = (
    "physical-ouv-proper-frame-basis-coupled-rp2-plane-and-raster-reflection/v2"
)
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_pose_v2.py",
    "arbitrary_plane_geometry.py",
    "arbitrary_plane_realization_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_manifest.py",
)
_OUV_KEYS = (
    "full_raster_best_fit_physical_ouv_ap_dv_ml_um_float64",
    "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64",
    "model_raster_physical_ouv_ap_dv_ml_um_float64",
)
_ARRAY_KEYS = {
    *_OUV_KEYS,
    "full_raster_quicknii_ouv_ml_ap_dv_float64",
    "cropped_pre_reflection_quicknii_ouv_ml_ap_dv_float64",
    "model_raster_quicknii_ouv_ml_ap_dv_float64",
    "support_origin_ap_dv_ml_um_float64",
    "full_raster_physical_center_ap_dv_ml_um_float64",
    "cropped_pre_reflection_physical_center_ap_dv_ml_um_float64",
    "proper_frame_ap_dv_ml_float64",
    "rotation_6d_ap_dv_ml_float64",
    "positive_upper_triangular_basis_um_float64",
    "log_positive_basis_diagonal_um_float64",
    "basis_shear_dimensionless_float64",
    "actual_plane_normal_and_signed_offset_um_float64",
    "canonical_plane_normal_and_signed_offset_um_float64",
    "actual_tangent_gauge_ap_dv_ml_float64",
    "canonical_tangent_gauge_ap_dv_ml_float64",
    "actual_to_canonical_tangent_offset_transport_float64",
    "canonical_to_actual_tangent_offset_transport_float64",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _array_receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {name: acquisition._array_receipt(value) for name, value in arrays.items()}


def _ouv_matrix(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("physical O/U/V must be a finite float64 3-by-3 array")
    edge_u, edge_v = array[1], array[2]
    scale = max(float(np.linalg.norm(edge_u)), float(np.linalg.norm(edge_v)), 1.0)
    if float(np.linalg.norm(np.cross(edge_u, edge_v))) <= 1e-12 * scale * scale:
        raise ValueError("physical O/U/V edges must be non-collinear")
    return np.ascontiguousarray(array)


def deterministic_rp2_tangent_gauge_v2(normal_ap_dv_ml: np.ndarray) -> np.ndarray:
    """Choose a deterministic right-handed tangent basis at a unit normal."""
    normal = np.array(normal_ap_dv_ml, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not np.isfinite(normal).all() or norm == 0.0:
        raise ValueError("normal must be one finite nonzero three-vector")
    normal = normal / norm
    anchor = np.zeros(3, dtype=np.float64)
    anchor[int(np.argmin(np.abs(normal)))] = 1.0
    first = anchor - float(anchor @ normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.ascontiguousarray(np.stack((first, second), axis=-1))


def antipodal_tangent_offset_transport_v2(
    source_normal_ap_dv_ml: np.ndarray,
    target_normal_ap_dv_ml: np.ndarray,
) -> np.ndarray:
    """Transport two tangent coordinates plus offset across an RP2 sign choice."""
    source = np.array(source_normal_ap_dv_ml, dtype=np.float64, copy=True)
    target = np.array(target_normal_ap_dv_ml, dtype=np.float64, copy=True)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if (
        source.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or source_norm == 0.0
        or target_norm == 0.0
    ):
        raise ValueError("source and target normals must be three-vectors")
    source /= source_norm
    target /= target_norm
    sign = 1.0 if float(source @ target) >= 0.0 else -1.0
    if not np.allclose(target, sign * source, rtol=0.0, atol=2e-12):
        raise ValueError("target normal must equal source normal or its antipode")
    source_gauge = deterministic_rp2_tangent_gauge_v2(source)
    target_gauge = deterministic_rp2_tangent_gauge_v2(target)
    result = np.zeros((3, 3), dtype=np.float64)
    result[:2, :2] = target_gauge.T @ (sign * source_gauge)
    result[2, 2] = sign
    return np.ascontiguousarray(result)


def _count_synthetic_realization_ids(value: object) -> int:
    if isinstance(value, Mapping):
        return int("synthetic_realization_id" in value) + sum(
            _count_synthetic_realization_ids(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_synthetic_realization_ids(item) for item in value)
    return 0


def _verify_final_reference(
    final_realization: Mapping[str, object], prepared_context: Mapping[str, object]
) -> None:
    acquisition._validate_v2_context(prepared_context)
    realization_v2._strict_structure(final_realization)
    frame = final_realization["frame_transform"]
    frame_arrays = frame["arrays"]
    prepared_receipt = acquisition._json_value(prepared_context["receipt"])
    prepared_binding = final_realization["upstream_reference"]["live_receipt_bindings"][
        "prepared_context"
    ]
    expected_frame_id = acquisition._payload_sha256(
        {
            key: value
            for key, value in frame.items()
            if key not in {"arrays", "frame_transform_id"}
        }
    )
    if (
        final_realization["schema_version"] != realization_v2.SYNTHETIC_REALIZATION_V2_SCHEMA
        or acquisition._json_value(final_realization["implementation_source_sha256"])
        != realization_v2._source_hashes()
        or final_realization["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or final_realization["upstream_reference"]["v2_context_sha256"]
        != prepared_context["v2_context_sha256"]
        or acquisition._json_value(prepared_binding["receipt_payload"]) != prepared_receipt
        or prepared_binding["receipt_sha256"] != acquisition._payload_sha256(prepared_receipt)
        or acquisition._json_value(frame["array_receipts"])
        != acquisition._json_value(_array_receipts(frame_arrays))
        or frame["frame_transform_id"] != expected_frame_id
        or final_realization["synthetic_realization_id"]
        != acquisition._payload_sha256(realization_v2._identity_payload(final_realization))
        or final_realization["receipt_sha256"]
        != acquisition._payload_sha256(
            realization_v2.synthetic_realization_receipt_v2(final_realization)
        )
        or _count_synthetic_realization_ids(final_realization) != 1
        or any(final_realization["asset_dependencies"].get(name) for name in (
            "learned_checkpoint_dependencies",
            "pretrained_feature_dependencies",
            "previous_model_dependencies",
        ))
    ):
        raise ValueError("final realization or prepared-context binding changed")
    for name in _OUV_KEYS:
        _ouv_matrix(frame_arrays[name])


def _factor_physical_ouv(ouv: np.ndarray):
    tensor = torch.from_numpy(np.ascontiguousarray(ouv.reshape(9)))
    center, frame, basis = physical_ouv_to_frame(tensor)
    rotation = frame_to_rotation_6d(frame)
    log_diagonal, shear = inplane_basis_to_parameters(basis)
    return tuple(
        np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float64)
        for value in (center, frame, basis, rotation, log_diagonal, shear)
    )


def _physical_to_quicknii_ouv(
    ouv: np.ndarray,
    origin_ap_dv_ml_um: np.ndarray,
    voxel_size_ap_dv_ml_um: np.ndarray,
    atlas_shape_ap_dv_ml: tuple[int, int, int],
) -> np.ndarray:
    values = torch.from_numpy(np.ascontiguousarray(ouv))
    origin = physical_um_to_allen_index_points(
        values[0], origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um
    )
    edges = physical_um_to_allen_index_vectors(
        values[1:], voxel_size_ap_dv_ml_um
    )
    quicknii = torch.stack(
        (
            allen_to_quicknii_points(origin, atlas_shape_ap_dv_ml),
            allen_to_quicknii_vectors(edges[0]),
            allen_to_quicknii_vectors(edges[1]),
        )
    )
    return np.ascontiguousarray(quicknii.detach().cpu().numpy(), dtype=np.float64)


def _expected_crop_and_reflection(
    parent_ouv: np.ndarray, frame_transform: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    parent = torch.from_numpy(np.ascontiguousarray(parent_ouv.reshape(9)))
    cropped = crop_quicknii_ouv(
        parent,
        tuple(frame_transform["parent_shape_h_w"]),
        tuple(frame_transform["top_left_y_x"]),
        tuple(frame_transform["output_shape_h_w"]),
    )
    model = cropped
    height, width = frame_transform["output_shape_h_w"]
    if frame_transform["horizontal_reflection"]:
        model = horizontal_flip_quicknii_ouv(model, width)
    if frame_transform["vertical_reflection"]:
        model = vertical_flip_quicknii_ouv(model, height)
    return tuple(
        np.ascontiguousarray(value.detach().cpu().numpy().reshape(3, 3), dtype=np.float64)
        for value in (cropped, model)
    )


def _identity_payload(pose_truth: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value({
        key: value
        for key, value in pose_truth.items()
        if key not in {"arrays", "finite_plane_pose_truth_id", "receipt_sha256"}
    })


def finite_plane_pose_truth_receipt_v2(
    pose_truth: Mapping[str, object],
) -> dict[str, object]:
    return {
        "finite_plane_pose_truth_id": pose_truth["finite_plane_pose_truth_id"],
        "identity_payload": _identity_payload(pose_truth),
    }


def _build_pose_truth(
    final_realization: Mapping[str, object], prepared_context: Mapping[str, object]
) -> Mapping[str, object]:
    transform = final_realization["frame_transform"]
    source_arrays = transform["arrays"]
    parent_ouv, cropped_ouv, model_ouv = (
        _ouv_matrix(source_arrays[name]) for name in _OUV_KEYS
    )
    expected_crop, expected_model = _expected_crop_and_reflection(parent_ouv, transform)
    scale = max(float(np.abs(parent_ouv).max()), 1.0)
    if (
        not np.allclose(cropped_ouv, expected_crop, rtol=2e-13, atol=2e-13 * scale)
        or not np.allclose(model_ouv, expected_model, rtol=2e-13, atol=2e-13 * scale)
    ):
        raise ValueError("final realization crop/reflection O/U/V algebra changed")

    full_center, full_frame, _, _, _, _ = _factor_physical_ouv(parent_ouv)
    center, frame, basis, rotation, log_diagonal, shear = _factor_physical_ouv(cropped_ouv)
    model_center, model_frame, _, _, _, _ = _factor_physical_ouv(model_ouv)
    reconstructed = frame_to_physical_ouv(
        torch.from_numpy(center), torch.from_numpy(frame), torch.from_numpy(basis)
    ).detach().numpy().reshape(3, 3)
    if (
        not np.allclose(reconstructed, cropped_ouv, rtol=2e-13, atol=2e-13 * scale)
        or not np.allclose(full_frame[:, 2], frame[:, 2], rtol=0.0, atol=2e-12)
    ):
        raise ValueError("physical O/U/V factorization is not equivariant to the crop")

    fov = prepared_context["receipt"]["global_reference_fov"]
    support_origin = np.ascontiguousarray(
        np.asarray(fov["support_origin_ap_dv_ml_um"], dtype=np.float64)
    )
    normal = np.ascontiguousarray(frame[:, 2])
    signed_offset = float(normal @ (center - support_origin))
    full_normal = np.ascontiguousarray(full_frame[:, 2])
    full_offset = float(full_normal @ (full_center - support_origin))
    model_normal = np.ascontiguousarray(model_frame[:, 2])
    model_offset = float(model_normal @ (model_center - support_origin))
    canonical_full = canonicalize_plane(full_normal, full_offset)
    canonical_model = canonicalize_plane(model_normal, model_offset)
    if (
        not np.isclose(full_offset, signed_offset, rtol=0.0, atol=2e-10 * scale)
        or not np.allclose(canonical_full[0], canonical_model[0], rtol=0.0, atol=2e-12)
        or not np.isclose(canonical_full[1], canonical_model[1], rtol=0.0, atol=2e-10 * scale)
    ):
        raise ValueError("crop changed the represented infinite plane")
    canonical_normal, canonical_offset, sign = canonicalize_plane(normal, signed_offset)
    canonical_normal = np.ascontiguousarray(canonical_normal, dtype=np.float64)
    actual_gauge = deterministic_rp2_tangent_gauge_v2(normal)
    canonical_gauge = deterministic_rp2_tangent_gauge_v2(canonical_normal)
    actual_to_canonical = antipodal_tangent_offset_transport_v2(
        normal, canonical_normal
    )
    canonical_to_actual = antipodal_tangent_offset_transport_v2(
        canonical_normal, normal
    )
    roll = float(
        np.arctan2(float(frame[:, 0] @ actual_gauge[:, 1]), float(frame[:, 0] @ actual_gauge[:, 0]))
    )

    support = acquisition._context_support(prepared_context)
    physical_origin = np.asarray(support["origin_um"], dtype=np.float64)
    voxel_size = np.asarray(support["voxel_size_um"], dtype=np.float64)
    atlas_shape = tuple(int(value) for value in support["annotation_shape"])
    quicknii_ouvs = tuple(
        _physical_to_quicknii_ouv(
            ouv, physical_origin, voxel_size, atlas_shape
        )
        for ouv in (parent_ouv, cropped_ouv, model_ouv)
    )

    arrays = {
        _OUV_KEYS[0]: parent_ouv,
        _OUV_KEYS[1]: cropped_ouv,
        _OUV_KEYS[2]: model_ouv,
        "full_raster_quicknii_ouv_ml_ap_dv_float64": quicknii_ouvs[0],
        "cropped_pre_reflection_quicknii_ouv_ml_ap_dv_float64": quicknii_ouvs[1],
        "model_raster_quicknii_ouv_ml_ap_dv_float64": quicknii_ouvs[2],
        "support_origin_ap_dv_ml_um_float64": support_origin,
        "full_raster_physical_center_ap_dv_ml_um_float64": full_center,
        "cropped_pre_reflection_physical_center_ap_dv_ml_um_float64": center,
        "proper_frame_ap_dv_ml_float64": frame,
        "rotation_6d_ap_dv_ml_float64": rotation,
        "positive_upper_triangular_basis_um_float64": basis,
        "log_positive_basis_diagonal_um_float64": log_diagonal,
        "basis_shear_dimensionless_float64": shear.reshape(1),
        "actual_plane_normal_and_signed_offset_um_float64": np.ascontiguousarray(
            np.r_[normal, signed_offset], dtype=np.float64
        ),
        "canonical_plane_normal_and_signed_offset_um_float64": np.ascontiguousarray(
            np.r_[canonical_normal, canonical_offset], dtype=np.float64
        ),
        "actual_tangent_gauge_ap_dv_ml_float64": actual_gauge,
        "canonical_tangent_gauge_ap_dv_ml_float64": canonical_gauge,
        "actual_to_canonical_tangent_offset_transport_float64": actual_to_canonical,
        "canonical_to_actual_tangent_offset_transport_float64": canonical_to_actual,
    }
    arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    array_receipts = _array_receipts(arrays)
    source_ouv_receipts = {
        name: acquisition._json_value(transform["array_receipts"][name])
        for name in _OUV_KEYS
    }
    if any(source_ouv_receipts[name] != array_receipts[name] for name in _OUV_KEYS):
        raise ValueError("copied O/U/V no longer matches the final realization receipt")

    pose_truth = {
        "schema_version": FINITE_PLANE_POSE_TRUTH_V2_SCHEMA,
        "algorithm": FINITE_PLANE_POSE_TRUTH_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_device": "cpu",
        },
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "scope": {
            "deterministic_truth_only": True,
            "posterior_or_probability_claim": False,
            "calibrated_uncertainty_claim": False,
            "tangent_coordinates_are_local_serialization_not_a_global_euclidean_chart": True,
        },
        "upstream_reference": {
            "synthetic_realization_id": final_realization["synthetic_realization_id"],
            "synthetic_realization_receipt_sha256": final_realization["receipt_sha256"],
            "training_row_id": final_realization["training_row_id"],
            "frame_transform_id": transform["frame_transform_id"],
            "parent_subject_centre_plane_fit_id": transform[
                "parent_subject_centre_plane_fit_id"
            ],
            "v2_context_sha256": prepared_context["v2_context_sha256"],
            "prepared_context_receipt_sha256": acquisition._payload_sha256(
                acquisition._json_value(prepared_context["receipt"])
            ),
            "support_index_sha256": prepared_context["receipt"]["support_index_sha256"],
            "global_reference_fov_id": fov["global_reference_fov_id"],
            "source_ouv_array_receipts": source_ouv_receipts,
        },
        "provenance": acquisition._json_value(final_realization["provenance"]),
        "coordinate_contract": {
            "physical_axis_order": ["AP", "DV", "ML"],
            "physical_unit": "um",
            "ouv_layout": "three rows [O,U,V]; O+(x/W)U+(y/H)V",
            "quicknii_axis_order": ["ML", "AP_size-AP", "DV_size-DV"],
            "quicknii_unit": "atlas index voxels",
            "atlas_shape_ap_dv_ml": list(atlas_shape),
            "voxel_size_ap_dv_ml_um": voxel_size.tolist(),
            "physical_origin_ap_dv_ml_um": physical_origin.tolist(),
            "state_raster": "cropped-pre-reflection",
            "proper_frame_columns": ["u", "v", "n"],
            "rotation_6d_columns": ["u", "v"],
            "basis_parameterization": "[[exp(log_a), shear*exp(log_b)],[0,exp(log_b)]]",
        },
        "reflection_state": {
            "horizontal_reflection": bool(transform["horizontal_reflection"]),
            "vertical_reflection": bool(transform["vertical_reflection"]),
            "reflection_order": ["horizontal", "vertical"],
            "discrete_state_is_not_absorbed_into_the_proper_frame": True,
            "model_raster_ouv_is_exact_reparameterization": True,
            "canonical_infinite_plane_is_reflection_invariant": True,
        },
        "plane_serialization": {
            "support_origin_binding": "global_reference_fov.support_origin_ap_dv_ml_um",
            "actual_representative": "proper-frame n and d=n dot (center-support_origin)",
            "canonical_equivalence": "(n,d)~(-n,-d); largest-absolute-normal component is nonnegative",
            "actual_to_canonical_sign": int(sign),
            "actual_roll_rad": roll,
            "roll_range": "[-pi,pi] from atan2",
            "roll_gauge": "atan2(u dot tangent_1, u dot tangent_0) at actual representative",
            "tangent_gauge": (
                "project the first AP/DV/ML axis with least absolute normal component; "
                "second tangent is normal cross first"
            ),
            "antipodal_transport_coordinates": ["tangent_0", "tangent_1", "signed_offset_um"],
            "tangent_coordinate_unit": "radians to first order",
        },
        "arrays": arrays,
        "array_receipts": array_receipts,
    }
    pose_truth["finite_plane_pose_truth_id"] = acquisition._payload_sha256(
        _identity_payload(pose_truth)
    )
    pose_truth["receipt_sha256"] = acquisition._payload_sha256(
        finite_plane_pose_truth_receipt_v2(pose_truth)
    )
    return acquisition._freeze_value(pose_truth)


def make_arbitrary_plane_pose_truth_v2(
    final_realization: Mapping[str, object], prepared_context: Mapping[str, object]
) -> Mapping[str, object]:
    """Derive deterministic finite-plane truth from one already final realization."""
    _verify_final_reference(final_realization, prepared_context)
    return _build_pose_truth(final_realization, prepared_context)


def replay_arbitrary_plane_pose_truth_v2(
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> Mapping[str, object]:
    """Replay a pose sidecar; the argument supplies no stochastic coordinates."""
    if pose_truth.get("schema_version") != FINITE_PLANE_POSE_TRUTH_V2_SCHEMA:
        raise ValueError("unsupported finite-plane pose truth")
    return make_arbitrary_plane_pose_truth_v2(final_realization, prepared_context)


def verify_arbitrary_plane_pose_truth_v2(
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    """Verify strict structure, provenance, deterministic replay, and pose algebra."""
    _verify_final_reference(final_realization, prepared_context)
    if (
        set(pose_truth)
        != {
            "schema_version", "algorithm", "implementation_source_sha256",
            "implementation_source_sha256_canonicalization", "runtime_dependencies",
            "asset_dependencies", "scope", "upstream_reference", "provenance",
            "coordinate_contract", "reflection_state", "plane_serialization", "arrays",
            "array_receipts", "finite_plane_pose_truth_id", "receipt_sha256",
        }
        or set(pose_truth.get("arrays", {})) != _ARRAY_KEYS
        or set(pose_truth.get("array_receipts", {})) != _ARRAY_KEYS
        or _count_synthetic_realization_ids(pose_truth) != 1
    ):
        raise ValueError("finite-plane pose truth has missing or extra fields")
    expected = _build_pose_truth(final_realization, prepared_context)
    if (
        pose_truth["schema_version"] != FINITE_PLANE_POSE_TRUTH_V2_SCHEMA
        or pose_truth["algorithm"] != FINITE_PLANE_POSE_TRUTH_V2_ALGORITHM
        or acquisition._json_value(pose_truth["implementation_source_sha256"])
        != _source_hashes()
        or pose_truth["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or pose_truth["scope"].get("posterior_or_probability_claim") is not False
        or pose_truth["scope"].get("calibrated_uncertainty_claim") is not False
        or any(pose_truth["asset_dependencies"].get(name) for name in (
            "learned_checkpoint_dependencies",
            "pretrained_feature_dependencies",
            "previous_model_dependencies",
        ))
        or acquisition._json_value(pose_truth["array_receipts"])
        != acquisition._json_value(_array_receipts(pose_truth["arrays"]))
        or pose_truth["finite_plane_pose_truth_id"]
        != acquisition._payload_sha256(_identity_payload(pose_truth))
        or pose_truth["receipt_sha256"]
        != acquisition._payload_sha256(finite_plane_pose_truth_receipt_v2(pose_truth))
        or acquisition._json_value(finite_plane_pose_truth_receipt_v2(pose_truth))
        != acquisition._json_value(finite_plane_pose_truth_receipt_v2(expected))
    ):
        raise ValueError("finite-plane pose truth metadata or receipt changed")
    if any(
        np.asarray(pose_truth["arrays"][name]).dtype != np.dtype(np.float64)
        or not np.array_equal(
            np.asarray(pose_truth["arrays"][name]), np.asarray(expected["arrays"][name])
        )
        for name in _ARRAY_KEYS
    ):
        raise ValueError("finite-plane pose truth arrays changed")

    arrays = pose_truth["arrays"]
    center = torch.from_numpy(np.array(arrays[
        "cropped_pre_reflection_physical_center_ap_dv_ml_um_float64"
    ], copy=True))
    frame = torch.from_numpy(np.array(arrays["proper_frame_ap_dv_ml_float64"], copy=True))
    basis = torch.from_numpy(np.array(arrays[
        "positive_upper_triangular_basis_um_float64"
    ], copy=True))
    rotation = torch.from_numpy(np.array(arrays["rotation_6d_ap_dv_ml_float64"], copy=True))
    log_diagonal = torch.from_numpy(np.array(arrays[
        "log_positive_basis_diagonal_um_float64"
    ], copy=True))
    shear = torch.from_numpy(
        np.array(arrays["basis_shear_dimensionless_float64"], copy=True)
    ).reshape(())
    reconstructed = frame_to_physical_ouv(center, frame, basis).numpy().reshape(3, 3)
    reconstructed_basis = positive_inplane_basis(log_diagonal, shear).numpy()
    transport = np.asarray(arrays[
        "actual_to_canonical_tangent_offset_transport_float64"
    ])
    inverse_transport = np.asarray(arrays[
        "canonical_to_actual_tangent_offset_transport_float64"
    ])
    if (
        not np.allclose(
            reconstructed,
            arrays[_OUV_KEYS[1]],
            rtol=2e-13,
            atol=2e-10,
        )
        or not np.allclose(rotation_6d_to_frame(rotation).numpy(), frame, rtol=0.0, atol=2e-12)
        or not np.allclose(reconstructed_basis, basis, rtol=2e-13, atol=2e-12)
        or not np.allclose(transport @ inverse_transport, np.eye(3), rtol=0.0, atol=2e-12)
    ):
        raise ValueError("finite-plane pose truth reconstruction changed")
