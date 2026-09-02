"""Read-only streaming truth-capture audit for the v3 pose catalogue."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_batch_v3 as batch_v3
import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_joint_loss as joint_loss_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_UPDATE_SIZE,
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_recurrent_model import (
    compose_antipodal_plane_frame_residual,
)


CATALOGUE_CAPTURE_AUDIT_V3_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-catalogue-capture-audit/v3"
)
CATALOGUE_CAPTURE_AUDIT_V3_ALGORITHM = (
    "nearest-cell-analytic-antipodal-direct-residual-capture/v3"
)
RESIDUAL_COMPONENTS = (
    "normal_tangent_u_rad",
    "normal_tangent_v_rad",
    "support_origin_normal_offset_um",
    "post_plane_roll_rad",
    "inplane_translation_u_um",
    "inplane_translation_v_um",
    "delta_log_basis_u",
    "delta_log_basis_v",
    "delta_upper_triangular_shear",
)
RESIDUAL_UNITS = (
    "rad",
    "rad",
    "um",
    "rad",
    "um",
    "um",
    "log-ratio",
    "log-ratio",
    "dimensionless",
)
ABSOLUTE_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
MAX_CENTER_ERROR_UM = 1.0e-7
MAX_FRAME_ERROR = 1.0e-10
MAX_BASIS_ERROR_UM = 1.0e-7
MAX_LANDMARK_ERROR_UM = 2.0e-7
CUMULATIVE_ENVELOPE_DESCRIPTION = (
    "conservative cumulative-component envelope only; not an exact nonlinear "
    "multi-step reachability proof"
)
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_catalogue_capture_audit_v3.py",
    "training/arbitrary_plane_batch_v3.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_joint_loss.py",
    "training/arbitrary_plane_joint_model.py",
    "training/arbitrary_plane_coarse_proposal_v5.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_row_cache_v3.py",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("catalogue-capture receipts require finite values")
    return value


def _hash_json(value):
    encoded = json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256():
    return {
        name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _verify_catalogue_snapshot_v3(catalogue, expected_receipt_sha256):
    arrays = catalogue.get("arrays", {})
    required_arrays = {
        "cell_id_int64",
        "cell_states_float64",
        "cell_log_mass_float64",
        "cell_normal_ap_dv_ml_float64",
        "cell_signed_offset_um_float64",
        "cell_roll_rad_float64",
        "cell_support_intersection_margin_um_float64",
        "normal_offset_table_um_float64",
        "representation_log_weight_float64",
        "representation_to_canonical_raster_affine_float64",
    }
    cell_count = catalogue.get("counts", {}).get("cell_count")
    if (
        catalogue.get("schema_version") != catalogue_v3.CATALOGUE_V3_SCHEMA
        or set(arrays) != required_arrays
        or catalogue.get("array_receipts")
        != {name: catalogue_v3._array_receipt(value) for name, value in arrays.items()}
        or catalogue.get("receipt_sha256")
        != catalogue_v3._hash(catalogue_v3.catalogue_receipt_v3(catalogue))
        or catalogue.get("receipt_sha256") != expected_receipt_sha256
        or isinstance(cell_count, bool)
        or not isinstance(cell_count, int)
        or cell_count < 1
        or np.asarray(arrays["cell_states_float64"]).shape != (cell_count, 12)
        or not np.isfinite(np.asarray(arrays["cell_states_float64"])).all()
        or not np.array_equal(
            np.asarray(arrays["cell_id_int64"]), np.arange(cell_count)
        )
    ):
        raise ValueError("catalogue snapshot failed its exact immutable receipt")
    support_origin = np.asarray(
        catalogue.get("support_geometry", {}).get(
            "support_origin_ap_dv_ml_um", ()
        ),
        dtype=np.float64,
    )
    if support_origin.shape != (3,) or not np.isfinite(support_origin).all():
        raise ValueError("catalogue support origin is invalid")
    return support_origin


def _aligned_truth_state_v3(base_state, truth_state):
    base_center, base_frame, _ = full_frame_state_to_components(base_state)
    truth_center, truth_frame, truth_basis = full_frame_state_to_components(
        truth_state
    )
    sign = torch.where(
        (base_frame[..., :, 2] * truth_frame[..., :, 2]).sum(dim=-1) < 0.0,
        -torch.ones_like(base_center[..., 0]),
        torch.ones_like(base_center[..., 0]),
    )
    frame_sign = torch.stack((sign, torch.ones_like(sign), sign), dim=-1)
    basis_sign = torch.stack((sign, torch.ones_like(sign)), dim=-1)
    aligned_frame = truth_frame * frame_sign[..., None, :]
    aligned_basis = (
        basis_sign[..., :, None] * truth_basis * basis_sign[..., None, :]
    )
    return (
        full_frame_state_from_components(
            truth_center, aligned_frame, aligned_basis
        ),
        sign,
    )


def decompose_catalogue_capture_residual_v3(
    catalogue_state,
    truth_state,
    support_origin_ap_dv_ml_um,
):
    """Analytically invert one direct antipodal plane/frame composition."""
    base = torch.as_tensor(catalogue_state, dtype=torch.float64)
    truth = torch.as_tensor(truth_state, dtype=torch.float64)
    if base.shape != (12,) or truth.shape != (12,):
        raise ValueError("catalogue and truth states must each contain 12 values")
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError("catalogue and truth states must be finite")
    origin = torch.as_tensor(support_origin_ap_dv_ml_um, dtype=torch.float64)
    if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
        raise ValueError("support origin must be one finite physical 3-vector")

    aligned_truth, antipodal_sign = _aligned_truth_state_v3(base, truth)
    base_center, base_frame, base_basis = full_frame_state_to_components(base)
    truth_center, truth_frame, truth_basis = full_frame_state_to_components(
        aligned_truth
    )
    base_u, base_v, base_normal = base_frame.unbind(dim=-1)
    truth_normal = truth_frame[:, 2]
    cosine = (base_normal * truth_normal).sum().clamp(-1.0, 1.0)
    tangent_direction = truth_normal - cosine * base_normal
    sine = torch.linalg.vector_norm(tangent_direction)
    angle = torch.atan2(sine, cosine)
    tangent = torch.where(
        sine > 1.0e-12,
        tangent_direction * (angle / sine.clamp_min(1.0e-300)),
        tangent_direction,
    )
    base_offset = ((base_center - origin) * base_normal).sum()
    truth_offset = ((truth_center - origin) * truth_normal).sum()
    residual = torch.zeros(FULL_FRAME_UPDATE_SIZE, dtype=torch.float64)
    residual[0] = (tangent * base_u).sum()
    residual[1] = (tangent * base_v).sum()
    residual[2] = truth_offset - base_offset

    post_plane = compose_antipodal_plane_frame_residual(base, residual, origin)
    post_center, post_frame, _ = full_frame_state_to_components(post_plane)
    post_u, post_v, _ = post_frame.unbind(dim=-1)
    residual[3] = torch.atan2(
        (truth_frame[:, 0] * post_v).sum(),
        (truth_frame[:, 0] * post_u).sum(),
    )
    center_delta = truth_center - post_center
    residual[4] = (center_delta * post_u).sum()
    residual[5] = (center_delta * post_v).sum()
    delta_basis = torch.linalg.solve(base_basis, truth_basis)
    if (
        bool((torch.diagonal(delta_basis) <= 0.0).any())
        or float(delta_basis[1, 0].abs()) > 1.0e-10
    ):
        raise ValueError("analytical upper-triangular basis delta is invalid")
    residual[6:8] = torch.log(torch.diagonal(delta_basis))
    residual[8] = delta_basis[0, 1] / delta_basis[1, 1]
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("analytical catalogue-capture residual became nonfinite")

    recomposed = compose_antipodal_plane_frame_residual(base, residual, origin)
    observed_center, observed_frame, observed_basis = full_frame_state_to_components(
        recomposed
    )
    center_error = float((observed_center - truth_center).abs().max())
    frame_error = float((observed_frame - truth_frame).abs().max())
    basis_error = float((observed_basis - truth_basis).abs().max())
    observed_landmarks = joint_loss_v3.physical_frame_landmarks(recomposed)
    raw_truth_landmarks = joint_loss_v3.physical_frame_landmarks(truth)
    landmark_permutation = (
        (0, 1, 2, 3, 4)
        if int(antipodal_sign) == 1
        else (1, 0, 3, 2, 4)
    )
    truth_landmarks = raw_truth_landmarks[
        torch.tensor(landmark_permutation, dtype=torch.long)
    ]
    aligned_truth_landmarks = joint_loss_v3.physical_frame_landmarks(aligned_truth)
    representation_equivalence_error = float(
        torch.linalg.vector_norm(
            aligned_truth_landmarks - truth_landmarks, dim=-1
        ).max()
    )
    landmark_errors = torch.linalg.vector_norm(
        observed_landmarks - truth_landmarks, dim=-1
    )
    landmark_error = float(landmark_errors.max())
    if (
        center_error > MAX_CENTER_ERROR_UM
        or frame_error > MAX_FRAME_ERROR
        or basis_error > MAX_BASIS_ERROR_UM
        or landmark_error > MAX_LANDMARK_ERROR_UM
        or representation_equivalence_error > MAX_LANDMARK_ERROR_UM
    ):
        raise ValueError("catalogue-capture analytical residual failed exact recomposition")
    return {
        "residual": residual,
        "antipodal_truth_normal_sign": int(antipodal_sign),
        "projective_normal_angle_rad": float(angle),
        "recomposition": {
            "center_max_abs_error_um": center_error,
            "frame_max_abs_error": frame_error,
            "basis_max_abs_error_um": basis_error,
            "physical_landmark_error_um": landmark_errors.tolist(),
            "physical_landmark_max_error_um": landmark_error,
            "truth_landmark_permutation": list(landmark_permutation),
            "antipodal_representation_equivalence_max_error_um": (
                representation_equivalence_error
            ),
            "truth_representation": (
                "canonical" if int(antipodal_sign) == 1 else "horizontal-antipodal-lift"
            ),
        },
    }


def _component_summary(values, envelope):
    matrix = np.asarray(values, dtype=np.float64)
    absolute = np.abs(matrix)
    summary = {}
    for index, (name, unit) in enumerate(zip(RESIDUAL_COMPONENTS, RESIDUAL_UNITS)):
        quantiles = np.quantile(absolute[:, index], ABSOLUTE_QUANTILES)
        maximum = float(absolute[:, index].max())
        summary[name] = {
            "unit": unit,
            "maximum_absolute": maximum,
            "absolute_quantiles": {
                f"q{int(round(100 * quantile)):02d}": float(value)
                for quantile, value in zip(ABSOLUTE_QUANTILES, quantiles)
            },
            "cumulative_component_envelope": float(envelope[index]),
            "maximum_fraction_of_cumulative_envelope": (
                maximum / float(envelope[index])
            ),
        }
    return summary


def catalogue_capture_audit_receipt_v3(report):
    return {key: value for key, value in report.items() if key != "receipt_sha256"}


def verify_catalogue_capture_audit_report_v3(report):
    payload = catalogue_capture_audit_receipt_v3(report)
    if (
        report.get("schema_version") != CATALOGUE_CAPTURE_AUDIT_V3_SCHEMA
        or report.get("algorithm") != CATALOGUE_CAPTURE_AUDIT_V3_ALGORITHM
        or report.get("receipt_sha256") != _hash_json(payload)
        or report.get("all_rows_captured") is not True
        or report.get("failure_count") != 0
    ):
        raise ValueError("catalogue-capture audit report failed its receipt")
    return True


def audit_catalogue_capture_v3(
    cache_directory,
    catalogue,
    *,
    atlas_shape_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    update_limits,
    refinement_steps,
    expected_cache_manifest_receipt_sha256,
    expected_catalogue_receipt_sha256,
):
    """Stream a frozen row cache and prove one-residual catalogue truth capture."""
    support_origin = _verify_catalogue_snapshot_v3(
        catalogue, expected_catalogue_receipt_sha256
    )
    raw_shape = tuple(atlas_shape_ap_dv_ml)
    if (
        len(raw_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_shape
        )
    ):
        raise ValueError("atlas shape must contain three explicit integers")
    shape = tuple(int(value) for value in raw_shape)
    origin = np.asarray(origin_ap_dv_ml_um, dtype=np.float64)
    spacing = np.asarray(voxel_size_ap_dv_ml_um, dtype=np.float64)
    support_geometry = catalogue["support_geometry"]
    if (
        len(shape) != 3
        or any(value < 2 for value in shape)
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0.0)
        or list(shape) != support_geometry["support_mask_receipt"]["shape"]
        or not np.array_equal(origin, support_geometry["origin_ap_dv_ml_um"])
        or not np.array_equal(spacing, support_geometry["voxel_size_ap_dv_ml_um"])
    ):
        raise ValueError("atlas geometry differs from the receipt-bound catalogue")
    limits = np.asarray(update_limits, dtype=np.float64)
    if (
        limits.shape != (FULL_FRAME_UPDATE_SIZE,)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0.0)
        or isinstance(refinement_steps, bool)
        or not isinstance(refinement_steps, int)
        or refinement_steps < 1
    ):
        raise ValueError("model update limits and refinement steps are invalid")
    recurrent_update_count = refinement_steps + 1
    envelope = recurrent_update_count * limits

    root = row_cache_v3._i_path(cache_directory)
    manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        root, expected_receipt_sha256=expected_cache_manifest_receipt_sha256
    )
    if manifest["status"] != row_cache_v3.FROZEN_CACHE_STATUS:
        raise ValueError("catalogue-capture audit requires one frozen row cache")
    geometry_gauge_contract = manifest["generator_binding"][
        "geometry_gauge_contract"
    ]
    composite_contract = row_cache_v3._composite_cache_contract_v3(
        manifest["generation_config"], manifest["generator_binding"]
    )
    source_sha256 = _source_sha256()
    catalogue_states = torch.from_numpy(
        np.asarray(catalogue["arrays"]["cell_states_float64"])
    )
    catalogue_for_nearest = {
        **catalogue,
        "tensors": {"cell_states": catalogue_states[None]},
    }
    rows = []
    residual_values = []
    maximum_recomposition = {
        "center_max_abs_error_um": 0.0,
        "frame_max_abs_error": 0.0,
        "basis_max_abs_error_um": 0.0,
        "physical_landmark_max_error_um": 0.0,
    }
    for record in manifest["rows"]:
        row = row_cache_v3._load_record(
            root, record, geometry_gauge_contract
        )
        if composite_contract is not None:
            expected = row_cache_v3._composite_row_receipts_v3(
                row, record["row_index"], composite_contract
            )
            if any(record.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    "composite cached row differs from its binding receipts"
                )
        truth_state = batch_v3.physical_state_from_quicknii_ouv_v3(
            row["canonical_effective_quicknii_ouv_float64"],
            shape,
            origin,
            spacing,
        )
        cell_index = int(
            batch_v3.nearest_catalogue_cell_v3(
                truth_state, catalogue_for_nearest
            )[0]
        )
        decomposition = decompose_catalogue_capture_residual_v3(
            catalogue_states[cell_index], truth_state, support_origin
        )
        residual = decomposition["residual"].detach().cpu().numpy()
        violation = np.flatnonzero(np.abs(residual) > envelope + 1.0e-12)
        if violation.size:
            names = [RESIDUAL_COMPONENTS[index] for index in violation]
            raise ValueError(
                "catalogue-capture row exceeds the cumulative-component envelope: "
                f"{row['training_row_id']} {names}"
            )
        residual_values.append(residual)
        recomposition = decomposition["recomposition"]
        for name in maximum_recomposition:
            maximum_recomposition[name] = max(
                maximum_recomposition[name], float(recomposition[name])
            )
        rows.append(
            {
                "row_index": record["row_index"],
                "training_row_id": row["training_row_id"],
                "training_row_receipt_sha256": row["receipt_sha256"],
                "animal_id": row["lineage"]["animal_id"],
                "specimen_id": row["lineage"]["specimen_id"],
                "experiment_id": row["lineage"]["experiment_id"],
                "synthetic_animal_id": row["lineage"]["synthetic_animal_id"],
                "section_id": row["lineage"]["section_id"],
                "selected_mode": row["selected_mode"],
                "reflection_state": row["reflection_state"],
                "nearest_catalogue_cell_index": cell_index,
                "nearest_catalogue_cell_id": int(
                    catalogue["arrays"]["cell_id_int64"][cell_index]
                ),
                "antipodal_truth_normal_sign": decomposition[
                    "antipodal_truth_normal_sign"
                ],
                "projective_normal_angle_rad": decomposition[
                    "projective_normal_angle_rad"
                ],
                "direct_residual": {
                    name: float(residual[index])
                    for index, name in enumerate(RESIDUAL_COMPONENTS)
                },
                "recomposition": recomposition,
            }
        )
        del row, truth_state, decomposition

    if not rows:
        raise ValueError("catalogue-capture audit requires at least one cached row")
    if _source_sha256() != source_sha256:
        raise RuntimeError("catalogue-capture audit source changed during execution")
    model_contract = {
        "update_limits": limits.tolist(),
        "refinement_steps": refinement_steps,
        "recurrent_update_count": recurrent_update_count,
        "cumulative_component_envelope": envelope.tolist(),
        "cumulative_component_envelope_description": (
            CUMULATIVE_ENVELOPE_DESCRIPTION
        ),
        "model_source_sha256": {
            name: source_sha256[name]
            for name in (
                "training/arbitrary_plane_full_frame_primitives.py",
                "training/arbitrary_plane_joint_loss.py",
                "training/arbitrary_plane_joint_model.py",
                "training/arbitrary_plane_coarse_proposal_v5.py",
                "training/arbitrary_plane_recurrent_model.py",
            )
        },
    }
    model_contract["receipt_sha256"] = _hash_json(model_contract)
    payload = {
        "schema_version": CATALOGUE_CAPTURE_AUDIT_V3_SCHEMA,
        "algorithm": CATALOGUE_CAPTURE_AUDIT_V3_ALGORITHM,
        "data_role": "development-training prelaunch truth-capture gate",
        "catalogue_binding": {
            "catalogue_id": catalogue["catalogue_id"],
            "receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": catalogue["counts"]["cell_count"],
            "array_receipts": catalogue["array_receipts"],
        },
        "row_cache_binding": {
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "generator_binding_receipt_sha256": manifest["generator_binding"][
                "receipt_sha256"
            ],
            "row_count": manifest["row_count"],
            "freeze_audit": manifest["freeze_audit"],
        },
        "atlas_geometry_binding": {
            "shape_ap_dv_ml": list(shape),
            "origin_ap_dv_ml_um": origin.tolist(),
            "voxel_size_ap_dv_ml_um": spacing.tolist(),
            "support_origin_ap_dv_ml_um": support_origin.tolist(),
            "receipt_sha256": _hash_json(
                {
                    "shape_ap_dv_ml": shape,
                    "origin_ap_dv_ml_um": origin,
                    "voxel_size_ap_dv_ml_um": spacing,
                    "support_origin_ap_dv_ml_um": support_origin,
                }
            ),
        },
        "model_capture_contract": model_contract,
        "source_sha256": source_sha256,
        "row_count": len(rows),
        "rows": rows,
        "component_summary": _component_summary(residual_values, envelope),
        "maximum_recomposition_error": maximum_recomposition,
        "all_rows_captured": True,
        "failure_count": 0,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    report = {**payload, "receipt_sha256": _hash_json(payload)}
    verify_catalogue_capture_audit_report_v3(report)
    return report


__all__ = [
    "ABSOLUTE_QUANTILES",
    "CATALOGUE_CAPTURE_AUDIT_V3_ALGORITHM",
    "CATALOGUE_CAPTURE_AUDIT_V3_SCHEMA",
    "CUMULATIVE_ENVELOPE_DESCRIPTION",
    "RESIDUAL_COMPONENTS",
    "audit_catalogue_capture_v3",
    "catalogue_capture_audit_receipt_v3",
    "decompose_catalogue_capture_residual_v3",
    "verify_catalogue_capture_audit_report_v3",
]
