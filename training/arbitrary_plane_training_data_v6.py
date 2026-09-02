"""Authenticated finite-row tensors for the complete v6 catalogue.

The frozen cache-manifest receipt is the stable run-level data binding.  Each
selected minibatch additionally carries its own selection receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import torch

from training.arbitrary_plane_catalogue_binding_v3 import (
    verify_catalogue_binding_v3,
)
import training.arbitrary_plane_finite_row_binding_v6 as finite_rows_v6
from training.arbitrary_plane_geometry import (
    allen_index_to_physical_um_points,
    allen_index_to_physical_um_vectors,
    physical_ouv_to_frame,
    quicknii_to_allen_points,
    quicknii_to_allen_vectors,
)
from training.arbitrary_plane_catalogue_runtime_v6 import (
    CompleteCatalogueRuntimeV6,
    verify_bound_complete_catalogue_batch_v6,
    verify_complete_catalogue_runtime_v6,
)
from training.arbitrary_plane_full_frame_primitives import (
    FULL_FRAME_UPDATE_SIZE,
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_recurrent_model import (
    compose_antipodal_plane_frame_residual,
)


TRAINING_DATA_V6_SCHEMA = "anatomy-tracker.arbitrary-plane-training-data/v6"
FROZEN_ROWS_V6_SCHEMA = "anatomy-tracker.frozen-generated-row-payloads/v6"
CATALOGUE_TRUTH_V6_SCHEMA = "anatomy-tracker.catalogue-truth-mapping/v6"
FULL_CATALOGUE_CELL_COUNT_V6 = 98_304
MODE_TO_INPUT_V6 = {
    "smart-brush-absent": "raw",
    "smart-brush-accurate": "black-exterior",
    "smart-brush-imperfect": "imperfect-mask",
}
LINEAGE_KEYS_V6 = (
    "animal_id",
    "specimen_id",
    "experiment_id",
    "synthetic_animal_id",
    "section_id",
)
DEPENDENCY_KEYS = {
    "learned_dependencies",
    "previous_model_dependencies",
    "learned_style_model_dependencies",
    "prior_model_dependencies",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "pseudolabel_dependencies",
}
RECOMPOSITION_CENTER_ATOL_UM = 1.0e-7
RECOMPOSITION_FRAME_ATOL = 1.0e-10
RECOMPOSITION_BASIS_ATOL_UM = 1.0e-7
REQUIRED_MAPPING_ARRAYS = {
    "cell_id_int64",
    "cell_states_float64",
    "normal_offset_table_um_float64",
}


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_receipt(value) -> dict[str, object]:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(normalized.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "array_sha256": digest.hexdigest(),
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and not (set(value) - set("0123456789abcdef"))
    )


def _empty_dependency(value: object) -> bool:
    return value is None or (
        isinstance(value, (list, tuple, dict)) and len(value) == 0
    )


def _assert_untrained_dependencies(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                ("candidate_bank" in normalized)
                or ("training_bank" in normalized)
                or ("legacy_prediction" in normalized)
                or ("legacy_feature" in normalized)
            ):
                raise ValueError("v6 rows cannot contain an earlier retrieval data source")
            if normalized in DEPENDENCY_KEYS and not _empty_dependency(item):
                raise ValueError("v6 rows cannot contain learned dependencies")
            if normalized == "automatic_segmentation_dependency" and item is not False:
                raise ValueError("v6 input rows cannot require automatic segmentation")
            _assert_untrained_dependencies(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_untrained_dependencies(item)


def _verify_catalogue_snapshot(catalogue: Mapping[str, object]) -> None:
    arrays = catalogue.get("arrays", {})
    if not isinstance(arrays, Mapping) or not REQUIRED_MAPPING_ARRAYS.issubset(
        arrays
    ):
        raise ValueError(
            "catalogue lacks authenticated nearest-cell geometry; no truth label was assigned"
        )


def _physical_state_from_quicknii_ouv_v6(
    quicknii_ouv,
    atlas_shape_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
) -> torch.Tensor:
    quicknii = torch.as_tensor(quicknii_ouv, dtype=torch.float64).reshape(3, 3)
    origin = torch.as_tensor(origin_ap_dv_ml_um, dtype=torch.float64)
    spacing = torch.as_tensor(voxel_size_ap_dv_ml_um, dtype=torch.float64)
    physical_ouv = torch.cat(
        (
            allen_index_to_physical_um_points(
                quicknii_to_allen_points(quicknii[0], tuple(atlas_shape_ap_dv_ml)),
                origin,
                spacing,
            ),
            allen_index_to_physical_um_vectors(
                quicknii_to_allen_vectors(quicknii[1]), spacing
            ),
            allen_index_to_physical_um_vectors(
                quicknii_to_allen_vectors(quicknii[2]), spacing
            ),
        )
    )
    center, frame, basis = physical_ouv_to_frame(physical_ouv)
    return full_frame_state_from_components(center, frame, basis)


def _nearest_catalogue_cell_v6(truth_state, catalogue) -> torch.Tensor:
    truth = torch.as_tensor(truth_state, dtype=torch.float64)
    if truth.ndim == 1:
        truth = truth[None]
    states = catalogue["tensors"]["cell_states"][0].to(
        device=truth.device, dtype=torch.float64
    )
    support_origin = torch.as_tensor(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        device=truth.device,
        dtype=torch.float64,
    )
    candidate_center, candidate_frame, _ = full_frame_state_to_components(states)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    candidate_normal = candidate_frame[:, :, 2]
    truth_normal = truth_frame[:, :, 2]
    dot = truth_normal @ candidate_normal.T
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    normal_angle = torch.acos(dot.abs().clamp(0.0, 1.0))
    truth_offset = ((truth_center - support_origin) * truth_normal).sum(dim=-1)
    candidate_offset = (
        (candidate_center - support_origin) * candidate_normal
    ).sum(dim=-1)
    offset_error = (truth_offset[:, None] * sign - candidate_offset[None]).abs()
    offset_table = torch.as_tensor(
        catalogue["arrays"]["normal_offset_table_um_float64"],
        device=truth.device,
        dtype=torch.float64,
    )
    offset_step = torch.diff(offset_table, dim=1).abs().median().clamp_min(1.0)
    normal_scale = max(
        float(
            catalogue["coverage_audit"][
                "max_observed_rp2_angular_covering_radius_rad"
            ]
        ),
        1.0e-3,
    )
    truth_u = truth_frame[:, :, 0]
    aligned_candidate_u = candidate_frame[:, :, 0][None].expand(
        truth.shape[0], -1, -1
    )
    aligned_candidate_u = torch.where(
        (sign < 0.0)[..., None], -aligned_candidate_u, aligned_candidate_u
    )
    roll_error = torch.acos(
        (truth_u[:, None] * aligned_candidate_u)
        .sum(dim=-1)
        .clamp(min=-1.0, max=1.0)
    )
    roll_scale = np.pi / catalogue["counts"]["roll_count"]
    cost = (
        (normal_angle / normal_scale).square()
        + (offset_error / offset_step).square()
        + (roll_error / roll_scale).square()
    )
    return cost.argmin(dim=1)


def _finite_row_to_tensors_v6(
    row,
    *,
    atlas_shape_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    finite_psf_capability,
    device,
) -> dict[str, torch.Tensor]:
    finite_rows_v6.verify_finite_training_row_v6(
        row, finite_psf_capability=finite_psf_capability
    )
    arrays = row["arrays"]
    channels = torch.from_numpy(
        np.ascontiguousarray(arrays["model_input_channels_float32"])
    ).permute(2, 0, 1)[None]
    velocity = torch.from_numpy(
        np.ascontiguousarray(
            arrays["truth_section_pullback_stationary_velocity_yx_px_float64"]
        )
    ).permute(2, 0, 1)[None]
    pullback = torch.from_numpy(
        np.ascontiguousarray(arrays["truth_section_pullback_map_yx_px_float64"])
    ).permute(2, 0, 1)[None]
    deformation_valid = torch.from_numpy(
        np.ascontiguousarray(arrays["truth_section_deformation_valid_mask"])
    )[None, None]
    correspondence_valid = torch.from_numpy(
        np.ascontiguousarray(arrays["target_valid_correspondence_mask"])
    )[None, None]
    abstention = torch.from_numpy(
        np.ascontiguousarray(arrays["target_correspondence_abstention_mask"])
    )[None, None]
    correspondence_weight = torch.from_numpy(
        np.ascontiguousarray(arrays["target_correspondence_weight_float32"])
    )[None, None]
    loss_weight = (
        deformation_valid & correspondence_valid & ~abstention
    ).to(correspondence_weight) * correspondence_weight
    support_contract = row["upstream_reference"].get(
        "support_supervision_contract",
        {
            "point_pose_supervision_weight": 1.0,
            "dense_deformation_supervision_weight": 1.0,
        },
    )
    psf = finite_rows_v6.finite_psf_tensors_from_training_row_v6(
        row,
        finite_psf_capability=finite_psf_capability,
        device=device,
        dtype=torch.float32,
    )
    tensors = {
        "image": channels[:, :1],
        "outline": channels[:, 1:2],
        "outline_available": channels[:, 2].mean(dim=(-2, -1)),
        "truth_state": _physical_state_from_quicknii_ouv_v6(
            row["canonical_effective_quicknii_ouv_float64"],
            atlas_shape_ap_dv_ml,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
        )[None],
        "pose_supervision_weight": torch.tensor(
            [float(support_contract["point_pose_supervision_weight"])],
            dtype=torch.float32,
        ),
        "dense_deformation_supervision_weight": torch.tensor(
            [float(support_contract["dense_deformation_supervision_weight"])],
            dtype=torch.float32,
        ),
        "truth_stationary_velocity_yx_px": velocity,
        "truth_pullback_map_yx_px": pullback,
        "deformation_weight": loss_weight,
        **psf,
    }
    return {
        name: value.to(device=device, dtype=torch.float32)
        if torch.is_floating_point(value)
        else value.to(device=device)
        for name, value in tensors.items()
    }
    verify_catalogue_binding_v3(catalogue)
    cell_id = torch.as_tensor(arrays["cell_id_int64"], dtype=torch.long)
    if not torch.equal(cell_id, torch.arange(cell_id.numel())):
        raise ValueError("catalogue cell IDs must equal canonical row indices")
    geometry = catalogue.get("support_geometry", {})
    counts = catalogue.get("counts", {})
    coverage = catalogue.get("coverage_audit", {})
    support_origin = np.asarray(
        geometry.get("support_origin_ap_dv_ml_um", ()), dtype=np.float64
    )
    offset_table = np.asarray(arrays["normal_offset_table_um_float64"])
    states = np.asarray(arrays["cell_states_float64"])
    if (
        support_origin.shape != (3,)
        or not np.isfinite(support_origin).all()
        or states.shape != (cell_id.numel(), 12)
        or not np.isfinite(states).all()
        or counts.get("cell_count") != cell_id.numel()
        or offset_table.ndim != 2
        or counts.get("normal_count") != offset_table.shape[0]
        or counts.get("offset_count_per_normal") != offset_table.shape[1]
        or offset_table.shape[1] < 2
        or not np.isfinite(offset_table).all()
        or not isinstance(counts.get("roll_count"), int)
        or isinstance(counts.get("roll_count"), bool)
        or counts["roll_count"] < 1
        or not isinstance(
            coverage.get("max_observed_rp2_angular_covering_radius_rad"),
            (int, float),
        )
        or not math.isfinite(
            float(coverage["max_observed_rp2_angular_covering_radius_rad"])
        )
    ):
        raise ValueError(
            "catalogue lacks authenticated nearest-cell geometry; no truth label was assigned"
        )


def _frozen_rows_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "training_data_manifest_receipt_sha256",
            "cache_manifest_receipt_sha256",
            "generator_binding_receipt_sha256",
            "generation_lineage_sha256",
            "row_indices",
            "training_row_ids",
            "training_row_receipts_sha256",
        )
    }


def load_frozen_training_rows_v6(
    cache_directory,
    indices=None,
    *,
    expected_manifest_receipt_sha256,
) -> dict[str, object]:
    """Read exact rows through the standalone authenticated finite-row reader."""
    return finite_rows_v6.load_frozen_training_rows_v6(
        cache_directory,
        indices,
        expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
    )


def _rows_and_source(rows_or_payload) -> tuple[list[Mapping[str, object]], object]:
    if not isinstance(rows_or_payload, Mapping):
        raise ValueError("v6 model batches require an authenticated frozen-row selection")
    payload = rows_or_payload
    if (
        set(payload)
        != {
            "schema_version",
            "training_data_manifest_receipt_sha256",
            "cache_manifest_receipt_sha256",
            "generator_binding_receipt_sha256",
            "generation_lineage_sha256",
            "row_indices",
            "training_row_ids",
            "training_row_receipts_sha256",
            "rows",
            "selection_receipt_sha256",
        }
        or payload.get("schema_version") != FROZEN_ROWS_V6_SCHEMA
        or payload.get("selection_receipt_sha256")
        != _payload_sha256(_frozen_rows_receipt(payload))
        or not all(
            _valid_sha256(payload.get(name))
            for name in (
                "training_data_manifest_receipt_sha256",
                "cache_manifest_receipt_sha256",
                "generator_binding_receipt_sha256",
                "generation_lineage_sha256",
            )
        )
    ):
        raise ValueError("frozen generated-row payload receipt is invalid")
    rows = list(payload["rows"])
    if (
        payload["training_row_ids"] != [row.get("training_row_id") for row in rows]
        or payload["training_row_receipts_sha256"]
        != [row.get("receipt_sha256") for row in rows]
        or len(payload["row_indices"]) != len(rows)
    ):
        raise ValueError("frozen generated-row payload no longer matches its rows")
    source = {key: copy.deepcopy(value) for key, value in _frozen_rows_receipt(payload).items()}
    if (
        payload["training_data_manifest_receipt_sha256"]
        != payload["cache_manifest_receipt_sha256"]
    ):
        raise ValueError("frozen row selection differs from its run-level data manifest")
    source["selection_receipt_sha256"] = payload["selection_receipt_sha256"]
    return rows, source


def _aligned_truth_state(
    catalogue_state: torch.Tensor, truth_state: torch.Tensor
) -> tuple[torch.Tensor, int]:
    base_center, base_frame, _ = full_frame_state_to_components(catalogue_state)
    truth_center, truth_frame, truth_basis = full_frame_state_to_components(truth_state)
    sign = -1 if float((base_frame[:, 2] * truth_frame[:, 2]).sum()) < 0.0 else 1
    frame_sign = torch.tensor(
        (sign, 1.0, sign), dtype=truth_state.dtype, device=truth_state.device
    )
    basis_sign = torch.tensor(
        (sign, 1.0), dtype=truth_state.dtype, device=truth_state.device
    )
    aligned = full_frame_state_from_components(
        truth_center,
        truth_frame * frame_sign[None],
        basis_sign[:, None] * truth_basis * basis_sign[None, :],
    )
    return aligned, sign


def _decompose_truth_residual(
    catalogue_state: torch.Tensor,
    truth_state: torch.Tensor,
    support_origin_ap_dv_ml_um,
) -> dict[str, object]:
    base = torch.as_tensor(catalogue_state, dtype=torch.float64)
    truth = torch.as_tensor(truth_state, dtype=torch.float64)
    origin = torch.as_tensor(support_origin_ap_dv_ml_um, dtype=torch.float64)
    if base.shape != (12,) or truth.shape != (12,) or origin.shape != (3,):
        raise ValueError("catalogue truth decomposition requires 12-state and 3-origin geometry")
    if not bool(torch.isfinite(base).all() and torch.isfinite(truth).all() and torch.isfinite(origin).all()):
        raise ValueError("catalogue truth decomposition requires finite geometry")

    aligned_truth, sign = _aligned_truth_state(base, truth)
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
    residual = torch.zeros(FULL_FRAME_UPDATE_SIZE, dtype=torch.float64)
    residual[0] = (tangent * base_u).sum()
    residual[1] = (tangent * base_v).sum()
    base_offset = ((base_center - origin) * base_normal).sum()
    truth_offset = ((truth_center - origin) * truth_normal).sum()
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
    if bool((torch.diagonal(delta_basis) <= 0.0).any()) or float(
        delta_basis[1, 0].abs()
    ) > 1.0e-10:
        raise ValueError("catalogue truth has no positive upper-triangular residual")
    residual[6:8] = torch.log(torch.diagonal(delta_basis))
    residual[8] = delta_basis[0, 1] / delta_basis[1, 1]

    recomposed = compose_antipodal_plane_frame_residual(base, residual, origin)
    observed_center, observed_frame, observed_basis = full_frame_state_to_components(
        recomposed
    )
    center_error = float((observed_center - truth_center).abs().max())
    frame_error = float((observed_frame - truth_frame).abs().max())
    basis_error = float((observed_basis - truth_basis).abs().max())
    if (
        center_error > RECOMPOSITION_CENTER_ATOL_UM
        or frame_error > RECOMPOSITION_FRAME_ATOL
        or basis_error > RECOMPOSITION_BASIS_ATOL_UM
        or not bool(torch.isfinite(residual).all())
    ):
        raise ValueError("catalogue residual did not recompose the complete truth state")
    return {
        "residual": residual,
        "aligned_truth_state": aligned_truth,
        "recomposed_truth_state": recomposed,
        "antipodal_truth_normal_sign": sign,
        "recomposition": {
            "center_max_abs_error_um": center_error,
            "frame_max_abs_error": frame_error,
            "basis_max_abs_error_um": basis_error,
        },
    }


def catalogue_truth_targets_v6(
    truth_state,
    catalogue: Mapping[str, object],
) -> dict[str, object]:
    """Replay the canonical v3 nearest-cell rule and retain its exact residual."""
    _verify_catalogue_snapshot(catalogue)
    truth = torch.as_tensor(truth_state, dtype=torch.float64).cpu()
    if truth.ndim == 1:
        truth = truth[None]
    if truth.ndim != 2 or truth.shape[1] != 12 or not bool(torch.isfinite(truth).all()):
        raise ValueError("truth_state must have finite shape (B,12)")
    states = torch.as_tensor(
        catalogue["arrays"]["cell_states_float64"], dtype=torch.float64
    )
    support_origin = torch.as_tensor(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        dtype=torch.float64,
    )
    cell_center, cell_frame, _ = full_frame_state_to_components(states)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    cell_normal = cell_frame[:, :, 2]
    truth_normal = truth_frame[:, :, 2]
    dot = truth_normal @ cell_normal.T
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    normal_angle = torch.acos(dot.abs().clamp(0.0, 1.0))
    truth_offset = ((truth_center - support_origin) * truth_normal).sum(dim=-1)
    cell_offset = ((cell_center - support_origin) * cell_normal).sum(dim=-1)
    offset_error = (truth_offset[:, None] * sign - cell_offset[None]).abs()
    offset_table = torch.as_tensor(
        catalogue["arrays"]["normal_offset_table_um_float64"],
        dtype=torch.float64,
    )
    offset_scale = torch.diff(offset_table, dim=1).abs().median().clamp_min(1.0)
    normal_scale = max(
        float(
            catalogue["coverage_audit"][
                "max_observed_rp2_angular_covering_radius_rad"
            ]
        ),
        1.0e-3,
    )
    truth_u = truth_frame[:, :, 0]
    aligned_cell_u = cell_frame[:, :, 0][None].expand(truth.shape[0], -1, -1)
    aligned_cell_u = torch.where(
        (sign < 0.0)[..., None], -aligned_cell_u, aligned_cell_u
    )
    roll_error = torch.acos(
        (truth_u[:, None] * aligned_cell_u).sum(dim=-1).clamp(-1.0, 1.0)
    )
    roll_scale = math.pi / int(catalogue["counts"]["roll_count"])
    normal_cost = (normal_angle / normal_scale).square()
    offset_cost = (offset_error / offset_scale).square()
    roll_cost = (roll_error / roll_scale).square()
    total_cost = normal_cost + offset_cost + roll_cost
    index = total_cost.argmin(dim=1)

    replay_catalogue = {
        **catalogue,
        "tensors": {"cell_states": states[None]},
    }
    replay_index = _nearest_catalogue_cell_v6(truth, replay_catalogue)
    if not torch.equal(index, replay_index):
        raise RuntimeError("v6 nearest-cell audit differs from the verified mapping")
    cell_id = torch.as_tensor(
        catalogue["arrays"]["cell_id_int64"], dtype=torch.long
    )[index]
    if not torch.equal(cell_id, index):
        raise RuntimeError("canonical catalogue index and cell ID diverged")

    decomposed = [
        _decompose_truth_residual(states[int(item)], truth[row], support_origin)
        for row, item in enumerate(index)
    ]
    residual = torch.stack([item["residual"] for item in decomposed])
    aligned_truth = torch.stack([item["aligned_truth_state"] for item in decomposed])
    recomposed_truth = torch.stack(
        [item["recomposed_truth_state"] for item in decomposed]
    )
    audit = []
    for row, selected in enumerate(index.tolist()):
        selected_cost = float(total_cost[row, selected])
        values = {
            "schema_version": CATALOGUE_TRUTH_V6_SCHEMA,
            "selection_algorithm": (
                "standalone v6 normalized RP2-normal/offset/roll cost; "
                "first canonical index on an exact tie"
            ),
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "truth_state_receipt": _array_receipt(truth[row].numpy()),
            "truth_catalogue_index": selected,
            "truth_catalogue_cell_id": int(cell_id[row]),
            "normal_angle_rad": float(normal_angle[row, selected]),
            "normal_scale_rad": normal_scale,
            "normal_normalized_squared_distance": float(normal_cost[row, selected]),
            "normal_offset_abs_error_um": float(offset_error[row, selected]),
            "normal_offset_scale_um": float(offset_scale),
            "normal_offset_normalized_squared_distance": float(
                offset_cost[row, selected]
            ),
            "roll_angle_rad": float(roll_error[row, selected]),
            "roll_scale_rad": roll_scale,
            "roll_normalized_squared_distance": float(roll_cost[row, selected]),
            "total_normalized_squared_distance": selected_cost,
            "exact_minimum_tie_count": int(
                torch.count_nonzero(total_cost[row] == total_cost[row, selected])
            ),
            "antipodal_truth_normal_sign": decomposed[row][
                "antipodal_truth_normal_sign"
            ],
            "residual_receipt": _array_receipt(
                residual[row].numpy()
            ),
            "recomposition": decomposed[row]["recomposition"],
        }
        values["receipt_sha256"] = _payload_sha256(values)
        audit.append(values)
    return {
        "truth_catalogue_index": index,
        "truth_catalogue_cell_id": cell_id,
        "truth_catalogue_residual": residual,
        "truth_catalogue_residual_float64": residual,
        "truth_catalogue_aligned_state_float64": aligned_truth,
        "truth_catalogue_recomposed_state_float64": recomposed_truth,
        "catalogue_truth_mapping_audit": audit,
    }


def _input_semantics(row: Mapping[str, object]) -> dict[str, object]:
    mode = row.get("selected_mode")
    if mode not in MODE_TO_INPUT_V6:
        raise ValueError("v6 row has an unsupported input mode")
    channels = np.asarray(row["arrays"]["model_input_channels_float32"])
    if (
        channels.ndim != 3
        or channels.shape[-1] != 3
        or channels.dtype != np.float32
        or not np.isfinite(channels).all()
        or np.any((channels[..., :2] < 0.0) | (channels[..., :2] > 1.0))
        or np.any((channels[..., 1] != 0.0) & (channels[..., 1] != 1.0))
        or np.any((channels[..., 2] != 0.0) & (channels[..., 2] != 1.0))
        or not np.all(channels[..., 2] == channels[0, 0, 2])
    ):
        raise ValueError("v6 rows require finite image/outline/availability channels")
    expected_available = mode != "smart-brush-absent"
    available = bool(channels[0, 0, 2])
    upstream = row.get("upstream_reference", {})
    black_exterior = upstream.get("selected_black_exterior_exact")
    mask_receipt = upstream.get("selected_input_mask_receipt")
    if (
        not isinstance(mask_receipt, Mapping)
        or mask_receipt.get("shape") != list(channels.shape[:2])
        or mask_receipt.get("dtype") != np.dtype(bool).str
        or not _valid_sha256(mask_receipt.get("array_sha256"))
    ):
        raise ValueError("v6 row is missing its authenticated input-mask receipt")
    if available != expected_available:
        raise ValueError("row mode and outline availability disagree")
    if mode == "smart-brush-absent":
        if np.any(channels[..., 1] != 0.0) or black_exterior is not None:
            raise ValueError("raw mode must retain acquired background without an outline")
    elif black_exterior is not True:
        raise ValueError("available smart-brush rows require authenticated exact black exterior")
    return {
        "selected_mode": mode,
        "input_mode": MODE_TO_INPUT_V6[mode],
        "outline_available": available,
        "model_input_channels_receipt": _array_receipt(channels),
        "selected_input_mask_receipt": copy.deepcopy(mask_receipt),
        "black_exterior_exact": black_exterior,
        "automatic_segmentation_required": False,
    }


def _verify_row_v6(row: Mapping[str, object], capability: Mapping[str, object]) -> None:
    if row.get("schema_version") != finite_rows_v6.TRAINING_ROW_V4_SCHEMA:
        raise ValueError("v6 training accepts only finite authenticated training-row/v4 payloads")
    finite_rows_v6.verify_finite_training_row_v6(
        row, finite_psf_capability=capability
    )
    _assert_untrained_dependencies(row)
    lineage = row.get("lineage", {})
    if any(
        not isinstance(lineage.get(key), str) or not lineage[key]
        for key in LINEAGE_KEYS_V6
    ):
        raise ValueError("v6 rows require exact nonempty five-part string lineage")
    if lineage.get("split") != "train":
        raise ValueError("v6 optimization accepts only exact train-split rows")
    if not _valid_sha256(row.get("receipt_sha256")) or not _valid_sha256(
        row.get("synthetic_realization_id")
    ):
        raise ValueError("v6 row and realization receipts must be SHA-256 values")
    expected_id = _payload_sha256(
        {
            "domain": finite_rows_v6.TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": row["finite_psf_contract"]["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": row["finite_psf_contract"][
                "slab_observation_v4_receipt_sha256"
            ],
        }
    )
    if row.get("training_row_id") != expected_id:
        raise ValueError("v6 training-row ID differs from its exact inputs")
    provenance_receipt = row.get("upstream_reference", {}).get(
        "selected_synthetic_provenance_sha256"
    )
    if not _valid_sha256(provenance_receipt):
        raise ValueError("v6 row is missing its generated-provenance receipt")
    _input_semantics(row)


def _row_receipts(row: Mapping[str, object]) -> dict[str, object]:
    upstream = row["upstream_reference"]
    names = (
        "finite_parent_provenance_sha256",
        "finite_slab_adapter_receipt_sha256",
        "slab_observation_v4_receipt_sha256",
        "selected_synthetic_provenance_sha256",
        "selected_synthetic_lineage_sha256",
    )
    return {
        "training_row_id": row["training_row_id"],
        "training_row_receipt_sha256": row["receipt_sha256"],
        "synthetic_realization_id": row["synthetic_realization_id"],
        "source_observation_receipt_sha256": row[
            "source_observation_receipt_sha256"
        ],
        "finite_psf_sha256": row["finite_psf_contract"]["finite_psf_sha256"],
        **{name: upstream[name] for name in names if name in upstream},
    }


def model_ready_rows_v6(
    rows_or_payload,
    catalogue: Mapping[str, object],
    catalogue_runtime_v6: CompleteCatalogueRuntimeV6,
    atlas_volume,
    *,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    finite_psf_capability,
    expected_training_data_manifest_receipt_sha256,
) -> dict[str, object]:
    """Build one trainer-ready batch without a reduced cell bank or derived features."""
    rows, frozen_source = _rows_and_source(rows_or_payload)
    if (
        not _valid_sha256(expected_training_data_manifest_receipt_sha256)
        or frozen_source["training_data_manifest_receipt_sha256"]
        != expected_training_data_manifest_receipt_sha256
    ):
        raise ValueError("frozen row selection differs from the run-bound data manifest")
    if not rows:
        raise ValueError("a v6 batch requires at least one frozen generated row")
    finite_rows_v6.verify_finite_psf_model_capability_v6(finite_psf_capability)
    verify_complete_catalogue_runtime_v6(catalogue_runtime_v6)
    _verify_catalogue_snapshot(catalogue)
    binding = catalogue_runtime_v6.binding
    if (
        catalogue["counts"]["cell_count"] != FULL_CATALOGUE_CELL_COUNT_V6
        or binding["cell_count"] != FULL_CATALOGUE_CELL_COUNT_V6
        or catalogue["receipt_sha256"] != binding["catalogue_receipt_sha256"]
        or catalogue["catalogue_id"] != binding["catalogue_id"]
    ):
        raise ValueError("v6 rows require the bound complete 98,304-cell catalogue")
    geometry = catalogue.get("support_geometry", {})
    origin = np.asarray(origin_ap_dv_ml_um, dtype=np.float64)
    spacing = np.asarray(voxel_size_ap_dv_ml_um, dtype=np.float64)
    if (
        not np.array_equal(origin, np.asarray(geometry.get("origin_ap_dv_ml_um")))
        or not np.array_equal(
            spacing, np.asarray(geometry.get("voxel_size_ap_dv_ml_um"))
        )
        or tuple(binding["support_origin_ap_dv_ml_um"])
        != tuple(geometry.get("support_origin_ap_dv_ml_um", ()))
    ):
        raise ValueError("atlas geometry differs from the bound catalogue")

    catalogue_batch = catalogue_runtime_v6.expand(len(rows))
    catalogue_tensors = verify_bound_complete_catalogue_batch_v6(
        catalogue_batch, expected_runtime=catalogue_runtime_v6
    )
    target_device = catalogue_tensors["cell_states"].device
    atlas = torch.as_tensor(atlas_volume, device=target_device, dtype=torch.float32)
    if tuple(atlas.shape[-3:]) != tuple(geometry["support_mask_receipt"]["shape"]):
        raise ValueError("atlas spatial shape differs from the catalogue support asset")

    converted = []
    input_audits = []
    for row in rows:
        _verify_row_v6(row, finite_psf_capability)
        converted.append(
            _finite_row_to_tensors_v6(
                row,
                atlas_shape_ap_dv_ml=tuple(atlas.shape[-3:]),
                origin_ap_dv_ml_um=origin,
                voxel_size_ap_dv_ml_um=spacing,
                device=target_device,
                finite_psf_capability=finite_psf_capability,
            )
        )
        input_audits.append(_input_semantics(row))
    shapes = {tuple(item["image"].shape[-2:]) for item in converted}
    signatures = {
        (
            row["finite_psf_contract"]["render_mode"],
            row["finite_psf_contract"]["axial_sample_count"],
        )
        for row in rows
    }
    if shapes != {tuple(geometry["raster_shape_h_w"])}:
        raise ValueError("training-row canvas differs from the catalogue raster contract")
    if len(signatures) != 1:
        raise ValueError("one v6 batch requires one PSF mode and axial sample count")

    truth_state_float64 = torch.stack(
        [
            _physical_state_from_quicknii_ouv_v6(
                row["canonical_effective_quicknii_ouv_float64"],
                tuple(atlas.shape[-3:]),
                origin,
                spacing,
            )
            for row in rows
        ]
    )
    targets = catalogue_truth_targets_v6(truth_state_float64, catalogue)
    pose_weight = torch.cat([item["pose_supervision_weight"] for item in converted])
    dense_weight = torch.cat(
        [item["dense_deformation_supervision_weight"] for item in converted]
    )
    provenance = [
        {
            "specimen_id": row["lineage"]["specimen_id"],
            "animal_id": row["lineage"]["animal_id"],
            "experiment_id": row["lineage"]["experiment_id"],
            "section_id": row["lineage"]["section_id"],
            "synthetic_animal_id": row["lineage"]["synthetic_animal_id"],
            "training_row_id": row["training_row_id"],
            "training_row_receipt_sha256": row["receipt_sha256"],
            "provenance_sha256": row["upstream_reference"][
                "selected_synthetic_provenance_sha256"
            ],
        }
        for row in rows
    ]
    return {
        "schema_version": TRAINING_DATA_V6_SCHEMA,
        "catalogue_batch": catalogue_batch,
        "catalogue_id": catalogue["catalogue_id"],
        "catalogue_receipt_sha256": catalogue["receipt_sha256"],
        "full_catalogue_cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
        "frozen_row_source": frozen_source,
        "provenance": provenance,
        "row_receipts": [_row_receipts(row) for row in rows],
        "input_mode": [item["input_mode"] for item in input_audits],
        "input_semantics_audit": input_audits,
        "image": torch.cat([item["image"] for item in converted]),
        "outline": torch.cat([item["outline"] for item in converted]),
        "outline_available": torch.cat(
            [item["outline_available"] for item in converted]
        ),
        "atlas_volume": atlas,
        "output_shape_h_w": tuple(converted[0]["image"].shape[-2:]),
        "origin_ap_dv_ml_um": tuple(float(value) for value in origin),
        "voxel_size_ap_dv_ml_um": tuple(float(value) for value in spacing),
        "axial_offsets_um": torch.cat(
            [item["axial_offsets_um"] for item in converted]
        ),
        "axial_weights": torch.cat(
            [item["axial_weights"] for item in converted]
        ),
        "finite_psf_contracts": [
            copy.deepcopy(row["finite_psf_contract"]) for row in rows
        ],
        "truth_state": torch.cat([item["truth_state"] for item in converted]),
        "truth_state_float64": truth_state_float64.to(target_device),
        "truth_catalogue_index": targets["truth_catalogue_index"].to(target_device),
        "truth_catalogue_cell_id": targets["truth_catalogue_cell_id"].to(
            target_device
        ),
        "truth_catalogue_residual": targets["truth_catalogue_residual"].to(
            target_device
        ),
        "truth_catalogue_residual_float64": targets[
            "truth_catalogue_residual_float64"
        ].to(target_device),
        "truth_catalogue_aligned_state_float64": targets[
            "truth_catalogue_aligned_state_float64"
        ].to(target_device),
        "truth_catalogue_recomposed_state_float64": targets[
            "truth_catalogue_recomposed_state_float64"
        ].to(target_device),
        "catalogue_truth_mapping_audit": targets[
            "catalogue_truth_mapping_audit"
        ],
        "retrieval_supervision_weight": pose_weight.clone(),
        "pose_supervision_weight": pose_weight,
        "dense_deformation_supervision_weight": dense_weight,
        "truth_stationary_velocity_yx_px": torch.cat(
            [item["truth_stationary_velocity_yx_px"] for item in converted]
        ),
        "truth_pullback_map_yx_px": torch.cat(
            [item["truth_pullback_map_yx_px"] for item in converted]
        ),
        "deformation_weight": torch.cat(
            [item["deformation_weight"] for item in converted]
        ),
    }


__all__ = [
    "CATALOGUE_TRUTH_V6_SCHEMA",
    "FROZEN_ROWS_V6_SCHEMA",
    "FULL_CATALOGUE_CELL_COUNT_V6",
    "MODE_TO_INPUT_V6",
    "TRAINING_DATA_V6_SCHEMA",
    "catalogue_truth_targets_v6",
    "load_frozen_training_rows_v6",
    "model_ready_rows_v6",
]
