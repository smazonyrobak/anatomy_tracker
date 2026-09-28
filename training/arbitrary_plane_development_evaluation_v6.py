"""Receipt-bound all-row internal-development evaluation for v6."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from training import arbitrary_plane_finite_row_binding_v6 as finite_rows_v6
from training import arbitrary_plane_inference_v6 as inference_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
    full_frame_state_to_physical_ouv,
)
from training.arbitrary_plane_geometry import (
    allen_index_to_physical_um_points,
    allen_index_to_physical_um_vectors,
    physical_ouv_to_frame,
    quicknii_to_allen_points,
    quicknii_to_allen_vectors,
)


DEVELOPMENT_EVALUATION_V6_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-development-evaluation/v6"
)
DEVELOPMENT_EVALUATION_BUNDLE_V6_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-development-evaluation-bundle/v6"
)
RAW_PREDICTION_V6_SCHEMA = "anatomy-tracker.raw-development-prediction/v6"
DATA_ROLE = "internal-development-only"
PROBABILITY_STATUS = "raw_uncalibrated_non_release"
FIVE_IDS = tuple(inference_v6.CASE_ID_KEYS_V6)
MODE_TO_INPUT = {
    "smart-brush-absent": "raw",
    "smart-brush-accurate": "black-exterior",
    "smart-brush-imperfect": "imperfect-mask",
}
MODE_LABELS = {
    "smart-brush-absent": "raw acquired background without an outline",
    "smart-brush-accurate": "exact-black smart-brush exterior",
    "smart-brush-imperfect": "imperfect-mask input",
}
CHI2_DF3_QUANTILES = {
    "50": 2.3659738843753377,
    "80": 4.64162767608745,
    "90": 6.251388631170325,
    "95": 7.814727903251179,
}
RELIABILITY_BIN_EDGES = tuple(index / 10.0 for index in range(11))
_SOURCE_FILES = (
    "training/arbitrary_plane_development_evaluation_v6.py",
    "training/arbitrary_plane_inference_v6.py",
    "training/arbitrary_plane_finite_row_binding_v6.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_geometry.py",
)


def _plain(value):
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("v6 development-evaluation receipts require finite values")
    return value


def _canonical_json(value) -> bytes:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_receipts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _SOURCE_FILES}


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and not (set(value) - set("0123456789abcdef"))
    )


def _i_path(value, *, must_exist=False) -> Path:
    path = Path(value).resolve(strict=must_exist)
    if os.path.splitdrive(str(path))[0].upper() != "I:":
        raise ValueError("v6 development-evaluation file I/O is restricted to I:")
    return path


def _atomic_json(path: Path, value) -> None:
    target = _i_path(path)
    if os.path.lexists(target):
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.writing-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _atomic_torch(path: Path, value) -> None:
    target = _i_path(path)
    if os.path.lexists(target):
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.writing-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _tensor_receipt(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "kind": "tensor",
        "torch_dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _tree_receipt(value):
    if isinstance(value, torch.Tensor):
        return _tensor_receipt(value)
    if isinstance(value, Mapping):
        return {
            str(key): _tree_receipt(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_tree_receipt(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_tree_receipt(item) for item in value]}
    if isinstance(value, np.ndarray):
        return _tree_receipt(torch.from_numpy(np.ascontiguousarray(value)))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported raw prediction value {type(value)!r}")


def _raw_prediction_receipt(value) -> str:
    return _sha256_json(_tree_receipt(value))


def _development_manifest(cache_directory, expected_receipt):
    manifest = finite_rows_v6.load_frozen_row_cache_manifest_v6(
        cache_directory,
        expected_manifest_receipt_sha256=expected_receipt,
    )
    records = manifest["rows"]
    if (
        not records
        or manifest["status"] != finite_rows_v6.FROZEN_CACHE_STATUS
        or manifest["generation_lineage"]["split"] != "development"
        or any(record["lineage"]["split"] != "development" for record in records)
    ):
        raise ValueError("evaluation requires one exact frozen development-split cache")
    for record in records:
        if any(not isinstance(record["lineage"].get(key), str) or not record["lineage"][key] for key in FIVE_IDS):
            raise ValueError("every development row must preserve all five exact IDs")
        if record["selected_mode"] not in MODE_TO_INPUT:
            raise ValueError("development cache contains an unsupported input mode")
    return manifest


def _selection_receipt(manifest) -> str:
    payload = {
        "schema_version": finite_rows_v6.FROZEN_ROWS_V6_SCHEMA,
        "training_data_manifest_receipt_sha256": manifest["receipt_sha256"],
        "cache_manifest_receipt_sha256": manifest["receipt_sha256"],
        "generator_binding_receipt_sha256": manifest["generator_binding"]["receipt_sha256"],
        "generation_lineage_sha256": manifest["generator_binding"]["generation_lineage_sha256"],
        "row_indices": list(range(manifest["row_count"])),
        "training_row_ids": [record["training_row_id"] for record in manifest["rows"]],
        "training_row_receipts_sha256": [
            record["training_row_receipt_sha256"] for record in manifest["rows"]
        ],
    }
    return finite_rows_v6.frozen_row_selection_receipt_v6(payload)


def _identity_sets(records) -> dict[str, set[str]]:
    return {key: {record["lineage"][key] for record in records} for key in FIVE_IDS}


def _training_identity_sets(run_manifest) -> dict[str, set[str]]:
    records = run_manifest["training_data"]["ordered_row_identities"]
    return {key: {record[key] for record in records} for key in FIVE_IDS}


def _truth_state(row, atlas_shape, origin, spacing) -> torch.Tensor:
    quicknii = torch.as_tensor(
        row["canonical_effective_quicknii_ouv_float64"], dtype=torch.float64
    ).reshape(3, 3)
    origin = torch.as_tensor(origin, dtype=torch.float64)
    spacing = torch.as_tensor(spacing, dtype=torch.float64)
    physical_ouv = torch.cat(
        (
            allen_index_to_physical_um_points(
                quicknii_to_allen_points(quicknii[0], tuple(atlas_shape)), origin, spacing
            ),
            allen_index_to_physical_um_vectors(quicknii_to_allen_vectors(quicknii[1]), spacing),
            allen_index_to_physical_um_vectors(quicknii_to_allen_vectors(quicknii[2]), spacing),
        )
    )
    center, frame, basis = physical_ouv_to_frame(physical_ouv)
    return full_frame_state_from_components(center, frame, basis)


def _catalogue_metric_context(catalogue) -> dict[str, object]:
    states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"], dtype=torch.float64)
    cell_id = torch.as_tensor(catalogue["arrays"]["cell_id_int64"], dtype=torch.long)
    if states.shape != (cell_id.numel(), 12) or not torch.equal(cell_id, torch.arange(cell_id.numel())):
        raise ValueError("v6 evaluation requires canonical complete catalogue cell IDs")
    center, frame, _ = full_frame_state_to_components(states)
    offset_table = torch.as_tensor(
        catalogue["arrays"]["normal_offset_table_um_float64"], dtype=torch.float64
    )
    return {
        "states": states,
        "center": center,
        "frame": frame,
        "normal": frame[:, :, 2],
        "u": frame[:, :, 0],
        "support_origin": torch.as_tensor(
            catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64
        ),
        "offset_scale": torch.diff(offset_table, dim=1).abs().median().clamp_min(1.0),
        "normal_scale": max(
            float(catalogue["coverage_audit"]["max_observed_rp2_angular_covering_radius_rad"]),
            1.0e-3,
        ),
        "roll_scale": math.pi / int(catalogue["counts"]["roll_count"]),
    }


def _truth_catalogue_index(truth_state, context) -> int:
    truth = torch.as_tensor(truth_state, dtype=torch.float64).reshape(1, 12)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    truth_normal = truth_frame[0, :, 2]
    dot = context["normal"] @ truth_normal
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    normal_angle = torch.acos(dot.abs().clamp(0.0, 1.0))
    truth_offset = torch.dot(truth_center[0] - context["support_origin"], truth_normal)
    candidate_offset = (
        (context["center"] - context["support_origin"]) * context["normal"]
    ).sum(dim=-1)
    offset_error = (truth_offset * sign - candidate_offset).abs()
    aligned_u = torch.where((sign < 0.0)[:, None], -context["u"], context["u"])
    roll_error = torch.acos(
        (aligned_u * truth_frame[0, :, 0]).sum(dim=-1).clamp(-1.0, 1.0)
    )
    cost = (
        (normal_angle / context["normal_scale"]).square()
        + (offset_error / context["offset_scale"]).square()
        + (roll_error / context["roll_scale"]).square()
    )
    return int(cost.argmin())


def _rank(scores: torch.Tensor, index: int) -> int:
    values = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    target = values[index]
    indices = torch.arange(values.numel())
    return int(((values > target) | ((values == target) & (indices < index))).sum()) + 1


def _posterior_and_point(result, catalogue_context, evaluation_stage="joint"):
    posterior = result["posterior"]
    proposal_log = torch.as_tensor(
        posterior["raw_full_catalogue_proposal_log_probability"], dtype=torch.float64
    )[0]
    if evaluation_stage == "proposal":
        cell_count = int(catalogue_context["states"].shape[0])
        if proposal_log.shape != (cell_count,) or not torch.allclose(
            torch.logsumexp(proposal_log, dim=0),
            torch.zeros((), dtype=torch.float64), atol=2e-5, rtol=0.0,
        ):
            raise ValueError("proposal probabilities are incomplete or unnormalized")
        order = torch.argsort(proposal_log, descending=True, stable=True)
        map_index = int(order[0])
        return {
            "proposal_log": proposal_log,
            "hybrid_log": proposal_log,
            "final_log": proposal_log,
            "hybrid_topk": order[:min(32, cell_count)],
            "selected_index": torch.empty(0, dtype=torch.long),
            "retained_probability": 0.0,
            "map_catalogue_index": map_index,
            "map_refined_position": -1,
            "map_state": catalogue_context["states"][map_index],
        }
    hybrid = posterior["honest_hybrid_posterior"]
    hybrid_log = torch.as_tensor(hybrid["hybrid_cell_log_probability"], dtype=torch.float64)[0]
    cell_count = int(catalogue_context["states"].shape[0])
    if proposal_log.shape != (cell_count,) or hybrid_log.shape != (cell_count,) or not torch.allclose(
        torch.logsumexp(proposal_log, dim=0), torch.zeros((), dtype=torch.float64), atol=2e-5, rtol=0.0
    ) or not torch.allclose(
        torch.logsumexp(hybrid_log, dim=0), torch.zeros((), dtype=torch.float64), atol=2e-5, rtol=0.0
    ):
        raise ValueError("v6 raw full-catalogue probabilities are incomplete or unnormalized")
    final_log = hybrid_log.clone()
    selected = result["k_poses"]["catalogue_index"]
    refined = result["recurrent_output"]
    selected_index = torch.empty(0, dtype=torch.long)
    retained = 0.0
    if refined is not None:
        selected_index = torch.as_tensor(selected, dtype=torch.long)[0]
        pose = refined["pose"]
        selected_cell_id = torch.as_tensor(result["k_poses"]["cell_id"], dtype=torch.long)[0]
        if (
            selected_index.ndim != 1
            or not selected_index.numel()
            or bool(((selected_index < 0) | (selected_index >= cell_count)).any())
            or selected_index.unique().numel() != selected_index.numel()
            or not torch.equal(selected_cell_id, selected_index)
            or bool(torch.as_tensor(pose["retrieval_teacher_forced_mask"]).any())
        ):
            raise ValueError("development inference cannot contain teacher forcing")
        conditional = torch.as_tensor(
            pose["conditional_within_topk_cell_log_probability"], dtype=torch.float64
        )[0]
        retained = float(torch.as_tensor(pose["retrieval_topk_retained_probability"])[0])
        if not 0.0 < retained <= 1.0 or conditional.shape != selected_index.shape:
            raise ValueError("v6 refined posterior has an invalid retained partition")
        refined_log = math.log(max(retained, torch.finfo(torch.float64).tiny)) + conditional
        final_log[selected_index] = refined_log
    ready = bool(torch.as_tensor(result["abstention"]["ready_mask"])[0])
    abstained = bool(torch.as_tensor(result["abstention"]["abstained_mask"])[0])
    if ready == abstained or ready != (refined is not None):
        raise ValueError("v6 refinement and abstention dispositions disagree")
    if not torch.allclose(
        torch.logsumexp(final_log, dim=0), torch.zeros((), dtype=torch.float64), atol=3e-5, rtol=0.0
    ):
        raise ValueError("v6 refined posterior did not preserve complete-catalogue mass")
    map_index = int(final_log.argmax())
    match = torch.nonzero(selected_index == map_index, as_tuple=False).flatten()
    refined_position = int(match[0]) if match.numel() else -1
    if refined_position >= 0:
        state = torch.as_tensor(refined["pose"]["final_cell_state"], dtype=torch.float64)[
            0, refined_position
        ]
    else:
        state = catalogue_context["states"][map_index]
    return {
        "proposal_log": proposal_log,
        "hybrid_log": hybrid_log,
        "final_log": final_log,
        "hybrid_topk": torch.as_tensor(hybrid["hybrid_topk_catalogue_index"], dtype=torch.long)[0],
        "selected_index": selected_index,
        "retained_probability": retained,
        "map_catalogue_index": map_index,
        "map_refined_position": refined_position,
        "map_state": state,
    }


def _physical_landmarks(state: torch.Tensor) -> torch.Tensor:
    ouv = full_frame_state_to_physical_ouv(torch.as_tensor(state, dtype=torch.float64))
    origin, edge_u, edge_v = ouv[:3], ouv[3:6], ouv[6:9]
    return torch.stack(
        (origin, origin + edge_u, origin + edge_v, origin + edge_u + edge_v, origin + 0.5 * (edge_u + edge_v))
    )


def _pose_metrics(prediction, truth, support_origin) -> dict[str, object]:
    prediction = torch.as_tensor(prediction, dtype=torch.float64).reshape(1, 12)
    truth = torch.as_tensor(truth, dtype=torch.float64).reshape(1, 12)
    landmark_error = torch.linalg.vector_norm(
        _physical_landmarks(prediction[0]) - _physical_landmarks(truth[0]), dim=-1
    )
    pred_center, pred_frame, pred_basis = full_frame_state_to_components(prediction)
    truth_center, truth_frame, truth_basis = full_frame_state_to_components(truth)
    pred_normal = pred_frame[0, :, 2]
    truth_normal = truth_frame[0, :, 2]
    dot = torch.dot(pred_normal, truth_normal).clamp(-1.0, 1.0)
    sign = -1.0 if float(dot) < 0.0 else 1.0
    angle = torch.acos(dot.abs())
    origin = torch.as_tensor(support_origin, dtype=torch.float64)
    pred_offset = torch.dot(pred_center[0] - origin, pred_normal)
    truth_offset = sign * torch.dot(truth_center[0] - origin, truth_normal)
    relative = pred_frame[0].transpose(0, 1) @ truth_frame[0]
    frame_angle = torch.acos(((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0))
    center_delta = pred_center[0] - truth_center[0]
    center_error = torch.linalg.vector_norm(center_delta)
    basis_error = torch.linalg.matrix_norm(pred_basis[0] - truth_basis[0])
    return {
        "primary_internal_development_metric": "physical_finite_frame_landmark_mean_um",
        "physical_finite_frame_landmark_error_um": landmark_error.tolist(),
        "physical_finite_frame_landmark_mean_um": float(landmark_error.mean()),
        "physical_finite_frame_landmark_rms_um": float(landmark_error.square().mean().sqrt()),
        "physical_finite_frame_landmark_max_um": float(landmark_error.max()),
        "center_distance_error_um": float(center_error),
        "signed_center_error_ap_dv_ml_um": center_delta.tolist(),
        "absolute_center_error_ap_dv_ml_um": center_delta.abs().tolist(),
        "inplane_basis_frobenius_error_um": float(basis_error),
        "plane_normal_projective_error_rad": float(angle),
        "plane_normal_projective_error_deg": float(torch.rad2deg(angle)),
        "signed_plane_offset_error_um": float(pred_offset - truth_offset),
        "absolute_signed_plane_offset_error_um": abs(float(pred_offset - truth_offset)),
        "finite_frame_rotation_error_rad": float(frame_angle),
        "finite_frame_rotation_error_deg": float(torch.rad2deg(frame_angle)),
    }


def _plane_residual(prediction, truth, support_origin) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64).reshape(1, 12)
    truth = torch.as_tensor(truth, dtype=torch.float64).reshape(1, 12)
    pred_center, pred_frame, _ = full_frame_state_to_components(prediction)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    pred_u, pred_v, pred_normal = pred_frame[0].unbind(dim=-1)
    truth_normal = truth_frame[0, :, 2]
    dot = torch.dot(pred_normal, truth_normal)
    sign = -1.0 if float(dot) < 0.0 else 1.0
    aligned = sign * truth_normal
    cosine = torch.dot(pred_normal, aligned).clamp(-1.0, 1.0)
    direction = aligned - cosine * pred_normal
    norm = torch.linalg.vector_norm(direction)
    angle = torch.atan2(norm, cosine)
    tangent = direction * torch.where(norm > 1.0e-4, angle / norm.clamp_min(1.0e-7), 1.0 + norm.square() / 6.0)
    origin = torch.as_tensor(support_origin, dtype=torch.float64)
    pred_offset = torch.dot(pred_center[0] - origin, pred_normal)
    truth_offset = sign * torch.dot(truth_center[0] - origin, truth_normal)
    return torch.stack((torch.dot(tangent, pred_u), torch.dot(tangent, pred_v), truth_offset - pred_offset))


def _field_error(prediction, truth, valid) -> dict[str, object]:
    error = torch.linalg.vector_norm(prediction - truth, dim=0)[valid]
    if not error.numel():
        return {"valid_pixel_count": 0, "mean": None, "rms": None, "maximum": None}
    return {
        "valid_pixel_count": int(error.numel()),
        "mean": float(error.mean()),
        "rms": float(error.square().mean().sqrt()),
        "maximum": float(error.max()),
    }


def _deformation_metrics(result, position, row, minimum_jacobian, maximum_cycle_error):
    support = row["upstream_reference"].get("support_supervision_contract", {})
    identifiable = float(support.get("dense_deformation_supervision_weight", 1.0)) > 0.0
    if position < 0 or not identifiable:
        return {
            "reference_identifiable": identifiable,
            "available": False,
            "reason": "MAP state was not refined" if position < 0 else "row has no identifiable dense reference",
            "valid_pixel_count": 0,
            "svf_mean_endpoint_error_px": None,
            "map_mean_endpoint_error_px": None,
            "minimum_forward_jacobian": None,
            "topology_failure": False,
            "cycle_failure": False,
            "deformation_failure": False,
        }
    refined = result["recurrent_output"]
    arrays = row["arrays"]
    valid = torch.from_numpy(
        np.asarray(arrays["truth_section_deformation_valid_mask"], dtype=bool)
        & np.asarray(arrays["target_valid_correspondence_mask"], dtype=bool)
        & ~np.asarray(arrays["target_correspondence_abstention_mask"], dtype=bool)
    )
    truth_velocity = torch.from_numpy(
        np.asarray(arrays["truth_section_pullback_stationary_velocity_yx_px_float64"])
    ).permute(2, 0, 1).to(torch.float64)
    truth_map = torch.from_numpy(
        np.asarray(arrays["truth_section_pullback_map_yx_px_float64"])
    ).permute(2, 0, 1).to(torch.float64)
    velocity = torch.as_tensor(refined["final_stationary_velocity_yx_px"])[0, position].to(torch.float64)
    pullback = torch.as_tensor(refined["final_pullback_map_yx_px"])[0, position].to(torch.float64)
    svf = _field_error(velocity, truth_velocity, valid)
    mapping = _field_error(pullback, truth_map, valid)
    jacobian = torch.as_tensor(refined["final_forward_jacobian_determinant"])[
        0, position, 0
    ].to(torch.float64)[valid]
    forward_valid = torch.as_tensor(refined["final_forward_then_inverse_valid_mask"])[
        0, position, 0
    ].bool() & valid
    inverse_valid = torch.as_tensor(refined["final_inverse_then_forward_valid_mask"])[
        0, position, 0
    ].bool() & valid
    forward = torch.linalg.vector_norm(
        torch.as_tensor(refined["final_forward_then_inverse_error_yx"])[0, position].to(torch.float64), dim=0
    )[forward_valid]
    inverse = torch.linalg.vector_norm(
        torch.as_tensor(refined["final_inverse_then_forward_error_yx"])[0, position].to(torch.float64), dim=0
    )[inverse_valid]
    topology_failure = bool(not jacobian.numel() or (jacobian < minimum_jacobian).any())
    cycle_failure = bool(
        not forward.numel()
        or not inverse.numel()
        or (forward > maximum_cycle_error).any()
        or (inverse > maximum_cycle_error).any()
    )
    return {
        "reference_identifiable": True,
        "available": True,
        "valid_pixel_count": int(valid.sum()),
        "svf_mean_endpoint_error_px": svf["mean"],
        "svf_rms_endpoint_error_px": svf["rms"],
        "svf_max_endpoint_error_px": svf["maximum"],
        "map_mean_endpoint_error_px": mapping["mean"],
        "map_rms_endpoint_error_px": mapping["rms"],
        "map_max_endpoint_error_px": mapping["maximum"],
        "minimum_forward_jacobian": None if not jacobian.numel() else float(jacobian.min()),
        "nonpositive_jacobian_fraction": None if not jacobian.numel() else float((jacobian <= 0).double().mean()),
        "below_minimum_jacobian_fraction": None if not jacobian.numel() else float((jacobian < minimum_jacobian).double().mean()),
        "minimum_jacobian_threshold": float(minimum_jacobian),
        "forward_cycle_rms_error_px": None if not forward.numel() else float(forward.square().mean().sqrt()),
        "forward_cycle_max_error_px": None if not forward.numel() else float(forward.max()),
        "inverse_cycle_rms_error_px": None if not inverse.numel() else float(inverse.square().mean().sqrt()),
        "inverse_cycle_max_error_px": None if not inverse.numel() else float(inverse.max()),
        "maximum_cycle_error_threshold_px": float(maximum_cycle_error),
        "cycle_invalid_fraction": None if not valid.any() else float(1.0 - (forward_valid & inverse_valid).double().sum() / valid.sum()),
        "topology_failure": topology_failure,
        "cycle_failure": cycle_failure,
        "deformation_failure": bool(topology_failure or cycle_failure),
    }


def _overlap_metrics(result, position, row) -> dict[str, object]:
    regional = {
        "available": False,
        "reason": "the authenticated v6 run embeds atlas support but not regional annotation labels",
        "mean_dice_foreground": None,
    }
    if position < 0:
        return {
            "tissue_support": {"available": False, "reason": "MAP state was not refined", "dice": None, "iou": None},
            "regional": regional,
        }
    rendered = torch.as_tensor(result["recurrent_output"]["final_deformed_canonical_render"])
    if rendered.ndim != 5 or rendered.shape[2] < 2:
        return {
            "tissue_support": {"available": False, "reason": "render has no atlas-support channel", "dice": None, "iou": None},
            "regional": regional,
        }
    prediction = rendered[0, position, 1] >= 0.5
    truth = torch.from_numpy(np.asarray(row["arrays"]["source_tissue_ground_truth_mask"], dtype=bool))
    intersection = int((prediction & truth).sum())
    pred_count = int(prediction.sum())
    truth_count = int(truth.sum())
    union = int((prediction | truth).sum())
    denominator = pred_count + truth_count
    return {
        "tissue_support": {
            "available": True,
            "method": "threshold 0.5 on the finite-thickness rendered Allen support channel",
            "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
            "iou": 1.0 if union == 0 else intersection / union,
            "predicted_foreground_pixels": pred_count,
            "reference_foreground_pixels": truth_count,
        },
        "regional": regional,
    }


def _uncertainty_metrics(posterior, truth_index, truth_state, result, support_origin):
    final_log = posterior["final_log"]
    probability = final_log.exp()
    map_index = posterior["map_catalogue_index"]
    top_confidence = float(probability[map_index])
    truth_log_probability = float(final_log[truth_index])
    brier = float(probability.square().sum() - 2.0 * probability[truth_index] + 1.0)
    entropy = float(-(probability * final_log).sum())
    position = posterior["map_refined_position"]
    local = {
        "available": False,
        "scope": "raw MAP-component local plane Gaussian; excludes unrefined catalogue tail",
        "mahalanobis_squared_df3": None,
        "coverage": {key: None for key in CHI2_DF3_QUANTILES},
        "covariance_trace": None,
        "covariance_max_eigenvalue": None,
    }
    if position >= 0:
        refined = result["recurrent_output"]["pose"]
        covariance = torch.as_tensor(
            refined["final_cell_canonical_plane_covariance"], dtype=torch.float64
        )[0, position]
        residual = _plane_residual(
            posterior["map_state"], truth_state, support_origin
        )
        regularized = covariance + 1.0e-6 * torch.eye(3, dtype=torch.float64)
        cholesky = torch.linalg.cholesky(regularized)
        standardized = torch.linalg.solve_triangular(
            cholesky, residual[:, None], upper=False
        )[:, 0]
        mahalanobis = float(standardized.square().sum())
        eigenvalues = torch.linalg.eigvalsh(covariance)
        local = {
            "available": True,
            "scope": "raw MAP-component local plane Gaussian; excludes unrefined catalogue tail",
            "coordinate_units": ["radian", "radian", "micrometre"],
            "mahalanobis_squared_df3": mahalanobis,
            "coverage": {
                key: mahalanobis <= threshold for key, threshold in CHI2_DF3_QUANTILES.items()
            },
            "covariance_trace": float(torch.trace(covariance)),
            "covariance_max_eigenvalue": float(eigenvalues.max()),
        }
    return {
        "probabilities_calibrated": False,
        "status": PROBABILITY_STATUS,
        "calibration_fitted_by_evaluator": False,
        "coverage_claimed": False,
        "categorical_truth_nll_nats": -truth_log_probability,
        "categorical_brier_score": brier,
        "raw_retrieval_entropy_nats": entropy,
        "raw_normalized_retrieval_entropy": entropy / math.log(max(probability.numel(), 2)),
        "raw_top1_confidence": top_confidence,
        "raw_top1_correct": map_index == truth_index,
        "raw_truth_cell_probability": float(probability[truth_index]),
        "raw_refined_probability_mass": float(posterior["retained_probability"]),
        "raw_unrefined_probability_mass": 1.0 - float(posterior["retained_probability"]),
        "local_plane_gaussian": local,
    }


def _row_metrics(result, row, catalogue, catalogue_context, atlas_shape, config):
    geometry = catalogue["support_geometry"]
    truth_state = _truth_state(
        row,
        atlas_shape,
        geometry["origin_ap_dv_ml_um"],
        geometry["voxel_size_ap_dv_ml_um"],
    )
    truth_index = _truth_catalogue_index(truth_state, catalogue_context)
    proposal_only = config["evaluation_stage"] == "proposal"
    posterior = _posterior_and_point(result, catalogue_context, config["evaluation_stage"])
    proposal_rank = _rank(posterior["proposal_log"], truth_index)
    topk = posterior["hybrid_topk"]
    topk_match = torch.nonzero(topk == truth_index, as_tuple=False).flatten()
    position = posterior["map_refined_position"]
    deformation = _deformation_metrics(
        result,
        position,
        row,
        config["minimum_jacobian"],
        config["maximum_cycle_error_px"],
    )
    operational_abstention = False if proposal_only else bool(
        position < 0 or torch.as_tensor(result["abstention"]["abstained_mask"])[0]
    )
    reference_failure = not bool(topk_match.numel())
    overall_failure = reference_failure if proposal_only else bool(
        operational_abstention or reference_failure or deformation["deformation_failure"]
    )
    if proposal_only:
        deformation["deformation_failure"] = None
    return {
        "pose": _pose_metrics(posterior["map_state"], truth_state, catalogue_context["support_origin"]),
        "retrieval": {
            "catalogue_complete": True,
            "full_catalogue_cell_count": int(posterior["final_log"].numel()),
            "truth_catalogue_index": truth_index,
            "truth_catalogue_cell_id": truth_index,
            "proposal_truth_rank": proposal_rank,
            "proposal_truth_nll_nats": -float(posterior["proposal_log"][truth_index]),
            "proposal_top1_recall": proposal_rank <= 1,
            "proposal_top8_recall": proposal_rank <= 8,
            "proposal_top32_recall": proposal_rank <= 32,
            "hybrid_truth_rank": None if proposal_only else _rank(posterior["hybrid_log"], truth_index),
            "final_truth_rank": _rank(posterior["final_log"], truth_index),
            "honest_topk_catalogue_indices": topk.tolist(),
            "topk_scope": "proposal_top32" if proposal_only else "joint_hybrid_topk",
            "topk_recall": bool(topk_match.numel()),
            "truth_rank_within_topk": None if not topk_match.numel() else int(topk_match[0]) + 1,
            "map_catalogue_index": posterior["map_catalogue_index"],
            "map_component_refined": position >= 0,
            "raw_unrefined_probability_mass": 1.0 - posterior["retained_probability"],
            "reference_failure": reference_failure,
        },
        "deformation": deformation,
        "overlap": _overlap_metrics(result, position, row),
        "uncertainty": _uncertainty_metrics(
            posterior,
            truth_index,
            truth_state,
            result,
            catalogue_context["support_origin"],
        ),
        "operational_abstention": operational_abstention,
        "overall_failure": overall_failure,
    }


def _nested(record, path):
    value = record
    for name in path.split("."):
        value = value.get(name) if isinstance(value, dict) else None
    if isinstance(value, bool):
        return float(value)
    return value if isinstance(value, (int, float)) else None


ANIMAL_METRICS = (
    "pose.physical_finite_frame_landmark_mean_um",
    "pose.center_distance_error_um",
    "pose.plane_normal_projective_error_deg",
    "pose.absolute_signed_plane_offset_error_um",
    "pose.finite_frame_rotation_error_deg",
    "retrieval.proposal_truth_rank",
    "retrieval.proposal_truth_nll_nats",
    "retrieval.proposal_top1_recall",
    "retrieval.proposal_top8_recall",
    "retrieval.proposal_top32_recall",
    "retrieval.hybrid_truth_rank",
    "retrieval.final_truth_rank",
    "retrieval.topk_recall",
    "retrieval.reference_failure",
    "deformation.svf_mean_endpoint_error_px",
    "deformation.map_mean_endpoint_error_px",
    "deformation.deformation_failure",
    "overlap.tissue_support.dice",
    "uncertainty.categorical_truth_nll_nats",
    "uncertainty.categorical_brier_score",
    "uncertainty.raw_normalized_retrieval_entropy",
    "uncertainty.raw_top1_correct",
    "operational_abstention",
    "overall_failure",
)


def _summary(values) -> dict[str, object]:
    values = [float(value) for value in values if value is not None]
    return {
        "eligible_count": len(values),
        "mean": None if not values else float(np.mean(values)),
        "minimum": None if not values else float(np.min(values)),
        "maximum": None if not values else float(np.max(values)),
    }


def _animal_macro(row_reports) -> dict[str, object]:
    grouped = {}
    for report in row_reports:
        grouped.setdefault(report["animal_id"], []).append(report)
    per_animal = {}
    for animal_id, rows in sorted(grouped.items()):
        per_animal[animal_id] = {
            "row_count": len(rows),
            "metric_means": {
                path: _summary([_nested(row["metrics"], path) for row in rows])["mean"]
                for path in ANIMAL_METRICS
            },
        }
    return {
        "statistical_unit": "animal",
        "animal_count": len(per_animal),
        "per_animal": per_animal,
        "macro_across_animals": {
            path: _summary([record["metric_means"][path] for record in per_animal.values()])
            for path in ANIMAL_METRICS
        },
        "confidence_interval_status": "not estimated in internal development; reserved for predefined animal-level validation",
    }


def _calibration_diagnostics(row_reports) -> dict[str, object]:
    animal_counts = {}
    for row in row_reports:
        animal_counts[row["animal_id"]] = animal_counts.get(row["animal_id"], 0) + 1
    weights = np.asarray([
        1.0 / (len(animal_counts) * animal_counts[row["animal_id"]])
        for row in row_reports
    ])
    confidence = np.asarray(
        [row["metrics"]["uncertainty"]["raw_top1_confidence"] for row in row_reports], dtype=np.float64
    )
    correct = np.asarray(
        [row["metrics"]["uncertainty"]["raw_top1_correct"] for row in row_reports], dtype=np.float64
    )
    bins = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, (lower, upper) in enumerate(zip(RELIABILITY_BIN_EDGES[:-1], RELIABILITY_BIN_EDGES[1:])):
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == len(RELIABILITY_BIN_EDGES) - 2 else (confidence < upper)
        )
        count = int(selected.sum())
        mean_confidence = None if not count else float(np.average(confidence[selected], weights=weights[selected]))
        accuracy = None if not count else float(np.average(correct[selected], weights=weights[selected]))
        gap = None if not count else abs(mean_confidence - accuracy)
        weighted_gap += 0.0 if gap is None else float(weights[selected].sum()) * gap
        maximum_gap = max(maximum_gap, 0.0 if gap is None else gap)
        bins.append(
            {
                "lower_inclusive": lower,
                "upper_inclusive": index == len(RELIABILITY_BIN_EDGES) - 2,
                "upper": upper,
                "row_count": count,
                "animal_balanced_weight": float(weights[selected].sum()),
                "mean_confidence": mean_confidence,
                "top1_accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    local_rows = [
        row["metrics"]["uncertainty"]["local_plane_gaussian"]
        for row in row_reports
        if row["metrics"]["uncertainty"]["local_plane_gaussian"]["available"]
    ]
    local_animal_coverage = {}
    for row in row_reports:
        local = row["metrics"]["uncertainty"]["local_plane_gaussian"]
        if local["available"]:
            local_animal_coverage.setdefault(row["animal_id"], []).append(local["coverage"])
    return {
        "status": PROBABILITY_STATUS,
        "post_hoc_fitting_performed": False,
        "coverage_claimed": False,
        "weighting": "each animal has equal total weight; sections share their animal weight",
        "fixed_top1_reliability_bins": bins,
        "top1_expected_calibration_error": weighted_gap,
        "top1_maximum_calibration_error": maximum_gap,
        "animal_macro_categorical_truth_nll_nats": float(np.dot(weights, [
            row["metrics"]["uncertainty"]["categorical_truth_nll_nats"] for row in row_reports
        ])),
        "animal_macro_categorical_brier_score": float(np.dot(weights, [
            row["metrics"]["uncertainty"]["categorical_brier_score"] for row in row_reports
        ])),
        "local_plane_gaussian": {
            "scope": "raw MAP-component df=3 ellipsoid only; excludes unrefined catalogue tail",
            "eligible_row_count": len(local_rows),
            "eligible_animal_count": len(local_animal_coverage),
            "chi_square_df3_thresholds": CHI2_DF3_QUANTILES,
            "empirical_coverage": {
                key: None if not local_animal_coverage else float(np.mean([
                    np.mean([coverage[key] for coverage in animal_rows])
                    for animal_rows in local_animal_coverage.values()
                ]))
                for key in CHI2_DF3_QUANTILES
            },
        },
    }


def _mode_stratified(row_reports) -> dict[str, object]:
    result = {}
    for mode in MODE_TO_INPUT:
        rows = [row for row in row_reports if row["selected_mode"] == mode]
        result[mode] = {
            "input_condition": MODE_LABELS[mode],
            "row_count": len(rows),
            "animal_count": len({row["animal_id"] for row in rows}),
            "failed_row_count": sum(row["disposition"]["failed"] for row in rows),
            "abstained_row_count": sum(row["disposition"]["abstained"] for row in rows),
            "animal_macro_metrics": _animal_macro(rows),
            "raw_calibration_diagnostics": None if not rows else _calibration_diagnostics(rows),
        }
    return result


def _input_arguments(row, catalogue, evaluation_stage="joint"):
    channels = np.asarray(row["arrays"]["model_input_channels_float32"])
    if channels.ndim != 3 or channels.shape[2] != 3:
        raise ValueError("v6 development input must have exact HxWx3 model channels")
    availability = np.unique(channels[:, :, 2])
    mode = row["selected_mode"]
    expected_available = mode != "smart-brush-absent"
    if availability.size != 1 or float(availability[0]) != float(expected_available):
        raise ValueError("cached outline availability differs from its selected input mode")
    height, width = channels.shape[:2]
    span = tuple(float(value) for value in catalogue["support_geometry"]["raster_physical_span_y_x_um"])
    contract = row["finite_psf_contract"]
    return {
        "model_input_channels_float32": channels,
        "expected_model_input_receipt": row["array_receipts"]["model_input_channels_float32"],
        "prepared_source_receipt_sha256": row["receipt_sha256"],
        "input_mode": MODE_TO_INPUT[mode],
        "proposal_only": evaluation_stage == "proposal",
        "physical_fov_y_x_um": span,
        "pixel_size_y_x_um": (span[0] / height, span[1] / width),
        "nominal_cut_thickness_um": contract["nominal_cut_thickness_um"],
        "axial_offsets_um": contract["axial_offsets_um"],
        "axial_weights": contract["axial_weights"],
        "case_ids": {key: row["lineage"][key] for key in FIVE_IDS},
    }


def _expected_input_receipt(row, catalogue):
    arguments = _input_arguments(row, catalogue)
    channels = arguments["model_input_channels_float32"]
    payload = {
        "input_mode": arguments["input_mode"],
        "brush_available": bool(channels[0, 0, 2]),
        "input_source": "authenticated_frozen_prepared_channels",
        "prepared_source_receipt_sha256": row["receipt_sha256"],
        "model_input_channels_receipt": arguments["expected_model_input_receipt"],
        "model_outline": inference_v6._array_receipt(channels[:, :, 1]),
        "model_image": inference_v6._array_receipt(channels[:, :, 0]),
        "physical_fov_y_x_um": list(arguments["physical_fov_y_x_um"]),
        "pixel_size_y_x_um": list(arguments["pixel_size_y_x_um"]),
        "nominal_cut_thickness_um": float(arguments["nominal_cut_thickness_um"]),
        "axial_offsets_um": np.asarray(arguments["axial_offsets_um"], dtype=np.float64).tolist(),
        "axial_weights": np.asarray(arguments["axial_weights"], dtype=np.float64).tolist(),
        "case_ids": arguments["case_ids"],
    }
    return {**payload, "receipt_sha256": inference_v6._sha256_json(payload)}


def _run_binding(loaded) -> dict[str, object]:
    context = loaded["context"]
    manifest = context["manifest"]
    return {
        "run_directory": str(Path(context["run_directory"]).resolve()),
        "run_manifest_receipt_sha256": loaded["run_manifest_receipt_sha256"],
        "run_state_receipt_sha256": loaded["run_state_receipt_sha256"],
        "checkpoint_receipt_sha256": loaded["checkpoint_receipt_sha256"],
        "checkpoint_model_state_sha256": loaded["checkpoint_model_state_sha256"],
        "catalogue_id": context["catalogue"]["catalogue_id"],
        "catalogue_receipt_sha256": context["catalogue"]["receipt_sha256"],
        "catalogue_cell_count": int(context["catalogue"]["counts"]["cell_count"]),
        "training_data_manifest_receipt_sha256": manifest["training_data"][
            "training_data_manifest_receipt_sha256"
        ],
        "training_git_commit": manifest["git_commit"],
        "training_source_sha256": manifest["source_sha256"],
        "initialization": manifest["initialization"],
        "prior_model_weight_dependencies": manifest["prior_model_weight_dependencies"],
        "prior_feature_dependencies": manifest["prior_feature_dependencies"],
        "prior_pseudolabel_dependencies": manifest["prior_pseudolabel_dependencies"],
    }


def _inference_result_run_binding(loaded) -> dict[str, str]:
    return {
        "run_manifest_receipt_sha256": loaded["run_manifest_receipt_sha256"],
        "run_state_receipt_sha256": loaded["run_state_receipt_sha256"],
        "checkpoint_receipt_sha256": loaded["checkpoint_receipt_sha256"],
        "checkpoint_model_state_sha256": loaded["checkpoint_model_state_sha256"],
        "catalogue_receipt_sha256": loaded["context"]["catalogue"]["receipt_sha256"],
    }


def run_arbitrary_plane_development_evaluation_v6(
    cache_directory,
    run_directory,
    output_directory,
    *,
    expected_cache_manifest_receipt_sha256: str,
    expected_run_manifest_receipt_sha256: str,
    expected_inference_source_sha256: Mapping[str, str],
    evaluation_stage: str = "joint",
    row_chunk_size: int = 8,
    minimum_jacobian: float = 0.05,
    maximum_cycle_error_px: float = 1.0,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Evaluate every authenticated development row without fitting calibration."""
    if evaluation_stage not in {"proposal", "joint"}:
        raise ValueError("evaluation_stage must be proposal or joint")
    if not _valid_sha256(expected_cache_manifest_receipt_sha256) or not _valid_sha256(
        expected_run_manifest_receipt_sha256
    ):
        raise ValueError("trusted lowercase SHA-256 cache and run receipts are required")
    if not isinstance(row_chunk_size, int) or isinstance(row_chunk_size, bool) or row_chunk_size < 1:
        raise ValueError("row chunk size must be a positive integer")
    if minimum_jacobian <= 0.0 or maximum_cycle_error_px <= 0.0:
        raise ValueError("deformation failure thresholds must be positive")
    cache_root = _i_path(cache_directory, must_exist=True)
    run_root = _i_path(run_directory, must_exist=True)
    output_root = _i_path(output_directory)
    if os.path.lexists(output_root):
        raise FileExistsError("v6 development-evaluation output must not already exist")
    source = _source_receipts()
    manifest = _development_manifest(cache_root, expected_cache_manifest_receipt_sha256)
    loaded = inference_v6.load_arbitrary_plane_inference_v6(
        run_root,
        expected_run_manifest_receipt_sha256=expected_run_manifest_receipt_sha256,
        expected_inference_source_sha256=expected_inference_source_sha256,
        device=device,
    )
    catalogue = loaded["context"]["catalogue"]
    catalogue_context = _catalogue_metric_context(catalogue)
    run_binding = _run_binding(loaded)
    development_ids = _identity_sets(manifest["rows"])
    training_ids = _training_identity_sets(loaded["context"]["manifest"])
    overlap = {key: sorted(development_ids[key] & training_ids[key]) for key in FIVE_IDS}
    if any(overlap.values()):
        raise ValueError("development cache identity leaked into the training run")
    config = {
        "evaluation_stage": evaluation_stage,
        "row_chunk_size": row_chunk_size,
        "minimum_jacobian": float(minimum_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "reliability_bin_edges": list(RELIABILITY_BIN_EDGES),
        "local_plane_chi_square_df3_quantiles": CHI2_DF3_QUANTILES,
        "inference_batch_size": 1,
        "all_cache_rows_evaluated": True,
        "post_hoc_calibration_fitting": False,
        "automatic_segmentation": False,
        "model_input_source": "authenticated frozen image/boundary/availability channels",
        "brush_mask_reconstructed": False,
    }
    staging = output_root.with_name(
        f".{output_root.name}.writing-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    raw_root = staging / "raw_predictions"
    raw_root.mkdir()
    row_reports = []
    atlas_shape = tuple(np.asarray(loaded["context"]["atlas_volume"]).shape[-3:])
    try:
        for start in range(0, manifest["row_count"], row_chunk_size):
            indices = list(range(start, min(start + row_chunk_size, manifest["row_count"])))
            selection = finite_rows_v6.load_frozen_training_rows_v6(
                cache_root,
                indices,
                expected_manifest_receipt_sha256=expected_cache_manifest_receipt_sha256,
            )
            for cache_index, row in zip(indices, selection["rows"]):
                arguments = _input_arguments(row, catalogue, evaluation_stage)
                result = inference_v6.run_arbitrary_plane_prepared_inference_v6(loaded, **arguments)
                if (
                    result.get("probabilities_calibrated") is not False
                    or result.get("evaluation_stage") != evaluation_stage
                    or result.get("probability_status") != "raw_uncalibrated"
                    or result.get("run_binding") != _inference_result_run_binding(loaded)
                    or result.get("trusted_inference_source_sha256") != dict(expected_inference_source_sha256)
                    or result.get("input_receipt") != _expected_input_receipt(row, catalogue)
                ):
                    raise RuntimeError("truth-free inference result changed its authenticated contract")
                raw_receipt = _raw_prediction_receipt(result)
                artifact_payload = {
                    "schema_version": RAW_PREDICTION_V6_SCHEMA,
                    "cache_row_index": cache_index,
                    "training_row_id": row["training_row_id"],
                    "training_row_receipt_sha256": row["receipt_sha256"],
                    "identifiers": {key: row["lineage"][key] for key in FIVE_IDS},
                    "selected_mode": row["selected_mode"],
                    "input_mode": arguments["input_mode"],
                    "run_binding": result["run_binding"],
                    "raw_prediction_receipt_sha256": raw_receipt,
                }
                artifact = {
                    **artifact_payload,
                    "artifact_receipt_sha256": _sha256_json(artifact_payload),
                    "raw_inference_result": result,
                }
                raw_path = raw_root / f"row_{cache_index:06d}_{row['receipt_sha256'][:16]}.pt"
                _atomic_torch(raw_path, artifact)
                metrics = _row_metrics(result, row, catalogue, catalogue_context, atlas_shape, config)
                lineage = row["lineage"]
                row_reports.append(
                    {
                        "evaluation_order": len(row_reports),
                        "cache_row_index": cache_index,
                        "training_row_id": row["training_row_id"],
                        "training_row_receipt_sha256": row["receipt_sha256"],
                        "synthetic_realization_id": row["synthetic_realization_id"],
                        **{key: lineage[key] for key in FIVE_IDS},
                        "split": lineage["split"],
                        "selected_mode": row["selected_mode"],
                        "input_mode": arguments["input_mode"],
                        "finite_psf_sha256": row["finite_psf_contract"]["finite_psf_sha256"],
                        "raw_prediction": {
                            "relative_path": raw_path.relative_to(staging).as_posix(),
                            "file_sha256": _file_sha256(raw_path),
                            "byte_count": raw_path.stat().st_size,
                            "artifact_receipt_sha256": artifact["artifact_receipt_sha256"],
                            "raw_prediction_receipt_sha256": raw_receipt,
                        },
                        "disposition": {
                            "included_in_all_row_metrics": True,
                            "included_in_animal_macro_where_defined": True,
                            "failed": metrics["overall_failure"],
                            "abstained": metrics["operational_abstention"],
                            "no_silent_drop": True,
                        },
                        "metrics": metrics,
                    }
                )
                del result, artifact
        if _source_receipts() != source:
            raise RuntimeError("evaluation source changed while predictions were produced")
        identities = {
            key: [row[key] for row in row_reports] for key in FIVE_IDS
        }
        failed = [row["training_row_id"] for row in row_reports if row["disposition"]["failed"]]
        abstained = [row["training_row_id"] for row in row_reports if row["disposition"]["abstained"]]
        payload = {
            "schema_version": DEVELOPMENT_EVALUATION_V6_SCHEMA,
            "data_role": DATA_ROLE,
            "scientific_scope": "internal synthetic animal-disjoint development only; not calibration, public benchmark, external validation, qualification, or final test",
            "probability_status": PROBABILITY_STATUS,
            "probabilities_calibrated": False,
            "calibration_fitted": False,
            "release_qualifying": False,
            "public_benchmark_accessed": False,
            "external_validation_accessed": False,
            "final_test_accessed": False,
            "automatic_segmentation_used": False,
            "source_sha256": source,
            "trusted_inference_source_sha256": dict(expected_inference_source_sha256),
            "configuration": config,
            "configuration_receipt_sha256": _sha256_json(config),
            "cache_binding": {
                "directory": str(cache_root),
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "generator_binding_receipt_sha256": manifest["generator_binding"]["receipt_sha256"],
                "generation_lineage_sha256": manifest["generator_binding"]["generation_lineage_sha256"],
                "complete_row_selection_receipt_sha256": _selection_receipt(manifest),
                "row_count": manifest["row_count"],
                "split": "development",
                "freeze_audit": manifest["freeze_audit"],
            },
            "run_binding": run_binding,
            "identity_overlap_with_training": overlap,
            "identities": identities,
            "row_accounting": {
                "expected_row_count": manifest["row_count"],
                "reported_row_count": len(row_reports),
                "no_rows_dropped": len(row_reports) == manifest["row_count"],
                "failed_row_count": len(failed),
                "failed_training_row_ids": failed,
                "abstained_row_count": len(abstained),
                "abstained_training_row_ids": abstained,
            },
            "metric_families": [
                "physical_finite_frame_geometric_landmarks_not_expert_anatomical_landmarks",
                "plane_offset",
                "plane_angle",
                "finite_frame_angle",
                "catalogue_hit_and_rank",
                "identifiable_dense_deformation",
                "tissue_support_overlap",
                "failures_and_abstentions",
                "input_mode_stratification",
                "raw_uncalibrated_probability_diagnostics",
            ],
            "row_reports": row_reports,
            "animal_macro_metrics": _animal_macro(row_reports),
            "mode_stratified_metrics": _mode_stratified(row_reports),
            "raw_calibration_diagnostics": _calibration_diagnostics(row_reports),
            "learned_dependencies": {
                "prior_model_weights": [],
                "prior_features": [],
                "prior_pseudolabels": [],
            },
        }
        report = {**payload, "receipt_sha256": _sha256_json(payload)}
        report_path = staging / "development_evaluation_report.json"
        _atomic_json(report_path, report)
        bundle_payload = {
            "schema_version": DEVELOPMENT_EVALUATION_BUNDLE_V6_SCHEMA,
            "output_directory": str(output_root),
            "report_relative_path": report_path.relative_to(staging).as_posix(),
            "report_file_sha256": _file_sha256(report_path),
            "report_receipt_sha256": report["receipt_sha256"],
            "cache_manifest_receipt_sha256": expected_cache_manifest_receipt_sha256,
            "run_manifest_receipt_sha256": expected_run_manifest_receipt_sha256,
            "source_sha256": source,
        }
        bundle = {**bundle_payload, "receipt_sha256": _sha256_json(bundle_payload)}
        _atomic_json(staging / "bundle_receipt.json", bundle)
        os.replace(staging, output_root)
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    verify_arbitrary_plane_development_evaluation_v6(
        output_root,
        expected_bundle_receipt_sha256=bundle["receipt_sha256"],
        expected_cache_manifest_receipt_sha256=expected_cache_manifest_receipt_sha256,
        expected_run_manifest_receipt_sha256=expected_run_manifest_receipt_sha256,
        expected_inference_source_sha256=expected_inference_source_sha256,
        device=device,
    )
    return bundle


def verify_arbitrary_plane_development_evaluation_v6(
    output_directory,
    *,
    expected_bundle_receipt_sha256: str,
    expected_cache_manifest_receipt_sha256: str,
    expected_run_manifest_receipt_sha256: str,
    expected_inference_source_sha256: Mapping[str, str],
    device: str | torch.device | None = None,
) -> bool:
    """Independently replay all metrics from frozen rows and raw predictions."""
    if any(
        not _valid_sha256(value)
        for value in (
            expected_bundle_receipt_sha256,
            expected_cache_manifest_receipt_sha256,
            expected_run_manifest_receipt_sha256,
        )
    ):
        raise ValueError("trusted bundle, cache, and run SHA-256 receipts are required")
    output_root = _i_path(output_directory, must_exist=True)
    source = _source_receipts()
    bundle = json.loads((output_root / "bundle_receipt.json").read_text("ascii"))
    bundle_payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    if (
        bundle.get("schema_version") != DEVELOPMENT_EVALUATION_BUNDLE_V6_SCHEMA
        or bundle.get("receipt_sha256") != expected_bundle_receipt_sha256
        or bundle.get("receipt_sha256") != _sha256_json(bundle_payload)
        or bundle.get("output_directory") != str(output_root)
        or bundle.get("cache_manifest_receipt_sha256") != expected_cache_manifest_receipt_sha256
        or bundle.get("run_manifest_receipt_sha256") != expected_run_manifest_receipt_sha256
        or bundle.get("source_sha256") != source
    ):
        raise ValueError("v6 development-evaluation bundle failed authentication")
    report_path = (output_root / bundle["report_relative_path"]).resolve(strict=True)
    if output_root not in report_path.parents or _file_sha256(report_path) != bundle["report_file_sha256"]:
        raise ValueError("v6 development-evaluation report file differs")
    report = json.loads(report_path.read_text("ascii"))
    report_payload = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if (
        report.get("schema_version") != DEVELOPMENT_EVALUATION_V6_SCHEMA
        or report.get("receipt_sha256") != _sha256_json(report_payload)
        or report.get("receipt_sha256") != bundle["report_receipt_sha256"]
        or report.get("source_sha256") != source
        or report.get("trusted_inference_source_sha256") != dict(expected_inference_source_sha256)
        or report.get("configuration_receipt_sha256") != _sha256_json(report.get("configuration", {}))
        or report.get("data_role") != DATA_ROLE
        or report.get("probability_status") != PROBABILITY_STATUS
        or report.get("probabilities_calibrated") is not False
        or report.get("calibration_fitted") is not False
        or report.get("release_qualifying") is not False
        or report.get("public_benchmark_accessed") is not False
        or report.get("external_validation_accessed") is not False
        or report.get("final_test_accessed") is not False
        or report.get("automatic_segmentation_used") is not False
        or report.get("learned_dependencies")
        != {"prior_model_weights": [], "prior_features": [], "prior_pseudolabels": []}
    ):
        raise ValueError("v6 development-evaluation report failed authentication")
    cache = report["cache_binding"]
    if cache["manifest_receipt_sha256"] != expected_cache_manifest_receipt_sha256:
        raise ValueError("v6 development cache differs from the trusted receipt")
    manifest = _development_manifest(cache["directory"], expected_cache_manifest_receipt_sha256)
    loaded = inference_v6.load_arbitrary_plane_inference_v6(
        report["run_binding"]["run_directory"],
        expected_run_manifest_receipt_sha256=expected_run_manifest_receipt_sha256,
        expected_inference_source_sha256=expected_inference_source_sha256,
        device=device,
    )
    try:
        if (
            report["run_binding"] != _run_binding(loaded)
            or cache["complete_row_selection_receipt_sha256"] != _selection_receipt(manifest)
            or cache["row_count"] != manifest["row_count"]
            or cache["generator_binding_receipt_sha256"] != manifest["generator_binding"]["receipt_sha256"]
            or cache["generation_lineage_sha256"] != manifest["generator_binding"]["generation_lineage_sha256"]
            or cache["freeze_audit"] != manifest["freeze_audit"]
        ):
            raise ValueError("v6 evaluation cache or run binding changed")
        catalogue = loaded["context"]["catalogue"]
        catalogue_context = _catalogue_metric_context(catalogue)
        atlas_shape = tuple(np.asarray(loaded["context"]["atlas_volume"]).shape[-3:])
        training_ids = _training_identity_sets(loaded["context"]["manifest"])
        development_ids = _identity_sets(manifest["rows"])
        overlap = {key: sorted(training_ids[key] & development_ids[key]) for key in FIVE_IDS}
        if any(overlap.values()) or report["identity_overlap_with_training"] != overlap:
            raise ValueError("v6 evaluation training/development identity separation failed")
        row_reports = report["row_reports"]
        if len(row_reports) != manifest["row_count"]:
            raise ValueError("v6 evaluation did not report every development row")
        recomputed = []
        expected_raw_paths = set()
        for cache_index, row_report in enumerate(row_reports):
            selection = finite_rows_v6.load_frozen_training_rows_v6(
                cache["directory"],
                [cache_index],
                expected_manifest_receipt_sha256=expected_cache_manifest_receipt_sha256,
            )
            row = selection["rows"][0]
            record = manifest["rows"][cache_index]
            if (
                row_report["evaluation_order"] != cache_index
                or row_report["cache_row_index"] != cache_index
                or row_report["training_row_id"] != row["training_row_id"]
                or row_report["training_row_receipt_sha256"] != row["receipt_sha256"]
                or row_report["synthetic_realization_id"] != row["synthetic_realization_id"]
                or any(row_report[key] != row["lineage"][key] for key in FIVE_IDS)
                or row_report["split"] != "development"
                or row_report["selected_mode"] != row["selected_mode"]
                or row_report["finite_psf_sha256"] != row["finite_psf_contract"]["finite_psf_sha256"]
                or record["training_row_receipt_sha256"] != row["receipt_sha256"]
            ):
                raise ValueError("v6 row report differs from its authenticated cache row")
            raw_record = row_report["raw_prediction"]
            raw_path = (output_root / raw_record["relative_path"]).resolve(strict=True)
            if output_root not in raw_path.parents:
                raise ValueError("raw prediction escaped its evaluation output")
            expected_raw_paths.add(raw_path)
            if _file_sha256(raw_path) != raw_record["file_sha256"] or raw_path.stat().st_size != raw_record["byte_count"]:
                raise ValueError("raw prediction file receipt differs")
            artifact = torch.load(raw_path, map_location="cpu", weights_only=True)
            result = artifact.pop("raw_inference_result")
            artifact_receipt = artifact.pop("artifact_receipt_sha256")
            if (
                artifact.get("schema_version") != RAW_PREDICTION_V6_SCHEMA
                or artifact_receipt != _sha256_json(artifact)
                or artifact_receipt != raw_record["artifact_receipt_sha256"]
                or artifact.get("cache_row_index") != cache_index
                or artifact.get("training_row_id") != row["training_row_id"]
                or artifact.get("training_row_receipt_sha256") != row["receipt_sha256"]
                or artifact.get("identifiers") != {key: row["lineage"][key] for key in FIVE_IDS}
                or artifact.get("selected_mode") != row["selected_mode"]
                or artifact.get("input_mode") != row_report["input_mode"]
                or artifact.get("run_binding") != result.get("run_binding")
                or result.get("run_binding") != _inference_result_run_binding(loaded)
                or artifact.get("raw_prediction_receipt_sha256") != _raw_prediction_receipt(result)
                or artifact.get("raw_prediction_receipt_sha256") != raw_record["raw_prediction_receipt_sha256"]
                or result.get("input_receipt") != _expected_input_receipt(row, catalogue)
                or result.get("trusted_inference_source_sha256") != dict(expected_inference_source_sha256)
                or result.get("probabilities_calibrated") is not False
                or result.get("evaluation_stage") != report["configuration"]["evaluation_stage"]
                or result.get("probability_status") != "raw_uncalibrated"
            ):
                raise ValueError("raw v6 prediction artifact failed independent authentication")
            metrics = _row_metrics(result, row, catalogue, catalogue_context, atlas_shape, report["configuration"])
            disposition = {
                "included_in_all_row_metrics": True,
                "included_in_animal_macro_where_defined": True,
                "failed": metrics["overall_failure"],
                "abstained": metrics["operational_abstention"],
                "no_silent_drop": True,
            }
            if row_report["metrics"] != _plain(metrics) or row_report["disposition"] != disposition:
                raise ValueError("v6 row metrics do not replay from raw prediction and truth")
            recomputed.append(row_report)
        actual_raw_paths = {path.resolve() for path in (output_root / "raw_predictions").glob("*.pt")}
        if actual_raw_paths != expected_raw_paths:
            raise ValueError("v6 raw prediction inventory differs from all-row accounting")
        expected_identities = {key: [row[key] for row in recomputed] for key in FIVE_IDS}
        failed = [row["training_row_id"] for row in recomputed if row["disposition"]["failed"]]
        abstained = [row["training_row_id"] for row in recomputed if row["disposition"]["abstained"]]
        accounting = {
            "expected_row_count": manifest["row_count"],
            "reported_row_count": len(recomputed),
            "no_rows_dropped": len(recomputed) == manifest["row_count"],
            "failed_row_count": len(failed),
            "failed_training_row_ids": failed,
            "abstained_row_count": len(abstained),
            "abstained_training_row_ids": abstained,
        }
        if (
            report["identities"] != expected_identities
            or report["row_accounting"] != accounting
            or report["animal_macro_metrics"] != _plain(_animal_macro(recomputed))
            or report["mode_stratified_metrics"] != _plain(_mode_stratified(recomputed))
            or report["raw_calibration_diagnostics"] != _plain(_calibration_diagnostics(recomputed))
            or _source_receipts() != source
        ):
            raise ValueError("v6 evaluation aggregate results failed independent replay")
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return True


__all__ = [
    "DATA_ROLE",
    "DEVELOPMENT_EVALUATION_BUNDLE_V6_SCHEMA",
    "DEVELOPMENT_EVALUATION_V6_SCHEMA",
    "PROBABILITY_STATUS",
    "RAW_PREDICTION_V6_SCHEMA",
    "run_arbitrary_plane_development_evaluation_v6",
    "verify_arbitrary_plane_development_evaluation_v6",
]
