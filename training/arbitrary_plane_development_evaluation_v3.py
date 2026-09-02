"""Provenance-bound internal development evaluation for the v3 joint model."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import training.arbitrary_plane_batch_v3 as batch_v3
import training.arbitrary_plane_deformation_primitives as deformation_v3
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_joint_loss as joint_loss_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import (
    normalized_raster_to_ccf,
    physical_um_to_allen_index_points,
)


DEVELOPMENT_EVALUATION_V3_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-development-evaluation/v3"
)
DEVELOPMENT_EVALUATION_BUNDLE_V3_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-development-evaluation-bundle/v3"
)
DEVELOPMENT_EVALUATION_ROLE = "internal-development-only"
RAW_UNCERTAINTY_SCOPE = (
    "uncalibrated raw model scores/covariances; diagnostic only; no coverage claim"
)
DEFAULT_OMITTED_MASS_FAILURE_THRESHOLD = 0.35
DEFAULT_MINIMUM_JACOBIAN = 0.05
DEFAULT_MAXIMUM_CYCLE_ERROR_PX = 1.0
_MODE_LABELS = {
    "smart-brush-absent": "raw acquired background",
    "smart-brush-accurate": "exact-black smart brush",
    "smart-brush-imperfect": "imperfect smart brush",
}
_SOURCE_FILES = (
    Path(__file__),
    Path(batch_v3.__file__),
    Path(deformation_v3.__file__),
    Path(inference_v3.__file__),
    Path(joint_loss_v3.__file__),
    Path(row_cache_v3.__file__),
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("development-evaluation receipts require finite values")
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _hash_json(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_receipts():
    root = Path(__file__).parent.parent.resolve()
    return {
        path.resolve().relative_to(root).as_posix(): _file_sha256(path)
        for path in _SOURCE_FILES
    }


def _i_path(path, *, must_exist=False):
    target = Path(path).resolve(strict=must_exist)
    if target.drive.upper() != "I:":
        raise ValueError("development evaluation artifacts must stay on the I: drive")
    return target


def _atomic_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("development-evaluation temporary path already exists")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, target)


def _audit_unselected_row_files(cache_root, manifest, selected_indices):
    selected = set(selected_indices)
    for record in manifest["rows"]:
        if record["row_index"] in selected:
            continue
        metadata_path = (cache_root / record["metadata_relative_path"]).resolve()
        arrays_path = (cache_root / record["arrays_relative_path"]).resolve()
        if (
            cache_root not in metadata_path.parents
            or cache_root not in arrays_path.parents
            or _file_sha256(metadata_path) != record["metadata_file_sha256"]
            or _file_sha256(arrays_path) != record["arrays_file_sha256"]
        ):
            raise ValueError("cached training-row file hash differs")


def _metric_summary(values):
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"eligible_count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "eligible_count": len(finite),
        "mean": float(np.mean(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _nested(record, path):
    value = record
    for name in path.split("."):
        value = value.get(name) if isinstance(value, dict) else None
    if isinstance(value, bool):
        return float(value)
    return value if isinstance(value, (int, float)) else None


def _animal_macro(row_reports):
    metric_paths = (
        "pose.physical_finite_frame_landmark_mean_um",
        "pose.plane_normal_projective_error_deg",
        "pose.absolute_signed_plane_offset_error_um",
        "pose.absolute_center_ap_error_um",
        "retrieval.topk_recall",
        "retrieval.raw_exact_tail_probability",
        "retrieval.failure_flag",
        "deformation.svf_mean_endpoint_error_px",
        "deformation.map_mean_endpoint_error_px",
        "deformation.deformation_failure",
        "deformation.topology_failure",
        "deformation.cycle_failure",
        "regional_overlap.mean_dice_foreground",
        "uncertainty.raw_normalized_retrieval_entropy",
        "overall_failure",
        "operational_abstention",
    )
    animals = {}
    for report in row_reports:
        animals.setdefault(report["animal_id"], []).append(report)
    per_animal = {}
    for animal_id, records in animals.items():
        per_animal[animal_id] = {
            "row_count": len(records),
            "metric_means": {
                path: _metric_summary([_nested(item["metrics"], path) for item in records])[
                    "mean"
                ]
                for path in metric_paths
            },
        }
    return {
        "statistical_unit": "animal",
        "animal_count": len(per_animal),
        "per_animal": per_animal,
        "macro_across_animals": {
            path: _metric_summary(
                [entry["metric_means"][path] for entry in per_animal.values()]
            )
            for path in metric_paths
        },
    }


def _mode_stratified(row_reports):
    result = {}
    for mode, label in _MODE_LABELS.items():
        rows = [item for item in row_reports if item["selected_mode"] == mode]
        failed = [item["training_row_id"] for item in rows if item["disposition"]["failed"]]
        abstained = [
            item["training_row_id"] for item in rows if item["disposition"]["abstained"]
        ]
        result[mode] = {
            "input_condition": label,
            "row_count": len(rows),
            "animal_count": len({item["animal_id"] for item in rows}),
            "failed_row_count": len(failed),
            "failed_row_fraction": None if not rows else len(failed) / len(rows),
            "failed_training_row_ids": failed,
            "abstained_row_count": len(abstained),
            "abstained_row_fraction": None if not rows else len(abstained) / len(rows),
            "abstained_training_row_ids": abstained,
            "animal_macro_metrics": _animal_macro(rows),
        }
    return result


def _pose_metrics(predicted_state, truth_state, support_origin):
    predicted = torch.as_tensor(predicted_state, dtype=torch.float64).reshape(1, 12)
    truth = torch.as_tensor(truth_state, dtype=torch.float64).reshape(1, 12)
    predicted_landmarks = joint_loss_v3.physical_frame_landmarks(predicted)[0]
    truth_landmarks = joint_loss_v3.physical_frame_landmarks(truth)[0]
    landmark_error = torch.linalg.vector_norm(
        predicted_landmarks - truth_landmarks, dim=-1
    )
    predicted_center, predicted_frame, _ = full_frame_state_to_components(predicted)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    predicted_normal = predicted_frame[0, :, 2]
    truth_normal = truth_frame[0, :, 2]
    dot = torch.dot(predicted_normal, truth_normal).clamp(-1.0, 1.0)
    sign = -1.0 if float(dot) < 0.0 else 1.0
    projective_angle = torch.acos(dot.abs())
    origin = torch.as_tensor(support_origin, dtype=torch.float64)
    predicted_offset = torch.dot(predicted_center[0] - origin, predicted_normal)
    aligned_truth_offset = sign * torch.dot(truth_center[0] - origin, truth_normal)
    signed_offset_error = predicted_offset - aligned_truth_offset
    center_ap_error = predicted_center[0, 0] - truth_center[0, 0]
    relative_rotation = predicted_frame[0].transpose(0, 1) @ truth_frame[0]
    frame_angle = torch.acos(
        ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    )
    return {
        "primary_internal_development_metric": "physical_finite_frame_landmark_mean_um",
        "physical_finite_frame_landmark_error_um": landmark_error.tolist(),
        "physical_finite_frame_landmark_mean_um": float(landmark_error.mean()),
        "physical_finite_frame_landmark_rms_um": float(
            landmark_error.square().mean().sqrt()
        ),
        "physical_finite_frame_landmark_max_um": float(landmark_error.max()),
        "plane_normal_projective_error_rad": float(projective_angle),
        "plane_normal_projective_error_deg": float(torch.rad2deg(projective_angle)),
        "signed_plane_offset_error_um": float(signed_offset_error),
        "absolute_signed_plane_offset_error_um": abs(float(signed_offset_error)),
        "signed_center_ap_error_um": float(center_ap_error),
        "absolute_center_ap_error_um": abs(float(center_ap_error)),
        "finite_frame_rotation_error_rad": float(frame_angle),
        "finite_frame_rotation_error_deg": float(torch.rad2deg(frame_angle)),
    }


def _field_error(prediction, truth, valid):
    error = torch.linalg.vector_norm(prediction - truth, dim=0)[valid]
    if error.numel() == 0:
        return {"valid_pixel_count": 0, "mean": None, "rms": None, "maximum": None}
    return {
        "valid_pixel_count": int(error.numel()),
        "mean": float(error.mean()),
        "rms": float(error.square().mean().sqrt()),
        "maximum": float(error.max()),
    }


def _deformation_metrics(raw_joint, mode_index, row, minimum_jacobian, maximum_cycle_error):
    if mode_index < 0:
        return {
            "map_component_has_refined_deformation": False,
            "valid_pixel_count": 0,
            "svf_mean_endpoint_error_px": None,
            "svf_rms_endpoint_error_px": None,
            "svf_max_endpoint_error_px": None,
            "map_mean_endpoint_error_px": None,
            "map_rms_endpoint_error_px": None,
            "map_max_endpoint_error_px": None,
            "minimum_forward_jacobian": None,
            "nonpositive_jacobian_fraction": None,
            "below_minimum_jacobian_fraction": None,
            "forward_cycle_rms_error_px": None,
            "forward_cycle_max_error_px": None,
            "inverse_cycle_rms_error_px": None,
            "inverse_cycle_max_error_px": None,
            "cycle_invalid_fraction": None,
            "topology_failure": True,
            "cycle_failure": True,
            "deformation_failure": True,
            "failure_reason": "MAP component belongs to the exact unrefined retrieval tail",
        }
    arrays = row["arrays"]
    valid = torch.from_numpy(
        np.asarray(arrays["truth_section_deformation_valid_mask"], dtype=bool)
        & np.asarray(arrays["target_valid_correspondence_mask"], dtype=bool)
        & ~np.asarray(arrays["target_correspondence_abstention_mask"], dtype=bool)
    )
    truth_velocity = torch.from_numpy(
        np.asarray(
            arrays["truth_section_pullback_stationary_velocity_yx_px_float64"]
        )
    ).permute(2, 0, 1).to(torch.float64)
    truth_map = torch.from_numpy(
        np.asarray(arrays["truth_section_pullback_map_yx_px_float64"])
    ).permute(2, 0, 1).to(torch.float64)
    predicted_velocity = raw_joint["final_stationary_velocity_yx_px"][
        0, mode_index
    ].to(torch.float64)
    predicted_map = raw_joint["final_pullback_map_yx_px"][0, mode_index].to(
        torch.float64
    )
    svf = _field_error(predicted_velocity, truth_velocity, valid)
    pullback = _field_error(predicted_map, truth_map, valid)
    jacobian = raw_joint["final_forward_jacobian_determinant"][
        0, mode_index, 0
    ].to(torch.float64)[valid]
    forward_cycle_valid = raw_joint["final_forward_then_inverse_valid_mask"][
        0, mode_index, 0
    ].bool() & valid
    inverse_cycle_valid = raw_joint["final_inverse_then_forward_valid_mask"][
        0, mode_index, 0
    ].bool() & valid
    forward_cycle = torch.linalg.vector_norm(
        raw_joint["final_forward_then_inverse_error_yx"][0, mode_index].to(
            torch.float64
        ),
        dim=0,
    )[forward_cycle_valid]
    inverse_cycle = torch.linalg.vector_norm(
        raw_joint["final_inverse_then_forward_error_yx"][0, mode_index].to(
            torch.float64
        ),
        dim=0,
    )[inverse_cycle_valid]
    topology_failure = bool(
        jacobian.numel() == 0 or (jacobian < float(minimum_jacobian)).any()
    )
    cycle_failure = bool(
        forward_cycle.numel() == 0
        or inverse_cycle.numel() == 0
        or (forward_cycle > float(maximum_cycle_error)).any()
        or (inverse_cycle > float(maximum_cycle_error)).any()
    )
    valid_count = int(valid.sum())
    cycle_valid_both = forward_cycle_valid & inverse_cycle_valid
    return {
        "map_component_has_refined_deformation": True,
        "selected_topk_index": int(mode_index),
        "valid_pixel_count": valid_count,
        "svf_mean_endpoint_error_px": svf["mean"],
        "svf_rms_endpoint_error_px": svf["rms"],
        "svf_max_endpoint_error_px": svf["maximum"],
        "map_mean_endpoint_error_px": pullback["mean"],
        "map_rms_endpoint_error_px": pullback["rms"],
        "map_max_endpoint_error_px": pullback["maximum"],
        "minimum_forward_jacobian": None if jacobian.numel() == 0 else float(jacobian.min()),
        "nonpositive_jacobian_fraction": None
        if jacobian.numel() == 0
        else float((jacobian <= 0.0).to(torch.float64).mean()),
        "minimum_jacobian_threshold": float(minimum_jacobian),
        "below_minimum_jacobian_fraction": None
        if jacobian.numel() == 0
        else float((jacobian < float(minimum_jacobian)).to(torch.float64).mean()),
        "forward_cycle_rms_error_px": None
        if forward_cycle.numel() == 0
        else float(forward_cycle.square().mean().sqrt()),
        "forward_cycle_max_error_px": None
        if forward_cycle.numel() == 0
        else float(forward_cycle.max()),
        "inverse_cycle_rms_error_px": None
        if inverse_cycle.numel() == 0
        else float(inverse_cycle.square().mean().sqrt()),
        "inverse_cycle_max_error_px": None
        if inverse_cycle.numel() == 0
        else float(inverse_cycle.max()),
        "maximum_cycle_error_threshold_px": float(maximum_cycle_error),
        "cycle_invalid_fraction": None
        if valid_count == 0
        else float(1.0 - cycle_valid_both.sum() / valid_count),
        "topology_failure": topology_failure,
        "cycle_failure": cycle_failure,
        "deformation_failure": bool(topology_failure or cycle_failure),
        "failure_reason": None
        if not (topology_failure or cycle_failure)
        else "Jacobian or inverse-consistency threshold failed",
    }


def _render_annotation(annotation, state, pullback, shape_h_w, origin, spacing):
    labels = torch.as_tensor(annotation)
    center, frame, basis = full_frame_state_to_components(
        torch.as_tensor(state, dtype=torch.float64).reshape(1, 12)
    )
    height, width = shape_h_w
    s = torch.arange(width, dtype=torch.float64) / width
    t = torch.arange(height, dtype=torch.float64) / height
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    points_um = normalized_raster_to_ccf(
        center[:, None, None],
        frame[:, None, None],
        basis[:, None, None],
        torch.stack((ss, tt), dim=-1)[None],
    )
    points = physical_um_to_allen_index_points(points_um, origin, spacing)
    depth, native_height, native_width = labels.shape
    grid = torch.stack(
        (
            points[..., 2] / (native_width - 1) * 2.0 - 1.0,
            points[..., 1] / (native_height - 1) * 2.0 - 1.0,
            points[..., 0] / (depth - 1) * 2.0 - 1.0,
        ),
        dim=-1,
    )[:, None].to(torch.float32)
    canonical = F.grid_sample(
        labels.to(torch.float32)[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, 0]
    return deformation_v3.warp_tensor_with_map_yx(
        canonical,
        torch.as_tensor(pullback, dtype=torch.float32).reshape(1, 2, height, width),
        mode="nearest",
    )[0, 0].round().to(torch.int64)


def _regional_overlap(row, annotation, state, pullback, origin, spacing):
    if annotation is None:
        return {
            "available": False,
            "method": None,
            "mean_dice_foreground": None,
            "micro_dice_foreground": None,
            "per_region_dice": {},
        }
    if pullback is None:
        return {
            "available": False,
            "method": None,
            "mean_dice_foreground": None,
            "micro_dice_foreground": None,
            "per_region_dice": {},
            "reason": "MAP component has no refined deformation",
        }
    truth = torch.from_numpy(
        np.asarray(row["arrays"]["source_label_ground_truth_canvas_int64"])
    ).to(torch.int64)
    mask = torch.from_numpy(
        np.asarray(row["arrays"]["source_tissue_ground_truth_mask"], dtype=bool)
    )
    predicted = _render_annotation(
        annotation, state, pullback, tuple(truth.shape), origin, spacing
    )
    labels = sorted(
        int(value)
        for value in torch.unique(torch.cat((truth[mask], predicted[mask])))
        if int(value) != 0
    )
    dice = {}
    intersections = denominator = 0
    for label in labels:
        truth_region = (truth == label) & mask
        predicted_region = (predicted == label) & mask
        intersection = int((truth_region & predicted_region).sum())
        region_denominator = int(truth_region.sum() + predicted_region.sum())
        dice[str(label)] = (
            None if region_denominator == 0 else 2.0 * intersection / region_denominator
        )
        intersections += intersection
        denominator += region_denominator
    eligible = [value for value in dice.values() if value is not None]
    return {
        "available": True,
        "method": "single-plane nearest annotation at MAP finite frame, then MAP pullback; foreground tissue only",
        "mean_dice_foreground": None if not eligible else float(np.mean(eligible)),
        "micro_dice_foreground": None
        if denominator == 0
        else 2.0 * intersections / denominator,
        "per_region_dice": dice,
    }


def _uncertainty_diagnostics(posterior, mode_index):
    probability = torch.as_tensor(
        posterior["raw_complete_retrieval_probability"][0], dtype=torch.float64
    )
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum()
    covariance = torch.as_tensor(
        posterior["refined_mode_raw_plane_tangent_covariance"][0],
        dtype=torch.float64,
    )
    selected_covariance = covariance[mode_index] if mode_index >= 0 else None
    return {
        "scope": RAW_UNCERTAINTY_SCOPE,
        "calibration_fitted_by_evaluator": False,
        "coverage_claimed": False,
        "coverage_metrics": None,
        "raw_retrieval_entropy_nats": float(entropy),
        "raw_normalized_retrieval_entropy": float(
            entropy / math.log(max(int(probability.numel()), 2))
        ),
        "raw_max_complete_catalogue_probability": float(probability.max()),
        "raw_map_mode_plane_tangent_covariance_trace": None
        if selected_covariance is None
        else float(torch.trace(selected_covariance)),
        "raw_map_mode_plane_tangent_covariance_max_eigenvalue": None
        if selected_covariance is None
        else float(torch.linalg.eigvalsh(selected_covariance).max()),
    }


def _row_metrics(result, raw_joint, row, catalogue, truth_state, support_origin, annotation, origin, spacing, config):
    pose = raw_joint["pose"]
    if pose.get("catalogue_complete") is not True or bool(
        torch.as_tensor(pose["retrieval_teacher_forced_mask"]).any()
    ):
        raise ValueError("development evaluation requires honest complete-catalogue inference")
    topk_ids = torch.as_tensor(pose["retrieval_topk_cell_id"])[0].to(torch.long)
    truth_cell_id = int(batch_v3.nearest_catalogue_cell_v3(truth_state, catalogue)[0])
    matches = topk_ids == truth_cell_id
    rank = int(matches.to(torch.int64).argmax()) + 1 if bool(matches.any()) else None
    posterior = result["probabilistic_output"]
    predicted_state = posterior["point_estimate"]["map_component_mean_state"][0]
    mode_index = int(posterior["point_estimate"]["map_component_topk_index"][0])
    raw_tail = float(posterior["raw_exact_omitted_probability"][0])
    deformation = _deformation_metrics(
        raw_joint,
        mode_index,
        row,
        config["minimum_jacobian"],
        config["maximum_cycle_error_px"],
    )
    pullback = (
        None
        if mode_index < 0
        else raw_joint["final_pullback_map_yx_px"][0, mode_index]
    )
    regional = _regional_overlap(
        row, annotation, predicted_state, pullback, origin, spacing
    )
    return {
        "pose": _pose_metrics(predicted_state, truth_state, support_origin),
        "retrieval": {
            "catalogue_complete": True,
            "teacher_forcing_used": False,
            "full_catalogue_cell_count": int(catalogue["counts"]["cell_count"]),
            "truth_catalogue_cell_id": truth_cell_id,
            "retrieved_topk_cell_ids": topk_ids.tolist(),
            "topk_recall": bool(matches.any()),
            "truth_rank_within_topk": rank,
            "raw_exact_tail_probability": raw_tail,
            "failure_omitted_mass_threshold": config[
                "omitted_mass_failure_threshold"
            ],
            "tail_mass_failure": raw_tail
            > config["omitted_mass_failure_threshold"],
            "map_component_unrefined_tail": mode_index < 0,
            "failure_flag": bool(
                raw_tail > config["omitted_mass_failure_threshold"]
                or mode_index < 0
            ),
        },
        "deformation": deformation,
        "regional_overlap": regional,
        "uncertainty": _uncertainty_diagnostics(posterior, mode_index),
        "overall_failure": bool(
            raw_tail > config["omitted_mass_failure_threshold"]
            or mode_index < 0
            or deformation["deformation_failure"]
        ),
    }


def run_arbitrary_plane_development_evaluation_v3(
    cache_directory,
    checkpoint_path,
    catalogue,
    atlas_volume_c_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    output_directory,
    *,
    development_evaluation_animal_ids,
    row_indices=None,
    annotation_volume_ap_dv_ml=None,
    top_k=4,
    refinement_steps=3,
    pose_only_steps=2,
    retrieval_shape_h_w=(48, 64),
    catalogue_chunk_size=128,
    gauss_hermite_order=5,
    evaluation_seed=0,
    minimum_jacobian=DEFAULT_MINIMUM_JACOBIAN,
    maximum_cycle_error_px=DEFAULT_MAXIMUM_CYCLE_ERROR_PX,
    catalogue_feature_cache_path=None,
    device="cpu",
):
    """Run a bounded internal development evaluation without calibration fitting."""
    cache_root = _i_path(cache_directory, must_exist=True)
    checkpoint = _i_path(checkpoint_path, must_exist=True)
    output_root = _i_path(output_directory)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("development-evaluation output directory must be empty")
    manifest = row_cache_v3.load_training_row_cache_manifest_v3(cache_root)
    if manifest["status"] != row_cache_v3.FROZEN_CACHE_STATUS:
        raise ValueError("development evaluation requires a frozen row cache")
    selected_indices = (
        list(range(manifest["row_count"]))
        if row_indices is None
        else [int(index) for index in row_indices]
    )
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or any(index < 0 or index >= manifest["row_count"] for index in selected_indices)
    ):
        raise ValueError("development evaluation requires unique nonempty row indices")
    selected_records = [manifest["rows"][index] for index in selected_indices]
    for record in selected_records:
        split = str(record["lineage"]["split"]).lower()
        if "development" not in split or any(
            token in split
            for token in ("benchmark", "final", "test", "external", "validation", "calibration")
        ):
            raise ValueError("only internal development rows may enter this evaluator")
    declared_animals = list(development_evaluation_animal_ids)
    if (
        not declared_animals
        or len(set(declared_animals)) != len(declared_animals)
        or any(not isinstance(value, str) or not value for value in declared_animals)
    ):
        raise ValueError("development-evaluation animal IDs must be unique nonempty strings")
    row_animals = {record["lineage"]["animal_id"] for record in selected_records}
    if row_animals != set(declared_animals):
        raise ValueError("declared development-evaluation animals differ from selected rows")
    _audit_unselected_row_files(cache_root, manifest, selected_indices)
    session = inference_v3.open_arbitrary_plane_inference_session_v3(
        checkpoint,
        atlas_volume_c_ap_dv_ml,
        catalogue,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        axial_offsets_um,
        axial_weights,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        catalogue_feature_cache_path=None
        if catalogue_feature_cache_path is None
        else _i_path(catalogue_feature_cache_path, must_exist=True),
        device=device,
    )
    training_animals = set(
        session["checkpoint_receipt"]["training_receipt"]["training_animal_ids"]
    )
    leakage = training_animals & row_animals
    if leakage:
        raise ValueError("development-evaluation animal leakage into training IDs")
    config = {
        "top_k": int(top_k),
        "refinement_steps": int(refinement_steps),
        "pose_only_steps": int(pose_only_steps),
        "retrieval_shape_h_w": [int(value) for value in retrieval_shape_h_w],
        "catalogue_chunk_size": int(catalogue_chunk_size),
        "gauss_hermite_order": int(gauss_hermite_order),
        "evaluation_seed": int(evaluation_seed),
        "evaluation_rng_use": "no stochastic evaluator operation; seed is provenance-only",
        "minimum_jacobian": float(minimum_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "omitted_mass_failure_threshold": DEFAULT_OMITTED_MASS_FAILURE_THRESHOLD,
        "inference_batch_size": 1,
        "raw_prediction_granularity": "one immutable file per row",
        "catalogue_scope": "complete catalogue posterior/inference scope",
        "catalogue_feature_cache": None
        if session["feature_cache"] is None
        else {
            "cache_id": session["feature_cache"]["cache_receipt"]["cache_id"],
            "path": session["feature_cache"]["cache_path"],
            "file_sha256": session["feature_cache"]["cache_file_sha256"],
            "cache_receipt": inference_v3._json(
                session["feature_cache"]["cache_receipt"]
            ),
        },
    }
    if (
        config["top_k"] < 1
        or config["top_k"] > int(catalogue["counts"]["cell_count"])
        or config["refinement_steps"] < 1
        or not 0 <= config["pose_only_steps"] <= config["refinement_steps"]
        or min(config["retrieval_shape_h_w"]) < 4
        or config["catalogue_chunk_size"] < 1
        or not 3 <= config["gauss_hermite_order"] <= 9
        or not math.isfinite(config["minimum_jacobian"])
        or config["minimum_jacobian"] <= 0.0
        or not math.isfinite(config["maximum_cycle_error_px"])
        or config["maximum_cycle_error_px"] <= 0.0
    ):
        raise ValueError("development-evaluation configuration is invalid")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw_predictions"
    raw_root.mkdir()
    atlas_shape = tuple(torch.as_tensor(atlas_volume_c_ap_dv_ml).shape[-3:])
    row_reports = []
    geometry_gauge_contract = manifest["generator_binding"][
        "geometry_gauge_contract"
    ]
    for order, cache_index in enumerate(selected_indices):
        record = manifest["rows"][cache_index]
        row = row_cache_v3._load_record(
            cache_root, record, geometry_gauge_contract
        )
        lineage = row["lineage"]
        input_b3hw = torch.from_numpy(
            np.asarray(row["arrays"]["model_input_channels_float32"])
        ).permute(2, 0, 1)[None]
        raw_path = raw_root / f"row_{order:06d}_{row['receipt_sha256'][:16]}.pt"
        result = inference_v3.run_arbitrary_plane_inference_session_v3(
            session,
            input_b3hw,
            animal_ids=[lineage["animal_id"]],
            specimen_ids=[lineage["specimen_id"]],
            experiment_ids=[lineage["experiment_id"]],
            synthetic_animal_ids=[lineage["synthetic_animal_id"]],
            section_ids=[lineage["section_id"]],
            synthetic_realization_ids=[row["synthetic_realization_id"]],
            top_k=config["top_k"],
            refinement_steps=config["refinement_steps"],
            pose_only_steps=config["pose_only_steps"],
            retrieval_shape_h_w=tuple(config["retrieval_shape_h_w"]),
            catalogue_chunk_size=config["catalogue_chunk_size"],
            gauss_hermite_order=config["gauss_hermite_order"],
            raw_prediction_output_path=raw_path,
            return_raw_prediction=True,
        )
        raw_joint = result.pop("raw_prediction")
        converted = batch_v3.training_row_to_tensors_v3(
            row,
            atlas_shape_ap_dv_ml=atlas_shape,
            origin_ap_dv_ml_um=origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um=voxel_size_ap_dv_ml_um,
            device="cpu",
        )
        metrics = _row_metrics(
            result,
            raw_joint,
            row,
            catalogue,
            converted["tensors"]["truth_state"],
            catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
            annotation_volume_ap_dv_ml,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            config,
        )
        failure_reasons = []
        if metrics["retrieval"]["tail_mass_failure"]:
            failure_reasons.append("raw exact omitted retrieval mass exceeded threshold")
        if metrics["retrieval"]["map_component_unrefined_tail"]:
            failure_reasons.append("MAP component belonged to the unrefined retrieval tail")
        if metrics["deformation"]["topology_failure"]:
            failure_reasons.append("deformation topology threshold failed")
        if metrics["deformation"]["cycle_failure"]:
            failure_reasons.append("deformation inverse-consistency threshold failed")
        abstained = bool(
            not metrics["deformation"]["map_component_has_refined_deformation"]
            or metrics["deformation"]["valid_pixel_count"] == 0
        )
        metrics["operational_abstention"] = abstained
        row_reports.append(
            {
                "evaluation_order": order,
                "cache_row_index": cache_index,
                "training_row_id": row["training_row_id"],
                "training_row_receipt_sha256": row["receipt_sha256"],
                "synthetic_realization_id": row["synthetic_realization_id"],
                "animal_id": lineage["animal_id"],
                "specimen_id": lineage["specimen_id"],
                "experiment_id": lineage["experiment_id"],
                "synthetic_animal_id": lineage["synthetic_animal_id"],
                "section_id": lineage["section_id"],
                "split": lineage["split"],
                "selected_mode": row["selected_mode"],
                "input_condition": _MODE_LABELS[row["selected_mode"]],
                "disposition": {
                    "included_in_row_metrics": True,
                    "included_in_animal_macro_denominators_where_metric_defined": True,
                    "failed": metrics["overall_failure"],
                    "abstained": abstained,
                    "failure_reasons": failure_reasons,
                    "no_silent_drop": True,
                },
                "cache_record_file_receipts": {
                    "metadata_file_sha256": record["metadata_file_sha256"],
                    "arrays_file_sha256": record["arrays_file_sha256"],
                },
                "raw_prediction": {
                    "relative_path": raw_path.relative_to(output_root).as_posix(),
                    "file_sha256": result["raw_prediction_file_sha256"],
                    "prediction_receipt": result["raw_prediction_receipt"],
                    "inference_receipt_sha256": result[
                        "inference_receipt_sha256"
                    ],
                    "input_receipt": result["input_receipt"],
                    "atlas_receipt": result["atlas_receipt"],
                    "configuration_receipt": result["configuration_receipt"],
                },
                "metrics": metrics,
            }
        )
    identities = {
        "training_row_ids": [item["training_row_id"] for item in row_reports],
        "development_evaluation_animal_ids": declared_animals,
        "specimen_ids": [item["specimen_id"] for item in row_reports],
        "experiment_ids": [item["experiment_id"] for item in row_reports],
        "synthetic_animal_ids": [item["synthetic_animal_id"] for item in row_reports],
        "section_ids": [item["section_id"] for item in row_reports],
        "synthetic_realization_ids": [
            item["synthetic_realization_id"] for item in row_reports
        ],
    }
    failed_rows = [
        item["training_row_id"] for item in row_reports if item["disposition"]["failed"]
    ]
    abstained_rows = [
        item["training_row_id"] for item in row_reports if item["disposition"]["abstained"]
    ]
    payload = {
        "schema_version": DEVELOPMENT_EVALUATION_V3_SCHEMA,
        "data_role": DEVELOPMENT_EVALUATION_ROLE,
        "scientific_scope": "internal synthetic development evaluation; not validation, qualification, or benchmarking",
        "public_benchmark_accessed": False,
        "final_test_accessed": False,
        "external_validation_accessed": False,
        "calibration_fitted": False,
        "uncertainty_scope": RAW_UNCERTAINTY_SCOPE,
        "source_sha256": _source_receipts(),
        "configuration": config,
        "configuration_receipt_sha256": _hash_json(config),
        "cache_binding": {
            "directory": str(cache_root),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "status": manifest["status"],
            "row_count": manifest["row_count"],
            "selected_row_indices": selected_indices,
            "generator_binding": manifest["generator_binding"],
            "generation_config": manifest["generation_config"],
            "seed_record": manifest["seed_record"],
        },
        "checkpoint_binding": {
            "path": str(checkpoint),
            "file_sha256": session["checkpoint_file_sha256"],
            "checkpoint_id": session["checkpoint_id"],
            "checkpoint_binding_id": session["checkpoint_binding_id"],
            "model_state_sha256": session["model_state_sha256"],
            "training_receipt": inference_v3._json(
                session["checkpoint_receipt"]["training_receipt"]
            ),
            "training_animal_ids": sorted(training_animals),
            "runtime_source_sha256": inference_v3._json(
                session["checkpoint_receipt"]["runtime_source_sha256"]
            ),
        },
        "catalogue_binding": {
            "catalogue_id": catalogue["catalogue_id"],
            "receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        },
        "atlas_binding": inference_v3._json(session["inference_contract"]),
        "identities": identities,
        "row_accounting": {
            "selected_row_count": len(selected_indices),
            "reported_row_count": len(row_reports),
            "no_rows_dropped": len(selected_indices) == len(row_reports),
            "failed_row_count": len(failed_rows),
            "failed_training_row_ids": failed_rows,
            "abstained_row_count": len(abstained_rows),
            "abstained_training_row_ids": abstained_rows,
        },
        "row_reports": row_reports,
        "animal_macro_metrics": _animal_macro(row_reports),
        "mode_stratified_metrics": _mode_stratified(row_reports),
        "learned_dependencies": {
            "prior_model_weights": [],
            "prior_features": [],
            "prior_pseudolabels": [],
        },
    }
    report = {**payload, "receipt_sha256": _hash_json(payload)}
    report_path = output_root / "development_evaluation_report.json"
    _atomic_json(report_path, report)
    bundle_payload = {
        "schema_version": DEVELOPMENT_EVALUATION_BUNDLE_V3_SCHEMA,
        "output_directory": str(output_root),
        "report_relative_path": report_path.relative_to(output_root).as_posix(),
        "report_file_sha256": _file_sha256(report_path),
        "report_receipt_sha256": report["receipt_sha256"],
        "source_sha256": _source_receipts(),
    }
    bundle = {**bundle_payload, "receipt_sha256": _hash_json(bundle_payload)}
    _atomic_json(output_root / "bundle_receipt.json", bundle)
    return bundle


def verify_arbitrary_plane_development_evaluation_v3(output_directory, *, catalogue=None):
    """Independently authenticate a frozen development report and every raw prediction."""
    output_root = _i_path(output_directory, must_exist=True)
    bundle = json.loads((output_root / "bundle_receipt.json").read_text("ascii"))
    bundle_payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    if (
        bundle.get("schema_version") != DEVELOPMENT_EVALUATION_BUNDLE_V3_SCHEMA
        or bundle.get("receipt_sha256") != _hash_json(bundle_payload)
        or bundle.get("output_directory") != str(output_root)
        or bundle.get("source_sha256") != _source_receipts()
    ):
        raise ValueError("development-evaluation bundle receipt failed authentication")
    report_path = (output_root / bundle["report_relative_path"]).resolve()
    if output_root not in report_path.parents or _file_sha256(report_path) != bundle["report_file_sha256"]:
        raise ValueError("development-evaluation report file hash differs")
    report = json.loads(report_path.read_text("ascii"))
    report_payload = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if (
        report.get("schema_version") != DEVELOPMENT_EVALUATION_V3_SCHEMA
        or report.get("receipt_sha256") != _hash_json(report_payload)
        or report.get("receipt_sha256") != bundle["report_receipt_sha256"]
        or report.get("source_sha256") != _source_receipts()
        or report.get("data_role") != DEVELOPMENT_EVALUATION_ROLE
        or report.get("public_benchmark_accessed") is not False
        or report.get("final_test_accessed") is not False
        or report.get("external_validation_accessed") is not False
        or report.get("calibration_fitted") is not False
        or report.get("uncertainty_scope") != RAW_UNCERTAINTY_SCOPE
        or report.get("configuration_receipt_sha256")
        != _hash_json(report.get("configuration", {}))
        or report.get("learned_dependencies")
        != {
            "prior_model_weights": [],
            "prior_features": [],
            "prior_pseudolabels": [],
        }
    ):
        raise ValueError("development-evaluation report failed authentication")
    checkpoint = report["checkpoint_binding"]
    checkpoint_path = _i_path(checkpoint["path"], must_exist=True)
    feature_cache = report["configuration"].get("catalogue_feature_cache")
    if catalogue is None:
        if _file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
            raise ValueError("development-evaluation checkpoint file hash differs")
        if feature_cache is not None and _file_sha256(
            _i_path(feature_cache["path"], must_exist=True)
        ) != feature_cache["file_sha256"]:
            raise ValueError("development-evaluation catalogue feature cache file hash differs")
    else:
        loaded = inference_v3.load_arbitrary_plane_inference_v3(
            checkpoint_path, catalogue, device="cpu"
        )
        if (
            loaded["checkpoint_file_sha256"] != checkpoint["file_sha256"]
            or loaded["checkpoint_id"] != checkpoint["checkpoint_id"]
            or loaded["checkpoint_binding_id"] != checkpoint["checkpoint_binding_id"]
            or loaded["model_state_sha256"] != checkpoint["model_state_sha256"]
        ):
            raise ValueError("development-evaluation checkpoint binding differs")
        if feature_cache is not None:
            loaded_cache = inference_v3._load_catalogue_feature_cache_file_v3(
                feature_cache["path"]
            )
            inference_v3._verify_catalogue_feature_cache_contents_v3(
                loaded_cache,
                loaded,
                catalogue,
                loaded["model"],
                verify_file_binding=False,
            )
            if (
                loaded_cache["cache_file_sha256"] != feature_cache["file_sha256"]
                or loaded_cache["cache_receipt"] != feature_cache["cache_receipt"]
            ):
                raise ValueError(
                    "development-evaluation catalogue feature cache receipt differs"
                )
    cache = report["cache_binding"]
    manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        cache["directory"], expected_receipt_sha256=cache["manifest_receipt_sha256"]
    )
    if manifest["status"] != row_cache_v3.FROZEN_CACHE_STATUS:
        raise ValueError("development-evaluation row cache is no longer frozen")
    selected_indices = cache["selected_row_indices"]
    row_reports = report["row_reports"]
    if len(selected_indices) != len(row_reports):
        raise ValueError("development-evaluation row count differs")
    failed_rows = [
        item["training_row_id"] for item in row_reports if item["disposition"]["failed"]
    ]
    abstained_rows = [
        item["training_row_id"] for item in row_reports if item["disposition"]["abstained"]
    ]
    if (
        report.get("row_accounting")
        != {
            "selected_row_count": len(selected_indices),
            "reported_row_count": len(row_reports),
            "no_rows_dropped": True,
            "failed_row_count": len(failed_rows),
            "failed_training_row_ids": failed_rows,
            "abstained_row_count": len(abstained_rows),
            "abstained_training_row_ids": abstained_rows,
        }
        or report.get("animal_macro_metrics") != _animal_macro(row_reports)
        or report.get("mode_stratified_metrics") != _mode_stratified(row_reports)
    ):
        raise ValueError("development-evaluation row accounting or strata differ")
    identities = report.get("identities", {})
    declared_animals = identities.get("development_evaluation_animal_ids", [])
    expected_identities = {
        "training_row_ids": [item["training_row_id"] for item in row_reports],
        "development_evaluation_animal_ids": declared_animals,
        "specimen_ids": [item["specimen_id"] for item in row_reports],
        "experiment_ids": [item["experiment_id"] for item in row_reports],
        "synthetic_animal_ids": [item["synthetic_animal_id"] for item in row_reports],
        "section_ids": [item["section_id"] for item in row_reports],
        "synthetic_realization_ids": [
            item["synthetic_realization_id"] for item in row_reports
        ],
    }
    if (
        identities != expected_identities
        or not isinstance(declared_animals, list)
        or len(declared_animals) != len(set(declared_animals))
        or set(declared_animals) != {item["animal_id"] for item in row_reports}
    ):
        raise ValueError("development-evaluation identity summary differs")
    if set(checkpoint["training_animal_ids"]) & set(
        declared_animals
    ):
        raise ValueError("development-evaluation animal leakage into training IDs")
    raw_paths = set()
    cache_root = _i_path(cache["directory"], must_exist=True)
    geometry_gauge_contract = manifest["generator_binding"][
        "geometry_gauge_contract"
    ]
    for cache_index, row_report in zip(selected_indices, row_reports, strict=True):
        if not isinstance(cache_index, int) or not 0 <= cache_index < manifest["row_count"]:
            raise ValueError("development-evaluation row index is invalid")
        row = row_cache_v3._load_record(
            cache_root, manifest["rows"][cache_index], geometry_gauge_contract
        )
        if (
            row["training_row_id"] != row_report["training_row_id"]
            or row["receipt_sha256"] != row_report["training_row_receipt_sha256"]
            or row["lineage"]["animal_id"] != row_report["animal_id"]
            or row["lineage"]["specimen_id"] != row_report["specimen_id"]
            or row["lineage"]["experiment_id"] != row_report["experiment_id"]
            or row["lineage"]["synthetic_animal_id"]
            != row_report["synthetic_animal_id"]
            or row["lineage"]["section_id"] != row_report["section_id"]
            or row["synthetic_realization_id"]
            != row_report["synthetic_realization_id"]
            or row["selected_mode"] != row_report["selected_mode"]
            or row_report["input_condition"]
            != _MODE_LABELS.get(row["selected_mode"])
            or row_report.get("disposition", {}).get("no_silent_drop") is not True
            or row_report.get("disposition", {}).get("included_in_row_metrics")
            is not True
        ):
            raise ValueError("development-evaluation row identity differs")
        raw_record = row_report["raw_prediction"]
        raw_path = (output_root / raw_record["relative_path"]).resolve()
        if output_root not in raw_path.parents or raw_path in raw_paths:
            raise ValueError("development-evaluation raw prediction path is invalid")
        raw_paths.add(raw_path)
        if _file_sha256(raw_path) != raw_record["file_sha256"]:
            raise ValueError("development-evaluation raw prediction file hash differs")
        artifact = torch.load(raw_path, map_location="cpu", weights_only=True)
        lineage = {
            "animal_id": row_report["animal_id"],
            "specimen_id": row_report["specimen_id"],
            "experiment_id": row_report["experiment_id"],
            "synthetic_animal_id": row_report["synthetic_animal_id"],
            "section_id": row_report["section_id"],
            "synthetic_realization_id": row_report["synthetic_realization_id"],
        }
        identifiers = {
            "animal_ids": [row_report["animal_id"]],
            "specimen_ids": [row_report["specimen_id"]],
            "experiment_ids": [row_report["experiment_id"]],
            "synthetic_animal_ids": [row_report["synthetic_animal_id"]],
            "section_ids": [row_report["section_id"]],
            "synthetic_realization_ids": [row_report["synthetic_realization_id"]],
            "lineage": [lineage],
        }
        if (
            artifact.get("schema_version") != "anatomy-tracker.raw-joint-prediction/v3"
            or artifact.get("checkpoint_id") != checkpoint["checkpoint_id"]
            or artifact.get("checkpoint_binding_id")
            != checkpoint["checkpoint_binding_id"]
            or artifact.get("catalogue_id")
            != report["catalogue_binding"]["catalogue_id"]
            or artifact.get("identifiers") != identifiers
            or artifact.get("input_receipt") != raw_record["input_receipt"]
            or artifact.get("input_receipt", {}).get("identifiers") != identifiers
            or artifact.get("configuration_receipt")
            != raw_record["configuration_receipt"]
            or artifact.get("raw_prediction", {}).get("lineage") != [lineage]
            or artifact.get("raw_prediction_receipt")
            != raw_record["prediction_receipt"]
            or inference_v3._prediction_receipt(artifact.get("raw_prediction"))
            != raw_record["prediction_receipt"]
        ):
            raise ValueError("development-evaluation raw prediction receipt differs")
        inference_receipt_payload = {
            "schema_version": inference_v3.INFERENCE_V3_SCHEMA,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_binding_id": checkpoint["checkpoint_binding_id"],
            "checkpoint_file_sha256": checkpoint["file_sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "catalogue_id": report["catalogue_binding"]["catalogue_id"],
            "animal_ids": [row_report["animal_id"]],
            "specimen_ids": [row_report["specimen_id"]],
            "experiment_ids": [row_report["experiment_id"]],
            "synthetic_animal_ids": [row_report["synthetic_animal_id"]],
            "section_ids": [row_report["section_id"]],
            "synthetic_realization_ids": [row_report["synthetic_realization_id"]],
            "lineage": [lineage],
            "input_receipt": raw_record["input_receipt"],
            "atlas_receipt": raw_record["atlas_receipt"],
            "configuration_receipt": raw_record["configuration_receipt"],
            "raw_prediction_receipt": raw_record["prediction_receipt"],
            "raw_prediction_path": str(raw_path),
            "raw_prediction_file_sha256": raw_record["file_sha256"],
        }
        if (
            raw_record["atlas_receipt"] != report["atlas_binding"]
            or inference_v3._sha(inference_receipt_payload)
            != raw_record["inference_receipt_sha256"]
        ):
            raise ValueError("development-evaluation inference receipt differs")
    if catalogue is not None:
        if report["catalogue_binding"] != {
            "catalogue_id": catalogue["catalogue_id"],
            "receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        }:
            raise ValueError("development-evaluation catalogue binding differs")
    return True
