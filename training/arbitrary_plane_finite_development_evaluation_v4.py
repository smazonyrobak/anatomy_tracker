"""Authenticated all-row internal development evaluation for finite-PSF v4."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_batch_v3 as batch_v3
import training.arbitrary_plane_development_evaluation_v3 as metrics_v3
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as row_cache_v4


FINITE_DEVELOPMENT_EVALUATION_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-development-evaluation/v4"
)
FINITE_DEVELOPMENT_EVALUATION_BUNDLE_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-development-evaluation-bundle/v4"
)
FINITE_DEVELOPMENT_EVALUATION_ROLE = "internal-development-only"
UNCALIBRATED_SCOPE = (
    "uncalibrated raw model scores/covariances; diagnostic only; no coverage claim"
)
_MODE_LABELS = metrics_v3._MODE_LABELS
_SOURCE_FILES = (
    Path(__file__),
    Path(batch_v3.__file__),
    Path(metrics_v3.__file__),
    Path(inference_v3.__file__),
    Path(psf_v4.__file__),
    Path(row_cache_v4.__file__),
)


def _plain(value):
    if isinstance(value, Mapping):
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
        raise ValueError("finite development receipts require finite values")
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value), allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("ascii")


def _sha(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
        raise ValueError("finite development artifacts must stay on I:")
    return target


def _atomic_json_new(path, value):
    target = _i_path(path)
    if os.path.lexists(target):
        raise FileExistsError("finite development JSON target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError("finite development temporary target already exists")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _metric_summary(values):
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"eligible_animal_count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "eligible_animal_count": len(finite),
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


_ANIMAL_METRICS = (
    "pose.physical_finite_frame_landmark_mean_um",
    "pose.plane_normal_projective_error_deg",
    "pose.absolute_signed_plane_offset_error_um",
    "pose.finite_frame_rotation_error_deg",
    "retrieval.topk_recall",
    "retrieval.failure_flag",
    "dense.mean_endpoint_error_px",
    "dense.deformation_failure",
    "regional_overlap.mean_dice_foreground",
    "uncertainty.raw_normalized_retrieval_entropy",
    "overall_failure",
    "operational_abstention",
)


def _animal_macro(row_reports):
    animals = {}
    for report in row_reports:
        animals.setdefault(report["animal_id"], []).append(report)
    per_animal = {}
    for animal_id, reports in sorted(animals.items()):
        per_animal[animal_id] = {
            "row_count": len(reports),
            "metric_means": {
                path: (
                    None
                    if not [
                        value for value in (_nested(item["metrics"], path) for item in reports)
                        if value is not None
                    ]
                    else float(np.mean([
                        value for value in (_nested(item["metrics"], path) for item in reports)
                        if value is not None
                    ]))
                )
                for path in _ANIMAL_METRICS
            },
        }
    return {
        "statistical_unit": "animal",
        "animal_count": len(per_animal),
        "per_animal": per_animal,
        "macro_across_animals": {
            path: _metric_summary([
                record["metric_means"][path] for record in per_animal.values()
            ])
            for path in _ANIMAL_METRICS
        },
        "confidence_interval_status": (
            "deferred until the predefined animal-level validation protocol; "
            "this is internal development only"
        ),
    }


def _mode_stratified(row_reports):
    result = {}
    for mode, label in _MODE_LABELS.items():
        rows = [item for item in row_reports if item["selected_mode"] == mode]
        result[mode] = {
            "input_condition": label,
            "row_count": len(rows),
            "animal_count": len({item["animal_id"] for item in rows}),
            "failed_row_count": sum(item["disposition"]["failed"] for item in rows),
            "abstained_row_count": sum(item["disposition"]["abstained"] for item in rows),
            "animal_macro_metrics": _animal_macro(rows),
        }
    return result


def _dense_summary(deformation):
    return {
        "supervision_family": "identifiable nonrigid pullback/SVF",
        "valid_pixel_count": deformation["valid_pixel_count"],
        "mean_endpoint_error_px": deformation["map_mean_endpoint_error_px"],
        "rms_endpoint_error_px": deformation["map_rms_endpoint_error_px"],
        "maximum_endpoint_error_px": deformation["map_max_endpoint_error_px"],
        "minimum_forward_jacobian": deformation["minimum_forward_jacobian"],
        "topology_failure": deformation["topology_failure"],
        "cycle_failure": deformation["cycle_failure"],
        "deformation_failure": deformation["deformation_failure"],
    }


def _experiment_scope(manifest):
    contract = manifest["finite_psf_run_contract"]
    mode = contract["render_mode"]
    count = int(contract["axial_sample_count"])
    if mode == "finite_boxcar" and count == psf_v4.PRODUCTION_AXIAL_SAMPLE_COUNT:
        return "finite-thickness-production-s9"
    if mode == "centre_plane_ablation" and count == 1:
        return "exact-zero-thickness-ablation-s1"
    raise ValueError("finite production and zero-thickness ablation scopes cannot be mixed or aliased")


def _row_schedule_binding(row, runtime_contract):
    contract = row["finite_psf_contract"]
    return {
        "source": "authenticated-per-row-finite_psf_contract",
        "render_mode": contract["render_mode"],
        "nominal_cut_thickness_um": contract["nominal_cut_thickness_um"],
        "axial_sample_count": contract["axial_sample_count"],
        "axial_offsets_um": contract["axial_offsets_um"],
        "axial_weights": contract["axial_weights"],
        "finite_psf_sha256": contract["finite_psf_sha256"],
        "finite_psf_capability_sha256": contract["finite_psf_capability_sha256"],
        "slab_observation_v4_receipt_sha256": contract[
            "slab_observation_v4_receipt_sha256"
        ],
        "runtime_inference_contract": _plain(runtime_contract),
    }


def run_arbitrary_plane_finite_development_evaluation_v4(
    cache_directory,
    checkpoint_path,
    catalogue,
    atlas_volume_c_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    output_directory,
    *,
    development_evaluation_animal_ids,
    annotation_volume_ap_dv_ml=None,
    top_k=4,
    refinement_steps=3,
    pose_only_steps=2,
    retrieval_shape_h_w=(48, 64),
    catalogue_chunk_size=128,
    gauss_hermite_order=5,
    evaluation_seed=0,
    minimum_jacobian=metrics_v3.DEFAULT_MINIMUM_JACOBIAN,
    maximum_cycle_error_px=metrics_v3.DEFAULT_MAXIMUM_CYCLE_ERROR_PX,
    device="cpu",
):
    """Evaluate every row in one frozen homogeneous finite v4 development cache."""
    cache_root = _i_path(cache_directory, must_exist=True)
    checkpoint = _i_path(checkpoint_path, must_exist=True)
    output_root = _i_path(output_directory)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("finite development output directory must be empty")
    manifest = row_cache_v4.load_training_row_cache_manifest_v4(cache_root)
    if manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS:
        raise ValueError("finite development evaluation requires a frozen v4 cache")
    experiment_scope = _experiment_scope(manifest)
    selected_indices = list(range(int(manifest["row_count"])))
    if not selected_indices:
        raise ValueError("finite development evaluation requires at least one row")
    for record in manifest["rows"]:
        split = str(record["lineage"]["split"]).lower()
        if "development" not in split or any(
            token in split for token in (
                "benchmark", "final", "test", "external", "validation", "calibration"
            )
        ):
            raise ValueError("only untouched internal-development rows may enter evaluation")
    declared_animals = sorted(development_evaluation_animal_ids)
    row_animals = {record["lineage"]["animal_id"] for record in manifest["rows"]}
    if (
        not declared_animals
        or len(declared_animals) != len(set(declared_animals))
        or any(not isinstance(value, str) or not value for value in declared_animals)
        or row_animals != set(declared_animals)
    ):
        raise ValueError("declared development animals differ from the complete cache")
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint, catalogue, device=device
    )
    capability = loaded["inference_contract"].get("finite_psf_capability")
    if capability != manifest["finite_psf_capability"]:
        raise ValueError("checkpoint and development cache finite-PSF capabilities differ")
    training_animals = set(loaded["checkpoint_receipt"]["training_receipt"]["training_animal_ids"])
    if training_animals & row_animals:
        raise ValueError("development animals leak into training identities")
    checkpoint_binding = {
        "path": str(checkpoint),
        "file_sha256": loaded["checkpoint_file_sha256"],
        "checkpoint_id": loaded["checkpoint_id"],
        "checkpoint_binding_id": loaded["checkpoint_binding_id"],
        "model_state_sha256": loaded["model_state_sha256"],
        "training_receipt": inference_v3._json(loaded["checkpoint_receipt"]["training_receipt"]),
        "training_animal_ids": sorted(training_animals),
        "inference_contract": inference_v3._json(loaded["inference_contract"]),
    }
    config = {
        "top_k": int(top_k),
        "refinement_steps": int(refinement_steps),
        "pose_only_steps": int(pose_only_steps),
        "retrieval_shape_h_w": [int(value) for value in retrieval_shape_h_w],
        "catalogue_chunk_size": int(catalogue_chunk_size),
        "gauss_hermite_order": int(gauss_hermite_order),
        "evaluation_seed": int(evaluation_seed),
        "evaluation_rng_use": "provenance-only; inference is deterministic",
        "minimum_jacobian": float(minimum_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "omitted_mass_failure_threshold": metrics_v3.DEFAULT_OMITTED_MASS_FAILURE_THRESHOLD,
        "inference_batch_size": 1,
        "all_cache_rows_evaluated": True,
        "per_row_schedule_source": "finite_psf_contract",
        "global_schedule_fallback": None,
        "catalogue_feature_cache": None,
    }
    if (
        config["top_k"] < 1
        or config["top_k"] > int(catalogue["counts"]["cell_count"])
        or config["refinement_steps"] < 1
        or not 0 <= config["pose_only_steps"] <= config["refinement_steps"]
        or min(config["retrieval_shape_h_w"]) < 4
        or config["catalogue_chunk_size"] < 1
        or not 3 <= config["gauss_hermite_order"] <= 9
        or config["minimum_jacobian"] <= 0.0
        or config["maximum_cycle_error_px"] <= 0.0
    ):
        raise ValueError("finite development configuration is invalid")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw_predictions"
    raw_root.mkdir()
    atlas_shape = tuple(torch.as_tensor(atlas_volume_c_ap_dv_ml).shape[-3:])
    row_reports = []
    for order, cache_index in enumerate(selected_indices):
        record = manifest["rows"][cache_index]
        row = row_cache_v4._load_record(cache_root, record, manifest)
        contract = row["finite_psf_contract"]
        psf_v4.verify_training_row_psf_contract_v4(contract, capability=capability)
        session = inference_v3.prepare_arbitrary_plane_inference_session_v3(
            loaded,
            atlas_volume_c_ap_dv_ml,
            catalogue,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            contract["axial_offsets_um"],
            contract["axial_weights"],
            annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
            catalogue_feature_cache=None,
        )
        runtime_contract = session["runtime_inference_contract"]
        schedule_binding = _row_schedule_binding(row, runtime_contract)
        if runtime_contract["finite_psf_runtime_contract"]["receipt_sha256"] != (
            psf_v4.runtime_schedule_contract_v4(
                contract["axial_offsets_um"], contract["axial_weights"], capability=capability
            )["receipt_sha256"]
        ):
            raise RuntimeError("row runtime schedule changed after session authentication")
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
            finite_psf_capability=capability,
            device="cpu",
        )
        metrics = metrics_v3._row_metrics(
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
        metrics["dense"] = _dense_summary(metrics["deformation"])
        abstained = bool(
            not metrics["deformation"]["map_component_has_refined_deformation"]
            or metrics["deformation"]["valid_pixel_count"] == 0
        )
        metrics["operational_abstention"] = abstained
        support_contract = row.get("upstream_reference", {}).get(
            "support_supervision_contract", {}
        )
        row_reports.append({
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
            "finite_psf_schedule_binding": schedule_binding,
            "supervision_disposition": {
                "pose_weight": support_contract.get("point_pose_supervision_weight"),
                "dense_weight": support_contract.get("dense_deformation_supervision_weight"),
                "marginal_or_empty_rows_retained": True,
            },
            "disposition": {
                "included_in_row_metrics": True,
                "included_in_animal_macro_denominators_where_metric_defined": True,
                "failed": metrics["overall_failure"],
                "abstained": abstained,
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
                "inference_receipt_sha256": result["inference_receipt_sha256"],
                "input_receipt": result["input_receipt"],
                "atlas_receipt": result["atlas_receipt"],
                "configuration_receipt": result["configuration_receipt"],
            },
            "metrics": metrics,
        })
        del session
    identities = {
        "training_row_ids": [item["training_row_id"] for item in row_reports],
        "development_evaluation_animal_ids": declared_animals,
        "animal_ids": [item["animal_id"] for item in row_reports],
        "specimen_ids": [item["specimen_id"] for item in row_reports],
        "experiment_ids": [item["experiment_id"] for item in row_reports],
        "synthetic_animal_ids": [item["synthetic_animal_id"] for item in row_reports],
        "section_ids": [item["section_id"] for item in row_reports],
        "synthetic_realization_ids": [item["synthetic_realization_id"] for item in row_reports],
    }
    failed = [item["training_row_id"] for item in row_reports if item["disposition"]["failed"]]
    abstained = [item["training_row_id"] for item in row_reports if item["disposition"]["abstained"]]
    payload = {
        "schema_version": FINITE_DEVELOPMENT_EVALUATION_V4_SCHEMA,
        "data_role": FINITE_DEVELOPMENT_EVALUATION_ROLE,
        "scientific_scope": (
            "internal synthetic animal-disjoint development only; not validation, "
            "qualification, calibration, public benchmarking, external validation, or final test"
        ),
        "experiment_scope": experiment_scope,
        "public_benchmark_accessed": False,
        "final_test_accessed": False,
        "external_validation_accessed": False,
        "calibration_fitted": False,
        "uncertainty_scope": UNCALIBRATED_SCOPE,
        "source_sha256": _source_receipts(),
        "configuration": config,
        "configuration_receipt_sha256": _sha(config),
        "cache_binding": {
            "directory": str(cache_root),
            "schema_version": manifest["schema_version"],
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "status": manifest["status"],
            "row_count": manifest["row_count"],
            "selected_row_indices": selected_indices,
            "finite_psf_capability": manifest["finite_psf_capability"],
            "finite_psf_run_contract": manifest["finite_psf_run_contract"],
            "freeze_audit": manifest["freeze_audit"],
            "generator_binding": manifest["generator_binding"],
            "generation_config": manifest["generation_config"],
            "seed_record": manifest["seed_record"],
        },
        "checkpoint_binding": checkpoint_binding,
        "catalogue_binding": {
            "catalogue_id": catalogue["catalogue_id"],
            "receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        },
        "identities": identities,
        "row_accounting": {
            "selected_row_count": len(selected_indices),
            "reported_row_count": len(row_reports),
            "no_rows_dropped": len(row_reports) == len(selected_indices),
            "marginal_or_empty_rows_retained": True,
            "failed_row_count": len(failed),
            "failed_training_row_ids": failed,
            "abstained_row_count": len(abstained),
            "abstained_training_row_ids": abstained,
        },
        "metric_families": [
            "physical_landmark", "plane_offset", "plane_angle", "finite_frame_angle",
            "regional_overlap", "dense_deformation", "failures", "input_mode",
            "raw_uncalibrated_uncertainty",
        ],
        "row_reports": row_reports,
        "animal_macro_metrics": _animal_macro(row_reports),
        "mode_stratified_metrics": _mode_stratified(row_reports),
        "learned_dependencies": {
            "prior_model_weights": [], "prior_features": [], "prior_pseudolabels": []
        },
    }
    report = {**payload, "receipt_sha256": _sha(payload)}
    report_path = output_root / "finite_development_evaluation_report.json"
    _atomic_json_new(report_path, report)
    bundle_payload = {
        "schema_version": FINITE_DEVELOPMENT_EVALUATION_BUNDLE_V4_SCHEMA,
        "output_directory": str(output_root),
        "report_relative_path": report_path.relative_to(output_root).as_posix(),
        "report_file_sha256": _file_sha256(report_path),
        "report_receipt_sha256": report["receipt_sha256"],
        "source_sha256": _source_receipts(),
    }
    bundle = {**bundle_payload, "receipt_sha256": _sha(bundle_payload)}
    _atomic_json_new(output_root / "bundle_receipt.json", bundle)
    verify_arbitrary_plane_finite_development_evaluation_v4(output_root, catalogue=catalogue)
    return bundle


def verify_arbitrary_plane_finite_development_evaluation_v4(
    output_directory, *, catalogue=None
):
    """Independently bind every cached row, per-row schedule, and raw prediction."""
    output_root = _i_path(output_directory, must_exist=True)
    bundle_path = output_root / "bundle_receipt.json"
    bundle = json.loads(bundle_path.read_text("ascii"))
    bundle_payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    if (
        bundle.get("schema_version") != FINITE_DEVELOPMENT_EVALUATION_BUNDLE_V4_SCHEMA
        or bundle.get("receipt_sha256") != _sha(bundle_payload)
        or bundle.get("output_directory") != str(output_root)
        or bundle.get("source_sha256") != _source_receipts()
    ):
        raise ValueError("finite development bundle failed authentication")
    report_path = (output_root / bundle["report_relative_path"]).resolve(strict=True)
    if output_root not in report_path.parents or _file_sha256(report_path) != bundle["report_file_sha256"]:
        raise ValueError("finite development report file hash differs")
    report = json.loads(report_path.read_text("ascii"))
    report_payload = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if (
        report.get("schema_version") != FINITE_DEVELOPMENT_EVALUATION_V4_SCHEMA
        or report.get("receipt_sha256") != _sha(report_payload)
        or report.get("receipt_sha256") != bundle["report_receipt_sha256"]
        or report.get("source_sha256") != _source_receipts()
        or report.get("configuration_receipt_sha256") != _sha(report.get("configuration", {}))
        or report.get("public_benchmark_accessed") is not False
        or report.get("final_test_accessed") is not False
        or report.get("external_validation_accessed") is not False
        or report.get("calibration_fitted") is not False
        or report.get("uncertainty_scope") != UNCALIBRATED_SCOPE
        or report.get("learned_dependencies") != {
            "prior_model_weights": [], "prior_features": [], "prior_pseudolabels": []
        }
    ):
        raise ValueError("finite development report failed authentication")
    cache = report["cache_binding"]
    manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        cache["directory"], expected_receipt_sha256=cache["manifest_receipt_sha256"]
    )
    expected_indices = list(range(int(manifest["row_count"])))
    rows = report["row_reports"]
    if (
        manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS
        or report["experiment_scope"] != _experiment_scope(manifest)
        or cache["selected_row_indices"] != expected_indices
        or cache["finite_psf_capability"] != manifest["finite_psf_capability"]
        or cache["finite_psf_run_contract"] != manifest["finite_psf_run_contract"]
        or cache["freeze_audit"] != manifest["freeze_audit"]
        or len(rows) != manifest["row_count"]
        or report["row_accounting"]["no_rows_dropped"] is not True
        or report["row_accounting"]["marginal_or_empty_rows_retained"] is not True
    ):
        raise ValueError("finite development cache or all-row accounting differs")
    checkpoint = report["checkpoint_binding"]
    checkpoint_path = _i_path(checkpoint["path"], must_exist=True)
    if _file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
        raise ValueError("finite development checkpoint hash differs")
    if catalogue is not None:
        loaded = inference_v3.load_arbitrary_plane_inference_v3(
            checkpoint_path, catalogue, device="cpu"
        )
        if any(loaded[name] != checkpoint[name] for name in (
            "checkpoint_id", "checkpoint_binding_id", "model_state_sha256"
        )) or loaded["inference_contract"] != checkpoint["inference_contract"]:
            raise ValueError("finite development checkpoint binding differs")
        if report["catalogue_binding"] != {
            "catalogue_id": catalogue["catalogue_id"],
            "receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        }:
            raise ValueError("finite development catalogue binding differs")
    declared_animals = report["identities"]["development_evaluation_animal_ids"]
    if set(checkpoint["training_animal_ids"]) & set(declared_animals):
        raise ValueError("finite development animal leakage into training IDs")
    raw_paths = set()
    for index, row_report in enumerate(rows):
        row = row_cache_v4._load_record(
            _i_path(cache["directory"], must_exist=True), manifest["rows"][index], manifest
        )
        schedule = row_report["finite_psf_schedule_binding"]
        expected_runtime = inference_v3.make_runtime_inference_contract_v4(
            checkpoint["inference_contract"],
            row["finite_psf_contract"]["axial_offsets_um"],
            row["finite_psf_contract"]["axial_weights"],
        )
        if (
            row_report["cache_row_index"] != index
            or row_report["training_row_id"] != row["training_row_id"]
            or row_report["training_row_receipt_sha256"] != row["receipt_sha256"]
            or row_report["animal_id"] != row["lineage"]["animal_id"]
            or row_report["specimen_id"] != row["lineage"]["specimen_id"]
            or row_report["experiment_id"] != row["lineage"]["experiment_id"]
            or row_report["section_id"] != row["lineage"]["section_id"]
            or row_report["synthetic_realization_id"] != row["synthetic_realization_id"]
            or schedule != _row_schedule_binding(row, expected_runtime)
            or row_report["disposition"]["no_silent_drop"] is not True
            or row_report["disposition"]["included_in_row_metrics"] is not True
        ):
            raise ValueError("finite development row identity or schedule differs")
        raw = row_report["raw_prediction"]
        raw_path = (output_root / raw["relative_path"]).resolve(strict=True)
        if output_root not in raw_path.parents or raw_path in raw_paths:
            raise ValueError("finite development raw-prediction path is invalid")
        raw_paths.add(raw_path)
        if _file_sha256(raw_path) != raw["file_sha256"]:
            raise ValueError("finite development raw-prediction hash differs")
        artifact = torch.load(raw_path, map_location="cpu", weights_only=True)
        cached_input = torch.from_numpy(
            np.asarray(row["arrays"]["model_input_channels_float32"])
        ).permute(2, 0, 1)[None]
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
            artifact.get("checkpoint_id") != checkpoint["checkpoint_id"]
            or artifact.get("checkpoint_binding_id") != checkpoint["checkpoint_binding_id"]
            or artifact.get("catalogue_id") != report["catalogue_binding"]["catalogue_id"]
            or artifact.get("identifiers") != identifiers
            or artifact.get("input_receipt", {}).get("identifiers") != identifiers
            or artifact.get("input_receipt", {}).get("raw_input_receipt")
            != inference_v3._tensor_receipt(cached_input)
            or artifact.get("raw_prediction", {}).get("lineage") != [lineage]
            or artifact.get("raw_prediction_receipt") != raw["prediction_receipt"]
            or inference_v3._prediction_receipt(artifact.get("raw_prediction")) != raw["prediction_receipt"]
            or artifact.get("input_receipt") != raw["input_receipt"]
            or artifact.get("configuration_receipt") != raw["configuration_receipt"]
            or raw["atlas_receipt"] != expected_runtime
        ):
            raise ValueError("finite development raw-prediction receipt differs")
        inference_receipt_payload = {
            "schema_version": inference_v3.INFERENCE_V3_SCHEMA,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_binding_id": checkpoint["checkpoint_binding_id"],
            "checkpoint_file_sha256": checkpoint["file_sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "catalogue_id": report["catalogue_binding"]["catalogue_id"],
            **identifiers,
            "input_receipt": raw["input_receipt"],
            "atlas_receipt": raw["atlas_receipt"],
            "configuration_receipt": raw["configuration_receipt"],
            "raw_prediction_receipt": raw["prediction_receipt"],
            "raw_prediction_path": str(raw_path),
            "raw_prediction_file_sha256": raw["file_sha256"],
        }
        if inference_v3._sha(inference_receipt_payload) != raw["inference_receipt_sha256"]:
            raise ValueError("finite development inference receipt differs")
    failed = [item["training_row_id"] for item in rows if item["disposition"]["failed"]]
    abstained = [item["training_row_id"] for item in rows if item["disposition"]["abstained"]]
    if (
        report["row_accounting"] != {
            "selected_row_count": len(rows),
            "reported_row_count": len(rows),
            "no_rows_dropped": True,
            "marginal_or_empty_rows_retained": True,
            "failed_row_count": len(failed),
            "failed_training_row_ids": failed,
            "abstained_row_count": len(abstained),
            "abstained_training_row_ids": abstained,
        }
        or report["animal_macro_metrics"] != _animal_macro(rows)
        or report["mode_stratified_metrics"] != _mode_stratified(rows)
    ):
        raise ValueError("finite development metrics or accounting changed")
    expected_identities = {
        "training_row_ids": [item["training_row_id"] for item in rows],
        "development_evaluation_animal_ids": declared_animals,
        "animal_ids": [item["animal_id"] for item in rows],
        "specimen_ids": [item["specimen_id"] for item in rows],
        "experiment_ids": [item["experiment_id"] for item in rows],
        "synthetic_animal_ids": [item["synthetic_animal_id"] for item in rows],
        "section_ids": [item["section_id"] for item in rows],
        "synthetic_realization_ids": [item["synthetic_realization_id"] for item in rows],
    }
    if (
        report["identities"] != expected_identities
        or set(declared_animals) != {item["animal_id"] for item in rows}
    ):
        raise ValueError("finite development identity summary differs")
    return True


__all__ = [
    "FINITE_DEVELOPMENT_EVALUATION_BUNDLE_V4_SCHEMA",
    "FINITE_DEVELOPMENT_EVALUATION_V4_SCHEMA",
    "UNCALIBRATED_SCOPE",
    "run_arbitrary_plane_finite_development_evaluation_v4",
    "verify_arbitrary_plane_finite_development_evaluation_v4",
]
