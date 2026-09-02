"""Freeze, evaluate, and bind one completed authentic arbitrary-plane V3 run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_development_evaluation_v3 as evaluation_v3
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_run_export_v3 as export_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3


AUTHENTIC_POSTRUN_BUNDLE_V3_SCHEMA = (
    "anatomy-tracker.authentic-arbitrary-plane-postrun-bundle/v3"
)
POSTRUN_SCIENTIFIC_SCOPE = (
    "internal synthetic animal-disjoint development only; uncalibrated and not a "
    "public benchmark, qualification, external validation, or final test"
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("post-training receipts require finite values")
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path, *, must_exist=False):
    target = Path(path).resolve(strict=must_exist)
    if os.path.splitdrive(str(target))[0].upper() != "I:":
        raise ValueError("authentic post-training artifacts must use only I:")
    return target


def _atomic_json_new(path, value):
    target = _i_path(path)
    if os.path.lexists(target):
        raise FileExistsError("post-training JSON target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError("post-training JSON temporary target already exists")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _source_receipts():
    paths = {
        "postrun_orchestrator": Path(__file__).resolve(),
        "bundle_verifier": Path(__file__).with_name(
            "verify_arbitrary_plane_authentic_package_v3.py"
        ).resolve(),
        "training_run_export": Path(export_v3.__file__).resolve(),
        "inference": Path(inference_v3.__file__).resolve(),
        "development_evaluation": Path(evaluation_v3.__file__).resolve(),
    }
    return {name: _file_sha256(path) for name, path in sorted(paths.items())}


def _completed_context(run_directory):
    context = runner_v3.load_training_run_v3(_i_path(run_directory, must_exist=True))
    state = context["run_state"]
    manifest = context["manifest"]
    target = int(manifest["runner_config"]["target_applied_steps"])
    if int(state["applied_step_count"]) != target:
        raise ValueError("post-training export requires the exact completed training target")
    return context


def _component_file(output_root, relative_path):
    path = (output_root / relative_path).resolve()
    if output_root not in path.parents:
        raise ValueError("post-training component path escapes the bundle")
    return path


def run_arbitrary_plane_authentic_postrun_v3(
    run_directory,
    development_cache_directory,
    output_directory,
    *,
    atlas_semantics,
    development_evaluation_animal_ids,
    annotation_volume_ap_dv_ml=None,
    top_k=4,
    refinement_steps=3,
    pose_only_steps=2,
    retrieval_shape_h_w=(48, 64),
    catalogue_chunk_size=128,
    gauss_hermite_order=5,
    evaluation_seed=0,
    minimum_jacobian=evaluation_v3.DEFAULT_MINIMUM_JACOBIAN,
    maximum_cycle_error_px=evaluation_v3.DEFAULT_MAXIMUM_CYCLE_ERROR_PX,
    feature_cache_build_chunk_size=128,
    device="cpu",
):
    """Create one immutable exact-checkpoint internal-development bundle."""
    context = _completed_context(run_directory)
    output_root = _i_path(output_directory)
    if os.path.lexists(output_root):
        raise FileExistsError("post-training output directory must be new")
    development_cache = _i_path(development_cache_directory, must_exist=True)
    animals = list(development_evaluation_animal_ids)
    if (
        not animals
        or len(animals) != len(set(animals))
        or any(not isinstance(value, str) or not value for value in animals)
    ):
        raise ValueError("development-evaluation animal IDs must be unique nonempty strings")
    animals = sorted(animals)

    output_root.mkdir(parents=True)
    checkpoint_relative = "inference/completed_checkpoint.pt"
    export_report_relative = "inference/export_report.json"
    feature_cache_relative = "inference/complete_catalogue_features.pt"
    evaluation_relative = "internal_development_evaluation"
    checkpoint_path = _component_file(output_root, checkpoint_relative)
    export_report_path = _component_file(output_root, export_report_relative)
    feature_cache_path = _component_file(output_root, feature_cache_relative)
    evaluation_root = _component_file(output_root, evaluation_relative)

    export_report = export_v3.export_training_run_to_inference_checkpoint_v3(
        context["run_root"],
        checkpoint_path,
        atlas_semantics=atlas_semantics,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        safe_load_device="cpu",
    )
    if export_report["export_status"] != "completed":
        raise RuntimeError("post-training checkpoint export was not completed")
    _atomic_json_new(export_report_path, export_report)

    run_binding = export_report["dataset_provenance"]["run_binding"]
    if (
        run_binding["run_id"] != context["manifest"]["run_id"]
        or run_binding["run_manifest_receipt_sha256"]
        != context["manifest"]["receipt_sha256"]
        or run_binding["run_state_receipt_sha256"]
        != context["run_state"]["receipt_sha256"]
        or int(run_binding["applied_step_count"])
        != int(context["run_state"]["applied_step_count"])
    ):
        raise RuntimeError("completed run changed across exact checkpoint export")

    catalogue = context["catalogue"]
    atlas = context["atlas_volume"]
    geometry = catalogue["support_geometry"]
    psf = context["manifest"]["finite_psf_contract"]
    del context["training_state"], context["training_reports"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint_path, catalogue, device=device
    )
    feature_cache = inference_v3.make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        geometry["origin_ap_dv_ml_um"],
        geometry["voxel_size_ap_dv_ml_um"],
        psf["axial_offsets_um"],
        psf["axial_weights"],
        feature_cache_path,
        retrieval_shape_h_w=tuple(int(value) for value in retrieval_shape_h_w),
        build_chunk_size=int(feature_cache_build_chunk_size),
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
    )
    feature_cache_binding = {
        "relative_path": feature_cache_relative,
        "file_sha256": feature_cache["cache_file_sha256"],
        "cache_id": feature_cache["cache_receipt"]["cache_id"],
        "cache_receipt": _plain(feature_cache["cache_receipt"]),
    }
    del loaded, feature_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluation_bundle = evaluation_v3.run_arbitrary_plane_development_evaluation_v3(
        development_cache,
        checkpoint_path,
        catalogue,
        atlas,
        geometry["origin_ap_dv_ml_um"],
        geometry["voxel_size_ap_dv_ml_um"],
        psf["axial_offsets_um"],
        psf["axial_weights"],
        evaluation_root,
        development_evaluation_animal_ids=animals,
        row_indices=None,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        top_k=int(top_k),
        refinement_steps=int(refinement_steps),
        pose_only_steps=int(pose_only_steps),
        retrieval_shape_h_w=tuple(int(value) for value in retrieval_shape_h_w),
        catalogue_chunk_size=int(catalogue_chunk_size),
        gauss_hermite_order=int(gauss_hermite_order),
        evaluation_seed=int(evaluation_seed),
        minimum_jacobian=float(minimum_jacobian),
        maximum_cycle_error_px=float(maximum_cycle_error_px),
        catalogue_feature_cache_path=feature_cache_path,
        device=device,
    )
    evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(
        evaluation_root, catalogue=catalogue
    )
    development_manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        development_cache
    )
    configuration = {
        "all_development_cache_rows_evaluated": True,
        "development_evaluation_animal_ids": animals,
        "top_k": int(top_k),
        "refinement_steps": int(refinement_steps),
        "pose_only_steps": int(pose_only_steps),
        "retrieval_shape_h_w": [int(value) for value in retrieval_shape_h_w],
        "catalogue_chunk_size": int(catalogue_chunk_size),
        "gauss_hermite_order": int(gauss_hermite_order),
        "evaluation_seed": int(evaluation_seed),
        "minimum_jacobian": float(minimum_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "feature_cache_build_chunk_size": int(feature_cache_build_chunk_size),
        "device": str(torch.device(device)),
    }
    bundle_payload = {
        "schema_version": AUTHENTIC_POSTRUN_BUNDLE_V3_SCHEMA,
        "scientific_scope": POSTRUN_SCIENTIFIC_SCOPE,
        "output_directory": str(output_root),
        "source_sha256": _source_receipts(),
        "configuration": configuration,
        "configuration_receipt_sha256": _sha(configuration),
        "run_binding": {
            "directory": str(context["run_root"]),
            "run_id": context["manifest"]["run_id"],
            "run_manifest_receipt_sha256": context["manifest"]["receipt_sha256"],
            "run_state_receipt_sha256": context["run_state"]["receipt_sha256"],
            "target_applied_steps": int(
                context["manifest"]["runner_config"]["target_applied_steps"]
            ),
            "applied_step_count": int(context["run_state"]["applied_step_count"]),
            "status": "completed",
        },
        "development_cache_binding": {
            "directory": str(development_cache),
            "manifest_receipt_sha256": development_manifest["receipt_sha256"],
            "row_count": int(development_manifest["row_count"]),
            "selected_row_indices": list(range(int(development_manifest["row_count"]))),
        },
        "artifacts": {
            "checkpoint": {
                "relative_path": checkpoint_relative,
                "file_sha256": export_report["checkpoint"]["file_sha256"],
                "checkpoint_id": export_report["checkpoint"]["checkpoint_id"],
                "checkpoint_binding_id": export_report["checkpoint"][
                    "checkpoint_binding_id"
                ],
                "model_state_sha256": export_report["checkpoint"][
                    "model_state_sha256"
                ],
            },
            "export_report": {
                "relative_path": export_report_relative,
                "file_sha256": _file_sha256(export_report_path),
                "receipt_sha256": export_report["receipt_sha256"],
            },
            "catalogue_feature_cache": feature_cache_binding,
            "development_evaluation": {
                "relative_directory": evaluation_relative,
                "bundle_relative_path": "bundle_receipt.json",
                "bundle_file_sha256": _file_sha256(
                    evaluation_root / "bundle_receipt.json"
                ),
                "bundle_receipt_sha256": evaluation_bundle["receipt_sha256"],
            },
        },
        "calibration": {
            "status": export_v3.UNCALIBRATED_STATUS,
            "calibration_receipt": None,
        },
        "public_benchmark_accessed": False,
        "final_test_accessed": False,
        "external_validation_accessed": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    bundle = {**bundle_payload, "receipt_sha256": _sha(bundle_payload)}
    _atomic_json_new(output_root / "post_training_bundle_receipt.json", bundle)

    from training.verify_arbitrary_plane_authentic_package_v3 import (
        verify_arbitrary_plane_authentic_package_v3,
    )

    verify_arbitrary_plane_authentic_package_v3(output_root)
    return bundle


__all__ = [
    "AUTHENTIC_POSTRUN_BUNDLE_V3_SCHEMA",
    "POSTRUN_SCIENTIFIC_SCOPE",
    "run_arbitrary_plane_authentic_postrun_v3",
]
