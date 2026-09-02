"""Export, evaluate, and bind one completed finite-v4 development run."""

from __future__ import annotations

from pathlib import Path

import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.arbitrary_plane_finite_run_export_v4 as export_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_row_cache_v4 as row_cache_v4
import training.arbitrary_plane_run_export_v3 as export_v3


FINITE_POSTRUN_BUNDLE_V4_SCHEMA = (
    "anatomy-tracker.authentic-arbitrary-plane-finite-postrun-bundle/v4"
)
FINITE_POSTRUN_SCIENTIFIC_SCOPE = (
    "internal synthetic animal-disjoint development only; uncalibrated and not a "
    "public benchmark, qualification, external validation, or final test"
)


def _sha(value):
    return evaluation_v4._sha(value)


def _file_sha256(path):
    return evaluation_v4._file_sha256(path)


def _i_path(path, *, must_exist=False):
    return evaluation_v4._i_path(path, must_exist=must_exist)


def _source_receipts():
    paths = {
        "postrun": Path(__file__).resolve(),
        "verifier": Path(__file__).with_name(
            "verify_arbitrary_plane_finite_package_v4.py"
        ).resolve(),
        "export": Path(export_v4.__file__).resolve(),
        "evaluation": Path(evaluation_v4.__file__).resolve(),
        "runner": Path(runner_v4.__file__).resolve(),
        "cache": Path(row_cache_v4.__file__).resolve(),
    }
    return {name: _file_sha256(path) for name, path in sorted(paths.items())}


def run_arbitrary_plane_finite_postrun_v4(
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
    minimum_jacobian=evaluation_v4.metrics_v3.DEFAULT_MINIMUM_JACOBIAN,
    maximum_cycle_error_px=evaluation_v4.metrics_v3.DEFAULT_MAXIMUM_CYCLE_ERROR_PX,
    device="cpu",
):
    """Create one immutable capability-bound all-row development package."""
    context = runner_v4.load_finite_training_run_v4(
        _i_path(run_directory, must_exist=True)
    )
    manifest = context["manifest"]
    state = context["run_state"]
    target = int(manifest["runner_config"]["target_applied_steps"])
    if int(state["applied_step_count"]) != target:
        raise ValueError("finite-v4 postrun requires the exact completed target")
    development_root = _i_path(development_cache_directory, must_exist=True)
    development_manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        development_root
    )
    if (
        development_manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS
        or development_manifest["finite_psf_capability"]
        != manifest["finite_psf_capability"]
        or development_manifest["finite_psf_run_contract"]
        != context["cache_manifest"]["finite_psf_run_contract"]
    ):
        raise ValueError("training and development finite-PSF experiment scopes differ")
    output_root = _i_path(output_directory)
    if output_root.exists():
        raise FileExistsError("finite-v4 postrun output directory must be new")
    output_root.mkdir(parents=True)
    checkpoint_relative = "inference/completed_checkpoint.pt"
    export_relative = "inference/export_report.json"
    evaluation_relative = "internal_development_evaluation"
    checkpoint_path = output_root / checkpoint_relative
    export_path = output_root / export_relative
    evaluation_root = output_root / evaluation_relative
    export_report = export_v4.export_finite_training_run_to_inference_checkpoint_v4(
        context["run_root"],
        checkpoint_path,
        atlas_semantics=atlas_semantics,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        safe_load_device="cpu",
    )
    evaluation_bundle = (
        evaluation_v4.run_arbitrary_plane_finite_development_evaluation_v4(
            development_root,
            checkpoint_path,
            context["catalogue"],
            context["atlas_volume"],
            context["catalogue"]["support_geometry"]["origin_ap_dv_ml_um"],
            context["catalogue"]["support_geometry"]["voxel_size_ap_dv_ml_um"],
            evaluation_root,
            development_evaluation_animal_ids=development_evaluation_animal_ids,
            annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
            top_k=top_k,
            refinement_steps=refinement_steps,
            pose_only_steps=pose_only_steps,
            retrieval_shape_h_w=retrieval_shape_h_w,
            catalogue_chunk_size=catalogue_chunk_size,
            gauss_hermite_order=gauss_hermite_order,
            evaluation_seed=evaluation_seed,
            minimum_jacobian=minimum_jacobian,
            maximum_cycle_error_px=maximum_cycle_error_px,
            device=device,
        )
    )
    evaluation_v4._atomic_json_new(export_path, export_report)
    configuration = {
        "all_development_cache_rows_evaluated": True,
        "development_evaluation_animal_ids": sorted(development_evaluation_animal_ids),
        "top_k": int(top_k),
        "refinement_steps": int(refinement_steps),
        "pose_only_steps": int(pose_only_steps),
        "retrieval_shape_h_w": [int(value) for value in retrieval_shape_h_w],
        "catalogue_chunk_size": int(catalogue_chunk_size),
        "gauss_hermite_order": int(gauss_hermite_order),
        "evaluation_seed": int(evaluation_seed),
        "minimum_jacobian": float(minimum_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "device": str(device),
        "per_row_schedule_source": "finite_psf_contract",
        "global_schedule_fallback": None,
    }
    payload = {
        "schema_version": FINITE_POSTRUN_BUNDLE_V4_SCHEMA,
        "scientific_scope": FINITE_POSTRUN_SCIENTIFIC_SCOPE,
        "output_directory": str(output_root),
        "source_sha256": _source_receipts(),
        "configuration": configuration,
        "configuration_receipt_sha256": _sha(configuration),
        "run_binding": {
            "directory": str(context["run_root"]),
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "run_state_receipt_sha256": state["receipt_sha256"],
            "target_applied_steps": target,
            "applied_step_count": int(state["applied_step_count"]),
            "status": "completed",
            "finite_psf_capability": manifest["finite_psf_capability"],
            "finite_psf_run_contract": context["cache_manifest"]["finite_psf_run_contract"],
            "training_cache_manifest_receipt_sha256": context["cache_manifest"]["receipt_sha256"],
        },
        "development_cache_binding": {
            "directory": str(development_root),
            "manifest_receipt_sha256": development_manifest["receipt_sha256"],
            "row_count": int(development_manifest["row_count"]),
            "selected_row_indices": list(range(int(development_manifest["row_count"]))),
            "finite_psf_capability": development_manifest["finite_psf_capability"],
            "finite_psf_run_contract": development_manifest["finite_psf_run_contract"],
            "freeze_audit": development_manifest["freeze_audit"],
        },
        "artifacts": {
            "checkpoint": {
                **export_report["checkpoint"],
                "relative_path": checkpoint_relative,
            },
            "export_report": {
                "relative_path": export_relative,
                "file_sha256": _file_sha256(export_path),
                "receipt_sha256": export_report["receipt_sha256"],
            },
            "development_evaluation": {
                "relative_directory": evaluation_relative,
                "bundle_relative_path": "bundle_receipt.json",
                "bundle_file_sha256": _file_sha256(evaluation_root / "bundle_receipt.json"),
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
    bundle = {**payload, "receipt_sha256": _sha(payload)}
    evaluation_v4._atomic_json_new(
        output_root / "finite_postrun_bundle_receipt.json", bundle
    )
    from training.verify_arbitrary_plane_finite_package_v4 import (
        verify_arbitrary_plane_finite_package_v4,
    )
    verify_arbitrary_plane_finite_package_v4(output_root)
    return bundle


__all__ = [
    "FINITE_POSTRUN_BUNDLE_V4_SCHEMA",
    "FINITE_POSTRUN_SCIENTIFIC_SCOPE",
    "run_arbitrary_plane_finite_postrun_v4",
]
