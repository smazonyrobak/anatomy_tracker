"""Independent verifier for a completed finite-v4 development package."""

from __future__ import annotations

import json

import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.arbitrary_plane_finite_run_export_v4 as export_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v4 as row_cache_v4
import training.arbitrary_plane_run_export_v3 as export_v3
import training.run_arbitrary_plane_finite_postrun_v4 as postrun_v4


def _inside(root, relative_path, *, directory=False):
    path = (root / relative_path).resolve(strict=True)
    if root not in path.parents or (path.is_dir() if directory else path.is_file()) is not True:
        raise ValueError("finite-v4 package component path is invalid")
    return path


def verify_arbitrary_plane_finite_package_v4(output_directory):
    """Cross-bind run, caches, capability checkpoint, rows, and raw predictions."""
    root = postrun_v4._i_path(output_directory, must_exist=True)
    bundle_path = _inside(root, "finite_postrun_bundle_receipt.json")
    bundle = json.loads(bundle_path.read_text("ascii"))
    payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    if (
        bundle.get("schema_version") != postrun_v4.FINITE_POSTRUN_BUNDLE_V4_SCHEMA
        or bundle.get("scientific_scope") != postrun_v4.FINITE_POSTRUN_SCIENTIFIC_SCOPE
        or bundle.get("output_directory") != str(root)
        or bundle.get("source_sha256") != postrun_v4._source_receipts()
        or bundle.get("configuration_receipt_sha256")
        != postrun_v4._sha(bundle.get("configuration", {}))
        or bundle.get("receipt_sha256") != postrun_v4._sha(payload)
        or bundle.get("calibration") != {
            "status": export_v3.UNCALIBRATED_STATUS, "calibration_receipt": None
        }
        or bundle.get("public_benchmark_accessed") is not False
        or bundle.get("final_test_accessed") is not False
        or bundle.get("external_validation_accessed") is not False
        or any(bundle.get(name) != [] for name in (
            "prior_model_weight_dependencies", "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        ))
        or bundle.get("configuration", {}).get("global_schedule_fallback") is not None
        or bundle.get("configuration", {}).get("all_development_cache_rows_evaluated") is not True
    ):
        raise ValueError("finite-v4 postrun bundle failed authentication")
    run = bundle["run_binding"]
    context = runner_v4.load_finite_training_run_v4(
        postrun_v4._i_path(run["directory"], must_exist=True)
    )
    manifest = context["manifest"]
    state = context["run_state"]
    expected_run = {
        "directory": str(context["run_root"]),
        "run_id": manifest["run_id"],
        "run_manifest_receipt_sha256": manifest["receipt_sha256"],
        "run_state_receipt_sha256": state["receipt_sha256"],
        "target_applied_steps": int(manifest["runner_config"]["target_applied_steps"]),
        "applied_step_count": int(state["applied_step_count"]),
        "status": "completed",
        "finite_psf_capability": manifest["finite_psf_capability"],
        "finite_psf_run_contract": context["cache_manifest"]["finite_psf_run_contract"],
        "training_cache_manifest_receipt_sha256": context["cache_manifest"]["receipt_sha256"],
    }
    if run != expected_run or run["applied_step_count"] != run["target_applied_steps"]:
        raise ValueError("finite-v4 completed-run binding differs")
    artifacts = bundle["artifacts"]
    checkpoint_record = artifacts["checkpoint"]
    checkpoint_path = _inside(root, checkpoint_record["relative_path"])
    export_record = artifacts["export_report"]
    export_path = _inside(root, export_record["relative_path"])
    if postrun_v4._file_sha256(export_path) != export_record["file_sha256"]:
        raise ValueError("finite-v4 export report hash differs")
    export_report = json.loads(export_path.read_text("ascii"))
    export_v4.verify_finite_training_run_inference_export_report_v4(export_report)
    runner_v4.verify_finite_training_run_export_receipt_v4(
        export_report["finite_training_run_export_receipt"], run["directory"]
    )
    if (
        export_report["receipt_sha256"] != export_record["receipt_sha256"]
        or export_report["checkpoint"]["path"] != str(checkpoint_path)
        or any(export_report["checkpoint"][name] != checkpoint_record[name] for name in (
            "file_sha256", "checkpoint_id", "checkpoint_binding_id", "model_state_sha256"
        ))
        or export_report["dataset_provenance"]["run_binding"]["run_state_receipt_sha256"]
        != run["run_state_receipt_sha256"]
        or export_report["dataset_provenance"]["finite_psf_capability"]
        != run["finite_psf_capability"]
    ):
        raise ValueError("finite-v4 export/run/capability cross-binding differs")
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint_path, context["catalogue"], device="cpu"
    )
    if any(loaded[name] != checkpoint_record[name] for name in (
        "checkpoint_id", "checkpoint_binding_id", "model_state_sha256"
    )) or loaded["checkpoint_file_sha256"] != checkpoint_record["file_sha256"]:
        raise ValueError("finite-v4 capability checkpoint identity differs")
    development = bundle["development_cache_binding"]
    development_manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        postrun_v4._i_path(development["directory"], must_exist=True),
        expected_receipt_sha256=development["manifest_receipt_sha256"],
    )
    expected_indices = list(range(int(development_manifest["row_count"])))
    expected_development = {
        "directory": str(postrun_v4._i_path(development["directory"], must_exist=True)),
        "manifest_receipt_sha256": development_manifest["receipt_sha256"],
        "row_count": int(development_manifest["row_count"]),
        "selected_row_indices": expected_indices,
        "finite_psf_capability": development_manifest["finite_psf_capability"],
        "finite_psf_run_contract": development_manifest["finite_psf_run_contract"],
        "freeze_audit": development_manifest["freeze_audit"],
    }
    if (
        development != expected_development
        or development["finite_psf_capability"] != run["finite_psf_capability"]
        or development["finite_psf_run_contract"] != run["finite_psf_run_contract"]
    ):
        raise ValueError("finite-v4 development cache scope differs")
    evaluation_record = artifacts["development_evaluation"]
    evaluation_root = _inside(
        root, evaluation_record["relative_directory"], directory=True
    )
    evaluation_bundle_path = _inside(
        evaluation_root, evaluation_record["bundle_relative_path"]
    )
    if postrun_v4._file_sha256(evaluation_bundle_path) != evaluation_record["bundle_file_sha256"]:
        raise ValueError("finite-v4 evaluation bundle hash differs")
    evaluation_bundle = json.loads(evaluation_bundle_path.read_text("ascii"))
    if evaluation_bundle["receipt_sha256"] != evaluation_record["bundle_receipt_sha256"]:
        raise ValueError("finite-v4 evaluation bundle receipt differs")
    evaluation_v4.verify_arbitrary_plane_finite_development_evaluation_v4(
        evaluation_root, catalogue=context["catalogue"]
    )
    evaluation_report = json.loads(
        (evaluation_root / "finite_development_evaluation_report.json").read_text("ascii")
    )
    evaluation_cache = evaluation_report["cache_binding"]
    if (
        evaluation_cache["directory"] != development["directory"]
        or evaluation_cache["schema_version"] != development_manifest["schema_version"]
        or evaluation_cache["manifest_receipt_sha256"]
        != development["manifest_receipt_sha256"]
        or evaluation_cache["status"] != development_manifest["status"]
        or evaluation_cache["row_count"] != development_manifest["row_count"]
        or evaluation_cache["selected_row_indices"] != expected_indices
        or evaluation_cache["finite_psf_capability"]
        != development_manifest["finite_psf_capability"]
        or evaluation_cache["finite_psf_run_contract"]
        != development_manifest["finite_psf_run_contract"]
        or evaluation_cache["freeze_audit"] != development_manifest["freeze_audit"]
        or evaluation_cache["generator_binding"] != development_manifest["generator_binding"]
        or evaluation_cache["generation_config"] != development_manifest["generation_config"]
        or evaluation_cache["seed_record"] != development_manifest["seed_record"]
        or evaluation_report["row_accounting"]["reported_row_count"]
        != development["row_count"]
        or evaluation_report["row_accounting"]["no_rows_dropped"] is not True
        or evaluation_report["checkpoint_binding"]["checkpoint_id"]
        != checkpoint_record["checkpoint_id"]
        or evaluation_report["checkpoint_binding"]["checkpoint_binding_id"]
        != checkpoint_record["checkpoint_binding_id"]
        or evaluation_report["checkpoint_binding"]["file_sha256"]
        != checkpoint_record["file_sha256"]
        or sorted(evaluation_report["identities"]["development_evaluation_animal_ids"])
        != bundle["configuration"]["development_evaluation_animal_ids"]
    ):
        raise ValueError("finite-v4 package all-row cross-binding differs")
    return True


__all__ = ["verify_arbitrary_plane_finite_package_v4"]
