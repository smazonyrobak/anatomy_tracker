"""Export an authenticated completed finite-v4 run to capability-bound inference."""

from __future__ import annotations

from pathlib import Path

import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_run_export_v3 as export_v3


FINITE_TRAINING_RUN_INFERENCE_EXPORT_V4_SCHEMA = (
    "anatomy-tracker.finite-training-run-inference-export/v4"
)
FINITE_TRAINING_DATASET_PROVENANCE_V4_SCHEMA = (
    "anatomy-tracker.finite-training-run-dataset-provenance/v4"
)


def _plain(value):
    return export_v3._plain(value)


def _sha(value):
    return export_v3._sha(value)


def _file_sha256(path):
    return export_v3._file_sha256(path)


def _dataset_provenance(context, run_export_receipt, checkpoint_path):
    manifest = context["manifest"]
    state = context["run_state"]
    cache = context["cache_manifest"]
    payload = {
        "schema_version": FINITE_TRAINING_DATASET_PROVENANCE_V4_SCHEMA,
        "data_role": cache["data_role"],
        "frozen_cache": {
            "directory": manifest["cache"]["directory"],
            "schema_version": cache["schema_version"],
            "manifest_receipt_sha256": cache["receipt_sha256"],
            "status": cache["status"],
            "row_count": int(cache["row_count"]),
            "freeze_audit": _plain(cache["freeze_audit"]),
            "generation_config": _plain(cache["generation_config"]),
            "seed_record": _plain(cache["seed_record"]),
            "generator_binding": _plain(cache["generator_binding"]),
            "finite_psf_run_contract": _plain(cache["finite_psf_run_contract"]),
        },
        "run_binding": {
            "run_id": manifest["run_id"],
            "run_manifest_receipt_sha256": manifest["receipt_sha256"],
            "run_state_receipt_sha256": state["receipt_sha256"],
            "run_status": "completed",
            "target_applied_steps": int(manifest["runner_config"]["target_applied_steps"]),
            "applied_step_count": int(state["applied_step_count"]),
            "latest_staged_checkpoint": {
                **_plain(state["latest_checkpoint"]),
                "absolute_path": str(checkpoint_path),
            },
            "training_run_export_receipt_sha256": run_export_receipt["receipt_sha256"],
        },
        "catalogue_binding": {
            "catalogue_id": context["catalogue"]["catalogue_id"],
            "catalogue_receipt_sha256": context["catalogue"]["receipt_sha256"],
            "cell_count": int(context["catalogue"]["counts"]["cell_count"]),
        },
        "atlas_binding_receipt_sha256": manifest["atlas"]["binding"]["receipt_sha256"],
        "runner_source_sha256": _plain(manifest["runner_source_sha256"]),
        "finite_psf_capability": _plain(manifest["finite_psf_capability"]),
        "finite_psf_training_schedule_source": _plain(
            manifest["finite_psf_training_schedule_source"]
        ),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_finite_training_run_inference_export_report_v4(report):
    payload = (
        {key: value for key, value in report.items() if key != "receipt_sha256"}
        if isinstance(report, dict) else {}
    )
    provenance = payload.get("dataset_provenance", {})
    provenance_payload = {
        key: value for key, value in provenance.items() if key != "receipt_sha256"
    }
    checkpoint = payload.get("checkpoint", {})
    safe = payload.get("safe_load_verification", {})
    if (
        payload.get("schema_version") != FINITE_TRAINING_RUN_INFERENCE_EXPORT_V4_SCHEMA
        or payload.get("export_status") != "completed"
        or report.get("receipt_sha256") != _sha(payload)
        or provenance.get("schema_version") != FINITE_TRAINING_DATASET_PROVENANCE_V4_SCHEMA
        or provenance.get("receipt_sha256") != _sha(provenance_payload)
        or payload.get("calibration") != {
            "status": export_v3.UNCALIBRATED_STATUS, "calibration_receipt": None
        }
        or any(payload.get(name) != [] for name in (
            "prior_model_weight_dependencies", "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        ))
        or payload.get("public_benchmark_accessed") is not False
        or payload.get("final_test_accessed") is not False
        or payload.get("external_validation_accessed") is not False
        or any(checkpoint.get(name) != safe.get(name) for name in (
            "file_sha256", "checkpoint_id", "checkpoint_binding_id",
            "model_state_sha256", "inference_contract_receipt_sha256",
        ))
        or payload.get("finite_psf_schedule_scope") != {
            "checkpoint": "capability-only",
            "runtime": "caller-explicit authenticated per-row schedule",
            "global_schedule_fallback": None,
        }
    ):
        raise ValueError("finite-v4 inference export report failed authentication")
    inference_v3.verify_inference_contract_v3(payload["inference_contract"])
    if payload["inference_contract"].get("finite_psf_capability") != (
        provenance.get("finite_psf_capability")
    ):
        raise ValueError("finite-v4 export capability differs from provenance")
    return True


def export_finite_training_run_to_inference_checkpoint_v4(
    run_directory,
    checkpoint_path,
    *,
    atlas_semantics,
    annotation_volume_ap_dv_ml=None,
    safe_load_device="cpu",
):
    """Freeze the exact completed finite-v4 state as a capability-bound checkpoint."""
    context = runner_v4.load_finite_training_run_v4(run_directory)
    manifest = context["manifest"]
    state = context["run_state"]
    if int(state["applied_step_count"]) != int(
        manifest["runner_config"]["target_applied_steps"]
    ):
        raise ValueError("finite-v4 inference export requires the completed target")
    target = export_v3._i_path(checkpoint_path)
    if target.exists():
        raise FileExistsError("finite-v4 inference checkpoint target already exists")
    export_v3._verify_semantics_match_run(
        atlas_semantics, manifest["atlas"]["binding"]
    )
    run_receipt = runner_v4.make_finite_training_run_export_receipt_v4(
        run_directory
    )
    training_receipt = run_receipt["staged_training_export_receipt"]
    staged_path = (
        context["run_root"] / state["latest_checkpoint"]["relative_path"]
    ).resolve(strict=True)
    geometry = context["catalogue"]["support_geometry"]
    inference_contract = inference_v3.make_inference_contract_v3(
        context["atlas_volume"],
        geometry["origin_ap_dv_ml_um"],
        geometry["voxel_size_ap_dv_ml_um"],
        None,
        None,
        atlas_semantics=atlas_semantics,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        finite_psf_capability=manifest["finite_psf_capability"],
    )
    dataset_provenance = _dataset_provenance(
        context, run_receipt, staged_path
    )
    provenance = {
        "initialization": "fresh_random",
        "architecture_source": inference_v3.ARCHITECTURE_MODULE,
        "dataset_provenance": [dataset_provenance],
        "animal_specimen_experiment_id_contract": (
            "exact authenticated animal_id, specimen_id, and experiment_id values "
            "from the frozen finite-v4 cache and staged receipt"
        ),
        "prior_trained_model_dependencies": [],
        "prior_model_feature_dependencies": [],
        "pseudolabel_dependencies": [],
    }
    checkpoint = inference_v3.make_arbitrary_plane_joint_checkpoint_v3(
        context["training_state"]["model"],
        manifest["model_kwargs"],
        context["catalogue"],
        provenance,
        training_receipt,
        inference_contract=inference_contract,
        calibration_receipt=None,
    )
    export_v3._atomic_torch_save_new_i(target, checkpoint)
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        target, context["catalogue"], device=safe_load_device
    )
    report_payload = {
        "schema_version": FINITE_TRAINING_RUN_INFERENCE_EXPORT_V4_SCHEMA,
        "export_status": "completed",
        "exporter_source_sha256": _file_sha256(Path(__file__).resolve()),
        "dataset_provenance": dataset_provenance,
        "finite_training_run_export_receipt": _plain(run_receipt),
        "training_receipt": _plain(training_receipt),
        "inference_contract": _plain(inference_contract),
        "finite_psf_schedule_scope": {
            "checkpoint": "capability-only",
            "runtime": "caller-explicit authenticated per-row schedule",
            "global_schedule_fallback": None,
        },
        "checkpoint": {
            "path": str(target),
            "file_sha256": loaded["checkpoint_file_sha256"],
            "checkpoint_id": loaded["checkpoint_id"],
            "checkpoint_binding_id": loaded["checkpoint_binding_id"],
            "model_state_sha256": loaded["model_state_sha256"],
            "inference_contract_receipt_sha256": inference_contract["receipt_sha256"],
        },
        "safe_load_verification": {
            "device": loaded["device"],
            "catalogue_id": loaded["catalogue_id"],
            "file_sha256": loaded["checkpoint_file_sha256"],
            "checkpoint_id": loaded["checkpoint_id"],
            "checkpoint_binding_id": loaded["checkpoint_binding_id"],
            "model_state_sha256": loaded["model_state_sha256"],
            "inference_contract_receipt_sha256": loaded["inference_contract"]["receipt_sha256"],
            "verified": True,
        },
        "calibration": {
            "status": export_v3.UNCALIBRATED_STATUS, "calibration_receipt": None
        },
        "public_benchmark_accessed": False,
        "final_test_accessed": False,
        "external_validation_accessed": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    report = {**report_payload, "receipt_sha256": _sha(report_payload)}
    verify_finite_training_run_inference_export_report_v4(report)
    return report


__all__ = [
    "FINITE_TRAINING_RUN_INFERENCE_EXPORT_V4_SCHEMA",
    "export_finite_training_run_to_inference_checkpoint_v4",
    "verify_finite_training_run_inference_export_report_v4",
]
