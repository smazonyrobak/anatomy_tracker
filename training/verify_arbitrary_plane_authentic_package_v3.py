"""Independent verifier for an authentic arbitrary-plane V3 post-training bundle."""

from __future__ import annotations

import json
from pathlib import Path

import training.arbitrary_plane_development_evaluation_v3 as evaluation_v3
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_run_export_v3 as export_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3
import training.run_arbitrary_plane_authentic_postrun_v3 as postrun_v3


def _inside(root, relative_path, *, directory=False):
    path = (root / relative_path).resolve(strict=True)
    if root not in path.parents or (
        not path.is_dir() if directory else not path.is_file()
    ):
        raise ValueError("post-training artifact path is invalid")
    return path


def verify_arbitrary_plane_authentic_package_v3(output_directory):
    """Authenticate the exact checkpoint/cache/all-row development bundle."""
    output_root = postrun_v3._i_path(output_directory, must_exist=True)
    bundle_path = _inside(output_root, "post_training_bundle_receipt.json")
    bundle = json.loads(bundle_path.read_text("ascii"))
    payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    if (
        bundle.get("schema_version") != postrun_v3.AUTHENTIC_POSTRUN_BUNDLE_V3_SCHEMA
        or bundle.get("scientific_scope") != postrun_v3.POSTRUN_SCIENTIFIC_SCOPE
        or bundle.get("output_directory") != str(output_root)
        or bundle.get("source_sha256") != postrun_v3._source_receipts()
        or bundle.get("configuration_receipt_sha256")
        != postrun_v3._sha(bundle.get("configuration", {}))
        or bundle.get("receipt_sha256") != postrun_v3._sha(payload)
        or bundle.get("calibration")
        != {"status": export_v3.UNCALIBRATED_STATUS, "calibration_receipt": None}
        or bundle.get("public_benchmark_accessed") is not False
        or bundle.get("final_test_accessed") is not False
        or bundle.get("external_validation_accessed") is not False
        or bundle.get("prior_model_weight_dependencies") != []
        or bundle.get("prior_feature_dependencies") != []
        or bundle.get("prior_pseudolabel_dependencies") != []
    ):
        raise ValueError("post-training bundle receipt failed authentication")

    run = bundle["run_binding"]
    context = runner_v3.load_training_run_v3(
        postrun_v3._i_path(run["directory"], must_exist=True)
    )
    expected_run = {
        "directory": str(context["run_root"]),
        "run_id": context["manifest"]["run_id"],
        "run_manifest_receipt_sha256": context["manifest"]["receipt_sha256"],
        "run_state_receipt_sha256": context["run_state"]["receipt_sha256"],
        "target_applied_steps": int(
            context["manifest"]["runner_config"]["target_applied_steps"]
        ),
        "applied_step_count": int(context["run_state"]["applied_step_count"]),
        "status": "completed",
    }
    if run != expected_run or run["applied_step_count"] != run["target_applied_steps"]:
        raise ValueError("post-training completed-run binding differs")
    del context["training_state"], context["training_reports"]

    artifacts = bundle["artifacts"]
    checkpoint_record = artifacts["checkpoint"]
    checkpoint_path = _inside(output_root, checkpoint_record["relative_path"])
    export_record = artifacts["export_report"]
    export_path = _inside(output_root, export_record["relative_path"])
    if postrun_v3._file_sha256(export_path) != export_record["file_sha256"]:
        raise ValueError("post-training export report file hash differs")
    export_report = json.loads(export_path.read_text("ascii"))
    export_v3.verify_training_run_inference_export_report_v3(export_report)
    if (
        export_report["export_status"] != "completed"
        or export_report["receipt_sha256"] != export_record["receipt_sha256"]
        or export_report["checkpoint"]["path"] != str(checkpoint_path)
        or any(
            export_report["checkpoint"][name] != checkpoint_record[name]
            for name in (
                "file_sha256",
                "checkpoint_id",
                "checkpoint_binding_id",
                "model_state_sha256",
            )
        )
        or export_report["dataset_provenance"]["run_binding"]["run_state_receipt_sha256"]
        != run["run_state_receipt_sha256"]
    ):
        raise ValueError("post-training exact export binding differs")
    if postrun_v3._file_sha256(checkpoint_path) != checkpoint_record["file_sha256"]:
        raise ValueError("post-training checkpoint file hash differs")
    catalogue = context["catalogue"]
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint_path, catalogue, device="cpu"
    )
    if any(
        loaded[name] != checkpoint_record[name]
        for name in (
            "checkpoint_id",
            "checkpoint_binding_id",
            "model_state_sha256",
        )
    ):
        raise ValueError("post-training checkpoint identity differs")

    cache_record = artifacts["catalogue_feature_cache"]
    cache_path = _inside(output_root, cache_record["relative_path"])
    cache = inference_v3.load_arbitrary_plane_catalogue_feature_cache_v3(
        cache_path, loaded, catalogue
    )
    if (
        cache["cache_file_sha256"] != cache_record["file_sha256"]
        or cache["cache_receipt"]["cache_id"] != cache_record["cache_id"]
        or cache["cache_receipt"] != cache_record["cache_receipt"]
    ):
        raise ValueError("post-training same-checkpoint catalogue cache differs")

    evaluation_record = artifacts["development_evaluation"]
    evaluation_root = _inside(
        output_root, evaluation_record["relative_directory"], directory=True
    )
    evaluation_bundle_path = _inside(
        evaluation_root, evaluation_record["bundle_relative_path"]
    )
    if (
        postrun_v3._file_sha256(evaluation_bundle_path)
        != evaluation_record["bundle_file_sha256"]
    ):
        raise ValueError("post-training development bundle file hash differs")
    evaluation_bundle = json.loads(evaluation_bundle_path.read_text("ascii"))
    if evaluation_bundle["receipt_sha256"] != evaluation_record["bundle_receipt_sha256"]:
        raise ValueError("post-training development bundle receipt differs")
    evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(
        evaluation_root, catalogue=catalogue
    )
    evaluation_report = json.loads(
        (evaluation_root / "development_evaluation_report.json").read_text("ascii")
    )
    development = bundle["development_cache_binding"]
    development_manifest = row_cache_v3.load_training_row_cache_manifest_v3(
        postrun_v3._i_path(development["directory"], must_exist=True),
        expected_receipt_sha256=development["manifest_receipt_sha256"],
    )
    all_indices = list(range(int(development_manifest["row_count"])))
    expected_development = {
        "directory": str(Path(development["directory"]).resolve()),
        "manifest_receipt_sha256": development_manifest["receipt_sha256"],
        "row_count": int(development_manifest["row_count"]),
        "selected_row_indices": all_indices,
    }
    evaluation_cache = evaluation_report["cache_binding"]
    evaluation_feature_cache = evaluation_report["configuration"][
        "catalogue_feature_cache"
    ]
    evaluation_configuration = evaluation_report["configuration"]
    postrun_configuration = bundle["configuration"]
    matching_configuration = all(
        evaluation_configuration[name] == postrun_configuration[name]
        for name in (
            "top_k",
            "refinement_steps",
            "pose_only_steps",
            "retrieval_shape_h_w",
            "catalogue_chunk_size",
            "gauss_hermite_order",
            "evaluation_seed",
            "minimum_jacobian",
            "maximum_cycle_error_px",
        )
    )
    if (
        development != expected_development
        or bundle["configuration"]["all_development_cache_rows_evaluated"] is not True
        or not matching_configuration
        or cache["cache_receipt"]["render_and_storage_recipe"][
            "retrieval_shape_h_w"
        ]
        != postrun_configuration["retrieval_shape_h_w"]
        or cache["cache_receipt"]["render_and_storage_recipe"]["build_chunk_size"]
        != postrun_configuration["feature_cache_build_chunk_size"]
        or evaluation_cache["directory"] != development["directory"]
        or evaluation_cache["manifest_receipt_sha256"]
        != development["manifest_receipt_sha256"]
        or evaluation_cache["selected_row_indices"] != all_indices
        or evaluation_report["row_accounting"]["selected_row_count"]
        != development["row_count"]
        or evaluation_report["row_accounting"]["reported_row_count"]
        != development["row_count"]
        or evaluation_report["row_accounting"]["no_rows_dropped"] is not True
        or sorted(evaluation_report["identities"]["development_evaluation_animal_ids"])
        != bundle["configuration"]["development_evaluation_animal_ids"]
        or evaluation_report["checkpoint_binding"]["checkpoint_id"]
        != checkpoint_record["checkpoint_id"]
        or evaluation_report["checkpoint_binding"]["checkpoint_binding_id"]
        != checkpoint_record["checkpoint_binding_id"]
        or evaluation_report["checkpoint_binding"]["file_sha256"]
        != checkpoint_record["file_sha256"]
        or evaluation_feature_cache["path"] != str(cache_path)
        or evaluation_feature_cache["file_sha256"] != cache_record["file_sha256"]
        or evaluation_feature_cache["cache_id"] != cache_record["cache_id"]
        or evaluation_feature_cache["cache_receipt"] != cache_record["cache_receipt"]
    ):
        raise ValueError("post-training all-row evaluation cross-binding differs")
    return True


__all__ = ["verify_arbitrary_plane_authentic_package_v3"]
