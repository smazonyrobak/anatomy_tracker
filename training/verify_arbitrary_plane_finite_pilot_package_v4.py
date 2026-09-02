"""Independent verifier for the exact finite-v4 pilot postrun package."""

from __future__ import annotations

import json

import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.arbitrary_plane_run_export_v3 as export_v3
import training.run_arbitrary_plane_finite_pilot_postrun_v4 as pilot_v4
from training.verify_arbitrary_plane_finite_package_v4 import (
    verify_arbitrary_plane_finite_package_v4,
)


def _inside(root, relative_path, *, directory=False):
    path = (root / relative_path).resolve(strict=True)
    if root not in path.parents or (path.is_dir() if directory else path.is_file()) is not True:
        raise ValueError("finite-v4 pilot package component path is invalid")
    return path


def verify_arbitrary_plane_finite_pilot_package_v4(output_directory):
    """Verify the generic package and the frozen exact-pilot constraints."""
    root = evaluation_v4._i_path(output_directory, must_exist=True)
    bundle_path = _inside(root, pilot_v4.PILOT_BUNDLE_RELATIVE_PATH)
    bundle = json.loads(bundle_path.read_text("ascii"))
    payload = {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    plan = pilot_v4.finite_pilot_postrun_plan_v4()
    if (
        bundle.get("schema_version")
        != pilot_v4.FINITE_PILOT_POSTRUN_BUNDLE_V4_SCHEMA
        or bundle.get("scientific_scope")
        != pilot_v4.FINITE_PILOT_POSTRUN_SCIENTIFIC_SCOPE
        or bundle.get("output_directory") != str(root)
        or bundle.get("source_sha256") != pilot_v4._source_receipts()
        or bundle.get("plan") != plan
        or bundle.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or bundle.get("receipt_sha256") != evaluation_v4._sha(payload)
        or bundle.get("calibration") != {
            "status": export_v3.UNCALIBRATED_STATUS,
            "calibration_receipt": None,
        }
        or bundle.get("public_benchmark_accessed") is not False
        or bundle.get("external_validation_accessed") is not False
        or bundle.get("final_test_accessed") is not False
        or any(
            bundle.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("finite-v4 pilot wrapper failed authentication")
    package_record = bundle["package"]
    if package_record["relative_directory"] != pilot_v4.PILOT_PACKAGE_RELATIVE_DIRECTORY:
        raise ValueError("finite-v4 pilot generic package path differs")
    package_root = _inside(root, package_record["relative_directory"], directory=True)
    generic_bundle_path = _inside(
        package_root, package_record["generic_bundle_relative_path"]
    )
    if (
        package_record["generic_bundle_relative_path"]
        != "finite_postrun_bundle_receipt.json"
        or evaluation_v4._file_sha256(generic_bundle_path)
        != package_record["generic_bundle_file_sha256"]
    ):
        raise ValueError("finite-v4 pilot generic bundle file differs")
    verify_arbitrary_plane_finite_package_v4(package_root)
    generic = json.loads(generic_bundle_path.read_text("ascii"))
    evaluation_record = generic["artifacts"]["development_evaluation"]
    report_path = _inside(
        package_root / evaluation_record["relative_directory"],
        "finite_development_evaluation_report.json",
    )
    report = json.loads(report_path.read_text("ascii"))
    expected_evaluation = plan["evaluation"]
    generic_configuration = generic["configuration"]
    expected_configuration = {
        "all_development_cache_rows_evaluated": True,
        "development_evaluation_animal_ids": generic_configuration[
            "development_evaluation_animal_ids"
        ],
        "top_k": expected_evaluation["top_k"],
        "refinement_steps": expected_evaluation["refinement_steps"],
        "pose_only_steps": expected_evaluation["pose_only_steps"],
        "retrieval_shape_h_w": expected_evaluation["retrieval_shape_h_w"],
        "catalogue_chunk_size": expected_evaluation["catalogue_chunk_size"],
        "gauss_hermite_order": expected_evaluation["gauss_hermite_order"],
        "evaluation_seed": expected_evaluation["evaluation_seed"],
        "minimum_jacobian": expected_evaluation["minimum_jacobian"],
        "maximum_cycle_error_px": expected_evaluation["maximum_cycle_error_px"],
        "device": expected_evaluation["device"],
        "per_row_schedule_source": "finite_psf_contract",
        "global_schedule_fallback": None,
    }
    animals = generic_configuration["development_evaluation_animal_ids"]
    rows = report["row_reports"]
    annotation = generic["artifacts"]["regional_annotation"]
    expected_contract = {
        "render_mode": plan["finite_psf"]["render_mode"],
        "axial_sample_count": plan["finite_psf"]["axial_sample_count"],
    }
    schedules_are_exact = all(
        row["finite_psf_schedule_binding"]["source"]
        == plan["finite_psf"]["schedule_source"]
        and all(
            row["finite_psf_schedule_binding"][name] == value
            for name, value in expected_contract.items()
        )
        for row in rows
    )
    if (
        generic["receipt_sha256"]
        != package_record["generic_bundle_receipt_sha256"]
        or generic["output_directory"] != str(package_root)
        or generic_configuration != expected_configuration
        or len(animals) != plan["expected_development_animal_count"]
        or len(animals) != len(set(animals))
        or generic["run_binding"]["directory"] != plan["run_directory"]
        or generic["development_cache_binding"]["directory"]
        != plan["development_cache_directory"]
        or generic["development_cache_binding"]["row_count"]
        != plan["expected_development_row_count"]
        or generic["development_cache_binding"]["selected_row_indices"]
        != list(range(plan["expected_development_row_count"]))
        or generic["development_cache_binding"]["finite_psf_run_contract"][
            "render_mode"
        ]
        != plan["finite_psf"]["render_mode"]
        or generic["development_cache_binding"]["finite_psf_run_contract"][
            "axial_sample_count"
        ]
        != plan["finite_psf"]["axial_sample_count"]
        or generic["calibration"] != bundle["calibration"]
        or annotation is None
        or annotation.get("contains_atlas_intensity") is not False
        or report["experiment_scope"] != "finite-thickness-production-s9"
        or report["catalogue_binding"]["cell_count"]
        != plan["expected_catalogue"]["cell_count"]
        or report["row_accounting"]["selected_row_count"]
        != plan["expected_development_row_count"]
        or report["row_accounting"]["reported_row_count"]
        != plan["expected_development_row_count"]
        or report["row_accounting"]["no_rows_dropped"] is not True
        or len(rows) != plan["expected_development_row_count"]
        or not schedules_are_exact
        or {row["animal_id"] for row in rows} != set(animals)
        or report["animal_macro_metrics"]["statistical_unit"] != "animal"
        or report["animal_macro_metrics"]["animal_count"]
        != plan["expected_development_animal_count"]
        or set(report["animal_macro_metrics"]["per_animal"]) != set(animals)
        or report["uncertainty_scope"] != evaluation_v4.UNCALIBRATED_SCOPE
        or report["calibration_fitted"] is not False
        or report["regional_annotation_artifact"] is None
        or report["regional_annotation_artifact"]["file_sha256"]
        != annotation["file_sha256"]
    ):
        raise ValueError("finite-v4 package does not implement the exact pilot plan")
    if bundle["runtime_binding"] != pilot_v4._runtime_binding(generic, report):
        raise ValueError("finite-v4 pilot runtime binding differs")
    expected_children = {
        bundle_path.resolve(),
        package_root.resolve(),
    }
    if {path.resolve() for path in root.iterdir()} != expected_children:
        raise ValueError("finite-v4 pilot wrapper artifact set differs")
    return True


__all__ = ["verify_arbitrary_plane_finite_pilot_package_v4"]
