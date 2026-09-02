"""Run the exact immutable finite-v4 pilot post-training evaluation."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v4 as row_cache_v4
import training.arbitrary_plane_run_export_v3 as export_v3
import training.run_arbitrary_plane_authentic_finite_development_v4 as development_v4
import training.run_arbitrary_plane_finite_postrun_v4 as postrun_v4


FINITE_PILOT_POSTRUN_PLAN_V4_SCHEMA = (
    "anatomy-tracker.finite-pilot-postrun-plan/v4"
)
FINITE_PILOT_POSTRUN_BUNDLE_V4_SCHEMA = (
    "anatomy-tracker.finite-pilot-postrun-bundle/v4"
)
FINITE_PILOT_POSTRUN_SCIENTIFIC_SCOPE = (
    "exact finite-v4 pilot internal synthetic animal-disjoint development only; "
    "uncalibrated and not a public benchmark, qualification, calibration, "
    "external validation, or final test"
)
PILOT_PACKAGE_RELATIVE_DIRECTORY = "finite_package_v4"
PILOT_BUNDLE_RELATIVE_PATH = "finite_pilot_postrun_bundle_receipt.json"


def _source_receipts():
    paths = {
        "pilot_postrun": Path(__file__).resolve(),
        "pilot_verifier": Path(__file__).with_name(
            "verify_arbitrary_plane_finite_pilot_package_v4.py"
        ).resolve(),
        "finite_postrun": Path(postrun_v4.__file__).resolve(),
        "finite_development_profile": Path(development_v4.__file__).resolve(),
    }
    return {
        name: evaluation_v4._file_sha256(path)
        for name, path in sorted(paths.items())
    }


def finite_pilot_postrun_plan_v4():
    """Return the frozen exact pilot evaluation plan."""
    config = development_v4.finite_development_configuration_v4("pilot")
    development_v4.verify_finite_development_configuration_v4(config)
    catalogue_config = config["catalogue_config"]
    training_config = config["training_config"]
    training = config["partitions"]["training"]
    development = config["partitions"]["internal_development"]
    training_row_count = int(
        training["pose_row_count"] + training["joint_row_count"]
    )
    development_row_count = int(
        development["pose_row_count"] + development["joint_row_count"]
    )
    catalogue_cell_count = int(
        catalogue_config["normal_count"]
        * catalogue_config["offset_count"]
        * catalogue_config["roll_count"]
    )
    expected_animal_count = int(
        development["pose_row_count"] // config["sections_per_animal"]
        + development["joint_row_count"] // config["sections_per_animal"]
    )
    if (
        training_row_count != 5120
        or development_row_count != 640
        or catalogue_cell_count != 98_304
        or expected_animal_count != 40
        or any(
            partition[name] % config["sections_per_animal"]
            for partition in (training, development)
            for name in ("pose_row_count", "joint_row_count")
        )
        or catalogue_config != {
            "normal_count": 384,
            "offset_count": 16,
            "roll_count": 16,
            "raster_shape_h_w": [160, 160],
            "raster_physical_span_y_x_um": [12000.0, 12000.0],
        }
        or config["render_mode"] != "finite_boxcar"
        or config["axial_sample_count"] != 9
        or training_config["catalogue_chunk_size"] != 512
        or training_config["retrieval_shape_h_w"] != [48, 48]
        or training_config["top_k"] != 4
        or training_config["refinement_steps"] != 3
        or training_config["joint_pose_only_steps"] != 2
    ):
        raise ValueError("finite-v4 pilot postrun source configuration changed")
    payload = {
        "schema_version": FINITE_PILOT_POSTRUN_PLAN_V4_SCHEMA,
        "profile": "pilot",
        "profile_configuration_receipt_sha256": config["receipt_sha256"],
        "run_directory": config["training_run"],
        "development_cache_directory": config["internal_development_cache"],
        "configuration_snapshot": config["configuration_snapshot"],
        "partition_audit": config["partition_audit"],
        "output_directory": str(
            Path(config["output_root"]) / "finite_pilot_postrun_v4"
        ),
        "expected_training_row_count": training_row_count,
        "expected_development_row_count": development_row_count,
        "expected_development_animal_count": expected_animal_count,
        "expected_catalogue": {
            "normal_count": 384,
            "offset_count": 16,
            "roll_count": 16,
            "cell_count": catalogue_cell_count,
            "raster_shape_h_w": [160, 160],
        },
        "finite_psf": {
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
            "schedule_source": "authenticated-per-row-finite_psf_contract",
            "global_schedule_fallback": None,
            "finite_psf_capability_sha256": config[
                "finite_psf_capability_sha256"
            ],
        },
        "evaluation": {
            "all_development_cache_rows_evaluated": True,
            "top_k": 4,
            "refinement_steps": 3,
            "pose_only_steps": 2,
            "retrieval_shape_h_w": [48, 48],
            "catalogue_chunk_size": 512,
            "gauss_hermite_order": 5,
            "evaluation_seed": int(training_config["seed"]),
            "minimum_jacobian": float(
                evaluation_v4.metrics_v3.DEFAULT_MINIMUM_JACOBIAN
            ),
            "maximum_cycle_error_px": float(
                evaluation_v4.metrics_v3.DEFAULT_MAXIMUM_CYCLE_ERROR_PX
            ),
            "device": "cuda",
            "annotation_bound_regional_dice_required": True,
            "raw_prediction_per_row_required": True,
            "animal_macro_statistical_unit": "animal",
            "uncertainty_status": export_v3.UNCALIBRATED_STATUS,
        },
        "public_benchmark_accessed": False,
        "external_validation_accessed": False,
        "final_test_accessed": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": evaluation_v4._sha(payload)}


def _atlas_semantics(manifest):
    binding = manifest["atlas"]["binding"]
    return {
        "schema_version": inference_v3.ATLAS_SEMANTICS_V3_SCHEMA,
        "atlas_name": "Allen Common Coordinate Framework",
        "atlas_version": "CCFv3 2017 25um",
        "processed_channel_names": ["normalized-template-intensity", "annotation-support"],
        "processed_channel_recipes": [
            "float32 clip((template-9)/(273-9),0,1), exact zero outside annotation support",
            "float32(annotation != 0)",
        ],
        "source_assets": [
            {
                "asset_role": asset["role"],
                "uri": asset["path"],
                "sha256": asset["sha256"],
            }
            for asset in binding["source_assets"]
        ],
        "source_format": "NRRD decoded by pynrrd",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["posterior", "inferior", "right"],
        "voxel_center_convention": "integer array coordinates denote voxel centres",
        "normalization_parameters": copy.deepcopy(binding["preprocessing"]),
    }


def _load_pinned_annotation():
    import nrrd

    if (
        importlib.metadata.version("pynrrd") != development_v4.PYNRRD_VERSION
        or evaluation_v4._file_sha256(development_v4.TEMPLATE_PATH)
        != development_v4.TEMPLATE_SHA256
        or evaluation_v4._file_sha256(development_v4.ANNOTATION_PATH)
        != development_v4.ANNOTATION_SHA256
    ):
        raise RuntimeError("pinned Allen decoder or source hashes differ")
    annotation = np.ascontiguousarray(
        nrrd.read(str(development_v4.ANNOTATION_PATH), index_order="F")[0]
    )
    if annotation.ndim != 3 or annotation.dtype.kind not in "iu":
        raise RuntimeError("pinned Allen annotation has an invalid decoded representation")
    return annotation


def _load_and_validate_pilot_inputs(plan):
    config = development_v4.finite_development_configuration_v4("pilot")
    snapshot_path = evaluation_v4._i_path(
        plan["configuration_snapshot"], must_exist=True
    )
    snapshot = json.loads(snapshot_path.read_text("ascii"))
    if snapshot != config:
        raise ValueError("finite-v4 pilot configuration snapshot differs")
    run_root = evaluation_v4._i_path(plan["run_directory"], must_exist=True)
    manifest, training_manifest = runner_v4._load_manifest(run_root)
    state, _, _ = runner_v4._load_run_state(run_root, manifest)
    development_manifest = row_cache_v4.load_training_row_cache_manifest_v4(
        plan["development_cache_directory"]
    )
    expected_contract = row_cache_v4.make_finite_psf_cache_run_contract_v4(
        "finite_boxcar"
    )
    training_animals = {
        row["lineage"]["animal_id"] for row in training_manifest["rows"]
    }
    development_animals = {
        row["lineage"]["animal_id"] for row in development_manifest["rows"]
    }
    if (
        int(state["applied_step_count"])
        != int(manifest["runner_config"]["target_applied_steps"])
        or manifest["runner_config"] != config["runner_config"]
        or manifest["training_config"] != config["training_config"]
        or manifest["model_kwargs"] != config["model_kwargs"]
        or manifest["cache"]["directory"] != config["training_cache"]
        or int(training_manifest["row_count"])
        != plan["expected_training_row_count"]
        or training_manifest["finite_psf_run_contract"] != expected_contract
        or development_manifest["status"] != row_cache_v4.FROZEN_CACHE_STATUS
        or int(development_manifest["row_count"])
        != plan["expected_development_row_count"]
        or development_manifest["finite_psf_run_contract"] != expected_contract
        or development_manifest["finite_psf_capability"]
        != manifest["finite_psf_capability"]
        or int(manifest["catalogue"]["catalogue_cell_count"])
        != plan["expected_catalogue"]["cell_count"]
        or len(development_animals) != plan["expected_development_animal_count"]
        or training_animals & development_animals
    ):
        raise ValueError("completed run does not match the exact finite-v4 pilot")
    annotation = _load_pinned_annotation()
    return {
        "development_animals": sorted(development_animals),
        "annotation": annotation,
        "atlas_semantics": _atlas_semantics(manifest),
    }


def _runtime_binding(generic_bundle, evaluation_report):
    annotation = generic_bundle["artifacts"]["regional_annotation"]
    return {
        "run_id": generic_bundle["run_binding"]["run_id"],
        "run_manifest_receipt_sha256": generic_bundle["run_binding"][
            "run_manifest_receipt_sha256"
        ],
        "run_state_receipt_sha256": generic_bundle["run_binding"][
            "run_state_receipt_sha256"
        ],
        "development_manifest_receipt_sha256": generic_bundle[
            "development_cache_binding"
        ]["manifest_receipt_sha256"],
        "catalogue_id": evaluation_report["catalogue_binding"]["catalogue_id"],
        "catalogue_receipt_sha256": evaluation_report["catalogue_binding"][
            "receipt_sha256"
        ],
        "development_evaluation_animal_ids": generic_bundle["configuration"][
            "development_evaluation_animal_ids"
        ],
        "raw_prediction_count": len(evaluation_report["row_reports"]),
        "evaluation_report_receipt_sha256": evaluation_report["receipt_sha256"],
        "regional_annotation_file_sha256": annotation["file_sha256"],
    }


def run_arbitrary_plane_finite_pilot_postrun_v4():
    """Evaluate and package the exact completed pilot without any path overrides."""
    plan = finite_pilot_postrun_plan_v4()
    inputs = _load_and_validate_pilot_inputs(plan)
    output_root = evaluation_v4._i_path(plan["output_directory"])
    if os.path.lexists(output_root):
        raise FileExistsError("finite-v4 pilot postrun output directory must be new")
    package_root = output_root / PILOT_PACKAGE_RELATIVE_DIRECTORY
    output_root.mkdir(parents=True)
    evaluation = plan["evaluation"]
    generic_bundle = postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        plan["run_directory"],
        plan["development_cache_directory"],
        package_root,
        atlas_semantics=inputs["atlas_semantics"],
        development_evaluation_animal_ids=inputs["development_animals"],
        annotation_volume_ap_dv_ml=inputs["annotation"],
        top_k=evaluation["top_k"],
        refinement_steps=evaluation["refinement_steps"],
        pose_only_steps=evaluation["pose_only_steps"],
        retrieval_shape_h_w=tuple(evaluation["retrieval_shape_h_w"]),
        catalogue_chunk_size=evaluation["catalogue_chunk_size"],
        gauss_hermite_order=evaluation["gauss_hermite_order"],
        evaluation_seed=evaluation["evaluation_seed"],
        minimum_jacobian=evaluation["minimum_jacobian"],
        maximum_cycle_error_px=evaluation["maximum_cycle_error_px"],
        device=evaluation["device"],
    )
    generic_bundle_path = package_root / "finite_postrun_bundle_receipt.json"
    report_path = (
        package_root
        / "internal_development_evaluation"
        / "finite_development_evaluation_report.json"
    )
    evaluation_report = json.loads(report_path.read_text("ascii"))
    payload = {
        "schema_version": FINITE_PILOT_POSTRUN_BUNDLE_V4_SCHEMA,
        "scientific_scope": FINITE_PILOT_POSTRUN_SCIENTIFIC_SCOPE,
        "output_directory": str(output_root),
        "source_sha256": _source_receipts(),
        "plan": plan,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "package": {
            "relative_directory": PILOT_PACKAGE_RELATIVE_DIRECTORY,
            "generic_bundle_relative_path": "finite_postrun_bundle_receipt.json",
            "generic_bundle_file_sha256": evaluation_v4._file_sha256(
                generic_bundle_path
            ),
            "generic_bundle_receipt_sha256": generic_bundle["receipt_sha256"],
        },
        "runtime_binding": _runtime_binding(generic_bundle, evaluation_report),
        "calibration": {
            "status": export_v3.UNCALIBRATED_STATUS,
            "calibration_receipt": None,
        },
        "public_benchmark_accessed": False,
        "external_validation_accessed": False,
        "final_test_accessed": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    bundle = {**payload, "receipt_sha256": evaluation_v4._sha(payload)}
    evaluation_v4._atomic_json_new(
        output_root / PILOT_BUNDLE_RELATIVE_PATH, bundle
    )
    from training.verify_arbitrary_plane_finite_pilot_package_v4 import (
        verify_arbitrary_plane_finite_pilot_package_v4,
    )

    verify_arbitrary_plane_finite_pilot_package_v4(output_root)
    return bundle


def main():
    run_arbitrary_plane_finite_pilot_postrun_v4()


if __name__ == "__main__":
    main()


__all__ = [
    "FINITE_PILOT_POSTRUN_BUNDLE_V4_SCHEMA",
    "FINITE_PILOT_POSTRUN_PLAN_V4_SCHEMA",
    "FINITE_PILOT_POSTRUN_SCIENTIFIC_SCOPE",
    "finite_pilot_postrun_plan_v4",
    "run_arbitrary_plane_finite_pilot_postrun_v4",
]
