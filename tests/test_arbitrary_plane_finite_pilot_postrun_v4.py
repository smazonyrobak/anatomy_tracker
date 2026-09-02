import copy
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.run_arbitrary_plane_finite_pilot_postrun_v4 as pilot_v4
import training.verify_arbitrary_plane_finite_pilot_package_v4 as pilot_verify_v4


def _test_root():
    return (
        Path("I:/AnatomyTracker/test_tmp/finite_pilot_postrun_v4")
        / uuid.uuid4().hex
    )


def _plan(output_root):
    plan = copy.deepcopy(pilot_v4.finite_pilot_postrun_plan_v4())
    plan["output_directory"] = str(output_root.resolve())
    payload = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    plan["receipt_sha256"] = evaluation_v4._sha(payload)
    return plan


def test_frozen_pilot_postrun_plan_is_exact_and_internal_only():
    plan = pilot_v4.finite_pilot_postrun_plan_v4()
    assert plan["expected_development_row_count"] == 640
    assert plan["expected_development_animal_count"] == 40
    assert plan["expected_catalogue"] == {
        "normal_count": 384,
        "offset_count": 16,
        "roll_count": 16,
        "cell_count": 98_304,
        "raster_shape_h_w": [160, 160],
    }
    assert plan["finite_psf"]["render_mode"] == "finite_boxcar"
    assert plan["finite_psf"]["axial_sample_count"] == 9
    assert plan["finite_psf"]["global_schedule_fallback"] is None
    assert plan["evaluation"] == {
        "all_development_cache_rows_evaluated": True,
        "top_k": 4,
        "refinement_steps": 3,
        "pose_only_steps": 2,
        "retrieval_shape_h_w": [48, 48],
        "catalogue_chunk_size": 512,
        "gauss_hermite_order": 5,
        "evaluation_seed": 20260902,
        "minimum_jacobian": 0.05,
        "maximum_cycle_error_px": 1.0,
        "device": "cuda",
        "annotation_bound_regional_dice_required": True,
        "raw_prediction_per_row_required": True,
        "animal_macro_statistical_unit": "animal",
        "uncertainty_status": "absent-uncalibrated",
    }
    assert plan["public_benchmark_accessed"] is False
    assert plan["external_validation_accessed"] is False
    assert plan["final_test_accessed"] is False
    assert plan["receipt_sha256"] == evaluation_v4._sha({
        key: value for key, value in plan.items() if key != "receipt_sha256"
    })


def test_runner_passes_only_exact_pilot_evaluation_arguments_and_never_overwrites(monkeypatch):
    root = _test_root()
    output = root / "postrun"
    plan = _plan(output)
    animals = [f"development-animal-{index:02d}" for index in range(40)]
    calls = []

    def fake_inputs(received):
        assert received == plan
        return {
            "development_animals": animals,
            "annotation": np.ones((2, 2, 2), dtype=np.uint16),
            "atlas_semantics": {"fixed": "semantics"},
        }

    def fake_postrun(run, cache, package_root, **kwargs):
        calls.append((run, cache, package_root, kwargs))
        package_root.mkdir(parents=True)
        generic = {
            "receipt_sha256": "1" * 64,
            "run_binding": {
                "run_id": "pilot-run",
                "run_manifest_receipt_sha256": "2" * 64,
                "run_state_receipt_sha256": "3" * 64,
            },
            "development_cache_binding": {
                "manifest_receipt_sha256": "4" * 64,
            },
            "configuration": {
                "development_evaluation_animal_ids": animals,
            },
            "artifacts": {
                "regional_annotation": {
                    "file_sha256": "5" * 64,
                },
            },
        }
        (package_root / "finite_postrun_bundle_receipt.json").write_bytes(
            evaluation_v4._canonical_json(generic)
        )
        evaluation_root = package_root / "internal_development_evaluation"
        evaluation_root.mkdir()
        report = {
            "receipt_sha256": "6" * 64,
            "catalogue_binding": {
                "catalogue_id": "pilot-catalogue",
                "receipt_sha256": "7" * 64,
            },
            "row_reports": [{} for _ in range(640)],
        }
        (evaluation_root / "finite_development_evaluation_report.json").write_bytes(
            evaluation_v4._canonical_json(report)
        )
        return generic

    monkeypatch.setattr(pilot_v4, "finite_pilot_postrun_plan_v4", lambda: plan)
    monkeypatch.setattr(pilot_v4, "_load_and_validate_pilot_inputs", fake_inputs)
    monkeypatch.setattr(
        pilot_v4.postrun_v4,
        "run_arbitrary_plane_finite_postrun_v4",
        fake_postrun,
    )
    monkeypatch.setattr(
        pilot_verify_v4,
        "verify_arbitrary_plane_finite_pilot_package_v4",
        lambda value: value == output.resolve(),
    )
    try:
        bundle = pilot_v4.run_arbitrary_plane_finite_pilot_postrun_v4()
        assert bundle["runtime_binding"]["raw_prediction_count"] == 640
        assert len(calls) == 1
        run, cache, package_root, kwargs = calls[0]
        assert run == plan["run_directory"]
        assert cache == plan["development_cache_directory"]
        assert package_root == output.resolve() / pilot_v4.PILOT_PACKAGE_RELATIVE_DIRECTORY
        assert kwargs == {
            "atlas_semantics": {"fixed": "semantics"},
            "development_evaluation_animal_ids": animals,
            "annotation_volume_ap_dv_ml": pytest.approx(np.ones((2, 2, 2))),
            "top_k": 4,
            "refinement_steps": 3,
            "pose_only_steps": 2,
            "retrieval_shape_h_w": (48, 48),
            "catalogue_chunk_size": 512,
            "gauss_hermite_order": 5,
            "evaluation_seed": 20260902,
            "minimum_jacobian": 0.05,
            "maximum_cycle_error_px": 1.0,
            "device": "cuda",
        }
        with pytest.raises(FileExistsError, match="must be new"):
            pilot_v4.run_arbitrary_plane_finite_pilot_postrun_v4()
        assert len(calls) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fake_verified_package(root, plan):
    package_root = root / pilot_v4.PILOT_PACKAGE_RELATIVE_DIRECTORY
    evaluation_root = package_root / "internal_development_evaluation"
    evaluation_root.mkdir(parents=True)
    animals = [f"development-animal-{index:02d}" for index in range(40)]
    rows = [
        {
            "animal_id": animals[index % len(animals)],
            "finite_psf_schedule_binding": {
                "source": "authenticated-per-row-finite_psf_contract",
                "render_mode": "finite_boxcar",
                "axial_sample_count": 9,
            },
        }
        for index in range(640)
    ]
    annotation = {
        "file_sha256": "8" * 64,
        "contains_atlas_intensity": False,
    }
    generic_configuration = {
        "all_development_cache_rows_evaluated": True,
        "development_evaluation_animal_ids": animals,
        "top_k": 4,
        "refinement_steps": 3,
        "pose_only_steps": 2,
        "retrieval_shape_h_w": [48, 48],
        "catalogue_chunk_size": 512,
        "gauss_hermite_order": 5,
        "evaluation_seed": 20260902,
        "minimum_jacobian": 0.05,
        "maximum_cycle_error_px": 1.0,
        "device": "cuda",
        "per_row_schedule_source": "finite_psf_contract",
        "global_schedule_fallback": None,
    }
    generic_payload = {
        "output_directory": str(package_root.resolve()),
        "configuration": generic_configuration,
        "run_binding": {
            "directory": plan["run_directory"],
            "run_id": "pilot-run",
            "run_manifest_receipt_sha256": "2" * 64,
            "run_state_receipt_sha256": "3" * 64,
        },
        "development_cache_binding": {
            "directory": plan["development_cache_directory"],
            "manifest_receipt_sha256": "4" * 64,
            "row_count": 640,
            "selected_row_indices": list(range(640)),
            "finite_psf_run_contract": {
                "render_mode": "finite_boxcar",
                "axial_sample_count": 9,
            },
        },
        "calibration": {
            "status": "absent-uncalibrated",
            "calibration_receipt": None,
        },
        "artifacts": {
            "regional_annotation": annotation,
            "development_evaluation": {
                "relative_directory": "internal_development_evaluation",
            },
        },
    }
    generic = {
        **generic_payload,
        "receipt_sha256": evaluation_v4._sha(generic_payload),
    }
    generic_path = package_root / "finite_postrun_bundle_receipt.json"
    generic_path.write_bytes(evaluation_v4._canonical_json(generic))
    report_payload = {
        "experiment_scope": "finite-thickness-production-s9",
        "catalogue_binding": {
            "catalogue_id": "pilot-catalogue",
            "receipt_sha256": "7" * 64,
            "cell_count": 98_304,
        },
        "row_accounting": {
            "selected_row_count": 640,
            "reported_row_count": 640,
            "no_rows_dropped": True,
        },
        "row_reports": rows,
        "animal_macro_metrics": {
            "statistical_unit": "animal",
            "animal_count": 40,
            "per_animal": {animal: {} for animal in animals},
        },
        "uncertainty_scope": evaluation_v4.UNCALIBRATED_SCOPE,
        "calibration_fitted": False,
        "regional_annotation_artifact": annotation,
    }
    report = {
        **report_payload,
        "receipt_sha256": evaluation_v4._sha(report_payload),
    }
    (evaluation_root / "finite_development_evaluation_report.json").write_bytes(
        evaluation_v4._canonical_json(report)
    )
    package_record = {
        "relative_directory": pilot_v4.PILOT_PACKAGE_RELATIVE_DIRECTORY,
        "generic_bundle_relative_path": "finite_postrun_bundle_receipt.json",
        "generic_bundle_file_sha256": evaluation_v4._file_sha256(generic_path),
        "generic_bundle_receipt_sha256": generic["receipt_sha256"],
    }
    wrapper_payload = {
        "schema_version": pilot_v4.FINITE_PILOT_POSTRUN_BUNDLE_V4_SCHEMA,
        "scientific_scope": pilot_v4.FINITE_PILOT_POSTRUN_SCIENTIFIC_SCOPE,
        "output_directory": str(root.resolve()),
        "source_sha256": pilot_v4._source_receipts(),
        "plan": plan,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "package": package_record,
        "runtime_binding": pilot_v4._runtime_binding(generic, report),
        "calibration": generic["calibration"],
        "public_benchmark_accessed": False,
        "external_validation_accessed": False,
        "final_test_accessed": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    wrapper = {
        **wrapper_payload,
        "receipt_sha256": evaluation_v4._sha(wrapper_payload),
    }
    wrapper_path = root / pilot_v4.PILOT_BUNDLE_RELATIVE_PATH
    wrapper_path.write_bytes(evaluation_v4._canonical_json(wrapper))
    return generic_path, wrapper_path


def test_independent_pilot_verifier_enforces_full_catalogue_chunk_and_row_contract(monkeypatch):
    root = _test_root()
    plan = _plan(root)
    try:
        generic_path, wrapper_path = _fake_verified_package(root, plan)
        monkeypatch.setattr(pilot_v4, "finite_pilot_postrun_plan_v4", lambda: plan)
        monkeypatch.setattr(
            pilot_verify_v4,
            "verify_arbitrary_plane_finite_package_v4",
            lambda package: True,
        )
        assert pilot_verify_v4.verify_arbitrary_plane_finite_pilot_package_v4(root)

        generic = json.loads(generic_path.read_text("ascii"))
        generic["configuration"]["catalogue_chunk_size"] = 128
        generic_payload = {
            key: value for key, value in generic.items() if key != "receipt_sha256"
        }
        generic["receipt_sha256"] = evaluation_v4._sha(generic_payload)
        generic_path.write_bytes(evaluation_v4._canonical_json(generic))
        wrapper = json.loads(wrapper_path.read_text("ascii"))
        wrapper["package"]["generic_bundle_file_sha256"] = evaluation_v4._file_sha256(
            generic_path
        )
        wrapper["package"]["generic_bundle_receipt_sha256"] = generic[
            "receipt_sha256"
        ]
        wrapper_payload = {
            key: value for key, value in wrapper.items() if key != "receipt_sha256"
        }
        wrapper["receipt_sha256"] = evaluation_v4._sha(wrapper_payload)
        wrapper_path.write_bytes(evaluation_v4._canonical_json(wrapper))
        with pytest.raises(ValueError, match="exact pilot plan"):
            pilot_verify_v4.verify_arbitrary_plane_finite_pilot_package_v4(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
