import json

import numpy as np
import pytest
import torch

import test_arbitrary_plane_finite_training_runner_v4 as runner_fixture
import training.arbitrary_plane_finite_development_evaluation_v4 as evaluation_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v4 as cache_v4
import training.run_arbitrary_plane_finite_postrun_v4 as postrun_v4
from training.verify_arbitrary_plane_finite_package_v4 import (
    verify_arbitrary_plane_finite_package_v4,
)


def _semantics(manifest):
    return {
        "schema_version": inference_v3.ATLAS_SEMANTICS_V3_SCHEMA,
        "atlas_name": "finite postrun fixture",
        "atlas_version": "v4-test",
        "processed_channel_names": ["intensity", "boundary"],
        "processed_channel_recipes": ["fixed fixture", "fixed boundary"],
        "source_assets": [
            {
                "asset_role": asset["role"],
                "uri": asset["path"],
                "sha256": asset["sha256"],
            }
            for asset in manifest["atlas"]["binding"]["source_assets"]
        ],
        "source_format": "synthetic test tensor",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["positive AP", "positive DV", "positive ML"],
        "voxel_center_convention": "integer array coordinates denote voxel centres",
        "normalization_parameters": manifest["atlas"]["binding"]["preprocessing"],
    }


def _development_cache(root, *, render_mode="finite_boxcar"):
    binding = runner_fixture._binding(2, render_mode)
    cache = root / "development-cache"
    cache_v4.initialize_training_row_cache_v4(cache, generator_binding=binding)
    thickness = 0.0 if render_mode == "centre_plane_ablation" else 31.0
    rows = [
        runner_fixture._row(
            index, binding,
            thickness if render_mode == "centre_plane_ablation" else thickness + index,
            render_mode,
            zero=index == 2,
        )
        for index in (1, 2)
    ]
    cache_v4.append_training_rows_v4(cache, rows)
    return cache, cache_v4.freeze_training_row_cache_v4(cache), rows


def test_completed_finite_s9_package_binds_every_row_schedule_and_raw_prediction(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, frozen, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    bundle = postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        annotation_volume_ap_dv_ml=np.ones((10, 10, 10), dtype=np.int64),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    assert bundle["development_cache_binding"]["manifest_receipt_sha256"] == frozen["receipt_sha256"]
    assert bundle["run_binding"]["status"] == "completed"
    assert bundle["calibration"]["status"] == "absent-uncalibrated"
    assert verify_arbitrary_plane_finite_package_v4(output)
    report = json.loads(
        (output / "internal_development_evaluation" /
         "finite_development_evaluation_report.json").read_text("ascii")
    )
    assert report["experiment_scope"] == "finite-thickness-production-s9"
    assert report["row_accounting"]["reported_row_count"] == 2
    assert report["row_accounting"]["no_rows_dropped"] is True
    assert report["row_accounting"]["marginal_or_empty_rows_retained"] is True
    assert report["row_reports"][1]["supervision_disposition"]["pose_weight"] == 0.0
    assert report["row_reports"][1]["supervision_disposition"]["dense_weight"] == 0.0
    assert report["animal_macro_metrics"]["statistical_unit"] == "animal"
    assert [
        row["finite_psf_schedule_binding"]["nominal_cut_thickness_um"]
        for row in report["row_reports"]
    ] == [32.0, 33.0]
    assert all(
        row["finite_psf_schedule_binding"]["axial_sample_count"] == 9
        for row in report["row_reports"]
    )
    assert len(list((output / "internal_development_evaluation" / "raw_predictions").glob("*.pt"))) == 2
    raw_path = next(
        (output / "internal_development_evaluation" / "raw_predictions").glob("*.pt")
    )
    with raw_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="raw-prediction hash"):
        verify_arbitrary_plane_finite_package_v4(output)


def test_finite_postrun_rejects_incomplete_run_before_creating_output(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=2
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="exact completed target"):
        postrun_v4.run_arbitrary_plane_finite_postrun_v4(
            run,
            development,
            output,
            atlas_semantics=_semantics(manifest),
            development_evaluation_animal_ids=tuple(
                row["lineage"]["animal_id"] for row in rows
            ),
        )
    assert not output.exists()


def test_s1_ablation_cannot_be_packaged_as_s9(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1,
        render_mode="centre_plane_ablation",
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(
        tmp_path / "evaluation", render_mode="finite_boxcar"
    )
    with pytest.raises(ValueError, match="experiment scopes differ"):
        postrun_v4.run_arbitrary_plane_finite_postrun_v4(
            run,
            development,
            tmp_path / "package",
            atlas_semantics=_semantics(manifest),
            development_evaluation_animal_ids=tuple(
                row["lineage"]["animal_id"] for row in rows
            ),
        )


def test_s1_ablation_is_packaged_only_under_its_exact_scope(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1,
        render_mode="centre_plane_ablation",
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(
        tmp_path / "evaluation", render_mode="centre_plane_ablation"
    )
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    report = json.loads(
        (output / "internal_development_evaluation" /
         "finite_development_evaluation_report.json").read_text("ascii")
    )
    assert report["experiment_scope"] == "exact-zero-thickness-ablation-s1"
    assert all(
        row["finite_psf_schedule_binding"]["axial_sample_count"] == 1
        and row["finite_psf_schedule_binding"]["nominal_cut_thickness_um"] == 0.0
        for row in report["row_reports"]
    )
    assert verify_arbitrary_plane_finite_package_v4(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_evaluation_normalizes_raw_metrics_and_annotation_to_cpu(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        annotation_volume_ap_dv_ml=np.zeros((10, 10, 10), dtype=np.int64),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=1,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cuda",
    )
    assert verify_arbitrary_plane_finite_package_v4(output)


def test_resealed_metric_tamper_is_rejected_against_raw_prediction(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=1,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    evaluation_root = output / "internal_development_evaluation"
    report_path = evaluation_root / "finite_development_evaluation_report.json"
    bundle_path = evaluation_root / "bundle_receipt.json"
    report = json.loads(report_path.read_text("ascii"))
    report["row_reports"][0]["metrics"]["pose"][
        "physical_finite_frame_landmark_mean_um"
    ] += 1000.0
    report["receipt_sha256"] = evaluation_v4._sha({
        key: value for key, value in report.items() if key != "receipt_sha256"
    })
    report_path.write_bytes(evaluation_v4._canonical_json(report))
    bundle = json.loads(bundle_path.read_text("ascii"))
    bundle["report_file_sha256"] = evaluation_v4._file_sha256(report_path)
    bundle["report_receipt_sha256"] = report["receipt_sha256"]
    bundle["receipt_sha256"] = evaluation_v4._sha({
        key: value for key, value in bundle.items() if key != "receipt_sha256"
    })
    bundle_path.write_bytes(evaluation_v4._canonical_json(bundle))
    with pytest.raises(ValueError, match="metric differs from raw prediction"):
        evaluation_v4.verify_arbitrary_plane_finite_development_evaluation_v4(
            evaluation_root,
            catalogue=runner_v4.load_finite_training_run_v4(run)["catalogue"],
        )


def test_packaged_regional_annotation_tamper_is_rejected(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        annotation_volume_ap_dv_ml=np.ones((10, 10, 10), dtype=np.int64),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    annotation_path = (
        output / "internal_development_evaluation" /
        evaluation_v4.REGIONAL_ANNOTATION_RELATIVE_PATH
    )
    with np.load(annotation_path, allow_pickle=False) as archive:
        annotation = archive[evaluation_v4.REGIONAL_ANNOTATION_ARRAY_KEY].copy()
    annotation[0, 0, 0] = 2
    with annotation_path.open("wb") as handle:
        np.savez_compressed(handle, **{
            evaluation_v4.REGIONAL_ANNOTATION_ARRAY_KEY: annotation
        })
    with pytest.raises(ValueError, match="regional annotation"):
        verify_arbitrary_plane_finite_package_v4(output)


def test_resealed_regional_metric_tamper_is_recomputed_from_packaged_annotation(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        annotation_volume_ap_dv_ml=np.ones((10, 10, 10), dtype=np.int64),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    evaluation_root = output / "internal_development_evaluation"
    report_path = evaluation_root / "finite_development_evaluation_report.json"
    evaluation_bundle_path = evaluation_root / "bundle_receipt.json"
    package_bundle_path = output / "finite_postrun_bundle_receipt.json"
    report = json.loads(report_path.read_text("ascii"))
    regional = report["row_reports"][0]["metrics"]["regional_overlap"]
    regional["available"] = not regional["available"]
    report["receipt_sha256"] = evaluation_v4._sha({
        key: value for key, value in report.items() if key != "receipt_sha256"
    })
    report_path.write_bytes(evaluation_v4._canonical_json(report))
    evaluation_bundle = json.loads(evaluation_bundle_path.read_text("ascii"))
    evaluation_bundle["report_file_sha256"] = evaluation_v4._file_sha256(report_path)
    evaluation_bundle["report_receipt_sha256"] = report["receipt_sha256"]
    evaluation_bundle["receipt_sha256"] = evaluation_v4._sha({
        key: value for key, value in evaluation_bundle.items()
        if key != "receipt_sha256"
    })
    evaluation_bundle_path.write_bytes(evaluation_v4._canonical_json(evaluation_bundle))
    package_bundle = json.loads(package_bundle_path.read_text("ascii"))
    packaged_evaluation = package_bundle["artifacts"]["development_evaluation"]
    packaged_evaluation["bundle_file_sha256"] = evaluation_v4._file_sha256(
        evaluation_bundle_path
    )
    packaged_evaluation["bundle_receipt_sha256"] = evaluation_bundle[
        "receipt_sha256"
    ]
    package_bundle["receipt_sha256"] = evaluation_v4._sha({
        key: value for key, value in package_bundle.items()
        if key != "receipt_sha256"
    })
    package_bundle_path.write_bytes(evaluation_v4._canonical_json(package_bundle))
    with pytest.raises(ValueError, match="metric differs from raw prediction"):
        verify_arbitrary_plane_finite_package_v4(output)


def test_unreported_raw_prediction_file_is_rejected(tmp_path):
    run, _, _, manifest, _ = runner_fixture._prepared(
        tmp_path / "training", row_count=1, target=1
    )
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    development, _, rows = _development_cache(tmp_path / "evaluation")
    output = tmp_path / "package"
    postrun_v4.run_arbitrary_plane_finite_postrun_v4(
        run,
        development,
        output,
        atlas_semantics=_semantics(manifest),
        development_evaluation_animal_ids=tuple(
            row["lineage"]["animal_id"] for row in rows
        ),
        top_k=2,
        refinement_steps=1,
        pose_only_steps=1,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        device="cpu",
    )
    evaluation_root = output / "internal_development_evaluation"
    torch.save({}, evaluation_root / "raw_predictions" / "unreported.pt")
    with pytest.raises(ValueError, match="artifact set differs"):
        evaluation_v4.verify_arbitrary_plane_finite_development_evaluation_v4(
            evaluation_root,
            catalogue=runner_v4.load_finite_training_run_v4(run)["catalogue"],
        )
