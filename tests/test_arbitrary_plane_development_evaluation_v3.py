import hashlib
import json
import copy
import shutil

import numpy as np
import pytest
import torch

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_development_evaluation_v3 as evaluation_v3
import training.arbitrary_plane_inference_v3 as inference_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_staged_training as staged_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3
import training.arbitrary_plane_training_row_v3 as training_row_v3


def _row(index, split, mode):
    row = copy.deepcopy(fixture.row(index, split=split))
    row["selected_mode"] = mode
    if mode == "smart-brush-absent":
        row["arrays"]["model_input_channels_float32"][..., 1:] = 0.0
    row["array_receipts"] = {
        name: acquisition_v2._array_receipt(value)
        for name, value in row["arrays"].items()
    }
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        training_row_v3.training_row_receipt_v3(row)
    )
    return row


def _frozen_cache(path, rows):
    row_cache_v3.initialize_training_row_cache_v3(
        path,
        generator_binding=fixture.generator_binding(),
        generation_config={"plane_domain": "all brain-intersecting", "row_count": len(rows)},
        seed_record={"root_seed": "0x1234", "purpose": "development-evaluation-fixture"},
    )
    row_cache_v3.append_training_rows_v3(path, rows)
    return row_cache_v3.freeze_training_row_cache_v3(path)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("development-evaluation-v3")
    catalogue = fixture.catalogue()
    atlas = fixture.atlas()
    annotation = np.ones(atlas.shape[-3:], dtype=np.int64)
    train_cache = root / "training-cache"
    _frozen_cache(
        train_cache,
        [_row(0, "development-training", "smart-brush-accurate")],
    )
    source = root / "atlas-source.bin"
    source.write_bytes(b"authenticated Allen fixture")
    model_kwargs = {
        **fixture.model_kwargs(),
        "deformation_support_floor": 1e-4,
        "deformation_maximum_velocity_gradient": 0.35,
        "proposal_count": None,
        "proposal_channels": 16,
        "proposal_mixture_components": 8,
        "proposal_offset_scale_um": 10000.0,
    }
    run = root / "training-run"
    runner_v3.initialize_training_run_v3(
        run,
        cache_directory=train_cache,
        expected_generator_binding=fixture.generator_binding(),
        catalogue=catalogue,
        atlas_volume=atlas,
        atlas_source_assets=(
            {
                "path": str(source),
                "role": "test atlas",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "fixed test normalization"},
        model_kwargs=model_kwargs,
        training_config=fixture.training_config(),
        runner_config=fixture.runner_config(target=1),
        device="cpu",
    )
    runner_v3.run_training_attempts_v3(run, max_attempts=1)
    trained = runner_v3.load_training_run_v3(run)
    staged_path = trained["run_root"] / trained["run_state"]["latest_checkpoint"][
        "relative_path"
    ]
    training_receipt = staged_v3.make_staged_training_export_receipt_v3(staged_path)
    atlas_semantics = {
        "schema_version": "anatomy-tracker.atlas-semantics/v3",
        "atlas_name": "development evaluator test atlas",
        "atlas_version": "test-v1",
        "processed_channel_names": ["test_intensity", "test_support"],
        "processed_channel_recipes": ["fixed test scale", "test support"],
        "source_assets": [
            {"asset_role": "test", "uri": "test://atlas", "sha256": "d" * 64}
        ],
        "source_format": "synthetic test tensor",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["positive AP", "positive DV", "positive ML"],
        "voxel_center_convention": "integer array coordinates denote voxel centres",
        "normalization_parameters": {"divisor": 1.0},
    }
    inference_contract = inference_v3.make_inference_contract_v3(
        atlas,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-0.5, 0.0, 0.5),
        (0.25, 0.5, 0.25),
        atlas_semantics=atlas_semantics,
        annotation_volume_ap_dv_ml=annotation,
    )
    checkpoint = inference_v3.make_arbitrary_plane_joint_checkpoint_v3(
        trained["training_state"]["model"],
        model_kwargs,
        catalogue,
        {
            "initialization": "fresh_random",
            "architecture_source": "training.arbitrary_plane_joint_model",
            "prior_trained_model_dependencies": [],
            "prior_model_feature_dependencies": [],
            "pseudolabel_dependencies": [],
            "dataset_provenance": ["authenticated synthetic generator v3 test fixture"],
            "animal_specimen_experiment_id_contract": "exact IDs retained; animal-disjoint evaluation",
        },
        training_receipt,
        inference_contract=inference_contract,
    )
    checkpoint_path = root / "joint-inference-checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = inference_v3.load_arbitrary_plane_inference_v3(
        checkpoint_path, catalogue, device="cpu"
    )
    feature_cache_path = root / "complete-catalogue-features.pt"
    inference_v3.make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-0.5, 0.0, 0.5),
        (0.25, 0.5, 0.25),
        feature_cache_path,
        retrieval_shape_h_w=(4, 4),
        build_chunk_size=2,
        annotation_volume_ap_dv_ml=annotation,
    )
    development_cache = root / "development-cache"
    _frozen_cache(
        development_cache,
        [
            _row(1, "development-evaluation", "smart-brush-accurate"),
            _row(2, "development-evaluation", "smart-brush-imperfect"),
            _row(3, "development-evaluation", "smart-brush-absent"),
        ],
    )
    leaking_cache = root / "leaking-cache"
    _frozen_cache(
        leaking_cache,
        [_row(0, "development-evaluation", "smart-brush-accurate")],
    )
    return {
        "root": root,
        "catalogue": catalogue,
        "atlas": atlas,
        "annotation": annotation,
        "checkpoint": checkpoint_path,
        "feature_cache": feature_cache_path,
        "development_cache": development_cache,
        "leaking_cache": leaking_cache,
    }


def _run(
    prepared,
    output,
    cache=None,
    animals=("animal-1", "animal-2", "animal-3"),
    row_indices=None,
):
    return evaluation_v3.run_arbitrary_plane_development_evaluation_v3(
        prepared["development_cache"] if cache is None else cache,
        prepared["checkpoint"],
        prepared["catalogue"],
        prepared["atlas"],
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-0.5, 0.0, 0.5),
        (0.25, 0.5, 0.25),
        output,
        development_evaluation_animal_ids=animals,
        row_indices=row_indices,
        annotation_volume_ap_dv_ml=prepared["annotation"],
        top_k=3,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(4, 4),
        catalogue_chunk_size=2,
        gauss_hermite_order=3,
        evaluation_seed=719,
        catalogue_feature_cache_path=prepared["feature_cache"],
        device="cpu",
    )


def test_development_evaluator_runs_honest_complete_catalogue_and_binds_metrics(prepared):
    output = prepared["root"] / "evaluation"
    bundle = _run(prepared, output)
    assert bundle["schema_version"] == evaluation_v3.DEVELOPMENT_EVALUATION_BUNDLE_V3_SCHEMA
    assert evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(
        output, catalogue=prepared["catalogue"]
    )
    report = json.loads((output / "development_evaluation_report.json").read_text())
    assert report["data_role"] == "internal-development-only"
    assert report["public_benchmark_accessed"] is False
    assert report["final_test_accessed"] is False
    assert report["calibration_fitted"] is False
    assert report["configuration"]["catalogue_feature_cache"]["cache_id"]
    assert report["animal_macro_metrics"]["statistical_unit"] == "animal"
    assert report["animal_macro_metrics"]["animal_count"] == 3
    assert len(report["row_reports"]) == 3
    assert report["row_accounting"]["no_rows_dropped"] is True
    assert {
        mode: values["row_count"]
        for mode, values in report["mode_stratified_metrics"].items()
    } == {
        "smart-brush-absent": 1,
        "smart-brush-accurate": 1,
        "smart-brush-imperfect": 1,
    }
    for row in report["row_reports"]:
        assert row["metrics"]["pose"]["primary_internal_development_metric"] == (
            "physical_finite_frame_landmark_mean_um"
        )
        assert row["metrics"]["retrieval"]["catalogue_complete"] is True
        assert row["metrics"]["retrieval"]["teacher_forcing_used"] is False
        assert row["metrics"]["uncertainty"]["coverage_claimed"] is False
        assert row["metrics"]["uncertainty"]["calibration_fitted_by_evaluator"] is False
        assert row["metrics"]["regional_overlap"]["available"] is True
        assert row["disposition"]["no_silent_drop"] is True
        raw_path = output / row["raw_prediction"]["relative_path"]
        assert raw_path.is_file()
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        expected_lineage = {
            "animal_id": row["animal_id"],
            "specimen_id": row["specimen_id"],
            "experiment_id": row["experiment_id"],
            "synthetic_animal_id": row["synthetic_animal_id"],
            "section_id": row["section_id"],
            "synthetic_realization_id": row["synthetic_realization_id"],
        }
        assert raw["identifiers"]["lineage"] == [expected_lineage]
        assert raw["raw_prediction"]["lineage"] == [expected_lineage]


def test_development_evaluator_rejects_animal_leakage_and_raw_prediction_tamper(prepared):
    with pytest.raises(ValueError, match="animal leakage"):
        _run(
            prepared,
            prepared["root"] / "leakage-output",
            cache=prepared["leaking_cache"],
            animals=("animal-0",),
        )
    output = prepared["root"] / "tamper-evaluation"
    _run(prepared, output)
    report = json.loads((output / "development_evaluation_report.json").read_text())
    raw = output / report["row_reports"][0]["raw_prediction"]["relative_path"]
    with raw.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="raw prediction file hash"):
        evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(output)


@pytest.mark.parametrize(
    ("plural_field", "lineage_field"),
    (
        ("synthetic_animal_ids", "synthetic_animal_id"),
        ("section_ids", "section_id"),
        ("synthetic_realization_ids", "synthetic_realization_id"),
    ),
)
def test_raw_artifact_verifier_rejects_resealed_lineage_tamper(
    prepared, plural_field, lineage_field
):
    output = prepared["root"] / f"lineage-tamper-{lineage_field}"
    _run(prepared, output, animals=("animal-1",), row_indices=(0,))
    report_path = output / "development_evaluation_report.json"
    bundle_path = output / "bundle_receipt.json"
    report = json.loads(report_path.read_text())
    raw_record = report["row_reports"][0]["raw_prediction"]
    raw_path = output / raw_record["relative_path"]
    artifact = torch.load(raw_path, map_location="cpu", weights_only=True)
    artifact["identifiers"][plural_field][0] = "tampered-lineage"
    artifact["identifiers"]["lineage"][0][lineage_field] = "tampered-lineage"
    artifact["input_receipt"]["identifiers"] = copy.deepcopy(
        artifact["identifiers"]
    )
    input_payload = {
        key: value
        for key, value in artifact["input_receipt"].items()
        if key != "receipt_sha256"
    }
    artifact["input_receipt"]["receipt_sha256"] = inference_v3._sha(input_payload)
    artifact["raw_prediction"]["lineage"][0][lineage_field] = "tampered-lineage"
    artifact["raw_prediction_receipt"] = inference_v3._prediction_receipt(
        artifact["raw_prediction"]
    )
    torch.save(artifact, raw_path)
    raw_record["file_sha256"] = evaluation_v3._file_sha256(raw_path)
    raw_record["input_receipt"] = artifact["input_receipt"]
    raw_record["prediction_receipt"] = artifact["raw_prediction_receipt"]
    report_payload = {
        key: value for key, value in report.items() if key != "receipt_sha256"
    }
    report["receipt_sha256"] = evaluation_v3._hash_json(report_payload)
    report_path.write_bytes(evaluation_v3._canonical_json(report))
    bundle = json.loads(bundle_path.read_text())
    bundle["report_file_sha256"] = evaluation_v3._file_sha256(report_path)
    bundle["report_receipt_sha256"] = report["receipt_sha256"]
    bundle_payload = {
        key: value for key, value in bundle.items() if key != "receipt_sha256"
    }
    bundle["receipt_sha256"] = evaluation_v3._hash_json(bundle_payload)
    bundle_path.write_bytes(evaluation_v3._canonical_json(bundle))
    with pytest.raises(ValueError, match="raw prediction receipt"):
        evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(output)


def test_development_evaluator_streams_rows_and_predictions(prepared, monkeypatch):
    def no_bulk_rows(*args, **kwargs):
        raise AssertionError("development evaluator must not materialize selected rows")

    loaded_indices = []
    original = row_cache_v3._load_record

    def tracked(root, record, contract):
        loaded_indices.append(record["row_index"])
        return original(root, record, contract)

    monkeypatch.setattr(row_cache_v3, "load_training_rows_v3", no_bulk_rows)
    monkeypatch.setattr(row_cache_v3, "audit_training_row_cache_v3", no_bulk_rows)
    monkeypatch.setattr(row_cache_v3, "_load_record", tracked)
    output = prepared["root"] / "streamed-evaluation"
    _run(prepared, output)
    assert loaded_indices == [0, 1, 2]
    assert evaluation_v3.verify_arbitrary_plane_development_evaluation_v3(
        output, catalogue=prepared["catalogue"]
    )
    assert loaded_indices == [0, 1, 2, 0, 1, 2]


def test_development_evaluator_still_authenticates_unselected_rows(prepared):
    cache = prepared["root"] / "unselected-row-tamper-cache"
    shutil.copytree(prepared["development_cache"], cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    unselected = cache / manifest["rows"][1]["arrays_relative_path"]
    with unselected.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="cached training-row file hash"):
        _run(
            prepared,
            prepared["root"] / "unselected-row-tamper-evaluation",
            cache=cache,
            animals=("animal-1",),
            row_indices=(0,),
        )
