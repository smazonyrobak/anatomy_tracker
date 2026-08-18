import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import training.evaluate_sealed_registered_holdout as evaluator
from training.evaluate_sealed_registered_holdout import (
    SEALED_SPLIT,
    brain_masked_plane_distance,
    evaluation_domains,
    ordered_experiment_groups,
    paired_animal_bootstrap,
    paired_animal_joint_superiority,
    prediction_rows,
    require_complete_sealed_images,
    release_quality_gate,
    sealed_release_report,
    validate_sealed_boundary,
    verify_complete_sealed_image_hashes,
)
from training.atlas_pose_release_contract import RELEASE_GATE_THRESHOLDS
from training.deepslice_benchmark_orientation import (
    DEEPSLICE_ORIENTATION_ADAPTER_VERSION,
    horizontal_image_frame_ouv,
    horizontally_flipped_deepslice_inputs,
)
from source.atlas_pose_runtime import (
    ATLAS_POSE_PREPROCESSING_VERSION,
    AUTOMATIC_BRAIN_MASK_VERSION,
    QUICKNII_COORDINATE_CONTRACT_VERSION,
    atlas_pose_preprocessing_contract_sha256,
)
from training.train_atlas_pose_v7 import FINAL_GATE_THRESHOLDS as TRAINING_FINAL_GATE_THRESHOLDS


def record(section_id, experiment_id, specimen_id, section_number, split=SEALED_SPLIT):
    return {
        "section_image_id": section_id,
        "experiment_id": experiment_id,
        "specimen_id": specimen_id,
        "section_number": section_number,
        "split": split,
        "quicknii_ouv": [0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0],
        "relative_path": f"images/{split}/{experiment_id}/{section_id}.jpg",
        "in_training_ap_domain": True,
    }


def test_sealed_records_are_grouped_by_experiment_and_cutting_index():
    records = [
        record(30, 2, 20, 8),
        record(12, 1, 10, 4),
        record(11, 1, 10, 4),
        record(10, 1, 10, 2),
    ]
    groups = ordered_experiment_groups(validate_sealed_boundary(records))
    assert list(groups) == [1, 2]
    assert [(row["section_number"], row["section_image_id"]) for row in groups[1]] == [
        (2, 10),
        (4, 11),
        (4, 12),
    ]


def test_brain_masked_ouv_plane_distance_uses_corresponding_pixel_locations():
    ground_truth = np.asarray([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    translated = ground_truth.copy()
    translated[0] += 3.0
    mask = np.asarray([[True, False], [False, True]])
    assert brain_masked_plane_distance(ground_truth, translated, mask) == pytest.approx(3.0)

    sheared = ground_truth.copy()
    sheared[3] += 4.0
    left_column = np.asarray([[True, False], [True, False]])
    assert brain_masked_plane_distance(ground_truth, sheared, left_column) == pytest.approx(1.0)


def test_annotation_mask_maps_quicknii_r_directly_to_asymmetric_ccf_ml():
    annotation = np.zeros((5, 5, 456), dtype=np.uint8)
    annotation[2, 2, 51] = 1
    ouv = np.asarray([-126.0, 526.0, 318.0, 708.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = evaluator.annotation_brain_mask(ouv, annotation, shape=(1, 2))
    assert np.array_equal(mask, np.asarray([[True, False]]))


def test_deepslice_horizontal_adapter_preserves_asymmetric_physical_plane_and_laterality(
    tmp_path: Path,
):
    raw_ouv = np.asarray(
        [40.0, 275.0, 260.0, 390.0, 28.0, 12.0, 15.0, -18.0, -255.0]
    )
    raw_pose = evaluator.quicknii_to_tracker_pose(raw_ouv)
    assert abs(raw_pose[1]) > 1.0
    assert abs(raw_pose[2]) > 1.0

    canonical_ouv = horizontal_image_frame_ouv(raw_ouv)
    assert np.array_equal(horizontal_image_frame_ouv(canonical_ouv), raw_ouv)
    assert np.allclose(evaluator.quicknii_to_tracker_pose(canonical_ouv), raw_pose)
    for raw_s, vertical_t in (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.18, 0.37),
        (0.73, 0.62),
    ):
        raw_point = raw_ouv[:3] + raw_s * raw_ouv[3:6] + vertical_t * raw_ouv[6:9]
        canonical_point = (
            canonical_ouv[:3]
            + (1.0 - raw_s) * canonical_ouv[3:6]
            + vertical_t * canonical_ouv[6:9]
        )
        assert np.allclose(canonical_point, raw_point)

    unilateral_s, unilateral_t = 0.18, 0.37
    raw_landmark = (
        raw_ouv[:3]
        + unilateral_s * raw_ouv[3:6]
        + unilateral_t * raw_ouv[6:9]
    )
    assert raw_landmark[0] < evaluator.ATLAS_CENTER_ML_DV[0]
    atlas_reflection = raw_ouv.copy()
    atlas_reflection[0] = evaluator.QUICKNII_SHAPE_ML_AP_DV[0] - raw_ouv[0]
    atlas_reflection[3] = -raw_ouv[3]
    atlas_reflection[6] = -raw_ouv[6]
    reflected_landmark = (
        atlas_reflection[:3]
        + unilateral_s * atlas_reflection[3:6]
        + unilateral_t * atlas_reflection[6:9]
    )
    reflected_pose = evaluator.quicknii_to_tracker_pose(atlas_reflection)
    assert reflected_landmark[0] > evaluator.ATLAS_CENTER_ML_DV[0]
    assert reflected_pose[1] == pytest.approx(-raw_pose[1])
    assert not np.allclose(atlas_reflection, canonical_ouv)

    pixels = np.arange(15, dtype=np.uint8).reshape(3, 5)
    image_path = tmp_path / "asymmetric.png"
    Image.fromarray(pixels).save(image_path)
    raw_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    with horizontally_flipped_deepslice_inputs([image_path]) as flipped_paths:
        flipped = np.asarray(Image.open(flipped_paths[0]))
        assert np.array_equal(flipped, np.fliplr(pixels))
        assert not np.array_equal(flipped, pixels)
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == raw_sha256


def test_deepslice_modes_flip_once_then_return_postprocessed_plane_to_raw_frame(
    tmp_path: Path, monkeypatch
):
    from DeepSlice.coord_post_processing import spacing_and_indexing

    pixels = np.arange(15, dtype=np.uint8).reshape(3, 5)
    image_path = tmp_path / "1.png"
    Image.fromarray(pixels).save(image_path)
    raw_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    raw_ouv = np.asarray(
        [40.0, 275.0, 260.0, 390.0, 28.0, 12.0, 15.0, -18.0, -255.0]
    )
    canonical_ouv = horizontal_image_frame_ouv(raw_ouv)
    observed_inputs = []

    def observe(paths):
        observed_inputs.append(np.asarray(Image.open(paths[0])))

    def fake_inference(paths, _messages, _cancel):
        observe(paths)
        prediction = {
            "Filenames": Path(paths[0]).name,
            **dict(zip(evaluator.OUV_COLUMNS, canonical_ouv)),
            "width": 5,
            "height": 3,
        }
        return [prediction], "1.2.8", {"primary": "a", "secondary": "b"}, {}, {
            "backend": "ONNX Runtime CPU"
        }

    def fake_preprocess(paths):
        observe(paths)
        return np.zeros((1, 299, 299, 3), dtype=np.float32), [5], [3]

    class PrimarySession:
        def run(self, _outputs, _inputs):
            return [canonical_ouv[None, :]]

    def fake_propagate(table):
        result = table.copy()
        result.loc[:, ("ox", "oy", "oz")] = (
            result.loc[:, ("ox", "oy", "oz")].to_numpy()
            + 0.25 * result.loc[:, ("ux", "uy", "uz")].to_numpy()
        )
        return result

    monkeypatch.setattr(evaluator, "run_deepslice_inference", fake_inference)
    monkeypatch.setattr(evaluator, "preprocess_deepslice_images", fake_preprocess)
    monkeypatch.setattr(
        evaluator,
        "load_deepslice_onnx_sessions",
        lambda _force_cpu: ({"primary": PrimarySession()}, {}, "CPUExecutionProvider", None),
    )
    monkeypatch.setattr(evaluator, "_propagate_angles", fake_propagate)
    monkeypatch.setattr(spacing_and_indexing, "enforce_section_ordering", lambda table: table)
    monkeypatch.setattr(
        spacing_and_indexing,
        "space_according_to_index",
        lambda table, **_kwargs: table,
    )

    benchmark_record = record(1, 100, 200, 1)
    benchmark_record["relative_path"] = image_path.name
    modes, metadata = evaluator.run_deepslice_modes([benchmark_record], [image_path])

    assert len(observed_inputs) == 2
    assert all(np.array_equal(image, np.fliplr(pixels)) for image in observed_inputs)
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == raw_sha256
    postprocessed_canonical = canonical_ouv.copy()
    postprocessed_canonical[:3] += 0.25 * canonical_ouv[3:6]
    expected_raw = horizontal_image_frame_ouv(postprocessed_canonical)[None, :]
    assert all(np.allclose(prediction, expected_raw) for prediction in modes.values())
    assert metadata["runtime"]["orientation_adapter"] == {
        "version": DEEPSLICE_ORIENTATION_ADAPTER_VERSION,
        "input_horizontal_raster_flips": 1,
        "output_ouv_transform": "H(O,U,V)=(O+U,-U,V)",
        "output_transform_after_official_postprocessing": True,
        "raw_images_or_ground_truth_modified": False,
        "physical_atlas_ml_reflection": False,
    }


def test_empty_annotation_mask_only_omits_optional_plane_distance():
    row = {**record(1, 1, 1, 1), "product": "5"}
    ground_truth = np.asarray(row["quicknii_ouv"], dtype=np.float64)
    pose = evaluator.quicknii_to_tracker_pose(ground_truth)[None, :]
    result = prediction_rows(
        [row], "deepslice_ai", pose, ground_truth[None, :], np.zeros((1, 1, 1))
    )
    assert result[0]["plane_distance_voxels"] is None
    assert result[0]["plane_distance_um"] is None
    assert result[0]["absolute_error_ap_um"] == 0.0


def test_sealed_specimen_or_experiment_cannot_cross_the_training_boundary():
    sealed = record(1, 100, 200, 1)
    with pytest.raises(ValueError, match="sealed specimen or experiment"):
        validate_sealed_boundary([sealed, record(2, 101, 200, 2, "train")])
    with pytest.raises(ValueError, match="sealed specimen or experiment"):
        validate_sealed_boundary([sealed, record(2, 100, 201, 2, "validation")])


def test_inference_gate_refuses_an_incomplete_sealed_image_set(tmp_path: Path):
    rows = [record(1, 100, 200, 1)]
    with pytest.raises(RuntimeError, match="SEALED INFERENCE REFUSED"):
        require_complete_sealed_images(tmp_path, rows, expected_sections=1, expected_experiments=1)


def test_animal_bootstrap_weights_animals_not_their_section_counts():
    rows = pd.DataFrame(
        [
            {"method": method, "specimen_id": animal, "section_image_id": section, "error": error}
            for method, animal, section, error in (
                ("candidate", 1, 1, 0.0),
                ("reference", 1, 1, 2.0),
                ("candidate", 2, 2, 10.0),
                ("reference", 2, 2, 0.0),
                ("candidate", 2, 3, 10.0),
                ("reference", 2, 3, 0.0),
                ("candidate", 2, 4, 10.0),
                ("reference", 2, 4, 0.0),
            )
        ]
    )
    result = paired_animal_bootstrap(rows, "candidate", "reference", "error", iterations=100, seed=3)
    assert result["animal_count"] == 2
    assert result["paired_section_count"] == 4
    assert result["delta_candidate_minus_reference"] == pytest.approx(4.0)


def test_joint_superiority_uses_one_complete_paired_animal_resample():
    rows = pd.DataFrame(
        [
            {
                "method": method,
                "specimen_id": animal,
                "section_image_id": animal,
                **{
                    metric: value
                    for metric, value in zip(("ap", "lr", "dv"), errors)
                },
            }
            for animal in (1, 2, 3)
            for method, errors in (
                ("candidate", (1.0, 0.1, 0.2)),
                ("reference", (3.0, 0.4, 0.6)),
            )
        ]
    )
    result = paired_animal_joint_superiority(
        rows, "candidate", "reference", ("ap", "lr", "dv"), iterations=100, seed=4
    )
    assert result["probability_all_components_lower_error"] == 1.0
    assert result["simultaneous_superiority_passed"] is True
    incomplete = rows.drop(rows.index[-1])
    with pytest.raises(ValueError, match="Incomplete paired cohort"):
        paired_animal_joint_superiority(
            incomplete, "candidate", "reference", ("ap", "lr", "dv")
        )


def test_primary_sealed_metrics_exclude_out_of_training_domain_sections():
    table = pd.DataFrame(
        {
            "section_image_id": [1, 2, 3],
            "in_training_ap_domain": [True, False, True],
        }
    )
    primary, excluded = evaluation_domains(table)
    assert primary["section_image_id"].tolist() == [1, 3]
    assert excluded["section_image_id"].tolist() == [2]


def test_sealed_release_report_requires_quality_and_deepslice_superiority():
    assert RELEASE_GATE_THRESHOLDS == TRAINING_FINAL_GATE_THRESHOLDS
    rows = pd.DataFrame(
        [
            {
                "method": "atlas_pose",
                "specimen_id": specimen,
                "section_image_id": specimen,
                "product": str(5 + 3 * (specimen - 1)),
                "in_training_ap_domain": True,
                "gt_ap_um": -1000.0 - 500.0 * specimen,
                "pred_ap_um": -980.0 - 500.0 * specimen,
                "gt_lr_deg": 1.0,
                "pred_lr_deg": 1.2,
                "gt_dv_deg": -2.0,
                "pred_dv_deg": -1.7,
            }
            for specimen in (1, 2)
        ]
    )
    quality = release_quality_gate(rows)
    assert quality["all_gates_passed"] is True
    assert quality["thresholds"] == RELEASE_GATE_THRESHOLDS
    comparisons = [
        {
            "candidate": "atlas_pose",
            "reference": "deepslice_mens_ai_ci",
            "metric": f"absolute_error_{axis}",
            "delta_candidate_minus_reference": -1.0,
            "probability_candidate_lower_error": 0.99,
        }
        for axis in ("ap_um", "lr_deg", "dv_deg")
    ]
    joint = {
        "candidate": "atlas_pose",
        "reference": "deepslice_mens_ai_ci",
        "simultaneous_superiority_passed": True,
    }
    release = sealed_release_report(
        rows,
        comparisons,
        joint,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        {"trainer.py": "d" * 64},
        {"atlas": {"annotation": "e" * 64}},
        {"sections_sha256": "f" * 64},
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "now",
    )
    assert release["release_approved"] is True
    assert release["promotion_ready"] is True
    assert release["release_report_version"] == 3
    assert len(release["release_integrity_sha256"]) == 64

    comparisons[0]["probability_candidate_lower_error"] = 0.80
    rejected = sealed_release_report(
        rows,
        comparisons,
        joint,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        {"trainer.py": "d" * 64},
        {"atlas": {"annotation": "e" * 64}},
        {"sections_sha256": "f" * 64},
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "now",
    )
    assert rejected["release_approved"] is False
    assert rejected["promotion_ready"] is False


def _candidate_bundle(tmp_path: Path) -> Path:
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"frozen candidate")
    metadata = {
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "preprocessing_version": ATLAS_POSE_PREPROCESSING_VERSION,
        "automatic_brain_mask_version": AUTOMATIC_BRAIN_MASK_VERSION,
        "quicknii_coordinate_contract": QUICKNII_COORDINATE_CONTRACT_VERSION,
        "preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "source_sha256": {"trainer.py": "1" * 64},
        "manifest_sha256": {"train": "2" * 64},
        "atlas_data_sha256": {"annotation_25.nrrd": "3" * 64},
        "git": {
            "commit": "a" * 40,
            "tracked_source_dirty": False,
            "tracked_source_status": [],
        },
        "registered_data": {
            "sha256": {
                **{
                    name: str(index) * 64
                    for index, name in enumerate(evaluator.SEALED_SOURCE_FILES, 4)
                },
                "nonsealed_image_tree_sha256": "9" * 64,
            },
            "excluded_from_selection": [SEALED_SPLIT],
        },
    }
    model.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return model


def test_global_claim_precedes_any_sealed_access_and_failure_consumes(tmp_path, monkeypatch):
    model = _candidate_bundle(tmp_path)
    state = tmp_path / "state"
    monkeypatch.setattr(evaluator, "SEALED_EVALUATION_STATE_ROOT", state)
    monkeypatch.setattr(
        evaluator,
        "evaluator_environment_commitment",
        lambda: {"contract_version": 1, "commitment_sha256": "9" * 64},
    )

    def sealed_access(*_args):
        assert (state / evaluator.SEALED_CLAIM_NAME).is_file()
        raise RuntimeError("sealed sentinel")

    monkeypatch.setattr(evaluator, "verify_source_commitment", sealed_access)
    with pytest.raises(RuntimeError, match="sealed sentinel"):
        evaluator.run_evaluation(tmp_path / "sealed", model)
    receipt = json.loads((state / evaluator.SEALED_RECEIPT_NAME).read_text())
    assert receipt["status"] == "failed"
    with pytest.raises(RuntimeError, match="already consumed"):
        evaluator.run_evaluation(tmp_path / "sealed", model)


def test_audited_recovery_preserves_failed_receipt_and_frozen_candidate(tmp_path, monkeypatch):
    model = _candidate_bundle(tmp_path)
    state = tmp_path / "state"
    original_environment = {"contract_version": 1, "commitment_sha256": "8" * 64}
    recovery_environment = {"contract_version": 1, "commitment_sha256": "9" * 64}
    monkeypatch.setattr(evaluator, "SEALED_EVALUATION_STATE_ROOT", state)
    monkeypatch.setattr(evaluator, "evaluator_environment_commitment", lambda: original_environment)
    frozen = evaluator.freeze_candidate(model)
    _, claim_path, receipt_path = evaluator.claim_sealed_benchmark(frozen)
    failed = {
        "contract_version": 1,
        "benchmark_id": evaluator.SEALED_BENCHMARK_ID,
        "claim_sha256": evaluator.sha256(claim_path),
        "model_sha256": frozen["model_sha256"],
        "presealed_commitment_sha256": frozen["presealed_sha256"],
        "status": "failed",
        "failed_at_utc": "now",
        "failure": evaluator.SEALED_RECOVERABLE_FAILURE,
    }
    evaluator._atomic_json(receipt_path, failed)
    failed_sha256 = evaluator.sha256(receipt_path)
    monkeypatch.setattr(evaluator, "evaluator_environment_commitment", lambda: recovery_environment)

    recovered = evaluator.load_frozen_candidate_for_recovery(model)
    _, _, _, recovery = evaluator.recover_failed_sealed_benchmark(
        recovered, tmp_path / "sealed"
    )

    assert evaluator.sha256(recovered["model_path"]) == frozen["model_sha256"]
    assert recovery["failed_receipt_sha256"] == failed_sha256
    assert json.loads(recovery["failed_receipt_path"].read_text()) == failed
    assert recovery["commitment"]["recovery_evaluator_environment"] == recovery_environment
    assert json.loads(receipt_path.read_text())["status"] == "claimed_recovery"


def test_candidate_is_mandatory_before_sealed_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "verify_source_commitment",
        lambda *_args: pytest.fail("sealed source was accessed"),
    )
    with pytest.raises(ValueError, match="candidate is mandatory"):
        evaluator.run_evaluation(tmp_path, None)


def test_dirty_candidate_is_refused_before_freezing(tmp_path):
    model = _candidate_bundle(tmp_path)
    metadata_path = model.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git"]["tracked_source_dirty"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean training"):
        evaluator.freeze_candidate(model)


def test_quality_filter_cannot_shrink_the_sealed_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "EXPECTED_SECTIONS", 2)
    monkeypatch.setattr(evaluator, "EXPECTED_EXPERIMENTS", 1)
    rows = [record(1, 100, 200, 1), record(2, 100, 200, 2)]
    (tmp_path / "sections.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "datasets.jsonl").write_text(
        json.dumps(
            {"experiment_id": 100, "specimen_id": 200, "split": SEALED_SPLIT}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"sealed_deepslice_s2p_experiment_ids": [100]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        evaluator,
        "load_registered_image_quality_manifest",
        lambda _root: ({"rejected_records": []}, {1}, set()),
    )

    with pytest.raises(RuntimeError, match="quality filtering removed"):
        evaluator.load_sealed_holdout(tmp_path)


def test_every_sealed_jpeg_is_verified_against_download_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "EXPECTED_SECTIONS", 1)
    monkeypatch.setattr(evaluator, "EXPECTED_EXPERIMENTS", 1)
    row = record(1, 100, 200, 1)
    image = tmp_path / row["relative_path"]
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg payload")
    (tmp_path / "sections.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (tmp_path / "downloads.jsonl").write_text(
        json.dumps({"section_image_id": 1, "sha256": digest}) + "\n",
        encoding="utf-8",
    )
    assert len(verify_complete_sealed_image_hashes(tmp_path)) == 64
    image.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="image checksum failed"):
        verify_complete_sealed_image_hashes(tmp_path)


def test_sealed_evaluator_is_not_imported_by_training_or_model_selection_code():
    root = Path(__file__).parents[1]
    forbidden = "evaluate_sealed_registered_holdout"
    for path in (root / "training").glob("*.py"):
        if path.name == f"{forbidden}.py":
            continue
        assert forbidden not in path.read_text(encoding="utf-8")


def test_sealed_evaluator_uses_gui_free_deepslice_runtime():
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "_tracker_module" not in source
    assert "proprietary_trajectory_tool" not in source
    assert "sys.path" not in source
    assert evaluator.run_deepslice_inference.__module__ == "source.deepslice_runtime"
