import importlib.util
import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_pose_selector_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def _quicknii_prediction(filename: str, atlas_index: float, tilt_lr: float, tilt_dv: float) -> dict:
    ap_size, dv_size, ml_size = TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML
    lr_slope = np.tan(np.deg2rad(tilt_lr))
    dv_slope = np.tan(np.deg2rad(tilt_dv))
    record = {
        "Filenames": filename,
        "ox": float(ml_size),
        "oy": ap_size - (
            atlas_index - lr_slope * ((ml_size - 1) / 2.0) - dv_slope * ((dv_size - 1) / 2.0)
        ),
        "oz": float(dv_size),
        "ux": -float(ml_size),
        "uy": -lr_slope * ml_size,
        "uz": 0.0,
        "vx": 0.0,
        "vy": -dv_slope * dv_size,
        "vz": -float(dv_size),
        "width": ml_size,
        "height": dv_size,
    }
    record["raw_ensemble_ouv"] = [
        record[column]
        for column in ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
    ]
    return record


def test_weighted_vote_fuses_ap_directly_and_tilts_as_plane_normals():
    poses = np.asarray([[-600.0, 38.0, -21.0], [-1000.0, -14.0, 33.0]])
    weights = np.asarray([0.3, 0.7])

    fused = TRACKER.fuse_pose_predictions(poses, weights)
    expected_normal = sum(
        weight * TRACKER.plane_normal_from_tilts(pose[1], pose[2])
        for pose, weight in zip(poses, weights)
    )
    expected_tilts = TRACKER.tilts_from_plane_normal(expected_normal)

    assert fused[0] == pytest.approx(-880.0)
    assert fused[1:] == pytest.approx(expected_tilts)
    assert not np.allclose(fused[1:], np.average(poses[:, 1:], axis=0, weights=weights))


def test_weighted_predictor_uses_smart_mask_and_preserves_both_components(tmp_path, monkeypatch):
    image = np.zeros((90, 120), dtype=np.uint8)
    cv2.ellipse(image, (60, 45), (42, 30), 0, 0, 360, 180, -1)
    path = tmp_path / "slice.png"
    assert cv2.imwrite(str(path), image)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (60, 45), (42, 30), 0, 0, 360, 1, -1)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0]
    points = [tuple(point.astype(float)) for point in contour[:: max(1, len(contour) // 50)]]
    seen = {}

    def fake_deepslice(image_paths, _messages, _cancel):
        filename = Path(image_paths[0]).name
        record = _quicknii_prediction(filename, 240.0, 10.0, -4.0)
        return (
            [record],
            "1.2.8",
            {"primary": "a", "secondary": "b"},
            {filename: {"ap_um": 25.0, "lr_deg": 1.0, "dv_deg": 2.0}},
            {
                "backend": "mock DeepSlice",
                "device": "mock GPU",
                "onnxruntime_version": "mock",
                "inference_seconds": 0.1,
                "preintegration_tilt_spread_deg": [0.0, 0.0],
            },
        )

    def fake_own(images, masks, _model_path, _cancel):
        seen["image"] = images[0]
        seen["mask"] = masks[0]
        return np.asarray([[-1000.0, -6.0, 8.0]]), {
            "architecture": "mock CNN",
            "model_sha256": "own-hash",
            "backend": "mock ONNX",
            "device": "mock GPU",
            "onnxruntime_version": "mock",
            "inference_seconds": 0.2,
            "orientation_inverted": [False],
            "orientation_inverted_logit": [-4.0],
            "metadata": {},
        }

    monkeypatch.setattr(TRACKER, "run_deepslice_inference", fake_deepslice)
    monkeypatch.setattr(TRACKER, "run_atlas_pose_onnx", fake_own)
    records, disagreement, runtime, prepared = TRACKER.prepare_and_run_pose_predictions(
        [(str(path), 0.0, False, False, points, None, True, mask)],
        TRACKER.POSE_ENGINE_WEIGHTED,
        0.7,
        216.0,
        queue.SimpleQueue(),
        threading.Event(),
    )

    record = records[0]
    deep_pose = np.asarray([-600.0, 10.0, -4.0])
    own_pose = np.asarray([-1000.0, -6.0, 8.0])
    expected = TRACKER.fuse_pose_predictions(np.stack([deep_pose, own_pose]), np.asarray([0.3, 0.7]))
    assert record["pose_ap_um_lr_deg_dv_deg"] == pytest.approx(expected)
    assert record["predicted_atlas_index"] == pytest.approx(216.0 - expected[0] / TRACKER.VOXEL_UM)
    assert set(record["component_predictions"]) == {
        TRACKER.POSE_ENGINE_DEEPSLICE,
        TRACKER.POSE_ENGINE_OWN_CNN,
    }
    assert record["fusion"]["own_cnn_weight"] == pytest.approx(0.7)
    assert np.asarray(record["initial_slice_to_atlas"]).shape == (3, 3)
    assert runtime["component_provenance"][TRACKER.POSE_ENGINE_OWN_CNN]["model_sha256"] == "own-hash"
    assert disagreement["slice_0000.png"]["ap_um"] == pytest.approx(400.0)
    assert np.array_equal(seen["mask"], mask.astype(bool))
    assert np.array_equal(prepared["slice_0000.png"]["brain_mask"], mask.astype(bool))


def test_own_runtime_preprocesses_the_supplied_mask_and_missing_model_is_clear(tmp_path, monkeypatch):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"mock")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": runtime._file_sha256(model),
                "preprocessing_version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
                "preprocessing_contract_sha256": runtime.atlas_pose_preprocessing_contract_sha256(),
            }
        ),
        encoding="utf-8",
    )
    image = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    mask = np.zeros_like(image, dtype=bool)
    mask[10:70, 20:80] = True
    seen = {}

    def fake_preprocess(received_image, received_mask):
        seen["image"] = received_image
        seen["mask"] = received_mask
        return np.full((3, 299, 299), 0.25, dtype=np.float32)

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, inputs):
            assert inputs[runtime.POSE_INPUT_NAME].shape == (1, 3, 299, 299)
            return [
                np.asarray([[-1400.0, 3.0, -2.0]], dtype=np.float32),
                np.asarray([-4.0], dtype=np.float32),
            ]

    monkeypatch.setattr(runtime, "preprocess_atlas_pose_image", fake_preprocess)
    monkeypatch.setattr(runtime, "_load_atlas_pose_session", lambda *_args: (Session(), None))
    prediction, _ = runtime.run_atlas_pose_candidate_onnx([image], [mask], model)

    assert prediction[0] == pytest.approx([-1400.0, 3.0, -2.0])
    assert seen["image"] is image
    assert seen["mask"] is mask
    with pytest.raises(RuntimeError, match="Own CNN model is unavailable"):
        runtime.run_atlas_pose_candidate_onnx([image], [mask], tmp_path / "missing.onnx")
    no_sidecar = tmp_path / "no_sidecar.onnx"
    no_sidecar.write_bytes(b"mock")
    with pytest.raises(RuntimeError, match="metadata is unavailable"):
        runtime.run_atlas_pose_candidate_onnx([image], [mask], no_sidecar)


def test_own_runtime_batches_many_slices(tmp_path, monkeypatch):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"mock")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": runtime._file_sha256(model),
                "preprocessing_version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
                "preprocessing_contract_sha256": runtime.atlas_pose_preprocessing_contract_sha256(),
            }
        ),
        encoding="utf-8",
    )
    batch_sizes = []

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, inputs):
            count = len(inputs[runtime.POSE_INPUT_NAME])
            batch_sizes.append(count)
            return [np.zeros((count, 3), np.float32), np.zeros(count, np.float32)]

    monkeypatch.setattr(runtime, "preprocess_atlas_pose_image", lambda *_: np.zeros((3, 299, 299), np.float32))
    monkeypatch.setattr(runtime, "_load_atlas_pose_session", lambda *_args: (Session(), None))
    images = [np.zeros((10, 10)) for _ in range(runtime.POSE_INFERENCE_BATCH_SIZE + 1)]
    masks = [np.ones((10, 10), bool) for _ in images]
    prediction, info = runtime.run_atlas_pose_candidate_onnx(images, masks, model)

    assert prediction.shape == (len(images), 3)
    assert batch_sizes == [runtime.POSE_INFERENCE_BATCH_SIZE, 1]
    assert info["orientation_inverted"] == [False] * len(images)


def test_own_runtime_requires_and_verifies_source_pinned_release_evidence(tmp_path, monkeypatch):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"approved-model")
    metadata = {
        "sha256": runtime._file_sha256(model),
        "preprocessing_version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
        "preprocessing_contract_sha256": runtime.atlas_pose_preprocessing_contract_sha256(),
        "source_sha256": {"trainer.py": "1" * 64},
        "manifest_sha256": {"train": "2" * 64},
        "registered_data": {"sha256": {"sections.jsonl": "3" * 64}},
        "atlas_data_sha256": {"annotation_25.nrrd": "4" * 64},
    }
    metadata_path = model.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    sealed_source = {
        "sections_sha256": "5" * 64,
        "datasets_sha256": "6" * 64,
        "provenance_sha256": "7" * 64,
        "downloads_sha256": "8" * 64,
        "registered_image_quality_manifest_sha256": "9" * 64,
    }
    metrics_path = tmp_path / "SEALED_metrics.json"
    metrics_path.write_text(
        json.dumps({"source": sealed_source, "evaluator_sha256": "a" * 64}),
        encoding="utf-8",
    )
    evidence = {
        "release_report_version": 2,
        "sealed": True,
        "benchmark_role": "final_release_gate",
        "release_approved": True,
        "promotion_ready": True,
        "model_sha256": runtime._file_sha256(model),
        "metadata_sha256": runtime._file_sha256(metadata_path),
        "preprocessing_contract_sha256": runtime.atlas_pose_preprocessing_contract_sha256(),
        "training_source_sha256": metadata["source_sha256"],
        "training_data_sha256": {
            "synthetic_manifests": metadata["manifest_sha256"],
            "registered_data": metadata["registered_data"]["sha256"],
            "atlas_data": metadata["atlas_data_sha256"],
        },
        "sealed_data_sha256": sealed_source,
        "sealed_metrics_sha256": runtime._file_sha256(metrics_path),
        "evaluator_sha256": "a" * 64,
        "quality_gate": {"all_gates_passed": True, "passed": {"mean_ap_um": True}},
        "deepslice_component_passed": {"ap_um": True, "lr_deg": True, "dv_deg": True},
    }
    evidence["release_integrity_sha256"] = runtime._canonical_json_sha256(evidence)
    evidence_path = tmp_path / "RELEASE_REPORT.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    image = np.zeros((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(RuntimeError, match="not release-approved"):
        runtime.run_atlas_pose_onnx([image], [mask], model)

    monkeypatch.setattr(runtime, "APPROVED_ATLAS_POSE_MODEL_SHA256", runtime._file_sha256(model))
    monkeypatch.setattr(
        runtime, "APPROVED_ATLAS_POSE_METADATA_SHA256", runtime._file_sha256(metadata_path)
    )
    monkeypatch.setattr(
        runtime, "APPROVED_ATLAS_POSE_EVIDENCE_SHA256", runtime._file_sha256(evidence_path)
    )

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, _inputs):
            return [np.zeros((1, 3), np.float32), np.zeros(1, np.float32)]

    monkeypatch.setattr(
        runtime, "preprocess_atlas_pose_image", lambda *_: np.zeros((3, 299, 299), np.float32)
    )
    monkeypatch.setattr(runtime, "_load_atlas_pose_session", lambda *_args: (Session(), None))
    _, diagnostics = runtime.run_atlas_pose_onnx([image], [mask], model)
    assert diagnostics["metadata_sha256"] == runtime._file_sha256(metadata_path)
    assert diagnostics["release_evidence_sha256"] == runtime._file_sha256(evidence_path)

    model.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum|source-pinned hashes"):
        runtime.run_atlas_pose_onnx([image], [mask], model)


def test_brain_mask_affine_preserves_geometry_after_roll_and_inversion():
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    target = np.zeros((120, 180), np.uint8)
    cv2.ellipse(target, (90, 66), (66, 38), 0, 0, 360, 1, -1)
    cv2.rectangle(target, (72, 18), (92, 45), 1, -1)

    center = ((target.shape[1] - 1) / 2, (target.shape[0] - 1) / 2)
    rotation = cv2.getRotationMatrix2D(center, 30.0, 1.0)
    corners = np.asarray(
        [[0, 0, 1], [target.shape[1] - 1, 0, 1], [0, target.shape[0] - 1, 1],
         [target.shape[1] - 1, target.shape[0] - 1, 1]],
        dtype=float,
    )
    rotated_corners = (rotation @ corners.T).T
    low = rotated_corners.min(axis=0)
    high = rotated_corners.max(axis=0)
    rotation[:, 2] -= low
    size = tuple(np.ceil(high - low + 1).astype(int))
    rolled = cv2.warpAffine(target, rotation, size, flags=cv2.INTER_NEAREST)
    inverted = np.rot90(target, 2).copy()

    for source, orientation_inverted in ((rolled, False), (inverted, True)):
        matrix = runtime.brain_mask_affine(source, target, orientation_inverted)
        mapped = cv2.warpPerspective(source, matrix, (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST)
        intersection = np.count_nonzero((mapped > 0) & (target > 0))
        union = np.count_nonzero((mapped > 0) | (target > 0))
        assert intersection / union > 0.97


def test_own_runtime_rejects_a_sidecar_checksum_mismatch(tmp_path, monkeypatch):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"mock")
    model.with_suffix(".json").write_text(json.dumps({"sha256": "wrong"}), encoding="utf-8")

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, _inputs):
            return [np.zeros((1, 3), dtype=np.float32)]

    monkeypatch.setattr(runtime, "preprocess_atlas_pose_image", lambda *_: np.zeros((3, 299, 299), np.float32))
    monkeypatch.setattr(runtime, "_load_atlas_pose_session", lambda *_args: (Session(), None))
    with pytest.raises(RuntimeError, match="checksum"):
        runtime.run_atlas_pose_candidate_onnx(
            [np.zeros((10, 10))], [np.ones((10, 10), bool)], model
        )


def test_own_runtime_rejects_a_preprocessing_contract_mismatch(tmp_path):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"mock")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": runtime._file_sha256(model),
                "preprocessing_version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
                "preprocessing_contract_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="preprocessing checksum"):
        runtime.run_atlas_pose_candidate_onnx(
            [np.zeros((10, 10))],
            [np.ones((10, 10), bool)],
            model,
        )


@pytest.mark.parametrize(("cancel_on_cpu_load", "expected_runs"), [(False, []), (True, ["CUDAExecutionProvider"])])
def test_own_runtime_cancels_after_session_construction(tmp_path, monkeypatch, cancel_on_cpu_load, expected_runs):
    runtime = sys.modules[TRACKER.run_atlas_pose_onnx.__module__]
    model = tmp_path / "atlas_pose.onnx"
    model.write_bytes(b"mock")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": runtime._file_sha256(model),
                "preprocessing_version": runtime.ATLAS_POSE_PREPROCESSING_VERSION,
                "preprocessing_contract_sha256": runtime.atlas_pose_preprocessing_contract_sha256(),
            }
        ),
        encoding="utf-8",
    )
    cancel = threading.Event()
    runs = []

    class Session:
        def __init__(self, provider):
            self.provider = provider

        def get_providers(self):
            return [self.provider]

        def run(self, _outputs, _inputs):
            runs.append(self.provider)
            if self.provider == "CUDAExecutionProvider":
                raise RuntimeError("GPU failure")
            raise AssertionError("Cancelled work must not start CPU inference")

    def fake_load(_path, _modified_ns, force_cpu):
        if force_cpu == cancel_on_cpu_load:
            cancel.set()
        return Session("CPUExecutionProvider" if force_cpu else "CUDAExecutionProvider"), None

    monkeypatch.setattr(runtime, "preprocess_atlas_pose_image", lambda *_: np.zeros((3, 299, 299), np.float32))
    monkeypatch.setattr(runtime, "_load_atlas_pose_session", fake_load)
    with pytest.raises(InterruptedError):
        runtime.run_atlas_pose_candidate_onnx(
            [np.zeros((10, 10))], [np.ones((10, 10), bool)], model, cancel
        )
    assert runs == expected_runs


def test_open_surface_arc_uses_tissue_detection_instead_of_a_filled_hull(tmp_path, monkeypatch):
    image = np.zeros((60, 80), dtype=np.uint8)
    image[10:50, 15:65] = 150
    path = tmp_path / "slice.png"
    assert cv2.imwrite(str(path), image)
    detected = np.zeros_like(image, dtype=bool)
    detected[12:48, 17:63] = True
    seen = {}

    def fake_detection(received):
        seen["image"] = received
        return detected

    monkeypatch.setattr(TRACKER, "automatic_brain_mask", fake_detection)
    with tempfile.TemporaryDirectory() as temporary_folder:
        _, crops, prepared = TRACKER.prepare_pose_inputs(
            [(str(path), 0.0, False, False, [(15.0, 10.0), (65.0, 10.0)], None, False, None)],
            temporary_folder,
            queue.SimpleQueue(),
            threading.Event(),
        )

    assert seen["image"].shape == image.shape
    assert np.array_equal(prepared["slice_0000.png"]["brain_mask"], detected)
    assert crops["slice_0000.png"]["crop_y1_oriented_display_px"] >= 48
    assert crops["slice_0000.png"]["crop_y1_oriented_display_px"] - crops["slice_0000.png"]["crop_y0_oriented_display_px"] > 30


@pytest.mark.parametrize(("prediction", "expected_edge"), [(-100.0, 0), (700.0, 527)])
def test_unconstrained_ap_candidates_clamp_out_of_atlas_predictions(prediction, expected_edge):
    candidates = TRACKER.ap_candidate_indices(prediction, 528, None, 4)
    assert candidates
    assert expected_edge in candidates


def test_ui_exposes_three_pose_modes_and_defaults_weight_to_twenty_percent(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        assert [window.pose_engine.itemText(index) for index in range(window.pose_engine.count())] == list(
            TRACKER.POSE_ENGINES
        )
        assert window.pose_engine.currentText() == TRACKER.POSE_ENGINE_DEEPSLICE
        assert window.own_cnn_weight.value() == 20
        assert not window.own_cnn_weight.isEnabled()
        window.pose_engine.setCurrentText(TRACKER.POSE_ENGINE_WEIGHTED)
        assert window.own_cnn_weight.isEnabled()
    finally:
        window.close()
        app.processEvents()
