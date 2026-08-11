import importlib.util
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_runtime_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


@pytest.mark.parametrize("failing_model", ["primary", "secondary"])
def test_directml_run_failure_retries_both_models_on_cpu_once(monkeypatch, failing_model):
    prediction = np.asarray(
        [[0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0]],
        dtype=np.float32,
    )

    class Session:
        def __init__(self, *, fail=False):
            self.fail = fail
            self.calls = 0

        def run(self, *_args):
            self.calls += 1
            if self.fail:
                raise RuntimeError("simulated DirectML failure")
            return [prediction]

    dml = {name: Session(fail=name == failing_model) for name in ("primary", "secondary")}
    cpu = {"primary": Session(), "secondary": Session()}
    loader_calls = []

    def load_sessions(force_cpu=False):
        loader_calls.append(force_cpu)
        return (cpu, {}, "CPUExecutionProvider", None) if force_cpu else (
            dml,
            {},
            "DmlExecutionProvider",
            None,
        )

    monkeypatch.setattr(TRACKER, "load_deepslice_onnx_sessions", load_sessions)
    monkeypatch.setattr(
        TRACKER,
        "preprocess_deepslice_images",
        lambda _paths: (np.zeros((1, 299, 299, 3), dtype=np.float32), [456], [320]),
    )

    records, _, _, _, runtime = TRACKER.run_deepslice_inference(
        ["slice.png"],
        False,
        queue.SimpleQueue(),
        threading.Event(),
    )

    assert loader_calls == [False, True]
    assert [dml["primary"].calls, dml["secondary"].calls] == (
        [1, 0] if failing_model == "primary" else [1, 1]
    )
    assert [cpu["primary"].calls, cpu["secondary"].calls] == [1, 1]
    assert runtime["backend"] == "ONNX Runtime CPU"
    assert runtime["gpu_fallback_reason"] == "DirectML inference failed: RuntimeError: simulated DirectML failure"
    assert records[0]["Filenames"] == "slice.png"


def test_cancellation_between_models_skips_secondary_inference(monkeypatch):
    cancel = threading.Event()

    class Session:
        def __init__(self, cancel_after_run=False):
            self.cancel_after_run = cancel_after_run
            self.calls = 0

        def run(self, *_args):
            self.calls += 1
            if self.cancel_after_run:
                cancel.set()
            return [np.zeros((1, 9), dtype=np.float32)]

    sessions = {"primary": Session(True), "secondary": Session()}
    monkeypatch.setattr(
        TRACKER,
        "load_deepslice_onnx_sessions",
        lambda *_args: (sessions, {}, "DmlExecutionProvider", None),
    )
    monkeypatch.setattr(
        TRACKER,
        "preprocess_deepslice_images",
        lambda _paths: (np.zeros((1, 299, 299, 3), dtype=np.float32), [456], [320]),
    )

    with pytest.raises(InterruptedError):
        TRACKER.run_deepslice_inference(["slice.png"], False, queue.SimpleQueue(), cancel)

    assert [sessions["primary"].calls, sessions["secondary"].calls] == [1, 0]


def test_validated_deepslice_runtime_recovers_known_atlas_planes():
    expected_indices = np.asarray([120.0, 216.0, 320.0])
    image_paths = [
        str(ROOT / "tests" / "data" / f"allen_average_coronal_{index}.png")
        for index in expected_indices.astype(int)
    ]

    records, _, hashes, _, runtime = TRACKER.run_deepslice_inference(
        image_paths, False, queue.SimpleQueue(), threading.Event()
    )
    by_filename = {
        record["Filenames"]: TRACKER.quicknii_to_tracker_alignment(record, TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML)[:3]
        for record in records
    }
    alignments = np.asarray([by_filename[Path(path).name] for path in image_paths])

    assert hashes == TRACKER.DEEPSLICE_ONNX_SHA256
    assert runtime["backend"] in {"ONNX Runtime DirectML", "ONNX Runtime CPU"}
    assert np.max(np.abs(alignments[:, 0] - expected_indices)) < 5.0
    assert np.max(np.abs(alignments[:, 1:])) < TRACKER.DEEPSLICE_REVIEW_TILT_DEG


def test_smart_selection_crop_flip_and_surface_fit_recover_known_oblique_plane():
    source_path = ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_source_fliph.png"
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    oriented, _ = TRACKER.transform_slice_image(source, 0.0, True, False)
    points, selection = TRACKER.smart_brain_surface_selection(
        oriented,
        [(330.0, 240.0), (220.0, 240.0), (440.0, 240.0)],
        [(20.0, 20.0), (680.0, 480.0)],
        50.0,
        50,
    )
    y, x = np.nonzero(selection)
    crop = TRACKER.surface_crop_bounds(
        [(float(x.min()), float(y.min())), (float(x.max()), float(y.max()))],
        selection.shape,
        0.04,
    )
    messages = queue.SimpleQueue()
    cancel = threading.Event()
    records, _, _, _, runtime_info = TRACKER.prepare_and_run_deepslice(
        [(str(source_path), 0.0, True, False, points, crop, True)],
        False,
        messages,
        cancel,
    )
    mask = cv2.imread(
        str(ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    annotation = np.broadcast_to(mask, TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML)
    prepared, shared_tilt = TRACKER.solve_deepslice_alignment(
        records,
        {"slice_0000.png": 0},
        TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML,
        annotation,
        {0: points},
        None,
        [],
        runtime_info,
        global_alignment=False,
    )

    _, atlas_index, tilt_lr, tilt_dv, matrix, _, _, diagnostics = prepared[0]
    assert shared_tilt is None
    assert abs(atlas_index - 280) < 5
    assert abs(tilt_lr - 6.0) < 5.0
    assert abs(tilt_dv + 4.0) < 5.0
    assert matrix[0, 0] > 0.0
    assert diagnostics["surface_rms_after_atlas_px"] < 3.0
