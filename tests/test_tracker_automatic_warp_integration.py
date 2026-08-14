import importlib.util
import os
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_automatic_warp_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def identity_result(valid_mask=None):
    yy, xx = np.mgrid[: TRACKER.DENSE_REGISTRATION_SHAPE[0], : TRACKER.DENSE_REGISTRATION_SHAPE[1]].astype(
        np.float32
    )
    identity = np.stack((xx, yy), axis=-1)
    return {
        "atlas_to_affine_xy": identity,
        "affine_to_atlas_xy": identity,
        "valid_atlas_mask": (
            np.ones(TRACKER.DENSE_REGISTRATION_SHAPE, dtype=bool)
            if valid_mask is None
            else valid_mask
        ),
        "metadata": {
            "method": "dense-registration-onnx-v2",
            "model_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "provider": "DmlExecutionProvider",
            "preprocessing_contract": "test-v2",
        },
    }


class ControlledExecutor:
    def __init__(self):
        self.future = Future()
        self.arguments = None
        self.keywords = None

    def submit(self, function, *arguments, **keywords):
        self.arguments = (function, arguments)
        self.keywords = keywords
        return self.future

    def shutdown(self, **_):
        return None


def process_until(app, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def make_window_session(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    image_path = tmp_path / "slice.tif"
    yy, xx = np.mgrid[: TRACKER.DENSE_REGISTRATION_SHAPE[0], : TRACKER.DENSE_REGISTRATION_SHAPE[1]]
    image = ((3 * xx + 5 * yy) % 256).astype(np.uint8)
    tifffile.imwrite(image_path, image)
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(image_path)
    session = window.sessions[0]
    window.atlas_volume = np.stack((image, image), axis=0)
    window.annotation_volume = np.ones_like(window.atlas_volume, dtype=np.uint16)
    window.atlas_file_hashes = {"average_template_25.nrrd": "c" * 64, "annotation_25.nrrd": "d" * 64}
    window.current_atlas_image = image
    session.atlas_index = 0
    session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3), image.shape, TRACKER.DENSE_REGISTRATION_SHAPE
    )
    return app, window, session, image


def test_dense_canvases_use_nonidentity_affine_raw_grayscale_and_nearest_binary_mask():
    raw = np.arange(160 * 228, dtype=np.uint8).reshape(160, 228)
    mask = np.zeros_like(raw, dtype=bool)
    mask[25:120, 35:180] = True
    raw[5, 5] = 251
    atlas = np.arange(np.prod(TRACKER.DENSE_REGISTRATION_SHAPE), dtype=np.float32).reshape(
        TRACKER.DENSE_REGISTRATION_SHAPE
    )
    atlas_mask = np.ones(TRACKER.DENSE_REGISTRATION_SHAPE, dtype=bool)
    homography = np.diag([2.0, 2.0, 1.0])

    atlas_out, atlas_mask_out, slice_out, mask_out = TRACKER.dense_registration_canvases(
        raw, mask, atlas, atlas_mask, homography
    )

    expected_mask = cv2.warpPerspective(
        mask.astype(np.uint8),
        homography,
        (TRACKER.DENSE_REGISTRATION_SHAPE[1], TRACKER.DENSE_REGISTRATION_SHAPE[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    assert np.array_equal(atlas_out, atlas)
    assert np.array_equal(atlas_mask_out, atlas_mask)
    assert slice_out.dtype == np.uint8
    assert mask_out.dtype == bool
    assert np.array_equal(mask_out, expected_mask)
    assert slice_out[10, 10] == 251
    assert not mask_out[10, 10]


def test_registration_mask_priority_is_smart_then_outline_then_automatic(monkeypatch):
    image = np.zeros((20, 30), dtype=np.uint8)
    smart = np.zeros_like(image, dtype=bool)
    smart[1:4, 1:4] = True
    outline = [(10.0, 5.0), (25.0, 5.0), (25.0, 15.0), (10.0, 15.0)]
    assert np.array_equal(TRACKER.registration_brain_mask(image, outline, True, smart), smart)

    outlined = TRACKER.registration_brain_mask(image, outline, True, None)
    assert outlined[10, 15]
    assert not outlined[2, 2]

    automatic = np.zeros_like(image, dtype=bool)
    automatic[7:9, 7:9] = True
    monkeypatch.setattr(TRACKER, "automatic_brain_mask", lambda _: automatic)
    assert np.array_equal(TRACKER.registration_brain_mask(image, [], False, None), automatic)


def test_background_warp_uses_weight_image_not_brightness_curve_and_is_nonblocking(tmp_path, monkeypatch):
    app, window, session, image = make_window_session(tmp_path)
    model = tmp_path / "dense_registration.onnx"
    metadata = tmp_path / "dense_registration.metadata.json"
    model.write_bytes(b"model")
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_MODEL_PATH", model)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_METADATA_PATH", metadata)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_MODEL_SHA256", "a" * 64)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_METADATA_SHA256", "b" * 64)
    session.brain_brush_selection_mask = np.ones_like(image, dtype=bool)
    raw_for_model = session.weight_image.copy()
    session.curve_points = [(0.0, 255.0), (255.0, 0.0)]
    session.adjusted = 255 - raw_for_model
    session.rotated = session.adjusted.copy()
    window.alignment_executor.shutdown(wait=False)
    executor = ControlledExecutor()
    window.alignment_executor = executor
    runtime_calls = []

    def fake_runtime(
        model_path,
        atlas_image,
        atlas_mask,
        slice_image,
        slice_mask,
        *,
        expected_model_sha256,
        expected_metadata_sha256,
        metadata_path,
    ):
        runtime_calls.append(
            (
                model_path,
                atlas_image,
                atlas_mask,
                slice_image,
                slice_mask,
                expected_model_sha256,
                expected_metadata_sha256,
                metadata_path,
            )
        )
        return identity_result()

    monkeypatch.setattr(TRACKER, "run_dense_registration", fake_runtime)
    monkeypatch.setattr(
        TRACKER,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("GUI thread must not hash the slice file")),
    )
    try:
        started = time.monotonic()
        window.apply_automatic_anatomical_warp()
        assert time.monotonic() - started < 0.25
        assert window.auto_alignment_busy
        function, arguments = executor.arguments
        assert function is fake_runtime
        assert np.array_equal(arguments[3], raw_for_model)
        assert np.array_equal(arguments[4], session.brain_brush_selection_mask)
        assert executor.keywords == {
            "expected_model_sha256": "a" * 64,
            "expected_metadata_sha256": "b" * 64,
            "metadata_path": metadata,
        }
        executor.future.set_result(function(*arguments, **executor.keywords))
        process_until(app, lambda: not window.auto_alignment_busy)
        assert len(runtime_calls) == 1
        assert session.slice_atlas_transform.nonlinear
        assert session.auto_alignment_diagnostics["coordinate_registration"]["status"] == "applied"
        assert session.auto_alignment_diagnostics["coordinate_registration"]["kind"] == "automatic"
    finally:
        window.close()
        app.processEvents()


def test_mapping_is_disabled_while_alignment_runs_and_rejects_pending_coordinate_registration(
    tmp_path, monkeypatch
):
    app, window, session, _ = make_window_session(tmp_path)
    warnings = []
    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    session.probe_traces["imec0"] = TRACKER.ProbeTrace(
        slice_points=[(10.0, 10.0)],
        atlas_points=[(10.0, 10.0)],
        volume_points=[[0.0, 10.0, 10.0]],
    )
    try:
        window.auto_alignment_busy = True
        window._probe_selection_changed()
        assert not window.map_btn.isEnabled()
        window.map_channels_units()
        assert warnings[-1][0] == "Alignment still running"

        window.auto_alignment_busy = False
        (tmp_path / "channels.csv").write_text("channel\n", encoding="utf-8")
        (tmp_path / "units.csv").write_text("unit\n", encoding="utf-8")
        window.run_folder.setText(str(tmp_path))
        session.auto_alignment_diagnostics = {
            "coordinate_registration": {
                "kind": "unselected",
                "status": "pending",
                "reason": "section matched",
            }
        }
        window._probe_selection_changed()
        assert window.map_btn.isEnabled()
        window.map_channels_units()
        assert warnings[-1][0] == "Coordinate registration required"
        assert session.name in warnings[-1][1]
    finally:
        window.close()
        app.processEvents()


def test_manual_pose_summary_exposes_required_anatomical_warp_step(tmp_path):
    app, window, session, _ = make_window_session(tmp_path)
    session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3), session.rotated.shape, TRACKER.DENSE_REGISTRATION_SHAPE
    )
    session.auto_alignment_engine = None
    session.auto_alignment_diagnostics = {
        "anatomical_registration": {"status": "pending", "reason": "atlas pose changed"}
    }
    try:
        window._refresh_point_counts()
        summary = window.alignment_summary.text()
        assert summary.startswith("Manual atlas pose:")
        assert "automatic anatomical warp pending" in summary
        assert "before mapping" in summary
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize("outcome", ["cancel", "atlas-stale", "image-stale"])
def test_cancel_or_stale_background_result_keeps_previous_transform(tmp_path, monkeypatch, outcome):
    app, window, session, image = make_window_session(tmp_path)
    model = tmp_path / "dense_registration.onnx"
    metadata = tmp_path / "dense_registration.metadata.json"
    model.write_bytes(b"model")
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_MODEL_PATH", model)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_METADATA_PATH", metadata)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_MODEL_SHA256", "a" * 64)
    monkeypatch.setattr(TRACKER, "DENSE_REGISTRATION_METADATA_SHA256", "b" * 64)
    monkeypatch.setattr(TRACKER.QtWidgets.QMessageBox, "critical", lambda *args: None)
    session.brain_brush_selection_mask = np.ones_like(image, dtype=bool)
    previous = session.slice_atlas_transform
    window.alignment_executor.shutdown(wait=False)
    executor = ControlledExecutor()
    window.alignment_executor = executor
    try:
        window.apply_automatic_anatomical_warp()
        if outcome == "cancel":
            window._alignment_cancel_event.set()
        elif outcome == "atlas-stale":
            session.atlas_index = 1
        else:
            session.weight_image[0, 0] ^= np.uint8(1)
        executor.future.set_result(identity_result())
        process_until(app, lambda: not window.auto_alignment_busy)
        assert session.slice_atlas_transform is previous
    finally:
        window.close()
        app.processEvents()


def test_atomic_install_rolls_back_invalid_probe_and_remaps_every_probe(tmp_path):
    app, window, session, image = make_window_session(tmp_path)
    try:
        session.probe_traces = {
            "imec0": TRACKER.ProbeTrace(slice_points=[(10.0, 10.0)]),
            "imec1": TRACKER.ProbeTrace(slice_points=[(20.0, 30.0)]),
        }
        previous = session.slice_atlas_transform
        invalid = np.ones(TRACKER.DENSE_REGISTRATION_SHAPE, dtype=bool)
        invalid[10, 10] = False
        with pytest.raises(RuntimeError, match="outside valid registered tissue"):
            window._apply_automatic_anatomical_warp_result(
                session,
                np.eye(3),
                image.shape,
                "f" * 64,
                "e" * 64,
                identity_result(invalid),
            )
        assert session.slice_atlas_transform is previous
        assert not session.probe_traces["imec0"].atlas_points

        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float32)
        result = identity_result()
        result["atlas_to_affine_xy"] = np.stack((xx - 2.0, yy - 3.0), axis=-1)
        result["affine_to_atlas_xy"] = np.stack((xx + 2.0, yy + 3.0), axis=-1)
        window._apply_automatic_anatomical_warp_result(
            session,
            np.eye(3),
            image.shape,
            "f" * 64,
            "e" * 64,
            result,
        )
        assert np.allclose(session.probe_traces["imec0"].atlas_points, [(12.0, 13.0)])
        assert np.allclose(session.probe_traces["imec1"].atlas_points, [(22.0, 33.0)])
        assert all(trace.volume_points for trace in session.probe_traces.values())
    finally:
        window.close()
        app.processEvents()


def test_automatic_warp_always_rebuilds_mask_affine_then_refines_with_surface(monkeypatch):
    image = np.zeros(TRACKER.DENSE_REGISTRATION_SHAPE, dtype=np.uint8)
    slice_mask = np.ones_like(image, dtype=bool)
    atlas_mask = np.ones_like(image, dtype=bool)
    surface_points = [
        (30.0, 30.0),
        (100.0, 20.0),
        (200.0, 25.0),
        (300.0, 35.0),
        (350.0, 150.0),
        (300.0, 280.0),
        (150.0, 290.0),
        (40.0, 250.0),
    ]
    initial = np.array([[0.9, 0.0, 3.0], [0.0, 0.9, 5.0], [0.0, 0.0, 1.0]])
    fitted = np.array([[1.1, 0.0, 7.0], [0.0, 1.1, 9.0], [0.0, 0.0, 1.0]])
    calls = []

    def fake_fit(initial, points, mask):
        calls.append((initial.copy(), list(points), mask.copy()))
        return fitted, {}

    monkeypatch.setattr(TRACKER, "brain_mask_affine", lambda source, target: initial.copy())
    monkeypatch.setattr(TRACKER, "fit_surface_scale_translation", fake_fit)

    base = TRACKER.TrajectoryTrackerWindow._automatic_warp_base_affine(
        slice_mask,
        atlas_mask,
        surface_points,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], initial)
    assert calls[0][1] == surface_points
    assert np.array_equal(calls[0][2], atlas_mask)
    assert np.array_equal(base, fitted)


def test_flip_recomputes_probe_coordinates_once(tmp_path, monkeypatch):
    app, window, session, image = make_window_session(tmp_path)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float32)
    identity = np.stack((xx, yy), axis=-1)
    session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3), image.shape, image.shape, identity, identity, np.ones_like(image, dtype=bool)
    )
    calls = 0
    original = window._recompute_probe_points_from_slice_points

    def count_recompute(current):
        nonlocal calls
        calls += 1
        return original(current)

    monkeypatch.setattr(window, "_recompute_probe_points_from_slice_points", count_recompute)
    try:
        window._apply_slice_geometry(0.0, True, False)
        assert calls == 1
    finally:
        window.close()
        app.processEvents()


def test_coordinate_warp_invalidation_matrix(tmp_path):
    app, window, session, image = make_window_session(tmp_path)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float32)
    identity = np.stack((xx, yy), axis=-1)

    def attach_dense():
        session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
            np.eye(3), image.shape, image.shape, identity, identity, np.ones_like(image, dtype=bool)
        )
        session.auto_alignment_diagnostics = {
            "coordinate_registration": {"kind": "automatic", "status": "applied"}
        }

    def assert_pending():
        registration = window._coordinate_registration(session)
        assert registration["status"] == "pending"

    try:
        attach_dense()
        window._section_changed(1)
        assert not session.slice_atlas_transform.nonlinear
        assert_pending()

        attach_dense()
        window._atlas_tilt_changed()
        assert not session.slice_atlas_transform.nonlinear
        assert_pending()

        attach_dense()
        window.plane_box.blockSignals(True)
        window.plane_box.setCurrentText("horizontal")
        window.plane_box.blockSignals(False)
        window._plane_changed()
        assert session.slice_atlas_transform is None
        assert_pending()

        window.plane_box.blockSignals(True)
        window.plane_box.setCurrentText("coronal")
        window.plane_box.blockSignals(False)
        window.section_scroll.blockSignals(True)
        window.section_scroll.setRange(0, 1)
        window.section_scroll.setValue(0)
        window.section_scroll.blockSignals(False)
        session.atlas_plane = "coronal"
        session.atlas_index = 0
        window.current_atlas_image = image
        session.probe_traces = {"imec0": TRACKER.ProbeTrace(slice_points=[(10.0, 10.0)])}
        attach_dense()
        window._apply_slice_geometry(0.0, True, False)
        assert not session.slice_atlas_transform.nonlinear
        assert np.array_equal(session.slice_atlas_transform.display_to_affine_atlas_h, np.eye(3))
        assert np.allclose(
            session.probe_traces["imec0"].atlas_points,
            [(image.shape[1] - 1 - 10.0, 10.0)],
        )
        assert_pending()

        attach_dense()
        window._invalidate_auto_alignment_after_surface_edit(session)
        assert not session.slice_atlas_transform.nonlinear
        assert_pending()

        attach_dense()
        window._invalidate_transform_after_landmark_edit(session)
        assert not session.slice_atlas_transform.nonlinear
        assert window._coordinate_registration(session) == {
            "kind": "landmark",
            "status": "pending",
            "reason": "Landmarks changed; apply the landmark warp again",
        }
    finally:
        window.close()
        app.processEvents()
