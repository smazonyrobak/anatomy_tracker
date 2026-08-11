import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_alignment_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def test_unspecified_ap_search_is_local_not_whole_atlas():
    assert TRACKER.ap_candidate_indices(201.25, 528, None, 4) == [193, 197, 201, 205, 209, 210]


def test_reversed_ap_bounds_define_the_actual_candidate_grid():
    assert TRACKER.ap_candidate_indices(90.0, 528, (160, 100), 25) == [100, 125, 150, 160]


def test_partial_anterior_to_posterior_order_uses_lattice_and_leaves_unchecked_best():
    lattices = {
        0: {100: (0.1, {}), 110: (0.0, {}), 120: (0.2, {})},
        1: {100: (0.0, {}), 110: (0.5, {}), 120: (0.6, {})},
        2: {100: (0.0, {}), 110: (0.1, {}), 120: (0.2, {})},
    }

    assignments, _ = TRACKER.solve_ordered_lattice(lattices, [0, 2])

    assert assignments[0] < assignments[2]
    assert assignments[1] == 100


def test_order_rejects_a_range_without_enough_distinct_sections():
    lattices = {
        index: {100: (0.0, {}), 101: (0.0, {})}
        for index in range(3)
    }

    with pytest.raises(ValueError, match="too narrow"):
        TRACKER.solve_ordered_lattice(lattices, [0, 1, 2])


def _ellipse_case(missing_arc: bool = False):
    center = np.array([140.0, 110.0])
    axes = np.array([55.0, 38.0])
    atlas_mask = np.zeros((220, 260), dtype=np.uint8)
    cv2.ellipse(atlas_mask, (140, 110), (55, 38), 0, 0, 360, 1, -1)

    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = center + axes * np.column_stack([np.cos(angles), np.sin(angles)])
    if missing_arc:
        points = points[(angles > 0.6) & (angles < 5.4)]

    scale = 0.85
    translation = np.array([8.0, -6.0])
    initial = np.array(
        [
            [scale, 0.0, center[0] + translation[0] - scale * center[0]],
            [0.0, scale, center[1] + translation[1] - scale * center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return points, atlas_mask, initial


def test_surface_fit_recovers_synthetic_scale_and_translation():
    points, atlas_mask, initial = _ellipse_case()

    fitted, diagnostics = TRACKER.fit_surface_scale_translation(
        initial,
        [tuple(point) for point in points],
        atlas_mask,
    )

    assert diagnostics["scale"] == pytest.approx(1.0 / 0.85, abs=0.003)
    assert diagnostics["translation_x_atlas_px"] == pytest.approx(-8.0, abs=0.1)
    assert diagnostics["translation_y_atlas_px"] == pytest.approx(6.0, abs=0.1)
    assert fitted == pytest.approx(np.eye(3), abs=0.08)
    assert diagnostics["rms_after_atlas_px"] < 0.5
    assert diagnostics["rms_after_atlas_px"] < diagnostics["rms_before_atlas_px"] / 20.0


def test_surface_fit_uses_only_trusted_points_when_outline_has_a_missing_arc():
    points, atlas_mask, initial = _ellipse_case(missing_arc=True)

    fitted, diagnostics = TRACKER.fit_surface_scale_translation(
        initial,
        [tuple(point) for point in points],
        atlas_mask,
    )

    assert diagnostics["trusted_point_count"] == len(points)
    assert fitted == pytest.approx(np.eye(3), abs=0.21)
    assert diagnostics["rms_after_atlas_px"] < 0.5


def test_surface_fit_recovers_large_translation_from_multistart_initialization():
    atlas_mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.circle(atlas_mask, (110, 110), 60, 1, -1)
    angles = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    points = np.column_stack([80 + 48 * np.cos(angles), 95 + 48 * np.sin(angles)])

    fitted, diagnostics = TRACKER.fit_surface_scale_translation(
        np.eye(3),
        [tuple(point) for point in points],
        atlas_mask,
    )

    assert fitted[0, 0] == pytest.approx(1.25, abs=0.02)
    assert fitted[1, 1] == pytest.approx(1.25, abs=0.02)
    assert diagnostics["rms_after_atlas_px"] < 1.0
    assert diagnostics["rms_after_atlas_px"] < diagnostics["rms_before_atlas_px"] / 10.0


def test_surface_crop_isolates_the_selected_object_with_a_margin():
    crop = TRACKER.surface_crop_bounds(
        [(100.0, 80.0), (300.0, 240.0)],
        (500, 700),
        0.1,
    )

    assert crop == (80, 64, 321, 257)


def test_slice_points_follow_rotation_and_both_flips_exactly():
    image_shape = (120, 200)
    points = [(10.0, 20.0), (80.0, 60.0), (180.0, 100.0)]
    output_shape, unflipped = TRACKER.slice_geometry_matrix(image_shape, 27.0, False, False)
    flipped_shape, flipped = TRACKER.slice_geometry_matrix(image_shape, 27.0, True, True)

    unflipped_points = np.asarray(TRACKER.transform_points(points, unflipped))
    flipped_points = np.asarray(TRACKER.transform_points(points, flipped))
    expected = np.column_stack(
        [
            output_shape[1] - 1.0 - unflipped_points[:, 0],
            output_shape[0] - 1.0 - unflipped_points[:, 1],
        ]
    )

    assert flipped_shape == output_shape
    assert flipped_points == pytest.approx(expected)
    recovered = TRACKER.transform_points(
        [tuple(point) for point in flipped_points],
        np.linalg.inv(flipped),
    )
    assert np.asarray(recovered) == pytest.approx(np.asarray(points))


def test_offscreen_ui_requires_surfaces_and_global_alignment_uses_only_outlined_slices(tmp_path, monkeypatch):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        window.atlas_volume = np.zeros((8, 32, 40), dtype=np.uint8)
        window.annotation_volume = np.ones((8, 32, 40), dtype=np.uint8)
        image = np.zeros((32, 40), dtype=np.uint8)
        window.sessions = [
            TRACKER.SliceSession(name="outlined-a", raw_display=image, adjusted=image, rotated=image, weight_image=image),
            TRACKER.SliceSession(name="unoutlined", raw_display=image, adjusted=image, rotated=image, weight_image=image),
            TRACKER.SliceSession(name="outlined-b", raw_display=image, adjusted=image, rotated=image, weight_image=image),
        ]
        window.current_session_index = 0
        for session in window.sessions:
            window.slice_list.addItem(session.name)

        window._refresh_point_counts()
        assert not window.auto_align_btn.isEnabled()
        assert not window.auto_align_all_btn.isEnabled()

        outline = [(float(x), float(y)) for x, y in cv2.ellipse2Poly((20, 16), (12, 8), 0, 0, 315, 45)]
        assert len(outline) == 8
        window.sessions[0].brain_outline_points = outline
        window.sessions[2].brain_outline_points = outline
        window._refresh_point_counts()

        assert window.auto_align_btn.isEnabled()
        assert window.auto_align_all_btn.isEnabled()
        assert [index for index, _ in window._outlined_auto_sessions()] == [0, 2]
        assert [window.auto_slice_order.item(row).text() for row in range(window.auto_slice_order.count())] == [
            "outlined-a",
            "outlined-b",
        ]
        first_order_item = window.auto_slice_order.item(0)
        first_order_item.setCheckState(TRACKER.QtCore.Qt.CheckState.Checked)
        assert first_order_item.data(TRACKER.QtCore.Qt.ItemDataRole.UserRole) == 0

        called = {}

        def capture_start(indices, *, global_alignment):
            called["indices"] = indices
            called["global_alignment"] = global_alignment

        monkeypatch.setattr(window, "_start_deepslice_alignment", capture_start)
        window.auto_align_all_slices()
        assert called == {"indices": [0, 2], "global_alignment": True}
    finally:
        window.close()
        app.processEvents()


def _quicknii_record(filename, shape, atlas_index, tilt_lr, tilt_dv):
    ap_size, dv_size, ml_size = shape
    lr_slope = np.tan(np.deg2rad(tilt_lr))
    dv_slope = np.tan(np.deg2rad(tilt_dv))
    center_ml = (ml_size - 1) / 2.0
    center_dv = (dv_size - 1) / 2.0
    record = {
        "Filenames": filename,
        "ox": float(ml_size),
        "oy": ap_size - (atlas_index - lr_slope * center_ml - dv_slope * center_dv),
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
    record["shared_angle_ouv"] = None
    return record


def test_offscreen_result_application_stores_one_exact_shared_tilt(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (8, 32, 40)
        window.atlas_volume = np.zeros(shape, dtype=np.uint8)
        window.annotation_volume = np.ones(shape, dtype=np.uint8)
        window.bregma_voxel = np.array([3.0, 0.0, 0.0])
        image = np.zeros(shape[1:], dtype=np.uint8)
        surface = []
        for x in np.linspace(1.0, 38.0, 5):
            surface.extend([(float(x), 1.0), (float(x), 30.0)])
        for y in np.linspace(6.0, 25.0, 4):
            surface.extend([(1.0, float(y)), (38.0, float(y))])
        window.sessions = [
            TRACKER.SliceSession(
                name=name,
                raw_display=image,
                adjusted=image,
                rotated=image,
                weight_image=image,
                brain_outline_points=surface,
            )
            for name in ("a", "b")
        ]
        window.current_session_index = 0
        for session in window.sessions:
            window.slice_list.addItem(session.name)

        records = [
            _quicknii_record("slice_0000.png", shape, 2.0, 2.0, -1.0),
            _quicknii_record("slice_0001.png", shape, 5.0, 4.0, -3.0),
        ]
        for record in records:
            record["shared_angle_ouv"] = record["raw_ensemble_ouv"].copy()
        disagreement = {
            record["Filenames"]: {"ap_um": 1.0, "lr_deg": 1.0, "dv_deg": 1.0}
            for record in records
        }
        runtime_info = {
            "backend": "test",
            "device": "test",
            "alignment_run_id": "global-test",
            "preintegration_tilt_spread_deg": [2.0, 2.0],
        }
        shared_tilt = (3.0, -2.0)
        prepared = []
        for index, (record, atlas_index) in enumerate(zip(records, (2, 5))):
            diagnostics = {
                "raw_deepslice_ap_index": float(atlas_index),
                "refined_ap_index": float(atlas_index),
                "ap_search_shift_index": 0.0,
                "ap_search_bounds_index": None,
                "alignment_batch_session_indices": [0, 1],
                "alignment_run_id": "global-test",
                "surface_scale": 1.0,
                "surface_rms_after_atlas_px": 0.0,
            }
            prepared.append(
                (index, atlas_index, *shared_tilt, np.eye(3), np.eye(3), record, diagnostics)
            )
        window._apply_deepslice_results(
            prepared,
            "1.2.8",
            {},
            disagreement,
            runtime_info,
            None,
            [],
            global_alignment=True,
            shared_tilt=shared_tilt,
        )

        assert window.sessions[0].atlas_tilt_ml_deg == window.sessions[1].atlas_tilt_ml_deg
        assert window.sessions[0].atlas_tilt_dv_deg == window.sessions[1].atlas_tilt_dv_deg
        assert window.sessions[0].atlas_tilt_ml_deg == pytest.approx(3.0, abs=1e-9)
        assert window.sessions[0].atlas_tilt_dv_deg == pytest.approx(-2.0, abs=1e-9)
        assert [session.atlas_index for session in window.sessions] == [2, 5]
        assert all(session.auto_alignment_global for session in window.sessions)
        assert all(session.auto_alignment_scope == "global" for session in window.sessions)
        assert all(session.auto_alignment_run_id == "global-test" for session in window.sessions)
        assert [session.deepslice_shared_angle_ouv for session in window.sessions] == [
            record["shared_angle_ouv"] for record in records
        ]

        peer_transform = window.sessions[1].slice_to_atlas_x
        raw_ouv = window.sessions[0].deepslice_raw_ensemble_ouv.copy()
        window._section_changed(4)
        assert window.sessions[0].auto_alignment_scope == "manual-refined"
        assert window.sessions[0].manual_refined_from_run_id == "global-test"
        assert window.sessions[0].deepslice_raw_ensemble_ouv == raw_ouv
        assert window.sessions[1].auto_alignment_scope == "global"
        assert window.sessions[1].slice_to_atlas_x is peer_transform

        window._invalidate_auto_alignment_after_surface_edit(window.sessions[0])
        assert window.sessions[0].slice_to_atlas_x is None
        assert window.sessions[1].slice_to_atlas_x is peer_transform
        assert window.sessions[1].auto_alignment_diagnostics["alignment_run_stale"]
    finally:
        window.close()
        app.processEvents()


def test_auto_alignment_runs_without_blocking_slice_browsing(tmp_path, monkeypatch):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    release_worker = threading.Event()
    release_close_worker = threading.Event()
    try:
        shape = (8, 32, 40)
        image = np.zeros(shape[1:], dtype=np.uint8)
        outline = [(float(x), float(y)) for x, y in cv2.ellipse2Poly((20, 16), (12, 8), 0, 0, 315, 45)]
        window.atlas_volume = np.zeros(shape, dtype=np.uint8)
        window.annotation_volume = np.ones(shape, dtype=np.uint8)
        window.sessions = [
            TRACKER.SliceSession(
                name=name,
                path=str(tmp_path / f"{name}.tif"),
                raw_display=image,
                adjusted=image,
                rotated=image,
                weight_image=image,
                brain_outline_points=outline if index == 0 else [],
            )
            for index, name in enumerate(("target", "browse"))
        ]
        for session in window.sessions:
            window.slice_list.addItem(session.name)
        window.current_session_index = 0

        prediction = _quicknii_record("slice_0000.png", shape, 3.0, 1.0, -1.0)
        worker_started = threading.Event()

        def fake_worker(*_args):
            worker_started.set()
            release_worker.wait(3.0)
            diagnostics = {
                "raw_deepslice_ap_index": 3.0,
                "refined_ap_index": 3.0,
                "ap_search_shift_index": 0.0,
                "ap_search_bounds_index": None,
                "order_constraint_applied": False,
                "alignment_batch_session_indices": [0],
                "order_constraint_session_indices": [],
                "runtime_backend": "test",
                "runtime_device": "test",
                "alignment_run_id": "test-run",
                "preintegration_tilt_spread_deg": [0.0, 0.0],
                "surface_scale": 1.0,
                "surface_rms_after_atlas_px": 0.0,
            }
            prepared = [(0, 3, 1.0, -1.0, np.eye(3), np.eye(3), prediction, diagnostics)]
            disagreement = {"slice_0000.png": {"ap_um": 0.0, "lr_deg": 0.0, "dv_deg": 0.0}}
            return "1.2.8", {}, disagreement, {"device": "test"}, prepared, None

        monkeypatch.setattr(TRACKER, "prepare_run_and_solve_deepslice", fake_worker)
        started = time.perf_counter()
        window._start_deepslice_alignment([0], global_alignment=False)
        assert time.perf_counter() - started < 0.10
        assert window.auto_alignment_busy
        assert worker_started.wait(2.0)

        window._switch_slice(1)
        assert window.current_session_index == 1
        release_worker.set()
        deadline = time.perf_counter() + 3.0
        while window.auto_alignment_busy and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert not window.auto_alignment_busy
        assert window.current_session_index == 1
        assert window.sessions[0].auto_alignment_engine == "DeepSlice"
        assert window._deepslice_timer is None
        assert window._deepslice_progress is None

        close_worker_started = threading.Event()

        def held_worker(*args):
            close_worker_started.set()
            release_close_worker.wait(3.0)
            if args[-1].is_set():
                raise InterruptedError

        monkeypatch.setattr(TRACKER, "prepare_run_and_solve_deepslice", held_worker)
        window._start_deepslice_alignment([0], global_alignment=False)
        assert close_worker_started.wait(2.0)
        cancel_event = window._deepslice_cancel_event
        window.close()
        app.processEvents()

        assert cancel_event.is_set()
        assert not window.auto_alignment_busy
        assert window._deepslice_cancel_event is None
        assert window._deepslice_timer is None
        assert window._deepslice_progress is None
    finally:
        release_worker.set()
        release_close_worker.set()
        window.close()
        window.deepslice_executor.shutdown(wait=True, cancel_futures=True)
        app.processEvents()


def test_surface_edits_preserve_authoritative_crop_and_alignment_state(tmp_path, monkeypatch):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        image = np.zeros((32, 40), dtype=np.uint8)
        outline = [(float(x), float(y)) for x, y in cv2.ellipse2Poly((20, 16), (12, 8), 0, 0, 315, 45)]
        mask = np.zeros_like(image)
        mask[8:24, 10:30] = 1
        session = TRACKER.SliceSession(
            name="edited",
            path=str(tmp_path / "edited.tif"),
            raw_display=image,
            adjusted=image,
            rotated=image,
            weight_image=image,
            brain_outline_points=outline,
            brain_outline_closed=True,
            brain_brush_strokes=[(False, [(20.0, 16.0)])],
            brain_brush_selection_mask=mask,
        )
        window.atlas_volume = np.zeros((8, 32, 40), dtype=np.uint8)
        window.annotation_volume = np.ones((8, 32, 40), dtype=np.uint8)
        window.sessions = [session]
        window.slice_list.addItem(session.name)
        window.current_session_index = 0
        window._detach_smart_surface(session)

        captured = {}

        def capture_worker(*args):
            captured["image_jobs"] = args[0]
            raise InterruptedError

        monkeypatch.setattr(TRACKER, "prepare_run_and_solve_deepslice", capture_worker)
        window._start_deepslice_alignment([0], global_alignment=False)
        deadline = time.perf_counter() + 3.0
        while window.auto_alignment_busy and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert session.brain_brush_selection_mask is mask
        assert not session.brain_brush_strokes
        assert captured["image_jobs"][0][5] is None
        assert not window.auto_alignment_busy

        transform = object()
        session.brain_brush_strokes = [(False, [(20.0, 16.0)])]
        session.brain_outline_points = outline.copy()
        session.brain_brush_selection_mask = mask
        session.slice_to_atlas_x = transform
        session.atlas_to_slice_x = transform
        session.auto_alignment_engine = "DeepSlice"

        def fail_segmentation(*_args):
            raise RuntimeError("segmentation failed")

        monkeypatch.setattr(TRACKER, "smart_brain_surface_selection", fail_segmentation)
        window.smart_surface_mode.setChecked(True)
        window._smart_surface_stroke([(20.0, 16.0)], False)

        assert session.brain_brush_strokes == [(False, [(20.0, 16.0)])]
        assert session.brain_outline_points == outline
        assert session.brain_brush_selection_mask is mask
        assert session.slice_to_atlas_x is transform
        assert session.auto_alignment_engine == "DeepSlice"
    finally:
        window.close()
        app.processEvents()
