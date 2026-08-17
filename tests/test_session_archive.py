import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_session_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def test_complete_session_round_trip_embeds_images_and_state(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    first_path = tmp_path / "0.tif"
    second_path = tmp_path / "1.tif"
    first_image = np.arange(320, dtype=np.uint16).reshape(16, 20)
    second_image = np.flipud(first_image)
    tifffile.imwrite(first_path, first_image)
    tifffile.imwrite(second_path, second_image)

    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(first_path)
    window.load_slice(second_path)
    first, second = window.sessions
    first.brain_outline_points = [(1.0, 2.0)] * 8
    first.brain_outline_closed = True
    first.brain_brush_strokes = [(False, [(3.0, 4.0), (5.0, 6.0)])]
    first.brain_brush_selection_mask = np.eye(16, 20, dtype=bool)
    first.probe_traces["imec0"] = TRACKER.ProbeTrace(
        atlas_points=[(8.0, 9.0)],
        slice_points=[(4.0, 5.0)],
        volume_points=[[10.0, 11.0, 12.0]],
        signal_values=[13.0],
    )
    atlas_y, atlas_x = np.mgrid[:30, :40].astype(np.float32)
    dense_identity = np.stack((atlas_x, atlas_y), axis=-1)
    first.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3),
        first_image.shape,
        (30, 40),
        dense_identity,
        dense_identity,
        np.ones((30, 40), dtype=bool),
        '{"model":"AtlasWarp-test"}',
    )
    first.auto_alignment_engine = TRACKER.POSE_ENGINE_OWN_CNN
    first.auto_alignment_diagnostics = {"probe_geometry_constraints": {"applied": True}}
    first.alignment_source_sha256 = TRACKER.file_sha256(first_path)
    second.brain_outline_points = [(2.0, 3.0)] * 8
    second.brain_outline_closed = True

    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    window.probe_constraints["imec0"] = TRACKER.ProbeInsertionConstraint(
        True, -1400.0, -1600.0, 400.0, 50.0, 5.0, 10000.0
    )
    window.probe_endpoint_settings["imec0"] = ("known_insertion_depth", 3200.0)
    window.limit_auto_align_ap.setChecked(True)
    window.auto_align_ap_min.setValue(-2200)
    window.auto_align_ap_max.setValue(-1100)
    window.outline_point_count.setValue(75)
    window.atlas_opacity.setValue(37)
    window.brain_opacity.setValue(58)
    window._update_auto_order_labels()
    for row in range(window.auto_slice_order.count()):
        window.auto_slice_order.item(row).setCheckState(TRACKER.QtCore.Qt.CheckState.Checked)

    archive = tmp_path / "real_example.attracker"
    window.save_session_file(archive)
    first_path.unlink()
    second_path.unlink()

    restored = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    restored.load_session_file(archive)

    assert len(restored.sessions) == 2
    assert np.array_equal(restored.sessions[0].raw_display, TRACKER.normalize_u8(first_image))
    assert restored.sessions[0].brain_brush_selection_mask.sum() == 16
    assert restored.sessions[0].probe_traces["imec0"].slice_points == [[4.0, 5.0]]
    assert restored.sessions[0].slice_atlas_transform is not None
    assert restored.sessions[0].slice_atlas_transform.nonlinear
    assert np.array_equal(
        restored.sessions[0].slice_atlas_transform.atlas_to_affine_xy,
        dense_identity,
    )
    assert restored.sessions[0].slice_atlas_transform.registration_metadata_json == '{"model":"AtlasWarp-test"}'
    assert restored.sessions[0].auto_alignment_engine == TRACKER.POSE_ENGINE_OWN_CNN
    assert restored.probe_constraints["imec0"].radius_um == 400.0
    assert restored.probe_endpoint_settings["imec0"] == ("known_insertion_depth", 3200.0)
    assert restored.limit_auto_align_ap.isChecked()
    assert restored.auto_align_ap_min.value() == -2200
    assert restored.auto_align_ap_max.value() == -1100
    assert restored.outline_point_count.value() == 75
    assert restored.atlas_opacity.value() == 37
    assert restored.brain_opacity.value() == 58
    assert len(restored._auto_order_constraint_session_indices()) == 2

    restored.close()
    window.close()
    app.processEvents()


def test_session_load_preserves_an_unpaired_landmark_and_rebuilds_complete_pairs(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    image_path = tmp_path / "slice.tif"
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    tifffile.imwrite(image_path, image)
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(image_path)
    session = window.sessions[0]
    session.atlas_landmarks = [(2.0, 2.0), (30.0, 2.0), (2.0, 20.0)]
    session.slice_landmarks = [(3.0, 3.0), (31.0, 3.0), (3.0, 21.0), (18.0, 12.0)]
    archive = tmp_path / "unpaired.attracker"
    window.save_session_file(archive)

    restored = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    restored.load_session_file(archive)

    loaded = restored.sessions[0]
    assert loaded.atlas_landmarks == [list(point) for point in session.atlas_landmarks]
    assert loaded.slice_landmarks == [list(point) for point in session.slice_landmarks]
    assert loaded.slice_to_atlas_tps is not None
    mapped = TRACKER.map_session_display_to_atlas(
        loaded,
        np.asarray(TRACKER.transform_points(loaded.slice_landmarks[:3], loaded.slice_transform)),
    )
    assert np.allclose(mapped, loaded.atlas_landmarks, atol=1e-6)
    restored.close()
    window.close()
    app.processEvents()


def test_session_load_rejects_transform_shape_that_does_not_match_slice(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    image_path = tmp_path / "slice.tif"
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    tifffile.imwrite(image_path, image)
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(image_path)
    window.sessions[0].slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3), (31, 40), (30, 40)
    )
    archive = tmp_path / "bad-display-shape.attracker"
    window.save_session_file(archive)

    restored = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        with pytest.raises(ValueError, match="does not match its displayed slice shape"):
            restored.load_session_file(archive)
    finally:
        restored.close()
        window.close()
        app.processEvents()


def test_session_load_rejects_transform_shape_that_does_not_match_loaded_atlas(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    image_path = tmp_path / "slice.tif"
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    tifffile.imwrite(image_path, image)
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(image_path)
    window.sessions[0].slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3), image.shape, (31, 40)
    )
    archive = tmp_path / "bad-atlas-shape.attracker"
    window.save_session_file(archive)

    restored = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    restored.atlas_volume = np.zeros((2, 30, 40), dtype=np.uint8)
    restored.annotation_volume = np.zeros_like(restored.atlas_volume, dtype=np.uint16)
    try:
        with pytest.raises(ValueError, match="does not match the coronal atlas canvas"):
            restored.load_session_file(archive)
    finally:
        restored.close()
        window.close()
        app.processEvents()


def test_remove_all_slices_can_cancel_clear_and_load_another_brain(tmp_path, monkeypatch):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    paths = [tmp_path / name for name in ("0.tif", "1.tif", "new-brain.tif")]
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    for index, path in enumerate(paths):
        tifffile.imwrite(path, image + index)

    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    window.load_slice(paths[0])
    window.load_slice(paths[1])
    window.sessions[0].brain_outline_points = [(float(index), 2.0) for index in range(8)]
    window._update_auto_order_labels()

    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "question",
        lambda *args: TRACKER.QtWidgets.QMessageBox.StandardButton.No,
    )
    window.remove_all_slices_btn.click()
    assert len(window.sessions) == 2
    assert window.slice_list.count() == 2

    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "question",
        lambda *args: TRACKER.QtWidgets.QMessageBox.StandardButton.Yes,
    )
    window.remove_all_slices_btn.click()
    assert window.sessions == []
    assert window.current_session_index == -1
    assert window.slice_list.count() == 0
    assert window.slice_position.text() == "0 / 0"
    assert window.auto_slice_order.count() == 0
    assert window.slice_panel.image_shape is None
    assert not window.remove_all_slices_btn.isEnabled()
    assert all(path.exists() for path in paths[:2])

    window.load_slice(paths[2])
    assert len(window.sessions) == 1
    assert window.current_session() is window.sessions[0]
    assert window.slice_position.text() == "1 / 1"
    assert window.slice_panel.image_shape == image.shape
    assert window.remove_all_slices_btn.isEnabled()

    window.close()
    app.processEvents()


def test_remove_selected_slice_keeps_the_remaining_slice_set_consistent(tmp_path, monkeypatch):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    paths = [tmp_path / f"{index}.tif" for index in range(3)]
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    for index, path in enumerate(paths):
        tifffile.imwrite(path, image + index)

    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    for path in paths:
        window.load_slice(path)
    for session in window.sessions:
        session.brain_outline_points = [(float(index), 2.0) for index in range(8)]
    window._update_auto_order_labels()
    window.slice_list.setCurrentIndex(1)
    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "question",
        lambda *args: TRACKER.QtWidgets.QMessageBox.StandardButton.Yes,
    )

    window.remove_selected_slice_btn.click()

    assert [session.name for session in window.sessions] == ["0.tif", "2.tif"]
    assert [window.slice_list.itemText(index) for index in range(2)] == ["0.tif", "2.tif"]
    assert window.current_session() is window.sessions[1]
    assert window.slice_position.text() == "2 / 2"
    assert [
        window.auto_slice_order.item(row).data(TRACKER.QtCore.Qt.ItemDataRole.UserRole)
        for row in range(window.auto_slice_order.count())
    ] == [0, 1]
    assert paths[1].exists()
    assert window.remove_selected_slice_btn.isEnabled()

    window.close()
    app.processEvents()
