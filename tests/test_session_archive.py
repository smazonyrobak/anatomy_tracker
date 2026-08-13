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
    first.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3),
        first_image.shape,
        (30, 40),
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
    assert loaded.slice_to_atlas_x is not None
    mapped = TRACKER.map_session_display_to_atlas(
        loaded,
        np.asarray(TRACKER.transform_points(loaded.slice_landmarks[:3], loaded.slice_transform)),
    )
    assert np.allclose(mapped, loaded.atlas_landmarks, atol=1e-6)
    restored.close()
    window.close()
    app.processEvents()
