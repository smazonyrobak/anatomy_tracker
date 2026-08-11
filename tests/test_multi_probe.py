import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_multi_probe_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


@pytest.fixture
def window(tmp_path):
    application = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    tracker = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    yield tracker
    tracker.close()
    application.processEvents()


def test_probe_regressions_are_independent(window):
    first = TRACKER.SliceSession("first")
    second = TRACKER.SliceSession("second")
    first.probe_traces = {
        "imec0": TRACKER.ProbeTrace(volume_points=[[100, 220, 90], [100, 180, 90]]),
        "imec1": TRACKER.ProbeTrace(volume_points=[[80, 220, 120], [100, 180, 120]]),
    }
    second.probe_traces = {
        "imec0": TRACKER.ProbeTrace(volume_points=[[100, 140, 90], [100, 100, 90]]),
        "imec1": TRACKER.ProbeTrace(volume_points=[[120, 140, 120], [140, 100, 120]]),
    }
    window.sessions = [first, second]

    _, imec0 = window.probe_regression("imec0")
    _, imec1 = window.probe_regression("imec1")

    assert imec0 == pytest.approx([0, -1, 0], abs=1e-12)
    assert imec1 == pytest.approx(np.array([1, -2, 0]) / np.sqrt(5), abs=1e-12)
    assert not np.allclose(imec0, imec1)


def test_both_probe_regressions_render_together_with_distinct_colors(window):
    session = TRACKER.SliceSession("slice")
    session.probe_traces = {
        "imec0": TRACKER.ProbeTrace(volume_points=[[100, 220, 90], [100, 180, 90]]),
        "imec1": TRACKER.ProbeTrace(
            volume_points=[[80, 220, 120], [100, 180, 120], [120, 140, 120]]
        ),
    }
    window.sessions = [session]
    window.current_session_index = 0
    window.atlas_volume = None
    window.annotation_volume = None
    window.probe_name.addItems(["imec0", "imec1"])
    window.probe_name.setCurrentText("imec0")
    window._refresh_3d()

    def trajectory_lines():
        return [
            item
            for item in window.dynamic_gl_items
            if isinstance(item, TRACKER.gl.GLLinePlotItem)
        ]

    def color_key(item):
        return tuple(np.round(np.asarray(item.color[:3], dtype=float), 6))

    def expected_color(name):
        return tuple(np.round(np.asarray(TRACKER.probe_color(name)) / 255.0, 6))

    lines = trajectory_lines()
    assert len(lines) == 2
    by_color = {color_key(item): item for item in lines}
    assert by_color[expected_color("imec0")].width == 4
    assert by_color[expected_color("imec1")].width == 2
    assert not np.allclose(
        by_color[expected_color("imec0")].pos,
        by_color[expected_color("imec1")].pos,
    )
    assert all(item.pos[0, 2] > item.pos[1, 2] for item in lines)

    window.probe_name.setCurrentText("imec1")
    by_color = {color_key(item): item for item in trajectory_lines()}
    assert len(by_color) == 2
    assert by_color[expected_color("imec0")].width == 2
    assert by_color[expected_color("imec1")].width == 4


def test_3d_slice_planes_require_both_surface_and_completed_alignment(window):
    identity = np.eye(3)

    def aligned_session(name, *, surface):
        session = TRACKER.SliceSession(name, brain_outline_points=[(1.0, 1.0)] if surface else [])
        session.slice_to_atlas_x = TRACKER.AffineCoordinate(identity, 0)
        session.slice_to_atlas_y = TRACKER.AffineCoordinate(identity, 1)
        session.atlas_to_slice_x = TRACKER.AffineCoordinate(identity, 0)
        session.atlas_to_slice_y = TRACKER.AffineCoordinate(identity, 1)
        return session

    window.atlas_volume = np.zeros((10, 12, 14), dtype=np.uint8)
    window.sessions = [
        TRACKER.SliceSession("unused"),
        TRACKER.SliceSession("surface-only", brain_outline_points=[(1.0, 1.0)]),
        aligned_session("alignment-only", surface=False),
        aligned_session("registered", surface=True),
    ]
    window.current_session_index = 0
    window.show_all_slice_planes.setChecked(True)
    window._refresh_3d()

    meshes = [item for item in window.dynamic_gl_items if isinstance(item, TRACKER.gl.GLMeshItem)]
    assert len(meshes) == 1

    window.show_all_slice_planes.setChecked(False)
    window._refresh_3d()
    assert not any(isinstance(item, TRACKER.gl.GLMeshItem) for item in window.dynamic_gl_items)

    window.current_session_index = 3
    window._refresh_3d()
    assert sum(isinstance(item, TRACKER.gl.GLMeshItem) for item in window.dynamic_gl_items) == 1


def test_undo_and_clear_touch_only_selected_probe(window):
    session = TRACKER.SliceSession("slice")
    session.probe_traces = {
        "imec0": TRACKER.ProbeTrace(
            atlas_points=[(1, 1), (2, 2)],
            slice_points=[(1, 1), (2, 2)],
            volume_points=[[1, 1, 1], [2, 2, 2]],
            signal_values=[10, 20],
        ),
        "imec1": TRACKER.ProbeTrace(
            atlas_points=[(3, 3)],
            slice_points=[(3, 3)],
            volume_points=[[3, 3, 3]],
            signal_values=[30],
        ),
    }
    session.point_history = ["probe:imec0", "probe:imec1", "probe:imec0"]
    window.sessions = [session]
    window.current_session_index = 0
    window.probe_name.addItems(["imec0", "imec1"])
    window.probe_name.setCurrentText("imec0")
    window.probe_mode.setChecked(True)

    window.undo_last_point()
    assert session.probe_traces["imec0"].slice_points == [(1, 1)]
    assert session.probe_traces["imec1"].slice_points == [(3, 3)]

    window.clear_current_points()
    assert session.probe_traces["imec0"].slice_points == []
    assert session.probe_traces["imec1"].slice_points == [(3, 3)]
    assert session.point_history == ["probe:imec1"]


def test_pose_recompute_updates_every_probe_trace(window):
    session = TRACKER.SliceSession("slice")
    session.slice_to_atlas_x = TRACKER.AffineCoordinate(np.eye(3), 0)
    session.slice_to_atlas_y = TRACKER.AffineCoordinate(np.eye(3), 1)
    session.probe_traces = {
        "imec0": TRACKER.ProbeTrace(slice_points=[(10, 20)]),
        "imec1": TRACKER.ProbeTrace(slice_points=[(30, 40)]),
    }
    window.atlas_volume = np.zeros((100, 80, 60), dtype=np.uint8)

    window._recompute_probe_points_from_slice_points(session)

    assert session.probe_traces["imec0"].atlas_points == [(10, 20)]
    assert session.probe_traces["imec1"].atlas_points == [(30, 40)]
    assert len(session.probe_traces["imec0"].volume_points) == 1
    assert len(session.probe_traces["imec1"].volume_points) == 1


def test_manifest_contains_only_requested_probe_trace(window, tmp_path):
    session = TRACKER.SliceSession("slice")
    session.probe_traces = {
        "imec0": TRACKER.ProbeTrace(
            atlas_points=[(1, 2)],
            slice_points=[(3, 4)],
            volume_points=[[5, 6, 7]],
            signal_values=[8],
        ),
        "imec1": TRACKER.ProbeTrace(
            atlas_points=[(11, 12)],
            slice_points=[(13, 14)],
            volume_points=[[15, 16, 17]],
            signal_values=[18],
        ),
    }
    window.sessions = [session]

    window._write_manifest(
        tmp_path,
        "imec1",
        "y0_contact",
        np.array([15.0, 16.0, 17.0]),
        np.array([15.0, 16.0, 17.0]),
        np.array([0.0, -1.0, 0.0]),
    )

    manifest = json.loads(
        (tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec1.json").read_text()
    )
    assert manifest["probe_name"] == "imec1"
    assert manifest["slices"][0]["probe_atlas_points"] == [[11, 12]]
    assert manifest["slices"][0]["probe_volume_points"] == [[15, 16, 17]]
    assert TRACKER.probe_color("imec0") != TRACKER.probe_color("imec1")
