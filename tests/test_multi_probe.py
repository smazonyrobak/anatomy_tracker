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
        session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(identity, (12, 14), (12, 14))
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


def test_probe_marks_can_be_added_before_alignment(window):
    image = np.arange(80 * 100, dtype=np.uint16).reshape(80, 100)
    session = TRACKER.SliceSession(
        "raw-slice",
        raw_display=image,
        adjusted=image,
        rotated=image,
        weight_image=image,
    )
    window.sessions = [session]
    window.current_session_index = 0
    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    window.probe_mode.setChecked(True)

    window._slice_clicked(25.0, 30.0)

    trace = session.probe_traces["imec0"]
    assert trace.slice_points == pytest.approx([(25.0, 30.0)])
    assert trace.atlas_points == []
    assert trace.volume_points == []
    assert len(trace.signal_values) == 1
    assert "pre-alignment" in window.status.text()


def test_atlas_click_is_rejected_before_alignment_but_slice_marks_remain(window, monkeypatch):
    image = np.zeros((80, 100), dtype=np.uint8)
    session = TRACKER.SliceSession("raw-slice", rotated=image)
    window.sessions = [session]
    window.current_session_index = 0
    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    warnings = []
    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "warning",
        lambda *_args: warnings.append(_args[2]),
    )

    window._add_probe_point(atlas_point=(10.0, 20.0), slice_raw_point=None)

    assert "imec0" not in session.probe_traces
    assert warnings and "histology slice" in warnings[0]


def test_enabled_surgical_constraint_requires_pre_alignment_marks_in_selected_batch(
    window, monkeypatch, tmp_path
):
    source = tmp_path / "slice.tif"
    source.write_bytes(b"slice")
    image = np.zeros((40, 50), dtype=np.uint8)
    session = TRACKER.SliceSession(
        "slice",
        path=str(source),
        raw_display=image,
        adjusted=image,
        rotated=image,
        weight_image=image,
        brain_outline_points=[(float(i), float(i)) for i in range(8)],
        probe_traces={"imec0": TRACKER.ProbeTrace(slice_points=[(10.0, 12.0)])},
    )
    window.sessions = [session]
    window.current_session_index = 0
    window.atlas_volume = np.zeros((20, 40, 50), dtype=np.uint8)
    window.annotation_volume = np.ones((20, 40, 50), dtype=np.uint8)
    window.probe_constraints["imec0"] = TRACKER.ProbeInsertionConstraint(
        True, 0.0, 0.0, 250.0, 60.0, 5.0, 2000.0
    )
    warnings = []
    monkeypatch.setattr(
        TRACKER.QtWidgets.QMessageBox,
        "warning",
        lambda *_args: warnings.append(_args[2]),
    )
    submitted = []
    monkeypatch.setattr(window.alignment_executor, "submit", lambda *_args: submitted.append(_args))

    window._start_auto_alignment([0], global_alignment=False)

    assert not submitted
    assert warnings and "at least two marks" in warnings[0]


def test_pose_recompute_updates_every_probe_trace(window):
    session = TRACKER.SliceSession("slice")
    session.slice_atlas_transform = TRACKER.SliceAtlasTransform2D(
        np.eye(3),
        (80, 60),
        (80, 60),
    )
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
        "deepest_mark_is_tip",
        np.array([15.0, 10.0, 17.0]),
        np.array([15.0, 16.0, 17.0]),
        np.array([15.0, 16.0, 17.0]),
        np.array([15.0, 8.0, 17.0]),
        np.array([0.0, -1.0, 0.0]),
        150.0,
        150.0,
    )

    manifest = json.loads(
        (tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec1.json").read_text()
    )
    assert manifest["probe_name"] == "imec1"
    assert manifest["slices"][0]["probe_atlas_points"] == [[11, 12]]
    assert manifest["slices"][0]["probe_volume_points"] == [[15, 16, 17]]
    assert TRACKER.probe_color("imec0") != TRACKER.probe_color("imec1")


def test_insertion_constraints_are_retained_independently_per_probe(window):
    window.probe_name.addItems(["imec0", "imec1"])
    window.probe_name.setCurrentText("imec0")
    window.use_probe_constraints.setChecked(True)
    window.insertion_ap_um.setValue(-1400)
    window.insertion_ml_um.setValue(900)
    window.insertion_radius_um.setValue(200)
    window.attack_angle_deg.setValue(72.0)
    window.attack_angle_tolerance_deg.setValue(4.0)
    window.limit_insertion_depth.setChecked(True)
    window.max_insertion_depth_um.setValue(8500)

    window.probe_name.setCurrentText("imec1")
    assert not window.use_probe_constraints.isChecked()
    window.use_probe_constraints.setChecked(True)
    window.insertion_ap_um.setValue(-2200)
    window.insertion_ml_um.setValue(-700)

    window.probe_name.setCurrentText("imec0")
    assert window.use_probe_constraints.isChecked()
    assert window.insertion_ap_um.value() == -1400
    assert window.insertion_ml_um.value() == 900
    assert window.attack_angle_deg.value() == 72.0
    assert window.attack_angle_tolerance_deg.value() == 4.0
    assert window.max_insertion_depth_um.value() == 8500
    assert window.probe_constraints["imec0"].maximum_insertion_depth_um == 8500
    assert window.probe_constraints["imec1"].ap_um == -2200
    assert window.probe_constraints["imec1"].ml_um == -700


def test_tip_location_is_retained_independently_per_probe(window):
    window.probe_name.addItems(["imec0", "imec1"])
    window.probe_name.setCurrentText("imec0")
    window.endpoint_reference.setCurrentIndex(
        window.endpoint_reference.findData("known_insertion_depth")
    )
    window.mapping_insertion_depth_um.setValue(3200.0)

    window.probe_name.setCurrentText("imec1")
    assert window.endpoint_reference.currentIndex() == -1
    window.endpoint_reference.setCurrentIndex(
        window.endpoint_reference.findData("deepest_mark_is_tip")
    )

    window.probe_name.setCurrentText("imec0")
    assert window.endpoint_reference.currentData() == "known_insertion_depth"
    assert window.mapping_insertion_depth_um.value() == 3200.0
    window.probe_name.setCurrentText("imec1")
    assert window.endpoint_reference.currentData() == "deepest_mark_is_tip"


def test_enabled_constraint_is_written_with_explicit_angle_convention(window, tmp_path):
    window.probe_constraints["imec0"] = TRACKER.ProbeInsertionConstraint(
        enabled=True,
        ap_um=-1400,
        ml_um=800,
        radius_um=250,
        angle_deg=75,
        angle_tolerance_deg=5,
        maximum_insertion_depth_um=9000,
    )
    window._write_manifest(
        tmp_path,
        "imec0",
        "deepest_mark_is_tip",
        np.array([15.0, 10.0, 17.0]),
        np.array([15.0, 16.0, 17.0]),
        np.array([15.0, 16.0, 17.0]),
        np.array([15.0, 8.0, 17.0]),
        np.array([0.0, -1.0, 0.0]),
        150.0,
        150.0,
    )
    manifest = json.loads(
        (tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec0.json").read_text()
    )
    assert manifest["insertion_constraint"] == {
        "enabled": True,
        "ap_um": -1400,
        "ml_um": 800,
        "radius_um": 250,
        "angle_deg": 75,
        "angle_tolerance_deg": 5,
        "maximum_insertion_depth_um": 9000,
    }
    assert manifest["probe_attack_angle_convention"] == (
        "0 degrees horizontal; 90 degrees vertical"
    )
    assert manifest["probe_tip_location_mode"] == "deepest_mark_is_tip"
    assert manifest["physical_tip_to_y0_contact_um"] == 200.0
    assert manifest["insertion_depth_from_surface_um"] == 150.0


def test_constraint_circle_is_projected_onto_annotation_surface(window):
    window.atlas_volume = np.zeros((40, 30, 50), dtype=np.uint8)
    window.annotation_volume = np.zeros_like(window.atlas_volume)
    window.annotation_volume[:, 7:, :] = 1
    window.cortical_region_ids = {1}
    window.bregma_voxel = np.array([20.0, 10.0, 25.0])
    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    window.probe_constraints["imec0"] = TRACKER.ProbeInsertionConstraint(
        enabled=True,
        radius_um=100,
        angle_deg=90,
        angle_tolerance_deg=5,
    )
    window._refresh_3d()

    rings = [
        item for item in window.dynamic_gl_items
        if isinstance(item, TRACKER.gl.GLLinePlotItem) and len(item.pos) == 97
    ]
    assert len(rings) == 1
    # Annotation surface is DV index 7, rendered as GL z = 320 - 7.
    assert np.allclose(rings[0].pos[:, 2], TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML[1] - 7)


def test_constraint_surface_requires_dorsal_isocortex_not_deeper_cortex(window, tmp_path):
    query = tmp_path / "query.csv"
    TRACKER.pd.DataFrame(
        {
            "id": [1, 2],
            "structure_id_path": ["/997/315/1/", "/997/8/2/"],
        }
    ).to_csv(query, index=False)
    assert window._load_cortical_region_ids(query) == {1}

    window.annotation_volume = np.zeros((3, 6, 3), dtype=np.uint16)
    window.bregma_voxel = np.array([1.0, 0.0, 1.0])
    window.cortical_region_ids = {1}
    window.annotation_volume[1, 2, 1] = 2
    window.annotation_volume[1, 3, 1] = 1
    assert np.isnan(window._surface_dv_um(0.0, 0.0))
    window.annotation_volume[1, 2, 1] = 1
    assert window._surface_dv_um(0.0, 0.0) == pytest.approx(-50.0)


def test_unconstrained_probe_geometry_rejects_physical_shank_overrun(window):
    session = TRACKER.SliceSession("slice")
    session.probe_traces = {
        "imec0": TRACKER.ProbeTrace(volume_points=[[100, 520, 90], [100, -80, 90]])
    }
    window.sessions = [session]
    window.annotation_volume = np.zeros((220, 620, 180), dtype=np.uint8)
    window.annotation_volume[:, 100:500, :] = 1
    with pytest.raises(TRACKER.InfeasibleProbeConstraint, match="10 mm physical shank"):
        window.probe_brain_geometry("imec0")


def test_partial_cortical_target_ring_draws_only_finite_arcs_and_warns(window):
    window.bregma_voxel = np.asarray([2.0, 2.0, 2.0])
    window.annotation_volume = np.zeros((5, 5, 5), dtype=np.uint8)
    window.annotation_volume[1:4, 1:4, 1:4] = 1
    window.cortical_region_ids = {1}
    window.probe_name.addItem("imec0")
    window.probe_name.setCurrentText("imec0")
    window.probe_constraints["imec0"] = TRACKER.ProbeInsertionConstraint(
        True, 50.0, 0.0, 50.0, 60.0, 5.0
    )

    window._refresh_3d()

    lines = [
        item for item in window.dynamic_gl_items
        if isinstance(item, TRACKER.gl.GLLinePlotItem)
    ]
    assert lines
    assert all(np.isfinite(item.pos).all() for item in lines)
    assert "partly outside dorsal cortex" in window.probe_fit_summary.text()


def test_only_alignment_runs_that_used_a_probe_are_invalidated(window):
    constrained = TRACKER.SliceSession("constrained")
    constrained.auto_alignment_run_id = "run-constrained"
    constrained.auto_alignment_diagnostics = {
        "probe_geometry_constraints": {
            "applied": True,
            "probes": {"imec0": {}},
        }
    }
    image_only = TRACKER.SliceSession("image-only")
    image_only.auto_alignment_run_id = "run-image"
    image_only.auto_alignment_diagnostics = {"pose_search_score": 0.1}
    other_probe = TRACKER.SliceSession("other-probe")
    other_probe.auto_alignment_run_id = "run-other"
    other_probe.auto_alignment_diagnostics = {
        "probe_geometry_constraints": {
            "applied": True,
            "probes": {"imec1": {}},
        }
    }
    window.sessions = [constrained, image_only, other_probe]

    window._mark_probe_constrained_alignment_stale(
        constrained,
        "imec0",
        "imec0 surgical constraint changed",
    )

    assert constrained.auto_alignment_diagnostics["alignment_run_stale"]
    assert not image_only.auto_alignment_diagnostics.get("alignment_run_stale", False)
    assert not other_probe.auto_alignment_diagnostics.get("alignment_run_stale", False)
