import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_registration_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def test_coordinate_registration_reader_accepts_legacy_sessions_and_prefers_applied_state():
    diagnostics = {
        "anatomical_registration": {"status": "pending", "reason": "atlas changed"},
        "landmark_registration": {
            "status": "applied",
            "method": "thin_plate_spline",
            "landmark_pairs": 5,
        },
    }

    registration = TRACKER.coordinate_registration_record(diagnostics)

    assert registration["kind"] == "landmark"
    assert registration["status"] == "applied"
    assert registration["landmark_pairs"] == 5


def test_landmark_tps_preserves_an_affine_map_when_a_consistent_pair_is_added():
    four_points = np.asarray(
        [(0.0, 0.0), (1000.0, 0.0), (0.0, 800.0), (1000.0, 800.0)],
        dtype=np.float64,
    )
    affine = np.asarray(
        [[1.08, 0.12, 31.0], [-0.07, 0.94, -18.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def apply_affine(points):
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (affine @ homogeneous.T).T[:, :2]

    grid_y, grid_x = np.mgrid[0:801:41j, 0:1001:51j]
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    four_point_tps, _ = TRACKER.fit_landmark_tps(four_points, apply_affine(four_points))
    center = np.asarray([[500.0, 400.0]])
    five_points = np.vstack((four_points, center))
    five_point_tps, inverse_tps = TRACKER.fit_landmark_tps(five_points, apply_affine(five_points))

    expected = apply_affine(grid)
    four_result = four_point_tps(grid)
    five_result = five_point_tps(grid)
    assert np.max(np.abs(four_result - expected)) < 1e-8
    assert np.max(np.abs(five_result - expected)) < 1e-8
    assert np.max(np.abs(five_result - four_result)) < 1e-8
    assert np.max(np.abs(inverse_tps(five_result) - grid)) < 1e-8


def test_landmark_tps_rejects_incomplete_duplicate_and_collinear_pairs():
    triangle = np.asarray([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
    with pytest.raises(ValueError, match="counts must match"):
        TRACKER.fit_landmark_tps(triangle, triangle[:2])
    with pytest.raises(ValueError, match="must be unique"):
        TRACKER.fit_landmark_tps(triangle, np.asarray([(0.0, 0.0), (0.0, 0.0), (0.0, 10.0)]))
    with pytest.raises(ValueError, match="one line"):
        TRACKER.fit_landmark_tps(
            np.asarray([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]),
            triangle,
        )


def test_landmark_capture_keeps_pairs_synchronized_and_undoes_them_together(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        image = np.zeros((32, 48), dtype=np.uint8)
        session = TRACKER.SliceSession(
            "slice.tif",
            raw_display=image,
            adjusted=image,
            rotated=image,
            slice_transform=np.eye(3),
        )
        window.sessions = [session]
        window.current_session_index = 0
        window.alignment_tabs.setCurrentIndex(0)
        window.landmark_mode.setChecked(True)

        window._atlas_clicked(5.0, 6.0)
        window._atlas_clicked(7.0, 8.0)
        assert len(session.atlas_landmarks) == 1
        window._slice_clicked(9.0, 10.0)
        assert len(session.slice_landmarks) == 1

        window._slice_clicked(11.0, 12.0)
        window._slice_clicked(13.0, 14.0)
        assert len(session.slice_landmarks) == 2
        window._refresh_point_counts()
        assert not window.transform_btn.isEnabled()
        window.undo_last_point()
        assert len(session.atlas_landmarks) == len(session.slice_landmarks) == 1

        window._atlas_clicked(15.0, 16.0)
        window._slice_clicked(17.0, 18.0)
        window.undo_last_point()
        assert len(session.atlas_landmarks) == len(session.slice_landmarks) == 1
    finally:
        window.close()
        app.processEvents()


def test_landmark_warp_remaps_probe_coordinates_and_preserves_automatic_pose(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (64, 96)
        image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        slice_landmarks = [(10.0, 10.0), (85.0, 10.0), (10.0, 55.0), (85.0, 55.0), (45.0, 30.0)]
        atlas_landmarks = [(10.0, 10.0), (85.0, 10.0), (10.0, 55.0), (85.0, 55.0), (55.0, 35.0)]
        session = TRACKER.SliceSession(
            "slice.tif",
            raw_display=image,
            adjusted=image,
            rotated=image,
            weight_image=image,
            slice_transform=np.eye(3),
            atlas_index=12,
            atlas_tilt_ml_deg=3.0,
            atlas_tilt_dv_deg=-2.0,
            atlas_landmarks=atlas_landmarks,
            slice_landmarks=slice_landmarks,
            slice_atlas_transform=TRACKER.SliceAtlasTransform2D(np.eye(3), shape, shape),
            auto_alignment_engine=TRACKER.POSE_ENGINE_OWN_CNN,
            auto_alignment_run_id="pose-run",
            auto_alignment_diagnostics={
                "alignment_run_id": "pose-run",
                "coordinate_registration": {
                    "kind": "automatic",
                    "status": "pending",
                    "reason": "section matched",
                },
            },
            probe_traces={"imec0": TRACKER.ProbeTrace(slice_points=[(45.0, 30.0)])},
        )
        window.sessions = [session]
        window.current_session_index = 0
        window.current_atlas_image = image
        window.atlas_volume = np.zeros((30, *shape), dtype=np.uint8)

        assert window._rebuild_slice_transform(session) == 5
        window._recompute_probe_points_from_slice_points(session)

        trace = session.probe_traces["imec0"]
        assert np.allclose(trace.atlas_points[0], atlas_landmarks[-1], atol=1e-6)
        assert np.allclose(
            trace.volume_points[0],
            TRACKER.point_to_volume(atlas_landmarks[-1], "coronal", 12, window.atlas_volume.shape, 3.0, -2.0),
        )
        assert session.transformed_overlay is not None
        assert session.auto_alignment_engine == TRACKER.POSE_ENGINE_OWN_CNN
        assert session.auto_alignment_run_id == "pose-run"
        registration = session.auto_alignment_diagnostics["coordinate_registration"]
        assert registration["kind"] == "landmark"
        assert registration["status"] == "applied"
        assert registration["method"] == "thin_plate_spline"
        assert registration["landmark_pairs"] == 5
        assert "anatomical_registration" not in session.auto_alignment_diagnostics
        assert "landmark_registration" not in session.auto_alignment_diagnostics
    finally:
        window.close()
        app.processEvents()


def test_ui_separates_section_matching_from_landmark_registration(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        assert window.alignment_tabs.tabText(0) == "Landmark registration"
        assert window.alignment_tabs.tabText(1) == "Automatic section matching"
        assert window.transform_btn.text() == "Apply landmark warp"
        assert window.automatic_warp_btn.text() == "Apply automatic warp"
        assert "Alternative to landmark registration" in window.automatic_warp_btn.toolTip()
        assert "selected atlas section" in window.automatic_warp_btn.toolTip()
        assert "Review the atlas overlay" in window.automatic_warp_btn.toolTip()
        assert "smart brush" in window.curve_group.title().lower()
        window.alignment_tabs.setCurrentIndex(1)
        assert not window.curve_group.isHidden()
        window.alignment_tabs.setCurrentIndex(0)
        assert not window.curve_group.isHidden()
        assert not hasattr(window, "fit_anatomy_btn")
    finally:
        window.close()
        app.processEvents()


def test_removed_experimental_warp_loads_as_its_affine_pose(tmp_path):
    path = tmp_path / "legacy_transform.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray(3, dtype=np.uint16),
        coordinate_convention=np.asarray("display_xy->affine_atlas_xy->atlas_xy;pixel_centers"),
        display_shape=np.asarray((32, 48), dtype=np.int32),
        atlas_shape=np.asarray((40, 60), dtype=np.int32),
        display_to_affine_atlas_h=np.eye(3),
        nonlinear=np.asarray(True, dtype=np.uint8),
        atlas_to_affine_xy=np.zeros((40, 60, 2), dtype=np.float32),
        affine_to_atlas_xy=np.zeros((40, 60, 2), dtype=np.float32),
    )

    transform = TRACKER.SliceAtlasTransform2D.load_npz(path)

    assert np.array_equal(transform.display_to_affine_atlas_h, np.eye(3))
    assert transform.display_shape == (32, 48)
    assert transform.atlas_shape == (40, 60)


def test_manifest_saves_applied_dense_coordinate_map_and_probe_roll(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (32, 48)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        identity = np.stack((xx, yy), axis=-1)
        session = TRACKER.SliceSession(
            "slice",
            auto_alignment_diagnostics={
                "coordinate_registration": {
                    "kind": "automatic",
                    "status": "applied",
                    "method": "dense-registration-onnx-v2",
                    "model_sha256": "a" * 64,
                },
            },
            slice_atlas_transform=TRACKER.SliceAtlasTransform2D(
                np.eye(3),
                shape,
                shape,
                identity,
                identity,
                np.ones(shape, dtype=bool),
            ),
            probe_traces={
                "imec0": TRACKER.ProbeTrace(
                    slice_points=[(10.0, 10.0)],
                    atlas_points=[(10.0, 10.0)],
                    volume_points=[[1.0, 2.0, 3.0]],
                )
            },
        )
        window.sessions = [session]
        window._write_manifest(
            tmp_path,
            "imec0",
            "deepest_mark_is_tip",
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
            0.0,
            0.0,
        )
        manifest = json.loads(
            (tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec0.json").read_text(encoding="utf-8")
        )
        slice_record = manifest["slices"][0]
        assert slice_record["coordinate_registration"]["kind"] == "automatic"
        assert slice_record["coordinate_registration"]["method"] == "dense-registration-onnx-v2"
        transform_file = tmp_path / "anatomy" / slice_record["slice_atlas_transform_file"]
        assert transform_file.is_file()
        restored = TRACKER.SliceAtlasTransform2D.load_npz(transform_file)
        assert restored.nonlinear
        assert np.array_equal(restored.valid_atlas_mask, np.ones(shape, dtype=bool))
        assert manifest["fitted_probe_attack_angle_deg"] == pytest.approx(45.0)
        assert manifest["fitted_probe_roll_deg"] == 0.0
    finally:
        window.close()
        app.processEvents()


def test_mapping_manifest_is_promoted_last(tmp_path):
    staging = tmp_path / "stage"
    destination = tmp_path / "output"
    backup = tmp_path / "backup"
    relative_paths = [
        Path("anatomy/proprietary_trajectory_manifest_imec0.json"),
        Path("channels.csv"),
        Path("units.csv"),
    ]
    for index, relative_path in enumerate(relative_paths):
        path = staging / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"staged-{index}".encode())
    staged_hashes = {
        relative_path: TRACKER.file_sha256(staging / relative_path)
        for relative_path in relative_paths
    }
    promoted = []

    def record_replace(source, target):
        promoted.append(Path(target).relative_to(destination))
        os.replace(source, target)

    TRACKER.promote_staged_mapping_outputs(
        staging,
        destination,
        backup,
        staged_hashes,
        replace_file=record_replace,
    )

    assert promoted[-1] == Path("anatomy/proprietary_trajectory_manifest_imec0.json")
