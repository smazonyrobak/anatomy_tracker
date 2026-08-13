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
            auto_alignment_diagnostics={"alignment_run_id": "pose-run"},
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
        registration = session.auto_alignment_diagnostics["landmark_registration"]
        assert registration["method"] == "thin_plate_spline"
        assert registration["landmark_pairs"] == 5
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


def test_manifest_records_landmark_coordinate_map_and_probe_roll(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        session = TRACKER.SliceSession(
            "slice",
            auto_alignment_diagnostics={
                "landmark_registration": {
                    "status": "applied",
                    "method": "thin_plate_spline",
                    "landmark_pairs": 5,
                }
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
        assert manifest["slices"][0]["landmark_registration"]["method"] == "thin_plate_spline"
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
