import hashlib
import importlib.util
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_nonlinear_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)
from nonlinear_registration import RUNTIME_GATE_VERSION, array_sha256
from diffeomorphic_registration_runtime import _correspondence_diagnostics, _gray_unit


def _worker_inputs():
    shape = (4, 12, 16)
    image = np.arange(shape[1] * shape[2], dtype=np.uint8).reshape(shape[1:])
    mask = np.ones(shape[1:], dtype=bool)
    prediction = {"Filenames": "slice_0000.png"}
    transform = TRACKER.SliceAtlasTransform2D(np.eye(3), image.shape, image.shape)
    diagnostics = {
        "alignment_run_id": "test-run",
        "input_crop": {"source_image_sha256": "c" * 64},
    }
    prepared = [(0, 1, 2.0, -3.0, transform, prediction, diagnostics)]
    runtime = {"component_provenance": {}}
    return shape, image, mask, prediction, prepared, runtime


def _accepted_diagnostics(warp, atlas_mask, affine_mask, fixed, moving, source_sha256):
    correspondence = _correspondence_diagnostics(
        _gray_unit(fixed, atlas_mask),
        _gray_unit(moving, affine_mask),
        atlas_mask,
        affine_mask,
        warp.atlas_to_affine_xy,
    )
    return {
        **warp.diagnostics(atlas_mask, affine_mask),
        **correspondence,
        "modeled_trusted_fraction": 1.0,
        "rejection_probability": 0.0,
        "model_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "source_image_sha256": source_sha256,
        "atlas_image_sha256": array_sha256(fixed),
        "moving_affine_sha256": array_sha256(moving),
        "runtime_gate_version": RUNTIME_GATE_VERSION,
        "pixel_spacing_um": TRACKER.VOXEL_UM,
    }


def _bound_transform(window, source_path, atlas_index, warp, mask):
    raw = TRACKER.tifffile.imread(str(source_path))
    display_raw, _ = TRACKER.downsample_for_display(TRACKER.as_gray(raw))
    display_image, _ = TRACKER.transform_slice_image(
        TRACKER.normalize_u8(display_raw), 0.0, False, False
    )
    affine = TRACKER.SliceAtlasTransform2D(np.eye(3), display_image.shape, mask.shape)
    fixed = TRACKER.coronal_oblique_slice(
        window.atlas_volume, atlas_index, 0.0, 0.0, order=1
    )
    moving = affine.render_display_image_in_atlas(display_image)
    diagnostics = _accepted_diagnostics(
        warp,
        mask,
        mask,
        fixed,
        moving,
        TRACKER.file_sha256(source_path),
    )
    attestation = TRACKER.NonlinearWarpAttestation.from_runtime(
        warp, mask, mask, diagnostics
    )
    return TRACKER.SliceAtlasTransform2D(
        np.eye(3), display_image.shape, mask.shape, warp, attestation
    ), diagnostics


def test_worker_runs_post_affine_on_raw_normalized_image_and_exact_atlas_plane(monkeypatch):
    shape, image, mask, _prediction, prepared, _runtime = _worker_inputs()
    atlas = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    annotation = np.ones(shape, dtype=np.uint16)
    captured = {}
    monkeypatch.setattr(
        TRACKER,
        "prepare_pose_inputs",
        lambda *_args: (
            ["slice_0000.png"],
            {"slice_0000.png": {"source_image_sha256": "c" * 64}},
            {"slice_0000.png": {"image": image, "brain_mask": mask}},
        ),
    )

    from nonlinear_registration import NonlinearWarp2D

    def accept(fixed, moving, fixed_mask, moving_mask, *_args, **kwargs):
        captured.update(
            fixed=fixed.copy(), moving=moving.copy(),
            fixed_mask=fixed_mask.copy(), moving_mask=moving_mask.copy(),
        )
        warp = NonlinearWarp2D.identity(mask.shape)
        return warp, _accepted_diagnostics(
            warp, fixed_mask, moving_mask, fixed, moving, kwargs["source_image_sha256"]
        )

    monkeypatch.setattr(TRACKER, "run_diffeomorphic_registration", accept)
    messages = queue.SimpleQueue()
    transform, diagnostics = TRACKER.fit_slice_anatomy_to_atlas(
        (),
        prepared[0][4],
        1,
        2.0,
        -3.0,
        atlas,
        annotation,
        "model.onnx",
        messages,
        threading.Event(),
    )

    assert transform.nonlinear is not None
    assert diagnostics["status"] == "accepted"
    assert np.array_equal(captured["fixed"], TRACKER.coronal_oblique_slice(atlas, 1, 2.0, -3.0, order=1))
    assert np.array_equal(captured["moving"], image)
    assert np.array_equal(captured["fixed_mask"], mask)
    assert np.array_equal(captured["moving_mask"], mask)
    progress = []
    while not messages.empty():
        progress.append(messages.get()[0])
    assert 35 in progress


@pytest.mark.parametrize(
    ("category", "mapping_blocking"),
    (("correspondence", False), ("wrong_plane", True), ("affine_input", True)),
)
def test_worker_rejection_keeps_the_frozen_affine_transform(
    monkeypatch, category, mapping_blocking
):
    _, image, mask, _prediction, prepared, _runtime = _worker_inputs()
    atlas = np.zeros((4, *image.shape), dtype=np.float32)
    annotation = np.ones_like(atlas, dtype=np.uint16)
    affine = prepared[0][4]
    monkeypatch.setattr(
        TRACKER,
        "prepare_pose_inputs",
        lambda *_args: (
            ["slice_0000.png"],
            {"slice_0000.png": {"source_image_sha256": "c" * 64}},
            {"slice_0000.png": {"image": image, "brain_mask": mask}},
        ),
    )

    def reject(*_args, **_kwargs):
        raise TRACKER.DiffeomorphicRegistrationRejected(
            [(category, "pair rejected")], {"rejection_probability": 0.9}
        )

    monkeypatch.setattr(TRACKER, "run_diffeomorphic_registration", reject)
    messages = queue.SimpleQueue()
    transform, diagnostics = TRACKER.fit_slice_anatomy_to_atlas(
        (), affine, 1, 2.0, -3.0, atlas, annotation, "model.onnx", messages, threading.Event()
    )

    assert transform is affine
    assert transform.nonlinear is None
    assert diagnostics["status"] == "rejected"
    assert diagnostics["reason"] == "pair rejected"
    assert diagnostics["rejection_categories"] == [category]
    assert diagnostics["mapping_blocking"] is mapping_blocking
    assert not messages.empty()


def test_mapping_blocking_rejection_blocks_export_but_other_rejection_requires_review(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        trace = {"imec0": TRACKER.ProbeTrace(volume_points=[[1.0, 2.0, 3.0]])}
        session = TRACKER.SliceSession(
            "slice",
            auto_alignment_diagnostics={
                "nonlinear_refinement": {"status": "rejected", "mapping_blocking": True}
            },
            probe_traces=trace,
        )
        window.sessions = [session]
        arguments = (
            tmp_path, "imec0", "y0_contact",
            np.ones(3), np.ones(3), np.array([0.0, -1.0, 0.0]),
        )
        with pytest.raises(RuntimeError, match="pose/input"):
            window._write_manifest(*arguments)

        session.auto_alignment_diagnostics["nonlinear_refinement"] = {
            "status": "rejected", "mapping_blocking": False
        }
        with pytest.raises(RuntimeError, match="explicitly reviewed"):
            window._write_manifest(*arguments)
        session.auto_alignment_diagnostics["nonlinear_refinement"][
            "affine_review_acknowledged_at"
        ] = "2026-08-11T00:00:00+02:00"
        window._write_manifest(*arguments)
    finally:
        window.close()
        app.processEvents()


def test_mapping_blocked_points_remain_stored_but_do_not_enter_live_trajectory(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        blocked_trace = TRACKER.ProbeTrace(
            volume_points=[[100.0, 100.0, 100.0]], signal_values=[255.0]
        )
        valid_trace = TRACKER.ProbeTrace(
            volume_points=[[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]], signal_values=[10.0, 20.0]
        )
        window.sessions = [
            TRACKER.SliceSession(
                "blocked",
                auto_alignment_diagnostics={
                    "nonlinear_refinement": {"mapping_blocking": True}
                },
                probe_traces={"imec0": blocked_trace},
            ),
            TRACKER.SliceSession("valid", probe_traces={"imec0": valid_trace}),
        ]

        assert blocked_trace.volume_points == [[100.0, 100.0, 100.0]]
        assert np.array_equal(
            window.all_probe_volume_points("imec0"),
            np.asarray(valid_trace.volume_points),
        )
        assert np.array_equal(
            window.all_probe_signal_values("imec0"),
            np.asarray(valid_trace.signal_values),
        )
        center, _ = window.probe_regression("imec0")
        assert np.allclose(center, [2.0, 2.0, 3.0])
    finally:
        window.close()
        app.processEvents()


def test_changed_source_invalidates_alignment_when_slice_is_reloaded(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        source = tmp_path / "slice.tif"
        TRACKER.tifffile.imwrite(source, np.zeros((12, 16), dtype=np.uint8))
        session = TRACKER.SliceSession(
            "slice",
            path=str(source),
            slice_atlas_transform=TRACKER.SliceAtlasTransform2D(
                np.eye(3), (12, 16), (12, 16)
            ),
            auto_alignment_run_id="run",
            auto_alignment_diagnostics={"alignment_run_id": "run"},
            alignment_source_sha256=TRACKER.file_sha256(source),
            probe_traces={
                "imec0": TRACKER.ProbeTrace(
                    atlas_points=[(1.0, 2.0)], volume_points=[[1.0, 2.0, 3.0]]
                )
            },
        )
        window.sessions = [session]
        TRACKER.tifffile.imwrite(source, np.ones((12, 16), dtype=np.uint8))

        with pytest.raises(RuntimeError, match="source image changed"):
            window._write_manifest(
                tmp_path,
                "imec0",
                "y0_contact",
                np.ones(3),
                np.ones(3),
                np.array([0.0, -1.0, 0.0]),
            )
        assert window._load_session_image(session)
        assert session.slice_atlas_transform is None
        assert session.alignment_source_sha256 is None
        assert not session.probe_traces["imec0"].volume_points
    finally:
        window.close()
        app.processEvents()


def test_brightness_preserves_composite_geometry_but_rotation_invalidates_it(tmp_path):
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (12, 16)
        image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        transform = TRACKER.SliceAtlasTransform2D(np.eye(3), shape, shape)
        session = TRACKER.SliceSession(
            "slice",
            raw_display=image,
            adjusted=image,
            rotated=image,
            weight_image=image,
            slice_atlas_transform=transform,
        )
        window.sessions = [session]
        window.current_session_index = 0
        window.atlas_volume = np.zeros((4, *shape), dtype=np.uint8)
        window.annotation_volume = np.ones((4, *shape), dtype=np.uint8)
        window.current_atlas_image = image
        window._curve_changed([(0.0, 0.0), (255.0, 220.0)])
        assert session.slice_atlas_transform is transform
        assert session.transformed_overlay is not None

        window._apply_slice_geometry(5.0, False, False)
        assert session.slice_atlas_transform is None
    finally:
        window.close()
        app.processEvents()


def test_mapping_manifest_references_a_checksum_verified_nonlinear_sidecar(tmp_path, monkeypatch):
    from nonlinear_registration import NonlinearWarp2D

    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (12, 16)
        mask = np.ones(shape, dtype=bool)
        source_path = tmp_path / "slice.tif"
        TRACKER.tifffile.imwrite(source_path, np.arange(np.prod(shape), dtype=np.uint8).reshape(shape))
        window.atlas_volume = np.zeros((4, *shape), dtype=np.float32)
        warp = NonlinearWarp2D.identity(shape)
        transform, runtime = _bound_transform(
            window, source_path, 1, warp, mask
        )
        monkeypatch.setattr(
            TRACKER,
            "verify_diffeomorphic_model_bundle",
            lambda _path: ("a" * 64, "b" * 64, {}),
        )
        session = TRACKER.SliceSession(
            "slice",
            path=str(source_path),
            atlas_index=1,
            slice_atlas_transform=transform,
            auto_alignment_run_id="accepted-run",
            auto_alignment_diagnostics={"nonlinear_refinement": {"status": "accepted"}},
            alignment_source_sha256=runtime["source_image_sha256"],
        )
        window.sessions = [session]
        window._verify_nonlinear_binding(session, transform)
        window.atlas_volume[1, 0, 0] = 1.0
        with pytest.raises(RuntimeError, match="atlas plane"):
            window._verify_nonlinear_binding(session, transform)
        window.atlas_volume[1, 0, 0] = 0.0
        session.flip_horizontal = True
        with pytest.raises(RuntimeError, match="affine slice input"):
            window._verify_nonlinear_binding(session, transform)
        session.flip_horizontal = False
        window._write_manifest(
            tmp_path,
            "imec0",
            "y0_contact",
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, -1.0, 0.0]),
        )

        manifest_path = tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec0.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference = manifest["slices"][0]["slice_atlas_transform"]["sidecar"]
        sidecar_path = manifest_path.parent / reference["relative_path"]
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == reference["sha256"]
        restored = TRACKER.SliceAtlasTransform2D.load_npz(sidecar_path)
        assert restored.nonlinear is not None
        assert restored.nonlinear_attestation.model_sha256 == "a" * 64

        session.auto_alignment_run_id = "replacement-run"
        window._write_manifest(
            tmp_path,
            "imec0",
            "y0_contact",
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, -1.0, 0.0]),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacement = manifest["slices"][0]["slice_atlas_transform"]["sidecar"]
        assert replacement["relative_path"] == reference["relative_path"]
        assert len(list(sidecar_path.parent.glob("*.npz"))) == 1

        row = {"probe_name": "imec0", "probe_channel_number": 0}
        row.update({name: 1 for name in TRACKER.ANATOMY_MAPPING_COLUMNS})
        channels = TRACKER.pd.DataFrame([row])
        units = TRACKER.pd.DataFrame([{"unit_key": "imec0:0", **row}])
        channels.to_csv(tmp_path / "channels.csv", index=False)
        units.to_csv(tmp_path / "units.csv", index=False)
        TRACKER.write_anatomy_sidecars(tmp_path, channels, units)
        TRACKER.verify_staged_mapping_outputs(tmp_path, 1, 1, "imec0")

        for field, invalid in (
            ("coordinate_convention", "wrong convention"),
            ("model_sha256", "c" * 64),
            ("manifest_sha256", "d" * 64),
            ("source_image_sha256", "e" * 64),
            ("atlas_image_sha256", "f" * 64),
            ("moving_affine_sha256", "0" * 64),
            ("runtime_gate_version", 99),
            ("pixel_spacing_um", 50.0),
        ):
            changed = json.loads(json.dumps(manifest))
            changed["slices"][0]["slice_atlas_transform"]["sidecar"][field] = invalid
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with pytest.raises(RuntimeError, match="metadata disagrees"):
                TRACKER.verify_staged_mapping_outputs(tmp_path, 1, 1, "imec0")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        monkeypatch.setattr(
            TRACKER,
            "verify_diffeomorphic_model_bundle",
            lambda _path: ("c" * 64, "d" * 64, {}),
        )
        rejected_folder = tmp_path / "rejected"
        rejected_folder.mkdir()
        with pytest.raises(RuntimeError, match="installed model bundle"):
            window._write_manifest(
                rejected_folder,
                "imec0",
                "y0_contact",
                np.ones(3),
                np.ones(3),
                np.array([0.0, -1.0, 0.0]),
            )
    finally:
        window.close()
        app.processEvents()


def test_nonidentity_warp_has_one_overlay_probe_volume_and_export_convention(tmp_path, monkeypatch):
    from nonlinear_registration import NonlinearWarp2D

    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        shape = (65, 64)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        mask = np.zeros(shape, dtype=bool)
        mask[2:-2, 2:-2] = True
        shift = (
            0.8
            * np.sin(np.pi * xx / (shape[1] - 1.0)) ** 2
            * (
                np.sin(2.0 * np.pi * yy / (shape[0] - 1.0))
                - 2.0 * np.sin(4.0 * np.pi * yy / (shape[0] - 1.0))
            )
            * mask
        )
        basis = np.stack(
            (
                np.ones_like(xx),
                xx * (2.0 / (shape[1] - 1.0)) - 1.0,
                yy * (2.0 / (shape[0] - 1.0)) - 1.0,
            ),
            axis=-1,
        )
        affine = np.linalg.lstsq(basis[mask], shift[mask], rcond=None)[0]
        shift = (shift - basis @ affine) * mask
        inverse_x = xx.copy()
        for _ in range(50):
            sampled_shift = TRACKER.cv2.remap(
                shift.astype(np.float32),
                inverse_x.astype(np.float32),
                yy,
                TRACKER.cv2.INTER_LINEAR,
                borderMode=TRACKER.cv2.BORDER_REPLICATE,
            )
            inverse_x = xx - sampled_shift
        warp = NonlinearWarp2D(
            np.stack((xx + shift, yy), axis=-1),
            np.stack((inverse_x, yy), axis=-1),
        )
        expected_atlas = np.array([32.0, 16.0])
        image = np.broadcast_to(np.arange(shape[1], dtype=np.float32), shape).copy()
        source_image = np.random.default_rng(9).integers(0, 256, shape, dtype=np.uint8)
        source_path = tmp_path / "slice.tif"
        TRACKER.tifffile.imwrite(source_path, source_image)
        window.atlas_volume = np.zeros((40, *shape), dtype=np.uint8)
        normalized = TRACKER.normalize_u8(source_image)
        window.atlas_volume[20] = TRACKER.cv2.remap(
            normalized,
            warp.atlas_to_affine_xy[..., 0],
            warp.atlas_to_affine_xy[..., 1],
            TRACKER.cv2.INTER_LINEAR,
            borderMode=TRACKER.cv2.BORDER_CONSTANT,
        )
        transform, runtime = _bound_transform(window, source_path, 20, warp, mask)
        monkeypatch.setattr(
            TRACKER,
            "verify_diffeomorphic_model_bundle",
            lambda _path: ("a" * 64, "b" * 64, {}),
        )
        display_point = tuple(transform.map_atlas_to_display(expected_atlas[None])[0])
        session = TRACKER.SliceSession(
            "slice",
            path=str(source_path),
            raw_display=image,
            adjusted=image,
            rotated=image,
            weight_image=image,
            atlas_index=20,
            slice_atlas_transform=transform,
            auto_alignment_run_id="nonidentity-run",
            auto_alignment_diagnostics={"nonlinear_refinement": {"status": "accepted"}},
            alignment_source_sha256=runtime["source_image_sha256"],
            probe_traces={"imec0": TRACKER.ProbeTrace(slice_points=[display_point])},
        )
        window.sessions = [session]

        overlay = TRACKER.render_session_slice_in_atlas(session, image, shape)
        window._recompute_probe_points_from_slice_points(session)
        trace = session.probe_traces["imec0"]
        expected_volume = TRACKER.point_to_volume(
            expected_atlas, "coronal", 20, window.atlas_volume.shape
        )

        assert abs(display_point[0] - expected_atlas[0]) > 0.5
        assert overlay[16, 32] == pytest.approx(display_point[0], abs=0.05)
        assert np.allclose(trace.atlas_points[0], expected_atlas, atol=0.03)
        assert np.allclose(trace.volume_points[0], expected_volume, atol=0.03)
        assert np.allclose(window.all_probe_volume_points("imec0")[0], expected_volume, atol=0.03)

        window._write_manifest(
            tmp_path,
            "imec0",
            "y0_contact",
            expected_volume,
            expected_volume,
            np.array([0.0, -1.0, 0.0]),
        )
        manifest_path = tmp_path / "anatomy" / "proprietary_trajectory_manifest_imec0.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        slice_record = manifest["slices"][0]
        sidecar = manifest_path.parent / slice_record["slice_atlas_transform"]["sidecar"]["relative_path"]
        restored = TRACKER.SliceAtlasTransform2D.load_npz(sidecar)

        assert np.allclose(slice_record["probe_atlas_points"][0], expected_atlas, atol=0.03)
        assert np.allclose(slice_record["probe_volume_points"][0], expected_volume, atol=0.03)
        assert np.allclose(
            restored.map_display_to_atlas(np.asarray([display_point]))[0],
            expected_atlas,
            atol=0.03,
        )
        assert restored.render_display_image_in_atlas(image)[16, 32] == pytest.approx(
            display_point[0], abs=0.05
        )
    finally:
        window.close()
        app.processEvents()


def test_no_promoted_bundle_disables_dedicated_anatomical_fit(tmp_path):
    if TRACKER.NONLINEAR_MODEL_PATH.is_file():
        pytest.skip("A promoted nonlinear bundle is installed")
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        assert not window.fit_anatomy_btn.isEnabled()
        assert "unavailable" in window.nonlinear_model_status.text()
        window._set_auto_constraint_controls_enabled(False)
        window._set_auto_constraint_controls_enabled(True)
        assert not window.fit_anatomy_btn.isEnabled()
        image = np.zeros((12, 16), dtype=np.uint8)
        outline = [(float(index), float(index % 4)) for index in range(8)]
        window.atlas_volume = np.zeros((4, 12, 16), dtype=np.uint8)
        window.annotation_volume = np.ones((4, 12, 16), dtype=np.uint8)
        window.sessions = [
            TRACKER.SliceSession(
                "slice",
                raw_display=image,
                rotated=image,
                weight_image=image,
                brain_outline_points=outline,
                slice_atlas_transform=TRACKER.SliceAtlasTransform2D(
                    np.eye(3), image.shape, image.shape
                ),
            )
        ]
        window.current_session_index = 0
        with pytest.raises(RuntimeError, match="Diffeomorphic ONNX model is unavailable"):
            window._fit_current_slice_anatomy()
        assert not window.auto_alignment_busy
    finally:
        window.close()
        app.processEvents()


def test_source_approved_bundle_enables_dedicated_fit_for_an_affine_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TRACKER,
        "verify_diffeomorphic_model_bundle",
        lambda _path: ("a" * 64, "b" * 64, {}),
    )
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        image = np.zeros((12, 16), dtype=np.uint8)
        window.atlas_volume = np.zeros((4, *image.shape), dtype=np.uint8)
        window.annotation_volume = np.ones((4, *image.shape), dtype=np.uint8)
        window.sessions = [
            TRACKER.SliceSession(
                "slice",
                slice_atlas_transform=TRACKER.SliceAtlasTransform2D(
                    np.eye(3), image.shape, image.shape
                ),
            )
        ]
        window.current_session_index = 0
        window._update_nonlinear_fit_button()
        assert window.fit_anatomy_btn.isEnabled()
        window._set_auto_constraint_controls_enabled(False)
        assert not window.fit_anatomy_btn.isEnabled()
        window._set_auto_constraint_controls_enabled(True)
        assert window.fit_anatomy_btn.isEnabled()
    finally:
        window.close()
        app.processEvents()


def test_mapping_promotion_failure_rolls_back_every_output(tmp_path, monkeypatch):
    from nonlinear_registration import NonlinearWarp2D

    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    window = TRACKER.TrajectoryTrackerWindow(default_atlas_folder=tmp_path / "missing-atlas")
    try:
        channels_path = tmp_path / "channels.csv"
        units_path = tmp_path / "units.csv"
        channels = TRACKER.pd.DataFrame(
            {
                "probe_name": ["imec0", "imec0"],
                "probe_channel_number": [0, 1],
                "probe_horizontal_position": [0.0, 0.0],
                "probe_vertical_position": [0.0, 25.0],
            }
        )
        units = TRACKER.pd.DataFrame(
            {
                "unit_key": ["imec0:0", "imec0:1"],
                "unit_id": [0, 1],
                "probe_name": ["imec0", "imec0"],
                "probe_channel_number": [0, 1],
            }
        )
        channels.to_csv(channels_path, index=False)
        units.to_csv(units_path, index=False)
        original_channels = channels_path.read_bytes()
        original_units = units_path.read_bytes()

        anatomy_dir = tmp_path / "anatomy"
        transform_dir = anatomy_dir / "slice_atlas_transforms"
        transform_dir.mkdir(parents=True)
        old_outputs = {
            anatomy_dir / "channel_brain_regions.csv": b"old channel provenance\n",
            anatomy_dir / "unit_brain_region_assignments.csv": b"old unit provenance\n",
            anatomy_dir / "proprietary_trajectory_manifest_imec0.json": b'{"old": true}\n',
        }

        shape = (30, 30)
        mask = np.ones(shape, dtype=bool)
        source_path = tmp_path / "slice.tif"
        TRACKER.tifffile.imwrite(source_path, np.arange(np.prod(shape), dtype=np.uint16).reshape(shape))
        window.atlas_volume = np.zeros((30, *shape), dtype=np.uint8)
        window.annotation_volume = np.ones((30, *shape), dtype=np.uint16)
        warp = NonlinearWarp2D.identity(shape)
        transform, runtime = _bound_transform(
            window, source_path, 10, warp, mask
        )
        monkeypatch.setattr(
            TRACKER,
            "verify_diffeomorphic_model_bundle",
            lambda _path: ("a" * 64, "b" * 64, {}),
        )
        session = TRACKER.SliceSession(
            "slice",
            path=str(source_path),
            atlas_index=10,
            slice_atlas_transform=transform,
            auto_alignment_run_id="accepted-run",
            auto_alignment_diagnostics={"nonlinear_refinement": {"status": "accepted"}},
            alignment_source_sha256=runtime["source_image_sha256"],
            probe_traces={
                "imec0": TRACKER.ProbeTrace(
                    volume_points=[[10.0, 20.0, 10.0], [10.0, 15.0, 10.0]],
                )
            },
        )
        identity = hashlib.sha256(
            f"0|{source_path.resolve()}".encode()
        ).hexdigest()[:16]
        old_transform_path = transform_dir / f"slice_atlas_transform_{identity}.npz"
        old_outputs[old_transform_path] = b"old nonlinear provenance\n"
        for path, content in old_outputs.items():
            path.write_bytes(content)

        window.sessions = [session]
        window.bregma_voxel = np.array([10.0, 10.0, 10.0])
        window.region_names = {1: ("Brain", "BR")}
        window.run_folder.setText(str(tmp_path))
        window.probe_name.addItem("imec0")
        window.probe_name.setCurrentText("imec0")
        window.endpoint_reference.setCurrentIndex(0)

        original_promote = TRACKER.promote_staged_mapping_outputs

        def fail_during_promotion(staging_root, data_folder, backup_dir, staged_hashes):
            def injected_replace(source, destination):
                if Path(destination) == units_path:
                    raise OSError("injected promotion failure")
                os.replace(source, destination)

            return original_promote(
                staging_root,
                data_folder,
                backup_dir,
                staged_hashes,
                replace_file=injected_replace,
            )

        monkeypatch.setattr(TRACKER, "promote_staged_mapping_outputs", fail_during_promotion)
        with pytest.raises(OSError, match="injected promotion failure"):
            window.map_channels_units()

        assert channels_path.read_bytes() == original_channels
        assert units_path.read_bytes() == original_units
        for path, content in old_outputs.items():
            assert path.read_bytes() == content
        assert not list(tmp_path.glob(".trajectory_mapping_*"))
        mapping_files = {
            path.relative_to(anatomy_dir)
            for path in anatomy_dir.rglob("*")
            if path.is_file() and "backups" not in path.parts
        }
        assert mapping_files == {
            Path("channel_brain_regions.csv"),
            Path("unit_brain_region_assignments.csv"),
            Path("proprietary_trajectory_manifest_imec0.json"),
            Path("slice_atlas_transforms") / old_transform_path.name,
        }
        backup = max((anatomy_dir / "backups").iterdir()) / "replaced"
        assert (backup / "channels.csv").read_bytes() == original_channels
        assert (backup / "units.csv").read_bytes() == original_units
        for path, content in old_outputs.items():
            assert (backup / path.relative_to(tmp_path)).read_bytes() == content
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
        Path("anatomy/slice_atlas_transforms/transform.npz"),
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
