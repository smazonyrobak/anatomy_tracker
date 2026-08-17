import importlib.util
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from source import deepslice_runtime as DEEPSLICE_RUNTIME
from source.probe_constraints import direction_from_attack_angle


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

    monkeypatch.setattr(DEEPSLICE_RUNTIME, "load_deepslice_onnx_sessions", load_sessions)
    monkeypatch.setattr(
        DEEPSLICE_RUNTIME,
        "preprocess_deepslice_images",
        lambda _paths: (np.zeros((1, 299, 299, 3), dtype=np.float32), [456], [320]),
    )

    records, _, _, _, runtime = DEEPSLICE_RUNTIME.run_deepslice_inference(
        ["slice.png"],
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
        DEEPSLICE_RUNTIME,
        "load_deepslice_onnx_sessions",
        lambda *_args: (sessions, {}, "DmlExecutionProvider", None),
    )
    monkeypatch.setattr(
        DEEPSLICE_RUNTIME,
        "preprocess_deepslice_images",
        lambda _paths: (np.zeros((1, 299, 299, 3), dtype=np.float32), [456], [320]),
    )

    with pytest.raises(InterruptedError):
        DEEPSLICE_RUNTIME.run_deepslice_inference(["slice.png"], queue.SimpleQueue(), cancel)

    assert [sessions["primary"].calls, sessions["secondary"].calls] == [1, 0]


def test_validated_deepslice_runtime_recovers_known_atlas_planes():
    assert Path(TRACKER.run_deepslice_inference.__code__.co_filename).resolve() == (
        ROOT / "source" / "deepslice_runtime.py"
    ).resolve()
    expected_indices = np.asarray([120.0, 216.0, 320.0])
    image_paths = [
        str(ROOT / "tests" / "data" / f"allen_average_coronal_{index}.png")
        for index in expected_indices.astype(int)
    ]

    records, _, hashes, _, runtime = TRACKER.run_deepslice_inference(
        image_paths, queue.SimpleQueue(), threading.Event()
    )
    by_filename = {
        record["Filenames"]: TRACKER.quicknii_to_tracker_alignment(record, TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML)[:3]
        for record in records
    }
    alignments = np.asarray([by_filename[Path(path).name] for path in image_paths])

    assert hashes == DEEPSLICE_RUNTIME.DEEPSLICE_ONNX_SHA256
    assert runtime["backend"] in {"ONNX Runtime DirectML", "ONNX Runtime CPU"}
    assert np.max(np.abs(alignments[:, 0] - expected_indices)) < 5.0
    assert np.max(np.abs(alignments[:, 1:])) < TRACKER.DEEPSLICE_REVIEW_TILT_DEG


def test_real_deepslice_predictions_flow_into_physical_constraint_solver():
    expected_indices = np.asarray([120, 216, 320])
    image_paths = [
        str(ROOT / "tests" / "data" / f"allen_average_coronal_{index}.png")
        for index in expected_indices
    ]
    records, _, _, _, _ = TRACKER.run_deepslice_inference(
        image_paths, queue.SimpleQueue(), threading.Event()
    )
    predicted = {
        Path(record["Filenames"]).name: TRACKER.quicknii_to_tracker_alignment(
            record, TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML
        )[0]
        for record in records
    }
    model_indices = [
        predicted[f"allen_average_coronal_{index}.png"] for index in expected_indices
    ]
    lattices = {
        session: {
            int(expected): (0.02, {}),
            int(round(model)): (0.0, {}),
        }
        for session, (expected, model) in enumerate(zip(expected_indices, model_indices))
    }
    bregma = np.asarray([216.0, 160.0, 228.0])
    entry = np.asarray([2400.0, 0.0, 0.0])
    direction = direction_from_attack_angle(20.0, 180.0)
    volume_entry = TRACKER.probe_stereotaxic_to_volume(entry, bregma, TRACKER.VOXEL_UM)
    volume_direction = direction / (
        TRACKER.VOXEL_UM * TRACKER.STEREOTAXIC_AXIS_SIGN_AP_DV_ML
    )
    points = {}
    for session, atlas_index in enumerate(expected_indices):
        intersection = volume_entry + (
            (float(atlas_index) - volume_entry[0]) / volume_direction[0]
        ) * volume_direction
        points[session] = np.asarray([[intersection[2], intersection[1]]])

    assignment, _, diagnostics = TRACKER.solve_probe_constrained_lattice(
        lattices,
        [0, 1, 2],
        {
            "imec0": {
                "constraint": TRACKER.ProbeInsertionConstraint(
                    True, entry[0], entry[2], 25.0, 20.0, 1.0, 8000.0
                )
            }
        },
        lambda _probe, session, _ap, _lr, _dv: points[session],
        lambda _ap, _ml: 0.0,
        bregma,
        TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML,
        0.0,
        0.0,
    )

    errors = np.abs(np.asarray([assignment[index] for index in range(3)]) - expected_indices)
    assert errors.tolist() == [0, 0, 2]
    assert diagnostics["probes"]["imec0"]["angle_deg"] == pytest.approx(20.0, abs=1.0)


def test_smart_selection_crop_and_surface_fit_recover_known_oblique_plane():
    source_path = ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_source_fliph.png"
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    oriented, _ = TRACKER.transform_slice_image(source, 0.0, False, False)
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
    records, _, runtime_info, _ = TRACKER.prepare_and_run_pose_predictions(
        [(str(source_path), 0.0, False, False, points, crop, True, selection)],
        TRACKER.POSE_ENGINE_DEEPSLICE,
        TRACKER.DEFAULT_OWN_CNN_WEIGHT,
        float(TRACKER.DEFAULT_BREGMA_VOXEL_AP_DV_ML[0]),
        messages,
        cancel,
    )
    mask = cv2.imread(
        str(ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    atlas_index = records[0]["predicted_atlas_index"]
    tilt_lr = records[0]["predicted_tilt_lr_deg"]
    tilt_dv = records[0]["predicted_tilt_dv_deg"]
    matrix = np.asarray(records[0]["initial_slice_to_atlas"])
    matrix, diagnostics = TRACKER.fit_surface_scale_translation(matrix, points, mask)

    assert abs(atlas_index - 280) < 10
    assert abs(tilt_lr - 6.0) < 5.0
    assert abs(tilt_dv + 4.0) < 5.0
    matrix = TRACKER.orientation_preserving_slice_to_atlas(matrix, selection, mask)
    assert np.linalg.det(matrix[:2, :2]) > 0.0
    assert diagnostics["rms_after_atlas_px"] < 3.0


def test_bounded_mind_search_recovers_known_pose_instead_of_clipping_to_bound():
    atlas_folder = ROOT / "data" / "Allen Brain Atlas 25um"
    if not (atlas_folder / "average_template_25.nrrd").exists():
        pytest.skip("Local Allen atlas is not installed")

    source = cv2.imread(
        str(ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_source_fliph.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    oriented, _ = TRACKER.transform_slice_image(source, 0.0, True, False)
    _, selection = TRACKER.smart_brain_surface_selection(
        oriented,
        [(330.0, 240.0), (220.0, 240.0), (440.0, 240.0)],
        [(20.0, 20.0), (680.0, 480.0)],
        50.0,
        50,
    )
    atlas = TRACKER.nrrd.read(str(atlas_folder / "average_template_25.nrrd"))[0]
    annotation = TRACKER.nrrd.read(str(atlas_folder / "annotation_25.nrrd"))[0]

    pose, diagnostics, _ = TRACKER.refine_pose_search(
        {0: (250.0, 0.0, 0.0, np.eye(3))},
        {0: {"Filenames": "known.png"}},
        atlas,
        annotation,
        {"known.png": {"image": oriented, "brain_mask": selection}},
        {"known.png": {"ap_um": 800.0, "lr_deg": 8.0, "dv_deg": 8.0}},
        (260, 300),
        [],
        None,
        threading.Event(),
        global_alignment=False,
    )

    assert pose[0][0] == 280
    assert abs(pose[0][1] - 6.0) <= 0.25
    assert abs(pose[0][2] + 4.0) <= 0.25
    assert not diagnostics[0]["pose_search_boundary"]
    assert diagnostics[0]["pose_search_margin"] > 0.01


def test_global_mind_search_recovers_ordered_planes_with_one_shared_tilt():
    atlas_folder = ROOT / "data" / "Allen Brain Atlas 25um"
    if not (atlas_folder / "average_template_25.nrrd").exists():
        pytest.skip("Local Allen atlas is not installed")

    atlas = TRACKER.nrrd.read(str(atlas_folder / "average_template_25.nrrd"))[0]
    annotation = TRACKER.nrrd.read(str(atlas_folder / "annotation_25.nrrd"))[0]
    known = {0: (275, 5.0, -3.0), 1: (287, 5.0, -3.0)}
    prepared_inputs = {}
    records = {}
    converted = {}
    disagreement = {}
    for index, (ap, tilt_lr, tilt_dv) in known.items():
        filename = f"known_{index}.png"
        prepared_inputs[filename] = {
            "image": TRACKER.coronal_oblique_slice_resampled(
                atlas, ap, tilt_lr, tilt_dv, order=1
            ),
            "brain_mask": TRACKER.coronal_oblique_slice_resampled(
                annotation, ap, tilt_lr, tilt_dv, order=0
            )
            > 0,
        }
        records[index] = {"Filenames": filename}
        converted[index] = (292.0 - 22.0 * index, 0.0, 0.0, np.eye(3))
        disagreement[filename] = {"ap_um": 800.0, "lr_deg": 8.0, "dv_deg": 8.0}

    pose, _, shared_tilt = TRACKER.refine_pose_search(
        converted,
        records,
        atlas,
        annotation,
        prepared_inputs,
        disagreement,
        (265, 297),
        [0, 1],
        None,
        threading.Event(),
        global_alignment=True,
    )

    assert pose[0][0] < pose[1][0]
    assert pose[0][0] == pytest.approx(known[0][0], abs=1)
    assert pose[1][0] == pytest.approx(known[1][0], abs=1)
    assert shared_tilt == pytest.approx((5.0, -3.0), abs=1.0)
    assert pose[0][1:] == pose[1][1:] == shared_tilt
