import importlib.util
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_runtime_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def test_validated_deepslice_runtime_recovers_known_atlas_planes():
    expected_indices = np.asarray([120.0, 216.0, 320.0])
    image_paths = [
        str(ROOT / "tests" / "data" / f"allen_average_coronal_{index}.png")
        for index in expected_indices.astype(int)
    ]

    sessions, hashes, provider, _fallback_reason = TRACKER.load_deepslice_onnx_sessions()
    inputs, widths, heights = TRACKER.preprocess_deepslice_images(image_paths)
    primary = sessions["primary"].run(["Identity:0"], {"images": inputs})[0]
    secondary = sessions["secondary"].run(["Identity:0"], {"images": inputs})[0]
    predictions = (primary + secondary) / 2.0
    columns = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
    actual_indices = []
    actual_tilts = []
    for path, row, width, height in zip(image_paths, predictions, widths, heights):
        record = {
            "Filenames": Path(path).name,
            "width": width,
            "height": height,
            **{column: float(value) for column, value in zip(columns, row)},
        }
        alignment = TRACKER.quicknii_to_tracker_alignment(
            record,
            TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML,
        )
        actual_indices.append(alignment[0])
        actual_tilts.append(alignment[1:3])

    assert hashes == TRACKER.DEEPSLICE_ONNX_SHA256
    assert provider in {"DmlExecutionProvider", "CPUExecutionProvider"}
    assert np.max(np.abs(np.asarray(actual_indices) - expected_indices)) < 5.0
    assert np.max(np.abs(np.asarray(actual_tilts))) < TRACKER.DEEPSLICE_REVIEW_TILT_DEG


def test_smart_selection_crop_flip_and_surface_fit_recover_known_oblique_plane(tmp_path):
    source_path = ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_source_fliph.png"
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    oriented, _ = TRACKER.transform_slice_image(source, 0.0, True, False)
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
    model_input = tmp_path / "slice_0000.png"
    messages = queue.SimpleQueue()
    cancel = threading.Event()
    records, _, _, _, runtime_info = TRACKER.prepare_and_run_deepslice(
        [(str(source_path), str(model_input), 0.0, True, False, points, crop, True)],
        False,
        messages,
        cancel,
    )
    mask = cv2.imread(
        str(ROOT / "tests" / "data" / "allen_oblique_ap280_lr6_dv-4_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    annotation = np.broadcast_to(mask, TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML)
    prepared, shared_tilt = TRACKER.solve_deepslice_alignment(
        records,
        {"slice_0000.png": 0},
        TRACKER.ALLEN_CCF_25_SHAPE_AP_DV_ML,
        annotation,
        {0: points},
        None,
        [],
        runtime_info,
        global_alignment=False,
    )

    _, atlas_index, tilt_lr, tilt_dv, *_rest, diagnostics = prepared[0]
    assert shared_tilt is None
    assert abs(atlas_index - 280) < 5
    assert abs(tilt_lr - 6.0) < 10.0
    assert abs(tilt_dv + 4.0) < 10.0
    assert diagnostics["surface_rms_after_atlas_px"] < 3.0
