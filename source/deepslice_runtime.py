from __future__ import annotations

import hashlib
import queue
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


RESOURCE_DIR = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
VOXEL_UM = 25.0
ALLEN_CCF_25_SHAPE_AP_DV_ML = (528, 320, 456)
DEEPSLICE_VERSION = "1.2.8"
DEEPSLICE_ONNX_SHA256 = {
    "primary": "90ce8d4662f53a602035a99d5145c0e6ae8924cde7f9de440cf6b74f79c791ac",
    "secondary": "2d7b5e44d9dc4aa6009df6c3cc7e8a0cbb9fd33dc63a8bd2ac43ea5999237978",
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quicknii_to_tracker_alignment(
    prediction: dict,
    atlas_shape: tuple[int, int, int],
) -> tuple[float, float, float, np.ndarray]:
    origin = np.asarray(
        [prediction["ox"], prediction["oy"], prediction["oz"]], dtype=np.float64
    )
    horizontal = np.asarray(
        [prediction["ux"], prediction["uy"], prediction["uz"]], dtype=np.float64
    )
    vertical = np.asarray(
        [prediction["vx"], prediction["vy"], prediction["vz"]], dtype=np.float64
    )
    normal = np.cross(horizontal, vertical)
    if normal[1] < 0:
        normal = -normal
    if abs(normal[1]) < 1e-9:
        raise ValueError("DeepSlice returned a non-coronal plane")
    ap_per_ml = normal[0] / normal[1]
    ap_per_dv = -normal[2] / normal[1]
    center_ml = (atlas_shape[2] - 1) / 2.0
    center_dv = (atlas_shape[1] - 1) / 2.0
    origin_ml = origin[0]
    origin_ap = atlas_shape[0] - origin[1]
    origin_dv = atlas_shape[1] - origin[2]
    index = (
        origin_ap
        + ap_per_ml * (center_ml - origin_ml)
        + ap_per_dv * (center_dv - origin_dv)
    )
    width = float(prediction["width"])
    height = float(prediction["height"])
    matrix = np.asarray(
        [
            [horizontal[0] / width, vertical[0] / height, origin_ml],
            [-horizontal[2] / width, -vertical[2] / height, origin_dv],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (
        float(index),
        float(np.degrees(np.arctan(ap_per_ml))),
        float(np.degrees(np.arctan(ap_per_dv))),
        matrix,
    )


@lru_cache(maxsize=2)
def load_deepslice_onnx_sessions(force_cpu: bool = False):
    import onnxruntime as ort

    model_dir = RESOURCE_DIR / "models" / "DeepSlice"
    model_paths = {
        "primary": model_dir / "deepslice_mouse_primary_opset18.onnx",
        "secondary": model_dir / "deepslice_mouse_secondary_opset18.onnx",
    }
    model_hashes = {name: _file_sha256(path) for name, path in model_paths.items()}
    if model_hashes != DEEPSLICE_ONNX_SHA256:
        raise RuntimeError("Validated DeepSlice 1.2.8 ONNX model checksum validation failed")

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    use_directml = not force_cpu and "DmlExecutionProvider" in ort.get_available_providers()
    providers = (
        [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if use_directml
        else ["CPUExecutionProvider"]
    )
    fallback_reason = None
    try:
        sessions = {
            name: ort.InferenceSession(str(path), sess_options=options, providers=providers)
            for name, path in model_paths.items()
        }
    except Exception as exc:
        if not use_directml:
            raise
        fallback_reason = f"DirectML initialization failed: {type(exc).__name__}: {exc}"
        sessions = {
            name: ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name, path in model_paths.items()
        }
    for session in sessions.values():
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        if model_input.name != "images" or model_input.shape[1:] != [299, 299, 3]:
            raise RuntimeError("DeepSlice ONNX input contract does not match the validated model")
        if model_output.name != "Identity:0" or model_output.shape[-1] != 9:
            raise RuntimeError("DeepSlice ONNX output contract does not match the validated model")
    provider = sessions["primary"].get_providers()[0]
    return sessions, model_hashes, provider, fallback_reason


def preprocess_deepslice_images(
    image_paths: list[str],
) -> tuple[np.ndarray, list[int], list[int]]:
    images = []
    widths = []
    heights = []
    grayscale_weights = np.asarray([0.2125, 0.7154, 0.0721], dtype=np.float32)
    for path in image_paths:
        with Image.open(path) as image:
            widths.append(image.width)
            heights.append(image.height)
            rgb = np.asarray(
                image.convert("RGB").resize((299, 299), Image.Resampling.NEAREST),
                dtype=np.float32,
            )
        gray = rgb @ grayscale_weights
        gray -= np.mean(gray, keepdims=True)
        gray /= np.std(gray, keepdims=True) + np.float32(1e-6)
        images.append(np.repeat(gray[..., None], 3, axis=-1).astype(np.float32))
    return np.stack(images), widths, heights


def run_deepslice_inference(
    image_paths: list[str],
    progress_messages: queue.SimpleQueue,
    cancel_event: threading.Event,
) -> tuple[list[dict], str, dict[str, str], dict[str, dict[str, float]], dict]:
    import onnxruntime as ort

    progress_messages.put((5, "Loading the validated DeepSlice GPU runtime..."))
    started = time.perf_counter()
    sessions, model_hashes, provider, fallback_reason = load_deepslice_onnx_sessions()
    inputs, widths, heights = preprocess_deepslice_images(image_paths)
    if cancel_event.is_set():
        raise InterruptedError
    progress_messages.put((15, f"Running the DeepSlice two-model ensemble on {provider}..."))
    inference_started = time.perf_counter()
    inference_error = None
    try:
        primary_result = sessions["primary"].run(["Identity:0"], {"images": inputs})
    except Exception as exc:
        inference_error = exc
    if inference_error is None:
        if cancel_event.is_set():
            raise InterruptedError
        try:
            secondary_result = sessions["secondary"].run(["Identity:0"], {"images": inputs})
        except Exception as exc:
            inference_error = exc
    if inference_error is not None:
        if provider != "DmlExecutionProvider":
            raise inference_error
        fallback_reason = (
            f"DirectML inference failed: {type(inference_error).__name__}: {inference_error}"
        )
        progress_messages.put((15, "DirectML failed; retrying the validated models on CPU..."))
        sessions, model_hashes, provider, _ = load_deepslice_onnx_sessions(True)
        primary_result = sessions["primary"].run(["Identity:0"], {"images": inputs})
        if cancel_event.is_set():
            raise InterruptedError
        secondary_result = sessions["secondary"].run(["Identity:0"], {"images": inputs})
    if cancel_event.is_set():
        raise InterruptedError
    primary = primary_result[0].astype(np.float64)
    secondary = secondary_result[0].astype(np.float64)
    inference_seconds = time.perf_counter() - inference_started
    ensemble = np.mean([primary, secondary], axis=0)
    coordinate_columns = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")

    def records_from(values: np.ndarray) -> list[dict]:
        return [
            {
                "Filenames": Path(path).name,
                **{name: float(value) for name, value in zip(coordinate_columns, row)},
                "width": int(width),
                "height": int(height),
            }
            for path, row, width, height in zip(image_paths, values, widths, heights)
        ]

    primary_records = records_from(primary)
    secondary_records = records_from(secondary)
    records = sorted(records_from(ensemble), key=lambda record: record["oy"])
    for record in records:
        record["raw_ensemble_ouv"] = [float(record[column]) for column in coordinate_columns]
    disagreement = {}
    for primary_record, secondary_record in zip(primary_records, secondary_records):
        primary_alignment = quicknii_to_tracker_alignment(
            primary_record, ALLEN_CCF_25_SHAPE_AP_DV_ML
        )
        secondary_alignment = quicknii_to_tracker_alignment(
            secondary_record, ALLEN_CCF_25_SHAPE_AP_DV_ML
        )
        disagreement[primary_record["Filenames"]] = {
            "ap_um": abs(primary_alignment[0] - secondary_alignment[0]) * VOXEL_UM,
            "lr_deg": abs(primary_alignment[1] - secondary_alignment[1]),
            "dv_deg": abs(primary_alignment[2] - secondary_alignment[2]),
        }
    preintegration_tilts = np.asarray(
        [
            quicknii_to_tracker_alignment(record, ALLEN_CCF_25_SHAPE_AP_DV_ML)[1:3]
            for record in records
        ]
    )
    progress_messages.put((27, "Converting DeepSlice coordinates into the tracker atlas..."))
    runtime_info = {
        "backend": (
            "ONNX Runtime DirectML" if provider == "DmlExecutionProvider" else "ONNX Runtime CPU"
        ),
        "onnxruntime_version": ort.__version__,
        "device": "GPU (DirectML device 0)" if provider == "DmlExecutionProvider" else "CPU",
        "gpu_fallback_reason": fallback_reason,
        "inference_seconds": float(inference_seconds),
        "total_backend_seconds": float(time.perf_counter() - started),
        "preintegration_tilt_spread_deg": (
            np.ptp(preintegration_tilts, axis=0).tolist()
            if len(preintegration_tilts) > 1
            else [0.0, 0.0]
        ),
    }
    return records, DEEPSLICE_VERSION, dict(model_hashes), disagreement, runtime_info
