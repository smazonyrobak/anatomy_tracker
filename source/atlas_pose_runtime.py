from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


POSE_IMAGE_SIZE = 299
POSE_INPUT_NAME = "images"
POSE_OUTPUT_NAME = "pose_ap_um_lr_deg_dv_deg"
POSE_ORIENTATION_OUTPUT_NAME = "orientation_inverted_logit"
POSE_INFERENCE_BATCH_SIZE = 16
ATLAS_POSE_PREPROCESSING_VERSION = "smart-mask-scale-invariant-v1"


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32).mean(axis=-1)
    return image.astype(np.float32, copy=False)


def automatic_brain_mask(image: np.ndarray) -> np.ndarray:
    pixels = np.squeeze(np.asarray(image))
    pixels = pixels[..., None] if pixels.ndim == 2 else pixels[..., :3]
    pixels = pixels.astype(np.float32)
    original_height, original_width = pixels.shape[:2]
    scale = min(1.0, 1024.0 / max(original_height, original_width))
    if scale < 1.0:
        pixels = cv2.resize(
            pixels,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        pixels = pixels[..., None] if pixels.ndim == 2 else pixels
    flat = pixels.reshape(-1, pixels.shape[-1])
    low, high = np.percentile(flat, [0.2, 99.8], axis=0)
    normalized = np.clip((pixels - low) * 255.0 / np.maximum(high - low, 1e-6), 0.0, 255.0)
    height, width = normalized.shape[:2]
    margin = max(3, round(min(height, width) * 0.015))
    border = np.concatenate(
        (
            normalized[:margin].reshape(-1, normalized.shape[-1]),
            normalized[-margin:].reshape(-1, normalized.shape[-1]),
            normalized[:, :margin].reshape(-1, normalized.shape[-1]),
            normalized[:, -margin:].reshape(-1, normalized.shape[-1]),
        )
    )
    background = np.median(border, axis=0)
    border_distance = np.sqrt(np.mean((border - background) ** 2, axis=1))
    distance = np.sqrt(np.mean((normalized - background) ** 2, axis=2))
    threshold = max(3.0, float(np.percentile(border_distance, 75.0)) * 2.0 + 2.0)
    _, otsu_mask = cv2.threshold(
        np.clip(distance, 0.0, 255.0).astype(np.uint8),
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = (distance > threshold).astype(np.uint8)
    if otsu_mask.mean() > 0.45 and (mask.mean() < 0.25 or mask.mean() > 0.90):
        mask = otsu_mask
    radius = max(2, round(min(height, width) / 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening_radius = max(2, round(min(height, width) / 50))
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * opening_radius + 1, 2 * opening_radius + 1),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("No brain foreground was detected")
    areas = stats[1:, cv2.CC_STAT_AREA]
    center = np.asarray([width / 2.0, height / 2.0])
    distance_from_center = np.linalg.norm((centroids[1:] - center) / np.asarray([width, height]), axis=1)
    selected_labels = 1 + np.flatnonzero(
        (areas >= max(0.001 * height * width, 0.15 * areas.max())) & (distance_from_center < 0.55)
    )
    selected = np.isin(labels, selected_labels).astype(np.uint8)
    radius = max(2, round(min(height, width) / 70))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No brain foreground was detected")
    contour_areas = np.asarray([cv2.contourArea(contour) for contour in contours])
    contours = [
        contour
        for contour, area in zip(contours, contour_areas)
        if area >= max(0.001 * height * width, 0.15 * contour_areas.max())
    ]
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, 1, -1)
    if result.shape != (original_height, original_width):
        result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return result.astype(bool)


def brain_orientation_affine(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    if len(x) < 64:
        raise ValueError("The selected brain surface does not enclose enough tissue")
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    angle = np.degrees(0.5 * np.arctan2(2.0 * np.mean(x * y), np.mean(x * x) - np.mean(y * y)))
    center = ((mask.shape[1] - 1.0) / 2.0, (mask.shape[0] - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    corners = np.asarray(
        [[0.0, 0.0, 1.0], [mask.shape[1] - 1.0, 0.0, 1.0], [0.0, mask.shape[0] - 1.0, 1.0],
         [mask.shape[1] - 1.0, mask.shape[0] - 1.0, 1.0]]
    )
    rotated = (matrix @ corners.T).T
    low = rotated.min(axis=0)
    high = rotated.max(axis=0)
    matrix[:, 2] -= low
    size = tuple(np.ceil(high - low + 1.0).astype(int))
    return np.vstack((matrix, [0.0, 0.0, 1.0])), size


def canonicalize_brain_orientation(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = as_gray(image)
    mask = np.asarray(mask, dtype=bool)
    matrix, size = brain_orientation_affine(mask)
    oriented = cv2.warpAffine(gray, matrix[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    oriented_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix[:2],
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    return oriented, oriented_mask


def preprocess_atlas_pose_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray, mask = canonicalize_brain_orientation(image, mask)
    y, x = np.nonzero(mask)
    center_x = (float(x.min()) + float(x.max())) / 2.0
    center_y = (float(y.min()) + float(y.max())) / 2.0
    side = max(float(x.max() - x.min()), float(y.max() - y.min())) * 1.14
    axis = np.linspace(-0.5, 0.5, POSE_IMAGE_SIZE, dtype=np.float32)
    sample_x, sample_y = np.meshgrid(center_x + axis * side, center_y + axis * side)
    canonical = cv2.remap(gray, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    canonical_mask = cv2.remap(
        mask.astype(np.uint8),
        sample_x,
        sample_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    values = canonical[canonical_mask]
    canonical = np.clip((canonical - values.mean()) / max(float(values.std()), 1e-4) / 4.0 + 0.5, 0.0, 1.0)
    canonical[~canonical_mask] = 0.0
    return np.ascontiguousarray(np.repeat(canonical[None].astype(np.float32), 3, axis=0))


def plane_normal_from_tilts(tilt_lr_deg: float, tilt_dv_deg: float) -> np.ndarray:
    normal = np.asarray(
        [-np.tan(np.deg2rad(tilt_lr_deg)), 1.0, -np.tan(np.deg2rad(tilt_dv_deg))],
        dtype=np.float64,
    )
    return normal / np.linalg.norm(normal)


def tilts_from_plane_normal(normal: np.ndarray) -> tuple[float, float]:
    normal = np.asarray(normal, dtype=np.float64)
    if normal[1] < 0.0:
        normal = -normal
    if abs(normal[1]) < 1e-9:
        raise ValueError("The fused prediction is not a coronal plane")
    return (
        float(np.degrees(np.arctan(-normal[0] / normal[1]))),
        float(np.degrees(np.arctan(-normal[2] / normal[1]))),
    )


def fuse_pose_predictions(poses: np.ndarray, weights: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or weights.shape != (len(poses),):
        raise ValueError("Pose fusion expects [N, 3] predictions and N weights")
    if not np.isfinite(poses).all() or not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Pose fusion received invalid predictions or weights")
    weights = weights / weights.sum()
    normal = np.sum(
        np.stack([plane_normal_from_tilts(pose[1], pose[2]) for pose in poses]) * weights[:, None],
        axis=0,
    )
    tilt_lr, tilt_dv = tilts_from_plane_normal(normal)
    return np.asarray([np.dot(weights, poses[:, 0]), tilt_lr, tilt_dv], dtype=np.float64)


def brain_mask_affine(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    orientation_inverted: bool = False,
) -> np.ndarray:
    source_y, source_x = np.nonzero(np.asarray(source_mask, dtype=bool))
    target_y, target_x = np.nonzero(np.asarray(target_mask, dtype=bool))
    if min(len(source_x), len(target_x)) < 64:
        raise ValueError("A brain surface is missing from the slice or predicted atlas plane")
    orientation, oriented_size = brain_orientation_affine(source_mask)
    if orientation_inverted:
        orientation = np.asarray(
            [[-1.0, 0.0, oriented_size[0] - 1.0], [0.0, -1.0, oriented_size[1] - 1.0], [0.0, 0.0, 1.0]]
        ) @ orientation
    source_points = (orientation @ np.column_stack((source_x, source_y, np.ones(len(source_x)))).T).T[:, :2]
    source_span = np.ptp(source_points, axis=0)
    target_span = np.asarray([np.ptp(target_x), np.ptp(target_y)], dtype=np.float64)
    scale = float(np.median(target_span / np.maximum(source_span, 1.0)))
    source_center = (source_points.min(axis=0) + source_points.max(axis=0)) / 2.0
    target_center = np.asarray(
        [(target_x.min() + target_x.max()) / 2.0, (target_y.min() + target_y.max()) / 2.0]
    )
    scale_and_translation = np.asarray(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale_and_translation[:2, 2] = target_center - scale * source_center
    return scale_and_translation @ orientation


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_atlas_pose_session(model_path: str, modified_ns: int, force_cpu: bool):
    del modified_ns
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = ort.get_available_providers()
    accelerator = next(
        (provider for provider in ("CUDAExecutionProvider", "DmlExecutionProvider") if provider in available),
        None,
    )
    if force_cpu or accelerator is None:
        providers = ["CPUExecutionProvider"]
    elif accelerator == "DmlExecutionProvider":
        providers = [(accelerator, {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = [accelerator, "CPUExecutionProvider"]
    fallback_reason = None
    try:
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    except Exception as exc:
        if force_cpu or providers == ["CPUExecutionProvider"]:
            raise
        fallback_reason = f"{accelerator} initialization failed: {type(exc).__name__}: {exc}"
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    model_input = session.get_inputs()[0]
    model_outputs = {output.name: output for output in session.get_outputs()}
    if model_input.name != POSE_INPUT_NAME or model_input.shape[1:] != [3, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE]:
        raise RuntimeError("Own CNN ONNX input must be images with shape [batch, 3, 299, 299]")
    if POSE_OUTPUT_NAME not in model_outputs or model_outputs[POSE_OUTPUT_NAME].shape[-1] != 3:
        raise RuntimeError("Own CNN ONNX output must be pose_ap_um_lr_deg_dv_deg with shape [batch, 3]")
    if POSE_ORIENTATION_OUTPUT_NAME not in model_outputs:
        raise RuntimeError("Own CNN ONNX output must include orientation_inverted_logit")
    return session, fallback_reason


def run_atlas_pose_onnx(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    model_path: str | Path,
    cancel_event=None,
) -> tuple[np.ndarray, dict]:
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(
            f"Own CNN model is unavailable. Expected the trained ONNX model at: {path}"
        )
    if len(images) != len(masks) or not images:
        raise ValueError("Own CNN inference needs one non-empty mask for every image")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise RuntimeError(f"Own CNN metadata is unavailable. Expected: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_sha256 = _file_sha256(path)
    if metadata.get("sha256") != model_sha256:
        raise RuntimeError("Own CNN model checksum does not match atlas_pose.json")
    if metadata.get("preprocessing_version") != ATLAS_POSE_PREPROCESSING_VERSION:
        raise RuntimeError(
            "Own CNN preprocessing contract does not match this application runtime"
        )
    preprocessing_source_sha256 = metadata.get("preprocessing_source_sha256")
    if not isinstance(preprocessing_source_sha256, str) or len(preprocessing_source_sha256) != 64:
        raise RuntimeError("Own CNN metadata is missing its preprocessing source checksum")
    inputs = []
    for image, mask in zip(images, masks):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError
        inputs.append(preprocess_atlas_pose_image(image, mask))
    batch = np.stack(inputs).astype(np.float32, copy=False)
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError

    started = time.perf_counter()
    session, fallback_reason = _load_atlas_pose_session(str(path.resolve()), path.stat().st_mtime_ns, False)
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError
    provider = session.get_providers()[0]
    def infer(current_session):
        poses = []
        orientation_logits = []
        for start in range(0, len(batch), POSE_INFERENCE_BATCH_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError
            pose, orientation = current_session.run(
                [POSE_OUTPUT_NAME, POSE_ORIENTATION_OUTPUT_NAME],
                {POSE_INPUT_NAME: batch[start : start + POSE_INFERENCE_BATCH_SIZE]},
            )
            poses.append(pose)
            orientation_logits.append(orientation)
        return np.concatenate(poses), np.concatenate(orientation_logits)

    try:
        prediction, orientation_logit = infer(session)
    except Exception as exc:
        if provider == "CPUExecutionProvider":
            raise
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError
        fallback_reason = f"{provider} inference failed: {type(exc).__name__}: {exc}"
        session, _ = _load_atlas_pose_session(str(path.resolve()), path.stat().st_mtime_ns, True)
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError
        provider = session.get_providers()[0]
        prediction, orientation_logit = infer(session)
    prediction = np.asarray(prediction, dtype=np.float64)
    orientation_logit = np.asarray(orientation_logit, dtype=np.float64)
    if prediction.shape != (len(images), 3) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Own CNN returned invalid predictions with shape {prediction.shape}")
    if orientation_logit.shape != (len(images),) or not np.isfinite(orientation_logit).all():
        raise RuntimeError(f"Own CNN returned invalid orientation logits with shape {orientation_logit.shape}")
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError

    return prediction, {
        "backend": "ONNX Runtime",
        "device": provider,
        "onnxruntime_version": ort.__version__,
        "gpu_fallback_reason": fallback_reason,
        "inference_seconds": float(time.perf_counter() - started),
        "model_path": str(path),
        "model_sha256": model_sha256,
        "architecture": metadata.get("architecture"),
        "orientation_inverted": (orientation_logit > 0.0).tolist(),
        "orientation_inverted_logit": orientation_logit.tolist(),
        "preprocessing_version": ATLAS_POSE_PREPROCESSING_VERSION,
        "preprocessing_source_sha256": preprocessing_source_sha256,
        "metadata": metadata,
    }
