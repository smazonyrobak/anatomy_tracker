"""ONNX runtime for residual registration after pose and affine alignment."""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from nonlinear_registration import NonlinearWarp2D


MODEL_SHAPE = (320, 464)
INPUT_NAMES = ("fixed", "moving", "fixed_mask", "moving_mask")
OUTPUT_NAMES = ("atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit")


class DiffeomorphicRegistrationRejected(RuntimeError):
    """A model result that failed a geometric or correspondence gate."""

    def __init__(self, failures: list[str], diagnostics: dict):
        self.failures = tuple(failures)
        self.diagnostics = diagnostics
        super().__init__("Nonlinear registration rejected: " + "; ".join(failures))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gray_unit(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = np.squeeze(np.asarray(image))
    if gray.ndim == 3:
        gray = gray[..., :3].astype(np.float32).mean(axis=-1)
    if gray.ndim != 2 or gray.shape != mask.shape:
        raise ValueError("Registration images and masks must share one H x W canvas")
    gray = gray.astype(np.float32, copy=False)
    if not np.isfinite(gray).all():
        raise ValueError("Registration image contains non-finite values")
    values = gray[mask]
    low, high = np.percentile(values, (0.5, 99.5))
    normalized = np.zeros_like(gray, dtype=np.float32)
    if high > low:
        normalized = np.clip((gray - low) / (high - low), 0.0, 1.0).astype(np.float32)
    normalized[~mask] = 0.0
    return normalized


def _center_geometry(native_shape: tuple[int, int]) -> tuple[int, int, int, int, int, int]:
    height, width = native_shape
    model_height, model_width = MODEL_SHAPE
    copied_height = min(height, model_height)
    copied_width = min(width, model_width)
    source_top = max((height - model_height) // 2, 0)
    source_left = max((width - model_width) // 2, 0)
    model_top = max((model_height - height) // 2, 0)
    model_left = max((model_width - width) // 2, 0)
    return source_top, source_left, model_top, model_left, copied_height, copied_width


def _center_input(array: np.ndarray, geometry: tuple[int, int, int, int, int, int]) -> np.ndarray:
    source_top, source_left, model_top, model_left, height, width = geometry
    result = np.zeros(MODEL_SHAPE, dtype=np.float32)
    result[model_top : model_top + height, model_left : model_left + width] = array[
        source_top : source_top + height, source_left : source_left + width
    ]
    return result


def _native_map(
    model_map: np.ndarray,
    native_shape: tuple[int, int],
    geometry: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    source_top, source_left, model_top, model_left, height, width = geometry
    yy, xx = np.mgrid[: native_shape[0], : native_shape[1]].astype(np.float32)
    result = np.stack((xx, yy), axis=-1)
    mapped = np.moveaxis(model_map[0, :, model_top : model_top + height, model_left : model_left + width], 0, -1).copy()
    mapped[..., 0] += source_left - model_left
    mapped[..., 1] += source_top - model_top
    result[source_top : source_top + height, source_left : source_left + width] = mapped
    return result


def _global_affine_max(velocity: np.ndarray) -> float:
    height, width = velocity.shape[-2:]
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    basis = np.stack((np.ones_like(x), x, y)).reshape(3, -1)
    flat = velocity[0].reshape(2, -1).astype(np.float64)
    coefficients = flat @ basis.T / np.square(basis).sum(axis=1)[None]
    return float(np.abs((coefficients @ basis).reshape(2, height, width)).max())


def _cycle_error(first_map: np.ndarray, second_map: np.ndarray) -> np.ndarray:
    height, width = first_map.shape[:2]
    valid = (
        (first_map[..., 0] >= 0.0)
        & (first_map[..., 0] <= width - 1.0)
        & (first_map[..., 1] >= 0.0)
        & (first_map[..., 1] <= height - 1.0)
    )
    composed = np.stack(
        [
            cv2.remap(
                second_map[..., axis],
                first_map[..., 0],
                first_map[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for axis in range(2)
        ],
        axis=-1,
    )
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    error = np.linalg.norm(composed - np.stack((xx, yy), axis=-1), axis=2)
    error[~valid] = np.nan
    return error


def _validate_session_contract(session) -> None:
    inputs = {value.name: value.shape for value in session.get_inputs()}
    outputs = {value.name: value.shape for value in session.get_outputs()}
    if set(inputs) != set(INPUT_NAMES) or set(outputs) != set(OUTPUT_NAMES):
        raise RuntimeError("Diffeomorphic ONNX names do not match the exported training contract")
    for name in INPUT_NAMES:
        shape = inputs[name]
        if len(shape) != 4 or shape[1:] != [1, *MODEL_SHAPE]:
            raise RuntimeError(f"Diffeomorphic ONNX input {name} must have shape [batch, 1, 320, 464]")
    for name in OUTPUT_NAMES[:3]:
        shape = outputs[name]
        if len(shape) != 4 or shape[1:] != [2, *MODEL_SHAPE]:
            raise RuntimeError(f"Diffeomorphic ONNX output {name} must have shape [batch, 2, 320, 464]")


@lru_cache(maxsize=4)
def _load_session(model_path: str, modified_ns: int, force_cpu: bool):
    del modified_ns
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = ort.get_available_providers()
    accelerator = next(
        (name for name in ("CUDAExecutionProvider", "DmlExecutionProvider") if name in available),
        None,
    )
    if force_cpu or accelerator is None:
        providers = ["CPUExecutionProvider"]
    elif accelerator == "DmlExecutionProvider":
        providers = [(accelerator, {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = [accelerator, "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    _validate_session_contract(session)
    return session


def run_diffeomorphic_registration(
    fixed_atlas: np.ndarray,
    moving_affine_slice: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    model_path: str | Path | None = None,
    *,
    session=None,
    rejection_threshold: float = 0.5,
    minimum_jacobian: float = 0.0,
    maximum_inverse_p95_px: float = 1.0,
    maximum_residual_affine_px: float = 0.05,
    maximum_displacement_p95_px: float = 8.0,
    maximum_displacement_px: float = 12.0,
) -> tuple[NonlinearWarp2D, dict]:
    """Infer and gate one residual warp without mutating pose or affine state."""
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    moving_mask = np.asarray(moving_mask, dtype=bool)
    if fixed_mask.ndim != 2 or fixed_mask.shape != moving_mask.shape or min(fixed_mask.shape) < 2:
        raise ValueError("Registration masks must share one H x W canvas")
    if not fixed_mask.any() or not moving_mask.any():
        raise ValueError("Registration needs non-empty atlas and trusted slice masks")
    fixed = _gray_unit(fixed_atlas, fixed_mask)
    moving = _gray_unit(moving_affine_slice, moving_mask)
    native_shape = fixed_mask.shape
    geometry = _center_geometry(native_shape)
    source_top, source_left, model_top, model_left, copied_height, copied_width = geometry
    feeds = {
        "fixed": _center_input(fixed, geometry)[None, None],
        "moving": _center_input(moving, geometry)[None, None],
        "fixed_mask": _center_input(fixed_mask.astype(np.float32), geometry)[None, None],
        "moving_mask": _center_input(moving_mask.astype(np.float32), geometry)[None, None],
    }

    model_sha256 = None
    if session is None:
        path = Path(model_path) if model_path is not None else None
        if path is None or not path.is_file():
            raise RuntimeError(f"Diffeomorphic ONNX model is unavailable: {path}")
        model_sha256 = _file_sha256(path)
        session = _load_session(str(path.resolve()), path.stat().st_mtime_ns, False)
    else:
        _validate_session_contract(session)

    started = time.perf_counter()
    atlas_to_affine_output, affine_to_atlas_output, velocity, rejection_logit = session.run(
        list(OUTPUT_NAMES), feeds
    )
    inference_seconds = float(time.perf_counter() - started)
    atlas_to_affine_output = np.asarray(atlas_to_affine_output, dtype=np.float32)
    affine_to_atlas_output = np.asarray(affine_to_atlas_output, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    expected_map_shape = (1, 2, *MODEL_SHAPE)
    if (
        atlas_to_affine_output.shape != expected_map_shape
        or affine_to_atlas_output.shape != expected_map_shape
        or velocity.shape != expected_map_shape
        or not np.isfinite(atlas_to_affine_output).all()
        or not np.isfinite(affine_to_atlas_output).all()
        or not np.isfinite(velocity).all()
    ):
        raise RuntimeError("Diffeomorphic ONNX returned invalid map or velocity tensors")
    rejection_values = np.asarray(rejection_logit, dtype=np.float64).reshape(-1)
    if rejection_values.shape != (1,) or not np.isfinite(rejection_values).all():
        raise RuntimeError("Diffeomorphic ONNX returned an invalid rejection logit")

    atlas_to_affine = _native_map(atlas_to_affine_output, native_shape, geometry)
    affine_to_atlas = _native_map(affine_to_atlas_output, native_shape, geometry)
    warp = NonlinearWarp2D(atlas_to_affine, affine_to_atlas)
    reverse_warp = NonlinearWarp2D(affine_to_atlas, atlas_to_affine)
    forward_jacobian = warp.jacobian_determinant()
    inverse_jacobian = reverse_warp.jacobian_determinant()
    forward_cycle = _cycle_error(atlas_to_affine, affine_to_atlas)
    inverse_cycle = _cycle_error(affine_to_atlas, atlas_to_affine)
    cycle_values = np.concatenate((forward_cycle[fixed_mask], inverse_cycle[moving_mask]))
    finite_cycle = cycle_values[np.isfinite(cycle_values)]
    identity_y, identity_x = np.mgrid[: native_shape[0], : native_shape[1]].astype(np.float32)
    identity = np.stack((identity_x, identity_y), axis=-1)
    displacement_values = np.concatenate(
        (
            np.linalg.norm(atlas_to_affine - identity, axis=2)[fixed_mask],
            np.linalg.norm(affine_to_atlas - identity, axis=2)[moving_mask],
        )
    )
    rejection_probability = float(1.0 / (1.0 + np.exp(-np.clip(rejection_values[0], -80.0, 80.0))))
    inverse_p95 = float(np.percentile(finite_cycle, 95)) if len(finite_cycle) else float("inf")
    displacement_p95 = float(np.percentile(displacement_values, 95))
    displacement_max = float(displacement_values.max())
    residual_affine_max = _global_affine_max(velocity)
    trusted_pixels = fixed_mask | moving_mask
    modeled = np.zeros(native_shape, dtype=bool)
    modeled[source_top : source_top + copied_height, source_left : source_left + copied_width] = True
    modeled_trusted_fraction = float(modeled[trusted_pixels].mean())
    diagnostics = {
        "provider": session.get_providers()[0],
        "inference_seconds": inference_seconds,
        "model_sha256": model_sha256,
        "native_shape": native_shape,
        "model_shape": MODEL_SHAPE,
        "source_offset_yx": (source_top, source_left),
        "model_offset_yx": (model_top, model_left),
        "copied_shape": (copied_height, copied_width),
        "modeled_trusted_fraction": modeled_trusted_fraction,
        "rejection_logit": float(rejection_values[0]),
        "rejection_probability": rejection_probability,
        "minimum_forward_jacobian": float(forward_jacobian.min()),
        "minimum_inverse_jacobian": float(inverse_jacobian.min()),
        "fold_count": int((forward_jacobian <= minimum_jacobian).sum() + (inverse_jacobian <= minimum_jacobian).sum()),
        "inverse_finite_fraction": float(np.isfinite(cycle_values).mean()),
        "inverse_p95_px": inverse_p95,
        "residual_affine_max_px": residual_affine_max,
        "displacement_p95_px": displacement_p95,
        "displacement_max_px": displacement_max,
    }
    failures = []
    if modeled_trusted_fraction < 1.0:
        failures.append("trusted tissue lies outside the model field of view")
    if rejection_probability >= rejection_threshold:
        failures.append(f"model rejection probability {rejection_probability:.3f} exceeds {rejection_threshold:.3f}")
    if diagnostics["fold_count"]:
        failures.append(f"nonlinear maps contain {diagnostics['fold_count']} Jacobian failures")
    if diagnostics["inverse_finite_fraction"] < 1.0 or inverse_p95 > maximum_inverse_p95_px:
        failures.append(f"inverse-consistency p95 {inverse_p95:.3f} px exceeds {maximum_inverse_p95_px:.3f} px")
    if residual_affine_max > maximum_residual_affine_px:
        failures.append(f"residual global affine {residual_affine_max:.3f} px exceeds {maximum_residual_affine_px:.3f} px")
    if displacement_p95 > maximum_displacement_p95_px or displacement_max > maximum_displacement_px:
        failures.append(
            f"displacement p95/max {displacement_p95:.3f}/{displacement_max:.3f} px exceeds "
            f"{maximum_displacement_p95_px:.3f}/{maximum_displacement_px:.3f} px"
        )
    if failures:
        raise DiffeomorphicRegistrationRejected(failures, diagnostics)
    return warp, diagnostics
