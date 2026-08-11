"""ONNX runtime for residual registration after pose and affine alignment."""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from nonlinear_registration import (
    COORDINATE_CONVENTION,
    MODEL_CONTRACT_VERSION,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
    MODEL_PIXEL_SPACING_UM,
    MODEL_SHAPE,
    MODEL_SPATIAL_CONTRACT,
    NonlinearWarp2D,
    NonlinearWarpAttestation,
    RUNTIME_GATE_CONTRACT,
    RUNTIME_GATE_VERSION,
    array_sha256,
    nonlinear_runtime_acceptance_issues,
)


INPUT_NAMES = MODEL_INPUT_NAMES
OUTPUT_NAMES = MODEL_OUTPUT_NAMES
# Deliberately unset until native-secondary and independent internal-landmark evidence pass review.
APPROVED_NONLINEAR_RELEASE: dict[str, str] | None = None
APPROVED_RELEASE_KEYS = (
    "model_sha256",
    "manifest_sha256",
    "native_histology_secondary_gate_report_sha256",
    "native_histology_secondary_evaluation_manifest_sha256",
    "internal_landmark_gate_report_sha256",
    "internal_landmark_evaluation_manifest_sha256",
)


class DiffeomorphicRegistrationRejected(RuntimeError):
    """A model result that failed a geometric or correspondence gate."""

    def __init__(self, issues: list[tuple[str, str]], diagnostics: dict):
        self.categories = tuple(dict.fromkeys(category for category, _ in issues))
        self.failures = tuple(message for _, message in issues)
        self.diagnostics = diagnostics
        super().__init__("Nonlinear registration rejected: " + "; ".join(self.failures))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_model_manifest(model_path: Path) -> tuple[str, str, dict]:
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"Validated nonlinear model manifest is unavailable: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    model_sha256 = _file_sha256(model_path)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    expected = {
        "format_version": MODEL_CONTRACT_VERSION,
        "model_sha256": model_sha256,
        "model_shape": list(MODEL_SHAPE),
        "pixel_spacing_um": MODEL_PIXEL_SPACING_UM,
        "spatial_contract": MODEL_SPATIAL_CONTRACT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "runtime_gates": RUNTIME_GATE_CONTRACT,
        "onnx_gate_passed": True,
        "native_histology_secondary_gate_passed": True,
        "native_histology_secondary_benchmark_role": "locked_secondary_native_gate",
        "internal_landmark_gate_passed": True,
        "internal_landmark_benchmark_role": "locked_promotion_gate",
        "promotion_ready": True,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise RuntimeError("Nonlinear model manifest contract failed: " + ", ".join(mismatched))
    evidence_path = model_path.with_suffix(".prelocked.json")
    if (
        manifest.get("prelocked_evidence_file") != evidence_path.name
        or not evidence_path.is_file()
        or manifest.get("prelocked_evidence_sha256") != _file_sha256(evidence_path)
    ):
        raise RuntimeError("Nonlinear model prelocked evidence is missing or fails its commitment")
    prelocked_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if prelocked_evidence.get("model_sha256") != model_sha256:
        raise RuntimeError("Nonlinear model prelocked evidence belongs to a different model")
    for commitment_key, evaluation_key in (
        (
            "locked_native_histology_commitment",
            "native_histology_secondary_evaluation_manifest_sha256",
        ),
        (
            "locked_internal_landmark_commitment",
            "internal_landmark_evaluation_manifest_sha256",
        ),
    ):
        commitment = manifest.get(commitment_key)
        evaluation_sha256 = (
            commitment.get("evaluation_manifest_sha256") if isinstance(commitment, dict) else None
        )
        if (
            not isinstance(commitment, dict)
            or not isinstance(commitment.get("source"), dict)
            or not isinstance(evaluation_sha256, str)
            or len(evaluation_sha256) != 64
            or manifest.get(evaluation_key) != evaluation_sha256
            or prelocked_evidence.get(commitment_key) != commitment
        ):
            raise RuntimeError(f"Nonlinear model has no valid {commitment_key}")
    evidence_hash_keys = (
        "native_histology_secondary_gate_report_sha256",
        "native_histology_secondary_evaluation_manifest_sha256",
        "internal_landmark_gate_report_sha256",
        "internal_landmark_evaluation_manifest_sha256",
    )
    for key in evidence_hash_keys:
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value.lower()
        ):
            raise RuntimeError(f"Nonlinear model manifest has no valid {key}")
    if (
        manifest["native_histology_secondary_gate_report_sha256"].lower()
        == manifest["internal_landmark_gate_report_sha256"].lower()
        or manifest["native_histology_secondary_evaluation_manifest_sha256"].lower()
        == manifest["internal_landmark_evaluation_manifest_sha256"].lower()
    ):
        raise RuntimeError("Native-histology and internal-landmark evidence hashes must be independent")
    if APPROVED_NONLINEAR_RELEASE is None:
        raise RuntimeError("No nonlinear model release is source-approved yet")
    actual_release = {
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
        **{key: manifest[key].lower() for key in evidence_hash_keys},
    }
    release_mismatches = [
        key
        for key in APPROVED_RELEASE_KEYS
        if APPROVED_NONLINEAR_RELEASE.get(key) != actual_release[key]
    ]
    if release_mismatches:
        raise RuntimeError(
            "Nonlinear model does not match the source-approved release: "
            + ", ".join(release_mismatches)
        )
    return model_sha256, manifest_sha256, manifest


def verify_diffeomorphic_model_bundle(model_path: str | Path) -> tuple[str, str, dict]:
    """Verify that a promoted model and manifest satisfy the runtime contract."""
    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(f"Diffeomorphic ONNX model is unavailable: {path}")
    return _verified_model_manifest(path)


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
    low, high = np.quantile(values, (0.005, 0.995))
    normalized = np.zeros_like(gray, dtype=np.float32)
    if high > low:
        normalized = np.clip((gray - low) / (high - low), 0.0, 1.0).astype(np.float32)
    normalized[~mask] = 0.0
    return normalized


def _mind_descriptor(image: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(image, dtype=np.float32), 1, mode="edge")
    neighbours = np.stack((
        padded[1:-1, :-2], padded[1:-1, 2:],
        padded[:-2, 1:-1], padded[2:, 1:-1],
    ))
    distance = (image[None] - neighbours) ** 2
    distance = np.stack([
        cv2.boxFilter(channel, -1, (3, 3), normalize=True, borderType=cv2.BORDER_CONSTANT)
        for channel in distance
    ])
    descriptor = np.exp(-distance / (distance.mean(axis=0, keepdims=True) + 1e-6))
    return descriptor / np.maximum(descriptor.max(axis=0, keepdims=True), 1e-6)


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    return float(2.0 * np.count_nonzero(first & second) / max(
        np.count_nonzero(first) + np.count_nonzero(second), 1
    ))


def _correspondence_diagnostics(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    atlas_to_affine: np.ndarray,
) -> dict[str, float | int]:
    original_overlap = fixed_mask & moving_mask
    overlap_pixels = int(np.count_nonzero(original_overlap))
    overlap_fraction = float(
        overlap_pixels
        / max(np.count_nonzero(fixed_mask), np.count_nonzero(moving_mask), 1)
    )
    map_x, map_y = atlas_to_affine[..., 0], atlas_to_affine[..., 1]
    warped_mask = cv2.remap(
        moving_mask.astype(np.uint8), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0.5
    fixed_descriptor = _mind_descriptor(fixed)
    moving_descriptor = _mind_descriptor(moving)
    warped_descriptor = np.stack([
        cv2.remap(channel, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        for channel in moving_descriptor
    ])
    if overlap_pixels:
        before = float(np.mean(np.abs(
            fixed_descriptor[:, original_overlap] - moving_descriptor[:, original_overlap]
        )))
        after = float(np.mean(np.abs(
            fixed_descriptor[:, original_overlap] - warped_descriptor[:, original_overlap]
        )))
        retained = float(np.count_nonzero(original_overlap & warped_mask) / overlap_pixels)
    else:
        before = after = float("inf")
        retained = 0.0
    surface_before = _dice(fixed_mask, moving_mask)
    surface_after = _dice(fixed_mask, warped_mask)
    return {
        "prewarp_overlap_pixels": overlap_pixels,
        "prewarp_overlap_fraction": overlap_fraction,
        "mind_before": before,
        "mind_after": after,
        "mind_improvement": before - after,
        "surface_dice_before": surface_before,
        "surface_dice_after": surface_after,
        "surface_dice_delta": surface_after - surface_before,
        "retained_coverage": retained,
    }


def verify_diffeomorphic_attestation_inputs(
    warp: NonlinearWarp2D,
    attestation: NonlinearWarpAttestation,
    fixed_atlas: np.ndarray,
    moving_affine_slice: np.ndarray,
) -> None:
    if array_sha256(fixed_atlas) != attestation.atlas_image_sha256:
        raise RuntimeError("Nonlinear evidence does not match its atlas plane")
    if array_sha256(moving_affine_slice) != attestation.moving_affine_sha256:
        raise RuntimeError("Nonlinear evidence does not match its affine slice input")
    observed = {
        **_correspondence_diagnostics(
            _gray_unit(fixed_atlas, attestation.atlas_mask),
            _gray_unit(moving_affine_slice, attestation.affine_mask),
            attestation.atlas_mask,
            attestation.affine_mask,
            warp.atlas_to_affine_xy,
        ),
        **warp.diagnostics(attestation.atlas_mask, attestation.affine_mask),
    }
    attested = attestation.acceptance_diagnostics
    mismatched = [
        key
        for key, value in observed.items()
        if key not in attested
        or not np.isclose(float(attested[key]), float(value), rtol=1e-6, atol=1e-6)
    ]
    if mismatched:
        raise RuntimeError(
            "Nonlinear runtime evidence disagrees with recomputed diagnostics: "
            + ", ".join(mismatched)
        )


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
    pixel_spacing_um: float | None = None,
    source_image_sha256: str | None = None,
    session=None,
) -> tuple[NonlinearWarp2D, dict]:
    """Infer a residual warp at 25 um/pixel without resizing the native canvas."""
    if pixel_spacing_um is None or not np.isclose(float(pixel_spacing_um), MODEL_PIXEL_SPACING_UM):
        raise ValueError("Diffeomorphic registration requires explicit 25 um one-to-one atlas pixels")
    fixed_mask = np.asarray(fixed_mask) > 0.5
    moving_mask = np.asarray(moving_mask) > 0.5
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

    model_sha256 = manifest_sha256 = None
    if session is None:
        if (
            source_image_sha256 is None
            or len(source_image_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_image_sha256.lower())
        ):
            raise ValueError("Production nonlinear registration requires the source image SHA-256")
        path = Path(model_path) if model_path is not None else None
        if path is None or not path.is_file():
            raise RuntimeError(f"Diffeomorphic ONNX model is unavailable: {path}")
        model_sha256, manifest_sha256, _ = _verified_model_manifest(path)
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
    warp_diagnostics = warp.diagnostics(fixed_mask, moving_mask)
    correspondence = _correspondence_diagnostics(
        fixed, moving, fixed_mask, moving_mask, atlas_to_affine
    )
    rejection_probability = float(1.0 / (1.0 + np.exp(-np.clip(rejection_values[0], -80.0, 80.0))))
    trusted_pixels = fixed_mask | moving_mask
    modeled = np.zeros(native_shape, dtype=bool)
    modeled[source_top : source_top + copied_height, source_left : source_left + copied_width] = True
    modeled_trusted_fraction = float(modeled[trusted_pixels].mean())
    diagnostics = {
        "provider": session.get_providers()[0],
        "inference_seconds": inference_seconds,
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
        "source_image_sha256": source_image_sha256,
        "atlas_image_sha256": array_sha256(np.asarray(fixed_atlas)),
        "moving_affine_sha256": array_sha256(np.asarray(moving_affine_slice)),
        "runtime_gate_version": RUNTIME_GATE_VERSION,
        "native_shape": native_shape,
        "model_shape": MODEL_SHAPE,
        "pixel_spacing_um": MODEL_PIXEL_SPACING_UM,
        "spatial_contract": MODEL_SPATIAL_CONTRACT,
        "source_offset_yx": (source_top, source_left),
        "model_offset_yx": (model_top, model_left),
        "copied_shape": (copied_height, copied_width),
        "modeled_trusted_fraction": modeled_trusted_fraction,
        "rejection_logit": float(rejection_values[0]),
        "rejection_probability": rejection_probability,
        **correspondence,
        **warp_diagnostics,
    }
    issues = nonlinear_runtime_acceptance_issues(diagnostics)
    if issues:
        raise DiffeomorphicRegistrationRejected(issues, diagnostics)
    return warp, diagnostics
