from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


POSE_IMAGE_SIZE = 299
POSE_INPUT_NAME = "images"
POSE_OUTPUT_NAME = "pose_ap_um_lr_deg_dv_deg"
POSE_ORIENTATION_OUTPUT_NAME = "orientation_inverted_logit"
POSE_INFERENCE_BATCH_SIZE = 16
QUICKNII_COORDINATE_CONTRACT_VERSION = "quicknii-ras-to-allen-pir-v2"
ATLAS_POSE_PREPROCESSING_VERSION = "smart-mask-scale-invariant-v2"
AUTOMATIC_BRAIN_MASK_VERSION = "border-distance-conditional-hull-v6"
APPROVED_ATLAS_POSE_MODEL_SHA256: str | None = None
APPROVED_ATLAS_POSE_METADATA_SHA256: str | None = None
APPROVED_ATLAS_POSE_EVIDENCE_SHA256: str | None = None
ATLAS_POSE_RELEASE_GATE_THRESHOLDS = {
    "mean_ap_um": 60.0,
    "mean_lr_deg": 0.90,
    "mean_dv_deg": 1.75,
    "absolute_ap_bias_um": 25.0,
    "ap_p95_um": 150.0,
    "worst_ap_band_mae_um": 90.0,
    "worst_product_mae_um": 90.0,
    "ap_bootstrap_upper95_um": 60.0,
    "per_animal_p90_ap_um": 90.0,
    "per_animal_p90_lr_deg": 1.50,
    "per_animal_p90_dv_deg": 2.50,
    "worst_group_p90_ap_um": 90.0,
    "worst_group_p90_lr_deg": 1.50,
    "worst_group_p90_dv_deg": 2.50,
}
ATLAS_POSE_SEALED_METHODS = {
    "deepslice_ai",
    "deepslice_mens_ai",
    "deepslice_mens_ai_ci",
    "atlas_pose",
}
ATLAS_POSE_SEALED_SECTION_COUNT = 1400
ATLAS_POSE_SEALED_EXPERIMENT_COUNT = 10
ATLAS_POSE_SEALED_BENCHMARK_ID = "deepslice_s2p_1400_quicknii_ras_v2"
ATLAS_POSE_SEALED_SPLIT = "sealed_deepslice_s2p"
ATLAS_POSE_RELEASE_CONFIDENCE = 0.95
ATLAS_POSE_SEALED_SOURCE_FILES = (
    "datasets.jsonl",
    "sections.jsonl",
    "provenance.json",
    "downloads.jsonl",
    "registered_image_quality.json",
)


def _brain_mask_from_distance(distance: np.ndarray, threshold: float) -> np.ndarray | None:
    height, width = distance.shape
    mask = (distance > threshold).astype(np.uint8)
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
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    center = np.asarray([width / 2.0, height / 2.0])
    distance_from_center = np.linalg.norm(
        (centroids[1:] - center) / np.asarray([width, height]), axis=1
    )
    selected_labels = 1 + np.flatnonzero(
        (areas >= max(0.001 * height * width, 0.15 * areas.max()))
        & (distance_from_center < 0.55)
    )
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    interior_labels = selected_labels[~np.isin(selected_labels, border_labels)]
    if stats[interior_labels, cv2.CC_STAT_AREA].sum() >= 0.03 * height * width:
        selected_labels = interior_labels
    selected = np.isin(labels, selected_labels).astype(np.uint8)
    radius = max(2, round(min(height, width) / 70))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour_areas = np.asarray([cv2.contourArea(contour) for contour in contours])
    contours = [
        contour
        for contour, area in zip(contours, contour_areas)
        if area >= max(0.001 * height * width, 0.15 * contour_areas.max())
    ]
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, 1, -1)
    return result.astype(bool)


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
    border_threshold = float(np.percentile(border_distance, 75.0))
    otsu_threshold, _ = cv2.threshold(
        np.clip(distance, 0.0, 255.0).astype(np.uint8),
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_floor = 0.75 * float(otsu_threshold)
    result = _brain_mask_from_distance(
        distance,
        max(3.0, border_threshold * 2.0 + 2.0, otsu_floor),
    )
    if result is None or result.mean() < 0.20:
        border_retry = max(3.0, float(np.percentile(border_distance, 60.0)) * 1.25 + 2.0)
        retries = (
            _brain_mask_from_distance(distance, border_retry),
            _brain_mask_from_distance(distance, max(border_retry, 0.15 * float(otsu_threshold))),
            _brain_mask_from_distance(distance, max(border_retry, 0.20 * float(otsu_threshold))),
        )
        retries = [
            retry
            for retry in retries
            if retry is not None
            and 0.03 <= retry.mean() <= 0.75
            and np.concatenate((retry[0], retry[-1], retry[:, 0], retry[:, -1])).mean() < 0.20
        ]
        if retries:
            retry = max(retries, key=np.mean)
            current_area = 0.0 if result is None else float(result.mean())
            if current_area < 0.05 or retry.mean() > 1.5 * current_area:
                result = retry
    if result is None or result.mean() < 0.03:
        raise ValueError("No brain foreground was detected")
    y, x = np.nonzero(result)
    hull = np.zeros_like(result, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack((x, y)).astype(np.int32)), 1)
    if result.sum() < 0.9 * hull.sum():
        result = hull.astype(bool)
    if result.shape != (original_height, original_width):
        result = cv2.resize(
            result.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return result


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


def canonical_brain_sampling_grid(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    center_x = (float(x.min()) + float(x.max())) / 2.0
    center_y = (float(y.min()) + float(y.max())) / 2.0
    side = max(float(x.max() - x.min()), float(y.max() - y.min())) * 1.14
    axis = np.linspace(-0.5, 0.5, POSE_IMAGE_SIZE, dtype=np.float32)
    return np.meshgrid(center_x + axis * side, center_y + axis * side)


def preprocess_atlas_pose_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray, mask = canonicalize_brain_orientation(image, mask)
    sample_x, sample_y = canonical_brain_sampling_grid(mask)
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


ATLAS_POSE_PREPROCESSING_CONTRACT_FUNCTIONS = (
    "as_gray",
    "brain_orientation_affine",
    "canonicalize_brain_orientation",
    "canonical_brain_sampling_grid",
    "preprocess_atlas_pose_image",
)
# Generated from normalized source text for the functions above. Runtime uses this immutable
# value so the contract is identical across CPython versions and inside a PyInstaller bundle.
ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256 = (
    "0be931fd9b3d04d1ac04c5103905539e99b6ecb3bfc6b95e7895ffa9370d8cab"
)


def atlas_pose_preprocessing_contract_sha256() -> str:
    return ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256


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


def _canonical_json_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_hash_tree(value) -> bool:
    return (
        bool(value)
        and (
            _is_sha256(value)
            or isinstance(value, dict)
            and all(isinstance(key, str) and _valid_hash_tree(child) for key, child in value.items())
        )
    )


def atlas_pose_release_quality_gate_valid(quality: dict) -> bool:
    values = quality.get("values", {})
    thresholds = quality.get("thresholds", {})
    passed = quality.get("passed", {})
    names = set(ATLAS_POSE_RELEASE_GATE_THRESHOLDS)
    if thresholds != ATLAS_POSE_RELEASE_GATE_THRESHOLDS or set(values) != names or set(passed) != names:
        return False
    try:
        expected = {
            name: bool(
                np.isfinite(float(values[name]))
                and float(values[name]) <= ATLAS_POSE_RELEASE_GATE_THRESHOLDS[name]
            )
            for name in names
        }
    except (TypeError, ValueError):
        return False
    return (
        all(isinstance(passed[name], bool) for name in names)
        and passed == expected
        and quality.get("all_gates_passed") is all(expected.values())
    )


def verify_atlas_pose_sealed_predictions(path: str | Path) -> dict:
    required = {
        "sealed",
        "split",
        "method",
        "experiment_id",
        "specimen_id",
        "section_image_id",
        "section_number",
        "relative_path",
        "product",
        "ap_band",
        "in_training_ap_domain",
        *(
            f"{prefix}_{axis}"
            for prefix in ("gt", "pred", "error", "absolute_error")
            for axis in ("ap_um", "lr_deg", "dv_deg")
        ),
    }
    seen = set()
    sections = {method: set() for method in ATLAS_POSE_SEALED_METHODS}
    experiments = set()
    section_truth = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("AtlasPose sealed predictions have an incomplete schema")
        for row in reader:
            method = row["method"]
            if (
                method not in ATLAS_POSE_SEALED_METHODS
                or row["sealed"] != "True"
                or row["split"] != ATLAS_POSE_SEALED_SPLIT
                or row["in_training_ap_domain"] not in {"True", "False"}
            ):
                raise RuntimeError("AtlasPose sealed predictions contain an invalid cohort row")
            try:
                section_id = int(row["section_image_id"])
                experiment_id = int(row["experiment_id"])
                int(row["specimen_id"])
                int(row["section_number"])
                for axis in ("ap_um", "lr_deg", "dv_deg"):
                    ground_truth = float(row[f"gt_{axis}"])
                    prediction = float(row[f"pred_{axis}"])
                    error = float(row[f"error_{axis}"])
                    absolute_error = float(row[f"absolute_error_{axis}"])
                    if (
                        not np.isfinite((ground_truth, prediction, error, absolute_error)).all()
                        or not np.isclose(error, prediction - ground_truth, rtol=1e-12, atol=1e-9)
                        or not np.isclose(absolute_error, abs(error), rtol=1e-12, atol=1e-9)
                    ):
                        raise ValueError
            except (TypeError, ValueError) as error:
                raise RuntimeError("AtlasPose sealed predictions contain invalid pose values") from error
            key = (method, section_id)
            if key in seen:
                raise RuntimeError("AtlasPose sealed predictions contain duplicate method/section rows")
            truth = (
                experiment_id,
                int(row["specimen_id"]),
                int(row["section_number"]),
                row["relative_path"],
                row["product"],
                row["ap_band"],
                row["in_training_ap_domain"],
                *(float(row[f"gt_{axis}"]) for axis in ("ap_um", "lr_deg", "dv_deg")),
            )
            if section_id in section_truth and section_truth[section_id] != truth:
                raise RuntimeError("AtlasPose methods do not share one paired ground-truth cohort")
            section_truth[section_id] = truth
            seen.add(key)
            sections[method].add(section_id)
            experiments.add(experiment_id)
    cohort = next(iter(sections.values()))
    if (
        len(cohort) != ATLAS_POSE_SEALED_SECTION_COUNT
        or any(ids != cohort for ids in sections.values())
        or len(seen) != ATLAS_POSE_SEALED_SECTION_COUNT * len(ATLAS_POSE_SEALED_METHODS)
        or len(experiments) != ATLAS_POSE_SEALED_EXPERIMENT_COUNT
    ):
        raise RuntimeError("AtlasPose sealed predictions do not cover the complete paired cohort")
    return {
        "section_count": len(cohort),
        "experiment_count": len(experiments),
        "methods": sorted(sections),
    }


def atlas_pose_evaluator_environment_valid(environment: dict) -> bool:
    if not isinstance(environment, dict):
        return False
    payload = dict(environment)
    commitment = payload.pop("commitment_sha256", None)
    dependencies = payload.get("dependencies", {})
    return (
        payload.get("contract_version") == 1
        and _valid_hash_tree(payload.get("source_sha256"))
        and _valid_hash_tree(payload.get("deepslice_model_sha256"))
        and isinstance(dependencies, dict)
        and bool(dependencies)
        and all(isinstance(name, str) and isinstance(value, str) and value for name, value in dependencies.items())
        and commitment == _canonical_json_sha256(payload)
    )


def atlas_pose_evidence_timestamps_valid(claim: dict, receipt: dict) -> bool:
    try:
        claimed = datetime.fromisoformat(claim["claimed_at_utc"])
        completed = datetime.fromisoformat(receipt["completed_at_utc"])
    except (KeyError, TypeError, ValueError):
        return False
    return claimed.tzinfo is not None and completed.tzinfo is not None and claimed <= completed


def _metadata_training_data_sha256(metadata: dict) -> dict:
    return {
        "synthetic_manifests": metadata.get("manifest_sha256"),
        "registered_data": metadata.get("registered_data", {}).get("sha256"),
        "atlas_data": metadata.get("atlas_data_sha256"),
    }


def _metadata_sealed_source_sha256(metadata: dict) -> dict:
    registered = metadata.get("registered_data", {}).get("sha256", {})
    return {name: registered.get(name) for name in ATLAS_POSE_SEALED_SOURCE_FILES}


def verify_atlas_pose_candidate_bundle(model_path: str | Path) -> tuple[str, str, dict]:
    path = Path(model_path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"AtlasPose candidate model or metadata is unavailable: {path}")
    model_sha256 = _file_sha256(path)
    metadata_sha256 = _file_sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != model_sha256:
        raise RuntimeError("AtlasPose candidate model checksum does not match its metadata")
    if metadata.get("preprocessing_version") != ATLAS_POSE_PREPROCESSING_VERSION:
        raise RuntimeError("AtlasPose candidate preprocessing version differs from the runtime")
    if metadata.get("automatic_brain_mask_version") != AUTOMATIC_BRAIN_MASK_VERSION:
        raise RuntimeError("AtlasPose candidate brain-mask version differs from the runtime")
    if metadata.get("quicknii_coordinate_contract") != QUICKNII_COORDINATE_CONTRACT_VERSION:
        raise RuntimeError("AtlasPose candidate coordinate contract differs from the runtime")
    if metadata.get("preprocessing_contract_sha256") != atlas_pose_preprocessing_contract_sha256():
        raise RuntimeError("AtlasPose candidate preprocessing checksum differs from the runtime")
    return model_sha256, metadata_sha256, metadata


@lru_cache(maxsize=8)
def _verify_atlas_pose_candidate_bundle_cached(
    model_path: str,
    model_size: int,
    model_modified_ns: int,
    metadata_size: int,
    metadata_modified_ns: int,
) -> tuple[str, str, dict]:
    del model_size, model_modified_ns, metadata_size, metadata_modified_ns
    return verify_atlas_pose_candidate_bundle(model_path)


def _verified_atlas_pose_candidate_bundle(path: Path) -> tuple[str, str, dict]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return verify_atlas_pose_candidate_bundle(path)
    model_stat = path.stat()
    metadata_stat = metadata_path.stat()
    return _verify_atlas_pose_candidate_bundle_cached(
        str(path.resolve()),
        model_stat.st_size,
        model_stat.st_mtime_ns,
        metadata_stat.st_size,
        metadata_stat.st_mtime_ns,
    )


def verify_atlas_pose_release_bundle(
    model_path: str | Path,
    evidence_path: str | Path | None = None,
) -> tuple[str, str, str, dict, dict]:
    path = Path(model_path)
    metadata_path = path.with_suffix(".json")
    evidence_path = Path(evidence_path) if evidence_path is not None else path.with_name("RELEASE_REPORT.json")
    sealed_metrics_path = evidence_path.with_name("SEALED_metrics.json")
    sealed_predictions_path = evidence_path.with_name("SEALED_predictions.csv")
    presealed_path = evidence_path.with_name("PRESEALED_COMMITMENT.json")
    sealed_claim_path = evidence_path.with_name("SEALED_CLAIM.json")
    receipt_path = evidence_path.with_name("SEALED_CONSUMPTION_RECEIPT.json")
    missing = [
        candidate
        for candidate in (
            path,
            metadata_path,
            evidence_path,
            sealed_metrics_path,
            sealed_predictions_path,
            presealed_path,
            sealed_claim_path,
            receipt_path,
        )
        if not candidate.is_file()
    ]
    if missing:
        raise RuntimeError(f"AtlasPose approved release bundle is incomplete: {missing}")
    model_sha256, metadata_sha256, metadata = verify_atlas_pose_candidate_bundle(path)
    evidence_sha256 = _file_sha256(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    integrity = evidence.pop("release_integrity_sha256", None)
    if integrity != _canonical_json_sha256(evidence):
        raise RuntimeError("AtlasPose sealed release evidence integrity check failed")
    release_version = evidence.get("release_report_version")
    if release_version not in (3, 4):
        raise RuntimeError("AtlasPose sealed release evidence version is unsupported")
    recovery_paths = {
        "commitment": evidence_path.with_name("SEALED_RECOVERY_COMMITMENT.json"),
        "failed_claim": evidence_path.with_name("FAILED_ATTEMPT_CLAIM.json"),
        "failed_receipt": evidence_path.with_name("FAILED_ATTEMPT_RECEIPT.json"),
    }
    if release_version == 4 and any(not path.is_file() for path in recovery_paths.values()):
        raise RuntimeError("AtlasPose audited recovery evidence is incomplete")
    training_data_sha256 = _metadata_training_data_sha256(metadata)
    expected = {
        "release_report_version": release_version,
        "sealed": True,
        "benchmark_role": "final_release_gate",
        "release_approved": True,
        "promotion_ready": True,
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "preprocessing_contract_sha256": atlas_pose_preprocessing_contract_sha256(),
        "training_source_sha256": metadata.get("source_sha256"),
        "training_data_sha256": training_data_sha256,
        "sealed_metrics_sha256": _file_sha256(sealed_metrics_path),
        "sealed_predictions_sha256": _file_sha256(sealed_predictions_path),
        "presealed_commitment_sha256": _file_sha256(presealed_path),
        "sealed_claim_sha256": _file_sha256(sealed_claim_path),
        "consumption_receipt_sha256": _file_sha256(receipt_path),
    }
    mismatched = [key for key, value in expected.items() if evidence.get(key) != value]
    component_passed = evidence.get("deepslice_component_passed", {})
    hashes = (
        training_data_sha256,
        metadata.get("source_sha256"),
        evidence.get("sealed_data_sha256"),
        evidence.get("evaluator_sha256"),
        evidence.get("evaluator_environment_sha256"),
    )
    if mismatched or not all(_valid_hash_tree(value) for value in hashes):
        raise RuntimeError(
            "AtlasPose sealed release evidence contract failed"
            + (f": {', '.join(mismatched)}" if mismatched else "")
        )
    if set(component_passed) != {"ap_um", "lr_deg", "dv_deg"} or not all(component_passed.values()):
        raise RuntimeError("AtlasPose did not pass every sealed DeepSlice component gate")
    simultaneous = evidence.get("deepslice_simultaneous_superiority", {})
    if (
        simultaneous.get("candidate") != "atlas_pose"
        or simultaneous.get("reference") != "deepslice_mens_ai_ci"
        or simultaneous.get("simultaneous_superiority_passed") is not True
    ):
        raise RuntimeError("AtlasPose did not pass simultaneous sealed DeepSlice superiority")
    quality = evidence.get("quality_gate", {})
    if not atlas_pose_release_quality_gate_valid(quality) or not quality["all_gates_passed"]:
        raise RuntimeError("AtlasPose did not pass every sealed absolute-quality gate")
    sealed_metrics = json.loads(sealed_metrics_path.read_text(encoding="utf-8"))
    prediction_cohort = verify_atlas_pose_sealed_predictions(sealed_predictions_path)
    comparison_rows = [
        row
        for row in sealed_metrics.get("animal_level_paired_bootstrap", [])
        if row.get("candidate") == "atlas_pose"
        and row.get("reference") == "deepslice_mens_ai_ci"
        and row.get("metric") in {
            "absolute_error_ap_um",
            "absolute_error_lr_deg",
            "absolute_error_dv_deg",
        }
    ]
    comparisons = {
        row["metric"].removeprefix("absolute_error_"): row for row in comparison_rows
    }
    expected_component_passed = {
        axis: bool(
            row.get("delta_candidate_minus_reference", np.inf) < 0.0
            and row.get("probability_candidate_lower_error", 0.0)
            >= ATLAS_POSE_RELEASE_CONFIDENCE
        )
        for axis, row in comparisons.items()
    }
    sealed_data = {
        key: sealed_metrics.get("source", {}).get(key)
        for key in ATLAS_POSE_SEALED_SOURCE_FILES
    }
    sealed_data["sealed_image_tree_sha256"] = sealed_metrics.get("source", {}).get(
        "sealed_image_tree_sha256"
    )
    presealed = json.loads(presealed_path.read_text(encoding="utf-8"))
    claim = json.loads(sealed_claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recovery = None
    failed_claim = None
    failed_receipt = None
    if release_version == 4:
        recovery = json.loads(recovery_paths["commitment"].read_text(encoding="utf-8"))
        failed_claim = json.loads(recovery_paths["failed_claim"].read_text(encoding="utf-8"))
        failed_receipt = json.loads(recovery_paths["failed_receipt"].read_text(encoding="utf-8"))
        recovery_valid = (
            evidence.get("sealed_recovery_commitment_sha256")
            == _file_sha256(recovery_paths["commitment"])
            == receipt.get("sealed_recovery_commitment_sha256")
            and evidence.get("failed_attempt_receipt_sha256")
            == _file_sha256(recovery_paths["failed_receipt"])
            == receipt.get("failed_attempt_receipt_sha256")
            and _file_sha256(recovery_paths["failed_claim"]) == _file_sha256(sealed_claim_path)
            and recovery.get("contract_version") == 1
            and recovery.get("benchmark_id") == ATLAS_POSE_SEALED_BENCHMARK_ID
            and recovery.get("recovery_mode") == "diagnostic-empty-annotation-mask-v1"
            and recovery.get("model_sha256") == model_sha256
            and recovery.get("metadata_sha256") == metadata_sha256
            and recovery.get("presealed_commitment_sha256") == _file_sha256(presealed_path)
            and recovery.get("original_claim_sha256") == _file_sha256(sealed_claim_path)
            and recovery.get("failed_attempt_receipt_sha256")
            == _file_sha256(recovery_paths["failed_receipt"])
            and recovery.get("original_evaluator_environment_sha256")
            == presealed.get("evaluator_environment", {}).get("commitment_sha256")
            and recovery.get("recovery_evaluator_environment", {}).get("commitment_sha256")
            == evidence.get("evaluator_environment_sha256")
            and atlas_pose_evaluator_environment_valid(
                recovery.get("recovery_evaluator_environment")
            )
            and recovery.get("sealed_result_artifacts_existed_before_recovery") is False
            and failed_claim == claim
            and failed_receipt.get("status") == "failed"
            and failed_receipt.get("failure")
            == "ValueError: The plane-distance metric needs a non-empty 2-D brain mask"
            and failed_receipt.get("claim_sha256") == _file_sha256(sealed_claim_path)
            and failed_receipt.get("model_sha256") == model_sha256
            and failed_receipt.get("presealed_commitment_sha256")
            == _file_sha256(presealed_path)
        )
    else:
        recovery_valid = (
            presealed.get("evaluator_environment", {}).get("commitment_sha256")
            == evidence.get("evaluator_environment_sha256")
        )
    if (
        sealed_metrics.get("benchmark_id") != ATLAS_POSE_SEALED_BENCHMARK_ID
        or sealed_metrics.get("benchmark_role") != "final_test_only"
        or sealed_metrics.get("section_count") != prediction_cohort["section_count"]
        or sealed_metrics.get("experiment_count") != prediction_cohort["experiment_count"]
        or len(comparison_rows) != 3
        or set(comparisons) != {"ap_um", "lr_deg", "dv_deg"}
        or evidence.get("deepslice_comparisons") != comparisons
        or evidence.get("deepslice_component_passed") != expected_component_passed
        or evidence.get("sealed_data_sha256") != sealed_data
        or evidence.get("evaluator_sha256") != sealed_metrics.get("evaluator_sha256")
        or evidence.get("evaluator_environment_sha256")
        != sealed_metrics.get("evaluator_environment_sha256")
        or evidence.get("deepslice_simultaneous_superiority")
        != sealed_metrics.get("animal_level_joint_superiority")
        or presealed.get("contract_version") != 1
        or presealed.get("benchmark_id") != ATLAS_POSE_SEALED_BENCHMARK_ID
        or presealed.get("model_sha256") != model_sha256
        or presealed.get("metadata_sha256") != metadata_sha256
        or presealed.get("training_source_sha256") != metadata.get("source_sha256")
        or presealed.get("training_data_sha256") != training_data_sha256
        or presealed.get("sealed_source_sha256")
        != _metadata_sealed_source_sha256(metadata)
        or not recovery_valid
        or not atlas_pose_evaluator_environment_valid(presealed.get("evaluator_environment"))
        or claim.get("contract_version") != 1
        or claim.get("benchmark_id") != ATLAS_POSE_SEALED_BENCHMARK_ID
        or claim.get("sealed_access_permitted_after_claim_only") is not True
        or claim.get("model_sha256") != model_sha256
        or claim.get("metadata_sha256") != metadata_sha256
        or claim.get("presealed_commitment_sha256") != _file_sha256(presealed_path)
        or receipt.get("contract_version") != 1
        or receipt.get("benchmark_id") != ATLAS_POSE_SEALED_BENCHMARK_ID
        or receipt.get("status") != "completed"
        or receipt.get("model_sha256") != model_sha256
        or receipt.get("claim_sha256") != _file_sha256(sealed_claim_path)
        or receipt.get("presealed_commitment_sha256") != _file_sha256(presealed_path)
        or receipt.get("sealed_predictions_sha256") != _file_sha256(sealed_predictions_path)
        or receipt.get("sealed_metrics_sha256") != _file_sha256(sealed_metrics_path)
        or not atlas_pose_evidence_timestamps_valid(claim, receipt)
    ):
        raise RuntimeError("AtlasPose release evidence does not bind its sealed evaluation data")
    if metadata.get("sha256") != model_sha256:
        raise RuntimeError("AtlasPose metadata does not bind the approved model")
    return model_sha256, metadata_sha256, evidence_sha256, metadata, evidence


def verify_atlas_pose_model_bundle(model_path: str | Path) -> tuple[str, str, str, dict, dict]:
    pins = (
        APPROVED_ATLAS_POSE_MODEL_SHA256,
        APPROVED_ATLAS_POSE_METADATA_SHA256,
        APPROVED_ATLAS_POSE_EVIDENCE_SHA256,
    )
    if not all(_is_sha256(value) for value in pins):
        raise RuntimeError(
            "AtlasPose is not release-approved: model, metadata, and sealed-evidence hashes "
            "must be pinned in application source"
        )
    verified = verify_atlas_pose_release_bundle(model_path)
    if verified[:3] != pins:
        raise RuntimeError("AtlasPose release bundle does not match the source-pinned hashes")
    return verified


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


def _run_atlas_pose_onnx(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    model_path: str | Path,
    cancel_event=None,
    *,
    require_release_approval: bool,
) -> tuple[np.ndarray, dict]:
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(
            f"Own CNN model is unavailable. Expected the trained ONNX model at: {path}"
        )
    if len(images) != len(masks) or not images:
        raise ValueError("Own CNN inference needs one non-empty mask for every image")
    if require_release_approval:
        model_sha256, metadata_sha256, evidence_sha256, metadata, _ = (
            verify_atlas_pose_model_bundle(path)
        )
    else:
        model_sha256, metadata_sha256, metadata = _verified_atlas_pose_candidate_bundle(path)
        evidence_sha256 = None
    preprocessing_contract_sha256 = metadata["preprocessing_contract_sha256"]
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
        "metadata_sha256": metadata_sha256,
        "release_evidence_sha256": evidence_sha256,
        "architecture": metadata.get("architecture"),
        "orientation_inverted": (orientation_logit > 0.0).tolist(),
        "orientation_inverted_logit": orientation_logit.tolist(),
        "preprocessing_version": ATLAS_POSE_PREPROCESSING_VERSION,
        "automatic_brain_mask_version": AUTOMATIC_BRAIN_MASK_VERSION,
        "preprocessing_contract_sha256": preprocessing_contract_sha256,
        "metadata": metadata,
    }


def run_atlas_pose_onnx(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    model_path: str | Path,
    cancel_event=None,
) -> tuple[np.ndarray, dict]:
    """Run only a source-pinned AtlasPose release bundle."""
    return _run_atlas_pose_onnx(
        images,
        masks,
        model_path,
        cancel_event,
        require_release_approval=True,
    )


def run_atlas_pose_candidate_onnx(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    model_path: str | Path,
    cancel_event=None,
) -> tuple[np.ndarray, dict]:
    """Run a checksum-bound candidate for the isolated sealed evaluator only."""
    return _run_atlas_pose_onnx(
        images,
        masks,
        model_path,
        cancel_event,
        require_release_approval=False,
    )
