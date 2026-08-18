"""Cold-start data contract for joint CCF plane inference and registration.

This module contains data construction only.  It deliberately depends on the
CCF renderer, synthetic generator, and registered-section reader, but on no
learned model or checkpoint from this or another project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from source.dense_registration_preprocessing import (
    MODEL_SHAPE,
    NATIVE_SHAPE,
    PAD_X,
)
from training.registered_section_dataset import (
    RegisteredSectionDataset,
    registered_static_cache_contract,
    registered_static_cache_key,
)
from training.synthetic_registration import (
    AP_MAX_UM,
    AP_MIN_UM,
    BREGMA_AP_INDEX,
    VOXEL_UM,
    split_ap_indices,
)


INDEPENDENT_DATA_VERSION = 1
SUPERVISED_PRODUCT_IDS = (5,)
TILT_MIN_DEG = -35.0
TILT_MAX_DEG = 35.0
SOURCE_CANVAS_CONTRACT = (
    "one-channel-float32-320x464;binary-mask;slice-orientation-preserved;"
    "brain-bbox-isotropic-fit-into-320x456;pad4;available-outline-modes-zero-"
    "intensity-strictly-outside-input-mask;absent-mode-letterboxes-full-raw-frame"
)
MAP_CONTRACT = (
    "absolute-xy-pixels;truth_fixed_to_source_map samples source at each fixed "
    "pixel;truth_source_to_fixed_map samples fixed at each source pixel"
)
SOURCE_VIEW_CONTRACT = (
    "handedness-preserving center similarity on the unified source canvas;"
    "rotation stratified over [-180,180] degrees;scale stratified over [0.5,1.5];"
    "image/mask/labels and both absolute pixel maps transformed together"
)
HIGH_TILT_CONTRACT = (
    "positive-plane train-or-validation stream;LR-only,DV-only,and-both modes;"
    "active magnitudes stratified over [15,35] degrees;per-axis signs balanced"
)
OUTLINE_MODE_NAMES = ("accurate", "imperfect", "absent")
OUTLINE_MODE_PROBABILITIES = np.asarray((0.35, 0.35, 0.30), np.float64)
OUTLINE_INSIDE_FEATHER_PX = 3.0
OUTLINE_CURRICULUM_CONTRACT = (
    "three hash-bound largest-remainder-stratified modes with probabilities "
    "accurate=0.35,imperfect=0.35,absent=0.30;accurate-reference-outline-and-zero-outside;"
    "imperfect-independent-morphology-smooth-boundary-jitter-one-small-gap-one-"
    "small-island-and-zero-outside;absent-zero-mask-channel-and-raw-background-"
    "retained;available-mask intensity uses sin(pi/2*clip(inside_distance/3px)) "
    "inside-only feather and remains exactly zero outside;exact-tissue-damage-"
    "visible-validity-never-derived-from-input-outline"
)
SYNTHETIC_AP_LEVELS_UM = np.asarray(
    (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0), np.float32
)
SYNTHETIC_TILT_LEVELS_DEG = np.asarray(
    (0.25, 0.5, 1.0, 2.0, 5.0, 10.0), np.float32
)
PRODUCT5_CANDIDATE_SCHEDULE = (
    ("nearest", 25.0, 0.25),
    ("resolvable-boundary", 100.0, 1.0),
    ("wide-250-2", 250.0, 2.0),
    ("wide-500-5", 500.0, 5.0),
    ("wide-1000-10", 1000.0, 10.0),
)


def _canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _rng(seed: int, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"independent-joint-data-v1:{int(seed)}:{stream}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _stratified_uniform(
    rng: np.random.Generator, count: int, low: float, high: float
) -> np.ndarray:
    values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(values)
    return (low + (high - low) * values).astype(np.float32)


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32).mean(axis=-1)
    return image.astype(np.float32, copy=False)


def _outline_plan(count: int, seed: int, stream: str) -> dict:
    rng = _rng(seed, f"{stream}-outline-plan")
    expected = OUTLINE_MODE_PROBABILITIES * int(count)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(count) - int(counts.sum())
    tie_break = rng.random(3)
    order = sorted(
        range(3),
        key=lambda index: (expected[index] - counts[index], tie_break[index]),
        reverse=True,
    )
    counts[order[:remainder]] += 1
    mode = np.concatenate(
        [np.full(counts[index], index, np.int8) for index in range(3)]
    )
    rng.shuffle(mode)
    imperfect = mode == 1
    morphology = np.zeros(count, np.int8)
    morphology[imperfect] = rng.choice(
        np.asarray((-3, -2, -1, 1, 2, 3), np.int8), int(imperfect.sum())
    )
    jitter = np.zeros(count, np.float32)
    jitter[imperfect] = rng.uniform(0.5, 2.5, int(imperfect.sum())).astype(np.float32)
    jitter_seed = rng.integers(
        0, np.iinfo(np.uint64).max, count, dtype=np.uint64, endpoint=True
    )
    sample_receipts = [
        _payload_sha256(
            {
                "mode": int(mode[item]),
                "morphology_px": int(morphology[item]),
                "jitter_amplitude_px": float(jitter[item]),
                "jitter_seed": int(jitter_seed[item]),
                "gap_count": int(imperfect[item]),
                "island_count": int(imperfect[item]),
                "contract": OUTLINE_CURRICULUM_CONTRACT,
            }
        )
        for item in range(count)
    ]
    plan = {
        "mode_probabilities": OUTLINE_MODE_PROBABILITIES.astype(np.float32),
        "mode_counts": counts,
        "mode": mode,
        "morphology_px": morphology,
        "jitter_amplitude_px": jitter,
        "jitter_seed": jitter_seed,
        "sample_receipt_sha256": sample_receipts,
    }
    plan["plan_sha256"] = _payload_sha256(plan)
    return plan


def _imperfect_outline(
    mask: np.ndarray, morphology_px: int, jitter_amplitude_px: float, seed: int
) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    height, width = mask.shape
    rng = np.random.default_rng(int(seed))
    low_height = max(3, round(height / 48))
    low_width = max(3, round(width / 48))
    dx = cv2.resize(
        rng.normal(size=(low_height, low_width)).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    dy = cv2.resize(
        rng.normal(size=(low_height, low_width)).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    dx *= float(jitter_amplitude_px) / max(float(dx.std()), 1e-6)
    dy *= float(jitter_amplitude_px) / max(float(dy.std()), 1e-6)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    result = cv2.remap(
        mask,
        x + dx,
        y + dy,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    kernel = np.ones((3, 3), np.uint8)
    if morphology_px > 0:
        result = cv2.dilate(result, kernel, iterations=int(morphology_px))
    elif morphology_px < 0:
        result = cv2.erode(result, kernel, iterations=-int(morphology_px))

    boundary = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel).astype(bool)
    by, bx = np.nonzero(boundary)
    if len(bx):
        choice = int(rng.integers(len(bx)))
        radius_x = int(rng.integers(2, 7))
        radius_y = int(rng.integers(2, 7))
        cv2.ellipse(
            result,
            (int(bx[choice]), int(by[choice])),
            (radius_x, radius_y),
            float(rng.uniform(0.0, 180.0)),
            0,
            360,
            0,
            -1,
        )
    exterior_ring = cv2.dilate(result, np.ones((15, 15), np.uint8)) & ~result.astype(bool)
    iy, ix = np.nonzero(exterior_ring)
    if len(ix):
        choice = int(rng.integers(len(ix)))
        cv2.ellipse(
            result,
            (int(ix[choice]), int(iy[choice])),
            (int(rng.integers(2, 6)), int(rng.integers(2, 6))),
            float(rng.uniform(0.0, 180.0)),
            0,
            360,
            1,
            -1,
        )
    return result.astype(bool)


def _inside_outline_alpha(mask: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(
        np.ascontiguousarray(mask, dtype=np.uint8), cv2.DIST_L2, 3
    )
    return np.sin(
        0.5
        * np.pi
        * np.clip(distance / OUTLINE_INSIDE_FEATHER_PX, 0.0, 1.0)
    ).astype(np.float32)


def canonical_source_canvas(
    image: np.ndarray, mask: np.ndarray, *, zero_outside: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one raw section to the shared canvas without changing its orientation."""
    gray = np.clip(np.rint(_gray(image)), 0, 255).astype(np.uint8)
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    if len(x) < 64:
        raise ValueError("the source brain mask contains fewer than 64 pixels")
    native_height, native_width = NATIVE_SHAPE
    scale = min(
        (native_width - 1) / max(float(x.max() - x.min()), 1.0),
        (native_height - 1) / max(float(y.max() - y.min()), 1.0),
    )
    source_center = np.asarray(
        ((x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0), np.float32
    )
    target_center = np.asarray(
        ((native_width - 1) / 2.0, (native_height - 1) / 2.0), np.float32
    )
    affine = np.asarray([[scale, 0.0, 0.0], [0.0, scale, 0.0]], np.float32)
    affine[:, 2] = target_center - scale * source_center
    native_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        affine,
        (native_width, native_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    native_image = cv2.warpAffine(
        gray,
        affine,
        (native_width, native_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float32) / 255.0
    if zero_outside:
        native_image *= _inside_outline_alpha(native_mask)
    canvas = np.zeros(MODEL_SHAPE, np.float32)
    canvas_mask = np.zeros(MODEL_SHAPE, bool)
    canvas[:, PAD_X : PAD_X + native_width] = native_image
    canvas_mask[:, PAD_X : PAD_X + native_width] = native_mask
    return torch.from_numpy(canvas[None]), torch.from_numpy(canvas_mask[None])


def raw_source_canvas(image: np.ndarray) -> torch.Tensor:
    """Letterbox a full raw frame when no tissue outline is available."""
    gray = np.clip(np.rint(_gray(image)), 0, 255).astype(np.uint8)
    native_height, native_width = NATIVE_SHAPE
    height, width = gray.shape
    scale = min(native_width / float(width), native_height / float(height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    native = np.zeros(NATIVE_SHAPE, np.float32)
    left = (native_width - resized_width) // 2
    top = (native_height - resized_height) // 2
    native[top : top + resized_height, left : left + resized_width] = resized
    canvas = np.zeros(MODEL_SHAPE, np.float32)
    canvas[:, PAD_X : PAD_X + native_width] = native
    return torch.from_numpy(canvas[None])


def _registered_outline_canvas(
    image: np.ndarray,
    reference_mask: np.ndarray,
    mode: int,
    morphology_px: int,
    jitter_amplitude_px: float,
    jitter_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if mode == 2:
        image_canvas = raw_source_canvas(image)
        return image_canvas, torch.zeros_like(image_canvas, dtype=torch.bool), float("nan")
    raw_canvas, accurate_mask = canonical_source_canvas(
        image, reference_mask, zero_outside=False
    )
    selected_mask = accurate_mask.numpy()[0]
    if mode == 1:
        selected_mask = _imperfect_outline(
            selected_mask,
            morphology_px,
            jitter_amplitude_px,
            jitter_seed,
        )
    selected_mask = torch.from_numpy(selected_mask[None])
    intersection = (selected_mask & accurate_mask).sum().item()
    union = (selected_mask | accurate_mask).sum().item()
    quality = float(intersection / max(union, 1))
    alpha = torch.from_numpy(_inside_outline_alpha(selected_mask.numpy()[0])[None])
    return raw_canvas * alpha, selected_mask, quality


def product5_schedule_level(step: int) -> tuple[str, float, float]:
    return PRODUCT5_CANDIDATE_SCHEDULE[int(step) % len(PRODUCT5_CANDIDATE_SCHEDULE)]


def product5_candidate_offsets(
    count: int, schedule_step: int
) -> tuple[np.ndarray, dict[str, str | float | int]]:
    """Return exactly +/- AP, +/- L-R, and +/- D-V offsets for every sample."""
    name, ap_um, tilt_deg = product5_schedule_level(schedule_step)
    offsets = np.asarray(
        (
            (-ap_um, 0.0, 0.0),
            (+ap_um, 0.0, 0.0),
            (0.0, -tilt_deg, 0.0),
            (0.0, +tilt_deg, 0.0),
            (0.0, 0.0, -tilt_deg),
            (0.0, 0.0, +tilt_deg),
        ),
        np.float32,
    )
    return np.broadcast_to(offsets, (int(count), 6, 3)).copy(), {
        "index": int(schedule_step) % len(PRODUCT5_CANDIDATE_SCHEDULE),
        "name": name,
        "ap_um": float(ap_um),
        "tilt_deg": float(tilt_deg),
    }


def _poses_in_training_domain(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, np.float32)
    return (
        (AP_MIN_UM <= poses[..., 0])
        & (poses[..., 0] <= AP_MAX_UM)
        & (TILT_MIN_DEG <= poses[..., 1])
        & (poses[..., 1] <= TILT_MAX_DEG)
        & (TILT_MIN_DEG <= poses[..., 2])
        & (poses[..., 2] <= TILT_MAX_DEG)
    )


def _synthetic_wrong_offsets(
    true_pose: np.ndarray,
    split: str,
    count: int,
    seed: int,
    *,
    _final_capability=None,
) -> np.ndarray:
    if count < 1:
        raise ValueError("negative candidate count must be positive")
    rng = _rng(seed, "synthetic-wrong-planes")
    split_pool = set(
        int(value)
        for value in split_ap_indices(split, _final_capability=_final_capability)
    )
    wrong = np.empty((len(true_pose), count, 3), np.float32)
    signed_tilt = [
        sign * float(level)
        for level in SYNTHETIC_TILT_LEVELS_DEG
        for sign in (-1.0, 1.0)
    ]
    for item, pose in enumerate(np.asarray(true_pose, np.float32)):
        ap_offsets = [
            sign * float(level)
            for level in SYNTHETIC_AP_LEVELS_UM
            for sign in (-1.0, 1.0)
            if int(round(BREGMA_AP_INDEX - (pose[0] + sign * level) / VOXEL_UM))
            in split_pool
        ]
        lr_offsets = [
            value
            for value in signed_tilt
            if TILT_MIN_DEG <= pose[1] + value <= TILT_MAX_DEG
        ]
        dv_offsets = [
            value
            for value in signed_tilt
            if TILT_MIN_DEG <= pose[2] + value <= TILT_MAX_DEG
        ]
        adjacent_ap = [value for value in ap_offsets if abs(value) == VOXEL_UM]
        required = [
            (float(rng.choice(adjacent_ap or ap_offsets)), 0.0, 0.0)
        ]
        if count >= 2:
            required.append(
                (0.0, float(rng.choice([v for v in lr_offsets if abs(v) == 0.25])), 0.0)
            )
        if count >= 3:
            required.append(
                (0.0, 0.0, float(rng.choice([v for v in dv_offsets if abs(v) == 0.25])))
            )
        pool = (
            [(value, 0.0, 0.0) for value in ap_offsets]
            + [(0.0, value, 0.0) for value in lr_offsets]
            + [(0.0, 0.0, value) for value in dv_offsets]
        )
        pool = [offset for offset in pool if offset not in required]
        remaining = count - len(required)
        if remaining > len(pool):
            raise ValueError("too many distinct split-safe synthetic negatives requested")
        selected = rng.choice(len(pool), remaining, replace=False) if remaining else ()
        wrong[item] = np.asarray(
            required + [pool[int(index)] for index in selected], np.float32
        )
    return wrong


def _true_pose(base_manifest: dict) -> np.ndarray:
    return np.column_stack(
        (
            base_manifest["ap_um"],
            base_manifest["tilt_lr_deg"],
            base_manifest["tilt_dv_deg"],
        )
    ).astype(np.float32)


def _record_identity(record: dict, experiment: dict) -> dict:
    return {
        "animal_id": int(record["specimen_id"]),
        "specimen_id": int(record["specimen_id"]),
        "experiment_id": int(record["experiment_id"]),
        "section_image_id": int(record["section_image_id"]),
        "split": str(record["split"]),
        "product_ids": sorted(int(value) for value in experiment["product_ids"]),
        "relative_path": record.get("relative_path"),
        "section_record_sha256": _payload_sha256(record),
        "experiment_record_sha256": _payload_sha256(experiment),
    }


def _render_physical_poses(renderer, poses: torch.Tensor):
    poses = poses.to(renderer.device, dtype=torch.float32)
    ap_index = BREGMA_AP_INDEX - poses[:, 0] / VOXEL_UM
    return renderer.render_planes(ap_index, poses[:, 1], poses[:, 2])


def _identity_map(batch: int, device, dtype=torch.float32) -> torch.Tensor:
    height, width = MODEL_SHAPE
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y))[None].expand(batch, -1, -1, -1)


def _source_view_homography(
    rotation_deg: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    batch = len(rotation_deg)
    center = rotation_deg.new_tensor(
        ((MODEL_SHAPE[1] - 1.0) / 2.0, (MODEL_SHAPE[0] - 1.0) / 2.0)
    )
    angle = torch.deg2rad(rotation_deg)
    cosine, sine = angle.cos(), angle.sin()
    matrix = scale[:, None, None] * torch.stack(
        (cosine, -sine, sine, cosine), dim=1
    ).reshape(batch, 2, 2)
    homography = torch.eye(
        3, device=rotation_deg.device, dtype=rotation_deg.dtype
    )[None].repeat(batch, 1, 1)
    homography[:, :2, :2] = matrix
    homography[:, :2, 2] = center - torch.einsum("bij,j->bi", matrix, center)
    return homography


def _absolute_map_sample(
    tensor: torch.Tensor,
    pixel_map: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    grid = torch.stack(
        (
            pixel_map[:, 0] * (2.0 / (width - 1)) - 1.0,
            pixel_map[:, 1] * (2.0 / (height - 1)) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def _integer_map_sample(labels: torch.Tensor, pixel_map: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = labels.shape
    x = pixel_map[:, 0].round().long()
    y = pixel_map[:, 1].round().long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    linear = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).flatten(1)
    sampled = labels[:, 0].flatten(1).gather(1, linear).reshape(batch, 1, height, width)
    return torch.where(valid[:, None], sampled, 0)


def _synthetic_outline_input(pair: dict, plan: dict) -> tuple[torch.Tensor, torch.Tensor]:
    raw = pair["moving_raw_uint8"].float() / 255.0
    reference = pair["moving_brush_mask"].bool()
    masks = reference.detach().cpu().numpy()[:, 0]
    modes = np.asarray(plan["mode"], np.int8)
    input_masks = []
    input_alpha = []
    for item, mask in enumerate(masks):
        if modes[item] == 0:
            selected = mask
        elif modes[item] == 1:
            selected = _imperfect_outline(
                mask,
                int(plan["morphology_px"][item]),
                float(plan["jitter_amplitude_px"][item]),
                int(plan["jitter_seed"][item]),
            )
        else:
            selected = np.zeros_like(mask, dtype=bool)
        input_masks.append(selected)
        input_alpha.append(_inside_outline_alpha(selected))
    input_mask = torch.from_numpy(np.stack(input_masks)[:, None]).to(raw.device)
    alpha = torch.from_numpy(np.stack(input_alpha)[:, None]).to(raw.device)
    available = torch.as_tensor(modes != 2, device=raw.device)[:, None, None, None]
    input_image = torch.where(available, raw * alpha, raw)
    return input_image, input_mask


def _apply_source_view(
    pair: dict,
    manifest: dict,
    input_image: torch.Tensor,
    input_mask: torch.Tensor,
) -> dict:
    device = pair["moving"].device
    rotation = torch.as_tensor(
        manifest["source_view_rotation_deg"], device=device, dtype=torch.float32
    )
    scale = torch.as_tensor(
        manifest["source_view_scale"], device=device, dtype=torch.float32
    )
    view_h = _source_view_homography(rotation, scale)
    identity = _identity_map(len(rotation), device)
    inverse_h = torch.linalg.inv(view_h)
    homogeneous = torch.cat((identity, torch.ones_like(identity[:, :1])), dim=1)
    new_to_old = torch.einsum("bij,bjhw->bihw", inverse_h[:, :2], homogeneous)

    source_image = _absolute_map_sample(input_image, new_to_old)
    source_mask = _absolute_map_sample(
        input_mask.float(), new_to_old, mode="nearest"
    ) > 0.5
    outline_mode = torch.as_tensor(
        manifest["outline_plan"]["mode"], device=device, dtype=torch.int8
    )
    mask_available_flat = outline_mode != 2
    source_image = torch.where(
        mask_available_flat[:, None, None, None],
        source_image * source_mask,
        source_image,
    )
    source_tissue = _absolute_map_sample(
        pair["moving_tissue_mask"].float(), new_to_old, mode="nearest"
    ) > 0.5
    source_damage = _absolute_map_sample(
        pair["moving_damage_mask"].float(), new_to_old, mode="nearest"
    ) > 0.5
    source_valid = _absolute_map_sample(
        pair["moving_visible_mask"].float(), new_to_old, mode="nearest"
    ) > 0.5
    source_brush = _absolute_map_sample(
        pair["moving_brush_mask"].float(), new_to_old, mode="nearest"
    ) > 0.5
    source_labels = _integer_map_sample(pair["moving_labels"], new_to_old)
    source_to_fixed = _absolute_map_sample(
        pair["moving_to_fixed"], new_to_old, padding_mode="border"
    )
    fixed_homogeneous = torch.cat(
        (pair["fixed_to_moving"], torch.ones_like(pair["fixed_to_moving"][:, :1])),
        dim=1,
    )
    fixed_to_source = torch.einsum(
        "bij,bjhw->bihw", view_h[:, :2], fixed_homogeneous
    )
    height, width = MODEL_SHAPE
    fixed_inside = (
        (fixed_to_source[:, 0] >= 0.0)
        & (fixed_to_source[:, 0] <= width - 1.0)
        & (fixed_to_source[:, 1] >= 0.0)
        & (fixed_to_source[:, 1] <= height - 1.0)
    )[:, None]
    fixed_valid = pair["fixed_visible_mask"] & fixed_inside

    similarity_h = view_h @ pair["similarity_h"]
    similarity_matrix = similarity_h[:, :2, :2]
    similarity_scale = torch.sqrt(
        similarity_matrix.square().sum((1, 2)) / 2.0
    )
    similarity_rotation = torch.rad2deg(
        torch.atan2(similarity_matrix[:, 1, 0], similarity_matrix[:, 0, 0])
    )
    center = similarity_h.new_tensor(
        ((MODEL_SHAPE[1] - 1.0) / 2.0, (MODEL_SHAPE[0] - 1.0) / 2.0)
    )
    similarity_translation = (
        similarity_h[:, :2, 2]
        - center
        + torch.einsum("bij,j->bi", similarity_matrix, center)
    )
    intersection = (source_mask & source_brush).flatten(1).sum(1).float()
    union = (source_mask | source_brush).flatten(1).sum(1).float().clamp_min(1.0)
    outline_quality = intersection / union
    outline_quality = torch.where(
        mask_available_flat,
        outline_quality,
        torch.full_like(outline_quality, torch.nan),
    )
    return {
        "source_image": source_image.float(),
        "source_mask": source_mask,
        "mask_available": mask_available_flat[:, None, None, None].float(),
        "input_outline_mode": outline_mode,
        "input_outline_mode_name": [
            OUTLINE_MODE_NAMES[int(value)] for value in manifest["outline_plan"]["mode"]
        ],
        "input_outline_quality_iou": outline_quality,
        "input_outline_receipt_sha256": list(
            manifest["outline_plan"]["sample_receipt_sha256"]
        ),
        "input_outline_plan_sha256": manifest["outline_plan"]["plan_sha256"],
        "truth_source_labels": source_labels.long(),
        "truth_source_tissue_mask": source_tissue,
        "truth_source_damage_mask": source_damage,
        "truth_source_valid_mask": source_valid,
        "truth_source_brush_mask": source_brush,
        "truth_fixed_to_source_map": fixed_to_source.float(),
        "truth_source_to_fixed_map": source_to_fixed.float(),
        "truth_fixed_valid_mask": fixed_valid,
        "truth_source_view_h": view_h.float(),
        "truth_generator_similarity_h": pair["similarity_h"].float(),
        "truth_similarity_h": similarity_h.float(),
        "truth_similarity_parameters": torch.cat(
            (
                torch.cos(torch.deg2rad(similarity_rotation))[:, None],
                torch.sin(torch.deg2rad(similarity_rotation))[:, None],
                similarity_translation,
                similarity_scale.log()[:, None],
            ),
            dim=1,
        ).float(),
        "truth_similarity_rotation_deg": similarity_rotation.float(),
        "truth_similarity_scale": similarity_scale.float(),
        "truth_similarity_translation_xy": similarity_translation.float(),
        "truth_source_view_parameters": torch.stack((rotation, scale), dim=1),
    }


def _candidate_batch(
    renderer,
    true_pose: torch.Tensor,
    true_fixed: torch.Tensor,
    true_mask: torch.Tensor,
    true_labels: torch.Tensor,
    wrong_offset: np.ndarray,
) -> dict:
    wrong_offset_tensor = torch.as_tensor(
        wrong_offset, device=renderer.device, dtype=torch.float32
    )
    wrong_pose = true_pose[:, None] + wrong_offset_tensor
    batch, negatives = wrong_pose.shape[:2]
    wrong_fixed, wrong_mask, wrong_labels = _render_physical_poses(
        renderer, wrong_pose.reshape(batch * negatives, 3)
    )
    image_shape = wrong_fixed.shape[1:]
    wrong_fixed = wrong_fixed.reshape(batch, negatives, *image_shape)
    wrong_mask = wrong_mask.reshape(batch, negatives, *image_shape)
    wrong_labels = wrong_labels.reshape(batch, negatives, *image_shape)
    positive = torch.zeros(batch, device=renderer.device, dtype=torch.long)
    positive_mask = torch.zeros(
        batch, negatives + 1, device=renderer.device, dtype=torch.bool
    )
    positive_mask[:, 0] = True
    candidate_pose = torch.cat((true_pose[:, None], wrong_pose), dim=1)
    return {
        "wrong_candidate_offset": wrong_offset_tensor,
        "wrong_candidate_pose": wrong_pose,
        "wrong_candidate_fixed_image": wrong_fixed,
        "wrong_candidate_fixed_mask": wrong_mask,
        "wrong_candidate_fixed_labels": wrong_labels,
        "candidate_pose": candidate_pose,
        "candidate_fixed_image": torch.cat((true_fixed[:, None], wrong_fixed), dim=1),
        "candidate_fixed_mask": torch.cat((true_mask[:, None], wrong_mask), dim=1),
        "candidate_fixed_labels": torch.cat((true_labels[:, None], wrong_labels), dim=1),
        "candidate_in_training_domain": torch.as_tensor(
            _poses_in_training_domain(candidate_pose.detach().cpu().numpy()),
            device=renderer.device,
        ),
        "listwise_target_index": positive,
        "listwise_positive_mask": positive_mask,
    }


def _balanced_signs(count: int, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray((-1.0, 1.0), np.float32)
    base = np.roll(base, int(rng.integers(2)))
    signs = np.resize(base, int(count)).copy()
    rng.shuffle(signs)
    return signs


def _high_tilt_generator_manifest(base: dict, seed: int) -> tuple[dict, np.ndarray]:
    count = len(base["ap_um"])
    mode_rng = _rng(seed, "high-tilt-modes")
    base_modes = np.roll(np.asarray((0, 1, 2), np.int8), int(mode_rng.integers(3)))
    modes = np.resize(base_modes, count).copy()
    mode_rng.shuffle(modes)
    tilt_lr = np.zeros(count, np.float32)
    tilt_dv = np.zeros(count, np.float32)
    for name, axis, active in (
        ("lr", tilt_lr, modes != 1),
        ("dv", tilt_dv, modes != 0),
    ):
        active_count = int(active.sum())
        magnitude = _stratified_uniform(
            _rng(seed, f"high-tilt-magnitude-{name}"),
            active_count,
            15.0,
            35.0,
        )
        signs = _balanced_signs(
            active_count,
            _rng(seed, f"high-tilt-sign-{name}"),
        )
        axis[active] = magnitude * signs
    adjusted = dict(base)
    adjusted["tilt_lr_deg"] = tilt_lr
    adjusted["tilt_dv_deg"] = tilt_dv
    adjusted["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in adjusted.items() if key != "manifest_sha256"}
    )
    return adjusted, modes


class IndependentSyntheticData:
    """Exact synthetic supervision with no learned initializer dependency."""

    def __init__(self, generator):
        self.generator = generator
        contract = {
            "version": INDEPENDENT_DATA_VERSION,
            "source": "synthetic_ccf",
            "source_canvas_contract": SOURCE_CANVAS_CONTRACT,
            "source_view_contract": SOURCE_VIEW_CONTRACT,
            "high_tilt_contract": HIGH_TILT_CONTRACT,
            "outline_curriculum_contract": OUTLINE_CURRICULUM_CONTRACT,
            "mask_availability_contract": (
                "float32[B,1,1,1];1=source_mask supplied;0=mask absent and source_mask zero"
            ),
            "benchmark_primary_mask_policy": "no-user-mask-or-common-automatic-mask",
            "benchmark_assisted_mask_policy": "smart-brush-reported-separately",
            "map_contract": MAP_CONTRACT,
            "similarity_parameter_contract": (
                "[cos(theta),sin(theta),tx_px,ty_px,log(scale)];theta is source-"
                "view rotation;translation is center-relative;homography maps fixed "
                "local-deformation pixels into unified source pixels"
            ),
            "generator_contract_sha256": generator.contract["contract_sha256"],
            "generator_source_sha256": generator.contract.get("generator_source_sha256"),
            "average_template_sha256": generator.contract.get("average_template_sha256"),
            "annotation_sha256": generator.contract.get("annotation_sha256"),
            "adapter_source_sha256": _source_sha256(),
            "learned_checkpoint_dependencies": [],
        }
        contract["contract_sha256"] = _payload_sha256(contract)
        self.contract = contract

    def make_manifest(
        self,
        count: int,
        split: str,
        seed: int,
        stratum: str,
        negatives_per_sample: int = 6,
        *,
        pose_regime: str = "standard",
        _final_capability=None,
    ) -> dict:
        if pose_regime not in {"standard", "high_tilt"}:
            raise ValueError("pose_regime must be standard or high_tilt")
        if pose_regime == "high_tilt" and split not in {"train", "validation"}:
            raise ValueError("high-tilt positives are train/validation data only")
        base = self.generator.make_manifest(
            count, split, seed, stratum, _final_capability=_final_capability
        )
        high_tilt_mode = np.full(count, -1, np.int8)
        if pose_regime == "high_tilt":
            base, high_tilt_mode = _high_tilt_generator_manifest(base, seed)
        true_pose = _true_pose(base)
        source_view_rotation = _stratified_uniform(
            _rng(seed, f"{pose_regime}-source-view-rotation"), count, -180.0, 180.0
        )
        source_view_scale = _stratified_uniform(
            _rng(seed, f"{pose_regime}-source-view-scale"), count, 0.5, 1.5
        )
        outline_plan = _outline_plan(count, seed, pose_regime)
        manifest = {
            "version": INDEPENDENT_DATA_VERSION,
            "contract_sha256": self.contract["contract_sha256"],
            "split": split,
            "seed": int(seed),
            "stratum": stratum,
            "pose_regime": pose_regime,
            "high_tilt_mode": high_tilt_mode,
            "sample_count": int(count),
            "animal_id": np.full(count, -1, np.int64),
            "specimen_id": np.full(count, -1, np.int64),
            "negative_count": int(negatives_per_sample),
            "true_pose": true_pose,
            "source_view_rotation_deg": source_view_rotation,
            "source_view_scale": source_view_scale,
            "outline_plan": outline_plan,
            "wrong_candidate_offset": _synthetic_wrong_offsets(
                true_pose,
                split,
                negatives_per_sample,
                seed,
                _final_capability=_final_capability,
            ),
            "generator_manifest": base,
        }
        manifest["manifest_sha256"] = _payload_sha256(manifest)
        return manifest

    def batch(self, manifest: dict, *, qa: bool = False, _final_capability=None) -> dict:
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != _payload_sha256(payload):
            raise ValueError("independent synthetic manifest hash mismatch")
        if manifest.get("contract_sha256") != self.contract["contract_sha256"]:
            raise ValueError("independent synthetic data contract mismatch")
        pair = self.generator.batch(
            manifest["generator_manifest"], qa=True, _final_capability=_final_capability
        )
        input_image, input_mask = _synthetic_outline_input(
            pair, manifest["outline_plan"]
        )
        source_truth = _apply_source_view(pair, manifest, input_image, input_mask)
        true_pose = torch.as_tensor(
            manifest["true_pose"], device=self.generator.device, dtype=torch.float32
        )
        candidates = _candidate_batch(
            self.generator,
            true_pose,
            pair["fixed"],
            pair["fixed_mask"],
            pair["fixed_labels"],
            manifest["wrong_candidate_offset"],
        )
        result = {
            "source_type": "synthetic_ccf",
            "data_split": str(manifest["split"]),
            "pose_regime": str(manifest["pose_regime"]),
            "high_tilt_mode": torch.as_tensor(
                manifest["high_tilt_mode"],
                device=self.generator.device,
                dtype=torch.int8,
            ),
            "data_contract_sha256": self.contract["contract_sha256"],
            "sample_manifest_sha256": manifest["manifest_sha256"],
            "animal_id": torch.full(
                (len(true_pose),), -1, device=self.generator.device, dtype=torch.int64
            ),
            "specimen_id": torch.full(
                (len(true_pose),), -1, device=self.generator.device, dtype=torch.int64
            ),
            "true_pose": true_pose,
            "truth_fixed_image": pair["fixed"].float(),
            "truth_fixed_mask": pair["fixed_mask"].bool(),
            "truth_fixed_labels": pair["fixed_labels"].long(),
            "truth_svf": pair["local_velocity"].float(),
            "truth_generator_similarity_parameters": torch.as_tensor(
                np.column_stack(
                    (
                        manifest["generator_manifest"]["rotation_deg"],
                        manifest["generator_manifest"]["scale"],
                        manifest["generator_manifest"]["translation_xy"],
                    )
                ),
                device=self.generator.device,
                dtype=torch.float32,
            ),
            "dense_truth_valid": torch.ones(
                len(true_pose), device=self.generator.device, dtype=torch.bool
            ),
            "candidate_dense_truth_valid": torch.nn.functional.pad(
                torch.ones(
                    len(true_pose), 1, device=self.generator.device, dtype=torch.bool
                ),
                (0, int(manifest["negative_count"])),
                value=False,
            ),
            **source_truth,
            **candidates,
        }
        if qa:
            result["qa"] = {
                key: value
                for key, value in pair.items()
                if key.startswith("moving_") and key not in result
            }
        return result

    def generate(
        self,
        count: int,
        split: str,
        seed: int,
        stratum: str,
        negatives_per_sample: int = 6,
        *,
        qa: bool = False,
        pose_regime: str = "standard",
        _final_capability=None,
    ) -> dict:
        return self.batch(
            self.make_manifest(
                count,
                split,
                seed,
                stratum,
                negatives_per_sample,
                pose_regime=pose_regime,
                _final_capability=_final_capability,
            ),
            qa=qa,
            _final_capability=_final_capability,
        )

    def generate_high_tilt(
        self,
        count: int,
        split: str,
        seed: int,
        stratum: str,
        negatives_per_sample: int = 6,
        *,
        qa: bool = False,
    ) -> dict:
        return self.generate(
            count,
            split,
            seed,
            stratum,
            negatives_per_sample,
            qa=qa,
            pose_regime="high_tilt",
        )


class IndependentProduct5Data:
    """Specimen-balanced Product-5 pose/ranking data without dense supervision."""

    def __init__(
        self,
        manifest_root: str | Path,
        atlas_folder: str | Path,
        renderer,
        *,
        split: str = "train",
        dataset: RegisteredSectionDataset | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("Product-5 data supports train or validation only")
        self.root = Path(manifest_root)
        self.atlas_folder = Path(atlas_folder)
        self.renderer = renderer
        self.device = renderer.device
        self.split = split
        self.dataset = dataset or RegisteredSectionDataset(
            self.root,
            self.atlas_folder,
            split=split,
            include_anatomy=False,
            allowed_product_ids=SUPERVISED_PRODUCT_IDS,
        )
        static_contract = registered_static_cache_contract(
            self.root, self.atlas_folder
        )
        static_key = registered_static_cache_key(self.root, self.atlas_folder)
        static_root = self.root / ".atlas_pose_cache" / static_key
        static_contract_path = static_root / "contract.json"
        expected_static_contract = {
            "cache_key": static_key,
            "contract": static_contract,
        }
        self.static_mask_cache_folder = (
            static_root / "training_static"
            if static_contract_path.is_file()
            and json.loads(static_contract_path.read_text(encoding="utf-8"))
            == expected_static_contract
            else None
        )
        specimen_split: dict[int, str] = {}
        for experiment in self.dataset.datasets.values():
            specimen = int(experiment["specimen_id"])
            previous = specimen_split.setdefault(specimen, experiment["split"])
            if previous != experiment["split"]:
                raise RuntimeError("registered specimens overlap data splits")
        self.record_indices = []
        for index, record in enumerate(self.dataset.records):
            experiment = self.dataset.datasets[int(record["experiment_id"])]
            products = {int(value) for value in experiment.get("product_ids", ())}
            if record["split"] != split:
                raise RuntimeError(f"Product-5 {split} adapter contains another split")
            if experiment["split"] != split or int(experiment["specimen_id"]) != int(
                record["specimen_id"]
            ):
                raise RuntimeError("section and experiment specimen/split metadata disagree")
            if not products or not products.issubset(SUPERVISED_PRODUCT_IDS):
                raise RuntimeError("Product-5 adapter contains another Allen product")
            if bool(
                record.get(
                    "in_training_ap_domain",
                    AP_MIN_UM <= float(record["ap_um"]) <= AP_MAX_UM,
                )
            ):
                self.record_indices.append(index)
        if not self.record_indices:
            raise RuntimeError("Product-5 split has no in-domain registered sections")

        selected = [self.dataset.records[index] for index in self.record_indices]
        self.specimen_positions: dict[int, list[int]] = {}
        for position, record in enumerate(selected):
            self.specimen_positions.setdefault(int(record["specimen_id"]), []).append(position)
        specimen_count = len(self.specimen_positions)
        self.sampling_weights = np.asarray(
            [
                1.0
                / (
                    specimen_count
                    * len(self.specimen_positions[int(record["specimen_id"])])
                )
                for record in selected
            ],
            np.float64,
        )
        self.sampling_weights /= self.sampling_weights.sum()

        record_ids = [int(record["section_image_id"]) for record in selected]
        contract = {
            "version": INDEPENDENT_DATA_VERSION,
            "source": "allen_registered_product5",
            "split": split,
            "product_ids": list(SUPERVISED_PRODUCT_IDS),
            "source_canvas_contract": SOURCE_CANVAS_CONTRACT,
            "outline_curriculum_contract": OUTLINE_CURRICULUM_CONTRACT,
            "mask_availability_contract": (
                "float32[B,1,1,1];1=source_mask supplied;0=mask absent and source_mask zero"
            ),
            "benchmark_primary_mask_policy": "no-user-mask-or-common-automatic-mask",
            "benchmark_assisted_mask_policy": "smart-brush-reported-separately",
            "dense_truth_available": False,
            "candidate_schedule": PRODUCT5_CANDIDATE_SCHEDULE,
            "record_count": len(record_ids),
            "record_ids_sha256": _payload_sha256({"section_image_ids": record_ids}),
            "record_identity_fields": (
                "animal_id",
                "specimen_id",
                "experiment_id",
                "section_image_id",
                "split",
                "product_ids",
                "relative_path",
                "section_record_sha256",
                "experiment_record_sha256",
            ),
            "specimen_ids": sorted(self.specimen_positions),
            "quality_manifest_sha256": getattr(
                self.dataset, "quality_manifest_sha256", None
            ),
            "datasets_sha256": _file_sha256(self.root / "datasets.jsonl"),
            "sections_sha256": _file_sha256(self.root / "sections.jsonl"),
            "downloads_sha256": _file_sha256(self.root / "downloads.jsonl"),
            "provenance_sha256": _file_sha256(self.root / "provenance.json"),
            "renderer_contract_sha256": renderer.contract["contract_sha256"],
            "average_template_sha256": renderer.contract.get("average_template_sha256"),
            "annotation_sha256": renderer.contract.get("annotation_sha256"),
            "adapter_source_sha256": _source_sha256(),
            "registered_reader_source_sha256": _file_sha256(
                Path(__file__).with_name("registered_section_dataset.py")
            ),
            "registered_static_mask_contract_sha256": static_key,
            "learned_checkpoint_dependencies": [],
        }
        contract["contract_sha256"] = _payload_sha256(contract)
        self.contract = contract

    def provenance_manifest(self, positions=None) -> dict:
        """Materialize exact raw-record identities only when a run receipt needs them."""
        if positions is None:
            positions = np.arange(len(self.record_indices), dtype=np.int64)
        positions = np.asarray(positions, np.int64)
        records = [
            self.dataset.records[self.record_indices[int(position)]]
            for position in positions
        ]
        identities = [
            _record_identity(
                record, self.dataset.datasets[int(record["experiment_id"])]
            )
            for record in records
        ]
        manifest = {
            "version": INDEPENDENT_DATA_VERSION,
            "data_contract_sha256": self.contract["contract_sha256"],
            "split": self.split,
            "record_identities": identities,
        }
        manifest["manifest_sha256"] = _payload_sha256(manifest)
        return manifest

    def _raw_source(self, dataset_index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.dataset.records[dataset_index]
        if "relative_path" in record:
            with Image.open(self.root / record["relative_path"]) as source:
                image = np.asarray(source).copy()
            cache_path = (
                self.static_mask_cache_folder
                / record["split"]
                / f"{int(record['section_image_id'])}.npz"
                if self.static_mask_cache_folder is not None
                else None
            )
            if cache_path is not None and cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cached:
                    mask = cached["mask"].astype(bool)
            else:
                mask = np.asarray(self.dataset.brain_masker(image), dtype=bool)
            return image, mask
        item = self.dataset[dataset_index]
        return np.asarray(item["raw_image"]), np.asarray(item["raw_mask"], dtype=bool)

    def fixed_validation_positions(self, count: int, seed: int) -> np.ndarray:
        if self.split != "validation":
            raise RuntimeError("fixed validation positions require the validation split")
        rng = _rng(seed, "product5-validation-positions")
        specimens = np.asarray(sorted(self.specimen_positions), np.int64)
        specimens = specimens[rng.permutation(len(specimens))]
        positions = []
        cycle = 0
        while len(positions) < count:
            for specimen in specimens:
                choices = self.specimen_positions[int(specimen)]
                positions.append(choices[(cycle + int(rng.integers(len(choices)))) % len(choices)])
                if len(positions) == count:
                    break
            cycle += 1
        return np.asarray(positions, np.int64)

    def batch_positions(
        self, positions, seed: int, schedule_step: int
    ) -> dict:
        positions = np.asarray(positions, np.int64)
        dataset_indices = [self.record_indices[int(position)] for position in positions]
        records = [self.dataset.records[index] for index in dataset_indices]
        identities = [
            _record_identity(
                record, self.dataset.datasets[int(record["experiment_id"])]
            )
            for record in records
        ]
        outline_plan = _outline_plan(
            len(positions), seed, f"product5-{self.split}-{int(schedule_step)}"
        )
        raw_sources = [self._raw_source(index) for index in dataset_indices]
        canvases = [
            _registered_outline_canvas(
                image,
                mask,
                int(outline_plan["mode"][item]),
                int(outline_plan["morphology_px"][item]),
                float(outline_plan["jitter_amplitude_px"][item]),
                int(outline_plan["jitter_seed"][item]),
            )
            for item, (image, mask) in enumerate(raw_sources)
        ]
        source_image = torch.stack([canvas[0] for canvas in canvases]).to(self.device)
        source_mask = torch.stack([canvas[1] for canvas in canvases]).to(self.device)
        true_pose = torch.as_tensor(
            [
                (record["ap_um"], record["tilt_lr_deg"], record["tilt_dv_deg"])
                for record in records
            ],
            device=self.device,
            dtype=torch.float32,
        )
        true_fixed, true_mask, true_labels = _render_physical_poses(
            self.renderer, true_pose
        )
        offsets, level = product5_candidate_offsets(len(positions), schedule_step)
        candidates = _candidate_batch(
            self.renderer,
            true_pose,
            true_fixed,
            true_mask,
            true_labels,
            offsets,
        )
        record_provenance = [_payload_sha256(identity) for identity in identities]
        batch_manifest_sha256 = _payload_sha256(
            {
                "version": INDEPENDENT_DATA_VERSION,
                "data_contract_sha256": self.contract["contract_sha256"],
                "split": self.split,
                "seed": int(seed),
                "schedule_step": int(schedule_step),
                "candidate_level": level,
                "record_provenance_sha256": record_provenance,
                "outline_plan_sha256": outline_plan["plan_sha256"],
            }
        )
        return {
            "source_type": "allen_registered_product5",
            "data_contract_sha256": self.contract["contract_sha256"],
            "batch_manifest_sha256": batch_manifest_sha256,
            "data_split": self.split,
            "record_provenance_sha256": record_provenance,
            "source_relative_path": [identity["relative_path"] for identity in identities],
            "product_id": torch.full(
                (len(positions),), 5, device=self.device, dtype=torch.int64
            ),
            "source_image": source_image.float(),
            "source_mask": source_mask.bool(),
            "mask_available": torch.as_tensor(
                outline_plan["mode"] != 2, device=self.device, dtype=torch.float32
            )[:, None, None, None],
            "input_outline_mode": torch.as_tensor(
                outline_plan["mode"], device=self.device, dtype=torch.int8
            ),
            "input_outline_mode_name": [
                OUTLINE_MODE_NAMES[int(value)] for value in outline_plan["mode"]
            ],
            "input_outline_quality_iou": torch.as_tensor(
                [canvas[2] for canvas in canvases],
                device=self.device,
                dtype=torch.float32,
            ),
            "input_outline_receipt_sha256": list(
                outline_plan["sample_receipt_sha256"]
            ),
            "input_outline_plan_sha256": outline_plan["plan_sha256"],
            "true_pose": true_pose,
            "dense_truth_valid": torch.zeros(
                len(positions), device=self.device, dtype=torch.bool
            ),
            "candidate_dense_truth_valid": torch.zeros(
                len(positions), 7, device=self.device, dtype=torch.bool
            ),
            "candidate_schedule_step": int(schedule_step),
            "candidate_level_index": level["index"],
            "candidate_level_name": level["name"],
            "candidate_ap_level_um": level["ap_um"],
            "candidate_tilt_level_deg": level["tilt_deg"],
            "specimen_id": torch.as_tensor(
                [record["specimen_id"] for record in records],
                device=self.device,
                dtype=torch.int64,
            ),
            "animal_id": torch.as_tensor(
                [record["specimen_id"] for record in records],
                device=self.device,
                dtype=torch.int64,
            ),
            "experiment_id": torch.as_tensor(
                [record["experiment_id"] for record in records],
                device=self.device,
                dtype=torch.int64,
            ),
            "section_image_id": torch.as_tensor(
                [record["section_image_id"] for record in records],
                device=self.device,
                dtype=torch.int64,
            ),
            **candidates,
        }

    def generate(
        self, count: int, seed: int, schedule_step: int
    ) -> dict:
        rng = _rng(seed, "product5-sampling")
        positions = rng.choice(
            len(self.record_indices), int(count), replace=True, p=self.sampling_weights
        )
        return self.batch_positions(positions, seed, schedule_step)
