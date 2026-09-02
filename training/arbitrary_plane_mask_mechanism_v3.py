"""Deterministic smart-brush mechanism ablations for the frozen development replay."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import ndimage


MASK_MECHANISM_SCHEMA = "anatomy-tracker.arbitrary-plane-mask-mechanism/v3"
MASK_VARIANTS = (
    "accurate",
    "morphology-only",
    "jitter-gap-island-only",
    "full-imperfect",
)
SCIENTIFIC_AMBIGUITY = (
    "The morphology and jitter/gap/island stages are nonlinear and order-dependent. "
    "These paired ablations localize association with the frozen failure but are not "
    "an additive or uniquely causal decomposition of mask error."
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def array_receipt(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(np.asarray(array))
    storage = "packbits-little" if value.dtype == np.bool_ else "C-order"
    body = (
        np.packbits(value.reshape(-1), bitorder="little").tobytes()
        if value.dtype == np.bool_
        else value.tobytes(order="C")
    )
    header = _canonical_json(
        {"dtype": value.dtype.str, "shape": list(value.shape), "storage": storage}
    ).encode("utf-8")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "nbytes": int(value.nbytes),
        "stored_nbytes": len(body),
        "storage": storage,
        "array_sha256": hashlib.sha256(header + b"\0" + body).hexdigest(),
    }


def _smooth_unit_field(
    shape: tuple[int, int], rng: np.random.Generator, sigma_px: float
) -> np.ndarray:
    field = rng.normal(size=shape).astype(np.float32)
    if sigma_px > 0.0:
        field = ndimage.gaussian_filter(field, sigma=float(sigma_px), mode="reflect")
    field -= float(field.mean())
    scale = float(field.std())
    return field / scale if scale > 1e-6 else np.zeros(shape, np.float32)


def _ellipse(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
    angle_rad: float,
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    dx, dy = x - float(center_xy[0]), y - float(center_xy[1])
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rx, ry = cosine * dx + sine * dy, -sine * dx + cosine * dy
    return (rx / max(float(radius_xy[0]), 1.0)) ** 2 + (
        ry / max(float(radius_xy[1]), 1.0)
    ) ** 2 <= 1.0


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = (float(value) for value in bounds)
    if not np.isfinite((low, high)).all() or high < low:
        raise ValueError("range bounds must be finite and ordered")
    return low if low == high else float(rng.uniform(low, high))


def _mechanism_mask(
    footprint: np.ndarray,
    seed_uint64: int,
    *,
    apply_morphology: bool,
    apply_jitter_gap_island: bool,
    morphology_radius_px: int,
    jitter_amplitude_px: float,
    gap_radius_fraction: tuple[float, float],
    island_radius_fraction: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Replay the legacy stream while selectively suppressing its mask stages."""
    rng = np.random.Generator(np.random.PCG64(int(seed_uint64)))
    height, width = footprint.shape
    choices = np.asarray(
        [
            value
            for value in range(-int(morphology_radius_px), int(morphology_radius_px) + 1)
            if value
        ],
        dtype=np.int16,
    )
    sampled_morphology = int(rng.choice(choices)) if len(choices) else 0
    applied_morphology = sampled_morphology if apply_morphology else 0
    outline = footprint.copy()
    if applied_morphology > 0:
        outline = ndimage.binary_dilation(outline, iterations=applied_morphology)
    elif applied_morphology < 0:
        outline = ndimage.binary_erosion(outline, iterations=-applied_morphology)

    gap_applied = island_applied = False
    if apply_jitter_gap_island:
        amplitude = float(jitter_amplitude_px)
        if amplitude > 0.0:
            sigma = max(height, width) / 16.0
            dx = amplitude * _smooth_unit_field(footprint.shape, rng, sigma)
            dy = amplitude * _smooth_unit_field(footprint.shape, rng, sigma)
            y, x = np.mgrid[:height, :width].astype(np.float32)
            outline = ndimage.map_coordinates(
                outline.astype(np.uint8),
                (y + dy, x + dx),
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(bool)
        boundary = outline & ~ndimage.binary_erosion(outline)
        gap_yx = np.argwhere(boundary)
        if len(gap_yx):
            gap_y, gap_x = gap_yx[int(rng.integers(len(gap_yx)))]
            gap = _ellipse(
                footprint.shape,
                (float(gap_x), float(gap_y)),
                (
                    width * _uniform(rng, gap_radius_fraction),
                    height * _uniform(rng, gap_radius_fraction),
                ),
                float(rng.uniform(-np.pi, np.pi)),
            )
            outline[gap] = False
            gap_applied = True
        ring = ndimage.binary_dilation(
            outline, iterations=max(2, int(morphology_radius_px) + 2)
        ) & ~outline
        island_yx = np.argwhere(ring)
        if len(island_yx):
            island_y, island_x = island_yx[int(rng.integers(len(island_yx)))]
            island = _ellipse(
                footprint.shape,
                (float(island_x), float(island_y)),
                (
                    width * _uniform(rng, island_radius_fraction),
                    height * _uniform(rng, island_radius_fraction),
                ),
                float(rng.uniform(-np.pi, np.pi)),
            )
            outline[island] = True
            island_applied = True
    return np.ascontiguousarray(outline, dtype=bool), {
        "perturbation_seed_uint64": f"0x{int(seed_uint64):016x}",
        "sampled_morphology_px": sampled_morphology,
        "applied_morphology_px": applied_morphology,
        "jitter_amplitude_px": (
            float(jitter_amplitude_px) if apply_jitter_gap_island else 0.0
        ),
        "gap_applied": gap_applied,
        "island_applied": island_applied,
    }


def paired_smart_brush_masks(
    observable_footprint_mask: np.ndarray,
    perturbation_seed_uint64: int | str,
    *,
    morphology_radius_px: int = 4,
    jitter_amplitude_px: float = 1.92,
    gap_radius_fraction: tuple[float, float] = (0.02, 0.06),
    island_radius_fraction: tuple[float, float] = (0.01, 0.04),
) -> dict[str, dict[str, object]]:
    """Return the four paired masks from one frozen accepted perturbation stream."""
    footprint = np.ascontiguousarray(np.asarray(observable_footprint_mask, dtype=bool))
    if footprint.ndim != 2 or not footprint.any():
        raise ValueError("observable_footprint_mask must be one nonempty H-by-W mask")
    if isinstance(perturbation_seed_uint64, str):
        if (
            len(perturbation_seed_uint64) != 18
            or not perturbation_seed_uint64.startswith("0x")
            or perturbation_seed_uint64.lower() != perturbation_seed_uint64
        ):
            raise ValueError("perturbation seed must be canonical uint64 hex")
        seed = int(perturbation_seed_uint64, 16)
    else:
        seed = int(perturbation_seed_uint64)
    if not 0 <= seed < 2**64 or int(morphology_radius_px) < 0:
        raise ValueError("seed and morphology radius are outside their declared ranges")

    masks = {"accurate": footprint.copy()}
    parameters: dict[str, dict[str, object]] = {"accurate": {}}
    for name, morphology, nonlinear in (
        ("morphology-only", True, False),
        ("jitter-gap-island-only", False, True),
        ("full-imperfect", True, True),
    ):
        masks[name], parameters[name] = _mechanism_mask(
            footprint,
            seed,
            apply_morphology=morphology,
            apply_jitter_gap_island=nonlinear,
            morphology_radius_px=int(morphology_radius_px),
            jitter_amplitude_px=float(jitter_amplitude_px),
            gap_radius_fraction=gap_radius_fraction,
            island_radius_fraction=island_radius_fraction,
        )

    return {
        name: {
            "mask": masks[name],
            "mask_receipt": array_receipt(masks[name]),
            "quality_iou": float(
                np.count_nonzero(masks[name] & footprint)
                / np.count_nonzero(masks[name] | footprint)
            ),
            "parameters": parameters[name],
        }
        for name in MASK_VARIANTS
    }


def paired_smart_brush_inputs(
    acquired_image: np.ndarray,
    observable_footprint_mask: np.ndarray,
    perturbation_seed_uint64: int | str,
    **mask_options: object,
) -> dict[str, dict[str, object]]:
    """Apply every paired mask to the same acquired image with exact black exterior."""
    image = np.ascontiguousarray(np.asarray(acquired_image))
    if image.ndim != 2 or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("acquired_image must be one finite float32 H-by-W raster")
    variants = paired_smart_brush_masks(
        observable_footprint_mask, perturbation_seed_uint64, **mask_options
    )
    for item in variants.values():
        if item["mask"].shape != image.shape:
            raise ValueError("acquired image and footprint shapes differ")
        model_input = np.zeros(image.shape, dtype=np.float32)
        model_input[item["mask"]] = image[item["mask"]]
        item["model_input_image"] = model_input
        item["model_input_image_receipt"] = array_receipt(model_input)
        item["black_exterior_exact"] = bool(np.all(model_input[~item["mask"]] == 0.0))
    return variants
