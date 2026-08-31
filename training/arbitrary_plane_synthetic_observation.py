"""Procedural appearance, damage and optional-outline primitives for v3 slices.

All stochastic functions require an explicit ``numpy.random.Generator``.  The
module consumes no learned asset and deliberately has no dependency on any
historical model or synthetic generator.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


ACCURATE_OUTLINE = "accurate-outline-black-exterior"
IMPERFECT_OUTLINE = "imperfect-outline-black-exterior"
ABSENT_OUTLINE = "absent-outline-acquired-background"
SMART_BRUSH_MODES = (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)


def _rng(value: np.random.Generator) -> np.random.Generator:
    if not isinstance(value, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    return value


def _same_shape(reference: np.ndarray, **arrays: np.ndarray) -> None:
    for name, value in arrays.items():
        if np.asarray(value).shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}")


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = (float(value) for value in bounds)
    if not np.isfinite((low, high)).all() or high < low:
        raise ValueError("range bounds must be finite and ordered")
    return low if low == high else float(rng.uniform(low, high))


def _smooth_unit_field(
    shape: tuple[int, int], rng: np.random.Generator, sigma_px: float
) -> np.ndarray:
    field = rng.normal(size=shape).astype(np.float32)
    if sigma_px > 0.0:
        field = ndimage.gaussian_filter(field, sigma=float(sigma_px), mode="reflect")
    field -= float(field.mean())
    scale = float(field.std())
    return field / scale if scale > 1e-6 else np.zeros(shape, np.float32)


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source_h, source_w = image.shape
    target_h, target_w = shape
    y = np.linspace(0.0, max(source_h - 1, 0), target_h, dtype=np.float32)
    x = np.linspace(0.0, max(source_w - 1, 0), target_w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return ndimage.map_coordinates(
        image, (yy, xx), order=1, mode="nearest", prefilter=False
    ).astype(np.float32)


def _ellipse(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
    angle_rad: float,
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    dx = x - float(center_xy[0])
    dy = y - float(center_xy[1])
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rx = cosine * dx + sine * dy
    ry = -sine * dx + cosine * dy
    return (rx / max(float(radius_xy[0]), 1.0)) ** 2 + (
        ry / max(float(radius_xy[1]), 1.0)
    ) ** 2 <= 1.0


def robust_clean_normalization(
    scalar: np.ndarray,
    tissue_mask: np.ndarray,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """Winsorize and scale tissue intensities to [0, 1], leaving exterior zero."""
    return _robust_clean_normalization_with_receipt(
        scalar, tissue_mask, quantiles
    )[0]


def _robust_clean_normalization_with_receipt(
    scalar: np.ndarray,
    tissue_mask: np.ndarray,
    quantiles: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    scalar = np.asarray(scalar)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if scalar.ndim != 2:
        raise ValueError("scalar must be one H-by-W array")
    _same_shape(scalar, tissue_mask=tissue)
    if not np.issubdtype(scalar.dtype, np.number) or not np.isfinite(scalar).all():
        raise ValueError("scalar must be finite and numeric")
    lower_q, upper_q = (float(value) for value in quantiles)
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    result = np.zeros(scalar.shape, np.float32)
    if not tissue.any():
        return result, {
            "method": "empty-tissue",
            "tissue_pixel_count": 0,
            "quantiles": None,
            "lower": None,
            "upper": None,
        }
    values = scalar[tissue].astype(np.float64)
    if len(values) >= 256:
        lower, upper = np.quantile(values, (lower_q, upper_q))
        method = "quantile"
        effective_quantiles: list[float] | None = [lower_q, upper_q]
    else:
        lower, upper = float(values.min()), float(values.max())
        method = "min-max-fallback"
        effective_quantiles = None
    if upper <= lower:
        result[tissue] = 0.5
    else:
        result[tissue] = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return result, {
        "method": method,
        "tissue_pixel_count": int(len(values)),
        "quantiles": effective_quantiles,
        "lower": float(lower),
        "upper": float(upper),
    }


def mix_template_and_labels(
    normalized_template: np.ndarray,
    labels: np.ndarray,
    tissue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    template_weight: float = 1.0,
    label_weight: float = 0.0,
    label_level_range: tuple[float, float] = (0.15, 0.85),
) -> np.ndarray:
    """Mix template contrast with per-region draws without using label magnitudes."""
    return _mix_template_and_labels_with_receipt(
        normalized_template,
        labels,
        tissue_mask,
        rng,
        template_weight=template_weight,
        label_weight=label_weight,
        label_level_range=label_level_range,
    )[0]


def _mix_template_and_labels_with_receipt(
    normalized_template: np.ndarray,
    labels: np.ndarray,
    tissue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    template_weight: float,
    label_weight: float,
    label_level_range: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    rng = _rng(rng)
    template = np.asarray(normalized_template, dtype=np.float32)
    labels = np.asarray(labels)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if template.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("template must be H-by-W and labels must be integer")
    _same_shape(template, labels=labels, tissue_mask=tissue)
    if not np.isfinite(template).all():
        raise ValueError("template must be finite")
    template_weight, label_weight = float(template_weight), float(label_weight)
    if min(template_weight, label_weight) < 0.0 or template_weight + label_weight == 0.0:
        raise ValueError("mixing weights must be nonnegative and not both zero")
    mixed = np.zeros(template.shape, np.float32)
    if label_weight == 0.0 or not tissue.any():
        mixed[tissue] = template[tissue]
        return mixed, {
            "template_weight": template_weight,
            "label_weight": label_weight,
            "label_level_range": [float(value) for value in label_level_range],
            "region_ids": [],
            "region_levels": [],
        }
    label_image = template.copy()
    region_ids = np.unique(labels[tissue])
    levels = rng.uniform(*label_level_range, size=len(region_ids)).astype(np.float32)
    for ordinal, region_id in enumerate(region_ids):
        label_image[tissue & (labels == region_id)] = levels[ordinal]
    mixed[tissue] = (
        template_weight * template[tissue] + label_weight * label_image[tissue]
    ) / (template_weight + label_weight)
    return np.clip(mixed, 0.0, 1.0), {
        "template_weight": template_weight,
        "label_weight": label_weight,
        "label_level_range": [float(value) for value in label_level_range],
        "region_ids": [value.item() for value in region_ids],
        "region_levels": [float(value) for value in levels],
    }


def synthesize_appearance(
    scalar: np.ndarray,
    labels: np.ndarray,
    tissue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    clean_path: bool = False,
    normalization_quantiles: tuple[float, float] = (0.01, 0.99),
    template_weight: float = 0.75,
    label_weight: float = 0.25,
    label_level_range: tuple[float, float] = (0.15, 0.85),
    gain_range: tuple[float, float] = (0.75, 1.25),
    offset_range: tuple[float, float] = (-0.10, 0.10),
    gamma_range: tuple[float, float] = (0.7, 1.5),
    polarity_probability: float = 0.10,
    bias_strength_range: tuple[float, float] = (0.0, 0.30),
    blur_sigma_range: tuple[float, float] = (0.0, 1.25),
    downsample_factor_range: tuple[float, float] = (1.0, 2.0),
    noise_std_range: tuple[float, float] = (0.0, 0.04),
    background_level_range: tuple[float, float] = (0.02, 0.35),
    background_texture_range: tuple[float, float] = (0.0, 0.12),
    artifact_density_range: tuple[float, float] = (0.0, 0.0015),
) -> dict[str, object]:
    """Create clean/procedural grayscale tissue and an acquired background."""
    rng = _rng(rng)
    scalar = np.asarray(scalar)
    labels = np.asarray(labels)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if scalar.ndim != 2:
        raise ValueError("appearance inputs must be H-by-W arrays")
    _same_shape(scalar, labels=labels, tissue_mask=tissue)
    clean, normalization_receipt = _robust_clean_normalization_with_receipt(
        scalar, tissue, normalization_quantiles
    )
    mixed, label_style_receipt = _mix_template_and_labels_with_receipt(
        clean,
        labels,
        tissue,
        rng,
        template_weight=1.0 if clean_path else template_weight,
        label_weight=0.0 if clean_path else label_weight,
        label_level_range=label_level_range,
    )
    height, width = scalar.shape
    background_level = _uniform(rng, background_level_range)
    background_strength = _uniform(rng, background_texture_range)
    background = background_level + background_strength * _smooth_unit_field(
        scalar.shape, rng, max(height, width) / 12.0
    )
    background = np.clip(background, 0.0, 1.0).astype(np.float32)
    artifact_mask = np.zeros(scalar.shape, bool)
    parameters: dict[str, object] = {
        "path": "clean" if clean_path else "augmented",
        "normalization": normalization_receipt,
        "label_style": label_style_receipt,
        "background_level": background_level,
        "background_texture_strength": background_strength,
        "gain": 1.0,
        "offset": 0.0,
    }
    appearance = mixed.copy()
    if not clean_path:
        polarity = bool(rng.random() < float(polarity_probability))
        gamma = _uniform(rng, gamma_range)
        gain = _uniform(rng, gain_range)
        offset = _uniform(rng, offset_range)
        bias_strength = _uniform(rng, bias_strength_range)
        blur_sigma = _uniform(rng, blur_sigma_range)
        downsample_factor = _uniform(rng, downsample_factor_range)
        noise_std = _uniform(rng, noise_std_range)
        artifact_density = _uniform(rng, artifact_density_range)
        if polarity:
            appearance[tissue] = 1.0 - appearance[tissue]
        appearance[tissue] = np.power(
            np.clip(appearance[tissue], 0.0, 1.0), gamma
        )
        bias = np.exp(
            bias_strength
            * _smooth_unit_field(scalar.shape, rng, max(height, width) / 10.0)
        ).astype(np.float32)
        appearance[tissue] = appearance[tissue] * gain * bias[tissue] + offset
        if blur_sigma > 0.0:
            appearance = ndimage.gaussian_filter(
                appearance, sigma=blur_sigma, mode="nearest"
            ).astype(np.float32)
        if downsample_factor > 1.0:
            low_shape = (
                max(1, int(round(height / downsample_factor))),
                max(1, int(round(width / downsample_factor))),
            )
            appearance = _resize(_resize(appearance, low_shape), scalar.shape)
        if noise_std > 0.0:
            appearance[tissue] += rng.normal(0.0, noise_std, int(tissue.sum())).astype(
                np.float32
            )
        if artifact_density > 0.0:
            artifact_mask = rng.random(scalar.shape) < artifact_density
            artifact_mask = ndimage.maximum_filter(
                artifact_mask, size=3, mode="constant"
            ).astype(bool) & tissue
            artifact_value = float(rng.uniform(0.0, 1.0))
            appearance[artifact_mask] = artifact_value
        parameters.update(
            polarity=polarity,
            gamma=gamma,
            gain=gain,
            offset=offset,
            bias_strength=bias_strength,
            blur_sigma=blur_sigma,
            downsample_factor=downsample_factor,
            noise_std=noise_std,
            artifact_density=artifact_density,
        )
    appearance = np.clip(appearance, 0.0, 1.0).astype(np.float32)
    pre_damage = np.where(tissue, appearance, background).astype(np.float32)
    return {
        "clean_grayscale": clean,
        "mixed_clean_grayscale": mixed,
        "tissue_appearance": appearance,
        "acquired_background": background,
        "pre_damage_image": pre_damage,
        "appearance_artifact_mask": artifact_mask,
        "parameters": parameters,
    }


def sample_damage_masks(
    tissue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    physical_loss_probability: float = 0.35,
    occlusion_probability: float = 0.35,
    radius_fraction_range: tuple[float, float] = (0.06, 0.20),
) -> dict[str, object]:
    """Sample disjoint physical-loss and occlusion ellipses within tissue."""
    rng = _rng(rng)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if tissue.ndim != 2:
        raise ValueError("tissue_mask must be one H-by-W array")
    height, width = tissue.shape
    physical = np.zeros(tissue.shape, bool)
    occlusion = np.zeros(tissue.shape, bool)
    parameters: dict[str, object] = {}
    boundary = tissue & ~ndimage.binary_erosion(tissue)
    for name, probability, candidates in (
        ("physical_loss", physical_loss_probability, boundary),
        ("occlusion", occlusion_probability, tissue),
    ):
        enabled = bool(tissue.any() and rng.random() < float(probability))
        parameters[f"{name}_enabled"] = enabled
        if not enabled:
            continue
        yx = np.argwhere(candidates if candidates.any() else tissue)
        center_y, center_x = yx[int(rng.integers(len(yx)))]
        radius_x = max(1.0, width * _uniform(rng, radius_fraction_range))
        radius_y = max(1.0, height * _uniform(rng, radius_fraction_range))
        angle = float(rng.uniform(-np.pi, np.pi))
        mask = _ellipse(
            tissue.shape,
            (float(center_x), float(center_y)),
            (radius_x, radius_y),
            angle,
        ) & tissue
        if name == "physical_loss":
            physical = mask
        else:
            occlusion = mask & ~physical
        parameters[name] = {
            "center_xy": [float(center_x), float(center_y)],
            "radius_xy": [radius_x, radius_y],
            "angle_rad": angle,
        }
    return {
        "physical_loss_mask": physical,
        "occlusion_mask": occlusion,
        "parameters": parameters,
    }


def apply_damage(
    pre_damage_image: np.ndarray,
    acquired_background: np.ndarray,
    tissue_mask: np.ndarray,
    physical_loss_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    appearance_artifact_mask: np.ndarray,
    *,
    occlusion_value: float,
    map_domain_valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply damage after appearance and return explicit truth-mask algebra."""
    image = np.asarray(pre_damage_image, dtype=np.float32)
    background = np.asarray(acquired_background, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    physical = np.asarray(physical_loss_mask, dtype=bool)
    occlusion = np.asarray(occlusion_mask, dtype=bool)
    artifact = np.asarray(appearance_artifact_mask, dtype=bool)
    if image.ndim != 2:
        raise ValueError("damage inputs must be H-by-W arrays")
    _same_shape(
        image,
        acquired_background=background,
        tissue_mask=tissue,
        physical_loss_mask=physical,
        occlusion_mask=occlusion,
        appearance_artifact_mask=artifact,
    )
    if not np.isfinite(image).all() or not np.isfinite(background).all():
        raise ValueError("damage image and background must be finite")
    if np.any((physical | occlusion) & ~tissue) or np.any(physical & occlusion):
        raise ValueError("physical loss and occlusion must be disjoint tissue subsets")
    map_valid = (
        np.ones(image.shape, bool)
        if map_domain_valid_mask is None
        else np.asarray(map_domain_valid_mask, dtype=bool)
    )
    _same_shape(image, map_domain_valid_mask=map_valid)
    damaged = image.copy()
    damaged[physical] = background[physical]
    damaged[occlusion] = float(occlusion_value)
    damage = physical | occlusion
    footprint = tissue & ~physical
    observation_invalid = tissue & (damage | artifact)
    valid = map_valid & tissue & ~observation_invalid
    return {
        "damaged_acquired_image": np.clip(damaged, 0.0, 1.0).astype(np.float32),
        "physical_loss_mask": physical.copy(),
        "occlusion_mask": occlusion.copy(),
        "appearance_artifact_mask": artifact.copy(),
        "damage_mask": damage,
        "observable_footprint_mask": footprint,
        "observation_invalid_mask": observation_invalid,
        "valid_correspondence_mask": valid,
    }


def synthesize_damage(
    pre_damage_image: np.ndarray,
    acquired_background: np.ndarray,
    tissue_mask: np.ndarray,
    appearance_artifact_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    map_domain_valid_mask: np.ndarray | None = None,
    physical_loss_probability: float = 0.35,
    occlusion_probability: float = 0.35,
    radius_fraction_range: tuple[float, float] = (0.06, 0.20),
    occlusion_value_range: tuple[float, float] = (0.0, 0.25),
) -> dict[str, object]:
    """Sample and apply physical loss/occlusion after appearance."""
    rng = _rng(rng)
    sampled = sample_damage_masks(
        tissue_mask,
        rng,
        physical_loss_probability=physical_loss_probability,
        occlusion_probability=occlusion_probability,
        radius_fraction_range=radius_fraction_range,
    )
    occlusion_value = _uniform(rng, occlusion_value_range)
    result = apply_damage(
        pre_damage_image,
        acquired_background,
        tissue_mask,
        sampled["physical_loss_mask"],
        sampled["occlusion_mask"],
        appearance_artifact_mask,
        occlusion_value=occlusion_value,
        map_domain_valid_mask=map_domain_valid_mask,
    )
    result["parameters"] = {
        **sampled["parameters"],
        "occlusion_value": occlusion_value,
    }
    return result


def _imperfect_outline(
    footprint: np.ndarray,
    rng: np.random.Generator,
    morphology_radius_px: int,
    jitter_amplitude_px: float,
    gap_radius_fraction: tuple[float, float],
    island_radius_fraction: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = footprint.shape
    choices = np.asarray(
        [value for value in range(-morphology_radius_px, morphology_radius_px + 1) if value],
        dtype=np.int16,
    )
    morphology = int(rng.choice(choices)) if len(choices) else 0
    outline = footprint.copy()
    if morphology > 0:
        outline = ndimage.binary_dilation(outline, iterations=morphology)
    elif morphology < 0:
        outline = ndimage.binary_erosion(outline, iterations=-morphology)
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
    gap = None
    if len(gap_yx):
        gap_y, gap_x = gap_yx[int(rng.integers(len(gap_yx)))]
        gap_radius = (
            width * _uniform(rng, gap_radius_fraction),
            height * _uniform(rng, gap_radius_fraction),
        )
        gap = _ellipse(
            footprint.shape,
            (float(gap_x), float(gap_y)),
            gap_radius,
            float(rng.uniform(-np.pi, np.pi)),
        )
        outline[gap] = False
    ring = ndimage.binary_dilation(outline, iterations=max(2, morphology_radius_px + 2)) & ~outline
    island_yx = np.argwhere(ring)
    island = None
    if len(island_yx):
        island_y, island_x = island_yx[int(rng.integers(len(island_yx)))]
        island_radius = (
            width * _uniform(rng, island_radius_fraction),
            height * _uniform(rng, island_radius_fraction),
        )
        island = _ellipse(
            footprint.shape,
            (float(island_x), float(island_y)),
            island_radius,
            float(rng.uniform(-np.pi, np.pi)),
        )
        outline[island] = True
    return outline, {
        "morphology_px": morphology,
        "jitter_amplitude_px": amplitude,
        "gap_applied": gap is not None,
        "island_applied": island is not None,
    }


def smart_brush_input(
    acquired_image: np.ndarray,
    observable_footprint_mask: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    *,
    morphology_radius_px: int = 3,
    jitter_amplitude_px: float = 1.5,
    gap_radius_fraction: tuple[float, float] = (0.02, 0.06),
    island_radius_fraction: tuple[float, float] = (0.01, 0.04),
) -> dict[str, object]:
    """Create the optional model-input outline without constructing truth masks."""
    rng = _rng(rng)
    image = np.asarray(acquired_image, dtype=np.float32)
    footprint = np.asarray(observable_footprint_mask, dtype=bool)
    if image.ndim != 2 or mode not in SMART_BRUSH_MODES:
        raise ValueError("smart-brush input requires an H-by-W image and a named mode")
    _same_shape(image, observable_footprint_mask=footprint)
    if mode == ABSENT_OUTLINE:
        outline = np.zeros(image.shape, bool)
        input_image = image.copy()
        available = False
        quality = None
        parameters: dict[str, object] = {}
    else:
        outline = footprint.copy()
        parameters = {}
        if mode == IMPERFECT_OUTLINE:
            outline, parameters = _imperfect_outline(
                footprint,
                rng,
                int(morphology_radius_px),
                float(jitter_amplitude_px),
                gap_radius_fraction,
                island_radius_fraction,
            )
        input_image = np.zeros(image.shape, np.float32)
        input_image[outline] = image[outline]
        available = True
        union = int((outline | footprint).sum())
        quality = float((outline & footprint).sum() / union) if union else 1.0
    return {
        "input_image": input_image,
        "input_outline_mask": outline,
        "outline_available": available,
        "outline_mode": mode,
        "outline_quality_iou": quality,
        "parameters": parameters,
    }
