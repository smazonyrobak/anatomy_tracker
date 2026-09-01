"""Deterministic image-information primitives for the frozen arbitrary-plane pilot."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, convolve, distance_transform_edt, map_coordinates

from training.arbitrary_plane_synthetic_ops import bilinear_sample_scalar


IMAGE_INFORMATION_ALGORITHM = "arbitrary-plane-image-information/v1"
TIE_TOLERANCE = 1.0e-12
MIND_SEARCH_DISPLACEMENT_UM = 100.0
MIND_GAUSSIAN_SIGMA_UM = 50.0
MIND_GAUSSIAN_TRUNCATE_SIGMA = 3.0
MIND_GAUSSIAN_RADIUS_UM = 150.0
CONTEXT_RADIUS_UM = 1000.0
ORDINARY_SCALAR_PADDING = 0.0
CANDIDATE_INTENSITY_LOWER = 6.0
CANDIDATE_INTENSITY_RANGE = 281.0


def dewarp_target_float32(
    model_input_image: np.ndarray,
    fixed_to_source_map: np.ndarray,
) -> np.ndarray:
    """Apply the frozen float32 source-to-fixed bilinear dewarp."""
    image = np.asarray(model_input_image)
    pixel_map = np.asarray(fixed_to_source_map)
    if image.ndim != 2 or image.dtype != np.float32:
        raise ValueError("model_input_image must be one float32 H-by-W raster")
    if pixel_map.shape != (2, *image.shape) or pixel_map.dtype != np.float32:
        raise ValueError("fixed_to_source_map must be float32 with shape (2,H,W)")
    return bilinear_sample_scalar(image, pixel_map, padding_mode="zeros")


def dewarp_target_for_scoring(
    model_input_image: np.ndarray,
    fixed_to_source_map: np.ndarray,
) -> np.ndarray:
    """Dewarp in float32, then promote the exact result to C-order float64."""
    return np.array(
        dewarp_target_float32(model_input_image, fixed_to_source_map),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def scale_candidate_raster(candidate_scalar_float32: np.ndarray) -> np.ndarray:
    """Apply the frozen global scaling after rendering a float32 2-D raster."""
    scalar = np.asarray(candidate_scalar_float32)
    if scalar.ndim != 2 or scalar.dtype != np.float32:
        raise ValueError("candidate scalar must be one rendered float32 H-by-W raster")
    values = np.array(scalar, dtype=np.float64, copy=True, order="C")
    return np.ascontiguousarray(
        np.clip(
            (values - CANDIDATE_INTENSITY_LOWER) / CANDIDATE_INTENSITY_RANGE,
            0.0,
            1.0,
        )
    )


def four_corner_safe_mask(
    source_mask: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_to_source_map: np.ndarray,
) -> np.ndarray:
    """Require all four unconditional bilinear source corners and the fixed mask."""
    source = np.asarray(source_mask)
    fixed = np.asarray(fixed_mask)
    pixel_map = np.asarray(fixed_to_source_map)
    if source.ndim != 2 or fixed.ndim != 2 or source.dtype != np.bool_ or fixed.dtype != np.bool_:
        raise ValueError("source_mask and fixed_mask must be two-dimensional Boolean arrays")
    if pixel_map.shape != (2, *fixed.shape) or pixel_map.dtype != np.float32:
        raise ValueError("fixed_to_source_map must be float32 with shape (2,H,W)")
    if not np.isfinite(pixel_map).all():
        raise ValueError("fixed_to_source_map must be finite")

    x0 = np.floor(pixel_map[0]).astype(np.int64)
    y0 = np.floor(pixel_map[1]).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    height, width = source.shape
    inside = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)
    cx0 = np.clip(x0, 0, width - 1)
    cx1 = np.clip(x1, 0, width - 1)
    cy0 = np.clip(y0, 0, height - 1)
    cy1 = np.clip(y1, 0, height - 1)
    corners = (
        source[cy0, cx0]
        & source[cy0, cx1]
        & source[cy1, cx0]
        & source[cy1, cx1]
    )
    return np.ascontiguousarray(fixed & inside & corners)


def mind_parameters(pixel_pitch_um: float) -> dict[str, object]:
    """Construct the frozen physical MIND offsets, Gaussian kernel and footprint."""
    pitch = float(pixel_pitch_um)
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pixel_pitch_um must be finite and positive")
    axial = max(1, math.floor(MIND_SEARCH_DISPLACEMENT_UM / pitch + 0.5))
    diagonal = max(
        1,
        math.floor(MIND_SEARCH_DISPLACEMENT_UM / (math.sqrt(2.0) * pitch) + 0.5),
    )
    sigma = MIND_GAUSSIAN_SIGMA_UM / pitch
    radius = math.ceil(MIND_GAUSSIAN_RADIUS_UM / pitch)
    offsets = (
        (-axial, 0),
        (axial, 0),
        (0, -axial),
        (0, axial),
        (-diagonal, -diagonal),
        (-diagonal, diagonal),
        (diagonal, -diagonal),
        (diagonal, diagonal),
    )

    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    kernel /= kernel.sum(dtype=np.float64)
    kernel = np.ascontiguousarray(kernel)

    footprint_radius = radius + max(axial, diagonal)
    footprint = np.zeros(
        (2 * footprint_radius + 1, 2 * footprint_radius + 1), dtype=bool
    )
    for dy, dx in ((0, 0), *offsets):
        y0 = footprint_radius + dy - radius
        x0 = footprint_radius + dx - radius
        footprint[y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1] = True

    return {
        "axial_step_px": axial,
        "diagonal_step_px": diagonal,
        "gaussian_sigma_px": sigma,
        "gaussian_radius_px": radius,
        "offsets_dy_dx": offsets,
        "kernel": kernel,
        "footprint": np.ascontiguousarray(footprint),
    }


def target_score_masks(
    fixed_to_source_map: np.ndarray,
    source_map_domain_mask: np.ndarray,
    fixed_map_domain_mask: np.ndarray,
    source_valid_correspondence_mask: np.ndarray,
    fixed_valid_correspondence_mask: np.ndarray,
    pixel_pitch_um: float,
) -> dict[str, np.ndarray]:
    """Build the five frozen target-only masks, including the exact MIND erosion."""
    map_safe = four_corner_safe_mask(
        source_map_domain_mask, fixed_map_domain_mask, fixed_to_source_map
    )
    visible = four_corner_safe_mask(
        source_valid_correspondence_mask,
        fixed_valid_correspondence_mask,
        fixed_to_source_map,
    )
    parameters = mind_parameters(pixel_pitch_um)
    footprint = parameters["footprint"]
    core = binary_erosion(
        visible,
        structure=footprint,
        iterations=1,
        border_value=0,
        origin=0,
        brute_force=False,
    )
    safe_eroded = binary_erosion(
        map_safe,
        structure=footprint,
        iterations=1,
        border_value=0,
        origin=0,
        brute_force=False,
    )
    pitch = float(pixel_pitch_um)
    context = (
        distance_transform_edt(~visible, sampling=(pitch, pitch)) <= CONTEXT_RADIUS_UM
    ) & safe_eroded
    return {
        "map_safe": np.ascontiguousarray(map_safe),
        "visible": np.ascontiguousarray(visible),
        "core": np.ascontiguousarray(core),
        "context": np.ascontiguousarray(context),
        "boundary_ring": np.ascontiguousarray(context & ~visible),
    }


def _shift_scalar(image: np.ndarray, dy: int, dx: int, padding_value: float) -> np.ndarray:
    height, width = image.shape
    shifted = np.full((height, width), padding_value, dtype=np.float64)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    if y0 < y1 and x0 < x1:
        shifted[y0:y1, x0:x1] = image[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return shifted


def _mind_patch_distances(
    image: np.ndarray,
    parameters: dict[str, object],
    padding_value: float,
) -> np.ndarray:
    distances = []
    for dy, dx in parameters["offsets_dy_dx"]:
        shifted = _shift_scalar(image, dy, dx, padding_value)
        squared = (image - shifted) ** 2
        distances.append(
            convolve(
                squared,
                parameters["kernel"],
                mode="constant",
                cval=0.0,
                origin=0,
            )
        )
    return np.ascontiguousarray(np.stack(distances, axis=0), dtype=np.float64)


def _mind_descriptor_from_distances(
    distances: np.ndarray,
    domain_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    variance = np.mean(distances[:4], axis=0, dtype=np.float64)
    vbar = float(np.mean(variance[domain_mask], dtype=np.float64))
    lower = max(1.0e-12, 1.0e-3 * vbar)
    upper = max(lower, 1.0e3 * vbar)
    denominator = np.clip(variance, lower, upper)
    descriptor = np.exp(
        -(distances - np.min(distances, axis=0, keepdims=True)) / denominator[None]
    )
    return np.ascontiguousarray(descriptor, dtype=np.float64), vbar


def mind_descriptor(
    image_float64: np.ndarray,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
    *,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return the domain-normalized eight-channel MIND descriptor and vbar."""
    image = np.asarray(image_float64)
    domain = np.asarray(domain_mask)
    if image.ndim != 2 or image.dtype != np.float64 or domain.shape != image.shape or domain.dtype != np.bool_:
        raise ValueError("MIND requires one float64 image and one matching Boolean domain")
    if not np.isfinite(image).all() or not domain.any() or not np.isfinite(padding_value):
        raise ValueError("MIND image/padding must be finite and its domain nonempty")
    parameters = mind_parameters(pixel_pitch_um)
    distances = _mind_patch_distances(image, parameters, float(padding_value))
    return _mind_descriptor_from_distances(distances, domain)


def mind_pixel_loss(target_descriptor: np.ndarray, candidate_descriptor: np.ndarray) -> np.ndarray:
    """Mean squared descriptor difference across the eight ordered channels."""
    target = np.asarray(target_descriptor, dtype=np.float64)
    candidate = np.asarray(candidate_descriptor, dtype=np.float64)
    if target.shape != candidate.shape or target.ndim != 3 or target.shape[0] != 8:
        raise ValueError("MIND descriptors must have identical shape (8,H,W)")
    return np.ascontiguousarray(
        np.mean((target - candidate) ** 2, axis=0, dtype=np.float64)
    )


def score_mind_candidates(
    target_image_float64: np.ndarray,
    candidate_images_float64: np.ndarray,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
    *,
    padding_value: float = 0.0,
    chunk_size: int = 40,
) -> dict[str, object]:
    """Score candidates with descriptor-only MIND on one fixed target domain."""
    target = np.asarray(target_image_float64)
    candidates = np.asarray(candidate_images_float64)
    domain = np.asarray(domain_mask)
    if target.ndim != 2 or target.dtype != np.float64:
        raise ValueError("target image must be one float64 H-by-W raster")
    if (
        candidates.ndim != 3
        or candidates.dtype != np.float64
        or candidates.shape[1:] != target.shape
        or domain.shape != target.shape
        or domain.dtype != np.bool_
    ):
        raise ValueError("candidate images must be float64 with shape (N,H,W)")
    if not np.isfinite(target).all() or not np.isfinite(candidates).all() or not domain.any():
        raise ValueError("MIND inputs must be finite and the target domain nonempty")
    if candidates.shape[0] < 1 or not np.isfinite(padding_value):
        raise ValueError("MIND requires candidates and a finite scalar padding value")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    parameters = mind_parameters(pixel_pitch_um)
    target_distances = _mind_patch_distances(target, parameters, float(padding_value))
    target_descriptor, target_vbar = _mind_descriptor_from_distances(
        target_distances, domain
    )
    scores = np.empty(candidates.shape[0], dtype=np.float64)
    candidate_vbar = np.empty(candidates.shape[0], dtype=np.float64)
    for start in range(0, candidates.shape[0], chunk_size):
        for index in range(start, min(start + chunk_size, candidates.shape[0])):
            distances = _mind_patch_distances(
                candidates[index], parameters, float(padding_value)
            )
            descriptor, candidate_vbar[index] = _mind_descriptor_from_distances(
                distances, domain
            )
            loss = mind_pixel_loss(target_descriptor, descriptor)
            scores[index] = np.clip(
                1.0 - np.mean(loss[domain], dtype=np.float64), 0.0, 1.0
            )
    return {
        "scores": scores,
        "target_vbar": target_vbar,
        "candidate_vbar": candidate_vbar,
    }


def constant_within_support_null(
    candidate_image_float64: np.ndarray,
    candidate_support_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flatten supported intensity while preserving every exterior value exactly."""
    image = np.asarray(candidate_image_float64)
    support = np.asarray(candidate_support_mask)
    if image.ndim != 2 or image.dtype != np.float64 or support.shape != image.shape or support.dtype != np.bool_:
        raise ValueError("null construction requires a float64 image and matching support")
    if not np.isfinite(image).all() or not support.any():
        raise ValueError("null construction requires finite intensity and nonempty support")
    supported_mean = float(np.mean(image[support], dtype=np.float64))
    flattened = np.array(image, dtype=np.float64, copy=True, order="C")
    flattened[support] = supported_mean
    return flattened, supported_mean


def support_penalized_score(
    pixel_loss: np.ndarray,
    domain_mask: np.ndarray,
    target_visible_mask: np.ndarray,
    candidate_support_mask: np.ndarray,
) -> tuple[float, float]:
    """Apply the frozen native-only candidate-exterior MIND penalty."""
    loss = np.asarray(pixel_loss, dtype=np.float64)
    domain = np.asarray(domain_mask)
    visible = np.asarray(target_visible_mask)
    support = np.asarray(candidate_support_mask)
    if (
        loss.ndim != 2
        or domain.shape != loss.shape
        or visible.shape != loss.shape
        or support.shape != loss.shape
        or domain.dtype != np.bool_
        or visible.dtype != np.bool_
        or support.dtype != np.bool_
    ):
        raise ValueError("loss, domain, visible and support must share one H-by-W shape")
    if not np.isfinite(loss).all() or not domain.any() or not (domain & visible).any():
        raise ValueError("support-penalized scoring requires finite loss and nonempty denominators")
    mismatch = domain & visible & ~support
    penalized = np.array(loss, dtype=np.float64, copy=True, order="C")
    penalized[mismatch] = 1.0
    score = float(
        np.clip(1.0 - np.mean(penalized[domain], dtype=np.float64), 0.0, 1.0)
    )
    exterior_fraction = float(
        np.count_nonzero(mismatch) / np.count_nonzero(domain & visible)
    )
    return score, exterior_fraction


def score_support_penalized_mind_candidates(
    target_image_float64: np.ndarray,
    candidate_images_float64: np.ndarray,
    domain_mask: np.ndarray,
    target_visible_mask: np.ndarray,
    candidate_support_masks: np.ndarray,
    pixel_pitch_um: float,
    *,
    padding_value: float = 0.0,
    chunk_size: int = 40,
) -> dict[str, object]:
    """Score the separate native-only MIND+atlas-support ablation."""
    target = np.asarray(target_image_float64)
    candidates = np.asarray(candidate_images_float64)
    domain = np.asarray(domain_mask)
    visible = np.asarray(target_visible_mask)
    supports = np.asarray(candidate_support_masks)
    if target.ndim != 2 or target.dtype != np.float64:
        raise ValueError("target image must be one float64 H-by-W raster")
    if (
        candidates.ndim != 3
        or candidates.dtype != np.float64
        or candidates.shape[1:] != target.shape
        or supports.shape != candidates.shape
        or domain.shape != target.shape
        or visible.shape != target.shape
        or supports.dtype != np.bool_
        or domain.dtype != np.bool_
        or visible.dtype != np.bool_
    ):
        raise ValueError("support ablation inputs have inconsistent shapes or dtypes")
    if not np.isfinite(target).all() or not np.isfinite(candidates).all():
        raise ValueError("support ablation intensity inputs must be finite")
    if candidates.shape[0] < 1 or not np.isfinite(padding_value):
        raise ValueError("support ablation requires candidates and finite scalar padding")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    parameters = mind_parameters(pixel_pitch_um)
    target_distances = _mind_patch_distances(target, parameters, float(padding_value))
    target_descriptor, target_vbar = _mind_descriptor_from_distances(
        target_distances, domain
    )
    scores = np.empty(candidates.shape[0], dtype=np.float64)
    exterior_fractions = np.empty(candidates.shape[0], dtype=np.float64)
    candidate_vbar = np.empty(candidates.shape[0], dtype=np.float64)
    for start in range(0, candidates.shape[0], chunk_size):
        for index in range(start, min(start + chunk_size, candidates.shape[0])):
            distances = _mind_patch_distances(
                candidates[index], parameters, float(padding_value)
            )
            descriptor, candidate_vbar[index] = _mind_descriptor_from_distances(
                distances, domain
            )
            loss = mind_pixel_loss(target_descriptor, descriptor)
            scores[index], exterior_fractions[index] = support_penalized_score(
                loss, domain, visible, supports[index]
            )
    return {
        "scores": scores,
        "candidate_exterior_fractions": exterior_fractions,
        "target_vbar": target_vbar,
        "candidate_vbar": candidate_vbar,
    }


def common_lattice_map_yx(
    shape: tuple[int, int],
    source_pixel_pitch_um: float,
    target_pixel_pitch_um: float,
) -> np.ndarray:
    """Construct the frozen float64 target-to-source common-lattice coordinate map."""
    height, width = (int(value) for value in shape)
    source_pitch = float(source_pixel_pitch_um)
    target_pitch = float(target_pixel_pitch_um)
    if height < 1 or width < 1 or not np.isfinite(source_pitch) or source_pitch <= 0.0:
        raise ValueError("shape and source pixel pitch must be positive")
    if not np.isfinite(target_pitch) or target_pitch <= 0.0:
        raise ValueError("target pixel pitch must be finite and positive")
    y, x = np.mgrid[:height, :width]
    ratio = target_pitch / source_pitch
    source_x = width / 2.0 + (x.astype(np.float64) - width / 2.0) * ratio
    source_y = height / 2.0 + (y.astype(np.float64) - height / 2.0) * ratio
    return np.ascontiguousarray(np.stack((source_y, source_x), axis=0), dtype=np.float64)


def resample_common_lattice_intensity(
    candidate_image_float64: np.ndarray,
    source_coordinates_yx_float64: np.ndarray,
) -> np.ndarray:
    """Bilinear-zero resample one already-scaled candidate onto a target lattice."""
    image = np.asarray(candidate_image_float64)
    coordinates = np.asarray(source_coordinates_yx_float64)
    if image.ndim != 2 or image.dtype != np.float64:
        raise ValueError("candidate image must be one already-scaled float64 raster")
    if coordinates.shape != (2, *image.shape) or coordinates.dtype != np.float64:
        raise ValueError("common-lattice coordinates must be float64 with shape (2,H,W)")
    if not np.isfinite(image).all() or not np.isfinite(coordinates).all():
        raise ValueError("common-lattice intensity inputs must be finite")
    return np.ascontiguousarray(
        map_coordinates(
            image,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ),
        dtype=np.float64,
    )


def resample_common_lattice_support(
    candidate_support_mask: np.ndarray,
    source_coordinates_yx_float64: np.ndarray,
) -> np.ndarray:
    """Nearest-zero resample support using ties-to-even integer coordinates."""
    support = np.asarray(candidate_support_mask)
    coordinates = np.asarray(source_coordinates_yx_float64)
    if (
        support.ndim != 2
        or support.dtype != np.bool_
        or coordinates.shape != (2, *support.shape)
        or coordinates.dtype != np.float64
    ):
        raise ValueError("support and float64 common-lattice coordinates have inconsistent shapes")
    if not np.isfinite(coordinates).all():
        raise ValueError("common-lattice coordinates must be finite")
    source_y = np.rint(coordinates[0]).astype(np.int64)
    source_x = np.rint(coordinates[1]).astype(np.int64)
    height, width = support.shape
    valid = (
        (source_y >= 0)
        & (source_y < height)
        & (source_x >= 0)
        & (source_x < width)
    )
    result = np.zeros((height, width), dtype=bool)
    result[valid] = support[source_y[valid], source_x[valid]]
    return result


def rank_candidate_scores(
    scores: np.ndarray,
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
) -> dict[str, object]:
    """Apply the frozen conservative ranking and canonical candidate-ID tie rule."""
    values = np.asarray(scores, dtype=np.float64)
    candidate_ids = [str(value) for value in ordered_candidate_ids]
    truth_id = str(truth_candidate_id)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("scores must be one finite vector with at least two candidates")
    if len(candidate_ids) != values.size or len(set(candidate_ids)) != values.size:
        raise ValueError("ordered candidate IDs must be unique and match the scores")
    if candidate_ids.count(truth_id) != 1:
        raise ValueError("truth_candidate_id must occur exactly once")

    truth_index = candidate_ids.index(truth_id)
    truth_score = float(values[truth_index])
    decoys = np.delete(values, truth_index)
    maximum = float(np.max(values))
    tied_indices = np.flatnonzero(values >= maximum - TIE_TOLERANCE).tolist()
    tied_ids = sorted(candidate_ids[index] for index in tied_indices)
    true_rank = 1 + int(
        np.count_nonzero(decoys >= truth_score - TIE_TOLERANCE)
    )
    selected_index = tied_indices[0] if len(tied_indices) == 1 else None
    return {
        "truth_index": truth_index,
        "truth_candidate_id": truth_id,
        "truth_score": truth_score,
        "top1": tied_ids == [truth_id],
        "true_rank": true_rank,
        "reciprocal_rank": 1.0 / true_rank,
        "truth_versus_decoy_win_fraction": float(
            np.count_nonzero(truth_score > decoys + TIE_TOLERANCE) / decoys.size
        ),
        "truth_score_margin": truth_score - float(np.max(decoys)),
        "tied_maximum_indices": tied_indices,
        "tied_maximum_candidate_ids": tied_ids,
        "selected_index": selected_index,
        "selected_candidate_id": (
            None if selected_index is None else candidate_ids[selected_index]
        ),
    }
