"""Frozen secondary image descriptors for the arbitrary-plane pilot."""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter


def central_gradients(image, pixel_pitch_um, padding_value=0.0):
    image = np.asarray(image)
    pitch = float(pixel_pitch_um)
    padding = float(padding_value)
    if image.ndim != 2 or image.dtype != np.float64 or not np.isfinite(image).all():
        raise ValueError("image must be one finite float64 raster")
    if not np.isfinite(pitch) or pitch <= 0.0 or not np.isfinite(padding):
        raise ValueError("pixel pitch must be positive and padding finite")
    padded = np.pad(image, 1, constant_values=padding)
    scale = 2.0 * pitch
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / scale
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / scale
    return gx, gy


def hog_blocks(image, pixel_pitch_um, padding_value=0.0):
    image = np.asarray(image)
    p = float(pixel_pitch_um)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("pixel pitch must be finite and positive")
    q = max(4, int(np.floor(400.0 / p + 0.5)))
    gx, gy = central_gradients(image, p, padding_value)
    magnitude = np.hypot(gx, gy)
    z = 9.0 * np.mod(np.arctan2(gy, gx), np.pi) / np.pi
    floor_z = np.floor(z)
    b0 = floor_z.astype(np.int64) % 9
    b1 = (b0 + 1) % 9
    fraction = z - floor_z
    height, width = image.shape
    yy, xx = np.indices((height, width))
    votes = np.zeros((height, width, 9), dtype=np.float64)
    np.add.at(votes, (yy, xx, b0), magnitude * (1.0 - fraction))
    np.add.at(votes, (yy, xx, b1), magnitude * fraction)
    rows = (height + q - 1) // q
    columns = (width + q - 1) // q
    padded_votes = np.pad(votes, ((0, rows * q - height), (0, columns * q - width), (0, 0)))
    cells = padded_votes.reshape(rows, q, columns, q, 9).sum(axis=(1, 3))
    blocks = np.concatenate(
        (cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]), axis=-1
    )
    blocks /= np.sqrt(np.sum(blocks * blocks, axis=-1, keepdims=True) + 1e-24)
    blocks = np.minimum(blocks, 0.2)
    blocks /= np.sqrt(np.sum(blocks * blocks, axis=-1, keepdims=True) + 1e-24)
    return blocks, q


def hog_complete_block_mask(domain, q):
    domain = np.asarray(domain)
    if domain.ndim != 2 or domain.dtype != np.bool_:
        raise ValueError("HOG domain must be one Boolean raster")
    height, width = domain.shape
    rows = (height + q - 1) // q - 1
    columns = (width + q - 1) // q - 1
    eligible = np.zeros((rows, columns), dtype=bool)
    for row in range(rows):
        y0, y1 = row * q, (row + 2) * q
        for column in range(columns):
            x0, x1 = column * q, (column + 2) * q
            if y0 < 1 or x0 < 1 or y1 >= height or x1 >= width:
                continue
            eligible[row, column] = (
                domain[y0:y1, x0:x1].all()
                and domain[y0 - 1, x0:x1].all()
                and domain[y1, x0:x1].all()
                and domain[y0:y1, x0 - 1].all()
                and domain[y0:y1, x1].all()
            )
    return eligible


def hog_boundary_ring_weights(boundary_ring, q):
    boundary_ring = np.asarray(boundary_ring)
    if boundary_ring.ndim != 2 or boundary_ring.dtype != np.bool_:
        raise ValueError("HOG boundary ring must be one Boolean raster")
    height, width = boundary_ring.shape
    rows = (height + q - 1) // q - 1
    columns = (width + q - 1) // q - 1
    weights = np.zeros((rows, columns), dtype=np.int64)
    for row in range(rows):
        y0, y1 = row * q, (row + 2) * q
        for column in range(columns):
            x0, x1 = column * q, (column + 2) * q
            if y0 >= 1 and x0 >= 1 and y1 < height and x1 < width:
                weights[row, column] = np.count_nonzero(boundary_ring[y0:y1, x0:x1])
    return weights


def score_hog_candidates(
    target,
    candidates,
    domain,
    pixel_pitch_um,
    *,
    boundary_ring=False,
    padding_value=0.0,
    chunk_size=8,
):
    target = np.asarray(target)
    candidates = np.asarray(candidates)
    domain = np.asarray(domain)
    chunk_size = int(chunk_size)
    if (
        target.ndim != 2
        or target.dtype != np.float64
        or candidates.ndim != 3
        or candidates.dtype != np.float64
        or candidates.shape[0] < 1
        or candidates.shape[1:] != target.shape
        or domain.shape != target.shape
        or domain.dtype != np.bool_
        or not np.isfinite(target).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(float(padding_value))
        or chunk_size < 1
    ):
        raise ValueError("HOG inputs must be finite float64 images with one matching Boolean domain")
    target_blocks, q = hog_blocks(target, pixel_pitch_um, padding_value)
    if boundary_ring:
        weights = hog_boundary_ring_weights(domain, q)
        eligible_count = int(np.count_nonzero(weights))
    else:
        weights = hog_complete_block_mask(domain, q).astype(np.int64)
        eligible_count = int(weights.sum())
    if eligible_count == 0:
        return None
    scores = np.empty(len(candidates), dtype=np.float64)
    denominator = float(weights.sum())
    for start in range(0, len(candidates), chunk_size):
        for index in range(start, min(start + chunk_size, len(candidates))):
            candidate_blocks, candidate_q = hog_blocks(
                candidates[index], pixel_pitch_um, padding_value
            )
            if candidate_q != q:
                raise ValueError("target and candidate HOG cell widths differ")
            loss = np.minimum(
                1.0, 0.5 * np.sum((target_blocks - candidate_blocks) ** 2, axis=-1)
            )
            scores[index] = 1.0 - np.sum(weights * loss) / denominator
    return {
        "scores": np.clip(scores, 0.0, 1.0),
        "cell_pixels": q,
        "eligible_block_count": eligible_count,
        "block_weights": weights,
    }


def ngf_evaluation_domain(domain, pixel_pitch_um):
    domain = np.asarray(domain)
    pitch = float(pixel_pitch_um)
    if domain.ndim != 2 or domain.dtype != np.bool_:
        raise ValueError("NGF domain must be one Boolean raster")
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pixel pitch must be finite and positive")
    radius = int(np.ceil(300.0 / pitch)) + 1
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_erosion(
        domain,
        structure=structure,
        iterations=1,
        border_value=0,
        origin=0,
        brute_force=False,
    )


def ngf_gradients(image, pixel_pitch_um, padding_value=0.0):
    image = np.asarray(image)
    p = float(pixel_pitch_um)
    if not np.isfinite(p) or p <= 0.0 or not np.isfinite(float(padding_value)):
        raise ValueError("pixel pitch must be positive and padding finite")
    radius = int(np.ceil(300.0 / p))
    smoothed = gaussian_filter(
        image,
        sigma=100.0 / p,
        radius=radius,
        mode="constant",
        cval=float(padding_value),
        axes=(0, 1),
    )
    return central_gradients(smoothed, p, padding_value)


def score_ngf_candidates(
    target,
    candidates,
    domain,
    pixel_pitch_um,
    *,
    padding_value=0.0,
    chunk_size=8,
):
    target = np.asarray(target)
    candidates = np.asarray(candidates)
    domain = np.asarray(domain)
    chunk_size = int(chunk_size)
    if (
        target.ndim != 2
        or target.dtype != np.float64
        or candidates.ndim != 3
        or candidates.dtype != np.float64
        or candidates.shape[0] < 1
        or candidates.shape[1:] != target.shape
        or domain.shape != target.shape
        or domain.dtype != np.bool_
        or not np.isfinite(target).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(float(padding_value))
        or chunk_size < 1
    ):
        raise ValueError("NGF inputs must be finite float64 images with one matching Boolean domain")
    effective = ngf_evaluation_domain(domain, pixel_pitch_um)
    count = int(effective.sum())
    if count == 0:
        return None
    target_gx, target_gy = ngf_gradients(target, pixel_pitch_um, padding_value)
    target_magnitude = np.hypot(target_gx, target_gy)
    target_eta = max(
        1e-12,
        0.1 * float(np.quantile(target_magnitude[effective], 0.95, method="linear")),
    )
    scores = np.empty(len(candidates), dtype=np.float64)
    candidate_eta = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), chunk_size):
        for index in range(start, min(start + chunk_size, len(candidates))):
            candidate_gx, candidate_gy = ngf_gradients(
                candidates[index], pixel_pitch_um, padding_value
            )
            candidate_magnitude = np.hypot(candidate_gx, candidate_gy)
            candidate_eta[index] = max(
                1e-12,
                0.1
                * float(np.quantile(candidate_magnitude[effective], 0.95, method="linear")),
            )
            numerator = (
                np.abs(target_gx * candidate_gx + target_gy * candidate_gy)
                + target_eta * candidate_eta[index]
            ) ** 2
            denominator = (
                target_gx**2 + target_gy**2 + target_eta**2
            ) * (candidate_gx**2 + candidate_gy**2 + candidate_eta[index] ** 2)
            similarity = numerator / denominator
            scores[index] = np.mean(similarity[effective])
    return {
        "scores": np.clip(scores, 0.0, 1.0),
        "effective_domain_count": count,
        "gaussian_radius_px": int(np.ceil(300.0 / float(pixel_pitch_um))),
        "target_eta": target_eta,
        "candidate_eta": candidate_eta,
    }
