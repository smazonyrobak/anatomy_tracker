"""Deterministic finite coarse catalogue over RP2 plane pose and raster roll."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition


CATALOGUE_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-catalogue/v3"
CATALOGUE_V3_ALGORITHM = "hemisphere-fibonacci-support-projection-roll/v3"
CATALOGUE_V3_SUPPORT_INDEX_ALGORITHM = (
    "hemisphere-fibonacci-authenticated-support-interval-roll/v3"
)


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("catalogue receipts require finite values")
        return value
    return value


def _hash(value):
    encoded = json.dumps(
        _json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_receipt(value):
    array = np.ascontiguousarray(torch.as_tensor(value).detach().cpu().numpy())
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _normals(count):
    index = np.arange(count, dtype=np.float64)
    z = (index + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.ascontiguousarray(
        np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=-1)
    )


def _merged_projection_intervals(projected, half_extent):
    starts = np.sort(projected - half_extent)
    ends = np.sort(projected + half_extent)
    intervals = []
    start, end = float(starts[0]), float(ends[0])
    order = np.argsort(projected - half_extent)
    for item in order[1:]:
        next_start = float(projected[item] - half_extent)
        next_end = float(projected[item] + half_extent)
        if next_start <= end:
            end = max(end, next_end)
        else:
            intervals.append((start, end))
            start, end = next_start, next_end
    intervals.append((start, end))
    return intervals


def _support_offsets(projected, half_extent, count):
    intervals = _merged_projection_intervals(projected, half_extent)
    return _offsets_from_intervals(intervals, count), intervals


def _offsets_from_intervals(intervals, count):
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or np.any(
        intervals[:, 1] <= intervals[:, 0]
    ):
        raise ValueError("support projection intervals must be nonempty and ordered")
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.float64)
    cumulative = np.cumsum(lengths)
    targets = (np.arange(count, dtype=np.float64) + 0.5) * cumulative[-1] / count
    offsets = np.empty(count, dtype=np.float64)
    for index, target in enumerate(targets):
        interval_index = int(np.searchsorted(cumulative, target, side="right"))
        interval_index = min(interval_index, len(intervals) - 1)
        start, _ = intervals[interval_index]
        before = 0.0 if interval_index == 0 else cumulative[interval_index - 1]
        offsets[index] = start + target - before
    return offsets


def _interval_margin(offset, intervals):
    for start, end in intervals:
        if start <= offset <= end:
            return min(offset - start, end - offset)
    raise RuntimeError("catalogue offset escaped its support interval union")


def _base_frame(normal):
    reference = np.eye(3, dtype=np.float64)[np.argmin(np.abs(normal))]
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _observed_covering_radius(normals, probe_count=16384):
    probes = _normals(probe_count)
    best = np.zeros(probe_count, dtype=np.float64)
    for start in range(0, len(normals), 512):
        best = np.maximum(
            best,
            np.max(np.abs(probes @ normals[start : start + 512].T), axis=1),
        )
    return float(np.max(np.arccos(np.clip(best, 0.0, 1.0))))


def catalogue_receipt_v3(artifact):
    return {
        key: artifact[key]
        for key in (
            "schema_version",
            "algorithm",
            "provenance",
            "support_geometry",
            "counts",
            "coverage_audit",
            "representation_contract",
            "array_receipts",
            "catalogue_id",
        )
    }


def make_arbitrary_plane_catalogue_v3(
    atlas_support_mask,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    *,
    support_index=None,
    normal_count,
    offset_count,
    roll_count,
    raster_shape_h_w,
    raster_physical_span_y_x_um,
):
    origin = np.asarray(origin_ap_dv_ml_um, dtype=np.float64)
    spacing = np.asarray(voxel_size_ap_dv_ml_um, dtype=np.float64)
    counts = (normal_count, offset_count, roll_count)
    shape = tuple(raster_shape_h_w)
    span_y_x = np.asarray(raster_physical_span_y_x_um, dtype=np.float64)
    if (
        origin.shape != (3,)
        or spacing.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0.0)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 1 for value in shape)
        or span_y_x.shape != (2,)
        or not np.isfinite(span_y_x).all()
        or np.any(span_y_x <= 0.0)
    ):
        raise ValueError("catalogue inputs must be finite, positive, explicit, and nonempty")
    indexed_support = support_index is not None
    if indexed_support:
        acquisition.verify_annotation_support_index(support_index)
        if atlas_support_mask is not None:
            raise ValueError("authenticated support-index mode does not accept a duplicate mask")
        if not np.array_equal(origin, np.asarray(support_index["origin_um"])) or not np.array_equal(
            spacing, np.asarray(support_index["voxel_size_um"])
        ):
            raise ValueError("catalogue physical grid differs from its support index")
        support_origin = np.asarray(
            acquisition.global_reference_support_geometry(support_index)[
                "support_origin_ap_dv_ml_um"
            ],
            dtype=np.float64,
        )
        support_mask_receipt = {
            "shape": list(support_index["annotation_shape"]),
            "dtype": np.dtype(bool).str,
            "sha256": support_index["support_mask_sha256"],
            "source": "authenticated-support-index",
        }
        support_voxel_count = int(support_index["foreground_voxel_count"])
        algorithm = CATALOGUE_V3_SUPPORT_INDEX_ALGORITHM
    else:
        support = np.asarray(atlas_support_mask)
        if support.ndim != 3 or support.dtype != bool or not support.any():
            raise ValueError("catalogue support mask must be a nonempty Boolean volume")
        support_indices = np.argwhere(support).astype(np.float64)
        support_points = origin + (support_indices + 0.5) * spacing
        support_origin = support_points.mean(axis=0)
        relative = support_points - support_origin
        support_mask_receipt = _array_receipt(support)
        support_voxel_count = int(support.sum())
        algorithm = CATALOGUE_V3_ALGORITHM
    normals = _normals(normal_count)
    if indexed_support:
        normals = np.ascontiguousarray(
            acquisition.support_projection_bounds(normals, support_index)[
                "normal_rp2"
            ],
            dtype=np.float64,
        )
    rolls = 2.0 * math.pi * np.arange(roll_count, dtype=np.float64) / roll_count
    cell_count = normal_count * offset_count * roll_count
    cell_ids = np.arange(cell_count, dtype=np.int64)
    states = np.empty((cell_count, 12), dtype=np.float64)
    cell_normals = np.empty((cell_count, 3), dtype=np.float64)
    cell_offsets = np.empty(cell_count, dtype=np.float64)
    cell_rolls = np.empty(cell_count, dtype=np.float64)
    intersection_margin = np.empty(cell_count, dtype=np.float64)
    offset_tables = np.empty((normal_count, offset_count), dtype=np.float64)
    support_interval_receipts = []
    cursor = 0
    for normal_index, normal in enumerate(normals):
        if indexed_support:
            interval_bundle = acquisition.shifted_component_interval_union(
                normal, support_index
            )
            intervals = interval_bundle["support_origin_interval_union_um"]
            support_interval_receipts.append(
                interval_bundle["support_origin_interval_array_receipt"]
            )
            offsets = _offsets_from_intervals(intervals, offset_count)
        else:
            projected = relative @ normal
            half_extent = 0.5 * float(np.abs(normal) @ spacing)
            offsets, intervals = _support_offsets(projected, half_extent, offset_count)
        offset_tables[normal_index] = offsets
        base_u, base_v = _base_frame(normal)
        for offset in offsets:
            if indexed_support:
                projection_offset = float(offset) - float(
                    interval_bundle["projection_to_support_origin_shift_um"]
                )
                if not acquisition.plane_interval_membership_certificate(
                    normal, projection_offset, support_index
                )["intersects"]:
                    raise RuntimeError(
                        "authenticated support certificate rejected a catalogue cell"
                    )
            margin = (
                _interval_margin(offset, intervals)
                if indexed_support
                else half_extent - np.min(np.abs(projected - offset))
            )
            for roll in rolls:
                cosine, sine = math.cos(float(roll)), math.sin(float(roll))
                u = cosine * base_u + sine * base_v
                v = -sine * base_u + cosine * base_v
                frame = np.stack((u, v, normal), axis=-1)
                centre = support_origin + offset * normal
                basis = np.diag((span_y_x[1], span_y_x[0]))
                states[cursor] = np.concatenate(
                    (centre, u, v, np.log(np.diag(basis)), [0.0])
                )
                cell_normals[cursor] = normal
                cell_offsets[cursor] = offset
                cell_rolls[cursor] = roll
                intersection_margin[cursor] = margin
                cursor += 1
    if np.any(intersection_margin < -1.0e-10):
        raise RuntimeError("a catalogue cell does not intersect atlas support")
    log_mass = np.full(cell_count, -math.log(cell_count), dtype=np.float64)
    representation_log_weight = np.full((cell_count, 2), -math.log(2.0), dtype=np.float64)
    representation_affine = np.broadcast_to(
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        (cell_count, 2, 2, 3),
    ).copy()
    arrays = {
        "cell_id_int64": cell_ids,
        "cell_states_float64": states,
        "cell_log_mass_float64": log_mass,
        "cell_normal_ap_dv_ml_float64": cell_normals,
        "cell_signed_offset_um_float64": cell_offsets,
        "cell_roll_rad_float64": cell_rolls,
        "cell_support_intersection_margin_um_float64": intersection_margin,
        "normal_offset_table_um_float64": offset_tables,
        "representation_log_weight_float64": representation_log_weight,
        "representation_to_canonical_raster_affine_float64": representation_affine,
    }
    array_receipts = {name: _array_receipt(value) for name, value in arrays.items()}
    artifact = {
        "schema_version": CATALOGUE_V3_SCHEMA,
        "algorithm": algorithm,
        "provenance": {
            "dependency_contract": "pure NumPy/Torch geometry; no learned model, prior weights, checkpoint, or filesystem input",
            "normal_sequence": (
                "deterministic equal-area Fibonacci RP2 samples; authenticated-index "
                "mode uses the support certificate's largest-absolute-component-positive convention"
            ),
            "offset_sequence": "equal support-projection-union arc-length cells; every centre plane intersects a support voxel",
            "roll_sequence": "uniform periodic [0,2pi)",
        },
        "support_geometry": {
            "support_mask_receipt": support_mask_receipt,
            "support_index_sha256": (
                support_index["support_index_sha256"] if indexed_support else None
            ),
            "normal_support_interval_receipts": support_interval_receipts,
            "origin_ap_dv_ml_um": origin.tolist(),
            "voxel_size_ap_dv_ml_um": spacing.tolist(),
            "support_origin_ap_dv_ml_um": support_origin.tolist(),
            "support_voxel_count": support_voxel_count,
            "raster_shape_h_w": list(shape),
            "raster_physical_span_y_x_um": span_y_x.tolist(),
        },
        "counts": {
            "normal_count": normal_count,
            "offset_count_per_normal": offset_count,
            "roll_count": roll_count,
            "cell_count": cell_count,
            "representation_count": 2,
        },
        "coverage_audit": {
            "observed_probe_count": 16384,
            "max_observed_rp2_angular_covering_radius_rad": _observed_covering_radius(normals),
            "minimum_support_intersection_margin_um": float(intersection_margin.min()),
            "all_cells_support_membership_certified": bool(indexed_support),
        },
        "representation_contract": {
            "names": ["identity", "horizontal"],
            "conditional_log_mass": [-math.log(2.0), -math.log(2.0)],
            "affine_grid_align_corners": False,
            "horizontal_is_raster_only_not_atlas_ml_reflection": True,
        },
        "arrays": arrays,
        "array_receipts": array_receipts,
    }
    artifact["catalogue_id"] = _hash(
        {
            "domain": "anatomy-tracker.arbitrary-plane-catalogue/v3",
            "provenance": artifact["provenance"],
            "support_geometry": artifact["support_geometry"],
            "counts": artifact["counts"],
            "coverage_audit": artifact["coverage_audit"],
            "representation_contract": artifact["representation_contract"],
            "array_receipts": array_receipts,
        }
    )
    artifact["tensors"] = {
        "cell_id": torch.from_numpy(cell_ids),
        "cell_states": torch.from_numpy(states)[None],
        "cell_log_mass": torch.from_numpy(log_mass)[None],
        "representation_log_weight": torch.from_numpy(representation_log_weight)[None],
        "representation_to_canonical_raster_affine": torch.from_numpy(
            representation_affine
        )[None],
    }
    artifact["receipt_sha256"] = _hash(catalogue_receipt_v3(artifact))
    return artifact


def replay_arbitrary_plane_catalogue_v3(artifact, *args, **kwargs):
    return make_arbitrary_plane_catalogue_v3(*args, **kwargs)


def verify_arbitrary_plane_catalogue_v3(artifact, *args, **kwargs):
    replay = make_arbitrary_plane_catalogue_v3(*args, **kwargs)
    if (
        set(artifact) != set(replay)
        or artifact.get("array_receipts")
        != {name: _array_receipt(value) for name, value in artifact.get("arrays", {}).items()}
        or artifact.get("receipt_sha256") != _hash(catalogue_receipt_v3(artifact))
        or catalogue_receipt_v3(artifact) != catalogue_receipt_v3(replay)
        or set(artifact.get("arrays", {})) != set(replay["arrays"])
        or any(
            not np.array_equal(artifact["arrays"][name], replay["arrays"][name])
            for name in replay["arrays"]
        )
        or set(artifact.get("tensors", {})) != set(replay["tensors"])
        or any(
            not torch.equal(artifact["tensors"][name], replay["tensors"][name])
            for name in replay["tensors"]
        )
    ):
        raise ValueError("catalogue v3 does not replay exactly")
