"""Deterministic finite arbitrary-plane CCF rendering precursor.

This development-only generator samples geometry from random initialization and
uses no learned asset.  It binds atlas provenance and rendered arrays, but does
not issue a synthetic realization identifier: deformation, appearance, and
smart-brush background modes are deliberately outside this precursor.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from training.arbitrary_plane_geometry import (
    ARBITRARY_PLANE_GEOMETRY_VERSION,
    QUICKNII_RASTER_INDEX_SAMPLING,
    allen_to_quicknii_points,
    allen_to_quicknii_vectors,
    physical_um_to_allen_index_points,
    physical_um_to_allen_index_vectors,
    quicknii_ouv_to_frame,
    render_arbitrary_plane,
    normalized_raster_to_ccf,
)
from training.arbitrary_plane_manifest import MANIFEST_SCHEMA, SAMPLER_ALGORITHM, canonicalize_plane
from training.arbitrary_plane_support import (
    PROJECTION_ALGORITHM,
    SUPPORT_INDEX_ALGORITHM,
    SUPPORT_INDEX_SCHEMA,
    verify_annotation_support_index,
)


FINITE_RENDER_SCHEMA = "anatomy-tracker.finite-arbitrary-plane-render/v1"
FINITE_RENDER_ALGORITHM = "uniform-rp2-component-union-finite-render/v1"
REFERENCE_STRATUM = "reference"
BOUNDARY_STRESS_STRATUM = "boundary-stress"
UINT64_SEED_ENCODING = "canonical-lowercase-uint64-hex/v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    array = np.asarray(array)
    dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": dtype.str, "shape": list(array.shape)}).encode("utf-8"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask.reshape(-1, order="C"), bitorder="little")
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": "|b1", "shape": list(mask.shape), "bitorder": "little"}).encode("utf-8")
    )
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _source_sha256() -> str:
    return _LOADED_GENERATOR_SOURCE_SHA256


def _dependency_source_sha256() -> dict[str, str]:
    source_root = Path(__file__).parent
    return {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in (
            "arbitrary_plane_geometry.py",
            "arbitrary_plane_manifest.py",
            "arbitrary_plane_support.py",
        )
    }


_LOADED_GENERATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOADED_DEPENDENCY_SOURCE_SHA256 = _dependency_source_sha256()
_PREPARED_CONTEXT_TOKEN = object()


def _freeze_context_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        copy = np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype)
        return copy.reshape(value.shape)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_context_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_context_value(item) for item in value)
    return value


def _plain_json_value(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_json_value(item) for item in value]
    return value


def _uint64_seed(seed: int) -> int:
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("Seed must fit an unsigned 64-bit integer")
    return seed


def _seed_hex(seed: int) -> str:
    return f"0x{_uint64_seed(seed):016x}"


def _parse_seed_hex(seed: object) -> int:
    if not isinstance(seed, str) or re.fullmatch(r"0x[0-9a-f]{16}", seed) is None:
        raise ValueError("Seed must use canonical lowercase uint64 hexadecimal encoding")
    return int(seed, 16)


def _derived_seed(seed: int, split: str, sample_index: int, field: str, attempt: int) -> int:
    payload = (
        f"{FINITE_RENDER_ALGORITHM}\0{_uint64_seed(seed)}\0{split}\0{sample_index}"
        f"\0{field}\0{attempt}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _source_commit(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Generator source commit must be a 40-digit Git SHA or None")
    return value


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _validate_sha256(value: str, name: str) -> str:
    value = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _merge_intervals(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
        raise ValueError("Intervals must be a finite C-by-2 array")
    intervals = intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("Interval upper bounds must not precede lower bounds")
    merged = [intervals[0].tolist()]
    for lower, upper in intervals[1:]:
        if lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(upper))
        else:
            merged.append([float(lower), float(upper)])
    return np.asarray(merged, dtype=np.float64)


def oriented_support_projection_bounds(
    direction_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> np.ndarray:
    """Return per-component bounds in the requested oriented direction."""
    verify_annotation_support_index(support_index)
    return _oriented_support_projection_bounds_trusted(direction_ap_dv_ml, support_index)


def _oriented_support_projection_bounds_trusted(
    direction_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> np.ndarray:
    direction = np.asarray(direction_ap_dv_ml, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0.0:
        raise ValueError("Projection direction must be a finite nonzero 3-vector")
    direction /= np.linalg.norm(direction)
    spacing = np.asarray(support_index["voxel_size_um"], dtype=np.float64)
    origin = np.asarray(support_index["origin_um"], dtype=np.float64)
    projection_origin = np.asarray(support_index["projection_origin_um"], dtype=np.float64)
    half_extent = float(np.abs(direction) @ (spacing / 2.0))
    bounds = []
    for vertices in support_index["component_hull_indices"]:
        physical = origin + (np.asarray(vertices, dtype=np.float64) + 0.5) * spacing - projection_origin
        projected = physical @ direction
        bounds.append([float(projected.min() - half_extent), float(projected.max() + half_extent)])
    return np.asarray(bounds, dtype=np.float64)


def component_interval_union(
    normal_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> np.ndarray:
    """Return the exact merged union of component support intervals."""
    verify_annotation_support_index(support_index)
    return _component_interval_union_trusted(normal_ap_dv_ml, support_index)


def _component_interval_union_trusted(
    normal_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> np.ndarray:
    return _merge_intervals(_oriented_support_projection_bounds_trusted(normal_ap_dv_ml, support_index))


def _stratum_intervals(
    component_union_um: np.ndarray,
    stratum: str,
    boundary_fraction: float,
) -> np.ndarray:
    if stratum == REFERENCE_STRATUM:
        return np.asarray(component_union_um, dtype=np.float64)
    if stratum != BOUNDARY_STRESS_STRATUM:
        raise ValueError(f"Stratum must be {REFERENCE_STRATUM!r} or {BOUNDARY_STRESS_STRATUM!r}")
    if not 0.0 < boundary_fraction <= 0.5:
        raise ValueError("Boundary-stress fraction must lie in (0, 0.5]")
    bands = []
    for lower, upper in np.asarray(component_union_um, dtype=np.float64):
        width = upper - lower
        bands.extend(((lower, lower + boundary_fraction * width), (upper - boundary_fraction * width, upper)))
    return _merge_intervals(np.asarray(bands))


def sample_interval_union_offset(intervals_um: np.ndarray, seed: int) -> tuple[float, int, float]:
    """Sample uniformly with respect to length over a disjoint interval union."""
    intervals = _merge_intervals(intervals_um)
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Interval union must have positive finite length")
    draw = float(np.random.Generator(np.random.PCG64(_uint64_seed(seed))).uniform(0.0, total))
    index = min(int(np.searchsorted(np.cumsum(lengths), draw, side="right")), len(intervals) - 1)
    before = float(lengths[:index].sum())
    offset = float(intervals[index, 0] + draw - before)
    return offset, index, draw / total


def physical_plane_frame(normal_ap_dv_ml: np.ndarray, roll_rad: float) -> np.ndarray:
    """Construct right-handed physical AP-DV-ML columns ``[u,v,n]``."""
    normal, _, _ = canonicalize_plane(normal_ap_dv_ml, 0.0)
    if not np.isfinite(roll_rad):
        raise ValueError("Roll must be finite")
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    u0 = np.cross(reference, normal)
    u0 /= np.linalg.norm(u0)
    v0 = np.cross(normal, u0)
    cosine, sine = np.cos(float(roll_rad)), np.sin(float(roll_rad))
    u = cosine * u0 + sine * v0
    v = -sine * u0 + cosine * v0
    return np.stack((u, v, normal), axis=-1)


def finite_plane_raster_geometry(
    normal_ap_dv_ml: np.ndarray,
    signed_offset_um: float,
    roll_rad: float,
    support_index: dict[str, object],
    output_shape: tuple[int, int],
    margin_um: float | tuple[float, float] = 0.0,
) -> dict[str, object]:
    """Build a finite physical plane whose sampled endpoints cover atlas support."""
    verify_annotation_support_index(support_index)
    return _finite_plane_raster_geometry_trusted(
        normal_ap_dv_ml, signed_offset_um, roll_rad, support_index, output_shape, margin_um
    )


def _finite_plane_raster_geometry_trusted(
    normal_ap_dv_ml: np.ndarray,
    signed_offset_um: float,
    roll_rad: float,
    support_index: dict[str, object],
    output_shape: tuple[int, int],
    margin_um: float | tuple[float, float] = 0.0,
) -> dict[str, object]:
    normal, offset, _ = canonicalize_plane(normal_ap_dv_ml, signed_offset_um)
    height, width = (int(value) for value in output_shape)
    if height <= 1 or width <= 1:
        raise ValueError("Finite QuickNII rasters require H,W > 1")
    margins = np.broadcast_to(np.asarray(margin_um, dtype=np.float64), (2,)).copy()
    if not np.isfinite(margins).all() or np.any(margins < 0.0):
        raise ValueError("Raster margins must contain one or two finite nonnegative values")
    normal_component_bounds = _oriented_support_projection_bounds_trusted(normal, support_index)
    membership = (offset >= normal_component_bounds[:, 0]) & (offset <= normal_component_bounds[:, 1])
    if not bool(membership.any()):
        raise ValueError("Plane does not intersect the annotation support interval union")
    frame = physical_plane_frame(normal, roll_rad)
    u, v = frame[:, 0], frame[:, 1]
    u_components = _oriented_support_projection_bounds_trusted(u, support_index)
    v_components = _oriented_support_projection_bounds_trusted(v, support_index)
    required_u = np.asarray([u_components[:, 0].min() - margins[0], u_components[:, 1].max() + margins[0]])
    required_v = np.asarray([v_components[:, 0].min() - margins[1], v_components[:, 1].max() + margins[1]])
    reference_pitch = max(
        (required_u[1] - required_u[0]) / (width - 1),
        (required_v[1] - required_v[0]) / (height - 1),
    )
    reference_pitch = float(np.nextafter(reference_pitch, np.inf))
    sampled_u_span = reference_pitch * (width - 1)
    sampled_v_span = reference_pitch * (height - 1)
    sampled_u = required_u.mean() + np.asarray([-0.5, 0.5]) * sampled_u_span
    sampled_v = required_v.mean() + np.asarray([-0.5, 0.5]) * sampled_v_span
    projection_origin = np.asarray(support_index["projection_origin_um"], dtype=np.float64)
    plane_center = projection_origin + offset * normal
    origin = plane_center + sampled_u[0] * u + sampled_v[0] * v
    edge_u = (sampled_u[1] - sampled_u[0]) * width / (width - 1) * u
    edge_v = (sampled_v[1] - sampled_v[0]) * height / (height - 1) * v
    spacing = tuple(support_index["voxel_size_um"])
    atlas_origin = tuple(support_index["origin_um"])
    origin_index = physical_um_to_allen_index_points(torch.as_tensor(origin), atlas_origin, spacing).numpy()
    edge_u_index = physical_um_to_allen_index_vectors(torch.as_tensor(edge_u), spacing).numpy()
    edge_v_index = physical_um_to_allen_index_vectors(torch.as_tensor(edge_v), spacing).numpy()
    atlas_shape = tuple(int(value) for value in support_index["annotation_shape"])
    quicknii_ouv = np.concatenate(
        (
            allen_to_quicknii_points(torch.as_tensor(origin_index), atlas_shape).numpy(),
            allen_to_quicknii_vectors(torch.as_tensor(edge_u_index)).numpy(),
            allen_to_quicknii_vectors(torch.as_tensor(edge_v_index)).numpy(),
        )
    )
    center_index, frame_index, basis_index = quicknii_ouv_to_frame(
        torch.as_tensor(quicknii_ouv, dtype=torch.float64), atlas_shape
    )
    x = np.arange(width, dtype=np.float64) / width
    y = np.arange(height, dtype=np.float64) / height
    design_coordinate_raster = (
        origin_index[None, None]
        + x[None, :, None] * edge_u_index[None, None]
        + y[:, None, None] * edge_v_index[None, None]
    )
    effective_center = center_index.to(torch.float32)
    effective_frame = frame_index.to(torch.float32)
    effective_basis = basis_index.to(torch.float32)
    s = torch.arange(width, dtype=torch.float32) / width
    t = torch.arange(height, dtype=torch.float32) / height
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    st = torch.stack((ss, tt), -1)[None]
    effective_coordinate_raster = normalized_raster_to_ccf(
        effective_center[None, None, None],
        effective_frame[None, None, None],
        effective_basis[None, None, None],
        st,
    )[0]
    depth, native_height, native_width = atlas_shape
    effective_normalized_grid = torch.stack(
        (
            effective_coordinate_raster[..., 2] / (native_width - 1) * 2 - 1,
            effective_coordinate_raster[..., 1] / (native_height - 1) * 2 - 1,
            effective_coordinate_raster[..., 0] / (depth - 1) * 2 - 1,
        ),
        -1,
    )
    effective_edges = effective_frame[:, :2] @ effective_basis
    effective_origin = effective_center - 0.5 * effective_edges.sum(dim=1)
    effective_allen_ouv = torch.cat(
        (effective_origin, effective_edges[:, 0], effective_edges[:, 1])
    )
    effective_quicknii_ouv = torch.cat(
        (
            allen_to_quicknii_points(effective_allen_ouv[:3], atlas_shape),
            allen_to_quicknii_vectors(effective_allen_ouv[3:6]),
            allen_to_quicknii_vectors(effective_allen_ouv[6:9]),
        )
    )
    effective_allen_ouv_numpy = effective_allen_ouv.numpy().astype(np.float64)
    effective_physical_ouv = np.concatenate(
        (
            atlas_origin + (effective_allen_ouv_numpy[:3] + 0.5) * spacing,
            effective_allen_ouv_numpy[3:6] * spacing,
            effective_allen_ouv_numpy[6:9] * spacing,
        )
    )
    rounded = torch.round(effective_coordinate_raster).to(torch.int64).numpy()
    valid_atlas = np.ones((height, width), dtype=bool)
    for axis, size in enumerate(atlas_shape):
        valid_atlas &= (rounded[..., axis] >= 0) & (rounded[..., axis] < size)
    reference_axis_index = int(np.argmin(np.abs(normal)))
    arrays = {
        "frame_ap_dv_ml_physical": frame,
        "physical_ouv_ap_dv_ml_um": np.concatenate((origin, edge_u, edge_v)),
        "allen_index_ouv_ap_dv_ml": np.concatenate((origin_index, edge_u_index, edge_v_index)),
        "quicknii_ouv_ml_ap_dv": quicknii_ouv,
        "design_renderer_center_ap_dv_ml_float64": center_index.numpy(),
        "design_renderer_frame_ap_dv_ml_float64": frame_index.numpy(),
        "design_renderer_inplane_basis_float64": basis_index.numpy(),
        "design_coordinate_raster_allen_index_float64": design_coordinate_raster,
        "effective_renderer_center_ap_dv_ml_float32": effective_center.numpy(),
        "effective_renderer_frame_ap_dv_ml_float32": effective_frame.numpy(),
        "effective_renderer_inplane_basis_float32": effective_basis.numpy(),
        "effective_allen_index_ouv_ap_dv_ml_float32": effective_allen_ouv.numpy(),
        "effective_physical_ouv_ap_dv_ml_um_from_float32_state": effective_physical_ouv,
        "effective_quicknii_ouv_ml_ap_dv_float32": effective_quicknii_ouv.numpy(),
        "effective_coordinate_raster_allen_index_float32": effective_coordinate_raster.numpy(),
        "effective_normalized_interpolation_grid_xyz_float32": effective_normalized_grid.numpy(),
    }
    array_receipts = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape), "array_sha256": _array_sha256(value)}
        for name, value in arrays.items()
    }
    array_receipts["valid_atlas_label_sampling_mask"] = {
        "dtype": valid_atlas.dtype.str,
        "shape": list(valid_atlas.shape),
        "array_sha256": _mask_sha256(valid_atlas),
    }
    membership_payload = {
        "schema": "anatomy-tracker.trusted-support-membership/v1",
        "support_index_sha256": support_index["support_index_sha256"],
        "projection_algorithm": PROJECTION_ALGORITHM,
        "normal_rp2": normal.tolist(),
        "signed_offset_um": float(offset),
        "component_bounds_um": normal_component_bounds.tolist(),
        "component_membership": membership.tolist(),
    }
    geometry = {
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "signed_offset_um": float(offset),
        "roll_rad": float(roll_rad),
        "frame_ap_dv_ml_physical": frame.tolist(),
        "component_projection_bounds_u_um": u_components.tolist(),
        "component_projection_bounds_v_um": v_components.tolist(),
        "required_endpoint_bounds_u_um": required_u.tolist(),
        "required_endpoint_bounds_v_um": required_v.tolist(),
        "sampled_endpoint_bounds_u_um": sampled_u.tolist(),
        "sampled_endpoint_bounds_v_um": sampled_v.tolist(),
        "margin_u_v_um": margins.tolist(),
        "physical_ouv_ap_dv_ml_um": np.concatenate((origin, edge_u, edge_v)).tolist(),
        "allen_index_ouv_ap_dv_ml": np.concatenate((origin_index, edge_u_index, edge_v_index)).tolist(),
        "quicknii_ouv_ml_ap_dv": quicknii_ouv.tolist(),
        "design_renderer_center_ap_dv_ml_float64": center_index.tolist(),
        "design_renderer_frame_ap_dv_ml_float64": frame_index.tolist(),
        "design_renderer_inplane_basis_float64": basis_index.tolist(),
        "renderer_center_ap_dv_ml": effective_center.tolist(),
        "renderer_frame_ap_dv_ml": effective_frame.tolist(),
        "renderer_inplane_basis": effective_basis.tolist(),
        "effective_renderer_dtype": "<f4",
        "effective_physical_ouv_ap_dv_ml_um": effective_physical_ouv.tolist(),
        "effective_allen_index_ouv_ap_dv_ml": effective_allen_ouv.tolist(),
        "effective_quicknii_ouv_ml_ap_dv": effective_quicknii_ouv.tolist(),
        "array_receipts": array_receipts,
        "coordinate_raster_storage": (
            "hash-only; float64 design and normative effective float32 Allen/grid coordinates are both bound"
        ),
        "valid_atlas_label_sampling_pixel_count": int(valid_atlas.sum()),
        "output_shape_h_w": [height, width],
        "sampling_contract": QUICKNII_RASTER_INDEX_SAMPLING,
        "raster_endpoint_semantics": {
            "pixel_mapping": "P(x,y)=O+(x/W)U+(y/H)V for x=0..W-1,y=0..H-1",
            "u_edge_factor": float(width / (width - 1)),
            "v_edge_factor": float(height / (height - 1)),
            "first_sample": "O",
            "last_sample": "O+((W-1)/W)U+((H-1)/H)V",
        },
        "reference_aspect_policy": {
            "policy": "isotropic physical pixel pitch with symmetric padding of the shorter axis",
            "pixel_pitch_u_um": float(reference_pitch),
            "pixel_pitch_v_um": float(reference_pitch),
            "anisotropic_resize_or_shear": "excluded from reference geometry; reserved for named augmentation",
        },
        "reflection_state": {
            "horizontal": False,
            "vertical": False,
            "status": "no raster reflection sampled in finite-geometry precursor v1",
        },
        "tangent_construction": {
            "reference_axis_index_ap_dv_ml": reference_axis_index,
            "reference_axis_rule": "least absolute alignment with canonical normal; u0=cross(axis,n), v0=cross(n,u0)",
            "roll_rule": "u=cos(r)u0+sin(r)v0; v=-sin(r)u0+cos(r)v0",
        },
        "sampling_interpolation": {
            "scalar": "torch.grid_sample trilinear for 5-D input, zero padding, align_corners=True",
            "annotation": "nearest integer index via torch.round, zero outside atlas",
            "brain_mask": "rendered annotation != 0",
            "aliasing_status": "finite raster may miss continuous tissue; eligibility uses minimum rendered brain pixels",
        },
        "support_membership_certificate_sha256": _payload_sha256(membership_payload),
    }
    return {**geometry, "geometry_sha256": _payload_sha256(geometry)}


def effective_renderer_sampling_arrays(
    geometry: dict[str, object],
    atlas_shape_ap_dv_ml: tuple[int, int, int],
    *,
    origin_ap_dv_ml_um: tuple[float, float, float] | None = None,
    voxel_size_ap_dv_ml_um: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct the exact float32 Allen coordinates and grid consumed by rendering."""
    height, width = geometry["output_shape_h_w"]
    center = torch.as_tensor(geometry["renderer_center_ap_dv_ml"], dtype=torch.float32)
    frame = torch.as_tensor(geometry["renderer_frame_ap_dv_ml"], dtype=torch.float32)
    basis = torch.as_tensor(geometry["renderer_inplane_basis"], dtype=torch.float32)
    s = torch.arange(width, dtype=torch.float32) / width
    t = torch.arange(height, dtype=torch.float32) / height
    tt, ss = torch.meshgrid(t, s, indexing="ij")
    points = normalized_raster_to_ccf(
        center[None, None, None], frame[None, None, None], basis[None, None, None],
        torch.stack((ss, tt), -1)[None],
    )[0]
    depth, native_height, native_width = atlas_shape_ap_dv_ml
    grid = torch.stack(
        (
            points[..., 2] / (native_width - 1) * 2 - 1,
            points[..., 1] / (native_height - 1) * 2 - 1,
            points[..., 0] / (depth - 1) * 2 - 1,
        ),
        -1,
    )
    rounded = torch.round(points).to(torch.int64)
    valid = (
        (rounded[..., 0] >= 0) & (rounded[..., 0] < depth)
        & (rounded[..., 1] >= 0) & (rounded[..., 1] < native_height)
        & (rounded[..., 2] >= 0) & (rounded[..., 2] < native_width)
    )
    edges = frame[:, :2] @ basis
    origin = center - 0.5 * edges.sum(dim=1)
    allen_ouv = torch.cat((origin, edges[:, 0], edges[:, 1]))
    quicknii_ouv = torch.cat(
        (
            allen_to_quicknii_points(allen_ouv[:3], atlas_shape_ap_dv_ml),
            allen_to_quicknii_vectors(allen_ouv[3:6]),
            allen_to_quicknii_vectors(allen_ouv[6:9]),
        )
    )
    arrays = {
        "coordinate_raster_allen_index_float32": points.numpy(),
        "normalized_interpolation_grid_xyz_float32": grid.numpy(),
        "valid_atlas_label_sampling_mask": valid.numpy(),
        "allen_index_ouv_ap_dv_ml_float32": allen_ouv.numpy(),
        "quicknii_ouv_ml_ap_dv_float32": quicknii_ouv.numpy(),
    }
    if (origin_ap_dv_ml_um is None) != (voxel_size_ap_dv_ml_um is None):
        raise ValueError("Physical effective O/U/V reconstruction requires both origin and voxel size")
    if origin_ap_dv_ml_um is not None:
        atlas_origin = np.asarray(origin_ap_dv_ml_um, dtype=np.float64)
        spacing = np.asarray(voxel_size_ap_dv_ml_um, dtype=np.float64)
        allen = allen_ouv.numpy().astype(np.float64)
        arrays["physical_ouv_ap_dv_ml_um_from_float32_state"] = np.concatenate(
            (
                atlas_origin + (allen[:3] + 0.5) * spacing,
                allen[3:6] * spacing,
                allen[6:9] * spacing,
            )
        )
    return arrays


def render_finite_arbitrary_plane(
    scalar_volume_ap_dv_ml: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    geometry: dict[str, object],
) -> dict[str, np.ndarray | str | int]:
    """Render scalar intensity, nearest annotation, and its exact nonzero mask."""
    scalar = np.asarray(scalar_volume_ap_dv_ml)
    annotation = np.asarray(annotation_ap_dv_ml)
    if scalar.ndim != 3 or not np.issubdtype(scalar.dtype, np.floating):
        raise ValueError("Scalar CCF volume must be one floating-point 3-D array")
    if not np.isfinite(scalar).all():
        raise ValueError("Scalar CCF volume must be finite")
    if annotation.shape != scalar.shape or not np.issubdtype(annotation.dtype, np.integer):
        raise ValueError("Annotation and scalar CCF arrays must have identical shapes")
    if annotation.size and (annotation.min() < 0 or annotation.max() > np.iinfo(np.int64).max):
        raise ValueError("Annotation labels must convert losslessly to signed int64")
    return _render_finite_arbitrary_plane_trusted(scalar, annotation, geometry)


def _render_finite_arbitrary_plane_trusted(
    scalar: np.ndarray | torch.Tensor,
    annotation: np.ndarray | torch.Tensor,
    geometry: dict[str, object],
) -> dict[str, np.ndarray | str | int]:
    scalar_tensor = torch.as_tensor(scalar)
    annotation_tensor = torch.as_tensor(annotation)
    dtype = scalar_tensor.dtype
    image, labels = render_arbitrary_plane(
        scalar_tensor,
        torch.as_tensor(geometry["renderer_center_ap_dv_ml"], dtype=dtype),
        torch.as_tensor(geometry["renderer_frame_ap_dv_ml"], dtype=dtype),
        torch.as_tensor(geometry["renderer_inplane_basis"], dtype=dtype),
        tuple(geometry["output_shape_h_w"]),
        annotation_tensor,
        sampling_contract=QUICKNII_RASTER_INDEX_SAMPLING,
    )
    scalar_raster = image[0, 0].cpu().numpy()
    annotation_raster = labels[0, 0].to(torch.int64).cpu().numpy()
    brain_mask = annotation_raster != 0
    hashes = {
        "scalar_sha256": _array_sha256(scalar_raster),
        "annotation_sha256": _array_sha256(annotation_raster),
        "brain_mask_sha256": _mask_sha256(brain_mask),
    }
    hashes["combined_sha256"] = _payload_sha256(hashes)
    receipts = {
        "scalar": {
            "dtype": scalar_raster.dtype.str,
            "shape": list(scalar_raster.shape),
            "array_sha256": hashes["scalar_sha256"],
        },
        "annotation": {
            "dtype": annotation_raster.dtype.str,
            "shape": list(annotation_raster.shape),
            "array_sha256": hashes["annotation_sha256"],
        },
        "brain_mask": {
            "dtype": brain_mask.dtype.str,
            "shape": list(brain_mask.shape),
            "array_sha256": hashes["brain_mask_sha256"],
        },
    }
    return {
        "scalar": scalar_raster,
        "annotation": annotation_raster,
        "brain_mask": brain_mask,
        "brain_pixel_count": int(brain_mask.sum()),
        "array_receipts": receipts,
        **hashes,
    }


def _provenance(
    support_index: dict[str, object],
    asset_receipt: dict[str, object],
    animal_id: object,
    specimen_id: object,
    experiment_id: object,
) -> dict[str, object]:
    return {
        "atlas": _plain_json_value(support_index["atlas"]),
        "annotation_source": _plain_json_value(support_index["source"]),
        "annotation_decoded": _plain_json_value(asset_receipt["annotation_decoded"]),
        "annotation_sampling": _plain_json_value(asset_receipt["annotation_sampling"]),
        "scalar_source": {
            **_plain_json_value(asset_receipt["scalar_source"]),
            "decoded": _plain_json_value(asset_receipt["template_decoded"]),
            "float_conversion": _plain_json_value(asset_receipt["scalar_conversion"]),
        },
        "animal_id": _python_scalar(animal_id),
        "specimen_id": _python_scalar(specimen_id),
        "experiment_id": _python_scalar(experiment_id),
    }


def _finite_render_identity(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "generator_algorithm": artifact["generator_algorithm"],
        "support_index_sha256": artifact["support_index_sha256"],
        "implementation_sha256": artifact["generator"]["implementation"]["implementation_sha256"],
        "resolved_config_sha256": artifact["generator"]["resolved_config_sha256"],
        "provenance_sha256": artifact["provenance_sha256"],
        "plane_realization_id": artifact["plane_realization_id"],
        "finite_plane_geometry_sha256": artifact["finite_plane_geometry_sha256"],
        "rendered_artifacts_sha256": artifact["rendered_artifacts_sha256"],
        "model_independence_sha256": artifact["generator"]["model_independence_sha256"],
        "rejection_attempts_sha256": artifact["rejection_attempts_sha256"],
        "accepted_attempt_index": artifact["accepted_attempt_index"],
    }


def _plane_sampler_lineage() -> dict[str, object]:
    return {
        "schema": "anatomy-tracker.finite-plane-sampler-lineage/v1",
        "algorithm": FINITE_RENDER_ALGORITHM,
        "seed_derivation": "sha256(algorithm,root_seed,split,sample_index,field,attempt)->little-endian uint64",
        "normal_measure": "normalized isotropic Gaussian folded canonically to Haar-uniform RP2",
        "roll_measure": "independent uniform [0,2pi)",
        "offset_measure": "length-uniform over merged component support intervals or named boundary bands",
        "manifest_contract": {"schema": MANIFEST_SCHEMA, "sampler": SAMPLER_ALGORITHM},
        "support_contract": {
            "schema": SUPPORT_INDEX_SCHEMA,
            "support_algorithm": SUPPORT_INDEX_ALGORITHM,
            "projection_algorithm": PROJECTION_ALGORITHM,
        },
        "loaded_dependency_source_sha256": {
            key: _LOADED_DEPENDENCY_SOURCE_SHA256[key]
            for key in ("arbitrary_plane_manifest.py", "arbitrary_plane_support.py")
        },
    }


def _sampling_measure() -> dict[str, str]:
    return {
        "orientation": "orientation-balanced Haar-uniform RP2 normal; not Crofton plane measure",
        "roll": "independent uniform [0,2pi)",
        "reference_offset": "length-uniform over merged per-component support-interval union",
        "boundary_stress_offset": (
            "length-uniform over merged lower/upper boundary bands of each merged support interval"
        ),
        "conditioning": (
            "unconditioned by the finite raster: exactly one normal, roll, and support-chord "
            "offset draw is retained; raster support only determines explicit supervision "
            "identifiability metadata and never redraws pose"
        ),
    }


def _receipt_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"raster", "finite_render_receipt_sha256"}
    }


def prepare_finite_render_context(
    scalar_volume_ap_dv_ml: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    scalar_source_uri: str,
    scalar_source_sha256: str,
    scalar_source_entity_type: str = "atlas-template",
    scalar_dtype: str = "float32",
    template_decoder: str = "caller-supplied decoded array",
    template_index_order: str | None = None,
    annotation_decoder: str = "caller-supplied decoded array",
    annotation_index_order: str | None = None,
) -> dict[str, object]:
    """Verify and hash immutable atlas assets once for many deterministic samples."""
    verify_annotation_support_index(support_index)
    template_decoded = np.asarray(scalar_volume_ap_dv_ml)
    scalar_dtype = np.dtype(scalar_dtype)
    if scalar_dtype != np.dtype(np.float32):
        raise ValueError("Canonical finite-render conversion requires scalar_dtype='float32'")
    scalar = np.array(template_decoded, dtype=scalar_dtype, copy=True, order="C")
    annotation = np.array(annotation_ap_dv_ml, copy=True, order="C")
    if (
        template_decoded.ndim != 3
        or not (
            np.issubdtype(template_decoded.dtype, np.integer)
            or np.issubdtype(template_decoded.dtype, np.floating)
        )
        or not np.isfinite(template_decoded).all()
        or not np.isfinite(scalar).all()
    ):
        raise ValueError("Decoded template must be one finite numeric 3-D array")
    if (
        annotation.shape != scalar.shape
        or list(annotation.shape) != support_index["annotation_shape"]
        or not np.issubdtype(annotation.dtype, np.integer)
    ):
        raise ValueError("Scalar, annotation, and support-index shapes must agree")
    if annotation.size and (annotation.min() < 0 or annotation.max() > np.iinfo(np.int64).max):
        raise ValueError("Annotation labels must convert losslessly to signed int64")
    template_decoded_sha256 = _array_sha256(template_decoded)
    scalar_sha256 = _array_sha256(scalar)
    annotation_sha256 = _array_sha256(annotation)
    if annotation_sha256 != support_index["source"]["annotation_array_sha256"]:
        raise ValueError("Annotation array does not match the verified support index")
    asset_receipt = {
        "schema": "anatomy-tracker.prepared-finite-render-context/v1",
        "support_index_sha256": support_index["support_index_sha256"],
        "template_decoded": {
            "decoder": str(template_decoder),
            "index_order": None if template_index_order is None else str(template_index_order),
            "dtype": template_decoded.dtype.str,
            "shape": list(template_decoded.shape),
            "array_sha256": template_decoded_sha256,
        },
        "scalar_conversion": {
            "operation": "numpy.array(dtype=<f4, copy=True, order=C)",
            "normalization": "none",
            "dtype": scalar.dtype.str,
            "shape": list(scalar.shape),
            "array_sha256": scalar_sha256,
        },
        "annotation_decoded": {
            "decoder": str(annotation_decoder),
            "index_order": None if annotation_index_order is None else str(annotation_index_order),
            "dtype": annotation.dtype.str,
            "shape": list(annotation.shape),
            "array_sha256": annotation_sha256,
        },
        "annotation_sampling": {
            "operation": "native-dtype nearest indexing, then sampled H-by-W labels converted to torch.int64",
            "losslessness": "required nonnegative labels <= int64 maximum",
            "full_volume_copy": "none",
            "rendered_output_dtype": "<i8",
        },
        "scalar_source": {
            "source_entity_type": str(scalar_source_entity_type),
            "uri": str(scalar_source_uri),
            "source_sha256": _validate_sha256(scalar_source_sha256, "Scalar source SHA-256"),
            "source_sha256_semantics": "raw source bytes",
        },
    }
    scalar_tensor = torch.from_numpy(scalar).clone()
    annotation_tensor = torch.from_numpy(annotation).clone()
    frozen_receipt = _freeze_context_value(asset_receipt)
    frozen_support = _freeze_context_value(support_index)
    return MappingProxyType({
        "schema": asset_receipt["schema"],
        "_token": _PREPARED_CONTEXT_TOKEN,
        "asset_receipt": frozen_receipt,
        "prepared_context_sha256": _payload_sha256(asset_receipt),
        "support_index": frozen_support,
        "scalar_tensor": scalar_tensor,
        "annotation_tensor": annotation_tensor,
        "scalar_tensor_id": id(scalar_tensor),
        "annotation_tensor_id": id(annotation_tensor),
        "scalar_tensor_version": scalar_tensor._version,
        "annotation_tensor_version": annotation_tensor._version,
    })


def _validate_prepared_context(context: dict[str, object]) -> None:
    if (
        not isinstance(context, MappingProxyType)
        or context.get("_token") is not _PREPARED_CONTEXT_TOKEN
        or context.get("schema") != "anatomy-tracker.prepared-finite-render-context/v1"
        or context.get("prepared_context_sha256")
        != _payload_sha256(_plain_json_value(context["asset_receipt"]))
        or context["support_index"]["support_index_sha256"]
        != context["asset_receipt"]["support_index_sha256"]
    ):
        raise ValueError("Prepared finite-render context receipt does not match")
    scalar = context["scalar_tensor"]
    annotation = context["annotation_tensor"]
    if (
        id(scalar) != context["scalar_tensor_id"]
        or id(annotation) != context["annotation_tensor_id"]
        or scalar._version != context["scalar_tensor_version"]
        or annotation._version != context["annotation_tensor_version"]
        or scalar.dtype != torch.float32
        or list(scalar.shape) != list(context["asset_receipt"]["scalar_conversion"]["shape"])
        or str(annotation.numpy().dtype) != np.dtype(
            context["asset_receipt"]["annotation_decoded"]["dtype"]
        ).name
        or list(annotation.shape) != list(context["asset_receipt"]["annotation_decoded"]["shape"])
    ):
        raise ValueError("Prepared finite-render tensor identity or version changed")


def _make_finite_arbitrary_plane_render_prepared(
    context: dict[str, object],
    split: str,
    seed: int,
    output_shape: tuple[int, int],
    *,
    sample_index: int = 0,
    stratum: str = REFERENCE_STRATUM,
    boundary_stress_fraction: float = 0.10,
    margin_um: float | tuple[float, float] = 0.0,
    animal_id: str | int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    max_rejection_attempts: int = 64,
    minimum_brain_pixels: int = 1,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Sample and render one finite arbitrary-plane development precursor."""
    _validate_prepared_context(context)
    scalar = context["scalar_tensor"]
    annotation = context["annotation_tensor"]
    support_index = context["support_index"]
    asset_receipt = context["asset_receipt"]
    annotation_sha256 = asset_receipt["annotation_decoded"]["array_sha256"]
    split = str(split)
    if split not in {"train", "development"}:
        raise ValueError("Finite development renders permit only train or development splits")
    seed = _uint64_seed(seed)
    sample_index = int(sample_index)
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    height, width = (int(value) for value in output_shape)
    if height <= 1 or width <= 1:
        raise ValueError("Finite QuickNII rasters require H,W > 1")
    margins = np.broadcast_to(np.asarray(margin_um, dtype=np.float64), (2,)).copy()
    if not np.isfinite(margins).all() or np.any(margins < 0.0):
        raise ValueError("Raster margins must contain one or two finite nonnegative values")
    boundary_stress_fraction = float(boundary_stress_fraction)
    if stratum not in {REFERENCE_STRATUM, BOUNDARY_STRESS_STRATUM}:
        raise ValueError("Unknown finite-render stratum")
    if not 0.0 < boundary_stress_fraction <= 0.5:
        raise ValueError("Boundary-stress fraction must lie in (0, 0.5]")
    max_rejection_attempts = int(max_rejection_attempts)
    if max_rejection_attempts <= 0:
        raise ValueError("max_rejection_attempts must be positive")
    minimum_brain_pixels = int(minimum_brain_pixels)
    if minimum_brain_pixels <= 0:
        raise ValueError("minimum_brain_pixels must be positive")
    source_commit = _source_commit(generator_source_commit)
    provenance = _provenance(support_index, asset_receipt, animal_id, specimen_id, experiment_id)
    sampling_measure = _sampling_measure()
    resolved_config = {
        "schema_version": FINITE_RENDER_SCHEMA,
        "generator_algorithm": FINITE_RENDER_ALGORITHM,
        "split": split,
        "root_seed": _seed_hex(seed),
        "sample_index": sample_index,
        "seed_encoding": UINT64_SEED_ENCODING,
        "support_index_sha256": support_index["support_index_sha256"],
        "prepared_context_sha256": context["prepared_context_sha256"],
        "template_decoded_dtype": asset_receipt["template_decoded"]["dtype"],
        "template_decoded_shape": list(asset_receipt["template_decoded"]["shape"]),
        "template_decoded_array_sha256": asset_receipt["template_decoded"]["array_sha256"],
        "scalar_conversion": _plain_json_value(asset_receipt["scalar_conversion"]),
        "annotation_decoded_dtype": asset_receipt["annotation_decoded"]["dtype"],
        "annotation_decoded_shape": list(asset_receipt["annotation_decoded"]["shape"]),
        "annotation_array_sha256": annotation_sha256,
        "annotation_sampling": _plain_json_value(asset_receipt["annotation_sampling"]),
        "output_shape_h_w": [height, width],
        "stratum": stratum,
        "boundary_stress_fraction": boundary_stress_fraction,
        "margin_u_v_um": margins.tolist(),
        "scalar_source_uri": asset_receipt["scalar_source"]["uri"],
        "scalar_source_sha256": asset_receipt["scalar_source"]["source_sha256"],
        "scalar_source_entity_type": asset_receipt["scalar_source"]["source_entity_type"],
        "template_decoder": asset_receipt["template_decoded"]["decoder"],
        "template_index_order": asset_receipt["template_decoded"]["index_order"],
        "annotation_decoder": asset_receipt["annotation_decoded"]["decoder"],
        "annotation_index_order": asset_receipt["annotation_decoded"]["index_order"],
        "animal_id": _python_scalar(animal_id),
        "specimen_id": _python_scalar(specimen_id),
        "experiment_id": _python_scalar(experiment_id),
        "max_rejection_attempts": max_rejection_attempts,
        "minimum_brain_pixels": minimum_brain_pixels,
        "sampling_measure": sampling_measure,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    implementation = {
        "source_path": "training/arbitrary_plane_rendered_generator.py",
        "loaded_source_sha256": _LOADED_GENERATOR_SOURCE_SHA256,
        "source_commit": source_commit,
        "loaded_dependency_source_sha256": _LOADED_DEPENDENCY_SOURCE_SHA256,
        "dependency_contract_versions": {
            "geometry": ARBITRARY_PLANE_GEOMETRY_VERSION,
            "manifest_schema": MANIFEST_SCHEMA,
            "manifest_sampler": SAMPLER_ALGORITHM,
            "support_schema": SUPPORT_INDEX_SCHEMA,
            "support_algorithm": SUPPORT_INDEX_ALGORITHM,
            "projection_algorithm": PROJECTION_ALGORITHM,
        },
    }
    implementation["implementation_sha256"] = _payload_sha256(implementation)
    model_independence = {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "initialization": "random seed streams only; no learned initialization",
    }
    plane_sampler_lineage = _plane_sampler_lineage()
    generator = {
        "implementation": implementation,
        "resolved_config": resolved_config,
        "resolved_config_sha256": _payload_sha256(resolved_config),
        **model_independence,
        "model_independence_sha256": _payload_sha256(model_independence),
        "plane_sampler_lineage": plane_sampler_lineage,
        "plane_sampler_lineage_sha256": _payload_sha256(plane_sampler_lineage),
    }
    attempts = []
    accepted_geometry = None
    accepted_raster = None
    accepted_index = None
    for attempt in range(1):
        field_seeds = {
            field: _derived_seed(seed, split, sample_index, field, attempt)
            for field in ("normal", "roll", "offset")
        }
        normal_rng = np.random.Generator(np.random.PCG64(field_seeds["normal"]))
        normal, _, _ = canonicalize_plane(normal_rng.normal(size=3), 0.0)
        roll = float(np.random.Generator(np.random.PCG64(field_seeds["roll"])).uniform(0.0, 2.0 * np.pi))
        component_union = _component_interval_union_trusted(normal, support_index)
        sampling_intervals = _stratum_intervals(component_union, stratum, boundary_stress_fraction)
        offset, selected_interval, measure_fraction = sample_interval_union_offset(
            sampling_intervals, field_seeds["offset"]
        )
        geometry = _finite_plane_raster_geometry_trusted(
            normal, offset, roll, support_index, (height, width), tuple(margins)
        )
        raster = _render_finite_arbitrary_plane_trusted(scalar, annotation, geometry)
        attempt_record = {
            "attempt_index": attempt,
            "field_stream_seed_uint64": {key: _seed_hex(value) for key, value in field_seeds.items()},
            "normal_rp2_ap_dv_ml": normal.tolist(),
            "roll_rad": roll,
            "component_interval_union_um": component_union.tolist(),
            "stratum_sampling_interval_union_um": sampling_intervals.tolist(),
            "selected_interval_index": selected_interval,
            "offset_measure_fraction": measure_fraction,
            "signed_offset_um": offset,
            "geometry_sha256": geometry["geometry_sha256"],
            "raster_combined_sha256": raster["combined_sha256"],
            "brain_pixel_count": raster["brain_pixel_count"],
            "raster_support_meets_requested_identifiability_threshold": bool(
                raster["brain_pixel_count"] >= minimum_brain_pixels
            ),
            "accepted": True,
        }
        attempts.append(attempt_record)
        accepted_geometry, accepted_raster, accepted_index = geometry, raster, attempt
    raster_hashes = {
        key: accepted_raster[key]
        for key in ("scalar_sha256", "annotation_sha256", "brain_mask_sha256", "combined_sha256")
    }
    effective_sampling_receipts = {
        key: accepted_geometry["array_receipts"][key]
        for key in (
            "effective_coordinate_raster_allen_index_float32",
            "effective_normalized_interpolation_grid_xyz_float32",
            "valid_atlas_label_sampling_mask",
        )
    }
    rendered_artifacts_receipt = {
        "schema": "anatomy-tracker.rendered-finite-plane-artifacts/v1",
        "effective_sampling_array_receipts": effective_sampling_receipts,
        "raster_array_receipts": accepted_raster["array_receipts"],
    }
    rendered_artifacts_sha256 = _payload_sha256(rendered_artifacts_receipt)
    accepted_attempt = attempts[accepted_index]
    plane_identity = {
        "schema": "anatomy-tracker.finite-plane-realization/v1",
        "sampler_algorithm": FINITE_RENDER_ALGORITHM,
        "support_index_sha256": support_index["support_index_sha256"],
        "plane_sampler_lineage_sha256": generator["plane_sampler_lineage_sha256"],
        "split": split,
        "root_seed": _seed_hex(seed),
        "sample_index": sample_index,
        "accepted_attempt_index": accepted_index,
        "stratum": stratum,
        "field_stream_seed_uint64": accepted_attempt["field_stream_seed_uint64"],
        "normal_rp2_ap_dv_ml": accepted_attempt["normal_rp2_ap_dv_ml"],
        "signed_offset_um": accepted_attempt["signed_offset_um"],
        "roll_rad": accepted_attempt["roll_rad"],
        "animal_id": _python_scalar(animal_id),
        "specimen_id": _python_scalar(specimen_id),
        "experiment_id": _python_scalar(experiment_id),
    }
    plane_realization_id = _payload_sha256(plane_identity)
    finite_plane_geometry_sha256 = _payload_sha256(
        {
            "schema": "anatomy-tracker.finite-plane-geometry/v1",
            "plane_realization_id": plane_realization_id,
            "geometry_sha256": accepted_geometry["geometry_sha256"],
        }
    )
    artifact = {
        "schema_version": FINITE_RENDER_SCHEMA,
        "generator_algorithm": FINITE_RENDER_ALGORITHM,
        "split": split,
        "root_seed": _seed_hex(seed),
        "sample_index": sample_index,
        "stratum": stratum,
        "support_index_sha256": support_index["support_index_sha256"],
        "generator": generator,
        "provenance": provenance,
        "provenance_sha256": _payload_sha256(provenance),
        "accepted_attempt_index": accepted_index,
        "rejection_attempts": attempts,
        "rejection_attempts_sha256": _payload_sha256(attempts),
        "plane_realization_id": plane_realization_id,
        "finite_plane_geometry_sha256": finite_plane_geometry_sha256,
        "geometry": accepted_geometry,
        "raster": {
            "scalar": accepted_raster["scalar"],
            "annotation": accepted_raster["annotation"],
            "brain_mask": accepted_raster["brain_mask"],
        },
        "raster_hashes": raster_hashes,
        "raster_array_receipts": accepted_raster["array_receipts"],
        "rendered_artifacts_receipt": rendered_artifacts_receipt,
        "rendered_artifacts_sha256": rendered_artifacts_sha256,
        "acceptance_contract": {
            "predicate": "one authenticated continuous brain-intersecting plane draw; no raster-support rejection",
            "minimum_brain_pixels": minimum_brain_pixels,
            "brain_pixel_count": accepted_raster["brain_pixel_count"],
            "continuous_plane_intersection_authenticated": True,
            "pose_redrawn_for_raster_support": False,
            "pose_draw_count": 1,
            "raster_support_meets_requested_identifiability_threshold": bool(
                accepted_raster["brain_pixel_count"] >= minimum_brain_pixels
            ),
            "minimum_brain_pixels_role": (
                "requested downstream point/dense-supervision identifiability threshold; "
                "not an acceptance predicate"
            ),
        },
        "sampling_measure": sampling_measure,
        "development_scope": {
            "status": "finite rendered geometry precursor only",
            "identifier": "finite_plane_render_id",
            "excluded": [
                "deformation field",
                "histology appearance model",
                "smart-brush and raw-background modes",
                "calibrated pose posterior",
                "final-test data",
            ],
        },
    }
    artifact["finite_plane_render_id"] = _payload_sha256(_finite_render_identity(artifact))
    artifact["finite_render_receipt_sha256"] = _payload_sha256(_receipt_payload(artifact))
    return artifact


def make_finite_arbitrary_plane_render_from_context(
    context: dict[str, object],
    split: str,
    seed: int,
    output_shape: tuple[int, int],
    *,
    sample_index: int = 0,
    stratum: str = REFERENCE_STRATUM,
    boundary_stress_fraction: float = 0.10,
    margin_um: float | tuple[float, float] = 0.0,
    animal_id: str | int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    max_rejection_attempts: int = 64,
    minimum_brain_pixels: int = 1,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    return _make_finite_arbitrary_plane_render_prepared(
        context,
        split,
        seed,
        output_shape,
        sample_index=sample_index,
        stratum=stratum,
        boundary_stress_fraction=boundary_stress_fraction,
        margin_um=margin_um,
        animal_id=animal_id,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        max_rejection_attempts=max_rejection_attempts,
        minimum_brain_pixels=minimum_brain_pixels,
        generator_source_commit=generator_source_commit,
    )


def make_finite_arbitrary_plane_render(
    scalar_volume_ap_dv_ml: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    split: str,
    seed: int,
    output_shape: tuple[int, int],
    *,
    sample_index: int = 0,
    stratum: str = REFERENCE_STRATUM,
    boundary_stress_fraction: float = 0.10,
    margin_um: float | tuple[float, float] = 0.0,
    scalar_source_uri: str,
    scalar_source_sha256: str,
    scalar_source_entity_type: str = "atlas-template",
    scalar_dtype: str = "float32",
    template_decoder: str = "caller-supplied decoded array",
    template_index_order: str | None = None,
    annotation_decoder: str = "caller-supplied decoded array",
    annotation_index_order: str | None = None,
    animal_id: str | int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    max_rejection_attempts: int = 64,
    minimum_brain_pixels: int = 1,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    context = prepare_finite_render_context(
        scalar_volume_ap_dv_ml,
        annotation_ap_dv_ml,
        support_index,
        scalar_source_uri=scalar_source_uri,
        scalar_source_sha256=scalar_source_sha256,
        scalar_source_entity_type=scalar_source_entity_type,
        scalar_dtype=scalar_dtype,
        template_decoder=template_decoder,
        template_index_order=template_index_order,
        annotation_decoder=annotation_decoder,
        annotation_index_order=annotation_index_order,
    )
    return make_finite_arbitrary_plane_render_from_context(
        context,
        split,
        seed,
        output_shape,
        sample_index=sample_index,
        stratum=stratum,
        boundary_stress_fraction=boundary_stress_fraction,
        margin_um=margin_um,
        animal_id=animal_id,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        max_rejection_attempts=max_rejection_attempts,
        minimum_brain_pixels=minimum_brain_pixels,
        generator_source_commit=generator_source_commit,
    )


def finite_render_receipt(artifact: dict[str, object]) -> dict[str, object]:
    """Return the JSON-safe, hash-bound receipt without the NumPy raster payload."""
    return {key: value for key, value in artifact.items() if key != "raster"}


def verify_finite_arbitrary_plane_render(
    artifact: dict[str, object],
    support_index: dict[str, object],
    *,
    generator_source_commit: str | None = None,
    _support_preverified: bool = False,
) -> None:
    if not _support_preverified:
        verify_annotation_support_index(support_index)
    if artifact.get("schema_version") != FINITE_RENDER_SCHEMA:
        raise ValueError("Unsupported finite-render artifact schema")
    if artifact.get("generator_algorithm") != FINITE_RENDER_ALGORITHM:
        raise ValueError("Unsupported finite-render generator algorithm")
    if artifact.get("support_index_sha256") != support_index["support_index_sha256"]:
        raise ValueError("Finite render does not match the supplied support index")
    generator = artifact["generator"]
    config = generator["resolved_config"]
    if artifact.get("finite_render_receipt_sha256") != _payload_sha256(_receipt_payload(artifact)):
        raise ValueError("Finite-render JSON receipt hash does not match")
    if generator["resolved_config_sha256"] != _payload_sha256(config):
        raise ValueError("Finite-render resolved config hash does not match")
    implementation = generator["implementation"]
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "implementation_sha256"
    }
    if implementation["implementation_sha256"] != _payload_sha256(implementation_payload):
        raise ValueError("Finite-render implementation hash does not match")
    if implementation["loaded_source_sha256"] != _LOADED_GENERATOR_SOURCE_SHA256:
        raise ValueError("Finite-render generator source hash does not match")
    if implementation["loaded_dependency_source_sha256"] != _LOADED_DEPENDENCY_SOURCE_SHA256:
        raise ValueError("Finite-render dependency source hashes do not match")
    expected_dependency_contract_versions = {
        "geometry": ARBITRARY_PLANE_GEOMETRY_VERSION,
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_sampler": SAMPLER_ALGORITHM,
        "support_schema": SUPPORT_INDEX_SCHEMA,
        "support_algorithm": SUPPORT_INDEX_ALGORITHM,
        "projection_algorithm": PROJECTION_ALGORITHM,
    }
    if (
        implementation["source_path"] != "training/arbitrary_plane_rendered_generator.py"
        or implementation["dependency_contract_versions"] != expected_dependency_contract_versions
    ):
        raise ValueError("Finite-render implementation metadata does not match loaded contracts")
    if implementation["source_commit"] != _source_commit(generator_source_commit):
        raise ValueError("Finite-render generator source commit does not match")
    dependency_keys = {key for key in generator if key.endswith("_dependencies")}
    required_dependency_keys = {
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
    }
    if dependency_keys != required_dependency_keys or any(generator[key] != [] for key in required_dependency_keys):
        raise ValueError("Finite renderer must have exactly three empty model-dependency lists")
    model_independence = {
        key: generator[key]
        for key in (
            "learned_checkpoint_dependencies",
            "previous_model_dependencies",
            "pretrained_feature_dependencies",
        )
    }
    model_independence["initialization"] = generator["initialization"]
    if generator["initialization"] != "random seed streams only; no learned initialization":
        raise ValueError("Finite renderer must use random-only initialization")
    if generator["model_independence_sha256"] != _payload_sha256(model_independence):
        raise ValueError("Finite-render model-independence hash does not match")
    expected_sampler_lineage = _plane_sampler_lineage()
    if (
        generator["plane_sampler_lineage"] != expected_sampler_lineage
        or generator["plane_sampler_lineage_sha256"] != _payload_sha256(expected_sampler_lineage)
    ):
        raise ValueError("Finite-render plane-sampler lineage does not match")
    if artifact["split"] not in {"train", "development"} or artifact["split"] != config["split"]:
        raise ValueError("Finite-render split is invalid")
    if _parse_seed_hex(artifact["root_seed"]) != _parse_seed_hex(config["root_seed"]):
        raise ValueError("Finite-render root seed does not match config")
    if artifact["provenance_sha256"] != _payload_sha256(artifact["provenance"]):
        raise ValueError("Finite-render provenance hash does not match")
    provenance = artifact["provenance"]
    support_atlas = _plain_json_value(support_index["atlas"])
    support_source = _plain_json_value(support_index["source"])
    if provenance["atlas"] != support_atlas or provenance["annotation_source"] != support_source:
        raise ValueError("Finite-render annotation atlas/source provenance does not match support index")
    annotation_decoded = provenance["annotation_decoded"]
    if (
        annotation_decoded["shape"] != list(support_index["annotation_shape"])
        or annotation_decoded["array_sha256"] != support_index["source"]["annotation_array_sha256"]
    ):
        raise ValueError("Finite-render decoded annotation provenance does not match support index")
    scalar_source = provenance["scalar_source"]
    reconstructed_asset_receipt = {
        "schema": "anatomy-tracker.prepared-finite-render-context/v1",
        "support_index_sha256": support_index["support_index_sha256"],
        "template_decoded": scalar_source["decoded"],
        "scalar_conversion": scalar_source["float_conversion"],
        "annotation_decoded": annotation_decoded,
        "annotation_sampling": provenance["annotation_sampling"],
        "scalar_source": {
            key: scalar_source[key]
            for key in ("source_entity_type", "uri", "source_sha256", "source_sha256_semantics")
        },
    }
    if config["prepared_context_sha256"] != _payload_sha256(reconstructed_asset_receipt):
        raise ValueError("Finite-render prepared asset provenance does not match prepared_context_sha256")
    config_crosslinks = {
        "schema_version": artifact["schema_version"],
        "generator_algorithm": artifact["generator_algorithm"],
        "split": artifact["split"],
        "root_seed": artifact["root_seed"],
        "sample_index": artifact["sample_index"],
        "support_index_sha256": artifact["support_index_sha256"],
        "stratum": artifact["stratum"],
        "template_decoded_dtype": scalar_source["decoded"]["dtype"],
        "template_decoded_shape": scalar_source["decoded"]["shape"],
        "template_decoded_array_sha256": scalar_source["decoded"]["array_sha256"],
        "scalar_conversion": scalar_source["float_conversion"],
        "annotation_decoded_dtype": annotation_decoded["dtype"],
        "annotation_decoded_shape": annotation_decoded["shape"],
        "annotation_array_sha256": annotation_decoded["array_sha256"],
        "annotation_sampling": provenance["annotation_sampling"],
        "scalar_source_uri": scalar_source["uri"],
        "scalar_source_sha256": scalar_source["source_sha256"],
        "scalar_source_entity_type": scalar_source["source_entity_type"],
        "template_decoder": scalar_source["decoded"]["decoder"],
        "template_index_order": scalar_source["decoded"]["index_order"],
        "annotation_decoder": annotation_decoded["decoder"],
        "annotation_index_order": annotation_decoded["index_order"],
        "animal_id": provenance["animal_id"],
        "specimen_id": provenance["specimen_id"],
        "experiment_id": provenance["experiment_id"],
        "output_shape_h_w": artifact["geometry"]["output_shape_h_w"],
        "margin_u_v_um": artifact["geometry"]["margin_u_v_um"],
        "minimum_brain_pixels": artifact["acceptance_contract"]["minimum_brain_pixels"],
        "sampling_measure": artifact["sampling_measure"],
    }
    if any(config[key] != value for key, value in config_crosslinks.items()):
        raise ValueError("Finite-render config, provenance, and top-level fields are not cross-linked")
    if artifact["sampling_measure"] != _sampling_measure():
        raise ValueError("Finite-render sampling-measure claim does not match the implemented algorithm")
    if artifact["acceptance_contract"]["predicate"] != (
        "one authenticated continuous brain-intersecting plane draw; no raster-support rejection"
    ):
        raise ValueError("Finite-render acceptance predicate does not match the implemented algorithm")
    geometry = artifact["geometry"]
    if geometry["geometry_sha256"] != _payload_sha256(
        {key: value for key, value in geometry.items() if key != "geometry_sha256"}
    ):
        raise ValueError("Finite-render geometry hash does not match")
    reconstructed_geometry = _finite_plane_raster_geometry_trusted(
        np.asarray(geometry["normal_rp2_ap_dv_ml"], dtype=np.float64),
        float(geometry["signed_offset_um"]),
        float(geometry["roll_rad"]),
        support_index,
        tuple(geometry["output_shape_h_w"]),
        tuple(geometry["margin_u_v_um"]),
    )
    if geometry != reconstructed_geometry:
        raise ValueError("Finite-render geometry does not replay from pose and support")
    effective_arrays = effective_renderer_sampling_arrays(
        geometry,
        tuple(int(value) for value in support_index["annotation_shape"]),
        origin_ap_dv_ml_um=tuple(support_index["origin_um"]),
        voxel_size_ap_dv_ml_um=tuple(support_index["voxel_size_um"]),
    )
    effective_receipt_names = {
        "coordinate_raster_allen_index_float32": "effective_coordinate_raster_allen_index_float32",
        "normalized_interpolation_grid_xyz_float32": "effective_normalized_interpolation_grid_xyz_float32",
        "valid_atlas_label_sampling_mask": "valid_atlas_label_sampling_mask",
    }
    for array_name, receipt_name in effective_receipt_names.items():
        array = effective_arrays[array_name]
        expected_sha256 = _mask_sha256(array) if array.dtype == np.bool_ else _array_sha256(array)
        expected = {"dtype": array.dtype.str, "shape": list(array.shape), "array_sha256": expected_sha256}
        if geometry["array_receipts"][receipt_name] != expected:
            raise ValueError("Finite-render effective sampling-grid receipt does not match")
    effective_ouv_contracts = {
        "effective_allen_index_ouv_ap_dv_ml": (
            "allen_index_ouv_ap_dv_ml_float32",
            "effective_allen_index_ouv_ap_dv_ml_float32",
        ),
        "effective_physical_ouv_ap_dv_ml_um": (
            "physical_ouv_ap_dv_ml_um_from_float32_state",
            "effective_physical_ouv_ap_dv_ml_um_from_float32_state",
        ),
        "effective_quicknii_ouv_ml_ap_dv": (
            "quicknii_ouv_ml_ap_dv_float32",
            "effective_quicknii_ouv_ml_ap_dv_float32",
        ),
    }
    for value_name, (array_name, receipt_name) in effective_ouv_contracts.items():
        array = effective_arrays[array_name]
        expected = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "array_sha256": _array_sha256(array),
        }
        if (
            not np.array_equal(np.asarray(geometry[value_name]), array)
            or geometry["array_receipts"][receipt_name] != expected
        ):
            raise ValueError("Finite-render effective O/U/V serialization does not match renderer state")
    raster = artifact["raster"]
    expected_hashes = {
        "scalar_sha256": _array_sha256(raster["scalar"]),
        "annotation_sha256": _array_sha256(raster["annotation"]),
        "brain_mask_sha256": _mask_sha256(raster["brain_mask"]),
    }
    expected_hashes["combined_sha256"] = _payload_sha256(expected_hashes)
    if artifact["raster_hashes"] != expected_hashes:
        raise ValueError("Finite-render raster hashes do not match")
    expected_receipts = {
        "scalar": {
            "dtype": np.asarray(raster["scalar"]).dtype.str,
            "shape": list(np.asarray(raster["scalar"]).shape),
            "array_sha256": expected_hashes["scalar_sha256"],
        },
        "annotation": {
            "dtype": np.asarray(raster["annotation"]).dtype.str,
            "shape": list(np.asarray(raster["annotation"]).shape),
            "array_sha256": expected_hashes["annotation_sha256"],
        },
        "brain_mask": {
            "dtype": np.asarray(raster["brain_mask"]).dtype.str,
            "shape": list(np.asarray(raster["brain_mask"]).shape),
            "array_sha256": expected_hashes["brain_mask_sha256"],
        },
    }
    if artifact["raster_array_receipts"] != expected_receipts:
        raise ValueError("Finite-render raster dtype/shape receipts do not match")
    if not np.array_equal(np.asarray(raster["brain_mask"]), np.asarray(raster["annotation"]) != 0):
        raise ValueError("Finite-render brain mask does not match the annotation raster")
    attempts = artifact["rejection_attempts"]
    if artifact["rejection_attempts_sha256"] != _payload_sha256(attempts):
        raise ValueError("Finite-render rejection-attempt hash does not match")
    if (
        not attempts
        or int(artifact["sample_index"]) < 0
        or artifact["stratum"] not in {REFERENCE_STRATUM, BOUNDARY_STRESS_STRATUM}
        or int(config["max_rejection_attempts"]) <= 0
        or len(attempts) != 1
        or int(config["minimum_brain_pixels"]) <= 0
    ):
        raise ValueError("Finite-render sample/rejection eligibility config is invalid")
    root_seed = _parse_seed_hex(artifact["root_seed"])
    for attempt_index, attempt in enumerate(attempts):
        field_seeds = {
            field: _derived_seed(
                root_seed, artifact["split"], int(artifact["sample_index"]), field, attempt_index
            )
            for field in ("normal", "roll", "offset")
        }
        normal_raw = np.random.Generator(np.random.PCG64(field_seeds["normal"])).normal(size=3)
        normal, _, _ = canonicalize_plane(normal_raw, 0.0)
        roll = float(
            np.random.Generator(np.random.PCG64(field_seeds["roll"])).uniform(0.0, 2.0 * np.pi)
        )
        component_union = _component_interval_union_trusted(normal, support_index)
        sampling_intervals = _stratum_intervals(
            component_union, artifact["stratum"], float(config["boundary_stress_fraction"])
        )
        offset, selected_interval, measure_fraction = sample_interval_union_offset(
            sampling_intervals, field_seeds["offset"]
        )
        expected_attempt_pose = {
            "attempt_index": attempt_index,
            "field_stream_seed_uint64": {
                key: _seed_hex(value) for key, value in field_seeds.items()
            },
            "normal_rp2_ap_dv_ml": normal.tolist(),
            "roll_rad": roll,
            "component_interval_union_um": component_union.tolist(),
            "stratum_sampling_interval_union_um": sampling_intervals.tolist(),
            "selected_interval_index": selected_interval,
            "offset_measure_fraction": measure_fraction,
            "signed_offset_um": offset,
        }
        if any(attempt[key] != value for key, value in expected_attempt_pose.items()):
            raise ValueError("Finite-render attempt pose does not replay from seed and support")
    accepted = int(artifact["accepted_attempt_index"])
    if accepted != 0 or not attempts[accepted]["accepted"]:
        raise ValueError("Finite-render accepted attempt is inconsistent")
    accepted_attempt = attempts[accepted]
    brain_pixel_count = int(np.asarray(raster["brain_mask"]).sum())
    minimum_brain_pixels = int(config["minimum_brain_pixels"])
    if (
        accepted_attempt["geometry_sha256"] != geometry["geometry_sha256"]
        or accepted_attempt["raster_combined_sha256"] != expected_hashes["combined_sha256"]
        or accepted_attempt["brain_pixel_count"] != brain_pixel_count
        or artifact["acceptance_contract"]["brain_pixel_count"] != brain_pixel_count
        or artifact["acceptance_contract"]["minimum_brain_pixels"] != minimum_brain_pixels
        or artifact["acceptance_contract"].get(
            "continuous_plane_intersection_authenticated"
        ) is not True
        or artifact["acceptance_contract"].get("pose_redrawn_for_raster_support") is not False
        or artifact["acceptance_contract"].get("pose_draw_count") != 1
        or artifact["acceptance_contract"].get(
            "raster_support_meets_requested_identifiability_threshold"
        )
        != (brain_pixel_count >= minimum_brain_pixels)
        or artifact["acceptance_contract"].get("minimum_brain_pixels_role")
        != (
            "requested downstream point/dense-supervision identifiability threshold; "
            "not an acceptance predicate"
        )
    ):
        raise ValueError("Finite-render accepted attempt does not match installed geometry or raster")
    if any(
        bool(attempt["accepted"]) is not True
        or attempt.get("raster_support_meets_requested_identifiability_threshold")
        != (int(attempt["brain_pixel_count"]) >= minimum_brain_pixels)
        for attempt in attempts
    ):
        raise ValueError("Finite-render attempt acceptance predicate is inconsistent")
    if (
        accepted_attempt["normal_rp2_ap_dv_ml"] != geometry["normal_rp2_ap_dv_ml"]
        or accepted_attempt["signed_offset_um"] != geometry["signed_offset_um"]
        or accepted_attempt["roll_rad"] != geometry["roll_rad"]
    ):
        raise ValueError("Finite-render accepted pose does not match installed geometry")
    plane_identity = {
        "schema": "anatomy-tracker.finite-plane-realization/v1",
        "sampler_algorithm": FINITE_RENDER_ALGORITHM,
        "support_index_sha256": artifact["support_index_sha256"],
        "plane_sampler_lineage_sha256": generator["plane_sampler_lineage_sha256"],
        "split": artifact["split"],
        "root_seed": artifact["root_seed"],
        "sample_index": artifact["sample_index"],
        "accepted_attempt_index": accepted,
        "stratum": artifact["stratum"],
        "field_stream_seed_uint64": accepted_attempt["field_stream_seed_uint64"],
        "normal_rp2_ap_dv_ml": accepted_attempt["normal_rp2_ap_dv_ml"],
        "signed_offset_um": accepted_attempt["signed_offset_um"],
        "roll_rad": accepted_attempt["roll_rad"],
        "animal_id": artifact["provenance"]["animal_id"],
        "specimen_id": artifact["provenance"]["specimen_id"],
        "experiment_id": artifact["provenance"]["experiment_id"],
    }
    if artifact["plane_realization_id"] != _payload_sha256(plane_identity):
        raise ValueError("Finite-plane realization identifier does not match")
    expected_finite_geometry_sha256 = _payload_sha256(
        {
            "schema": "anatomy-tracker.finite-plane-geometry/v1",
            "plane_realization_id": artifact["plane_realization_id"],
            "geometry_sha256": geometry["geometry_sha256"],
        }
    )
    if artifact["finite_plane_geometry_sha256"] != expected_finite_geometry_sha256:
        raise ValueError("Finite-plane geometry identifier does not match")
    effective_sampling_receipts = {
        key: geometry["array_receipts"][key]
        for key in (
            "effective_coordinate_raster_allen_index_float32",
            "effective_normalized_interpolation_grid_xyz_float32",
            "valid_atlas_label_sampling_mask",
        )
    }
    expected_rendered_artifacts_receipt = {
        "schema": "anatomy-tracker.rendered-finite-plane-artifacts/v1",
        "effective_sampling_array_receipts": effective_sampling_receipts,
        "raster_array_receipts": expected_receipts,
    }
    if (
        artifact["rendered_artifacts_receipt"] != expected_rendered_artifacts_receipt
        or artifact["rendered_artifacts_sha256"]
        != _payload_sha256(expected_rendered_artifacts_receipt)
    ):
        raise ValueError("Rendered-artifacts identifier does not match")
    if artifact["finite_plane_render_id"] != _payload_sha256(_finite_render_identity(artifact)):
        raise ValueError("Finite-plane render identifier does not match")
    if "synthetic_realization_id" in artifact:
        raise ValueError("Finite geometry precursor must not issue a synthetic realization identifier")


def replay_finite_arbitrary_plane_render_from_context(
    artifact: dict[str, object],
    context: dict[str, object],
    *,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    _validate_prepared_context(context)
    support_index = context["support_index"]
    verify_finite_arbitrary_plane_render(
        artifact,
        support_index,
        generator_source_commit=generator_source_commit,
        _support_preverified=True,
    )
    config = artifact["generator"]["resolved_config"]
    if config["prepared_context_sha256"] != context["prepared_context_sha256"]:
        raise ValueError("Prepared context does not match finite-render config")
    replayed = make_finite_arbitrary_plane_render_from_context(
        context,
        config["split"],
        _parse_seed_hex(config["root_seed"]),
        tuple(config["output_shape_h_w"]),
        sample_index=config["sample_index"],
        stratum=config["stratum"],
        boundary_stress_fraction=config["boundary_stress_fraction"],
        margin_um=tuple(config["margin_u_v_um"]),
        animal_id=config["animal_id"],
        specimen_id=config["specimen_id"],
        experiment_id=config["experiment_id"],
        max_rejection_attempts=config["max_rejection_attempts"],
        minimum_brain_pixels=config["minimum_brain_pixels"],
        generator_source_commit=generator_source_commit,
    )
    if replayed["finite_plane_render_id"] != artifact["finite_plane_render_id"] or any(
        not np.array_equal(replayed["raster"][key], artifact["raster"][key])
        for key in artifact["raster"]
    ):
        raise ValueError("Prepared-context replay did not reproduce the finite render")
    return replayed


def replay_finite_arbitrary_plane_render(
    artifact: dict[str, object],
    scalar_volume_ap_dv_ml: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    generator_source_commit: str | None = None,
) -> dict[str, object]:
    verify_finite_arbitrary_plane_render(
        artifact, support_index, generator_source_commit=generator_source_commit
    )
    config = artifact["generator"]["resolved_config"]
    replayed = make_finite_arbitrary_plane_render(
        scalar_volume_ap_dv_ml,
        annotation_ap_dv_ml,
        support_index,
        config["split"],
        _parse_seed_hex(config["root_seed"]),
        tuple(config["output_shape_h_w"]),
        sample_index=config["sample_index"],
        stratum=config["stratum"],
        boundary_stress_fraction=config["boundary_stress_fraction"],
        margin_um=tuple(config["margin_u_v_um"]),
        scalar_source_uri=config["scalar_source_uri"],
        scalar_source_sha256=config["scalar_source_sha256"],
        scalar_source_entity_type=config["scalar_source_entity_type"],
        scalar_dtype=config["scalar_conversion"]["dtype"],
        template_decoder=config["template_decoder"],
        template_index_order=config["template_index_order"],
        annotation_decoder=config["annotation_decoder"],
        annotation_index_order=config["annotation_index_order"],
        animal_id=config["animal_id"],
        specimen_id=config["specimen_id"],
        experiment_id=config["experiment_id"],
        max_rejection_attempts=config["max_rejection_attempts"],
        minimum_brain_pixels=config["minimum_brain_pixels"],
        generator_source_commit=generator_source_commit,
    )
    if replayed["finite_plane_render_id"] != artifact["finite_plane_render_id"]:
        raise ValueError("Finite-render replay did not reproduce the identifier")
    return replayed
