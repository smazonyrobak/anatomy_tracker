from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import _qhull
from scipy.spatial import ConvexHull


SUPPORT_INDEX_SCHEMA = "anatomy-tracker.annotation-support-index/v1"
SUPPORT_INDEX_ALGORITHM = "six-connected-ap-line-endpoint-convex-hulls/v1"
PROJECTION_ALGORITHM = "component-hull-voxel-box-projection/v1"
INTERSECTION_ALGORITHM = "component-projection-interval-membership/v1"
DEFAULT_NORMAL_BATCH_SIZE = 1024


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
    packed = np.packbits(np.asarray(mask, dtype=bool).reshape(-1, order="C"), bitorder="little")
    digest = hashlib.sha256()
    digest.update(_canonical_json({"shape": list(mask.shape), "bitorder": "little"}).encode("utf-8"))
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _qhull_provenance() -> dict[str, str]:
    binary = Path(_qhull.__file__).read_bytes()
    match = re.search(rb"qhull_r ([0-9.]+) \(([^)]+)\)", binary)
    version = (
        f"{match.group(1).decode('ascii')} ({match.group(2).decode('ascii')})"
        if match is not None
        else "unreported-by-scipy-binary"
    )
    return {"qhull_version": version, "qhull_extension_sha256": hashlib.sha256(binary).hexdigest()}


def _scan_line_endpoints(mask: np.ndarray, scan_axis: int, offset: np.ndarray) -> np.ndarray:
    lines = np.moveaxis(mask, scan_axis, -1)
    present = lines.any(axis=-1)
    other = np.argwhere(present)
    lower = np.argmax(lines, axis=-1)[present]
    upper = lines.shape[-1] - 1 - np.argmax(lines[..., ::-1], axis=-1)[present]
    other_axes = [axis for axis in range(3) if axis != scan_axis]
    lower_points = np.empty((len(other), 3), dtype=np.int64)
    upper_points = np.empty((len(other), 3), dtype=np.int64)
    lower_points[:, scan_axis] = lower
    upper_points[:, scan_axis] = upper
    for column, axis in enumerate(other_axes):
        lower_points[:, axis] = other[:, column]
        upper_points[:, axis] = other[:, column]
    return np.unique(np.vstack((lower_points, upper_points)) + offset, axis=0)


def _compact_hull_vertices(endpoints: np.ndarray) -> tuple[np.ndarray, int, str]:
    points = np.asarray(endpoints, dtype=np.int64)
    differences = points - points[0]
    nonzero = np.flatnonzero(np.any(differences != 0, axis=1))
    if len(nonzero) == 0:
        return points[:1], 0, "singleton"
    direction = differences[nonzero[0]]
    crosses = np.cross(direction, differences)
    independent = np.flatnonzero(np.any(crosses != 0, axis=1))
    if len(independent) == 0:
        coordinate = differences @ direction
        selected = np.unique([int(np.argmin(coordinate)), int(np.argmax(coordinate))])
        return np.unique(points[selected], axis=0), 1, "integer-line-extrema"
    plane_normal = crosses[independent[0]]
    if not np.any(differences @ plane_normal):
        dropped_axis = int(np.argmax(np.abs(plane_normal)))
        projected = np.delete(points, dropped_axis, axis=1).astype(np.float64)
        hull = ConvexHull(projected, qhull_options="Qx")
        return np.unique(points[hull.vertices], axis=0), 2, "scipy.spatial.ConvexHull-2d-Qx"
    hull = ConvexHull(points.astype(np.float64), qhull_options="Qx")
    return np.unique(points[hull.vertices], axis=0), 3, "scipy.spatial.ConvexHull-3d-Qx"


def _component_hull(
    mask: np.ndarray, scan_axis: int, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    endpoints = _scan_line_endpoints(mask, scan_axis, offset)
    vertices, affine_rank, backend = _compact_hull_vertices(endpoints)
    maximum_index = int(endpoints.max())
    dtype = np.uint16 if maximum_index <= np.iinfo(np.uint16).max else np.uint32
    endpoints = endpoints.astype(dtype)
    vertices = vertices.astype(dtype)
    occupied_axes = [
        np.flatnonzero(mask.any(axis=tuple(other for other in range(3) if other != axis)))
        for axis in range(3)
    ]
    lower = np.asarray([values[0] for values in occupied_axes], dtype=np.int64) + offset
    upper = np.asarray([values[-1] for values in occupied_axes], dtype=np.int64) + offset
    metadata = {
        "occupied_voxel_count": int(mask.sum()),
        "bounding_box_index_inclusive": [lower.tolist(), upper.tolist()],
        "anchor_index": lower.tolist(),
        "line_endpoint_count": int(len(endpoints)),
        "line_endpoint_index_dtype": endpoints.dtype.str,
        "line_endpoint_indices_sha256": _array_sha256(endpoints),
        "hull_vertex_count": int(len(vertices)),
        "affine_rank": affine_rank,
        "hull_backend": backend,
        "hull_index_dtype": vertices.dtype.str,
        "hull_indices_sha256": _array_sha256(vertices),
    }
    return endpoints, vertices, metadata


def build_annotation_support_index(
    annotation: np.ndarray,
    *,
    atlas_id: str,
    atlas_version: str,
    source_uri: str,
    voxel_size_um: tuple[float, float, float],
    origin_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    projection_origin_um: tuple[float, float, float] | None = None,
    coordinate_axes: tuple[str, str, str] = ("AP", "DV", "ML"),
    source_entity_type: str,
    coordinate_axis_directions: tuple[str, str, str],
    scan_axis: str | int = "AP",
    source_sha256: str,
    normal_batch_size: int = DEFAULT_NORMAL_BATCH_SIZE,
) -> dict[str, object]:
    annotation = np.asarray(annotation)
    if annotation.ndim != 3:
        raise ValueError("Annotation must be three-dimensional")
    mask = annotation != 0
    if not mask.any():
        raise ValueError("Annotation support is empty")
    spacing = np.asarray(voxel_size_um, dtype=np.float64)
    origin = np.asarray(origin_um, dtype=np.float64)
    axes = tuple(str(axis) for axis in coordinate_axes)
    axis_directions = tuple(str(direction) for direction in coordinate_axis_directions)
    source_entity_type = str(source_entity_type)
    normal_batch_size = int(normal_batch_size)
    if spacing.shape != (3,) or np.any(spacing <= 0.0) or not np.isfinite(spacing).all():
        raise ValueError("voxel_size_um must contain three finite positive values")
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or len(axes) != 3
        or len(axis_directions) != 3
    ):
        raise ValueError(
            "origin_um, coordinate_axes and coordinate_axis_directions must contain three values"
        )
    if not source_entity_type:
        raise ValueError("source_entity_type must be nonempty")
    if normal_batch_size <= 0:
        raise ValueError("normal_batch_size must be positive")
    scan_axis_index = axes.index(scan_axis) if isinstance(scan_axis, str) else int(scan_axis)
    if scan_axis_index not in range(3):
        raise ValueError("scan_axis must identify one annotation axis")
    projection_origin = (
        origin + np.asarray(annotation.shape, dtype=np.float64) * spacing / 2.0
        if projection_origin_um is None
        else np.asarray(projection_origin_um, dtype=np.float64)
    )
    if projection_origin.shape != (3,) or not np.isfinite(projection_origin).all():
        raise ValueError("projection_origin_um must contain three finite values")
    annotation_sha256 = _array_sha256(annotation)
    source_sha256 = str(source_sha256)
    if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
        raise ValueError("source_sha256 must be the lowercase SHA-256 of the raw source bytes")
    labels, component_count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    component_endpoints = []
    component_hulls = []
    component_metadata = []
    if component_count == 1:
        endpoints, vertices, metadata = _component_hull(
            mask, scan_axis_index, np.zeros(3, dtype=np.int64)
        )
        component_endpoints.append(endpoints)
        component_hulls.append(vertices)
        component_metadata.append(metadata)
    else:
        for label, slices in enumerate(ndimage.find_objects(labels), 1):
            offset = np.asarray([axis_slice.start for axis_slice in slices], dtype=np.int64)
            endpoints, vertices, metadata = _component_hull(
                labels[slices] == label, scan_axis_index, offset
            )
            component_endpoints.append(endpoints)
            component_hulls.append(vertices)
            component_metadata.append(metadata)
    ordered = sorted(
        zip(component_metadata, component_endpoints, component_hulls),
        key=lambda item: (tuple(item[0]["anchor_index"]), item[0]["line_endpoint_indices_sha256"]),
    )
    component_metadata = []
    component_endpoints = []
    component_hulls = []
    for component_index, (metadata, endpoints, vertices) in enumerate(ordered):
        component_metadata.append({"component_index": component_index, **metadata})
        component_endpoints.append(endpoints)
        component_hulls.append(vertices)
    metadata = {
        "schema_version": SUPPORT_INDEX_SCHEMA,
        "algorithm": SUPPORT_INDEX_ALGORITHM,
        "atlas": {
            "id": str(atlas_id),
            "version": str(atlas_version),
            "coordinate_axes": list(axes),
            "coordinate_axis_directions": list(axis_directions),
            "coordinate_unit": "um",
        },
        "source": {
            "source_entity_type": source_entity_type,
            "annotation_uri": str(source_uri),
            "source_sha256": source_sha256,
            "source_sha256_semantics": "raw source bytes",
            "annotation_array_sha256": annotation_sha256,
        },
        "annotation_shape": list(annotation.shape),
        "voxel_size_um": spacing.tolist(),
        "origin_um": origin.tolist(),
        "projection_origin_um": projection_origin.tolist(),
        "voxel_coordinate_semantics": "origin-plus-index-plus-half-voxel; closed axis-aligned boxes",
        "foreground_voxel_count": int(mask.sum()),
        "support_mask_sha256": _mask_sha256(mask),
        "connectivity": "six-face-connected occupied voxel boxes",
        "component_count": int(component_count),
        "projection_interval_mode": (
            "single-connected-component-fast-path"
            if component_count == 1
            else "per-connected-component-interval-union"
        ),
        "scan_axis": {"index": scan_axis_index, "name": axes[scan_axis_index]},
        "projection_query": {"normal_batch_size": normal_batch_size},
        "convex_hull_dependency": {
            "implementation": "scipy.spatial.ConvexHull",
            "qhull_options": "Qx",
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            **_qhull_provenance(),
        },
        "exactness": (
            "scan-line endpoint reduction and connected-component projection intervals are exact for the "
            "declared voxel boxes; convex-hull construction is numerical under the pinned SciPy/Qhull backend"
        ),
        "components": component_metadata,
    }
    return {
        **metadata,
        "component_line_endpoint_indices": component_endpoints,
        "component_hull_indices": component_hulls,
        "support_index_sha256": _payload_sha256(metadata),
    }


def verify_annotation_support_index(index: dict[str, object]) -> None:
    if (
        index.get("schema_version") != SUPPORT_INDEX_SCHEMA
        or index.get("algorithm") != SUPPORT_INDEX_ALGORITHM
    ):
        raise ValueError("Unsupported annotation support index")
    metadata = {
        key: value
        for key, value in index.items()
        if key
        not in {
            "component_line_endpoint_indices",
            "component_hull_indices",
            "support_index_sha256",
        }
    }
    if index.get("support_index_sha256") != _payload_sha256(metadata):
        raise ValueError("Support-index metadata does not match support_index_sha256")
    endpoints = index["component_line_endpoint_indices"]
    hulls = index["component_hull_indices"]
    components = index["components"]
    if len(endpoints) != len(components) or len(hulls) != len(components):
        raise ValueError("Support-index component arrays are incomplete")
    for component, line_endpoints, vertices in zip(components, endpoints, hulls):
        if len(line_endpoints) != component["line_endpoint_count"]:
            raise ValueError(
                f"Support-index component {component['component_index']} endpoint count does not match"
            )
        if len(vertices) != component["hull_vertex_count"]:
            raise ValueError(
                f"Support-index component {component['component_index']} hull count does not match"
            )
        if component["line_endpoint_indices_sha256"] != _array_sha256(line_endpoints):
            raise ValueError(
                f"Support-index component {component['component_index']} endpoint hash does not match"
            )
        if component["hull_indices_sha256"] != _array_sha256(vertices):
            raise ValueError(f"Support-index component {component['component_index']} hull hash does not match")


def replay_annotation_support_index(annotation: np.ndarray, index: dict[str, object]) -> dict[str, object]:
    verify_annotation_support_index(index)
    replayed = build_annotation_support_index(
        annotation,
        atlas_id=index["atlas"]["id"],
        atlas_version=index["atlas"]["version"],
        source_uri=index["source"]["annotation_uri"],
        source_sha256=index["source"]["source_sha256"],
        voxel_size_um=tuple(index["voxel_size_um"]),
        origin_um=tuple(index["origin_um"]),
        projection_origin_um=tuple(index["projection_origin_um"]),
        coordinate_axes=tuple(index["atlas"]["coordinate_axes"]),
        coordinate_axis_directions=tuple(index["atlas"]["coordinate_axis_directions"]),
        source_entity_type=index["source"]["source_entity_type"],
        scan_axis=int(index["scan_axis"]["index"]),
        normal_batch_size=int(index["projection_query"]["normal_batch_size"]),
    )
    if replayed["support_index_sha256"] != index["support_index_sha256"]:
        raise ValueError("Annotation did not reproduce the support index")
    return replayed


def _canonicalize_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[-1] != 3:
        raise ValueError("Normals must have a final dimension of three")
    flat = normals.reshape(-1, 3)
    lengths = np.linalg.norm(flat, axis=1)
    if np.any(lengths == 0.0) or not np.isfinite(flat).all():
        raise ValueError("Normals must be finite and nonzero")
    unit = flat / lengths[:, None]
    pivots = np.argmax(np.abs(unit), axis=1)
    signs = np.where(unit[np.arange(len(unit)), pivots] < 0.0, -1.0, 1.0)
    canonical = unit * signs[:, None]
    canonical = np.where(canonical == 0.0, 0.0, canonical)
    return canonical, lengths, signs


def _component_bounds(
    canonical_normals: np.ndarray,
    index: dict[str, object],
    normal_batch_size: int | None,
) -> np.ndarray:
    spacing = np.asarray(index["voxel_size_um"], dtype=np.float64)
    origin = np.asarray(index["origin_um"], dtype=np.float64)
    projection_origin = np.asarray(index["projection_origin_um"], dtype=np.float64)
    batch_size = (
        int(index["projection_query"]["normal_batch_size"])
        if normal_batch_size is None
        else int(normal_batch_size)
    )
    if batch_size <= 0:
        raise ValueError("normal_batch_size must be positive")
    half_extent = np.abs(canonical_normals) @ (spacing / 2.0)
    bounds = np.empty((len(canonical_normals), len(index["component_hull_indices"]), 2), dtype=np.float64)
    for component, vertices in enumerate(index["component_hull_indices"]):
        physical = origin + (np.asarray(vertices, dtype=np.float64) + 0.5) * spacing - projection_origin
        for start in range(0, len(canonical_normals), batch_size):
            stop = min(start + batch_size, len(canonical_normals))
            batch = canonical_normals[start:stop]
            projected = batch[:, 0, None] * physical[None, :, 0]
            projected += batch[:, 1, None] * physical[None, :, 1]
            projected += batch[:, 2, None] * physical[None, :, 2]
            bounds[start:stop, component, 0] = projected.min(axis=1) - half_extent[start:stop]
            bounds[start:stop, component, 1] = projected.max(axis=1) + half_extent[start:stop]
    return bounds


def support_projection_bounds(
    normals: np.ndarray,
    index: dict[str, object],
    *,
    normal_batch_size: int | None = None,
) -> dict[str, object]:
    verify_annotation_support_index(index)
    input_shape = np.asarray(normals).shape[:-1]
    canonical, _, _ = _canonicalize_normals(normals)
    component_bounds = _component_bounds(canonical, index, normal_batch_size)
    global_bounds = np.stack((component_bounds[..., 0].min(axis=1), component_bounds[..., 1].max(axis=1)), axis=-1)
    canonical = canonical.reshape(*input_shape, 3)
    component_bounds = component_bounds.reshape(*input_shape, len(index["components"]), 2)
    global_bounds = global_bounds.reshape(*input_shape, 2)
    payload = {
        "schema": "anatomy-tracker.support-projection-batch/v1",
        "algorithm": PROJECTION_ALGORITHM,
        "support_index_sha256": index["support_index_sha256"],
        "normal_rp2_sha256": _array_sha256(canonical),
        "component_bounds_um_sha256": _array_sha256(component_bounds),
        "global_bounds_um_sha256": _array_sha256(global_bounds),
    }
    return {
        "algorithm": PROJECTION_ALGORITHM,
        "support_index_sha256": index["support_index_sha256"],
        "normal_rp2": canonical,
        "component_bounds_um": component_bounds,
        "global_bounds_um": global_bounds,
        "projection_sha256": _payload_sha256(payload),
    }


def plane_interval_membership_certificate(
    normals: np.ndarray,
    signed_offsets_um: np.ndarray | float,
    index: dict[str, object],
    *,
    normal_batch_size: int | None = None,
) -> dict[str, object]:
    verify_annotation_support_index(index)
    input_shape = np.asarray(normals).shape[:-1]
    canonical, lengths, signs = _canonicalize_normals(normals)
    offsets = np.broadcast_to(np.asarray(signed_offsets_um, dtype=np.float64), input_shape).reshape(-1)
    offsets = signs * offsets / lengths
    component_bounds = _component_bounds(canonical, index, normal_batch_size)
    component_membership = (offsets[:, None] >= component_bounds[..., 0]) & (
        offsets[:, None] <= component_bounds[..., 1]
    )
    intersects = component_membership.any(axis=1)
    canonical = canonical.reshape(*input_shape, 3)
    offsets = offsets.reshape(input_shape)
    component_bounds = component_bounds.reshape(*input_shape, len(index["components"]), 2)
    component_membership = component_membership.reshape(*input_shape, len(index["components"]))
    intersects = intersects.reshape(input_shape)
    payload = {
        "schema": "anatomy-tracker.support-interval-membership-certificate/v1",
        "algorithm": INTERSECTION_ALGORITHM,
        "support_index_sha256": index["support_index_sha256"],
        "normal_rp2_sha256": _array_sha256(canonical),
        "signed_offset_um_sha256": _array_sha256(offsets),
        "component_bounds_um_sha256": _array_sha256(component_bounds),
        "component_membership_sha256": _array_sha256(component_membership),
        "intersects_sha256": _array_sha256(intersects),
    }
    return {
        "algorithm": INTERSECTION_ALGORITHM,
        "comparison": "inclusive float64 interval membership without tolerance",
        "support_index_sha256": index["support_index_sha256"],
        "normal_rp2": canonical,
        "signed_offset_um": offsets,
        "component_bounds_um": component_bounds,
        "component_membership": component_membership,
        "intersects": intersects,
        "certificate_sha256": _payload_sha256(payload),
    }
