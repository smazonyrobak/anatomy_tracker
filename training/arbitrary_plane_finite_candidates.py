"""Deterministic fixed-canvas candidates around a verified arbitrary plane.

Candidate planes are transported from the parent physical centre/frame/basis;
the atlas support is never used to refit a candidate crop, scale, or pitch.
The truth keeps the verified finite-parent geometry and raster byte-for-byte.
No target observation, learned asset, or historical model enters generation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from training.arbitrary_plane_geometry import (
    QUICKNII_RASTER_INDEX_SAMPLING,
    allen_to_quicknii_points,
    allen_to_quicknii_vectors,
    physical_um_to_allen_index_points,
    physical_um_to_allen_index_vectors,
    quicknii_ouv_to_frame,
)
from training.arbitrary_plane_manifest import canonicalize_plane
from training.arbitrary_plane_rendered_generator import (
    _component_interval_union_trusted,
    effective_renderer_sampling_arrays,
    finite_render_receipt,
    verify_finite_arbitrary_plane_render,
)
from training.arbitrary_plane_support import verify_annotation_support_index


FINITE_CANDIDATE_SCHEMA = "anatomy-tracker.arbitrary-plane-finite-candidate-bank/v1"
FINITE_CANDIDATE_ALGORITHM = "arbitrary-plane-finite-candidates/v1"
FINITE_CANDIDATE_RENDER_SCHEMA = "anatomy-tracker.fixed-canvas-candidate-label-render/v1"
DEFAULT_CANDIDATE_ROOT_SEED = 0xCAAD1DA7E0000001
DEFAULT_SHUFFLE_ROOT_SEED = 0x5C0E5EED00000001
MINIMUM_CANDIDATE_BRAIN_PIXELS = 64
MAXIMUM_GLOBAL_ATTEMPTS = 4096
ORDINARY_OFFSET_MARGIN_UM = 550.0
DESIGN_PLANE_TOLERANCE_UM = 1e-9
EFFECTIVE_PLANE_TOLERANCE_UM = 0.01
FINITE_CANDIDATE_CONTEXT_SCHEMA = "anatomy-tracker.prepared-finite-candidate-annotation/v1"

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = {
    "candidate_generator": Path(__file__).resolve(),
    "finite_renderer": Path(__file__).with_name("arbitrary_plane_rendered_generator.py"),
    "geometry": Path(__file__).with_name("arbitrary_plane_geometry.py"),
    "manifest": Path(__file__).with_name("arbitrary_plane_manifest.py"),
    "support": Path(__file__).with_name("arbitrary_plane_support.py"),
    "predeclared_protocol": _ROOT / "publication" / "arbitrary_plane_oracle_pose_ranking_preflight.yaml",
}
_LOADED_SOURCE_SHA256 = {
    name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in _SOURCE_PATHS.items()
}
_PREPARED_ANNOTATION_CONTEXT_TOKEN = object()


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_scalar,
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    dtype = value.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(value.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": dtype.str, "shape": list(value.shape)}).encode("utf-8"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask, dtype=bool)
    packed = np.packbits(value.reshape(-1, order="C"), bitorder="little")
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": "|b1", "shape": list(value.shape), "bitorder": "little"}).encode(
            "utf-8"
        )
    )
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "array_sha256": _mask_sha256(value) if value.dtype == np.bool_ else _array_sha256(value),
    }


def prepare_arbitrary_plane_finite_candidate_context(
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> MappingProxyType:
    """Verify/hash the decoded atlas annotation once and freeze an owned C view."""
    verify_annotation_support_index(support_index)
    decoded = np.asarray(annotation_ap_dv_ml)
    if decoded.shape != tuple(support_index["annotation_shape"]) or not np.issubdtype(
        decoded.dtype, np.integer
    ):
        raise ValueError("Annotation must match the verified support shape and integer dtype")
    if decoded.size and (decoded.min() < 0 or decoded.max() > np.iinfo(np.int64).max):
        raise ValueError("Annotation labels must convert losslessly to signed int64")
    contiguous = np.ascontiguousarray(decoded)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    annotation_sha256 = _array_sha256(frozen)
    if annotation_sha256 != support_index["source"]["annotation_array_sha256"]:
        raise ValueError("Annotation array does not match the verified support source")
    receipt = {
        "schema": FINITE_CANDIDATE_CONTEXT_SCHEMA,
        "support_index_sha256": support_index["support_index_sha256"],
        "annotation": {
            "dtype": frozen.dtype.str,
            "shape": list(frozen.shape),
            "array_sha256": annotation_sha256,
            "storage": "owned immutable C-order bytes",
        },
    }
    return MappingProxyType(
        {
            "schema": FINITE_CANDIDATE_CONTEXT_SCHEMA,
            "_token": _PREPARED_ANNOTATION_CONTEXT_TOKEN,
            "annotation": frozen,
            "annotation_id": id(frozen),
            "annotation_base_id": id(frozen.base),
            "receipt": receipt,
            "prepared_context_sha256": _payload_sha256(receipt),
        }
    )


def _validate_prepared_annotation_context(
    context: MappingProxyType,
    support_index: dict[str, object],
) -> tuple[np.ndarray, str, str]:
    if (
        not isinstance(context, MappingProxyType)
        or context.get("schema") != FINITE_CANDIDATE_CONTEXT_SCHEMA
        or context.get("_token") is not _PREPARED_ANNOTATION_CONTEXT_TOKEN
    ):
        raise ValueError("Unsupported or forged prepared candidate annotation context")
    annotation = context["annotation"]
    receipt = context["receipt"]
    if (
        not isinstance(annotation, np.ndarray)
        or id(annotation) != context["annotation_id"]
        or id(annotation.base) != context["annotation_base_id"]
        or annotation.flags.writeable
        or not annotation.flags.c_contiguous
        or annotation.dtype.str != receipt["annotation"]["dtype"]
        or list(annotation.shape) != receipt["annotation"]["shape"]
        or context["prepared_context_sha256"] != _payload_sha256(receipt)
        or receipt["support_index_sha256"] != support_index["support_index_sha256"]
        or receipt["annotation"]["array_sha256"]
        != support_index["source"]["annotation_array_sha256"]
    ):
        raise ValueError("Prepared candidate annotation context identity or receipt changed")
    return annotation, receipt["annotation"]["array_sha256"], context["prepared_context_sha256"]


def _uint64(value: int | str) -> int:
    if isinstance(value, str):
        if re.fullmatch(r"0x[0-9a-f]{16}", value) is None:
            raise ValueError("Seed strings must use canonical lowercase uint64 hexadecimal encoding")
        value = int(value, 16)
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("Seed must fit an unsigned 64-bit integer")
    return value


def _seed_hex(value: int | str) -> str:
    return f"0x{_uint64(value):016x}"


def derive_finite_candidate_seed(
    root_seed: int | str,
    base_plane_realization_id: str,
    candidate_class: str,
    slot: int | str,
    attempt: int,
) -> int:
    """Derive the exact predeclared independent candidate stream seed."""
    if re.fullmatch(r"[0-9a-f]{64}", str(base_plane_realization_id)) is None:
        raise ValueError("base_plane_realization_id must be a lowercase SHA-256 digest")
    if int(attempt) < 0:
        raise ValueError("attempt must be nonnegative")
    payload = (
        f"{FINITE_CANDIDATE_ALGORITHM}\0{_seed_hex(root_seed)}\0{base_plane_realization_id}"
        f"\0{candidate_class}\0{slot}\0{int(attempt)}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def align_rp2_pose_to_reference(
    normal_ap_dv_ml: np.ndarray,
    signed_offset_um: float,
    reference_normal_ap_dv_ml: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize a plane and choose the RP2 sign nearest the reference normal."""
    normal = np.asarray(normal_ap_dv_ml, dtype=np.float64)
    reference = np.asarray(reference_normal_ap_dv_ml, dtype=np.float64)
    offset = float(signed_offset_um)
    if normal.shape != (3,) or reference.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("Plane normals must be finite three-vectors")
    if not np.isfinite(reference).all() or not np.isfinite(offset):
        raise ValueError("Reference normal and signed offset must be finite")
    reference_scale = float(np.linalg.norm(reference))
    if float(np.linalg.norm(normal)) == 0.0 or reference_scale == 0.0:
        raise ValueError("Plane normals must be nonzero")
    normal, offset, _ = canonicalize_plane(normal, offset)
    reference = reference / reference_scale
    if float(normal @ reference) < 0.0:
        normal = -normal
        offset = -offset
    return normal, float(offset)


def minimal_normal_rotation(
    parent_normal_ap_dv_ml: np.ndarray,
    candidate_normal_ap_dv_ml: np.ndarray,
) -> np.ndarray:
    """Return the minimal proper rotation taking one aligned unit normal to another."""
    parent = np.array(parent_normal_ap_dv_ml, dtype=np.float64, copy=True)
    candidate = np.array(candidate_normal_ap_dv_ml, dtype=np.float64, copy=True)
    if parent.shape != (3,) or candidate.shape != (3,):
        raise ValueError("Normals must be three-vectors")
    parent /= np.linalg.norm(parent)
    candidate /= np.linalg.norm(candidate)
    cosine = float(np.clip(parent @ candidate, -1.0, 1.0))
    if cosine < -1.0 + 1e-12:
        raise ValueError("Candidate normal must be sign-aligned before minimal transport")
    cross = np.cross(parent, candidate)
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-15:
        return np.eye(3, dtype=np.float64)
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def _parent_physical_state(parent_geometry: dict[str, object]) -> dict[str, np.ndarray | float]:
    if parent_geometry["sampling_contract"] != QUICKNII_RASTER_INDEX_SAMPLING:
        raise ValueError("Finite candidates require the current x/W,y/H sampling contract")
    frame = np.asarray(parent_geometry["frame_ap_dv_ml_physical"], dtype=np.float64)
    ouv = np.asarray(parent_geometry["physical_ouv_ap_dv_ml_um"], dtype=np.float64)
    normal = np.asarray(parent_geometry["normal_rp2_ap_dv_ml"], dtype=np.float64)
    offset = float(parent_geometry["signed_offset_um"])
    if frame.shape != (3, 3) or ouv.shape != (9,) or not np.isfinite(frame).all() or not np.isfinite(ouv).all():
        raise ValueError("Parent physical frame and O/U/V must be finite")
    edges = np.stack((ouv[3:6], ouv[6:9]), axis=-1)
    basis = frame[:, :2].T @ edges
    if (
        not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10)
        or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-10)
        or np.linalg.det(basis) <= 0.0
        or not np.allclose(frame[:, :2] @ basis, edges, atol=1e-8)
    ):
        raise ValueError("Parent physical frame/basis is not right-handed and finite")
    center = ouv[:3] + 0.5 * edges.sum(axis=1)
    return {"center": center, "frame": frame, "basis": basis, "normal": normal, "offset": offset}


def transport_finite_candidate_pose(
    parent_geometry: dict[str, object],
    support_index: dict[str, object],
    candidate_normal_ap_dv_ml: np.ndarray,
    candidate_signed_offset_um: float,
    roll_delta_rad: float,
) -> dict[str, object]:
    """Transport one proposal without changing the parent's finite canvas."""
    parent = _parent_physical_state(parent_geometry)
    n0 = np.asarray(parent["normal"])
    nc, dc = align_rp2_pose_to_reference(candidate_normal_ap_dv_ml, candidate_signed_offset_um, n0)
    roll_delta = float(roll_delta_rad)
    if not np.isfinite(roll_delta):
        raise ValueError("Roll delta must be finite")
    rotation = minimal_normal_rotation(n0, nc)
    q = np.asarray(support_index["projection_origin_um"], dtype=np.float64)
    c0 = np.asarray(parent["center"])
    d0 = float(parent["offset"])
    r0 = c0 - q - d0 * n0
    if abs(float(n0 @ r0)) > DESIGN_PLANE_TOLERANCE_UM:
        raise ValueError("Parent finite centre violates its float64 design plane equation")
    center = q + dc * nc + rotation @ r0
    transported = rotation @ np.asarray(parent["frame"])
    cosine, sine = math.cos(roll_delta), math.sin(roll_delta)
    u = cosine * transported[:, 0] + sine * transported[:, 1]
    v = -sine * transported[:, 0] + cosine * transported[:, 1]
    frame = np.stack((u, v, nc), axis=-1)
    basis = np.asarray(parent["basis"])
    edges = frame[:, :2] @ basis
    origin = center - 0.5 * edges.sum(axis=1)
    design_residual = float(nc @ (center - q) - dc)
    if abs(design_residual) > DESIGN_PLANE_TOLERANCE_UM:
        raise ValueError("Transported centre violates the predeclared float64 plane equation")

    spacing = tuple(float(value) for value in support_index["voxel_size_um"])
    atlas_origin = tuple(float(value) for value in support_index["origin_um"])
    atlas_shape = tuple(int(value) for value in support_index["annotation_shape"])
    origin_index = physical_um_to_allen_index_points(
        torch.as_tensor(origin, dtype=torch.float64), atlas_origin, spacing
    ).numpy()
    edge_u_index = physical_um_to_allen_index_vectors(
        torch.as_tensor(edges[:, 0], dtype=torch.float64), spacing
    ).numpy()
    edge_v_index = physical_um_to_allen_index_vectors(
        torch.as_tensor(edges[:, 1], dtype=torch.float64), spacing
    ).numpy()
    quicknii_ouv = np.concatenate(
        (
            allen_to_quicknii_points(torch.as_tensor(origin_index), atlas_shape).numpy(),
            allen_to_quicknii_vectors(torch.as_tensor(edge_u_index)).numpy(),
            allen_to_quicknii_vectors(torch.as_tensor(edge_v_index)).numpy(),
        )
    )
    renderer_center, renderer_frame, renderer_basis = quicknii_ouv_to_frame(
        torch.as_tensor(quicknii_ouv, dtype=torch.float64), atlas_shape
    )
    geometry = {
        "normal_rp2_sign_aligned_ap_dv_ml": nc.tolist(),
        "signed_offset_um": dc,
        "roll_delta_rad_from_parallel_transport": roll_delta,
        "center_ap_dv_ml_um": center.tolist(),
        "frame_ap_dv_ml_physical": frame.tolist(),
        "inplane_basis_u_v_um": basis.tolist(),
        "physical_ouv_ap_dv_ml_um": np.concatenate((origin, edges[:, 0], edges[:, 1])).tolist(),
        "renderer_center_ap_dv_ml": renderer_center.to(torch.float32).tolist(),
        "renderer_frame_ap_dv_ml": renderer_frame.to(torch.float32).tolist(),
        "renderer_inplane_basis": renderer_basis.to(torch.float32).tolist(),
        "effective_renderer_dtype": "<f4",
        "output_shape_h_w": copy.deepcopy(parent_geometry["output_shape_h_w"]),
        "sampling_contract": parent_geometry["sampling_contract"],
        "reflection_state": copy.deepcopy(parent_geometry["reflection_state"]),
        "physical_pixel_pitch_u_v_um": [
            float(parent_geometry["reference_aspect_policy"]["pixel_pitch_u_um"]),
            float(parent_geometry["reference_aspect_policy"]["pixel_pitch_v_um"]),
        ],
        "parent_canvas": {
            "parent_geometry_sha256": parent_geometry["geometry_sha256"],
            "policy": "transport centre/frame/basis; never refit crop, scale, pitch, spans, or reflection",
        },
        "design_plane_equation": {
            "formula": "dot(n,c-q)-d",
            "residual_um": design_residual,
            "absolute_tolerance_um": DESIGN_PLANE_TOLERANCE_UM,
        },
    }
    effective = effective_renderer_sampling_arrays(
        geometry,
        atlas_shape,
        origin_ap_dv_ml_um=atlas_origin,
        voxel_size_ap_dv_ml_um=spacing,
    )
    if any(
        not np.isfinite(array).all()
        for array in effective.values()
        if np.asarray(array).dtype != np.bool_
    ):
        raise ValueError("Effective candidate renderer coordinates must be finite")
    effective_ouv = effective["physical_ouv_ap_dv_ml_um_from_float32_state"]
    effective_center = effective_ouv[:3] + 0.5 * (effective_ouv[3:6] + effective_ouv[6:9])
    effective_residual = float(nc @ (effective_center - q) - dc)
    effective_tolerance = EFFECTIVE_PLANE_TOLERANCE_UM
    if abs(effective_residual) > effective_tolerance:
        raise ValueError("Effective float32 candidate drift exceeds its declared tolerance")
    intervals = _component_interval_union_trusted(nc, support_index)
    membership = (dc >= intervals[:, 0]) & (dc <= intervals[:, 1])
    geometry.update(
        {
            "effective_allen_index_ouv_ap_dv_ml": effective[
                "allen_index_ouv_ap_dv_ml_float32"
            ].tolist(),
            "effective_physical_ouv_ap_dv_ml_um": effective_ouv.tolist(),
            "effective_quicknii_ouv_ml_ap_dv": effective[
                "quicknii_ouv_ml_ap_dv_float32"
            ].tolist(),
            "effective_plane_equation": {
                "formula": "dot(n,c_effective-q)-d",
                "residual_um": effective_residual,
                "absolute_tolerance_um": effective_tolerance,
                "status": "diagnostic tolerance for the actual float32 renderer state",
            },
            "support_intersection": {
                "component_interval_union_um": intervals.tolist(),
                "component_membership": membership.tolist(),
                "brain_intersection": bool(membership.any()),
                "acceptance_independence": "computed from candidate and atlas support only; target overlap absent",
            },
            "array_receipts": {
                name: _array_receipt(array) for name, array in effective.items()
            },
        }
    )
    geometry["candidate_geometry_sha256"] = _payload_sha256(geometry)
    return geometry


def render_finite_candidate_annotation(
    annotation_ap_dv_ml: np.ndarray,
    geometry: dict[str, object],
    atlas_shape_ap_dv_ml: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-render atlas IDs on the candidate x/W,y/H lattice."""
    annotation = np.asarray(annotation_ap_dv_ml)
    if annotation.ndim != 3 or not np.issubdtype(annotation.dtype, np.integer):
        raise ValueError("Annotation must be one integer AP-DV-ML array")
    shape = tuple(annotation.shape) if atlas_shape_ap_dv_ml is None else tuple(atlas_shape_ap_dv_ml)
    if tuple(annotation.shape) != shape:
        raise ValueError("Annotation and atlas shape must match")
    effective = effective_renderer_sampling_arrays(geometry, shape)
    indices = torch.round(torch.from_numpy(effective["coordinate_raster_allen_index_float32"])).to(
        torch.int64
    ).numpy()
    valid = np.ones(indices.shape[:2], dtype=bool)
    for axis, size in enumerate(shape):
        valid &= (indices[..., axis] >= 0) & (indices[..., axis] < size)
    clipped = np.stack(
        [np.clip(indices[..., axis], 0, size - 1) for axis, size in enumerate(shape)], axis=-1
    )
    labels = annotation[clipped[..., 0], clipped[..., 1], clipped[..., 2]].astype(np.int64, copy=False)
    labels = np.where(valid, labels, 0).astype(np.int64, copy=False)
    return labels, labels != 0


def _parent_identity(parent: dict[str, object]) -> dict[str, object]:
    return {
        key: parent[key]
        for key in (
            "support_index_sha256",
            "plane_realization_id",
            "finite_plane_geometry_sha256",
            "rendered_artifacts_sha256",
            "finite_plane_render_id",
            "finite_render_receipt_sha256",
        )
    }


def _candidate_identity(candidate: dict[str, object], shared: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "anatomy-tracker.finite-candidate-identity/v1",
        **shared,
        "candidate_class": candidate["candidate_class"],
        "slot": candidate["slot"],
        "accepted_attempt_index": candidate["accepted_attempt_index"],
        "field_stream_seed_uint64": candidate["field_stream_seed_uint64"],
        "proposal": candidate["proposal"],
        "pose_sha256": candidate["pose_sha256"],
        "effective_geometry_sha256": candidate["effective_geometry_sha256"],
        "geometry_uniqueness_sha256": candidate["geometry_uniqueness_sha256"],
        "candidate_geometry_sha256": candidate["candidate_geometry_sha256"],
        "render_schema": candidate["render_schema"],
        "render_array_receipts": candidate["render_array_receipts"],
        "rendered_artifacts_sha256": candidate["rendered_artifacts_sha256"],
        "truth_parent_binding": candidate["truth_parent_binding"],
    }


def _candidate_pose_sha256(geometry: dict[str, object]) -> str:
    return _payload_sha256(
        {
            "normal": geometry["normal_rp2_sign_aligned_ap_dv_ml"],
            "offset": geometry["signed_offset_um"],
            "roll_delta": geometry["roll_delta_rad_from_parallel_transport"],
            "center": geometry["center_ap_dv_ml_um"],
            "frame": geometry["frame_ap_dv_ml_physical"],
            "basis": geometry["inplane_basis_u_v_um"],
        }
    )


def _candidate_geometry_uniqueness_sha256(
    physical_pose: dict[str, object], geometry: dict[str, object]
) -> str:
    return _payload_sha256(
        {
            "schema": "anatomy-tracker.finite-candidate-design-effective-geometry-key/v1",
            "design_physical_pose": physical_pose,
            "effective_renderer_center_ap_dv_ml": geometry["renderer_center_ap_dv_ml"],
            "effective_renderer_frame_ap_dv_ml": geometry["renderer_frame_ap_dv_ml"],
            "effective_renderer_inplane_basis": geometry["renderer_inplane_basis"],
            "effective_physical_ouv_ap_dv_ml_um": geometry[
                "effective_physical_ouv_ap_dv_ml_um"
            ],
        }
    )


def _candidate_effective_geometry_sha256(geometry: dict[str, object]) -> str:
    return _payload_sha256(
        {
            "schema": "anatomy-tracker.finite-candidate-effective-float32-geometry-key/v1",
            "effective_renderer_center_ap_dv_ml": geometry["renderer_center_ap_dv_ml"],
            "effective_renderer_frame_ap_dv_ml": geometry["renderer_frame_ap_dv_ml"],
            "effective_renderer_inplane_basis": geometry["renderer_inplane_basis"],
            "effective_physical_ouv_ap_dv_ml_um": geometry[
                "effective_physical_ouv_ap_dv_ml_um"
            ],
        }
    )


def _sample_interval_union(intervals: np.ndarray, rng: np.random.Generator) -> tuple[float, int, float]:
    intervals = np.asarray(intervals, dtype=np.float64)
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    draw = float(rng.uniform(0.0, total))
    index = min(int(np.searchsorted(np.cumsum(lengths), draw, side="right")), len(intervals) - 1)
    before = float(lengths[:index].sum())
    return float(intervals[index, 0] + draw - before), index, draw / total


def _render_payload(labels: np.ndarray, brain_mask: np.ndarray) -> tuple[dict[str, object], str]:
    receipts = {
        "annotation": _array_receipt(labels),
        "brain_mask": _array_receipt(brain_mask),
    }
    payload = {"schema": FINITE_CANDIDATE_RENDER_SCHEMA, "arrays": receipts}
    return receipts, _payload_sha256(payload)


def _truth_copy_receipt(parent: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "anatomy-tracker.exact-finite-parent-truth-copy/v1",
        "finite_plane_geometry_sha256": parent["finite_plane_geometry_sha256"],
        "parent_geometry_sha256": parent["geometry"]["geometry_sha256"],
        "rendered_artifacts_sha256": parent["rendered_artifacts_sha256"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "raster_hashes": copy.deepcopy(parent["raster_hashes"]),
        "raster_array_receipts": copy.deepcopy(parent["raster_array_receipts"]),
    }


def _truth_candidate(
    parent: dict[str, object],
    root_seed: int,
    shared_identity: dict[str, object],
    truth_copy_receipt: dict[str, object],
) -> dict[str, object]:
    labels = np.array(parent["raster"]["annotation"], copy=True, order="K")
    brain = np.array(parent["raster"]["brain_mask"], copy=True, order="K")
    parent_state = _parent_physical_state(parent["geometry"])
    physical_pose = {
        "normal_rp2_sign_aligned_ap_dv_ml": np.asarray(parent_state["normal"]).tolist(),
        "signed_offset_um": float(parent_state["offset"]),
        "roll_delta_rad_from_parallel_transport": 0.0,
        "center_ap_dv_ml_um": np.asarray(parent_state["center"]).tolist(),
        "frame_ap_dv_ml_physical": np.asarray(parent_state["frame"]).tolist(),
        "inplane_basis_u_v_um": np.asarray(parent_state["basis"]).tolist(),
    }
    pose_sha256 = _candidate_pose_sha256(physical_pose)
    effective_geometry_sha256 = _candidate_effective_geometry_sha256(parent["geometry"])
    geometry_uniqueness_sha256 = _candidate_geometry_uniqueness_sha256(
        physical_pose, parent["geometry"]
    )
    receipts, render_sha256 = _render_payload(labels, brain)
    seed = derive_finite_candidate_seed(
        root_seed, parent["plane_realization_id"], "truth", 0, 0
    )
    candidate = {
        "candidate_class": "truth",
        "slot": 0,
        "accepted_attempt_index": 0,
        "field_stream_seed_uint64": _seed_hex(seed),
        "proposal": {
            "normal_rp2_sign_aligned_ap_dv_ml": physical_pose[
                "normal_rp2_sign_aligned_ap_dv_ml"
            ],
            "signed_offset_um": physical_pose["signed_offset_um"],
            "roll_delta_rad": 0.0,
            "source": "verified parent geometry reused byte-for-byte",
        },
        "physical_pose": physical_pose,
        "pose_sha256": pose_sha256,
        "effective_geometry_sha256": effective_geometry_sha256,
        "geometry_uniqueness_sha256": geometry_uniqueness_sha256,
        "geometry_storage": "truth_parent_geometry",
        "candidate_geometry_sha256": parent["finite_plane_geometry_sha256"],
        "render_schema": "anatomy-tracker.exact-finite-parent-truth-copy/v1",
        "rendered_annotation": labels,
        "brain_mask": brain,
        "brain_pixel_count": int(brain.sum()),
        "render_array_receipts": receipts,
        "rendered_artifacts_sha256": render_sha256,
        "truth_parent_binding": copy.deepcopy(truth_copy_receipt),
    }
    candidate["candidate_id"] = _payload_sha256(_candidate_identity(candidate, shared_identity))
    return candidate


def _decoy_candidate(
    parent: dict[str, object],
    support_index: dict[str, object],
    annotation: np.ndarray,
    candidate_class: str,
    slot: int,
    attempt: int,
    seed: int,
    normal: np.ndarray,
    offset: float,
    roll_delta: float,
    shared_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    geometry = transport_finite_candidate_pose(
        parent["geometry"], support_index, normal, offset, roll_delta
    )
    labels, brain = render_finite_candidate_annotation(annotation, geometry)
    receipts, render_sha256 = _render_payload(labels, brain)
    pose_sha256 = _candidate_pose_sha256(geometry)
    proposal = {
        "normal_rp2_sign_aligned_ap_dv_ml": geometry["normal_rp2_sign_aligned_ap_dv_ml"],
        "signed_offset_um": geometry["signed_offset_um"],
        "roll_delta_rad": geometry["roll_delta_rad_from_parallel_transport"],
    }
    candidate = {
        "candidate_class": candidate_class,
        "slot": int(slot),
        "accepted_attempt_index": int(attempt),
        "field_stream_seed_uint64": _seed_hex(seed),
        "proposal": proposal,
        "physical_pose": {
            key: geometry[key]
            for key in (
                "normal_rp2_sign_aligned_ap_dv_ml",
                "signed_offset_um",
                "roll_delta_rad_from_parallel_transport",
                "center_ap_dv_ml_um",
                "frame_ap_dv_ml_physical",
                "inplane_basis_u_v_um",
            )
        },
        "pose_sha256": pose_sha256,
        "effective_geometry_sha256": _candidate_effective_geometry_sha256(geometry),
        "geometry_uniqueness_sha256": _candidate_geometry_uniqueness_sha256(
            {
                key: geometry[key]
                for key in (
                    "normal_rp2_sign_aligned_ap_dv_ml",
                    "signed_offset_um",
                    "roll_delta_rad_from_parallel_transport",
                    "center_ap_dv_ml_um",
                    "frame_ap_dv_ml_physical",
                    "inplane_basis_u_v_um",
                )
            },
            geometry,
        ),
        "geometry_storage": "candidate",
        "geometry": geometry,
        "candidate_geometry_sha256": geometry["candidate_geometry_sha256"],
        "render_schema": FINITE_CANDIDATE_RENDER_SCHEMA,
        "rendered_annotation": labels,
        "brain_mask": brain,
        "brain_pixel_count": int(brain.sum()),
        "render_array_receipts": receipts,
        "rendered_artifacts_sha256": render_sha256,
        "truth_parent_binding": None,
    }
    candidate["candidate_id"] = _payload_sha256(_candidate_identity(candidate, shared_identity))
    attempt_record = {
        "candidate_class": candidate_class,
        "slot": int(slot),
        "attempt_index": int(attempt),
        "field_stream_seed_uint64": _seed_hex(seed),
        "proposal": proposal,
        "pose_sha256": pose_sha256,
        "effective_geometry_sha256": candidate["effective_geometry_sha256"],
        "geometry_uniqueness_sha256": candidate["geometry_uniqueness_sha256"],
        "candidate_geometry_sha256": geometry["candidate_geometry_sha256"],
        "rendered_artifacts_sha256": render_sha256,
        "brain_intersection": geometry["support_intersection"]["brain_intersection"],
        "brain_pixel_count": int(brain.sum()),
    }
    return candidate, attempt_record


def _local_proposals(parent_geometry: dict[str, object]) -> list[tuple[str, int, np.ndarray, float, float]]:
    state = _parent_physical_state(parent_geometry)
    n0 = np.asarray(state["normal"])
    d0 = float(state["offset"])
    u0, v0 = np.asarray(state["frame"])[:, :2].T
    proposals = []
    slot = 0
    for magnitude in (100.0, 250.0, 500.0):
        for sign in (-1.0, 1.0):
            proposals.append(("offset_only", slot, n0, d0 + sign * magnitude, 0.0))
            slot += 1
    slot = 0
    for axis in (u0, v0):
        for magnitude in (1.0, 3.0, 7.0, 12.0):
            for sign in (-1.0, 1.0):
                theta = math.radians(sign * magnitude)
                proposals.append(
                    ("normal_angle_only", slot, math.cos(theta) * n0 + math.sin(theta) * axis, d0, 0.0)
                )
                slot += 1
    slot = 0
    for magnitude in (3.0, 10.0, 30.0):
        for sign in (-1.0, 1.0):
            proposals.append(("roll_only", slot, n0, d0, math.radians(sign * magnitude)))
            slot += 1
    coupled = (
        (3.0, u0, 250.0, 10.0),
        (-3.0, u0, -250.0, -10.0),
        (7.0, v0, -100.0, 30.0),
        (-7.0, v0, 100.0, -30.0),
        (12.0, (u0 + v0) / math.sqrt(2.0), 500.0, 15.0),
    )
    for slot, (angle, axis, offset_delta, roll) in enumerate(coupled):
        theta = math.radians(angle)
        proposals.append(
            (
                "coupled_local",
                slot,
                math.cos(theta) * n0 + math.sin(theta) * axis,
                d0 + offset_delta,
                math.radians(roll),
            )
        )
    return proposals


def _case_rejection(reason: str, attempts: list[dict[str, object]]) -> ValueError:
    receipt = {
        "schema": "anatomy-tracker.finite-candidate-case-rejection/v1",
        "reason": reason,
        "candidate_attempts": attempts,
        "candidate_attempts_sha256": _payload_sha256(attempts),
    }
    return ValueError(f"finite candidate case rejected: {_canonical_json(receipt)}")


def _generate_bank(
    finite_parent: dict[str, object],
    annotation_ap_dv_ml: np.ndarray,
    annotation_sha256: str,
    prepared_context_sha256: str,
    support_index: dict[str, object],
    candidate_root_seed: int,
    shuffle_root_seed: int,
    *,
    finite_parent_generator_source_commit: str | None,
) -> dict[str, object]:
    verify_finite_arbitrary_plane_render(
        finite_parent,
        support_index,
        generator_source_commit=finite_parent_generator_source_commit,
    )
    annotation = np.asarray(annotation_ap_dv_ml)
    if annotation.shape != tuple(support_index["annotation_shape"]) or not np.issubdtype(
        annotation.dtype, np.integer
    ):
        raise ValueError("Annotation must match the verified support shape and integer dtype")
    if annotation_sha256 != support_index["source"]["annotation_array_sha256"]:
        raise ValueError("Annotation array does not match the verified support source")
    if annotation_sha256 != finite_parent["provenance"]["annotation_decoded"]["array_sha256"]:
        raise ValueError("Annotation array does not match the finite parent")
    candidate_root_seed = _uint64(candidate_root_seed)
    shuffle_root_seed = _uint64(shuffle_root_seed)
    if candidate_root_seed != DEFAULT_CANDIDATE_ROOT_SEED or shuffle_root_seed != DEFAULT_SHUFFLE_ROOT_SEED:
        raise ValueError("Finite candidate generation requires the two predeclared root seeds")
    parent_identity = _parent_identity(finite_parent)
    config = {
        "schema_version": FINITE_CANDIDATE_SCHEMA,
        "algorithm": FINITE_CANDIDATE_ALGORITHM,
        "candidate_root_seed": _seed_hex(candidate_root_seed),
        "shuffle_root_seed": _seed_hex(shuffle_root_seed),
        "base_plane_realization_id": finite_parent["plane_realization_id"],
        "support_index_sha256": support_index["support_index_sha256"],
        "annotation_array_sha256": annotation_sha256,
        "prepared_annotation_context_schema": FINITE_CANDIDATE_CONTEXT_SCHEMA,
        "prepared_annotation_context_sha256": prepared_context_sha256,
        "output_shape_h_w": copy.deepcopy(finite_parent["geometry"]["output_shape_h_w"]),
        "minimum_candidate_brain_pixels": MINIMUM_CANDIDATE_BRAIN_PIXELS,
        "maximum_global_attempts_per_slot": MAXIMUM_GLOBAL_ATTEMPTS,
        "ordinary_offset_margin_um": ORDINARY_OFFSET_MARGIN_UM,
        "candidate_composition": {
            "truth": 1,
            "offset_only": 6,
            "normal_angle_only": 16,
            "roll_only": 6,
            "coupled_local": 5,
            "global_hard_negative": 6,
        },
        "sampling_contract": QUICKNII_RASTER_INDEX_SAMPLING,
        "target_overlap_used_for_acceptance": False,
    }
    implementation = {
        "source_path": "training/arbitrary_plane_finite_candidates.py",
        "loaded_source_sha256": copy.deepcopy(_LOADED_SOURCE_SHA256),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    model_independence = {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "initialization": "deterministic random candidate and case seed streams only",
    }
    generator = {
        "implementation": implementation,
        "implementation_sha256": _payload_sha256(implementation),
        "resolved_config": config,
        "resolved_config_sha256": _payload_sha256(config),
        **model_independence,
        "model_independence_sha256": _payload_sha256(model_independence),
    }
    shared_identity = {
        "parent_identity": parent_identity,
        "support_index_sha256": support_index["support_index_sha256"],
        "annotation_array_sha256": config["annotation_array_sha256"],
        "implementation_sha256": generator["implementation_sha256"],
        "resolved_config_sha256": generator["resolved_config_sha256"],
    }
    truth_copy_receipt = _truth_copy_receipt(finite_parent)
    truth_copy_receipt_sha256 = _payload_sha256(truth_copy_receipt)
    candidates = [
        _truth_candidate(finite_parent, candidate_root_seed, shared_identity, truth_copy_receipt)
    ]
    attempts: list[dict[str, object]] = [
        {
            "candidate_class": "truth",
            "slot": 0,
            "attempt_index": 0,
            "field_stream_seed_uint64": candidates[0]["field_stream_seed_uint64"],
            "pose_sha256": candidates[0]["pose_sha256"],
            "effective_geometry_sha256": candidates[0]["effective_geometry_sha256"],
            "geometry_uniqueness_sha256": candidates[0]["geometry_uniqueness_sha256"],
            "candidate_geometry_sha256": candidates[0]["candidate_geometry_sha256"],
            "rendered_artifacts_sha256": candidates[0]["rendered_artifacts_sha256"],
            "brain_intersection": True,
            "brain_pixel_count": candidates[0]["brain_pixel_count"],
            "accepted": candidates[0]["brain_pixel_count"] >= MINIMUM_CANDIDATE_BRAIN_PIXELS,
            "reason": "accepted" if candidates[0]["brain_pixel_count"] >= MINIMUM_CANDIDATE_BRAIN_PIXELS else "fewer-than-64-candidate-brain-pixels",
        }
    ]
    if not attempts[0]["accepted"]:
        raise _case_rejection("truth has fewer than 64 candidate-brain pixels", attempts)
    parent_state = _parent_physical_state(finite_parent["geometry"])
    truth_intervals = _component_interval_union_trusted(
        np.asarray(parent_state["normal"]), support_index
    )
    d0 = float(parent_state["offset"])
    containing = truth_intervals[(d0 >= truth_intervals[:, 0]) & (d0 <= truth_intervals[:, 1])]
    if len(containing) != 1 or min(d0 - containing[0, 0], containing[0, 1] - d0) < ORDINARY_OFFSET_MARGIN_UM:
        raise _case_rejection("truth offset lacks the predeclared 550-um ordinary margin", attempts)

    pose_ids = {candidates[0]["pose_sha256"]}
    effective_geometry_ids = {candidates[0]["effective_geometry_sha256"]}
    geometry_keys = {candidates[0]["geometry_uniqueness_sha256"]}
    for candidate_class, slot, normal, offset, roll_delta in _local_proposals(
        finite_parent["geometry"]
    ):
        seed = derive_finite_candidate_seed(
            candidate_root_seed,
            finite_parent["plane_realization_id"],
            candidate_class,
            slot,
            0,
        )
        candidate, record = _decoy_candidate(
            finite_parent,
            support_index,
            annotation,
            candidate_class,
            slot,
            0,
            seed,
            normal,
            offset,
            roll_delta,
            shared_identity,
        )
        reason = "accepted"
        if not record["brain_intersection"]:
            reason = "nonintersecting-candidate-plane"
        elif record["brain_pixel_count"] < MINIMUM_CANDIDATE_BRAIN_PIXELS:
            reason = "fewer-than-64-candidate-brain-pixels"
        elif record["pose_sha256"] in pose_ids or record["effective_geometry_sha256"] in effective_geometry_ids:
            reason = "duplicate-design-or-effective-candidate-geometry"
        record.update(accepted=reason == "accepted", reason=reason)
        attempts.append(record)
        if reason != "accepted":
            raise _case_rejection(f"fixed local slot {candidate_class}/{slot} is invalid", attempts)
        candidates.append(candidate)
        pose_ids.add(record["pose_sha256"])
        effective_geometry_ids.add(record["effective_geometry_sha256"])
        geometry_keys.add(record["geometry_uniqueness_sha256"])

    n0 = np.asarray(parent_state["normal"])
    for slot in range(6):
        accepted = False
        for attempt in range(MAXIMUM_GLOBAL_ATTEMPTS):
            seed = derive_finite_candidate_seed(
                candidate_root_seed,
                finite_parent["plane_realization_id"],
                "global_hard_negative",
                slot,
                attempt,
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            raw = rng.normal(size=3)
            normal, _ = align_rp2_pose_to_reference(raw, 0.0, n0)
            angle_deg = math.degrees(math.acos(float(np.clip(normal @ n0, -1.0, 1.0))))
            if not 20.0 <= angle_deg <= 60.0:
                attempts.append(
                    {
                        "candidate_class": "global_hard_negative",
                        "slot": slot,
                        "attempt_index": attempt,
                        "field_stream_seed_uint64": _seed_hex(seed),
                        "sampled_normal_angle_deg": angle_deg,
                        "accepted": False,
                        "reason": "normal-angle-outside-20-to-60-deg",
                    }
                )
                continue
            intervals = _component_interval_union_trusted(normal, support_index)
            offset, interval_index, fraction = _sample_interval_union(intervals, rng)
            roll_delta = float(rng.uniform(-math.pi, math.pi))
            candidate, record = _decoy_candidate(
                finite_parent,
                support_index,
                annotation,
                "global_hard_negative",
                slot,
                attempt,
                seed,
                normal,
                offset,
                roll_delta,
                shared_identity,
            )
            record.update(
                sampled_normal_angle_deg=angle_deg,
                selected_interval_index=interval_index,
                offset_measure_fraction=fraction,
            )
            reason = "accepted"
            if not record["brain_intersection"]:
                reason = "nonintersecting-candidate-plane"
            elif record["brain_pixel_count"] < MINIMUM_CANDIDATE_BRAIN_PIXELS:
                reason = "fewer-than-64-candidate-brain-pixels"
            elif record["pose_sha256"] in pose_ids or record["effective_geometry_sha256"] in effective_geometry_ids:
                reason = "duplicate-design-or-effective-candidate-geometry"
            record.update(accepted=reason == "accepted", reason=reason)
            attempts.append(record)
            if reason == "accepted":
                candidates.append(candidate)
                pose_ids.add(record["pose_sha256"])
                effective_geometry_ids.add(record["effective_geometry_sha256"])
                geometry_keys.add(record["geometry_uniqueness_sha256"])
                accepted = True
                break
        if not accepted:
            raise _case_rejection(f"global hard-negative slot {slot} exhausted", attempts)

    if (
        len(candidates) != 40
        or len({item["candidate_id"] for item in candidates}) != 40
        or len(pose_ids) != 40
        or len(effective_geometry_ids) != 40
        or len(geometry_keys) != 40
    ):
        raise _case_rejection("candidate bank is not exactly 40 unique identities", attempts)
    canonical_candidate_ids = [item["candidate_id"] for item in candidates]
    shuffle_seed = derive_finite_candidate_seed(
        shuffle_root_seed,
        finite_parent["plane_realization_id"],
        "final-order",
        "bank",
        0,
    )
    order = np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(40).tolist()
    candidates = [candidates[index] for index in order]
    ordered_candidate_ids = [item["candidate_id"] for item in candidates]
    candidate_attempts_sha256 = _payload_sha256(attempts)
    candidate_set_identity = {
        "schema": "anatomy-tracker.finite-candidate-set/v1",
        **shared_identity,
        "candidate_attempts_sha256": candidate_attempts_sha256,
        "canonical_candidate_ids": canonical_candidate_ids,
    }
    candidate_set_id = _payload_sha256(candidate_set_identity)
    bank_identity = {
        "schema_version": FINITE_CANDIDATE_SCHEMA,
        "algorithm": FINITE_CANDIDATE_ALGORITHM,
        "candidate_set_id": candidate_set_id,
        "shuffle_field_stream_seed_uint64": _seed_hex(shuffle_seed),
        "ordered_candidate_ids": ordered_candidate_ids,
        "truth_copy_receipt_sha256": truth_copy_receipt_sha256,
        "provenance_sha256": finite_parent["provenance_sha256"],
        "model_independence_sha256": generator["model_independence_sha256"],
    }
    artifact = {
        "schema_version": FINITE_CANDIDATE_SCHEMA,
        "generator_algorithm": FINITE_CANDIDATE_ALGORITHM,
        "finite_parent_identity": parent_identity,
        "finite_parent_receipt": finite_render_receipt(finite_parent),
        "truth_parent_geometry": copy.deepcopy(finite_parent["geometry"]),
        "truth_parent_raster": {
            key: np.array(value, copy=True, order="K") for key, value in finite_parent["raster"].items()
        },
        "truth_copy_receipt": truth_copy_receipt,
        "truth_copy_receipt_sha256": truth_copy_receipt_sha256,
        "support_index_sha256": support_index["support_index_sha256"],
        "provenance": copy.deepcopy(finite_parent["provenance"]),
        "provenance_sha256": finite_parent["provenance_sha256"],
        "generator": generator,
        "candidate_attempts": attempts,
        "candidate_attempts_sha256": candidate_attempts_sha256,
        "canonical_candidate_ids": canonical_candidate_ids,
        "shuffle_field_stream_seed_uint64": _seed_hex(shuffle_seed),
        "final_order_canonical_indices": order,
        "ordered_candidate_ids": ordered_candidate_ids,
        "candidates": candidates,
        "candidate_set_id": candidate_set_id,
        "acceptance_contract": {
            "predicate": "candidate brain intersection and candidate brain_pixel_count >= 64",
            "fixed_canvas": True,
            "candidate_target_overlap_used": False,
            "target_or_target_mask_argument": None,
            "local_invalid_policy": "reject whole base case without tuning or substitution",
            "global_invalid_policy": "deterministic per-slot resampling up to 4096 attempts",
        },
        "development_scope": {
            "status": "finite semantic-oracle candidate representation only",
            "model_training": False,
            "benchmark": False,
            "qualification": False,
            "final_test_access": False,
        },
    }
    artifact["finite_candidate_bank_id"] = _payload_sha256(bank_identity)
    artifact["finite_candidate_receipt_sha256"] = _payload_sha256(
        finite_candidate_bank_receipt(artifact)
    )
    return artifact


def make_arbitrary_plane_finite_candidate_bank_from_context(
    finite_parent: dict[str, object],
    prepared_context: MappingProxyType,
    support_index: dict[str, object],
    *,
    candidate_root_seed: int | str = DEFAULT_CANDIDATE_ROOT_SEED,
    shuffle_root_seed: int | str = DEFAULT_SHUFFLE_ROOT_SEED,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Create one bank without re-hashing or expanding the prepared atlas."""
    annotation, annotation_sha256, context_sha256 = _validate_prepared_annotation_context(
        prepared_context, support_index
    )
    return _generate_bank(
        finite_parent,
        annotation,
        annotation_sha256,
        context_sha256,
        support_index,
        _uint64(candidate_root_seed),
        _uint64(shuffle_root_seed),
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def make_arbitrary_plane_finite_candidate_bank(
    finite_parent: dict[str, object],
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    candidate_root_seed: int | str = DEFAULT_CANDIDATE_ROOT_SEED,
    shuffle_root_seed: int | str = DEFAULT_SHUFFLE_ROOT_SEED,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """One-shot wrapper; repeated runs should prepare one context explicitly."""
    context = prepare_arbitrary_plane_finite_candidate_context(annotation_ap_dv_ml, support_index)
    return make_arbitrary_plane_finite_candidate_bank_from_context(
        finite_parent,
        context,
        support_index,
        candidate_root_seed=candidate_root_seed,
        shuffle_root_seed=shuffle_root_seed,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def finite_candidate_bank_receipt(artifact: dict[str, object]) -> dict[str, object]:
    """Return the JSON-safe receipt while retaining every array hash and pose."""
    receipt = {
        key: copy.deepcopy(value)
        for key, value in artifact.items()
        if key not in {"truth_parent_raster", "candidates", "finite_candidate_receipt_sha256"}
    }
    receipt["candidates"] = [
        {
            key: copy.deepcopy(value)
            for key, value in candidate.items()
            if key not in {"rendered_annotation", "brain_mask"}
        }
        for candidate in artifact["candidates"]
    ]
    json.dumps(receipt, allow_nan=False, default=_json_scalar)
    return receipt


def _arrays_byte_equal(first: np.ndarray, second: np.ndarray) -> bool:
    left, right = np.asarray(first), np.asarray(second)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="A") == right.tobytes(order="A")


def _verified_replay_from_context(
    artifact: dict[str, object],
    finite_parent: dict[str, object],
    prepared_context: MappingProxyType,
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None,
) -> dict[str, object]:
    required = {
        "schema_version",
        "generator_algorithm",
        "truth_parent_geometry",
        "truth_parent_raster",
        "candidates",
        "generator",
        "finite_candidate_bank_id",
        "finite_candidate_receipt_sha256",
    }
    if not required <= artifact.keys():
        raise ValueError("Finite candidate artifact is incomplete")
    annotation, annotation_sha256, context_sha256 = _validate_prepared_annotation_context(
        prepared_context, support_index
    )
    config = artifact["generator"]["resolved_config"]
    replayed = _generate_bank(
        finite_parent,
        annotation,
        annotation_sha256,
        context_sha256,
        support_index,
        _uint64(config["candidate_root_seed"]),
        _uint64(config["shuffle_root_seed"]),
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )
    if finite_candidate_bank_receipt(artifact) != finite_candidate_bank_receipt(replayed):
        raise ValueError("Finite candidate receipt or hash-bound metadata does not replay exactly")
    if artifact["finite_candidate_receipt_sha256"] != _payload_sha256(
        finite_candidate_bank_receipt(artifact)
    ):
        raise ValueError("Finite candidate receipt SHA-256 does not match")
    if artifact["truth_parent_geometry"] != finite_parent["geometry"]:
        raise ValueError("Truth parent geometry was not reused byte-for-byte")
    if artifact["truth_parent_raster"].keys() != finite_parent["raster"].keys():
        raise ValueError("Truth parent raster keys do not match")
    for key in finite_parent["raster"]:
        if not _arrays_byte_equal(artifact["truth_parent_raster"][key], finite_parent["raster"][key]):
            raise ValueError("Truth parent raster was not reused byte-for-byte")
    for actual, expected in zip(artifact["candidates"], replayed["candidates"], strict=True):
        for key in ("rendered_annotation", "brain_mask"):
            if not _arrays_byte_equal(actual[key], expected[key]):
                raise ValueError("Candidate label render does not replay byte-for-byte")
    truth = next(item for item in artifact["candidates"] if item["candidate_class"] == "truth")
    for key, parent_key in (("rendered_annotation", "annotation"), ("brain_mask", "brain_mask")):
        if not _arrays_byte_equal(truth[key], finite_parent["raster"][parent_key]):
            raise ValueError("Truth candidate does not expose the exact parent raster")
    return replayed


def verify_arbitrary_plane_finite_candidate_bank_from_context(
    artifact: dict[str, object],
    finite_parent: dict[str, object],
    prepared_context: MappingProxyType,
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> None:
    """Verify a bank without re-hashing the decoded atlas annotation."""
    _verified_replay_from_context(
        artifact,
        finite_parent,
        prepared_context,
        support_index,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def replay_arbitrary_plane_finite_candidate_bank_from_context(
    artifact: dict[str, object],
    finite_parent: dict[str, object],
    prepared_context: MappingProxyType,
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Replay a bank without re-hashing or batch-expanding the atlas."""
    return _verified_replay_from_context(
        artifact,
        finite_parent,
        prepared_context,
        support_index,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def verify_arbitrary_plane_finite_candidate_bank(
    artifact: dict[str, object],
    finite_parent: dict[str, object],
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> None:
    """Verify identities, exact truth reuse, and deterministic candidate replay."""
    context = prepare_arbitrary_plane_finite_candidate_context(annotation_ap_dv_ml, support_index)
    _verified_replay_from_context(
        artifact,
        finite_parent,
        context,
        support_index,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def replay_arbitrary_plane_finite_candidate_bank(
    artifact: dict[str, object],
    finite_parent: dict[str, object],
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Return the exact regenerated bank after first verifying the supplied artifact."""
    context = prepare_arbitrary_plane_finite_candidate_context(annotation_ap_dv_ml, support_index)
    return _verified_replay_from_context(
        artifact,
        finite_parent,
        context,
        support_index,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )
