"""Provenance-bound v2 arbitrary-plane acquisition primitives."""

from __future__ import annotations

import hashlib
import json
import math
import platform
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
    render_arbitrary_plane,
)
from training.arbitrary_plane_manifest import canonicalize_plane
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
    physical_plane_frame,
    prepare_finite_render_context,
)
from training.arbitrary_plane_support import (
    plane_interval_membership_certificate,
    support_projection_bounds,
    verify_annotation_support_index,
)


V2_SCHEMA = "anatomy-tracker.arbitrary-plane-synthetic-realization/v2"
V2_PLANE_SCHEMA = "anatomy-tracker.arbitrary-plane-global-reference-centre-render/v2"
V2_PLANE_ALGORITHM = "rejection-free-rp2-global-reference-centre-render/v2"
V2_GENERIC_PLANE_SCHEMA = (
    "anatomy-tracker.authenticated-generic-arbitrary-plane-centre-render/v2"
)
V2_GENERIC_PLANE_ALGORITHM = (
    "domain-separated-rp2-global-reference-centre-render/v2"
)
V2_RNG_DOMAIN = "anatomy-tracker.arbitrary-plane-acquisition-v2"
V2_PLANE_STRATA = ("near_AP", "near_DV", "near_ML", "general_oblique", "edge_or_partial")
V2_GENERIC_PLANE_STRATA = (
    "reference",
    "near_AP",
    "near_DV",
    "near_ML",
    "general_oblique",
    "edge_or_partial",
)
V2_SOURCE_SHA256_CANONICALIZATION = "CRLF and CR normalized to LF before SHA-256"
V2_PREFLIGHT_CANONICAL_SHA256 = "0d06c26e1eb793b66da437db33933a09d304d2989b6383df8846106057da9ad9"
_SOURCE_ROOT = Path(__file__).parent
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
_PREFLIGHT_PATH = _REPOSITORY_ROOT / "publication" / "arbitrary_plane_acquisition_hardening_preflight.yaml"
_V2_CONTEXT_TOKEN = object()
_SMOKE_ROOT_SEED = "0x415154564f320001"
_SMOKE_PARENT_SHAPE = (256, 256)
V2_SMOKE_ASSIGNMENTS = (
    ("near_AP", "standard", "none", "centre_plane_ablation", 50.0, None, 0.0),
    ("near_AP", "standard", "horizontal", "finite_boxcar", 25.0, None, None),
    ("near_AP", "standard", "none", "finite_boxcar", 50.0, None, None),
    ("near_AP", "broad", "horizontal", "finite_boxcar", 60.0, None, None),
    ("near_DV", "stress", "none", "finite_boxcar", 100.0, "named_thick_stress", None),
    ("near_DV", "standard", "horizontal", "centre_plane_ablation", 60.0, None, 0.0),
    ("near_DV", "standard", "none", "finite_boxcar", 30.0, None, None),
    ("near_DV", "standard", "horizontal", "finite_boxcar", 55.0, None, None),
    ("near_ML", "broad", "none", "finite_boxcar", 85.0, None, None),
    ("near_ML", "standard", "horizontal", "finite_boxcar", 32.5, None, None),
    ("near_ML", "standard", "none", "centre_plane_ablation", 25.0, None, 0.0),
    ("near_ML", "standard", "horizontal", "finite_boxcar", 45.0, None, None),
    ("general_oblique", "standard", "none", "finite_boxcar", 64.0, None, None),
    ("general_oblique", "broad", "horizontal", "finite_boxcar", 75.0, None, None),
    ("general_oblique", "standard", "none", "finite_boxcar", 99.0, None, None),
    ("general_oblique", "standard", "horizontal", "finite_boxcar", 27.5, None, None),
    ("edge_or_partial", "standard", "none", "finite_boxcar", 52.5, None, None),
    ("edge_or_partial", "standard", "horizontal", "finite_boxcar", 87.5, None, None),
    ("edge_or_partial", "broad", "none", "finite_boxcar", 47.5, None, None),
    ("edge_or_partial", "stress", "horizontal", "finite_boxcar", 62.5, None, None),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    dtype = values.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(values.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": dtype.str, "shape": list(values.shape)}).encode("utf-8"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    return {"dtype": values.dtype.str, "shape": list(values.shape), "array_sha256": _array_sha256(values)}


def _plain_value(value: object) -> object:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen
    return value


def _json_value(value: object) -> object:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _merge_intervals(intervals: np.ndarray) -> np.ndarray:
    ordered = np.asarray(intervals, dtype=np.float64)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
    merged = [ordered[0].tolist()]
    for lower, upper in ordered[1:]:
        if lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(upper))
        else:
            merged.append([float(lower), float(upper)])
    return np.asarray(merged, dtype=np.float64)


def _normalized_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        name: _normalized_text_sha256(_SOURCE_ROOT / name)
        for name in (
            "arbitrary_plane_acquisition_v2.py",
            "arbitrary_plane_geometry.py",
            "arbitrary_plane_manifest.py",
            "arbitrary_plane_rendered_generator.py",
            "arbitrary_plane_support.py",
        )
    }


def _preflight_provenance() -> dict[str, str]:
    file_sha256 = _normalized_text_sha256(_PREFLIGHT_PATH)
    if file_sha256 != V2_PREFLIGHT_CANONICAL_SHA256:
        raise ValueError("v2 acquisition hardening preflight content does not match")
    return {
        "receipt_id": "arbitrary-plane-acquisition-hardening-2026-09-01",
        "path": "publication/arbitrary_plane_acquisition_hardening_preflight.yaml",
        "file_sha256": file_sha256,
        "file_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
        "preflight_commit": "cd51b9d9ba9e8843d5a0d9f6405da676f73a47d5",
    }


def _runtime_receipt() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_device": "cpu",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
    }


def prepare_arbitrary_plane_acquisition_context_v2(
    scalar_volume_ap_dv_ml: np.ndarray,
    annotation_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
    *,
    scalar_source_uri: str,
    scalar_source_sha256: str,
    scalar_source_entity_type: str = "atlas-template",
    template_decoder: str = "caller-supplied decoded array",
    template_index_order: str | None = None,
    annotation_decoder: str = "caller-supplied decoded array",
    annotation_index_order: str | None = None,
) -> MappingProxyType:
    """Authenticate immutable atlas assets once and wrap them in a v2 context."""
    parent = prepare_finite_render_context(
        scalar_volume_ap_dv_ml,
        annotation_ap_dv_ml,
        support_index,
        scalar_source_uri=scalar_source_uri,
        scalar_source_sha256=scalar_source_sha256,
        scalar_source_entity_type=scalar_source_entity_type,
        template_decoder=template_decoder,
        template_index_order=template_index_order,
        annotation_decoder=annotation_decoder,
        annotation_index_order=annotation_index_order,
    )
    support = _plain_value(parent["support_index"])
    verify_annotation_support_index(support)
    fov = global_reference_support_geometry(support)
    receipt = {
        "schema": "anatomy-tracker.prepared-arbitrary-plane-acquisition-context/v2",
        "opaque_v1_prepared_context_sha256": parent["prepared_context_sha256"],
        "support_index_sha256": support["support_index_sha256"],
        "annotation_array_sha256": parent["asset_receipt"]["annotation_decoded"]["array_sha256"],
        "scalar_array_sha256": parent["asset_receipt"]["scalar_conversion"]["array_sha256"],
        "scalar_source": dict(parent["asset_receipt"]["scalar_source"]),
        "global_reference_fov": fov,
        "source_sha256": _source_hashes(),
        "source_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
    }
    frozen_receipt = _freeze_value(receipt)
    return MappingProxyType(
        {
            "schema": receipt["schema"],
            "_token": _V2_CONTEXT_TOKEN,
            "receipt": frozen_receipt,
            "v2_context_sha256": _payload_sha256(receipt),
            "opaque_v1_context": parent,
        }
    )


def _validate_v2_context(context: dict[str, object]) -> None:
    parent = context["opaque_v1_context"]
    receipt = context["receipt"]
    support = _plain_value(parent["support_index"])
    verify_annotation_support_index(support)
    if (
        not isinstance(context, MappingProxyType)
        or context.get("_token") is not _V2_CONTEXT_TOKEN
        or context.get("schema") != "anatomy-tracker.prepared-arbitrary-plane-acquisition-context/v2"
        or context.get("v2_context_sha256") != _payload_sha256(_json_value(receipt))
        or _json_value(receipt["source_sha256"]) != _source_hashes()
        or receipt.get("source_sha256_canonicalization")
        != V2_SOURCE_SHA256_CANONICALIZATION
        or parent["prepared_context_sha256"] != receipt["opaque_v1_prepared_context_sha256"]
        or support["support_index_sha256"] != receipt["support_index_sha256"]
    ):
        raise ValueError("prepared v2 acquisition context receipt does not match")
    scalar = parent["scalar_tensor"]
    annotation = parent["annotation_tensor"]
    if (
        id(scalar) != parent["scalar_tensor_id"]
        or id(annotation) != parent["annotation_tensor_id"]
        or scalar._version != parent["scalar_tensor_version"]
        or annotation._version != parent["annotation_tensor_version"]
        or scalar.device.type != "cpu"
        or annotation.device.type != "cpu"
        or scalar.dtype != torch.float32
        or list(scalar.shape) != list(parent["asset_receipt"]["scalar_conversion"]["shape"])
        or list(annotation.shape) != list(parent["asset_receipt"]["annotation_decoded"]["shape"])
        or _array_sha256(scalar.detach().numpy()) != receipt["scalar_array_sha256"]
        or _array_sha256(annotation.detach().numpy()) != receipt["annotation_array_sha256"]
    ):
        raise ValueError("prepared v2 acquisition tensors changed")


def _context_support(context: dict[str, object]) -> dict[str, object]:
    return _plain_value(context["opaque_v1_context"]["support_index"])


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be a nonnegative uint64 integer")
    return result


def _parent_shape(parent_shape_h_w: object) -> tuple[int, int]:
    if not isinstance(parent_shape_h_w, (tuple, list, np.ndarray)) or len(parent_shape_h_w) != 2:
        raise ValueError("parent_shape_h_w must contain two integers")
    return tuple(
        _nonnegative_integer(value, "parent_shape_h_w dimension")
        for value in parent_shape_h_w
    )


def _root_seed_hex(root_seed: int | str) -> str:
    if isinstance(root_seed, str):
        if len(root_seed) != 18 or not root_seed.startswith("0x"):
            raise ValueError("root_seed string must be 0x plus 16 lowercase hexadecimal digits")
        digits = root_seed[2:]
        if any(character not in "0123456789abcdef" for character in digits):
            raise ValueError("root_seed string must be 0x plus 16 lowercase hexadecimal digits")
        return digits
    if isinstance(root_seed, (bool, np.bool_)) or not isinstance(
        root_seed, (int, np.integer)
    ):
        raise ValueError("root_seed must be a uint64 integer or canonical hex string")
    value = int(root_seed)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("root_seed must fit uint64")
    return f"{value:016x}"


def derive_v2_field_seed(
    root_seed: int | str,
    split: str,
    sample_index: int,
    stage: str,
    field: str,
    attempt: int = 0,
) -> int:
    """Derive one replayable PCG64DXSM uint64 seed from the frozen v2 tuple."""
    if not isinstance(split, str) or not split:
        raise ValueError("split must be nonempty")
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    attempt = _nonnegative_integer(attempt, "attempt")
    if not isinstance(stage, str) or not stage or not isinstance(field, str) or not field:
        raise ValueError("sample_index/attempt must be nonnegative and stage/field nonempty")
    components = (
        V2_RNG_DOMAIN,
        V2_SCHEMA,
        split,
        _root_seed_hex(root_seed),
        str(sample_index),
        stage,
        field,
        str(attempt),
    )
    encoded = b"".join(len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8") for value in components)
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-ACQ-V2").digest(), "big"
    )


def _field_rng(
    root_seed: int | str,
    split: str,
    sample_index: int,
    stage: str,
    field: str,
) -> tuple[np.random.Generator, str]:
    seed = derive_v2_field_seed(root_seed, split, sample_index, stage, field, 0)
    return np.random.Generator(np.random.PCG64DXSM(seed)), f"0x{seed:016x}"


def _smoke_assignment(sample_index: int) -> dict[str, object]:
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    row = V2_SMOKE_ASSIGNMENTS[sample_index]
    return {
        "sample_index": sample_index,
        "plane_stratum": row[0],
        "window_plan_severity": row[1],
        "reflection": row[2],
        "render_mode": row[3],
        "nominal_cut_thickness_um": row[4],
        "thickness_class": row[5],
        "effective_optical_support_um": row[6],
    }


def global_reference_support_geometry(
    support_index: dict[str, object], parent_shape_h_w: tuple[int, int] = _SMOKE_PARENT_SHAPE
) -> dict[str, object]:
    """Return the orientation-independent closed-face support sphere and FOV."""
    verify_annotation_support_index(support_index)
    height, width = _parent_shape(parent_shape_h_w)
    if (height, width) != _SMOKE_PARENT_SHAPE:
        raise ValueError("v2 acquisition-core parent shape is frozen at 256x256")
    boxes = np.asarray(
        [component["bounding_box_index_inclusive"] for component in support_index["components"]],
        dtype=np.int64,
    )
    minimum_index = boxes[:, 0].min(axis=0)
    maximum_index = boxes[:, 1].max(axis=0)
    origin = np.asarray(support_index["origin_um"], dtype=np.float64)
    spacing = np.asarray(support_index["voxel_size_um"], dtype=np.float64)
    lower = origin + minimum_index * spacing
    upper = origin + (maximum_index + 1) * spacing
    support_origin = (lower + upper) / 2.0
    corners = np.stack(
        np.meshgrid(*[(lower[axis], upper[axis]) for axis in range(3)], indexing="ij"), -1
    ).reshape(-1, 3)
    radius = float(np.linalg.norm(corners - support_origin, axis=1).max())
    margin = float(np.linalg.norm(spacing))
    diameter = 2.0 * (radius + margin)
    payload = {
        "support_index_sha256": support_index["support_index_sha256"],
        "minimum_occupied_index_inclusive": minimum_index.tolist(),
        "maximum_occupied_index_inclusive": maximum_index.tolist(),
        "closed_face_lower_ap_dv_ml_um": lower.tolist(),
        "closed_face_upper_ap_dv_ml_um": upper.tolist(),
        "support_origin_ap_dv_ml_um": support_origin.tolist(),
        "occupied_corner_radius_um": radius,
        "margin_um": margin,
        "diameter_um": diameter,
        "parent_shape_h_w": [height, width],
        "raster_sampling_contract": QUICKNII_RASTER_INDEX_SAMPLING,
    }
    return {**payload, "global_reference_fov_id": _payload_sha256(payload)}


def shifted_component_interval_union(
    normal_ap_dv_ml: np.ndarray,
    support_index: dict[str, object],
) -> dict[str, object]:
    """Express frozen projection-origin intervals about the v2 support origin."""
    projection = support_projection_bounds(np.asarray(normal_ap_dv_ml, dtype=np.float64), support_index)
    normal = np.asarray(projection["normal_rp2"], dtype=np.float64)
    base_components = np.asarray(projection["component_bounds_um"], dtype=np.float64)
    projection_origin = np.asarray(support_index["projection_origin_um"], dtype=np.float64)
    support_origin = np.asarray(
        global_reference_support_geometry(support_index)["support_origin_ap_dv_ml_um"],
        dtype=np.float64,
    )
    shift = float(normal @ (projection_origin - support_origin))
    shifted_components = np.asarray(base_components + shift, dtype=np.float64)
    shifted = _merge_intervals(shifted_components)
    payload = {
        "support_index_sha256": support_index["support_index_sha256"],
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "projection_origin_ap_dv_ml_um": projection_origin.tolist(),
        "support_origin_ap_dv_ml_um": support_origin.tolist(),
        "projection_to_support_origin_shift_um": shift,
        "projection_origin_component_bounds_um": base_components.tolist(),
        "projection_origin_component_bounds_receipt": _array_receipt(base_components),
        "support_origin_component_bounds_um": shifted_components.tolist(),
        "support_origin_component_bounds_receipt": _array_receipt(shifted_components),
        "support_origin_interval_union_um": shifted.tolist(),
        "support_origin_interval_array_receipt": _array_receipt(shifted),
    }
    return {**payload, "shifted_interval_receipt_sha256": _payload_sha256(payload)}


def _offset_at_measure_fraction(intervals_um: np.ndarray, fraction: float) -> tuple[float, int]:
    intervals = np.asarray(intervals_um, dtype=np.float64)
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    if intervals.ndim != 2 or intervals.shape[1] != 2 or total <= 0.0 or not 0.0 <= fraction < 1.0:
        raise ValueError("intervals and measure fraction are invalid")
    draw = float(fraction * total)
    index = min(int(np.searchsorted(np.cumsum(lengths), draw, side="right")), len(intervals) - 1)
    return float(intervals[index, 0] + draw - lengths[:index].sum()), index


def sample_v2_smoke_plane_pose(
    support_index: dict[str, object],
    split: str,
    root_seed: int | str,
    sample_index: int,
    plane_stratum: str,
    parent_shape_h_w: tuple[int, int] = (256, 256),
) -> dict[str, object]:
    """Sample one of the twenty predeclared development-smoke planes."""
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    assignment = V2_SMOKE_ASSIGNMENTS[sample_index] if 0 <= sample_index < 20 else None
    expected = assignment[0] if assignment is not None else None
    if (
        split != "development"
        or _root_seed_hex(root_seed) != _SMOKE_ROOT_SEED[2:]
        or plane_stratum != expected
    ):
        raise ValueError("v2 smoke requires development indices 0..19 in four-case stratum blocks")
    fov = global_reference_support_geometry(support_index, parent_shape_h_w)
    seeds: dict[str, str] = {}

    def uniform(field: str, low: float = 0.0, high: float = 1.0) -> float:
        rng, seed = _field_rng(root_seed, split, sample_index, "pose", field)
        seeds[field] = seed
        return float(rng.uniform(low, high))

    if plane_stratum.startswith("near_"):
        axis = {"near_AP": 0, "near_DV": 1, "near_ML": 2}[plane_stratum]
        cosine = uniform("axis-cosine", 0.90, 0.985)
        azimuth = 2.0 * math.pi * uniform("axis-azimuth")
        other = [value for value in range(3) if value != axis]
        raw_normal = np.zeros(3, dtype=np.float64)
        raw_normal[axis] = cosine
        radius = math.sqrt(1.0 - cosine * cosine)
        raw_normal[other] = radius * np.asarray((math.cos(azimuth), math.sin(azimuth)))
    elif plane_stratum == "general_oblique":
        magnitudes = np.asarray(
            [uniform(f"oblique-magnitude-{axis}", 0.35, 0.75) for axis in range(3)]
        )
        signs = np.asarray(
            [1.0, -1.0 if uniform("oblique-sign-1") < 0.5 else 1.0,
             -1.0 if uniform("oblique-sign-2") < 0.5 else 1.0]
        )
        raw_normal = magnitudes * signs
    else:
        z = 2.0 * uniform("edge-sphere-z") - 1.0
        azimuth = 2.0 * math.pi * uniform("edge-sphere-azimuth")
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        raw_normal = np.asarray((z, radius * math.cos(azimuth), radius * math.sin(azimuth)))
    normal, _, _ = canonicalize_plane(raw_normal, 0.0)
    roll = 2.0 * math.pi * uniform("roll")
    intervals = shifted_component_interval_union(normal, support_index)
    if plane_stratum == "edge_or_partial":
        draw = uniform("edge-offset-fraction")
        measure_fraction = 0.01 + 0.02 * draw if int(sample_index) % 2 == 0 else 0.99 - 0.02 * draw
    else:
        measure_fraction = 0.15 + 0.70 * uniform("offset-fraction")
    signed_offset, selected_interval = _offset_at_measure_fraction(
        np.asarray(intervals["support_origin_interval_union_um"]), measure_fraction
    )
    frame = physical_plane_frame(normal, roll)
    payload = {
        "plane_stratum": plane_stratum,
        "normal_draw_ap_dv_ml": raw_normal.tolist(),
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "roll_rad": roll,
        "signed_offset_um_about_support_origin": signed_offset,
        "offset_measure_fraction": measure_fraction,
        "selected_interval_index": selected_interval,
        "shifted_intervals": intervals,
        "frame_ap_dv_ml_physical": frame.tolist(),
        "field_stream_seed_uint64": seeds,
        "field_stream_stage": "pose",
        "field_stream_attempt_index": {field: 0 for field in seeds},
        "rejection_attempts": [],
    }
    return {**payload, "plane_sampler_receipt_sha256": _payload_sha256(payload)}


def sample_v2_generic_plane_pose(
    support_index: dict[str, object],
    split: str,
    root_seed: int | str,
    sample_index: int,
    plane_stratum: str,
    parent_shape_h_w: tuple[int, int] = (256, 256),
) -> dict[str, object]:
    """Sample one authenticated arbitrary plane without using animal labels."""
    if not isinstance(split, str) or not split:
        raise ValueError("generic plane split must be a nonempty string")
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    if sample_index < 0 or plane_stratum not in V2_GENERIC_PLANE_STRATA:
        raise ValueError("generic plane split/index/stratum is invalid")
    _root_seed_hex(root_seed)
    global_reference_support_geometry(support_index, parent_shape_h_w)
    seeds: dict[str, str] = {}
    stream_stage = f"generic-pose/{plane_stratum}"

    def generator(field: str) -> np.random.Generator:
        rng, seed = _field_rng(root_seed, split, sample_index, stream_stage, field)
        seeds[field] = seed
        return rng

    def uniform(field: str, low: float = 0.0, high: float = 1.0) -> float:
        return float(generator(field).uniform(low, high))

    if plane_stratum in {"reference", "edge_or_partial"}:
        raw_normal = generator("isotropic-gaussian-normal").normal(size=3).astype(np.float64)
        normal_measure = "Haar-uniform RP2 from normalized isotropic Gaussian"
    elif plane_stratum.startswith("near_"):
        axis = {"near_AP": 0, "near_DV": 1, "near_ML": 2}[plane_stratum]
        cosine = uniform("axis-cosine", 0.90, 0.985)
        azimuth = 2.0 * math.pi * uniform("axis-azimuth")
        other = [value for value in range(3) if value != axis]
        raw_normal = np.zeros(3, dtype=np.float64)
        raw_normal[axis] = cosine
        radius = math.sqrt(1.0 - cosine * cosine)
        raw_normal[other] = radius * np.asarray((math.cos(azimuth), math.sin(azimuth)))
        normal_measure = "named near-cardinal stress stratum; not the reference measure"
    else:
        magnitudes = np.asarray(
            [uniform(f"oblique-magnitude-{axis}", 0.35, 0.75) for axis in range(3)],
            dtype=np.float64,
        )
        signs = np.asarray(
            [
                1.0,
                -1.0 if uniform("oblique-sign-1") < 0.5 else 1.0,
                -1.0 if uniform("oblique-sign-2") < 0.5 else 1.0,
            ],
            dtype=np.float64,
        )
        raw_normal = magnitudes * signs
        normal_measure = "named general-oblique stress stratum; not the reference measure"
    normal, _, _ = canonicalize_plane(raw_normal, 0.0)
    roll = 2.0 * math.pi * uniform("roll")
    intervals = shifted_component_interval_union(normal, support_index)
    if plane_stratum == "edge_or_partial":
        edge_depth = uniform("edge-depth-fraction", 0.01, 0.03)
        measure_fraction = (
            edge_depth if uniform("edge-side") < 0.5 else 1.0 - edge_depth
        )
        offset_measure = "named edge/partial tail stress; not the reference measure"
    else:
        measure_fraction = uniform("offset-measure-fraction")
        offset_measure = "length-uniform over authenticated merged brain-intersection intervals"
    signed_offset, selected_interval = _offset_at_measure_fraction(
        np.asarray(intervals["support_origin_interval_union_um"]), measure_fraction
    )
    frame = physical_plane_frame(normal, roll)
    payload = {
        "plane_stratum": plane_stratum,
        "reference_measure": plane_stratum == "reference",
        "normal_sampling_measure": normal_measure,
        "offset_sampling_measure": offset_measure,
        "stress_strata_do_not_change_reference_measure": True,
        "normal_draw_ap_dv_ml": raw_normal.tolist(),
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "roll_rad": roll,
        "signed_offset_um_about_support_origin": signed_offset,
        "offset_measure_fraction": measure_fraction,
        "selected_interval_index": selected_interval,
        "shifted_intervals": intervals,
        "frame_ap_dv_ml_physical": frame.tolist(),
        "field_stream_seed_uint64": seeds,
        "field_stream_stage": stream_stage,
        "field_stream_attempt_index": {field: 0 for field in seeds},
        "rejection_attempts": [],
        "animal_label_rng_dependencies": [],
    }
    return {**payload, "plane_sampler_receipt_sha256": _payload_sha256(payload)}


def global_reference_plane_geometry(
    normal_ap_dv_ml: np.ndarray,
    signed_offset_um_about_support_origin: float,
    roll_rad: float,
    support_index: dict[str, object],
    parent_shape_h_w: tuple[int, int] = (256, 256),
) -> dict[str, object]:
    """Construct the exact fixed-FOV v2 parent grid for one physical plane."""
    verify_annotation_support_index(support_index)
    normal, signed_offset, _ = canonicalize_plane(
        normal_ap_dv_ml, signed_offset_um_about_support_origin
    )
    fov = global_reference_support_geometry(support_index, parent_shape_h_w)
    height, width = fov["parent_shape_h_w"]
    support_origin = np.asarray(fov["support_origin_ap_dv_ml_um"], dtype=np.float64)
    intervals = shifted_component_interval_union(normal, support_index)
    interval_array = np.asarray(intervals["support_origin_interval_union_um"])
    if not np.any((signed_offset >= interval_array[:, 0]) & (signed_offset <= interval_array[:, 1])):
        raise ValueError("plane does not intersect the support-origin interval union")
    projection_offset = signed_offset - float(intervals["projection_to_support_origin_shift_um"])
    membership = _json_value(
        plane_interval_membership_certificate(normal, projection_offset, support_index)
    )
    if not membership["intersects"]:
        raise ValueError("shifted plane failed the authenticated v1 support membership certificate")
    frame = physical_plane_frame(normal, roll_rad)
    u, v = frame[:, 0], frame[:, 1]
    diameter = float(fov["diameter_um"])
    plane_center = support_origin + signed_offset * normal
    origin_physical = plane_center - 0.5 * diameter * u - 0.5 * diameter * v
    edge_u_physical = diameter * width / (width - 1.0) * u
    edge_v_physical = diameter * height / (height - 1.0) * v
    atlas_origin = tuple(support_index["origin_um"])
    spacing = tuple(support_index["voxel_size_um"])
    origin_index = physical_um_to_allen_index_points(
        torch.as_tensor(origin_physical), atlas_origin, spacing
    ).numpy()
    edge_u_index = physical_um_to_allen_index_vectors(
        torch.as_tensor(edge_u_physical), spacing
    ).numpy()
    edge_v_index = physical_um_to_allen_index_vectors(
        torch.as_tensor(edge_v_physical), spacing
    ).numpy()
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
    renderer_geometry = {
        "output_shape_h_w": [height, width],
        "renderer_center_ap_dv_ml": center_index.to(torch.float32).tolist(),
        "renderer_frame_ap_dv_ml": frame_index.to(torch.float32).tolist(),
        "renderer_inplane_basis": basis_index.to(torch.float32).tolist(),
    }
    effective = effective_renderer_sampling_arrays(
        renderer_geometry,
        atlas_shape,
        origin_ap_dv_ml_um=atlas_origin,
        voxel_size_ap_dv_ml_um=spacing,
    )
    independent_center64, independent_frame64, independent_basis64 = quicknii_ouv_to_frame(
        torch.as_tensor(quicknii_ouv, dtype=torch.float64), atlas_shape
    )
    effective_center = independent_center64.to(torch.float32)
    effective_frame = independent_frame64.to(torch.float32)
    effective_basis = independent_basis64.to(torch.float32)
    if not all(
        _array_receipt(values.numpy()) == _array_receipt(np.asarray(renderer_geometry[name], np.float32))
        for name, values in (
            ("renderer_center_ap_dv_ml", effective_center),
            ("renderer_frame_ap_dv_ml", effective_frame),
            ("renderer_inplane_basis", effective_basis),
        )
    ):
        raise ValueError("independent design O/U/V factorization changed the effective renderer state")
    s32 = torch.arange(width, dtype=torch.float32) / width
    t32 = torch.arange(height, dtype=torch.float32) / height
    tt32, ss32 = torch.meshgrid(t32, s32, indexing="ij")
    st32 = torch.stack((ss32, tt32), -1)
    edges32 = effective_frame[:, :2] @ effective_basis
    independent_points = effective_center + torch.matmul(
        edges32, (st32 - 0.5).unsqueeze(-1)
    ).squeeze(-1)
    independent_grid = torch.stack(
        (
            independent_points[..., 2] / (atlas_shape[2] - 1) * 2 - 1,
            independent_points[..., 1] / (atlas_shape[1] - 1) * 2 - 1,
            independent_points[..., 0] / (atlas_shape[0] - 1) * 2 - 1,
        ),
        -1,
    )
    independent_valid = torch.ones((height, width), dtype=torch.bool)
    rounded = torch.round(independent_points).to(torch.int64)
    for axis, size in enumerate(atlas_shape):
        independent_valid &= (rounded[..., axis] >= 0) & (rounded[..., axis] < size)
    independent_grid_receipts = {
        "coordinate_raster_allen_index_float32": _array_receipt(independent_points.numpy()),
        "normalized_interpolation_grid_xyz_float32": _array_receipt(independent_grid.numpy()),
        "valid_atlas_label_sampling_mask": _array_receipt(independent_valid.numpy()),
    }
    if any(independent_grid_receipts[name] != _array_receipt(effective[name]) for name in independent_grid_receipts):
        raise ValueError("independent float32 O/U/V grid reconstruction is not byte-identical")
    effective_origin = effective_center - 0.5 * edges32.sum(dim=1)
    independent_allen_ouv = torch.cat((effective_origin, edges32[:, 0], edges32[:, 1]))
    independent_quicknii_ouv = torch.cat(
        (
            allen_to_quicknii_points(independent_allen_ouv[:3], atlas_shape),
            allen_to_quicknii_vectors(independent_allen_ouv[3:6]),
            allen_to_quicknii_vectors(independent_allen_ouv[6:9]),
        )
    )
    independent_allen_numpy = independent_allen_ouv.numpy().astype(np.float64)
    independent_physical_ouv = np.concatenate(
        (
            np.asarray(atlas_origin) + (independent_allen_numpy[:3] + 0.5) * np.asarray(spacing),
            independent_allen_numpy[3:6] * np.asarray(spacing),
            independent_allen_numpy[6:9] * np.asarray(spacing),
        )
    )
    independent_ouv = {
        "allen_index_ouv_ap_dv_ml_float32": independent_allen_ouv.numpy(),
        "quicknii_ouv_ml_ap_dv_float32": independent_quicknii_ouv.numpy(),
        "physical_ouv_ap_dv_ml_um_from_float32_state": independent_physical_ouv,
    }
    if any(_array_receipt(values) != _array_receipt(effective[name]) for name, values in independent_ouv.items()):
        raise ValueError("independent effective O/U/V reconstruction is not byte-identical")
    ouv_points = (
        effective_origin
        + ss32[..., None] * edges32[:, 0]
        + tt32[..., None] * edges32[:, 1]
    )
    ouv_grid = torch.stack(
        (
            ouv_points[..., 2] / (atlas_shape[2] - 1) * 2 - 1,
            ouv_points[..., 1] / (atlas_shape[1] - 1) * 2 - 1,
            ouv_points[..., 0] / (atlas_shape[0] - 1) * 2 - 1,
        ),
        -1,
    )
    ouv_parameterization_max_abs_index = float(torch.max(torch.abs(ouv_points - independent_points)))
    ouv_parameterization_max_abs_normalized = float(torch.max(torch.abs(ouv_grid - independent_grid)))
    if ouv_parameterization_max_abs_index > 1e-5 or ouv_parameterization_max_abs_normalized > 1e-6:
        raise ValueError("effective O/U/V parameterization exceeds its float32 equivalence gate")
    x = np.arange(width, dtype=np.float64) / width
    y = np.arange(height, dtype=np.float64) / height
    design_physical_grid = (
        origin_physical[None, None]
        + x[None, :, None] * edge_u_physical[None, None]
        + y[:, None, None] * edge_v_physical[None, None]
    )
    design_index_grid = (
        origin_index[None, None]
        + x[None, :, None] * edge_u_index[None, None]
        + y[:, None, None] * edge_v_index[None, None]
    )
    reconstructed_physical = np.asarray(atlas_origin) + (design_index_grid + 0.5) * np.asarray(spacing)
    lower = np.asarray(fov["closed_face_lower_ap_dv_ml_um"])
    upper = np.asarray(fov["closed_face_upper_ap_dv_ml_um"])
    corners = np.stack(
        np.meshgrid(*[(lower[axis], upper[axis]) for axis in range(3)], indexing="ij"), -1
    ).reshape(-1, 3)
    projected_u = (corners - plane_center) @ u
    projected_v = (corners - plane_center) @ v
    clearance = min(
        float(projected_u.min() + diameter / 2.0),
        float(diameter / 2.0 - projected_u.max()),
        float(projected_v.min() + diameter / 2.0),
        float(diameter / 2.0 - projected_v.max()),
    )
    arrays = {
        "design_physical_coordinate_raster_ap_dv_ml_um_float64": design_physical_grid,
        "design_allen_index_coordinate_raster_float64": design_index_grid,
        **effective,
        "independent_ouv_parameterized_coordinate_raster_float32": ouv_points.numpy(),
        "independent_ouv_parameterized_normalized_grid_float32": ouv_grid.numpy(),
    }
    array_receipts = {name: _array_receipt(value) for name, value in arrays.items()}
    diagnostics = {
        "normal_norm_error": abs(float(np.linalg.norm(normal)) - 1.0),
        "frame_orthogonality_max_abs": float(np.max(np.abs(frame.T @ frame - np.eye(3)))),
        "frame_determinant_error": abs(float(np.linalg.det(frame)) - 1.0),
        "physical_index_roundtrip_max_abs_um": float(
            np.max(np.abs(reconstructed_physical - design_physical_grid))
        ),
        "plane_residual_max_abs_um": float(
            np.max(np.abs((design_physical_grid - support_origin) @ normal - signed_offset))
        ),
        "support_corner_minimum_inplane_clearance_um": clearance,
        "independent_effective_grid_byte_equal": True,
        "independent_effective_ouv_byte_equal": True,
        "ouv_parameterization_max_abs_index": ouv_parameterization_max_abs_index,
        "ouv_parameterization_max_abs_normalized": ouv_parameterization_max_abs_normalized,
    }
    if (
        diagnostics["normal_norm_error"] > 1e-12
        or diagnostics["frame_orthogonality_max_abs"] > 1e-12
        or diagnostics["frame_determinant_error"] > 1e-12
        or diagnostics["physical_index_roundtrip_max_abs_um"] > 1e-9
        or diagnostics["plane_residual_max_abs_um"] > 1e-9
        or clearance < float(fov["margin_um"]) - 1e-9
    ):
        raise ValueError("global-reference plane geometry failed its frozen numerical gates")
    physical_ouv = np.concatenate((origin_physical, edge_u_physical, edge_v_physical))
    allen_ouv = np.concatenate((origin_index, edge_u_index, edge_v_index))
    payload = {
        **renderer_geometry,
        "normal_rp2_ap_dv_ml": normal.tolist(),
        "signed_offset_um_about_support_origin": signed_offset,
        "roll_rad": float(roll_rad),
        "frame_ap_dv_ml_physical": frame.tolist(),
        "plane_center_ap_dv_ml_um": plane_center.tolist(),
        "physical_ouv_ap_dv_ml_um": physical_ouv.tolist(),
        "allen_index_ouv_ap_dv_ml": allen_ouv.tolist(),
        "quicknii_ouv_ml_ap_dv": quicknii_ouv.tolist(),
        "pose_state": {
            "center_ap_dv_ml_um": plane_center.tolist(),
            "proper_frame_ap_dv_ml": frame.tolist(),
            "positive_inplane_basis_um": [
                [float(np.linalg.norm(edge_u_physical)), 0.0],
                [0.0, float(np.linalg.norm(edge_v_physical))],
            ],
        },
        "global_reference_fov": fov,
        "shifted_intervals": intervals,
        "projection_origin_membership_certificate": membership,
        "array_receipts": array_receipts,
        "diagnostics": diagnostics,
        "raster_endpoint_semantics": {
            "pixel_mapping": "P(x,y)=O+(x/W)U+(y/H)V",
            "first_sample_ap_dv_ml_um": design_physical_grid[0, 0].tolist(),
            "last_sample_ap_dv_ml_um": design_physical_grid[-1, -1].tolist(),
            "u_edge_factor": width / (width - 1.0),
            "v_edge_factor": height / (height - 1.0),
        },
    }
    return {**payload, "global_reference_grid_id": _payload_sha256(payload)}


def _context_provenance(
    context: dict[str, object],
    animal_id: str | int | None,
    animal_index: int | None,
    specimen_id: str | int | None,
    experiment_id: str | int | None,
) -> dict[str, object]:
    receipt = context["receipt"]
    support = _context_support(context)
    resolved_animal_index = (
        None
        if animal_index is None
        else _nonnegative_integer(animal_index, "animal_index")
    )
    return {
        "animal_id": animal_id,
        "animal_index": resolved_animal_index,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "v2_context_sha256": context["v2_context_sha256"],
        "opaque_v1_prepared_context_sha256": receipt["opaque_v1_prepared_context_sha256"],
        "support_index_sha256": support["support_index_sha256"],
        "annotation_array_sha256": receipt["annotation_array_sha256"],
        "scalar_source_sha256": receipt["scalar_source"]["source_sha256"],
    }


def _validate_generic_animal_lineage(
    animal_id: str | int | None, animal_index: int | None
) -> None:
    if animal_id is None or animal_index is None:
        raise ValueError(
            "generic authenticated training animal lineage requires non-null "
            "animal_id and nonnegative animal_index"
        )
    _nonnegative_integer(animal_index, "animal_index")


def _v2_raster_metadata(
    scalar: np.ndarray, annotation: np.ndarray, brain_mask: np.ndarray
) -> dict[str, object]:
    scalar = np.asarray(scalar)
    annotation = np.asarray(annotation)
    brain_mask = np.asarray(brain_mask)
    if (
        scalar.dtype != np.dtype(np.float32)
        or annotation.dtype != np.dtype(np.int64)
        or brain_mask.dtype != np.dtype(bool)
        or scalar.shape != annotation.shape
        or scalar.shape != brain_mask.shape
        or not np.isfinite(scalar).all()
        or not np.array_equal(brain_mask, annotation != 0)
    ):
        raise ValueError("v2 centre-render arrays have the wrong dtype, shape, or support semantics")
    receipts = {
        "scalar": _array_receipt(scalar),
        "annotation": _array_receipt(annotation),
        "brain_mask": _array_receipt(brain_mask),
    }
    hashes = {f"{name}_sha256": receipt["array_sha256"] for name, receipt in receipts.items()}
    combined = _payload_sha256(
        {"schema": "anatomy-tracker.v2-centre-render-arrays/v1", "array_receipts": receipts}
    )
    return {
        "array_receipts": receipts,
        **hashes,
        "combined_sha256": combined,
        "brain_pixel_count": int(brain_mask.sum()),
    }


def _render_v2_centre_plane(
    context: dict[str, object], geometry: dict[str, object]
) -> dict[str, object]:
    parent = context["opaque_v1_context"]
    scalar_tensor = parent["scalar_tensor"]
    annotation_tensor = parent["annotation_tensor"]
    image, labels = render_arbitrary_plane(
        scalar_tensor,
        torch.as_tensor(geometry["renderer_center_ap_dv_ml"], dtype=scalar_tensor.dtype),
        torch.as_tensor(geometry["renderer_frame_ap_dv_ml"], dtype=scalar_tensor.dtype),
        torch.as_tensor(geometry["renderer_inplane_basis"], dtype=scalar_tensor.dtype),
        tuple(geometry["output_shape_h_w"]),
        annotation_tensor,
        sampling_contract=QUICKNII_RASTER_INDEX_SAMPLING,
    )
    scalar = image[0, 0].cpu().numpy()
    annotation = labels[0, 0].to(torch.int64).cpu().numpy()
    brain_mask = annotation != 0
    return {
        "scalar": scalar,
        "annotation": annotation,
        "brain_mask": brain_mask,
        **_v2_raster_metadata(scalar, annotation, brain_mask),
    }


def make_v2_smoke_global_reference_centre_render(
    prepared_context: dict[str, object],
    split: str,
    root_seed: int | str,
    sample_index: int,
    plane_stratum: str,
    *,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    animal_id: str | int | None = None,
    animal_index: int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
) -> dict[str, object]:
    """Create the v2 plane manifest, fixed global grid, and centre-plane render."""
    _validate_v2_context(prepared_context)
    root_hex = _root_seed_hex(root_seed)
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    parent_shape_h_w = _parent_shape(parent_shape_h_w)
    support = _context_support(prepared_context)
    if tuple(parent_shape_h_w) != _SMOKE_PARENT_SHAPE or tuple(parent_shape_h_w) != tuple(
        prepared_context["receipt"]["global_reference_fov"]["parent_shape_h_w"]
    ):
        raise ValueError("v2 smoke parent shape must match the frozen context-wide FOV")
    sampling = sample_v2_smoke_plane_pose(
        support, split, f"0x{root_hex}", sample_index, plane_stratum, parent_shape_h_w
    )
    smoke_case_assignment = _smoke_assignment(sample_index)
    geometry = global_reference_plane_geometry(
        np.asarray(sampling["normal_rp2_ap_dv_ml"]),
        float(sampling["signed_offset_um_about_support_origin"]),
        float(sampling["roll_rad"]),
        support,
        parent_shape_h_w,
    )
    provenance = _context_provenance(
        prepared_context, animal_id, animal_index, specimen_id, experiment_id
    )
    resolved_config = {
        "schema_version": V2_PLANE_SCHEMA,
        "algorithm": V2_PLANE_ALGORITHM,
        "split": split,
        "root_seed_uint64": f"0x{root_hex}",
        "sample_index": sample_index,
        "plane_stratum": plane_stratum,
        "parent_shape_h_w": list(parent_shape_h_w),
        "rng": {
            "derivation": "length-prefixed-v2-domain/schema/split/root/sample/stage/field/attempt",
            "digest": "BLAKE2b-64 person=AP-ACQ-V2; unsigned big-endian",
            "generator": "NumPy PCG64DXSM",
        },
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "learned_style_model_dependencies": [],
        "source_sha256": _source_hashes(),
        "source_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
        "preflight": _preflight_provenance(),
        "runtime": _runtime_receipt(),
    }
    resolved_config_sha256 = _payload_sha256(resolved_config)
    plane_payload = {
        "schema_version": V2_PLANE_SCHEMA,
        "algorithm": V2_PLANE_ALGORITHM,
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "provenance": provenance,
        "sampling": sampling,
        "geometry": geometry,
    }
    plane_id = _payload_sha256(plane_payload)
    raster = _render_v2_centre_plane(prepared_context, geometry)
    render_payload = {
        "v2_plane_realization_id": plane_id,
        "v2_context_sha256": prepared_context["v2_context_sha256"],
        "operator": "direct frozen render_arbitrary_plane offset-zero scalar/nearest-label primitive",
        "array_receipts": raster["array_receipts"],
        "combined_sha256": raster["combined_sha256"],
    }
    artifact = {
        "schema_version": V2_PLANE_SCHEMA,
        "v2_plane_realization_id": plane_id,
        "centre_plane_render_id": _payload_sha256(render_payload),
        "generator": {
            "resolved_config": resolved_config,
            "resolved_config_sha256": resolved_config_sha256,
        },
        "provenance": provenance,
        "smoke_case_assignment": smoke_case_assignment,
        "sampling": sampling,
        "geometry": geometry,
        "raster": raster,
    }
    artifact["receipt_sha256"] = _payload_sha256(v2_centre_render_receipt(artifact))
    return artifact


def v2_centre_render_receipt(artifact: dict[str, object]) -> dict[str, object]:
    raster = artifact["raster"]
    return {
        "schema_version": artifact["schema_version"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "centre_plane_render_id": artifact["centre_plane_render_id"],
        "generator": artifact["generator"],
        "provenance": artifact["provenance"],
        "smoke_case_assignment": artifact["smoke_case_assignment"],
        "sampling": artifact["sampling"],
        "geometry": artifact["geometry"],
        "raster_receipt": {
            "array_receipts": raster["array_receipts"],
            "scalar_sha256": raster["scalar_sha256"],
            "annotation_sha256": raster["annotation_sha256"],
            "brain_mask_sha256": raster["brain_mask_sha256"],
            "combined_sha256": raster["combined_sha256"],
            "brain_pixel_count": raster["brain_pixel_count"],
        },
    }


def replay_v2_smoke_global_reference_centre_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    config = artifact["generator"]["resolved_config"]
    provenance = artifact["provenance"]
    return make_v2_smoke_global_reference_centre_render(
        prepared_context,
        config["split"],
        config["root_seed_uint64"],
        config["sample_index"],
        config["plane_stratum"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=provenance["animal_id"],
        animal_index=provenance["animal_index"],
        specimen_id=provenance["specimen_id"],
        experiment_id=provenance["experiment_id"],
    )


def verify_v2_smoke_global_reference_centre_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> None:
    expected_artifact_keys = {
        "schema_version",
        "v2_plane_realization_id",
        "centre_plane_render_id",
        "generator",
        "provenance",
        "smoke_case_assignment",
        "sampling",
        "geometry",
        "raster",
        "receipt_sha256",
    }
    expected_raster_keys = {
        "scalar",
        "annotation",
        "brain_mask",
        "array_receipts",
        "scalar_sha256",
        "annotation_sha256",
        "brain_mask_sha256",
        "combined_sha256",
        "brain_pixel_count",
    }
    if set(artifact) != expected_artifact_keys or set(artifact.get("raster", {})) != expected_raster_keys:
        raise ValueError("v2 centre-render artifact contains missing or unauthenticated extra fields")
    if artifact.get("schema_version") != V2_PLANE_SCHEMA:
        raise ValueError("v2 centre-render schema does not match")
    for forbidden in (
        "slab_recipe_id",
        "slab_render_id",
        "acquisition_window_realization_id",
        "reflection_transform_id",
        "reflection_realization_id",
        "v2_acquisition_realization_id",
        "synthetic_realization_id",
    ):
        if forbidden in artifact:
            raise ValueError("v2 centre-render precursor contains a premature downstream ID")
    raster = artifact["raster"]
    live_metadata = _v2_raster_metadata(
        raster["scalar"], raster["annotation"], raster["brain_mask"]
    )
    if any(raster.get(key) != value for key, value in live_metadata.items()):
        raise ValueError("v2 centre-render live array receipts do not match")
    if artifact.get("receipt_sha256") != _payload_sha256(v2_centre_render_receipt(artifact)):
        raise ValueError("v2 centre-render receipt does not match")
    replayed = replay_v2_smoke_global_reference_centre_render(artifact, prepared_context)
    if v2_centre_render_receipt(artifact) != v2_centre_render_receipt(replayed):
        raise ValueError("v2 centre-render replay receipt does not match")
    for name in ("scalar", "annotation", "brain_mask"):
        if (
            _array_receipt(artifact["raster"][name])
            != _array_receipt(replayed["raster"][name])
            or not np.array_equal(artifact["raster"][name], replayed["raster"][name])
        ):
            raise ValueError("v2 centre-render replay arrays do not match")


def _generic_plane_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "resolved_config": artifact["generator"]["resolved_config"],
        "resolved_config_sha256": artifact["generator"]["resolved_config_sha256"],
        "provenance": artifact["provenance"],
        "sampling": artifact["sampling"],
        "geometry": artifact["geometry"],
    }


def _generic_centre_render_identity_payload(
    artifact: dict[str, object],
) -> dict[str, object]:
    raster = artifact["raster"]
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "v2_context_sha256": artifact["provenance"]["v2_context_sha256"],
        "operator": "direct frozen render_arbitrary_plane offset-zero scalar/nearest-label primitive",
        "array_receipts": raster["array_receipts"],
        "combined_sha256": raster["combined_sha256"],
    }


def make_v2_generic_global_reference_centre_render(
    prepared_context: dict[str, object],
    split: str,
    root_seed: int | str,
    sample_index: int,
    plane_stratum: str,
    *,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    animal_id: str | int | None = None,
    animal_index: int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
) -> dict[str, object]:
    """Render one generic authenticated arbitrary-plane training precursor."""
    _validate_generic_animal_lineage(animal_id, animal_index)
    _validate_v2_context(prepared_context)
    root_hex = _root_seed_hex(root_seed)
    sample_index = _nonnegative_integer(sample_index, "sample_index")
    parent_shape_h_w = _parent_shape(parent_shape_h_w)
    support = _context_support(prepared_context)
    if tuple(parent_shape_h_w) != tuple(
        prepared_context["receipt"]["global_reference_fov"]["parent_shape_h_w"]
    ):
        raise ValueError("generic plane canvas must match the context-wide FOV")
    sampling = sample_v2_generic_plane_pose(
        support,
        split,
        f"0x{root_hex}",
        sample_index,
        plane_stratum,
        parent_shape_h_w,
    )
    geometry = global_reference_plane_geometry(
        np.asarray(sampling["normal_rp2_ap_dv_ml"]),
        float(sampling["signed_offset_um_about_support_origin"]),
        float(sampling["roll_rad"]),
        support,
        parent_shape_h_w,
    )
    provenance = _context_provenance(
        prepared_context, animal_id, animal_index, specimen_id, experiment_id
    )
    resolved_config = {
        "schema_version": V2_GENERIC_PLANE_SCHEMA,
        "algorithm": V2_GENERIC_PLANE_ALGORITHM,
        "split": split,
        "root_seed_uint64": f"0x{root_hex}",
        "sample_index": sample_index,
        "plane_stratum": plane_stratum,
        "parent_shape_h_w": list(parent_shape_h_w),
        "rng": {
            "derivation": "length-prefixed-v2-domain/schema/split/root/sample/stage/field/attempt",
            "digest": "BLAKE2b-64 person=AP-ACQ-V2; unsigned big-endian",
            "generator": "NumPy PCG64DXSM",
            "animal_label_inputs": [],
        },
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "learned_style_model_dependencies": [],
        "source_sha256": _source_hashes(),
        "source_sha256_canonicalization": V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime": _runtime_receipt(),
    }
    artifact = {
        "schema_version": V2_GENERIC_PLANE_SCHEMA,
        "algorithm": V2_GENERIC_PLANE_ALGORITHM,
        "v2_plane_realization_id": None,
        "centre_plane_render_id": None,
        "generator": {
            "resolved_config": resolved_config,
            "resolved_config_sha256": _payload_sha256(resolved_config),
        },
        "provenance": provenance,
        "sampling": sampling,
        "geometry": geometry,
        "raster": _render_v2_centre_plane(prepared_context, geometry),
    }
    artifact["v2_plane_realization_id"] = _payload_sha256(
        _generic_plane_identity_payload(artifact)
    )
    artifact["centre_plane_render_id"] = _payload_sha256(
        _generic_centre_render_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = _payload_sha256(
        v2_generic_centre_render_receipt(artifact)
    )
    return artifact


def v2_generic_centre_render_receipt(
    artifact: dict[str, object],
) -> dict[str, object]:
    raster = artifact["raster"]
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "centre_plane_render_id": artifact["centre_plane_render_id"],
        "generator": artifact["generator"],
        "provenance": artifact["provenance"],
        "sampling": artifact["sampling"],
        "geometry": artifact["geometry"],
        "raster_receipt": {
            "array_receipts": raster["array_receipts"],
            "scalar_sha256": raster["scalar_sha256"],
            "annotation_sha256": raster["annotation_sha256"],
            "brain_mask_sha256": raster["brain_mask_sha256"],
            "combined_sha256": raster["combined_sha256"],
            "brain_pixel_count": raster["brain_pixel_count"],
        },
    }


def replay_v2_generic_global_reference_centre_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    config = artifact["generator"]["resolved_config"]
    provenance = artifact["provenance"]
    return make_v2_generic_global_reference_centre_render(
        prepared_context,
        config["split"],
        config["root_seed_uint64"],
        config["sample_index"],
        config["plane_stratum"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=provenance["animal_id"],
        animal_index=provenance["animal_index"],
        specimen_id=provenance["specimen_id"],
        experiment_id=provenance["experiment_id"],
    )


def verify_v2_generic_global_reference_centre_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> None:
    artifact_keys = {
        "schema_version",
        "algorithm",
        "v2_plane_realization_id",
        "centre_plane_render_id",
        "generator",
        "provenance",
        "sampling",
        "geometry",
        "raster",
        "receipt_sha256",
    }
    raster_keys = {
        "scalar",
        "annotation",
        "brain_mask",
        "array_receipts",
        "scalar_sha256",
        "annotation_sha256",
        "brain_mask_sha256",
        "combined_sha256",
        "brain_pixel_count",
    }
    config_keys = {
        "schema_version",
        "algorithm",
        "split",
        "root_seed_uint64",
        "sample_index",
        "plane_stratum",
        "parent_shape_h_w",
        "rng",
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
        "learned_style_model_dependencies",
        "source_sha256",
        "source_sha256_canonicalization",
        "runtime",
    }
    if (
        set(artifact) != artifact_keys
        or set(artifact.get("generator", {}))
        != {"resolved_config", "resolved_config_sha256"}
        or set(artifact.get("generator", {}).get("resolved_config", {})) != config_keys
        or set(artifact.get("provenance", {}))
        != {
            "animal_id",
            "animal_index",
            "specimen_id",
            "experiment_id",
            "v2_context_sha256",
            "opaque_v1_prepared_context_sha256",
            "support_index_sha256",
            "annotation_array_sha256",
            "scalar_source_sha256",
        }
        or set(artifact.get("raster", {})) != raster_keys
    ):
        raise ValueError("generic centre render has missing or unauthenticated extra fields")
    config = artifact["generator"]["resolved_config"]
    _validate_generic_animal_lineage(
        artifact["provenance"]["animal_id"],
        artifact["provenance"]["animal_index"],
    )
    if (
        artifact["schema_version"] != V2_GENERIC_PLANE_SCHEMA
        or artifact["algorithm"] != V2_GENERIC_PLANE_ALGORITHM
        or config["schema_version"] != V2_GENERIC_PLANE_SCHEMA
        or config["algorithm"] != V2_GENERIC_PLANE_ALGORITHM
        or "preflight" in config
        or config["source_sha256"] != _source_hashes()
        or config["source_sha256_canonicalization"]
        != V2_SOURCE_SHA256_CANONICALIZATION
        or any(
            config[name]
            for name in (
                "learned_checkpoint_dependencies",
                "previous_model_dependencies",
                "pretrained_feature_dependencies",
                "learned_style_model_dependencies",
            )
        )
        or artifact["provenance"]["v2_context_sha256"]
        != prepared_context["v2_context_sha256"]
        or artifact["generator"]["resolved_config_sha256"]
        != _payload_sha256(config)
    ):
        raise ValueError("generic centre render schema, source, context, or dependencies disagree")
    for forbidden in (
        "slab_recipe_id",
        "slab_render_id",
        "acquisition_window_realization_id",
        "reflection_transform_id",
        "reflection_realization_id",
        "v2_acquisition_realization_id",
        "synthetic_realization_id",
    ):
        if forbidden in artifact:
            raise ValueError("generic centre render contains a premature downstream ID")
    _validate_v2_context(prepared_context)
    raster = artifact["raster"]
    live = _v2_raster_metadata(
        raster["scalar"], raster["annotation"], raster["brain_mask"]
    )
    if (
        any(raster.get(key) != value for key, value in live.items())
        or artifact["v2_plane_realization_id"]
        != _payload_sha256(_generic_plane_identity_payload(artifact))
        or artifact["centre_plane_render_id"]
        != _payload_sha256(_generic_centre_render_identity_payload(artifact))
        or artifact["receipt_sha256"]
        != _payload_sha256(v2_generic_centre_render_receipt(artifact))
    ):
        raise ValueError("generic centre render live receipt or identity disagrees")
    replayed = replay_v2_generic_global_reference_centre_render(
        artifact, prepared_context
    )
    if v2_generic_centre_render_receipt(artifact) != v2_generic_centre_render_receipt(
        replayed
    ):
        raise ValueError("generic centre render deterministic replay receipt disagrees")
    for name in ("scalar", "annotation", "brain_mask"):
        if not np.array_equal(artifact["raster"][name], replayed["raster"][name]):
            raise ValueError("generic centre render deterministic replay arrays disagree")
