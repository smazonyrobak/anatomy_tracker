"""Authenticated finite pose-ranking proposals for one final v2 realization.

The bank is deliberately forced to contain one truth and a nonuniform set of
local/global perturbations.  It is a ranking proposal set, never posterior mass.
Candidate construction and acceptance use atlas support only, never target overlap.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_pose_v2 as pose_v2
import training.arbitrary_plane_realization_v2 as realization_v2
from training.arbitrary_plane_geometry import (
    horizontal_flip_quicknii_ouv,
    physical_um_to_allen_index_points,
    vertical_flip_quicknii_ouv,
)
from training.arbitrary_plane_manifest import canonicalize_plane


CANDIDATE_BANK_V2_SCHEMA = "anatomy-tracker.arbitrary-plane-candidate-bank/v2"
CANDIDATE_BANK_V2_ALGORITHM = (
    "forced-truth-support-measure-local-global-fixed-canvas-ranking-proposals/v2"
)
CANDIDATE_BANK_V2_RNG_DOMAIN = "anatomy-tracker.candidate-bank-rng/v2"
DEFAULT_CANDIDATE_ROOT_SEED = 0x43414E4442414E4B
DEFAULT_MAXIMUM_GLOBAL_ATTEMPTS = 4096
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_candidate_bank_v2.py",
    "arbitrary_plane_pose_v2.py",
    "arbitrary_plane_realization_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_geometry.py",
    "arbitrary_plane_manifest.py",
)
_ARRAY_KEYS = {
    "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64",
    "model_raster_physical_ouv_ap_dv_ml_um_float64",
    "rendered_annotation_int64",
    "brain_mask",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _root_seed_uint64(value: int | str) -> int:
    if isinstance(value, str):
        if (
            len(value) != 18
            or not value.startswith("0x")
            or any(character not in "0123456789abcdef" for character in value[2:])
        ):
            raise ValueError("candidate root seed must be canonical uint64 hexadecimal")
        return int(value[2:], 16)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("candidate root seed must be an integer or canonical hexadecimal")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError("candidate root seed must fit uint64")
    return result


def _schedule_uint64(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"{name} must fit uint64")
    return result


def derive_candidate_bank_seed_v2(
    root_seed: int | str,
    split_index: int,
    animal_index: int,
    section_index: int,
    observation_index: int,
    realization_index: int,
    candidate_class: str,
    slot: int | str,
    attempt_index: int,
) -> int:
    """Derive one stream from numeric lineage; labels and artifact IDs are absent."""
    numeric = tuple(
        _schedule_uint64(value, name)
        for value, name in zip(
            (
                split_index,
                animal_index,
                section_index,
                observation_index,
                realization_index,
                attempt_index,
            ),
            (
                "split_index",
                "animal_index",
                "section_index",
                "observation_index",
                "realization_index",
                "attempt_index",
            ),
            strict=True,
        )
    )
    if not isinstance(candidate_class, str) or not candidate_class:
        raise ValueError("candidate_class must be a nonempty string")
    if isinstance(slot, str):
        slot_text = slot
    else:
        slot_text = str(_schedule_uint64(slot, "slot"))
    components = (
        CANDIDATE_BANK_V2_RNG_DOMAIN,
        CANDIDATE_BANK_V2_SCHEMA,
        f"0x{_root_seed_uint64(root_seed):016x}",
        *(str(value) for value in numeric[:-1]),
        candidate_class,
        slot_text,
        str(numeric[-1]),
    )
    encoded = b"".join(
        len(component.encode("utf-8")).to_bytes(4, "big")
        + component.encode("utf-8")
        for component in components
    )
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-CAND-V2").digest(),
        "big",
    )


def _seed_receipt(
    provenance: Mapping[str, object],
    root_seed: int,
    candidate_class: str,
    slot: int | str,
    attempt_index: int,
) -> dict[str, object]:
    seed = derive_candidate_bank_seed_v2(
        root_seed,
        provenance["split_index"],
        provenance["animal_index"],
        provenance["section_index"],
        provenance["observation_index"],
        provenance["realization_index"],
        candidate_class,
        slot,
        attempt_index,
    )
    return {
        "domain": CANDIDATE_BANK_V2_RNG_DOMAIN,
        "root_seed_uint64": f"0x{root_seed:016x}",
        "split_index": int(provenance["split_index"]),
        "animal_index": int(provenance["animal_index"]),
        "section_index": int(provenance["section_index"]),
        "observation_index": int(provenance["observation_index"]),
        "realization_index": int(provenance["realization_index"]),
        "candidate_class": candidate_class,
        "slot": slot,
        "attempt_index": int(attempt_index),
        "seed_uint64": f"0x{seed:016x}",
        "generator": "numpy.random.PCG64DXSM",
        "animal_label_and_artifact_ids_excluded": True,
    }


def _ouv(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("candidate physical O/U/V must be finite float64 [3,3]")
    if float(np.linalg.norm(np.cross(array[1], array[2]))) <= 1e-12:
        raise ValueError("candidate physical O/U/V edges are collinear")
    return np.ascontiguousarray(array)


def _physical_raster(ouv: np.ndarray, shape_h_w: tuple[int, int]) -> np.ndarray:
    height, width = (int(value) for value in shape_h_w)
    if min(height, width) < 2:
        raise ValueError("candidate raster must have H,W >= 2")
    y, x = np.indices((height, width), dtype=np.float64)
    values = _ouv(ouv)
    return np.ascontiguousarray(
        values[0][None, None]
        + (x / width)[..., None] * values[1][None, None]
        + (y / height)[..., None] * values[2][None, None]
    )


def _render_physical_ouv_annotation_validated_v2(
    prepared_context: Mapping[str, object],
    physical_ouv_ap_dv_ml_um_float64: np.ndarray,
    output_shape_h_w: tuple[int, int],
) -> dict[str, object]:
    support = acquisition._context_support(prepared_context)
    physical = _physical_raster(
        physical_ouv_ap_dv_ml_um_float64, output_shape_h_w
    )
    allen = np.ascontiguousarray(
        physical_um_to_allen_index_points(
            torch.from_numpy(physical),
            tuple(float(value) for value in support["origin_um"]),
            tuple(float(value) for value in support["voxel_size_um"]),
        )
        .to(torch.float32)
        .numpy(),
        dtype=np.float32,
    )
    annotation = prepared_context["opaque_v1_context"]["annotation_tensor"]
    coordinates = torch.from_numpy(allen)
    rounded = torch.round(coordinates).to(torch.long)
    shape = annotation.shape
    inside = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < shape[0])
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < shape[1])
        & (rounded[..., 2] >= 0)
        & (rounded[..., 2] < shape[2])
    )
    sampled = torch.zeros(inside.shape, dtype=torch.int64)
    if bool(inside.any()):
        indices = rounded[inside]
        sampled[inside] = annotation[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ].to(torch.int64)
    labels = np.ascontiguousarray(sampled.numpy(), dtype=np.int64)
    mask = np.ascontiguousarray(labels != 0)
    return {
        "rendered_annotation_int64": labels,
        "brain_mask": mask,
        "physical_coordinate_raster_receipt": acquisition._array_receipt(physical),
        "allen_index_coordinate_raster_float32_receipt": acquisition._array_receipt(allen),
        "annotation_array_sha256": prepared_context["receipt"]["annotation_array_sha256"],
        "sampling_contract": "O+(x/W)U+(y/H)V; nearest torch.round ties-to-even; zero outside",
    }


def render_physical_ouv_annotation_v2(
    prepared_context: Mapping[str, object],
    physical_ouv_ap_dv_ml_um_float64: np.ndarray,
    output_shape_h_w: tuple[int, int],
) -> dict[str, object]:
    """Independently nearest-render Allen IDs under exact x/W,y/H sampling."""
    acquisition._validate_v2_context(prepared_context)
    return _render_physical_ouv_annotation_validated_v2(
        prepared_context,
        physical_ouv_ap_dv_ml_um_float64,
        output_shape_h_w,
    )


def _aligned_interval_union(
    normal: np.ndarray, prepared_context: Mapping[str, object]
) -> np.ndarray:
    support = acquisition._context_support(prepared_context)
    result = acquisition.shifted_component_interval_union(normal, support)
    source_normal = np.asarray(result["normal_rp2_ap_dv_ml"], dtype=np.float64)
    intervals = np.asarray(result["support_origin_interval_union_um"], dtype=np.float64)
    if float(source_normal @ normal) < 0.0:
        intervals = np.stack((-intervals[:, 1], -intervals[:, 0]), axis=-1)[::-1]
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not np.isfinite(intervals).all()
        or np.any(intervals[:, 1] <= intervals[:, 0])
    ):
        raise ValueError("candidate support interval union is invalid")
    return np.ascontiguousarray(intervals)


def _measure_state(intervals: np.ndarray, offset: float) -> dict[str, object]:
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    distances = np.maximum(intervals[:, 0] - offset, 0.0) + np.maximum(
        offset - intervals[:, 1], 0.0
    )
    index = int(np.argmin(distances))
    projected = float(np.clip(offset, intervals[index, 0], intervals[index, 1]))
    measure = float(lengths[:index].sum() + projected - intervals[index, 0])
    fraction = min(measure / total, float(np.nextafter(1.0, 0.0)))
    membership = bool(np.any((offset >= intervals[:, 0]) & (offset <= intervals[:, 1])))
    return {
        "interval_union_um": intervals.tolist(),
        "total_measure_um": total,
        "input_signed_offset_um": float(offset),
        "input_is_member": membership,
        "nearest_member_signed_offset_um": projected,
        "nearest_interval_index": index,
        "measure_um": measure,
        "measure_fraction": fraction,
    }


def _offset_at_measure(intervals: np.ndarray, measure_um: float) -> tuple[float, int, float]:
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    wrapped = float(measure_um % total)
    cumulative = np.cumsum(lengths)
    index = min(int(np.searchsorted(cumulative, wrapped, side="right")), len(intervals) - 1)
    before = float(lengths[:index].sum())
    return float(intervals[index, 0] + wrapped - before), index, wrapped / total


def _offset_at_fraction(intervals: np.ndarray, fraction: float) -> tuple[float, int, float]:
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("support-measure fraction must lie in [0,1)")
    return _offset_at_measure(intervals, fraction * float(np.diff(intervals, axis=1).sum()))


def _minimal_rotation(parent_normal: np.ndarray, candidate_normal: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_normal, dtype=np.float64)
    candidate = np.asarray(candidate_normal, dtype=np.float64)
    parent = parent / np.linalg.norm(parent)
    candidate = candidate / np.linalg.norm(candidate)
    if float(parent @ candidate) < 0.0:
        candidate = -candidate
    cross = np.cross(parent, candidate)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(parent @ candidate, -1.0, 1.0))
    if sine <= 1e-15:
        return np.eye(3, dtype=np.float64)
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def _transport_pre_reflection_ouv(
    base: Mapping[str, np.ndarray | float],
    candidate_normal: np.ndarray,
    candidate_offset_um: float,
    roll_delta_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    normal0 = np.asarray(base["normal"], dtype=np.float64)
    normal = np.asarray(candidate_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if float(normal @ normal0) < 0.0:
        normal = -normal
        candidate_offset_um = -float(candidate_offset_um)
    rotation = _minimal_rotation(normal0, normal)
    transported = rotation @ np.asarray(base["frame"], dtype=np.float64)
    cosine, sine = math.cos(float(roll_delta_rad)), math.sin(float(roll_delta_rad))
    frame = np.stack(
        (
            cosine * transported[:, 0] + sine * transported[:, 1],
            -sine * transported[:, 0] + cosine * transported[:, 1],
            normal,
        ),
        axis=-1,
    )
    support_origin = np.asarray(base["support_origin"], dtype=np.float64)
    centre = (
        support_origin
        + float(candidate_offset_um) * normal
        + rotation @ np.asarray(base["tangent_center_offset"], dtype=np.float64)
    )
    edges = frame[:, :2] @ np.asarray(base["basis"], dtype=np.float64)
    ouv = np.stack((centre - 0.5 * edges.sum(axis=1), edges[:, 0], edges[:, 1]))
    return np.ascontiguousarray(ouv), np.ascontiguousarray(frame)


def _reflect_ouv(
    pre_reflection_ouv: np.ndarray,
    output_shape_h_w: tuple[int, int],
    horizontal: bool,
    vertical: bool,
) -> np.ndarray:
    height, width = output_shape_h_w
    result = torch.from_numpy(
        np.array(pre_reflection_ouv, dtype=np.float64, copy=True, order="C").reshape(9)
    )
    if horizontal:
        result = horizontal_flip_quicknii_ouv(result, width)
    if vertical:
        result = vertical_flip_quicknii_ouv(result, height)
    return np.ascontiguousarray(result.numpy().reshape(3, 3), dtype=np.float64)


def _candidate_identity(candidate: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value({
        key: value
        for key, value in candidate.items()
        if key not in {"arrays", "candidate_id"}
    })


def _make_candidate(
    prepared_context: Mapping[str, object],
    base: Mapping[str, object],
    output_shape: tuple[int, int],
    reflection: tuple[bool, bool],
    candidate_class: str,
    slot: int,
    seed_receipt: Mapping[str, object],
    normal: np.ndarray,
    offset: float,
    roll: float,
    proposal: Mapping[str, object],
    *,
    exact_truth_ouvs: tuple[np.ndarray, np.ndarray] | None = None,
) -> Mapping[str, object]:
    if exact_truth_ouvs is None:
        pre_ouv, frame = _transport_pre_reflection_ouv(base, normal, offset, roll)
        model_ouv = _reflect_ouv(pre_ouv, output_shape, *reflection)
    else:
        pre_ouv, model_ouv = (_ouv(value) for value in exact_truth_ouvs)
        frame = np.asarray(base["frame"], dtype=np.float64)
    rendered = _render_physical_ouv_annotation_validated_v2(
        prepared_context, model_ouv, output_shape
    )
    canonical_normal, canonical_offset, _ = canonicalize_plane(normal, offset)
    support_envelope = _measure_state(
        _aligned_interval_union(np.asarray(normal, dtype=np.float64), prepared_context),
        float(offset),
    )
    arrays = {
        "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64": pre_ouv,
        "model_raster_physical_ouv_ap_dv_ml_um_float64": model_ouv,
        "rendered_annotation_int64": rendered["rendered_annotation_int64"],
        "brain_mask": rendered["brain_mask"],
    }
    candidate = {
        "candidate_class": candidate_class,
        "slot": int(slot),
        "field_seed": acquisition._json_value(seed_receipt),
        "proposal": acquisition._json_value(proposal),
        "pose": {
            "actual_normal_ap_dv_ml": np.asarray(normal, dtype=np.float64).tolist(),
            "actual_signed_offset_um": float(offset),
            "canonical_normal_ap_dv_ml": canonical_normal.tolist(),
            "canonical_signed_offset_um": float(canonical_offset),
            "roll_delta_rad_from_truth": float(roll),
            "proper_frame_ap_dv_ml": frame.tolist(),
        },
        "render_contract": {
            "sampling_contract": rendered["sampling_contract"],
            "physical_coordinate_raster_receipt": rendered[
                "physical_coordinate_raster_receipt"
            ],
            "allen_index_coordinate_raster_float32_receipt": rendered[
                "allen_index_coordinate_raster_float32_receipt"
            ],
            "annotation_array_sha256": rendered["annotation_array_sha256"],
            "target_overlap_used_for_construction_or_acceptance": False,
        },
        "infinite_plane_support_envelope": {
            "reference": "3-D nonzero annotation support projected onto candidate normal",
            "signed_offset_interval_state": support_envelope,
            "plane_intersects_support_envelope": support_envelope["input_is_member"],
            "not_a_finite_raster_support_claim": True,
        },
        "brain_pixel_count": int(rendered["brain_mask"].sum()),
        "finite_raster_support": bool(rendered["brain_mask"].any()),
        "arrays": arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value) for name, value in arrays.items()
        },
    }
    candidate["candidate_id"] = acquisition._payload_sha256(
        _candidate_identity(candidate)
    )
    return acquisition._freeze_value(candidate)


def _local_schedule(base: Mapping[str, object]) -> list[dict[str, object]]:
    normal = np.asarray(base["normal"], dtype=np.float64)
    frame = np.asarray(base["frame"], dtype=np.float64)
    schedule = []
    for slot, delta in enumerate((-100.0, 100.0, -250.0, 250.0, -500.0, 500.0)):
        schedule.append({"candidate_class": "offset_only", "slot": slot, "normal": normal, "measure_delta_um": delta, "roll": 0.0})
    slot = 0
    for axis_name, axis in (("u", frame[:, 0]), ("v", frame[:, 1])):
        for magnitude in (1.0, 3.0, 7.0, 12.0):
            for sign in (-1.0, 1.0):
                angle = sign * magnitude
                theta = math.radians(angle)
                schedule.append({"candidate_class": "normal_angle_only", "slot": slot, "normal": math.cos(theta) * normal + math.sin(theta) * axis, "measure_delta_um": 0.0, "roll": 0.0, "angle_deg": angle, "axis": axis_name})
                slot += 1
    for slot, roll_deg in enumerate((-3.0, 3.0, -10.0, 10.0, -30.0, 30.0)):
        schedule.append({"candidate_class": "roll_only", "slot": slot, "normal": normal, "measure_delta_um": 0.0, "roll": math.radians(roll_deg), "roll_deg": roll_deg})
    coupled = (
        (3.0, frame[:, 0], "u", 250.0, 10.0),
        (-3.0, frame[:, 0], "u", -250.0, -10.0),
        (7.0, frame[:, 1], "v", -100.0, 30.0),
        (-7.0, frame[:, 1], "v", 100.0, -30.0),
        (12.0, (frame[:, 0] + frame[:, 1]) / math.sqrt(2.0), "u_plus_v", 500.0, 15.0),
    )
    for slot, (angle, axis, axis_name, delta, roll_deg) in enumerate(coupled):
        theta = math.radians(angle)
        schedule.append({"candidate_class": "coupled_local", "slot": slot, "normal": math.cos(theta) * normal + math.sin(theta) * axis, "measure_delta_um": delta, "roll": math.radians(roll_deg), "angle_deg": angle, "axis": axis_name, "roll_deg": roll_deg})
    return schedule


def _model_grid_reference(final: Mapping[str, object]) -> dict[str, object]:
    frame_shape = tuple(final["frame_transform"]["output_shape_h_w"])
    input_shape = tuple(final["model_input"]["spatial_shape_h_w"])
    if frame_shape != input_shape or len(frame_shape) != 2 or min(frame_shape) < 2:
        raise ValueError("final frame and model-input spatial shapes differ")
    return {
        "output_shape_h_w": list(frame_shape),
        "shape_sources": [
            "frame_transform.output_shape_h_w",
            "model_input.spatial_shape_h_w",
        ],
        "target_arrays_or_receipts_accessed": False,
    }


def _source_lineage_reference(final: Mapping[str, object]) -> dict[str, object]:
    provenance = final["provenance"]
    upstream = final["upstream_reference"]
    binding = upstream["live_receipt_bindings"].get("support_resolution")
    if not isinstance(binding, Mapping):
        raise ValueError("final realization lacks its support-resolution receipt binding")
    payload = acquisition._json_value(binding.get("receipt_payload"))
    if binding.get("receipt_sha256") != acquisition._payload_sha256(payload):
        raise ValueError("support-resolution live receipt binding changed")
    try:
        plan = payload["plan_identity_payload"]
        lineage = plan["lineage"]
        configuration = plan["configuration"]
    except (KeyError, TypeError) as error:
        raise ValueError("support-resolution source lineage receipt is incomplete") from error
    if (
        lineage["split"] != provenance["split"]
        or lineage["animal_index"] != provenance["animal_index"]
        or lineage["animal_id"] != provenance["animal_id"]
        or configuration["split_index"] != provenance["split_index"]
        or configuration["animal_index"] != provenance["animal_index"]
        or configuration["section_index"] != provenance["section_index"]
        or plan["support_resolution_plan_id"]
        != upstream["support_resolution_plan_id"]
    ):
        raise ValueError("support-resolution and final source lineage differ")
    plane_stratum = configuration["plane_stratum"]
    if plane_stratum not in acquisition.V2_GENERIC_PLANE_STRATA:
        raise ValueError("source plane_stratum is outside the frozen v2 strata")
    return {
        "support_resolution_plan_id": plan["support_resolution_plan_id"],
        "support_resolution_receipt_sha256": binding["receipt_sha256"],
        "split": lineage["split"],
        "split_index": configuration["split_index"],
        "animal_id": acquisition._json_value(lineage["animal_id"]),
        "animal_index": configuration["animal_index"],
        "specimen_id": acquisition._json_value(lineage["specimen_id"]),
        "experiment_id": acquisition._json_value(lineage["experiment_id"]),
        "section_index": configuration["section_index"],
        "plane_stratum": plane_stratum,
    }


def _bank_identity(bank: Mapping[str, object]) -> dict[str, object]:
    return acquisition._json_value({
        key: value
        for key, value in bank.items()
        if key not in {"candidates", "candidate_bank_id", "receipt_sha256"}
    }) | {
        "candidate_receipts": [_candidate_identity(item) | {"candidate_id": item["candidate_id"]} for item in bank["candidates"]]
    }


def arbitrary_plane_candidate_bank_receipt_v2(
    bank: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate_bank_id": bank["candidate_bank_id"],
        "identity_payload": _bank_identity(bank),
    }


def make_arbitrary_plane_candidate_bank_v2(
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    candidate_root_seed: int | str = DEFAULT_CANDIDATE_ROOT_SEED,
    maximum_global_attempts: int = DEFAULT_MAXIMUM_GLOBAL_ATTEMPTS,
) -> Mapping[str, object]:
    """Make the fixed 40-proposal bank after verifying pose/final/context lineage."""
    pose_v2.verify_arbitrary_plane_pose_truth_v2(
        pose_truth, final_realization, prepared_context
    )
    root_seed = _root_seed_uint64(candidate_root_seed)
    maximum_global_attempts = _schedule_uint64(
        maximum_global_attempts, "maximum_global_attempts"
    )
    if not 1 <= maximum_global_attempts <= DEFAULT_MAXIMUM_GLOBAL_ATTEMPTS:
        raise ValueError("maximum_global_attempts must be in [1,4096]")
    provenance = final_realization["provenance"]
    numeric_names = (
        "split_index", "animal_index", "section_index", "observation_index", "realization_index"
    )
    numeric = {name: _schedule_uint64(provenance[name], name) for name in numeric_names}
    provenance = acquisition._json_value(provenance)
    if any(provenance[name] != numeric[name] for name in numeric_names):
        raise ValueError("final realization numeric provenance changed")

    model_grid_reference = _model_grid_reference(final_realization)
    source_lineage = _source_lineage_reference(final_realization)
    output_shape = tuple(model_grid_reference["output_shape_h_w"])
    pose_arrays = pose_truth["arrays"]
    pre_truth = _ouv(pose_arrays[
        "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
    ])
    model_truth = _ouv(pose_arrays[
        "model_raster_physical_ouv_ap_dv_ml_um_float64"
    ])
    final_model = _ouv(final_realization["frame_transform"]["arrays"][
        "model_raster_physical_ouv_ap_dv_ml_um_float64"
    ])
    nominal = _physical_raster(model_truth, output_shape)
    saved_nominal = np.asarray(final_realization["factor_truth"]["arrays"][
        "nominal_physical_map_ap_dv_ml_um_float64"
    ])
    if not np.array_equal(model_truth, final_model) or not np.array_equal(nominal, saved_nominal):
        raise ValueError("pose truth does not reproduce final model O/U/V and nominal map")

    frame = np.asarray(pose_arrays["proper_frame_ap_dv_ml_float64"], dtype=np.float64)
    basis = np.asarray(
        pose_arrays["positive_upper_triangular_basis_um_float64"], dtype=np.float64
    )
    center = np.asarray(
        pose_arrays["cropped_pre_reflection_physical_center_ap_dv_ml_um_float64"],
        dtype=np.float64,
    )
    support_origin = np.asarray(
        pose_arrays["support_origin_ap_dv_ml_um_float64"], dtype=np.float64
    )
    actual = np.asarray(
        pose_arrays["actual_plane_normal_and_signed_offset_um_float64"], dtype=np.float64
    )
    normal0, offset0 = actual[:3], float(actual[3])
    base = {
        "normal": normal0,
        "offset": offset0,
        "frame": frame,
        "basis": basis,
        "support_origin": support_origin,
        "tangent_center_offset": center - support_origin - offset0 * normal0,
    }
    base_intervals = _aligned_interval_union(normal0, prepared_context)
    base_measure = _measure_state(base_intervals, offset0)
    horizontal = bool(pose_truth["reflection_state"]["horizontal_reflection"])
    vertical = bool(pose_truth["reflection_state"]["vertical_reflection"])
    reflection = (horizontal, vertical)

    attempts: list[dict[str, object]] = []
    canonical_candidates: list[Mapping[str, object]] = []
    truth_seed = _seed_receipt(provenance, root_seed, "truth", 0, 0)
    truth = _make_candidate(
        prepared_context,
        base,
        output_shape,
        reflection,
        "truth",
        0,
        truth_seed,
        normal0,
        offset0,
        0.0,
        {
            "source": "exact verified pose-truth cropped/model O/U/V",
            "forced_candidate": True,
        },
        exact_truth_ouvs=(pre_truth, model_truth),
    )
    canonical_candidates.append(truth)
    attempts.append({
        "candidate_class": "truth", "slot": 0, "attempt_index": 0,
        "field_seed": truth_seed, "candidate_id": truth["candidate_id"],
        "accepted": True, "reason": "forced exact truth; target overlap absent",
    })

    for item in _local_schedule(base):
        candidate_class, slot = item["candidate_class"], int(item["slot"])
        normal = np.array(item["normal"], dtype=np.float64, copy=True)
        normal /= np.linalg.norm(normal)
        intervals = _aligned_interval_union(normal, prepared_context)
        total = float(np.diff(intervals, axis=1).sum())
        if candidate_class in {"offset_only", "roll_only"}:
            origin_measure = float(base_measure["measure_um"])
        else:
            origin_measure = float(base_measure["measure_fraction"]) * total
        offset, interval_index, fraction = _offset_at_measure(
            intervals, origin_measure + float(item["measure_delta_um"])
        )
        seed = _seed_receipt(provenance, root_seed, candidate_class, slot, 0)
        proposal = {
            key: value
            for key, value in item.items()
            if key not in {"candidate_class", "slot", "normal", "roll"}
        }
        proposal.update({
            "edge_safe_offset_policy": "cumulative merged-interval measure modulo total length",
            "base_measure_fraction": base_measure["measure_fraction"],
            "candidate_interval_index": interval_index,
            "candidate_measure_fraction": fraction,
            "candidate_signed_offset_um": offset,
            "target_overlap_used": False,
        })
        candidate = _make_candidate(
            prepared_context, base, output_shape, reflection,
            candidate_class, slot, seed, normal, offset, float(item["roll"]), proposal,
        )
        canonical_candidates.append(candidate)
        attempts.append({
            "candidate_class": candidate_class, "slot": slot, "attempt_index": 0,
            "field_seed": seed, "candidate_id": candidate["candidate_id"],
            "accepted": True, "reason": "fixed support-measure proposal; target overlap absent",
        })

    occupied_geometry = {
        candidate["array_receipts"][
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ]["array_sha256"]
        for candidate in canonical_candidates
    }
    if len(occupied_geometry) != len(canonical_candidates):
        raise RuntimeError(
            "fixed local candidate schedule has duplicate pre-reflection O/U/V coverage"
        )
    for slot in range(6):
        accepted = False
        for attempt_index in range(maximum_global_attempts):
            seed = _seed_receipt(
                provenance, root_seed, "global_hard_negative", slot, attempt_index
            )
            rng = np.random.Generator(np.random.PCG64DXSM(int(seed["seed_uint64"], 16)))
            normal = rng.normal(size=3).astype(np.float64)
            normal /= np.linalg.norm(normal)
            if float(normal @ normal0) < 0.0:
                normal = -normal
            angle = math.degrees(math.acos(float(np.clip(normal @ normal0, -1.0, 1.0))))
            if not 20.0 <= angle <= 60.0:
                attempts.append({
                    "candidate_class": "global_hard_negative", "slot": slot,
                    "attempt_index": attempt_index, "field_seed": seed,
                    "sampled_normal_angle_deg": angle, "accepted": False,
                    "reason": "normal-angle-outside-20-to-60-deg",
                })
                continue
            intervals = _aligned_interval_union(normal, prepared_context)
            fraction = float(rng.random())
            offset, interval_index, _ = _offset_at_fraction(intervals, fraction)
            roll = float(rng.uniform(-math.pi, math.pi))
            proposal = {
                "normal_sampling": "normalized isotropic Gaussian, sign-aligned to truth",
                "sampled_normal_angle_deg": angle,
                "offset_sampling": "length-uniform over aligned merged interval union",
                "candidate_interval_index": interval_index,
                "candidate_measure_fraction": fraction,
                "candidate_signed_offset_um": offset,
                "roll_sampling": "uniform [-pi,pi)",
                "target_overlap_used": False,
            }
            candidate = _make_candidate(
                prepared_context, base, output_shape, reflection,
                "global_hard_negative", slot, seed, normal, offset, roll, proposal,
            )
            geometry_id = candidate["array_receipts"][
                "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
            ]["array_sha256"]
            reason = "accepted" if geometry_id not in occupied_geometry else "duplicate-physical-ouv"
            attempts.append({
                "candidate_class": "global_hard_negative", "slot": slot,
                "attempt_index": attempt_index, "field_seed": seed,
                "sampled_normal_angle_deg": angle,
                "candidate_id": candidate["candidate_id"],
                "accepted": reason == "accepted", "reason": reason,
            })
            if reason == "accepted":
                canonical_candidates.append(candidate)
                occupied_geometry.add(geometry_id)
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"global candidate slot {slot} exhausted deterministic attempts")

    if len(canonical_candidates) != 40:
        raise RuntimeError("candidate schedule did not produce exactly forty proposals")
    order_seed = _seed_receipt(provenance, root_seed, "final_order", "bank", 0)
    order = np.random.Generator(
        np.random.PCG64DXSM(int(order_seed["seed_uint64"], 16))
    ).permutation(40).tolist()
    canonical_ids = [candidate["candidate_id"] for candidate in canonical_candidates]
    candidates = [canonical_candidates[index] for index in order]
    ordered_ids = [candidate["candidate_id"] for candidate in candidates]

    pose_receipt = acquisition._json_value(
        pose_v2.finite_plane_pose_truth_receipt_v2(pose_truth)
    )
    context_receipt = acquisition._json_value(prepared_context["receipt"])
    bank = {
        "schema_version": CANDIDATE_BANK_V2_SCHEMA,
        "algorithm": CANDIDATE_BANK_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {
            "numpy_version": np.__version__, "torch_version": torch.__version__, "torch_device": "cpu"
        },
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "scope": {
            "purpose": "finite semantic pose-ranking proposal bank only",
            "forced_truth_candidate": True,
            "nonuniform_proposal_bank": True,
            "posterior_mass_claim": False,
            "calibrated_uncertainty_claim": False,
            "semantic_scores_present": False,
            "target_overlap_used_for_candidate_construction_or_acceptance": False,
        },
        "upstream_reference": {
            "v2_context_sha256": prepared_context["v2_context_sha256"],
            "prepared_context_receipt_sha256": acquisition._payload_sha256(context_receipt),
            "support_index_sha256": prepared_context["receipt"]["support_index_sha256"],
            "annotation_array_sha256": prepared_context["receipt"]["annotation_array_sha256"],
            "finite_plane_pose_truth_id": pose_truth["finite_plane_pose_truth_id"],
            "finite_plane_pose_truth_receipt": pose_receipt,
            "finite_plane_pose_truth_receipt_sha256": acquisition._payload_sha256(pose_receipt),
            "synthetic_realization_id": final_realization["synthetic_realization_id"],
            "synthetic_realization_receipt_sha256": final_realization["receipt_sha256"],
            "training_row_id": final_realization["training_row_id"],
            "frame_transform_id": final_realization["frame_transform"]["frame_transform_id"],
        },
        "provenance": provenance,
        "source_lineage": source_lineage,
        "reflection_state": {
            "horizontal_reflection": horizontal,
            "vertical_reflection": vertical,
            "order": ["horizontal", "vertical"],
            "semantics": "exact finite-raster O/U/V reparameterization after pre-reflection transport",
        },
        "rng_contract": {
            "candidate_root_seed_uint64": f"0x{root_seed:016x}",
            "coordinates": [*numeric_names, "candidate_class", "slot", "attempt_index"],
            "excluded": ["split", "animal_id", "specimen_id", "experiment_id", "artifact_ids"],
            "maximum_global_attempts": int(maximum_global_attempts),
            "final_order_seed": order_seed,
        },
        "schedule": {
            "truth": 1, "offset_only": 6, "normal_angle_only": 16,
            "roll_only": 6, "coupled_local": 5, "global_hard_negative": 6,
            "total": 40,
            "edge_policy": "cumulative aligned merged-interval measure; no ordinary-margin gate",
            "zero_support_wrong_candidate_policy": "retain",
            "all_pre_reflection_ouv_receipts_unique": True,
        },
        "model_grid_reference": model_grid_reference,
        "truth_evaluability": {
            "independent_truth_brain_pixel_count": truth["brain_pixel_count"],
            "evaluable": truth["brain_pixel_count"] > 0,
            "zero_support_policy": "mark unevaluable; never redraw or inspect target overlap",
            "truth_plane_interval_measure": base_measure,
        },
        "candidate_attempts": attempts,
        "candidate_attempts_sha256": acquisition._payload_sha256(attempts),
        "canonical_candidate_ids": canonical_ids,
        "final_order_canonical_indices": order,
        "ordered_candidate_ids": ordered_ids,
        "candidates": candidates,
    }
    bank["candidate_bank_id"] = acquisition._payload_sha256(_bank_identity(bank))
    bank["receipt_sha256"] = acquisition._payload_sha256(
        arbitrary_plane_candidate_bank_receipt_v2(bank)
    )
    return acquisition._freeze_value(bank)


def replay_arbitrary_plane_candidate_bank_v2(
    bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> Mapping[str, object]:
    config = bank["rng_contract"]
    return make_arbitrary_plane_candidate_bank_v2(
        pose_truth,
        final_realization,
        prepared_context,
        candidate_root_seed=config["candidate_root_seed_uint64"],
        maximum_global_attempts=config["maximum_global_attempts"],
    )


def verify_arbitrary_plane_candidate_bank_v2(
    bank: Mapping[str, object],
    pose_truth: Mapping[str, object],
    final_realization: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    required = {
        "schema_version", "algorithm", "implementation_source_sha256",
        "implementation_source_sha256_canonicalization", "runtime_dependencies",
        "asset_dependencies", "scope", "upstream_reference", "provenance",
        "source_lineage", "reflection_state", "rng_contract", "schedule",
        "model_grid_reference", "truth_evaluability",
        "candidate_attempts", "candidate_attempts_sha256", "canonical_candidate_ids",
        "final_order_canonical_indices", "ordered_candidate_ids", "candidates",
        "candidate_bank_id", "receipt_sha256",
    }
    if (
        set(bank) != required
        or bank.get("schema_version") != CANDIDATE_BANK_V2_SCHEMA
        or bank.get("algorithm") != CANDIDATE_BANK_V2_ALGORITHM
        or len(bank.get("candidates", ())) != 40
        or [item.get("candidate_id") for item in bank.get("candidates", ())]
        != list(bank.get("ordered_candidate_ids", ()))
        or sum(item.get("candidate_class") == "truth" for item in bank.get("candidates", ())) != 1
        or any(bank.get("asset_dependencies", {}).values())
        or bank.get("scope", {}).get("posterior_mass_claim") is not False
        or bank.get("scope", {}).get("semantic_scores_present") is not False
        or bank.get("scope", {}).get("target_overlap_used_for_candidate_construction_or_acceptance") is not False
        or acquisition._json_value(bank.get("model_grid_reference"))
        != acquisition._json_value(_model_grid_reference(final_realization))
        or bank.get("source_lineage") != _source_lineage_reference(final_realization)
        or bank.get("reflection_state", {}).get("horizontal_reflection")
        is not pose_truth["reflection_state"]["horizontal_reflection"]
        or bank.get("reflection_state", {}).get("vertical_reflection")
        is not pose_truth["reflection_state"]["vertical_reflection"]
    ):
        raise ValueError("candidate bank has missing, extra, learned, posterior, or scoring fields")
    expected_slots = {
        "truth": set(range(1)),
        "offset_only": set(range(6)),
        "normal_angle_only": set(range(16)),
        "roll_only": set(range(6)),
        "coupled_local": set(range(5)),
        "global_hard_negative": set(range(6)),
    }
    actual_slots = {
        name: {
            int(item["slot"])
            for item in bank["candidates"]
            if item["candidate_class"] == name
        }
        for name in expected_slots
    }
    if (
        Counter(item["candidate_class"] for item in bank["candidates"])
        != Counter({name: len(slots) for name, slots in expected_slots.items()})
        or actual_slots != expected_slots
        or len(set(bank["ordered_candidate_ids"])) != 40
        or len(set(bank["canonical_candidate_ids"])) != 40
        or set(bank["ordered_candidate_ids"]) != set(bank["canonical_candidate_ids"])
        or sorted(bank["final_order_canonical_indices"]) != list(range(40))
    ):
        raise ValueError("candidate bank class, slot, identity, or order coverage changed")
    for candidate in bank["candidates"]:
        support_state = _measure_state(
            _aligned_interval_union(
                np.asarray(candidate["pose"]["actual_normal_ap_dv_ml"], dtype=np.float64),
                prepared_context,
            ),
            float(candidate["pose"]["actual_signed_offset_um"]),
        )
        if (
            set(candidate.get("arrays", {})) != _ARRAY_KEYS
            or set(candidate.get("array_receipts", {})) != _ARRAY_KEYS
            or any(
                acquisition._json_value(candidate["array_receipts"][name])
                != acquisition._json_value(acquisition._array_receipt(candidate["arrays"][name]))
                for name in _ARRAY_KEYS
            )
            or candidate["candidate_id"] != acquisition._payload_sha256(
                _candidate_identity(candidate)
            )
            or candidate["render_contract"][
                "target_overlap_used_for_construction_or_acceptance"
            ] is not False
            or candidate["infinite_plane_support_envelope"].get(
                "not_a_finite_raster_support_claim"
            ) is not True
            or acquisition._json_value(
                candidate["infinite_plane_support_envelope"][
                    "signed_offset_interval_state"
                ]
            )
            != acquisition._json_value(support_state)
            or candidate["infinite_plane_support_envelope"][
                "plane_intersects_support_envelope"
            ]
            is not support_state["input_is_member"]
            or candidate["brain_pixel_count"]
            != int(np.asarray(candidate["arrays"]["brain_mask"]).sum())
            or candidate["finite_raster_support"]
            is not bool(np.asarray(candidate["arrays"]["brain_mask"]).any())
        ):
            raise ValueError("candidate arrays, identity, or target-independence changed")
    geometry_receipts = [
        item["array_receipts"][
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64"
        ]["array_sha256"]
        for item in bank["candidates"]
    ]
    if len(set(geometry_receipts)) != 40:
        raise ValueError("candidate bank pre-reflection O/U/V coverage is not unique")
    if (
        bank["candidate_attempts_sha256"]
        != acquisition._payload_sha256(
            acquisition._json_value(bank["candidate_attempts"])
        )
        or bank["candidate_bank_id"] != acquisition._payload_sha256(_bank_identity(bank))
        or bank["receipt_sha256"]
        != acquisition._payload_sha256(arbitrary_plane_candidate_bank_receipt_v2(bank))
    ):
        raise ValueError("candidate bank live identity or receipt changed")
    expected = replay_arbitrary_plane_candidate_bank_v2(
        bank, pose_truth, final_realization, prepared_context
    )
    if acquisition._json_value(arbitrary_plane_candidate_bank_receipt_v2(bank)) != acquisition._json_value(
        arbitrary_plane_candidate_bank_receipt_v2(expected)
    ):
        raise ValueError("candidate bank deterministic receipt replay changed")
    for actual, replayed in zip(bank["candidates"], expected["candidates"], strict=True):
        if any(
            np.asarray(actual["arrays"][name]).dtype != np.asarray(replayed["arrays"][name]).dtype
            or not np.array_equal(actual["arrays"][name], replayed["arrays"][name])
            for name in _ARRAY_KEYS
        ):
            raise ValueError("candidate bank array replay changed")
