"""Pose- and tissue-blind fixed-canvas acquisition windows for arbitrary planes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

import training.arbitrary_plane_acquisition_v2 as acquisition


WINDOW_PLAN_V3_SCHEMA = "anatomy-tracker.acquisition-window-plan/v3"
WINDOW_REALIZATION_V3_SCHEMA = "anatomy-tracker.acquisition-window-realization/v3"
WINDOW_V3_ALGORITHM = "pose-tissue-blind-fixed-canvas-affine/v3"
DEFAULT_CANVAS_SHAPE_H_W = (192, 256)
DEFAULT_PARENT_SHAPE_H_W = (256, 256)
ASPECT_RATIO_RANGE = (0.75, 1.33)
SEVERITY_PROBABILITIES = {"standard": 0.70, "broad": 0.20, "stress": 0.10}
SEVERITY_RANGES = {
    "standard": {"content_scale": (0.9, 1.1), "offset": (-0.05, 0.05)},
    "broad": {"content_scale": (0.7, 1.3), "offset": (-0.15, 0.15)},
    "stress": {"content_scale": (0.5, 1.5), "offset": (-0.30, 0.30)},
}
ARRAY_ROLES = (
    "scalar",
    "annotation",
    "mask",
    "abstention-mask",
    "ccf-coordinate",
    "absolute-map-yx",
    "vector-yx",
)
VALIDITY_SUFFIX = "__valid_mask"
ABSTENTION_SUFFIX = "__abstention_mask"
RESERVED_ARRAY_NAMES = {
    "parent_sampling_domain_mask",
    "window_abstention_mask",
}
UPSTREAM_REALIZATION_ID_FIELDS = (
    "v2_plane_realization_id",
    "slab_render_id",
    "section_processing_render_id",
    "section_processing_plan_id",
    "section_processing_realization_id",
    "synthetic_section_processing_id",
    "subject_slab_render_id",
)
LINEAGE_FIELDS = (
    "split",
    "animal_index",
    "animal_id",
    "specimen_id",
    "experiment_id",
    "synthetic_animal_id",
    "section_index",
    "section_id",
)
_SOURCE = Path(__file__)


def _view_plan_identity(plan: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in plan.items()
        if key not in ("view_plan_id", "plan_receipt_sha256")
    }


def _uint64(value: int | str) -> int:
    if isinstance(value, str):
        if (
            len(value) != 18
            or not value.startswith("0x")
            or any(character not in "0123456789abcdef" for character in value[2:])
        ):
            raise ValueError("root_seed must be uint64 or 0x plus 16 lowercase hex digits")
        return int(value[2:], 16)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("root_seed must be uint64 or canonical hexadecimal text")
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("root_seed must fit uint64")
    return value


def _index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must fit uint64")
    return value


def derive_acquisition_window_seed_v3(
    root_seed: int | str,
    split: str,
    sample_index: int,
    field: str,
) -> int:
    """Use the frozen v2 stage/field seed derivation without pose or tissue."""
    if split not in ("train", "development") or not isinstance(field, str) or not field:
        raise ValueError("split or field is outside the frozen acquisition scope")
    return acquisition.derive_v2_field_seed(
        root_seed,
        split,
        _index(sample_index, "sample_index"),
        "window",
        field,
        0,
    )


def _field_rng(
    root_seed: int | str,
    split: str,
    sample_index: int,
    field: str,
) -> tuple[np.random.Generator, dict[str, object]]:
    seed = derive_acquisition_window_seed_v3(
        root_seed,
        split,
        sample_index,
        field,
    )
    return np.random.Generator(np.random.PCG64DXSM(seed)), {
        "stage": "window",
        "field": field,
        "attempt": 0,
        "seed_uint64": f"0x{seed:016x}",
        "generator": "NumPy PCG64DXSM",
    }


def sample_acquisition_window_plan_v3(
    *,
    root_seed: int | str,
    split: str,
    sample_index: int,
) -> dict[str, object]:
    """Sample a fixed-canvas view without accepting pose, tissue, or label inputs."""
    root_seed = _uint64(root_seed)
    if split not in ("train", "development"):
        raise ValueError("window plans are limited to train/development scope")
    sample_index = _index(sample_index, "sample_index")
    parent_height, parent_width = DEFAULT_PARENT_SHAPE_H_W
    canvas_height, canvas_width = DEFAULT_CANVAS_SHAPE_H_W
    fields = {}
    severity_rng, fields["window/plan-severity"] = _field_rng(
        root_seed,
        split,
        sample_index,
        "plan-severity",
    )
    smoke_root = int(acquisition._SMOKE_ROOT_SEED[2:], 16)
    if (
        root_seed == smoke_root
        and split == "development"
        and sample_index < len(acquisition.V2_SMOKE_ASSIGNMENTS)
    ):
        plan_severity = acquisition.V2_SMOKE_ASSIGNMENTS[sample_index][1]
        severity_policy = "frozen 20-case development smoke assignment; no caller override"
    else:
        severity_draw = float(severity_rng.random())
        plan_severity = (
            "standard"
            if severity_draw < 0.70
            else "broad"
            if severity_draw < 0.90
            else "stress"
        )
        severity_policy = "one frozen categorical draw; no caller override"
    values = {}
    for field in (
        "window/content-scale",
        "window/aspect",
        "window/offset-u",
        "window/offset-v",
    ):
        values[field], fields[field] = _field_rng(
            root_seed,
            split,
            sample_index,
            field.split("/", 1)[1],
        )
    bounds = SEVERITY_RANGES[plan_severity]
    content_scale = float(values["window/content-scale"].uniform(*bounds["content_scale"]))
    aspect_ratio = float(
        np.exp(values["window/aspect"].uniform(*np.log(ASPECT_RATIO_RANGE)))
    )
    offset_u = float(values["window/offset-u"].uniform(*bounds["offset"]))
    offset_v = float(values["window/offset-v"].uniform(*bounds["offset"]))
    scale_u = content_scale * np.sqrt(aspect_ratio)
    scale_v = content_scale / np.sqrt(aspect_ratio)
    canvas_to_parent = np.array(
        [
            [
                parent_width / (canvas_width * scale_u),
                0.0,
                parent_width * (0.5 + offset_u - 0.5 / scale_u),
            ],
            [
                0.0,
                parent_height / (canvas_height * scale_v),
                parent_height * (0.5 + offset_v - 0.5 / scale_v),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    parent_to_canvas = np.linalg.inv(canvas_to_parent)
    plan_seed = derive_acquisition_window_seed_v3(
        root_seed,
        split,
        sample_index,
        "view-plan",
    )
    plan = {
        "schema_version": WINDOW_PLAN_V3_SCHEMA,
        "algorithm": WINDOW_V3_ALGORITHM,
        "implementation_source_sha256": acquisition._normalized_text_sha256(_SOURCE),
        "preflight": acquisition._preflight_provenance(),
        "dependency_source_sha256": acquisition._source_hashes(),
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "view_plan_seed_uint64": f"0x{plan_seed:016x}",
        "plan_generated_without_pose_or_tissue": True,
        "plan_input_allowlist": [
            "root_seed_uint64",
            "split",
            "sample_index",
        ],
        "provenance": {
            "root_seed_uint64": f"0x{root_seed:016x}",
            "split": split,
            "sample_index": sample_index,
            "identity_exclusion": (
                "animal_id, specimen_id, experiment_id, artifact IDs, pose, tissue, "
                "support masks, projected bounds, and rejection outcomes never enter RNG"
            ),
        },
        "parent_shape_h_w": [parent_height, parent_width],
        "canvas_shape_h_w": [canvas_height, canvas_width],
        "plan_severity": plan_severity,
        "severity_selection_policy": severity_policy,
        "severity_probability": SEVERITY_PROBABILITIES[plan_severity],
        "content_scale": content_scale,
        "aspect_ratio": aspect_ratio,
        "axis_content_scale_u_v": [float(scale_u), float(scale_v)],
        "centre_offset_uv": [offset_u, offset_v],
        "canvas_to_parent_affine_float64": canvas_to_parent.tolist(),
        "parent_to_canvas_affine_float64": parent_to_canvas.tolist(),
        "effective_canvas_to_parent_affine_float32": canvas_to_parent[:2].astype(
            np.float32
        ).tolist(),
        "rng_sources": fields,
        "sampling_policy": (
            "one draw per named field at attempt zero, except the predeclared 20-case "
            "smoke severity assignment; no inspection, rejection, redraw, tight fit, or "
            "post-hoc repair"
        ),
    }
    plan["view_plan_id"] = acquisition._payload_sha256(_view_plan_identity(plan))
    plan["plan_receipt_sha256"] = acquisition._payload_sha256(plan)
    return plan


make_acquisition_window_plan_v3 = sample_acquisition_window_plan_v3


def replay_acquisition_window_plan_v3(plan: dict[str, object]) -> dict[str, object]:
    provenance = plan["provenance"]
    return sample_acquisition_window_plan_v3(
        root_seed=provenance["root_seed_uint64"],
        split=provenance["split"],
        sample_index=provenance["sample_index"],
    )


def verify_acquisition_window_plan_v3(plan: dict[str, object]) -> None:
    replay = replay_acquisition_window_plan_v3(plan)
    if plan != replay:
        raise ValueError("acquisition-window plan does not replay exactly")
    canvas_to_parent = np.asarray(plan["canvas_to_parent_affine_float64"])
    parent_to_canvas = np.asarray(plan["parent_to_canvas_affine_float64"])
    if (
        np.linalg.det(canvas_to_parent[:2, :2]) <= 0.0
        or not np.allclose(canvas_to_parent @ parent_to_canvas, np.eye(3), atol=1.0e-12)
    ):
        raise ValueError("acquisition-window affine is not positive and invertible")


def _sampling_grid(plan: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas_height, canvas_width = plan["canvas_shape_h_w"]
    parent_height, parent_width = plan["parent_shape_h_w"]
    y, x = np.indices((canvas_height, canvas_width), dtype=np.float64)
    affine = np.asarray(plan["canvas_to_parent_affine_float64"], dtype=np.float64)
    parent_x = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]
    parent_y = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]
    inside = (
        (parent_x >= 0.0)
        & (parent_x <= parent_width - 1)
        & (parent_y >= 0.0)
        & (parent_y <= parent_height - 1)
    )
    return parent_y, parent_x, inside


def _validate_source_array(
    name: str,
    array: np.ndarray,
    role: str,
    parent_shape: tuple[int, int],
) -> np.ndarray:
    if (
        not isinstance(name, str)
        or not name
        or name in RESERVED_ARRAY_NAMES
        or name.endswith(VALIDITY_SUFFIX)
        or name.endswith(ABSTENTION_SUFFIX)
    ):
        raise ValueError("source array name is empty or reserved")
    array = np.asarray(array)
    if array.shape[:2] != parent_shape:
        raise ValueError("source-array parent shapes do not match the frozen plan")
    if role == "scalar":
        valid_role = array.ndim >= 2 and np.issubdtype(array.dtype, np.floating)
    elif role == "annotation":
        valid_role = array.ndim == 2 and np.issubdtype(array.dtype, np.integer)
    elif role in ("mask", "abstention-mask"):
        valid_role = array.ndim == 2 and array.dtype == bool
    elif role == "ccf-coordinate":
        valid_role = array.shape == parent_shape + (3,) and np.issubdtype(
            array.dtype, np.floating
        )
    elif role in ("absolute-map-yx", "vector-yx"):
        valid_role = array.shape == parent_shape + (2,) and np.issubdtype(
            array.dtype, np.floating
        )
    else:
        valid_role = False
    if not valid_role:
        raise ValueError(f"source array {name!r} does not satisfy role {role!r}")
    return array


def _map_coordinates(
    array: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray],
    order: int,
    exterior: int | float,
) -> np.ndarray:
    output = np.empty(grid[0].shape + array.shape[2:], dtype=array.dtype)
    if array.ndim == 2:
        output = ndimage.map_coordinates(
            array, grid, order=order, mode="constant", cval=exterior, prefilter=False
        )
    else:
        for index in np.ndindex(array.shape[2:]):
            output[(slice(None), slice(None)) + index] = ndimage.map_coordinates(
                array[(slice(None), slice(None)) + index],
                grid,
                order=order,
                mode="constant",
                cval=exterior,
                prefilter=False,
            )
    return output


def _resample(
    array: np.ndarray,
    validity: np.ndarray,
    role: str,
    grid: tuple[np.ndarray, np.ndarray],
    inside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = 0 if role in ("annotation", "mask", "abstention-mask") else 1
    finite = (
        np.isfinite(array).all(axis=tuple(range(2, array.ndim)))
        if array.ndim > 2
        else np.isfinite(array)
    )
    source_valid = np.asarray(validity, dtype=bool) & finite
    expanded_valid = source_valid[(...,) + (None,) * (array.ndim - 2)]
    exterior = 1 if role == "abstention-mask" else 0
    safe = np.where(expanded_valid, array, exterior).astype(array.dtype, copy=False)
    sampled = _map_coordinates(safe, grid, order, exterior)
    sampled_validity = ndimage.map_coordinates(
        source_valid.astype(np.float32),
        grid,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    threshold = 0.5 if order == 0 else 1.0 - 8.0 * np.finfo(np.float32).eps
    output_valid = inside & (sampled_validity >= threshold)
    output_finite = (
        np.isfinite(sampled).all(axis=tuple(range(2, sampled.ndim)))
        if sampled.ndim > 2
        else np.isfinite(sampled)
    )
    output_valid &= output_finite
    if role in ("mask", "abstention-mask"):
        sampled = sampled.astype(bool)
    invalid_fill = True if role == "abstention-mask" else 0
    sampled[~output_valid] = invalid_fill
    return np.ascontiguousarray(sampled, dtype=array.dtype), np.ascontiguousarray(
        output_valid
    )


def _transform_map_values(
    array: np.ndarray,
    validity: np.ndarray,
    role: str,
    plan: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
    if role not in ("absolute-map-yx", "vector-yx"):
        return array, validity
    inverse = np.asarray(plan["parent_to_canvas_affine_float64"], dtype=np.float64)
    parent_xy = np.stack((array[..., 1], array[..., 0]), axis=-1).astype(np.float64)
    if role == "absolute-map-yx":
        canvas_xy = parent_xy @ inverse[:2, :2].T + inverse[:2, 2]
    else:
        canvas_xy = parent_xy @ inverse[:2, :2].T
    transformed = np.stack((canvas_xy[..., 1], canvas_xy[..., 0]), axis=-1)
    validity = validity & np.isfinite(transformed).all(axis=-1)
    if role == "absolute-map-yx":
        canvas_height, canvas_width = plan["canvas_shape_h_w"]
        validity &= (
            (transformed[..., 1] >= 0.0)
            & (transformed[..., 1] <= canvas_width - 1)
            & (transformed[..., 0] >= 0.0)
            & (transformed[..., 0] <= canvas_height - 1)
        )
    transformed[~validity] = 0.0
    return np.ascontiguousarray(transformed, dtype=array.dtype), np.ascontiguousarray(
        validity
    )


def _retained_fraction(mask: np.ndarray, plan: dict[str, object]) -> float:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return 0.0
    y, x = np.indices(mask.shape, dtype=np.float64)
    inverse = np.asarray(plan["parent_to_canvas_affine_float64"], dtype=np.float64)
    canvas_x = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
    canvas_y = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
    canvas_height, canvas_width = plan["canvas_shape_h_w"]
    retained = (
        (canvas_x >= 0.0)
        & (canvas_x <= canvas_width - 1)
        & (canvas_y >= 0.0)
        & (canvas_y <= canvas_height - 1)
    )
    return float((mask & retained).sum() / count)


def _retained_mass_fraction(mass: np.ndarray, plan: dict[str, object]) -> float:
    mass = np.asarray(mass, dtype=np.float64)
    total = float(mass.sum())
    if total == 0.0:
        return 0.0
    y, x = np.indices(mass.shape, dtype=np.float64)
    inverse = np.asarray(plan["parent_to_canvas_affine_float64"], dtype=np.float64)
    canvas_x = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
    canvas_y = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
    canvas_height, canvas_width = plan["canvas_shape_h_w"]
    retained = (
        (canvas_x >= 0.0)
        & (canvas_x <= canvas_width - 1)
        & (canvas_y >= 0.0)
        & (canvas_y <= canvas_height - 1)
    )
    return float(mass[retained].sum() / total)


def _byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )


def _window_status(retained_fraction: float) -> str:
    if retained_fraction >= 0.9:
        return "near_full"
    if retained_fraction >= 0.7:
        return "mild_partial"
    if retained_fraction >= 0.4:
        return "severe_ambiguity"
    return "below_0_4_abstention_ood_stress"


def _effective_ouv(
    design_quicknii_ouv: np.ndarray, plan: dict[str, object]
) -> np.ndarray:
    design = np.asarray(design_quicknii_ouv, dtype=np.float64)
    if design.shape != (3, 3) or not np.isfinite(design).all():
        raise ValueError("design_quicknii_ouv must contain finite O/U/V rows")
    origin, edge_u, edge_v = design
    parent_height, parent_width = plan["parent_shape_h_w"]
    canvas_height, canvas_width = plan["canvas_shape_h_w"]
    affine = np.asarray(plan["canvas_to_parent_affine_float64"], dtype=np.float64)
    return np.ascontiguousarray(
        np.stack(
            (
                origin
                + (affine[0, 2] / parent_width) * edge_u
                + (affine[1, 2] / parent_height) * edge_v,
                (canvas_width * affine[0, 0] / parent_width) * edge_u
                + (canvas_width * affine[1, 0] / parent_height) * edge_v,
                (canvas_height * affine[0, 1] / parent_width) * edge_u
                + (canvas_height * affine[1, 1] / parent_height) * edge_v,
            )
        )
    )


def _validate_global_reference_geometry(
    global_reference_fov_uv_um: tuple[float, float],
    design_quicknii_ouv: np.ndarray,
    plan: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    fov = np.asarray(global_reference_fov_uv_um, dtype=np.float64)
    design = np.ascontiguousarray(design_quicknii_ouv, dtype=np.float64)
    if (
        fov.shape != (2,)
        or not np.isfinite(fov).all()
        or np.any(fov <= 0.0)
        or design.shape != (3, 3)
        or not np.isfinite(design).all()
    ):
        raise ValueError("physical reference requires two finite positive FOV values")
    parent_height, parent_width = plan["parent_shape_h_w"]
    expected = np.array(
        [
            fov[0] * parent_width / (parent_width - 1),
            fov[1] * parent_height / (parent_height - 1),
        ],
        dtype=np.float64,
    )
    actual = np.array(
        [np.linalg.norm(design[1]), np.linalg.norm(design[2])], dtype=np.float64
    )
    area_scale = actual[0] * actual[1]
    if (
        not np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-9)
        or float(np.linalg.norm(np.cross(design[1], design[2])))
        <= np.finfo(np.float64).eps * max(area_scale, 1.0)
    ):
        raise ValueError(
            "design O/U/V does not match its physical FOV endpoints or is degenerate"
        )
    return fov, design


def transform_parent_landmarks_yx_v3(
    plan: dict[str, object], parent_landmarks_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    landmarks = np.asarray(parent_landmarks_yx, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or not np.isfinite(landmarks).all():
        raise ValueError("parent landmarks must be a finite N-by-2 y-x array")
    inverse = np.asarray(plan["parent_to_canvas_affine_float64"], dtype=np.float64)
    parent_xy = landmarks[:, ::-1]
    canvas_xy = parent_xy @ inverse[:2, :2].T + inverse[:2, 2]
    transformed = canvas_xy[:, ::-1]
    canvas_height, canvas_width = plan["canvas_shape_h_w"]
    validity = (
        (transformed[:, 1] >= 0.0)
        & (transformed[:, 1] <= canvas_width - 1)
        & (transformed[:, 0] >= 0.0)
        & (transformed[:, 0] <= canvas_height - 1)
    )
    transformed[~validity] = 0.0
    return np.ascontiguousarray(transformed), np.ascontiguousarray(validity)


def acquisition_window_realization_receipt_v3(
    artifact: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "window_plan": artifact["window_plan"],
        "source_binding": artifact["source_binding"],
        "lineage": artifact["lineage"],
        "transform": artifact["transform"],
        "retention_audit": artifact["retention_audit"],
        "array_receipts": artifact["array_receipts"],
        "landmark_receipts": artifact["landmark_receipts"],
        "landmark_validity_receipts": artifact["landmark_validity_receipts"],
        "acquisition_window_realization_id": artifact[
            "acquisition_window_realization_id"
        ],
    }


def apply_acquisition_window_v3(
    plan: dict[str, object],
    source_arrays: dict[str, np.ndarray],
    array_roles: dict[str, str],
    *,
    source_validity: dict[str, np.ndarray],
    global_reference_grid_id: str,
    global_reference_fov_uv_um: tuple[float, float],
    design_quicknii_ouv: np.ndarray,
    centre_plane_support_mask: np.ndarray,
    optical_slab_support_mass: np.ndarray,
    upstream_realization_ids: dict[str, str],
    section_processing_receipt: dict[str, object],
    section_processing_receipt_sha256: str,
    lineage: dict[str, object],
    parent_landmarks_yx: dict[str, np.ndarray],
) -> dict[str, object]:
    """Apply one already-frozen plan coherently, without rejection or redraw."""
    verify_acquisition_window_plan_v3(plan)
    parent_shape = tuple(plan["parent_shape_h_w"])
    if (
        not source_arrays
        or set(source_arrays) != set(array_roles)
        or set(source_arrays) != set(source_validity)
        or any(
        role not in ARRAY_ROLES for role in array_roles.values()
        )
    ):
        raise ValueError("every source array needs one declared role and validity raster")
    validated_arrays = {}
    validated_validity = {}
    for name, array in source_arrays.items():
        validated_arrays[name] = _validate_source_array(
            name, array, array_roles[name], parent_shape
        )
        validity = np.asarray(source_validity[name])
        if validity.shape != parent_shape or validity.dtype != bool:
            raise ValueError("source validity rasters must be Boolean parent rasters")
        validated_validity[name] = np.ascontiguousarray(validity)
    centre_support = np.asarray(centre_plane_support_mask)
    optical_support = np.asarray(optical_slab_support_mass)
    if (
        centre_support.shape != parent_shape
        or optical_support.shape != parent_shape
        or centre_support.dtype != bool
        or not np.issubdtype(optical_support.dtype, np.floating)
        or not np.isfinite(optical_support).all()
        or np.any(optical_support < 0.0)
    ):
        raise ValueError("centre support must be Boolean and optical support finite mass")
    if not isinstance(global_reference_grid_id, str) or not global_reference_grid_id:
        raise ValueError("global_reference_grid_id must be nonempty")
    fov, design = _validate_global_reference_geometry(
        global_reference_fov_uv_um, design_quicknii_ouv, plan
    )
    if (
        set(upstream_realization_ids) != set(UPSTREAM_REALIZATION_ID_FIELDS)
        or any(
            not isinstance(value, str) or not value
            for value in upstream_realization_ids.values()
        )
    ):
        raise ValueError("all frozen upstream realization IDs are required")
    if (
        not isinstance(section_processing_receipt, dict)
        or not section_processing_receipt
        or not isinstance(section_processing_receipt_sha256, str)
        or len(section_processing_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in section_processing_receipt_sha256)
        or acquisition._payload_sha256(section_processing_receipt)
        != section_processing_receipt_sha256
        or section_processing_receipt.get("section_processing_render_id")
        != upstream_realization_ids["section_processing_render_id"]
        or section_processing_receipt.get("receipt_sha256")
        != acquisition._payload_sha256(
            {
                key: value
                for key, value in section_processing_receipt.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise ValueError("section-processing receipt is not authenticated")
    if set(lineage) != set(LINEAGE_FIELDS):
        raise ValueError("authenticated lineage is incomplete")
    bound_lineage = {name: acquisition._json_value(lineage[name]) for name in LINEAGE_FIELDS}
    if (
        bound_lineage["split"] != plan["provenance"]["split"]
        or _index(bound_lineage["animal_index"], "animal_index")
        != bound_lineage["animal_index"]
        or _index(bound_lineage["section_index"], "section_index")
        != bound_lineage["section_index"]
        or any(
            bound_lineage[name] is None
            or (isinstance(bound_lineage[name], str) and not bound_lineage[name])
            for name in (
                "animal_id",
                "specimen_id",
                "experiment_id",
                "synthetic_animal_id",
                "section_id",
            )
        )
    ):
        raise ValueError("authenticated lineage does not match the window plan")
    if not isinstance(parent_landmarks_yx, dict):
        raise ValueError("parent_landmarks_yx must be a named mapping")
    parent_y, parent_x, inside = _sampling_grid(plan)
    arrays = {}
    window_abstention = ~inside.copy()
    for name, array in validated_arrays.items():
        role = array_roles[name]
        sampled, output_valid = _resample(
            array,
            validated_validity[name],
            role,
            (parent_y, parent_x),
            inside,
        )
        sampled, output_valid = _transform_map_values(
            sampled, output_valid, role, plan
        )
        if role == "abstention-mask":
            sampled[~output_valid] = True
        else:
            sampled[~output_valid] = 0
        arrays[name] = np.ascontiguousarray(sampled)
        arrays[name + VALIDITY_SUFFIX] = np.ascontiguousarray(output_valid)
        arrays[name + ABSTENTION_SUFFIX] = np.ascontiguousarray(~output_valid)
        window_abstention |= ~output_valid
    arrays["parent_sampling_domain_mask"] = np.ascontiguousarray(inside)
    arrays["window_abstention_mask"] = np.ascontiguousarray(window_abstention)
    effective = _effective_ouv(design, plan)
    landmarks = {}
    landmark_validity = {}
    for name, parent_landmarks in parent_landmarks_yx.items():
        if (
            not isinstance(name, str)
            or not name
            or name in RESERVED_ARRAY_NAMES
            or name.endswith(VALIDITY_SUFFIX)
            or name.endswith(ABSTENTION_SUFFIX)
        ):
            raise ValueError("landmark name is empty or reserved")
        landmarks[name], landmark_validity[name] = transform_parent_landmarks_yx_v3(
            plan, parent_landmarks
        )
    centre_fraction = _retained_fraction(centre_support, plan)
    optical_fraction = _retained_mass_fraction(optical_support, plan)
    source_receipts = {
        name: acquisition._array_receipt(array) for name, array in validated_arrays.items()
    }
    source_validity_receipts = {
        name: acquisition._array_receipt(validity)
        for name, validity in validated_validity.items()
    }
    output_receipts = {
        name: acquisition._array_receipt(array) for name, array in arrays.items()
    }
    artifact = {
        "schema_version": WINDOW_REALIZATION_V3_SCHEMA,
        "algorithm": WINDOW_V3_ALGORITHM,
        "window_plan": plan,
        "source_binding": {
            "upstream_realization_ids": dict(upstream_realization_ids),
            "section_processing_receipt": section_processing_receipt,
            "section_processing_receipt_sha256": section_processing_receipt_sha256,
            "global_reference_grid_id": global_reference_grid_id,
            "global_reference_fov_uv_um": fov.tolist(),
            "design_quicknii_ouv": design.tolist(),
            "design_quicknii_ouv_receipt": acquisition._array_receipt(design),
            "centre_plane_support_receipt": acquisition._array_receipt(centre_support),
            "optical_slab_support_mass_receipt": acquisition._array_receipt(optical_support),
            "source_array_roles": dict(array_roles),
            "source_array_receipts": source_receipts,
            "source_validity_receipts": source_validity_receipts,
        },
        "lineage": bound_lineage,
        "transform": {
            "canvas_shape_h_w": list(plan["canvas_shape_h_w"]),
            "canvas_to_parent_affine_float64": plan[
                "canvas_to_parent_affine_float64"
            ],
            "parent_to_canvas_affine_float64": plan[
                "parent_to_canvas_affine_float64"
            ],
            "effective_canvas_to_parent_affine_float32": plan[
                "effective_canvas_to_parent_affine_float32"
            ],
            "design_quicknii_ouv": design.tolist(),
            "effective_quicknii_ouv": effective.tolist(),
            "interpolation": {
                "scalar": "bilinear-zero",
                "annotation": "nearest-zero integer",
                "mask": "nearest-zero Boolean",
                "abstention-mask": "nearest with true exterior",
                "ccf-coordinate": "bilinear with explicit invalid zero exterior",
                "absolute-map-yx": "bilinear then T^-1(F_parent(Tx))",
                "vector-yx": "bilinear then inverse-linear vector conjugation",
            },
        },
        "retention_audit": {
            "retained_centre_plane_support_fraction": centre_fraction,
            "retained_optical_slab_support_fraction": optical_fraction,
            "window_status": _window_status(centre_fraction),
            "policy": "post-hoc audit only; never reject, redraw, repair, or reweight",
        },
        "landmarks_yx": landmarks,
        "landmark_validity": landmark_validity,
        "arrays": arrays,
        "array_receipts": output_receipts,
        "landmark_receipts": {
            name: acquisition._array_receipt(value) for name, value in landmarks.items()
        },
        "landmark_validity_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in landmark_validity.items()
        },
    }
    artifact["acquisition_window_realization_id"] = acquisition._payload_sha256(
        {
            key: value
            for key, value in acquisition_window_realization_receipt_v3(
                {**artifact, "acquisition_window_realization_id": "pending"}
            ).items()
            if key != "acquisition_window_realization_id"
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        acquisition_window_realization_receipt_v3(artifact)
    )
    return artifact


make_acquisition_window_realization_v3 = apply_acquisition_window_v3


def verify_acquisition_window_realization_v3(
    artifact: dict[str, object],
    source_arrays: dict[str, np.ndarray],
    array_roles: dict[str, str],
    *,
    source_validity: dict[str, np.ndarray],
    global_reference_grid_id: str,
    global_reference_fov_uv_um: tuple[float, float],
    design_quicknii_ouv: np.ndarray,
    centre_plane_support_mask: np.ndarray,
    optical_slab_support_mass: np.ndarray,
    upstream_realization_ids: dict[str, str],
    section_processing_receipt: dict[str, object],
    section_processing_receipt_sha256: str,
    lineage: dict[str, object],
    parent_landmarks_yx: dict[str, np.ndarray],
) -> None:
    replay = replay_acquisition_window_realization_v3(
        artifact["window_plan"],
        source_arrays,
        array_roles,
        source_validity=source_validity,
        global_reference_grid_id=global_reference_grid_id,
        global_reference_fov_uv_um=global_reference_fov_uv_um,
        design_quicknii_ouv=design_quicknii_ouv,
        centre_plane_support_mask=centre_plane_support_mask,
        optical_slab_support_mass=optical_slab_support_mass,
        upstream_realization_ids=upstream_realization_ids,
        section_processing_receipt=section_processing_receipt,
        section_processing_receipt_sha256=section_processing_receipt_sha256,
        lineage=lineage,
        parent_landmarks_yx=parent_landmarks_yx,
    )
    if (
        set(artifact) != set(replay)
        or artifact.get("receipt_sha256") != replay["receipt_sha256"]
        or acquisition_window_realization_receipt_v3(artifact)
        != acquisition_window_realization_receipt_v3(replay)
        or set(artifact.get("arrays", {})) != set(replay["arrays"])
        or any(
            not _byte_equal(artifact["arrays"][name], replay["arrays"][name])
            for name in replay["arrays"]
        )
        or set(artifact.get("landmarks_yx", {})) != set(replay["landmarks_yx"])
        or any(
            not _byte_equal(artifact["landmarks_yx"][name], replay["landmarks_yx"][name])
            for name in replay["landmarks_yx"]
        )
        or set(artifact.get("landmark_validity", {}))
        != set(replay["landmark_validity"])
        or any(
            not _byte_equal(
                artifact["landmark_validity"][name], replay["landmark_validity"][name]
            )
            for name in replay["landmark_validity"]
        )
    ):
        raise ValueError("acquisition-window realization does not replay exactly")


def replay_acquisition_window_realization_v3(
    plan: dict[str, object],
    source_arrays: dict[str, np.ndarray],
    array_roles: dict[str, str],
    *,
    source_validity: dict[str, np.ndarray],
    global_reference_grid_id: str,
    global_reference_fov_uv_um: tuple[float, float],
    design_quicknii_ouv: np.ndarray,
    centre_plane_support_mask: np.ndarray,
    optical_slab_support_mass: np.ndarray,
    upstream_realization_ids: dict[str, str],
    section_processing_receipt: dict[str, object],
    section_processing_receipt_sha256: str,
    lineage: dict[str, object],
    parent_landmarks_yx: dict[str, np.ndarray],
) -> dict[str, object]:
    return apply_acquisition_window_v3(
        plan,
        source_arrays,
        array_roles,
        source_validity=source_validity,
        global_reference_grid_id=global_reference_grid_id,
        global_reference_fov_uv_um=global_reference_fov_uv_um,
        design_quicknii_ouv=design_quicknii_ouv,
        centre_plane_support_mask=centre_plane_support_mask,
        optical_slab_support_mass=optical_slab_support_mass,
        upstream_realization_ids=upstream_realization_ids,
        section_processing_receipt=section_processing_receipt,
        section_processing_receipt_sha256=section_processing_receipt_sha256,
        lineage=lineage,
        parent_landmarks_yx=parent_landmarks_yx,
    )
