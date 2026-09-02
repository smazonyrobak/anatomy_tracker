"""Authenticated variable finite-thickness observations for verified v3 planes.

The adapter never samples or redraws pose.  It applies an independently
receipted, normalized through-plane boxcar PSF to a verified finite-FOV v3
parent while retaining the parent's centre-plane pose, labels, and CCF
coordinates as the authoritative targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_rendered_generator as parent_generator
from training.arbitrary_plane_geometry import (
    QUICKNII_RASTER_INDEX_SAMPLING,
    allen_index_to_physical_um_vectors,
    physical_um_to_allen_index_vectors,
    render_arbitrary_plane,
)
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
    verify_finite_arbitrary_plane_render,
)


FINITE_SLAB_V4_SCHEMA = "anatomy-tracker.authenticated-finite-thickness-slab-render/v4"
FINITE_SLAB_V4_ALGORITHM = "verified-v3-parent-physical-normal-boxcar/v4"
FINITE_PSF_V4_SCHEMA = psf_v4.FINITE_PSF_V4_SCHEMA
FINITE_PSF_CAPABILITY_V4_SCHEMA = psf_v4.FINITE_PSF_CAPABILITY_V4_SCHEMA
SLAB_OBSERVATION_V4_SCHEMA = "anatomy-tracker.slab-observation/v4"
FINITE_BOXCAR = "finite_boxcar"
CENTRE_PLANE_ABLATION = "centre_plane_ablation"
PRODUCTION_THICKNESS_RANGE_UM = psf_v4.PRODUCTION_THICKNESS_RANGE_UM
PRODUCTION_AXIAL_SAMPLE_COUNT = psf_v4.PRODUCTION_AXIAL_SAMPLE_COUNT
PRODUCTION_AXIAL_STEP_UM_MAX = psf_v4.AXIAL_STEP_UM_MAX
PHYSICAL_DISPLACEMENT_TOLERANCE_UM = 0.01
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


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    array = np.asarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "array_sha256": _array_sha256(array),
    }


def _source_commit(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Generator source commit must be a 40-digit Git SHA or None")
    return value


def _uint64_seed(seed: int) -> int:
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("Thickness seed must fit an unsigned 64-bit integer")
    return seed


def _seed_hex(seed: int) -> str:
    return f"0x{_uint64_seed(seed):016x}"


def _parse_seed_hex(seed: object) -> int:
    if not isinstance(seed, str) or re.fullmatch(r"0x[0-9a-f]{16}", seed) is None:
        raise ValueError("Thickness seed must use canonical lowercase uint64 hexadecimal encoding")
    return int(seed, 16)


_SOURCE_ROOT = Path(__file__).parent
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_DEPENDENCY_FILES = (
    "arbitrary_plane_geometry.py",
    "arbitrary_plane_psf_v4.py",
    "arbitrary_plane_rendered_generator.py",
)
_LOADED_DEPENDENCY_SOURCE_SHA256 = {
    name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
    for name in _DEPENDENCY_FILES
}


def finite_psf_capability_v4() -> dict[str, object]:
    return psf_v4.finite_psf_model_capability_v4()


def _resolve_thickness_selection(
    render_mode: str,
    thickness_seed: int | None,
    nominal_cut_thickness_um: float | None,
) -> dict[str, object]:
    if render_mode == CENTRE_PLANE_ABLATION:
        if thickness_seed is not None or (
            nominal_cut_thickness_um is not None and float(nominal_cut_thickness_um) != 0.0
        ):
            raise ValueError("Centre-plane ablation requires zero thickness and no thickness seed")
        payload = {
            "selection_mode": "zero-thickness-ablation",
            "seed_encoding": UINT64_SEED_ENCODING,
            "thickness_seed_uint64": None,
            "draw_fraction": None,
            "distribution": "point mass at zero; excluded from production thickness sampling",
            "production_thickness_range_um": list(PRODUCTION_THICKNESS_RANGE_UM),
            "nominal_cut_thickness_um": 0.0,
        }
    elif render_mode == FINITE_BOXCAR:
        if (thickness_seed is None) == (nominal_cut_thickness_um is None):
            raise ValueError("Finite boxcar requires exactly one of thickness_seed or explicit thickness")
        if thickness_seed is not None:
            seed = _uint64_seed(thickness_seed)
            fraction = float(np.random.Generator(np.random.PCG64(seed)).random())
            lower, upper = PRODUCTION_THICKNESS_RANGE_UM
            thickness = lower + (upper - lower) * fraction
            payload = {
                "selection_mode": "independent-seeded-uniform",
                "seed_encoding": UINT64_SEED_ENCODING,
                "thickness_seed_uint64": _seed_hex(seed),
                "draw_fraction": fraction,
                "distribution": "uniform continuous over the inclusive declared capability interval",
                "production_thickness_range_um": [lower, upper],
                "nominal_cut_thickness_um": thickness,
            }
        else:
            thickness = float(nominal_cut_thickness_um)
            lower, upper = PRODUCTION_THICKNESS_RANGE_UM
            if not math.isfinite(thickness) or not lower <= thickness <= upper:
                raise ValueError("Explicit finite thickness must lie in [25,100] um")
            payload = {
                "selection_mode": "explicit-receipted-thickness",
                "seed_encoding": UINT64_SEED_ENCODING,
                "thickness_seed_uint64": None,
                "draw_fraction": None,
                "distribution": "explicit caller-supplied thickness within declared capability",
                "production_thickness_range_um": [lower, upper],
                "nominal_cut_thickness_um": thickness,
            }
    else:
        raise ValueError("Unknown finite-slab render mode")
    return {**payload, "thickness_selection_sha256": _payload_sha256(payload)}


def finite_psf_v4(
    render_mode: str,
    nominal_cut_thickness_um: float,
    *,
    thickness_selection_sha256: str,
) -> dict[str, object]:
    """Build the exact fixed-S production schedule or zero-thickness ablation."""
    contract = psf_v4.make_finite_psf_schedule_v4(
        render_mode,
        nominal_cut_thickness_um,
        thickness_selection_sha256=thickness_selection_sha256,
    )
    psf_v4.verify_finite_psf_schedule_v4(
        contract,
        capability=finite_psf_capability_v4(),
    )
    return contract


def _reduce_samples(
    scalar_samples: np.ndarray,
    annotation_samples: np.ndarray,
    integer_masses: np.ndarray,
    centre_index: int,
) -> dict[str, np.ndarray]:
    scalar = np.asarray(scalar_samples)
    annotation = np.asarray(annotation_samples)
    masses = np.asarray(integer_masses)
    if (
        scalar.ndim != 3
        or scalar.dtype != np.dtype(np.float32)
        or annotation.shape != scalar.shape
        or annotation.dtype != np.dtype(np.int64)
        or masses.shape != (scalar.shape[0],)
        or masses.dtype != np.dtype(np.int64)
        or np.any(masses <= 0)
        or not np.isfinite(scalar).all()
        or not 0 <= int(centre_index) < scalar.shape[0]
    ):
        raise ValueError("Finite-slab samples have invalid shape, dtype, or masses")
    total_mass = int(masses.sum())
    scalar_accumulator = np.zeros(scalar.shape[1:], dtype=np.float64)
    occupancy_mass = np.zeros(scalar.shape[1:], dtype=np.int64)
    for sample_index, mass in enumerate(masses):
        scalar_accumulator += int(mass) * scalar[sample_index].astype(np.float64)
        occupancy_mass += int(mass) * (annotation[sample_index] != 0)
    centre_annotation = annotation[int(centre_index)].copy()
    centre_support = centre_annotation != 0
    winning_mass = np.full(scalar.shape[1:], -1, dtype=np.int64)
    modal_annotation = np.full(scalar.shape[1:], np.iinfo(np.int64).max, dtype=np.int64)
    for candidate_index in range(annotation.shape[0]):
        candidate = annotation[candidate_index]
        candidate_mass = np.zeros(scalar.shape[1:], dtype=np.int64)
        for sample_index, mass in enumerate(masses):
            candidate_mass += int(mass) * (annotation[sample_index] == candidate)
        update = (candidate_mass > winning_mass) | (
            (candidate_mass == winning_mass) & (candidate < modal_annotation)
        )
        winning_mass[update] = candidate_mass[update]
        modal_annotation[update] = candidate[update]
    centre_label_mass = np.zeros(scalar.shape[1:], dtype=np.int64)
    for sample_index, mass in enumerate(masses):
        centre_label_mass += int(mass) * (annotation[sample_index] == centre_annotation)
    centre_label_psf_mass = (centre_label_mass / total_mass).astype(np.float32)
    dense_weight = np.where(
        centre_support,
        np.clip((centre_label_psf_mass.astype(np.float64) - 0.5) / 0.3, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    return {
        "observed_scalar_float32": (scalar_accumulator / total_mass).astype(np.float32),
        "centre_plane_annotation_int64": centre_annotation,
        "centre_plane_support_mask": centre_support,
        "slab_brain_occupancy_float32": (occupancy_mass / total_mass).astype(np.float32),
        "slab_observable_support_mask": occupancy_mass > 0,
        "slab_modal_annotation_int64": modal_annotation,
        "slab_modal_purity_float32": (winning_mass / total_mass).astype(np.float32),
        "centre_label_psf_mass_float32": centre_label_psf_mass,
        "dense_correspondence_weight_float32": dense_weight,
        "dense_correspondence_abstention_mask": (~centre_support) | (centre_label_psf_mass <= 0.5),
    }


def _offset_error_components(
    observed: np.ndarray,
    expected: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, float, float]:
    error = observed - expected
    axial = error @ normal
    tangential = error - axial[..., None] * normal
    return (
        float(np.max(np.abs(error))),
        float(np.max(np.abs(axial))),
        float(np.max(np.linalg.norm(tangential, axis=-1))),
    )


def _render_samples(
    parent: dict[str, object],
    context: dict[str, object],
    finite_psf: dict[str, object],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    geometry = parent["geometry"]
    support = context["support_index"]
    spacing = np.asarray(support["voxel_size_um"], dtype=np.float64)
    atlas_origin = np.asarray(support["origin_um"], dtype=np.float64)
    atlas_shape = tuple(int(value) for value in support["annotation_shape"])
    normal = np.asarray(geometry["normal_rp2_ap_dv_ml"], dtype=np.float64)
    offsets = np.asarray(finite_psf["axial_offsets_um"], dtype=np.float64)
    masses = np.asarray(finite_psf["axial_integer_masses"], dtype=np.int64)
    base_center = torch.as_tensor(geometry["renderer_center_ap_dv_ml"], dtype=torch.float32)
    frame = torch.as_tensor(geometry["renderer_frame_ap_dv_ml"], dtype=torch.float32)
    basis = torch.as_tensor(geometry["renderer_inplane_basis"], dtype=torch.float32)
    base_geometry = dict(geometry)
    base_geometry["renderer_center_ap_dv_ml"] = base_center.tolist()
    base_grid = effective_renderer_sampling_arrays(
        base_geometry,
        atlas_shape,
        origin_ap_dv_ml_um=tuple(atlas_origin),
        voxel_size_ap_dv_ml_um=tuple(spacing),
    )
    base_points = base_grid["coordinate_raster_allen_index_float32"].astype(np.float64)
    centre_ccf_um = np.ascontiguousarray(
        (atlas_origin + (base_points + 0.5) * spacing).astype(np.float32)
    )
    centre_scalar = np.asarray(parent["raster"]["scalar"])
    centre_annotation = np.asarray(parent["raster"]["annotation"])
    scalar_samples: list[np.ndarray] = []
    annotation_samples: list[np.ndarray] = []
    offset_receipts: list[dict[str, object]] = []
    maxima = np.zeros(6, dtype=np.float64)
    centre_indices = np.flatnonzero(offsets == 0.0)
    if centre_indices.size != 1:
        raise ValueError("Finite PSF must contain exactly one zero offset")
    centre_index = int(centre_indices[0])
    sample_specs: list[dict[str, object]] = []
    nonzero_centers: list[torch.Tensor] = []
    for offset_index, (offset, mass) in enumerate(zip(offsets, masses)):
        design_displacement = float(offset) * normal
        delta_index = physical_um_to_allen_index_vectors(
            torch.as_tensor(design_displacement, dtype=torch.float64), tuple(spacing)
        ).to(torch.float32)
        shifted_center = base_center + delta_index
        effective_delta_index = shifted_center - base_center
        effective_displacement = allen_index_to_physical_um_vectors(
            effective_delta_index.to(torch.float64), tuple(spacing)
        ).numpy()
        shifted_geometry = dict(geometry)
        shifted_geometry["renderer_center_ap_dv_ml"] = shifted_center.tolist()
        shifted_grid = effective_renderer_sampling_arrays(
            shifted_geometry,
            atlas_shape,
            origin_ap_dv_ml_um=tuple(atlas_origin),
            voxel_size_ap_dv_ml_um=tuple(spacing),
        )
        grid_displacement = (
            shifted_grid["coordinate_raster_allen_index_float32"].astype(np.float64) - base_points
        ) * spacing
        centre_errors = _offset_error_components(effective_displacement, design_displacement, normal)
        grid_errors = _offset_error_components(grid_displacement, design_displacement, normal)
        maxima = np.maximum(maxima, np.asarray((*centre_errors, *grid_errors)))
        batch_index = None
        if float(offset) != 0.0:
            batch_index = len(nonzero_centers)
            nonzero_centers.append(shifted_center)
        sample_specs.append(
            {
                "offset_index": offset_index,
                "offset": float(offset),
                "mass": int(mass),
                "design_displacement": design_displacement,
                "effective_displacement": effective_displacement,
                "shifted_center": shifted_center,
                "centre_errors": centre_errors,
                "grid_errors": grid_errors,
                "batch_index": batch_index,
            }
        )
    if nonzero_centers:
        centers = torch.stack(nonzero_centers)
        count = centers.shape[0]
        batch_image, batch_labels = render_arbitrary_plane(
            context["scalar_tensor"],
            centers,
            frame[None].expand(count, -1, -1),
            basis[None].expand(count, -1, -1),
            tuple(geometry["output_shape_h_w"]),
            context["annotation_tensor"],
            sampling_contract=QUICKNII_RASTER_INDEX_SAMPLING,
        )
    else:
        batch_image = batch_labels = None
    for spec in sample_specs:
        offset_index = int(spec["offset_index"])
        offset = float(spec["offset"])
        mass = int(spec["mass"])
        batch_index = spec["batch_index"]
        if float(offset) == 0.0:
            scalar_raster = centre_scalar.copy()
            annotation_raster = centre_annotation.copy()
            reused_centre = True
        else:
            scalar_raster = np.ascontiguousarray(
                batch_image[int(batch_index), 0].cpu().numpy().astype(np.float32, copy=False)
            )
            annotation_raster = np.ascontiguousarray(
                batch_labels[int(batch_index), 0].to(torch.int64).cpu().numpy()
            )
            reused_centre = False
        scalar_samples.append(scalar_raster)
        annotation_samples.append(annotation_raster)
        payload = {
            "offset_index": offset_index,
            "offset_um": float(offset),
            "integer_mass": int(mass),
            "normalized_weight": float(mass / int(masses.sum())),
            "nonzero_render_batch_index": batch_index,
            "design_physical_displacement_ap_dv_ml_um": spec["design_displacement"].tolist(),
            "effective_physical_displacement_ap_dv_ml_um": spec["effective_displacement"].tolist(),
            "renderer_center_ap_dv_ml_float32": spec["shifted_center"].numpy().tolist(),
            "centre_displacement_max_abs_error_um": spec["centre_errors"][0],
            "centre_displacement_axial_error_um": spec["centre_errors"][1],
            "centre_displacement_tangential_error_um": spec["centre_errors"][2],
            "coordinate_raster_displacement_max_abs_error_um": spec["grid_errors"][0],
            "coordinate_raster_displacement_axial_error_um": spec["grid_errors"][1],
            "coordinate_raster_displacement_tangential_error_um": spec["grid_errors"][2],
            "reused_authenticated_centre_plane_render": reused_centre,
            "scalar_array_receipt": _array_receipt(scalar_raster),
            "annotation_array_receipt": _array_receipt(annotation_raster),
        }
        offset_receipts.append({**payload, "offset_render_receipt_sha256": _payload_sha256(payload)})
    if finite_psf["render_mode"] == CENTRE_PLANE_ABLATION:
        centre_support = np.asarray(parent["raster"]["brain_mask"]).copy()
        ones = np.ones(centre_scalar.shape, dtype=np.float32)
        reduced = {
            "observed_scalar_float32": centre_scalar.copy(),
            "centre_plane_annotation_int64": centre_annotation.copy(),
            "centre_plane_support_mask": centre_support,
            "slab_brain_occupancy_float32": centre_support.astype(np.float32),
            "slab_observable_support_mask": centre_support.copy(),
            "slab_modal_annotation_int64": centre_annotation.copy(),
            "slab_modal_purity_float32": ones.copy(),
            "centre_label_psf_mass_float32": ones.copy(),
            "dense_correspondence_weight_float32": centre_support.astype(np.float32),
            "dense_correspondence_abstention_mask": ~centre_support,
        }
    else:
        reduced = _reduce_samples(
            np.stack(scalar_samples),
            np.stack(annotation_samples),
            masses,
            centre_index,
        )
    reduced["centre_plane_ccf_um_float32"] = centre_ccf_um
    if (
        not np.array_equal(reduced["centre_plane_annotation_int64"], centre_annotation)
        or not np.array_equal(reduced["centre_plane_support_mask"], parent["raster"]["brain_mask"])
        or (
            finite_psf["render_mode"] == CENTRE_PLANE_ABLATION
            and not np.array_equal(reduced["observed_scalar_float32"], centre_scalar)
        )
    ):
        raise ValueError("Finite slab changed an authoritative centre-plane target")
    diagnostics = {
        "sampling_axis": "physical canonical arbitrary-plane normal, never atlas AP",
        "offset_render_order": "ascending axial_offsets_um",
        "zero_offset_index": centre_index,
        "zero_offset_render_reused": True,
        "nonzero_offset_render_count": int(offsets.size - 1),
        "nonzero_offsets_rendered_in_one_batch": True,
        "renderer_call_count": int(bool(nonzero_centers)),
        "maximum_centre_displacement_error_um": float(maxima[0]),
        "maximum_centre_axial_displacement_error_um": float(maxima[1]),
        "maximum_centre_tangential_displacement_error_um": float(maxima[2]),
        "maximum_coordinate_raster_displacement_error_um": float(maxima[3]),
        "maximum_coordinate_raster_axial_displacement_error_um": float(maxima[4]),
        "maximum_coordinate_raster_tangential_displacement_error_um": float(maxima[5]),
        "physical_displacement_tolerance_um": PHYSICAL_DISPLACEMENT_TOLERANCE_UM,
        "pose_draw_count": 0,
        "pose_or_tissue_conditioned_rejection_count": 0,
    }
    if float(max(maxima[1], maxima[2], maxima[4], maxima[5])) > PHYSICAL_DISPLACEMENT_TOLERANCE_UM:
        raise ValueError("Finite-slab physical-normal displacement exceeds tolerance")
    return reduced, offset_receipts, diagnostics


_SLAB_OBSERVATION_ARRAY_DTYPES = {
    "observed_scalar_float32": np.dtype(np.float32),
    "slab_brain_occupancy_float32": np.dtype(np.float32),
    "slab_observable_support_mask": np.dtype(bool),
    "slab_modal_annotation_int64": np.dtype(np.int64),
    "slab_modal_purity_float32": np.dtype(np.float32),
    "centre_label_psf_mass_float32": np.dtype(np.float32),
    "dense_correspondence_weight_float32": np.dtype(np.float32),
    "dense_correspondence_abstention_mask": np.dtype(bool),
}

_CENTRE_TARGET_ARRAY_DTYPES = {
    "centre_plane_annotation_int64": np.dtype(np.int64),
    "centre_plane_support_mask": np.dtype(bool),
    "centre_plane_ccf_um_float32": np.dtype(np.float32),
}


def _observation_arrays(observation: dict[str, object]) -> dict[str, np.ndarray]:
    return {name: np.asarray(observation[name]) for name in _SLAB_OBSERVATION_ARRAY_DTYPES}


def _centre_target_arrays(targets: dict[str, object]) -> dict[str, np.ndarray]:
    return {name: np.asarray(targets[name]) for name in _CENTRE_TARGET_ARRAY_DTYPES}


def _centre_target_metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    shape = arrays["centre_plane_annotation_int64"].shape
    if (
        len(shape) != 2
        or arrays["centre_plane_support_mask"].shape != shape
        or arrays["centre_plane_ccf_um_float32"].shape != shape + (3,)
        or any(arrays[name].dtype != dtype for name, dtype in _CENTRE_TARGET_ARRAY_DTYPES.items())
        or not np.isfinite(arrays["centre_plane_ccf_um_float32"]).all()
        or not np.array_equal(
            arrays["centre_plane_support_mask"], arrays["centre_plane_annotation_int64"] != 0
        )
    ):
        raise ValueError("Centre-plane target arrays violate v4 semantics")
    receipts = {name: _array_receipt(array) for name, array in arrays.items()}
    return {
        "array_receipts": receipts,
        "combined_sha256": _payload_sha256(
            {"schema": "anatomy-tracker.centre-plane-target-arrays/v4", "array_receipts": receipts}
        ),
    }


def _centre_target_receipt_payload(targets: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in targets.items()
        if key not in {*_CENTRE_TARGET_ARRAY_DTYPES, "receipt_sha256"}
    }


def _observation_metadata(
    arrays: dict[str, np.ndarray],
    centre_targets: dict[str, np.ndarray],
) -> dict[str, object]:
    shape = arrays["observed_scalar_float32"].shape
    centre_support = centre_targets["centre_plane_support_mask"]
    if (
        len(shape) != 2
        or any(array.shape != shape for array in arrays.values())
        or centre_support.shape != shape
        or any(arrays[name].dtype != dtype for name, dtype in _SLAB_OBSERVATION_ARRAY_DTYPES.items())
        or any(
            not np.isfinite(array).all()
            for name, array in arrays.items()
            if np.issubdtype(array.dtype, np.floating)
        )
        or not np.array_equal(
            arrays["slab_observable_support_mask"], arrays["slab_brain_occupancy_float32"] > 0
        )
        or any(
            np.any(arrays[name] < 0) or np.any(arrays[name] > 1)
            for name in (
                "slab_brain_occupancy_float32",
                "slab_modal_purity_float32",
                "centre_label_psf_mass_float32",
                "dense_correspondence_weight_float32",
            )
        )
        or not np.array_equal(
            arrays["dense_correspondence_abstention_mask"],
            (~centre_support)
            | (arrays["centre_label_psf_mass_float32"] <= 0.5),
        )
        or not np.array_equal(
            arrays["dense_correspondence_weight_float32"],
            np.where(
                centre_support,
                np.clip(
                    (arrays["centre_label_psf_mass_float32"].astype(np.float64) - 0.5) / 0.3,
                    0.0,
                    1.0,
                ),
                0.0,
            ).astype(np.float32),
        )
    ):
        raise ValueError("Finite-slab observation arrays violate v4 semantics")
    receipts = {name: _array_receipt(array) for name, array in arrays.items()}
    return {
        "array_receipts": receipts,
        "combined_sha256": _payload_sha256(
            {"schema": "anatomy-tracker.slab-observation-arrays/v4", "array_receipts": receipts}
        ),
        "centre_plane_brain_pixel_count": int(centre_support.sum()),
        "slab_observable_pixel_count": int(arrays["slab_observable_support_mask"].sum()),
        "slab_effective_brain_pixel_mass": float(
            arrays["slab_brain_occupancy_float32"].astype(np.float64).sum()
        ),
        "dense_abstention_pixel_count": int(arrays["dense_correspondence_abstention_mask"].sum()),
        "dense_eligible_pixel_count": int((~arrays["dense_correspondence_abstention_mask"]).sum()),
        "dense_effective_supervision_mass": float(
            arrays["dense_correspondence_weight_float32"].astype(np.float64).sum()
        ),
    }


def _parent_reference(parent: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "anatomy-tracker.verified-finite-parent-reference/v4",
        "schema_version": parent["schema_version"],
        "generator_algorithm": parent["generator_algorithm"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "finite_plane_geometry_sha256": parent["finite_plane_geometry_sha256"],
        "rendered_artifacts_sha256": parent["rendered_artifacts_sha256"],
        "support_index_sha256": parent["support_index_sha256"],
        "prepared_context_sha256": context["prepared_context_sha256"],
        "provenance_sha256": parent["provenance_sha256"],
        "centre_scalar_array_receipt": parent["raster_array_receipts"]["scalar"],
        "centre_annotation_array_receipt": parent["raster_array_receipts"]["annotation"],
        "centre_brain_mask_array_receipt": parent["raster_array_receipts"]["brain_mask"],
    }


def _block_receipt_payload(artifact: dict[str, object]) -> dict[str, object]:
    block = artifact["slab_observation_v4"]
    return {
        key: value
        for key, value in block.items()
        if key not in {*_SLAB_OBSERVATION_ARRAY_DTYPES, "receipt_sha256"}
    }


def finite_slab_render_receipt_v4(adapter_result: dict[str, object]) -> dict[str, object]:
    artifact = adapter_result["artifact"]
    block = artifact["slab_observation_v4"]
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "parent_reference": artifact["parent_reference"],
        "generator": artifact["generator"],
        "provenance": artifact["provenance"],
        "provenance_sha256": artifact["provenance_sha256"],
        "offset_render_receipts": artifact["offset_render_receipts"],
        "diagnostics": artifact["diagnostics"],
        "centre_plane_targets_receipt": _centre_target_receipt_payload(
            artifact["centre_plane_targets"]
        ),
        "centre_plane_targets_receipt_sha256": artifact["centre_plane_targets"][
            "receipt_sha256"
        ],
        "slab_observation_v4_receipt": _block_receipt_payload(artifact),
        "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
    }


def make_finite_slab_render_v4(
    parent: dict[str, object],
    prepared_context: dict[str, object],
    *,
    render_mode: str = FINITE_BOXCAR,
    thickness_seed: int | None = None,
    nominal_cut_thickness_um: float | None = None,
    generator_source_commit: str | None = None,
    parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Create a standalone authenticated slab observation without redrawing pose."""
    parent_generator._validate_prepared_context(prepared_context)
    support = prepared_context["support_index"]
    verify_finite_arbitrary_plane_render(
        parent,
        support,
        generator_source_commit=parent_generator_source_commit,
        _support_preverified=True,
    )
    parent_config = parent["generator"]["resolved_config"]
    if parent_config["prepared_context_sha256"] != prepared_context["prepared_context_sha256"]:
        raise ValueError("Prepared context does not match the verified finite parent")
    selection = _resolve_thickness_selection(render_mode, thickness_seed, nominal_cut_thickness_um)
    finite_psf = finite_psf_v4(
        render_mode,
        selection["nominal_cut_thickness_um"],
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    parent_reference = _parent_reference(parent, prepared_context)
    implementation = {
        "source_path": "training/arbitrary_plane_finite_slab_v4.py",
        "loaded_source_sha256": _LOADED_SOURCE_SHA256,
        "loaded_dependency_source_sha256": _LOADED_DEPENDENCY_SOURCE_SHA256,
        "source_commit": _source_commit(generator_source_commit),
        "parent_generator_source_commit": _source_commit(parent_generator_source_commit),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    implementation["implementation_sha256"] = _payload_sha256(implementation)
    model_independence = {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "initialization": "independent thickness seed or explicit physical thickness only; no learned initialization",
    }
    resolved_config = {
        "schema_version": FINITE_SLAB_V4_SCHEMA,
        "algorithm": FINITE_SLAB_V4_ALGORITHM,
        "parent_reference": parent_reference,
        "thickness_selection": selection,
        "finite_psf_capability": finite_psf_capability_v4(),
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
    }
    generator = {
        "implementation": implementation,
        "resolved_config": resolved_config,
        "resolved_config_sha256": _payload_sha256(resolved_config),
        **model_independence,
        "model_independence_sha256": _payload_sha256(model_independence),
    }
    rendered_arrays, offset_receipts, diagnostics = _render_samples(
        parent, prepared_context, finite_psf
    )
    centre_arrays = {
        name: rendered_arrays[name] for name in _CENTRE_TARGET_ARRAY_DTYPES
    }
    arrays = {
        name: rendered_arrays[name] for name in _SLAB_OBSERVATION_ARRAY_DTYPES
    }
    centre_metadata = _centre_target_metadata(centre_arrays)
    metadata = _observation_metadata(arrays, centre_arrays)
    provenance = json.loads(_canonical_json(parent["provenance"]))
    artifact: dict[str, object] = {
        "schema_version": FINITE_SLAB_V4_SCHEMA,
        "algorithm": FINITE_SLAB_V4_ALGORITHM,
        "parent_reference": parent_reference,
        "generator": generator,
        "provenance": provenance,
        "provenance_sha256": _payload_sha256(provenance),
        "offset_render_receipts": offset_receipts,
        "diagnostics": diagnostics,
    }
    centre_targets: dict[str, object] = {
        "schema": "anatomy-tracker.authoritative-centre-plane-targets/v4",
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "support_index_sha256": parent["support_index_sha256"],
        **centre_arrays,
        **centre_metadata,
    }
    centre_targets["receipt_sha256"] = _payload_sha256(
        _centre_target_receipt_payload(centre_targets)
    )
    artifact["centre_plane_targets"] = centre_targets
    block: dict[str, object] = {
        "schema": SLAB_OBSERVATION_V4_SCHEMA,
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "support_index_sha256": parent["support_index_sha256"],
        "provenance_sha256": parent["provenance_sha256"],
        "split": parent["split"],
        "sample_index": parent["sample_index"],
        "animal_id": provenance["animal_id"],
        "specimen_id": provenance["specimen_id"],
        "experiment_id": provenance["experiment_id"],
        "centre_plane_targets_receipt_sha256": centre_targets["receipt_sha256"],
        "thickness_selection": selection,
        "finite_psf": finite_psf,
        **arrays,
        **metadata,
    }
    identity = {
        "schema": "anatomy-tracker.slab-observation-identity/v4",
        "parent_reference": parent_reference,
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
        "combined_sha256": metadata["combined_sha256"],
        "centre_plane_targets_receipt_sha256": centre_targets["receipt_sha256"],
        "generator_implementation_sha256": implementation["implementation_sha256"],
        "model_independence_sha256": generator["model_independence_sha256"],
        "provenance_sha256": artifact["provenance_sha256"],
        "offset_render_receipt_sha256": _payload_sha256(offset_receipts),
        "diagnostics_sha256": _payload_sha256(diagnostics),
    }
    block["slab_observation_id"] = _payload_sha256(identity)
    artifact["slab_observation_v4"] = block
    block["receipt_sha256"] = _payload_sha256(_block_receipt_payload(artifact))
    result = {"artifact": artifact}
    artifact["receipt_sha256"] = _payload_sha256(finite_slab_render_receipt_v4(result))
    return result


def _replay_from_receipt(
    adapter_result: dict[str, object],
    parent: dict[str, object],
    prepared_context: dict[str, object],
    *,
    generator_source_commit: str | None,
    parent_generator_source_commit: str | None,
) -> dict[str, object]:
    selection = adapter_result["artifact"]["slab_observation_v4"]["thickness_selection"]
    render_mode = adapter_result["artifact"]["slab_observation_v4"]["finite_psf"]["render_mode"]
    if selection["selection_mode"] == "independent-seeded-uniform":
        return make_finite_slab_render_v4(
            parent,
            prepared_context,
            render_mode=render_mode,
            thickness_seed=_parse_seed_hex(selection["thickness_seed_uint64"]),
            generator_source_commit=generator_source_commit,
            parent_generator_source_commit=parent_generator_source_commit,
        )
    return make_finite_slab_render_v4(
        parent,
        prepared_context,
        render_mode=render_mode,
        nominal_cut_thickness_um=selection["nominal_cut_thickness_um"],
        generator_source_commit=generator_source_commit,
        parent_generator_source_commit=parent_generator_source_commit,
    )


def verify_finite_slab_render_v4(
    adapter_result: dict[str, object],
    parent: dict[str, object],
    prepared_context: dict[str, object],
    *,
    generator_source_commit: str | None = None,
    parent_generator_source_commit: str | None = None,
) -> None:
    if set(adapter_result) != {"artifact"}:
        raise ValueError("Finite-slab adapter result must contain exactly one artifact")
    artifact = adapter_result.get("artifact", {})
    required_artifact_keys = {
        "schema_version",
        "algorithm",
        "parent_reference",
        "generator",
        "provenance",
        "provenance_sha256",
        "offset_render_receipts",
        "diagnostics",
        "centre_plane_targets",
        "slab_observation_v4",
        "receipt_sha256",
    }
    if set(artifact) != required_artifact_keys:
        raise ValueError("Finite-slab artifact contains missing or unauthenticated extra fields")
    if (
        artifact["schema_version"] != FINITE_SLAB_V4_SCHEMA
        or artifact["algorithm"] != FINITE_SLAB_V4_ALGORITHM
    ):
        raise ValueError("Unsupported finite-slab schema or algorithm")
    parent_generator._validate_prepared_context(prepared_context)
    support = prepared_context["support_index"]
    verify_finite_arbitrary_plane_render(
        parent,
        support,
        generator_source_commit=parent_generator_source_commit,
        _support_preverified=True,
    )
    if parent["generator"]["resolved_config"]["prepared_context_sha256"] != prepared_context["prepared_context_sha256"]:
        raise ValueError("Prepared context does not match verified finite parent")
    expected_parent_reference = _parent_reference(parent, prepared_context)
    if artifact["parent_reference"] != expected_parent_reference:
        raise ValueError("Finite-slab parent reference does not match verified parent/context")
    if (
        artifact["provenance"] != parent["provenance"]
        or artifact["provenance_sha256"] != parent["provenance_sha256"]
        or artifact["provenance_sha256"] != _payload_sha256(artifact["provenance"])
    ):
        raise ValueError("Finite-slab provenance does not match verified parent")
    generator = artifact["generator"]
    implementation = generator["implementation"]
    implementation_payload = {key: value for key, value in implementation.items() if key != "implementation_sha256"}
    if (
        implementation["implementation_sha256"] != _payload_sha256(implementation_payload)
        or implementation["source_path"] != "training/arbitrary_plane_finite_slab_v4.py"
        or implementation["loaded_source_sha256"] != _LOADED_SOURCE_SHA256
        or implementation["loaded_dependency_source_sha256"] != _LOADED_DEPENDENCY_SOURCE_SHA256
        or implementation["source_commit"] != _source_commit(generator_source_commit)
        or implementation["parent_generator_source_commit"] != _source_commit(parent_generator_source_commit)
        or implementation["numpy_version"] != np.__version__
        or implementation["torch_version"] != torch.__version__
    ):
        raise ValueError("Finite-slab implementation/runtime receipt does not match")
    dependency_keys = {key for key in generator if key.endswith("_dependencies")}
    required_dependencies = {
        "learned_checkpoint_dependencies",
        "previous_model_dependencies",
        "pretrained_feature_dependencies",
    }
    if dependency_keys != required_dependencies or any(generator[key] != [] for key in required_dependencies):
        raise ValueError("Finite-slab adapter must have exactly three empty learned-dependency lists")
    model_independence = {
        key: generator[key]
        for key in (
            "learned_checkpoint_dependencies",
            "previous_model_dependencies",
            "pretrained_feature_dependencies",
        )
    }
    model_independence["initialization"] = generator["initialization"]
    if (
        generator["initialization"]
        != "independent thickness seed or explicit physical thickness only; no learned initialization"
        or generator["model_independence_sha256"] != _payload_sha256(model_independence)
    ):
        raise ValueError("Finite-slab model-independence receipt does not match")
    config = generator["resolved_config"]
    if generator["resolved_config_sha256"] != _payload_sha256(config):
        raise ValueError("Finite-slab resolved-config receipt does not match")
    block = artifact["slab_observation_v4"]
    required_block_keys = {
        "schema",
        "slab_observation_id",
        "finite_plane_render_id",
        "finite_render_receipt_sha256",
        "plane_realization_id",
        "support_index_sha256",
        "provenance_sha256",
        "split",
        "sample_index",
        "animal_id",
        "specimen_id",
        "experiment_id",
        "thickness_selection",
        "finite_psf",
        "centre_plane_targets_receipt_sha256",
        *_SLAB_OBSERVATION_ARRAY_DTYPES,
        "array_receipts",
        "combined_sha256",
        "centre_plane_brain_pixel_count",
        "slab_observable_pixel_count",
        "slab_effective_brain_pixel_mass",
        "dense_abstention_pixel_count",
        "dense_eligible_pixel_count",
        "dense_effective_supervision_mass",
        "receipt_sha256",
    }
    if set(block) != required_block_keys or block["schema"] != SLAB_OBSERVATION_V4_SCHEMA:
        raise ValueError("Finite-slab observation block contains missing or unauthenticated fields")
    expected_core_ids = {
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "support_index_sha256": parent["support_index_sha256"],
        "provenance_sha256": parent["provenance_sha256"],
        "split": parent["split"],
        "sample_index": parent["sample_index"],
        "animal_id": parent["provenance"]["animal_id"],
        "specimen_id": parent["provenance"]["specimen_id"],
        "experiment_id": parent["provenance"]["experiment_id"],
    }
    if any(block[key] != value for key, value in expected_core_ids.items()):
        raise ValueError("Finite-slab observation core IDs do not match verified parent")
    centre_targets = artifact["centre_plane_targets"]
    required_centre_target_keys = {
        "schema",
        "finite_plane_render_id",
        "finite_render_receipt_sha256",
        "plane_realization_id",
        "support_index_sha256",
        *_CENTRE_TARGET_ARRAY_DTYPES,
        "array_receipts",
        "combined_sha256",
        "receipt_sha256",
    }
    if (
        set(centre_targets) != required_centre_target_keys
        or centre_targets["schema"] != "anatomy-tracker.authoritative-centre-plane-targets/v4"
    ):
        raise ValueError("Centre-plane target block contains missing or unauthenticated fields")
    expected_centre_target_ids = {
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "plane_realization_id": parent["plane_realization_id"],
        "support_index_sha256": parent["support_index_sha256"],
    }
    if any(centre_targets[key] != value for key, value in expected_centre_target_ids.items()):
        raise ValueError("Centre-plane target IDs do not match verified parent")
    centre_arrays = _centre_target_arrays(centre_targets)
    expected_centre_metadata = _centre_target_metadata(centre_arrays)
    if any(centre_targets[key] != value for key, value in expected_centre_metadata.items()):
        raise ValueError("Centre-plane target live array receipts do not match")
    if (
        not np.array_equal(centre_arrays["centre_plane_annotation_int64"], parent["raster"]["annotation"])
        or not np.array_equal(centre_arrays["centre_plane_support_mask"], parent["raster"]["brain_mask"])
        or centre_targets["receipt_sha256"]
        != _payload_sha256(_centre_target_receipt_payload(centre_targets))
        or block["centre_plane_targets_receipt_sha256"] != centre_targets["receipt_sha256"]
    ):
        raise ValueError("Centre-plane targets do not match verified parent or receipt")
    selection = block["thickness_selection"]
    mode = block["finite_psf"]["render_mode"]
    if selection["selection_mode"] == "independent-seeded-uniform":
        expected_selection = _resolve_thickness_selection(
            mode, _parse_seed_hex(selection["thickness_seed_uint64"]), None
        )
    else:
        expected_selection = _resolve_thickness_selection(
            mode, None, selection["nominal_cut_thickness_um"]
        )
    if selection != expected_selection:
        raise ValueError("Finite-slab thickness selection does not replay")
    expected_psf = finite_psf_v4(
        mode,
        selection["nominal_cut_thickness_um"],
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    if block["finite_psf"] != expected_psf:
        raise ValueError("Finite-slab PSF contract does not match thickness selection")
    expected_capability = finite_psf_capability_v4()
    expected_config = {
        "schema_version": FINITE_SLAB_V4_SCHEMA,
        "algorithm": FINITE_SLAB_V4_ALGORITHM,
        "parent_reference": expected_parent_reference,
        "thickness_selection": expected_selection,
        "finite_psf_capability": expected_capability,
        "finite_psf_sha256": expected_psf["finite_psf_sha256"],
    }
    if config != expected_config:
        raise ValueError("Finite-slab resolved config does not match installed contracts")
    arrays = _observation_arrays(block)
    expected_metadata = _observation_metadata(arrays, centre_arrays)
    if any(block[key] != value for key, value in expected_metadata.items()):
        raise ValueError("Finite-slab live array receipts do not match")
    identity = {
        "schema": "anatomy-tracker.slab-observation-identity/v4",
        "parent_reference": expected_parent_reference,
        "finite_psf_sha256": expected_psf["finite_psf_sha256"],
        "combined_sha256": expected_metadata["combined_sha256"],
        "centre_plane_targets_receipt_sha256": centre_targets["receipt_sha256"],
        "generator_implementation_sha256": implementation["implementation_sha256"],
        "model_independence_sha256": generator["model_independence_sha256"],
        "provenance_sha256": artifact["provenance_sha256"],
        "offset_render_receipt_sha256": _payload_sha256(artifact["offset_render_receipts"]),
        "diagnostics_sha256": _payload_sha256(artifact["diagnostics"]),
    }
    if block["slab_observation_id"] != _payload_sha256(identity):
        raise ValueError("Finite-slab observation ID does not match")
    if block["receipt_sha256"] != _payload_sha256(_block_receipt_payload(artifact)):
        raise ValueError("Finite-slab observation block receipt does not match")
    if artifact["receipt_sha256"] != _payload_sha256(finite_slab_render_receipt_v4(adapter_result)):
        raise ValueError("Finite-slab artifact receipt does not match")
    replayed = _replay_from_receipt(
        adapter_result,
        parent,
        prepared_context,
        generator_source_commit=generator_source_commit,
        parent_generator_source_commit=parent_generator_source_commit,
    )
    if finite_slab_render_receipt_v4(replayed) != finite_slab_render_receipt_v4(adapter_result):
        raise ValueError("Finite-slab replay receipt does not match")
    replayed_arrays = _observation_arrays(replayed["artifact"]["slab_observation_v4"])
    if any(not np.array_equal(array, replayed_arrays[name]) for name, array in arrays.items()):
        raise ValueError("Finite-slab replay arrays do not match")
    replayed_centre = _centre_target_arrays(replayed["artifact"]["centre_plane_targets"])
    if any(not np.array_equal(array, replayed_centre[name]) for name, array in centre_arrays.items()):
        raise ValueError("Finite-slab replay centre targets do not match")


def replay_finite_slab_render_v4(
    adapter_result: dict[str, object],
    parent: dict[str, object],
    prepared_context: dict[str, object],
    *,
    generator_source_commit: str | None = None,
    parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    verify_finite_slab_render_v4(
        adapter_result,
        parent,
        prepared_context,
        generator_source_commit=generator_source_commit,
        parent_generator_source_commit=parent_generator_source_commit,
    )
    return _replay_from_receipt(
        adapter_result,
        parent,
        prepared_context,
        generator_source_commit=generator_source_commit,
        parent_generator_source_commit=parent_generator_source_commit,
    )
