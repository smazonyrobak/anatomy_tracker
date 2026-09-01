"""Finite-thickness v2 arbitrary-plane acquisition stage."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_geometry import (
    QUICKNII_RASTER_INDEX_SAMPLING,
    allen_index_to_physical_um_vectors,
    physical_um_to_allen_index_vectors,
    render_arbitrary_plane,
)
from training.arbitrary_plane_rendered_generator import effective_renderer_sampling_arrays


V2_SLAB_SCHEMA = "anatomy-tracker.arbitrary-plane-slab-render/v2"
V2_SLAB_ALGORITHM = "physical-normal-symmetric-trapezoid-boxcar/v2"
V2_GENERIC_SLAB_SCHEMA = "anatomy-tracker.authenticated-generic-arbitrary-plane-slab-render/v2"
V2_GENERIC_SLAB_ALGORITHM = "physical-normal-generic-symmetric-trapezoid-boxcar/v2"
_SOURCE_ROOT = Path(__file__).parent
_AXIAL_STEP_UM_MAX = 12.5
_PHYSICAL_DISPLACEMENT_TOLERANCE_UM = 0.01


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in (
            "arbitrary_plane_synthetic_generator_v2.py",
            "arbitrary_plane_acquisition_v2.py",
            "arbitrary_plane_geometry.py",
            "arbitrary_plane_rendered_generator.py",
        )
    }


def finite_boxcar_kernel(
    render_mode: str,
    nominal_cut_thickness_um: float,
    *,
    axial_step_um_max: float = _AXIAL_STEP_UM_MAX,
) -> dict[str, object]:
    """Return the frozen ablation or symmetric finite-boxcar schedule."""
    thickness = float(nominal_cut_thickness_um)
    step_max = float(axial_step_um_max)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("nominal cut thickness must be finite and positive")
    if not math.isfinite(step_max) or step_max <= 0.0 or step_max > _AXIAL_STEP_UM_MAX:
        raise ValueError("axial step maximum must be in (0, 12.5] um")
    if render_mode == "centre_plane_ablation":
        offsets = np.asarray([0.0], dtype=np.float64)
        masses = np.asarray([1], dtype=np.int64)
        operator = "direct-centre-plane-ablation"
        effective_support = 0.0
    elif render_mode == "finite_boxcar":
        half_intervals = int(math.ceil(thickness / (2.0 * step_max)))
        offsets = np.linspace(
            -thickness / 2.0, thickness / 2.0, 2 * half_intervals + 1, dtype=np.float64
        )
        offsets[half_intervals] = 0.0
        offsets[:half_intervals] = -offsets[:half_intervals:-1]
        masses = np.full(offsets.size, 2, dtype=np.int64)
        masses[[0, -1]] = 1
        operator = "finite-full-slab-boxcar-development-abstraction"
        effective_support = thickness
    else:
        raise ValueError("unknown v2 slab render mode")
    weights = masses.astype(np.float64) / int(masses.sum())
    mean_offset = math.fsum(
        float(weight) * float(offset) for weight, offset in zip(weights, offsets)
    )
    second_moment = math.fsum(
        float(weight) * (float(offset) - mean_offset) ** 2
        for weight, offset in zip(weights, offsets)
    )
    if (
        not np.array_equal(offsets, -offsets[::-1])
        or not np.array_equal(masses, masses[::-1])
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or math.fsum(weights.tolist()) != 1.0
        or mean_offset != 0.0
    ):
        raise ValueError("v2 slab kernel failed symmetry or normalization")
    return {
        "render_mode": render_mode,
        "nominal_cut_thickness_um": thickness,
        "material_thickness_um": thickness,
        "section_spacing_um": None,
        "optical_plane_depths_um": [],
        "effective_optical_support_um": effective_support,
        "projection_operator": operator,
        "modality": "unknown-development-abstraction",
        "optical_kernel_offsets_um": offsets.tolist(),
        "optical_kernel_integer_masses": masses.tolist(),
        "optical_kernel_weights": weights.tolist(),
        "axial_step_um_max": step_max,
        "axial_sample_count": int(offsets.size),
        "axial_support_second_moment_um2": second_moment,
    }


def reduce_v2_slab_samples(
    scalar_samples: np.ndarray,
    annotation_samples: np.ndarray,
    integer_masses: np.ndarray,
    centre_index: int,
) -> dict[str, object]:
    """Reduce ordered axial samples with exact integer categorical masses."""
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
        raise ValueError("v2 slab samples have the wrong shape, dtype, or masses")
    total_mass = int(masses.sum())
    scalar_accumulator = np.zeros(scalar.shape[1:], dtype=np.float64)
    occupancy_mass = np.zeros(scalar.shape[1:], dtype=np.int64)
    for index, mass in enumerate(masses):
        scalar_accumulator += int(mass) * scalar[index].astype(np.float64)
        occupancy_mass += int(mass) * (annotation[index] != 0)
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
    centre_support_mass = np.zeros(scalar.shape[1:], dtype=np.int64)
    for sample_index, mass in enumerate(masses):
        centre_support_mass += int(mass) * (annotation[sample_index] == centre_annotation)
    centre_support_weight = (centre_support_mass / total_mass).astype(np.float32)
    supervision_weight = np.where(
        centre_support,
        np.clip((centre_support_weight.astype(np.float64) - 0.5) / 0.3, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    return {
        "scalar": (scalar_accumulator / total_mass).astype(np.float32),
        "centre_plane_annotation": centre_annotation,
        "centre_plane_support_mask": centre_support,
        "slab_brain_occupancy": (occupancy_mass / total_mass).astype(np.float32),
        "slab_observable_support_mask": occupancy_mass > 0,
        "slab_modal_annotation": modal_annotation,
        "slab_label_purity": (winning_mass / total_mass).astype(np.float32),
        "centre_label_support_weight": centre_support_weight,
        "slab_supervision_weight_or_abstention": {
            "dense_correspondence_weight": supervision_weight,
            "abstention_mask": (~centre_support) | (centre_support_weight <= 0.5),
        },
    }


def _slab_arrays(raster: dict[str, object]) -> dict[str, np.ndarray]:
    supervision = raster["slab_supervision_weight_or_abstention"]
    return {
        "scalar": np.asarray(raster["scalar"]),
        "centre_plane_annotation": np.asarray(raster["centre_plane_annotation"]),
        "centre_plane_support_mask": np.asarray(raster["centre_plane_support_mask"]),
        "slab_brain_occupancy": np.asarray(raster["slab_brain_occupancy"]),
        "slab_observable_support_mask": np.asarray(raster["slab_observable_support_mask"]),
        "slab_modal_annotation": np.asarray(raster["slab_modal_annotation"]),
        "slab_label_purity": np.asarray(raster["slab_label_purity"]),
        "centre_label_support_weight": np.asarray(raster["centre_label_support_weight"]),
        "dense_correspondence_weight": np.asarray(supervision["dense_correspondence_weight"]),
        "dense_correspondence_abstention_mask": np.asarray(supervision["abstention_mask"]),
    }


def _slab_raster_metadata(raster: dict[str, object]) -> dict[str, object]:
    arrays = _slab_arrays(raster)
    shape = arrays["scalar"].shape
    dtypes = {
        "scalar": np.dtype(np.float32),
        "centre_plane_annotation": np.dtype(np.int64),
        "centre_plane_support_mask": np.dtype(bool),
        "slab_brain_occupancy": np.dtype(np.float32),
        "slab_observable_support_mask": np.dtype(bool),
        "slab_modal_annotation": np.dtype(np.int64),
        "slab_label_purity": np.dtype(np.float32),
        "centre_label_support_weight": np.dtype(np.float32),
        "dense_correspondence_weight": np.dtype(np.float32),
        "dense_correspondence_abstention_mask": np.dtype(bool),
    }
    if (
        len(shape) != 2
        or any(array.shape != shape for array in arrays.values())
        or any(arrays[name].dtype != dtype for name, dtype in dtypes.items())
        or any(not array.all() for array in [
            np.isfinite(arrays[name]) for name, dtype in dtypes.items() if dtype.kind == "f"
        ])
        or not np.array_equal(
            arrays["centre_plane_support_mask"], arrays["centre_plane_annotation"] != 0
        )
        or not np.array_equal(
            arrays["slab_observable_support_mask"], arrays["slab_brain_occupancy"] > 0
        )
        or any(
            np.any(arrays[name] < 0) or np.any(arrays[name] > 1)
            for name in (
                "slab_brain_occupancy",
                "slab_label_purity",
                "centre_label_support_weight",
            )
        )
        or not np.array_equal(
            arrays["dense_correspondence_abstention_mask"],
            (~arrays["centre_plane_support_mask"])
            | (arrays["centre_label_support_weight"] <= 0.5),
        )
        or not np.array_equal(
            arrays["dense_correspondence_weight"],
            np.where(
                arrays["centre_plane_support_mask"],
                np.clip(
                    (arrays["centre_label_support_weight"].astype(np.float64) - 0.5)
                    / 0.3,
                    0.0,
                    1.0,
                ),
                0.0,
            ).astype(np.float32),
        )
    ):
        raise ValueError("v2 slab raster arrays violate their frozen semantics")
    receipts = {name: acquisition._array_receipt(array) for name, array in arrays.items()}
    return {
        "array_receipts": receipts,
        "combined_sha256": acquisition._payload_sha256(
            {"schema": "anatomy-tracker.v2-slab-arrays/v1", "array_receipts": receipts}
        ),
        "centre_plane_brain_pixel_count": int(arrays["centre_plane_support_mask"].sum()),
        "slab_observable_pixel_count": int(arrays["slab_observable_support_mask"].sum()),
        "dense_abstention_pixel_count": int(
            arrays["dense_correspondence_abstention_mask"].sum()
        ),
    }


def _slab_recipe(
    precursor: dict[str, object],
    kernel: dict[str, object],
    *,
    schema_version: str = V2_SLAB_SCHEMA,
    algorithm: str = V2_SLAB_ALGORITHM,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "algorithm": algorithm,
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "global_reference_grid_id": precursor["geometry"]["global_reference_grid_id"],
        **kernel,
        "sampling_direction": "canonical physical AP-DV-ML unit normal",
        "scalar_reduction": "ascending-offset float64 ordered sum of integer_mass*float32 samples; divide once; cast float32",
        "support_reduction": "ascending-offset exact int64 occupied mass; divide once; cast float32",
        "categorical_reduction": "exact int64 weighted mode including zero; ties choose smallest annotation ID",
        "centre_plane_label_target": "unchanged direct offset-zero nearest-neighbour int64 annotation",
        "slab_supervision_rule": "inside centre-plane brain: weight=clip((centre_label_support_weight-0.5)/0.3,0,1), abstain at <=0.5; outside: weight=0 and abstain",
        "outside_atlas_rule": "scalar, support, and annotation samples are zero",
        "physical_displacement_tolerance_um": _PHYSICAL_DISPLACEMENT_TOLERANCE_UM,
        "output_dtypes": {
            "scalar_and_weight_maps": "float32",
            "annotations": "int64",
            "masks": "bool",
        },
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
    }


def _offset_error_components(
    observed: np.ndarray, expected: np.ndarray, normal: np.ndarray
) -> tuple[float, float, float]:
    error = observed - expected
    axial = error @ normal
    tangential = error - axial[..., None] * normal
    return (
        float(np.max(np.abs(error))),
        float(np.max(np.abs(axial))),
        float(np.max(np.linalg.norm(tangential, axis=-1))),
    )


def _render_offset_samples(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    kernel: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    parent = prepared_context["opaque_v1_context"]
    support = acquisition._context_support(prepared_context)
    geometry = precursor["geometry"]
    spacing = np.asarray(support["voxel_size_um"], dtype=np.float64)
    atlas_origin = tuple(float(value) for value in support["origin_um"])
    atlas_shape = tuple(int(value) for value in support["annotation_shape"])
    normal = np.asarray(geometry["normal_rp2_ap_dv_ml"], dtype=np.float64)
    offsets = np.asarray(kernel["optical_kernel_offsets_um"], dtype=np.float64)
    masses = np.asarray(kernel["optical_kernel_integer_masses"], dtype=np.int64)
    base_center = torch.as_tensor(
        geometry["renderer_center_ap_dv_ml"], dtype=torch.float32
    )
    frame = torch.as_tensor(geometry["renderer_frame_ap_dv_ml"], dtype=torch.float32)
    basis = torch.as_tensor(geometry["renderer_inplane_basis"], dtype=torch.float32)
    base_geometry = dict(geometry)
    base_geometry["renderer_center_ap_dv_ml"] = base_center.tolist()
    base_grid = effective_renderer_sampling_arrays(
        base_geometry,
        atlas_shape,
        origin_ap_dv_ml_um=atlas_origin,
        voxel_size_ap_dv_ml_um=tuple(spacing),
    )
    base_points = base_grid["coordinate_raster_allen_index_float32"].astype(np.float64)
    scalar_samples = []
    annotation_samples = []
    offset_receipts = []
    centre_index = int(np.flatnonzero(offsets == 0.0)[0])
    centre_scalar = np.asarray(precursor["raster"]["scalar"])
    centre_annotation = np.asarray(precursor["raster"]["annotation"])
    maxima = np.zeros(6, dtype=np.float64)
    for index, (offset, mass) in enumerate(zip(offsets, masses)):
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
        grid = effective_renderer_sampling_arrays(
            shifted_geometry,
            atlas_shape,
            origin_ap_dv_ml_um=atlas_origin,
            voxel_size_ap_dv_ml_um=tuple(spacing),
        )
        grid_displacement = (
            grid["coordinate_raster_allen_index_float32"].astype(np.float64) - base_points
        ) * spacing
        centre_errors = _offset_error_components(
            effective_displacement, design_displacement, normal
        )
        grid_errors = _offset_error_components(grid_displacement, design_displacement, normal)
        maxima = np.maximum(maxima, np.asarray((*centre_errors, *grid_errors)))
        if float(offset) == 0.0:
            scalar_raster = centre_scalar
            annotation_raster = centre_annotation
            reused_centre = True
        else:
            image, labels = render_arbitrary_plane(
                parent["scalar_tensor"],
                shifted_center,
                frame,
                basis,
                tuple(geometry["output_shape_h_w"]),
                parent["annotation_tensor"],
                sampling_contract=QUICKNII_RASTER_INDEX_SAMPLING,
            )
            scalar_raster = image[0, 0].cpu().numpy()
            annotation_raster = labels[0, 0].to(torch.int64).cpu().numpy()
            reused_centre = False
        brain_mask = annotation_raster != 0
        scalar_samples.append(scalar_raster)
        annotation_samples.append(annotation_raster)
        grid_receipts = {
            name: acquisition._array_receipt(array) for name, array in grid.items()
        }
        raster_receipts = {
            "scalar": acquisition._array_receipt(scalar_raster),
            "annotation": acquisition._array_receipt(annotation_raster),
            "brain_mask": acquisition._array_receipt(brain_mask),
        }
        offset_payload = {
            "offset_index": index,
            "offset_um": float(offset),
            "integer_mass": int(mass),
            "normalized_weight": float(mass / int(masses.sum())),
            "design_physical_displacement_ap_dv_ml_um": design_displacement.tolist(),
            "effective_physical_displacement_ap_dv_ml_um": effective_displacement.tolist(),
            "renderer_center_ap_dv_ml_float32": shifted_center.numpy().tolist(),
            "centre_displacement_max_abs_error_um": centre_errors[0],
            "centre_displacement_axial_error_um": centre_errors[1],
            "centre_displacement_tangential_error_um": centre_errors[2],
            "coordinate_raster_displacement_max_abs_error_um": grid_errors[0],
            "coordinate_raster_displacement_axial_error_um": grid_errors[1],
            "coordinate_raster_displacement_tangential_error_um": grid_errors[2],
            "reused_authenticated_centre_plane_render": reused_centre,
            "grid_array_receipts": grid_receipts,
            "raster_array_receipts": raster_receipts,
        }
        offset_receipts.append(
            {**offset_payload, "offset_render_receipt_sha256": acquisition._payload_sha256(offset_payload)}
        )
    if kernel["render_mode"] == "centre_plane_ablation":
        centre_support = np.asarray(precursor["raster"]["brain_mask"]).copy()
        unit_weight = np.ones(centre_scalar.shape, dtype=np.float32)
        reduced = {
            "scalar": centre_scalar.copy(),
            "centre_plane_annotation": centre_annotation.copy(),
            "centre_plane_support_mask": centre_support,
            "slab_brain_occupancy": centre_support.astype(np.float32),
            "slab_observable_support_mask": centre_support.copy(),
            "slab_modal_annotation": centre_annotation.copy(),
            "slab_label_purity": unit_weight.copy(),
            "centre_label_support_weight": unit_weight.copy(),
            "slab_supervision_weight_or_abstention": {
                "dense_correspondence_weight": centre_support.astype(np.float32),
                "abstention_mask": ~centre_support,
            },
        }
    else:
        reduced = reduce_v2_slab_samples(
            np.stack(scalar_samples), np.stack(annotation_samples), masses, centre_index
        )
    if (
        not np.array_equal(reduced["centre_plane_annotation"], centre_annotation)
        or not np.array_equal(
            reduced["centre_plane_support_mask"], precursor["raster"]["brain_mask"]
        )
        or (
            kernel["render_mode"] == "centre_plane_ablation"
            and acquisition._array_receipt(reduced["scalar"])
            != acquisition._array_receipt(centre_scalar)
        )
    ):
        raise ValueError("v2 slab changed the authenticated centre-plane target")
    diagnostics = {
        "sampling_axis": "physical canonical arbitrary-plane normal, never atlas AP",
        "offset_render_order": "ascending optical_kernel_offsets_um",
        "zero_offset_index": centre_index,
        "zero_offset_render_reused": True,
        "nonzero_offset_render_count": int(offsets.size - 1),
        "maximum_centre_displacement_error_um": float(maxima[0]),
        "maximum_centre_axial_displacement_error_um": float(maxima[1]),
        "maximum_centre_tangential_displacement_error_um": float(maxima[2]),
        "maximum_coordinate_raster_displacement_error_um": float(maxima[3]),
        "maximum_coordinate_raster_axial_displacement_error_um": float(maxima[4]),
        "maximum_coordinate_raster_tangential_displacement_error_um": float(maxima[5]),
        "physical_displacement_tolerance_um": _PHYSICAL_DISPLACEMENT_TOLERANCE_UM,
        "pose_or_tissue_conditioned_rejection_count": 0,
    }
    if float(max(maxima[1], maxima[2], maxima[4], maxima[5])) > _PHYSICAL_DISPLACEMENT_TOLERANCE_UM:
        raise ValueError("v2 slab physical-normal displacement exceeds its frozen tolerance")
    return reduced, offset_receipts, diagnostics


def _centre_plane_reference(precursor: dict[str, object]) -> dict[str, object]:
    raster = precursor["raster"]
    return {
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "global_reference_grid_id": precursor["geometry"]["global_reference_grid_id"],
        "centre_plane_receipt_sha256": precursor["receipt_sha256"],
        "scalar_array_receipt": raster["array_receipts"]["scalar"],
        "annotation_array_receipt": raster["array_receipts"]["annotation"],
        "brain_mask_array_receipt": raster["array_receipts"]["brain_mask"],
        "combined_sha256": raster["combined_sha256"],
    }


def _raster_receipt(raster: dict[str, object]) -> dict[str, object]:
    return {
        "array_receipts": raster["array_receipts"],
        "combined_sha256": raster["combined_sha256"],
        "centre_plane_brain_pixel_count": raster["centre_plane_brain_pixel_count"],
        "slab_observable_pixel_count": raster["slab_observable_pixel_count"],
        "dense_abstention_pixel_count": raster["dense_abstention_pixel_count"],
    }


def _slab_render_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "centre_plane_render_id": artifact["centre_plane_render_id"],
        "v2_context_sha256": artifact["provenance"]["v2_context_sha256"],
        "slab_recipe_id": artifact["slab_recipe_id"],
        "centre_plane_reference": artifact["centre_plane_reference"],
        "offset_render_receipts": artifact["offset_render_receipts"],
        "diagnostics": artifact["diagnostics"],
        "raster_receipt": _raster_receipt(artifact["raster"]),
    }


def make_v2_smoke_global_reference_slab_render(
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
    """Render the predeclared finite-thickness stage for one v2 smoke plane."""
    precursor = acquisition.make_v2_smoke_global_reference_centre_render(
        prepared_context,
        split,
        root_seed,
        sample_index,
        plane_stratum,
        parent_shape_h_w=parent_shape_h_w,
        animal_id=animal_id,
        animal_index=animal_index,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
    )
    assignment = precursor["smoke_case_assignment"]
    kernel = finite_boxcar_kernel(
        assignment["render_mode"], assignment["nominal_cut_thickness_um"]
    )
    if (
        assignment["effective_optical_support_um"] is not None
        and assignment["effective_optical_support_um"] != kernel["effective_optical_support_um"]
    ):
        raise ValueError("fixed smoke assignment and v2 slab kernel disagree")
    recipe = _slab_recipe(precursor, kernel)
    reduced, offset_receipts, diagnostics = _render_offset_samples(
        prepared_context, precursor, kernel
    )
    raster = {**reduced, **_slab_raster_metadata(reduced)}
    artifact = {
        "schema_version": V2_SLAB_SCHEMA,
        "algorithm": V2_SLAB_ALGORITHM,
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "slab_recipe_id": acquisition._payload_sha256(recipe),
        "generator": precursor["generator"],
        "provenance": precursor["provenance"],
        "smoke_case_assignment": assignment,
        "sampling": precursor["sampling"],
        "geometry": precursor["geometry"],
        "centre_plane_reference": _centre_plane_reference(precursor),
        "slab_recipe": recipe,
        "offset_render_receipts": offset_receipts,
        "diagnostics": diagnostics,
        "raster": raster,
    }
    artifact["slab_render_id"] = acquisition._payload_sha256(
        _slab_render_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(v2_slab_render_receipt(artifact))
    return artifact


def v2_slab_render_receipt(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "centre_plane_render_id": artifact["centre_plane_render_id"],
        "slab_recipe_id": artifact["slab_recipe_id"],
        "slab_render_id": artifact["slab_render_id"],
        "generator": artifact["generator"],
        "provenance": artifact["provenance"],
        "smoke_case_assignment": artifact["smoke_case_assignment"],
        "sampling": artifact["sampling"],
        "geometry": artifact["geometry"],
        "centre_plane_reference": artifact["centre_plane_reference"],
        "slab_recipe": artifact["slab_recipe"],
        "offset_render_receipts": artifact["offset_render_receipts"],
        "diagnostics": artifact["diagnostics"],
        "raster_receipt": _raster_receipt(artifact["raster"]),
    }


def replay_v2_smoke_global_reference_slab_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    config = artifact["generator"]["resolved_config"]
    provenance = artifact["provenance"]
    return make_v2_smoke_global_reference_slab_render(
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


def verify_v2_smoke_global_reference_slab_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> None:
    artifact_keys = {
        "schema_version", "algorithm", "v2_plane_realization_id",
        "centre_plane_render_id", "slab_recipe_id", "slab_render_id", "generator",
        "provenance", "smoke_case_assignment", "sampling", "geometry",
        "centre_plane_reference", "slab_recipe", "offset_render_receipts",
        "diagnostics", "raster", "receipt_sha256",
    }
    raster_keys = {
        "scalar", "centre_plane_annotation", "centre_plane_support_mask",
        "slab_brain_occupancy", "slab_observable_support_mask",
        "slab_modal_annotation", "slab_label_purity", "centre_label_support_weight",
        "slab_supervision_weight_or_abstention", "array_receipts", "combined_sha256",
        "centre_plane_brain_pixel_count", "slab_observable_pixel_count",
        "dense_abstention_pixel_count",
    }
    if (
        set(artifact) != artifact_keys
        or set(artifact.get("raster", {})) != raster_keys
        or set(artifact.get("raster", {}).get("slab_supervision_weight_or_abstention", {}))
        != {"dense_correspondence_weight", "abstention_mask"}
    ):
        raise ValueError("v2 slab artifact contains missing or unauthenticated extra fields")
    if artifact["schema_version"] != V2_SLAB_SCHEMA or artifact["algorithm"] != V2_SLAB_ALGORITHM:
        raise ValueError("v2 slab schema or algorithm does not match")
    for forbidden in (
        "acquisition_window_realization_id", "reflection_transform_id",
        "reflection_realization_id", "v2_acquisition_realization_id",
        "synthetic_realization_id",
    ):
        if forbidden in artifact:
            raise ValueError("v2 slab precursor contains a premature downstream ID")
    live_metadata = _slab_raster_metadata(artifact["raster"])
    if any(artifact["raster"].get(key) != value for key, value in live_metadata.items()):
        raise ValueError("v2 slab live array receipts do not match")
    if artifact["slab_recipe_id"] != acquisition._payload_sha256(artifact["slab_recipe"]):
        raise ValueError("v2 slab recipe ID does not match")
    if artifact["slab_render_id"] != acquisition._payload_sha256(
        _slab_render_identity_payload(artifact)
    ):
        raise ValueError("v2 slab render ID does not match")
    if artifact["receipt_sha256"] != acquisition._payload_sha256(v2_slab_render_receipt(artifact)):
        raise ValueError("v2 slab receipt does not match")
    replayed = replay_v2_smoke_global_reference_slab_render(artifact, prepared_context)
    if v2_slab_render_receipt(artifact) != v2_slab_render_receipt(replayed):
        raise ValueError("v2 slab replay receipt does not match")
    replayed_arrays = _slab_arrays(replayed["raster"])
    for name, array in _slab_arrays(artifact["raster"]).items():
        if (
            acquisition._array_receipt(array) != acquisition._array_receipt(replayed_arrays[name])
            or not np.array_equal(array, replayed_arrays[name])
        ):
            raise ValueError("v2 slab replay arrays do not match")


def make_v2_generic_global_reference_slab_render(
    prepared_context: dict[str, object],
    split: str,
    root_seed: int | str,
    sample_index: int,
    plane_stratum: str,
    *,
    nominal_cut_thickness_um: float,
    axial_step_um_max: float = _AXIAL_STEP_UM_MAX,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    animal_id: str | int | None = None,
    animal_index: int | None = None,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
) -> dict[str, object]:
    """Render a generic authenticated finite-thickness arbitrary-plane slab."""
    acquisition._validate_generic_animal_lineage(animal_id, animal_index)
    precursor = acquisition.make_v2_generic_global_reference_centre_render(
        prepared_context,
        split,
        root_seed,
        sample_index,
        plane_stratum,
        parent_shape_h_w=parent_shape_h_w,
        animal_id=animal_id,
        animal_index=animal_index,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
    )
    kernel = finite_boxcar_kernel(
        "finite_boxcar",
        nominal_cut_thickness_um,
        axial_step_um_max=axial_step_um_max,
    )
    recipe = _slab_recipe(
        precursor,
        kernel,
        schema_version=V2_GENERIC_SLAB_SCHEMA,
        algorithm=V2_GENERIC_SLAB_ALGORITHM,
    )
    reduced, offset_receipts, diagnostics = _render_offset_samples(
        prepared_context, precursor, kernel
    )
    raster = {**reduced, **_slab_raster_metadata(reduced)}
    artifact = {
        "schema_version": V2_GENERIC_SLAB_SCHEMA,
        "algorithm": V2_GENERIC_SLAB_ALGORITHM,
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "slab_recipe_id": acquisition._payload_sha256(recipe),
        "generator": precursor["generator"],
        "provenance": precursor["provenance"],
        "sampling": precursor["sampling"],
        "geometry": precursor["geometry"],
        "centre_plane_reference": _centre_plane_reference(precursor),
        "slab_recipe": recipe,
        "offset_render_receipts": offset_receipts,
        "diagnostics": diagnostics,
        "raster": raster,
    }
    artifact["slab_render_id"] = acquisition._payload_sha256(
        _slab_render_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        v2_generic_slab_render_receipt(artifact)
    )
    return artifact


def v2_generic_slab_render_receipt(
    artifact: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "centre_plane_render_id": artifact["centre_plane_render_id"],
        "slab_recipe_id": artifact["slab_recipe_id"],
        "slab_render_id": artifact["slab_render_id"],
        "generator": artifact["generator"],
        "provenance": artifact["provenance"],
        "sampling": artifact["sampling"],
        "geometry": artifact["geometry"],
        "centre_plane_reference": artifact["centre_plane_reference"],
        "slab_recipe": artifact["slab_recipe"],
        "offset_render_receipts": artifact["offset_render_receipts"],
        "diagnostics": artifact["diagnostics"],
        "raster_receipt": _raster_receipt(artifact["raster"]),
    }


def replay_v2_generic_global_reference_slab_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    config = artifact["generator"]["resolved_config"]
    provenance = artifact["provenance"]
    recipe = artifact["slab_recipe"]
    return make_v2_generic_global_reference_slab_render(
        prepared_context,
        config["split"],
        config["root_seed_uint64"],
        config["sample_index"],
        config["plane_stratum"],
        nominal_cut_thickness_um=recipe["nominal_cut_thickness_um"],
        axial_step_um_max=recipe["axial_step_um_max"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=provenance["animal_id"],
        animal_index=provenance["animal_index"],
        specimen_id=provenance["specimen_id"],
        experiment_id=provenance["experiment_id"],
    )


def verify_v2_generic_global_reference_slab_render(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> None:
    artifact_keys = {
        "schema_version",
        "algorithm",
        "v2_plane_realization_id",
        "centre_plane_render_id",
        "slab_recipe_id",
        "slab_render_id",
        "generator",
        "provenance",
        "sampling",
        "geometry",
        "centre_plane_reference",
        "slab_recipe",
        "offset_render_receipts",
        "diagnostics",
        "raster",
        "receipt_sha256",
    }
    raster_keys = {
        "scalar",
        "centre_plane_annotation",
        "centre_plane_support_mask",
        "slab_brain_occupancy",
        "slab_observable_support_mask",
        "slab_modal_annotation",
        "slab_label_purity",
        "centre_label_support_weight",
        "slab_supervision_weight_or_abstention",
        "array_receipts",
        "combined_sha256",
        "centre_plane_brain_pixel_count",
        "slab_observable_pixel_count",
        "dense_abstention_pixel_count",
    }
    if (
        set(artifact) != artifact_keys
        or set(artifact.get("raster", {})) != raster_keys
        or set(
            artifact.get("raster", {}).get(
                "slab_supervision_weight_or_abstention", {}
            )
        )
        != {"dense_correspondence_weight", "abstention_mask"}
    ):
        raise ValueError("generic slab has missing or unauthenticated extra fields")
    acquisition._validate_generic_animal_lineage(
        artifact["provenance"]["animal_id"],
        artifact["provenance"]["animal_index"],
    )
    config = artifact["generator"]["resolved_config"]
    if (
        artifact["schema_version"] != V2_GENERIC_SLAB_SCHEMA
        or artifact["algorithm"] != V2_GENERIC_SLAB_ALGORITHM
        or artifact["slab_recipe"].get("schema_version") != V2_GENERIC_SLAB_SCHEMA
        or artifact["slab_recipe"].get("algorithm") != V2_GENERIC_SLAB_ALGORITHM
        or config.get("schema_version") != acquisition.V2_GENERIC_PLANE_SCHEMA
        or config.get("algorithm") != acquisition.V2_GENERIC_PLANE_ALGORITHM
        or "preflight" in config
        or any(
            config.get(name)
            for name in (
                "learned_checkpoint_dependencies",
                "previous_model_dependencies",
                "pretrained_feature_dependencies",
                "learned_style_model_dependencies",
            )
        )
    ):
        raise ValueError("generic slab schema, precursor, or dependencies disagree")
    for forbidden in (
        "acquisition_window_realization_id",
        "reflection_transform_id",
        "reflection_realization_id",
        "v2_acquisition_realization_id",
        "synthetic_realization_id",
    ):
        if forbidden in artifact:
            raise ValueError("generic slab contains a premature downstream ID")
    live_metadata = _slab_raster_metadata(artifact["raster"])
    if (
        any(artifact["raster"].get(key) != value for key, value in live_metadata.items())
        or artifact["slab_recipe_id"]
        != acquisition._payload_sha256(artifact["slab_recipe"])
        or artifact["slab_render_id"]
        != acquisition._payload_sha256(_slab_render_identity_payload(artifact))
        or artifact["receipt_sha256"]
        != acquisition._payload_sha256(v2_generic_slab_render_receipt(artifact))
    ):
        raise ValueError("generic slab live receipt or identity disagrees")
    replayed = replay_v2_generic_global_reference_slab_render(
        artifact, prepared_context
    )
    if v2_generic_slab_render_receipt(artifact) != v2_generic_slab_render_receipt(
        replayed
    ):
        raise ValueError("generic slab deterministic replay receipt disagrees")
    replayed_arrays = _slab_arrays(replayed["raster"])
    for name, array in _slab_arrays(artifact["raster"]).items():
        if not np.array_equal(array, replayed_arrays[name]):
            raise ValueError("generic slab deterministic replay arrays disagree")


def compare_v2_slab_axial_refinement(
    artifact: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    """Evaluate and receipt the predeclared 12.5-to-6.25-um refinement gate."""
    verify_v2_smoke_global_reference_slab_render(artifact, prepared_context)
    if artifact["slab_recipe"]["render_mode"] != "finite_boxcar":
        raise ValueError("axial refinement applies only to finite-boxcar renders")
    config = artifact["generator"]["resolved_config"]
    provenance = artifact["provenance"]
    precursor = acquisition.make_v2_smoke_global_reference_centre_render(
        prepared_context,
        config["split"], config["root_seed_uint64"], config["sample_index"],
        config["plane_stratum"], parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=provenance["animal_id"], animal_index=provenance["animal_index"],
        specimen_id=provenance["specimen_id"],
        experiment_id=provenance["experiment_id"],
    )
    fine_kernel = finite_boxcar_kernel(
        "finite_boxcar", artifact["slab_recipe"]["nominal_cut_thickness_um"],
        axial_step_um_max=6.25,
    )
    fine, fine_offsets, fine_diagnostics = _render_offset_samples(
        prepared_context, precursor, fine_kernel
    )
    fine_metadata = _slab_raster_metadata(fine)
    coarse_scalar = np.asarray(artifact["raster"]["scalar"], dtype=np.float64)
    fine_scalar = np.asarray(fine["scalar"], dtype=np.float64)
    coarse_support = np.asarray(artifact["raster"]["slab_brain_occupancy"], dtype=np.float64)
    fine_support = np.asarray(fine["slab_brain_occupancy"], dtype=np.float64)
    union = (coarse_support > 0.0) | (fine_support > 0.0)
    if not union.any():
        raise ValueError("axial refinement has no nonzero support pixels")
    if (
        not np.array_equal(
            artifact["raster"]["centre_plane_annotation"],
            fine["centre_plane_annotation"],
        )
        or not np.array_equal(
            artifact["raster"]["centre_plane_support_mask"],
            fine["centre_plane_support_mask"],
        )
    ):
        raise ValueError("axial refinement changed the immutable centre-plane target")
    scalar_tensor = prepared_context["opaque_v1_context"]["scalar_tensor"]
    scalar_range = max(float(scalar_tensor.max().item() - scalar_tensor.min().item()), 1.0)
    scalar_error = np.abs(coarse_scalar[union] - fine_scalar[union]) / scalar_range
    support_error = np.abs(coarse_support[union] - fine_support[union])
    purity_error = np.abs(
        np.asarray(artifact["raster"]["slab_label_purity"], dtype=np.float64)[union]
        - np.asarray(fine["slab_label_purity"], dtype=np.float64)[union]
    )
    centre_support_error = np.abs(
        np.asarray(
            artifact["raster"]["centre_label_support_weight"], dtype=np.float64
        )[union]
        - np.asarray(fine["centre_label_support_weight"], dtype=np.float64)[union]
    )
    coarse_dense = artifact["raster"]["slab_supervision_weight_or_abstention"]
    fine_dense = fine["slab_supervision_weight_or_abstention"]
    dense_weight_error = np.abs(
        np.asarray(coarse_dense["dense_correspondence_weight"], dtype=np.float64)[union]
        - np.asarray(fine_dense["dense_correspondence_weight"], dtype=np.float64)[union]
    )
    metrics = {
        "normalized_scalar_mae": float(scalar_error.mean()),
        "normalized_scalar_absolute_error_p99": float(
            np.quantile(scalar_error, 0.99, method="linear")
        ),
        "support_mass_mae": float(support_error.mean()),
        "support_mass_absolute_error_p99": float(
            np.quantile(support_error, 0.99, method="linear")
        ),
        "slab_label_purity_mae": float(purity_error.mean()),
        "slab_label_purity_absolute_error_p99": float(
            np.quantile(purity_error, 0.99, method="linear")
        ),
        "centre_label_support_weight_mae": float(centre_support_error.mean()),
        "centre_label_support_weight_absolute_error_p99": float(
            np.quantile(centre_support_error, 0.99, method="linear")
        ),
        "dense_correspondence_weight_mae": float(dense_weight_error.mean()),
        "dense_correspondence_weight_absolute_error_p99": float(
            np.quantile(dense_weight_error, 0.99, method="linear")
        ),
        "slab_modal_annotation_disagreement_fraction": float(
            np.mean(
                np.asarray(artifact["raster"]["slab_modal_annotation"])[union]
                != np.asarray(fine["slab_modal_annotation"])[union]
            )
        ),
        "slab_observable_support_mask_disagreement_fraction": float(
            np.mean(
                np.asarray(artifact["raster"]["slab_observable_support_mask"])[union]
                != np.asarray(fine["slab_observable_support_mask"])[union]
            )
        ),
        "dense_correspondence_abstention_disagreement_fraction": float(
            np.mean(
                np.asarray(coarse_dense["abstention_mask"])[union]
                != np.asarray(fine_dense["abstention_mask"])[union]
            )
        ),
    }
    payload = {
        "schema_version": "anatomy-tracker.v2-slab-axial-refinement-case/v1",
        "sample_index": config["sample_index"],
        "plane_stratum": config["plane_stratum"],
        "v2_context_sha256": provenance["v2_context_sha256"],
        "v2_plane_realization_id": artifact["v2_plane_realization_id"],
        "coarse_slab_recipe_id": artifact["slab_recipe_id"],
        "coarse_slab_render_id": artifact["slab_render_id"],
        "coarse_axial_step_um_max": 12.5,
        "refined_axial_step_um_max": 6.25,
        "refined_kernel": fine_kernel,
        "refined_offset_render_receipts": fine_offsets,
        "refined_diagnostics": fine_diagnostics,
        "refined_raster_receipt": fine_metadata,
        "union_nonzero_support_pixel_count": int(union.sum()),
        "authenticated_scalar_range_denominator": scalar_range,
        "metrics": metrics,
        "thresholds": {"mae": 0.02, "absolute_error_p99": 0.10},
    }
    payload["passed"] = bool(
        all(
            value <= 0.02
            for name, value in metrics.items()
            if not name.endswith("absolute_error_p99")
        )
        and all(
            value <= 0.10
            for name, value in metrics.items()
            if name.endswith("absolute_error_p99")
        )
    )
    payload["case_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def evaluate_v2_slab_refinement_smoke(
    prepared_context: dict[str, object],
) -> dict[str, object]:
    """Run the provenance-bound refinement gate over all 17 finite CCF cases."""
    cases = []
    for sample_index, assignment in enumerate(acquisition.V2_SMOKE_ASSIGNMENTS):
        if assignment[3] != "finite_boxcar":
            continue
        artifact = make_v2_smoke_global_reference_slab_render(
            prepared_context,
            "development",
            "0x415154564f320001",
            sample_index,
            acquisition.V2_PLANE_STRATA[sample_index // 4],
        )
        cases.append(compare_v2_slab_axial_refinement(artifact, prepared_context))
    payload = {
        "schema_version": "anatomy-tracker.v2-slab-axial-refinement-smoke/v1",
        "claim_scope": "undeformed CCF precursor quadrature qualification only; no model scoring or benchmark",
        "subject_deformed_qualification_status": "pending; not evaluated by this report",
        "split": "development",
        "root_seed_uint64": "0x415154564f320001",
        "v2_context_sha256": prepared_context["v2_context_sha256"],
        "finite_case_indices": [case["sample_index"] for case in cases],
        "case_count": len(cases),
        "cases": cases,
        "all_cases_passed": all(case["passed"] for case in cases),
        "source_sha256": _source_hashes(),
        "source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runner_source_sha256": acquisition._normalized_text_sha256(
            _SOURCE_ROOT / "run_arbitrary_plane_slab_qualification.py"
        ),
    }
    payload["qualification_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def verify_v2_slab_refinement_qualification(
    report: dict[str, object], prepared_context: dict[str, object]
) -> None:
    expected_indices = [
        index for index, assignment in enumerate(acquisition.V2_SMOKE_ASSIGNMENTS)
        if assignment[3] == "finite_boxcar"
    ]
    report_keys = {
        "schema_version", "claim_scope", "subject_deformed_qualification_status",
        "split", "root_seed_uint64",
        "v2_context_sha256", "finite_case_indices", "case_count", "cases",
        "all_cases_passed", "source_sha256", "source_sha256_canonicalization",
        "runner_source_sha256", "qualification_receipt_sha256",
    }
    case_keys = {
        "schema_version", "sample_index", "plane_stratum", "v2_context_sha256",
        "v2_plane_realization_id", "coarse_slab_recipe_id", "coarse_slab_render_id",
        "coarse_axial_step_um_max", "refined_axial_step_um_max", "refined_kernel",
        "refined_offset_render_receipts", "refined_diagnostics", "refined_raster_receipt",
        "union_nonzero_support_pixel_count", "authenticated_scalar_range_denominator",
        "metrics", "thresholds", "passed", "case_receipt_sha256",
    }
    metric_keys = {
        "normalized_scalar_mae", "normalized_scalar_absolute_error_p99",
        "support_mass_mae", "support_mass_absolute_error_p99",
        "slab_label_purity_mae", "slab_label_purity_absolute_error_p99",
        "centre_label_support_weight_mae",
        "centre_label_support_weight_absolute_error_p99",
        "dense_correspondence_weight_mae",
        "dense_correspondence_weight_absolute_error_p99",
        "slab_modal_annotation_disagreement_fraction",
        "slab_observable_support_mask_disagreement_fraction",
        "dense_correspondence_abstention_disagreement_fraction",
    }
    if set(report) != report_keys or not isinstance(report["cases"], list):
        raise ValueError("v2 slab refinement qualification schema does not match")
    cases = report["cases"]
    semantic_cases_match = True
    for expected_index, case in zip(expected_indices, cases):
        metrics = case.get("metrics", {})
        thresholds = case.get("thresholds", {})
        expected_pass = (
            set(metrics) == metric_keys
            and set(thresholds) == {"mae", "absolute_error_p99"}
            and all(math.isfinite(float(value)) for value in metrics.values())
            and thresholds == {"mae": 0.02, "absolute_error_p99": 0.10}
            and all(
                value <= 0.02
                for name, value in metrics.items()
                if not name.endswith("absolute_error_p99")
            )
            and all(
                value <= 0.10
                for name, value in metrics.items()
                if name.endswith("absolute_error_p99")
            )
        )
        semantic_cases_match &= (
            set(case) == case_keys
            and case.get("schema_version")
            == "anatomy-tracker.v2-slab-axial-refinement-case/v1"
            and case.get("sample_index") == expected_index
            and case.get("plane_stratum") == acquisition.V2_PLANE_STRATA[expected_index // 4]
            and case.get("v2_context_sha256") == prepared_context["v2_context_sha256"]
            and case.get("coarse_axial_step_um_max") == 12.5
            and case.get("refined_axial_step_um_max") == 6.25
            and case.get("union_nonzero_support_pixel_count", 0) > 0
            and case.get("authenticated_scalar_range_denominator", 0.0) > 0.0
            and case.get("passed") is expected_pass
            and case.get("case_receipt_sha256")
            == acquisition._payload_sha256(
                {key: value for key, value in case.items() if key != "case_receipt_sha256"}
            )
        )
    payload = {
        key: value for key, value in report.items() if key != "qualification_receipt_sha256"
    }
    if (
        report["qualification_receipt_sha256"] != acquisition._payload_sha256(payload)
        or report["schema_version"]
        != "anatomy-tracker.v2-slab-axial-refinement-smoke/v1"
        or report["claim_scope"]
        != "undeformed CCF precursor quadrature qualification only; no model scoring or benchmark"
        or report["subject_deformed_qualification_status"]
        != "pending; not evaluated by this report"
        or report["split"] != "development"
        or report["root_seed_uint64"] != "0x415154564f320001"
        or report["v2_context_sha256"] != prepared_context["v2_context_sha256"]
        or report["finite_case_indices"] != expected_indices
        or report["case_count"] != 17
        or len(cases) != 17
        or not semantic_cases_match
        or report["all_cases_passed"] is not all(case["passed"] for case in cases)
        or report["all_cases_passed"] is not True
        or report["source_sha256"] != _source_hashes()
        or report["source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or report["runner_source_sha256"]
        != acquisition._normalized_text_sha256(
            _SOURCE_ROOT / "run_arbitrary_plane_slab_qualification.py"
        )
    ):
        raise ValueError("v2 slab refinement qualification did not pass all 17 finite cases")
    expected = evaluate_v2_slab_refinement_smoke(prepared_context)
    if acquisition._canonical_json(report) != acquisition._canonical_json(expected):
        raise ValueError("v2 slab refinement qualification replay does not match")


def save_v2_slab_refinement_qualification(
    path: str | Path, report: dict[str, object]
) -> None:
    """Persist raw per-case metrics and provenance before enforcing the gate."""
    expected = acquisition._payload_sha256(
        {key: value for key, value in report.items() if key != "qualification_receipt_sha256"}
    )
    if report.get("qualification_receipt_sha256") != expected:
        raise ValueError("v2 slab refinement qualification receipt does not match")
    Path(path).write_text(
        acquisition._canonical_json(report) + "\n", encoding="utf-8", newline="\n"
    )
