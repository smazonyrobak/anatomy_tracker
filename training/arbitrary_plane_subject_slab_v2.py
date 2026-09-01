"""Authenticated subject-space slab pullback and reduction stage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_synthetic_generator_v2 as synthetic_generator
from training.arbitrary_plane_geometry import (
    allen_index_to_physical_um_points,
    physical_um_to_allen_index_points,
    physical_um_to_allen_index_vectors,
)
from training.arbitrary_plane_rendered_generator import effective_renderer_sampling_arrays
from training.arbitrary_plane_subject_deformation_v2 import (
    _verified_subject_to_ccf_mapper_v2,
    subject_deformation_plan_receipt_v2,
)
from training.arbitrary_plane_subject_section_v2 import (
    fit_subject_centre_plane_and_residual_v2,
    sample_coordinate_rasters_v2,
    sample_nearest_annotation_coordinate_rasters_v2,
    verify_subject_centre_plane_fit_v2,
)
from training.arbitrary_plane_synthetic_generator_v2 import (
    V2_GENERIC_SLAB_ALGORITHM,
    V2_GENERIC_SLAB_SCHEMA,
    V2_SLAB_ALGORITHM,
    V2_SLAB_SCHEMA,
    reduce_v2_slab_samples,
    v2_generic_slab_render_receipt,
    v2_slab_render_receipt,
    verify_v2_generic_global_reference_slab_render,
    verify_v2_smoke_global_reference_slab_render,
)


SUBJECT_COORDINATE_MAP_V2_SCHEMA = "anatomy-tracker.subject-coordinate-map/v2"
SUBJECT_COORDINATE_MAP_V2_ALGORITHM = (
    "legacy-float32-subject-offset-grid-then-subject-to-ccf-pullback/v2"
)
SUBJECT_CENTRE_SUPPORT_PROBE_V2_SCHEMA = (
    "anatomy-tracker.subject-centre-support-probe/v2"
)
SUBJECT_CENTRE_SUPPORT_PROBE_V2_ALGORITHM = (
    "mapped-centre-nearest-annotation-support-only/v2"
)
SUBJECT_SLAB_RENDER_V2_SCHEMA = "anatomy-tracker.subject-slab-render/v2"
SUBJECT_SLAB_RENDER_V2_ALGORITHM = (
    "authenticated-ccf-sample-and-precursor-parity-finite-boxcar-reduce/v2"
)
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_section_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_geometry.py",
    "arbitrary_plane_rendered_generator.py",
)
_COORDINATE_ARRAY_KEYS = {
    "subject_renderer_centres_allen_index_float32",
    "subject_allen_index_coordinates_float32",
    "subject_physical_coordinates_ap_dv_ml_um_float64",
    "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64",
    "mapped_allen_index_coordinates_float32",
}
_REDUCED_ARRAY_KEYS = {
    "scalar",
    "centre_plane_annotation",
    "centre_plane_support_mask",
    "slab_brain_occupancy",
    "slab_observable_support_mask",
    "slab_modal_annotation",
    "slab_label_purity",
    "centre_label_support_weight",
    "dense_correspondence_weight",
    "dense_correspondence_abstention_mask",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _receipts(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {name: acquisition._array_receipt(value) for name, value in arrays.items()}


def _byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _context_reference(prepared_context: dict[str, object]) -> dict[str, object]:
    return {
        "schema": prepared_context["schema"],
        "v2_context_sha256": prepared_context["v2_context_sha256"],
        "prepared_context_receipt_sha256": acquisition._payload_sha256(
            acquisition._json_value(prepared_context["receipt"])
        ),
    }


def _verified_subject_to_ccf_mapper_for_context_v2(
    prepared_context: dict[str, object],
    subject_plan: dict[str, object] | None,
) -> Callable[..., np.ndarray] | None:
    if subject_plan is None:
        return None
    acquisition._validate_v2_context(prepared_context)
    support = acquisition._context_support(prepared_context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(
        support["annotation_shape"], dtype=np.float64
    ) * np.asarray(support["voxel_size_um"], dtype=np.float64)
    return _verified_subject_to_ccf_mapper_v2(
        subject_plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )


def _precursor_contract_and_receipt(
    precursor: dict[str, object], prepared_context: dict[str, object] | None = None
) -> tuple[str, dict[str, object]]:
    identity = (precursor.get("schema_version"), precursor.get("algorithm"))
    if identity == (V2_SLAB_SCHEMA, V2_SLAB_ALGORITHM):
        if prepared_context is not None:
            verify_v2_smoke_global_reference_slab_render(precursor, prepared_context)
        return "frozen-smoke-v2", v2_slab_render_receipt(precursor)
    if identity == (V2_GENERIC_SLAB_SCHEMA, V2_GENERIC_SLAB_ALGORITHM):
        if prepared_context is not None:
            verify_v2_generic_global_reference_slab_render(precursor, prepared_context)
        return "authenticated-generic-v2", v2_generic_slab_render_receipt(precursor)
    raise ValueError("subject slab precursor schema/algorithm pair is not supported")


def _verify_precursor_receipt_binding_without_render_replay(
    precursor: dict[str, object], prepared_context: dict[str, object]
) -> None:
    contract, receipt = _precursor_contract_and_receipt(precursor)
    config = precursor["generator"]["resolved_config"]
    provenance = precursor["provenance"]
    context_receipt = prepared_context["receipt"]
    recipe = precursor["slab_recipe"]
    if contract == "authenticated-generic-v2":
        acquisition._validate_generic_animal_lineage(
            provenance["animal_id"], provenance["animal_index"]
        )
    if (
        precursor["receipt_sha256"] != acquisition._payload_sha256(receipt)
        or precursor["slab_recipe_id"] != acquisition._payload_sha256(recipe)
        or precursor["slab_render_id"]
        != acquisition._payload_sha256(
            synthetic_generator._slab_render_identity_payload(precursor)
        )
        or precursor["generator"]["resolved_config_sha256"]
        != acquisition._payload_sha256(config)
        or config["source_sha256"] != acquisition._source_hashes()
        or config["source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or config["runtime"] != acquisition._runtime_receipt()
        or recipe["implementation_source_sha256"]
        != synthetic_generator._source_hashes()
        or recipe["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or any(
            config[name]
            for name in (
                "learned_checkpoint_dependencies",
                "previous_model_dependencies",
                "pretrained_feature_dependencies",
                "learned_style_model_dependencies",
            )
        )
        or (contract == "authenticated-generic-v2" and "preflight" in config)
        or (contract == "frozen-smoke-v2" and "preflight" not in config)
        or provenance["v2_context_sha256"]
        != prepared_context["v2_context_sha256"]
        or provenance["opaque_v1_prepared_context_sha256"]
        != context_receipt["opaque_v1_prepared_context_sha256"]
        or provenance["support_index_sha256"]
        != context_receipt["support_index_sha256"]
        or provenance["annotation_array_sha256"]
        != context_receipt["annotation_array_sha256"]
        or provenance["scalar_source_sha256"]
        != context_receipt["scalar_source"]["source_sha256"]
        or any(
            name in precursor
            for name in (
                "acquisition_window_realization_id",
                "reflection_transform_id",
                "reflection_realization_id",
                "v2_acquisition_realization_id",
                "synthetic_realization_id",
            )
        )
    ):
        raise ValueError("subject support precursor source, context, or live receipt disagrees")


def _precursor_reference(precursor: dict[str, object]) -> dict[str, object]:
    config = precursor["generator"]["resolved_config"]
    provenance = precursor["provenance"]
    contract, receipt = _precursor_contract_and_receipt(precursor)
    return {
        "precursor_contract": contract,
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "global_reference_grid_id": precursor["geometry"]["global_reference_grid_id"],
        "slab_recipe_id": precursor["slab_recipe_id"],
        "slab_render_id": precursor["slab_render_id"],
        "v2_slab_render_receipt_sha256": acquisition._payload_sha256(
            receipt
        ),
        "v2_context_sha256": precursor["provenance"]["v2_context_sha256"],
        "animal_id": provenance["animal_id"],
        "animal_index": provenance["animal_index"],
        "split": config["split"],
        "plane_sample_index": config["sample_index"],
    }


def _deformation_reference(subject_plan: dict[str, object] | None) -> dict[str, object]:
    return {
        "mode": (
            "authenticated-identity-reference"
            if subject_plan is None
            else "accepted-subject-deformation"
        ),
        "subject_deformation_plan_receipt": (
            None
            if subject_plan is None
            else acquisition._json_value(subject_deformation_plan_receipt_v2(subject_plan))
        ),
        "synthetic_animal_id": (
            None if subject_plan is None else subject_plan["synthetic_animal_id"]
        ),
    }


def _lineage_reference(
    precursor: dict[str, object], subject_plan: dict[str, object] | None
) -> dict[str, object]:
    provenance = precursor["provenance"]
    config = precursor["generator"]["resolved_config"]
    return {
        "split": config["split"],
        "plane_sample_index": config["sample_index"],
        "animal_id": acquisition._json_value(provenance["animal_id"]),
        "animal_index": provenance["animal_index"],
        "specimen_id": acquisition._json_value(provenance["specimen_id"]),
        "experiment_id": acquisition._json_value(provenance["experiment_id"]),
        "synthetic_animal_id": (
            None if subject_plan is None else subject_plan["synthetic_animal_id"]
        ),
    }


def _atlas_domain(support: dict[str, object]) -> dict[str, object]:
    origin = np.asarray(support["origin_um"], dtype=np.float64)
    spacing = np.asarray(support["voxel_size_um"], dtype=np.float64)
    shape = np.asarray(support["annotation_shape"], dtype=np.int64)
    return {
        "axis_order": ["AP", "DV", "ML"],
        "origin_ap_dv_ml_um": origin.tolist(),
        "voxel_size_ap_dv_ml_um": spacing.tolist(),
        "shape_ap_dv_ml": shape.tolist(),
        "voxel_face_lower_closed_ap_dv_ml_um": origin.tolist(),
        "voxel_face_upper_closed_ap_dv_ml_um": (origin + shape * spacing).tolist(),
        "voxel_centre_convention": "origin + (allen_index + 0.5) * voxel_size",
    }


def _reduced_arrays(raster: dict[str, object]) -> dict[str, np.ndarray]:
    supervision = raster["slab_supervision_weight_or_abstention"]
    return {
        "scalar": np.asarray(raster["scalar"]),
        "centre_plane_annotation": np.asarray(raster["centre_plane_annotation"]),
        "centre_plane_support_mask": np.asarray(raster["centre_plane_support_mask"]),
        "slab_brain_occupancy": np.asarray(raster["slab_brain_occupancy"]),
        "slab_observable_support_mask": np.asarray(
            raster["slab_observable_support_mask"]
        ),
        "slab_modal_annotation": np.asarray(raster["slab_modal_annotation"]),
        "slab_label_purity": np.asarray(raster["slab_label_purity"]),
        "centre_label_support_weight": np.asarray(
            raster["centre_label_support_weight"]
        ),
        "dense_correspondence_weight": np.asarray(
            supervision["dense_correspondence_weight"]
        ),
        "dense_correspondence_abstention_mask": np.asarray(
            supervision["abstention_mask"]
        ),
    }


def _nest_reduced(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        **{name: value for name, value in arrays.items() if not name.startswith("dense_")},
        "slab_supervision_weight_or_abstention": {
            "dense_correspondence_weight": arrays["dense_correspondence_weight"],
            "abstention_mask": arrays["dense_correspondence_abstention_mask"],
        },
    }


def _reduce_samples_like_precursor(
    scalar_samples: np.ndarray,
    annotation_samples: np.ndarray,
    integer_masses: np.ndarray,
    centre_index: int,
    render_mode: str,
) -> dict[str, np.ndarray]:
    if render_mode != "centre_plane_ablation":
        return _reduced_arrays(
            reduce_v2_slab_samples(
                scalar_samples, annotation_samples, integer_masses, centre_index
            )
        )
    scalar = scalar_samples[centre_index].copy()
    annotation = annotation_samples[centre_index].copy()
    support = annotation != 0
    unit_weight = np.ones(scalar.shape, dtype=np.float32)
    return {
        "scalar": scalar,
        "centre_plane_annotation": annotation,
        "centre_plane_support_mask": support,
        "slab_brain_occupancy": support.astype(np.float32),
        "slab_observable_support_mask": support.copy(),
        "slab_modal_annotation": annotation.copy(),
        "slab_label_purity": unit_weight.copy(),
        "centre_label_support_weight": unit_weight.copy(),
        "dense_correspondence_weight": support.astype(np.float32),
        "dense_correspondence_abstention_mask": ~support,
    }


def _subject_offset_grids(
    precursor: dict[str, object],
    support: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    geometry = precursor["geometry"]
    recipe = precursor["slab_recipe"]
    origin = tuple(float(value) for value in support["origin_um"])
    spacing = tuple(float(value) for value in support["voxel_size_um"])
    shape = tuple(int(value) for value in support["annotation_shape"])
    normal = np.asarray(geometry["normal_rp2_ap_dv_ml"], dtype=np.float64)
    offsets = np.asarray(recipe["optical_kernel_offsets_um"], dtype=np.float64)
    masses = np.asarray(recipe["optical_kernel_integer_masses"], dtype=np.int64)
    base_center = torch.as_tensor(
        geometry["renderer_center_ap_dv_ml"], dtype=torch.float32
    )
    centres = []
    coordinates = []
    if len(precursor["offset_render_receipts"]) != len(offsets):
        raise ValueError("authenticated precursor offset count does not match its kernel")
    for index, (offset, mass) in enumerate(zip(offsets, masses)):
        delta_index = physical_um_to_allen_index_vectors(
            torch.as_tensor(float(offset) * normal, dtype=torch.float64), spacing
        ).to(torch.float32)
        shifted_center = base_center + delta_index
        shifted_geometry = dict(geometry)
        shifted_geometry["renderer_center_ap_dv_ml"] = shifted_center.tolist()
        grid = effective_renderer_sampling_arrays(
            shifted_geometry,
            shape,
            origin_ap_dv_ml_um=origin,
            voxel_size_ap_dv_ml_um=spacing,
        )
        precursor_offset = precursor["offset_render_receipts"][index]
        if (
            precursor_offset["offset_index"] != index
            or precursor_offset["offset_um"] != float(offset)
            or precursor_offset["integer_mass"] != int(mass)
            or precursor_offset["renderer_center_ap_dv_ml_float32"]
            != shifted_center.numpy().tolist()
            or precursor_offset["grid_array_receipts"] != _receipts(grid)
        ):
            raise ValueError(
                "subject offset grid does not reproduce the authenticated legacy renderer"
            )
        centres.append(shifted_center.numpy())
        coordinates.append(grid["coordinate_raster_allen_index_float32"])
    return (
        np.ascontiguousarray(np.stack(centres), dtype=np.float32),
        np.ascontiguousarray(np.stack(coordinates), dtype=np.float32),
    )


def _subject_domain_state(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    full_precursor_verification: bool,
) -> dict[str, object]:
    acquisition._validate_v2_context(prepared_context)
    if full_precursor_verification:
        _precursor_contract_and_receipt(precursor, prepared_context)
    else:
        _verify_precursor_receipt_binding_without_render_replay(
            precursor, prepared_context
        )
    support = acquisition._context_support(prepared_context)
    annotation_volume = prepared_context["opaque_v1_context"]["annotation_tensor"]
    origin = tuple(float(value) for value in support["origin_um"])
    spacing = tuple(float(value) for value in support["voxel_size_um"])
    shape = tuple(int(value) for value in support["annotation_shape"])
    lower = np.asarray(origin, dtype=np.float64)
    upper = lower + np.asarray(shape, dtype=np.float64) * np.asarray(
        spacing, dtype=np.float64
    )
    if tuple(annotation_volume.shape) != shape:
        raise ValueError("prepared context annotation tensor and support geometry disagree")
    return {
        "support": support,
        "annotation_volume": annotation_volume,
        "origin": origin,
        "spacing": spacing,
        "shape": shape,
        "lower": lower,
        "upper": upper,
    }


def _map_subject_physical_points(
    subject_physical: np.ndarray,
    subject_allen: np.ndarray,
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    domain: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str | None,
    Callable[..., np.ndarray] | None,
]:
    if subject_plan is None:
        mapped_physical = np.array(subject_physical, copy=True, order="C")
        mapped_allen = np.array(subject_allen, copy=True, order="C")
        return mapped_physical, mapped_allen, None, None

    if subject_to_ccf_mapper is None:
        subject_to_ccf_mapper = _verified_subject_to_ccf_mapper_v2(
            subject_plan,
            expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
            expected_full_ccf_lower_um=domain["lower"],
            expected_full_ccf_upper_um=domain["upper"],
        )
    verified_snapshot = getattr(
        subject_to_ccf_mapper,
        "_verified_subject_deformation_snapshot_v2",
        None,
    )
    if (
        getattr(
            subject_to_ccf_mapper,
            "_verified_subject_deformation_plan_v2",
            None,
        )
        is not subject_plan
        or verified_snapshot is None
        or subject_deformation_plan_receipt_v2(verified_snapshot)
        != subject_deformation_plan_receipt_v2(subject_plan)
    ):
        raise ValueError("verified subject mapper does not capture the exact plan")
    plan_provenance = subject_plan["provenance"]
    precursor_provenance = precursor["provenance"]
    precursor_config = precursor["generator"]["resolved_config"]
    if (
        precursor_provenance["animal_id"] != plan_provenance["animal_id"]
        or precursor_provenance["animal_index"] != plan_provenance["animal_index"]
        or precursor_config["split"] != plan_provenance["split"]
    ):
        raise ValueError("subject slab precursor and deformation animal lineage disagree")
    mapped_physical = np.ascontiguousarray(
        subject_to_ccf_mapper(subject_physical, batch_size=batch_size),
        dtype=np.float64,
    )
    mapped_allen = np.ascontiguousarray(
        physical_um_to_allen_index_points(
            torch.from_numpy(mapped_physical),
            domain["origin"],
            domain["spacing"],
        )
        .to(torch.float32)
        .numpy(),
        dtype=np.float32,
    )
    return (
        mapped_physical,
        mapped_allen,
        subject_plan["synthetic_animal_id"],
        subject_to_ccf_mapper,
    )


def _subject_centre_grid(
    precursor: dict[str, object], support: dict[str, object]
) -> tuple[np.ndarray, int]:
    geometry = precursor["geometry"]
    offsets = np.asarray(
        precursor["slab_recipe"]["optical_kernel_offsets_um"], dtype=np.float64
    )
    masses = np.asarray(
        precursor["slab_recipe"]["optical_kernel_integer_masses"], dtype=np.int64
    )
    centre_index = int(np.flatnonzero(offsets == 0.0)[0])
    grid = effective_renderer_sampling_arrays(
        geometry,
        tuple(int(value) for value in support["annotation_shape"]),
        origin_ap_dv_ml_um=tuple(float(value) for value in support["origin_um"]),
        voxel_size_ap_dv_ml_um=tuple(
            float(value) for value in support["voxel_size_um"]
        ),
    )
    centre_receipt = precursor["offset_render_receipts"][centre_index]
    renderer_center = torch.as_tensor(
        geometry["renderer_center_ap_dv_ml"], dtype=torch.float32
    ).numpy()
    if (
        centre_receipt["offset_index"] != centre_index
        or centre_receipt["offset_um"] != 0.0
        or centre_receipt["integer_mass"] != int(masses[centre_index])
        or centre_receipt["renderer_center_ap_dv_ml_float32"]
        != renderer_center.tolist()
        or centre_receipt["grid_array_receipts"] != _receipts(grid)
    ):
        raise ValueError(
            "subject centre grid does not reproduce the authenticated legacy renderer"
        )
    return (
        np.ascontiguousarray(
            grid["coordinate_raster_allen_index_float32"], dtype=np.float32
        ),
        centre_index,
    )


def _subject_centre_support_state(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> dict[str, object]:
    domain = _subject_domain_state(
        prepared_context, precursor, full_precursor_verification=False
    )
    subject_allen, centre_index = _subject_centre_grid(
        precursor, domain["support"]
    )
    subject_physical = np.ascontiguousarray(
        allen_index_to_physical_um_points(
            torch.from_numpy(subject_allen).to(torch.float64),
            domain["origin"],
            domain["spacing"],
        ).numpy(),
        dtype=np.float64,
    )
    _, mapped_allen, synthetic_animal_id, subject_to_ccf_mapper = (
        _map_subject_physical_points(
            subject_physical,
            subject_allen,
            prepared_context,
            precursor,
            domain,
            subject_plan=subject_plan,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
    )
    return {
        "annotation_volume": domain["annotation_volume"],
        "mapped_centre_allen": mapped_allen,
        "centre_index": centre_index,
        "source_hashes": _source_hashes(),
        "context_reference": _context_reference(prepared_context),
        "precursor_reference": _precursor_reference(precursor),
        "deformation_reference": _deformation_reference(subject_plan),
        "synthetic_animal_id": synthetic_animal_id,
        "subject_to_ccf_mapper": subject_to_ccf_mapper,
    }


def _subject_coordinate_state(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> dict[str, object]:
    domain = _subject_domain_state(
        prepared_context, precursor, full_precursor_verification=True
    )
    support = domain["support"]
    renderer_centres, subject_allen = _subject_offset_grids(precursor, support)
    subject_physical = np.ascontiguousarray(
        allen_index_to_physical_um_points(
            torch.from_numpy(subject_allen).to(torch.float64),
            domain["origin"],
            domain["spacing"],
        ).numpy(),
        dtype=np.float64,
    )
    identity = subject_plan is None
    mapped_physical, mapped_allen, synthetic_animal_id, _ = (
        _map_subject_physical_points(
            subject_physical,
            subject_allen,
            prepared_context,
            precursor,
            domain,
            subject_plan=subject_plan,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
    )

    offsets = np.asarray(
        precursor["slab_recipe"]["optical_kernel_offsets_um"], dtype=np.float64
    )
    masses = np.asarray(
        precursor["slab_recipe"]["optical_kernel_integer_masses"], dtype=np.int64
    )
    centre_index = int(np.flatnonzero(offsets == 0.0)[0])
    fit = fit_subject_centre_plane_and_residual_v2(mapped_physical[centre_index])
    coordinate_arrays = {
        "subject_renderer_centres_allen_index_float32": renderer_centres,
        "subject_allen_index_coordinates_float32": subject_allen,
        "subject_physical_coordinates_ap_dv_ml_um_float64": subject_physical,
        "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64": mapped_physical,
        "mapped_allen_index_coordinates_float32": mapped_allen,
    }
    source_hashes = _source_hashes()
    context_reference = _context_reference(prepared_context)
    precursor_reference = _precursor_reference(precursor)
    coordinate_map = {
        "schema_version": SUBJECT_COORDINATE_MAP_V2_SCHEMA,
        "algorithm": SUBJECT_COORDINATE_MAP_V2_ALGORITHM,
        "implementation_source_sha256": source_hashes,
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "context_reference": context_reference,
        "precursor_reference": precursor_reference,
        "deformation_reference": _deformation_reference(subject_plan),
        "synthetic_animal_id": synthetic_animal_id,
        "atlas_domain": _atlas_domain(support),
        "kernel": {
            "render_mode": precursor["slab_recipe"]["render_mode"],
            "offsets_um": offsets.tolist(),
            "integer_masses": masses.tolist(),
            "centre_index": centre_index,
            "physical_subject_normal_ap_dv_ml": precursor["geometry"][
                "normal_rp2_ap_dv_ml"
            ],
            "subject_grid_construction": (
                "float32 renderer centre + float32(physical offset / frozen voxel size); "
                "effective_renderer_sampling_arrays for every offset"
            ),
            "mapping_direction": "subject-space query point to CCF-space sample point",
        },
        "arrays": coordinate_arrays,
        "array_receipts": _receipts(coordinate_arrays),
        "centre_plane_fit": fit,
    }
    coordinate_map["subject_coordinate_map_id"] = acquisition._payload_sha256(
        _coordinate_identity_payload(coordinate_map)
    )
    return {
        "support": support,
        "annotation_volume": domain["annotation_volume"],
        "identity": identity,
        "synthetic_animal_id": synthetic_animal_id,
        "offsets": offsets,
        "masses": masses,
        "centre_index": centre_index,
        "source_hashes": source_hashes,
        "context_reference": context_reference,
        "precursor_reference": precursor_reference,
        "coordinate_map": coordinate_map,
    }


def _coordinate_identity_payload(stage: dict[str, object]) -> dict[str, object]:
    return {
        "domain": SUBJECT_COORDINATE_MAP_V2_SCHEMA,
        "schema_version": stage["schema_version"],
        "algorithm": stage["algorithm"],
        "implementation_source_sha256": stage["implementation_source_sha256"],
        "implementation_source_sha256_canonicalization": stage[
            "implementation_source_sha256_canonicalization"
        ],
        "context_reference": stage["context_reference"],
        "precursor_reference": stage["precursor_reference"],
        "deformation_reference": stage["deformation_reference"],
        "synthetic_animal_id": stage["synthetic_animal_id"],
        "atlas_domain": stage["atlas_domain"],
        "kernel": stage["kernel"],
        "array_receipts": stage["array_receipts"],
        "centre_plane_fit_id": stage["centre_plane_fit"][
            "subject_centre_plane_fit_id"
        ],
    }


def _support_acceptance(centre_plane_brain_pixel_count: int) -> dict[str, object]:
    return {
        "rule": "at least one mapped centre-plane annotation pixel is nonzero",
        "centre_plane_brain_pixel_count": int(centre_plane_brain_pixel_count),
        "accepted": int(centre_plane_brain_pixel_count) > 0,
        "target_image_overlap_used": False,
        "redraw_attempted": False,
    }


def _support_probe_sampling_contract() -> dict[str, object]:
    return {
        "coordinate_space": "zero-based Allen AP-DV-ML voxel-centre index",
        "annotation_operator": "torch.round ties-to-even nearest label; zero outside atlas",
        "sampled_slab_level": "authenticated zero-offset centre plane only",
        "decision_rule": "count sampled annotation labels that are nonzero",
    }


def _support_probe_disclosure() -> dict[str, object]:
    return {
        "decision_inputs": [
            "mapped centre-plane Allen-index coordinates",
            "authenticated atlas annotation",
        ],
        "scalar_samples_computed": False,
        "appearance_used": False,
        "target_image_overlap_used": False,
    }


def _support_probe_identity_payload(stage: dict[str, object]) -> dict[str, object]:
    return {
        "domain": SUBJECT_CENTRE_SUPPORT_PROBE_V2_SCHEMA,
        "schema_version": stage["schema_version"],
        "algorithm": stage["algorithm"],
        "implementation_source_sha256": stage["implementation_source_sha256"],
        "implementation_source_sha256_canonicalization": stage[
            "implementation_source_sha256_canonicalization"
        ],
        "context_reference": stage["context_reference"],
        "precursor_reference": stage["precursor_reference"],
        "deformation_reference": stage["deformation_reference"],
        "lineage": stage["lineage"],
        "lineage_receipt_sha256": stage["lineage_receipt_sha256"],
        "sampling_contract": stage["sampling_contract"],
        "decision_disclosure": stage["decision_disclosure"],
        "mapped_centre_coordinate_receipt": stage[
            "mapped_centre_coordinate_receipt"
        ],
        "centre_annotation_receipt": stage["centre_annotation_receipt"],
        "support_acceptance": stage["support_acceptance"],
    }


def subject_centre_support_probe_receipt_v2(
    stage: dict[str, object],
) -> dict[str, object]:
    return {
        "subject_centre_support_probe_id": stage["subject_centre_support_probe_id"],
        "identity_payload": _support_probe_identity_payload(stage),
    }


def _make_subject_centre_support_probe_from_state(
    state: dict[str, object],
    precursor: dict[str, object],
    subject_plan: dict[str, object] | None,
) -> dict[str, object]:
    mapped_centre = np.ascontiguousarray(
        state["mapped_centre_allen"],
        dtype=np.float32,
    )
    sampled_annotation = sample_nearest_annotation_coordinate_rasters_v2(
        state["annotation_volume"].to(torch.int64),
        torch.from_numpy(mapped_centre[None]),
    )[0]
    centre_annotation = np.ascontiguousarray(
        sampled_annotation.cpu().numpy(), dtype=np.int64
    )
    count = int((centre_annotation != 0).sum())
    lineage = _lineage_reference(precursor, subject_plan)
    artifact = {
        "schema_version": SUBJECT_CENTRE_SUPPORT_PROBE_V2_SCHEMA,
        "algorithm": SUBJECT_CENTRE_SUPPORT_PROBE_V2_ALGORITHM,
        "implementation_source_sha256": state["source_hashes"],
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "context_reference": state["context_reference"],
        "precursor_reference": state["precursor_reference"],
        "deformation_reference": state["deformation_reference"],
        "lineage": lineage,
        "lineage_receipt_sha256": acquisition._payload_sha256(lineage),
        "sampling_contract": _support_probe_sampling_contract(),
        "decision_disclosure": _support_probe_disclosure(),
        "mapped_centre_coordinate_receipt": acquisition._array_receipt(
            mapped_centre
        ),
        "centre_annotation_receipt": acquisition._array_receipt(centre_annotation),
        "support_acceptance": _support_acceptance(count),
    }
    artifact["subject_centre_support_probe_id"] = acquisition._payload_sha256(
        _support_probe_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        subject_centre_support_probe_receipt_v2(artifact)
    )
    return artifact


def _make_subject_centre_support_probe_with_mapper_v2(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> tuple[dict[str, object], Callable[..., np.ndarray] | None]:
    state = _subject_centre_support_state(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    return (
        _make_subject_centre_support_probe_from_state(
            state, precursor, subject_plan
        ),
        state["subject_to_ccf_mapper"],
    )


def make_subject_centre_support_probe_v2(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Authenticate post-deformation centre support without sampling scalar appearance."""
    artifact, _ = _make_subject_centre_support_probe_with_mapper_v2(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
    )
    return artifact


def _replay_subject_centre_support_probe_with_mapper_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> dict[str, object]:
    replay, _ = _make_subject_centre_support_probe_with_mapper_v2(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    return replay


def replay_subject_centre_support_probe_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    return _replay_subject_centre_support_probe_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
    )


def _validate_support_probe_structure(artifact: dict[str, object]) -> None:
    if (
        set(artifact)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "context_reference",
            "precursor_reference",
            "deformation_reference",
            "lineage",
            "lineage_receipt_sha256",
            "sampling_contract",
            "decision_disclosure",
            "mapped_centre_coordinate_receipt",
            "centre_annotation_receipt",
            "support_acceptance",
            "subject_centre_support_probe_id",
            "receipt_sha256",
        }
        or set(artifact.get("context_reference", {}))
        != {"schema", "v2_context_sha256", "prepared_context_receipt_sha256"}
        or set(artifact.get("precursor_reference", {}))
        != {
            "precursor_contract",
            "v2_plane_realization_id",
            "centre_plane_render_id",
            "global_reference_grid_id",
            "slab_recipe_id",
            "slab_render_id",
            "v2_slab_render_receipt_sha256",
            "v2_context_sha256",
            "animal_id",
            "animal_index",
            "split",
            "plane_sample_index",
        }
        or set(artifact.get("deformation_reference", {}))
        != {"mode", "subject_deformation_plan_receipt", "synthetic_animal_id"}
        or set(artifact.get("lineage", {}))
        != {
            "split",
            "plane_sample_index",
            "animal_id",
            "animal_index",
            "specimen_id",
            "experiment_id",
            "synthetic_animal_id",
        }
        or set(artifact.get("sampling_contract", {}))
        != {
            "coordinate_space",
            "annotation_operator",
            "sampled_slab_level",
            "decision_rule",
        }
        or set(artifact.get("decision_disclosure", {}))
        != {
            "decision_inputs",
            "scalar_samples_computed",
            "appearance_used",
            "target_image_overlap_used",
        }
        or set(artifact.get("support_acceptance", {}))
        != {
            "rule",
            "centre_plane_brain_pixel_count",
            "accepted",
            "target_image_overlap_used",
            "redraw_attempted",
        }
    ):
        raise ValueError("subject centre support probe has missing or extra fields")


def _verify_subject_centre_support_probe_with_mapper_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None = None,
) -> None:
    _validate_support_probe_structure(artifact)
    replay = _replay_subject_centre_support_probe_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    if (
        artifact["schema_version"] != SUBJECT_CENTRE_SUPPORT_PROBE_V2_SCHEMA
        or artifact["algorithm"] != SUBJECT_CENTRE_SUPPORT_PROBE_V2_ALGORITHM
        or artifact["implementation_source_sha256"] != _source_hashes()
        or artifact["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or artifact["context_reference"] != _context_reference(prepared_context)
        or artifact["precursor_reference"] != _precursor_reference(precursor)
        or artifact["deformation_reference"] != _deformation_reference(subject_plan)
        or artifact["lineage"] != _lineage_reference(precursor, subject_plan)
        or artifact["lineage_receipt_sha256"]
        != acquisition._payload_sha256(artifact["lineage"])
        or artifact["sampling_contract"] != _support_probe_sampling_contract()
        or artifact["decision_disclosure"] != _support_probe_disclosure()
        or artifact["support_acceptance"]
        != _support_acceptance(
            artifact["support_acceptance"]["centre_plane_brain_pixel_count"]
        )
        or artifact["subject_centre_support_probe_id"]
        != acquisition._payload_sha256(_support_probe_identity_payload(artifact))
        or artifact["receipt_sha256"]
        != acquisition._payload_sha256(
            subject_centre_support_probe_receipt_v2(artifact)
        )
    ):
        raise ValueError("subject centre support probe source or live receipt does not match")
    if artifact != replay:
        raise ValueError("subject centre support probe deterministic replay does not match")


def verify_subject_centre_support_probe_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> None:
    _verify_subject_centre_support_probe_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
    )


def _render_identity_payload(stage: dict[str, object]) -> dict[str, object]:
    return {
        "domain": SUBJECT_SLAB_RENDER_V2_SCHEMA,
        "schema_version": stage["schema_version"],
        "algorithm": stage["algorithm"],
        "implementation_source_sha256": stage["implementation_source_sha256"],
        "implementation_source_sha256_canonicalization": stage[
            "implementation_source_sha256_canonicalization"
        ],
        "subject_coordinate_map_id": stage["subject_coordinate_map_id"],
        "context_reference": stage["context_reference"],
        "precursor_reference": stage["precursor_reference"],
        "identity_reference_path": stage["identity_reference_path"],
        "synthetic_animal_id": stage["synthetic_animal_id"],
        "support_probe_reference": stage["support_probe_reference"],
        "support_acceptance": stage["support_acceptance"],
        "sample_array_receipts": stage["sample_array_receipts"],
        "raster_array_receipts": stage["raster_array_receipts"],
    }


def subject_slab_render_receipt_v2(stage: dict[str, object]) -> dict[str, object]:
    return {
        "subject_coordinate_map_id": stage["subject_coordinate_map_id"],
        "subject_slab_render_id": stage["subject_slab_render_id"],
        "coordinate_identity_payload": _coordinate_identity_payload(
            stage["coordinate_map"]
        ),
        "render_identity_payload": _render_identity_payload(stage),
    }


def _make_subject_slab_render_with_mapper_v2(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None,
) -> dict[str, object]:
    support_state = _subject_centre_support_state(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    support_probe = _make_subject_centre_support_probe_from_state(
        support_state, precursor, subject_plan
    )
    support_acceptance = support_probe["support_acceptance"]
    if not support_acceptance["accepted"]:
        raise ValueError("subject slab mapped centre plane has no brain support")

    state = _subject_coordinate_state(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=support_state["subject_to_ccf_mapper"],
    )
    support = state["support"]
    parent = prepared_context["opaque_v1_context"]
    scalar_volume = parent["scalar_tensor"]
    annotation_volume = state["annotation_volume"]
    shape = tuple(int(value) for value in support["annotation_shape"])
    if tuple(scalar_volume.shape) != shape or tuple(annotation_volume.shape) != shape:
        raise ValueError("prepared context tensors and support geometry disagree")
    identity = bool(state["identity"])
    synthetic_animal_id = state["synthetic_animal_id"]
    masses = state["masses"]
    centre_index = int(state["centre_index"])
    source_hashes = state["source_hashes"]
    context_reference = state["context_reference"]
    precursor_reference = state["precursor_reference"]
    coordinate_map = state["coordinate_map"]
    mapped_allen = coordinate_map["arrays"][
        "mapped_allen_index_coordinates_float32"
    ]

    sampled_scalar, sampled_annotation = sample_coordinate_rasters_v2(
        scalar_volume,
        annotation_volume.to(torch.int64),
        torch.from_numpy(mapped_allen),
    )
    sample_arrays = {
        "scalar_samples_float32": np.ascontiguousarray(
            sampled_scalar.cpu().numpy(), dtype=np.float32
        ),
        "annotation_samples_int64": np.ascontiguousarray(
            sampled_annotation.cpu().numpy(), dtype=np.int64
        ),
    }
    reduced_arrays = _reduce_samples_like_precursor(
        sample_arrays["scalar_samples_float32"],
        sample_arrays["annotation_samples_int64"],
        masses,
        centre_index,
        precursor["slab_recipe"]["render_mode"],
    )
    centre_plane_brain_pixel_count = int(
        (sample_arrays["annotation_samples_int64"][centre_index] != 0).sum()
    )
    full_support_acceptance = _support_acceptance(centre_plane_brain_pixel_count)
    if not full_support_acceptance["accepted"]:
        raise ValueError("subject slab mapped centre plane has no brain support")
    if (
        support_probe["mapped_centre_coordinate_receipt"]
        != acquisition._array_receipt(mapped_allen[centre_index])
        or support_probe["centre_annotation_receipt"]
        != acquisition._array_receipt(
            sample_arrays["annotation_samples_int64"][centre_index]
        )
        or support_acceptance != full_support_acceptance
    ):
        raise ValueError("subject support probe and full slab sampling disagree")
    if identity:
        for index, offset_receipt in enumerate(precursor["offset_render_receipts"]):
            if (
                acquisition._array_receipt(
                    sample_arrays["scalar_samples_float32"][index]
                )
                != offset_receipt["raster_array_receipts"]["scalar"]
                or acquisition._array_receipt(
                    sample_arrays["annotation_samples_int64"][index]
                )
                != offset_receipt["raster_array_receipts"]["annotation"]
            ):
                raise ValueError(
                    "identity samples do not reproduce authenticated precursor offsets"
                )
        precursor_arrays = _reduced_arrays(precursor["raster"])
        if any(
            not _byte_equal(reduced_arrays[name], precursor_arrays[name])
            for name in _REDUCED_ARRAY_KEYS
        ):
            raise ValueError(
                "identity reduction is not byte-identical to authenticated precursor"
            )
        reduced_arrays = {
            name: np.array(value, copy=True, order="C")
            for name, value in precursor_arrays.items()
        }

    render = {
        "schema_version": SUBJECT_SLAB_RENDER_V2_SCHEMA,
        "algorithm": SUBJECT_SLAB_RENDER_V2_ALGORITHM,
        "implementation_source_sha256": source_hashes,
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "subject_coordinate_map_id": coordinate_map["subject_coordinate_map_id"],
        "context_reference": context_reference,
        "precursor_reference": precursor_reference,
        "identity_reference_path": identity,
        "synthetic_animal_id": synthetic_animal_id,
        "support_probe_reference": {
            "subject_centre_support_probe_id": support_probe[
                "subject_centre_support_probe_id"
            ],
            "receipt_sha256": support_probe["receipt_sha256"],
        },
        "support_acceptance": support_acceptance,
        "coordinate_map": coordinate_map,
        "sample_arrays": sample_arrays,
        "sample_array_receipts": _receipts(sample_arrays),
        "raster": _nest_reduced(reduced_arrays),
        "raster_array_receipts": _receipts(reduced_arrays),
    }
    render["subject_slab_render_id"] = acquisition._payload_sha256(
        _render_identity_payload(render)
    )
    render["receipt_sha256"] = acquisition._payload_sha256(
        subject_slab_render_receipt_v2(render)
    )
    return render


def make_subject_slab_render_v2(
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Authenticate a plan, then map and sample an immutable subject slab."""
    return _make_subject_slab_render_with_mapper_v2(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )


def _replay_subject_slab_render_with_mapper_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None,
) -> dict[str, object]:
    return _make_subject_slab_render_with_mapper_v2(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def replay_subject_slab_render_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    return _replay_subject_slab_render_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )


def _validate_structure(artifact: dict[str, object]) -> None:
    coordinate = artifact.get("coordinate_map", {})
    raster = artifact.get("raster", {})
    if (
        set(artifact)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "subject_coordinate_map_id",
            "context_reference",
            "precursor_reference",
            "identity_reference_path",
            "synthetic_animal_id",
            "support_probe_reference",
            "support_acceptance",
            "coordinate_map",
            "sample_arrays",
            "sample_array_receipts",
            "raster",
            "raster_array_receipts",
            "subject_slab_render_id",
            "receipt_sha256",
        }
        or set(coordinate)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "context_reference",
            "precursor_reference",
            "deformation_reference",
            "synthetic_animal_id",
            "atlas_domain",
            "kernel",
            "arrays",
            "array_receipts",
            "centre_plane_fit",
            "subject_coordinate_map_id",
        }
        or set(coordinate.get("context_reference", {}))
        != {"schema", "v2_context_sha256", "prepared_context_receipt_sha256"}
        or set(coordinate.get("precursor_reference", {}))
        != {
            "precursor_contract",
            "v2_plane_realization_id",
            "centre_plane_render_id",
            "global_reference_grid_id",
            "slab_recipe_id",
            "slab_render_id",
            "v2_slab_render_receipt_sha256",
            "v2_context_sha256",
            "animal_id",
            "animal_index",
            "split",
            "plane_sample_index",
        }
        or set(coordinate.get("deformation_reference", {}))
        != {"mode", "subject_deformation_plan_receipt", "synthetic_animal_id"}
        or set(coordinate.get("atlas_domain", {}))
        != {
            "axis_order",
            "origin_ap_dv_ml_um",
            "voxel_size_ap_dv_ml_um",
            "shape_ap_dv_ml",
            "voxel_face_lower_closed_ap_dv_ml_um",
            "voxel_face_upper_closed_ap_dv_ml_um",
            "voxel_centre_convention",
        }
        or set(coordinate.get("kernel", {}))
        != {
            "render_mode",
            "offsets_um",
            "integer_masses",
            "centre_index",
            "physical_subject_normal_ap_dv_ml",
            "subject_grid_construction",
            "mapping_direction",
        }
        or set(coordinate.get("arrays", {})) != _COORDINATE_ARRAY_KEYS
        or set(coordinate.get("array_receipts", {})) != _COORDINATE_ARRAY_KEYS
        or set(artifact.get("sample_arrays", {}))
        != {"scalar_samples_float32", "annotation_samples_int64"}
        or set(artifact.get("sample_array_receipts", {}))
        != {"scalar_samples_float32", "annotation_samples_int64"}
        or set(raster)
        != {
            "scalar",
            "centre_plane_annotation",
            "centre_plane_support_mask",
            "slab_brain_occupancy",
            "slab_observable_support_mask",
            "slab_modal_annotation",
            "slab_label_purity",
            "centre_label_support_weight",
            "slab_supervision_weight_or_abstention",
        }
        or set(raster.get("slab_supervision_weight_or_abstention", {}))
        != {"dense_correspondence_weight", "abstention_mask"}
        or set(artifact.get("raster_array_receipts", {})) != _REDUCED_ARRAY_KEYS
        or set(artifact.get("support_acceptance", {}))
        != {
            "rule",
            "centre_plane_brain_pixel_count",
            "accepted",
            "target_image_overlap_used",
            "redraw_attempted",
        }
        or set(artifact.get("support_probe_reference", {}))
        != {"subject_centre_support_probe_id", "receipt_sha256"}
    ):
        raise ValueError("subject slab contains missing or unauthenticated extra fields")


def _verify_subject_slab_render_with_mapper_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper: Callable[..., np.ndarray] | None,
) -> None:
    acquisition._validate_v2_context(prepared_context)
    _precursor_contract_and_receipt(precursor, prepared_context)
    _validate_structure(artifact)
    coordinate = artifact["coordinate_map"]
    source_hashes = _source_hashes()
    context_reference = _context_reference(prepared_context)
    precursor_reference = _precursor_reference(precursor)
    synthetic_animal_id = (
        None if subject_plan is None else subject_plan["synthetic_animal_id"]
    )
    support = acquisition._context_support(prepared_context)
    centre_index = coordinate["kernel"]["centre_index"]
    centre_plane_brain_pixel_count = int(
        (
            artifact["sample_arrays"]["annotation_samples_int64"][centre_index]
            != 0
        ).sum()
    )
    expected_support_acceptance = _support_acceptance(
        centre_plane_brain_pixel_count
    )
    if (
        artifact["schema_version"] != SUBJECT_SLAB_RENDER_V2_SCHEMA
        or artifact["algorithm"] != SUBJECT_SLAB_RENDER_V2_ALGORITHM
        or coordinate["schema_version"] != SUBJECT_COORDINATE_MAP_V2_SCHEMA
        or coordinate["algorithm"] != SUBJECT_COORDINATE_MAP_V2_ALGORITHM
        or artifact["implementation_source_sha256"] != source_hashes
        or coordinate["implementation_source_sha256"] != source_hashes
        or artifact["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or coordinate["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or artifact["context_reference"] != context_reference
        or coordinate["context_reference"] != context_reference
        or artifact["precursor_reference"] != precursor_reference
        or coordinate["precursor_reference"] != precursor_reference
        or coordinate["deformation_reference"]
        != _deformation_reference(subject_plan)
        or artifact["synthetic_animal_id"] != synthetic_animal_id
        or coordinate["synthetic_animal_id"] != synthetic_animal_id
        or coordinate["atlas_domain"] != _atlas_domain(support)
        or artifact["identity_reference_path"] != (subject_plan is None)
        or artifact["subject_coordinate_map_id"]
        != coordinate["subject_coordinate_map_id"]
        or artifact["support_acceptance"] != expected_support_acceptance
        or expected_support_acceptance["accepted"] is not True
    ):
        raise ValueError("subject slab schema, source, or authoritative binding does not match")
    verify_subject_centre_plane_fit_v2(
        coordinate["centre_plane_fit"],
        coordinate["arrays"][
            "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
        ][coordinate["kernel"]["centre_index"]],
    )
    if (
        coordinate["array_receipts"] != _receipts(coordinate["arrays"])
        or coordinate["subject_coordinate_map_id"]
        != acquisition._payload_sha256(_coordinate_identity_payload(coordinate))
        or artifact["sample_array_receipts"]
        != _receipts(artifact["sample_arrays"])
        or artifact["raster_array_receipts"]
        != _receipts(_reduced_arrays(artifact["raster"]))
        or artifact["subject_slab_render_id"]
        != acquisition._payload_sha256(_render_identity_payload(artifact))
        or artifact["receipt_sha256"]
        != acquisition._payload_sha256(subject_slab_render_receipt_v2(artifact))
    ):
        raise ValueError("subject slab live array receipt or identity does not match")
    replay = _replay_subject_slab_render_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    if subject_slab_render_receipt_v2(artifact) != subject_slab_render_receipt_v2(
        replay
    ):
        raise ValueError("subject slab deterministic replay receipt does not match")
    for left, right in (
        (coordinate["arrays"], replay["coordinate_map"]["arrays"]),
        (artifact["sample_arrays"], replay["sample_arrays"]),
        (_reduced_arrays(artifact["raster"]), _reduced_arrays(replay["raster"])),
    ):
        if any(not _byte_equal(left[name], right[name]) for name in left):
            raise ValueError("subject slab deterministic replay arrays do not match")


def verify_subject_slab_render_v2(
    artifact: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> None:
    _verify_subject_slab_render_with_mapper_v2(
        artifact,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )
