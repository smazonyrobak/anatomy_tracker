"""Final provenance-bound arbitrary-plane synthetic training realization."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from inspect import signature
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_observation_v2 import (
    observation_bundle_receipt_v2,
    verify_arbitrary_plane_observation_v2,
)
from training.arbitrary_plane_section_processing_v2 import (
    section_processing_plan_receipt_v2,
    section_processing_render_receipt_v2,
)
from training.arbitrary_plane_subject_slab_v2 import (
    _precursor_contract_and_receipt,
    subject_slab_render_receipt_v2,
)
from training.arbitrary_plane_support_resolution_v2 import (
    subject_support_resolution_receipt_v2,
    verify_accepted_subject_slab_matches_support_resolution_v2,
    verify_subject_support_resolution_v2,
)


SYNTHETIC_REALIZATION_V2_SCHEMA = "anatomy-tracker.synthetic-realization/v2"
SYNTHETIC_REALIZATION_V2_ALGORITHM = (
    "verified-observation-uniform-mode-crop-quicknii-reflection-and-exact-residual/v2"
)
SYNTHETIC_REALIZATION_V2_RNG_DOMAIN = "anatomy-tracker.synthetic-realization-rng/v2"
TRAINABLE_INPUT_MODES = (
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
)
MODEL_INPUT_CHANNEL_NAMES = (
    "grayscale_image",
    "selected_input_mask",
    "brush_availability",
)
_OBSERVATION_TARGET_KEYS = (
    "source_label_ground_truth_crop_int64",
    "source_tissue_ground_truth_mask",
    "source_correspondence_domain_mask",
    "source_dense_correspondence_weight_float32",
    "source_dense_correspondence_abstention_mask",
    "processed_mapped_ccf_physical_coordinates_crop_float64",
    "processed_bilinear_domain_valid_mask",
    "processed_nearest_domain_valid_mask",
    "processed_dense_coordinate_valid_mask",
    "physical_loss_mask",
    "occlusion_mask",
    "appearance_artifact_mask",
    "damage_union_mask",
    "observable_footprint_mask",
    "observation_invalid_mask",
    "outside_correspondence_domain_mask",
    "valid_correspondence_mask",
    "valid_correspondence_weight_float32",
)
_TARGET_ARRAY_KEYS = set(_OBSERVATION_TARGET_KEYS) | {
    "selected_brush_mask_error_mask"
}
_FACTOR_ARRAY_KEYS = {
    "section_pullback_parent_index_yx_float64",
    "nominal_physical_map_ap_dv_ml_um_float64",
    "nominal_at_section_pullback_ap_dv_ml_um_float64",
    "animal_residual_at_section_pullback_ap_dv_ml_um_float64",
    "section_plane_displacement_ap_dv_ml_um_float64",
    "composed_coordinate_residual_ap_dv_ml_um_float64",
}
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_realization_v2.py",
    "arbitrary_plane_support_resolution_v2.py",
    "arbitrary_plane_observation_v2.py",
    "arbitrary_plane_section_processing_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_section_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _root_seed_uint64(root_seed: int | str) -> int:
    if isinstance(root_seed, str):
        if (
            len(root_seed) != 18
            or not root_seed.startswith("0x")
            or any(character not in "0123456789abcdef" for character in root_seed[2:])
        ):
            raise ValueError("root_seed must be uint64 or 0x plus 16 lowercase hex digits")
        root_seed = int(root_seed[2:], 16)
    elif isinstance(root_seed, (bool, np.bool_)) or not isinstance(
        root_seed, (int, np.integer)
    ):
        raise TypeError("root_seed must be an integer or canonical uint64 hex string")
    root_seed = int(root_seed)
    if root_seed < 0 or root_seed >= 2**64:
        raise ValueError("root_seed must be uint64")
    return root_seed


def _schedule_uint64(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"{name} must fit uint64")
    return result


def derive_synthetic_realization_seed_v2(
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    section_index: int,
    observation_index: int,
    realization_index: int,
    field: str,
) -> int:
    """Derive a final-stage seed from numeric lineage; animal labels are absent."""
    coordinates = tuple(
        _schedule_uint64(value, name)
        for value, name in zip(
            (
                split_index,
                animal_index,
                section_index,
                observation_index,
                realization_index,
            ),
            (
                "split_index",
                "animal_index",
                "section_index",
                "observation_index",
                "realization_index",
            ),
        )
    )
    if (
        not isinstance(split, str)
        or not split
        or not field
    ):
        raise ValueError("synthetic-realization RNG coordinates are invalid")
    components = (
        SYNTHETIC_REALIZATION_V2_RNG_DOMAIN,
        SYNTHETIC_REALIZATION_V2_SCHEMA,
        f"0x{_root_seed_uint64(root_seed):016x}",
        split,
        *(str(value) for value in coordinates),
        str(field),
    )
    encoded = b"".join(
        len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8")
        for value in components
    )
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"AP-REAL-V2").digest(),
        "big",
    )


def _rng_receipt(provenance: dict[str, object], realization_index: int, field: str):
    seed = derive_synthetic_realization_seed_v2(
        provenance["root_seed_uint64"],
        provenance["split"],
        provenance["split_index"],
        provenance["animal_index"],
        provenance["section_index"],
        provenance["observation_index"],
        realization_index,
        field,
    )
    return seed, {
        "split": provenance["split"],
        "split_index": int(provenance["split_index"]),
        "animal_index": int(provenance["animal_index"]),
        "section_index": int(provenance["section_index"]),
        "observation_index": int(provenance["observation_index"]),
        "realization_index": int(realization_index),
        "field": field,
        "seed_uint64": f"0x{seed:016x}",
        "generator": "numpy.random.PCG64DXSM",
        "animal_label_excluded": True,
    }


def sample_synthetic_realization_choice_v2(
    observation_bundle: dict[str, object], realization_index: int
) -> dict[str, object]:
    provenance = observation_bundle["provenance"]
    realization_index = _schedule_uint64(realization_index, "realization_index")
    receipts = {}
    draws = {}
    for field, high in (
        ("trainable-input-mode", len(TRAINABLE_INPUT_MODES)),
        ("horizontal-reflection", 2),
        ("vertical-reflection", 2),
    ):
        seed, receipt = _rng_receipt(provenance, realization_index, field)
        receipts[field] = receipt
        draws[field] = int(np.random.Generator(np.random.PCG64DXSM(seed)).integers(high))
    return {
        "selected_mode_index": draws["trainable-input-mode"],
        "selected_mode": TRAINABLE_INPUT_MODES[draws["trainable-input-mode"]],
        "horizontal_reflection": bool(draws["horizontal-reflection"]),
        "vertical_reflection": bool(draws["vertical-reflection"]),
        "rng_sources": receipts,
    }


def _flip(array: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    result = np.asarray(array)
    if horizontal:
        result = result[:, ::-1, ...]
    if vertical:
        result = result[::-1, :, ...]
    return np.ascontiguousarray(result)


def _ouv_matrix(flat_ouv: np.ndarray) -> np.ndarray:
    ouv = np.asarray(flat_ouv, dtype=np.float64)
    if ouv.shape == (9,):
        ouv = ouv.reshape(3, 3)
    if ouv.shape != (3, 3) or not np.isfinite(ouv).all():
        raise ValueError("full-raster physical O/U/V is invalid")
    return np.ascontiguousarray(ouv)


def _crop_and_reflect_ouv(
    parent_ouv: np.ndarray,
    parent_shape_h_w: tuple[int, int],
    top_left_y_x: tuple[int, int],
    output_shape_h_w: tuple[int, int],
    horizontal: bool,
    vertical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    parent_height, parent_width = (int(value) for value in parent_shape_h_w)
    top, left = (int(value) for value in top_left_y_x)
    height, width = (int(value) for value in output_shape_h_w)
    if (
        min(parent_height, parent_width, height, width) < 2
        or top < 0
        or left < 0
        or top + height > parent_height
        or left + width > parent_width
    ):
        raise ValueError("final crop window is invalid")
    origin, edge_u, edge_v = _ouv_matrix(parent_ouv)
    cropped = np.stack(
        (
            origin + (left / parent_width) * edge_u + (top / parent_height) * edge_v,
            (width / parent_width) * edge_u,
            (height / parent_height) * edge_v,
        )
    )
    reflected = cropped.copy()
    if horizontal:
        reflected[0] += ((width - 1) / width) * reflected[1]
        reflected[1] *= -1.0
    if vertical:
        reflected[0] += ((height - 1) / height) * reflected[2]
        reflected[2] *= -1.0
    return np.ascontiguousarray(cropped), np.ascontiguousarray(reflected)


def _quicknii_map(ouv: np.ndarray, shape_h_w: tuple[int, int]) -> np.ndarray:
    height, width = (int(value) for value in shape_h_w)
    y, x = np.indices((height, width), dtype=np.float64)
    origin, edge_u, edge_v = _ouv_matrix(ouv)
    return np.ascontiguousarray(
        origin[None, None]
        + (x / width)[..., None] * edge_u
        + (y / height)[..., None] * edge_v
    )


def _parent_plane_at_indices(
    parent_ouv: np.ndarray,
    parent_shape_h_w: tuple[int, int],
    index_yx: np.ndarray,
) -> np.ndarray:
    height, width = (int(value) for value in parent_shape_h_w)
    indices = np.asarray(index_yx, dtype=np.float64)
    origin, edge_u, edge_v = _ouv_matrix(parent_ouv)
    return np.ascontiguousarray(
        origin
        + (indices[..., 1] / width)[..., None] * edge_u
        + (indices[..., 0] / height)[..., None] * edge_v
    )


def _array_receipts(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {name: acquisition._array_receipt(value) for name, value in arrays.items()}


def _count_synthetic_realization_ids(value: object) -> int:
    if isinstance(value, Mapping):
        return int("synthetic_realization_id" in value) + sum(
            _count_synthetic_realization_ids(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_synthetic_realization_ids(item) for item in value)
    return 0


def _upstream_reference(
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
) -> dict[str, object]:
    coordinate = subject_slab_render["coordinate_map"]
    resolution = support_resolution["resolution"]
    _, precursor_receipt = _precursor_contract_and_receipt(precursor)
    plan_receipt = acquisition._json_value(
        section_processing_plan_receipt_v2(section_processing_plan)
    )
    plan_receipt_payload = {
        key: value for key, value in plan_receipt.items() if key != "receipt_sha256"
    }
    if (
        "receipt_sha256" in plan_receipt
        and plan_receipt["receipt_sha256"]
        != acquisition._payload_sha256(plan_receipt_payload)
    ):
        raise ValueError("verified section plan live receipt changed")
    receipt_payloads = {
        "prepared_context": acquisition._json_value(prepared_context["receipt"]),
        "support_resolution": acquisition._json_value(
            subject_support_resolution_receipt_v2(resolution)
        ),
        "precursor": acquisition._json_value(precursor_receipt),
        "subject_slab": acquisition._json_value(
            subject_slab_render_receipt_v2(subject_slab_render)
        ),
        "section_processing_plan": plan_receipt_payload,
        "section_processing_render": acquisition._json_value(
            section_processing_render_receipt_v2(processed_render)
        ),
        "observation_bundle": acquisition._json_value(
            observation_bundle_receipt_v2(observation_bundle)
        ),
    }
    receipt_bindings = {
        name: {
            "receipt_payload": payload,
            "receipt_sha256": acquisition._payload_sha256(payload),
        }
        for name, payload in receipt_payloads.items()
    }
    stored_receipts = {
        "support_resolution": resolution["receipt_sha256"],
        "precursor": precursor["receipt_sha256"],
        "subject_slab": subject_slab_render["receipt_sha256"],
        "section_processing_plan": section_processing_plan["receipt_sha256"],
        "section_processing_render": processed_render["receipt_sha256"],
        "observation_bundle": observation_bundle["receipt_sha256"],
    }
    if any(
        receipt_bindings[name]["receipt_sha256"] != stored
        for name, stored in stored_receipts.items()
    ):
        raise ValueError("verified upstream live receipt changed")
    return {
        "v2_context_sha256": prepared_context["v2_context_sha256"],
        "support_resolution_plan_id": resolution["support_resolution_plan_id"],
        "subject_support_resolution_id": resolution[
            "subject_support_resolution_id"
        ],
        "support_resolution_accepted_attempt_index": resolution[
            "accepted_attempt_index"
        ],
        "accepted_support_precursor_reference": acquisition._json_value(
            resolution["accepted_precursor_reference"]
        ),
        "accepted_support_probe_reference": acquisition._json_value(
            resolution["accepted_probe_reference"]
        ),
        "precursor_slab_render_id": precursor["slab_render_id"],
        "subject_slab_render_id": subject_slab_render["subject_slab_render_id"],
        "subject_coordinate_map_id": coordinate["subject_coordinate_map_id"],
        "subject_deformation_reference": acquisition._json_value(
            coordinate["deformation_reference"]
        ),
        "section_processing_plan_id": section_processing_plan[
            "section_processing_plan_id"
        ],
        "section_processing_realization_id": section_processing_plan[
            "section_processing_realization_id"
        ],
        "section_processing_render_id": processed_render[
            "section_processing_render_id"
        ],
        "observation_bundle_id": observation_bundle["observation_bundle_id"],
        "observation_receipt_sha256": observation_bundle["receipt_sha256"],
        "acquired_observation_id": observation_bundle["acquired_observation_id"],
        "crop_window_id": observation_bundle["crop_window_id"],
        "live_receipt_bindings": receipt_bindings,
    }


def _verify_upstream(
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
    subject_plan: dict[str, object] | None,
) -> None:
    provenance = observation_bundle["provenance"]
    resolution = support_resolution["resolution"]
    config = resolution["configuration"]
    lineage = resolution["lineage"]
    verify_subject_support_resolution_v2(
        support_resolution,
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=config["master_root_seed_uint64"],
        split=lineage["split"],
        split_index=config["split_index"],
        animal_index=config["animal_index"],
        animal_id=lineage["animal_id"],
        section_index=config["section_index"],
        plane_stratum=config["plane_stratum"],
        nominal_cut_thickness_um=config["nominal_cut_thickness_um"],
        specimen_id=lineage["specimen_id"],
        experiment_id=lineage["experiment_id"],
        axial_step_um_max=config["axial_step_um_max"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        max_attempts=config["max_attempts"],
    )
    verify_accepted_subject_slab_matches_support_resolution_v2(
        support_resolution, subject_slab_render
    )
    accepted_precursor = resolution["accepted_precursor_reference"]
    if (
        resolution["status"] != "accepted"
        or lineage["split"] != provenance["split"]
        or config["split_index"] != provenance["split_index"]
        or config["animal_index"] != provenance["animal_index"]
        or config["section_index"] != provenance["section_index"]
        or acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(lineage["animal_id"])}
        )
        != acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(provenance["animal_id"])}
        )
        or any(
            name in provenance
            and acquisition._json_value(lineage[name])
            != acquisition._json_value(provenance[name])
            for name in ("specimen_id", "experiment_id")
        )
    ):
        raise ValueError("support-resolution and observation scheduling lineage differ")
    if (
        resolution["status"] != "accepted"
        or precursor["slab_render_id"] != accepted_precursor["slab_render_id"]
        or precursor["receipt_sha256"] != accepted_precursor["receipt_sha256"]
    ):
        raise ValueError("precursor is not the accepted support-resolution attempt")
    verify_arbitrary_plane_observation_v2(
        observation_bundle,
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        root_seed=provenance["root_seed_uint64"],
        split=provenance["split"],
        split_index=provenance["split_index"],
        animal_index=provenance["animal_index"],
        animal_id=provenance["animal_id"],
        section_index=provenance["section_index"],
        observation_index=provenance["observation_index"],
        modality=observation_bundle["modality"],
    )
    if any(
        _count_synthetic_realization_ids(value)
        for value in (
            prepared_context,
            support_resolution,
            precursor,
            subject_slab_render,
            section_processing_plan,
            processed_render,
            observation_bundle,
            subject_plan,
        )
    ):
        raise ValueError("an upstream stage issued a premature synthetic_realization_id")


def _identity_payload(realization: dict[str, object]) -> dict[str, object]:
    return acquisition._json_value({
        "domain": SYNTHETIC_REALIZATION_V2_SCHEMA,
        "schema_version": realization["schema_version"],
        "algorithm": realization["algorithm"],
        "implementation_source_sha256": realization["implementation_source_sha256"],
        "implementation_source_sha256_canonicalization": realization[
            "implementation_source_sha256_canonicalization"
        ],
        "runtime_dependencies": realization["runtime_dependencies"],
        "asset_dependencies": realization["asset_dependencies"],
        "upstream_reference": realization["upstream_reference"],
        "provenance": realization["provenance"],
        "rng_sources": realization["rng_sources"],
        "mode_selection": realization["mode_selection"],
        "paired_mode_sensitivity_reference": realization[
            "paired_mode_sensitivity_reference"
        ],
        "frame_transform": {
            key: value
            for key, value in realization["frame_transform"].items()
            if key != "arrays"
        },
        "model_input": {
            key: value
            for key, value in realization["model_input"].items()
            if key != "channels_float32"
        },
        "target_policy": realization["target_policy"],
        "target_array_receipts": realization["target_array_receipts"],
        "factor_truth": {
            key: value
            for key, value in realization["factor_truth"].items()
            if key != "arrays"
        },
        "training_row_id": realization["training_row_id"],
    })


def synthetic_realization_receipt_v2(realization: dict[str, object]) -> dict[str, object]:
    return {
        "synthetic_realization_id": realization["synthetic_realization_id"],
        "identity_payload": _identity_payload(realization),
    }


def _build_realization(
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
    realization_index: int,
) -> dict[str, object]:
    choice = sample_synthetic_realization_choice_v2(
        observation_bundle, realization_index
    )
    mode = choice["selected_mode"]
    horizontal = choice["horizontal_reflection"]
    vertical = choice["vertical_reflection"]
    descendant = observation_bundle["descendants"][mode]
    raw_descendant = observation_bundle["descendants"]["raw"]
    if (
        not descendant["trainable"]
        or raw_descendant["trainable"]
        or mode not in TRAINABLE_INPUT_MODES
    ):
        raise ValueError("observation descendant trainability policy changed")

    crop = observation_bundle["crop_window"]
    parent_shape = tuple(int(value) for value in crop["parent_shape_h_w"])
    top_left = tuple(int(value) for value in crop["top_left_y_x"])
    output_shape = tuple(int(value) for value in crop["output_shape_h_w"])
    fit = subject_slab_render["coordinate_map"]["centre_plane_fit"]
    parent_ouv = _ouv_matrix(
        fit["arrays"]["physical_ouv_ap_dv_ml_um_float64"]
    )
    if tuple(fit["output_shape_h_w"]) != parent_shape:
        raise ValueError("full-raster plane fit and observation crop shapes differ")
    cropped_ouv, model_ouv = _crop_and_reflect_ouv(
        parent_ouv,
        parent_shape,
        top_left,
        output_shape,
        horizontal,
        vertical,
    )
    frame_arrays = {
        "full_raster_best_fit_physical_ouv_ap_dv_ml_um_float64": parent_ouv,
        "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64": cropped_ouv,
        "model_raster_physical_ouv_ap_dv_ml_um_float64": model_ouv,
    }
    frame = {
        "quicknii_pixel_contract": "O + (x/W) U + (y/H) V; no half-pixel term",
        "crop_is_upstream_observation_crop": True,
        "crop_window_id": observation_bundle["crop_window_id"],
        "parent_shape_h_w": list(parent_shape),
        "top_left_y_x": list(top_left),
        "output_shape_h_w": list(output_shape),
        "horizontal_reflection": horizontal,
        "vertical_reflection": vertical,
        "reflection_order": ["horizontal", "vertical"],
        "reflection_is_raster_reparameterization_not_physical_mirror": True,
        "crop_formula": (
            "Oc=O+(left/W)U+(top/H)V; Uc=(w/W)U; Vc=(h/H)V"
        ),
        "horizontal_reflection_formula": "O'=O+((w-1)/w)U; U'=-U",
        "vertical_reflection_formula": "O'=O+((h-1)/h)V; V'=-V",
        "parent_subject_centre_plane_fit_id": fit["subject_centre_plane_fit_id"],
        "arrays": frame_arrays,
        "array_receipts": _array_receipts(frame_arrays),
    }
    frame["frame_transform_id"] = acquisition._payload_sha256(
        {key: value for key, value in frame.items() if key != "arrays"}
    )

    image = _flip(
        descendant["arrays"]["model_input_image_float32"], horizontal, vertical
    ).astype(np.float32, copy=False)
    selected_mask = _flip(
        descendant["arrays"]["selected_input_mask"], horizontal, vertical
    ).astype(bool, copy=False)
    available = bool(descendant["brush_available"])
    availability = np.full(output_shape, np.float32(available), dtype=np.float32)
    channels = np.ascontiguousarray(
        np.stack((image, selected_mask.astype(np.float32), availability)),
        dtype=np.float32,
    )
    model_input = {
        "channel_names": list(MODEL_INPUT_CHANNEL_NAMES),
        "channels_float32": channels,
        "channels_array_receipt": acquisition._array_receipt(channels),
        "spatial_shape_h_w": list(output_shape),
        "strict_allowlist": list(MODEL_INPUT_CHANNEL_NAMES),
    }

    target_arrays = {
        name: _flip(observation_bundle["arrays"][name], horizontal, vertical)
        for name in _OBSERVATION_TARGET_KEYS
    }
    target_arrays["selected_brush_mask_error_mask"] = _flip(
        descendant["arrays"]["brush_mask_error_mask"], horizontal, vertical
    ).astype(bool, copy=False)
    target_arrays = {
        name: np.ascontiguousarray(value) for name, value in target_arrays.items()
    }

    top, left = top_left
    height, width = output_shape
    parent_source_index = np.asarray(
        processed_render["state"]["source_index_yx"], dtype=np.float64
    )
    if parent_source_index.shape != parent_shape + (2,):
        raise ValueError("section-processing pullback shape changed")
    section_source_index = _flip(
        parent_source_index[top : top + height, left : left + width],
        horizontal,
        vertical,
    )
    nominal = _quicknii_map(model_ouv, output_shape)
    nominal_at_pullback = _parent_plane_at_indices(
        parent_ouv, parent_shape, section_source_index
    )
    exact = np.asarray(
        target_arrays[
            "processed_mapped_ccf_physical_coordinates_crop_float64"
        ],
        dtype=np.float64,
    )
    animal_residual = np.ascontiguousarray(exact - nominal_at_pullback)
    section_displacement = np.ascontiguousarray(nominal_at_pullback - nominal)
    composed_residual = np.ascontiguousarray(exact - nominal)
    factor_arrays = {
        "section_pullback_parent_index_yx_float64": np.ascontiguousarray(
            section_source_index, dtype=np.float64
        ),
        "nominal_physical_map_ap_dv_ml_um_float64": nominal,
        "nominal_at_section_pullback_ap_dv_ml_um_float64": nominal_at_pullback,
        "animal_residual_at_section_pullback_ap_dv_ml_um_float64": animal_residual,
        "section_plane_displacement_ap_dv_ml_um_float64": section_displacement,
        "composed_coordinate_residual_ap_dv_ml_um_float64": composed_residual,
    }
    finite = np.isfinite(exact).all(axis=-1)
    reconstruction_error = (
        float(
            np.max(
                np.abs(
                    animal_residual[finite]
                    + section_displacement[finite]
                    - composed_residual[finite]
                )
            )
        )
        if finite.any()
        else 0.0
    )
    factor_truth = {
        "arrays": factor_arrays,
        "array_receipts": _array_receipts(factor_arrays),
        "animal_factor": (
            "exact mapped CCF coordinate minus the full-raster fitted plane evaluated "
            "at the section pullback coordinate"
        ),
        "section_factor": (
            "full-raster fitted plane at section pullback minus the analytically "
            "crop/reflection-reparameterized nominal plane"
        ),
        "composed_residual": "exact mapped CCF coordinate minus nominal model-raster plane",
        "factor_sum_reconstruction_max_abs_um": reconstruction_error,
        "normal_component_policy": (
            "preserved in exact 3-D residual; never silently folded into plane pose; "
            "a 2-D head supervises representable tangential/section components only"
        ),
        "subject_deformation_reference": acquisition._json_value(
            subject_slab_render["coordinate_map"]["deformation_reference"]
        ),
        "section_processing_realization_id": section_processing_plan[
            "section_processing_realization_id"
        ],
    }

    provenance = {
        **acquisition._json_value(observation_bundle["provenance"]),
        "realization_index": int(realization_index),
        "rng_identity_policy": (
            "root seed plus split and numeric split/animal/section/observation/realization "
            "coordinates only; animal label strings and artifact IDs are excluded"
        ),
    }
    mode_selection = {
        "eligible_modes": list(TRAINABLE_INPUT_MODES),
        "uniform_probability_numerator": 1,
        "uniform_probability_denominator": len(TRAINABLE_INPUT_MODES),
        "selected_mode_index": choice["selected_mode_index"],
        "selected_mode": mode,
        "selected_descendant_id": descendant["descendant_id"],
        "selected_brush_available": available,
        "raw_mode_trainable": False,
        "raw_descendant_id": raw_descendant["descendant_id"],
        "raw_exclusion_policy": (
            "raw is a nontrainable audit mirror of smart-brush-absent and never emits a row"
        ),
        "emitted_training_row_count": 1,
    }
    paired_modes = {}
    for paired_mode in TRAINABLE_INPUT_MODES:
        paired_descendant = observation_bundle["descendants"][paired_mode]
        paired_image = _flip(
            paired_descendant["arrays"]["model_input_image_float32"],
            horizontal,
            vertical,
        ).astype(np.float32, copy=False)
        paired_mask = _flip(
            paired_descendant["arrays"]["selected_input_mask"],
            horizontal,
            vertical,
        ).astype(bool, copy=False)
        paired_modes[paired_mode] = {
            "descendant_id": paired_descendant["descendant_id"],
            "brush_available": bool(paired_descendant["brush_available"]),
            "reflected_model_input_image_receipt": acquisition._array_receipt(
                paired_image
            ),
            "reflected_selected_input_mask_receipt": acquisition._array_receipt(
                paired_mask
            ),
        }
    paired_mode_sensitivity_reference = {
        "acquired_observation_id": observation_bundle["acquired_observation_id"],
        "frame_transform_id": frame["frame_transform_id"],
        "horizontal_reflection": horizontal,
        "vertical_reflection": vertical,
        "trainable_modes": paired_modes,
        "raw_exclusion_reference": {
            "descendant_id": raw_descendant["descendant_id"],
            "trainable": False,
            "equivalent_trainable_mode": "smart-brush-absent",
        },
        "emitted_training_row_count": 1,
    }
    target_policy = {
        "model_input_channel_names": list(MODEL_INPUT_CHANNEL_NAMES),
        "forbidden_model_input_families": [
            "brush_mask_error",
            "tissue_or_damage_truth",
            "correspondence_or_validity",
            "weights_or_abstention",
            "atlas_labels",
            "physical_coordinates_or_residuals",
        ],
        "selected_input_mask_is_side_information_only": True,
        "selected_input_mask_may_gate_any_loss": False,
        "registration_loss_gate": "valid_correspondence_mask",
        "brush_error_role": "audit target only; never a model input or loss gate",
        "pose_target_policy": (
            "full-raster best-fit O/U/V is analytically reparameterized by the stored crop "
            "and raster reflections; it is never refit on cropped, visible, or valid pixels"
        ),
    }
    upstream = _upstream_reference(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
    )
    training_row_id = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.synthetic-training-row/v2",
            "acquired_observation_id": observation_bundle["acquired_observation_id"],
            "selected_descendant_id": descendant["descendant_id"],
            "frame_transform_id": frame["frame_transform_id"],
            "realization_index": int(realization_index),
        }
    )
    realization = {
        "schema_version": SYNTHETIC_REALIZATION_V2_SCHEMA,
        "algorithm": SYNTHETIC_REALIZATION_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "reflection_operator": "exact array reversal; no interpolation",
        },
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "upstream_reference": upstream,
        "provenance": provenance,
        "rng_sources": choice["rng_sources"],
        "mode_selection": mode_selection,
        "paired_mode_sensitivity_reference": paired_mode_sensitivity_reference,
        "frame_transform": frame,
        "model_input": model_input,
        "targets": target_arrays,
        "target_array_receipts": _array_receipts(target_arrays),
        "target_policy": target_policy,
        "factor_truth": factor_truth,
        "training_row_id": training_row_id,
    }
    realization["synthetic_realization_id"] = acquisition._payload_sha256(
        _identity_payload(realization)
    )
    realization["receipt_sha256"] = acquisition._payload_sha256(
        synthetic_realization_receipt_v2(realization)
    )
    return acquisition._freeze_value(realization)


def make_arbitrary_plane_realization_v2(
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    realization_index: int,
) -> dict[str, object]:
    """Verify every upstream stage and emit the only final synthetic realization ID."""
    _verify_upstream(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
        subject_plan,
    )
    return _build_realization(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
        _schedule_uint64(realization_index, "realization_index"),
    )


def replay_arbitrary_plane_realization_v2(
    realization: dict[str, object],
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
) -> dict[str, object]:
    return make_arbitrary_plane_realization_v2(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
        subject_plan=subject_plan,
        realization_index=realization["provenance"]["realization_index"],
    )


def _byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _strict_structure(realization: dict[str, object]) -> None:
    if (
        set(realization)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "runtime_dependencies",
            "asset_dependencies",
            "upstream_reference",
            "provenance",
            "rng_sources",
            "mode_selection",
            "paired_mode_sensitivity_reference",
            "frame_transform",
            "model_input",
            "targets",
            "target_array_receipts",
            "target_policy",
            "factor_truth",
            "training_row_id",
            "synthetic_realization_id",
            "receipt_sha256",
        }
        or set(realization.get("model_input", {}))
        != {
            "channel_names",
            "channels_float32",
            "channels_array_receipt",
            "spatial_shape_h_w",
            "strict_allowlist",
        }
        or set(realization.get("paired_mode_sensitivity_reference", {}))
        != {
            "acquired_observation_id",
            "frame_transform_id",
            "horizontal_reflection",
            "vertical_reflection",
            "trainable_modes",
            "raw_exclusion_reference",
            "emitted_training_row_count",
        }
        or set(
            realization.get("paired_mode_sensitivity_reference", {}).get(
                "trainable_modes", {}
            )
        )
        != set(TRAINABLE_INPUT_MODES)
        or set(realization.get("targets", {})) != _TARGET_ARRAY_KEYS
        or set(realization.get("target_array_receipts", {})) != _TARGET_ARRAY_KEYS
        or set(realization.get("factor_truth", {}).get("arrays", {}))
        != _FACTOR_ARRAY_KEYS
        or set(realization.get("factor_truth", {}).get("array_receipts", {}))
        != _FACTOR_ARRAY_KEYS
        or set(realization.get("frame_transform", {}).get("arrays", {}))
        != {
            "full_raster_best_fit_physical_ouv_ap_dv_ml_um_float64",
            "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64",
            "model_raster_physical_ouv_ap_dv_ml_um_float64",
        }
        or _count_synthetic_realization_ids(realization) != 1
    ):
        raise ValueError("synthetic realization has missing or extra fields")


def verify_arbitrary_plane_realization_v2(
    realization: dict[str, object],
    prepared_context: dict[str, object],
    support_resolution: dict[str, object],
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    processed_render: dict[str, object],
    observation_bundle: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
) -> None:
    """Verify upstream lineage, strict input separation, receipts, algebra, and replay."""
    _verify_upstream(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
        subject_plan,
    )
    _strict_structure(realization)
    expected = _build_realization(
        prepared_context,
        support_resolution,
        precursor,
        subject_slab_render,
        section_processing_plan,
        processed_render,
        observation_bundle,
        _schedule_uint64(
            realization["provenance"]["realization_index"], "realization_index"
        ),
    )
    channels = np.asarray(realization["model_input"]["channels_float32"])
    height, width = realization["frame_transform"]["output_shape_h_w"]
    mode = realization["mode_selection"]["selected_mode"]
    available = mode != "smart-brush-absent"
    if (
        realization["schema_version"] != SYNTHETIC_REALIZATION_V2_SCHEMA
        or realization["algorithm"] != SYNTHETIC_REALIZATION_V2_ALGORITHM
        or acquisition._json_value(realization["implementation_source_sha256"])
        != _source_hashes()
        or realization["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or tuple(realization["model_input"]["channel_names"])
        != MODEL_INPUT_CHANNEL_NAMES
        or tuple(realization["model_input"]["strict_allowlist"])
        != MODEL_INPUT_CHANNEL_NAMES
        or channels.shape != (3, height, width)
        or channels.dtype != np.float32
        or not np.array_equal(channels[1] > 0.5, channels[1].astype(bool))
        or not np.all(channels[2] == np.float32(available))
        or realization["target_policy"]["selected_input_mask_may_gate_any_loss"]
        is not False
        or tuple(realization["mode_selection"]["eligible_modes"])
        != TRAINABLE_INPUT_MODES
        or mode not in TRAINABLE_INPUT_MODES
        or realization["mode_selection"]["raw_mode_trainable"] is not False
        or realization["mode_selection"]["emitted_training_row_count"] != 1
        or realization["paired_mode_sensitivity_reference"][
            "emitted_training_row_count"
        ]
        != 1
        or realization["paired_mode_sensitivity_reference"][
            "raw_exclusion_reference"
        ]
        != {
            "descendant_id": observation_bundle["descendants"]["raw"][
                "descendant_id"
            ],
            "trainable": False,
            "equivalent_trainable_mode": "smart-brush-absent",
        }
        or acquisition._json_value(
            realization["model_input"]["channels_array_receipt"]
        )
        != acquisition._json_value(acquisition._array_receipt(channels))
        or acquisition._json_value(realization["target_array_receipts"])
        != acquisition._json_value(_array_receipts(realization["targets"]))
        or acquisition._json_value(realization["factor_truth"]["array_receipts"])
        != acquisition._json_value(
            _array_receipts(realization["factor_truth"]["arrays"])
        )
        or acquisition._json_value(realization["frame_transform"]["array_receipts"])
        != acquisition._json_value(
            _array_receipts(realization["frame_transform"]["arrays"])
        )
        or realization["synthetic_realization_id"]
        != acquisition._payload_sha256(_identity_payload(realization))
        or realization["receipt_sha256"]
        != acquisition._payload_sha256(synthetic_realization_receipt_v2(realization))
        or acquisition._json_value(synthetic_realization_receipt_v2(realization))
        != acquisition._json_value(synthetic_realization_receipt_v2(expected))
    ):
        raise ValueError("synthetic realization metadata, input allowlist, or receipt changed")
    for group, keys in (
        ("targets", _TARGET_ARRAY_KEYS),
        ("factor_truth", _FACTOR_ARRAY_KEYS),
    ):
        actual_arrays = (
            realization[group]
            if group == "targets"
            else realization[group]["arrays"]
        )
        expected_arrays = (
            expected[group] if group == "targets" else expected[group]["arrays"]
        )
        if any(
            not _byte_equal(actual_arrays[name], expected_arrays[name]) for name in keys
        ):
            raise ValueError("synthetic realization target or factor arrays changed")
    if not _byte_equal(channels, expected["model_input"]["channels_float32"]):
        raise ValueError("synthetic realization model input changed")
    if any(
        not _byte_equal(
            realization["frame_transform"]["arrays"][name],
            expected["frame_transform"]["arrays"][name],
        )
        for name in realization["frame_transform"]["arrays"]
    ):
        raise ValueError("synthetic realization O/U/V frame changed")
    exact = realization["targets"][
        "processed_mapped_ccf_physical_coordinates_crop_float64"
    ]
    factor = realization["factor_truth"]["arrays"]
    nominal = factor["nominal_physical_map_ap_dv_ml_um_float64"]
    composed = factor["composed_coordinate_residual_ap_dv_ml_um_float64"]
    finite = np.isfinite(exact).all(axis=-1)
    if finite.any() and (
        not np.allclose(nominal[finite] + composed[finite], exact[finite], rtol=0.0, atol=1e-10)
        or not np.allclose(
            factor["animal_residual_at_section_pullback_ap_dv_ml_um_float64"][finite]
            + factor["section_plane_displacement_ap_dv_ml_um_float64"][finite],
            composed[finite],
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError("synthetic realization exact coordinate reconstruction failed")


if "animal_id" in signature(derive_synthetic_realization_seed_v2).parameters:
    raise RuntimeError("synthetic-realization RNG must never accept animal_id")
