"""Random-only modality, damage, crop, and optional smart-brush observations."""

from __future__ import annotations

import hashlib
from inspect import signature
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_section_processing_v2 import (
    section_processing_render_receipt_v2,
    verify_section_processing_render_v2,
)


OBSERVATION_V2_SCHEMA = "anatomy-tracker.arbitrary-plane-observation/v2"
OBSERVATION_V2_ALGORITHM = (
    "modality-forward-model-damage-window-and-paired-smart-brush/v2"
)
OBSERVATION_DESCENDANT_V2_SCHEMA = "anatomy-tracker.observation-descendant/v2"
MODALITIES = ("brightfield-nissl-like", "fluorescence")
DESCENDANT_MODES = (
    "raw",
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
)
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_observation_v2.py",
    "arbitrary_plane_section_processing_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_section_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)
_ARRAY_KEYS = {
    "source_scalar_crop_float32",
    "source_label_ground_truth_crop_int64",
    "source_tissue_ground_truth_mask",
    "source_correspondence_domain_mask",
    "source_dense_correspondence_weight_float32",
    "source_dense_correspondence_abstention_mask",
    "processed_mapped_ccf_physical_coordinates_crop_float64",
    "processed_bilinear_domain_valid_mask",
    "processed_nearest_domain_valid_mask",
    "processed_dense_coordinate_valid_mask",
    "normalized_template_float32",
    "label_conditioned_latent_float32",
    "acquired_background_float32",
    "pre_damage_acquired_image_float32",
    "raw_acquired_image_float32",
    "physical_loss_mask",
    "occlusion_mask",
    "appearance_artifact_mask",
    "damage_union_mask",
    "observable_footprint_mask",
    "observation_invalid_mask",
    "outside_correspondence_domain_mask",
    "valid_correspondence_mask",
    "valid_correspondence_weight_float32",
}
_DESCENDANT_ARRAY_KEYS = {
    "model_input_image_float32",
    "selected_input_mask",
    "brush_mask_error_mask",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be a nonnegative uint64 integer")
    return result


def _root_seed_uint64(root_seed: int | str) -> int:
    if isinstance(root_seed, str):
        if (
            len(root_seed) != 18
            or not root_seed.startswith("0x")
            or any(character not in "0123456789abcdef" for character in root_seed[2:])
        ):
            raise ValueError("root_seed must be uint64 or 0x plus 16 lowercase hex digits")
        return int(root_seed[2:], 16)
    if isinstance(root_seed, (bool, np.bool_)) or not isinstance(
        root_seed, (int, np.integer)
    ):
        raise ValueError("root_seed must be a uint64 integer or canonical hex string")
    value = int(root_seed)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("root_seed must fit uint64")
    return value


def derive_observation_seed_v2(
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    section_index: int,
    observation_index: int,
    stage: str,
    field: str,
    attempt: int = 0,
) -> int:
    """Derive one PCG64DXSM seed without consuming any specimen or artifact ID."""
    if not isinstance(split, str) or not split:
        raise ValueError("observation RNG split must be a nonempty string")
    if not isinstance(stage, str) or not stage or not isinstance(field, str) or not field:
        raise ValueError("observation RNG stage and field must be nonempty strings")
    numeric = (
        _root_seed_uint64(root_seed),
        _nonnegative_integer(split_index, "split_index"),
        _nonnegative_integer(animal_index, "animal_index"),
        _nonnegative_integer(section_index, "section_index"),
        _nonnegative_integer(observation_index, "observation_index"),
        _nonnegative_integer(attempt, "attempt"),
    )
    digest = hashlib.blake2b(digest_size=8, person=b"AT-OBS-V2")
    parts = (
        b"anatomy-tracker.observation-rng/v2",
        numeric[0].to_bytes(8, "little", signed=False),
        split.encode("utf-8"),
        *(value.to_bytes(8, "little", signed=False) for value in numeric[1:]),
        stage.encode("utf-8"),
        field.encode("utf-8"),
    )
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return int.from_bytes(digest.digest(), "little", signed=False)


def _rng(
    provenance: dict[str, object],
    stage: str,
    field: str,
    receipts: dict[str, dict[str, object]],
    attempt: int = 0,
) -> np.random.Generator:
    seed = derive_observation_seed_v2(
        provenance["root_seed_uint64"],
        provenance["split"],
        provenance["split_index"],
        provenance["animal_index"],
        provenance["section_index"],
        provenance["observation_index"],
        stage,
        field,
        attempt,
    )
    key = f"{stage}/{field}/attempt-{attempt}"
    receipts[key] = {
        "split": provenance["split"],
        "split_index": provenance["split_index"],
        "animal_index": provenance["animal_index"],
        "section_index": provenance["section_index"],
        "observation_index": provenance["observation_index"],
        "stage": stage,
        "field": field,
        "attempt": attempt,
        "seed_uint64": f"0x{seed:016x}",
        "generator": "NumPy PCG64DXSM",
    }
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _engineering_priors() -> dict[str, object]:
    return {
        "status": (
            "initial engineering priors for synthetic coverage; not fitted population "
            "statistics and subject to later empirical refinement"
        ),
        "crop": {
            "height_fraction_range": [0.82, 0.98],
            "width_fraction_range": [0.82, 0.98],
            "centre_jitter_fraction": 0.08,
            "operator": "integer parent-raster window; no interpolation or resize",
        },
        "modalities": {
            "brightfield-nissl-like": {
                "forward_assumption": (
                    "Beer-Lambert-like transmitted-light attenuation through stain; "
                    "bright glass/background"
                ),
                "noise_assumption": "Poisson photon counting plus additive Gaussian read noise",
                "background_assumption": (
                    "bright glass/background under high smooth illumination with "
                    "low-amplitude sensor noise"
                ),
                "label_blend_weight": 0.22,
                "label_level_range": [0.15, 0.85],
                "optical_density_base_range": [0.18, 0.38],
                "optical_density_contrast_range": [0.9, 1.8],
                "photon_count_range": [800.0, 2200.0],
                "read_noise_std_range": [0.002, 0.012],
                "background_level_range": [0.78, 0.98],
                "background_texture_range": [0.01, 0.06],
            },
            "fluorescence": {
                "forward_assumption": (
                    "additive positive fluorophore emission with nonlinear response; "
                    "no transmitted-light inversion"
                ),
                "noise_assumption": "lower-count Poisson shot noise plus additive Gaussian read noise",
                "background_assumption": "dark field with smooth autofluorescence and sensor offset",
                "label_blend_weight": 0.30,
                "label_level_range": [0.08, 1.0],
                "emission_baseline_range": [0.01, 0.08],
                "emission_gain_range": [0.55, 1.0],
                "emission_gamma_range": [0.8, 1.4],
                "photon_count_range": [150.0, 700.0],
                "read_noise_std_range": [0.004, 0.025],
                "background_level_range": [0.0, 0.08],
                "background_texture_range": [0.005, 0.035],
            },
        },
        "damage": {
            "sampling_policy": (
                "sample one clean/mild/moderate/severe specimen stratum, then bounded "
                "disjoint non-invertible events; initial engineering prior, tune later"
            ),
            "stratum_probabilities": {
                "clean": 0.10,
                "mild": 0.45,
                "moderate": 0.35,
                "severe": 0.10,
            },
            "strata": {
                "clean": {
                    "event_count_range_inclusive": [0, 0],
                    "radius_fraction_range": [0.0, 0.0],
                    "maximum_damaged_tissue_fraction": 0.0,
                },
                "mild": {
                    "event_count_range_inclusive": [1, 1],
                    "radius_fraction_range": [0.020, 0.050],
                    "maximum_damaged_tissue_fraction": 0.08,
                },
                "moderate": {
                    "event_count_range_inclusive": [2, 3],
                    "radius_fraction_range": [0.035, 0.085],
                    "maximum_damaged_tissue_fraction": 0.22,
                },
                "severe": {
                    "event_count_range_inclusive": [4, 6],
                    "radius_fraction_range": [0.055, 0.140],
                    "maximum_damaged_tissue_fraction": 0.45,
                },
            },
            "event_category_probabilities": {
                "physical-loss": 0.35,
                "occlusion": 0.35,
                "appearance-artifact": 0.30,
            },
            "geometry_families": {
                "physical-loss": ["edge-bite", "tear-stripe"],
                "occlusion": ["ellipse", "stripe"],
                "appearance-artifact": ["ellipse", "stripe"],
            },
            "physical_loss_operator": "replace missing tissue with the already acquired background",
            "occlusion_operator": "replace tissue with a modality-conditioned opaque value",
            "appearance_artifact_operator": "replace tissue with a modality-conditioned saturated value",
            "category_precedence": [
                "physical_loss",
                "occlusion",
                "appearance_artifact",
            ],
        },
        "smart_brush": {
            "morphology_radius_px": 2,
            "jitter_amplitude_px": 1.25,
            "gap_radius_fraction": [0.02, 0.05],
            "island_radius_fraction": [0.01, 0.035],
            "available_background": "exact positive float32 zero outside the selected mask",
            "absent_background": "byte-identical raw acquired image",
        },
    }


def _dependencies() -> dict[str, list[object]]:
    return {
        "learned_checkpoint_dependencies": [],
        "pretrained_feature_dependencies": [],
        "previous_model_dependencies": [],
        "learned_style_model_dependencies": [],
    }


def _disclosure() -> dict[str, object]:
    return {
        "atlas_labels_used_as_synthetic_ground_truth_only": True,
        "atlas_label_integer_magnitudes_used_as_appearance_values": False,
        "atlas_labels_exposed_to_model_inputs": False,
        "label_conditioning_role": (
            "per-region random appearance levels blended with the authenticated atlas template"
        ),
        "smart_brush_role": (
            "optional input descendant only; never a tissue, damage, or correspondence target"
        ),
        "trainable_input_modes": [
            "smart-brush-accurate",
            "smart-brush-imperfect",
            "smart-brush-absent",
        ],
        "raw_descendant_role": (
            "nontrainable audit mirror of smart-brush-absent; never sample both as "
            "independent training inputs"
        ),
        "brush_availability_model_input": (
            "explicit descendant scalar; the model loader broadcasts it as a constant channel"
        ),
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


def _positive_black(array: np.ndarray, mask: np.ndarray) -> bool:
    outside = np.ascontiguousarray(np.asarray(array, dtype=np.float32)[~mask])
    return bool(np.all(outside.view(np.uint32) == 0))


def _smooth_unit_field(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    field = rng.normal(size=shape).astype(np.float32)
    field = ndimage.gaussian_filter(
        field, sigma=max(shape) / 12.0, mode="reflect"
    ).astype(np.float32)
    field -= np.float32(field.mean(dtype=np.float64))
    scale = float(field.std(dtype=np.float64))
    return field / np.float32(scale) if scale > 1.0e-7 else np.zeros(shape, np.float32)


def _ellipse(
    shape: tuple[int, int], center_yx: tuple[float, float], radius_yx: tuple[float, float]
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    ry, rx = radius_yx
    return ((y - cy) / max(ry, 1.0)) ** 2 + ((x - cx) / max(rx, 1.0)) ** 2 <= 1.0


def _upstream_reference(
    processed_render: dict[str, object],
    section_processing_plan: dict[str, object],
) -> dict[str, object]:
    plan = processed_render["plan_reference"]
    provenance = section_processing_plan["provenance"]
    pose = processed_render["pose_anatomy_policy"]["pose_anatomy_reference"]
    source_stage = processed_render["source_input_reference"]["source_stage_receipt"]
    return {
        "section_processing_render_id": processed_render[
            "section_processing_render_id"
        ],
        "section_processing_render_receipt_sha256": acquisition._payload_sha256(
            section_processing_render_receipt_v2(processed_render)
        ),
        "section_processing_plan_id": plan["section_processing_plan_id"],
        "section_processing_realization_id": plan[
            "section_processing_realization_id"
        ],
        "synthetic_section_processing_id": plan["synthetic_section_processing_id"],
        "split": provenance["split"],
        "animal_index": provenance["animal_index"],
        "animal_id": provenance["animal_id"],
        "section_index": provenance["section_index"],
        "source_input_receipt_sha256": processed_render["source_input_reference"][
            "source_input_receipt_sha256"
        ],
        "subject_slab_render_id": source_stage["subject_slab_render_id"],
        "source_subject_coordinate_map_id": pose[
            "source_subject_coordinate_map_id"
        ],
        "v2_context_sha256": pose["context_reference"]["v2_context_sha256"],
        "precursor_slab_render_id": pose["precursor_reference"]["slab_render_id"],
    }


def _source_arrays(
    processed_render: dict[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    raster = processed_render["raster"]
    state = processed_render["state"]
    scalar = np.asarray(raster["scalar"])
    labels = np.asarray(raster["slab_modal_annotation"])
    tissue = np.asarray(raster["slab_observable_support_mask"])
    abstention = np.asarray(
        raster["slab_supervision_weight_or_abstention"]["abstention_mask"]
    )
    dense_weight = np.asarray(
        raster["slab_supervision_weight_or_abstention"][
            "dense_correspondence_weight"
        ]
    )
    mapped = np.asarray(
        processed_render["mapped_ccf_physical_coordinates_ap_dv_ml_um"]
    )
    bilinear_valid = np.asarray(state["bilinear_domain_valid_mask"])
    nearest_valid = np.asarray(state["nearest_domain_valid_mask"])
    dense_valid = np.asarray(state["dense_coordinate_valid_mask"])
    if (
        scalar.ndim != 2
        or scalar.dtype != np.float32
        or labels.shape != scalar.shape
        or labels.dtype != np.int64
        or tissue.shape != scalar.shape
        or tissue.dtype != bool
        or abstention.shape != scalar.shape
        or abstention.dtype != bool
        or dense_weight.shape != scalar.shape
        or dense_weight.dtype != np.float32
        or not np.isfinite(dense_weight).all()
        or np.any((dense_weight < 0.0) | (dense_weight > 1.0))
        or np.any(dense_weight[abstention] != 0.0)
        or np.any(dense_weight[~tissue] != 0.0)
        or mapped.shape != scalar.shape + (3,)
        or mapped.dtype != np.float64
        or any(
            mask.shape != scalar.shape or mask.dtype != bool
            for mask in (bilinear_valid, nearest_valid, dense_valid)
        )
        or not np.isfinite(scalar).all()
    ):
        raise ValueError("authenticated processed arrays do not satisfy observation inputs")
    correspondence = (
        dense_valid
        & np.isfinite(mapped).all(axis=-1)
        & (dense_weight > 0.0)
        & ~abstention
    )
    return (
        scalar,
        labels,
        tissue,
        correspondence,
        mapped,
        bilinear_valid,
        nearest_valid,
        dense_valid,
        dense_weight,
        abstention,
    )


def _sample_crop_window(
    tissue: np.ndarray,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[int, int, int, int]:
    height, width = tissue.shape
    crop_prior = priors["crop"]
    scale_y = float(
        _rng(provenance, "crop", "height-fraction", rng_receipts).uniform(
            *crop_prior["height_fraction_range"]
        )
    )
    scale_x = float(
        _rng(provenance, "crop", "width-fraction", rng_receipts).uniform(
            *crop_prior["width_fraction_range"]
        )
    )
    crop_height = min(height, max(2, int(round(scale_y * height))))
    crop_width = min(width, max(2, int(round(scale_x * width))))
    if tissue.any():
        center_y, center_x = np.argwhere(tissue).mean(axis=0)
    else:
        center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    jitter = float(crop_prior["centre_jitter_fraction"])
    center_y += float(
        _rng(provenance, "crop", "centre-jitter-y", rng_receipts).uniform(
            -jitter * crop_height, jitter * crop_height
        )
    )
    center_x += float(
        _rng(provenance, "crop", "centre-jitter-x", rng_receipts).uniform(
            -jitter * crop_width, jitter * crop_width
        )
    )
    top = int(np.clip(round(center_y - crop_height / 2.0), 0, height - crop_height))
    left = int(np.clip(round(center_x - crop_width / 2.0), 0, width - crop_width))
    return top, left, crop_height, crop_width


def _crop_metadata(
    processed_render: dict[str, object],
    section_processing_plan: dict[str, object],
    window: tuple[int, int, int, int],
) -> dict[str, object]:
    top, left, height, width = window
    parent_height, parent_width = processed_render["raster"]["scalar"].shape
    pitch = np.asarray(
        section_processing_plan["resolved_config"]["pixel_pitch_y_x_um"],
        dtype=np.float64,
    )
    mapped = np.ascontiguousarray(
        processed_render["mapped_ccf_physical_coordinates_ap_dv_ml_um"][
            top : top + height, left : left + width
        ],
        dtype=np.float64,
    )
    pose = processed_render["pose_anatomy_policy"]["pose_anatomy_reference"]
    return {
        "parent_shape_h_w": [parent_height, parent_width],
        "top_left_y_x": [top, left],
        "output_shape_h_w": [height, width],
        "parent_window_half_open_y_x": [top, top + height, left, left + width],
        "operator": "integer parent-raster slice; no interpolation or resize",
        "processed_pixel_pitch_y_x_um": pitch.tolist(),
        "processed_closed_face_window_y_x_um": [
            [float(top * pitch[0]), float(left * pitch[1])],
            [float((top + height) * pitch[0]), float((left + width) * pitch[1])],
        ],
        "processed_mapped_ccf_coordinate_crop_receipt": acquisition._array_receipt(
            mapped
        ),
        "pose_anatomy_reference_sha256": acquisition._payload_sha256(
            acquisition._json_value(pose)
        ),
        "processed_mapping_contract_sha256": acquisition._payload_sha256(
            acquisition._json_value(processed_render["mapping_contract"])
        ),
        "plane_target_policy": (
            "observation-only window; the upstream global arbitrary-plane target is unchanged"
        ),
        "downstream_coordinate_contract": (
            "model preprocessing and later composition must use the stored cropped mapped-CCF "
            "coordinates and validity masks, then place pixel outputs through the exact parent "
            "half-open window; the crop must not be reinterpreted as a new plane"
        ),
        "nonlinear_coordinate_policy": (
            "the same integer crop applies to the processed image, full mapped-CCF "
            "coordinate raster, and all processed validity masks; upstream pose, anatomy, "
            "and nonlinear residual remain bound by the processed-render receipt"
        ),
    }


def _normalize_template(
    scalar: np.ndarray, tissue: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    normalized = np.zeros(scalar.shape, dtype=np.float32)
    values = scalar[tissue].astype(np.float64)
    if not len(values):
        raise ValueError("observation stage requires nonempty section tissue")
    if len(values) >= 256:
        lower, upper = np.quantile(values, (0.01, 0.99), method="linear")
        method = "q01-q99"
    else:
        lower, upper = float(values.min()), float(values.max())
        method = "min-max-small-support"
    if upper <= lower:
        normalized[tissue] = np.float32(0.5)
    else:
        normalized[tissue] = np.clip(
            (values - lower) / (upper - lower), 0.0, 1.0
        ).astype(np.float32)
    return normalized, {
        "method": method,
        "lower": float(lower),
        "upper": float(upper),
        "tissue_pixel_count": int(tissue.sum()),
    }


def _label_conditioned_latent(
    normalized: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    modality: str,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    modality_prior: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    region_ids = np.unique(labels[tissue])
    region_levels = _rng(
        provenance, "appearance", "label-region-levels", rng_receipts
    ).uniform(*modality_prior["label_level_range"], size=len(region_ids)).astype(np.float32)
    label_image = np.zeros(normalized.shape, dtype=np.float32)
    for ordinal, region_id in enumerate(region_ids):
        label_image[tissue & (labels == region_id)] = region_levels[ordinal]
    weight = float(modality_prior["label_blend_weight"])
    latent = np.zeros(normalized.shape, dtype=np.float32)
    latent[tissue] = (
        (1.0 - weight) * normalized[tissue] + weight * label_image[tissue]
    )
    return np.clip(latent, 0.0, 1.0).astype(np.float32), {
        "modality": modality,
        "operator": "equality-defined atlas regions receive random levels; label integers are not intensities",
        "template_weight": 1.0 - weight,
        "label_weight": weight,
        "region_ids": [int(value) for value in region_ids],
        "region_levels": [float(value) for value in region_levels],
    }


def _uniform(
    provenance: dict[str, object],
    stage: str,
    field: str,
    receipts: dict[str, dict[str, object]],
    bounds: list[float],
) -> float:
    return float(_rng(provenance, stage, field, receipts).uniform(*bounds))


def _appearance(
    scalar: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    modality: str,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    modality_prior = priors["modalities"][modality]
    normalized, normalization = _normalize_template(scalar, tissue)
    latent, label_conditioning = _label_conditioned_latent(
        normalized,
        labels,
        tissue,
        modality,
        provenance,
        rng_receipts,
        modality_prior,
    )
    background_level = _uniform(
        provenance,
        "appearance",
        "background-level",
        rng_receipts,
        modality_prior["background_level_range"],
    )
    background_texture = _uniform(
        provenance,
        "appearance",
        "background-texture-strength",
        rng_receipts,
        modality_prior["background_texture_range"],
    )
    background = background_level + background_texture * _smooth_unit_field(
        scalar.shape,
        _rng(provenance, "appearance", "background-field", rng_receipts),
    )
    background += _rng(
        provenance, "appearance", "background-sensor-noise", rng_receipts
    ).normal(0.0, 0.002 if modality == "brightfield-nissl-like" else 0.004, scalar.shape)
    background = np.clip(background, 0.0, 1.0).astype(np.float32)

    photon_count = _uniform(
        provenance,
        "appearance",
        "photon-count",
        rng_receipts,
        modality_prior["photon_count_range"],
    )
    read_noise = _uniform(
        provenance,
        "appearance",
        "read-noise-std",
        rng_receipts,
        modality_prior["read_noise_std_range"],
    )
    bias = 1.0 + np.float32(0.08) * _smooth_unit_field(
        scalar.shape, _rng(provenance, "appearance", "tissue-bias-field", rng_receipts)
    )
    if modality == "brightfield-nissl-like":
        base = _uniform(
            provenance,
            "appearance",
            "optical-density-base",
            rng_receipts,
            modality_prior["optical_density_base_range"],
        )
        contrast = _uniform(
            provenance,
            "appearance",
            "optical-density-contrast",
            rng_receipts,
            modality_prior["optical_density_contrast_range"],
        )
        expected = np.clip(bias * np.exp(-(base + contrast * (1.0 - latent))), 0.0, 1.0)
        forward_parameters = {
            "family": "transmitted-light Beer-Lambert-like attenuation",
            "optical_density_base": base,
            "optical_density_contrast": contrast,
        }
    else:
        baseline = _uniform(
            provenance,
            "appearance",
            "emission-baseline",
            rng_receipts,
            modality_prior["emission_baseline_range"],
        )
        gain = _uniform(
            provenance,
            "appearance",
            "emission-gain",
            rng_receipts,
            modality_prior["emission_gain_range"],
        )
        gamma = _uniform(
            provenance,
            "appearance",
            "emission-gamma",
            rng_receipts,
            modality_prior["emission_gamma_range"],
        )
        expected = np.clip(baseline + gain * bias * np.power(latent, gamma), 0.0, 1.0)
        forward_parameters = {
            "family": "additive positive fluorescence emission",
            "emission_baseline": baseline,
            "emission_gain": gain,
            "emission_gamma": gamma,
        }
    shot = _rng(provenance, "appearance", "shot-noise", rng_receipts).poisson(
        np.asarray(expected, dtype=np.float64) * photon_count
    ) / photon_count
    shot += _rng(provenance, "appearance", "read-noise", rng_receipts).normal(
        0.0, read_noise, scalar.shape
    )
    tissue_image = np.clip(shot, 0.0, 1.0).astype(np.float32)
    pre_damage = np.where(tissue, tissue_image, background).astype(np.float32)
    return {
        "normalized_template_float32": normalized,
        "label_conditioned_latent_float32": latent,
        "acquired_background_float32": background,
        "pre_damage_acquired_image_float32": pre_damage,
    }, {
        "normalization": normalization,
        "label_conditioning": label_conditioning,
        "forward_parameters": forward_parameters,
        "photon_count": photon_count,
        "read_noise_std": read_noise,
        "operator_order": [
            "template normalization",
            "ground-truth label-conditioned latent appearance",
            "modality forward model",
            "Poisson shot noise",
            "Gaussian read noise",
            "independently acquired background",
        ],
    }


def _event_mask(
    tissue: np.ndarray,
    excluded: np.ndarray,
    category: str,
    event_index: int,
    radius_fraction_range: list[float],
    maximum_new_pixels: int,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    remaining = tissue & ~excluded
    candidates = remaining
    if category == "physical-loss":
        boundary = remaining & ~ndimage.binary_erosion(remaining)
        candidates = boundary if boundary.any() else remaining
    if not candidates.any():
        raise ValueError("insufficient tissue for disjoint observation damage categories")
    rng = _rng(
        provenance,
        "damage",
        f"event-{event_index:02d}-{category}-geometry",
        rng_receipts,
    )
    yx = np.argwhere(candidates)
    center_y, center_x = yx[int(rng.integers(len(yx)))]
    low, high = radius_fraction_range
    radius_y = max(1.0, tissue.shape[0] * float(rng.uniform(low, high)))
    radius_x = max(1.0, tissue.shape[1] * float(rng.uniform(low, high)))
    geometries = priors["damage"]["geometry_families"][category]
    geometry = str(rng.choice(np.asarray(geometries)))
    angle_radians = float(rng.uniform(-np.pi, np.pi))
    if geometry in {"ellipse", "edge-bite"}:
        mask = _ellipse(
            tissue.shape,
            (float(center_y), float(center_x)),
            (radius_y, radius_x),
        )
    else:
        y, x = np.mgrid[: tissue.shape[0], : tissue.shape[1]]
        delta_y, delta_x = y - center_y, x - center_x
        along = np.cos(angle_radians) * delta_x + np.sin(angle_radians) * delta_y
        across = -np.sin(angle_radians) * delta_x + np.cos(angle_radians) * delta_y
        mask = (
            (np.abs(along) <= 1.8 * max(radius_y, radius_x))
            & (np.abs(across) <= 0.45 * min(radius_y, radius_x))
        )
    mask &= remaining
    if not mask.any():
        mask[center_y, center_x] = True
    proposed_pixel_count = int(mask.sum())
    budget_clipped = proposed_pixel_count > int(maximum_new_pixels)
    if int(mask.sum()) > int(maximum_new_pixels):
        indices = np.argwhere(mask)
        distance = np.sum(
            (indices - np.asarray([center_y, center_x], dtype=np.float64)) ** 2,
            axis=1,
        )
        retained = indices[np.argsort(distance, kind="stable")[:maximum_new_pixels]]
        mask = np.zeros(tissue.shape, dtype=bool)
        mask[retained[:, 0], retained[:, 1]] = True
    return mask, {
        "event_index": int(event_index),
        "category": category,
        "geometry": geometry,
        "center_y_x": [int(center_y), int(center_x)],
        "radius_y_x_px": [radius_y, radius_x],
        "angle_radians": angle_radians,
        "proposed_pixel_count": proposed_pixel_count,
        "retained_pixel_count": int(mask.sum()),
        "budget_clipped": budget_clipped,
    }


def _damage(
    pre_damage: np.ndarray,
    background: np.ndarray,
    tissue: np.ndarray,
    correspondence: np.ndarray,
    correspondence_weight: np.ndarray,
    modality: str,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    damage_prior = priors["damage"]
    probabilities = damage_prior["stratum_probabilities"]
    strata = tuple(probabilities)
    stratum = str(
        _rng(provenance, "damage", "stratum", rng_receipts).choice(
            np.asarray(strata), p=np.asarray([probabilities[name] for name in strata])
        )
    )
    specification = damage_prior["strata"][stratum]
    minimum_events, maximum_events = specification["event_count_range_inclusive"]
    if minimum_events == maximum_events:
        requested_event_count = int(minimum_events)
    else:
        requested_event_count = int(
            _rng(provenance, "damage", "event-count", rng_receipts).integers(
                minimum_events, maximum_events + 1
            )
        )
    tissue_count = int(tissue.sum())
    maximum_fraction = float(specification["maximum_damaged_tissue_fraction"])
    damage_budget = int(np.floor(maximum_fraction * tissue_count))
    realized_event_count = min(requested_event_count, damage_budget)
    masks = {
        "physical-loss": np.zeros(tissue.shape, dtype=bool),
        "occlusion": np.zeros(tissue.shape, dtype=bool),
        "appearance-artifact": np.zeros(tissue.shape, dtype=bool),
    }
    events = []
    if realized_event_count:
        categories = tuple(damage_prior["event_category_probabilities"])
        category_probabilities = np.asarray(
            [damage_prior["event_category_probabilities"][name] for name in categories]
        )
        schedule = _rng(
            provenance, "damage", "event-category-schedule", rng_receipts
        ).choice(
            np.asarray(categories), size=realized_event_count, p=category_probabilities
        )
        for event_index, category_value in enumerate(schedule):
            category = str(category_value)
            excluded = masks["physical-loss"] | masks["occlusion"] | masks[
                "appearance-artifact"
            ]
            events_remaining = realized_event_count - event_index - 1
            maximum_new_pixels = damage_budget - int(excluded.sum()) - events_remaining
            event, parameters = _event_mask(
                tissue,
                excluded,
                category,
                event_index,
                specification["radius_fraction_range"],
                maximum_new_pixels,
                provenance,
                rng_receipts,
                priors,
            )
            masks[category] |= event
            events.append(parameters)
    physical = masks["physical-loss"]
    occlusion = masks["occlusion"]
    appearance_artifact = masks["appearance-artifact"]
    if not realized_event_count:
        occlusion_value = 0.06 if modality == "brightfield-nissl-like" else 0.91
        artifact_value = 0.94 if modality == "brightfield-nissl-like" else 0.80
    elif modality == "brightfield-nissl-like":
        occlusion_value = float(
            _rng(provenance, "damage", "occlusion-value", rng_receipts).uniform(0.0, 0.12)
        )
        artifact_value = float(
            _rng(provenance, "damage", "appearance-artifact-value", rng_receipts).uniform(0.88, 1.0)
        )
    else:
        occlusion_value = float(
            _rng(provenance, "damage", "occlusion-value", rng_receipts).uniform(0.82, 1.0)
        )
        artifact_value = float(
            _rng(provenance, "damage", "appearance-artifact-value", rng_receipts).uniform(0.65, 0.95)
        )
    raw = pre_damage.copy()
    raw[physical] = background[physical]
    raw[occlusion] = np.float32(occlusion_value)
    raw[appearance_artifact] = np.float32(artifact_value)
    damage_union = physical | occlusion | appearance_artifact
    footprint = tissue & ~physical
    invalid = tissue & damage_union
    outside = ~correspondence
    valid = correspondence & tissue & ~damage_union
    valid_weight = np.where(valid, correspondence_weight, np.float32(0)).astype(
        np.float32
    )
    return {
        "raw_acquired_image_float32": np.ascontiguousarray(raw, dtype=np.float32),
        "physical_loss_mask": physical,
        "occlusion_mask": occlusion,
        "appearance_artifact_mask": appearance_artifact,
        "damage_union_mask": damage_union,
        "observable_footprint_mask": footprint,
        "observation_invalid_mask": invalid,
        "outside_correspondence_domain_mask": outside,
        "valid_correspondence_mask": valid,
        "valid_correspondence_weight_float32": valid_weight,
    }, {
        "stratum": stratum,
        "stratum_probability": float(probabilities[stratum]),
        "requested_event_count": requested_event_count,
        "realized_event_count": realized_event_count,
        "damage_budget_pixels": damage_budget,
        "support_limited": realized_event_count != requested_event_count,
        "event_realization_policy": (
            "realized_event_count=min(requested_event_count,damage_budget_pixels); "
            "authenticated cropped tissue support only"
        ),
        "no_redraw": True,
        "no_target_overlap_conditioning": True,
        "maximum_damaged_tissue_fraction": maximum_fraction,
        "damaged_tissue_fraction": float(damage_union.sum() / tissue_count),
        "events": events,
        "occlusion_value": occlusion_value,
        "appearance_artifact_value": artifact_value,
        "noninvertibility": {
            "physical_loss": "source tissue pixels replaced by independent acquired background",
            "occlusion": "source tissue pixels replaced by opaque constant",
            "appearance_artifact": "source tissue pixels replaced by saturated constant",
        },
    }


def _imperfect_brush_mask(
    footprint: np.ndarray,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    brush = priors["smart_brush"]
    radius = int(brush["morphology_radius_px"])
    morphology = int(
        _rng(provenance, "brush", "morphology", rng_receipts).choice(
            np.asarray([-radius, -1, 1, radius], dtype=np.int64)
        )
    )
    mask = (
        ndimage.binary_dilation(footprint, iterations=morphology)
        if morphology > 0
        else ndimage.binary_erosion(footprint, iterations=-morphology)
    )
    amplitude = float(brush["jitter_amplitude_px"])
    y, x = np.mgrid[: footprint.shape[0], : footprint.shape[1]].astype(np.float32)
    dx = amplitude * _smooth_unit_field(
        footprint.shape, _rng(provenance, "brush", "jitter-x", rng_receipts)
    )
    dy = amplitude * _smooth_unit_field(
        footprint.shape, _rng(provenance, "brush", "jitter-y", rng_receipts)
    )
    mask = ndimage.map_coordinates(
        mask.astype(np.uint8),
        (y + dy, x + dx),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    gap_candidates = np.argwhere(mask & ~ndimage.binary_erosion(mask))
    if len(gap_candidates):
        gap_rng = _rng(provenance, "brush", "gap", rng_receipts)
        gap_y, gap_x = gap_candidates[int(gap_rng.integers(len(gap_candidates)))]
        low, high = brush["gap_radius_fraction"]
        gap = _ellipse(
            footprint.shape,
            (float(gap_y), float(gap_x)),
            (
                footprint.shape[0] * float(gap_rng.uniform(low, high)),
                footprint.shape[1] * float(gap_rng.uniform(low, high)),
            ),
        )
        mask[gap] = False
    ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    island_candidates = np.argwhere(ring)
    if len(island_candidates):
        island_rng = _rng(provenance, "brush", "island", rng_receipts)
        island_y, island_x = island_candidates[
            int(island_rng.integers(len(island_candidates)))
        ]
        low, high = brush["island_radius_fraction"]
        island = _ellipse(
            footprint.shape,
            (float(island_y), float(island_x)),
            (
                footprint.shape[0] * float(island_rng.uniform(low, high)),
                footprint.shape[1] * float(island_rng.uniform(low, high)),
            ),
        )
        mask[island] = True
    if np.array_equal(mask, footprint):
        candidates = np.argwhere(mask)
        if len(candidates):
            fallback_rng = _rng(provenance, "brush", "difference-fallback", rng_receipts)
            yy, xx = candidates[int(fallback_rng.integers(len(candidates)))]
            mask[yy, xx] = False
    union = int((mask | footprint).sum())
    empty_selection = not bool(mask.any())
    return mask, {
        "operator": "independent morphology, smooth boundary jitter, gap, and exterior island",
        "morphology_px": morphology,
        "quality_iou": float((mask & footprint).sum() / union) if union else 1.0,
        "empty_selection": empty_selection,
        "selection_failure_tag": (
            "empty-imperfect-brush-selection" if empty_selection else "none"
        ),
        "independent_of_damage_rng": True,
        "used_as_correspondence_truth": False,
    }


def _descendant(
    mode: str,
    raw: np.ndarray,
    selected: np.ndarray,
    brush_error: np.ndarray,
    acquired_observation_id: str,
    parameters: dict[str, object],
    *,
    trainable: bool = True,
) -> dict[str, object]:
    available = mode in {"smart-brush-accurate", "smart-brush-imperfect"}
    if available:
        image = np.zeros(raw.shape, dtype=np.float32)
        image[selected] = raw[selected]
        background_policy = "exact-positive-float32-black-outside-selected-mask"
    else:
        image = raw.copy()
        background_policy = "byte-identical-raw-acquired-background"
    arrays = {
        "model_input_image_float32": image,
        "selected_input_mask": selected.copy(),
        "brush_mask_error_mask": brush_error.copy(),
    }
    result = {
        "schema_version": OBSERVATION_DESCENDANT_V2_SCHEMA,
        "mode": mode,
        "trainable": bool(trainable),
        "brush_available": available,
        "acquired_observation_id": acquired_observation_id,
        "background_policy": background_policy,
        "parameters": parameters,
        "arrays": arrays,
        "array_receipts": _receipts(arrays),
    }
    result["descendant_id"] = acquisition._payload_sha256(
        _descendant_identity_payload(result)
    )
    return result


def _plan_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "domain": "anatomy-tracker.observation-plan/v2",
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "implementation_source_sha256": artifact["implementation_source_sha256"],
        "implementation_source_sha256_canonicalization": artifact[
            "implementation_source_sha256_canonicalization"
        ],
        "runtime_dependencies": artifact["runtime_dependencies"],
        "asset_dependencies": artifact["asset_dependencies"],
        "upstream_reference": artifact["upstream_reference"],
        "provenance": artifact["provenance"],
        "modality": artifact["modality"],
        "modality_model": artifact["modality_model"],
        "engineering_priors": artifact["engineering_priors"],
        "disclosure": artifact["disclosure"],
    }


def _crop_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "domain": "anatomy-tracker.observation-crop-window/v2",
        "observation_plan_id": artifact["observation_plan_id"],
        "crop_window": artifact["crop_window"],
    }


def _acquired_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "domain": "anatomy-tracker.acquired-observation/v2",
        "observation_plan_id": artifact["observation_plan_id"],
        "crop_window_id": artifact["crop_window_id"],
        "parameters": artifact["parameters"],
        "acquisition_rng_sources": artifact["rng_sources"]["acquisition"],
        "array_receipts": artifact["array_receipts"],
    }


def _descendant_identity_payload(descendant: dict[str, object]) -> dict[str, object]:
    return {
        "domain": OBSERVATION_DESCENDANT_V2_SCHEMA,
        "schema_version": descendant["schema_version"],
        "mode": descendant["mode"],
        "trainable": descendant["trainable"],
        "brush_available": descendant["brush_available"],
        "acquired_observation_id": descendant["acquired_observation_id"],
        "background_policy": descendant["background_policy"],
        "parameters": descendant["parameters"],
        "array_receipts": descendant["array_receipts"],
    }


def _bundle_identity_payload(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "domain": "anatomy-tracker.observation-bundle/v2",
        "observation_plan_id": artifact["observation_plan_id"],
        "acquired_observation_id": artifact["acquired_observation_id"],
        "descendant_ids": {
            name: artifact["descendants"][name]["descendant_id"]
            for name in DESCENDANT_MODES
        },
        "brush_rng_sources": artifact["rng_sources"]["brush"],
    }


def observation_bundle_receipt_v2(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "observation_plan_id": artifact["observation_plan_id"],
        "crop_window_id": artifact["crop_window_id"],
        "acquired_observation_id": artifact["acquired_observation_id"],
        "observation_bundle_id": artifact["observation_bundle_id"],
        "plan_identity_payload": _plan_identity_payload(artifact),
        "crop_identity_payload": _crop_identity_payload(artifact),
        "acquired_identity_payload": _acquired_identity_payload(artifact),
        "bundle_identity_payload": _bundle_identity_payload(artifact),
        "descendant_identity_payloads": {
            name: _descendant_identity_payload(artifact["descendants"][name])
            for name in DESCENDANT_MODES
        },
    }


def make_arbitrary_plane_observation_v2(
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    observation_index: int,
    modality: str,
) -> dict[str, object]:
    """Create one acquired observation plus paired raw/brush input descendants."""
    root_seed_value = _root_seed_uint64(root_seed)
    if not isinstance(split, str) or not split:
        raise ValueError("observation split must be a nonempty string")
    split_index = _nonnegative_integer(split_index, "split_index")
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    section_index = _nonnegative_integer(section_index, "section_index")
    observation_index = _nonnegative_integer(observation_index, "observation_index")
    verify_section_processing_render_v2(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
    )
    upstream_provenance = section_processing_plan["provenance"]
    animal_id = acquisition._json_value(animal_id)
    if (
        modality not in MODALITIES
        or animal_id == ""
        or split != upstream_provenance["split"]
        or animal_index != upstream_provenance["animal_index"]
        or acquisition._payload_sha256({"animal_id": animal_id})
        != acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(upstream_provenance["animal_id"])}
        )
        or section_index != upstream_provenance["section_index"]
    ):
        raise ValueError("observation modality or authoritative upstream lineage does not match")
    provenance = {
        "root_seed_uint64": f"0x{root_seed_value:016x}",
        "split": split,
        "split_index": split_index,
        "animal_index": animal_index,
        "animal_id": animal_id,
        "section_index": section_index,
        "observation_index": observation_index,
        "rng_dynamic_coordinates": (
            "observation augmentation stream only: root_seed and split_index are "
            "observation-stage coordinates independent of authenticated upstream generator "
            "coordinates; split/animal_index/section_index remain bound to upstream; "
            "animal_id and artifact IDs are excluded from RNG"
        ),
    }
    if min(
        provenance["split_index"],
        provenance["animal_index"],
        provenance["section_index"],
        provenance["observation_index"],
    ) < 0:
        raise ValueError("numeric observation provenance must be nonnegative")
    priors = _engineering_priors()
    artifact = {
        "schema_version": OBSERVATION_V2_SCHEMA,
        "algorithm": OBSERVATION_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "ndimage_boundary_modes": "reflect for fields; constant zero for brush warp",
        },
        "asset_dependencies": _dependencies(),
        "upstream_reference": _upstream_reference(
            processed_render, section_processing_plan
        ),
        "provenance": provenance,
        "modality": modality,
        "modality_model": priors["modalities"][modality],
        "engineering_priors": priors,
        "disclosure": _disclosure(),
    }
    artifact["observation_plan_id"] = acquisition._payload_sha256(
        _plan_identity_payload(artifact)
    )

    acquisition_rng_sources: dict[str, dict[str, object]] = {}
    (
        scalar,
        labels,
        tissue,
        correspondence,
        mapped,
        bilinear_valid,
        nearest_valid,
        dense_valid,
        dense_weight,
        abstention,
    ) = _source_arrays(processed_render)
    window = _sample_crop_window(
        tissue, provenance, acquisition_rng_sources, priors
    )
    top, left, height, width = window
    window_slice = np.s_[top : top + height, left : left + width]
    scalar = np.ascontiguousarray(scalar[window_slice], dtype=np.float32)
    labels = np.ascontiguousarray(labels[window_slice], dtype=np.int64)
    tissue = np.ascontiguousarray(tissue[window_slice], dtype=bool)
    correspondence = np.ascontiguousarray(correspondence[window_slice], dtype=bool)
    mapped = np.ascontiguousarray(mapped[window_slice], dtype=np.float64)
    bilinear_valid = np.ascontiguousarray(bilinear_valid[window_slice], dtype=bool)
    nearest_valid = np.ascontiguousarray(nearest_valid[window_slice], dtype=bool)
    dense_valid = np.ascontiguousarray(dense_valid[window_slice], dtype=bool)
    dense_weight = np.ascontiguousarray(dense_weight[window_slice], dtype=np.float32)
    abstention = np.ascontiguousarray(abstention[window_slice], dtype=bool)
    if int(tissue.sum()) < 1:
        raise ValueError("cropped observation does not intersect authenticated tissue")
    crop_window = _crop_metadata(
        processed_render, section_processing_plan, window
    )
    artifact["crop_window"] = crop_window
    artifact["crop_window_id"] = acquisition._payload_sha256(
        _crop_identity_payload(artifact)
    )

    appearance_arrays, appearance_parameters = _appearance(
        scalar,
        labels,
        tissue,
        modality,
        provenance,
        acquisition_rng_sources,
        priors,
    )
    damage_arrays, damage_parameters = _damage(
        appearance_arrays["pre_damage_acquired_image_float32"],
        appearance_arrays["acquired_background_float32"],
        tissue,
        correspondence,
        dense_weight,
        modality,
        provenance,
        acquisition_rng_sources,
        priors,
    )
    arrays = {
        "source_scalar_crop_float32": scalar,
        "source_label_ground_truth_crop_int64": labels,
        "source_tissue_ground_truth_mask": tissue,
        "source_correspondence_domain_mask": correspondence,
        "source_dense_correspondence_weight_float32": dense_weight,
        "source_dense_correspondence_abstention_mask": abstention,
        "processed_mapped_ccf_physical_coordinates_crop_float64": mapped,
        "processed_bilinear_domain_valid_mask": bilinear_valid,
        "processed_nearest_domain_valid_mask": nearest_valid,
        "processed_dense_coordinate_valid_mask": dense_valid,
        **appearance_arrays,
        **damage_arrays,
    }
    artifact["parameters"] = {
        "appearance": appearance_parameters,
        "damage": damage_parameters,
        "mask_algebra": {
            "damage_categories": "pairwise disjoint subsets of source tissue",
            "observable_footprint": "source_tissue AND NOT physical_loss",
            "observation_invalid": "source_tissue AND damage_union",
            "valid_correspondence": (
                "source_correspondence_domain AND source_tissue AND NOT damage_union"
            ),
            "valid_correspondence_weight": (
                "source_dense_correspondence_weight * valid_correspondence_mask; "
                "exact float32 zero otherwise"
            ),
            "brush_mask_error_excluded_from_truth": True,
        },
    }
    artifact["rng_sources"] = {
        "acquisition": acquisition_rng_sources,
        "brush": {},
    }
    artifact["arrays"] = arrays
    artifact["array_receipts"] = _receipts(arrays)
    artifact["acquired_observation_id"] = acquisition._payload_sha256(
        _acquired_identity_payload(artifact)
    )

    raw = arrays["raw_acquired_image_float32"]
    footprint = arrays["observable_footprint_mask"]
    brush_rng_sources: dict[str, dict[str, object]] = {}
    imperfect, imperfect_parameters = _imperfect_brush_mask(
        footprint, provenance, brush_rng_sources, priors
    )
    zeros = np.zeros(footprint.shape, dtype=bool)
    artifact["rng_sources"]["brush"] = brush_rng_sources
    artifact["descendants"] = {
        "raw": _descendant(
            "raw",
            raw,
            zeros,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "nontrainable acquired-input audit mirror",
                "equivalent_trainable_mode": "smart-brush-absent",
                "sampling_policy": "never count raw and absent as separate training inputs",
            },
            trainable=False,
        ),
        "smart-brush-accurate": _descendant(
            "smart-brush-accurate",
            raw,
            footprint,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "oracle-quality smart-brush development descendant",
                "selection_source": "observable footprint after physical loss only",
                "quality_iou": 1.0,
            },
        ),
        "smart-brush-imperfect": _descendant(
            "smart-brush-imperfect",
            raw,
            imperfect,
            imperfect ^ footprint,
            artifact["acquired_observation_id"],
            imperfect_parameters,
        ),
        "smart-brush-absent": _descendant(
            "smart-brush-absent",
            raw,
            zeros,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "trainable no-brush descendant; raw background retained exactly",
                "raw_audit_mirror_mode": "raw",
            },
        ),
    }
    artifact["observation_bundle_id"] = acquisition._payload_sha256(
        _bundle_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        observation_bundle_receipt_v2(artifact)
    )
    return artifact


def replay_arbitrary_plane_observation_v2(
    artifact: dict[str, object],
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    observation_index: int,
    modality: str,
) -> dict[str, object]:
    return make_arbitrary_plane_observation_v2(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        root_seed=root_seed,
        split=split,
        split_index=split_index,
        animal_index=animal_index,
        animal_id=animal_id,
        section_index=section_index,
        observation_index=observation_index,
        modality=modality,
    )


def _contains_forbidden_final_id(value: object) -> bool:
    if isinstance(value, dict):
        return "synthetic_realization_id" in value or any(
            _contains_forbidden_final_id(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_final_id(item) for item in value)
    return False


def _validate_structure(artifact: dict[str, object]) -> None:
    if (
        set(artifact)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "runtime_dependencies",
            "asset_dependencies",
            "upstream_reference",
            "provenance",
            "modality",
            "modality_model",
            "engineering_priors",
            "disclosure",
            "observation_plan_id",
            "crop_window",
            "crop_window_id",
            "parameters",
            "rng_sources",
            "arrays",
            "array_receipts",
            "acquired_observation_id",
            "descendants",
            "observation_bundle_id",
            "receipt_sha256",
        }
        or set(artifact.get("runtime_dependencies", {}))
        != {"numpy_version", "scipy_version", "ndimage_boundary_modes"}
        or set(artifact.get("asset_dependencies", {})) != set(_dependencies())
        or set(artifact.get("upstream_reference", {}))
        != {
            "section_processing_render_id",
            "section_processing_render_receipt_sha256",
            "section_processing_plan_id",
            "section_processing_realization_id",
            "synthetic_section_processing_id",
            "split",
            "animal_index",
            "animal_id",
            "section_index",
            "source_input_receipt_sha256",
            "subject_slab_render_id",
            "source_subject_coordinate_map_id",
            "v2_context_sha256",
            "precursor_slab_render_id",
        }
        or set(artifact.get("provenance", {}))
        != {
            "root_seed_uint64",
            "split",
            "split_index",
            "animal_index",
            "animal_id",
            "section_index",
            "observation_index",
            "rng_dynamic_coordinates",
        }
        or set(artifact.get("crop_window", {}))
        != {
            "parent_shape_h_w",
            "top_left_y_x",
            "output_shape_h_w",
            "parent_window_half_open_y_x",
            "operator",
            "processed_pixel_pitch_y_x_um",
            "processed_closed_face_window_y_x_um",
            "processed_mapped_ccf_coordinate_crop_receipt",
            "pose_anatomy_reference_sha256",
            "processed_mapping_contract_sha256",
            "plane_target_policy",
            "downstream_coordinate_contract",
            "nonlinear_coordinate_policy",
        }
        or set(artifact.get("rng_sources", {})) != {"acquisition", "brush"}
        or set(artifact.get("parameters", {}))
        != {"appearance", "damage", "mask_algebra"}
        or set(artifact.get("parameters", {}).get("appearance", {}))
        != {
            "normalization",
            "label_conditioning",
            "forward_parameters",
            "photon_count",
            "read_noise_std",
            "operator_order",
        }
        or set(artifact.get("parameters", {}).get("damage", {}))
        != {
            "stratum",
            "stratum_probability",
            "requested_event_count",
            "realized_event_count",
            "damage_budget_pixels",
            "support_limited",
            "event_realization_policy",
            "no_redraw",
            "no_target_overlap_conditioning",
            "maximum_damaged_tissue_fraction",
            "damaged_tissue_fraction",
            "events",
            "occlusion_value",
            "appearance_artifact_value",
            "noninvertibility",
        }
        or set(artifact.get("parameters", {}).get("mask_algebra", {}))
        != {
            "damage_categories",
            "observable_footprint",
            "observation_invalid",
            "valid_correspondence",
            "valid_correspondence_weight",
            "brush_mask_error_excluded_from_truth",
        }
        or set(artifact.get("arrays", {})) != _ARRAY_KEYS
        or set(artifact.get("array_receipts", {})) != _ARRAY_KEYS
        or set(artifact.get("descendants", {})) != set(DESCENDANT_MODES)
        or _contains_forbidden_final_id(artifact)
    ):
        raise ValueError("observation artifact has missing, extra, or premature final fields")
    rng_receipt_keys = {
        "split",
        "split_index",
        "animal_index",
        "section_index",
        "observation_index",
        "stage",
        "field",
        "attempt",
        "seed_uint64",
        "generator",
    }
    if any(
        set(receipt) != rng_receipt_keys
        for branch in artifact["rng_sources"].values()
        for receipt in branch.values()
    ):
        raise ValueError("observation RNG receipt has missing or extra fields")
    for name in DESCENDANT_MODES:
        descendant = artifact["descendants"][name]
        if (
            set(descendant)
            != {
                "schema_version",
                "mode",
                "trainable",
                "brush_available",
                "acquired_observation_id",
                "background_policy",
                "parameters",
                "arrays",
                "array_receipts",
                "descendant_id",
            }
            or descendant["mode"] != name
            or set(descendant.get("arrays", {})) != _DESCENDANT_ARRAY_KEYS
            or set(descendant.get("array_receipts", {})) != _DESCENDANT_ARRAY_KEYS
        ):
            raise ValueError("observation descendant has missing or extra fields")


def _validate_mask_and_descendant_algebra(artifact: dict[str, object]) -> None:
    arrays = artifact["arrays"]
    scalar = arrays["source_scalar_crop_float32"]
    labels = arrays["source_label_ground_truth_crop_int64"]
    tissue = arrays["source_tissue_ground_truth_mask"]
    correspondence = arrays["source_correspondence_domain_mask"]
    source_weight = arrays["source_dense_correspondence_weight_float32"]
    source_abstention = arrays["source_dense_correspondence_abstention_mask"]
    mapped = arrays[
        "processed_mapped_ccf_physical_coordinates_crop_float64"
    ]
    bilinear_valid = arrays["processed_bilinear_domain_valid_mask"]
    nearest_valid = arrays["processed_nearest_domain_valid_mask"]
    dense_valid = arrays["processed_dense_coordinate_valid_mask"]
    valid_weight = arrays["valid_correspondence_weight_float32"]
    physical = arrays["physical_loss_mask"]
    occlusion = arrays["occlusion_mask"]
    appearance = arrays["appearance_artifact_mask"]
    damage = physical | occlusion | appearance
    raw = arrays["raw_acquired_image_float32"]
    background = arrays["acquired_background_float32"]
    pre_damage = arrays["pre_damage_acquired_image_float32"]
    damage_parameters = artifact["parameters"]["damage"]
    damage_prior = artifact["engineering_priors"]["damage"]
    stratum = damage_parameters["stratum"]
    specification = damage_prior["strata"].get(stratum)
    requested_event_count = damage_parameters["requested_event_count"]
    realized_event_count = damage_parameters["realized_event_count"]
    events = damage_parameters["events"]
    tissue_count = int(tissue.sum())
    if tissue_count < 1:
        raise ValueError("observation artifact has no authenticated tissue")
    actual_damage_fraction = float(damage.sum() / tissue_count)
    expected_correspondence = (
        dense_valid
        & np.isfinite(mapped).all(axis=-1)
        & (source_weight > 0.0)
        & ~source_abstention
    )
    expected_valid = correspondence & tissue & ~damage
    expected_valid_weight = np.where(
        expected_valid, source_weight, np.float32(0)
    ).astype(np.float32)
    event_keys = {
        "event_index",
        "category",
        "geometry",
        "center_y_x",
        "radius_y_x_px",
        "angle_radians",
        "proposed_pixel_count",
        "retained_pixel_count",
        "budget_clipped",
    }
    if (
        scalar.ndim != 2
        or scalar.dtype != np.float32
        or labels.shape != scalar.shape
        or labels.dtype != np.int64
        or tissue.shape != scalar.shape
        or tissue.dtype != bool
        or correspondence.dtype != bool
        or correspondence.shape != scalar.shape
        or source_weight.shape != scalar.shape
        or source_weight.dtype != np.float32
        or not np.isfinite(source_weight).all()
        or np.any((source_weight < 0.0) | (source_weight > 1.0))
        or source_abstention.shape != scalar.shape
        or source_abstention.dtype != bool
        or np.any(source_weight[source_abstention] != 0.0)
        or np.any(source_weight[~tissue] != 0.0)
        or valid_weight.shape != scalar.shape
        or valid_weight.dtype != np.float32
        or not np.isfinite(valid_weight).all()
        or mapped.shape != scalar.shape + (3,)
        or mapped.dtype != np.float64
        or any(
            mask.shape != scalar.shape or mask.dtype != bool
            for mask in (bilinear_valid, nearest_valid, dense_valid)
        )
        or artifact["crop_window"][
            "processed_mapped_ccf_coordinate_crop_receipt"
        ]
        != acquisition._array_receipt(mapped)
        or not np.array_equal(correspondence, expected_correspondence)
        or np.any(tissue & ~nearest_valid)
        or any(array.dtype != bool for array in (physical, occlusion, appearance))
        or np.any(physical & occlusion)
        or np.any(physical & appearance)
        or np.any(occlusion & appearance)
        or np.any(damage & ~tissue)
        or not np.array_equal(arrays["damage_union_mask"], damage)
        or not np.array_equal(arrays["observable_footprint_mask"], tissue & ~physical)
        or not np.array_equal(arrays["observation_invalid_mask"], tissue & damage)
        or not np.array_equal(arrays["outside_correspondence_domain_mask"], ~correspondence)
        or not np.array_equal(
            arrays["valid_correspondence_mask"], expected_valid
        )
        or not _byte_equal(valid_weight, expected_valid_weight)
        or np.any(valid_weight[source_abstention | damage | ~tissue] != 0.0)
        or specification is None
        or damage_parameters["stratum_probability"]
        != damage_prior["stratum_probabilities"].get(stratum)
        or not (
            specification["event_count_range_inclusive"][0]
            <= requested_event_count
            <= specification["event_count_range_inclusive"][1]
        )
        or damage_parameters["damage_budget_pixels"]
        != int(np.floor(specification["maximum_damaged_tissue_fraction"] * tissue_count))
        or realized_event_count
        != min(requested_event_count, damage_parameters["damage_budget_pixels"])
        or damage_parameters["support_limited"]
        is not (realized_event_count != requested_event_count)
        or damage_parameters["event_realization_policy"]
        != (
            "realized_event_count=min(requested_event_count,damage_budget_pixels); "
            "authenticated cropped tissue support only"
        )
        or damage_parameters["no_redraw"] is not True
        or damage_parameters["no_target_overlap_conditioning"] is not True
        or len(events) != realized_event_count
        or any(
            set(event) != event_keys
            or event.get("event_index") != index
            or event.get("category") not in damage_prior["event_category_probabilities"]
            or event.get("geometry")
            not in damage_prior["geometry_families"][event.get("category")]
            or event.get("proposed_pixel_count", 0) < event.get("retained_pixel_count", 0)
            or event.get("retained_pixel_count", 0) <= 0
            or event.get("budget_clipped")
            is not (event.get("proposed_pixel_count") > event.get("retained_pixel_count"))
            for index, event in enumerate(events)
        )
        or sum(event["retained_pixel_count"] for event in events) != int(damage.sum())
        or damage_parameters["maximum_damaged_tissue_fraction"]
        != specification["maximum_damaged_tissue_fraction"]
        or damage_parameters["damaged_tissue_fraction"] != actual_damage_fraction
        or actual_damage_fraction
        > specification["maximum_damaged_tissue_fraction"] + 1.0e-15
        or (stratum == "clean" and (damage.any() or not _byte_equal(raw, pre_damage)))
        or np.any(arrays["valid_correspondence_mask"] & (damage | ~correspondence))
        or not _byte_equal(raw[physical], background[physical])
        or not _byte_equal(pre_damage[~tissue], background[~tissue])
        or not _byte_equal(raw[~tissue], background[~tissue])
        or not np.all(
            raw[occlusion] == np.float32(damage_parameters["occlusion_value"])
        )
        or not np.all(
            raw[appearance]
            == np.float32(damage_parameters["appearance_artifact_value"])
        )
    ):
        raise ValueError("observation damage or correspondence mask algebra does not match")
    descendants = artifact["descendants"]
    accurate = descendants["smart-brush-accurate"]
    imperfect = descendants["smart-brush-imperfect"]
    absent = descendants["smart-brush-absent"]
    raw_descendant = descendants["raw"]
    footprint = arrays["observable_footprint_mask"]
    imperfect_mask = imperfect["arrays"]["selected_input_mask"]
    imperfect_parameters = imperfect["parameters"]
    empty_imperfect = not bool(imperfect_mask.any())
    if (
        any(
            descendant["schema_version"] != OBSERVATION_DESCENDANT_V2_SCHEMA
            or descendant["trainable"]
            is not (name != "raw")
            or descendant["acquired_observation_id"]
            != artifact["acquired_observation_id"]
            for name, descendant in descendants.items()
        )
        or not np.array_equal(accurate["arrays"]["selected_input_mask"], footprint)
        or accurate["arrays"]["brush_mask_error_mask"].any()
        or not np.array_equal(
            imperfect["arrays"]["brush_mask_error_mask"], imperfect_mask ^ footprint
        )
        or np.array_equal(imperfect_mask, footprint)
        or imperfect_parameters.get("empty_selection") is not empty_imperfect
        or imperfect_parameters.get("selection_failure_tag")
        != ("empty-imperfect-brush-selection" if empty_imperfect else "none")
        or not _positive_black(
            accurate["arrays"]["model_input_image_float32"],
            accurate["arrays"]["selected_input_mask"],
        )
        or not _positive_black(
            imperfect["arrays"]["model_input_image_float32"], imperfect_mask
        )
        or not _byte_equal(absent["arrays"]["model_input_image_float32"], raw)
        or not _byte_equal(raw_descendant["arrays"]["model_input_image_float32"], raw)
        or raw_descendant["parameters"]
        != {
            "role": "nontrainable acquired-input audit mirror",
            "equivalent_trainable_mode": "smart-brush-absent",
            "sampling_policy": "never count raw and absent as separate training inputs",
        }
        or absent["parameters"]
        != {
            "role": "trainable no-brush descendant; raw background retained exactly",
            "raw_audit_mirror_mode": "raw",
        }
        or absent["arrays"]["selected_input_mask"].any()
        or raw_descendant["arrays"]["selected_input_mask"].any()
        or absent["arrays"]["brush_mask_error_mask"].any()
        or raw_descendant["arrays"]["brush_mask_error_mask"].any()
    ):
        raise ValueError("raw or smart-brush descendant algebra does not match")


def verify_arbitrary_plane_observation_v2(
    artifact: dict[str, object],
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    observation_index: int,
    modality: str,
) -> None:
    root_seed_value = _root_seed_uint64(root_seed)
    if not isinstance(split, str) or not split:
        raise ValueError("observation split must be a nonempty string")
    split_index = _nonnegative_integer(split_index, "split_index")
    animal_index = _nonnegative_integer(animal_index, "animal_index")
    section_index = _nonnegative_integer(section_index, "section_index")
    observation_index = _nonnegative_integer(observation_index, "observation_index")
    verify_section_processing_render_v2(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
    )
    upstream_provenance = section_processing_plan["provenance"]
    if (
        split != upstream_provenance["split"]
        or animal_index != upstream_provenance["animal_index"]
        or acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(animal_id)}
        )
        != acquisition._payload_sha256(
            {"animal_id": acquisition._json_value(upstream_provenance["animal_id"])}
        )
        or section_index != upstream_provenance["section_index"]
    ):
        raise ValueError("observation authoritative upstream lineage does not match")
    _validate_structure(artifact)
    expected_provenance = {
        "root_seed_uint64": f"0x{root_seed_value:016x}",
        "split": split,
        "split_index": split_index,
        "animal_index": animal_index,
        "animal_id": acquisition._json_value(animal_id),
        "section_index": section_index,
        "observation_index": observation_index,
        "rng_dynamic_coordinates": (
            "observation augmentation stream only: root_seed and split_index are "
            "observation-stage coordinates independent of authenticated upstream generator "
            "coordinates; split/animal_index/section_index remain bound to upstream; "
            "animal_id and artifact IDs are excluded from RNG"
        ),
    }
    priors = _engineering_priors()
    if (
        artifact["schema_version"] != OBSERVATION_V2_SCHEMA
        or artifact["algorithm"] != OBSERVATION_V2_ALGORITHM
        or artifact["implementation_source_sha256"] != _source_hashes()
        or artifact["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or artifact["runtime_dependencies"]
        != {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "ndimage_boundary_modes": "reflect for fields; constant zero for brush warp",
        }
        or artifact["asset_dependencies"] != _dependencies()
        or artifact["upstream_reference"]
        != _upstream_reference(processed_render, section_processing_plan)
        or artifact["provenance"] != expected_provenance
        or artifact["modality"] != modality
        or artifact["modality_model"] != priors["modalities"][modality]
        or artifact["engineering_priors"] != priors
        or artifact["disclosure"] != _disclosure()
        or artifact["observation_plan_id"]
        != acquisition._payload_sha256(_plan_identity_payload(artifact))
        or artifact["crop_window_id"]
        != acquisition._payload_sha256(_crop_identity_payload(artifact))
        or artifact["array_receipts"] != _receipts(artifact["arrays"])
        or artifact["acquired_observation_id"]
        != acquisition._payload_sha256(_acquired_identity_payload(artifact))
    ):
        raise ValueError("observation source, plan, crop, or acquired receipt does not match")
    for name in DESCENDANT_MODES:
        descendant = artifact["descendants"][name]
        if (
            descendant["array_receipts"] != _receipts(descendant["arrays"])
            or descendant["descendant_id"]
            != acquisition._payload_sha256(_descendant_identity_payload(descendant))
        ):
            raise ValueError("observation descendant live receipt does not match")
    if (
        artifact["observation_bundle_id"]
        != acquisition._payload_sha256(_bundle_identity_payload(artifact))
        or artifact["receipt_sha256"]
        != acquisition._payload_sha256(observation_bundle_receipt_v2(artifact))
    ):
        raise ValueError("observation bundle receipt does not match")
    _validate_mask_and_descendant_algebra(artifact)
    replay = replay_arbitrary_plane_observation_v2(
        artifact,
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        root_seed=root_seed,
        split=split,
        split_index=split_index,
        animal_index=animal_index,
        animal_id=animal_id,
        section_index=section_index,
        observation_index=observation_index,
        modality=modality,
    )
    if observation_bundle_receipt_v2(artifact) != observation_bundle_receipt_v2(
        replay
    ):
        raise ValueError("observation deterministic replay receipt does not match")
    for name in _ARRAY_KEYS:
        if not _byte_equal(artifact["arrays"][name], replay["arrays"][name]):
            raise ValueError("observation deterministic replay arrays do not match")
    for mode in DESCENDANT_MODES:
        for name in _DESCENDANT_ARRAY_KEYS:
            if not _byte_equal(
                artifact["descendants"][mode]["arrays"][name],
                replay["descendants"][mode]["arrays"][name],
            ):
                raise ValueError("observation descendant replay arrays do not match")


if "animal_id" in signature(derive_observation_seed_v2).parameters:
    raise RuntimeError("observation RNG must never accept animal_id")
