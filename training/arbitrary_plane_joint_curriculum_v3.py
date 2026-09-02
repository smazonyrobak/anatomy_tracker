"""Direct nonidentity pose/deformation curriculum from audited arbitrary planes."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v4 as direct_deformation_target
import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
from training.arbitrary_plane_rendered_generator import (
    make_finite_arbitrary_plane_render_from_context,
)
from training.arbitrary_plane_synthetic_generator import (
    make_arbitrary_plane_synthetic_realization,
)
from training.arbitrary_plane_synthetic_ops import (
    FIXED_SEVEN_DECODER_INTEGRATION,
    UNIFORM_CANVAS_AFFINE_PROJECTION,
)


JOINT_CURRICULUM_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-joint-curriculum/v4"
JOINT_CURRICULUM_V4_ALGORITHM = (
    "unconditioned-uniform-rp2-direct-preintegration-affine-free-source-to-fixed-g1-varied-g2-g3/v4"
)
# Compatibility names keep the existing runner API while every emitted artifact
# explicitly declares the v4 schema and algorithm above.
JOINT_CURRICULUM_V3_SCHEMA = JOINT_CURRICULUM_V4_SCHEMA
JOINT_CURRICULUM_V3_ALGORITHM = JOINT_CURRICULUM_V4_ALGORITHM
LEGACY_V3_RNG_DOMAIN = "anatomy-tracker.arbitrary-plane-joint-curriculum/v3"
COMPOSITE_CURRICULUM_V3_SCHEMA = (
    "anatomy-tracker.pose-and-joint-curriculum-cache-config/v3"
)
COMPOSITE_ROW_ORDER_POLICY = (
    "append authenticated pose rows then authenticated joint rows"
)
DEFORMATION_AMPLITUDE_BANDS = {
    "mild": (0.0025, 0.0050),
    "moderate": (0.0050, 0.0080),
}
JOINT_G1_FIXED_OVERRIDES = {
    "identity_probability": 0.0,
    "analytic_probability": 0.0,
    "similarity_angle_rad": (0.0, 0.0),
    "similarity_scale": (1.0, 1.0),
    "similarity_translation_over_D": (0.0, 0.0),
    "affine_projection_contract": UNIFORM_CANVAS_AFFINE_PROJECTION,
    "integration_contract": FIXED_SEVEN_DECODER_INTEGRATION,
}
GAUGE_RECOMPOSITION_REJECTION = (
    "affine-gauge pose/deformation recomposition exceeds the production bound"
)
ZERO_AFFINE_FREE_REJECTION = "affine-free deformation is zero after gauge projection"
RETRYABLE_REJECTION_STAGES = {
    **pose_curriculum.RETRYABLE_REJECTION_STAGES,
    GAUGE_RECOMPOSITION_REJECTION: "deformation-gauge",
    ZERO_AFFINE_FREE_REJECTION: "deformation-gauge",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_joint_curriculum_v3.py",
    "training/arbitrary_plane_pose_curriculum_v3.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_synthetic_generator.py",
    "training/arbitrary_plane_synthetic_ops.py",
    "training/arbitrary_plane_synthetic_observation.py",
    "training/arbitrary_plane_deformation_gauge_v4.py",
    "training/arbitrary_plane_deformation_primitives.py",
    "training/arbitrary_plane_training_row_v3.py",
)


def _source_sha256():
    return {
        name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _seed(value):
    parsed = int(value, 16) if isinstance(value, str) else int(value)
    if not 0 <= parsed < 2**64:
        raise ValueError("root seed must be uint64")
    return f"0x{parsed:016x}"


def _derived_seed(root_seed, sample_index, attempt_index, domain):
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{LEGACY_V3_RNG_DOMAIN}/{domain}",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
                "attempt_index": int(attempt_index),
            }
        )[:16],
        16,
    )


def joint_attempt_index_v3(root_seed, sample_index, attempt_index):
    attempt_index = int(attempt_index)
    if attempt_index < 0:
        raise ValueError("joint attempt index must be nonnegative")
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{LEGACY_V3_RNG_DOMAIN}/plane-attempt-index",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
                "attempt_index": attempt_index,
            }
        )[:15],
        16,
    )


def _attempt_receipt(root_seed, sample_index, attempt_index, reason):
    return {
        "attempt_index": int(attempt_index),
        "derived_plane_sample_index": joint_attempt_index_v3(root_seed, sample_index, 0),
        "finite_render_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, 0, 'finite-render'):016x}"
        ),
        "synthetic_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, attempt_index, 'synthetic'):016x}"
        ),
        "error_type": "ValueError",
        "stage": RETRYABLE_REJECTION_STAGES[reason],
        "reason": reason,
    }


def _verified_rejection_history(root_seed, sample_index, attempt_index, history):
    history = copy.deepcopy(list(history or ()))
    if len(history) != int(attempt_index):
        raise ValueError("joint rejection history length must equal accepted attempt index")
    for index, entry in enumerate(history):
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if reason not in RETRYABLE_REJECTION_STAGES or entry != _attempt_receipt(
            root_seed, sample_index, index, reason
        ):
            raise ValueError("joint rejection history is not canonical")
    return history


def joint_g1_overrides_v3(amplitude_band):
    if amplitude_band not in DEFORMATION_AMPLITUDE_BANDS:
        raise ValueError("unknown joint deformation amplitude band")
    return {
        **{key: list(value) if isinstance(value, tuple) else value for key, value in JOINT_G1_FIXED_OVERRIDES.items()},
        "target_rms_displacement_over_D": list(
            DEFORMATION_AMPLITUDE_BANDS[amplitude_band]
        ),
    }


def joint_curriculum_generation_config_v3(
    prepared_context,
    *,
    root_seed,
    start_index,
    row_count,
    output_shape_h_w,
    identity_prefix,
    sections_per_animal=4,
    amplitude_band_cycle=("mild", "moderate"),
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=320,
    maximum_rejection_attempts=64,
    maximum_joint_rejection_attempts=16,
    finite_parent_generator_source_commit=None,
):
    if not amplitude_band_cycle or any(
        name not in DEFORMATION_AMPLITUDE_BANDS for name in amplitude_band_cycle
    ):
        raise ValueError("joint amplitude-band cycle is invalid")
    return {
        "schema_version": JOINT_CURRICULUM_V3_SCHEMA,
        "algorithm": JOINT_CURRICULUM_V3_ALGORITHM,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": prepared_context["support_index"]["support_index_sha256"],
        "plane_domain": "all brain-intersecting planes",
        "normal_measure": "normalized isotropic Gaussian canonically folded to Haar-uniform RP2",
        "offset_measure": "length-uniform over authenticated merged brain-intersection intervals",
        "pose_acceptance": (
            "exactly one pose draw per global logical sample; finite-raster support never redraws pose"
        ),
        "marginal_support_policy": (
            "retain every continuous brain-intersecting plane; marginal rasters use an explicit "
            "identity/censored path with zero point-pose and dense-deformation loss weight"
        ),
        "render_thickness_scope": (
            pose_curriculum.SINGLE_PLANE_RENDER_SCOPE
        ),
        "required_runner_psf": {
            "axial_offsets_um": [0.0],
            "axial_weights": [1.0],
            "interpretation": "single centre-plane sample matching the direct curriculum raster",
        },
        "deformation_amplitude_bands_target_rms_over_section_D": {
            name: list(bounds) for name, bounds in DEFORMATION_AMPLITUDE_BANDS.items()
        },
        "amplitude_band_cycle": list(amplitude_band_cycle),
        "g1_fixed_overrides": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in JOINT_G1_FIXED_OVERRIDES.items()
        },
        "identifiability": (
            "no sampled G1 similarity; G1 is projected into the decoder's uniform-full-canvas "
            "affine-free gauge before integration; parent pose is unchanged"
        ),
        "direct_deformation_target_contract": (
            direct_deformation_target.direct_deformation_target_contract_v4()
        ),
        "rng_domain_compatibility": LEGACY_V3_RNG_DOMAIN,
        "appearance_damage": "audited varied G2/G3",
        "trainable_modes": list(training_row.TRAINABLE_MODES),
        "horizontal_representation_augmentation": list(training_row.REFLECTION_STATES),
        "root_seed_uint64": _seed(root_seed),
        "start_index": int(start_index),
        "row_count": int(row_count),
        "output_shape_h_w": [int(value) for value in output_shape_h_w],
        "identity_prefix": str(identity_prefix),
        "sections_per_animal": int(sections_per_animal),
        "split": str(split),
        "stratum": str(stratum),
        "margin_u_v_um": np.broadcast_to(np.asarray(margin_um, dtype=float), (2,)).tolist(),
        "minimum_brain_pixels": int(minimum_brain_pixels),
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "maximum_joint_rejection_attempts": int(maximum_joint_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "maximum_direct_target_certification_error_px": (
            direct_deformation_target.MAXIMUM_CERTIFICATION_ERROR_PX
        ),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def joint_curriculum_generator_binding_v3(generation_config):
    return row_cache.make_generator_binding_v3(
        generator_ids=(JOINT_CURRICULUM_V3_ALGORITHM,),
        source_sha256=_source_sha256(),
        geometry_gauge_contract=(
            direct_deformation_target.direct_deformation_target_contract_v4()
        ),
        generator_config=generation_config,
    )


def composite_curriculum_generation_config_v3(
    pose_generation_config,
    joint_generation_config,
    *,
    row_order_policy=COMPOSITE_ROW_ORDER_POLICY,
):
    pose_config = copy.deepcopy(pose_generation_config)
    joint_config = copy.deepcopy(joint_generation_config)
    pose_binding = pose_curriculum.pose_curriculum_generator_binding_v3(pose_config)
    joint_binding = joint_curriculum_generator_binding_v3(joint_config)
    if (
        pose_config.get("schema_version") != pose_curriculum.POSE_CURRICULUM_V3_SCHEMA
        or pose_config.get("algorithm") != pose_curriculum.POSE_CURRICULUM_V3_ALGORITHM
        or joint_config.get("schema_version") != JOINT_CURRICULUM_V3_SCHEMA
        or joint_config.get("algorithm") != JOINT_CURRICULUM_V3_ALGORITHM
        or pose_config.get("prepared_context_sha256")
        != joint_config.get("prepared_context_sha256")
        or pose_config.get("support_index_sha256")
        != joint_config.get("support_index_sha256")
        or pose_config.get("required_runner_psf")
        != joint_config.get("required_runner_psf")
        or row_order_policy != COMPOSITE_ROW_ORDER_POLICY
        or any(
            isinstance(config.get("row_count"), bool)
            or not isinstance(config.get("row_count"), int)
            or config["row_count"] < 1
            for config in (pose_config, joint_config)
        )
        or any(
            config.get(name) != []
            for config in (pose_config, joint_config)
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("pose/joint cache configs are incompatible or learned-dependent")
    return {
        "schema_version": COMPOSITE_CURRICULUM_V3_SCHEMA,
        "generator_ids": [
            pose_curriculum.POSE_CURRICULUM_V3_ALGORITHM,
            JOINT_CURRICULUM_V3_ALGORITHM,
        ],
        "prepared_context_sha256": pose_config["prepared_context_sha256"],
        "support_index_sha256": pose_config["support_index_sha256"],
        "component_generation_configs": {
            "identity_pose_curriculum": pose_config,
            "nonidentity_joint_curriculum": joint_config,
        },
        "component_generator_bindings": {
            "identity_pose_curriculum": pose_binding,
            "nonidentity_joint_curriculum": joint_binding,
        },
        "component_row_counts": {
            "identity_pose_curriculum": int(pose_config["row_count"]),
            "nonidentity_joint_curriculum": int(joint_config["row_count"]),
        },
        "row_order_policy": COMPOSITE_ROW_ORDER_POLICY,
        "single_frozen_cache": True,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def composite_curriculum_generator_binding_v3(composite_generation_config):
    config = copy.deepcopy(composite_generation_config)
    if config.get("schema_version") != COMPOSITE_CURRICULUM_V3_SCHEMA:
        raise ValueError("composite curriculum config schema is invalid")
    components = config.get("component_generation_configs", {})
    declared_bindings = config.get("component_generator_bindings", {})
    if set(components) != {
        "identity_pose_curriculum",
        "nonidentity_joint_curriculum",
    } or set(declared_bindings) != set(components):
        raise ValueError("composite curriculum config lacks exact component configs")
    bindings = (
        pose_curriculum.pose_curriculum_generator_binding_v3(
            components["identity_pose_curriculum"]
        ),
        joint_curriculum_generator_binding_v3(
            components["nonidentity_joint_curriculum"]
        ),
    )
    if declared_bindings != {
        "identity_pose_curriculum": bindings[0],
        "nonidentity_joint_curriculum": bindings[1],
    }:
        raise ValueError("composite curriculum component source/config bindings changed")
    source_sha256 = {}
    for binding in bindings:
        for name, digest in binding["source_sha256"].items():
            if name in source_sha256 and source_sha256[name] != digest:
                raise ValueError("composite curriculum source hashes disagree")
            source_sha256[name] = digest
    return row_cache.make_generator_binding_v3(
        generator_ids=(
            pose_curriculum.POSE_CURRICULUM_V3_ALGORITHM,
            JOINT_CURRICULUM_V3_ALGORITHM,
        ),
        source_sha256=source_sha256,
        geometry_gauge_contract=(
            direct_deformation_target.direct_deformation_target_contract_v4()
        ),
        generator_config=config,
    )


def make_joint_curriculum_training_row_v3(
    prepared_context,
    *,
    root_seed,
    sample_index,
    output_shape_h_w,
    selected_mode,
    reflection_state,
    amplitude_band,
    animal_id,
    specimen_id,
    experiment_id,
    synthetic_animal_id,
    section_id,
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=320,
    maximum_rejection_attempts=64,
    finite_parent_generator_source_commit=None,
    joint_attempt_number=0,
    joint_rejection_history=None,
):
    if selected_mode not in pose_curriculum.MODE_TO_OUTLINE:
        raise ValueError("joint curriculum mode is invalid")
    if reflection_state not in training_row.REFLECTION_STATES:
        raise ValueError("joint curriculum reflection state is invalid")
    overrides = joint_g1_overrides_v3(amplitude_band)
    identities = {
        "animal_id": animal_id,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "synthetic_animal_id": synthetic_animal_id,
        "section_id": section_id,
    }
    if any(value in (None, "") for value in identities.values()):
        raise ValueError("every joint curriculum row requires complete lineage IDs")
    sample_index = int(sample_index)
    joint_attempt_number = int(joint_attempt_number)
    rejection_history = _verified_rejection_history(
        root_seed,
        sample_index,
        joint_attempt_number,
        joint_rejection_history,
    )
    plane_sample_index = joint_attempt_index_v3(root_seed, sample_index, 0)
    finite_seed = _derived_seed(
        root_seed, sample_index, 0, "finite-render"
    )
    synthetic_seed = _derived_seed(
        root_seed, sample_index, joint_attempt_number, "synthetic"
    )
    parent = make_finite_arbitrary_plane_render_from_context(
        prepared_context,
        split,
        finite_seed,
        tuple(int(value) for value in output_shape_h_w),
        sample_index=plane_sample_index,
        stratum=stratum,
        margin_um=margin_um,
        animal_id=animal_id,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        max_rejection_attempts=int(maximum_rejection_attempts),
        minimum_brain_pixels=int(minimum_brain_pixels),
        generator_source_commit=finite_parent_generator_source_commit,
    )
    brain_pixel_count = int(parent["acceptance_contract"]["brain_pixel_count"])
    support_identifiable = bool(brain_pixel_count >= int(minimum_brain_pixels))
    support_supervision_contract = {
        "continuous_plane_sample_retained": True,
        "pose_redrawn_for_raster_support": False,
        "raster_brain_pixel_count": brain_pixel_count,
        "requested_identifiability_threshold_pixels": int(minimum_brain_pixels),
        "point_pose_supervision_identifiable": support_identifiable,
        "point_pose_supervision_weight": float(support_identifiable),
        "dense_deformation_supervision_identifiable": support_identifiable,
        "dense_deformation_supervision_weight": float(support_identifiable),
        "marginal_observation_role": (
            "ordinary point-pose plus nonrigid supervision"
            if support_identifiable
            else "retained censored observation; identity realization only and no unique point/dense target"
        ),
    }
    if not support_identifiable:
        overrides = {
            **overrides,
            "identity_probability": 1.0,
        }
    support = pose_curriculum._mutable_support(prepared_context["support_index"])
    paired = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            root_seed=synthetic_seed,
            sample_index=sample_index,
            outline_mode=outline_mode,
            config_overrides={"g1": overrides},
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
        for mode, outline_mode in pose_curriculum.MODE_TO_OUTLINE.items()
    }
    if len({value["paired_view_group_id"] for value in paired.values()}) != 1:
        raise RuntimeError("paired joint-curriculum modes do not share one latent realization")
    selected = paired[selected_mode]
    accepted_g1 = selected["g1"]["parameters"]["accepted_attempt"]
    similarity = accepted_g1["similarity"]
    if (
        accepted_g1["identity_path"] == support_identifiable
        or similarity["angle_rad"] != 0.0
        or similarity["scale"] != 1.0
        or similarity["translation_xy_px"] != [0.0, 0.0]
        or accepted_g1["affine_projection_contract"]
        != UNIFORM_CANVAS_AFFINE_PROJECTION
        or accepted_g1["integration_contract"]
        != FIXED_SEVEN_DECODER_INTEGRATION
        or accepted_g1["integration_steps"] != 7
        or accepted_g1[
            "identity_similarity_inverse_composition_error_max_abs_px"
        ]
        != 0.0
    ):
        raise RuntimeError(
            "joint curriculum requires direct affine-free fixed-seven G1 with identity similarity"
        )
    source = selected["arrays"]
    velocity_xy = source["velocity_xy_px"]
    source_to_fixed_xy = source["source_to_fixed_map"]
    if support_identifiable and not np.any(velocity_xy != 0.0):
        raise ValueError(ZERO_AFFINE_FREE_REJECTION)
    target_pullback_velocity_yx = np.ascontiguousarray(
        -np.moveaxis(velocity_xy, 0, -1)[..., ::-1], dtype=np.float32
    )
    pullback_yx = np.ascontiguousarray(
        np.moveaxis(source_to_fixed_xy, 0, -1)[..., ::-1], dtype=np.float32
    )
    effective_pose = np.asarray(
        parent["geometry"]["effective_quicknii_ouv_ml_ap_dv"], dtype=np.float64
    ).reshape(3, 3)
    deformation_valid = np.asarray(source["source_map_domain_mask"], dtype=bool)
    direct_target = direct_deformation_target.certify_direct_deformation_target_v4(
        target_pullback_velocity_yx,
        pullback_yx,
        effective_pose,
        deformation_valid,
    )
    affine_free_velocity = direct_target["arrays"][
        "target_pullback_stationary_velocity_yx_px_float32"
    ]
    if support_identifiable and not np.any(affine_free_velocity != 0.0):
        raise ValueError(ZERO_AFFINE_FREE_REJECTION)
    height, width = source["model_input_image"].shape
    canonical = effective_pose.copy()
    horizontal = reflection_state == "horizontal"
    observed, affine, representation_index = pose_curriculum._reflection_geometry(
        canonical, (height, width), reflection_state
    )
    reflected_pullback = pose_curriculum._reflect(
        direct_target["arrays"]["certified_pullback_map_yx_px_float32"], horizontal
    ).copy()
    reflected_velocity = pose_curriculum._reflect(
        affine_free_velocity, horizontal
    ).copy()
    if horizontal:
        reflected_pullback[..., 1] = width - 1.0 - reflected_pullback[..., 1]
        reflected_velocity[..., 1] *= -1.0
    valid = source["source_valid_correspondence_mask"]
    tissue = source["source_clean_tissue_mask"]
    arrays = {
        "model_input_channels_float32": pose_curriculum._input_channels(
            selected, horizontal
        ),
        "source_label_ground_truth_canvas_int64": pose_curriculum._reflect(
            source["source_annotation"], horizontal
        ).astype(np.int64),
        "source_tissue_ground_truth_mask": pose_curriculum._reflect(
            tissue, horizontal
        ),
        "target_ccf_coordinates_ap_dv_ml_um_float64": pose_curriculum._reflect(
            source["source_ccf_ap_dv_ml_um"], horizontal
        ).astype(np.float64),
        "target_valid_correspondence_mask": pose_curriculum._reflect(
            valid, horizontal
        ),
        "target_correspondence_weight_float32": pose_curriculum._reflect(
            valid.astype(np.float32), horizontal
        ),
        "target_correspondence_abstention_mask": pose_curriculum._reflect(
            tissue & ~valid, horizontal
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(
            reflected_pullback, dtype=np.float64
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            reflected_velocity, dtype=np.float64
        ),
        "truth_section_deformation_valid_mask": pose_curriculum._reflect(
            deformation_valid, horizontal
        ),
    }
    paired_receipts = {
        mode: acquisition._array_receipt(
            pose_curriculum._input_channels(realization, horizontal)
        )
        for mode, realization in paired.items()
    }
    source_bundle = acquisition._payload_sha256(
        {
            "domain": f"{JOINT_CURRICULUM_V3_SCHEMA}/paired-source",
            "amplitude_band": amplitude_band,
            "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
            "paired_synthetic_receipts_sha256": {
                mode: realization["synthetic_receipt_sha256"]
                for mode, realization in paired.items()
            },
            "direct_deformation_target_id": direct_target[
                "direct_deformation_target_id"
            ],
        }
    )
    transform_id = acquisition._payload_sha256(
        {
            "domain": f"{JOINT_CURRICULUM_V3_SCHEMA}/reflection-transform",
            "state": reflection_state,
            "canvas_width": width,
            "affine_xy": affine.tolist(),
        }
    )
    adapter_configuration = {
        "root_seed": _seed(root_seed),
        "sample_index": sample_index,
        "joint_attempt_number": joint_attempt_number,
        "joint_rejection_history": rejection_history,
        "output_shape_h_w": [height, width],
        "selected_mode": selected_mode,
        "reflection_state": reflection_state,
        "amplitude_band": amplitude_band,
        **identities,
        "split": split,
        "stratum": stratum,
        "margin_um": np.broadcast_to(np.asarray(margin_um, dtype=float), (2,)).tolist(),
        "minimum_brain_pixels": int(minimum_brain_pixels),
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
    }
    numeric = {
        "schema_version": JOINT_CURRICULUM_V3_SCHEMA,
        "root_seed_uint64": _seed(root_seed),
        "sample_index": sample_index,
        "joint_attempt_number": joint_attempt_number,
        "derived_plane_sample_index": plane_sample_index,
        "finite_render_seed_uint64": f"0x{finite_seed:016x}",
        "synthetic_seed_uint64": f"0x{synthetic_seed:016x}",
    }
    artifact = {
        "schema_version": training_row.TRAINING_ROW_V3_SCHEMA,
        "source_observation_receipt_sha256": source_bundle,
        "lineage": {**identities, "split": split},
        "upstream_reference": {
            "schema_version": JOINT_CURRICULUM_V3_SCHEMA,
            "algorithm": JOINT_CURRICULUM_V3_ALGORITHM,
            "implementation_source_sha256": _source_sha256(),
            "adapter_configuration": adapter_configuration,
            "prepared_context_sha256": prepared_context["prepared_context_sha256"],
            "support_index_sha256": support["support_index_sha256"],
            "joint_rejection_history": rejection_history,
            "deformation_amplitude_band": amplitude_band,
            "target_rms_displacement_over_section_D": list(
                DEFORMATION_AMPLITUDE_BANDS[amplitude_band]
            ),
            "g1_overrides": overrides,
            "finite_parent_generator_binding": copy.deepcopy(parent["generator"]),
            "finite_parent_provenance": copy.deepcopy(parent["provenance"]),
            "finite_parent_provenance_sha256": parent["provenance_sha256"],
            "finite_plane_render_id": parent["finite_plane_render_id"],
            "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
            "effective_pose_source_key": "parent['geometry']['effective_quicknii_ouv_ml_ap_dv']",
            "effective_quicknii_ouv_ml_ap_dv_before_gauge": effective_pose.tolist(),
            "effective_quicknii_ouv_ml_ap_dv_after_gauge": canonical.tolist(),
            "plane_sampling_measure": copy.deepcopy(parent["sampling_measure"]),
            "render_thickness_scope": (
                "single centre-plane finite-FOV raster; no through-plane PSF integration"
            ),
            "brain_pixel_count": parent["acceptance_contract"]["brain_pixel_count"],
            "support_supervision_contract": support_supervision_contract,
            "selected_synthetic_receipt_sha256": selected["synthetic_receipt_sha256"],
            "selected_synthetic_generator_binding": copy.deepcopy(selected["generator"]),
            "selected_synthetic_provenance_sha256": selected["provenance_sha256"],
            "paired_synthetic_receipts_sha256": {
                mode: realization["synthetic_receipt_sha256"]
                for mode, realization in paired.items()
            },
            "selected_stage_realization_ids": {
                "g1": selected["g1"]["deformation_realization_id"],
                "g2": selected["g2"]["appearance_realization_id"],
                "g3": selected["g3"]["damage_realization_id"],
                "outline": selected["outline"]["outline_realization_id"],
            },
            "selected_g1_accepted_attempt": copy.deepcopy(accepted_g1),
            "g1_nonidentity_forced": support_identifiable,
            "marginal_support_identity_forced": not support_identifiable,
            "sampled_similarity_forced_identity": True,
            "direct_deformation_target_certification_summary": (
                direct_deformation_target.direct_deformation_target_summary_v4(
                    direct_target
                )
            ),
            "selected_input_mask_receipt": acquisition._array_receipt(
                source["input_outline_mask"]
            ),
            "selected_black_exterior_exact": selected["outline"]["parameters"][
                "black_exterior_exact"
            ],
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
        },
        "numeric_rng_provenance": numeric,
        "rng_sources": {
            "finite_render_accepted_attempt": parent["rejection_attempts"][
                parent["accepted_attempt_index"]
            ]["field_stream_seed_uint64"],
            "synthetic_g1_accepted_attempt": accepted_g1["field_stream_seed_uint64"],
        },
        "selected_mode": selected_mode,
        "selected_descendant_id": selected["synthetic_realization_id"],
        "deformation_pose_gauge_reference": direct_deformation_target.direct_deformation_target_reference_v4(
            direct_target
        ),
        "reflection_state": reflection_state,
        "reflection_representation_index": representation_index,
        "reflection_representation_affine_xy_float64": affine.tolist(),
        "canonical_effective_quicknii_ouv_float64": canonical.tolist(),
        "observed_effective_quicknii_ouv_float64": observed.tolist(),
        "proper_physical_pose_unchanged": canonical.tolist(),
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "reflection_transform_id": transform_id,
        "reflection_realization_id": acquisition._payload_sha256(
            {
                "domain": f"{JOINT_CURRICULUM_V3_SCHEMA}/reflection-realization",
                "source_bundle": source_bundle,
                "transform_id": transform_id,
                "numeric_rng_provenance": numeric,
            }
        ),
        "paired_view_group_id": acquisition._payload_sha256(
            {
                "domain": f"{JOINT_CURRICULUM_V3_SCHEMA}/paired-view",
                "latent_group": selected["paired_view_group_id"],
                "direct_deformation_target_id": direct_target[
                    "direct_deformation_target_id"
                ],
                "transform_id": transform_id,
            }
        ),
        "paired_mode_reflected_receipts": paired_receipts,
        "arrays": arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value) for name, value in arrays.items()
        },
    }
    artifact["synthetic_realization_id"] = acquisition._payload_sha256(
        {
            "domain": f"{JOINT_CURRICULUM_V3_SCHEMA}/training-realization",
            "source_bundle": source_bundle,
            "selected_descendant_id": artifact["selected_descendant_id"],
            "reflection_realization_id": artifact["reflection_realization_id"],
        }
    )
    artifact["training_row_id"] = acquisition._payload_sha256(
        {
            "domain": training_row.TRAINING_ROW_V3_SCHEMA,
            "synthetic_realization_id": artifact["synthetic_realization_id"],
            "array_receipts": artifact["array_receipts"],
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        training_row.training_row_receipt_v3(artifact)
    )
    row_cache.verify_cached_training_row_v3(artifact)
    return artifact


def replay_joint_curriculum_training_row_v3(row, prepared_context):
    config = copy.deepcopy(row["upstream_reference"]["adapter_configuration"])
    return make_joint_curriculum_training_row_v3(prepared_context, **config)


def verify_joint_curriculum_training_row_v3(row, prepared_context):
    row_cache.verify_cached_training_row_v3(row)
    if row["upstream_reference"].get("implementation_source_sha256") != _source_sha256():
        raise ValueError("joint curriculum implementation source binding changed")
    adapter = copy.deepcopy(row["upstream_reference"]["adapter_configuration"])
    history = adapter["joint_rejection_history"]
    for attempt_index, expected in enumerate(history):
        rejected = copy.deepcopy(adapter)
        rejected["joint_attempt_number"] = attempt_index
        rejected["joint_rejection_history"] = history[:attempt_index]
        try:
            make_joint_curriculum_training_row_v3(prepared_context, **rejected)
        except ValueError as error:
            if str(error) != expected["reason"]:
                raise ValueError("joint rejection history does not replay exactly") from error
        else:
            raise ValueError("joint rejection history names an accepted attempt")
    replay = replay_joint_curriculum_training_row_v3(row, prepared_context)
    if (
        set(row) != set(replay)
        or training_row.training_row_receipt_v3(row)
        != training_row.training_row_receipt_v3(replay)
        or any(
            np.asarray(row["arrays"][name]).dtype
            != np.asarray(replay["arrays"][name]).dtype
            or not np.array_equal(row["arrays"][name], replay["arrays"][name])
            for name in training_row._ARRAY_KEYS
        )
    ):
        raise ValueError("joint curriculum row does not replay exactly")
    return True


def make_joint_curriculum_training_rows_v3(
    prepared_context,
    *,
    root_seed,
    start_index,
    row_count,
    output_shape_h_w,
    identity_prefix,
    sections_per_animal=4,
    amplitude_band_cycle=("mild", "moderate"),
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=320,
    maximum_rejection_attempts=64,
    maximum_joint_rejection_attempts=16,
    finite_parent_generator_source_commit=None,
):
    if (
        int(sections_per_animal) <= 0
        or int(maximum_joint_rejection_attempts) <= 0
        or not amplitude_band_cycle
        or any(name not in DEFORMATION_AMPLITUDE_BANDS for name in amplitude_band_cycle)
    ):
        raise ValueError("joint batch grouping, amplitude cycle, or retry count is invalid")
    rows = []
    for offset in range(int(row_count)):
        sample_index = int(start_index) + offset
        animal_index = sample_index // int(sections_per_animal)
        amplitude_band = amplitude_band_cycle[
            sample_index % len(amplitude_band_cycle)
        ]
        rejection_history = []
        for joint_attempt_number in range(int(maximum_joint_rejection_attempts)):
            try:
                row = make_joint_curriculum_training_row_v3(
                    prepared_context,
                    root_seed=root_seed,
                    sample_index=sample_index,
                    output_shape_h_w=output_shape_h_w,
                    selected_mode=training_row.TRAINABLE_MODES[
                        sample_index % len(training_row.TRAINABLE_MODES)
                    ],
                    reflection_state=training_row.REFLECTION_STATES[
                        (sample_index // len(training_row.TRAINABLE_MODES))
                        % len(training_row.REFLECTION_STATES)
                    ],
                    amplitude_band=amplitude_band,
                    animal_id=f"{identity_prefix}-animal-{animal_index:08d}",
                    specimen_id=f"{identity_prefix}-specimen-{animal_index:08d}",
                    experiment_id=f"{identity_prefix}-experiment-{animal_index:08d}",
                    synthetic_animal_id=f"{identity_prefix}-synthetic-animal-{animal_index:08d}",
                    section_id=f"{identity_prefix}-section-{sample_index:08d}",
                    split=split,
                    stratum=stratum,
                    margin_um=margin_um,
                    minimum_brain_pixels=minimum_brain_pixels,
                    maximum_rejection_attempts=maximum_rejection_attempts,
                    finite_parent_generator_source_commit=finite_parent_generator_source_commit,
                    joint_attempt_number=joint_attempt_number,
                    joint_rejection_history=rejection_history,
                )
            except ValueError as error:
                reason = str(error)
                if reason not in RETRYABLE_REJECTION_STAGES:
                    raise
                rejection_history.append(
                    _attempt_receipt(
                        root_seed,
                        sample_index,
                        joint_attempt_number,
                        reason,
                    )
                )
            else:
                rows.append(row)
                break
        else:
            raise RuntimeError(
                f"no verified joint row after {maximum_joint_rejection_attempts} "
                f"deterministic attempts for logical sample {sample_index}: {rejection_history}"
            )
    return rows


joint_attempt_index_v4 = joint_attempt_index_v3
joint_g1_overrides_v4 = joint_g1_overrides_v3
joint_curriculum_generation_config_v4 = joint_curriculum_generation_config_v3
joint_curriculum_generator_binding_v4 = joint_curriculum_generator_binding_v3
make_joint_curriculum_training_row_v4 = make_joint_curriculum_training_row_v3
make_joint_curriculum_training_rows_v4 = make_joint_curriculum_training_rows_v3
replay_joint_curriculum_training_row_v4 = replay_joint_curriculum_training_row_v3
verify_joint_curriculum_training_row_v4 = verify_joint_curriculum_training_row_v3


__all__ = [
    "COMPOSITE_CURRICULUM_V3_SCHEMA",
    "DEFORMATION_AMPLITUDE_BANDS",
    "JOINT_CURRICULUM_V3_ALGORITHM",
    "JOINT_CURRICULUM_V3_SCHEMA",
    "JOINT_CURRICULUM_V4_ALGORITHM",
    "JOINT_CURRICULUM_V4_SCHEMA",
    "JOINT_G1_FIXED_OVERRIDES",
    "RETRYABLE_REJECTION_STAGES",
    "joint_attempt_index_v3",
    "composite_curriculum_generation_config_v3",
    "composite_curriculum_generator_binding_v3",
    "joint_curriculum_generation_config_v3",
    "joint_curriculum_generator_binding_v3",
    "joint_g1_overrides_v3",
    "make_joint_curriculum_training_row_v3",
    "make_joint_curriculum_training_rows_v3",
    "replay_joint_curriculum_training_row_v3",
    "verify_joint_curriculum_training_row_v3",
    "joint_attempt_index_v4",
    "joint_g1_overrides_v4",
    "joint_curriculum_generation_config_v4",
    "joint_curriculum_generator_binding_v4",
    "make_joint_curriculum_training_row_v4",
    "make_joint_curriculum_training_rows_v4",
    "replay_joint_curriculum_training_row_v4",
    "verify_joint_curriculum_training_row_v4",
]
