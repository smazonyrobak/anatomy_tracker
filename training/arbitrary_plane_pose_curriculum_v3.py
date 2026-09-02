"""Fast identity-deformation pose curriculum from the audited arbitrary-plane chain."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v3 as deformation_gauge
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
from training.arbitrary_plane_rendered_generator import (
    make_finite_arbitrary_plane_render_from_context,
)
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    make_arbitrary_plane_synthetic_realization,
)
from training.arbitrary_plane_synthetic_ops import identity_pixel_map


POSE_CURRICULUM_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-pose-curriculum/v3"
POSE_CURRICULUM_V3_ALGORITHM = (
    "unconditioned-uniform-rp2-finite-render-identity-g1-varied-g2-g3-paired-outline/v3"
)
PARENT_GEOMETRY_REJECTION = (
    "Finite-render accepted pose does not match installed geometry"
)
RETRYABLE_REJECTION_STAGES = {
    PARENT_GEOMETRY_REJECTION: "finite-parent-verification",
    "finite parent has too little tissue for a synthetic realization": "synthetic-g1",
    "ordinary synthetic stratum does not meet the predeclared clean-brain-pixel gate": "synthetic-g1",
    "no G1 realization passed every predeclared topology, cycle, displacement, and FOV gate": "synthetic-g1",
    "G2 realization failed all deterministic information-content rejection attempts": "synthetic-g2",
    "G3 realization failed all deterministic damage/visibility rejection attempts": "synthetic-g3",
    "imperfect outline did not meet its predeclared IoU gate": "synthetic-outline",
}
SINGLE_PLANE_RENDER_SCOPE = (
    "single centre-plane finite-FOV raster; no through-plane PSF integration"
)
MODE_TO_OUTLINE = {
    "smart-brush-accurate": ACCURATE_OUTLINE,
    "smart-brush-imperfect": IMPERFECT_OUTLINE,
    "smart-brush-absent": ABSENT_OUTLINE,
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_pose_curriculum_v3.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_synthetic_generator.py",
    "training/arbitrary_plane_synthetic_ops.py",
    "training/arbitrary_plane_synthetic_observation.py",
    "training/arbitrary_plane_deformation_gauge_v3.py",
    "training/arbitrary_plane_training_row_v3.py",
)


def _source_sha256():
    return {
        name: hashlib.sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _seed(value):
    if isinstance(value, str):
        parsed = int(value, 16)
    else:
        parsed = int(value)
    if not 0 <= parsed < 2**64:
        raise ValueError("root seed must be uint64")
    return f"0x{parsed:016x}"


def _derived_seed(root_seed, sample_index, domain):
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/{domain}",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
            }
        )[:16],
        16,
    )


def plane_attempt_index_v3(root_seed, sample_index, attempt_index):
    attempt_index = int(attempt_index)
    if attempt_index < 0:
        raise ValueError("plane attempt index must be nonnegative")
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/plane-attempt-index",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
                "attempt_index": attempt_index,
            }
        )[:15],
        16,
    )


def _verified_parent_rejection_history(root_seed, sample_index, attempt_index, history):
    history = copy.deepcopy(list(history or ()))
    if len(history) != int(attempt_index):
        raise ValueError("pose rejection history length must equal the accepted attempt index")
    for index, entry in enumerate(history):
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if reason not in RETRYABLE_REJECTION_STAGES or entry != _attempt_receipt(
            root_seed, sample_index, index, reason
        ):
            raise ValueError("pose rejection history is not canonical")
    return history


def _attempt_receipt(root_seed, sample_index, attempt_index, reason):
    return {
        "attempt_index": int(attempt_index),
        "derived_plane_sample_index": plane_attempt_index_v3(root_seed, sample_index, 0),
        "finite_render_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, 'finite-render/plane-attempt-0'):016x}"
        ),
        "appearance_damage_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, f'appearance-damage/plane-attempt-{attempt_index}'):016x}"
        ),
        "error_type": "ValueError",
        "stage": RETRYABLE_REJECTION_STAGES[reason],
        "reason": reason,
    }


def single_plane_curriculum_runner_config_v3(runner_config):
    """Return the exact runner PSF matching these zero-thickness direct rasters."""
    config = copy.deepcopy(dict(runner_config))
    config["axial_offsets_um"] = [0.0]
    config["axial_weights"] = [1.0]
    return config


def _reflect(value, horizontal):
    array = np.asarray(value)
    return np.ascontiguousarray(array[:, ::-1] if horizontal else array)


def _mutable_support(value):
    if isinstance(value, Mapping):
        return {key: _mutable_support(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_support(item) for item in value]
    return value


def _outline(mask):
    mask = np.asarray(mask, dtype=bool)
    eroded = mask.copy()
    eroded[1:] &= mask[:-1]
    eroded[:-1] &= mask[1:]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    eroded[[0, -1], :] = False
    eroded[:, [0, -1]] = False
    return np.ascontiguousarray(mask & ~eroded)


def _input_channels(realization, horizontal):
    arrays = realization["arrays"]
    image = _reflect(arrays["model_input_image"], horizontal).astype(np.float32)
    outline = _reflect(_outline(arrays["input_outline_mask"]), horizontal)
    available = float(realization["outline"]["parameters"]["outline_available"])
    return np.ascontiguousarray(
        np.stack((image, outline.astype(np.float32), np.full_like(image, available)), axis=-1),
        dtype=np.float32,
    )


def _reflection_geometry(canonical, shape_h_w, reflection_state):
    height, width = shape_h_w
    if reflection_state == "none":
        return canonical.copy(), np.eye(3, dtype=np.float64), 0
    affine = np.array(
        [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observed = canonical.copy()
    observed[0] = canonical[0] + ((width - 1.0) / width) * canonical[1]
    observed[1] = -canonical[1]
    return observed, affine, 1


def pose_curriculum_generation_config_v3(
    prepared_context,
    *,
    root_seed,
    start_index,
    row_count,
    output_shape_h_w,
    identity_prefix,
    sections_per_animal=4,
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=64,
    maximum_rejection_attempts=64,
    maximum_parent_geometry_retries=16,
    finite_parent_generator_source_commit=None,
):
    return {
        "schema_version": POSE_CURRICULUM_V3_SCHEMA,
        "algorithm": POSE_CURRICULUM_V3_ALGORITHM,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": prepared_context["support_index"]["support_index_sha256"],
        "plane_domain": "all brain-intersecting planes",
        "normal_measure": "normalized isotropic Gaussian canonically folded to Haar-uniform RP2",
        "offset_measure": "length-uniform over authenticated merged brain-intersection intervals",
        "pose_acceptance": (
            "exactly one pose draw per global logical sample; finite-raster support never redraws pose"
        ),
        "marginal_support_policy": (
            "retain every continuous brain-intersecting plane and bind point/dense supervision "
            "eligibility to explicit raster-support metadata"
        ),
        "render_thickness_scope": SINGLE_PLANE_RENDER_SCOPE,
        "required_runner_psf": {
            "axial_offsets_um": [0.0],
            "axial_weights": [1.0],
            "interpretation": "single centre-plane sample matching the direct curriculum raster",
        },
        "g1_deformation": "forced exact identity",
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
        "maximum_parent_geometry_retries": int(maximum_parent_geometry_retries),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def pose_curriculum_generator_binding_v3(generation_config):
    return row_cache.make_generator_binding_v3(
        generator_ids=(POSE_CURRICULUM_V3_ALGORITHM,),
        source_sha256=_source_sha256(),
        geometry_gauge_contract={
            "schema_version": deformation_gauge.DEFORMATION_GAUGE_V3_SCHEMA,
            "algorithm": deformation_gauge.DEFORMATION_GAUGE_V3_ALGORITHM,
            "projection_weighting": row_cache.DEFORMATION_GAUGE_PROJECTION_WEIGHTING,
        },
        generator_config=generation_config,
    )


def make_pose_curriculum_training_row_v3(
    prepared_context,
    *,
    root_seed,
    sample_index,
    output_shape_h_w,
    selected_mode,
    reflection_state,
    animal_id,
    specimen_id,
    experiment_id,
    synthetic_animal_id,
    section_id,
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=64,
    maximum_rejection_attempts=64,
    finite_parent_generator_source_commit=None,
    plane_attempt_number=0,
    plane_parent_rejection_history=None,
):
    """Generate one authenticated pose-only row from a preloaded atlas context."""
    if selected_mode not in MODE_TO_OUTLINE or reflection_state not in training_row.REFLECTION_STATES:
        raise ValueError("pose curriculum mode or reflection state is invalid")
    identities = {
        "animal_id": animal_id,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "synthetic_animal_id": synthetic_animal_id,
        "section_id": section_id,
    }
    if any(value in (None, "") for value in identities.values()):
        raise ValueError("every pose curriculum row requires complete lineage IDs")
    sample_index = int(sample_index)
    plane_attempt_number = int(plane_attempt_number)
    rejection_history = _verified_parent_rejection_history(
        root_seed,
        sample_index,
        plane_attempt_number,
        plane_parent_rejection_history,
    )
    plane_sample_index = plane_attempt_index_v3(root_seed, sample_index, 0)
    finite_seed = _derived_seed(
        root_seed, sample_index, "finite-render/plane-attempt-0"
    )
    synthetic_seed = _derived_seed(
        root_seed,
        sample_index,
        f"appearance-damage/plane-attempt-{plane_attempt_number}",
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
            "ordinary point/dense supervision"
            if support_identifiable
            else "retained censored observation; no unique point-pose or dense-deformation target"
        ),
    }
    support = _mutable_support(prepared_context["support_index"])
    paired = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            root_seed=synthetic_seed,
            sample_index=sample_index,
            outline_mode=outline_mode,
            config_overrides={"g1": {"identity_probability": 1.0}},
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
        for mode, outline_mode in MODE_TO_OUTLINE.items()
    }
    if len({value["paired_view_group_id"] for value in paired.values()}) != 1:
        raise RuntimeError("paired pose-curriculum modes do not share one latent realization")
    selected = paired[selected_mode]
    height, width = selected["arrays"]["model_input_image"].shape
    identity_xy = identity_pixel_map((height, width))
    fixed_to_source_xy = selected["arrays"]["fixed_to_source_map"]
    velocity_xy = selected["arrays"]["velocity_xy_px"]
    if (
        not selected["g1"]["parameters"]["accepted_attempt"]["identity_path"]
        or not np.array_equal(fixed_to_source_xy, identity_xy)
        or np.any(velocity_xy != 0.0)
    ):
        raise RuntimeError("pose curriculum G1 must be exact identity")
    pullback_yx = np.ascontiguousarray(
        np.moveaxis(fixed_to_source_xy, 0, -1)[..., ::-1], dtype=np.float64
    )
    velocity_yx = np.ascontiguousarray(
        np.moveaxis(velocity_xy, 0, -1)[..., ::-1], dtype=np.float64
    )
    effective_pose = np.asarray(
        parent["geometry"]["effective_quicknii_ouv_ml_ap_dv"], dtype=np.float64
    ).reshape(3, 3)
    deformation_valid = np.ones((height, width), dtype=bool)
    gauge = deformation_gauge.gauge_fix_canvas_deformation_v3(
        velocity_yx,
        pullback_yx,
        effective_pose,
        deformation_valid,
    )
    canonical = gauge["arrays"]["pose_adjusted_effective_quicknii_ouv_float64"]
    horizontal = reflection_state == "horizontal"
    observed, affine, representation_index = _reflection_geometry(
        canonical, (height, width), reflection_state
    )
    source = selected["arrays"]
    reflected_pullback = _reflect(
        gauge["arrays"]["affine_free_pullback_map_yx_px_float64"], horizontal
    ).copy()
    reflected_velocity = _reflect(
        gauge["arrays"]["affine_free_stationary_velocity_yx_px_float64"], horizontal
    ).copy()
    if horizontal:
        reflected_pullback[..., 1] = width - 1.0 - reflected_pullback[..., 1]
        reflected_velocity[..., 1] *= -1.0
    valid = source["source_valid_correspondence_mask"]
    tissue = source["source_clean_tissue_mask"]
    arrays = {
        "model_input_channels_float32": _input_channels(selected, horizontal),
        "source_label_ground_truth_canvas_int64": _reflect(
            source["source_annotation"], horizontal
        ).astype(np.int64),
        "source_tissue_ground_truth_mask": _reflect(tissue, horizontal),
        "target_ccf_coordinates_ap_dv_ml_um_float64": _reflect(
            source["source_ccf_ap_dv_ml_um"], horizontal
        ).astype(np.float64),
        "target_valid_correspondence_mask": _reflect(valid, horizontal),
        "target_correspondence_weight_float32": _reflect(
            valid.astype(np.float32), horizontal
        ),
        "target_correspondence_abstention_mask": _reflect(
            tissue & ~valid, horizontal
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(
            reflected_pullback
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            reflected_velocity
        ),
        "truth_section_deformation_valid_mask": _reflect(
            deformation_valid, horizontal
        ),
    }
    paired_receipts = {
        mode: acquisition._array_receipt(_input_channels(realization, horizontal))
        for mode, realization in paired.items()
    }
    source_bundle = acquisition._payload_sha256(
        {
            "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/paired-source",
            "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
            "paired_synthetic_receipts_sha256": {
                mode: realization["synthetic_receipt_sha256"]
                for mode, realization in paired.items()
            },
        }
    )
    transform_id = acquisition._payload_sha256(
        {
            "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/reflection-transform",
            "state": reflection_state,
            "canvas_width": width,
            "affine_xy": affine.tolist(),
        }
    )
    adapter_configuration = {
        "root_seed": _seed(root_seed),
        "sample_index": sample_index,
        "plane_attempt_number": plane_attempt_number,
        "plane_parent_rejection_history": rejection_history,
        "output_shape_h_w": [height, width],
        "selected_mode": selected_mode,
        "reflection_state": reflection_state,
        **identities,
        "split": split,
        "stratum": stratum,
        "margin_um": np.broadcast_to(np.asarray(margin_um, dtype=float), (2,)).tolist(),
        "minimum_brain_pixels": int(minimum_brain_pixels),
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
    }
    numeric = {
        "schema_version": POSE_CURRICULUM_V3_SCHEMA,
        "root_seed_uint64": _seed(root_seed),
        "sample_index": sample_index,
        "plane_attempt_number": plane_attempt_number,
        "derived_plane_sample_index": plane_sample_index,
        "finite_render_seed_uint64": f"0x{finite_seed:016x}",
        "appearance_damage_seed_uint64": f"0x{synthetic_seed:016x}",
    }
    artifact = {
        "schema_version": training_row.TRAINING_ROW_V3_SCHEMA,
        "source_observation_receipt_sha256": source_bundle,
        "lineage": {**identities, "split": split},
        "upstream_reference": {
            "schema_version": POSE_CURRICULUM_V3_SCHEMA,
            "algorithm": POSE_CURRICULUM_V3_ALGORITHM,
            "implementation_source_sha256": _source_sha256(),
            "adapter_configuration": adapter_configuration,
            "prepared_context_sha256": prepared_context["prepared_context_sha256"],
            "support_index_sha256": support["support_index_sha256"],
            "plane_parent_rejection_history": rejection_history,
            "finite_parent_generator_binding": copy.deepcopy(parent["generator"]),
            "finite_parent_provenance": copy.deepcopy(parent["provenance"]),
            "finite_parent_provenance_sha256": parent["provenance_sha256"],
            "finite_plane_render_id": parent["finite_plane_render_id"],
            "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
            "effective_pose_source_key": "parent['geometry']['effective_quicknii_ouv_ml_ap_dv']",
            "effective_quicknii_ouv_ml_ap_dv": effective_pose.tolist(),
            "plane_sampling_measure": copy.deepcopy(parent["sampling_measure"]),
            "render_thickness_scope": SINGLE_PLANE_RENDER_SCOPE,
            "brain_pixel_count": parent["acceptance_contract"]["brain_pixel_count"],
            "support_supervision_contract": support_supervision_contract,
            "selected_synthetic_receipt_sha256": selected["synthetic_receipt_sha256"],
            "selected_synthetic_generator_binding": copy.deepcopy(
                selected["generator"]
            ),
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
            "g1_identity_forced": True,
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
            "synthetic_g1_accepted_attempt": selected["g1"]["parameters"][
                "accepted_attempt"
            ]["field_stream_seed_uint64"],
        },
        "selected_mode": selected_mode,
        "selected_descendant_id": selected["synthetic_realization_id"],
        "deformation_pose_gauge_reference": deformation_gauge.deformation_pose_gauge_reference_v3(
            gauge
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
                "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/reflection-realization",
                "source_bundle": source_bundle,
                "transform_id": transform_id,
                "numeric_rng_provenance": numeric,
            }
        ),
        "paired_view_group_id": acquisition._payload_sha256(
            {
                "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/paired-view",
                "latent_group": selected["paired_view_group_id"],
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
            "domain": f"{POSE_CURRICULUM_V3_SCHEMA}/training-realization",
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


def replay_pose_curriculum_training_row_v3(row, prepared_context):
    config = copy.deepcopy(row["upstream_reference"]["adapter_configuration"])
    return make_pose_curriculum_training_row_v3(prepared_context, **config)


def verify_pose_curriculum_training_row_v3(row, prepared_context):
    row_cache.verify_cached_training_row_v3(row)
    if row["upstream_reference"].get("implementation_source_sha256") != _source_sha256():
        raise ValueError("pose curriculum implementation source binding changed")
    adapter = copy.deepcopy(row["upstream_reference"]["adapter_configuration"])
    history = adapter["plane_parent_rejection_history"]
    for attempt_index, expected in enumerate(history):
        rejected = copy.deepcopy(adapter)
        rejected["plane_attempt_number"] = attempt_index
        rejected["plane_parent_rejection_history"] = history[:attempt_index]
        try:
            make_pose_curriculum_training_row_v3(prepared_context, **rejected)
        except ValueError as error:
            if str(error) != expected["reason"]:
                raise ValueError("pose rejection history does not replay exactly") from error
        else:
            raise ValueError("pose rejection history names an accepted attempt")
    replay = replay_pose_curriculum_training_row_v3(row, prepared_context)
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
        raise ValueError("pose curriculum row does not replay exactly")
    return True


def make_pose_curriculum_training_rows_v3(
    prepared_context,
    *,
    root_seed,
    start_index,
    row_count,
    output_shape_h_w,
    identity_prefix,
    sections_per_animal=4,
    split="development",
    stratum="reference",
    margin_um=(0.0, 0.0),
    minimum_brain_pixels=64,
    maximum_rejection_attempts=64,
    maximum_parent_geometry_retries=16,
    finite_parent_generator_source_commit=None,
):
    """Deterministically cycle all modes and horizontal representations."""
    rows = []
    maximum_parent_geometry_retries = int(maximum_parent_geometry_retries)
    if int(sections_per_animal) <= 0 or maximum_parent_geometry_retries <= 0:
        raise ValueError("sections per animal and parent retry count must be positive")
    for offset in range(int(row_count)):
        sample_index = int(start_index) + offset
        animal_index = sample_index // int(sections_per_animal)
        rejection_history = []
        for plane_attempt_number in range(maximum_parent_geometry_retries):
            try:
                row = make_pose_curriculum_training_row_v3(
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
                    plane_attempt_number=plane_attempt_number,
                    plane_parent_rejection_history=rejection_history,
                )
            except ValueError as error:
                reason = str(error)
                if reason not in RETRYABLE_REJECTION_STAGES:
                    raise
                rejection_history.append(
                    _attempt_receipt(
                        root_seed,
                        sample_index,
                        plane_attempt_number,
                        reason,
                    )
                )
            else:
                rows.append(row)
                break
        else:
            raise RuntimeError(
                f"no verified finite parent after {maximum_parent_geometry_retries} "
                f"deterministic attempts for logical sample {sample_index}: {rejection_history}"
            )
    return rows


__all__ = [
    "MODE_TO_OUTLINE",
    "POSE_CURRICULUM_V3_ALGORITHM",
    "POSE_CURRICULUM_V3_SCHEMA",
    "PARENT_GEOMETRY_REJECTION",
    "RETRYABLE_REJECTION_STAGES",
    "SINGLE_PLANE_RENDER_SCOPE",
    "make_pose_curriculum_training_row_v3",
    "make_pose_curriculum_training_rows_v3",
    "pose_curriculum_generation_config_v3",
    "pose_curriculum_generator_binding_v3",
    "plane_attempt_index_v3",
    "replay_pose_curriculum_training_row_v3",
    "single_plane_curriculum_runner_config_v3",
    "verify_pose_curriculum_training_row_v3",
]
