"""Finite-thickness pose-only curriculum from immutable v3 plane parents."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge
import training.arbitrary_plane_finite_slab_v4 as finite_slab
import training.arbitrary_plane_pose_curriculum_v3 as pose_v3
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
from training.arbitrary_plane_rendered_generator import (
    make_finite_arbitrary_plane_render_from_context,
)
from training.arbitrary_plane_synthetic_generator import (
    make_arbitrary_plane_synthetic_realization,
)
from training.arbitrary_plane_synthetic_ops import identity_pixel_map


FINITE_POSE_CURRICULUM_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-pose-curriculum/v4"
)
FINITE_POSE_CURRICULUM_V4_ALGORITHM = (
    "unconditioned-uniform-rp2-independent-finite-boxcar-identity-g1-"
    "varied-g2-g3-paired-outline/v4"
)
FINITE_POSE_NO_DROP_POLICY = (
    "one authenticated row per logical sample; every retry reuses the exact parent "
    "plane and finite slab; bounded exhaustion uses a fresh identity-G1 realization "
    "with information-gate bypass and never relabels a rejected image"
)
NO_DROP_FALLBACK_REASON = "bounded pose appearance/damage retries exhausted"
RETRYABLE_REJECTION_STAGES = {
    reason: stage
    for reason, stage in pose_v3.RETRYABLE_REJECTION_STAGES.items()
    if stage.startswith("synthetic-")
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_finite_pose_curriculum_v4.py",
    "training/arbitrary_plane_finite_slab_v4.py",
    "training/arbitrary_plane_psf_v4.py",
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


def _derived_seed(root_seed, sample_index, domain):
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/{domain}",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
            }
        )[:16],
        16,
    )


def finite_pose_plane_sample_index_v4(root_seed, sample_index):
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/plane-sample-index",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
            }
        )[:15],
        16,
    )


def _parent_request(root_seed, sample_index, identities):
    return {
        "logical_root_seed_uint64": _seed(root_seed),
        "logical_sample_index": int(sample_index),
        "derived_plane_sample_index": finite_pose_plane_sample_index_v4(
            root_seed, sample_index
        ),
        "finite_render_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, 'finite-render'):016x}"
        ),
        "lineage_ids": copy.deepcopy(identities),
    }


def _parent_identity(root_seed, sample_index, identities, parent):
    request = _parent_request(root_seed, sample_index, identities)
    identity = {
        **request,
        "finite_parent_root_seed_uint64": parent["root_seed"],
        "finite_parent_sample_index": int(parent["sample_index"]),
        "plane_realization_id": parent["plane_realization_id"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "finite_parent_provenance_sha256": parent["provenance_sha256"],
    }
    if (
        identity["finite_parent_root_seed_uint64"]
        != request["finite_render_seed_uint64"]
        or identity["finite_parent_sample_index"]
        != request["derived_plane_sample_index"]
        or any(
            parent["provenance"].get(name) != value
            for name, value in identities.items()
            if name in ("animal_id", "specimen_id", "experiment_id")
        )
    ):
        raise ValueError("finite parent differs from the logical pose-row request")
    return identity


def _slab_identity(root_seed, sample_index, parent_identity, slab_result):
    artifact = slab_result["artifact"]
    block = artifact["slab_observation_v4"]
    selection = block["thickness_selection"]
    finite_psf = block["finite_psf"]
    thickness_seed = _derived_seed(root_seed, sample_index, "finite-slab-thickness")
    identity = {
        "logical_root_seed_uint64": _seed(root_seed),
        "logical_sample_index": int(sample_index),
        "thickness_seed_uint64": f"0x{thickness_seed:016x}",
        "finite_parent_identity": copy.deepcopy(parent_identity),
        "slab_parent_sample_index": int(block["sample_index"]),
        "slab_observation_id": block["slab_observation_id"],
        "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
        "finite_slab_artifact_receipt_sha256": artifact["receipt_sha256"],
        "centre_plane_targets_receipt_sha256": block[
            "centre_plane_targets_receipt_sha256"
        ],
        "thickness_selection_sha256": selection["thickness_selection_sha256"],
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
        "finite_psf_capability_sha256": finite_psf[
            "finite_psf_capability_sha256"
        ],
        "render_mode": finite_psf["render_mode"],
        "nominal_cut_thickness_um": finite_psf["nominal_cut_thickness_um"],
    }
    if (
        artifact.get("receipt_sha256")
        != finite_slab._payload_sha256(
            finite_slab.finite_slab_render_receipt_v4(slab_result)
        )
        or artifact.get("provenance_sha256")
        != parent_identity["finite_parent_provenance_sha256"]
        or selection.get("selection_mode") != "independent-seeded-uniform"
        or selection.get("thickness_seed_uint64") != identity["thickness_seed_uint64"]
        or block.get("finite_plane_render_id")
        != parent_identity["finite_plane_render_id"]
        or block.get("finite_render_receipt_sha256")
        != parent_identity["finite_render_receipt_sha256"]
        or block.get("plane_realization_id")
        != parent_identity["plane_realization_id"]
        or identity["slab_parent_sample_index"]
        != parent_identity["derived_plane_sample_index"]
        or finite_psf.get("render_mode") != finite_slab.FINITE_BOXCAR
    ):
        raise ValueError("finite slab differs from its parent or independent seed")
    return identity


def _attempt_receipt(
    root_seed,
    sample_index,
    attempt_index,
    reason,
    *,
    parent_identity,
    slab_identity,
):
    return {
        "attempt_index": int(attempt_index),
        "appearance_damage_seed_uint64": (
            f"0x{_derived_seed(root_seed, sample_index, f'appearance-damage/attempt-{attempt_index}'):016x}"
        ),
        "error_type": "ValueError",
        "stage": RETRYABLE_REJECTION_STAGES[reason],
        "reason": reason,
        "finite_parent_identity": copy.deepcopy(parent_identity),
        "finite_slab_identity": copy.deepcopy(slab_identity),
    }


def _verified_rejection_history(
    root_seed,
    sample_index,
    attempt_index,
    history,
):
    history = copy.deepcopy(list(history or ()))
    if len(history) != int(attempt_index):
        raise ValueError("finite-pose rejection history length differs from attempt")
    for index, entry in enumerate(history):
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if reason not in RETRYABLE_REJECTION_STAGES or entry != _attempt_receipt(
            root_seed,
            sample_index,
            index,
            reason,
            parent_identity=entry.get("finite_parent_identity"),
            slab_identity=entry.get("finite_slab_identity"),
        ):
            raise ValueError("finite-pose rejection history is not canonical")
    return history


def _make_parent(
    prepared_context,
    *,
    root_seed,
    sample_index,
    output_shape_h_w,
    split,
    stratum,
    margin_um,
    minimum_brain_pixels,
    maximum_rejection_attempts,
    finite_parent_generator_source_commit,
    identities,
):
    plane_sample_index = finite_pose_plane_sample_index_v4(root_seed, sample_index)
    finite_seed = _derived_seed(root_seed, sample_index, "finite-render")
    parent = make_finite_arbitrary_plane_render_from_context(
        prepared_context,
        split,
        finite_seed,
        tuple(int(value) for value in output_shape_h_w),
        sample_index=plane_sample_index,
        stratum=stratum,
        margin_um=margin_um,
        animal_id=identities["animal_id"],
        specimen_id=identities["specimen_id"],
        experiment_id=identities["experiment_id"],
        max_rejection_attempts=int(maximum_rejection_attempts),
        minimum_brain_pixels=int(minimum_brain_pixels),
        generator_source_commit=finite_parent_generator_source_commit,
    )
    return parent, plane_sample_index, finite_seed


def _make_slab(
    prepared_context,
    parent,
    *,
    root_seed,
    sample_index,
    finite_slab_generator_source_commit,
    finite_parent_generator_source_commit,
):
    return finite_slab.make_finite_slab_render_v4(
        parent,
        prepared_context,
        render_mode=finite_slab.FINITE_BOXCAR,
        thickness_seed=_derived_seed(
            root_seed, sample_index, "finite-slab-thickness"
        ),
        generator_source_commit=finite_slab_generator_source_commit,
        parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def finite_pose_curriculum_generation_config_v4(
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
    maximum_pose_rejection_attempts=16,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
):
    return {
        "schema_version": FINITE_POSE_CURRICULUM_V4_SCHEMA,
        "algorithm": FINITE_POSE_CURRICULUM_V4_ALGORITHM,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": prepared_context["support_index"][
            "support_index_sha256"
        ],
        "plane_domain": "all brain-intersecting planes",
        "normal_measure": "normalized isotropic Gaussian canonically folded to Haar-uniform RP2",
        "offset_measure": "length-uniform over authenticated merged brain-intersection intervals",
        "pose_acceptance": "one pose draw per logical row; never redraw for raster or slab support",
        "finite_psf_capability": psf_v4.finite_psf_model_capability_v4(),
        "finite_psf_render_mode": finite_slab.FINITE_BOXCAR,
        "thickness_selection": "independent seeded uniform continuous 25-100 um",
        "thickness_seed_domain": (
            f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/finite-slab-thickness"
        ),
        "point_pose_evidence_metric": (
            "post-G1 sum(source_slab_brain_occupancy_float32)"
        ),
        "dense_deformation_supervision": "always zero for exact identity-G1 pose rows",
        "finite_pose_no_drop_policy": FINITE_POSE_NO_DROP_POLICY,
        "root_seed_uint64": _seed(root_seed),
        "start_index": int(start_index),
        "row_count": int(row_count),
        "output_shape_h_w": [int(value) for value in output_shape_h_w],
        "identity_prefix": str(identity_prefix),
        "sections_per_animal": int(sections_per_animal),
        "split": str(split),
        "stratum": str(stratum),
        "margin_u_v_um": np.broadcast_to(
            np.asarray(margin_um, dtype=float), (2,)
        ).tolist(),
        "minimum_brain_pixels": int(minimum_brain_pixels),
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "maximum_pose_rejection_attempts": int(maximum_pose_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def finite_pose_curriculum_generator_binding_v4(generation_config):
    return row_cache.make_generator_binding_v3(
        generator_ids=(FINITE_POSE_CURRICULUM_V4_ALGORITHM,),
        source_sha256=_source_sha256(),
        geometry_gauge_contract=deformation_gauge.direct_deformation_target_contract_v4(),
        generator_config=generation_config,
    )


def make_finite_pose_curriculum_training_row_v4(
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
    maximum_pose_rejection_attempts=16,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
    pose_attempt_number=0,
    pose_rejection_history=None,
    no_drop_fallback=False,
    finite_parent_identity=None,
    finite_slab_identity=None,
    _finite_parent_artifact=None,
    _finite_slab_result=None,
):
    if (
        selected_mode not in pose_v3.MODE_TO_OUTLINE
        or reflection_state not in training_row.REFLECTION_STATES
    ):
        raise ValueError("finite-pose mode or reflection state is invalid")
    identities = {
        "animal_id": animal_id,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "synthetic_animal_id": synthetic_animal_id,
        "section_id": section_id,
    }
    if any(value in (None, "") for value in identities.values()):
        raise ValueError("every finite-pose row requires complete lineage IDs")
    sample_index = int(sample_index)
    pose_attempt_number = int(pose_attempt_number)
    maximum_pose_rejection_attempts = int(maximum_pose_rejection_attempts)
    no_drop_fallback = bool(no_drop_fallback)
    if (
        maximum_pose_rejection_attempts <= 0
        or pose_attempt_number < 0
        or pose_attempt_number > maximum_pose_rejection_attempts
        or no_drop_fallback
        != (pose_attempt_number == maximum_pose_rejection_attempts)
    ):
        raise ValueError("finite-pose attempt/fallback state is invalid")
    history = _verified_rejection_history(
        root_seed, sample_index, pose_attempt_number, pose_rejection_history
    )
    if _finite_parent_artifact is None:
        parent, plane_sample_index, finite_seed = _make_parent(
            prepared_context,
            root_seed=root_seed,
            sample_index=sample_index,
            output_shape_h_w=output_shape_h_w,
            split=split,
            stratum=stratum,
            margin_um=margin_um,
            minimum_brain_pixels=minimum_brain_pixels,
            maximum_rejection_attempts=maximum_rejection_attempts,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
            identities=identities,
        )
    else:
        parent = _finite_parent_artifact
        plane_sample_index = finite_pose_plane_sample_index_v4(
            root_seed, sample_index
        )
        finite_seed = _derived_seed(root_seed, sample_index, "finite-render")
    actual_parent_identity = _parent_identity(
        root_seed, sample_index, identities, parent
    )
    if finite_parent_identity is not None and finite_parent_identity != actual_parent_identity:
        raise ValueError("finite parent identity changed across pose retries")
    if _finite_slab_result is None:
        slab_result = _make_slab(
            prepared_context,
            parent,
            root_seed=root_seed,
            sample_index=sample_index,
            finite_slab_generator_source_commit=finite_slab_generator_source_commit,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
    else:
        slab_result = _finite_slab_result
    actual_slab_identity = _slab_identity(
        root_seed, sample_index, actual_parent_identity, slab_result
    )
    if finite_slab_identity is not None and finite_slab_identity != actual_slab_identity:
        raise ValueError("finite slab identity changed across pose retries")
    if any(
        entry["finite_parent_identity"] != actual_parent_identity
        or entry["finite_slab_identity"] != actual_slab_identity
        for entry in history
    ):
        raise ValueError("finite parent or slab changed inside pose rejection history")
    block = slab_result["artifact"]["slab_observation_v4"]
    pre_g1_slab_mass = float(block["slab_effective_brain_pixel_mass"])
    requested_threshold = int(minimum_brain_pixels)
    slab_marginal = pre_g1_slab_mass < requested_threshold
    synthetic_seed = _derived_seed(
        root_seed, sample_index, f"appearance-damage/attempt-{pose_attempt_number}"
    )
    overrides = {"g1": {"identity_probability": 1.0}}
    if slab_marginal or no_drop_fallback:
        height, width = map(int, output_shape_h_w)
        overrides["ordinary_minimum_clean_brain_pixels_floor"] = height * width + 1
    support = pose_v3._mutable_support(prepared_context["support_index"])
    paired = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            slab_observation_v4=block,
            root_seed=synthetic_seed,
            sample_index=sample_index,
            outline_mode=outline_mode,
            config_overrides=overrides,
            lineage={**identities, "split": split},
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
        for mode, outline_mode in pose_v3.MODE_TO_OUTLINE.items()
    }
    if (
        len({value["paired_view_group_id"] for value in paired.values()}) != 1
        or any(
            value["slab_observation_v4_identity"]
            != paired[selected_mode]["slab_observation_v4_identity"]
            for value in paired.values()
        )
    ):
        raise RuntimeError("paired finite-pose modes do not share one slab realization")
    selected = paired[selected_mode]
    if selected["lineage"] != {**identities, "split": split}:
        raise RuntimeError("finite-pose synthetic lineage differs from its row lineage")
    source = selected["arrays"]
    height, width = source["model_input_image"].shape
    identity_xy = identity_pixel_map((height, width))
    if (
        not selected["g1"]["parameters"]["accepted_attempt"]["identity_path"]
        or not np.array_equal(source["source_to_fixed_map"], identity_xy)
        or np.any(source["velocity_xy_px"] != 0.0)
    ):
        raise RuntimeError("finite-pose curriculum G1 must be exact identity")
    paired_pose_mass = {
        mode: float(
            realization["support_supervision"][
                "point_pose_evidence_effective_brain_pixel_mass"
            ]
        )
        for mode, realization in paired.items()
    }
    if len(set(paired_pose_mass.values())) != 1:
        raise RuntimeError("paired finite-pose modes changed point-pose evidence mass")
    post_g1_pose_mass = paired_pose_mass[selected_mode]
    point_pose_identifiable = bool(post_g1_pose_mass >= requested_threshold)
    final_dense_mass = float(
        np.asarray(
            source["source_dense_correspondence_weight_float32"],
            dtype=np.float64,
        ).sum()
    )
    pullback_yx = np.ascontiguousarray(
        np.moveaxis(source["source_to_fixed_map"], 0, -1)[..., ::-1],
        dtype=np.float32,
    )
    velocity_yx = np.ascontiguousarray(
        -np.moveaxis(source["velocity_xy_px"], 0, -1)[..., ::-1],
        dtype=np.float32,
    )
    effective_pose = np.asarray(
        parent["geometry"]["effective_quicknii_ouv_ml_ap_dv"], dtype=np.float64
    ).reshape(3, 3)
    deformation_valid = np.asarray(source["source_map_domain_mask"], dtype=bool)
    gauge = deformation_gauge.certify_direct_deformation_target_v4(
        velocity_yx,
        pullback_yx,
        effective_pose,
        deformation_valid,
    )
    canonical = effective_pose.copy()
    horizontal = reflection_state == "horizontal"
    observed, affine, representation_index = pose_v3._reflection_geometry(
        canonical, (height, width), reflection_state
    )
    reflected_pullback = pose_v3._reflect(
        gauge["arrays"]["certified_pullback_map_yx_px_float32"], horizontal
    ).copy()
    reflected_velocity = pose_v3._reflect(
        gauge["arrays"]["target_pullback_stationary_velocity_yx_px_float32"],
        horizontal,
    ).copy()
    if horizontal:
        reflected_pullback[..., 1] = width - 1.0 - reflected_pullback[..., 1]
        reflected_velocity[..., 1] *= -1.0
    valid = source["source_valid_correspondence_mask"]
    tissue = source["source_clean_tissue_mask"]
    arrays = {
        "model_input_channels_float32": pose_v3._input_channels(
            selected, horizontal
        ),
        "source_label_ground_truth_canvas_int64": pose_v3._reflect(
            source["source_annotation"], horizontal
        ).astype(np.int64),
        "source_tissue_ground_truth_mask": pose_v3._reflect(tissue, horizontal),
        "target_ccf_coordinates_ap_dv_ml_um_float64": pose_v3._reflect(
            source["source_ccf_ap_dv_ml_um"], horizontal
        ).astype(np.float64),
        "target_valid_correspondence_mask": pose_v3._reflect(valid, horizontal),
        "target_correspondence_weight_float32": pose_v3._reflect(
            source["source_dense_correspondence_weight_float32"], horizontal
        ).astype(np.float32),
        "target_correspondence_abstention_mask": pose_v3._reflect(
            source["source_dense_correspondence_abstention_mask"], horizontal
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(
            reflected_pullback, dtype=np.float64
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            reflected_velocity, dtype=np.float64
        ),
        "truth_section_deformation_valid_mask": pose_v3._reflect(
            deformation_valid, horizontal
        ),
    }
    support_contract = {
        "continuous_plane_sample_retained": True,
        "pose_redrawn_for_raster_or_slab_support": False,
        "centre_plane_brain_pixel_count": int(
            block["centre_plane_brain_pixel_count"]
        ),
        "pre_g1_slab_effective_brain_pixel_mass": pre_g1_slab_mass,
        "post_g1_point_pose_evidence_effective_brain_pixel_mass": post_g1_pose_mass,
        "requested_identifiability_threshold_effective_pixels": requested_threshold,
        "point_pose_supervision_evidence_metric": (
            "post-G1 sum(source_slab_brain_occupancy_float32)"
        ),
        "point_pose_supervision_identifiable": point_pose_identifiable,
        "point_pose_supervision_weight": float(point_pose_identifiable),
        "post_g3_dense_effective_supervision_mass": final_dense_mass,
        "dense_deformation_supervision_identifiable": False,
        "dense_deformation_supervision_weight": 0.0,
        "dense_deformation_censor_reason": "exact identity-G1 pose-only curriculum",
        "marginal_observation_role": (
            "ordinary probabilistic point-pose supervision; dense identity target censored"
            if point_pose_identifiable
            else "retained low-evidence observation; point and dense losses censored"
        ),
    }
    paired_receipts = {
        mode: acquisition._array_receipt(
            pose_v3._input_channels(realization, horizontal)
        )
        for mode, realization in paired.items()
    }
    source_bundle = acquisition._payload_sha256(
        {
            "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/paired-source",
            "finite_render_receipt_sha256": parent[
                "finite_render_receipt_sha256"
            ],
            "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
            "finite_psf_sha256": block["finite_psf"]["finite_psf_sha256"],
            "paired_synthetic_receipts_sha256": {
                mode: realization["synthetic_receipt_sha256"]
                for mode, realization in paired.items()
            },
        }
    )
    transform_id = acquisition._payload_sha256(
        {
            "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/reflection-transform",
            "state": reflection_state,
            "canvas_width": width,
            "affine_xy": affine.tolist(),
        }
    )
    actual_parent_identity = copy.deepcopy(actual_parent_identity)
    actual_slab_identity = copy.deepcopy(actual_slab_identity)
    adapter_configuration = {
        "root_seed": _seed(root_seed),
        "sample_index": sample_index,
        "pose_attempt_number": pose_attempt_number,
        "pose_rejection_history": history,
        "maximum_pose_rejection_attempts": maximum_pose_rejection_attempts,
        "no_drop_fallback": no_drop_fallback,
        "finite_parent_identity": actual_parent_identity,
        "finite_slab_identity": actual_slab_identity,
        "output_shape_h_w": [height, width],
        "selected_mode": selected_mode,
        "reflection_state": reflection_state,
        **identities,
        "split": split,
        "stratum": stratum,
        "margin_um": np.broadcast_to(
            np.asarray(margin_um, dtype=float), (2,)
        ).tolist(),
        "minimum_brain_pixels": requested_threshold,
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
    }
    numeric = {
        "schema_version": FINITE_POSE_CURRICULUM_V4_SCHEMA,
        "root_seed_uint64": _seed(root_seed),
        "sample_index": sample_index,
        "pose_attempt_number": pose_attempt_number,
        "derived_plane_sample_index": plane_sample_index,
        "finite_render_seed_uint64": f"0x{finite_seed:016x}",
        "finite_slab_thickness_seed_uint64": actual_slab_identity[
            "thickness_seed_uint64"
        ],
        "appearance_damage_seed_uint64": f"0x{synthetic_seed:016x}",
        "thickness_selection_sha256": actual_slab_identity[
            "thickness_selection_sha256"
        ],
        "finite_psf_sha256": actual_slab_identity["finite_psf_sha256"],
        "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
        "no_drop_fallback": no_drop_fallback,
    }
    finite_reference = {
        **actual_slab_identity,
        "finite_slab_schema_version": slab_result["artifact"]["schema_version"],
        "finite_slab_algorithm": slab_result["artifact"]["algorithm"],
        "pre_g1_slab_observable_pixel_count": int(
            block["slab_observable_pixel_count"]
        ),
        "pre_g1_dense_effective_supervision_mass": float(
            block["dense_effective_supervision_mass"]
        ),
    }
    upstream = {
        "schema_version": FINITE_POSE_CURRICULUM_V4_SCHEMA,
        "algorithm": FINITE_POSE_CURRICULUM_V4_ALGORITHM,
        "implementation_source_sha256": _source_sha256(),
        "adapter_configuration": adapter_configuration,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": support["support_index_sha256"],
        "pose_rejection_history": history,
        "finite_parent_identity": actual_parent_identity,
        "finite_parent_generator_binding": copy.deepcopy(parent["generator"]),
        "finite_parent_provenance": copy.deepcopy(parent["provenance"]),
        "finite_parent_provenance_sha256": parent["provenance_sha256"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "finite_slab_reference": finite_reference,
        "slab_observation_id": block["slab_observation_id"],
        "centre_plane_targets_receipt_sha256": block[
            "centre_plane_targets_receipt_sha256"
        ],
        "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
        "finite_psf_sha256": block["finite_psf"]["finite_psf_sha256"],
        "finite_psf_capability_sha256": block["finite_psf"][
            "finite_psf_capability_sha256"
        ],
        "effective_pose_source_key": (
            "finite parent geometry effective_quicknii_ouv_ml_ap_dv"
        ),
        "effective_quicknii_ouv_ml_ap_dv": effective_pose.tolist(),
        "plane_sampling_measure": copy.deepcopy(parent["sampling_measure"]),
        "support_supervision_contract": support_contract,
        "selected_synthetic_receipt_sha256": selected[
            "synthetic_receipt_sha256"
        ],
        "selected_synthetic_generator_binding": copy.deepcopy(
            selected["generator"]
        ),
        "selected_synthetic_provenance_sha256": selected[
            "provenance_sha256"
        ],
        "selected_synthetic_lineage": copy.deepcopy(selected["lineage"]),
        "selected_synthetic_lineage_sha256": selected["lineage_sha256"],
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
        "no_drop_fallback": no_drop_fallback,
        "no_drop_fallback_reason": (
            NO_DROP_FALLBACK_REASON if no_drop_fallback else None
        ),
        "direct_deformation_target_certification_summary": (
            deformation_gauge.direct_deformation_target_summary_v4(gauge)
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
    }
    row_like = {
        "schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "source_observation_receipt_sha256": source_bundle,
        "lineage": {**identities, "split": split},
        "upstream_reference": upstream,
        "numeric_rng_provenance": numeric,
        "rng_sources": {
            "finite_render_accepted_attempt": parent["rejection_attempts"][
                parent["accepted_attempt_index"]
            ]["field_stream_seed_uint64"],
            "finite_slab_thickness_seed_uint64": actual_slab_identity[
                "thickness_seed_uint64"
            ],
            "synthetic_g1_accepted_attempt": selected["g1"]["parameters"][
                "accepted_attempt"
            ]["field_stream_seed_uint64"],
        },
        "selected_mode": selected_mode,
        "selected_descendant_id": selected["synthetic_realization_id"],
        "deformation_pose_gauge_reference": deformation_gauge.direct_deformation_target_reference_v4(
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
                "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/reflection-realization",
                "source_bundle": source_bundle,
                "transform_id": transform_id,
                "numeric_rng_provenance": numeric,
            }
        ),
        "paired_view_group_id": acquisition._payload_sha256(
            {
                "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/paired-view",
                "latent_group": selected["paired_view_group_id"],
                "slab_observation_id": block["slab_observation_id"],
                "transform_id": transform_id,
            }
        ),
        "paired_mode_reflected_receipts": paired_receipts,
        "arrays": arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in arrays.items()
        },
    }
    row_like["synthetic_realization_id"] = acquisition._payload_sha256(
        {
            "domain": f"{FINITE_POSE_CURRICULUM_V4_SCHEMA}/training-realization",
            "source_bundle": source_bundle,
            "selected_descendant_id": row_like["selected_descendant_id"],
            "reflection_realization_id": row_like["reflection_realization_id"],
            "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
            "finite_psf_sha256": block["finite_psf"]["finite_psf_sha256"],
        }
    )
    return psf_v4.finalize_training_row_v4(
        row_like,
        block,
        capability=psf_v4.finite_psf_model_capability_v4(),
    )


def replay_finite_pose_curriculum_training_row_v4(row, prepared_context):
    return make_finite_pose_curriculum_training_row_v4(
        prepared_context,
        **copy.deepcopy(row["upstream_reference"]["adapter_configuration"]),
    )


def verify_finite_pose_curriculum_training_row_v4(row, prepared_context):
    psf_v4.verify_training_row_v4(
        row, capability=psf_v4.finite_psf_model_capability_v4()
    )
    if row["upstream_reference"].get("implementation_source_sha256") != _source_sha256():
        raise ValueError("finite-pose curriculum implementation source binding changed")
    adapter = copy.deepcopy(row["upstream_reference"]["adapter_configuration"])
    history = adapter["pose_rejection_history"]
    for attempt_index, expected in enumerate(history):
        rejected = copy.deepcopy(adapter)
        rejected["pose_attempt_number"] = attempt_index
        rejected["pose_rejection_history"] = history[:attempt_index]
        rejected["no_drop_fallback"] = False
        try:
            make_finite_pose_curriculum_training_row_v4(
                prepared_context, **rejected
            )
        except ValueError as error:
            if str(error) != expected["reason"]:
                raise ValueError(
                    "finite-pose rejection history does not replay exactly"
                ) from error
        else:
            raise ValueError("finite-pose rejection history names an accepted attempt")
    replay = replay_finite_pose_curriculum_training_row_v4(row, prepared_context)
    if (
        set(row) != set(replay)
        or psf_v4.training_row_receipt_v4(row)
        != psf_v4.training_row_receipt_v4(replay)
        or any(
            np.asarray(row["arrays"][name]).dtype
            != np.asarray(replay["arrays"][name]).dtype
            or not np.array_equal(row["arrays"][name], replay["arrays"][name])
            for name in training_row._ARRAY_KEYS
        )
    ):
        raise ValueError("finite-pose curriculum row does not replay exactly")
    return True


def make_finite_pose_curriculum_training_rows_v4(
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
    maximum_pose_rejection_attempts=16,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
):
    if int(sections_per_animal) <= 0 or int(maximum_pose_rejection_attempts) <= 0:
        raise ValueError("finite-pose grouping and retry count must be positive")
    rows = []
    for offset in range(int(row_count)):
        sample_index = int(start_index) + offset
        animal_index = sample_index // int(sections_per_animal)
        identities = {
            "animal_id": f"{identity_prefix}-animal-{animal_index:08d}",
            "specimen_id": f"{identity_prefix}-specimen-{animal_index:08d}",
            "experiment_id": f"{identity_prefix}-experiment-{animal_index:08d}",
            "synthetic_animal_id": (
                f"{identity_prefix}-synthetic-animal-{animal_index:08d}"
            ),
            "section_id": f"{identity_prefix}-section-{sample_index:08d}",
        }
        parent, _, _ = _make_parent(
            prepared_context,
            root_seed=root_seed,
            sample_index=sample_index,
            output_shape_h_w=output_shape_h_w,
            split=split,
            stratum=stratum,
            margin_um=margin_um,
            minimum_brain_pixels=minimum_brain_pixels,
            maximum_rejection_attempts=maximum_rejection_attempts,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
            identities=identities,
        )
        parent_identity = _parent_identity(
            root_seed, sample_index, identities, parent
        )
        slab_result = _make_slab(
            prepared_context,
            parent,
            root_seed=root_seed,
            sample_index=sample_index,
            finite_slab_generator_source_commit=finite_slab_generator_source_commit,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
        slab_identity = _slab_identity(
            root_seed, sample_index, parent_identity, slab_result
        )
        common = {
            "root_seed": root_seed,
            "sample_index": sample_index,
            "output_shape_h_w": output_shape_h_w,
            "selected_mode": training_row.TRAINABLE_MODES[
                sample_index % len(training_row.TRAINABLE_MODES)
            ],
            "reflection_state": training_row.REFLECTION_STATES[
                (sample_index // len(training_row.TRAINABLE_MODES))
                % len(training_row.REFLECTION_STATES)
            ],
            **identities,
            "split": split,
            "stratum": stratum,
            "margin_um": margin_um,
            "minimum_brain_pixels": minimum_brain_pixels,
            "maximum_rejection_attempts": maximum_rejection_attempts,
            "maximum_pose_rejection_attempts": int(
                maximum_pose_rejection_attempts
            ),
            "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
            "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
            "finite_parent_identity": parent_identity,
            "finite_slab_identity": slab_identity,
            "_finite_parent_artifact": parent,
            "_finite_slab_result": slab_result,
        }
        history = []
        for attempt_index in range(int(maximum_pose_rejection_attempts)):
            try:
                row = make_finite_pose_curriculum_training_row_v4(
                    prepared_context,
                    **common,
                    pose_attempt_number=attempt_index,
                    pose_rejection_history=history,
                )
            except ValueError as error:
                reason = str(error)
                if reason not in RETRYABLE_REJECTION_STAGES:
                    raise
                history.append(
                    _attempt_receipt(
                        root_seed,
                        sample_index,
                        attempt_index,
                        reason,
                        parent_identity=parent_identity,
                        slab_identity=slab_identity,
                    )
                )
            else:
                rows.append(row)
                break
        else:
            rows.append(
                make_finite_pose_curriculum_training_row_v4(
                    prepared_context,
                    **common,
                    pose_attempt_number=int(maximum_pose_rejection_attempts),
                    pose_rejection_history=history,
                    no_drop_fallback=True,
                )
            )
    return rows


__all__ = [
    "FINITE_POSE_CURRICULUM_V4_ALGORITHM",
    "FINITE_POSE_CURRICULUM_V4_SCHEMA",
    "FINITE_POSE_NO_DROP_POLICY",
    "NO_DROP_FALLBACK_REASON",
    "RETRYABLE_REJECTION_STAGES",
    "finite_pose_curriculum_generation_config_v4",
    "finite_pose_curriculum_generator_binding_v4",
    "finite_pose_plane_sample_index_v4",
    "make_finite_pose_curriculum_training_row_v4",
    "make_finite_pose_curriculum_training_rows_v4",
    "replay_finite_pose_curriculum_training_row_v4",
    "verify_finite_pose_curriculum_training_row_v4",
]
