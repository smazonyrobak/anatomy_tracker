"""Finite-thickness joint pose/deformation curriculum.

This is a new namespaced path.  It does not alter or alias the frozen pose-v3
or direct-joint-v4 entry points.  Every logical sample owns exactly one
arbitrary-plane parent and one independently sampled finite slab.  Those two
artifacts are reused across brush modes, bounded nonidentity attempts, and the
fresh identity-G1 no-drop fallback.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v4 as direct_target
import training.arbitrary_plane_finite_slab_v4 as finite_slab
import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_training_row_v3 as training_row_v3
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


FINITE_JOINT_CURRICULUM_V5_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-joint-curriculum/v5"
)
FINITE_JOINT_CURRICULUM_V5_ALGORITHM = (
    "one-parent-one-slab-paired-brush-direct-affine-free-joint-curriculum/v5"
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
NONIDENTITY_RETRY_EXHAUSTION_CENSOR_REASON = (
    "bounded nonidentity G1 realization retries exhausted"
)
MARGINAL_CENTRE_GAUGE_CENSOR_REASON = (
    "post-G1 centre/gauge support is below the identifiability threshold"
)
ZERO_POST_G3_DENSE_MASS_CENSOR_REASON = (
    "post-G3 dense correspondence weight has zero effective mass"
)
UNCENSORED_DEFORMATION_STATUS = "uncensored-direct-nonidentity-finite-slab"
IDENTITY_FALLBACK_CENSOR_STATUS = (
    "censored-to-fresh-identity-g1-after-bounded-nonidentity-retries"
)
MARGINAL_CENTRE_GAUGE_CENSOR_STATUS = "censored-marginal-centre-gauge-support"
ZERO_POST_G3_DENSE_MASS_CENSOR_STATUS = "censored-zero-post-g3-dense-mass"
FINITE_JOINT_NO_DROP_POLICY = (
    "one authenticated row per logical sample; all modes and retries reuse the exact "
    "finite parent and independently sampled slab; rejected images are never relabelled; "
    "retry exhaustion generates a fresh identity-G1 pose-only realization"
)
RETRYABLE_REJECTION_STAGES = {
    **pose_curriculum.RETRYABLE_REJECTION_STAGES,
    GAUGE_RECOMPOSITION_REJECTION: "deformation-gauge",
    ZERO_AFFINE_FREE_REJECTION: "deformation-gauge",
}
INTEGRATION_DEPENDENCIES = {
    "cache": "v4 cache/composite integration is intentionally out of scope",
    "batch": "consumer must accept authenticated training-row/v4 and its per-row finite_psf_contract",
    "runner": "runner must use authenticated per-row schedules; no global fixed schedule",
    "model": "checkpoint binds finite-PSF capability, while each row/session binds an exact schedule",
}

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "training/arbitrary_plane_finite_joint_curriculum_v5.py",
    "training/arbitrary_plane_finite_slab_v4.py",
    "training/arbitrary_plane_psf_v4.py",
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
                "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/{domain}",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
                "attempt_index": int(attempt_index),
            }
        )[:16],
        16,
    )


def finite_joint_plane_sample_index_v5(root_seed, sample_index):
    return int(
        acquisition._payload_sha256(
            {
                "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/plane-sample-index",
                "root_seed_uint64": _seed(root_seed),
                "sample_index": int(sample_index),
            }
        )[:15],
        16,
    )


def finite_joint_g1_overrides_v5(amplitude_band):
    if amplitude_band not in DEFORMATION_AMPLITUDE_BANDS:
        raise ValueError("unknown finite-joint deformation amplitude band")
    return {
        **{
            key: list(value) if isinstance(value, tuple) else value
            for key, value in JOINT_G1_FIXED_OVERRIDES.items()
        },
        "target_rms_displacement_over_D": list(
            DEFORMATION_AMPLITUDE_BANDS[amplitude_band]
        ),
    }


def _parent_request(root_seed, sample_index, identities):
    return {
        "logical_root_seed_uint64": _seed(root_seed),
        "logical_sample_index": int(sample_index),
        "derived_plane_sample_index": finite_joint_plane_sample_index_v5(
            root_seed, sample_index
        ),
        "finite_render_seed_uint64": _seed(
            _derived_seed(root_seed, sample_index, 0, "finite-render")
        ),
        "lineage_ids": copy.deepcopy(identities),
    }


def _parent_identity(root_seed, sample_index, identities, parent):
    request = _parent_request(root_seed, sample_index, identities)
    actual = {
        **request,
        "finite_parent_root_seed_uint64": parent["root_seed"],
        "finite_parent_sample_index": int(parent["sample_index"]),
        "plane_realization_id": parent["plane_realization_id"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "finite_parent_provenance_sha256": parent["provenance_sha256"],
    }
    if (
        actual["finite_parent_root_seed_uint64"]
        != request["finite_render_seed_uint64"]
        or actual["finite_parent_sample_index"]
        != request["derived_plane_sample_index"]
        or any(
            parent["provenance"].get(name) != identities[name]
            for name in ("animal_id", "specimen_id", "experiment_id")
        )
    ):
        raise ValueError("finite-joint parent differs from its logical request")
    return actual


def _slab_identity(adapter_result, parent_identity, thickness_seed_uint64):
    artifact = adapter_result["artifact"]
    block = artifact["slab_observation_v4"]
    actual = {
        "independent_thickness_seed_uint64": thickness_seed_uint64,
        "finite_slab_adapter_receipt_sha256": artifact["receipt_sha256"],
        "slab_observation_v4_receipt_sha256": block["receipt_sha256"],
        "slab_observation_id": block["slab_observation_id"],
        "centre_plane_targets_receipt_sha256": block[
            "centre_plane_targets_receipt_sha256"
        ],
        "finite_psf_sha256": block["finite_psf"]["finite_psf_sha256"],
        "finite_psf_capability_sha256": block["finite_psf"][
            "finite_psf_capability_sha256"
        ],
        "thickness_selection": copy.deepcopy(block["thickness_selection"]),
        "finite_parent_identity": copy.deepcopy(parent_identity),
    }
    if (
        block["finite_plane_render_id"]
        != parent_identity["finite_plane_render_id"]
        or block["finite_render_receipt_sha256"]
        != parent_identity["finite_render_receipt_sha256"]
        or block["plane_realization_id"] != parent_identity["plane_realization_id"]
    ):
        raise ValueError("finite slab changed the authenticated parent pose")
    return actual


def _make_parent_and_slab(
    prepared_context,
    *,
    root_seed,
    sample_index,
    output_shape_h_w,
    identities,
    split,
    stratum,
    margin_um,
    minimum_brain_pixels,
    maximum_rejection_attempts,
    render_mode,
    nominal_cut_thickness_um,
    finite_parent_generator_source_commit,
    finite_slab_generator_source_commit,
):
    plane_sample_index = finite_joint_plane_sample_index_v5(root_seed, sample_index)
    finite_seed = _derived_seed(root_seed, sample_index, 0, "finite-render")
    thickness_seed = _derived_seed(root_seed, sample_index, 0, "finite-thickness")
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
    slab_kwargs = {}
    if render_mode == finite_slab.FINITE_BOXCAR:
        slab_kwargs = (
            {"thickness_seed": thickness_seed}
            if nominal_cut_thickness_um is None
            else {"nominal_cut_thickness_um": float(nominal_cut_thickness_um)}
        )
    elif render_mode != finite_slab.CENTRE_PLANE_ABLATION:
        raise ValueError("finite-joint render mode is unsupported")
    adapter = finite_slab.make_finite_slab_render_v4(
        parent,
        prepared_context,
        render_mode=render_mode,
        generator_source_commit=finite_slab_generator_source_commit,
        parent_generator_source_commit=finite_parent_generator_source_commit,
        **slab_kwargs,
    )
    return parent, adapter, plane_sample_index, finite_seed, thickness_seed


def _attempt_receipt(
    root_seed,
    sample_index,
    attempt_index,
    reason,
    *,
    amplitude_band,
    identities,
    parent_identity,
    slab_identity,
):
    return {
        "attempt_index": int(attempt_index),
        "synthetic_seed_uint64": _seed(
            _derived_seed(root_seed, sample_index, attempt_index, "synthetic")
        ),
        "error_type": "ValueError",
        "stage": RETRYABLE_REJECTION_STAGES[reason],
        "reason": reason,
        "requested_deformation_amplitude_band": amplitude_band,
        "finite_parent_request": _parent_request(root_seed, sample_index, identities),
        "finite_parent_identity": copy.deepcopy(parent_identity),
        "finite_slab_identity": copy.deepcopy(slab_identity),
    }


def _verified_rejection_history(
    root_seed,
    sample_index,
    attempt_index,
    history,
    *,
    amplitude_band,
    identities,
    parent_identity,
    slab_identity,
):
    history = copy.deepcopy(list(history or ()))
    if len(history) != int(attempt_index):
        raise ValueError("finite-joint rejection history length changed")
    for index, entry in enumerate(history):
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if reason not in RETRYABLE_REJECTION_STAGES or entry != _attempt_receipt(
            root_seed,
            sample_index,
            index,
            reason,
            amplitude_band=amplitude_band,
            identities=identities,
            parent_identity=parent_identity,
            slab_identity=slab_identity,
        ):
            raise ValueError("finite-joint rejection history is not canonical")
    return history


def _supervision_evidence(paired, minimum_brain_pixels, identity_fallback):
    modes = tuple(pose_curriculum.MODE_TO_OUTLINE)
    post_g1_mass = {
        mode: float(
            paired[mode]["support_supervision"][
                "point_pose_evidence_effective_brain_pixel_mass"
            ]
        )
        for mode in modes
    }
    recomputed_post_g1_mass = {
        mode: float(
            np.asarray(
                paired[mode]["arrays"]["source_slab_brain_occupancy_float32"],
                dtype=np.float64,
            ).sum()
        )
        for mode in modes
    }
    center_gauge_masks = {
        mode: np.asarray(
            paired[mode]["arrays"]["source_map_domain_mask"], dtype=bool
        )
        & np.asarray(
            paired[mode]["arrays"]["source_clean_tissue_mask"], dtype=bool
        )
        for mode in modes
    }
    center_gauge_counts = {
        mode: int(center_gauge_masks[mode].sum()) for mode in modes
    }
    dense_masses = {
        mode: float(
            np.asarray(
                paired[mode]["arrays"][
                    "source_dense_correspondence_weight_float32"
                ],
                dtype=np.float64,
            ).sum()
        )
        for mode in modes
    }
    dense_valid_counts = {
        mode: int(
            (
                np.asarray(
                    paired[mode]["arrays"][
                        "source_dense_correspondence_weight_float32"
                    ]
                )
                > 0.0
            ).sum()
        )
        for mode in modes
    }
    if (
        post_g1_mass != recomputed_post_g1_mass
        or len(set(post_g1_mass.values())) != 1
        or len(set(center_gauge_counts.values())) != 1
        or len(set(dense_masses.values())) != 1
        or len(set(dense_valid_counts.values())) != 1
        or any(
            not np.array_equal(
                paired[modes[0]]["arrays"][name],
                paired[mode]["arrays"][name],
            )
            for mode in modes[1:]
            for name in (
                "source_slab_brain_occupancy_float32",
                "source_dense_correspondence_weight_float32",
                "source_dense_correspondence_abstention_mask",
            )
        )
    ):
        raise RuntimeError("paired brush modes changed pose or dense supervision evidence")
    post_mass = post_g1_mass[modes[0]]
    center_count = center_gauge_counts[modes[0]]
    dense_mass = dense_masses[modes[0]]
    point_identifiable = bool(post_mass >= int(minimum_brain_pixels))
    center_identifiable = bool(center_count >= int(minimum_brain_pixels))
    dense_identifiable = bool(
        center_identifiable and dense_mass > 0.0 and not identity_fallback
    )
    return {
        "point_pose_metric": "post-G1 sum(source_slab_brain_occupancy_float32)",
        "post_g1_slab_effective_brain_pixel_mass": post_mass,
        "paired_post_g1_slab_effective_brain_pixel_mass": post_g1_mass,
        "point_pose_supervision_identifiable": point_identifiable,
        "point_pose_supervision_weight": float(point_identifiable),
        "center_gauge_support_metric": (
            "post-G1 source_map_domain_mask & source_clean_tissue_mask"
        ),
        "center_gauge_support_pixel_count": center_count,
        "center_gauge_support_identifiable": center_identifiable,
        "post_g3_dense_correspondence_weight_mass": dense_mass,
        "post_g3_dense_positive_weight_pixel_count": dense_valid_counts[modes[0]],
        "dense_deformation_supervision_identifiable": dense_identifiable,
        "dense_deformation_supervision_weight": float(dense_identifiable),
        "identity_g1_pose_only_fallback": bool(identity_fallback),
    }


def finite_joint_curriculum_generation_config_v5(
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
    render_mode=finite_slab.FINITE_BOXCAR,
    nominal_cut_thickness_um=None,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
):
    if not amplitude_band_cycle or any(
        band not in DEFORMATION_AMPLITUDE_BANDS for band in amplitude_band_cycle
    ):
        raise ValueError("finite-joint amplitude cycle is invalid")
    return {
        "schema_version": FINITE_JOINT_CURRICULUM_V5_SCHEMA,
        "algorithm": FINITE_JOINT_CURRICULUM_V5_ALGORITHM,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": prepared_context["support_index"][
            "support_index_sha256"
        ],
        "plane_domain": "all continuous brain-intersecting arbitrary planes",
        "parent_policy": "one immutable RP2 parent per logical sample; no raster redraw",
        "slab_policy": (
            "one independent thickness descendant per parent; never pose/tissue conditioned"
        ),
        "paired_mode_policy": "same parent, slab, G1, G2, and G3 across all brush modes",
        "point_pose_gate": (
            "post-G1 sum(source_slab_brain_occupancy_float32) against requested threshold"
        ),
        "dense_gate": (
            "post-G1 center/gauge support threshold and positive post-G3 dense weight mass"
        ),
        "no_drop_policy": FINITE_JOINT_NO_DROP_POLICY,
        "finite_psf_model_capability": psf_v4.finite_psf_model_capability_v4(),
        "render_mode": render_mode,
        "nominal_cut_thickness_um": (
            None
            if nominal_cut_thickness_um is None
            else float(nominal_cut_thickness_um)
        ),
        "root_seed_uint64": _seed(root_seed),
        "start_index": int(start_index),
        "row_count": int(row_count),
        "output_shape_h_w": [int(value) for value in output_shape_h_w],
        "identity_prefix": str(identity_prefix),
        "sections_per_animal": int(sections_per_animal),
        "amplitude_band_cycle": list(amplitude_band_cycle),
        "split": str(split),
        "stratum": str(stratum),
        "margin_u_v_um": np.broadcast_to(
            np.asarray(margin_um, dtype=float), (2,)
        ).tolist(),
        "minimum_brain_pixels": int(minimum_brain_pixels),
        "maximum_rejection_attempts": int(maximum_rejection_attempts),
        "maximum_joint_rejection_attempts": int(maximum_joint_rejection_attempts),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }


def _finalize_row_v4(row_like, slab_block):
    return psf_v4.finalize_training_row_v4(
        row_like,
        slab_block,
        capability=psf_v4.finite_psf_model_capability_v4(),
    )


def make_finite_joint_curriculum_training_row_v5(
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
    maximum_joint_rejection_attempts=16,
    render_mode=finite_slab.FINITE_BOXCAR,
    nominal_cut_thickness_um=None,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
    joint_attempt_number=0,
    joint_rejection_history=None,
    identity_g1_pose_only_fallback=False,
    requested_deformation_amplitude_band=None,
    deformation_censor_status=None,
    deformation_censor_reason=None,
    fallback_attempt_number=None,
    fallback_synthetic_seed_uint64=None,
    finite_parent_identity=None,
    finite_slab_identity=None,
    supervision_evidence=None,
    thickness_seed_uint64=None,
    _finite_parent_artifact=None,
    _finite_slab_adapter_result=None,
):
    if selected_mode not in pose_curriculum.MODE_TO_OUTLINE:
        raise ValueError("finite-joint brush mode is invalid")
    if reflection_state not in training_row_v3.REFLECTION_STATES:
        raise ValueError("finite-joint reflection state is invalid")
    identities = {
        "animal_id": animal_id,
        "specimen_id": specimen_id,
        "experiment_id": experiment_id,
        "synthetic_animal_id": synthetic_animal_id,
        "section_id": section_id,
    }
    if any(value in (None, "") for value in identities.values()):
        raise ValueError("every finite-joint row requires complete lineage IDs")
    sample_index = int(sample_index)
    attempt = int(joint_attempt_number)
    maximum_attempts = int(maximum_joint_rejection_attempts)
    identity_fallback = bool(identity_g1_pose_only_fallback)
    if (
        maximum_attempts <= 0
        or attempt < 0
        or attempt > maximum_attempts
        or identity_fallback != (attempt == maximum_attempts)
    ):
        raise ValueError("finite-joint attempt/fallback state is invalid")
    if (
        requested_deformation_amplitude_band is not None
        and requested_deformation_amplitude_band != amplitude_band
    ):
        raise ValueError("requested finite-joint amplitude band changed")
    requested_deformation_amplitude_band = amplitude_band
    expected_thickness_seed = _seed(
        _derived_seed(root_seed, sample_index, 0, "finite-thickness")
    )
    if thickness_seed_uint64 is not None and thickness_seed_uint64 != expected_thickness_seed:
        raise ValueError("independent thickness seed changed")
    thickness_seed_uint64 = expected_thickness_seed
    if (_finite_parent_artifact is None) != (_finite_slab_adapter_result is None):
        raise ValueError("finite parent and slab must be supplied together")
    if _finite_parent_artifact is None:
        parent, slab_adapter, plane_index, finite_seed, _ = _make_parent_and_slab(
            prepared_context,
            root_seed=root_seed,
            sample_index=sample_index,
            output_shape_h_w=output_shape_h_w,
            identities=identities,
            split=split,
            stratum=stratum,
            margin_um=margin_um,
            minimum_brain_pixels=minimum_brain_pixels,
            maximum_rejection_attempts=maximum_rejection_attempts,
            render_mode=render_mode,
            nominal_cut_thickness_um=nominal_cut_thickness_um,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
            finite_slab_generator_source_commit=finite_slab_generator_source_commit,
        )
    else:
        parent = _finite_parent_artifact
        slab_adapter = _finite_slab_adapter_result
        plane_index = finite_joint_plane_sample_index_v5(root_seed, sample_index)
        finite_seed = _derived_seed(root_seed, sample_index, 0, "finite-render")
    actual_parent_identity = _parent_identity(
        root_seed, sample_index, identities, parent
    )
    actual_slab_identity = _slab_identity(
        slab_adapter, actual_parent_identity, thickness_seed_uint64
    )
    if finite_parent_identity is not None and finite_parent_identity != actual_parent_identity:
        raise ValueError("finite parent changed across retries")
    if finite_slab_identity is not None and finite_slab_identity != actual_slab_identity:
        raise ValueError("finite slab changed across retries")
    finite_parent_identity = actual_parent_identity
    finite_slab_identity = actual_slab_identity
    history = _verified_rejection_history(
        root_seed,
        sample_index,
        attempt,
        joint_rejection_history,
        amplitude_band=amplitude_band,
        identities=identities,
        parent_identity=finite_parent_identity,
        slab_identity=finite_slab_identity,
    )
    block = slab_adapter["artifact"]["slab_observation_v4"]
    pre_slab_receipt = block["receipt_sha256"]
    pre_adapter_receipt = slab_adapter["artifact"]["receipt_sha256"]
    synthetic_seed = _derived_seed(root_seed, sample_index, attempt, "synthetic")
    overrides = finite_joint_g1_overrides_v5(amplitude_band)
    parent_center_count = int(parent["acceptance_contract"]["brain_pixel_count"])
    parent_center_identifiable = bool(
        parent_center_count >= int(minimum_brain_pixels)
    )
    if identity_fallback or not parent_center_identifiable:
        overrides = {**overrides, "identity_probability": 1.0}
    support = pose_curriculum._mutable_support(prepared_context["support_index"])
    paired = {
        mode: make_arbitrary_plane_synthetic_realization(
            parent,
            support,
            root_seed=synthetic_seed,
            sample_index=sample_index,
            synthetic_stratum=(
                "low-information-stress" if identity_fallback else "ordinary"
            ),
            outline_mode=outline_mode,
            config_overrides={
                "ordinary_minimum_clean_brain_pixels_floor": int(
                    minimum_brain_pixels
                ),
                "ordinary_minimum_clean_brain_fraction": 0.0,
                "g1": overrides,
                **(
                    {"g3": {"imperfect_iou": [0.0, 1.0]}}
                    if identity_fallback
                    else {}
                ),
            },
            slab_observation_v4=block,
            lineage={**identities, "split": split},
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
        )
        for mode, outline_mode in pose_curriculum.MODE_TO_OUTLINE.items()
    }
    if (
        len({value["paired_view_group_id"] for value in paired.values()}) != 1
        or any(
            value["slab_observation_v4"]["receipt_sha256"] != pre_slab_receipt
            for value in paired.values()
        )
        or block["receipt_sha256"] != pre_slab_receipt
        or slab_adapter["artifact"]["receipt_sha256"] != pre_adapter_receipt
    ):
        raise RuntimeError("paired modes did not preserve one immutable parent and slab")
    selected = paired[selected_mode]
    if selected["lineage"] != {**identities, "split": split}:
        raise RuntimeError("finite-joint synthetic lineage differs from its row lineage")
    source = selected["arrays"]
    accepted_g1 = selected["g1"]["parameters"]["accepted_attempt"]
    expected_identity = bool(identity_fallback or not parent_center_identifiable)
    similarity = accepted_g1["similarity"]
    if (
        accepted_g1["identity_path"] != expected_identity
        or similarity["angle_rad"] != 0.0
        or similarity["scale"] != 1.0
        or similarity["translation_xy_px"] != [0.0, 0.0]
        or accepted_g1["affine_projection_contract"]
        != UNIFORM_CANVAS_AFFINE_PROJECTION
        or accepted_g1["integration_contract"]
        != FIXED_SEVEN_DECODER_INTEGRATION
        or accepted_g1["integration_steps"] != 7
    ):
        raise RuntimeError("finite-joint G1 is outside the direct affine-free gauge")
    actual_evidence = _supervision_evidence(
        paired, minimum_brain_pixels, identity_fallback
    )
    actual_evidence["pre_g1_slab_effective_brain_pixel_mass"] = float(
        block["slab_effective_brain_pixel_mass"]
    )
    actual_evidence["requested_identifiability_threshold_pixels"] = int(
        minimum_brain_pixels
    )
    if supervision_evidence is not None and supervision_evidence != actual_evidence:
        raise ValueError("finite-joint supervision evidence changed")
    supervision_evidence = actual_evidence
    nonidentity = not accepted_g1["identity_path"]
    if nonidentity and not np.any(source["velocity_xy_px"] != 0.0):
        raise ValueError(ZERO_AFFINE_FREE_REJECTION)
    center_gauge_mask = np.asarray(source["source_map_domain_mask"], dtype=bool) & np.asarray(
        source["source_clean_tissue_mask"], dtype=bool
    )
    certification_mask = (
        center_gauge_mask
        if center_gauge_mask.any()
        else np.asarray(source["source_map_domain_mask"], dtype=bool)
    )
    target_velocity = np.ascontiguousarray(
        -np.moveaxis(source["velocity_xy_px"], 0, -1)[..., ::-1],
        dtype=np.float32,
    )
    pullback_yx = np.ascontiguousarray(
        np.moveaxis(source["source_to_fixed_map"], 0, -1)[..., ::-1],
        dtype=np.float32,
    )
    effective_pose = np.asarray(
        parent["geometry"]["effective_quicknii_ouv_ml_ap_dv"], dtype=np.float64
    ).reshape(3, 3)
    certified = direct_target.certify_direct_deformation_target_v4(
        target_velocity,
        pullback_yx,
        effective_pose,
        certification_mask,
    )
    affine_free_velocity = certified["arrays"][
        "target_pullback_stationary_velocity_yx_px_float32"
    ]
    if nonidentity and not np.any(affine_free_velocity != 0.0):
        raise ValueError(ZERO_AFFINE_FREE_REJECTION)
    dense_identifiable = bool(
        supervision_evidence["dense_deformation_supervision_identifiable"]
        and nonidentity
    )
    supervision_evidence["dense_deformation_supervision_identifiable"] = dense_identifiable
    supervision_evidence["dense_deformation_supervision_weight"] = float(
        dense_identifiable
    )
    if identity_fallback:
        actual_status = IDENTITY_FALLBACK_CENSOR_STATUS
        actual_reason = NONIDENTITY_RETRY_EXHAUSTION_CENSOR_REASON
    elif not supervision_evidence["center_gauge_support_identifiable"] or not nonidentity:
        actual_status = MARGINAL_CENTRE_GAUGE_CENSOR_STATUS
        actual_reason = MARGINAL_CENTRE_GAUGE_CENSOR_REASON
    elif supervision_evidence["post_g3_dense_correspondence_weight_mass"] <= 0.0:
        actual_status = ZERO_POST_G3_DENSE_MASS_CENSOR_STATUS
        actual_reason = ZERO_POST_G3_DENSE_MASS_CENSOR_REASON
    else:
        actual_status = UNCENSORED_DEFORMATION_STATUS
        actual_reason = None
    if deformation_censor_status is not None and deformation_censor_status != actual_status:
        raise ValueError("finite-joint deformation censor status changed")
    if deformation_censor_reason is not None and deformation_censor_reason != actual_reason:
        raise ValueError("finite-joint deformation censor reason changed")
    deformation_censor_status = actual_status
    deformation_censor_reason = actual_reason
    actual_fallback_attempt = attempt if identity_fallback else None
    actual_fallback_seed = _seed(synthetic_seed) if identity_fallback else None
    if fallback_attempt_number is not None and fallback_attempt_number != actual_fallback_attempt:
        raise ValueError("finite-joint fallback attempt changed")
    if (
        fallback_synthetic_seed_uint64 is not None
        and fallback_synthetic_seed_uint64 != actual_fallback_seed
    ):
        raise ValueError("finite-joint fallback seed changed")
    fallback_attempt_number = actual_fallback_attempt
    fallback_synthetic_seed_uint64 = actual_fallback_seed
    height, width = source["model_input_image"].shape
    canonical = effective_pose.copy()
    horizontal = reflection_state == "horizontal"
    observed, affine, representation_index = pose_curriculum._reflection_geometry(
        canonical, (height, width), reflection_state
    )
    reflected_pullback = pose_curriculum._reflect(
        certified["arrays"]["certified_pullback_map_yx_px_float32"], horizontal
    ).copy()
    reflected_velocity = pose_curriculum._reflect(
        affine_free_velocity, horizontal
    ).copy()
    if horizontal:
        reflected_pullback[..., 1] = width - 1.0 - reflected_pullback[..., 1]
        reflected_velocity[..., 1] *= -1.0
    arrays = {
        "model_input_channels_float32": pose_curriculum._input_channels(
            selected, horizontal
        ),
        "source_label_ground_truth_canvas_int64": pose_curriculum._reflect(
            source["source_annotation"], horizontal
        ).astype(np.int64),
        "source_tissue_ground_truth_mask": pose_curriculum._reflect(
            source["source_clean_tissue_mask"], horizontal
        ),
        "target_ccf_coordinates_ap_dv_ml_um_float64": pose_curriculum._reflect(
            source["source_ccf_ap_dv_ml_um"], horizontal
        ).astype(np.float64),
        "target_valid_correspondence_mask": pose_curriculum._reflect(
            source["source_valid_correspondence_mask"], horizontal
        ),
        "target_correspondence_weight_float32": pose_curriculum._reflect(
            source["source_dense_correspondence_weight_float32"], horizontal
        ).astype(np.float32),
        "target_correspondence_abstention_mask": pose_curriculum._reflect(
            source["source_dense_correspondence_abstention_mask"], horizontal
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(
            reflected_pullback, dtype=np.float64
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.ascontiguousarray(
            reflected_velocity, dtype=np.float64
        ),
        "truth_section_deformation_valid_mask": pose_curriculum._reflect(
            center_gauge_mask, horizontal
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
            "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/paired-source",
            "finite_parent_receipt_sha256": parent["finite_render_receipt_sha256"],
            "finite_slab_adapter_receipt_sha256": pre_adapter_receipt,
            "slab_observation_v4_receipt_sha256": pre_slab_receipt,
            "paired_synthetic_receipts_sha256": {
                mode: realization["synthetic_receipt_sha256"]
                for mode, realization in paired.items()
            },
            "direct_deformation_target_id": certified["direct_deformation_target_id"],
            "supervision_evidence": supervision_evidence,
            "deformation_censor_status": deformation_censor_status,
        }
    )
    transform_id = acquisition._payload_sha256(
        {
            "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/reflection-transform",
            "state": reflection_state,
            "canvas_width": width,
            "affine_xy": affine.tolist(),
        }
    )
    adapter_configuration = {
        "root_seed": _seed(root_seed),
        "sample_index": sample_index,
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
        "maximum_joint_rejection_attempts": maximum_attempts,
        "render_mode": render_mode,
        "nominal_cut_thickness_um": (
            None
            if nominal_cut_thickness_um is None
            else float(nominal_cut_thickness_um)
        ),
        "finite_parent_generator_source_commit": finite_parent_generator_source_commit,
        "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
        "joint_attempt_number": attempt,
        "joint_rejection_history": history,
        "identity_g1_pose_only_fallback": identity_fallback,
        "requested_deformation_amplitude_band": requested_deformation_amplitude_band,
        "deformation_censor_status": deformation_censor_status,
        "deformation_censor_reason": deformation_censor_reason,
        "fallback_attempt_number": fallback_attempt_number,
        "fallback_synthetic_seed_uint64": fallback_synthetic_seed_uint64,
        "finite_parent_identity": copy.deepcopy(finite_parent_identity),
        "finite_slab_identity": copy.deepcopy(finite_slab_identity),
        "supervision_evidence": copy.deepcopy(supervision_evidence),
        "thickness_seed_uint64": thickness_seed_uint64,
    }
    support_contract = {
        "continuous_plane_sample_retained": True,
        "pose_redrawn_for_raster_support": False,
        "finite_parent_reused_across_retries": True,
        "finite_slab_reused_across_retries": True,
        "paired_brush_modes_share_post_g1_evidence": True,
        "fallback_generation_stratum": (
            "low-information-stress" if identity_fallback else None
        ),
        **copy.deepcopy(supervision_evidence),
        "pre_g1_slab_observation_role": "diagnostic and lineage only",
        "post_g1_slab_observation_role": "point-pose loss decision",
        "post_g3_dense_observation_role": "dense-deformation loss decision",
    }
    upstream = {
        "schema_version": FINITE_JOINT_CURRICULUM_V5_SCHEMA,
        "algorithm": FINITE_JOINT_CURRICULUM_V5_ALGORITHM,
        "implementation_source_sha256": _source_sha256(),
        "adapter_configuration": adapter_configuration,
        "prepared_context_sha256": prepared_context["prepared_context_sha256"],
        "support_index_sha256": support["support_index_sha256"],
        "joint_rejection_history": history,
        "deformation_amplitude_band": amplitude_band,
        "requested_deformation_amplitude_band": requested_deformation_amplitude_band,
        "deformation_censoring_contract": {
            "status": deformation_censor_status,
            "reason": deformation_censor_reason,
            "identity_g1_pose_only_fallback": identity_fallback,
            "fallback_attempt_number": fallback_attempt_number,
            "fallback_synthetic_seed_uint64": fallback_synthetic_seed_uint64,
            "fresh_identity_g1_realization": identity_fallback,
            "rejected_nonidentity_image_relabeling_allowed": False,
        },
        "support_supervision_contract": support_contract,
        "g1_overrides": overrides,
        "finite_parent_identity": copy.deepcopy(finite_parent_identity),
        "finite_parent_generator_binding": copy.deepcopy(parent["generator"]),
        "finite_parent_provenance": copy.deepcopy(parent["provenance"]),
        "finite_parent_provenance_sha256": parent["provenance_sha256"],
        "finite_plane_render_id": parent["finite_plane_render_id"],
        "finite_render_receipt_sha256": parent["finite_render_receipt_sha256"],
        "finite_slab_identity": copy.deepcopy(finite_slab_identity),
        "finite_slab_generator_binding": copy.deepcopy(
            slab_adapter["artifact"]["generator"]
        ),
        "finite_slab_provenance": copy.deepcopy(
            slab_adapter["artifact"]["provenance"]
        ),
        "finite_slab_adapter_receipt_sha256": pre_adapter_receipt,
        "slab_observation_id": block["slab_observation_id"],
        "centre_plane_targets_receipt_sha256": block[
            "centre_plane_targets_receipt_sha256"
        ],
        "slab_observation_v4_receipt_sha256": pre_slab_receipt,
        "finite_psf_sha256": block["finite_psf"]["finite_psf_sha256"],
        "finite_psf_capability_sha256": block["finite_psf"][
            "finite_psf_capability_sha256"
        ],
        "finite_psf": copy.deepcopy(block["finite_psf"]),
        "thickness_selection": copy.deepcopy(block["thickness_selection"]),
        "pre_g1_slab_effective_brain_pixel_mass": float(
            block["slab_effective_brain_pixel_mass"]
        ),
        "effective_pose_source_key": (
            "parent['geometry']['effective_quicknii_ouv_ml_ap_dv']"
        ),
        "effective_quicknii_ouv_ml_ap_dv_before_gauge": effective_pose.tolist(),
        "effective_quicknii_ouv_ml_ap_dv_after_gauge": canonical.tolist(),
        "plane_sampling_measure": copy.deepcopy(parent["sampling_measure"]),
        "selected_synthetic_receipt_sha256": selected["synthetic_receipt_sha256"],
        "selected_synthetic_generator_binding": copy.deepcopy(selected["generator"]),
        "selected_synthetic_provenance_sha256": selected["provenance_sha256"],
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
        "selected_g1_accepted_attempt": copy.deepcopy(accepted_g1),
        "direct_deformation_target_certification_summary": (
            direct_target.direct_deformation_target_summary_v4(certified)
        ),
        "direct_target_certification_mask_role": (
            "post-G1 centre/gauge support"
            if center_gauge_mask.any()
            else "full map domain numeric certificate only; dense target remains abstained"
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
    numeric = {
        "schema_version": FINITE_JOINT_CURRICULUM_V5_SCHEMA,
        "root_seed_uint64": _seed(root_seed),
        "sample_index": sample_index,
        "joint_attempt_number": attempt,
        "derived_plane_sample_index": plane_index,
        "finite_render_seed_uint64": _seed(finite_seed),
        "finite_thickness_seed_uint64": thickness_seed_uint64,
        "synthetic_seed_uint64": _seed(synthetic_seed),
    }
    if identity_fallback:
        numeric.update(
            {
                "identity_g1_pose_only_fallback": True,
                "fallback_attempt_number": fallback_attempt_number,
                "fallback_synthetic_seed_uint64": fallback_synthetic_seed_uint64,
            }
        )
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
            "finite_thickness_seed_uint64": thickness_seed_uint64,
            "synthetic_g1_accepted_attempt": accepted_g1["field_stream_seed_uint64"],
        },
        "selected_mode": selected_mode,
        "selected_descendant_id": selected["synthetic_realization_id"],
        "deformation_pose_gauge_reference": (
            direct_target.direct_deformation_target_reference_v4(certified)
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
                "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/reflection-realization",
                "source_bundle": source_bundle,
                "transform_id": transform_id,
                "numeric_rng_provenance": numeric,
            }
        ),
        "paired_view_group_id": acquisition._payload_sha256(
            {
                "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/paired-view",
                "latent_group": selected["paired_view_group_id"],
                "slab_observation_id": block["slab_observation_id"],
                "direct_deformation_target_id": certified[
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
    row_like["synthetic_realization_id"] = acquisition._payload_sha256(
        {
            "domain": f"{FINITE_JOINT_CURRICULUM_V5_SCHEMA}/training-realization",
            "source_bundle": source_bundle,
            "selected_descendant_id": row_like["selected_descendant_id"],
            "reflection_realization_id": row_like["reflection_realization_id"],
        }
    )
    return _finalize_row_v4(row_like, block)


def replay_finite_joint_curriculum_training_row_v5(row, prepared_context):
    return make_finite_joint_curriculum_training_row_v5(
        prepared_context,
        **copy.deepcopy(row["upstream_reference"]["adapter_configuration"]),
    )


def verify_finite_joint_curriculum_training_row_v5(row, prepared_context):
    psf_v4.verify_training_row_v4(
        row, capability=psf_v4.finite_psf_model_capability_v4()
    )
    upstream = row.get("upstream_reference", {})
    if (
        upstream.get("schema_version") != FINITE_JOINT_CURRICULUM_V5_SCHEMA
        or upstream.get("algorithm") != FINITE_JOINT_CURRICULUM_V5_ALGORITHM
        or upstream.get("implementation_source_sha256") != _source_sha256()
        or any(
            upstream.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("finite-joint implementation or independence binding changed")
    adapter = copy.deepcopy(upstream["adapter_configuration"])
    history = adapter["joint_rejection_history"]
    for attempt_index, expected in enumerate(history):
        rejected = copy.deepcopy(adapter)
        rejected["joint_attempt_number"] = attempt_index
        rejected["joint_rejection_history"] = history[:attempt_index]
        rejected["identity_g1_pose_only_fallback"] = False
        rejected["deformation_censor_status"] = None
        rejected["deformation_censor_reason"] = None
        rejected["fallback_attempt_number"] = None
        rejected["fallback_synthetic_seed_uint64"] = None
        rejected["supervision_evidence"] = None
        try:
            make_finite_joint_curriculum_training_row_v5(
                prepared_context, **rejected
            )
        except ValueError as error:
            if str(error) != expected["reason"]:
                raise ValueError(
                    "finite-joint rejection history does not replay exactly"
                ) from error
        else:
            raise ValueError("finite-joint rejection history names an accepted attempt")
    replay = replay_finite_joint_curriculum_training_row_v5(row, prepared_context)
    if (
        set(row) != set(replay)
        or psf_v4.training_row_receipt_v4(row)
        != psf_v4.training_row_receipt_v4(replay)
        or any(
            np.asarray(row["arrays"][name]).dtype
            != np.asarray(replay["arrays"][name]).dtype
            or not np.array_equal(row["arrays"][name], replay["arrays"][name])
            for name in training_row_v3._ARRAY_KEYS
        )
    ):
        raise ValueError("finite-joint row does not replay exactly")
    return True


def make_finite_joint_curriculum_training_rows_v5(
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
    render_mode=finite_slab.FINITE_BOXCAR,
    nominal_cut_thickness_um=None,
    finite_parent_generator_source_commit=None,
    finite_slab_generator_source_commit=None,
):
    if (
        int(sections_per_animal) <= 0
        or int(maximum_joint_rejection_attempts) <= 0
        or not amplitude_band_cycle
        or any(
            band not in DEFORMATION_AMPLITUDE_BANDS
            for band in amplitude_band_cycle
        )
    ):
        raise ValueError("finite-joint batch grouping or retry config is invalid")
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
        parent, slab_adapter, _, _, thickness_seed = _make_parent_and_slab(
            prepared_context,
            root_seed=root_seed,
            sample_index=sample_index,
            output_shape_h_w=output_shape_h_w,
            identities=identities,
            split=split,
            stratum=stratum,
            margin_um=margin_um,
            minimum_brain_pixels=minimum_brain_pixels,
            maximum_rejection_attempts=maximum_rejection_attempts,
            render_mode=render_mode,
            nominal_cut_thickness_um=nominal_cut_thickness_um,
            finite_parent_generator_source_commit=finite_parent_generator_source_commit,
            finite_slab_generator_source_commit=finite_slab_generator_source_commit,
        )
        parent_identity = _parent_identity(
            root_seed, sample_index, identities, parent
        )
        slab_identity = _slab_identity(
            slab_adapter, parent_identity, _seed(thickness_seed)
        )
        amplitude_band = amplitude_band_cycle[
            sample_index % len(amplitude_band_cycle)
        ]
        common = {
            "root_seed": root_seed,
            "sample_index": sample_index,
            "output_shape_h_w": output_shape_h_w,
            "selected_mode": training_row_v3.TRAINABLE_MODES[
                sample_index % len(training_row_v3.TRAINABLE_MODES)
            ],
            "reflection_state": training_row_v3.REFLECTION_STATES[
                (sample_index // len(training_row_v3.TRAINABLE_MODES))
                % len(training_row_v3.REFLECTION_STATES)
            ],
            "amplitude_band": amplitude_band,
            **identities,
            "split": split,
            "stratum": stratum,
            "margin_um": margin_um,
            "minimum_brain_pixels": minimum_brain_pixels,
            "maximum_rejection_attempts": maximum_rejection_attempts,
            "maximum_joint_rejection_attempts": int(
                maximum_joint_rejection_attempts
            ),
            "render_mode": render_mode,
            "nominal_cut_thickness_um": nominal_cut_thickness_um,
            "finite_parent_generator_source_commit": (
                finite_parent_generator_source_commit
            ),
            "finite_slab_generator_source_commit": finite_slab_generator_source_commit,
            "finite_parent_identity": parent_identity,
            "finite_slab_identity": slab_identity,
            "thickness_seed_uint64": _seed(thickness_seed),
            "_finite_parent_artifact": parent,
            "_finite_slab_adapter_result": slab_adapter,
        }
        rejection_history = []
        for attempt in range(int(maximum_joint_rejection_attempts)):
            try:
                row = make_finite_joint_curriculum_training_row_v5(
                    prepared_context,
                    **common,
                    joint_attempt_number=attempt,
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
                        attempt,
                        reason,
                        amplitude_band=amplitude_band,
                        identities=identities,
                        parent_identity=parent_identity,
                        slab_identity=slab_identity,
                    )
                )
            else:
                rows.append(row)
                break
        else:
            fallback_attempt = int(maximum_joint_rejection_attempts)
            rows.append(
                make_finite_joint_curriculum_training_row_v5(
                    prepared_context,
                    **common,
                    joint_attempt_number=fallback_attempt,
                    joint_rejection_history=rejection_history,
                    identity_g1_pose_only_fallback=True,
                    requested_deformation_amplitude_band=amplitude_band,
                    deformation_censor_status=IDENTITY_FALLBACK_CENSOR_STATUS,
                    deformation_censor_reason=(
                        NONIDENTITY_RETRY_EXHAUSTION_CENSOR_REASON
                    ),
                    fallback_attempt_number=fallback_attempt,
                    fallback_synthetic_seed_uint64=_seed(
                        _derived_seed(
                            root_seed,
                            sample_index,
                            fallback_attempt,
                            "synthetic",
                        )
                    ),
                )
            )
    return rows


__all__ = [
    "DEFORMATION_AMPLITUDE_BANDS",
    "FINITE_JOINT_CURRICULUM_V5_ALGORITHM",
    "FINITE_JOINT_CURRICULUM_V5_SCHEMA",
    "FINITE_JOINT_NO_DROP_POLICY",
    "IDENTITY_FALLBACK_CENSOR_STATUS",
    "INTEGRATION_DEPENDENCIES",
    "MARGINAL_CENTRE_GAUGE_CENSOR_STATUS",
    "NONIDENTITY_RETRY_EXHAUSTION_CENSOR_REASON",
    "RETRYABLE_REJECTION_STAGES",
    "UNCENSORED_DEFORMATION_STATUS",
    "ZERO_POST_G3_DENSE_MASS_CENSOR_STATUS",
    "finite_joint_curriculum_generation_config_v5",
    "finite_joint_g1_overrides_v5",
    "finite_joint_plane_sample_index_v5",
    "make_finite_joint_curriculum_training_row_v5",
    "make_finite_joint_curriculum_training_rows_v5",
    "replay_finite_joint_curriculum_training_row_v5",
    "verify_finite_joint_curriculum_training_row_v5",
]
