"""End-to-end provenance-bound arbitrary-plane training-row generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_acquisition_window_v3 as window_v3
import training.arbitrary_plane_legacy_chain_v3 as legacy_chain_v3
import training.arbitrary_plane_observation_v3 as observation_v3
import training.arbitrary_plane_section_processing_v2 as section_processing_v2
import training.arbitrary_plane_subject_deformation_v2 as subject_deformation_v2
import training.arbitrary_plane_training_row_v3 as training_row_v3


TRAINING_BUNDLE_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-bundle/v3"
PREPARED_TRAINING_SECTION_V3_SCHEMA = (
    "anatomy-tracker.prepared-training-section/v3"
)
PREPARED_TRAINING_SECTION_V3_ALGORITHM = (
    "authenticate-support-subject-section-once-for-many-descendants/v3"
)
DEFAULT_V3_POINT_BATCH_SIZE = 65536
_SOURCE_ROOT = Path(__file__).parent
_PREPARATION_SPEC_KEYS = (
    "support_root_seed",
    "split",
    "split_index",
    "animal_index",
    "animal_id",
    "section_index",
    "plane_stratum",
    "nominal_cut_thickness_um",
    "specimen_id",
    "experiment_id",
    "axial_step_um_max",
    "maximum_support_attempts",
    "section_root_seed",
    "section_id",
    "section_deformation_mode",
)


def _preparation_spec(spec):
    defaults = {
        "nominal_cut_thickness_um": 55.0,
        "axial_step_um_max": 12.5,
        "maximum_support_attempts": 8,
        "section_deformation_mode": "standard",
    }
    return {
        name: acquisition._json_value(
            spec[name] if name in spec else defaults[name]
        )
        for name in _PREPARATION_SPEC_KEYS
    }


def _prepared_source_hashes():
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in (
            "arbitrary_plane_training_data_v3.py",
            "arbitrary_plane_legacy_chain_v3.py",
            "arbitrary_plane_observation_v3.py",
        )
    }


def training_bundle_receipt_v3(bundle):
    return {
        "schema_version": bundle["schema_version"],
        "generation_spec": bundle["generation_spec"],
        "prepared_section_receipt_sha256": bundle[
            "prepared_section_receipt_sha256"
        ],
        "legacy_chain_adapter_v3": bundle["legacy_chain_adapter_v3"],
        "subject_deformation_plan_id": bundle["subject_plan"][
            "subject_deformation_plan_id"
        ],
        "support_resolution_receipt_sha256": bundle["support_resolution"][
            "receipt_sha256"
        ],
        "precursor_receipt_sha256": bundle["precursor"]["receipt_sha256"],
        "subject_slab_receipt_sha256": bundle["subject_slab"]["receipt_sha256"],
        "section_plan_receipt_sha256": bundle["section_plan"]["receipt_sha256"],
        "section_render_receipt_sha256": bundle["section_render"][
            "receipt_sha256"
        ],
        "window_plan_receipt_sha256": bundle["window_plan"][
            "plan_receipt_sha256"
        ],
        "observation_receipt_sha256": bundle["observation"]["receipt_sha256"],
        "training_row_receipt_sha256": bundle["training_row"]["receipt_sha256"],
    }


def make_training_subject_v3(
    prepared_context,
    *,
    root_seed,
    split,
    animal_index,
    animal_id,
    deformation_stratum="standard",
):
    support = acquisition._context_support(prepared_context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    return subject_deformation_v2.sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed=root_seed,
        split=split,
        animal_index=animal_index,
        animal_id=animal_id,
        ccf_context_sha256=prepared_context["v2_context_sha256"],
        deformation_stratum=deformation_stratum,
    )


def prepared_training_section_receipt_v3(prepared):
    return {
        "schema_version": prepared["schema_version"],
        "algorithm": prepared["algorithm"],
        "implementation_source_sha256": prepared[
            "implementation_source_sha256"
        ],
        "preparation_spec": prepared["preparation_spec"],
        "point_batch_size": prepared["point_batch_size"],
        "legacy_chain_adapter_v3": prepared["legacy_chain_adapter_v3"],
        "subject_deformation_plan_id": prepared["subject_plan"][
            "subject_deformation_plan_id"
        ],
        "support_resolution_receipt_sha256": prepared["support_resolution"][
            "receipt_sha256"
        ],
        "precursor_receipt_sha256": prepared["precursor"]["receipt_sha256"],
        "subject_slab_receipt_sha256": prepared["subject_slab"][
            "receipt_sha256"
        ],
        "section_plan_receipt_sha256": prepared["section_plan"][
            "receipt_sha256"
        ],
        "section_render_receipt_sha256": prepared["section_render"][
            "receipt_sha256"
        ],
        "parent_authentication_receipt_sha256": prepared[
            "parent_authentication_v3"
        ]["receipt_sha256"],
        "prepared_training_section_id": prepared["prepared_training_section_id"],
    }


def make_prepared_training_section_v3(
    prepared_context, subject_plan, spec, *, batch_size=None
):
    if (
        subject_plan is None or "subject_deformation_plan_id" not in subject_plan
    ):
        raise ValueError("prepared v3 training sections require a subject plan")
    point_batch_size = (
        DEFAULT_V3_POINT_BATCH_SIZE if batch_size is None else int(batch_size)
    )
    support_bundle = legacy_chain_v3.resolve_subject_support_v3(
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=spec["support_root_seed"],
        split=spec["split"],
        split_index=spec["split_index"],
        animal_index=spec["animal_index"],
        animal_id=spec["animal_id"],
        section_index=spec["section_index"],
        plane_stratum=spec["plane_stratum"],
        nominal_cut_thickness_um=spec.get("nominal_cut_thickness_um", 55.0),
        specimen_id=spec["specimen_id"],
        experiment_id=spec["experiment_id"],
        axial_step_um_max=spec.get("axial_step_um_max", 12.5),
        parent_shape_h_w=(256, 256),
        max_attempts=spec.get("maximum_support_attempts", 8),
        batch_size=point_batch_size,
    )
    if support_bundle["resolution"]["status"] != "accepted":
        raise RuntimeError("bounded support resolution exhausted")
    precursor = support_bundle["accepted_precursor"]
    subject_slab = legacy_chain_v3.make_subject_slab_render_v3(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=point_batch_size,
    )
    centre_index = int(subject_slab["coordinate_map"]["kernel"]["centre_index"])
    subject_physical = np.asarray(
        subject_slab["coordinate_map"]["arrays"][
            "subject_physical_coordinates_ap_dv_ml_um_float64"
        ][centre_index]
    )
    pixel_pitch_y_x_um, _, _ = section_processing_v2._orthogonal_section_pixel_metric(
        subject_physical
    )
    section_plan = section_processing_v2.sample_section_processing_plan_v2(
        tuple(subject_physical.shape[:2]),
        tuple(pixel_pitch_y_x_um),
        root_seed=spec["section_root_seed"],
        split=spec["split"],
        animal_index=spec["animal_index"],
        section_index=spec["section_index"],
        animal_id=spec["animal_id"],
        section_id=spec["section_id"],
        deformation_mode=spec.get("section_deformation_mode", "standard"),
    )
    section_render = (
        legacy_chain_v3.make_section_processing_render_from_generated_subject_v3(
            subject_slab,
            section_plan,
            batch_size=point_batch_size,
        )
    )
    parent_authentication = observation_v3.authenticate_observation_parent_v3(
        section_render,
        subject_slab,
        section_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=point_batch_size,
    )
    prepared = {
        "schema_version": PREPARED_TRAINING_SECTION_V3_SCHEMA,
        "algorithm": PREPARED_TRAINING_SECTION_V3_ALGORITHM,
        "implementation_source_sha256": _prepared_source_hashes(),
        "preparation_spec": _preparation_spec(spec),
        "point_batch_size": point_batch_size,
        "legacy_chain_adapter_v3": legacy_chain_v3.adapter_receipt_v3(precursor),
        "subject_plan": subject_plan,
        "support_resolution": support_bundle["resolution"],
        "precursor": precursor,
        "subject_slab": subject_slab,
        "section_plan": section_plan,
        "section_render": section_render,
        "parent_authentication_v3": parent_authentication,
    }
    prepared["prepared_training_section_id"] = acquisition._payload_sha256(
        {
            "domain": PREPARED_TRAINING_SECTION_V3_SCHEMA,
            "preparation_spec": prepared["preparation_spec"],
            "legacy_chain_adapter_v3": prepared["legacy_chain_adapter_v3"],
            "subject_deformation_plan_id": subject_plan[
                "subject_deformation_plan_id"
            ],
            "parent_authentication_receipt_sha256": parent_authentication[
                "receipt_sha256"
            ],
        }
    )
    prepared["receipt_sha256"] = acquisition._payload_sha256(
        prepared_training_section_receipt_v3(prepared)
    )
    return prepared


def verify_prepared_training_section_v3(prepared, prepared_context):
    expected_keys = {
        "schema_version",
        "algorithm",
        "implementation_source_sha256",
        "preparation_spec",
        "point_batch_size",
        "legacy_chain_adapter_v3",
        "subject_plan",
        "support_resolution",
        "precursor",
        "subject_slab",
        "section_plan",
        "section_render",
        "parent_authentication_v3",
        "prepared_training_section_id",
        "receipt_sha256",
    }
    if (
        set(prepared) != expected_keys
        or prepared["schema_version"] != PREPARED_TRAINING_SECTION_V3_SCHEMA
        or prepared["algorithm"] != PREPARED_TRAINING_SECTION_V3_ALGORITHM
        or prepared["implementation_source_sha256"] != _prepared_source_hashes()
        or prepared["legacy_chain_adapter_v3"]
        != legacy_chain_v3.adapter_receipt_v3(prepared["precursor"])
        or prepared["receipt_sha256"]
        != acquisition._payload_sha256(prepared_training_section_receipt_v3(prepared))
    ):
        raise ValueError("prepared training section structure or receipt changed")
    observation_v3.verify_observation_parent_authentication_v3(
        prepared["parent_authentication_v3"],
        prepared["section_render"],
        prepared["subject_slab"],
        prepared["section_plan"],
        prepared_context,
        prepared["precursor"],
    )
    return True


def make_training_bundle_from_prepared_section_v3(
    prepared, prepared_context, spec
):
    verify_prepared_training_section_v3(prepared, prepared_context)
    if _preparation_spec(spec) != prepared["preparation_spec"]:
        raise ValueError("descendant generation spec changes its prepared section")
    precursor = prepared["precursor"]
    subject_plan = prepared["subject_plan"]
    subject_slab = prepared["subject_slab"]
    section_plan = prepared["section_plan"]
    section_render = prepared["section_render"]
    point_batch_size = prepared["point_batch_size"]
    sample_index = int(precursor["generator"]["resolved_config"]["sample_index"])
    window_plan = window_v3.sample_acquisition_window_plan_v3(
        root_seed=spec["window_root_seed"],
        split=spec["split"],
        sample_index=sample_index,
    )
    observation = observation_v3.make_arbitrary_plane_observation_v3(
        section_render,
        subject_slab,
        section_plan,
        prepared_context,
        precursor,
        window_plan,
        subject_plan=subject_plan,
        root_seed=spec["observation_root_seed"],
        split=spec["split"],
        split_index=spec["split_index"],
        animal_index=spec["animal_index"],
        animal_id=spec["animal_id"],
        section_index=spec["section_index"],
        observation_index=spec["observation_index"],
        modality=spec["modality"],
        batch_size=point_batch_size,
        authenticated_parent_v3=prepared["parent_authentication_v3"],
    )
    row = training_row_v3.make_arbitrary_plane_training_row_v3(
        observation, spec["realization_index"]
    )
    bundle = {
        "schema_version": TRAINING_BUNDLE_V3_SCHEMA,
        "generation_spec": acquisition._json_value(spec),
        "prepared_section_receipt_sha256": prepared["receipt_sha256"],
        "legacy_chain_adapter_v3": prepared["legacy_chain_adapter_v3"],
        "subject_plan": subject_plan,
        "support_resolution": prepared["support_resolution"],
        "precursor": precursor,
        "subject_slab": subject_slab,
        "section_plan": section_plan,
        "section_render": section_render,
        "window_plan": window_plan,
        "observation": observation,
        "training_row": row,
    }
    bundle["receipt_sha256"] = acquisition._payload_sha256(
        training_bundle_receipt_v3(bundle)
    )
    return bundle


def make_training_bundle_v3(prepared_context, subject_plan, spec, *, batch_size=None):
    prepared = make_prepared_training_section_v3(
        prepared_context, subject_plan, spec, batch_size=batch_size
    )
    return make_training_bundle_from_prepared_section_v3(
        prepared, prepared_context, spec
    )


def replay_training_bundle_v3(
    bundle, prepared_context, subject_plan, *, batch_size=None
):
    return make_training_bundle_v3(
        prepared_context,
        subject_plan,
        bundle["generation_spec"],
        batch_size=batch_size,
    )


def verify_training_bundle_v3(
    bundle, prepared_context, subject_plan, *, batch_size=None
):
    expected_keys = {
        "schema_version",
        "generation_spec",
        "prepared_section_receipt_sha256",
        "legacy_chain_adapter_v3",
        "subject_plan",
        "support_resolution",
        "precursor",
        "subject_slab",
        "section_plan",
        "section_render",
        "window_plan",
        "observation",
        "training_row",
        "receipt_sha256",
    }
    if (
        set(bundle) != expected_keys
        or bundle["schema_version"] != TRAINING_BUNDLE_V3_SCHEMA
        or bundle["legacy_chain_adapter_v3"]
        != legacy_chain_v3.adapter_receipt_v3(bundle["precursor"])
        or bundle["receipt_sha256"]
        != acquisition._payload_sha256(training_bundle_receipt_v3(bundle))
    ):
        raise ValueError("training bundle v3 structure or live receipt changed")
    replay = replay_training_bundle_v3(
        bundle, prepared_context, subject_plan, batch_size=batch_size
    )
    if training_bundle_receipt_v3(bundle) != training_bundle_receipt_v3(replay):
        raise ValueError("training bundle v3 deterministic replay receipt changed")
    for stage in ("observation", "training_row"):
        for name, array in bundle[stage]["arrays"].items():
            replayed = replay[stage]["arrays"][name]
            if (
                np.asarray(array).dtype != np.asarray(replayed).dtype
                or np.asarray(array).shape != np.asarray(replayed).shape
                or np.ascontiguousarray(array).tobytes(order="C")
                != np.ascontiguousarray(replayed).tobytes(order="C")
            ):
                raise ValueError(
                    f"training bundle v3 {stage} array {name} did not replay"
                )
