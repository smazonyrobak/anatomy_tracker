"""Bounded, authenticated support-only resolution for generic subject planes."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab


SUBJECT_SUPPORT_RESOLUTION_V2_SCHEMA = (
    "anatomy-tracker.subject-support-resolution/v2"
)
SUBJECT_SUPPORT_RESOLUTION_V2_ALGORITHM = (
    "bounded-first-nonzero-post-deformation-centre-support/v2"
)
SUPPORT_ATTEMPT_RNG_DOMAIN = "anatomy-tracker.subject-support-attempt-root/v2"
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_support_resolution_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_section_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_geometry.py",
    "arbitrary_plane_rendered_generator.py",
)
_LEARNED_DEPENDENCY_KEYS = {
    "learned_checkpoint_dependencies",
    "previous_model_dependencies",
    "pretrained_feature_dependencies",
    "learned_style_model_dependencies",
}


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
            raise ValueError("master_root_seed must be uint64 or 0x plus 16 lowercase hex digits")
        return int(root_seed[2:], 16)
    if isinstance(root_seed, (bool, np.bool_)) or not isinstance(
        root_seed, (int, np.integer)
    ):
        raise TypeError("master_root_seed must be an integer or canonical uint64 hex string")
    value = int(root_seed)
    if value < 0 or value >= 1 << 64:
        raise ValueError("master_root_seed must fit uint64")
    return value


def _schedule_uint64(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"{name} must fit uint64")
    return result


def derive_subject_support_attempt_root_seed_v2(
    master_root_seed: int | str,
    split_index: int,
    animal_index: int,
    section_index: int,
    attempt_index: int,
) -> int:
    """Derive an attempt root from numeric scheduling coordinates only."""
    root = _root_seed_uint64(master_root_seed)
    numeric = tuple(
        _schedule_uint64(value, name)
        for value, name in zip(
            (split_index, animal_index, section_index, attempt_index),
            ("split_index", "animal_index", "section_index", "attempt_index"),
        )
    )
    digest = hashlib.blake2b(digest_size=8, person=b"AT-SUP-RES-V2")
    parts = (
        SUPPORT_ATTEMPT_RNG_DOMAIN.encode("utf-8"),
        root.to_bytes(8, "little", signed=False),
        *(value.to_bytes(8, "little", signed=False) for value in numeric),
    )
    for part in parts:
        digest.update(len(part).to_bytes(8, "little", signed=False))
        digest.update(part)
    return int.from_bytes(digest.digest(), "little", signed=False)


def _attempt_seed_receipt(
    config: dict[str, object], attempt_index: int
) -> dict[str, object]:
    seed = derive_subject_support_attempt_root_seed_v2(
        config["master_root_seed_uint64"],
        config["split_index"],
        config["animal_index"],
        config["section_index"],
        attempt_index,
    )
    return {
        "domain": SUPPORT_ATTEMPT_RNG_DOMAIN,
        "digest": "BLAKE2b-64 person=AT-SUP-RES-V2; unsigned little-endian",
        "master_root_seed_uint64": config["master_root_seed_uint64"],
        "split_index": config["split_index"],
        "animal_index": config["animal_index"],
        "section_index": config["section_index"],
        "attempt_index": _schedule_uint64(attempt_index, "attempt_index"),
        "attempt_root_seed_uint64": f"0x{seed:016x}",
    }


def _rng_policy() -> dict[str, object]:
    return {
        "attempt_root_coordinates": [
            "master_root_seed_uint64",
            "split_index",
            "animal_index",
            "section_index",
            "attempt_index",
        ],
        "excluded_from_attempt_root": [
            "split",
            "animal_id",
            "specimen_id",
            "experiment_id",
            "artifact_ids",
        ],
        "generic_plane_sample_index": "exactly section_index for every attempt",
        "generic_precursor_rng_note": (
            "the authenticated generic precursor additionally binds its human split and "
            "fixed sample_index under its own frozen RNG contract"
        ),
    }


def _decision_disclosure() -> dict[str, object]:
    return {
        "decision": "first attempt with at least one nonzero mapped centre annotation pixel",
        "fixed_plane_stratum_within_resolution": True,
        "bounded_first_success_no_redraw_after_acceptance": True,
        "precursor_reference_scalar_rendered": True,
        "precursor_reference_scalar_used_for_decision": False,
        "post_deformation_scalar_sampled": False,
        "appearance_used": False,
        "target_image_overlap_used": False,
    }


def _learned_dependencies() -> dict[str, list[object]]:
    return {name: [] for name in sorted(_LEARNED_DEPENDENCY_KEYS)}


def _lineage(
    *,
    split: str,
    animal_id: str | int,
    animal_index: int,
    specimen_id: str | int | None,
    experiment_id: str | int | None,
) -> dict[str, object]:
    return {
        "split": split,
        "animal_id": acquisition._json_value(animal_id),
        "animal_index": _schedule_uint64(animal_index, "animal_index"),
        "specimen_id": acquisition._json_value(specimen_id),
        "experiment_id": acquisition._json_value(experiment_id),
    }


def _configuration(
    *,
    master_root_seed: int | str,
    split_index: int,
    animal_index: int,
    section_index: int,
    plane_stratum: str,
    nominal_cut_thickness_um: float,
    axial_step_um_max: float,
    parent_shape_h_w: tuple[int, int],
    max_attempts: int,
) -> dict[str, object]:
    shape = tuple(
        _schedule_uint64(value, f"parent_shape_h_w[{index}]")
        for index, value in enumerate(parent_shape_h_w)
    )
    numeric = (
        _schedule_uint64(split_index, "split_index"),
        _schedule_uint64(animal_index, "animal_index"),
        _schedule_uint64(section_index, "section_index"),
        _schedule_uint64(max_attempts, "max_attempts"),
    )
    if (
        numeric[3] < 1
        or numeric[3] > 64
        or len(shape) != 2
        or min(shape) < 2
        or plane_stratum not in acquisition.V2_GENERIC_PLANE_STRATA
        or not math.isfinite(float(nominal_cut_thickness_um))
        or float(nominal_cut_thickness_um) <= 0.0
        or not math.isfinite(float(axial_step_um_max))
        or float(axial_step_um_max) <= 0.0
        or float(axial_step_um_max) > 12.5
    ):
        raise ValueError("subject support-resolution configuration is invalid")
    return {
        "master_root_seed_uint64": f"0x{_root_seed_uint64(master_root_seed):016x}",
        "split_index": numeric[0],
        "animal_index": numeric[1],
        "section_index": numeric[2],
        "plane_stratum": plane_stratum,
        "nominal_cut_thickness_um": float(nominal_cut_thickness_um),
        "axial_step_um_max": float(axial_step_um_max),
        "parent_shape_h_w": list(shape),
        "max_attempts": numeric[3],
    }


def _precursor_reference(precursor: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": precursor["schema_version"],
        "algorithm": precursor["algorithm"],
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "centre_plane_render_id": precursor["centre_plane_render_id"],
        "slab_recipe_id": precursor["slab_recipe_id"],
        "slab_render_id": precursor["slab_render_id"],
        "receipt_sha256": precursor["receipt_sha256"],
    }


def _probe_reference(probe: dict[str, object]) -> dict[str, object]:
    return {
        "subject_centre_support_probe_id": probe[
            "subject_centre_support_probe_id"
        ],
        "receipt_sha256": probe["receipt_sha256"],
        "mapped_centre_coordinate_receipt": probe[
            "mapped_centre_coordinate_receipt"
        ],
        "centre_annotation_receipt": probe["centre_annotation_receipt"],
        "support_acceptance": probe["support_acceptance"],
    }


def _make_verified_attempt(
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    lineage: dict[str, object],
    config: dict[str, object],
    attempt_seed: dict[str, object],
    batch_size: int | None,
    subject_to_ccf_mapper,
) -> tuple[dict[str, object], dict[str, object]]:
    precursor = slab.make_v2_generic_global_reference_slab_render(
        prepared_context,
        lineage["split"],
        attempt_seed["attempt_root_seed_uint64"],
        config["section_index"],
        config["plane_stratum"],
        nominal_cut_thickness_um=config["nominal_cut_thickness_um"],
        axial_step_um_max=config["axial_step_um_max"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=lineage["animal_id"],
        animal_index=lineage["animal_index"],
        specimen_id=lineage["specimen_id"],
        experiment_id=lineage["experiment_id"],
    )
    slab.verify_v2_generic_global_reference_slab_render(
        precursor, prepared_context
    )
    probe, subject_to_ccf_mapper = (
        subject_slab._make_subject_centre_support_probe_with_mapper_v2(
            prepared_context,
            precursor,
            subject_plan=subject_plan,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
    )
    subject_slab._verify_subject_centre_support_probe_with_mapper_v2(
        probe,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    return precursor, probe


def _plan_identity_payload(resolution: dict[str, object]) -> dict[str, object]:
    return {
        "domain": "anatomy-tracker.subject-support-resolution-plan/v2",
        "schema_version": resolution["schema_version"],
        "algorithm": resolution["algorithm"],
        "implementation_source_sha256": resolution[
            "implementation_source_sha256"
        ],
        "implementation_source_sha256_canonicalization": resolution[
            "implementation_source_sha256_canonicalization"
        ],
        "learned_dependencies": resolution["learned_dependencies"],
        "context_reference": resolution["context_reference"],
        "deformation_reference": resolution["deformation_reference"],
        "configuration": resolution["configuration"],
        "lineage": resolution["lineage"],
        "rng_policy": resolution["rng_policy"],
        "decision_disclosure": resolution["decision_disclosure"],
    }


def _resolution_identity_payload(resolution: dict[str, object]) -> dict[str, object]:
    return {
        "domain": SUBJECT_SUPPORT_RESOLUTION_V2_SCHEMA,
        "support_resolution_plan_id": resolution["support_resolution_plan_id"],
        "attempts": resolution["attempts"],
        "status": resolution["status"],
        "accepted_attempt_index": resolution["accepted_attempt_index"],
        "accepted_precursor_reference": resolution[
            "accepted_precursor_reference"
        ],
        "accepted_probe_reference": resolution["accepted_probe_reference"],
    }


def subject_support_resolution_receipt_v2(
    resolution: dict[str, object],
) -> dict[str, object]:
    return {
        "subject_support_resolution_id": resolution[
            "subject_support_resolution_id"
        ],
        "plan_identity_payload": _plan_identity_payload(resolution),
        "resolution_identity_payload": _resolution_identity_payload(resolution),
    }


def _resolve_subject_support_with_mapper_v2(
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    master_root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    plane_stratum: str,
    nominal_cut_thickness_um: float,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    axial_step_um_max: float = 12.5,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    max_attempts: int = 8,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> dict[str, object]:
    if not isinstance(split, str) or not split or animal_id is None:
        raise ValueError("support-resolution split and animal_id must be nonempty/non-null")
    acquisition._validate_v2_context(prepared_context)
    config = _configuration(
        master_root_seed=master_root_seed,
        split_index=split_index,
        animal_index=animal_index,
        section_index=section_index,
        plane_stratum=plane_stratum,
        nominal_cut_thickness_um=nominal_cut_thickness_um,
        axial_step_um_max=axial_step_um_max,
        parent_shape_h_w=parent_shape_h_w,
        max_attempts=max_attempts,
    )
    lineage = _lineage(
        split=split,
        animal_id=animal_id,
        animal_index=animal_index,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
    )
    if lineage["animal_index"] != config["animal_index"]:
        raise ValueError("support-resolution numeric animal lineage disagrees")

    resolution = {
        "schema_version": SUBJECT_SUPPORT_RESOLUTION_V2_SCHEMA,
        "algorithm": SUBJECT_SUPPORT_RESOLUTION_V2_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "learned_dependencies": _learned_dependencies(),
        "context_reference": subject_slab._context_reference(prepared_context),
        "deformation_reference": subject_slab._deformation_reference(subject_plan),
        "configuration": config,
        "lineage": lineage,
        "rng_policy": _rng_policy(),
        "decision_disclosure": _decision_disclosure(),
    }
    resolution["support_resolution_plan_id"] = acquisition._payload_sha256(
        _plan_identity_payload(resolution)
    )
    attempts = []
    accepted_precursor = None
    accepted_probe = None
    for attempt_index in range(config["max_attempts"]):
        attempt_seed = _attempt_seed_receipt(config, attempt_index)
        precursor, probe = _make_verified_attempt(
            prepared_context,
            subject_plan=subject_plan,
            lineage=lineage,
            config=config,
            attempt_seed=attempt_seed,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
        acceptance = probe["support_acceptance"]
        attempts.append(
            {
                "attempt_index": attempt_index,
                "attempt_seed": attempt_seed,
                "plane_sample_index": config["section_index"],
                "plane_stratum": config["plane_stratum"],
                "precursor_reference": _precursor_reference(precursor),
                "probe_reference": _probe_reference(probe),
                "centre_plane_brain_pixel_count": acceptance[
                    "centre_plane_brain_pixel_count"
                ],
                "accepted": acceptance["accepted"],
            }
        )
        if acceptance["accepted"]:
            accepted_precursor = precursor
            accepted_probe = probe
            break

    accepted = accepted_precursor is not None
    resolution.update(
        {
            "attempts": attempts,
            "status": "accepted" if accepted else "exhausted",
            "accepted_attempt_index": attempts[-1]["attempt_index"] if accepted else None,
            "accepted_precursor_reference": (
                _precursor_reference(accepted_precursor) if accepted else None
            ),
            "accepted_probe_reference": (
                _probe_reference(accepted_probe) if accepted else None
            ),
        }
    )
    resolution["subject_support_resolution_id"] = acquisition._payload_sha256(
        _resolution_identity_payload(resolution)
    )
    resolution["receipt_sha256"] = acquisition._payload_sha256(
        subject_support_resolution_receipt_v2(resolution)
    )
    return {
        "resolution": resolution,
        "accepted_precursor": accepted_precursor,
        "accepted_probe": accepted_probe,
    }


def resolve_subject_support_v2(
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    master_root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    plane_stratum: str,
    nominal_cut_thickness_um: float,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    axial_step_um_max: float = 12.5,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    max_attempts: int = 8,
    batch_size: int | None = None,
) -> dict[str, object]:
    subject_to_ccf_mapper = (
        subject_slab._verified_subject_to_ccf_mapper_for_context_v2(
            prepared_context, subject_plan
        )
    )
    return _resolve_subject_support_with_mapper_v2(
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=master_root_seed,
        split=split,
        split_index=split_index,
        animal_index=animal_index,
        animal_id=animal_id,
        section_index=section_index,
        plane_stratum=plane_stratum,
        nominal_cut_thickness_um=nominal_cut_thickness_um,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        axial_step_um_max=axial_step_um_max,
        parent_shape_h_w=parent_shape_h_w,
        max_attempts=max_attempts,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def _contains_final_id(value: object) -> bool:
    if isinstance(value, dict):
        return "synthetic_realization_id" in value or any(
            _contains_final_id(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_final_id(item) for item in value)
    return False


def _validate_structure(bundle: dict[str, object]) -> None:
    if set(bundle) != {"resolution", "accepted_precursor", "accepted_probe"}:
        raise ValueError("support-resolution result has missing or extra fields")
    resolution = bundle.get("resolution", {})
    if (
        set(resolution)
        != {
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "learned_dependencies",
            "context_reference",
            "deformation_reference",
            "configuration",
            "lineage",
            "rng_policy",
            "decision_disclosure",
            "support_resolution_plan_id",
            "attempts",
            "status",
            "accepted_attempt_index",
            "accepted_precursor_reference",
            "accepted_probe_reference",
            "subject_support_resolution_id",
            "receipt_sha256",
        }
        or set(resolution.get("learned_dependencies", {}))
        != _LEARNED_DEPENDENCY_KEYS
        or set(resolution.get("context_reference", {}))
        != {"schema", "v2_context_sha256", "prepared_context_receipt_sha256"}
        or set(resolution.get("deformation_reference", {}))
        != {"mode", "subject_deformation_plan_receipt", "synthetic_animal_id"}
        or set(resolution.get("configuration", {}))
        != {
            "master_root_seed_uint64",
            "split_index",
            "animal_index",
            "section_index",
            "plane_stratum",
            "nominal_cut_thickness_um",
            "axial_step_um_max",
            "parent_shape_h_w",
            "max_attempts",
        }
        or set(resolution.get("lineage", {}))
        != {"split", "animal_id", "animal_index", "specimen_id", "experiment_id"}
        or set(resolution.get("rng_policy", {})) != set(_rng_policy())
        or set(resolution.get("decision_disclosure", {}))
        != set(_decision_disclosure())
        or _contains_final_id(bundle)
    ):
        raise ValueError("support resolution has missing, extra, learned, or final fields")
    precursor_keys = {
        "schema_version",
        "algorithm",
        "v2_plane_realization_id",
        "centre_plane_render_id",
        "slab_recipe_id",
        "slab_render_id",
        "receipt_sha256",
    }
    probe_keys = {
        "subject_centre_support_probe_id",
        "receipt_sha256",
        "mapped_centre_coordinate_receipt",
        "centre_annotation_receipt",
        "support_acceptance",
    }
    attempt_keys = {
        "attempt_index",
        "attempt_seed",
        "plane_sample_index",
        "plane_stratum",
        "precursor_reference",
        "probe_reference",
        "centre_plane_brain_pixel_count",
        "accepted",
    }
    seed_keys = {
        "domain",
        "digest",
        "master_root_seed_uint64",
        "split_index",
        "animal_index",
        "section_index",
        "attempt_index",
        "attempt_root_seed_uint64",
    }
    for attempt in resolution.get("attempts", []):
        if (
            set(attempt) != attempt_keys
            or set(attempt.get("attempt_seed", {})) != seed_keys
            or set(attempt.get("precursor_reference", {})) != precursor_keys
            or set(attempt.get("probe_reference", {})) != probe_keys
        ):
            raise ValueError("support-resolution attempt has missing or extra fields")


def _validate_semantics(bundle: dict[str, object]) -> None:
    resolution = bundle["resolution"]
    config = resolution["configuration"]
    attempts = resolution["attempts"]
    if (
        not attempts
        or len(attempts) > config["max_attempts"]
        or resolution["lineage"]["animal_index"] != config["animal_index"]
        or resolution["rng_policy"] != _rng_policy()
        or resolution["decision_disclosure"] != _decision_disclosure()
        or any(resolution["learned_dependencies"].values())
    ):
        raise ValueError("support-resolution policy or lineage is invalid")
    for index, attempt in enumerate(attempts):
        if (
            attempt["attempt_index"] != index
            or attempt["attempt_seed"] != _attempt_seed_receipt(config, index)
            or attempt["plane_sample_index"] != config["section_index"]
            or attempt["plane_stratum"] != config["plane_stratum"]
            or attempt["probe_reference"]["support_acceptance"][
                "centre_plane_brain_pixel_count"
            ]
            != attempt["centre_plane_brain_pixel_count"]
            or attempt["probe_reference"]["support_acceptance"]["accepted"]
            != attempt["accepted"]
            or attempt["probe_reference"]["support_acceptance"][
                "target_image_overlap_used"
            ]
            is not False
        ):
            raise ValueError("support-resolution attempt order, seed, or decision disagrees")
    accepted_indices = [
        index for index, attempt in enumerate(attempts) if attempt["accepted"]
    ]
    if resolution["status"] == "accepted":
        if (
            accepted_indices != [len(attempts) - 1]
            or resolution["accepted_attempt_index"] != len(attempts) - 1
            or resolution["accepted_precursor_reference"]
            != attempts[-1]["precursor_reference"]
            or resolution["accepted_probe_reference"]
            != attempts[-1]["probe_reference"]
            or bundle["accepted_precursor"] is None
            or bundle["accepted_probe"] is None
        ):
            raise ValueError("accepted support resolution is not first-success consistent")
    elif resolution["status"] == "exhausted":
        if (
            accepted_indices
            or len(attempts) != config["max_attempts"]
            or resolution["accepted_attempt_index"] is not None
            or resolution["accepted_precursor_reference"] is not None
            or resolution["accepted_probe_reference"] is not None
            or bundle["accepted_precursor"] is not None
            or bundle["accepted_probe"] is not None
        ):
            raise ValueError("exhausted support resolution is inconsistent")
    else:
        raise ValueError("support resolution status is invalid")
    if (
        resolution["support_resolution_plan_id"]
        != acquisition._payload_sha256(_plan_identity_payload(resolution))
        or resolution["subject_support_resolution_id"]
        != acquisition._payload_sha256(_resolution_identity_payload(resolution))
        or resolution["receipt_sha256"]
        != acquisition._payload_sha256(
            subject_support_resolution_receipt_v2(resolution)
        )
    ):
        raise ValueError("support-resolution live identity or receipt disagrees")


def _replay_subject_support_resolution_with_mapper_v2(
    bundle: dict[str, object],
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> dict[str, object]:
    _validate_structure(bundle)
    _validate_semantics(bundle)
    resolution = bundle["resolution"]
    config = resolution["configuration"]
    lineage = resolution["lineage"]
    return _resolve_subject_support_with_mapper_v2(
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
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def replay_subject_support_resolution_v2(
    bundle: dict[str, object],
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    subject_to_ccf_mapper = (
        subject_slab._verified_subject_to_ccf_mapper_for_context_v2(
            prepared_context, subject_plan
        )
    )
    return _replay_subject_support_resolution_with_mapper_v2(
        bundle,
        prepared_context,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def _verify_subject_support_resolution_with_mapper_v2(
    bundle: dict[str, object],
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    master_root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    plane_stratum: str,
    nominal_cut_thickness_um: float,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    axial_step_um_max: float = 12.5,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    max_attempts: int = 8,
    batch_size: int | None = None,
    subject_to_ccf_mapper=None,
) -> None:
    _validate_structure(bundle)
    _validate_semantics(bundle)
    resolution = bundle["resolution"]
    if (
        resolution["schema_version"] != SUBJECT_SUPPORT_RESOLUTION_V2_SCHEMA
        or resolution["algorithm"] != SUBJECT_SUPPORT_RESOLUTION_V2_ALGORITHM
        or resolution["implementation_source_sha256"] != _source_hashes()
        or resolution["implementation_source_sha256_canonicalization"]
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or resolution["learned_dependencies"] != _learned_dependencies()
        or resolution["context_reference"]
        != subject_slab._context_reference(prepared_context)
        or resolution["deformation_reference"]
        != subject_slab._deformation_reference(subject_plan)
    ):
        raise ValueError("support-resolution source, context, or deformation disagrees")
    replay = _resolve_subject_support_with_mapper_v2(
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=master_root_seed,
        split=split,
        split_index=split_index,
        animal_index=animal_index,
        animal_id=animal_id,
        section_index=section_index,
        plane_stratum=plane_stratum,
        nominal_cut_thickness_um=nominal_cut_thickness_um,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        axial_step_um_max=axial_step_um_max,
        parent_shape_h_w=parent_shape_h_w,
        max_attempts=max_attempts,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    if resolution != replay["resolution"]:
        raise ValueError("support-resolution deterministic replay does not match")
    if resolution["status"] == "accepted":
        precursor = bundle["accepted_precursor"]
        probe = bundle["accepted_probe"]
        slab.verify_v2_generic_global_reference_slab_render(
            precursor, prepared_context
        )
        subject_slab._verify_subject_centre_support_probe_with_mapper_v2(
            probe,
            prepared_context,
            precursor,
            subject_plan=subject_plan,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
        if (
            _precursor_reference(precursor)
            != resolution["accepted_precursor_reference"]
            or _probe_reference(probe) != resolution["accepted_probe_reference"]
            or slab.v2_generic_slab_render_receipt(precursor)
            != slab.v2_generic_slab_render_receipt(replay["accepted_precursor"])
            or probe != replay["accepted_probe"]
        ):
            raise ValueError("accepted support-resolution artifacts do not match")


def verify_subject_support_resolution_v2(
    bundle: dict[str, object],
    prepared_context: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    master_root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    plane_stratum: str,
    nominal_cut_thickness_um: float,
    specimen_id: str | int | None = None,
    experiment_id: str | int | None = None,
    axial_step_um_max: float = 12.5,
    parent_shape_h_w: tuple[int, int] = (256, 256),
    max_attempts: int = 8,
    batch_size: int | None = None,
) -> None:
    subject_to_ccf_mapper = (
        subject_slab._verified_subject_to_ccf_mapper_for_context_v2(
            prepared_context, subject_plan
        )
    )
    _verify_subject_support_resolution_with_mapper_v2(
        bundle,
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=master_root_seed,
        split=split,
        split_index=split_index,
        animal_index=animal_index,
        animal_id=animal_id,
        section_index=section_index,
        plane_stratum=plane_stratum,
        nominal_cut_thickness_um=nominal_cut_thickness_um,
        specimen_id=specimen_id,
        experiment_id=experiment_id,
        axial_step_um_max=axial_step_um_max,
        parent_shape_h_w=parent_shape_h_w,
        max_attempts=max_attempts,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def verify_accepted_subject_slab_matches_support_resolution_v2(
    bundle: dict[str, object], subject_slab_render: dict[str, object]
) -> None:
    _validate_structure(bundle)
    _validate_semantics(bundle)
    resolution = bundle["resolution"]
    if resolution["status"] != "accepted":
        raise ValueError("exhausted support resolution has no accepted subject slab")
    probe_reference = resolution["accepted_probe_reference"]
    precursor_reference = resolution["accepted_precursor_reference"]
    coordinate = subject_slab_render["coordinate_map"]
    centre_index = int(coordinate["kernel"]["centre_index"])
    mapped_receipt = acquisition._array_receipt(
        coordinate["arrays"]["mapped_allen_index_coordinates_float32"][
            centre_index
        ]
    )
    annotation_receipt = acquisition._array_receipt(
        subject_slab_render["sample_arrays"]["annotation_samples_int64"][
            centre_index
        ]
    )
    if (
        subject_slab_render["precursor_reference"]["slab_render_id"]
        != precursor_reference["slab_render_id"]
        or subject_slab_render["precursor_reference"][
            "v2_slab_render_receipt_sha256"
        ]
        != precursor_reference["receipt_sha256"]
        or subject_slab_render["support_probe_reference"]
        != {
            "subject_centre_support_probe_id": probe_reference[
                "subject_centre_support_probe_id"
            ],
            "receipt_sha256": probe_reference["receipt_sha256"],
        }
        or subject_slab_render["support_acceptance"]
        != probe_reference["support_acceptance"]
        or mapped_receipt != probe_reference["mapped_centre_coordinate_receipt"]
        or annotation_receipt != probe_reference["centre_annotation_receipt"]
    ):
        raise ValueError("accepted full subject slab does not match support resolution")
