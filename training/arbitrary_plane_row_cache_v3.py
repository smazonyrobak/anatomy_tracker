"""Authenticated, resumable I:-drive cache for provenance-bound v3 rows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_deformation_gauge_v3 as deformation_gauge_v3
import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge_v4
import training.arbitrary_plane_training_row_v3 as training_row_v3


ROW_CACHE_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-row-cache/v3"
GENERATOR_BINDING_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-generator-binding/v3"
DEVELOPMENT_DATA_ROLE = "development-training"
LEGACY_DEFORMATION_GAUGE_REFERENCE_KEYS = {
    "schema_version",
    "algorithm",
    "projection_weighting",
    "deformation_pose_gauge_id",
    "receipt_sha256",
}
DEFORMATION_GAUGE_REFERENCE_KEYS = {
    "schema_version",
    "algorithm",
    "projection_weighting",
    "target_direction",
    "numeric_contract",
    "runtime_versions",
    "direct_deformation_target_id",
    "receipt_sha256",
}
DEFORMATION_GAUGE_PROJECTION_WEIGHTING = (
    "fixed uniform full canvas, matching decoder gauge"
)
OPEN_CACHE_STATUS = "OPEN"
FROZEN_CACHE_STATUS = "FROZEN"
COMPOSITE_CURRICULUM_V3_SCHEMA = (
    "anatomy-tracker.pose-and-joint-curriculum-cache-config/v3"
)
COMPOSITE_ROW_ORDER_POLICY = (
    "append authenticated pose rows then authenticated joint rows"
)
JOINT_NO_DROP_POLICY = (
    "one authenticated row per logical sample; all nonidentity retries reuse the exact "
    "finite parent; exhaustion generates a fresh identity-G1 pose-only realization and "
    "never relabels a rejected nonidentity image"
)
JOINT_FALLBACK_CENSOR_STATUS = (
    "censored-to-fresh-identity-g1-after-bounded-nonidentity-retries"
)
JOINT_FALLBACK_CENSOR_REASON = (
    "bounded nonidentity G1 realization retries exhausted"
)
JOINT_UNCENSORED_STATUS = "uncensored-direct-nonidentity-g1"
JOINT_MARGINAL_CENSOR_STATUS = "censored-marginal-support-identity-g1"
JOINT_MARGINAL_CENSOR_REASON = (
    "finite parent raster support is below the requested identifiability threshold"
)
COMPOSITE_COMPONENTS = (
    (
        "identity_pose_curriculum",
        "anatomy-tracker.arbitrary-plane-pose-curriculum/v3",
        "unconditioned-uniform-rp2-finite-render-identity-g1-varied-g2-g3-paired-outline/v3",
    ),
    (
        "nonidentity_joint_curriculum",
        "anatomy-tracker.arbitrary-plane-joint-curriculum/v4",
        "unconditioned-uniform-rp2-direct-preintegration-affine-free-source-to-fixed-g1-varied-g2-g3/v4",
    ),
)
FORBIDDEN_SPLIT_TOKENS = (
    "test",
    "benchmark",
    "qualification",
    "external",
    "validation",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("arbitrary-plane row caches must be stored only on I:")
    return resolved


def _atomic_json(path, value):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_npz(path, arrays):
    target = _i_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.ascontiguousarray(value)
                for name, value in sorted(arrays.items())
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_geometry_gauge_contract(contract):
    if not isinstance(contract, dict):
        return False
    if contract == deformation_gauge_v4.direct_deformation_target_contract_v4():
        return True
    return (
        set(contract) == {"schema_version", "algorithm", "projection_weighting"}
        and contract.get("schema_version")
        == deformation_gauge_v3.DEFORMATION_GAUGE_V3_SCHEMA
        and contract.get("algorithm")
        == deformation_gauge_v3.DEFORMATION_GAUGE_V3_ALGORITHM
        and contract.get("projection_weighting")
        == DEFORMATION_GAUGE_PROJECTION_WEIGHTING
    )


def _valid_gauge_reference(reference):
    if not isinstance(reference, dict):
        return False
    if set(reference) == DEFORMATION_GAUGE_REFERENCE_KEYS:
        contract = deformation_gauge_v4.direct_deformation_target_contract_v4()
        return (
            all(reference.get(name) == contract[name] for name in contract)
            and _valid_sha256(reference.get("direct_deformation_target_id"))
            and _valid_sha256(reference.get("receipt_sha256"))
        )
    return (
        set(reference) == LEGACY_DEFORMATION_GAUGE_REFERENCE_KEYS
        and reference.get("schema_version")
        == deformation_gauge_v3.DEFORMATION_GAUGE_V3_SCHEMA
        and reference.get("algorithm")
        == deformation_gauge_v3.DEFORMATION_GAUGE_V3_ALGORITHM
        and reference.get("projection_weighting")
        == DEFORMATION_GAUGE_PROJECTION_WEIGHTING
        and _valid_sha256(reference.get("deformation_pose_gauge_id"))
        and _valid_sha256(reference.get("receipt_sha256"))
    )


def make_generator_binding_v3(
    *,
    generator_ids,
    source_sha256,
    geometry_gauge_contract,
    generator_config,
):
    """Freeze the exact source/config/gauge contract used to make cached rows."""
    generator_ids = sorted({str(value) for value in generator_ids})
    source_sha256 = {str(key): str(value) for key, value in source_sha256.items()}
    if (
        not generator_ids
        or not source_sha256
        or any(not _valid_sha256(value) for value in source_sha256.values())
        or not _valid_geometry_gauge_contract(geometry_gauge_contract)
        or not isinstance(generator_config, dict)
        or not generator_config
    ):
        raise ValueError("generator binding requires IDs, source hashes, config, and gauge contract")
    payload = {
        "schema_version": GENERATOR_BINDING_V3_SCHEMA,
        "generator_ids": generator_ids,
        "source_sha256": source_sha256,
        "geometry_gauge_contract": _plain(geometry_gauge_contract),
        "geometry_gauge_contract_sha256": _hash_json(geometry_gauge_contract),
        "generator_config": _plain(generator_config),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    _canonical_json(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def verify_generator_binding_v3(binding):
    payload = {key: value for key, value in binding.items() if key != "receipt_sha256"}
    if (
        binding.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != GENERATOR_BINDING_V3_SCHEMA
        or not payload.get("generator_ids")
        or payload.get("generator_ids") != sorted(set(payload["generator_ids"]))
        or not payload.get("source_sha256")
        or any(not _valid_sha256(value) for value in payload["source_sha256"].values())
        or not _valid_geometry_gauge_contract(
            payload.get("geometry_gauge_contract")
        )
        or payload.get("geometry_gauge_contract_sha256")
        != _hash_json(payload.get("geometry_gauge_contract"))
        or not payload.get("generator_config")
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("generator binding is invalid or contains learned dependencies")
    return True


def _composite_cache_contract_v3(generation_config, generator_binding):
    """Authenticate the closed pose-then-joint composite cache declaration."""
    config_is_composite = (
        isinstance(generation_config, dict)
        and generation_config.get("schema_version")
        == COMPOSITE_CURRICULUM_V3_SCHEMA
    )
    bound_config = generator_binding.get("generator_config", {})
    binding_is_composite = (
        isinstance(bound_config, dict)
        and bound_config.get("schema_version")
        == COMPOSITE_CURRICULUM_V3_SCHEMA
    )
    if not config_is_composite and not binding_is_composite:
        return None
    if not config_is_composite or not binding_is_composite:
        raise ValueError("composite cache config and generator binding disagree")
    config = _plain(generation_config)
    if config != _plain(bound_config):
        raise ValueError("composite cache generation config differs from its binding")

    labels = tuple(item[0] for item in COMPOSITE_COMPONENTS)
    component_configs = config.get("component_generation_configs", {})
    component_bindings = config.get("component_generator_bindings", {})
    component_counts = config.get("component_row_counts", {})
    expected_algorithms = [item[2] for item in COMPOSITE_COMPONENTS]
    if (
        tuple(component_configs) != labels
        or tuple(component_bindings) != labels
        or tuple(component_counts) != labels
        or config.get("generator_ids") != expected_algorithms
        or generator_binding.get("generator_ids") != sorted(expected_algorithms)
        or config.get("row_order_policy") != COMPOSITE_ROW_ORDER_POLICY
        or config.get("single_frozen_cache") is not True
        or any(
            config.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("composite cache membership, order, or dependency contract is invalid")

    merged_sources = {}
    normalized = []
    required_runner_psf = None
    for label, schema, algorithm in COMPOSITE_COMPONENTS:
        component_config = component_configs.get(label, {})
        component_binding = component_bindings.get(label, {})
        verify_generator_binding_v3(component_binding)
        count = component_counts.get(label)
        if (
            component_config.get("schema_version") != schema
            or component_config.get("algorithm") != algorithm
            or component_binding.get("generator_ids") != [algorithm]
            or component_binding.get("generator_config") != component_config
            or component_binding.get("geometry_gauge_contract")
            != generator_binding.get("geometry_gauge_contract")
            or component_config.get("prepared_context_sha256")
            != config.get("prepared_context_sha256")
            or component_config.get("support_index_sha256")
            != config.get("support_index_sha256")
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or component_config.get("row_count") != count
            or "finite_parent_generator_source_commit" not in component_config
            or not isinstance(component_config.get("required_runner_psf"), dict)
            or any(
                component_config.get(name) != []
                for name in (
                    "prior_model_weight_dependencies",
                    "prior_feature_dependencies",
                    "prior_pseudolabel_dependencies",
                )
            )
        ):
            raise ValueError(
                f"composite component {label!r} has a stale config, count, or binding"
            )
        if label == "nonidentity_joint_curriculum" and (
            component_config.get("joint_no_drop_policy") != JOINT_NO_DROP_POLICY
            or component_config.get("nonidentity_retry_exhaustion_censor_reason")
            != JOINT_FALLBACK_CENSOR_REASON
            or component_config.get("deformation_censor_statuses")
            != {
                "direct_success": JOINT_UNCENSORED_STATUS,
                "bounded_retry_fallback": JOINT_FALLBACK_CENSOR_STATUS,
                "marginal_support": JOINT_MARGINAL_CENSOR_STATUS,
            }
            or component_config.get("marginal_support_censor_reason")
            != JOINT_MARGINAL_CENSOR_REASON
            or component_config.get("fallback_attempt_index")
            != component_config.get("maximum_joint_rejection_attempts")
            or component_config.get("rejected_nonidentity_image_relabeling_allowed")
            is not False
            or not isinstance(component_config.get("fallback_seed_derivation"), str)
        ):
            raise ValueError("composite joint no-drop/censor contract is stale")
        if required_runner_psf is None:
            required_runner_psf = component_config["required_runner_psf"]
        elif component_config["required_runner_psf"] != required_runner_psf:
            raise ValueError("composite component finite-thickness PSF configs disagree")
        for name, digest in component_binding["source_sha256"].items():
            if name in merged_sources and merged_sources[name] != digest:
                raise ValueError("composite component source bindings disagree")
            merged_sources[name] = digest
        normalized.append(
            {
                "label": label,
                "schema": schema,
                "algorithm": algorithm,
                "config": component_config,
                "binding": component_binding,
                "count": count,
            }
        )
    if merged_sources != generator_binding.get("source_sha256"):
        raise ValueError("composite cache source binding is not the exact component union")
    return {
        "components": normalized,
        "total_row_count": sum(item["count"] for item in normalized),
    }


def _composite_component_at_v3(contract, row_index):
    index = int(row_index)
    if index < 0:
        raise ValueError("composite cache row index is invalid")
    for component in contract["components"]:
        if index < component["count"]:
            return component, index
        index -= component["count"]
    raise ValueError("composite cache exceeds its declared component counts")


def _verify_composite_seed_record_v3(contract, seed_record):
    expected = {
        "pose_root_seed_uint64": contract["components"][0]["config"][
            "root_seed_uint64"
        ],
        "joint_root_seed_uint64": contract["components"][1]["config"][
            "root_seed_uint64"
        ],
    }
    if not isinstance(seed_record, dict) or any(
        seed_record.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("composite cache seed record differs from component configs")
    return True


def _composite_row_receipts_v3(row, row_index, contract):
    component, local_index = _composite_component_at_v3(contract, row_index)
    config = component["config"]
    binding = component["binding"]
    upstream = row.get("upstream_reference", {})
    adapter = upstream.get("adapter_configuration", {})
    numeric = row.get("numeric_rng_provenance", {})
    sample_index = int(config["start_index"]) + local_index
    modes = config.get("trainable_modes", [])
    reflections = config.get("horizontal_representation_augmentation", [])
    sections_per_animal = config.get("sections_per_animal")
    if (
        not isinstance(adapter, dict)
        or not modes
        or not reflections
        or isinstance(sections_per_animal, bool)
        or not isinstance(sections_per_animal, int)
        or sections_per_animal < 1
    ):
        raise ValueError("composite component adapter contract is invalid")
    animal_index = sample_index // sections_per_animal
    identity_prefix = config["identity_prefix"]
    expected_adapter = {
        "root_seed": config["root_seed_uint64"],
        "sample_index": sample_index,
        "output_shape_h_w": config["output_shape_h_w"],
        "selected_mode": modes[sample_index % len(modes)],
        "reflection_state": reflections[
            (sample_index // len(modes)) % len(reflections)
        ],
        "animal_id": f"{identity_prefix}-animal-{animal_index:08d}",
        "specimen_id": f"{identity_prefix}-specimen-{animal_index:08d}",
        "experiment_id": f"{identity_prefix}-experiment-{animal_index:08d}",
        "synthetic_animal_id": f"{identity_prefix}-synthetic-animal-{animal_index:08d}",
        "section_id": f"{identity_prefix}-section-{sample_index:08d}",
        "split": config["split"],
        "stratum": config["stratum"],
        "margin_um": config["margin_u_v_um"],
        "minimum_brain_pixels": config["minimum_brain_pixels"],
        "maximum_rejection_attempts": config["maximum_rejection_attempts"],
        "finite_parent_generator_source_commit": config[
            "finite_parent_generator_source_commit"
        ],
    }
    if any(adapter.get(name) != value for name, value in expected_adapter.items()):
        raise ValueError("composite row adapter differs from its declared component config")
    if (
        upstream.get("schema_version") != component["schema"]
        or upstream.get("algorithm") != component["algorithm"]
        or upstream.get("implementation_source_sha256") != binding["source_sha256"]
        or upstream.get("prepared_context_sha256")
        != config["prepared_context_sha256"]
        or upstream.get("support_index_sha256") != config["support_index_sha256"]
        or row.get("selected_mode") != expected_adapter["selected_mode"]
        or row.get("reflection_state") != expected_adapter["reflection_state"]
        or any(
            row.get("lineage", {}).get(name) != expected_adapter[name]
            for name in (
                "animal_id",
                "specimen_id",
                "experiment_id",
                "synthetic_animal_id",
                "section_id",
                "split",
            )
        )
        or numeric.get("schema_version") != component["schema"]
        or numeric.get("root_seed_uint64") != config["root_seed_uint64"]
        or numeric.get("sample_index") != sample_index
    ):
        raise ValueError("composite row source, algorithm, order, or identity changed")

    if component["label"] == "identity_pose_curriculum":
        attempt_key = "plane_attempt_number"
        history_key = "plane_parent_rejection_history"
        maximum_attempts = config["maximum_parent_geometry_retries"]
    else:
        attempt_key = "joint_attempt_number"
        history_key = "joint_rejection_history"
        maximum_attempts = config["maximum_joint_rejection_attempts"]
        expected_adapter["maximum_joint_rejection_attempts"] = maximum_attempts
        amplitude_cycle = config.get("amplitude_band_cycle", [])
        if not amplitude_cycle:
            raise ValueError("composite joint amplitude cycle is empty")
        expected_amplitude = amplitude_cycle[sample_index % len(amplitude_cycle)]
        if (
            adapter.get("amplitude_band") != expected_amplitude
            or upstream.get("deformation_amplitude_band") != expected_amplitude
        ):
            raise ValueError("composite joint row amplitude differs from its config")
    if any(adapter.get(name) != value for name, value in expected_adapter.items()):
        raise ValueError("composite row adapter differs from its declared component config")
    attempt = adapter.get(attempt_key)
    history = adapter.get(history_key)
    identity_fallback = bool(
        component["label"] == "nonidentity_joint_curriculum"
        and adapter.get("identity_g1_pose_only_fallback") is True
    )
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 0
        or (
            attempt != maximum_attempts
            if identity_fallback
            else attempt >= maximum_attempts
        )
        or not isinstance(history, list)
        or len(history) != attempt
        or upstream.get(history_key) != history
        or numeric.get(attempt_key) != attempt
    ):
        raise ValueError("composite row retry provenance differs from its config")
    if component["label"] == "nonidentity_joint_curriculum":
        parent_identity = adapter.get("finite_parent_identity")
        dense_support = adapter.get("effective_dense_support")
        censoring = upstream.get("deformation_censoring_contract", {})
        support = upstream.get("support_supervision_contract", {})
        expected_parent_request = {
            "logical_root_seed_uint64": config["root_seed_uint64"],
            "logical_sample_index": sample_index,
            "derived_plane_sample_index": numeric.get(
                "derived_plane_sample_index"
            ),
            "finite_render_seed_uint64": numeric.get(
                "finite_render_seed_uint64"
            ),
            "lineage_ids": {
                name: expected_adapter[name]
                for name in (
                    "animal_id",
                    "specimen_id",
                    "experiment_id",
                    "synthetic_animal_id",
                    "section_id",
                )
            },
        }
        if (
            not isinstance(parent_identity, dict)
            or parent_identity != upstream.get("finite_parent_identity")
            or any(
                parent_identity.get(name) != value
                for name, value in expected_parent_request.items()
            )
            or parent_identity.get("finite_parent_root_seed_uint64")
            != parent_identity.get("finite_render_seed_uint64")
            or parent_identity.get("finite_parent_sample_index")
            != parent_identity.get("derived_plane_sample_index")
            or parent_identity.get("derived_plane_sample_index")
            != numeric.get("derived_plane_sample_index")
            or parent_identity.get("finite_render_seed_uint64")
            != numeric.get("finite_render_seed_uint64")
            or parent_identity.get("finite_plane_render_id")
            != upstream.get("finite_plane_render_id")
            or parent_identity.get("finite_render_receipt_sha256")
            != upstream.get("finite_render_receipt_sha256")
            or parent_identity.get("finite_parent_provenance_sha256")
            != upstream.get("finite_parent_provenance_sha256")
            or not isinstance(dense_support, dict)
            or dense_support != upstream.get("deformation_censoring_contract", {}).get(
                "effective_dense_support"
            )
            or dense_support != support.get("effective_dense_support")
            or adapter.get("requested_deformation_amplitude_band")
            != adapter.get("amplitude_band")
            or upstream.get("requested_deformation_amplitude_band")
            != adapter.get("amplitude_band")
            or any(
                entry.get("requested_deformation_amplitude_band")
                != adapter.get("amplitude_band")
                or entry.get("finite_parent_identity") != parent_identity
                or entry.get("finite_parent_request") != expected_parent_request
                for entry in history
            )
            or adapter.get("deformation_censor_status")
            != censoring.get("status")
            or adapter.get("deformation_censor_reason")
            != censoring.get("reason")
            or adapter.get("fallback_attempt_number")
            != censoring.get("fallback_attempt_number")
            or adapter.get("fallback_synthetic_seed_uint64")
            != censoring.get("fallback_synthetic_seed_uint64")
            or adapter.get("identity_g1_pose_only_fallback")
            is not identity_fallback
            or censoring.get("rejected_nonidentity_image_relabeling_allowed")
            is not False
            or bool(numeric.get("identity_g1_pose_only_fallback", False))
            is not identity_fallback
        ):
            raise ValueError("composite joint parent/censor/dense-support provenance changed")
        support_identifiable = bool(dense_support.get("raster_support_identifiable"))
        expected_pose_weight = float(support_identifiable)
        expected_dense_weight = float(support_identifiable and not identity_fallback)
        if (
            support.get("point_pose_supervision_weight") != expected_pose_weight
            or support.get("dense_deformation_supervision_weight")
            != expected_dense_weight
            or dense_support.get(
                "effective_dense_deformation_supervision_weight"
            )
            != expected_dense_weight
        ):
            raise ValueError("composite joint supervision weights violate censor policy")
        if identity_fallback:
            expected_seed = numeric.get("synthetic_seed_uint64")
            pullback = np.asarray(
                row["arrays"]["truth_section_pullback_map_yx_px_float64"]
            )
            height, width = pullback.shape[:2]
            y, x = np.mgrid[:height, :width]
            identity_pullback = np.stack((y, x), axis=-1).astype(
                pullback.dtype
            )
            if (
                adapter.get("fallback_attempt_number") != maximum_attempts
                or adapter.get("fallback_synthetic_seed_uint64") != expected_seed
                or numeric.get("fallback_attempt_number") != maximum_attempts
                or numeric.get("fallback_synthetic_seed_uint64") != expected_seed
                or censoring.get("status") != JOINT_FALLBACK_CENSOR_STATUS
                or censoring.get("reason") != JOINT_FALLBACK_CENSOR_REASON
                or censoring.get("fresh_identity_g1_realization") is not True
                or upstream.get("selected_g1_accepted_attempt", {}).get(
                    "identity_path"
                )
                is not True
                or np.count_nonzero(
                    row["arrays"][
                        "truth_section_pullback_stationary_velocity_yx_px_float64"
                    ]
                )
                != 0
                or not np.array_equal(pullback, identity_pullback)
            ):
                raise ValueError("composite joint fallback provenance changed")
        elif support_identifiable:
            if (
                censoring.get("status") != JOINT_UNCENSORED_STATUS
                or censoring.get("reason") is not None
                or adapter.get("fallback_attempt_number") is not None
                or adapter.get("fallback_synthetic_seed_uint64") is not None
                or upstream.get("selected_g1_accepted_attempt", {}).get(
                    "identity_path"
                )
                is not False
            ):
                raise ValueError("composite joint direct-success censor state changed")
        elif (
            censoring.get("status") != JOINT_MARGINAL_CENSOR_STATUS
            or censoring.get("reason") != JOINT_MARGINAL_CENSOR_REASON
            or support.get("point_pose_supervision_weight") != 0.0
            or support.get("dense_deformation_supervision_weight") != 0.0
        ):
            raise ValueError("composite joint marginal support must remain 0/0")
    return {
        "composite_component": component["label"],
        "upstream_algorithm": component["algorithm"],
        "component_generator_binding_receipt_sha256": binding["receipt_sha256"],
        "component_generation_config_receipt_sha256": _hash_json(config),
        "implementation_source_sha256_receipt": _hash_json(binding["source_sha256"]),
        "adapter_configuration_receipt_sha256": _hash_json(adapter),
    }


def verify_cached_training_row_v3(row, *, expected_geometry_gauge_contract=None):
    lineage = row.get("lineage", {})
    split = str(lineage.get("split", "")).lower()
    gauge_reference = row.get("deformation_pose_gauge_reference", {})
    required_lineage = (
        "animal_id",
        "specimen_id",
        "experiment_id",
        "synthetic_animal_id",
        "section_id",
        "split",
    )
    if (
        not _valid_gauge_reference(gauge_reference)
    ):
        raise ValueError("training row deformation-pose gauge reference is invalid or stale")
    if expected_geometry_gauge_contract is not None and any(
        gauge_reference.get(name) != value
        for name, value in expected_geometry_gauge_contract.items()
    ):
        raise ValueError("training row uses a stale or mismatched deformation-pose gauge")
    if (
        row.get("schema_version") != training_row_v3.TRAINING_ROW_V3_SCHEMA
        or any(token in split for token in FORBIDDEN_SPLIT_TOKENS)
        or any(name not in lineage or lineage[name] in (None, "") for name in required_lineage)
        or any(
            row.get(name) != []
            for name in (
                "prior_model_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
        or set(row.get("arrays", {})) != training_row_v3._ARRAY_KEYS
        or row.get("array_receipts")
        != {
            name: acquisition_v2._array_receipt(value)
            for name, value in row.get("arrays", {}).items()
        }
        or row.get("receipt_sha256")
        != acquisition_v2._payload_sha256(training_row_v3.training_row_receipt_v3(row))
    ):
        raise ValueError("training row is unauthenticated, non-development, or learned-dependent")
    return True


def _manifest_payload(manifest):
    return {key: value for key, value in manifest.items() if key != "receipt_sha256"}


def _with_manifest_receipt(payload):
    payload = _plain(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def initialize_training_row_cache_v3(
    cache_directory,
    *,
    generator_binding,
    generation_config,
    seed_record,
):
    root = _i_path(cache_directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("row-cache directory must be empty before initialization")
    verify_generator_binding_v3(generator_binding)
    if not isinstance(generation_config, dict) or not generation_config:
        raise ValueError("the exact generation config is required")
    if not isinstance(seed_record, dict) or not seed_record:
        raise ValueError("the exact generation seeds are required")
    _canonical_json(generation_config)
    _canonical_json(seed_record)
    composite_contract = _composite_cache_contract_v3(
        generation_config, generator_binding
    )
    if composite_contract is not None:
        _verify_composite_seed_record_v3(composite_contract, seed_record)
    root.mkdir(parents=True, exist_ok=True)
    (root / "rows").mkdir(exist_ok=True)
    payload = {
        "schema_version": ROW_CACHE_V3_SCHEMA,
        "data_role": DEVELOPMENT_DATA_ROLE,
        "generator_binding": _plain(generator_binding),
        "generation_config": _plain(generation_config),
        "seed_record": _plain(seed_record),
        "status": OPEN_CACHE_STATUS,
        "freeze_audit": None,
        "row_count": 0,
        "rows": [],
        "forbidden_sources": [
            "public benchmark",
            "validation animals",
            "external-validation animals",
            "final-test animals",
        ],
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    manifest = _with_manifest_receipt(payload)
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def load_training_row_cache_manifest_v3(
    cache_directory,
    *,
    expected_generator_binding=None,
    expected_receipt_sha256=None,
):
    root = _i_path(cache_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = _manifest_payload(manifest)
    rows = payload.get("rows", [])
    row_ids = [item.get("training_row_id") for item in rows]
    receipts = [item.get("training_row_receipt_sha256") for item in rows]
    if (
        manifest.get("receipt_sha256") != _hash_json(payload)
        or payload.get("schema_version") != ROW_CACHE_V3_SCHEMA
        or payload.get("data_role") != DEVELOPMENT_DATA_ROLE
        or payload.get("status") not in (OPEN_CACHE_STATUS, FROZEN_CACHE_STATUS)
        or (
            payload.get("status") == OPEN_CACHE_STATUS
            and payload.get("freeze_audit") is not None
        )
        or (
            payload.get("status") == FROZEN_CACHE_STATUS
            and payload.get("freeze_audit")
            != {
                "row_count": len(rows),
                "ordered_training_row_receipts_sha256": _hash_json(receipts),
                "all_rows_authenticated": True,
            }
        )
        or payload.get("row_count") != len(rows)
        or len(set(row_ids)) != len(row_ids)
        or len(set(receipts)) != len(receipts)
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("row-cache manifest failed authentication")
    verify_generator_binding_v3(payload.get("generator_binding", {}))
    composite_contract = _composite_cache_contract_v3(
        payload.get("generation_config", {}), payload["generator_binding"]
    )
    if composite_contract is not None:
        _verify_composite_seed_record_v3(
            composite_contract, payload.get("seed_record", {})
        )
    if expected_generator_binding is not None and payload["generator_binding"] != _plain(expected_generator_binding):
        raise ValueError("row-cache generator binding differs")
    if expected_receipt_sha256 is not None and manifest["receipt_sha256"] != expected_receipt_sha256:
        raise ValueError("row-cache manifest receipt differs")
    for index, record in enumerate(rows):
        lineage = record.get("lineage", {})
        gauge_reference = record.get("deformation_pose_gauge_reference", {})
        if (
            record.get("row_index") != index
            or not _valid_gauge_reference(gauge_reference)
            or any(
                gauge_reference.get(name) != value
                for name, value in payload["generator_binding"][
                    "geometry_gauge_contract"
                ].items()
            )
            or any(token in str(lineage.get("split", "")).lower() for token in FORBIDDEN_SPLIT_TOKENS)
            or any(
                record.get(name) in (None, "")
                for name in (
                    "training_row_id",
                    "training_row_receipt_sha256",
                    "metadata_relative_path",
                    "metadata_file_sha256",
                    "arrays_relative_path",
                    "arrays_file_sha256",
                )
            )
            or any(
                lineage.get(name) in (None, "")
                for name in (
                    "animal_id",
                    "specimen_id",
                    "experiment_id",
                    "synthetic_animal_id",
                    "section_id",
                    "split",
                )
            )
        ):
            raise ValueError("row-cache record is malformed or non-development")
        for name in ("metadata_relative_path", "arrays_relative_path"):
            path = (root / record[name]).resolve()
            if root not in path.parents:
                raise ValueError("row-cache record escapes its I:-drive cache")
        if composite_contract is not None:
            expected_component, _ = _composite_component_at_v3(
                composite_contract, index
            )
            if (
                record.get("composite_component") != expected_component["label"]
                or record.get("upstream_algorithm")
                != expected_component["algorithm"]
                or record.get("component_generator_binding_receipt_sha256")
                != expected_component["binding"]["receipt_sha256"]
                or record.get("component_generation_config_receipt_sha256")
                != _hash_json(expected_component["config"])
                or record.get("implementation_source_sha256_receipt")
                != _hash_json(expected_component["binding"]["source_sha256"])
                or not _valid_sha256(
                    record.get("adapter_configuration_receipt_sha256")
                )
            ):
                raise ValueError("composite row record membership or binding changed")
    if composite_contract is not None and (
        len(rows) > composite_contract["total_row_count"]
        or (
            payload["status"] == FROZEN_CACHE_STATUS
            and len(rows) != composite_contract["total_row_count"]
        )
    ):
        raise ValueError("composite cache component counts differ from the declaration")
    return manifest


def append_training_rows_v3(cache_directory, rows):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v3(root)
    if manifest["status"] != OPEN_CACHE_STATUS:
        raise ValueError("frozen row caches cannot be appended")
    composite_contract = _composite_cache_contract_v3(
        manifest["generation_config"], manifest["generator_binding"]
    )
    existing = {
        record["training_row_id"]: record for record in manifest["rows"]
    }
    seen_receipts = {
        row_id: record["training_row_receipt_sha256"]
        for row_id, record in existing.items()
    }
    pending = []
    for row in rows:
        verify_cached_training_row_v3(
            row,
            expected_geometry_gauge_contract=manifest["generator_binding"][
                "geometry_gauge_contract"
            ],
        )
        row_id = row["training_row_id"]
        receipt = row["receipt_sha256"]
        if row_id in seen_receipts:
            if seen_receipts[row_id] != receipt:
                raise ValueError("a cached training-row ID was reused with different content")
            if composite_contract is not None and row_id in existing:
                expected = _composite_row_receipts_v3(
                    row, existing[row_id]["row_index"], composite_contract
                )
                if any(
                    existing[row_id].get(name) != value
                    for name, value in expected.items()
                ):
                    raise ValueError("composite cached row record differs from its row")
            continue
        row_index = len(manifest["rows"]) + len(pending)
        composite_receipts = (
            _composite_row_receipts_v3(row, row_index, composite_contract)
            if composite_contract is not None
            else {}
        )
        pending.append((row, composite_receipts))
        seen_receipts[row_id] = receipt

    appended = []
    for row, composite_receipts in pending:
        row_id = row["training_row_id"]
        receipt = row["receipt_sha256"]
        stem = receipt
        metadata_relative = f"rows/{stem}.json"
        arrays_relative = f"rows/{stem}.npz"
        metadata = {key: value for key, value in row.items() if key != "arrays"}
        _atomic_npz(root / arrays_relative, row["arrays"])
        _atomic_json(root / metadata_relative, metadata)
        lineage = row["lineage"]
        record = {
            "row_index": len(manifest["rows"]) + len(appended),
            "training_row_id": row_id,
            "training_row_receipt_sha256": receipt,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "lineage": {
                name: _plain(lineage[name])
                for name in (
                    "animal_id",
                    "specimen_id",
                    "experiment_id",
                    "synthetic_animal_id",
                    "section_id",
                    "split",
                )
            },
            "selected_mode": row["selected_mode"],
            "reflection_state": row["reflection_state"],
            "deformation_pose_gauge_reference": _plain(
                row["deformation_pose_gauge_reference"]
            ),
            "metadata_relative_path": metadata_relative,
            "metadata_file_sha256": _file_sha256(root / metadata_relative),
            "arrays_relative_path": arrays_relative,
            "arrays_file_sha256": _file_sha256(root / arrays_relative),
            **composite_receipts,
        }
        appended.append(record)
        existing[row_id] = record
    if appended:
        payload = _manifest_payload(manifest)
        payload["rows"] = [*payload["rows"], *appended]
        payload["row_count"] = len(payload["rows"])
        manifest = _with_manifest_receipt(payload)
        _atomic_json(root / "manifest.json", manifest)
    return manifest


def _load_record(root, record, geometry_gauge_contract):
    metadata_path = root / record["metadata_relative_path"]
    arrays_path = root / record["arrays_relative_path"]
    if (
        _file_sha256(metadata_path) != record["metadata_file_sha256"]
        or _file_sha256(arrays_path) != record["arrays_file_sha256"]
    ):
        raise ValueError("cached training-row file hash differs")
    row = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as stored:
        row["arrays"] = {
            name: np.ascontiguousarray(stored[name]) for name in stored.files
        }
    verify_cached_training_row_v3(
        row, expected_geometry_gauge_contract=geometry_gauge_contract
    )
    if (
        row["training_row_id"] != record["training_row_id"]
        or row["receipt_sha256"] != record["training_row_receipt_sha256"]
        or row["synthetic_realization_id"] != record["synthetic_realization_id"]
        or row["deformation_pose_gauge_reference"]
        != record["deformation_pose_gauge_reference"]
        or {name: row["lineage"][name] for name in record["lineage"]}
        != record["lineage"]
        or row["selected_mode"] != record["selected_mode"]
        or row["reflection_state"] != record["reflection_state"]
    ):
        raise ValueError("cached row differs from its manifest identity")
    return row


def load_training_rows_v3(
    cache_directory,
    indices=None,
    *,
    expected_manifest_receipt_sha256=None,
):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v3(
        root, expected_receipt_sha256=expected_manifest_receipt_sha256
    )
    if indices is None:
        indices = range(manifest["row_count"])
    selected = [int(index) for index in indices]
    if any(index < 0 or index >= manifest["row_count"] for index in selected):
        raise IndexError("row-cache index is out of range")
    geometry_gauge_contract = manifest["generator_binding"][
        "geometry_gauge_contract"
    ]
    composite_contract = _composite_cache_contract_v3(
        manifest["generation_config"], manifest["generator_binding"]
    )
    loaded = []
    for index in selected:
        row = _load_record(root, manifest["rows"][index], geometry_gauge_contract)
        if composite_contract is not None:
            expected = _composite_row_receipts_v3(row, index, composite_contract)
            if any(
                manifest["rows"][index].get(name) != value
                for name, value in expected.items()
            ):
                raise ValueError("composite cached row differs from its binding receipts")
        loaded.append(row)
    return loaded


def audit_training_row_cache_v3(cache_directory):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v3(root)
    geometry_gauge_contract = manifest["generator_binding"][
        "geometry_gauge_contract"
    ]
    composite_contract = _composite_cache_contract_v3(
        manifest["generation_config"], manifest["generator_binding"]
    )
    if (
        composite_contract is not None
        and manifest["row_count"] != composite_contract["total_row_count"]
    ):
        raise ValueError("composite cache component counts differ from the declaration")
    for record in manifest["rows"]:
        row = _load_record(root, record, geometry_gauge_contract)
        if composite_contract is not None:
            expected = _composite_row_receipts_v3(
                row, record["row_index"], composite_contract
            )
            if any(
                record.get(name) != value for name, value in expected.items()
            ):
                raise ValueError("composite cached row differs from its binding receipts")
    return {
        "schema_version": ROW_CACHE_V3_SCHEMA,
        "manifest_receipt_sha256": manifest["receipt_sha256"],
        "row_count": manifest["row_count"],
        "all_rows_authenticated": True,
        "learned_dependencies": [],
        "data_role": DEVELOPMENT_DATA_ROLE,
    }


def freeze_training_row_cache_v3(cache_directory):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v3(root)
    if manifest["status"] == FROZEN_CACHE_STATUS:
        audit_training_row_cache_v3(root)
        return manifest
    audit_training_row_cache_v3(root)
    payload = _manifest_payload(manifest)
    payload["status"] = FROZEN_CACHE_STATUS
    payload["freeze_audit"] = {
        "row_count": manifest["row_count"],
        "ordered_training_row_receipts_sha256": _hash_json(
            [record["training_row_receipt_sha256"] for record in manifest["rows"]]
        ),
        "all_rows_authenticated": True,
    }
    frozen = _with_manifest_receipt(payload)
    _atomic_json(root / "manifest.json", frozen)
    return frozen
