"""Strict authenticated I:-drive cache for finite-thickness v4 training rows."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge_v4
import training.arbitrary_plane_psf_v4 as psf_v4


ROW_CACHE_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-row-cache/v4"
GENERATOR_BINDING_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-generator-binding/v4"
)
FINITE_PSF_CACHE_RUN_V4_SCHEMA = (
    "anatomy-tracker.finite-psf-cache-run/v4"
)
ROW_CACHE_AUDIT_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-row-cache-audit/v4"
GENERATION_LINEAGE_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-cache-generation-lineage/v4"
)
EXPECTED_FINITE_PSF_CAPABILITY_SHA256 = (
    "bcd6441a685e902fb5b59e85bb7003ef3261207d906a0b9390d4a219c3ae3d3e"
)
DEVELOPMENT_DATA_ROLE = "development-training"
OPEN_CACHE_STATUS = "OPEN"
FROZEN_CACHE_STATUS = "FROZEN"
FORBIDDEN_SPLIT_TOKENS = (
    "test",
    "benchmark",
    "qualification",
    "external",
    "validation",
)
LEARNED_DEPENDENCY_KEYS = {
    "learned_dependencies",
    "prior_model_dependencies",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
}
ROW_LINEAGE_KEYS = {
    "animal_id",
    "specimen_id",
    "experiment_id",
    "synthetic_animal_id",
    "section_id",
    "split",
}
GENERATION_LINEAGE_REQUIRED_KEYS = {
    "schema_version",
    "generation_run_id",
    "source_commit",
    "split",
}
RUN_CONTRACT_KEYS = {
    "schema_version",
    "training_row_schema_version",
    "finite_psf_capability_sha256",
    "render_mode",
    "axial_sample_count",
    "nominal_cut_thickness_policy",
    "receipt_sha256",
}
GENERATOR_BINDING_KEYS = {
    "schema_version",
    "training_row_schema_version",
    "finite_psf_capability",
    "finite_psf_capability_sha256",
    "finite_psf_run_contract",
    "generator_ids",
    "source_sha256",
    "geometry_gauge_contract",
    "geometry_gauge_contract_sha256",
    "generation_config",
    "generation_config_sha256",
    "seed_record",
    "seed_record_sha256",
    "generation_lineage",
    "generation_lineage_sha256",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "receipt_sha256",
}
MANIFEST_KEYS = {
    "schema_version",
    "data_role",
    "training_row_schema_version",
    "finite_psf_capability",
    "finite_psf_capability_sha256",
    "finite_psf_run_contract",
    "generator_binding",
    "generation_config",
    "seed_record",
    "generation_lineage",
    "status",
    "freeze_audit",
    "row_count",
    "rows",
    "forbidden_sources",
    "prior_model_weight_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "receipt_sha256",
}
ROW_RECORD_KEYS = {
    "row_index",
    "training_row_schema_version",
    "training_row_id",
    "training_row_receipt_sha256",
    "synthetic_realization_id",
    "lineage",
    "selected_mode",
    "reflection_state",
    "finite_psf_capability_sha256",
    "finite_psf_sha256",
    "slab_observation_v4_receipt_sha256",
    "nominal_cut_thickness_um",
    "render_mode",
    "axial_sample_count",
    "metadata_relative_path",
    "metadata_file_sha256",
    "arrays_relative_path",
    "arrays_file_sha256",
}
FREEZE_AUDIT_KEYS = {
    "row_count",
    "ordered_training_row_receipts_sha256",
    "ordered_finite_psf_sha256",
    "ordered_slab_observation_v4_receipts_sha256",
    "ordered_nominal_cut_thickness_um_sha256",
    "all_rows_authenticated",
    "learned_dependencies",
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
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


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _valid_git_commit(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and not (set(value) - set("0123456789abcdef"))
    )


def _i_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("finite-thickness v4 row caches must be stored only on I:")
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


def _assert_no_learned_dependencies(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LEARNED_DEPENDENCY_KEYS and item != []:
                raise ValueError("finite v4 caches cannot bind learned dependencies")
            _assert_no_learned_dependencies(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_learned_dependencies(item)


def _verify_development_split(split):
    normalized = str(split).strip().lower()
    if not normalized or any(token in normalized for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("finite v4 caches are restricted to development splits")


def _finite_psf_capability_v4():
    capability = psf_v4.finite_psf_model_capability_v4()
    psf_v4.verify_finite_psf_model_capability_v4(capability)
    if capability["receipt_sha256"] != EXPECTED_FINITE_PSF_CAPABILITY_SHA256:
        raise ValueError("finite-PSF capability changed; a new cache schema is required")
    return capability


def make_finite_psf_cache_run_contract_v4(render_mode):
    """Declare one homogeneous finite-PSF family for an entire v4 cache."""
    capability = _finite_psf_capability_v4()
    if render_mode == "finite_boxcar":
        sample_count = psf_v4.PRODUCTION_AXIAL_SAMPLE_COUNT
        thickness_policy = {
            "kind": "authenticated-per-row-closed-interval",
            "minimum_um": psf_v4.PRODUCTION_THICKNESS_RANGE_UM[0],
            "maximum_um": psf_v4.PRODUCTION_THICKNESS_RANGE_UM[1],
        }
    elif render_mode == "centre_plane_ablation":
        sample_count = 1
        thickness_policy = {"kind": "exact", "nominal_cut_thickness_um": 0.0}
    else:
        raise ValueError("v4 cache render mode must be finite_boxcar or centre_plane_ablation")
    payload = {
        "schema_version": FINITE_PSF_CACHE_RUN_V4_SCHEMA,
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "finite_psf_capability_sha256": capability["receipt_sha256"],
        "render_mode": render_mode,
        "axial_sample_count": sample_count,
        "nominal_cut_thickness_policy": thickness_policy,
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def verify_finite_psf_cache_run_contract_v4(contract):
    if not isinstance(contract, dict) or set(contract) != RUN_CONTRACT_KEYS:
        raise ValueError("finite-PSF cache-run contract fields changed")
    expected = make_finite_psf_cache_run_contract_v4(contract.get("render_mode"))
    if contract != expected:
        raise ValueError("finite-PSF cache-run contract changed")
    return True


def _verify_generation_config(config):
    if (
        not isinstance(config, dict)
        or not config
        or not isinstance(config.get("schema_version"), str)
        or not config["schema_version"]
        or not isinstance(config.get("algorithm"), str)
        or not config["algorithm"]
        or isinstance(config.get("row_count"), bool)
        or not isinstance(config.get("row_count"), int)
        or config["row_count"] < 1
    ):
        raise ValueError(
            "exact versioned generation config, algorithm, and positive row count are required"
        )
    _canonical_json(config)
    _assert_no_learned_dependencies(config)


def _verify_seed_record(seed_record):
    if not isinstance(seed_record, dict) or not seed_record:
        raise ValueError("exact nonempty generation seed record is required")
    _canonical_json(seed_record)


def _verify_generation_lineage(lineage):
    if (
        not isinstance(lineage, dict)
        or not GENERATION_LINEAGE_REQUIRED_KEYS.issubset(lineage)
        or lineage.get("schema_version") != GENERATION_LINEAGE_V4_SCHEMA
        or not isinstance(lineage.get("generation_run_id"), str)
        or not lineage["generation_run_id"]
        or not _valid_git_commit(lineage.get("source_commit"))
    ):
        raise ValueError("exact v4 generation run, commit, and split lineage are required")
    _verify_development_split(lineage["split"])
    _canonical_json(lineage)
    _assert_no_learned_dependencies(lineage)


def make_generator_binding_v4(
    *,
    generator_ids,
    source_sha256,
    geometry_gauge_contract,
    generation_config,
    seed_record,
    generation_lineage,
    finite_psf_run_contract,
):
    """Bind every executable, numeric, lineage, and finite-PSF input to a cache."""
    if (
        not isinstance(generator_ids, (list, tuple, set))
        or any(not isinstance(value, str) or not value for value in generator_ids)
        or not isinstance(source_sha256, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in source_sha256.items()
        )
        or not isinstance(geometry_gauge_contract, dict)
    ):
        raise ValueError("generator IDs, source hashes, and gauge contract are required")
    generator_ids = sorted(set(generator_ids))
    source_sha256 = {
        key: value for key, value in sorted(source_sha256.items())
    }
    _verify_generation_config(generation_config)
    _verify_seed_record(seed_record)
    _verify_generation_lineage(generation_lineage)
    verify_finite_psf_cache_run_contract_v4(finite_psf_run_contract)
    capability = _finite_psf_capability_v4()
    if (
        not generator_ids
        or not source_sha256
        or any(not _valid_sha256(value) for value in source_sha256.values())
        or geometry_gauge_contract
        != deformation_gauge_v4.direct_deformation_target_contract_v4()
    ):
        raise ValueError("generator IDs, source hashes, or direct gauge contract are invalid")
    payload = {
        "schema_version": GENERATOR_BINDING_V4_SCHEMA,
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "finite_psf_capability": capability,
        "finite_psf_capability_sha256": capability["receipt_sha256"],
        "finite_psf_run_contract": _plain(finite_psf_run_contract),
        "generator_ids": generator_ids,
        "source_sha256": source_sha256,
        "geometry_gauge_contract": _plain(geometry_gauge_contract),
        "geometry_gauge_contract_sha256": _hash_json(geometry_gauge_contract),
        "generation_config": _plain(generation_config),
        "generation_config_sha256": _hash_json(generation_config),
        "seed_record": _plain(seed_record),
        "seed_record_sha256": _hash_json(seed_record),
        "generation_lineage": _plain(generation_lineage),
        "generation_lineage_sha256": _hash_json(generation_lineage),
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def verify_generator_binding_v4(binding):
    if not isinstance(binding, dict) or set(binding) != GENERATOR_BINDING_KEYS:
        raise ValueError("finite v4 generator-binding fields changed")
    payload = {key: value for key, value in binding.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload.get("generator_ids"), list)
        or any(not isinstance(value, str) or not value for value in payload["generator_ids"])
        or not isinstance(payload.get("source_sha256"), dict)
        or not isinstance(payload.get("geometry_gauge_contract"), dict)
    ):
        raise ValueError("finite v4 generator binding is malformed")
    capability = _finite_psf_capability_v4()
    _verify_generation_config(payload.get("generation_config"))
    _verify_seed_record(payload.get("seed_record"))
    _verify_generation_lineage(payload.get("generation_lineage"))
    verify_finite_psf_cache_run_contract_v4(payload.get("finite_psf_run_contract"))
    if (
        binding["receipt_sha256"] != _hash_json(payload)
        or payload.get("schema_version") != GENERATOR_BINDING_V4_SCHEMA
        or payload.get("training_row_schema_version") != psf_v4.TRAINING_ROW_V4_SCHEMA
        or payload.get("finite_psf_capability") != capability
        or payload.get("finite_psf_capability_sha256")
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or not payload.get("generator_ids")
        or payload["generator_ids"] != sorted(set(payload["generator_ids"]))
        or not payload.get("source_sha256")
        or any(not _valid_sha256(value) for value in payload["source_sha256"].values())
        or payload.get("geometry_gauge_contract")
        != deformation_gauge_v4.direct_deformation_target_contract_v4()
        or payload.get("geometry_gauge_contract_sha256")
        != _hash_json(payload["geometry_gauge_contract"])
        or payload.get("generation_config_sha256")
        != _hash_json(payload["generation_config"])
        or payload.get("seed_record_sha256") != _hash_json(payload["seed_record"])
        or payload.get("generation_lineage_sha256")
        != _hash_json(payload["generation_lineage"])
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("finite v4 generator binding is invalid or stale")
    _assert_no_learned_dependencies(payload)
    return True


def _manifest_payload(manifest):
    return {key: value for key, value in manifest.items() if key != "receipt_sha256"}


def _with_manifest_receipt(payload):
    payload = _plain(payload)
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _freeze_audit(manifest):
    rows = manifest["rows"]
    return {
        "row_count": len(rows),
        "ordered_training_row_receipts_sha256": _hash_json(
            [record["training_row_receipt_sha256"] for record in rows]
        ),
        "ordered_finite_psf_sha256": _hash_json(
            [record["finite_psf_sha256"] for record in rows]
        ),
        "ordered_slab_observation_v4_receipts_sha256": _hash_json(
            [record["slab_observation_v4_receipt_sha256"] for record in rows]
        ),
        "ordered_nominal_cut_thickness_um_sha256": _hash_json(
            [record["nominal_cut_thickness_um"] for record in rows]
        ),
        "all_rows_authenticated": True,
        "learned_dependencies": [],
    }


def initialize_training_row_cache_v4(cache_directory, *, generator_binding):
    root = _i_path(cache_directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("v4 row-cache directory must be empty before initialization")
    verify_generator_binding_v4(generator_binding)
    root.mkdir(parents=True, exist_ok=True)
    (root / "rows").mkdir(exist_ok=True)
    payload = {
        "schema_version": ROW_CACHE_V4_SCHEMA,
        "data_role": DEVELOPMENT_DATA_ROLE,
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "finite_psf_capability": _plain(generator_binding["finite_psf_capability"]),
        "finite_psf_capability_sha256": EXPECTED_FINITE_PSF_CAPABILITY_SHA256,
        "finite_psf_run_contract": _plain(
            generator_binding["finite_psf_run_contract"]
        ),
        "generator_binding": _plain(generator_binding),
        "generation_config": _plain(generator_binding["generation_config"]),
        "seed_record": _plain(generator_binding["seed_record"]),
        "generation_lineage": _plain(generator_binding["generation_lineage"]),
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


def _verify_row_lineage(lineage, generation_lineage):
    if (
        not isinstance(lineage, dict)
        or not ROW_LINEAGE_KEYS.issubset(lineage)
        or any(lineage.get(name) in (None, "") for name in ROW_LINEAGE_KEYS)
    ):
        raise ValueError("v4 training row is missing animal/specimen/experiment lineage")
    _verify_development_split(lineage["split"])
    if lineage["split"] != generation_lineage["split"]:
        raise ValueError("v4 row split differs from cache generation lineage")
    _canonical_json(lineage)


def _verify_row_against_run(row, run_contract, generation_lineage, generator_binding):
    if row.get("schema_version") != psf_v4.TRAINING_ROW_V4_SCHEMA:
        raise ValueError("finite v4 caches accept only authenticated training-row/v4 rows")
    psf_v4.verify_training_row_v4(
        row, capability=_finite_psf_capability_v4()
    )
    _verify_row_lineage(row.get("lineage"), generation_lineage)
    _assert_no_learned_dependencies(row)
    if any(
        row.get(name) != []
        for name in (
            "prior_model_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        )
    ):
        raise ValueError("finite v4 training rows cannot use learned dependencies")
    if not _valid_sha256(row.get("training_row_id")) or not _valid_sha256(
        row.get("synthetic_realization_id")
    ):
        raise ValueError("finite v4 training row identities must be SHA-256 values")
    expected_row_id = acquisition_v2._payload_sha256(
        {
            "domain": psf_v4.TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": row["finite_psf_contract"][
                "finite_psf_sha256"
            ],
            "slab_observation_v4_receipt_sha256": row[
                "finite_psf_contract"
            ]["slab_observation_v4_receipt_sha256"],
        }
    )
    if row["training_row_id"] != expected_row_id:
        raise ValueError("finite v4 training-row identity differs from its exact inputs")
    gauge_reference = row.get("deformation_pose_gauge_reference", {})
    gauge_contract = generator_binding["geometry_gauge_contract"]
    if not isinstance(gauge_reference, dict) or any(
        gauge_reference.get(name) != value for name, value in gauge_contract.items()
    ) or any(
        not _valid_sha256(gauge_reference.get(name))
        for name in ("direct_deformation_target_id", "receipt_sha256")
    ):
        raise ValueError("finite v4 row direct deformation gauge differs from binding")
    upstream = row.get("upstream_reference", {})
    implementation_sources = upstream.get("implementation_source_sha256")
    if (
        upstream.get("algorithm") not in generator_binding["generator_ids"]
        or not isinstance(implementation_sources, dict)
        or not implementation_sources
        or any(
            generator_binding["source_sha256"].get(name) != digest
            for name, digest in implementation_sources.items()
        )
    ):
        raise ValueError("finite v4 row implementation differs from generator binding")
    contract = row["finite_psf_contract"]
    if (
        contract["finite_psf_capability_sha256"]
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or contract["render_mode"] != run_contract["render_mode"]
        or contract["axial_sample_count"] != run_contract["axial_sample_count"]
    ):
        raise ValueError("mixed finite-PSF render modes or sample counts are forbidden")
    thickness = float(contract["nominal_cut_thickness_um"])
    policy = run_contract["nominal_cut_thickness_policy"]
    if run_contract["render_mode"] == "finite_boxcar":
        valid_thickness = policy["minimum_um"] <= thickness <= policy["maximum_um"]
    else:
        valid_thickness = thickness == policy["nominal_cut_thickness_um"]
    if not math.isfinite(thickness) or not valid_thickness:
        raise ValueError("v4 row thickness differs from the cache-run contract")
    return True


def _verify_record(record, index, root, manifest):
    if not isinstance(record, dict) or set(record) != ROW_RECORD_KEYS:
        raise ValueError("finite v4 cache row-record fields changed")
    run_contract = manifest["finite_psf_run_contract"]
    lineage = record.get("lineage")
    expected_stem = record.get("training_row_receipt_sha256")
    if (
        record.get("row_index") != index
        or record.get("training_row_schema_version")
        != psf_v4.TRAINING_ROW_V4_SCHEMA
        or any(
            not _valid_sha256(record.get(name))
            for name in (
                "training_row_id",
                "training_row_receipt_sha256",
                "synthetic_realization_id",
                "finite_psf_capability_sha256",
                "finite_psf_sha256",
                "slab_observation_v4_receipt_sha256",
                "metadata_file_sha256",
                "arrays_file_sha256",
            )
        )
        or record.get("finite_psf_capability_sha256")
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or record.get("render_mode") != run_contract["render_mode"]
        or record.get("axial_sample_count") != run_contract["axial_sample_count"]
        or record.get("metadata_relative_path") != f"rows/{expected_stem}.json"
        or record.get("arrays_relative_path") != f"rows/{expected_stem}.npz"
    ):
        raise ValueError("finite v4 cache row record is malformed or mixed")
    _verify_row_lineage(lineage, manifest["generation_lineage"])
    thickness = record.get("nominal_cut_thickness_um")
    if not isinstance(thickness, (int, float)) or isinstance(thickness, bool):
        raise ValueError("finite v4 row record thickness is invalid")
    policy = run_contract["nominal_cut_thickness_policy"]
    if run_contract["render_mode"] == "finite_boxcar":
        valid_thickness = (
            math.isfinite(float(thickness))
            and policy["minimum_um"] <= float(thickness) <= policy["maximum_um"]
        )
    else:
        valid_thickness = float(thickness) == policy["nominal_cut_thickness_um"]
    if not valid_thickness:
        raise ValueError("finite v4 row-record thickness violates its run contract")
    for name in ("metadata_relative_path", "arrays_relative_path"):
        path = (root / record[name]).resolve()
        if root not in path.parents:
            raise ValueError("finite v4 cache record escapes its I:-drive root")


def load_training_row_cache_manifest_v4(
    cache_directory,
    *,
    expected_generator_binding=None,
    expected_receipt_sha256=None,
):
    root = _i_path(cache_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("finite v4 row-cache manifest fields changed")
    payload = _manifest_payload(manifest)
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(record, dict) for record in rows):
        raise ValueError("finite v4 row-cache records are invalid")
    row_ids = [record.get("training_row_id") for record in rows]
    row_receipts = [record.get("training_row_receipt_sha256") for record in rows]
    slab_receipts = [
        record.get("slab_observation_v4_receipt_sha256") for record in rows
    ]
    if (
        manifest["receipt_sha256"] != _hash_json(payload)
        or payload.get("schema_version") != ROW_CACHE_V4_SCHEMA
        or payload.get("data_role") != DEVELOPMENT_DATA_ROLE
        or payload.get("training_row_schema_version")
        != psf_v4.TRAINING_ROW_V4_SCHEMA
        or payload.get("finite_psf_capability") != _finite_psf_capability_v4()
        or payload.get("finite_psf_capability_sha256")
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or payload.get("status") not in (OPEN_CACHE_STATUS, FROZEN_CACHE_STATUS)
        or payload.get("row_count") != len(rows)
        or len(set(row_ids)) != len(row_ids)
        or len(set(row_receipts)) != len(row_receipts)
        or len(set(slab_receipts)) != len(slab_receipts)
        or any(
            payload.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("finite v4 row-cache manifest failed authentication")
    verify_finite_psf_cache_run_contract_v4(payload["finite_psf_run_contract"])
    verify_generator_binding_v4(payload["generator_binding"])
    binding = payload["generator_binding"]
    if (
        payload["finite_psf_run_contract"] != binding["finite_psf_run_contract"]
        or payload["generation_config"] != binding["generation_config"]
        or payload["seed_record"] != binding["seed_record"]
        or payload["generation_lineage"] != binding["generation_lineage"]
    ):
        raise ValueError("finite v4 manifest differs from its generator binding")
    _verify_generation_config(payload["generation_config"])
    _verify_seed_record(payload["seed_record"])
    _verify_generation_lineage(payload["generation_lineage"])
    if payload["status"] == OPEN_CACHE_STATUS:
        if payload["freeze_audit"] is not None:
            raise ValueError("open finite v4 cache cannot contain a freeze audit")
    elif (
        not isinstance(payload["freeze_audit"], dict)
        or set(payload["freeze_audit"]) != FREEZE_AUDIT_KEYS
        or payload["freeze_audit"] != _freeze_audit(payload)
    ):
        raise ValueError("frozen finite v4 cache audit changed")
    for index, record in enumerate(rows):
        _verify_record(record, index, root, payload)
    if expected_generator_binding is not None and binding != _plain(
        expected_generator_binding
    ):
        raise ValueError("finite v4 row-cache generator binding differs")
    if (
        expected_receipt_sha256 is not None
        and manifest["receipt_sha256"] != expected_receipt_sha256
    ):
        raise ValueError("finite v4 row-cache manifest receipt differs")
    _assert_no_learned_dependencies(payload)
    return manifest


def _record_from_row(root, row, row_index):
    receipt = row["receipt_sha256"]
    metadata_relative = f"rows/{receipt}.json"
    arrays_relative = f"rows/{receipt}.npz"
    metadata = {key: value for key, value in row.items() if key != "arrays"}
    _atomic_npz(root / arrays_relative, row["arrays"])
    _atomic_json(root / metadata_relative, metadata)
    finite_psf = row["finite_psf_contract"]
    return {
        "row_index": row_index,
        "training_row_schema_version": row["schema_version"],
        "training_row_id": row["training_row_id"],
        "training_row_receipt_sha256": receipt,
        "synthetic_realization_id": row["synthetic_realization_id"],
        "lineage": _plain(row["lineage"]),
        "selected_mode": row["selected_mode"],
        "reflection_state": row["reflection_state"],
        "finite_psf_capability_sha256": finite_psf[
            "finite_psf_capability_sha256"
        ],
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
        "slab_observation_v4_receipt_sha256": finite_psf[
            "slab_observation_v4_receipt_sha256"
        ],
        "nominal_cut_thickness_um": finite_psf["nominal_cut_thickness_um"],
        "render_mode": finite_psf["render_mode"],
        "axial_sample_count": finite_psf["axial_sample_count"],
        "metadata_relative_path": metadata_relative,
        "metadata_file_sha256": _file_sha256(root / metadata_relative),
        "arrays_relative_path": arrays_relative,
        "arrays_file_sha256": _file_sha256(root / arrays_relative),
    }


def _load_record(root, record, manifest):
    metadata_path = root / record["metadata_relative_path"]
    arrays_path = root / record["arrays_relative_path"]
    if (
        _file_sha256(metadata_path) != record["metadata_file_sha256"]
        or _file_sha256(arrays_path) != record["arrays_file_sha256"]
    ):
        raise ValueError("finite v4 cached training-row file hash differs")
    row = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as stored:
        row["arrays"] = {
            name: np.ascontiguousarray(stored[name]) for name in stored.files
        }
    _verify_row_against_run(
        row,
        manifest["finite_psf_run_contract"],
        manifest["generation_lineage"],
        manifest["generator_binding"],
    )
    finite_psf = row["finite_psf_contract"]
    expected = {
        "training_row_schema_version": row["schema_version"],
        "training_row_id": row["training_row_id"],
        "training_row_receipt_sha256": row["receipt_sha256"],
        "synthetic_realization_id": row["synthetic_realization_id"],
        "lineage": row["lineage"],
        "selected_mode": row["selected_mode"],
        "reflection_state": row["reflection_state"],
        "finite_psf_capability_sha256": finite_psf[
            "finite_psf_capability_sha256"
        ],
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
        "slab_observation_v4_receipt_sha256": finite_psf[
            "slab_observation_v4_receipt_sha256"
        ],
        "nominal_cut_thickness_um": finite_psf["nominal_cut_thickness_um"],
        "render_mode": finite_psf["render_mode"],
        "axial_sample_count": finite_psf["axial_sample_count"],
    }
    if any(record[name] != value for name, value in expected.items()):
        raise ValueError("finite v4 cached row differs from ordered manifest record")
    return row


def append_training_rows_v4(cache_directory, rows):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v4(root)
    if manifest["status"] != OPEN_CACHE_STATUS:
        raise ValueError("frozen finite v4 row caches cannot be appended")
    rows = list(rows)
    for row in rows:
        _verify_row_against_run(
            row,
            manifest["finite_psf_run_contract"],
            manifest["generation_lineage"],
            manifest["generator_binding"],
        )
    existing_ids = {record["training_row_id"] for record in manifest["rows"]}
    incoming_unique_ids = {row["training_row_id"] for row in rows} - existing_ids
    if (
        manifest["row_count"] + len(incoming_unique_ids)
        > manifest["generation_config"]["row_count"]
    ):
        raise ValueError("finite v4 cache would exceed its declared generation row count")
    existing = {record["training_row_id"]: record for record in manifest["rows"]}
    existing_receipts = {
        record["training_row_receipt_sha256"]: record["training_row_id"]
        for record in manifest["rows"]
    }
    existing_slabs = {
        record["slab_observation_v4_receipt_sha256"]: record["training_row_id"]
        for record in manifest["rows"]
    }
    incoming = {}
    incoming_receipts = {}
    incoming_slabs = {}
    for row in rows:
        row_id = row["training_row_id"]
        receipt = row["receipt_sha256"]
        slab = row["finite_psf_contract"][
            "slab_observation_v4_receipt_sha256"
        ]
        if row_id in incoming and incoming[row_id] != receipt:
            raise ValueError(
                "finite v4 training-row ID was reused with different content"
            )
        if receipt in incoming_receipts and incoming_receipts[receipt] != row_id:
            raise ValueError("finite v4 training-row receipt was reused")
        if slab in incoming_slabs and incoming_slabs[slab] != row_id:
            raise ValueError("finite v4 slab receipt was reused")
        if row_id in existing and existing[row_id][
            "training_row_receipt_sha256"
        ] != receipt:
            raise ValueError(
                "finite v4 training-row ID was reused with different content"
            )
        if receipt in existing_receipts and existing_receipts[receipt] != row_id:
            raise ValueError("finite v4 training-row receipt was reused")
        if slab in existing_slabs and existing_slabs[slab] != row_id:
            raise ValueError("finite v4 slab receipt was reused")
        incoming[row_id] = receipt
        incoming_receipts[receipt] = row_id
        incoming_slabs[slab] = row_id
    for row in rows:
        row_id = row["training_row_id"]
        if row_id in existing:
            record = existing[row_id]
            if record["training_row_receipt_sha256"] != row["receipt_sha256"]:
                raise ValueError("finite v4 training-row ID was reused with different content")
            _load_record(root, record, manifest)
            continue
        if any(
            record["training_row_receipt_sha256"] == row["receipt_sha256"]
            or record["slab_observation_v4_receipt_sha256"]
            == row["finite_psf_contract"][
                "slab_observation_v4_receipt_sha256"
            ]
            for record in existing.values()
        ):
            raise ValueError("finite v4 row or slab receipt was reused")
        record = _record_from_row(root, row, len(manifest["rows"]))
        payload = _manifest_payload(manifest)
        payload["rows"] = [*payload["rows"], record]
        payload["row_count"] = len(payload["rows"])
        manifest = _with_manifest_receipt(payload)
        _atomic_json(root / "manifest.json", manifest)
        existing[row_id] = record
    return manifest


def load_training_rows_v4(
    cache_directory,
    indices=None,
    *,
    expected_manifest_receipt_sha256=None,
):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v4(
        root, expected_receipt_sha256=expected_manifest_receipt_sha256
    )
    selected = (
        list(range(manifest["row_count"]))
        if indices is None
        else [int(index) for index in indices]
    )
    if any(index < 0 or index >= manifest["row_count"] for index in selected):
        raise IndexError("finite v4 row-cache index is out of range")
    return [_load_record(root, manifest["rows"][index], manifest) for index in selected]


def audit_training_row_cache_v4(cache_directory):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v4(root)
    for record in manifest["rows"]:
        _load_record(root, record, manifest)
    expected_files = {
        record[name]
        for record in manifest["rows"]
        for name in ("metadata_relative_path", "arrays_relative_path")
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / "rows").iterdir()
        if path.is_file() and path.suffix in (".json", ".npz")
    }
    if actual_files != expected_files:
        raise ValueError("finite v4 cache contains missing or unrecorded finalized row files")
    temporary_count = sum(
        path.is_file() and path.name.endswith(".tmp")
        for path in (root / "rows").iterdir()
    )
    rows = manifest["rows"]
    thicknesses = [record["nominal_cut_thickness_um"] for record in rows]
    payload = {
        "schema_version": ROW_CACHE_AUDIT_V4_SCHEMA,
        "manifest_receipt_sha256": manifest["receipt_sha256"],
        "generator_binding_receipt_sha256": manifest["generator_binding"][
            "receipt_sha256"
        ],
        "training_row_schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "finite_psf_capability_sha256": EXPECTED_FINITE_PSF_CAPABILITY_SHA256,
        "finite_psf_run_contract_receipt_sha256": manifest[
            "finite_psf_run_contract"
        ]["receipt_sha256"],
        "render_mode": manifest["finite_psf_run_contract"]["render_mode"],
        "axial_sample_count": manifest["finite_psf_run_contract"][
            "axial_sample_count"
        ],
        "row_count": manifest["row_count"],
        "nominal_cut_thickness_um_min": min(thicknesses) if thicknesses else None,
        "nominal_cut_thickness_um_max": max(thicknesses) if thicknesses else None,
        "ordered_training_row_receipts_sha256": _hash_json(
            [record["training_row_receipt_sha256"] for record in rows]
        ),
        "ordered_finite_psf_sha256": _hash_json(
            [record["finite_psf_sha256"] for record in rows]
        ),
        "ordered_slab_observation_v4_receipts_sha256": _hash_json(
            [record["slab_observation_v4_receipt_sha256"] for record in rows]
        ),
        "all_rows_authenticated": True,
        "learned_dependencies": [],
        "temporary_file_count": temporary_count,
        "data_role": DEVELOPMENT_DATA_ROLE,
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def freeze_training_row_cache_v4(cache_directory):
    root = _i_path(cache_directory)
    manifest = load_training_row_cache_manifest_v4(root)
    if manifest["status"] == FROZEN_CACHE_STATUS:
        audit_training_row_cache_v4(root)
        return manifest
    if manifest["row_count"] < 1:
        raise ValueError("an empty finite v4 cache cannot be frozen")
    if manifest["row_count"] != manifest["generation_config"]["row_count"]:
        raise ValueError("finite v4 cache cannot freeze before its declared row count")
    audit_training_row_cache_v4(root)
    payload = _manifest_payload(manifest)
    payload["status"] = FROZEN_CACHE_STATUS
    payload["freeze_audit"] = _freeze_audit(payload)
    frozen = _with_manifest_receipt(payload)
    _atomic_json(root / "manifest.json", frozen)
    return frozen
