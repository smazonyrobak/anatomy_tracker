"""Pure authenticated reader for frozen finite-thickness v4 row caches."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


ROW_CACHE_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-row-cache/v4"
GENERATOR_BINDING_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-generator-binding/v4"
)
FINITE_PSF_CACHE_RUN_V4_SCHEMA = "anatomy-tracker.finite-psf-cache-run/v4"
GENERATION_LINEAGE_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-cache-generation-lineage/v4"
)
FINITE_PSF_V4_SCHEMA = "anatomy-tracker.finite-psf-contract/v4"
FINITE_PSF_CAPABILITY_V4_SCHEMA = (
    "anatomy-tracker.finite-psf-model-capability/v4"
)
TRAINING_ROW_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-training-row/v4"
FROZEN_ROWS_V6_SCHEMA = "anatomy-tracker.frozen-generated-row-payloads/v6"
EXPECTED_FINITE_PSF_CAPABILITY_SHA256 = (
    "bcd6441a685e902fb5b59e85bb7003ef3261207d906a0b9390d4a219c3ae3d3e"
)
DEVELOPMENT_DATA_ROLE = "development-training"
FROZEN_CACHE_STATUS = "FROZEN"
PRODUCTION_AXIAL_SAMPLE_COUNT = 9
PRODUCTION_THICKNESS_RANGE_UM = (25.0, 100.0)
PRODUCTION_INTEGER_MASSES = (1, 2, 2, 2, 2, 2, 2, 2, 1)
AXIAL_STEP_UM_MAX = 12.5
SAMPLING_DIRECTION = "canonical physical AP-DV-ML arbitrary-plane normal"
NORMALIZATION = "global unit-mass PSF; no per-pixel in-bounds renormalization"
OUTSIDE_ATLAS_RULE = "zero padding before global weighted sum"
DEFORMATION_GAUGE_V4_SCHEMA = "anatomy-tracker.direct-deformation-target/v4"
DEFORMATION_GAUGE_V4_ALGORITHM = (
    "preintegration-uniform-canvas-affine-free-source-to-fixed-pullback-certification/v4"
)
DEFORMATION_PROJECTION_WEIGHTING = (
    "fixed uniform full canvas, matching decoder gauge"
)
DEFORMATION_TARGET_DIRECTION = (
    "source-to-fixed pullback exp(-v); target stationary velocity is -v"
)
DEFORMATION_NUMERIC_CONTRACT = (
    "float32; y-x channel order; absolute pixel-centre map; align_corners=True; "
    "border displacement composition; exactly seven scaling-and-squaring steps"
)
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
FINITE_PSF_KEYS = {
    "schema",
    "family",
    "render_mode",
    "nominal_cut_thickness_um",
    "production_thickness_range_um",
    "axial_sample_count",
    "axial_offsets_um",
    "axial_integer_masses",
    "axial_weights",
    "axial_step_um",
    "axial_step_um_max",
    "sampling_direction",
    "normalization",
    "outside_atlas_rule",
    "projection_operator",
    "thickness_selection_sha256",
    "finite_psf_capability_sha256",
    "finite_psf_sha256",
}
ROW_PSF_KEYS = FINITE_PSF_KEYS | {"slab_observation_v4_receipt_sha256"}
TRAINING_ROW_ARRAY_KEYS = {
    "model_input_channels_float32",
    "source_label_ground_truth_canvas_int64",
    "source_tissue_ground_truth_mask",
    "target_ccf_coordinates_ap_dv_ml_um_float64",
    "target_valid_correspondence_mask",
    "target_correspondence_weight_float32",
    "target_correspondence_abstention_mask",
    "truth_section_pullback_map_yx_px_float64",
    "truth_section_pullback_stationary_velocity_yx_px_float64",
    "truth_section_deformation_valid_mask",
}
TRAINING_ROW_V4_KEYS = {
    "schema_version",
    "source_observation_receipt_sha256",
    "lineage",
    "upstream_reference",
    "numeric_rng_provenance",
    "rng_sources",
    "selected_mode",
    "selected_descendant_id",
    "deformation_pose_gauge_reference",
    "reflection_state",
    "reflection_representation_index",
    "reflection_representation_affine_xy_float64",
    "canonical_effective_quicknii_ouv_float64",
    "observed_effective_quicknii_ouv_float64",
    "proper_physical_pose_unchanged",
    "prior_model_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "reflection_transform_id",
    "reflection_realization_id",
    "paired_view_group_id",
    "synthetic_realization_id",
    "paired_mode_reflected_receipts",
    "arrays",
    "array_receipts",
    "training_row_id",
    "finite_psf_contract",
    "receipt_sha256",
}
TRAINING_ROW_RECEIPT_KEYS = (
    "schema_version",
    "source_observation_receipt_sha256",
    "lineage",
    "upstream_reference",
    "numeric_rng_provenance",
    "rng_sources",
    "selected_mode",
    "selected_descendant_id",
    "deformation_pose_gauge_reference",
    "reflection_state",
    "reflection_representation_index",
    "reflection_representation_affine_xy_float64",
    "canonical_effective_quicknii_ouv_float64",
    "observed_effective_quicknii_ouv_float64",
    "proper_physical_pose_unchanged",
    "prior_model_dependencies",
    "prior_feature_dependencies",
    "prior_pseudolabel_dependencies",
    "reflection_transform_id",
    "reflection_realization_id",
    "paired_view_group_id",
    "synthetic_realization_id",
    "paired_mode_reflected_receipts",
    "array_receipts",
    "training_row_id",
)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _cache_canonical_json(value):
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _cache_sha256(value):
    return hashlib.sha256(_cache_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_sha256(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_receipt(value):
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(normalized.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "array_sha256": digest.hexdigest(),
    }


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _valid_psf_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value.lower()) - set("0123456789abcdef"))
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


def _assert_no_learned_dependencies(value):
    if isinstance(value, Mapping):
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


def _finite_psf_model_capability_v4():
    payload = {
        "schema_version": FINITE_PSF_CAPABILITY_V4_SCHEMA,
        "family": "boxcar",
        "sampling_direction": SAMPLING_DIRECTION,
        "training_schedule_scope": "authenticated-exact-per-row",
        "runtime_schedule_scope": (
            "caller-explicit-exact-inference-session-or-feature-cache-bound"
        ),
        "production": {
            "render_mode": "finite_boxcar",
            "nominal_cut_thickness_range_um": list(
                PRODUCTION_THICKNESS_RANGE_UM
            ),
            "axial_sample_count": PRODUCTION_AXIAL_SAMPLE_COUNT,
            "axial_integer_masses": list(PRODUCTION_INTEGER_MASSES),
            "axial_step_um_max": AXIAL_STEP_UM_MAX,
        },
        "zero_thickness_ablation": {
            "render_mode": "centre_plane_ablation",
            "nominal_cut_thickness_um": 0.0,
            "axial_offsets_um": [0.0],
            "axial_weights": [1.0],
        },
        "unknown_thickness_policy": "reject",
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return {**payload, "receipt_sha256": _payload_sha256(payload)}


def verify_finite_psf_model_capability_v6(capability):
    if capability != _finite_psf_model_capability_v4():
        raise ValueError("finite-PSF model capability is unsupported or changed")
    return True


def _finite_psf_payload(contract):
    return {
        key: contract[key]
        for key in sorted(FINITE_PSF_KEYS - {"finite_psf_sha256"})
    }


def _canonical_axial_offsets_v4(thickness):
    offsets = np.linspace(
        -float(thickness) / 2.0,
        float(thickness) / 2.0,
        PRODUCTION_AXIAL_SAMPLE_COUNT,
        dtype=np.float64,
    )
    centre = PRODUCTION_AXIAL_SAMPLE_COUNT // 2
    offsets[centre] = 0.0
    offsets[:centre] = -offsets[:centre:-1]
    return offsets


def _verify_finite_psf_schedule_v4(contract, capability):
    if not isinstance(contract, dict) or set(contract) != FINITE_PSF_KEYS:
        raise ValueError("v4 finite-PSF schedule fields are incomplete or unknown")
    verify_finite_psf_model_capability_v6(capability)
    offsets = np.asarray(contract.get("axial_offsets_um", ()), dtype=np.float64)
    masses = np.asarray(contract.get("axial_integer_masses", ()), dtype=np.int64)
    weights = np.asarray(contract.get("axial_weights", ()), dtype=np.float64)
    thickness = contract.get("nominal_cut_thickness_um")
    mode = contract.get("render_mode")
    valid = (
        contract.get("schema") == FINITE_PSF_V4_SCHEMA
        and contract.get("family") == "boxcar"
        and contract.get("production_thickness_range_um")
        == list(PRODUCTION_THICKNESS_RANGE_UM)
        and contract.get("sampling_direction") == SAMPLING_DIRECTION
        and contract.get("normalization") == NORMALIZATION
        and contract.get("outside_atlas_rule") == OUTSIDE_ATLAS_RULE
        and contract.get("axial_step_um_max") == AXIAL_STEP_UM_MAX
        and _valid_psf_sha256(contract.get("thickness_selection_sha256"))
        and contract.get("finite_psf_capability_sha256")
        == capability["receipt_sha256"]
        and contract.get("finite_psf_sha256")
        == _payload_sha256(_finite_psf_payload(contract))
        and isinstance(thickness, (int, float))
        and not isinstance(thickness, bool)
        and math.isfinite(float(thickness))
        and offsets.ndim == masses.ndim == weights.ndim == 1
        and offsets.shape == masses.shape == weights.shape
        and contract.get("axial_sample_count") == offsets.size
        and np.isfinite(offsets).all()
        and np.isfinite(weights).all()
        and np.all(weights > 0.0)
        and np.array_equal(offsets, -offsets[::-1])
        and np.array_equal(masses, masses[::-1])
        and np.array_equal(weights, weights[::-1])
        and np.array_equal(
            weights, masses.astype(np.float64) / int(masses.sum())
        )
    )
    if not valid:
        raise ValueError("v4 finite-PSF schedule failed capability or receipt validation")
    if mode == "finite_boxcar":
        valid_mode = (
            PRODUCTION_THICKNESS_RANGE_UM[0]
            <= float(thickness)
            <= PRODUCTION_THICKNESS_RANGE_UM[1]
            and offsets.size == PRODUCTION_AXIAL_SAMPLE_COUNT
            and np.array_equal(masses, np.asarray(PRODUCTION_INTEGER_MASSES))
            and np.array_equal(offsets, _canonical_axial_offsets_v4(thickness))
            and contract.get("axial_step_um") == float(thickness) / 8.0
            and contract.get("projection_operator")
            == "finite-full-slab-boxcar-trapezoidal-quadrature"
        )
    elif mode == "centre_plane_ablation":
        valid_mode = (
            float(thickness) == 0.0
            and np.array_equal(offsets, np.asarray([0.0]))
            and np.array_equal(masses, np.asarray([1]))
            and np.array_equal(weights, np.asarray([1.0]))
            and contract.get("axial_step_um") == 0.0
            and contract.get("projection_operator")
            == "direct-centre-plane-ablation"
        )
    else:
        valid_mode = False
    if not valid_mode:
        raise ValueError("finite-PSF thickness or axial schedule is unsupported")


def _verify_training_row_psf_contract_v4(contract, capability):
    if not isinstance(contract, dict) or set(contract) != ROW_PSF_KEYS:
        raise ValueError("v4 row finite-PSF contract is incomplete or ambiguous")
    if not _valid_psf_sha256(
        contract.get("slab_observation_v4_receipt_sha256")
    ):
        raise ValueError("v4 row finite-PSF source receipt is invalid")
    _verify_finite_psf_schedule_v4(
        {key: contract[key] for key in FINITE_PSF_KEYS}, capability
    )


def _training_row_receipt_v4(row):
    return {
        **{key: row[key] for key in TRAINING_ROW_RECEIPT_KEYS},
        "finite_psf_contract": row["finite_psf_contract"],
    }


def verify_finite_training_row_v6(
    row,
    *,
    finite_psf_capability,
    cache_manifest=None,
):
    arrays = row.get("arrays", {})
    contract = row.get("finite_psf_contract", {})
    upstream = row.get("upstream_reference", {})
    source_binding_valid = (
        isinstance(upstream, dict)
        and _valid_psf_sha256(upstream.get("slab_observation_id"))
        and _valid_psf_sha256(
            upstream.get("centre_plane_targets_receipt_sha256")
        )
        and upstream.get("slab_observation_v4_receipt_sha256")
        == contract.get("slab_observation_v4_receipt_sha256")
        and upstream.get("finite_psf_sha256")
        == contract.get("finite_psf_sha256")
        and upstream.get("finite_psf_capability_sha256")
        == contract.get("finite_psf_capability_sha256")
    )
    if (
        set(row) != TRAINING_ROW_V4_KEYS
        or row.get("schema_version") != TRAINING_ROW_V4_SCHEMA
        or set(arrays) != TRAINING_ROW_ARRAY_KEYS
        or row.get("array_receipts")
        != {name: _array_receipt(value) for name, value in arrays.items()}
        or row.get("receipt_sha256")
        != _payload_sha256(_training_row_receipt_v4(row))
        or not source_binding_valid
    ):
        raise ValueError("v4 training-row receipt, arrays, or source binding changed")
    verify_finite_psf_model_capability_v6(finite_psf_capability)
    _verify_training_row_psf_contract_v4(contract, finite_psf_capability)
    expected_row_id = _payload_sha256(
        {
            "domain": TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": contract["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": contract[
                "slab_observation_v4_receipt_sha256"
            ],
        }
    )
    if row.get("training_row_id") != expected_row_id:
        raise ValueError("finite v4 training-row identity differs from its exact inputs")
    if cache_manifest is not None:
        if finite_psf_capability != cache_manifest.get("finite_psf_capability"):
            raise ValueError("finite v4 row capability differs from its cache manifest")
        _verify_row_source_context(row, cache_manifest)
    return True


def finite_psf_tensors_from_training_row_v6(
    row,
    *,
    finite_psf_capability,
    device=None,
    dtype=torch.float32,
):
    verify_finite_training_row_v6(
        row, finite_psf_capability=finite_psf_capability
    )
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point() or probe.is_complex():
        raise ValueError("finite-PSF tensor dtype must be real floating point")
    contract = row["finite_psf_contract"]
    return {
        "axial_offsets_um": torch.as_tensor(
            contract["axial_offsets_um"], device=device, dtype=dtype
        )[None].contiguous(),
        "axial_weights": torch.as_tensor(
            contract["axial_weights"], device=device, dtype=dtype
        )[None].contiguous(),
    }


def _direct_deformation_target_contract_v4():
    return {
        "schema_version": DEFORMATION_GAUGE_V4_SCHEMA,
        "algorithm": DEFORMATION_GAUGE_V4_ALGORITHM,
        "projection_weighting": DEFORMATION_PROJECTION_WEIGHTING,
        "target_direction": DEFORMATION_TARGET_DIRECTION,
        "numeric_contract": DEFORMATION_NUMERIC_CONTRACT,
        "runtime_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def _expected_run_contract(render_mode):
    if render_mode == "finite_boxcar":
        sample_count = PRODUCTION_AXIAL_SAMPLE_COUNT
        thickness_policy = {
            "kind": "authenticated-per-row-closed-interval",
            "minimum_um": PRODUCTION_THICKNESS_RANGE_UM[0],
            "maximum_um": PRODUCTION_THICKNESS_RANGE_UM[1],
        }
    elif render_mode == "centre_plane_ablation":
        sample_count = 1
        thickness_policy = {"kind": "exact", "nominal_cut_thickness_um": 0.0}
    else:
        raise ValueError(
            "v4 cache render mode must be finite_boxcar or centre_plane_ablation"
        )
    payload = {
        "schema_version": FINITE_PSF_CACHE_RUN_V4_SCHEMA,
        "training_row_schema_version": TRAINING_ROW_V4_SCHEMA,
        "finite_psf_capability_sha256": EXPECTED_FINITE_PSF_CAPABILITY_SHA256,
        "render_mode": render_mode,
        "axial_sample_count": sample_count,
        "nominal_cut_thickness_policy": thickness_policy,
    }
    return {**payload, "receipt_sha256": _cache_sha256(payload)}


def _verify_run_contract(contract):
    if not isinstance(contract, dict) or set(contract) != RUN_CONTRACT_KEYS:
        raise ValueError("finite-PSF cache-run contract fields changed")
    if contract != _expected_run_contract(contract.get("render_mode")):
        raise ValueError("finite-PSF cache-run contract changed")


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
    _cache_canonical_json(config)
    _assert_no_learned_dependencies(config)


def _verify_seed_record(seed_record):
    if not isinstance(seed_record, dict) or not seed_record:
        raise ValueError("exact nonempty generation seed record is required")
    _cache_canonical_json(seed_record)


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
    _cache_canonical_json(lineage)
    _assert_no_learned_dependencies(lineage)


def _verify_generator_binding(binding):
    if not isinstance(binding, dict) or set(binding) != GENERATOR_BINDING_KEYS:
        raise ValueError("finite v4 generator-binding fields changed")
    payload = {key: value for key, value in binding.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload.get("generator_ids"), list)
        or any(
            not isinstance(value, str) or not value
            for value in payload["generator_ids"]
        )
        or not isinstance(payload.get("source_sha256"), dict)
        or not isinstance(payload.get("geometry_gauge_contract"), dict)
    ):
        raise ValueError("finite v4 generator binding is malformed")
    _verify_generation_config(payload.get("generation_config"))
    _verify_seed_record(payload.get("seed_record"))
    _verify_generation_lineage(payload.get("generation_lineage"))
    _verify_run_contract(payload.get("finite_psf_run_contract"))
    capability = _finite_psf_model_capability_v4()
    if (
        binding["receipt_sha256"] != _cache_sha256(payload)
        or payload.get("schema_version") != GENERATOR_BINDING_V4_SCHEMA
        or payload.get("training_row_schema_version") != TRAINING_ROW_V4_SCHEMA
        or payload.get("finite_psf_capability") != capability
        or payload.get("finite_psf_capability_sha256")
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or not payload.get("generator_ids")
        or payload["generator_ids"] != sorted(set(payload["generator_ids"]))
        or not payload.get("source_sha256")
        or any(
            not _valid_sha256(value)
            for value in payload["source_sha256"].values()
        )
        or payload.get("geometry_gauge_contract")
        != _direct_deformation_target_contract_v4()
        or payload.get("geometry_gauge_contract_sha256")
        != _cache_sha256(payload["geometry_gauge_contract"])
        or payload.get("generation_config_sha256")
        != _cache_sha256(payload["generation_config"])
        or payload.get("seed_record_sha256")
        != _cache_sha256(payload["seed_record"])
        or payload.get("generation_lineage_sha256")
        != _cache_sha256(payload["generation_lineage"])
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
    _cache_canonical_json(lineage)


def _verify_row_source_context(row, manifest):
    _verify_row_lineage(row.get("lineage"), manifest["generation_lineage"])
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
    gauge_reference = row.get("deformation_pose_gauge_reference", {})
    gauge_contract = manifest["generator_binding"]["geometry_gauge_contract"]
    if (
        not isinstance(gauge_reference, dict)
        or any(
            gauge_reference.get(name) != value
            for name, value in gauge_contract.items()
        )
        or any(
            not _valid_sha256(gauge_reference.get(name))
            for name in ("direct_deformation_target_id", "receipt_sha256")
        )
    ):
        raise ValueError("finite v4 row direct deformation gauge differs from binding")
    upstream = row.get("upstream_reference", {})
    implementation_sources = upstream.get("implementation_source_sha256")
    binding = manifest["generator_binding"]
    if (
        upstream.get("algorithm") not in binding["generator_ids"]
        or not isinstance(implementation_sources, dict)
        or not implementation_sources
        or any(
            binding["source_sha256"].get(name) != digest
            for name, digest in implementation_sources.items()
        )
    ):
        raise ValueError("finite v4 row implementation differs from generator binding")
    contract = row["finite_psf_contract"]
    run_contract = manifest["finite_psf_run_contract"]
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


def _freeze_audit(manifest):
    rows = manifest["rows"]
    return {
        "row_count": len(rows),
        "ordered_training_row_receipts_sha256": _cache_sha256(
            [record["training_row_receipt_sha256"] for record in rows]
        ),
        "ordered_finite_psf_sha256": _cache_sha256(
            [record["finite_psf_sha256"] for record in rows]
        ),
        "ordered_slab_observation_v4_receipts_sha256": _cache_sha256(
            [record["slab_observation_v4_receipt_sha256"] for record in rows]
        ),
        "ordered_nominal_cut_thickness_um_sha256": _cache_sha256(
            [record["nominal_cut_thickness_um"] for record in rows]
        ),
        "all_rows_authenticated": True,
        "learned_dependencies": [],
    }


def _verify_record(record, index, root, manifest):
    if not isinstance(record, dict) or set(record) != ROW_RECORD_KEYS:
        raise ValueError("finite v4 cache row-record fields changed")
    run_contract = manifest["finite_psf_run_contract"]
    expected_stem = record.get("training_row_receipt_sha256")
    if (
        record.get("row_index") != index
        or record.get("training_row_schema_version") != TRAINING_ROW_V4_SCHEMA
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
    _verify_row_lineage(record.get("lineage"), manifest["generation_lineage"])
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


def load_frozen_row_cache_manifest_v6(
    cache_directory,
    *,
    expected_manifest_receipt_sha256,
):
    if not _valid_sha256(expected_manifest_receipt_sha256):
        raise ValueError("a trusted frozen-cache manifest SHA-256 is required")
    root = _i_path(cache_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("finite v4 row-cache manifest fields changed")
    payload = {
        key: value for key, value in manifest.items() if key != "receipt_sha256"
    }
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(record, dict) for record in rows):
        raise ValueError("finite v4 row-cache records are invalid")
    row_ids = [record.get("training_row_id") for record in rows]
    row_receipts = [record.get("training_row_receipt_sha256") for record in rows]
    slab_receipts = [
        record.get("slab_observation_v4_receipt_sha256") for record in rows
    ]
    if (
        manifest["receipt_sha256"] != _cache_sha256(payload)
        or manifest["receipt_sha256"] != expected_manifest_receipt_sha256
        or payload.get("schema_version") != ROW_CACHE_V4_SCHEMA
        or payload.get("data_role") != DEVELOPMENT_DATA_ROLE
        or payload.get("training_row_schema_version") != TRAINING_ROW_V4_SCHEMA
        or payload.get("finite_psf_capability")
        != _finite_psf_model_capability_v4()
        or payload.get("finite_psf_capability_sha256")
        != EXPECTED_FINITE_PSF_CAPABILITY_SHA256
        or payload.get("status") != FROZEN_CACHE_STATUS
        or payload.get("row_count") != len(rows)
        or payload.get("row_count") != payload.get("generation_config", {}).get("row_count")
        or payload.get("row_count", 0) < 1
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
        raise ValueError("finite v4 frozen row-cache manifest failed authentication")
    _verify_run_contract(payload["finite_psf_run_contract"])
    _verify_generator_binding(payload["generator_binding"])
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
    if (
        not isinstance(payload["freeze_audit"], dict)
        or set(payload["freeze_audit"]) != FREEZE_AUDIT_KEYS
        or payload["freeze_audit"] != _freeze_audit(payload)
    ):
        raise ValueError("frozen finite v4 cache audit changed")
    for index, record in enumerate(rows):
        _verify_record(record, index, root, payload)
    _assert_no_learned_dependencies(payload)
    return manifest


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
    verify_finite_training_row_v6(
        row,
        finite_psf_capability=manifest["finite_psf_capability"],
        cache_manifest=manifest,
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


def _frozen_rows_receipt(payload):
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "training_data_manifest_receipt_sha256",
            "cache_manifest_receipt_sha256",
            "generator_binding_receipt_sha256",
            "generation_lineage_sha256",
            "row_indices",
            "training_row_ids",
            "training_row_receipts_sha256",
        )
    }


def frozen_row_selection_receipt_v6(payload):
    """Hash one row selection without loading its dense arrays."""
    return _payload_sha256(_frozen_rows_receipt(payload))


def load_frozen_training_rows_v6(
    cache_directory,
    indices=None,
    *,
    expected_manifest_receipt_sha256,
):
    root = _i_path(cache_directory)
    manifest = load_frozen_row_cache_manifest_v6(
        root,
        expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
    )
    selected = (
        list(range(manifest["row_count"]))
        if indices is None
        else [int(index) for index in indices]
    )
    if any(index < 0 or index >= manifest["row_count"] for index in selected):
        raise IndexError("finite v4 row-cache index is out of range")
    rows = [
        _load_record(root, manifest["rows"][index], manifest)
        for index in selected
    ]
    payload = {
        "schema_version": FROZEN_ROWS_V6_SCHEMA,
        "training_data_manifest_receipt_sha256": manifest["receipt_sha256"],
        "cache_manifest_receipt_sha256": manifest["receipt_sha256"],
        "generator_binding_receipt_sha256": manifest["generator_binding"][
            "receipt_sha256"
        ],
        "generation_lineage_sha256": manifest["generator_binding"][
            "generation_lineage_sha256"
        ],
        "row_indices": selected,
        "training_row_ids": [row["training_row_id"] for row in rows],
        "training_row_receipts_sha256": [
            row["receipt_sha256"] for row in rows
        ],
        "rows": rows,
    }
    payload["selection_receipt_sha256"] = frozen_row_selection_receipt_v6(payload)
    return payload


__all__ = [
    "EXPECTED_FINITE_PSF_CAPABILITY_SHA256",
    "FINITE_PSF_CAPABILITY_V4_SCHEMA",
    "FINITE_PSF_V4_SCHEMA",
    "FROZEN_ROWS_V6_SCHEMA",
    "TRAINING_ROW_V4_SCHEMA",
    "finite_psf_tensors_from_training_row_v6",
    "frozen_row_selection_receipt_v6",
    "load_frozen_row_cache_manifest_v6",
    "load_frozen_training_rows_v6",
    "verify_finite_psf_model_capability_v6",
    "verify_finite_training_row_v6",
]
