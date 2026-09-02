"""Authenticated per-row finite-PSF schedules for v4 training rows."""

from __future__ import annotations

import copy
import math

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_training_row_v3 as training_row_v3


FINITE_PSF_V4_SCHEMA = "anatomy-tracker.finite-psf-contract/v4"
FINITE_PSF_CAPABILITY_V4_SCHEMA = (
    "anatomy-tracker.finite-psf-model-capability/v4"
)
TRAINING_ROW_V4_SCHEMA = "anatomy-tracker.arbitrary-plane-training-row/v4"
PRODUCTION_AXIAL_SAMPLE_COUNT = 9
PRODUCTION_THICKNESS_RANGE_UM = (25.0, 100.0)
PRODUCTION_INTEGER_MASSES = (1, 2, 2, 2, 2, 2, 2, 2, 1)
AXIAL_STEP_UM_MAX = 12.5
SAMPLING_DIRECTION = "canonical physical AP-DV-ML arbitrary-plane normal"
NORMALIZATION = "global unit-mass PSF; no per-pixel in-bounds renormalization"
OUTSIDE_ATLAS_RULE = "zero padding before global weighted sum"
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
PREFINAL_TRAINING_ROW_V4_KEYS = TRAINING_ROW_V4_KEYS - {
    "training_row_id",
    "finite_psf_contract",
    "receipt_sha256",
}


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value.lower()) - set("0123456789abcdef"))
    )


def finite_psf_model_capability_v4():
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
    return {**payload, "receipt_sha256": acquisition_v2._payload_sha256(payload)}


def verify_finite_psf_model_capability_v4(capability):
    if capability != finite_psf_model_capability_v4():
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


def make_finite_psf_schedule_v4(
    render_mode,
    nominal_cut_thickness_um,
    *,
    thickness_selection_sha256,
):
    capability = finite_psf_model_capability_v4()
    thickness = float(nominal_cut_thickness_um)
    if render_mode == "finite_boxcar":
        offsets = _canonical_axial_offsets_v4(thickness)
        masses = np.asarray(PRODUCTION_INTEGER_MASSES, dtype=np.int64)
        step = thickness / 8.0
        projection = "finite-full-slab-boxcar-trapezoidal-quadrature"
    elif render_mode == "centre_plane_ablation":
        offsets = np.asarray([0.0], dtype=np.float64)
        masses = np.asarray([1], dtype=np.int64)
        step = 0.0
        projection = "direct-centre-plane-ablation"
    else:
        raise ValueError("unknown v4 finite-PSF render mode")
    payload = {
        "schema": FINITE_PSF_V4_SCHEMA,
        "family": "boxcar",
        "render_mode": render_mode,
        "nominal_cut_thickness_um": thickness,
        "production_thickness_range_um": list(PRODUCTION_THICKNESS_RANGE_UM),
        "axial_sample_count": int(offsets.size),
        "axial_offsets_um": offsets.tolist(),
        "axial_integer_masses": masses.tolist(),
        "axial_weights": (masses / int(masses.sum())).tolist(),
        "axial_step_um": step,
        "axial_step_um_max": AXIAL_STEP_UM_MAX,
        "sampling_direction": SAMPLING_DIRECTION,
        "normalization": NORMALIZATION,
        "outside_atlas_rule": OUTSIDE_ATLAS_RULE,
        "projection_operator": projection,
        "thickness_selection_sha256": str(thickness_selection_sha256),
        "finite_psf_capability_sha256": capability["receipt_sha256"],
    }
    contract = {
        **payload,
        "finite_psf_sha256": acquisition_v2._payload_sha256(payload),
    }
    verify_finite_psf_schedule_v4(contract, capability=capability)
    return contract


def verify_finite_psf_schedule_v4(contract, *, capability=None):
    if not isinstance(contract, dict) or set(contract) != FINITE_PSF_KEYS:
        raise ValueError("v4 finite-PSF schedule fields are incomplete or unknown")
    expected_capability = finite_psf_model_capability_v4()
    if capability is not None:
        verify_finite_psf_model_capability_v4(capability)
        expected_capability = capability
    offsets = np.asarray(contract.get("axial_offsets_um", ()), dtype=np.float64)
    masses = np.asarray(contract.get("axial_integer_masses", ()), dtype=np.int64)
    weights = np.asarray(contract.get("axial_weights", ()), dtype=np.float64)
    thickness = contract.get("nominal_cut_thickness_um")
    mode = contract.get("render_mode")
    basic = (
        contract.get("schema") == FINITE_PSF_V4_SCHEMA
        and contract.get("family") == "boxcar"
        and contract.get("production_thickness_range_um")
        == list(PRODUCTION_THICKNESS_RANGE_UM)
        and contract.get("sampling_direction") == SAMPLING_DIRECTION
        and contract.get("normalization") == NORMALIZATION
        and contract.get("outside_atlas_rule") == OUTSIDE_ATLAS_RULE
        and contract.get("axial_step_um_max") == AXIAL_STEP_UM_MAX
        and _valid_sha256(contract.get("thickness_selection_sha256"))
        and contract.get("finite_psf_capability_sha256")
        == expected_capability["receipt_sha256"]
        and contract.get("finite_psf_sha256")
        == acquisition_v2._payload_sha256(_finite_psf_payload(contract))
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
        and np.array_equal(weights, masses.astype(np.float64) / int(masses.sum()))
    )
    if not basic:
        raise ValueError("v4 finite-PSF schedule failed capability or receipt validation")
    if mode == "finite_boxcar":
        expected_offsets = _canonical_axial_offsets_v4(thickness)
        valid = (
            PRODUCTION_THICKNESS_RANGE_UM[0]
            <= float(thickness)
            <= PRODUCTION_THICKNESS_RANGE_UM[1]
            and offsets.size == PRODUCTION_AXIAL_SAMPLE_COUNT
            and np.array_equal(masses, np.asarray(PRODUCTION_INTEGER_MASSES))
            and np.array_equal(offsets, expected_offsets)
            and contract.get("axial_step_um") == float(thickness) / 8.0
            and contract.get("projection_operator")
            == "finite-full-slab-boxcar-trapezoidal-quadrature"
        )
    elif mode == "centre_plane_ablation":
        valid = (
            float(thickness) == 0.0
            and np.array_equal(offsets, np.asarray([0.0]))
            and np.array_equal(masses, np.asarray([1]))
            and np.array_equal(weights, np.asarray([1.0]))
            and contract.get("axial_step_um") == 0.0
            and contract.get("projection_operator")
            == "direct-centre-plane-ablation"
        )
    else:
        valid = False
    if not valid:
        raise ValueError("finite-PSF thickness or axial schedule is unsupported")
    return True


def verify_training_row_psf_contract_v4(contract, *, capability=None):
    if not isinstance(contract, dict) or set(contract) != ROW_PSF_KEYS:
        raise ValueError("v4 row finite-PSF contract is incomplete or ambiguous")
    if not _valid_sha256(contract.get("slab_observation_v4_receipt_sha256")):
        raise ValueError("v4 row finite-PSF source receipt is invalid")
    verify_finite_psf_schedule_v4(
        {key: contract[key] for key in FINITE_PSF_KEYS},
        capability=capability,
    )
    return True


def finalize_training_row_v4(
    row_like,
    slab_observation_v4,
    *,
    capability=None,
):
    arrays = row_like.get("arrays", {})
    upstream = row_like.get("upstream_reference", {})
    if (
        set(row_like) != PREFINAL_TRAINING_ROW_V4_KEYS
        or row_like.get("schema_version") != TRAINING_ROW_V4_SCHEMA
        or set(arrays) != training_row_v3._ARRAY_KEYS
        or row_like.get("array_receipts")
        != {
            name: acquisition_v2._array_receipt(value)
            for name, value in arrays.items()
        }
        or not _valid_sha256(row_like.get("source_observation_receipt_sha256"))
        or not _valid_sha256(row_like.get("synthetic_realization_id"))
        or not isinstance(upstream, dict)
    ):
        raise ValueError(
            "v4 finalization requires an unfinalized slab-derived v4 row"
        )
    if not isinstance(slab_observation_v4, dict):
        raise ValueError("canonical slab_observation_v4 block is missing")
    finite_psf = slab_observation_v4.get("finite_psf")
    slab_receipt = slab_observation_v4.get("receipt_sha256")
    verify_finite_psf_schedule_v4(finite_psf, capability=capability)
    expected_source_binding = {
        "slab_observation_id": slab_observation_v4.get(
            "slab_observation_id"
        ),
        "centre_plane_targets_receipt_sha256": slab_observation_v4.get(
            "centre_plane_targets_receipt_sha256"
        ),
        "slab_observation_v4_receipt_sha256": slab_receipt,
        "finite_psf_sha256": finite_psf["finite_psf_sha256"],
        "finite_psf_capability_sha256": finite_psf[
            "finite_psf_capability_sha256"
        ],
    }
    if (
        not all(_valid_sha256(value) for value in expected_source_binding.values())
        or any(upstream.get(key) != value for key, value in expected_source_binding.items())
    ):
        raise ValueError(
            "v4 row upstream source does not bind the canonical slab observation"
        )
    row = copy.deepcopy(row_like)
    row["finite_psf_contract"] = {
        **copy.deepcopy(finite_psf),
        "slab_observation_v4_receipt_sha256": slab_receipt,
    }
    row["training_row_id"] = acquisition_v2._payload_sha256(
        {
            "domain": TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": row["synthetic_realization_id"],
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": finite_psf["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": slab_receipt,
        }
    )
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        training_row_receipt_v4(row)
    )
    verify_training_row_v4(row, capability=capability)
    return row


def training_row_receipt_v4(row):
    receipt = training_row_v3.training_row_receipt_v3(row)
    return {**receipt, "finite_psf_contract": row["finite_psf_contract"]}


def verify_training_row_v4(row, *, capability=None):
    arrays = row.get("arrays", {})
    contract = row.get("finite_psf_contract", {})
    upstream = row.get("upstream_reference", {})
    source_binding_valid = (
        isinstance(upstream, dict)
        and _valid_sha256(upstream.get("slab_observation_id"))
        and _valid_sha256(
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
        or set(arrays) != training_row_v3._ARRAY_KEYS
        or row.get("array_receipts")
        != {
            name: acquisition_v2._array_receipt(value)
            for name, value in arrays.items()
        }
        or row.get("receipt_sha256")
        != acquisition_v2._payload_sha256(training_row_receipt_v4(row))
        or not source_binding_valid
    ):
        raise ValueError("v4 training-row receipt or arrays changed")
    verify_training_row_psf_contract_v4(
        contract, capability=capability
    )
    return True


def schedule_tensors_from_training_row_v4(row, *, capability=None, device=None):
    verify_training_row_v4(row, capability=capability)
    contract = row["finite_psf_contract"]
    return {
        "axial_offsets_um": np.asarray(
            contract["axial_offsets_um"], dtype=np.float32
        ),
        "axial_weights": np.asarray(contract["axial_weights"], dtype=np.float32),
    }


def runtime_schedule_contract_v4(
    axial_offsets_um,
    axial_weights,
    *,
    capability=None,
):
    expected_capability = finite_psf_model_capability_v4()
    if capability is not None:
        verify_finite_psf_model_capability_v4(capability)
        expected_capability = capability
    offsets = np.asarray(axial_offsets_um, dtype=np.float64)
    weights = np.asarray(axial_weights, dtype=np.float64)
    masses = np.asarray(PRODUCTION_INTEGER_MASSES, dtype=np.int64)
    if offsets.shape == (1,) and weights.shape == (1,):
        mode, thickness, step, expected_masses = (
            "centre_plane_ablation",
            0.0,
            0.0,
            np.asarray([1], dtype=np.int64),
        )
    elif offsets.shape == (PRODUCTION_AXIAL_SAMPLE_COUNT,) and weights.shape == offsets.shape:
        mode = "finite_boxcar"
        thickness = float(offsets[-1] - offsets[0])
        step = thickness / 8.0
        expected_masses = masses
    else:
        raise ValueError("runtime finite-PSF schedule has unsupported sample count")
    if (
        not np.array_equal(weights, expected_masses / int(expected_masses.sum()))
        or (
            mode == "centre_plane_ablation"
            and not np.array_equal(offsets, np.asarray([0.0]))
        )
        or (
            mode == "finite_boxcar"
            and (
                not PRODUCTION_THICKNESS_RANGE_UM[0]
                <= thickness
                <= PRODUCTION_THICKNESS_RANGE_UM[1]
                or not np.array_equal(
                    offsets,
                    _canonical_axial_offsets_v4(thickness),
                )
            )
        )
    ):
        raise ValueError("runtime finite-PSF schedule or thickness is unsupported")
    payload = {
        "schema_version": "anatomy-tracker.finite-psf-runtime-schedule/v4",
        "finite_psf_capability_receipt_sha256": expected_capability[
            "receipt_sha256"
        ],
        "family": "boxcar",
        "render_mode": mode,
        "nominal_cut_thickness_um": thickness,
        "axial_sample_count": int(offsets.size),
        "axial_offsets_um": offsets.tolist(),
        "axial_weights": weights.tolist(),
        "axial_step_um": step,
        "sampling_direction": SAMPLING_DIRECTION,
    }
    return {**payload, "receipt_sha256": acquisition_v2._payload_sha256(payload)}


def verify_runtime_schedule_contract_v4(contract, capability):
    verify_finite_psf_model_capability_v4(capability)
    if not isinstance(contract, dict):
        raise ValueError("runtime finite-PSF schedule contract is missing")
    expected = runtime_schedule_contract_v4(
        contract.get("axial_offsets_um", ()),
        contract.get("axial_weights", ()),
        capability=capability,
    )
    if contract != expected:
        raise ValueError("runtime finite-PSF schedule contract changed")
    return True
