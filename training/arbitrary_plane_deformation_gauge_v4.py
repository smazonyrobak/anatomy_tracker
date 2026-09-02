"""Direct affine-free pullback targets with decoder-identical integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_deformation_primitives import (
    scaling_and_squaring_yx,
    support_weighted_affine_projection_yx,
)


DEFORMATION_GAUGE_V4_SCHEMA = "anatomy-tracker.direct-deformation-target/v4"
DEFORMATION_GAUGE_V4_ALGORITHM = (
    "preintegration-uniform-canvas-affine-free-source-to-fixed-pullback-certification/v4"
)
PROJECTION_WEIGHTING = "fixed uniform full canvas, matching decoder gauge"
TARGET_DIRECTION = (
    "source-to-fixed pullback exp(-v); target stationary velocity is -v"
)
NUMERIC_CONTRACT = (
    "float32; y-x channel order; absolute pixel-centre map; align_corners=True; "
    "border displacement composition; exactly seven scaling-and-squaring steps"
)
INTEGRATION_STEPS = 7
MAXIMUM_AFFINE_COEFFICIENT_ABS = 1e-4
MAXIMUM_CERTIFICATION_ERROR_PX = 1e-6
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_deformation_gauge_v4.py",
    "arbitrary_plane_deformation_primitives.py",
)


def _source_hashes():
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _valid_mask(value, shape):
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError("direct target validity mask shape disagrees")
    if np.issubdtype(raw.dtype, np.number) and not np.isfinite(raw).all():
        raise ValueError("direct target validity mask must be finite")
    if raw.dtype != np.bool_ and not np.all((raw == 0) | (raw == 1)):
        raise ValueError("direct target validity mask must be boolean")
    valid = raw.astype(bool, copy=False)
    if not valid.any():
        raise ValueError("direct target validity mask must contain a valid pixel")
    return np.ascontiguousarray(valid)


def direct_deformation_target_receipt_v4(artifact):
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "implementation_source_sha256": artifact[
            "implementation_source_sha256"
        ],
        "projection_weighting": artifact["projection_weighting"],
        "target_direction": artifact["target_direction"],
        "numeric_contract": artifact["numeric_contract"],
        "runtime_versions": artifact["runtime_versions"],
        "integration_steps": artifact["integration_steps"],
        "maximum_affine_coefficient_abs": artifact[
            "maximum_affine_coefficient_abs"
        ],
        "maximum_certification_error_px": artifact[
            "maximum_certification_error_px"
        ],
        "input_array_receipts": artifact["input_array_receipts"],
        "array_receipts": artifact["array_receipts"],
        "diagnostics": artifact["diagnostics"],
        "direct_deformation_target_id": artifact[
            "direct_deformation_target_id"
        ],
    }


def direct_deformation_target_summary_v4(artifact):
    return {
        **direct_deformation_target_receipt_v4(artifact),
        "receipt_sha256": artifact["receipt_sha256"],
    }


def direct_deformation_target_reference_v4(artifact):
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "projection_weighting": artifact["projection_weighting"],
        "target_direction": artifact["target_direction"],
        "numeric_contract": artifact["numeric_contract"],
        "runtime_versions": artifact["runtime_versions"],
        "direct_deformation_target_id": artifact[
            "direct_deformation_target_id"
        ],
        "receipt_sha256": artifact["receipt_sha256"],
    }


def direct_deformation_target_contract_v4():
    return {
        "schema_version": DEFORMATION_GAUGE_V4_SCHEMA,
        "algorithm": DEFORMATION_GAUGE_V4_ALGORITHM,
        "projection_weighting": PROJECTION_WEIGHTING,
        "target_direction": TARGET_DIRECTION,
        "numeric_contract": NUMERIC_CONTRACT,
        "runtime_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def certify_direct_deformation_target_v4(
    target_pullback_stationary_velocity_yx,
    source_to_fixed_pullback_map_yx,
    parent_effective_quicknii_ouv,
    valid_mask,
):
    velocity = np.asarray(
        target_pullback_stationary_velocity_yx, dtype=np.float32
    )
    pullback = np.asarray(source_to_fixed_pullback_map_yx, dtype=np.float32)
    pose = np.asarray(parent_effective_quicknii_ouv, dtype=np.float64).reshape(3, 3)
    if (
        velocity.ndim != 3
        or velocity.shape[-1] != 2
        or pullback.shape != velocity.shape
        or not np.isfinite(velocity).all()
        or not np.isfinite(pullback).all()
        or not np.isfinite(pose).all()
    ):
        raise ValueError(
            "direct target velocity, source-to-fixed pullback, and parent pose must be finite"
        )
    valid = _valid_mask(valid_mask, velocity.shape[:2])
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(velocity, -1, 0))
    ).unsqueeze(0)
    support = torch.ones(
        (1, 1, *velocity.shape[:2]), dtype=torch.float32
    )
    projected, removed, post, _ = support_weighted_affine_projection_yx(
        tensor, support
    )
    certified = scaling_and_squaring_yx(tensor, INTEGRATION_STEPS)
    certified_hwc = np.ascontiguousarray(
        np.moveaxis(certified[0].detach().cpu().numpy(), 0, -1),
        dtype=np.float32,
    )
    error = np.linalg.norm(
        certified_hwc.astype(np.float64) - pullback.astype(np.float64), axis=-1
    )
    finite_valid = valid & np.isfinite(error)
    if not finite_valid.any():
        raise ValueError("direct target has no finite valid certification pixels")
    affine_max = float(removed.abs().max().item())
    post_max = float(post.abs().max().item())
    certification_max = float(error[finite_valid].max())
    if (
        affine_max > MAXIMUM_AFFINE_COEFFICIENT_ABS
        or post_max > MAXIMUM_AFFINE_COEFFICIENT_ABS
        or certification_max > MAXIMUM_CERTIFICATION_ERROR_PX
    ):
        raise ValueError(
            "direct pullback target is not in the certified decoder gauge"
        )
    arrays = {
        "target_pullback_stationary_velocity_yx_px_float32": np.ascontiguousarray(
            velocity
        ),
        "source_to_fixed_pullback_map_yx_px_float32": np.ascontiguousarray(
            pullback
        ),
        "certified_pullback_map_yx_px_float32": certified_hwc,
        "parent_effective_quicknii_ouv_float64": np.ascontiguousarray(pose),
        "uniform_canvas_affine_coefficients_yx_float32": np.ascontiguousarray(
            removed[0].detach().cpu().numpy(), dtype=np.float32
        ),
        "postprojection_affine_coefficients_yx_float32": np.ascontiguousarray(
            post[0].detach().cpu().numpy(), dtype=np.float32
        ),
        "projected_stationary_velocity_yx_px_float32": np.ascontiguousarray(
            np.moveaxis(projected[0].detach().cpu().numpy(), 0, -1),
            dtype=np.float32,
        ),
        "valid_mask": valid,
    }
    input_receipts = {
        name: acquisition._array_receipt(value)
        for name, value in {
            "target_pullback_stationary_velocity_yx_px_float32": velocity,
            "source_to_fixed_pullback_map_yx_px_float32": pullback,
            "parent_effective_quicknii_ouv_float64": pose,
            "valid_mask": valid,
        }.items()
    }
    diagnostics = {
        "uniform_canvas_affine_coefficient_max_abs": affine_max,
        "postprojection_affine_coefficient_max_abs": post_max,
        "valid_certification_error_mean_px": float(error[finite_valid].mean()),
        "valid_certification_error_max_px": certification_max,
        "parent_pose_adjustment_max_abs": 0.0,
    }
    artifact = {
        "schema_version": DEFORMATION_GAUGE_V4_SCHEMA,
        "algorithm": DEFORMATION_GAUGE_V4_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "projection_weighting": PROJECTION_WEIGHTING,
        "target_direction": TARGET_DIRECTION,
        "numeric_contract": NUMERIC_CONTRACT,
        "runtime_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "integration_steps": INTEGRATION_STEPS,
        "maximum_affine_coefficient_abs": MAXIMUM_AFFINE_COEFFICIENT_ABS,
        "maximum_certification_error_px": MAXIMUM_CERTIFICATION_ERROR_PX,
        "input_array_receipts": input_receipts,
        "arrays": arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in arrays.items()
        },
        "diagnostics": diagnostics,
    }
    artifact["direct_deformation_target_id"] = acquisition._payload_sha256(
        {
            "domain": DEFORMATION_GAUGE_V4_SCHEMA,
            "implementation_source_sha256": artifact[
                "implementation_source_sha256"
            ],
            "projection_weighting": artifact["projection_weighting"],
            "target_direction": artifact["target_direction"],
            "numeric_contract": artifact["numeric_contract"],
            "runtime_versions": artifact["runtime_versions"],
            "input_array_receipts": input_receipts,
            "array_receipts": artifact["array_receipts"],
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        direct_deformation_target_receipt_v4(artifact)
    )
    return artifact


def replay_direct_deformation_target_v4(artifact):
    arrays = artifact["arrays"]
    return certify_direct_deformation_target_v4(
        arrays["target_pullback_stationary_velocity_yx_px_float32"],
        arrays["source_to_fixed_pullback_map_yx_px_float32"],
        arrays["parent_effective_quicknii_ouv_float64"],
        arrays["valid_mask"],
    )


def verify_direct_deformation_target_v4(artifact):
    if (
        artifact.get("schema_version") != DEFORMATION_GAUGE_V4_SCHEMA
        or artifact.get("algorithm") != DEFORMATION_GAUGE_V4_ALGORITHM
        or artifact.get("implementation_source_sha256") != _source_hashes()
        or artifact.get("receipt_sha256")
        != acquisition._payload_sha256(
            direct_deformation_target_receipt_v4(artifact)
        )
        or artifact.get("array_receipts")
        != {
            name: acquisition._array_receipt(value)
            for name, value in artifact.get("arrays", {}).items()
        }
    ):
        raise ValueError("direct deformation target receipt or source binding changed")
    replay = replay_direct_deformation_target_v4(artifact)
    if (
        direct_deformation_target_receipt_v4(replay)
        != direct_deformation_target_receipt_v4(artifact)
        or set(replay["arrays"]) != set(artifact["arrays"])
        or any(
            np.asarray(replay["arrays"][name]).dtype
            != np.asarray(artifact["arrays"][name]).dtype
            or not np.array_equal(
                replay["arrays"][name], artifact["arrays"][name]
            )
            for name in replay["arrays"]
        )
    ):
        raise ValueError("direct deformation target does not replay exactly")
    return True


__all__ = [
    "DEFORMATION_GAUGE_V4_ALGORITHM",
    "DEFORMATION_GAUGE_V4_SCHEMA",
    "INTEGRATION_STEPS",
    "MAXIMUM_AFFINE_COEFFICIENT_ABS",
    "MAXIMUM_CERTIFICATION_ERROR_PX",
    "NUMERIC_CONTRACT",
    "PROJECTION_WEIGHTING",
    "TARGET_DIRECTION",
    "certify_direct_deformation_target_v4",
    "direct_deformation_target_receipt_v4",
    "direct_deformation_target_contract_v4",
    "direct_deformation_target_reference_v4",
    "direct_deformation_target_summary_v4",
    "replay_direct_deformation_target_v4",
    "verify_direct_deformation_target_v4",
]
