"""Verified ONNX boundary for atlas-to-slice dense registration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .dense_registration_preprocessing import (
        MASK_CONTRACT_SHA256,
        MODEL_SHAPE,
        NATIVE_SHAPE,
        PAD_X,
        PREPROCESSING_CONTRACT_V2,
        numpy_cosine_mask_feather,
    )
    from .nonlinear_registration import SliceAtlasTransform2D
else:
    from dense_registration_preprocessing import (
        MASK_CONTRACT_SHA256,
        MODEL_SHAPE,
        NATIVE_SHAPE,
        PAD_X,
        PREPROCESSING_CONTRACT_V2,
        numpy_cosine_mask_feather,
    )
    from nonlinear_registration import SliceAtlasTransform2D


INPUT_NAMES = ("fixed_atlas_and_mask", "moving_slice_and_mask")
OUTPUT_NAMES = ("fixed_to_moving_map", "moving_to_fixed_map")
PRODUCTION_PROVIDER = "DmlExecutionProvider"
PINNED_ONNXRUNTIME_DIRECTML_VERSION = "1.24.4"
INPUT_CONTRACT = "two grayscale/mask tensors; channels=[image,brain-outline/tissue-mask]"
OUTPUT_CONTRACT = "absolute x,y pixel maps fixed->moving and moving->fixed"

DENSE_REGISTRATION_V2_RELEASE_PROTOCOL = {
    "protocol_version": 2,
    "benchmark_id": "allen-dense-registration-v2",
    "scope": (
        "Synthetic deformation, grayscale appearance, damage, and mask robustness "
        "on Allen CCFv3; no claim of real-histology accuracy."
    ),
    "generator_profile": "v2",
    "strata": ["clean", "mild", "hard"],
    "cohorts": {
        "qualification": {
            "split": "validation",
            "seeds": [83117, 83129],
            "samples_per_stratum": 128,
        },
        "mask_stress": {"samples_per_stratum": 64, "offsets": [-3, 3]},
        "sealed": {
            "split": "sealed-test",
            "seed": "cooperative-local one-shot CSPRNG uint31 committed before inference",
            "samples_per_stratum": 256,
        },
        "onnx_parity": {
            "seed": 1931771,
            "appearances": ["template", "label"],
            "mask_offsets": [-3, 0, 3],
            "production_provider": "DmlExecutionProvider",
            "diagnostic_provider": "CPUExecutionProvider",
            "onnxruntime_directml_version": PINNED_ONNXRUNTIME_DIRECTML_VERSION,
            "maximum_absolute_px": 0.05,
            "minimum_jacobian": 0.01,
            "metric_absolute_delta_bounds": {
                "foreground_correspondence": 0.002,
                "analytic_foreground_correspondence": 0.002,
                "macro_region_dice": 0.002,
                "boundary_f1_2px": 0.002,
                "boundary_mean_distance_px": 0.05,
                "endpoint_p50_px": 0.05,
                "endpoint_p95_px": 0.05,
                "endpoint_p99_px": 0.05,
                "inverse_endpoint_p50_px": 0.05,
                "inverse_endpoint_p95_px": 0.05,
                "inverse_endpoint_p99_px": 0.05,
                "damage_endpoint_p95_px": 0.05,
                "inverse_damage_endpoint_p95_px": 0.05,
                "inverse_cycle_p95_px": 0.05,
                "reverse_cycle_p95_px": 0.05,
                "fold_count": 0.0,
                "fold_fraction": 0.0,
                "jacobian_min": 0.05,
                "inverse_fold_count": 0.0,
                "inverse_fold_fraction": 0.0,
                "inverse_jacobian_min": 0.05,
            },
        },
    },
    "gates": {
        "policy": {
            "primary_target": (
                "At least 98% mean exact foreground Allen annotation-ID correspondence "
                "across the equally sampled clean, mild, and hard cohort."
            ),
            "stratum_floors": (
                "Severity-specific minimums prevent the overall mean from hiding a failed "
                "subgroup; they are robustness floors, not replacements for the 98% target."
            ),
        },
        "core": {
            "overall_metrics": {
                "foreground_correspondence": {"operator": ">=", "threshold": 0.98},
                "macro_region_dice": {"operator": ">=", "threshold": 0.95},
                "boundary_f1_2px": {"operator": ">=", "threshold": 0.95},
                "boundary_mean_distance_px": {"operator": "<=", "threshold": 1.0},
                "endpoint_p95_px": {"operator": "<=", "threshold": 2.0},
                "endpoint_p99_px": {"operator": "<=", "threshold": 4.0},
                "inverse_endpoint_p95_px": {"operator": "<=", "threshold": 2.0},
                "inverse_endpoint_p99_px": {"operator": "<=", "threshold": 4.0},
                "sample_endpoint_p95_q95_px": {"operator": "<=", "threshold": 3.0},
                "sample_inverse_endpoint_p95_q95_px": {"operator": "<=", "threshold": 3.0},
                "damage_endpoint_p95_px": {"operator": "<=", "threshold": 4.0},
                "inverse_damage_endpoint_p95_px": {"operator": "<=", "threshold": 4.0},
                "damage_pixel_count": {"operator": ">=", "threshold": 100.0},
                "inverse_damage_pixel_count": {"operator": ">=", "threshold": 100.0},
                "damaged_sample_count": {"operator": ">=", "threshold": 1.0},
                "inverse_damaged_sample_count": {"operator": ">=", "threshold": 1.0},
                "sample_foreground_correspondence_q05": {"operator": ">=", "threshold": 0.96},
                "sample_macro_region_dice_q05": {"operator": ">=", "threshold": 0.90},
                "inverse_cycle_p95_px": {"operator": "<=", "threshold": 1.0},
                "reverse_cycle_p95_px": {"operator": "<=", "threshold": 1.0},
                "fold_fraction": {"operator": "==", "threshold": 0.0},
                "inverse_fold_fraction": {"operator": "==", "threshold": 0.0},
                "jacobian_min": {"operator": ">=", "threshold": 0.01},
                "inverse_jacobian_min": {"operator": ">=", "threshold": 0.01},
            },
            "per_stratum": {
                "shared_metrics": {
                    "inverse_cycle_p95_px": {"operator": "<=", "threshold": 1.0},
                    "reverse_cycle_p95_px": {"operator": "<=", "threshold": 1.0},
                    "fold_fraction": {"operator": "==", "threshold": 0.0},
                    "inverse_fold_fraction": {"operator": "==", "threshold": 0.0},
                    "jacobian_min": {"operator": ">=", "threshold": 0.01},
                    "inverse_jacobian_min": {"operator": ">=", "threshold": 0.01},
                },
                "metrics": {
                    "clean": {
                        "foreground_correspondence": {"operator": ">=", "threshold": 0.985},
                        "macro_region_dice": {"operator": ">=", "threshold": 0.96},
                        "boundary_f1_2px": {"operator": ">=", "threshold": 0.97},
                        "boundary_mean_distance_px": {"operator": "<=", "threshold": 0.75},
                        "endpoint_p95_px": {"operator": "<=", "threshold": 1.5},
                        "inverse_endpoint_p95_px": {"operator": "<=", "threshold": 1.5},
                        "endpoint_p99_px": {"operator": "<=", "threshold": 3.0},
                        "inverse_endpoint_p99_px": {"operator": "<=", "threshold": 3.0},
                        "sample_endpoint_p95_q95_px": {"operator": "<=", "threshold": 2.0},
                        "sample_inverse_endpoint_p95_q95_px": {"operator": "<=", "threshold": 2.0},
                        "sample_foreground_correspondence_q05": {"operator": ">=", "threshold": 0.97},
                        "sample_macro_region_dice_q05": {"operator": ">=", "threshold": 0.93},
                    },
                    "mild": {
                        "foreground_correspondence": {"operator": ">=", "threshold": 0.975},
                        "macro_region_dice": {"operator": ">=", "threshold": 0.94},
                        "boundary_f1_2px": {"operator": ">=", "threshold": 0.95},
                        "boundary_mean_distance_px": {"operator": "<=", "threshold": 1.0},
                        "endpoint_p95_px": {"operator": "<=", "threshold": 2.25},
                        "inverse_endpoint_p95_px": {"operator": "<=", "threshold": 2.25},
                        "endpoint_p99_px": {"operator": "<=", "threshold": 4.5},
                        "inverse_endpoint_p99_px": {"operator": "<=", "threshold": 4.5},
                        "sample_endpoint_p95_q95_px": {"operator": "<=", "threshold": 3.5},
                        "sample_inverse_endpoint_p95_q95_px": {"operator": "<=", "threshold": 3.5},
                        "sample_foreground_correspondence_q05": {"operator": ">=", "threshold": 0.945},
                        "sample_macro_region_dice_q05": {"operator": ">=", "threshold": 0.87},
                    },
                    "hard": {
                        "foreground_correspondence": {"operator": ">=", "threshold": 0.95},
                        "macro_region_dice": {"operator": ">=", "threshold": 0.90},
                        "boundary_f1_2px": {"operator": ">=", "threshold": 0.90},
                        "boundary_mean_distance_px": {"operator": "<=", "threshold": 1.5},
                        "endpoint_p95_px": {"operator": "<=", "threshold": 3.0},
                        "inverse_endpoint_p95_px": {"operator": "<=", "threshold": 3.0},
                        "endpoint_p99_px": {"operator": "<=", "threshold": 6.0},
                        "inverse_endpoint_p99_px": {"operator": "<=", "threshold": 6.0},
                        "sample_endpoint_p95_q95_px": {"operator": "<=", "threshold": 4.5},
                        "sample_inverse_endpoint_p95_q95_px": {"operator": "<=", "threshold": 4.5},
                        "sample_foreground_correspondence_q05": {"operator": ">=", "threshold": 0.90},
                        "sample_macro_region_dice_q05": {"operator": ">=", "threshold": 0.80},
                    },
                },
            },
        },
        "appearance": {
            "groups": ["template", "label"],
            "metrics": {
                "foreground_correspondence": {"operator": ">=", "threshold": 0.98},
                "macro_region_dice": {"operator": ">=", "threshold": 0.95},
            },
        },
        "mask_offset_stress": {
            "groups": ["-3", "3"],
            "metrics": {
                "foreground_correspondence": {"operator": ">=", "threshold": 0.975},
                "macro_region_dice": {"operator": ">=", "threshold": 0.94},
                "endpoint_p95_px": {"operator": "<=", "threshold": 2.5},
                "inverse_endpoint_p95_px": {"operator": "<=", "threshold": 2.5},
                "fold_fraction": {"operator": "==", "threshold": 0.0},
                "inverse_fold_fraction": {"operator": "==", "threshold": 0.0},
                "jacobian_min": {"operator": ">=", "threshold": 0.01},
                "inverse_jacobian_min": {"operator": ">=", "threshold": 0.01},
            },
        },
    },
}
DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        DENSE_REGISTRATION_V2_RELEASE_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_value(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _metric_gate(metrics: dict, schema: dict) -> dict:
    checks = {}
    for name, rule in schema.items():
        value = metrics.get(name)
        operator, threshold = rule["operator"], rule["threshold"]
        checks[name] = bool(
            isinstance(value, (int, float))
            and np.isfinite(value)
            and (
                value >= threshold if operator == ">="
                else value <= threshold if operator == "<="
                else value == threshold if operator == "=="
                else False
            )
        )
    return {"passed": all(checks.values()), "checks": checks, "schema": schema}


def dense_registration_v2_gate_report(main: dict, mask_stress: dict) -> dict:
    protocol = DENSE_REGISTRATION_V2_RELEASE_PROTOCOL
    gates = protocol["gates"]
    overall_schema = gates["core"]["overall_metrics"]
    stratum_gate = gates["core"]["per_stratum"]
    shared_stratum_schema = stratum_gate["shared_metrics"]
    appearance_schema = gates["appearance"]["metrics"]
    stress_schema = gates["mask_offset_stress"]["metrics"]
    core = {
        "overall": _metric_gate(main.get("overall", {}), overall_schema),
        "per_stratum": {
            stratum: _metric_gate(
                main.get("per_stratum", {}).get(stratum, {}),
                {**shared_stratum_schema, **stratum_gate["metrics"][stratum]},
            )
            for stratum in protocol["strata"]
        },
    }
    appearance = {
        group: _metric_gate(
            main.get("appearance_subgroups", {}).get(group, {}), appearance_schema
        )
        for group in gates["appearance"]["groups"]
    }
    stress = {
        group: _metric_gate(
            mask_stress.get("mask_offset_subgroups", {}).get(group, {}), stress_schema
        )
        for group in gates["mask_offset_stress"]["groups"]
    }
    passed = (
        core["overall"]["passed"]
        and all(value["passed"] for value in core["per_stratum"].values())
        and all(value["passed"] for value in appearance.values())
        and all(value["passed"] for value in stress.values())
    )
    return {
        "release_protocol_sha256": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256,
        "passed": passed,
        "core": core,
        "appearance": appearance,
        "mask_offset_stress": stress,
    }


_METADATA_KEYS = {
    "format_version", "benchmark_id", "release_protocol",
    "release_protocol_sha256", "scope", "candidate",
    "candidate_payload_sha256", "candidate_file_sha256",
    "candidate_checkpoint_file_sha256", "checkpoint",
    "onnx_model_file_sha256", "model_shape", "model_config",
    "preprocessing_contract", "mask_contract_payload_sha256",
    "appearance_contract_sha256", "query_sha256",
    "generator_contract_payload_sha256", "generator_contract",
    "input_contract", "output_contract", "sealed_test",
    "production_provider", "onnxruntime_directml_version",
    "onnxruntime_parity", "environment", "source_file_sha256",
}

_EVALUATED_V2_METADATA_KEYS = {
    "format_version", "bundle_kind", "release_approved", "benchmark_id",
    "release_protocol_sha256", "scope", "checkpoint",
    "candidate_checkpoint_file_sha256", "onnx_model_file_sha256",
    "model_shape", "model_config", "preprocessing_contract",
    "mask_contract_payload_sha256", "input_contract", "output_contract",
    "qualification", "sealed_test", "production_provider",
    "onnxruntime_directml_version", "onnxruntime_parity", "environment",
    "source_file_sha256",
}
_EVALUATED_V2_QUALIFICATION_KEYS = {
    "status", "receipt_file_sha256", "receipt", "per_seed",
}
_EVALUATED_V2_SEED_SUMMARY_KEYS = {
    "foreground_correspondence", "macro_region_dice", "endpoint_p95_px",
    "release_gate_passed",
}


def _verify_v2_contract(metadata: dict) -> None:
    if set(metadata) != _METADATA_KEYS:
        raise RuntimeError("Dense-registration metadata schema is invalid")
    if (
        metadata["format_version"] != 2
        or metadata["benchmark_id"]
        != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["benchmark_id"]
        or metadata["release_protocol"] != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL
        or metadata["release_protocol_sha256"]
        != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256
        or tuple(metadata["model_shape"]) != MODEL_SHAPE
        or metadata["preprocessing_contract"] != PREPROCESSING_CONTRACT_V2
        or metadata["mask_contract_payload_sha256"] != MASK_CONTRACT_SHA256
        or metadata["input_contract"] != INPUT_CONTRACT
        or metadata["output_contract"] != OUTPUT_CONTRACT
    ):
        raise RuntimeError("Dense-registration v2 model contract is unsupported")
    if metadata["production_provider"] != PRODUCTION_PROVIDER:
        raise RuntimeError("Dense-registration production provider is unsupported")
    if (
        metadata["onnxruntime_directml_version"]
        != PINNED_ONNXRUNTIME_DIRECTML_VERSION
        or importlib.metadata.version("onnxruntime-directml")
        != PINNED_ONNXRUNTIME_DIRECTML_VERSION
    ):
        raise RuntimeError("Dense-registration DirectML version is unsupported")
    if not _sha256_value(metadata["candidate_checkpoint_file_sha256"]):
        raise RuntimeError("Dense-registration checkpoint identity is invalid")


def _verify_evaluated_v2_contract(metadata: dict) -> None:
    if set(metadata) != _EVALUATED_V2_METADATA_KEYS:
        raise RuntimeError("Evaluated dense-registration metadata schema is invalid")
    if (
        metadata["format_version"] != 2
        or metadata["bundle_kind"] != "evaluated-v2"
        or metadata["release_approved"] is not False
        or metadata["benchmark_id"]
        != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["benchmark_id"]
        or metadata["release_protocol_sha256"]
        != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256
        or metadata["scope"] != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["scope"]
        or tuple(metadata["model_shape"]) != MODEL_SHAPE
        or metadata["preprocessing_contract"] != PREPROCESSING_CONTRACT_V2
        or metadata["mask_contract_payload_sha256"] != MASK_CONTRACT_SHA256
        or metadata["input_contract"] != INPUT_CONTRACT
        or metadata["output_contract"] != OUTPUT_CONTRACT
    ):
        raise RuntimeError("Evaluated dense-registration v2 model contract is unsupported")
    if metadata["production_provider"] != PRODUCTION_PROVIDER:
        raise RuntimeError("Dense-registration production provider is unsupported")
    if (
        metadata["onnxruntime_directml_version"]
        != PINNED_ONNXRUNTIME_DIRECTML_VERSION
        or importlib.metadata.version("onnxruntime-directml")
        != PINNED_ONNXRUNTIME_DIRECTML_VERSION
    ):
        raise RuntimeError("Dense-registration DirectML version is unsupported")
    checkpoint_sha256 = metadata["candidate_checkpoint_file_sha256"]
    if not _sha256_value(checkpoint_sha256):
        raise RuntimeError("Dense-registration checkpoint identity is invalid")
    checkpoint = metadata["checkpoint"]
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_file_sha256") != checkpoint_sha256
    ):
        raise RuntimeError("Evaluated dense-registration checkpoint is inconsistent")

    qualification = metadata["qualification"]
    if (
        not isinstance(qualification, dict)
        or set(qualification) != _EVALUATED_V2_QUALIFICATION_KEYS
        or qualification["status"] != "rejected"
        or not _sha256_value(qualification["receipt_file_sha256"])
    ):
        raise RuntimeError("Evaluated dense-registration qualification is invalid")
    receipt = qualification["receipt"]
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "rejected"
        or receipt.get("checkpoint") != checkpoint
        or receipt.get("release_protocol_sha256")
        != DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256
    ):
        raise RuntimeError("Evaluated dense-registration qualification receipt is invalid")
    per_seed = qualification["per_seed"]
    expected_seeds = {
        str(seed)
        for seed in DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["cohorts"]["qualification"]["seeds"]
    }
    if not isinstance(per_seed, dict) or set(per_seed) != expected_seeds:
        raise RuntimeError("Evaluated dense-registration qualification summaries are invalid")
    gate_results = []
    for summary in per_seed.values():
        if not isinstance(summary, dict) or set(summary) != _EVALUATED_V2_SEED_SUMMARY_KEYS:
            raise RuntimeError("Evaluated dense-registration qualification summaries are invalid")
        metrics = [
            summary["foreground_correspondence"],
            summary["macro_region_dice"],
            summary["endpoint_p95_px"],
        ]
        if (
            any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                for value in metrics
            )
            or not isinstance(summary["release_gate_passed"], bool)
        ):
            raise RuntimeError("Evaluated dense-registration qualification summaries are invalid")
        gate_results.append(summary["release_gate_passed"])
    if all(gate_results):
        raise RuntimeError("Evaluated dense-registration qualification must be rejected")
    if metadata["sealed_test"] != {"status": "not_run"}:
        raise RuntimeError("Evaluated dense-registration sealed test must be marked not_run")

    parity = metadata["onnxruntime_parity"]
    if not isinstance(parity, dict):
        raise RuntimeError("Evaluated dense-registration CPU/DirectML parity did not pass")
    provider_aggregates = parity.get("provider_aggregates")
    required_providers = {PRODUCTION_PROVIDER, "CPUExecutionProvider"}
    if (
        parity.get("passed") is not True
        or parity.get("production_provider") != PRODUCTION_PROVIDER
        or parity.get("diagnostic_provider") != "CPUExecutionProvider"
        or parity.get("onnxruntime_directml_version")
        != PINNED_ONNXRUNTIME_DIRECTML_VERSION
        or not isinstance(provider_aggregates, dict)
        or set(provider_aggregates) != required_providers
        or any(
            not isinstance(provider_aggregates[provider], dict)
            or provider_aggregates[provider].get("passed") is not True
            for provider in required_providers
        )
    ):
        raise RuntimeError("Evaluated dense-registration CPU/DirectML parity did not pass")


def _load_bundle(
    model_path: str | Path,
    metadata_path: str | Path | None,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> tuple[Path, dict, str, str]:
    if (expected_model_sha256 is None) != (expected_metadata_sha256 is None):
        raise ValueError("Dense-registration bundle pins must be supplied together")
    if expected_model_sha256 is not None and (
        not _sha256_value(expected_model_sha256)
        or not _sha256_value(expected_metadata_sha256)
    ):
        raise ValueError("Dense-registration inference requires external SHA-256 pins")
    model_path = Path(model_path)
    metadata_path = Path(metadata_path) if metadata_path else model_path.with_suffix(".metadata.json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("Dense-registration ONNX model or metadata is unavailable")
    model_file_sha256 = sha256_file(model_path)
    metadata_file_sha256 = sha256_file(metadata_path)
    if expected_model_sha256 is not None and model_file_sha256 != expected_model_sha256:
        raise RuntimeError("Dense-registration model differs from the external SHA-256 pin")
    if expected_metadata_sha256 is not None and metadata_file_sha256 != expected_metadata_sha256:
        raise RuntimeError("Dense-registration metadata differs from the external SHA-256 pin")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("Dense-registration metadata is invalid")
    if metadata.get("onnx_model_file_sha256") != model_file_sha256:
        raise RuntimeError("Dense-registration ONNX checksum does not match its metadata")
    return model_path, metadata, model_file_sha256, metadata_file_sha256


def verify_dense_registration_v2_bundle(
    model_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> dict:
    _, metadata, model_file_sha256, metadata_file_sha256 = _load_bundle(
        model_path,
        metadata_path,
        expected_model_sha256,
        expected_metadata_sha256,
    )
    _verify_v2_contract(metadata)
    verified = dict(metadata)
    verified.update(
        model_file_sha256=model_file_sha256,
        metadata_file_sha256=metadata_file_sha256,
    )
    return verified


def verify_dense_registration_evaluated_bundle(
    model_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> dict:
    _, metadata, model_file_sha256, metadata_file_sha256 = _load_bundle(
        model_path,
        metadata_path,
        expected_model_sha256,
        expected_metadata_sha256,
    )
    _verify_evaluated_v2_contract(metadata)
    verified = dict(metadata)
    verified.update(
        model_file_sha256=model_file_sha256,
        metadata_file_sha256=metadata_file_sha256,
    )
    return verified


def verify_dense_registration_bundle(
    model_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> dict:
    _, metadata, model_file_sha256, metadata_file_sha256 = _load_bundle(
        model_path,
        metadata_path,
        expected_model_sha256,
        expected_metadata_sha256,
    )
    if metadata.get("bundle_kind") == "evaluated-v2":
        _verify_evaluated_v2_contract(metadata)
    else:
        _verify_v2_contract(metadata)
    verified = dict(metadata)
    verified.update(
        model_file_sha256=model_file_sha256,
        metadata_file_sha256=metadata_file_sha256,
    )
    return verified


def _native_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.shape != NATIVE_SHAPE or not np.all((values == 0) | (values == 1)):
        raise ValueError(f"Brain mask must be binary with shape {NATIVE_SHAPE}")
    return np.ascontiguousarray(values, dtype=bool)


def _normalized_atlas(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.shape != NATIVE_SHAPE:
        raise ValueError(f"Raw grayscale image must be a NumPy array with shape {NATIVE_SHAPE}")
    values = np.asarray(image, dtype=np.float32)
    if not np.isfinite(values).all() or not mask.any():
        raise ValueError("Raw grayscale image and brain mask must contain finite tissue pixels")
    low, high = np.quantile(values[mask], (0.005, 0.995))
    normalized = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32)
    normalized[~mask] = 0.0
    return normalized


def _normalized_slice_v2(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.shape != NATIVE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"Canonical slice image must be uint8 with shape {NATIVE_SHAPE}")
    return (
        np.ascontiguousarray(image, dtype=np.float32)
        * (numpy_cosine_mask_feather(mask) / 255.0)
    )


def _padded_tensor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tensor = np.zeros((1, 2, *MODEL_SHAPE), dtype=np.float32)
    tensor[0, 0, :, PAD_X : PAD_X + NATIVE_SHAPE[1]] = image
    tensor[0, 1, :, PAD_X : PAD_X + NATIVE_SHAPE[1]] = mask
    return tensor


def preprocess_dense_registration_inputs(
    atlas_image: np.ndarray,
    atlas_mask: np.ndarray,
    slice_image: np.ndarray,
    slice_mask: np.ndarray,
    *,
    preprocessing_contract: str = PREPROCESSING_CONTRACT_V2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build model tensors directly from native raw arrays, never GUI display curves."""
    atlas_mask = _native_mask(atlas_mask)
    slice_mask = _native_mask(slice_mask)
    fixed = _normalized_atlas(atlas_image, atlas_mask)
    if preprocessing_contract != PREPROCESSING_CONTRACT_V2:
        raise ValueError("Unsupported dense-registration preprocessing contract")
    moving = _normalized_slice_v2(slice_image, slice_mask)
    return _padded_tensor(fixed, atlas_mask), _padded_tensor(moving, slice_mask)


def native_absolute_map(model_output: np.ndarray) -> np.ndarray:
    output = np.asarray(model_output, dtype=np.float32)
    if output.shape != (1, 2, *MODEL_SHAPE) or not np.isfinite(output).all():
        raise RuntimeError(f"Dense-registration output must have shape {(1, 2, *MODEL_SHAPE)}")
    native = np.ascontiguousarray(
        output[0, :, :, PAD_X : PAD_X + NATIVE_SHAPE[1]].transpose(1, 2, 0)
    )
    native[..., 0] -= PAD_X
    return native


def _check_session_contract(session) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if tuple(value.name for value in inputs) != INPUT_NAMES or tuple(value.name for value in outputs) != OUTPUT_NAMES:
        raise RuntimeError("Dense-registration ONNX input/output names differ from the runtime contract")
    for value in (*inputs, *outputs):
        shape = tuple(value.shape)
        if (
            len(shape) != 4
            or shape[0] not in (None, 1, "batch")
            or tuple(shape[1:]) != (2, *MODEL_SHAPE)
            or getattr(value, "type", "tensor(float)") != "tensor(float)"
        ):
            raise RuntimeError("Dense-registration ONNX tensor shapes differ from the runtime contract")


def run_dense_registration(
    model_path: str | Path,
    atlas_image: np.ndarray,
    atlas_mask: np.ndarray,
    slice_image: np.ndarray,
    slice_mask: np.ndarray,
    *,
    expected_model_sha256: str,
    expected_metadata_sha256: str,
    metadata_path: str | Path | None = None,
    ort_module=None,
) -> dict:
    if not _sha256_value(expected_model_sha256) or not _sha256_value(
        expected_metadata_sha256
    ):
        raise ValueError("Dense-registration inference requires external SHA-256 pins")
    verified = verify_dense_registration_bundle(
        model_path,
        metadata_path,
        expected_model_sha256=expected_model_sha256,
        expected_metadata_sha256=expected_metadata_sha256,
    )
    fixed, moving = preprocess_dense_registration_inputs(
        atlas_image,
        atlas_mask,
        slice_image,
        slice_mask,
        preprocessing_contract=verified["preprocessing_contract"],
    )
    if ort_module is None:
        import onnxruntime as ort_module
    if getattr(ort_module, "__version__", None) != PINNED_ONNXRUNTIME_DIRECTML_VERSION:
        raise RuntimeError("Dense-registration DirectML runtime version is unsupported")
    if PRODUCTION_PROVIDER not in ort_module.get_available_providers():
        raise RuntimeError(
            "The required DmlExecutionProvider is unavailable"
        )
    session = ort_module.InferenceSession(
        str(model_path), providers=[PRODUCTION_PROVIDER]
    )
    if session.get_providers()[0] != PRODUCTION_PROVIDER:
        raise RuntimeError("ONNX Runtime did not activate the required primary provider")
    _check_session_contract(session)
    forward, inverse = session.run(
        list(OUTPUT_NAMES),
        {INPUT_NAMES[0]: fixed, INPUT_NAMES[1]: moving},
    )
    atlas_to_affine = native_absolute_map(forward)
    affine_to_atlas = native_absolute_map(inverse)
    atlas_mask_native = _native_mask(atlas_mask)
    slice_mask_native = _native_mask(slice_mask)
    valid = atlas_mask_native & cv2.remap(
        slice_mask_native.astype(np.uint8),
        atlas_to_affine[..., 0],
        atlas_to_affine[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    output_diagnostics = SliceAtlasTransform2D(
        np.eye(3),
        NATIVE_SHAPE,
        NATIVE_SHAPE,
        atlas_to_affine,
        affine_to_atlas,
        valid,
    ).check_invariants()
    runtime_metadata = {
        "method": (
            "dense-registration-onnx-evaluated-v2"
            if verified.get("bundle_kind") == "evaluated-v2"
            else "dense-registration-onnx-v2"
        ),
        "model_file_sha256": verified["model_file_sha256"],
        "metadata_file_sha256": verified["metadata_file_sha256"],
        "candidate_checkpoint_file_sha256": verified[
            "candidate_checkpoint_file_sha256"
        ],
        "provider": session.get_providers()[0],
        "preprocessing_contract": verified["preprocessing_contract"],
        "map_contract": "native absolute atlas->affine and affine->atlas x,y pixel maps",
        "output_diagnostics": output_diagnostics,
    }
    if verified.get("bundle_kind") == "evaluated-v2":
        runtime_metadata.update(
            bundle_kind="evaluated-v2",
            release_approved=False,
            qualification_status="rejected",
        )
    return {
        "atlas_to_affine_xy": atlas_to_affine,
        "affine_to_atlas_xy": affine_to_atlas,
        "valid_atlas_mask": np.ascontiguousarray(valid),
        "metadata": runtime_metadata,
        "registration_metadata_json": json.dumps(
            runtime_metadata, sort_keys=True, separators=(",", ":")
        ),
    }
