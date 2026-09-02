"""Truth-free, receipt-bound inference for the standalone arbitrary-plane v6 model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from training import arbitrary_plane_receipt_bound_training_runner_v6 as runner_v6
from training import arbitrary_plane_staged_trainer_v6 as trainer_v6


INFERENCE_V6_SCHEMA = "anatomy-tracker-arbitrary-plane-inference-v6"
INPUT_MODES_V6 = ("raw", "black-exterior", "imperfect-mask")
CASE_ID_KEYS_V6 = (
    "animal_id",
    "specimen_id",
    "experiment_id",
    "section_id",
    "synthetic_animal_id",
)
_SOURCE_FILES = (
    "training/arbitrary_plane_inference_v6.py",
    "training/arbitrary_plane_receipt_bound_training_runner_v6.py",
    "training/arbitrary_plane_staged_trainer_v6.py",
    "training/arbitrary_plane_joint_model_v6.py",
    "training/arbitrary_plane_recurrent_model_v6.py",
    "training/arbitrary_plane_catalogue_runtime_v6.py",
)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _source_receipts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _context_seals(context: Mapping[str, object]) -> dict[str, object]:
    return {
        "manifest_sha256": _sha256_json(context["manifest"]),
        "run_state_sha256": _sha256_json(context["run_state"]),
        "catalogue_sha256": _sha256_json(context["catalogue"]),
        "decoded_atlas": _array_receipt(np.asarray(context["atlas_volume"])),
    }


def _i_path(value, *, must_exist: bool) -> Path:
    path = Path(value).resolve()
    if os.path.splitdrive(str(path))[0].upper() != "I:":
        raise ValueError("v6 inference file I/O is restricted to I:")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_path(context: Mapping[str, object]) -> Path:
    root = _i_path(context["run_directory"], must_exist=True)
    relative = context["run_state"]["latest_checkpoint"]["relative_path"]
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("v6 inference checkpoint escaped its authenticated run")
    return _i_path(path, must_exist=True)


def _verify_fresh_lineage(context, checkpoint) -> None:
    manifest = context["manifest"]
    checkpoint_manifest = checkpoint["manifest"]
    if (
        manifest.get("initialization") != "fresh_random_only"
        or checkpoint_manifest.get("initialization") != "fresh_random_only"
        or checkpoint.get("initialization_receipt", {}).get("algorithm")
        != "fresh-pytorch-random-initialization"
        or checkpoint.get("learned_dependencies")
        != {"model_weights": [], "features": [], "pseudolabels": []}
        or any(
            checkpoint_manifest.get(key) != []
            for key in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
        or checkpoint.get("probabilities_calibrated") is not False
        or checkpoint.get("uncertainty_status") != "raw_uncalibrated"
    ):
        raise ValueError("v6 inference accepts only fresh-lineage raw-uncalibrated runs")


def load_arbitrary_plane_inference_v6(
    run_directory,
    *,
    expected_run_manifest_receipt_sha256: str,
    expected_inference_source_sha256: Mapping[str, str],
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Authenticate a run and code closure against caller-held trust anchors."""
    if not _is_sha256(expected_run_manifest_receipt_sha256):
        raise ValueError("a trusted lowercase SHA-256 run-manifest receipt is required")
    if not isinstance(expected_inference_source_sha256, Mapping):
        raise ValueError("a trusted inference-source receipt map is required")
    trusted_source = dict(expected_inference_source_sha256)
    if (
        set(trusted_source) != set(_SOURCE_FILES)
        or any(not _is_sha256(value) for value in trusted_source.values())
        or trusted_source != _source_receipts()
    ):
        raise ValueError("inference sources differ from the trusted external receipt map")
    root = _i_path(run_directory, must_exist=True)
    context = runner_v6.load_receipt_bound_training_run_v6(
        root,
        expected_run_manifest_receipt_sha256=expected_run_manifest_receipt_sha256,
        device=device,
    )
    checkpoint = trainer_v6.load_staged_checkpoint_v6(_checkpoint_path(context))
    trainer_v6.verify_staged_checkpoint_v6(checkpoint)
    _verify_fresh_lineage(context, checkpoint)
    model = context["trainer_state"]["model"]
    restored = model.state_dict()
    if set(restored) != set(checkpoint["model_state"]) or any(
        not torch.equal(restored[name].detach().cpu(), checkpoint["model_state"][name])
        for name in restored
    ):
        raise ValueError("restored v6 model differs from its authenticated checkpoint")
    model.eval()
    return {
        "schema_version": INFERENCE_V6_SCHEMA,
        "context": context,
        "checkpoint": checkpoint,
        "model": model,
        "run_manifest_receipt_sha256": context["manifest"]["receipt_sha256"],
        "run_state_receipt_sha256": context["run_state"]["receipt_sha256"],
        "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
        "trusted_inference_source_sha256": trusted_source,
        "authenticated_context_seals": _context_seals(context),
    }


def _case_ids(value) -> dict[str, str | None]:
    supplied = {} if value is None else dict(value)
    if set(supplied) - set(CASE_ID_KEYS_V6) or any(
        item is not None and (not isinstance(item, str) or not item)
        for item in supplied.values()
    ):
        raise ValueError("case IDs must be nonempty strings or null and use only v6 ID keys")
    return {key: supplied.get(key) for key in CASE_ID_KEYS_V6}


def _scalar_image(value) -> np.ndarray:
    image = np.asarray(value)
    if (
        image.ndim != 2
        or image.dtype.kind not in "fiu"
        or not np.isfinite(image).all()
        or np.any((image < 0.0) | (image > 1.0))
    ):
        raise ValueError("image must be one finite normalized [0,1] scalar HxW array")
    return np.ascontiguousarray(image, dtype=np.float32)


def _prepare_input(image, input_mode, outline, outline_available):
    raw = _scalar_image(image)
    if input_mode not in INPUT_MODES_V6 or not isinstance(outline_available, bool):
        raise ValueError("input mode or explicit outline availability is invalid")
    if input_mode == "raw":
        if outline_available or outline is not None:
            raise ValueError("raw mode requires an explicitly unavailable, absent outline")
        mask = np.zeros(raw.shape, dtype=np.float32)
        return raw, mask, raw.copy()
    if not outline_available or outline is None:
        raise ValueError("assisted modes require an explicitly available user outline")
    mask = np.asarray(outline)
    if mask.shape != raw.shape or mask.dtype.kind not in "bifu" or not np.isfinite(mask).all():
        raise ValueError("outline must be a finite binary array matching the image")
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    if not np.logical_or(mask == 0.0, mask == 1.0).all():
        raise ValueError("outline must be exactly binary")
    model_image = np.ascontiguousarray(raw * mask, dtype=np.float32)
    model_image[mask == 0.0] = 0.0
    return raw, mask, model_image


def _detach_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    return value


def run_arbitrary_plane_inference_v6(
    loaded: Mapping[str, object],
    image,
    *,
    input_mode: str,
    outline,
    outline_available: bool,
    physical_fov_y_x_um,
    pixel_size_y_x_um,
    nominal_cut_thickness_um: float,
    axial_offsets_um,
    axial_weights,
    case_ids: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Run one real section through the authenticated model with no truth path."""
    if loaded.get("schema_version") != INFERENCE_V6_SCHEMA:
        raise ValueError("loaded inference state is not v6")
    context = loaded["context"]
    checkpoint = loaded["checkpoint"]
    trainer_v6.verify_staged_checkpoint_v6(checkpoint)
    _verify_fresh_lineage(context, checkpoint)
    if (
        loaded.get("run_manifest_receipt_sha256")
        != context["manifest"].get("receipt_sha256")
        or loaded.get("run_state_receipt_sha256")
        != context["run_state"].get("receipt_sha256")
        or loaded.get("checkpoint_receipt_sha256") != checkpoint.get("receipt_sha256")
        or loaded.get("checkpoint_model_state_sha256")
        != checkpoint.get("model_state_sha256")
        or loaded.get("trusted_inference_source_sha256") != _source_receipts()
        or loaded.get("authenticated_context_seals") != _context_seals(context)
    ):
        raise ValueError("v6 inference run or source binding changed after loading")
    model = loaded["model"]
    state = model.state_dict()
    if set(state) != set(checkpoint["model_state"]) or any(
        not torch.equal(state[name].detach().cpu(), checkpoint["model_state"][name])
        for name in state
    ):
        raise ValueError("v6 inference model state changed after authentication")

    raw, mask, model_image = _prepare_input(
        image, input_mode, outline, outline_available
    )
    geometry = context["catalogue"]["support_geometry"]
    output_shape = tuple(raw.shape)
    fov = tuple(float(item) for item in physical_fov_y_x_um)
    pixels = tuple(float(item) for item in pixel_size_y_x_um)
    expected_fov = tuple(float(item) for item in geometry["raster_physical_span_y_x_um"])
    expected_pixels = (
        expected_fov[0] / output_shape[0],
        expected_fov[1] / output_shape[1],
    )
    if fov != expected_fov or pixels != expected_pixels:
        raise ValueError("physical FOV and pixel geometry must exactly describe the image")

    thickness = float(nominal_cut_thickness_um)
    offsets = np.ascontiguousarray(axial_offsets_um, dtype=np.float64)
    weights = np.ascontiguousarray(axial_weights, dtype=np.float64)
    expected_offsets = np.linspace(-thickness / 2.0, thickness / 2.0, 9)
    expected_weights = np.asarray([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float64) / 16.0
    if (
        not 25.0 <= thickness <= 100.0
        or offsets.shape != (9,)
        or weights.shape != (9,)
        or not np.array_equal(offsets, expected_offsets)
        or not np.array_equal(weights, expected_weights)
    ):
        raise ValueError("inference requires the exact normalized finite S=9 PSF schedule")

    ids = _case_ids(case_ids)
    input_payload = {
        "input_mode": input_mode,
        "outline_available": outline_available,
        "raw_image": _array_receipt(raw),
        "outline": None if outline is None else _array_receipt(mask),
        "model_image": _array_receipt(model_image),
        "physical_fov_y_x_um": list(fov),
        "pixel_size_y_x_um": list(pixels),
        "nominal_cut_thickness_um": thickness,
        "axial_offsets_um": offsets.tolist(),
        "axial_weights": weights.tolist(),
        "case_ids": ids,
    }
    model.eval()
    device = next(model.parameters()).device
    config = context["manifest"]["training_config"]
    with torch.no_grad():
        output = model(
            torch.from_numpy(model_image)[None, None].to(device),
            torch.from_numpy(mask)[None, None].to(device),
            torch.tensor([outline_available], dtype=torch.bool, device=device),
            torch.as_tensor(context["atlas_volume"], dtype=torch.float32, device=device),
            context["catalogue_runtime"].expand(1),
            output_shape,
            tuple(config["retrieval_shape_h_w"]),
            tuple(geometry["origin_ap_dv_ml_um"]),
            tuple(geometry["voxel_size_ap_dv_ml_um"]),
            torch.from_numpy(offsets.astype(np.float32)).to(device),
            torch.from_numpy(weights.astype(np.float32)).to(device),
            proposal_top_m=int(config["proposal_top_m"]),
            top_k=int(config["top_k"]),
            refinement_steps=int(config["refinement_steps"]),
        )
    output = _detach_cpu(output)
    cascade = output["cascade"]
    refined = output["refined_output"]
    run_binding = {
        "run_manifest_receipt_sha256": loaded["run_manifest_receipt_sha256"],
        "run_state_receipt_sha256": loaded["run_state_receipt_sha256"],
        "checkpoint_receipt_sha256": loaded["checkpoint_receipt_sha256"],
        "checkpoint_model_state_sha256": loaded["checkpoint_model_state_sha256"],
        "catalogue_receipt_sha256": context["catalogue"]["receipt_sha256"],
    }
    return {
        "schema_version": INFERENCE_V6_SCHEMA,
        "probabilities_calibrated": False,
        "probability_status": "raw_uncalibrated",
        "input_receipt": {**input_payload, "receipt_sha256": _sha256_json(input_payload)},
        "run_binding": run_binding,
        "trusted_inference_source_sha256": dict(
            loaded["trusted_inference_source_sha256"]
        ),
        "posterior": {
            "raw_full_catalogue_proposal_log_probability": cascade[
                "raw_full_catalogue_proposal_log_probability"
            ],
            "honest_hybrid_posterior": cascade["honest_hybrid_posterior"],
        },
        "k_poses": {
            "catalogue_index": output["refinement_selected_catalogue_index"],
            "cell_id": output["refinement_selected_cell_id"],
            "pose": None if refined is None else refined["pose"],
        },
        "recurrent_output": refined,
        "deformation": None
        if refined is None
        else {
            key: value
            for key, value in refined.items()
            if "deformation" in key or "deformed" in key
        },
        "abstention": {
            "ready_mask": output["refinement_ready_mask"],
            "abstained_mask": output["refinement_abstained_mask"],
            "reason": cascade["honest_refinement_abstention_reason"],
        },
    }


__all__ = [
    "CASE_ID_KEYS_V6",
    "INFERENCE_V6_SCHEMA",
    "INPUT_MODES_V6",
    "load_arbitrary_plane_inference_v6",
    "run_arbitrary_plane_inference_v6",
]
