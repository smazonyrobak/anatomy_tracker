"""Provenance-checked standalone inference for the fresh arbitrary-plane model."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import torch

import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_psf_v4 as psf_v4
from training.arbitrary_plane_coarse_proposal_v5 import COARSE_PROPOSAL_GEOMETRY
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel
from training.arbitrary_plane_recurrent_model import (
    _VERIFIED_CATALOGUE_FEATURE_CACHE_TOKEN,
)
from training.arbitrary_plane_staged_training import (
    STAGED_TRAINING_EXPORT_SCHEMA,
    verify_staged_training_export_receipt_v3,
)
from training.arbitrary_plane_uncertainty_v3 import (
    posterior_summary_v3,
    propagate_electrode_trajectory_v3,
    verify_temperature_calibration_receipt_v3,
)


CHECKPOINT_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-joint-checkpoint/v3"
INFERENCE_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-inference/v3"
INFERENCE_CONTRACT_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-inference-contract/v3"
RUNTIME_INFERENCE_CONTRACT_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-runtime-inference-contract/v4"
)
ATLAS_SEMANTICS_V3_SCHEMA = "anatomy-tracker.atlas-semantics/v3"
CATALOGUE_FEATURE_CACHE_V3_SCHEMA = (
    "anatomy-tracker.complete-catalogue-atlas-feature-cache/v3"
)
INFERENCE_SESSION_V3_SCHEMA = "anatomy-tracker.verified-inference-session/v3"
CACHE_NUMERICAL_ATOL = 2e-6
CACHE_NUMERICAL_RTOL = 2e-6
ARCHITECTURE_NAME = "ArbitraryPlaneJointModel"
ARCHITECTURE_MODULE = "training.arbitrary_plane_joint_model"
INFERENCE_SOURCE_FILES = (
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_deformation_primitives.py",
    "training/arbitrary_plane_coarse_proposal_v5.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_joint_model.py",
    "training/arbitrary_plane_acquisition_v2.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_psf_v4.py",
    "training/arbitrary_plane_uncertainty_v3.py",
    "training/arbitrary_plane_staged_training.py",
    "training/arbitrary_plane_inference_v3.py",
)
FEATURE_RECIPE_V3 = {
    "schema_version": "anatomy-tracker.arbitrary-plane-feature-recipe/v3",
    "input_channel_names": [
        "brightfield_intensity",
        "outline_mask",
        "outline_availability_constant_binary_plane",
    ],
    "input_channel_count": 3,
    "model_image_channels": ["brightfield_intensity", "outline_mask"],
    "outline_availability_extraction": "channel_2[0,0] after exact constant-binary-plane verification",
    "numeric_preprocessing": "caller supplies scalar brightfield and outline in [0,1]; no learned normalization; values are cast to checkpoint dtype without rescaling",
    "photometric_contract": "raw color/dtype conversion is upstream and must yield one scalar brightfield channel in [0,1]; 0-255 floats are invalid",
    "spatial_recipe": "input is already placed on the immutable catalogue canonical H-W canvas; inference performs no resize, crop, or segmentation",
    "external_or_legacy_feature_dependencies": [],
}
CHECKPOINT_FINITE_PSF_CAPABILITY_SCOPE_V4 = {
    "schedule_scope": "caller-explicit-known-thickness-at-session-or-cache-creation",
    "normalization": psf_v4.NORMALIZATION,
    "unknown_thickness_policy": "reject",
}
_MODEL_KEYS = {
    name
    for name in inspect.signature(ArbitraryPlaneJointModel).parameters
    if name != "self"
}
_INFERENCE_SESSION_TOKEN = object()


def _verify_training_export_receipt(
    receipt,
    *,
    model_kwargs,
    catalogue_id,
    catalogue_receipt_sha256,
    catalogue_cell_count,
    model_state_sha256,
    require_source_file=False,
):
    """Dispatch only between the authenticated v3 and finite-v4 training receipts."""
    schema = receipt.get("schema_version") if isinstance(receipt, dict) else None
    arguments = {
        "model_kwargs": model_kwargs,
        "catalogue_id": catalogue_id,
        "catalogue_receipt_sha256": catalogue_receipt_sha256,
        "catalogue_cell_count": catalogue_cell_count,
        "model_state_sha256": model_state_sha256,
        "require_source_file": require_source_file,
    }
    if schema == STAGED_TRAINING_EXPORT_SCHEMA:
        return verify_staged_training_export_receipt_v3(receipt, **arguments)
    from training.arbitrary_plane_finite_training_runner_v4 import (
        FINITE_STAGED_TRAINING_EXPORT_V4_SCHEMA,
        verify_finite_staged_training_export_receipt_v4,
    )

    if schema == FINITE_STAGED_TRAINING_EXPORT_V4_SCHEMA:
        return verify_finite_staged_training_export_receipt_v4(
            receipt, **arguments
        )
    raise ValueError("staged-training export receipt schema is unsupported")


def _json(value):
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json(value.item())
        return _json(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint receipts require finite values")
        return value
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_receipt(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _large_tensor_receipt(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(memoryview(raw).cast("B")).hexdigest(),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_source_receipts():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in INFERENCE_SOURCE_FILES
    }


def _model_executable_contract(model):
    proposal = model.pose_model.coarse_proposal
    return {
        "deformation_integration_steps": int(
            model.deformation_decoder.integration_steps
        ),
        "deformation_support_floor": float(model.deformation_decoder.support_floor),
        "deformation_maximum_velocity_gradient": float(
            model.deformation_decoder.maximum_velocity_gradient
        ),
        "max_velocity_fraction_yx": model.deformation_decoder.max_velocity_fraction_yx.detach()
        .cpu()
        .tolist(),
        "update_limits": model.pose_model.update_limits.detach().cpu().tolist(),
        "plane_tangent_scales": model.pose_model.plane_tangent_scales.detach()
        .cpu()
        .tolist(),
        "coarse_proposal": (
            None
            if proposal is None
            else {
                "proposal_count": int(model.pose_model.proposal_count),
                "proposal_channels": int(proposal.proposal_channels),
                "mixture_components": int(proposal.mixture_components),
                "offset_scale_um": float(proposal.offset_scale_um),
                "geometry_contract": list(COARSE_PROPOSAL_GEOMETRY),
                "probabilities_calibrated": False,
                "exact_render_scope": "top-M only",
            }
        ),
    }


def _state_receipts(state):
    return {name: _tensor_receipt(value) for name, value in sorted(state.items())}


def _model_state_sha256(state_receipts):
    return _sha(
        {
            "domain": "anatomy-tracker.arbitrary-plane-model-state/v3",
            "tensor_receipts": state_receipts,
        }
    )


def _prediction_receipt(value):
    def visit(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_receipt": _tensor_receipt(item)}
        if isinstance(item, dict):
            return {str(key): visit(child) for key, child in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        return _json(item)

    tree = visit(value)
    return {"prediction_tree_receipt": tree, "receipt_sha256": _sha(tree)}


def _detach_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _detach_cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_detach_cpu_tree(child) for child in value]
    return value


def _verify_atlas_semantics_v3(semantics, channel_count):
    required = {
        "schema_version",
        "atlas_name",
        "atlas_version",
        "processed_channel_names",
        "processed_channel_recipes",
        "source_assets",
        "source_format",
        "nrrd_index_order",
        "array_axis_order",
        "positive_axis_directions",
        "voxel_center_convention",
        "normalization_parameters",
    }
    assets = semantics.get("source_assets", ()) if isinstance(semantics, dict) else ()
    valid = (
        isinstance(semantics, dict)
        and set(semantics) == required
        and semantics.get("schema_version") == ATLAS_SEMANTICS_V3_SCHEMA
        and all(
            isinstance(semantics.get(name), str) and bool(semantics[name])
            for name in ("atlas_name", "atlas_version", "source_format", "voxel_center_convention")
        )
        and semantics.get("nrrd_index_order") == "F"
        and semantics.get("array_axis_order") == ["AP", "DV", "ML"]
        and isinstance(semantics.get("positive_axis_directions"), list)
        and len(semantics["positive_axis_directions"]) == 3
        and all(
            isinstance(value, str) and bool(value)
            for value in semantics["positive_axis_directions"]
        )
        and isinstance(semantics.get("processed_channel_names"), list)
        and len(semantics["processed_channel_names"]) == int(channel_count)
        and len(set(semantics["processed_channel_names"])) == int(channel_count)
        and all(
            isinstance(value, str) and bool(value)
            for value in semantics["processed_channel_names"]
        )
        and isinstance(semantics.get("processed_channel_recipes"), list)
        and len(semantics["processed_channel_recipes"]) == int(channel_count)
        and all(
            isinstance(value, str) and bool(value)
            for value in semantics["processed_channel_recipes"]
        )
        and isinstance(assets, list)
        and bool(assets)
        and all(
            isinstance(asset, dict)
            and set(asset) == {"asset_role", "uri", "sha256"}
            and isinstance(asset["asset_role"], str)
            and bool(asset["asset_role"])
            and isinstance(asset["uri"], str)
            and bool(asset["uri"])
            and isinstance(asset["sha256"], str)
            and len(asset["sha256"]) == 64
            and not (set(asset["sha256"].lower()) - set("0123456789abcdef"))
            for asset in assets
        )
        and isinstance(semantics.get("normalization_parameters"), dict)
        and bool(semantics["normalization_parameters"])
    )
    if not valid:
        raise ValueError("atlas semantic construction contract is incomplete or invalid")
    _json(semantics["normalization_parameters"])
    return True


def make_inference_contract_v3(
    atlas_volume_c_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    *,
    atlas_semantics,
    annotation_volume_ap_dv_ml=None,
    finite_psf_capability=None,
):
    """Bind immutable atlas assets, geometry, feature recipe, and finite PSF."""
    atlas = torch.as_tensor(atlas_volume_c_ap_dv_ml)
    origin = torch.as_tensor(origin_ap_dv_ml_um, dtype=torch.float64).cpu()
    spacing = torch.as_tensor(voxel_size_ap_dv_ml_um, dtype=torch.float64).cpu()
    capability_bound = finite_psf_capability is not None
    if capability_bound:
        psf_v4.verify_finite_psf_model_capability_v4(finite_psf_capability)
        if axial_offsets_um is not None or axial_weights is not None:
            raise ValueError(
                "v4 checkpoints bind PSF capability only; exact schedules belong to sessions and caches"
            )
        offsets = weights = None
    else:
        offsets = torch.as_tensor(axial_offsets_um, dtype=torch.float64).cpu()
        weights = torch.as_tensor(axial_weights, dtype=torch.float64).cpu()
    annotation = (
        None
        if annotation_volume_ap_dv_ml is None
        else torch.as_tensor(annotation_volume_ap_dv_ml)
    )
    if (
        atlas.ndim != 4
        or any(value < 1 for value in atlas.shape)
        or not torch.is_floating_point(atlas)
        or not bool(torch.isfinite(atlas).all())
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or not bool(torch.isfinite(origin).all())
        or not bool(torch.isfinite(spacing).all())
        or bool((spacing <= 0.0).any())
    ):
        raise ValueError("atlas contract requires a finite C-AP-DV-ML asset and physical geometry")
    if not capability_bound and (
        offsets.ndim != 1
        or weights.shape != offsets.shape
        or offsets.numel() < 1
        or not bool(torch.isfinite(offsets).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0.0).any())
        or not torch.allclose(weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-12, rtol=1e-12)
        or not torch.allclose(offsets, -offsets.flip(0), atol=1e-12, rtol=1e-12)
        or not torch.allclose(weights, weights.flip(0), atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("finite PSF must be explicit, symmetric, positive, and unit mass")
    if annotation is not None and (
        annotation.ndim != 3
        or tuple(annotation.shape) != tuple(atlas.shape[1:])
        or torch.is_floating_point(annotation)
    ):
        raise ValueError("annotation asset must match the atlas spatial geometry")
    _verify_atlas_semantics_v3(atlas_semantics, atlas.shape[0])
    payload = {
        "schema_version": INFERENCE_CONTRACT_V3_SCHEMA,
        "atlas_geometry": {
            "shape_c_ap_dv_ml": list(atlas.shape),
            "origin_ap_dv_ml_um": origin.tolist(),
            "voxel_size_ap_dv_ml_um": spacing.tolist(),
        },
        "atlas_assets": {
            "atlas_volume_receipt": _tensor_receipt(atlas),
            "annotation_volume_receipt": None
            if annotation is None
            else _tensor_receipt(annotation),
        },
        "atlas_semantics": _json(atlas_semantics),
        "feature_recipe": dict(FEATURE_RECIPE_V3),
        "finite_psf": (
            dict(CHECKPOINT_FINITE_PSF_CAPABILITY_SCOPE_V4)
            if capability_bound
            else {
                "axial_offsets_um": offsets.tolist(),
                "axial_weights": weights.tolist(),
                "axial_offsets_receipt": _tensor_receipt(offsets),
                "axial_weights_receipt": _tensor_receipt(weights),
                "normalization": "positive symmetric discrete unit mass",
            }
        ),
    }
    if capability_bound:
        payload["finite_psf_capability"] = _json(finite_psf_capability)
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_inference_contract_v3(contract):
    payload = {key: value for key, value in contract.items() if key != "receipt_sha256"}
    geometry = contract.get("atlas_geometry", {})
    assets = contract.get("atlas_assets", {})
    psf = contract.get("finite_psf", {})
    shape = geometry.get("shape_c_ap_dv_ml", ())
    offsets = torch.as_tensor(psf.get("axial_offsets_um", ()), dtype=torch.float64)
    weights = torch.as_tensor(psf.get("axial_weights", ()), dtype=torch.float64)
    atlas_receipt = assets.get("atlas_volume_receipt", {})
    annotation_receipt = assets.get("annotation_volume_receipt")
    semantics_valid = True
    try:
        _verify_atlas_semantics_v3(contract.get("atlas_semantics"), shape[0])
    except (KeyError, TypeError, ValueError):
        semantics_valid = False
    atlas_receipt_valid = (
        isinstance(atlas_receipt, dict)
        and atlas_receipt.get("shape") == shape
        and isinstance(atlas_receipt.get("dtype"), str)
        and atlas_receipt["dtype"]
        in {
            "torch.float16",
            "torch.float32",
            "torch.float64",
            "torch.bfloat16",
        }
        and isinstance(atlas_receipt.get("sha256"), str)
        and len(atlas_receipt["sha256"]) == 64
        and not (set(atlas_receipt["sha256"].lower()) - set("0123456789abcdef"))
    )
    annotation_receipt_valid = annotation_receipt is None or (
        isinstance(annotation_receipt, dict)
        and annotation_receipt.get("shape") == shape[1:]
        and isinstance(annotation_receipt.get("dtype"), str)
        and not any(
            token in annotation_receipt["dtype"]
            for token in ("float", "complex", "bool")
        )
        and isinstance(annotation_receipt.get("sha256"), str)
        and len(annotation_receipt["sha256"]) == 64
        and not (
            set(annotation_receipt["sha256"].lower()) - set("0123456789abcdef")
        )
    )
    capability = contract.get("finite_psf_capability")
    if capability is None:
        schedule_valid = (
            "finite_psf_capability" not in contract
            and "finite_psf_runtime_contract" not in contract
            and offsets.ndim == 1
            and offsets.numel() > 0
            and weights.shape == offsets.shape
            and bool(torch.isfinite(offsets).all())
            and bool(torch.isfinite(weights).all())
            and bool((weights > 0.0).all())
            and torch.allclose(
                weights.sum(),
                torch.ones((), dtype=weights.dtype),
                atol=1e-12,
                rtol=1e-12,
            )
            and torch.allclose(
                offsets, -offsets.flip(0), atol=1e-12, rtol=1e-12
            )
            and torch.allclose(
                weights, weights.flip(0), atol=1e-12, rtol=1e-12
            )
            and psf.get("axial_offsets_receipt") == _tensor_receipt(offsets)
            and psf.get("axial_weights_receipt") == _tensor_receipt(weights)
            and psf.get("normalization")
            == "positive symmetric discrete unit mass"
        )
    else:
        try:
            psf_v4.verify_finite_psf_model_capability_v4(capability)
            schedule_valid = (
                psf == CHECKPOINT_FINITE_PSF_CAPABILITY_SCOPE_V4
                and "finite_psf_runtime_contract" not in contract
            )
        except (TypeError, ValueError):
            schedule_valid = False
    valid = (
        contract.get("schema_version") == INFERENCE_CONTRACT_V3_SCHEMA
        and contract.get("feature_recipe") == FEATURE_RECIPE_V3
        and isinstance(shape, list)
        and len(shape) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
        and atlas_receipt_valid
        and annotation_receipt_valid
        and semantics_valid
        and torch.as_tensor(geometry.get("origin_ap_dv_ml_um", ())).shape == (3,)
        and torch.as_tensor(geometry.get("voxel_size_ap_dv_ml_um", ())).shape == (3,)
        and schedule_valid
        and contract.get("receipt_sha256") == _sha(payload)
    )
    if not valid:
        raise ValueError("atlas/feature/PSF inference contract failed verification")
    spacing = torch.as_tensor(geometry["voxel_size_ap_dv_ml_um"], dtype=torch.float64)
    origin = torch.as_tensor(geometry["origin_ap_dv_ml_um"], dtype=torch.float64)
    if not bool(torch.isfinite(origin).all()) or not bool(torch.isfinite(spacing).all()) or bool(
        (spacing <= 0.0).any()
    ):
        raise ValueError("atlas geometry in the inference contract is invalid")
    return True


def make_runtime_inference_contract_v4(
    checkpoint_inference_contract,
    axial_offsets_um,
    axial_weights,
):
    verify_inference_contract_v3(checkpoint_inference_contract)
    capability = checkpoint_inference_contract.get("finite_psf_capability")
    if capability is None:
        raise ValueError("v4 runtime contracts require a capability-bound checkpoint")
    runtime_psf = psf_v4.runtime_schedule_contract_v4(
        axial_offsets_um,
        axial_weights,
        capability=capability,
    )
    payload = {
        "schema_version": RUNTIME_INFERENCE_CONTRACT_V4_SCHEMA,
        "checkpoint_inference_contract": _json(checkpoint_inference_contract),
        "finite_psf_capability_receipt_sha256": capability["receipt_sha256"],
        "finite_psf_runtime_contract": runtime_psf,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_runtime_inference_contract_v4(
    runtime_contract,
    checkpoint_inference_contract,
):
    expected = make_runtime_inference_contract_v4(
        checkpoint_inference_contract,
        runtime_contract.get("finite_psf_runtime_contract", {}).get(
            "axial_offsets_um", ()
        ),
        runtime_contract.get("finite_psf_runtime_contract", {}).get(
            "axial_weights", ()
        ),
    )
    if runtime_contract != expected:
        raise ValueError("runtime inference PSF or checkpoint binding changed")
    return True


def verify_catalogue_binding_v3(catalogue):
    arrays = catalogue.get("arrays", {})
    receipts = catalogue.get("array_receipts", {})
    tensors = catalogue.get("tensors", {})
    tensor_to_array = {
        "cell_id": "cell_id_int64",
        "cell_states": "cell_states_float64",
        "cell_log_mass": "cell_log_mass_float64",
        "representation_log_weight": "representation_log_weight_float64",
        "representation_to_canonical_raster_affine": "representation_to_canonical_raster_affine_float64",
    }
    valid = (
        catalogue.get("schema_version") == catalogue_v3.CATALOGUE_V3_SCHEMA
        and set(arrays) == set(receipts)
        and receipts
        and all(
            catalogue_v3._array_receipt(value) == receipts[name]
            for name, value in arrays.items()
        )
        and catalogue.get("receipt_sha256")
        == catalogue_v3._hash(catalogue_v3.catalogue_receipt_v3(catalogue))
        and set(tensors) == set(tensor_to_array)
        and all(
            torch.equal(
                torch.as_tensor(tensors[name]),
                torch.as_tensor(arrays[array_name])[
                    None if name != "cell_id" else slice(None)
                ],
            )
            for name, array_name in tensor_to_array.items()
        )
    )
    if not valid:
        raise ValueError("catalogue arrays or immutable receipt are invalid")
    cell_id = torch.as_tensor(arrays["cell_id_int64"])
    if not torch.equal(cell_id, torch.arange(cell_id.numel())):
        raise ValueError("catalogue cell IDs must be complete, unique, and canonical")
    return True


def _verified_provenance(provenance):
    required_empty = (
        "prior_trained_model_dependencies",
        "prior_model_feature_dependencies",
        "pseudolabel_dependencies",
    )
    valid = (
        isinstance(provenance, dict)
        and provenance.get("initialization") == "fresh_random"
        and provenance.get("architecture_source") == ARCHITECTURE_MODULE
        and all(provenance.get(name) == [] for name in required_empty)
        and isinstance(provenance.get("dataset_provenance"), list)
        and bool(provenance.get("dataset_provenance"))
        and isinstance(provenance.get("animal_specimen_experiment_id_contract"), str)
        and bool(provenance.get("animal_specimen_experiment_id_contract"))
    )
    if not valid:
        raise ValueError("checkpoint does not prove fresh standalone training provenance")
    return True


def checkpoint_receipt_v3(checkpoint):
    return {
        key: checkpoint[key]
        for key in (
            "schema_version",
            "architecture_name",
            "architecture_module",
            "model_config",
            "catalogue_binding",
            "provenance",
            "training_receipt",
            "inference_contract",
            "runtime_source_sha256",
            "model_executable_contract",
            "model_state_sha256",
            "checkpoint_binding_id",
            "calibration_receipt",
            "model_state_receipts",
        )
    }


def _checkpoint_binding_payload(checkpoint):
    return {
        "domain": "anatomy-tracker.arbitrary-plane-checkpoint-binding/v3",
        **{
            key: checkpoint[key]
            for key in (
                "schema_version",
                "architecture_name",
                "architecture_module",
                "model_config",
                "catalogue_binding",
                "provenance",
                "training_receipt",
                "inference_contract",
                "runtime_source_sha256",
                "model_executable_contract",
                "model_state_sha256",
                "model_state_receipts",
            )
        },
    }


def _verify_contract_catalogue_v3(contract, catalogue, atlas_channels):
    verify_inference_contract_v3(contract)
    geometry = contract["atlas_geometry"]
    support = catalogue["support_geometry"]
    valid = (
        geometry["shape_c_ap_dv_ml"][0] == int(atlas_channels)
        and geometry["shape_c_ap_dv_ml"][1:]
        == support["support_mask_receipt"]["shape"]
        and geometry["origin_ap_dv_ml_um"] == support["origin_ap_dv_ml_um"]
        and geometry["voxel_size_ap_dv_ml_um"]
        == support["voxel_size_ap_dv_ml_um"]
    )
    if not valid:
        raise ValueError("checkpoint atlas geometry disagrees with its catalogue or model")
    return True


def make_arbitrary_plane_joint_checkpoint_v3(
    model,
    model_config,
    catalogue,
    provenance,
    training_receipt,
    *,
    inference_contract,
    calibration_receipt=None,
):
    """Freeze one auditable checkpoint without importing any prior model state."""
    if type(model) is not ArbitraryPlaneJointModel:
        raise ValueError("only the standalone arbitrary-plane joint architecture is accepted")
    if set(model_config) != _MODEL_KEYS:
        raise ValueError("model config must explicitly bind every constructor field")
    verify_catalogue_binding_v3(catalogue)
    _verified_provenance(provenance)
    _verify_contract_catalogue_v3(
        inference_contract, catalogue, model_config["atlas_channels"]
    )
    reference = ArbitraryPlaneJointModel(**model_config)
    executable_contract = _model_executable_contract(reference)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if set(reference.state_dict()) != set(state) or any(
        reference.state_dict()[name].shape != value.shape for name, value in state.items()
    ):
        raise ValueError("model instance does not match its declared constructor config")
    if _model_executable_contract(model) != executable_contract:
        raise ValueError("model instance hyperparameters do not match its declared config")
    state_receipts = _state_receipts(state)
    model_state_sha256 = _model_state_sha256(state_receipts)
    _verify_training_export_receipt(
        training_receipt,
        model_kwargs=model_config,
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        model_state_sha256=model_state_sha256,
        require_source_file=True,
    )
    if training_receipt.get("binding", {}).get("finite_psf_capability") != (
        inference_contract.get("finite_psf_capability")
    ):
        raise ValueError("training and inference finite-PSF capabilities differ")
    binding_payload = {
        "schema_version": CHECKPOINT_V3_SCHEMA,
        "architecture_name": ARCHITECTURE_NAME,
        "architecture_module": ARCHITECTURE_MODULE,
        "model_config": dict(model_config),
        "catalogue_binding": {
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        },
        "provenance": dict(provenance),
        "training_receipt": dict(training_receipt),
        "inference_contract": dict(inference_contract),
        "runtime_source_sha256": _inference_source_receipts(),
        "model_executable_contract": executable_contract,
        "model_state_sha256": model_state_sha256,
        "model_state_receipts": state_receipts,
    }
    checkpoint_binding_id = _sha(
        {
            "domain": "anatomy-tracker.arbitrary-plane-checkpoint-binding/v3",
            **binding_payload,
        }
    )
    if calibration_receipt is not None:
        verify_temperature_calibration_receipt_v3(
            calibration_receipt,
            catalogue["catalogue_id"],
            checkpoint_binding_id=checkpoint_binding_id,
            model_state_sha256=model_state_sha256,
        )
        training_animals = {
            str(value) for value in training_receipt.get("training_animal_ids", ())
        }
        if training_animals != set(calibration_receipt["training_animal_ids"]):
            raise ValueError("calibration receipt training animals do not bind to training")
    payload = {
        **binding_payload,
        "checkpoint_binding_id": checkpoint_binding_id,
        "calibration_receipt": calibration_receipt,
    }
    return {**payload, "state_dict": state, "checkpoint_id": _sha(payload)}


def verify_arbitrary_plane_joint_checkpoint_v3(checkpoint, catalogue):
    verify_catalogue_binding_v3(catalogue)
    _verified_provenance(checkpoint.get("provenance"))
    config = checkpoint.get("model_config", {})
    binding = checkpoint.get("catalogue_binding", {})
    state = checkpoint.get("state_dict", {})
    state_receipts = checkpoint.get("model_state_receipts", {})
    inference_contract = checkpoint.get("inference_contract", {})
    if isinstance(config, dict) and "atlas_channels" in config:
        _verify_contract_catalogue_v3(
            inference_contract, catalogue, config["atlas_channels"]
        )
    computed_state_receipts = (
        _state_receipts(state) if isinstance(state, dict) and state else {}
    )
    computed_model_state_sha256 = (
        _model_state_sha256(computed_state_receipts)
        if computed_state_receipts
        else None
    )
    computed_binding_id = (
        _sha(_checkpoint_binding_payload(checkpoint))
        if all(
            key in checkpoint
            for key in (
                "schema_version",
                "architecture_name",
                "architecture_module",
                "model_config",
                "catalogue_binding",
                "provenance",
                "training_receipt",
                "inference_contract",
                "runtime_source_sha256",
                "model_executable_contract",
                "model_state_sha256",
                "model_state_receipts",
            )
        )
        else None
    )
    valid = (
        checkpoint.get("schema_version") == CHECKPOINT_V3_SCHEMA
        and checkpoint.get("architecture_name") == ARCHITECTURE_NAME
        and checkpoint.get("architecture_module") == ARCHITECTURE_MODULE
        and isinstance(config, dict)
        and set(config) == _MODEL_KEYS
        and binding
        == {
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
        }
        and isinstance(checkpoint.get("training_receipt"), dict)
        and bool(checkpoint["training_receipt"])
        and checkpoint.get("runtime_source_sha256") == _inference_source_receipts()
        and checkpoint.get("model_executable_contract")
        == _model_executable_contract(ArbitraryPlaneJointModel(**config))
        and isinstance(state, dict)
        and state
        and set(state) == set(state_receipts)
        and computed_state_receipts == state_receipts
        and checkpoint.get("model_state_sha256") == computed_model_state_sha256
        and checkpoint.get("checkpoint_binding_id") == computed_binding_id
        and checkpoint.get("checkpoint_id") == _sha(checkpoint_receipt_v3(checkpoint))
    )
    if not valid:
        raise ValueError("joint checkpoint or dependency binding failed verification")
    _verify_training_export_receipt(
        checkpoint["training_receipt"],
        model_kwargs=config,
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        model_state_sha256=checkpoint["model_state_sha256"],
        require_source_file=False,
    )
    if checkpoint["training_receipt"].get("binding", {}).get(
        "finite_psf_capability"
    ) != inference_contract.get("finite_psf_capability"):
        raise ValueError("checkpoint finite-PSF capability differs from training")
    if checkpoint.get("calibration_receipt") is not None:
        verify_temperature_calibration_receipt_v3(
            checkpoint["calibration_receipt"],
            catalogue["catalogue_id"],
            checkpoint_binding_id=checkpoint["checkpoint_binding_id"],
            model_state_sha256=checkpoint["model_state_sha256"],
        )
        if {
            str(value)
            for value in checkpoint["training_receipt"].get(
                "training_animal_ids", ()
            )
        } != set(checkpoint["calibration_receipt"]["training_animal_ids"]):
            raise ValueError("calibration receipt training animals do not bind to training")
    expected = ArbitraryPlaneJointModel(**config).state_dict()
    if set(expected) != set(state) or any(
        tuple(expected[name].shape) != tuple(state[name].shape) for name in expected
    ):
        raise ValueError("checkpoint tensors do not match the declared architecture")
    return True


def load_arbitrary_plane_inference_v3(checkpoint_path, catalogue, *, device="cpu"):
    """Safely fresh-load only the declared standalone joint architecture."""
    path = Path(checkpoint_path).resolve(strict=True)
    if not path.is_file() or path.drive.upper() != "I:":
        raise ValueError("v3 checkpoints must be regular files resolved on the I: drive")
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )
    verify_arbitrary_plane_joint_checkpoint_v3(checkpoint, catalogue)
    model = ArbitraryPlaneJointModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    if _model_executable_contract(model) != checkpoint["model_executable_contract"]:
        raise ValueError("loaded model executable attributes differ from checkpoint")
    return {
        "model": model,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_receipt": checkpoint_receipt_v3(checkpoint),
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": _file_sha256(path),
        "checkpoint_binding_id": checkpoint["checkpoint_binding_id"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "model_config": checkpoint["model_config"],
        "runtime_source_sha256": checkpoint["runtime_source_sha256"],
        "model_executable_contract": checkpoint["model_executable_contract"],
        "catalogue_id": catalogue["catalogue_id"],
        "calibration_receipt": checkpoint.get("calibration_receipt"),
        "inference_contract": checkpoint["inference_contract"],
        "device": str(torch.device(device)),
    }


def _verify_loaded_for_catalogue_cache_v3(loaded, catalogue):
    verify_catalogue_binding_v3(catalogue)
    model = loaded.get("model")
    if type(model) is not ArbitraryPlaneJointModel or model.training:
        raise ValueError("catalogue feature caches require a verified evaluation model")
    path = Path(loaded.get("checkpoint_path", "")).resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    verify_arbitrary_plane_joint_checkpoint_v3(checkpoint, catalogue)
    valid = (
        path.drive.upper() == "I:"
        and path.is_file()
        and _file_sha256(path) == loaded.get("checkpoint_file_sha256")
        and checkpoint["checkpoint_id"] == loaded.get("checkpoint_id")
        and checkpoint["checkpoint_binding_id"] == loaded.get("checkpoint_binding_id")
        and checkpoint["model_state_sha256"] == loaded.get("model_state_sha256")
        and loaded.get("checkpoint_receipt") == checkpoint_receipt_v3(checkpoint)
        and loaded.get("runtime_source_sha256") == _inference_source_receipts()
        and loaded.get("model_executable_contract") == _model_executable_contract(model)
        and _model_state_sha256(_state_receipts(model.state_dict()))
        == loaded.get("model_state_sha256")
        and loaded.get("catalogue_id") == catalogue.get("catalogue_id")
    )
    if not valid:
        raise ValueError("loaded checkpoint is not valid for catalogue feature caching")
    _verify_contract_catalogue_v3(
        loaded["inference_contract"], catalogue, model.pose_model.atlas_channels
    )
    return model, path


def _catalogue_feature_cache_payload_v3(
    loaded,
    catalogue,
    retrieval_shape_h_w,
    build_chunk_size,
    atlas_features_receipt,
    finite_psf_runtime_contract=None,
):
    shape = atlas_features_receipt["shape"]
    render_recipe = {
        "retrieval_shape_h_w": list(retrieval_shape_h_w),
        "raster_support_geometry": _json(catalogue["support_geometry"]),
        "representation_affine_receipt": catalogue["array_receipts"][
            "representation_to_canonical_raster_affine_float64"
        ],
        "rendering": "finite-thickness atlas render followed by the frozen atlas stem and shared encoder",
        "source_resize": "bilinear align_corners=False before histology encoding; atlas renders are produced directly at retrieval size",
        "tensor_layout": "canonical_cell,representation,feature_channel,encoded_h,encoded_w; contiguous row-major",
        "tensor_dtype": atlas_features_receipt["dtype"],
        "tensor_shape": shape,
        "build_chunk_size": int(build_chunk_size),
    }
    if "finite_psf_capability" in loaded["inference_contract"]:
        psf_v4.verify_runtime_schedule_contract_v4(
            finite_psf_runtime_contract,
            loaded["inference_contract"]["finite_psf_capability"],
        )
        render_recipe["finite_psf_capability"] = loaded["inference_contract"][
            "finite_psf_capability"
        ]
        render_recipe["finite_psf_runtime_contract"] = finite_psf_runtime_contract
    else:
        if finite_psf_runtime_contract is not None:
            raise ValueError("v3 feature caches cannot add a v4 runtime PSF")
        render_recipe["finite_thickness_psf"] = loaded["inference_contract"][
            "finite_psf"
        ]
    return {
        "schema_version": CATALOGUE_FEATURE_CACHE_V3_SCHEMA,
        "feature_origin": {
            "description": "atlas features emitted by this exact frozen checkpoint pose encoder",
            "external_or_prior_model_dependencies": [],
            "approximate_candidate_pruning": False,
        },
        "checkpoint_binding": {
            "checkpoint_id": loaded["checkpoint_id"],
            "checkpoint_binding_id": loaded["checkpoint_binding_id"],
            "checkpoint_path": loaded["checkpoint_path"],
            "checkpoint_file_sha256": loaded["checkpoint_file_sha256"],
            "model_state_sha256": loaded["model_state_sha256"],
            "runtime_source_sha256": loaded["runtime_source_sha256"],
            "model_executable_contract": loaded["model_executable_contract"],
        },
        "catalogue_binding": {
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "cell_count": int(catalogue["counts"]["cell_count"]),
            "representation_count": int(catalogue["counts"]["representation_count"]),
            "cell_id_receipt": catalogue["array_receipts"]["cell_id_int64"],
        },
        "atlas_inference_contract": loaded["inference_contract"],
        "render_and_storage_recipe": render_recipe,
        "complete_coverage": {
            "cell_id_min": 0,
            "cell_id_max": int(catalogue["counts"]["cell_count"]) - 1,
            "cell_count": int(catalogue["counts"]["cell_count"]),
            "all_cells_exactly_once": True,
        },
        "numerical_equivalence_contract": {
            "absolute_tolerance": CACHE_NUMERICAL_ATOL,
            "relative_tolerance": CACHE_NUMERICAL_RTOL,
            "scope": "complete normalized retrieval log probabilities/probabilities, stable top-K IDs/log probabilities, retained mass, and omitted tail mass",
            "same_dtype_no_compression": True,
        },
        "atlas_features_receipt": atlas_features_receipt,
    }


def _verify_catalogue_feature_cache_contents_v3(
    cache,
    loaded,
    catalogue,
    model,
    *,
    verify_file_binding,
):
    receipt = cache.get("cache_receipt", {})
    features = cache.get("atlas_features")
    cell_id = cache.get("cell_id")
    if not isinstance(features, torch.Tensor) or not isinstance(cell_id, torch.Tensor):
        raise ValueError("catalogue feature cache tensors are missing")
    feature_receipt = _large_tensor_receipt(features)
    recipe = receipt.get("render_and_storage_recipe", {})
    expected = _catalogue_feature_cache_payload_v3(
        loaded,
        catalogue,
        recipe.get("retrieval_shape_h_w", ()),
        recipe.get("build_chunk_size", 0),
        feature_receipt,
        recipe.get("finite_psf_runtime_contract"),
    )
    expected_cells = int(catalogue["counts"]["cell_count"])
    expected_representations = int(catalogue["counts"]["representation_count"])
    retrieval_shape = recipe.get("retrieval_shape_h_w", ())
    recipe_valid = (
        isinstance(retrieval_shape, list)
        and len(retrieval_shape) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 4
            for value in retrieval_shape
        )
        and isinstance(recipe.get("build_chunk_size"), int)
        and not isinstance(recipe.get("build_chunk_size"), bool)
        and recipe["build_chunk_size"] >= 1
    )
    encoded_shape = (
        (int(retrieval_shape[0]) + 3) // 4,
        (int(retrieval_shape[1]) + 3) // 4,
    ) if recipe_valid else ()
    valid = (
        receipt.get("schema_version") == CATALOGUE_FEATURE_CACHE_V3_SCHEMA
        and recipe_valid
        and receipt.get("cache_id") == _sha(expected)
        and {key: value for key, value in receipt.items() if key != "cache_id"} == expected
        and features.device.type == "cpu"
        and features.is_contiguous()
        and torch.is_floating_point(features)
        and bool(torch.isfinite(features).all())
        and features.dtype == next(model.parameters()).dtype
        and tuple(features.shape[:3])
        == (
            expected_cells,
            expected_representations,
            int(loaded["model_config"]["feature_channels"]),
        )
        and tuple(features.shape[-2:]) == encoded_shape
        and cell_id.device.type == "cpu"
        and cell_id.dtype == torch.long
        and torch.equal(cell_id, torch.arange(expected_cells))
        and cache.get("cache_path") is not None
        and cache.get("cache_file_sha256") is not None
    )
    if not valid:
        raise ValueError("catalogue feature cache binding or complete coverage is invalid")
    path = Path(cache["cache_path"]).resolve(strict=True)
    if path.drive.upper() != "I:" or not path.is_file() or (
        verify_file_binding
        and _file_sha256(path) != cache["cache_file_sha256"]
    ):
        raise ValueError("catalogue feature cache file binding is invalid")
    return True


def verify_arbitrary_plane_catalogue_feature_cache_v3(cache, loaded, catalogue):
    """Verify one complete, exact, same-checkpoint inference cache."""
    model, _ = _verify_loaded_for_catalogue_cache_v3(loaded, catalogue)
    return _verify_catalogue_feature_cache_contents_v3(
        cache,
        loaded,
        catalogue,
        model,
        verify_file_binding=True,
    )


def make_arbitrary_plane_catalogue_feature_cache_v3(
    loaded,
    atlas_volume_c_ap_dv_ml,
    catalogue,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    cache_output_path,
    *,
    retrieval_shape_h_w=(48, 64),
    build_chunk_size=128,
    annotation_volume_ap_dv_ml=None,
):
    """Freeze all current-checkpoint atlas features after model training ends."""
    model, _ = _verify_loaded_for_catalogue_cache_v3(loaded, catalogue)
    if model.pose_model.coarse_proposal is not None:
        raise ValueError(
            "amortized proposal checkpoints do not build complete-catalogue feature caches"
        )
    if (
        len(retrieval_shape_h_w) != 2
        or min(retrieval_shape_h_w) < 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in retrieval_shape_h_w
        )
        or not isinstance(build_chunk_size, int)
        or isinstance(build_chunk_size, bool)
        or build_chunk_size < 1
    ):
        raise ValueError("cache retrieval shape and build chunk size are invalid")
    capability = loaded["inference_contract"].get("finite_psf_capability")
    checkpoint_contract = make_inference_contract_v3(
        atlas_volume_c_ap_dv_ml,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        None if capability is not None else axial_offsets_um,
        None if capability is not None else axial_weights,
        atlas_semantics=loaded["inference_contract"]["atlas_semantics"],
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        finite_psf_capability=capability,
    )
    if checkpoint_contract != loaded["inference_contract"]:
        raise ValueError("cache atlas assets, geometry, semantics, or PSF do not match checkpoint")
    finite_psf_runtime_contract = None
    if capability is not None:
        finite_psf_runtime_contract = make_runtime_inference_contract_v4(
            checkpoint_contract,
            axial_offsets_um,
            axial_weights,
        )["finite_psf_runtime_contract"]
    output = Path(cache_output_path).resolve()
    if output.drive.upper() != "I:" or output.exists():
        raise ValueError("catalogue feature caches require a new file path on the I: drive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise ValueError("catalogue feature cache temporary path already exists")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    atlas = torch.as_tensor(atlas_volume_c_ap_dv_ml, device=device, dtype=dtype)
    tensors = catalogue["tensors"]
    states = torch.as_tensor(tensors["cell_states"], device=device, dtype=dtype)
    affine = torch.as_tensor(
        tensors["representation_to_canonical_raster_affine"],
        device=device,
        dtype=dtype,
    )
    count = int(catalogue["counts"]["cell_count"])
    features = None
    with torch.inference_mode():
        for start in range(0, count, build_chunk_size):
            stop = min(start + build_chunk_size, count)
            rendered = model.pose_model._render_representations(
                atlas,
                states[:, start:stop],
                affine[:, start:stop],
                tuple(retrieval_shape_h_w),
                origin_ap_dv_ml_um,
                voxel_size_ap_dv_ml_um,
                axial_offsets_um,
                axial_weights,
            )
            flat_encoded = model.pose_model._encode_atlas(
                rendered.reshape(
                    -1,
                    model.pose_model.atlas_channels,
                    *tuple(retrieval_shape_h_w),
                )
            )
            encoded = flat_encoded.reshape(
                stop - start,
                rendered.shape[2],
                flat_encoded.shape[1],
                *flat_encoded.shape[-2:],
            )
            if features is None:
                features = torch.empty(
                    (count, *encoded.shape[1:]), dtype=encoded.dtype, device="cpu"
                )
            features[start:stop].copy_(encoded.detach().cpu())
    features = features.contiguous()
    feature_receipt = _large_tensor_receipt(features)
    payload = _catalogue_feature_cache_payload_v3(
        loaded,
        catalogue,
        tuple(int(value) for value in retrieval_shape_h_w),
        int(build_chunk_size),
        feature_receipt,
        finite_psf_runtime_contract,
    )
    artifact = {
        "cache_receipt": {**payload, "cache_id": _sha(payload)},
        "cell_id": torch.arange(count, dtype=torch.long),
        "atlas_features": features,
    }
    torch.save(artifact, temporary)
    os.replace(temporary, output)
    return load_arbitrary_plane_catalogue_feature_cache_v3(output, loaded, catalogue)


def _load_catalogue_feature_cache_file_v3(path):
    cache_path = Path(path).resolve(strict=True)
    if cache_path.drive.upper() != "I:" or not cache_path.is_file():
        raise ValueError("catalogue feature caches must be regular files on the I: drive")
    artifact = torch.load(cache_path, map_location="cpu", weights_only=True)
    return {
        **artifact,
        "cache_path": str(cache_path),
        "cache_file_sha256": _file_sha256(cache_path),
    }


def load_arbitrary_plane_catalogue_feature_cache_v3(path, loaded, catalogue):
    cache = _load_catalogue_feature_cache_file_v3(path)
    model, _ = _verify_loaded_for_catalogue_cache_v3(loaded, catalogue)
    _verify_catalogue_feature_cache_contents_v3(
        cache,
        loaded,
        catalogue,
        model,
        verify_file_binding=False,
    )
    return cache


def _tensor_handle_state(value):
    tensor = torch.as_tensor(value)
    return (
        id(tensor),
        int(tensor._version),
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _model_handle_state(model):
    return tuple(
        (name, *_tensor_handle_state(value))
        for name, value in sorted(
            (*model.named_parameters(), *model.named_buffers()), key=lambda item: item[0]
        )
    )


def _verify_loaded_checkpoint_runtime_v3(loaded, catalogue):
    verify_catalogue_binding_v3(catalogue)
    model = loaded.get("model")
    if type(model) is not ArbitraryPlaneJointModel or model.training:
        raise ValueError("inference requires the verified joint model in evaluation mode")
    checkpoint_receipt = loaded.get("checkpoint_receipt")
    checkpoint_path = Path(loaded.get("checkpoint_path", "")).resolve(strict=True)
    if (
        not isinstance(checkpoint_receipt, dict)
        or checkpoint_path.drive.upper() != "I:"
        or not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != loaded.get("checkpoint_file_sha256")
        or loaded.get("checkpoint_id") != _sha(checkpoint_receipt)
        or loaded.get("checkpoint_binding_id")
        != checkpoint_receipt.get("checkpoint_binding_id")
        or loaded.get("model_state_sha256")
        != checkpoint_receipt.get("model_state_sha256")
        or loaded.get("calibration_receipt")
        != checkpoint_receipt.get("calibration_receipt")
        or loaded.get("catalogue_id") != catalogue.get("catalogue_id")
    ):
        raise ValueError("loaded checkpoint provenance receipt is invalid")
    if _model_state_sha256(_state_receipts(model.state_dict())) != loaded[
        "model_state_sha256"
    ]:
        raise ValueError("loaded model state changed after checkpoint verification")
    if (
        loaded.get("runtime_source_sha256") != _inference_source_receipts()
        or loaded.get("runtime_source_sha256")
        != checkpoint_receipt.get("runtime_source_sha256")
        or loaded.get("model_executable_contract")
        != _model_executable_contract(model)
        or loaded.get("model_executable_contract")
        != checkpoint_receipt.get("model_executable_contract")
        or _model_executable_contract(ArbitraryPlaneJointModel(**loaded["model_config"]))
        != loaded.get("model_executable_contract")
    ):
        raise ValueError("runtime source or executable model contract changed")
    if loaded.get("calibration_receipt") is not None:
        verify_temperature_calibration_receipt_v3(
            loaded["calibration_receipt"],
            catalogue["catalogue_id"],
            checkpoint_binding_id=loaded["checkpoint_binding_id"],
            model_state_sha256=loaded["model_state_sha256"],
        )
    inference_contract = loaded.get("inference_contract")
    if inference_contract != checkpoint_receipt.get("inference_contract"):
        raise ValueError("loaded inference contract is not checkpoint-bound")
    _verify_contract_catalogue_v3(
        inference_contract, catalogue, model.pose_model.atlas_channels
    )
    return model


def _make_arbitrary_plane_inference_session_v3(
    loaded,
    atlas_volume_c_ap_dv_ml,
    catalogue,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    *,
    annotation_volume_ap_dv_ml=None,
    catalogue_feature_cache=None,
    catalogue_feature_cache_path=None,
    loaded_already_verified=False,
):
    if catalogue_feature_cache is not None and catalogue_feature_cache_path is not None:
        raise ValueError("provide one catalogue feature cache object or path, not both")
    model = loaded.get("model")
    if loaded_already_verified:
        if (
            type(model) is not ArbitraryPlaneJointModel
            or model.training
            or loaded.get("catalogue_id") != catalogue.get("catalogue_id")
        ):
            raise ValueError("freshly loaded checkpoint handle is invalid")
    else:
        model = _verify_loaded_checkpoint_runtime_v3(loaded, catalogue)
    inference_contract = loaded["inference_contract"]
    capability = inference_contract.get("finite_psf_capability")
    checkpoint_contract = make_inference_contract_v3(
        atlas_volume_c_ap_dv_ml,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        None if capability is not None else axial_offsets_um,
        None if capability is not None else axial_weights,
        atlas_semantics=inference_contract["atlas_semantics"],
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        finite_psf_capability=capability,
    )
    if checkpoint_contract != inference_contract:
        raise ValueError(
            "runtime atlas assets, geometry, feature recipe, or PSF do not match checkpoint"
        )
    runtime_contract = (
        make_runtime_inference_contract_v4(
            checkpoint_contract,
            axial_offsets_um,
            axial_weights,
        )
        if capability is not None
        else checkpoint_contract
    )
    parameter = next(model.parameters())
    device, model_dtype = parameter.device, parameter.dtype
    atlas = torch.as_tensor(
        atlas_volume_c_ap_dv_ml, device=device, dtype=model_dtype
    ).contiguous().clone()
    if atlas.ndim != 4 or atlas.shape[0] != model.pose_model.atlas_channels:
        raise ValueError("atlas channels do not match the checkpoint architecture")
    annotation = (
        None
        if annotation_volume_ap_dv_ml is None
        else torch.as_tensor(annotation_volume_ap_dv_ml, device=device).contiguous().clone()
    )
    tensors = {}
    for name, value in catalogue["tensors"].items():
        tensor = torch.as_tensor(value, device=device)
        if name != "cell_id":
            tensor = tensor.to(dtype=model_dtype)
        tensors[name] = tensor.contiguous().clone()
    origin = torch.as_tensor(
        origin_ap_dv_ml_um, device=device, dtype=model_dtype
    ).clone()
    voxel_size = torch.as_tensor(
        voxel_size_ap_dv_ml_um, device=device, dtype=model_dtype
    ).clone()
    axial_offsets = torch.as_tensor(
        axial_offsets_um, device=device, dtype=model_dtype
    ).clone()
    weights = torch.as_tensor(
        axial_weights, device=device, dtype=model_dtype
    ).clone()
    feature_cache = catalogue_feature_cache
    feature_cache_fresh = False
    if catalogue_feature_cache_path is not None:
        feature_cache = _load_catalogue_feature_cache_file_v3(
            catalogue_feature_cache_path
        )
        feature_cache_fresh = True
    if feature_cache is not None:
        if loaded["model"].pose_model.coarse_proposal is not None:
            raise ValueError(
                "amortized proposal checkpoints do not consume complete-catalogue feature caches"
            )
        _verify_catalogue_feature_cache_contents_v3(
            feature_cache,
            loaded,
            catalogue,
            model,
            verify_file_binding=not feature_cache_fresh,
        )
        if capability is not None and feature_cache["cache_receipt"][
            "render_and_storage_recipe"
        ].get("finite_psf_runtime_contract") != runtime_contract[
            "finite_psf_runtime_contract"
        ]:
            raise ValueError("catalogue feature cache runtime PSF differs from session")
    cache_binding = None
    sealed_feature_cache = None
    if feature_cache is not None:
        cache_receipt = feature_cache["cache_receipt"]
        sealed_feature_cache = MappingProxyType(
            {
                "atlas_features": feature_cache["atlas_features"],
                "cell_id": feature_cache["cell_id"],
                "cache_path": feature_cache["cache_path"],
                "cache_file_sha256": feature_cache["cache_file_sha256"],
                "cache_receipt": _freeze(cache_receipt),
            }
        )
        cache_binding = {
            "cache_id": cache_receipt["cache_id"],
            "cache_path": feature_cache["cache_path"],
            "cache_file_sha256": feature_cache["cache_file_sha256"],
            "atlas_features_receipt": cache_receipt["atlas_features_receipt"],
            "numerical_equivalence_contract": cache_receipt[
                "numerical_equivalence_contract"
            ],
        }
    receipt_payload = {
        "schema_version": INFERENCE_SESSION_V3_SCHEMA,
        "checkpoint_id": loaded["checkpoint_id"],
        "checkpoint_binding_id": loaded["checkpoint_binding_id"],
        "checkpoint_file_sha256": loaded["checkpoint_file_sha256"],
        "model_state_sha256": loaded["model_state_sha256"],
        "catalogue_id": catalogue["catalogue_id"],
        "catalogue_receipt_sha256": catalogue["receipt_sha256"],
        "atlas_receipt": runtime_contract,
        "catalogue_feature_cache": cache_binding,
        "device": str(device),
        "model_dtype": str(model_dtype),
    }
    receipt = _freeze(
        {**receipt_payload, "receipt_sha256": _sha(receipt_payload)}
    )
    tracked = [atlas, *tensors.values(), origin, voxel_size, axial_offsets, weights]
    if annotation is not None:
        tracked.append(annotation)
    if sealed_feature_cache is not None:
        tracked.extend(
            (
                sealed_feature_cache["atlas_features"],
                sealed_feature_cache["cell_id"],
            )
        )
    return MappingProxyType(
        {
            "schema_version": INFERENCE_SESSION_V3_SCHEMA,
            "_token": _INFERENCE_SESSION_TOKEN,
            "receipt": receipt,
            "model": model,
            "model_handle_state": _model_handle_state(model),
            "tracked_tensors": tuple(tracked),
            "tracked_tensor_states": tuple(_tensor_handle_state(value) for value in tracked),
            "catalogue_tensors": MappingProxyType(tensors),
            "posterior_catalogue": MappingProxyType(
                {
                    "catalogue_id": catalogue["catalogue_id"],
                    "tensors": MappingProxyType(
                        {
                            "cell_id": tensors["cell_id"],
                            "cell_states": tensors["cell_states"],
                        }
                    ),
                }
            ),
            "catalogue_counts": _freeze(catalogue["counts"]),
            "support_origin_ap_dv_ml_um": tuple(
                float(value)
                for value in catalogue["support_geometry"][
                    "support_origin_ap_dv_ml_um"
                ]
            ),
            "support_raster_shape_h_w": tuple(
                int(value)
                for value in catalogue["support_geometry"]["raster_shape_h_w"]
            ),
            "atlas": atlas,
            "annotation": annotation,
            "origin_ap_dv_ml_um": origin,
            "voxel_size_ap_dv_ml_um": voxel_size,
            "axial_offsets_um": axial_offsets,
            "axial_weights": weights,
            "checkpoint_receipt": _freeze(loaded["checkpoint_receipt"]),
            "checkpoint_id": loaded["checkpoint_id"],
            "checkpoint_binding_id": loaded["checkpoint_binding_id"],
            "checkpoint_file_sha256": loaded["checkpoint_file_sha256"],
            "model_state_sha256": loaded["model_state_sha256"],
            "calibration_receipt": _freeze(loaded["calibration_receipt"])
            if loaded.get("calibration_receipt") is not None
            else None,
            "inference_contract": _freeze(inference_contract),
            "runtime_inference_contract": _freeze(runtime_contract),
            "model_executable_contract": _freeze(
                loaded["model_executable_contract"]
            ),
            "feature_cache": sealed_feature_cache,
            "feature_cache_receipt": sealed_feature_cache["cache_receipt"]
            if sealed_feature_cache is not None
            else None,
            "cache_binding": _freeze(cache_binding) if cache_binding is not None else None,
            "device": device,
            "model_dtype": model_dtype,
        }
    )


def prepare_arbitrary_plane_inference_session_v3(
    loaded,
    atlas_volume_c_ap_dv_ml,
    catalogue,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    *,
    annotation_volume_ap_dv_ml=None,
    catalogue_feature_cache=None,
):
    """Fully authenticate runtime dependencies once and seal an in-process handle."""
    return _make_arbitrary_plane_inference_session_v3(
        loaded,
        atlas_volume_c_ap_dv_ml,
        catalogue,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        axial_offsets_um,
        axial_weights,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        catalogue_feature_cache=catalogue_feature_cache,
    )


def open_arbitrary_plane_inference_session_v3(
    checkpoint_path,
    atlas_volume_c_ap_dv_ml,
    catalogue,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    *,
    annotation_volume_ap_dv_ml=None,
    catalogue_feature_cache_path=None,
    device="cpu",
):
    """Load and authenticate a checkpoint/cache exactly once for repeated inference."""
    loaded = load_arbitrary_plane_inference_v3(
        checkpoint_path, catalogue, device=device
    )
    return _make_arbitrary_plane_inference_session_v3(
        loaded,
        atlas_volume_c_ap_dv_ml,
        catalogue,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        axial_offsets_um,
        axial_weights,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        catalogue_feature_cache_path=catalogue_feature_cache_path,
        loaded_already_verified=True,
    )


def _verify_arbitrary_plane_inference_session_v3(session):
    if (
        not isinstance(session, MappingProxyType)
        or session.get("_token") is not _INFERENCE_SESSION_TOKEN
        or session.get("schema_version") != INFERENCE_SESSION_V3_SCHEMA
    ):
        raise ValueError("inference requires a sealed verified session handle")
    receipt = session.get("receipt", {})
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    model = session.get("model")
    valid = (
        receipt.get("schema_version") == INFERENCE_SESSION_V3_SCHEMA
        and receipt.get("receipt_sha256") == _sha(payload)
        and type(model) is ArbitraryPlaneJointModel
        and not model.training
        and _model_handle_state(model) == session.get("model_handle_state")
        and _model_executable_contract(model)
        == _json(session.get("model_executable_contract"))
        and tuple(
            _tensor_handle_state(value) for value in session.get("tracked_tensors", ())
        )
        == session.get("tracked_tensor_states")
    )
    if not valid:
        raise ValueError("sealed inference session changed after verification")
    return True


def _validate_raw_inference_input_v3(input_b3hw):
    raw_image = torch.as_tensor(input_b3hw)
    if not torch.is_floating_point(raw_image):
        raise ValueError("v3 input must be floating point with shape (B,3,H,W)")
    image = raw_image
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("v3 input must be floating point with shape (B,3,H,W)")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("v3 input must be finite")
    if bool(((image[:, :2] < 0.0) | (image[:, :2] > 1.0)).any()):
        raise ValueError("brightfield and outline channels must be in the trained [0,1] domain")
    availability_plane = image[:, 2]
    availability = availability_plane[:, 0, 0]
    if not bool((availability_plane == availability[:, None, None]).all()) or not bool(
        ((availability == 0.0) | (availability == 1.0)).all()
    ):
        raise ValueError("outline availability channel must be a constant binary plane")
    return raw_image


def run_arbitrary_plane_inference_session_v3(
    session,
    input_b3hw,
    *,
    animal_ids,
    specimen_ids,
    experiment_ids,
    synthetic_animal_ids=None,
    section_ids=None,
    synthetic_realization_ids=None,
    top_k=4,
    refinement_steps=3,
    pose_only_steps=2,
    retrieval_shape_h_w=(48, 64),
    catalogue_chunk_size=128,
    electrode_points_yx_px=None,
    gauss_hermite_order=5,
    raw_prediction_output_path=None,
    return_raw_prediction=False,
):
    """Run one batch against a previously authenticated immutable session."""
    _verify_arbitrary_plane_inference_session_v3(session)
    model = session["model"]
    device, model_dtype = session["device"], session["model_dtype"]
    inference_contract = session["inference_contract"]
    runtime_contract = _json(session["runtime_inference_contract"])
    atlas = session["atlas"]
    tensors = session["catalogue_tensors"]
    raw_image = _validate_raw_inference_input_v3(input_b3hw)
    raw_input_receipt = _tensor_receipt(raw_image)
    image = raw_image.to(device=device, dtype=model_dtype)
    if image.ndim == 3:
        image = image[None]
    if tuple(image.shape[-2:]) != session["support_raster_shape_h_w"]:
        raise ValueError("input raster geometry does not match the checkpoint catalogue recipe")
    availability_plane = image[:, 2]
    availability = availability_plane[:, 0, 0]
    batch = image.shape[0]

    def identifiers(values, name, *, nullable=False):
        if values is None:
            if nullable:
                return [None] * batch
            raise ValueError(f"{name} must identify every inference sample")
        if isinstance(values, (str, int)):
            values = [values]
        values = list(values)
        if len(values) != batch or any(
            (value is None and not nullable)
            or (
                value is not None
                and (
                    not isinstance(value, (str, int))
                    or isinstance(value, bool)
                    or (isinstance(value, str) and not value)
                )
            )
            for value in values
        ):
            raise ValueError(f"{name} must identify every inference sample")
        return values

    animal_ids = identifiers(animal_ids, "animal_ids")
    specimen_ids = identifiers(specimen_ids, "specimen_ids")
    experiment_ids = identifiers(experiment_ids, "experiment_ids")
    synthetic_animal_ids = identifiers(
        synthetic_animal_ids, "synthetic_animal_ids", nullable=True
    )
    section_ids = identifiers(section_ids, "section_ids", nullable=True)
    synthetic_realization_ids = identifiers(
        synthetic_realization_ids, "synthetic_realization_ids", nullable=True
    )

    def expanded(name, dtype=model_dtype):
        value = tensors[name]
        if name == "cell_id":
            return value
        value = value.to(dtype=dtype)
        return value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value

    support_origin = session["support_origin_ap_dv_ml_um"]
    feature_cache = session["feature_cache"]
    cache_binding = session["cache_binding"]
    if feature_cache is not None:
        cache_receipt = session["feature_cache_receipt"]
        if tuple(
            cache_receipt["render_and_storage_recipe"]["retrieval_shape_h_w"]
        ) != tuple(retrieval_shape_h_w):
            raise ValueError("catalogue feature cache retrieval shape does not match inference")
        if cache_receipt["render_and_storage_recipe"]["build_chunk_size"] != int(
            catalogue_chunk_size
        ):
            raise ValueError(
                "catalogue feature cache build chunking must match exact inference scoring chunking"
            )
    config_payload = {
        "top_k": int(top_k),
        "refinement_steps": int(refinement_steps),
        "pose_only_steps": int(pose_only_steps),
        "retrieval_shape_h_w": list(retrieval_shape_h_w),
        "catalogue_chunk_size": int(catalogue_chunk_size),
        "gauss_hermite_order": int(gauss_hermite_order),
        "electrode_points_receipt": None
        if electrode_points_yx_px is None
        else _tensor_receipt(electrode_points_yx_px),
        "model_dtype": str(model_dtype),
        "device": str(device),
        "catalogue_feature_cache": None
        if cache_binding is None
        else _json(cache_binding),
    }
    configuration_receipt = {
        **config_payload,
        "receipt_sha256": _sha(config_payload),
    }
    cache_context = (
        nullcontext()
        if feature_cache is None
        else model.pose_model.use_complete_catalogue_feature_cache(
            feature_cache["atlas_features"],
            feature_cache["cell_id"],
            tuple(retrieval_shape_h_w),
            _verification_token=_VERIFIED_CATALOGUE_FEATURE_CACHE_TOKEN,
        )
    )
    with torch.inference_mode(), cache_context:
        joint = model(
            image[:, :1],
            image[:, 1:2],
            availability,
            atlas,
            expanded("cell_id", None),
            expanded("cell_states"),
            expanded("cell_log_mass"),
            expanded("representation_log_weight"),
            expanded("representation_to_canonical_raster_affine"),
            tuple(image.shape[-2:]),
            session["origin_ap_dv_ml_um"],
            session["voxel_size_ap_dv_ml_um"],
            support_origin,
            session["axial_offsets_um"],
            session["axial_weights"],
            expected_catalogue_cell_count=int(
                session["catalogue_counts"]["cell_count"]
            ),
            top_k=int(top_k),
            refinement_steps=int(refinement_steps),
            pose_only_steps=int(pose_only_steps),
            retrieval_shape_h_w=tuple(retrieval_shape_h_w),
            catalogue_chunk_size=int(catalogue_chunk_size),
        )
        posterior = posterior_summary_v3(
            joint,
            session["posterior_catalogue"],
            support_origin,
            calibration_receipt=session["calibration_receipt"],
            checkpoint_binding_id=session["checkpoint_binding_id"],
            model_state_sha256=session["model_state_sha256"],
            gauss_hermite_order=int(gauss_hermite_order),
        )
        trajectory = None
        if electrode_points_yx_px is not None:
            trajectory = propagate_electrode_trajectory_v3(
                posterior,
                joint,
                electrode_points_yx_px,
                session["origin_ap_dv_ml_um"],
                session["voxel_size_ap_dv_ml_um"],
                annotation_volume_ap_dv_ml=session["annotation"],
                atlas_shape_ap_dv_ml=tuple(atlas.shape[1:]),
            )
    lineage = [
        {
            "animal_id": animal_ids[index],
            "specimen_id": specimen_ids[index],
            "experiment_id": experiment_ids[index],
            "synthetic_animal_id": synthetic_animal_ids[index],
            "section_id": section_ids[index],
            "synthetic_realization_id": synthetic_realization_ids[index],
        }
        for index in range(batch)
    ]
    identifier_payload = {
        "animal_ids": animal_ids,
        "specimen_ids": specimen_ids,
        "experiment_ids": experiment_ids,
        "synthetic_animal_ids": synthetic_animal_ids,
        "section_ids": section_ids,
        "synthetic_realization_ids": synthetic_realization_ids,
        "lineage": lineage,
    }
    input_payload = {
        "raw_input_receipt": raw_input_receipt,
        "model_input_receipt": _tensor_receipt(image),
        "feature_recipe": _json(inference_contract["feature_recipe"]),
        "identifiers": identifier_payload,
    }
    input_receipt = {**input_payload, "receipt_sha256": _sha(input_payload)}
    raw_prediction = {"lineage": lineage, **_detach_cpu_tree(joint)}
    raw_prediction_receipt = _prediction_receipt(raw_prediction)
    if raw_prediction_output_path is None:
        raise ValueError("successful inference requires an I:-drive raw prediction output path")
    raw_path = Path(raw_prediction_output_path).resolve()
    if raw_path.drive.upper() != "I:" or raw_path.exists():
        raise ValueError("raw predictions require a new regular-file path on the I: drive")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_artifact = {
        "schema_version": "anatomy-tracker.raw-joint-prediction/v3",
        "checkpoint_id": session["checkpoint_id"],
        "checkpoint_binding_id": session["checkpoint_binding_id"],
        "catalogue_id": session["receipt"]["catalogue_id"],
        "identifiers": identifier_payload,
        "input_receipt": input_receipt,
        "configuration_receipt": configuration_receipt,
        "raw_prediction_receipt": raw_prediction_receipt,
        "raw_prediction": raw_prediction,
    }
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError("raw prediction temporary path already exists")
    torch.save(raw_artifact, temporary)
    os.replace(temporary, raw_path)
    raw_prediction_file_sha256 = _file_sha256(raw_path)
    result = {
        "schema_version": INFERENCE_V3_SCHEMA,
        "checkpoint_id": session["checkpoint_id"],
        "checkpoint_binding_id": session["checkpoint_binding_id"],
        "checkpoint_file_sha256": session["checkpoint_file_sha256"],
        "model_state_sha256": session["model_state_sha256"],
        "catalogue_id": session["receipt"]["catalogue_id"],
        **identifier_payload,
        "input_receipt": input_receipt,
        "atlas_receipt": runtime_contract,
        "configuration_receipt": configuration_receipt,
        "raw_prediction_receipt": raw_prediction_receipt,
        "raw_prediction_path": str(raw_path),
        "raw_prediction_file_sha256": raw_prediction_file_sha256,
        "point_estimate": posterior["point_estimate"],
        "probabilistic_output": posterior,
        "trajectory_credible_spatial_volume": trajectory,
        "deformation_pullback_yx_px": joint["final_pullback_map_yx_px"],
        "deformation_jacobian_determinant": joint["final_forward_jacobian_determinant"],
    }
    inference_receipt_payload = {
        key: result[key]
        for key in (
            "schema_version",
            "checkpoint_id",
            "checkpoint_binding_id",
            "checkpoint_file_sha256",
            "model_state_sha256",
            "catalogue_id",
            "animal_ids",
            "specimen_ids",
            "experiment_ids",
            "synthetic_animal_ids",
            "section_ids",
            "synthetic_realization_ids",
            "lineage",
            "input_receipt",
            "atlas_receipt",
            "configuration_receipt",
            "raw_prediction_receipt",
            "raw_prediction_path",
            "raw_prediction_file_sha256",
        )
    }
    return {
        **result,
        "inference_receipt_sha256": _sha(inference_receipt_payload),
        **({"raw_prediction": raw_prediction} if return_raw_prediction else {}),
    }


def run_arbitrary_plane_inference_v3(
    loaded,
    input_b3hw,
    atlas_volume_c_ap_dv_ml,
    catalogue,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    *,
    animal_ids,
    specimen_ids,
    experiment_ids,
    synthetic_animal_ids=None,
    section_ids=None,
    synthetic_realization_ids=None,
    top_k=4,
    refinement_steps=3,
    pose_only_steps=2,
    retrieval_shape_h_w=(48, 64),
    catalogue_chunk_size=128,
    electrode_points_yx_px=None,
    annotation_volume_ap_dv_ml=None,
    gauss_hermite_order=5,
    raw_prediction_output_path=None,
    catalogue_feature_cache=None,
):
    """Authenticate one standalone inference call, then use the sealed path."""
    _validate_raw_inference_input_v3(input_b3hw)
    session = prepare_arbitrary_plane_inference_session_v3(
        loaded,
        atlas_volume_c_ap_dv_ml,
        catalogue,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        axial_offsets_um,
        axial_weights,
        annotation_volume_ap_dv_ml=annotation_volume_ap_dv_ml,
        catalogue_feature_cache=catalogue_feature_cache,
    )
    return run_arbitrary_plane_inference_session_v3(
        session,
        input_b3hw,
        animal_ids=animal_ids,
        specimen_ids=specimen_ids,
        experiment_ids=experiment_ids,
        synthetic_animal_ids=synthetic_animal_ids,
        section_ids=section_ids,
        synthetic_realization_ids=synthetic_realization_ids,
        top_k=top_k,
        refinement_steps=refinement_steps,
        pose_only_steps=pose_only_steps,
        retrieval_shape_h_w=retrieval_shape_h_w,
        catalogue_chunk_size=catalogue_chunk_size,
        electrode_points_yx_px=electrode_points_yx_px,
        gauss_hermite_order=gauss_hermite_order,
        raw_prediction_output_path=raw_prediction_output_path,
    )
