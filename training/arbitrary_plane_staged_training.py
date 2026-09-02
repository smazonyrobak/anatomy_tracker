"""Small deterministic staged trainer for the standalone arbitrary-plane model."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_training_row_v3 as training_row_v3
from training.arbitrary_plane_batch_v3 import (
    nearest_catalogue_cell_v3,
    training_row_to_tensors_v3,
)
from training.arbitrary_plane_joint_loss import (
    arbitrary_plane_joint_loss,
)
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel
from training.arbitrary_plane_training_bank_v3 import (
    COMPLETE_CATALOGUE_SCOPE,
    TRAINING_CANDIDATE_BANK_SCOPE,
    verify_training_candidate_bank_receipt_v3,
    verify_training_catalogue_batch_v3,
)


STAGED_TRAINING_SCHEMA = "anatomy-tracker.arbitrary-plane-staged-training/v3"
STAGED_TRAINING_EXPORT_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-staged-training-export/v3"
)
TRAINING_STEP_LEDGER_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-training-step-ledger/v3"
)
TRAINING_REPORT_LEDGER_EVIDENCE_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-training-report-ledger-evidence/v3"
)
DEVELOPMENT_DATA_ROLE = "development-training"
SOURCE_FILES = (
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_deformation_primitives.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_joint_model.py",
    "training/arbitrary_plane_joint_loss.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_batch_v3.py",
    "training/arbitrary_plane_psf_v4.py",
    "training/arbitrary_plane_training_bank_v3.py",
    "training/arbitrary_plane_training_row_v3.py",
    "training/arbitrary_plane_staged_training.py",
)


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    return value


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(
            _json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value.lower()) - set("0123456789abcdef"))
    )


def _source_receipts():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_receipt(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _model_state_receipts(state):
    return {name: _tensor_receipt(value) for name, value in sorted(state.items())}


def _model_state_receipt_sha256(receipts):
    return _hash_json(
        {
            "domain": "anatomy-tracker.arbitrary-plane-model-state/v3",
            "tensor_receipts": receipts,
        }
    )


def _state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]


def _development_split(split):
    value = str(split).lower()
    forbidden = ("test", "benchmark", "qualification", "external", "validation")
    if any(token in value for token in forbidden):
        raise ValueError("benchmark, validation, and final-test rows are forbidden in training")


def _row_identity(row):
    lineage = row["lineage"]
    identity = {
        "training_row_id": row["training_row_id"],
        "training_row_receipt_sha256": row["receipt_sha256"],
        "synthetic_realization_id": row["synthetic_realization_id"],
        "animal_id": lineage["animal_id"],
        "specimen_id": lineage["specimen_id"],
        "experiment_id": lineage["experiment_id"],
        "synthetic_animal_id": lineage["synthetic_animal_id"],
        "section_id": lineage["section_id"],
        "split": lineage["split"],
    }
    if row.get("schema_version") == psf_v4.TRAINING_ROW_V4_SCHEMA:
        contract = row["finite_psf_contract"]
        identity["finite_psf"] = {
            "finite_psf_sha256": contract["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": contract[
                "slab_observation_v4_receipt_sha256"
            ],
            "render_mode": contract["render_mode"],
            "nominal_cut_thickness_um": contract[
                "nominal_cut_thickness_um"
            ],
        }
    return identity


def model_ready_rows_v3(
    rows,
    catalogue,
    atlas_volume,
    *,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    support_origin_ap_dv_ml_um,
    axial_offsets_um,
    axial_weights,
    device="cpu",
    data_role=DEVELOPMENT_DATA_ROLE,
    finite_psf_capability=None,
):
    """Convert authenticated v3 rows and a fixed catalogue to model/loss tensors."""
    if data_role != DEVELOPMENT_DATA_ROLE:
        raise ValueError("staged optimization accepts development-training rows only")
    geometry = catalogue.get("support_geometry", {})
    required_geometry = {
        "origin_ap_dv_ml_um",
        "voxel_size_ap_dv_ml_um",
        "support_origin_ap_dv_ml_um",
        "support_mask_receipt",
        "raster_shape_h_w",
        "raster_physical_span_y_x_um",
    }
    if not required_geometry.issubset(geometry):
        raise ValueError("catalogue physical geometry binding is incomplete")
    if (
        not np.array_equal(
            np.asarray(origin_ap_dv_ml_um, dtype=np.float64),
            np.asarray(geometry["origin_ap_dv_ml_um"], dtype=np.float64),
        )
        or not np.array_equal(
            np.asarray(voxel_size_ap_dv_ml_um, dtype=np.float64),
            np.asarray(geometry["voxel_size_ap_dv_ml_um"], dtype=np.float64),
        )
        or not np.array_equal(
            np.asarray(support_origin_ap_dv_ml_um, dtype=np.float64),
            np.asarray(geometry["support_origin_ap_dv_ml_um"], dtype=np.float64),
        )
    ):
        raise ValueError("atlas physical geometry differs from the catalogue")
    row_schemas = {row.get("schema_version") for row in rows}
    if len(row_schemas) != 1:
        raise ValueError("one model-ready batch cannot mix v3 and v4 PSF row contracts")
    row_schema = next(iter(row_schemas), None)
    if finite_psf_capability is not None:
        psf_v4.verify_finite_psf_model_capability_v4(finite_psf_capability)
        if np.asarray(axial_offsets_um).size or np.asarray(axial_weights).size:
            raise ValueError(
                "v4 training schedules come only from authenticated rows; global PSF arrays must be empty"
            )
    for row in rows:
        if row_schema == psf_v4.TRAINING_ROW_V4_SCHEMA:
            if finite_psf_capability is None:
                raise ValueError("v4 rows require a checkpoint-bound finite-PSF capability")
            psf_v4.verify_training_row_v4(
                row, capability=finite_psf_capability
            )
        elif row_schema == training_row_v3.TRAINING_ROW_V3_SCHEMA:
            if finite_psf_capability is not None:
                raise ValueError("v3 rows cannot be reinterpreted as per-row PSF rows")
        else:
            raise ValueError("only provenance-bound v3 or v4 training rows are accepted")
        _development_split(row["lineage"]["split"])
        if any(
            row.get(name) != []
            for name in (
                "prior_model_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        ):
            raise ValueError("v3 rows must not contain learned dependencies")
        if row_schema == training_row_v3.TRAINING_ROW_V3_SCHEMA and (
            row.get("array_receipts")
            != {
                name: acquisition_v2._array_receipt(value)
                for name, value in row.get("arrays", {}).items()
            }
            or row.get("receipt_sha256")
            != acquisition_v2._payload_sha256(
                training_row_v3.training_row_receipt_v3(row)
            )
        ):
            raise ValueError("v3 training-row receipt or arrays changed")
    if row_schema == psf_v4.TRAINING_ROW_V4_SCHEMA:
        schedule_signatures = {
            (
                row["finite_psf_contract"]["render_mode"],
                row["finite_psf_contract"]["axial_sample_count"],
            )
            for row in rows
        }
        if len(schedule_signatures) != 1:
            raise ValueError(
                "one v4 batch must use one render mode and axial sample count"
            )
    target_device = torch.device(device)
    atlas = torch.as_tensor(atlas_volume, device=target_device, dtype=torch.float32)
    if tuple(atlas.shape[-3:]) != tuple(geometry["support_mask_receipt"]["shape"]):
        raise ValueError("atlas spatial shape differs from the catalogue support asset")
    converted = [
        training_row_to_tensors_v3(
            row,
            atlas_shape_ap_dv_ml=tuple(atlas.shape[-3:]),
            origin_ap_dv_ml_um=origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um=voxel_size_ap_dv_ml_um,
            device=target_device,
            finite_psf_capability=finite_psf_capability,
        )["tensors"]
        for row in rows
    ]
    truth_state = torch.cat([item["truth_state"] for item in converted])
    if any(
        tuple(item["image"].shape[-2:]) != tuple(geometry["raster_shape_h_w"])
        for item in converted
    ):
        raise ValueError("training-row canvas differs from the catalogue raster contract")
    catalogue_tensors = catalogue["tensors"]
    states = torch.as_tensor(
        catalogue_tensors["cell_states"], device=target_device, dtype=atlas.dtype
    )
    batch_size = len(rows)
    states = states.expand(batch_size, -1, -1)
    truth_catalogue_cell_index = nearest_catalogue_cell_v3(truth_state, catalogue).to(
        target_device
    )
    cell_id = torch.as_tensor(
        catalogue_tensors["cell_id"], device=target_device, dtype=torch.long
    )
    if not torch.equal(cell_id, torch.arange(cell_id.numel(), device=target_device)):
        raise ValueError("model-ready catalogue cell IDs must equal canonical row indices")
    truth_catalogue_cell_id = cell_id[truth_catalogue_cell_index]
    if not torch.equal(truth_catalogue_cell_id, truth_catalogue_cell_index):
        raise RuntimeError("canonical catalogue source-index and cell-ID contracts diverged")
    expand = lambda value: torch.as_tensor(value, device=target_device, dtype=atlas.dtype)
    axial_offsets = (
        torch.cat([item["axial_offsets_um"] for item in converted])
        if row_schema == psf_v4.TRAINING_ROW_V4_SCHEMA
        else expand(axial_offsets_um)
    )
    axial_weight_tensor = (
        torch.cat([item["axial_weights"] for item in converted])
        if row_schema == psf_v4.TRAINING_ROW_V4_SCHEMA
        else expand(axial_weights)
    )
    return {
        "data_role": data_role,
        "catalogue_id": catalogue["catalogue_id"],
        "catalogue_receipt_sha256": catalogue["receipt_sha256"],
        "full_catalogue_cell_count": int(catalogue["counts"]["cell_count"]),
        "catalogue_scope": COMPLETE_CATALOGUE_SCOPE,
        "row_identity": [_row_identity(row) for row in rows],
        "image": torch.cat([item["image"] for item in converted]),
        "outline": torch.cat([item["outline"] for item in converted]),
        "outline_available": torch.cat(
            [item["outline_available"] for item in converted]
        ),
        "atlas_volume": atlas,
        "cell_id": cell_id,
        "cell_states": states,
        "cell_log_mass": expand(catalogue_tensors["cell_log_mass"]).expand(batch_size, -1),
        "representation_log_weight": expand(
            catalogue_tensors["representation_log_weight"]
        ).expand(batch_size, -1, -1),
        "representation_to_canonical_raster_affine": expand(
            catalogue_tensors["representation_to_canonical_raster_affine"]
        ).expand(batch_size, -1, -1, -1, -1),
        "output_shape_h_w": tuple(converted[0]["image"].shape[-2:]),
        "origin_ap_dv_ml_um": tuple(origin_ap_dv_ml_um),
        "voxel_size_ap_dv_ml_um": tuple(voxel_size_ap_dv_ml_um),
        "support_origin_ap_dv_ml_um": tuple(support_origin_ap_dv_ml_um),
        "axial_offsets_um": axial_offsets,
        "axial_weights": axial_weight_tensor,
        "truth_state": truth_state,
        "pose_supervision_weight": torch.cat(
            [item["pose_supervision_weight"] for item in converted]
        ),
        "dense_deformation_supervision_weight": torch.cat(
            [item["dense_deformation_supervision_weight"] for item in converted]
        ),
        "truth_catalogue_cell_index": truth_catalogue_cell_index,
        "truth_catalogue_cell_source_index": truth_catalogue_cell_index.clone(),
        "truth_catalogue_cell_id": truth_catalogue_cell_id,
        "truth_stationary_velocity_yx_px": torch.cat(
            [item["truth_stationary_velocity_yx_px"] for item in converted]
        ),
        "truth_pullback_map_yx_px": torch.cat(
            [item["truth_pullback_map_yx_px"] for item in converted]
        ),
        "deformation_weight": torch.cat(
            [item["deformation_weight"] for item in converted]
        ),
    }


def initialize_staged_training(
    model_kwargs,
    training_config,
    *,
    catalogue_id,
    catalogue_receipt_sha256,
    catalogue_cell_count,
    generator_ids,
    device="cuda",
    finite_psf_capability=None,
):
    """Fresh random initialization; learned dependency inputs do not exist."""
    config = dict(training_config)
    required = {
        "seed",
        "pose_warmup_steps",
        "learning_rate",
        "weight_decay",
        "top_k",
        "refinement_steps",
        "joint_pose_only_steps",
        "retrieval_shape_h_w",
        "catalogue_chunk_size",
        "amp",
        "amp_initial_scale",
        "gradient_clip_norm",
    }
    if set(config) != required:
        raise ValueError(f"training config keys must be exactly {sorted(required)}")
    if config["pose_warmup_steps"] < 1 or config["joint_pose_only_steps"] < 0:
        raise ValueError("the fixed schedule requires a positive pose warmup")
    if (
        float(config["gradient_clip_norm"]) <= 0.0
        or float(config["amp_initial_scale"]) <= 0.0
    ):
        raise ValueError("gradient clipping norm and AMP initial scale must be positive")
    if config["joint_pose_only_steps"] > config["refinement_steps"] + 1:
        raise ValueError("joint pose-only iterations exceed recurrent iterations")
    seed = int(config["seed"])
    target_device = torch.device(device)
    generator_ids = tuple(str(value) for value in generator_ids)
    if finite_psf_capability is not None:
        psf_v4.verify_finite_psf_model_capability_v4(finite_psf_capability)
    if (
        not str(catalogue_id)
        or not str(catalogue_receipt_sha256)
        or not isinstance(catalogue_cell_count, int)
        or isinstance(catalogue_cell_count, bool)
        or catalogue_cell_count < 1
        or not generator_ids
    ):
        raise ValueError("catalogue and generator IDs must be nonempty")
    _set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ArbitraryPlaneJointModel(**model_kwargs).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["amp"] and target_device.type == "cuda"),
        init_scale=float(config["amp_initial_scale"]),
    )
    source_receipts = _source_receipts()
    binding = {
        "schema_version": STAGED_TRAINING_SCHEMA,
        "source_sha256": source_receipts,
        "model_kwargs": _json(model_kwargs),
        "training_config": _json(config),
        "config_id": _hash_json({"model": model_kwargs, "training": config}),
        "catalogue_id": str(catalogue_id),
        "catalogue_receipt_sha256": str(catalogue_receipt_sha256),
        "catalogue_cell_count": int(catalogue_cell_count),
        "generator_ids": sorted(generator_ids),
        "data_role": DEVELOPMENT_DATA_ROLE,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    if finite_psf_capability is not None:
        binding["finite_psf_capability"] = _json(finite_psf_capability)
    initial_hash = _state_sha256(model)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "global_step": 0,
        "binding": binding,
        "initialization_receipt": {
            "algorithm": "fresh-pytorch-random-initialization",
            "seed": seed,
            "model_state_sha256": initial_hash,
            "receipt_sha256": _hash_json(
                {"seed": seed, "model_state_sha256": initial_hash, "binding": binding}
            ),
        },
        "row_identity_records": [],
        "training_step_ledger": [],
        "device": str(target_device),
    }


def _set_deformation_trainable(model, enabled):
    for parameter in model.deformation_decoder.parameters():
        parameter.requires_grad_(enabled)


def _model_forward(state, batch, phase):
    config = state["binding"]["training_config"]
    recurrent_iterations = int(config["refinement_steps"]) + 1
    pose_only_steps = (
        recurrent_iterations
        if phase == "pose-warmup"
        else int(config["joint_pose_only_steps"])
    )
    return state["model"](
        batch["image"],
        batch["outline"],
        batch["outline_available"],
        batch["atlas_volume"],
        batch["cell_id"],
        batch["cell_states"],
        batch["cell_log_mass"],
        batch["representation_log_weight"],
        batch["representation_to_canonical_raster_affine"],
        batch["output_shape_h_w"],
        batch["origin_ap_dv_ml_um"],
        batch["voxel_size_ap_dv_ml_um"],
        batch["support_origin_ap_dv_ml_um"],
        batch["axial_offsets_um"],
        batch["axial_weights"],
        expected_catalogue_cell_count=batch["cell_states"].shape[1],
        top_k=int(config["top_k"]),
        refinement_steps=int(config["refinement_steps"]),
        pose_only_steps=pose_only_steps,
        retrieval_shape_h_w=tuple(config["retrieval_shape_h_w"]),
        catalogue_chunk_size=int(config["catalogue_chunk_size"]),
        training_truth_catalogue_index=batch["truth_catalogue_cell_id"],
        dense_deformation_supervision_weight=batch.get(
            "dense_deformation_supervision_weight",
            torch.ones(
                batch["truth_state"].shape[0],
                device=batch["truth_state"].device,
                dtype=batch["truth_state"].dtype,
            ),
        ),
    )


def _topk_truth_index(output, truth_cell_id):
    matches = output["pose"]["retrieval_topk_cell_id"] == truth_cell_id[:, None]
    return torch.where(
        matches.any(dim=1),
        matches.to(torch.int64).argmax(dim=1),
        torch.full_like(truth_cell_id, -1),
    )


def _pose_objective(losses):
    return (
        losses["retrieval_nll"]
        + 0.1 * losses["initial_plane_mixture_nll"]
        + 0.5 * losses["final_plane_mixture_nll"]
        + losses["final_landmark_mixture_nll"]
    )


def train_staged_step(state, batch):
    """Run one fixed-schedule streamed-retrieval update."""
    if batch.get("data_role") != DEVELOPMENT_DATA_ROLE:
        raise ValueError("benchmark and final-test access is forbidden")
    verify_training_catalogue_batch_v3(
        batch,
        expected_catalogue_id=state["binding"]["catalogue_id"],
        expected_catalogue_receipt_sha256=state["binding"][
            "catalogue_receipt_sha256"
        ],
        expected_full_catalogue_cell_count=state["binding"][
            "catalogue_cell_count"
        ],
    )
    existing_identity_by_row_id = {
        identity["training_row_id"]: _hash_json(identity)
        for identity in state["row_identity_records"]
    }
    batch_identity_by_row_id = {}
    for identity in batch["row_identity"]:
        row_id = identity["training_row_id"]
        identity_hash = _hash_json(identity)
        if (
            row_id in existing_identity_by_row_id
            and existing_identity_by_row_id[row_id] != identity_hash
        ) or (
            row_id in batch_identity_by_row_id
            and batch_identity_by_row_id[row_id] != identity_hash
        ):
            raise ValueError("one training row ID cannot identify multiple row receipts")
        batch_identity_by_row_id[row_id] = identity_hash
    bank_receipts = batch.get("training_candidate_bank_receipts", [])
    step = int(state["global_step"])
    warmup = int(state["binding"]["training_config"]["pose_warmup_steps"])
    phase = "pose-warmup" if step < warmup else "joint"
    joint = phase == "joint"
    _set_deformation_trainable(state["model"], joint)
    state["model"].train()
    state["optimizer"].zero_grad(set_to_none=True)
    use_amp = bool(
        state["binding"]["training_config"]["amp"]
        and torch.device(state["device"]).type == "cuda"
    )
    with torch.amp.autocast("cuda", enabled=use_amp):
        output = _model_forward(state, batch, phase)
        truth_topk_index = _topk_truth_index(
            output, batch["truth_catalogue_cell_id"]
        )
        losses = arbitrary_plane_joint_loss(
            output,
            batch["truth_state"],
            batch["truth_catalogue_cell_id"],
            truth_topk_index,
            batch["truth_stationary_velocity_yx_px"],
            batch["truth_pullback_map_yx_px"],
            batch["deformation_weight"],
            batch["support_origin_ap_dv_ml_um"],
            pose_supervision_weight=batch.get(
                "pose_supervision_weight",
                torch.ones(
                    batch["truth_state"].shape[0],
                    device=batch["truth_state"].device,
                    dtype=batch["truth_state"].dtype,
                ),
            ),
            dense_deformation_supervision_weight=batch.get(
                "dense_deformation_supervision_weight",
                torch.ones(
                    batch["truth_state"].shape[0],
                    device=batch["truth_state"].device,
                    dtype=batch["truth_state"].dtype,
                ),
            ),
        )
        objective = losses["total"] if joint else _pose_objective(losses)
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("nonfinite staged-training objective")
    state["scaler"].scale(objective).backward()
    amp_scale_before = float(state["scaler"].get_scale())
    state["scaler"].unscale_(state["optimizer"])
    nonfinite_gradient = any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in state["model"].parameters()
    )
    if nonfinite_gradient:
        if not use_amp:
            raise FloatingPointError("nonfinite staged-training gradient")
        state["scaler"].step(state["optimizer"])
        state["scaler"].update()
        state["optimizer"].zero_grad(set_to_none=True)
        return {
            "step": step,
            "phase": phase,
            "objective": float(objective.detach()),
            "optimizer_step_applied": False,
            "amp_overflow": True,
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": float(state["scaler"].get_scale()),
            "deformation_decoder_called": bool(
                output["deformation_active_sequence"].any()
            ),
            "deformation_loss_enabled": joint,
            "retrieval_scope": batch.get(
                "catalogue_scope", COMPLETE_CATALOGUE_SCOPE
            ),
        }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        state["model"].parameters(),
        float(state["binding"]["training_config"]["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    state["scaler"].step(state["optimizer"])
    state["scaler"].update()
    amp_scale_after = float(state["scaler"].get_scale())
    known_by_row_id = {
        item["training_row_id"]: _hash_json(item)
        for item in state["row_identity_records"]
    }
    for identity in batch["row_identity"]:
        key = _hash_json(identity)
        row_id = identity["training_row_id"]
        if row_id in known_by_row_id and known_by_row_id[row_id] != key:
            raise ValueError("one training row ID cannot identify multiple row receipts")
        if row_id not in known_by_row_id:
            state["row_identity_records"].append(_json(identity))
            known_by_row_id[row_id] = key
    state["global_step"] = step + 1
    ledger_payload = {
        "step": step,
        "catalogue_scope": batch["catalogue_scope"],
        "training_row_ids": [
            identity["training_row_id"] for identity in batch["row_identity"]
        ],
        "training_row_receipt_sha256": [
            identity["training_row_receipt_sha256"]
            for identity in batch["row_identity"]
        ],
        "training_row_identity_sha256": [
            _hash_json(identity) for identity in batch["row_identity"]
        ],
        "training_candidate_bank_receipt_sha256": [
            receipt["receipt_sha256"] for receipt in bank_receipts
        ],
    }
    state["training_step_ledger"].append(
        _training_step_ledger_entry_v3(
            state["binding"], state["training_step_ledger"], ledger_payload
        )
    )
    report = {name: float(value.detach()) for name, value in losses.items()}
    if not joint:
        for name in tuple(report):
            if name.startswith("deformation_"):
                report[name] = 0.0
    honest_matches = output["pose"]["honest_retrieval_topk_cell_id"].eq(
        batch["truth_catalogue_cell_id"][:, None]
    )
    identifiable = batch.get(
        "pose_supervision_weight",
        torch.ones(
            batch["truth_state"].shape[0],
            device=batch["truth_state"].device,
            dtype=batch["truth_state"].dtype,
        ),
    ) > 0.0
    identifiable_count = identifiable.to(torch.float32).sum().clamp_min(1.0)
    return {
        "step": step,
        "phase": phase,
        "objective": float(objective.detach()),
        "optimizer_step_applied": True,
        "amp_overflow": False,
        "amp_scale_before": amp_scale_before,
        "amp_scale_after": amp_scale_after,
        "deformation_decoder_called": bool(
            output["deformation_active_sequence"].any()
        ),
        "deformation_loss_enabled": joint,
        "truth_in_topk_fraction": float(
            (
                honest_matches.any(dim=1).to(torch.float32)
                * identifiable.to(torch.float32)
            ).sum()
            / identifiable_count
        ),
        "pose_identifiable_fraction": float(identifiable.to(torch.float32).mean()),
        "truth_forced_refinement_fraction": float(
            output["pose"]["retrieval_teacher_forced_mask"].float().mean()
        ),
        "retrieval_scope": batch.get(
            "catalogue_scope", COMPLETE_CATALOGUE_SCOPE
        ),
        "preclip_gradient_norm": float(gradient_norm),
        "losses": report,
    }


def _checkpoint_path(path):
    resolved = Path(path).resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("training checkpoints must be written only on I:")
    return resolved


_TRAINING_STEP_PAYLOAD_KEYS = (
    "step",
    "catalogue_scope",
    "training_row_ids",
    "training_row_receipt_sha256",
    "training_row_identity_sha256",
    "training_candidate_bank_receipt_sha256",
)


def _training_step_ledger_genesis_v3(binding):
    return _hash_json(
        {
            "domain": f"{TRAINING_STEP_LEDGER_SCHEMA}/genesis",
            "binding": binding,
        }
    )


def _training_step_ledger_entry_v3(binding, preceding_entries, payload):
    previous = (
        preceding_entries[-1]["chain_sha256"]
        if preceding_entries
        else _training_step_ledger_genesis_v3(binding)
    )
    entry_sha256 = _hash_json(payload)
    chain_sha256 = _hash_json(
        {
            "domain": f"{TRAINING_STEP_LEDGER_SCHEMA}/ordered-entry",
            "previous_chain_sha256": previous,
            "entry_sha256": entry_sha256,
            "step": payload["step"],
        }
    )
    return {
        **_json(payload),
        "entry_sha256": entry_sha256,
        "previous_chain_sha256": previous,
        "chain_sha256": chain_sha256,
    }


def _normalize_training_step_ledger_v3(binding, step_ledger, *, allow_legacy):
    normalized = []
    for entry in step_ledger:
        if not isinstance(entry, dict):
            raise ValueError("training step ledger failed verification")
        payload = {key: entry.get(key) for key in _TRAINING_STEP_PAYLOAD_KEYS}
        expected = _training_step_ledger_entry_v3(binding, normalized, payload)
        if entry.get("entry_sha256") != expected["entry_sha256"]:
            raise ValueError("training step ledger failed verification")
        has_chain = "previous_chain_sha256" in entry or "chain_sha256" in entry
        if has_chain and (
            entry.get("previous_chain_sha256")
            != expected["previous_chain_sha256"]
            or entry.get("chain_sha256") != expected["chain_sha256"]
        ):
            raise ValueError("training step ledger hash chain failed verification")
        if not has_chain and not allow_legacy:
            raise ValueError("training step ledger hash chain is missing")
        if set(entry) - {
            *_TRAINING_STEP_PAYLOAD_KEYS,
            "entry_sha256",
            "previous_chain_sha256",
            "chain_sha256",
        }:
            raise ValueError("training step ledger contains unknown fields")
        normalized.append(expected)
    return normalized


def _training_step_ledger_summary_v3(binding, step_ledger):
    genesis = _training_step_ledger_genesis_v3(binding)
    payload = {
        "schema_version": TRAINING_STEP_LEDGER_SCHEMA,
        "entry_count": len(step_ledger),
        "genesis_chain_sha256": genesis,
        "final_chain_sha256": (
            step_ledger[-1]["chain_sha256"] if step_ledger else genesis
        ),
        "compact_ledger_sha256": _hash_json(step_ledger),
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def _verify_training_history(
    binding,
    global_step,
    row_identity_records,
    step_ledger,
    *,
    summary=None,
    allow_legacy_chain=False,
):
    identity_by_row_id = {}
    for identity in row_identity_records:
        row_id = identity["training_row_id"]
        identity_hash = _hash_json(identity)
        if row_id in identity_by_row_id and identity_by_row_id[row_id][0] != identity_hash:
            raise ValueError("one training row ID identifies multiple row receipts")
        identity_by_row_id[row_id] = (identity_hash, identity)
    seen_rows = set(identity_by_row_id)
    normalized = _normalize_training_step_ledger_v3(
        binding, step_ledger, allow_legacy=allow_legacy_chain
    )
    if len(step_ledger) != int(global_step):
        raise ValueError("training step ledger length differs from global step")
    for expected_step, entry in enumerate(normalized):
        payload = {key: entry[key] for key in _TRAINING_STEP_PAYLOAD_KEYS}
        scope = payload.get("catalogue_scope")
        row_ids = payload.get("training_row_ids", [])
        row_receipts = payload.get("training_row_receipt_sha256", [])
        row_identity_hashes = payload.get("training_row_identity_sha256", [])
        receipt_ids = payload.get("training_candidate_bank_receipt_sha256", [])
        if (
            payload.get("step") != expected_step
            or not row_ids
            or len(row_receipts) != len(row_ids)
            or len(row_identity_hashes) != len(row_ids)
            or any(row_id not in seen_rows for row_id in row_ids)
            or any(
                identity_by_row_id[row_id][1]["training_row_receipt_sha256"]
                != row_receipt
                or identity_by_row_id[row_id][0] != identity_hash
                for row_id, row_receipt, identity_hash in zip(
                    row_ids, row_receipts, row_identity_hashes
                )
            )
            or scope not in (COMPLETE_CATALOGUE_SCOPE, TRAINING_CANDIDATE_BANK_SCOPE)
            or (scope == COMPLETE_CATALOGUE_SCOPE and receipt_ids)
            or (
                scope == TRAINING_CANDIDATE_BANK_SCOPE
                and len(receipt_ids) != len(row_ids)
            )
            or any(not _is_sha256(receipt_id) for receipt_id in receipt_ids)
        ):
            raise ValueError("training step ledger failed verification")
    expected_summary = _training_step_ledger_summary_v3(binding, normalized)
    if summary is not None and summary != expected_summary:
        raise ValueError("training step ledger summary failed verification")
    return normalized, expected_summary


def _seen_identity_values(records, name):
    return sorted(
        {_hash_json(item[name]): item[name] for item in records}.values(),
        key=_hash_json,
    )


def verify_staged_training_checkpoint_payload_v3(
    checkpoint, *, expected_binding=None, replay_initialization=True
):
    """Verify one complete staged checkpoint without mutating its contents."""
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != STAGED_TRAINING_SCHEMA:
        raise ValueError("staged-training checkpoint schema differs")
    binding = checkpoint.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("staged-training checkpoint binding is missing")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("checkpoint source/config/catalogue/generator binding differs")
    if binding.get("source_sha256") != _source_receipts():
        raise ValueError("checkpoint source hashes differ from the running code")
    if "finite_psf_capability" in binding:
        psf_v4.verify_finite_psf_model_capability_v4(
            binding["finite_psf_capability"]
        )
    dependencies = checkpoint.get("learned_dependency_arrays")
    if dependencies != {
        "prior_model_weights": [],
        "prior_features": [],
        "prior_pseudolabels": [],
    }:
        raise ValueError("checkpoint contains forbidden learned dependencies")
    records = checkpoint.get("row_identity_records")
    if not isinstance(records, list):
        raise ValueError("training row identity history is missing")
    _verify_training_history(
        binding,
        checkpoint.get("global_step"),
        records,
        checkpoint.get("training_step_ledger", []),
        summary=checkpoint.get("training_step_ledger_summary"),
    )
    if "training_candidate_bank_receipts" in checkpoint:
        raise ValueError("checkpoint contains noncompact candidate-bank receipts")
    expected_seen = {
        "seen_training_row_ids": sorted(
            {item["training_row_id"] for item in records}
        ),
        "seen_animal_ids": _seen_identity_values(records, "animal_id"),
        "seen_specimen_ids": _seen_identity_values(records, "specimen_id"),
        "seen_experiment_ids": _seen_identity_values(records, "experiment_id"),
    }
    if any(checkpoint.get(name) != value for name, value in expected_seen.items()):
        raise ValueError("checkpoint seen-ID summaries differ from row provenance")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or not model_state or not all(
        isinstance(value, torch.Tensor) for value in model_state.values()
    ):
        raise ValueError("checkpoint model state is missing")
    initialization = checkpoint.get("initialization_receipt")
    if (
        not isinstance(initialization, dict)
        or initialization.get("algorithm")
        != "fresh-pytorch-random-initialization"
        or initialization.get("seed") != binding["training_config"]["seed"]
        or initialization.get("receipt_sha256")
        != _hash_json(
            {
                "seed": initialization.get("seed"),
                "model_state_sha256": initialization.get("model_state_sha256"),
                "binding": binding,
            }
        )
    ):
        raise ValueError("fresh initialization receipt is invalid")
    if replay_initialization:
        saved_rng = _rng_state()
        try:
            replay = initialize_staged_training(
                binding["model_kwargs"],
                binding["training_config"],
                catalogue_id=binding["catalogue_id"],
                catalogue_receipt_sha256=binding["catalogue_receipt_sha256"],
                catalogue_cell_count=binding["catalogue_cell_count"],
                generator_ids=binding["generator_ids"],
                device="cpu",
                finite_psf_capability=binding.get("finite_psf_capability"),
            )
            if replay["initialization_receipt"] != initialization:
                raise ValueError("fresh initialization receipt does not replay")
            replay["model"].load_state_dict(model_state, strict=True)
        finally:
            _restore_rng_state(saved_rng)
    return True


def verify_training_report_ledger_correspondence_v3(
    checkpoint,
    reports,
    *,
    expected_run_id=None,
    expected_run_manifest_receipt_sha256=None,
):
    """Verify full immutable report receipts against the compact checkpoint ledger."""
    verify_staged_training_checkpoint_payload_v3(
        checkpoint, replay_initialization=False
    )
    binding = checkpoint["binding"]
    compact_ledger = checkpoint["training_step_ledger"]
    identity_by_row_id = {
        identity["training_row_id"]: identity
        for identity in checkpoint["row_identity_records"]
    }
    applied = 0
    report_receipts = []
    run_id = expected_run_id
    manifest_receipt = expected_run_manifest_receipt_sha256
    if reports:
        if run_id is None:
            run_id = reports[0].get("run_id")
        if manifest_receipt is None:
            manifest_receipt = reports[0].get(
                "run_manifest_receipt_sha256"
            )
    report_chain = _hash_json(
        {
            "domain": f"{TRAINING_REPORT_LEDGER_EVIDENCE_SCHEMA}/genesis",
            "run_id": run_id,
            "run_manifest_receipt_sha256": manifest_receipt,
        }
    )
    for attempt_index, report in enumerate(reports):
        if not isinstance(report, dict):
            raise ValueError("training report ledger correspondence failed")
        report_payload = {
            key: value for key, value in report.items() if key != "receipt_sha256"
        }
        report_receipt = report.get("receipt_sha256")
        if report_receipt != _hash_json(report_payload):
            raise ValueError("training report receipt failed authentication")
        row_identity = report_payload.get("row_identity")
        receipts = report_payload.get("training_candidate_bank_receipts")
        scope = report_payload.get("retrieval_scope")
        before = report_payload.get("global_step_before")
        after = report_payload.get("global_step_after")
        train_report = report_payload.get("training_report", {})
        was_applied = bool(train_report.get("optimizer_step_applied"))
        if (
            report_payload.get("schema_version")
            != "anatomy-tracker.arbitrary-plane-training-step-report/v3"
            or report_payload.get("attempt_index") != attempt_index
            or report_payload.get("run_id") != run_id
            or report_payload.get("run_manifest_receipt_sha256")
            != manifest_receipt
            or before != applied
            or after not in (applied, applied + 1)
            or was_applied != (after == applied + 1)
            or train_report.get("retrieval_scope") != scope
            or not isinstance(row_identity, list)
            or not row_identity
            or not isinstance(receipts, list)
            or scope
            not in (COMPLETE_CATALOGUE_SCOPE, TRAINING_CANDIDATE_BANK_SCOPE)
            or (scope == COMPLETE_CATALOGUE_SCOPE and receipts)
            or (
                scope == TRAINING_CANDIDATE_BANK_SCOPE
                and len(receipts) != len(row_identity)
            )
        ):
            raise ValueError("training report ledger correspondence failed")
        for identity, receipt in zip(row_identity, receipts):
            verify_training_candidate_bank_receipt_v3(
                receipt,
                expected_catalogue_id=binding["catalogue_id"],
                expected_catalogue_receipt_sha256=binding[
                    "catalogue_receipt_sha256"
                ],
                expected_training_row_id=identity["training_row_id"],
                expected_training_row_receipt_sha256=identity[
                    "training_row_receipt_sha256"
                ],
                expected_training_row_identity_sha256=_hash_json(identity),
            )
        if was_applied:
            payload = {
                "step": applied,
                "catalogue_scope": scope,
                "training_row_ids": [
                    identity["training_row_id"] for identity in row_identity
                ],
                "training_row_receipt_sha256": [
                    identity["training_row_receipt_sha256"]
                    for identity in row_identity
                ],
                "training_row_identity_sha256": [
                    _hash_json(identity) for identity in row_identity
                ],
                "training_candidate_bank_receipt_sha256": [
                    receipt["receipt_sha256"] for receipt in receipts
                ],
            }
            if (
                applied >= len(compact_ledger)
                or any(
                    compact_ledger[applied].get(key) != value
                    for key, value in payload.items()
                )
                or any(
                    identity_by_row_id.get(identity["training_row_id"])
                    != identity
                    for identity in row_identity
                )
            ):
                raise ValueError(
                    "checkpoint and training report ledgers differ"
                )
            applied += 1
        report_receipts.append(report_receipt)
        report_chain = _hash_json(
            {
                "domain": (
                    f"{TRAINING_REPORT_LEDGER_EVIDENCE_SCHEMA}/ordered-report"
                ),
                "previous_chain_sha256": report_chain,
                "attempt_index": attempt_index,
                "report_receipt_sha256": report_receipt,
            }
        )
    if applied != int(checkpoint["global_step"]):
        raise ValueError("checkpoint and training report applied-step counts differ")
    evidence_payload = {
        "schema_version": TRAINING_REPORT_LEDGER_EVIDENCE_SCHEMA,
        "run_id": run_id,
        "run_manifest_receipt_sha256": manifest_receipt,
        "report_count": len(reports),
        "applied_step_count": applied,
        "ordered_report_receipts_sha256": _hash_json(report_receipts),
        "final_report_chain_sha256": report_chain,
    }
    return {
        **evidence_payload,
        "receipt_sha256": _hash_json(evidence_payload),
    }


def _runner_report_ledger_for_checkpoint_v3(target, checkpoint):
    if target.parent.name != "checkpoints":
        raise ValueError(
            "sampled-bank export requires its authenticated training-run reports"
        )
    run_root = target.parent.parent.resolve()
    manifest_path = run_root / "run_manifest.json"
    run_state_path = run_root / "run_state.json"
    if not manifest_path.is_file() or not run_state_path.is_file():
        raise ValueError(
            "sampled-bank export requires its authenticated training-run reports"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "receipt_sha256"
    }
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    run_state_payload = {
        key: value for key, value in run_state.items() if key != "receipt_sha256"
    }
    records = run_state_payload.get("committed_reports", [])
    if (
        manifest.get("receipt_sha256") != _hash_json(manifest_payload)
        or manifest_payload.get("schema_version")
        != "anatomy-tracker.arbitrary-plane-training-run/v3"
        or manifest_payload.get("staged_training_binding")
        != checkpoint["binding"]
        or run_state.get("receipt_sha256") != _hash_json(run_state_payload)
        or run_state_payload.get("schema_version")
        != "anatomy-tracker.arbitrary-plane-training-run-state/v3"
        or run_state_payload.get("run_id") != manifest_payload.get("run_id")
        or run_state_payload.get("run_manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or run_state_payload.get("attempt_count") != len(records)
    ):
        raise ValueError("training-run report ledger failed authentication")
    target_relative = target.relative_to(run_root).as_posix()
    target_sha256 = _file_sha256(target)
    reports = []
    target_report_index = None
    for attempt_index, record in enumerate(records):
        report_path = (run_root / record["relative_path"]).resolve()
        if run_root not in report_path.parents:
            raise ValueError("training report path escapes its run directory")
        if _file_sha256(report_path) != record.get("file_sha256"):
            raise ValueError("committed training-step report hash differs")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload = {
            key: value for key, value in report.items() if key != "receipt_sha256"
        }
        if (
            report.get("receipt_sha256") != _hash_json(report_payload)
            or report.get("receipt_sha256")
            != record.get("report_receipt_sha256")
            or report_payload.get("attempt_index") != attempt_index
        ):
            raise ValueError("committed training-step report failed authentication")
        reports.append(report)
        references = [report_payload.get("checkpoint")]
        if report_payload.get("archive_checkpoint") is not None:
            references.append(report_payload["archive_checkpoint"])
        if any(
            reference.get("relative_path") == target_relative
            and reference.get("file_sha256") == target_sha256
            for reference in references
            if isinstance(reference, dict)
        ):
            target_report_index = attempt_index
    if target_report_index is None:
        raise ValueError("checkpoint is not bound by its training report ledger")
    prefix = reports[: target_report_index + 1]
    if prefix[-1].get("global_step_after") != checkpoint["global_step"]:
        raise ValueError("checkpoint and training report global steps differ")
    return prefix, manifest


def make_staged_training_export_receipt_v3(path, *, training_report_ledger=None):
    """Export a file-bound, inference-safe receipt from one verified staged checkpoint."""
    target = _checkpoint_path(path)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    verify_staged_training_checkpoint_payload_v3(checkpoint)
    records = checkpoint["row_identity_records"]
    if int(checkpoint["global_step"]) < 1 or not records:
        raise ValueError("an inference export requires at least one applied training step and row")
    sampled_bank_step_count = sum(
        entry["catalogue_scope"] == TRAINING_CANDIDATE_BANK_SCOPE
        for entry in checkpoint["training_step_ledger"]
    )
    report_evidence = None
    if sampled_bank_step_count:
        expected_run_id = None
        expected_manifest_receipt = None
        if training_report_ledger is None:
            training_report_ledger, manifest = (
                _runner_report_ledger_for_checkpoint_v3(target, checkpoint)
            )
            expected_run_id = manifest["run_id"]
            expected_manifest_receipt = manifest["receipt_sha256"]
        report_evidence = verify_training_report_ledger_correspondence_v3(
            checkpoint,
            training_report_ledger,
            expected_run_id=expected_run_id,
            expected_run_manifest_receipt_sha256=expected_manifest_receipt,
        )
    model_receipts = _model_state_receipts(checkpoint["model_state"])
    payload = {
        "schema_version": STAGED_TRAINING_EXPORT_SCHEMA,
        "staged_training_schema_version": STAGED_TRAINING_SCHEMA,
        "staged_checkpoint_path": str(target),
        "staged_checkpoint_file_sha256": _file_sha256(target),
        "binding": checkpoint["binding"],
        "initialization_receipt": checkpoint["initialization_receipt"],
        "global_step": int(checkpoint["global_step"]),
        "model_state_receipts": model_receipts,
        "model_state_sha256": _model_state_receipt_sha256(model_receipts),
        "row_identity_record_count": len(records),
        "row_identity_records_sha256": _hash_json(records),
        "candidate_bank_receipt_storage": "immutable-training-run-reports-only",
        "sampled_bank_step_count": sampled_bank_step_count,
        "training_step_ledger_count": len(checkpoint["training_step_ledger"]),
        "training_step_ledger_sha256": _hash_json(checkpoint["training_step_ledger"]),
        "training_step_ledger_summary": checkpoint[
            "training_step_ledger_summary"
        ],
        "training_report_ledger_evidence": report_evidence,
        "training_row_ids": checkpoint["seen_training_row_ids"],
        "training_animal_ids": checkpoint["seen_animal_ids"],
        "training_specimen_ids": checkpoint["seen_specimen_ids"],
        "training_experiment_ids": checkpoint["seen_experiment_ids"],
        "learned_dependency_arrays": checkpoint["learned_dependency_arrays"],
    }
    return {**payload, "receipt_sha256": _hash_json(payload)}


def verify_staged_training_export_receipt_v3(
    receipt,
    *,
    model_kwargs,
    catalogue_id,
    catalogue_receipt_sha256,
    catalogue_cell_count,
    model_state_sha256,
    require_source_file=False,
):
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    } if isinstance(receipt, dict) else {}
    binding = receipt.get("binding", {}) if isinstance(receipt, dict) else {}
    valid = (
        isinstance(receipt, dict)
        and receipt.get("schema_version") == STAGED_TRAINING_EXPORT_SCHEMA
        and receipt.get("staged_training_schema_version") == STAGED_TRAINING_SCHEMA
        and receipt.get("receipt_sha256") == _hash_json(payload)
        and binding.get("schema_version") == STAGED_TRAINING_SCHEMA
        and binding.get("source_sha256") == _source_receipts()
        and binding.get("model_kwargs") == _json(model_kwargs)
        and binding.get("catalogue_id") == str(catalogue_id)
        and binding.get("catalogue_receipt_sha256")
        == str(catalogue_receipt_sha256)
        and binding.get("catalogue_cell_count") == int(catalogue_cell_count)
        and binding.get("prior_model_weight_dependencies") == []
        and binding.get("prior_feature_dependencies") == []
        and binding.get("prior_pseudolabel_dependencies") == []
        and receipt.get("model_state_sha256") == model_state_sha256
        and isinstance(receipt.get("model_state_receipts"), dict)
        and receipt.get("model_state_sha256")
        == _model_state_receipt_sha256(receipt["model_state_receipts"])
        and isinstance(receipt.get("global_step"), int)
        and receipt["global_step"] >= 1
        and receipt.get("training_step_ledger_count") == receipt["global_step"]
        and isinstance(receipt.get("sampled_bank_step_count"), int)
        and 0 <= receipt["sampled_bank_step_count"] <= receipt["global_step"]
        and receipt.get("candidate_bank_receipt_storage")
        == "immutable-training-run-reports-only"
        and isinstance(receipt.get("training_step_ledger_summary"), dict)
        and receipt["training_step_ledger_summary"].get("schema_version")
        == TRAINING_STEP_LEDGER_SCHEMA
        and receipt["training_step_ledger_summary"].get("entry_count")
        == receipt["global_step"]
        and receipt["training_step_ledger_summary"].get("compact_ledger_sha256")
        == receipt.get("training_step_ledger_sha256")
        and receipt["training_step_ledger_summary"].get("receipt_sha256")
        == _hash_json(
            {
                key: value
                for key, value in receipt["training_step_ledger_summary"].items()
                if key != "receipt_sha256"
            }
        )
        and (
            (
                receipt["sampled_bank_step_count"] == 0
                and receipt.get("training_report_ledger_evidence") is None
            )
            or (
                receipt["sampled_bank_step_count"] > 0
                and isinstance(
                    receipt.get("training_report_ledger_evidence"), dict
                )
                and receipt["training_report_ledger_evidence"].get(
                    "schema_version"
                )
                == TRAINING_REPORT_LEDGER_EVIDENCE_SCHEMA
                and receipt["training_report_ledger_evidence"].get(
                    "applied_step_count"
                )
                == receipt["global_step"]
                and receipt["training_report_ledger_evidence"].get(
                    "receipt_sha256"
                )
                == _hash_json(
                    {
                        key: value
                        for key, value in receipt[
                            "training_report_ledger_evidence"
                        ].items()
                        if key != "receipt_sha256"
                    }
                )
            )
        )
        and isinstance(receipt.get("row_identity_record_count"), int)
        and receipt["row_identity_record_count"] >= 1
        and all(
            isinstance(receipt.get(name), list) and bool(receipt[name])
            for name in (
                "training_row_ids",
                "training_animal_ids",
                "training_specimen_ids",
                "training_experiment_ids",
            )
        )
        and receipt.get("learned_dependency_arrays")
        == {
            "prior_model_weights": [],
            "prior_features": [],
            "prior_pseudolabels": [],
        }
        and all(
            isinstance(receipt.get(name), str)
            and len(receipt[name]) == 64
            and not (set(receipt[name].lower()) - set("0123456789abcdef"))
            for name in (
                "staged_checkpoint_file_sha256",
                "row_identity_records_sha256",
                "training_step_ledger_sha256",
            )
        )
    )
    if valid and "finite_psf_capability" in binding:
        try:
            psf_v4.verify_finite_psf_model_capability_v4(
                binding["finite_psf_capability"]
            )
        except (TypeError, ValueError):
            valid = False
    if valid and require_source_file:
        try:
            target = _checkpoint_path(receipt["staged_checkpoint_path"])
            valid = (
                target.is_file()
                and _file_sha256(target)
                == receipt["staged_checkpoint_file_sha256"]
                and make_staged_training_export_receipt_v3(target) == receipt
            )
        except (OSError, TypeError, ValueError):
            valid = False
    if not valid:
        raise ValueError("staged-training export receipt is invalid or mismatched")
    return True


def save_staged_training_checkpoint(state, path):
    """Atomically save exact model, optimizer, scaler, RNG, and provenance state."""
    target = _checkpoint_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(state["row_identity_records"])
    compact_ledger, ledger_summary = _verify_training_history(
        state["binding"],
        state["global_step"],
        records,
        state["training_step_ledger"],
        allow_legacy_chain=True,
    )
    state["training_step_ledger"] = compact_ledger
    checkpoint = {
        "schema_version": STAGED_TRAINING_SCHEMA,
        "binding": state["binding"],
        "initialization_receipt": state["initialization_receipt"],
        "global_step": int(state["global_step"]),
        "model_state": state["model"].state_dict(),
        "optimizer_state": state["optimizer"].state_dict(),
        "scaler_state": state["scaler"].state_dict(),
        "rng_state": _rng_state(),
        "row_identity_records": records,
        "training_step_ledger": compact_ledger,
        "training_step_ledger_summary": ledger_summary,
        "seen_training_row_ids": sorted({item["training_row_id"] for item in records}),
        "seen_animal_ids": sorted({_hash_json(item["animal_id"]): item["animal_id"] for item in records}.values(), key=_hash_json),
        "seen_specimen_ids": sorted({_hash_json(item["specimen_id"]): item["specimen_id"] for item in records}.values(), key=_hash_json),
        "seen_experiment_ids": sorted({_hash_json(item["experiment_id"]): item["experiment_id"] for item in records}.values(), key=_hash_json),
        "learned_dependency_arrays": {
            "prior_model_weights": [],
            "prior_features": [],
            "prior_pseudolabels": [],
        },
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_staged_training_checkpoint(
    path,
    *,
    device="cuda",
    expected_binding=None,
    training_report_ledger=None,
):
    """Restore an exact between-step state and all random number generators."""
    target = _checkpoint_path(path)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    verify_staged_training_checkpoint_payload_v3(
        checkpoint,
        expected_binding=expected_binding,
        replay_initialization=False,
    )
    sampled_bank_history = any(
        entry["catalogue_scope"] == TRAINING_CANDIDATE_BANK_SCOPE
        for entry in checkpoint["training_step_ledger"]
    )
    if sampled_bank_history:
        expected_run_id = None
        expected_manifest_receipt = None
        if training_report_ledger is None:
            training_report_ledger, manifest = (
                _runner_report_ledger_for_checkpoint_v3(target, checkpoint)
            )
            expected_run_id = manifest["run_id"]
            expected_manifest_receipt = manifest["receipt_sha256"]
        verify_training_report_ledger_correspondence_v3(
            checkpoint,
            training_report_ledger,
            expected_run_id=expected_run_id,
            expected_run_manifest_receipt_sha256=expected_manifest_receipt,
        )
    binding = checkpoint["binding"]
    state = initialize_staged_training(
        binding["model_kwargs"],
        binding["training_config"],
        catalogue_id=binding["catalogue_id"],
        catalogue_receipt_sha256=binding["catalogue_receipt_sha256"],
        catalogue_cell_count=binding["catalogue_cell_count"],
        generator_ids=binding["generator_ids"],
        device=device,
        finite_psf_capability=binding.get("finite_psf_capability"),
    )
    if state["initialization_receipt"] != checkpoint["initialization_receipt"]:
        raise ValueError("fresh initialization receipt does not replay")
    state["model"].load_state_dict(checkpoint["model_state"])
    state["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    state["scaler"].load_state_dict(checkpoint["scaler_state"])
    state["global_step"] = int(checkpoint["global_step"])
    state["row_identity_records"] = list(checkpoint["row_identity_records"])
    state["training_step_ledger"] = list(checkpoint.get("training_step_ledger", []))
    _restore_rng_state(checkpoint["rng_state"])
    return state
