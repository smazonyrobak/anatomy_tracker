"""Fresh-init three-stage trainer for the complete v6 retrieval cascade."""

from __future__ import annotations

import hashlib
import json
import os
import random
import copy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_catalogue_runtime_v6 import (
    CompleteCatalogueRuntimeV6,
    verify_bound_complete_catalogue_batch_v6,
    verify_complete_catalogue_runtime_v6,
)
from training.arbitrary_plane_joint_loss_v6 import arbitrary_plane_joint_loss_v6
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_retrieval_loss_v6 import (
    full_catalogue_proposal_nll_v6,
    selected_exact_rerank_nll_v6,
)


STAGED_TRAINER_V6_SCHEMA = "anatomy-tracker.staged-trainer/v6"
STAGED_TRAINER_V6_MANIFEST_SCHEMA = "anatomy-tracker.staged-trainer-manifest/v6"
STAGED_TRAINER_V6_CHECKPOINT_SCHEMA = "anatomy-tracker.staged-trainer-checkpoint/v6"
FROZEN_ROWS_V6_SCHEMA = "anatomy-tracker.frozen-generated-row-payloads/v6"
ROW_RECEIPTS_V6_SCHEMA = "anatomy-tracker.training-row-receipts/v6"
FULL_CATALOGUE_CELL_COUNT_V6 = 98_304
RAW_INPUT_MODE_V6 = "raw"
BLACK_EXTERIOR_INPUT_MODE_V6 = "black-exterior"
IMPERFECT_MASK_INPUT_MODE_V6 = "imperfect-mask"
INPUT_MODES_V6 = (
    RAW_INPUT_MODE_V6,
    BLACK_EXTERIOR_INPUT_MODE_V6,
    IMPERFECT_MASK_INPUT_MODE_V6,
)
PROVENANCE_KEYS_V6 = (
    "specimen_id",
    "animal_id",
    "experiment_id",
    "section_id",
    "synthetic_animal_id",
)
OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6 = (
    "training_row_id",
    "training_row_receipt_sha256",
    "row_id",
    "row_receipt_sha256",
    "provenance_sha256",
    "provenance_receipt_sha256",
)
_CONFIG_KEYS = {
    "seed",
    "proposal_only_steps",
    "pose_rerank_steps",
    "learning_rate",
    "weight_decay",
    "proposal_top_m",
    "top_k",
    "refinement_steps",
    "joint_pose_only_steps",
    "retrieval_shape_h_w",
    "amp",
    "amp_initial_scale",
    "gradient_clip_norm",
    "proposal_loss_weight",
    "rerank_loss_weight",
}
_SOURCE_FILES = (
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_full_frame_primitives.py",
    "training/arbitrary_plane_deformation_primitives.py",
    "training/arbitrary_plane_manifest.py",
    "training/arbitrary_plane_support.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_acquisition_v2.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_catalogue_binding_v3.py",
    "training/arbitrary_plane_finite_row_binding_v6.py",
    "training/arbitrary_plane_training_data_v6.py",
    "training/arbitrary_plane_catalogue_runtime_v6.py",
    "training/arbitrary_plane_coarse_proposal_v6.py",
    "training/arbitrary_plane_hybrid_posterior_v6.py",
    "training/arbitrary_plane_recurrent_model_v6.py",
    "training/arbitrary_plane_joint_model_v6.py",
    "training/arbitrary_plane_retrieval_loss_v6.py",
    "training/arbitrary_plane_joint_loss_v6.py",
    "training/arbitrary_plane_staged_trainer_v6.py",
)
_SUPPORTED_RUNTIME_DTYPES = {
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
    "torch.float64",
}
_TRAINING_RUN_BINDING_KEYS = (
    "run_manifest_receipt_sha256",
    "atlas_binding_receipt_sha256",
    "training_data_manifest_receipt_sha256",
)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _hash_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and not (set(value) - set("0123456789abcdef"))


def _object_receipt(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _object_receipt(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_object_receipt(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_receipts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _state_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    return _state_mapping_sha256(module.state_dict())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _phase(config: Mapping[str, object], step: int) -> str:
    proposal_end = int(config["proposal_only_steps"])
    rerank_end = proposal_end + int(config["pose_rerank_steps"])
    return "proposal-only" if step < proposal_end else "pose-rerank" if step < rerank_end else "joint"


def _same_device(left: torch.device | str, right: torch.device | str) -> bool:
    left = torch.device(left)
    right = torch.device(right)
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return left == right
    left_index = torch.cuda.current_device() if left.index is None else left.index
    right_index = torch.cuda.current_device() if right.index is None else right.index
    return left_index == right_index


def _validated_frozen_row_source(
    source: object, expected_manifest_receipt_sha256: str, batch_size: int
) -> dict[str, object]:
    keys = {
        "schema_version",
        "training_data_manifest_receipt_sha256",
        "cache_manifest_receipt_sha256",
        "generator_binding_receipt_sha256",
        "generation_lineage_sha256",
        "row_indices",
        "training_row_ids",
        "training_row_receipts_sha256",
        "selection_receipt_sha256",
    }
    if not isinstance(source, Mapping) or set(source) != keys:
        raise ValueError("v6 training requires one authenticated frozen-row selection")
    receipt_payload = {
        key: source[key] for key in keys if key != "selection_receipt_sha256"
    }
    row_indices = source["row_indices"]
    row_ids = source["training_row_ids"]
    row_receipts = source["training_row_receipts_sha256"]
    if (
        source["schema_version"] != FROZEN_ROWS_V6_SCHEMA
        or source["training_data_manifest_receipt_sha256"]
        != expected_manifest_receipt_sha256
        or source["cache_manifest_receipt_sha256"]
        != expected_manifest_receipt_sha256
        or any(
            not _is_sha256(source[key])
            for key in (
                "generator_binding_receipt_sha256",
                "generation_lineage_sha256",
                "selection_receipt_sha256",
            )
        )
        or source["selection_receipt_sha256"] != _hash_json(receipt_payload)
        or not isinstance(row_indices, list)
        or not isinstance(row_ids, list)
        or not isinstance(row_receipts, list)
        or len(row_indices) != batch_size
        or len(row_ids) != batch_size
        or len(row_receipts) != batch_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in row_indices
        )
        or any(not isinstance(value, str) or not value for value in row_ids)
        or any(not _is_sha256(value) for value in row_receipts)
    ):
        raise ValueError("v6 frozen-row selection receipt or run binding is invalid")
    return _plain(source)


def _validated_row_receipts(
    value: object,
    frozen_source: Mapping[str, object],
    batch_size: int,
) -> tuple[list[dict[str, object]], str]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError("v6 training requires one exact receipt record per row")
    receipts = _plain(value)
    for index, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("training_row_id")
            != frozen_source["training_row_ids"][index]
            or receipt.get("training_row_receipt_sha256")
            != frozen_source["training_row_receipts_sha256"][index]
            or any(not isinstance(key, str) or not key for key in receipt)
            or any(not isinstance(item, str) or not item for item in receipt.values())
            or any(
                not _is_sha256(item)
                for key, item in receipt.items()
                if key.endswith("sha256")
            )
        ):
            raise ValueError("v6 row receipts differ from the frozen-row selection")
    receipt_sha256 = _hash_json(
        {
            "schema_version": ROW_RECEIPTS_V6_SCHEMA,
            "row_receipts": receipts,
        }
    )
    return receipts, receipt_sha256


def _validated_config(training_config: Mapping[str, object]) -> dict[str, object]:
    config = dict(training_config)
    if set(config) != _CONFIG_KEYS:
        raise ValueError(f"v6 training config keys must be exactly {sorted(_CONFIG_KEYS)}")
    integer_keys = (
        "seed",
        "proposal_only_steps",
        "pose_rerank_steps",
        "proposal_top_m",
        "top_k",
        "refinement_steps",
        "joint_pose_only_steps",
    )
    if any(not isinstance(config[key], int) or isinstance(config[key], bool) for key in integer_keys):
        raise ValueError("v6 schedule and selection counts must be integers")
    if config["proposal_only_steps"] < 1 or config["pose_rerank_steps"] < 1:
        raise ValueError("v6 requires nonempty proposal-only and pose-rerank stages")
    if not 1 <= config["top_k"] <= config["proposal_top_m"] <= FULL_CATALOGUE_CELL_COUNT_V6:
        raise ValueError("v6 proposal/top-K counts are invalid")
    if config["refinement_steps"] < 1 or not 0 <= config["joint_pose_only_steps"] <= config["refinement_steps"] + 1:
        raise ValueError("joint pose-only prefix must fit the recurrent schedule")
    shape = config["retrieval_shape_h_w"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 4 for size in shape
    ):
        raise ValueError("retrieval shape must contain two integer sizes of at least four")
    positive = ("learning_rate", "amp_initial_scale", "gradient_clip_norm")
    nonnegative = ("weight_decay", "proposal_loss_weight", "rerank_loss_weight")
    if any(not np.isfinite(float(config[key])) or float(config[key]) <= 0.0 for key in positive) or any(
        not np.isfinite(float(config[key])) or float(config[key]) < 0.0 for key in nonnegative
    ):
        raise ValueError("v6 optimizer and loss scalars are invalid")
    if not isinstance(config["amp"], bool):
        raise ValueError("amp must be Boolean")
    config["retrieval_shape_h_w"] = tuple(shape)
    return config


def initialize_staged_trainer_v6(
    catalogue_runtime_v6: CompleteCatalogueRuntimeV6,
    atlas_channels: int,
    model_kwargs: Mapping[str, object],
    training_config: Mapping[str, object],
    *,
    training_run_binding: Mapping[str, object],
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    """Construct the v6 model from fresh random parameters; no weight input exists."""
    verify_complete_catalogue_runtime_v6(catalogue_runtime_v6)
    if catalogue_runtime_v6.cell_count != FULL_CATALOGUE_CELL_COUNT_V6:
        raise ValueError("v6 training requires the complete 98,304-cell catalogue")
    kwargs = dict(model_kwargs)
    if "catalogue_runtime_v6" in kwargs or "pose_only_steps" in kwargs:
        raise ValueError("runtime and fixed pose-only prefix come from the v6 trainer")
    config = _validated_config(training_config)
    render_budget = kwargs.get("cascade_max_rendered_cells_per_sample", 64)
    if (
        not isinstance(render_budget, int)
        or isinstance(render_budget, bool)
        or render_budget < 1
        or int(config["proposal_top_m"]) > render_budget
    ):
        raise ValueError("proposal_top_m must not exceed the effective v6 cascade render budget")
    if not isinstance(training_run_binding, Mapping) or any(
        not _is_sha256(training_run_binding.get(key))
        for key in _TRAINING_RUN_BINDING_KEYS
    ):
        raise ValueError("v6 training requires receipt-bound run, atlas, and training-data manifests")
    run_binding = _plain(training_run_binding)
    seed = int(config["seed"])
    target_device = torch.device(device)
    runtime_binding = catalogue_runtime_v6.binding
    if not _same_device(runtime_binding["device"], target_device):
        raise ValueError("v6 catalogue runtime device must exactly match the trainer device")
    if runtime_binding.get("dtype") not in _SUPPORTED_RUNTIME_DTYPES:
        raise ValueError("v6 catalogue runtime dtype is not a supported floating dtype")
    _set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ArbitraryPlaneJointModelV6(
        catalogue_runtime_v6,
        atlas_channels=int(atlas_channels),
        pose_only_steps=int(config["joint_pose_only_steps"]),
        **kwargs,
    ).to(target_device)
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
    manifest_payload = {
        "schema_version": STAGED_TRAINER_V6_MANIFEST_SCHEMA,
        "trainer_schema_version": STAGED_TRAINER_V6_SCHEMA,
        "source_sha256": _source_receipts(),
        "model_kwargs": _plain(kwargs),
        "atlas_channels": int(atlas_channels),
        "training_config": _plain(config),
        "catalogue_binding": _plain(catalogue_runtime_v6.binding),
        "catalogue_cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
        "catalogue_scope": "complete_98304_cell_catalogue",
        "cascade_max_rendered_cells_per_sample": render_budget,
        "initialization": "fresh_random_only",
        "input_modes": list(INPUT_MODES_V6),
        "provenance_keys": list(PROVENANCE_KEYS_V6),
        "probabilities_calibrated": False,
        "uncertainty_status": "raw_uncalibrated",
        "training_run_binding": run_binding,
        "checkpoint_scope": "trainer_state_subordinate_to_receipt_bound_v6_runner",
        "release_qualifying": False,
        "atlas_bytes_verified_by_trainer": False,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    manifest = {**manifest_payload, "receipt_sha256": _hash_json(manifest_payload)}
    initial_hash = _state_sha256(model)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "catalogue_runtime_v6": catalogue_runtime_v6,
        "device": str(target_device),
        "global_step": 0,
        "manifest": manifest,
        "initialization_receipt": {
            "algorithm": "fresh-pytorch-random-initialization",
            "seed": seed,
            "model_state_sha256": initial_hash,
            "receipt_sha256": _hash_json(
                {
                    "algorithm": "fresh-pytorch-random-initialization",
                    "seed": seed,
                    "model_state_sha256": initial_hash,
                    "manifest_receipt_sha256": manifest["receipt_sha256"],
                }
            ),
        },
        "provenance_records": [],
        "training_step_ledger": [],
    }


def _validated_batch(
    state: Mapping[str, object], batch: Mapping[str, object]
) -> tuple[
    torch.Tensor,
    list[dict[str, str]],
    list[str],
    dict[str, object],
    str,
]:
    image = torch.as_tensor(batch["image"])
    outline = torch.as_tensor(batch["outline"], device=image.device)
    available = torch.as_tensor(batch["outline_available"], device=image.device)
    if image.ndim != 4 or image.shape[1] != 1 or outline.shape != image.shape:
        raise ValueError("v6 image and outline must share shape (B,1,H,W)")
    if not _same_device(image.device, state["device"]):
        raise ValueError("v6 model inputs, catalogue runtime, and trainer must share one device")
    if available.shape == (image.shape[0], 1):
        available = available[:, 0]
    if available.shape != (image.shape[0],) or not bool(((available == 0) | (available == 1)).all()):
        raise ValueError("outline availability must be binary with shape (B,)")
    modes = [str(value) for value in batch["input_mode"]]
    if len(modes) != image.shape[0] or any(value not in INPUT_MODES_V6 for value in modes):
        raise ValueError("each v6 row must declare raw, black-exterior, or imperfect-mask input")
    frozen_source = _validated_frozen_row_source(
        batch.get("frozen_row_source"),
        state["manifest"]["training_run_binding"][
            "training_data_manifest_receipt_sha256"
        ],
        image.shape[0],
    )
    provenance = []
    for value in batch["provenance"]:
        allowed = set(PROVENANCE_KEYS_V6) | set(OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6)
        if not isinstance(value, Mapping) or not set(PROVENANCE_KEYS_V6).issubset(value) or set(value) - allowed or any(
            not isinstance(value[key], str) or not value[key] for key in PROVENANCE_KEYS_V6
        ) or any(
            not isinstance(value[key], str) or not value[key]
            for key in OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6
            if key in value
        ) or any(
            not _is_sha256(value[key])
            for key in OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6
            if key.endswith("sha256") and key in value
        ):
            raise ValueError("each row needs exact specimen/animal/experiment/section/synthetic-animal provenance")
        provenance.append({key: value[key] for key in (*PROVENANCE_KEYS_V6, *OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6) if key in value})
    if len(provenance) != image.shape[0]:
        raise ValueError("row provenance count must equal batch size")
    if (
        [item.get("training_row_id") for item in provenance]
        != frozen_source["training_row_ids"]
        or [item.get("training_row_receipt_sha256") for item in provenance]
        != frozen_source["training_row_receipts_sha256"]
    ):
        raise ValueError("v6 row provenance differs from the frozen-row selection")
    _, row_receipts_sha256 = _validated_row_receipts(
        batch.get("row_receipts"), frozen_source, image.shape[0]
    )
    runtime = state["catalogue_runtime_v6"]
    catalogue_batch = batch["catalogue_batch"]
    verify_bound_complete_catalogue_batch_v6(catalogue_batch, expected_runtime=runtime)
    if catalogue_batch.batch_size != image.shape[0] or runtime.cell_count != FULL_CATALOGUE_CELL_COUNT_V6:
        raise ValueError("batch must bind the complete 98,304-cell runtime")
    truth = torch.as_tensor(batch["truth_catalogue_index"], device=image.device)
    if truth.dtype == torch.bool or torch.is_floating_point(truth) or torch.is_complex(truth):
        raise ValueError("truth catalogue indices must be integers")
    truth = truth.to(torch.long)
    if truth.shape != (image.shape[0],) or bool(((truth < 0) | (truth >= FULL_CATALOGUE_CELL_COUNT_V6)).any()):
        raise ValueError("truth catalogue indices must address the complete catalogue")
    return truth, provenance, modes, frozen_source, row_receipts_sha256


def _model_arguments(state: Mapping[str, object], batch: Mapping[str, object]) -> tuple[object, ...]:
    config = state["manifest"]["training_config"]
    return (
        batch["image"],
        batch["outline"],
        batch["outline_available"],
        batch["atlas_volume"],
        batch["catalogue_batch"],
        tuple(batch["output_shape_h_w"]),
        tuple(config["retrieval_shape_h_w"]),
        batch["origin_ap_dv_ml_um"],
        batch["voxel_size_ap_dv_ml_um"],
        batch["axial_offsets_um"],
        batch["axial_weights"],
    )


def _proposal_loss(output: Mapping[str, object], truth: torch.Tensor, weight: torch.Tensor | None) -> dict[str, object]:
    if output.get("atlas_render_count") != 0 or output.get("cascade_boundary") != "full_catalogue_proposal_only":
        raise RuntimeError("proposal-only phase must perform zero atlas renders")
    if output.get("probabilities_calibrated") is not False:
        raise RuntimeError("v6 proposal uncertainty must remain explicitly uncalibrated")
    result = full_catalogue_proposal_nll_v6(
        output["raw_full_catalogue_cell_log_probability"],
        truth,
        expected_catalogue_cell_count=FULL_CATALOGUE_CELL_COUNT_V6,
        supervision_weight=weight,
    )
    return {"total": result["loss"], "full_catalogue_proposal_nll": result["loss"]}


def _membership(index: torch.Tensor, valid: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    match = index.eq(truth[:, None]) & valid
    hit = match.any(dim=1)
    return hit, torch.where(hit, match.to(torch.long).argmax(dim=1), torch.full_like(truth, -1))


def _verify_joint_schedule(output: Mapping[str, object], config: Mapping[str, object]) -> None:
    refined = output.get("refined_output")
    ready = output.get("refinement_ready_mask")
    source = output.get("refinement_source_batch_index")
    if refined is None:
        if (
            not isinstance(ready, torch.Tensor)
            or ready.dtype != torch.bool
            or bool(ready.any())
            or not isinstance(source, torch.Tensor)
            or source.numel() != 0
            or output.get("refinement_performed") is not False
        ):
            raise RuntimeError("an empty joint refinement must be an exact all-abstained R=0 batch")
        return
    if not isinstance(refined, Mapping):
        raise RuntimeError("joint refined output must be a mapping or an exact all-abstained R=0")
    audit = refined.get("deformation_gating_audit")
    active = refined.get("deformation_active_sequence")
    prefix = int(config["joint_pose_only_steps"])
    iterations = int(config["refinement_steps"]) + 1
    expected = torch.arange(iterations, device=active.device) >= prefix if isinstance(active, torch.Tensor) else None
    if (
        not isinstance(audit, Mapping)
        or audit.get("pose_only_steps") != prefix
        or not isinstance(active, torch.Tensor)
        or active.dtype != torch.bool
        or active.shape != (iterations,)
        or not torch.equal(active, expected)
    ):
        raise RuntimeError("joint deformation must start only after the fixed pose-only recurrent prefix")


def _pose_rerank_loss(output: Mapping[str, object], truth: torch.Tensor, weight: torch.Tensor | None, config: Mapping[str, object]) -> dict[str, object]:
    proposal_log = output["raw_full_catalogue_proposal_log_probability"]
    if proposal_log.shape != (truth.shape[0], FULL_CATALOGUE_CELL_COUNT_V6):
        raise RuntimeError("pose-rerank must preserve the full proposal q")
    if output.get("probability_status") != "raw_uncalibrated" or output.get("training_truth_leakage_into_honest_hybrid") is not False:
        raise RuntimeError("pose-rerank uncertainty or honest-selection semantics changed")
    honest = output["honest_hybrid_posterior"]
    if honest.get("selection_scope") != "honest_proposal_plus_adaptive_closure_no_truth":
        raise RuntimeError("pose-rerank must use honest adaptive selection")
    honest_index = honest["selected_catalogue_index"]
    honest_valid = honest["selected_valid_mask"]
    honest_hit, honest_position = _membership(honest_index, honest_valid, truth)
    forced = output["training_truth_forced_mask"]
    if not torch.equal(forced, ~honest_hit):
        raise RuntimeError("teacher union is permitted only on honest misses")
    training_index = output["training_selected_catalogue_index"]
    training_valid = output["training_selected_valid_mask"]
    teacher = output["training_teacher_forced_hybrid_posterior"]
    teacher_hit, teacher_position = _membership(training_index, training_valid, truth)
    honest_count = honest_valid.sum(dim=1)
    training_count = training_valid.sum(dim=1)
    if not torch.equal(training_count, honest_count + forced.to(torch.long)) or not bool(teacher_hit.all()):
        raise RuntimeError("teacher selection must be the honest set union truth on misses")
    for row in range(truth.shape[0]):
        count = int(honest_count[row].item())
        if not torch.equal(training_index[row, :count], honest_index[row, honest_valid[row]]):
            raise RuntimeError("teacher selection must preserve the honest adaptive selection")
        if bool(forced[row]) and training_index[row, count] != truth[row]:
            raise RuntimeError("teacher union may append only the missing truth")
    proposal = full_catalogue_proposal_nll_v6(
        proposal_log,
        truth,
        expected_catalogue_cell_count=FULL_CATALOGUE_CELL_COUNT_V6,
        supervision_weight=weight,
    )["loss"]
    honest_result = selected_exact_rerank_nll_v6(
        honest["selected_conditional_log_probability"],
        honest_valid,
        honest_position,
        honest_hit,
        supervision_weight=weight,
    )
    teacher_result = selected_exact_rerank_nll_v6(
        teacher["selected_conditional_log_probability"],
        training_valid,
        teacher_position,
        forced,
        supervision_weight=weight,
    )
    row_weight = torch.ones_like(proposal_log[:, 0]) if weight is None else torch.as_tensor(weight, device=proposal_log.device, dtype=proposal_log.dtype)
    rerank_per_row = honest_result["per_row_nll"] + teacher_result["per_row_nll"]
    denominator = row_weight.sum()
    rerank = (rerank_per_row * row_weight).sum() / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    total = float(config["proposal_loss_weight"]) * proposal + float(config["rerank_loss_weight"]) * rerank
    return {
        "total": total,
        "full_catalogue_proposal_nll": proposal,
        "selected_finite_render_conditional_rerank_nll": rerank,
        "teacher_forced_row_count": int(forced.sum().item()),
    }


def train_staged_step_v6(state: dict[str, object], batch: Mapping[str, object]) -> dict[str, object]:
    """Apply one update at the immutable proposal, rerank, or joint phase boundary."""
    truth, provenance, modes, frozen_source, row_receipts_sha256 = (
        _validated_batch(state, batch)
    )
    config = state["manifest"]["training_config"]
    step = int(state["global_step"])
    phase = _phase(config, step)
    model = state["model"]
    if model.pose_model.cascade_max_rendered_cells_per_sample != state["manifest"]["cascade_max_rendered_cells_per_sample"]:
        raise RuntimeError("live cascade render budget differs from the frozen v6 manifest")
    model.train()
    state["optimizer"].zero_grad(set_to_none=True)
    use_amp = bool(config["amp"] and torch.device(state["device"]).type == "cuda")
    retrieval_weight = batch.get("retrieval_supervision_weight")
    with torch.amp.autocast("cuda", enabled=use_amp):
        if phase == "proposal-only":
            output = model.pose_model.forward_proposal_only(
                batch["image"], batch["outline"], batch["outline_available"],
                batch["catalogue_batch"], tuple(config["retrieval_shape_h_w"]),
            )
            losses = _proposal_loss(output, truth, retrieval_weight)
        elif phase == "pose-rerank":
            args = _model_arguments(state, batch)
            output = model.pose_model.forward_proposed(
                args[0], args[1], args[2], args[3], args[4], args[6], args[7], args[8], args[9], args[10],
                proposal_top_m=int(config["proposal_top_m"]),
                top_k=int(config["top_k"]),
                training_truth_catalogue_index=truth,
            )
            losses = _pose_rerank_loss(output, truth, retrieval_weight, config)
        else:
            output = model(
                *_model_arguments(state, batch),
                proposal_top_m=int(config["proposal_top_m"]),
                top_k=int(config["top_k"]),
                refinement_steps=int(config["refinement_steps"]),
                training_truth_catalogue_index=truth,
                dense_deformation_supervision_weight=batch.get("dense_deformation_supervision_weight"),
            )
            _verify_joint_schedule(output, config)
            losses = arbitrary_plane_joint_loss_v6(
                output,
                batch["truth_state"],
                truth,
                batch["truth_stationary_velocity_yx_px"],
                batch["truth_pullback_map_yx_px"],
                batch["deformation_weight"],
                state["catalogue_runtime_v6"].binding["support_origin_ap_dv_ml_um"],
                expected_catalogue_cell_count=FULL_CATALOGUE_CELL_COUNT_V6,
                retrieval_supervision_weight=retrieval_weight,
                pose_supervision_weight=batch.get("pose_supervision_weight"),
                dense_deformation_supervision_weight=batch.get("dense_deformation_supervision_weight"),
                proposal_loss_weight=float(config["proposal_loss_weight"]),
                rerank_loss_weight=float(config["rerank_loss_weight"]),
            )
            if losses.get("probability_status") != "raw_uncalibrated" or losses.get("probabilities_calibrated") is not False:
                raise RuntimeError("joint loss must report raw uncalibrated uncertainty")
    objective = losses["total"]
    if not isinstance(objective, torch.Tensor) or objective.ndim != 0 or not bool(torch.isfinite(objective)):
        raise FloatingPointError("v6 training objective must be one finite scalar tensor")
    state["scaler"].scale(objective).backward()
    state["scaler"].unscale_(state["optimizer"])
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["gradient_clip_norm"]), error_if_nonfinite=True
    )
    state["scaler"].step(state["optimizer"])
    state["scaler"].update()
    record_start = len(state["provenance_records"])
    state["provenance_records"].extend(_plain(provenance))
    ready = output.get("refinement_ready_mask") if phase == "joint" else None
    ready_count = int(ready.sum().item()) if isinstance(ready, torch.Tensor) else None
    abstained_count = truth.shape[0] - ready_count if ready_count is not None else None
    trainer_output_payload = {
        "schema_version": STAGED_TRAINER_V6_SCHEMA,
        "step": step,
        "phase": phase,
        "objective": float(objective.detach()),
        "preclip_gradient_norm": float(gradient_norm),
        "optimizer_step_applied": True,
        "catalogue_cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
        "probabilities_calibrated": False,
        "probability_status": "raw_uncalibrated",
        "refinement_ready_row_count": ready_count,
        "refinement_abstained_row_count": abstained_count,
        "losses": {
            key: float(value.detach()) if isinstance(value, torch.Tensor) and value.ndim == 0 else value
            for key, value in losses.items()
            if isinstance(value, (torch.Tensor, int, float, bool, str))
        },
    }
    trainer_output_receipt_sha256 = _hash_json(trainer_output_payload)
    ledger_payload = {
        "step": step,
        "phase": phase,
        "provenance_record_indices": list(range(record_start, record_start + len(provenance))),
        "input_mode": modes,
        "frozen_row_selection": frozen_source,
        "row_receipts_sha256": row_receipts_sha256,
        "trainer_output_receipt_sha256": trainer_output_receipt_sha256,
        "catalogue_cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
        "probability_status": "raw_uncalibrated",
        "refinement_ready_row_count": ready_count,
        "refinement_abstained_row_count": abstained_count,
    }
    previous = state["training_step_ledger"][-1]["receipt_sha256"] if state["training_step_ledger"] else state["manifest"]["receipt_sha256"]
    ledger_payload["previous_receipt_sha256"] = previous
    state["training_step_ledger"].append({**ledger_payload, "receipt_sha256": _hash_json(ledger_payload)})
    state["global_step"] = step + 1
    return {
        **trainer_output_payload,
        "receipt_sha256": trainer_output_receipt_sha256,
    }


def make_staged_checkpoint_v6(state: Mapping[str, object]) -> dict[str, object]:
    model_state = {name: value.detach().cpu().clone() for name, value in state["model"].state_dict().items()}
    payload = {
        "schema_version": STAGED_TRAINER_V6_CHECKPOINT_SCHEMA,
        "manifest": _plain(state["manifest"]),
        "initialization_receipt": _plain(state["initialization_receipt"]),
        "global_step": int(state["global_step"]),
        "model_state": model_state,
        "model_state_sha256": _state_sha256(state["model"]),
        "optimizer_state": copy.deepcopy(state["optimizer"].state_dict()),
        "scaler_state": copy.deepcopy(state["scaler"].state_dict()),
        "rng_state": copy.deepcopy(_rng_state()),
        "provenance_records": _plain(state["provenance_records"]),
        "training_step_ledger": _plain(state["training_step_ledger"]),
        "seen_specimen_ids": sorted({item["specimen_id"] for item in state["provenance_records"]}),
        "seen_animal_ids": sorted({item["animal_id"] for item in state["provenance_records"]}),
        "seen_experiment_ids": sorted({item["experiment_id"] for item in state["provenance_records"]}),
        "seen_section_ids": sorted({item["section_id"] for item in state["provenance_records"]}),
        "seen_synthetic_animal_ids": sorted({item["synthetic_animal_id"] for item in state["provenance_records"]}),
        "learned_dependencies": {"model_weights": [], "features": [], "pseudolabels": []},
        "probabilities_calibrated": False,
        "uncertainty_status": "raw_uncalibrated",
    }
    payload["optimizer_state_sha256"] = _hash_json(_object_receipt(payload["optimizer_state"]))
    payload["scaler_state_sha256"] = _hash_json(_object_receipt(payload["scaler_state"]))
    payload["rng_state_sha256"] = _hash_json(_object_receipt(payload["rng_state"]))
    receipt_payload = {
        key: value for key, value in payload.items()
        if key not in ("model_state", "optimizer_state", "scaler_state", "rng_state")
    }
    payload["receipt_sha256"] = _hash_json(receipt_payload)
    verify_staged_checkpoint_v6(payload, verify_sources=False)
    return payload


def verify_staged_checkpoint_v6(checkpoint: Mapping[str, object], *, verify_sources: bool = True) -> bool:
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != STAGED_TRAINER_V6_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a v6 staged-trainer checkpoint")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != STAGED_TRAINER_V6_MANIFEST_SCHEMA:
        raise ValueError("v6 checkpoint manifest is missing")
    manifest_payload = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
    if manifest.get("receipt_sha256") != _hash_json(manifest_payload):
        raise ValueError("v6 manifest receipt is invalid")
    if verify_sources and manifest.get("source_sha256") != _source_receipts():
        raise ValueError("v6 checkpoint source receipts differ from running code")
    if (
        manifest.get("trainer_schema_version") != STAGED_TRAINER_V6_SCHEMA
        or manifest.get("catalogue_cell_count") != FULL_CATALOGUE_CELL_COUNT_V6
        or manifest.get("catalogue_scope") != "complete_98304_cell_catalogue"
        or manifest.get("cascade_max_rendered_cells_per_sample")
        != manifest.get("model_kwargs", {}).get("cascade_max_rendered_cells_per_sample", 64)
        or manifest["training_config"]["proposal_top_m"]
        > manifest.get("cascade_max_rendered_cells_per_sample", 0)
        or manifest.get("initialization") != "fresh_random_only"
        or manifest.get("input_modes") != list(INPUT_MODES_V6)
        or manifest.get("provenance_keys") != list(PROVENANCE_KEYS_V6)
        or manifest.get("probabilities_calibrated") is not False
        or manifest.get("uncertainty_status") != "raw_uncalibrated"
        or manifest.get("checkpoint_scope") != "trainer_state_subordinate_to_receipt_bound_v6_runner"
        or manifest.get("release_qualifying") is not False
        or manifest.get("atlas_bytes_verified_by_trainer") is not False
        or not isinstance(manifest.get("catalogue_binding"), Mapping)
        or manifest["catalogue_binding"].get("cell_count") != FULL_CATALOGUE_CELL_COUNT_V6
        or manifest["catalogue_binding"].get("dtype") not in _SUPPORTED_RUNTIME_DTYPES
    ):
        raise ValueError("v6 checkpoint catalogue or uncertainty contract differs")
    run_binding = manifest.get("training_run_binding")
    if not isinstance(run_binding, Mapping) or any(
        not _is_sha256(run_binding.get(key)) for key in _TRAINING_RUN_BINDING_KEYS
    ):
        raise ValueError("v6 checkpoint lacks its receipt-bound training run")
    _validated_config(manifest.get("training_config", {}))
    if manifest.get("prior_model_weight_dependencies") != [] or manifest.get("prior_feature_dependencies") != [] or manifest.get("prior_pseudolabel_dependencies") != []:
        raise ValueError("v6 manifest contains forbidden learned dependencies")
    initialization = checkpoint.get("initialization_receipt")
    initialization_payload = {
        "algorithm": "fresh-pytorch-random-initialization",
        "seed": manifest["training_config"]["seed"],
        "model_state_sha256": initialization.get("model_state_sha256") if isinstance(initialization, Mapping) else None,
        "manifest_receipt_sha256": manifest["receipt_sha256"],
    }
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("algorithm") != initialization_payload["algorithm"]
        or initialization.get("seed") != initialization_payload["seed"]
        or initialization.get("receipt_sha256") != _hash_json(initialization_payload)
    ):
        raise ValueError("v6 fresh initialization receipt is invalid")
    records = checkpoint.get("provenance_records")
    ledger = checkpoint.get("training_step_ledger")
    if not isinstance(records, list) or not isinstance(ledger, list) or len(ledger) != checkpoint.get("global_step"):
        raise ValueError("v6 provenance or step ledger is incomplete")
    for record in records:
        allowed = set(PROVENANCE_KEYS_V6) | set(OPTIONAL_PROVENANCE_RECEIPT_KEYS_V6)
        if not isinstance(record, Mapping) or not set(PROVENANCE_KEYS_V6).issubset(record) or set(record) - allowed or any(not isinstance(record[key], str) or not record[key] for key in record) or any(not _is_sha256(record[key]) for key in record if key.endswith("sha256")):
            raise ValueError("v6 checkpoint provenance is not exact")
    previous = manifest["receipt_sha256"]
    next_record_index = 0
    for step, entry in enumerate(ledger):
        entry_payload = {key: value for key, value in entry.items() if key != "receipt_sha256"}
        indices = entry.get("provenance_record_indices")
        phase = _phase(manifest["training_config"], step)
        ready_count = entry.get("refinement_ready_row_count")
        abstained_count = entry.get("refinement_abstained_row_count")
        try:
            frozen_source = _validated_frozen_row_source(
                entry.get("frozen_row_selection"),
                run_binding["training_data_manifest_receipt_sha256"],
                len(indices) if isinstance(indices, list) else -1,
            )
        except ValueError:
            raise ValueError("v6 training step ledger failed verification") from None
        if (
            entry.get("step") != step
            or entry.get("phase") != phase
            or entry.get("previous_receipt_sha256") != previous
            or entry.get("receipt_sha256") != _hash_json(entry_payload)
            or not isinstance(indices, list)
            or not indices
            or indices != list(range(next_record_index, next_record_index + len(indices)))
            or len(entry.get("input_mode", [])) != len(indices)
            or any(mode not in INPUT_MODES_V6 for mode in entry.get("input_mode", []))
            or entry.get("catalogue_cell_count") != FULL_CATALOGUE_CELL_COUNT_V6
            or entry.get("probability_status") != "raw_uncalibrated"
            or not _is_sha256(entry.get("row_receipts_sha256"))
            or not _is_sha256(entry.get("trainer_output_receipt_sha256"))
            or [records[index].get("training_row_id") for index in indices]
            != frozen_source["training_row_ids"]
            or [
                records[index].get("training_row_receipt_sha256")
                for index in indices
            ]
            != frozen_source["training_row_receipts_sha256"]
            or (
                phase == "joint"
                and (
                    not isinstance(ready_count, int)
                    or isinstance(ready_count, bool)
                    or not isinstance(abstained_count, int)
                    or isinstance(abstained_count, bool)
                    or ready_count < 0
                    or abstained_count < 0
                    or ready_count + abstained_count != len(indices)
                )
            )
            or (phase != "joint" and (ready_count is not None or abstained_count is not None))
        ):
            raise ValueError("v6 training step ledger failed verification")
        previous = entry["receipt_sha256"]
        next_record_index += len(indices)
    if next_record_index != len(records):
        raise ValueError("v6 step ledger does not cover each provenance record exactly once")
    expected_seen = {
        "seen_specimen_ids": sorted({item["specimen_id"] for item in records}),
        "seen_animal_ids": sorted({item["animal_id"] for item in records}),
        "seen_experiment_ids": sorted({item["experiment_id"] for item in records}),
        "seen_section_ids": sorted({item["section_id"] for item in records}),
        "seen_synthetic_animal_ids": sorted({item["synthetic_animal_id"] for item in records}),
    }
    if any(checkpoint.get(key) != value for key, value in expected_seen.items()):
        raise ValueError("v6 seen provenance summaries differ")
    if checkpoint.get("learned_dependencies") != {"model_weights": [], "features": [], "pseudolabels": []}:
        raise ValueError("v6 checkpoint contains forbidden learned dependencies")
    if checkpoint.get("probabilities_calibrated") is not False or checkpoint.get("uncertainty_status") != "raw_uncalibrated":
        raise ValueError("v6 checkpoint uncertainty contract differs")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state or not all(isinstance(value, torch.Tensor) for value in model_state.values()):
        raise ValueError("v6 checkpoint model state is missing")
    if checkpoint.get("model_state_sha256") != _state_mapping_sha256(model_state):
        raise ValueError("v6 checkpoint model state receipt is invalid")
    if (
        checkpoint.get("optimizer_state_sha256") != _hash_json(_object_receipt(checkpoint.get("optimizer_state")))
        or checkpoint.get("scaler_state_sha256") != _hash_json(_object_receipt(checkpoint.get("scaler_state")))
        or checkpoint.get("rng_state_sha256") != _hash_json(_object_receipt(checkpoint.get("rng_state")))
    ):
        raise ValueError("v6 optimizer, scaler, or RNG receipt is invalid")
    receipt_payload = {
        key: value for key, value in checkpoint.items()
        if key not in ("model_state", "optimizer_state", "scaler_state", "rng_state", "receipt_sha256")
    }
    if checkpoint.get("receipt_sha256") != _hash_json(receipt_payload):
        raise ValueError("v6 checkpoint receipt is invalid")
    return True


def save_staged_checkpoint_v6(state: Mapping[str, object], path: str | Path) -> Path:
    target = Path(path).resolve()
    if os.path.splitdrive(str(target))[0].upper() != "I:":
        raise ValueError("v6 checkpoints may be written only on I:")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(make_staged_checkpoint_v6(state), temporary)
    os.replace(temporary, target)
    return target


def load_staged_checkpoint_v6(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    if os.path.splitdrive(str(source))[0].upper() != "I:":
        raise ValueError("v6 checkpoints may be loaded only from I:")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    verify_staged_checkpoint_v6(checkpoint)
    return checkpoint


def restore_staged_trainer_v6(
    checkpoint: Mapping[str, object],
    catalogue_runtime_v6: CompleteCatalogueRuntimeV6,
    *,
    training_run_binding: Mapping[str, object],
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    """Resume only an authenticated v6 checkpoint through its fresh-init lineage."""
    verify_staged_checkpoint_v6(checkpoint)
    verify_complete_catalogue_runtime_v6(catalogue_runtime_v6)
    if _plain(catalogue_runtime_v6.binding) != checkpoint["manifest"]["catalogue_binding"]:
        raise ValueError("resume catalogue binding differs from the v6 checkpoint")
    manifest = checkpoint["manifest"]
    if _plain(training_run_binding) != manifest["training_run_binding"]:
        raise ValueError("resume training-run binding differs from the v6 checkpoint")
    saved_rng = _rng_state()
    try:
        state = initialize_staged_trainer_v6(
            catalogue_runtime_v6,
            manifest["atlas_channels"],
            manifest["model_kwargs"],
            manifest["training_config"],
            training_run_binding=training_run_binding,
            device=device,
        )
        if state["initialization_receipt"] != checkpoint["initialization_receipt"]:
            raise ValueError("v6 checkpoint does not replay its fresh initialization")
        state["model"].load_state_dict(checkpoint["model_state"], strict=True)
        state["optimizer"].load_state_dict(checkpoint["optimizer_state"])
        state["scaler"].load_state_dict(checkpoint["scaler_state"])
        state["global_step"] = int(checkpoint["global_step"])
        state["provenance_records"] = list(checkpoint["provenance_records"])
        state["training_step_ledger"] = list(checkpoint["training_step_ledger"])
    except Exception:
        _restore_rng_state(saved_rng)
        raise
    _restore_rng_state(checkpoint["rng_state"])
    return state


__all__ = [
    "BLACK_EXTERIOR_INPUT_MODE_V6",
    "FROZEN_ROWS_V6_SCHEMA",
    "FULL_CATALOGUE_CELL_COUNT_V6",
    "IMPERFECT_MASK_INPUT_MODE_V6",
    "INPUT_MODES_V6",
    "RAW_INPUT_MODE_V6",
    "STAGED_TRAINER_V6_CHECKPOINT_SCHEMA",
    "STAGED_TRAINER_V6_MANIFEST_SCHEMA",
    "STAGED_TRAINER_V6_SCHEMA",
    "initialize_staged_trainer_v6",
    "load_staged_checkpoint_v6",
    "make_staged_checkpoint_v6",
    "restore_staged_trainer_v6",
    "save_staged_checkpoint_v6",
    "train_staged_step_v6",
    "verify_staged_checkpoint_v6",
]
