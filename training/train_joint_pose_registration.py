"""Train the joint AtlasPose, candidate-review, and dense-registration model.

The trainer deliberately keeps atlas rendering outside the model.  Wrong atlas
planes supervise pose correction and candidate ranking only; exact dense-flow
targets are consumed exclusively by the known true atlas plane.

Checkpoint resume restores model, optimizer, scheduler, RNG, and data-stream
state. CPU preflights require exact continuation; CUDA ``grid_sample`` backward
can be nondeterministic, so no bitwise CUDA-resume claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from source.atlas_pose_runtime import (
    ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256,
    ATLAS_POSE_PREPROCESSING_VERSION,
)
from source.dense_registration_preprocessing import (
    MASK_CONTRACT_SHA256,
    PREPROCESSING_CONTRACT_V2,
)
from training.atlas_pose_models_v7 import (
    AP_MAX_UM,
    AP_MIN_UM,
    PHYSICAL_POSE_LOSS_SCALE,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    atlas_pose_v7_loss,
    ouv_pose_loss,
    pose_to_quicknii_ouv,
)
from training.dense_registration_model import warp_tensor
from training.joint_pose_registration_data import (
    TILT_OFFSET_LEVELS_DEG,
    JointSyntheticData,
)
from training.joint_pose_registration_model import JointPoseRegistrationModel
from training.joint_pose_registration_release import (
    JOINT_CHECKPOINT_FORMAT_VERSION,
    normalized_source_sha256,
)
from training.joint_registered_data import (
    JointRegisteredData,
    apply_homography_to_map,
    mask_normalized_moving,
)
from training.synthetic_registration import (
    STRATA,
    SyntheticRegistrationGenerator,
    _payload_sha256,
)
from training.train_dense_registration import (
    ExponentialMovingAverage,
    _cosine_multiplier,
    atomic_json,
    capture_rng_state,
    load_checkpoint,
    registration_loss,
    restore_rng_state,
    sample_integer_labels,
    save_checkpoint,
    set_determinism,
    sha256_file,
    training_batch_seed,
)


FORMAT_VERSION = JOINT_CHECKPOINT_FORMAT_VERSION
DEFAULT_STAGES = (
    {"name": "review", "until_fraction": 0.20},
    {"name": "geometry", "until_fraction": 0.55},
    {"name": "joint", "until_fraction": 1.00},
)
DEFAULT_LOSS_WEIGHTS = {
    "initializer_pose": 1.0,
    "refined_pose": 1.0,
    "plane_anchor": 0.15,
    "ranking": 0.25,
    "dense": 1.0,
}
DEFAULT_STRATUM_PROBABILITIES = {
    "review": (0.30, 0.50, 0.20),
    "geometry": (0.15, 0.50, 0.35),
    "joint": (0.10, 0.45, 0.45),
}
DEFAULT_REGISTERED_FRACTIONS = {"review": 0.0, "geometry": 0.25, "joint": 0.50}
DEFAULT_HIGH_TILT_FRACTIONS = {"review": 0.20, "geometry": 0.15, "joint": 0.15}
CATASTROPHIC_POSE_ERROR = (250.0, 5.0, 5.0)
P95_MIN_SAMPLE_COUNT = 20
TAIL_SELECTION_WEIGHT = 0.10
CATASTROPHIC_SELECTION_WEIGHT = 0.50
WORST_GROUP_SELECTION_WEIGHT = 0.20
_TRUE_DENSE_KEYS = (
    "fixed",
    "fixed_mask",
    "fixed_visible_mask",
    "fixed_damage_mask",
    "moving",
    "moving_tissue_mask",
    "moving_damage_mask",
    "moving_visible_mask",
    "moving_model_mask",
    "fixed_to_moving",
    "moving_to_fixed",
    "similarity_h",
    "fixed_labels",
    "moving_labels",
    "local_velocity",
)


def _json_copy(value):
    return json.loads(json.dumps(value))


def _checkpoint_provenance(path: str | Path, role: str, state: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "state": state,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def warm_start_model(
    model: JointPoseRegistrationModel,
    pose_checkpoint: str | Path,
    dense_checkpoint: str | Path,
) -> dict:
    """Load the selected AtlasPose optimizer model and dense EMA candidate."""
    pose_record = _checkpoint_provenance(
        pose_checkpoint, "AtlasPose initializer", "model"
    )
    dense_record = _checkpoint_provenance(
        dense_checkpoint, "dense registrar", "ema.shadow"
    )
    pose_payload = load_checkpoint(pose_checkpoint, "cpu")
    dense_payload = load_checkpoint(dense_checkpoint, "cpu")
    pose_state = pose_payload.get("model")
    dense_state = dense_payload.get("ema", {}).get("shadow")
    if not isinstance(pose_state, dict):
        raise ValueError("AtlasPose checkpoint has no model state")
    if not isinstance(dense_state, dict):
        raise ValueError("dense checkpoint has no EMA shadow state")
    model.pose_initializer.load_state_dict(pose_state, strict=True)
    model.registrar.load_state_dict(dense_state, strict=True)
    return {"pose": pose_record, "dense": dense_record}


def normalized_pose_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.new_tensor(PHYSICAL_POSE_LOSS_SCALE)
    error = (prediction - target) / scale
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0)


def deep_pose_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Supervise every candidate/iteration, not only the last pose proposal."""
    if predictions.ndim == 2:
        predictions = predictions[:, None]
    if predictions.ndim != 3 or predictions.shape[0] != target.shape[0]:
        raise ValueError("pose predictions must have shape [B,S,3]")
    expanded_target = target[:, None].expand_as(predictions)
    return normalized_pose_loss(predictions, expanded_target)


def quicknii_plane_anchor_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Measure whole-plane OUV displacement, including coupled AP/tilt error."""
    if predictions.ndim == 2:
        predictions = predictions[:, None]
    expanded_target = target[:, None].expand_as(predictions)
    return ouv_pose_loss(
        pose_to_quicknii_ouv(predictions.reshape(-1, 3)),
        expanded_target.reshape(-1, 3),
    )


def _two_channel(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.cat((image, mask.to(dtype=image.dtype)), dim=1)


def candidate_review_outputs(
    model: JointPoseRegistrationModel,
    batch: dict,
    initialization: dict,
    *,
    candidate_chunk_size: int,
    gradient_checkpointing: bool,
    prepare_moving=None,
) -> dict:
    """Evaluate true, initial-offset, and hard-negative planes in one batch."""
    true_pose = batch["true_pose"]
    batch_size = true_pose.shape[0]
    wrong_pose = batch["wrong_candidate_pose"]
    if wrong_pose.ndim != 3 or wrong_pose.shape[:1] != (batch_size,):
        raise ValueError("wrong candidate poses must have shape [B,K,3]")
    if (
        "wrong_candidate_dense_target_valid" in batch
        and bool(batch["wrong_candidate_dense_target_valid"].any())
    ):
        raise ValueError("wrong atlas candidates cannot carry dense-flow targets")

    true_fixed = _two_channel(batch["fixed"], batch["fixed_mask"])
    initial_fixed = _two_channel(
        batch["initial_fixed"], batch["initial_fixed_mask"]
    )
    wrong_fixed = torch.cat(
        (
            batch["wrong_candidate_fixed"],
            batch["wrong_candidate_fixed_mask"].to(
                dtype=batch["wrong_candidate_fixed"].dtype
            ),
        ),
        dim=2,
    )
    candidate_fixed = torch.cat(
        (true_fixed[:, None], initial_fixed[:, None], wrong_fixed), dim=1
    )
    candidate_pose = torch.cat(
        (
            true_pose[:, None],
            batch["initial_pose"][:, None],
            wrong_pose,
        ),
        dim=1,
    )
    candidate_count = candidate_pose.shape[1]
    if prepare_moving is None:
        moving_mask = (
            batch["moving_model_mask"]
            if "moving_model_mask" in batch
            else batch["moving_tissue_mask"]
        )
        moving = _two_channel(batch["moving"], moving_mask)
        moving = moving[:, None].expand(-1, candidate_count, -1, -1, -1)
    else:
        prepared = prepare_moving(batch, candidate_fixed[:, :, 1:2].flatten(0, 1))
        candidate_moving, candidate_moving_mask = prepared[:2]
        moving = _two_channel(candidate_moving, candidate_moving_mask).reshape(
            batch_size, candidate_count, 2, *candidate_fixed.shape[-2:]
        )
    features = initialization["pose_features"]
    features = features[:, None].expand(-1, candidate_count, -1)
    flat_fixed = candidate_fixed.flatten(0, 1)
    flat_moving = moving.flatten(0, 1)
    flat_pose = candidate_pose.flatten(0, 1)
    flat_features = features.flatten(0, 1)
    refined, logits = [], []
    for start in range(0, len(flat_pose), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(flat_pose))
        pose, logit = _compact_refine(
            model,
            flat_fixed[start:stop],
            flat_moving[start:stop],
            flat_pose[start:stop],
            flat_features[start:stop],
            gradient_checkpointing=gradient_checkpointing,
        )
        refined.append(pose)
        logits.append(logit)
    return {
        "candidate_pose": candidate_pose,
        "refined_pose": torch.cat(refined).reshape(batch_size, candidate_count, 3),
        "compatibility_logits": torch.cat(logits).reshape(batch_size, candidate_count),
    }


def _compact_refine(
    model: JointPoseRegistrationModel,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pose: torch.Tensor,
    pose_features: torch.Tensor,
    *,
    gradient_checkpointing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    def forward(fixed, moving, pose, pose_features):
        return model.review_once(fixed, moving, pose, pose_features)

    if gradient_checkpointing and torch.is_grad_enabled():
        return checkpoint(
            forward,
            fixed,
            moving,
            pose,
            pose_features,
            use_reentrant=False,
        )
    return forward(fixed, moving, pose, pose_features)


def recurrent_training_rollout(
    model: JointPoseRegistrationModel,
    batch: dict,
    initialization: dict,
    render_pose,
    *,
    refinement_steps: int,
    live_initializer_fraction: float,
    gradient_checkpointing: bool,
    prepare_moving=None,
    compute_final_registration: bool = True,
) -> dict:
    """Unroll the shared reviewer with a newly rendered plane after every update."""
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be positive")
    if not 0.0 <= live_initializer_fraction <= 1.0:
        raise ValueError("live initializer fraction must be between zero and one")
    live_pose = initialization["pose"]
    if live_initializer_fraction == 0.0:
        pose = batch["initial_pose"]
        live_mask = torch.zeros(len(pose), device=pose.device, dtype=torch.bool)
    elif live_initializer_fraction == 1.0:
        pose = live_pose
        live_mask = torch.ones(len(pose), device=pose.device, dtype=torch.bool)
    else:
        live_mask = torch.rand(len(live_pose), device=live_pose.device) < live_initializer_fraction
        pose = torch.where(live_mask[:, None], live_pose, batch["initial_pose"])
    features = initialization["pose_features"]
    poses, logits = [], []
    for _ in range(refinement_steps):
        fixed, fixed_mask, _ = render_pose(pose)
        if prepare_moving is None:
            moving_mask = (
                batch["moving_model_mask"]
                if "moving_model_mask" in batch
                else batch["moving_tissue_mask"]
            )
            moving = _two_channel(batch["moving"], moving_mask)
        else:
            prepared = prepare_moving(batch, fixed_mask)
            aligned_moving, aligned_mask = prepared[:2]
            moving = _two_channel(aligned_moving, aligned_mask)
        pose, logit = _compact_refine(
            model,
            _two_channel(fixed, fixed_mask),
            moving,
            pose,
            features,
            gradient_checkpointing=gradient_checkpointing,
        )
        poses.append(pose)
        logits.append(logit)

    result = {
        "start_pose": torch.where(live_mask[:, None], live_pose, batch["initial_pose"]),
        "live_initializer_mask": live_mask,
        "pose_sequence": torch.stack(poses, dim=1),
        "compatibility_logits": torch.stack(logits, dim=1),
        "pose": pose,
    }
    if not compute_final_registration:
        return result

    # Validation and inference maps are always recomputed at the settled pose.
    with torch.no_grad():
        final_fixed, final_mask, final_labels = render_pose(pose.detach())
        final_alignment = {
            "source_to_aligned_h": torch.eye(
                3, device=pose.device, dtype=pose.dtype
            ).expand(len(pose), -1, -1)
        }
        if prepare_moving is not None:
            prepared = prepare_moving(batch, final_mask)
            aligned_moving, aligned_mask = prepared[:2]
            if len(prepared) > 2:
                final_alignment = prepared[2]
            moving = _two_channel(aligned_moving, aligned_mask)
        source_moving = batch.get("_outline_source_moving", batch["moving"])
        final_alignment = dict(
            final_alignment,
            map_pose=pose.detach(),
            source_shape=tuple(int(value) for value in source_moving.shape[-2:]),
        )
        final = model.register_final_pose(
            _two_channel(final_fixed, final_mask),
            moving,
            pose.detach(),
            features.detach(),
            final_alignment,
        )
        final["fixed_labels"] = final_labels
        final["fixed_mask"] = final_mask
    result["final_registration"] = final
    return result


def teacher_forced_dense_batch(batch: dict) -> dict:
    """Expose only the true-plane v2 contract to the dense objective."""
    return {name: batch[name] for name in _TRUE_DENSE_KEYS if name in batch}


def prepare_moving_for_fixed(
    batch: dict,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Normalize the original slice outline to each currently rendered atlas mask."""
    moving, mask, homography, _ = mask_normalized_moving(
        batch.get("_outline_source_moving", batch["moving"]),
        batch.get("_outline_source_mask", batch["moving_model_mask"]),
        target_mask,
        apply_cosine_feather=bool(batch.get("_outline_apply_cosine_feather", False)),
    )
    return moving, mask, {
        "source_to_aligned_h": homography,
    }


def normalize_synthetic_dense_contract(batch: dict) -> dict:
    """Use the runtime outline affine and compose every exact dense target into it."""
    result = dict(batch)
    raw_moving = batch.get("moving_raw_uint8")
    result["_outline_source_moving"] = (
        raw_moving.float() / 255.0 if raw_moving is not None else batch["moving"]
    )
    result["_outline_source_mask"] = batch["moving_model_mask"]
    result["_outline_apply_cosine_feather"] = raw_moving is not None
    result["_outline_source_labels"] = batch["moving_labels"]
    result["_outline_source_visible_mask"] = batch["moving_visible_mask"]
    for name in (
        "moving_raw_uint8",
        "moving_clean",
        "moving_appearance_clean",
        "moving_brush_mask",
    ):
        result.pop(name, None)
    moving, model_mask, homography, inverse_map = mask_normalized_moving(
        result["_outline_source_moving"],
        batch["moving_model_mask"],
        batch["fixed_mask"],
        apply_cosine_feather=result["_outline_apply_cosine_feather"],
    )
    result["moving"] = moving
    result["moving_model_mask"] = model_mask
    for name in (
        "moving_tissue_mask",
        "moving_damage_mask",
        "moving_visible_mask",
    ):
        result[name] = warp_tensor(
            batch[name].float(), inverse_map, mode="nearest", padding_mode="zeros"
        ) > 0.5
    result["moving_labels"] = sample_integer_labels(batch["moving_labels"], inverse_map)
    result["fixed_to_moving"] = apply_homography_to_map(
        homography, batch["fixed_to_moving"]
    )
    result["moving_to_fixed"] = warp_tensor(
        batch["moving_to_fixed"], inverse_map, padding_mode="border"
    )
    result["similarity_h"] = homography @ batch["similarity_h"]
    sampled_visible = warp_tensor(
        result["moving_visible_mask"].float(),
        result["fixed_to_moving"],
        mode="nearest",
        padding_mode="zeros",
    ) > 0.5
    result["fixed_visible_mask"] = batch["fixed_mask"] & sampled_visible
    result["fixed_damage_mask"] = batch["fixed_mask"] & ~result["fixed_visible_mask"]
    return result


def pose_review_objective(
    model: JointPoseRegistrationModel,
    batch: dict,
    *,
    render_pose,
    refinement_steps: int,
    live_initializer_fraction: float,
    candidate_chunk_size: int,
    gradient_checkpointing: bool,
    prepare_moving=None,
    compute_final_registration: bool = True,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict]:
    weights = DEFAULT_LOSS_WEIGHTS if weights is None else weights
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError("joint loss weights differ from the declared objective")
    initialization = model.initialize(batch["pose_image"])
    orientation_target = batch.get("orientation_inverted_target")
    required_atlas_outputs = {
        "image_frame_pose",
        "orientation_inverted_logit",
    }
    if orientation_target is not None and required_atlas_outputs <= set(initialization):
        initializer_pose, _ = atlas_pose_v7_loss(
            initialization,
            batch["true_pose"],
            orientation_target.float(),
            return_components=True,
        )
    else:
        initializer_pose = normalized_pose_loss(
            initialization["pose"], batch["true_pose"]
        )
    candidates = candidate_review_outputs(
        model,
        batch,
        initialization,
        candidate_chunk_size=candidate_chunk_size,
        gradient_checkpointing=gradient_checkpointing,
        prepare_moving=prepare_moving,
    )
    recurrent = recurrent_training_rollout(
        model,
        batch,
        initialization,
        render_pose,
        refinement_steps=refinement_steps,
        live_initializer_fraction=live_initializer_fraction,
        gradient_checkpointing=gradient_checkpointing,
        prepare_moving=prepare_moving,
        compute_final_registration=compute_final_registration,
    )
    refined_pose = 0.5 * (
        deep_pose_loss(candidates["refined_pose"], batch["true_pose"])
        + deep_pose_loss(recurrent["pose_sequence"], batch["true_pose"])
    )
    plane_anchor = 0.5 * (
        quicknii_plane_anchor_loss(candidates["refined_pose"], batch["true_pose"])
        + quicknii_plane_anchor_loss(recurrent["pose_sequence"], batch["true_pose"])
    )
    ranking = F.cross_entropy(
        candidates["compatibility_logits"],
        torch.zeros(
            batch["true_pose"].shape[0],
            device=batch["true_pose"].device,
            dtype=torch.long,
        ),
    )
    tensor_terms = {
        "initializer_pose": initializer_pose,
        "refined_pose": refined_pose,
        "plane_anchor": plane_anchor,
        "ranking": ranking,
    }
    total = sum(float(weights[name]) * value for name, value in tensor_terms.items())
    scalars = {name: float(value.detach()) for name, value in tensor_terms.items()}
    scalars["total"] = float(total.detach())
    outputs = {
        "initialization": initialization,
        "candidates": candidates,
        "recurrent": recurrent,
    }
    return total, scalars, outputs


def registered_objective(
    model: JointPoseRegistrationModel,
    batch: dict,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float], dict]:
    """Real sections supervise pose/review only and never receive dense targets."""
    return pose_review_objective(model, batch, **kwargs)


def generate_high_tilt_retention_batch(
    data: JointSyntheticData,
    count: int,
    split: str,
    seed: int,
    stratum: str,
    negatives_per_sample: int,
    forced_regime: str | None = None,
) -> dict:
    """Deterministic 15-35 degree exact pairs for pose and dense retention."""
    manifest = data.make_manifest(
        count, split, seed, stratum, negatives_per_sample
    )
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), 0x3515)))
    unit = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    high_lr = (15.0 + 20.0 * unit).astype(np.float32)
    high_dv = (15.0 + 20.0 * unit[::-1]).astype(np.float32)
    high_lr *= rng.choice(np.asarray((-1.0, 1.0), np.float32), count)
    high_dv *= rng.choice(np.asarray((-1.0, 1.0), np.float32), count)
    low_lr = rng.uniform(-15.0, 15.0, count).astype(np.float32)
    low_dv = rng.uniform(-15.0, 15.0, count).astype(np.float32)
    regime_names = ("lr_only", "dv_only", "both")
    if forced_regime is None:
        regimes = (np.arange(count) + int(seed) % 3) % 3
        rng.shuffle(regimes)
    else:
        if forced_regime not in regime_names:
            raise ValueError(f"unknown high-tilt regime: {forced_regime}")
        regimes = np.full(count, regime_names.index(forced_regime), dtype=np.int64)
    lr = np.where(regimes == 1, low_lr, high_lr).astype(np.float32)
    dv = np.where(regimes == 0, low_dv, high_dv).astype(np.float32)
    true_pose = np.asarray(manifest["true_pose"], np.float32).copy()
    true_pose[:, 1] = lr
    true_pose[:, 2] = dv

    initial_offset = np.asarray(manifest["initial_pose_offset"], np.float32).copy()
    wrong_offset = np.asarray(manifest["wrong_candidate_offset"], np.float32).copy()
    signed_tilt = np.concatenate((-TILT_OFFSET_LEVELS_DEG, TILT_OFFSET_LEVELS_DEG))
    for item, pose in enumerate(true_pose):
        def inside_pose_domain(offset):
            candidate = pose + np.asarray(offset, np.float32)
            return (
                AP_MIN_UM <= candidate[0] <= AP_MAX_UM
                and TILT_MIN_DEG <= candidate[1] <= TILT_MAX_DEG
                and TILT_MIN_DEG <= candidate[2] <= TILT_MAX_DEG
            )

        valid_lr = signed_tilt[
            (pose[1] + signed_tilt >= TILT_MIN_DEG)
            & (pose[1] + signed_tilt <= TILT_MAX_DEG)
        ]
        valid_dv = signed_tilt[
            (pose[2] + signed_tilt >= TILT_MIN_DEG)
            & (pose[2] + signed_tilt <= TILT_MAX_DEG)
        ]
        initial_offset[item, 1] = rng.choice(valid_lr)
        initial_offset[item, 2] = rng.choice(valid_dv)
        required = [tuple(float(value) for value in wrong_offset[item, 0])]
        if negatives_per_sample >= 2:
            adjacent = valid_lr[np.abs(valid_lr) == TILT_OFFSET_LEVELS_DEG[0]]
            required.append((0.0, float(rng.choice(adjacent)), 0.0))
        if negatives_per_sample >= 3:
            adjacent = valid_dv[np.abs(valid_dv) == TILT_OFFSET_LEVELS_DEG[0]]
            required.append((0.0, 0.0, float(rng.choice(adjacent))))
        pool = [
            tuple(float(value) for value in offset)
            for offset in wrong_offset[item]
        ]
        pool += [(0.0, float(value), 0.0) for value in valid_lr]
        pool += [(0.0, 0.0, float(value)) for value in valid_dv]
        if not inside_pose_domain(initial_offset[item]):
            raise RuntimeError("high-tilt initial pose escaped the canonical pose domain")
        if not all(inside_pose_domain(value) for value in required):
            raise RuntimeError("required high-tilt negative escaped the canonical pose domain")
        pool = list(
            dict.fromkeys(
                value
                for value in pool
                if value not in required and inside_pose_domain(value)
            )
        )
        selected = rng.choice(
            len(pool), negatives_per_sample - len(required), replace=False
        )
        wrong_offset[item] = np.asarray(
            required + [pool[int(index)] for index in selected], np.float32
        )

    manifest["true_pose"] = true_pose
    manifest["initial_pose_offset"] = initial_offset
    manifest["wrong_candidate_offset"] = wrong_offset
    manifest["registration__tilt_lr_deg"] = lr
    manifest["registration__tilt_dv_deg"] = dv
    registration = {
        key[len("registration__") :]: value
        for key, value in manifest.items()
        if key.startswith("registration__")
    }
    registration["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in registration.items() if key != "manifest_sha256"}
    )
    manifest["registration__manifest_sha256"] = registration["manifest_sha256"]
    manifest["registration_manifest_sha256"] = registration["manifest_sha256"]
    manifest["joint_manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "joint_manifest_sha256"}
    )
    batch = normalize_synthetic_dense_contract(data.batch(manifest, qa=True))
    batch["source"] = f"synthetic_high_tilt_{stratum}"
    batch["high_tilt_retention"] = True
    batch["high_tilt_regime"] = tuple(
        regime_names[int(regime)] for regime in regimes
    )
    return batch


def joint_objective(
    model: JointPoseRegistrationModel,
    batch: dict,
    *,
    dense_loss_fn=registration_loss,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float], dict]:
    if not bool(batch["true_dense_target_valid"].all()):
        raise ValueError("the teacher-forced synthetic plane must have exact dense targets")
    total, scalars, outputs = pose_review_objective(model, batch, **kwargs)
    if model.training and not any(
        parameter.requires_grad for parameter in model.registrar.parameters()
    ):
        scalars["dense_skipped"] = 1.0
        scalars["total"] = float(total.detach())
        return total, scalars, outputs

    weights = DEFAULT_LOSS_WEIGHTS if kwargs.get("weights") is None else kwargs["weights"]
    dense, dense_terms, dense_details = dense_loss_fn(
        model.registrar,
        teacher_forced_dense_batch(batch),
    )
    total = total + float(weights["dense"]) * dense
    scalars["dense"] = float(dense.detach())
    scalars["dense_skipped"] = 0.0
    scalars["total"] = float(total.detach())
    scalars.update({f"dense_{name}": float(value) for name, value in dense_terms.items()})
    outputs["dense"] = dense_details
    return total, scalars, outputs


def _stage_schedule(config: dict) -> tuple[dict, ...]:
    total = int(config["total_views"])
    stages = config.get("stages", DEFAULT_STAGES)
    resolved = []
    previous = 0
    previous_rank = -1
    stage_rank = {"review": 0, "geometry": 1, "joint": 2}
    for item in stages:
        name = str(item["name"])
        until = item.get("until_views")
        if until is None:
            until = math.ceil(float(item["until_fraction"]) * total)
        until = min(int(until), total)
        rank = stage_rank.get(name, -1)
        if rank <= previous_rank or until <= previous:
            raise ValueError("training stages must be ordered review/geometry/joint intervals")
        resolved.append({"name": name, "until_views": until})
        previous = until
        previous_rank = rank
    if not resolved or resolved[-1]["until_views"] != total:
        raise ValueError("the final training stage must end at total_views")
    return tuple(resolved)


def training_stage(schedule: tuple[dict, ...], completed_views: int) -> str:
    for stage in schedule:
        if completed_views < stage["until_views"]:
            return stage["name"]
    return schedule[-1]["name"]


def sample_training_stratum(
    data_rng: np.random.Generator,
    stage: str,
    probabilities: dict[str, list[float] | tuple[float, ...]],
) -> str:
    strata = tuple(STRATA)
    return str(data_rng.choice(strata, p=probabilities[stage]))


def apply_training_stage(model: JointPoseRegistrationModel, stage: str) -> int:
    model.set_training_stage(stage)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count == 0:
        raise RuntimeError(f"training stage {stage} has no trainable parameters")
    return count


def _batch_metrics(batch: dict, outputs: dict) -> dict[str, float]:
    candidates = outputs["candidates"]
    selected = candidates["compatibility_logits"].argmax(dim=1)
    pose = outputs["recurrent"]["pose"]
    error = (pose - batch["true_pose"]).abs().mean(dim=0)
    initial_error = (
        outputs["initialization"]["pose"] - batch["true_pose"]
    ).abs().mean(dim=0)
    final = outputs["recurrent"]["final_registration"]
    moving_labels = batch["moving_labels"]
    moving_visible = batch["moving_visible_mask"]
    fixed_to_moving = final["fixed_to_source_model_map"]
    if "_outline_source_labels" in batch:
        moving_labels = batch["_outline_source_labels"]
        moving_visible = batch["_outline_source_visible_mask"]
    sampled_moving_labels = sample_integer_labels(
        moving_labels, fixed_to_moving
    )
    sampled_visible = warp_tensor(
        moving_visible.float(),
        fixed_to_moving,
        mode="nearest",
        padding_mode="zeros",
    ) > 0.5
    valid = final["fixed_mask"].bool() & sampled_visible
    agreement = (sampled_moving_labels == final["fixed_labels"]) & valid
    valid_count = valid.flatten(1).sum(1)
    fixed_count = final["fixed_mask"].flatten(1).sum(1).clamp_min(1)
    correspondence = (agreement.flatten(1).sum(1) / valid_count.clamp_min(1)).mean()
    valid_fraction = (valid_count / fixed_count).mean()
    invalid_endpoint_fraction = (valid_count == 0).float().mean()
    expected_visible_fraction = (
        batch["fixed_visible_mask"].flatten(1).sum(1)
        / batch["fixed_mask"].flatten(1).sum(1).clamp_min(1)
    )
    retained_coverage = (valid_count / fixed_count) / expected_visible_fraction.clamp_min(1e-6)
    retained_coverage = retained_coverage.clamp(max=1.0)
    coverage_failure_fraction = (retained_coverage < 0.80).float().mean()
    sample_dice = []
    for item in range(len(pose)):
        expected = final["fixed_labels"][item, 0]
        observed = sampled_moving_labels[item, 0]
        item_valid = valid[item, 0]
        region_ids = torch.unique(torch.cat((expected[item_valid], observed[item_valid])))
        region_ids = region_ids[region_ids != 0]
        region_dice = []
        for region_id in region_ids:
            expected_region = (expected == region_id) & item_valid
            observed_region = (observed == region_id) & item_valid
            denominator = expected_region.sum() + observed_region.sum()
            if denominator:
                region_dice.append(
                    2.0 * (expected_region & observed_region).sum() / denominator
                )
        sample_dice.append(
            torch.stack(region_dice).mean()
            if region_dice
            else correspondence.new_tensor(0.0)
        )
    macro_dice = torch.stack(sample_dice).mean()
    metrics = {
        "count": float(batch["true_pose"].shape[0]),
        "ap_mae_um": float(error[0].detach()),
        "lr_mae_deg": float(error[1].detach()),
        "dv_mae_deg": float(error[2].detach()),
        "initial_ap_mae_um": float(initial_error[0].detach()),
        "initial_lr_mae_deg": float(initial_error[1].detach()),
        "initial_dv_mae_deg": float(initial_error[2].detach()),
        "ranking_accuracy": float((selected == 0).float().mean().detach()),
        "end_to_end_region_correspondence": float(correspondence.detach()),
        "end_to_end_macro_region_dice": float(macro_dice.detach()),
        "end_to_end_valid_fraction": float(valid_fraction.detach()),
        "end_to_end_retained_coverage": float(retained_coverage.mean().detach()),
        "end_to_end_retained_coverage_p10": float(
            torch.quantile(retained_coverage.float(), 0.10).detach()
        ),
        "coverage_failure_fraction": float(coverage_failure_fraction.detach()),
        "invalid_endpoint_fraction": float(invalid_endpoint_fraction.detach()),
    }
    if (
        "orientation_inverted_logit" in outputs["initialization"]
        and "orientation_inverted_target" in batch
    ):
        predicted = outputs["initialization"]["orientation_inverted_logit"] >= 0.0
        metrics["orientation_accuracy"] = float(
            (predicted == batch["orientation_inverted_target"]).float().mean().detach()
        )
    return metrics


def _weighted_validation_report(rows: list[dict]) -> dict[str, float]:
    total = sum(row["count"] for row in rows)
    report = {
        name: sum(row[name] * row["count"] for row in rows) / total
        for name in rows[0]
        if name
        not in {"count", "artifact_stratum", "high_tilt_regime", "_pose_errors"}
    }
    report["count"] = int(total)
    return report


def _pose_error_tail_report(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    p95 = np.quantile(errors, 0.95, axis=0)
    catastrophic = np.any(
        errors > np.asarray(CATASTROPHIC_POSE_ERROR, dtype=np.float64), axis=1
    )
    return {
        "ap_p95_um": float(p95[0]),
        "lr_p95_deg": float(p95[1]),
        "dv_p95_deg": float(p95[2]),
        "catastrophic_pose_failure_rate": float(catastrophic.mean()),
        "pose_tail_sample_count": int(len(errors)),
        "pose_p95_supported": bool(len(errors) >= P95_MIN_SAMPLE_COUNT),
        "catastrophic_pose_threshold": {
            "ap_um": CATASTROPHIC_POSE_ERROR[0],
            "lr_deg": CATASTROPHIC_POSE_ERROR[1],
            "dv_deg": CATASTROPHIC_POSE_ERROR[2],
        },
        "robust_selection_contract": {
            "pose_p95_weight": TAIL_SELECTION_WEIGHT,
            "catastrophic_failure_weight": CATASTROPHIC_SELECTION_WEIGHT,
            "worst_group_gap_weight": WORST_GROUP_SELECTION_WEIGHT,
            "pose_scale": {
                "ap_um": PHYSICAL_POSE_LOSS_SCALE[0],
                "lr_deg": PHYSICAL_POSE_LOSS_SCALE[1],
                "dv_deg": PHYSICAL_POSE_LOSS_SCALE[2],
            },
        },
    }


def robust_validation_selection_score(
    pooled_score: float,
    tail_report: dict,
    group_scores: tuple[float, ...] | list[float] = (),
) -> tuple[float, dict[str, float]]:
    """Apply the prespecified tail, catastrophic, and worst-group penalties."""
    tail_ratio = (
        tail_report["ap_p95_um"] / PHYSICAL_POSE_LOSS_SCALE[0]
        + tail_report["lr_p95_deg"] / PHYSICAL_POSE_LOSS_SCALE[1]
        + tail_report["dv_p95_deg"] / PHYSICAL_POSE_LOSS_SCALE[2]
    )
    tail_penalty = TAIL_SELECTION_WEIGHT * tail_ratio
    catastrophic_penalty = (
        CATASTROPHIC_SELECTION_WEIGHT
        * tail_report["catastrophic_pose_failure_rate"]
    )
    worst_group_gap = (
        max(0.0, float(pooled_score) - min(map(float, group_scores)))
        if group_scores
        else 0.0
    )
    penalties = {
        "pose_p95": float(tail_penalty),
        "catastrophic_pose_failures": float(catastrophic_penalty),
        "worst_group_gap": float(WORST_GROUP_SELECTION_WEIGHT * worst_group_gap),
    }
    score = float(pooled_score) - sum(penalties.values())
    if not math.isfinite(score):
        raise RuntimeError("robust validation selection score is non-finite")
    return score, penalties


@torch.no_grad()
def evaluate_model(
    model: JointPoseRegistrationModel,
    data: JointSyntheticData,
    *,
    count: int,
    batch_size: int,
    seed: int,
    negatives_per_sample: int,
    refinement_steps: int,
    candidate_chunk_size: int,
    weights: dict[str, float] | None = None,
    dense_loss_fn=registration_loss,
    high_tilt_count_per_stratum: int = 8,
    high_tilt_seed: int = 41003,
    high_tilt_selection_weight: float = 0.20,
) -> dict:
    model.eval()
    rows = []
    ordinal = 0
    for stratum_index, stratum in enumerate(STRATA):
        remaining = count
        while remaining:
            current = min(batch_size, remaining)
            batch_seed = (1 << 62) | (int(seed) << 24) | (stratum_index << 20) | ordinal
            batch = data.generate(
                current,
                "validation",
                batch_seed,
                stratum,
                negatives_per_sample,
                qa=True,
            )
            batch = normalize_synthetic_dense_contract(batch)
            loss, terms, outputs = joint_objective(
                model,
                batch,
                render_pose=data.render_pose,
                refinement_steps=refinement_steps,
                live_initializer_fraction=1.0,
                candidate_chunk_size=candidate_chunk_size,
                gradient_checkpointing=False,
                prepare_moving=prepare_moving_for_fixed,
                weights=weights,
                dense_loss_fn=dense_loss_fn,
            )
            pose_errors = (
                outputs["recurrent"]["pose"] - batch["true_pose"]
            ).abs().detach().cpu().numpy()
            rows.append(
                {
                    **_batch_metrics(batch, outputs),
                    **terms,
                    "loss": float(loss),
                    "artifact_stratum": stratum,
                    "_pose_errors": pose_errors,
                }
            )
            remaining -= current
            ordinal += 1
    report = _weighted_validation_report(rows)
    report.update(
        _pose_error_tail_report(
            np.concatenate([row["_pose_errors"] for row in rows])
        )
    )
    report["by_artifact_stratum"] = {}
    for stratum in STRATA:
        values = [row for row in rows if row["artifact_stratum"] == stratum]
        summary = _weighted_validation_report(values)
        summary.update(
            _pose_error_tail_report(
                np.concatenate([row["_pose_errors"] for row in values])
            )
        )
        pooled_score = validation_score(summary)
        summary["pooled_selection_score_before_robustness"] = pooled_score
        summary["selection_score"], summary["selection_penalties"] = (
            robust_validation_selection_score(pooled_score, summary)
        )
        report["by_artifact_stratum"][stratum] = summary
    pooled_score = validation_score(report)
    report["pooled_selection_score_before_robustness"] = pooled_score
    report["selection_score"], report["selection_penalties"] = (
        robust_validation_selection_score(
            pooled_score,
            report,
            [
                summary["selection_score"]
                for summary in report["by_artifact_stratum"].values()
            ],
        )
    )
    if high_tilt_count_per_stratum:
        high_tilt = evaluate_high_tilt_retention(
            model,
            data,
            count_per_stratum=high_tilt_count_per_stratum,
            batch_size=batch_size,
            seed=high_tilt_seed,
            negatives_per_sample=negatives_per_sample,
            refinement_steps=refinement_steps,
            candidate_chunk_size=candidate_chunk_size,
            weights=weights,
            dense_loss_fn=dense_loss_fn,
        )
        base_score = report["selection_score"]
        report.update(
            base_synthetic_selection_score=base_score,
            high_tilt_retention=high_tilt,
            high_tilt_selection_weight=float(high_tilt_selection_weight),
            selection_score=(1.0 - high_tilt_selection_weight) * base_score
            + high_tilt_selection_weight * high_tilt["selection_score"],
        )
    return report


def _registered_batch_metrics(batch: dict, outputs: dict) -> dict[str, float]:
    selected = outputs["candidates"]["compatibility_logits"].argmax(dim=1)
    final_error = (outputs["recurrent"]["pose"] - batch["true_pose"]).abs().mean(0)
    initial_error = (
        outputs["initialization"]["pose"] - batch["true_pose"]
    ).abs().mean(0)
    metrics = {
        "count": float(len(batch["true_pose"])),
        "ap_mae_um": float(final_error[0]),
        "lr_mae_deg": float(final_error[1]),
        "dv_mae_deg": float(final_error[2]),
        "initial_ap_mae_um": float(initial_error[0]),
        "initial_lr_mae_deg": float(initial_error[1]),
        "initial_dv_mae_deg": float(initial_error[2]),
        "ranking_accuracy": float((selected == 0).float().mean()),
    }
    if (
        "orientation_inverted_logit" in outputs["initialization"]
        and "orientation_inverted_target" in batch
    ):
        predicted = outputs["initialization"]["orientation_inverted_logit"] >= 0.0
        metrics["orientation_accuracy"] = float(
            (predicted == batch["orientation_inverted_target"]).float().mean()
        )
    return metrics


def registered_validation_score(report: dict) -> float:
    score = -float(
        report["ap_mae_um"] / PHYSICAL_POSE_LOSS_SCALE[0]
        + report["lr_mae_deg"] / PHYSICAL_POSE_LOSS_SCALE[1]
        + report["dv_mae_deg"] / PHYSICAL_POSE_LOSS_SCALE[2]
        + (1.0 - report["ranking_accuracy"])
        + (1.0 - report.get("orientation_accuracy", 1.0))
    )
    if not math.isfinite(score):
        raise RuntimeError("Product-5 validation selection score is non-finite")
    return score


@torch.no_grad()
def evaluate_registered_model(
    model: JointPoseRegistrationModel,
    data: JointRegisteredData,
    *,
    count: int,
    batch_size: int,
    seed: int,
    negatives_per_sample: int,
    refinement_steps: int,
    candidate_chunk_size: int,
    weights: dict[str, float] | None = None,
) -> dict:
    """Development-consumed, specimen-disjoint Product-5 validation."""
    model.eval()
    positions = data.fixed_validation_positions(count, seed)
    rows = []
    for ordinal, start in enumerate(range(0, count, batch_size)):
        batch = data.batch_positions(
            positions[start : start + batch_size],
            (1 << 61) | (int(seed) << 20) | ordinal,
            negatives_per_sample,
        )
        loss, terms, outputs = registered_objective(
            model,
            batch,
            render_pose=data.render_pose,
            refinement_steps=refinement_steps,
            live_initializer_fraction=1.0,
            candidate_chunk_size=candidate_chunk_size,
            gradient_checkpointing=False,
            prepare_moving=data.moving_for_fixed,
            weights=weights,
        )
        pose_errors = (
            outputs["recurrent"]["pose"] - batch["true_pose"]
        ).abs().detach().cpu().numpy()
        rows.append(
            {
                **_registered_batch_metrics(batch, outputs),
                "loss": float(loss),
                **terms,
                "_pose_errors": pose_errors,
            }
        )
    report = _weighted_validation_report(rows)
    report.update(
        _pose_error_tail_report(
            np.concatenate([row["_pose_errors"] for row in rows])
        )
    )
    report.update(
        role="development_consumed_product5_validation",
        registered_contract_sha256=data.contract["contract_sha256"],
    )
    pooled_score = registered_validation_score(report)
    report["pooled_selection_score_before_robustness"] = pooled_score
    report["selection_score"], report["selection_penalties"] = (
        robust_validation_selection_score(pooled_score, report)
    )
    return report


@torch.no_grad()
def evaluate_high_tilt_retention(
    model: JointPoseRegistrationModel,
    data: JointSyntheticData,
    *,
    count_per_stratum: int,
    batch_size: int,
    seed: int,
    negatives_per_sample: int,
    refinement_steps: int,
    candidate_chunk_size: int,
    weights: dict[str, float] | None = None,
    dense_loss_fn=registration_loss,
) -> dict:
    model.eval()
    rows = []
    regime_names = ("lr_only", "dv_only", "both")
    ordinal = 0
    for stratum_index, stratum in enumerate(STRATA):
        assignments = [
            regime_names[(stratum_index + item) % len(regime_names)]
            for item in range(count_per_stratum)
        ]
        for regime_index, regime in enumerate(regime_names):
            remaining = assignments.count(regime)
            while remaining:
                current = min(batch_size, remaining)
                batch_seed = (
                    (1 << 60)
                    | (int(seed) << 24)
                    | (stratum_index << 20)
                    | (regime_index << 16)
                    | ordinal
                )
                batch = generate_high_tilt_retention_batch(
                    data,
                    current,
                    "validation",
                    batch_seed,
                    stratum,
                    negatives_per_sample,
                    forced_regime=regime,
                )
                loss, terms, outputs = joint_objective(
                    model,
                    batch,
                    render_pose=data.render_pose,
                    refinement_steps=refinement_steps,
                    live_initializer_fraction=1.0,
                    candidate_chunk_size=candidate_chunk_size,
                    gradient_checkpointing=False,
                    prepare_moving=prepare_moving_for_fixed,
                    weights=weights,
                    dense_loss_fn=dense_loss_fn,
                )
                rows.append(
                    {
                        **_batch_metrics(batch, outputs),
                        "loss": float(loss),
                        **terms,
                        "artifact_stratum": stratum,
                        "high_tilt_regime": regime,
                        "_pose_errors": (
                            outputs["recurrent"]["pose"] - batch["true_pose"]
                        ).abs().detach().cpu().numpy(),
                    }
                )
                remaining -= current
                ordinal += 1
    report = _weighted_validation_report(rows)
    report.update(
        _pose_error_tail_report(
            np.concatenate([row["_pose_errors"] for row in rows])
        )
    )
    report.update(role="development_high_tilt_15_35_exact_dense_retention")
    report["by_regime"] = {}
    report["by_artifact_stratum"] = {}
    for group_name, group_values in (
        ("by_regime", regime_names),
        ("by_artifact_stratum", STRATA),
    ):
        field = "high_tilt_regime" if group_name == "by_regime" else "artifact_stratum"
        for value in group_values:
            values = [row for row in rows if row[field] == value]
            if not values:
                continue
            summary = _weighted_validation_report(values)
            summary.update(
                _pose_error_tail_report(
                    np.concatenate([row["_pose_errors"] for row in values])
                )
            )
            pooled_score = validation_score(summary)
            summary["pooled_selection_score_before_robustness"] = pooled_score
            summary["selection_score"], summary["selection_penalties"] = (
                robust_validation_selection_score(pooled_score, summary)
            )
            report[group_name][value] = summary
    pooled_score = validation_score(report)
    report["pooled_selection_score_before_robustness"] = pooled_score
    group_scores = [
        summary["selection_score"]
        for group_name in ("by_regime", "by_artifact_stratum")
        for summary in report[group_name].values()
    ]
    report["selection_score"], report["selection_penalties"] = (
        robust_validation_selection_score(
            pooled_score,
            report,
            group_scores,
        )
    )
    return report


def combine_validation_reports(
    synthetic: dict,
    registered: dict,
    registered_weight: float,
) -> dict:
    synthetic_score = float(synthetic["selection_score"])
    registered_score = float(registered["selection_score"])
    score = (1.0 - registered_weight) * synthetic_score + registered_weight * registered_score
    if not math.isfinite(score):
        raise RuntimeError("combined validation selection score is non-finite")
    report = dict(synthetic)
    report.update(
        synthetic_selection_score=synthetic_score,
        registered_product5=registered,
        registered_validation_weight=float(registered_weight),
        selection_score=score,
        selection_role="synthetic_and_development_consumed_product5_validation",
    )
    return report


def validation_score(report: dict) -> float:
    """Validation-only scalar; no training statistic can promote a checkpoint."""
    pose_ratio = (
        report["ap_mae_um"] / PHYSICAL_POSE_LOSS_SCALE[0]
        + report["lr_mae_deg"] / PHYSICAL_POSE_LOSS_SCALE[1]
        + report["dv_mae_deg"] / PHYSICAL_POSE_LOSS_SCALE[2]
    )
    endpoint_penalty = (
        2.0 * (1.0 - report["end_to_end_region_correspondence"])
        + (1.0 - report["end_to_end_macro_region_dice"])
        + (1.0 - report.get("end_to_end_retained_coverage", 1.0))
        + report.get("coverage_failure_fraction", 0.0)
        + report.get("invalid_endpoint_fraction", 0.0)
        + (1.0 - report.get("orientation_accuracy", 1.0))
    )
    score = -float(
        pose_ratio
        + endpoint_penalty
        + (1.0 - report["ranking_accuracy"])
        + report["dense"]
    )
    if not math.isfinite(score):
        raise RuntimeError("synthetic validation selection score is non-finite")
    return score


def _normalized_config(config: dict) -> dict:
    result = _json_copy(config)
    for runtime_key in ("resume", "stop_after_views"):
        result.pop(runtime_key, None)
    result["stages"] = list(_stage_schedule(config))
    result["loss_weights"] = {
        name: float(config.get("loss_weights", DEFAULT_LOSS_WEIGHTS)[name])
        for name in DEFAULT_LOSS_WEIGHTS
    }
    result["refinement_steps"] = int(config.get("refinement_steps", 3))
    result["candidate_chunk_size"] = int(config.get("candidate_chunk_size", 2))
    result["validation_negatives_per_sample"] = int(
        config.get("validation_negatives_per_sample", config["negatives_per_sample"])
    )
    result["gradient_checkpointing"] = bool(config.get("gradient_checkpointing", True))
    result["amp_initial_scale"] = float(config.get("amp_initial_scale", 65536.0))
    result["early_stopping_patience_validations"] = int(
        config.get("early_stopping_patience_validations", 0)
    )
    result["checkpoint_every_views"] = int(
        config.get("checkpoint_every_views", config["validation_every_views"])
    )
    live = config.get(
        "live_initializer_fraction_by_stage",
        {"review": 0.25, "geometry": 0.50, "joint": 0.75},
    )
    result["live_initializer_fraction_by_stage"] = {
        name: float(live[name]) for name in ("review", "geometry", "joint")
    }
    stratum_probabilities = config.get(
        "stratum_probabilities_by_stage", DEFAULT_STRATUM_PROBABILITIES
    )
    result["stratum_probabilities_by_stage"] = {
        stage: [float(value) for value in stratum_probabilities[stage]]
        for stage in ("review", "geometry", "joint")
    }
    registered_fractions = config.get(
        "registered_fraction_by_stage", DEFAULT_REGISTERED_FRACTIONS
    )
    result["registered_fraction_by_stage"] = {
        stage: float(registered_fractions[stage])
        for stage in ("review", "geometry", "joint")
    }
    high_tilt_fractions = config.get(
        "high_tilt_fraction_by_stage", DEFAULT_HIGH_TILT_FRACTIONS
    )
    result["high_tilt_fraction_by_stage"] = {
        stage: float(high_tilt_fractions[stage])
        for stage in ("review", "geometry", "joint")
    }
    result["high_tilt_validation_count_per_stratum"] = int(
        config.get("high_tilt_validation_count_per_stratum", 8)
    )
    result["high_tilt_validation_seed"] = int(
        config.get("high_tilt_validation_seed", int(config["validation_seed"]) + 65537)
    )
    result["high_tilt_selection_weight"] = float(
        config.get("high_tilt_selection_weight", 0.20)
    )
    result["registered_validation_count"] = int(
        config.get("registered_validation_count", 96)
    )
    result["registered_validation_batch_size"] = int(
        config.get("registered_validation_batch_size", 1)
    )
    result["registered_validation_seed"] = int(
        config.get("registered_validation_seed", int(config["validation_seed"]) + 104729)
    )
    result["registered_validation_weight"] = float(
        config.get("registered_validation_weight", 0.5)
    )
    if (
        result["refinement_steps"] < 1
        or result["candidate_chunk_size"] < 1
        or result["validation_negatives_per_sample"] < 1
    ):
        raise ValueError(
            "refinement steps, candidate chunk size, and validation negatives "
            "must be positive"
        )
    if (
        not math.isfinite(result["amp_initial_scale"])
        or result["amp_initial_scale"] <= 0.0
    ):
        raise ValueError("AMP initial scale must be finite and positive")
    if result["early_stopping_patience_validations"] < 0:
        raise ValueError("early-stopping patience cannot be negative")
    if result["checkpoint_every_views"] < 1:
        raise ValueError("checkpoint interval must be positive")
    if any(not 0.0 <= value <= 1.0 for value in result["live_initializer_fraction_by_stage"].values()):
        raise ValueError("live initializer fractions must lie between zero and one")
    for probabilities in result["stratum_probabilities_by_stage"].values():
        if len(probabilities) != len(STRATA) or any(value < 0.0 for value in probabilities):
            raise ValueError("each stage needs one nonnegative probability per artifact stratum")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("artifact-stratum probabilities must sum to one")
    if any(not 0.0 <= value < 1.0 for value in result["registered_fraction_by_stage"].values()):
        raise ValueError("registered fractions must lie in [0, 1)")
    if any(not 0.0 <= value < 1.0 for value in result["high_tilt_fraction_by_stage"].values()):
        raise ValueError("high-tilt retention fractions must lie in [0, 1)")
    if any(
        result["registered_fraction_by_stage"][stage]
        + result["high_tilt_fraction_by_stage"][stage]
        >= 1.0
        for stage in ("review", "geometry", "joint")
    ):
        raise ValueError("registered and high-tilt fractions must sum to less than one")
    if result["high_tilt_validation_count_per_stratum"] < 0:
        raise ValueError("high-tilt validation count cannot be negative")
    if not 0.0 <= result["high_tilt_selection_weight"] <= 1.0:
        raise ValueError("high-tilt validation weight must lie between zero and one")
    if result["registered_validation_count"] < 1 or result["registered_validation_batch_size"] < 1:
        raise ValueError("registered validation count and batch size must be positive")
    if not 0.0 <= result["registered_validation_weight"] <= 1.0:
        raise ValueError("registered validation weight must lie between zero and one")
    if "registered_root" in result:
        result["registered_root"] = str(Path(result["registered_root"]).resolve())
    if "pose_checkpoint" in result:
        result["pose_checkpoint"] = str(Path(result["pose_checkpoint"]).resolve())
        result["pose_checkpoint_sha256"] = sha256_file(result["pose_checkpoint"])
    if "dense_checkpoint" in result:
        result["dense_checkpoint"] = str(Path(result["dense_checkpoint"]).resolve())
        result["dense_checkpoint_sha256"] = sha256_file(result["dense_checkpoint"])
    return result


def _generator_contract(data, registered_data=None, registered_validation_data=None) -> dict:
    generator = getattr(data, "generator", None)
    folder = Path(__file__).parent
    source_folder = folder.parent / "source"
    return {
        "format_version": FORMAT_VERSION,
        "source_sha256": {
            "trainer": normalized_source_sha256(Path(__file__)),
            "model": normalized_source_sha256(folder / "joint_pose_registration_model.py"),
            "synthetic_adapter": normalized_source_sha256(
                folder / "joint_pose_registration_data.py"
            ),
            "registered_adapter_and_canvas": normalized_source_sha256(
                folder / "joint_registered_data.py"
            ),
            "atlas_pose_models": normalized_source_sha256(
                folder / "atlas_pose_models_v7.py"
            ),
            "dense_registration_model": normalized_source_sha256(
                folder / "dense_registration_model.py"
            ),
            "dense_loss_ema_and_checkpoint": normalized_source_sha256(
                folder / "train_dense_registration.py"
            ),
            "atlas_pose_preprocessing": normalized_source_sha256(
                source_folder / "atlas_pose_runtime.py"
            ),
            "dense_registration_preprocessing": normalized_source_sha256(
                source_folder / "dense_registration_preprocessing.py"
            ),
        },
        "preprocessing_contract": {
            "atlas_pose_version": ATLAS_POSE_PREPROCESSING_VERSION,
            "atlas_pose_sha256": ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256,
            "dense_registration_version": PREPROCESSING_CONTRACT_V2,
            "dense_mask_sha256": MASK_CONTRACT_SHA256,
        },
        "synthetic": _json_copy(
            getattr(generator, "contract", getattr(data, "contract", {}))
        ),
        "registered_train": _json_copy(
            getattr(registered_data, "contract", None)
        ),
        "registered_validation": _json_copy(
            getattr(registered_validation_data, "contract", None)
        ),
    }


def _verify_resume_generator_contract(checkpoint_contract: dict, current_contract: dict) -> None:
    if checkpoint_contract != current_contract:
        raise ValueError("resume generator contract differs")


def _save_state(
    *,
    config: dict,
    model: JointPoseRegistrationModel,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    data_rng,
    step: int,
    batch_ordinal: int,
    completed_views: int,
    best_score: float,
    validations_without_improvement: int,
    latest_validation: dict | None,
    generator_contract: dict,
    warm_start: dict,
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "config": config,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(data_rng),
        "step": step,
        "batch_ordinal": batch_ordinal,
        "completed_views": completed_views,
        "best_validation_score": best_score,
        "validation_checkpoints_without_improvement": validations_without_improvement,
        "latest_validation": latest_validation,
        "generator_contract": generator_contract,
        "warm_start": warm_start,
    }


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line.rstrip() + "\n")


def _write_progress(path: Path, payload: dict, log_path: Path) -> None:
    atomic_json(path, payload)
    percent = 100.0 * payload["completed_views"] / max(payload["total_views"], 1)
    eta = payload.get("eta_seconds")
    eta_text = "--:--:--" if eta is None else time.strftime(
        "%H:%M:%S", time.gmtime(max(0.0, eta))
    )
    line = (
        f"[{percent:6.2f}%] {payload['completed_views']:,}/{payload['total_views']:,} "
        f"views | {payload['views_per_second']:.2f} views/s | ETA {eta_text} | "
        f"{payload['stage']} | loss {payload['smoothed_terms'].get('total', math.nan):.4f}"
    )
    print(line, flush=True)
    _append_log(log_path, line)


def _build_production_components(config: dict, device: torch.device):
    generator = SyntheticRegistrationGenerator(config["atlas"], device)
    synthetic = JointSyntheticData(generator)
    if "registered_root" in config:
        registered = JointRegisteredData(
            config["registered_root"], config["atlas"], synthetic, split="train"
        )
        registered_validation = JointRegisteredData(
            config["registered_root"], config["atlas"], synthetic, split="validation"
        )
    else:
        registered = registered_validation = None
    return (
        JointPoseRegistrationModel().to(device),
        synthetic,
        registered,
        registered_validation,
    )


def train(
    config: dict,
    *,
    model: JointPoseRegistrationModel | None = None,
    data: JointSyntheticData | None = None,
    registered_data: JointRegisteredData | None = None,
    registered_validation_data: JointRegisteredData | None = None,
    dense_loss_fn=registration_loss,
    evaluation_fn=evaluate_model,
    registered_evaluation_fn=evaluate_registered_model,
) -> Path:
    """Train or state/RNG-resume one validation-selected joint run."""
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    set_determinism(int(config["seed"]))
    normalized = _normalized_config(config)
    schedule = tuple(normalized["stages"])
    run_folder = Path(config["workspace"]) / "runs" / config["run_name"]
    run_folder.mkdir(parents=True, exist_ok=True)
    config_path = run_folder / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != normalized:
            raise ValueError("resume config differs from the run's immutable config.json")
    else:
        atomic_json(config_path, normalized)

    components_injected = model is not None and data is not None
    if model is None or data is None:
        if model is not None or data is not None:
            raise ValueError("model and data must be supplied together")
        model, data, registered_data, registered_validation_data = (
            _build_production_components(config, device)
        )
    else:
        model = model.to(device)
    latest_path = run_folder / "latest.pt"
    best_path = run_folder / "best-validation.pt"
    warm_start = {}
    if not latest_path.is_file():
        if "pose_checkpoint" in normalized and "dense_checkpoint" in normalized:
            warm_start = warm_start_model(
                model,
                normalized["pose_checkpoint"],
                normalized["dense_checkpoint"],
            )
        elif not components_injected:
            raise ValueError("a production run requires pose and dense warm-start checkpoints")
    if latest_path.is_file() and not config.get("resume", True):
        raise ValueError("run already has a checkpoint; choose a new run name or resume it")
    if (
        any(normalized["registered_fraction_by_stage"].values())
        and (registered_data is None or registered_validation_data is None)
    ):
        raise ValueError(
            "specimen-disjoint registered train and validation data are required "
            "when a registered fraction is nonzero"
        )
    if (
        any(normalized["registered_fraction_by_stage"].values())
        and normalized["registered_validation_weight"] <= 0.0
    ):
        raise ValueError("registered training requires nonzero Product-5 validation weight")

    ema = ExponentialMovingAverage(model, float(config["ema_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = math.ceil(config["total_views"] / config["batch_size"])
    warmup_steps = max(
        1, math.ceil(config["scheduler_warmup_views"] / config["batch_size"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(step, total_steps, warmup_steps),
    )
    amp_enabled = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=normalized["amp_initial_scale"],
    )
    data_rng = np.random.default_rng(int(config["data_seed"]))
    step = batch_ordinal = completed_views = 0
    best_score = -math.inf
    validations_without_improvement = 0
    latest_validation = None
    contract = _generator_contract(data, registered_data, registered_validation_data)
    if config.get("resume", True) and latest_path.is_file():
        checkpoint = load_checkpoint(latest_path, device)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("resume checkpoint format differs from the trainer contract")
        if checkpoint.get("config") != normalized:
            raise ValueError("resume checkpoint config differs from immutable config")
        _verify_resume_generator_contract(
            checkpoint.get("generator_contract"), contract
        )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"], data_rng)
        step = int(checkpoint["step"])
        batch_ordinal = int(checkpoint["batch_ordinal"])
        completed_views = int(checkpoint["completed_views"])
        best_score = float(checkpoint["best_validation_score"])
        validations_without_improvement = int(
            checkpoint.get("validation_checkpoints_without_improvement", 0)
        )
        latest_validation = checkpoint.get("latest_validation")
        warm_start = checkpoint["warm_start"]

    early_stopping_patience = normalized["early_stopping_patience_validations"]
    if (
        early_stopping_patience
        and validations_without_improvement >= early_stopping_patience
        and completed_views < config["total_views"]
    ):
        if not best_path.is_file():
            raise RuntimeError("early-stopped run is missing best-validation.pt")
        return best_path

    started = time.monotonic()
    initial_views = completed_views
    next_validation = (
        completed_views // config["validation_every_views"] + 1
    ) * config["validation_every_views"]
    next_checkpoint = (
        completed_views // normalized["checkpoint_every_views"] + 1
    ) * normalized["checkpoint_every_views"]
    stop_at = min(
        int(config["total_views"]),
        completed_views + int(config.get("stop_after_views", config["total_views"])),
    )
    if (
        stop_at < int(config["total_views"])
        and (stop_at - completed_views) % int(config["batch_size"])
    ):
        raise ValueError("stop_after_views must end on a complete training batch")
    smoothed = {}
    last_progress = -math.inf
    current_stage = ""
    stopped_early = False
    model.train()
    while completed_views < stop_at:
        stage = training_stage(schedule, completed_views)
        if stage != current_stage:
            trainable = apply_training_stage(model, stage)
            current_stage = stage
            _append_log(run_folder / "training.log", f"stage {stage} | {trainable:,} trainable parameters")
        count = min(
            int(config["batch_size"]),
            int(config["total_views"]) - completed_views,
            stop_at - completed_views,
        )
        seed = training_batch_seed(int(config["data_seed"]), batch_ordinal)
        registered_fraction = normalized["registered_fraction_by_stage"][stage]
        high_tilt_fraction = normalized["high_tilt_fraction_by_stage"][stage]
        source_draw = data_rng.random()
        use_registered = source_draw < registered_fraction
        use_high_tilt = (
            not use_registered
            and source_draw < registered_fraction + high_tilt_fraction
        )
        batch_ordinal += 1
        if use_registered:
            batch = registered_data.generate(
                count, seed, int(config["negatives_per_sample"])
            )
            source = "registered_product5"
        elif use_high_tilt:
            stratum = sample_training_stratum(
                data_rng,
                stage,
                normalized["stratum_probabilities_by_stage"],
            )
            batch = generate_high_tilt_retention_batch(
                data,
                count,
                "train",
                seed,
                stratum,
                int(config["negatives_per_sample"]),
            )
            source = f"synthetic_high_tilt_{stratum}"
        else:
            stratum = sample_training_stratum(
                data_rng,
                stage,
                normalized["stratum_probabilities_by_stage"],
            )
            batch = data.generate(
                count,
                "train",
                seed,
                stratum,
                int(config["negatives_per_sample"]),
                qa=True,
            )
            batch = normalize_synthetic_dense_contract(batch)
            source = f"synthetic_{stratum}"
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            objective_kwargs = {
                "render_pose": data.render_pose,
                "refinement_steps": normalized["refinement_steps"],
                "live_initializer_fraction": normalized[
                    "live_initializer_fraction_by_stage"
                ][stage],
                "candidate_chunk_size": normalized["candidate_chunk_size"],
                "gradient_checkpointing": normalized["gradient_checkpointing"],
                "weights": normalized["loss_weights"],
                "compute_final_registration": False,
            }
            if use_registered:
                loss, terms, _ = registered_objective(
                    model,
                    batch,
                    prepare_moving=registered_data.moving_for_fixed,
                    **objective_kwargs,
                )
            else:
                loss, terms, _ = joint_objective(
                    model,
                    batch,
                    prepare_moving=prepare_moving_for_fixed,
                    dense_loss_fn=dense_loss_fn,
                    **objective_kwargs,
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["gradient_clip"]),
        )
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if amp_enabled and scaler.get_scale() < scale_before:
            _append_log(run_folder / "training.log", "AMP overflow | update skipped")
            continue
        scheduler.step()
        ema.update(model)
        step += 1
        completed_views += count
        for name, value in terms.items():
            smoothed[name] = value if name not in smoothed else 0.95 * smoothed[name] + 0.05 * value

        now = time.monotonic()
        if now - last_progress >= float(config["progress_every_seconds"]):
            elapsed = max(now - started, 1e-6)
            rate = (completed_views - initial_views) / elapsed
            progress = {
                "run_name": config["run_name"],
                "status": "training",
                "step": step,
                "completed_views": completed_views,
                "total_views": int(config["total_views"]),
                "stage": stage,
                "source": source,
                "views_per_second": rate,
                "eta_seconds": (config["total_views"] - completed_views) / rate if rate else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "smoothed_terms": smoothed,
                "latest_validation": latest_validation,
            }
            _write_progress(run_folder / "progress.json", progress, run_folder / "training.log")
            last_progress = now

        should_validate = completed_views >= next_validation or completed_views == config["total_views"]
        should_checkpoint = completed_views >= next_checkpoint
        interrupted = completed_views == stop_at and completed_views < config["total_views"]
        improved = False
        if should_validate:
            with ema.applied(model):
                synthetic_validation = evaluation_fn(
                    model,
                    data,
                    count=int(config["validation_count_per_stratum"]),
                    batch_size=int(config["validation_batch_size"]),
                    seed=int(config["validation_seed"]),
                    negatives_per_sample=normalized[
                        "validation_negatives_per_sample"
                    ],
                    refinement_steps=normalized["refinement_steps"],
                    candidate_chunk_size=normalized["candidate_chunk_size"],
                    weights=normalized["loss_weights"],
                    dense_loss_fn=dense_loss_fn,
                    high_tilt_count_per_stratum=normalized[
                        "high_tilt_validation_count_per_stratum"
                    ],
                    high_tilt_seed=normalized["high_tilt_validation_seed"],
                    high_tilt_selection_weight=normalized[
                        "high_tilt_selection_weight"
                    ],
                )
                if registered_validation_data is not None:
                    registered_validation = registered_evaluation_fn(
                        model,
                        registered_validation_data,
                        count=normalized["registered_validation_count"],
                        batch_size=normalized["registered_validation_batch_size"],
                        seed=normalized["registered_validation_seed"],
                        negatives_per_sample=normalized[
                            "validation_negatives_per_sample"
                        ],
                        refinement_steps=normalized["refinement_steps"],
                        candidate_chunk_size=normalized["candidate_chunk_size"],
                        weights=normalized["loss_weights"],
                    )
                    latest_validation = combine_validation_reports(
                        synthetic_validation,
                        registered_validation,
                        normalized["registered_validation_weight"],
                    )
                else:
                    latest_validation = synthetic_validation
            score = float(latest_validation["selection_score"])
            if not math.isfinite(score):
                raise RuntimeError("validation selection score is non-finite")
            improved = score > best_score
            validations_without_improvement = (
                0 if improved else validations_without_improvement + 1
            )
            best_score = max(best_score, score)
            atomic_json(run_folder / "validation-latest.json", latest_validation)
            line = (
                f"validation | score {score:.6f} | AP {latest_validation['ap_mae_um']:.2f} um | "
                f"L-R {latest_validation['lr_mae_deg']:.3f} deg | "
                f"D-V {latest_validation['dv_mae_deg']:.3f} deg | "
                f"rank {latest_validation['ranking_accuracy']:.3f}"
            )
            if "registered_product5" in latest_validation:
                real = latest_validation["registered_product5"]
                line += (
                    f" | Product-5 AP {real['ap_mae_um']:.2f} um | "
                    f"L-R {real['lr_mae_deg']:.3f} deg | "
                    f"D-V {real['dv_mae_deg']:.3f} deg | rank {real['ranking_accuracy']:.3f}"
                )
            print(line, flush=True)
            _append_log(run_folder / "training.log", line)
            model.train()
            apply_training_stage(model, stage)
            while next_validation <= completed_views:
                next_validation += int(config["validation_every_views"])

        if should_checkpoint or should_validate or interrupted:
            state = _save_state(
                config=normalized,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_rng=data_rng,
                step=step,
                batch_ordinal=batch_ordinal,
                completed_views=completed_views,
                best_score=best_score,
                validations_without_improvement=validations_without_improvement,
                latest_validation=latest_validation,
                generator_contract=contract,
                warm_start=warm_start,
            )
            save_checkpoint(latest_path, state)
            if should_checkpoint:
                save_checkpoint(
                    run_folder / "checkpoints" / f"views-{completed_views:09d}.pt",
                    state,
                )
                while next_checkpoint <= completed_views:
                    next_checkpoint += normalized["checkpoint_every_views"]
            if improved:
                best_state = dict(state)
                best_state["release_selection"] = {
                    "state": "ema.shadow",
                    "criterion": "validation_selection_score",
                    "validation_score": score,
                    "completed_views": completed_views,
                }
                save_checkpoint(best_path, best_state)
        if (
            should_validate
            and early_stopping_patience
            and validations_without_improvement >= early_stopping_patience
            and completed_views < config["total_views"]
        ):
            stopped_early = True
            line = (
                "early stopping | validation score did not improve for "
                f"{validations_without_improvement} validations"
            )
            print(line, flush=True)
            _append_log(run_folder / "training.log", line)
            break

    status = (
        "complete"
        if completed_views == config["total_views"]
        else "early_stopped" if stopped_early else "interrupted"
    )
    elapsed = max(time.monotonic() - started, 1e-6)
    final_progress = {
        "run_name": config["run_name"],
        "status": status,
        "step": step,
        "completed_views": completed_views,
        "total_views": int(config["total_views"]),
        "stage": training_stage(schedule, min(completed_views, config["total_views"] - 1)),
        "views_per_second": (completed_views - initial_views) / elapsed,
        "eta_seconds": 0.0 if status in {"complete", "early_stopped"} else None,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "smoothed_terms": smoothed,
        "latest_validation": latest_validation,
    }
    atomic_json(run_folder / "progress.json", final_progress)
    _append_log(run_folder / "training.log", status)
    return best_path if status in {"complete", "early_stopped"} else latest_path


def micro_overfit(
    model: JointPoseRegistrationModel,
    batch: dict,
    *,
    render_pose,
    stage: str = "review",
    refinement_steps: int = 2,
    candidate_chunk_size: int = 2,
    steps: int = 20,
    learning_rate: float = 1e-3,
    dense_loss_fn=registration_loss,
    normalize_outline: bool = True,
) -> list[float]:
    """Small preflight hook: one fixed batch should be learnable before a long run."""
    model.train()
    model.set_training_stage(stage)
    if normalize_outline:
        batch = normalize_synthetic_dense_contract(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = joint_objective(
            model,
            batch,
            render_pose=render_pose,
            refinement_steps=refinement_steps,
            live_initializer_fraction=1.0,
            candidate_chunk_size=candidate_chunk_size,
            gradient_checkpointing=False,
            prepare_moving=(prepare_moving_for_fixed if normalize_outline else None),
            dense_loss_fn=dense_loss_fn,
            compute_final_registration=False,
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON training configuration")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(train(config), flush=True)


if __name__ == "__main__":
    main()
