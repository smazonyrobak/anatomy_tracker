"""Audit Product-5 candidate ranking as a function of physical plane offset.

This is a development diagnostic, not a release evaluator.  It explicitly loads
``ema.shadow`` from a stopped joint-training checkpoint, reconstructs the same
fixed Product-5 validation panel used by the trainer, and compares each metadata
plane independently with every declared signed one-axis offset.  Each candidate
is rendered from the CCF and receives its own outline-derived affine and cosine
feather before the frozen registrar and reviewer are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from source.atlas_pose_runtime import (
    ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256,
    ATLAS_POSE_PREPROCESSING_VERSION,
    preprocess_atlas_pose_image,
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
    pose_to_quicknii_ouv,
    quicknii_ouv_to_pose,
)
from training.dense_registration_model import (
    jacobian_determinant,
    modality_independent_descriptor,
    warp_tensor,
)
from training.evaluate_joint_registration_premise import SCORE_WEIGHTS
from training import evaluate_joint_registration_premise
from training.joint_pose_registration_data import (
    AP_OFFSET_LEVELS_UM,
    TILT_OFFSET_LEVELS_DEG,
    JointSyntheticData,
)
from training.joint_pose_registration_model import (
    JointPoseRegistrationModel,
    project_pose_to_domain,
)
from training.joint_registered_data import (
    JointRegisteredData,
    canonical_registration_image,
)
from training.quicknii_plane_metric import (
    QUICKNII_PIXEL_GRID_SHAPE,
    QUICKNII_PLANE_DISTANCE_CONTRACT,
    torch_annotation_brain_mask,
    torch_brain_masked_plane_distance,
)
from training.synthetic_registration import SyntheticRegistrationGenerator
from training.train_dense_registration import set_determinism, sha256_file
from training.train_joint_pose_registration import (
    FORMAT_VERSION as JOINT_CHECKPOINT_FORMAT_VERSION,
    _two_channel,
    recurrent_training_rollout,
)


FORMAT_VERSION = 2
CONTROL_RUN_NAME = "joint-review-mixed-2000-r4322"
CONTROL_SOURCE_CONFIG = Path(__file__).parent / "configs" / "joint_review_mixed_2000_r4322.json"
CONTROL_SOURCE_CONFIG_SHA256 = "dfb4714d2cfc4e369e5d9ca3aab0ad81841dfe36ca20ac0953938c9500f9057b"
CONTROL_CHECKPOINT_SHA256 = "0b7a940ee586cf4f91a9dc72895b50136804c9b8164490f4ad37eda28aa52790"
CONTROL_EMA_SHA256 = "b947cabbd372cee3641fe37a79cf90a14ef51c3ec3d36ef5b28b5e08389d6628"
CONTROL_SUBTREE_SHA256 = {
    "pose_initializer": "c8a18e21172ed656bd4ca4ef0b91e9996a00bc707a0b085bfc39798b7a5654d1",
    "registrar": "7505b7572f2697a78c6d7c8851d673e5b44c87222dcf3d0463a0edb4acd37a76",
    "review_head": "c173743f4553e7c7dff4349573ddc57756ab80fbaeb83c1abcd00f17c273f63a",
}
CONTROL_CONFIG_SHA256 = "8b8f11060378e6b6020a1492e95943f49f526c18cb564984f3ea060833a04fe1"
CONTROL_CRITICAL_CONFIG = {
    "registered_validation_count": 96,
    "registered_validation_seed": 1094740,
    "registered_validation_batch_size": 2,
    "validation_negatives_per_sample": 6,
    "refinement_steps": 2,
}
AXES = ("ap", "lr", "dv")
AXIS_UNITS = {"ap": "um", "lr": "deg", "dv": "deg"}
OFFSET_LEVELS = {
    "ap": tuple(float(value) for value in AP_OFFSET_LEVELS_UM),
    "lr": tuple(float(value) for value in TILT_OFFSET_LEVELS_DEG),
    "dv": tuple(float(value) for value in TILT_OFFSET_LEVELS_DEG),
}
NEAREST_LEVEL = {"ap": 25.0, "lr": 0.25, "dv": 0.25}
RESOLVABLE_LEVEL = {"ap": 100.0, "lr": 1.0, "dv": 1.0}
REGISTRATION_COMPONENT_COLUMNS = {
    "mind": "postwarp_mind_residual",
    "outline_dice": "outline_dice",
    "common_support_fraction": "common_support_fraction",
    "aligned_mask_dice": "aligned_mask_dice",
    "velocity": "normalized_velocity_magnitude",
    "topology": "topology_penalty",
    "similarity": "normalized_similarity_magnitude",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 930_731
OUV_RESIDUAL_AXES = ("ap_um", "lr_deg", "dv_deg")
OUV_FLOAT32_TOLERANCE = {"ap_um": 0.01, "lr_deg": 1e-4, "dv_deg": 1e-4}
FRAME_CONTROL_POSE = (-1450.0, 7.0, -4.0)
FRAME_CONTROL_TRANSFORM = {
    "rotation_deg": 8.0,
    "scale": 1.07,
    "translation_xy_pixels": (9.0, -6.0),
}
FRAME_CONTROL_OFFSETS = (
    (100.0, 0.0, 0.0),
    (-100.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
VARIANTS = ("identity", "horizontal", "vertical", "horizontal_vertical")
CSV_FIELDS = (
    "sample_index",
    "section_image_id",
    "specimen_id",
    "raw_image_sha256",
    "axis",
    "sign",
    "magnitude",
    "unit",
    "offset_ap_um",
    "offset_lr_deg",
    "offset_dv_deg",
    "candidate_in_domain",
    "truth_ap_um",
    "truth_lr_deg",
    "truth_dv_deg",
    "candidate_ap_um",
    "candidate_lr_deg",
    "candidate_dv_deg",
    "reviewer_truth_logit",
    "reviewer_candidate_logit",
    "reviewer_margin",
    "reviewer_truth_win",
    "reviewer_pairwise_ce",
    "registration_truth_evidence",
    "registration_candidate_evidence",
    "registration_margin",
    "registration_truth_win",
    "registration_pairwise_ce",
    "physical_corresponding_plane_distance_um",
    "candidate_refined_physical_plane_distance_um",
    "truth_refined_physical_plane_distance_um",
    "truth_postwarp_mind_residual",
    "candidate_postwarp_mind_residual",
    "truth_outline_dice",
    "candidate_outline_dice",
    "truth_common_support_fraction",
    "candidate_common_support_fraction",
    "truth_aligned_mask_dice",
    "candidate_aligned_mask_dice",
    "truth_normalized_velocity_magnitude",
    "candidate_normalized_velocity_magnitude",
    "truth_topology_penalty",
    "candidate_topology_penalty",
    "truth_normalized_similarity_magnitude",
    "candidate_normalized_similarity_magnitude",
    "candidate_initial_E",
    "candidate_refined_E",
    "candidate_refined_ap_error_um",
    "candidate_refined_lr_error_deg",
    "candidate_refined_dv_error_deg",
    "truth_refined_E",
    "truth_refined_ap_error_um",
    "truth_refined_lr_error_deg",
    "truth_refined_dv_error_deg",
    "candidate_source_to_aligned_h",
)
ORIENTATION_FIELDS = (
    "sample_index",
    "section_image_id",
    "specimen_id",
    "variant",
    "pose_input_sha256",
    "moving_input_sha256",
    "initializer_ap_um",
    "initializer_lr_deg",
    "initializer_dv_deg",
    "orientation_inverted_logit",
    "true_registration_evidence",
    "mean_reviewer_margin",
    "mean_registration_margin",
    "registration_evidence_improvement_vs_identity",
    "reviewer_margin_improvement_vs_identity",
)


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _subtree_sha256(state: dict[str, torch.Tensor], prefix: str) -> str:
    return _state_sha256(
        {name: value for name, value in state.items() if name.startswith(prefix + ".")}
    )


def _source_sha256(path: str | Path) -> str:
    source = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _control_identity(payload: dict, checkpoint_path: Path) -> dict:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("joint checkpoint has no immutable config")
    source_config_sha = sha256_file(CONTROL_SOURCE_CONFIG)
    checkpoint_sha = sha256_file(checkpoint_path)
    config_sha = _json_sha256(config)
    if source_config_sha != CONTROL_SOURCE_CONFIG_SHA256:
        raise ValueError("tracked reviewer-control source config differs from the frozen audit")
    if checkpoint_sha != CONTROL_CHECKPOINT_SHA256:
        raise ValueError("checkpoint is not the frozen reviewer-only 2,000-view control")
    if config_sha != CONTROL_CONFIG_SHA256 or config.get("run_name") != CONTROL_RUN_NAME:
        raise ValueError("checkpoint config is not the frozen reviewer-only control")
    if config.get("stages") != [{"name": "review", "until_views": 5000}]:
        raise ValueError("Product-5 audit requires the review-only control schedule")
    if any(config.get(name) != value for name, value in CONTROL_CRITICAL_CONFIG.items()):
        raise ValueError("reviewer-control validation tuple differs from the frozen audit")
    state = payload.get("ema", {}).get("shadow")
    hashes = {
        "full_ema": _state_sha256(state),
        **{name: _subtree_sha256(state, name) for name in CONTROL_SUBTREE_SHA256},
    }
    if hashes["full_ema"] != CONTROL_EMA_SHA256 or any(
        hashes[name] != expected for name, expected in CONTROL_SUBTREE_SHA256.items()
    ):
        raise ValueError("reviewer-control EMA or a frozen model subtree differs")
    return {
        "run_name": CONTROL_RUN_NAME,
        "source_config_path": str(CONTROL_SOURCE_CONFIG.resolve()),
        "source_config_sha256": source_config_sha,
        "embedded_config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "critical_config": dict(CONTROL_CRITICAL_CONFIG),
        "schedule": config["stages"],
        "ema_decay": float(payload["ema"]["decay"]),
        "state_sha256": hashes,
        "warm_start": payload.get("warm_start"),
    }


def _wilson(successes: int, total: int) -> dict[str, float | int | None]:
    if total == 0:
        return {"successes": 0, "total": 0, "rate": None, "low": None, "high": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(rate),
        "low": float(center - radius),
        "high": float(center + radius),
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None
    rng = np.random.default_rng(int(seed))
    means = np.empty(replicates, dtype=np.float64)
    chunk = 512
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        means[start : start + count] = values[
            rng.integers(0, len(values), size=(count, len(values)))
        ].mean(1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _specimen_cluster_values(values: list[float], specimen_ids: list[int]) -> np.ndarray:
    grouped = defaultdict(list)
    for specimen, value in zip(specimen_ids, values):
        grouped[int(specimen)].append(float(value))
    return np.asarray(
        [np.mean(grouped[specimen]) for specimen in sorted(grouped)], dtype=np.float64
    )


def _clustered_summary(values: list[float], specimen_ids: list[int], *, seed: int) -> dict:
    array = _specimen_cluster_values(values, specimen_ids)
    low, high = _bootstrap_mean_interval(array, seed=seed)
    return {
        "pair_count": int(len(values)),
        "specimen_count": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "specimen_cluster_bootstrap_mean_95_ci": [low, high],
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(values.dtype)
    return (values * mask).flatten(1).sum(1) / mask.expand_as(values).flatten(1).sum(1).clamp_min(1.0)


def _registration_evidence(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    details: dict,
) -> dict[str, torch.Tensor]:
    fixed_mask = fixed[:, 1:2].clamp(0.0, 1.0)
    warped = warp_tensor(moving, details["fixed_to_moving_map"], padding_mode="zeros")
    warped_mask = warped[:, 1:2].clamp(0.0, 1.0)
    overlap = fixed_mask * warped_mask
    if bool((overlap.flatten(1).sum(1) < 1.0).any()):
        raise RuntimeError("candidate has no valid post-warp tissue overlap")
    mind = _masked_mean(
        (
            modality_independent_descriptor(fixed[:, :1])
            - modality_independent_descriptor(warped[:, :1])
        ).abs(),
        overlap,
    )
    intersection = (fixed_mask * warped_mask).flatten(1).sum(1)
    outline_dice = (2.0 * intersection + 1e-6) / (
        fixed_mask.flatten(1).sum(1) + warped_mask.flatten(1).sum(1) + 1e-6
    )
    aligned_mask = moving[:, 1:2].clamp(0.0, 1.0)
    aligned_intersection = (fixed_mask * aligned_mask).flatten(1).sum(1)
    aligned_mask_dice = (2.0 * aligned_intersection + 1e-6) / (
        fixed_mask.flatten(1).sum(1) + aligned_mask.flatten(1).sum(1) + 1e-6
    )
    common_support_fraction = intersection / fixed_mask.flatten(1).sum(1).clamp_min(1.0)
    velocity = details["local_velocity"]
    height, width = fixed.shape[-2:]
    velocity_magnitude = _masked_mean(
        torch.linalg.vector_norm(velocity, dim=1, keepdim=True), fixed_mask
    ) / max(0.15 * min(height, width), 1.0)
    determinant = jacobian_determinant(details["fixed_to_moving_map"])
    topology = _masked_mean(torch.relu(0.05 - determinant), fixed_mask) + _masked_mean(
        (determinant <= 0.0).float(), fixed_mask
    )
    similarity = details["similarity_parameters"]
    scale = similarity.new_tensor(
        (math.radians(15.0), width * 0.05, height * 0.05, math.log(1.1))
    )
    similarity_magnitude = torch.sqrt(((similarity / scale).square()).mean(1))
    score = (
        SCORE_WEIGHTS["postwarp_mind_residual"] * mind
        + SCORE_WEIGHTS["outline_mismatch"] * (1.0 - outline_dice)
        + SCORE_WEIGHTS["normalized_velocity_magnitude"] * velocity_magnitude
        + SCORE_WEIGHTS["topology_penalty"] * topology
        + SCORE_WEIGHTS["normalized_similarity_magnitude"] * similarity_magnitude
    )
    return {
        "score": score,
        "mind": mind,
        "outline_dice": outline_dice,
        "common_support_fraction": common_support_fraction,
        "aligned_mask_dice": aligned_mask_dice,
        "velocity": velocity_magnitude,
        "topology": topology,
        "similarity": similarity_magnitude,
    }


@torch.inference_mode()
def score_candidates(
    model: JointPoseRegistrationModel,
    data: JointRegisteredData,
    batch: dict,
    poses: torch.Tensor,
    pose_features: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Render and score candidate planes with the exact registered-data contract."""
    fixed, fixed_mask, _ = data.render_pose(poses)
    aligned, aligned_mask, map_receipt = data.moving_for_fixed(batch, fixed_mask)
    fixed_input = _two_channel(fixed, fixed_mask)
    moving_input = _two_channel(aligned, aligned_mask)
    features = pose_features.expand(len(poses), -1)
    refined, logits, evidence = [], [], defaultdict(list)
    for start in range(0, len(poses), chunk_size):
        stop = min(start + chunk_size, len(poses))
        candidate_fixed = fixed_input[start:stop]
        candidate_moving = moving_input[start:stop]
        details = model.registrar.forward_with_details(candidate_fixed, candidate_moving)
        warped = warp_tensor(
            candidate_moving,
            details["fixed_to_moving_map"],
            padding_mode="border",
        )
        delta, logit = model.review_head(
            candidate_fixed,
            warped,
            poses[start:stop],
            features[start:stop],
            details["similarity_parameters"],
            details["local_velocity"],
        )
        refined.append(project_pose_to_domain(poses[start:stop] + delta))
        logits.append(logit)
        for name, value in _registration_evidence(
            candidate_fixed, candidate_moving, details
        ).items():
            evidence[name].append(value)
    result = {
        "refined_pose": torch.cat(refined).float().cpu().numpy(),
        "reviewer_logit": torch.cat(logits).float().cpu().numpy(),
        "source_to_aligned_h": map_receipt["source_to_aligned_h"].float().cpu().numpy(),
    }
    result.update(
        {f"registration_{name}": torch.cat(value).float().cpu().numpy() for name, value in evidence.items()}
    )
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("candidate scorer produced a non-finite value")
    return result


def _offset_records(true_pose: np.ndarray) -> list[dict]:
    records = []
    minimum = np.asarray((AP_MIN_UM, TILT_MIN_DEG, TILT_MIN_DEG), np.float64)
    maximum = np.asarray((AP_MAX_UM, TILT_MAX_DEG, TILT_MAX_DEG), np.float64)
    for axis_index, axis in enumerate(AXES):
        for magnitude in OFFSET_LEVELS[axis]:
            for sign in (-1, 1):
                offset = np.zeros(3, np.float64)
                offset[axis_index] = sign * magnitude
                pose = np.asarray(true_pose, np.float64) + offset
                records.append(
                    {
                        "axis": axis,
                        "sign": int(sign),
                        "magnitude": float(magnitude),
                        "offset": offset,
                        "pose": pose,
                        "in_domain": bool(np.all((minimum <= pose) & (pose <= maximum))),
                    }
                )
    return records


def _offset_contract() -> dict:
    return {
        "signed_one_axis_levels": OFFSET_LEVELS,
        "nearest_neighbor_levels": NEAREST_LEVEL,
        "resolvable_at_or_above": RESOLVABLE_LEVEL,
        "pairwise_truth_win": "strict margin > 0; zero margin is not a win",
        "top1": "candidate-zero-first argmax; a tie at the maximum counts as truth top-1",
        "candidate_domain": {
            "ap_um": [AP_MIN_UM, AP_MAX_UM],
            "tilt_deg": [TILT_MIN_DEG, TILT_MAX_DEG],
        },
    }


def _pose_key(pose: np.ndarray | torch.Tensor) -> tuple[float, float, float]:
    values = np.asarray(pose.detach().cpu() if torch.is_tensor(pose) else pose, np.float64)
    return tuple(np.round(values, 6).tolist())


def _E(error: np.ndarray) -> float:
    return float(np.sum(np.asarray(error, np.float64) / np.asarray(PHYSICAL_POSE_LOSS_SCALE)))


def _physical_plane_distances_um(
    data: JointRegisteredData,
    truth_pose: np.ndarray | torch.Tensor,
    candidate_poses: np.ndarray | torch.Tensor,
) -> np.ndarray:
    truth = torch.as_tensor(truth_pose, device=data.device, dtype=torch.float64).reshape(1, 3)
    candidates = torch.as_tensor(
        candidate_poses, device=data.device, dtype=torch.float64
    ).reshape(-1, 3)
    truth_ouv = pose_to_quicknii_ouv(truth)
    support = torch_annotation_brain_mask(
        truth_ouv,
        data.joint_synthetic_data.generator.annotation,
        QUICKNII_PIXEL_GRID_SHAPE,
    )
    repeated_truth = truth_ouv.expand(len(candidates), -1)
    repeated_support = support.expand(len(candidates), -1, -1)
    distance = torch_brain_masked_plane_distance(
        repeated_truth,
        pose_to_quicknii_ouv(candidates),
        repeated_support,
    )
    return (distance * 25.0).cpu().numpy()


def _pairwise_ce(margin: float) -> float:
    return float(np.logaddexp(0.0, -float(margin)))


def _slice_batch(batch: dict, item: int) -> dict:
    batch_size = len(batch["true_pose"])
    return {
        name: (value[item : item + 1] if torch.is_tensor(value) and value.ndim and len(value) == batch_size else value)
        for name, value in batch.items()
    }


def _variant_arrays(image: np.ndarray, mask: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if variant == "identity":
        return np.asarray(image).copy(), np.asarray(mask).copy()
    if variant == "horizontal":
        return np.fliplr(image).copy(), np.fliplr(mask).copy()
    if variant == "vertical":
        return np.flipud(image).copy(), np.flipud(mask).copy()
    if variant == "horizontal_vertical":
        return np.flip(image, axis=(0, 1)).copy(), np.flip(mask, axis=(0, 1)).copy()
    raise ValueError(f"unknown orientation variant: {variant}")


def _variant_batch(
    data: JointRegisteredData,
    position: int,
    base: dict,
    variant: str,
) -> tuple[dict, dict[str, str]]:
    dataset_index = data.record_indices[int(position)]
    image, mask = data._raw_image_and_mask(dataset_index)
    image, mask = _variant_arrays(image, mask, variant)
    moving, moving_mask = canonical_registration_image(image, mask)
    result = dict(base)
    result.update(
        pose_image=torch.from_numpy(preprocess_atlas_pose_image(image, mask))[None].to(data.device),
        moving=moving[None].to(data.device),
        moving_model_mask=moving_mask[None].to(data.device),
        moving_tissue_mask=moving_mask[None].to(data.device),
        moving_visible_mask=moving_mask[None].to(data.device),
    )
    return result, {
        "pose_input_sha256": _array_sha256(result["pose_image"].cpu().numpy()),
        "moving_input_sha256": _array_sha256(result["moving"].cpu().numpy()),
    }


def _top1(scores: list[float]) -> bool:
    return bool(scores and scores[0] >= max(scores[1:], default=-math.inf))


def _candidate_resolvable(offset: np.ndarray) -> bool:
    return bool(
        abs(float(offset[0])) >= RESOLVABLE_LEVEL["ap"]
        or abs(float(offset[1])) >= RESOLVABLE_LEVEL["lr"]
        or abs(float(offset[2])) >= RESOLVABLE_LEVEL["dv"]
    )


def _candidate_nearest(offset: np.ndarray) -> bool:
    nonzero = np.flatnonzero(np.asarray(offset) != 0.0)
    return bool(
        len(nonzero) == 1
        and math.isclose(
            abs(float(offset[nonzero[0]])),
            NEAREST_LEVEL[AXES[int(nonzero[0])]],
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    )


def _sample_top1(scores: list[float], offsets: list[np.ndarray]) -> dict[str, bool]:
    keep_not_nearest = [0] + [index for index, offset in enumerate(offsets[1:], 1) if not _candidate_nearest(offset)]
    keep_resolvable = [0] + [index for index, offset in enumerate(offsets[1:], 1) if _candidate_resolvable(offset)]
    return {
        "all": _top1(scores),
        "nearest_neighbor_excluded": _top1([scores[index] for index in keep_not_nearest]),
        "resolvable_only": _top1([scores[index] for index in keep_resolvable]),
    }


def _rankdata(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    ranked_x, ranked_y = _rankdata(x), _rankdata(y)
    if ranked_x.std() == 0.0 or ranked_y.std() == 0.0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def _within_specimen_monotonicity(
    rows: list[dict], axis: str, sign: int, margin_name: str, *, seed: int
) -> dict:
    by_specimen = defaultdict(dict)
    for row in rows:
        if row["axis"] == axis and row["sign"] == sign:
            by_specimen[int(row["specimen_id"])][float(row["magnitude"])] = float(
                row[margin_name]
            )
    correlations, monotonic, specimens = [], [], []
    point_counts = {specimen: len(values) for specimen, values in by_specimen.items()}
    for specimen in sorted(by_specimen):
        magnitudes = sorted(by_specimen[specimen])
        if len(magnitudes) < 3:
            continue
        margins = [by_specimen[specimen][magnitude] for magnitude in magnitudes]
        correlation = _spearman(magnitudes, margins)
        if correlation is not None:
            correlations.append(correlation)
            specimens.append(specimen)
        monotonic.append(all(right >= left for left, right in zip(margins, margins[1:])))
    counts = np.asarray(list(point_counts.values()), dtype=np.int64)
    full_lattice = set(OFFSET_LEVELS[axis])
    return {
        "axis": axis,
        "sign": sign,
        "specimen_count": len(by_specimen),
        "minimum_points_for_gate": 3,
        "eligible_specimen_count": len(monotonic),
        "full_lattice_specimen_count": sum(
            set(values) == full_lattice for values in by_specimen.values()
        ),
        "valid_magnitude_count_min_median_max": (
            [int(counts.min()), float(np.median(counts)), int(counts.max())]
            if len(counts)
            else [None, None, None]
        ),
        "per_specimen_valid_magnitude_count_sha256": _json_sha256(point_counts),
        "within_specimen_spearman": _clustered_summary(
            correlations, specimens, seed=seed
        ),
        "monotonic_specimen_fraction_wilson_95_ci": _wilson(
            sum(monotonic), len(monotonic)
        ),
    }


def _aggregate_pairs(rows: list[dict], metric: str) -> dict:
    margin_name = f"{metric}_margin"
    win_name = f"{metric}_truth_win"
    ce_name = f"{metric}_pairwise_ce"
    valid = [row for row in rows if row["candidate_in_domain"]]
    groups = defaultdict(list)
    for row in valid:
        groups[(row["axis"], int(row["sign"]), float(row["magnitude"]))].append(row)

    result = []
    for group_index, (key, values) in enumerate(sorted(groups.items())):
        margins = [float(value[margin_name]) for value in values]
        specimen_ids = [int(value["specimen_id"]) for value in values]
        wins = sum(bool(value[win_name]) for value in values)
        result.append(
            {
                "axis": key[0],
                "sign": key[1],
                "magnitude": key[2],
                "unit": AXIS_UNITS[key[0]],
                "truth_win_wilson_95_ci": _wilson(wins, len(values)),
                "margin": _clustered_summary(
                    margins, specimen_ids, seed=BOOTSTRAP_SEED + group_index
                ),
                "pairwise_ce": _clustered_summary(
                    [float(value[ce_name]) for value in values],
                    specimen_ids,
                    seed=BOOTSTRAP_SEED + 50 + group_index,
                ),
                "physical_corresponding_plane_distance_um": _clustered_summary(
                    [float(value["physical_corresponding_plane_distance_um"]) for value in values],
                    specimen_ids,
                    seed=BOOTSTRAP_SEED + 75 + group_index,
                ),
            }
        )

    strata = {}
    for stratum, predicate in (
        ("all", lambda row: True),
        ("nearest_neighbor", lambda row: math.isclose(row["magnitude"], NEAREST_LEVEL[row["axis"]])),
        ("sub_resolution", lambda row: row["magnitude"] < RESOLVABLE_LEVEL[row["axis"]]),
        ("resolvable", lambda row: row["magnitude"] >= RESOLVABLE_LEVEL[row["axis"]]),
    ):
        values = [row for row in valid if predicate(row)]
        margins = [float(value[margin_name]) for value in values]
        specimen_ids = [int(value["specimen_id"]) for value in values]
        strata[stratum] = {
            "descriptive_pair_wilson_95_ci": _wilson(
                sum(bool(value[win_name]) for value in values), len(values)
            ),
            "truth_win_specimen_cluster": _clustered_summary(
                [float(bool(value[win_name])) for value in values],
                specimen_ids,
                seed=BOOTSTRAP_SEED + 100 + len(strata),
            ),
            "margin": _clustered_summary(
                margins, specimen_ids, seed=BOOTSTRAP_SEED + 110 + len(strata)
            ),
            "pairwise_ce": _clustered_summary(
                [float(value[ce_name]) for value in values],
                specimen_ids,
                seed=BOOTSTRAP_SEED + 120 + len(strata),
            ),
            "physical_corresponding_plane_distance_um": _clustered_summary(
                [float(value["physical_corresponding_plane_distance_um"]) for value in values],
                specimen_ids,
                seed=BOOTSTRAP_SEED + 130 + len(strata),
            ),
        }

    monotonicity = [
        _within_specimen_monotonicity(
            valid,
            axis,
            sign,
            margin_name,
            seed=BOOTSTRAP_SEED + 200 + 2 * axis_index + (sign > 0),
        )
        for axis_index, axis in enumerate(AXES)
        for sign in (-1, 1)
    ]
    return {"by_axis_sign_magnitude": result, "strata": strata, "monotonicity": monotonicity}


def _top1_aggregate(sample_metrics: list[dict]) -> dict:
    result = {}
    for lattice in ("current", "exhaustive"):
        for metric in ("reviewer", "registration"):
            result[f"{lattice}_{metric}"] = {
                name: _wilson(
                    sum(bool(row[f"{lattice}_{metric}_{name}"]) for row in sample_metrics),
                    len(sample_metrics),
                )
                for name in ("all", "nearest_neighbor_excluded", "resolvable_only")
            }
    return result


def _registration_component_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["candidate_in_domain"]]
    specimens = [int(row["specimen_id"]) for row in valid]
    fields = tuple(REGISTRATION_COMPONENT_COLUMNS.values())
    return {
        field: {
            role: _clustered_summary(
                [float(row[f"{role}_{field}"]) for row in valid],
                specimens,
                seed=BOOTSTRAP_SEED + 400 + 2 * index + (role == "candidate"),
            )
            for role in ("truth", "candidate")
        }
        for index, field in enumerate(fields)
    }


def _registration_evidence_receipt() -> dict:
    return {
        "score_weights": dict(SCORE_WEIGHTS),
        "score_weights_sha256": _json_sha256(SCORE_WEIGHTS),
        "source_hash_normalization": "CRLF and CR normalized to LF",
        "premise_evaluator_source": str(
            Path(evaluate_joint_registration_premise.__file__).resolve()
        ),
        "premise_evaluator_source_sha256": _source_sha256(
            evaluate_joint_registration_premise.__file__
        ),
    }


def _current_execution_contract() -> dict:
    folder = Path(__file__).parent
    source_folder = folder.parent / "source"
    return {
        "source_sha256": {
            "trainer": _source_sha256(folder / "train_joint_pose_registration.py"),
            "model": _source_sha256(folder / "joint_pose_registration_model.py"),
            "synthetic_adapter": _source_sha256(folder / "joint_pose_registration_data.py"),
            "registered_adapter_and_canvas": _source_sha256(folder / "joint_registered_data.py"),
            "atlas_pose_models": _source_sha256(folder / "atlas_pose_models_v7.py"),
            "dense_registration_model": _source_sha256(folder / "dense_registration_model.py"),
            "dense_loss_ema_and_checkpoint": _source_sha256(folder / "train_dense_registration.py"),
            "atlas_pose_preprocessing": _source_sha256(source_folder / "atlas_pose_runtime.py"),
            "dense_registration_preprocessing": _source_sha256(
                source_folder / "dense_registration_preprocessing.py"
            ),
        },
        "preprocessing_contract": {
            "atlas_pose_version": ATLAS_POSE_PREPROCESSING_VERSION,
            "atlas_pose_sha256": ATLAS_POSE_PREPROCESSING_CONTRACT_SHA256,
            "dense_registration_version": PREPROCESSING_CONTRACT_V2,
            "dense_mask_sha256": MASK_CONTRACT_SHA256,
        },
    }


def _ouv_rederivation_receipt(rows: list[dict]) -> dict:
    maxima = np.asarray([row["absolute_pose_residual"] for row in rows]).max(axis=0)
    tolerances = np.asarray([OUV_FLOAT32_TOLERANCE[axis] for axis in OUV_RESIDUAL_AXES])
    component_pass = maxima <= tolerances
    return {
        "metadata_ouv_rederivation_max_absolute_residual": dict(
            zip(OUV_RESIDUAL_AXES, maxima.tolist())
        ),
        "metadata_ouv_rederivation_float32_tolerance": dict(OUV_FLOAT32_TOLERANCE),
        "metadata_ouv_rederivation_tolerance_rationale": (
            "Per-axis allowance for float32 metadata roundtrip only; 0.01 um and "
            "0.0001 deg are far below the atlas sampling and evaluated pose resolution."
        ),
        "metadata_ouv_rederivation_component_pass": dict(
            zip(OUV_RESIDUAL_AXES, component_pass.tolist())
        ),
        "metadata_ouv_rederivation_pass": bool(component_pass.all()),
        "metadata_ouv_rederivation_rows": rows,
    }


def _bind_current_data_contract(checkpoint_receipt: dict, data: JointRegisteredData) -> None:
    checkpoint_contract = checkpoint_receipt.get("checkpoint_generator_contract")
    if not isinstance(checkpoint_contract, dict):
        raise ValueError("reviewer-control checkpoint has no generator contract")
    expected_execution = {
        "source_sha256": checkpoint_contract.get("source_sha256"),
        "preprocessing_contract": checkpoint_contract.get("preprocessing_contract"),
    }
    current_execution = _current_execution_contract()
    if current_execution != expected_execution:
        raise ValueError(
            "current execution source or preprocessing contract differs from the control"
        )
    expected = {
        "atlas_generator": checkpoint_contract.get("synthetic"),
        "registered_validation": checkpoint_contract.get("registered_validation"),
    }
    current = {
        "atlas_generator": data.joint_synthetic_data.generator.contract,
        "registered_validation": data.contract,
    }
    expected_hashes = {name: _json_sha256(value) for name, value in expected.items()}
    current_hashes = {name: _json_sha256(value) for name, value in current.items()}
    if current_hashes != expected_hashes:
        raise ValueError("current atlas or Product-5 validation contract differs from the control")
    checkpoint_receipt["current_data_contract"] = {
        "comparison": (
            "exact source/preprocessing equality and canonical JSON equality with "
            "checkpoint synthetic and registered_validation"
        ),
        "source_sha256": current_execution["source_sha256"],
        "source_sha256_sha256": _json_sha256(current_execution["source_sha256"]),
        "preprocessing_contract": current_execution["preprocessing_contract"],
        "preprocessing_contract_sha256": _json_sha256(
            current_execution["preprocessing_contract"]
        ),
        "atlas_generator_sha256": current_hashes["atlas_generator"],
        "registered_validation_sha256": current_hashes["registered_validation"],
    }


def _orientation_variant_report(rows: list[dict]) -> dict:
    variants, registration_flags, reviewer_flags = {}, [], []
    for variant_index, variant in enumerate(VARIANTS[1:], 1):
        selected = [row for row in rows if row["variant"] == variant]
        specimens = [int(row["specimen_id"]) for row in selected]
        variants[variant] = {
            "registration_evidence_improvement": _clustered_summary(
                [row["registration_evidence_improvement_vs_identity"] for row in selected],
                specimens,
                seed=BOOTSTRAP_SEED + 500 + variant_index,
            ),
            "reviewer_margin_improvement": _clustered_summary(
                [row["reviewer_margin_improvement_vs_identity"] for row in selected],
                specimens,
                seed=BOOTSTRAP_SEED + 510 + variant_index,
            ),
        }
        registration_ci = variants[variant]["registration_evidence_improvement"][
            "specimen_cluster_bootstrap_mean_95_ci"
        ]
        reviewer_ci = variants[variant]["reviewer_margin_improvement"][
            "specimen_cluster_bootstrap_mean_95_ci"
        ]
        registration_flag = bool(registration_ci[0] is not None and registration_ci[0] > 0.0)
        reviewer_flag = bool(reviewer_ci[0] is not None and reviewer_ci[0] > 0.0)
        variants[variant]["frozen_registration_frame_flag"] = registration_flag
        variants[variant]["reviewer_variant_flag"] = reviewer_flag
        registration_flags.append(registration_flag)
        reviewer_flags.append(reviewer_flag)
    return {
        "variants": variants,
        "frozen_registration_frame_flag": any(registration_flags),
        "reviewer_variant_flag": any(reviewer_flags),
    }


def _raw_input_receipt(data: JointRegisteredData, positions: np.ndarray) -> list[dict]:
    records = []
    for position in positions:
        record = data.dataset.records[data.record_indices[int(position)]]
        path = data.root / record["relative_path"]
        records.append(
            {
                "position": int(position),
                "section_image_id": int(record["section_image_id"]),
                "specimen_id": int(record["specimen_id"]),
                "relative_path": str(record["relative_path"]),
                "raw_image_sha256": sha256_file(path),
            }
        )
    return records


def _product5_sampling_receipt(data: JointRegisteredData) -> dict:
    corpus = type(data.dataset)(
        data.root,
        data.atlas_folder,
        split=None,
        include_anatomy=False,
        allowed_product_ids=(5,),
    )
    records = corpus.records
    thickness = np.asarray(
        [
            float(corpus.datasets[int(record["experiment_id"])]["section_thickness_um"])
            for record in records
        ],
        dtype=np.float64,
    )
    by_specimen = defaultdict(list)
    for record in records:
        by_specimen[int(record["specimen_id"])].append(float(record["ap_um"]))
    spacing = np.concatenate(
        [
            np.abs(np.diff(np.sort(np.asarray(values, dtype=np.float64))))
            for values in by_specimen.values()
            if len(values) > 1
        ]
    )
    return {
        "role": "empirical full Product-5 development corpus sampling; not assumed label uncertainty",
        "section_count": len(records),
        "specimen_count": len(by_specimen),
        "section_thickness_um": {
            "unique": np.unique(thickness).tolist(),
            "median": float(np.median(thickness)),
        },
        "within_specimen_adjacent_ap_spacing_um": {
            "count": int(len(spacing)),
            "q05_median_q95": np.quantile(spacing, (0.05, 0.5, 0.95)).tolist(),
            "values_sha256": _array_sha256(spacing),
        },
    }


def load_development_ema(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    expected_completed_views: int = 2000,
) -> tuple[JointPoseRegistrationModel, dict, dict]:
    """Load the stopped development EMA without invoking the release loader."""
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", -1)) != JOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("development checkpoint format differs from the joint trainer")
    if int(payload.get("completed_views", -1)) != int(expected_completed_views):
        raise ValueError("development checkpoint is not the prespecified completed-view state")
    state = payload.get("ema", {}).get("shadow")
    if not isinstance(state, dict):
        raise ValueError("development checkpoint has no ema.shadow state")
    identity = _control_identity(payload, path)
    model = JointPoseRegistrationModel()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    receipt = {
        "role": "development_consumed_product5_offset_diagnostic",
        "release_claim": False,
        "checkpoint_path": str(path),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_format_version": int(payload["format_version"]),
        "completed_views": int(payload["completed_views"]),
        "selected_state": "ema.shadow",
        "state_selection": "explicit stopped-development EMA; release loader not used",
        "ema_state_sha256": identity["state_sha256"]["full_ema"],
        "control_identity": identity,
        "checkpoint_generator_contract": payload.get("generator_contract"),
    }
    return model, payload["config"], receipt


def _positive_determinant_frame_source(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(
        ((width - 1.0) / 2.0, (height - 1.0) / 2.0),
        FRAME_CONTROL_TRANSFORM["rotation_deg"],
        FRAME_CONTROL_TRANSFORM["scale"],
    )
    matrix[:, 2] += FRAME_CONTROL_TRANSFORM["translation_xy_pixels"]
    determinant = float(np.linalg.det(matrix[:, :2]))
    if determinant <= 0.0:
        raise ValueError("atlas self-render control affine must preserve handedness")
    transformed_image = cv2.warpAffine(
        np.asarray(image, np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    transformed_mask = cv2.warpAffine(
        np.asarray(mask, np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    asymmetry = {}
    for name, flipped_image, flipped_mask in (
        ("horizontal", np.fliplr(transformed_image), np.fliplr(transformed_mask)),
        ("vertical", np.flipud(transformed_image), np.flipud(transformed_mask)),
    ):
        support = transformed_mask | flipped_mask
        asymmetry[name] = float(
            np.abs(transformed_image - flipped_image)[support].mean()
        )
    return transformed_image, transformed_mask, {
        "source_to_control_affine": matrix.tolist(),
        "linear_determinant": determinant,
        **{name: list(value) if name == "translation_xy_pixels" else value for name, value in FRAME_CONTROL_TRANSFORM.items()},
        "measured_mean_absolute_reflection_asymmetry": asymmetry,
        "asymmetric_source": bool(min(asymmetry.values()) > 0.0),
    }


@torch.inference_mode()
def _atlas_frame_control(
    model: JointPoseRegistrationModel,
    data: JointRegisteredData,
    *,
    chunk_size: int,
) -> dict:
    truth = torch.tensor([FRAME_CONTROL_POSE], device=data.device, dtype=torch.float32)
    fixed, mask, _ = data.render_pose(truth)
    image = fixed[0, 0].cpu().numpy()
    source_mask = mask[0, 0].cpu().numpy()
    image, source_mask, affine_receipt = _positive_determinant_frame_source(
        image, source_mask
    )
    poses = torch.cat(
        (truth, truth + torch.tensor(FRAME_CONTROL_OFFSETS, device=data.device)), dim=0
    )
    rows = []
    for variant in VARIANTS:
        variant_image, variant_mask = _variant_arrays(image, source_mask, variant)
        moving = torch.from_numpy(variant_image)[None, None].to(data.device, dtype=torch.float32)
        moving_mask = torch.from_numpy(variant_mask)[None, None].to(data.device, dtype=torch.bool)
        batch = {
            "moving": moving,
            "moving_model_mask": moving_mask,
            "pose_image": torch.from_numpy(
                preprocess_atlas_pose_image(variant_image, variant_mask)
            )[None].to(data.device),
        }
        initialization = model.initialize(batch["pose_image"])
        scored = score_candidates(
            model,
            data,
            batch,
            poses,
            initialization["pose_features"],
            chunk_size=chunk_size,
        )
        reviewer = scored["reviewer_logit"].tolist()
        registration = (-scored["registration_score"]).tolist()
        rows.append(
            {
                "variant": variant,
                "reviewer_zero_offset_top1": _top1(reviewer),
                "registration_zero_offset_top1": _top1(registration),
                "reviewer_truth_logit": float(reviewer[0]),
                "registration_truth_evidence": float(-registration[0]),
                "mean_reviewer_margin": float(np.mean(np.asarray(reviewer[0]) - reviewer[1:])),
                "mean_registration_margin": float(np.mean(np.asarray(registration[0]) - registration[1:])),
                "moving_input_sha256": _array_sha256(moving.cpu().numpy()),
            }
        )
    return {
        "pose_ap_lr_dv": list(FRAME_CONTROL_POSE),
        "candidate_offsets": [list(value) for value in FRAME_CONTROL_OFFSETS],
        "asymmetric_render_sha256": _array_sha256(image),
        "source_control": affine_receipt,
        "variants": rows,
        "registration_identity_pass": bool(rows[0]["registration_zero_offset_top1"]),
        "reviewer_identity_pass": bool(rows[0]["reviewer_zero_offset_top1"]),
    }


@torch.inference_mode()
def evaluate_registered_candidate_offsets(
    model: JointPoseRegistrationModel,
    data: JointRegisteredData,
    *,
    checkpoint_receipt: dict,
    count: int,
    batch_size: int,
    seed: int,
    current_negatives: int,
    refinement_steps: int,
    chunk_size: int,
    orientation_variants: bool,
) -> tuple[list[dict], list[dict], dict]:
    """Evaluate the frozen reviewer, registrar evidence, and exact current lattice."""
    model.eval()
    positions = data.fixed_validation_positions(count, seed)
    input_rows = _raw_input_receipt(data, positions)
    input_by_section = {row["section_image_id"]: row for row in input_rows}
    rows, orientation_rows, sample_metrics, ouv_rows = [], [], [], []

    for ordinal, start in enumerate(range(0, count, batch_size)):
        current = data.batch_positions(
            positions[start : start + batch_size],
            (1 << 61) | (int(seed) << 20) | ordinal,
            current_negatives,
        )
        initialization = model.initialize(current["pose_image"])
        recurrent = recurrent_training_rollout(
            model,
            current,
            initialization,
            data.render_pose,
            refinement_steps=refinement_steps,
            live_initializer_fraction=1.0,
            gradient_checkpointing=False,
            prepare_moving=data.moving_for_fixed,
            compute_final_registration=False,
        )
        for item in range(len(current["true_pose"])):
            sample_index = start + item
            position = int(positions[sample_index])
            batch = _slice_batch(current, item)
            features = initialization["pose_features"][item : item + 1]
            true_pose = batch["true_pose"][0].cpu().numpy().astype(np.float64)
            offsets = _offset_records(true_pose)
            valid = [record for record in offsets if record["in_domain"]]
            current_pose = np.concatenate(
                (
                    true_pose[None],
                    batch["initial_pose"].cpu().numpy(),
                    batch["wrong_candidate_pose"][0].cpu().numpy(),
                )
            )
            union = [true_pose] + [record["pose"] for record in valid]
            for pose in current_pose[1:]:
                if _pose_key(pose) not in {_pose_key(value) for value in union}:
                    union.append(pose)
            union_tensor = torch.tensor(np.asarray(union), device=data.device, dtype=torch.float32)
            scored = score_candidates(
                model, data, batch, union_tensor, features, chunk_size=chunk_size
            )
            plane_distance = _physical_plane_distances_um(data, true_pose, union_tensor)
            refined_plane_distance = _physical_plane_distances_um(
                data, true_pose, scored["refined_pose"]
            )
            index = {_pose_key(pose): value for value, pose in enumerate(union)}
            truth_index = index[_pose_key(true_pose)]
            truth_logit = float(scored["reviewer_logit"][truth_index])
            truth_evidence = float(scored["registration_score"][truth_index])
            truth_refined_error = np.abs(scored["refined_pose"][truth_index] - true_pose)

            record = data.dataset.records[data.record_indices[position]]
            section_id = int(record["section_image_id"])
            specimen_id = int(record["specimen_id"])
            metadata_ouv = torch.tensor(record["quicknii_ouv"], dtype=torch.float64)
            rederived_pose = quicknii_ouv_to_pose(metadata_ouv).numpy()
            canonical_ouv = pose_to_quicknii_ouv(
                torch.tensor(true_pose, dtype=torch.float64)
            ).numpy()
            ouv_rows.append(
                {
                    "section_image_id": section_id,
                    "metadata_ouv": list(map(float, record["quicknii_ouv"])),
                    "stored_pose": true_pose.tolist(),
                    "rederived_pose": rederived_pose.tolist(),
                    "absolute_pose_residual": np.abs(rederived_pose - true_pose).tolist(),
                    "canonical_ouv_from_stored_pose": canonical_ouv.tolist(),
                }
            )

            for offset in offsets:
                base = {
                    "sample_index": sample_index,
                    "section_image_id": section_id,
                    "specimen_id": specimen_id,
                    "raw_image_sha256": input_by_section[section_id]["raw_image_sha256"],
                    "axis": offset["axis"],
                    "sign": offset["sign"],
                    "magnitude": offset["magnitude"],
                    "unit": AXIS_UNITS[offset["axis"]],
                    "offset_ap_um": float(offset["offset"][0]),
                    "offset_lr_deg": float(offset["offset"][1]),
                    "offset_dv_deg": float(offset["offset"][2]),
                    "candidate_in_domain": bool(offset["in_domain"]),
                    "truth_ap_um": float(true_pose[0]),
                    "truth_lr_deg": float(true_pose[1]),
                    "truth_dv_deg": float(true_pose[2]),
                    "candidate_ap_um": float(offset["pose"][0]),
                    "candidate_lr_deg": float(offset["pose"][1]),
                    "candidate_dv_deg": float(offset["pose"][2]),
                }
                if not offset["in_domain"]:
                    rows.append({**base, **{name: None for name in CSV_FIELDS if name not in base}})
                    continue
                candidate_index = index[_pose_key(offset["pose"])]
                candidate_logit = float(scored["reviewer_logit"][candidate_index])
                candidate_evidence = float(scored["registration_score"][candidate_index])
                reviewer_margin = truth_logit - candidate_logit
                registration_margin = candidate_evidence - truth_evidence
                candidate_refined_error = np.abs(
                    scored["refined_pose"][candidate_index] - true_pose
                )
                components = {
                    f"{role}_{column}": float(
                        scored[f"registration_{source}"][value_index]
                    )
                    for role, value_index in (
                        ("truth", truth_index),
                        ("candidate", candidate_index),
                    )
                    for source, column in REGISTRATION_COMPONENT_COLUMNS.items()
                }
                rows.append(
                    {
                        **base,
                        **components,
                        "reviewer_truth_logit": truth_logit,
                        "reviewer_candidate_logit": candidate_logit,
                        "reviewer_margin": reviewer_margin,
                        "reviewer_truth_win": reviewer_margin > 0.0,
                        "reviewer_pairwise_ce": _pairwise_ce(reviewer_margin),
                        "registration_truth_evidence": truth_evidence,
                        "registration_candidate_evidence": candidate_evidence,
                        "registration_margin": registration_margin,
                        "registration_truth_win": registration_margin > 0.0,
                        "registration_pairwise_ce": _pairwise_ce(registration_margin),
                        "physical_corresponding_plane_distance_um": float(
                            plane_distance[candidate_index]
                        ),
                        "candidate_refined_physical_plane_distance_um": float(
                            refined_plane_distance[candidate_index]
                        ),
                        "truth_refined_physical_plane_distance_um": float(
                            refined_plane_distance[truth_index]
                        ),
                        "candidate_initial_E": _E(np.abs(offset["offset"])),
                        "candidate_refined_E": _E(candidate_refined_error),
                        "candidate_refined_ap_error_um": float(candidate_refined_error[0]),
                        "candidate_refined_lr_error_deg": float(candidate_refined_error[1]),
                        "candidate_refined_dv_error_deg": float(candidate_refined_error[2]),
                        "truth_refined_E": _E(truth_refined_error),
                        "truth_refined_ap_error_um": float(truth_refined_error[0]),
                        "truth_refined_lr_error_deg": float(truth_refined_error[1]),
                        "truth_refined_dv_error_deg": float(truth_refined_error[2]),
                        "candidate_source_to_aligned_h": json.dumps(
                            scored["source_to_aligned_h"][candidate_index].tolist(),
                            separators=(",", ":"),
                        ),
                    }
                )

            current_indices = [index[_pose_key(pose)] for pose in current_pose]
            current_offsets = [np.zeros(3)] + [pose - true_pose for pose in current_pose[1:]]
            exhaustive_indices = [truth_index] + [index[_pose_key(record["pose"])] for record in valid]
            exhaustive_offsets = [np.zeros(3)] + [record["offset"] for record in valid]
            metric = {
                "sample_index": sample_index,
                "section_image_id": section_id,
                "specimen_id": specimen_id,
                "initializer_pose": initialization["pose"][item].cpu().tolist(),
                "initializer_absolute_error": np.abs(
                    initialization["pose"][item].cpu().numpy() - true_pose
                ).tolist(),
                "recurrent_pose": recurrent["pose"][item].cpu().tolist(),
                "recurrent_absolute_error": np.abs(
                    recurrent["pose"][item].cpu().numpy() - true_pose
                ).tolist(),
                "initializer_physical_corresponding_plane_distance_um": float(
                    _physical_plane_distances_um(
                        data, true_pose, initialization["pose"][item]
                    )[0]
                ),
                "recurrent_physical_corresponding_plane_distance_um": float(
                    _physical_plane_distances_um(data, true_pose, recurrent["pose"][item])[0]
                ),
            }
            for name, selected, selected_offsets in (
                ("current", current_indices, current_offsets),
                ("exhaustive", exhaustive_indices, exhaustive_offsets),
            ):
                for scorer, values in (
                    ("reviewer", scored["reviewer_logit"]),
                    ("registration", -scored["registration_score"]),
                ):
                    top1 = _sample_top1(
                        [float(values[value]) for value in selected], selected_offsets
                    )
                    metric.update(
                        {f"{name}_{scorer}_{key}": value for key, value in top1.items()}
                    )
            sample_metrics.append(metric)

            if orientation_variants:
                current_reviewer = scored["reviewer_logit"][current_indices]
                current_registration = -scored["registration_score"][current_indices]
                variant_values = [
                    {
                        "sample_index": sample_index,
                        "section_image_id": section_id,
                        "specimen_id": specimen_id,
                        "variant": "identity",
                        "pose_input_sha256": _array_sha256(batch["pose_image"].cpu().numpy()),
                        "moving_input_sha256": _array_sha256(batch["moving"].cpu().numpy()),
                        "initializer_ap_um": float(initialization["pose"][item, 0]),
                        "initializer_lr_deg": float(initialization["pose"][item, 1]),
                        "initializer_dv_deg": float(initialization["pose"][item, 2]),
                        "orientation_inverted_logit": float(
                            initialization["orientation_inverted_logit"][item]
                        ),
                        "true_registration_evidence": truth_evidence,
                        "mean_reviewer_margin": float(
                            np.mean(current_reviewer[0] - current_reviewer[1:])
                        ),
                        "mean_registration_margin": float(
                            np.mean(current_registration[0] - current_registration[1:])
                        ),
                    }
                ]
                for variant in VARIANTS[1:]:
                    variant_batch, hashes = _variant_batch(data, position, batch, variant)
                    variant_initialization = model.initialize(variant_batch["pose_image"])
                    variant_scored = score_candidates(
                        model,
                        data,
                        variant_batch,
                        torch.tensor(current_pose, device=data.device, dtype=torch.float32),
                        variant_initialization["pose_features"],
                        chunk_size=chunk_size,
                    )
                    reviewer_scores = variant_scored["reviewer_logit"]
                    registration_scores = -variant_scored["registration_score"]
                    output = {
                        "sample_index": sample_index,
                        "section_image_id": section_id,
                        "specimen_id": specimen_id,
                        "variant": variant,
                        **hashes,
                        "initializer_ap_um": float(variant_initialization["pose"][0, 0]),
                        "initializer_lr_deg": float(variant_initialization["pose"][0, 1]),
                        "initializer_dv_deg": float(variant_initialization["pose"][0, 2]),
                        "orientation_inverted_logit": float(
                            variant_initialization["orientation_inverted_logit"][0]
                        ),
                        "true_registration_evidence": float(-registration_scores[0]),
                        "mean_reviewer_margin": float(np.mean(reviewer_scores[0] - reviewer_scores[1:])),
                        "mean_registration_margin": float(np.mean(registration_scores[0] - registration_scores[1:])),
                    }
                    variant_values.append(output)
                identity = variant_values[0]
                for output in variant_values:
                    output["registration_evidence_improvement_vs_identity"] = (
                        identity["true_registration_evidence"] - output["true_registration_evidence"]
                    )
                    output["reviewer_margin_improvement_vs_identity"] = (
                        output["mean_reviewer_margin"] - identity["mean_reviewer_margin"]
                    )
                orientation_rows.extend(variant_values)

    valid_rows = [row for row in rows if row["candidate_in_domain"]]
    atlas_self_render = _atlas_frame_control(model, data, chunk_size=chunk_size)
    report = {
        "format_version": FORMAT_VERSION,
        "status": "development_diagnostic_complete",
        "interpretation": (
            "Development-consumed Product-5 rank-by-offset evidence only; this is not "
            "a release, external generalization benchmark, or permission to alter a locked gate."
        ),
        "checkpoint": checkpoint_receipt,
        "offset_contract": _offset_contract(),
        "panel_receipt": {
            "count": int(count),
            "seed": int(seed),
            "positions_sha256": _array_sha256(positions),
            "registered_contract": data.contract,
            "product5_sampling": _product5_sampling_receipt(data),
            "selected_inputs": input_rows,
            "selected_inputs_sha256": _json_sha256(input_rows),
        },
        "map_receipt": {
            "atlas_generator_contract": data.joint_synthetic_data.generator.contract,
            "render": "CCF average and annotation sampled at each physical AP/L-R/D-V candidate",
            "moving_preprocessing": (
                "candidate-specific detached outline PCA/orientation, isotropic span scaling, "
                "centering, and shared three-ring cosine feather"
            ),
            "dense_flow_target_used": False,
            "registration_evidence": _registration_evidence_receipt(),
            "physical_plane_distance_contract": QUICKNII_PLANE_DISTANCE_CONTRACT,
            "physical_plane_distance_contract_sha256": _json_sha256(
                QUICKNII_PLANE_DISTANCE_CONTRACT
            ),
            "candidate_homographies_sha256": _json_sha256(
                [
                    [
                        row["section_image_id"],
                        row["offset_ap_um"],
                        row["offset_lr_deg"],
                        row["offset_dv_deg"],
                        row["candidate_source_to_aligned_h"],
                    ]
                    for row in valid_rows
                ]
            ),
        },
        "frame_receipt": {
            **_ouv_rederivation_receipt(ouv_rows),
            "atlas_self_render": atlas_self_render,
        },
        "pair_count": len(rows),
        "scored_pair_count": len(valid_rows),
        "outside_domain_pair_count": len(rows) - len(valid_rows),
        "reviewer": _aggregate_pairs(rows, "reviewer"),
        "registration_evidence": _aggregate_pairs(rows, "registration"),
        "registration_evidence_components": _registration_component_summary(rows),
        "top1": _top1_aggregate(sample_metrics),
        "pose": {
            "sample_rows": sample_metrics,
            "initializer_mae_ap_lr_dv": np.mean(
                [row["initializer_absolute_error"] for row in sample_metrics], axis=0
            ).tolist(),
            "recurrent_mae_ap_lr_dv": np.mean(
                [row["recurrent_absolute_error"] for row in sample_metrics], axis=0
            ).tolist(),
            "initializer_physical_corresponding_plane_distance_mean_um": float(
                np.mean(
                    [
                        row["initializer_physical_corresponding_plane_distance_um"]
                        for row in sample_metrics
                    ]
                )
            ),
            "recurrent_physical_corresponding_plane_distance_mean_um": float(
                np.mean(
                    [
                        row["recurrent_physical_corresponding_plane_distance_um"]
                        for row in sample_metrics
                    ]
                )
            ),
        },
    }
    if orientation_rows:
        variant_report = _orientation_variant_report(orientation_rows)
        report["frame_receipt"]["real_source_orientation_variants"] = variant_report[
            "variants"
        ]
        report["frame_receipt"]["frozen_registration_frame_flag"] = variant_report[
            "frozen_registration_frame_flag"
        ]
        report["frame_receipt"]["reviewer_variant_flag"] = variant_report[
            "reviewer_variant_flag"
        ]
        report["frame_receipt"]["frame_integrity_pass"] = bool(
            report["frame_receipt"]["metadata_ouv_rederivation_pass"]
            and atlas_self_render["registration_identity_pass"]
            and not variant_report["frozen_registration_frame_flag"]
        )
    else:
        report["frame_receipt"]["frame_integrity_pass"] = None
        report["frame_receipt"]["frame_integrity_incomplete_reason"] = (
            "real H/V/HV frozen-registration controls were explicitly skipped"
        )
    return rows, orientation_rows, report


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    checkpoint_path: str | Path,
    output_folder: str | Path,
    *,
    device: str = "cuda",
    expected_completed_views: int = 2000,
    orientation_variants: bool = True,
) -> Path:
    set_determinism(4322)
    model, config, checkpoint_receipt = load_development_ema(
        checkpoint_path, device, expected_completed_views=expected_completed_views
    )
    count = int(config["registered_validation_count"])
    if count != 96:
        raise ValueError("the prespecified Product-5 offset diagnostic requires 96 sections")
    generator = SyntheticRegistrationGenerator(config["atlas"], torch.device(device))
    synthetic = JointSyntheticData(generator)
    data = JointRegisteredData(
        config["registered_root"], config["atlas"], synthetic, split="validation"
    )
    _bind_current_data_contract(checkpoint_receipt, data)
    rows, orientation_rows, report = evaluate_registered_candidate_offsets(
        model,
        data,
        checkpoint_receipt=checkpoint_receipt,
        count=count,
        batch_size=int(config["registered_validation_batch_size"]),
        seed=int(config["registered_validation_seed"]),
        current_negatives=int(config["validation_negatives_per_sample"]),
        refinement_steps=int(config["refinement_steps"]),
        chunk_size=int(config["candidate_chunk_size"]),
        orientation_variants=orientation_variants,
    )
    folder = Path(output_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    rows_path = folder / "pairs.csv"
    orientation_path = folder / "orientation-variants.csv"
    _write_csv(rows_path, CSV_FIELDS, rows)
    if orientation_rows:
        _write_csv(orientation_path, ORIENTATION_FIELDS, orientation_rows)
    report["output_receipt"] = {
        "evaluator_source_sha256": _source_sha256(__file__),
        "pairs_csv": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "orientation_variants_csv": (
            {"path": str(orientation_path), "sha256": sha256_file(orientation_path)}
            if orientation_rows
            else None
        ),
    }
    report_path = folder / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-completed-views", type=int, default=2000)
    parser.add_argument("--skip-orientation-variants", action="store_true")
    args = parser.parse_args()
    path = run(
        args.checkpoint,
        args.output,
        device=args.device,
        expected_completed_views=args.expected_completed_views,
        orientation_variants=not args.skip_orientation_variants,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
