"""Cold-start training core for :mod:`training.independent_joint_model`.

The trainer deliberately owns no atlas data and no pretrained weights.  It
consumes the exact batches emitted by ``independent_joint_data`` and keeps pose
search, recurrent pose correction, and truth-plane dense registration as three
separately supervised operations.

Dense gradients train shared anatomy features, but this core does not yet feed
warp quality back into pose search; that closed-loop policy remains a later,
separately validated extension rather than a claimed property of this trainer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import training.independent_joint_data as independent_data
import training.independent_joint_model as independent_model
from training.independent_joint_model import (
    IndependentJointModel,
    identity_pixel_map,
    jacobian_determinant,
    project_affine_free_velocity,
    warp_tensor,
)


TRAINER_VERSION = 1
LEARNED_CHECKPOINT_DEPENDENCIES: tuple[str, ...] = ()
CURRICULUM_STREAMS = ("regular_synthetic", "high_tilt", "product5")
DEFAULT_CURRICULUM = (
    "regular_synthetic",
    "regular_synthetic",
    "high_tilt",
    "product5",
)
ACCURACY_POSE_SCALE = (100.0, 2.0, 2.0)
QUICKNII_ANCHORS_ML_DV_UM = (
    (0.0, 0.0),
    (-2500.0, 0.0),
    (2500.0, 0.0),
    (0.0, -2000.0),
    (0.0, 2000.0),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_hash(value) -> str | None:
    if value is None:
        return None
    value = str(value).lower().removeprefix("sha256:")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"not a normalized SHA-256: {value!r}")
    return value


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stream_contract_hashes(data_contract: dict) -> dict[str, str]:
    if set(data_contract) == set(CURRICULUM_STREAMS):
        result = {}
        for name in CURRICULUM_STREAMS:
            value = data_contract[name]
            result[name] = _normal_hash(
                value.get("contract_sha256") if isinstance(value, dict) else value
            )
            if result[name] is None:
                raise ValueError(f"missing SHA-256 for data stream {name}")
        return result
    if "contract_sha256" in data_contract:
        value = _normal_hash(data_contract["contract_sha256"])
        if value is None:
            raise ValueError("missing data contract SHA-256")
        return {name: value for name in CURRICULUM_STREAMS}
    raise ValueError(f"data contract must bind every stream: {CURRICULUM_STREAMS}")


def _model_constructor_contract(model: IndependentJointModel) -> dict:
    declared = getattr(model, "constructor_contract", None)
    state_schema = [
        (name, list(value.shape), str(value.dtype)) for name, value in model.state_dict().items()
    ]
    contract = {
        "class": type(model).__name__,
        "fully_qualified_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "architecture_family": getattr(model, "architecture_family", type(model).__name__),
        "architecture_source_sha256": _sha256(inspect.getfile(type(model))),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_schema_sha256": hashlib.sha256(
            json.dumps(state_schema, separators=(",", ":")).encode()
        ).hexdigest(),
        "module_graph_sha256": hashlib.sha256(repr(model).encode()).hexdigest(),
        "declared_constructor_contract": declared,
    }
    for name in (
        "integration_steps",
        "maximum_translation_pixels",
        "maximum_velocity_fraction",
        "uses_recurrent_state",
        "comparison_refinement_steps",
    ):
        if hasattr(model, name):
            contract[name] = getattr(model, name)
    if hasattr(model, "pyramid") and hasattr(model.pyramid, "channels"):
        contract["pyramid_channels"] = list(model.pyramid.channels)
    if hasattr(model, "maximum_pose_delta"):
        contract["maximum_pose_delta"] = model.maximum_pose_delta.detach().cpu().tolist()
    if hasattr(model, "log_scale_limits"):
        contract["minimum_scale"] = float(torch.exp(-model.log_scale_limits[0]).cpu())
        contract["maximum_scale"] = float(torch.exp(model.log_scale_limits[1]).cpu())
    if hasattr(model, "pose_delta_head") and hasattr(model.pose_delta_head, "in_features"):
        contract["hidden_channels"] = model.pose_delta_head.in_features
    if hasattr(model, "pose_head"):
        context = next(
            (layer for layer in model.pose_head.context if isinstance(layer, torch.nn.Linear)), None
        )
        if context is not None:
            contract["pose_context_features"] = context.out_features
    return contract


def training_lineage(
    model: IndependentJointModel,
    data_contract: dict,
    atlas_contract: dict,
    run_config: dict | None = None,
    initial_state_sha256: str | None = None,
) -> dict:
    """Return the complete normalized cold-start lineage stored in checkpoints."""
    dependencies = tuple(getattr(model, "learned_weight_dependencies", ()))
    if dependencies or getattr(model, "initialization", None) != "random":
        raise RuntimeError("independent training requires random initialization and no learned dependencies")
    stream_hashes = _stream_contract_hashes(data_contract)
    composite_data_hash = hashlib.sha256(
        json.dumps(stream_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lineage = {
        "trainer_version": TRAINER_VERSION,
        "initialization": "random",
        "learned_checkpoint_dependencies": [],
        "trainer_source_sha256": _sha256(__file__),
        "model_source_sha256": _sha256(independent_model.__file__),
        "data_source_sha256": _sha256(independent_data.__file__),
        "data_contract_sha256": composite_data_hash,
        "data_stream_contract_sha256": stream_hashes,
        "atlas_renderer_contract_sha256": _normal_hash(atlas_contract.get("contract_sha256")),
        "atlas_average_template_sha256": _normal_hash(atlas_contract.get("average_template_sha256")),
        "atlas_annotation_sha256": _normal_hash(atlas_contract.get("annotation_sha256")),
        "model_constructor": _model_constructor_contract(model),
        "data_contract_payload_sha256": hashlib.sha256(
            json.dumps(data_contract, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "run_config": run_config or {},
        "initial_state_sha256": _normal_hash(initial_state_sha256 or _state_sha256(model)),
    }
    lineage["run_config_sha256"] = hashlib.sha256(
        json.dumps(lineage["run_config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
    lineage["lineage_sha256"] = hashlib.sha256(payload).hexdigest()
    return lineage


def _row_seed(seed: int, counter: int, row: int) -> int:
    payload = f"independent-candidate-shuffle:{int(seed)}:{int(counter)}:{int(row)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def shuffle_candidates(batch: dict, seed: int, counter: int) -> dict:
    """Shuffle each BxC candidate lattice and move every aligned field together.

    The positive destination cycles over all positions as ``counter`` advances;
    the remaining candidates use a hash-seeded permutation.  Thus ranking cannot
    exploit a fixed positive index, while an interrupted run reproduces exactly.
    """
    poses = batch["candidate_pose"]
    batch_size, candidate_count = poses.shape[:2]
    old_target = batch["listwise_target_index"].detach().cpu().long()
    permutation = torch.empty(batch_size, candidate_count, dtype=torch.long)
    new_target = torch.empty(batch_size, dtype=torch.long)
    for row in range(batch_size):
        generator = torch.Generator().manual_seed(_row_seed(seed, counter, row))
        target = int(old_target[row])
        destination = (_row_seed(seed, 0, row) + int(counter)) % candidate_count
        negative = torch.tensor(
            [index for index in range(candidate_count) if index != target], dtype=torch.long
        )
        negative = negative[torch.randperm(len(negative), generator=generator)]
        available = [index for index in range(candidate_count) if index != destination]
        permutation[row, destination] = target
        permutation[row, available] = negative
        new_target[row] = destination

    result = dict(batch)
    permutation_device = permutation.to(poses.device)
    aligned = (
        "candidate_pose",
        "candidate_fixed_image",
        "candidate_fixed_mask",
        "candidate_fixed_labels",
        "candidate_in_training_domain",
        "candidate_dense_truth_valid",
        "listwise_positive_mask",
    )
    for name in aligned:
        if name not in batch:
            continue
        value = batch[name]
        index = permutation_device.reshape(
            batch_size, candidate_count, *([1] * (value.ndim - 2))
        ).expand_as(value)
        result[name] = value.gather(1, index)
    result["listwise_target_index"] = new_target.to(poses.device)
    result["listwise_positive_mask"] = F.one_hot(
        result["listwise_target_index"], candidate_count
    ).bool()
    result["candidate_permutation"] = permutation_device
    result["candidate_inverse_permutation"] = permutation_device.argsort(1)
    return result


def normalized_full_cholesky_nll(
    mean: torch.Tensor,
    target: torch.Tensor,
    cholesky: torch.Tensor,
    physical_scale: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL in normalized pose units using the complete Cholesky factor."""
    scale = physical_scale.to(device=mean.device, dtype=mean.dtype)
    error = ((target - mean) / scale).unsqueeze(-1)
    normalized_cholesky = cholesky / scale[None, :, None]
    whitened = torch.linalg.solve_triangular(normalized_cholesky, error, upper=False)
    log_determinant = 2.0 * torch.log(
        torch.diagonal(normalized_cholesky, dim1=-2, dim2=-1)
    ).sum(1)
    return 0.5 * (
        whitened.square().sum((1, 2))
        + log_determinant
        + mean.shape[1] * math.log(2.0 * math.pi)
    ).mean()


def initializer_pose_losses(
    initialization: dict[str, torch.Tensor],
    true_pose: torch.Tensor,
    model: IndependentJointModel,
) -> dict[str, torch.Tensor]:
    centers = (
        model.pose_head.ap_centers,
        model.pose_head.tilt_centers,
        model.pose_head.tilt_centers,
    )
    logits = (
        initialization["ap_logits"],
        initialization["lr_logits"],
        initialization["dv_logits"],
    )
    target_bin = [
        (true_pose[:, axis, None] - axis_centers[None]).abs().argmin(1)
        for axis, axis_centers in enumerate(centers)
    ]
    categorical = torch.stack(
        [F.cross_entropy(axis_logits, axis_target) for axis_logits, axis_target in zip(logits, target_bin)]
    ).mean()
    selected_centers = torch.stack(
        [axis_centers.index_select(0, axis_target) for axis_centers, axis_target in zip(centers, target_bin)],
        dim=1,
    )
    maximum = model.pose_head.maximum_residual.to(true_pose)
    sub_bin = F.smooth_l1_loss(
        initialization["continuous_residual"] / maximum,
        (true_pose - selected_centers) / maximum,
    )
    gaussian = normalized_full_cholesky_nll(
        initialization["pose"],
        true_pose,
        initialization["pose_cholesky"],
        true_pose.new_tensor(ACCURACY_POSE_SCALE),
    )
    anchor = F.smooth_l1_loss(
        _plane_anchor_ap(initialization["pose"]) / ACCURACY_POSE_SCALE[0],
        _plane_anchor_ap(true_pose) / ACCURACY_POSE_SCALE[0],
    )
    return {
        "initializer_categorical": categorical,
        "initializer_sub_bin": sub_bin,
        "initializer_gaussian_nll": gaussian,
        "initializer_plane_anchor": anchor,
    }


def _plane_anchor_ap(pose: torch.Tensor) -> torch.Tensor:
    anchors = pose.new_tensor(QUICKNII_ANCHORS_ML_DV_UM)
    lr = torch.tan(torch.deg2rad(pose[:, 1]))[:, None]
    dv = torch.tan(torch.deg2rad(pose[:, 2]))[:, None]
    return pose[:, 0, None] - lr * anchors[None, :, 0] - dv * anchors[None, :, 1]


def _render_pose(renderer, pose: torch.Tensor):
    ap_index = independent_data.BREGMA_AP_INDEX - pose[:, 0] / independent_data.VOXEL_UM
    return renderer.render_planes(ap_index, pose[:, 1], pose[:, 2])


def _flat_candidates(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def independent_joint_forward(
    model: IndependentJointModel,
    batch: dict,
    renderer,
    *,
    recurrent_steps: int = 3,
    ranking_temperature: float = 1.0,
) -> dict:
    """Run cached ranking, three cheap recurrent corrections, and routed dense decode."""
    if recurrent_steps != 3:
        raise ValueError("the frozen training recurrence uses exactly T=3")
    source_image = batch["source_image"]
    source_mask = batch["source_mask"]
    mask_available = batch["mask_available"]
    batch_size, candidate_count = batch["candidate_pose"].shape[:2]
    source_index = torch.arange(batch_size, device=source_image.device).repeat_interleave(
        candidate_count
    )

    # This is the sole source encoder call in the entire batch path.
    source_features = model.encode_source(source_image, source_mask, mask_available)
    initialization = model.pose_head(source_features)
    candidate_image = _flat_candidates(batch["candidate_fixed_image"])
    candidate_mask = _flat_candidates(batch["candidate_fixed_mask"])
    candidate_pose = _flat_candidates(batch["candidate_pose"])
    atlas_available = torch.ones(
        len(candidate_pose), 1, 1, 1, device=source_image.device, dtype=source_image.dtype
    )
    ranked = model.score_candidate_from_features(
        candidate_image,
        candidate_mask,
        atlas_available,
        candidate_pose,
        initialization["pose_context"],
        None,
        source_features,
        source_index,
    )
    ranking_logits = ranked["compatibility_logit"].reshape(batch_size, candidate_count)
    in_domain = batch["candidate_in_training_domain"].bool()
    target = batch["listwise_target_index"]
    if not torch.all(in_domain.gather(1, target[:, None])):
        raise RuntimeError("the listwise positive candidate must be inside the training domain")
    expected_positive = F.one_hot(target, candidate_count).bool()
    if not torch.equal(batch["listwise_positive_mask"].bool(), expected_positive):
        raise RuntimeError("listwise target index and singleton positive mask disagree")
    ranking_logits_masked = ranking_logits.masked_fill(~in_domain, -1e4)
    ranking_probability = torch.softmax(ranking_logits_masked / ranking_temperature, dim=1)
    # The truth-centred candidate lattice is listwise supervision only.  Using
    # its (symmetric) poses or hidden states to seed recurrence would reveal the
    # target even when every compatibility logit is uniform.
    current_pose = initialization["pose"]
    hidden = None

    recurrent = []
    single_source_index = torch.arange(batch_size, device=source_image.device)
    for _ in range(recurrent_steps):
        atlas_image, atlas_mask, _ = _render_pose(renderer, current_pose)
        step = model.score_candidate_from_features(
            atlas_image,
            atlas_mask,
            torch.ones(batch_size, 1, 1, 1, device=source_image.device, dtype=source_image.dtype),
            current_pose,
            initialization["pose_context"],
            hidden,
            source_features,
            single_source_index,
        )
        recurrent.append(step)
        current_pose, hidden = step["pose"], step["hidden_state"]

    # Rerender and rescore the settled pose, but do not apply the proposed delta.
    final_image, final_mask, final_labels = _render_pose(renderer, current_pose)
    final_receipt = model.score_candidate_from_features(
        final_image,
        final_mask,
        torch.ones(batch_size, 1, 1, 1, device=source_image.device, dtype=source_image.dtype),
        current_pose,
        initialization["pose_context"],
        hidden,
        source_features,
        single_source_index,
    )

    positive_dense = batch["candidate_dense_truth_valid"].gather(
        1, batch["listwise_target_index"][:, None]
    )[:, 0].bool()
    dense_valid = batch["dense_truth_valid"].bool()
    if not torch.equal(positive_dense, dense_valid):
        raise RuntimeError("dense truth must belong only to the shuffled positive candidate")
    if not torch.equal(
        batch["candidate_dense_truth_valid"].sum(1), dense_valid.to(torch.int64)
    ):
        raise RuntimeError("wrong-plane candidates may never carry dense truth")
    dense_index = dense_valid.nonzero(as_tuple=False).flatten()
    dense = None
    if len(dense_index):
        required = (
            "truth_fixed_image",
            "truth_fixed_mask",
            "truth_svf",
            "truth_fixed_to_source_map",
            "truth_source_to_fixed_map",
            "truth_similarity_parameters",
        )
        if any(name not in batch for name in required):
            raise RuntimeError("dense-valid samples require exact synthetic truth")
        positive_index = batch["listwise_target_index"].index_select(0, dense_index)
        row = torch.arange(len(dense_index), device=dense_index.device)
        positive_pose = batch["candidate_pose"].index_select(0, dense_index)[row, positive_index]
        positive_image = batch["candidate_fixed_image"].index_select(0, dense_index)[row, positive_index]
        positive_mask = batch["candidate_fixed_mask"].index_select(0, dense_index)[row, positive_index]
        if not torch.equal(positive_pose, batch["true_pose"].index_select(0, dense_index)):
            raise RuntimeError("dense supervision pose is not the listwise positive truth plane")
        if not torch.equal(positive_image, batch["truth_fixed_image"].index_select(0, dense_index)):
            raise RuntimeError("dense supervision image is not the listwise positive truth plane")
        if not torch.equal(positive_mask, batch["truth_fixed_mask"].index_select(0, dense_index)):
            raise RuntimeError("dense supervision mask is not the listwise positive truth plane")
        dense = model.refine_from_features(
            batch["truth_fixed_image"].index_select(0, dense_index),
            batch["truth_fixed_mask"].index_select(0, dense_index),
            torch.ones(len(dense_index), 1, 1, 1, device=source_image.device, dtype=source_image.dtype),
            batch["true_pose"].index_select(0, dense_index),
            initialization["pose_context"],
            None,
            source_features,
            dense_index,
        )

    return {
        "initialization": initialization,
        "source_features": source_features,
        "source_index": source_index,
        "ranking": ranked,
        "ranking_logits": ranking_logits,
        "ranking_logits_masked": ranking_logits_masked,
        "ranking_probability": ranking_probability,
        "recurrent": recurrent,
        "settled_pose": current_pose,
        "final_render_pose": current_pose,
        "final_atlas_image": final_image,
        "final_atlas_mask": final_mask,
        "final_atlas_labels": final_labels,
        "final_receipt": final_receipt,
        "dense": dense,
        "dense_sample_index": dense_index,
        "dense_binding_pose": batch["true_pose"].index_select(0, dense_index),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_smooth_l1(prediction, target, mask) -> torch.Tensor:
    return _masked_mean(F.smooth_l1_loss(prediction, target, reduction="none"), mask)


def _sampled_exact_regions(
    fixed_labels: torch.Tensor,
    source_labels: torch.Tensor,
    maximum_regions: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed_channels, source_channels, available = [], [], []
    for item in range(len(fixed_labels)):
        labels, counts = torch.unique(fixed_labels[item, 0][fixed_labels[item, 0] > 0], return_counts=True)
        labels = labels[counts.argsort(descending=True)[:maximum_regions]]
        padded = F.pad(labels, (0, maximum_regions - len(labels)))
        valid = torch.arange(maximum_regions, device=labels.device) < len(labels)
        fixed_channels.append((fixed_labels[item, 0] == padded[:, None, None]).float())
        source_channels.append((source_labels[item, 0] == padded[:, None, None]).float())
        available.append(valid)
    return torch.stack(fixed_channels), torch.stack(source_channels), torch.stack(available)


def _region_boundary(regions: torch.Tensor) -> torch.Tensor:
    dilation = F.max_pool2d(regions, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-regions, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0.0, 1.0)


def dense_registration_losses(output: dict, batch: dict, index: torch.Tensor) -> dict:
    """Exact synthetic-only registration losses for the teacher-forced truth plane."""
    select = lambda name: batch[name].index_select(0, index)
    forward_valid = select("truth_fixed_valid_mask").bool()
    inverse_valid = select("truth_source_valid_mask").bool()
    height, width = output["fixed_to_moving_map"].shape[-2:]
    map_scale = output["fixed_to_moving_map"].new_tensor((width - 1, height - 1))[
        None, :, None, None
    ]
    truth_forward = select("truth_fixed_to_source_map")
    truth_inverse = select("truth_source_to_fixed_map")
    map_forward = _masked_smooth_l1(
        output["fixed_to_moving_map"] / map_scale,
        truth_forward / map_scale,
        forward_valid,
    )
    map_inverse = _masked_smooth_l1(
        output["moving_to_fixed_map"] / map_scale,
        truth_inverse / map_scale,
        inverse_valid,
    )

    truth_similarity = select("truth_similarity_parameters")
    similarity_scale = output["similarity_parameters"].new_tensor((1.0, 1.0, width, height, 1.0))
    similarity = F.smooth_l1_loss(
        output["similarity_parameters"] / similarity_scale,
        truth_similarity / similarity_scale,
    )
    truth_velocity, _ = project_affine_free_velocity(select("truth_svf"))
    velocity_scale = float(min(height, width))
    svf = _masked_smooth_l1(
        output["stationary_velocity"] / velocity_scale,
        truth_velocity / velocity_scale,
        forward_valid,
    )
    affine_free = output["affine_velocity_coefficients"].square().mean()
    validity = F.binary_cross_entropy_with_logits(
        output["validity_logits"], forward_valid.to(output["validity_logits"].dtype)
    )

    velocity = output["stationary_velocity"] / velocity_scale
    smoothness = (
        (velocity[:, :, 1:] - velocity[:, :, :-1]).square().mean()
        + (velocity[:, :, :, 1:] - velocity[:, :, :, :-1]).square().mean()
    )
    identity = identity_pixel_map(
        len(index), height, width, device=velocity.device, dtype=velocity.dtype
    )
    forward_cycle = warp_tensor(output["moving_to_fixed_map"], output["fixed_to_moving_map"])
    inverse_cycle = warp_tensor(output["fixed_to_moving_map"], output["moving_to_fixed_map"])
    cycle = 0.5 * (
        _masked_smooth_l1(forward_cycle / map_scale, identity / map_scale, forward_valid)
        + _masked_smooth_l1(inverse_cycle / map_scale, identity / map_scale, inverse_valid)
    )

    predicted_jacobian = jacobian_determinant(output["fixed_to_moving_map"])
    truth_jacobian = jacobian_determinant(truth_forward)
    jacobian_mask = forward_valid[:, 0, 1:, 1:]
    jacobian = _masked_smooth_l1(predicted_jacobian, truth_jacobian, jacobian_mask)
    topology = _masked_mean(F.relu(0.05 - predicted_jacobian), jacobian_mask)

    fixed_regions, source_regions, region_available = _sampled_exact_regions(
        select("truth_fixed_labels"), select("truth_source_labels")
    )
    warped_regions = warp_tensor(
        source_regions, output["fixed_to_moving_map"], padding_mode="zeros"
    ).clamp(0.0, 1.0)
    spatial_mask = forward_valid.to(warped_regions.dtype)
    intersection = (warped_regions * fixed_regions * spatial_mask).sum((-2, -1))
    denominator = ((warped_regions + fixed_regions) * spatial_mask).sum((-2, -1))
    region_dice_per_channel = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    region_dice = (region_dice_per_channel * region_available).sum() / region_available.sum().clamp_min(1)
    fixed_boundary = _region_boundary(fixed_regions)
    source_boundary = _region_boundary(source_regions)
    warped_boundary = warp_tensor(
        source_boundary, output["fixed_to_moving_map"], padding_mode="zeros"
    )
    boundary_mask = forward_valid & region_available[:, :, None, None]
    region_boundary = _masked_smooth_l1(warped_boundary, fixed_boundary, boundary_mask)
    return {
        "dense_map_forward": map_forward,
        "dense_map_inverse": map_inverse,
        "dense_similarity": similarity,
        "dense_svf": svf,
        "dense_affine_free": affine_free,
        "dense_validity": validity,
        "dense_smoothness": smoothness,
        "dense_cycle": cycle,
        "dense_jacobian": jacobian,
        "dense_topology": topology,
        "dense_region_dice": region_dice,
        "dense_region_boundary": region_boundary,
    }


DEFAULT_LOSS_WEIGHTS = {
    "initializer_categorical": 1.0,
    "initializer_sub_bin": 0.5,
    "initializer_gaussian_nll": 0.05,
    "initializer_plane_anchor": 0.25,
    "candidate_ranking": 1.0,
    "recurrent_pose": 1.0,
    "recurrent_plane_anchor": 0.5,
    "final_compatibility": 0.1,
    "dense_map_forward": 1.0,
    "dense_map_inverse": 1.0,
    "dense_similarity": 0.5,
    "dense_svf": 0.5,
    "dense_affine_free": 0.05,
    "dense_validity": 0.2,
    "dense_smoothness": 0.05,
    "dense_cycle": 0.1,
    "dense_jacobian": 0.05,
    "dense_topology": 0.1,
    "dense_region_dice": 0.15,
    "dense_region_boundary": 0.1,
}


def independent_joint_loss(
    model: IndependentJointModel,
    output: dict,
    batch: dict,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = initializer_pose_losses(output["initialization"], batch["true_pose"], model)
    components["candidate_ranking"] = F.cross_entropy(
        output["ranking_logits_masked"], batch["listwise_target_index"]
    )
    pose_scale = batch["true_pose"].new_tensor(ACCURACY_POSE_SCALE)
    recurrent_weights = output["settled_pose"].new_tensor((0.5, 0.75, 1.0))
    recurrent_values = torch.stack(
        [
            F.smooth_l1_loss(step["pose"] / pose_scale, batch["true_pose"] / pose_scale)
            for step in output["recurrent"]
        ]
    )
    components["recurrent_pose"] = (
        recurrent_values * recurrent_weights
    ).sum() / recurrent_weights.sum()
    recurrent_anchor_values = torch.stack(
        [
            F.smooth_l1_loss(
                _plane_anchor_ap(step["pose"]) / ACCURACY_POSE_SCALE[0],
                _plane_anchor_ap(batch["true_pose"]) / ACCURACY_POSE_SCALE[0],
            )
            for step in output["recurrent"]
        ]
    )
    components["recurrent_plane_anchor"] = (
        recurrent_anchor_values * recurrent_weights
    ).sum() / recurrent_weights.sum()
    normalized_error = ((output["settled_pose"] - batch["true_pose"]) / pose_scale).square().sum(1)
    compatibility_target = torch.exp(-0.5 * normalized_error).detach()
    components["final_compatibility"] = F.binary_cross_entropy_with_logits(
        output["final_receipt"]["compatibility_logit"], compatibility_target
    )
    if output["dense"] is not None:
        components.update(
            dense_registration_losses(output["dense"], batch, output["dense_sample_index"])
        )
    selected_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        selected_weights.update(weights)
    total = sum(selected_weights[name] * value for name, value in components.items())
    return total, components


def raw_prediction_records(output: dict, batch: dict) -> list[dict]:
    """Return unaggregated animal/sample predictions with their raw provenance."""
    records = []
    batch_size = len(batch["true_pose"])
    tensor_ids = ("animal_id", "specimen_id", "experiment_id", "section_image_id", "product_id")
    for item in range(batch_size):
        record = {
            "source_type": str(batch.get("source_type", "unknown")),
            "data_split": str(batch.get("data_split", "unknown")),
            "sample_manifest_sha256": batch.get("sample_manifest_sha256"),
            "batch_manifest_sha256": batch.get("batch_manifest_sha256"),
            "record_provenance_sha256": (
                batch.get(
                    "record_provenance_sha256",
                    [batch.get("sample_manifest_sha256")] * batch_size,
                )[item]
            ),
            "data_contract_sha256": batch.get("data_contract_sha256"),
            "input_outline_receipt_sha256": (
                batch.get("input_outline_receipt_sha256", [None] * batch_size)[item]
            ),
            "source_relative_path": batch.get("source_relative_path", [None] * batch_size)[item]
            if "source_relative_path" in batch else None,
            "input_outline_mode": int(batch["input_outline_mode"][item].detach().cpu()),
            "mask_available": float(batch["mask_available"][item].detach().cpu().reshape(-1)[0]),
            "true_pose": batch["true_pose"][item].detach().cpu().tolist(),
            "initializer_pose": output["initialization"]["pose"][item].detach().cpu().tolist(),
            "initializer_cholesky": output["initialization"]["pose_cholesky"][item].detach().cpu().tolist(),
            "initializer_covariance": output["initialization"]["pose_covariance"][item].detach().cpu().tolist(),
            "initializer_axis_probability_uncalibrated": {
                "ap": output["initialization"]["ap_probability"][item].detach().cpu().tolist(),
                "lr": output["initialization"]["lr_probability"][item].detach().cpu().tolist(),
                "dv": output["initialization"]["dv_probability"][item].detach().cpu().tolist(),
            },
            "settled_pose": output["settled_pose"][item].detach().cpu().tolist(),
            "final_render_pose": output["final_render_pose"][item].detach().cpu().tolist(),
            "final_proposed_pose_not_applied": output["final_receipt"]["pose"][item].detach().cpu().tolist(),
            "candidate_pose": batch["candidate_pose"][item].detach().cpu().tolist(),
            "candidate_compatibility_logit": output["ranking_logits"][item].detach().cpu().tolist(),
            "candidate_score_softmax_uncalibrated": output["ranking_probability"][item].detach().cpu().tolist(),
            "candidate_in_training_domain": batch["candidate_in_training_domain"][item].detach().cpu().tolist(),
            "candidate_target_index": int(batch["listwise_target_index"][item].detach().cpu()),
            "final_compatibility_logit": float(
                output["final_receipt"]["compatibility_logit"][item].detach().cpu()
            ),
            "probabilities_calibrated": False,
        }
        for name in tensor_ids:
            if name in batch:
                record[name] = int(batch[name][item].detach().cpu())
        records.append(record)
    return records


def curriculum_batch(providers: dict, schedule, step: int, data_counter: int) -> tuple[str, dict]:
    """Select regular synthetic, high-tilt, or Product-5 without hiding counters."""
    schedule = tuple(schedule)
    if not schedule or any(name not in CURRICULUM_STREAMS for name in schedule):
        raise ValueError(f"curriculum must use only {CURRICULUM_STREAMS}")
    name = schedule[int(data_counter) % len(schedule)]
    if name not in providers:
        raise KeyError(f"missing curriculum provider {name!r}")
    return name, providers[name](int(step), int(data_counter))


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _update_ema(ema: dict[str, torch.Tensor], model: IndependentJointModel, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[name].copy_(value)


def _scheduled_learning_rate(
    update: int,
    total_updates: int,
    base_learning_rate: float,
    warmup_updates: int,
    minimum_fraction: float,
) -> float:
    if warmup_updates and update < warmup_updates:
        return base_learning_rate * float(update + 1) / float(warmup_updates)
    cosine_updates = max(total_updates - warmup_updates, 1)
    progress = min(max((update - warmup_updates) / cosine_updates, 0.0), 1.0)
    multiplier = minimum_fraction + (1.0 - minimum_fraction) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return base_learning_rate * multiplier


def _evaluate_ema(model, ema, evaluator, step):
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema)
    model.eval()
    try:
        with torch.no_grad():
            return evaluator(model, step)
    finally:
        model.load_state_dict(raw)
        model.train()


def _validate_development_panel(
    panel: dict,
    step: int,
    train_animal_ids,
    expected_contract_sha256: str | None = None,
) -> tuple[float, str]:
    partition = str(panel.get("partition", "")).lower()
    if partition not in {"development", "validation"}:
        raise RuntimeError("best-checkpoint selection is development/validation only")
    if panel.get("fresh_checkpoint_step") != step:
        raise RuntimeError("checkpoint selection requires a freshly evaluated panel")
    panel_hash = _normal_hash(panel.get("panel_manifest_sha256"))
    if expected_contract_sha256 is not None and _normal_hash(
        panel.get("panel_contract_sha256")
    ) != _normal_hash(expected_contract_sha256):
        raise RuntimeError("development panel contract differs from the frozen run config")
    animal_ids = {int(value) for value in panel.get("animal_ids", ())}
    if not animal_ids or animal_ids.intersection({int(value) for value in train_animal_ids}):
        raise RuntimeError("development animals must be nonempty and disjoint from training animals")
    metric = float(panel["selection_metric"])
    if not math.isfinite(metric):
        raise RuntimeError("development selection metric must be finite")
    raw = panel.get("raw_predictions")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("development panel must retain nonempty raw per-animal predictions")
    required = {
        "animal_id",
        "record_provenance_sha256",
        "candidate_score_softmax_uncalibrated",
        "initializer_covariance",
    }
    if any(not required.issubset(record) for record in raw):
        raise RuntimeError("development raw predictions lack animal/provenance/probability fields")
    for record in raw:
        _normal_hash(record["record_provenance_sha256"])
        if not isinstance(record["candidate_score_softmax_uncalibrated"], list) or not record[
            "candidate_score_softmax_uncalibrated"
        ]:
            raise RuntimeError("development raw candidate scores are empty")
        if not isinstance(record["initializer_covariance"], list) or not record[
            "initializer_covariance"
        ]:
            raise RuntimeError("development raw initializer covariance is empty")
    if {int(record["animal_id"]) for record in raw} != animal_ids:
        raise RuntimeError("development raw prediction animals differ from the panel manifest")
    return metric, panel_hash


def train_independent_joint(
    model: IndependentJointModel,
    renderer,
    providers: dict,
    data_contract: dict,
    checkpoint_folder: str | Path,
    steps: int,
    *,
    seed: int = 4322,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    amp: bool = True,
    ema_decay: float = 0.999,
    curriculum=DEFAULT_CURRICULUM,
    loss_weights: dict[str, float] | None = None,
    recurrent_steps: int = 3,
    warmup_steps: int = 1000,
    minimum_learning_rate_fraction: float = 0.05,
    gradient_clip_norm: float = 1.0,
    checkpoint_interval: int = 100,
    max_steps_this_call: int | None = None,
    resume: bool = True,
    development_evaluator=None,
    evaluate_every: int = 0,
    development_panel_contract_sha256: str | None = None,
    train_animal_ids=None,
) -> dict:
    """One compact AdamW+AMP+EMA loop with deterministic atomic resume.

    The caller constructs the randomly initialized model; its exact initial
    state receipt is frozen into lineage.  ``seed`` governs subsequent training
    and data order, rather than silently reinitializing architecture-specific
    heads after construction.
    """
    device = next(model.parameters()).device
    if recurrent_steps != 3:
        raise ValueError("the frozen training recurrence uses exactly T=3")
    if checkpoint_interval <= 0 or gradient_clip_norm <= 0:
        raise ValueError("checkpoint interval and gradient clip norm must be positive")
    if development_evaluator is not None and development_panel_contract_sha256 is None:
        raise ValueError("a frozen development-panel contract hash is required for selection")
    development_panel_contract_sha256 = _normal_hash(
        development_panel_contract_sha256
    )
    if set(data_contract) != set(CURRICULUM_STREAMS):
        raise ValueError(f"training requires an explicit contract for every stream: {CURRICULUM_STREAMS}")
    stream_contracts = _stream_contract_hashes(data_contract)
    frozen_product5_ids = {
        int(value) for value in data_contract["product5"].get("specimen_ids", ())
    }
    if "product5" in tuple(curriculum) or development_evaluator is not None:
        if not frozen_product5_ids:
            raise ValueError("the Product-5 contract must contain its nonempty training specimen_ids")
        supplied_ids = set() if train_animal_ids is None else {
            int(value) for value in train_animal_ids
        }
        if not supplied_ids or supplied_ids != frozen_product5_ids:
            raise ValueError(
                "train_animal_ids must exactly match the frozen Product-5 specimen_ids"
            )
    else:
        supplied_ids = set() if train_animal_ids is None else {
            int(value) for value in train_animal_ids
        }
    selected_loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if loss_weights:
        selected_loss_weights.update(loss_weights)
    run_config = {
        "seed": int(seed),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "amp_requested": bool(amp),
        "amp_enabled": bool(amp and device.type == "cuda"),
        "device_type": device.type,
        "ema_decay": float(ema_decay),
        "curriculum": list(curriculum),
        "loss_weights": selected_loss_weights,
        "recurrent_steps": int(recurrent_steps),
        "warmup_steps": int(warmup_steps),
        "minimum_learning_rate_fraction": float(minimum_learning_rate_fraction),
        "gradient_clip_norm": float(gradient_clip_norm),
        "checkpoint_interval": int(checkpoint_interval),
        "planned_steps": int(steps),
        "evaluate_every": int(evaluate_every),
        "development_panel_contract_sha256": development_panel_contract_sha256,
        "checkpoint_selection_state": "ema",
        "selection_direction": "minimize",
        "train_animal_ids": sorted(supplied_ids),
        "accuracy_pose_scale": list(ACCURACY_POSE_SCALE),
        "quicknii_anchors_ml_dv_um": [list(value) for value in QUICKNII_ANCHORS_ML_DV_UM],
    }
    checkpoint_folder = Path(checkpoint_folder)
    latest_path = checkpoint_folder / "latest.pt"
    best_path = checkpoint_folder / "best.pt"
    saved = (
        torch.load(latest_path, map_location=device, weights_only=False)
        if resume and latest_path.is_file()
        else None
    )
    initial_state_sha256 = (
        saved["lineage"]["initial_state_sha256"] if saved is not None else _state_sha256(model)
    )
    lineage = training_lineage(
        model, data_contract, renderer.contract, run_config, initial_state_sha256
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    step = 0
    data_counter = 0
    best_metric = math.inf
    evaluated_panels: list[str] = []

    if saved is not None:
        if saved["lineage"] != lineage or saved.get("learned_checkpoint_dependencies") != []:
            raise RuntimeError("resume lineage differs from this cold-start run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        ema = saved["ema"]
        step = int(saved["step"])
        data_counter = int(saved["data_counter"])
        best_metric = float(saved["best_metric"])
        evaluated_panels = list(saved["evaluated_panels"])
        _set_rng_state(saved["rng_state"])
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    last_loss = None
    last_records = []
    call_stop = int(steps)
    if max_steps_this_call is not None:
        call_stop = min(call_stop, step + int(max_steps_this_call))
    while step < call_stop:
        model.train()
        source_name, raw_batch = curriculum_batch(providers, curriculum, step, data_counter)
        if _normal_hash(raw_batch.get("data_contract_sha256")) != stream_contracts[source_name]:
            raise RuntimeError(f"{source_name} batch does not match its frozen data contract")
        if raw_batch.get("data_split") != "train":
            raise RuntimeError("training providers may not expose validation/calibration/final samples")
        if source_name == "product5":
            batch_animals = {int(value) for value in raw_batch["animal_id"].detach().cpu().tolist()}
            if not batch_animals or not batch_animals.issubset(frozen_product5_ids):
                raise RuntimeError("Product-5 batch animals differ from its frozen training contract")
        batch = shuffle_candidates(raw_batch, seed, data_counter)
        learning_rate_now = _scheduled_learning_rate(
            step, int(steps), learning_rate, warmup_steps, minimum_learning_rate_fraction
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_now
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = independent_joint_forward(
                model, batch, renderer, recurrent_steps=recurrent_steps
            )
            loss, components = independent_joint_loss(
                model, output, batch, weights=selected_loss_weights
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        _update_ema(ema, model, ema_decay)
        step += 1
        data_counter += 1
        last_loss = float(loss.detach().cpu())
        last_records = raw_prediction_records(output, batch)

        panel = None
        improved = False
        if development_evaluator is not None and evaluate_every and step % evaluate_every == 0:
            panel = _evaluate_ema(model, ema, development_evaluator, step)
            metric, panel_hash = _validate_development_panel(
                panel,
                step,
                supplied_ids,
                development_panel_contract_sha256,
            )
            if panel_hash in evaluated_panels:
                raise RuntimeError("development panel receipt was reused")
            evaluated_panels.append(panel_hash)
            improved = metric < best_metric
            if improved:
                best_metric = metric

        checkpoint = {
            "format": "independent-joint-cold-start-v1",
            "lineage": lineage,
            "learned_checkpoint_dependencies": [],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema,
            "step": step,
            "data_counter": data_counter,
            "rng_state": _rng_state(),
            "best_metric": best_metric,
            "evaluated_panels": evaluated_panels,
            "last_source": source_name,
            "last_loss": last_loss,
            "last_learning_rate": learning_rate_now,
            "last_gradient_norm": float(gradient_norm.detach().cpu()),
            "last_loss_components": {
                name: float(value.detach().cpu()) for name, value in components.items()
            },
            "last_raw_predictions": last_records,
            "development_panel": panel,
            "checkpoint_selection_state": "ema",
        }
        if panel is not None or step % checkpoint_interval == 0 or step == call_stop or step == int(steps):
            _atomic_save(checkpoint, latest_path)
        if improved:
            _atomic_save(checkpoint, best_path)

    return {
        "step": step,
        "data_counter": data_counter,
        "best_metric": best_metric,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path if best_path.is_file() else None,
        "last_loss": last_loss,
        "raw_predictions": last_records,
        "lineage": lineage,
    }
