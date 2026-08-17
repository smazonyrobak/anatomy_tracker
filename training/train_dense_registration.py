"""Train and validate the Allen slice dense-registration model."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from training.dense_registration_model import (
    DenseRegistrationModel,
    compose_pixel_maps,
    identity_pixel_map,
    jacobian_determinant,
    modality_independent_descriptor,
    resize_vector_field,
    warp_tensor,
)
from training.synthetic_registration import (
    STRATA,
    SyntheticRegistrationGenerator,
)


# Canonical dense v2 trainer; qualification and release remain in dense_registration_release.py.
DEFAULT_WORKSPACE = Path.home() / "AtlasWarpTraining"
DEFAULT_ATLAS = Path(__file__).resolve().parents[1] / "data" / "Allen Brain Atlas 25um"
FORMAT_VERSION = 1

LOSS_WEIGHTS = {
    "forward_flow": 1.0,
    "inverse_flow": 0.7,
    "similarity": 0.80,
    "deep_flow": 0.25,
    "regions": 0.20,
    "structure": 0.02,
    "smoothness": 0.015,
    "inverse_cycle": 0.05,
    "topology": 0.10,
}

FIXED_MODEL_PARAMETERS = {
    "integration_steps": 7,
    "maximum_rotation_degrees": 30.0,
    "maximum_translation_fraction": 0.10,
    "maximum_scale": 1.25,
    "maximum_local_velocity_fraction": 0.15,
}
LEGACY_V13_MODEL_PARAMETERS = {
    "input_channels": 2,
    "full_resolution_refiner": "one_shot",
    "registration_estimator": "residual5",
    "self_similarity_features": False,
}

def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def payload_sha256(payload: dict | list) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_batch_seed(data_seed: int, batch_ordinal: int) -> int:
    if not 0 <= int(data_seed) < 2**31 or not 0 <= int(batch_ordinal) < 2**32:
        raise ValueError("training seed and batch ordinal exceed the collision-free seed domain")
    return (int(data_seed) << 32) | int(batch_ordinal)


def evaluation_sample_seed(base_seed: int, stratum_index: int, sample_index: int) -> int:
    if not 0 <= int(base_seed) < 2**31:
        raise ValueError("evaluation seed exceeds the collision-free seed domain")
    if not 0 <= int(stratum_index) < 256 or not 0 <= int(sample_index) < 2**24:
        raise ValueError("evaluation sample ordinal exceeds the collision-free seed domain")
    return (
        (1 << 63)
        | (int(base_seed) << 32)
        | (int(stratum_index) << 24)
        | int(sample_index)
    )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state(data_rng: np.random.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "data_rng": copy.deepcopy(data_rng.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict, data_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    data_rng.bit_generator.state = state["data_rng"]
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def model_config(
    channels: tuple[int, ...],
    correlation_radii: tuple[int, ...],
) -> dict:
    channels = tuple(int(value) for value in channels)
    correlation_radii = tuple(int(value) for value in correlation_radii)
    if len(channels) != len(correlation_radii):
        raise ValueError("channels and correlation radii must contain the same number of stages")
    return {
        "channels": tuple(int(value) for value in channels),
        "correlation_radii": correlation_radii,
        **FIXED_MODEL_PARAMETERS,
    }


def canonical_model_config(config: dict, *, allow_legacy_v13: bool = False) -> dict:
    normalized = dict(config)
    legacy_keys = set(LEGACY_V13_MODEL_PARAMETERS)
    present_legacy = legacy_keys & set(normalized)
    if present_legacy:
        if (
            not allow_legacy_v13
            or present_legacy != legacy_keys
            or any(
                normalized[name] != expected
                for name, expected in LEGACY_V13_MODEL_PARAMETERS.items()
            )
        ):
            raise ValueError("legacy model config is not the selected v13 winner")
        for name in legacy_keys:
            normalized.pop(name)
    expected_keys = {"channels", "correlation_radii", *FIXED_MODEL_PARAMETERS}
    missing = expected_keys - set(normalized)
    unknown = set(normalized) - expected_keys
    if missing or unknown:
        raise ValueError(
            f"model config keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for name, expected in FIXED_MODEL_PARAMETERS.items():
        if normalized[name] != expected:
            raise ValueError(f"model config {name} must equal {expected}")
    return model_config(
        tuple(normalized["channels"]),
        tuple(normalized["correlation_radii"]),
    )


def build_model(config: dict, device: str | torch.device) -> DenseRegistrationModel:
    return DenseRegistrationModel(
        input_channels=2,
        **canonical_model_config(config),
    ).to(device)


# EMA weights are the validation and release candidate; optimizer weights exist for exact resume.
class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        current = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(current)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]


def similarity_parameters_from_homography(
    homography: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Recover [angle, tx, ty, log(scale)] used by the model's centre transform."""
    height, width = image_size
    matrix = homography[:, :2, :2]
    angle = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
    scale = torch.sqrt(torch.det(matrix).clamp_min(1e-12))
    centre = homography.new_tensor(((width - 1) / 2.0, (height - 1) / 2.0))
    translation = homography[:, :2, 2] - centre + torch.einsum(
        "bij,j->bi", matrix, centre
    )
    return torch.cat((angle[:, None], translation, torch.log(scale)[:, None]), dim=1)


def identity_training_batch(batch: dict) -> dict:
    """Return a geometry-identity warm-up batch without changing its atlas planes."""
    result = dict(batch)
    fixed = batch["fixed"]
    identity = identity_pixel_map(
        fixed.shape[0], fixed.shape[-2], fixed.shape[-1],
        device=fixed.device, dtype=fixed.dtype,
    )
    result.update(
        moving=fixed,
        moving_tissue_mask=batch["fixed_mask"],
        moving_damage_mask=torch.zeros_like(batch["fixed_mask"]),
        moving_visible_mask=batch["fixed_mask"],
        moving_model_mask=batch["fixed_mask"],
        fixed_damage_mask=torch.zeros_like(batch["fixed_mask"]),
        fixed_visible_mask=batch["fixed_mask"],
        moving_labels=batch["fixed_labels"],
        fixed_to_moving=identity,
        moving_to_fixed=identity,
        local_velocity=torch.zeros_like(identity),
        similarity_h=torch.eye(3, device=fixed.device)[None].repeat(fixed.shape[0], 1, 1),
    )
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def _robust_endpoint_loss(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = torch.sqrt((predicted - expected).square().sum(1) + 0.01)
    return _masked_mean(error, mask[:, 0] if mask.ndim == 4 else mask)


def label_boundary(labels: torch.Tensor) -> torch.Tensor:
    labels = labels[:, 0]
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary[:, None]


def internal_label_boundary(labels: torch.Tensor) -> torch.Tensor:
    """Allen-region boundaries excluding the outer tissue/background edge."""
    labels = labels[:, 0]
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = (
        (labels[:, :, 1:] != labels[:, :, :-1])
        & (labels[:, :, 1:] > 0)
        & (labels[:, :, :-1] > 0)
    )
    vertical = (
        (labels[:, 1:, :] != labels[:, :-1, :])
        & (labels[:, 1:, :] > 0)
        & (labels[:, :-1, :] > 0)
    )
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary[:, None]


def _training_masks(
    batch: dict,
    damage_flow_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["moving_model_mask"],
        batch["fixed_visible_mask"].float()
        + damage_flow_weight * batch["fixed_damage_mask"].float(),
        batch["moving_visible_mask"].float()
        + damage_flow_weight * batch["moving_damage_mask"].float(),
        batch["fixed_visible_mask"],
    )


def sample_integer_labels(labels: torch.Tensor, pixel_map: torch.Tensor) -> torch.Tensor:
    """Nearest-sample full Allen IDs without a lossy float conversion."""
    batch, _, height, width = labels.shape
    x = pixel_map[:, 0].round().long()
    y = pixel_map[:, 1].round().long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    linear = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).flatten(1)
    sampled = labels[:, 0].flatten(1).gather(1, linear).reshape(batch, 1, height, width)
    return torch.where(valid[:, None], sampled, 0)


def sampled_region_loss(
    fixed_labels: torch.Tensor,
    moving_labels: torch.Tensor,
    fixed_to_moving_map: torch.Tensor,
    valid_mask: torch.Tensor,
    maximum_regions: int = 12,
) -> torch.Tensor:
    """Differentiable soft Dice over full Allen IDs, sampled without one-hot expansion."""
    losses = []
    for item in range(fixed_labels.shape[0]):
        labels, counts = torch.unique(
            fixed_labels[item, 0][valid_mask[item, 0]], return_counts=True
        )
        keep = labels > 0
        labels, counts = labels[keep], counts[keep]
        if labels.numel() == 0:
            continue
        selected = labels[torch.argsort(counts, descending=True)[:maximum_regions]]
        moving_masks = (
            moving_labels[item : item + 1] == selected[None, :, None, None]
        ).float()
        fixed_masks = (
            fixed_labels[item : item + 1] == selected[None, :, None, None]
        ).float()
        predicted = warp_tensor(
            moving_masks,
            fixed_to_moving_map[item : item + 1],
            padding_mode="zeros",
        )
        valid = valid_mask[item : item + 1].float()
        intersection = (predicted * fixed_masks * valid).sum((-2, -1))
        denominator = ((predicted + fixed_masks) * valid).sum((-2, -1))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        point_loss = ((predicted - fixed_masks).square() * valid).sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1.0)
        losses.append((dice_loss + point_loss).mean())
    return torch.stack(losses).mean() if losses else fixed_to_moving_map.sum() * 0.0


def registration_loss(
    model: DenseRegistrationModel,
    batch: dict,
    weights: dict[str, float] | None = None,
    *,
    boundary_endpoint_weight: float = 1.0,
    damage_flow_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float], dict]:
    boundary_endpoint_weight = float(boundary_endpoint_weight)
    if not math.isfinite(boundary_endpoint_weight) or boundary_endpoint_weight < 1.0:
        raise ValueError("boundary_endpoint_weight must be finite and at least 1.0")
    damage_flow_weight = float(damage_flow_weight)
    if not math.isfinite(damage_flow_weight) or not 0.0 <= damage_flow_weight <= 1.0:
        raise ValueError("damage_flow_weight must be finite and between 0 and 1")
    weights = LOSS_WEIGHTS if weights is None else weights
    moving_model_mask, forward_flow_mask, inverse_flow_mask, visible_mask = (
        _training_masks(batch, damage_flow_weight)
    )
    fixed_input = torch.cat((batch["fixed"], batch["fixed_mask"].float()), dim=1)
    moving_input = torch.cat(
        (batch["moving"], moving_model_mask.float()), dim=1
    )
    details = model.forward_with_details(fixed_input, moving_input)
    forward = details["fixed_to_moving_map"]
    inverse = details["moving_to_fixed_map"]
    forward_endpoint_mask = forward_flow_mask * (
        1.0
        + (boundary_endpoint_weight - 1.0)
        * label_boundary(batch["fixed_labels"]).float()
    )
    inverse_endpoint_mask = inverse_flow_mask * (
        1.0
        + (boundary_endpoint_weight - 1.0)
        * label_boundary(batch["moving_labels"]).float()
    )
    target_similarity = similarity_parameters_from_homography(
        batch["similarity_h"], batch["fixed"].shape[-2:]
    )
    height, width = batch["fixed"].shape[-2:]
    similarity_scale = forward.new_tensor(
        (math.radians(15.0), width * 0.05, height * 0.05, math.log(1.1))
    )
    terms: dict[str, torch.Tensor] = {
        "forward_flow": _robust_endpoint_loss(
            forward, batch["fixed_to_moving"], forward_endpoint_mask
        ),
        "inverse_flow": _robust_endpoint_loss(
            inverse, batch["moving_to_fixed"], inverse_endpoint_mask
        ),
        "similarity": F.smooth_l1_loss(
            details["similarity_parameters"] / similarity_scale,
            target_similarity / similarity_scale,
        ),
    }
    deep = []
    pyramid_velocities = details["pyramid_velocities"]
    for velocity in pyramid_velocities:
        target = resize_vector_field(batch["local_velocity"], velocity.shape[-2:])
        target_mask = F.interpolate(
            forward_flow_mask, size=velocity.shape[-2:], mode="nearest"
        )
        deep.append(_robust_endpoint_loss(velocity, target, target_mask))
    terms["deep_flow"] = torch.stack(deep).mean()
    terms["regions"] = sampled_region_loss(
        batch["fixed_labels"], batch["moving_labels"], forward, visible_mask
    )
    fixed_descriptor = modality_independent_descriptor(batch["fixed"])
    moving_descriptor = modality_independent_descriptor(batch["moving"])
    warped_descriptor = warp_tensor(moving_descriptor, forward, padding_mode="border")
    terms["structure"] = _masked_mean(
        (fixed_descriptor - warped_descriptor).abs(), visible_mask
    )
    velocity = details["local_velocity"]
    terms["smoothness"] = (
        (velocity[..., 1:] - velocity[..., :-1]).square().mean()
        + (velocity[..., 1:, :] - velocity[..., :-1, :]).square().mean()
    )
    identity = identity_pixel_map(
        forward.shape[0], height, width, device=forward.device, dtype=forward.dtype
    )
    cycle = compose_pixel_maps(forward, inverse)
    terms["inverse_cycle"] = _robust_endpoint_loss(
        cycle, identity, batch["fixed_mask"]
    )
    determinant = jacobian_determinant(forward)
    terms["topology"] = _masked_mean(
        F.relu(0.05 - determinant).square(), batch["fixed_mask"]
    )
    total = sum(weights[name] * value for name, value in terms.items())
    scalars = {name: float(value.detach()) for name, value in terms.items()}
    scalars["total"] = float(total.detach())
    return total, scalars, details


def _boundary_statistics(expected: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    expected = np.asarray(expected, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if not expected.any() and not observed.any():
        return 1.0, 0.0
    if not expected.any() or not observed.any():
        return 0.0, float(math.hypot(*expected.shape))
    distance_to_observed = distance_transform_edt(~observed)
    distance_to_expected = distance_transform_edt(~expected)
    recall = float((distance_to_observed[expected] <= 2.0).mean())
    precision = float((distance_to_expected[observed] <= 2.0).mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    distance = 0.5 * (
        float(distance_to_observed[expected].mean())
        + float(distance_to_expected[observed].mean())
    )
    return f1, distance


def _sample_metrics(batch: dict, forward: torch.Tensor, inverse: torch.Tensor) -> list[dict]:
    fixed_labels = batch["fixed_labels"]
    warped_labels = sample_integer_labels(batch["moving_labels"], forward)
    recovered_fixed = compose_pixel_maps(forward, batch["moving_to_fixed"])
    analytic_labels = sample_integer_labels(fixed_labels, recovered_fixed)
    fixed_tissue = batch["fixed_mask"]
    moving_tissue = batch["moving_tissue_mask"]
    fixed_damage = batch["fixed_damage_mask"]
    moving_damage = batch["moving_damage_mask"]
    target_forward = batch["fixed_to_moving"]
    endpoint = torch.linalg.vector_norm(forward - target_forward, dim=1)
    inverse_endpoint = torch.linalg.vector_norm(
        inverse - batch["moving_to_fixed"], dim=1
    )
    identity = identity_pixel_map(
        forward.shape[0], forward.shape[-2], forward.shape[-1],
        device=forward.device, dtype=forward.dtype,
    )
    cycle = torch.linalg.vector_norm(compose_pixel_maps(forward, inverse) - identity, dim=1)
    reverse_cycle = torch.linalg.vector_norm(
        compose_pixel_maps(inverse, forward) - identity, dim=1
    )
    determinant = jacobian_determinant(forward)[:, 0]
    inverse_determinant = jacobian_determinant(inverse)[:, 0]
    expected_boundaries = internal_label_boundary(fixed_labels)
    observed_boundaries = internal_label_boundary(warped_labels)
    metrics = []
    for item in range(forward.shape[0]):
        mask = fixed_tissue[item, 0]
        expected = fixed_labels[item, 0][mask]
        observed = warped_labels[item, 0][mask]
        exact = float((expected == observed).float().mean()) if expected.numel() else 0.0
        analytic = analytic_labels[item, 0][mask]
        analytic_exact = (
            float((expected == analytic).float().mean()) if expected.numel() else 0.0
        )
        region_dice = []
        for region in torch.unique(torch.cat((expected, observed))):
            if int(region) == 0:
                continue
            expected_region = expected == region
            observed_region = observed == region
            denominator = int(expected_region.sum() + observed_region.sum())
            if denominator:
                region_dice.append(
                    2.0 * float((expected_region & observed_region).sum()) / denominator
                )
        endpoint_values = endpoint[item][mask]
        inverse_mask = moving_tissue[item, 0]
        inverse_endpoint_values = inverse_endpoint[item][inverse_mask]
        damage_endpoint_values = endpoint[item][fixed_damage[item, 0]]
        inverse_damage_endpoint_values = inverse_endpoint[item][moving_damage[item, 0]]
        cycle_values = cycle[item][mask]
        reverse_cycle_values = reverse_cycle[item][inverse_mask]
        tissue = fixed_tissue[item, 0]
        determinant_values = determinant[item][tissue]
        inverse_tissue = moving_tissue[item, 0]
        inverse_determinant_values = inverse_determinant[item][inverse_tissue]
        boundary_mask = mask.detach().cpu().numpy()
        boundary_f1, boundary_distance = _boundary_statistics(
            expected_boundaries[item, 0].detach().cpu().numpy() & boundary_mask,
            observed_boundaries[item, 0].detach().cpu().numpy() & boundary_mask,
        )
        metrics.append(
            {
                "foreground_correspondence": exact,
                "analytic_foreground_correspondence": analytic_exact,
                "macro_region_dice": float(np.mean(region_dice)) if region_dice else 1.0,
                "boundary_f1_2px": boundary_f1,
                "boundary_mean_distance_px": boundary_distance,
                "endpoint_values": endpoint_values.detach().float().cpu().numpy(),
                "inverse_endpoint_values": inverse_endpoint_values.detach().float().cpu().numpy(),
                "damage_endpoint_values": damage_endpoint_values.detach().float().cpu().numpy(),
                "inverse_damage_endpoint_values": (
                    inverse_damage_endpoint_values.detach().float().cpu().numpy()
                ),
                "endpoint_p95_px": float(torch.quantile(endpoint_values, 0.95)),
                "inverse_endpoint_p95_px": float(
                    torch.quantile(inverse_endpoint_values, 0.95)
                ),
                "damage_endpoint_p95_px": (
                    float(torch.quantile(damage_endpoint_values, 0.95))
                    if damage_endpoint_values.numel() else None
                ),
                "inverse_damage_endpoint_p95_px": (
                    float(torch.quantile(inverse_damage_endpoint_values, 0.95))
                    if inverse_damage_endpoint_values.numel() else None
                ),
                "inverse_cycle_values": cycle_values.detach().float().cpu().numpy(),
                "reverse_cycle_values": reverse_cycle_values.detach().float().cpu().numpy(),
                "fold_count": int((determinant_values <= 0.0).sum()),
                "jacobian_count": int(determinant_values.numel()),
                "jacobian_min": float(determinant_values.min()) if determinant_values.numel() else 1.0,
                "inverse_fold_count": int((inverse_determinant_values <= 0.0).sum()),
                "inverse_jacobian_count": int(inverse_determinant_values.numel()),
                "inverse_jacobian_min": (
                    float(inverse_determinant_values.min())
                    if inverse_determinant_values.numel() else 1.0
                ),
            }
        )
    return metrics


def summarize_metrics(samples: list[dict]) -> dict[str, float | int | bool | None]:
    def concatenated(name: str) -> np.ndarray:
        arrays = [np.asarray(sample[name]).reshape(-1) for sample in samples]
        nonempty = [array for array in arrays if array.size]
        return np.concatenate(nonempty) if nonempty else np.empty(0, np.float32)

    def quantile(values: np.ndarray, probability: float) -> float:
        return float(np.quantile(values, probability)) if values.size else 0.0

    endpoints = concatenated("endpoint_values")
    inverse_endpoints = concatenated("inverse_endpoint_values")
    damage_endpoints = concatenated("damage_endpoint_values")
    inverse_damage_endpoints = concatenated("inverse_damage_endpoint_values")
    cycles = concatenated("inverse_cycle_values")
    reverse_cycles = concatenated("reverse_cycle_values")
    fold_count = sum(sample["fold_count"] for sample in samples)
    jacobian_count = sum(sample["jacobian_count"] for sample in samples)
    inverse_fold_count = sum(sample["inverse_fold_count"] for sample in samples)
    inverse_jacobian_count = sum(
        sample["inverse_jacobian_count"] for sample in samples
    )
    def scalar_quantile(name: str, probability: float) -> float | None:
        values = [sample[name] for sample in samples if sample[name] is not None]
        return float(np.quantile(values, probability)) if values else None

    result: dict[str, float | int | bool | None] = {
        "sample_count": len(samples),
        "foreground_correspondence": float(np.mean([sample["foreground_correspondence"] for sample in samples])),
        "analytic_foreground_correspondence": float(np.mean([
            sample["analytic_foreground_correspondence"] for sample in samples
        ])),
        "macro_region_dice": float(np.mean([sample["macro_region_dice"] for sample in samples])),
        "boundary_f1_2px": float(np.mean([sample["boundary_f1_2px"] for sample in samples])),
        "boundary_mean_distance_px": float(np.mean([sample["boundary_mean_distance_px"] for sample in samples])),
        "sample_foreground_correspondence_q05": float(np.quantile(
            [sample["foreground_correspondence"] for sample in samples], 0.05
        )),
        "sample_macro_region_dice_q05": float(np.quantile(
            [sample["macro_region_dice"] for sample in samples], 0.05
        )),
        "endpoint_p50_px": quantile(endpoints, 0.50),
        "endpoint_p95_px": quantile(endpoints, 0.95),
        "endpoint_p99_px": quantile(endpoints, 0.99),
        "inverse_endpoint_p50_px": quantile(inverse_endpoints, 0.50),
        "inverse_endpoint_p95_px": quantile(inverse_endpoints, 0.95),
        "inverse_endpoint_p99_px": quantile(inverse_endpoints, 0.99),
        "damage_endpoint_p95_px": quantile(damage_endpoints, 0.95),
        "inverse_damage_endpoint_p95_px": quantile(
            inverse_damage_endpoints, 0.95
        ),
        "damage_pixel_count": int(damage_endpoints.size),
        "inverse_damage_pixel_count": int(inverse_damage_endpoints.size),
        "damaged_sample_count": sum(
            sample["damage_endpoint_p95_px"] is not None for sample in samples
        ),
        "inverse_damaged_sample_count": sum(
            sample["inverse_damage_endpoint_p95_px"] is not None for sample in samples
        ),
        "sample_endpoint_p95_q95_px": scalar_quantile(
            "endpoint_p95_px", 0.95
        ),
        "sample_inverse_endpoint_p95_q95_px": scalar_quantile(
            "inverse_endpoint_p95_px", 0.95
        ),
        "sample_damage_endpoint_p95_q95_px": scalar_quantile(
            "damage_endpoint_p95_px", 0.95
        ),
        "sample_inverse_damage_endpoint_p95_q95_px": scalar_quantile(
            "inverse_damage_endpoint_p95_px", 0.95
        ),
        "inverse_cycle_p95_px": quantile(cycles, 0.95),
        "reverse_cycle_p95_px": quantile(reverse_cycles, 0.95),
        "fold_count": fold_count,
        "fold_fraction": fold_count / max(jacobian_count, 1),
        "jacobian_min": float(min(sample["jacobian_min"] for sample in samples)),
        "inverse_fold_count": inverse_fold_count,
        "inverse_fold_fraction": inverse_fold_count / max(inverse_jacobian_count, 1),
        "inverse_jacobian_min": float(
            min(sample["inverse_jacobian_min"] for sample in samples)
        ),
    }
    return result


def _stack_pairs(pairs: list[dict]) -> dict:
    return {
        name: torch.cat([pair[name] for pair in pairs], dim=0)
        for name, value in pairs[0].items()
        if torch.is_tensor(value)
    }


def _cpu_tensor_pair(pair: dict) -> dict:
    return {
        name: value.detach().cpu()
        for name, value in pair.items()
        if torch.is_tensor(value)
    }


def _prepare_evaluation_records(
    generator: SyntheticRegistrationGenerator,
    split: str,
    samples_per_stratum: int,
    seed: int,
    strata: tuple[str, ...],
) -> list[dict]:
    records = []
    for stratum_index, stratum in enumerate(strata):
        for sample_index in range(samples_per_stratum):
            sample_seed = evaluation_sample_seed(seed, stratum_index, sample_index)
            manifest = generator.make_manifest(
                1,
                split,
                sample_seed,
                stratum,
            )
            descriptors = {}
            for name in ("moving_appearance_mode", "mask_offset_px"):
                if name in manifest:
                    value = np.asarray(manifest[name]).reshape(-1)[0]
                    descriptors[name] = value.item() if isinstance(value, np.generic) else value
            records.append(
                {
                    "stratum": stratum,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest": manifest,
                    **descriptors,
                }
            )
    return records


def _evaluation_record_descriptors(records: list[dict]) -> list[dict]:
    order = {name: index for index, name in enumerate(STRATA)}
    return sorted(
        [
            {
                key: record[key]
                for key in (
                    "stratum", "sample_index", "seed", "manifest_sha256",
                    "moving_appearance_mode", "mask_offset_px",
                )
                if key in record
            }
            for record in records
        ],
        key=lambda record: (order[record["stratum"]], record["sample_index"]),
    )


@torch.inference_mode()
def _evaluate_records(
    model: DenseRegistrationModel,
    generator: SyntheticRegistrationGenerator,
    records: list[dict],
    batch_size: int,
    strata: tuple[str, ...],
    *,
    pair_cache: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    parameter = next(model.parameters(), None)
    model_device = parameter.device if parameter is not None else torch.device("cpu")
    samples = {name: [] for name in strata}
    oracle_samples = {name: [] for name in strata}
    for offset in range(0, len(records), batch_size):
        selected = records[offset : offset + batch_size]
        if pair_cache is None:
            pairs = [generator.batch(record["manifest"]) for record in selected]
        else:
            pairs = []
            for record in selected:
                key = record["manifest_sha256"]
                if key not in pair_cache:
                    pair_cache[key] = _cpu_tensor_pair(
                        generator.batch(record["manifest"])
                    )
                pairs.append(pair_cache[key])
        pair = {
            name: value.to(model_device)
            for name, value in _stack_pairs(pairs).items()
        }
        fixed_input = torch.cat((pair["fixed"], pair["fixed_mask"].float()), dim=1)
        moving_input = torch.cat(
            (pair["moving"], pair["moving_model_mask"].float()), dim=1
        )
        forward, inverse = model(fixed_input, moving_input)
        predicted = _sample_metrics(pair, forward.float(), inverse.float())
        oracle = _sample_metrics(
            pair, pair["fixed_to_moving"], pair["moving_to_fixed"]
        )
        for record, predicted_sample, oracle_sample in zip(selected, predicted, oracle):
            for name in ("moving_appearance_mode", "mask_offset_px"):
                if name in record:
                    predicted_sample[name] = record[name]
                    oracle_sample[name] = record[name]
            samples[record["stratum"]].append(predicted_sample)
            oracle_samples[record["stratum"]].append(oracle_sample)
    per_stratum = {
        stratum: summarize_metrics(samples[stratum]) for stratum in strata
    }
    oracle_per_stratum = {
        stratum: summarize_metrics(oracle_samples[stratum]) for stratum in strata
    }
    def result(values: dict[str, list[dict]], per_stratum: dict) -> dict:
        all_samples = [sample for stratum in strata for sample in values[stratum]]
        report = {
            "per_stratum": per_stratum,
            "overall": summarize_metrics(all_samples),
        }
        for descriptor, output_name in (
            ("moving_appearance_mode", "appearance_subgroups"),
            ("mask_offset_px", "mask_offset_subgroups"),
        ):
            observed = sorted(
                {sample[descriptor] for sample in all_samples if descriptor in sample},
                key=str,
            )
            if observed:
                report[output_name] = {
                    str(value): summarize_metrics(
                        [sample for sample in all_samples if sample.get(descriptor) == value]
                    )
                    for value in observed
                }
        return report

    return result(samples, per_stratum), result(oracle_samples, oracle_per_stratum)


@torch.inference_mode()
# Development evaluation consumes fixed validation manifests and never selects on sealed-test samples.
def evaluate_model(
    model: DenseRegistrationModel,
    generator: SyntheticRegistrationGenerator,
    *,
    split: str = "validation",
    samples_per_stratum: int = 24,
    batch_size: int = 2,
    seed: int = 73001,
    pair_cache: dict[str, dict] | None = None,
) -> dict:
    if split != "validation":
        raise ValueError("development evaluation supports only the validation split")
    strata = tuple(STRATA)
    model.eval()
    records = _prepare_evaluation_records(
        generator,
        split,
        samples_per_stratum,
        seed,
        strata,
    )
    metrics, oracle = _evaluate_records(
        model,
        generator,
        records,
        batch_size,
        strata,
        pair_cache=pair_cache,
    )
    descriptors = _evaluation_record_descriptors(records)
    return {
        "format_version": FORMAT_VERSION,
        "split": split,
        "seed": seed,
        "samples_per_stratum": samples_per_stratum,
        **metrics,
        "oracle_ceiling": oracle,
        "evaluation_samples": descriptors,
        "evaluation_manifest_sha256": payload_sha256(descriptors),
    }


def validation_score(metrics: dict) -> float:
    overall = metrics["overall"]
    return (
        float(overall["foreground_correspondence"])
        + 0.15 * float(overall["macro_region_dice"])
        + 0.10 * float(overall["boundary_f1_2px"])
        - 0.005 * (
            float(overall["endpoint_p95_px"])
            + float(overall["inverse_endpoint_p95_px"])
        )
        - 0.002 * (
            float(overall["inverse_cycle_p95_px"])
            + float(overall["reverse_cycle_p95_px"])
        )
        - 10.0 * (
            float(overall["fold_fraction"])
            + float(overall["inverse_fold_fraction"])
        )
    )


def write_registration_qa(
    model: DenseRegistrationModel,
    pair: dict,
    path: str | Path,
    maximum_items: int = 3,
) -> Path:
    """Write fixed/moving/predicted-warp/label-disagreement panels."""
    model.eval()
    with torch.inference_mode():
        fixed_input = torch.cat((pair["fixed"], pair["fixed_mask"].float()), dim=1)
        moving_input = torch.cat(
            (pair["moving"], pair["moving_model_mask"].float()), dim=1
        )
        forward, _ = model(fixed_input, moving_input)
        warped = warp_tensor(pair["moving"], forward, padding_mode="zeros")
        predicted_labels = sample_integer_labels(pair["moving_labels"], forward)
    count = min(maximum_items, pair["fixed"].shape[0])
    height, width = pair["fixed"].shape[-2:]
    titles = ("fixed atlas", "moving slice", "model warp", "Allen-ID disagreement")
    montage = Image.new("RGB", (width * 4, count * (height + 24)), (0, 0, 0))
    draw = ImageDraw.Draw(montage)
    for row in range(count):
        fixed = pair["fixed"][row, 0].detach().cpu().clamp(0, 1).numpy()
        moving = pair["moving"][row, 0].detach().cpu().clamp(0, 1).numpy()
        aligned = warped[row, 0].detach().cpu().clamp(0, 1).numpy()
        valid = pair["fixed_mask"][row, 0].detach().cpu().numpy()
        disagreement = (
            (predicted_labels[row, 0] != pair["fixed_labels"][row, 0])
            .detach().cpu().numpy() & valid
        )
        panels = []
        for image in (fixed, moving, aligned):
            gray = (image * 255.0).astype(np.uint8)
            panels.append(np.repeat(gray[..., None], 3, axis=2))
        diagnostic = panels[0].copy()
        diagnostic[disagreement] = (255, 35, 35)
        panels.append(diagnostic)
        top = row * (height + 24)
        for column, (title, panel) in enumerate(zip(titles, panels)):
            left = column * width
            montage.paste(Image.fromarray(panel), (left, top + 24))
            draw.text((left + 5, top + 5), title, fill=(230, 230, 230))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination)
    return destination


def save_checkpoint(path: str | Path, state: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)


def _curriculum_stratum(
    completed_views: int,
    identity_warmup_views: int,
    total_views: int,
    rng: np.random.Generator,
) -> tuple[str, bool]:
    if completed_views < identity_warmup_views:
        return "clean", True
    progress = (completed_views - identity_warmup_views) / max(
        total_views - identity_warmup_views, 1
    )
    if progress < 0.20:
        probabilities = (0.65, 0.30, 0.05)
    elif progress < 0.55:
        probabilities = (0.30, 0.50, 0.20)
    else:
        probabilities = (0.20, 0.45, 0.35)
    return str(rng.choice(tuple(STRATA), p=probabilities)), False


def _cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# Resume captures every state that can change the next sample or optimizer update.
def _checkpoint_state(
    *,
    config: dict,
    model: DenseRegistrationModel,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    data_rng: np.random.Generator,
    step: int,
    batch_ordinal: int,
    completed_views: int,
    best_validation_score: float,
    validation_checkpoints_without_improvement: int,
    latest_validation: dict | None,
    generator_contract: dict,
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "config": config,
        "model_config": config["model"],
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(data_rng),
        "step": step,
        "batch_ordinal": batch_ordinal,
        "completed_views": completed_views,
        "best_validation_score": best_validation_score,
        "validation_checkpoints_without_improvement": (
            validation_checkpoints_without_improvement
        ),
        "latest_validation": latest_validation,
        "generator_contract": generator_contract,
    }


def _progress_line(progress: dict) -> str:
    percent = 100.0 * progress["completed_views"] / max(progress["total_views"], 1)
    eta_seconds = progress["eta_seconds"]
    eta = "--:--:--" if eta_seconds is None else time.strftime(
        "%H:%M:%S", time.gmtime(max(0.0, eta_seconds))
    )
    terms = progress["smoothed_terms"]
    return (
        f"[{percent:6.2f}%] {progress['completed_views']:,}/{progress['total_views']:,} "
        f"views | {progress['views_per_second']:.2f} views/s | ETA {eta} | "
        f"{progress['stratum']} | loss {terms.get('total', float('nan')):.4f} | "
        f"flow {terms.get('forward_flow', float('nan')):.3f}/{terms.get('inverse_flow', float('nan')):.3f} | "
        f"regions {terms.get('regions', float('nan')):.3f}"
    )


# Early stopping is validation-only; best-validation.pt is the handoff to qualification.
def train(config: dict) -> Path:
    """Train/resume one development run; checkpoint selection uses validation only."""
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    obsolete = {"generator_profile", "recurrent_sequence_supervision"} & set(config)
    if obsolete:
        raise ValueError(f"obsolete training config keys are not accepted: {sorted(obsolete)}")
    model_configuration = canonical_model_config(config["model"])
    damage_flow_weight = float(config.get("damage_flow_weight", 0.10))
    if not math.isfinite(damage_flow_weight) or not 0.0 <= damage_flow_weight <= 1.0:
        raise ValueError("damage_flow_weight must be finite and between 0 and 1")
    boundary_endpoint_weight = float(config.get("boundary_endpoint_weight", 1.0))
    if not math.isfinite(boundary_endpoint_weight) or boundary_endpoint_weight < 1.0:
        raise ValueError("boundary_endpoint_weight must be finite and at least 1.0")
    early_stopping_patience = int(config.get(
        "early_stopping_patience_checkpoints", 0
    ))
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience_checkpoints cannot be negative")
    set_determinism(int(config["seed"]))
    run_folder = Path(config["workspace"]) / "runs" / config["run_name"]
    run_folder.mkdir(parents=True, exist_ok=True)
    latest_path = run_folder / "latest.pt"
    config_path = run_folder / "config.json"
    normalized_config = json.loads(json.dumps(config))
    normalized_config.pop("resume", None)
    normalized_config["model"] = json.loads(json.dumps(model_configuration))
    initialization_path = normalized_config.get("initialize_from")
    if initialization_path is not None:
        initialization_path = str(Path(initialization_path).resolve())
        normalized_config["initialize_from"] = initialization_path
        normalized_config["initialize_from_sha256"] = sha256_file(
            initialization_path
        )
    normalized_config["boundary_endpoint_weight"] = boundary_endpoint_weight
    normalized_config["damage_flow_weight"] = damage_flow_weight
    normalized_config["early_stopping_patience_checkpoints"] = (
        early_stopping_patience
    )
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != normalized_config:
            raise ValueError("resume config differs from the run's immutable config.json")
    else:
        atomic_json(config_path, normalized_config)

    generator = SyntheticRegistrationGenerator(config["atlas"], device)
    model = build_model(model_configuration, device)
    if initialization_path is not None and not latest_path.is_file():
        load_ema_initialization(
            model, model_configuration, initialization_path
        )
    ema = ExponentialMovingAverage(model, config["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = math.ceil(config["total_views"] / config["batch_size"])
    warmup_steps = max(1, math.ceil(config["scheduler_warmup_views"] / config["batch_size"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(step, total_steps, warmup_steps),
    )
    amp_enabled = device.type == "cuda" and bool(config["amp"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled, init_scale=64.0, growth_interval=2000
    )
    data_rng = np.random.default_rng(config["data_seed"])
    step = completed_views = batch_ordinal = 0
    best_score = -math.inf
    validations_without_improvement = 0
    latest_validation = None
    validation_pair_cache: dict[str, dict] = {}
    if config.get("resume", True) and latest_path.is_file():
        checkpoint = load_checkpoint(latest_path, device)
        checkpoint_config = dict(checkpoint.get("config", {}))
        if checkpoint_config != normalized_config:
            raise ValueError(
                "resume checkpoint config differs from the immutable run config"
            )
        if checkpoint["generator_contract"] != generator.contract:
            raise ValueError("checkpoint generator contract differs from installed atlas")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"], data_rng)
        step = int(checkpoint["step"])
        batch_ordinal = int(checkpoint.get("batch_ordinal", step))
        completed_views = int(checkpoint["completed_views"])
        best_score = float(checkpoint["best_validation_score"])
        validations_without_improvement = int(
            checkpoint.get("validation_checkpoints_without_improvement", 0)
        )
        latest_validation = checkpoint.get("latest_validation")

    best_path = run_folder / "best-validation.pt"
    if (
        early_stopping_patience
        and validations_without_improvement >= early_stopping_patience
        and completed_views < config["total_views"]
    ):
        if not best_path.is_file():
            raise RuntimeError("early-stopped run is missing best-validation.pt")
        progress_path = run_folder / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["status"] = "early_stopped"
        progress["eta_seconds"] = 0.0
        progress["latest_validation"] = latest_validation
        atomic_json(progress_path, progress)
        return best_path

    started = time.monotonic()
    initial_views = completed_views
    next_checkpoint = (
        (completed_views // config["checkpoint_every_views"] + 1)
        * config["checkpoint_every_views"]
    )
    smoothed: dict[str, float] = {}
    last_progress = 0.0
    stopped_early = False
    model.train()
    while completed_views < config["total_views"]:
        count = min(config["batch_size"], config["total_views"] - completed_views)
        stratum, identity = _curriculum_stratum(
            completed_views,
            config["identity_warmup_views"],
            config["total_views"],
            data_rng,
        )
        batch_seed = training_batch_seed(config["data_seed"], batch_ordinal)
        batch_ordinal += 1
        batch = generator.generate(count, "train", batch_seed, stratum)
        if identity:
            batch = identity_training_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss, terms, _ = registration_loss(
                model,
                batch,
                boundary_endpoint_weight=boundary_endpoint_weight,
                damage_flow_weight=damage_flow_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = not amp_enabled or scaler.get_scale() >= scale_before
        if not optimizer_stepped:
            print("AMP overflow: update skipped and views not counted", flush=True)
            continue
        scheduler.step()
        ema.update(model)
        step += 1
        completed_views += count
        for name, value in terms.items():
            smoothed[name] = value if name not in smoothed else 0.95 * smoothed[name] + 0.05 * value

        now = time.monotonic()
        if now - last_progress >= config["progress_every_seconds"]:
            elapsed = max(now - started, 1e-6)
            rate = (completed_views - initial_views) / elapsed
            remaining = config["total_views"] - completed_views
            progress = {
                "run_name": config["run_name"],
                "status": "training",
                "step": step,
                "completed_views": completed_views,
                "total_views": config["total_views"],
                "stratum": "identity" if identity else stratum,
                "views_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "smoothed_terms": smoothed,
                "latest_validation": latest_validation,
            }
            atomic_json(run_folder / "progress.json", progress)
            print(_progress_line(progress), flush=True)
            last_progress = now

        if completed_views >= next_checkpoint or completed_views == config["total_views"]:
            with ema.applied(model):
                latest_validation = evaluate_model(
                    model,
                    generator,
                    samples_per_stratum=config["validation_samples_per_stratum"],
                    batch_size=config["validation_batch_size"],
                    seed=config["validation_seed"],
                    pair_cache=validation_pair_cache,
                )
                score = validation_score(latest_validation)
                qa_pair = generator.generate(2, "validation", config["validation_seed"] + 991, "hard")
                write_registration_qa(
                    model, qa_pair,
                    run_folder / "qa" / f"views-{completed_views:09d}.png",
                    maximum_items=2,
                )
            improved = score > best_score
            validations_without_improvement = (
                0 if improved else validations_without_improvement + 1
            )
            state = _checkpoint_state(
                config=normalized_config, model=model, ema=ema,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                data_rng=data_rng, step=step, batch_ordinal=batch_ordinal,
                completed_views=completed_views,
                best_validation_score=max(best_score, score),
                validation_checkpoints_without_improvement=(
                    validations_without_improvement
                ),
                latest_validation=latest_validation,
                generator_contract=generator.contract,
            )
            save_checkpoint(run_folder / "checkpoints" / f"views-{completed_views:09d}.pt", state)
            if improved:
                best_score = score
                save_checkpoint(best_path, state)
            save_checkpoint(latest_path, state)
            atomic_json(run_folder / "validation-latest.json", latest_validation)
            overall = latest_validation["overall"]
            print(
                "validation | "
                f"correspondence {overall['foreground_correspondence']:.4%} | "
                f"macro Dice {overall['macro_region_dice']:.4f} | "
                f"boundary F1 {overall['boundary_f1_2px']:.4f} | "
                f"EPE p95 {overall['endpoint_p95_px']:.3f}px | "
                f"folds {overall['fold_count']}",
                flush=True,
            )
            model.train()
            next_checkpoint += config["checkpoint_every_views"]
            if (
                early_stopping_patience
                and validations_without_improvement >= early_stopping_patience
                and completed_views < config["total_views"]
            ):
                stopped_early = True
                print(
                    "early stopping | validation score did not improve for "
                    f"{validations_without_improvement} checkpoints",
                    flush=True,
                )
                break

    completed_progress = json.loads((run_folder / "progress.json").read_text(encoding="utf-8"))
    completed_progress["status"] = "early_stopped" if stopped_early else "complete"
    completed_progress["step"] = step
    completed_progress["completed_views"] = completed_views
    completed_progress["views_per_second"] = (
        (completed_views - initial_views) / max(time.monotonic() - started, 1e-6)
    )
    completed_progress["eta_seconds"] = 0.0
    completed_progress["latest_validation"] = latest_validation
    atomic_json(run_folder / "progress.json", completed_progress)
    return run_folder / "best-validation.pt"


def model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    use_ema: bool = True,
) -> tuple[DenseRegistrationModel, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_model_configuration = canonical_model_config(
        checkpoint["model_config"],
        allow_legacy_v13=True,
    )
    model = build_model(checkpoint_model_configuration, device)
    if use_ema and "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"]["shadow"])
    else:
        model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def load_ema_initialization(
    model: DenseRegistrationModel,
    model_configuration: dict,
    checkpoint_path: str | Path,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    target_configuration = canonical_model_config(model_configuration)
    checkpoint_model_configuration = canonical_model_config(
        checkpoint["model_config"],
        allow_legacy_v13=True,
    )
    if checkpoint_model_configuration != target_configuration:
        raise ValueError("initialization checkpoint model architecture differs")
    shadow = checkpoint.get("ema", {}).get("shadow")
    if not isinstance(shadow, dict) or not shadow:
        raise ValueError("initialization checkpoint has no EMA weights")
    model.load_state_dict(shadow)


# Standalone validation is diagnostic and cannot promote or release a checkpoint.
def validate_checkpoint(
    checkpoint_path: str | Path,
    *,
    atlas: str | Path = DEFAULT_ATLAS,
    device: str = "cuda",
    samples_per_stratum: int = 48,
    batch_size: int = 2,
    seed: int = 73001,
) -> dict:
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    generator = SyntheticRegistrationGenerator(atlas, device)
    if checkpoint["generator_contract"] != generator.contract:
        raise ValueError("checkpoint and installed atlas generator contracts differ")
    return evaluate_model(
        model, generator, split="validation",
        samples_per_stratum=samples_per_stratum,
        batch_size=batch_size, seed=seed,
    )


def training_config_from_args(args) -> dict:
    channels = tuple(int(value) for value in args.channels.split(","))
    correlation_radii = (
        tuple(int(value) for value in args.correlation_radii.split(","))
        if args.correlation_radii else ()
    )
    if (
        not math.isfinite(args.boundary_endpoint_weight)
        or args.boundary_endpoint_weight < 1.0
    ):
        raise ValueError("boundary_endpoint_weight must be finite and at least 1.0")
    if args.early_stopping_patience_checkpoints < 0:
        raise ValueError("early_stopping_patience_checkpoints cannot be negative")
    config = {
        "format_version": FORMAT_VERSION,
        "workspace": str(Path(args.workspace).resolve()),
        "atlas": str(Path(args.atlas).resolve()),
        "run_name": args.run_name,
        "device": args.device,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "model": model_config(channels, correlation_radii),
        "total_views": args.total_views,
        "batch_size": args.batch_size,
        "identity_warmup_views": args.identity_warmup_views,
        "scheduler_warmup_views": args.scheduler_warmup_views,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "damage_flow_weight": args.damage_flow_weight,
        "boundary_endpoint_weight": args.boundary_endpoint_weight,
        "early_stopping_patience_checkpoints": (
            args.early_stopping_patience_checkpoints
        ),
        "ema_decay": args.ema_decay,
        "amp": not args.no_amp,
        "checkpoint_every_views": args.checkpoint_every_views,
        "progress_every_seconds": args.progress_every_seconds,
        "validation_samples_per_stratum": args.validation_samples_per_stratum,
        "validation_batch_size": args.validation_batch_size,
        "validation_seed": args.validation_seed,
        "resume": not args.no_resume,
    }
    if args.initialize_from:
        config["initialize_from"] = str(Path(args.initialize_from).resolve())
    return config


def boundary_endpoint_weight_argument(value: str) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 1.0:
        raise argparse.ArgumentTypeError("must be finite and at least 1.0")
    return weight


def damage_flow_weight_argument(value: str) -> float:
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train/resume using train+validation only")
    train_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    train_parser.add_argument("--atlas", default=str(DEFAULT_ATLAS))
    train_parser.add_argument("--run-name", default="dense-reg-v2-residual5")
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument(
        "--damage-flow-weight",
        type=damage_flow_weight_argument,
        default=0.10,
        help="endpoint/deep-flow weight on damaged tissue",
    )
    train_parser.add_argument("--channels", default="16,24,32,48")
    train_parser.add_argument(
        "--correlation-radii",
        default="4,3,2,2",
        help="comma-separated local matching radii in coarse-to-fine stage order",
    )
    train_parser.add_argument("--total-views", type=int, default=100_000)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--identity-warmup-views", type=int, default=2_000)
    train_parser.add_argument("--scheduler-warmup-views", type=int, default=2_000)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gradient-clip", type=float, default=2.0)
    train_parser.add_argument(
        "--boundary-endpoint-weight",
        type=boundary_endpoint_weight_argument,
        default=1.0,
        help="relative endpoint-loss weight on Allen-label boundary pixels (>= 1)",
    )
    train_parser.add_argument("--ema-decay", type=float, default=0.999)
    train_parser.add_argument("--checkpoint-every-views", type=int, default=5_000)
    train_parser.add_argument("--progress-every-seconds", type=float, default=5.0)
    train_parser.add_argument("--validation-samples-per-stratum", type=int, default=48)
    train_parser.add_argument("--validation-batch-size", type=int, default=2)
    train_parser.add_argument("--validation-seed", type=int, default=73_001)
    train_parser.add_argument(
        "--early-stopping-patience-checkpoints",
        type=int,
        default=0,
        help="stop after this many validation checkpoints without improvement (0 disables)",
    )
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--data-seed", type=int, default=18)
    train_parser.add_argument(
        "--initialize-from",
        help="start a new run from compatible EMA weights without resuming optimizer state",
    )
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--no-resume", action="store_true")

    validate_parser = commands.add_parser("validate", help="re-run the development validation set")
    validate_parser.add_argument("checkpoint")
    validate_parser.add_argument("--atlas", default=str(DEFAULT_ATLAS))
    validate_parser.add_argument("--device", default="cuda")
    validate_parser.add_argument("--samples-per-stratum", type=int, default=48)
    validate_parser.add_argument("--batch-size", type=int, default=2)
    validate_parser.add_argument("--seed", type=int, default=73_001)
    validate_parser.add_argument("--output")

    qa_parser = commands.add_parser("qa", help="write a prediction QA montage")
    qa_parser.add_argument("checkpoint")
    qa_parser.add_argument("output")
    qa_parser.add_argument("--atlas", default=str(DEFAULT_ATLAS))
    qa_parser.add_argument("--device", default="cuda")
    qa_parser.add_argument("--stratum", choices=tuple(STRATA), default="hard")
    qa_parser.add_argument("--count", type=int, default=3)
    qa_parser.add_argument("--seed", type=int, default=80_117)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        print(train(training_config_from_args(args)), flush=True)
    elif args.command == "validate":
        result = validate_checkpoint(
            args.checkpoint, atlas=args.atlas, device=args.device,
            samples_per_stratum=args.samples_per_stratum,
            batch_size=args.batch_size, seed=args.seed,
        )
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2), flush=True)
    elif args.command == "qa":
        model, checkpoint = model_from_checkpoint(args.checkpoint, args.device)
        generator = SyntheticRegistrationGenerator(args.atlas, args.device)
        if checkpoint["generator_contract"] != generator.contract:
            raise ValueError("checkpoint and installed atlas generator contracts differ")
        pair = generator.generate(args.count, "validation", args.seed, args.stratum)
        print(write_registration_qa(model, pair, args.output, args.count), flush=True)


if __name__ == "__main__":
    main()
