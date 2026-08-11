"""Train and gate the post-pose, residual in-plane registration model.

Run as ``python -m training.train_diffeomorphic_registration``. Configuration
is read from ``DIFFEO_*`` environment variables; all generated artifacts stay
under ``J:/AtlasPoseDiffeomorphic`` by default. Nothing is promoted to the GUI.
"""

from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from training.diffeomorphic_registration_model import (
    DiffeomorphicRegistrationUNet,
    compose_pixel_maps,
    integrate_stationary_velocity,
    inverse_consistency_loss,
    jacobian_determinant,
    mind_loss,
    pixel_identity_grid,
    remove_global_affine,
    sample_at_pixel_map,
    smoothness_loss,
    synthetic_flow_loss,
    topology_loss,
)
from training.synthetic_atlas import AP_MAX_UM, AP_MIN_UM, BREGMA_AP_INDEX, VOXEL_UM, SyntheticAtlas, make_manifest


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "Allen Brain Atlas 25um"
DEFAULT_WORKSPACE = Path("J:/AtlasPoseDiffeomorphic")
PADDED_SIZE = (320, 464)
CLASS_COUNT = 9
VALIDATION_STRATA = ("identity_extreme", "smooth_deformation", "nuisance_damage", "wrong_ap")


def workspace_path() -> Path:
    return Path(os.environ.get("DIFFEO_WORKSPACE", str(DEFAULT_WORKSPACE)))


def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _rand(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device)


def _pad_to(tensor: torch.Tensor, shape: tuple[int, int] = PADDED_SIZE) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_y = shape[0] - height
    pad_x = shape[1] - width
    return F.pad(tensor, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))


def one_hot_labels(labels: torch.Tensor) -> torch.Tensor:
    return F.one_hot(labels[:, 0].long(), CLASS_COUNT).permute(0, 3, 1, 2).float()


def synthesize_modality(
    template: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator,
    *,
    extreme: bool = False,
) -> torch.Tensor:
    """Generate a label-conditioned arbitrary contrast while preserving geometry."""
    batch, _, height, width = template.shape
    device = template.device
    palette = 0.05 + 0.9 * _rand((batch, CLASS_COUNT, 1, 1), generator, device)
    regional = palette.expand(-1, -1, height, width).gather(1, labels.long())
    edge_x = F.pad((labels[:, :, :, 1:] != labels[:, :, :, :-1]).float(), (0, 1, 0, 0))
    edge_y = F.pad((labels[:, :, 1:, :] != labels[:, :, :-1, :]).float(), (0, 0, 0, 1))
    edges = F.max_pool2d(torch.maximum(edge_x, edge_y), 3, stride=1, padding=1)
    mix_low, mix_high = ((0.02, 0.30) if extreme else (0.15, 0.82))
    mix = mix_low + (mix_high - mix_low) * _rand((batch, 1, 1, 1), generator, device)
    edge_weight = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * (0.9 if extreme else 0.5)
    image = mix * template + (1.0 - mix) * regional + edge_weight * edges

    axis_y = torch.linspace(-1.0, 1.0, height, device=device)
    axis_x = torch.linspace(-1.0, 1.0, width, device=device)
    y, x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    coefficients = (_rand((batch, 5, 1, 1), generator, device) - 0.5) * (1.2 if extreme else 0.6)
    bias = (
        coefficients[:, 0:1] * x
        + coefficients[:, 1:2] * y
        + coefficients[:, 2:3] * x * y
        + coefficients[:, 3:4] * x.square()
        + coefficients[:, 4:5] * y.square()
    )
    image = image * torch.exp(bias)
    gamma_range = 2.2 if extreme else 1.2
    gamma = torch.exp((_rand((batch, 1, 1, 1), generator, device) - 0.5) * gamma_range)
    gain = torch.exp((_rand((batch, 1, 1, 1), generator, device) - 0.5) * gamma_range)
    offset = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * (0.7 if extreme else 0.3)
    image = (image.clamp(0.0, 1.0).pow(gamma) * gain + offset).clamp(0.0, 1.0)
    inverted = _rand((batch, 1, 1, 1), generator, device) < 0.5
    image = torch.where(inverted, 1.0 - image, image)
    background = _rand((batch, 1, 1, 1), generator, device) * (0.35 if extreme else 0.15)
    background = background + torch.randn(image.shape, generator=generator, device=device) * 0.015
    return torch.where(mask > 0.5, image, background.clamp(0.0, 1.0))


def sample_anatomical_velocity(
    mask: torch.Tensor,
    generator: torch.Generator,
    max_velocity_px: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an affine-free smooth SVF and shrink it until its discrete map has positive J."""
    batch, _, height, width = mask.shape
    device = mask.device
    low = torch.randn((batch, 2, 6, 8), generator=generator, device=device)
    velocity = F.interpolate(low, (height, width), mode="bicubic", align_corners=True)
    velocity = F.avg_pool2d(velocity, 11, stride=1, padding=5)
    support = F.avg_pool2d(mask.float(), 31, stride=1, padding=15).clamp(0.0, 1.0)
    velocity = remove_global_affine(velocity * support)
    amplitude = 1.5 + (max_velocity_px - 1.5) * _rand((batch, 1, 1, 1), generator, device)
    peak = velocity.abs().flatten(1).amax(dim=1).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    velocity = velocity * amplitude / peak
    for _ in range(8):
        forward = integrate_stationary_velocity(velocity, steps=7)
        shrink = torch.where(
            jacobian_determinant(forward).amin(dim=(1, 2)) < 0.20,
            torch.full((batch,), 0.65, device=device),
            torch.ones(batch, device=device),
        )
        velocity = velocity * shrink[:, None, None, None]
    velocity = remove_global_affine(velocity)
    return (
        velocity,
        integrate_stationary_velocity(velocity, steps=7),
        integrate_stationary_velocity(-velocity, steps=7),
    )


def add_nuisance_damage(
    image: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add non-geometric damage; these corruptions never alter the target map."""
    batch, _, height, width = image.shape
    device = image.device
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    x, y = x[None, None], y[None, None]
    offset = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * 1.2
    slope = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * 1.5
    curve = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * 0.6
    width_px = 0.007 + 0.025 * _rand((batch, 1, 1, 1), generator, device)
    signed_line = y - offset - slope * x - curve * x.square()
    tear_enabled = _rand((batch, 1, 1, 1), generator, device) < 0.65
    tear = (signed_line.abs() < width_px) & tear_enabled

    angle = _rand((batch, 1, 1, 1), generator, device) * (2.0 * math.pi)
    center_x, center_y = 0.88 * torch.cos(angle), 0.88 * torch.sin(angle)
    radius_x = 0.14 + 0.32 * _rand((batch, 1, 1, 1), generator, device)
    radius_y = 0.12 + 0.35 * _rand((batch, 1, 1, 1), generator, device)
    missing = (((x - center_x) / radius_x).square() + ((y - center_y) / radius_y).square() < 1.0)
    missing &= _rand((batch, 1, 1, 1), generator, device) < 0.60
    visible = (mask > 0.5) & ~tear & ~missing

    fold_width = width_px * 0.8
    fold_light = torch.exp(-((signed_line - width_px * 2.0) / fold_width).square())
    fold_dark = torch.exp(-((signed_line + width_px * 2.0) / fold_width).square())
    image = image + 0.55 * fold_light - 0.38 * fold_dark
    background = 0.02 + 0.25 * _rand((batch, 1, 1, 1), generator, device)
    image = torch.where(visible, image, background)

    period_x = 24.0 + 65.0 * _rand((batch, 1, 1, 1), generator, device)
    period_y = 24.0 + 65.0 * _rand((batch, 1, 1, 1), generator, device)
    pixel_x = (x + 1.0) * (width - 1.0) / 2.0
    pixel_y = (y + 1.0) * (height - 1.0) / 2.0
    fraction_x = torch.remainder(pixel_x / period_x, 1.0) - 0.5
    fraction_y = torch.remainder(pixel_y / period_y, 1.0) - 0.5
    vignette = 1.0 - (0.08 + 0.30 * _rand((batch, 1, 1, 1), generator, device)) * (
        fraction_x.square() + fraction_y.square()
    )
    seams = ((fraction_x.abs() > 0.47) | (fraction_y.abs() > 0.47)).float()
    image = image * vignette + seams * (_rand((batch, 1, 1, 1), generator, device) - 0.5) * 0.18

    density = 0.0003 + 0.003 * _rand((batch, 1, 1, 1), generator, device)
    specks = (_rand(image.shape, generator, device) < density).float()
    specks = F.max_pool2d(specks, 3, stride=1, padding=1)
    image = torch.maximum(image, specks * (0.75 + 0.25 * _rand((batch, 1, 1, 1), generator, device)))
    return image.clamp(0.0, 1.0), visible.float()


def make_synthetic_pair(
    template: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    seed: int,
    stratum: str,
    wrong_template: torch.Tensor | None = None,
    wrong_labels: torch.Tensor | None = None,
    wrong_mask: torch.Tensor | None = None,
    max_velocity_px: float = 7.0,
) -> dict[str, torch.Tensor]:
    """Construct an exact pair; only the known SVF contributes to its target flow."""
    device = template.device
    generator = _torch_generator(seed, device)
    batch, _, height, width = template.shape
    identity = pixel_identity_grid(batch, height, width, device=device, dtype=template.dtype)
    is_wrong = stratum == "wrong_ap"
    extreme = stratum == "identity_extreme"
    source_template = wrong_template if is_wrong else template
    source_labels = wrong_labels if is_wrong else labels
    source_mask = wrong_mask if is_wrong else mask
    if is_wrong and source_template is None:
        raise ValueError("wrong_ap pairs require a second, known-wrong plane")

    fixed = synthesize_modality(template, labels, mask, generator, extreme=extreme)
    moving_base = synthesize_modality(source_template, source_labels, source_mask, generator, extreme=extreme)
    if stratum in {"smooth_deformation", "nuisance_damage"}:
        target_velocity, atlas_to_affine, affine_to_atlas = sample_anatomical_velocity(
            mask, generator, max_velocity_px
        )
    else:
        target_velocity = torch.zeros_like(identity)
        atlas_to_affine = identity
        affine_to_atlas = identity

    moving = sample_at_pixel_map(moving_base, affine_to_atlas, padding_mode="zeros")
    moving_mask = sample_at_pixel_map(source_mask.float(), affine_to_atlas, padding_mode="zeros")
    moving_labels = sample_at_pixel_map(one_hot_labels(source_labels), affine_to_atlas, padding_mode="zeros")
    if stratum == "nuisance_damage":
        moving, moving_mask = add_nuisance_damage(moving, moving_mask, generator)
    return {
        "fixed": fixed,
        "moving": moving,
        "fixed_mask": mask.float(),
        "moving_mask": moving_mask.clamp(0.0, 1.0),
        "fixed_labels": one_hot_labels(labels),
        "moving_labels": moving_labels,
        "target_atlas_to_affine": atlas_to_affine,
        "target_affine_to_atlas": affine_to_atlas,
        "target_velocity": target_velocity,
        "wrong_pair": torch.full((batch,), is_wrong, device=device, dtype=torch.bool),
    }


class AllenObliquePairGenerator:
    """Exact Allen template/annotation planes; no pose or surface-affine augmentation."""

    def __init__(self, atlas_folder: str | Path = ATLAS, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.atlas = SyntheticAtlas(Path(atlas_folder), str(self.device))

    @staticmethod
    def _neutral_manifest(count: int, seed: int) -> dict[str, np.ndarray]:
        manifest = make_manifest(count, "train", seed)
        manifest["anatomy_mix"][:] = 0.0
        manifest["anatomy_edge_strength"][:] = 0.0
        return manifest

    def _render(self, manifest: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template, mask, labels = self.atlas._render(manifest, slice(0, len(manifest["ap_um"])), True)
        return _pad_to(template), _pad_to(mask.float()), _pad_to(labels.long())

    def batch(self, seed: int, count: int, stratum: str) -> dict[str, torch.Tensor]:
        manifest = self._neutral_manifest(count, seed)
        template, mask, labels = self._render(manifest)
        wrong_template = wrong_mask = wrong_labels = None
        moving_pose = np.column_stack(
            (manifest["ap_um"], manifest["tilt_lr_deg"], manifest["tilt_dv_deg"])
        ).astype(np.float32)
        if stratum == "wrong_ap":
            wrong_manifest = {name: np.array(value, copy=True) for name, value in manifest.items()}
            rng = np.random.default_rng(seed ^ 0x51CE)
            displacement = rng.uniform(500.0, 1500.0, count).astype(np.float32)
            direction = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), count)
            candidate = manifest["ap_um"] + displacement * direction
            outside = (candidate < AP_MIN_UM) | (candidate > AP_MAX_UM)
            candidate[outside] = manifest["ap_um"][outside] - displacement[outside] * direction[outside]
            wrong_manifest["ap_um"] = candidate.clip(AP_MIN_UM, AP_MAX_UM)
            wrong_manifest["ap_index"] = BREGMA_AP_INDEX - wrong_manifest["ap_um"] / VOXEL_UM
            wrong_template, wrong_mask, wrong_labels = self._render(wrong_manifest)
            moving_pose[:, 0] = wrong_manifest["ap_um"]
        pair = make_synthetic_pair(
            template,
            labels,
            mask,
            seed=seed ^ 0xD1FF30,
            stratum=stratum,
            wrong_template=wrong_template,
            wrong_labels=wrong_labels,
            wrong_mask=wrong_mask,
        )
        pair["fixed_pose"] = torch.from_numpy(
            np.column_stack((manifest["ap_um"], manifest["tilt_lr_deg"], manifest["tilt_dv_deg"]))
        ).to(self.device)
        pair["moving_pose"] = torch.from_numpy(moving_pose).to(self.device)
        pair["surface_affine"] = torch.eye(3, device=self.device)[:2].expand(count, -1, -1).clone()
        return pair


class RegistrationWithRejector(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.registration = DiffeomorphicRegistrationUNet(base_channels=base_channels)
        self.rejector = nn.Sequential(
            nn.Conv2d(4, base_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, 1),
        )

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atlas_to_affine, affine_to_atlas, velocity = self.registration(fixed, moving, fixed_mask, moving_mask)
        rejection_logit = self.rejector(torch.cat((fixed, moving, fixed_mask, moving_mask), dim=1))[:, 0]
        return atlas_to_affine, affine_to_atlas, velocity, rejection_logit


def hierarchical_label_dice_loss(
    fixed_labels: torch.Tensor,
    moving_labels: torch.Tensor,
    atlas_to_affine: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    warped = sample_at_pixel_map(moving_labels, atlas_to_affine, padding_mode="zeros")
    losses = []
    for scale in (1, 2, 4):
        fixed_level = fixed_labels if scale == 1 else F.avg_pool2d(fixed_labels, scale)
        warped_level = warped if scale == 1 else F.avg_pool2d(warped, scale)
        mask_level = mask if scale == 1 else F.avg_pool2d(mask, scale)
        intersection = (fixed_level[:, 1:] * warped_level[:, 1:] * mask_level).sum(dim=(-2, -1))
        denominator = ((fixed_level[:, 1:] + warped_level[:, 1:]) * mask_level).sum(dim=(-2, -1))
        present = denominator > 1.0
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        losses.append((1.0 - dice)[present].mean())
    return torch.stack(losses).mean()


def label_dice_score(
    fixed_labels: torch.Tensor,
    moving_labels: torch.Tensor,
    atlas_to_affine: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - hierarchical_label_dice_loss(fixed_labels, moving_labels, atlas_to_affine, mask)


def registration_objective(
    model: RegistrationWithRejector,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted, predicted_inverse, velocity, rejection_logit = model(
        batch["fixed"], batch["moving"], batch["fixed_mask"], batch["moving_mask"]
    )
    wrong = batch["wrong_pair"]
    rejection = F.binary_cross_entropy_with_logits(rejection_logit, wrong.float())
    zero = rejection * 0.0
    terms = {"mind": zero, "flow": zero, "dice": zero, "inverse": zero, "smooth": zero, "topology": zero}
    correct = ~wrong
    if bool(correct.any()):
        predicted_mask = sample_at_pixel_map(batch["moving_mask"][correct], predicted[correct], padding_mode="zeros")
        valid = batch["fixed_mask"][correct] * (predicted_mask > 0.5)
        terms["mind"] = mind_loss(
            batch["fixed"][correct], batch["moving"][correct], predicted[correct], valid
        )
        terms["flow"] = 0.5 * (
            synthetic_flow_loss(predicted[correct], batch["target_atlas_to_affine"][correct], valid)
            + synthetic_flow_loss(predicted_inverse[correct], batch["target_affine_to_atlas"][correct], valid)
        )
        terms["dice"] = hierarchical_label_dice_loss(
            batch["fixed_labels"][correct], batch["moving_labels"][correct], predicted[correct], valid
        )
        terms["inverse"] = inverse_consistency_loss(predicted[correct], predicted_inverse[correct], valid)
        terms["smooth"] = smoothness_loss(velocity[correct], batch["fixed_mask"][correct])
        terms["topology"] = topology_loss(predicted[correct], 0.05) + topology_loss(
            predicted_inverse[correct], 0.05
        )
    wrong_identity = zero
    if bool(wrong.any()):
        identity = pixel_identity_grid(
            int(wrong.sum()),
            predicted.shape[-2],
            predicted.shape[-1],
            device=predicted.device,
            dtype=predicted.dtype,
        )
        wrong_identity = synthetic_flow_loss(predicted[wrong], identity, batch["fixed_mask"][wrong])
    total = (
        terms["mind"]
        + 2.0 * terms["flow"]
        + terms["dice"]
        + 0.2 * terms["inverse"]
        + 0.05 * terms["smooth"]
        + 5.0 * terms["topology"]
        + 0.75 * rejection
        + 0.5 * wrong_identity
    )
    terms.update({"rejection": rejection, "wrong_identity": wrong_identity})
    return total, terms


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.lerp_(source[name], 1.0 - self.decay)
            else:
                value.copy_(source[name])


def _masked_percentile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> float:
    selected = values[mask.expand_as(values) > 0.5]
    return float(torch.quantile(selected.float(), quantile)) if selected.numel() else float("nan")


@torch.inference_mode()
def validation_report(
    model: RegistrationWithRejector,
    generator: AllenObliquePairGenerator,
    batches_per_stratum: int,
    batch_size: int,
) -> dict:
    model.eval()
    strata = {}
    for stratum_index, stratum in enumerate(VALIDATION_STRATA):
        rows = []
        for batch_index in range(batches_per_stratum):
            batch = generator.batch(910_000 + stratum_index * 10_000 + batch_index, batch_size, stratum)
            forward, inverse, velocity, reject_logit = model(
                batch["fixed"], batch["moving"], batch["fixed_mask"], batch["moving_mask"]
            )
            identity = pixel_identity_grid(
                batch_size, forward.shape[-2], forward.shape[-1], device=forward.device, dtype=forward.dtype
            )
            displacement = (forward - identity).square().sum(1, keepdim=True).sqrt()
            cycle = (compose_pixel_maps(forward, inverse) - identity).square().sum(1, keepdim=True).sqrt()
            target_error = (forward - batch["target_atlas_to_affine"]).square().sum(1, keepdim=True).sqrt()
            affine_error = (identity - batch["target_atlas_to_affine"]).square().sum(1, keepdim=True).sqrt()
            residual_affine = velocity - remove_global_affine(velocity)
            predicted_visible = batch["fixed_mask"] * (
                sample_at_pixel_map(batch["moving_mask"], forward, padding_mode="zeros") > 0.5
            )
            affine_visible = batch["fixed_mask"] * (batch["moving_mask"] > 0.5)
            folded = int((jacobian_determinant(forward) <= 0.0).sum() + (jacobian_determinant(inverse) <= 0.0).sum())
            row = {
                "folded_voxels": folded,
                "roundtrip_p95_px": _masked_percentile(cycle, batch["fixed_mask"], 0.95),
                "residual_affine_max_px": float(residual_affine.abs().max()),
                "landmark_tre_px": _masked_percentile(target_error[:, :, ::8, ::8], batch["fixed_mask"][:, :, ::8, ::8], 0.50),
                "affine_landmark_tre_px": _masked_percentile(affine_error[:, :, ::8, ::8], batch["fixed_mask"][:, :, ::8, ::8], 0.50),
                "label_dice": float(label_dice_score(batch["fixed_labels"], batch["moving_labels"], forward, predicted_visible)),
                "affine_label_dice": float(label_dice_score(batch["fixed_labels"], batch["moving_labels"], identity, affine_visible)),
                "identity_tre_p95_px": _masked_percentile(target_error, batch["fixed_mask"], 0.95),
                "wrong_reject_rate": float((torch.sigmoid(reject_logit) >= 0.5).float().mean()),
                "wrong_displacement_p95_px": _masked_percentile(displacement, batch["fixed_mask"], 0.95),
                "valid_reject_rate": float((torch.sigmoid(reject_logit) >= 0.5).float().mean()),
            }
            rows.append(row)
        strata[stratum] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    correct_deformation = ("smooth_deformation", "nuisance_damage")
    gates = {
        "folded_voxels": sum(strata[name]["folded_voxels"] for name in VALIDATION_STRATA[:-1]),
        "roundtrip_p95_px": max(strata[name]["roundtrip_p95_px"] for name in VALIDATION_STRATA[:-1]),
        "residual_affine_max_px": max(strata[name]["residual_affine_max_px"] for name in VALIDATION_STRATA),
        "landmark_tre_px": float(np.mean([strata[name]["landmark_tre_px"] for name in correct_deformation])),
        "affine_landmark_tre_px": float(np.mean([strata[name]["affine_landmark_tre_px"] for name in correct_deformation])),
        "label_dice": float(np.mean([strata[name]["label_dice"] for name in correct_deformation])),
        "affine_label_dice": float(np.mean([strata[name]["affine_label_dice"] for name in correct_deformation])),
        "identity_tre_p95_px": strata["identity_extreme"]["identity_tre_p95_px"],
        "wrong_reject_rate": strata["wrong_ap"]["wrong_reject_rate"],
        "wrong_displacement_p95_px": strata["wrong_ap"]["wrong_displacement_p95_px"],
        "valid_reject_rate": float(np.mean([strata[name]["valid_reject_rate"] for name in VALIDATION_STRATA[:-1]])),
    }
    return {"strata": strata, "gates": gates}


@torch.inference_mode()
def benchmark_runtime(model: nn.Module, device: torch.device, trials: int = 20) -> float:
    inputs = tuple(torch.zeros(1, 1, *PADDED_SIZE, device=device) for _ in range(4))
    model.eval()
    for _ in range(3):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(trials):
        start = time.perf_counter()
        model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def export_gate_failures(metrics: dict) -> list[str]:
    gates = metrics["gates"]
    failures = []
    checks = (
        (gates["folded_voxels"] == 0, "predicted map contains a fold"),
        (gates["roundtrip_p95_px"] <= 1.0, "round-trip p95 exceeds 1 px"),
        (gates["residual_affine_max_px"] <= 0.05, "residual global affine exceeds 0.05 px"),
        (gates["landmark_tre_px"] < gates["affine_landmark_tre_px"], "landmark TRE did not improve over affine"),
        (gates["label_dice"] > gates["affine_label_dice"], "label Dice did not improve over affine"),
        (gates["identity_tre_p95_px"] <= 1.0, "identity/extreme-modality case moved by more than 1 px"),
        (gates["wrong_reject_rate"] >= 0.95, "wrong-AP rejection is below 95%"),
        (gates["wrong_displacement_p95_px"] <= 1.0, "wrong-AP cases were warped instead of rejected"),
        (gates["valid_reject_rate"] <= 0.05, "valid-pair false rejection exceeds 5%"),
        (gates["runtime_ms"] <= gates["runtime_limit_ms"], "runtime benchmark exceeds its gate"),
    )
    for passed, message in checks:
        if not passed:
            failures.append(message)
    return failures


def export_validated_model(model: nn.Module, metrics: dict, destination: Path) -> tuple[bool, list[str]]:
    failures = export_gate_failures(metrics)
    if failures:
        return False, failures
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = tuple(torch.zeros(1, 1, *PADDED_SIZE, device=next(model.parameters()).device) for _ in range(4))
    torch.onnx.export(
        model.eval(),
        inputs,
        destination,
        input_names=("fixed", "moving", "fixed_mask", "moving_mask"),
        output_names=("atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit"),
        dynamic_axes={name: {0: "batch"} for name in (
            "fixed", "moving", "fixed_mask", "moving_mask",
            "atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit",
        )},
        opset_version=17,
        dynamo=False,
    )
    return True, []


def train() -> dict:
    workspace = workspace_path()
    if workspace.drive.upper() != "J:":
        raise ValueError("DIFFEO_WORKSPACE must be on J:")
    workspace.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("DIFFEO_BATCH_SIZE", "8"))
    total_steps = int(os.environ.get("DIFFEO_TRAIN_STEPS", "50000"))
    validation_interval = int(os.environ.get("DIFFEO_VALIDATION_INTERVAL", "1000"))
    patience = int(os.environ.get("DIFFEO_EARLY_STOPPING_PATIENCE", "10"))
    validation_batches = int(os.environ.get("DIFFEO_VALIDATION_BATCHES", "8"))
    torch.manual_seed(int(os.environ.get("DIFFEO_SEED", "73051")))

    generator = AllenObliquePairGenerator(ATLAS, device)
    model = RegistrationWithRejector().to(device)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    rng = np.random.default_rng(73051)
    strata = np.asarray(VALIDATION_STRATA)
    probabilities = np.asarray([0.10, 0.45, 0.30, 0.15])
    history = []
    best_score = float("inf")
    best_state = None
    stale = 0

    for step in range(1, total_steps + 1):
        stratum = str(rng.choice(strata, p=probabilities))
        batch = generator.batch(step * 1009 + 17, batch_size, stratum)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, terms = registration_objective(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        if step % validation_interval == 0 or step == total_steps:
            report = validation_report(ema.model, generator, validation_batches, batch_size)
            gates = report["gates"]
            score = (
                gates["landmark_tre_px"]
                + 5.0 * (1.0 - gates["label_dice"])
                + 5.0 * (1.0 - gates["wrong_reject_rate"])
            )
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "terms": {name: float(value.detach()) for name, value in terms.items()},
                "validation": report,
            })
            if score < best_score:
                best_score = score
                best_state = deepcopy(ema.model.state_dict())
                stale = 0
                torch.save({"model": best_state, "step": step, "history": history}, workspace / "best.pt")
            else:
                stale += 1
            (workspace / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if stale >= patience:
                break

    ema.model.load_state_dict(best_state)
    final = validation_report(ema.model, generator, validation_batches * 2, batch_size)
    final["gates"]["runtime_ms"] = benchmark_runtime(ema.model, device)
    final["gates"]["runtime_limit_ms"] = float(
        os.environ.get("DIFFEO_RUNTIME_LIMIT_MS", "250" if device.type == "cuda" else "2500")
    )
    final["device"] = str(device)
    final["training_steps"] = history[-1]["step"]
    exported, failures = export_validated_model(ema.model, final, workspace / "validated" / "diffeomorphic.onnx")
    final["exported"] = exported
    final["export_gate_failures"] = failures
    (workspace / "final_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
