"""Train and gate the post-pose, residual in-plane registration model.

Run as ``python -m training.train_diffeomorphic_registration``. Configuration
is read from ``DIFFEO_*`` environment variables; all generated artifacts stay
under ``J:/AtlasPoseDiffeomorphic`` by default. Nothing is promoted to the GUI.
"""

from __future__ import annotations

import hashlib
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

from source.nonlinear_registration import (
    COORDINATE_CONVENTION,
    MAXIMUM_ABS_LOG_JACOBIAN,
    MAXIMUM_ABS_LOG_JACOBIAN_P99,
    MAXIMUM_DISPLACEMENT_P95_PX,
    MAXIMUM_DISPLACEMENT_PX,
    MAXIMUM_INVERSE_P95_PX,
    MAXIMUM_INVERSE_PX,
    MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX,
    MAXIMUM_RESIDUAL_AFFINE_PX,
    MINIMUM_JACOBIAN,
    MODEL_CONTRACT_VERSION,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
    MODEL_PIXEL_SPACING_UM,
    MODEL_SHAPE,
    MODEL_SPATIAL_CONTRACT,
    RUNTIME_GATE_CONTRACT,
)
from training.diffeomorphic_registration_model import (
    DiffeomorphicRegistrationUNet,
    MAX_DEFORMATION_PX,
    compose_pixel_maps,
    hard_cell_mask,
    integrate_stationary_velocity,
    inverse_consistency_loss,
    jacobian_determinant,
    mind_loss,
    pixel_identity_grid,
    preprocess_registration_tensor,
    remove_tissue_affine,
    sample_at_pixel_map,
    smoothness_loss,
    synthetic_flow_loss,
    tissue_affine_component,
    topology_loss,
    soft_tissue_support,
)
from training.real_histology_registration import (
    NATIVE_WRONG_KINDS,
    REAL_HISTOLOGY_LOCKED_SEED,
    REAL_HISTOLOGY_SELECTION_SEED,
    RegisteredHistologySource,
    evaluate_real_histology,
    native_registration_batch,
    real_histology_gate_violation,
    select_native_wrong_entries,
    surface_affine_target_to_source,
    torch_model_sha256,
)
from training.synthetic_atlas import AP_MAX_UM, AP_MIN_UM, BREGMA_AP_INDEX, VOXEL_UM, SyntheticAtlas, make_manifest


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "Allen Brain Atlas 25um"
DEFAULT_WORKSPACE = Path("J:/AtlasPoseDiffeomorphic")
CLASS_COUNT = 9
SELECTION_STRATA = (
    "identity_extreme", "smooth_deformation", "nuisance_damage",
    "wrong_ap_near", "wrong_ap_far", "wrong_tilt",
)
LOCKED_STRATA = tuple(f"{name}_label_free" for name in SELECTION_STRATA)
SELECTION_SEED_BASE = 1_000_000_000
LOCKED_SEED_BASE = 2_000_000_000
# At the native 25 um grid these cap median/p95 dense landmark error at 25/50 um.
MAX_LANDMARK_TRE_MEDIAN_PX = 1.0
MAX_LANDMARK_TRE_P95_PX = 2.0
MIN_TRE_IMPROVEMENT_PX = 0.50
MIN_TRE_RELATIVE_IMPROVEMENT = 0.25
MIN_LABEL_DICE = 0.85
MIN_LABEL_DICE_IMPROVEMENT = 0.05


def workspace_path() -> Path:
    return Path(os.environ.get("DIFFEO_WORKSPACE", str(DEFAULT_WORKSPACE)))


def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _rand(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device)


def _pad_to(tensor: torch.Tensor, shape: tuple[int, int] = MODEL_SHAPE) -> torch.Tensor:
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
    label_appearance: bool = True,
) -> torch.Tensor:
    """Generate a label-conditioned arbitrary contrast while preserving geometry."""
    batch, _, height, width = template.shape
    device = template.device
    if label_appearance:
        palette = 0.05 + 0.9 * _rand((batch, CLASS_COUNT, 1, 1), generator, device)
        regional = palette.expand(-1, -1, height, width).gather(1, labels.long())
        edge_x = F.pad((labels[:, :, :, 1:] != labels[:, :, :, :-1]).float(), (0, 1, 0, 0))
        edge_y = F.pad((labels[:, :, 1:, :] != labels[:, :, :-1, :]).float(), (0, 0, 0, 1))
        edges = F.max_pool2d(torch.maximum(edge_x, edge_y), 3, stride=1, padding=1)
        mix_low, mix_high = ((0.02, 0.30) if extreme else (0.15, 0.82))
        mix = mix_low + (mix_high - mix_low) * _rand((batch, 1, 1, 1), generator, device)
        edge_weight = (_rand((batch, 1, 1, 1), generator, device) - 0.5) * (0.9 if extreme else 0.5)
        image = mix * template + (1.0 - mix) * regional + edge_weight * edges
    else:
        image = template

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
    max_velocity_px: float = MAX_DEFORMATION_PX,
    *,
    interior_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an affine-free smooth SVF and shrink it until its discrete map has positive J."""
    batch, _, height, width = mask.shape
    device = mask.device
    mask = (mask > 0.5).float()
    cell_mask = hard_cell_mask(mask)[:, 0]
    low = torch.randn((batch, 2, 6, 8), generator=generator, device=device)
    velocity = F.interpolate(low, (height, width), mode="bicubic", align_corners=True)
    velocity = F.avg_pool2d(velocity, 11, stride=1, padding=5)
    support = soft_tissue_support(mask.float(), mask.float())
    if interior_only:
        interior = 1.0 - F.max_pool2d(1.0 - mask, 11, stride=1, padding=5)
        support = support * interior
    velocity = remove_tissue_affine(velocity, support, mask.float())
    minimum_amplitude = 3.0 if interior_only else 1.5
    amplitude = minimum_amplitude + (max_velocity_px - minimum_amplitude) * _rand(
        (batch, 1, 1, 1), generator, device
    )
    peak = velocity.abs().flatten(1).amax(dim=1).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    velocity = velocity * amplitude / peak
    for _ in range(8):
        forward = integrate_stationary_velocity(velocity, steps=7)
        inverse = integrate_stationary_velocity(-velocity, steps=7)
        forward_jacobian = jacobian_determinant(forward)
        inverse_jacobian = jacobian_determinant(inverse)
        trusted_log_jacobian = [
            torch.cat((forward_jacobian[item][cell_mask[item]], inverse_jacobian[item][cell_mask[item]]))
            .clamp_min(1e-8)
            .log()
            .abs()
            for item in range(batch)
        ]
        log_tail = torch.stack([torch.quantile(values, 0.99) for values in trusted_log_jacobian])
        log_max = torch.stack([values.max() for values in trusted_log_jacobian])
        shrink = torch.where(
            (forward_jacobian.amin(dim=(1, 2)) < MINIMUM_JACOBIAN + 0.05)
            | (inverse_jacobian.amin(dim=(1, 2)) < MINIMUM_JACOBIAN + 0.05)
            | (log_tail > MAXIMUM_ABS_LOG_JACOBIAN_P99 - 0.20)
            | (log_max > MAXIMUM_ABS_LOG_JACOBIAN - 0.10),
            torch.full((batch,), 0.65, device=device),
            torch.ones(batch, device=device),
        )
        velocity = velocity * shrink[:, None, None, None]
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


def surface_affine_calibrate(
    template: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match centroid and axis scales so negative pairs cannot be rejected by outline size."""
    target_to_source = surface_affine_target_to_source(mask.float(), target_mask.float())
    calibrated_template = sample_at_pixel_map(template, target_to_source, padding_mode="zeros")
    calibrated_mask = sample_at_pixel_map(mask.float(), target_to_source, padding_mode="zeros")
    calibrated_labels = sample_at_pixel_map(one_hot_labels(labels), target_to_source, padding_mode="zeros")
    return calibrated_template, calibrated_labels.argmax(1, keepdim=True), calibrated_mask


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
    max_velocity_px: float = MAX_DEFORMATION_PX,
) -> dict[str, torch.Tensor]:
    """Construct an exact pair; only the known SVF contributes to its target flow."""
    device = template.device
    generator = _torch_generator(seed, device)
    batch, _, height, width = template.shape
    mask = (mask > 0.5).float()
    if wrong_mask is not None:
        wrong_mask = (wrong_mask > 0.5).float()
    identity = pixel_identity_grid(batch, height, width, device=device, dtype=template.dtype)
    label_appearance = not stratum.endswith("_label_free")
    base_stratum = stratum.removesuffix("_label_free")
    is_wrong = base_stratum.startswith("wrong_")
    extreme = base_stratum == "identity_extreme"
    source_template = wrong_template if is_wrong else template
    source_labels = wrong_labels if is_wrong else labels
    source_mask = wrong_mask if is_wrong else mask
    if is_wrong and source_template is None:
        raise ValueError("wrong_ap pairs require a second, known-wrong plane")

    fixed = synthesize_modality(
        template, labels, mask, generator, extreme=extreme, label_appearance=label_appearance
    )
    moving_base = synthesize_modality(
        source_template, source_labels, source_mask, generator,
        extreme=extreme, label_appearance=label_appearance,
    )
    if base_stratum in {"smooth_deformation", "nuisance_damage", "real_histology_interior"}:
        target_velocity, atlas_to_affine, affine_to_atlas = sample_anatomical_velocity(
            mask, generator, max_velocity_px,
            interior_only=base_stratum == "real_histology_interior",
        )
    else:
        target_velocity = torch.zeros_like(identity)
        atlas_to_affine = identity
        affine_to_atlas = identity

    moving = sample_at_pixel_map(moving_base, affine_to_atlas, padding_mode="zeros")
    moving_mask = (
        source_mask.float()
        if base_stratum == "real_histology_interior"
        else (sample_at_pixel_map(source_mask.float(), affine_to_atlas, padding_mode="zeros") > 0.5).float()
    )
    moving_labels = sample_at_pixel_map(one_hot_labels(source_labels), affine_to_atlas, padding_mode="zeros")
    if base_stratum == "nuisance_damage":
        moving, moving_mask = add_nuisance_damage(moving, moving_mask, generator)
    atlas_supervision_mask = mask.float() * (
        sample_at_pixel_map(moving_mask, atlas_to_affine, padding_mode="zeros") > 0.5
    )
    affine_supervision_mask = moving_mask * (
        sample_at_pixel_map(mask.float(), affine_to_atlas, padding_mode="zeros") > 0.5
    )
    fixed = preprocess_registration_tensor(fixed, mask.float())
    moving = preprocess_registration_tensor(moving, moving_mask)
    return {
        "fixed": fixed,
        "moving": moving,
        "fixed_mask": mask.float(),
        "moving_mask": moving_mask,
        "fixed_labels": one_hot_labels(labels),
        "moving_labels": moving_labels,
        "target_atlas_to_affine": atlas_to_affine,
        "target_affine_to_atlas": affine_to_atlas,
        "target_velocity": target_velocity,
        "atlas_supervision_mask": atlas_supervision_mask.float(),
        "affine_supervision_mask": affine_supervision_mask.float(),
        "retained_overlap": atlas_supervision_mask.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0),
        "wrong_pair": torch.full((batch,), is_wrong, device=device, dtype=torch.bool),
        "similarity_supervision": torch.full((batch,), not is_wrong, device=device, dtype=torch.bool),
        "dense_supervision": torch.full((batch,), not is_wrong, device=device, dtype=torch.bool),
        "label_supervision": torch.full(
            (batch,), not is_wrong and label_appearance, device=device, dtype=torch.bool
        ),
        "geometry_supervision": torch.full((batch,), not is_wrong, device=device, dtype=torch.bool),
        "support_supervision": torch.zeros(batch, device=device, dtype=torch.bool),
        "plane_basis_um": torch.tensor(
            ((0.0, 0.0), (0.0, MODEL_PIXEL_SPACING_UM), (MODEL_PIXEL_SPACING_UM, 0.0)),
            device=device,
            dtype=template.dtype,
        )[None].expand(batch, -1, -1),
    }


def real_histology_training_batch(
    source: RegisteredHistologySource,
    training_bank: dict,
    seed: int,
    count: int,
    device: torch.device,
    mode: str = "synthetic",
) -> dict[str, torch.Tensor]:
    records = training_bank["entries"]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(records), count, replace=len(records) < count)
    target_records = [records[int(index)] for index in selected]
    sections = [source.section(int(record["section_image_id"])) for record in target_records]
    if mode == "synthetic":
        images = torch.stack([torch.from_numpy(section["moving"]) for section in sections])[:, None].to(device)
        masks = torch.stack(
            [torch.from_numpy(section["moving_mask"]) for section in sections]
        )[:, None].to(device).float()
        labels = torch.zeros_like(images, dtype=torch.long)
        stratum = "nuisance_damage_label_free" if seed % 2 else "smooth_deformation_label_free"
        pair = make_synthetic_pair(images, labels, masks, seed=seed, stratum=stratum)
        pair["plane_basis_um"] = torch.stack(
            [torch.from_numpy(section["plane_basis_um"]) for section in sections]
        ).to(device)
        return pair
    if mode == "native_positive":
        return native_registration_batch(sections, device)
    if mode in NATIVE_WRONG_KINDS:
        wrong_records = select_native_wrong_entries(records, target_records, mode, seed)
        wrong_sections = [source.section(int(record["section_image_id"])) for record in wrong_records]
        return native_registration_batch(
            sections, device, wrong_sections=wrong_sections, wrong_kind=mode
        )
    raise ValueError(f"Unknown real-histology training mode: {mode}")


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
        base_stratum = stratum.removesuffix("_label_free")
        manifest = self._neutral_manifest(count, seed)
        template, mask, labels = self._render(manifest)
        wrong_template = wrong_mask = wrong_labels = None
        if base_stratum.startswith("wrong_"):
            wrong_manifest = {name: np.array(value, copy=True) for name, value in manifest.items()}
            rng = np.random.default_rng(seed ^ 0x51CE)
            if base_stratum in {"wrong_ap_near", "wrong_ap_far"}:
                bounds = (25.0, 500.0) if base_stratum == "wrong_ap_near" else (500.0, 1500.0)
                displacement = rng.uniform(*bounds, count).astype(np.float32)
                direction = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), count)
                candidate = manifest["ap_um"] + displacement * direction
                outside = (candidate < AP_MIN_UM) | (candidate > AP_MAX_UM)
                candidate[outside] = manifest["ap_um"][outside] - displacement[outside] * direction[outside]
                wrong_manifest["ap_um"] = candidate.clip(AP_MIN_UM, AP_MAX_UM)
                wrong_manifest["ap_index"] = BREGMA_AP_INDEX - wrong_manifest["ap_um"] / VOXEL_UM
            else:
                tilt_delta = rng.uniform(4.0, 15.0, (count, 2)).astype(np.float32)
                tilt_delta *= rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), (count, 2))
                wrong_tilt = np.column_stack((manifest["tilt_lr_deg"], manifest["tilt_dv_deg"])) + tilt_delta
                outside = (wrong_tilt < -35.0) | (wrong_tilt > 35.0)
                wrong_tilt[outside] -= 2.0 * tilt_delta[outside]
                wrong_manifest["tilt_lr_deg"] = wrong_tilt[:, 0]
                wrong_manifest["tilt_dv_deg"] = wrong_tilt[:, 1]
            wrong_template, wrong_mask, wrong_labels = self._render(wrong_manifest)
            wrong_template, wrong_labels, wrong_mask = surface_affine_calibrate(
                wrong_template, wrong_labels, wrong_mask, mask
            )
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
        present_loss = (1.0 - dice)[present]
        losses.append(present_loss.mean() if present_loss.numel() else warped.sum() * 0.0)
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
    terms = {
        "mind": zero, "flow": zero, "dice": zero, "inverse": zero,
        "smooth": zero, "topology": zero, "affine": zero, "support": zero,
    }
    similarity = batch["similarity_supervision"]
    if bool(similarity.any()):
        terms["mind"] = mind_loss(
            batch["fixed"][similarity], batch["moving"][similarity], predicted[similarity],
            batch["atlas_supervision_mask"][similarity],
        )
    dense = batch["dense_supervision"]
    if bool(dense.any()):
        terms["flow"] = 0.5 * (
            synthetic_flow_loss(
                predicted[dense], batch["target_atlas_to_affine"][dense],
                batch["atlas_supervision_mask"][dense],
            )
            + synthetic_flow_loss(
                predicted_inverse[dense], batch["target_affine_to_atlas"][dense],
                batch["affine_supervision_mask"][dense],
            )
        )
    labelled = batch["label_supervision"]
    if bool(labelled.any()):
        terms["dice"] = hierarchical_label_dice_loss(
            batch["fixed_labels"][labelled], batch["moving_labels"][labelled], predicted[labelled],
            batch["atlas_supervision_mask"][labelled],
        )
    geometry = batch["geometry_supervision"]
    if bool(geometry.any()):
        atlas_mask = batch["atlas_supervision_mask"][geometry]
        affine_mask = batch["affine_supervision_mask"][geometry]
        terms["inverse"] = inverse_consistency_loss(
            predicted[geometry], predicted_inverse[geometry], atlas_mask, affine_mask
        )
        terms["smooth"] = smoothness_loss(velocity[geometry])
        terms["topology"] = topology_loss(predicted[geometry], MINIMUM_JACOBIAN) + topology_loss(
            predicted_inverse[geometry], MINIMUM_JACOBIAN
        )
        forward_affine = tissue_affine_component(predicted[geometry], batch["fixed_mask"][geometry])
        inverse_affine = tissue_affine_component(
            predicted_inverse[geometry], batch["moving_mask"][geometry]
        )
        terms["affine"] = 0.5 * (forward_affine.square().mean() + inverse_affine.square().mean())
    support = batch["support_supervision"]
    if bool(support.any()):
        fixed_mask = batch["fixed_mask"][support]
        moving_mask = batch["moving_mask"][support]
        warped_mask = sample_at_pixel_map(
            moving_mask, predicted[support], padding_mode="zeros"
        ).clamp(0.0, 1.0)
        before = 2.0 * (fixed_mask * moving_mask).sum((-2, -1)) / (
            fixed_mask.sum((-2, -1)) + moving_mask.sum((-2, -1))
        ).clamp_min(1.0)
        after = 2.0 * (fixed_mask * warped_mask).sum((-2, -1)) / (
            fixed_mask.sum((-2, -1)) + warped_mask.sum((-2, -1))
        ).clamp_min(1.0)
        original_overlap = fixed_mask * moving_mask
        retained = (original_overlap * warped_mask).sum((-2, -1)) / original_overlap.sum(
            (-2, -1)
        ).clamp_min(1.0)
        terms["support"] = (F.relu(before - after) + F.relu(0.95 - retained)).mean()
    wrong_identity = zero
    if bool(wrong.any()):
        identity = pixel_identity_grid(
            int(wrong.sum()),
            predicted.shape[-2],
            predicted.shape[-1],
            device=predicted.device,
            dtype=predicted.dtype,
        )
        wrong_identity = 0.5 * (
            synthetic_flow_loss(predicted[wrong], identity, batch["fixed_mask"][wrong])
            + synthetic_flow_loss(predicted_inverse[wrong], identity, batch["moving_mask"][wrong])
        )
    total = (
        terms["mind"]
        + 2.0 * terms["flow"]
        + terms["dice"]
        + 0.2 * terms["inverse"]
        + 0.05 * terms["smooth"]
        + 5.0 * terms["topology"]
        + 2.0 * terms["affine"]
        + 2.0 * terms["support"]
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


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask.expand_as(values) > 0.5]
    return float(selected.abs().max()) if selected.numel() else float("nan")


@torch.inference_mode()
def validation_report(
    model: RegistrationWithRejector,
    generator: AllenObliquePairGenerator,
    batches_per_stratum: int,
    batch_size: int,
    *,
    strata_names: tuple[str, ...] = SELECTION_STRATA,
    seed_base: int = SELECTION_SEED_BASE,
) -> dict:
    model.eval()
    strata = {}
    aggregation = {
        "folded_voxels": np.sum,
        "minimum_jacobian": np.min,
        "maximum_abs_log_jacobian_p99": np.max,
        "maximum_abs_log_jacobian": np.max,
        "roundtrip_p95_px": np.max,
        "roundtrip_max_px": np.max,
        "inverse_finite_fraction": np.min,
        "map_finite_fraction": np.min,
        "residual_affine_max_px": np.max,
        "outside_tissue_displacement_max_px": np.max,
        "displacement_p95_px": np.max,
        "displacement_max_px": np.max,
        "landmark_tre_px": np.max,
        "landmark_tre_p95_px": np.max,
        "tre_improvement_px": np.min,
        "tre_relative_improvement": np.min,
        "tre_p95_improvement_px": np.min,
        "tre_p95_relative_improvement": np.min,
        "label_dice": np.min,
        "label_dice_improvement": np.min,
        "identity_tre_p95_px": np.max,
        "wrong_displacement_p95_px": np.max,
        "retained_overlap": np.min,
    }
    for stratum_index, stratum in enumerate(strata_names):
        rows = []
        for batch_index in range(batches_per_stratum):
            batch = generator.batch(seed_base + stratum_index * 10_000 + batch_index, batch_size, stratum)
            forward, inverse, _, reject_logit = model(
                batch["fixed"], batch["moving"], batch["fixed_mask"], batch["moving_mask"]
            )
            identity = pixel_identity_grid(
                batch_size, forward.shape[-2], forward.shape[-1], device=forward.device, dtype=forward.dtype
            )
            forward_displacement = (forward - identity).square().sum(1, keepdim=True).sqrt()
            inverse_displacement = (inverse - identity).square().sum(1, keepdim=True).sqrt()
            forward_cycle = (compose_pixel_maps(forward, inverse) - identity).square().sum(1, keepdim=True).sqrt()
            inverse_cycle = (compose_pixel_maps(inverse, forward) - identity).square().sum(1, keepdim=True).sqrt()
            forward_in_bounds = (
                (forward[:, :1] >= 0.0) & (forward[:, :1] <= forward.shape[-1] - 1.0)
                & (forward[:, 1:] >= 0.0) & (forward[:, 1:] <= forward.shape[-2] - 1.0)
            )
            inverse_in_bounds = (
                (inverse[:, :1] >= 0.0) & (inverse[:, :1] <= inverse.shape[-1] - 1.0)
                & (inverse[:, 1:] >= 0.0) & (inverse[:, 1:] <= inverse.shape[-2] - 1.0)
            )
            target_error = (forward - batch["target_atlas_to_affine"]).square().sum(1, keepdim=True).sqrt()
            affine_error = (identity - batch["target_atlas_to_affine"]).square().sum(1, keepdim=True).sqrt()
            forward_affine = tissue_affine_component(forward, batch["fixed_mask"])
            inverse_affine = tissue_affine_component(inverse, batch["moving_mask"])
            forward_jacobian = jacobian_determinant(forward)
            inverse_jacobian = jacobian_determinant(inverse)
            fixed_hard = batch["fixed_mask"] > 0.5
            moving_hard = batch["moving_mask"] > 0.5
            jacobian = torch.cat((forward_jacobian.flatten(), inverse_jacobian.flatten()))
            trusted_jacobian = torch.cat(
                (
                    forward_jacobian[hard_cell_mask(fixed_hard)[:, 0]],
                    inverse_jacobian[hard_cell_mask(moving_hard)[:, 0]],
                )
            )
            absolute_log_jacobian = trusted_jacobian.clamp_min(1e-8).log().abs()
            common_mask = batch["atlas_supervision_mask"] > 0.5
            union_mask = fixed_hard | moving_hard
            outside = (~union_mask) * torch.maximum(
                forward_displacement,
                inverse_displacement,
            )
            accepted_displacement = torch.cat(
                (forward_displacement[fixed_hard], inverse_displacement[moving_hard])
            )
            landmark_tre = _masked_percentile(
                target_error[:, :, ::8, ::8], common_mask[:, :, ::8, ::8], 0.50
            )
            affine_landmark_tre = _masked_percentile(
                affine_error[:, :, ::8, ::8], common_mask[:, :, ::8, ::8], 0.50
            )
            landmark_tre_p95 = _masked_percentile(target_error, common_mask, 0.95)
            affine_landmark_tre_p95 = _masked_percentile(affine_error, common_mask, 0.95)
            dice = float(label_dice_score(batch["fixed_labels"], batch["moving_labels"], forward, common_mask))
            affine_dice = float(
                label_dice_score(batch["fixed_labels"], batch["moving_labels"], identity, common_mask)
            )
            row = {
                "folded_voxels": int((jacobian <= 0.0).sum()),
                "minimum_jacobian": float(jacobian.min()),
                "maximum_abs_log_jacobian_p99": float(torch.quantile(absolute_log_jacobian, 0.99)),
                "maximum_abs_log_jacobian": float(absolute_log_jacobian.max()),
                "roundtrip_p95_px": max(
                    _masked_percentile(forward_cycle, fixed_hard * forward_in_bounds, 0.95),
                    _masked_percentile(inverse_cycle, moving_hard * inverse_in_bounds, 0.95),
                ),
                "roundtrip_max_px": max(
                    _masked_max(forward_cycle, fixed_hard * forward_in_bounds),
                    _masked_max(inverse_cycle, moving_hard * inverse_in_bounds),
                ),
                "inverse_finite_fraction": float(
                    (
                        (fixed_hard * forward_in_bounds).sum()
                        + (moving_hard * inverse_in_bounds).sum()
                    ) / (fixed_hard.sum() + moving_hard.sum()).clamp_min(1.0)
                ),
                "map_finite_fraction": float(
                    torch.cat((torch.isfinite(forward).flatten(), torch.isfinite(inverse).flatten())).float().mean()
                ),
                "residual_affine_max_px": max(
                    _masked_max(forward_affine, fixed_hard),
                    _masked_max(inverse_affine, moving_hard),
                ),
                "outside_tissue_displacement_max_px": float(outside.max()),
                "displacement_p95_px": float(torch.quantile(accepted_displacement.float(), 0.95)),
                "displacement_max_px": float(accepted_displacement.max()),
                "landmark_tre_px": landmark_tre,
                "affine_landmark_tre_px": affine_landmark_tre,
                "landmark_tre_p95_px": landmark_tre_p95,
                "affine_landmark_tre_p95_px": affine_landmark_tre_p95,
                "tre_improvement_px": affine_landmark_tre - landmark_tre,
                "tre_relative_improvement": 1.0 - landmark_tre / max(affine_landmark_tre, 1e-6),
                "tre_p95_improvement_px": affine_landmark_tre_p95 - landmark_tre_p95,
                "tre_p95_relative_improvement": 1.0 - landmark_tre_p95 / max(affine_landmark_tre_p95, 1e-6),
                "label_dice": dice,
                "affine_label_dice": affine_dice,
                "label_dice_improvement": dice - affine_dice,
                "identity_tre_p95_px": _masked_percentile(target_error, batch["fixed_mask"], 0.95),
                "wrong_reject_rate": float((torch.sigmoid(reject_logit) >= 0.5).float().mean()),
                "wrong_displacement_p95_px": max(
                    _masked_percentile(forward_displacement, fixed_hard, 0.95),
                    _masked_percentile(inverse_displacement, moving_hard, 0.95),
                ),
                "valid_reject_rate": float((torch.sigmoid(reject_logit) >= 0.5).float().mean()),
                "retained_overlap": float(batch["retained_overlap"].min()),
            }
            rows.append(row)
        strata[stratum] = {
            key: float(aggregation.get(key, np.mean)([row[key] for row in rows]))
            for key in rows[0]
        }

    valid_names = tuple(name for name in strata_names if not name.removesuffix("_label_free").startswith("wrong_"))
    correct_deformation = tuple(
        name for name in valid_names
        if name.removesuffix("_label_free") in {"smooth_deformation", "nuisance_damage"}
    )
    wrong_names = tuple(name for name in strata_names if name.removesuffix("_label_free").startswith("wrong_"))
    identity_name = next(name for name in valid_names if name.removesuffix("_label_free") == "identity_extreme")
    gates = {
        "folded_voxels": sum(strata[name]["folded_voxels"] for name in strata_names),
        "minimum_jacobian": min(strata[name]["minimum_jacobian"] for name in strata_names),
        "maximum_abs_log_jacobian_p99": max(strata[name]["maximum_abs_log_jacobian_p99"] for name in strata_names),
        "maximum_abs_log_jacobian": max(strata[name]["maximum_abs_log_jacobian"] for name in strata_names),
        "roundtrip_p95_px": max(strata[name]["roundtrip_p95_px"] for name in strata_names),
        "roundtrip_max_px": max(strata[name]["roundtrip_max_px"] for name in strata_names),
        "inverse_finite_fraction": min(strata[name]["inverse_finite_fraction"] for name in strata_names),
        "map_finite_fraction": min(strata[name]["map_finite_fraction"] for name in strata_names),
        "residual_affine_max_px": max(strata[name]["residual_affine_max_px"] for name in strata_names),
        "outside_tissue_displacement_max_px": max(
            strata[name]["outside_tissue_displacement_max_px"] for name in strata_names
        ),
        "displacement_p95_px": max(strata[name]["displacement_p95_px"] for name in valid_names),
        "displacement_max_px": max(strata[name]["displacement_max_px"] for name in valid_names),
        "landmark_tre_px": max(strata[name]["landmark_tre_px"] for name in correct_deformation),
        "landmark_tre_p95_px": max(strata[name]["landmark_tre_p95_px"] for name in correct_deformation),
        "affine_landmark_tre_px": max(
            strata[name]["affine_landmark_tre_px"] for name in correct_deformation
        ),
        "affine_landmark_tre_p95_px": max(
            strata[name]["affine_landmark_tre_p95_px"] for name in correct_deformation
        ),
        "tre_improvement_px": min(strata[name]["tre_improvement_px"] for name in correct_deformation),
        "tre_relative_improvement": min(
            strata[name]["tre_relative_improvement"] for name in correct_deformation
        ),
        "tre_p95_improvement_px": min(
            strata[name]["tre_p95_improvement_px"] for name in correct_deformation
        ),
        "tre_p95_relative_improvement": min(
            strata[name]["tre_p95_relative_improvement"] for name in correct_deformation
        ),
        "label_dice": min(strata[name]["label_dice"] for name in correct_deformation),
        "affine_label_dice": min(strata[name]["affine_label_dice"] for name in correct_deformation),
        "label_dice_improvement": min(
            strata[name]["label_dice_improvement"] for name in correct_deformation
        ),
        "identity_tre_p95_px": strata[identity_name]["identity_tre_p95_px"],
        "wrong_reject_rate": min(strata[name]["wrong_reject_rate"] for name in wrong_names),
        "wrong_reject_rate_by_stratum": {name: strata[name]["wrong_reject_rate"] for name in wrong_names},
        "wrong_displacement_p95_px": max(strata[name]["wrong_displacement_p95_px"] for name in wrong_names),
        "valid_reject_rate": float(np.mean([strata[name]["valid_reject_rate"] for name in valid_names])),
        "retained_overlap": min(strata[name]["retained_overlap"] for name in valid_names),
    }
    return {"strata": strata, "gates": gates}


@torch.inference_mode()
def benchmark_runtime(model: nn.Module, device: torch.device, trials: int = 20) -> float:
    inputs = tuple(torch.zeros(1, 1, *MODEL_SHAPE, device=device) for _ in range(4))
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
        (gates["minimum_jacobian"] >= MINIMUM_JACOBIAN, "minimum Jacobian is below 0.20"),
        (
            gates["maximum_abs_log_jacobian_p99"] <= MAXIMUM_ABS_LOG_JACOBIAN_P99,
            "absolute log-Jacobian p99 exceeds its limit",
        ),
        (
            gates["maximum_abs_log_jacobian"] <= MAXIMUM_ABS_LOG_JACOBIAN,
            "absolute log-Jacobian maximum exceeds its limit",
        ),
        (gates["inverse_finite_fraction"] == 1.0, "a forward/inverse cycle leaves the finite canvas"),
        (gates["map_finite_fraction"] == 1.0, "a predicted map contains non-finite coordinates"),
        (gates["roundtrip_p95_px"] <= MAXIMUM_INVERSE_P95_PX, "round-trip p95 exceeds 1 px"),
        (gates["roundtrip_max_px"] <= MAXIMUM_INVERSE_PX, "round-trip maximum exceeds 2 px"),
        (
            gates["residual_affine_max_px"] <= MAXIMUM_RESIDUAL_AFFINE_PX,
            "residual global affine exceeds 0.05 px",
        ),
        (
            gates["outside_tissue_displacement_max_px"] <= MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX,
            "outside-tissue map is not identity",
        ),
        (
            gates["displacement_p95_px"] <= MAXIMUM_DISPLACEMENT_P95_PX,
            "accepted-pair displacement p95 exceeds 8 px",
        ),
        (
            gates["displacement_max_px"] <= MAXIMUM_DISPLACEMENT_PX,
            "accepted-pair displacement maximum exceeds 12 px",
        ),
        (gates["landmark_tre_px"] <= MAX_LANDMARK_TRE_MEDIAN_PX, "landmark median TRE exceeds 1 px"),
        (gates["landmark_tre_p95_px"] <= MAX_LANDMARK_TRE_P95_PX, "landmark TRE p95 exceeds 2 px"),
        (
            gates["tre_improvement_px"] >= MIN_TRE_IMPROVEMENT_PX,
            "landmark median TRE improvement is below 0.5 px",
        ),
        (
            gates["tre_relative_improvement"] >= MIN_TRE_RELATIVE_IMPROVEMENT,
            "landmark median TRE relative improvement is below 25%",
        ),
        (
            gates["tre_p95_improvement_px"] >= MIN_TRE_IMPROVEMENT_PX,
            "landmark TRE p95 improvement is below 0.5 px",
        ),
        (
            gates["tre_p95_relative_improvement"] >= MIN_TRE_RELATIVE_IMPROVEMENT,
            "landmark TRE p95 relative improvement is below 25%",
        ),
        (gates["label_dice"] >= MIN_LABEL_DICE, "hierarchical anatomical Dice is below 0.85"),
        (
            gates["label_dice_improvement"] >= MIN_LABEL_DICE_IMPROVEMENT,
            "hierarchical anatomical Dice improvement is below 0.05",
        ),
        (gates["identity_tre_p95_px"] <= 1.0, "identity/extreme-modality case moved by more than 1 px"),
        (gates["wrong_reject_rate"] >= 0.95, "wrong-AP rejection is below 95%"),
        (gates["wrong_displacement_p95_px"] <= 1.0, "wrong-AP cases were warped instead of rejected"),
        (gates["valid_reject_rate"] <= 0.05, "valid-pair false rejection exceeds 5%"),
        (gates["retained_overlap"] >= 0.40, "retained trusted overlap is below 40%"),
        (gates["runtime_ms"] <= gates["runtime_limit_ms"], "runtime benchmark exceeds its gate"),
    )
    for passed, message in checks:
        if not passed:
            failures.append(message)
    return failures


def synthetic_gate_violation(gates: dict) -> float:
    """Return dimensionless validation-gate violation, excluding runtime."""
    upper = lambda value, limit: max(float(value) / float(limit) - 1.0, 0.0)
    lower = lambda value, limit: max(1.0 - float(value) / float(limit), 0.0)
    violations = (
        float(gates["folded_voxels"]),
        lower(gates["minimum_jacobian"], MINIMUM_JACOBIAN),
        upper(gates["maximum_abs_log_jacobian_p99"], MAXIMUM_ABS_LOG_JACOBIAN_P99),
        upper(gates["maximum_abs_log_jacobian"], MAXIMUM_ABS_LOG_JACOBIAN),
        1.0 - min(float(gates["inverse_finite_fraction"]), 1.0),
        1.0 - min(float(gates["map_finite_fraction"]), 1.0),
        upper(gates["roundtrip_p95_px"], MAXIMUM_INVERSE_P95_PX),
        upper(gates["roundtrip_max_px"], MAXIMUM_INVERSE_PX),
        upper(gates["residual_affine_max_px"], MAXIMUM_RESIDUAL_AFFINE_PX),
        upper(
            gates["outside_tissue_displacement_max_px"],
            MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX,
        ) if MAXIMUM_OUTSIDE_TISSUE_DISPLACEMENT_PX else float(
            gates["outside_tissue_displacement_max_px"] > 0.0
        ),
        upper(gates["displacement_p95_px"], MAXIMUM_DISPLACEMENT_P95_PX),
        upper(gates["displacement_max_px"], MAXIMUM_DISPLACEMENT_PX),
        upper(gates["landmark_tre_px"], MAX_LANDMARK_TRE_MEDIAN_PX),
        upper(gates["landmark_tre_p95_px"], MAX_LANDMARK_TRE_P95_PX),
        lower(gates["tre_improvement_px"], MIN_TRE_IMPROVEMENT_PX),
        lower(gates["tre_relative_improvement"], MIN_TRE_RELATIVE_IMPROVEMENT),
        lower(gates["tre_p95_improvement_px"], MIN_TRE_IMPROVEMENT_PX),
        lower(gates["tre_p95_relative_improvement"], MIN_TRE_RELATIVE_IMPROVEMENT),
        lower(gates["label_dice"], MIN_LABEL_DICE),
        lower(gates["label_dice_improvement"], MIN_LABEL_DICE_IMPROVEMENT),
        upper(gates["identity_tre_p95_px"], 1.0),
        lower(gates["wrong_reject_rate"], 0.95),
        upper(gates["wrong_displacement_p95_px"], 1.0),
        upper(gates["valid_reject_rate"], 0.05),
        lower(gates["retained_overlap"], 0.40),
    )
    return float(sum(violations))


def checkpoint_selection_key(
    validation: dict,
    real_validation: dict | None,
    scientific_score: float,
) -> tuple[int, float, float]:
    """Prefer releasable checkpoints, then least gate violation, then scientific score."""
    violation = synthetic_gate_violation(validation["gates"])
    if real_validation is not None:
        violation += real_histology_gate_violation(real_validation["gates"])
    return (0 if violation == 0.0 else 1, violation, float(scientific_score))


def export_candidate_model(model: nn.Module, metrics: dict, destination: Path) -> tuple[bool, list[str]]:
    failures = export_gate_failures(metrics)
    if failures:
        return False, failures
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = tuple(torch.zeros(1, 1, *MODEL_SHAPE, device=next(model.parameters()).device) for _ in range(4))
    torch.onnx.export(
        model.eval(),
        inputs,
        destination,
        input_names=MODEL_INPUT_NAMES,
        output_names=MODEL_OUTPUT_NAMES,
        dynamic_axes={name: {0: "batch"} for name in (*MODEL_INPUT_NAMES, *MODEL_OUTPUT_NAMES)},
        opset_version=17,
        dynamo=False,
    )
    return True, []


def write_prelocked_evidence(model_path: Path, report: dict) -> tuple[Path, str]:
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = {
        "format_version": 1,
        "model_sha256": model_sha256,
        "synthetic_gate": report["synthetic_gate"],
        "onnx_gate": report["onnx_gate"],
        "onnx_parity": report.get("onnx_parity"),
        "locked_pytorch": report.get("locked_pytorch"),
        "locked_onnx": report.get("locked_onnx"),
        "locked_real_histology_commitment": report.get("locked_real_histology_commitment"),
    }
    evidence_path = model_path.with_suffix(".prelocked.json")
    evidence_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    evidence_path.write_bytes(evidence_bytes)
    return evidence_path, hashlib.sha256(evidence_bytes).hexdigest()


def write_model_manifest(model_path: Path, report: dict) -> tuple[Path, str]:
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    evidence_path, evidence_sha256 = write_prelocked_evidence(model_path, report)
    real_gate = report["real_histology_ground_truth_gate"]
    payload = {
        "format_version": MODEL_CONTRACT_VERSION,
        "model_sha256": model_sha256,
        "model_shape": list(MODEL_SHAPE),
        "pixel_spacing_um": MODEL_PIXEL_SPACING_UM,
        "spatial_contract": MODEL_SPATIAL_CONTRACT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "input_names": list(MODEL_INPUT_NAMES),
        "output_names": list(MODEL_OUTPUT_NAMES),
        "runtime_gates": RUNTIME_GATE_CONTRACT,
        "prelocked_evidence_file": evidence_path.name,
        "prelocked_evidence_sha256": evidence_sha256,
        "locked_real_histology_commitment": report.get("locked_real_histology_commitment"),
        "synthetic_gate_passed": bool(report["synthetic_gate"]["passed"]),
        "onnx_gate_passed": bool(report["onnx_gate"]["passed"]),
        "real_histology_gate_passed": bool(real_gate["passed"]),
        "real_histology_gate_report_sha256": real_gate.get("report_sha256"),
        "real_histology_evaluation_manifest_sha256": real_gate.get("evaluation_manifest_sha256"),
        "real_histology_source": real_gate.get("source"),
        "real_histology_benchmark_role": real_gate.get("benchmark_role"),
        "promotion_ready": bool(report["promotion_ready"]),
    }
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, hashlib.sha256(manifest_bytes).hexdigest()


class OnnxRegistrationModel:
    def __init__(self, path: Path):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "DmlExecutionProvider") if name in available]
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.provider = self.session.get_providers()[0]

    def eval(self) -> "OnnxRegistrationModel":
        return self

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feeds = {
            name: value.detach().float().cpu().numpy()
            for name, value in zip(MODEL_INPUT_NAMES, inputs)
        }
        outputs = self.session.run(None, feeds)
        return tuple(torch.from_numpy(value).to(inputs[0].device) for value in outputs)


@torch.inference_mode()
def onnx_parity_report(
    pytorch_model: nn.Module,
    onnx_model: OnnxRegistrationModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, float | str | bool]:
    inputs = tuple(batch[name] for name in MODEL_INPUT_NAMES)
    expected = pytorch_model.eval()(*inputs)
    observed = onnx_model(*inputs)
    errors = [float((left - right).abs().max()) for left, right in zip(expected, observed)]
    return {
        "provider": onnx_model.provider,
        "atlas_to_affine_max_abs": errors[0],
        "affine_to_atlas_max_abs": errors[1],
        "velocity_max_abs": errors[2],
        "rejection_logit_max_abs": errors[3],
        "passed": max(errors[:3]) <= 1e-3 and errors[3] <= 1e-4,
    }


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
    seed = int(os.environ.get("DIFFEO_SEED", "73051"))
    torch.manual_seed(seed)

    generator = AllenObliquePairGenerator(ATLAS, device)
    registered_root = os.environ.get("DIFFEO_REGISTERED_ROOT")
    real_source = RegisteredHistologySource(registered_root, ATLAS) if registered_root else None
    real_train_fraction = float(os.environ.get("DIFFEO_REAL_HISTOLOGY_TRAIN_FRACTION", "0.35"))
    real_training_bank = real_source.training_bank_manifest() if real_source else None
    real_selection_manifest = (
        real_source.evaluation_manifest("validation", REAL_HISTOLOGY_SELECTION_SEED)
        if real_source else None
    )
    model = RegistrationWithRejector().to(device)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    strata = np.asarray(SELECTION_STRATA)
    probabilities = np.asarray([0.10, 0.25, 0.25, 0.15, 0.15, 0.10])
    history = []
    best_key = None
    best_state = None
    best_selection = None
    best_real_selection = None
    best_step = None
    stale = 0

    for step in range(1, total_steps + 1):
        stratum = str(rng.choice(strata, p=probabilities))
        if rng.random() < 0.35:
            stratum += "_label_free"
        training_seed = (seed + step * 1009 + 17) % 900_000_000
        if real_source and rng.random() < real_train_fraction:
            real_mode = str(rng.choice(
                np.asarray(("synthetic", "native_positive", *NATIVE_WRONG_KINDS)),
                p=np.asarray((0.50, 0.25, 0.25 / 3.0, 0.25 / 3.0, 0.25 / 3.0)),
            ))
            batch = real_histology_training_batch(
                real_source, real_training_bank, training_seed, batch_size, device, real_mode
            )
        else:
            batch = generator.batch(training_seed, batch_size, stratum)
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
            report = validation_report(
                ema.model,
                generator,
                validation_batches,
                batch_size,
                strata_names=SELECTION_STRATA,
                seed_base=SELECTION_SEED_BASE,
            )
            real_selection = (
                evaluate_real_histology(
                    ema.model,
                    real_source,
                    real_selection_manifest,
                    make_synthetic_pair,
                    device,
                    torch_model_sha256(ema.model),
                    batch_size=min(batch_size, 4),
                )
                if real_source else None
            )
            gates = report["gates"]
            score = (
                gates["landmark_tre_px"]
                + 5.0 * (1.0 - gates["label_dice"])
                + 5.0 * (1.0 - gates["wrong_reject_rate"])
            )
            if real_selection:
                real_gates = real_selection["gates"]
                score += (
                    real_gates["dense_epe_median_px"]
                    + 0.5 * real_gates["dense_epe_p95_px"]
                    + 10.0 * real_gates["native_mind_delta"]
                    + 5.0 * (1.0 - real_gates["native_accept_rate"])
                    + 5.0 * (1.0 - real_gates["native_wrong_reject_rate"])
                )
            if not np.isfinite(score):
                raise RuntimeError("Selection metrics became non-finite")
            selection_key = checkpoint_selection_key(report, real_selection, score)
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "terms": {name: float(value.detach()) for name, value in terms.items()},
                "validation": report,
                "real_histology_validation": real_selection,
                "selection_key": list(selection_key),
            })
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_state = deepcopy(ema.model.state_dict())
                best_selection = report
                best_real_selection = real_selection
                best_step = step
                stale = 0
                torch.save(
                    {
                        "model": best_state,
                        "step": step,
                        "selection": best_selection,
                        "real_histology_selection": best_real_selection,
                        "real_histology_training_bank": real_training_bank,
                        "history": history,
                    },
                    workspace / "best.pt",
                )
            else:
                stale += 1
            (workspace / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if stale >= patience:
                break

    ema.model.load_state_dict(best_state)
    locked = validation_report(
        ema.model,
        generator,
        validation_batches * 2,
        batch_size,
        strata_names=LOCKED_STRATA,
        seed_base=LOCKED_SEED_BASE,
    )
    locked["gates"]["runtime_ms"] = benchmark_runtime(ema.model, device)
    locked["gates"]["runtime_limit_ms"] = float(
        os.environ.get("DIFFEO_RUNTIME_LIMIT_MS", "250" if device.type == "cuda" else "2500")
    )
    synthetic_failures = export_gate_failures(locked)
    candidate_path = workspace / f"candidate-seed-{seed}-step-{best_step}" / "diffeomorphic.onnx"
    candidate_exported, export_failures = export_candidate_model(ema.model, locked, candidate_path)
    parity = None
    onnx_locked = None
    onnx_failures = (
        [] if candidate_exported
        else ["ONNX candidate was not exported because the PyTorch locked gates failed"]
    )
    if candidate_exported:
        onnx_model = OnnxRegistrationModel(candidate_path)
        parity_batch = generator.batch(
            LOCKED_SEED_BASE + 99_000_000,
            min(batch_size, 2),
            "nuisance_damage_label_free",
        )
        parity = onnx_parity_report(ema.model, onnx_model, parity_batch)
        if parity["passed"]:
            onnx_locked = validation_report(
                onnx_model,
                generator,
                validation_batches * 2,
                batch_size,
                strata_names=LOCKED_STRATA,
                seed_base=LOCKED_SEED_BASE,
            )
            onnx_locked["gates"]["runtime_ms"] = benchmark_runtime(onnx_model, device)
            onnx_locked["gates"]["runtime_limit_ms"] = locked["gates"]["runtime_limit_ms"]
            onnx_failures = export_gate_failures(onnx_locked)
        else:
            onnx_failures = ["ONNX outputs do not match PyTorch within the parity tolerances"]

    real_histology_gate = {
        "status": "blocked",
        "passed": False,
        "reason": (
            "The locked animal-disjoint test gate is deliberately separate from training and checkpoint selection. "
            "Run training.evaluate_locked_nonlinear_histology once for the frozen ONNX candidate."
        ),
    }
    locked_real_histology_commitment = None
    if real_source is not None:
        locked_manifest = real_source.evaluation_manifest("test", REAL_HISTOLOGY_LOCKED_SEED)
        locked_real_histology_commitment = {
            "source": real_source.contract,
            "evaluation_manifest_sha256": locked_manifest["manifest_sha256"],
        }
    final = {
        "device": str(device),
        "training_steps": history[-1]["step"],
        "selected_step": best_step,
        "selected_checkpoint_key": list(best_key),
        "split_seeds": {
            "training_range": [0, 899_999_999],
            "selection_base": SELECTION_SEED_BASE,
            "locked_base": LOCKED_SEED_BASE,
            "real_histology_selection": REAL_HISTOLOGY_SELECTION_SEED,
            "sealed_used_for_tuning": False,
        },
        "selection": best_selection,
        "real_histology_selection": best_real_selection,
        "real_histology_training_bank": real_training_bank,
        "locked_pytorch": locked,
        "synthetic_gate": {"passed": not synthetic_failures, "failures": synthetic_failures},
        "candidate_exported": candidate_exported,
        "candidate_path": str(candidate_path) if candidate_exported else None,
        "candidate_export_failures": export_failures,
        "onnx_parity": parity,
        "locked_onnx": onnx_locked,
        "onnx_gate": {"passed": bool(parity and parity["passed"] and not onnx_failures), "failures": onnx_failures},
        "real_histology_ground_truth_gate": real_histology_gate,
        "locked_real_histology_commitment": locked_real_histology_commitment,
        "real_histology_gate_artifact": None,
        "promotion_ready": False,
        "promoted": False,
    }
    if candidate_exported:
        manifest_path, manifest_sha256 = write_model_manifest(candidate_path, final)
        final["candidate_manifest_path"] = str(manifest_path)
        final["candidate_manifest_sha256"] = manifest_sha256
    (workspace / "final_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
