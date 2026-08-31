"""Parameter-matched global versus fixed 2x2 Haar correlation aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.independent_atlas_pair_energy import (
    AtlasPairEnergyModel,
    _local_correlation,
    _masked_statistics,
)


CORRELATION_CHANNELS = (2 * AtlasPairEnergyModel.radius + 1) ** 2
STATISTICS_DIMENSION = 5 * CORRELATION_CHANNELS
EXPECTED_PARAMETER_COUNT = 271_780


def _symmetric_halves(
    length: int, *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate = torch.arange(length, device=device, dtype=dtype)
    midpoint = (length - 1) / 2
    first = (coordinate < midpoint).to(dtype)
    if length % 2:
        first[length // 2] = 0.5
    return first, 1.0 - first


def _haar_2x2_ac_weights(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return top-bottom, left-right, and diagonal quadrant contrasts / 2."""
    top, bottom = _symmetric_halves(height, device=device, dtype=dtype)
    left, right = _symmetric_halves(width, device=device, dtype=dtype)
    top_left = top[:, None] * left[None]
    top_right = top[:, None] * right[None]
    bottom_left = bottom[:, None] * left[None]
    bottom_right = bottom[:, None] * right[None]
    return 0.5 * torch.stack(
        (
            top_left + top_right - bottom_left - bottom_right,
            top_left + bottom_left - top_right - bottom_right,
            top_left + bottom_right - top_right - bottom_left,
        )
    )


def _masked_global_haar_statistics(
    correlation: torch.Tensor,
    atlas_mask: torch.Tensor,
    *,
    use_haar_coefficients: bool,
) -> torch.Tensor:
    global_statistics = _masked_statistics(correlation, atlas_mask)
    channels = correlation.shape[1]
    if not use_haar_coefficients:
        return torch.cat(
            (global_statistics, correlation.new_zeros(len(correlation), 3 * channels)),
            dim=1,
        )

    mask = F.interpolate(
        atlas_mask.float(), correlation.shape[-2:], mode="nearest"
    ).squeeze(1).to(correlation)
    support = mask.sum((-2, -1)).clamp_min(1.0)
    mean = global_statistics[:, :channels]
    centered = (correlation - mean[:, :, None, None]) * mask[:, None]
    weights = _haar_2x2_ac_weights(
        correlation.shape[-2],
        correlation.shape[-1],
        device=correlation.device,
        dtype=correlation.dtype,
    )
    coefficients = (
        centered[:, :, None] * weights[None, None]
    ).sum((-2, -1)) / support[:, None, None]
    return torch.cat(
        (global_statistics, coefficients.permute(0, 2, 1).reshape(len(correlation), -1)),
        dim=1,
    )


def _energy_head() -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(STATISTICS_DIMENSION),
        nn.Linear(STATISTICS_DIMENSION, 25),
        nn.GELU(),
        nn.Linear(25, 1),
    )


class _AtlasPairEnergyMatchedSpatialAggregation(AtlasPairEnergyModel):
    use_haar_coefficients = False

    def __init__(self):
        super().__init__()
        self.energy8 = _energy_head()
        self.energy16 = _energy_head()

    def score_encoded(
        self,
        source_features: tuple[torch.Tensor, torch.Tensor],
        atlas_image: torch.Tensor,
        atlas_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        atlas8, atlas16 = self.encode_atlas(atlas_image, atlas_mask)
        source8, source16 = source_features
        if len(source8) != len(atlas8):
            if len(atlas8) % len(source8):
                raise ValueError("candidate batch must be a multiple of source batch")
            repeats = len(atlas8) // len(source8)
            source8 = source8.repeat_interleave(repeats, 0)
            source16 = source16.repeat_interleave(repeats, 0)
        statistics8 = _masked_global_haar_statistics(
            _local_correlation(source8, atlas8, self.radius),
            atlas_mask,
            use_haar_coefficients=self.use_haar_coefficients,
        )
        statistics16 = _masked_global_haar_statistics(
            _local_correlation(source16, atlas16, self.radius),
            atlas_mask,
            use_haar_coefficients=self.use_haar_coefficients,
        )
        energy8 = self.energy8(statistics8).squeeze(1)
        energy16 = self.energy16(statistics16).squeeze(1)
        return {
            "energy": 0.5 * (energy8 + energy16),
            "energy8": energy8,
            "energy16": energy16,
        }


class AtlasPairEnergyGlobalAggregationControl(
    _AtlasPairEnergyMatchedSpatialAggregation
):
    """Matched 405-D head with all fixed Haar AC inputs held at zero."""


class AtlasPairEnergyHaar2x2SpatialAggregation(
    _AtlasPairEnergyMatchedSpatialAggregation
):
    """Matched head using three fixed 2x2 Haar AC coefficients per displacement."""

    use_haar_coefficients = True
