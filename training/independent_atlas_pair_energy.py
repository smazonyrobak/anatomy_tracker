"""Compact random-init atlas-pair energy model for the oracle premise."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_INPUT_SHAPE = (160, 232)
POSE_SCALE = (2500.0, 35.0, 35.0)


class _Block(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride: int):
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                input_channels, output_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )


class _Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.level2 = _Block(3, 24, 2)
        self.level4 = _Block(24, 32, 2)
        self.level8 = _Block(32, 48, 2)
        self.level16 = _Block(48, 64, 2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.level2(value)
        value = self.level4(value)
        level8 = self.level8(value)
        return level8, self.level16(level8)


def _local_correlation(
    source: torch.Tensor, atlas: torch.Tensor, radius: int
) -> torch.Tensor:
    """Return one normalized dot-product channel per local displacement."""
    if source.shape != atlas.shape or source.ndim != 4:
        raise ValueError("source and atlas features must have matching NCHW shapes")
    source = F.normalize(source.float(), dim=1, eps=1e-6)
    atlas = F.normalize(atlas.float(), dim=1, eps=1e-6)
    diameter = 2 * radius + 1
    patches = F.unfold(atlas, diameter, padding=radius).reshape(
        len(atlas), atlas.shape[1], diameter * diameter, *atlas.shape[-2:]
    )
    return (source[:, :, None] * patches).sum(1)


def _masked_statistics(
    correlation: torch.Tensor, atlas_mask: torch.Tensor
) -> torch.Tensor:
    mask = F.interpolate(
        atlas_mask.float(), correlation.shape[-2:], mode="nearest"
    ).squeeze(1)
    support = mask.sum((-2, -1)).clamp_min(1.0)
    mean = (correlation * mask[:, None]).sum((-2, -1)) / support[:, None]
    maximum = correlation.masked_fill(~mask[:, None].bool(), -torch.inf).amax((-2, -1))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


class AtlasPairEnergyModel(nn.Module):
    """Pose-blind atlas/source matcher with cached source pyramid features."""

    radius = 4

    def __init__(self):
        super().__init__()
        self.source_stem = _Stem()
        self.atlas_stem = _Stem()
        self.source_projection8 = nn.Conv2d(48, 24, 1, bias=False)
        self.atlas_projection8 = nn.Conv2d(48, 24, 1, bias=False)
        self.source_projection16 = nn.Conv2d(64, 32, 1, bias=False)
        self.atlas_projection16 = nn.Conv2d(64, 32, 1, bias=False)
        statistics = 2 * (2 * self.radius + 1) ** 2
        self.energy8 = nn.Sequential(nn.LayerNorm(statistics), nn.Linear(statistics, 64), nn.GELU(), nn.Linear(64, 1))
        self.energy16 = nn.Sequential(nn.LayerNorm(statistics), nn.Linear(statistics, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _source_channels(
        image: torch.Tensor, mask: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        availability = available.to(image).expand(-1, -1, *image.shape[-2:])
        return torch.cat((image, mask.to(image), availability), dim=1)

    @staticmethod
    def _atlas_channels(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((image, mask.to(image), torch.ones_like(image)), dim=1)

    def encode_source(
        self, image: torch.Tensor, mask: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level8, level16 = self.source_stem(
            self._source_channels(image, mask, available)
        )
        return self.source_projection8(level8), self.source_projection16(level16)

    def encode_atlas(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level8, level16 = self.atlas_stem(self._atlas_channels(image, mask))
        return self.atlas_projection8(level8), self.atlas_projection16(level16)

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
        statistics8 = _masked_statistics(
            _local_correlation(source8, atlas8, self.radius), atlas_mask
        )
        statistics16 = _masked_statistics(
            _local_correlation(source16, atlas16, self.radius), atlas_mask
        )
        energy8 = self.energy8(statistics8).squeeze(1)
        energy16 = self.energy16(statistics16).squeeze(1)
        return {
            "energy": 0.5 * (energy8 + energy16),
            "energy8": energy8,
            "energy16": energy16,
        }

    def forward(
        self,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        mask_available: torch.Tensor,
        candidate_image: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        candidate_chunk_size: int = 8,
    ) -> dict[str, torch.Tensor]:
        if candidate_image.ndim != 5 or candidate_image.shape[:2] != candidate_mask.shape[:2]:
            raise ValueError("candidates must have matching BxCx1xHxW image and mask tensors")
        batch, candidates = candidate_image.shape[:2]
        source_features = self.encode_source(source_image, source_mask, mask_available)
        pieces = {name: [] for name in ("energy", "energy8", "energy16")}
        for start in range(0, candidates, candidate_chunk_size):
            stop = min(start + candidate_chunk_size, candidates)
            image = candidate_image[:, start:stop].flatten(0, 1)
            mask = candidate_mask[:, start:stop].flatten(0, 1)
            scored = self.score_encoded(source_features, image, mask)
            for name, value in scored.items():
                pieces[name].append(value.reshape(batch, stop - start))
        return {name: torch.cat(value, 1) for name, value in pieces.items()}


def atlas_pair_loss(
    output: dict[str, torch.Tensor],
    candidate_pose: torch.Tensor,
    truth_pose: torch.Tensor,
    target_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = -output["energy"]
    ranking = F.cross_entropy(logits, target_index)
    auxiliary = 0.5 * (
        F.cross_entropy(-output["energy8"], target_index)
        + F.cross_entropy(-output["energy16"], target_index)
    )
    probability = logits.softmax(1)
    posterior_mean = (probability[:, :, None] * candidate_pose).sum(1)
    scale = truth_pose.new_tensor(POSE_SCALE)
    point = F.smooth_l1_loss(posterior_mean / scale, truth_pose / scale)
    total = ranking + 0.25 * auxiliary + 0.25 * point
    return {
        "total": total,
        "ranking": ranking,
        "auxiliary_ranking": auxiliary,
        "point": point,
        "probability": probability,
        "posterior_mean": posterior_mean,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())
