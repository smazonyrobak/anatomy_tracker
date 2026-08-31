"""Matched native versus fixed-random cost-volume adjacency diagnostic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.independent_atlas_pair_energy import (
    AtlasPairEnergyModel,
    _local_correlation,
    _masked_statistics,
    atlas_pair_loss,
)


CORRELATION_CHANNELS = (2 * AtlasPairEnergyModel.radius + 1) ** 2
COST_VOLUME_CHANNELS = CORRELATION_CHANNELS + 1
TOPOLOGY_CHANNELS = 32
TOPOLOGY_OUTPUT_CHANNELS = 16
STATISTICS_DIMENSION = 2 * CORRELATION_CHANNELS + 2 * TOPOLOGY_OUTPUT_CHANNELS
EXPECTED_PARAMETER_COUNT = 284_058


def _precommitted_permutation(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(size, generator=generator)
    return permutation, permutation.argsort()


PERMUTATION8, INVERSE_PERMUTATION8 = _precommitted_permutation(20 * 29, 180432208)
PERMUTATION16, INVERSE_PERMUTATION16 = _precommitted_permutation(10 * 15, 180432216)


def permutation_is_bijective(
    permutation: torch.Tensor, inverse: torch.Tensor
) -> bool:
    positions = torch.arange(len(permutation), device=permutation.device)
    return bool(
        permutation.ndim == inverse.ndim == 1
        and len(permutation) == len(inverse)
        and torch.equal(permutation.sort().values, positions)
        and torch.equal(inverse.sort().values, positions)
        and torch.equal(permutation[inverse], positions)
        and torch.equal(inverse[permutation], positions)
    )


def permute_lattice(value: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return value.flatten(2).index_select(2, permutation).reshape_as(value)


def restore_lattice(value: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    return value.flatten(2).index_select(2, inverse).reshape_as(value)


def lattice_permutation_audit(
    value: torch.Tensor, permutation: torch.Tensor, inverse: torch.Tensor
) -> dict[str, bool]:
    scrambled = permute_lattice(value, permutation)
    recovered = restore_lattice(scrambled, inverse)
    multiset_exact = True
    for original_item, scrambled_item in zip(value, scrambled):
        original_vectors = original_item.flatten(1).T.contiguous()
        scrambled_vectors = scrambled_item.flatten(1).T.contiguous()
        original_unique, original_counts = torch.unique(
            original_vectors, dim=0, return_counts=True
        )
        scrambled_unique, scrambled_counts = torch.unique(
            scrambled_vectors, dim=0, return_counts=True
        )
        multiset_exact &= torch.equal(original_unique, scrambled_unique) and torch.equal(
            original_counts, scrambled_counts
        )
    return {
        "bijection": permutation_is_bijective(permutation, inverse),
        "vector_multiset_exact": bool(multiset_exact),
        "recovery_exact": torch.equal(recovered, value),
    }


class _PixelLayerNorm(nn.LayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.movedim(1, -1)).movedim(-1, 1)


def _pointwise(value: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
    weight = layer.weight[:, :, 0, 0]
    return F.linear(value.movedim(1, -1), weight).movedim(-1, 1)


class _TopologyResidualBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            TOPOLOGY_CHANNELS,
            TOPOLOGY_CHANNELS,
            3,
            padding=dilation,
            dilation=dilation,
            groups=TOPOLOGY_CHANNELS,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            TOPOLOGY_CHANNELS, TOPOLOGY_CHANNELS, 1, bias=False
        )
        self.normalization = _PixelLayerNorm(TOPOLOGY_CHANNELS)
        self.activation = nn.GELU()
        with torch.no_grad():
            center = self.depthwise.weight[:, :, 1, 1].clone()
            self.depthwise.weight.zero_()
            self.depthwise.weight[:, :, 1, 1] = center

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        center = self.depthwise.weight[:, 0, 1, 1].view(1, -1, 1, 1) * value
        off_center = self.depthwise.weight.clone()
        off_center[:, :, 1, 1] = 0
        update = center + F.conv2d(
            value,
            off_center,
            padding=self.depthwise.padding,
            dilation=self.depthwise.dilation,
            groups=TOPOLOGY_CHANNELS,
        )
        update = _pointwise(update, self.pointwise)
        update = self.normalization(update)
        return value + self.activation(update)


class _TopologyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(
            COST_VOLUME_CHANNELS, TOPOLOGY_CHANNELS, 1, bias=False
        )
        self.input_normalization = _PixelLayerNorm(TOPOLOGY_CHANNELS)
        self.input_activation = nn.GELU()
        self.blocks = nn.ModuleList(
            _TopologyResidualBlock(dilation) for dilation in (1, 2, 4)
        )
        self.output_projection = nn.Conv2d(
            TOPOLOGY_CHANNELS, TOPOLOGY_OUTPUT_CHANNELS, 1, bias=False
        )
        self.output_activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_activation(
            self.input_normalization(_pointwise(value, self.input_projection))
        )
        for block in self.blocks:
            value = block(value)
        return self.output_activation(_pointwise(value, self.output_projection))


def off_center_depthwise_coefficients(
    model: nn.Module, *, gradients: bool = False
) -> torch.Tensor:
    values = []
    for module in model.modules():
        if isinstance(module, _TopologyResidualBlock):
            value = module.depthwise.weight.grad if gradients else module.depthwise.weight
            if value is None:
                raise RuntimeError("depthwise gradients are unavailable")
            mask = torch.ones(3, 3, dtype=torch.bool, device=value.device)
            mask[1, 1] = False
            values.append(value[..., mask].reshape(-1))
    return torch.cat(values)


def _energy_head() -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(STATISTICS_DIMENSION),
        nn.Linear(STATISTICS_DIMENSION, 48),
        nn.GELU(),
        nn.Linear(48, 1),
    )


class _AtlasPairEnergyMatchedTopology(AtlasPairEnergyModel):
    scramble_topology = False

    def __init__(self):
        super().__init__()
        self.topology8 = _TopologyEncoder()
        self.topology16 = _TopologyEncoder()
        self.energy8 = _energy_head()
        self.energy16 = _energy_head()
        self.register_buffer("permutation8", PERMUTATION8.clone())
        self.register_buffer("inverse_permutation8", INVERSE_PERMUTATION8.clone())
        self.register_buffer("permutation16", PERMUTATION16.clone())
        self.register_buffer("inverse_permutation16", INVERSE_PERMUTATION16.clone())

    def _level_statistics(
        self,
        correlation: torch.Tensor,
        atlas_mask: torch.Tensor,
        topology: _TopologyEncoder,
        permutation: torch.Tensor,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        mask = F.interpolate(
            atlas_mask.float(), correlation.shape[-2:], mode="nearest"
        ).to(correlation)
        lattice = torch.cat((correlation, mask), dim=1)
        if lattice.shape[-2] * lattice.shape[-1] != len(permutation):
            raise ValueError("cost-volume lattice does not match the committed grid")
        if self.scramble_topology:
            lattice = permute_lattice(lattice, permutation)
        features = topology(lattice)
        if self.scramble_topology:
            features = restore_lattice(features, inverse)
        features = features.flatten(2).contiguous()
        pooled = torch.cat(
            (features.mean(2), features.amax(2)), dim=1
        )
        return torch.cat((_masked_statistics(correlation, atlas_mask), pooled), dim=1)

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
        statistics8 = self._level_statistics(
            _local_correlation(source8, atlas8, self.radius),
            atlas_mask,
            self.topology8,
            self.permutation8,
            self.inverse_permutation8,
        )
        statistics16 = self._level_statistics(
            _local_correlation(source16, atlas16, self.radius),
            atlas_mask,
            self.topology16,
            self.permutation16,
            self.inverse_permutation16,
        )
        energy8 = self.energy8(statistics8).squeeze(1)
        energy16 = self.energy16(statistics16).squeeze(1)
        return {
            "energy": 0.5 * (energy8 + energy16),
            "energy8": energy8,
            "energy16": energy16,
        }


class AtlasPairEnergyScrambledTopologyControl(_AtlasPairEnergyMatchedTopology):
    """Null arm: apply the spatial CNN on a fixed scrambled lattice."""

    scramble_topology = True


class AtlasPairEnergyNativeTopology(_AtlasPairEnergyMatchedTopology):
    """Treatment arm: apply the same spatial CNN on the native lattice."""


def make_atlas_pair_topology_pair() -> tuple[
    AtlasPairEnergyScrambledTopologyControl, AtlasPairEnergyNativeTopology
]:
    control = AtlasPairEnergyScrambledTopologyControl()
    treatment = AtlasPairEnergyNativeTopology()
    treatment.load_state_dict(control.state_dict(), strict=True)
    return control, treatment
