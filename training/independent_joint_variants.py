"""Cold-start controls for the independent joint pose-registration model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.independent_joint_model import (
    IndependentCachedRefinerExport,
    IndependentCandidateScorerExport,
    IndependentInitializerExport,
    IndependentJointModel,
    ProbabilisticPoseHead,
    _ResidualBlock,
    _group_count,
    project_pose_to_domain,
)


def _candidate_sources(
    model: IndependentJointModel,
    fixed_features: tuple[torch.Tensor, ...],
    source_features: tuple[torch.Tensor, ...],
    pose_context: torch.Tensor,
    current_pose: torch.Tensor,
    source_index: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if source_index is None:
        source_features = model._expand_source_features(
            source_features, fixed_features[0].shape[0]
        )
        pose_context = pose_context.expand(current_pose.shape[0], -1)
    else:
        source_features = tuple(
            feature.index_select(0, source_index) for feature in source_features
        )
        pose_context = pose_context.index_select(0, source_index)
    return source_features, pose_context


def _projection(input_channels: int, output_channels: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.GELU(),
        *[_ResidualBlock(output_channels) for _ in range(blocks)],
    )


class SpatialMomentProbabilisticPoseHead(ProbabilisticPoseHead):
    """Add DSNT-style spatial moments to the existing multilevel GAP pose context.

    Four learned heatmaps are normalized over space and summarized by their first
    and second moments.  This is the parameter-light spatial-softmax construction
    of Finn et al. (arXiv:1509.06113), extended with differentiable variance and
    covariance statistics; the normalized coordinate expectation follows DSNT
    (Nibali et al., arXiv:1801.07372).
    """

    attention_map_count = 4
    moments_per_map = 5

    def __init__(
        self,
        feature_channels: tuple[int, ...],
        context_features: int = 128,
        ap_bins: int = 41,
        tilt_bins: int = 29,
    ):
        moment_features = self.attention_map_count * self.moments_per_map
        super().__init__(
            (*feature_channels, moment_features),
            context_features=context_features,
            ap_bins=ap_bins,
            tilt_bins=tilt_bins,
        )
        self.spatial_attention_logits = nn.Conv2d(
            feature_channels[-1], self.attention_map_count, 1
        )
        nn.init.normal_(self.spatial_attention_logits.weight, std=1e-3)
        nn.init.zeros_(self.spatial_attention_logits.bias)

    @staticmethod
    def moments_from_logits(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized maps and flattened ``(mux,muy,varx,vary,covxy)``."""
        batch, maps, height, width = logits.shape
        weights = torch.softmax(logits.flatten(2), dim=2).reshape(
            batch, maps, height, width
        )
        x = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        x = x[None, None, None, :]
        y = y[None, None, :, None]
        mean_x = (weights * x).sum(dim=(-2, -1))
        mean_y = (weights * y).sum(dim=(-2, -1))
        centered_x = x - mean_x[:, :, None, None]
        centered_y = y - mean_y[:, :, None, None]
        variance_x = (weights * centered_x.square()).sum(dim=(-2, -1))
        variance_y = (weights * centered_y.square()).sum(dim=(-2, -1))
        covariance_xy = (weights * centered_x * centered_y).sum(dim=(-2, -1))
        moments = torch.stack(
            (mean_x, mean_y, variance_x, variance_y, covariance_xy), dim=2
        ).flatten(1)
        return weights, moments

    def spatial_attention_moments(
        self, final_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.moments_from_logits(self.spatial_attention_logits(final_feature))

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        _, moments = self.spatial_attention_moments(features[-1])
        return super().forward((*features, moments[:, :, None, None]))


class IndependentJointSpatialMomentModel(IndependentJointModel):
    """Leader model with a spatially aware pose initializer and unchanged refiner."""

    architecture_family = "recurrent_spatial_moment_initializer"
    uses_recurrent_state = True
    comparison_refinement_steps = 3

    def __init__(
        self,
        pyramid_channels: tuple[int, ...] = (24, 40, 64, 96),
        pose_context_features: int = 192,
        pair_features: int = 96,
        hidden_channels: int = 96,
        integration_steps: int = 6,
        maximum_pose_delta: tuple[float, float, float] = (750.0, 7.5, 7.5),
        maximum_translation_pixels: float = 32.0,
        minimum_scale: float = 0.40,
        maximum_scale: float = 2.00,
        maximum_velocity_fraction: float = 0.12,
    ):
        super().__init__(
            pyramid_channels=pyramid_channels,
            pose_context_features=pose_context_features,
            pair_features=pair_features,
            hidden_channels=hidden_channels,
            integration_steps=integration_steps,
            maximum_pose_delta=maximum_pose_delta,
            maximum_translation_pixels=maximum_translation_pixels,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
            maximum_velocity_fraction=maximum_velocity_fraction,
        )
        self.pose_head = SpatialMomentProbabilisticPoseHead(
            pyramid_channels, context_features=pose_context_features
        )


class SupervisedSimilarityCanonicalizer(nn.Module):
    """Tiny diagnostic STN for known synthetic source-view rotation and scale.

    The predicted affine maps canonical output coordinates into the observed
    source image.  This is the forward source-view homography used by
    ``independent_joint_data._apply_source_view``, not its inverse.
    """

    initialization_seed = 12731

    def __init__(self):
        super().__init__()
        self.localizer = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(8), 8),
            nn.GELU(),
            nn.Conv2d(8, 12, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(12), 12),
            nn.GELU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(16), 16),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.parameters_head = nn.Linear(16, 2)
        nn.init.zeros_(self.parameters_head.weight)
        nn.init.zeros_(self.parameters_head.bias)
        self.register_buffer("maximum_rotation_deg", torch.tensor(45.0))
        self.register_buffer(
            "maximum_log_scale", torch.tensor(float(math.log(1.5)))
        )

    def predict_parameters(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.parameters_head(self.localizer(planes))
        rotation_deg = torch.tanh(raw[:, 0]) * self.maximum_rotation_deg
        log_scale = torch.tanh(raw[:, 1]) * self.maximum_log_scale
        return {
            "source_view_rotation_deg": rotation_deg,
            "source_view_log_scale": log_scale,
            "source_view_scale": torch.exp(log_scale),
        }

    @staticmethod
    def sampling_theta(
        rotation_deg: torch.Tensor,
        log_scale: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Normalized output-to-input affine for a centered pixel similarity."""
        angle = rotation_deg * (math.pi / 180.0)
        scale = torch.exp(log_scale)
        cosine = scale * torch.cos(angle)
        sine = scale * torch.sin(angle)
        zeros = torch.zeros_like(cosine)
        x_per_y = float(height - 1) / float(width - 1)
        y_per_x = float(width - 1) / float(height - 1)
        return torch.stack(
            (
                torch.stack((cosine, -sine * x_per_y, zeros), dim=1),
                torch.stack((sine * y_per_x, cosine, zeros), dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def warp_with_parameters(
        planes: torch.Tensor,
        rotation_deg: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        theta = SupervisedSimilarityCanonicalizer.sampling_theta(
            rotation_deg, log_scale, planes.shape[-2], planes.shape[-1]
        )
        grid = F.affine_grid(theta, planes.shape, align_corners=True)
        return F.grid_sample(
            planes,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(
        self, planes: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parameters = self.predict_parameters(planes)
        # This causal diagnostic allows only the exact nuisance labels to train
        # the localizer.  Pose gradients still train the encoder downstream of
        # the warped pixels, but cannot repurpose theta to absorb anatomy.
        warped = self.warp_with_parameters(
            planes,
            parameters["source_view_rotation_deg"].detach(),
            parameters["source_view_log_scale"].detach(),
        )
        return warped, parameters


class _SupervisedSimilarityCanonicalized:
    """Diagnostic-only one-warp source canonicalization mixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SupervisedSimilarityCanonicalizer.initialization_seed)
            canonicalizer = SupervisedSimilarityCanonicalizer()
        self.source_view_canonicalizer = canonicalizer

    def _canonicalized_source_features(
        self,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        planes = self.pyramid._input(
            source_image, source_outline_mask, source_mask_available
        )
        planes, parameters = self.source_view_canonicalizer(planes)
        feature = self.pyramid.slice_stem(planes)
        features = []
        for level in self.pyramid.levels:
            feature = level(feature)
            features.append(feature)
        return tuple(features), parameters

    def encode_source(
        self,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features, _ = self._canonicalized_source_features(
            source_image, source_outline_mask, source_mask_available
        )
        return features

    def initialize(
        self,
        slice_image: torch.Tensor,
        slice_outline_mask: torch.Tensor,
        slice_mask_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features, parameters = self._canonicalized_source_features(
            slice_image, slice_outline_mask, slice_mask_available
        )
        output = self.pose_head(features)
        output.update(parameters)
        return output


class IndependentJointSimilarityCanonicalizedModel(
    _SupervisedSimilarityCanonicalized, IndependentJointModel
):
    """Base GAP initializer preceded by the supervised diagnostic STN."""

    architecture_family = "recurrent_supervised_similarity_initializer"


class IndependentJointSpatialMomentSimilarityCanonicalizedModel(
    _SupervisedSimilarityCanonicalized, IndependentJointSpatialMomentModel
):
    """Spatial-moment initializer preceded by the same supervised diagnostic STN."""

    architecture_family = "recurrent_spatial_moment_supervised_similarity_initializer"


class IndependentJointOracleSimilarityModel(IndependentJointModel):
    """Marker class for runner-side exact source-view canonicalization."""

    architecture_family = "recurrent_oracle_similarity_initializer"


class IndependentJointSpatialMomentOracleSimilarityModel(
    IndependentJointSpatialMomentModel
):
    """Spatial-moment marker for runner-side exact source-view canonicalization."""

    architecture_family = "recurrent_spatial_moment_oracle_similarity_initializer"


class FactorizedCNNControl(IndependentJointModel):
    """Stateless CNN control for three factorized render-and-correct updates."""

    architecture_family = "factorized_cnn_control"
    uses_recurrent_state = False
    comparison_refinement_steps = 3

    def __init__(
        self,
        pyramid_channels: tuple[int, ...] = (24, 40, 64, 96),
        pose_context_features: int = 192,
        pair_features: int = 96,
        hidden_channels: int = 96,
        fusion_channels: int = 112,
        integration_steps: int = 6,
        maximum_pose_delta: tuple[float, float, float] = (750.0, 7.5, 7.5),
        maximum_translation_pixels: float = 32.0,
        minimum_scale: float = 0.40,
        maximum_scale: float = 2.00,
        maximum_velocity_fraction: float = 0.12,
    ):
        super().__init__(
            pyramid_channels=pyramid_channels,
            pose_context_features=pose_context_features,
            pair_features=pair_features,
            hidden_channels=hidden_channels,
            integration_steps=integration_steps,
            maximum_pose_delta=maximum_pose_delta,
            maximum_translation_pixels=maximum_translation_pixels,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
            maximum_velocity_fraction=maximum_velocity_fraction,
        )
        coarse_channels = pyramid_channels[-1]
        middle_channels = pyramid_channels[-2]
        condition_channels = max(pair_features // 2, 16)
        self.pair_projection = nn.Identity()
        self.recurrent = nn.Identity()
        self.factor_coarse = _projection(coarse_channels * 3, fusion_channels, 2)
        self.factor_middle = _projection(middle_channels * 3, fusion_channels, 2)
        self.factor_fusion = nn.Sequential(
            nn.Conv2d(
                fusion_channels * 2 + condition_channels,
                hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            _ResidualBlock(hidden_channels),
            _ResidualBlock(hidden_channels),
        )
        self.stateless_hidden_channels = int(hidden_channels)

    def initial_hidden_state(self, atlas_image: torch.Tensor) -> torch.Tensor:
        divisor = 2 ** len(self.pyramid.channels)
        height = (atlas_image.shape[-2] + divisor - 1) // divisor
        width = (atlas_image.shape[-1] + divisor - 1) // divisor
        return atlas_image.new_zeros(
            atlas_image.shape[0], self.stateless_hidden_channels, height, width
        )

    def _coarse_update(
        self,
        fixed_features: tuple[torch.Tensor, ...],
        source_features: tuple[torch.Tensor, ...],
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None,
        source_index: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]
    ]:
        source_features, pose_context = _candidate_sources(
            self,
            fixed_features,
            source_features,
            pose_context,
            current_pose,
            source_index,
        )
        coarse = self.factor_coarse(
            torch.cat(
                (
                    fixed_features[-1],
                    source_features[-1],
                    torch.abs(fixed_features[-1] - source_features[-1]),
                ),
                dim=1,
            )
        )
        middle = self.factor_middle(
            torch.cat(
                (
                    fixed_features[-2],
                    source_features[-2],
                    torch.abs(fixed_features[-2] - source_features[-2]),
                ),
                dim=1,
            )
        )
        middle = F.interpolate(
            middle, size=coarse.shape[-2:], mode="bilinear", align_corners=True
        )
        normalized_pose = (current_pose - self.pose_center) / self.pose_scale
        condition = self.condition(torch.cat((pose_context, normalized_pose), dim=1))
        condition = condition[:, :, None, None].expand(
            -1, -1, coarse.shape[-2], coarse.shape[-1]
        )
        state = self.factor_fusion(torch.cat((coarse, middle, condition), dim=1))
        if hidden_state is not None:
            state = state + (hidden_state - hidden_state)
        pooled = state.mean(dim=(-2, -1))
        pose_delta = torch.tanh(self.pose_delta_head(pooled)) * self.maximum_pose_delta
        outputs = {
            "pose": project_pose_to_domain(current_pose + pose_delta),
            "pose_delta": pose_delta,
            "compatibility_logit": self.compatibility_head(pooled).squeeze(1),
            "hidden_state": state,
        }
        return outputs, pooled, source_features


def _local_windows(features: torch.Tensor, radius: int) -> torch.Tensor:
    height, width = features.shape[-2:]
    padded = F.pad(features, (radius, radius, radius, radius), mode="replicate")
    return torch.stack(
        [
            padded[
                :,
                :,
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
            for offset_y in range(2 * radius + 1)
            for offset_x in range(2 * radius + 1)
        ],
        dim=2,
    )


class _BidirectionalLocalAttention(nn.Module):
    def __init__(self, input_channels: int, attention_channels: int, radius: int):
        super().__init__()
        self.query = nn.Conv2d(input_channels, attention_channels, 1, bias=False)
        self.key = nn.Conv2d(input_channels, attention_channels, 1, bias=False)
        self.value = nn.Conv2d(input_channels, attention_channels, 1, bias=False)
        self.radius = int(radius)
        self.attention_channels = int(attention_channels)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        keys = _local_windows(key, self.radius)
        values = _local_windows(value, self.radius)
        logits = (query[:, :, None] * keys).sum(dim=1) / self.attention_channels**0.5
        weights = torch.softmax(logits, dim=1)
        return (weights[:, None] * values).sum(dim=2)

    def forward(self, fixed: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        fixed_query = self.query(fixed)
        fixed_key = self.key(fixed)
        fixed_value = self.value(fixed)
        source_query = self.query(source)
        source_key = self.key(source)
        source_value = self.value(source)
        return torch.cat(
            (
                self._attend(fixed_query, source_key, source_value),
                self._attend(source_query, fixed_key, fixed_value),
            ),
            dim=1,
        )


class RecurrentAttentionVariant(IndependentJointModel):
    """Leader recurrence with bidirectional local attention for coarse matching."""

    architecture_family = "recurrent_windowed_attention"
    uses_recurrent_state = True
    comparison_refinement_steps = 3

    def __init__(
        self,
        pyramid_channels: tuple[int, ...] = (24, 40, 64, 96),
        pose_context_features: int = 192,
        pair_features: int = 96,
        hidden_channels: int = 96,
        attention_channels: int = 32,
        integration_steps: int = 6,
        maximum_pose_delta: tuple[float, float, float] = (750.0, 7.5, 7.5),
        maximum_translation_pixels: float = 32.0,
        minimum_scale: float = 0.40,
        maximum_scale: float = 2.00,
        maximum_velocity_fraction: float = 0.12,
    ):
        super().__init__(
            pyramid_channels=pyramid_channels,
            pose_context_features=pose_context_features,
            pair_features=pair_features,
            hidden_channels=hidden_channels,
            integration_steps=integration_steps,
            maximum_pose_delta=maximum_pose_delta,
            maximum_translation_pixels=maximum_translation_pixels,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
            maximum_velocity_fraction=maximum_velocity_fraction,
        )
        coarse_channels = pyramid_channels[-1]
        self.coarse_attention = _BidirectionalLocalAttention(
            coarse_channels, attention_channels, radius=2
        )
        self.middle_attention = _BidirectionalLocalAttention(
            pyramid_channels[-2], attention_channels, radius=1
        )
        self.pair_projection = nn.Sequential(
            nn.Conv2d(
                coarse_channels * 2 + attention_channels * 4,
                pair_features,
                1,
            ),
            nn.GroupNorm(_group_count(pair_features), pair_features),
            nn.GELU(),
            _ResidualBlock(pair_features),
        )

    def _coarse_update(
        self,
        fixed_features: tuple[torch.Tensor, ...],
        source_features: tuple[torch.Tensor, ...],
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None,
        source_index: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]
    ]:
        source_features, pose_context = _candidate_sources(
            self,
            fixed_features,
            source_features,
            pose_context,
            current_pose,
            source_index,
        )
        coarse_fixed = fixed_features[-1]
        coarse_source = source_features[-1]
        coarse_attention = self.coarse_attention(coarse_fixed, coarse_source)
        middle_attention = self.middle_attention(
            fixed_features[-2], source_features[-2]
        )
        middle_attention = F.interpolate(
            middle_attention,
            size=coarse_fixed.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        pair = self.pair_projection(
            torch.cat(
                (
                    torch.abs(coarse_fixed - coarse_source),
                    coarse_fixed * coarse_source,
                    coarse_attention,
                    middle_attention,
                ),
                dim=1,
            )
        )
        normalized_pose = (current_pose - self.pose_center) / self.pose_scale
        condition = self.condition(torch.cat((pose_context, normalized_pose), dim=1))
        condition = condition[:, :, None, None].expand(
            -1, -1, pair.shape[-2], pair.shape[-1]
        )
        if hidden_state is None:
            hidden_state = pair.new_zeros(
                pair.shape[0], self.recurrent.gates.out_channels // 2, *pair.shape[-2:]
            )
        next_hidden = self.recurrent(torch.cat((pair, condition), dim=1), hidden_state)
        pooled = next_hidden.mean(dim=(-2, -1))
        pose_delta = torch.tanh(self.pose_delta_head(pooled)) * self.maximum_pose_delta
        outputs = {
            "pose": project_pose_to_domain(current_pose + pose_delta),
            "pose_delta": pose_delta,
            "compatibility_logit": self.compatibility_head(pooled).squeeze(1),
            "hidden_state": next_hidden,
        }
        return outputs, pooled, source_features


class VariantInitializerExport(IndependentInitializerExport):
    pass


class VariantCandidateScorerExport(IndependentCandidateScorerExport):
    pass


class VariantCachedRefinerExport(IndependentCachedRefinerExport):
    pass


LEADER_PARAMETER_REFERENCE = 1_369_070
DEFAULT_VARIANT_PARAMETERS = {
    IndependentJointSpatialMomentModel.architecture_family: 1_373_338,
    IndependentJointSimilarityCanonicalizedModel.architecture_family: 1_373_904,
    IndependentJointSpatialMomentSimilarityCanonicalizedModel.architecture_family: 1_378_172,
    FactorizedCNNControl.architecture_family: 1_387_342,
    RecurrentAttentionVariant.architecture_family: 1_393_454,
}


def install_resource_hooks(
    model: nn.Module,
) -> tuple[dict[str, int], list[torch.utils.hooks.RemovableHandle]]:
    """Install MAC counters; call ``resource_snapshot`` after a representative pass."""
    statistics = {"macs": 0}
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def count(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(module, nn.Conv2d):
            statistics["macs"] += int(
                output.numel()
                * module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                / module.groups
            )
        elif isinstance(module, nn.Linear):
            statistics["macs"] += int(output.numel() * module.in_features)
        elif isinstance(module, _BidirectionalLocalAttention):
            batch, _, height, width = inputs[0].shape
            window = (2 * module.radius + 1) ** 2
            statistics["macs"] += int(
                4 * batch * height * width * window * module.attention_channels
            )

    handles = [
        module.register_forward_hook(count)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear, _BidirectionalLocalAttention))
    ]
    return statistics, handles


def resource_snapshot(model: nn.Module, statistics: dict[str, int]) -> dict[str, int]:
    device = next(model.parameters()).device
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs": int(statistics["macs"]),
        "peak_vram_bytes": peak_vram,
    }
