"""Cold-start recurrent CCF pose and nonlinear-registration network.

This module is intentionally self-contained: it has no learned-weight dependency
and does not import either legacy pose or registration model.  A deterministic
host renderer supplies an atlas plane at ``current_pose`` between recurrent
calls.  Every image uses the explicit three-channel contract ``intensity,
outline, mask_available``.  The outline is an optional input prior, never a
damage/visibility target.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


AP_MIN_UM = -4500.0
AP_MAX_UM = 500.0
TILT_MIN_DEG = -35.0
TILT_MAX_DEG = 35.0
POSE_CENTER = (-2000.0, 0.0, 0.0)
POSE_SCALE = (2500.0, 35.0, 35.0)


def project_pose_to_domain(pose: torch.Tensor) -> torch.Tensor:
    """Project ``[AP_um, LR_deg, DV_deg]`` onto the supported CCF domain."""
    minimum = pose.new_tensor((AP_MIN_UM, TILT_MIN_DEG, TILT_MIN_DEG))
    maximum = pose.new_tensor((AP_MAX_UM, TILT_MAX_DEG, TILT_MAX_DEG))
    return torch.maximum(minimum, torch.minimum(maximum, pose))


def identity_pixel_map(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``B,2,H,W`` output-to-input map in ``(x, y)`` pixels."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def pixel_map_to_normalized_grid(pixel_map: torch.Tensor) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    x = 2.0 * pixel_map[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * pixel_map[:, 1] / max(height - 1, 1) - 1.0
    return torch.stack((x, y), dim=-1)


def warp_tensor(
    image: torch.Tensor,
    pixel_map: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    return F.grid_sample(
        image,
        pixel_map_to_normalized_grid(pixel_map),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def integrate_stationary_velocity(
    velocity: torch.Tensor,
    integration_steps: int = 6,
) -> torch.Tensor:
    """Exponentiate a stationary pixel-velocity field by scaling and squaring."""
    if integration_steps < 0:
        raise ValueError("integration_steps cannot be negative")
    identity = identity_pixel_map(
        velocity.shape[0],
        velocity.shape[-2],
        velocity.shape[-1],
        device=velocity.device,
        dtype=velocity.dtype,
    )
    transform = identity + velocity / float(2**integration_steps)
    for _ in range(integration_steps):
        transform = warp_tensor(transform, transform)
    return transform


def _similarity_map(
    parameters: torch.Tensor,
    height: int,
    width: int,
    *,
    inverse: bool,
) -> torch.Tensor:
    identity = identity_pixel_map(
        parameters.shape[0],
        height,
        width,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    x = identity[:, 0] - (width - 1.0) / 2.0
    y = identity[:, 1] - (height - 1.0) / 2.0
    cosine, sine, translation_x, translation_y, log_scale = parameters.unbind(dim=1)
    cosine = cosine[:, None, None]
    sine = sine[:, None, None]
    scale = torch.exp(log_scale)[:, None, None]
    if inverse:
        x = x - translation_x[:, None, None]
        y = y - translation_y[:, None, None]
        mapped_x = (cosine * x + sine * y) / scale
        mapped_y = (-sine * x + cosine * y) / scale
    else:
        mapped_x = scale * (cosine * x - sine * y) + translation_x[:, None, None]
        mapped_y = scale * (sine * x + cosine * y) + translation_y[:, None, None]
    return torch.stack(
        (mapped_x + (width - 1.0) / 2.0, mapped_y + (height - 1.0) / 2.0),
        dim=1,
    )


def _apply_similarity(pixel_map: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    x = pixel_map[:, 0] - (width - 1.0) / 2.0
    y = pixel_map[:, 1] - (height - 1.0) / 2.0
    cosine, sine, translation_x, translation_y, log_scale = parameters.unbind(dim=1)
    cosine = cosine[:, None, None]
    sine = sine[:, None, None]
    scale = torch.exp(log_scale)[:, None, None]
    mapped_x = scale * (cosine * x - sine * y) + translation_x[:, None, None]
    mapped_y = scale * (sine * x + cosine * y) + translation_y[:, None, None]
    return torch.stack(
        (mapped_x + (width - 1.0) / 2.0, mapped_y + (height - 1.0) / 2.0),
        dim=1,
    )


def registration_maps(
    similarity_parameters: torch.Tensor,
    stationary_velocity: torch.Tensor,
    integration_steps: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-to-moving and moving-to-fixed output-to-input maps."""
    local_forward = integrate_stationary_velocity(stationary_velocity, integration_steps)
    local_inverse = integrate_stationary_velocity(-stationary_velocity, integration_steps)
    forward = _apply_similarity(local_forward, similarity_parameters)
    inverse_similarity = _similarity_map(
        similarity_parameters,
        stationary_velocity.shape[-2],
        stationary_velocity.shape[-1],
        inverse=True,
    )
    inverse = warp_tensor(local_inverse, inverse_similarity)
    return forward, inverse


def jacobian_determinant(pixel_map: torch.Tensor) -> torch.Tensor:
    """Finite-difference determinant on interior map cells."""
    dx = pixel_map[:, :, 1:, 1:] - pixel_map[:, :, 1:, :-1]
    dy = pixel_map[:, :, 1:, 1:] - pixel_map[:, :, :-1, 1:]
    return dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]


def affine_velocity_coefficients(velocity: torch.Tensor) -> torch.Tensor:
    """Project a pixel velocity onto ``translation + x + y`` affine bases.

    Coordinates are centered and normalized, so the three bases are mutually
    orthogonal on the rectangular model canvas.  The returned shape is
    ``B,2,3``: one ``(translation, x slope, y slope)`` row per vector component.
    """
    height, width = velocity.shape[-2:]
    x = torch.linspace(-1.0, 1.0, width, device=velocity.device, dtype=velocity.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=velocity.device, dtype=velocity.dtype)
    x = x[None, None, None, :]
    y = y[None, None, :, None]
    translation = velocity.mean(dim=(-2, -1))
    x_denominator = x.square().sum() * float(height)
    y_denominator = y.square().sum() * float(width)
    x_slope = (velocity * x).sum(dim=(-2, -1)) / x_denominator.clamp_min(1e-6)
    y_slope = (velocity * y).sum(dim=(-2, -1)) / y_denominator.clamp_min(1e-6)
    return torch.stack((translation, x_slope, y_slope), dim=2)


def project_affine_free_velocity(
    velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the best whole-canvas affine field and return its coefficients."""
    height, width = velocity.shape[-2:]
    x = torch.linspace(-1.0, 1.0, width, device=velocity.device, dtype=velocity.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=velocity.device, dtype=velocity.dtype)
    coefficients = affine_velocity_coefficients(velocity)
    fitted = (
        coefficients[:, :, 0, None, None]
        + coefficients[:, :, 1, None, None] * x[None, None, None, :]
        + coefficients[:, :, 2, None, None] * y[None, None, :, None]
    )
    return velocity - fitted, coefficients


def local_correlation(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """ONNX-friendly local normalized cost volume around every fixed pixel."""
    fixed = F.normalize(fixed, dim=1, eps=1e-6)
    moving = F.normalize(moving, dim=1, eps=1e-6)
    height, width = fixed.shape[-2:]
    padded = F.pad(moving, (radius, radius, radius, radius), mode="replicate")
    correlations = []
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            shifted = padded[
                :,
                :,
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
            correlations.append((fixed * shifted).sum(dim=1, keepdim=True))
    return torch.cat(correlations, dim=1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.project = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(image)
        residual = self.norm(residual)
        residual = self.project(F.gelu(self.expand(residual)))
        return image + residual


class _PyramidLevel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )
        self.residual = _ResidualBlock(output_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.residual(self.transition(image))


class StructuralPyramid(nn.Module):
    """Modality-specific three-channel stems and one tied anatomical encoder."""

    def __init__(self, channels: tuple[int, ...] = (24, 40, 64, 96)):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("the structural pyramid requires four levels")
        self.channels = tuple(channels)
        self.slice_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.atlas_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.levels = nn.ModuleList(
            [_PyramidLevel(channels[0], channels[0], 1)]
            + [
                _PyramidLevel(previous, current, 2)
                for previous, current in zip(channels[:-1], channels[1:])
            ]
        )

    @staticmethod
    def _input(
        image: torch.Tensor,
        outline_mask: torch.Tensor,
        mask_available: torch.Tensor,
    ) -> torch.Tensor:
        """Build the public image ABI without treating outline as tissue truth.

        Available outlines suppress pixels outside the user/automatic outline;
        absent outlines retain the raw background.  A third plane makes those
        cases distinguishable to the network, including imperfect outlines.
        """
        outline_mask = outline_mask.to(dtype=image.dtype)
        mask_available = mask_available.to(dtype=image.dtype).expand(
            -1, -1, image.shape[-2], image.shape[-1]
        )
        canonical_intensity = image * (
            1.0 - mask_available + mask_available * outline_mask
        )
        return torch.cat((canonical_intensity, outline_mask, mask_available), dim=1)

    def _encode(
        self,
        image: torch.Tensor,
        outline_mask: torch.Tensor,
        mask_available: torch.Tensor,
        stem: nn.Module,
    ):
        feature = stem(self._input(image, outline_mask, mask_available))
        features = []
        for level in self.levels:
            feature = level(feature)
            features.append(feature)
        return tuple(features)

    def encode_slice(
        self,
        image: torch.Tensor,
        outline_mask: torch.Tensor,
        mask_available: torch.Tensor,
    ):
        return self._encode(image, outline_mask, mask_available, self.slice_stem)

    def encode_atlas(
        self,
        image: torch.Tensor,
        outline_mask: torch.Tensor,
        mask_available: torch.Tensor,
    ):
        return self._encode(image, outline_mask, mask_available, self.atlas_stem)


class ProbabilisticPoseHead(nn.Module):
    """Coarse axis distributions with a continuous, physically bounded residual."""

    def __init__(
        self,
        feature_channels: tuple[int, ...],
        context_features: int = 128,
        ap_bins: int = 41,
        tilt_bins: int = 29,
    ):
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(sum(feature_channels)),
            nn.Linear(sum(feature_channels), context_features),
            nn.GELU(),
            nn.Linear(context_features, context_features),
            nn.GELU(),
        )
        self.ap_logits = nn.Linear(context_features, ap_bins)
        self.lr_logits = nn.Linear(context_features, tilt_bins)
        self.dv_logits = nn.Linear(context_features, tilt_bins)
        self.residual = nn.Linear(context_features, 3)
        self.local_cholesky = nn.Linear(context_features, 6)
        self.register_buffer("ap_centers", torch.linspace(AP_MIN_UM, AP_MAX_UM, ap_bins))
        self.register_buffer(
            "tilt_centers", torch.linspace(TILT_MIN_DEG, TILT_MAX_DEG, tilt_bins)
        )
        self.register_buffer(
            "maximum_residual",
            torch.tensor(
                (
                    (AP_MAX_UM - AP_MIN_UM) / (ap_bins - 1) / 2.0,
                    (TILT_MAX_DEG - TILT_MIN_DEG) / (tilt_bins - 1) / 2.0,
                    (TILT_MAX_DEG - TILT_MIN_DEG) / (tilt_bins - 1) / 2.0,
                )
            ),
        )
        self.register_buffer("physical_pose_scale", torch.tensor(POSE_SCALE))
        for head in (self.ap_logits, self.lr_logits, self.dv_logits):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.normal_(self.residual.weight, std=1e-3)
        nn.init.zeros_(self.residual.bias)
        nn.init.normal_(self.local_cholesky.weight, std=1e-3)
        nn.init.zeros_(self.local_cholesky.bias)
        nn.init.constant_(self.local_cholesky.bias[:3], -3.5)

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        pooled = torch.cat([feature.mean(dim=(-2, -1)) for feature in features], dim=1)
        context = self.context(pooled)
        ap_logits = self.ap_logits(context)
        lr_logits = self.lr_logits(context)
        dv_logits = self.dv_logits(context)
        ap_probability = torch.softmax(ap_logits, dim=1)
        lr_probability = torch.softmax(lr_logits, dim=1)
        dv_probability = torch.softmax(dv_logits, dim=1)
        coarse_pose = torch.stack(
            (
                (ap_probability * self.ap_centers).sum(dim=1),
                (lr_probability * self.tilt_centers).sum(dim=1),
                (dv_probability * self.tilt_centers).sum(dim=1),
            ),
            dim=1,
        )
        continuous_residual = torch.tanh(self.residual(context)) * self.maximum_residual
        # AP uncertainty is measured in micrometres, so its variance routinely
        # exceeds float16's finite range.  Keep this probabilistic branch in
        # float32 even when the image encoder and point heads run under AMP.
        with torch.amp.autocast(device_type=context.device.type, enabled=False):
            raw_cholesky = self.local_cholesky(context.float())
            diagonal = F.softplus(raw_cholesky[:, :3]) + 1e-4
            off_diagonal = 0.25 * torch.tanh(raw_cholesky[:, 3:])
            zeros = torch.zeros_like(diagonal[:, 0])
            normalized_cholesky = torch.stack(
                (
                    torch.stack((diagonal[:, 0], zeros, zeros), dim=1),
                    torch.stack((off_diagonal[:, 0], diagonal[:, 1], zeros), dim=1),
                    torch.stack(
                        (off_diagonal[:, 1], off_diagonal[:, 2], diagonal[:, 2]),
                        dim=1,
                    ),
                ),
                dim=1,
            )
            physical_scale = self.physical_pose_scale.float()[None, :, None]
            pose_cholesky = normalized_cholesky * physical_scale
            pose_covariance = pose_cholesky @ pose_cholesky.transpose(1, 2)
        return {
            "pose": project_pose_to_domain(coarse_pose + continuous_residual),
            "coarse_pose": coarse_pose,
            "continuous_residual": continuous_residual,
            "ap_logits": ap_logits,
            "lr_logits": lr_logits,
            "dv_logits": dv_logits,
            "ap_probability": ap_probability,
            "lr_probability": lr_probability,
            "dv_probability": dv_probability,
            "pose_cholesky": pose_cholesky,
            "pose_covariance": pose_covariance,
            "pose_context": context,
        }


class ConvGRUCell(nn.Module):
    """Spatial recurrent state shared across render-compare-correct iterations."""

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        joined = input_channels + hidden_channels
        self.gates = nn.Conv2d(joined, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(joined, hidden_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class _DenseDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        pyramid_channels: tuple[int, ...],
    ):
        super().__init__()
        fine, low, middle, _ = pyramid_channels
        self.middle = _PyramidLevel(hidden_channels + middle + 9, hidden_channels, 1)
        self.low = _PyramidLevel(hidden_channels + low + 9, middle, 1)
        self.fine = _PyramidLevel(middle + fine, low, 1)
        self.velocity_head = nn.Conv2d(low, 2, 3, padding=1)
        self.validity_head = nn.Conv2d(low, 1, 1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        fixed_features: tuple[torch.Tensor, ...],
        moving_features: tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = F.interpolate(
            hidden, size=fixed_features[2].shape[-2:], mode="bilinear", align_corners=True
        )
        middle_correlation = local_correlation(fixed_features[2], moving_features[2], 1)
        feature = self.middle(
            torch.cat(
                (feature, torch.abs(fixed_features[2] - moving_features[2]), middle_correlation),
                dim=1,
            )
        )
        feature = F.interpolate(
            feature, size=fixed_features[1].shape[-2:], mode="bilinear", align_corners=True
        )
        low_correlation = local_correlation(fixed_features[1], moving_features[1], 1)
        feature = self.low(
            torch.cat(
                (feature, torch.abs(fixed_features[1] - moving_features[1]), low_correlation),
                dim=1,
            )
        )
        feature = F.interpolate(
            feature, size=fixed_features[0].shape[-2:], mode="bilinear", align_corners=True
        )
        feature = self.fine(
            torch.cat((feature, torch.abs(fixed_features[0] - moving_features[0])), dim=1)
        )
        velocity = F.interpolate(
            self.velocity_head(feature), size=output_size, mode="bilinear", align_corners=True
        )
        validity_logits = F.interpolate(
            self.validity_head(feature), size=output_size, mode="bilinear", align_corners=True
        )
        return velocity, validity_logits


class IndependentJointModel(nn.Module):
    """From-scratch recurrent model for atlas-plane pose and nonlinear maps."""

    learned_weight_dependencies: tuple[str, ...] = ()
    initialization: str = "random"

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
        super().__init__()
        self.pyramid = StructuralPyramid(pyramid_channels)
        self.pose_head = ProbabilisticPoseHead(
            pyramid_channels, context_features=pose_context_features
        )
        coarse_channels = pyramid_channels[-1]
        correlation_channels = 25 + 9
        self.pair_projection = nn.Sequential(
            nn.Conv2d(coarse_channels * 2 + correlation_channels, pair_features, 1),
            nn.GroupNorm(_group_count(pair_features), pair_features),
            nn.GELU(),
            _ResidualBlock(pair_features),
        )
        condition_features = max(pair_features // 2, 16)
        self.condition = nn.Sequential(
            nn.LayerNorm(pose_context_features + 3),
            nn.Linear(pose_context_features + 3, condition_features),
            nn.GELU(),
        )
        self.recurrent = ConvGRUCell(pair_features + condition_features, hidden_channels)
        self.pose_delta_head = nn.Linear(hidden_channels, 3)
        self.similarity_head = nn.Linear(hidden_channels, 5)
        self.compatibility_head = nn.Linear(hidden_channels, 1)
        self.decoder = _DenseDecoder(hidden_channels, pyramid_channels)
        self.integration_steps = int(integration_steps)
        self.maximum_translation_pixels = float(maximum_translation_pixels)
        self.maximum_velocity_fraction = float(maximum_velocity_fraction)
        self.register_buffer("pose_center", torch.tensor(POSE_CENTER))
        self.register_buffer("pose_scale", torch.tensor(POSE_SCALE))
        self.register_buffer("maximum_pose_delta", torch.tensor(maximum_pose_delta))
        self.register_buffer(
            "log_scale_limits",
            torch.tensor((-math.log(minimum_scale), math.log(maximum_scale))),
        )
        nn.init.zeros_(self.pose_delta_head.weight)
        nn.init.zeros_(self.pose_delta_head.bias)
        nn.init.zeros_(self.similarity_head.weight)
        nn.init.zeros_(self.similarity_head.bias)
        with torch.no_grad():
            self.similarity_head.bias[0] = 1.0

    def initialize(
        self,
        slice_image: torch.Tensor,
        slice_outline_mask: torch.Tensor,
        slice_mask_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_source(
            slice_image, slice_outline_mask, slice_mask_available
        )
        return self.pose_head(features)

    def encode_source(
        self,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Cacheable source encoding shared by every rendered atlas candidate."""
        return self.pyramid.encode_slice(
            source_image, source_outline_mask, source_mask_available
        )

    def initial_hidden_state(
        self,
        atlas_image: torch.Tensor,
    ) -> torch.Tensor:
        divisor = 2 ** len(self.pyramid.channels)
        height = (atlas_image.shape[-2] + divisor - 1) // divisor
        width = (atlas_image.shape[-1] + divisor - 1) // divisor
        return atlas_image.new_zeros(
            atlas_image.shape[0], self.recurrent.gates.out_channels // 2, height, width
        )

    @staticmethod
    def _expand_source_features(
        source_features: tuple[torch.Tensor, ...], batch_size: int
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            feature.expand(batch_size, -1, -1, -1) for feature in source_features
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
        if source_index is None:
            source_features = self._expand_source_features(
                source_features, fixed_features[0].shape[0]
            )
            pose_context = pose_context.expand(current_pose.shape[0], -1)
        else:
            source_features = tuple(
                feature.index_select(0, source_index) for feature in source_features
            )
            pose_context = pose_context.index_select(0, source_index)
        coarse_fixed = fixed_features[-1]
        coarse_source = source_features[-1]
        coarse_correlation = local_correlation(coarse_fixed, coarse_source, 2)
        middle_correlation = local_correlation(fixed_features[-2], source_features[-2], 1)
        middle_correlation = F.interpolate(
            middle_correlation,
            size=coarse_fixed.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        pair = self.pair_projection(
            torch.cat(
                (
                    torch.abs(coarse_fixed - coarse_source),
                    coarse_fixed * coarse_source,
                    coarse_correlation,
                    middle_correlation,
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

    def score_candidate_from_features(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None,
        source_features: tuple[torch.Tensor, ...],
        source_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score/update one or many poses without dense decoding or map integration."""
        fixed_features = self.pyramid.encode_atlas(
            atlas_image, atlas_outline_mask, atlas_mask_available
        )
        outputs, _, _ = self._coarse_update(
            fixed_features,
            source_features,
            current_pose,
            pose_context,
            hidden_state,
            source_index,
        )
        return outputs

    def score_candidate(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        source_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_features = self.encode_source(
            source_image, source_outline_mask, source_mask_available
        )
        return self.score_candidate_from_features(
            atlas_image,
            atlas_outline_mask,
            atlas_mask_available,
            current_pose,
            pose_context,
            hidden_state,
            source_features,
            source_index,
        )

    def refine_from_features(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None,
        source_features: tuple[torch.Tensor, ...],
        source_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fixed_features = self.pyramid.encode_atlas(
            atlas_image, atlas_outline_mask, atlas_mask_available
        )
        outputs, pooled, source_features = self._coarse_update(
            fixed_features,
            source_features,
            current_pose,
            pose_context,
            hidden_state,
            source_index,
        )
        raw_similarity = self.similarity_head(pooled)
        angle_vector = F.normalize(raw_similarity[:, :2], dim=1, eps=1e-6)
        normalized_scale = torch.tanh(raw_similarity[:, 4])
        log_scale = torch.where(
            normalized_scale >= 0.0,
            normalized_scale * self.log_scale_limits[1],
            normalized_scale * self.log_scale_limits[0],
        )
        similarity = torch.stack(
            (
                angle_vector[:, 0],
                angle_vector[:, 1],
                torch.tanh(raw_similarity[:, 2]) * self.maximum_translation_pixels,
                torch.tanh(raw_similarity[:, 3]) * self.maximum_translation_pixels,
                log_scale,
            ),
            dim=1,
        )
        height, width = atlas_image.shape[-2:]
        raw_velocity, validity_logits = self.decoder(
            outputs["hidden_state"], fixed_features, source_features, (height, width)
        )
        raw_velocity = F.avg_pool2d(
            raw_velocity, kernel_size=5, stride=1, padding=2, count_include_pad=False
        )
        bounded_velocity = (
            torch.tanh(raw_velocity)
            * self.maximum_velocity_fraction
            * float(min(height, width))
        )
        stationary_velocity, affine_coefficients = project_affine_free_velocity(
            bounded_velocity
        )
        fixed_to_moving, moving_to_fixed = registration_maps(
            similarity, stationary_velocity, self.integration_steps
        )
        outputs.update(
            {
                "similarity_parameters": similarity,
                "stationary_velocity": stationary_velocity,
                "affine_velocity_coefficients": affine_coefficients,
                "fixed_to_moving_map": fixed_to_moving,
                "moving_to_fixed_map": moving_to_fixed,
                "validity_logits": validity_logits,
                "validity_probability": torch.sigmoid(validity_logits),
            }
        )
        return outputs

    def refine_once(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        source_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_features = self.encode_source(
            source_image, source_outline_mask, source_mask_available
        )
        return self.refine_from_features(
            atlas_image,
            atlas_outline_mask,
            atlas_mask_available,
            current_pose,
            pose_context,
            hidden_state,
            source_features,
            source_index,
        )

    def forward(
        self,
        slice_image: torch.Tensor,
        slice_outline_mask: torch.Tensor,
        slice_mask_available: torch.Tensor,
    ) -> torch.Tensor:
        return self.initialize(
            slice_image, slice_outline_mask, slice_mask_available
        )["pose"]


class IndependentInitializerExport(nn.Module):
    """Initial pose distribution plus cacheable source pyramid for ONNX."""

    def __init__(self, model: IndependentJointModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        slice_image: torch.Tensor,
        slice_outline_mask: torch.Tensor,
        slice_mask_available: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features = self.model.encode_source(
            slice_image, slice_outline_mask, slice_mask_available
        )
        outputs = self.model.pose_head(features)
        return (
            outputs["pose"],
            outputs["pose_context"],
            outputs["ap_logits"],
            outputs["lr_logits"],
            outputs["dv_logits"],
            outputs["pose_cholesky"],
            *features,
        )


class IndependentCandidateScorerExport(nn.Module):
    """Cheap cached-feature recurrent candidate scorer for ONNX runtimes."""

    def __init__(self, model: IndependentJointModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor,
        source_index: torch.Tensor,
        source_feature_2: torch.Tensor,
        source_feature_3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model.score_candidate_from_features(
            atlas_image,
            atlas_outline_mask,
            atlas_mask_available,
            current_pose,
            pose_context,
            hidden_state,
            (source_feature_2, source_feature_3),
            source_index,
        )
        return (
            outputs["pose"],
            outputs["pose_delta"],
            outputs["compatibility_logit"],
            outputs["hidden_state"],
        )


class IndependentCachedRefinerExport(nn.Module):
    """Final dense decode using source features cached by the initializer."""

    def __init__(self, model: IndependentJointModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor,
        source_index: torch.Tensor,
        source_feature_0: torch.Tensor,
        source_feature_1: torch.Tensor,
        source_feature_2: torch.Tensor,
        source_feature_3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model.refine_from_features(
            atlas_image,
            atlas_outline_mask,
            atlas_mask_available,
            current_pose,
            pose_context,
            hidden_state,
            (source_feature_0, source_feature_1, source_feature_2, source_feature_3),
            source_index,
        )
        return (
            outputs["pose"],
            outputs["pose_delta"],
            outputs["similarity_parameters"],
            outputs["stationary_velocity"],
            outputs["affine_velocity_coefficients"],
            outputs["fixed_to_moving_map"],
            outputs["moving_to_fixed_map"],
            outputs["compatibility_logit"],
            outputs["validity_logits"],
            outputs["hidden_state"],
        )


class IndependentRefinerExport(nn.Module):
    """ONNX entry graph for one shared recurrent render-compare-correct step."""

    def __init__(self, model: IndependentJointModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        atlas_image: torch.Tensor,
        atlas_outline_mask: torch.Tensor,
        atlas_mask_available: torch.Tensor,
        source_image: torch.Tensor,
        source_outline_mask: torch.Tensor,
        source_mask_available: torch.Tensor,
        current_pose: torch.Tensor,
        pose_context: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model.refine_once(
            atlas_image,
            atlas_outline_mask,
            atlas_mask_available,
            source_image,
            source_outline_mask,
            source_mask_available,
            current_pose,
            pose_context,
            hidden_state,
        )
        return (
            outputs["pose"],
            outputs["pose_delta"],
            outputs["similarity_parameters"],
            outputs["stationary_velocity"],
            outputs["affine_velocity_coefficients"],
            outputs["fixed_to_moving_map"],
            outputs["moving_to_fixed_map"],
            outputs["compatibility_logit"],
            outputs["validity_logits"],
            outputs["hidden_state"],
        )
