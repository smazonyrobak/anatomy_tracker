from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# Canonical dense v2 utilities use absolute (x, y) pixel maps with align_corners=True sampling.
def identity_pixel_map(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return B,2,H,W identity coordinates ordered as x, y in pixels."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def pixel_map_to_normalized_grid(
    pixel_map: torch.Tensor,
    input_size: tuple[int, int],
) -> torch.Tensor:
    """Convert an x,y pixel map to an align_corners=True grid_sample grid."""
    height, width = input_size
    x = pixel_map[:, 0] * (2.0 / max(width - 1, 1)) - 1.0
    y = pixel_map[:, 1] * (2.0 / max(height - 1, 1)) - 1.0
    return torch.stack((x, y), dim=-1)


def warp_tensor(
    tensor: torch.Tensor,
    output_to_input_map: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Sample an input B,C,H,W tensor onto the map's output pixel grid."""
    return F.grid_sample(
        tensor,
        pixel_map_to_normalized_grid(output_to_input_map, tensor.shape[-2:]),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def compose_pixel_maps(
    output_to_middle: torch.Tensor,
    middle_to_input: torch.Tensor,
) -> torch.Tensor:
    """Compose maps as middle_to_input(output_to_middle(x))."""
    return warp_tensor(
        middle_to_input,
        output_to_middle,
        padding_mode="border",
    )


def resize_vector_field(
    vector_field: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Resize an x,y pixel vector field while preserving physical displacement."""
    old_height, old_width = vector_field.shape[-2:]
    new_height, new_width = output_size
    resized = F.interpolate(
        vector_field,
        size=output_size,
        mode="bilinear",
        align_corners=True,
    )
    scale = resized.new_tensor(
        (
            (new_width - 1) / max(old_width - 1, 1),
            (new_height - 1) / max(old_height - 1, 1),
        )
    )[None, :, None, None]
    return resized * scale


def local_dot_product_correlation(
    fixed_feature: torch.Tensor,
    moving_feature: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Return L2-normalized local fixed/moving correlations.

    Channels are ordered row-major from offset ``(-radius, -radius)`` to
    ``(+radius, +radius)``. Explicit pad/slices keep the operation exportable
    with the legacy ONNX path used by the packaged application.
    """
    if radius < 0:
        raise ValueError("correlation radius cannot be negative")
    fixed = F.normalize(fixed_feature, p=2.0, dim=1, eps=1e-6)
    moving = F.normalize(moving_feature, p=2.0, dim=1, eps=1e-6)
    if radius == 0:
        return (fixed * moving).sum(dim=1, keepdim=True)
    height, width = fixed.shape[-2:]
    padded = F.pad(moving, (radius, radius, radius, radius))
    correlations = []
    for offset_y in range(-radius, radius + 1):
        top = radius + offset_y
        for offset_x in range(-radius, radius + 1):
            left = radius + offset_x
            shifted = padded[..., top : top + height, left : left + width]
            correlations.append((fixed * shifted).sum(dim=1, keepdim=True))
    return torch.cat(correlations, dim=1)


def _shift_replicate(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    y0, x0 = 1 + dy, 1 + dx
    return padded[..., y0 : y0 + image.shape[-2], x0 : x0 + image.shape[-1]]


def modality_independent_descriptor(image: torch.Tensor) -> torch.Tensor:
    """Return a compact six-neighbour 2-D MIND-style descriptor."""
    descriptors = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)):
        difference = (image - _shift_replicate(image, dx, dy)).square()
        descriptors.append(F.avg_pool2d(difference, 5, stride=1, padding=2))
    descriptor = torch.cat(descriptors, dim=1)
    descriptor = descriptor - descriptor.amin(dim=1, keepdim=True)
    scale = descriptor.mean(dim=1, keepdim=True).clamp_min(1e-4)
    return torch.exp(-descriptor / scale)


def scaling_and_squaring(
    stationary_velocity: torch.Tensor,
    steps: int = 7,
) -> torch.Tensor:
    """Exponentiate a stationary x,y velocity field into a pixel coordinate map."""
    displacement = stationary_velocity / float(2**steps)
    identity = identity_pixel_map(
        stationary_velocity.shape[0],
        stationary_velocity.shape[-2],
        stationary_velocity.shape[-1],
        device=stationary_velocity.device,
        dtype=stationary_velocity.dtype,
    )
    for _ in range(steps):
        displacement = displacement + warp_tensor(
            displacement,
            identity + displacement,
            padding_mode="border",
        )
    return identity + displacement


def integrate_stationary_velocity(
    stationary_velocity: torch.Tensor,
    steps: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exp(v) and exp(-v), a paired diffeomorphic map and inverse."""
    return (
        scaling_and_squaring(stationary_velocity, steps),
        scaling_and_squaring(-stationary_velocity, steps),
    )


def apply_similarity_transform(
    pixel_map: torch.Tensor,
    parameters: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply [angle_rad, tx_px, ty_px, log_scale] about the image centre."""
    height, width = pixel_map.shape[-2:]
    angle, translation_x, translation_y, log_scale = parameters.unbind(dim=1)
    cosine = torch.cos(angle)[:, None, None]
    sine = torch.sin(angle)[:, None, None]
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    if inverse:
        x = pixel_map[:, 0] - centre_x - translation_x[:, None, None]
        y = pixel_map[:, 1] - centre_y - translation_y[:, None, None]
        scale = torch.exp(-log_scale)[:, None, None]
        transformed_x = scale * (cosine * x + sine * y) + centre_x
        transformed_y = scale * (-sine * x + cosine * y) + centre_y
    else:
        x = pixel_map[:, 0] - centre_x
        y = pixel_map[:, 1] - centre_y
        scale = torch.exp(log_scale)[:, None, None]
        transformed_x = (
            scale * (cosine * x - sine * y)
            + centre_x
            + translation_x[:, None, None]
        )
        transformed_y = (
            scale * (sine * x + cosine * y)
            + centre_y
            + translation_y[:, None, None]
        )
    return torch.stack((transformed_x, transformed_y), dim=1)


def resize_similarity_parameters(
    parameters: torch.Tensor,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Express similarity translations in another pyramid level's pixel units."""
    input_height, input_width = input_size
    output_height, output_width = output_size
    translation_scale = parameters.new_tensor(
        (
            (output_width - 1) / max(input_width - 1, 1),
            (output_height - 1) / max(input_height - 1, 1),
        )
    )
    return torch.cat(
        (
            parameters[:, :1],
            parameters[:, 1:3] * translation_scale,
            parameters[:, 3:4],
        ),
        dim=1,
    )


# Similarity and integrated stationary velocity compose explicitly into forward and inverse maps.
def registration_maps(
    similarity_parameters: torch.Tensor,
    local_velocity: torch.Tensor,
    steps: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-atlas->moving-slice and moving-slice->fixed-atlas maps.

    The local map acts in fixed coordinates and the global similarity remains an
    explicit transform: forward = similarity o exp(v). The inverse is derived
    from the same parameters: exp(-v) o inverse(similarity).
    """
    local_forward, local_inverse = integrate_stationary_velocity(local_velocity, steps)
    fixed_to_moving = apply_similarity_transform(
        local_forward,
        similarity_parameters,
    )
    identity = identity_pixel_map(
        local_velocity.shape[0],
        local_velocity.shape[-2],
        local_velocity.shape[-1],
        device=local_velocity.device,
        dtype=local_velocity.dtype,
    )
    moving_to_similarity_fixed = apply_similarity_transform(
        identity,
        similarity_parameters,
        inverse=True,
    )
    moving_to_fixed = moving_to_similarity_fixed + warp_tensor(
        local_inverse - identity,
        moving_to_similarity_fixed,
        padding_mode="border",
    )
    return fixed_to_moving, moving_to_fixed


def jacobian_determinant(pixel_map: torch.Tensor) -> torch.Tensor:
    """Return B,1,H,W determinants of d(input x,y)/d(output x,y)."""
    derivative_x = _finite_difference(pixel_map, dimension=-1)
    derivative_y = _finite_difference(pixel_map, dimension=-2)
    determinant = (
        derivative_x[:, 0] * derivative_y[:, 1]
        - derivative_y[:, 0] * derivative_x[:, 1]
    )
    return determinant[:, None]


def _finite_difference(tensor: torch.Tensor, dimension: int) -> torch.Tensor:
    length = tensor.shape[dimension]
    if length == 1:
        return torch.zeros_like(tensor)
    slices = [slice(None)] * tensor.ndim
    first = slices.copy()
    first[dimension] = slice(0, 1)
    second = slices.copy()
    second[dimension] = slice(1, 2)
    before_last = slices.copy()
    before_last[dimension] = slice(-2, -1)
    last = slices.copy()
    last[dimension] = slice(-1, None)
    leading = tensor[tuple(second)] - tensor[tuple(first)]
    trailing = tensor[tuple(last)] - tensor[tuple(before_last)]
    if length == 2:
        return torch.cat((leading, trailing), dim=dimension)
    left = slices.copy()
    left[dimension] = slice(0, -2)
    right = slices.copy()
    right[dimension] = slice(2, None)
    centre = 0.5 * (tensor[tuple(right)] - tensor[tuple(left)])
    return torch.cat((leading, centre, trailing), dim=dimension)


def _group_count(channel_count: int) -> int:
    for group_count in (8, 4, 2):
        if channel_count % group_count == 0:
            return group_count
    return 1


class _FeatureBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class _PreActivationResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.activation1(self.norm1(inputs)))
        residual = self.conv2(self.activation2(self.norm2(residual)))
        return inputs + residual


class _Residual5RegistrationEstimator(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual_blocks = nn.ModuleList(
            _PreActivationResidualBlock(output_channels) for _ in range(5)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(inputs)
        for block in self.residual_blocks:
            features = block(features)
        return features


class _SiameseEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: tuple[int, ...]):
        super().__init__()
        self.stem = _FeatureBlock(input_channels, channels[0])
        self.down = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(previous, current, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_group_count(current), current),
                nn.LeakyReLU(0.2, inplace=True),
                _FeatureBlock(current, current),
            )
            for previous, current in zip(channels[:-1], channels[1:])
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.stem(image)
        features = [feature]
        for block in self.down:
            feature = block(feature)
            features.append(feature)
        return tuple(features)


class _ResidualSimilarityHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        maximum_rotation_degrees: float,
        maximum_translation_fraction: float,
        maximum_scale: float,
    ):
        super().__init__()
        self.maximum_rotation_radians = math.radians(maximum_rotation_degrees)
        self.maximum_translation_fraction = maximum_translation_fraction
        self.maximum_log_scale = math.log(maximum_scale)
        spatial_channels = max(feature_channels // 2, 8)
        self.spatial = nn.Sequential(
            nn.Conv2d(feature_channels * 3, spatial_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(spatial_channels), spatial_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(size=(4, 6), mode="bilinear", align_corners=True),
        )
        self.network = nn.Sequential(
            nn.Linear(spatial_channels * 24, feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feature_channels * 2, 4),
        )
        nn.init.normal_(self.network[-1].weight, std=1e-5)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        fixed_feature: torch.Tensor,
        moving_feature: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        pair_features = torch.cat(
            (
                fixed_feature,
                moving_feature,
                torch.abs(fixed_feature - moving_feature),
            ),
            dim=1,
        )
        raw = self.network(self.spatial(pair_features).flatten(1))
        height, width = image_size
        return torch.cat(
            (
                torch.tanh(raw[:, :1]) * self.maximum_rotation_radians,
                torch.tanh(raw[:, 1:2])
                * self.maximum_translation_fraction
                * (width - 1),
                torch.tanh(raw[:, 2:3])
                * self.maximum_translation_fraction
                * (height - 1),
                torch.tanh(raw[:, 3:4]) * self.maximum_log_scale,
            ),
            dim=1,
        )


# Siamese multiscale features drive global similarity and diffeomorphic residual refinement.
class DenseRegistrationModel(nn.Module):
    """Selected coarse-to-fine Siamese deformable registration network.

    ``forward(fixed_atlas, moving_slice)`` returns two B,2,H,W pixel maps:
    fixed-atlas output -> moving-slice input, followed by its derived inverse.
    """

    def __init__(
        self,
        input_channels: int = 2,
        channels: tuple[int, ...] = (16, 24, 32, 48),
        integration_steps: int = 7,
        maximum_rotation_degrees: float = 30.0,
        maximum_translation_fraction: float = 0.10,
        maximum_scale: float = 1.25,
        maximum_local_velocity_fraction: float = 0.15,
        correlation_radii: tuple[int, ...] | list[int] = (4, 3, 2, 2),
    ):
        super().__init__()
        self.integration_steps = integration_steps
        self.maximum_local_velocity_fraction = maximum_local_velocity_fraction
        self.correlation_radii = tuple(int(radius) for radius in correlation_radii)
        if len(self.correlation_radii) != len(channels):
            raise ValueError("correlation_radii must contain one coarse-to-fine radius per stage")
        if any(radius < 0 for radius in self.correlation_radii):
            raise ValueError("correlation radii cannot be negative")
        self.encoder = _SiameseEncoder(input_channels, channels)
        self.similarity_head = _ResidualSimilarityHead(
            channels[-1],
            maximum_rotation_degrees,
            maximum_translation_fraction,
            maximum_scale,
        )
        coarse_to_fine_channels = tuple(reversed(channels))
        previous_context_channels = 0
        stages = []
        velocity_heads = []
        for feature_channels, correlation_radius in zip(
            coarse_to_fine_channels, self.correlation_radii
        ):
            correlation_channels = (
                (2 * correlation_radius + 1) ** 2 if correlation_radius else 0
            )
            stage_input_channels = (
                feature_channels * 3
                + previous_context_channels
                + correlation_channels
            )
            stages.append(
                _Residual5RegistrationEstimator(
                    stage_input_channels,
                    feature_channels,
                )
            )
            velocity_head = nn.Conv2d(feature_channels, 2, 3, padding=1)
            nn.init.normal_(velocity_head.weight, std=1e-5)
            nn.init.zeros_(velocity_head.bias)
            velocity_heads.append(velocity_head)
            previous_context_channels = feature_channels
        self.registration_stages = nn.ModuleList(stages)
        self.velocity_heads = nn.ModuleList(velocity_heads)

    def _predict(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        fixed_features = self.encoder(fixed_atlas)
        moving_features = self.encoder(moving_slice)
        image_size = fixed_atlas.shape[-2:]
        similarity_parameters = self.similarity_head(
            fixed_features[-1],
            moving_features[-1],
            image_size,
        )
        cumulative_velocity = None
        context = None
        pyramid_velocities = []
        feature_steps = max(self.integration_steps - 2, 0)
        for (
            fixed_feature,
            moving_feature,
            stage,
            velocity_head,
            correlation_radius,
        ) in zip(
            reversed(fixed_features),
            reversed(moving_features),
            self.registration_stages,
            self.velocity_heads,
            self.correlation_radii,
        ):
            level_size = fixed_feature.shape[-2:]
            if cumulative_velocity is None:
                cumulative_velocity = fixed_feature.new_zeros(
                    (fixed_feature.shape[0], 2, level_size[0], level_size[1])
                )
            else:
                cumulative_velocity = resize_vector_field(cumulative_velocity, level_size)
            level_similarity = resize_similarity_parameters(
                similarity_parameters,
                image_size,
                level_size,
            )
            local_map = scaling_and_squaring(cumulative_velocity, feature_steps)
            fixed_to_moving = apply_similarity_transform(local_map, level_similarity)
            warped_moving = warp_tensor(
                moving_feature,
                fixed_to_moving,
                padding_mode="border",
            )
            stage_inputs = [
                fixed_feature,
                warped_moving,
                torch.abs(fixed_feature - warped_moving),
            ]
            if correlation_radius:
                stage_inputs.append(
                    local_dot_product_correlation(
                        fixed_feature,
                        warped_moving,
                        correlation_radius,
                    )
                )
            if context is not None:
                stage_inputs.append(
                    F.interpolate(
                        context,
                        size=level_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            context = stage(torch.cat(stage_inputs, dim=1))
            raw_velocity = F.avg_pool2d(
                velocity_head(context),
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            )
            cumulative_velocity = cumulative_velocity + (
                torch.tanh(raw_velocity)
                * self.maximum_local_velocity_fraction
                * min(level_size)
            )
            pyramid_velocities.append(cumulative_velocity)
        return (
            similarity_parameters,
            cumulative_velocity,
            tuple(pyramid_velocities),
        )

    def forward_with_details(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        similarity, velocity, pyramid = self._predict(fixed_atlas, moving_slice)
        fixed_to_moving, moving_to_fixed = registration_maps(
            similarity,
            velocity,
            self.integration_steps,
        )
        return {
            "fixed_to_moving_map": fixed_to_moving,
            "moving_to_fixed_map": moving_to_fixed,
            "similarity_parameters": similarity,
            "local_velocity": velocity,
            "pyramid_velocities": pyramid,
        }

    def forward_for_review(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Return reviewer inputs without integrating the unused inverse map."""
        similarity, velocity, pyramid = self._predict(fixed_atlas, moving_slice)
        local_forward = scaling_and_squaring(velocity, self.integration_steps)
        return {
            "fixed_to_moving_map": apply_similarity_transform(
                local_forward, similarity
            ),
            "similarity_parameters": similarity,
            "local_velocity": velocity,
            "pyramid_velocities": pyramid,
        }

    def forward(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarity, velocity, _ = self._predict(fixed_atlas, moving_slice)
        return registration_maps(similarity, velocity, self.integration_steps)
