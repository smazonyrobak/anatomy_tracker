"""Residual in-plane diffeomorphic registration after atlas pose is frozen.

Maps use absolute pixel coordinates in ``(x, y)`` channel order.  Thus
``warp_atlas_to_affine(moving, atlas_to_affine)`` resamples the affine-aligned
slice onto the atlas grid.  AP position, section tilts, and the surface affine
must be solved before this module is called; the velocity projection below
removes translation and every linear (scale, rotation, and shear) component.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


PADDED_SIZE = (320, 464)


def pixel_identity_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def _normalized_sampling_grid(pixel_map: torch.Tensor) -> torch.Tensor:
    height, width = pixel_map.shape[-2:]
    x = pixel_map[:, 0] * (2.0 / (width - 1)) - 1.0
    y = pixel_map[:, 1] * (2.0 / (height - 1)) - 1.0
    return torch.stack((x, y), dim=-1)


def sample_at_pixel_map(
    image: torch.Tensor,
    pixel_map: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    return F.grid_sample(
        image,
        _normalized_sampling_grid(pixel_map),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def remove_global_affine(velocity: torch.Tensor) -> torch.Tensor:
    """Orthogonally remove the best field ``a + b*x + c*y`` per batch/channel."""
    batch, _, height, width = velocity.shape
    grid = pixel_identity_grid(batch, height, width, device=velocity.device, dtype=velocity.dtype)
    x = grid[:, :1] * (2.0 / (width - 1)) - 1.0
    y = grid[:, 1:] * (2.0 / (height - 1)) - 1.0
    basis = torch.cat((torch.ones_like(x), x, y), dim=1).flatten(2)
    flat = velocity.flatten(2)
    coefficients = torch.matmul(flat, basis.transpose(1, 2)) / basis.square().sum(dim=2).unsqueeze(1)
    affine = torch.matmul(coefficients, basis).reshape_as(velocity)
    return velocity - affine


def integrate_stationary_velocity(velocity: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Return the absolute pixel map ``exp(velocity)`` by scaling and squaring."""
    displacement = velocity / float(2**steps)
    identity = pixel_identity_grid(
        velocity.shape[0],
        velocity.shape[-2],
        velocity.shape[-1],
        device=velocity.device,
        dtype=velocity.dtype,
    )
    for _ in range(steps):
        displacement = displacement + sample_at_pixel_map(displacement, identity + displacement)
    return identity + displacement


def compose_pixel_maps(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose maps as ``second(first(x))``."""
    return sample_at_pixel_map(second, first)


def jacobian_determinant(pixel_map: torch.Tensor) -> torch.Tensor:
    d_dx = pixel_map[:, :, :-1, 1:] - pixel_map[:, :, :-1, :-1]
    d_dy = pixel_map[:, :, 1:, :-1] - pixel_map[:, :, :-1, :-1]
    return d_dx[:, 0] * d_dy[:, 1] - d_dx[:, 1] * d_dy[:, 0]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    return (values * mask).sum() / (mask.sum() * (values.shape[1] if values.ndim == 4 else 1) + 1e-6)


def mind_descriptor(image: torch.Tensor, patch_size: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Compact 2-D MIND-like descriptor invariant to intensity offset and scale."""
    gray = image.mean(dim=1, keepdim=True)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    neighbours = torch.cat(
        (
            padded[:, :, 1:-1, :-2],
            padded[:, :, 1:-1, 2:],
            padded[:, :, :-2, 1:-1],
            padded[:, :, 2:, 1:-1],
        ),
        dim=1,
    )
    distance = F.avg_pool2d((gray - neighbours).square(), patch_size, stride=1, padding=patch_size // 2)
    descriptor = torch.exp(-distance / (distance.mean(dim=1, keepdim=True) + eps))
    return descriptor / descriptor.amax(dim=1, keepdim=True).clamp_min(eps)


def mind_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    atlas_to_affine: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    warped_descriptor = sample_at_pixel_map(
        mind_descriptor(moving), atlas_to_affine, padding_mode="zeros"
    )
    return _masked_mean((mind_descriptor(fixed) - warped_descriptor).abs(), mask)


def inverse_consistency_loss(
    atlas_to_affine: torch.Tensor,
    affine_to_atlas: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    identity = pixel_identity_grid(
        atlas_to_affine.shape[0],
        atlas_to_affine.shape[-2],
        atlas_to_affine.shape[-1],
        device=atlas_to_affine.device,
        dtype=atlas_to_affine.dtype,
    )
    atlas_cycle = compose_pixel_maps(atlas_to_affine, affine_to_atlas)
    affine_cycle = compose_pixel_maps(affine_to_atlas, atlas_to_affine)
    return 0.5 * (
        _masked_mean((atlas_cycle - identity).square(), mask)
        + _masked_mean((affine_cycle - identity).square(), mask)
    )


def smoothness_loss(velocity: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    dx = velocity[:, :, :, 1:] - velocity[:, :, :, :-1]
    dy = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
    mask_x = None if mask is None else mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = None if mask is None else mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return 0.5 * (_masked_mean(dx.square(), mask_x) + _masked_mean(dy.square(), mask_y))


def topology_loss(pixel_map: torch.Tensor, minimum_jacobian: float = 0.0) -> torch.Tensor:
    return F.relu(minimum_jacobian - jacobian_determinant(pixel_map)).square().mean()


def synthetic_flow_loss(
    predicted_map: torch.Tensor,
    known_map: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    error = F.smooth_l1_loss(predicted_map, known_map, reduction="none", beta=0.25)
    return _masked_mean(error, mask)


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class DiffeomorphicRegistrationUNet(nn.Module):
    """Small residual U-Net for fixed 320x464 atlas/affine-slice pairs."""

    def __init__(
        self,
        structural_channels: int = 1,
        base_channels: int = 16,
        max_velocity_px: float = 6.0,
        integration_steps: int = 7,
    ):
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.max_velocity_px = max_velocity_px
        self.integration_steps = integration_steps
        self.encoder_0 = _ConvBlock(structural_channels * 2 + 2, channels[0])
        self.encoder_1 = _ConvBlock(channels[0], channels[1])
        self.encoder_2 = _ConvBlock(channels[1], channels[2])
        self.bottleneck = _ConvBlock(channels[2], channels[3])
        self.decoder_2 = _ConvBlock(channels[3] + channels[2], channels[2])
        self.decoder_1 = _ConvBlock(channels[2] + channels[1], channels[1])
        self.decoder_0 = _ConvBlock(channels[1] + channels[0], channels[0])
        self.velocity_head = nn.Conv2d(channels[0], 2, 3, padding=1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @staticmethod
    def _down(image: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(image, 2)

    @staticmethod
    def _up(image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=True)

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_0 = self.encoder_0(torch.cat((fixed, moving, fixed_mask, moving_mask), dim=1))
        level_1 = self.encoder_1(self._down(level_0))
        level_2 = self.encoder_2(self._down(level_1))
        encoded = self.bottleneck(self._down(level_2))
        decoded_2 = self.decoder_2(torch.cat((self._up(encoded, level_2), level_2), dim=1))
        decoded_1 = self.decoder_1(torch.cat((self._up(decoded_2, level_1), level_1), dim=1))
        decoded_0 = self.decoder_0(torch.cat((self._up(decoded_1, level_0), level_0), dim=1))

        support = torch.maximum(fixed_mask, moving_mask)
        velocity = torch.tanh(self.velocity_head(decoded_0)) * support * self.max_velocity_px
        velocity = remove_global_affine(velocity)
        peak = velocity.abs().flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
        velocity = velocity * torch.clamp(self.max_velocity_px / (peak + 1e-6), max=1.0)
        atlas_to_affine = integrate_stationary_velocity(velocity, self.integration_steps)
        affine_to_atlas = integrate_stationary_velocity(-velocity, self.integration_steps)
        return atlas_to_affine, affine_to_atlas, velocity
