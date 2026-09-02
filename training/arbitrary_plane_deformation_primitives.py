"""Fresh pure-Torch primitives for affine-free 2-D slice deformation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


AFFINE_FREE_DEFORMATION_TENSOR_KEYS = (
    "raw_velocity_fraction_yx_lowres",
    "support_logits_lowres",
    "raw_velocity_fraction_yx",
    "raw_velocity_yx_px",
    "support_logits",
    "support_probability",
    "projection_support_weight",
    "stationary_velocity_yx_px",
    "removed_affine_coefficients_yx",
    "postprojection_affine_coefficients_yx",
    "velocity_gradient_rescale",
    "prelimit_maximum_velocity_gradient",
    "forward_map_yx_px",
    "pullback_map_yx_px",
    "inverse_map_yx_px",
    "forward_jacobian_determinant",
    "inverse_jacobian_determinant",
    "forward_then_inverse_map_yx",
    "inverse_then_forward_map_yx",
    "forward_then_inverse_error_yx",
    "inverse_then_forward_error_yx",
    "forward_then_inverse_valid_mask",
    "inverse_then_forward_valid_mask",
)


def identity_pixel_map_yx(
    batch_size: int,
    shape_h_w: tuple[int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a ``B,2,H,W`` identity sampling map in y-x pixel order."""
    height, width = shape_h_w
    if batch_size < 1 or height < 2 or width < 2:
        raise ValueError("pixel maps require positive batches and spatial sizes >= 2")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((y, x), dim=0)[None].expand(batch_size, -1, -1, -1)


def inactive_affine_free_deformation(
    context: torch.Tensor,
    output_shape_h_w: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Return the exact identity member of the affine-free deformation family."""
    count, _, low_height, low_width = context.shape
    height, width = output_shape_h_w
    identity = identity_pixel_map_yx(
        count,
        output_shape_h_w,
        device=context.device,
        dtype=context.dtype,
    )
    zero_vector = context.new_zeros(count, 2, height, width)
    zero_scalar = context.new_zeros(count, 1, height, width)
    zero_vector_low = context.new_zeros(count, 2, low_height, low_width)
    zero_scalar_low = context.new_zeros(count, 1, low_height, low_width)
    zero_coefficients = context.new_zeros(count, 2, 3)
    one_scalar = context.new_ones(count, 1, height, width)
    one_summary = context.new_ones(count, 1, 1, 1)
    zero_summary = context.new_zeros(count, 1, 1, 1)
    valid = torch.ones(
        count, 1, height, width, device=context.device, dtype=torch.bool
    )
    return {
        "raw_velocity_fraction_yx_lowres": zero_vector_low,
        "support_logits_lowres": zero_scalar_low,
        "raw_velocity_fraction_yx": zero_vector,
        "raw_velocity_yx_px": zero_vector,
        "support_logits": zero_scalar,
        "support_probability": zero_scalar,
        "projection_support_weight": zero_scalar,
        "stationary_velocity_yx_px": zero_vector,
        "removed_affine_coefficients_yx": zero_coefficients,
        "postprojection_affine_coefficients_yx": zero_coefficients,
        "velocity_gradient_rescale": one_summary,
        "prelimit_maximum_velocity_gradient": zero_summary,
        "forward_map_yx_px": identity,
        "pullback_map_yx_px": identity,
        "inverse_map_yx_px": identity,
        "forward_jacobian_determinant": one_scalar,
        "inverse_jacobian_determinant": one_scalar,
        "forward_then_inverse_map_yx": identity,
        "inverse_then_forward_map_yx": identity,
        "forward_then_inverse_error_yx": zero_vector,
        "inverse_then_forward_error_yx": zero_vector,
        "forward_then_inverse_valid_mask": valid,
        "inverse_then_forward_valid_mask": valid,
    }


def pixel_map_yx_to_normalized_grid_xy(
    pixel_map_yx: torch.Tensor,
    source_shape_h_w: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert a y-x pixel sampling map to a grid-sample x-y grid."""
    if pixel_map_yx.ndim != 4 or pixel_map_yx.shape[1] != 2:
        raise ValueError("pixel maps must have shape (B,2,H,W)")
    source_height, source_width = (
        pixel_map_yx.shape[-2:] if source_shape_h_w is None else source_shape_h_w
    )
    if source_height < 2 or source_width < 2:
        raise ValueError("source spatial sizes must be >= 2")
    y, x = pixel_map_yx.unbind(dim=1)
    return torch.stack(
        (
            2.0 * x / (source_width - 1) - 1.0,
            2.0 * y / (source_height - 1) - 1.0,
        ),
        dim=-1,
    )


def sampling_map_in_bounds_yx(
    pixel_map_yx: torch.Tensor,
    source_shape_h_w: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return a ``B,1,H,W`` inclusive pixel-centre-domain mask."""
    if pixel_map_yx.ndim != 4 or pixel_map_yx.shape[1] != 2:
        raise ValueError("pixel maps must have shape (B,2,H,W)")
    source_height, source_width = (
        pixel_map_yx.shape[-2:] if source_shape_h_w is None else source_shape_h_w
    )
    y, x = pixel_map_yx.unbind(dim=1)
    return (
        torch.isfinite(pixel_map_yx).all(dim=1)
        & (y >= 0.0)
        & (y <= source_height - 1)
        & (x >= 0.0)
        & (x <= source_width - 1)
    )[:, None]


def warp_tensor_with_map_yx(
    source: torch.Tensor,
    pixel_map_yx: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Sample ``source`` at a y-x pixel map; image exterior is zero by default."""
    if source.ndim != 4 or pixel_map_yx.ndim != 4 or pixel_map_yx.shape[1] != 2:
        raise ValueError("source and map must have shapes (B,C,H,W) and (B,2,h,w)")
    if source.shape[0] != pixel_map_yx.shape[0]:
        raise ValueError("source and map batch sizes must match")
    return F.grid_sample(
        source,
        pixel_map_yx_to_normalized_grid_xy(pixel_map_yx, source.shape[-2:]),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def compose_sampling_maps_yx(
    outer_map_yx: torch.Tensor,
    inner_map_yx: torch.Tensor,
) -> torch.Tensor:
    """Return ``outer(inner(.))`` by sampling the outer displacement field."""
    if outer_map_yx.shape != inner_map_yx.shape:
        raise ValueError("sampling maps must share shape")
    identity = identity_pixel_map_yx(
        outer_map_yx.shape[0],
        outer_map_yx.shape[-2:],
        device=outer_map_yx.device,
        dtype=outer_map_yx.dtype,
    )
    outer_displacement = outer_map_yx - identity
    return inner_map_yx + warp_tensor_with_map_yx(
        outer_displacement,
        inner_map_yx,
        padding_mode="border",
    )


def scaling_and_squaring_yx(
    stationary_velocity_yx: torch.Tensor,
    steps: int = 7,
) -> torch.Tensor:
    """Exponentiate a stationary y-x pixel velocity into a sampling map."""
    if stationary_velocity_yx.ndim != 4 or stationary_velocity_yx.shape[1] != 2:
        raise ValueError("stationary velocity must have shape (B,2,H,W)")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("scaling-and-squaring steps must be a nonnegative integer")
    identity = identity_pixel_map_yx(
        stationary_velocity_yx.shape[0],
        stationary_velocity_yx.shape[-2:],
        device=stationary_velocity_yx.device,
        dtype=stationary_velocity_yx.dtype,
    )
    displacement = stationary_velocity_yx / float(2**steps)
    for _ in range(steps):
        displacement = displacement + warp_tensor_with_map_yx(
            displacement,
            identity + displacement,
            padding_mode="border",
        )
    return identity + displacement


def integrate_stationary_velocity_yx(
    stationary_velocity_yx: torch.Tensor,
    steps: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paired ``exp(v)`` and ``exp(-v)`` y-x sampling maps."""
    return (
        scaling_and_squaring_yx(stationary_velocity_yx, steps),
        scaling_and_squaring_yx(-stationary_velocity_yx, steps),
    )


def _finite_difference(value: torch.Tensor, dimension: int) -> torch.Tensor:
    length = value.shape[dimension]
    if length < 2:
        raise ValueError("finite differences require spatial sizes >= 2")
    indices = [slice(None)] * value.ndim
    first, second = indices.copy(), indices.copy()
    first[dimension], second[dimension] = slice(0, 1), slice(1, 2)
    last, before_last = indices.copy(), indices.copy()
    last[dimension], before_last[dimension] = slice(-1, None), slice(-2, -1)
    leading = value[tuple(second)] - value[tuple(first)]
    trailing = value[tuple(last)] - value[tuple(before_last)]
    if length == 2:
        return torch.cat((leading, trailing), dim=dimension)
    left, right = indices.copy(), indices.copy()
    left[dimension], right[dimension] = slice(0, -2), slice(2, None)
    middle = 0.5 * (value[tuple(right)] - value[tuple(left)])
    return torch.cat((leading, middle, trailing), dim=dimension)


def jacobian_determinant_yx(pixel_map_yx: torch.Tensor) -> torch.Tensor:
    """Return ``B,1,H,W`` determinants for a y-x sampling map."""
    if pixel_map_yx.ndim != 4 or pixel_map_yx.shape[1] != 2:
        raise ValueError("pixel maps must have shape (B,2,H,W)")
    derivative_y = _finite_difference(pixel_map_yx, -2)
    derivative_x = _finite_difference(pixel_map_yx, -1)
    determinant = (
        derivative_y[:, 0] * derivative_x[:, 1]
        - derivative_x[:, 0] * derivative_y[:, 1]
    )
    return determinant[:, None]


def inverse_consistency_yx(
    forward_map_yx: torch.Tensor,
    inverse_map_yx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return both cycle maps, errors, and strict in-bounds validity masks."""
    if forward_map_yx.shape != inverse_map_yx.shape:
        raise ValueError("forward and inverse maps must share shape")
    identity = identity_pixel_map_yx(
        forward_map_yx.shape[0],
        forward_map_yx.shape[-2:],
        device=forward_map_yx.device,
        dtype=forward_map_yx.dtype,
    )
    forward_then_inverse = compose_sampling_maps_yx(inverse_map_yx, forward_map_yx)
    inverse_then_forward = compose_sampling_maps_yx(forward_map_yx, inverse_map_yx)
    forward_valid = sampling_map_in_bounds_yx(forward_map_yx)
    inverse_valid = sampling_map_in_bounds_yx(inverse_map_yx)
    forward_then_inverse_valid = forward_valid & (
        warp_tensor_with_map_yx(
            inverse_valid.to(forward_map_yx.dtype),
            forward_map_yx,
            mode="nearest",
            padding_mode="zeros",
        )
        > 0.5
    )
    inverse_then_forward_valid = inverse_valid & (
        warp_tensor_with_map_yx(
            forward_valid.to(inverse_map_yx.dtype),
            inverse_map_yx,
            mode="nearest",
            padding_mode="zeros",
        )
        > 0.5
    )
    return {
        "forward_then_inverse_map_yx": forward_then_inverse,
        "inverse_then_forward_map_yx": inverse_then_forward,
        "forward_then_inverse_error_yx": forward_then_inverse - identity,
        "inverse_then_forward_error_yx": inverse_then_forward - identity,
        "forward_then_inverse_valid_mask": forward_then_inverse_valid,
        "inverse_then_forward_valid_mask": inverse_then_forward_valid,
    }


def limit_velocity_gradient_yx(
    velocity_yx: torch.Tensor,
    maximum_gradient: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Globally rescale each affine-free SVF to a strict spatial-gradient bound."""
    derivative_y = _finite_difference(velocity_yx, -2)
    derivative_x = _finite_difference(velocity_yx, -1)
    frobenius = torch.sqrt(
        derivative_y.square().sum(dim=1, keepdim=True)
        + derivative_x.square().sum(dim=1, keepdim=True)
    )
    observed = frobenius.amax(dim=(-2, -1), keepdim=True)
    scale = (float(maximum_gradient) / observed.clamp_min(1e-12)).clamp(max=1.0)
    return velocity_yx * scale, scale, observed


def support_weighted_affine_projection_yx(
    velocity_yx: torch.Tensor,
    support_weight: torch.Tensor,
    *,
    support_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove weighted translation plus unconstrained 2-by-2 linear velocity."""
    if velocity_yx.ndim != 4 or velocity_yx.shape[1] != 2:
        raise ValueError("velocity must have shape (B,2,H,W)")
    if support_weight.ndim == 3:
        support_weight = support_weight[:, None]
    if support_weight.shape != velocity_yx.shape[:1] + (1,) + velocity_yx.shape[-2:]:
        raise ValueError("support weight must have shape (B,1,H,W)")
    if support_floor <= 0.0:
        raise ValueError("support floor must be positive")
    if not bool(torch.isfinite(velocity_yx).all()) or not bool(
        torch.isfinite(support_weight).all()
    ) or bool((support_weight < 0.0).any()):
        raise ValueError("velocity and nonnegative support weights must be finite")

    input_dtype = velocity_yx.dtype
    work_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )
    with torch.autocast(device_type=velocity_yx.device.type, enabled=False):
        velocity = velocity_yx.to(work_dtype)
        weight = support_weight.detach().to(work_dtype).clamp_min(support_floor)
        weight = weight / weight.sum(dim=(-2, -1), keepdim=True)
        height, width = velocity.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=velocity.device, dtype=work_dtype),
            torch.linspace(-1.0, 1.0, width, device=velocity.device, dtype=work_dtype),
            indexing="ij",
        )
        basis = torch.stack((torch.ones_like(y), y, x), dim=0)[None]
        gram = torch.einsum("bphw,bqhw->bpq", weight * basis, basis)
        right = torch.einsum("bchw,bphw->bcp", weight * velocity, basis)
        coefficients = torch.linalg.solve(
            gram, right.transpose(-2, -1)
        ).transpose(-2, -1)
        fitted = torch.einsum("bcp,bphw->bchw", coefficients, basis)
        residual = velocity - fitted
        post_right = torch.einsum("bchw,bphw->bcp", weight * residual, basis)
        post_coefficients = torch.linalg.solve(
            gram, post_right.transpose(-2, -1)
        ).transpose(-2, -1)
    return (
        residual.to(input_dtype),
        coefficients.to(input_dtype),
        post_coefficients.to(input_dtype),
        weight.to(input_dtype),
    )


class AffineFreeSVFDecoder(nn.Module):
    """Decode one bounded affine-free y-x SVF from each canonical cell context."""

    def __init__(
        self,
        hidden_channels: int,
        max_velocity_fraction_yx: tuple[float, float] = (0.08, 0.08),
        integration_steps: int = 7,
        support_floor: float = 1e-4,
        maximum_velocity_gradient: float = 0.35,
    ):
        super().__init__()
        if hidden_channels < 1 or len(max_velocity_fraction_yx) != 2 or any(
            value <= 0.0 for value in max_velocity_fraction_yx
        ):
            raise ValueError("decoder channels and velocity limits must be positive")
        if (
            integration_steps < 0
            or support_floor <= 0.0
            or not 0.0 < maximum_velocity_gradient < 1.0
        ):
            raise ValueError("integration steps and support floor are invalid")
        self.integration_steps = int(integration_steps)
        self.support_floor = float(support_floor)
        self.maximum_velocity_gradient = float(maximum_velocity_gradient)
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.velocity_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        self.support_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.register_buffer(
            "max_velocity_fraction_yx",
            torch.tensor(max_velocity_fraction_yx, dtype=torch.float32),
        )
        nn.init.normal_(self.velocity_head.weight, std=1e-4)
        nn.init.zeros_(self.velocity_head.bias)
        nn.init.normal_(self.support_head.weight, std=1e-3)
        nn.init.zeros_(self.support_head.bias)

    def forward(
        self,
        canonical_cell_context: torch.Tensor,
        output_shape_h_w: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if canonical_cell_context.ndim != 4:
            raise ValueError("canonical cell context must have shape (N,C,h,w)")
        height, width = output_shape_h_w
        if height < 2 or width < 2:
            raise ValueError("deformation output spatial sizes must be >= 2")
        features = self.trunk(canonical_cell_context)
        raw_fraction_low = torch.tanh(self.velocity_head(features))
        raw_fraction_low = raw_fraction_low * self.max_velocity_fraction_yx.to(
            raw_fraction_low
        )[None, :, None, None]
        support_logits_low = self.support_head(features)
        raw_fraction = F.interpolate(
            raw_fraction_low,
            output_shape_h_w,
            mode="bilinear",
            align_corners=True,
        )
        support_logits = F.interpolate(
            support_logits_low,
            output_shape_h_w,
            mode="bilinear",
            align_corners=True,
        )
        pixel_scale = raw_fraction.new_tensor((height - 1, width - 1))[
            None, :, None, None
        ]
        raw_velocity = raw_fraction * pixel_scale
        support_probability = torch.sigmoid(support_logits)
        gauge_weight = torch.ones_like(support_probability)
        velocity, removed, post, projection_weight = (
            support_weighted_affine_projection_yx(
                raw_velocity,
                gauge_weight,
                support_floor=self.support_floor,
            )
        )
        velocity, gradient_scale, prelimit_maximum_gradient = limit_velocity_gradient_yx(
            velocity, self.maximum_velocity_gradient
        )
        forward, inverse = integrate_stationary_velocity_yx(
            velocity, self.integration_steps
        )
        cycles = inverse_consistency_yx(forward, inverse)
        return {
            "raw_velocity_fraction_yx_lowres": raw_fraction_low,
            "support_logits_lowres": support_logits_low,
            "raw_velocity_fraction_yx": raw_fraction,
            "raw_velocity_yx_px": raw_velocity,
            "support_logits": support_logits,
            "support_probability": support_probability,
            "projection_support_weight": projection_weight,
            "affine_projection_gauge": "uniform_canvas",
            "stationary_velocity_yx_px": velocity,
            "removed_affine_coefficients_yx": removed,
            "postprojection_affine_coefficients_yx": post,
            "velocity_gradient_rescale": gradient_scale,
            "prelimit_maximum_velocity_gradient": prelimit_maximum_gradient,
            "forward_map_yx_px": forward,
            "pullback_map_yx_px": forward,
            "inverse_map_yx_px": inverse,
            "forward_jacobian_determinant": jacobian_determinant_yx(forward),
            "inverse_jacobian_determinant": jacobian_determinant_yx(inverse),
            **cycles,
        }
