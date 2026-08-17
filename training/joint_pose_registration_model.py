from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from training.atlas_pose_models_v7 import (
    AP_MAX_UM,
    AP_MIN_UM,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    AtlasPoseV7,
)
from training.dense_registration_model import (
    DenseRegistrationModel,
    identity_pixel_map,
    warp_tensor,
)


POSE_CENTER = (-2000.0, 0.0, 0.0)
POSE_SCALE = (2500.0, 20.0, 20.0)
DEFAULT_MAXIMUM_POSE_DELTA = (500.0, 5.0, 5.0)
POSE_MINIMUM = (AP_MIN_UM, TILT_MIN_DEG, TILT_MIN_DEG)
POSE_MAXIMUM = (AP_MAX_UM, TILT_MAX_DEG, TILT_MAX_DEG)


def project_pose_to_domain(pose: torch.Tensor) -> torch.Tensor:
    """Differentiably project AP/L-R/D-V onto the canonical AtlasPose domain."""
    minimum = pose.new_tensor(POSE_MINIMUM)
    maximum = pose.new_tensor(POSE_MAXIMUM)
    return torch.maximum(torch.minimum(pose, maximum), minimum)


def _apply_homography(pixel_map: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((pixel_map, torch.ones_like(pixel_map[:, :1])), dim=1)
    mapped = torch.einsum("bij,bjhw->bihw", homography, homogeneous)
    denominator = mapped[:, 2:3]
    safe = torch.where(
        denominator.abs() >= 1e-8,
        denominator,
        torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
        * 1e-8,
    )
    return mapped[:, :2] / safe


def compose_aligned_maps_to_source_model(
    fixed_to_aligned_moving_map: torch.Tensor,
    aligned_moving_to_fixed_map: torch.Tensor,
    source_to_aligned_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose aligned maps onto the canonical pre-refiner source model canvas."""
    aligned_to_source_h = torch.linalg.inv(source_to_aligned_h)
    fixed_to_source = _apply_homography(
        fixed_to_aligned_moving_map, aligned_to_source_h
    )
    source_grid = identity_pixel_map(
        fixed_to_aligned_moving_map.shape[0],
        fixed_to_aligned_moving_map.shape[-2],
        fixed_to_aligned_moving_map.shape[-1],
        device=fixed_to_aligned_moving_map.device,
        dtype=fixed_to_aligned_moving_map.dtype,
    )
    source_to_aligned_map = _apply_homography(source_grid, source_to_aligned_h)
    source_to_fixed = warp_tensor(
        aligned_moving_to_fixed_map,
        source_to_aligned_map,
        padding_mode="border",
    )
    return fixed_to_source, source_to_fixed


def _group_count(channel_count: int) -> int:
    for group_count in (8, 4, 2):
        if channel_count % group_count == 0:
            return group_count
    return 1


def _validated_alignment_receipt(
    receipt: object,
    pose: torch.Tensor,
    source_shape: tuple[int, int],
) -> tuple[dict, torch.Tensor]:
    if not isinstance(receipt, dict):
        raise ValueError("prepare_pair must return an alignment receipt dictionary")
    if "map_pose" not in receipt or "source_to_aligned_h" not in receipt:
        raise ValueError("alignment receipt requires map_pose and source_to_aligned_h")
    declared_shape = tuple(int(value) for value in receipt.get("source_shape", ()))
    if declared_shape != tuple(source_shape):
        raise ValueError("alignment receipt source_shape does not match the model canvas")
    receipt_pose = torch.as_tensor(
        receipt["map_pose"], device=pose.device, dtype=pose.dtype
    )
    if receipt_pose.shape != pose.shape or not torch.allclose(
        receipt_pose, pose, rtol=0.0, atol=1e-5
    ):
        raise ValueError("alignment receipt map_pose is stale or mismatched")
    homography = torch.as_tensor(
        receipt["source_to_aligned_h"], device=pose.device, dtype=pose.dtype
    )
    if homography.ndim == 2:
        homography = homography.unsqueeze(0)
    if homography.shape != (pose.shape[0], 3, 3):
        raise ValueError("source_to_aligned_h must have shape [batch, 3, 3]")
    if not bool(torch.isfinite(homography).all()):
        raise ValueError("source_to_aligned_h must be finite")
    if bool((torch.linalg.det(homography).abs() < 1e-8).any()):
        raise ValueError("source_to_aligned_h must be invertible")
    bound = dict(receipt)
    bound.update(
        map_pose=pose,
        source_to_aligned_h=homography,
        source_shape=tuple(source_shape),
        map_space="source-model-canvas",
        aligned_map_space="candidate-aligned-moving",
        composition="candidate-aligned dense maps composed through source_to_aligned_h",
    )
    return bound, homography


class _ReviewEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: tuple[int, ...]):
        super().__init__()
        blocks = []
        previous_channels = input_channels
        for channel_count in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        previous_channels,
                        channel_count,
                        3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(channel_count), channel_count),
                    nn.GELU(),
                    nn.Conv2d(
                        channel_count,
                        channel_count,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(channel_count), channel_count),
                    nn.GELU(),
                )
            )
            previous_channels = channel_count
        self.blocks = nn.ModuleList(blocks)
        self.output_features = sum(channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = []
        for block in self.blocks:
            image = block(image)
            pooled.append(image.mean(dim=(-2, -1)))
        return torch.cat(pooled, dim=1)


class PoseReviewHead(nn.Module):
    """Shared recurrent head that scores and corrects one rendered atlas candidate."""

    def __init__(
        self,
        pose_feature_count: int = 512,
        registration_channels: int = 2,
        channels: tuple[int, ...] = (16, 24, 32),
        hidden_features: int = 256,
        maximum_pose_delta: tuple[float, float, float] = DEFAULT_MAXIMUM_POSE_DELTA,
    ):
        super().__init__()
        self.encoder = _ReviewEncoder(registration_channels * 3, channels)
        context_features = self.encoder.output_features + pose_feature_count + 3 + 4 + 4
        self.context = nn.Sequential(
            nn.LayerNorm(context_features),
            nn.Linear(context_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
        )
        self.pose_delta_head = nn.Linear(hidden_features, 3)
        self.compatibility_head = nn.Linear(hidden_features, 1)
        self.register_buffer("pose_center", torch.tensor(POSE_CENTER))
        self.register_buffer("pose_scale", torch.tensor(POSE_SCALE))
        self.register_buffer("maximum_pose_delta", torch.tensor(maximum_pose_delta))
        nn.init.zeros_(self.pose_delta_head.weight)
        nn.init.zeros_(self.pose_delta_head.bias)

    def forward(
        self,
        fixed_atlas: torch.Tensor,
        warped_moving_slice: torch.Tensor,
        current_pose: torch.Tensor,
        pose_features: torch.Tensor,
        similarity_parameters: torch.Tensor,
        local_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paired_images = torch.cat(
            (
                fixed_atlas,
                warped_moving_slice,
                torch.abs(fixed_atlas - warped_moving_slice),
            ),
            dim=1,
        )
        velocity_summary = torch.cat(
            (
                local_velocity.mean(dim=(-2, -1)),
                local_velocity.abs().mean(dim=(-2, -1)),
            ),
            dim=1,
        )
        normalized_pose = (current_pose - self.pose_center) / self.pose_scale
        context = self.context(
            torch.cat(
                (
                    self.encoder(paired_images),
                    pose_features,
                    normalized_pose,
                    similarity_parameters,
                    velocity_summary,
                ),
                dim=1,
            )
        )
        pose_delta = torch.tanh(self.pose_delta_head(context)) * self.maximum_pose_delta
        compatibility_logit = self.compatibility_head(context).squeeze(1)
        return pose_delta, compatibility_logit


class JointPoseRegistrationModel(nn.Module):
    """One checkpoint coupling AtlasPose, dense registration, and recurrent review.

    The atlas and outline-affine preprocessing remain outside this module.
    ``rollout`` requests a newly rendered atlas and candidate-normalized moving
    canvas after every pose update, then requests and scores the pair once more at
    the final pose. It retains the aligned maps for audit and composes released
    maps through the required host affine into the canonical pre-refiner
    ``source-model-canvas``. The GUI must separately compose its original-display
    coordinates into that 320x464-style model canvas.
    """

    def __init__(
        self,
        pose_initializer: AtlasPoseV7 | None = None,
        registrar: DenseRegistrationModel | None = None,
        pose_feature_count: int = 512,
        registration_channels: int = 2,
        review_channels: tuple[int, ...] = (16, 24, 32),
        review_hidden_features: int = 256,
        maximum_pose_delta: tuple[float, float, float] = DEFAULT_MAXIMUM_POSE_DELTA,
    ):
        super().__init__()
        self.pose_initializer = (
            pose_initializer if pose_initializer is not None else AtlasPoseV7(pretrained=False)
        )
        self.registrar = (
            registrar
            if registrar is not None
            else DenseRegistrationModel(input_channels=registration_channels)
        )
        self.review_head = PoseReviewHead(
            pose_feature_count=pose_feature_count,
            registration_channels=registration_channels,
            channels=review_channels,
            hidden_features=review_hidden_features,
            maximum_pose_delta=maximum_pose_delta,
        )

    def initialize(self, pose_image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = dict(
            self.pose_initializer.training_outputs(
                pose_image,
                include_anatomy=False,
            )
        )
        outputs["pose"] = project_pose_to_domain(outputs["pose"])
        outputs["pose_features"] = outputs["pooled_features"]
        return outputs

    def refine_once(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
        current_pose: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        registration = self.registrar.forward_with_details(fixed_atlas, moving_slice)
        warped_moving_slice = warp_tensor(
            moving_slice,
            registration["fixed_to_moving_map"],
            padding_mode="border",
        )
        pose_delta, compatibility_logit = self.review_head(
            fixed_atlas,
            warped_moving_slice,
            current_pose,
            pose_features,
            registration["similarity_parameters"],
            registration["local_velocity"],
        )
        return {
            "pose": project_pose_to_domain(current_pose + pose_delta),
            "map_pose": current_pose,
            "pose_delta": pose_delta,
            "compatibility_logit": compatibility_logit,
            "warped_moving_slice": warped_moving_slice,
            **registration,
        }

    def register_final_pose(
        self,
        fixed_atlas_at_final_pose: torch.Tensor,
        moving_slice: torch.Tensor,
        final_pose: torch.Tensor,
        pose_features: torch.Tensor,
        map_domain_receipt: dict,
    ) -> dict[str, object]:
        """Score the final candidate and bind maps to the source model canvas."""
        review = self.refine_once(
            fixed_atlas_at_final_pose,
            moving_slice,
            final_pose,
            pose_features,
        )
        receipt, source_to_aligned_h = _validated_alignment_receipt(
            map_domain_receipt, final_pose, review["fixed_to_moving_map"].shape[-2:]
        )
        fixed_to_source, source_to_fixed = compose_aligned_maps_to_source_model(
            review["fixed_to_moving_map"],
            review["moving_to_fixed_map"],
            source_to_aligned_h,
        )
        return {
            "pose": final_pose,
            "map_pose": final_pose,
            "suggested_next_pose": review["pose"],
            "pose_delta": review["pose_delta"],
            "final_compatibility_logit": review["compatibility_logit"],
            "fixed_atlas": fixed_atlas_at_final_pose,
            "moving_slice": moving_slice,
            "warped_moving_slice": review["warped_moving_slice"],
            "fixed_to_source_model_map": fixed_to_source,
            "source_model_to_fixed_map": source_to_fixed,
            "fixed_to_aligned_moving_map": review["fixed_to_moving_map"],
            "aligned_moving_to_fixed_map": review["moving_to_fixed_map"],
            "map_space": "source-model-canvas",
            "aligned_map_space": "candidate-aligned-moving",
            "map_domain_receipt": receipt,
            "similarity_parameters": review["similarity_parameters"],
            "local_velocity": review["local_velocity"],
            "pyramid_velocities": review["pyramid_velocities"],
        }

    def rollout(
        self,
        pose_image: torch.Tensor,
        prepare_pair: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, dict],
        ],
        refinement_steps: int,
    ) -> dict[str, object]:
        initialization = self.initialize(pose_image)
        pose = initialization["pose"]
        pose_features = initialization["pose_features"]
        steps = []
        for _ in range(refinement_steps):
            prepared = prepare_pair(pose)
            if not isinstance(prepared, tuple) or len(prepared) != 3:
                raise ValueError("prepare_pair must return fixed, aligned moving, and receipt")
            fixed_atlas, moving_slice, receipt = prepared
            step = self.refine_once(
                fixed_atlas,
                moving_slice,
                pose,
                pose_features,
            )
            bound_receipt, source_to_aligned_h = _validated_alignment_receipt(
                receipt,
                step["map_pose"],
                step["fixed_to_moving_map"].shape[-2:],
            )
            step["fixed_to_aligned_moving_map"] = step["fixed_to_moving_map"]
            step["aligned_moving_to_fixed_map"] = step["moving_to_fixed_map"]
            step["fixed_to_source_model_map"], step["source_model_to_fixed_map"] = (
                compose_aligned_maps_to_source_model(
                    step["fixed_to_aligned_moving_map"],
                    step["aligned_moving_to_fixed_map"],
                    source_to_aligned_h,
                )
            )
            del step["fixed_to_moving_map"], step["moving_to_fixed_map"]
            step["map_space"] = "source-model-canvas"
            step["aligned_map_space"] = "candidate-aligned-moving"
            step["map_domain_receipt"] = bound_receipt
            steps.append(step)
            pose = step["pose"]

        prepared = prepare_pair(pose)
        if not isinstance(prepared, tuple) or len(prepared) != 3:
            raise ValueError("prepare_pair must return fixed, aligned moving, and receipt")
        fixed_atlas_at_final_pose, moving_slice_at_final_pose, receipt = prepared
        final_registration = self.register_final_pose(
            fixed_atlas_at_final_pose,
            moving_slice_at_final_pose,
            pose,
            pose_features,
            receipt,
        )
        compatibility_logits = (
            torch.stack([step["compatibility_logit"] for step in steps], dim=1)
            if steps
            else pose.new_empty((pose.shape[0], 0))
        )
        return {
            **final_registration,
            "initial_pose": initialization["pose"],
            "orientation_inverted_logit": initialization["orientation_inverted_logit"],
            "pose_features": pose_features,
            "compatibility_logits": compatibility_logits,
            "steps": tuple(steps),
            "initializer_outputs": initialization,
        }

    def set_training_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "joint")
        if stage == "joint":
            return
        if stage not in {"review", "geometry"}:
            raise ValueError(f"Unknown joint-training stage: {stage}")
        for parameter in self.review_head.parameters():
            parameter.requires_grad_(True)
        if stage == "geometry":
            for module_name in ("feature_head", "pose_head", "orientation_head"):
                for parameter in getattr(self.pose_initializer, module_name).parameters():
                    parameter.requires_grad_(True)
            for parameter in self.registrar.similarity_head.parameters():
                parameter.requires_grad_(True)
            for modules in (
                self.registrar.registration_stages,
                self.registrar.velocity_heads,
            ):
                for module in modules[-2:]:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)


class JointInitializerExport(nn.Module):
    def __init__(self, model: JointPoseRegistrationModel):
        super().__init__()
        self.model = model

    def forward(self, pose_image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model.initialize(pose_image)
        return (
            outputs["pose"],
            outputs["orientation_inverted_logit"],
            outputs["pose_features"],
        )


class JointRefinerExport(nn.Module):
    """Score/register ``current_pose`` and propose, but do not render, the next pose.

    The returned maps and compatibility logit describe the supplied current pose;
    the first output is the proposed next pose that the host may render afterward.
    """

    def __init__(self, model: JointPoseRegistrationModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        fixed_atlas: torch.Tensor,
        moving_slice: torch.Tensor,
        current_pose: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model.refine_once(
            fixed_atlas,
            moving_slice,
            current_pose,
            pose_features,
        )
        return (
            outputs["pose"],
            outputs["pose_delta"],
            outputs["compatibility_logit"],
            outputs["fixed_to_moving_map"],
            outputs["moving_to_fixed_map"],
            outputs["similarity_parameters"],
            outputs["local_velocity"],
        )
