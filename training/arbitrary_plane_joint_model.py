"""Standalone arbitrary-plane pose model with pose-gated affine-free deformation."""

from __future__ import annotations

import torch
from torch import nn

from training.arbitrary_plane_deformation_primitives import (
    AFFINE_FREE_DEFORMATION_TENSOR_KEYS,
    AffineFreeSVFDecoder,
    identity_pixel_map_yx,
    inactive_affine_free_deformation,
    warp_tensor_with_map_yx,
)
from training.arbitrary_plane_recurrent_model import (
    ArbitraryPlaneRetrievalRefinementModel,
)


DEFORMATION_GATE_POLICY = "fixed_iteration_index_only"
DEFORMATION_UPDATE_SEMANTICS = "absolute_per_iteration_not_accumulated"

_SEQUENCE_KEYS = AFFINE_FREE_DEFORMATION_TENSOR_KEYS


def _inactive_deformation(
    context: torch.Tensor,
    output_shape_h_w: tuple[int, int],
) -> dict[str, torch.Tensor]:
    return inactive_affine_free_deformation(context, output_shape_h_w)


class ArbitraryPlaneJointModel(nn.Module):
    """Retrieve coarse pose modes, then jointly refine pose and deformation."""

    def __init__(
        self,
        atlas_channels: int,
        feature_channels: int = 16,
        hidden_channels: int = 32,
        correlation_radius: int = 2,
        update_limits: tuple[float, ...] = (
            0.18,
            0.18,
            600.0,
            0.18,
            600.0,
            600.0,
            0.12,
            0.12,
            0.12,
        ),
        plane_tangent_scales: tuple[float, float, float] = (0.18, 0.18, 600.0),
        max_velocity_fraction_yx: tuple[float, float] = (0.08, 0.08),
        deformation_integration_steps: int = 7,
        deformation_support_floor: float = 1e-4,
        deformation_maximum_velocity_gradient: float = 0.35,
    ):
        super().__init__()
        self.pose_model = ArbitraryPlaneRetrievalRefinementModel(
            atlas_channels=atlas_channels,
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
            correlation_radius=correlation_radius,
            update_limits=update_limits,
            plane_tangent_scales=plane_tangent_scales,
        )
        self.deformation_decoder = AffineFreeSVFDecoder(
            hidden_channels,
            max_velocity_fraction_yx=max_velocity_fraction_yx,
            integration_steps=deformation_integration_steps,
            support_floor=deformation_support_floor,
            maximum_velocity_gradient=deformation_maximum_velocity_gradient,
        )

    def forward(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        cell_id: torch.Tensor,
        cell_states: torch.Tensor,
        cell_log_mass: torch.Tensor,
        representation_log_weight: torch.Tensor,
        representation_to_canonical_raster_affine: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        expected_catalogue_cell_count: int,
        top_k: int = 4,
        refinement_steps: int = 3,
        pose_only_steps: int = 2,
        retrieval_shape_h_w: tuple[int, int] | None = None,
        catalogue_chunk_size: int | None = None,
        training_truth_catalogue_index: torch.Tensor | None = None,
    ) -> dict[str, object]:
        pose_arguments = (
            image,
            outline,
            outline_available,
            atlas_volume,
            cell_id,
            cell_states,
            cell_log_mass,
            representation_log_weight,
            representation_to_canonical_raster_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            support_origin_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        pose_keywords = dict(
            expected_catalogue_cell_count=expected_catalogue_cell_count,
            top_k=top_k,
            refinement_steps=refinement_steps,
            deformation_decoder=self.deformation_decoder,
            pose_only_steps=pose_only_steps,
        )
        if (retrieval_shape_h_w is None) != (catalogue_chunk_size is None):
            raise ValueError(
                "streamed retrieval shape and catalogue chunk size must be set together"
            )
        if training_truth_catalogue_index is not None and retrieval_shape_h_w is None:
            raise ValueError("truth-forced refinement requires streamed retrieval")
        pose = (
            self.pose_model(*pose_arguments, **pose_keywords)
            if retrieval_shape_h_w is None
            else self.pose_model.forward_streamed(
                *pose_arguments[:9],
                output_shape_h_w,
                retrieval_shape_h_w,
                *pose_arguments[10:],
                catalogue_chunk_size=catalogue_chunk_size,
                training_truth_catalogue_index=training_truth_catalogue_index,
                **pose_keywords,
            )
        )
        sequences = pose.pop("joint_deformation_output_sequences")
        cell_contexts = pose.pop("joint_deformation_cell_context_sequence")
        representation_probability = pose.pop(
            "joint_deformation_representation_probability_sequence"
        )
        active = pose.pop("joint_deformation_active_sequence")
        feedback_render = pose.pop(
            "joint_final_feedback_deformed_canonical_render"
        )
        batch, cells = cell_contexts.shape[:2]
        final_render = pose["final_canonical_render"]
        final_probability = representation_probability[..., -1]
        marginalized_render = (
            final_probability.to(final_render)[..., None, None, None] * final_render
        ).sum(dim=2)
        final_map = sequences["forward_map_yx_px_sequence"][:, :, -1]
        deformed_render = warp_tensor_with_map_yx(
            marginalized_render.reshape(
                batch * cells, *marginalized_render.shape[2:]
            ),
            final_map.to(marginalized_render).reshape(
                batch * cells, *final_map.shape[2:]
            ),
        ).reshape_as(marginalized_render)
        final_outputs = {
            f"final_{key.removesuffix('_sequence')}": value[:, :, -1]
            for key, value in sequences.items()
        }
        return {
            "pose": pose,
            "deformation_representation_probability_sequence": representation_probability,
            "deformation_cell_context_sequence": cell_contexts,
            "deformation_active_sequence": active,
            "deformation_gating_audit": {
                "pose_only_steps": int(pose_only_steps),
                "gate_policy": DEFORMATION_GATE_POLICY,
                "update_semantics": DEFORMATION_UPDATE_SEMANTICS,
                "representation_probabilities_detached": True,
                "feedback_semantics": "absolute_deformation_warps_next_finite_thickness_render",
                "shared_recurrent_context": True,
            },
            **sequences,
            **final_outputs,
            "final_representation_marginalized_canonical_render": marginalized_render,
            "final_feedback_deformed_canonical_render": (
                final_probability.to(feedback_render)[..., None, None, None]
                * feedback_render
            ).sum(dim=2),
            "final_deformed_canonical_render": deformed_render,
        }
