"""Compact joint refinement after the bound v6 retrieval cascade."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_catalogue_runtime_v6 import (
    BoundCompleteCatalogueBatchV6,
    CompleteCatalogueRuntimeV6,
    verify_bound_complete_catalogue_batch_v6,
)
from training.arbitrary_plane_deformation_primitives import (
    AFFINE_FREE_DEFORMATION_TENSOR_KEYS,
    AffineFreeSVFDecoder,
    warp_tensor_with_map_yx,
)
from training.arbitrary_plane_joint_model import (
    DEFORMATION_GATE_POLICY,
    DEFORMATION_UPDATE_SEMANTICS,
)
from training.arbitrary_plane_recurrent_model import (
    compose_antipodal_plane_frame_residual,
)
from training.arbitrary_plane_recurrent_model_v6 import (
    ArbitraryPlaneRetrievalRefinementModelV6,
)


JOINT_MODEL_V6_SCHEMA = "anatomy-tracker.joint-model/v6"


class ArbitraryPlaneJointModelV6(nn.Module):
    """Refine only honest finite-render-closed modes in a compact batch."""

    def __init__(
        self,
        catalogue_runtime_v6: CompleteCatalogueRuntimeV6,
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
        proposal_channels: int = 16,
        proposal_mixture_components: int = 8,
        proposal_spatial_bins_h_w: tuple[int, int] = (4, 4),
        proposal_offset_scale_um: float = 10000.0,
        cascade_max_rendered_cells_per_sample: int = 64,
        cascade_max_closure_rounds: int = 4,
        pose_only_steps: int = 2,
        max_velocity_fraction_yx: tuple[float, float] = (0.08, 0.08),
        deformation_integration_steps: int = 7,
        deformation_support_floor: float = 1e-4,
        deformation_maximum_velocity_gradient: float = 0.35,
    ):
        super().__init__()
        if (
            not isinstance(pose_only_steps, int)
            or isinstance(pose_only_steps, bool)
            or pose_only_steps < 0
        ):
            raise ValueError("pose-only steps must be one fixed nonnegative integer")
        self.pose_only_steps = pose_only_steps
        self.pose_model = ArbitraryPlaneRetrievalRefinementModelV6(
            catalogue_runtime_v6=catalogue_runtime_v6,
            atlas_channels=atlas_channels,
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
            correlation_radius=correlation_radius,
            update_limits=update_limits,
            plane_tangent_scales=plane_tangent_scales,
            proposal_channels=proposal_channels,
            proposal_mixture_components=proposal_mixture_components,
            proposal_spatial_bins_h_w=proposal_spatial_bins_h_w,
            proposal_offset_scale_um=proposal_offset_scale_um,
            cascade_max_rendered_cells_per_sample=(
                cascade_max_rendered_cells_per_sample
            ),
            cascade_max_closure_rounds=cascade_max_closure_rounds,
        )
        self.deformation_decoder = AffineFreeSVFDecoder(
            hidden_channels,
            max_velocity_fraction_yx=max_velocity_fraction_yx,
            integration_steps=deformation_integration_steps,
            support_floor=deformation_support_floor,
            maximum_velocity_gradient=deformation_maximum_velocity_gradient,
        )

    @staticmethod
    def _row_schedule(
        value: torch.Tensor,
        row: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        value = torch.as_tensor(value)
        if value.ndim == 1:
            return value
        if value.ndim == 2 and value.shape[0] == batch:
            return value.index_select(0, row.reshape(1))
        raise ValueError("axial schedules must have shape (S,) or (B,S)")

    @staticmethod
    def _compact_schedule(
        value: torch.Tensor,
        row: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        value = torch.as_tensor(value)
        if value.ndim == 1:
            return value
        if value.ndim == 2 and value.shape[0] == batch:
            return value.index_select(0, row)
        raise ValueError("axial schedules must have shape (S,) or (B,S)")

    @staticmethod
    def _gather_cells(
        value: torch.Tensor,
        source_row: torch.Tensor,
        catalogue_index: torch.Tensor,
    ) -> torch.Tensor:
        compact = value.index_select(0, source_row)
        index = catalogue_index.reshape(
            *catalogue_index.shape, *([1] * (value.ndim - 2))
        ).expand(*catalogue_index.shape, *value.shape[2:])
        return torch.gather(compact, 1, index)

    def _full_resolution_topk_score(
        self,
        source_features: torch.Tensor,
        atlas_volume: torch.Tensor,
        catalogue: dict[str, torch.Tensor],
        source_row: torch.Tensor,
        catalogue_index: torch.Tensor,
        output_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = catalogue["cell_states"].shape[0]
        states = self._gather_cells(
            catalogue["cell_states"], source_row, catalogue_index
        )
        log_mass = self._gather_cells(
            catalogue["cell_log_mass"], source_row, catalogue_index
        )
        log_weight = self._gather_cells(
            catalogue["representation_log_weight"], source_row, catalogue_index
        )
        affine = self._gather_cells(
            catalogue["representation_to_canonical_raster_affine"],
            source_row,
            catalogue_index,
        )
        chunks = []
        for local_row in range(source_row.numel()):
            original_row = source_row[local_row]
            canonical_id = catalogue["cell_id"][catalogue_index[local_row]]
            chunks.append(
                self.pose_model.score_catalogue_chunk(
                    source_features[local_row : local_row + 1],
                    atlas_volume,
                    canonical_id,
                    states[local_row : local_row + 1],
                    log_mass[local_row : local_row + 1],
                    log_weight[local_row : local_row + 1],
                    affine[local_row : local_row + 1],
                    output_shape_h_w,
                    origin_ap_dv_ml_um,
                    voxel_size_ap_dv_ml_um,
                    catalogue["support_origin_ap_dv_ml_um"],
                    self._row_schedule(axial_offsets_um, original_row, batch),
                    self._row_schedule(axial_weights, original_row, batch),
                )
            )
        keys = tuple(key for key in chunks[0] if key != "cell_id")
        result = {key: torch.cat([chunk[key] for chunk in chunks]) for key in keys}
        result["cell_id"] = torch.stack([chunk["cell_id"] for chunk in chunks])
        expected_id = catalogue["cell_id"][catalogue_index]
        if not torch.equal(result["cell_id"], expected_id):
            raise RuntimeError("full-resolution scoring lost canonical catalogue IDs")
        return result

    @staticmethod
    def _ranker_detached_initialization(
        top_score: dict[str, torch.Tensor],
        cell_states: torch.Tensor,
        support_origin_ap_dv_ml_um: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical_update = top_score["initial_representation_canonical_residual"]
        probability = top_score[
            "representation_log_conditional_within_cell"
        ].detach().exp()
        accumulation_dtype = (
            torch.float32
            if canonical_update.dtype in (torch.float16, torch.bfloat16)
            else canonical_update.dtype
        )
        cell_update = (
            probability.to(accumulation_dtype)[..., None]
            * canonical_update.to(accumulation_dtype)
        ).sum(dim=2).to(cell_states)
        with torch.autocast(device_type=cell_states.device.type, enabled=False):
            initial_state = compose_antipodal_plane_frame_residual(
                cell_states.reshape(-1, cell_states.shape[-1]),
                cell_update.reshape(-1, cell_update.shape[-1]),
                support_origin_ap_dv_ml_um,
            ).reshape_as(cell_states)
            representation_covariance = top_score[
                "initial_representation_canonical_plane_covariance"
            ]
            covariance_dtype = torch.promote_types(
                representation_covariance.dtype, cell_states.dtype
            )
            if covariance_dtype in (torch.float16, torch.bfloat16):
                covariance_dtype = torch.float32
            difference = (
                canonical_update.to(covariance_dtype)[..., :3]
                - cell_update.to(covariance_dtype)[..., None, :3]
            )
            cell_covariance = (
                probability.to(covariance_dtype)[..., None, None]
                * (
                    representation_covariance.to(covariance_dtype)
                    + difference[..., :, None] @ difference[..., None, :]
                )
            ).sum(dim=2)
        return initial_state, cell_covariance

    @staticmethod
    def _legacy_joint_output(
        pose: dict[str, object],
        pose_only_steps: int,
    ) -> dict[str, object]:
        pose = dict(pose)
        sequences = pose.pop("joint_deformation_output_sequences")
        contexts = pose.pop("joint_deformation_cell_context_sequence")
        representation_probability = pose.pop(
            "joint_deformation_representation_probability_sequence"
        )
        active = pose.pop("joint_deformation_active_sequence")
        feedback_render = pose.pop("joint_final_feedback_deformed_canonical_render")
        feedback_maps = pose.pop("joint_deformation_feedback_map_yx_px_sequence")
        feedback_enabled = pose.pop("joint_deformation_feedback_enabled_mask")
        batch, cells = contexts.shape[:2]
        final_render = pose["final_canonical_render"]
        final_probability = representation_probability[..., -1]
        marginalized_render = (
            final_probability.to(final_render)[..., None, None, None] * final_render
        ).sum(dim=2)
        final_map = feedback_maps[:, :, -1]
        deformed_render = warp_tensor_with_map_yx(
            marginalized_render.reshape(batch * cells, *marginalized_render.shape[2:]),
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
            "deformation_representation_probability_sequence": (
                representation_probability
            ),
            "deformation_cell_context_sequence": contexts,
            "deformation_active_sequence": active,
            "deformation_feedback_map_yx_px_sequence": feedback_maps,
            "deformation_feedback_enabled_mask": feedback_enabled,
            "deformation_gating_audit": {
                "pose_only_steps": int(pose_only_steps),
                "gate_policy": DEFORMATION_GATE_POLICY,
                "update_semantics": DEFORMATION_UPDATE_SEMANTICS,
                "representation_probabilities_detached": True,
                "dense_supervision_feedback_gate": (
                    "positive_weight_only; censored rows use detached identity"
                ),
                "feedback_semantics": (
                    "absolute_deformation_warps_next_finite_thickness_render"
                ),
                "shared_recurrent_context": True,
            },
            **sequences,
            **final_outputs,
            "final_representation_marginalized_canonical_render": (
                marginalized_render
            ),
            "final_feedback_deformed_canonical_render": (
                final_probability.to(feedback_render)[..., None, None, None]
                * feedback_render
            ).sum(dim=2),
            "final_deformed_canonical_render": deformed_render,
        }

    def forward(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        catalogue_batch: BoundCompleteCatalogueBatchV6,
        output_shape_h_w: tuple[int, int],
        retrieval_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        proposal_top_m: int,
        top_k: int,
        refinement_steps: int,
        training_truth_catalogue_index: torch.Tensor | None = None,
        dense_deformation_supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if self.pose_only_steps > refinement_steps + 1:
            raise ValueError("fixed pose-only steps must be between zero and T")
        truth_input = None
        if training_truth_catalogue_index is not None:
            truth_input = torch.as_tensor(
                training_truth_catalogue_index, device=image.device
            )
            if (
                truth_input.dtype == torch.bool
                or torch.is_floating_point(truth_input)
                or torch.is_complex(truth_input)
            ):
                raise ValueError("training truth catalogue indices must be integers")
            if truth_input.shape != (image.shape[0],) or bool(
                (
                    (truth_input < 0)
                    | (truth_input >= self.pose_model.catalogue_runtime_v6.cell_count)
                ).any()
            ):
                raise ValueError("training truth indices must have shape (B,) in catalogue")
            truth_input = truth_input.to(torch.long)
        cascade = self.pose_model.forward_proposed(
            image,
            outline,
            outline_available,
            atlas_volume,
            catalogue_batch,
            retrieval_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
            proposal_top_m=proposal_top_m,
            top_k=top_k,
            training_truth_catalogue_index=truth_input,
        )
        catalogue = verify_bound_complete_catalogue_batch_v6(
            catalogue_batch,
            expected_runtime=self.pose_model.catalogue_runtime_v6,
        )
        honest = cascade["honest_hybrid_posterior"]
        ready = cascade["honest_refinement_ready_mask"]
        finite_topk = honest["hybrid_topk_finite_rendered_mask"]
        if (
            ready.shape != (image.shape[0],)
            or ready.dtype != torch.bool
            or finite_topk.shape[0] != image.shape[0]
            or finite_topk.dtype != torch.bool
            or not torch.equal(ready, finite_topk.all(dim=1))
        ):
            raise RuntimeError("cascade readiness must exactly match finite-rendered top-K")
        source_row = torch.nonzero(ready, as_tuple=False).flatten()
        teacher_forced = torch.zeros_like(ready)
        scope = ["abstained_unclosed_honest_topk" for _ in range(image.shape[0])]
        if source_row.numel() == 0:
            return {
                "schema_version": JOINT_MODEL_V6_SCHEMA,
                "probabilities_calibrated": False,
                "probability_status": "raw_uncalibrated",
                "catalogue_binding": dict(catalogue_batch.binding),
                "cascade": cascade,
                "refinement_ready_mask": ready,
                "refinement_abstained_mask": ~ready,
                "refinement_source_batch_index": source_row,
                "refinement_teacher_forced_mask": teacher_forced,
                "refinement_selection_scope_by_sample": tuple(scope),
                "refinement_selected_catalogue_index": None,
                "refinement_selected_cell_id": None,
                "refinement_initial_honest_topk_catalogue_index": None,
                "refinement_initial_honest_mode_mask": None,
                "refinement_final_honest_mode_mask": None,
                "refinement_final_teacher_forced_mode_mask": None,
                "refinement_selected_full_catalogue_log_probability": None,
                "refinement_initial_topk_log_probability": None,
                "refinement_retained_probability": None,
                "refinement_omitted_probability": None,
                "refinement_truth_topk_index": None,
                "refined_topk_full_catalogue_log_probability": None,
                "refinement_performed_mask": ready,
                "refinement_performed": False,
                "refined_output": None,
            }

        selected = honest["hybrid_topk_catalogue_index"].index_select(
            0, source_row
        ).clone()
        initial_honest_selected = selected.clone()
        honest_full_log = honest["hybrid_cell_log_probability"].index_select(
            0, source_row
        )
        selected_full_log_source = honest_full_log
        truth_topk_index = selected.new_full((source_row.numel(),), -1)
        if self.training and truth_input is not None:
            truth = truth_input.index_select(0, source_row)
            match = selected.eq(truth[:, None])
            missing = ~match.any(dim=1)
            selected[missing, -1] = truth[missing]
            teacher_forced[source_row[missing]] = True
            teacher = cascade["training_teacher_forced_hybrid_posterior"]
            teacher_full_log = teacher["hybrid_cell_log_probability"].index_select(
                0, source_row
            )
            selected_full_log_source = torch.where(
                missing[:, None], teacher_full_log, honest_full_log
            )
            match = selected.eq(truth[:, None])
            truth_topk_index = match.to(torch.long).argmax(dim=1)
            training_finite = teacher["finite_rendered_mask"].index_select(
                0, source_row
            )
            if not bool(torch.gather(training_finite, 1, selected).all()):
                raise RuntimeError("teacher bootstrap selected a cell without a finite render")
            for row, forced in zip(source_row.tolist(), missing.tolist()):
                scope[row] = (
                    "teacher_forced_training_only_exact_truth_replaced_last"
                    if forced
                    else "honest_closed_topk_truth_already_present"
                )
        else:
            for row in source_row.tolist():
                scope[row] = "honest_closed_topk"

        selected_log_probability = torch.gather(
            selected_full_log_source, 1, selected
        )
        refinement_initial_log_probability = torch.log_softmax(
            selected_log_probability, dim=1
        ).detach()
        retained = selected_log_probability.exp().sum(dim=1)
        omitted = (1.0 - retained).clamp(0.0, 1.0)
        canonical_id = catalogue["cell_id"][selected]
        full_image = F.interpolate(
            image.index_select(0, source_row),
            output_shape_h_w,
            mode="bilinear",
            align_corners=False,
        )
        full_outline = F.interpolate(
            outline.index_select(0, source_row),
            output_shape_h_w,
            mode="bilinear",
            align_corners=False,
        )
        full_source = self.pose_model.encode_histology(
            full_image,
            full_outline,
            torch.as_tensor(outline_available, device=image.device).index_select(
                0, source_row
            ),
        )
        top_score = self._full_resolution_topk_score(
            full_source,
            atlas_volume,
            catalogue,
            source_row,
            selected,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            axial_offsets_um,
            axial_weights,
        )
        top_states = self._gather_cells(
            catalogue["cell_states"], source_row, selected
        )
        initial_state, initial_covariance = self._ranker_detached_initialization(
            top_score,
            top_states,
            catalogue["support_origin_ap_dv_ml_um"],
        )
        initial_joint_log_probability = (
            refinement_initial_log_probability[..., None]
            + top_score["representation_log_conditional_within_cell"]
        ).detach()
        top_affine = self._gather_cells(
            catalogue["representation_to_canonical_raster_affine"],
            source_row,
            selected,
        )
        dense_weight = (
            None
            if dense_deformation_supervision_weight is None
            else torch.as_tensor(
                dense_deformation_supervision_weight,
                device=image.device,
                dtype=image.dtype,
            ).index_select(0, source_row)
        )
        refinement = self.pose_model.refine(
            full_source,
            atlas_volume,
            initial_state,
            initial_joint_log_probability,
            top_affine,
            output_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            catalogue["support_origin_ap_dv_ml_um"],
            self._compact_schedule(axial_offsets_um, source_row, image.shape[0]),
            self._compact_schedule(axial_weights, source_row, image.shape[0]),
            refinement_steps,
            deformation_decoder=self.deformation_decoder,
            pose_only_steps=self.pose_only_steps,
            dense_deformation_supervision_weight=dense_weight,
        )
        compact_teacher = teacher_forced.index_select(0, source_row)
        full_log = torch.where(
            compact_teacher[:, None], selected_full_log_source, honest_full_log
        )
        pose = {
            "retrieval_cell_id": catalogue["cell_id"],
            "retrieval_cell_log_probability": full_log,
            "retrieval_cell_probability": full_log.exp(),
            "retrieval_topk_catalogue_index": selected,
            "retrieval_topk_cell_id": canonical_id,
            "retrieval_topk_log_probability": selected_log_probability,
            "retrieval_topk_retained_probability": retained,
            "retrieval_omitted_probability": omitted,
            "retrieval_teacher_forced_mask": compact_teacher,
            "catalogue_complete": True,
            "probabilities_calibrated": False,
            "retrieval_tail_scope": "complete_hybrid_catalogue_before_refinement",
            "topk_initial_representation_log_score": top_score[
                "representation_log_score"
            ],
            "topk_initial_representation_log_conditional_within_cell": top_score[
                "representation_log_conditional_within_cell"
            ],
            "topk_initial_cell_state": initial_state,
            "topk_initial_cell_canonical_plane_covariance": initial_covariance,
            "refinement_probability_scope": (
                "conditional_within_finite_render_closed_topk"
            ),
            "refinement_initial_topk_log_probability": (
                refinement_initial_log_probability
            ),
            **refinement,
        }
        refined_output = self._legacy_joint_output(pose, self.pose_only_steps)
        refined_partition = (
            retained[:, None].log()
            + refined_output["pose"][
                "conditional_within_topk_cell_log_probability"
            ]
        )
        if not torch.allclose(
            refined_partition.exp().sum(dim=1), retained, atol=3e-6, rtol=0.0
        ):
            raise RuntimeError("refinement changed retained full-catalogue mass")
        return {
            "schema_version": JOINT_MODEL_V6_SCHEMA,
            "probabilities_calibrated": False,
            "probability_status": "raw_uncalibrated",
            "catalogue_binding": dict(catalogue_batch.binding),
            "cascade": cascade,
            "refinement_ready_mask": ready,
            "refinement_abstained_mask": ~ready,
            "refinement_source_batch_index": source_row,
            "refinement_teacher_forced_mask": teacher_forced,
            "refinement_selection_scope_by_sample": tuple(scope),
            "refinement_selected_catalogue_index": selected,
            "refinement_selected_cell_id": canonical_id,
            "refinement_initial_honest_topk_catalogue_index": (
                initial_honest_selected
            ),
            "refinement_initial_honest_mode_mask": torch.ones_like(
                initial_honest_selected, dtype=torch.bool
            ),
            "refinement_final_honest_mode_mask": selected[..., None].eq(
                initial_honest_selected[:, None]
            ).any(dim=-1),
            "refinement_final_teacher_forced_mode_mask": ~selected[..., None].eq(
                initial_honest_selected[:, None]
            ).any(dim=-1),
            "refinement_selected_full_catalogue_log_probability": (
                selected_log_probability
            ),
            "refinement_initial_topk_log_probability": (
                refinement_initial_log_probability
            ),
            "refinement_retained_probability": retained,
            "refinement_omitted_probability": omitted,
            "refinement_truth_topk_index": truth_topk_index,
            "refined_topk_full_catalogue_log_probability": refined_partition,
            "refinement_performed_mask": ready,
            "refinement_performed": True,
            "refined_output": refined_output,
        }


__all__ = ["ArbitraryPlaneJointModelV6", "JOINT_MODEL_V6_SCHEMA"]
