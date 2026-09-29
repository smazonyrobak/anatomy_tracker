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
from training.arbitrary_plane_recurrent_model import (
    _pair_evidence,
    compose_antipodal_plane_frame_residual,
)
from training.arbitrary_plane_recurrent_model_v6 import (
    ArbitraryPlaneRetrievalRefinementModelV6,
)
from training.arbitrary_plane_joint_uncertainty import joint_lowrank_parameters
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_ribbon_v6 import compose_curved_ribbon_coordinates


JOINT_MODEL_V6_SCHEMA = "anatomy-tracker.joint-model/v6"
DEFORMATION_GATE_POLICY = "fixed_iteration_index_and_dense_supervision"
DEFORMATION_UPDATE_SEMANTICS = "absolute_per_iteration_not_accumulated"


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
        joint_uncertainty_rank: int | None = None,
        joint_uncertainty_coordinate_scale: tuple[float, ...] = (
            (0.1,) * 3 + (500.0,) * 3 + (0.1,) * 3 + (1.0,) * 26
        ),
        proposal_normal_readout_count: int | None = None,
        spatial_residual_blocks: int = 0,
        frame_centre_offset_conditioning: bool = False,
        coordinate_evidence_conditioning: bool = False,
        image_key_descriptor_dim: int | None = None,
        ribbon_deformation: bool = False,
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
            proposal_normal_readout_count=proposal_normal_readout_count,
            spatial_residual_blocks=spatial_residual_blocks,
            frame_centre_offset_conditioning=frame_centre_offset_conditioning,
            coordinate_evidence_conditioning=coordinate_evidence_conditioning,
            image_key_descriptor_dim=image_key_descriptor_dim,
        )
        self.deformation_decoder = AffineFreeSVFDecoder(
            hidden_channels,
            max_velocity_fraction_yx=max_velocity_fraction_yx,
            integration_steps=deformation_integration_steps,
            support_floor=deformation_support_floor,
            maximum_velocity_gradient=deformation_maximum_velocity_gradient,
        )
        # Opt-in only: old initialization, RNG state and state_dict stay exact.
        self.joint_uncertainty_rank = joint_uncertainty_rank
        if joint_uncertainty_rank is not None:
            if joint_uncertainty_rank < 1:
                raise ValueError("joint uncertainty rank must be positive")
            coordinate_scale = torch.tensor(joint_uncertainty_coordinate_scale)
            if coordinate_scale.shape != (35,) or not bool(
                torch.isfinite(coordinate_scale).all() and (coordinate_scale > 0).all()
            ):
                raise ValueError("joint uncertainty requires 35 positive physical coordinate scales")
            self.register_buffer("joint_uncertainty_coordinate_scale", coordinate_scale)
            self.joint_uncertainty_head = nn.Linear(
                hidden_channels, 35 * (1 + joint_uncertainty_rank)
            )
            nn.init.normal_(self.joint_uncertainty_head.weight, std=1e-3)
            nn.init.zeros_(self.joint_uncertainty_head.bias)
            with torch.no_grad():
                self.joint_uncertainty_head.bias[:35].fill_(
                    torch.expm1(torch.tensor(1.0 - 1e-4)).log().item()
                )
        self.ribbon_deformation = ribbon_deformation
        if ribbon_deformation:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() ^ 0x71BB06)
                self.ribbon_field_head = nn.Conv2d(hidden_channels, 6, 3, padding=1, device="cpu")
                nn.init.zeros_(self.ribbon_field_head.weight)
                nn.init.zeros_(self.ribbon_field_head.bias)

    def refine_ribbon(
        self, image, outline, outline_available, atlas_volume,
        initial_states, initial_component_log_mass, horizontal_reflection,
        output_shape_h_w, origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um,
        support_origin_ap_dv_ml_um, axial_offsets_um, axial_weights, steps,
    ):
        """Conditional native 3D refinement of supplied B,K states and B,K,R masses.

        Reflection flags R act only on the raster width axis. Hidden states,
        poses and ribbon fields remain distinct for every (sample,cell,flag).
        Known PSF schedules are S or B,S; section-processing maps are identity.
        Each sample must have at least one finite supplied component log mass.
        No global retrieval/tail, acquired constraints, SVF or covariance head
        is used. Returned probabilities normalize ONLY the supplied components
        using initial mass plus the final re-render score once; uncalibrated.
        """
        if not self.ribbon_deformation:
            raise ValueError("ribbon refinement requires ribbon_deformation=True")
        if initial_states.ndim != 3 or initial_states.shape[-1] != 12:
            raise ValueError("initial ribbon states must have shape (B,K,12)")
        batch, cells = initial_states.shape[:2]
        flags = torch.as_tensor(horizontal_reflection, device=image.device, dtype=torch.bool)
        if flags.ndim != 1 or initial_component_log_mass.shape != (batch, cells, flags.numel()):
            raise ValueError("ribbon reflection flags R and initial masses B,K,R must agree")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("ribbon steps must be a nonnegative integer")
        representations = flags.numel()
        count = batch * cells * representations
        height, width = output_shape_h_w
        pose = self.pose_model
        source_features = pose.encode_histology(
            F.interpolate(image, output_shape_h_w, mode="bilinear", align_corners=False),
            F.interpolate(outline, output_shape_h_w, mode="bilinear", align_corners=False),
            outline_available,
        )
        source = source_features[:, None, None].expand(
            batch, cells, representations, *source_features.shape[1:]
        ).reshape(count, *source_features.shape[1:])
        hidden = source.new_zeros(count, pose.recurrent_cell.candidate.out_channels, *source.shape[-2:])
        geometry_dtype = torch.float64 if initial_states.dtype == torch.float64 else torch.float32
        state = initial_states[:, :, None].expand(-1, -1, representations, -1).reshape(count, 12).to(dtype=geometry_dtype)
        flip = flags[None, None].expand(batch, cells, -1).reshape(count)
        offsets = pose._expanded_axial_schedule(
            axial_offsets_um, batch, cells * representations, state.device, geometry_dtype,
        )
        weights = pose._expanded_axial_schedule(
            axial_weights, batch, cells * representations, state.device, geometry_dtype,
        )
        states, residuals, directors, surfaces = [], [], [], []
        derivative_bounds, rescale_factors, updates = [], [], []
        for iteration in range(steps + 1):
            raw_field = (
                self.ribbon_field_head(hidden)
                if iteration >= self.pose_only_steps
                else hidden.new_zeros(count, 6, *hidden.shape[-2:])
            )
            with torch.autocast(device_type=state.device.type, enabled=False):
                raw_field = F.interpolate(raw_field.to(state), output_shape_h_w, mode="bilinear", align_corners=False)
                canonical_field = torch.where(flip[:, None, None, None], raw_field.flip(-1), raw_field)
                ribbon = compose_curved_ribbon_coordinates(
                    state, canonical_field[:, :3].tanh() * 200.,
                    canonical_field[:, 3:].tanh() * .2, offsets, .35,
                )
                canonical_grid = ribbon["ccf_coordinates_ap_dv_ml_um"]
                observed_grid = torch.where(
                    flip[:, None, None, None, None], canonical_grid.flip(-2), canonical_grid,
                )
                canonical_surface = ribbon["centre_surface_ccf_ap_dv_ml_um"]
                observed_surface = torch.where(
                    flip[:, None, None, None], canonical_surface.flip(-2), canonical_surface,
                )
                rendered = render_finite_thickness_coordinate_grid(
                    atlas_volume, observed_grid, origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um, weights,
                )
            atlas_features = pose._encode_atlas(rendered)
            evidence = pose.refinement_pair_encoder(_pair_evidence(source, atlas_features, pose.correlation_radius))
            if pose.coordinate_evidence_conditioning:
                with torch.autocast(device_type=state.device.type, enabled=False):
                    h, w = evidence.shape[-2:]
                    y, x = torch.meshgrid(
                        torch.arange(h, device=state.device, dtype=state.dtype) * 4,
                        torch.arange(w, device=state.device, dtype=state.dtype) * 4, indexing="ij",
                    )
                    raster_xy = 2 * (torch.stack((x, y)) + .5) / state.new_tensor((width, height))[:, None, None] - 1
                    signs = torch.stack((1. - 2. * flip.to(state), torch.ones_like(flip, dtype=state.dtype)), dim=1)
                    origin = torch.as_tensor(support_origin_ap_dv_ml_um, device=state.device, dtype=state.dtype)
                    centre_ccf = observed_surface[:, ::4, ::4].permute(0, 3, 1, 2)
                    coordinates = torch.cat((
                        (centre_ccf - origin[None, :, None, None]) / 10000.,
                        raster_xy[None].expand(count, -1, -1, -1),
                        signs[..., None, None].expand(-1, -1, h, w),
                    ), dim=1)
                    addition = pose.coordinate_evidence(coordinates.to(pose.coordinate_evidence.weight))
                evidence = evidence + addition.to(evidence)
            hidden = pose.recurrent_cell(evidence, hidden)
            states.append(state.reshape(batch, cells, representations, 12))
            residuals.append(ribbon["residual_local_um"].reshape(batch, cells, representations, 3, height, width))
            directors.append(ribbon["director_delta_local"].reshape(batch, cells, representations, 3, height, width))
            surfaces.append(observed_surface.reshape(batch, cells, representations, height, width, 3))
            derivative_bounds.append(ribbon["postlimit_derivative_frobenius_bound"].reshape(batch, cells, representations))
            rescale_factors.append(ribbon["deformation_rescale"].reshape(batch, cells, representations))
            if iteration < steps:
                update = pose._bounded_update(pose.recurrent_update(hidden.mean(dim=(-2, -1)))).to(state)
                with torch.autocast(device_type=state.device.type, enabled=False):
                    state = compose_antipodal_plane_frame_residual(state, update, support_origin_ap_dv_ml_um)
                updates.append(update.reshape(batch, cells, representations, 9))
        with torch.autocast(device_type=state.device.type, enabled=False):
            score = pose.recurrent_log_likelihood(hidden.mean(dim=(-2, -1)).to(pose.recurrent_log_likelihood.weight))
            score = score.reshape(batch, cells, representations).to(dtype=geometry_dtype)
            component_log_mass = initial_component_log_mass.to(score) + score
            log_probability = F.log_softmax(component_log_mass.reshape(batch, -1), dim=-1).reshape_as(score)
        return {
            "component_state_sequence": torch.stack(states, dim=3),
            "canonical_residual_local_um_sequence": torch.stack(residuals, dim=3),
            "canonical_director_delta_local_sequence": torch.stack(directors, dim=3),
            "observed_centre_ccf_ap_dv_ml_um_sequence": torch.stack(surfaces, dim=3),
            "postlimit_derivative_frobenius_bound_sequence": torch.stack(derivative_bounds, dim=3),
            "deformation_rescale_sequence": torch.stack(rescale_factors, dim=3),
            "component_pose_update_sequence": torch.stack(updates, dim=3) if updates else state.new_empty(batch, cells, representations, 0, 9),
            "deformation_active_sequence": torch.arange(steps + 1, device=state.device) >= self.pose_only_steps,
            "final_component_state": states[-1],
            "final_canonical_residual_local_um": residuals[-1],
            "final_canonical_director_delta_local": directors[-1],
            "final_observed_ccf_slab_ap_dv_ml_um": observed_grid.reshape(batch, cells, representations, *observed_grid.shape[1:]),
            "final_observed_centre_ccf_ap_dv_ml_um": surfaces[-1],
            "final_observed_render": rendered.reshape(batch, cells, representations, *rendered.shape[1:]),
            "final_component_refinement_log_score": score,
            "conditional_kept_component_log_probability": log_probability,
            "conditional_kept_component_probability": log_probability.exp(),
            "horizontal_reflection": flags,
            "probabilities_calibrated": False,
            "probability_scope": "conditional_within_supplied_kept_components_no_tail",
            "processing_scope": "identity_section_processing_known_psf_no_acquired_constraints",
            "field_semantics": "absolute_per_iteration_canonical_proper_frame_ribbon_not_2d_svf",
        }

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

    def _joint_uncertainty_output(
        self, refined_output, top_affine, retained, omitted, cell_id, teacher_forced,
    ):
        """Decode covariance from each final (cell, reflection) recurrent context."""
        pose = refined_output["pose"]
        context = pose["refinement_representation_context_sequence"][:, :, :, -1]
        with torch.autocast(device_type=context.device.type, enabled=False):
            raw = self.joint_uncertainty_head(context.float().mean(dim=(-2, -1)))
            diagonal_scale, factor = joint_lowrank_parameters(raw, self.joint_uncertainty_rank)
        representations = diagonal_scale.shape[2]
        dense_enabled = refined_output["deformation_feedback_enabled_mask"]
        mean_velocity = torch.where(
            dense_enabled[:, None, None, None, None],
            refined_output["final_stationary_velocity_yx_px"],
            torch.zeros_like(refined_output["final_stationary_velocity_yx_px"]),
        )
        return {
            "diagonal_scale": diagonal_scale,
            "factor": factor,
            "coordinate_scale": self.joint_uncertainty_coordinate_scale,
            "mean_frame_state": pose["final_cell_state"][:, :, None].expand(-1, -1, representations, -1),
            "mean_stationary_velocity_yx_px": mean_velocity[:, :, None].expand(-1, -1, representations, -1, -1, -1),
            "representation_to_canonical_raster_affine": top_affine,
            "conditional_cell_log_probability": pose["conditional_within_topk_cell_log_probability"],
            "representation_log_conditional_within_cell": pose["final_representation_log_conditional_within_cell"],
            "retained_probability": retained,
            "omitted_probability": omitted,
            "cell_id": cell_id,
            "teacher_forced_mask": teacher_forced,
            "velocity_supervision_eligible": dense_enabled & refined_output["deformation_active_sequence"][-1],
            "affine_projection_gauge": "uniform_canvas",
            "mean_scope": "cell_shared_frame_and_velocity_representation_specific_covariance",
            "probabilities_calibrated": False,
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
        frame_centre_offset_observation: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if self.pose_only_steps > refinement_steps + 1:
            raise ValueError("fixed pose-only steps must be between zero and T")
        if frame_centre_offset_observation is not None:
            frame_centre_offset_observation = torch.as_tensor(frame_centre_offset_observation, device=image.device)
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
            frame_centre_offset_observation=frame_centre_offset_observation,
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
            frame_centre_offset_observation=(
                None if frame_centre_offset_observation is None
                else frame_centre_offset_observation.index_select(0, source_row)
            ),
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
        if self.joint_uncertainty_rank is not None:
            refined_output["joint_uncertainty"] = self._joint_uncertainty_output(
                refined_output, top_affine, retained, omitted, canonical_id,
                teacher_forced.index_select(0, source_row),
            )
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
