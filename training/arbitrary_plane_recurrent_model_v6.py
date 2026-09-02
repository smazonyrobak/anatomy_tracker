"""Fresh-init full-catalogue proposal and bounded finite-render cascade."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from training.arbitrary_plane_catalogue_runtime_v6 import (
    BoundCompleteCatalogueBatchV6,
    CompleteCatalogueRuntimeV6,
    verify_bound_complete_catalogue_batch_v6,
    verify_complete_catalogue_runtime_v6,
)
from training.arbitrary_plane_coarse_proposal_v6 import AntipodalPlaneProposalV6
from training.arbitrary_plane_hybrid_posterior_v6 import (
    hybrid_full_catalogue_posterior_v6,
)
from training.arbitrary_plane_recurrent_model import (
    ArbitraryPlaneRetrievalRefinementModel,
)


RECURRENT_CASCADE_V6_SCHEMA = "anatomy-tracker.recurrent-proposal-cascade/v6"


class ArbitraryPlaneRetrievalRefinementModelV6(
    ArbitraryPlaneRetrievalRefinementModel
):
    """Stop at a closed, honest hybrid-catalogue boundary before refinement."""

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
    ):
        verify_complete_catalogue_runtime_v6(catalogue_runtime_v6)
        if (
            not isinstance(cascade_max_rendered_cells_per_sample, int)
            or isinstance(cascade_max_rendered_cells_per_sample, bool)
            or cascade_max_rendered_cells_per_sample < 1
        ):
            raise ValueError("cascade render budget must be a positive integer")
        if (
            not isinstance(cascade_max_closure_rounds, int)
            or isinstance(cascade_max_closure_rounds, bool)
            or cascade_max_closure_rounds < 0
        ):
            raise ValueError("cascade closure rounds must be a nonnegative integer")
        super().__init__(
            atlas_channels=atlas_channels,
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
            correlation_radius=correlation_radius,
            update_limits=update_limits,
            plane_tangent_scales=plane_tangent_scales,
        )
        self.catalogue_runtime_v6 = catalogue_runtime_v6
        self.cascade_max_rendered_cells_per_sample = (
            cascade_max_rendered_cells_per_sample
        )
        self.cascade_max_closure_rounds = cascade_max_closure_rounds
        self.proposal_head_v6 = AntipodalPlaneProposalV6(
            feature_channels=feature_channels,
            proposal_channels=proposal_channels,
            mixture_components=proposal_mixture_components,
            spatial_bins_h_w=proposal_spatial_bins_h_w,
            offset_scale_um=proposal_offset_scale_um,
        )

    @staticmethod
    def _validate_retrieval_shape(
        retrieval_shape_h_w: tuple[int, int],
    ) -> tuple[int, int]:
        if (
            not isinstance(retrieval_shape_h_w, tuple)
            or len(retrieval_shape_h_w) != 2
            or any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 4
                for size in retrieval_shape_h_w
            )
        ):
            raise ValueError("retrieval spatial sizes must be two integers of at least four")
        return retrieval_shape_h_w

    def _proposal_features(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        retrieval_shape_h_w: tuple[int, int],
    ) -> torch.Tensor:
        """Apply the identical explicit preprocessing in every proposal stage."""
        shape = self._validate_retrieval_shape(retrieval_shape_h_w)
        retrieval_image = F.interpolate(
            image, shape, mode="bilinear", align_corners=False
        )
        retrieval_outline = F.interpolate(
            outline, shape, mode="bilinear", align_corners=False
        )
        return self.encode_histology(
            retrieval_image, retrieval_outline, outline_available
        )

    def _catalogue(
        self,
        catalogue_batch: BoundCompleteCatalogueBatchV6,
        image: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        catalogue = verify_bound_complete_catalogue_batch_v6(
            catalogue_batch, expected_runtime=self.catalogue_runtime_v6
        )
        if catalogue_batch.batch_size != image.shape[0]:
            raise ValueError("bound catalogue and image must share batch size")
        if any(value.device != image.device for value in catalogue.values()):
            raise ValueError("bound catalogue and image must share device")
        return catalogue

    @staticmethod
    def _gather_row(
        value: torch.Tensor, row: int, index: torch.Tensor
    ) -> torch.Tensor:
        return value[row : row + 1, index]

    @staticmethod
    def _pad_selection(
        selected_by_sample: list[torch.Tensor],
        evidence_by_sample: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(index.numel() for index in selected_by_sample)
        padded_index = []
        padded_evidence = []
        valid_mask = []
        for index, evidence in zip(selected_by_sample, evidence_by_sample):
            padding = width - index.numel()
            padded_index.append(
                torch.cat((index, index.new_zeros(padding)), dim=0)
            )
            padded_evidence.append(
                torch.cat((evidence, evidence.new_zeros(padding)), dim=0)
            )
            valid_mask.append(
                torch.cat(
                    (
                        torch.ones_like(index, dtype=torch.bool),
                        torch.zeros(padding, device=index.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
            )
        return (
            torch.stack(padded_index),
            torch.stack(padded_evidence),
            torch.stack(valid_mask),
        )

    def _score_row(
        self,
        row: int,
        selected_index: torch.Tensor,
        source_features: torch.Tensor,
        atlas_volume: torch.Tensor,
        catalogue: Mapping[str, torch.Tensor],
        retrieval_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Render one row so legacy cell IDs remain canonical catalogue IDs."""
        canonical_id = catalogue["cell_id"][selected_index]
        batch = source_features.shape[0]
        offsets = torch.as_tensor(axial_offsets_um)
        weights = torch.as_tensor(axial_weights)
        if (
            offsets.shape != weights.shape
            or offsets.ndim not in (1, 2)
            or (offsets.ndim == 2 and offsets.shape[0] != batch)
        ):
            raise ValueError(
                "paired axial schedules must have shape (S,) or original-batch (B,S)"
            )
        row_offsets = offsets if offsets.ndim == 1 else offsets[row : row + 1]
        row_weights = weights if weights.ndim == 1 else weights[row : row + 1]
        return self.score_catalogue_chunk(
            source_features[row : row + 1],
            atlas_volume,
            canonical_id,
            self._gather_row(catalogue["cell_states"], row, selected_index),
            self._gather_row(catalogue["cell_log_mass"], row, selected_index),
            self._gather_row(
                catalogue["representation_log_weight"], row, selected_index
            ),
            self._gather_row(
                catalogue["representation_to_canonical_raster_affine"],
                row,
                selected_index,
            ),
            retrieval_shape_h_w,
            origin_ap_dv_ml_um,
            voxel_size_ap_dv_ml_um,
            catalogue["support_origin_ap_dv_ml_um"],
            row_offsets,
            row_weights,
        )

    @staticmethod
    def _public_hybrid(
        hybrid: Mapping[str, torch.Tensor | bool | str],
        *,
        selection_scope: str,
    ) -> dict[str, object]:
        """Name finite-render scores without implying an exact likelihood."""
        return {
            "schema_version": hybrid["schema_version"],
            "probabilities_calibrated": hybrid["probabilities_calibrated"],
            "probability_scope": hybrid["probability_scope"],
            "selection_scope": selection_scope,
            "finite_render_score_semantics": (
                "learned_discriminative_log_score_from_finite_thickness_atlas_render;"
                "not_an_exact_likelihood"
            ),
            "tail_semantics": hybrid["tail_semantics"],
            "proposal_cell_log_probability": hybrid[
                "proposal_cell_log_probability"
            ],
            "proposal_cell_probability": hybrid["proposal_cell_probability"],
            "selected_catalogue_index": hybrid["selected_catalogue_index"],
            "selected_valid_mask": hybrid["selected_valid_mask"],
            "selected_finite_rendered_learned_log_score": hybrid[
                "selected_exact_log_evidence"
            ],
            "selected_conditional_log_probability": hybrid[
                "selected_exact_conditional_log_probability"
            ],
            "selected_conditional_probability": hybrid[
                "selected_exact_conditional_probability"
            ],
            "selected_proposal_probability_mass": hybrid[
                "selected_proposal_probability_mass"
            ],
            "selected_hybrid_probability_mass": hybrid[
                "selected_hybrid_probability_mass"
            ],
            "tail_probability_mass": hybrid["tail_probability_mass"],
            "hybrid_cell_log_probability": hybrid[
                "hybrid_cell_log_probability"
            ],
            "hybrid_cell_probability": hybrid["hybrid_cell_probability"],
            "finite_rendered_mask": hybrid["exact_evaluated_mask"],
            "hybrid_topk_catalogue_index": hybrid[
                "hybrid_topk_catalogue_index"
            ],
            "hybrid_topk_log_probability": hybrid[
                "hybrid_topk_log_probability"
            ],
            "hybrid_topk_probability": hybrid["hybrid_topk_probability"],
            "hybrid_topk_finite_rendered_mask": hybrid[
                "hybrid_topk_exact_evaluated_mask"
            ],
            "hybrid_topk_retained_probability": hybrid[
                "hybrid_topk_retained_probability"
            ],
            "hybrid_omitted_probability": hybrid[
                "hybrid_omitted_probability"
            ],
            "entropy": hybrid["entropy"],
            "normalized_entropy": hybrid["normalized_entropy"],
        }

    def forward_proposal_only(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        catalogue_batch: BoundCompleteCatalogueBatchV6,
        retrieval_shape_h_w: tuple[int, int],
    ) -> dict[str, object]:
        """Score the bound complete catalogue without invoking an atlas renderer."""
        catalogue = self._catalogue(catalogue_batch, image)
        source_features = self._proposal_features(
            image, outline, outline_available, retrieval_shape_h_w
        )
        proposal = self.proposal_head_v6(
            source_features,
            catalogue["cell_states"],
            catalogue["cell_log_mass"],
            catalogue["support_origin_ap_dv_ml_um"],
            expected_catalogue_cell_count=self.catalogue_runtime_v6.cell_count,
        )
        return {
            **proposal,
            "cascade_schema_version": RECURRENT_CASCADE_V6_SCHEMA,
            "catalogue_binding": dict(catalogue_batch.binding),
            "retrieval_shape_h_w": retrieval_shape_h_w,
            "atlas_render_count": 0,
            "cascade_boundary": "full_catalogue_proposal_only",
        }

    def forward_proposed(
        self,
        image: torch.Tensor,
        outline: torch.Tensor,
        outline_available: torch.Tensor,
        atlas_volume: torch.Tensor,
        catalogue_batch: BoundCompleteCatalogueBatchV6,
        retrieval_shape_h_w: tuple[int, int],
        origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        voxel_size_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
        axial_offsets_um: torch.Tensor,
        axial_weights: torch.Tensor,
        *,
        proposal_top_m: int,
        top_k: int,
        training_truth_catalogue_index: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Propose globally and close honest top-K modes under a fixed render budget."""
        catalogue = self._catalogue(catalogue_batch, image)
        cells = self.catalogue_runtime_v6.cell_count
        if (
            not isinstance(proposal_top_m, int)
            or isinstance(proposal_top_m, bool)
            or not 1 <= proposal_top_m <= cells
        ):
            raise ValueError("proposal_top_m must select between one and all catalogue cells")
        if proposal_top_m > self.cascade_max_rendered_cells_per_sample:
            raise ValueError("proposal_top_m exceeds the frozen cascade render budget")
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= cells
        ):
            raise ValueError("top_k must select between one and all catalogue cells")

        source_features = self._proposal_features(
            image, outline, outline_available, retrieval_shape_h_w
        )
        proposal = self.proposal_head_v6(
            source_features,
            catalogue["cell_states"],
            catalogue["cell_log_mass"],
            catalogue["support_origin_ap_dv_ml_um"],
            expected_catalogue_cell_count=cells,
        )
        proposal_log_probability = proposal[
            "raw_full_catalogue_cell_log_probability"
        ]
        honest_topm = torch.argsort(
            proposal_log_probability, dim=1, descending=True, stable=True
        )[:, :proposal_top_m]
        honest_topm_log_probability = torch.gather(
            proposal_log_probability, 1, honest_topm
        )

        selected_by_sample = [row.clone() for row in honest_topm]
        evidence_by_sample: list[torch.Tensor] = []
        honest_chunks_by_sample: list[list[dict[str, torch.Tensor]]] = []
        for row, selected in enumerate(selected_by_sample):
            chunk = self._score_row(
                row,
                selected,
                source_features,
                atlas_volume,
                catalogue,
                retrieval_shape_h_w,
                origin_ap_dv_ml_um,
                voxel_size_ap_dv_ml_um,
                axial_offsets_um,
                axial_weights,
            )
            evidence_by_sample.append(chunk["cell_log_evidence"][0])
            honest_chunks_by_sample.append([chunk])

        closure_rounds = torch.zeros(
            image.shape[0], device=image.device, dtype=torch.long
        )
        for _ in range(self.cascade_max_closure_rounds + 1):
            padded_selected, padded_evidence, selected_valid = self._pad_selection(
                selected_by_sample, evidence_by_sample
            )
            honest_hybrid_raw = hybrid_full_catalogue_posterior_v6(
                proposal_log_probability,
                padded_selected,
                padded_evidence,
                top_k=top_k,
                selected_valid_mask=selected_valid,
            )
            topk_index = honest_hybrid_raw["hybrid_topk_catalogue_index"]
            topk_rendered = honest_hybrid_raw[
                "hybrid_topk_exact_evaluated_mask"
            ]
            if bool(topk_rendered.all()):
                break
            if int(closure_rounds.max().item()) >= self.cascade_max_closure_rounds:
                break

            added_any = False
            for row in range(image.shape[0]):
                outside = topk_index[row, ~topk_rendered[row]]
                remaining = (
                    self.cascade_max_rendered_cells_per_sample
                    - selected_by_sample[row].numel()
                )
                if outside.numel() == 0 or remaining == 0:
                    continue
                add = outside[:remaining]
                already_selected = torch.isin(add, selected_by_sample[row])
                add = add[~already_selected]
                if add.numel() == 0:
                    continue
                chunk = self._score_row(
                    row,
                    add,
                    source_features,
                    atlas_volume,
                    catalogue,
                    retrieval_shape_h_w,
                    origin_ap_dv_ml_um,
                    voxel_size_ap_dv_ml_um,
                    axial_offsets_um,
                    axial_weights,
                )
                selected_by_sample[row] = torch.cat(
                    (selected_by_sample[row], add), dim=0
                )
                evidence_by_sample[row] = torch.cat(
                    (evidence_by_sample[row], chunk["cell_log_evidence"][0]),
                    dim=0,
                )
                honest_chunks_by_sample[row].append(chunk)
                closure_rounds[row] += 1
                added_any = True
            if not added_any:
                break

        padded_selected, padded_evidence, selected_valid = self._pad_selection(
            selected_by_sample, evidence_by_sample
        )
        honest_hybrid_raw = hybrid_full_catalogue_posterior_v6(
            proposal_log_probability,
            padded_selected,
            padded_evidence,
            top_k=top_k,
            selected_valid_mask=selected_valid,
        )
        honest_topk_rendered = honest_hybrid_raw[
            "hybrid_topk_exact_evaluated_mask"
        ]
        refinement_ready = honest_topk_rendered.all(dim=1)
        rendered_count = selected_valid.sum(dim=1)
        budget_exhausted = (
            ~refinement_ready
            & rendered_count.eq(self.cascade_max_rendered_cells_per_sample)
        )
        round_limit = (
            ~refinement_ready
            & closure_rounds.eq(self.cascade_max_closure_rounds)
            & ~budget_exhausted
        )
        abstention_reason = tuple(
            "topk_closed"
            if bool(refinement_ready[row])
            else "render_budget_exhausted"
            if bool(budget_exhausted[row])
            else "closure_round_limit_reached"
            if bool(round_limit[row])
            else "closure_could_not_progress"
            for row in range(image.shape[0])
        )

        truth = None
        honest_topm_truth_hit = None
        training_truth_position = None
        training_truth_forced = torch.zeros(
            image.shape[0], device=image.device, dtype=torch.bool
        )
        truth_chunks_by_sample: list[dict[str, torch.Tensor] | None] = [
            None for _ in range(image.shape[0])
        ]
        teacher_hybrid = None
        if training_truth_catalogue_index is not None:
            if not self.training:
                raise ValueError("truth-augmented finite rendering is training-only")
            truth = torch.as_tensor(
                training_truth_catalogue_index, device=image.device
            )
            if (
                truth.dtype == torch.bool
                or torch.is_floating_point(truth)
                or torch.is_complex(truth)
            ):
                raise ValueError("training truth catalogue indices must be integers")
            truth = truth.to(torch.long)
            if truth.shape != (image.shape[0],) or bool(
                ((truth < 0) | (truth >= cells)).any()
            ):
                raise ValueError("training truth indices must have shape (B,) in catalogue")
            honest_topm_truth_hit = honest_topm.eq(truth[:, None]).any(dim=1)
            truth_positions = []
            training_selected_by_sample = []
            training_evidence_by_sample = []
            for row in range(image.shape[0]):
                match = selected_by_sample[row].eq(truth[row])
                if bool(match.any()):
                    truth_positions.append(
                        match.to(torch.long).argmax().reshape(())
                    )
                    training_selected_by_sample.append(selected_by_sample[row])
                    training_evidence_by_sample.append(evidence_by_sample[row])
                    continue
                truth_index = truth[row : row + 1]
                chunk = self._score_row(
                    row,
                    truth_index,
                    source_features,
                    atlas_volume,
                    catalogue,
                    retrieval_shape_h_w,
                    origin_ap_dv_ml_um,
                    voxel_size_ap_dv_ml_um,
                    axial_offsets_um,
                    axial_weights,
                )
                truth_chunks_by_sample[row] = chunk
                training_truth_forced[row] = True
                truth_positions.append(
                    truth.new_tensor(selected_by_sample[row].numel())
                )
                training_selected_by_sample.append(
                    torch.cat((selected_by_sample[row], truth_index), dim=0)
                )
                training_evidence_by_sample.append(
                    torch.cat(
                        (evidence_by_sample[row], chunk["cell_log_evidence"][0]),
                        dim=0,
                    )
                )
            training_truth_position = torch.stack(truth_positions)
            (
                training_selected,
                training_evidence,
                training_valid,
            ) = self._pad_selection(
                training_selected_by_sample, training_evidence_by_sample
            )
            teacher_hybrid_raw = hybrid_full_catalogue_posterior_v6(
                proposal_log_probability,
                training_selected,
                training_evidence,
                top_k=top_k,
                selected_valid_mask=training_valid,
            )
            teacher_hybrid = self._public_hybrid(
                teacher_hybrid_raw,
                selection_scope="teacher_forced_training_only_truth_augmented",
            )
        else:
            training_selected, training_evidence, training_valid = (
                padded_selected,
                padded_evidence,
                selected_valid,
            )

        honest_hybrid = self._public_hybrid(
            honest_hybrid_raw,
            selection_scope="honest_proposal_plus_adaptive_closure_no_truth",
        )
        truth_extra_count = training_truth_forced.to(torch.long)
        return {
            "schema_version": RECURRENT_CASCADE_V6_SCHEMA,
            "probabilities_calibrated": False,
            "probability_status": "raw_uncalibrated",
            "catalogue_binding": dict(catalogue_batch.binding),
            "retrieval_shape_h_w": retrieval_shape_h_w,
            "proposal": proposal,
            "raw_full_catalogue_proposal_log_probability": proposal_log_probability,
            "honest_initial_topm_catalogue_index": honest_topm,
            "honest_initial_topm_log_probability": honest_topm_log_probability,
            "honest_initial_topm_truth_hit": honest_topm_truth_hit,
            "honest_selected_catalogue_index": padded_selected,
            "honest_selected_valid_mask": selected_valid,
            "honest_finite_rendered_learned_log_score": padded_evidence,
            "honest_finite_render_score_chunks_by_sample": tuple(
                tuple(chunks) for chunks in honest_chunks_by_sample
            ),
            "honest_hybrid_posterior": honest_hybrid,
            "honest_global_topk_outside_finite_rendered_mask": (
                ~honest_topk_rendered
            ),
            "honest_global_topk_has_unrendered_state": (
                ~honest_topk_rendered
            ).any(dim=1),
            "honest_refinement_ready_mask": refinement_ready,
            "honest_refinement_abstained_mask": ~refinement_ready,
            "honest_refinement_abstention_reason": abstention_reason,
            "honest_render_budget_exhausted_mask": budget_exhausted,
            "honest_closure_round_limit_reached_mask": round_limit,
            "honest_closure_rounds_used_per_sample": closure_rounds,
            "frozen_honest_closure_max_rendered_cells_per_sample": (
                self.cascade_max_rendered_cells_per_sample
            ),
            "honest_closure_render_budget_scope": (
                "honest_inference_available_selection_only;"
                "training_truth_extra_render_of_at_most_one_cell_is_excluded"
            ),
            "frozen_max_closure_rounds": self.cascade_max_closure_rounds,
            "honest_finite_rendered_cell_count_per_sample": rendered_count,
            "honest_finite_rendered_physical_batch_slots": int(
                rendered_count.sum().item()
            ),
            "training_truth_catalogue_index": truth,
            "training_truth_position": training_truth_position,
            "training_truth_forced_mask": training_truth_forced,
            "training_selection_scope": (
                "teacher_forced_training_only_truth_augmented"
                if truth is not None
                else "honest_no_truth_requested"
            ),
            "training_selected_catalogue_index": training_selected,
            "training_selected_valid_mask": training_valid,
            "training_finite_rendered_learned_log_score": training_evidence,
            "training_truth_finite_render_score_chunk_by_sample": tuple(
                truth_chunks_by_sample
            ),
            "training_teacher_forced_hybrid_posterior": teacher_hybrid,
            "training_truth_leakage_into_honest_hybrid": False,
            "training_truth_extra_rendered_cell_count_per_sample": truth_extra_count,
            "total_finite_rendered_cell_count_per_sample": (
                rendered_count + truth_extra_count
            ),
            "total_finite_rendered_physical_batch_slots": int(
                (rendered_count + truth_extra_count).sum().item()
            ),
            "refinement_performed": False,
            "cascade_boundary": "adaptive_honest_hybrid_global_topk_before_refinement",
            "honest_global_topk_closed_mask": refinement_ready,
            "finite_render_score_semantics": (
                "learned_discriminative_score_from_finite_thickness_atlas_render;"
                "not_an_exact_likelihood"
            ),
        }


__all__ = [
    "ArbitraryPlaneRetrievalRefinementModelV6",
    "RECURRENT_CASCADE_V6_SCHEMA",
]
