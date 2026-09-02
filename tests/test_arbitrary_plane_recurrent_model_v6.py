import types

import pytest
import torch
from torch import nn

from tests.arbitrary_plane_production_v3_fixtures import atlas, catalogue
from training.arbitrary_plane_catalogue_runtime_v6 import (
    make_complete_catalogue_runtime_v6,
)
from training.arbitrary_plane_coarse_proposal_v6 import AntipodalPlaneProposalV6
from training.arbitrary_plane_recurrent_model_v6 import (
    ArbitraryPlaneRetrievalRefinementModelV6,
)


def _runtime():
    artifact = catalogue()
    return make_complete_catalogue_runtime_v6(
        artifact,
        expected_catalogue_receipt_sha256=artifact["receipt_sha256"],
        device="cpu",
        dtype=torch.float32,
    )


def _fixture(runtime, batch=1):
    image = torch.rand(batch, 1, 8, 8)
    return {
        "image": image,
        "outline": torch.ones_like(image),
        "available": torch.ones(batch),
        "volume": torch.from_numpy(atlas()),
        "catalogue": runtime.expand(batch),
    }


class _CountingModel(ArbitraryPlaneRetrievalRefinementModelV6):
    def __init__(self, runtime, *, budget=6, rounds=4):
        super().__init__(
            catalogue_runtime_v6=runtime,
            atlas_channels=2,
            feature_channels=4,
            hidden_channels=6,
            correlation_radius=1,
            proposal_channels=4,
            proposal_mixture_components=2,
            proposal_spatial_bins_h_w=(2, 2),
            cascade_max_rendered_cells_per_sample=budget,
            cascade_max_closure_rounds=rounds,
        )
        self.rendered_cell_counts = []
        self.proposal_input_shapes = []

    def encode_histology(self, image, outline, outline_available):
        self.proposal_input_shapes.append(tuple(image.shape[-2:]))
        return super().encode_histology(image, outline, outline_available)

    def _render_representations(self, atlas_volume, cell_states, *args, **kwargs):
        self.rendered_cell_counts.append(cell_states.shape[1])
        return super()._render_representations(atlas_volume, cell_states, *args, **kwargs)


def _proposal_only(model, fixture):
    return model.forward_proposal_only(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["catalogue"],
        (8, 8),
    )


def _proposed(
    model,
    fixture,
    axial_offsets=None,
    axial_weights=None,
    **kwargs,
):
    return model.forward_proposed(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["catalogue"],
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]) if axial_offsets is None else axial_offsets,
        torch.tensor([0.25, 0.5, 0.25]) if axial_weights is None else axial_weights,
        proposal_top_m=2,
        top_k=2,
        **kwargs,
    )


def test_bound_proposal_only_is_fresh_full_catalogue_zero_render_and_uses_shared_resolution():
    torch.manual_seed(1)
    runtime = _runtime()
    model = _CountingModel(runtime).eval()
    fixture = _fixture(runtime)
    assert isinstance(model.proposal_head_v6, AntipodalPlaneProposalV6)
    proposal_only = _proposal_only(model, fixture)
    assert proposal_only["catalogue_cell_count"] == runtime.cell_count
    assert proposal_only["atlas_render_count"] == 0
    assert proposal_only["retrieval_shape_h_w"] == (8, 8)
    assert model.rendered_cell_counts == []

    _proposed(model, fixture)
    assert model.proposal_input_shapes == [(8, 8), (8, 8)]

    forged = {}
    with pytest.raises(ValueError, match="not issued by the expected runtime"):
        model.forward_proposal_only(
            fixture["image"],
            fixture["outline"],
            fixture["available"],
            forged,
            (8, 8),
        )


class _FixedProposal(nn.Module):
    def forward(
        self,
        source_features,
        cell_states,
        cell_log_mass,
        support_origin,
        *,
        expected_catalogue_cell_count,
    ):
        probability = source_features.new_tensor(
            [0.29, 0.28, 0.18, 0.17, 0.04, 0.04]
        )
        assert probability.numel() == expected_catalogue_cell_count
        log_probability = probability.log().expand(source_features.shape[0], -1)
        return {
            "raw_full_catalogue_cell_log_probability": log_probability,
            "cell_log_probability": log_probability,
            "cell_probability": log_probability.exp(),
            "catalogue_complete": True,
        }


def _install_score(model, score_by_id):
    model.scored_canonical_ids = []
    model.scored_axial_shapes = []

    def score(self, source_features, atlas_volume, cell_id, cell_states, *args, **kwargs):
        self.scored_canonical_ids.append(cell_id.detach().cpu().tolist())
        self.scored_axial_shapes.append(
            (tuple(torch.as_tensor(args[-2]).shape), tuple(torch.as_tensor(args[-1]).shape))
        )
        score = source_features.new_tensor(
            [score_by_id.get(int(index), 0.0) for index in cell_id]
        )[None]
        return {"cell_id": cell_id, "cell_log_evidence": score}

    model.score_catalogue_chunk = types.MethodType(score, model)


def test_mixed_batch_renders_truth_only_for_miss_and_keeps_honest_hybrid_separate():
    runtime = _runtime()
    model = _CountingModel(runtime).train()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {})
    fixture = _fixture(runtime, batch=2)
    honest_reference = _proposed(model, fixture)
    _install_score(model, {})
    result = _proposed(
        model,
        fixture,
        training_truth_catalogue_index=torch.tensor([0, 3]),
    )

    assert model.scored_canonical_ids == [[0, 1], [0, 1], [3]]
    assert result["honest_initial_topm_truth_hit"].tolist() == [True, False]
    assert result["training_truth_forced_mask"].tolist() == [False, True]
    assert result["training_truth_position"].tolist() == [0, 2]
    assert result["training_selected_valid_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert result["honest_finite_rendered_cell_count_per_sample"].tolist() == [2, 2]
    assert result["training_truth_extra_rendered_cell_count_per_sample"].tolist() == [0, 1]
    assert result["total_finite_rendered_cell_count_per_sample"].tolist() == [2, 3]
    assert result["total_finite_rendered_physical_batch_slots"] == 5
    assert not result["probabilities_calibrated"]
    assert result["probability_status"] == "raw_uncalibrated"
    assert result["training_truth_finite_render_score_chunk_by_sample"][0] is None
    assert result["training_truth_finite_render_score_chunk_by_sample"][1][
        "cell_id"
    ].tolist() == [3]
    assert result["training_truth_leakage_into_honest_hybrid"] is False
    assert torch.equal(
        result["honest_hybrid_posterior"]["hybrid_cell_log_probability"],
        honest_reference["honest_hybrid_posterior"][
            "hybrid_cell_log_probability"
        ],
    )
    assert result["honest_hybrid_posterior"]["selection_scope"].endswith(
        "no_truth"
    )
    assert result["training_teacher_forced_hybrid_posterior"][
        "selection_scope"
    ].startswith("teacher_forced_training_only")
    assert "not_an_exact_likelihood" in result["finite_render_score_semantics"]


def test_adaptive_closure_renders_outside_mode_by_canonical_id_and_becomes_ready():
    runtime = _runtime()
    model = _CountingModel(runtime, budget=4, rounds=2).eval()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {1: -20.0})
    result = _proposed(model, _fixture(runtime))

    assert model.scored_canonical_ids == [[0, 1], [2]]
    assert result["honest_selected_catalogue_index"].tolist() == [[0, 1, 2]]
    assert result["honest_closure_rounds_used_per_sample"].tolist() == [1]
    assert result["honest_finite_rendered_cell_count_per_sample"].tolist() == [3]
    assert result["honest_hybrid_posterior"][
        "hybrid_topk_catalogue_index"
    ].tolist() == [[0, 2]]
    assert result["honest_hybrid_posterior"][
        "hybrid_topk_finite_rendered_mask"
    ].tolist() == [[True, True]]
    assert result["honest_refinement_ready_mask"].tolist() == [True]
    assert result["honest_refinement_abstention_reason"] == ("topk_closed",)


def test_frozen_render_budget_causes_explicit_abstention_without_hidden_render():
    runtime = _runtime()
    model = _CountingModel(runtime, budget=2, rounds=3).eval()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {1: -20.0})
    result = _proposed(model, _fixture(runtime))

    assert model.scored_canonical_ids == [[0, 1]]
    assert result["frozen_honest_closure_max_rendered_cells_per_sample"] == 2
    assert result["honest_refinement_ready_mask"].tolist() == [False]
    assert result["honest_refinement_abstained_mask"].tolist() == [True]
    assert result["honest_render_budget_exhausted_mask"].tolist() == [True]
    assert result["honest_refinement_abstention_reason"] == (
        "render_budget_exhausted",
    )
    assert result["honest_global_topk_has_unrendered_state"].tolist() == [True]
    assert result["honest_finite_rendered_cell_count_per_sample"].tolist() == [2]
    assert result["honest_global_topk_closed_mask"].tolist() == [False]
    assert "adaptive" in result["cascade_boundary"]


def test_per_sample_axial_schedules_are_sliced_by_original_row():
    runtime = _runtime()
    model = _CountingModel(runtime).eval()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {})
    fixture = _fixture(runtime, batch=2)
    offsets = torch.tensor([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0]])
    weights = torch.tensor([[0.25, 0.5, 0.25], [0.2, 0.6, 0.2]])
    _proposed(
        model,
        fixture,
        axial_offsets=offsets,
        axial_weights=weights,
    )
    assert model.scored_axial_shapes == [((1, 3), (1, 3)), ((1, 3), (1, 3))]


def test_teacher_truth_render_is_explicitly_outside_honest_closure_budget():
    runtime = _runtime()
    model = _CountingModel(runtime, budget=2).train()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {})
    result = _proposed(
        model,
        _fixture(runtime),
        training_truth_catalogue_index=torch.tensor([3]),
    )
    assert result["honest_finite_rendered_cell_count_per_sample"].tolist() == [2]
    assert result["training_truth_extra_rendered_cell_count_per_sample"].tolist() == [1]
    assert result["total_finite_rendered_cell_count_per_sample"].tolist() == [3]
    assert "training_truth_extra" in result["honest_closure_render_budget_scope"]


@pytest.mark.parametrize(
    "truth",
    [torch.tensor([3.0]), torch.tensor([True]), torch.tensor([3.0 + 0.0j])],
)
def test_training_truth_indices_must_be_integer(truth):
    runtime = _runtime()
    model = _CountingModel(runtime).train()
    model.proposal_head_v6 = _FixedProposal()
    _install_score(model, {})
    with pytest.raises(ValueError, match="must be integers"):
        _proposed(
            model,
            _fixture(runtime),
            training_truth_catalogue_index=truth,
        )


def test_proposal_gradients_remain_finite_without_any_finite_render():
    torch.manual_seed(7)
    runtime = _runtime()
    model = _CountingModel(runtime).train()
    result = _proposal_only(model, _fixture(runtime, batch=2))
    loss = -result["cell_log_probability"][:, 0].mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.proposal_head_v6.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
