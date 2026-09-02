import math

import pytest
import torch

from training.arbitrary_plane_hybrid_posterior_v6 import (
    hybrid_full_catalogue_posterior_v6,
    training_selection_with_truth_v6,
)


def _log(probability):
    return torch.tensor(probability, dtype=torch.float64).log()


def test_selected_mass_is_conserved_and_unselected_probabilities_are_unchanged():
    proposal = _log([[0.4, 0.3, 0.2, 0.1]])
    result = hybrid_full_catalogue_posterior_v6(
        proposal,
        torch.tensor([[0, 2]]),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        top_k=3,
    )
    hybrid = result["hybrid_cell_probability"]
    assert torch.allclose(hybrid.sum(1), torch.ones(1, dtype=torch.float64))
    assert torch.allclose(
        hybrid[:, [0, 2]].sum(1), torch.tensor([0.6], dtype=torch.float64)
    )
    assert torch.equal(hybrid[:, [1, 3]], proposal.exp()[:, [1, 3]])
    assert torch.allclose(
        result["selected_proposal_probability_mass"],
        result["selected_hybrid_probability_mass"],
    )
    assert not result["probabilities_calibrated"]


@pytest.mark.parametrize(
    "selected,evidence",
    [
        ([[0, 2]], [[3.0, 3.0]]),
        ([[1]], [[-5000.0]]),
    ],
)
def test_constant_or_singleton_exact_evidence_recovers_the_proposal(selected, evidence):
    proposal = _log([[0.4, 0.3, 0.2, 0.1]])
    result = hybrid_full_catalogue_posterior_v6(
        proposal,
        torch.tensor(selected),
        torch.tensor(evidence, dtype=torch.float64),
        top_k=2,
    )
    assert torch.allclose(result["hybrid_cell_probability"], proposal.exp())


def test_selecting_all_cells_matches_standard_log_domain_reweighting():
    proposal = _log([[0.4, 0.3, 0.2, 0.1]])
    evidence = torch.tensor([[10000.0, -10000.0, 2.0, -1.0]], dtype=torch.float64)
    result = hybrid_full_catalogue_posterior_v6(
        proposal,
        torch.tensor([[0, 1, 2, 3]]),
        evidence,
        top_k=2,
    )
    expected = torch.softmax(proposal + evidence, dim=1)
    assert torch.allclose(result["hybrid_cell_probability"], expected)
    assert torch.isfinite(result["hybrid_cell_log_probability"]).all()


def test_global_topk_can_include_an_unrendered_tail_cell():
    proposal = _log([[0.30, 0.29, 0.21, 0.20]])
    result = hybrid_full_catalogue_posterior_v6(
        proposal,
        torch.tensor([[0, 1]]),
        torch.tensor([[0.0, -20.0]], dtype=torch.float64),
        top_k=2,
    )
    assert result["hybrid_topk_catalogue_index"].tolist() == [[0, 2]]
    assert result["hybrid_topk_exact_evaluated_mask"].tolist() == [[True, False]]


def test_masked_training_union_is_batched_and_does_not_change_honest_recall():
    selection = training_selection_with_truth_v6(
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([1, 0]),
        4,
    )
    assert selection["honest_topm_truth_hit"].tolist() == [True, False]
    assert selection["training_truth_forced_mask"].tolist() == [False, True]
    assert selection["training_selected_valid_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    proposal = _log([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.4, 0.3]])
    evidence = torch.tensor(
        [[0.0, 1.0, float("nan")], [0.0, 1.0, 2.0]], dtype=torch.float64
    )
    result = hybrid_full_catalogue_posterior_v6(
        proposal,
        selection["training_selected_catalogue_index"],
        evidence,
        selected_valid_mask=selection["training_selected_valid_mask"],
        top_k=2,
    )
    assert torch.allclose(
        result["hybrid_cell_probability"].sum(1),
        torch.ones(2, dtype=torch.float64),
    )


def test_gradients_are_finite_and_valid_duplicates_are_rejected():
    logits = torch.tensor([[0.3, 0.2, -0.1, -0.4]], requires_grad=True)
    proposal = torch.log_softmax(logits, dim=1)
    evidence = torch.tensor([[0.1, -0.2]], requires_grad=True)
    result = hybrid_full_catalogue_posterior_v6(
        proposal, torch.tensor([[0, 2]]), evidence, top_k=2
    )
    (-result["hybrid_cell_log_probability"][0, 0]).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(evidence.grad).all()
    with pytest.raises(ValueError, match="unique per row"):
        hybrid_full_catalogue_posterior_v6(
            proposal.detach(),
            torch.tensor([[0, 0]]),
            torch.zeros(1, 2),
            top_k=2,
        )


def test_invalid_probability_or_evidence_contracts_fail_closed():
    with pytest.raises(ValueError, match="unit full-catalogue mass"):
        hybrid_full_catalogue_posterior_v6(
            torch.zeros(1, 4), torch.tensor([[0]]), torch.zeros(1, 1), top_k=1
        )
    with pytest.raises(ValueError, match="finite"):
        hybrid_full_catalogue_posterior_v6(
            torch.full((1, 4), -math.log(4.0)),
            torch.tensor([[0]]),
            torch.tensor([[float("nan")]]),
            top_k=1,
        )


def test_extreme_finite_evidence_has_zero_safe_finite_entropy():
    maximum = torch.finfo(torch.float32).max
    result = hybrid_full_catalogue_posterior_v6(
        torch.full((1, 4), -math.log(4.0)),
        torch.tensor([[0, 1]]),
        torch.tensor([[maximum, -maximum]]),
        top_k=2,
    )
    assert torch.isfinite(result["entropy"]).all()
    assert torch.isfinite(result["normalized_entropy"]).all()


def test_ties_are_stable_and_non_boolean_masks_or_truth_ids_are_rejected():
    proposal = torch.full((1, 4), -math.log(4.0))
    result = hybrid_full_catalogue_posterior_v6(
        proposal, torch.tensor([[0, 1]]), torch.zeros(1, 2), top_k=3
    )
    assert result["hybrid_topk_catalogue_index"].tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError, match="Boolean"):
        hybrid_full_catalogue_posterior_v6(
            proposal,
            torch.tensor([[0, 1]]),
            torch.zeros(1, 2),
            selected_valid_mask=torch.ones(1, 2),
            top_k=2,
        )
    with pytest.raises(ValueError, match="must be integers"):
        training_selection_with_truth_v6(
            torch.tensor([[0, 1]]), torch.tensor([1.9]), 4
        )
    with pytest.raises(ValueError, match="must be integers"):
        training_selection_with_truth_v6(
            torch.tensor([[0, 1]]), torch.tensor([True]), 4
        )
