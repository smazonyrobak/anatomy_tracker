import pytest
import torch

from training.arbitrary_plane_retrieval_loss_v6 import (
    full_catalogue_proposal_nll_v6,
    selected_exact_rerank_nll_v6,
)


def test_full_catalogue_proposal_nll_uses_truth_probability_and_weights():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], requires_grad=True)
    log_probability = torch.log_softmax(logits, dim=1)
    result = full_catalogue_proposal_nll_v6(
        log_probability,
        torch.tensor([0, 1]),
        expected_catalogue_cell_count=3,
        supervision_weight=torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(result["loss"], -log_probability[0, 0])
    assert result["eligible_row_count"] == 1
    assert result["full_catalogue_cell_count"] == 3
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_proposal_fractional_weights_are_scale_invariant_and_zero_weights_have_zero_gradient():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    log_probability = torch.log_softmax(logits, dim=1)
    half = full_catalogue_proposal_nll_v6(
        log_probability,
        torch.tensor([0, 1]),
        expected_catalogue_cell_count=2,
        supervision_weight=torch.tensor([0.25, 0.25]),
    )["loss"]
    full = full_catalogue_proposal_nll_v6(
        log_probability,
        torch.tensor([0, 1]),
        expected_catalogue_cell_count=2,
        supervision_weight=torch.tensor([1.0, 1.0]),
    )["loss"]
    assert torch.allclose(half, full)
    zero = full_catalogue_proposal_nll_v6(
        log_probability,
        torch.tensor([0, 1]),
        expected_catalogue_cell_count=2,
        supervision_weight=torch.zeros(2),
    )["loss"]
    assert zero.item() == 0.0
    zero.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_proposal_nll_rejects_local_or_unnormalized_probabilities():
    with pytest.raises(ValueError, match="complete-catalogue"):
        full_catalogue_proposal_nll_v6(
            torch.log_softmax(torch.zeros(1, 2), dim=1),
            torch.tensor([0]),
            expected_catalogue_cell_count=3,
        )


@pytest.mark.parametrize(
    "truth",
    [
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.tensor([0.0 + 0.0j]),
    ],
)
def test_proposal_nll_rejects_noninteger_truth_and_empty_batches(truth):
    with pytest.raises(ValueError, match="truth indices must be integers"):
        full_catalogue_proposal_nll_v6(
            torch.log_softmax(torch.zeros(1, 2), dim=1),
            truth,
            expected_catalogue_cell_count=2,
        )
    with pytest.raises(ValueError, match="complete-catalogue"):
        full_catalogue_proposal_nll_v6(
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.long),
            expected_catalogue_cell_count=2,
        )
    with pytest.raises(ValueError, match="unit full-catalogue"):
        full_catalogue_proposal_nll_v6(
            torch.zeros(1, 3),
            torch.tensor([0]),
            expected_catalogue_cell_count=3,
        )


def test_exact_rerank_nll_ignores_honest_misses_and_masked_padding():
    logits = torch.tensor([[2.0, 0.0, -4.0], [0.0, 1.0, 2.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    conditional_log = torch.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=1)
    result = selected_exact_rerank_nll_v6(
        conditional_log,
        valid,
        torch.tensor([1, -1]),
        torch.tensor([True, False]),
    )
    assert torch.allclose(result["loss"], -conditional_log[0, 1])
    assert result["eligible_row_count"] == 1
    assert result["truth_selected_row_count"] == 1
    assert result["truth_omitted_row_count"] == 1
    assert result["eligible_weight_sum"].item() == 1.0
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))


def test_exact_rerank_nll_requires_valid_selected_truth_and_normalization():
    valid = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="valid exact-rerank"):
        selected_exact_rerank_nll_v6(
            torch.tensor([[0.0, -torch.inf]]),
            valid,
            torch.tensor([1]),
            torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="unit selected mass"):
        selected_exact_rerank_nll_v6(
            torch.tensor([[-1.0, -torch.inf]]),
            valid,
            torch.tensor([0]),
            torch.tensor([True]),
        )


def test_exact_rerank_fractional_weights_are_scale_invariant_and_zero_eligible_is_differentiable():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    conditional_log = torch.log_softmax(logits, dim=1)
    valid = torch.ones_like(logits, dtype=torch.bool)
    truth_position = torch.tensor([0, 1])
    selected = torch.tensor([True, True])
    half = selected_exact_rerank_nll_v6(
        conditional_log,
        valid,
        truth_position,
        selected,
        supervision_weight=torch.tensor([0.25, 0.25]),
    )["loss"]
    full = selected_exact_rerank_nll_v6(
        conditional_log,
        valid,
        truth_position,
        selected,
        supervision_weight=torch.ones(2),
    )["loss"]
    assert torch.allclose(half, full)
    zero = selected_exact_rerank_nll_v6(
        conditional_log,
        valid,
        torch.tensor([-1, -1]),
        torch.tensor([False, False]),
    )["loss"]
    assert zero.item() == 0.0
    zero.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "position",
    [
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.tensor([0.0 + 0.0j]),
    ],
)
def test_exact_rerank_rejects_noninteger_positions_and_empty_batches(position):
    with pytest.raises(ValueError, match="truth positions must be integers"):
        selected_exact_rerank_nll_v6(
            torch.tensor([[0.0]]),
            torch.tensor([[True]]),
            position,
            torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="nonempty"):
        selected_exact_rerank_nll_v6(
            torch.empty(0, 1),
            torch.empty(0, 1, dtype=torch.bool),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.bool),
        )
