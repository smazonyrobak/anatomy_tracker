"""Full-catalogue proposal and selected exact-rerank losses for v6."""

from __future__ import annotations

import torch
import torch.nn.functional as F


RETRIEVAL_LOSS_V6_SCHEMA = "anatomy-tracker.retrieval-loss/v6"


def _supervision_weight(
    value: torch.Tensor | None,
    batch: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    weight = (
        torch.ones(batch, device=reference.device, dtype=reference.dtype)
        if value is None
        else torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    )
    if weight.shape != (batch,) or not bool(
        torch.isfinite(weight).all()
        and (weight >= 0.0).all()
        and (weight <= 1.0).all()
    ):
        raise ValueError("retrieval supervision weight must be finite in [0,1] with shape (B,)")
    return weight


def full_catalogue_proposal_nll_v6(
    proposal_cell_log_probability: torch.Tensor,
    truth_catalogue_index: torch.Tensor,
    *,
    expected_catalogue_cell_count: int,
    supervision_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | str]:
    proposal_log = torch.as_tensor(proposal_cell_log_probability)
    if (
        not isinstance(expected_catalogue_cell_count, int)
        or isinstance(expected_catalogue_cell_count, bool)
        or expected_catalogue_cell_count < 1
        or proposal_log.ndim != 2
        or proposal_log.shape[0] < 1
        or not torch.is_floating_point(proposal_log)
        or not bool(torch.isfinite(proposal_log).all())
        or proposal_log.shape[1] != expected_catalogue_cell_count
    ):
        raise ValueError("proposal loss requires finite complete-catalogue log probabilities")
    if not torch.allclose(
        torch.logsumexp(proposal_log, dim=1),
        torch.zeros(proposal_log.shape[0], device=proposal_log.device, dtype=proposal_log.dtype),
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("proposal loss requires unit full-catalogue probability mass")
    truth = torch.as_tensor(truth_catalogue_index, device=proposal_log.device)
    if (
        truth.dtype == torch.bool
        or torch.is_floating_point(truth)
        or torch.is_complex(truth)
    ):
        raise ValueError("proposal truth indices must be integers")
    truth = truth.to(torch.long)
    if truth.shape != (proposal_log.shape[0],) or bool(
        ((truth < 0) | (truth >= proposal_log.shape[1])).any()
    ):
        raise ValueError("proposal truth indices must have shape (B,) within the catalogue")
    weight = _supervision_weight(supervision_weight, proposal_log.shape[0], proposal_log)
    per_row = F.nll_loss(proposal_log, truth, reduction="none")
    denominator = weight.sum()
    numerator = (per_row * weight).sum()
    loss = numerator / torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator)
    )
    return {
        "schema_version": RETRIEVAL_LOSS_V6_SCHEMA,
        "loss": loss,
        "per_row_nll": per_row,
        "eligible_row_count": int((weight > 0.0).sum().item()),
        "supervision_weight_sum": denominator,
        "full_catalogue_cell_count": int(expected_catalogue_cell_count),
    }


def selected_exact_rerank_nll_v6(
    selected_exact_conditional_log_probability: torch.Tensor,
    selected_valid_mask: torch.Tensor,
    truth_selected_position: torch.Tensor,
    truth_selected_mask: torch.Tensor,
    *,
    supervision_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | str]:
    conditional_log = torch.as_tensor(selected_exact_conditional_log_probability)
    valid = torch.as_tensor(selected_valid_mask, device=conditional_log.device)
    if conditional_log.ndim != 2 or not torch.is_floating_point(conditional_log):
        raise ValueError("exact-rerank conditional log probabilities must have shape (B,M)")
    if conditional_log.shape[0] < 1 or conditional_log.shape[1] < 1:
        raise ValueError("exact-rerank loss requires a nonempty batch and selection")
    if valid.shape != conditional_log.shape or valid.dtype != torch.bool:
        raise ValueError("exact-rerank valid mask must be one Boolean (B,M) tensor")
    if not bool(valid.any(dim=1).all()) or not bool(torch.isfinite(conditional_log[valid]).all()):
        raise ValueError("each exact-rerank row needs finite probability on a valid cell")
    if bool((~valid & ~torch.isneginf(conditional_log)).any()):
        raise ValueError("invalid exact-rerank positions must have negative-infinite log mass")
    normalized = torch.logsumexp(
        conditional_log.masked_fill(~valid, -torch.inf), dim=1
    )
    if not torch.allclose(
        normalized,
        torch.zeros_like(normalized),
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("exact-rerank conditional probabilities must have unit selected mass")

    truth_position = torch.as_tensor(
        truth_selected_position, device=conditional_log.device
    )
    if (
        truth_position.dtype == torch.bool
        or torch.is_floating_point(truth_position)
        or torch.is_complex(truth_position)
    ):
        raise ValueError("exact-rerank truth positions must be integers")
    truth_position = truth_position.to(torch.long)
    truth_selected = torch.as_tensor(
        truth_selected_mask, device=conditional_log.device
    )
    batch, selected_count = conditional_log.shape
    if truth_position.shape != (batch,) or truth_selected.shape != (batch,):
        raise ValueError("exact-rerank truth position and mask must have shape (B,)")
    if truth_selected.dtype != torch.bool:
        raise ValueError("exact-rerank truth-selected mask must be Boolean")
    safe_position = truth_position.clamp(0, selected_count - 1)
    if bool((
        truth_selected
        & (
            (truth_position < 0)
            | (truth_position >= selected_count)
            | ~torch.gather(valid, 1, safe_position[:, None]).squeeze(1)
        )
    ).any()):
        raise ValueError("selected truth positions must identify valid exact-rerank cells")
    weight = _supervision_weight(supervision_weight, batch, conditional_log)
    eligible_weight = weight * truth_selected.to(weight)
    gathered = -torch.gather(
        conditional_log, 1, safe_position[:, None]
    ).squeeze(1)
    per_row = torch.where(truth_selected, gathered, torch.zeros_like(gathered))
    supplied_weight_sum = weight.sum()
    eligible_weight_sum = eligible_weight.sum()
    numerator = (per_row * eligible_weight).sum()
    loss = numerator / torch.where(
        eligible_weight_sum > 0.0,
        eligible_weight_sum,
        torch.ones_like(eligible_weight_sum),
    )
    return {
        "schema_version": RETRIEVAL_LOSS_V6_SCHEMA,
        "loss": loss,
        "per_row_nll": per_row,
        "eligible_row_count": int((eligible_weight > 0.0).sum().item()),
        "truth_selected_row_count": int(truth_selected.sum().item()),
        "truth_omitted_row_count": int((~truth_selected).sum().item()),
        "supplied_supervision_weight_sum": supplied_weight_sum,
        "eligible_weight_sum": eligible_weight_sum,
    }


__all__ = [
    "RETRIEVAL_LOSS_V6_SCHEMA",
    "full_catalogue_proposal_nll_v6",
    "selected_exact_rerank_nll_v6",
]
