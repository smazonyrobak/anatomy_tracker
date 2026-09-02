"""Uncalibrated full-catalogue proposal with selected finite-render reranking."""

from __future__ import annotations

import math

import torch


HYBRID_POSTERIOR_V6_SCHEMA = "anatomy-tracker.hybrid-catalogue-posterior/v6"
HYBRID_POSTERIOR_V6_CALIBRATED = False


def training_selection_with_truth_v6(
    honest_topm_catalogue_index: torch.Tensor,
    truth_catalogue_index: torch.Tensor,
    catalogue_cell_count: int,
) -> dict[str, torch.Tensor]:
    """Append truth only when absent, without changing honest retrieval evidence."""
    honest = torch.as_tensor(honest_topm_catalogue_index)
    truth = torch.as_tensor(truth_catalogue_index, device=honest.device)
    if honest.ndim != 2 or honest.dtype == torch.bool or torch.is_floating_point(honest):
        raise ValueError("honest top-M indices must be one integer matrix")
    honest = honest.to(torch.long)
    if truth.dtype == torch.bool or torch.is_floating_point(truth):
        raise ValueError("truth catalogue indices must be integers")
    truth = truth.to(torch.long)
    if truth.shape != (honest.shape[0],):
        raise ValueError("truth catalogue indices must have shape (B,)")
    if (
        not isinstance(catalogue_cell_count, int)
        or isinstance(catalogue_cell_count, bool)
        or catalogue_cell_count < 1
        or bool(((honest < 0) | (honest >= catalogue_cell_count)).any())
        or bool(((truth < 0) | (truth >= catalogue_cell_count)).any())
    ):
        raise ValueError("training selection indices must lie in the declared catalogue")
    if honest.shape[1] < 1 or bool(
        (torch.sort(honest, dim=1).values[:, 1:]
         == torch.sort(honest, dim=1).values[:, :-1]).any()
    ):
        raise ValueError("honest top-M indices must be unique within every row")

    honest_hit = honest.eq(truth[:, None]).any(dim=1)
    selected = torch.cat((honest, truth[:, None]), dim=1)
    valid = torch.cat(
        (
            torch.ones_like(honest, dtype=torch.bool),
            (~honest_hit)[:, None],
        ),
        dim=1,
    )
    truth_position = torch.where(
        honest_hit,
        honest.eq(truth[:, None]).to(torch.long).argmax(dim=1),
        torch.full_like(truth, honest.shape[1]),
    )
    return {
        "honest_topm_catalogue_index": honest,
        "honest_topm_truth_hit": honest_hit,
        "training_selected_catalogue_index": selected,
        "training_selected_valid_mask": valid,
        "training_truth_position": truth_position,
        "training_truth_forced_mask": ~honest_hit,
    }


def hybrid_full_catalogue_posterior_v6(
    proposal_cell_log_probability: torch.Tensor,
    selected_catalogue_index: torch.Tensor,
    selected_exact_log_evidence: torch.Tensor,
    *,
    top_k: int,
    selected_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | bool | str]:
    """Conserve proposal mass on selected cells while reranking them exactly.

    The returned hybrid is approximate: exact finite-render evidence is available
    only on the selected set. Proposal probabilities outside that set are copied
    unchanged, and the selected set retains its original total proposal mass.
    """
    proposal_log = torch.as_tensor(proposal_cell_log_probability)
    selected = torch.as_tensor(
        selected_catalogue_index, device=proposal_log.device
    )
    evidence = torch.as_tensor(
        selected_exact_log_evidence,
        device=proposal_log.device,
        dtype=proposal_log.dtype,
    )
    if proposal_log.ndim != 2 or not torch.is_floating_point(proposal_log):
        raise ValueError("proposal log probability must have shape (B,K)")
    if not bool(torch.isfinite(proposal_log).all()):
        raise ValueError("proposal log probabilities must be finite")
    batch, cells = proposal_log.shape
    if selected.ndim != 2 or selected.shape[0] != batch:
        raise ValueError("selected catalogue indices must have shape (B,M)")
    if selected.dtype == torch.bool or torch.is_floating_point(selected):
        raise ValueError("selected catalogue indices must be integers")
    selected = selected.to(torch.long)
    if evidence.shape != selected.shape:
        raise ValueError("selected exact evidence must have shape (B,M)")
    if selected_valid_mask is None:
        valid = torch.ones_like(selected, dtype=torch.bool)
    else:
        valid = torch.as_tensor(selected_valid_mask, device=proposal_log.device)
        if valid.dtype != torch.bool:
            raise ValueError("selected valid mask must be Boolean")
    if valid.shape != selected.shape or not bool(valid.any(dim=1).all()):
        raise ValueError("every row must contain at least one valid selected cell")
    if bool(((selected < 0) | (selected >= cells)).any()):
        raise ValueError("selected catalogue indices are out of range")
    if not bool(torch.isfinite(evidence[valid]).all()):
        raise ValueError("valid selected exact evidence must be finite")
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= cells
    ):
        raise ValueError("top_k must select between one and all catalogue cells")
    normalized = torch.logsumexp(proposal_log, dim=1)
    if not torch.allclose(
        normalized,
        torch.zeros_like(normalized),
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("proposal probabilities must have unit full-catalogue mass")

    for row in range(batch):
        row_selected = selected[row, valid[row]]
        if torch.unique(row_selected).numel() != row_selected.numel():
            raise ValueError("valid selected catalogue indices must be unique per row")

    selected_proposal_log = torch.gather(proposal_log, 1, selected)
    negative_infinity = torch.full_like(selected_proposal_log, -torch.inf)
    valid_selected_proposal_log = torch.where(
        valid, selected_proposal_log, negative_infinity
    )
    selected_proposal_log_mass = torch.logsumexp(
        valid_selected_proposal_log, dim=1
    )
    exact_conditional_logit = torch.where(
        valid,
        selected_proposal_log + evidence,
        negative_infinity,
    )
    exact_conditional_log_probability = exact_conditional_logit - torch.logsumexp(
        exact_conditional_logit, dim=1, keepdim=True
    )
    selected_hybrid_log_probability = (
        selected_proposal_log_mass[:, None]
        + exact_conditional_log_probability
    )

    hybrid_log_probability = proposal_log.clone()
    batch_index = torch.arange(batch, device=proposal_log.device)[:, None].expand_as(
        selected
    )
    hybrid_log_probability[
        batch_index[valid], selected[valid]
    ] = selected_hybrid_log_probability[valid]
    hybrid_probability = hybrid_log_probability.exp()
    exact_evaluated_mask = torch.zeros_like(proposal_log, dtype=torch.bool)
    exact_evaluated_mask[batch_index[valid], selected[valid]] = True
    selected_proposal_probability_mass = selected_proposal_log_mass.exp()
    selected_hybrid_probability_mass = hybrid_probability.masked_fill(
        ~exact_evaluated_mask, 0.0
    ).sum(dim=1)
    tail_probability_mass = hybrid_probability.masked_fill(
        exact_evaluated_mask, 0.0
    ).sum(dim=1)
    if not torch.allclose(
        selected_hybrid_probability_mass,
        selected_proposal_probability_mass,
        atol=3e-6,
        rtol=0.0,
    ) or not torch.allclose(
        hybrid_probability.sum(dim=1),
        torch.ones(batch, device=proposal_log.device, dtype=proposal_log.dtype),
        atol=3e-6,
        rtol=0.0,
    ):
        raise RuntimeError("hybrid posterior failed its mass-conservation contract")

    topk_catalogue_index = torch.argsort(
        hybrid_log_probability, dim=1, descending=True, stable=True
    )[:, :top_k]
    topk_log_probability = torch.gather(
        hybrid_log_probability, 1, topk_catalogue_index
    )
    topk_probability = topk_log_probability.exp()
    topk_exact_evaluated = torch.gather(
        exact_evaluated_mask, 1, topk_catalogue_index
    )
    topk_retained_probability = topk_probability.sum(dim=1)
    entropy = torch.special.entr(hybrid_probability).sum(dim=1)
    selected_exact_conditional_probability = torch.where(
        valid,
        exact_conditional_log_probability.exp(),
        torch.zeros_like(exact_conditional_log_probability),
    )
    return {
        "schema_version": HYBRID_POSTERIOR_V6_SCHEMA,
        "probabilities_calibrated": HYBRID_POSTERIOR_V6_CALIBRATED,
        "probability_scope": "hybrid_full_catalogue",
        "exact_evidence_scope": "selected_cells_only",
        "tail_semantics": "unselected_proposal_probabilities_unchanged",
        "proposal_cell_log_probability": proposal_log,
        "proposal_cell_probability": proposal_log.exp(),
        "selected_catalogue_index": selected,
        "selected_valid_mask": valid,
        "selected_exact_log_evidence": evidence,
        "selected_exact_conditional_log_probability": exact_conditional_log_probability,
        "selected_exact_conditional_probability": selected_exact_conditional_probability,
        "selected_proposal_probability_mass": selected_proposal_probability_mass,
        "selected_hybrid_probability_mass": selected_hybrid_probability_mass,
        "tail_probability_mass": tail_probability_mass,
        "hybrid_cell_log_probability": hybrid_log_probability,
        "hybrid_cell_probability": hybrid_probability,
        "exact_evaluated_mask": exact_evaluated_mask,
        "hybrid_topk_catalogue_index": topk_catalogue_index,
        "hybrid_topk_log_probability": topk_log_probability,
        "hybrid_topk_probability": topk_probability,
        "hybrid_topk_exact_evaluated_mask": topk_exact_evaluated,
        "hybrid_topk_retained_probability": topk_retained_probability,
        "hybrid_omitted_probability": 1.0 - topk_retained_probability,
        "entropy": entropy,
        "normalized_entropy": entropy / max(math.log(cells), 1.0e-12),
    }


__all__ = [
    "HYBRID_POSTERIOR_V6_CALIBRATED",
    "HYBRID_POSTERIOR_V6_SCHEMA",
    "hybrid_full_catalogue_posterior_v6",
    "training_selection_with_truth_v6",
]
