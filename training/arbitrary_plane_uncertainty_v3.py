"""Multimodal pose uncertainty, calibration, and trajectory propagation."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_recurrent_model import (
    RETRIEVAL_TAIL_SCOPE,
    compose_antipodal_plane_frame_residual,
)


UNCERTAINTY_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-uncertainty/v3"
CALIBRATION_V3_SCHEMA = "anatomy-tracker.hierarchical-calibration/v3"
CREDIBLE_LEVELS = (0.50, 0.80, 0.90, 0.95)
_CHI2_DF3_QUANTILES = (2.365973884, 4.641627676, 6.251388631, 7.814727903)
_ANIMAL_BOOTSTRAP_REPLICATES = 2048
_ANIMAL_BOOTSTRAP_SEED = 1729


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json(value.item())
        return _json(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("receipts require finite values")
        return value
    return value


def _sha(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_receipt(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def categorical_calibration_metrics_v3(
    logits: torch.Tensor,
    target_cell_index: torch.Tensor,
    temperature: float = 1.0,
    *,
    ece_bin_count: int = 15,
):
    """Categorical NLL, ECE, and minimum-posterior-set coverage."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    target = torch.as_tensor(target_cell_index, device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or target.shape != (logits.shape[0],):
        raise ValueError("logits and targets must have shapes (N,C) and (N,)")
    if logits.shape[0] < 1 or logits.shape[1] < 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("calibration requires finite nonempty multiclass logits")
    if bool(((target < 0) | (target >= logits.shape[1])).any()):
        raise ValueError("target catalogue indices are out of range")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(ece_bin_count, int) or isinstance(ece_bin_count, bool) or ece_bin_count < 2:
        raise ValueError("ECE bin count must be at least two")

    log_probability = torch.log_softmax(logits / float(temperature), dim=1)
    probability = log_probability.exp()
    nll = -log_probability.gather(1, target[:, None]).mean()
    confidence, prediction = probability.max(dim=1)
    correct = prediction.eq(target).to(probability)
    ece = probability.new_zeros(())
    edges = torch.linspace(0.0, 1.0, ece_bin_count + 1, device=probability.device)
    for index in range(ece_bin_count):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == ece_bin_count - 1
            else confidence < edges[index + 1]
        )
        if bool(selected.any()):
            ece = ece + selected.to(probability).mean() * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()

    sorted_probability, order = probability.sort(dim=1, descending=True, stable=True)
    cumulative = sorted_probability.cumsum(dim=1)
    target_rank = order.eq(target[:, None]).to(torch.int64).argmax(dim=1)
    coverage = {}
    for level in CREDIBLE_LEVELS:
        prefix_length = (cumulative < level).sum(dim=1).clamp_max(logits.shape[1] - 1)
        coverage[f"{int(level * 100)}"] = float(
            (target_rank <= prefix_length).to(probability).mean().item()
        )
    return {
        "sample_count": int(logits.shape[0]),
        "class_count": int(logits.shape[1]),
        "temperature": float(temperature),
        "nll": float(nll.item()),
        "ece": float(ece.item()),
        "minimum_posterior_set_coverage": coverage,
        "credible_levels": list(CREDIBLE_LEVELS),
    }


def continuous_calibration_metrics_v3(
    tangent_residual: torch.Tensor,
    tangent_covariance: torch.Tensor,
    covariance_scale: float = 1.0,
):
    """Gaussian tangent-space NLL and ellipsoid coverage."""
    residual = torch.as_tensor(tangent_residual, dtype=torch.float64)
    covariance = torch.as_tensor(
        tangent_covariance, device=residual.device, dtype=torch.float64
    )
    if residual.ndim != 2 or residual.shape[1] != 3 or covariance.shape != residual.shape[:1] + (3, 3):
        raise ValueError("continuous residuals/covariances must have shapes (N,3) and (N,3,3)")
    if residual.shape[0] < 1 or not bool(torch.isfinite(residual).all()) or not bool(
        torch.isfinite(covariance).all()
    ):
        raise ValueError("continuous calibration arrays must be finite and nonempty")
    if not math.isfinite(float(covariance_scale)) or float(covariance_scale) <= 0.0:
        raise ValueError("continuous covariance scale must be finite and positive")
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    if bool((torch.linalg.eigvalsh(covariance) <= 0.0).any()):
        raise ValueError("continuous calibration covariances must be positive definite")
    scaled = covariance * float(covariance_scale)
    mahalanobis_squared = (
        residual[..., None, :]
        @ torch.linalg.solve(scaled, residual[..., :, None])
    ).squeeze(-1).squeeze(-1)
    nll = 0.5 * (
        3.0 * math.log(2.0 * math.pi)
        + torch.linalg.slogdet(scaled).logabsdet
        + mahalanobis_squared
    )
    return {
        "sample_count": int(residual.shape[0]),
        "dimension": 3,
        "covariance_scale": float(covariance_scale),
        "gaussian_nll": float(nll.mean().item()),
        "mean_mahalanobis_squared": float(mahalanobis_squared.mean().item()),
        "ellipsoid_coverage": {
            str(int(level * 100)): float(
                (mahalanobis_squared <= quantile).to(residual).mean().item()
            )
            for level, quantile in zip(CREDIBLE_LEVELS, _CHI2_DF3_QUANTILES)
        },
        "credible_levels": list(CREDIBLE_LEVELS),
        "coverage_reference": "chi-square with three degrees of freedom",
    }


def _fit_temperature(logits, target, animal_ids, iterations):
    animal_order = tuple(sorted(set(animal_ids)))
    animal_masks = tuple(
        torch.tensor([animal == value for animal in animal_ids], dtype=torch.bool)
        for value in animal_order
    )

    def nll(log_temperature):
        losses = F.cross_entropy(
            logits / math.exp(log_temperature), target, reduction="none"
        )
        return float(torch.stack(tuple(losses[mask].mean() for mask in animal_masks)).mean().item())

    left, right = math.log(0.01), math.log(100.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = nll(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = nll(x2)
    return math.exp(0.5 * (left + right))


def _bootstrap_macro_ci(values):
    value = torch.as_tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(_ANIMAL_BOOTSTRAP_SEED)
    index = torch.randint(
        value.numel(),
        (_ANIMAL_BOOTSTRAP_REPLICATES, value.numel()),
        generator=generator,
    )
    means = value[index].mean(dim=1)
    return {
        "animal_macro_mean": float(value.mean().item()),
        "animal_cluster_bootstrap_95_ci": [
            float(torch.quantile(means, 0.025).item()),
            float(torch.quantile(means, 0.975).item()),
        ],
    }


def _categorical_animal_report(logits, target, animal_ids, temperature):
    animal_order = tuple(sorted(set(animal_ids)))
    per_animal = {}
    for animal in animal_order:
        mask = torch.tensor([value == animal for value in animal_ids], dtype=torch.bool)
        per_animal[animal] = categorical_calibration_metrics_v3(
            logits[mask], target[mask], temperature
        )
    return {
        "animal_count": len(animal_order),
        "per_animal": per_animal,
        "animal_macro_nll": _bootstrap_macro_ci(
            [per_animal[animal]["nll"] for animal in animal_order]
        ),
        "animal_macro_ece": _bootstrap_macro_ci(
            [per_animal[animal]["ece"] for animal in animal_order]
        ),
        "animal_macro_minimum_posterior_set_coverage": {
            str(int(level * 100)): _bootstrap_macro_ci(
                [
                    per_animal[animal]["minimum_posterior_set_coverage"][str(int(level * 100))]
                    for animal in animal_order
                ]
            )
            for level in CREDIBLE_LEVELS
        },
        "interval_scope": "animal-cluster bootstrap of animal-macro metrics",
        "bootstrap_seed": _ANIMAL_BOOTSTRAP_SEED,
        "bootstrap_replicates": _ANIMAL_BOOTSTRAP_REPLICATES,
    }


def _continuous_animal_report(residual, covariance, animal_ids, covariance_scale):
    animal_order = tuple(sorted(set(animal_ids)))
    per_animal = {}
    for animal in animal_order:
        mask = torch.tensor([value == animal for value in animal_ids], dtype=torch.bool)
        per_animal[animal] = continuous_calibration_metrics_v3(
            residual[mask], covariance[mask], covariance_scale
        )
    return {
        "animal_count": len(animal_order),
        "per_animal": per_animal,
        "animal_macro_gaussian_nll": _bootstrap_macro_ci(
            [per_animal[animal]["gaussian_nll"] for animal in animal_order]
        ),
        "animal_macro_mean_mahalanobis_squared": _bootstrap_macro_ci(
            [per_animal[animal]["mean_mahalanobis_squared"] for animal in animal_order]
        ),
        "animal_macro_ellipsoid_coverage": {
            str(int(level * 100)): _bootstrap_macro_ci(
                [
                    per_animal[animal]["ellipsoid_coverage"][str(int(level * 100))]
                    for animal in animal_order
                ]
            )
            for level in CREDIBLE_LEVELS
        },
        "interval_scope": "animal-cluster bootstrap of animal-macro metrics",
        "bootstrap_seed": _ANIMAL_BOOTSTRAP_SEED,
        "bootstrap_replicates": _ANIMAL_BOOTSTRAP_REPLICATES,
    }


def _valid_animal_report(report, animal_ids, *, continuous):
    if (
        not isinstance(report, dict)
        or report.get("animal_count") != len(animal_ids)
        or set(report.get("per_animal", {})) != set(animal_ids)
        or report.get("interval_scope")
        != "animal-cluster bootstrap of animal-macro metrics"
        or report.get("bootstrap_seed") != _ANIMAL_BOOTSTRAP_SEED
        or report.get("bootstrap_replicates") != _ANIMAL_BOOTSTRAP_REPLICATES
    ):
        return False
    names = (
        ("animal_macro_gaussian_nll", "animal_macro_mean_mahalanobis_squared")
        if continuous
        else ("animal_macro_nll", "animal_macro_ece")
    )
    coverage_name = (
        "animal_macro_ellipsoid_coverage"
        if continuous
        else "animal_macro_minimum_posterior_set_coverage"
    )
    coverage = report.get(coverage_name, {})
    if set(coverage) != {str(int(level * 100)) for level in CREDIBLE_LEVELS}:
        return False
    entries = [report.get(name) for name in names]
    entries.extend(coverage.values())
    return len(entries) == len(names) + len(CREDIBLE_LEVELS) and all(
        isinstance(entry, dict)
        and math.isfinite(float(entry.get("animal_macro_mean", float("nan"))))
        and isinstance(entry.get("animal_cluster_bootstrap_95_ci"), list)
        and len(entry["animal_cluster_bootstrap_95_ci"]) == 2
        and all(math.isfinite(float(value)) for value in entry["animal_cluster_bootstrap_95_ci"])
        and entry["animal_cluster_bootstrap_95_ci"][0]
        <= entry["animal_cluster_bootstrap_95_ci"][1]
        for entry in entries
    )


def fit_temperature_on_heldout_animals_v3(
    heldout_logits: torch.Tensor,
    heldout_target_cell_index: torch.Tensor,
    animal_ids,
    heldout_animal_ids,
    final_test_animal_ids,
    catalogue_id: str,
    *,
    training_animal_ids,
    checkpoint_binding_id: str,
    model_state_sha256: str,
    heldout_truth_in_topk_mask=None,
    heldout_topk_catalogue_cell_index=None,
    heldout_refinement_logits=None,
    heldout_target_refined_mode_index=None,
    heldout_continuous_tangent_residual=None,
    heldout_continuous_tangent_covariance=None,
    iterations: int = 96,
):
    """Fit scoped hierarchical calibration on animal-disjoint held-out labels."""
    logits = torch.as_tensor(heldout_logits, dtype=torch.float64).detach().cpu()
    target = torch.as_tensor(heldout_target_cell_index, dtype=torch.long).detach().cpu()
    animals = tuple(str(value) for value in animal_ids)
    calibration_animals = tuple(sorted({str(value) for value in heldout_animal_ids}))
    training_animals = tuple(sorted({str(value) for value in training_animal_ids}))
    final_animals = tuple(sorted({str(value) for value in final_test_animal_ids}))
    if logits.ndim != 2 or target.shape != (logits.shape[0],) or len(animals) != logits.shape[0]:
        raise ValueError("held-out logits, targets, and animal IDs must align")
    split_sets = tuple(map(set, (training_animals, calibration_animals, final_animals)))
    if (
        not animals
        or not all(split_sets)
        or not isinstance(catalogue_id, str)
        or not catalogue_id
        or not isinstance(checkpoint_binding_id, str)
        or len(checkpoint_binding_id) != 64
        or not isinstance(model_state_sha256, str)
        or len(model_state_sha256) != 64
    ):
        raise ValueError("calibration requires immutable model/catalogue bindings and all animal splits")
    if any(set(checkpoint_binding_id.lower()) - set("0123456789abcdef")) or any(
        set(model_state_sha256.lower()) - set("0123456789abcdef")
    ):
        raise ValueError("checkpoint and model-state bindings must be SHA-256 digests")
    if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("training, calibration, and final-test animal sets must be strictly disjoint")
    if not set(animals) <= set(calibration_animals):
        raise ValueError("every supplied label must belong to a held-out calibration animal")
    if not set(calibration_animals) <= set(animals):
        raise ValueError("every declared calibration animal must contribute labels")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 16:
        raise ValueError("temperature search requires at least sixteen iterations")
    categorical_calibration_metrics_v3(logits, target)
    retrieval_temperature = _fit_temperature(logits, target, animals, iterations)

    refinement_supplied = heldout_refinement_logits is not None
    continuous_supplied = (
        heldout_continuous_tangent_residual is not None
        or heldout_continuous_tangent_covariance is not None
    )
    if continuous_supplied and (
        heldout_continuous_tangent_residual is None
        or heldout_continuous_tangent_covariance is None
    ):
        raise ValueError("continuous residuals and covariances must be supplied together")
    if heldout_target_refined_mode_index is not None and not (
        refinement_supplied or continuous_supplied
    ):
        raise ValueError("a refined target is only valid with a conditional calibration head")

    conditional_supplied = bool(refinement_supplied or continuous_supplied)
    truth_in_topk = topk_cell_index = refinement_target = None
    conditional_animals = ()
    conditional_calibration_animals = ()
    eligibility_by_animal = None
    if conditional_supplied:
        if (
            heldout_truth_in_topk_mask is None
            or heldout_topk_catalogue_cell_index is None
            or heldout_target_refined_mode_index is None
        ):
            raise ValueError(
                "conditional calibration requires an explicit truth-in-top-K mask, "
                "top-K catalogue cell IDs, and truth mode targets"
            )
        raw_mask = torch.as_tensor(heldout_truth_in_topk_mask).detach().cpu()
        if raw_mask.dtype != torch.bool or raw_mask.shape != (len(animals),):
            raise ValueError("truth-in-top-K eligibility must be a boolean retrieval-row mask")
        truth_in_topk = raw_mask
        eligible_index = truth_in_topk.nonzero(as_tuple=False).flatten()
        if eligible_index.numel() < 1:
            raise ValueError("conditional calibration requires at least one truth-in-top-K case")
        conditional_animals = tuple(animals[index] for index in eligible_index.tolist())
        conditional_calibration_animals = tuple(sorted(set(conditional_animals)))
        topk_cell_index = torch.as_tensor(
            heldout_topk_catalogue_cell_index, dtype=torch.long
        ).detach().cpu()
        refinement_target = torch.as_tensor(
            heldout_target_refined_mode_index, dtype=torch.long
        ).detach().cpu()
        if (
            topk_cell_index.ndim != 2
            or topk_cell_index.shape[0] != eligible_index.numel()
            or topk_cell_index.shape[1] < 1
            or refinement_target.shape != (eligible_index.numel(),)
            or bool((topk_cell_index < 0).any())
            or bool((topk_cell_index >= logits.shape[1]).any())
            or bool((refinement_target < 0).any())
            or bool((refinement_target >= topk_cell_index.shape[1]).any())
        ):
            raise ValueError("conditional top-K identities and targets have invalid shapes or values")
        if topk_cell_index.shape[1] > 1 and bool(
            (
                topk_cell_index.sort(dim=1).values[:, 1:]
                == topk_cell_index.sort(dim=1).values[:, :-1]
            ).any()
        ):
            raise ValueError("each conditional top-K row must contain unique catalogue cells")
        eligible_reference = target[eligible_index]
        selected_reference = topk_cell_index.gather(
            1, refinement_target[:, None]
        ).squeeze(1)
        if not torch.equal(selected_reference, eligible_reference):
            raise ValueError(
                "each conditional refinement target must identify its reference catalogue cell"
            )
        eligibility_by_animal = {}
        for animal in calibration_animals:
            animal_mask = torch.tensor(
                [value == animal for value in animals], dtype=torch.bool
            )
            eligible = int((animal_mask & truth_in_topk).sum().item())
            total = int(animal_mask.sum().item())
            eligibility_by_animal[animal] = {
                "retrieval_sample_count": total,
                "truth_in_topk_count": eligible,
                "truth_omitted_count": total - eligible,
                "truth_in_topk_rate": eligible / total,
            }

    refinement_logits = None
    refinement_temperature = 1.0
    if refinement_supplied:
        refinement_logits = torch.as_tensor(
            heldout_refinement_logits, dtype=torch.float64
        ).detach().cpu()
        if refinement_logits.shape != topk_cell_index.shape:
            raise ValueError(
                "refinement logits must align only with ordered truth-in-top-K rows and modes"
            )
        categorical_calibration_metrics_v3(refinement_logits, refinement_target)
        refinement_temperature = _fit_temperature(
            refinement_logits, refinement_target, conditional_animals, iterations
        )

    residual = covariance = None
    mode_residual = mode_covariance = None
    covariance_scale = 1.0
    if continuous_supplied:
        mode_residual = torch.as_tensor(
            heldout_continuous_tangent_residual, dtype=torch.float64
        ).detach().cpu()
        mode_covariance = torch.as_tensor(
            heldout_continuous_tangent_covariance, dtype=torch.float64
        ).detach().cpu()
        if (
            mode_residual.shape != topk_cell_index.shape + (3,)
            or mode_covariance.shape != topk_cell_index.shape + (3, 3)
        ):
            raise ValueError(
                "continuous arrays must provide every verified truth-in-top-K mode"
            )
        residual = mode_residual.gather(
            1,
            refinement_target[:, None, None].expand(-1, 1, 3),
        ).squeeze(1)
        covariance = mode_covariance.gather(
            1,
            refinement_target[:, None, None, None].expand(-1, 1, 3, 3),
        ).squeeze(1)
        continuous_calibration_metrics_v3(residual, covariance)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        inverse = torch.linalg.solve(covariance, residual[..., :, None])
        squared = (residual[..., None, :] @ inverse).squeeze(-1).squeeze(-1)
        per_animal = torch.stack(
            tuple(
                squared[
                    torch.tensor(
                        [value == animal for value in conditional_animals],
                        dtype=torch.bool,
                    )
                ].mean()
                for animal in conditional_calibration_animals
            )
        )
        covariance_scale = float((per_animal.mean() / 3.0).clamp_min(1e-12).item())

    fully_calibrated = bool(refinement_supplied and continuous_supplied)
    payload = {
        "schema_version": CALIBRATION_V3_SCHEMA,
        "fit_scope": "animal-disjoint-heldout-calibration-only",
        "label_access_contract": "no final-test animal label is accepted by this fit",
        "catalogue_id": catalogue_id,
        "checkpoint_binding_id": checkpoint_binding_id,
        "checkpoint_binding_scope": "calibration-independent checkpoint digest over architecture/config/catalogue/provenance/training/inference-contract/model-state receipts",
        "model_state_sha256": model_state_sha256,
        "training_animal_ids": list(training_animals),
        "calibration_animal_ids": list(calibration_animals),
        "final_test_animal_ids": list(final_animals),
        "sample_animal_ids": list(animals),
        "sample_count": len(animals),
        "truth_in_topk_mask_receipt": (
            None if truth_in_topk is None else _tensor_receipt(truth_in_topk)
        ),
        "conditional_topk_catalogue_cell_index_receipt": (
            None if topk_cell_index is None else _tensor_receipt(topk_cell_index)
        ),
        "conditional_reference_catalogue_cell_index_receipt": (
            None
            if truth_in_topk is None
            else _tensor_receipt(target[truth_in_topk])
        ),
        "conditional_truth_mode_target_receipt": (
            None if refinement_target is None else _tensor_receipt(refinement_target)
        ),
        "conditional_sample_animal_ids": list(conditional_animals),
        "conditional_calibration_animal_ids": list(conditional_calibration_animals),
        "conditional_sample_count": len(conditional_animals),
        "conditional_omitted_count": len(animals) - len(conditional_animals),
        "truth_in_topk_by_animal": eligibility_by_animal,
        "retrieval_temperature": retrieval_temperature,
        "refinement_temperature": refinement_temperature,
        "continuous_covariance_scale": covariance_scale,
        "retrieval_temperature_scope": "complete-catalogue retrieval logits",
        "refinement_temperature_scope": "conditional-within-retrieved-top-K refinement logits",
        "continuous_covariance_scale_scope": (
            "conditional-on-truth-retrieved global scalar for 3D plane-tangent covariance"
            if continuous_supplied
            else "not calibrated; identity scale retained"
        ),
        "retrieval_categorical_calibration_verified": True,
        "refinement_categorical_calibration_verified": bool(refinement_supplied),
        "continuous_covariance_calibration_verified": bool(continuous_supplied),
        "fully_calibrated": fully_calibrated,
        "calibration_scope": (
            "complete retrieval plus truth-retrieved-conditional refinement and continuous calibration"
            if fully_calibrated
            else "partial calibration; unverified components retain unit temperature/scale"
        ),
        "retrieval_logits_receipt": _tensor_receipt(logits),
        "retrieval_target_receipt": _tensor_receipt(target),
        "retrieval_metrics_before": categorical_calibration_metrics_v3(logits, target),
        "retrieval_metrics_after": categorical_calibration_metrics_v3(
            logits, target, retrieval_temperature
        ),
        "retrieval_animal_metrics_before": _categorical_animal_report(
            logits, target, animals, 1.0
        ),
        "retrieval_animal_metrics_after": _categorical_animal_report(
            logits, target, animals, retrieval_temperature
        ),
        "refinement_logits_receipt": None if refinement_logits is None else _tensor_receipt(refinement_logits),
        "refinement_target_receipt": None if refinement_logits is None else _tensor_receipt(refinement_target),
        "refinement_metrics_before": None if refinement_logits is None else categorical_calibration_metrics_v3(refinement_logits, refinement_target),
        "refinement_metrics_after": None if refinement_logits is None else categorical_calibration_metrics_v3(refinement_logits, refinement_target, refinement_temperature),
        "refinement_animal_metrics_before": None if refinement_logits is None else _categorical_animal_report(refinement_logits, refinement_target, conditional_animals, 1.0),
        "refinement_animal_metrics_after": None if refinement_logits is None else _categorical_animal_report(refinement_logits, refinement_target, conditional_animals, refinement_temperature),
        "continuous_residual_receipt": None if residual is None else _tensor_receipt(residual),
        "continuous_covariance_receipt": None if covariance is None else _tensor_receipt(covariance),
        "continuous_mode_residual_receipt": None if mode_residual is None else _tensor_receipt(mode_residual),
        "continuous_mode_covariance_receipt": None if mode_covariance is None else _tensor_receipt(mode_covariance),
        "continuous_metrics_before": None if residual is None else continuous_calibration_metrics_v3(residual, covariance),
        "continuous_metrics_after": None if residual is None else continuous_calibration_metrics_v3(residual, covariance, covariance_scale),
        "continuous_animal_metrics_before": None if residual is None else _continuous_animal_report(residual, covariance, conditional_animals, 1.0),
        "continuous_animal_metrics_after": None if residual is None else _continuous_animal_report(residual, covariance, conditional_animals, covariance_scale),
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def _valid_tensor_receipt_shape(receipt, shape, dtype):
    return (
        isinstance(receipt, dict)
        and receipt.get("shape") == list(shape)
        and receipt.get("dtype") == dtype
        and isinstance(receipt.get("sha256"), str)
        and len(receipt["sha256"]) == 64
        and not (set(receipt["sha256"].lower()) - set("0123456789abcdef"))
    )


def _valid_conditional_calibration_binding(receipt, calibration_animals, enabled):
    sample_count = receipt.get("sample_count")
    conditional_count = receipt.get("conditional_sample_count")
    omitted_count = receipt.get("conditional_omitted_count")
    conditional_animals = receipt.get("conditional_sample_animal_ids")
    conditional_calibration_animals = receipt.get(
        "conditional_calibration_animal_ids"
    )
    eligibility = receipt.get("truth_in_topk_by_animal")
    if not enabled:
        return (
            conditional_count == 0
            and omitted_count == sample_count
            and conditional_animals == []
            and conditional_calibration_animals == []
            and eligibility is None
            and all(
                receipt.get(name) is None
                for name in (
                    "truth_in_topk_mask_receipt",
                    "conditional_topk_catalogue_cell_index_receipt",
                    "conditional_reference_catalogue_cell_index_receipt",
                    "conditional_truth_mode_target_receipt",
                )
            )
        )
    if (
        not isinstance(sample_count, int)
        or not isinstance(conditional_count, int)
        or conditional_count < 1
        or not isinstance(omitted_count, int)
        or omitted_count != sample_count - conditional_count
        or not isinstance(conditional_animals, list)
        or len(conditional_animals) != conditional_count
        or not isinstance(conditional_calibration_animals, list)
        or conditional_calibration_animals
        != sorted(set(conditional_calibration_animals))
        or set(conditional_calibration_animals) != set(conditional_animals)
        or not set(conditional_calibration_animals) <= set(calibration_animals)
        or not isinstance(eligibility, dict)
        or set(eligibility) != set(calibration_animals)
        or not _valid_tensor_receipt_shape(
            receipt.get("truth_in_topk_mask_receipt"),
            (sample_count,),
            "torch.bool",
        )
    ):
        return False
    total = eligible = omitted = 0
    for animal in calibration_animals:
        entry = eligibility.get(animal)
        if not isinstance(entry, dict):
            return False
        animal_total = entry.get("retrieval_sample_count")
        animal_eligible = entry.get("truth_in_topk_count")
        animal_omitted = entry.get("truth_omitted_count")
        rate = entry.get("truth_in_topk_rate")
        if (
            not isinstance(animal_total, int)
            or animal_total < 1
            or not isinstance(animal_eligible, int)
            or animal_eligible < 0
            or not isinstance(animal_omitted, int)
            or animal_omitted != animal_total - animal_eligible
            or not math.isfinite(float(rate))
            or abs(float(rate) - animal_eligible / animal_total) > 1e-12
            or conditional_animals.count(animal) != animal_eligible
        ):
            return False
        total += animal_total
        eligible += animal_eligible
        omitted += animal_omitted
    topk_receipt = receipt.get("conditional_topk_catalogue_cell_index_receipt")
    reference_receipt = receipt.get(
        "conditional_reference_catalogue_cell_index_receipt"
    )
    target_receipt = receipt.get("conditional_truth_mode_target_receipt")
    if (
        total != sample_count
        or eligible != conditional_count
        or omitted != omitted_count
        or not isinstance(topk_receipt, dict)
        or len(topk_receipt.get("shape", ())) != 2
        or topk_receipt["shape"][0] != conditional_count
        or topk_receipt["shape"][1] < 1
        or not _valid_tensor_receipt_shape(
            topk_receipt,
            tuple(topk_receipt["shape"]),
            "torch.int64",
        )
        or not _valid_tensor_receipt_shape(
            reference_receipt, (conditional_count,), "torch.int64"
        )
        or not _valid_tensor_receipt_shape(
            target_receipt, (conditional_count,), "torch.int64"
        )
    ):
        return False
    return True


def verify_temperature_calibration_receipt_v3(
    receipt,
    catalogue_id: str,
    *,
    checkpoint_binding_id: str | None = None,
    model_state_sha256: str | None = None,
):
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    training = set(receipt.get("training_animal_ids", ()))
    calibration = set(receipt.get("calibration_animal_ids", ()))
    final = set(receipt.get("final_test_animal_ids", ()))
    split_sets = (training, calibration, final)
    refinement_verified = receipt.get("refinement_categorical_calibration_verified") is True
    continuous_verified = receipt.get("continuous_covariance_calibration_verified") is True
    conditional_verified = refinement_verified or continuous_verified
    conditional_calibration = set(
        receipt.get("conditional_calibration_animal_ids", ())
    )
    valid = (
        isinstance(receipt, dict)
        and receipt.get("schema_version") == CALIBRATION_V3_SCHEMA
        and receipt.get("fit_scope") == "animal-disjoint-heldout-calibration-only"
        and receipt.get("catalogue_id") == catalogue_id
        and all(split_sets)
        and all(
            len(receipt.get(name, ())) == len(set(receipt.get(name, ())))
            for name in (
                "training_animal_ids",
                "calibration_animal_ids",
                "final_test_animal_ids",
            )
        )
        and not any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3))
        and set(receipt.get("sample_animal_ids", ())) == calibration
        and receipt.get("sample_count") == len(receipt.get("sample_animal_ids", ()))
        and receipt.get("retrieval_categorical_calibration_verified") is True
        and math.isfinite(float(receipt.get("retrieval_temperature", float("nan"))))
        and float(receipt.get("retrieval_temperature", 0.0)) > 0.0
        and math.isfinite(float(receipt.get("refinement_temperature", float("nan"))))
        and float(receipt.get("refinement_temperature", 0.0)) > 0.0
        and math.isfinite(float(receipt.get("continuous_covariance_scale", float("nan"))))
        and float(receipt.get("continuous_covariance_scale", 0.0)) > 0.0
        and receipt.get("retrieval_temperature_scope")
        == "complete-catalogue retrieval logits"
        and receipt.get("refinement_temperature_scope")
        == "conditional-within-retrieved-top-K refinement logits"
        and receipt.get("continuous_covariance_scale_scope")
        == (
            "conditional-on-truth-retrieved global scalar for 3D plane-tangent covariance"
            if continuous_verified
            else "not calibrated; identity scale retained"
        )
        and receipt.get("fully_calibrated") is (refinement_verified and continuous_verified)
        and receipt.get("calibration_scope")
        == (
            "complete retrieval plus truth-retrieved-conditional refinement and continuous calibration"
            if refinement_verified and continuous_verified
            else "partial calibration; unverified components retain unit temperature/scale"
        )
        and _valid_conditional_calibration_binding(
            receipt, calibration, conditional_verified
        )
        and _valid_animal_report(
            receipt.get("retrieval_animal_metrics_before"), calibration, continuous=False
        )
        and _valid_animal_report(
            receipt.get("retrieval_animal_metrics_after"), calibration, continuous=False
        )
        and (
            refinement_verified
            and _valid_animal_report(
                receipt.get("refinement_animal_metrics_before"),
                conditional_calibration,
                continuous=False,
            )
            and _valid_animal_report(
                receipt.get("refinement_animal_metrics_after"),
                conditional_calibration,
                continuous=False,
            )
            and _valid_tensor_receipt_shape(
                receipt.get("refinement_logits_receipt"),
                tuple(
                    receipt["conditional_topk_catalogue_cell_index_receipt"][
                        "shape"
                    ]
                ),
                "torch.float64",
            )
            and _valid_tensor_receipt_shape(
                receipt.get("refinement_target_receipt"),
                (receipt["conditional_sample_count"],),
                "torch.int64",
            )
            or not refinement_verified
            and receipt.get("refinement_temperature") == 1.0
            and all(
                receipt.get(name) is None
                for name in (
                    "refinement_logits_receipt",
                    "refinement_target_receipt",
                    "refinement_metrics_before",
                    "refinement_metrics_after",
                    "refinement_animal_metrics_before",
                    "refinement_animal_metrics_after",
                )
            )
        )
        and (
            continuous_verified
            and _valid_animal_report(
                receipt.get("continuous_animal_metrics_before"),
                conditional_calibration,
                continuous=True,
            )
            and _valid_animal_report(
                receipt.get("continuous_animal_metrics_after"),
                conditional_calibration,
                continuous=True,
            )
            and _valid_tensor_receipt_shape(
                receipt.get("continuous_residual_receipt"),
                (receipt["conditional_sample_count"], 3),
                "torch.float64",
            )
            and _valid_tensor_receipt_shape(
                receipt.get("continuous_covariance_receipt"),
                (receipt["conditional_sample_count"], 3, 3),
                "torch.float64",
            )
            and _valid_tensor_receipt_shape(
                receipt.get("continuous_mode_residual_receipt"),
                tuple(
                    receipt["conditional_topk_catalogue_cell_index_receipt"][
                        "shape"
                    ]
                )
                + (3,),
                "torch.float64",
            )
            and _valid_tensor_receipt_shape(
                receipt.get("continuous_mode_covariance_receipt"),
                tuple(
                    receipt["conditional_topk_catalogue_cell_index_receipt"][
                        "shape"
                    ]
                )
                + (3, 3),
                "torch.float64",
            )
            or not continuous_verified
            and receipt.get("continuous_covariance_scale") == 1.0
            and all(
                receipt.get(name) is None
                for name in (
                    "continuous_residual_receipt",
                    "continuous_covariance_receipt",
                    "continuous_mode_residual_receipt",
                    "continuous_mode_covariance_receipt",
                    "continuous_metrics_before",
                    "continuous_metrics_after",
                    "continuous_animal_metrics_before",
                    "continuous_animal_metrics_after",
                )
            )
        )
        and isinstance(receipt.get("checkpoint_binding_id"), str)
        and receipt.get("checkpoint_binding_scope")
        == "calibration-independent checkpoint digest over architecture/config/catalogue/provenance/training/inference-contract/model-state receipts"
        and len(receipt["checkpoint_binding_id"]) == 64
        and not (set(receipt["checkpoint_binding_id"].lower()) - set("0123456789abcdef"))
        and isinstance(receipt.get("model_state_sha256"), str)
        and len(receipt["model_state_sha256"]) == 64
        and not (set(receipt["model_state_sha256"].lower()) - set("0123456789abcdef"))
        and (checkpoint_binding_id is None or receipt.get("checkpoint_binding_id") == checkpoint_binding_id)
        and (model_state_sha256 is None or receipt.get("model_state_sha256") == model_state_sha256)
        and receipt.get("receipt_sha256") == _sha(payload)
    )
    if not valid:
        raise ValueError("temperature calibration receipt is invalid or mismatched")
    return True


def _catalogue_value(catalogue, name, *, device, dtype=None):
    source = catalogue.get("tensors", {}).get(name)
    if source is None:
        source = catalogue["arrays"][
            {
                "cell_id": "cell_id_int64",
                "cell_states": "cell_states_float64",
            }[name]
        ]
    value = torch.as_tensor(source, device=device)
    return value if dtype is None else value.to(dtype=dtype)


def _weighted_quantile(values, weights, levels=CREDIBLE_LEVELS):
    """Weighted quantiles along the final dimension."""
    values, weights = torch.broadcast_tensors(values, weights)
    order = values.argsort(dim=-1, stable=True)
    sorted_value = values.gather(-1, order)
    sorted_weight = weights.gather(-1, order)
    cumulative = sorted_weight.cumsum(dim=-1)
    cumulative = cumulative / cumulative[..., -1:].clamp_min(torch.finfo(weights.dtype).tiny)
    result = []
    for level in levels:
        index = (cumulative < float(level)).sum(dim=-1).clamp_max(values.shape[-1] - 1)
        result.append(sorted_value.gather(-1, index[..., None]).squeeze(-1))
    return torch.stack(result, dim=-1)


def _gauss_hermite_grid(order, *, device, dtype):
    if not isinstance(order, int) or isinstance(order, bool) or order < 3 or order > 9:
        raise ValueError("Gauss-Hermite order must be an integer in [3,9]")
    node, weight = np.polynomial.hermite.hermgauss(order)
    node = torch.as_tensor(node * math.sqrt(2.0), device=device, dtype=dtype)
    weight = torch.as_tensor(weight / math.sqrt(math.pi), device=device, dtype=dtype)
    node_grid = torch.stack(torch.meshgrid(node, node, node, indexing="ij"), dim=-1).reshape(-1, 3)
    weight_grid = (
        weight[:, None, None] * weight[None, :, None] * weight[None, None, :]
    ).reshape(-1)
    return node_grid, weight_grid


def posterior_summary_v3(
    joint_output,
    catalogue,
    support_origin_ap_dv_ml_um,
    *,
    calibration_receipt=None,
    checkpoint_binding_id=None,
    model_state_sha256=None,
    failure_omitted_mass_threshold: float = 0.35,
    gauss_hermite_order: int = 5,
):
    """Build a truncated hierarchical mixture with an exact retrieval tail mass."""
    if (
        not math.isfinite(float(failure_omitted_mass_threshold))
        or not 0.0 <= float(failure_omitted_mass_threshold) <= 1.0
    ):
        raise ValueError("omitted-mass failure threshold must lie in [0,1]")
    pose = joint_output["pose"]
    log_probability = torch.as_tensor(pose["retrieval_cell_log_probability"])
    if log_probability.ndim != 2 or not bool(torch.isfinite(log_probability).all()):
        raise ValueError("complete retrieval log posterior must have shape (B,N)")
    normalization_tolerance = 64.0 * torch.finfo(log_probability.dtype).eps
    if (
        pose.get("catalogue_complete") is not True
        or pose.get("retrieval_tail_scope") != RETRIEVAL_TAIL_SCOPE
        or not torch.allclose(
            torch.logsumexp(log_probability, dim=1),
            torch.zeros(log_probability.shape[0], device=log_probability.device, dtype=log_probability.dtype),
            atol=normalization_tolerance,
            rtol=0.0,
        )
    ):
        raise ValueError("retrieval output is not an explicitly normalized complete-catalogue posterior")
    batch, count = log_probability.shape
    catalogue_id = catalogue.get("catalogue_id")
    if not isinstance(catalogue_id, str) or not catalogue_id:
        raise ValueError("catalogue must carry its immutable ID")
    cell_id = _catalogue_value(catalogue, "cell_id", device=log_probability.device)
    states = _catalogue_value(
        catalogue, "cell_states", device=log_probability.device, dtype=log_probability.dtype
    )
    states = states[0] if states.ndim == 3 and states.shape[0] == 1 else states
    if cell_id.shape != (count,) or states.shape != (count, 12):
        raise ValueError("catalogue IDs/states disagree with complete retrieval posterior")
    if not torch.equal(torch.as_tensor(pose["retrieval_cell_id"], device=cell_id.device), cell_id):
        raise ValueError("retrieval and catalogue cell IDs do not match exactly")

    calibration_applied = calibration_receipt is not None
    retrieval_temperature = 1.0
    refinement_temperature = 1.0
    covariance_scale = 1.0
    retrieval_calibrated = refinement_calibrated = continuous_calibrated = False
    fully_calibrated = False
    if calibration_applied:
        if checkpoint_binding_id is None or model_state_sha256 is None:
            raise ValueError("calibration application requires its checkpoint/model-state binding")
        verify_temperature_calibration_receipt_v3(
            calibration_receipt,
            catalogue_id,
            checkpoint_binding_id=checkpoint_binding_id,
            model_state_sha256=model_state_sha256,
        )
        retrieval_temperature = float(calibration_receipt["retrieval_temperature"])
        retrieval_calibrated = calibration_receipt[
            "retrieval_categorical_calibration_verified"
        ]
        refinement_calibrated = calibration_receipt[
            "refinement_categorical_calibration_verified"
        ]
        continuous_calibrated = calibration_receipt[
            "continuous_covariance_calibration_verified"
        ]
        if refinement_calibrated:
            refinement_temperature = float(
                calibration_receipt["refinement_temperature"]
            )
        if continuous_calibrated:
            covariance_scale = float(
                calibration_receipt["continuous_covariance_scale"]
            )
        fully_calibrated = calibration_receipt["fully_calibrated"]
    probability = torch.softmax(log_probability / retrieval_temperature, dim=1)
    raw_probability = log_probability.exp()
    if pose.get("retrieval_cell_probability") is None:
        raise ValueError("complete retrieval output must include its probability tensor")
    reported_probability = torch.as_tensor(
        pose["retrieval_cell_probability"],
        device=raw_probability.device,
        dtype=raw_probability.dtype,
    )
    if reported_probability.shape != raw_probability.shape or not torch.allclose(
        reported_probability,
        raw_probability,
        atol=normalization_tolerance,
        rtol=normalization_tolerance,
    ):
        raise ValueError("reported retrieval probabilities disagree with normalized log probabilities")
    top_index = torch.as_tensor(
        pose["retrieval_topk_catalogue_index"], device=probability.device, dtype=torch.long
    )
    if top_index.ndim != 2 or top_index.shape[0] != batch:
        raise ValueError("top-K catalogue indices must have shape (B,K)")
    if bool(((top_index < 0) | (top_index >= count)).any()) or (
        top_index.shape[1] > 1
        and bool((top_index.sort(dim=1).values[:, 1:] == top_index.sort(dim=1).values[:, :-1]).any())
    ):
        raise ValueError("top-K catalogue indices must be in range and unique per batch")
    top_id = torch.as_tensor(pose["retrieval_topk_cell_id"], device=cell_id.device)
    if not torch.equal(top_id, cell_id[top_index]):
        raise ValueError("top-K IDs do not bind to catalogue indices")
    selected = torch.zeros_like(probability, dtype=torch.bool).scatter(1, top_index, True)
    raw_omitted = raw_probability.masked_fill(selected, 0.0).sum(dim=1)
    reported_omitted = torch.as_tensor(
        pose["retrieval_omitted_probability"], device=probability.device, dtype=probability.dtype
    )
    if reported_omitted.shape != (batch,) or not bool(torch.isfinite(reported_omitted).all()):
        raise ValueError("reported omitted mass must align with posterior batches")
    if not torch.allclose(raw_omitted, reported_omitted, atol=2e-6, rtol=2e-6):
        raise ValueError("reported omitted mass is not the exact complete-catalogue tail")

    omitted_weight = probability.masked_fill(selected, 0.0)
    retained = probability.masked_fill(~selected, 0.0).sum(dim=1)
    omitted = omitted_weight.sum(dim=1)
    refined_state = torch.as_tensor(pose["final_cell_state"], device=probability.device)
    raw_covariance = torch.as_tensor(
        pose["final_cell_canonical_plane_covariance"], device=probability.device
    ).to(refined_state)
    if refined_state.shape != top_index.shape + (12,) or raw_covariance.shape != top_index.shape + (3, 3):
        raise ValueError("top-K refined states/covariances do not align")
    if not bool(torch.isfinite(refined_state).all()) or not bool(
        torch.isfinite(raw_covariance).all()
    ):
        raise ValueError("refined states/covariances must be finite")
    covariance = raw_covariance * covariance_scale
    conditional_log_probability = torch.as_tensor(
        pose["conditional_within_topk_cell_log_probability"], device=probability.device
    )
    if conditional_log_probability.shape != top_index.shape or not bool(
        torch.isfinite(conditional_log_probability).all()
    ):
        raise ValueError("refinement log probabilities must align with top-K modes")
    if not torch.allclose(
        torch.logsumexp(conditional_log_probability, dim=1),
        torch.zeros(batch, device=probability.device, dtype=probability.dtype),
        atol=normalization_tolerance,
        rtol=0.0,
    ):
        raise ValueError("conditional refinement log probabilities must be normalized")
    mode_probability = retained[:, None] * torch.softmax(
        conditional_log_probability / refinement_temperature, dim=1
    )

    symmetric_covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalue = torch.linalg.eigvalsh(symmetric_covariance)
    covariance_scale_reference = symmetric_covariance.abs().amax(dim=(-2, -1)).clamp_min(1.0)
    covariance_negative_tolerance = (
        128.0 * torch.finfo(symmetric_covariance.dtype).eps * covariance_scale_reference
    )
    minimum_eigenvalue = eigenvalue[..., 0]
    if bool((minimum_eigenvalue < -covariance_negative_tolerance).any()):
        raise ValueError("refined plane covariance is materially non-positive-semidefinite")
    covariance_roundoff_jitter = (-minimum_eigenvalue).clamp_min(0.0)
    symmetric_covariance = symmetric_covariance + covariance_roundoff_jitter[..., None, None] * torch.eye(
        3, device=covariance.device, dtype=covariance.dtype
    )
    covariance = symmetric_covariance
    eigenvalue, eigenvector = torch.linalg.eigh(symmetric_covariance)
    columns = eigenvector * eigenvalue.clamp_min(0.0).sqrt()[..., None, :]
    quadrature_node, quadrature_base_weight = _gauss_hermite_grid(
        gauss_hermite_order, device=covariance.device, dtype=covariance.dtype
    )
    delta = torch.einsum("...ij,qj->...qi", columns, quadrature_node)
    residual = torch.zeros(
        *delta.shape[:-1], 9, device=delta.device, dtype=delta.dtype
    )
    residual[..., :3] = delta
    quadrature_state = compose_antipodal_plane_frame_residual(
        refined_state[..., None, :].expand(*delta.shape[:-1], 12).reshape(-1, 12),
        residual.reshape(-1, 9),
        support_origin_ap_dv_ml_um,
    ).reshape(*delta.shape[:-1], 12)
    quadrature_weight = mode_probability[..., None] * quadrature_base_weight

    support_state = torch.cat(
        (
            states[None].expand(batch, -1, -1),
            quadrature_state.reshape(batch, -1, 12),
        ),
        dim=1,
    )
    support_weight = torch.cat((omitted_weight, quadrature_weight.reshape(batch, -1)), dim=1)
    quadrature_count = gauss_hermite_order ** 3
    support_cell_id = torch.cat(
        (
            cell_id[None].expand(batch, -1),
            top_id[..., None]
            .expand(-1, -1, quadrature_count)
            .reshape(batch, -1),
        ),
        dim=1,
    )
    quadrature_top_index = torch.arange(
        top_index.shape[1], device=top_index.device, dtype=torch.long
    )[None, :, None].expand(batch, -1, quadrature_count).reshape(batch, -1)
    support_topk_index = torch.cat(
        (
            torch.full(
                (batch, count), -1, device=top_index.device, dtype=torch.long
            ),
            quadrature_top_index,
        ),
        dim=1,
    )
    if not torch.allclose(
        support_weight.sum(dim=1), torch.ones(batch, device=probability.device, dtype=probability.dtype),
        atol=2e-6,
        rtol=2e-6,
    ):
        raise RuntimeError("posterior support does not preserve unit mass")

    row = torch.arange(batch, device=probability.device)
    component_state = torch.cat(
        (states[None].expand(batch, -1, -1), refined_state), dim=1
    )
    component_mass = torch.cat((omitted_weight, mode_probability), dim=1)
    component_cell_id = torch.cat((cell_id[None].expand(batch, -1), top_id), dim=1)
    component_topk_index = torch.cat(
        (
            torch.full(
                (batch, count), -1, device=top_index.device, dtype=torch.long
            ),
            torch.arange(
                top_index.shape[1], device=top_index.device, dtype=torch.long
            )[None].expand(batch, -1),
        ),
        dim=1,
    )
    map_component_index = component_mass.argmax(dim=1)
    map_state = component_state[row, map_component_index]
    map_center, map_frame, _ = full_frame_state_to_components(map_state)
    map_normal = map_frame[..., :, 2]

    center, frame, _ = full_frame_state_to_components(support_state)
    normal = frame[..., :, 2]
    sign = torch.where(
        (normal * map_normal[:, None]).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(normal[..., :1]),
        torch.ones_like(normal[..., :1]),
    )
    aligned_normal = normal * sign
    scatter = (
        support_weight[..., None, None]
        * aligned_normal[..., :, None]
        * aligned_normal[..., None, :]
    ).sum(dim=1)
    _, projective_vector = torch.linalg.eigh(scatter)
    mean_normal = projective_vector[..., -1]
    mean_normal = mean_normal * torch.where(
        (mean_normal * map_normal).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(mean_normal[..., :1]),
        torch.ones_like(mean_normal[..., :1]),
    )
    support_origin = torch.as_tensor(
        support_origin_ap_dv_ml_um, device=center.device, dtype=center.dtype
    )
    if support_origin.shape != (3,) or not bool(torch.isfinite(support_origin).all()):
        raise ValueError("support origin must be one finite AP-DV-ML point")
    signed_offset = ((center - support_origin) * aligned_normal).sum(dim=-1)
    map_signed_offset = ((map_center - support_origin) * map_normal).sum(dim=-1)
    center_ap = center[..., 0]
    tilt_dv = torch.atan2(aligned_normal[..., 1], aligned_normal[..., 0])
    tilt_ml = torch.atan2(
        aligned_normal[..., 2], torch.linalg.vector_norm(aligned_normal[..., :2], dim=-1)
    )
    lower_levels = tuple((1.0 - level) / 2.0 for level in CREDIBLE_LEVELS)
    upper_levels = tuple(1.0 - value for value in lower_levels)
    signed_offset_interval = torch.stack(
        (
            _weighted_quantile(signed_offset, support_weight, lower_levels),
            _weighted_quantile(signed_offset, support_weight, upper_levels),
        ),
        dim=-1,
    )
    center_ap_interval = torch.stack(
        (
            _weighted_quantile(center_ap, support_weight, lower_levels),
            _weighted_quantile(center_ap, support_weight, upper_levels),
        ),
        dim=-1,
    )
    map_tilt_dv = torch.atan2(map_normal[..., 1], map_normal[..., 0])
    tilt_dv_residual = torch.atan2(
        torch.sin(tilt_dv - map_tilt_dv[:, None]),
        torch.cos(tilt_dv - map_tilt_dv[:, None]),
    )
    tilt_dv_interval = map_tilt_dv[:, None, None] + torch.stack(
        (
            _weighted_quantile(tilt_dv_residual, support_weight, lower_levels),
            _weighted_quantile(tilt_dv_residual, support_weight, upper_levels),
        ),
        dim=-1,
    )
    tilt_ml_interval = torch.stack(
        (_weighted_quantile(tilt_ml, support_weight, lower_levels), _weighted_quantile(tilt_ml, support_weight, upper_levels)),
        dim=-1,
    )
    angular_distance = torch.acos(
        (normal * map_normal[:, None]).sum(dim=-1).abs().clamp(0.0, 1.0)
    )
    normal_radius = _weighted_quantile(angular_distance, support_weight)
    sorted_weight, support_order = support_weight.sort(dim=1, descending=True, stable=True)
    cumulative = sorted_weight.cumsum(dim=1)
    quadrature_mass_set_masks = []
    for level in CREDIBLE_LEVELS:
        rank_mask = cumulative - sorted_weight < level
        quadrature_mass_set_masks.append(
            torch.zeros_like(rank_mask).scatter(1, support_order, rank_mask)
        )
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(max(count, 2))
    failure = omitted > float(failure_omitted_mass_threshold)

    return {
        "schema_version": UNCERTAINTY_V3_SCHEMA,
        "catalogue_id": catalogue_id,
        "posterior_scope": "hierarchical/truncated top-K refinement with exact complete-catalogue retrieval tail mass",
        "posterior_exact": False,
        "posterior_approximation": "continuous refined modes use deterministic tensor-product Gauss-Hermite quadrature; omitted cells retain catalogue mean states without local refinement",
        "retrieval_tail_mass_exact": True,
        "calibration_applied": bool(calibration_applied),
        "retrieval_probabilities_calibrated": bool(retrieval_calibrated),
        "refinement_probabilities_calibrated": bool(refinement_calibrated),
        "continuous_covariance_calibrated": bool(continuous_calibrated),
        "probabilities_calibrated": bool(
            retrieval_calibrated and refinement_calibrated
        ),
        "fully_calibrated": bool(fully_calibrated),
        "calibration_status": (
            "fully_calibrated"
            if fully_calibrated
            else "partially_calibrated"
            if calibration_applied
            else "uncalibrated"
        ),
        "calibration_receipt_sha256": None if not calibration_applied else calibration_receipt["receipt_sha256"],
        "retrieval_temperature": retrieval_temperature,
        "refinement_temperature": refinement_temperature,
        "continuous_covariance_scale": covariance_scale,
        "retrieval_temperature_scope": "complete-catalogue retrieval logits",
        "refinement_temperature_scope": "conditional-within-retrieved-top-K refinement logits",
        "continuous_covariance_scale_scope": "one global scalar multiplier for every 3D plane-tangent covariance"
        if continuous_calibrated
        else "not calibrated; identity scale retained",
        "retrieval_catalogue_cell_id": cell_id,
        "raw_complete_retrieval_probability": raw_probability,
        "complete_retrieval_probability": probability,
        "raw_exact_omitted_probability": reported_omitted,
        "exact_complete_retrieval_tail_probability": omitted,
        "retained_probability": retained,
        "omitted_probability": omitted,
        "retrieval_tail_scope": RETRIEVAL_TAIL_SCOPE,
        "refined_mode_state": refined_state,
        "refined_mode_probability": mode_probability,
        "refined_mode_raw_plane_tangent_covariance": raw_covariance,
        "refined_mode_plane_tangent_covariance": covariance,
        "refined_mode_covariance_roundoff_negative_tolerance": covariance_negative_tolerance,
        "refined_mode_covariance_roundoff_diagonal_jitter": covariance_roundoff_jitter,
        "materially_non_psd_refined_covariance_rejected": True,
        "gauss_hermite_order_per_dimension": gauss_hermite_order,
        "gauss_hermite_node_count_per_mode": quadrature_count,
        "refined_mode_gauss_hermite_state": quadrature_state,
        "refined_mode_gauss_hermite_weight": quadrature_weight,
        "posterior_component_mean_state": component_state,
        "posterior_component_mass": component_mass,
        "posterior_component_cell_id": component_cell_id,
        "posterior_component_topk_index": component_topk_index,
        "posterior_support_state": support_state,
        "posterior_support_weight": support_weight,
        "posterior_support_semantics": "deterministic integration support, not posterior atoms and not the MAP definition",
        "posterior_support_cell_id": support_cell_id,
        "posterior_support_topk_index": support_topk_index,
        "posterior_support_center_ap_dv_ml_um": center,
        "posterior_support_plane_normal_ap_dv_ml": normal,
        "posterior_support_antipodally_aligned_plane_normal_ap_dv_ml": aligned_normal,
        "posterior_support_signed_plane_offset_um": signed_offset,
        "posterior_support_center_ap_um": center_ap,
        "posterior_support_tilt_dv_from_ap_rad": tilt_dv,
        "posterior_support_tilt_ml_from_apdv_rad": tilt_ml,
        "credible_levels": torch.tensor(CREDIBLE_LEVELS, device=probability.device, dtype=probability.dtype),
        "quadrature_minimum_mass_support_mask": torch.stack(
            quadrature_mass_set_masks, dim=1
        ),
        "quadrature_minimum_mass_support_scope": "ranked quadrature contributions for the hierarchical/truncated posterior; not a highest-density region",
        "signed_plane_offset_central_credible_interval_um": signed_offset_interval,
        "center_ap_central_credible_interval_um": center_ap_interval,
        "tilt_dv_central_credible_interval_rad": tilt_dv_interval,
        "tilt_ml_central_credible_interval_rad": tilt_ml_interval,
        "plane_normal_map_axis_ap_dv_ml": map_normal,
        "plane_normal_projective_credible_radius_rad": normal_radius,
        "retrieval_entropy_nats": entropy,
        "normalized_retrieval_entropy": normalized_entropy,
        "failure_flag": failure,
        "failure_reason": "exact_omitted_mass_above_threshold",
        "ambiguity_flag": normalized_entropy > 0.75,
        "point_estimate": {
            "summary_semantics": "posterior means are quadrature summaries; MAP selects the largest mixture component mass and returns that component mean state; the hierarchical/truncated mixture remains authoritative",
            "posterior_mean_center_ap_dv_ml_um": (support_weight[..., None] * center).sum(dim=1),
            "posterior_mean_center_ap_um": (support_weight * center_ap).sum(dim=1),
            "posterior_mean_signed_plane_offset_um": (
                support_weight * signed_offset
            ).sum(dim=1),
            "posterior_projective_mean_plane_normal_ap_dv_ml": mean_normal,
            "map_component_mean_state": map_state,
            "map_component_mass": component_mass[row, map_component_index],
            "map_center_ap_dv_ml_um": map_center,
            "map_center_ap_um": map_center[..., 0],
            "map_signed_plane_offset_um": map_signed_offset,
            "map_catalogue_cell_id": component_cell_id[row, map_component_index],
            "map_component_topk_index": component_topk_index[
                row, map_component_index
            ],
        },
    }


def posterior_coverage_metrics_v3(
    summary, reference_signed_plane_offset_um, reference_normal_ap_dv_ml
):
    """Coverage of signed plane-offset and projective-normal credible regions."""
    interval = torch.as_tensor(
        summary["signed_plane_offset_central_credible_interval_um"]
    )
    reference_offset = torch.as_tensor(
        reference_signed_plane_offset_um, device=interval.device, dtype=interval.dtype
    )
    reference_normal = torch.as_tensor(
        reference_normal_ap_dv_ml, device=interval.device, dtype=interval.dtype
    )
    if reference_offset.shape != interval.shape[:1] or reference_normal.shape != interval.shape[:1] + (3,):
        raise ValueError("references must align one-to-one with posterior batches")
    reference_norm = torch.linalg.vector_norm(reference_normal, dim=-1, keepdim=True)
    if (
        not bool(torch.isfinite(reference_offset).all())
        or not bool(torch.isfinite(reference_norm).all())
        or bool((reference_norm <= 0.0).any())
    ):
        raise ValueError("reference offsets and nonzero normals must be finite")
    reference_normal = reference_normal / reference_norm
    map_normal = torch.as_tensor(
        summary["plane_normal_map_axis_ap_dv_ml"],
        device=interval.device,
        dtype=interval.dtype,
    )
    signed_dot = (reference_normal * map_normal).sum(dim=-1)
    reference_offset = reference_offset * torch.where(
        signed_dot < 0.0, -torch.ones_like(signed_dot), torch.ones_like(signed_dot)
    )
    angle = torch.acos(signed_dot.abs().clamp(0.0, 1.0))
    radius = summary["plane_normal_projective_credible_radius_rad"]
    result = {"sample_count": int(reference_offset.numel()), "credible_levels": list(CREDIBLE_LEVELS)}
    result["signed_plane_offset_coverage"] = {
        str(int(level * 100)): float(
            (
                (reference_offset >= interval[:, index, 0])
                & (reference_offset <= interval[:, index, 1])
            )
            .float()
            .mean()
            .item()
        )
        for index, level in enumerate(CREDIBLE_LEVELS)
    }
    result["plane_normal_coverage"] = {
        str(int(level * 100)): float((angle <= radius[:, index]).float().mean().item())
        for index, level in enumerate(CREDIBLE_LEVELS)
    }
    return result


def propagate_electrode_trajectory_v3(
    summary,
    joint_output,
    electrode_points_yx_px,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    *,
    annotation_volume_ap_dv_ml=None,
    atlas_shape_ap_dv_ml=None,
    minimum_deformation_jacobian: float = 0.05,
    maximum_cycle_error_px: float = 1.0,
):
    """Propagate the deterministic per-mode SVF approximation to a trajectory tube."""
    states = torch.as_tensor(summary["posterior_support_state"])
    weights = torch.as_tensor(summary["posterior_support_weight"], device=states.device, dtype=states.dtype)
    top_index = torch.as_tensor(summary["posterior_support_topk_index"], device=states.device)
    points = torch.as_tensor(electrode_points_yx_px, device=states.device, dtype=states.dtype)
    if points.ndim == 2:
        points = points[None].expand(states.shape[0], -1, -1)
    if points.shape[0] != states.shape[0] or points.shape[-1] != 2 or not bool(
        torch.isfinite(points).all()
    ):
        raise ValueError("electrode points must have shape (B,P,2) in y-x order")
    pullback_source = (
        joint_output["final_pullback_map_yx_px"]
        if "final_pullback_map_yx_px" in joint_output
        else joint_output["final_forward_map_yx_px"]
    )
    pullback = torch.as_tensor(
        pullback_source,
        device=states.device,
    ).to(states)
    if pullback.ndim != 5 or pullback.shape[2] != 2 or not bool(
        torch.isfinite(pullback).all()
    ):
        raise ValueError("deformation maps must have finite shape (B,K,2,H,W)")
    batch, top_k, _, height, width = pullback.shape
    if batch != states.shape[0] or height < 2 or width < 2:
        raise ValueError("deformation maps and posterior batches disagree")
    if (
        not math.isfinite(float(minimum_deformation_jacobian))
        or float(minimum_deformation_jacobian) <= 0.0
        or not math.isfinite(float(maximum_cycle_error_px))
        or float(maximum_cycle_error_px) <= 0.0
    ):
        raise ValueError("deformation validity thresholds must be finite and positive")
    if top_index.shape != states.shape[:2] or bool(
        ((top_index < -1) | (top_index >= top_k)).any()
    ):
        raise ValueError("posterior top-K indices must be exactly -1 or valid mode indices")
    y, x = points.unbind(dim=-1)
    input_raster_valid = (y >= 0.0) & (y <= height - 1) & (x >= 0.0) & (x <= width - 1)
    query = torch.stack((2.0 * x / (width - 1) - 1.0, 2.0 * y / (height - 1) - 1.0), dim=-1)
    sampled = F.grid_sample(
        pullback.reshape(batch * top_k, 2, height, width),
        query[:, None].expand(-1, top_k, -1, -1).reshape(batch * top_k, 1, -1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, :, 0].transpose(1, 2).reshape(batch, top_k, points.shape[1], 2)

    observed_mode_points = points[:, None].expand(-1, top_k, -1, -1)
    canonical_mode_y, canonical_mode_x = sampled.unbind(dim=-1)
    canonical_mode_in_bounds = (
        torch.isfinite(canonical_mode_y)
        & torch.isfinite(canonical_mode_x)
        & (canonical_mode_y >= 0.0)
        & (canonical_mode_y <= height - 1)
        & (canonical_mode_x >= 0.0)
        & (canonical_mode_x <= width - 1)
    )

    def sample_mode_field(name, channels, query_yx_px, mode="bilinear"):
        if name not in joint_output:
            raise ValueError(f"joint output lacks required deformation diagnostic {name}")
        field = torch.as_tensor(joint_output[name], device=states.device).to(states)
        if channels == 1 and field.ndim == 4:
            field = field[:, :, None]
        if field.shape != (batch, top_k, channels, height, width):
            raise ValueError(f"deformation diagnostic {name} has the wrong shape")
        field_y, field_x = query_yx_px.unbind(dim=-1)
        field_query = torch.stack(
            (
                2.0 * field_x / (width - 1) - 1.0,
                2.0 * field_y / (height - 1) - 1.0,
            ),
            dim=-1,
        )
        return F.grid_sample(
            field.reshape(batch * top_k, channels, height, width),
            field_query.reshape(batch * top_k, 1, -1, 2),
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )[:, :, 0].transpose(1, 2).reshape(
            batch, top_k, points.shape[1], channels
        )

    sampled_jacobian = sample_mode_field(
        "final_forward_jacobian_determinant", 1, observed_mode_points
    ).squeeze(-1)
    sampled_forward_cycle_valid = sample_mode_field(
        "final_forward_then_inverse_valid_mask",
        1,
        observed_mode_points,
        mode="nearest",
    ).squeeze(-1) > 0.5
    sampled_inverse_cycle_valid = sample_mode_field(
        "final_inverse_then_forward_valid_mask", 1, sampled, mode="nearest"
    ).squeeze(-1) > 0.5
    sampled_forward_cycle_error = torch.linalg.vector_norm(
        sample_mode_field(
            "final_forward_then_inverse_error_yx", 2, observed_mode_points
        ),
        dim=-1,
    )
    sampled_inverse_cycle_error = torch.linalg.vector_norm(
        sample_mode_field("final_inverse_then_forward_error_yx", 2, sampled), dim=-1
    )
    mode_deformation_valid = (
        canonical_mode_in_bounds
        & torch.isfinite(sampled_jacobian)
        & (sampled_jacobian >= float(minimum_deformation_jacobian))
        & sampled_forward_cycle_valid
        & sampled_inverse_cycle_valid
        & torch.isfinite(sampled_forward_cycle_error)
        & torch.isfinite(sampled_inverse_cycle_error)
        & (sampled_forward_cycle_error <= float(maximum_cycle_error_px))
        & (sampled_inverse_cycle_error <= float(maximum_cycle_error_px))
    )
    canonical_points = points[:, None].expand(-1, states.shape[1], -1, -1).clone()
    deformation_valid = torch.ones(
        states.shape[0],
        states.shape[1],
        points.shape[1],
        device=states.device,
        dtype=torch.bool,
    )
    for mode in range(top_k):
        selected_mode = top_index == mode
        canonical_points = torch.where(
            selected_mode[..., None, None], sampled[:, mode, None], canonical_points
        )
        deformation_valid = torch.where(
            selected_mode[..., None],
            mode_deformation_valid[:, mode, None],
            deformation_valid,
        )
    canonical_y, canonical_x = canonical_points.unbind(dim=-1)
    raster_valid = (
        input_raster_valid[:, None]
        & torch.isfinite(canonical_y)
        & torch.isfinite(canonical_x)
        & (canonical_y >= 0.0)
        & (canonical_y <= height - 1)
        & (canonical_x >= 0.0)
        & (canonical_x <= width - 1)
    )
    st = torch.stack(
        (canonical_points[..., 1] / width, canonical_points[..., 0] / height), dim=-1
    )
    center, frame, basis = full_frame_state_to_components(states)
    edges = frame[..., :, :2] @ basis
    trajectory = center[:, :, None] + (
        edges[:, :, None] @ (st - 0.5)[..., None]
    ).squeeze(-1)
    origin = torch.as_tensor(
        origin_ap_dv_ml_um, device=states.device, dtype=states.dtype
    )
    spacing = torch.as_tensor(
        voxel_size_ap_dv_ml_um, device=states.device, dtype=states.dtype
    )
    if (
        origin.shape != (3,)
        or spacing.shape != (3,)
        or not bool(torch.isfinite(origin).all())
        or not bool(torch.isfinite(spacing).all())
        or bool((spacing <= 0.0).any())
    ):
        raise ValueError("atlas origin/voxel size must be finite AP-DV-ML triples")
    if atlas_shape_ap_dv_ml is None and annotation_volume_ap_dv_ml is not None:
        atlas_shape_ap_dv_ml = tuple(torch.as_tensor(annotation_volume_ap_dv_ml).shape)
    atlas_valid = torch.ones_like(raster_valid)
    atlas_index = torch.floor((trajectory - origin) / spacing).to(torch.long)
    if atlas_shape_ap_dv_ml is not None:
        atlas_shape = tuple(int(value) for value in atlas_shape_ap_dv_ml)
        if len(atlas_shape) != 3 or any(value < 1 for value in atlas_shape):
            raise ValueError("atlas shape must contain three positive dimensions")
        for axis, size in enumerate(atlas_shape):
            atlas_valid &= (atlas_index[..., axis] >= 0) & (
                atlas_index[..., axis] < size
            )
    trajectory_valid = (
        raster_valid
        & atlas_valid
        & deformation_valid
        & torch.isfinite(trajectory).all(dim=-1)
    )
    deformation_failure_probability = (
        weights[..., None] * (~deformation_valid).to(weights)
    ).sum(dim=1)
    valid_weight = weights[..., None] * trajectory_valid.to(weights)
    valid_probability = valid_weight.sum(dim=1)
    abstain = valid_probability <= torch.finfo(weights.dtype).tiny
    normalized_valid_weight = valid_weight / valid_probability[:, None].clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    mean = (normalized_valid_weight[..., None] * trajectory).sum(dim=1)
    safe_mean = torch.where(abstain[..., None], torch.zeros_like(mean), mean)
    covariance = (
        normalized_valid_weight[..., None, None]
        * (trajectory - safe_mean[:, None])[..., :, None]
        * (trajectory - safe_mean[:, None])[..., None, :]
    ).sum(dim=1)
    radius = torch.linalg.vector_norm(
        trajectory - safe_mean[:, None], dim=-1
    ).transpose(1, 2)
    tube_radius = _weighted_quantile(
        radius, normalized_valid_weight.transpose(1, 2)
    )
    nan = torch.full((), float("nan"), device=states.device, dtype=states.dtype)
    mean = torch.where(abstain[..., None], nan, mean)
    covariance = torch.where(abstain[..., None, None], nan, covariance)
    tube_radius = torch.where(abstain[..., None], nan, tube_radius)

    whole_trajectory_valid = trajectory_valid.all(dim=-1)
    whole_weight = weights * whole_trajectory_valid.to(weights)
    whole_valid_probability = whole_weight.sum(dim=1)
    whole_abstain = whole_valid_probability <= torch.finfo(weights.dtype).tiny
    normalized_whole_weight = whole_weight / whole_valid_probability[:, None].clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    simultaneous_center = (
        normalized_whole_weight[..., None, None] * trajectory
    ).sum(dim=1)
    simultaneous_distance = torch.linalg.vector_norm(
        trajectory - simultaneous_center[:, None], dim=-1
    ).amax(dim=-1)
    simultaneous_radius = _weighted_quantile(
        simultaneous_distance, normalized_whole_weight
    )
    simultaneous_center = torch.where(
        whole_abstain[:, None, None], nan, simultaneous_center
    )
    simultaneous_radius = torch.where(
        whole_abstain[:, None], nan, simultaneous_radius
    )
    result = {
        "trajectory_sample_points_ap_dv_ml_um": trajectory,
        "trajectory_sample_weights": weights,
        "input_raster_validity_mask": input_raster_valid,
        "canonical_raster_validity_mask": raster_valid,
        "atlas_validity_mask": atlas_valid,
        "trajectory_sample_validity_mask": trajectory_valid,
        "deformation_topology_validity_mask": deformation_valid,
        "sampled_mode_forward_jacobian_determinant": sampled_jacobian,
        "sampled_mode_forward_cycle_error_px": sampled_forward_cycle_error,
        "sampled_mode_inverse_cycle_error_px": sampled_inverse_cycle_error,
        "minimum_deformation_jacobian": float(minimum_deformation_jacobian),
        "maximum_cycle_error_px": float(maximum_cycle_error_px),
        "posterior_deformation_failure_probability": deformation_failure_probability,
        "atlas_voxel_index_ap_dv_ml": atlas_index,
        "credible_spatial_volume_representation": "unconditional weighted samples plus validity-conditioned pointwise and simultaneous summaries",
        "trajectory_uncertainty_scope": "deterministic one-SVF-per-refined-mode propagation; Gauss-Hermite pose nodes within a mode share that mode SVF; omitted retrieval-tail components use identity deformation",
        "trajectory_approximation_exact": False,
        "validity_conditioned_point_probability": valid_probability,
        "trajectory_point_abstention_mask": abstain,
        "validity_conditioned_pointwise_center_ap_dv_ml_um": mean,
        "validity_conditioned_pointwise_covariance_um2": covariance,
        "validity_conditioned_pointwise_credible_radius_um": tube_radius,
        "pointwise_credible_radius_scope": "marginal per electrode point, conditional on spatial and deformation validity; not simultaneous",
        "whole_trajectory_valid_probability": whole_valid_probability,
        "whole_trajectory_abstention_mask": whole_abstain,
        "validity_conditioned_simultaneous_center_ap_dv_ml_um": simultaneous_center,
        "validity_conditioned_simultaneous_uniform_credible_radius_um": simultaneous_radius,
        "simultaneous_credible_radius_scope": "posterior quantile of each whole-valid trajectory sample's maximum Euclidean deviation over all electrode points",
        "credible_levels": summary["credible_levels"],
        "unrefined_deformation_probability": summary["omitted_probability"],
        "omitted_mode_deformation_contract": "identity-tail approximation; omitted modes have exact retrieval mass but no refined SVF",
    }
    if annotation_volume_ap_dv_ml is not None:
        annotation = torch.as_tensor(annotation_volume_ap_dv_ml, device=states.device)
        if annotation.ndim != 3 or tuple(annotation.shape) != tuple(atlas_shape_ap_dv_ml):
            raise ValueError("annotation and verified atlas geometry must match exactly")
        if torch.is_floating_point(annotation):
            raise ValueError("atlas annotation labels must use an integer dtype")
        clipped = torch.stack(
            tuple(
                atlas_index[..., axis].clamp(0, annotation.shape[axis] - 1)
                for axis in range(3)
            ),
            dim=-1,
        )
        label = annotation[clipped[..., 0], clipped[..., 1], clipped[..., 2]].to(torch.long)
        valid_region = trajectory_valid & (label > 0)
        label = torch.where(valid_region, label, torch.full_like(label, -1))
        label_values = torch.unique(label[label > 0], sorted=True)
        probability = (
            weights[..., None, None]
            * label[..., None].eq(label_values).to(weights)
        ).sum(dim=1)
        invalid_probability = (
            weights[..., None] * label.eq(-1).to(weights)
        ).sum(dim=1)
        valid_probability = probability.sum(dim=-1)
        conditional_probability = probability / valid_probability[..., None].clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        confidence = (
            probability.max(dim=-1).values
            if label_values.numel()
            else torch.zeros_like(valid_probability)
        )
        result.update(
            {
                "region_label_per_trajectory_sample": label,
                "region_label_validity_mask": valid_region,
                "region_label_values": label_values,
                "region_assignment_probability": probability,
                "region_assignment_conditional_probability_given_valid": conditional_probability,
                "valid_region_probability": valid_probability,
                "invalid_region_label": -1,
                "invalid_region_probability": invalid_probability,
                "region_validity_contract": "strictly positive integer atlas annotation IDs are valid; zero, negative, raster-invalid, and atlas-invalid samples map to -1",
                "region_assignment_confidence": confidence,
            }
        )
    return result
