"""Native uncalibrated losses for the v6 retrieval/refinement cascade."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components,
    full_frame_state_to_physical_ouv,
)
from training.arbitrary_plane_retrieval_loss_v6 import (
    full_catalogue_proposal_nll_v6,
    selected_exact_rerank_nll_v6,
)


JOINT_LOSS_V6_SCHEMA = "anatomy-tracker.joint-loss/v6"
JOINT_LOSS_V6_CALIBRATED = False
RECURRENT_DEFORMATION_LOSS_GAMMA_V6 = 0.8


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a tensor")
    return value


def _integer_vector(value: object, length: int, device: torch.device, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if (
        tensor.shape != (length,)
        or tensor.dtype == torch.bool
        or torch.is_floating_point(tensor)
        or torch.is_complex(tensor)
    ):
        raise ValueError(f"{name} must be an integer vector with shape ({length},)")
    return tensor.to(torch.long)


def _weight(
    value: torch.Tensor | None,
    batch: int,
    reference: torch.Tensor,
    name: str,
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
        raise ValueError(f"{name} must be finite in [0,1] with shape (B,)")
    return weight


def _normalized_log_probability(
    value: torch.Tensor,
    valid: torch.Tensor,
    name: str,
) -> None:
    if value.ndim != 2 or not torch.is_floating_point(value) or value.shape != valid.shape:
        raise ValueError(f"{name} must be a floating (N,J) tensor matching its valid mask")
    if valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
        raise ValueError(f"{name} requires at least one valid component per row")
    if not bool(torch.isfinite(value[valid]).all()) or bool(
        (~valid & ~torch.isneginf(value)).any()
    ):
        raise ValueError(f"{name} has invalid or nonfinite component mass")
    partition = torch.logsumexp(value.masked_fill(~valid, -torch.inf), dim=1)
    if not torch.allclose(partition, torch.zeros_like(partition), atol=2e-6, rtol=0.0):
        raise ValueError(f"{name} must be normalized across its valid components")


def _truth_membership(
    selected: torch.Tensor,
    valid: torch.Tensor,
    truth: torch.Tensor,
    catalogue_count: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        selected.ndim != 2
        or selected.shape[1] < 1
        or selected.shape != valid.shape
        or selected.shape[0] != truth.shape[0]
        or selected.dtype == torch.bool
        or torch.is_floating_point(selected)
        or torch.is_complex(selected)
        or valid.dtype != torch.bool
        or not bool(valid.any(dim=1).all())
        or selected.device != truth.device
        or bool(((selected[valid] < 0) | (selected[valid] >= catalogue_count)).any())
    ):
        raise ValueError(f"{name} must provide integer (B,M) IDs and a Boolean valid mask")
    for row in range(selected.shape[0]):
        if torch.unique(selected[row, valid[row]]).numel() != int(valid[row].sum().item()):
            raise ValueError(f"{name} contains duplicate valid canonical IDs")
    match = selected.to(torch.long).eq(truth[:, None]) & valid
    if bool((match.sum(dim=1) > 1).any()):
        raise ValueError(f"{name} contains the truth more than once")
    hit = match.any(dim=1)
    return hit, match.to(torch.long).argmax(dim=1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denominator = weight.sum()
    numerator = (value * weight).sum()
    return numerator / torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator)
    )


def _cohort_mean(
    value: torch.Tensor, base_weight: torch.Tensor, cohort: torch.Tensor
) -> torch.Tensor:
    return _weighted_mean(value, base_weight * cohort.to(base_weight))


def _physical_frame_landmarks(state: torch.Tensor) -> torch.Tensor:
    ouv = full_frame_state_to_physical_ouv(state)
    origin, edge_u, edge_v = ouv[..., :3], ouv[..., 3:6], ouv[..., 6:9]
    return torch.stack(
        (
            origin,
            origin + edge_u,
            origin + edge_v,
            origin + edge_u + edge_v,
            origin + 0.5 * (edge_u + edge_v),
        ),
        dim=-2,
    )


def _plane_tangent_residual(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    pred_center, pred_frame, _ = full_frame_state_to_components(prediction)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth)
    pred_u, pred_v, pred_normal = pred_frame.unbind(dim=-1)
    truth_normal = truth_frame[..., :, 2]
    dot = (pred_normal * truth_normal).sum(dim=-1)
    sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
    aligned_normal = truth_normal * sign[..., None]
    cosine = (pred_normal * aligned_normal).sum(dim=-1).clamp(-1.0, 1.0)
    tangent_direction = aligned_normal - cosine[..., None] * pred_normal
    tangent_norm = torch.linalg.vector_norm(tangent_direction, dim=-1)
    angle = torch.atan2(tangent_norm, cosine)
    ratio = angle / tangent_norm.clamp_min(1e-7)
    ratio = torch.where(tangent_norm > 1e-4, ratio, 1.0 + tangent_norm.square() / 6.0)
    tangent = tangent_direction * ratio[..., None]
    origin = torch.as_tensor(
        support_origin_ap_dv_ml_um,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
        raise ValueError("support origin must contain three finite physical coordinates")
    pred_offset = ((pred_center - origin) * pred_normal).sum(dim=-1)
    truth_offset = ((truth_center - origin) * truth_normal).sum(dim=-1) * sign
    return torch.stack(
        (
            (tangent * pred_u).sum(dim=-1),
            (tangent * pred_v).sum(dim=-1),
            truth_offset - pred_offset,
        ),
        dim=-1,
    )


def _plane_mixture_nll_per_row(
    state: torch.Tensor,
    covariance: torch.Tensor,
    component_log_probability: torch.Tensor,
    truth_state: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    work_dtype = torch.promote_types(state.dtype, covariance.dtype)
    work_dtype = torch.promote_types(work_dtype, truth_state.dtype)
    work_dtype = torch.promote_types(work_dtype, component_log_probability.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    with torch.autocast(device_type=state.device.type, enabled=False):
        work_state = state.to(work_dtype)
        truth = truth_state.to(work_dtype)[:, None].expand_as(work_state)
        residual = _plane_tangent_residual(
            work_state, truth, support_origin_ap_dv_ml_um
        )
        work_covariance = covariance.to(work_dtype)
        identity = torch.eye(3, device=state.device, dtype=work_dtype)
        cholesky = torch.linalg.cholesky(work_covariance + 1e-6 * identity)
        standardized = torch.linalg.solve_triangular(
            cholesky, residual[..., None], upper=False
        ).squeeze(-1)
        log_density = -0.5 * (
            standardized.square().sum(dim=-1)
            + 2.0
            * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
            + 3.0 * math.log(2.0 * math.pi)
        )
        return -torch.logsumexp(
            component_log_probability.to(work_dtype) + log_density, dim=1
        )


def _landmark_mixture_nll_per_row(
    state: torch.Tensor,
    component_log_probability: torch.Tensor,
    truth_state: torch.Tensor,
    landmark_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = _physical_frame_landmarks(state)
    truth = _physical_frame_landmarks(truth_state)[:, None]
    error_um = torch.linalg.vector_norm(predicted - truth, dim=-1).mean(dim=-1)
    log_likelihood = -F.smooth_l1_loss(
        error_um / landmark_scale_um,
        torch.zeros_like(error_um),
        beta=1.0,
        reduction="none",
    )
    return -torch.logsumexp(component_log_probability + log_likelihood, dim=1), error_um


def _gather_mode(value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    index = position.reshape(position.shape + (1,) * (value.ndim - 2)).expand(
        position.shape + value.shape[2:]
    )
    return torch.gather(value, 1, index[:, None]).squeeze(1)


def _dense_mean(
    value: torch.Tensor, element_weight: torch.Tensor, row_weight: torch.Tensor
) -> torch.Tensor:
    expanded = element_weight.expand_as(value)
    dimensions = tuple(range(1, value.ndim))
    denominator = expanded.sum(dim=dimensions)
    numerator = (value * expanded).sum(dim=dimensions)
    per_row = numerator / torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator)
    )
    return _weighted_mean(
        per_row, row_weight * (denominator > 0.0).to(row_weight)
    )


def arbitrary_plane_joint_loss_v6(
    output: Mapping[str, object],
    truth_state: torch.Tensor,
    truth_catalogue_index: torch.Tensor,
    truth_stationary_velocity_yx_px: torch.Tensor,
    truth_pullback_map_yx_px: torch.Tensor,
    deformation_weight: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    *,
    expected_catalogue_cell_count: int,
    retrieval_supervision_weight: torch.Tensor | None = None,
    pose_supervision_weight: torch.Tensor | None = None,
    dense_deformation_supervision_weight: torch.Tensor | None = None,
    landmark_scale_um: float = 250.0,
    proposal_loss_weight: float = 1.0,
    rerank_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor | int | bool | str]:
    """Loss over full-B retrieval and compact R-only refinement outputs."""
    output = _mapping(output, "v6 joint output")
    cascade = _mapping(output.get("cascade"), "cascade")
    proposal_log = _tensor(cascade, "raw_full_catalogue_proposal_log_probability")
    if proposal_log.ndim != 2:
        raise ValueError("full-catalogue proposal log probability must have shape (B,K)")
    batch = proposal_log.shape[0]
    truth_state = torch.as_tensor(
        truth_state, device=proposal_log.device, dtype=proposal_log.dtype
    )
    if truth_state.shape != (batch, 12):
        raise ValueError("truth state must have shape (B,12)")
    truth_index = _integer_vector(
        truth_catalogue_index, batch, proposal_log.device, "truth catalogue index"
    )
    retrieval_weight = _weight(
        retrieval_supervision_weight, batch, proposal_log, "retrieval supervision weight"
    )
    pose_weight = _weight(
        pose_supervision_weight, batch, proposal_log, "pose supervision weight"
    )
    dense_weight = _weight(
        dense_deformation_supervision_weight,
        batch,
        proposal_log,
        "dense deformation supervision weight",
    )
    if (
        not isinstance(expected_catalogue_cell_count, int)
        or isinstance(expected_catalogue_cell_count, bool)
        or expected_catalogue_cell_count < 1
        or not isinstance(landmark_scale_um, (int, float))
        or isinstance(landmark_scale_um, bool)
        or not math.isfinite(float(landmark_scale_um))
        or landmark_scale_um <= 0.0
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0.0
            for value in (proposal_loss_weight, rerank_loss_weight)
        )
    ):
        raise ValueError("catalogue count, landmark scale, and loss weights are invalid")

    proposal = full_catalogue_proposal_nll_v6(
        proposal_log,
        truth_index,
        expected_catalogue_cell_count=expected_catalogue_cell_count,
        supervision_weight=retrieval_weight,
    )
    declared_training_truth = cascade.get("training_truth_catalogue_index")
    if declared_training_truth is not None and not torch.equal(
        _integer_vector(
            declared_training_truth,
            batch,
            proposal_log.device,
            "declared training truth catalogue index",
        ),
        truth_index,
    ):
        raise ValueError("declared training truth does not match loss truth")
    initial_index = _tensor(cascade, "honest_initial_topm_catalogue_index")
    initial_hit, _ = _truth_membership(
        initial_index,
        torch.ones_like(initial_index, dtype=torch.bool),
        truth_index,
        expected_catalogue_cell_count,
        "honest initial top-M",
    )
    declared_initial_hit = cascade.get("honest_initial_topm_truth_hit")
    if declared_initial_hit is not None:
        declared_initial_hit = torch.as_tensor(
            declared_initial_hit, device=proposal_log.device
        )
        if declared_initial_hit.dtype != torch.bool or not torch.equal(
            declared_initial_hit, initial_hit
        ):
            raise ValueError("declared initial top-M truth hits do not match canonical IDs")

    honest = _mapping(cascade.get("honest_hybrid_posterior"), "honest hybrid posterior")
    honest_index = _tensor(honest, "selected_catalogue_index")
    honest_valid = _tensor(honest, "selected_valid_mask")
    honest_conditional = _tensor(honest, "selected_conditional_log_probability")
    honest_hit, honest_position = _truth_membership(
        honest_index,
        honest_valid,
        truth_index,
        expected_catalogue_cell_count,
        "honest final rendered selection",
    )
    honest_rerank = selected_exact_rerank_nll_v6(
        honest_conditional,
        honest_valid,
        honest_position,
        honest_hit,
        supervision_weight=retrieval_weight,
    )

    forced = _tensor(cascade, "training_truth_forced_mask").to(
        device=proposal_log.device
    )
    if forced.shape != (batch,) or forced.dtype != torch.bool or bool(
        (forced & honest_hit).any()
    ):
        raise ValueError("teacher-forced mask must contain only honest final misses")
    teacher = cascade.get("training_teacher_forced_hybrid_posterior")
    if teacher is None:
        if bool(forced.any()):
            raise ValueError("teacher-forced rows require a teacher posterior")
        teacher_per_row = proposal_log[:, 0] * 0.0
        teacher_hit = torch.zeros_like(forced)
    else:
        teacher = _mapping(teacher, "teacher-forced hybrid posterior")
        teacher_index = _tensor(teacher, "selected_catalogue_index")
        teacher_valid = _tensor(teacher, "selected_valid_mask")
        teacher_conditional = _tensor(teacher, "selected_conditional_log_probability")
        teacher_hit, teacher_position = _truth_membership(
            teacher_index,
            teacher_valid,
            truth_index,
            expected_catalogue_cell_count,
            "teacher-forced selection",
        )
        if not torch.equal(forced, ~honest_hit) or not bool(teacher_hit[forced].all()):
            raise ValueError("every teacher-forced row must contain its canonical truth")
        declared_teacher_index = cascade.get("training_selected_catalogue_index")
        declared_teacher_valid = cascade.get("training_selected_valid_mask")
        if not isinstance(declared_teacher_index, torch.Tensor) or not isinstance(
            declared_teacher_valid, torch.Tensor
        ) or not torch.equal(declared_teacher_index, teacher_index) or not torch.equal(
            declared_teacher_valid, teacher_valid
        ):
            raise ValueError("teacher posterior does not match the declared training selection")
        teacher_rerank = selected_exact_rerank_nll_v6(
            teacher_conditional,
            teacher_valid,
            teacher_position,
            forced,
            supervision_weight=retrieval_weight,
        )
        teacher_per_row = teacher_rerank["per_row_nll"]
    rerank_weight = retrieval_weight * (honest_hit | forced).to(retrieval_weight)
    rerank_per_row = honest_rerank["per_row_nll"] + teacher_per_row
    rerank_nll = _weighted_mean(rerank_per_row, rerank_weight)
    retrieval_nll = (
        float(proposal_loss_weight) * proposal["loss"]
        + float(rerank_loss_weight) * rerank_nll
    )

    ready = _tensor(cascade, "honest_refinement_ready_mask").to(
        device=proposal_log.device
    )
    if ready.shape != (batch,) or ready.dtype != torch.bool:
        raise ValueError("honest refinement-ready mask must be Boolean with shape (B,)")
    for key, expected_mask in (
        ("refinement_ready_mask", ready),
        ("refinement_abstained_mask", ~ready),
        ("refinement_performed_mask", ready),
    ):
        declared_mask = _tensor(output, key)
        if declared_mask.dtype != torch.bool or not torch.equal(
            declared_mask.to(device=ready.device), expected_mask
        ):
            raise ValueError(f"{key} does not match the honest cascade boundary")
    source = _integer_vector(
        output.get("refinement_source_batch_index"),
        int(ready.sum().item()),
        proposal_log.device,
        "refinement source batch index",
    )
    expected_source = torch.nonzero(ready, as_tuple=False).squeeze(1)
    if not torch.equal(source, expected_source):
        raise ValueError("refinement source mapping must exactly match honest ready rows")
    refined_rows = source.numel()
    if refined_rows == 0:
        if output.get("refined_output") is not None:
            raise ValueError("all-abstained output must not contain a refined payload")
        zero = proposal_log.sum() * 0.0
        return {
            "schema_version": JOINT_LOSS_V6_SCHEMA,
            "probabilities_calibrated": JOINT_LOSS_V6_CALIBRATED,
            "probability_status": "raw_uncalibrated",
            "total": retrieval_nll,
            "retrieval_nll": retrieval_nll,
            "full_catalogue_proposal_nll": proposal["loss"],
            "selected_finite_render_conditional_rerank_nll": rerank_nll,
            "refinement_total": zero,
            "initial_plane_mixture_nll": zero,
            "final_plane_mixture_nll": zero,
            "final_landmark_mixture_nll": zero,
            "deformation_svf": zero,
            "deformation_map": zero,
            "deformation_support": zero,
            "deformation_topology": zero,
            "deformation_smoothness": zero,
            "deformation_inverse_consistency": zero,
            "mean_best_topk_landmark_error_um": zero,
            "proposal_eligible_row_count": proposal["eligible_row_count"],
            "proposal_supervision_weight_sum": proposal["supervision_weight_sum"],
            "honest_initial_selected_row_count": int(initial_hit.sum().item()),
            "honest_initial_miss_row_count": int((~initial_hit).sum().item()),
            "honest_initial_selected_proposal_nll": _cohort_mean(
                proposal["per_row_nll"], retrieval_weight, initial_hit
            ),
            "honest_initial_miss_proposal_nll": _cohort_mean(
                proposal["per_row_nll"], retrieval_weight, ~initial_hit
            ),
            "honest_final_selected_row_count": int(honest_hit.sum().item()),
            "honest_final_miss_row_count": int((~honest_hit).sum().item()),
            "honest_final_selected_rerank_nll": honest_rerank["loss"],
            "teacher_forced_row_count": int(forced.sum().item()),
            "teacher_forced_rerank_nll": _cohort_mean(
                teacher_per_row, retrieval_weight, forced
            ),
            "rerank_eligible_row_count": int((rerank_weight > 0.0).sum().item()),
            "rerank_eligible_weight_sum": rerank_weight.sum(),
            "refinement_ready_row_count": 0,
            "refinement_abstained_row_count": batch,
            "refinement_topk_truth_selected_row_count": 0,
            "refinement_topk_truth_miss_row_count": 0,
            "refinement_pose_eligible_weight_sum": pose_weight.sum() * 0.0,
            "refinement_dense_eligible_weight_sum": dense_weight.sum() * 0.0,
        }

    refined = _mapping(output.get("refined_output"), "refined_output")
    pose = _mapping(refined.get("pose"), "refined_output.pose")

    topk_index = _tensor(pose, "retrieval_topk_catalogue_index")
    topk_cell_id = _tensor(pose, "retrieval_topk_cell_id")
    if (
        topk_index.ndim != 2
        or topk_index.shape[0] != refined_rows
        or topk_index.shape[1] < 1
        or topk_index.dtype == torch.bool
        or torch.is_floating_point(topk_index)
        or topk_cell_id.dtype == torch.bool
        or torch.is_floating_point(topk_cell_id)
        or torch.is_complex(topk_index)
        or torch.is_complex(topk_cell_id)
        or not torch.equal(topk_index.to(torch.long), topk_cell_id.to(torch.long))
    ):
        raise ValueError("compact refined top-K must provide identical canonical IDs")
    honest_topk = _tensor(honest, "hybrid_topk_catalogue_index")
    if honest_topk.ndim != 2 or honest_topk.shape[0] != batch:
        raise ValueError("honest hybrid top-K must have shape (B,J)")
    _truth_membership(
        honest_topk,
        torch.ones_like(honest_topk, dtype=torch.bool),
        truth_index,
        expected_catalogue_cell_count,
        "honest hybrid top-K",
    )
    declared_selected = _tensor(output, "refinement_selected_catalogue_index")
    declared_selected_cell_id = _tensor(output, "refinement_selected_cell_id")
    initial_honest_topk = _tensor(
        output, "refinement_initial_honest_topk_catalogue_index"
    )
    initial_honest_mask = _tensor(output, "refinement_initial_honest_mode_mask")
    final_honest_mask = _tensor(output, "refinement_final_honest_mode_mask")
    final_teacher_mask = _tensor(
        output, "refinement_final_teacher_forced_mode_mask"
    )
    refinement_teacher_full = _tensor(output, "refinement_teacher_forced_mask")
    if (
        not torch.equal(declared_selected, topk_index)
        or not torch.equal(declared_selected_cell_id, topk_cell_id)
        or not torch.equal(initial_honest_topk.to(torch.long), honest_topk[source].to(torch.long))
        or initial_honest_mask.shape != topk_index.shape
        or final_honest_mask.shape != topk_index.shape
        or final_teacher_mask.shape != topk_index.shape
        or any(
            mask.dtype != torch.bool
            for mask in (initial_honest_mask, final_honest_mask, final_teacher_mask)
        )
        or not bool(initial_honest_mask.all())
        or bool((final_honest_mask & final_teacher_mask).any())
        or not bool((final_honest_mask | final_teacher_mask).all())
        or refinement_teacher_full.shape != (batch,)
        or refinement_teacher_full.dtype != torch.bool
        or bool((refinement_teacher_full & ~ready).any())
    ):
        raise ValueError("refinement honest/teacher selection declarations are inconsistent")
    refined_truth = truth_index[source]
    compact_teacher = refinement_teacher_full[source]
    expected_selected = initial_honest_topk.to(torch.long).clone()
    expected_selected[compact_teacher, -1] = refined_truth[compact_teacher]
    expected_teacher_mask = ~expected_selected[..., None].eq(
        initial_honest_topk.to(torch.long)[:, None]
    ).any(dim=-1)
    if (
        not torch.equal(topk_index.to(torch.long), expected_selected)
        or not torch.equal(final_teacher_mask, expected_teacher_mask)
        or not torch.equal(final_honest_mask, ~expected_teacher_mask)
        or not torch.equal(expected_teacher_mask.any(dim=1), compact_teacher)
        or bool((expected_teacher_mask.sum(dim=1) > 1).any())
    ):
        raise ValueError("teacher refinement must be one exact canonical truth replacement")
    topk_valid = torch.ones_like(topk_index, dtype=torch.bool)
    topk_hit, truth_topk_position = _truth_membership(
        topk_index,
        topk_valid,
        refined_truth,
        expected_catalogue_cell_count,
        "compact refined top-K",
    )
    initial_log_probability = _tensor(pose, "refinement_initial_topk_log_probability")
    declared_initial_log_probability = _tensor(
        output, "refinement_initial_topk_log_probability"
    )
    if not torch.equal(declared_initial_log_probability, initial_log_probability):
        raise ValueError("top-level and compact initial refinement probabilities differ")
    final_log_probability = _tensor(
        pose, "conditional_within_topk_cell_log_probability"
    )
    _normalized_log_probability(
        initial_log_probability, topk_valid, "initial refinement top-K probability"
    )
    _normalized_log_probability(
        final_log_probability, topk_valid, "final refinement top-K probability"
    )
    initial_state = _tensor(pose, "topk_initial_cell_state")
    initial_covariance = _tensor(
        pose, "topk_initial_cell_canonical_plane_covariance"
    )
    final_state = _tensor(pose, "final_cell_state")
    final_covariance = _tensor(pose, "final_cell_canonical_plane_covariance")
    expected_state_shape = topk_index.shape + (12,)
    expected_covariance_shape = topk_index.shape + (3, 3)
    if (
        initial_state.shape != expected_state_shape
        or final_state.shape != expected_state_shape
        or initial_covariance.shape != expected_covariance_shape
        or final_covariance.shape != expected_covariance_shape
        or any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in (initial_state, final_state, initial_covariance, final_covariance)
        )
        or any(
            value.device != proposal_log.device
            for value in (
                initial_log_probability,
                final_log_probability,
                initial_state,
                final_state,
                initial_covariance,
                final_covariance,
            )
        )
    ):
        raise ValueError("compact pose states and covariances have invalid shape or values")
    if refined_rows and bool(
        (
            torch.linalg.cholesky_ex(
                initial_covariance
                + 1e-6
                * torch.eye(3, device=initial_covariance.device, dtype=initial_covariance.dtype)
            ).info
            != 0
        ).any()
        or (
            torch.linalg.cholesky_ex(
                final_covariance
                + 1e-6
                * torch.eye(3, device=final_covariance.device, dtype=final_covariance.dtype)
            ).info
            != 0
        ).any()
    ):
        raise ValueError("compact pose covariances must be positive definite")
    if not torch.allclose(
        initial_covariance,
        initial_covariance.transpose(-1, -2),
        atol=1e-6,
        rtol=0.0,
    ) or not torch.allclose(
        final_covariance,
        final_covariance.transpose(-1, -2),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("compact pose covariances must be symmetric")
    graph_zero = (
        proposal_log.sum() * 0.0
        + initial_state.sum() * 0.0
        + final_state.sum() * 0.0
    )
    compact_pose_weight = pose_weight[source] * topk_hit.to(pose_weight)
    eligible = topk_hit & (compact_pose_weight > 0.0)
    if not bool(torch.isfinite(truth_state[source][eligible]).all()):
        raise ValueError("pose-eligible compact truth states must be finite")
    if refined_rows and bool(eligible.any()):
        initial_plane_per_row = _plane_mixture_nll_per_row(
            initial_state[eligible],
            initial_covariance[eligible],
            initial_log_probability[eligible],
            truth_state[source][eligible],
            support_origin_ap_dv_ml_um,
        )
        final_plane_per_row = _plane_mixture_nll_per_row(
            final_state[eligible],
            final_covariance[eligible],
            final_log_probability[eligible],
            truth_state[source][eligible],
            support_origin_ap_dv_ml_um,
        )
        final_landmark_per_row, landmark_error_um = _landmark_mixture_nll_per_row(
            final_state[eligible],
            final_log_probability[eligible],
            truth_state[source][eligible],
            float(landmark_scale_um),
        )
        eligible_weight = compact_pose_weight[eligible]
        initial_plane_nll = _weighted_mean(initial_plane_per_row, eligible_weight)
        final_plane_nll = _weighted_mean(final_plane_per_row, eligible_weight)
        final_landmark_nll = _weighted_mean(final_landmark_per_row, eligible_weight)
        mean_best_landmark_error_um = _weighted_mean(
            landmark_error_um.min(dim=1).values, eligible_weight
        )
    else:
        initial_plane_nll = graph_zero
        final_plane_nll = graph_zero
        final_landmark_nll = graph_zero
        mean_best_landmark_error_um = graph_zero

    velocity = _tensor(refined, "stationary_velocity_yx_px_sequence")
    pullback = _tensor(refined, "pullback_map_yx_px_sequence")
    support_logits = _tensor(refined, "support_logits_sequence")
    jacobian = _tensor(refined, "forward_jacobian_determinant_sequence")
    cycle_forward = _tensor(refined, "forward_then_inverse_error_yx_sequence")
    cycle_inverse = _tensor(refined, "inverse_then_forward_error_yx_sequence")
    cycle_forward_valid = _tensor(refined, "forward_then_inverse_valid_mask_sequence")
    cycle_inverse_valid = _tensor(refined, "inverse_then_forward_valid_mask_sequence")
    active = _tensor(refined, "deformation_active_sequence")
    dense_values = (
        velocity,
        pullback,
        support_logits,
        jacobian,
        cycle_forward,
        cycle_inverse,
        cycle_forward_valid,
        cycle_inverse_valid,
    )
    if any(value.ndim < 3 or value.shape[:2] != topk_index.shape for value in dense_values):
        raise ValueError("compact deformation tensors must begin with identical (R,J) axes")
    selected_velocity = _gather_mode(velocity, truth_topk_position)
    selected_pullback = _gather_mode(pullback, truth_topk_position)
    selected_support = _gather_mode(support_logits, truth_topk_position)
    selected_jacobian = _gather_mode(jacobian, truth_topk_position)
    selected_cycle_forward = _gather_mode(cycle_forward, truth_topk_position)
    selected_cycle_inverse = _gather_mode(cycle_inverse, truth_topk_position)
    selected_cycle_forward_valid = _gather_mode(cycle_forward_valid, truth_topk_position)
    selected_cycle_inverse_valid = _gather_mode(cycle_inverse_valid, truth_topk_position)
    expected_scalar_shape = selected_velocity.shape[:2] + (1,) + selected_velocity.shape[3:]
    if (
        selected_velocity.ndim != 5
        or selected_velocity.shape[2] != 2
        or selected_pullback.shape != selected_velocity.shape
        or selected_cycle_forward.shape != selected_velocity.shape
        or selected_cycle_inverse.shape != selected_velocity.shape
        or selected_support.shape != expected_scalar_shape
        or selected_jacobian.shape != expected_scalar_shape
        or selected_cycle_forward_valid.shape != expected_scalar_shape
        or selected_cycle_inverse_valid.shape != expected_scalar_shape
        or active.shape != (selected_velocity.shape[1],)
        or active.dtype != torch.bool
        or any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in dense_values[:6]
        )
        or selected_cycle_forward_valid.dtype != torch.bool
        or selected_cycle_inverse_valid.dtype != torch.bool
    ):
        raise ValueError("compact deformation sequence shapes or values are invalid")
    height, width = selected_velocity.shape[-2:]
    truth_velocity = torch.as_tensor(
        truth_stationary_velocity_yx_px,
        device=proposal_log.device,
        dtype=selected_velocity.dtype,
    )
    truth_pullback = torch.as_tensor(
        truth_pullback_map_yx_px,
        device=proposal_log.device,
        dtype=selected_pullback.dtype,
    )
    pixel_weight = torch.as_tensor(
        deformation_weight,
        device=proposal_log.device,
        dtype=selected_velocity.dtype,
    )
    if (
        truth_velocity.shape != (batch, 2, height, width)
        or truth_pullback.shape != (batch, 2, height, width)
        or pixel_weight.shape != (batch, 1, height, width)
    ):
        raise ValueError("dense truths and deformation weights have invalid shapes")

    dense_row_weight_all = compact_pose_weight * dense_weight[source]
    dense_eligible = dense_row_weight_all > 0.0
    dense_row_weight = dense_row_weight_all[dense_eligible]
    dense_source = source[dense_eligible]
    selected_velocity = selected_velocity[dense_eligible]
    selected_pullback = selected_pullback[dense_eligible]
    selected_support = selected_support[dense_eligible]
    selected_jacobian = selected_jacobian[dense_eligible]
    selected_cycle_forward = selected_cycle_forward[dense_eligible]
    selected_cycle_inverse = selected_cycle_inverse[dense_eligible]
    selected_cycle_forward_valid = selected_cycle_forward_valid[dense_eligible]
    selected_cycle_inverse_valid = selected_cycle_inverse_valid[dense_eligible]
    if (
        not bool(torch.isfinite(truth_velocity[dense_source]).all())
        or not bool(torch.isfinite(truth_pullback[dense_source]).all())
        or not bool(torch.isfinite(pixel_weight[dense_source]).all())
        or bool(
            (
                (pixel_weight[dense_source] < 0.0)
                | (pixel_weight[dense_source] > 1.0)
            ).any()
        )
    ):
        raise ValueError("dense-eligible truths and deformation weights are invalid")
    sequence = active.to(selected_velocity) * RECURRENT_DEFORMATION_LOSS_GAMMA_V6 ** torch.arange(
        active.numel() - 1,
        -1,
        -1,
        device=active.device,
        dtype=selected_velocity.dtype,
    )
    sequence = sequence.reshape(1, -1, 1, 1, 1)
    selected_pixel_weight = pixel_weight[dense_source, None]
    vector_element_weight = (selected_pixel_weight * sequence).expand_as(selected_velocity)
    scalar_sequence_weight = sequence.expand_as(selected_support)
    scalar_element_weight = (selected_pixel_weight * sequence).expand_as(selected_support)
    deformation_svf = _dense_mean(
        F.smooth_l1_loss(
            selected_velocity,
            truth_velocity[dense_source, None].expand_as(selected_velocity),
            beta=0.5,
            reduction="none",
        ),
        vector_element_weight,
        dense_row_weight,
    )
    deformation_map = _dense_mean(
        F.smooth_l1_loss(
            selected_pullback,
            truth_pullback[dense_source, None].expand_as(selected_pullback),
            beta=0.5,
            reduction="none",
        ),
        vector_element_weight,
        dense_row_weight,
    )
    deformation_support = _dense_mean(
        F.binary_cross_entropy_with_logits(
            selected_support,
            pixel_weight[dense_source, None].expand_as(selected_support),
            reduction="none",
        ),
        scalar_sequence_weight,
        dense_row_weight,
    )
    deformation_topology = _dense_mean(
        F.relu(0.05 - selected_jacobian).square(),
        scalar_element_weight,
        dense_row_weight,
    )
    difference_y = selected_velocity[..., 1:, :] - selected_velocity[..., :-1, :]
    difference_x = selected_velocity[..., :, 1:] - selected_velocity[..., :, :-1]
    weight_y = vector_element_weight[..., 1:, :] * vector_element_weight[..., :-1, :]
    weight_x = vector_element_weight[..., :, 1:] * vector_element_weight[..., :, :-1]
    deformation_smoothness = 0.5 * (
        _dense_mean(difference_y.square(), weight_y, dense_row_weight)
        + _dense_mean(difference_x.square(), weight_x, dense_row_weight)
    )
    deformation_inverse_consistency = 0.5 * (
        _dense_mean(
            selected_cycle_forward.square(),
            vector_element_weight * selected_cycle_forward_valid.to(vector_element_weight),
            dense_row_weight,
        )
        + _dense_mean(
            selected_cycle_inverse.square(),
            vector_element_weight * selected_cycle_inverse_valid.to(vector_element_weight),
            dense_row_weight,
        )
    )
    refinement_total = (
        0.1 * initial_plane_nll
        + 0.5 * final_plane_nll
        + final_landmark_nll
        + 0.5 * deformation_svf
        + 0.25 * deformation_map
        + 0.05 * deformation_support
        + 0.1 * deformation_topology
        + 0.05 * deformation_smoothness
        + 0.05 * deformation_inverse_consistency
    )
    total = retrieval_nll + refinement_total
    return {
        "schema_version": JOINT_LOSS_V6_SCHEMA,
        "probabilities_calibrated": JOINT_LOSS_V6_CALIBRATED,
        "probability_status": "raw_uncalibrated",
        "total": total,
        "retrieval_nll": retrieval_nll,
        "full_catalogue_proposal_nll": proposal["loss"],
        "selected_finite_render_conditional_rerank_nll": rerank_nll,
        "refinement_total": refinement_total,
        "initial_plane_mixture_nll": initial_plane_nll,
        "final_plane_mixture_nll": final_plane_nll,
        "final_landmark_mixture_nll": final_landmark_nll,
        "deformation_svf": deformation_svf,
        "deformation_map": deformation_map,
        "deformation_support": deformation_support,
        "deformation_topology": deformation_topology,
        "deformation_smoothness": deformation_smoothness,
        "deformation_inverse_consistency": deformation_inverse_consistency,
        "mean_best_topk_landmark_error_um": mean_best_landmark_error_um,
        "proposal_eligible_row_count": proposal["eligible_row_count"],
        "proposal_supervision_weight_sum": proposal["supervision_weight_sum"],
        "honest_initial_selected_row_count": int(initial_hit.sum().item()),
        "honest_initial_miss_row_count": int((~initial_hit).sum().item()),
        "honest_initial_selected_proposal_nll": _cohort_mean(
            proposal["per_row_nll"], retrieval_weight, initial_hit
        ),
        "honest_initial_miss_proposal_nll": _cohort_mean(
            proposal["per_row_nll"], retrieval_weight, ~initial_hit
        ),
        "honest_final_selected_row_count": int(honest_hit.sum().item()),
        "honest_final_miss_row_count": int((~honest_hit).sum().item()),
        "honest_final_selected_rerank_nll": honest_rerank["loss"],
        "teacher_forced_row_count": int(forced.sum().item()),
        "teacher_forced_rerank_nll": _cohort_mean(
            teacher_per_row, retrieval_weight, forced
        ),
        "rerank_eligible_row_count": int((rerank_weight > 0.0).sum().item()),
        "rerank_eligible_weight_sum": rerank_weight.sum(),
        "refinement_ready_row_count": refined_rows,
        "refinement_abstained_row_count": batch - refined_rows,
        "refinement_topk_truth_selected_row_count": int(topk_hit.sum().item()),
        "refinement_topk_truth_miss_row_count": int((~topk_hit).sum().item()),
        "refinement_pose_eligible_weight_sum": compact_pose_weight.sum(),
        "refinement_dense_eligible_weight_sum": dense_row_weight_all.sum(),
    }


__all__ = [
    "JOINT_LOSS_V6_CALIBRATED",
    "JOINT_LOSS_V6_SCHEMA",
    "arbitrary_plane_joint_loss_v6",
]
