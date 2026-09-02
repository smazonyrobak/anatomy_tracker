"""Physical probabilistic losses for the fresh arbitrary-plane joint model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components,
    full_frame_state_to_physical_ouv,
)


RECURRENT_DEFORMATION_LOSS_GAMMA = 0.8


def plane_tangent_residual(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Return exact sphere-log normal residuals plus antipodal signed offset."""
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


def physical_frame_landmarks(state: torch.Tensor) -> torch.Tensor:
    """Return four finite-frame corners and centre in physical AP/DV/ML um."""
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


def gaussian_plane_mixture_nll(
    state: torch.Tensor,
    covariance: torch.Tensor,
    component_log_mass: torch.Tensor,
    truth_state: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    truth = truth_state[:, None].expand_as(state)
    residual = plane_tangent_residual(state, truth, support_origin_ap_dv_ml_um)
    identity = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    cholesky = torch.linalg.cholesky(covariance + 1e-6 * identity)
    standardized = torch.linalg.solve_triangular(
        cholesky, residual[..., None], upper=False
    ).squeeze(-1)
    log_density = -0.5 * (
        standardized.square().sum(dim=-1)
        + 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
        + 3.0 * math.log(2.0 * math.pi)
    )
    return -torch.logsumexp(component_log_mass + log_density, dim=1).mean()


def landmark_mixture_nll(
    state: torch.Tensor,
    component_log_mass: torch.Tensor,
    truth_state: torch.Tensor,
    landmark_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = physical_frame_landmarks(state)
    truth = physical_frame_landmarks(truth_state)[:, None]
    error_um = torch.linalg.vector_norm(predicted - truth, dim=-1).mean(dim=-1)
    log_likelihood = -F.smooth_l1_loss(
        error_um / landmark_scale_um,
        torch.zeros_like(error_um),
        beta=1.0,
        reduction="none",
    )
    return -torch.logsumexp(component_log_mass + log_likelihood, dim=1).mean(), error_um


def _gather_mode(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    safe = index.clamp_min(0)
    gather_index = safe.reshape(safe.shape + (1,) * (value.ndim - 2)).expand(
        safe.shape + value.shape[2:]
    )
    return torch.gather(value, 1, gather_index[:, None]).squeeze(1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def arbitrary_plane_joint_loss(
    output: dict[str, object],
    truth_state: torch.Tensor,
    truth_catalogue_cell_id: torch.Tensor,
    truth_topk_index: torch.Tensor,
    truth_stationary_velocity_yx_px: torch.Tensor,
    truth_pullback_map_yx_px: torch.Tensor,
    deformation_weight: torch.Tensor,
    support_origin_ap_dv_ml_um: torch.Tensor | tuple[float, float, float],
    *,
    landmark_scale_um: float = 250.0,
    pose_supervision_weight: torch.Tensor | None = None,
    dense_deformation_supervision_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combine pose losses with independently gated dense deformation losses."""
    pose = output["pose"]
    batch = truth_state.shape[0]
    if pose_supervision_weight is None:
        pose_weight = torch.ones(
            batch,
            device=truth_state.device,
            dtype=truth_state.dtype,
        )
    else:
        pose_weight = torch.as_tensor(
            pose_supervision_weight,
            device=truth_state.device,
            dtype=truth_state.dtype,
        )
        if pose_weight.shape != (batch,) or not bool(
            torch.isfinite(pose_weight).all()
            and (pose_weight >= 0.0).all()
            and (pose_weight <= 1.0).all()
        ):
            raise ValueError("pose supervision weight must be finite in [0,1] with shape (B,)")
    if dense_deformation_supervision_weight is None:
        dense_weight = torch.ones(
            batch,
            device=truth_state.device,
            dtype=truth_state.dtype,
        )
    else:
        dense_weight = torch.as_tensor(
            dense_deformation_supervision_weight,
            device=truth_state.device,
            dtype=truth_state.dtype,
        )
        if dense_weight.shape != (batch,) or not bool(
            torch.isfinite(dense_weight).all()
            and (dense_weight >= 0.0).all()
            and (dense_weight <= 1.0).all()
        ):
            raise ValueError(
                "dense deformation supervision weight must be finite in [0,1] with shape (B,)"
            )
    retrieval_nll = _weighted_mean(
        F.nll_loss(
            pose["retrieval_cell_log_probability"],
            truth_catalogue_cell_id.long(),
            reduction="none",
        ),
        pose_weight,
    )
    retained_log_mass = torch.log(
        pose["retrieval_topk_retained_probability"].clamp_min(1e-8)
    )[:, None]
    final_log_mass = (
        retained_log_mass
        + pose["conditional_within_topk_cell_log_probability"]
    )
    eligible = truth_topk_index >= 0
    point_eligible = eligible & (pose_weight > 0.0)
    dense_eligible = point_eligible & (dense_weight > 0.0)
    landmark_error_um = torch.linalg.vector_norm(
        physical_frame_landmarks(pose["final_cell_state"])
        - physical_frame_landmarks(truth_state)[:, None],
        dim=-1,
    ).mean(dim=-1)
    if bool(point_eligible.any()):
        initial_plane_nll = gaussian_plane_mixture_nll(
            pose["topk_initial_cell_state"][point_eligible],
            pose["topk_initial_cell_canonical_plane_covariance"][point_eligible],
            pose["retrieval_topk_log_probability"][point_eligible],
            truth_state[point_eligible],
            support_origin_ap_dv_ml_um,
        )
        final_landmark_nll, _ = landmark_mixture_nll(
            pose["final_cell_state"][point_eligible],
            final_log_mass[point_eligible],
            truth_state[point_eligible],
            landmark_scale_um,
        )
        final_plane_nll = gaussian_plane_mixture_nll(
            pose["final_cell_state"][point_eligible],
            pose["final_cell_canonical_plane_covariance"][point_eligible],
            final_log_mass[point_eligible],
            truth_state[point_eligible],
            support_origin_ap_dv_ml_um,
        )
    else:
        zero = pose["final_cell_state"].sum() * 0.0
        initial_plane_nll = zero
        final_landmark_nll = zero
        final_plane_nll = zero
    selected_velocity = _gather_mode(
        output["stationary_velocity_yx_px_sequence"], truth_topk_index
    )
    selected_map = _gather_mode(
        output["pullback_map_yx_px_sequence"], truth_topk_index
    )
    selected_support_logits = _gather_mode(
        output["support_logits_sequence"], truth_topk_index
    )
    selected_jacobian = _gather_mode(
        output["forward_jacobian_determinant_sequence"], truth_topk_index
    )
    selected_cycle_forward = _gather_mode(
        output["forward_then_inverse_error_yx_sequence"], truth_topk_index
    )
    selected_cycle_inverse = _gather_mode(
        output["inverse_then_forward_error_yx_sequence"], truth_topk_index
    )
    selected_cycle_forward_valid = _gather_mode(
        output["forward_then_inverse_valid_mask_sequence"], truth_topk_index
    )
    selected_cycle_inverse_valid = _gather_mode(
        output["inverse_then_forward_valid_mask_sequence"], truth_topk_index
    )
    active = torch.as_tensor(
        output["deformation_active_sequence"],
        device=selected_velocity.device,
        dtype=selected_velocity.dtype,
    )
    iteration = torch.arange(
        active.numel(), device=active.device, dtype=active.dtype
    )
    sequence_weight = active * RECURRENT_DEFORMATION_LOSS_GAMMA ** (
        active.numel() - 1 - iteration
    )
    sequence_weight = sequence_weight.reshape(1, -1, 1, 1, 1)
    dense_gate = (
        dense_weight.to(selected_velocity)
        * dense_eligible.to(selected_velocity)
    ).reshape(batch, 1, 1, 1, 1)
    weight = (
        deformation_weight.to(selected_velocity)[:, None]
        * dense_gate
        * sequence_weight
    )
    vector_weight = weight.expand(-1, -1, 2, -1, -1)
    deformation_svf = _weighted_mean(
        F.smooth_l1_loss(
            selected_velocity,
            truth_stationary_velocity_yx_px[:, None].expand_as(selected_velocity),
            beta=0.5,
            reduction="none",
        ),
        vector_weight.expand_as(selected_velocity),
    )
    deformation_map = _weighted_mean(
        F.smooth_l1_loss(
            selected_map,
            truth_pullback_map_yx_px[:, None].expand_as(selected_map),
            beta=0.5,
            reduction="none",
        ),
        vector_weight.expand_as(selected_map),
    )
    support = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            selected_support_logits,
            deformation_weight.to(selected_support_logits)[:, None]
            .expand_as(selected_support_logits)
            .clamp(0.0, 1.0),
            reduction="none",
        ),
        dense_gate.to(selected_support_logits) * sequence_weight,
    )
    topology = _weighted_mean(
        F.relu(0.05 - selected_jacobian).square(), weight
    )
    velocity_difference_y = selected_velocity[..., 1:, :] - selected_velocity[..., :-1, :]
    velocity_difference_x = selected_velocity[..., :, 1:] - selected_velocity[..., :, :-1]
    smoothness = 0.5 * (
        _weighted_mean(
            velocity_difference_y.square(),
            vector_weight[..., 1:, :] * vector_weight[..., :-1, :],
        )
        + _weighted_mean(
            velocity_difference_x.square(),
            vector_weight[..., :, 1:] * vector_weight[..., :, :-1],
        )
    )
    inverse_consistency = 0.5 * (
        _weighted_mean(
            selected_cycle_forward.square(),
            vector_weight * selected_cycle_forward_valid.to(vector_weight),
        )
        + _weighted_mean(
            selected_cycle_inverse.square(),
            vector_weight * selected_cycle_inverse_valid.to(vector_weight),
        )
    )
    total = (
        retrieval_nll
        + 0.1 * initial_plane_nll
        + 0.5 * final_plane_nll
        + final_landmark_nll
        + 0.5 * deformation_svf
        + 0.25 * deformation_map
        + 0.05 * support
        + 0.1 * topology
        + 0.05 * smoothness
        + 0.05 * inverse_consistency
    )
    return {
        "total": total,
        "retrieval_nll": retrieval_nll,
        "initial_plane_mixture_nll": initial_plane_nll,
        "final_plane_mixture_nll": final_plane_nll,
        "final_landmark_mixture_nll": final_landmark_nll,
        "deformation_svf": deformation_svf,
        "deformation_map": deformation_map,
        "deformation_support": support,
        "deformation_topology": topology,
        "deformation_smoothness": smoothness,
        "deformation_inverse_consistency": inverse_consistency,
        "mean_best_topk_landmark_error_um": _weighted_mean(
            landmark_error_um.min(dim=1).values,
            pose_weight,
        ),
        "pose_identifiable_fraction": (pose_weight > 0.0).to(truth_state.dtype).mean(),
        "deformation_eligible_fraction": dense_eligible.to(truth_state.dtype).mean(),
    }
