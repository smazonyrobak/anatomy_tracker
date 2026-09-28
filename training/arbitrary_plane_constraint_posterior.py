"""Conditional physical evidence for a full cell/representation hypothesis table."""

import torch

from training.arbitrary_plane_deformation_primitives import (
    sampling_map_in_bounds_yx, warp_tensor_with_map_yx,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_joint_constraints import (
    attack_angle_factor, electrode_ray_factor, entry_position_factor,
    hard_constraint_factors,
)


def constraint_conditioned_posterior(
    image_log_probability, cell_states, representation_to_canonical_raster_affine,
    raster_shape_h_w, *, source_points_yx=None, source_to_fixed_yx_px=None,
    mark_available=None, point_sigma_um=None, entry_ap_dv_ml_um=None,
    direction_ap_dv_ml=None, depth_bounds_um=None, measured_entry_ap_ml_um=None,
    entry_covariance_um2=None, measured_angle_deg=None, angle_sigma_deg=None,
    azimuth_deg=None, direction_concentration=None, hard_entry_center_ap_ml_um=None,
    hard_entry_radius_um=None, hard_attack_angle_bounds_deg=None,
    plane_domain_normal=None, plane_reference_ap_dv_ml_um=None,
    plane_offset_bounds_um=None,
):
    """Condition normalized image log probabilities [B,C,R] on an explicit ray.

    One B row is one animal hypothesis table, not one independently conditioned
    slice. Frames are [B,S,C,12], affines [B,S,C,R,2,3], source marks [B,S,P,2],
    and optional cell-shared source-to-fixed maps [B,S,C,2,H,W]. The identity
    map is used when absent. C/R columns must identify coherent joint hypotheses
    across S sections; unrelated section marginals must NOT be multiplied here.
    The input image prior contains the image evidence and prior mass only once.

    Ray entry/direction [B,C,R,3] and finite depth bounds [B,C,R,2] describe one
    shared trajectory hypothesis across sections. Nothing fixes or estimates an
    unknown ray implicitly: joint inference must marginalize these hypotheses.
    Missing surgical evidence leaves the image posterior exactly unchanged when
    marks/plane bounds are also absent. This is not a calibration guarantee.

    Mark availability and Gaussian sigma are [B,S,P]. Entry/angle measurements
    have B-leading dimensions (or are shared constants); plane bounds/reference
    have B,S-leading dimensions. Hard radius/tolerance are never Gaussian sigma.
    Each observed mark contributes once; each surgical factor contributes once
    per animal hypothesis, not per mark or section. Duplicated marks are not
    independent observations. Do not feed this posterior back as the image prior
    on subsequent recurrent steps: recompute factors from the original prior.

    Mapping order is source pixel -> cell warp -> exact representation affine
    -> canonical frame. Affines use normalized x/y, align_corners=False, as the
    renderer; physical coordinates then use x/W,y/H. Mirrored modes stay separate.
    An available mark mapped outside the raster makes that hypothesis infeasible;
    it is not silently discarded. Outputs include per-section/per-mode/per-mark
    residuals for recurrence and separate shared factors for audit.
    """
    batch, cells, representations = image_log_probability.shape
    sections = cell_states.shape[1]
    zero = torch.zeros_like(image_log_probability)
    result = {"log_probability": image_log_probability,
              "section_mark_log_likelihood": zero[:, None].expand(-1, sections, -1, -1),
              "shared_surgery_log_likelihood": zero,
              "point_residual": None, "points_ap_dv_ml_um": None,
              "mark_valid": None, "entry_residual": None, "angle_residual": None,
              "hard_violations": {}, "feasible": torch.isfinite(image_log_probability),
              "all_infeasible": ~torch.isfinite(image_log_probability).flatten(1).any(-1),
              "constraint_log_evidence": zero[:, 0, 0]}
    if all(value is None for value in (
        source_points_yx, measured_entry_ap_ml_um, measured_angle_deg,
        hard_entry_center_ap_ml_um, hard_attack_angle_bounds_deg, plane_offset_bounds_um,
    )):
        return result
    if any(value is not None for value in (
        source_points_yx, measured_entry_ap_ml_um, measured_angle_deg,
        hard_entry_center_ap_ml_um, hard_attack_angle_bounds_deg,
    )) and (entry_ap_dv_ml_um is None or direction_ap_dv_ml is None):
        raise ValueError("Physical ray evidence requires explicit entry and direction hypotheses")
    tensor_options = {"device": cell_states.device, "dtype": cell_states.dtype}
    feasible = torch.ones_like(image_log_probability, dtype=torch.bool)
    if source_points_yx is not None:
        height, width = raster_shape_h_w
        points = source_points_yx
        available = torch.ones_like(points[..., 0], dtype=torch.bool) if mark_available is None else mark_available
        points = torch.where(available[..., None], points, 0)
        count = points.shape[-2]
        query = points[:, :, None].expand(-1, -1, cells, -1, -1).reshape(-1, count, 2).transpose(1, 2)[:, :, None]
        sampled = query if source_to_fixed_yx_px is None else warp_tensor_with_map_yx(
            source_to_fixed_yx_px.reshape(-1, 2, height, width), query, padding_mode="border",
        )
        valid = sampling_map_in_bounds_yx(query, (height, width)) & sampling_map_in_bounds_yx(sampled, (height, width))
        warped_xy = sampled[:, :, 0].transpose(1, 2).reshape(batch, sections, cells, count, 2).flip(-1)
        size = points.new_tensor([width, height])
        affine = representation_to_canonical_raster_affine
        # Conjugate align_corners=False into pixel coordinates. Exact signed
        # reflections become x -> W-1-x, without endpoint roundoff from [-1,1].
        linear = affine[..., :2] * (size[:, None] / size[None, :])
        translation = (size * affine[..., 2] + size - 1 - (linear @ (size - 1)[..., None]).squeeze(-1)) / 2
        canonical_xy = (linear[..., None, :, :] @ warped_xy[:, :, :, None, :, :, None]).squeeze(-1) + translation[..., None, :]
        canonical_valid = torch.isfinite(canonical_xy).all(-1) & (canonical_xy >= 0).all(-1) & (canonical_xy <= size - 1).all(-1)
        valid = valid.reshape(batch, sections, cells, 1, count) & canonical_valid
        observed = available[:, :, None, None, :]
        feasible = feasible & (valid | ~observed).all(-1).all(1)
        center, frame, basis = full_frame_state_to_components(cell_states)
        physical = normalized_raster_to_ccf(
            center[..., None, None, :], frame[..., None, None, :, :],
            basis[..., None, None, :, :], canonical_xy / size,
        )
        sigma = torch.as_tensor(point_sigma_um, **tensor_options)
        sigma = torch.broadcast_to(sigma, (batch, sections, count))[:, :, None, None, :]
        marks = electrode_ray_factor(
            physical, entry_ap_dv_ml_um[:, None], direction_ap_dv_ml[:, None],
            sigma, depth_bounds_um[:, None], available=valid & observed,
        )
        result.update(section_mark_log_likelihood=marks["log_likelihood"],
                      point_residual=marks["residual"], points_ap_dv_ml_um=physical,
                      mark_valid=valid & observed)
    if measured_entry_ap_ml_um is not None:
        entry = entry_position_factor(
            entry_ap_dv_ml_um,
            torch.as_tensor(measured_entry_ap_ml_um, **tensor_options)[..., None, None, :],
            torch.as_tensor(entry_covariance_um2, **tensor_options)[..., None, None, :, :],
        )
        result["shared_surgery_log_likelihood"] = result["shared_surgery_log_likelihood"] + entry["log_likelihood"]
        result["entry_residual"] = entry["residual"]
    if measured_angle_deg is not None:
        angle = attack_angle_factor(
            direction_ap_dv_ml,
            torch.as_tensor(measured_angle_deg, **tensor_options)[..., None, None],
            None if angle_sigma_deg is None else torch.as_tensor(angle_sigma_deg, **tensor_options)[..., None, None],
            azimuth_deg=None if azimuth_deg is None else torch.as_tensor(azimuth_deg, **tensor_options)[..., None, None],
            direction_concentration=None if direction_concentration is None else torch.as_tensor(direction_concentration, **tensor_options)[..., None, None],
        )
        result["shared_surgery_log_likelihood"] = result["shared_surgery_log_likelihood"] + angle["log_likelihood"]
        result["angle_residual"] = angle["residual"]
    surgery_hard = hard_constraint_factors(
        cell_states[:, 0, :, None], entry_ap_dv_ml_um=entry_ap_dv_ml_um,
        direction_ap_dv_ml=direction_ap_dv_ml,
        entry_center_ap_ml_um=None if hard_entry_center_ap_ml_um is None else torch.as_tensor(hard_entry_center_ap_ml_um, **tensor_options)[..., None, None, :],
        entry_radius_um=None if hard_entry_radius_um is None else torch.as_tensor(hard_entry_radius_um, **tensor_options)[..., None, None],
        attack_angle_bounds_deg=None if hard_attack_angle_bounds_deg is None else torch.as_tensor(hard_attack_angle_bounds_deg, **tensor_options)[..., None, None, :],
    )
    plane_hard = hard_constraint_factors(
        cell_states,
        plane_domain_normal=None if plane_domain_normal is None else torch.as_tensor(plane_domain_normal, **tensor_options)[..., None, :],
        plane_reference_ap_dv_ml_um=None if plane_reference_ap_dv_ml_um is None else torch.as_tensor(plane_reference_ap_dv_ml_um, **tensor_options)[..., None, :],
        plane_offset_bounds_um=None if plane_offset_bounds_um is None else torch.as_tensor(plane_offset_bounds_um, **tensor_options)[..., None, :],
    )
    feasible = feasible & surgery_hard["feasible"] & plane_hard["feasible"].all(1)[..., None]
    logits = image_log_probability + result["section_mark_log_likelihood"].sum(1) + result["shared_surgery_log_likelihood"]
    logits = logits.masked_fill(~feasible, -torch.inf)
    any_finite = torch.isfinite(logits).flatten(1).any(-1)
    safe_logits = torch.where(any_finite[:, None, None], logits, 0)
    log_normalizer = torch.logsumexp(safe_logits.flatten(1), -1)
    result.update(
        log_probability=torch.where(any_finite[:, None, None], logits - log_normalizer[:, None, None], -torch.inf),
        constraint_log_evidence=torch.where(any_finite, log_normalizer, -torch.inf),
        feasible=feasible & torch.isfinite(image_log_probability), all_infeasible=~any_finite,
        hard_violations={"shared_surgery": surgery_hard["violations"], "section_plane": plane_hard["violations"]},
    )
    return result
