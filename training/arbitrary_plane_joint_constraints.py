"""Differentiable physical constraint factors; all distances are micrometres."""

import math

import torch

from training.arbitrary_plane_deformation_primitives import (
    sampling_map_in_bounds_yx, warp_tensor_with_map_yx,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_geometry import normalized_raster_to_ccf


def marked_points_to_ccf(source_points, source_to_fixed_yx_px, frame_state, *, order="yx"):
    """Map source marks [...,P,2] through [...,2,H,W] pullback and [...,12] frame.

    Source-to-fixed means the map queried at a source pixel RETURNS its fixed
    atlas-raster y,x. Pixel centres are 0..H-1,0..W-1; physical s,t use x/W,y/H
    exactly as the atlas renderer. Return CCF AP,DV,ML points and a validity mask;
    callers must omit invalid marks, never use padded values as observations.
    """
    points = source_points.flip(-1) if order == "xy" else source_points
    leading = torch.broadcast_shapes(points.shape[:-2], source_to_fixed_yx_px.shape[:-3], frame_state.shape[:-1])
    height, width = source_to_fixed_yx_px.shape[-2:]
    count = points.shape[-2]
    query = points.expand(*leading, count, 2).reshape(-1, count, 2).transpose(1, 2)[:, :, None]
    mapping = source_to_fixed_yx_px.expand(*leading, 2, height, width).reshape(-1, 2, height, width)
    sampled = warp_tensor_with_map_yx(mapping, query, padding_mode="border")
    valid = sampling_map_in_bounds_yx(query, (height, width)) & sampling_map_in_bounds_yx(sampled, (height, width))
    fixed_yx = sampled[:, :, 0].transpose(1, 2).reshape(*leading, count, 2)
    st = fixed_yx.flip(-1) / fixed_yx.new_tensor([width, height])
    center, frame, basis = full_frame_state_to_components(frame_state.expand(*leading, 12))
    physical = normalized_raster_to_ccf(center[..., None, :], frame[..., None, :, :], basis[..., None, :, :], st)
    return {"points_ap_dv_ml_um": physical, "valid_mask": valid.reshape(*leading, count), "fixed_yx_px": fixed_yx}


def electrode_ray_factor(points_ap_dv_ml_um, entry_ap_dv_ml_um, direction_ap_dv_ml,
                         point_sigma_um, depth_bounds_um, *, available=None):
    """Point likelihood conditional on an explicit shared entry/direction ray.

    Points [...,P,3], entry/direction [...,3], sigma [...,P], bounds [...,2].
    Unit direction points from entry toward tip. Unknown depth is integrated
    exactly under a uniform prior on finite bounds 0 <= a < b. Noise is isotropic
    Gaussian in physical 3-D coordinates. Marks are conditionally independent;
    multiply this factor once per observation, not again as surgical metadata.
    Missing points return zero; available excludes missing/invalid marks.
    """
    if points_ap_dv_ml_um is None:
        return {"log_likelihood": entry_ap_dv_ml_um[..., 0] * 0, "residual": None}
    direction = direction_ap_dv_ml / torch.linalg.vector_norm(direction_ap_dv_ml, dim=-1, keepdim=True)
    mask = torch.ones_like(points_ap_dv_ml_um[..., 0], dtype=torch.bool) if available is None else available
    points = torch.where(mask[..., None], points_ap_dv_ml_um, entry_ap_dv_ml_um[..., None, :])
    delta = points - entry_ap_dv_ml_um[..., None, :]
    depth = (delta * direction[..., None, :]).sum(-1)
    perpendicular = delta - depth[..., None] * direction[..., None, :]
    sigma = torch.as_tensor(point_sigma_um, device=points.device, dtype=points.dtype)
    lower, upper = torch.as_tensor(depth_bounds_um, device=points.device, dtype=points.dtype).unbind(-1)
    lo, hi = (lower[..., None] - depth) / sigma, (upper[..., None] - depth) / sigma
    # Reflect positive-tail intervals to avoid subtracting two CDFs near one.
    left, right = torch.where(lo > 0, -hi, lo), torch.where(lo > 0, -lo, hi)
    log_left, log_right = torch.special.log_ndtr(left), torch.special.log_ndtr(right)
    log_mass = log_right + torch.log(-torch.expm1(log_left - log_right))
    per_point = (-0.5 * (perpendicular / sigma[..., None]).square().sum(-1)
                 - math.log(2 * math.pi) - 2 * sigma.log()
                 - (upper - lower).log()[..., None] + log_mass)
    closest_depth = torch.maximum(lower[..., None], torch.minimum(upper[..., None], depth))
    residual = (delta - closest_depth[..., None] * direction[..., None, :]) / sigma[..., None]
    return {"log_likelihood": torch.where(mask, per_point, 0).sum(-1),
            "residual": torch.where(mask[..., None], residual, 0), "point_depth_um": depth}


def entry_position_factor(entry_ap_dv_ml_um, measured_ap_ml_um=None, covariance_um2=None):
    """Normalized AP/ML Gaussian, with measured coordinates already in CCF axes.

    Covariance [...,2,2] is positive definite; a GUI hard radius is NOT a sigma.
    Bregma-to-CCF conversion and its uncertainty belong to the caller.
    """
    if measured_ap_ml_um is None:
        return {"log_likelihood": entry_ap_dv_ml_um[..., 0] * 0, "residual": None}
    covariance = torch.as_tensor(covariance_um2, device=entry_ap_dv_ml_um.device, dtype=entry_ap_dv_ml_um.dtype)
    chol = torch.linalg.cholesky(covariance)
    delta = entry_ap_dv_ml_um[..., (0, 2)] - measured_ap_ml_um
    residual = torch.linalg.solve_triangular(chol, delta[..., None], upper=False).squeeze(-1)
    loglike = -0.5 * residual.square().sum(-1) - chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) - math.log(2 * math.pi)
    return {"log_likelihood": loglike, "residual": residual}


def attack_angle_factor(direction_ap_dv_ml, measured_angle_deg=None, sigma_deg=None,
                        *, azimuth_deg=None, direction_concentration=None):
    """Elevation is 0 horizontal, +90 ventral (+DV), -90 dorsal; azimuth AP->ML.

    With no azimuth, Gaussian measurement noise concerns elevation in radians.
    With azimuth, use normalized S2 vMF and explicit concentration kappa > 0;
    sigma_deg is unused. Hard GUI angle tolerances never enter this function.
    """
    if measured_angle_deg is None:
        return {"log_likelihood": direction_ap_dv_ml[..., 0] * 0, "residual": None}
    direction = direction_ap_dv_ml / torch.linalg.vector_norm(direction_ap_dv_ml, dim=-1, keepdim=True)
    angle = torch.deg2rad(torch.as_tensor(measured_angle_deg, device=direction.device, dtype=direction.dtype))
    if azimuth_deg is None:
        elevation = torch.atan2(direction[..., 1], torch.linalg.vector_norm(direction[..., (0, 2)], dim=-1))
        sigma = torch.deg2rad(torch.as_tensor(sigma_deg, device=direction.device, dtype=direction.dtype))
        residual = (elevation - angle) / sigma
        return {"log_likelihood": -0.5 * residual.square() - sigma.log() - 0.5 * math.log(2 * math.pi), "residual": residual[..., None]}
    azimuth = torch.deg2rad(torch.as_tensor(azimuth_deg, device=direction.device, dtype=direction.dtype))
    angle, azimuth = torch.broadcast_tensors(angle, azimuth)
    mean = torch.stack((angle.cos() * azimuth.cos(), angle.sin(), angle.cos() * azimuth.sin()), -1)
    kappa = torch.as_tensor(direction_concentration, device=direction.device, dtype=direction.dtype)
    log_normalizer = kappa.log() - math.log(2 * math.pi) - kappa - torch.log(-torch.expm1(-2 * kappa))
    return {"log_likelihood": log_normalizer + kappa * (direction * mean).sum(-1),
            "residual": kappa.sqrt()[..., None] * (direction - mean)}


def hard_constraint_factors(frame_state, *, entry_ap_dv_ml_um=None, direction_ap_dv_ml=None,
                           entry_center_ap_ml_um=None, entry_radius_um=None,
                           attack_angle_bounds_deg=None, plane_domain_normal=None,
                           plane_reference_ap_dv_ml_um=None, plane_offset_bounds_um=None):
    """Exact inclusive hard bounds: return feasibility, 0/-inf mask and violations.

    Plane offset is dot(frame centre - reference, normalized declared normal).
    It bounds centre position along that declared axis, not the candidate's own
    normal or orientation. Entry disks are AP/ML; angles follow attack_angle_factor.
    All vectors use physical CCF AP,DV,ML. Absent constraints contribute nothing.
    """
    center = frame_state[..., :3]
    feasible = torch.ones_like(center[..., 0], dtype=torch.bool)
    violations = {}
    if entry_center_ap_ml_um is not None:
        distance = torch.linalg.vector_norm(entry_ap_dv_ml_um[..., (0, 2)] - entry_center_ap_ml_um, dim=-1)
        violations["entry_um"] = (distance - entry_radius_um).clamp_min(0)
        feasible = feasible & (distance <= entry_radius_um)
    if attack_angle_bounds_deg is not None:
        angle = torch.rad2deg(torch.atan2(direction_ap_dv_ml[..., 1], torch.linalg.vector_norm(direction_ap_dv_ml[..., (0, 2)], dim=-1)))
        bounds = torch.as_tensor(attack_angle_bounds_deg, device=center.device, dtype=center.dtype)
        violations["attack_deg"] = torch.maximum(bounds[..., 0] - angle, angle - bounds[..., 1]).clamp_min(0)
        feasible = feasible & (angle >= bounds[..., 0]) & (angle <= bounds[..., 1])
    if plane_offset_bounds_um is not None:
        normal = torch.as_tensor(plane_domain_normal, device=center.device, dtype=center.dtype)
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        offset = ((center - plane_reference_ap_dv_ml_um) * normal).sum(-1)
        bounds = torch.as_tensor(plane_offset_bounds_um, device=center.device, dtype=center.dtype)
        violations["plane_offset_um"] = torch.maximum(bounds[..., 0] - offset, offset - bounds[..., 1]).clamp_min(0)
        feasible = feasible & (offset >= bounds[..., 0]) & (offset <= bounds[..., 1])
    return {"feasible": feasible, "log_mask": torch.where(feasible, center[..., 0] * 0, -torch.inf), "violations": violations}
