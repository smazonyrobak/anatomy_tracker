from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares


AXIS_SIGN_AP_DV_ML = np.asarray((-1.0, -1.0, 1.0), dtype=np.float64)


class InfeasibleProbeConstraint(ValueError):
    pass


@dataclass(frozen=True)
class ProbeInsertionConstraint:
    enabled: bool = False
    ap_um: float = 0.0
    ml_um: float = 0.0
    radius_um: float = 0.0
    angle_deg: float = 90.0
    angle_tolerance_deg: float = 0.0
    maximum_insertion_depth_um: float | None = None


@dataclass(frozen=True)
class SlicePlane:
    ap_index: float
    tilt_lr_deg: float = 0.0
    tilt_dv_deg: float = 0.0


@dataclass(frozen=True)
class ProbeRayFit:
    entry_ap_dv_ml_um: np.ndarray
    direction_ap_dv_ml: np.ndarray
    angle_deg: float
    azimuth_deg: float | None
    loss: float
    diagnostics: dict


def volume_to_stereotaxic_um(
    volume_points: np.ndarray,
    bregma_voxel: np.ndarray,
    voxel_um: float = 25.0,
) -> np.ndarray:
    return (
        (np.asarray(volume_points, dtype=np.float64) - np.asarray(bregma_voxel, dtype=np.float64))
        * float(voxel_um)
        * AXIS_SIGN_AP_DV_ML
    )


def stereotaxic_to_volume(
    stereotaxic_points: np.ndarray,
    bregma_voxel: np.ndarray,
    voxel_um: float = 25.0,
) -> np.ndarray:
    return (
        np.asarray(stereotaxic_points, dtype=np.float64)
        / (float(voxel_um) * AXIS_SIGN_AP_DV_ML)
        + np.asarray(bregma_voxel, dtype=np.float64)
    )


def direction_from_attack_angle(angle_deg: float, azimuth_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    azimuth = np.deg2rad(float(azimuth_deg))
    return np.asarray(
        [np.cos(angle) * np.cos(azimuth), -np.sin(angle), np.cos(angle) * np.sin(azimuth)],
        dtype=np.float64,
    )


def insertion_plan_plane_feasibility(
    constraint: ProbeInsertionConstraint,
    plane: SlicePlane,
    bregma_voxel: np.ndarray,
    volume_shape: Sequence[int],
    surface_dv: Callable[[float, float], float],
    voxel_um: float = 25.0,
    prepared_entries: np.ndarray | None = None,
) -> dict:
    """Test whether any allowed insertion ray can intersect a candidate slice plane."""
    angle_low, angle_high = _constraint_bounds(constraint)
    maximum_depth = float(constraint.maximum_insertion_depth_um or 10000.0)
    if prepared_entries is None:
        prepared_entries = prepare_insertion_surface_entries(constraint, surface_dv)

    tangent_lr = np.tan(np.deg2rad(float(plane.tilt_lr_deg)))
    tangent_dv = np.tan(np.deg2rad(float(plane.tilt_dv_deg)))
    normal = np.asarray([-1.0, tangent_dv, -tangent_lr], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    constant = float(voxel_um) * (
        float(bregma_voxel[0])
        - float(plane.ap_index)
        - tangent_lr * (float(bregma_voxel[2]) - (float(volume_shape[2]) - 1.0) / 2.0)
        - tangent_dv * (float(bregma_voxel[1]) - (float(volume_shape[1]) - 1.0) / 2.0)
    ) / np.linalg.norm(np.asarray([-1.0, tangent_dv, -tangent_lr], dtype=np.float64))

    angles = np.deg2rad(np.linspace(angle_low, angle_high, max(2, int(np.ceil(angle_high - angle_low)) + 1)))
    horizontal_normal = float(np.hypot(normal[0], normal[2]))
    derivative_center = -normal[1] * np.sin(angles)
    derivative_radius = horizontal_normal * np.cos(angles)
    minimum_derivative = float(np.min(derivative_center - derivative_radius))
    maximum_derivative = float(np.max(derivative_center + derivative_radius))

    required_depths = []
    valid_entries = 0
    for ap_um, dv_um, ml_um in np.asarray(prepared_entries, dtype=np.float64).reshape(-1, 3):
        valid_entries += 1
        signed_distance = float(normal @ np.asarray([ap_um, dv_um, ml_um]) + constant)
        if abs(signed_distance) <= float(voxel_um) / 2.0:
            required_depths.append(0.0)
        elif signed_distance > 0.0 and minimum_derivative < -1e-9:
            required_depths.append(signed_distance / -minimum_derivative)
        elif signed_distance < 0.0 and maximum_derivative > 1e-9:
            required_depths.append(-signed_distance / maximum_derivative)
    minimum_depth = min(required_depths, default=float("inf"))
    return {
        "feasible": bool(valid_entries and minimum_depth <= maximum_depth),
        "minimum_required_depth_um": float(minimum_depth),
        "maximum_depth_um": maximum_depth,
        "valid_surface_entry_samples": int(valid_entries),
    }


def prepare_insertion_surface_entries(
    constraint: ProbeInsertionConstraint,
    surface_dv: Callable[[float, float], float],
) -> np.ndarray:
    """Sample the insertion disk once; candidate planes reuse these surface points."""
    _constraint_bounds(constraint)
    phases = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    entries = [(constraint.ap_um, constraint.ml_um)]
    for fraction in (0.5, 1.0):
        entries.extend(
            (
                constraint.ap_um + constraint.radius_um * fraction * np.cos(phase),
                constraint.ml_um + constraint.radius_um * fraction * np.sin(phase),
            )
            for phase in phases
        )
    sampled = [
        (float(ap), float(dv), float(ml))
        for ap, ml in entries
        if np.isfinite(dv := surface_dv(float(ap), float(ml)))
    ]
    return np.asarray(sampled, dtype=np.float64).reshape(-1, 3)


def fit_observed_probe_ray(
    observations_by_slice: Mapping[Hashable, np.ndarray],
    constraint: ProbeInsertionConstraint,
    surface_dv: Callable[[float, float], float],
    *,
    robust_scale_um: float = 50.0,
    axial_tolerance_um: float = 25.0,
) -> ProbeRayFit:
    """Validate the ordinary displayed regression itself against hard surgical bounds."""
    angle_low, angle_high = _constraint_bounds(constraint)
    groups = {
        label: np.asarray(values, dtype=np.float64).reshape(-1, 3)
        for label, values in observations_by_slice.items()
        if len(values)
    }
    points = np.concatenate(list(groups.values())) if groups else np.empty((0, 3))
    if len(points) < 2 or not np.isfinite(points).all():
        raise InfeasibleProbeConstraint("At least two finite probe observations are required")
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    direction = axes[0]
    if direction[1] > 0.0:
        direction = -direction
    direction /= np.linalg.norm(direction)
    angle = attack_angle_deg(direction)
    tolerance = 1e-7
    if angle < angle_low - tolerance or angle > angle_high + tolerance:
        raise InfeasibleProbeConstraint(
            f"Observed attack angle {angle:.3f} deg is outside the allowed "
            f"{angle_low:.3f}-{angle_high:.3f} deg range"
        )

    projected = (points - center) @ direction
    shallowest = float(projected.min())
    maximum_depth = float(constraint.maximum_insertion_depth_um or 10000.0)
    parameters = np.linspace(shallowest - maximum_depth, shallowest + axial_tolerance_um, 257)

    def surface_offset(parameter: float) -> float:
        point = center + float(parameter) * direction
        surface = float(surface_dv(float(point[0]), float(point[2])))
        return float(point[1] - surface) if np.isfinite(surface) else float("nan")

    offsets = np.asarray([surface_offset(value) for value in parameters])
    entries = []
    for first, second, value_first, value_second in zip(
        parameters[:-1], parameters[1:], offsets[:-1], offsets[1:]
    ):
        if not np.isfinite(value_first) or not np.isfinite(value_second):
            continue
        if value_first == 0.0:
            entries.append(center + first * direction)
        elif value_first * value_second < 0.0:
            root = brentq(surface_offset, float(first), float(second), xtol=1e-6)
            entries.append(center + root * direction)
    if not entries:
        raise InfeasibleProbeConstraint("Observed trajectory does not intersect the valid cortical surface")
    entry = min(entries, key=lambda value: abs(float(((points - value) @ direction).min())))
    disk_distance = float(np.hypot(entry[0] - constraint.ap_um, entry[2] - constraint.ml_um))
    if disk_distance > constraint.radius_um + 1e-6:
        raise InfeasibleProbeConstraint(
            f"Observed cortical entry is {disk_distance:.1f} um from target, outside the "
            f"allowed {constraint.radius_um:.1f} um radius"
        )
    residuals, axial = _ray_residuals(points, entry, direction, constraint.maximum_insertion_depth_um)
    distances = np.linalg.norm(residuals, axis=1)
    if np.any(axial < -axial_tolerance_um):
        raise InfeasibleProbeConstraint("Observed points extend above the planned cortical entry")
    if constraint.maximum_insertion_depth_um is not None and np.any(
        axial > constraint.maximum_insertion_depth_um + axial_tolerance_um
    ):
        raise InfeasibleProbeConstraint("Observed points exceed the allowed insertion depth")
    if np.median(distances) > 3.0 * robust_scale_um:
        raise InfeasibleProbeConstraint("Observed points do not form one feasible probe trajectory")
    for label, group in groups.items():
        group_residuals, group_axial = _ray_residuals(
            group, entry, direction, constraint.maximum_insertion_depth_um
        )
        if not np.any(
            (np.linalg.norm(group_residuals, axis=1) <= 3.0 * robust_scale_um)
            & (group_axial >= -axial_tolerance_um)
            & (group_axial <= maximum_depth + axial_tolerance_um)
        ):
            raise InfeasibleProbeConstraint(f"Slice {label} has no observation on the feasible segment")
    diagnostics = {
        "feasible": True,
        "fit_kind": "ordinary_observed_regression_hard_bounds",
        "angle_range_deg": [float(angle_low), float(angle_high)],
        "entry_disk_distance_um": disk_distance,
        "entry_disk_slack_um": float(constraint.radius_um - disk_distance),
        "angle_slack_deg": float(min(angle - angle_low, angle_high - angle)),
        "maximum_insertion_depth_um": constraint.maximum_insertion_depth_um,
        "axial_min_um": float(axial.min()),
        "axial_max_um": float(axial.max()),
        "median_orthogonal_residual_um": float(np.median(distances)),
    }
    return ProbeRayFit(
        entry,
        direction,
        float(angle),
        float(np.rad2deg(np.arctan2(direction[2], direction[0])) % 360.0),
        float(np.mean((distances / robust_scale_um) ** 2)),
        diagnostics,
    )


def attack_angle_deg(direction_ap_dv_ml: np.ndarray) -> float:
    direction = np.asarray(direction_ap_dv_ml, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    if direction[1] > 0.0:
        direction = -direction
    return float(np.rad2deg(np.arcsin(np.clip(-direction[1], 0.0, 1.0))))


def _constraint_bounds(constraint: ProbeInsertionConstraint) -> tuple[float, float]:
    values = np.asarray(
        [constraint.ap_um, constraint.ml_um, constraint.radius_um, constraint.angle_deg,
         constraint.angle_tolerance_deg],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or constraint.radius_um < 0.0 or constraint.angle_tolerance_deg < 0.0:
        raise InfeasibleProbeConstraint("Insertion constraint contains invalid values")
    low = max(0.0, constraint.angle_deg - constraint.angle_tolerance_deg)
    high = min(90.0, constraint.angle_deg + constraint.angle_tolerance_deg)
    if low > high or constraint.angle_deg < 0.0 or constraint.angle_deg > 90.0:
        raise InfeasibleProbeConstraint("Attack angle must use the 0 horizontal / 90 vertical convention")
    if constraint.maximum_insertion_depth_um is not None and (
        not np.isfinite(constraint.maximum_insertion_depth_um)
        or constraint.maximum_insertion_depth_um <= 0.0
    ):
        raise InfeasibleProbeConstraint("Maximum insertion depth must be positive")
    return low, high


def _ray_residuals(
    points: np.ndarray,
    entry: np.ndarray,
    direction: np.ndarray,
    maximum_depth_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    delta = points - entry
    axial = delta @ direction
    upper = np.inf if maximum_depth_um is None else float(maximum_depth_um)
    closest_axial = np.clip(axial, 0.0, upper)
    return delta - closest_axial[:, None] * direction, axial


def fit_probe_ray(
    observations_by_slice: Mapping[Hashable, np.ndarray],
    constraint: ProbeInsertionConstraint,
    surface_dv: Callable[[float, float], float],
    *,
    surface_is_valid: Callable[[float, float, float], bool] | None = None,
    robust_scale_um: float = 50.0,
    axial_tolerance_um: float = 25.0,
    max_starts: int | None = None,
) -> ProbeRayFit | None:
    """Fit a dorsal-surface-to-deep ray in stereotaxic ``(AP,DV,ML)`` microns.

    Each slice is normalized by its click count and must retain compatible
    observations after robust fitting. Disabled constraints deliberately
    return ``None`` and perform no fitting.
    """
    if not constraint.enabled:
        return None
    angle_low, angle_high = _constraint_bounds(constraint)
    if not np.isfinite(robust_scale_um) or robust_scale_um <= 0.0:
        raise ValueError("robust_scale_um must be positive")
    if not np.isfinite(axial_tolerance_um) or axial_tolerance_um < 0.0:
        raise ValueError("axial_tolerance_um must be non-negative")

    groups = []
    labels = []
    for label, values in observations_by_slice.items():
        points = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        if len(points):
            if not np.isfinite(points).all():
                raise InfeasibleProbeConstraint("Probe observations must be finite")
            groups.append(points)
            labels.append(label)
    if sum(map(len, groups)) < 2:
        raise InfeasibleProbeConstraint("At least two probe observations are required")
    points = np.concatenate(groups)

    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    seed_direction = axes[0]
    if seed_direction[1] > 0.0:
        seed_direction = -seed_direction
    seed_angle = float(np.clip(attack_angle_deg(seed_direction), angle_low, angle_high))
    seed_azimuth = float(np.rad2deg(np.arctan2(seed_direction[2], seed_direction[0])))

    radius = float(constraint.radius_um)

    def geometry(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        radial_fraction, entry_phase, angle, azimuth = parameters
        ap = constraint.ap_um + radius * radial_fraction * np.cos(entry_phase)
        ml = constraint.ml_um + radius * radial_fraction * np.sin(entry_phase)
        dv = float(surface_dv(float(ap), float(ml)))
        if not np.isfinite(dv):
            dv = 1e12
        return np.asarray([ap, dv, ml]), direction_from_attack_angle(angle, azimuth), angle, azimuth

    def residuals(parameters: np.ndarray) -> np.ndarray:
        entry, direction, _, _ = geometry(parameters)
        residual_groups = []
        for group in groups:
            residual, _ = _ray_residuals(
                group, entry, direction, constraint.maximum_insertion_depth_um
            )
            residual_groups.append(
                residual.reshape(-1) / (robust_scale_um * np.sqrt(len(group)))
            )
        return np.concatenate(residual_groups)

    azimuths = (seed_azimuth, seed_azimuth + 180.0)
    starts = [np.asarray([0.0, 0.0, seed_angle, seed_azimuth])]
    if radius > 0.0:
        phases = np.linspace(-np.pi, np.pi, 4, endpoint=False)
        starts.extend(
            np.asarray([1.0, phase, seed_angle, seed_azimuth], dtype=np.float64)
            for phase in phases
        )
    starts.append(np.asarray([0.0, 0.0, seed_angle, seed_azimuth + 180.0]))
    if radius > 0.0:
        starts.extend(
            np.asarray([radial, phase, seed_angle, azimuth], dtype=np.float64)
            for radial in (1.0, 0.65)
            for azimuth in azimuths
            for phase in phases
            if radial < 1.0 or azimuth != seed_azimuth
        )
    if max_starts is not None:
        starts = starts[:max_starts]
    lower = np.asarray([0.0, -2.0 * np.pi, angle_low, -360.0])
    upper = np.asarray([1.0, 2.0 * np.pi, angle_high, 360.0])
    equal = lower == upper
    lower[equal] -= 1e-10
    upper[equal] += 1e-10
    results = [
        least_squares(
            residuals,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=300,
        )
        for start in starts
    ]
    result = min(results, key=lambda item: item.cost)
    entry, direction, angle, azimuth = geometry(result.x)
    surface_value = float(surface_dv(float(entry[0]), float(entry[2])))
    if not np.isfinite(surface_value) or not np.isfinite(entry).all() or (
        surface_is_valid is not None and not surface_is_valid(*map(float, entry))
    ):
        raise InfeasibleProbeConstraint("No valid cortical entry exists inside the target disk")

    residual_vectors, axial = _ray_residuals(
        points, entry, direction, constraint.maximum_insertion_depth_um
    )
    maximum = constraint.maximum_insertion_depth_um
    distances = np.linalg.norm(residual_vectors, axis=1)
    inlier_limit = 3.0 * robust_scale_um
    within_segment = (axial >= -axial_tolerance_um) & (
        True
        if maximum is None
        else axial <= float(maximum) + axial_tolerance_um
    )
    inliers = (distances <= inlier_limit) & within_segment
    if np.count_nonzero(inliers) < max(2, int(np.ceil(0.7 * len(points)))) or np.median(distances) > inlier_limit:
        raise InfeasibleProbeConstraint(
            "Probe observations are inconsistent with the insertion constraints"
        )
    inlier_axial = axial[inliers]

    per_slice = {}
    offset = 0
    for label, group in zip(labels, groups):
        selected = distances[offset : offset + len(group)]
        selected_axial = axial[offset : offset + len(group)]
        selected_within_segment = (selected_axial >= -axial_tolerance_um) & (
            True
            if maximum is None
            else selected_axial <= float(maximum) + axial_tolerance_um
        )
        if not np.any(selected_within_segment & (selected <= inlier_limit)):
            raise InfeasibleProbeConstraint(
                f"Slice {label} has no probe observation compatible with the surgical constraints"
            )
        compatible = selected_within_segment & (selected <= inlier_limit)
        if not np.any(compatible):
            raise InfeasibleProbeConstraint(
                f"Slice {label} is inconsistent with the fitted probe trajectory"
            )
        if len(group) >= 5 and np.count_nonzero(compatible) <= int(np.floor(0.8 * len(group))):
            raise InfeasibleProbeConstraint(
                f"Slice {label} contains probe observations outside the physical insertion segment"
            )
        per_slice[str(label)] = float(np.sqrt(np.mean(selected**2)))
        offset += len(group)
    disk_distance = float(np.hypot(entry[0] - constraint.ap_um, entry[2] - constraint.ml_um))
    depth = constraint.maximum_insertion_depth_um
    diagnostics = {
        "feasible": True,
        "slice_count": len(groups),
        "point_count": len(points),
        "median_orthogonal_residual_um": float(np.median(distances)),
        "rms_orthogonal_residual_um": float(np.sqrt(np.mean(distances**2))),
        "p95_orthogonal_residual_um": float(np.percentile(distances, 95.0)),
        "per_slice_rms_um": per_slice,
        "outlier_count": int(np.count_nonzero(~inliers)),
        "axial_min_um": float(inlier_axial.min()),
        "axial_max_um": float(inlier_axial.max()),
        "axial_span_um": float(np.ptp(inlier_axial)),
        "entry_disk_distance_um": disk_distance,
        "entry_disk_slack_um": float(radius - disk_distance),
        "angle_target_deg": float(constraint.angle_deg),
        "angle_tolerance_deg": float(constraint.angle_tolerance_deg),
        "angle_slack_deg": float(min(angle - angle_low, angle_high - angle)),
        "maximum_insertion_depth_um": None if depth is None else float(depth),
        "at_depth_bound": bool(depth is not None and inlier_axial.max() >= depth - robust_scale_um),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "azimuth_identifiable": bool(angle < 89.9),
    }
    return ProbeRayFit(
        entry_ap_dv_ml_um=entry,
        direction_ap_dv_ml=direction,
        angle_deg=float(angle),
        azimuth_deg=(float(azimuth % 360.0) if angle < 89.9 else None),
        loss=float(result.cost),
        diagnostics=diagnostics,
    )


def atlas_points_to_stereotaxic_um(
    atlas_points_xy: np.ndarray,
    plane: SlicePlane,
    bregma_voxel: np.ndarray,
    volume_shape: Sequence[int],
    voxel_um: float = 25.0,
) -> np.ndarray:
    points = np.asarray(atlas_points_xy, dtype=np.float64).reshape(-1, 2)
    ml, dv = points[:, 0], points[:, 1]
    ap = (
        float(plane.ap_index)
        + np.tan(np.deg2rad(float(plane.tilt_lr_deg))) * (ml - (float(volume_shape[2]) - 1.0) / 2.0)
        + np.tan(np.deg2rad(float(plane.tilt_dv_deg))) * (dv - (float(volume_shape[1]) - 1.0) / 2.0)
    )
    return volume_to_stereotaxic_um(np.column_stack((ap, dv, ml)), bregma_voxel, voxel_um)


def score_candidate_slice_plane(
    atlas_points_xy: np.ndarray,
    plane: SlicePlane,
    fit: ProbeRayFit,
    bregma_voxel: np.ndarray,
    volume_shape: Sequence[int],
    *,
    voxel_um: float = 25.0,
    robust_scale_um: float = 50.0,
    axial_tolerance_um: float = 25.0,
    brain_mask: np.ndarray | None = None,
) -> dict:
    points = atlas_points_to_stereotaxic_um(
        atlas_points_xy, plane, bregma_voxel, volume_shape, voxel_um
    )
    residuals, axial = _ray_residuals(
        points,
        fit.entry_ap_dv_ml_um,
        fit.direction_ap_dv_ml,
        fit.diagnostics.get("maximum_insertion_depth_um"),
    )
    distances = np.linalg.norm(residuals, axis=1)
    maximum = fit.diagnostics.get("maximum_insertion_depth_um")
    within_segment = (axial >= -float(axial_tolerance_um)) & (
        True if maximum is None else axial <= float(maximum) + float(axial_tolerance_um)
    )
    inside_brain = np.ones(len(points), dtype=bool)
    if brain_mask is not None and len(points):
        mask = np.asarray(brain_mask, dtype=bool)
        indices = np.rint(np.asarray(atlas_points_xy, dtype=np.float64)).astype(int)
        inside_bounds = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < mask.shape[1])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < mask.shape[0])
        )
        inside_brain = np.zeros(len(points), dtype=bool)
        inside_brain[inside_bounds] = mask[
            indices[inside_bounds, 1],
            indices[inside_bounds, 0],
        ]
    normalized = distances / float(robust_scale_um)
    robust = 2.0 * (np.sqrt(1.0 + normalized**2) - 1.0)
    return {
        "feasible": bool(
            len(points)
            and np.all(within_segment)
            and np.all(inside_brain)
            and np.median(distances) <= 3.0 * robust_scale_um
        ),
        "score": float(np.mean(robust)) if len(points) else float("inf"),
        "median_orthogonal_residual_um": float(np.median(distances)) if len(points) else float("inf"),
        "rms_orthogonal_residual_um": float(np.sqrt(np.mean(distances**2))) if len(points) else float("inf"),
        "axial_min_um": float(axial.min()) if len(points) else float("nan"),
        "axial_max_um": float(axial.max()) if len(points) else float("nan"),
    }
