import numpy as np
import pytest

from source.probe_constraints import (
    InfeasibleProbeConstraint,
    ProbeInsertionConstraint,
    ProbeRayFit,
    SlicePlane,
    atlas_points_to_stereotaxic_um,
    attack_angle_deg,
    direction_from_attack_angle,
    fit_probe_ray,
    insertion_plan_plane_feasibility,
    score_candidate_slice_plane,
    stereotaxic_to_volume,
    volume_to_stereotaxic_um,
)


BREGMA = np.asarray([216.0, 332.0 / 25.0, 5739.0 / 25.0])
SHAPE = (528, 320, 456)


def ray_points(entry, angle, azimuth, depths):
    direction = direction_from_attack_angle(angle, azimuth)
    return np.asarray(entry) + np.asarray(depths)[:, None] * direction, direction


def plane_through(point_um, tilt_lr=0.0, tilt_dv=0.0):
    ap, dv, ml = stereotaxic_to_volume(point_um, BREGMA)
    index = (
        ap
        - np.tan(np.deg2rad(tilt_lr)) * (ml - (SHAPE[2] - 1.0) / 2.0)
        - np.tan(np.deg2rad(tilt_dv)) * (dv - (SHAPE[1] - 1.0) / 2.0)
    )
    return SlicePlane(index, tilt_lr, tilt_dv), np.asarray([ml, dv])


def test_insertion_plan_filters_only_planes_the_allowed_shank_can_reach():
    constraint = ProbeInsertionConstraint(
        enabled=True,
        ap_um=-1400.0,
        ml_um=-1600.0,
        radius_um=0.0,
        angle_deg=90.0,
        angle_tolerance_deg=0.0,
        maximum_insertion_depth_um=1000.0,
    )
    surface_dv = lambda _ap, _ml: 0.0

    intersecting, _ = plane_through([-1400.0, -500.0, -1600.0])
    unreachable, _ = plane_through([-3000.0, -500.0, -1600.0])

    assert insertion_plan_plane_feasibility(
        constraint, intersecting, BREGMA, SHAPE, surface_dv
    )["feasible"]
    assert not insertion_plan_plane_feasibility(
        constraint, unreachable, BREGMA, SHAPE, surface_dv
    )["feasible"]


def test_coordinate_conversion_is_exact_and_uses_tracker_axis_signs():
    voxels = BREGMA + np.asarray([[-40.0, -20.0, 12.0], [5.5, 7.25, -3.0]])
    stereotaxic = volume_to_stereotaxic_um(voxels, BREGMA)
    assert stereotaxic[0] == pytest.approx([1000.0, 500.0, 300.0])
    assert stereotaxic_to_volume(stereotaxic, BREGMA) == pytest.approx(voxels)


def test_disabled_constraint_is_a_strict_noop_and_invalid_inputs_fail_explicitly():
    disabled = ProbeInsertionConstraint()
    assert fit_probe_ray({}, disabled, lambda _ap, _ml: 0.0) is None
    with pytest.raises(InfeasibleProbeConstraint, match="At least two"):
        fit_probe_ray(
            {"slice": np.zeros((1, 3))},
            ProbeInsertionConstraint(enabled=True),
            lambda _ap, _ml: 0.0,
        )
    with pytest.raises(InfeasibleProbeConstraint, match="Attack angle"):
        fit_probe_ray(
            {"slice": np.zeros((2, 3))},
            ProbeInsertionConstraint(enabled=True, angle_deg=91.0),
            lambda _ap, _ml: 0.0,
        )


def test_missing_or_rejected_surface_fails_explicitly():
    points = np.asarray([[0.0, -100.0, 0.0], [0.0, -500.0, 0.0]])
    constraint = ProbeInsertionConstraint(enabled=True, angle_deg=90.0)
    with pytest.raises(InfeasibleProbeConstraint, match="valid cortical entry"):
        fit_probe_ray({"slice": points}, constraint, lambda _ap, _ml: np.nan)
    with pytest.raises(InfeasibleProbeConstraint, match="valid cortical entry"):
        fit_probe_ray(
            {"slice": points},
            constraint,
            lambda _ap, _ml: 0.0,
            surface_is_valid=lambda _ap, _dv, _ml: False,
        )


@pytest.mark.parametrize("angle,azimuth", [(0.0, -45.0), (35.0, 120.0), (90.0, 0.0)])
def test_attack_angle_convention_has_stable_horizontal_and_vertical_edges(angle, azimuth):
    direction = direction_from_attack_angle(angle, azimuth)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[1] <= 0.0
    assert attack_angle_deg(direction) == pytest.approx(angle)


@pytest.mark.parametrize("angle,azimuth", [(0.0, 32.0), (68.0, -25.0), (90.0, 73.0)])
def test_exact_ray_fit_recovers_horizontal_oblique_and_vertical_tracks(angle, azimuth):
    entry = np.asarray([-1400.0, -120.0, 800.0])
    points, expected_direction = ray_points(entry, angle, azimuth, [150.0, 700.0, 1300.0, 2100.0])
    fit = fit_probe_ray(
        {"a": points[:2], "b": points[2:]},
        ProbeInsertionConstraint(
            enabled=True,
            ap_um=entry[0],
            ml_um=entry[2],
            radius_um=0.0,
            angle_deg=angle,
            angle_tolerance_deg=0.0,
            maximum_insertion_depth_um=2500.0,
        ),
        lambda _ap, _ml: entry[1],
        robust_scale_um=25.0,
    )
    assert fit.entry_ap_dv_ml_um == pytest.approx(entry, abs=1e-5)
    assert fit.direction_ap_dv_ml == pytest.approx(expected_direction, abs=1e-5)
    assert fit.angle_deg == pytest.approx(angle, abs=1e-5)
    assert fit.diagnostics["median_orthogonal_residual_um"] < 1e-4
    assert fit.azimuth_deg is None if angle == 90.0 else fit.azimuth_deg == pytest.approx(azimuth % 360)


def test_disk_is_a_bound_not_a_center_prior():
    center = np.asarray([-1300.0, 700.0])
    entry = np.asarray([-1450.0, -80.0, 900.0])
    points, _ = ray_points(entry, 72.0, 18.0, [400.0, 900.0, 1500.0, 2300.0])
    fit = fit_probe_ray(
        {0: points[:2], 1: points[2:]},
        ProbeInsertionConstraint(True, center[0], center[1], 260.0, 72.0, 0.0, 3000.0),
        lambda _ap, _ml: entry[1],
        robust_scale_um=25.0,
    )
    assert fit.entry_ap_dv_ml_um == pytest.approx(entry, abs=1e-3)
    assert fit.diagnostics["entry_disk_distance_um"] == pytest.approx(250.0, abs=1e-3)


def test_robust_equal_slice_fit_tolerates_noise_outlier_and_unequal_click_counts():
    rng = np.random.default_rng(1841)
    entry = np.asarray([-1600.0, -110.0, 650.0])
    direction = direction_from_attack_angle(74.0, 35.0)
    many = entry + np.linspace(300.0, 2300.0, 40)[:, None] * direction
    few = entry + np.asarray([500.0, 1400.0, 2200.0])[:, None] * direction
    many += rng.normal(0.0, 18.0, many.shape)
    few += rng.normal(0.0, 18.0, few.shape)
    many[7] += np.asarray([350.0, -250.0, 300.0])
    fit = fit_probe_ray(
        {"many": many, "few": few},
        ProbeInsertionConstraint(True, -1550.0, 700.0, 180.0, 74.0, 4.0, 2800.0),
        lambda _ap, _ml: entry[1],
        robust_scale_um=45.0,
    )
    assert np.linalg.norm(fit.entry_ap_dv_ml_um - entry) < 80.0
    assert np.rad2deg(np.arccos(np.clip(fit.direction_ap_dv_ml @ direction, -1.0, 1.0))) < 2.0
    assert fit.diagnostics["outlier_count"] >= 1
    assert set(fit.diagnostics["per_slice_rms_um"]) == {"many", "few"}


def test_robust_fit_ignores_isolated_axial_click_outliers():
    entry = np.asarray([-1600.0, -110.0, 650.0])
    points, direction = ray_points(entry, 74.0, 35.0, [300.0, 700.0, 1200.0, 1800.0, 2300.0])
    points = np.vstack((points, entry - 500.0 * direction, entry + 4000.0 * direction))
    fit = fit_probe_ray(
        {"a": points[:4], "b": points[4:]},
        ProbeInsertionConstraint(True, entry[0], entry[2], 0.0, 74.0, 0.0, 2800.0),
        lambda _ap, _ml: entry[1],
        robust_scale_um=30.0,
    )
    assert fit.entry_ap_dv_ml_um == pytest.approx(entry, abs=1e-3)
    direction_error = np.rad2deg(
        np.arccos(np.clip(fit.direction_ap_dv_ml @ direction, -1.0, 1.0))
    )
    assert direction_error < 0.25
    assert fit.diagnostics["outlier_count"] == 2
    assert fit.diagnostics["axial_max_um"] == pytest.approx(2300.0, abs=0.05)


def test_wrong_target_angle_and_length_constraints_are_rejected():
    entry = np.asarray([-1200.0, -100.0, 500.0])
    points, _ = ray_points(entry, 70.0, 20.0, [100.0, 500.0, 900.0])
    with pytest.raises(InfeasibleProbeConstraint, match="inconsistent"):
        fit_probe_ray(
            {0: points},
            ProbeInsertionConstraint(True, 200.0, -1000.0, 50.0, 70.0, 0.0, None),
            lambda _ap, _ml: entry[1],
            robust_scale_um=25.0,
        )
    with pytest.raises(InfeasibleProbeConstraint, match="inconsistent"):
        fit_probe_ray(
            {0: points},
            ProbeInsertionConstraint(True, entry[0], entry[2], 0.0, 20.0, 0.0, None),
            lambda _ap, _ml: entry[1],
            robust_scale_um=25.0,
        )
    beyond, _ = ray_points(entry, 70.0, 20.0, [100.0, 500.0, 1150.0])
    with pytest.raises(InfeasibleProbeConstraint, match="inconsistent"):
        fit_probe_ray(
            {0: beyond},
            ProbeInsertionConstraint(True, entry[0], entry[2], 0.0, 70.0, 0.0, 1000.0),
            lambda _ap, _ml: entry[1],
            robust_scale_um=20.0,
        )


def test_physical_depth_cannot_be_overridden_by_a_minority_of_clicks():
    entry = np.asarray([-1200.0, -100.0, 500.0])
    points, _ = ray_points(entry, 70.0, 20.0, [100.0, 300.0, 600.0, 900.0, 5000.0])
    with pytest.raises(InfeasibleProbeConstraint, match="physical insertion segment"):
        fit_probe_ray(
            {"slice": points},
            ProbeInsertionConstraint(True, entry[0], entry[2], 0.0, 70.0, 0.0, 1000.0),
            lambda _ap, _ml: entry[1],
            robust_scale_um=20.0,
        )


def test_every_contributing_slice_requires_compatible_probe_points():
    entry = np.asarray([-1200.0, -100.0, 500.0])
    many, _ = ray_points(entry, 70.0, 20.0, np.linspace(100.0, 900.0, 40))
    few = many[:2] + np.asarray([0.0, 0.0, 2000.0])
    with pytest.raises(InfeasibleProbeConstraint, match="inconsistent|Slice few"):
        fit_probe_ray(
            {"many": many, "few": few},
            ProbeInsertionConstraint(True, entry[0], entry[2], 0.0, 70.0, 0.0, 1000.0),
            lambda _ap, _ml: entry[1],
            robust_scale_um=25.0,
        )


def test_slice_plane_equation_matches_tracker_point_to_volume_property():
    rng = np.random.default_rng(7391)
    for _ in range(200):
        plane = SlicePlane(
            rng.uniform(20.0, SHAPE[0] - 20.0),
            rng.uniform(-35.0, 35.0),
            rng.uniform(-35.0, 35.0),
        )
        ml = rng.uniform(0.0, SHAPE[2] - 1.0)
        dv = rng.uniform(0.0, SHAPE[1] - 1.0)
        ap = (
            plane.ap_index
            + np.tan(np.deg2rad(plane.tilt_lr_deg)) * (ml - (SHAPE[2] - 1.0) / 2.0)
            + np.tan(np.deg2rad(plane.tilt_dv_deg)) * (dv - (SHAPE[1] - 1.0) / 2.0)
        )
        point = volume_to_stereotaxic_um(np.asarray([ap, dv, ml]), BREGMA)
        assert atlas_points_to_stereotaxic_um([[ml, dv]], plane, BREGMA, SHAPE)[0] == pytest.approx(point)


def test_candidate_scoring_prefers_true_ordered_planes_to_ap_decoys():
    entry = np.asarray([-1100.0, -90.0, 600.0])
    points, direction = ray_points(entry, 67.0, 28.0, [500.0, 1000.0, 1600.0])
    fit = ProbeRayFit(entry, direction, 67.0, 28.0, 0.0, {"maximum_insertion_depth_um": 2200.0})
    true_scores = []
    decoy_scores = []
    indices = []
    for point in points:
        plane, atlas_point = plane_through(point, tilt_lr=7.0, tilt_dv=-5.0)
        indices.append(plane.ap_index)
        true_scores.append(score_candidate_slice_plane([atlas_point], plane, fit, BREGMA, SHAPE)["score"])
        decoy = SlicePlane(plane.ap_index + 10.0, plane.tilt_lr_deg, plane.tilt_dv_deg)
        decoy_scores.append(score_candidate_slice_plane([atlas_point], decoy, fit, BREGMA, SHAPE)["score"])
    assert max(true_scores) < 1e-20
    assert min(decoy_scores) > max(true_scores)
    assert np.all(np.diff(indices) > 0.0) or np.all(np.diff(indices) < 0.0)


def test_candidate_scoring_rejects_points_outside_atlas_brain():
    entry = np.asarray([-1100.0, -90.0, 600.0])
    point, direction = ray_points(entry, 67.0, 28.0, [500.0])
    fit = ProbeRayFit(entry, direction, 67.0, 28.0, 0.0, {"maximum_insertion_depth_um": 2200.0})
    plane, atlas_point = plane_through(point[0], tilt_lr=7.0, tilt_dv=-5.0)
    mask = np.ones((SHAPE[1], SHAPE[2]), dtype=bool)
    assert score_candidate_slice_plane(
        [atlas_point], plane, fit, BREGMA, SHAPE, brain_mask=mask
    )["feasible"]
    x, y = np.rint(atlas_point).astype(int)
    mask[y, x] = False
    assert not score_candidate_slice_plane(
        [atlas_point], plane, fit, BREGMA, SHAPE, brain_mask=mask
    )["feasible"]
