import hashlib
import inspect

import numpy as np
import pytest
from scipy.linalg import expm

from training.arbitrary_plane_subject_deformation_v2 import (
    SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS,
    SUBJECT_DEFORMATION_V2_RK4_ORIENTATION_CERTIFICATE,
    _candidate_audit,
    _combined_cubic_bspline_bounds,
    _domain_id,
    _json_value,
    _rk4_step_map_orientation_certificate,
    apply_positive_diagonal_scale,
    ccf_to_subject_points_v2,
    compose_diagonal_scale_svf_forward,
    compose_diagonal_scale_svf_inverse,
    cubic_bspline_basis,
    cubic_bspline_velocity,
    derive_subject_deformation_seed_v2,
    integrate_stationary_velocity,
    replay_animal_subject_deformation_plan_v2,
    sample_animal_subject_deformation_plan_v2,
    subject_deformation_plan_receipt_v2,
    subject_to_ccf_points_v2,
    verify_subject_deformation_plan_v2,
)


LOWER = np.asarray([0.0, 0.0, 0.0])
UPPER = np.asarray([1000.0, 900.0, 800.0])
CCF_CONTEXT = hashlib.sha256(b"frozen full CCF context").hexdigest()


def _callable_affine_velocity(matrix, offset):
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)

    def field(points, *, return_gradient=False):
        points = np.asarray(points)
        velocity = points @ matrix.T + offset
        if not return_gradient:
            return velocity
        gradient = np.broadcast_to(matrix, points.shape[:-1] + (3, 3)).copy()
        return velocity, gradient

    return field


def _plan(**overrides):
    parameters = {
        "root_seed": "0x415154564f320001",
        "split": "development",
        "animal_index": 3,
        "animal_id": "animal-003",
        "ccf_context_sha256": CCF_CONTEXT,
        "coarse_spacing_um": 500.0,
        "fine_spacing_um": 250.0,
        "coarse_padding_um": 2000.0,
        "fine_padding_um": 1000.0,
        "a0_um": 25.0,
        "minimum_halo_um": 0.0,
    }
    parameters.update(overrides)
    return sample_animal_subject_deformation_plan_v2(LOWER, UPPER, **parameters)


@pytest.fixture(scope="module")
def standard_plan():
    return _plan()


def test_subject_rng_is_length_prefixed_domain_separated_and_label_independent():
    seed = derive_subject_deformation_seed_v2(
        "0x415154564f320001",
        "development",
        3,
        "subject-deformation",
        "animal-realization",
        "coarse-cubic-bspline-svf",
        0,
    )

    assert seed == 9653814047834250562
    assert seed != derive_subject_deformation_seed_v2(
        "0x415154564f320001", "development", 4,
        "subject-deformation", "animal-realization", "coarse-cubic-bspline-svf", 0,
    )
    assert derive_subject_deformation_seed_v2(
        7, "train", 2, "ab", "c", "d", 0
    ) != derive_subject_deformation_seed_v2(
        7, "train", 2, "a", "bc", "d", 0
    )
    assert "animal_id" not in inspect.signature(derive_subject_deformation_seed_v2).parameters


def test_cubic_bspline_basis_partitions_unity_and_derivative_zero():
    t = np.linspace(0.0, 1.0, 101, endpoint=False)
    weights, derivatives = cubic_bspline_basis(t)

    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=3e-16, rtol=0.0)
    assert np.allclose(derivatives.sum(axis=-1), 0.0, atol=2e-16, rtol=0.0)


def test_cubic_bspline_reproduces_affine_fields_and_analytic_gradient():
    shape = (10, 11, 12)
    origin = np.asarray([-7.0, 13.0, 31.0])
    spacing = np.asarray([1.5, 2.0, 2.5])
    coordinates = np.stack(
        np.meshgrid(
            *[origin[axis] + spacing[axis] * np.arange(shape[axis]) for axis in range(3)],
            indexing="ij",
        ),
        -1,
    )
    matrix = np.asarray(
        [[0.07, -0.03, 0.02], [0.01, 0.04, -0.05], [-0.02, 0.06, 0.03]]
    )
    offset = np.asarray([4.0, -7.0, 11.0])
    coefficients = coordinates @ matrix.T + offset
    points = origin + np.asarray([[2.2, 3.1, 4.4], [4.7, 5.5, 6.2]]) * spacing

    velocity, gradient = cubic_bspline_velocity(
        points, coefficients, origin, spacing, return_gradient=True
    )

    assert np.allclose(velocity, points @ matrix.T + offset, atol=2e-13, rtol=0.0)
    assert np.allclose(gradient, matrix, atol=2e-14, rtol=0.0)


def test_combined_coefficient_bounds_cover_speed_derivatives_gradient_and_divergence():
    rng = np.random.Generator(np.random.PCG64DXSM(71))
    coarse = rng.normal(size=(8, 8, 8, 3)).astype("<f4")
    fine = rng.normal(size=(12, 12, 12, 3)).astype("<f4")
    coarse_origin = np.asarray([-2.0, -2.0, -2.0])
    fine_origin = np.asarray([-1.0, -1.0, -1.0])
    coarse_spacing = np.asarray([1.0, 1.0, 1.0])
    fine_spacing = np.asarray([0.5, 0.5, 0.5])
    points = rng.uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0], size=(400, 3))
    coarse_value, coarse_gradient = cubic_bspline_velocity(
        points, coarse, coarse_origin, coarse_spacing, return_gradient=True
    )
    fine_value, fine_gradient = cubic_bspline_velocity(
        points, fine, fine_origin, fine_spacing, return_gradient=True
    )
    value = coarse_value + fine_value
    gradient = coarse_gradient + fine_gradient
    bounds = _combined_cubic_bspline_bounds(
        coarse, coarse_spacing, fine, fine_spacing
    )

    assert np.all(
        np.abs(value) <= np.asarray(bounds["component_speed_abs_bound_um"]) + 1e-12
    )
    assert np.all(
        np.abs(gradient)
        <= np.asarray(bounds["component_derivative_abs_bound"]) + 1e-12
    )
    assert np.max(np.linalg.norm(value, axis=1)) <= bounds["speed_l2_bound_um"] + 1e-12
    assert np.max(np.linalg.norm(gradient, axis=(1, 2))) <= bounds[
        "gradient_frobenius_bound"
    ] + 1e-12
    assert np.max(np.abs(np.trace(gradient, axis1=1, axis2=2))) <= bounds[
        "divergence_abs_bound"
    ] + 1e-12


def test_rk4_step_map_orientation_certificate_accepts_safe_and_rejects_unsafe_bounds():
    safe = _rk4_step_map_orientation_certificate(2.0, 8)
    unsafe = _rk4_step_map_orientation_certificate(2.0, 1)

    assert safe["rk4_step_gradient_product_bound"] == 0.25
    assert safe["rk4_step_map_jacobian_perturbation_bound"] == (
        0.25 + 0.25**2 / 2.0 + 0.25**3 / 6.0 + 0.25**4 / 24.0
    )
    assert safe["rk4_step_map_orientation_margin"] > 0.0
    assert safe["rk4_step_map_orientation_certified"]
    assert unsafe["rk4_step_map_jacobian_perturbation_bound"] == 6.0
    assert unsafe["rk4_step_map_orientation_margin"] < 0.0
    assert not unsafe["rk4_step_map_orientation_certified"]


def test_candidate_rejects_an_uncertified_rk4_step_map():
    shape = (12, 12, 12)
    origin = np.full(3, -6.0)
    spacing = np.ones(3)
    parity = np.indices(shape).sum(axis=0) % 2
    coarse = np.repeat((2.0 * parity - 1.0)[..., None], 3, axis=-1).astype("<f4")
    fine = np.zeros_like(coarse)
    modes = np.zeros((12,) + shape + (3,), dtype=np.float64)
    grid = np.stack(
        np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    limits = {
        "local_jacobian_det_min": 1.0e-9,
        "local_jacobian_det_max": 1.0e9,
        "composed_jacobian_det_floor": 1.0e-9,
        "cycle_max_um": 1.0e9,
        "max_local_displacement_um": 1.0e9,
        "component_derivative_abs_max": 1.0e9,
        "gradient_frobenius_bound": 1.0e9,
        "divergence_abs_bound": 1.0e9,
        "speed_l2_bound_um": 1.0e9,
        "rk4_step_map_jacobian_perturbation_bound": 1.0,
        "physical_affine_residual_max_um": 1.0e9,
        "minimum_halo_um": 0.0,
    }

    audit, _, _, _ = _candidate_audit(
        grid,
        grid,
        np.ones(3),
        coarse,
        modes,
        np.eye(12),
        origin,
        spacing,
        fine,
        origin,
        spacing,
        1.0,
        np.ones(3),
        np.zeros(3),
        1,
        limits,
    )

    assert audit["gate_values"]["rk4_step_map_jacobian_perturbation_bound"] > 1.0
    assert "rk4_step_map_orientation_certificate" in audit["failed_gates"]
    assert not audit["accepted"]


def test_fixed_rk4_zero_constant_and_linear_flows_have_variational_jacobians():
    points = np.asarray([[1.0, -2.0, 3.0], [5.0, 7.0, -11.0]])
    zero = _callable_affine_velocity(np.zeros((3, 3)), np.zeros(3))
    constant = _callable_affine_velocity(np.zeros((3, 3)), [2.0, -3.0, 5.0])
    matrix = np.asarray(
        [[0.02, -0.05, 0.01], [0.05, -0.01, 0.02], [0.0, -0.02, 0.03]]
    )
    linear = _callable_affine_velocity(matrix, np.zeros(3))

    identity_points, identity_jacobian = integrate_stationary_velocity(
        points, zero, return_jacobian=True
    )
    translated, translated_jacobian = integrate_stationary_velocity(
        points, constant, return_jacobian=True
    )
    mapped, jacobian = integrate_stationary_velocity(points, linear, return_jacobian=True)
    expected_jacobian = expm(matrix)

    assert np.array_equal(identity_points, points)
    assert np.array_equal(identity_jacobian, np.broadcast_to(np.eye(3), (2, 3, 3)))
    assert np.allclose(translated, points + [2.0, -3.0, 5.0], atol=2e-14, rtol=0.0)
    assert np.array_equal(translated_jacobian, np.broadcast_to(np.eye(3), (2, 3, 3)))
    assert np.allclose(mapped, points @ expected_jacobian.T, atol=2e-10, rtol=0.0)
    assert np.allclose(jacobian, expected_jacobian, atol=2e-11, rtol=0.0)


def test_positive_diagonal_scale_composition_has_correct_noncommuting_inverse():
    matrix = np.asarray([[0.01, 0.03, 0.0], [-0.02, 0.0, 0.01], [0.0, -0.01, 0.02]])
    field = _callable_affine_velocity(matrix, np.zeros(3))
    scale = np.asarray([1.04, 0.97, 1.02])
    center = np.asarray([20.0, 10.0, -5.0])
    points = np.asarray([[100.0, -30.0, 70.0], [-20.0, 40.0, 90.0]])
    forward, forward_jacobian = compose_diagonal_scale_svf_forward(
        points, field, scale, center, return_jacobian=True
    )
    inverse, inverse_jacobian = compose_diagonal_scale_svf_inverse(
        forward, field, scale, center, return_jacobian=True
    )

    assert np.allclose(inverse, points, atol=2e-9, rtol=0.0)
    assert np.allclose(
        np.matmul(inverse_jacobian, forward_jacobian), np.eye(3), atol=3e-10, rtol=0.0
    )
    assert np.allclose(
        apply_positive_diagonal_scale(
            apply_positive_diagonal_scale(points, scale, center), scale, center, inverse=True
        ),
        points,
        atol=5e-15,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="positive"):
        apply_positive_diagonal_scale(points, [-1.0, 1.0, 1.0], center)


def test_plan_uses_only_full_ccf_fixed_grids_and_post_float32_affine_gate(standard_plan):
    plan = standard_plan
    state = plan["state"]
    projection = state["projection_grid_um"]
    design = np.column_stack(
        (
            np.ones(len(projection)),
            (projection - (LOWER + UPPER) / 2.0) / ((UPPER - LOWER) / 2.0),
        )
    )
    combined = cubic_bspline_velocity(
        projection,
        state["projected_coarse_unit_coefficients"],
        state["coarse_origin_um"],
        state["coarse_spacing_um"],
    ) + cubic_bspline_velocity(
        projection,
        state["projected_fine_unit_coefficients"],
        state["fine_origin_um"],
        state["fine_spacing_um"],
    )
    residual = np.linalg.lstsq(design, combined, rcond=None)[0]
    maximum_spacing = state["fine_spacing_um"] / 2.0
    segments = np.ceil((UPPER - LOWER) / maximum_spacing).astype(int)
    expected_shape = tuple(segments + 1)
    expected_spacing = (UPPER - LOWER) / segments

    assert "brain_support_points" not in inspect.signature(
        sample_animal_subject_deformation_plan_v2
    ).parameters
    assert np.array_equal(projection.min(axis=0), LOWER)
    assert np.array_equal(projection.max(axis=0), UPPER)
    assert np.array_equal(state["audit_grid_um"].min(axis=0), LOWER)
    assert np.array_equal(state["audit_grid_um"].max(axis=0), UPPER)
    assert projection.shape == (int(np.prod(expected_shape)), 3)
    assert tuple(plan["resolved_config"]["projection_grid_shape"]) == expected_shape
    assert tuple(plan["resolved_config"]["audit_grid_shape"]) == expected_shape
    assert np.array_equal(state["grid_maximum_spacing_um"], maximum_spacing)
    assert np.array_equal(state["projection_grid_spacing_um"], expected_spacing)
    assert np.array_equal(state["audit_grid_spacing_um"], expected_spacing)
    assert np.all(expected_spacing <= maximum_spacing)
    assert state["raw_coarse_coefficients"].dtype == np.dtype("<f4")
    assert state["projected_coarse_unit_coefficients"].dtype == np.dtype("<f4")
    assert np.max(np.abs(residual)) <= plan["resolved_config"][
        "post_float32_affine_residual_max"
    ]
    assert np.isclose(
        np.max(np.abs(residual)),
        plan["local_svf"]["projection"]["post_float32_complete_affine_residual_max"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_plan_ids_replay_and_rng_are_provenance_bound_but_label_independent(standard_plan):
    first = standard_plan
    other_label = _plan(animal_id="animal-004")
    other_context = _plan(ccf_context_sha256=hashlib.sha256(b"other CCF").hexdigest())
    other_index = _plan(animal_index=4, animal_id="animal-004")
    replay = replay_animal_subject_deformation_plan_v2(first)

    verify_subject_deformation_plan_v2(
        first,
        expected_ccf_context_sha256=CCF_CONTEXT,
        expected_full_ccf_lower_um=LOWER,
        expected_full_ccf_upper_um=UPPER,
    )
    assert subject_deformation_plan_receipt_v2(first) == subject_deformation_plan_receipt_v2(
        replay
    )
    for plan in (other_label, other_context):
        assert np.array_equal(
            first["state"]["raw_coarse_coefficients"],
            plan["state"]["raw_coarse_coefficients"],
        )
        assert np.array_equal(first["state"]["global_scale"], plan["state"]["global_scale"])
        assert first["rng_sources"] == plan["rng_sources"]
        assert first["subject_deformation_plan_id"] != plan["subject_deformation_plan_id"]
        assert first["subject_deformation_realization_id"] != plan[
            "subject_deformation_realization_id"
        ]
        assert first["synthetic_animal_id"] != plan["synthetic_animal_id"]
    assert not np.array_equal(
        first["state"]["raw_coarse_coefficients"],
        other_index["state"]["raw_coarse_coefficients"],
    )
    assert set(first["rng_sources"]) == {
        "global_scale_x", "global_scale_y", "global_scale_z", "coarse_svf", "fine_svf"
    }
    assert len(
        {first["rng_sources"][f"global_scale_{axis}"]["seed_uint64"] for axis in "xyz"}
    ) == 3
    assert first["resolved_config"]["learned_checkpoint_dependencies"] == ()
    assert first["resolved_config"]["previous_model_dependencies"] == ()
    assert first["resolved_config"]["pretrained_feature_dependencies"] == ()


def test_accepted_fixed_audit_maps_jacobians_determinants_and_cycles_are_persisted(
    standard_plan,
):
    plan = standard_plan
    state = plan["state"]
    audit = state["audit_grid_um"]
    forward, forward_jacobian = ccf_to_subject_points_v2(
        audit, plan, return_jacobian=True
    )
    inverse, inverse_jacobian = subject_to_ccf_points_v2(
        audit, plan, return_jacobian=True
    )
    forward_cycle = subject_to_ccf_points_v2(forward, plan)
    inverse_cycle = ccf_to_subject_points_v2(inverse, plan)

    assert forward.dtype.str == "<f8"
    assert inverse.dtype.str == "<f8"
    assert np.array_equal(forward, state["accepted_audit_forward_subject_um"])
    assert np.array_equal(inverse, state["accepted_audit_inverse_ccf_um"])
    assert np.array_equal(forward_jacobian, state["accepted_audit_forward_jacobian"])
    assert np.array_equal(inverse_jacobian, state["accepted_audit_inverse_jacobian"])
    assert np.allclose(
        np.linalg.det(forward_jacobian),
        state["accepted_audit_forward_jacobian_det"],
        atol=2e-15,
        rtol=0.0,
    )
    assert np.allclose(
        np.linalg.det(inverse_jacobian),
        state["accepted_audit_inverse_jacobian_det"],
        atol=2e-15,
        rtol=0.0,
    )
    assert np.array_equal(
        forward_cycle - audit,
        state["accepted_audit_forward_then_inverse_cycle_error_um"],
    )
    assert np.array_equal(
        inverse_cycle - audit,
        state["accepted_audit_inverse_then_forward_cycle_error_um"],
    )
    assert np.all(state["accepted_audit_forward_jacobian_det"] > 0.0)
    assert np.all(state["accepted_audit_inverse_jacobian_det"] > 0.0)
    accepted_index = plan["realization"]["accepted_candidate_index"]
    accepted = plan["realization"]["candidate_audits"][accepted_index]
    bounds = accepted["field_bounds"]
    gates = accepted["gate_values"]
    limits = plan["realization"]["gate_limits"]
    certificate = _rk4_step_map_orientation_certificate(
        bounds["gradient_frobenius_bound"], plan["resolved_config"]["flow"]["steps"]
    )
    assert np.array_equal(
        bounds["component_speed_abs_bound_um"],
        state["accepted_component_speed_abs_bound_um"],
    )
    assert np.array_equal(
        bounds["component_derivative_abs_bound"],
        state["accepted_component_derivative_abs_bound"],
    )
    assert np.isclose(
        gates["minimum_interpolation_halo_um"],
        gates["minimum_interpolation_start_halo_um"] - bounds["speed_l2_bound_um"],
        rtol=0.0,
        atol=1e-12,
    )
    assert plan["resolved_config"][
        "numerical_orientation_certificate"
    ] == SUBJECT_DEFORMATION_V2_RK4_ORIENTATION_CERTIFICATE
    assert limits["rk4_step_map_jacobian_perturbation_bound"] == 1.0
    assert gates["rk4_step_gradient_product_bound"] == certificate[
        "rk4_step_gradient_product_bound"
    ]
    assert gates["rk4_step_map_jacobian_perturbation_bound"] == certificate[
        "rk4_step_map_jacobian_perturbation_bound"
    ]
    assert gates["rk4_step_map_orientation_margin"] == certificate[
        "rk4_step_map_orientation_margin"
    ]
    assert certificate["rk4_step_map_orientation_certified"]
    assert "rk4_step_map_orientation_certificate" not in accepted["failed_gates"]
    for name in (
        "component_derivative_abs_max",
        "gradient_frobenius_bound",
        "divergence_abs_bound",
        "speed_l2_bound_um",
        "physical_affine_residual_max_um",
    ):
        assert gates[name] <= limits[name]
    projection = state["projection_grid_um"]
    design = np.column_stack(
        (
            np.ones(len(projection)),
            (projection - (LOWER + UPPER) / 2.0) / ((UPPER - LOWER) / 2.0),
        )
    )
    accepted_velocity = cubic_bspline_velocity(
        projection,
        state["accepted_coarse_coefficients_um"],
        state["coarse_origin_um"],
        state["coarse_spacing_um"],
    ) + cubic_bspline_velocity(
        projection,
        state["accepted_fine_coefficients_um"],
        state["fine_origin_um"],
        state["fine_spacing_um"],
    )
    final_affine_residual = np.max(
        np.abs(np.linalg.lstsq(design, accepted_velocity, rcond=None)[0])
    )
    assert np.isclose(
        final_affine_residual,
        gates["physical_affine_residual_max_um"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert plan["realization"][
        "accepted_candidate_physical_affine_residual_max_um"
    ] == gates["physical_affine_residual_max_um"]
    assert plan["realization"]["candidate_schedule_um"] == tuple(
        25.0 * factor for factor in SUBJECT_DEFORMATION_V2_CANDIDATE_FACTORS
    )


def test_standard_mapping_is_batch_and_order_invariant(standard_plan):
    rng = np.random.Generator(np.random.PCG64DXSM(19))
    points = rng.uniform(LOWER + 50.0, UPPER - 50.0, size=(17, 3))
    unbatched = ccf_to_subject_points_v2(points, standard_plan)
    batched = ccf_to_subject_points_v2(points, standard_plan, batch_size=5)
    permutation = rng.permutation(len(points))
    permuted = ccf_to_subject_points_v2(
        points[permutation], standard_plan, batch_size=4
    )
    recovered = subject_to_ccf_points_v2(unbatched, standard_plan, batch_size=6)

    assert np.array_equal(unbatched, batched)
    assert np.array_equal(unbatched[permutation], permuted)
    assert np.max(np.linalg.norm(recovered - points, axis=1)) < 1e-5


def test_identity_is_bitwise_and_an_accepted_realization_is_mandatory():
    identity = _plan(deformation_stratum="identity")
    points = np.asarray([[-0.0, 1.0, 2.0], [3.0, -0.0, 4.0]], dtype=np.float32)
    forward, jacobian = ccf_to_subject_points_v2(points, identity, return_jacobian=True)
    inverse = subject_to_ccf_points_v2(points, identity)

    assert forward.tobytes() == points.tobytes()
    assert inverse.tobytes() == points.tobytes()
    assert np.array_equal(jacobian, np.broadcast_to(np.eye(3), (2, 3, 3)))
    assert np.array_equal(identity["state"]["global_scale"], np.ones(3))
    assert not np.any(identity["state"]["accepted_coarse_coefficients_um"])
    assert identity["realization"]["accepted_amplitude_um"] == 0.0

    rejected = dict(identity)
    rejected_realization = dict(identity["realization"])
    rejected_realization["accepted_candidate_index"] = None
    rejected["realization"] = rejected_realization
    with pytest.raises(ValueError, match="accepted"):
        ccf_to_subject_points_v2(points, rejected)
    with pytest.raises(ValueError, match="accepted"):
        subject_to_ccf_points_v2(points, rejected)


def test_candidate_failure_hard_fails_and_exact_key_verifier_rejects_extras(standard_plan):
    with pytest.raises(ValueError, match="no deterministic"):
        _plan(deformation_stratum="identity", minimum_halo_um=1.0e9)

    extra = dict(standard_plan)
    extra["unauthenticated"] = 1
    with pytest.raises(ValueError, match="unauthenticated"):
        verify_subject_deformation_plan_v2(
            extra,
            expected_ccf_context_sha256=CCF_CONTEXT,
            expected_full_ccf_lower_um=LOWER,
            expected_full_ccf_upper_um=UPPER,
        )


def test_verifier_rejects_mismatched_authoritative_ccf_context_and_bounds(standard_plan):
    arguments = {
        "expected_ccf_context_sha256": CCF_CONTEXT,
        "expected_full_ccf_lower_um": LOWER,
        "expected_full_ccf_upper_um": UPPER,
    }
    with pytest.raises(ValueError, match="authoritative"):
        verify_subject_deformation_plan_v2(
            standard_plan,
            **{**arguments, "expected_ccf_context_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="authoritative"):
        verify_subject_deformation_plan_v2(
            standard_plan,
            **{**arguments, "expected_full_ccf_lower_um": LOWER + [1.0, 0.0, 0.0]},
        )
    with pytest.raises(ValueError, match="authoritative"):
        verify_subject_deformation_plan_v2(
            standard_plan,
            **{**arguments, "expected_full_ccf_upper_um": UPPER + [0.0, 0.0, 1.0]},
        )


def test_verifier_rejects_coherently_reidentified_orientation_diagnostic_tamper(
    standard_plan,
):
    tampered = dict(standard_plan)
    realization = dict(standard_plan["realization"])
    audits = list(realization["candidate_audits"])
    accepted_index = realization["accepted_candidate_index"]
    accepted = dict(audits[accepted_index])
    gate_values = dict(accepted["gate_values"])
    gate_values["rk4_step_map_orientation_margin"] += 1.0e-6
    accepted["gate_values"] = gate_values
    candidate_payload = {
        key: accepted[key]
        for key in (
            "amplitude_um",
            "gate_values",
            "failed_gates",
            "accepted",
            "coarse_coefficients_receipt",
            "fine_coefficients_receipt",
            "candidate_affine_correction_receipt",
            "field_bounds",
            "candidate_state_receipts",
        )
    }
    accepted["candidate_id"] = _domain_id(
        "anatomy-tracker.subject-deformation-candidate/v2",
        _json_value(candidate_payload),
    )
    audits[accepted_index] = accepted
    realization["candidate_audits"] = tuple(audits)
    tampered["realization"] = realization

    with pytest.raises(ValueError, match="orientation certificate"):
        verify_subject_deformation_plan_v2(
            tampered,
            expected_ccf_context_sha256=CCF_CONTEXT,
            expected_full_ccf_lower_um=LOWER,
            expected_full_ccf_upper_um=UPPER,
        )
