import numpy as np
import pytest

from training.arbitrary_plane_synthetic_ops import (
    bilinear_sample_field,
    bilinear_sample_scalar,
    compose_pixel_maps,
    fixed_source_maps,
    identity_pixel_map,
    integrate_stationary_velocity,
    jacobian_determinant,
    nearest_sample_labels,
    physical_velocity_to_pixel,
    remove_tissue_affine_component,
    sample_multiscale_physical_velocity,
    similarity_maps,
    topology_acceptance_metrics,
)


def test_arbitrary_shape_sampling_and_full_integer_labels():
    shape = (7, 11)
    identity = identity_pixel_map(shape)
    y, x = np.mgrid[: shape[0], : shape[1]]
    scalar = (3.0 * x + 5.0 * y + 7.0).astype(np.float32)
    pullback = identity[:, 1:-1, 1:-1] + np.asarray((0.25, -0.4), np.float32)[:, None, None]
    expected = 3.0 * pullback[0] + 5.0 * pullback[1] + 7.0
    assert identity.shape == (2, *shape) and identity.dtype == np.float32
    assert np.allclose(bilinear_sample_scalar(scalar, pullback), expected, atol=2e-6)

    field = np.stack((scalar, 2.0 * scalar))
    sampled_field = bilinear_sample_field(field, pullback)
    assert np.allclose(sampled_field[0], expected, atol=2e-6)
    assert np.allclose(sampled_field[1], 2.0 * expected, atol=4e-6)

    labels = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) + 2**40
    label_map = identity.copy()
    label_map[0] += 0.51
    sampled_labels = nearest_sample_labels(labels, label_map)
    assert sampled_labels.dtype == np.int64
    assert np.array_equal(sampled_labels[:, :-1], labels[:, 1:])
    assert np.all(sampled_labels[:, -1] == 0)


def test_map_composition_uses_absolute_pixel_center_semantics():
    shape = (9, 13)
    identity = identity_pixel_map(shape)
    first = identity + np.asarray((0.25, 0.5), np.float32)[:, None, None]
    second = np.empty_like(identity)
    second[0] = 2.0 * identity[0] + 0.5 * identity[1] + 1.0
    second[1] = -0.25 * identity[0] + 1.5 * identity[1] - 2.0
    composed = compose_pixel_maps(first, second)
    interior = (slice(None), slice(1, -2), slice(1, -2))
    expected = np.empty_like(first)
    expected[0] = 2.0 * first[0] + 0.5 * first[1] + 1.0
    expected[1] = -0.25 * first[0] + 1.5 * first[1] - 2.0
    assert np.allclose(composed[interior], expected[interior], atol=2e-6)


def test_physical_basis_conversion_and_affine_component_removal():
    shape = (12, 17)
    pixel_velocity = np.empty((2, *shape), np.float32)
    y, x = np.mgrid[: shape[0], : shape[1]]
    pixel_velocity[0] = 1.5 + 0.03 * x - 0.02 * y
    pixel_velocity[1] = -0.4 + 0.01 * x + 0.04 * y
    basis = np.asarray(((25.0, 6.0), (0.0, 20.0)))
    physical = (basis @ pixel_velocity.reshape(2, -1)).reshape(pixel_velocity.shape)
    assert np.allclose(
        physical_velocity_to_pixel(physical, basis), pixel_velocity, atol=2e-6
    )

    tissue = np.zeros(shape, bool)
    tissue[2:-2, 3:-3] = True
    residual = remove_tissue_affine_component(pixel_velocity, tissue)
    assert np.max(np.abs(residual)) < 2e-6
    with pytest.raises(ValueError, match="positive-orientation"):
        physical_velocity_to_pixel(physical, np.diag((25.0, -20.0)))


def test_multiscale_physical_velocity_is_explicit_deterministic_and_smooth():
    arguments = {
        "correlation_lengths_px": (2.0, 5.0),
        "rms_amplitudes_um": (12.0, 25.0),
    }
    first = sample_multiscale_physical_velocity(
        np.random.default_rng(9182), (31, 47), **arguments
    )
    repeat = sample_multiscale_physical_velocity(
        np.random.default_rng(9182), (31, 47), **arguments
    )
    changed = sample_multiscale_physical_velocity(
        np.random.default_rng(9183), (31, 47), **arguments
    )
    assert first.shape == (2, 31, 47) and first.dtype == np.float32
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, changed)
    assert np.isfinite(first).all()
    assert np.mean(np.abs(np.diff(first, axis=1))) < 0.25 * np.std(first)
    assert np.mean(np.abs(np.diff(first, axis=2))) < 0.25 * np.std(first)


def test_adaptive_stationary_velocity_exponential_uses_shared_steps():
    velocity = np.zeros((2, 8, 14), np.float32)
    velocity[0] = 4.0
    velocity[1] = -2.0
    forward, inverse, steps = integrate_stationary_velocity(velocity)
    identity = identity_pixel_map((8, 14))
    assert steps == 4
    assert np.allclose(forward, identity + velocity, atol=2e-6)
    assert np.allclose(inverse, identity - velocity, atol=2e-6)
    with pytest.raises(ValueError, match=r"\(0,0.5\]"):
        integrate_stationary_velocity(velocity, max_scaled_displacement_px=0.51)


def test_similarity_is_positive_orientation_and_exactly_inverted_in_frame():
    shape = (17, 23)
    forward, inverse = similarity_maps(
        shape, angle_rad=0.31, scale=1.2, translation_xy=(1.25, -0.75)
    )
    identity = identity_pixel_map(shape)
    valid = (
        (forward[0] >= 1.0)
        & (forward[0] <= shape[1] - 2.0)
        & (forward[1] >= 1.0)
        & (forward[1] <= shape[0] - 2.0)
    )
    cycle = compose_pixel_maps(forward, inverse)
    assert np.allclose(cycle[:, valid], identity[:, valid], atol=4e-5)
    assert np.allclose(jacobian_determinant(forward), 1.2**2, atol=5e-6)
    with pytest.raises(ValueError, match="positive scale"):
        similarity_maps(shape, angle_rad=0.0, scale=-1.0, translation_xy=(0.0, 0.0))


def test_fixed_source_pullback_and_topology_acceptance_are_two_way():
    shape = (35, 49)
    physical = sample_multiscale_physical_velocity(
        np.random.default_rng(771),
        shape,
        correlation_lengths_px=(3.0, 8.0),
        rms_amplitudes_um=(1.5, 2.0),
    )
    tissue = np.zeros(shape, bool)
    tissue[4:-4, 5:-5] = True
    velocity_px = physical_velocity_to_pixel(physical, np.diag((25.0, 25.0)))
    velocity_px = remove_tissue_affine_component(velocity_px, tissue)
    physical = velocity_px * 25.0
    maps = fixed_source_maps(
        physical,
        np.diag((25.0, 25.0)),
        angle_rad=0.025,
        scale=1.01,
        translation_xy=(0.2, -0.15),
    )
    metrics = topology_acceptance_metrics(
        maps["fixed_to_source_map"],
        maps["source_to_fixed_map"],
        minimum_jacobian=0.90,
    )
    assert maps["fixed_to_source_map"].shape == (2, *shape)
    assert maps["source_to_fixed_map"].shape == (2, *shape)
    assert metrics["accepted"] is True
    assert metrics["forward_nonpositive_jacobian_count"] == 0
    assert metrics["inverse_nonpositive_jacobian_count"] == 0

    reflected = identity_pixel_map(shape)
    reflected[0] = shape[1] - 1 - reflected[0]
    rejected = topology_acceptance_metrics(
        reflected,
        reflected,
    )
    assert rejected["accepted"] is False
    assert rejected["forward_nonpositive_jacobian_count"] == (shape[0] - 1) * (shape[1] - 1)


def test_every_predeclared_topology_gate_can_reject_independently():
    identity = identity_pixel_map((12, 18))
    common = {
        "minimum_jacobian": 0.20,
        "maximum_jacobian": 5.0,
        "maximum_cycle_rms_px": 0.05,
        "maximum_cycle_q99_px": 0.25,
        "maximum_cycle_max_px": 0.50,
    }
    assert topology_acceptance_metrics(identity, identity, **common)["accepted"] is True

    minimum = topology_acceptance_metrics(
        identity, identity, **{**common, "minimum_jacobian": 1.01}
    )
    maximum = topology_acceptance_metrics(
        identity, identity, **{**common, "maximum_jacobian": 0.99}
    )
    shifted = identity + np.asarray((0.10, 0.0), np.float32)[:, None, None]
    rms = topology_acceptance_metrics(
        shifted,
        identity,
        **{
            **common,
            "maximum_cycle_rms_px": 0.05,
            "maximum_cycle_q99_px": 1.0,
            "maximum_cycle_max_px": 1.0,
        },
    )
    q99 = topology_acceptance_metrics(
        shifted,
        identity,
        **{
            **common,
            "maximum_cycle_rms_px": 1.0,
            "maximum_cycle_q99_px": 0.05,
            "maximum_cycle_max_px": 1.0,
        },
    )
    maximum_cycle = topology_acceptance_metrics(
        shifted,
        identity,
        **{
            **common,
            "maximum_cycle_rms_px": 1.0,
            "maximum_cycle_q99_px": 1.0,
            "maximum_cycle_max_px": 0.05,
        },
    )
    assert minimum["jacobian_min_passed"] is False
    assert maximum["jacobian_max_passed"] is False
    assert rms["cycle_rms_passed"] is False
    assert q99["cycle_q99_passed"] is False
    assert maximum_cycle["cycle_max_passed"] is False
    assert not any(
        result["accepted"] for result in (minimum, maximum, rms, q99, maximum_cycle)
    )
