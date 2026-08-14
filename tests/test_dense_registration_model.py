from inspect import signature
from io import BytesIO

import pytest
import torch

from training.dense_registration_model import (
    DenseRegistrationModel,
    _PreActivationResidualBlock,
    _Residual5RegistrationEstimator,
    apply_similarity_transform,
    compose_pixel_maps,
    identity_pixel_map,
    integrate_stationary_velocity,
    jacobian_determinant,
    local_dot_product_correlation,
    modality_independent_descriptor,
    registration_maps,
    resize_similarity_parameters,
    resize_vector_field,
    warp_tensor,
)


def test_identity_pixel_map_and_warp_use_xy_pixel_coordinates():
    identity = identity_pixel_map(1, 4, 5, dtype=torch.float64)
    assert identity.shape == (1, 2, 4, 5)
    assert torch.equal(identity[0, 0, 0], torch.arange(5, dtype=torch.float64))
    assert torch.equal(identity[0, 1, :, 0], torch.arange(4, dtype=torch.float64))
    image = torch.arange(20, dtype=torch.float64).reshape(1, 1, 4, 5)
    assert torch.allclose(warp_tensor(image, identity), image)
    shifted = identity.clone()
    shifted[:, 0] += 1.0
    assert torch.allclose(warp_tensor(image, shifted)[..., :-1], image[..., 1:])


def test_similarity_forward_inverse_and_jacobian_are_analytic():
    parameters = torch.tensor([[0.19, 2.0, -1.5, 0.12]], dtype=torch.float64)
    identity = identity_pixel_map(1, 41, 47, dtype=torch.float64)
    forward = apply_similarity_transform(identity, parameters)
    inverse = apply_similarity_transform(identity, parameters, inverse=True)
    round_trip = compose_pixel_maps(forward, inverse)
    assert torch.allclose(round_trip[..., 8:-8, 8:-8], identity[..., 8:-8, 8:-8], atol=1e-10)
    expected_determinant = torch.exp(2.0 * parameters[:, 3])
    assert torch.allclose(
        jacobian_determinant(forward)[..., 1:-1, 1:-1],
        expected_determinant[:, None, None, None],
        atol=1e-10,
    )


def test_similarity_parameter_and_vector_resizing_preserve_pixel_geometry():
    parameters = torch.tensor([[0.2, 3.0, -4.0, 0.1]])
    resized_parameters = resize_similarity_parameters(parameters, (11, 21), (21, 41))
    assert torch.allclose(resized_parameters, torch.tensor([[0.2, 6.0, -8.0, 0.1]]))
    field = torch.zeros(1, 2, 11, 21)
    field[:, 0] = 3.0
    field[:, 1] = -4.0
    resized_field = resize_vector_field(field, (21, 41))
    assert torch.allclose(resized_field[:, 0], torch.full((1, 21, 41), 6.0))
    assert torch.allclose(resized_field[:, 1], torch.full((1, 21, 41), -8.0))


def test_scaling_and_squaring_returns_paired_maps_for_constant_velocity():
    velocity = torch.zeros(1, 2, 31, 35, dtype=torch.float64)
    velocity[:, 0] = 1.25
    velocity[:, 1] = -0.75
    forward, inverse = integrate_stationary_velocity(velocity, steps=6)
    identity = identity_pixel_map(1, 31, 35, dtype=torch.float64)
    assert torch.allclose(forward, identity + velocity, atol=1e-12)
    assert torch.allclose(inverse, identity - velocity, atol=1e-12)
    assert torch.allclose(
        compose_pixel_maps(forward, inverse)[..., 3:-3, 3:-3],
        identity[..., 3:-3, 3:-3],
        atol=1e-12,
    )


def test_smooth_stationary_velocity_is_invertible_and_topology_preserving():
    identity = identity_pixel_map(1, 49, 53, dtype=torch.float64)
    x = identity[:, :1]
    y = identity[:, 1:]
    velocity = torch.cat(
        (
            0.7 * torch.sin(y * (2.0 * torch.pi / 48.0)),
            0.5 * torch.sin(x * (2.0 * torch.pi / 52.0)),
        ),
        dim=1,
    )
    forward, inverse = integrate_stationary_velocity(velocity, steps=7)
    error = torch.linalg.vector_norm(
        compose_pixel_maps(forward, inverse) - identity,
        dim=1,
    )
    assert error[:, 4:-4, 4:-4].max() < 0.01
    assert jacobian_determinant(forward).min() > 0.85


def test_registration_maps_preserve_global_similarity_and_inverse_order():
    identity = identity_pixel_map(1, 49, 53, dtype=torch.float64)
    x = identity[:, :1]
    y = identity[:, 1:]
    velocity = torch.cat(
        (
            0.7 * torch.sin(y * (2.0 * torch.pi / 48.0)),
            0.5 * torch.sin(x * (2.0 * torch.pi / 52.0)),
        ),
        dim=1,
    )
    parameters = torch.tensor([[0.11, 1.5, -2.0, -0.06]], dtype=torch.float64)
    forward, inverse = registration_maps(parameters, velocity, steps=7)
    error = torch.linalg.vector_norm(
        compose_pixel_maps(forward, inverse) - identity,
        dim=1,
    )
    assert error[:, 8:-8, 8:-8].max() < 0.01
    assert jacobian_determinant(forward).min() > 0.85

    zero_velocity = torch.zeros_like(velocity)
    similarity_forward, similarity_inverse = registration_maps(
        parameters,
        zero_velocity,
        steps=7,
    )
    assert torch.allclose(
        similarity_forward,
        apply_similarity_transform(identity, parameters),
        atol=1e-12,
    )
    assert torch.allclose(
        similarity_inverse,
        apply_similarity_transform(identity, parameters, inverse=True),
        atol=1e-12,
    )


def test_registration_map_gradients_reach_similarity_and_local_velocity():
    parameters = torch.tensor([[0.03, 0.5, -0.25, 0.02]], requires_grad=True)
    velocity = torch.randn(1, 2, 17, 19) * 0.01
    velocity.requires_grad_()
    forward, inverse = registration_maps(parameters, velocity, steps=4)
    (forward.square().mean() + inverse.square().mean()).backward()
    assert parameters.grad is not None and torch.isfinite(parameters.grad).all()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()


def test_similarity_transform_applies_about_image_centre():
    identity = identity_pixel_map(1, 5, 7)
    parameters = torch.tensor([[torch.pi / 2.0, 0.0, 0.0, 0.0]])
    transformed = apply_similarity_transform(identity, parameters)
    assert torch.allclose(transformed[..., 2:3, 3:4], identity[..., 2:3, 3:4], atol=1e-6)
    assert torch.allclose(transformed[0, :, 2, 4], torch.tensor([3.0, 3.0]), atol=1e-6)


def test_local_correlation_is_normalized_row_major_and_differentiable():
    fixed = torch.zeros(1, 2, 5, 7)
    moving = torch.zeros_like(fixed)
    fixed[:, 0, 2, 3] = 7.0
    moving[:, 1, 2, 3] = 5.0
    moving[:, 0, 1, 4] = 11.0
    correlation = local_dot_product_correlation(fixed, moving, radius=1)
    assert correlation.shape == (1, 9, 5, 7)
    assert int(correlation[0, :, 2, 3].argmax()) == 2
    assert correlation[0, 2, 2, 3] == 1.0
    assert correlation[0, 4, 2, 3] == 0.0
    assert torch.equal(
        correlation,
        local_dot_product_correlation(fixed * 3.0, moving * 9.0, radius=1),
    )

    fixed = torch.randn(2, 5, 7, 9, requires_grad=True)
    moving = torch.randn(2, 5, 7, 9, requires_grad=True)
    local_dot_product_correlation(fixed, moving, radius=2).square().mean().backward()
    assert fixed.grad is not None and torch.count_nonzero(fixed.grad) > 0
    assert moving.grad is not None and torch.count_nonzero(moving.grad) > 0


def test_modality_independent_descriptor_is_gain_and_offset_invariant():
    torch.manual_seed(29)
    image = 0.1 + 0.8 * torch.rand(2, 1, 25, 31)
    expected = modality_independent_descriptor(image)
    observed = modality_independent_descriptor(0.23 + 0.61 * image)
    assert expected.shape == (2, 6, 25, 31)
    assert torch.allclose(
        expected[..., 2:-2, 2:-2],
        observed[..., 2:-2, 2:-2],
        atol=2e-5,
    )


def test_selected_production_architecture_is_the_only_model_path():
    model = DenseRegistrationModel()
    parameters = signature(DenseRegistrationModel).parameters
    assert set(parameters) == {
        "input_channels",
        "channels",
        "integration_steps",
        "maximum_rotation_degrees",
        "maximum_translation_fraction",
        "maximum_scale",
        "maximum_local_velocity_fraction",
        "correlation_radii",
    }
    assert model.encoder.stem[0].in_channels == 2
    assert model.correlation_radii == (4, 3, 2, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 758_500
    assert all(
        isinstance(stage, _Residual5RegistrationEstimator)
        for stage in model.registration_stages
    )
    assert all(len(stage.residual_blocks) == 5 for stage in model.registration_stages)
    assert all(
        isinstance(block, _PreActivationResidualBlock)
        for stage in model.registration_stages
        for block in stage.residual_blocks
    )
    assert model.registration_stages[0].input_projection[0].in_channels == 48 * 3 + 81
    assert not hasattr(model, "full_resolution_convgru")


def test_channels_and_correlation_radii_remain_configurable():
    model = DenseRegistrationModel(
        input_channels=1,
        channels=(4, 8, 8),
        correlation_radii=(2, 1, 1),
    )
    assert model.encoder.stem[0].in_channels == 1
    assert model.correlation_radii == (2, 1, 1)
    assert model.registration_stages[0].input_projection[0].in_channels == 8 * 3 + 25
    with pytest.raises(ValueError, match="one coarse-to-fine radius"):
        DenseRegistrationModel(channels=(4, 8), correlation_radii=(1,))
    with pytest.raises(ValueError, match="cannot be negative"):
        DenseRegistrationModel(channels=(4, 8), correlation_radii=(1, -1))


def test_selected_model_returns_deep_supervision_maps_and_backpropagates():
    torch.manual_seed(4)
    model = DenseRegistrationModel(
        channels=(4, 8, 8),
        correlation_radii=(2, 1, 1),
        integration_steps=2,
    )
    fixed = torch.rand(2, 2, 47, 51)
    moving = torch.rand(2, 2, 47, 51)
    details = model.forward_with_details(fixed, moving)
    assert set(details) == {
        "fixed_to_moving_map",
        "moving_to_fixed_map",
        "similarity_parameters",
        "local_velocity",
        "pyramid_velocities",
    }
    assert details["fixed_to_moving_map"].shape == (2, 2, 47, 51)
    assert details["moving_to_fixed_map"].shape == (2, 2, 47, 51)
    assert details["similarity_parameters"].shape == (2, 4)
    assert details["local_velocity"].shape == (2, 2, 47, 51)
    assert [item.shape[-2:] for item in details["pyramid_velocities"]] == [
        (12, 13),
        (24, 26),
        (47, 51),
    ]
    loss = (
        details["fixed_to_moving_map"].square().mean()
        + details["moving_to_fixed_map"].square().mean()
    )
    loss.backward()
    assert model.velocity_heads[-1].weight.grad is not None
    assert model.similarity_head.network[-1].weight.grad is not None
    assert model.encoder.stem[0].weight.grad is not None
    for stage in model.registration_stages:
        assert stage.input_projection[0].weight.grad is not None
        for block in stage.residual_blocks:
            assert block.conv1.weight.grad is not None
            assert block.conv2.weight.grad is not None


def test_each_local_velocity_update_is_bounded_and_smoothed():
    model = DenseRegistrationModel(
        input_channels=1,
        channels=(4, 8, 8),
        correlation_radii=(2, 1, 1),
        integration_steps=2,
        maximum_local_velocity_fraction=0.10,
    ).eval()
    for head in model.velocity_heads:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.constant_(head.bias, 100.0)
    pyramid = model.forward_with_details(
        torch.zeros(1, 1, 24, 28),
        torch.zeros(1, 1, 24, 28),
    )["pyramid_velocities"]
    previous = None
    for velocity in pyramid:
        residual = (
            velocity
            if previous is None
            else velocity - resize_vector_field(previous, velocity.shape[-2:])
        )
        assert residual.abs().max() <= 0.10 * min(residual.shape[-2:]) + 1e-6
        previous = velocity


def test_selected_model_loads_its_state_dict_strictly():
    expected = DenseRegistrationModel()
    observed = DenseRegistrationModel()
    assert observed.load_state_dict(expected.state_dict(), strict=True).missing_keys == []
    assert observed.state_dict().keys() == expected.state_dict().keys()


def test_selected_model_has_minimal_onnx_contract_and_cpu_parity():
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = DenseRegistrationModel(
        channels=(4, 8),
        correlation_radii=(1, 1),
        integration_steps=2,
    ).eval()
    fixed = torch.rand(1, 2, 16, 20)
    moving = torch.rand(1, 2, 16, 20)
    expected = model(fixed, moving)
    stream = BytesIO()
    torch.onnx.export(
        model,
        (fixed, moving),
        stream,
        input_names=["fixed_atlas", "moving_slice"],
        output_names=["fixed_to_moving_map", "moving_to_fixed_map"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(stream.getvalue(), providers=["CPUExecutionProvider"])
    observed = session.run(
        None,
        {"fixed_atlas": fixed.numpy(), "moving_slice": moving.numpy()},
    )
    assert torch.allclose(torch.from_numpy(observed[0]), expected[0], atol=2e-4)
    assert torch.allclose(torch.from_numpy(observed[1]), expected[1], atol=2e-4)
