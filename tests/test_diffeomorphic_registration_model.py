from io import BytesIO

import pytest
import torch

from training.diffeomorphic_registration_model import (
    DiffeomorphicRegistrationUNet,
    compose_pixel_maps,
    integrate_stationary_velocity,
    inverse_consistency_loss,
    jacobian_determinant,
    mind_loss,
    pixel_identity_grid,
    remove_global_affine,
    smoothness_loss,
    synthetic_flow_loss,
    topology_loss,
)


def test_zero_initialized_model_is_exact_identity_with_expected_shapes():
    model = DiffeomorphicRegistrationUNet(base_channels=4, integration_steps=4).eval()
    fixed = torch.rand(1, 1, 320, 464)
    moving = torch.rand_like(fixed)
    mask = torch.ones(1, 1, 320, 464)
    atlas_to_affine, affine_to_atlas, velocity = model(fixed, moving, mask, mask)
    identity = pixel_identity_grid(1, 320, 464)

    assert atlas_to_affine.shape == affine_to_atlas.shape == velocity.shape == (1, 2, 320, 464)
    assert torch.equal(velocity, torch.zeros_like(velocity))
    assert torch.equal(atlas_to_affine, identity)
    assert torch.equal(affine_to_atlas, identity)


def test_affine_projection_removes_translation_scale_rotation_and_shear():
    identity = pixel_identity_grid(2, 40, 52)
    x, y = identity[:, 0], identity[:, 1]
    affine = torch.stack((3.0 + 0.1 * x - 0.2 * y, -4.0 + 0.3 * x + 0.07 * y), dim=1)
    assert remove_global_affine(affine).abs().max() < 2e-5


def test_smooth_analytic_exponential_has_positive_jacobian_and_accurate_inverse():
    identity = pixel_identity_grid(1, 64, 80)
    x = identity[:, 0] / 79.0
    y = identity[:, 1] / 63.0
    velocity = torch.stack(
        (1.25 * torch.sin(2.0 * torch.pi * x) * torch.sin(torch.pi * y),
         0.9 * torch.sin(torch.pi * x) * torch.sin(2.0 * torch.pi * y)),
        dim=1,
    )
    velocity = remove_global_affine(velocity)
    forward = integrate_stationary_velocity(velocity, steps=8)
    inverse = integrate_stationary_velocity(-velocity, steps=8)
    round_trip = compose_pixel_maps(forward, inverse)
    interior = (..., slice(4, -4), slice(4, -4))

    assert jacobian_determinant(forward).min() > 0.0
    assert (round_trip - identity)[interior].abs().max() < 0.25
    assert inverse_consistency_loss(forward, inverse, torch.ones(1, 1, 64, 80)) < 0.02


def test_deliberate_fold_has_nonzero_topology_loss():
    folded = pixel_identity_grid(1, 32, 40)
    folded[:, 0] = 39.0 - folded[:, 0]
    assert jacobian_determinant(folded).max() < 0.0
    assert topology_loss(folded) > 0.9


def test_registration_losses_propagate_gradients():
    torch.manual_seed(4)
    moving = torch.rand(1, 1, 32, 40)
    fixed = torch.roll(moving, shifts=1, dims=-1)
    mask = torch.ones_like(fixed)
    velocity = (torch.randn(1, 2, 32, 40) * 0.02).requires_grad_()
    forward = integrate_stationary_velocity(remove_global_affine(velocity), steps=4)
    inverse = integrate_stationary_velocity(-remove_global_affine(velocity), steps=4)
    known = pixel_identity_grid(1, 32, 40)
    loss = (
        mind_loss(fixed, moving, forward, mask)
        + inverse_consistency_loss(forward, inverse, mask)
        + smoothness_loss(velocity, mask)
        + topology_loss(forward, minimum_jacobian=0.2)
        + synthetic_flow_loss(forward, known, mask)
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()
    assert velocity.grad.abs().sum() > 0.0


def test_untrained_model_exports_maps_and_velocity_to_onnx():
    onnx = pytest.importorskip("onnx")
    model = DiffeomorphicRegistrationUNet(base_channels=4, integration_steps=2).eval()
    inputs = tuple(torch.zeros(1, 1, 320, 464) for _ in range(4))
    stream = BytesIO()
    torch.onnx.export(
        model,
        inputs,
        stream,
        input_names=["fixed", "moving", "fixed_mask", "moving_mask"],
        output_names=["atlas_to_affine", "affine_to_atlas", "velocity"],
        dynamic_axes={name: {0: "batch"} for name in (
            "fixed", "moving", "fixed_mask", "moving_mask",
            "atlas_to_affine", "affine_to_atlas", "velocity",
        )},
        opset_version=17,
        dynamo=False,
    )
    exported = onnx.load_model_from_string(stream.getvalue())
    onnx.checker.check_model(exported)
    assert [output.name for output in exported.graph.output] == [
        "atlas_to_affine",
        "affine_to_atlas",
        "velocity",
    ]
