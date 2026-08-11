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
    preprocess_registration_tensor,
    remove_global_affine,
    remove_tissue_affine,
    soft_tissue_support,
    smoothness_loss,
    synthetic_flow_loss,
    tissue_affine_component,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mixed precision is unavailable")
def test_geometric_projection_stays_finite_in_cuda_mixed_precision():
    model = DiffeomorphicRegistrationUNet(base_channels=4).cuda().eval()
    image = torch.rand(1, 1, 320, 464, device="cuda")
    mask = torch.ones_like(image)
    with torch.autocast("cuda", dtype=torch.float16):
        forward, inverse, velocity = model(image, image, mask, mask)

    assert forward.dtype == inverse.dtype == velocity.dtype == torch.float32
    assert torch.isfinite(forward).all()
    assert torch.isfinite(inverse).all()
    assert torch.isfinite(velocity).all()


def test_affine_projection_removes_translation_scale_rotation_and_shear():
    identity = pixel_identity_grid(2, 40, 52)
    x, y = identity[:, 0], identity[:, 1]
    affine = torch.stack((3.0 + 0.1 * x - 0.2 * y, -4.0 + 0.3 * x + 0.07 * y), dim=1)
    assert remove_global_affine(affine).abs().max() < 2e-5


def test_tissue_projection_catches_affine_hidden_by_the_padded_canvas_and_is_identity_outside():
    identity = pixel_identity_grid(1, 80, 120)
    x, y = identity[:, 0], identity[:, 1]
    tissue = (((x - 60.0) / 24.0).square() + ((y - 40.0) / 18.0).square() < 1.0)[:, None].float()
    raw = torch.stack((2.5 + 0.04 * x, -1.75 + 0.03 * y), dim=1) * tissue
    support = soft_tissue_support(tissue, tissue)
    projected = remove_tissue_affine(raw, support, tissue)
    final_map = identity + projected

    assert tissue_affine_component(final_map, tissue).abs().max() < 2e-4
    assert torch.equal(projected * (1.0 - tissue), torch.zeros_like(projected))


def test_training_preprocessing_is_masked_percentile_normalization():
    image = torch.linspace(-2.0, 5.0, 80).reshape(1, 1, 8, 10)
    mask = torch.zeros_like(image)
    mask[:, :, 1:7, 2:9] = 1.0
    processed = preprocess_registration_tensor(image, mask)
    values = image[mask > 0.5]
    low, high = torch.quantile(values, torch.tensor([0.005, 0.995]))
    expected = ((image - low) / (high - low)).clamp(0.0, 1.0) * mask

    assert torch.allclose(processed, expected)


def test_fractional_trusted_masks_are_hard_thresholded_everywhere():
    model = DiffeomorphicRegistrationUNet(base_channels=4).eval()
    image = torch.rand(1, 1, 32, 48)
    fractional = torch.zeros_like(image)
    fractional[:, :, 4:-4, 6:-6] = 0.51
    fractional[:, :, 0, 0] = 0.49
    hard = (fractional > 0.5).float()
    with torch.no_grad():
        fractional_outputs = model(image, image, fractional, fractional)
        hard_outputs = model(image, image, hard, hard)
    assert all(torch.equal(left, right) for left, right in zip(fractional_outputs, hard_outputs))
    assert torch.equal(
        preprocess_registration_tensor(image, fractional),
        preprocess_registration_tensor(image, hard),
    )


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
