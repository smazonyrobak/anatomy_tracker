import ast
import inspect
import math

import pytest
import torch

import training.arbitrary_plane_deformation_primitives as deformation


def _identity(batch=1, shape=(25, 31), dtype=torch.float64):
    return deformation.identity_pixel_map_yx(batch, shape, dtype=dtype)


def test_identity_and_constant_translation_integrate_to_exact_paired_maps():
    identity = _identity()
    zero = torch.zeros_like(identity)
    forward, inverse = deformation.integrate_stationary_velocity_yx(zero, steps=7)
    assert torch.equal(forward, identity)
    assert torch.equal(inverse, identity)
    assert torch.equal(
        deformation.jacobian_determinant_yx(forward),
        torch.ones(1, 1, 25, 31, dtype=torch.float64),
    )

    velocity = torch.zeros_like(identity)
    velocity[:, 0] = 0.75
    velocity[:, 1] = -1.25
    forward, inverse = deformation.integrate_stationary_velocity_yx(velocity, steps=6)
    assert torch.allclose(forward, identity + velocity, atol=1e-11, rtol=0.0)
    assert torch.allclose(inverse, identity - velocity, atol=1e-11, rtol=0.0)
    cycles = deformation.inverse_consistency_yx(forward, inverse)
    assert cycles["forward_then_inverse_error_yx"].abs().max() < 1e-11
    assert cycles["inverse_then_forward_error_yx"].abs().max() < 1e-11
    assert not bool(cycles["forward_then_inverse_valid_mask"].all())


def test_smooth_stationary_velocity_has_positive_topology_and_small_cycles():
    height, width = 31, 37
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)[None, None, :, None]
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)[None, None, None, :]
    velocity = torch.cat(
        (
            (0.22 * torch.sin(math.pi * y) * torch.cos(math.pi * x)).expand(
                1, 1, height, width
            ),
            (0.18 * torch.cos(math.pi * y) * torch.sin(math.pi * x)).expand(
                1, 1, height, width
            ),
        ),
        dim=1,
    )
    forward, inverse = deformation.integrate_stationary_velocity_yx(velocity, steps=7)
    cycles = deformation.inverse_consistency_yx(forward, inverse)
    assert deformation.jacobian_determinant_yx(forward).min() > 0.9
    assert deformation.jacobian_determinant_yx(inverse).min() > 0.9
    assert cycles["forward_then_inverse_error_yx"][..., 3:-3, 3:-3].abs().max() < 1e-3
    assert cycles["inverse_then_forward_error_yx"][..., 3:-3, 3:-3].abs().max() < 1e-3


def test_partial_support_projection_removes_all_six_affine_dofs_exactly():
    height, width = 25, 31
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)[None, None, :, None]
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)[None, None, None, :]
    velocity = torch.cat(
        (
            (2.0 - 0.5 * y + 3.0 * x).expand(1, 1, height, width),
            (-1.0 + 4.0 * y + 0.25 * x).expand(1, 1, height, width),
        ),
        dim=1,
    )
    support = (((y / 0.75).square() + ((x + 0.3) / 0.55).square()) < 1.0).to(
        torch.float64
    )
    residual, coefficients, post, projection_weight = (
        deformation.support_weighted_affine_projection_yx(velocity, support)
    )
    expected = torch.tensor(
        [[[2.0, -0.5, 3.0], [-1.0, 4.0, 0.25]]], dtype=torch.float64
    )
    assert torch.allclose(coefficients, expected, atol=2e-12, rtol=0.0)
    assert residual.abs().max() < 3e-12
    assert post.abs().max() < 2e-12
    assert torch.allclose(
        projection_weight.sum(dim=(-2, -1)),
        torch.ones(1, 1, dtype=torch.float64),
        atol=1e-14,
    )


def test_projection_keeps_local_structure_and_detaches_support_gauge():
    height, width = 23, 29
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)[None, None, :, None]
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)[None, None, None, :]
    local = torch.cat(
        (
            (y.square() + 0.2 * x * y).expand(1, 1, height, width),
            (x.square() - 0.15 * x * y).expand(1, 1, height, width),
        ),
        dim=1,
    ).requires_grad_()
    support = torch.sigmoid(4.0 - 7.0 * (x.square() + y.square())).requires_grad_()
    residual, _, post, _ = deformation.support_weighted_affine_projection_yx(
        local, support
    )
    assert residual.square().mean() > 0.01
    assert post.abs().max() < 2e-12
    (residual.square().mean()).backward()
    assert local.grad is not None and torch.isfinite(local.grad).all()
    assert torch.count_nonzero(local.grad) > 0
    assert support.grad is None


def test_yx_channel_order_and_zero_image_padding_are_explicit():
    height, width = 5, 6
    identity = _identity(shape=(height, width), dtype=torch.float32)
    source = torch.arange(height, dtype=torch.float32)[None, None, :, None].expand(
        1, 1, height, width
    )
    shift_down_source = identity.clone()
    shift_down_source[:, 0] += 1.0
    observed = deformation.warp_tensor_with_map_yx(source, shift_down_source)
    assert torch.equal(observed[:, :, :-1], source[:, :, 1:])
    assert torch.count_nonzero(observed[:, :, -1]) == 0

    outside = identity.clone()
    outside[:, 1] += width + 1
    assert torch.count_nonzero(
        deformation.warp_tensor_with_map_yx(torch.ones_like(source), outside)
    ) == 0


def test_jacobian_matches_known_scale_and_detects_reflection():
    scaled = _identity(shape=(17, 19))
    scaled[:, 0] *= 1.2
    scaled[:, 1] *= 0.8
    determinant = deformation.jacobian_determinant_yx(scaled)
    assert torch.allclose(
        determinant,
        torch.full_like(determinant, 0.96),
        atol=2e-14,
        rtol=0.0,
    )

    reflected = _identity(shape=(17, 19))
    reflected[:, 1] = 18.0 - reflected[:, 1]
    assert torch.allclose(
        deformation.jacobian_determinant_yx(reflected),
        -torch.ones(1, 1, 17, 19, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )


def test_integration_and_topology_have_finite_nonzero_gradients():
    torch.manual_seed(17)
    velocity = (0.02 * torch.randn(1, 2, 9, 11, dtype=torch.float64)).requires_grad_()
    forward, inverse = deformation.integrate_stationary_velocity_yx(velocity, steps=4)
    cycles = deformation.inverse_consistency_yx(forward, inverse)
    weight = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64)[None, None, None]
    loss = (
        (forward * weight).sum()
        + deformation.jacobian_determinant_yx(forward).square().mean()
        + cycles["forward_then_inverse_error_yx"].square().mean()
    )
    loss.backward()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()
    assert torch.count_nonzero(velocity.grad) > 0


def test_decoder_is_bounded_affine_free_near_identity_and_fully_differentiable():
    torch.manual_seed(29)
    decoder = deformation.AffineFreeSVFDecoder(
        hidden_channels=8,
        max_velocity_fraction_yx=(0.05, 0.04),
        integration_steps=5,
    )
    context = torch.randn(3, 8, 4, 5, requires_grad=True)
    output = decoder(context, (16, 20))
    assert output["raw_velocity_fraction_yx_lowres"].shape == (3, 2, 4, 5)
    assert output["support_logits_lowres"].shape == (3, 1, 4, 5)
    assert output["stationary_velocity_yx_px"].shape == (3, 2, 16, 20)
    assert output["forward_map_yx_px"].shape == (3, 2, 16, 20)
    assert output["forward_jacobian_determinant"].shape == (3, 1, 16, 20)
    assert output["forward_then_inverse_valid_mask"].dtype == torch.bool
    assert output["raw_velocity_fraction_yx"][:, 0].abs().max() <= 0.05 + 1e-7
    assert output["raw_velocity_fraction_yx"][:, 1].abs().max() <= 0.04 + 1e-7
    assert output["stationary_velocity_yx_px"].abs().max() < 0.1
    assert output["postprojection_affine_coefficients_yx"].abs().max() < 2e-6
    assert output["forward_jacobian_determinant"].min() > 0.95
    assert torch.equal(output["pullback_map_yx_px"], output["forward_map_yx_px"])
    assert output["velocity_gradient_rescale"].shape == (3, 1, 1, 1)
    assert torch.allclose(
        output["projection_support_weight"].sum(dim=(-2, -1)),
        torch.ones(3, 1),
        atol=2e-6,
    )
    assert output["affine_projection_gauge"] == "uniform_canvas"
    assert torch.allclose(
        output["projection_support_weight"],
        torch.full_like(output["projection_support_weight"], 1.0 / (16 * 20)),
        atol=1e-7,
    )

    spatial_weight = torch.linspace(-1.0, 1.0, 20)[None, None, None]
    loss = (
        (output["stationary_velocity_yx_px"] * spatial_weight).sum()
        + output["support_logits"].square().mean()
        + output["forward_then_inverse_error_yx"].square().mean()
    )
    loss.backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    for parameter in decoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_decoder_cuda_amp_keeps_affine_projection_solve_in_float32():
    decoder = deformation.AffineFreeSVFDecoder(8, integration_steps=3).cuda().train()
    context = torch.randn(2, 8, 5, 7, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = decoder(context, (20, 28))
        loss = (
            output["stationary_velocity_yx_px"].square().mean()
            + output["support_logits"].square().mean()
        )
    loss.backward()
    assert output["stationary_velocity_yx_px"].dtype in (torch.float16, torch.float32)
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def test_saturated_decoder_remains_topology_preserving_at_production_raster_size():
    torch.manual_seed(811)
    decoder = deformation.AffineFreeSVFDecoder(
        8,
        max_velocity_fraction_yx=(0.08, 0.08),
        integration_steps=7,
        maximum_velocity_gradient=0.35,
    )
    with torch.no_grad():
        decoder.velocity_head.weight.mul_(5000.0)
        decoder.velocity_head.bias.uniform_(-20.0, 20.0)
    context = 20.0 * torch.randn(3, 8, 24, 32)
    output = decoder(context, (192, 256))
    interior = output["forward_jacobian_determinant"][..., 4:-4, 4:-4]
    assert output["velocity_gradient_rescale"].min() < 1.0
    assert interior.min() > 0.05
    assert torch.isfinite(output["forward_then_inverse_error_yx"]).all()


def test_new_primitives_have_no_legacy_checkpoint_or_filesystem_dependency():
    tree = ast.parse(inspect.getsource(deformation))
    imports = set()
    calls = set()

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            calls.add(dotted_name(node.func))

    forbidden = (
        "pathlib",
        "pickle",
        "timm",
        "torchvision",
        "training.dense_registration_model",
        "training.independent_joint_model",
        "training.joint_pose_registration",
        "training.atlas_pose",
    )
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden
    )
    assert not ({"torch.load", "open"} & calls)
