import ast
import inspect

import pytest
import torch

import training.arbitrary_plane_full_frame_primitives as primitives
import training.arbitrary_plane_synthetic_generator_v2 as frozen_slab
from training.arbitrary_plane_geometry import (
    frame_to_physical_ouv,
    positive_inplane_basis,
    render_arbitrary_plane,
    rotation_6d_to_frame,
)


def _state(center, rotation_6d, spans, shear=0.0):
    center = torch.as_tensor(center)
    rotation_6d = torch.as_tensor(
        rotation_6d, device=center.device, dtype=center.dtype
    )
    spans = torch.as_tensor(spans, device=center.device, dtype=center.dtype)
    shear = torch.as_tensor(shear, device=center.device, dtype=center.dtype)
    return primitives.full_frame_state_from_components(
        center,
        rotation_6d_to_frame(rotation_6d),
        positive_inplane_basis(torch.log(spans), shear),
    )


def _delta_basis(update):
    diagonal = torch.exp(update[..., 6:8])
    shear = update[..., 8]
    zero = torch.zeros_like(shear)
    return torch.stack(
        (
            torch.stack((diagonal[..., 0], shear * diagonal[..., 1]), -1),
            torch.stack((zero, diagonal[..., 1]), -1),
        ),
        -2,
    )


def _physical_volume(shape, origin, spacing, coefficients):
    axes = torch.meshgrid(
        *(torch.arange(size, dtype=origin.dtype) for size in shape), indexing="ij"
    )
    points = torch.stack(axes, -1)
    physical = origin + (points + 0.5) * spacing
    return (physical * coefficients).sum(-1)


def test_full_frame_state_round_trip_and_physical_ouv_decode():
    center = torch.tensor(
        [[120.0, 210.0, 330.0], [-40.0, 95.0, 410.0]], dtype=torch.float64
    )
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.0, -2.0, 4.0, 1.0]],
        dtype=torch.float64,
    )
    frame = rotation_6d_to_frame(rotation)
    basis = positive_inplane_basis(
        torch.log(torch.tensor([[420.0, 280.0], [315.0, 205.0]], dtype=torch.float64)),
        torch.tensor([0.0, -0.37], dtype=torch.float64),
    )

    state = primitives.full_frame_state_from_components(center, frame, basis)
    recovered = primitives.full_frame_state_to_components(state)

    assert state.shape == (2, primitives.FULL_FRAME_STATE_SIZE)
    assert torch.allclose(recovered[0], center, atol=0.0, rtol=0.0)
    assert torch.allclose(recovered[1], frame, atol=1e-12, rtol=0.0)
    assert torch.allclose(recovered[2], basis, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        primitives.full_frame_state_to_physical_ouv(state),
        frame_to_physical_ouv(center, frame, basis),
        atol=1e-12,
        rtol=0.0,
    )


def test_so3_exp_is_exact_at_zero_proper_and_has_finite_zero_gradient():
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    identity = primitives.so3_exp_map(zero)
    assert torch.equal(identity, torch.eye(3, dtype=torch.float64))
    weights = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    (identity * weights).sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.count_nonzero(zero.grad) > 0

    quarter_turn = primitives.so3_exp_map(
        torch.tensor([0.0, 0.0, torch.pi / 2.0], dtype=torch.float64)
    )
    expected = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(quarter_turn, expected, atol=1e-12, rtol=0.0)

    rotations = primitives.so3_exp_map(
        torch.tensor([[0.2, -0.3, 0.4], [-0.7, 0.1, 0.2]], dtype=torch.float64)
    )
    eye = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    assert torch.allclose(rotations.transpose(-1, -2) @ rotations, eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_full_frame_update_uses_preupdate_local_frame_and_right_composition():
    state = _state(
        torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [420.0, 275.0],
        0.2,
    )
    update = torch.tensor(
        [0.17, -0.11, 0.09, 4.0, -3.0, 2.0, 0.03, -0.04, 0.07],
        dtype=torch.float64,
    )
    center, frame, basis = primitives.full_frame_state_to_components(state)
    composed = primitives.compose_full_frame_state(state, update)
    observed_center, observed_frame, observed_basis = (
        primitives.full_frame_state_to_components(composed)
    )

    assert torch.allclose(
        observed_center,
        center + frame @ update[3:6],
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        observed_frame,
        frame @ primitives.so3_exp_map(update[:3]),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        observed_basis, basis @ _delta_basis(update), atol=1e-11, rtol=0.0
    )
    assert torch.allclose(
        observed_frame.transpose(-1, -2) @ observed_frame,
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.linalg.det(observed_frame) > 0.0
    assert observed_basis[1, 0] == 0.0
    assert torch.linalg.det(observed_basis) > 0.0

    zero_update = torch.zeros(primitives.FULL_FRAME_UPDATE_SIZE, dtype=torch.float64)
    assert torch.allclose(
        primitives.compose_full_frame_state(state, zero_update), state, atol=1e-12, rtol=0.0
    )


def test_two_full_frame_updates_match_manual_sequential_products():
    state = _state(
        torch.tensor([30.0, 40.0, 50.0], dtype=torch.float64),
        [1.0, 2.0, 3.0, -2.0, 4.0, 1.0],
        [18.0, 15.0],
        -0.25,
    )
    first = torch.tensor(
        [0.05, -0.03, 0.02, 1.0, -2.0, 0.5, 0.02, -0.01, 0.04],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [-0.02, 0.04, 0.01, -0.3, 0.7, 1.2, -0.01, 0.03, -0.02],
        dtype=torch.float64,
    )
    center, frame, basis = primitives.full_frame_state_to_components(state)
    first_frame = frame @ primitives.so3_exp_map(first[:3])
    expected_center = center + frame @ first[3:6] + first_frame @ second[3:6]
    expected_frame = first_frame @ primitives.so3_exp_map(second[:3])
    expected_basis = basis @ _delta_basis(first) @ _delta_basis(second)

    observed = primitives.compose_full_frame_state(
        primitives.compose_full_frame_state(state, first), second
    )
    observed_center, observed_frame, observed_basis = (
        primitives.full_frame_state_to_components(observed)
    )
    assert torch.allclose(observed_center, expected_center, atol=1e-11, rtol=0.0)
    assert torch.allclose(observed_frame, expected_frame, atol=1e-11, rtol=0.0)
    assert torch.allclose(observed_basis, expected_basis, atol=1e-11, rtol=0.0)


def test_zero_update_has_finite_nonzero_gradient_in_all_nine_coordinates():
    state = _state(
        torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
        [1.0, 2.0, 3.0, -2.0, 4.0, 1.0],
        [420.0, 275.0],
        0.2,
    )
    update = torch.zeros(
        primitives.FULL_FRAME_UPDATE_SIZE, dtype=torch.float64, requires_grad=True
    )
    center, frame, basis = primitives.full_frame_state_to_components(
        primitives.compose_full_frame_state(state, update)
    )
    loss = (
        (center * center.new_tensor([0.7, -1.1, 2.3])).sum()
        + (frame * frame.new_tensor([[0.2, -0.5, 1.3], [0.7, 0.4, -0.8], [1.1, -0.9, 0.6]])).sum()
        + (basis * basis.new_tensor([[0.3, -0.7], [0.4, 1.2]])).sum()
    )
    loss.backward()
    assert torch.isfinite(update.grad).all()
    assert torch.count_nonzero(update.grad) == primitives.FULL_FRAME_UPDATE_SIZE


def test_boxcar_kernel_matches_frozen_trapezoid_schedule_within_float_roundoff():
    assert [
        primitives.finite_psf_axial_sample_count(thickness, 12.5)
        for thickness in (25.0, 55.0, 100.0)
    ] == [3, 7, 9]
    offsets, weights = primitives.normalized_finite_psf_kernel(
        torch.tensor(55.0, dtype=torch.float64), 7, "boxcar"
    )
    reference = frozen_slab.finite_boxcar_kernel("finite_boxcar", 55.0)
    expected_offsets = torch.tensor(
        reference["optical_kernel_offsets_um"], dtype=torch.float64
    )
    expected_masses = torch.tensor([1, 2, 2, 2, 2, 2, 1], dtype=torch.float64)
    assert torch.allclose(offsets, expected_offsets, atol=8e-15, rtol=0.0)
    assert torch.equal(
        weights,
        torch.tensor(reference["optical_kernel_weights"], dtype=torch.float64),
    )
    assert torch.equal(weights, expected_masses / expected_masses.sum())
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-15)
    assert torch.dot(offsets, weights).item() == pytest.approx(0.0, abs=1e-15)


def test_finite_gaussian_kernel_is_explicit_symmetric_positive_and_differentiable():
    thickness = torch.tensor([40.0, 80.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([9.0, 18.0], dtype=torch.float64, requires_grad=True)
    offsets, weights = primitives.normalized_finite_psf_kernel(
        thickness, 7, "gaussian", gaussian_sigma_um=sigma
    )

    assert offsets.shape == weights.shape == (2, 7)
    assert torch.equal(offsets, -offsets.flip(-1))
    assert torch.equal(weights, weights.flip(-1))
    assert torch.all(weights > 0.0)
    assert torch.allclose(weights.sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-15)
    assert torch.allclose((weights * offsets).sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-14)

    second_moment = (weights * offsets.square()).sum()
    second_moment.backward()
    assert torch.isfinite(thickness.grad).all() and torch.count_nonzero(thickness.grad) == 2
    assert torch.isfinite(sigma.grad).all() and torch.count_nonzero(sigma.grad) == 2

    with pytest.raises(ValueError, match="odd integer"):
        primitives.normalized_finite_psf_kernel(thickness.detach(), 4, "boxcar")
    with pytest.raises(ValueError, match="requires an explicit sigma"):
        primitives.normalized_finite_psf_kernel(thickness.detach(), 7, "gaussian")


def test_narrow_float32_gaussian_stays_positive_and_differentiable():
    thickness = torch.tensor(100.0, dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor(100.0 / 30.0, dtype=torch.float32, requires_grad=True)
    offsets, weights = primitives.normalized_finite_psf_kernel(
        thickness, 7, "gaussian", gaussian_sigma_um=sigma
    )
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0.0)
    assert weights.sum().item() == pytest.approx(1.0, abs=2e-7)
    assert torch.equal(offsets, -offsets.flip(-1))
    assert torch.equal(weights, weights.flip(-1))
    (weights * offsets.square()).sum().backward()
    assert torch.isfinite(thickness.grad) and thickness.grad != 0.0
    assert torch.isfinite(sigma.grad) and sigma.grad != 0.0

    tiny_sigma = torch.tensor(1e-30, dtype=torch.float32, requires_grad=True)
    tiny_offsets, tiny_weights = primitives.normalized_finite_psf_kernel(
        thickness.detach(), 7, "gaussian", gaussian_sigma_um=tiny_sigma
    )
    (tiny_weights * tiny_offsets.square()).sum().backward()
    assert torch.isfinite(tiny_weights).all() and torch.all(tiny_weights > 0.0)
    assert torch.isfinite(tiny_sigma.grad)


def test_one_sample_render_matches_existing_quicknii_plane_contract():
    torch.manual_seed(17)
    volume = torch.rand(7, 6, 9, dtype=torch.float64)
    index_center = torch.tensor([3.2, 2.8, 4.4], dtype=torch.float64)
    frame = rotation_6d_to_frame(
        torch.tensor([0.8, -0.3, 1.1, 0.2, 1.3, -0.4], dtype=torch.float64)
    )
    basis = positive_inplane_basis(
        torch.log(torch.tensor([4.0, 3.0], dtype=torch.float64)),
        torch.tensor(0.17, dtype=torch.float64),
    )
    physical_state = primitives.full_frame_state_from_components(
        index_center + 0.5, frame, basis
    )

    expected, _ = render_arbitrary_plane(
        volume, index_center, frame, basis, (3, 4)
    )
    observed = primitives.render_finite_thickness_plane(
        volume[None],
        physical_state,
        (3, 4),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([7.0], dtype=torch.float64),
    )
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "rotation",
    (
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 2.0, 3.0, -2.0, 4.0, 1.0],
    ),
)
def test_physical_normal_psf_preserves_linear_ramp_with_anisotropic_axis_order(rotation):
    shape = (9, 10, 11)
    origin = torch.tensor([-10.0, 20.0, 5.0], dtype=torch.float64)
    spacing = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    coefficients = torch.tensor([5.0, -2.0, 0.75], dtype=torch.float64)
    volume = _physical_volume(shape, origin, spacing, coefficients)[None]
    center = origin + (torch.tensor([4.2, 5.1, 5.3], dtype=torch.float64) + 0.5) * spacing
    state = _state(center, rotation, [7.0, 6.0], 0.11)
    offsets, weights = primitives.normalized_finite_psf_kernel(
        torch.tensor(4.0, dtype=torch.float64), 3, "boxcar"
    )

    centre_plane = primitives.render_finite_thickness_plane(
        volume, state, (3, 4), origin, spacing, offsets.new_zeros(1), offsets.new_ones(1)
    )
    slab = primitives.render_finite_thickness_plane(
        volume, state, (3, 4), origin, spacing, offsets, weights
    )
    assert torch.allclose(slab, centre_plane, atol=2e-12, rtol=0.0)


def test_quadratic_axial_ramp_matches_kernel_second_moment():
    shape = (7, 5, 5)
    ap = torch.arange(shape[0], dtype=torch.float64)[:, None, None]
    volume = (ap - 3.0).square().expand(shape)[None]
    frame = rotation_6d_to_frame(
        torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
    )
    basis = torch.eye(2, dtype=torch.float64)
    target = torch.tensor([3.5, 2.5, 2.5], dtype=torch.float64)
    state = primitives.full_frame_state_from_components(
        target + 0.5 * (frame[:, 0] + frame[:, 1]), frame, basis
    )
    offsets, weights = primitives.normalized_finite_psf_kernel(
        torch.tensor(2.0, dtype=torch.float64), 3, "boxcar"
    )
    rendered = primitives.render_finite_thickness_plane(
        volume,
        state,
        (1, 1),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        offsets,
        weights,
    )
    assert rendered.item() == pytest.approx(
        torch.dot(weights, offsets.square()).item(), abs=1e-12
    )


def test_zero_padding_attenuates_edge_without_per_pixel_psf_renormalization():
    volume = torch.ones(1, 5, 5, 5, dtype=torch.float64)
    frame = torch.eye(3, dtype=torch.float64)
    basis = torch.eye(2, dtype=torch.float64)
    first_ml_voxel = torch.tensor([2.5, 2.5, 0.5], dtype=torch.float64)
    state = primitives.full_frame_state_from_components(
        first_ml_voxel + 0.5 * (frame[:, 0] + frame[:, 1]), frame, basis
    )
    rendered = primitives.render_finite_thickness_plane(
        volume,
        state,
        (1, 1),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64),
    )
    assert rendered.item() == pytest.approx(0.75, abs=1e-12)


def test_multichannel_one_hot_render_and_aligned_batch_match_individual_calls():
    shape = (7, 7, 7)
    labels = torch.arange(shape[0])[:, None, None] < 3
    first = labels.expand(shape).to(torch.float64)
    volume = torch.stack((first, 1.0 - first))
    states = torch.stack(
        (
            _state(torch.tensor([3.7, 3.6, 3.8], dtype=torch.float64),
                   [1.0, 0.2, 0.1, -0.1, 1.0, 0.3], [2.0, 1.7], 0.05),
            _state(torch.tensor([3.4, 3.8, 3.5], dtype=torch.float64),
                   [0.2, 1.0, 0.1, 1.0, -0.1, 0.2], [1.8, 2.1], -0.08),
        )
    )
    offsets, weights = primitives.normalized_finite_psf_kernel(
        torch.tensor(1.0, dtype=torch.float64), 3, "boxcar"
    )
    batched = primitives.render_finite_thickness_plane(
        volume,
        states,
        (2, 3),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        offsets,
        weights,
    )
    individual = torch.cat(
        [
            primitives.render_finite_thickness_plane(
                volume,
                state,
                (2, 3),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                offsets,
                weights,
            )
            for state in states
        ]
    )
    assert torch.allclose(batched, individual, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        batched.sum(dim=1), torch.ones_like(batched[:, 0]), atol=1e-12, rtol=0.0
    )


def test_finite_thickness_renderer_gradcheck_reaches_all_continuous_inputs():
    torch.manual_seed(31)
    volume = torch.rand(1, 5, 6, 7, dtype=torch.float64)
    center = torch.tensor([2.7, 3.1, 3.6], dtype=torch.float64, requires_grad=True)
    rotation = torch.tensor(
        [0.8, -0.2, 1.1, 0.1, 1.2, -0.3], dtype=torch.float64, requires_grad=True
    )
    log_diagonal = torch.log(
        torch.tensor([1.4, 1.2], dtype=torch.float64)
    ).requires_grad_()
    shear = torch.tensor(0.09, dtype=torch.float64, requires_grad=True)
    thickness = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)

    def render(c, r, log_d, sh, thick, sig):
        state = torch.cat((c, r, log_d, sh[None]))
        offsets, weights = primitives.normalized_finite_psf_kernel(
            thick, 3, "gaussian", gaussian_sigma_um=sig
        )
        return primitives.render_finite_thickness_plane(
            volume,
            state,
            (2, 2),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            offsets,
            weights,
        )

    inputs = (center, rotation, log_diagonal, shear, thickness, sigma)
    assert torch.autograd.gradcheck(
        render, inputs, eps=1e-6, atol=3e-4, rtol=3e-3, fast_mode=True
    )
    render(*inputs).square().sum().backward()
    for value in inputs:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


def test_new_primitive_source_has_no_legacy_model_checkpoint_or_filesystem_dependency():
    tree = ast.parse(inspect.getsource(primitives))
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

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert imports <= {
        "__future__",
        "math",
        "torch",
        "torch.nn.functional",
        "training.arbitrary_plane_geometry",
    }
    forbidden_import_prefixes = (
        "os",
        "pathlib",
        "pickle",
        "timm",
        "torchvision",
        "training.atlas_pose_models",
        "training.dense_registration_model",
        "training.independent_joint",
        "training.joint_pose_registration",
    )
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden_import_prefixes
    )
    assert not ({"torch.load", "load_state_dict", "open"} & calls)
    assert primitives.FULL_FRAME_STATE_SIZE == 12
    assert primitives.FULL_FRAME_UPDATE_SIZE == 9
    for function in (
        primitives.full_frame_state_from_components,
        primitives.full_frame_state_to_components,
        primitives.compose_full_frame_state,
        primitives.render_finite_thickness_plane,
    ):
        assert not any(
            "reflect" in parameter
            for parameter in inspect.signature(function).parameters
        )
    source_strings = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any(
        token in value
        for value in source_strings
        for token in ("product5", "checkpoint", "resume_state", ".pt")
    )
