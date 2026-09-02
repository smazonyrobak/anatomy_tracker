import ast
import inspect
import math

import pytest
import torch

import training.arbitrary_plane_joint_model as joint_module
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
)
from training.arbitrary_plane_geometry import positive_inplane_basis


def _state(center, dtype=torch.float32):
    center = torch.tensor(center, dtype=dtype)
    frame = torch.eye(3, dtype=dtype)
    basis = positive_inplane_basis(
        torch.log(torch.tensor((4.0, 3.5), dtype=dtype)),
        torch.tensor(0.03, dtype=dtype),
    )
    return full_frame_state_from_components(center, frame, basis)


def _fixture(batch=1, cells=3, representations=2, dtype=torch.float32):
    torch.manual_seed(43)
    states = torch.stack(
        [
            torch.stack(
                [
                    _state((4.7 + 0.2 * cell, 5.0 + 0.1 * row, 5.1), dtype)
                    for cell in range(cells)
                ]
            )
            for row in range(batch)
        ]
    )
    raster = torch.eye(2, 3, dtype=dtype).expand(
        batch, cells, representations, 2, 3
    ).clone()
    return {
        "volume": torch.rand(2, 10, 10, 10, dtype=dtype),
        "image": torch.rand(batch, 1, 8, 8, dtype=dtype),
        "outline": torch.ones(batch, 1, 8, 8, dtype=dtype),
        "available": torch.tensor(
            [row % 2 for row in range(batch)], dtype=dtype
        ),
        "cell_id": torch.arange(cells),
        "states": states,
        "log_mass": torch.log(
            torch.tensor([1.0 + cell for cell in range(cells)], dtype=dtype)
        )[None].expand(batch, -1),
        "log_weight": torch.full(
            (batch, cells, representations),
            -math.log(representations),
            dtype=dtype,
        ),
        "raster": raster,
    }


def _model(dtype=torch.float32):
    torch.manual_seed(47)
    return joint_module.ArbitraryPlaneJointModel(
        atlas_channels=2,
        feature_channels=4,
        hidden_channels=6,
        correlation_radius=1,
        update_limits=(0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        plane_tangent_scales=(0.08, 0.08, 0.5),
        max_velocity_fraction_yx=(0.05, 0.04),
        deformation_integration_steps=4,
    ).to(dtype=dtype)


def _forward(model, fixture, **kwargs):
    top_k = kwargs.pop("top_k", 2)
    refinement_steps = kwargs.pop("refinement_steps", 3)
    pose_only_steps = kwargs.pop("pose_only_steps", 2)
    return model(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["cell_id"],
        fixture["states"],
        fixture["log_mass"],
        fixture["log_weight"],
        fixture["raster"],
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        torch.tensor([-0.5, 0.0, 0.5], dtype=fixture["volume"].dtype),
        torch.tensor([0.25, 0.5, 0.25], dtype=fixture["volume"].dtype),
        expected_catalogue_cell_count=fixture["states"].shape[1],
        top_k=top_k,
        refinement_steps=refinement_steps,
        pose_only_steps=pose_only_steps,
        **kwargs,
    )


def test_joint_shapes_fixed_gate_and_exact_inactive_identity():
    output = _forward(_model(), _fixture())
    pose = output["pose"]
    assert pose["refinement_representation_context_sequence"].shape == (
        1,
        2,
        2,
        4,
        6,
        2,
        2,
    )
    assert torch.equal(
        output["deformation_active_sequence"],
        torch.tensor([False, False, True, True]),
    )
    assert output["stationary_velocity_yx_px_sequence"].shape == (
        1,
        2,
        4,
        2,
        8,
        8,
    )
    assert output["support_logits_sequence"].shape == (1, 2, 4, 1, 8, 8)
    assert output["forward_jacobian_determinant_sequence"].shape == (
        1,
        2,
        4,
        1,
        8,
        8,
    )
    assert not output["deformation_representation_probability_sequence"].requires_grad
    assert torch.count_nonzero(
        output["stationary_velocity_yx_px_sequence"][:, :, :2]
    ) == 0
    assert torch.count_nonzero(
        output["forward_then_inverse_error_yx_sequence"][:, :, :2]
    ) == 0
    identity = joint_module.identity_pixel_map_yx(1, (8, 8))
    assert torch.equal(
        output["forward_map_yx_px_sequence"][:, :, :2],
        identity[:, None, None].expand(1, 2, 2, 2, 8, 8),
    )
    assert torch.equal(
        output["forward_jacobian_determinant_sequence"][:, :, :2],
        torch.ones(1, 2, 2, 1, 8, 8),
    )
    assert output["final_deformed_canonical_render"].shape == (1, 2, 2, 8, 8)
    assert torch.allclose(
        output["final_deformed_canonical_render"],
        output["final_feedback_deformed_canonical_render"],
        atol=2e-6,
        rtol=0.0,
    )
    assert output["deformation_gating_audit"] == {
        "pose_only_steps": 2,
        "gate_policy": "fixed_iteration_index_and_dense_supervision",
        "update_semantics": "absolute_per_iteration_not_accumulated",
        "representation_probabilities_detached": True,
        "dense_supervision_feedback_gate": "positive_weight_only; censored rows use detached identity",
        "feedback_semantics": "absolute_deformation_warps_next_finite_thickness_render",
        "shared_recurrent_context": True,
    }


def test_representation_permutation_and_exact_duplicate_split_are_invariant():
    model = _model().eval()
    fixture = _fixture(representations=2)
    fixture["raster"][:, :, 1, 1, 1] = -1.0
    first = _forward(model, fixture)
    permutation = torch.tensor([1, 0])
    fixture["raster"] = fixture["raster"][:, :, permutation]
    fixture["log_weight"] = fixture["log_weight"][:, :, permutation]
    permuted = _forward(model, fixture)
    for key in (
        "stationary_velocity_yx_px_sequence",
        "forward_map_yx_px_sequence",
        "final_representation_marginalized_canonical_render",
        "final_deformed_canonical_render",
    ):
        assert torch.allclose(first[key], permuted[key], atol=2e-6, rtol=0.0)

    single = _forward(model, _fixture(representations=1))
    duplicate = _forward(model, _fixture(representations=2))
    for key in (
        "stationary_velocity_yx_px_sequence",
        "forward_map_yx_px_sequence",
        "final_representation_marginalized_canonical_render",
        "final_deformed_canonical_render",
    ):
        assert torch.allclose(single[key], duplicate[key], atol=2e-6, rtol=0.0)


def test_deformation_feedback_changes_only_the_active_pose_suffix():
    model = _model().eval()
    fixture = _fixture()
    first = _forward(model, fixture)
    with torch.no_grad():
        for parameter in model.deformation_decoder.parameters():
            parameter.normal_(mean=0.5, std=0.2)
    second = _forward(model, fixture)
    assert torch.equal(
        first["pose"]["refinement_representation_context_sequence"][:, :, :, :2],
        second["pose"]["refinement_representation_context_sequence"][:, :, :, :2],
    )
    assert torch.equal(
        first["pose"]["refined_cell_state_sequence"][:, :, :3],
        second["pose"]["refined_cell_state_sequence"][:, :, :3],
    )
    assert torch.equal(
        first["pose"]["refinement_cell_update_sequence"][:, :, :2],
        second["pose"]["refinement_cell_update_sequence"][:, :, :2],
    )
    assert not torch.equal(
        first["pose"]["refinement_representation_context_sequence"][:, :, :, 2:],
        second["pose"]["refinement_representation_context_sequence"][:, :, :, 2:],
    )
    assert not torch.equal(
        first["pose"]["final_cell_state"], second["pose"]["final_cell_state"]
    )
    assert not torch.allclose(
        first["final_stationary_velocity_yx_px"],
        second["final_stationary_velocity_yx_px"],
    )


def test_each_active_iteration_is_an_absolute_shared_decoder_prediction():
    model = _model().eval()
    output = _forward(model, _fixture())
    contexts = output["deformation_cell_context_sequence"]
    for iteration in (2, 3):
        context = contexts[:, :, iteration].reshape(-1, *contexts.shape[3:])
        expected = model.deformation_decoder(context, (8, 8))[
            "stationary_velocity_yx_px"
        ].reshape(1, 2, 2, 8, 8)
        assert torch.allclose(
            output["stationary_velocity_yx_px_sequence"][:, :, iteration],
            expected,
            atol=0.0,
            rtol=0.0,
        )
    previous = output["stationary_velocity_yx_px_sequence"][:, :, 2]
    final = output["final_stationary_velocity_yx_px"]
    assert torch.count_nonzero(previous) > 0
    assert not torch.allclose(final, previous + final, atol=1e-8, rtol=0.0)


def test_final_deformed_render_uses_yx_output_to_input_pullback(monkeypatch):
    model = _model().eval()

    def one_row_pullback(context, output_shape_h_w):
        result = joint_module._inactive_deformation(context, output_shape_h_w)
        result["forward_map_yx_px"] = result["forward_map_yx_px"].clone()
        result["forward_map_yx_px"][:, 0] += 1.0
        return result

    monkeypatch.setattr(model.deformation_decoder, "forward", one_row_pullback)
    output = _forward(model, _fixture(), pose_only_steps=3)
    source = output["final_representation_marginalized_canonical_render"]
    observed = output["final_deformed_canonical_render"]
    assert torch.allclose(observed[..., :-1, :], source[..., 1:, :], atol=1e-6, rtol=0.0)
    assert torch.allclose(
        observed[..., -1, :], torch.zeros_like(observed[..., -1, :]), atol=1e-6, rtol=0.0
    )


def test_joint_losses_reach_pose_and_deformation_parameters():
    fixture = _fixture()
    fixture["volume"].requires_grad_()
    fixture["image"].requires_grad_()
    fixture["states"].requires_grad_()
    model = _model().train()
    output = _forward(model, fixture)
    spatial_weight = torch.linspace(-1.0, 1.0, 8)[None, None, None, None]
    loss = (
        output["pose"]["cell_log_unnormalized_mass"].square().mean()
        + output["pose"]["final_cell_state"].square().mean()
        + (output["final_stationary_velocity_yx_px"] * spatial_weight).sum()
        + output["final_support_logits"].square().mean()
        + output["final_deformed_canonical_render"].square().mean()
    )
    loss.backward()
    for value in (fixture["volume"], fixture["image"], fixture["states"]):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    for parameter in (
        model.pose_model.candidate_log_likelihood.weight,
        model.pose_model.recurrent_cell.gates.weight,
        model.pose_model.recurrent_update.weight,
        model.deformation_decoder.trunk[0].weight,
        model.deformation_decoder.velocity_head.weight,
        model.deformation_decoder.support_head.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_final_pose_objective_reaches_decoder_only_through_recurrent_feedback():
    model = _model().train()
    output = _forward(model, _fixture(), pose_only_steps=2, refinement_steps=3)
    loss = output["pose"]["final_cell_state"].square().mean()
    loss.backward()
    gradient = model.deformation_decoder.velocity_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_dense_censored_sample_uses_detached_identity_feedback_for_pose():
    model = _model().train()
    output = _forward(
        model,
        _fixture(),
        pose_only_steps=0,
        refinement_steps=3,
        dense_deformation_supervision_weight=torch.zeros(1),
    )
    identity = joint_module.identity_pixel_map_yx(1, (8, 8))
    assert not output["deformation_feedback_enabled_mask"].any()
    assert torch.equal(
        output["deformation_feedback_map_yx_px_sequence"],
        identity[:, None, None].expand(1, 2, 4, 2, 8, 8),
    )
    loss = output["pose"]["final_cell_state"].square().mean()
    loss.backward()
    pose_gradient = model.pose_model.recurrent_update.weight.grad
    assert pose_gradient is not None and torch.count_nonzero(pose_gradient) > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.deformation_decoder.parameters()
    )


def test_explicit_dense_one_preserves_normal_joint_outputs_exactly():
    model = _model().eval()
    fixture = _fixture()
    default = _forward(model, fixture)
    explicit = _forward(
        model,
        fixture,
        dense_deformation_supervision_weight=torch.ones(1),
    )
    assert explicit["deformation_feedback_enabled_mask"].all()
    for key in (
        "stationary_velocity_yx_px_sequence",
        "deformation_feedback_map_yx_px_sequence",
        "final_deformed_canonical_render",
    ):
        assert torch.equal(default[key], explicit[key])
    assert torch.equal(
        default["pose"]["final_cell_state"], explicit["pose"]["final_cell_state"]
    )


def test_dense_feedback_gate_is_independent_for_each_batch_row():
    output = _forward(
        _model().eval(),
        _fixture(batch=2),
        pose_only_steps=0,
        dense_deformation_supervision_weight=torch.tensor([0.0, 1.0]),
    )
    identity = joint_module.identity_pixel_map_yx(1, (8, 8))
    assert torch.equal(
        output["deformation_feedback_enabled_mask"], torch.tensor([False, True])
    )
    assert torch.equal(
        output["deformation_feedback_map_yx_px_sequence"][0],
        identity[:, None].expand(2, 4, 2, 8, 8),
    )
    assert torch.equal(
        output["deformation_feedback_map_yx_px_sequence"][1],
        output["forward_map_yx_px_sequence"][1],
    )


def test_gate_rejects_data_dependent_or_out_of_range_values():
    with pytest.raises(ValueError, match="between zero and T"):
        _forward(_model(), _fixture(), pose_only_steps=5)
    with pytest.raises(ValueError, match="between zero and T"):
        _forward(_model(), _fixture(), pose_only_steps=True)


def test_joint_model_has_no_prior_model_checkpoint_or_filesystem_dependency():
    tree = ast.parse(inspect.getsource(joint_module))
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
    signature = inspect.signature(joint_module.ArbitraryPlaneJointModel)
    assert not any(
        token in name.lower()
        for name in signature.parameters
        for token in ("checkpoint", "pretrained", "legacy")
    )


def test_joint_model_uses_streamed_low_resolution_catalogue_path():
    model = _model().eval()
    output = _forward(
        model,
        _fixture(cells=5),
        top_k=2,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=2,
    )
    assert output["pose"]["retrieval_execution"] == "checkpointed_low_resolution_chunks"
    assert output["pose"]["catalogue_chunk_size"] == 2
    assert output["pose"]["cell_log_unnormalized_mass"].shape == (1, 5)
    assert output["final_stationary_velocity_yx_px"].shape == (1, 2, 2, 8, 8)
