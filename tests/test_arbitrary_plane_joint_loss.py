import math

import torch

import training.arbitrary_plane_joint_loss as joint_loss
from training.arbitrary_plane_deformation_primitives import identity_pixel_map_yx
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import positive_inplane_basis
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel


def _state(center):
    return full_frame_state_from_components(
        torch.tensor(center, dtype=torch.float32),
        torch.eye(3),
        positive_inplane_basis(
            torch.log(torch.tensor((4.0, 3.5))), torch.tensor(0.03)
        ),
    )


def _model_output(batch=2):
    torch.manual_seed(47)
    model = ArbitraryPlaneJointModel(
        atlas_channels=2,
        feature_channels=4,
        hidden_channels=6,
        correlation_radius=1,
        update_limits=(0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        plane_tangent_scales=(0.08, 0.08, 0.5),
        max_velocity_fraction_yx=(0.05, 0.04),
        deformation_integration_steps=4,
    )
    cells = 3
    states = torch.stack(
        [
            torch.stack(
                [_state((4.7 + 0.2 * cell, 5.0 + 0.1 * row, 5.1)) for cell in range(cells)]
            )
            for row in range(batch)
        ]
    )
    raster = torch.eye(2, 3).expand(batch, cells, 2, 2, 3).clone()
    output = model(
        torch.rand(batch, 1, 8, 8),
        torch.ones(batch, 1, 8, 8),
        torch.tensor([row % 2 for row in range(batch)], dtype=torch.float32),
        torch.rand(2, 10, 10, 10),
        torch.arange(cells),
        states,
        torch.full((batch, cells), -math.log(cells)),
        torch.full((batch, cells, 2), -math.log(2.0)),
        raster,
        (8, 8),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        expected_catalogue_cell_count=cells,
        top_k=2,
        refinement_steps=3,
        pose_only_steps=2,
    )
    return model, output


def _truth(output):
    state = output["pose"]["final_cell_state"][:, 0].detach().clone()
    state[:, 0] += 0.15
    batch = state.shape[0]
    return dict(
        truth_state=state,
        truth_catalogue_cell_id=output["pose"]["retrieval_topk_cell_id"][:, 0],
        truth_topk_index=torch.zeros(batch, dtype=torch.long),
        truth_stationary_velocity_yx_px=torch.zeros(batch, 2, 8, 8),
        truth_pullback_map_yx_px=identity_pixel_map_yx(batch, (8, 8)),
        deformation_weight=torch.ones(batch, 1, 8, 8),
        support_origin_ap_dv_ml_um=(5.0, 5.0, 5.0),
    )


def test_plane_tangent_residual_has_exact_offset_and_antipodal_plane_semantics():
    prediction = _state((5.0, 5.0, 6.0))[None]
    truth = _state((5.0, 5.0, 7.25))[None]
    residual = joint_loss.plane_tangent_residual(
        prediction, truth, (5.0, 5.0, 5.0)
    )
    assert torch.allclose(residual, torch.tensor([[0.0, 0.0, 1.25]]), atol=1e-6)

    flipped_frame = torch.diag(torch.tensor((1.0, -1.0, -1.0)))
    flipped = full_frame_state_from_components(
        torch.tensor((5.0, 5.0, 7.25)),
        flipped_frame,
        positive_inplane_basis(torch.log(torch.tensor((4.0, 3.5))), torch.tensor(0.03)),
    )[None]
    assert torch.allclose(
        joint_loss.plane_tangent_residual(prediction, flipped, (5.0, 5.0, 5.0)),
        torch.tensor([[0.0, 0.0, 1.25]]),
        atol=1e-6,
    )


def test_joint_loss_is_finite_probabilistic_pose_gated_and_fully_differentiable():
    model, output = _model_output()
    losses = joint_loss.arbitrary_plane_joint_loss(output, **_truth(output))
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    for parameter in (
        model.pose_model.candidate_log_likelihood.weight,
        model.pose_model.candidate_plane_cholesky.weight,
        model.pose_model.recurrent_update.weight,
        model.pose_model.recurrent_plane_cholesky.weight,
        model.deformation_decoder.velocity_head.weight,
        model.deformation_decoder.support_head.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_deformation_losses_abstain_when_truth_cell_is_outside_topk():
    _, output = _model_output(batch=1)
    truth = _truth(output)
    truth["truth_topk_index"] = torch.tensor([-1])
    losses = joint_loss.arbitrary_plane_joint_loss(output, **truth)
    for key in (
        "deformation_svf",
        "deformation_map",
        "deformation_support",
        "deformation_topology",
        "deformation_smoothness",
        "deformation_inverse_consistency",
    ):
        assert losses[key] == 0.0


def test_topk_miss_trains_retrieval_without_inflating_local_uncertainty():
    model, output = _model_output(batch=1)
    truth = _truth(output)
    truth["truth_catalogue_cell_id"] = torch.tensor([2])
    truth["truth_topk_index"] = torch.tensor([-1])
    losses = joint_loss.arbitrary_plane_joint_loss(output, **truth)
    assert losses["retrieval_nll"] > 0.0
    assert losses["initial_plane_mixture_nll"] == 0.0
    assert losses["final_plane_mixture_nll"] == 0.0
    assert losses["final_landmark_mixture_nll"] == 0.0
    losses["total"].backward()
    assert model.pose_model.candidate_log_likelihood.weight.grad is not None
    assert torch.count_nonzero(model.pose_model.candidate_log_likelihood.weight.grad) > 0
    for parameter in (
        model.pose_model.candidate_plane_cholesky.weight,
        model.pose_model.recurrent_plane_cholesky.weight,
    ):
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_censored_marginal_support_has_no_false_point_or_dense_supervision():
    model, output = _model_output(batch=1)
    truth = _truth(output)
    truth["pose_supervision_weight"] = torch.zeros(1)
    truth["deformation_weight"] = torch.zeros_like(truth["deformation_weight"])
    losses = joint_loss.arbitrary_plane_joint_loss(output, **truth)
    for name in (
        "retrieval_nll",
        "initial_plane_mixture_nll",
        "final_plane_mixture_nll",
        "final_landmark_mixture_nll",
        "deformation_svf",
        "deformation_map",
        "deformation_support",
        "deformation_topology",
        "deformation_smoothness",
        "deformation_inverse_consistency",
    ):
        assert losses[name] == 0.0
    assert losses["pose_identifiable_fraction"] == 0.0
    losses["total"].backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.parameters()
    )


def test_fold_penalty_detects_negative_pullback_jacobians():
    _, output = _model_output(batch=1)
    truth = _truth(output)
    normal = joint_loss.arbitrary_plane_joint_loss(output, **truth)
    folded = dict(output)
    folded["forward_jacobian_determinant_sequence"] = -torch.ones_like(
        output["forward_jacobian_determinant_sequence"]
    )
    penalized = joint_loss.arbitrary_plane_joint_loss(folded, **truth)
    assert penalized["deformation_topology"] > normal["deformation_topology"]


def test_cycle_loss_ignores_out_of_domain_composition_residuals():
    _, output = _model_output(batch=1)
    truth = _truth(output)
    invalid = dict(output)
    invalid["forward_then_inverse_error_yx_sequence"] = torch.full_like(
        output["forward_then_inverse_error_yx_sequence"], 1.0e4
    )
    invalid["inverse_then_forward_error_yx_sequence"] = torch.full_like(
        output["inverse_then_forward_error_yx_sequence"], -1.0e4
    )
    invalid["forward_then_inverse_valid_mask_sequence"] = torch.zeros_like(
        output["forward_then_inverse_valid_mask_sequence"]
    )
    invalid["inverse_then_forward_valid_mask_sequence"] = torch.zeros_like(
        output["inverse_then_forward_valid_mask_sequence"]
    )
    losses = joint_loss.arbitrary_plane_joint_loss(invalid, **truth)
    assert losses["deformation_inverse_consistency"] == 0.0


def test_active_intermediate_deformations_receive_later_weighted_supervision():
    _, output = _model_output(batch=1)
    truth = _truth(output)
    baseline = joint_loss.arbitrary_plane_joint_loss(output, **truth)
    changed = dict(output)
    changed["stationary_velocity_yx_px_sequence"] = output[
        "stationary_velocity_yx_px_sequence"
    ].clone()
    first_active = int(
        torch.nonzero(output["deformation_active_sequence"], as_tuple=False)[0]
    )
    changed["stationary_velocity_yx_px_sequence"][:, :, first_active] += 2.0
    assert torch.equal(
        changed["stationary_velocity_yx_px_sequence"][:, :, -1],
        output["stationary_velocity_yx_px_sequence"][:, :, -1],
    )
    supervised = joint_loss.arbitrary_plane_joint_loss(changed, **truth)
    assert supervised["deformation_svf"] > baseline["deformation_svf"]
