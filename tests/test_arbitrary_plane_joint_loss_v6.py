import math

import pytest
import torch

from training.arbitrary_plane_deformation_primitives import identity_pixel_map_yx
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import positive_inplane_basis
from training.arbitrary_plane_joint_loss_v6 import arbitrary_plane_joint_loss_v6
from tests.test_arbitrary_plane_joint_model_v6 import (
    _fixture as _joint_fixture,
    _forward as _joint_forward,
    _install_cascade,
    _model as _joint_model,
    _runtime,
)


def _state(ap):
    return full_frame_state_from_components(
        torch.tensor((ap, 2.0, 2.0)),
        torch.eye(3),
        positive_inplane_basis(torch.log(torch.tensor((3.0, 3.0))), torch.tensor(0.0)),
    )


def _log_probability(probability, valid=None):
    probability = torch.tensor(probability, dtype=torch.float32)
    if valid is None:
        valid = torch.ones_like(probability, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid, dtype=torch.bool)
    return torch.where(valid, probability.log(), torch.full_like(probability, -torch.inf))


def _case():
    batch, cells, refined_rows, topk, steps, height, width = 3, 4, 2, 2, 2, 3, 3
    proposal_logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [0.0, 1.0, 2.0, -1.0], [0.0, 1.0, 2.0, -1.0]],
        requires_grad=True,
    )
    honest_selected = torch.tensor([[0, 1], [0, 1], [1, 2]])
    honest_valid = torch.ones(batch, 2, dtype=torch.bool)
    honest_conditional = _log_probability([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    teacher_selected = torch.tensor([[0, 1, 0], [0, 1, 2], [1, 2, 0]])
    teacher_valid = torch.tensor(
        [[True, True, False], [True, True, True], [True, True, False]]
    )
    teacher_conditional = _log_probability(
        [[0.8, 0.2, 0.0], [0.1, 0.1, 0.8], [0.3, 0.7, 0.0]],
        teacher_valid,
    )
    honest_topk = torch.tensor([[0, 1], [0, 1], [2, 1]])
    cascade = {
        "raw_full_catalogue_proposal_log_probability": torch.log_softmax(
            proposal_logits, dim=1
        ),
        "honest_initial_topm_catalogue_index": torch.tensor(
            [[0, 1], [0, 1], [1, 2]]
        ),
        "honest_initial_topm_truth_hit": torch.tensor([True, False, True]),
        "honest_hybrid_posterior": {
            "selected_catalogue_index": honest_selected,
            "selected_valid_mask": honest_valid,
            "selected_conditional_log_probability": honest_conditional,
            "hybrid_topk_catalogue_index": honest_topk,
        },
        "training_truth_catalogue_index": torch.tensor([0, 2, 2]),
        "training_truth_forced_mask": torch.tensor([False, True, False]),
        "training_selected_catalogue_index": teacher_selected,
        "training_selected_valid_mask": teacher_valid,
        "training_teacher_forced_hybrid_posterior": {
            "selected_catalogue_index": teacher_selected,
            "selected_valid_mask": teacher_valid,
            "selected_conditional_log_probability": teacher_conditional,
        },
        "honest_refinement_ready_mask": torch.tensor([True, False, True]),
    }
    truth_state = torch.stack((_state(1.0), _state(2.0), _state(3.0)))
    initial_state = torch.stack(
        (
            torch.stack((_state(1.2), _state(4.0))),
            torch.stack((_state(4.0), _state(2.0))),
        )
    ).requires_grad_()
    final_state = torch.stack(
        (
            torch.stack((_state(1.1), _state(3.0))),
            torch.stack((_state(3.8), _state(2.5))),
        )
    ).requires_grad_()
    covariance = torch.eye(3).expand(refined_rows, topk, 3, 3).clone()
    legacy_ranker_logits = torch.zeros(refined_rows, topk, requires_grad=True)
    pose = {
        "retrieval_topk_catalogue_index": honest_topk[[0, 2]].clone(),
        "retrieval_topk_cell_id": honest_topk[[0, 2]].clone(),
        "retrieval_topk_log_probability": torch.log_softmax(
            legacy_ranker_logits, dim=1
        ),
        "refinement_initial_topk_log_probability": _log_probability(
            [[0.7, 0.3], [0.4, 0.6]]
        ),
        "conditional_within_topk_cell_log_probability": _log_probability(
            [[0.9, 0.1], [0.6, 0.4]]
        ),
        "topk_initial_cell_state": initial_state,
        "topk_initial_cell_canonical_plane_covariance": covariance.clone(),
        "final_cell_state": final_state,
        "final_cell_canonical_plane_covariance": covariance.clone(),
    }
    velocity = torch.zeros(refined_rows, topk, steps, 2, height, width)
    velocity[0, 0] = 0.2
    velocity[1, 0] = 1.0
    velocity.requires_grad_()
    identity = identity_pixel_map_yx(refined_rows, (height, width))
    pullback = identity[:, None, None].expand(-1, topk, steps, -1, -1, -1).clone()
    pullback[0, 0] += 0.1
    pullback[1, 0] += 0.8
    pullback.requires_grad_()
    vector_shape = (refined_rows, topk, steps, 2, height, width)
    scalar_shape = (refined_rows, topk, steps, 1, height, width)
    refined = {
        "pose": pose,
        "stationary_velocity_yx_px_sequence": velocity,
        "pullback_map_yx_px_sequence": pullback,
        "support_logits_sequence": torch.zeros(scalar_shape, requires_grad=True),
        "forward_jacobian_determinant_sequence": torch.ones(scalar_shape),
        "forward_then_inverse_error_yx_sequence": torch.zeros(vector_shape),
        "inverse_then_forward_error_yx_sequence": torch.zeros(vector_shape),
        "forward_then_inverse_valid_mask_sequence": torch.ones(
            scalar_shape, dtype=torch.bool
        ),
        "inverse_then_forward_valid_mask_sequence": torch.ones(
            scalar_shape, dtype=torch.bool
        ),
        "deformation_active_sequence": torch.ones(steps, dtype=torch.bool),
    }
    output = {
        "cascade": cascade,
        "refined_output": refined,
        "refinement_ready_mask": torch.tensor([True, False, True]),
        "refinement_abstained_mask": torch.tensor([False, True, False]),
        "refinement_performed_mask": torch.tensor([True, False, True]),
        "refinement_source_batch_index": torch.tensor([0, 2]),
        "refinement_teacher_forced_mask": torch.tensor([False, False, False]),
        "refinement_selected_catalogue_index": honest_topk[[0, 2]].clone(),
        "refinement_selected_cell_id": honest_topk[[0, 2]].clone(),
        "refinement_initial_honest_topk_catalogue_index": honest_topk[[0, 2]].clone(),
        "refinement_initial_honest_mode_mask": torch.ones(
            refined_rows, topk, dtype=torch.bool
        ),
        "refinement_final_honest_mode_mask": torch.ones(
            refined_rows, topk, dtype=torch.bool
        ),
        "refinement_final_teacher_forced_mode_mask": torch.zeros(
            refined_rows, topk, dtype=torch.bool
        ),
        "refinement_truth_topk_index": torch.tensor([0, 0]),
        "refinement_initial_topk_log_probability": pose[
            "refinement_initial_topk_log_probability"
        ],
    }
    truth_map = identity_pixel_map_yx(batch, (height, width))
    arguments = {
        "output": output,
        "truth_state": truth_state,
        "truth_catalogue_index": torch.tensor([0, 2, 2]),
        "truth_stationary_velocity_yx_px": torch.zeros(batch, 2, height, width),
        "truth_pullback_map_yx_px": truth_map,
        "deformation_weight": torch.ones(batch, 1, height, width),
        "support_origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
        "expected_catalogue_cell_count": cells,
    }
    return arguments, proposal_logits, legacy_ranker_logits, initial_state


def _loss(arguments, **updates):
    return arbitrary_plane_joint_loss_v6(**(arguments | updates))


def _clone(value):
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _joint_cascade(model, ready, truth, teacher=False):
    batch = len(ready)
    cells = model.pose_model.catalogue_runtime_v6.cell_count
    proposal_logits = torch.linspace(1.0, -1.0, cells).repeat(batch, 1)
    proposal_logits.requires_grad_()
    full_log = torch.log_softmax(proposal_logits, dim=1)
    topk = torch.tensor([[0, 1], [2, 3]][:batch])
    selected_log = torch.gather(full_log, 1, topk)
    selected_conditional = torch.log_softmax(selected_log, dim=1)
    truth = torch.tensor(truth, dtype=torch.long)
    hit = topk.eq(truth[:, None]).any(dim=1)
    finite = torch.ones_like(topk, dtype=torch.bool)
    for row, is_ready in enumerate(ready):
        if not is_ready:
            finite[row, -1] = False
    honest = {
        "selected_catalogue_index": topk,
        "selected_valid_mask": torch.ones_like(topk, dtype=torch.bool),
        "selected_conditional_log_probability": selected_conditional,
        "hybrid_topk_catalogue_index": topk,
        "hybrid_cell_log_probability": full_log,
        "hybrid_topk_finite_rendered_mask": finite,
        "hybrid_topk_retained_probability": selected_log.exp().sum(1),
        "hybrid_omitted_probability": 1.0 - selected_log.exp().sum(1),
    }
    cascade = {
        "raw_full_catalogue_proposal_log_probability": full_log,
        "honest_initial_topm_catalogue_index": topk,
        "honest_initial_topm_truth_hit": hit,
        "honest_hybrid_posterior": honest,
        "training_truth_catalogue_index": truth if teacher else None,
        "training_truth_forced_mask": ~hit if teacher else torch.zeros_like(hit),
        "honest_refinement_ready_mask": torch.tensor(ready, dtype=torch.bool),
    }
    if teacher:
        teacher_index = torch.cat((topk, truth[:, None]), dim=1)
        teacher_valid = torch.cat(
            (torch.ones_like(topk, dtype=torch.bool), (~hit)[:, None]), dim=1
        )
        teacher_selected_log = torch.gather(full_log, 1, teacher_index)
        teacher_conditional = torch.log_softmax(
            teacher_selected_log.masked_fill(~teacher_valid, -torch.inf), dim=1
        )
        teacher_hybrid = {
            "selected_catalogue_index": teacher_index,
            "selected_valid_mask": teacher_valid,
            "selected_conditional_log_probability": teacher_conditional,
            "hybrid_cell_log_probability": full_log,
            "finite_rendered_mask": torch.ones_like(full_log, dtype=torch.bool),
        }
        cascade["training_selected_catalogue_index"] = teacher_index
        cascade["training_selected_valid_mask"] = teacher_valid
        cascade["training_teacher_forced_hybrid_posterior"] = teacher_hybrid
    else:
        cascade["training_teacher_forced_hybrid_posterior"] = None
    return cascade, proposal_logits


def _joint_loss_arguments(output, truth_state, truth_index, shape=(8, 8)):
    batch = truth_state.shape[0]
    return {
        "output": output,
        "truth_state": truth_state,
        "truth_catalogue_index": truth_index,
        "truth_stationary_velocity_yx_px": torch.zeros(batch, 2, *shape),
        "truth_pullback_map_yx_px": identity_pixel_map_yx(batch, shape),
        "deformation_weight": torch.ones(batch, 1, *shape),
        "support_origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
        "expected_catalogue_cell_count": output["cascade"][
            "raw_full_catalogue_proposal_log_probability"
        ].shape[1],
    }


def test_fractional_pose_weights_use_exact_per_row_normalization():
    arguments, _, _, _ = _case()
    first = _loss(arguments, pose_supervision_weight=torch.tensor([1.0, 0.0, 0.0]))
    second = _loss(arguments, pose_supervision_weight=torch.tensor([0.0, 0.0, 1.0]))
    mixed = _loss(arguments, pose_supervision_weight=torch.tensor([0.1, 0.0, 0.3]))
    for name in (
        "initial_plane_mixture_nll",
        "final_plane_mixture_nll",
        "final_landmark_mixture_nll",
        "deformation_svf",
        "deformation_map",
        "deformation_support",
    ):
        torch.testing.assert_close(mixed[name], 0.25 * first[name] + 0.75 * second[name])
    assert mixed["refinement_pose_eligible_weight_sum"] == pytest.approx(0.4)


def test_teacher_forced_and_honest_retrieval_statistics_are_disjoint_and_exact():
    arguments, _, _, _ = _case()
    losses = _loss(arguments)
    expected = -(math.log(0.8) + math.log(0.8) + math.log(0.7)) / 3.0
    assert losses["honest_initial_selected_row_count"] == 2
    assert losses["honest_initial_miss_row_count"] == 1
    assert losses["honest_final_selected_row_count"] == 2
    assert losses["honest_final_miss_row_count"] == 1
    assert losses["teacher_forced_row_count"] == 1
    assert losses["rerank_eligible_row_count"] == 3
    assert losses["selected_finite_render_conditional_rerank_nll"] == pytest.approx(
        expected
    )
    assert losses["teacher_forced_rerank_nll"] == pytest.approx(-math.log(0.8))


def test_legacy_ranker_and_retained_mass_cannot_enter_v6_refinement_loss():
    arguments, proposal_logits, legacy_ranker_logits, _ = _case()
    losses = _loss(arguments)
    losses["total"].backward()
    assert torch.count_nonzero(proposal_logits.grad) > 0
    assert legacy_ranker_logits.grad is None
    changed = _clone(arguments)
    replacement = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    changed["output"]["refined_output"]["pose"]["retrieval_topk_log_probability"] = (
        torch.log_softmax(replacement, dim=1)
    )
    for name in (
        "initial_plane_mixture_nll",
        "final_plane_mixture_nll",
        "final_landmark_mixture_nll",
        "refinement_total",
    ):
        torch.testing.assert_close(_loss(changed)[name], losses[name])


def test_zero_refined_rows_are_graph_connected_and_zero_safe():
    arguments, proposal_logits, _, initial_state = _case()
    output = _clone(arguments["output"])
    output["cascade"]["honest_refinement_ready_mask"] = torch.zeros(3, dtype=torch.bool)
    output["refinement_ready_mask"] = torch.zeros(3, dtype=torch.bool)
    output["refinement_abstained_mask"] = torch.ones(3, dtype=torch.bool)
    output["refinement_performed_mask"] = torch.zeros(3, dtype=torch.bool)
    output["refinement_source_batch_index"] = torch.empty(0, dtype=torch.long)
    output["refined_output"] = None
    losses = _loss(
        arguments,
        output=output,
        retrieval_supervision_weight=torch.zeros(3),
    )
    assert losses["total"] == 0.0
    assert losses["refinement_total"] == 0.0
    assert losses["refinement_ready_row_count"] == 0
    assert losses["refinement_abstained_row_count"] == 3
    losses["total"].backward()
    assert torch.equal(proposal_logits.grad, torch.zeros_like(proposal_logits))
    assert initial_state.grad is None or torch.equal(
        initial_state.grad, torch.zeros_like(initial_state)
    )


def test_abstained_full_batch_truth_is_never_read_by_refinement_terms():
    arguments, _, _, _ = _case()
    baseline = _loss(arguments)
    poisoned = dict(arguments)
    poisoned["truth_state"] = arguments["truth_state"].clone()
    poisoned["truth_state"][1] = torch.nan
    poisoned["truth_stationary_velocity_yx_px"] = arguments[
        "truth_stationary_velocity_yx_px"
    ].clone()
    poisoned["truth_stationary_velocity_yx_px"][1] = torch.nan
    poisoned["truth_pullback_map_yx_px"] = arguments["truth_pullback_map_yx_px"].clone()
    poisoned["truth_pullback_map_yx_px"][1] = torch.inf
    poisoned["deformation_weight"] = arguments["deformation_weight"].clone()
    poisoned["deformation_weight"][1] = torch.nan
    changed = _loss(poisoned)
    for name in (
        "initial_plane_mixture_nll",
        "final_plane_mixture_nll",
        "final_landmark_mixture_nll",
        "deformation_svf",
        "deformation_map",
        "refinement_total",
    ):
        torch.testing.assert_close(changed[name], baseline[name], rtol=0.0, atol=0.0)


def test_mapping_and_canonical_probability_contracts_fail_closed():
    arguments, _, _, _ = _case()
    wrong_mapping = _clone(arguments)
    wrong_mapping["output"]["refinement_source_batch_index"] = torch.tensor([2, 0])
    with pytest.raises(ValueError, match="exactly match honest ready rows"):
        _loss(wrong_mapping)

    wrong_ids = _clone(arguments)
    wrong_ids["output"]["refined_output"]["pose"]["retrieval_topk_cell_id"][0, 0] = 3
    with pytest.raises(ValueError, match="identical canonical IDs"):
        _loss(wrong_ids)

    retained_mass = _clone(arguments)
    retained_mass["output"]["refined_output"]["pose"][
        "refinement_initial_topk_log_probability"
    ] += math.log(0.5)
    retained_mass["output"]["refinement_initial_topk_log_probability"] += math.log(
        0.5
    )
    with pytest.raises(ValueError, match="normalized across"):
        _loss(retained_mass)


def test_actual_joint_model_mixed_compact_output_flows_into_native_loss():
    runtime = _runtime()
    model = _joint_model(runtime).train()
    fixture = _joint_fixture(runtime)
    cascade, _ = _joint_cascade(model, [True, False], [0, 2])
    _install_cascade(model, cascade)
    output = _joint_forward(model, fixture)
    truth_state = torch.stack(
        (output["refined_output"]["pose"]["final_cell_state"][0, 0].detach(), _state(2.0))
    )
    losses = _loss(_joint_loss_arguments(output, truth_state, torch.tensor([0, 2])))
    assert losses["refinement_ready_row_count"] == 1
    assert losses["refinement_abstained_row_count"] == 1
    losses["refinement_total"].backward()
    ranker_gradient = model.pose_model.candidate_log_likelihood.weight.grad
    proposal_gradients = [
        parameter.grad for parameter in model.pose_model.proposal_head_v6.parameters()
    ]
    assert ranker_gradient is None or torch.count_nonzero(ranker_gradient) == 0
    assert all(
        gradient is None or torch.count_nonzero(gradient) == 0
        for gradient in proposal_gradients
    )
    assert torch.count_nonzero(model.pose_model.recurrent_update.weight.grad) > 0


def test_actual_joint_model_all_abstained_is_retrieval_only_and_backward_safe():
    runtime = _runtime()
    model = _joint_model(runtime).eval()
    fixture = _joint_fixture(runtime)
    cascade, proposal_logits = _joint_cascade(model, [False, False], [0, 2])
    _install_cascade(model, cascade)
    output = _joint_forward(model, fixture)
    assert output["refined_output"] is None
    arguments = _joint_loss_arguments(
        output, torch.stack((_state(1.0), _state(2.0))), torch.tensor([0, 2])
    )
    losses = _loss(arguments, retrieval_supervision_weight=torch.zeros(2))
    assert losses["total"] == 0.0
    assert losses["refinement_total"] == 0.0
    losses["total"].backward()
    assert torch.equal(proposal_logits.grad, torch.zeros_like(proposal_logits))


def test_actual_joint_model_teacher_truth_replacement_is_derived_canonically():
    runtime = _runtime()
    model = _joint_model(runtime).train()
    fixture = _joint_fixture(runtime, batch=1)
    cascade, _ = _joint_cascade(model, [True], [3], teacher=True)
    _install_cascade(model, cascade)
    output = _joint_forward(
        model, fixture, training_truth_catalogue_index=torch.tensor([3])
    )
    assert output["refinement_selected_catalogue_index"].tolist() == [[0, 3]]
    truth_state = output["refined_output"]["pose"]["final_cell_state"][:, 1].detach()
    losses = _loss(_joint_loss_arguments(output, truth_state, torch.tensor([3])))
    assert losses["honest_final_miss_row_count"] == 1
    assert losses["teacher_forced_row_count"] == 1
    assert losses["refinement_topk_truth_selected_row_count"] == 1
    assert torch.isfinite(losses["total"])
