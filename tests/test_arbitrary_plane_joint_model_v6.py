import math
import types

import pytest
import torch

from tests.arbitrary_plane_production_v3_fixtures import atlas, catalogue
from training.arbitrary_plane_catalogue_runtime_v6 import (
    make_complete_catalogue_runtime_v6,
)
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6


def _runtime():
    artifact = catalogue()
    return make_complete_catalogue_runtime_v6(
        artifact,
        expected_catalogue_receipt_sha256=artifact["receipt_sha256"],
        device="cpu",
        dtype=torch.float32,
    )


def _model(runtime, pose_only_steps=1):
    return ArbitraryPlaneJointModelV6(
        runtime,
        atlas_channels=2,
        feature_channels=4,
        hidden_channels=6,
        correlation_radius=1,
        proposal_channels=4,
        proposal_mixture_components=2,
        proposal_spatial_bins_h_w=(2, 2),
        cascade_max_rendered_cells_per_sample=4,
        cascade_max_closure_rounds=1,
        pose_only_steps=pose_only_steps,
        max_velocity_fraction_yx=(0.04, 0.04),
        deformation_integration_steps=2,
    )


def _fixture(runtime, batch=2):
    torch.manual_seed(73)
    image = torch.rand(batch, 1, 8, 8)
    return {
        "image": image,
        "outline": torch.ones_like(image),
        "available": torch.ones(batch),
        "volume": torch.from_numpy(atlas()),
        "bound": runtime.expand(batch),
    }


def _cascade(model, ready, topk=((0, 1), (2, 3)), teacher=False):
    batch = len(ready)
    cells = model.pose_model.catalogue_runtime_v6.cell_count
    probability = torch.tensor(
        [0.30, 0.24, 0.18, 0.14, 0.08, 0.06], dtype=torch.float32
    )
    probability = probability[:cells]
    probability = probability / probability.sum()
    full_log = probability.log()[None].expand(batch, -1).clone()
    index = torch.tensor(topk[:batch], dtype=torch.long)
    finite = torch.ones_like(index, dtype=torch.bool)
    for row, is_ready in enumerate(ready):
        if not is_ready:
            finite[row, -1] = False
    honest = {
        "hybrid_topk_catalogue_index": index,
        "hybrid_cell_log_probability": full_log,
        "hybrid_topk_finite_rendered_mask": finite,
        "hybrid_topk_retained_probability": torch.gather(
            full_log, 1, index
        ).exp().sum(1),
        "hybrid_omitted_probability": 1.0
        - torch.gather(full_log, 1, index).exp().sum(1),
    }
    output = {
        "honest_hybrid_posterior": honest,
        "honest_refinement_ready_mask": torch.tensor(ready, dtype=torch.bool),
    }
    if teacher:
        output["training_teacher_forced_hybrid_posterior"] = {
            "hybrid_cell_log_probability": full_log,
            "finite_rendered_mask": torch.ones_like(full_log, dtype=torch.bool),
        }
    return output


def _install_cascade(model, output):
    def forward_proposed(self, *args, **kwargs):
        return output

    model.pose_model.forward_proposed = types.MethodType(
        forward_proposed, model.pose_model
    )


def _forward(model, fixture, **kwargs):
    return model(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["bound"],
        (8, 8),
        (4, 4),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        proposal_top_m=2,
        top_k=2,
        refinement_steps=2,
        **kwargs,
    )


def test_mixed_ready_batch_refines_only_compact_honest_rows_and_preserves_mass():
    runtime = _runtime()
    model = _model(runtime).eval()
    fixture = _fixture(runtime)
    cascade = _cascade(model, [True, False])
    _install_cascade(model, cascade)
    output = _forward(model, fixture)

    assert output["refinement_source_batch_index"].tolist() == [0]
    assert output["refinement_ready_mask"].tolist() == [True, False]
    assert output["refined_output"]["pose"]["final_cell_state"].shape[:2] == (1, 2)
    assert output["refinement_selection_scope_by_sample"] == (
        "honest_closed_topk",
        "abstained_unclosed_honest_topk",
    )
    torch.testing.assert_close(
        output["refined_topk_full_catalogue_log_probability"].exp().sum(1),
        output["refinement_retained_probability"],
    )
    torch.testing.assert_close(
        output["refinement_retained_probability"]
        + output["refinement_omitted_probability"],
        torch.ones(1),
    )
    assert output["cascade"] is cascade
    assert not output["probabilities_calibrated"]
    assert output["probability_status"] == "raw_uncalibrated"


def test_all_abstained_returns_no_refinement_and_never_calls_refine():
    runtime = _runtime()
    model = _model(runtime).eval()
    fixture = _fixture(runtime)
    _install_cascade(model, _cascade(model, [False, False]))

    def forbidden(*args, **kwargs):
        raise AssertionError("refine must not run for an all-abstained batch")

    model.pose_model.refine = forbidden
    output = _forward(model, fixture)
    assert output["refined_output"] is None
    assert output["refinement_source_batch_index"].numel() == 0
    assert output["refinement_selected_catalogue_index"] is None
    assert output["refined_topk_full_catalogue_log_probability"] is None
    assert not output["refinement_performed"]


def test_training_replaces_last_mode_only_with_exactly_rendered_truth():
    runtime = _runtime()
    model = _model(runtime).train()
    fixture = _fixture(runtime, batch=1)
    cascade = _cascade(model, [True], topk=((0, 1),), teacher=True)
    _install_cascade(model, cascade)
    output = _forward(
        model,
        fixture,
        training_truth_catalogue_index=torch.tensor([3]),
    )

    assert output["refinement_selected_catalogue_index"].tolist() == [[0, 3]]
    assert output["refinement_selected_cell_id"].tolist() == [[0, 3]]
    assert output["refinement_teacher_forced_mask"].tolist() == [True]
    assert output["refinement_truth_topk_index"].tolist() == [1]
    assert output["refinement_initial_honest_mode_mask"].tolist() == [[True, True]]
    assert output["refinement_final_honest_mode_mask"].tolist() == [[True, False]]
    assert output["refinement_final_teacher_forced_mode_mask"].tolist() == [
        [False, True]
    ]
    assert not output["refinement_initial_topk_log_probability"].requires_grad
    torch.testing.assert_close(
        torch.logsumexp(output["refinement_initial_topk_log_probability"], dim=1),
        torch.zeros(1),
    )
    assert output["refinement_selection_scope_by_sample"] == (
        "teacher_forced_training_only_exact_truth_replaced_last",
    )
    assert cascade["honest_hybrid_posterior"][
        "hybrid_topk_catalogue_index"
    ].tolist() == [[0, 1]]


def test_canonical_ids_and_per_sample_axial_schedules_reach_each_row_score():
    runtime = _runtime()
    model = _model(runtime).eval()
    fixture = _fixture(runtime)
    _install_cascade(model, _cascade(model, [True, True]))
    offsets = torch.tensor([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0]])
    weights = torch.tensor([[0.25, 0.5, 0.25], [0.2, 0.6, 0.2]])
    observed = []
    original = model.pose_model.score_catalogue_chunk

    def score(self, *args, **kwargs):
        observed.append((args[2].clone(), args[-2].clone(), args[-1].clone()))
        return original(*args, **kwargs)

    model.pose_model.score_catalogue_chunk = types.MethodType(score, model.pose_model)
    output = model(
        fixture["image"],
        fixture["outline"],
        fixture["available"],
        fixture["volume"],
        fixture["bound"],
        (8, 8),
        (4, 4),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        offsets,
        weights,
        proposal_top_m=2,
        top_k=2,
        refinement_steps=2,
    )
    assert [item[0].tolist() for item in observed] == [[0, 1], [2, 3]]
    assert [item[1].tolist() for item in observed] == [
        [[-0.5, 0.0, 0.5]],
        [[-1.0, 0.0, 1.0]],
    ]
    torch.testing.assert_close(observed[0][2], weights[0:1])
    torch.testing.assert_close(observed[1][2], weights[1:2])
    assert output["refined_output"]["pose"]["final_cell_state"].shape[0] == 2


def test_fixed_deformation_gate_and_joint_gradients_reach_fresh_heads():
    runtime = _runtime()
    model = _model(runtime, pose_only_steps=1).train()
    fixture = _fixture(runtime, batch=1)
    fixture["image"].requires_grad_()
    _install_cascade(model, _cascade(model, [True], topk=((0, 1),)))
    output = _forward(model, fixture)
    refined = output["refined_output"]

    assert torch.equal(
        refined["deformation_active_sequence"],
        torch.tensor([False, True, True]),
    )
    loss = (
        refined["pose"]["final_cell_state"].square().mean()
        + refined["final_stationary_velocity_yx_px"].square().mean()
        + refined["final_support_logits"].square().mean()
    )
    loss.backward()
    assert fixture["image"].grad is not None
    assert torch.count_nonzero(fixture["image"].grad) > 0
    for parameter in (
        model.pose_model.recurrent_update.weight,
        model.pose_model.candidate_update.weight,
        model.deformation_decoder.velocity_head.weight,
        model.deformation_decoder.support_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    ranker_gradient = model.pose_model.candidate_log_likelihood.weight.grad
    assert ranker_gradient is None or torch.count_nonzero(ranker_gradient) == 0


def test_global_axial_schedule_is_not_indexed_as_a_batch():
    runtime = _runtime()
    model = _model(runtime).eval()
    fixture = _fixture(runtime)
    _install_cascade(model, _cascade(model, [True, False]))
    output = _forward(model, fixture)
    assert output["refined_output"] is not None
    assert math.isclose(
        float(output["refinement_retained_probability"][0]), 0.54, abs_tol=1e-6
    )


@pytest.mark.parametrize(
    "truth",
    [torch.tensor([3.0]), torch.tensor([True]), torch.tensor([3.0 + 0.0j])],
)
def test_joint_model_rejects_noninteger_training_truth_before_cascade(truth):
    runtime = _runtime()
    model = _model(runtime).train()
    fixture = _fixture(runtime, batch=1)
    with pytest.raises(ValueError, match="must be integers"):
        _forward(
            model,
            fixture,
            training_truth_catalogue_index=truth,
        )
