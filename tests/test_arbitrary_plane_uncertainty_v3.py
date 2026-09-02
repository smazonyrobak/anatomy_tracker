import copy
import math

import numpy as np
import pytest
import torch

import training.arbitrary_plane_uncertainty_v3 as uncertainty_v3
from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
from training.arbitrary_plane_deformation_primitives import identity_pixel_map_yx
from training.arbitrary_plane_uncertainty_v3 import (
    _gauss_hermite_grid,
    categorical_calibration_metrics_v3,
    continuous_calibration_metrics_v3,
    fit_temperature_on_heldout_animals_v3,
    posterior_coverage_metrics_v3,
    posterior_summary_v3,
    propagate_electrode_trajectory_v3,
    verify_temperature_calibration_receipt_v3,
)


def _catalogue(count=3):
    return make_arbitrary_plane_catalogue_v3(
        np.ones((6, 6, 6), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=count,
        offset_count=1,
        roll_count=1,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(4.0, 4.0),
    )


def _joint(catalogue):
    probability = torch.tensor([[0.45, 0.35, 0.20]], dtype=torch.float64)
    top = torch.tensor([[0, 1]])
    state = catalogue["tensors"]["cell_states"][:, :2].clone()
    state[:, 0, 0] -= 1.5
    state[:, 1, 0] += 1.5
    covariance = torch.diag(torch.tensor([0.01, 0.02, 0.09], dtype=torch.float64))
    identity = identity_pixel_map_yx(2, (8, 8), dtype=torch.float64)[None]
    return {
        "pose": {
            "retrieval_cell_id": torch.arange(3),
            "retrieval_cell_log_probability": probability.log(),
            "retrieval_cell_probability": probability,
            "retrieval_topk_catalogue_index": top,
            "retrieval_topk_cell_id": top,
            "retrieval_omitted_probability": torch.tensor([0.20], dtype=torch.float64),
            "catalogue_complete": True,
            "retrieval_tail_scope": "complete_catalogue_at_retrieval",
            "final_cell_state": state,
            "final_cell_canonical_plane_covariance": covariance[None, None].expand(1, 2, -1, -1),
            "conditional_within_topk_cell_log_probability": torch.tensor([[0.6, 0.4]], dtype=torch.float64).log(),
        },
        "final_forward_map_yx_px": identity,
        "final_forward_jacobian_determinant": torch.ones(
            1, 2, 8, 8, dtype=torch.float64
        ),
        "final_forward_then_inverse_valid_mask": torch.ones(
            1, 2, 8, 8, dtype=torch.bool
        ),
        "final_inverse_then_forward_valid_mask": torch.ones(
            1, 2, 8, 8, dtype=torch.bool
        ),
        "final_forward_then_inverse_error_yx": torch.zeros(
            1, 2, 2, 8, 8, dtype=torch.float64
        ),
        "final_inverse_then_forward_error_yx": torch.zeros(
            1, 2, 2, 8, 8, dtype=torch.float64
        ),
    }


def test_tensor_product_gauss_hermite_integrates_standard_normal_moments():
    node, weight = _gauss_hermite_grid(5, device=torch.device("cpu"), dtype=torch.float64)
    assert node.shape == (125, 3)
    assert weight.sum().item() == pytest.approx(1.0)
    assert torch.allclose((weight[:, None] * node).sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-14)
    second_moment = torch.einsum("q,qi,qj->ij", weight, node, node)
    assert torch.allclose(second_moment, torch.eye(3, dtype=torch.float64), atol=1e-14)


def test_multimodal_summary_preserves_exact_tail_and_local_covariance():
    catalogue = _catalogue()
    summary = posterior_summary_v3(
        _joint(catalogue), catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    assert summary["probabilities_calibrated"] is False
    assert summary["raw_exact_omitted_probability"].item() == pytest.approx(0.2)
    assert summary["omitted_probability"].item() == pytest.approx(0.2)
    assert summary["retained_probability"].item() == pytest.approx(0.8)
    assert summary["posterior_scope"].startswith("hierarchical/truncated")
    assert summary["posterior_exact"] is False
    assert summary["retrieval_tail_mass_exact"] is True
    assert summary["gauss_hermite_order_per_dimension"] == 5
    assert summary["gauss_hermite_node_count_per_mode"] == 125
    assert torch.unique(summary["refined_mode_gauss_hermite_weight"][0, 0]).numel() > 1
    assert summary["posterior_support_state"].shape == (1, 253, 12)
    assert summary["posterior_support_weight"].sum().item() == pytest.approx(1.0)
    assert (summary["posterior_support_weight"][0, :2] == 0).all()
    assert summary["posterior_support_weight"][0, 2].item() == pytest.approx(0.2)
    assert summary["quadrature_minimum_mass_support_mask"].shape == (1, 4, 253)
    assert summary["signed_plane_offset_central_credible_interval_um"].shape == (1, 4, 2)
    assert summary["center_ap_central_credible_interval_um"].shape == (1, 4, 2)
    assert summary["plane_normal_projective_credible_radius_rad"].shape == (1, 4)
    assert "largest mixture component mass" in summary["point_estimate"]["summary_semantics"]
    assert torch.equal(
        summary["point_estimate"]["map_component_mean_state"],
        summary["refined_mode_state"][:, :1].squeeze(1),
    )
    origin = torch.tensor(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        dtype=torch.float64,
    )
    expected_offset = (
        (summary["posterior_support_center_ap_dv_ml_um"] - origin)
        * summary["posterior_support_antipodally_aligned_plane_normal_ap_dv_ml"]
    ).sum(dim=-1)
    assert torch.allclose(summary["posterior_support_signed_plane_offset_um"], expected_offset)
    map_center = summary["point_estimate"]["map_center_ap_dv_ml_um"]
    map_normal = summary["plane_normal_map_axis_ap_dv_ml"]
    assert summary["point_estimate"]["map_signed_plane_offset_um"].item() == pytest.approx(
        ((map_center - origin) * map_normal).sum().item()
    )


def test_reported_tail_is_verified_against_complete_catalogue():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    joint["pose"]["retrieval_omitted_probability"] = torch.tensor([0.19], dtype=torch.float64)
    with pytest.raises(ValueError, match="exact complete-catalogue tail"):
        posterior_summary_v3(
            joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        )


def test_posterior_rejects_shifted_log_mass_missing_scope_and_non_psd_covariance():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    joint["pose"]["retrieval_cell_log_probability"] += 0.3
    joint["pose"]["retrieval_cell_probability"] *= math.exp(0.3)
    joint["pose"]["retrieval_omitted_probability"] *= math.exp(0.3)
    with pytest.raises(ValueError, match="normalized complete-catalogue"):
        posterior_summary_v3(
            joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        )
    joint = _joint(catalogue)
    del joint["pose"]["retrieval_tail_scope"]
    with pytest.raises(ValueError, match="normalized complete-catalogue"):
        posterior_summary_v3(
            joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        )
    joint = _joint(catalogue)
    joint["pose"]["final_cell_canonical_plane_covariance"] = joint["pose"][
        "final_cell_canonical_plane_covariance"
    ].clone()
    joint["pose"]["final_cell_canonical_plane_covariance"][0, 0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="non-positive-semidefinite"):
        posterior_summary_v3(
            joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
        )


def test_temperature_is_fit_only_on_disjoint_heldout_animals_and_is_receipted():
    logits = torch.tensor(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]], dtype=torch.float64
    )
    target = torch.tensor([0, 1, 1, 0])
    receipt = fit_temperature_on_heldout_animals_v3(
        logits,
        target,
        ["cal-a", "cal-a", "cal-b", "cal-b"],
        ["cal-a", "cal-b"],
        ["test-z"],
        "catalogue-id",
        training_animal_ids=["train-x"],
        checkpoint_binding_id="a" * 64,
        model_state_sha256="b" * 64,
        heldout_truth_in_topk_mask=torch.tensor([True, False, True, True]),
        heldout_topk_catalogue_cell_index=torch.tensor(
            [[0, 1], [1, 0], [1, 0]]
        ),
        heldout_refinement_logits=logits[[0, 2, 3]],
        heldout_target_refined_mode_index=torch.tensor([0, 0, 1]),
        heldout_continuous_tangent_residual=torch.tensor(
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.6], [0.3, 0.2, 0.1]],
            dtype=torch.float64,
        )[:, None].expand(-1, 2, -1),
        heldout_continuous_tangent_covariance=torch.eye(3, dtype=torch.float64)[None, None].expand(3, 2, -1, -1),
    )
    assert verify_temperature_calibration_receipt_v3(receipt, "catalogue-id")
    assert receipt["fully_calibrated"] is True
    assert receipt["conditional_sample_count"] == 3
    assert receipt["conditional_omitted_count"] == 1
    assert receipt["truth_in_topk_by_animal"]["cal-a"]["truth_omitted_count"] == 1
    assert receipt["retrieval_metrics_after"]["nll"] <= receipt["retrieval_metrics_before"]["nll"]
    assert set(receipt["retrieval_metrics_after"]["minimum_posterior_set_coverage"]) == {
        "50",
        "80",
        "90",
        "95",
    }
    assert receipt["retrieval_animal_metrics_after"]["animal_count"] == 2
    assert "animal_cluster_bootstrap_95_ci" in receipt["continuous_animal_metrics_after"][
        "animal_macro_gaussian_nll"
    ]
    tampered_seed = copy.deepcopy(receipt)
    tampered_seed["retrieval_animal_metrics_before"]["bootstrap_seed"] = 1730
    tampered_seed["receipt_sha256"] = uncertainty_v3._sha(
        {
            key: value
            for key, value in tampered_seed.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="calibration receipt"):
        verify_temperature_calibration_receipt_v3(tampered_seed, "catalogue-id")
    with pytest.raises(ValueError, match="reference catalogue cell"):
        fit_temperature_on_heldout_animals_v3(
            logits,
            target,
            ["cal-a", "cal-a", "cal-b", "cal-b"],
            ["cal-a", "cal-b"],
            ["test-z"],
            "catalogue-id",
            training_animal_ids=["train-x"],
            checkpoint_binding_id="a" * 64,
            model_state_sha256="b" * 64,
            heldout_truth_in_topk_mask=torch.tensor([True, False, True, True]),
            heldout_topk_catalogue_cell_index=torch.tensor(
                [[1, 0], [1, 0], [1, 0]]
            ),
            heldout_refinement_logits=logits[[0, 2, 3]],
            heldout_target_refined_mode_index=torch.tensor([0, 0, 1]),
        )
    with pytest.raises(ValueError, match="held-out calibration animal"):
        fit_temperature_on_heldout_animals_v3(
            logits[:1],
            target[:1],
            ["test-z"],
            ["cal-a"],
            ["test-z"],
            "catalogue-id",
            training_animal_ids=["train-x"],
            checkpoint_binding_id="a" * 64,
            model_state_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="strictly disjoint"):
        fit_temperature_on_heldout_animals_v3(
            logits[:2],
            target[:2],
            ["cal-a", "cal-a"],
            ["cal-a"],
            ["test-z"],
            "catalogue-id",
            training_animal_ids=["cal-a"],
            checkpoint_binding_id="a" * 64,
            model_state_sha256="b" * 64,
        )


def test_categorical_and_continuous_coverage_metrics_are_explicit():
    metrics = categorical_calibration_metrics_v3(
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1])
    )
    assert metrics["nll"] < 0.1
    assert metrics["minimum_posterior_set_coverage"]["90"] == 1.0
    continuous = continuous_calibration_metrics_v3(
        torch.zeros(2, 3), torch.eye(3)[None].expand(2, -1, -1)
    )
    assert continuous["mean_mahalanobis_squared"] == 0.0
    catalogue = _catalogue()
    summary = posterior_summary_v3(
        _joint(catalogue), catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    coverage = posterior_coverage_metrics_v3(
        summary,
        summary["point_estimate"]["posterior_mean_signed_plane_offset_um"],
        summary["plane_normal_map_axis_ap_dv_ml"],
    )
    assert coverage["sample_count"] == 1
    assert set(coverage["signed_plane_offset_coverage"]) == {"50", "80", "90", "95"}
    antipodal = posterior_coverage_metrics_v3(
        summary,
        -summary["point_estimate"]["posterior_mean_signed_plane_offset_um"],
        -summary["plane_normal_map_axis_ap_dv_ml"],
    )
    assert antipodal == coverage


def test_pose_and_pullback_samples_propagate_to_trajectory_and_region_confidence():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    summary = posterior_summary_v3(
        joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    annotation = torch.zeros(6, 6, 6, dtype=torch.long)
    annotation[:3] = 7
    annotation[3:] = 11
    result = propagate_electrode_trajectory_v3(
        summary,
        joint,
        torch.tensor([[2.0, 2.0], [5.0, 5.0]], dtype=torch.float64),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        annotation_volume_ap_dv_ml=annotation,
    )
    assert result["trajectory_sample_points_ap_dv_ml_um"].shape == (1, 253, 2, 3)
    assert result["validity_conditioned_pointwise_center_ap_dv_ml_um"].shape == (1, 2, 3)
    assert result["validity_conditioned_pointwise_credible_radius_um"].shape == (1, 2, 4)
    assert result["validity_conditioned_simultaneous_uniform_credible_radius_um"].shape == (1, 4)
    assert result["posterior_deformation_failure_probability"].max().item() == 0.0
    assert (
        result["region_assignment_probability"].sum(dim=-1)
        + result["invalid_region_probability"]
    ).allclose(torch.ones(1, 2, dtype=torch.float64))
    assert (result["region_assignment_confidence"] <= 1.0).all()
    assert result["trajectory_approximation_exact"] is False
    assert "one-SVF-per-refined-mode" in result["trajectory_uncertainty_scope"]


def test_pullback_selection_is_lazy_and_invalid_raster_region_mass_is_explicit():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    joint["final_pullback_map_yx_px"] = joint.pop("final_forward_map_yx_px")
    summary = posterior_summary_v3(
        joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    result = propagate_electrode_trajectory_v3(
        summary,
        joint,
        torch.tensor([[-1.0, 2.0], [2.0, 2.0]], dtype=torch.float64),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        annotation_volume_ap_dv_ml=torch.zeros(6, 6, 6, dtype=torch.long),
        atlas_shape_ap_dv_ml=(6, 6, 6),
    )
    assert result["input_raster_validity_mask"].tolist() == [[False, True]]
    assert torch.allclose(
        result["invalid_region_probability"], torch.ones(1, 2, dtype=torch.float64)
    )
    assert result["region_assignment_probability"].numel() == 0


def test_trajectory_abstains_on_deformation_topology_failure_and_rejects_bad_mode_index():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    summary = posterior_summary_v3(
        joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    joint["final_forward_jacobian_determinant"] = joint[
        "final_forward_jacobian_determinant"
    ].clone()
    joint["final_forward_jacobian_determinant"][:, 0] = 0.0
    result = propagate_electrode_trajectory_v3(
        summary,
        joint,
        torch.tensor([[2.0, 2.0]], dtype=torch.float64),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        atlas_shape_ap_dv_ml=(6, 6, 6),
    )
    assert result["posterior_deformation_failure_probability"].item() > 0.0
    assert result["validity_conditioned_point_probability"].item() < 1.0
    bad = dict(summary)
    bad["posterior_support_topk_index"] = summary[
        "posterior_support_topk_index"
    ].clone()
    bad["posterior_support_topk_index"][0, 0] = 99
    with pytest.raises(ValueError, match="valid mode indices"):
        propagate_electrode_trajectory_v3(
            bad,
            joint,
            torch.tensor([[2.0, 2.0]], dtype=torch.float64),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            atlas_shape_ap_dv_ml=(6, 6, 6),
        )


def test_inverse_cycle_diagnostic_is_sampled_at_mode_specific_canonical_point():
    catalogue = _catalogue()
    joint = _joint(catalogue)
    joint["final_forward_map_yx_px"] = joint["final_forward_map_yx_px"].clone()
    joint["final_forward_map_yx_px"][:, :, 1] += 3.0
    joint["final_inverse_then_forward_valid_mask"] = joint[
        "final_inverse_then_forward_valid_mask"
    ].clone()
    joint["final_inverse_then_forward_valid_mask"][:, :, 2, 5] = False
    summary = posterior_summary_v3(
        joint, catalogue, catalogue["support_geometry"]["support_origin_ap_dv_ml_um"]
    )
    result = propagate_electrode_trajectory_v3(
        summary,
        joint,
        torch.tensor([[2.0, 2.0]], dtype=torch.float64),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        atlas_shape_ap_dv_ml=(6, 6, 6),
    )
    assert result["posterior_deformation_failure_probability"].item() == pytest.approx(
        0.8
    )
