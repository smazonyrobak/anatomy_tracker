import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training import evaluate_registered_candidate_offsets as evaluator
from training.dense_registration_model import identity_pixel_map
from training.joint_registered_data import mask_normalized_moving


def test_repository_control_config_is_frozen_to_reviewer_only_run():
    assert evaluator.CONTROL_RUN_NAME == "joint-review-mixed-2000-r4322"
    assert evaluator.sha256_file(evaluator.CONTROL_SOURCE_CONFIG) == (
        evaluator.CONTROL_SOURCE_CONFIG_SHA256
    )
    assert evaluator.CONTROL_CRITICAL_CONFIG == {
        "registered_validation_count": 96,
        "registered_validation_seed": 1094740,
        "registered_validation_batch_size": 2,
        "validation_negatives_per_sample": 6,
        "refinement_steps": 2,
    }


def test_offset_lattice_is_signed_exhaustive_and_marks_domain_edges():
    centered = evaluator._offset_records(np.asarray((-2000.0, 0.0, 0.0)))
    assert len(centered) == 36
    assert all(record["in_domain"] for record in centered)
    assert {
        (record["axis"], record["sign"], record["magnitude"])
        for record in centered
    } == {
        (axis, sign, magnitude)
        for axis in evaluator.AXES
        for magnitude in evaluator.OFFSET_LEVELS[axis]
        for sign in (-1, 1)
    }

    boundary = evaluator._offset_records(np.asarray((500.0, 35.0, -35.0)))
    assert not next(
        record
        for record in boundary
        if record["axis"] == "ap" and record["sign"] == 1
    )["in_domain"]
    assert not next(
        record
        for record in boundary
        if record["axis"] == "lr" and record["sign"] == 1
    )["in_domain"]
    assert not next(
        record
        for record in boundary
        if record["axis"] == "dv" and record["sign"] == -1
    )["in_domain"]


def test_wilson_pair_aggregation_and_monotonicity_are_deterministic():
    rows = []
    for sample in range(4):
        for magnitude, margin in ((25.0, -0.1), (100.0, 0.2), (250.0, 0.5)):
            rows.append(
                    {
                        "section_image_id": sample,
                        "specimen_id": sample,
                    "axis": "ap",
                    "sign": 1,
                    "magnitude": magnitude,
                    "candidate_in_domain": True,
                    "reviewer_margin": margin + 0.01 * sample,
                    "reviewer_truth_win": margin + 0.01 * sample > 0.0,
                        "reviewer_pairwise_ce": evaluator._pairwise_ce(margin + 0.01 * sample),
                        "physical_corresponding_plane_distance_um": magnitude,
                }
            )
    first = evaluator._aggregate_pairs(rows, "reviewer")
    second = evaluator._aggregate_pairs(rows, "reviewer")
    assert first == second
    assert first["strata"]["nearest_neighbor"]["descriptive_pair_wilson_95_ci"]["total"] == 4
    assert first["strata"]["resolvable"]["truth_win_specimen_cluster"]["mean"] == 1.0
    ap_positive = next(
        row for row in first["monotonicity"] if row["axis"] == "ap" and row["sign"] == 1
    )
    assert ap_positive["minimum_points_for_gate"] == 3
    assert ap_positive["eligible_specimen_count"] == 4
    assert ap_positive["valid_magnitude_count_min_median_max"] == [3, 3.0, 3]
    assert ap_positive["within_specimen_spearman"]["mean"] == 1.0
    assert ap_positive["monotonic_specimen_fraction_wilson_95_ci"]["rate"] == 1.0


def test_pooled_bootstrap_clusters_by_specimen_instead_of_rows():
    summary = evaluator._clustered_summary(
        [0.0, 0.0, 0.0, 1.0], [11, 11, 11, 22], seed=7
    )
    assert summary["pair_count"] == 4
    assert summary["specimen_count"] == 2
    assert summary["mean"] == 0.5


def test_monotonicity_keeps_each_specimens_valid_lattice_and_requires_three_points():
    rows = []
    for specimen, magnitudes in ((1, (25, 50, 100, 250, 500, 1000)), (2, (25, 50))):
        rows.extend(
            {
                "specimen_id": specimen,
                "axis": "ap",
                "sign": -1,
                "magnitude": float(magnitude),
                "reviewer_margin": float(magnitude),
            }
            for magnitude in magnitudes
        )
    result = evaluator._within_specimen_monotonicity(
        rows, "ap", -1, "reviewer_margin", seed=9
    )
    assert result["specimen_count"] == 2
    assert result["eligible_specimen_count"] == 1
    assert result["full_lattice_specimen_count"] == 1
    assert result["valid_magnitude_count_min_median_max"] == [2, 4.0, 6]
    assert result["within_specimen_spearman"]["mean"] == 1.0
    assert result["monotonic_specimen_fraction_wilson_95_ci"]["total"] == 1


class _Registrar(nn.Module):
    def forward_with_details(self, fixed, moving):
        batch, _, height, width = fixed.shape
        return {
            "fixed_to_moving_map": identity_pixel_map(
                batch, height, width, device=fixed.device, dtype=fixed.dtype
            ),
            "similarity_parameters": torch.zeros(batch, 4, device=fixed.device),
            "local_velocity": torch.zeros(batch, 2, height, width, device=fixed.device),
        }


class _Review(nn.Module):
    def forward(self, fixed, warped, pose, features, similarity, velocity):
        del features, similarity, velocity
        logit = -(fixed[:, :1] - warped[:, :1]).abs().mean(dim=(1, 2, 3))
        return torch.zeros_like(pose), logit


class _ScoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.registrar = _Registrar()
        self.review_head = _Review()


class _ScoringData:
    device = torch.device("cpu")

    def render_pose(self, poses):
        y, x = torch.meshgrid(torch.arange(24.0), torch.arange(24.0), indexing="ij")
        images, masks = [], []
        for pose in poses:
            radius = 7.0 + 0.02 * pose[0]
            mask = ((x - 12.0) ** 2 / radius**2 + (y - 12.0) ** 2 / 7.0**2) <= 1.0
            image = (0.2 + x / 30.0 + y / 60.0) * mask
            images.append(image)
            masks.append(mask)
        image = torch.stack(images)[:, None]
        mask = torch.stack(masks)[:, None]
        return image, mask, torch.zeros_like(mask, dtype=torch.long)

    def moving_for_fixed(self, batch, target_mask):
        moving, mask, homography, _ = mask_normalized_moving(
            batch["moving"],
            batch["moving_model_mask"],
            target_mask,
            apply_cosine_feather=True,
        )
        return moving, mask, {"source_to_aligned_h": homography}


def test_candidate_scoring_uses_one_candidate_specific_map_per_rendered_plane():
    data = _ScoringData()
    poses = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    moving, moving_mask, _ = data.render_pose(poses[:1])
    result = evaluator.score_candidates(
        _ScoringModel(),
        data,
        {"moving": moving, "moving_model_mask": moving_mask},
        poses,
        torch.zeros(1, 4),
        chunk_size=1,
    )
    assert result["reviewer_logit"].shape == (2,)
    assert result["registration_score"].shape == (2,)
    assert result["registration_mind"].shape == (2,)
    assert result["registration_common_support_fraction"].shape == (2,)
    assert result["registration_aligned_mask_dice"].shape == (2,)
    assert result["source_to_aligned_h"].shape == (2, 3, 3)
    assert np.isfinite(result["registration_score"]).all()
    assert not np.array_equal(
        result["source_to_aligned_h"][0], result["source_to_aligned_h"][1]
    )


def test_development_loader_selects_ema_without_a_release_claim(tmp_path, monkeypatch):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.pose_initializer = nn.Linear(1, 1, bias=False)
            self.registrar = nn.Linear(1, 1, bias=False)
            self.review_head = nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(evaluator, "JointPoseRegistrationModel", Tiny)
    source_config = tmp_path / "control.json"
    source_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluator, "CONTROL_SOURCE_CONFIG", source_config)
    monkeypatch.setattr(
        evaluator, "CONTROL_SOURCE_CONFIG_SHA256", evaluator.sha256_file(source_config)
    )
    config = {
        "run_name": evaluator.CONTROL_RUN_NAME,
        "stages": [{"name": "review", "until_views": 5000}],
        **evaluator.CONTROL_CRITICAL_CONFIG,
    }
    state = {
        "pose_initializer.weight": torch.tensor([[1.0]]),
        "registrar.weight": torch.tensor([[2.0]]),
        "review_head.weight": torch.tensor([[3.0]]),
    }
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "format_version": evaluator.JOINT_CHECKPOINT_FORMAT_VERSION,
            "completed_views": 2000,
            "config": config,
            "ema": {"decay": 0.995, "shadow": state},
            "release_selection": {"state": "model"},
        },
        checkpoint,
    )
    monkeypatch.setattr(evaluator, "CONTROL_CHECKPOINT_SHA256", evaluator.sha256_file(checkpoint))
    monkeypatch.setattr(evaluator, "CONTROL_CONFIG_SHA256", evaluator._json_sha256(config))
    monkeypatch.setattr(evaluator, "CONTROL_EMA_SHA256", evaluator._state_sha256(state))
    monkeypatch.setattr(
        evaluator,
        "CONTROL_SUBTREE_SHA256",
        {
            name: evaluator._subtree_sha256(state, name)
            for name in ("pose_initializer", "registrar", "review_head")
        },
    )
    model, config, receipt = evaluator.load_development_ema(checkpoint, "cpu")
    assert float(model.registrar.weight.detach()) == 2.0
    assert config["registered_validation_count"] == 96
    assert receipt["selected_state"] == "ema.shadow"
    assert receipt["release_claim"] is False
    assert "release loader not used" in receipt["state_selection"]
    assert receipt["control_identity"]["schedule"] == [
        {"name": "review", "until_views": 5000}
    ]
    assert set(receipt["control_identity"]["state_sha256"]) == {
        "full_ema",
        "pose_initializer",
        "registrar",
        "review_head",
    }

    with pytest.raises(ValueError, match="prespecified completed-view"):
        evaluator.load_development_ema(checkpoint, "cpu", expected_completed_views=1500)

    staged = torch.load(checkpoint, weights_only=False)
    staged["config"] = {**config, "stages": [{"name": "review", "until_views": 600}, {"name": "geometry", "until_views": 5000}]}
    staged_path = tmp_path / "staged.pt"
    torch.save(staged, staged_path)
    monkeypatch.setattr(evaluator, "CONTROL_CHECKPOINT_SHA256", evaluator.sha256_file(staged_path))
    monkeypatch.setattr(
        evaluator, "CONTROL_CONFIG_SHA256", evaluator._json_sha256(staged["config"])
    )
    with pytest.raises(ValueError, match="review-only control schedule"):
        evaluator.load_development_ema(staged_path, "cpu")


def test_current_top1_views_have_explicit_nearest_and_resolvable_semantics():
    scores = [1.0, 1.2, 0.9, 0.8]
    offsets = [
        np.zeros(3),
        np.asarray((25.0, 0.0, 0.0)),
        np.asarray((50.0, 0.0, 0.0)),
        np.asarray((100.0, 0.0, 0.0)),
    ]
    result = evaluator._sample_top1(scores, offsets)
    assert result == {
        "all": False,
        "nearest_neighbor_excluded": True,
        "resolvable_only": True,
    }
    interval = evaluator._wilson(3, 4)
    assert interval["rate"] == 0.75
    assert 0.0 < interval["low"] < interval["high"] < 1.0
    assert math.isclose(evaluator._pairwise_ce(0.0), math.log(2.0))

    for candidate_scores in ([1.0, 0.0, -1.0], [1.0, 1.0, 0.0], [0.0, 1.0, -1.0]):
        expected = int(torch.tensor(candidate_scores).argmax()) == 0
        observed = evaluator._sample_top1(
            list(candidate_scores),
            [np.zeros(3), np.asarray((25.0, 0.0, 0.0)), np.asarray((100.0, 0.0, 0.0))],
        )["all"]
        assert observed == expected
    contract = evaluator._offset_contract()
    assert "strict margin > 0" in contract["pairwise_truth_win"]
    assert "tie at the maximum counts as truth" in contract["top1"]


def test_frame_control_is_handedness_preserving_and_flags_evidence_independently():
    image = np.zeros((40, 60), np.float32)
    image[8:30, 7:22] = np.linspace(0.1, 1.0, 22)[:, None]
    mask = image > 0.0
    _, _, receipt = evaluator._positive_determinant_frame_source(image, mask)
    assert receipt["linear_determinant"] > 0.0
    assert receipt["asymmetric_source"]

    rows = []
    for specimen in range(4):
        for variant in evaluator.VARIANTS[1:]:
            rows.append(
                {
                    "variant": variant,
                    "specimen_id": specimen,
                    "registration_evidence_improvement_vs_identity": 1.0 if variant == "horizontal" else -1.0,
                    "reviewer_margin_improvement_vs_identity": -1.0,
                }
            )
    report = evaluator._orientation_variant_report(rows)
    assert report["frozen_registration_frame_flag"] is True
    assert report["reviewer_variant_flag"] is False


def test_ouv_rederivation_uses_componentwise_float32_tolerances():
    observed = evaluator._ouv_rederivation_receipt(
        [{"absolute_pose_residual": [0.000194514, 1.13e-7, 3.90e-7]}]
    )
    assert evaluator.FORMAT_VERSION == 2
    assert observed["metadata_ouv_rederivation_pass"] is True
    assert all(observed["metadata_ouv_rederivation_component_pass"].values())
    assert observed["metadata_ouv_rederivation_float32_tolerance"] == {
        "ap_um": 0.01,
        "lr_deg": 1e-4,
        "dv_deg": 1e-4,
    }
    assert "far below" in observed["metadata_ouv_rederivation_tolerance_rationale"]

    drifted = evaluator._ouv_rederivation_receipt(
        [{"absolute_pose_residual": [0.1, 0.01, 0.01]}]
    )
    assert drifted["metadata_ouv_rederivation_pass"] is False
    assert not any(drifted["metadata_ouv_rederivation_component_pass"].values())


def test_physical_plane_distance_and_evidence_receipts_are_explicit(monkeypatch):
    data = SimpleNamespace(
        device=torch.device("cpu"),
        joint_synthetic_data=SimpleNamespace(
            generator=SimpleNamespace(annotation=torch.empty(0))
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "torch_annotation_brain_mask",
        lambda truth, annotation, shape: torch.ones((len(truth), 2, 2), dtype=torch.bool),
    )
    distance = evaluator._physical_plane_distances_um(
        data,
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray(((0.0, 0.0, 0.0), (25.0, 0.0, 0.0))),
    )
    assert distance[0] == pytest.approx(0.0, abs=1e-9)
    assert distance[1] == pytest.approx(25.0, abs=1e-6)
    evidence = evaluator._registration_evidence_receipt()
    assert evidence["score_weights"] == evaluator.SCORE_WEIGHTS
    assert evidence["score_weights_sha256"] == evaluator._json_sha256(
        evaluator.SCORE_WEIGHTS
    )
    assert len(evidence["premise_evaluator_source_sha256"]) == 64


def test_current_execution_atlas_and_product5_contracts_must_equal_checkpoint_contracts(
    monkeypatch,
):
    expected_atlas_contract = {
        "contract_sha256": "atlas",
        "model_shape": [320, 464],
    }
    current_atlas_contract = {
        "contract_sha256": "atlas",
        "model_shape": (320, 464),
    }
    expected_registered_contract = {
        "contract_sha256": "registered",
        "fixed_positions": [[1, 2, 3]],
    }
    current_registered_contract = {
        "contract_sha256": "registered",
        "fixed_positions": [(1, 2, 3)],
    }
    execution_contract = {
        "source_sha256": {"model": "model-source"},
        "preprocessing_contract": {"version": "preprocessing-v1"},
    }
    monkeypatch.setattr(
        evaluator, "_current_execution_contract", lambda: execution_contract
    )
    receipt = {
        "checkpoint_generator_contract": {
            **execution_contract,
            "synthetic": expected_atlas_contract,
            "registered_validation": expected_registered_contract,
        }
    }
    data = SimpleNamespace(
        contract=current_registered_contract,
        joint_synthetic_data=SimpleNamespace(
            generator=SimpleNamespace(contract=current_atlas_contract)
        ),
    )
    evaluator._bind_current_data_contract(receipt, data)
    assert "canonical JSON equality" in receipt["current_data_contract"]["comparison"]
    assert receipt["current_data_contract"]["atlas_generator_sha256"] == (
        evaluator._json_sha256(expected_atlas_contract)
    )
    assert receipt["current_data_contract"]["source_sha256"] == execution_contract[
        "source_sha256"
    ]
    data.joint_synthetic_data.generator.contract["model_shape"] = (320, 465)
    with pytest.raises(ValueError, match="current atlas or Product-5"):
        evaluator._bind_current_data_contract(receipt, data)


@pytest.mark.parametrize("contract_key", ("source_sha256", "preprocessing_contract"))
def test_current_execution_contract_mismatch_is_fail_closed(monkeypatch, contract_key):
    execution_contract = {
        "source_sha256": {"model": "checkpoint-source"},
        "preprocessing_contract": {"version": "checkpoint-preprocessing"},
    }
    current = {
        "source_sha256": dict(execution_contract["source_sha256"]),
        "preprocessing_contract": dict(execution_contract["preprocessing_contract"]),
    }
    current[contract_key][next(iter(current[contract_key]))] = "changed"
    monkeypatch.setattr(evaluator, "_current_execution_contract", lambda: current)
    receipt = {
        "checkpoint_generator_contract": {
            **execution_contract,
            "synthetic": {},
            "registered_validation": {},
        }
    }
    data = SimpleNamespace(
        contract={},
        joint_synthetic_data=SimpleNamespace(generator=SimpleNamespace(contract={})),
    )
    with pytest.raises(ValueError, match="execution source or preprocessing"):
        evaluator._bind_current_data_contract(receipt, data)
