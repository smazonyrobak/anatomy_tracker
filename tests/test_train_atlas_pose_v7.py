import csv
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn
import training.train_atlas_pose_v7 as trainer

from training.train_atlas_pose_v7 import (
    COMPARISON_SEEDS,
    FINAL_GATE_THRESHOLDS,
    atlas_data_hashes,
    bootstrap_seed_group_comparison,
    checkpoint_selection_improved,
    checkpoint_validation_key,
    cosine_learning_rate,
    evaluated_rows_sha256,
    ema_state,
    ensure_fixed_manifest,
    ensure_paired_manifest,
    export_onnx,
    final_acceptance_summary,
    file_sha256,
    held_out_reports,
    manifest_sha256,
    module_state_sha256,
    paired_invariance,
    promote_export,
    registered_data_hashes,
    registered_sampling_weights,
    registered_report,
    registered_rows_for_products,
    registered_domain_reports,
    representative_onnx_batch,
    rotation_180_counterfactual_diagnostics,
    seed_animal_component_errors,
    select_model_family,
    renderer_variant,
    registered_style,
    stratified_pose_metrics,
    synthetic_acceptance_summary,
    specimen_median_tilt_diagnostics,
    training_objective,
    update_ema,
    validation_selection_summary,
    validation_selection_key,
)
from training.synthetic_atlas import APPEARANCE_MANIFEST_KEYS
from source.atlas_pose_runtime import _canonical_json_sha256


class ToyPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 3, padding=1)
        self.features = nn.Linear(8, 16)
        self.pose_head = nn.Linear(16, 3)
        self.orientation_head = nn.Linear(16, 1)
        self.anatomy_head = nn.Conv2d(8, 9, 1)
        self.register_buffer("center", torch.tensor((-2000.0, 0.0, 0.0)))
        self.register_buffer("scale", torch.tensor((2500.0, 20.0, 20.0)))

    def training_outputs(self, image, include_anatomy=True):
        feature_map = torch.relu(self.encoder(image))
        features = self.features(feature_map.mean((-2, -1)))
        normalized_pose = self.pose_head(features)
        image_frame_pose = normalized_pose * self.scale + self.center
        orientation = self.orientation_head(features).squeeze(1)
        sign = torch.where(orientation > 0.0, -torch.ones_like(orientation), torch.ones_like(orientation))
        pose = torch.cat((image_frame_pose[:, :1], image_frame_pose[:, 1:] * sign[:, None]), 1)
        output = {
            "pose": pose,
            "image_frame_pose": image_frame_pose,
            "normalized_pose": normalized_pose,
            "orientation_inverted_logit": orientation,
            "pooled_features": features,
        }
        if include_anatomy:
            output["anatomy_logits"] = self.anatomy_head(feature_map)
        return output

    def forward_with_orientation(self, image):
        output = self.training_outputs(image)
        return output["pose"], output["orientation_inverted_logit"]


def test_training_source_commitment_includes_release_contract():
    assert "training/atlas_pose_release_contract.py" in trainer.training_source_hashes()


def test_git_source_provenance_includes_untracked_files(monkeypatch):
    commands = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(command, **_kwargs):
        commands.append(command)
        return Result("a" * 40 + "\n" if command[1] == "rev-parse" else "?? training/new.py\n")

    monkeypatch.setattr(trainer.subprocess, "run", run)
    provenance = trainer.git_source_provenance()

    assert "--untracked-files=all" in commands[1]
    assert provenance["tracked_source_dirty"] is True


def test_export_refuses_dirty_training_source_before_writing(tmp_path):
    with pytest.raises(RuntimeError, match="tracked-clean"):
        export_onnx(
            ToyPoseModel(),
            tmp_path / "export",
            {"git": {"tracked_source_dirty": True}},
        )
    assert not (tmp_path / "export").exists()


def validation_rows(split="validation"):
    return [
        {
            "split": split,
            "specimen_id": specimen,
            "experiment_id": specimen,
            "section_image_id": specimen * 10 + section,
            "product": str(5 + specimen % 2 * 3),
            "target_ap": -1000.0 - section * 100.0,
            "target_lr": float(specimen),
            "target_dv": -float(section),
            "prediction_ap": -980.0 - section * 100.0,
            "prediction_lr": float(specimen) + 0.5,
            "prediction_dv": -float(section) - 0.25,
        }
        for specimen in (1, 2)
        for section in (0, 1)
    ]


def gate_eligible_validation_rows():
    rows = []
    lr_values = (-2.0, 0.0, 2.0)
    dv_values = (-8.0, -4.0, 0.0)
    section_id = 0
    for specimen in range(30):
        for target_ap in np.arange(-4250.0, 500.0, 500.0):
            rows.append(
                {
                    "split": "validation",
                    "specimen_id": specimen,
                    "experiment_id": specimen,
                    "section_image_id": section_id,
                    "product": "5" if specimen < 15 else "8",
                    "target_ap": float(target_ap),
                    "target_lr": lr_values[specimen % 3],
                    "target_dv": dv_values[specimen % 3],
                    "prediction_ap": float(target_ap + 20.0),
                    "prediction_lr": lr_values[specimen % 3] + 0.5,
                    "prediction_dv": dv_values[specimen % 3] + 1.0,
                    "in_training_ap_domain": True,
                }
            )
            section_id += 1
    return rows


def passing_synthetic_report():
    component = {"count": 64, "mae": [50.0, 0.7, 1.2]}
    return {
        "overall": dict(component),
        "artifact_severity": {
            name: dict(component) for name in ("clean", "mild", "moderate", "severe")
        },
        "tilt_bands": {
            name: dict(component) for name in ("0:5", "5:15", "15:25", "25:35")
        },
        "artifact_invariance": {
            "mean_absolute_prediction_shift": [40.0, 0.6, 1.0],
            "p95_absolute_prediction_shift": [80.0, 1.2, 2.0],
            "mean_absolute_error_change": [0.0, 0.0, 0.0],
        },
    }


def test_first_stage_defaults_and_environment_overrides(monkeypatch):
    assert trainer.DEFAULTS["data_workers"] == 8
    assert trainer.DEFAULTS["registered_fraction"] == 0.50
    assert trainer.DEFAULTS["validation_interval"] == 1_000
    captured = {}
    monkeypatch.setenv("ATLAS_POSE_V7_EXPERIMENT", "worker_override")
    monkeypatch.setenv("ATLAS_POSE_V7_SAMPLES", "12")
    monkeypatch.setenv("ATLAS_POSE_V7_DATA_WORKERS", "3")
    monkeypatch.setenv("ATLAS_POSE_V7_VALIDATION_INTERVAL", "250")
    monkeypatch.setattr(
        trainer,
        "run_experiment",
        lambda config, export=False: captured.update(config=config, export=export),
    )
    trainer.main()
    assert captured["config"]["data_workers"] == 3
    assert captured["config"]["validation_interval"] == 250


def test_main_refuses_an_implicit_multi_stage_schedule(monkeypatch):
    monkeypatch.delenv("ATLAS_POSE_V7_EXPERIMENT", raising=False)
    monkeypatch.setattr(
        trainer,
        "run_experiment",
        lambda *_args, **_kwargs: pytest.fail("an implicit training run was started"),
    )
    with pytest.raises(RuntimeError, match="ATLAS_POSE_V7_EXPERIMENT"):
        trainer.main()


def test_registered_sampling_is_product_and_specimen_balanced_without_tilt_strata():
    dataset = SimpleNamespace(
        datasets={
            1: {"product_ids": [5]},
            2: {"product_ids": [5]},
            3: {"product_ids": [8]},
        },
        records=[
            {"experiment_id": 1, "specimen_id": 101, "tilt_lr_deg": 0.0},
            {"experiment_id": 1, "specimen_id": 101, "tilt_lr_deg": 30.0},
            {"experiment_id": 2, "specimen_id": 102, "tilt_lr_deg": 0.0},
            {"experiment_id": 3, "specimen_id": 103, "tilt_lr_deg": -30.0},
            {"experiment_id": 3, "specimen_id": 103, "tilt_lr_deg": 0.0},
            {"experiment_id": 3, "specimen_id": 103, "tilt_lr_deg": 30.0},
        ],
    )
    weights = registered_sampling_weights(dataset).numpy()
    assert weights[:2].sum() == pytest.approx(0.5)
    assert weights[2] == pytest.approx(0.5)
    assert weights[3:].sum() == pytest.approx(1.0)
    assert weights[0] == weights[1]


def test_fixed_latents_and_paired_views_are_reproducible_without_image_cache(tmp_path):
    first, path = ensure_fixed_manifest(tmp_path, "train", 32, 17)
    second, repeated_path = ensure_fixed_manifest(tmp_path, "train", 32, 17)
    paired, paired_path = ensure_paired_manifest(tmp_path, first, "train", 23)
    assert path == repeated_path
    assert manifest_sha256(first) == manifest_sha256(second)
    assert paired_path.is_file()
    for key in first.keys() - set(APPEARANCE_MANIFEST_KEYS):
        assert np.array_equal(first[key], paired[key])
    minimal = renderer_variant(first, "minimal")
    assert not minimal["warp"].any()
    assert not minimal["flaw_mask"].any()
    assert not minimal["sensor_enabled"].any()
    assert not minimal["occlusion_type"].any()
    assert not list(tmp_path.rglob("*.npy"))


def test_paired_manifest_cache_is_keyed_by_the_base_latents(tmp_path):
    base, _ = ensure_fixed_manifest(tmp_path, "train", 8, 19)
    minimal = renderer_variant(base, "minimal")
    _, full_path = ensure_paired_manifest(tmp_path, base, "train", 23)
    paired_minimal, minimal_path = ensure_paired_manifest(tmp_path, minimal, "train", 23)

    assert full_path != minimal_path
    for key in minimal.keys() - set(APPEARANCE_MANIFEST_KEYS):
        assert np.array_equal(minimal[key], paired_minimal[key])


def test_fixed_synthetic_validation_cache_is_lossless_and_renders_only_once(tmp_path):
    class Renderer:
        atlas_folder = tmp_path
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def batch(self, manifest, start, count):
            self.calls += 1
            pixels = torch.from_numpy(manifest["pixels"][start : start + count])
            image = pixels[:, None, None, None].expand(-1, 3, 5, 5).clone()
            target = torch.from_numpy(
                np.column_stack(
                    (
                        manifest["ap_um"][start : start + count],
                        np.zeros((count, 2), dtype=np.float32),
                    )
                )
            )
            return image, torch.zeros_like(target), target

    manifest = {
        "ap_um": np.asarray([-100.0, -200.0, -300.0, -400.0], dtype=np.float32),
        "cohort": np.arange(4, dtype=np.uint8),
        "pixels": np.asarray([0.0012345, 0.1723456, 0.5012345, 0.9987654], dtype=np.float32),
    }
    paired = {
        **manifest,
        "pixels": np.asarray([0.9981234, 0.5123456, 0.0612345, 0.004321], dtype=np.float32),
    }
    renderer = Renderer()
    trainer._SYNTHETIC_VALIDATION_CACHE.clear()
    first = trainer._synthetic_validation_images(renderer, manifest, paired, batch_size=2)
    second = trainer._synthetic_validation_images(renderer, manifest, paired, batch_size=2)
    assert renderer.calls == 4
    assert all(left.data_ptr() == right.data_ptr() for left, right in zip(first, second))
    assert torch.equal(first[0], torch.from_numpy(manifest["pixels"])[:, None, None].expand(-1, 5, 5))
    assert torch.equal(first[1], torch.from_numpy(paired["pixels"])[:, None, None].expand(-1, 5, 5))
    assert torch.equal(first[2][:, 0], torch.from_numpy(manifest["ap_um"]))


def test_paired_objective_updates_pose_orientation_consistency_and_anatomy():
    torch.manual_seed(3)
    model = ToyPoseModel()
    images = torch.rand(4, 2, 3, 16, 16)
    pose = torch.tensor([[-1200.0, 4.0, -2.0], [-1800.0, -5.0, 7.0], [-2400.0, 8.0, 1.0], [-3000.0, -3.0, -6.0]])
    orientation = torch.tensor([0.0, 1.0, 0.0, 1.0])
    anatomy = torch.randint(0, 9, (4, 16, 16))
    loss, components = training_objective(model, images, pose, orientation, anatomy, 0.15, 0.20)
    loss.backward()
    assert set(components) == {
        "total",
        "pose",
        "image_frame_pose",
        "soft_physical_tilt",
        "weighted_soft_physical_tilt",
        "representation_auxiliary",
        "weighted_representation_auxiliary",
        "orientation",
        "weighted_orientation",
        "feature_consistency",
        "prediction_consistency",
        "anatomy",
    }
    assert components["feature_consistency"] > 0.0
    assert components["prediction_consistency"] > 0.0
    assert model.pose_head.weight.grad is not None
    assert model.orientation_head.weight.grad is not None
    assert model.anatomy_head.weight.grad is not None
    assert model.encoder.weight.grad is not None


def test_no_anatomy_objective_does_not_execute_or_update_anatomy_decoder():
    model = ToyPoseModel()
    image = torch.rand(2, 3, 16, 16)
    pose = torch.zeros(2, 3)
    loss, components = training_objective(
        model,
        image,
        pose,
        torch.zeros(2),
        None,
        consistency_weight=0.0,
        anatomy_weight=0.0,
    )
    loss.backward()
    assert components["anatomy"] == 0.0
    assert model.anatomy_head.weight.grad is None


def test_metrics_cover_cohorts_bands_bias_calibration_and_registered_groups():
    target = np.asarray([
        [-4450.0, 2.0, -1.0], [-3900.0, 8.0, 4.0], [-2400.0, 18.0, -3.0], [-1200.0, 28.0, -32.0]
    ])
    prediction = target + np.asarray((25.0, 1.0, -0.5))
    report = stratified_pose_metrics(target, prediction, np.arange(4))
    assert set(report["artifact_severity"]) == {"clean", "mild", "moderate", "severe"}
    assert len(report["ap_500um_bands"]) == 4
    assert len(report["tilt_bands"]) == 4
    assert report["overall"]["bias"] == pytest.approx([25.0, 1.0, -0.5])
    assert report["overall"]["calibration_slope"] == pytest.approx([1.0, 1.0, 1.0])

    rows = validation_rows()
    registered = registered_report(rows)
    assert set(registered["per_specimen"]) == {"1", "2"}
    assert set(registered["per_product"]) == {"5", "8"}
    assert registered["worst_ap_bin"] in registered["ap_500um_bands"]
    assert validation_selection_summary(rows)["composite_score"] > 0.0
    for forbidden in ("test", "sealed_deepslice_s2p"):
        with pytest.raises(RuntimeError, match="validation"):
            validation_selection_summary(validation_rows(forbidden))

    for row in rows:
        row["in_training_ap_domain"] = True
    excluded = {
        **rows[0],
        "section_image_id": 999,
        "target_ap": -5000.0,
        "prediction_ap": -4900.0,
        "in_training_ap_domain": False,
    }
    domains = registered_domain_reports([*rows, excluded])
    assert domains["primary_in_training_ap_domain"]["overall"]["count"] == len(rows)
    assert domains["out_of_domain"]["overall"]["count"] == 1


def test_validation_selection_equal_weights_each_specimen_ap_bin_and_enforces_axis_gates():
    rows = []
    for section in range(100):
        rows.append(
            {
                **validation_rows()[0],
                "section_image_id": section,
                "target_ap": -750.0,
                "prediction_ap": -750.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            }
        )
    rows.extend(
        (
            {
                **validation_rows()[0],
                "section_image_id": 101,
                "target_ap": -1750.0,
                "prediction_ap": -1630.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            },
            {
                **validation_rows()[2],
                "section_image_id": 201,
                "target_ap": -750.0,
                "prediction_ap": -690.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            },
        )
    )
    summary = validation_selection_summary(rows)
    assert summary["component_mae"]["ap_um"] == pytest.approx(60.0)
    assert summary["all_mean_gates_passed"] is True
    final_gate = final_acceptance_summary(rows, "validation")
    assert final_gate["all_gates_passed"] is False
    assert final_gate["passed"]["worst_ap_band_mae_um"] is False
    assert final_gate["thresholds"] == FINAL_GATE_THRESHOLDS

    for row in rows:
        row["prediction_lr"] = row["target_lr"] + 0.91
    failed = validation_selection_summary(rows)
    assert failed["component_passed"]["ap_um"] is True
    assert failed["component_passed"]["lr_deg"] is False
    assert failed["all_mean_gates_passed"] is False


def test_checkpoint_key_prioritizes_full_performance_then_worst_ratio_then_composite():
    higher_composite = {"composite_score": 0.95}
    lower_composite = {"composite_score": 0.40}
    passing = {"all_performance_gates_passed": True, "worst_gate_ratio": 0.99}
    failing = {"all_performance_gates_passed": False, "worst_gate_ratio": 1.20}
    lower_worst = {**failing, "worst_gate_ratio": 1.10}

    assert validation_selection_key(higher_composite, passing) < validation_selection_key(
        lower_composite, failing
    )
    assert validation_selection_key(higher_composite, lower_worst) < validation_selection_key(
        lower_composite, failing
    )
    assert validation_selection_key(lower_composite, failing) < validation_selection_key(
        higher_composite, failing
    )


def test_checkpoint_key_uses_the_worst_trusted_or_synthetic_gate():
    summary = {"composite_score": 0.4}
    registered = {"worst_gate_ratio": 0.8}
    synthetic = {"worst_gate_ratio": 1.7}
    assert checkpoint_validation_key(summary, registered, synthetic) == (1.7, 0.8, 0.4)


def test_product_8_rows_are_available_for_diagnostics_but_not_trusted_selection():
    rows = gate_eligible_validation_rows()
    rows.append({**rows[0], "section_image_id": -1, "product": "5+8"})
    trusted = registered_rows_for_products(rows, ("5",))
    assert trusted
    assert {row["product"] for row in trusted} == {"5"}
    with pytest.raises(RuntimeError, match="products"):
        registered_rows_for_products(trusted, ("8",))

    all_product_5 = [{**row, "product": "5"} for row in rows]
    assert final_acceptance_summary(
        all_product_5,
        "validation",
        ("5",),
    )["all_gates_passed"] is True
    assert final_acceptance_summary(all_product_5, "validation")["all_gates_passed"] is False


def test_release_gate_requires_preregistered_real_data_coverage_and_animal_tails():
    smoke = final_acceptance_summary(validation_rows(), "validation")
    assert smoke["coverage"]["eligible"] is False
    assert smoke["coverage"]["passed"]["animals"] is False
    assert smoke["all_gates_passed"] is False

    rows = gate_eligible_validation_rows()
    passing = final_acceptance_summary(rows, "validation")
    assert passing["coverage"]["eligible"] is True
    assert passing["coverage"]["counts"]["animals"] == 30
    assert passing["coverage"]["counts"]["animals_by_required_product"] == {"5": 15, "8": 15}
    assert min(passing["coverage"]["counts"]["animals_by_ap_band"].values()) == 30
    assert min(passing["coverage"]["counts"]["animals_by_lr_bin"].values()) == 10
    assert min(passing["coverage"]["counts"]["animals_by_dv_bin"].values()) == 10
    assert passing["values"]["ap_bootstrap_upper95_um"] == pytest.approx(20.0)
    assert passing["values"]["per_animal_p90_ap_um"] == pytest.approx(20.0)
    assert passing["values"]["worst_group_p90_dv_deg"] == pytest.approx(1.0)
    assert passing["all_gates_passed"] is True

    for row in rows:
        if row["specimen_id"] < 4:
            row["prediction_dv"] = row["target_dv"] + 4.0
    failed = final_acceptance_summary(rows, "validation")
    assert failed["passed"]["per_animal_p90_dv_deg"] is False
    assert failed["passed"]["worst_group_p90_dv_deg"] is False
    assert failed["all_gates_passed"] is False


def test_selection_does_not_trade_a_product_tail_failure_for_a_better_global_mean():
    tail_failure = gate_eligible_validation_rows()
    robust = gate_eligible_validation_rows()
    for row in tail_failure:
        error = 100.0 if row["specimen_id"] >= 28 else (15.0 if row["specimen_id"] % 2 else -15.0)
        row["prediction_ap"] = row["target_ap"] + error
        row["product"] = "8" if row["specimen_id"] >= 28 else "5"
    for row in robust:
        error = 40.0 if row["specimen_id"] % 2 else -40.0
        row["prediction_ap"] = row["target_ap"] + error
        row["product"] = "8" if row["specimen_id"] >= 28 else "5"

    tail_selection = validation_selection_summary(tail_failure)
    robust_selection = validation_selection_summary(robust)
    tail_gate = final_acceptance_summary(tail_failure, "validation")
    robust_gate = final_acceptance_summary(robust, "validation")

    assert tail_selection["composite_score"] < robust_selection["composite_score"]
    assert tail_gate["passed"]["worst_product_mae_um"] is False
    assert robust_gate["all_performance_gates_passed"] is True
    assert validation_selection_key(robust_selection, robust_gate) < validation_selection_key(
        tail_selection, tail_gate
    )


def test_synthetic_robustness_is_a_locked_eligibility_gate():
    report = passing_synthetic_report()
    passing = synthetic_acceptance_summary(report)
    assert passing["coverage"]["eligible"] is True
    assert passing["all_gates_passed"] is True

    report["artifact_severity"]["severe"]["mae"] = [91.0, 0.7, 1.2]
    failed = synthetic_acceptance_summary(report)
    assert failed["passed"]["worst_artifact_mae_ap_um"] is False
    assert failed["all_gates_passed"] is False

    report = passing_synthetic_report()
    del report["artifact_invariance"]
    missing_pair = synthetic_acceptance_summary(report)
    assert missing_pair["coverage"]["passed"]["paired_artifact_invariance"] is False
    assert missing_pair["all_gates_passed"] is False

    assert checkpoint_selection_improved(True, False, (10.0, 10.0, 10.0), (0.1, 0.1, 0.1), 0.0)
    assert not checkpoint_selection_improved(False, True, (0.1, 0.1, 0.1), (10.0, 10.0, 10.0), 0.0)


def test_evaluated_row_hash_is_order_invariant_and_prediction_bound():
    rows = gate_eligible_validation_rows()
    digest = evaluated_rows_sha256(rows)
    assert digest == evaluated_rows_sha256(list(reversed(rows)))
    rows[0]["prediction_ap"] += 1.0
    assert evaluated_rows_sha256(rows) != digest


def test_paired_invariance_reports_prediction_shift():
    target = np.zeros((6, 3))
    first = np.ones((6, 3))
    second = np.full((6, 3), 2.0)
    invariant = paired_invariance(target, first, second)
    assert invariant["mean_absolute_prediction_shift"] == [1.0, 1.0, 1.0]


def test_nonselection_tilt_pooling_and_rotation_counterfactual_diagnostics():
    rows = [
        {
            "specimen_id": specimen,
            "target_lr": float(specimen),
            "target_dv": -2.0,
            "prediction_lr": float(specimen) + lr_error,
            "prediction_dv": -1.75,
        }
        for specimen in (1, 2)
        for lr_error in (0.5, 0.5, 100.0)
    ]
    pooled = specimen_median_tilt_diagnostics(rows)
    assert pooled["role"] == "diagnostic_only_not_used_for_selection"
    assert pooled["mae_deg"] == pytest.approx({"lr": 0.5, "dv": 0.25})

    prediction = np.asarray([[10.0, 1.0, -2.0], [20.0, -1.0, 2.0]])
    rotated = prediction + np.asarray([2.0, 0.2, -0.3])
    diagnostic = rotation_180_counterfactual_diagnostics(
        prediction,
        rotated,
        np.asarray([-2.0, 3.0]),
        np.asarray([2.0, -3.0]),
    )
    assert diagnostic["mean_absolute_prediction_shift"] == pytest.approx([2.0, 0.2, 0.3])
    assert diagnostic["orientation_logit_sign_flip_fraction"] == 1.0
    assert diagnostic["mean_absolute_orientation_logit_sum"] == 0.0


def test_model_family_selection_uses_paired_seed_and_animal_uncertainty(tmp_path):
    def results(name, ap_errors, synthetic_passed=True, synthetic_worst=0.5):
        output = []
        for seed, ap_error in zip(COMPARISON_SEEDS, ap_errors):
            folder = tmp_path / f"{name}_{seed}"
            folder.mkdir()
            rows = validation_rows()
            for row in rows:
                row["product"] = "5"
                row["prediction_ap"] = row["target_ap"] + ap_error
                row["prediction_lr"] = row["target_lr"]
                row["prediction_dv"] = row["target_dv"]
            with (folder / "validation_registered.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output.append(
                {
                    "selection_split": "validation",
                    "best_checkpoint": str(folder / "best.pt"),
                    "config": {"training_seed": seed},
                    "synthetic_validation_gate": {
                        "all_gates_passed": synthetic_passed,
                        "worst_gate_ratio": synthetic_worst,
                    },
                }
            )
        return output

    candidate = results("candidate", (60.0, 120.0, 180.0))
    reference = results("reference", (120.0, 180.0, 240.0))
    seeds, animals, errors = seed_animal_component_errors(candidate)
    assert seeds.tolist() == sorted(COMPARISON_SEEDS)
    assert animals.tolist() == [1, 2]
    assert errors.shape == (3, 2, 3)
    comparison = bootstrap_seed_group_comparison(
        candidate,
        reference,
        tmp_path / "comparison.json",
        iterations=1000,
        seed=7,
    )
    assert comparison["unit"] == "paired training_seed x specimen_id"
    assert comparison["seed_count"] == 3
    assert comparison["animal_count"] == 2
    assert comparison["components"]["ap_um"]["candidate_minus_reference"] == pytest.approx(-60.0)
    assert comparison["probability_candidate_better"] == 1.0

    tied = results("tied", (60.0, 120.0, 180.0))
    decision = select_model_family(
        {"preferred": tied, "point_best": candidate},
        "test family",
        ("preferred", "point_best"),
        tmp_path / "family",
    )
    assert decision["decision"] == "prespecified_tie_priority"
    assert decision["winner"] == "preferred"

    synthetic_failure = results("synthetic_failure", (0.0, 0.0, 0.0), False, 2.0)
    robust = results("robust", (20.0, 20.0, 20.0), True, 0.8)
    decision = select_model_family(
        {"synthetic_failure": synthetic_failure, "robust": robust},
        "synthetic gate",
        ("synthetic_failure", "robust"),
        tmp_path / "synthetic_gate",
    )
    assert decision["point_estimate_best"] == "robust"
    assert decision["winner"] == "robust"


def test_registered_style_is_deterministic_local_and_geometry_preserving():
    image = np.tile(np.arange(192, dtype=np.uint8), (128, 1))
    first = registered_style(image, np.random.default_rng(18))
    second = registered_style(image, np.random.default_rng(18))
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert np.array_equal(first, second)
    design = np.column_stack((image.ravel(), np.ones(image.size)))
    residual = first.ravel() - design @ np.linalg.lstsq(design, first.ravel(), rcond=None)[0]
    assert residual.std() > 1.0


def test_ema_and_warmup_cosine_are_deterministic():
    model = nn.Linear(2, 1)
    ema = ema_state(model)
    with torch.no_grad():
        model.weight.add_(2.0)
    old = ema["weight"].clone()
    update_ema(ema, model, 0.75, 100)
    assert torch.allclose(ema["weight"], old + 0.5)
    ema = ema_state(nn.Linear(2, 1))
    source = nn.Linear(2, 1)
    source.load_state_dict({name: value.clone() for name, value in ema.items()})
    with torch.no_grad():
        source.weight.add_(2.0)
    update_ema(ema, source, 0.999, 1)
    assert torch.allclose(ema["weight"], source.weight - 2.0 * (2.0 / 11.0))
    rates = [cosine_learning_rate(step, 10, 2, 1e-3) for step in range(10)]
    assert rates[:2] == pytest.approx([5e-4, 1e-3])
    assert rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_foreach_ema_matches_the_scalar_update_formula_exactly():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    model.register_buffer("updates", torch.tensor(7, dtype=torch.int64))
    ema = ema_state(model)
    expected = {name: value.clone() for name, value in ema.items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(torch.linspace(0.1, 0.9, value.numel()).reshape_as(value))
        model.updates.add_(1)
    decay = min(0.999, (1.0 + 17) / (10.0 + 17))
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            expected[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            expected[name].copy_(value)
    update_ema(ema, model, 0.999, 17)
    assert all(torch.equal(ema[name], expected[name]) for name in ema)


def test_export_is_verified_and_promotion_is_explicit(tmp_path):
    model = ToyPoseModel().eval()
    export_folder = tmp_path / "workspace" / "run" / "export"
    registered_hashes = {
        name: str(index) * 64
        for index, name in enumerate(
            (
                "datasets.jsonl",
                "sections.jsonl",
                "provenance.json",
                "downloads.jsonl",
                "registered_image_quality.json",
            ),
            1,
        )
    }
    registered_hashes["nonsealed_image_tree_sha256"] = "9" * 64
    metadata = export_onnx(
        model,
        export_folder,
        {
            "selection_split": "validation",
            "manifest_sha256": {"toy": "a" * 64},
            "registered_data": {
                "sha256": registered_hashes,
                "excluded_from_selection": ["test", "sealed_deepslice_s2p"],
            },
            "atlas_data_sha256": {"average_template_25.nrrd": "c" * 64},
            "git": {
                "commit": "d" * 40,
                "tracked_source_dirty": False,
                "tracked_source_status": [],
            },
        },
        representative_onnx_batch()[:2],
    )
    model_path = export_folder / "atlas_pose.onnx"
    assert metadata["sha256"] == file_sha256(model_path)
    assert metadata["preprocessing_version"] == "smart-mask-scale-invariant-v2"
    assert metadata["automatic_brain_mask_version"] == "border-distance-conditional-hull-v6"
    assert metadata["quicknii_coordinate_contract"] == "quicknii-ras-to-allen-pir-v2"
    assert metadata["preprocessing_contract_sha256"] == trainer.atlas_pose_preprocessing_contract_sha256()
    assert metadata["verification_sample_count"] == 2
    assert "CPUExecutionProvider" in metadata["verification_by_provider"]
    for provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if provider in ort.get_available_providers():
            assert provider in metadata["verification_by_provider"]
    assert len(metadata["verification_input_sha256"]) == 64
    assert (export_folder / "atlas_pose.json").is_file()
    provenance = json.loads((export_folder / "provenance.json").read_text())
    assert provenance["registered_data_sha256"] == registered_hashes
    assert provenance["excluded_from_selection"] == [
        "registered_test", "sealed_deepslice_s2p"
    ]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == [
        "pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"
    ]
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    metrics = sealed / "SEALED_metrics.json"
    predictions = sealed / "SEALED_predictions.csv"
    import pandas as pd
    from training.evaluate_sealed_registered_holdout import (
        POSE_AXES,
        paired_animal_bootstrap,
        paired_animal_joint_superiority,
        sealed_release_report,
    )

    methods = {
        "deepslice_ai": (120.0, 2.5, 3.5),
        "deepslice_mens_ai": (110.0, 2.25, 3.25),
        "deepslice_mens_ai_ci": (100.0, 2.0, 3.0),
        "atlas_pose": (1.0, 0.01, 0.01),
    }
    rows = []
    for section_id in range(1400):
        experiment_id = section_id // 140 + 1
        ground_truth = (
            -4500.0 + 5000.0 * section_id / 1399.0,
            (-3.0, 0.0, 3.0)[section_id % 3],
            (-10.0, -5.0, 1.0)[section_id % 3],
        )
        for method, errors in methods.items():
            row = {
                "sealed": True,
                "split": "sealed_deepslice_s2p",
                "method": method,
                "experiment_id": experiment_id,
                "specimen_id": experiment_id,
                "section_image_id": section_id,
                "section_number": section_id % 140,
                "relative_path": f"images/{section_id}.jpg",
                "product": "5" if experiment_id <= 5 else "8",
                "ap_band": "in_domain",
                "in_training_ap_domain": True,
            }
            for axis, truth, error in zip(POSE_AXES, ground_truth, errors):
                row[f"gt_{axis}"] = truth
                row[f"pred_{axis}"] = truth + error
                row[f"error_{axis}"] = error
                row[f"absolute_error_{axis}"] = abs(error)
            rows.append(row)
    prediction_table = pd.DataFrame(rows)
    prediction_table.to_csv(predictions, index=False)
    sealed_source = {
        **{
            name: value
            for name, value in registered_hashes.items()
            if name != "nonsealed_image_tree_sha256"
        },
        "sealed_image_tree_sha256": "6" * 64,
    }
    evaluator_environment = {
        "contract_version": 1,
        "source_sha256": {"evaluator.py": "7" * 64},
        "deepslice_model_sha256": {"primary": "8" * 64, "secondary": "9" * 64},
        "dependencies": {"python": "3.11"},
    }
    evaluator_environment["commitment_sha256"] = _canonical_json_sha256(
        evaluator_environment
    )
    evaluator_environment_sha256 = evaluator_environment["commitment_sha256"]
    comparisons = [
        paired_animal_bootstrap(
            prediction_table,
            "atlas_pose",
            "deepslice_mens_ai_ci",
            f"absolute_error_{axis}",
        )
        for axis in POSE_AXES
    ]
    joint = paired_animal_joint_superiority(
        prediction_table,
        "atlas_pose",
        "deepslice_mens_ai_ci",
        tuple(f"absolute_error_{axis}" for axis in POSE_AXES),
    )
    metrics.write_text(
        json.dumps(
            {
                "benchmark_id": "deepslice_s2p_1400_quicknii_ras_v2",
                "benchmark_role": "final_test_only",
                "section_count": 1400,
                "experiment_count": 10,
                "source": sealed_source,
                "evaluator_sha256": "8" * 64,
                "evaluator_environment_sha256": evaluator_environment_sha256,
                "animal_level_paired_bootstrap": comparisons,
                "animal_level_joint_superiority": joint,
            }
        ),
        encoding="utf-8",
    )
    metadata_path = export_folder / "atlas_pose.json"
    training_data = {
        "synthetic_manifests": metadata["manifest_sha256"],
        "registered_data": metadata["registered_data"]["sha256"],
        "atlas_data": metadata["atlas_data_sha256"],
    }
    presealed = sealed / "PRESEALED_COMMITMENT.json"
    presealed.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "benchmark_id": "deepslice_s2p_1400_quicknii_ras_v2",
                "model_sha256": file_sha256(model_path),
                "metadata_sha256": file_sha256(metadata_path),
                "training_source_sha256": metadata["source_sha256"],
                "training_data_sha256": training_data,
                "sealed_source_sha256": {
                    name: value
                    for name, value in registered_hashes.items()
                    if name != "nonsealed_image_tree_sha256"
                },
                "evaluator_environment": evaluator_environment,
            }
        ),
        encoding="utf-8",
    )
    claim = sealed / "SEALED_CLAIM.json"
    claim.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "benchmark_id": "deepslice_s2p_1400_quicknii_ras_v2",
                "model_sha256": file_sha256(model_path),
                "metadata_sha256": file_sha256(metadata_path),
                "presealed_commitment_sha256": file_sha256(presealed),
                "sealed_access_permitted_after_claim_only": True,
                "claimed_at_utc": "2026-08-11T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    receipt = sealed / "SEALED_CONSUMPTION_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "benchmark_id": "deepslice_s2p_1400_quicknii_ras_v2",
                "status": "completed",
                "model_sha256": file_sha256(model_path),
                "claim_sha256": file_sha256(claim),
                "presealed_commitment_sha256": file_sha256(presealed),
                "sealed_predictions_sha256": file_sha256(predictions),
                "sealed_metrics_sha256": file_sha256(metrics),
                "completed_at_utc": "2026-08-11T10:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    release_payload = sealed_release_report(
        prediction_table,
        comparisons,
        joint,
        file_sha256(model_path),
        file_sha256(metadata_path),
        metadata["preprocessing_contract_sha256"],
        metadata["source_sha256"],
        training_data,
        sealed_source,
        file_sha256(metrics),
        file_sha256(predictions),
        "8" * 64,
        evaluator_environment_sha256,
        file_sha256(presealed),
        file_sha256(claim),
        file_sha256(receipt),
        "2026-08-11T10:01:00+00:00",
    )
    release = sealed / "RELEASE_REPORT.json"
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    destination = tmp_path / "promoted"
    assert not destination.exists()
    pins = promote_export(export_folder, release, destination)
    assert pins == {
        "APPROVED_ATLAS_POSE_MODEL_SHA256": file_sha256(model_path),
        "APPROVED_ATLAS_POSE_METADATA_SHA256": file_sha256(metadata_path),
        "APPROVED_ATLAS_POSE_EVIDENCE_SHA256": file_sha256(release),
    }
    assert {path.name for path in destination.iterdir()} == {
        "atlas_pose.onnx",
        "atlas_pose.json",
        "provenance.json",
        "RELEASE_REPORT.json",
        "SEALED_metrics.json",
        "SEALED_predictions.csv",
        "PRESEALED_COMMITMENT.json",
        "SEALED_CLAIM.json",
        "SEALED_CONSUMPTION_RECEIPT.json",
    }
    predictions.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed_predictions_sha256|raw predictions"):
        promote_export(export_folder, release, tmp_path / "tampered-promotion")


def test_registered_provenance_cache_reuses_stats_and_detects_tampering(tmp_path, monkeypatch):
    images = (tmp_path / "images" / "1.jpg", tmp_path / "images" / "2.jpg")
    images[0].parent.mkdir()
    images[0].write_bytes(b"registered-image-1")
    images[1].write_bytes(b"registered-image-2")
    (tmp_path / "datasets.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "sections.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "section_image_id": 1,
                    "split": "train",
                    "relative_path": "images/1.jpg",
                },
                {
                    "section_image_id": 2,
                    "split": "validation",
                    "relative_path": "images/2.jpg",
                },
                {
                    "section_image_id": 3,
                    "split": "sealed_deepslice_s2p",
                    "relative_path": "images/3.jpg",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "downloads.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"section_image_id": 1, "sha256": file_sha256(images[0])}),
                json.dumps({"section_image_id": 2, "sha256": file_sha256(images[1])}),
                json.dumps({"section_image_id": 3, "sha256": "f" * 64}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    (tmp_path / "registered_image_quality.json").write_text("{}", encoding="utf-8")
    original_file_sha256 = trainer.file_sha256
    image_hash_calls = []

    def tracked_file_sha256(path):
        if Path(path) in images:
            image_hash_calls.append(Path(path))
        return original_file_sha256(path)

    monkeypatch.setattr(trainer, "file_sha256", tracked_file_sha256)
    hashes = registered_data_hashes(tmp_path)
    assert image_hash_calls == list(images)
    receipt = tmp_path / ".atlas_pose_cache" / "registered_data_hashes_v1.json"
    assert receipt.is_file()
    receipt_mtime_ns = receipt.stat().st_mtime_ns
    assert registered_data_hashes(tmp_path) == hashes
    assert image_hash_calls == list(images)
    assert receipt.stat().st_mtime_ns == receipt_mtime_ns
    image_stat = images[1].stat()
    os.utime(images[1], ns=(image_stat.st_atime_ns, image_stat.st_mtime_ns + 1_000_000_000))
    assert registered_data_hashes(tmp_path) == hashes
    assert image_hash_calls == [*images, images[1]]
    (tmp_path / "provenance.json").write_text('{"revision":2}', encoding="utf-8")
    refreshed_hashes = registered_data_hashes(tmp_path)
    assert refreshed_hashes["provenance.json"] != hashes["provenance.json"]
    assert image_hash_calls == [*images, images[1], *images]
    assert set(hashes) == {
        "datasets.jsonl",
        "sections.jsonl",
        "provenance.json",
        "downloads.jsonl",
        "registered_image_quality.json",
        "nonsealed_image_tree_sha256",
    }
    images[0].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="image checksum"):
        registered_data_hashes(tmp_path)
    assert image_hash_calls == [*images, images[1], *images, images[0]]
    assert "sealed_deepslice_s2p" not in inspect.getsource(held_out_reports)


def test_atlas_and_initial_model_state_provenance_are_content_bound(tmp_path):
    names = ("average_template_25.nrrd", "annotation_25.nrrd", "query.csv", "atlas_labels.pkl")
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(bytes((index, index + 1)))
    hashes = atlas_data_hashes(tmp_path)
    assert set(hashes) == set(names)
    (tmp_path / "query.csv").write_bytes(b"changed")
    assert atlas_data_hashes(tmp_path)["query.csv"] != hashes["query.csv"]

    module = nn.Linear(3, 2)
    digest = module_state_sha256(module)
    assert digest == module_state_sha256(module)
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    assert module_state_sha256(module) != digest


def test_unknown_renderer_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown AtlasPose renderer"):
        renderer_variant({"cohort": np.zeros(1)}, "typo")
