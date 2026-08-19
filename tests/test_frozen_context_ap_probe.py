import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import training.run_frozen_context_ap_probe as probe


ROOT = Path(__file__).parents[1]
CONFIGS = [
    ROOT
    / "training/configs/independent_oracle_frozen_context_ap_probe_fresh504322.json",
    ROOT
    / "training/configs/independent_spatial_moment_oracle_frozen_context_ap_probe_fresh504322.json",
]


def test_frozen_configs_bind_exact_sources_fresh_transfer_and_solver():
    configs = [probe.load_ap_probe_config(path) for path in CONFIGS]
    assert [value["arm"] for value in configs] == [
        "global_average",
        "spatial_moment",
    ]
    assert all(
        value["source_checkpoint_commit"] == probe.SOURCE_CHECKPOINT_COMMIT
        and value["source_panel_contract_sha256"]
        == probe.SOURCE_PANEL_CONTRACT_SHA256
        and value["data"]["primary_fresh_held_generator_seed"] == 504322
        and value["data"]["consumed_descriptive_held_generator_seed"] == 404322
        and value["data"]["consumed_404322_use"].startswith("excluded")
        and value["solver"] == probe.SOLVER_CONTRACT
        and value["fits"] == probe.FIT_CONTRACT
        and not value["product5_access"]
        and not value["calibration_access"]
        and not value["final_test_access"]
        and not value["learned_candidate_promotion"]
        for value in configs
    )
    assert configs[0]["data"] == configs[1]["data"]
    assert configs[0]["gates"] == configs[1]["gates"] == probe.GATE_CONTRACT
    assert all(
        value["source_checkpoint"]["terminal_model_state_sha256"]
        != value["source_checkpoint"]["terminal_named_parameter_sha256"]
        for value in configs
    )


def test_config_parser_rejects_a_rehashed_ignored_top_level_field(tmp_path):
    payload = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
    payload["ignored_adaptive_field"] = {"fresh_held_influences_fit": True}
    payload_without_hash = {
        name: value for name, value in payload.items() if name != "contract_sha256"
    }
    payload["contract_sha256"] = probe.foundation._canonical_sha256(
        payload_without_hash
    )
    path = tmp_path / "rehashed-extra-field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level fields changed"):
        probe.inspect_ap_probe_config(path)


def test_ce41_analytic_gradient_matches_central_difference():
    rng = np.random.default_rng(7)
    context = rng.normal(size=(7, 4))
    target = np.asarray([0, 1, 3, 2, 1, 0, 3])
    vector = rng.normal(scale=0.2, size=4 * 4 + 4)
    _, gradient = probe._ce41_objective_and_gradient(
        vector.copy(), context, target, 4, 1e-4
    )
    epsilon = 1e-6
    for index in (0, 3, 7, 11, 15, 16, 19):
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = probe._ce41_objective_and_gradient(
            plus, context, target, 4, 1e-4
        )[0]
        minus_loss = probe._ce41_objective_and_gradient(
            minus, context, target, 4, 1e-4
        )[0]
        numerical = (plus_loss - minus_loss) / (2 * epsilon)
        assert gradient[index] == pytest.approx(numerical, abs=1e-8)


def test_fit_passes_terminal_head_and_exact_options_to_scipy(monkeypatch):
    context = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10
    target = torch.tensor([0, 1, 2, 3])
    weight = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 100
    bias = torch.arange(4, dtype=torch.float64) / 100
    solver = {
        **probe.SOLVER_CONTRACT,
        "class_count": 4,
    }
    captured = {}

    def fake_minimize(fun, initial, args, method, jac, options):
        captured.update(
            initial=initial.copy(), args=args, method=method, jac=jac, options=options
        )
        return SimpleNamespace(
            x=initial,
            success=True,
            status=0,
            message="ok",
            nit=0,
            nfev=1,
        )

    monkeypatch.setattr(probe, "minimize", fake_minimize)
    fitted = probe._fit_ap_head(context, target, weight, bias, solver)
    assert np.array_equal(captured["initial"][:12], weight.numpy().ravel())
    assert np.array_equal(captured["initial"][12:], bias.numpy())
    assert captured["method"] == "L-BFGS-B"
    assert captured["jac"] is True
    assert captured["options"] == {
        "maxiter": 1000,
        "maxfun": 1250,
        "gtol": 1e-8,
        "ftol": 1e-10,
        "maxls": 20,
        "maxcor": 10,
    }
    assert captured["args"][0].dtype == np.float64
    assert fitted["weight"].dtype == fitted["bias"].dtype == torch.float64
    assert fitted["solver"]["independent_verification_function_evaluations"] == 1


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap_logits = nn.Linear(2, 41)
        self.register_buffer("ap_centers", torch.linspace(-4500.0, 500.0, 41))


class _ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_head = _Head()


def _panel(context_value):
    context = torch.full((24, 2), float(context_value))
    target = torch.tensor([3, 11, 19, 27, 35, 39]).repeat_interleave(4)
    centers = torch.linspace(-4500.0, 500.0, 41)
    return {
        "context": context,
        "truth_ap_um": centers.index_select(0, target),
        "target_ap_bin": target,
        "terminal_ap_residual_um": torch.zeros(24),
        "manifest_sha256": str(context_value),
        "oracle_source_image_sha256": str(context_value),
    }


def test_exact_three_fits_never_pass_fresh_held_context_to_optimizer(monkeypatch):
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(100), _panel(200)],
    }
    model = _ProbeModel()
    captured = []

    def fake_fit(context, target, initial_weight, initial_bias, solver):
        captured.append(
            {
                "values": sorted(context[:, 0].unique().tolist()),
                "weight": initial_weight.clone(),
                "bias": initial_bias.clone(),
            }
        )
        return {
            "weight": initial_weight.clone(),
            "bias": initial_bias.clone(),
            "solver": {
                "success": True,
                "status": 0,
                "message": "ok",
                "iterations": 0,
                "function_evaluations": 1,
                "independent_verification_function_evaluations": 1,
                "final_objective": 0.0,
                "independent_final_gradient_inf_norm": 0.0,
            },
        }

    monkeypatch.setattr(probe, "_fit_ap_head", fake_fit)
    probe._run_fits(
        {"fits": copy.deepcopy(probe.FIT_CONTRACT), "solver": probe.SOLVER_CONTRACT},
        model,
        features,
    )
    assert [value["values"] for value in captured] == [[1.0], [2.0], [1.0, 2.0]]
    assert all(
        torch.equal(value["weight"], model.pose_head.ap_logits.weight.double())
        and torch.equal(value["bias"], model.pose_head.ap_logits.bias.double())
        for value in captured
    )


class _ContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.pose_head = _Head()

    def initialize(self, image, mask, available):
        assert not mask.any() and not available.any()
        context = image.mean(dim=(-2, -1)).repeat(1, 2) * self.scale
        return {
            "pose_context": context,
            "continuous_residual": torch.zeros(len(image), 3, device=image.device),
        }


def test_context_extraction_is_eval_frozen_and_receives_no_pose_label():
    model = _ContextModel().eval()
    for value in model.parameters():
        value.requires_grad_(False)
    panels = {"seen": [], "fresh_held": []}
    for partition in panels:
        for index in range(2):
            panels[partition].append(
                {
                    "oracle_source_image": torch.ones(24, 1, 4, 5) * (index + 1),
                    "source_mask": torch.zeros(24, 1, 4, 5, dtype=torch.bool),
                    "mask_available": torch.zeros(24, 1, 1, 1),
                    "true_pose": torch.column_stack(
                        (torch.linspace(-4000, 0, 24), torch.zeros(24), torch.zeros(24))
                    ),
                    "manifest_sha256": f"{partition}-{index}",
                    "oracle_source_image_sha256": f"image-{partition}-{index}",
                }
            )
    before = probe.foundation._state_sha256(model)
    features = probe._extract_contexts(model, panels, torch.device("cpu"))
    assert probe.foundation._state_sha256(model) == before
    assert all(not value.requires_grad for value in model.parameters())
    assert torch.equal(features["seen"][0]["context"], torch.full((24, 2), 2.0))
    assert probe._input_integrity(panels) == {
        "source_mask_nonzero_count": 0,
        "mask_available_nonzero_count": 0,
    }
    panels["fresh_held"][1]["mask_available"][0] = 1
    with pytest.raises(RuntimeError, match="absent-outline input integrity"):
        probe._input_integrity(panels)


def test_fresh_manifest_disjointness_fails_before_tensor_generation(monkeypatch):
    def manifest(generator, panel, outline):
        return {
            "generator_manifest": {"manifest_sha256": generator},
            "manifest_sha256": panel,
            "outline_plan": {"plan_sha256": outline},
        }

    manifests = {
        "seen": [manifest("seen-generator", "seen-0", "outline-0")],
        "held": [manifest("fresh-generator", "fresh-0", "outline-1")],
    }
    expected = probe._manifest_hashes(manifests)
    monkeypatch.setattr(
        probe.pose_diagnostic, "fixed_panel_manifests", lambda synthetic, config: manifests
    )
    monkeypatch.setattr(
        probe.pose_diagnostic,
        "_prepare_fixed_panels",
        lambda *args: pytest.fail("image tensors generated before disjointness gate"),
    )
    source_receipt = {
        "fixed_panel_contract": {
            "generator_realization_sha256": {"seen": "fresh-generator"},
            "manifests": {"seen": ["seen-0"]},
        }
    }
    with pytest.raises(RuntimeError, match="generator realization is not disjoint"):
        probe._prepare_probe_panels(
            {
                "data": {
                    "primary_fresh_held_generator_seed": 504322,
                    "expected_manifest_sha256": expected,
                }
            },
            {"data": {"held_base_seed": 404322}},
            source_receipt,
            object(),
            object(),
            torch.device("cpu"),
        )


def _prediction(correct, count, mae=100.0, prediction_sd=1000.0, truth_sd=1000.0):
    target = torch.arange(count) % 41
    predicted = target.clone()
    if correct < count:
        predicted[correct:] = (predicted[correct:] + 1) % 41
    return {
        "logits": torch.zeros(count, 41, dtype=torch.float64),
        "predicted_bin": predicted,
        "target_bin": target,
        "truth_ap_um": torch.zeros(count, dtype=torch.float64),
        "bin_center_ap_um": torch.zeros(count, dtype=torch.float64),
        "terminal_residual_added_ap_um": torch.zeros(count, dtype=torch.float64),
        "correct_count": correct,
        "sample_count": count,
        "bin_center_ap_mae_um": mae,
        "terminal_residual_added_ap_mae_um": 10000.0,
        "prediction_sd_um": prediction_sd,
        "truth_sd_um": truth_sd,
    }


def _fit(train_correct, train_count, test_correct, test_count, mae=100.0):
    return {
        "weight": torch.zeros(41, 2, dtype=torch.float64),
        "bias": torch.zeros(41, dtype=torch.float64),
        "solver": {
            "success": True,
            "status": 0,
            "message": "ok",
            "iterations": 2,
            "function_evaluations": 3,
            "independent_verification_function_evaluations": 1,
            "final_objective": 0.1,
            "independent_final_gradient_inf_norm": 1e-9,
        },
        "train": _prediction(train_correct, train_count),
        "test": _prediction(test_correct, test_count, mae=mae),
    }


def _qualification_inputs():
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    fits = {
        "seen_panel_0_to_1": _fit(24, 24, 23, 24),
        "seen_panel_1_to_0": _fit(24, 24, 23, 24),
        "seen_to_fresh_held": _fit(48, 48, 48, 48, mae=250.0),
    }
    config = {"gates": copy.deepcopy(probe.GATE_CONTRACT)}
    return config, features, fits


def test_qualification_uses_independent_solver_and_primary_bin_center_gates():
    config, features, fits = _qualification_inputs()
    passed = probe.qualification_status(config, features, fits)
    assert passed["decision"] == "go"
    assert passed["classification"] == "joint-ap-head-optimization-underconverged"
    assert passed["observed"]["fresh_held_terminal_residual_added_ap_mae_um"] == 10000

    failed = copy.deepcopy(fits)
    failed["seen_panel_0_to_1"]["solver"]["success"] = False
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == "ap-head-solver-result-not-successful"

    failed = copy.deepcopy(fits)
    failed["seen_panel_0_to_1"]["solver"][
        "independent_final_gradient_inf_norm"
    ] = 1.1e-8
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == (
        "ap-head-independent-gradient-convergence-failed"
    )

    failed = copy.deepcopy(fits)
    failed["seen_panel_0_to_1"]["solver"]["function_evaluations"] = 1251
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == "ap-head-preregistered-budget-exceeded"
    assert not result["checks"]["solver_function_evaluation_budget"]["passed"]

    failed = copy.deepcopy(fits)
    for value in failed.values():
        value["solver"]["function_evaluations"] = 1250
    failed["seen_to_fresh_held"]["train"]["sample_count"] = 49
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == "ap-head-preregistered-budget-exceeded"
    assert not result["checks"]["optimizer_head_sample_presentation_budget"]["passed"]

    failed = copy.deepcopy(fits)
    failed["seen_panel_0_to_1"]["train"]["correct_count"] = 23
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == (
        "preregistered-regularized-ap-head-training-fit-insufficient"
    )

    failed = copy.deepcopy(fits)
    failed["seen_panel_0_to_1"]["test"]["correct_count"] = 22
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == "terminal-ap-context-panel-resampling-instability"

    failed = copy.deepcopy(fits)
    failed["seen_to_fresh_held"]["test"]["bin_center_ap_mae_um"] = 250.1
    result = probe.qualification_status(config, features, failed)
    assert result["classification"] == (
        "terminal-ap-context-generator-generalization-insufficient"
    )
