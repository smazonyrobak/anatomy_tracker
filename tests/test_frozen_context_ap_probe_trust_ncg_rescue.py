import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import training.run_frozen_context_ap_probe_trust_ncg_rescue as rescue


ROOT = Path(__file__).parents[1]
CONFIGS = [
    ROOT
    / "training/configs/independent_oracle_frozen_context_ap_probe_trust_ncg_rescue.json",
    ROOT
    / "training/configs/independent_spatial_moment_oracle_frozen_context_ap_probe_trust_ncg_rescue.json",
]


def test_rescue_source_has_no_model_or_data_generation_path():
    source = (
        ROOT / "training/run_frozen_context_ap_probe_trust_ncg_rescue.py"
    ).read_text(encoding="utf-8")
    assert "SyntheticRegistrationGenerator" not in source
    assert "IndependentSyntheticData" not in source
    assert ".initialize(" not in source
    assert "torch.load(tensor_path" in source


def test_frozen_configs_bind_consumed_stops_and_exact_solver_contract():
    configs = [rescue.load_rescue_config(path) for path in CONFIGS]
    assert [value["arm"] for value in configs] == ["global_average", "spatial_moment"]
    assert all(
        value["solver"] == rescue.SOLVER_CONTRACT
        and value["fits"] == rescue.source_probe.FIT_CONTRACT
        and value["gates"] == rescue.source_probe.GATE_CONTRACT
        and value["environment"] == rescue.ENVIRONMENT_CONTRACT
        and value["source_probe"]["source_tensor_artifact_sha256"]
        in {
            "0179acda8bba37e7a2c06153f7a99723a46faf81af3e89271571d19d98552f46",
            "2e73edb82e03aba04fe93e308df035bec33e2a0d002de463b054b28c61946405",
        }
        and not value["product5_access"]
        and not value["calibration_access"]
        and not value["final_test_access"]
        and not value["learned_candidate_promotion"]
        for value in configs
    )


def test_config_parser_rejects_rehashed_extra_field(tmp_path):
    payload = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
    payload["adaptive_retry"] = True
    unhashed = {name: value for name, value in payload.items() if name != "contract_sha256"}
    payload["contract_sha256"] = rescue.foundation._canonical_sha256(unhashed)
    path = tmp_path / "rehashed-extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level fields changed"):
        rescue.inspect_rescue_config(path)


def _fixture(n):
    rng = np.random.Generator(np.random.PCG64(1729 + n))
    context = rng.standard_normal((n, 192))
    target = np.arange(n, dtype=np.int64) % 41
    vector = 0.02 * rng.standard_normal(41 * 192 + 41)
    direction = rng.standard_normal(len(vector))
    direction /= np.linalg.norm(direction)
    return context, target, vector, direction


@pytest.mark.parametrize("n", [24, 48])
def test_analytic_hvp_matches_gradient_central_difference(n):
    context, target, vector, direction = _fixture(n)
    epsilon = 1e-6
    plus = rescue.source_probe._ce41_objective_and_gradient(
        vector + epsilon * direction, context, target, 41, 1e-4
    )[1]
    minus = rescue.source_probe._ce41_objective_and_gradient(
        vector - epsilon * direction, context, target, 41, 1e-4
    )[1]
    numerical = (plus - minus) / (2 * epsilon)
    analytic = rescue._ce41_hessian_vector_product(
        vector, direction, context, target, 41, 1e-4
    )
    assert np.max(np.abs(analytic - numerical)) < 1e-9
    assert float(direction @ analytic) >= 1e-4 * float(direction @ direction)


def test_trust_ncg_fixture_converges_and_repeats_bitwise():
    context, target, vector, _ = _fixture(24)
    weight = torch.from_numpy(vector[: 41 * 192].reshape(41, 192).copy())
    bias = torch.from_numpy(vector[41 * 192 :].copy())
    first = rescue._fit_ap_head_trust_ncg(
        torch.from_numpy(context), torch.from_numpy(target), weight, bias, rescue.SOLVER_CONTRACT
    )
    second = rescue._fit_ap_head_trust_ncg(
        torch.from_numpy(context), torch.from_numpy(target), weight, bias, rescue.SOLVER_CONTRACT
    )
    assert first["solver"]["success"]
    assert first["solver"]["independent_final_gradient_inf_norm"] <= 1e-8
    assert first["solver"]["function_evaluations"] < 1250
    assert torch.equal(first["weight"], second["weight"])
    assert torch.equal(first["bias"], second["bias"])
    assert first["solver"] == second["solver"]


def test_preregistered_extreme_scale_fixture_converges_within_operator_cap():
    rng = np.random.default_rng(104)
    context = rng.standard_normal((48, 192)) * np.logspace(-4, 4, 192)
    target = np.arange(48, dtype=np.int64) % 41
    vector = 0.02 * rng.standard_normal(41 * 192 + 41)
    result = rescue._fit_ap_head_trust_ncg(
        torch.from_numpy(context),
        torch.from_numpy(target),
        torch.from_numpy(vector[: 41 * 192].reshape(41, 192).copy()),
        torch.from_numpy(vector[41 * 192 :].copy()),
        rescue.SOLVER_CONTRACT,
    )
    assert result["solver"]["success"]
    assert result["solver"]["function_evaluations"] <= 1250
    assert result["solver"]["independent_final_gradient_inf_norm"] <= 1e-8


def test_combined_operator_cap_is_checked_before_calls(monkeypatch):
    def fake_minimize(fun, initial, args, method, jac, hessp, callback, options):
        direction = np.ones_like(initial)
        fun(initial, *args)
        hessp(initial, direction, *args)
        fun(initial, *args)
        hessp(initial, direction, *args)

    monkeypatch.setattr(rescue, "minimize", fake_minimize)
    solver = {
        **rescue.SOLVER_CONTRACT,
        "class_count": 2,
        "combined_fg_hvp_call_limit_each_fit": 3,
    }
    result = rescue._fit_ap_head_trust_ncg(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
        torch.zeros(2, 2),
        torch.zeros(2),
        solver,
    )
    assert result["solver"]["budget_exhausted"]
    assert not result["solver"]["success"]
    assert result["solver"]["function_evaluations"] == 3
    assert result["solver"]["objective_gradient_evaluations"] == 2
    assert result["solver"]["hessian_vector_evaluations"] == 1
    assert result["solver"]["independent_verification_function_evaluations"] == 1


def test_exact_trust_ncg_options_are_passed_to_scipy(monkeypatch):
    captured = {}

    def fake_minimize(fun, initial, args, method, jac, hessp, callback, options):
        captured.update(method=method, jac=jac, options=options)
        fun(initial, *args)
        hessp(initial, np.ones_like(initial), *args)
        return SimpleNamespace(
            x=initial,
            success=True,
            status=0,
            message="ok",
            nit=0,
            nfev=1,
            njev=1,
            nhev=2,
        )

    monkeypatch.setattr(rescue, "minimize", fake_minimize)
    rescue._fit_ap_head_trust_ncg(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
        torch.zeros(2, 2),
        torch.zeros(2),
        {**rescue.SOLVER_CONTRACT, "class_count": 2},
    )
    assert captured == {
        "method": "trust-ncg",
        "jac": True,
        "options": {
            "gtol": 1e-8,
            "maxiter": 250,
            "initial_trust_radius": 1.0,
            "max_trust_radius": 1000.0,
            "eta": 0.15,
            "disp": False,
            "return_all": False,
        },
    }


def _panel(value):
    target = torch.tensor([3, 11, 19, 27, 35, 39]).repeat_interleave(4)
    centers = torch.linspace(-4500.0, 500.0, 41)
    return {
        "context": torch.full((24, 192), float(value)),
        "truth_ap_um": centers.index_select(0, target),
        "target_ap_bin": target,
        "terminal_ap_residual_um": torch.zeros(24),
        "manifest_sha256": str(value),
        "oracle_source_image_sha256": str(value),
    }


def test_exact_three_fits_use_only_seen_context_for_optimization(monkeypatch):
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(100), _panel(200)],
    }
    terminal = {"weight": torch.zeros(41, 192), "bias": torch.zeros(41)}
    captured = []

    def fake_fit(context, target, initial_weight, initial_bias, solver):
        captured.append(
            {
                "contexts": sorted(context[:, 0].unique().tolist()),
                "initial": torch.cat((initial_weight.flatten(), initial_bias)).clone(),
            }
        )
        return {
            "weight": initial_weight.clone(),
            "bias": initial_bias.clone(),
            "solver": {
                "success": True,
                "status": 0,
                "message": "ok",
                "iterations": 1,
                "function_evaluations": 2,
                "objective_gradient_evaluations": 1,
                "hessian_vector_evaluations": 1,
                "independent_verification_function_evaluations": 1,
                "budget_call_limit": 1250,
                "budget_exhausted": False,
                "final_objective": 0.1,
                "independent_final_gradient_inf_norm": 1e-9,
                "independent_final_gradient_l2_norm": 2e-9,
                "initial_vector_sha256": "initial",
                "last_accepted_vector_sha256": "final",
                "scipy_nfev": 1,
                "scipy_njev": 1,
                "scipy_nhev": 2,
            },
        }

    monkeypatch.setattr(rescue, "_fit_ap_head_trust_ncg", fake_fit)
    rescue._run_fits(
        {"fits": copy.deepcopy(rescue.source_probe.FIT_CONTRACT), "solver": rescue.SOLVER_CONTRACT},
        features,
        terminal,
    )
    assert [value["contexts"] for value in captured] == [
        [1.0],
        [2.0],
        [1.0, 2.0],
    ]
    expected = torch.cat((terminal["weight"].double().flatten(), terminal["bias"].double()))
    assert all(torch.equal(value["initial"], expected) for value in captured)


def _prediction(correct, count, mae=100.0):
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
        "terminal_residual_added_ap_mae_um": mae,
        "prediction_sd_um": 1000.0,
        "truth_sd_um": 1000.0,
    }


def _fit(train_count, test_correct, test_count, exhausted=False):
    return {
        "weight": torch.zeros(41, 192, dtype=torch.float64),
        "bias": torch.zeros(41, dtype=torch.float64),
        "solver": {
            "success": not exhausted,
            "status": 4 if exhausted else 0,
            "message": "budget" if exhausted else "ok",
            "iterations": 2,
            "function_evaluations": 3,
            "objective_gradient_evaluations": 2,
            "hessian_vector_evaluations": 1,
            "independent_verification_function_evaluations": 1,
            "budget_call_limit": 1250,
            "budget_exhausted": exhausted,
            "final_objective": 0.1,
            "independent_final_gradient_inf_norm": 1e-9,
            "independent_final_gradient_l2_norm": 2e-9,
        },
        "train": _prediction(train_count, train_count),
        "test": _prediction(test_correct, test_count),
    }


def test_budget_exhaustion_precedes_downstream_interpretation():
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    fits = {
        "seen_panel_0_to_1": _fit(24, 23, 24, exhausted=True),
        "seen_panel_1_to_0": _fit(24, 23, 24),
        "seen_to_fresh_held": _fit(48, 48, 48),
    }
    result = rescue.qualification_status(
        {
            "gates": copy.deepcopy(rescue.source_probe.GATE_CONTRACT),
            "solver": rescue.SOLVER_CONTRACT,
        },
        features,
        fits,
    )
    assert result["decision"] == "stop"
    assert result["classification"] == "ap-head-preregistered-budget-exhausted"
    assert not result["causal_classification_allowed"]
    assert not result["fresh_held_transfer_interpretation_allowed"]


def test_converged_scientific_stop_remains_causally_interpretable():
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    fits = {
        "seen_panel_0_to_1": _fit(24, 23, 24),
        "seen_panel_1_to_0": _fit(24, 23, 24),
        "seen_to_fresh_held": _fit(48, 48, 48),
    }
    fits["seen_to_fresh_held"]["test"]["bin_center_ap_mae_um"] = 251.0
    result = rescue.qualification_status(
        {
            "gates": copy.deepcopy(rescue.source_probe.GATE_CONTRACT),
            "solver": rescue.SOLVER_CONTRACT,
        },
        features,
        fits,
    )
    assert result["decision"] == "stop"
    assert result["classification"] == (
        "terminal-ap-context-generator-generalization-insufficient"
    )
    assert result["causal_classification_allowed"]
    assert result["fresh_held_transfer_interpretation_allowed"]


@pytest.mark.parametrize("iterations", [250, 251])
def test_iteration_exhaustion_precedes_generic_solver_failure(iterations):
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    fits = {
        "seen_panel_0_to_1": _fit(24, 23, 24),
        "seen_panel_1_to_0": _fit(24, 23, 24),
        "seen_to_fresh_held": _fit(48, 48, 48),
    }
    fits["seen_panel_0_to_1"]["solver"]["success"] = False
    fits["seen_panel_0_to_1"]["solver"]["iterations"] = iterations
    result = rescue.qualification_status(
        {
            "gates": copy.deepcopy(rescue.source_probe.GATE_CONTRACT),
            "solver": rescue.SOLVER_CONTRACT,
        },
        features,
        fits,
    )
    assert result["classification"] == (
        "ap-head-preregistered-iteration-budget-exhausted"
    )
    assert not result["causal_classification_allowed"]


def test_operator_accounting_is_fail_closed():
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    fits = {
        "seen_panel_0_to_1": _fit(24, 23, 24),
        "seen_panel_1_to_0": _fit(24, 23, 24),
        "seen_to_fresh_held": _fit(48, 48, 48),
    }
    fits["seen_panel_0_to_1"]["solver"]["function_evaluations"] = 4
    result = rescue.qualification_status(
        {
            "gates": copy.deepcopy(rescue.source_probe.GATE_CONTRACT),
            "solver": rescue.SOLVER_CONTRACT,
        },
        features,
        fits,
    )
    assert result["classification"] == "ap-head-operator-accounting-invalid"
    assert not result["causal_classification_allowed"]


def test_context_target_and_terminal_head_hashes_are_fail_closed():
    features = {
        "seen": [_panel(1), _panel(2)],
        "fresh_held": [_panel(3), _panel(4)],
    }
    terminal = {"weight": torch.zeros(41, 192), "bias": torch.zeros(41)}
    binding = {
        "feature_sha256": rescue._feature_hashes(features),
        "terminal_ap_head_sha256": {
            name: rescue.foundation._tensor_sha256(value) for name, value in terminal.items()
        },
    }
    rescue._validate_tensor_contract(features, terminal, binding)
    features["fresh_held"][0]["target_ap_bin"][0] += 1
    with pytest.raises(RuntimeError, match="context or target tensor hash changed"):
        rescue._validate_tensor_contract(features, terminal, binding)


def test_binary_receipt_gate_precedes_tensor_deserialization(tmp_path, monkeypatch):
    source_run = tmp_path / "consumed-stop"
    source_run.mkdir()
    monkeypatch.setattr(rescue.foundation, "_binary_sha256", lambda path: "wrong")
    monkeypatch.setattr(
        rescue.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("untrusted tensor was deserialized"),
    )
    with pytest.raises(RuntimeError, match="receipt file hash changed"):
        rescue._load_bound_artifact(
            {"source_probe": rescue.ARM_CONTRACTS["global_average"]}, source_run
        )


def test_environment_requires_exact_versions_and_prestart_thread_binding(monkeypatch):
    for name, value in rescue.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    observed = rescue._validate_environment({"environment": rescue.ENVIRONMENT_CONTRACT})
    assert observed["blas_thread_environment"] == rescue.THREAD_ENVIRONMENT
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="before Python starts"):
        rescue._validate_environment({"environment": rescue.ENVIRONMENT_CONTRACT})
