"""Solver-only trust-NCG rescue for the consumed frozen-context AP probe."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import threadpoolctl
from scipy.optimize import minimize

import training.run_frozen_context_ap_probe as source_probe
import training.run_independent_initializer_foundation as foundation


REPOSITORY_ROOT = Path(__file__).parents[1]
SCHEMA_VERSION = 1
PURPOSE = "development-only-frozen-context-ap-trust-ncg-solver-rescue"
FORMAT = "frozen-context-ap-trust-ncg-rescue-tensors-v1"
SOURCE_FILES = (
    "training/run_frozen_context_ap_probe.py",
    "training/run_frozen_context_ap_probe_trust_ncg_rescue.py",
    "training/run_independent_initializer_foundation.py",
    "training/train_independent_joint.py",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ENVIRONMENT_CONTRACT = {
    "numpy_version": "2.4.4",
    "scipy_version": "1.17.1",
    "torch_version": "2.11.0+cu128",
    "threadpoolctl_version": "3.6.0",
    "blas_thread_environment": THREAD_ENVIRONMENT,
    "blas_thread_environment_set_before_python": True,
    "loaded_blas_and_openmp_thread_count": 1,
    "torch_intraop_thread_count": 1,
    "torch_interop_thread_count": 8,
}
SOLVER_CONTRACT = {
    "implementation": "scipy.optimize.minimize",
    "method": "trust-ncg",
    "jacobian": "combined-analytic-objective-gradient",
    "hessian_vector_product": "analytic-full-coordinate",
    "dtype": "float64-cpu",
    "initialization": "consumed-artifact-terminal-ap-head-weight-and-bias",
    "objective": source_probe.SOLVER_CONTRACT["objective"],
    "class_count": 41,
    "l2_coefficient": 0.0001,
    "ap_bin_center_minimum_um": -4500.0,
    "ap_bin_center_maximum_um": 500.0,
    "ap_bin_center_count": 41,
    "ap_bin_center_dtype": "torch-float32",
    "gtol": 1e-8,
    "maxiter": 250,
    "initial_trust_radius": 1.0,
    "max_trust_radius": 1000.0,
    "eta": 0.15,
    "disp": False,
    "return_all": False,
    "combined_fg_hvp_call_limit_each_fit": 1250,
    "held_access_during_fit": False,
    "feature_normalization": None,
    "retry": None,
    "fallback_solver": None,
    "cross_fit_warm_start": False,
}


def _features(
    seen_context_0,
    seen_context_1,
    held_context_0,
    held_context_1,
    seen_residual_0,
    seen_residual_1,
    held_residual_0,
    held_residual_1,
):
    target = "8ba5e88b39ec13cf5115797db594ec817065fb5c48773f7f8f74232e024fc539"
    truth = "c8c393fb6a537634ca68d89ec6509a300428f85f98573a4cf85a5461220296a1"
    return {
        "seen": [
            {
                "context": seen_context_0,
                "target_ap_bin": target,
                "terminal_ap_residual_um": seen_residual_0,
                "truth_ap_um": truth,
            },
            {
                "context": seen_context_1,
                "target_ap_bin": target,
                "terminal_ap_residual_um": seen_residual_1,
                "truth_ap_um": truth,
            },
        ],
        "fresh_held": [
            {
                "context": held_context_0,
                "target_ap_bin": target,
                "terminal_ap_residual_um": held_residual_0,
                "truth_ap_um": truth,
            },
            {
                "context": held_context_1,
                "target_ap_bin": target,
                "terminal_ap_residual_um": held_residual_1,
                "truth_ap_um": truth,
            },
        ],
    }


ARM_CONTRACTS = {
    "global_average": {
        "output_name": "independent-oracle-frozen-context-ap-probe-trust-ncg-rescue-v1",
        "source_run_name": "independent-oracle-frozen-context-ap-probe-fresh504322-v1",
        "source_config_contract_sha256": "bcfaeac84717ae702f4b239f40004b3525bccb5301e0ddbdade3bae8cd47a2a5",
        "source_config_file_sha256": "84cf8e0ba4e65a0f91c01add46a9f194d1031924225876fbbcda37bd407ff57d",
        "source_receipt_file_sha256": "7f8a234f3df05414821a82f15a5f5943b7a49c160af55a5de5da8b57470a271c",
        "source_setup_sha256": "d680f5e379c902d80becb89e7c2acb8e2137daebfea4a1f649db689c8b09eef5",
        "source_result_sha256": "e2cf23031378beeffadd3d4805095e0787956ab25ed70052c219de183f0b0b59",
        "source_tensor_artifact_sha256": "0179acda8bba37e7a2c06153f7a99723a46faf81af3e89271571d19d98552f46",
        "terminal_ap_head_sha256": {
            "weight": "099b9e432fa05af7f615cc4d0e97054a31b0521efe5bdfc5500ddc4d21bdc3f6",
            "bias": "6690d2fe7f0ac20b1924b4247de20315d172d8c6daf55845ab9a163838bca8f6",
        },
        "feature_sha256": _features(
            "3072249352428091a621278278b30202d62c1bbe0dedb414bdc22d436f636352",
            "098065d46956134e5526f67ffd82431df47b5da396db21af0dbe7b2bfb468c90",
            "9df51f21c7c7f1d0d5b9e49467da36e5547e9ffcacec719eadf7bf4eb590734d",
            "7195660f0e17e64c52f2546bcae5e69079e253d2db8b29b711013ec9dddaa85f",
            "8c96d3da0b42e42bcf086001d8070274f1e0b1b39e18119994b60dc70ee6aa7a",
            "8f530b97bb69797daca312b4455bc470b33282db5d2c16595d9d9689cce8c45f",
            "2b2381a5300f8b8b98c50176efddc88f3dfcebf2ec8e5457ac3d914295323de4",
            "5d277bfc10358bea1d513371fadb95cbb8c450cf78d13dd654452bed8043a71e",
        ),
        "learned_checkpoint_dependencies": [
            {
                "role": "diagnostic-terminal-context-source-only",
                "run_name": "independent-oracle-similarity-pose-identifiability-300-r4322-v1",
                "resume_state_sha256": "c2a355b079e59be28fb84b74c34cd3cf380e83788711cf2a52be4f23d8518aa9",
                "terminal_model_state_sha256": "0efb78806303ec88b4465f3f9b19eb0898d214f5aba36b31829c353948061626",
            }
        ],
    },
    "spatial_moment": {
        "output_name": "independent-spatial-moment-oracle-frozen-context-ap-probe-trust-ncg-rescue-v1",
        "source_run_name": "independent-spatial-moment-oracle-frozen-context-ap-probe-fresh504322-v1",
        "source_config_contract_sha256": "50650ba2244d1cd25cef1528cbd73afe2416f73f5f9736173d8f6540421b35d3",
        "source_config_file_sha256": "9b1e7eba683a94c83e81cc7e1ab59b05dc939a99d74dc15f8ba0c22d1896b8df",
        "source_receipt_file_sha256": "1fea7d3e24b8c48bd08d75d8ad77b038326d3f8a5753f7ca79adc948a6d8f3cb",
        "source_setup_sha256": "3485d060b3269d7f475e9ff1de52581d3f8737ff966f8acd49e8a42b5a9003c0",
        "source_result_sha256": "395a6f0d0304994db12d296c1ec6229ae9c533552ab57c35a4bfaded600bbd43",
        "source_tensor_artifact_sha256": "2e73edb82e03aba04fe93e308df035bec33e2a0d002de463b054b28c61946405",
        "terminal_ap_head_sha256": {
            "weight": "6aea16c42edf586a489b0068dca74d5778f730cc4f395cd76504806437c332c7",
            "bias": "08f450533ea2a8a72406caf9d63e1f8392681be9d0cb708777bc5cc7d2ce5c77",
        },
        "feature_sha256": _features(
            "9af826587452d891b697dd16bab1be2e1a28e53f030aeb5023f83f7e12faaf7a",
            "ecd7a3e016e819807f27505f731d28d5f96602c9511dc16c375201df4e0e5d75",
            "a6281657f1ea6d591670f82162ca7d53bf156398993bb97b54ae858491242018",
            "868acb95d7caaf294e04344086c5436540d9c4d7dda96beb871776285b4460c9",
            "b031a8f7c355d846a679137cf6d84681cdff5fb3728855c426f77083f9e25996",
            "614d98d1ce41697530d578cdd0235a080d5d2e98e4dbcc3ea2cd695d0f26782d",
            "907e96dd88d5416d46092d39cadb9c54759c7c6f2453553e8d2731a2cec6e96b",
            "789ca0d0aebf062cfb41b161d04948478b91438aa805903e7eca586a0dc1254a",
        ),
        "learned_checkpoint_dependencies": [
            {
                "role": "diagnostic-terminal-context-source-only",
                "run_name": "independent-spatial-moment-oracle-similarity-pose-identifiability-300-r4322-v1",
                "resume_state_sha256": "812b044b5d082aa0119c5c4d49c4dad9f6080a82b3819289164e2e7ea773d615",
                "terminal_model_state_sha256": "061b0bd978e00005c57a5c6c125d6481f47420f152af6cf81017edd6cf8f4363",
            }
        ],
    },
}


class TrustNcgBudgetExhausted(RuntimeError):
    pass


def inspect_rescue_config(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = raw.pop("contract_sha256")
    if foundation._canonical_sha256(raw) != contract:
        raise ValueError("trust-NCG rescue config hash differs from its payload")
    expected_fields = {
        "schema_version",
        "frozen",
        "name",
        "purpose",
        "role",
        "arm",
        "product5_access",
        "calibration_access",
        "final_test_access",
        "learned_candidate_promotion",
        "source_probe",
        "diagnostic_artifact_dependencies",
        "learned_checkpoint_dependencies",
        "paths",
        "fits",
        "solver",
        "gates",
        "environment",
        "artifacts",
        "lineage",
    }
    if set(raw) != expected_fields:
        raise ValueError("trust-NCG rescue config top-level fields changed")
    arm = raw.get("arm")
    if arm not in ARM_CONTRACTS:
        raise ValueError("trust-NCG rescue arm is not allowlisted")
    source = ARM_CONTRACTS[arm]
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("frozen") is not True:
        raise ValueError("trust-NCG rescue config is not frozen schema v1")
    if raw.get("name") != source["output_name"] or raw.get("purpose") != PURPOSE:
        raise ValueError("trust-NCG rescue identity changed")
    if raw.get("role") != "diagnostic-solver-rescue-not-model-selection":
        raise ValueError("trust-NCG rescue role changed")
    if any(
        raw.get(name) is not False
        for name in ("product5_access", "calibration_access", "final_test_access")
    ) or raw.get("learned_candidate_promotion") is not False:
        raise ValueError("trust-NCG rescue must remain non-protected and non-promotional")
    expected_source = {
        name: value
        for name, value in source.items()
        if name not in ("output_name", "learned_checkpoint_dependencies")
    }
    if raw.get("source_probe") != expected_source:
        raise ValueError("trust-NCG rescue source artifact binding changed")
    if raw.get("diagnostic_artifact_dependencies") != [
        {
            "role": "consumed-frozen-context-tensor-source-only",
            "run_name": source["source_run_name"],
            "tensor_artifact_sha256": source["source_tensor_artifact_sha256"],
            "result_sha256": source["source_result_sha256"],
        }
    ]:
        raise ValueError("trust-NCG rescue diagnostic dependency changed")
    if raw.get("learned_checkpoint_dependencies") != source[
        "learned_checkpoint_dependencies"
    ]:
        raise ValueError("trust-NCG rescue learned dependency changed")
    if raw.get("paths") != {"run_root_env": "ATLAS_JOINT_RUN_ROOT"}:
        raise ValueError("trust-NCG rescue paths changed")
    if raw.get("fits") != source_probe.FIT_CONTRACT:
        raise ValueError("trust-NCG rescue fits changed")
    if raw.get("solver") != SOLVER_CONTRACT:
        raise ValueError("trust-NCG rescue solver changed")
    if raw.get("gates") != source_probe.GATE_CONTRACT:
        raise ValueError("trust-NCG rescue scientific gates changed")
    if raw.get("environment") != ENVIRONMENT_CONTRACT:
        raise ValueError("trust-NCG rescue environment changed")
    if raw.get("artifacts") != {
        "tensor_payload": "trust_ncg_rescue_tensors.pt",
        "receipt": "diagnostic_receipt.json",
        "raw_predictions": True,
        "save_logits": True,
        "save_fitted_heads": True,
        "copy_contexts": False,
    }:
        raise ValueError("trust-NCG rescue artifacts changed")
    if set(raw.get("lineage", {})) != {"source_sha256"} or set(
        raw["lineage"]["source_sha256"]
    ) != set(SOURCE_FILES):
        raise ValueError("trust-NCG rescue source lineage is incomplete")
    raw["contract_sha256"] = contract
    raw["config_file_sha256"] = foundation._source_sha256(path)
    return raw


def load_rescue_config(path: str | Path) -> dict:
    config = inspect_rescue_config(path)
    for relative, expected in config["lineage"]["source_sha256"].items():
        if foundation._source_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"source lineage changed: {relative}")
    return config


def _validate_environment(config: dict) -> dict:
    observed = {
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": str(torch.__version__),
        "threadpoolctl_version": threadpoolctl.__version__,
        "blas_thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "loaded_threadpools": [
            {
                name: value.get(name)
                for name in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "version",
                    "threading_layer",
                    "architecture",
                )
            }
            for value in threadpoolctl.threadpool_info()
            if value.get("user_api") in {"blas", "openmp"}
        ],
        "torch_intraop_thread_count": torch.get_num_threads(),
        "torch_interop_thread_count": torch.get_num_interop_threads(),
    }
    expected = config["environment"]
    if observed["numpy_version"] != expected["numpy_version"]:
        raise RuntimeError("NumPy version differs from the frozen rescue environment")
    if observed["scipy_version"] != expected["scipy_version"]:
        raise RuntimeError("SciPy version differs from the frozen rescue environment")
    if observed["torch_version"] != expected["torch_version"]:
        raise RuntimeError("PyTorch version differs from the frozen rescue environment")
    if observed["threadpoolctl_version"] != expected["threadpoolctl_version"]:
        raise RuntimeError("threadpoolctl version differs from the frozen rescue environment")
    if observed["blas_thread_environment"] != expected["blas_thread_environment"]:
        raise RuntimeError("BLAS thread environment must be set before Python starts")
    if not observed["loaded_threadpools"] or any(
        value["num_threads"] != expected["loaded_blas_and_openmp_thread_count"]
        for value in observed["loaded_threadpools"]
    ):
        raise RuntimeError("loaded BLAS and OpenMP runtimes must use one thread")
    if (
        observed["torch_intraop_thread_count"]
        != expected["torch_intraop_thread_count"]
        or observed["torch_interop_thread_count"]
        != expected["torch_interop_thread_count"]
    ):
        raise RuntimeError("PyTorch thread counts differ from the frozen environment")
    return observed


def _resolve_paths(config: dict) -> tuple[Path, Path]:
    try:
        root = Path(os.environ[config["paths"]["run_root_env"]]).resolve()
    except KeyError as error:
        raise RuntimeError("ATLAS_JOINT_RUN_ROOT is required") from error
    return root / config["source_probe"]["source_run_name"], root / config["name"]


def _feature_hashes(features: dict) -> dict:
    return {
        partition: [
            {
                name: foundation._tensor_sha256(panel[name])
                for name in (
                    "context",
                    "target_ap_bin",
                    "terminal_ap_residual_um",
                    "truth_ap_um",
                )
            }
            for panel in panels
        ]
        for partition, panels in features.items()
    }


def _validate_tensor_contract(features: dict, terminal_head: dict, binding: dict) -> None:
    if set(features) != {"seen", "fresh_held"} or any(
        len(features[name]) != 2 for name in features
    ):
        raise RuntimeError("consumed rescue feature partitions changed")
    if _feature_hashes(features) != binding["feature_sha256"]:
        raise RuntimeError("consumed rescue context or target tensor hash changed")
    if {
        name: foundation._tensor_sha256(value) for name, value in terminal_head.items()
    } != binding["terminal_ap_head_sha256"]:
        raise RuntimeError("consumed rescue terminal AP head hash changed")
    if terminal_head["weight"].shape != (41, 192) or terminal_head["bias"].shape != (41,):
        raise RuntimeError("consumed rescue terminal AP head shape changed")
    for panels in features.values():
        for panel in panels:
            if panel["context"].shape != (24, 192) or any(
                panel[name].shape != (24,)
                for name in (
                    "target_ap_bin",
                    "terminal_ap_residual_um",
                    "truth_ap_um",
                )
            ):
                raise RuntimeError("consumed rescue feature shape changed")
            if any(
                not bool(torch.isfinite(panel[name]).all())
                for name in ("context", "terminal_ap_residual_um", "truth_ap_um")
            ):
                raise RuntimeError("consumed rescue input contains nonfinite values")


def _load_bound_artifact(config: dict, source_run: Path):
    binding = config["source_probe"]
    receipt_path = source_run / "diagnostic_receipt.json"
    tensor_path = source_run / "ap_probe_tensors.pt"
    if foundation._binary_sha256(receipt_path) != binding["source_receipt_file_sha256"]:
        raise RuntimeError("consumed rescue receipt file hash changed")
    if foundation._binary_sha256(tensor_path) != binding["source_tensor_artifact_sha256"]:
        raise RuntimeError("consumed rescue tensor artifact hash changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if any(
        receipt.get(name) is not False
        for name in ("product5_access", "calibration_access", "final_test_access")
    ) or receipt.get("learned_candidate_promotion") is not False:
        raise RuntimeError("consumed rescue source protected-access flags changed")
    if (
        receipt.get("status") != "stop"
        or receipt.get("qualification", {}).get("classification")
        != "ap-head-solver-result-not-successful"
        or receipt.get("setup_sha256") != binding["source_setup_sha256"]
        or receipt.get("result_sha256") != binding["source_result_sha256"]
        or receipt.get("tensor_artifact_sha256")
        != binding["source_tensor_artifact_sha256"]
        or receipt.get("config", {}).get("contract_sha256")
        != binding["source_config_contract_sha256"]
        or receipt.get("config", {}).get("config_file_sha256")
        != binding["source_config_file_sha256"]
    ):
        raise RuntimeError("consumed rescue receipt contract changed")
    if (
        receipt.get("config", {}).get("fits") != source_probe.FIT_CONTRACT
        or receipt.get("config", {}).get("gates") != source_probe.GATE_CONTRACT
        or receipt.get("config", {}).get("solver") != source_probe.SOLVER_CONTRACT
        or receipt.get("tensor_sha256", {}).get("features")
        != binding["feature_sha256"]
        or receipt.get("tensor_sha256", {}).get("terminal_ap_head")
        != binding["terminal_ap_head_sha256"]
    ):
        raise RuntimeError("consumed rescue objective, split, gate, or tensor hashes changed")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if set(payload) != {"format", "setup_sha256", "terminal_ap_head", "contexts", "fits"}:
        raise RuntimeError("consumed rescue tensor payload fields changed")
    if payload["format"] != source_probe.FORMAT or payload["setup_sha256"] != binding[
        "source_setup_sha256"
    ]:
        raise RuntimeError("consumed rescue tensor payload identity changed")
    _validate_tensor_contract(payload["contexts"], payload["terminal_ap_head"], binding)
    return receipt, payload["contexts"], payload["terminal_ap_head"]


def _ce41_hessian_vector_product(
    vector: np.ndarray,
    direction: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    class_count: int,
    l2_coefficient: float,
) -> np.ndarray:
    del target
    feature_count = context.shape[1]
    split = class_count * feature_count
    weight = vector[:split].reshape(class_count, feature_count)
    bias = vector[split:]
    weight_direction = direction[:split].reshape(class_count, feature_count)
    bias_direction = direction[split:]
    logits = context @ weight.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    logit_direction = context @ weight_direction.T + bias_direction
    response = probability * (
        logit_direction
        - (probability * logit_direction).sum(axis=1, keepdims=True)
    )
    weight_result = response.T @ context / len(context) + l2_coefficient * weight_direction
    bias_result = response.sum(axis=0) / len(context) + l2_coefficient * bias_direction
    return np.concatenate((weight_result.ravel(), bias_result))


def _fit_ap_head_trust_ncg(
    context: torch.Tensor,
    target: torch.Tensor,
    initial_weight: torch.Tensor,
    initial_bias: torch.Tensor,
    solver: dict,
) -> dict:
    x = np.ascontiguousarray(context.detach().cpu().numpy(), dtype=np.float64)
    y = np.ascontiguousarray(target.detach().cpu().numpy(), dtype=np.int64)
    weight = np.ascontiguousarray(initial_weight.detach().cpu().numpy(), dtype=np.float64)
    bias = np.ascontiguousarray(initial_bias.detach().cpu().numpy(), dtype=np.float64)
    initial = np.concatenate((weight.ravel(), bias))
    arguments = (x, y, int(solver["class_count"]), float(solver["l2_coefficient"]))
    counters = {"n_fg": 0, "n_hvp": 0}
    accepted = {"x": initial.copy(), "iterations": 0}
    limit = int(solver["combined_fg_hvp_call_limit_each_fit"])

    def reserve(name):
        if counters["n_fg"] + counters["n_hvp"] >= limit:
            raise TrustNcgBudgetExhausted
        counters[name] += 1

    def objective(vector, *args):
        reserve("n_fg")
        return source_probe._ce41_objective_and_gradient(vector, *args)

    def hessian_product(vector, direction, *args):
        reserve("n_hvp")
        return _ce41_hessian_vector_product(vector, direction, *args)

    def callback(iterate):
        accepted["x"] = np.asarray(iterate, dtype=np.float64).copy()
        accepted["iterations"] += 1

    budget_exhausted = False
    try:
        result = minimize(
            objective,
            initial,
            args=arguments,
            method=solver["method"],
            jac=True,
            hessp=hessian_product,
            callback=callback,
            options={
                "gtol": float(solver["gtol"]),
                "maxiter": int(solver["maxiter"]),
                "initial_trust_radius": float(solver["initial_trust_radius"]),
                "max_trust_radius": float(solver["max_trust_radius"]),
                "eta": float(solver["eta"]),
                "disp": bool(solver["disp"]),
                "return_all": bool(solver["return_all"]),
            },
        )
        final = np.asarray(result.x, dtype=np.float64)
        success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
        iterations = int(result.nit)
        scipy_counts = {
            "scipy_nfev": int(result.nfev),
            "scipy_njev": int(result.njev),
            "scipy_nhev": int(result.nhev),
        }
    except TrustNcgBudgetExhausted:
        final = accepted["x"]
        success = False
        status = 4
        message = "STOP: PREREGISTERED COMBINED FG+HVP CALL LIMIT REACHED"
        iterations = int(accepted["iterations"])
        budget_exhausted = True
        scipy_counts = {"scipy_nfev": None, "scipy_njev": None, "scipy_nhev": None}

    final_loss, final_gradient = source_probe._ce41_objective_and_gradient(
        final, *arguments
    )
    split = int(solver["class_count"]) * x.shape[1]
    total_calls = counters["n_fg"] + counters["n_hvp"]
    return {
        "weight": torch.from_numpy(final[:split].reshape(weight.shape).copy()),
        "bias": torch.from_numpy(final[split:].copy()),
        "solver": {
            "success": success,
            "status": status,
            "message": message,
            "iterations": iterations,
            "function_evaluations": total_calls,
            "objective_gradient_evaluations": counters["n_fg"],
            "hessian_vector_evaluations": counters["n_hvp"],
            "independent_verification_function_evaluations": 1,
            "budget_call_limit": limit,
            "budget_exhausted": budget_exhausted,
            "final_objective": float(final_loss),
            "independent_final_gradient_inf_norm": float(np.abs(final_gradient).max()),
            "independent_final_gradient_l2_norm": float(np.linalg.norm(final_gradient)),
            "initial_vector_sha256": foundation._tensor_sha256(
                torch.from_numpy(initial.copy())
            ),
            "last_accepted_vector_sha256": foundation._tensor_sha256(
                torch.from_numpy(final.copy())
            ),
            **scipy_counts,
        },
    }


def _run_fits(config: dict, features: dict, terminal_head: dict) -> dict:
    initial_weight = terminal_head["weight"].detach().cpu().double()
    initial_bias = terminal_head["bias"].detach().cpu().double()
    ap_centers = torch.linspace(
        float(config["solver"]["ap_bin_center_minimum_um"]),
        float(config["solver"]["ap_bin_center_maximum_um"]),
        int(config["solver"]["ap_bin_center_count"]),
        dtype=torch.float32,
    )
    result = {}
    for specification in config["fits"]:
        train = source_probe._stack_panels(features, specification["train"])
        test = source_probe._stack_panels(features, specification["test"])
        fit = _fit_ap_head_trust_ncg(
            train["context"],
            train["target_ap_bin"],
            initial_weight,
            initial_bias,
            config["solver"],
        )
        fit["train"] = source_probe._prediction_payload(
            fit["weight"], fit["bias"], train, ap_centers
        )
        fit["test"] = source_probe._prediction_payload(
            fit["weight"], fit["bias"], test, ap_centers
        )
        result[specification["id"]] = fit
    return result


def qualification_status(config: dict, features: dict, fits: dict) -> dict:
    result = source_probe.qualification_status(config, features, fits)
    accounting_valid = all(
        all(
            isinstance(fit["solver"][name], int)
            and not isinstance(fit["solver"][name], bool)
            and fit["solver"][name] >= 0
            for name in (
                "objective_gradient_evaluations",
                "hessian_vector_evaluations",
                "function_evaluations",
            )
        )
        and fit["solver"]["function_evaluations"]
        == fit["solver"]["objective_gradient_evaluations"]
        + fit["solver"]["hessian_vector_evaluations"]
        and fit["solver"]["function_evaluations"]
        <= int(config["solver"]["combined_fg_hvp_call_limit_each_fit"])
        and fit["solver"]["budget_call_limit"]
        == int(config["solver"]["combined_fg_hvp_call_limit_each_fit"])
        for fit in fits.values()
    )
    within_cap = all(not fit["solver"]["budget_exhausted"] for fit in fits.values())
    iteration_counts_valid = all(
        isinstance(fit["solver"]["iterations"], int)
        and not isinstance(fit["solver"]["iterations"], bool)
        and 0 <= fit["solver"]["iterations"] <= int(config["solver"]["maxiter"])
        for fit in fits.values()
    )
    iteration_budget_not_exhausted = iteration_counts_valid and all(
        fit["solver"]["success"]
        or fit["solver"]["iterations"] < int(config["solver"]["maxiter"])
        for fit in fits.values()
    )
    result["checks"]["trust_ncg_operator_accounting_integrity"] = {
        "passed": accounting_valid
    }
    result["checks"]["trust_ncg_hard_cap_not_exhausted"] = {
        "passed": within_cap
    }
    result["checks"]["trust_ncg_iteration_budget_not_exhausted"] = {
        "passed": iteration_budget_not_exhausted
    }
    result["observed"]["trust_ncg_operator_calls"] = {
        name: {
            key: value["solver"][key]
            for key in (
                "objective_gradient_evaluations",
                "hessian_vector_evaluations",
                "function_evaluations",
                "budget_call_limit",
                "budget_exhausted",
            )
        }
        for name, value in fits.items()
    }
    if result["checks"]["finite"]["passed"]:
        if not accounting_valid:
            result["classification"] = "ap-head-operator-accounting-invalid"
        elif not within_cap:
            result["classification"] = "ap-head-preregistered-budget-exhausted"
        elif not iteration_budget_not_exhausted:
            result["classification"] = (
                "ap-head-preregistered-iteration-budget-exhausted"
            )
    result["decision"] = (
        "go"
        if result["decision"] == "go"
        and accounting_valid
        and within_cap
        and iteration_budget_not_exhausted
        else "stop"
    )
    solver_validity_checks = (
        "finite",
        "solver_result_success",
        "independent_gradient_inf_norm",
        "solver_iteration_budget",
        "solver_function_evaluation_budget",
        "optimizer_head_sample_presentation_budget",
        "independent_verification_sample_presentations",
        "total_head_sample_presentation_budget",
        "trust_ncg_operator_accounting_integrity",
        "trust_ncg_hard_cap_not_exhausted",
        "trust_ncg_iteration_budget_not_exhausted",
    )
    causal_validity = all(
        result["checks"][name]["passed"] for name in solver_validity_checks
    )
    result["causal_classification_allowed"] = causal_validity
    result["fresh_held_transfer_interpretation_allowed"] = causal_validity and all(
        result["checks"][name]["passed"]
        for name in (
            "single_panel_training_fit",
            "pooled_seen_training_fit",
            "opposite_seen_panel_transfer",
        )
    )
    result["learned_candidate_promotion_allowed"] = False
    return result


def run_rescue(path: str | Path) -> dict:
    config = load_rescue_config(path)
    environment = _validate_environment(config)
    source_run, output = _resolve_paths(config)
    if source_run.resolve() == output.resolve():
        raise RuntimeError("trust-NCG rescue source and output paths must differ")
    tensor_path = output / config["artifacts"]["tensor_payload"]
    receipt_path = output / config["artifacts"]["receipt"]
    if output.exists():
        raise RuntimeError("trust-NCG rescue output already exists")
    source_receipt, features, terminal_head = _load_bound_artifact(config, source_run)
    fits = _run_fits(config, features, terminal_head)
    qualification = qualification_status(config, features, fits)
    hashes = source_probe._tensor_hashes(features, fits, terminal_head)
    source_hashes = {
        relative: foundation._source_sha256(REPOSITORY_ROOT / relative)
        for relative in SOURCE_FILES
    }
    setup = {
        "version": 1,
        "purpose": PURPOSE,
        "role": "diagnostic-solver-rescue-not-model-selection",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_candidate_promotion": False,
        "config": config,
        "source_sha256": source_hashes,
        "source_probe": config["source_probe"],
        "diagnostic_artifact_dependencies": config[
            "diagnostic_artifact_dependencies"
        ],
        "learned_checkpoint_dependencies": config["learned_checkpoint_dependencies"],
        "source_probe_status": source_receipt["status"],
        "source_probe_qualification": source_receipt["qualification"],
        "tensor_sha256": hashes,
        "solver": config["solver"],
        "environment": environment,
    }
    setup["setup_sha256"] = foundation._canonical_sha256(setup)
    tensor_payload = {
        "format": FORMAT,
        "setup_sha256": setup["setup_sha256"],
        "source_tensor_artifact_sha256": config["source_probe"][
            "source_tensor_artifact_sha256"
        ],
        "terminal_ap_head": terminal_head,
        "fits": fits,
    }
    final_config = load_rescue_config(path)
    if (
        final_config["contract_sha256"] != config["contract_sha256"]
        or final_config["config_file_sha256"] != config["config_file_sha256"]
        or output.exists()
    ):
        raise RuntimeError("trust-NCG rescue lineage or output changed before write")
    source_probe._atomic_save(tensor_payload, tensor_path)
    receipt = {
        **setup,
        "status": qualification["decision"],
        "qualification": qualification,
        "prediction_interpretation": {
            name: (
                "debug-only-non-inferential-hard-cap-output"
                if fit["solver"]["budget_exhausted"]
                else "subject-to-recorded-qualification"
            )
            for name, fit in fits.items()
        },
        "raw_predictions": source_probe._raw_predictions(fits),
        "tensor_artifact": str(tensor_path),
        "tensor_artifact_sha256": foundation._binary_sha256(tensor_path),
    }
    receipt["result_sha256"] = foundation._canonical_sha256(receipt)
    foundation._atomic_json(receipt, receipt_path)
    return {
        "status": qualification["decision"],
        "qualification": qualification,
        "receipt_path": receipt_path,
        "tensor_path": tensor_path,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m training.run_frozen_context_ap_probe_trust_ncg_rescue CONFIG.json"
        )
    print(json.dumps(foundation._canonical(run_rescue(sys.argv[1])), indent=2))
