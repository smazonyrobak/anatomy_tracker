"""Frozen-context AP decoder convergence and generator-transfer diagnostic."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.optimize import minimize

import training.independent_joint_data as independent_data
import training.run_independent_initializer_foundation as foundation
import training.run_independent_pose_identifiability as pose_diagnostic
from training.synthetic_registration import SyntheticRegistrationGenerator
from training.train_independent_joint import _atomic_save


REPOSITORY_ROOT = Path(__file__).parents[1]
SCHEMA_VERSION = 1
PURPOSE = "development-only-frozen-context-ap-convergence-transfer-diagnostic"
FORMAT = "frozen-context-ap-probe-tensors-v1"
SOURCE_CHECKPOINT_COMMIT = "db6ed2163e1181dc8ddb1efdbee8461caa6d9286"
SOURCE_PANEL_CONTRACT_SHA256 = (
    "886d24c3e6993bc0beff01e5c4a1e75249b3e47bed1b192a848a3fa7e290583e"
)
SOURCE_FILES = (
    "training/independent_joint_model.py",
    "training/independent_joint_variants.py",
    "training/independent_joint_data.py",
    "training/train_independent_joint.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "training/run_independent_initializer_foundation.py",
    "training/run_independent_pose_identifiability.py",
    "training/run_frozen_context_ap_probe.py",
    "source/dense_registration_preprocessing.py",
)
ARM_CONTRACTS = {
    "global_average": {
        "output_name": "independent-oracle-frozen-context-ap-probe-fresh504322-v1",
        "model_class": (
            "training.independent_joint_variants.IndependentJointOracleSimilarityModel"
        ),
        "parameter_count": 1_369_070,
        "source_config_relative": (
            "training/configs/"
            "independent_pose_identifiability_oracle_similarity_300_r4322.json"
        ),
        "source_config_contract_sha256": (
            "3202f03d7214ee4767ccbbecedd6b5e1f1e18529de8453e284785cded8976d87"
        ),
        "source_config_file_sha256": (
            "4e9b517498ecb33abe35d9478fa453cc5b060be826624ad297e851406bfe0fa7"
        ),
        "source_run_name": (
            "independent-oracle-similarity-pose-identifiability-300-r4322-v1"
        ),
        "source_setup_sha256": (
            "54b0208e61919ef1cbb4786c2274d93d06a677d72273eb57e1f2fa253e0e8b1d"
        ),
        "source_resume_state_sha256": (
            "c2a355b079e59be28fb84b74c34cd3cf380e83788711cf2a52be4f23d8518aa9"
        ),
        "terminal_model_state_sha256": (
            "0efb78806303ec88b4465f3f9b19eb0898d214f5aba36b31829c353948061626"
        ),
        "terminal_named_parameter_sha256": (
            "63b5fb0acf439e4a259e1cfe061e4b4c3ff9cf7b8bed94db1bdbf5e125aa7252"
        ),
    },
    "spatial_moment": {
        "output_name": (
            "independent-spatial-moment-oracle-frozen-context-ap-probe-"
            "fresh504322-v1"
        ),
        "model_class": (
            "training.independent_joint_variants."
            "IndependentJointSpatialMomentOracleSimilarityModel"
        ),
        "parameter_count": 1_373_338,
        "source_config_relative": (
            "training/configs/"
            "independent_pose_identifiability_spatial_moment_oracle_similarity_"
            "300_r4322.json"
        ),
        "source_config_contract_sha256": (
            "4b4b0ffc8f545dbe7ce631b9c8c3f6a9e6160f8aa2a2858f90c55f0a08e7552d"
        ),
        "source_config_file_sha256": (
            "997cc7537c0d2d20cb5780f86bed033bce40fd118522b3ab49b76a64d01f0677"
        ),
        "source_run_name": (
            "independent-spatial-moment-oracle-similarity-pose-identifiability-"
            "300-r4322-v1"
        ),
        "source_setup_sha256": (
            "84f6c220aa120800d2933daff5fde8d047137da49da9695b48c3152ec15ef9c5"
        ),
        "source_resume_state_sha256": (
            "812b044b5d082aa0119c5c4d49c4dad9f6080a82b3819289164e2e7ea773d615"
        ),
        "terminal_model_state_sha256": (
            "061b0bd978e00005c57a5c6c125d6481f47420f152af6cf81017edd6cf8f4363"
        ),
        "terminal_named_parameter_sha256": (
            "1ffd3d67f135c225a5f76bcdc78d4af28aa14928c6b8a4828c82833b73a6d38b"
        ),
    },
}
DATA_CONTRACT = {
    "source_seen_generator_seed": 304322,
    "consumed_descriptive_held_generator_seed": 404322,
    "primary_fresh_held_generator_seed": 504322,
    "latent_pose_count": 24,
    "seen_panels": 2,
    "fresh_held_panels": 2,
    "outline_mode": "absent",
    "source_view_intervention": "exact-oracle-forward-sampling-once",
    "fresh_held_nuisance_policy": (
        "reuse-frozen-404322-held-rotation-scale-values-and-pair-assignments"
    ),
    "consumed_404322_use": "excluded-from-fitting-and-primary-transfer-evaluation",
    "expected_manifest_sha256": {
        "seen": {
            "generator": (
                "b7520c0039fc0bd6dc90dbe52fe85b30b3c11a67f16e57b5cdf0841985f7870b"
            ),
            "panels": [
                "a55104400404b055186c00aea9004a3acdbfe8482be38f9f50749c7084c2eda6",
                "575d97a5d9a07d9e99095030754825450fdeedc00d5b2b8bf3df19cf22a5c734",
            ],
            "outlines": [
                "404d9d259aa40169012ae401184899508ccf32bfdaaa4b703b35c712715cfd01",
                "cd9d8e871eda2c738b2d203efbc7573e68fb52a40e26487c8f4c3ec5cbc32343",
            ],
        },
        "fresh_held": {
            "generator": (
                "1311ed79a32982a31cedf464c906eca1addab083cd916d1b334e125388ea418f"
            ),
            "panels": [
                "826726adf1fa1827af94ae29237589c691a8f83c29646190db1a274d66189bbc",
                "e86857da57e0b03e3b9d480ae66f61b9985114a4f465b2c7f98109375434a580",
            ],
            "outlines": [
                "e2fad680bf4b766aa2201c6a88c7c7e378071cd81b72167109c121972ec27c7e",
                "eb1f6f086f1e971cae40f2e0f5e3211e220c2cad83f6cfd1365d151dd8048c70",
            ],
        },
    },
}
FIT_CONTRACT = [
    {
        "id": "seen_panel_0_to_1",
        "train": [{"partition": "seen", "panel": 0}],
        "test": [{"partition": "seen", "panel": 1}],
    },
    {
        "id": "seen_panel_1_to_0",
        "train": [{"partition": "seen", "panel": 1}],
        "test": [{"partition": "seen", "panel": 0}],
    },
    {
        "id": "seen_to_fresh_held",
        "train": [
            {"partition": "seen", "panel": 0},
            {"partition": "seen", "panel": 1},
        ],
        "test": [
            {"partition": "fresh_held", "panel": 0},
            {"partition": "fresh_held", "panel": 1},
        ],
    },
]
SOLVER_CONTRACT = {
    "implementation": "scipy.optimize.minimize",
    "method": "L-BFGS-B",
    "jacobian": "analytic",
    "dtype": "float64-cpu",
    "initialization": "terminal-ap-head-weight-and-bias",
    "objective": "mean-cross-entropy-41-plus-half-1e-4-l2-on-weight-and-bias",
    "class_count": 41,
    "l2_coefficient": 0.0001,
    "maxiter": 1000,
    "maxfun": 1250,
    "gtol": 1e-8,
    "ftol": 1e-10,
    "maxls": 20,
    "maxcor": 10,
    "maximum_optimizer_objective_sample_presentations_per_arm": 120000,
    "independent_verification_objective_sample_presentations_per_arm": 96,
    "maximum_total_objective_sample_presentations_per_arm": 120096,
    "held_access_during_fit": False,
    "feature_normalization": None,
    "early_stopping": None,
}
GATE_CONTRACT = {
    "nonfinite_count_maximum": 0,
    "solver_result_success_required": True,
    "independent_final_gradient_inf_norm_maximum": 1e-8,
    "solver_iterations_maximum_each_fit": 1000,
    "solver_function_evaluations_maximum_each_fit": 1250,
    "optimizer_head_sample_presentations_maximum_per_arm": 120000,
    "independent_verification_sample_presentations_per_arm": 96,
    "total_head_sample_presentations_maximum_per_arm": 120096,
    "single_panel_training_correct_required": 24,
    "pooled_seen_training_correct_required": 48,
    "opposite_seen_panel_correct_minimum": 23,
    "opposite_seen_panel_sample_count": 24,
    "fresh_held_bin_center_ap_mae_um_maximum": 250.0,
    "fresh_held_prediction_to_truth_sd_ratio_minimum": 0.75,
}


def inspect_ap_probe_config(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = raw.pop("contract_sha256")
    if foundation._canonical_sha256(raw) != contract:
        raise ValueError("AP-probe config hash differs from its payload")
    expected_top_level = {
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
        "source_checkpoint_commit",
        "source_panel_contract_sha256",
        "source_checkpoint",
        "learned_checkpoint_dependencies",
        "paths",
        "feature_extraction",
        "data",
        "fits",
        "solver",
        "gates",
        "artifacts",
        "lineage",
    }
    if set(raw) != expected_top_level:
        raise ValueError("AP-probe config top-level fields changed")
    arm = raw.get("arm")
    if arm not in ARM_CONTRACTS:
        raise ValueError("AP-probe arm is not allowlisted")
    source = ARM_CONTRACTS[arm]
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("frozen") is not True:
        raise ValueError("AP-probe config is not frozen schema v1")
    if raw.get("purpose") != PURPOSE or raw.get("role") != "diagnostic-not-model-selection":
        raise ValueError("AP-probe purpose or role changed")
    if raw.get("name") != source["output_name"]:
        raise ValueError("AP-probe name changed")
    if any(
        raw.get(name) is not False
        for name in ("product5_access", "calibration_access", "final_test_access")
    ):
        raise ValueError("AP probe must remain synthetic-development-only")
    if raw.get("learned_candidate_promotion") is not False:
        raise ValueError("AP probe cannot promote a learned candidate")
    if raw.get("source_checkpoint_commit") != SOURCE_CHECKPOINT_COMMIT:
        raise ValueError("AP-probe source checkpoint commit changed")
    if raw.get("source_panel_contract_sha256") != SOURCE_PANEL_CONTRACT_SHA256:
        raise ValueError("AP-probe source panel contract changed")
    expected_checkpoint = {
        name: value for name, value in source.items() if name != "output_name"
    }
    if raw.get("source_checkpoint") != expected_checkpoint:
        raise ValueError("AP-probe source checkpoint binding changed")
    if raw.get("learned_checkpoint_dependencies") != [
        {
            "role": "diagnostic-terminal-context-source-only",
            "run_name": source["source_run_name"],
            "resume_state_sha256": source["source_resume_state_sha256"],
            "terminal_model_state_sha256": source["terminal_model_state_sha256"],
        }
    ]:
        raise ValueError("AP-probe checkpoint dependency changed")
    if raw.get("paths") != {
        "atlas_repo_relative": "data/Allen Brain Atlas 25um",
        "run_root_env": "ATLAS_JOINT_RUN_ROOT",
    }:
        raise ValueError("AP-probe paths changed")
    if raw.get("feature_extraction") != {
        "device": "cuda",
        "model_mode": "eval",
        "autocast": False,
        "model_parameters_trainable": False,
        "pose_context_features": 192,
        "model_input": "oracle-source-image-zero-mask-zero-availability",
        "anatomical_pose_label_input_to_model": False,
    }:
        raise ValueError("AP-probe feature extraction changed")
    if raw.get("data") != DATA_CONTRACT:
        raise ValueError("AP-probe data contract changed")
    if raw.get("fits") != FIT_CONTRACT:
        raise ValueError("AP-probe fit contract changed")
    if raw.get("solver") != SOLVER_CONTRACT:
        raise ValueError("AP-probe solver contract changed")
    if raw.get("gates") != GATE_CONTRACT:
        raise ValueError("AP-probe gates changed")
    if raw.get("artifacts") != {
        "tensor_payload": "ap_probe_tensors.pt",
        "receipt": "diagnostic_receipt.json",
        "raw_predictions": True,
        "save_contexts": True,
        "save_logits": True,
        "save_fitted_heads": True,
    }:
        raise ValueError("AP-probe artifact contract changed")
    lineage = raw.get("lineage", {}).get("source_sha256", {})
    if set(lineage) != set(SOURCE_FILES):
        raise ValueError("AP-probe source lineage is incomplete")
    raw["contract_sha256"] = contract
    raw["config_file_sha256"] = foundation._source_sha256(path)
    return raw


def load_ap_probe_config(path: str | Path) -> dict:
    config = inspect_ap_probe_config(path)
    for relative, expected in config["lineage"]["source_sha256"].items():
        if foundation._source_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"source lineage changed: {relative}")
    return config


def _resolve_paths(config: dict) -> tuple[Path, Path, Path]:
    atlas = (REPOSITORY_ROOT / config["paths"]["atlas_repo_relative"]).resolve()
    try:
        run_root = Path(os.environ[config["paths"]["run_root_env"]]).resolve()
    except KeyError as error:
        raise RuntimeError("ATLAS_JOINT_RUN_ROOT is required") from error
    source_run = run_root / config["source_checkpoint"]["source_run_name"]
    return atlas, source_run, run_root / config["name"]


def _seed_model_construction(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_bound_source(config: dict, source_run: Path, device: torch.device):
    binding = config["source_checkpoint"]
    source_config_path = REPOSITORY_ROOT / binding["source_config_relative"]
    if foundation._source_sha256(source_config_path) != binding["source_config_file_sha256"]:
        raise RuntimeError("source oracle config file hash changed")
    source_config = pose_diagnostic.load_pose_identifiability_config(source_config_path)
    if source_config["contract_sha256"] != binding["source_config_contract_sha256"]:
        raise RuntimeError("source oracle config contract changed")
    if source_config["model"]["class"] != binding["model_class"]:
        raise RuntimeError("source oracle model class changed")

    receipt_path = source_run / "diagnostic_receipt.json"
    state_path = source_run / "resume_state.pt"
    if foundation._binary_sha256(state_path) != binding["source_resume_state_sha256"]:
        raise RuntimeError("source resume artifact hash changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("resume_state_sha256") != binding["source_resume_state_sha256"]:
        raise RuntimeError("source receipt and resume artifact disagree")
    if receipt.get("setup_sha256") != binding["source_setup_sha256"]:
        raise RuntimeError("source setup hash changed")
    if receipt.get("status") != "stop" or receipt.get("progress") != {
        "optimizer_updates": 300,
        "sample_presentations": 7200,
    }:
        raise RuntimeError("source diagnostic is not the bound terminal state")
    if any(
        receipt.get(name) is not False
        for name in ("product5_access", "calibration_access", "final_test_access")
    ) or receipt.get("learned_checkpoint_dependencies") != []:
        raise RuntimeError("source diagnostic protected-access contract changed")
    if (
        receipt.get("fixed_panel_contract", {}).get("contract_sha256")
        != SOURCE_PANEL_CONTRACT_SHA256
    ):
        raise RuntimeError("source fixed-panel contract changed")
    oracle = receipt["fixed_panel_contract"]["oracle_source_view_canonicalization"]
    if any(
        oracle.get(name) != value
        for name, value in {
            "parameter_mismatch_count": 0,
            "canonicalized_nonfinite_count": 0,
            "fixed_panel_count": 4,
            "observed_warps_per_fixed_panel": 1,
        }.items()
    ):
        raise RuntimeError("source oracle integrity contract failed")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        state.get("format") != pose_diagnostic.FORMAT
        or state.get("status") != "stop"
        or state.get("update") != 300
        or state.get("learned_checkpoint_dependencies") != []
    ):
        raise RuntimeError("source resume payload changed")
    _seed_model_construction(int(source_config["seed"]))
    model = pose_diagnostic._model_contract(source_config)["factory"](
        **source_config["model"]["kwargs"]
    ).to(device)
    if sum(value.numel() for value in model.parameters()) != binding["parameter_count"]:
        raise RuntimeError("source model parameter count changed")
    model.load_state_dict(state["model"])
    if foundation._state_sha256(model) != binding["terminal_model_state_sha256"]:
        raise RuntimeError("terminal model state hash changed")
    parameter_names = [name for name, _ in model.named_parameters()]
    if (
        foundation._named_parameter_sha256(model, parameter_names)
        != binding["terminal_named_parameter_sha256"]
    ):
        raise RuntimeError("terminal ordered parameter hash changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return source_config, receipt, model


def _manifest_hashes(manifests: dict) -> dict:
    return {
        "seen": {
            "generator": manifests["seen"][0]["generator_manifest"]["manifest_sha256"],
            "panels": [value["manifest_sha256"] for value in manifests["seen"]],
            "outlines": [value["outline_plan"]["plan_sha256"] for value in manifests["seen"]],
        },
        "fresh_held": {
            "generator": manifests["held"][0]["generator_manifest"]["manifest_sha256"],
            "panels": [value["manifest_sha256"] for value in manifests["held"]],
            "outlines": [value["outline_plan"]["plan_sha256"] for value in manifests["held"]],
        },
    }


def _prepare_probe_panels(config, source_config, source_receipt, generator, synthetic, device):
    generation_config = copy.deepcopy(source_config)
    generation_config["data"]["held_base_seed"] = int(
        config["data"]["primary_fresh_held_generator_seed"]
    )
    manifests = pose_diagnostic.fixed_panel_manifests(synthetic, generation_config)
    observed_manifests = _manifest_hashes(manifests)
    if observed_manifests != config["data"]["expected_manifest_sha256"]:
        raise RuntimeError("predeclared AP-probe manifests changed")
    source_panel = source_receipt["fixed_panel_contract"]
    if observed_manifests["fresh_held"]["generator"] in set(
        source_panel["generator_realization_sha256"].values()
    ):
        raise RuntimeError("fresh-held generator realization is not disjoint")
    if not set(observed_manifests["fresh_held"]["panels"]).isdisjoint(
        manifest
        for values in source_panel["manifests"].values()
        for manifest in values
    ):
        raise RuntimeError("fresh-held panel manifests are not disjoint")

    panel_contract, panels, _ = pose_diagnostic._prepare_fixed_panels(
        generation_config, synthetic, generator
    )
    panel_contract, panels = pose_diagnostic._prepare_oracle_source_view_panels(
        panel_contract, panels, device
    )
    if panel_contract["input_sha256"]["seen"] != source_panel["input_sha256"]["seen"]:
        raise RuntimeError("regenerated seen raw panels differ from the bound source panels")
    if (
        panel_contract["oracle_source_view_canonicalization"]
        ["canonicalized_source_image_sha256"]["seen"]
        != source_panel["oracle_source_view_canonicalization"]
        ["canonicalized_source_image_sha256"]["seen"]
    ):
        raise RuntimeError("regenerated seen oracle panels differ from the bound source panels")
    if panel_contract["manifests"]["seen"] != observed_manifests["seen"]["panels"]:
        raise RuntimeError("seen panel manifest lineage changed")
    if panel_contract["manifests"]["held"] != observed_manifests["fresh_held"]["panels"]:
        raise RuntimeError("fresh-held panel manifest lineage changed")
    panels = {"seen": panels["seen"], "fresh_held": panels["held"]}
    return panel_contract, panels, observed_manifests


def _extract_contexts(model, panels, device: torch.device) -> dict:
    before = foundation._state_sha256(model)
    result = {"seen": [], "fresh_held": []}
    centers = model.pose_head.ap_centers.detach().cpu()
    with torch.inference_mode():
        for partition in result:
            for panel in panels[partition]:
                if bool(torch.count_nonzero(panel["source_mask"])) or bool(
                    torch.count_nonzero(panel["mask_available"])
                ):
                    raise RuntimeError(
                        "AP probe requires zero outline mask and availability"
                    )
                output = model.initialize(
                    panel["oracle_source_image"].to(device),
                    panel["source_mask"].to(device),
                    panel["mask_available"].to(device),
                )
                context = output["pose_context"].detach().cpu().contiguous()
                truth_ap = panel["true_pose"][:, 0].detach().cpu().contiguous()
                target_bin = (truth_ap[:, None] - centers[None]).abs().argmin(1)
                result[partition].append(
                    {
                        "context": context,
                        "truth_ap_um": truth_ap,
                        "target_ap_bin": target_bin,
                        "terminal_ap_residual_um": output["continuous_residual"][:, 0]
                        .detach()
                        .cpu()
                        .contiguous(),
                        "manifest_sha256": panel["manifest_sha256"],
                        "oracle_source_image_sha256": panel[
                            "oracle_source_image_sha256"
                        ],
                    }
                )
    after = foundation._state_sha256(model)
    if before != after:
        raise RuntimeError("frozen source model changed during context extraction")
    return result


def _input_integrity(panels: dict) -> dict:
    mask_nonzero = sum(
        int(torch.count_nonzero(panel["source_mask"]))
        for values in panels.values()
        for panel in values
    )
    availability_nonzero = sum(
        int(torch.count_nonzero(panel["mask_available"]))
        for values in panels.values()
        for panel in values
    )
    if mask_nonzero or availability_nonzero:
        raise RuntimeError("AP-probe absent-outline input integrity failed")
    return {
        "source_mask_nonzero_count": mask_nonzero,
        "mask_available_nonzero_count": availability_nonzero,
    }


def _ce41_objective_and_gradient(
    vector: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    class_count: int,
    l2_coefficient: float,
) -> tuple[float, np.ndarray]:
    feature_count = context.shape[1]
    split = class_count * feature_count
    weight = vector[:split].reshape(class_count, feature_count)
    bias = vector[split:]
    logits = context @ weight.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probability = exponential / exponential.sum(axis=1, keepdims=True)
    loss = (
        np.log(exponential.sum(axis=1)) - logits[np.arange(len(target)), target]
    ).mean()
    loss += 0.5 * l2_coefficient * (
        np.square(weight).sum() + np.square(bias).sum()
    )
    probability[np.arange(len(target)), target] -= 1.0
    probability /= len(target)
    weight_gradient = probability.T @ context + l2_coefficient * weight
    bias_gradient = probability.sum(axis=0) + l2_coefficient * bias
    gradient = np.concatenate((weight_gradient.ravel(), bias_gradient))
    return float(loss), gradient


def _fit_ap_head(
    context: torch.Tensor,
    target: torch.Tensor,
    initial_weight: torch.Tensor,
    initial_bias: torch.Tensor,
    solver: dict,
) -> dict:
    x = context.detach().cpu().numpy().astype(np.float64, copy=False)
    y = target.detach().cpu().numpy().astype(np.int64, copy=False)
    weight = initial_weight.detach().cpu().numpy().astype(np.float64, copy=True)
    bias = initial_bias.detach().cpu().numpy().astype(np.float64, copy=True)
    initial = np.concatenate((weight.ravel(), bias))
    arguments = (x, y, int(solver["class_count"]), float(solver["l2_coefficient"]))
    result = minimize(
        _ce41_objective_and_gradient,
        initial,
        args=arguments,
        method=solver["method"],
        jac=True,
        options={
            "maxiter": int(solver["maxiter"]),
            "maxfun": int(solver["maxfun"]),
            "gtol": float(solver["gtol"]),
            "ftol": float(solver["ftol"]),
            "maxls": int(solver["maxls"]),
            "maxcor": int(solver["maxcor"]),
        },
    )
    final_loss, final_gradient = _ce41_objective_and_gradient(
        result.x, *arguments
    )
    split = int(solver["class_count"]) * x.shape[1]
    return {
        "weight": torch.from_numpy(result.x[:split].reshape(weight.shape).copy()),
        "bias": torch.from_numpy(result.x[split:].copy()),
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "independent_verification_function_evaluations": 1,
            "final_objective": float(final_loss),
            "independent_final_gradient_inf_norm": float(
                np.abs(final_gradient).max()
            ),
        },
    }


def _stack_panels(features: dict, specifications: list[dict]) -> dict:
    values = [features[item["partition"]][int(item["panel"])] for item in specifications]
    result = {
        name: torch.cat([value[name] for value in values])
        for name in (
            "context",
            "truth_ap_um",
            "target_ap_bin",
            "terminal_ap_residual_um",
        )
    }
    result["sample_provenance"] = [
        {
            "partition": item["partition"],
            "panel_index": int(item["panel"]),
            "latent_pose_index": latent,
            "manifest_sha256": features[item["partition"]][int(item["panel"])][
                "manifest_sha256"
            ],
        }
        for item in specifications
        for latent in range(
            len(features[item["partition"]][int(item["panel"])]["context"])
        )
    ]
    return result


def _prediction_payload(
    weight: torch.Tensor,
    bias: torch.Tensor,
    batch: dict,
    ap_centers: torch.Tensor,
) -> dict:
    context = batch["context"].double()
    logits = context @ weight.T + bias
    predicted_bin = logits.argmax(1)
    bin_center = ap_centers.double().index_select(0, predicted_bin)
    residual_added = torch.clamp(
        bin_center + batch["terminal_ap_residual_um"].double(), -4500.0, 500.0
    )
    truth = batch["truth_ap_um"].double()
    return {
        "logits": logits,
        "predicted_bin": predicted_bin,
        "target_bin": batch["target_ap_bin"].long(),
        "truth_ap_um": truth,
        "bin_center_ap_um": bin_center,
        "terminal_residual_added_ap_um": residual_added,
        "correct_count": int((predicted_bin == batch["target_ap_bin"]).sum()),
        "sample_count": len(truth),
        "bin_center_ap_mae_um": float((bin_center - truth).abs().mean()),
        "terminal_residual_added_ap_mae_um": float(
            (residual_added - truth).abs().mean()
        ),
        "prediction_sd_um": float(bin_center.std(unbiased=False)),
        "truth_sd_um": float(truth.std(unbiased=False)),
        "sample_provenance": batch["sample_provenance"],
    }


def _run_fits(config: dict, model, features: dict) -> dict:
    initial_weight = model.pose_head.ap_logits.weight.detach().cpu().double()
    initial_bias = model.pose_head.ap_logits.bias.detach().cpu().double()
    ap_centers = model.pose_head.ap_centers.detach().cpu()
    result = {}
    for specification in config["fits"]:
        train = _stack_panels(features, specification["train"])
        test = _stack_panels(features, specification["test"])
        fit = _fit_ap_head(
            train["context"],
            train["target_ap_bin"],
            initial_weight,
            initial_bias,
            config["solver"],
        )
        fit["train"] = _prediction_payload(
            fit["weight"], fit["bias"], train, ap_centers
        )
        fit["test"] = _prediction_payload(
            fit["weight"], fit["bias"], test, ap_centers
        )
        result[specification["id"]] = fit
    return result


def _nonfinite_count(features: dict, fits: dict) -> int:
    tensors = []
    for panels in features.values():
        for panel in panels:
            tensors.extend(
                panel[name]
                for name in (
                    "context",
                    "truth_ap_um",
                    "terminal_ap_residual_um",
                )
            )
    for fit in fits.values():
        tensors.extend((fit["weight"], fit["bias"]))
        if not math.isfinite(fit["solver"]["final_objective"]):
            tensors.append(torch.tensor(float("nan")))
        if not math.isfinite(
            fit["solver"]["independent_final_gradient_inf_norm"]
        ):
            tensors.append(torch.tensor(float("nan")))
        for partition in ("train", "test"):
            tensors.extend(
                fit[partition][name]
                for name in (
                    "logits",
                    "truth_ap_um",
                    "bin_center_ap_um",
                    "terminal_residual_added_ap_um",
                )
            )
    return sum(int(torch.count_nonzero(~torch.isfinite(value))) for value in tensors)


def qualification_status(config: dict, features: dict, fits: dict) -> dict:
    gates = config["gates"]
    nonfinite = _nonfinite_count(features, fits)
    fold_0 = fits["seen_panel_0_to_1"]
    fold_1 = fits["seen_panel_1_to_0"]
    transfer = fits["seen_to_fresh_held"]
    held = transfer["test"]
    held_ratio = held["prediction_sd_um"] / held["truth_sd_um"]
    optimizer_sample_presentations = sum(
        value["solver"]["function_evaluations"] * value["train"]["sample_count"]
        for value in fits.values()
    )
    verification_sample_presentations = sum(
        value["solver"]["independent_verification_function_evaluations"]
        * value["train"]["sample_count"]
        for value in fits.values()
    )
    total_sample_presentations = (
        optimizer_sample_presentations + verification_sample_presentations
    )
    checks = {
        "finite": nonfinite <= int(gates["nonfinite_count_maximum"]),
        "solver_result_success": all(
            value["solver"]["success"] for value in fits.values()
        ),
        "independent_gradient_inf_norm": all(
            value["solver"]["independent_final_gradient_inf_norm"]
            <= float(gates["independent_final_gradient_inf_norm_maximum"])
            for value in fits.values()
        ),
        "solver_iteration_budget": all(
            value["solver"]["iterations"]
            <= int(gates["solver_iterations_maximum_each_fit"])
            for value in fits.values()
        ),
        "solver_function_evaluation_budget": all(
            value["solver"]["function_evaluations"]
            <= int(gates["solver_function_evaluations_maximum_each_fit"])
            for value in fits.values()
        ),
        "optimizer_head_sample_presentation_budget": (
            optimizer_sample_presentations
            <= int(gates["optimizer_head_sample_presentations_maximum_per_arm"])
        ),
        "independent_verification_sample_presentations": (
            verification_sample_presentations
            == int(gates["independent_verification_sample_presentations_per_arm"])
        ),
        "total_head_sample_presentation_budget": (
            total_sample_presentations
            <= int(gates["total_head_sample_presentations_maximum_per_arm"])
        ),
        "single_panel_training_fit": all(
            value["train"]["correct_count"]
            == value["train"]["sample_count"]
            == int(gates["single_panel_training_correct_required"])
            for value in (fold_0, fold_1)
        ),
        "pooled_seen_training_fit": (
            transfer["train"]["correct_count"]
            == transfer["train"]["sample_count"]
            == int(gates["pooled_seen_training_correct_required"])
        ),
        "opposite_seen_panel_transfer": all(
            value["test"]["correct_count"]
            >= int(gates["opposite_seen_panel_correct_minimum"])
            and value["test"]["sample_count"]
            == int(gates["opposite_seen_panel_sample_count"])
            for value in (fold_0, fold_1)
        ),
        "fresh_held_bin_center_ap_mae": (
            held["bin_center_ap_mae_um"]
            <= float(gates["fresh_held_bin_center_ap_mae_um_maximum"])
        ),
        "fresh_held_prediction_sd": (
            held_ratio
            >= float(gates["fresh_held_prediction_to_truth_sd_ratio_minimum"])
        ),
    }
    if not checks["finite"]:
        classification = "numerical-failure"
    elif not checks["solver_result_success"]:
        classification = "ap-head-solver-result-not-successful"
    elif not checks["independent_gradient_inf_norm"]:
        classification = "ap-head-independent-gradient-convergence-failed"
    elif not (
        checks["solver_iteration_budget"]
        and checks["solver_function_evaluation_budget"]
        and checks["optimizer_head_sample_presentation_budget"]
        and checks["independent_verification_sample_presentations"]
        and checks["total_head_sample_presentation_budget"]
    ):
        classification = "ap-head-preregistered-budget-exceeded"
    elif not (
        checks["single_panel_training_fit"] and checks["pooled_seen_training_fit"]
    ):
        classification = "preregistered-regularized-ap-head-training-fit-insufficient"
    elif not checks["opposite_seen_panel_transfer"]:
        classification = "terminal-ap-context-panel-resampling-instability"
    elif not (
        checks["fresh_held_bin_center_ap_mae"]
        and checks["fresh_held_prediction_sd"]
    ):
        classification = "terminal-ap-context-generator-generalization-insufficient"
    else:
        classification = "joint-ap-head-optimization-underconverged"
    return {
        "decision": "go" if all(checks.values()) else "stop",
        "classification": classification,
        "checks": {name: {"passed": value} for name, value in checks.items()},
        "observed": {
            "nonfinite_count": nonfinite,
            "single_panel_training_correct": [
                fold_0["train"]["correct_count"],
                fold_1["train"]["correct_count"],
            ],
            "pooled_seen_training_correct": transfer["train"]["correct_count"],
            "opposite_seen_panel_correct": [
                fold_0["test"]["correct_count"],
                fold_1["test"]["correct_count"],
            ],
            "fresh_held_bin_center_ap_mae_um": held["bin_center_ap_mae_um"],
            "fresh_held_terminal_residual_added_ap_mae_um": held[
                "terminal_residual_added_ap_mae_um"
            ],
            "fresh_held_prediction_to_truth_sd_ratio": held_ratio,
            "solver": {
                name: value["solver"] for name, value in fits.items()
            },
            "optimizer_head_sample_presentations": optimizer_sample_presentations,
            "independent_verification_sample_presentations": (
                verification_sample_presentations
            ),
            "total_head_sample_presentations": total_sample_presentations,
        },
    }


def _tensor_hashes(features: dict, fits: dict, initial_ap_head: dict) -> dict:
    return {
        "terminal_ap_head": {
            name: foundation._tensor_sha256(value)
            for name, value in initial_ap_head.items()
        },
        "features": {
            partition: [
                {
                    name: foundation._tensor_sha256(panel[name])
                    for name in (
                        "context",
                        "truth_ap_um",
                        "target_ap_bin",
                        "terminal_ap_residual_um",
                    )
                }
                for panel in panels
            ]
            for partition, panels in features.items()
        },
        "fits": {
            name: {
                "weight": foundation._tensor_sha256(value["weight"]),
                "bias": foundation._tensor_sha256(value["bias"]),
                "train_logits": foundation._tensor_sha256(value["train"]["logits"]),
                "test_logits": foundation._tensor_sha256(value["test"]["logits"]),
                "train_prediction": foundation._tensor_sha256(
                    value["train"]["predicted_bin"]
                ),
                "test_prediction": foundation._tensor_sha256(
                    value["test"]["predicted_bin"]
                ),
            }
            for name, value in fits.items()
        },
    }


def _raw_predictions(fits: dict) -> dict:
    result = {}
    for name, fit in fits.items():
        result[name] = {}
        for role in ("train", "test"):
            value = fit[role]
            records = []
            for index in range(value["sample_count"]):
                record = {
                    "fit": name,
                    "role": role,
                    "sample_index": index,
                    **value["sample_provenance"][index],
                    "true_ap_um": float(value["truth_ap_um"][index]),
                    "target_ap_bin": int(value["target_bin"][index]),
                    "predicted_ap_bin": int(value["predicted_bin"][index]),
                    "bin_center_ap_um": float(value["bin_center_ap_um"][index]),
                    "terminal_residual_added_ap_um": float(
                        value["terminal_residual_added_ap_um"][index]
                    ),
                }
                record["record_sha256"] = foundation._canonical_sha256(record)
                records.append(record)
            result[name][role] = records
    return result


def run_frozen_context_ap_probe(path: str | Path) -> dict:
    config = load_ap_probe_config(path)
    if not torch.cuda.is_available():
        raise RuntimeError("bound oracle context extraction requires CUDA")
    device = torch.device("cuda")
    atlas, source_run, output = _resolve_paths(config)
    tensor_path = output / config["artifacts"]["tensor_payload"]
    receipt_path = output / config["artifacts"]["receipt"]
    if tensor_path.exists() or receipt_path.exists():
        raise RuntimeError("AP-probe output already exists")

    source_config, source_receipt, model = _load_bound_source(
        config, source_run, device
    )
    generator = SyntheticRegistrationGenerator(atlas, device=device)
    synthetic = independent_data.IndependentSyntheticData(generator)
    panel_contract, panels, manifest_hashes = _prepare_probe_panels(
        config, source_config, source_receipt, generator, synthetic, device
    )
    input_integrity = _input_integrity(panels)
    features = _extract_contexts(model, panels, device)
    initial_ap_head = {
        "weight": model.pose_head.ap_logits.weight.detach().cpu().double(),
        "bias": model.pose_head.ap_logits.bias.detach().cpu().double(),
    }
    fits = _run_fits(config, model, features)
    qualification = qualification_status(config, features, fits)
    hashes = _tensor_hashes(features, fits, initial_ap_head)
    source_hashes = {
        relative: foundation._source_sha256(REPOSITORY_ROOT / relative)
        for relative in SOURCE_FILES
    }
    setup = {
        "version": 1,
        "purpose": PURPOSE,
        "role": "diagnostic-not-model-selection",
        "product5_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "learned_candidate_promotion": False,
        "config": config,
        "source_sha256": source_hashes,
        "source_checkpoint_commit": SOURCE_CHECKPOINT_COMMIT,
        "source_checkpoint": config["source_checkpoint"],
        "source_panel_contract_sha256": SOURCE_PANEL_CONTRACT_SHA256,
        "fresh_panel_contract": panel_contract,
        "manifest_sha256": manifest_hashes,
        "input_integrity": input_integrity,
        "tensor_sha256": hashes,
        "feature_extraction": {
            "device": str(device),
            "model_mode": "eval",
            "model_parameters_trainable": False,
            "model_state_before_and_after_sha256": config["source_checkpoint"][
                "terminal_model_state_sha256"
            ],
            "anatomical_pose_label_input_to_model": False,
        },
        "solver": config["solver"],
        "execution_environment": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    setup["setup_sha256"] = foundation._canonical_sha256(setup)
    tensor_payload = {
        "format": FORMAT,
        "setup_sha256": setup["setup_sha256"],
        "terminal_ap_head": initial_ap_head,
        "contexts": features,
        "fits": fits,
    }
    _atomic_save(tensor_payload, tensor_path)
    receipt = {
        **setup,
        "status": qualification["decision"],
        "qualification": qualification,
        "raw_predictions": _raw_predictions(fits),
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
            "usage: python -m training.run_frozen_context_ap_probe CONFIG.json"
        )
    print(
        json.dumps(
            foundation._canonical(run_frozen_context_ap_probe(sys.argv[1])), indent=2
        )
    )
