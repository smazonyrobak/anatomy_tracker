from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.independent_joint_model import IndependentJointModel
from training.independent_joint_variants import (
    FactorizedCNNControl,
    RecurrentAttentionVariant,
)
from training.preflight_independent_architectures import (
    FIXED_WORKLOAD,
    MASK_CURRICULUM,
    PREFLIGHT_PROTOCOL,
    _fixed_batch,
    _fixed_forward,
    build_model,
    load_frozen_config,
    profile_model,
)


CONFIG_FOLDER = Path(__file__).parents[1] / "training" / "configs"
CONFIG_PATHS = (
    CONFIG_FOLDER / "independent_architecture_preflight_leader.json",
    CONFIG_FOLDER / "independent_architecture_preflight_factorized_t3.json",
    CONFIG_FOLDER / "independent_architecture_preflight_attention_t3.json",
)


def _tiny_workload() -> dict:
    return {
        **FIXED_WORKLOAD,
        "candidates_per_source": 2,
        "height": 32,
        "width": 40,
        "warmup_iterations": 0,
        "measured_iterations": 1,
        "amp": False,
        "onnx_provider": "CPUExecutionProvider",
    }


def _tiny_models():
    common = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
        "integration_steps": 3,
    }
    return (
        IndependentJointModel(**common),
        FactorizedCNNControl(**common, fusion_channels=16),
        RecurrentAttentionVariant(**common, attention_channels=4),
    )


def test_three_configs_are_hash_bound_cold_start_and_workload_identical():
    configurations = [load_frozen_config(path) for path in CONFIG_PATHS]
    models = [build_model(config) for config in configurations]
    assert {type(model) for model in models} == {
        IndependentJointModel,
        FactorizedCNNControl,
        RecurrentAttentionVariant,
    }
    assert all(config["workload"] == FIXED_WORKLOAD for config in configurations)
    assert all(config["mask_curriculum"] == MASK_CURRICULUM for config in configurations)
    assert all(
        config["training_protocol"] == PREFLIGHT_PROTOCOL
        for config in configurations
    )
    assert all(config["architecture"]["recurrent_steps"] == 3 for config in configurations)
    assert all(config["contract_sha256"] != "0" * 64 for config in configurations)
    assert all(config["config_file_sha256"] for config in configurations)
    assert all(model.learned_weight_dependencies == () for model in models)
    assert all(model.initialization == "random" for model in models)


def test_frozen_config_rejects_payload_tampering(tmp_path):
    payload = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    payload["workload"]["candidates_per_source"] += 1
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_frozen_config(altered)


def test_fixed_batch_exercises_all_mask_modes_and_three_update_contract():
    workload = _tiny_workload()
    batch = _fixed_batch(workload, torch.device("cpu"))
    assert torch.equal(
        batch["mask_available"].flatten(), torch.tensor([1.0, 1.0, 0.0])
    )
    assert torch.count_nonzero(batch["source_mask"][0]) > 0
    assert torch.count_nonzero(batch["source_mask"][1]) > 0
    assert torch.count_nonzero(batch["source_mask"][2]) == 0
    assert torch.count_nonzero(batch["source_image"][2]) > 0

    for model in _tiny_models():
        with torch.no_grad():
            outputs, loss = _fixed_forward(model.eval(), batch, recurrent_steps=3)
        assert len(outputs["recurrent"]) == 3
        assert outputs["ranking"]["pose"].shape == (6, 3)
        assert outputs["dense"]["fixed_to_moving_map"].shape == (3, 2, 32, 40)
        assert torch.isfinite(loss)


@pytest.mark.parametrize("model", _tiny_models())
def test_tiny_cpu_profiler_records_fixed_workflow_and_onnx_provider(model):
    result = profile_model(
        model,
        _tiny_workload(),
        device="cpu",
        onnx_provider="CPUExecutionProvider",
    )
    assert result["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert result["mac_proxy"] > 0
    assert "excludes elementwise" in result["mac_proxy_scope"]
    assert result["peak_vram_bytes"] == 0
    assert result["forward_median_seconds"] > 0.0
    assert result["backward_median_seconds"] > 0.0
    assert len(result["initial_state_sha256"]) == 64
    assert result["onnx"]["checker_passed"] is True
    assert result["onnx"]["runtime_passed"] is True
    assert result["onnx"]["requested_provider"] == "CPUExecutionProvider"
    assert result["onnx"]["max_abs_error"] <= 5e-3


def test_harness_contains_no_optimizer_or_checkpoint_training_path():
    source = (
        Path(__file__).parents[1]
        / "training"
        / "preflight_independent_architectures.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "torch.load" not in source
    assert "torch.save" not in source
