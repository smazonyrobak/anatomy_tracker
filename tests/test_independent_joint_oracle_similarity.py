from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.independent_joint_data as independent_data
import training.run_independent_pose_identifiability as diagnostic
from source.dense_registration_preprocessing import MODEL_SHAPE
from training.independent_joint_model import IndependentJointModel, StructuralPyramid
from training.independent_joint_variants import (
    IndependentJointOracleSimilarityModel,
    IndependentJointSimilarityCanonicalizedModel,
    IndependentJointSpatialMomentModel,
    IndependentJointSpatialMomentOracleSimilarityModel,
    IndependentJointSpatialMomentSimilarityCanonicalizedModel,
    SupervisedSimilarityCanonicalizer,
)


ROOT = Path(__file__).parents[1]
BASE_CONFIG = (
    ROOT
    / "training/configs/independent_pose_identifiability_oracle_similarity_300_r4322.json"
)
MOMENT_CONFIG = (
    ROOT
    / "training/configs/independent_pose_identifiability_spatial_moment_oracle_similarity_300_r4322.json"
)
BASE_CONFIG_SHA256 = "4e9b517498ecb33abe35d9478fa453cc5b060be826624ad297e851406bfe0fa7"
MOMENT_CONFIG_SHA256 = "997cc7537c0d2d20cb5780f86bed033bce40fd118522b3ab49b76a64d01f0677"


def _small(model_class):
    return model_class(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def _source(batch=2):
    image = torch.linspace(0.0, 1.0, batch * 32 * 40).reshape(batch, 1, 32, 40)
    return (
        image,
        torch.zeros_like(image, dtype=torch.bool),
        torch.zeros(batch, 1, 1, 1),
    )


def test_oracle_marker_models_preserve_default_abi_state_and_parameter_boundary():
    assert inspect.signature(IndependentJointOracleSimilarityModel.initialize) == inspect.signature(
        IndependentJointModel.initialize
    )
    assert inspect.signature(
        IndependentJointSpatialMomentOracleSimilarityModel.initialize
    ) == inspect.signature(IndependentJointSpatialMomentModel.initialize)
    pairs = (
        (IndependentJointModel, IndependentJointOracleSimilarityModel),
        (
            IndependentJointSpatialMomentModel,
            IndependentJointSpatialMomentOracleSimilarityModel,
        ),
    )
    for reference_class, oracle_class in pairs:
        torch.manual_seed(4322)
        reference = reference_class()
        torch.manual_seed(4322)
        oracle = oracle_class()
        assert all(
            torch.equal(value, oracle.state_dict()[name])
            for name, value in reference.state_dict().items()
        )
        assert not hasattr(oracle, "source_view_canonicalizer")
        assert set(oracle.initialize(*_source())) == set(reference.initialize(*_source()))
        assert diagnostic._model_contract(
            {
                "model": {
                    "class": f"training.independent_joint_variants.{oracle_class.__name__}"
                }
            }
        )["oracle_source_view"]
    for oracle_class, supervised_class in (
        (
            IndependentJointOracleSimilarityModel,
            IndependentJointSimilarityCanonicalizedModel,
        ),
        (
            IndependentJointSpatialMomentOracleSimilarityModel,
            IndependentJointSpatialMomentSimilarityCanonicalizedModel,
        ),
    ):
        torch.manual_seed(4322)
        oracle = oracle_class()
        torch.manual_seed(4322)
        supervised = supervised_class()
        assert all(
            torch.equal(value, supervised.state_dict()[name])
            for name, value in oracle.state_dict().items()
        )


def _marker_panel():
    height, width = MODEL_SHAPE
    rotation = torch.tensor([27.0])
    scale = torch.tensor([0.85])
    marker = torch.zeros(1, 1, height, width)
    marker[:, :, 72:112, 91:151] = 0.65
    marker[:, :, 112:151, 91:111] = 1.0
    marker[:, :, 83:95, 139:181] = 0.35
    false_mask = torch.zeros_like(marker, dtype=torch.bool)
    identity = independent_data._identity_map(1, torch.device("cpu"))
    pair = {
        "moving": marker,
        "moving_tissue_mask": false_mask,
        "moving_damage_mask": false_mask,
        "moving_visible_mask": torch.ones_like(false_mask),
        "moving_brush_mask": false_mask,
        "moving_labels": torch.zeros_like(marker, dtype=torch.long),
        "moving_to_fixed": identity,
        "fixed_to_moving": identity,
        "fixed_visible_mask": torch.ones_like(false_mask),
        "similarity_h": torch.eye(3)[None],
    }
    manifest = {
        "source_view_rotation_deg": rotation.numpy(),
        "source_view_scale": scale.numpy(),
        "outline_plan": {
            "mode": [2],
            "sample_receipt_sha256": ["marker"],
            "plan_sha256": "marker-plan",
        },
    }
    viewed = independent_data._apply_source_view(pair, manifest, marker, false_mask)
    panel = {
        "source_image": viewed["source_image"],
        "source_mask": viewed["source_mask"],
        "mask_available": viewed["mask_available"],
        "truth_source_view_parameters": viewed["truth_source_view_parameters"],
        "source_view_rotation_deg": rotation.numpy(),
        "source_view_scale": scale.numpy(),
    }
    return marker, panel, rotation, scale


def test_oracle_prewarp_uses_exact_forward_view_once_without_pose_or_rng(monkeypatch):
    marker, panel, rotation, scale = _marker_panel()
    original = SupervisedSimilarityCanonicalizer.warp_with_parameters
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        SupervisedSimilarityCanonicalizer, "warp_with_parameters", counted
    )
    rng_before = torch.get_rng_state().clone()
    contract, panels = diagnostic._prepare_oracle_source_view_panels(
        {"contract_sha256": "r" * 64}, {"seen": [panel], "held": []}, torch.device("cpu")
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert len(calls) == 1
    restored = panels["seen"][0]["oracle_source_image"]
    assert (restored - marker).abs().mean() < 0.002

    planes = StructuralPyramid._input(
        panel["source_image"], panel["source_mask"], panel["mask_available"]
    )
    expected = original(planes, rotation, torch.log(scale))[:, :1]
    assert torch.equal(restored, expected)
    wrong = original(panel["source_image"], -rotation, -torch.log(scale))
    assert (restored - marker).abs().mean() < (wrong - marker).abs().mean() * 0.2
    twice = original(restored, rotation, torch.log(scale))
    assert not torch.allclose(twice, restored)
    assert (twice - marker).abs().mean() > (restored - marker).abs().mean() * 5

    oracle = contract["oracle_source_view_canonicalization"]
    assert oracle["parameter_mismatch_count"] == 0
    assert oracle["canonicalized_nonfinite_count"] == 0
    assert oracle["canonicalization_warps_per_fixed_panel"] == 1
    assert oracle["observed_canonicalization_warp_count"] == 1
    assert oracle["observed_warps_per_fixed_panel"] == 1
    assert oracle["canonicalization_sampling_direction"] == (
        "observed-source-sampled-with-forward-source-view-h-output-to-input"
    )
    assert "true_pose" not in panel
    assert torch.count_nonzero(panel["source_mask"]) == 0
    assert torch.count_nonzero(panel["mask_available"]) == 0


def test_oracle_identity_is_noop_and_available_outline_fails_closed():
    image, mask, available = _source(1)
    panel = {
        "source_image": image,
        "source_mask": mask,
        "mask_available": available,
        "truth_source_view_parameters": torch.tensor([[0.0, 1.0]]),
        "source_view_rotation_deg": np.asarray([0.0], np.float32),
        "source_view_scale": np.asarray([1.0], np.float32),
    }
    _, panels = diagnostic._prepare_oracle_source_view_panels(
        {"contract_sha256": "r" * 64},
        {"seen": [copy.deepcopy(panel)], "held": []},
        torch.device("cpu"),
    )
    assert torch.allclose(
        panels["seen"][0]["oracle_source_image"], image, atol=5e-6, rtol=0.0
    )
    panel["source_mask"][0, 0, 1, 1] = True
    panel["mask_available"].fill_(1.0)
    with pytest.raises(RuntimeError, match="requires absent outlines"):
        diagnostic._prepare_oracle_source_view_panels(
            {"contract_sha256": "r" * 64},
            {"seen": [panel], "held": []},
            torch.device("cpu"),
        )


def test_frozen_oracle_configs_are_strict_matched_cold_start_contracts():
    assert hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest() == BASE_CONFIG_SHA256
    assert hashlib.sha256(MOMENT_CONFIG.read_bytes()).hexdigest() == MOMENT_CONFIG_SHA256
    base = diagnostic.load_pose_identifiability_config(BASE_CONFIG)
    moment = diagnostic.load_pose_identifiability_config(MOMENT_CONFIG)
    for name in (
        "schema_version",
        "frozen",
        "purpose",
        "role",
        "product5_access",
        "calibration_access",
        "final_test_access",
        "learned_checkpoint_dependencies",
        "seed",
        "device",
        "paths",
        "data",
        "training",
        "evaluation",
        "gates",
    ):
        assert base[name] == moment[name]
    assert base["training"]["loss_weights"] == {
        "categorical": 1.0,
        "sub_bin_residual": 0.5,
    }
    assert base["training"]["oracle_source_view_contract"] == (
        diagnostic.ORACLE_SOURCE_VIEW_CONTRACT
    )
    assert "source_view_supervision_contract" not in base["training"]
    assert not any(
        name.startswith("seen_source_view_") or name.startswith("held_source_view_")
        for name in base["gates"]
    )
    assert not base["product5_access"]
    assert not base["calibration_access"]
    assert not base["final_test_access"]
    assert base["learned_checkpoint_dependencies"] == []
    assert base["model"]["expected_parameter_count"] == 1_369_070
    assert moment["model"]["expected_parameter_count"] == 1_373_338
    assert base["lineage"]["source_sha256"] == moment["lineage"]["source_sha256"]
    assert "training/independent_joint_variants.py" in base["lineage"]["source_sha256"]
    assert hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest() == base[
        "config_file_sha256"
    ]
    assert hashlib.sha256(MOMENT_CONFIG.read_bytes()).hexdigest() == moment[
        "config_file_sha256"
    ]


def _passing_evaluation():
    truth_sd = torch.tensor([1400.0, 12.5, 17.5], dtype=torch.float64)
    return {
        "seen": {
            "bin_accuracy": torch.tensor([0.95, 0.90, 0.90], dtype=torch.float64),
            "residual_improvement_over_zero": torch.tensor(
                [0.20, 0.20, 0.20], dtype=torch.float64
            ),
        },
        "held": {
            "mae": torch.tensor([250.0, 3.0, 3.0], dtype=torch.float64),
            "prediction_sd": truth_sd * 0.75,
            "truth_sd": truth_sd,
            "physical_improvement_over_constant_prior": 0.50,
            "residual_improvement_over_zero": torch.tensor(
                [0.20, 0.20, 0.20], dtype=torch.float64
            ),
        },
        "oracle_source_view_canonicalization": {
            **diagnostic.ORACLE_SOURCE_VIEW_CONTRACT,
            "parameter_mismatch_count": 0,
            "canonicalized_nonfinite_count": 0,
            "fixed_panel_count": 4,
            "observed_warps_per_fixed_panel": 1,
        },
        "nonfinite_output_count": 0,
    }


def test_oracle_integrity_precedes_conditional_pose_and_generator_classification():
    config = diagnostic.load_pose_identifiability_config(BASE_CONFIG)
    gradients = [{"update": value, "clipped": False} for value in range(1, 301)]
    evaluation = _passing_evaluation()
    passed = diagnostic.qualification_status(evaluation, gradients, 0, config)
    assert passed["decision"] == "go"
    assert passed["classification"] == (
        "pose-identifiability-and-held-generator-generalization-demonstrated-"
        "conditional-on-oracle-source-view-canonicalization"
    )

    evaluation["held"]["mae"][0] = 251.0
    held = diagnostic.qualification_status(evaluation, gradients, 0, config)
    assert "held-generator-generalization-insufficient" in held["classification"]
    assert "invariance" not in held["classification"]

    evaluation["oracle_source_view_canonicalization"]["parameter_mismatch_count"] = 1
    integrity = diagnostic.qualification_status(evaluation, gradients, 0, config)
    assert integrity["classification"] == (
        "oracle-source-view-canonicalization-integrity-failed"
    )


class _FakeGenerator:
    def __init__(self, atlas_folder, device):
        self.device = torch.device(device)
        self.contract = {"contract_sha256": "a" * 64}
        self.annotation = torch.ones(2, 2, 2, dtype=torch.int16)


class _FakeSynthetic:
    def __init__(self, generator):
        self.contract = {"contract_sha256": "d" * 64}


def _fake_panels(config, synthetic, generator):
    truth = torch.from_numpy(diagnostic.latent_pose_table(config))
    transforms = diagnostic.nuisance_transform_tables(config)
    source = torch.linspace(0.0, 1.0, 24 * 32 * 40).reshape(24, 1, 32, 40)
    panels = {"seen": [], "held": []}
    for kind in panels:
        for index in range(2):
            nuisance = torch.from_numpy(transforms[kind][index]).float()
            panels[kind].append(
                {
                    "source_image": torch.roll(
                        source, index + int(kind == "held"), dims=-1
                    ),
                    "source_mask": torch.zeros(24, 1, 32, 40, dtype=torch.bool),
                    "mask_available": torch.zeros(24, 1, 1, 1),
                    "true_pose": truth.clone(),
                    "truth_source_view_parameters": nuisance,
                    "source_view_rotation_deg": nuisance[:, 0].numpy(),
                    "source_view_scale": nuisance[:, 1].numpy(),
                    "manifest_sha256": f"{kind}-{index}",
                    "generator_manifest_sha256": "g" * 64,
                    "outline_plan_sha256": "o" * 64,
                    "data_contract_sha256": synthetic.contract["contract_sha256"],
                }
            )
    return {"contract_sha256": "p" * 64}, panels, torch.ones(24, 7, 9, dtype=torch.bool)


def test_matched_oracle_arms_precompute_bit_exact_fixed_inputs():
    configs = [
        diagnostic.load_pose_identifiability_config(BASE_CONFIG),
        diagnostic.load_pose_identifiability_config(MOMENT_CONFIG),
    ]
    results = []
    for config in configs:
        generator = _FakeGenerator(None, "cpu")
        synthetic = _FakeSynthetic(generator)
        contract, panels, _ = _fake_panels(config, synthetic, generator)
        contract, panels = diagnostic._prepare_oracle_source_view_panels(
            contract, panels, torch.device("cpu")
        )
        results.append((contract, panels))
    assert results[0][0]["oracle_source_view_canonicalization"] == results[1][0][
        "oracle_source_view_canonicalization"
    ]
    assert all(
        torch.equal(
            results[0][1][kind][panel]["oracle_source_image"],
            results[1][1][kind][panel]["oracle_source_image"],
        )
        for kind in ("seen", "held")
        for panel in range(2)
    )


def test_oracle_runner_pause_resume_records_prewarp_without_nuisance_loss(
    tmp_path, monkeypatch
):
    config = copy.deepcopy(diagnostic.load_pose_identifiability_config(BASE_CONFIG))
    config["device"] = "cpu"
    config["name"] = "oracle-resume-smoke"
    config["training"]["amp"] = False
    config["training"]["max_updates"] = 3
    config["training"]["gradient_clip_warmup_updates"] = 0
    config["training"]["resume_state_every_updates"] = 1
    config["model"]["kwargs"] = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
        "integration_steps": 3,
    }
    config["model"]["expected_parameter_count"] = sum(
        value.numel() for value in _small(IndependentJointOracleSimilarityModel).parameters()
    )
    monkeypatch.setattr(
        diagnostic, "load_pose_identifiability_config", lambda path: copy.deepcopy(config)
    )
    monkeypatch.setattr(diagnostic, "SyntheticRegistrationGenerator", _FakeGenerator)
    monkeypatch.setattr(
        diagnostic.independent_data, "IndependentSyntheticData", _FakeSynthetic
    )
    monkeypatch.setattr(diagnostic, "_prepare_fixed_panels", _fake_panels)
    model_source_calls = []
    original_panel_source = diagnostic._panel_source_image

    def recorded_panel_source(panel, oracle_source_view):
        model_source_calls.append(oracle_source_view)
        value = original_panel_source(panel, oracle_source_view)
        assert value is panel["oracle_source_image"]
        return value

    monkeypatch.setattr(diagnostic, "_panel_source_image", recorded_panel_source)
    monkeypatch.setattr(
        diagnostic,
        "_resolve_paths",
        lambda loaded: (tmp_path / "atlas", tmp_path / "run"),
    )

    first = diagnostic.run_pose_identifiability(BASE_CONFIG, max_updates_this_call=1)
    second = diagnostic.run_pose_identifiability(BASE_CONFIG, max_updates_this_call=1)
    assert first["status"] == second["status"] == "paused"
    assert first["updates"] == 1
    assert second["updates"] == 2
    receipt = json.loads(second["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["source_view_supervision"]["enabled"] is False
    oracle = receipt["oracle_source_view_canonicalization"]
    assert oracle["enabled"] is True
    assert oracle["model_receives_nuisance_parameters"] is False
    assert oracle["model_receives_anatomical_pose_target"] is False
    assert oracle["nuisance_loss"] is False
    assert oracle["learned_parameters"] is False
    assert oracle["contract"]["fixed_panel_count"] == 4
    assert oracle["contract"]["parameter_mismatch_count"] == 0
    assert all(
        len(value) == 2
        for value in oracle["contract"]["canonicalized_source_image_sha256"].values()
    )
    assert receipt["progress"] == {
        "optimizer_updates": 2,
        "sample_presentations": 48,
    }
    assert len(receipt["gradient_records"]) == 2
    assert all(
        not any("source_view" in name or "canonicalizer" in name for name in record)
        for record in receipt["gradient_records"]
    )

    third = diagnostic.run_pose_identifiability(BASE_CONFIG, max_updates_this_call=1)
    assert third["updates"] == 3
    assert third["status"] in {"go", "stop"}
    terminal = json.loads(third["receipt_path"].read_text(encoding="utf-8"))
    assert terminal["evaluation"]["oracle_source_view_canonicalization"] == oracle[
        "contract"
    ]
    assert all(
        "oracle_source_image_sha256" in record
        for kind in ("seen", "held")
        for record in terminal["evaluation"][kind]["raw_predictions"]
    )
    assert model_source_calls == [True] * 7
