from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.independent_joint_data as independent_data
import training.run_independent_initializer_foundation as foundation
from training.independent_joint_model import IndependentJointModel
from training.quicknii_plane_metric import torch_brain_masked_plane_distance
from training.train_independent_joint import initializer_pose_losses


CONFIG = (
    Path(__file__).parents[1]
    / "training"
    / "configs"
    / "independent_initializer_foundation_2500_r4322.json"
)


class FakeSynthetic:
    contract = {"contract_sha256": "f" * 64, "learned_checkpoint_dependencies": []}

    def __init__(self, generator=None):
        self.generator = generator

    def make_manifest(
        self, count, split, seed, stratum, negatives_per_sample=1, *, pose_regime="standard"
    ):
        true_pose = np.column_stack(
            (
                np.linspace(-2200.0, -1800.0, count, dtype=np.float32),
                np.linspace(-4.0, 4.0, count, dtype=np.float32),
                np.linspace(3.0, -3.0, count, dtype=np.float32),
            )
        )
        manifest = {
            "version": 1,
            "contract_sha256": self.contract["contract_sha256"],
            "split": split,
            "seed": int(seed),
            "stratum": stratum,
            "pose_regime": pose_regime,
            "sample_count": int(count),
            "negative_count": int(negatives_per_sample),
            "true_pose": true_pose,
            "source_view_rotation_deg": np.zeros(count, np.float32),
            "source_view_scale": np.ones(count, np.float32),
            "outline_plan": independent_data._outline_plan(count, seed, "fake"),
            "generator_manifest": {"manifest_sha256": f"generator-{seed}"},
        }
        manifest["manifest_sha256"] = independent_data._payload_sha256(manifest)
        return manifest

    def batch(self, manifest):
        count = int(manifest["sample_count"])
        generator = torch.Generator().manual_seed(int(manifest["seed"]))
        image = torch.rand(count, 1, 32, 40, generator=generator)
        mode = torch.as_tensor(manifest["outline_plan"]["mode"], dtype=torch.int8)
        available = (mode != 2)[:, None, None, None].float()
        outline = torch.ones_like(image) * available
        return {
            "source_type": "synthetic_ccf",
            "data_split": manifest["split"],
            "pose_regime": manifest["pose_regime"],
            "data_contract_sha256": self.contract["contract_sha256"],
            "sample_manifest_sha256": manifest["manifest_sha256"],
            "source_image": image,
            "source_mask": outline.bool(),
            "mask_available": available,
            "input_outline_mode": mode,
            "input_outline_receipt_sha256": manifest["outline_plan"][
                "sample_receipt_sha256"
            ],
            "true_pose": torch.from_numpy(manifest["true_pose"]),
        }


def _rehash_config(payload):
    payload = copy.deepcopy(payload)
    payload.pop("contract_sha256", None)
    payload["contract_sha256"] = foundation._canonical_sha256(payload)
    return payload


def test_frozen_foundation_config_is_source_bound_synthetic_only_and_not_selection(tmp_path):
    config = foundation.load_foundation_config(CONFIG)
    assert config["role"] == "diagnostic-not-architecture-selection"
    assert config["learned_checkpoint_dependencies"] == []
    assert not config["product5_access"]
    assert not config["calibration_access"]
    assert not config["final_test_access"]
    assert config["development"]["evaluation_views"] == [0, 500, 1000, 1500, 2000, 2500]
    source = (Path(__file__).parents[1] / "training" / "run_independent_initializer_foundation.py").read_text()
    assert "IndependentProduct5Data" not in source
    assert "atlas_pose_models_v7" not in source
    assert "independent_joint_forward" not in source

    raw = json.loads(CONFIG.read_text())
    raw["gates"]["final_overall_physical_reduction_minimum"] = 0.34
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(_rehash_config(raw)))
    with pytest.raises(ValueError, match="gates changed"):
        foundation.load_foundation_config(changed)

    raw = json.loads(CONFIG.read_text())
    raw["lineage"]["source_sha256"]["training/independent_joint_model.py"] = "0" * 64
    changed.write_text(json.dumps(_rehash_config(raw)))
    with pytest.raises(ValueError, match="source lineage changed"):
        foundation.load_foundation_config(changed)

    raw = json.loads(CONFIG.read_text())
    raw["training"]["amp"] = False
    changed.write_text(json.dumps(_rehash_config(raw)))
    with pytest.raises(ValueError, match="loss, AMP, checkpoint, or resume contract changed"):
        foundation.load_foundation_config(changed)


def test_global_view_schedule_is_exact_unique_and_has_no_batch_two_absent_bug():
    config = foundation.load_foundation_config(CONFIG)
    data = config["data"]
    global_plan = independent_data._outline_plan(
        2500, data["outline_seed"], "initializer-foundation-global"
    )
    assert global_plan["mode_counts"].tolist() == [875, 875, 750]
    assert independent_data._outline_plan(2, 1, "old-per-batch")["mode_counts"].tolist() == [1, 1, 0]

    synthetic = FakeSynthetic()
    modes, strata, identities = [], [], []
    for update in range(1250):
        manifest = foundation.training_manifest(synthetic, config, update, global_plan)
        assert manifest["split"] == "train"
        assert manifest["pose_regime"] == "standard"
        modes.extend(manifest["outline_plan"]["mode"].tolist())
        strata.extend([manifest["stratum"]] * 2)
        identities.extend(
            record["view_content_sha256"]
            for record in foundation._view_receipts(manifest, update)
        )
        for row, limits in enumerate(manifest["foundation_source_view_limits"]):
            assert abs(manifest["source_view_rotation_deg"][row]) <= limits["rotation_abs_max_deg"]
            assert limits["scale_minimum"] <= manifest["source_view_scale"][row] <= limits["scale_maximum"]
    assert np.bincount(modes, minlength=3).tolist() == [875, 875, 750]
    assert strata.count("clean") == 1750
    assert strata.count("mild") == 750
    assert len(identities) == len(set(identities)) == 2500

    assert foundation.nuisance_limits(0, 2500) == foundation.nuisance_limits(1249, 2500) == {
        "rotation_abs_max_deg": 30.0,
        "scale_minimum": 0.8,
        "scale_maximum": 1.2,
        "second_half_progress": 0.0,
    }
    assert foundation.nuisance_limits(1250, 2500)["second_half_progress"] == 0.0
    final_limits = foundation.nuisance_limits(2499, 2500)
    assert final_limits["rotation_abs_max_deg"] == pytest.approx(90.0)
    assert final_limits["scale_minimum"] == pytest.approx(0.7)
    assert final_limits["scale_maximum"] == pytest.approx(1.3)
    assert final_limits["second_half_progress"] == 1.0
    assert foundation._learning_rate(0, 250, 2e-4) == pytest.approx(8e-7)
    assert foundation._learning_rate(249, 250, 2e-4) == pytest.approx(2e-4)
    assert foundation._learning_rate(900, 250, 2e-4) == pytest.approx(2e-4)
    assert foundation._gaussian_weight(0, 1250, 0.01, 0.05) == pytest.approx(0.01)
    assert foundation._gaussian_weight(1249, 1250, 0.01, 0.05) == pytest.approx(0.05)


def _tiny_model():
    torch.manual_seed(3)
    return IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def test_initializer_step_freezes_and_never_calls_atlas_pair_recurrent_or_dense(monkeypatch):
    model = _tiny_model().train()
    encoder, head = foundation._initializer_parameter_groups(model)
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    frozen_before = foundation._named_parameter_sha256(model, frozen_names)

    def forbidden(*args, **kwargs):
        raise AssertionError("non-initializer module was called")

    for module in (
        model.pyramid.atlas_stem,
        model.pair_projection,
        model.condition,
        model.recurrent,
        model.pose_delta_head,
        model.similarity_head,
        model.compatibility_head,
        model.decoder,
    ):
        monkeypatch.setattr(module, "forward", forbidden)

    batch = FakeSynthetic().batch(
        foundation.training_manifest(
            FakeSynthetic(),
            foundation.load_foundation_config(CONFIG),
            0,
            independent_data._outline_plan(2500, 204322, "initializer-foundation-global"),
        )
    )
    output = model.initialize(
        batch["source_image"], batch["source_mask"], batch["mask_available"]
    )
    components = initializer_pose_losses(output, batch["true_pose"], model)
    loss = (
        components["initializer_categorical"]
        + 0.5 * components["initializer_sub_bin"]
        + 0.01 * components["initializer_gaussian_nll"]
        + 0.25 * components["initializer_plane_anchor"]
    )
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in encoder + head)
    assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_ema_names = sorted(set(model.state_dict()) - set(trainable_names))
    frozen_ema_before = foundation._state_subset_sha256(ema, frozen_ema_names)
    optimizer = torch.optim.AdamW(
        ({"params": encoder}, {"params": head}), lr=2e-4, weight_decay=1e-4
    )
    config = foundation.load_foundation_config(CONFIG)
    foundation._validate_optimizer_contract(optimizer, encoder, head, config)
    optimizer.param_groups[0]["weight_decay"] = 0.0
    with pytest.raises(RuntimeError, match="optimizer hyperparameters"):
        foundation._validate_optimizer_contract(optimizer, encoder, head, config)
    optimizer.param_groups[0]["weight_decay"] = 1e-4
    optimizer.step()
    foundation._update_initializer_ema(ema, model, 0.99, trainable_names)
    assert foundation._named_parameter_sha256(model, frozen_names) == frozen_before
    assert foundation._state_subset_sha256(ema, frozen_ema_names) == frozen_ema_before
    assert any(
        not torch.equal(ema[name], model.state_dict()[name]) for name in trainable_names
    )


def _panel(views, physical, ap, lr, dv, absent=None, nonfinite=0):
    absent = physical if absent is None else absent
    return {
        "fresh_checkpoint_views": views,
        "nonfinite_output_count": nonfinite,
        "overall": {
            "physical_corresponding_plane_error_um": physical,
            "ap_mae_um": ap,
            "lr_mae_deg": lr,
            "dv_mae_deg": dv,
        },
        "by_outline_mode": {
            "absent": {"physical_corresponding_plane_error_um": absent}
        },
    }


def test_physical_metric_and_stop_go_gates_are_exact_and_fail_closed():
    truth = torch.tensor([[-2000.0, 0.0, 0.0]], dtype=torch.float64)
    shifted = truth + torch.tensor([[25.0, 0.0, 0.0]], dtype=torch.float64)
    mask = torch.ones(1, 7, 9, dtype=torch.bool)
    same = torch_brain_masked_plane_distance(
        foundation._pose_to_quicknii_ouv(truth), foundation._pose_to_quicknii_ouv(truth), mask
    ) * 25.0
    distance = torch_brain_masked_plane_distance(
        foundation._pose_to_quicknii_ouv(truth), foundation._pose_to_quicknii_ouv(shifted), mask
    ) * 25.0
    assert same.item() == pytest.approx(0.0)
    assert distance.item() == pytest.approx(25.0)

    config = foundation.load_foundation_config(CONFIG)
    evaluations = [
        _panel(0, 100.0, 100.0, 10.0, 5.0, absent=120.0),
        _panel(1000, 85.0, 90.0, 9.0, 4.5, absent=110.0),
        _panel(2500, 65.0, 80.0, 8.0, 4.0, absent=90.0),
    ]
    gradients = []
    for index, factor in enumerate((0.1, 0.1, 0.1, 1.0)):
        gradients.append(
            {
                "views_after": 502 + 2 * index,
                "encoder": {"clip_factor": factor},
                "head": {"clip_factor": factor},
            }
        )
    result = foundation.qualification_status(evaluations, gradients, 0, config)
    assert result["decision"] == "go"
    assert all(value["passed"] for value in result["checks"].values())

    failed = copy.deepcopy(evaluations)
    failed[1]["overall"]["physical_corresponding_plane_error_um"] = 85.01
    assert foundation.qualification_status(failed[:2], gradients, 0, config)["decision"] == "stop"
    assert foundation.qualification_status(evaluations, gradients, 1, config)["decision"] == "stop"
    missing_baseline = foundation.qualification_status(evaluations[1:], gradients, 0, config)
    assert missing_baseline["decision"] == "stop"
    assert missing_baseline["checks"]["interim_overall_physical_reduction"]["observed"] is None
    missing_interim = foundation.qualification_status(
        [evaluations[0], evaluations[2]], gradients, 0, config
    )
    assert missing_interim["decision"] == "stop"
    assert not missing_interim["checks"]["interim_overall_physical_reduction"]["passed"]


def test_nonfinite_development_output_is_serializable_and_stops_at_baseline():
    batch = {
        "source_image": torch.zeros(1, 1, 2, 2),
        "source_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "mask_available": torch.ones(1, 1, 1, 1),
        "true_pose": torch.tensor([[-2000.0, 0.0, 0.0]]),
        "sample_manifest_sha256": "m" * 64,
        "input_outline_receipt_sha256": ["o" * 64],
    }

    class NonfiniteInitializer:
        def initialize(self, image, outline, available):
            count = len(image)
            return {
                "pose": torch.full((count, 3), float("nan")),
                "pose_cholesky": torch.eye(3).expand(count, -1, -1).clone(),
                "pose_covariance": torch.eye(3).expand(count, -1, -1).clone(),
            }

    evaluate = foundation._development_evaluator(
        {"contract_sha256": "d" * 64},
        {name: batch for name in independent_data.OUTLINE_MODE_NAMES},
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.device("cpu"),
    )
    panel = evaluate(NonfiniteInitializer(), 0)
    assert panel["nonfinite_output_count"] == 9
    assert isinstance(panel["panel_manifest_sha256"], str)
    result = foundation.qualification_status(
        [panel], [], 0, foundation.load_foundation_config(CONFIG)
    )
    assert result["decision"] == "stop"
    assert not result["checks"]["nonfinite_count"]["passed"]


class FakeGenerator:
    def __init__(self, atlas_folder, device):
        self.device = torch.device(device)
        self.contract = {"contract_sha256": "a" * 64}
        self.annotation = torch.ones(2, 2, 2, dtype=torch.int16)


def _fake_evaluator(panel_contract, batches, brain_mask, device):
    def evaluate(model, views):
        by_mode, raw = {}, []
        for mode in independent_data.OUTLINE_MODE_NAMES:
            by_mode[mode] = {
                "physical_corresponding_plane_error_um": 100.0,
                "ap_mae_um": 100.0,
                "lr_mae_deg": 10.0,
                "dv_mae_deg": 5.0,
                "sample_count": 8,
            }
            for item in range(8):
                record = {
                    "panel_contract_sha256": panel_contract["contract_sha256"],
                    "views": views,
                    "outline_mode": mode,
                    "sample_index": item,
                    "sample_manifest_sha256": f"manifest-{mode}",
                    "input_outline_receipt_sha256": f"outline-{mode}-{item}",
                    "true_pose": torch.zeros(3),
                    "predicted_pose": torch.zeros(3),
                    "pose_cholesky": torch.eye(3),
                    "pose_covariance": torch.eye(3),
                    "physical_corresponding_plane_error_um": 100.0,
                }
                record["record_provenance_sha256"] = foundation._canonical_sha256(
                    {
                        "panel_contract_sha256": panel_contract["contract_sha256"],
                        "outline_mode": mode,
                        "sample_index": item,
                        "sample_manifest_sha256": f"manifest-{mode}",
                    }
                )
                raw.append(record)
        result = {
            "partition": "development",
            "fresh_checkpoint_views": views,
            "model_state": "ema",
            "panel_contract_sha256": panel_contract["contract_sha256"],
            "primary_outline_mode": "absent",
            "overall": {
                "physical_corresponding_plane_error_um": 100.0,
                "ap_mae_um": 100.0,
                "lr_mae_deg": 10.0,
                "dv_mae_deg": 5.0,
                "sample_count": 24,
            },
            "by_outline_mode": by_mode,
            "nonfinite_output_count": 0,
            "raw_predictions": raw,
        }
        result["panel_manifest_sha256"] = foundation._canonical_sha256(result)
        return result

    return evaluate


def test_cpu_micro_run_resumes_exactly_with_atomic_unique_view_receipts(tmp_path, monkeypatch):
    config = foundation.load_foundation_config(CONFIG)
    config = copy.deepcopy(config)
    config["device"] = "cpu"
    config["training"]["amp"] = False
    config["model"]["kwargs"] = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
        "integration_steps": 3,
    }
    config["model"]["expected_parameter_count"] = sum(
        value.numel() for value in _tiny_model().parameters()
    )
    run_root = tmp_path / "resume"
    monkeypatch.setattr(foundation, "load_foundation_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(foundation, "SyntheticRegistrationGenerator", FakeGenerator)
    monkeypatch.setattr(foundation.independent_data, "IndependentSyntheticData", FakeSynthetic)
    monkeypatch.setattr(
        foundation,
        "_development_setup",
        lambda config, synthetic, generator: (
            {"contract_sha256": "d" * 64}, {}, torch.ones(1, 1, 1, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(foundation, "_development_evaluator", _fake_evaluator)
    monkeypatch.setattr(
        foundation, "_resolve_paths", lambda config: (tmp_path / "atlas", run_root)
    )

    first = foundation.run_initializer_foundation(CONFIG, max_updates_this_call=1)
    second = foundation.run_initializer_foundation(CONFIG, max_updates_this_call=1)
    assert first["views"] == 2
    assert second["views"] == 4
    running_receipt = json.loads(second["receipt_path"].read_text())
    assert running_receipt["execution_environment"]["resolved_device"] == "cpu"
    assert not running_receipt["execution_environment"]["amp_enabled"]
    assert running_receipt["execution_environment"]["torch_version"] == torch.__version__
    resumed = torch.load(
        second["checkpoint_folder"] / "latest.pt", map_location="cpu", weights_only=False
    )
    assert isinstance(resumed["rng_state"]["torch"], torch.ByteTensor)
    assert resumed["rng_state"]["cuda"] is None or (
        isinstance(resumed["rng_state"]["cuda"], list)
        and all(
            isinstance(value, torch.ByteTensor)
            for value in resumed["rng_state"]["cuda"]
        )
    )
    foundation._validate_rng_state_types(
        {"torch": torch.get_rng_state(), "cuda": [torch.get_rng_state().clone()]}
    )
    with pytest.raises(RuntimeError, match="CUDA RNG state"):
        foundation._validate_rng_state_types(
            {"torch": torch.get_rng_state(), "cuda": torch.get_rng_state()}
        )
    identities = [value["view_content_sha256"] for value in resumed["view_receipts"]]
    assert len(identities) == len(set(identities)) == 4
    assert not list(second["checkpoint_folder"].glob("*.tmp"))

    run_root = tmp_path / "direct"
    direct_result = foundation.run_initializer_foundation(CONFIG, max_updates_this_call=2)
    direct = torch.load(
        direct_result["checkpoint_folder"] / "latest.pt", map_location="cpu", weights_only=False
    )
    assert direct["views"] == resumed["views"] == 4
    assert all(
        torch.equal(resumed["model"][name], direct["model"][name])
        for name in resumed["model"]
    )
    assert all(
        torch.equal(resumed["ema"][name], direct["ema"][name])
        for name in resumed["ema"]
    )
    assert resumed["gradient_records"] == direct["gradient_records"]

    foundation._validate_gradient_records(direct["gradient_records"], config)
    bad_gradients = copy.deepcopy(direct["gradient_records"])
    bad_gradients[0]["update"] = 2
    with pytest.raises(RuntimeError, match="gradient sequence"):
        foundation._validate_gradient_records(bad_gradients, config)
    bad_clipping = copy.deepcopy(direct["gradient_records"])
    bad_clipping[0]["encoder"].update(
        {"preclip_norm": 4.0, "postclip_norm": 2.0, "clip_factor": 1.0, "clipped": False}
    )
    with pytest.raises(RuntimeError, match="clipping telemetry"):
        foundation._validate_gradient_records(bad_clipping, config)
    foundation._validate_evaluations(
        direct["evaluations"], direct["views"], {"contract_sha256": "d" * 64}, config
    )
    bad_evaluations = copy.deepcopy(direct["evaluations"])
    bad_evaluations[0]["overall"]["ap_mae_um"] = 99.0
    with pytest.raises(RuntimeError, match="panel provenance"):
        foundation._validate_evaluations(
            bad_evaluations,
            direct["views"],
            {"contract_sha256": "d" * 64},
            config,
        )

    terminal = torch.load(
        second["checkpoint_folder"] / "latest.pt", map_location="cpu", weights_only=False
    )
    terminal["status"] = "stop"
    terminal["nonfinite_training_count"] = 1
    torch.save(terminal, second["checkpoint_folder"] / "latest.pt")
    final_path = second["checkpoint_folder"] / "final.pt"
    if final_path.exists():
        final_path.unlink()
    run_root = tmp_path / "resume"
    repaired = foundation.run_initializer_foundation(CONFIG, max_updates_this_call=0)
    assert repaired["status"] == "stop"
    assert final_path.is_file()
    assert json.loads(repaired["receipt_path"].read_text())["status"] == "stop"

    receipt_path = repaired["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["role"] = "architecture-selection"
    receipt_path.write_text(json.dumps(receipt))
    run_root = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="receipt differs"):
        foundation.run_initializer_foundation(CONFIG, max_updates_this_call=1)
