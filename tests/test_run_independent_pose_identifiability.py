import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.independent_joint_data as independent_data
import training.run_independent_pose_identifiability as diagnostic
from training.independent_joint_model import IndependentJointModel


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "training/configs/independent_pose_identifiability_300_r4322.json"
CONFIG_SHA256 = "efcd541ed9824ca286ff065a0cd7693091cac13bbb353459a8d7f289b2aada6b"


def _historical_config():
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    config = diagnostic.inspect_pose_identifiability_config(CONFIG)
    with pytest.raises(ValueError, match="source lineage changed"):
        diagnostic.load_pose_identifiability_config(CONFIG)
    return config


def _tiny_model():
    torch.manual_seed(3)
    return IndependentJointModel(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def test_frozen_config_has_balanced_split_safe_poses_and_disjoint_fixed_transforms():
    config = _historical_config()
    poses = diagnostic.latent_pose_table(config)
    assert poses.shape == (24, 3)
    assert np.unique(poses[:, 0]).size == 6
    assert np.all(np.unique(poses[:, 0], return_counts=True)[1] == 4)
    assert np.array_equal(np.unique(poses[:, 1]), [-13.25, 13.25])
    assert np.array_equal(np.unique(poses[:, 2]), [-18.25, 18.25])
    indices = np.rint(
        independent_data.BREGMA_AP_INDEX - poses[:, 0] / independent_data.VOXEL_UM
    ).astype(np.int32)
    assert np.isin(indices, diagnostic.split_ap_indices("train")).all()
    centers = (
        np.linspace(-4500.0, 500.0, 41),
        np.linspace(-35.0, 35.0, 29),
        np.linspace(-35.0, 35.0, 29),
    )
    residual = np.column_stack([
        poses[:, axis] - value[np.abs(poses[:, axis, None] - value).argmin(1)]
        for axis, value in enumerate(centers)
    ])
    assert not np.isclose(residual, 0.0).any()
    assert (np.abs(residual) <= [62.5, 1.25, 1.25]).all()

    tables = diagnostic.nuisance_transform_tables(config)
    assert all(len(value) == 2 for value in tables.values())
    assert all(panel.shape == (24, 2) for value in tables.values() for panel in value)
    for item in range(24):
        assert not np.array_equal(tables["seen"][0][item], tables["seen"][1][item])
        assert not np.array_equal(tables["held"][0][item], tables["held"][1][item])
    seen = {tuple(row) for panel in tables["seen"] for row in panel.tolist()}
    held = {tuple(row) for panel in tables["held"] for row in panel.tolist()}
    assert seen.isdisjoint(held)
    assert np.all(diagnostic.nuisance_shortcut_accuracy(config, "seen") <= [0.50, 2 / 3, 2 / 3])
    assert config["learned_checkpoint_dependencies"] == []
    assert not config["product5_access"]
    assert not config["calibration_access"]
    assert not config["final_test_access"]


class FakeManifestSource:
    def make_manifest(self, count, split, seed, stratum, negatives, pose_regime):
        generator = {
            "appearance_seed": seed,
            "ap_um": np.zeros(count, np.float32),
            "ap_index": np.zeros(count, np.float32),
            "tilt_lr_deg": np.zeros(count, np.float32),
            "tilt_dv_deg": np.zeros(count, np.float32),
            "manifest_sha256": "old-generator",
        }
        return {
            "sample_count": count,
            "true_pose": np.zeros((count, 3), np.float32),
            "generator_manifest": generator,
            "source_view_rotation_deg": np.zeros(count, np.float32),
            "source_view_scale": np.ones(count, np.float32),
            "negative_count": negatives,
            "wrong_candidate_offset": np.zeros((count, negatives, 3), np.float32),
            "manifest_sha256": "old-outer",
        }


def test_fixed_manifests_are_exact_absent_outline_and_map_consistent():
    config = _historical_config()
    poses = diagnostic.latent_pose_table(config)
    transforms = diagnostic.nuisance_transform_tables(config)
    manifests = diagnostic.fixed_panel_manifests(FakeManifestSource(), config)
    assert (
        manifests["seen"][0]["generator_manifest"]["manifest_sha256"]
        != manifests["held"][0]["generator_manifest"]["manifest_sha256"]
    )
    for kind in ("seen", "held"):
        for panel_index, manifest in enumerate(manifests[kind]):
            generator = manifest["generator_manifest"]
            assert np.array_equal(manifest["true_pose"], poses)
            assert np.array_equal(generator["ap_um"], poses[:, 0])
            assert np.array_equal(generator["tilt_lr_deg"], poses[:, 1])
            assert np.array_equal(generator["tilt_dv_deg"], poses[:, 2])
            assert np.array_equal(
                manifest["source_view_rotation_deg"], transforms[kind][panel_index][:, 0]
            )
            assert np.array_equal(
                manifest["source_view_scale"], transforms[kind][panel_index][:, 1]
            )
            assert np.array_equal(manifest["outline_plan"]["mode"], np.full(24, 2))
            assert "wrong_candidate_offset" not in manifest
            assert "negative_count" not in manifest
            assert generator["manifest_sha256"] == independent_data._payload_sha256(
                {key: value for key, value in generator.items() if key != "manifest_sha256"}
            )
            assert manifest["manifest_sha256"] == independent_data._payload_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )


def test_trainable_boundary_and_loss_exclude_covariance_anchor_pair_recurrent_and_dense():
    model = _tiny_model().train()
    parameters = diagnostic._pose_parameter_group(model)
    trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable_names
    assert all(
        name.startswith((
            "pyramid.slice_stem", "pyramid.levels", "pose_head.context",
            "pose_head.ap_logits", "pose_head.lr_logits", "pose_head.dv_logits",
            "pose_head.residual",
        ))
        for name in trainable_names
    )
    assert all(not value.requires_grad for value in model.pose_head.local_cholesky.parameters())
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    frozen_before = diagnostic.foundation._named_parameter_sha256(model, frozen_names)
    image = torch.rand(4, 1, 64, 64)
    mask = torch.zeros(4, 1, 64, 64, dtype=torch.bool)
    available = torch.zeros(4, 1, 1, 1)
    truth = torch.tensor([
        [-4125.0, -12.5, -17.5], [-3125.0, -12.5, 17.5],
        [-2125.0, 12.5, -17.5], [-1125.0, 12.5, 17.5],
    ])
    output = model.initialize(image, mask, available)
    losses = diagnostic.categorical_residual_loss(output, truth, model)
    loss = losses["categorical"] + 0.5 * losses["sub_bin_residual"]
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)
    assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    optimizer.step()
    assert diagnostic.foundation._named_parameter_sha256(model, frozen_names) == frozen_before
    assert losses["target_bins"].shape == (4, 3)


def _evaluation(
    seen=(0.95, 0.90, 0.90),
    mae=(250.0, 3.0, 3.0),
    ratios=(0.75, 0.75, 0.75),
    improvement=0.50,
    residual=(0.20, 0.20, 0.20),
    nonfinite=0,
):
    truth_sd = torch.tensor([1400.0, 12.5, 17.5])
    return {
        "seen": {
            "bin_accuracy": torch.tensor(seen, dtype=torch.float64),
            "residual_improvement_over_zero": torch.tensor(residual, dtype=torch.float64),
        },
        "held": {
            "mae": torch.tensor(mae, dtype=torch.float64),
            "prediction_sd": truth_sd.double() * torch.tensor(ratios, dtype=torch.float64),
            "truth_sd": truth_sd.double(),
            "physical_improvement_over_constant_prior": improvement,
            "residual_improvement_over_zero": torch.tensor(residual, dtype=torch.float64),
        },
        "nonfinite_output_count": nonfinite,
    }


def _gradients(clipped_count=134):
    return [
        {"update": update, "clipped": update <= 30 or update <= 30 + clipped_count}
        for update in range(1, 301)
    ]


def test_predeclared_gates_are_exact_strict_and_classify_representation_vs_invariance():
    config = _historical_config()
    passed = diagnostic.qualification_status(_evaluation(), _gradients(), 0, config)
    assert passed["decision"] == "go"
    assert passed["classification"] == "pose-representation-and-held-transform-invariance-demonstrated"
    assert all(value["passed"] for value in passed["checks"].values())

    half_clipped = diagnostic.qualification_status(
        _evaluation(), _gradients(clipped_count=135), 0, config
    )
    assert half_clipped["decision"] == "stop"
    assert not half_clipped["checks"]["postwarm_clipping"]["passed"]
    assert half_clipped["classification"].endswith("training-stability-gate-failed")

    unidentifiable = diagnostic.qualification_status(
        _evaluation(seen=(0.949, 1.0, 1.0)), _gradients(), 0, config
    )
    assert unidentifiable["classification"] == "pose-representation-not-identifiable-on-seen-transforms"
    zero_residual = diagnostic.qualification_status(
        _evaluation(residual=(0.0, 0.20, 0.20)), _gradients(), 0, config
    )
    assert zero_residual["decision"] == "stop"
    assert not zero_residual["checks"]["seen_residual_improvement"]["passed"]
    numerical = diagnostic.qualification_status(
        _evaluation(nonfinite=1), _gradients(), 0, config
    )
    assert numerical["classification"] == "numerical-failure"


class FakeGenerator:
    def __init__(self, atlas_folder, device):
        self.device = torch.device(device)
        self.contract = {"contract_sha256": "a" * 64}
        self.annotation = torch.ones(2, 2, 2, dtype=torch.int16)


class FakeSynthetic:
    def __init__(self, generator):
        self.contract = {"contract_sha256": "d" * 64}


def _fake_panels(config, synthetic, generator):
    truth = torch.from_numpy(diagnostic.latent_pose_table(config))
    panels = {"seen": [], "held": []}
    transforms = diagnostic.nuisance_transform_tables(config)
    base = torch.linspace(0.0, 1.0, 24 * 32 * 32).reshape(24, 1, 32, 32)
    for kind in panels:
        for index in range(2):
            panels[kind].append({
                "source_image": torch.roll(base, index + (kind == "held"), dims=-1),
                "source_mask": torch.zeros(24, 1, 32, 32, dtype=torch.bool),
                "mask_available": torch.zeros(24, 1, 1, 1),
                "true_pose": truth.clone(),
                "manifest_sha256": f"{kind}-{index}",
                "generator_manifest_sha256": "g" * 64,
                "outline_plan_sha256": "o" * 64,
                "data_contract_sha256": synthetic.contract["contract_sha256"],
                "source_view_rotation_deg": transforms[kind][index][:, 0],
                "source_view_scale": transforms[kind][index][:, 1],
            })
    contract = {"contract_sha256": "p" * 64}
    return contract, panels, torch.ones(24, 7, 9, dtype=torch.bool)


def test_cpu_micro_run_resumes_bit_exact_with_single_atomic_state(tmp_path, monkeypatch):
    config = copy.deepcopy(_historical_config())
    config["device"] = "cpu"
    config["training"]["amp"] = False
    config["training"]["max_updates"] = 4
    config["training"]["gradient_clip_warmup_updates"] = 0
    config["training"]["resume_state_every_updates"] = 1
    config["evaluation"]["evaluate_at_update"] = 4
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
    monkeypatch.setattr(
        diagnostic, "load_pose_identifiability_config", lambda path: copy.deepcopy(config)
    )
    monkeypatch.setattr(diagnostic, "SyntheticRegistrationGenerator", FakeGenerator)
    monkeypatch.setattr(diagnostic.independent_data, "IndependentSyntheticData", FakeSynthetic)
    monkeypatch.setattr(diagnostic, "_prepare_fixed_panels", _fake_panels)
    monkeypatch.setattr(
        diagnostic, "_resolve_paths", lambda config: (tmp_path / "atlas", run_root)
    )

    first = diagnostic.run_pose_identifiability(CONFIG, max_updates_this_call=1)
    second = diagnostic.run_pose_identifiability(CONFIG, max_updates_this_call=1)
    assert first["status"] == second["status"] == "paused"
    assert first["updates"] == 1
    assert second["updates"] == 2
    resumed = torch.load(second["state_path"], map_location="cpu", weights_only=False)
    assert resumed["format"] == diagnostic.FORMAT
    assert resumed["learned_checkpoint_dependencies"] == []
    assert isinstance(resumed["rng_state"]["torch"], torch.ByteTensor)
    assert resumed["rng_state"]["cuda"] is None or all(
        isinstance(value, torch.ByteTensor) for value in resumed["rng_state"]["cuda"]
    )
    assert not list(second["state_path"].parent.glob("*.tmp"))
    receipt = json.loads(second["receipt_path"].read_text())
    assert receipt["artifact_policy"] == "single-atomic-resume-state-not-a-selected-model-checkpoint"
    assert receipt["progress"] == {"optimizer_updates": 2, "sample_presentations": 48}

    run_root = tmp_path / "direct"
    direct_result = diagnostic.run_pose_identifiability(CONFIG, max_updates_this_call=2)
    direct = torch.load(direct_result["state_path"], map_location="cpu", weights_only=False)
    assert direct["update"] == resumed["update"] == 2
    assert all(torch.equal(resumed["model"][name], direct["model"][name]) for name in resumed["model"])
    assert resumed["gradient_records"] == direct["gradient_records"]
