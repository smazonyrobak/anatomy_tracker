from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.run_independent_architecture_screen as screen
from training.train_independent_joint import _validate_development_panel


HASH = "a" * 64
CONFIG = (
    Path(__file__).parents[1]
    / "training"
    / "configs"
    / "independent_architecture_development_screen.json"
)


class FakeRenderer:
    device = torch.device("cpu")
    contract = {
        "contract_sha256": "1" * 64,
        "average_template_sha256": "2" * 64,
        "annotation_sha256": "3" * 64,
    }

    def __init__(self, atlas_folder, device="cpu"):
        self.atlas_folder = Path(atlas_folder)


def _batch(count, *, source_type="synthetic_ccf", split="validation", mode=None):
    modes = (
        torch.arange(count, dtype=torch.int8) % 3
        if mode is None
        else torch.full((count,), int(mode), dtype=torch.int8)
    )
    true_pose = torch.stack(
        (
            -1000.0 - 25.0 * torch.arange(count),
            torch.arange(count).float(),
            -torch.arange(count).float(),
        ),
        dim=1,
    )
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [-100.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    candidates = true_pose[:, None] + offsets[None]
    image = torch.ones(count, 7, 1, 4, 5)
    mask = torch.ones_like(image, dtype=torch.bool)
    labels = torch.ones_like(image, dtype=torch.long)
    dense = source_type == "synthetic_ccf"
    result = {
        "source_type": source_type,
        "data_split": split,
        "data_contract_sha256": "4" * 64,
        "source_image": torch.ones(count, 1, 4, 5),
        "source_mask": (modes != 2)[:, None, None, None].expand(count, 1, 4, 5),
        "mask_available": (modes != 2).float()[:, None, None, None],
        "input_outline_mode": modes,
        "input_outline_receipt_sha256": [f"{item + 1:x}" * 64 for item in range(count)],
        "true_pose": true_pose,
        "candidate_pose": candidates,
        "candidate_fixed_image": image,
        "candidate_fixed_mask": mask,
        "candidate_fixed_labels": labels,
        "candidate_in_training_domain": torch.ones(count, 7, dtype=torch.bool),
        "candidate_dense_truth_valid": torch.zeros(count, 7, dtype=torch.bool),
        "listwise_target_index": torch.zeros(count, dtype=torch.long),
        "listwise_positive_mask": torch.nn.functional.one_hot(
            torch.zeros(count, dtype=torch.long), 7
        ).bool(),
        "dense_truth_valid": torch.full((count,), dense, dtype=torch.bool),
        "animal_id": torch.full((count,), -1, dtype=torch.long),
        "specimen_id": torch.full((count,), -1, dtype=torch.long),
    }
    if dense:
        identity = torch.stack(
            torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")[::-1]
        ).float()[None].expand(count, -1, -1, -1)
        result.update(
            {
                "sample_manifest_sha256": HASH,
                "truth_fixed_image": image[:, 0],
                "truth_fixed_mask": mask[:, 0],
                "truth_fixed_labels": labels[:, 0],
                "truth_source_labels": labels[:, 0].clone(),
                "truth_source_tissue_mask": mask[:, 0].clone(),
                "truth_source_brush_mask": mask[:, 0].clone(),
                "truth_svf": torch.zeros(count, 2, 4, 5),
                "truth_fixed_to_source_map": identity.clone(),
                "truth_source_to_fixed_map": identity.clone(),
                "truth_fixed_valid_mask": mask[:, 0],
                "truth_source_valid_mask": mask[:, 0],
                "truth_source_damage_mask": torch.zeros(count, 1, 4, 5, dtype=torch.bool),
                "truth_source_view_h": torch.eye(3)[None].expand(count, -1, -1).clone(),
                "truth_generator_similarity_h": torch.eye(3)[None].expand(count, -1, -1).clone(),
                "truth_similarity_h": torch.eye(3)[None].expand(count, -1, -1).clone(),
                "truth_similarity_parameters": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0, 0.0]]
                ).expand(count, -1),
                "truth_source_view_parameters": torch.tensor([[0.0, 1.0]]).expand(count, -1),
            }
        )
        result["candidate_dense_truth_valid"][:, 0] = True
    else:
        result.update(
            {
                "batch_manifest_sha256": "5" * 64,
                "record_provenance_sha256": [f"{item + 5:x}" * 64 for item in range(count)],
                "source_relative_path": [f"animal/{item}.jpg" for item in range(count)],
                "product_id": torch.full((count,), 5),
                "animal_id": 20 + torch.arange(count) % 2,
                "specimen_id": 20 + torch.arange(count) % 2,
                "experiment_id": 200 + torch.arange(count),
                "section_image_id": 300 + torch.arange(count),
            }
        )
    return result


class FakeSynthetic:
    contract = {"contract_sha256": "4" * 64}

    def __init__(self, generator):
        self.generator = generator

    def make_manifest(self, count, split, seed, stratum, *, pose_regime="standard"):
        manifest = {
            "version": 1,
            "contract_sha256": self.contract["contract_sha256"],
            "sample_count": int(count),
            "split": split,
            "seed": int(seed),
            "stratum": stratum,
            "pose_regime": pose_regime,
            "outline_plan": {"mode": np.arange(count, dtype=np.int8) % 3},
            "generator_manifest": {
                "manifest_sha256": "e" * 64,
                "seed": int(seed),
                "sample_count": int(count),
            },
        }
        manifest["manifest_sha256"] = screen.independent_data._payload_sha256(manifest)
        return manifest

    def batch(self, manifest):
        modes = np.asarray(manifest["outline_plan"]["mode"])
        batch = _batch(len(modes), mode=int(modes[0]) if len(modes) == 1 else None)
        batch["input_outline_mode"] = torch.as_tensor(modes, dtype=torch.int8)
        batch["mask_available"] = (batch["input_outline_mode"] != 2).float()[:, None, None, None]
        batch["source_mask"] = batch["mask_available"].bool().expand(-1, 1, 4, 5)
        batch["source_image"] = batch["source_image"] * (
            0.5 + 0.25 * batch["input_outline_mode"].float()[:, None, None, None]
        )
        batch["sample_manifest_sha256"] = manifest["manifest_sha256"]
        return batch

    def generate(self, count, split, seed, stratum, *, pose_regime="standard"):
        return self.batch(self.make_manifest(count, split, seed, stratum, pose_regime=pose_regime))

    def generate_high_tilt(self, count, split, seed, stratum):
        return self.generate(count, split, seed, stratum, pose_regime="high_tilt")


class FakeProduct5:
    calls = []

    def __init__(self, root, atlas_folder, renderer, *, split):
        self.split = split
        self.contract = {
            "contract_sha256": ("6" if split == "train" else "7") * 64,
            "specimen_ids": [10] if split == "train" else [20, 21],
        }
        self.calls.append(split)

    def fixed_validation_positions(self, count, seed):
        return np.arange(count, dtype=np.int64)

    def provenance_manifest(self, positions=None):
        positions = np.arange(2) if positions is None else np.asarray(positions)
        return {
            "split": self.split,
            "record_identities": [
                {"animal_id": 10 if self.split == "train" else 20 + int(item) % 2}
                for item in positions
            ],
            "manifest_sha256": ("8" if self.split == "train" else "9") * 64,
        }

    def batch_positions(self, positions, seed, schedule_step):
        batch = _batch(len(positions), source_type="allen_registered_product5")
        batch["data_contract_sha256"] = self.contract["contract_sha256"]
        return batch

    def generate(self, count, seed, schedule_step):
        batch = _batch(count, source_type="allen_registered_product5", split="train")
        batch["data_contract_sha256"] = self.contract["contract_sha256"]
        batch["animal_id"][:] = 10
        batch["specimen_id"][:] = 10
        return batch


def _write_protocol(tmp_path, **training_changes):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload.pop("contract_sha256")
    payload["training"].update(training_changes)
    if int(payload["development"]["evaluate_every"]) > int(payload["training"]["steps"]):
        payload["development"]["evaluate_every"] = int(payload["training"]["steps"])
    payload["contract_sha256"] = screen._canonical_sha256(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checked_protocol_is_portable_frozen_and_source_bound():
    protocol = screen.load_screen_protocol(CONFIG)
    assert protocol["paths"] == {
        "atlas_repo_relative": "data/Allen Brain Atlas 25um",
        "product5_root_env": "ATLAS_PRODUCT5_ROOT",
        "run_root_env": "ATLAS_JOINT_RUN_ROOT",
    }
    assert protocol["purpose"] == screen.PROTOCOL_PURPOSE
    assert protocol["calibration_access"] is protocol["final_test_access"] is False
    assert protocol["learned_checkpoint_dependencies"] == []
    assert protocol["training"]["steps"] == 2000
    assert protocol["development"]["metric"]["primary_panel_weights"] == {
        "product5_absent": 0.5,
        "paired_outline_absent": 0.25,
        "high_tilt_absent": 0.25,
    }


def test_protocol_tampering_and_forbidden_legacy_dependencies_are_rejected(tmp_path):
    tampered = json.loads(CONFIG.read_text(encoding="utf-8"))
    tampered["final_test_access"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        screen.load_screen_protocol(path)

    tree = ast.parse(Path(screen.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        forbidden in name
        for name in imports
        for forbidden in ("atlas_pose_models", "dense_registration_model", "joint_atlas")
    )


def test_paired_outline_panel_changes_only_model_input_outline():
    synthetic = FakeSynthetic(FakeRenderer("atlas"))
    manifests, batches = screen._paired_outline_batches(
        synthetic, {"count": 8, "seed": 17, "stratum": "hard", "shuffle_seed": 23}
    )
    assert {name: set(batch["input_outline_mode"].tolist()) for name, batch in batches.items()} == {
        "accurate": {0},
        "imperfect": {1},
        "absent": {2},
    }
    assert not batches["absent"]["mask_available"].any()
    assert not batches["absent"]["source_mask"].any()
    assert batches["accurate"]["mask_available"].all()
    for name in ("imperfect", "absent"):
        assert torch.equal(
            batches["accurate"]["truth_source_damage_mask"],
            batches[name]["truth_source_damage_mask"],
        )
        assert torch.equal(
            batches["accurate"]["truth_fixed_to_source_map"],
            batches[name]["truth_fixed_to_source_map"],
        )
        assert manifests[name]["generator_manifest"] == manifests["accurate"]["generator_manifest"]


def test_development_evaluator_uses_absent_primary_and_reports_paired_deltas(
    tmp_path, monkeypatch
):
    product = _batch(3, source_type="allen_registered_product5")
    paired = {name: _batch(1, mode=mode) for name, mode in screen.OUTLINE_MODES.items()}
    high = _batch(3)
    panel = {
        "contract_sha256": "b" * 64,
        "paired_outline_base_generator_manifest_sha256": "e" * 64,
        "metric": {
            "component_weights": {
                "pose": 1.0,
                "ranking": 0.0,
                "dense_map": 0.0,
                "dense_region": 0.0,
                "dense_validity": 0.0,
            },
            "primary_panel_weights": {
                "product5_absent": 0.5,
                "paired_outline_absent": 0.25,
                "high_tilt_absent": 0.25,
            },
        },
    }

    def forward(model, batch, renderer):
        mode_error = batch["input_outline_mode"].float()[:, None]
        dense_index = batch["dense_truth_valid"].nonzero(as_tuple=False).flatten()
        return {
            "settled_pose": batch["true_pose"] + mode_error * torch.tensor([100.0, 2.0, 2.0]),
            "ranking_logits_masked": torch.zeros(len(mode_error), 7),
            "dense_sample_index": dense_index,
            "dense": {"dummy": torch.zeros(len(dense_index), 1)} if len(dense_index) else None,
        }

    def records(output, batch):
        provenance = batch.get("record_provenance_sha256", [HASH] * len(batch["true_pose"]))
        return [
                {
                    "source_type": batch["source_type"],
                    "data_contract_sha256": batch["data_contract_sha256"],
                    "sample_manifest_sha256": batch.get("sample_manifest_sha256"),
                    "animal_id": int(batch["animal_id"][item]),
                "record_provenance_sha256": provenance[item],
                "candidate_score_softmax_uncalibrated": [1 / 7] * 7,
                "initializer_covariance": [[1.0, 0.0, 0.0]] * 3,
            }
            for item in range(len(batch["true_pose"]))
        ]

    monkeypatch.setattr(screen, "independent_joint_forward", forward)
    monkeypatch.setattr(screen, "raw_prediction_records", records)
    monkeypatch.setattr(
        screen,
        "dense_registration_losses",
        lambda output, batch, index: {
            "dense_map_forward": torch.tensor(0.1),
            "dense_map_inverse": torch.tensor(0.1),
            "dense_region_dice": torch.tensor(0.2),
            "dense_region_boundary": torch.tensor(0.2),
            "dense_validity": torch.tensor(0.3),
        },
    )
    result = screen._development_evaluator(
        FakeRenderer("atlas"), "c" * 64, panel, product, paired, high, tmp_path
    )(object(), 11)
    assert result["primary_endpoint"] == "absent/no-user-mask"
    assert set(result["paired_outline_metrics"]) == {"accurate", "imperfect", "absent"}
    assert result["paired_outline_deltas_vs_absent"]["accurate"]["pose_normalized_mean_l2"] < 0
    assert result["selection_metric"] == pytest.approx(12**0.5)
    assert set(result["animal_ids"]) == {20, 21}
    assert len(result["raw_predictions"]) == 3
    assert len(result["synthetic_raw_predictions"]) == 6
    paired_provenance = {
        record["paired_base_sample_provenance_sha256"]
        for record in result["synthetic_raw_predictions"]
        if record["panel_name"].startswith("paired_outline_")
    }
    assert len(paired_provenance) == 1
    metric, _ = _validate_development_panel(result, 11, [10], "b" * 64)
    assert metric == pytest.approx(result["selection_metric"])
    assert json.loads((tmp_path / "development_panel_latest.json").read_text())["raw_predictions"]


def test_zero_step_driver_wires_scratch_providers_animal_custody_and_receipt(
    tmp_path, monkeypatch
):
    protocol_path = _write_protocol(tmp_path, steps=1, max_steps_this_call=0)
    product_root = tmp_path / "product5"
    run_root = tmp_path / "runs"
    product_root.mkdir()
    monkeypatch.setenv("ATLAS_PRODUCT5_ROOT", str(product_root))
    monkeypatch.setenv("ATLAS_JOINT_RUN_ROOT", str(run_root))
    architecture = {
        "name": "tiny-scratch",
        "contract_sha256": "c" * 64,
        "workload": {"seed": 4322},
    }
    model = torch.nn.Linear(1, 1)
    model.learned_weight_dependencies = ()
    captured = {}

    def fake_train(model_arg, renderer, providers, contracts, checkpoint, steps, **kwargs):
        captured.update(
            {
                "model": model_arg,
                "providers": providers,
                "contracts": contracts,
                "checkpoint": Path(checkpoint),
                "steps": steps,
                "kwargs": kwargs,
            }
        )
        for name, provider in providers.items():
            first = provider(0, 0)
            second = provider(0, 0)
            assert torch.equal(first["true_pose"], second["true_pose"]), name
        lineage = {"lineage_sha256": "d" * 64}
        panel = {
            "partition": "validation",
            "fresh_checkpoint_step": 1,
            "panel_contract_sha256": kwargs["development_panel_contract_sha256"],
            "panel_manifest_sha256": "f" * 64,
            "animal_ids": [20],
            "selection_metric": 1.25,
            "raw_predictions": [
                {
                    "animal_id": 20,
                    "record_provenance_sha256": "e" * 64,
                    "candidate_score_softmax_uncalibrated": [1 / 7] * 7,
                    "initializer_covariance": [[1.0, 0.0, 0.0]] * 3,
                }
            ],
            "synthetic_raw_predictions": [{"synthetic_sample_index": 0}],
        }
        best_checkpoint = Path(checkpoint) / "best.pt"
        torch.save(
            {
                "format": "independent-joint-cold-start-v1",
                "learned_checkpoint_dependencies": [],
                "checkpoint_selection_state": "ema",
                "lineage": lineage,
                "step": 1,
                "best_metric": 1.25,
                "development_panel": panel,
            },
            best_checkpoint,
        )
        return {
            "step": 1,
            "best_metric": 1.25,
            "best_checkpoint": best_checkpoint,
            "last_loss": 2.5,
            "lineage": lineage,
            "raw_predictions": [{"training_sample": 1}],
        }

    FakeProduct5.calls.clear()
    monkeypatch.setattr(screen, "load_architecture_config", lambda path: architecture)
    monkeypatch.setattr(screen, "build_model", lambda config: model)
    monkeypatch.setattr(screen, "SyntheticRegistrationGenerator", FakeRenderer)
    monkeypatch.setattr(screen, "IndependentSyntheticData", FakeSynthetic)
    monkeypatch.setattr(screen, "IndependentProduct5Data", FakeProduct5)
    monkeypatch.setattr(screen, "train_independent_joint", fake_train)
    result = screen.run_architecture_screen(protocol_path, tmp_path / "architecture.json")

    assert FakeProduct5.calls == ["train", "validation"]
    assert result["train_animal_ids"] == [10]
    assert result["validation_animal_ids"] == [20, 21]
    assert captured["steps"] == 1
    assert captured["kwargs"]["train_animal_ids"] == [10]
    assert captured["kwargs"]["development_panel_contract_sha256"] == result[
        "panel_contract_sha256"
    ]
    assert set(captured["contracts"]) == {
        "regular_synthetic",
        "high_tilt",
        "product5",
    }
    receipt = json.loads(result["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["learned_checkpoint_dependencies"] == []
    assert receipt["train_animal_ids"] == [10]
    assert receipt["validation_animal_ids"] == [20, 21]
    assert receipt["development_panel"]["purpose"] == "animal-disjoint-development-selection-only"
    assert receipt["development_panel"]["historically_consumed"] is True
    assert receipt["development_panel"]["untouched_final_test"] is False
    assert receipt["architecture_initial_state_sha256"] == screen._state_sha256(model)
    assert Path(receipt["development_panel_registry"]).is_file()
    assert receipt["best_development_panel"]["selection_metric"] == 1.25
    assert receipt["best_development_panel"]["checkpoint_selection_state"] == "ema"
    assert receipt["best_development_panel"]["trainer_best_checkpoint_sha256"] == screen._binary_sha256(
        captured["checkpoint"] / "best.pt"
    )
    assert receipt["development_receipts"]["best"].endswith("development_panel_best.json")
    assert receipt["last_training_raw_predictions"] == [{"training_sample": 1}]

    best_checkpoint = captured["checkpoint"] / "best.pt"
    monkeypatch.setattr(
        screen,
        "train_independent_joint",
        lambda *args, **kwargs: {
            "step": 1,
            "best_metric": 1.25,
            "best_checkpoint": best_checkpoint,
            "last_loss": None,
            "lineage": {"lineage_sha256": "d" * 64},
            "raw_predictions": [],
        },
    )
    screen.run_architecture_screen(protocol_path, tmp_path / "architecture.json")
    resumed_receipt = json.loads(result["receipt_path"].read_text())
    assert resumed_receipt["last_training_raw_predictions"] == [{"training_sample": 1}]
    assert resumed_receipt["training_result"]["last_loss"] == 2.5
    before_failed_resume = result["receipt_path"].read_bytes()
    monkeypatch.setattr(
        screen,
        "train_independent_joint",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("resume lineage mismatch")),
    )
    with pytest.raises(RuntimeError, match="resume lineage mismatch"):
        screen.run_architecture_screen(protocol_path, tmp_path / "architecture.json")
    assert result["receipt_path"].read_bytes() == before_failed_resume


def test_driver_rejects_overlapping_product5_animals(tmp_path, monkeypatch):
    protocol_path = _write_protocol(tmp_path, steps=1, max_steps_this_call=0)
    monkeypatch.setenv("ATLAS_PRODUCT5_ROOT", str(tmp_path))
    monkeypatch.setenv("ATLAS_JOINT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(
        screen,
        "load_architecture_config",
        lambda path: {"name": "tiny", "contract_sha256": "c" * 64, "workload": {"seed": 4322}},
    )
    model = torch.nn.Linear(1, 1)
    model.learned_weight_dependencies = ()
    monkeypatch.setattr(screen, "build_model", lambda config: model)
    monkeypatch.setattr(screen, "SyntheticRegistrationGenerator", FakeRenderer)
    monkeypatch.setattr(screen, "IndependentSyntheticData", FakeSynthetic)

    class Overlapping(FakeProduct5):
        def __init__(self, root, atlas_folder, renderer, *, split):
            super().__init__(root, atlas_folder, renderer, split=split)
            self.contract["specimen_ids"] = [10]

    monkeypatch.setattr(screen, "IndependentProduct5Data", Overlapping)
    with pytest.raises(RuntimeError, match="animal-disjoint"):
        screen.run_architecture_screen(protocol_path, tmp_path / "architecture.json")
