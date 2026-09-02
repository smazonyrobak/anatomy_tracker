import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import training.arbitrary_plane_receipt_bound_training_runner_v6 as runner


STABLE_DATA_RECEIPT = "d" * 64


class _Runtime:
    cell_count = runner.FULL_CATALOGUE_CELL_COUNT_V6
    binding = {
        "schema_version": "anatomy-tracker.complete-catalogue-runtime/v6",
        "catalogue_id": "complete-catalogue-v6",
        "catalogue_receipt_sha256": "c" * 64,
        "cell_count": runner.FULL_CATALOGUE_CELL_COUNT_V6,
        "representation_count": 2,
        "device": "cpu",
        "dtype": "torch.float32",
        "support_origin_ap_dv_ml_um": (0.0, 0.0, 0.0),
    }


def _rows(split="train"):
    animals = ("animal-a", "animal-a", "animal-b", "animal-c")
    return [
        {
            "lineage": {
                "animal_id": animal,
                "specimen_id": f"specimen-{index}",
                "experiment_id": f"experiment-{index}",
                "synthetic_animal_id": f"synthetic-{index}",
                "section_id": f"section-{index}",
                "split": split,
            },
            "training_row_id": f"training-row-{index}",
            "receipt_sha256": f"{index + 1:x}" * 64,
        }
        for index, animal in enumerate(animals)
    ]


def _config():
    return {
        "seed": 7,
        "proposal_only_steps": 1,
        "pose_rerank_steps": 1,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "proposal_top_m": 2,
        "top_k": 2,
        "refinement_steps": 1,
        "joint_pose_only_steps": 1,
        "retrieval_shape_h_w": (4, 4),
        "amp": False,
        "amp_initial_scale": 16.0,
        "gradient_clip_norm": 10.0,
        "proposal_loss_weight": 1.0,
        "rerank_loss_weight": 1.0,
    }


@pytest.fixture
def i_root():
    parent = Path("I:/AnatomyTracker/pytest_tmp")
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="receipt-bound-v6-", dir=parent))
    yield path
    shutil.rmtree(path)


def _install_fakes(monkeypatch, *, split="train"):
    rows = _rows(split)

    def load_manifest(cache_directory, *, expected_manifest_receipt_sha256):
        assert expected_manifest_receipt_sha256 == STABLE_DATA_RECEIPT
        return {
            "receipt_sha256": STABLE_DATA_RECEIPT,
            "row_count": len(rows),
            "generation_lineage": {"split": split},
            "rows": [
                {
                    "lineage": row["lineage"],
                    "training_row_id": row["training_row_id"],
                    "training_row_receipt_sha256": row["receipt_sha256"],
                }
                for row in rows
            ],
            "generator_binding": {
                "receipt_sha256": "e" * 64,
                "generation_lineage_sha256": "f" * 64,
            },
        }

    def load_frozen(cache_directory, indices=None, *, expected_manifest_receipt_sha256):
        assert expected_manifest_receipt_sha256 == STABLE_DATA_RECEIPT
        load_frozen.calls.append(indices)
        selected = list(range(len(rows))) if indices is None else list(indices)
        chosen = [rows[index] for index in selected]
        return {
            "schema_version": "anatomy-tracker.frozen-generated-row-payloads/v6",
            "training_data_manifest_receipt_sha256": STABLE_DATA_RECEIPT,
            "cache_manifest_receipt_sha256": STABLE_DATA_RECEIPT,
            "generator_binding_receipt_sha256": "e" * 64,
            "generation_lineage_sha256": "f" * 64,
            "row_indices": selected,
            "training_row_ids": [row["training_row_id"] for row in chosen],
            "training_row_receipts_sha256": [
                row["receipt_sha256"] for row in chosen
            ],
            "rows": chosen,
            "selection_receipt_sha256": runner._hash_json(
                {
                    "stable": STABLE_DATA_RECEIPT,
                    "indices": selected,
                    "ids": [row["training_row_id"] for row in chosen],
                }
            ),
        }

    load_frozen.calls = []

    def model_ready(
        frozen,
        catalogue,
        runtime,
        atlas,
        *,
        origin_ap_dv_ml_um,
        voxel_size_ap_dv_ml_um,
        finite_psf_capability,
        expected_training_data_manifest_receipt_sha256,
    ):
        assert expected_training_data_manifest_receipt_sha256 == STABLE_DATA_RECEIPT
        provenance = [
            {
                **{key: row["lineage"][key] for key in runner._FIVE_IDS},
                "training_row_id": row["training_row_id"],
                "training_row_receipt_sha256": row["receipt_sha256"],
            }
            for row in frozen["rows"]
        ]
        return {
            "frozen_row_source": {
                key: frozen[key]
                for key in (
                    "training_data_manifest_receipt_sha256",
                    "cache_manifest_receipt_sha256",
                    "selection_receipt_sha256",
                    "row_indices",
                    "training_row_ids",
                    "training_row_receipts_sha256",
                )
            },
            "provenance": provenance,
            "input_mode": ["raw" for _ in frozen["rows"]],
            "row_receipts": [
                {
                    "training_row_id": row["training_row_id"],
                    "training_row_receipt_sha256": row["receipt_sha256"],
                }
                for row in frozen["rows"]
            ],
        }

    def initialize(runtime, atlas_channels, model_kwargs, training_config, *, training_run_binding, device):
        trainer_manifest = {
            "receipt_sha256": runner._hash_json(
                {"binding": training_run_binding, "seed": training_config["seed"]}
            ),
            "training_run_binding": runner._plain(training_run_binding),
            "catalogue_binding": runner._plain(runtime.binding),
            "model_kwargs": runner._plain(model_kwargs),
            "training_config": runner._plain(training_config),
            "atlas_channels": atlas_channels,
        }
        return {
            "global_step": 0,
            "manifest": trainer_manifest,
            "provenance_records": [],
            "training_step_ledger": [],
        }

    def checkpoint(state):
        receipt = runner._hash_json(
            {
                "step": state["global_step"],
                "trainer_manifest": state["manifest"]["receipt_sha256"],
            }
        )
        return {
            "receipt_sha256": receipt,
            "global_step": state["global_step"],
            "manifest": state["manifest"],
            "provenance_records": state["provenance_records"],
            "training_step_ledger": state["training_step_ledger"],
            "probabilities_calibrated": False,
            "uncertainty_status": runner.RAW_UNCALIBRATED,
            "learned_dependencies": {
                "model_weights": [],
                "features": [],
                "pseudolabels": [],
            },
        }

    def save(state, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint(state), temporary)
        os.replace(temporary, path)
        return path

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def restore(checkpoint_value, runtime, *, training_run_binding, device):
        assert checkpoint_value["manifest"]["training_run_binding"] == training_run_binding
        return {
            "global_step": checkpoint_value["global_step"],
            "manifest": checkpoint_value["manifest"],
            "provenance_records": list(checkpoint_value["provenance_records"]),
            "training_step_ledger": list(
                checkpoint_value["training_step_ledger"]
            ),
        }

    def train(state, batch):
        train.calls += 1
        step = state["global_step"]
        phase = (
            "proposal-only"
            if step == 0
            else "pose-rerank"
            if step == 1
            else "joint"
        )
        ready = 0 if phase == "joint" else None
        abstained = len(batch["provenance"]) if phase == "joint" else None
        objective = 1.0 / (step + 1)
        output = {
            "schema_version": runner.staged_trainer_v6.STAGED_TRAINER_V6_SCHEMA,
            "step": step,
            "phase": phase,
            "objective": objective,
            "preclip_gradient_norm": objective,
            "optimizer_step_applied": True,
            "catalogue_cell_count": runner.FULL_CATALOGUE_CELL_COUNT_V6,
            "probabilities_calibrated": False,
            "probability_status": runner.RAW_UNCALIBRATED,
            "refinement_ready_row_count": ready,
            "refinement_abstained_row_count": abstained,
            "losses": {"total": objective},
        }
        output["receipt_sha256"] = runner._hash_json(output)
        state["provenance_records"].extend(batch["provenance"])
        ledger = {
            "step": step,
            "phase": phase,
            "input_mode": list(batch.get("input_mode", [])),
            "frozen_row_selection": runner._plain(batch["frozen_row_source"]),
            "row_receipts_sha256": runner._hash_json(
                {
                    "schema_version": runner.staged_trainer_v6.ROW_RECEIPTS_V6_SCHEMA,
                    "row_receipts": runner._plain(batch["row_receipts"]),
                }
            ),
            "trainer_output_receipt_sha256": output["receipt_sha256"],
            "refinement_ready_row_count": ready,
            "refinement_abstained_row_count": abstained,
        }
        ledger["receipt_sha256"] = runner._hash_json(ledger)
        state["training_step_ledger"].append(ledger)
        state["global_step"] += 1
        return output

    train.calls = 0

    monkeypatch.setattr(runner, "_current_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        runner, "_verify_declared_sources_match_git_commit", lambda commit: None
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "verify_bound_allen_atlas_v6",
        lambda bundle: True,
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "resolve_bound_allen_atlas_v6",
        lambda bundle: bundle,
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "verify_pinned_allen_raw_sources_v6",
        lambda binding=None: True,
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "restore_bound_allen_atlas_v6",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        runner.finite_row_binding_v6,
        "verify_finite_psf_model_capability_v6",
        lambda capability: True,
    )
    monkeypatch.setattr(
        runner.finite_row_binding_v6,
        "load_frozen_row_cache_manifest_v6",
        load_manifest,
    )
    monkeypatch.setattr(
        runner.training_data_v6, "load_frozen_training_rows_v6", load_frozen
    )
    monkeypatch.setattr(runner.training_data_v6, "model_ready_rows_v6", model_ready)
    monkeypatch.setattr(
        runner.catalogue_binding_v3,
        "verify_catalogue_binding_v3",
        lambda catalogue: True,
    )
    monkeypatch.setattr(
        runner.catalogue_runtime_v6,
        "verify_complete_catalogue_runtime_v6",
        lambda runtime: True,
    )
    monkeypatch.setattr(
        runner.catalogue_runtime_v6,
        "make_complete_catalogue_runtime_v6",
        lambda *args, **kwargs: _Runtime(),
    )
    monkeypatch.setattr(
        runner.staged_trainer_v6, "initialize_staged_trainer_v6", initialize
    )
    monkeypatch.setattr(runner.staged_trainer_v6, "save_staged_checkpoint_v6", save)
    monkeypatch.setattr(runner.staged_trainer_v6, "load_staged_checkpoint_v6", load)
    monkeypatch.setattr(
        runner.staged_trainer_v6, "restore_staged_trainer_v6", restore
    )
    monkeypatch.setattr(runner.staged_trainer_v6, "train_staged_step_v6", train)


def _initialize(monkeypatch, i_root):
    _install_fakes(monkeypatch)
    cache = i_root / "cache"
    cache.mkdir()
    catalogue = {
        "catalogue_id": _Runtime.binding["catalogue_id"],
        "receipt_sha256": _Runtime.binding["catalogue_receipt_sha256"],
        "support_geometry": {
            "support_mask_receipt": {"shape": [2, 2, 2]},
            "origin_ap_dv_ml_um": [0.0, 0.0, 0.0],
            "voxel_size_ap_dv_ml_um": [25.0, 25.0, 25.0],
        },
    }
    atlas = np.ones((2, 2, 2, 2), dtype=np.float32)
    atlas_receipt = {
        "shape": [2, 2, 2, 2],
        "dtype": "<f4",
        "sha256": "b" * 64,
    }
    binding = runner._with_receipt(
        {
            "schema_version": runner.ATLAS_BINDING_V6_SCHEMA,
            "decoded_atlas_receipt": atlas_receipt,
            "geometry": {
                "spatial_shape_ap_dv_ml": [2, 2, 2],
                "origin_ap_dv_ml_um": [0.0, 0.0, 0.0],
                "voxel_size_ap_dv_ml_um": [25.0, 25.0, 25.0],
            },
            "catalogue": {
                "cell_count": runner.FULL_CATALOGUE_CELL_COUNT_V6,
                "catalogue_id": _Runtime.binding["catalogue_id"],
                "catalogue_receipt_sha256": _Runtime.binding[
                    "catalogue_receipt_sha256"
                ],
            },
        }
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "ATLAS_FLOAT32_RECEIPT_V6",
        atlas_receipt,
    )
    monkeypatch.setattr(
        runner.allen_atlas_v6,
        "ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6",
        {(2, 2): binding["receipt_sha256"]},
    )
    bundle = {
        "schema_version": runner.allen_atlas_v6.ALLEN_ATLAS_BUNDLE_V6_SCHEMA,
        "atlas_volume_float32": atlas,
        "support_index": {"support": "fake"},
        "catalogue": catalogue,
        "binding": binding,
    }
    run = i_root / "run"
    initialized = runner.initialize_receipt_bound_training_run_v6(
        run,
        cache_directory=cache,
        expected_training_data_manifest_receipt_sha256=STABLE_DATA_RECEIPT,
        allen_atlas_bundle=bundle,
        finite_psf_capability={"receipt_sha256": "9" * 64},
        model_kwargs={},
        training_config=_config(),
        runner_config={
            "batch_size": 2,
            "row_selection_seed": 17,
            "archive_checkpoint_interval_steps": 1,
        },
        device="cpu",
        expected_git_commit="a" * 40,
    )
    return run, initialized, run / "inputs" / "atlas_volume_float32.npy"


def _rewrite_step_transaction(run, run_state, report, raw):
    state = json.loads(json.dumps(run_state))
    committed = state["committed_steps"][0]
    report_path = run / committed["report"]["relative_path"]
    raw_path = run / committed["raw_output"]["relative_path"]
    runner._atomic_json(report_path, report)
    runner._atomic_json(raw_path, raw)
    committed["report"]["file_sha256"] = runner._file_sha256(report_path)
    committed["report"]["receipt_sha256"] = report["receipt_sha256"]
    committed["raw_output"]["file_sha256"] = runner._file_sha256(raw_path)
    committed["raw_output"]["receipt_sha256"] = raw["receipt_sha256"]
    transaction_path = run / committed["transaction"]["relative_path"]
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["committed_step"] = {
        key: value for key, value in committed.items() if key != "transaction"
    }
    transaction = runner._with_receipt(runner._payload(transaction))
    runner._atomic_json(transaction_path, transaction)
    committed["transaction"]["file_sha256"] = runner._file_sha256(
        transaction_path
    )
    committed["transaction"]["receipt_sha256"] = transaction[
        "receipt_sha256"
    ]
    state = runner._with_receipt(runner._payload(state))
    runner._atomic_json(run / "run_state.json", state)


def test_initialization_binds_raw_and_decoded_atlas_complete_catalogue_and_static_data_manifest(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    manifest = initialized["manifest"]
    assert Path(initialized["run_directory"]).drive.upper() == "I:"
    assert manifest["training_data"]["training_data_manifest_receipt_sha256"] == STABLE_DATA_RECEIPT
    assert manifest["training_data"]["initial_full_selection_receipt_sha256"] != STABLE_DATA_RECEIPT
    assert manifest["catalogue"]["cell_count"] == 98_304
    assert manifest["atlas"]["binding"]["decoded_atlas_receipt"]["shape"] == [2, 2, 2, 2]
    assert manifest["execution"]["atlas_channels"] == 2
    assert manifest["atlas"]["support_index_relative_path"] == "inputs/allen_support_index.pt"
    assert manifest["initialization"] == "fresh_random_only"
    assert manifest["probability_status"] == "raw_uncalibrated"
    assert manifest["release_qualifying"] is False
    selected = runner.sample_training_row_indices_v6(manifest, 0)
    animals = [
        manifest["training_data"]["ordered_row_identities"][index]["animal_id"]
        for index in selected
    ]
    assert len(animals) == len(set(animals))
    assert (run / "checkpoints" / "resume_slot_0.pt").is_file()
    assert runner.training_data_v6.load_frozen_training_rows_v6.calls == []


def test_unrelated_head_advance_does_not_invalidate_byte_identical_frozen_run(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    monkeypatch.setattr(runner, "_current_git_commit", lambda: "b" * 40)
    restored = runner.load_receipt_bound_training_run_v6(
        run,
        expected_run_manifest_receipt_sha256=initialized["manifest"][
            "receipt_sha256"
        ],
    )
    assert restored["manifest"]["git_commit"] == "a" * 40


def test_optimizer_boundary_rejects_held_out_development_cache(monkeypatch, i_root):
    _install_fakes(monkeypatch, split="development")
    cache = i_root / "development-cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="exact train-split"):
        runner._training_data_record(cache, STABLE_DATA_RECEIPT)


def test_declared_source_bytes_must_match_recorded_git_blob(monkeypatch, i_root):
    source = i_root / "source.py"
    source.write_bytes(b"dirty bytes\n")
    monkeypatch.setattr(runner, "_SOURCE_ROOT", i_root)
    monkeypatch.setattr(runner, "_SOURCE_FILES", ("source.py",))
    monkeypatch.setattr(runner.staged_trainer_v6, "_SOURCE_FILES", ())

    class Result:
        stdout = b"committed bytes\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match="differs from recorded git commit"):
        runner._verify_declared_sources_match_git_commit("a" * 40)


def test_one_step_then_verified_resume_preserves_ids_and_selection_receipts(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    receipt = initialized["manifest"]["receipt_sha256"]
    first = runner.run_receipt_bound_training_steps_v6(
        run, expected_run_manifest_receipt_sha256=receipt
    )
    assert len(runner.training_data_v6.load_frozen_training_rows_v6.calls) == 1
    assert first["run_state"]["global_step"] == 1
    report_path = run / first["run_state"]["committed_steps"][0]["report"]["relative_path"]
    first_report_bytes = report_path.read_bytes()
    report = json.loads(first_report_bytes)
    assert report["selection_receipt_sha256"] != STABLE_DATA_RECEIPT
    assert report["training_data_manifest_receipt_sha256"] == STABLE_DATA_RECEIPT
    assert all(set(runner._FIVE_IDS).issubset(item) for item in report["row_identities"])
    second = runner.run_receipt_bound_training_steps_v6(
        run, expected_run_manifest_receipt_sha256=receipt
    )
    assert second["run_state"]["global_step"] == 2
    assert len(second["run_state"]["immutable_archives"]) == 2
    assert report_path.read_bytes() == first_report_bytes
    assert second["run_state"]["latest_checkpoint"]["relative_path"] == "checkpoints/resume_slot_0.pt"


def test_complete_pre_state_transaction_is_adopted_without_cuda_style_replay(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    receipt = initialized["manifest"]["receipt_sha256"]
    original_atomic_json = runner._atomic_json
    crashed = False

    def fail_before_state(path, value):
        nonlocal crashed
        if Path(path).name == "run_state.json" and value.get("global_step") == 1 and not crashed:
            crashed = True
            raise RuntimeError("injected pre-state crash")
        return original_atomic_json(path, value)

    monkeypatch.setattr(runner, "_atomic_json", fail_before_state)
    with pytest.raises(RuntimeError, match="injected pre-state crash"):
        runner.run_receipt_bound_training_steps_v6(
            run, expected_run_manifest_receipt_sha256=receipt
        )
    assert runner.staged_trainer_v6.train_staged_step_v6.calls == 1
    assert (run / "transactions" / "step_00000000" / "transaction.json").is_file()

    monkeypatch.setattr(runner, "_atomic_json", original_atomic_json)
    recovered = runner.run_receipt_bound_training_steps_v6(
        run, expected_run_manifest_receipt_sha256=receipt
    )
    assert recovered["run_state"]["global_step"] == 1
    assert runner.staged_trainer_v6.train_staged_step_v6.calls == 1


def test_load_rejects_tampered_stored_atlas_and_step_report(monkeypatch, i_root):
    run, initialized, stored_atlas = _initialize(monkeypatch, i_root)
    receipt = initialized["manifest"]["receipt_sha256"]
    original_atlas = stored_atlas.read_bytes()
    stored_atlas.write_bytes(original_atlas + b"tampered")
    with pytest.raises(ValueError, match="stored Allen"):
        runner.load_receipt_bound_training_run_v6(
            run, expected_run_manifest_receipt_sha256=receipt
        )
    stored_atlas.write_bytes(original_atlas)
    context = runner.run_receipt_bound_training_steps_v6(
        run, expected_run_manifest_receipt_sha256=receipt
    )
    report_path = run / context["run_state"]["committed_steps"][0]["report"]["relative_path"]
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact file changed"):
        runner.load_receipt_bound_training_run_v6(run)


def test_load_rejects_re_receipted_false_report_checkpoint_link(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    context = runner.run_receipt_bound_training_steps_v6(
        run,
        expected_run_manifest_receipt_sha256=initialized["manifest"][
            "receipt_sha256"
        ],
    )
    state = context["run_state"]
    committed = state["committed_steps"][0]
    report_path = run / committed["report"]["relative_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["committed_checkpoint_receipt_sha256"] = "0" * 64
    report = runner._with_receipt(runner._payload(report))
    runner._atomic_json(report_path, report)
    committed["committed_checkpoint_receipt_sha256"] = "0" * 64
    committed["report"]["file_sha256"] = runner._file_sha256(report_path)
    committed["report"]["receipt_sha256"] = report["receipt_sha256"]
    state = runner._with_receipt(runner._payload(state))
    runner._atomic_json(run / "run_state.json", state)
    with pytest.raises(ValueError, match="step ledger failed exact replay"):
        runner.load_receipt_bound_training_run_v6(run)


def test_load_rejects_fully_re_receipted_row_evidence_attack(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    context = runner.run_receipt_bound_training_steps_v6(
        run,
        expected_run_manifest_receipt_sha256=initialized["manifest"][
            "receipt_sha256"
        ],
    )
    committed = context["run_state"]["committed_steps"][0]
    report = json.loads(
        (run / committed["report"]["relative_path"]).read_text(encoding="utf-8")
    )
    raw = json.loads(
        (run / committed["raw_output"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    report["row_receipts"][0]["training_row_receipt_sha256"] = "0" * 64
    report["row_receipts_sha256"] = runner._hash_json(
        {
            "schema_version": runner.staged_trainer_v6.ROW_RECEIPTS_V6_SCHEMA,
            "row_receipts": report["row_receipts"],
        }
    )
    report = runner._with_receipt(runner._payload(report))
    _rewrite_step_transaction(run, context["run_state"], report, raw)
    with pytest.raises(ValueError, match="step ledger failed exact replay"):
        runner.load_receipt_bound_training_run_v6(run)


def test_load_rejects_fully_re_receipted_trainer_output_attack(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    context = runner.run_receipt_bound_training_steps_v6(
        run,
        expected_run_manifest_receipt_sha256=initialized["manifest"][
            "receipt_sha256"
        ],
    )
    committed = context["run_state"]["committed_steps"][0]
    report = json.loads(
        (run / committed["report"]["relative_path"]).read_text(encoding="utf-8")
    )
    raw = json.loads(
        (run / committed["raw_output"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    raw["trainer_output"]["phase"] = "joint"
    raw["trainer_output"]["objective"] = 12345.0
    raw["trainer_output"]["losses"]["total"] = 12345.0
    raw["trainer_output"] = runner._with_receipt(
        runner._payload(raw["trainer_output"])
    )
    raw = runner._with_receipt(runner._payload(raw))
    report["trainer_output_receipt_sha256"] = raw["trainer_output"][
        "receipt_sha256"
    ]
    report["raw_output_receipt_sha256"] = raw["receipt_sha256"]
    report = runner._with_receipt(runner._payload(report))
    _rewrite_step_transaction(run, context["run_state"], report, raw)
    with pytest.raises(ValueError, match="step ledger failed exact replay"):
        runner.load_receipt_bound_training_run_v6(run)


def test_load_rejects_re_receipted_raw_and_report_run_id_attack(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    context = runner.run_receipt_bound_training_steps_v6(
        run,
        expected_run_manifest_receipt_sha256=initialized["manifest"][
            "receipt_sha256"
        ],
    )
    committed = context["run_state"]["committed_steps"][0]
    report = json.loads(
        (run / committed["report"]["relative_path"]).read_text(encoding="utf-8")
    )
    raw = json.loads(
        (run / committed["raw_output"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    report["run_id"] = "0" * 64
    raw["run_id"] = "0" * 64
    raw = runner._with_receipt(runner._payload(raw))
    report["raw_output_receipt_sha256"] = raw["receipt_sha256"]
    report = runner._with_receipt(runner._payload(report))
    _rewrite_step_transaction(run, context["run_state"], report, raw)
    with pytest.raises(ValueError, match="step ledger failed exact replay"):
        runner.load_receipt_bound_training_run_v6(run)


def test_batch_with_wrong_static_frozen_source_binding_is_rejected(monkeypatch, i_root):
    run, initialized, _ = _initialize(monkeypatch, i_root)
    original = runner.training_data_v6.model_ready_rows_v6

    def wrong_source(*args, **kwargs):
        batch = original(*args, **kwargs)
        batch["frozen_row_source"]["cache_manifest_receipt_sha256"] = "0" * 64
        return batch

    monkeypatch.setattr(runner.training_data_v6, "model_ready_rows_v6", wrong_source)
    with pytest.raises(ValueError, match="bound frozen cache"):
        runner.run_receipt_bound_training_steps_v6(
            run,
            expected_run_manifest_receipt_sha256=initialized["manifest"][
                "receipt_sha256"
            ],
        )


def test_runner_rejects_non_i_run_root_before_writing(monkeypatch, i_root):
    _install_fakes(monkeypatch)
    with pytest.raises(ValueError, match="must be on I"):
        runner.initialize_receipt_bound_training_run_v6(
            "C:/receipt-bound-v6-must-not-exist",
            cache_directory=i_root,
            expected_training_data_manifest_receipt_sha256=STABLE_DATA_RECEIPT,
            allen_atlas_bundle=None,
            finite_psf_capability={},
            model_kwargs={},
            training_config=_config(),
            runner_config={
                "batch_size": 1,
                "row_selection_seed": 0,
                "archive_checkpoint_interval_steps": 1,
            },
            device="cpu",
        )


def test_default_cuda_alias_matches_the_canonical_runtime_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert runner._same_device("cuda", "cuda:0")
    assert not runner._same_device("cuda", "cuda:1")


def test_clean_subprocess_import_has_no_forbidden_training_modules():
    code = """
import sys
import training.arbitrary_plane_receipt_bound_training_runner_v6
forbidden = (
    'training.arbitrary_plane_training_runner_v3',
    'training.arbitrary_plane_staged_training',
    'training.arbitrary_plane_training_bank_v3',
    'training.arbitrary_plane_candidate_bank',
    'training.arbitrary_plane_legacy_chain_v3',
)
loaded = sorted(name for name in sys.modules if any(token in name for token in forbidden))
print('\\n'.join(loaded))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_clean_subprocess_binds_every_loaded_repository_training_module():
    code = """
from pathlib import Path
import sys
import training.arbitrary_plane_receipt_bound_training_runner_v6 as runner
root = Path(runner.__file__).resolve().parents[1]
declared = set(runner._SOURCE_FILES) | set(runner.staged_trainer_v6._SOURCE_FILES)
loaded = set()
for module in tuple(sys.modules.values()):
    path = getattr(module, '__file__', None)
    if path is None:
        continue
    try:
        relative = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.startswith('training/') and relative.endswith('.py'):
        loaded.add(relative)
print('\\n'.join(sorted(loaded - declared)))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
