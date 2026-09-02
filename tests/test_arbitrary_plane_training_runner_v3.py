import hashlib
import json
import copy

import pytest
import torch

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_staged_training as staged_v3
import training.arbitrary_plane_training_runner_v3 as runner_v3
from training.arbitrary_plane_training_bank_v3 import TRAINING_CANDIDATE_BANK_SCOPE


def _prepared(tmp_path, target=2):
    cache = tmp_path / "cache"
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=fixture.generator_binding(),
        generation_config={"row_count": 2, "plane_domain": "all brain-intersecting"},
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(cache, [fixture.row(0), fixture.row(1)])
    frozen = row_cache_v3.freeze_training_row_cache_v3(cache)
    atlas_source = tmp_path / "allen-source.bin"
    atlas_source.write_bytes(b"authenticated Allen source fixture")
    run = tmp_path / "run"
    manifest, state = runner_v3.initialize_training_run_v3(
        run,
        cache_directory=cache,
        expected_generator_binding=fixture.generator_binding(),
        catalogue=fixture.catalogue(),
        atlas_volume=fixture.atlas(),
        atlas_source_assets=(
            {
                "path": str(atlas_source),
                "role": "Allen template and annotation test asset",
                "sha256": hashlib.sha256(atlas_source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "fixed deterministic fixture"},
        model_kwargs=fixture.model_kwargs(),
        training_config=fixture.training_config(),
        runner_config=fixture.runner_config(target),
        device="cpu",
    )
    return run, frozen, manifest, state


def test_runner_resumes_exact_conditional_training_with_atomic_reports(tmp_path):
    run, frozen, manifest, initial_state = _prepared(tmp_path, target=2)
    assert manifest["data_role"] == "development-training"
    assert manifest["retrieval_scope"] == TRAINING_CANDIDATE_BANK_SCOPE
    assert manifest["finite_psf_contract"]["strictly_positive_weights"]
    assert manifest["finite_psf_contract"]["receipt_sha256"]
    assert manifest["row_sampling_policy"]["population"] == (
        "every row in the exact frozen cache manifest"
    )
    assert manifest["row_sampling_policy"]["phase_aware_resampling"] is False
    assert "identity-pose and nonidentity-deformed" in manifest[
        "row_sampling_policy"
    ]["pose_warmup_eligibility"]
    assert manifest["cache"]["manifest_receipt_sha256"] == frozen["receipt_sha256"]
    assert manifest["seed_record"]["model_initialization_seed"] == 173
    assert initial_state["applied_step_count"] == 0

    first = runner_v3.run_training_attempts_v3(run, max_attempts=1)
    assert len(first) == 1
    assert first[0]["retrieval_scope"] == TRAINING_CANDIDATE_BANK_SCOPE
    assert first[0]["training_report"]["optimizer_step_applied"]
    assert first[0]["optimizer_learning_rates_after"] == [2e-3]
    assert first[0]["global_step_after"] == 1
    assert first[0]["row_identity"][0]["animal_id"].startswith("animal-")
    assert first[0]["row_identity"][0]["specimen_id"].startswith("specimen-")
    assert first[0]["row_identity"][0]["experiment_id"].startswith("experiment-")
    assert all(
        receipt["inference_scope"] is False
        for receipt in first[0]["training_candidate_bank_receipts"]
    )

    resumed = runner_v3.load_training_run_v3(run)
    assert resumed["training_state"]["global_step"] == 1
    second = runner_v3.run_training_attempts_v3(run, max_attempts=1)
    assert second[0]["global_step_after"] == 2
    complete = runner_v3.load_training_run_v3(run)
    assert complete["training_state"]["global_step"] == 2
    assert complete["run_state"]["attempt_count"] == 2
    assert len(complete["training_state"]["training_step_ledger"]) == 2
    checkpoint = complete["run_root"] / complete["run_state"]["latest_checkpoint"][
        "relative_path"
    ]
    assert checkpoint.exists()
    frozen_checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    assert "training_candidate_bank_receipts" not in frozen_checkpoint
    assert frozen_checkpoint["training_step_ledger_summary"]["entry_count"] == 2
    export = runner_v3.make_training_run_export_receipt_v3(run)
    assert export["sampled_bank_step_count"] == 2
    assert export["training_report_ledger_evidence"]["applied_step_count"] == 2
    assert {path.name for path in (complete["run_root"] / "checkpoints").iterdir()} == {
        "resume_slot_0.pt",
        "resume_slot_1.pt",
        "archive_step_00000002.pt",
    }


def test_runner_detects_persisted_input_and_report_tampering(tmp_path):
    run, _, manifest, _ = _prepared(tmp_path, target=1)
    atlas_path = run / manifest["atlas"]["relative_path"]
    with atlas_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="atlas input hash"):
        runner_v3.load_training_run_v3(run)

    run_two, _, _, _ = _prepared(tmp_path / "second", target=1)
    report = runner_v3.run_training_attempts_v3(run_two, max_attempts=1)[0]
    state = json.loads((run_two / "run_state.json").read_text(encoding="utf-8"))
    report_path = run_two / state["committed_reports"][0]["relative_path"]
    report_path.write_text(json.dumps({**report, "global_step_after": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="report hash"):
        runner_v3.load_training_run_v3(run_two)


def test_runner_rejects_rehashed_checkpoint_report_ledger_mismatch(tmp_path):
    run, _, _, _ = _prepared(tmp_path, target=1)
    runner_v3.run_training_attempts_v3(run, max_attempts=1)
    run_state_path = run / "run_state.json"
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    checkpoint_path = run / run_state["latest_checkpoint"]["relative_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = {
        key: checkpoint["training_step_ledger"][0][key]
        for key in staged_v3._TRAINING_STEP_PAYLOAD_KEYS
    }
    payload["training_candidate_bank_receipt_sha256"] = ["0" * 64]
    checkpoint["training_step_ledger"] = [
        staged_v3._training_step_ledger_entry_v3(
            checkpoint["binding"], [], payload
        )
    ]
    checkpoint["training_step_ledger_summary"] = (
        staged_v3._training_step_ledger_summary_v3(
            checkpoint["binding"], checkpoint["training_step_ledger"]
        )
    )
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = runner_v3._file_sha256(checkpoint_path)

    record = run_state["committed_reports"][0]
    report_path = run / record["relative_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"]["file_sha256"] = checkpoint_sha
    report_payload = {
        key: value for key, value in report.items() if key != "receipt_sha256"
    }
    report["receipt_sha256"] = runner_v3._hash_json(report_payload)
    runner_v3._atomic_json(report_path, report)
    record["file_sha256"] = runner_v3._file_sha256(report_path)
    record["report_receipt_sha256"] = report["receipt_sha256"]
    run_state["latest_checkpoint"]["file_sha256"] = checkpoint_sha
    run_state_payload = {
        key: value for key, value in run_state.items() if key != "receipt_sha256"
    }
    run_state["receipt_sha256"] = runner_v3._hash_json(run_state_payload)
    runner_v3._atomic_json(run_state_path, run_state)

    with pytest.raises(ValueError, match="checkpoint and training report ledgers differ"):
        runner_v3.load_training_run_v3(run)


def test_runner_requires_frozen_cache_and_i_drive(tmp_path):
    cache = tmp_path / "open-cache"
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=fixture.generator_binding(),
        generation_config={"row_count": 1},
        seed_record={"root_seed": 1},
    )
    row_cache_v3.append_training_rows_v3(cache, [fixture.row(0)])
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    kwargs = {
        "cache_directory": cache,
        "expected_generator_binding": fixture.generator_binding(),
        "catalogue": fixture.catalogue(),
        "atlas_volume": fixture.atlas(),
        "atlas_source_assets": (
            {
                "path": str(source),
                "role": "fixture",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ),
        "atlas_preprocessing": {"normalization": "fixture"},
        "model_kwargs": fixture.model_kwargs(),
        "training_config": fixture.training_config(),
        "runner_config": fixture.runner_config(target=1),
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="frozen"):
        runner_v3.initialize_training_run_v3(tmp_path / "run", **kwargs)
    with pytest.raises(ValueError, match="only I"):
        runner_v3.initialize_training_run_v3("C:\\forbidden-run-v3", **kwargs)


def test_runner_freeze_rejects_invalid_psf_and_topk_bank_mismatch(tmp_path):
    cache = tmp_path / "cache"
    binding = fixture.generator_binding()
    row_cache_v3.initialize_training_row_cache_v3(
        cache,
        generator_binding=binding,
        generation_config={"row_count": 1},
        seed_record={"root_seed": 1},
    )
    row_cache_v3.append_training_rows_v3(cache, [fixture.row(0)])
    row_cache_v3.freeze_training_row_cache_v3(cache)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    base = {
        "cache_directory": cache,
        "expected_generator_binding": binding,
        "catalogue": fixture.catalogue(),
        "atlas_volume": fixture.atlas(),
        "atlas_source_assets": (
            {
                "path": str(source),
                "role": "fixture",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ),
        "atlas_preprocessing": {"normalization": "fixture"},
        "model_kwargs": fixture.model_kwargs(),
        "training_config": fixture.training_config(),
        "runner_config": fixture.runner_config(target=1),
        "device": "cpu",
    }
    zero_weight = copy.deepcopy(base)
    zero_weight["runner_config"]["axial_weights"] = (0.5, 0.0, 0.5)
    with pytest.raises(ValueError, match="normalized PSF"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "zero-weight-run", **zero_weight
        )
    asymmetric = copy.deepcopy(base)
    asymmetric["runner_config"]["axial_offsets_um"] = (-0.5, 0.0, 0.4)
    with pytest.raises(ValueError, match="normalized PSF"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "asymmetric-run", **asymmetric
        )
    oversized_topk = copy.deepcopy(base)
    oversized_topk["training_config"]["top_k"] = 6
    with pytest.raises(ValueError, match="conditional training"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "topk-run", **oversized_topk
        )
    inactive_deformation = copy.deepcopy(base)
    inactive_deformation["runner_config"]["target_applied_steps"] = 2
    inactive_deformation["training_config"]["joint_pose_only_steps"] = 2
    with pytest.raises(ValueError, match="conditional training"):
        runner_v3.initialize_training_run_v3(
            tmp_path / "inactive-deformation-run", **inactive_deformation
        )
