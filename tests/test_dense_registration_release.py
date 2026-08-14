import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.dense_registration_release as release


def passing_metrics(schema):
    values = {}
    for name, rule in schema.items():
        operator, threshold = rule["operator"], rule["threshold"]
        if operator == ">=":
            values[name] = threshold + (1.0 if "count" in name else 0.01)
        elif operator == "<=":
            values[name] = max(0.0, threshold - 0.1)
        else:
            values[name] = threshold
    return values


def passing_reports():
    gates = release.V2_RELEASE_PROTOCOL["gates"]
    overall = passing_metrics(gates["core"]["overall_metrics"])
    shared = gates["core"]["per_stratum"]["shared_metrics"]
    main = {
        "overall": overall,
        "per_stratum": {
            stratum: passing_metrics({
                **shared,
                **gates["core"]["per_stratum"]["metrics"][stratum],
            })
            for stratum in release.synthetic.STRATA
        },
        "appearance_subgroups": {
            mode: {
                "foreground_correspondence": 0.99,
                "macro_region_dice": 0.97,
            }
            for mode in ("template", "label")
        },
    }
    stress = {
        "mask_offset_subgroups": {
            str(offset): {
                "foreground_correspondence": 0.985,
                "macro_region_dice": 0.96,
                "endpoint_p95_px": 2.0,
                "inverse_endpoint_p95_px": 2.0,
                "fold_fraction": 0.0,
                "inverse_fold_fraction": 0.0,
                "jacobian_min": 0.1,
                "inverse_jacobian_min": 0.1,
            }
            for offset in (-3, 3)
        }
    }
    return main, stress


def checkpoint_payload(profile="v2", with_ema=True):
    generator_contract = {"profile": profile, "atlas": "fixture"}
    generator_contract["contract_sha256"] = release.synthetic._payload_sha256(
        generator_contract
    )
    payload = {
        "config": {},
        "model_config": {"channels": [4, 8]},
        "generator_contract": generator_contract,
    }
    if with_ema:
        payload["ema"] = {
            "shadow": {
                "scalar": torch.tensor(3.0),
                "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            }
        }
    return payload


def test_release_gate_keeps_98_percent_overall_and_severity_specific_stratum_floors():
    main, stress = passing_reports()
    assert release.release_gate_report(main, stress)["passed"]

    main["overall"]["foreground_correspondence"] = 0.979
    report = release.release_gate_report(main, stress)
    assert not report["passed"]
    assert not report["core"]["overall"]["checks"]["foreground_correspondence"]

    main, stress = passing_reports()
    main["per_stratum"]["hard"]["foreground_correspondence"] = 0.951
    assert release.release_gate_report(main, stress)["passed"]
    main["per_stratum"]["hard"]["foreground_correspondence"] = 0.949
    report = release.release_gate_report(main, stress)
    assert not report["passed"]
    assert not report["core"]["per_stratum"]["hard"]["checks"]["foreground_correspondence"]


def test_release_gate_requires_both_appearance_groups_and_balanced_mask_stress():
    main, stress = passing_reports()

    del main["appearance_subgroups"]["label"]
    assert not release.release_gate_report(main, stress)["passed"]

    main, stress = passing_reports()
    del stress["mask_offset_subgroups"]["-3"]
    assert not release.release_gate_report(main, stress)["passed"]


def test_release_gate_rejects_empty_damage_coverage_and_per_sample_epe_tail():
    main, stress = passing_reports()
    main["overall"]["damage_pixel_count"] = 0
    assert not release.release_gate_report(main, stress)["passed"]

    main, stress = passing_reports()
    main["overall"]["sample_endpoint_p95_q95_px"] = 3.01
    assert not release.release_gate_report(main, stress)["passed"]


def test_checkpoint_identity_binds_ema_and_rejects_wrong_profile_or_missing_ema(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(checkpoint_payload(), checkpoint)
    identity, loaded = release.checkpoint_identity(checkpoint)
    assert identity["weights"] == "ema"
    assert len(identity["checkpoint_file_sha256"]) == 64
    assert len(identity["ema_state_payload_sha256"]) == 64
    assert loaded["generator_contract"]["profile"] == "v2"

    torch.save(checkpoint_payload(profile="v1"), checkpoint)
    with pytest.raises(ValueError, match="committed v2 generator contract"):
        release.checkpoint_identity(checkpoint)
    torch.save(checkpoint_payload(with_ema=False), checkpoint)
    with pytest.raises(ValueError, match="EMA weights"):
        release.checkpoint_identity(checkpoint)


def test_locked_evidence_is_idempotent_but_cannot_change(tmp_path):
    path = tmp_path / "locked.json"
    assert release._write_once(path, {"value": 1}) == path
    assert release._write_once(path, {"value": 1}) == path
    with pytest.raises(ValueError, match="locked evidence differs"):
        release._write_once(path, {"value": 2})


def test_forced_mask_stress_records_have_committed_minus_or_plus_three_masks():
    class Generator:
        def make_manifest(self, count, split, seed, stratum, _final_capability=None):
            manifest = {
                "contract_sha256": "contract",
                "split": split,
                "ap_index": np.asarray([11]),
                "moving_appearance_mode": np.asarray([seed % 2], dtype=np.uint8),
                "mask_offset_px": np.asarray([0], dtype=np.int8),
            }
            manifest["manifest_sha256"] = release.synthetic._payload_sha256(manifest)
            return manifest

    records = release._records(
        Generator().make_manifest, split="validation", seed=71,
        samples_per_stratum=2, forced_mask_offset=-3,
    )
    assert len(records) == 2 * len(release.synthetic.STRATA)
    assert {record["mask_offset_px"] for record in records} == {-3}
    for record in records:
        manifest = record["manifest"]
        unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        assert manifest["manifest_sha256"] == release.synthetic._payload_sha256(unhashed)


def test_locked_manifest_is_bound_to_generator_and_exactly_regenerated(monkeypatch):
    monkeypatch.setattr(release, "QUALIFICATION_SAMPLES_PER_STRATUM", 1)
    monkeypatch.setattr(release, "MASK_STRESS_SAMPLES_PER_STRATUM", 1)
    contract = {"profile": "v2", "atlas": "fixture"}
    contract["contract_sha256"] = release.synthetic._payload_sha256(contract)
    identity = {
        "checkpoint_file_sha256": "a" * 64,
        "ema_state_payload_sha256": "b" * 64,
        "weights": "ema",
        "model_config_payload_sha256": "c" * 64,
        "generator_contract_payload_sha256": release.canonical_json_sha256(contract),
    }
    class Generator:
        def __init__(self, generator_contract):
            self.contract = generator_contract

        def make_manifest(self, count, split, seed, stratum, _final_capability=None):
            return release.synthetic.make_registration_manifest(
                self.contract, count, split, seed, stratum,
                _final_capability=_final_capability,
            )

    generator = Generator(contract)
    cohorts = release._cohorts(generator.make_manifest, "validation", 97, 1, 1)
    manifest = release._locked_manifest(identity, generator, "validation", 97, cohorts)
    release._verify_manifest(manifest, identity, contract, "validation", 97)

    manifest["cohorts"]["main"][0]["manifest_sha256"] = "f" * 64
    manifest["manifest_payload_sha256"] = release.canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    )
    with pytest.raises(ValueError, match="exactly regenerate"):
        release._verify_manifest(manifest, identity, contract, "validation", 97)


def test_qualification_runs_fresh_evaluation_for_both_locked_seeds(tmp_path, monkeypatch):
    identity = {
        "checkpoint_file_sha256": "a" * 64,
        "ema_state_payload_sha256": "b" * 64,
        "weights": "ema",
        "model_config_payload_sha256": "c" * 64,
        "generator_contract_payload_sha256": "d" * 64,
    }

    class Generator:
        contract = {"profile": "v2", "contract_sha256": "contract"}
        make_manifest = None

    calls = []
    main, stress = passing_reports()
    main.update(sample_evidence=[], sample_evidence_payload_sha256=release.canonical_json_sha256([]))
    stress.update(sample_evidence=[], sample_evidence_payload_sha256=release.canonical_json_sha256([]))
    monkeypatch.setattr(
        release, "_load_release_components",
        lambda *_args, **_kwargs: (identity, {}, object(), Generator()),
    )
    monkeypatch.setattr(release, "checkpoint_identity", lambda _path: (identity, {}))
    monkeypatch.setattr(release, "deterministic_environment", lambda _device: {"device": "cpu"})
    monkeypatch.setattr(release, "source_hashes", lambda: {"source": "hash"})
    monkeypatch.setattr(
        release, "_cohorts",
        lambda _generator, _split, seed, _main, _stress: {
            "main": [], "mask-minus-3": [], "mask-plus-3": [], "seed": seed
        },
    )

    def evaluate(_model, _generator, cohorts, _batch_size, final):
        calls.append((cohorts["seed"], final))
        return json.loads(json.dumps(main)), json.loads(json.dumps(stress))

    monkeypatch.setattr(release, "_evaluate_cohorts", evaluate)
    monkeypatch.setattr(
        release, "_locked_manifest",
        lambda _identity, _generator, split, seed, _cohorts: {
            "format_version": 2,
            "split": split,
            "seed": seed,
            "manifest_payload_sha256": str(seed),
        },
    )
    receipt_path = release.qualify_checkpoint(
        tmp_path / "ignored.pt", tmp_path / "evidence", device="cpu"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == [(seed, False) for seed in release.QUALIFICATION_SEEDS]
    assert receipt["status"] == "passed"
    assert receipt["qualification_seeds"] == list(release.QUALIFICATION_SEEDS)
    for seed in release.QUALIFICATION_SEEDS:
        assert (receipt_path.parent / f"qualification-{seed}.manifest.json").is_file()
        assert (receipt_path.parent / f"qualification-{seed}.result.json").is_file()


def test_qualification_and_freeze_verify_existing_artifacts_before_new_timestamps(tmp_path, monkeypatch):
    identity = {
        "checkpoint_file_sha256": "a" * 64,
        "ema_state_payload_sha256": "b" * 64,
        "weights": "ema",
        "model_config_payload_sha256": "c" * 64,
        "generator_contract_payload_sha256": "d" * 64,
    }
    receipt = tmp_path / "evidence" / identity["checkpoint_file_sha256"] / "qualification-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    verified = []
    monkeypatch.setattr(release, "checkpoint_identity", lambda _path: (identity, {}))
    monkeypatch.setattr(
        release, "verify_qualification",
        lambda _checkpoint, path: verified.append(Path(path)) or {},
    )
    monkeypatch.setattr(
        release, "deterministic_environment",
        lambda _device: (_ for _ in ()).throw(AssertionError("must not rerun")),
    )
    assert release.qualify_checkpoint("checkpoint.pt", tmp_path / "evidence") == receipt
    assert verified == [receipt]

    workspace = tmp_path / "workspace"
    candidate_path = workspace / "frozen-candidates-v2" / identity["checkpoint_file_sha256"] / "candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        release, "_verified_candidate",
        lambda path: ({"identity": identity}, candidate_path.parent / "model.pt"),
    )
    monkeypatch.setattr(
        release, "_utc_now",
        lambda: (_ for _ in ()).throw(AssertionError("must not create timestamp")),
    )
    assert release.freeze_qualified_candidate("checkpoint.pt", receipt, workspace) == candidate_path


def test_failed_sealed_attempt_consumes_claim_before_model_inference(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    candidate = {
        "candidate_payload_sha256": "c" * 64,
        "identity": {"checkpoint_file_sha256": "d" * 64},
    }
    calls = 0

    def fail_load(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("deliberate inference failure")

    monkeypatch.setattr(release, "_verified_candidate", lambda _path: (candidate, tmp_path / "model.pt"))
    monkeypatch.setattr(release, "_load_release_components", fail_load)
    monkeypatch.setattr(release, "deterministic_environment", lambda _device: {"device": "cpu"})
    monkeypatch.setattr(release.secrets, "randbelow", lambda _maximum: 12345)
    monkeypatch.setattr(release, "_utc_now", lambda: "2026-08-14T00:00:00Z")
    state = tmp_path / "sealed"
    monkeypatch.setattr(release, "_LOCAL_SEALED_STATE_ROOT", state)
    with pytest.raises(RuntimeError, match="deliberate inference failure"):
        release.run_sealed_benchmark(
            candidate_path, confirmation=release.SEALED_CONFIRMATION,
            device="cpu",
        )
    claim = json.loads((state / "claim.json").read_text(encoding="utf-8"))
    receipt = json.loads((state / "receipt.json").read_text(encoding="utf-8"))
    assert claim["seed"] == 12345
    assert receipt["status"] == "failed"
    with pytest.raises(RuntimeError, match="already been consumed"):
        release.run_sealed_benchmark(
            candidate_path, confirmation=release.SEALED_CONFIRMATION,
            device="cpu",
        )
    assert calls == 1


def test_deterministic_setup_failure_is_receipted_and_consumes_sealed_state(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    candidate = {
        "candidate_payload_sha256": "c" * 64,
        "identity": {"checkpoint_file_sha256": "d" * 64},
    }
    state = tmp_path / "sealed"
    monkeypatch.setattr(release, "_LOCAL_SEALED_STATE_ROOT", state)
    monkeypatch.setattr(release, "_verified_candidate", lambda _path: (candidate, tmp_path / "model.pt"))
    monkeypatch.setattr(
        release, "deterministic_environment",
        lambda _device: (_ for _ in ()).throw(RuntimeError("determinism unavailable")),
    )
    with pytest.raises(RuntimeError, match="determinism unavailable"):
        release.run_sealed_benchmark(
            candidate_path, confirmation=release.SEALED_CONFIRMATION, device="cpu"
        )
    receipt = json.loads((state / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["environment"]["deterministic_setup"] is False
    with pytest.raises(RuntimeError, match="already been consumed"):
        release.run_sealed_benchmark(
            candidate_path, confirmation=release.SEALED_CONFIRMATION, device="cpu"
        )


def test_export_checks_passing_sealed_receipt_before_writing_onnx(tmp_path, monkeypatch):
    called = False

    def export_sentinel(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        release, "verify_passing_sealed_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("receipt required")),
    )
    monkeypatch.setattr(torch.onnx, "export", export_sentinel)
    with pytest.raises(RuntimeError, match="receipt required"):
        release.export_onnx_release(
            tmp_path / "candidate.json", tmp_path / "output", device="cpu"
        )
    assert not called
    assert not (tmp_path / "output").exists()


def test_export_publishes_staged_pair_atomically_and_reuses_verified_bundle(tmp_path, monkeypatch):
    commitment = "7" * 64
    identity = {"checkpoint_file_sha256": "4" * 64}
    candidate = {"candidate_payload_sha256": commitment, "identity": identity}
    sealed = {"seed": 1}
    sealed_evidence = {"claim": {}}
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    exports = []
    export_kwargs = []
    export_model = torch.nn.Identity()
    monkeypatch.setattr(
        release,
        "verify_passing_sealed_receipt",
        lambda _path: (candidate, sealed, sealed_evidence),
    )
    monkeypatch.setattr(release, "_verified_candidate", lambda _path: (candidate, tmp_path / "model.pt"))
    monkeypatch.setattr(
        release,
        "_load_release_components",
        lambda *_args, **_kwargs: (
            identity,
            {"model_config": {}, "generator_contract": {}},
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        release,
        "_onnx_parity_inputs",
        lambda _device: (torch.zeros(1), torch.zeros(1)),
    )
    monkeypatch.setattr(
        release.training,
        "model_from_checkpoint",
        lambda _path, device, *, use_ema: (export_model, {})
        if device == "cpu" and use_ema
        else (_ for _ in ()).throw(AssertionError("CPU EMA export expected")),
    )

    def fake_export(_model, _inputs, path, **_kwargs):
        assert _model is export_model
        exports.append(Path(path))
        export_kwargs.append(_kwargs)
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(release, "run_onnx_parity", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(release, "deterministic_environment", lambda _device: {})
    monkeypatch.setattr(
        release,
        "build_v2_release_metadata",
        lambda **_kwargs: {"candidate_payload_sha256": commitment},
    )

    def verify(model_path, metadata_path):
        assert Path(model_path).is_file() and Path(metadata_path).is_file()
        return {"candidate_payload_sha256": commitment}

    monkeypatch.setattr(release, "verify_dense_registration_v2_bundle", verify)
    output = tmp_path / "published"
    first = release.export_onnx_release(candidate_path, output, device="cuda")
    assert first == (
        output / "dense_registration.onnx",
        output / "dense_registration.metadata.json",
    )
    assert len(exports) == 1
    assert "dynamic_axes" not in export_kwargs[0]
    assert not any(path.name.startswith(".published-") for path in tmp_path.iterdir())

    assert release.export_onnx_release(candidate_path, output, device="cuda") == first
    assert len(exports) == 1


def test_export_publish_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    commitment = "7" * 64
    identity = {"checkpoint_file_sha256": "4" * 64}
    candidate = {"candidate_payload_sha256": commitment, "identity": identity}
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        release, "verify_passing_sealed_receipt",
        lambda _path: (candidate, {"seed": 1}, {"claim": {}}),
    )
    monkeypatch.setattr(
        release, "_verified_candidate",
        lambda _path: (candidate, tmp_path / "model.pt"),
    )
    monkeypatch.setattr(
        release, "_load_release_components",
        lambda *_args, **_kwargs: (
            identity, {"model_config": {}, "generator_contract": {}}, object(), object()
        ),
    )
    monkeypatch.setattr(
        release, "_onnx_parity_inputs",
        lambda _device: (torch.zeros(1), torch.zeros(1)),
    )
    monkeypatch.setattr(
        torch.onnx, "export",
        lambda _model, _inputs, path, **_kwargs: Path(path).write_bytes(b"onnx"),
    )
    monkeypatch.setattr(release, "run_onnx_parity", lambda *_args: {"passed": True})
    monkeypatch.setattr(release, "deterministic_environment", lambda _device: {})
    monkeypatch.setattr(
        release, "build_v2_release_metadata",
        lambda **_kwargs: {"candidate_payload_sha256": commitment},
    )
    monkeypatch.setattr(
        release, "verify_dense_registration_v2_bundle",
        lambda *_args: {"candidate_payload_sha256": commitment},
    )
    destination = tmp_path / "published"
    original_replace = release.os.replace

    def fail_publish(source, target):
        if Path(target) == destination:
            raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(release.os, "replace", fail_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        release.export_onnx_release(candidate_path, destination, device="cpu")
    assert not destination.exists()
    assert not any(path.name.startswith(".published-") for path in tmp_path.iterdir())
