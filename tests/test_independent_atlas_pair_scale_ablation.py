import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.run_independent_atlas_pair_scale_ablation as ablation


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "training/configs/independent_oracle_atlas_pair_scale_ablation.json"


def _parent_manifest(count=2):
    manifest = {
        "contract_sha256": "generator-contract",
        "seed": 1104322,
        "ap_index": np.arange(count, dtype=np.float32),
        "ap_um": np.arange(count, dtype=np.float32) * 25,
        "tilt_lr_deg": np.arange(count, dtype=np.float32),
        "tilt_dv_deg": -np.arange(count, dtype=np.float32),
        "scale": np.asarray([0.99, 1.01], dtype=np.float32)[:count],
        "rotation_deg": np.asarray([1.0, -1.0], dtype=np.float32)[:count],
        "translation_xy": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        "noise": np.asarray([0.01, 0.02], dtype=np.float32)[:count],
        "appearance": np.arange(count, dtype=np.int16),
    }
    return ablation._commit_manifest(manifest)


def _receipt_fixture():
    config = ablation._expected_config()
    config["contract_sha256"] = "config-contract"
    config["config_file_sha256"] = "config-file"
    kinds = ["truth", *(["axis"] * 12), *(["joint-global"] * 3)]
    raw = []
    records = []
    scale = np.linspace(0.981, 1.019, 48, dtype=np.float64)
    for item in range(48):
        baseline = np.full(16, 2.0, dtype=np.float32)
        treatment = np.full(16, 2.0, dtype=np.float32)
        baseline[0], baseline[1] = ((0.0, 1.0) if item < 16 else (1.0, 0.0))
        treatment[0], treatment[1] = ((0.0, 1.0) if item < 24 else (1.0, 0.0))
        if item == 0:
            treatment[13] = treatment[0]
        realization_id = f"{item:064x}"
        raw.append(
            {
                "sample_index": item,
                "synthetic_realization_id": realization_id,
                "candidate_kind": kinds,
                "target_index": 0,
                "baseline": {"normal_energy": baseline},
                "treatment": {"normal_energy": treatment},
            }
        )
        records.append(
            {
                "sample_index": item,
                "synthetic_realization_id": realization_id,
                "baseline_scale": float(scale[item]),
                "treatment_scale": 1.0,
                "mutated_fields": ["scale", "manifest_sha256"],
                "moving_raw_uint8_changed": True,
                "source_160x232_changed": True,
                "baseline_realization_manifest_sha256": f"b{item:063x}",
                "treatment_realization_manifest_sha256": f"c{item:063x}",
                "baseline_moving_raw_uint8_sha256": f"d{item:063x}",
                "treatment_moving_raw_uint8_sha256": f"e{item:063x}",
                "baseline_source_160x232_sha256": f"f{item:063x}",
                "treatment_source_160x232_sha256": f"a{item:063x}",
            }
        )
    scored = {
        "correct": {
            "baseline": {
                "normal": 16,
                "broken_atlas_binding": 0,
                "broken_source_pairing": 1,
            },
            "treatment": {
                "normal": 24,
                "broken_atlas_binding": 0,
                "broken_source_pairing": 1,
            },
        },
        "nonfinite_count": 0,
        "invalid_render_count": 0,
        "raw": raw,
    }
    summary = ablation.summarize(scored, {"scale": scale}, config)
    status = ablation.gate_status(
        scored, summary, {"invalid_source_count": 0}, config
    )
    receipt = {
        "format": ablation.FORMAT,
        "purpose": ablation.PURPOSE,
        "role": config["role"],
        "scope": config["interpretation"]["scope"],
        "product5_access": False,
        "qualification_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "training_performed": False,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": ablation.source_hashes(),
        "checkpoint_file_sha256": config["artifacts"]["checkpoint_file_sha256"],
        "checkpoint_model_state_sha256": "1" * 64,
        "development_manifest_file_sha256": config["artifacts"]["development_manifest_file_sha256"],
        "development_manifest_sha256": config["artifacts"]["development_manifest_sha256"],
        "development_result_file_sha256": config["artifacts"]["development_result_file_sha256"],
        "development_result_sha256": config["artifacts"]["development_result_sha256"],
        "generator_contract_sha256": config["artifacts"]["generator_contract_sha256"],
        "atlas_sha256": config["artifacts"]["atlas_sha256"],
        "intervention": config["intervention"],
        "statistics_contract": config["statistics"],
        "training_scale_reference": config["training_scale_reference"],
        "source_realizations": records,
        "intervention_integrity": {
            "all_baseline_scales_differ_from_one": True,
            "all_treatment_scales_equal_one": True,
            "only_scale_and_manifest_commitment_changed": True,
            "all_raw_and_downsampled_source_hashes_changed": True,
        },
        "fixed_candidates": scored,
        "summary": summary,
        "invalid_source_count": 0,
        "status": status,
    }
    receipt["receipt_sha256"] = ablation._canonical_sha256(receipt)
    return config, receipt


def test_frozen_config_is_committed_and_development_only():
    config = ablation.inspect_config(CONFIG)
    assert config["training"] is False
    assert config["qualification_access"] is False
    assert config["product5_access"] is False
    assert config["calibration_access"] is False
    assert config["final_test_access"] is False
    assert config["data"]["role"] == "stored-development-only"
    assert config["data"]["qualification_policy"] == "no-read-no-generation-no-scoring"
    assert Path(config["paths"]["base_run_folder"]) == ablation.BASE_RUN
    assert Path(config["paths"]["output_run_folder"]) == ablation.OUTPUT_RUN


def test_stored_development_contract_loads_without_generating_tensors():
    parent, result = ablation._load_committed_development(ablation.inspect_config(CONFIG))
    assert len(parent["scale"]) == 48
    assert float(np.min(parent["scale"])) == 0.9804836511611938
    assert float(np.max(parent["scale"])) == 1.0191631317138672
    assert not bool(np.equal(parent["scale"], 1.0).any())
    assert result["fixed_candidates"]["correct"] == {
        "normal": 16,
        "broken_atlas_binding": 0,
        "broken_source_pairing": 1,
    }
    raw = result["fixed_candidates"]["raw"]
    assert [value["sample_index"] for value in raw] == list(range(48))
    assert len({value["realization"]["synthetic_realization_id"] for value in raw}) == 48
    baseline, treatment, _ = ablation.paired_scale_manifests(
        parent, 0, result["generator_contract_sha256"]
    )
    assert baseline["manifest_sha256"] == raw[0]["realization"][
        "realization_manifest_sha256"
    ]
    assert float(treatment["scale"][0]) == 1.0


def test_paired_manifest_mutates_only_scale_and_commitment():
    parent = _parent_manifest()
    baseline, treatment, realization = ablation.paired_scale_manifests(
        parent, 1, "generator-contract"
    )
    assert float(baseline["scale"][0]) == pytest.approx(1.01)
    assert float(treatment["scale"][0]) == 1.0
    assert baseline["synthetic_realization_id"] == treatment["synthetic_realization_id"]
    assert baseline["realization_seed"] == treatment["realization_seed"]
    assert realization["synthetic_realization_id"] == baseline["synthetic_realization_id"]
    for name in baseline:
        if name not in {"scale", "manifest_sha256"}:
            assert ablation._canonical(baseline[name]) == ablation._canonical(treatment[name])
    treatment_payload = {
        name: value for name, value in treatment.items() if name != "manifest_sha256"
    }
    assert treatment["manifest_sha256"] == ablation._payload_sha256(treatment_payload)


def test_materialization_rejects_an_intervention_with_unchanged_model_input():
    parent = _parent_manifest(1)
    raw = torch.arange(20 * 30, dtype=torch.uint8).reshape(1, 1, 20, 30)

    class Generator:
        contract = {"contract_sha256": "generator-contract"}

        def batch(self, manifest, qa=False):
            return {"moving_raw_uint8": raw}

    _, realization = ablation.realization_manifest(parent, 0, "generator-contract")
    source = ablation.oracle_source({"moving_raw_uint8": raw})[0]
    realization["moving_raw_uint8_sha256"] = ablation._tensor_sha256(raw)
    realization["source_160x232_sha256"] = ablation._tensor_sha256(source)
    with pytest.raises(RuntimeError, match="observably change"):
        ablation._materialize_pairs(
            Generator(), parent, [{"realization": realization}]
        )


def test_exact_mcnemar_gate_boundary_is_preregistered():
    baseline = np.zeros(48, dtype=bool)
    baseline[:16] = True
    treatment = baseline.copy()
    treatment[16:24] = True
    result = ablation.exact_mcnemar(baseline, treatment)
    assert result == {
        "baseline_wrong_treatment_correct": 8,
        "baseline_correct_treatment_wrong": 0,
        "discordant_count": 8,
        "net_corrections": 8,
        "exact_two_sided_p": 0.0078125,
    }


def test_global_pair_gate_requires_all_144_declared_pairs():
    config = ablation._expected_config()
    scored = {
        "correct": {
            "baseline": {"broken_atlas_binding": 0, "broken_source_pairing": 1},
            "treatment": {"broken_atlas_binding": 0, "broken_source_pairing": 1},
        },
        "nonfinite_count": 0,
        "invalid_render_count": 0,
    }
    summary = {
        "baseline_top1": 16,
        "treatment_top1": 24,
        "mcnemar": {
            "net_corrections": 8,
            "exact_two_sided_p": 0.0078125,
        },
        "median_paired_truth_gap_improvement": 0.25,
        "global_pair_accuracy": 1.0,
        "global_pair_count": 143,
    }
    sources = {"invalid_source_count": 0}
    assert not ablation.gate_status(scored, summary, sources, config)["checks"][
        "global_pair_accuracy"
    ]
    summary["global_pair_count"] = 144
    status = ablation.gate_status(scored, summary, sources, config)
    assert status["passed"]
    assert status["interpretation_branch"] == "all_gates_pass"


def test_summary_uses_paired_gap_sign_strict_global_comparison_and_pair_identities():
    config, receipt = _receipt_fixture()
    summary = receipt["summary"]
    assert summary["global_pair_count"] == 144
    assert summary["global_pair_wins"] == 143
    assert summary["global_pair_accuracy"] == 143 / 144
    assert summary["paired_truth_gap_improvement"][16] == 2.0
    assert receipt["fixed_candidates"]["raw"][16]["baseline_truth_gap"] == -1.0
    assert receipt["fixed_candidates"]["raw"][16]["treatment_truth_gap"] == 1.0
    assert summary["mcnemar"] == {
        "baseline_wrong_treatment_correct": 8,
        "baseline_correct_treatment_wrong": 0,
        "discordant_count": 8,
        "net_corrections": 8,
        "exact_two_sided_p": 0.0078125,
    }
    assert config["statistics"]["paired_truth_gap_improvement"].startswith(
        "treatment_truth_gap_minus"
    )


def test_baseline_energy_replay_is_bit_exact():
    values = torch.tensor([0.25, 0.5], dtype=torch.float32)
    paired = {
        "baseline": {
            name: values.clone()
            for name in (
                "normal_energy",
                "normal_energy8",
                "normal_energy16",
                "broken_atlas_binding_energy",
                "broken_atlas_binding_energy8",
                "broken_atlas_binding_energy16",
                "broken_source_pairing_energy",
                "broken_source_pairing_energy8",
                "broken_source_pairing_energy16",
            )
        }
    }
    stored = {name: values.tolist() for name in paired["baseline"]}
    config = ablation._expected_config()
    ablation._verify_baseline_replay(
        [paired], [stored], {"normal": 16, "broken_atlas_binding": 0, "broken_source_pairing": 1}, config
    )
    changed = copy.deepcopy(paired)
    changed["baseline"]["normal_energy"][0] = torch.nextafter(
        changed["baseline"]["normal_energy"][0], torch.tensor(float("inf"))
    )
    with pytest.raises(RuntimeError, match="bit-exactly"):
        ablation._verify_baseline_replay(
            [changed], [stored], {"normal": 16, "broken_atlas_binding": 0, "broken_source_pairing": 1}, config
        )


def test_receipt_write_is_atomic_and_never_overwrites(tmp_path):
    assert tmp_path.drive.upper() == "I:"
    path = tmp_path / "diagnostic_receipt.json"
    first = {"value": 1}
    ablation._write_immutable_json(path, first)
    with pytest.raises(FileExistsError):
        ablation._write_immutable_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_self_committed_receipt_cannot_relax_access_contract(tmp_path):
    assert tmp_path.drive.upper() == "I:"
    config = ablation._expected_config()
    config["contract_sha256"] = "config-contract"
    config["config_file_sha256"] = "config-file"
    receipt = {
        "format": ablation.FORMAT,
        "purpose": ablation.PURPOSE,
        "qualification_access": True,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": ablation.source_hashes(),
        "checkpoint_file_sha256": config["artifacts"]["checkpoint_file_sha256"],
        "development_manifest_file_sha256": config["artifacts"]["development_manifest_file_sha256"],
        "development_result_file_sha256": config["artifacts"]["development_result_file_sha256"],
    }
    receipt["receipt_sha256"] = ablation._canonical_sha256(receipt)
    path = tmp_path / "diagnostic_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lineage differs"):
        ablation._existing_receipt(path, config)


def test_complete_existing_receipt_is_recomputed_before_acceptance(tmp_path):
    assert tmp_path.drive.upper() == "I:"
    config, receipt = _receipt_fixture()
    path = tmp_path / "diagnostic_receipt.json"
    ablation._write_immutable_json(path, receipt)
    assert ablation._existing_receipt(path, config)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("role", "performance"),
        ("scope", "promotion"),
        ("product5_access", True),
        ("qualification_access", True),
        ("calibration_access", True),
        ("final_test_access", True),
        ("training_performed", True),
        ("config_contract_sha256", "changed"),
        ("config_file_sha256", "changed"),
        ("source_sha256", {}),
        ("checkpoint_file_sha256", "changed"),
        ("development_manifest_file_sha256", "changed"),
        ("development_manifest_sha256", "changed"),
        ("development_result_file_sha256", "changed"),
        ("development_result_sha256", "changed"),
        ("generator_contract_sha256", "changed"),
        ("atlas_sha256", {}),
        ("intervention", {}),
        ("statistics_contract", {}),
        ("training_scale_reference", {}),
    ],
)
def test_recommitted_receipt_cannot_change_a_frozen_header(
    tmp_path, field, changed
):
    assert tmp_path.drive.upper() == "I:"
    config, receipt = _receipt_fixture()
    receipt[field] = changed
    receipt["receipt_sha256"] = ablation._canonical_sha256(
        {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    )
    path = tmp_path / "diagnostic_receipt.json"
    path.write_text(json.dumps(ablation._canonical(receipt)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lineage differs"):
        ablation._existing_receipt(path, config)


@pytest.mark.parametrize("mutation", ["missing-summary", "wrong-status"])
def test_recommitted_receipt_cannot_hide_incomplete_or_wrong_statistics(
    tmp_path, mutation
):
    assert tmp_path.drive.upper() == "I:"
    config, receipt = _receipt_fixture()
    if mutation == "missing-summary":
        del receipt["summary"]
    else:
        receipt["status"]["checks"]["mcnemar"] = False
    receipt["receipt_sha256"] = ablation._canonical_sha256(
        {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    )
    path = tmp_path / "diagnostic_receipt.json"
    path.write_text(json.dumps(ablation._canonical(receipt)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ"):
        ablation._existing_receipt(path, config)
