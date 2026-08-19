"""Run one frozen-checkpoint, development-only paired source-scale ablation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from training.independent_atlas_pair_energy import AtlasPairEnergyModel, parameter_count
from training.run_independent_atlas_pair_energy import (
    ROOT,
    _canonical,
    _canonical_bytes,
    _canonical_sha256,
    _commit_manifest,
    _derangement,
    _payload_sha256,
    _pose_from_manifest,
    _sha256,
    _tensor_sha256,
    candidate_pose_table,
    oracle_source,
    realization_manifest,
    render_candidate_poses,
)
from training.synthetic_registration import SyntheticRegistrationGenerator


PURPOSE = "development-only-paired-source-scale-causal-diagnostic"
FORMAT = "independent-atlas-pair-source-scale-ablation-receipt-v1"
BASE_RUN = Path(
    "I:/AtlasJointProject/runs/AtlasJointTraining/"
    "independent-oracle-atlas-pair-energy-1500-v1"
)
OUTPUT_RUN = Path(
    "I:/AtlasJointProject/runs/AtlasJointTraining/"
    "independent-oracle-atlas-pair-source-scale-1p0-ablation-v1"
)
SOURCE_FILES = (
    "training/run_independent_atlas_pair_scale_ablation.py",
    "training/independent_atlas_pair_energy.py",
    "training/run_independent_atlas_pair_energy.py",
    "training/synthetic_registration.py",
    "training/quicknii_plane_metric.py",
    "source/dense_registration_preprocessing.py",
)
BASE_SOURCE_SHA256 = {
    "source/dense_registration_preprocessing.py": "cad4a5fdbaa4be638abe6dbaf1add2c95c74bb10d4e2d851b2a8ea8682cf4f60",
    "training/independent_atlas_pair_energy.py": "6187cb051d048d1e5eec3137b9edc6ac09706cecffcf989507951708681589ec",
    "training/quicknii_plane_metric.py": "fd3f3bc0a1f2e0b57b33db706c2d5d73134b3525bdc6989dfb4bcf2defd6dc31",
    "training/run_independent_atlas_pair_energy.py": "21c73f88a48ca87ac0a44ff022993eea5dc2cfbb1f8c72237ec4b05fa4445b19",
    "training/synthetic_registration.py": "5a7274b56cdfa95fdb410b1b441325c4ee9ab3b15eb8076be2781e92c00c1dde",
}
ATLAS_SHA256 = {
    "average_template_sha256": "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b",
    "annotation_sha256": "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42",
    "query_sha256": "5347daf90e02ac1d1cfcbf9c8af86ff23a2fb32cd7e7a2ba2881951931286dbd",
}


def source_hashes() -> dict[str, str]:
    return {name: _sha256(ROOT / name) for name in SOURCE_FILES}


def _expected_config() -> dict:
    return {
        "schema_version": 1,
        "frozen": True,
        "name": OUTPUT_RUN.name,
        "purpose": PURPOSE,
        "role": "causal-diagnostic-not-performance-or-promotion",
        "product5_access": False,
        "qualification_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "training": False,
        "device": "cuda",
        "paths": {
            "base_run_folder": str(BASE_RUN).replace("\\", "/"),
            "output_run_folder": str(OUTPUT_RUN).replace("\\", "/"),
            "atlas_repo_relative": "data/Allen Brain Atlas 25um",
            "checkpoint_file": "final_checkpoint.pt",
            "development_manifest_file": "development_manifest.json",
            "development_result_file": "development_1500.json",
            "receipt_file": "diagnostic_receipt.json",
        },
        "artifacts": {
            "checkpoint_file_sha256": "1f451b0dfff79629b46fdd0bd7149c9a722be4e6c456fa8d714b0291e0f422ce",
            "development_manifest_file_sha256": "b3b3cdd046519a6e79bd50a6cf1c075e569d50e9c71df8dee22bf94e46c93722",
            "development_manifest_sha256": "c77d427ce5a3e5efbc98677daf11743643446873528643c20f629ff457a3385c",
            "development_result_file_sha256": "89245f1c0fdc6094a4841ca89e95941a33e970d98c7790e1e61df4fdc912dd0c",
            "development_result_sha256": "f9cc842c1f54e8cbe3e893e52a4ef601c0c3b94b5e4f8e0cdeaf96dd3cb75df3",
            "base_config_contract_sha256": "506ebbb88857bd4242939220d0c98e3c5ac99c389690d051c2cab36adac89ccb",
            "base_config_file_sha256": "8e747c82ba8f477c0a71ae803cf214e052e14c4b824c1c9859c6e1b00ad061e2",
            "base_training_history_sha256": "55d202fa695486564d592a7d53f8e931bac54258ad14237e8a665fe193785fe9",
            "base_data_lineage_sha256": "9b14f3e0395166e26cbc662515c675f336f514db623f13d2609fdd584b42c032",
            "generator_contract_sha256": "5ff5a29ccc7f0c554020dfc8e7e07d2af59a662f9420422bffc9a86edcd73872",
            "atlas_sha256": ATLAS_SHA256,
            "base_source_sha256": BASE_SOURCE_SHA256,
        },
        "data": {
            "role": "stored-development-only",
            "seed": 1104322,
            "count": 48,
            "singleton_realizations": True,
            "source_tensor": "moving_raw_uint8_pre_source_view_downsampled_bilinear_160x232",
            "source_masks": "absent",
            "candidate_count": 16,
            "candidate_chunk_size": 8,
            "batch_size": 2,
            "qualification_policy": "no-read-no-generation-no-scoring",
        },
        "intervention": {
            "name": "do(source_scale=1.0)",
            "baseline_scale_range": [0.9804836511611938, 1.0191631317138672],
            "treatment_scale": 1.0,
            "mutated_child_fields": ["scale", "manifest_sha256"],
            "preserved_pairing_identity": "synthetic_realization_id",
            "preserved_generator_fields": "all_except_scale_and_manifest_sha256",
            "same_rotation_translation_deformation_appearance_noise": True,
            "same_frozen_model_and_rendered_candidate_tensors": True,
        },
        "statistics": {
            "truth_gap": "minimum_nontruth_energy_minus_truth_energy",
            "paired_truth_gap_improvement": "treatment_truth_gap_minus_baseline_truth_gap",
            "mcnemar": "exact_two_sided_binomial_on_discordant_top1_pairs",
            "mcnemar_net_corrections": "baseline_wrong_treatment_correct_minus_baseline_correct_treatment_wrong",
            "global_pair_accuracy": "truth_energy_strictly_less_than_each_of_three_joint_global_energies_over_144_pairs",
            "scale_strata": "low_if_abs_log_original_scale_le_development_median_else_high",
        },
        "gates": {
            "baseline_source_hashes_exact": True,
            "baseline_energies_bit_exact": True,
            "baseline_top1_exact": 16,
            "treatment_top1_minimum": 24,
            "net_corrections_minimum": 8,
            "mcnemar_exact_two_sided_p_maximum": 0.01,
            "median_paired_truth_gap_improvement_minimum": 0.25,
            "global_pair_accuracy_minimum": 0.97,
            "broken_atlas_correct_maximum": 12,
            "broken_source_correct_maximum": 12,
            "nonfinite_count_maximum": 0,
            "invalid_source_or_render_count_maximum": 0,
        },
        "interpretation": {
            "all_gates_pass": "source-scale nuisance causally suppresses local discrimination; preregister learned or marginalized scale handling next",
            "replay_gate_fails": "diagnostic invalid because the frozen baseline did not replay exactly",
            "causal_gates_fail": "scale association is not a sufficient causal explanation; prioritize a separate spatial-aggregation diagnostic",
            "specificity_gate_fails": "reject apparent correction as nonspecific or degenerate scoring",
            "scope": "development-only mechanism evidence; no performance, promotion, benchmark, or deployment claim",
        },
        "training_scale_reference": {
            "manifest_file_sha256": "c6ce463b728175561e53d2f424ca5cc4812dfe9d78026ff9834462408f32e445",
            "count": 2048,
            "observed_range": [0.980411946773529, 1.0199992656707764],
            "observed_median_abs_log_scale": 0.0098871793436826,
            "same_clean_generator_distribution_as_development": True,
            "runtime_access": False,
        },
        "lineage": {"source_sha256": source_hashes()},
    }


def inspect_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    if config_path.drive.upper() != "I:":
        raise ValueError("scale ablation config must be on I:")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    commitment = raw.pop("contract_sha256", None)
    if commitment != _canonical_sha256(raw):
        raise ValueError("scale ablation config commitment differs from its payload")
    if raw != _expected_config():
        raise ValueError("scale ablation frozen contract changed")
    raw["contract_sha256"] = commitment
    raw["config_file_sha256"] = _sha256(config_path)
    raw["config_path"] = str(config_path)
    return raw


def _load_committed_development(config: dict) -> tuple[dict, dict]:
    paths = config["paths"]
    artifacts = config["artifacts"]
    base = Path(paths["base_run_folder"])
    if base.resolve() != BASE_RUN.resolve() or base.drive.upper() != "I:":
        raise RuntimeError("development replay root changed")
    manifest_path = base / paths["development_manifest_file"]
    result_path = base / paths["development_result_file"]
    if (
        manifest_path.name != "development_manifest.json"
        or result_path.name != "development_1500.json"
        or _sha256(manifest_path) != artifacts["development_manifest_file_sha256"]
        or _sha256(result_path) != artifacts["development_result_file_sha256"]
    ):
        raise RuntimeError("stored development artifact hash changed")
    manifest_receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_commitment = manifest_receipt.pop("manifest_file_payload_sha256", None)
    if manifest_commitment != _canonical_sha256(manifest_receipt):
        raise RuntimeError("stored development manifest receipt commitment changed")
    parent = manifest_receipt.get("manifest", {})
    parent_payload = {name: value for name, value in parent.items() if name != "manifest_sha256"}
    if (
        parent.get("manifest_sha256") != artifacts["development_manifest_sha256"]
        or parent.get("manifest_sha256") != _payload_sha256(parent_payload)
        or manifest_receipt.get("role") != "development"
        or manifest_receipt.get("generator_contract_sha256") != artifacts["generator_contract_sha256"]
        or manifest_receipt.get("config_contract_sha256") != artifacts["base_config_contract_sha256"]
        or manifest_receipt.get("config_file_sha256") != artifacts["base_config_file_sha256"]
        or manifest_receipt.get("source_sha256") != artifacts["base_source_sha256"]
        or int(parent.get("seed", -1)) != config["data"]["seed"]
        or len(parent.get("ap_index", [])) != config["data"]["count"]
    ):
        raise RuntimeError("stored development manifest lineage changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload = {
        name: value
        for name, value in result.items()
        if name not in {"result_sha256", "status", "update"}
    }
    if (
        result.get("result_sha256") != artifacts["development_result_sha256"]
        or result.get("result_sha256") != _canonical_sha256(result_payload)
        or result.get("update") != 1500
        or result.get("seed") != config["data"]["seed"]
        or result.get("sample_count") != config["data"]["count"]
        or result.get("panel_manifest_sha256") != parent["manifest_sha256"]
        or result.get("generator_contract_sha256") != artifacts["generator_contract_sha256"]
        or result.get("atlas_sha256") != artifacts["atlas_sha256"]
        or result.get("fixed_candidates", {}).get("correct", {}).get("normal")
        != config["gates"]["baseline_top1_exact"]
        or len(result.get("fixed_candidates", {}).get("raw", [])) != config["data"]["count"]
    ):
        raise RuntimeError("stored terminal development result lineage changed")
    return _array_manifest(parent), result


def _array_manifest(manifest: dict) -> dict:
    count = len(manifest["ap_index"])
    return {
        name: (
            np.asarray(value, dtype=np.uint64)
            if name == "label_style_seed"
            else np.asarray(value)
        )
        if isinstance(value, list) and len(value) == count
        else value
        for name, value in manifest.items()
    }


def _load_checkpoint(config: dict, device: torch.device) -> tuple[AtlasPairEnergyModel, dict]:
    checkpoint_path = Path(config["paths"]["base_run_folder"]) / config["paths"]["checkpoint_file"]
    if (
        checkpoint_path.resolve() != (BASE_RUN / "final_checkpoint.pt").resolve()
        or checkpoint_path.drive.upper() != "I:"
        or _sha256(checkpoint_path) != config["artifacts"]["checkpoint_file_sha256"]
    ):
        raise RuntimeError("frozen checkpoint path or hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lineage = checkpoint.get("data_lineage", {})
    lineage_payload = {name: value for name, value in lineage.items() if name != "data_lineage_sha256"}
    generator_contract = lineage.get("generator_contract", {})
    generator_payload = {
        name: value for name, value in generator_contract.items() if name != "contract_sha256"
    }
    if (
        checkpoint.get("format") != "independent-atlas-pair-energy-final-v1"
        or checkpoint.get("update") != 1500
        or checkpoint.get("config_contract_sha256") != config["artifacts"]["base_config_contract_sha256"]
        or checkpoint.get("config_file_sha256") != config["artifacts"]["base_config_file_sha256"]
        or checkpoint.get("source_sha256") != config["artifacts"]["base_source_sha256"]
        or checkpoint.get("training_history_sha256") != config["artifacts"]["base_training_history_sha256"]
        or checkpoint.get("training_history_sha256") != _canonical_sha256(checkpoint.get("training_history", []))
        or lineage.get("data_lineage_sha256") != config["artifacts"]["base_data_lineage_sha256"]
        or lineage.get("data_lineage_sha256") != _canonical_sha256(lineage_payload)
        or lineage.get("generator_contract_sha256") != config["artifacts"]["generator_contract_sha256"]
        or generator_contract.get("contract_sha256") != _payload_sha256(generator_payload)
        or lineage.get("atlas_sha256") != config["artifacts"]["atlas_sha256"]
        or len(checkpoint.get("training_history", [])) != 1500
        or checkpoint.get("learned_checkpoint_dependencies") != []
    ):
        raise RuntimeError("frozen checkpoint lineage changed")
    model = AtlasPairEnergyModel().to(device)
    if parameter_count(model) != 271450:
        raise RuntimeError("frozen model architecture changed")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def paired_scale_manifests(
    parent: dict, sample_index: int, generator_contract_sha256: str
) -> tuple[dict, dict, dict]:
    baseline, realization = realization_manifest(parent, sample_index, generator_contract_sha256)
    treatment = copy.deepcopy(baseline)
    original = np.asarray(treatment["scale"])
    if original.shape != (1,) or not np.issubdtype(original.dtype, np.floating):
        raise RuntimeError("source scale field is not the frozen singleton floating value")
    treatment["scale"] = np.ones_like(original)
    treatment = _commit_manifest(treatment)
    changed = [
        name
        for name in baseline
        if _canonical(baseline[name]) != _canonical(treatment[name])
    ]
    if set(changed) != {"manifest_sha256", "scale"} or treatment["synthetic_realization_id"] != baseline["synthetic_realization_id"]:
        raise RuntimeError("paired intervention changed a field other than scale and its commitment")
    return baseline, treatment, realization


def _materialize_pairs(
    generator: SyntheticRegistrationGenerator,
    parent: dict,
    stored_raw: list[dict],
) -> dict:
    baseline_sources, treatment_sources, records = [], [], []
    invalid_source = 0
    contract = generator.contract["contract_sha256"]
    if bool(np.equal(np.asarray(parent["scale"], dtype=np.float64), 1.0).any()):
        raise RuntimeError("scale intervention requires every baseline scale to differ from 1.0")
    for sample_index, stored in enumerate(stored_raw):
        baseline_manifest, treatment_manifest, realization = paired_scale_manifests(
            parent, sample_index, contract
        )
        if realization != {
            name: stored["realization"][name]
            for name in realization
        }:
            raise RuntimeError("stored singleton realization lineage changed")
        baseline_pair = generator.batch(baseline_manifest, qa=True)
        treatment_pair = generator.batch(treatment_manifest, qa=True)
        baseline, baseline_mask, baseline_available = oracle_source(baseline_pair)
        treatment, treatment_mask, treatment_available = oracle_source(treatment_pair)
        baseline_raw_hash = _tensor_sha256(baseline_pair["moving_raw_uint8"])
        baseline_source_hash = _tensor_sha256(baseline)
        treatment_raw_hash = _tensor_sha256(treatment_pair["moving_raw_uint8"])
        treatment_source_hash = _tensor_sha256(treatment)
        if (
            baseline_manifest["manifest_sha256"] != stored["realization"]["realization_manifest_sha256"]
            or baseline_raw_hash != stored["realization"]["moving_raw_uint8_sha256"]
            or baseline_source_hash != stored["realization"]["source_160x232_sha256"]
        ):
            raise RuntimeError("baseline singleton source hash did not replay exactly")
        if (
            float(np.asarray(treatment_manifest["scale"])[0]) != 1.0
            or treatment_raw_hash == baseline_raw_hash
            or treatment_source_hash == baseline_source_hash
        ):
            raise RuntimeError("scale intervention did not observably change the paired source")
        source_values = (baseline, treatment)
        invalid_source += sum(
            int(
                value.shape != (1, 1, 160, 232)
                or not bool(torch.isfinite(value).all())
                or not bool(value.ne(0).any())
            )
            for value in source_values
        )
        if any(
            bool(value.any())
            for value in (
                baseline_mask,
                baseline_available,
                treatment_mask,
                treatment_available,
            )
        ):
            raise RuntimeError("oracle scale diagnostic requires absent source masks")
        baseline_sources.append(baseline.cpu())
        treatment_sources.append(treatment.cpu())
        records.append(
            {
                "sample_index": sample_index,
                "synthetic_realization_id": realization["synthetic_realization_id"],
                "baseline_realization_manifest_sha256": baseline_manifest["manifest_sha256"],
                "treatment_realization_manifest_sha256": treatment_manifest["manifest_sha256"],
                "baseline_scale": float(np.asarray(baseline_manifest["scale"])[0]),
                "treatment_scale": float(np.asarray(treatment_manifest["scale"])[0]),
                "baseline_moving_raw_uint8_sha256": baseline_raw_hash,
                "treatment_moving_raw_uint8_sha256": treatment_raw_hash,
                "baseline_source_160x232_sha256": baseline_source_hash,
                "treatment_source_160x232_sha256": treatment_source_hash,
                "preserved_realization_seed": int(baseline_manifest["realization_seed"]),
                "mutated_fields": ["scale", "manifest_sha256"],
                "moving_raw_uint8_changed": True,
                "source_160x232_changed": True,
            }
        )
    return {
        "baseline": torch.cat(baseline_sources),
        "treatment": torch.cat(treatment_sources),
        "records": records,
        "invalid_source_count": invalid_source,
        "intervention_integrity": {
            "all_baseline_scales_differ_from_one": True,
            "all_treatment_scales_equal_one": True,
            "only_scale_and_manifest_commitment_changed": True,
            "all_raw_and_downsampled_source_hashes_changed": True,
        },
    }


def _score_pair(
    model: AtlasPairEnergyModel,
    renderer: SyntheticRegistrationGenerator,
    sources: dict,
    parent: dict,
    stored_result: dict,
    config: dict,
) -> dict:
    device = renderer.device
    baseline_all = sources["baseline"].to(device)
    treatment_all = sources["treatment"].to(device)
    stored_raw = stored_result["fixed_candidates"]["raw"]
    truth_all = _pose_from_manifest(parent, device)
    source_order = _derangement(truth_all, config["data"]["seed"] + 101)
    condition_sources = {"baseline": baseline_all, "treatment": treatment_all}
    correct = {
        name: {kind: 0 for kind in ("normal", "broken_atlas_binding", "broken_source_pairing")}
        for name in condition_sources
    }
    nonfinite = 0
    invalid_render = 0
    raw = []
    batch_size = config["data"]["batch_size"]
    chunk_size = config["data"]["candidate_chunk_size"]
    for start in range(0, len(truth_all), batch_size):
        stop = min(start + batch_size, len(truth_all))
        rows = stored_raw[start:stop]
        candidate_pose = torch.as_tensor(
            [value["candidate_pose"] for value in rows], device=device, dtype=torch.float32
        )
        target = torch.as_tensor(
            [value["target_index"] for value in rows], device=device, dtype=torch.long
        )
        kinds = [value["candidate_kind"] for value in rows]
        expected_pose, expected_target, expected_kinds = candidate_pose_table(
            truth_all[start:stop], config["data"]["seed"], np.arange(start, stop)
        )
        if (
            not torch.equal(candidate_pose, expected_pose)
            or not torch.equal(target, expected_target)
            or kinds != expected_kinds
            or any(value.count("truth") != 1 or value.count("joint-global") != 3 for value in kinds)
        ):
            raise RuntimeError("stored fixed candidate construction changed")
        candidate_image, candidate_mask = render_candidate_poses(renderer, candidate_pose)
        invalid = (
            ~torch.isfinite(candidate_image).flatten(2).all(2)
            | ~candidate_mask.flatten(2).any(2)
        )
        invalid_render += int(invalid.sum())
        batch_outputs = {}
        for condition, source_all in condition_sources.items():
            source = source_all[start:stop]
            zeros = torch.zeros_like(source, dtype=torch.bool)
            available = torch.zeros(stop - start, 1, 1, 1, device=device)
            normal = model(
                source, zeros, available, candidate_image, candidate_mask,
                candidate_chunk_size=chunk_size,
            )
            broken_atlas = model(
                source, zeros, available, candidate_image.roll(1, 1), candidate_mask.roll(1, 1),
                candidate_chunk_size=chunk_size,
            )
            source_indices = torch.as_tensor(source_order[start:stop], device=device)
            broken_source = model(
                source_all[source_indices], zeros, available, candidate_image, candidate_mask,
                candidate_chunk_size=chunk_size,
            )
            outputs = {
                "normal": normal,
                "broken_atlas_binding": broken_atlas,
                "broken_source_pairing": broken_source,
            }
            batch_outputs[condition] = outputs
            for kind, output in outputs.items():
                correct[condition][kind] += int((output["energy"].argmin(1) == target).sum())
                nonfinite += sum(int((~torch.isfinite(value)).sum()) for value in output.values())
        for offset, stored in enumerate(rows):
            item = start + offset
            paired = {
                "sample_index": item,
                "synthetic_realization_id": sources["records"][item]["synthetic_realization_id"],
                "true_pose": truth_all[item].detach().cpu(),
                "candidate_pose": candidate_pose[offset].detach().cpu(),
                "candidate_kind": kinds[offset],
                "target_index": int(target[offset]),
                "candidate_image_sha256": _tensor_sha256(candidate_image[offset]),
                "candidate_mask_sha256": _tensor_sha256(candidate_mask[offset]),
                "source": sources["records"][item],
            }
            for condition in condition_sources:
                paired[condition] = {
                    f"{kind}_{scale}": batch_outputs[condition][kind][scale][offset].detach().cpu()
                    for kind in ("normal", "broken_atlas_binding", "broken_source_pairing")
                    for scale in ("energy", "energy8", "energy16")
                }
            raw.append(paired)
    _verify_baseline_replay(raw, stored_raw, correct["baseline"], config)
    return {
        "correct": correct,
        "nonfinite_count": nonfinite,
        "invalid_render_count": invalid_render,
        "raw": raw,
    }


def _verify_baseline_replay(
    paired_raw: list[dict], stored_raw: list[dict], correct: dict, config: dict
) -> None:
    names = {
        "normal_energy": "normal_energy",
        "normal_energy8": "normal_energy8",
        "normal_energy16": "normal_energy16",
        "broken_atlas_binding_energy": "broken_atlas_binding_energy",
        "broken_atlas_binding_energy8": "broken_atlas_binding_energy8",
        "broken_atlas_binding_energy16": "broken_atlas_binding_energy16",
        "broken_source_pairing_energy": "broken_source_pairing_energy",
        "broken_source_pairing_energy8": "broken_source_pairing_energy8",
        "broken_source_pairing_energy16": "broken_source_pairing_energy16",
    }
    for paired, stored in zip(paired_raw, stored_raw):
        for observed_name, stored_name in names.items():
            observed = paired["baseline"][observed_name]
            expected = torch.as_tensor(stored[stored_name], dtype=torch.float32)
            if not torch.equal(observed, expected):
                raise RuntimeError("baseline fixed-candidate energies did not replay bit-exactly")
    if correct != stored_counts(config):
        raise RuntimeError("baseline fixed-candidate decisions did not replay exactly")


def stored_counts(config: dict) -> dict:
    return {
        "normal": config["gates"]["baseline_top1_exact"],
        "broken_atlas_binding": 0,
        "broken_source_pairing": 1,
    }


def exact_mcnemar(baseline_correct: np.ndarray, treatment_correct: np.ndarray) -> dict:
    corrected = int((~baseline_correct & treatment_correct).sum())
    regressed = int((baseline_correct & ~treatment_correct).sum())
    discordant = corrected + regressed
    if discordant:
        tail = sum(math.comb(discordant, value) for value in range(min(corrected, regressed) + 1))
        p = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p = 1.0
    return {
        "baseline_wrong_treatment_correct": corrected,
        "baseline_correct_treatment_wrong": regressed,
        "discordant_count": discordant,
        "net_corrections": corrected - regressed,
        "exact_two_sided_p": p,
    }


def summarize(scored: dict, parent: dict, config: dict) -> dict:
    baseline_correct, treatment_correct = [], []
    gap_improvement = []
    global_wins = 0
    global_pairs = 0
    for row in scored["raw"]:
        target = row["target_index"]
        baseline = np.asarray(row["baseline"]["normal_energy"], dtype=np.float32)
        treatment = np.asarray(row["treatment"]["normal_energy"], dtype=np.float32)
        baseline_correct.append(int(np.argmin(baseline)) == target)
        treatment_correct.append(int(np.argmin(treatment)) == target)
        not_truth = np.arange(len(baseline)) != target
        baseline_gap = float(baseline[not_truth].min() - baseline[target])
        treatment_gap = float(treatment[not_truth].min() - treatment[target])
        gap_improvement.append(treatment_gap - baseline_gap)
        global_indices = np.flatnonzero(np.asarray(row["candidate_kind"]) == "joint-global")
        global_wins += int((treatment[target] < treatment[global_indices]).sum())
        global_pairs += len(global_indices)
        row["baseline_truth_gap"] = baseline_gap
        row["treatment_truth_gap"] = treatment_gap
        row["paired_truth_gap_improvement"] = treatment_gap - baseline_gap
    baseline_correct = np.asarray(baseline_correct, bool)
    treatment_correct = np.asarray(treatment_correct, bool)
    mcnemar = exact_mcnemar(baseline_correct, treatment_correct)
    scale = np.asarray(parent["scale"], dtype=np.float64)
    absolute_log_scale = np.abs(np.log(scale))
    median_scale = float(np.median(absolute_log_scale))
    low = absolute_log_scale <= median_scale
    strata = {}
    for name, mask in (("low", low), ("high", ~low)):
        indices = np.flatnonzero(mask)
        values = np.asarray(gap_improvement)[indices]
        strata[name] = {
            "definition": "abs(log(original_scale)) <= median" if name == "low" else "abs(log(original_scale)) > median",
            "count": len(indices),
            "sample_indices": indices,
            "baseline_top1": int(baseline_correct[indices].sum()),
            "treatment_top1": int(treatment_correct[indices].sum()),
            "median_paired_truth_gap_improvement": float(np.median(values)),
        }
    return {
        "baseline_top1": int(baseline_correct.sum()),
        "treatment_top1": int(treatment_correct.sum()),
        "mcnemar": mcnemar,
        "median_paired_truth_gap_improvement": float(np.median(gap_improvement)),
        "paired_truth_gap_improvement": gap_improvement,
        "global_pair_accuracy": global_wins / global_pairs,
        "global_pair_wins": global_wins,
        "global_pair_count": global_pairs,
        "original_scale": scale,
        "absolute_log_original_scale": absolute_log_scale,
        "median_absolute_log_original_scale": median_scale,
        "scale_strata": strata,
    }


def gate_status(scored: dict, summary: dict, sources: dict, config: dict) -> dict:
    gates = config["gates"]
    mcnemar = summary["mcnemar"]
    replay = {
        "baseline_source_hashes_exact": True,
        "baseline_energies_bit_exact": True,
        "baseline_top1_exact": summary["baseline_top1"] == gates["baseline_top1_exact"],
    }
    causal = {
        "treatment_top1": summary["treatment_top1"] >= gates["treatment_top1_minimum"],
        "net_corrections": mcnemar["net_corrections"] >= gates["net_corrections_minimum"],
        "mcnemar": mcnemar["exact_two_sided_p"] <= gates["mcnemar_exact_two_sided_p_maximum"],
        "truth_gap": summary["median_paired_truth_gap_improvement"]
        >= gates["median_paired_truth_gap_improvement_minimum"],
    }
    specificity = {
        "global_pair_accuracy": (
            summary["global_pair_count"] == config["data"]["count"] * 3
            and summary["global_pair_accuracy"] >= gates["global_pair_accuracy_minimum"]
        ),
        "baseline_broken_atlas": scored["correct"]["baseline"]["broken_atlas_binding"]
        <= gates["broken_atlas_correct_maximum"],
        "baseline_broken_source": scored["correct"]["baseline"]["broken_source_pairing"]
        <= gates["broken_source_correct_maximum"],
        "treatment_broken_atlas": scored["correct"]["treatment"]["broken_atlas_binding"]
        <= gates["broken_atlas_correct_maximum"],
        "treatment_broken_source": scored["correct"]["treatment"]["broken_source_pairing"]
        <= gates["broken_source_correct_maximum"],
        "nonfinite": scored["nonfinite_count"] <= gates["nonfinite_count_maximum"],
        "invalid_source_or_render": sources["invalid_source_count"] + scored["invalid_render_count"]
        <= gates["invalid_source_or_render_count_maximum"],
    }
    if not all(replay.values()):
        branch = "replay_gate_fails"
    elif not all(specificity.values()):
        branch = "specificity_gate_fails"
    elif not all(causal.values()):
        branch = "causal_gates_fail"
    else:
        branch = "all_gates_pass"
    checks = {**replay, **causal, **specificity}
    return {
        "passed": all(checks.values()),
        "interpretation_branch": branch,
        "interpretation": config["interpretation"][branch],
        "checks": checks,
        "status_sha256": _canonical_sha256(checks),
    }


def _write_immutable_json(path: Path, payload: dict) -> None:
    encoded = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _existing_receipt(path: Path, config: dict) -> bool:
    if not path.exists():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    commitment = receipt.pop("receipt_sha256", None)
    fixed = {
        "format": FORMAT,
        "purpose": PURPOSE,
        "role": config["role"],
        "scope": config["interpretation"]["scope"],
        "product5_access": False,
        "qualification_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "training_performed": False,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": source_hashes(),
        "checkpoint_file_sha256": config["artifacts"]["checkpoint_file_sha256"],
        "development_manifest_file_sha256": config["artifacts"]["development_manifest_file_sha256"],
        "development_manifest_sha256": config["artifacts"]["development_manifest_sha256"],
        "development_result_file_sha256": config["artifacts"]["development_result_file_sha256"],
        "development_result_sha256": config["artifacts"]["development_result_sha256"],
        "generator_contract_sha256": config["artifacts"]["generator_contract_sha256"],
        "atlas_sha256": config["artifacts"]["atlas_sha256"],
        "intervention": config["intervention"],
        "statistics_contract": config["statistics"],
        "training_scale_reference": config["training_scale_reference"],
    }
    dynamic = {
        "checkpoint_model_state_sha256",
        "source_realizations",
        "intervention_integrity",
        "fixed_candidates",
        "summary",
        "invalid_source_count",
        "status",
    }
    if (
        commitment != _canonical_sha256(receipt)
        or set(receipt) != set(fixed) | dynamic
        or any(
        receipt.get(name) != value for name, value in fixed.items()
        )
    ):
        raise RuntimeError("existing immutable scale-ablation receipt lineage differs")
    records = receipt["source_realizations"]
    raw = receipt["fixed_candidates"].get("raw", [])
    expected_integrity = {
        "all_baseline_scales_differ_from_one": True,
        "all_treatment_scales_equal_one": True,
        "only_scale_and_manifest_commitment_changed": True,
        "all_raw_and_downsampled_source_hashes_changed": True,
    }
    if (
        len(records) != config["data"]["count"]
        or len(raw) != config["data"]["count"]
        or len(receipt.get("checkpoint_model_state_sha256", "")) != 64
        or receipt["intervention_integrity"] != expected_integrity
        or any(
            value.get("baseline_scale") == 1.0
            or value.get("treatment_scale") != 1.0
            or value.get("mutated_fields") != ["scale", "manifest_sha256"]
            or value.get("moving_raw_uint8_changed") is not True
            or value.get("source_160x232_changed") is not True
            or value.get("baseline_realization_manifest_sha256")
            == value.get("treatment_realization_manifest_sha256")
            or value.get("baseline_moving_raw_uint8_sha256")
            == value.get("treatment_moving_raw_uint8_sha256")
            or value.get("baseline_source_160x232_sha256")
            == value.get("treatment_source_160x232_sha256")
            for value in records
        )
        or len({value.get("synthetic_realization_id") for value in records})
        != config["data"]["count"]
        or [value.get("sample_index") for value in records]
        != list(range(config["data"]["count"]))
        or [value.get("synthetic_realization_id") for value in raw]
        != [value.get("synthetic_realization_id") for value in records]
        or any(
            len(value.get("candidate_kind", [])) != config["data"]["candidate_count"]
            or value.get("candidate_kind", []).count("truth") != 1
            or value.get("candidate_kind", []).count("joint-global") != 3
            for value in raw
        )
    ):
        raise RuntimeError("existing immutable scale-ablation intervention differs")
    replayed_summary = summarize(
        receipt["fixed_candidates"],
        {"scale": [value["baseline_scale"] for value in records]},
        config,
    )
    replayed_status = gate_status(
        receipt["fixed_candidates"],
        replayed_summary,
        {"invalid_source_count": receipt["invalid_source_count"]},
        config,
    )
    if (
        _canonical(receipt["summary"]) != _canonical(replayed_summary)
        or receipt["status"] != _canonical(replayed_status)
    ):
        raise RuntimeError("existing immutable scale-ablation statistics differ")
    return True


def run(config_path: str | Path) -> Path:
    config = inspect_config(config_path)
    output = Path(config["paths"]["output_run_folder"])
    receipt_path = output / config["paths"]["receipt_file"]
    if output.resolve() != OUTPUT_RUN.resolve() or output.drive.upper() != "I:":
        raise RuntimeError("scale ablation output path changed")
    if _existing_receipt(receipt_path, config):
        return receipt_path
    if not torch.cuda.is_available():
        raise RuntimeError("bit-exact baseline replay requires the original CUDA execution path")
    device = torch.device("cuda")
    parent, stored_result = _load_committed_development(config)
    model, checkpoint = _load_checkpoint(config, device)
    generator = SyntheticRegistrationGenerator(
        ROOT / config["paths"]["atlas_repo_relative"], device
    )
    if (
        generator.contract != checkpoint["data_lineage"]["generator_contract"]
        or generator.contract["contract_sha256"] != config["artifacts"]["generator_contract_sha256"]
    ):
        raise RuntimeError("atlas generator differs from the frozen checkpoint")
    with torch.no_grad():
        sources = _materialize_pairs(
            generator, parent, stored_result["fixed_candidates"]["raw"]
        )
        scored = _score_pair(model, generator, sources, parent, stored_result, config)
    summary = summarize(scored, parent, config)
    status = gate_status(scored, summary, sources, config)
    receipt = {
        "format": FORMAT,
        "purpose": PURPOSE,
        "role": config["role"],
        "scope": config["interpretation"]["scope"],
        "product5_access": False,
        "qualification_access": False,
        "calibration_access": False,
        "final_test_access": False,
        "training_performed": False,
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "source_sha256": source_hashes(),
        "checkpoint_file_sha256": config["artifacts"]["checkpoint_file_sha256"],
        "checkpoint_model_state_sha256": _canonical_sha256(checkpoint["model"]),
        "development_manifest_file_sha256": config["artifacts"]["development_manifest_file_sha256"],
        "development_manifest_sha256": config["artifacts"]["development_manifest_sha256"],
        "development_result_file_sha256": config["artifacts"]["development_result_file_sha256"],
        "development_result_sha256": config["artifacts"]["development_result_sha256"],
        "generator_contract_sha256": config["artifacts"]["generator_contract_sha256"],
        "atlas_sha256": config["artifacts"]["atlas_sha256"],
        "intervention": config["intervention"],
        "statistics_contract": config["statistics"],
        "training_scale_reference": config["training_scale_reference"],
        "source_realizations": sources["records"],
        "intervention_integrity": sources["intervention_integrity"],
        "fixed_candidates": scored,
        "summary": summary,
        "invalid_source_count": sources["invalid_source_count"],
        "status": status,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_immutable_json(receipt_path, receipt)
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit(
            "usage: python -m training.run_independent_atlas_pair_scale_ablation CONFIG"
        )
    print(run(arguments[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
