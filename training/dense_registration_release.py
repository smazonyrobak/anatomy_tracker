"""Auditable qualification and release protocol for dense registration v2.

This module deliberately owns no training logic. A trained EMA is evaluated on
two locked validation cohorts, frozen only after both pass, and then evaluated
once under a cooperative local sealed-test convention. Deleting local state can
reset that convention; it is an audit aid, not a security boundary. The evidence
is synthetic Allen CCFv3 evidence, not a claim of real-histology accuracy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from training import synthetic_registration as synthetic
from training import train_dense_registration as training
from source.dense_registration_preprocessing import (
    MASK_CONTRACT_SHA256,
    MODEL_SHAPE,
    PREPROCESSING_CONTRACT_V2,
)
from source.dense_registration_runtime import (
    DENSE_REGISTRATION_V2_RELEASE_PROTOCOL as V2_RELEASE_PROTOCOL,
    DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256 as V2_RELEASE_PROTOCOL_SHA256,
    dense_registration_v2_gate_report as release_gate_report,
    verify_dense_registration_v2_bundle,
)


FORMAT_VERSION = V2_RELEASE_PROTOCOL["protocol_version"]
GENERATOR_PROFILE = V2_RELEASE_PROTOCOL["generator_profile"]
BENCHMARK_ID = V2_RELEASE_PROTOCOL["benchmark_id"]
RELEASE_STRATA = tuple(V2_RELEASE_PROTOCOL["strata"])
QUALIFICATION_SEEDS = tuple(V2_RELEASE_PROTOCOL["cohorts"]["qualification"]["seeds"])
QUALIFICATION_SAMPLES_PER_STRATUM = V2_RELEASE_PROTOCOL["cohorts"]["qualification"]["samples_per_stratum"]
MASK_STRESS_SAMPLES_PER_STRATUM = V2_RELEASE_PROTOCOL["cohorts"]["mask_stress"]["samples_per_stratum"]
SEALED_SAMPLES_PER_STRATUM = V2_RELEASE_PROTOCOL["cohorts"]["sealed"]["samples_per_stratum"]
SEALED_CONFIRMATION = "COOPERATIVE-LOCAL-SEALED-EVALUATION-V2"
EVIDENCE_SCOPE = V2_RELEASE_PROTOCOL["scope"]
_LOCAL_SEALED_STATE_ROOT = (
    Path.home() / ".proprietary-anatomy-tracker" / "final-holdout" / BENCHMARK_ID
)


def _canonical_bytes(payload: dict | list) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(payload: dict | list) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_once(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _canonical_bytes(existing) != _canonical_bytes(payload):
            raise ValueError(f"locked evidence differs: {destination}")
        return destination
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _claim_once(path: str | Path, payload: dict) -> Path:
    """Create a cooperative local one-shot claim; deleting it resets the convention."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = (
        Path(__file__).resolve(),
        root / "train_dense_registration.py",
        root / "dense_registration_model.py",
        root / "synthetic_registration.py",
        root.parent / "source" / "dense_registration_preprocessing.py",
        root.parent / "source" / "dense_registration_runtime.py",
    )
    return {path.relative_to(root.parent).as_posix(): sha256_file(path) for path in files}


def _enable_strict_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def deterministic_environment(device: str | torch.device) -> dict:
    """Enable strict deterministic inference and return its reproducibility record."""
    _enable_strict_determinism()
    selected = torch.device(device)

    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "scipy": version("scipy"),
        "pynrrd": version("pynrrd"),
        "onnx": version("onnx"),
        "onnxruntime": version("onnxruntime-gpu") or version("onnxruntime"),
        "onnxruntime_directml": version("onnxruntime-directml"),
        "device": str(selected),
        "device_name": (
            torch.cuda.get_device_name(selected)
            if selected.type == "cuda" and torch.cuda.is_available()
            else platform.processor()
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_identity(checkpoint_path: str | Path) -> tuple[dict, dict]:
    _enable_strict_determinism()
    path = Path(checkpoint_path).resolve()
    checkpoint = training.load_checkpoint(path, "cpu")
    generator_contract = checkpoint.get("generator_contract", {})
    contract_payload = dict(generator_contract)
    contract_commitment = contract_payload.pop("contract_sha256", None)
    if (
        generator_contract.get("profile") != GENERATOR_PROFILE
        or not contract_commitment
        or synthetic._payload_sha256(contract_payload) != contract_commitment
    ):
        raise ValueError("release v2 requires a committed v2 generator contract")
    shadow = checkpoint.get("ema", {}).get("shadow")
    if not isinstance(shadow, dict) or not shadow:
        raise ValueError("release v2 requires EMA weights")
    identity = {
        "checkpoint_file_sha256": sha256_file(path),
        "ema_state_payload_sha256": _state_sha256(shadow),
        "weights": "ema",
        "model_config_payload_sha256": canonical_json_sha256(checkpoint["model_config"]),
        "generator_contract_payload_sha256": canonical_json_sha256(generator_contract),
    }
    return identity, checkpoint


def _record_descriptor(record: dict) -> dict:
    mode = int(record["moving_appearance_mode"])
    return {
        "stratum": record["stratum"],
        "sample_index": int(record["sample_index"]),
        "seed": int(record["seed"]),
        "manifest_sha256": record["manifest_sha256"],
        "moving_appearance_mode": "label" if mode else "template",
        "mask_offset_px": int(record["mask_offset_px"]),
    }


def _records(
    manifest_factory,
    *,
    split: str,
    seed: int,
    samples_per_stratum: int,
    forced_mask_offset: int | None = None,
) -> list[dict]:
    final = split == "sealed-test"
    capability = synthetic._FINAL_HOLDOUT_CAPABILITY if final else None
    records = []
    for stratum_index, stratum in enumerate(RELEASE_STRATA):
        for sample_index in range(samples_per_stratum):
            sample_seed = training.evaluation_sample_seed(seed, stratum_index, sample_index)
            manifest = manifest_factory(
                1, split, sample_seed, stratum, _final_capability=capability
            )
            if forced_mask_offset is not None:
                manifest["mask_offset_px"] = np.asarray(
                    [forced_mask_offset], dtype=np.int8
                )
                unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
                manifest["manifest_sha256"] = synthetic._payload_sha256(unhashed)
            records.append(
                {
                    "stratum": stratum,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "moving_appearance_mode": int(manifest["moving_appearance_mode"][0]),
                    "mask_offset_px": int(manifest["mask_offset_px"][0]),
                    "manifest": manifest,
                }
            )
    return records


def _cohorts(manifest_factory, split: str, seed: int, main_count: int, stress_count: int) -> dict:
    return {
        "main": _records(
            manifest_factory, split=split, seed=seed,
            samples_per_stratum=main_count,
        ),
        "mask-minus-3": _records(
            manifest_factory, split=split, seed=seed,
            samples_per_stratum=stress_count, forced_mask_offset=-3,
        ),
        "mask-plus-3": _records(
            manifest_factory, split=split, seed=seed,
            samples_per_stratum=stress_count, forced_mask_offset=3,
        ),
    }


def _locked_manifest(
    identity: dict,
    generator,
    split: str,
    seed: int,
    cohorts: dict,
    *,
    claim_file_sha256: str | None = None,
) -> dict:
    payload = {
        "format_version": FORMAT_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
        "scope": EVIDENCE_SCOPE,
        "split": split,
        "seed": seed,
        "profile": GENERATOR_PROFILE,
        "checkpoint": identity,
        "generator_contract": generator.contract,
        "source_file_sha256": source_hashes(),
        "cohorts": {
            name: [_record_descriptor(record) for record in records]
            for name, records in cohorts.items()
        },
    }
    if claim_file_sha256 is not None:
        payload["claim_file_sha256"] = claim_file_sha256
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _vector_evidence(values) -> dict:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    if not array.size:
        return {"count": 0, "float32_bytes_sha256": digest, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "float32_bytes_sha256": digest,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _sample_evidence(record: dict, sample: dict) -> dict:
    scalar_names = (
        "foreground_correspondence", "analytic_foreground_correspondence",
        "macro_region_dice", "boundary_f1_2px", "boundary_mean_distance_px",
        "fold_count", "jacobian_count", "jacobian_min", "inverse_fold_count",
        "inverse_jacobian_count", "inverse_jacobian_min",
    )
    evidence = _record_descriptor(record)
    evidence.update({name: sample[name] for name in scalar_names})
    for name in (
        "endpoint_values", "inverse_endpoint_values", "damage_endpoint_values",
        "inverse_damage_endpoint_values", "inverse_cycle_values", "reverse_cycle_values",
    ):
        evidence[name] = _vector_evidence(sample[name])
    return evidence


@torch.inference_mode()
def _evaluate_records(model, generator, records: list[dict], batch_size: int, *, final: bool) -> dict:
    grouped = {stratum: [] for stratum in RELEASE_STRATA}
    evidence = []
    capability = synthetic._FINAL_HOLDOUT_CAPABILITY if final else None
    for offset in range(0, len(records), batch_size):
        selected = records[offset : offset + batch_size]
        pairs = [
            generator.batch(record["manifest"], _final_capability=capability)
            for record in selected
        ]
        pair = training._stack_pairs(pairs)
        moving_mask = pair.get("moving_model_mask", pair["moving_tissue_mask"])
        fixed_input = torch.cat((pair["fixed"], pair["fixed_mask"].float()), dim=1)
        moving_input = torch.cat((pair["moving"], moving_mask.float()), dim=1)
        forward, inverse = model(fixed_input, moving_input)
        samples = training._sample_metrics(pair, forward.float(), inverse.float())
        for record, sample in zip(selected, samples):
            sample["moving_appearance_mode"] = "label" if record["moving_appearance_mode"] else "template"
            sample["mask_offset_px"] = str(record["mask_offset_px"])
            grouped[record["stratum"]].append(sample)
            evidence.append(_sample_evidence(record, sample))
    all_samples = [sample for stratum in RELEASE_STRATA for sample in grouped[stratum]]
    report = {
        "overall": training.summarize_metrics(all_samples),
        "per_stratum": {
            stratum: training.summarize_metrics(grouped[stratum])
            for stratum in RELEASE_STRATA
        },
        "appearance_subgroups": {
            mode: training.summarize_metrics(
                [sample for sample in all_samples if sample["moving_appearance_mode"] == mode]
            )
            for mode in ("template", "label")
            if any(sample["moving_appearance_mode"] == mode for sample in all_samples)
        },
        "mask_offset_subgroups": {
            str(value): training.summarize_metrics(
                [sample for sample in all_samples if sample["mask_offset_px"] == str(value)]
            )
            for value in (-3, 3)
            if any(sample["mask_offset_px"] == str(value) for sample in all_samples)
        },
        "sample_evidence": evidence,
    }
    report["sample_evidence_payload_sha256"] = canonical_json_sha256(evidence)
    return report


def _evaluate_cohorts(model, generator, cohorts: dict, batch_size: int, *, final: bool) -> tuple[dict, dict]:
    main = _evaluate_records(model, generator, cohorts["main"], batch_size, final=final)
    stress_parts = [
        _evaluate_records(model, generator, cohorts[name], batch_size, final=final)
        for name in ("mask-minus-3", "mask-plus-3")
    ]
    stress_evidence = sum((part["sample_evidence"] for part in stress_parts), [])
    stress = {
        "mask_offset_subgroups": {
            str(offset): stress_parts[index]["overall"]
            for index, offset in enumerate((-3, 3))
        },
        "sample_evidence": stress_evidence,
        "sample_evidence_payload_sha256": canonical_json_sha256(stress_evidence),
    }
    return main, stress


def _load_release_components(checkpoint_path: str | Path, atlas: str | Path, device):
    _enable_strict_determinism()
    identity, checkpoint = checkpoint_identity(checkpoint_path)
    model, loaded = training.model_from_checkpoint(checkpoint_path, device, use_ema=True)
    generator = synthetic.SyntheticRegistrationGenerator(atlas, device)
    if loaded["generator_contract"] != generator.contract:
        raise ValueError("checkpoint and installed v2 generator contracts differ")
    return identity, checkpoint, model.eval(), generator


def qualify_checkpoint(
    checkpoint_path: str | Path,
    evidence_root: str | Path,
    *,
    atlas: str | Path = training.DEFAULT_ATLAS,
    device: str = "cuda",
    batch_size: int = 2,
) -> Path:
    """Freshly evaluate the checkpoint EMA on both declared qualification seeds."""
    identity, _ = checkpoint_identity(checkpoint_path)
    root = Path(evidence_root).resolve() / identity["checkpoint_file_sha256"]
    receipt_path = root / "qualification-receipt.json"
    if receipt_path.exists():
        verify_qualification(checkpoint_path, receipt_path)
        return receipt_path
    environment = deterministic_environment(device)
    identity, _, model, generator = _load_release_components(checkpoint_path, atlas, device)
    result_files = []
    for seed in QUALIFICATION_SEEDS:
        cohorts = _cohorts(
            generator.make_manifest, "validation", seed,
            QUALIFICATION_SAMPLES_PER_STRATUM, MASK_STRESS_SAMPLES_PER_STRATUM,
        )
        manifest = _locked_manifest(identity, generator, "validation", seed, cohorts)
        manifest_path = _write_once(root / f"qualification-{seed}.manifest.json", manifest)
        main, stress = _evaluate_cohorts(model, generator, cohorts, batch_size, final=False)
        gate = release_gate_report(main, stress)
        result = {
            "format_version": FORMAT_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
            "scope": EVIDENCE_SCOPE,
            "seed": seed,
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "checkpoint": identity,
            "environment": environment,
            "main": main,
            "mask_stress": stress,
            "release_gate": gate,
        }
        result_files.append(_write_once(root / f"qualification-{seed}.result.json", result))
    receipt = {
        "format_version": FORMAT_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
        "scope": EVIDENCE_SCOPE,
        "status": "passed" if all(
            json.loads(path.read_text(encoding="utf-8"))["release_gate"]["passed"]
            for path in result_files
        ) else "rejected",
        "completed_utc": _utc_now(),
        "checkpoint": identity,
        "qualification_seeds": list(QUALIFICATION_SEEDS),
        "results": {
            str(seed): {
                "manifest_file_sha256": sha256_file(root / f"qualification-{seed}.manifest.json"),
                "result_file_sha256": sha256_file(root / f"qualification-{seed}.result.json"),
            }
            for seed in QUALIFICATION_SEEDS
        },
        "source_file_sha256": source_hashes(),
        "environment": environment,
    }
    return _write_once(receipt_path, receipt)


def verify_qualification(
    checkpoint_path: str | Path, receipt_path: str | Path
) -> dict:
    identity, checkpoint = checkpoint_identity(checkpoint_path)
    path = Path(receipt_path).resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("checkpoint") != identity:
        raise RuntimeError("a passing qualification receipt for this EMA is required")
    if receipt.get("release_protocol_sha256") != V2_RELEASE_PROTOCOL_SHA256:
        raise ValueError("qualification release protocol differs")
    if receipt.get("qualification_seeds") != list(QUALIFICATION_SEEDS):
        raise ValueError("qualification seeds differ from the locked protocol")
    if receipt.get("source_file_sha256") != source_hashes():
        raise ValueError("qualification evaluator source has changed")
    for seed in QUALIFICATION_SEEDS:
        manifest_path = path.parent / f"qualification-{seed}.manifest.json"
        result_path = path.parent / f"qualification-{seed}.result.json"
        committed = receipt["results"][str(seed)]
        if sha256_file(manifest_path) != committed["manifest_file_sha256"]:
            raise ValueError("qualification manifest hash mismatch")
        if sha256_file(result_path) != committed["result_file_sha256"]:
            raise ValueError("qualification result hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _verify_manifest(
            manifest, identity, checkpoint["generator_contract"], "validation", seed
        )
        if (
            result.get("checkpoint") != identity
            or result.get("release_protocol_sha256") != V2_RELEASE_PROTOCOL_SHA256
            or result.get("manifest_payload_sha256")
            != manifest["manifest_payload_sha256"]
            or result.get("manifest_file_sha256") != sha256_file(manifest_path)
        ):
            raise ValueError("qualification result is detached from its locked cohort")
        _verify_sample_evidence(result["main"], manifest["cohorts"]["main"])
        stress_descriptors = (
            manifest["cohorts"]["mask-minus-3"]
            + manifest["cohorts"]["mask-plus-3"]
        )
        _verify_sample_evidence(result["mask_stress"], stress_descriptors)
        expected = release_gate_report(result["main"], result["mask_stress"])
        if result.get("release_gate") != expected or not expected["passed"]:
            raise RuntimeError("qualification evidence does not pass the release gates")
    return receipt


def freeze_qualified_candidate(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    workspace: str | Path,
) -> Path:
    """Create a content-addressed candidate only after fresh EMA qualification."""
    receipt = verify_qualification(checkpoint_path, receipt_path)
    identity, checkpoint = checkpoint_identity(checkpoint_path)
    folder = Path(workspace).resolve() / "frozen-candidates-v2" / identity["checkpoint_file_sha256"]
    candidate_path = folder / "candidate.json"
    if candidate_path.exists():
        candidate, _ = _verified_candidate(candidate_path)
        if candidate["identity"] != identity:
            raise ValueError("existing candidate belongs to a different checkpoint")
        return candidate_path
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "model.pt"
    if frozen.exists():
        if sha256_file(frozen) != identity["checkpoint_file_sha256"]:
            raise ValueError("frozen candidate checkpoint hash mismatch")
    else:
        temporary = folder / "model.pt.tmp"
        shutil.copyfile(Path(checkpoint_path).resolve(), temporary)
        os.replace(temporary, frozen)
    qualification_folder = folder / "qualification"
    qualification_folder.mkdir(exist_ok=True)
    evidence_sources = [Path(receipt_path).resolve()]
    for seed in QUALIFICATION_SEEDS:
        evidence_sources.extend(
            (
                Path(receipt_path).resolve().parent / f"qualification-{seed}.manifest.json",
                Path(receipt_path).resolve().parent / f"qualification-{seed}.result.json",
            )
        )
    qualification_evidence = {}
    for source in evidence_sources:
        destination = qualification_folder / source.name
        digest = sha256_file(source)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError("frozen qualification evidence hash mismatch")
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        qualification_evidence[f"qualification/{source.name}"] = digest
    manifest = {
        "format_version": FORMAT_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "release_protocol": V2_RELEASE_PROTOCOL,
        "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
        "scope": EVIDENCE_SCOPE,
        "created_utc": _utc_now(),
        "checkpoint": "model.pt",
        "identity": identity,
        "model_config": checkpoint["model_config"],
        "generator_contract": checkpoint["generator_contract"],
        "qualification_receipt_file_sha256": sha256_file(receipt_path),
        "qualification_receipt": receipt,
        "qualification_evidence": qualification_evidence,
        "source_file_sha256": source_hashes(),
    }
    manifest["candidate_payload_sha256"] = canonical_json_sha256(manifest)
    return _write_once(candidate_path, manifest)


def _verified_candidate(candidate_path: str | Path) -> tuple[dict, Path]:
    path = Path(candidate_path).resolve()
    candidate = json.loads(path.read_text(encoding="utf-8"))
    commitment = candidate.pop("candidate_payload_sha256", None)
    if not commitment or canonical_json_sha256(candidate) != commitment:
        raise ValueError("candidate commitment mismatch")
    candidate["candidate_payload_sha256"] = commitment
    checkpoint = path.parent / candidate["checkpoint"]
    identity, _ = checkpoint_identity(checkpoint)
    if identity != candidate.get("identity") or path.parent.name != identity["checkpoint_file_sha256"]:
        raise ValueError("candidate checkpoint identity mismatch")
    if (
        canonical_json_sha256(candidate.get("model_config", {}))
        != identity["model_config_payload_sha256"]
        or canonical_json_sha256(candidate.get("generator_contract", {}))
        != identity["generator_contract_payload_sha256"]
    ):
        raise ValueError("candidate configuration differs from its checkpoint")
    if candidate.get("source_file_sha256") != source_hashes():
        raise ValueError("candidate evaluator source has changed")
    if (
        candidate.get("release_protocol") != V2_RELEASE_PROTOCOL
        or candidate.get("release_protocol_sha256") != V2_RELEASE_PROTOCOL_SHA256
    ):
        raise ValueError("candidate release protocol differs")
    qualification = candidate.get("qualification_receipt", {})
    if (
        qualification.get("status") != "passed"
        or qualification.get("checkpoint") != identity
        or qualification.get("qualification_seeds") != list(QUALIFICATION_SEEDS)
        or qualification.get("source_file_sha256") != source_hashes()
    ):
        raise ValueError("candidate is not backed by the locked qualification protocol")
    encoded_receipt = json.dumps(
        qualification, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded_receipt).hexdigest() != candidate.get(
        "qualification_receipt_file_sha256"
    ):
        raise ValueError("embedded qualification receipt hash mismatch")
    evidence = candidate.get("qualification_evidence", {})
    expected_names = {"qualification/qualification-receipt.json"}
    for seed in QUALIFICATION_SEEDS:
        expected_names.update(
            {
                f"qualification/qualification-{seed}.manifest.json",
                f"qualification/qualification-{seed}.result.json",
            }
        )
    if set(evidence) != expected_names:
        raise ValueError("frozen qualification evidence set is incomplete")
    for relative, digest in evidence.items():
        evidence_path = path.parent / relative
        if not evidence_path.is_file() or sha256_file(evidence_path) != digest:
            raise ValueError("frozen qualification evidence hash mismatch")
    verified_qualification = verify_qualification(
        checkpoint,
        path.parent / "qualification" / "qualification-receipt.json",
    )
    if _canonical_bytes(verified_qualification) != _canonical_bytes(qualification):
        raise ValueError("embedded qualification receipt differs from frozen evidence")
    return candidate, checkpoint


def _verify_manifest(
    manifest: dict,
    identity: dict,
    generator_contract: dict,
    split: str,
    seed: int,
) -> None:
    commitment = manifest.get("manifest_payload_sha256")
    uncommitted = dict(manifest)
    uncommitted.pop("manifest_payload_sha256", None)
    if not commitment or canonical_json_sha256(uncommitted) != commitment:
        raise ValueError("evaluation manifest commitment mismatch")
    if (
        manifest.get("benchmark_id") != BENCHMARK_ID
        or manifest.get("release_protocol_sha256") != V2_RELEASE_PROTOCOL_SHA256
        or manifest.get("profile") != GENERATOR_PROFILE
        or manifest.get("split") != split
        or manifest.get("seed") != seed
        or manifest.get("checkpoint") != identity
        or manifest.get("source_file_sha256") != source_hashes()
        or _canonical_bytes(manifest.get("generator_contract", {}))
        != _canonical_bytes(generator_contract)
        or canonical_json_sha256(generator_contract)
        != identity["generator_contract_payload_sha256"]
    ):
        raise ValueError("evaluation manifest differs from its locked protocol")
    generator_payload = dict(generator_contract)
    generator_commitment = generator_payload.pop("contract_sha256", None)
    if (
        generator_contract.get("profile") != GENERATOR_PROFILE
        or not generator_commitment
        or synthetic._payload_sha256(generator_payload) != generator_commitment
    ):
        raise ValueError("evaluation generator contract commitment mismatch")
    main_count = (
        SEALED_SAMPLES_PER_STRATUM
        if split == "sealed-test"
        else QUALIFICATION_SAMPLES_PER_STRATUM
    )
    expected_counts = {
        "main": main_count,
        "mask-minus-3": MASK_STRESS_SAMPLES_PER_STRATUM,
        "mask-plus-3": MASK_STRESS_SAMPLES_PER_STRATUM,
    }
    cohorts = manifest.get("cohorts", {})
    if set(cohorts) != set(expected_counts):
        raise ValueError("evaluation manifest cohort set differs")
    for cohort_name, count in expected_counts.items():
        records = cohorts[cohort_name]
        if len(records) != count * len(RELEASE_STRATA):
            raise ValueError("evaluation manifest cohort is incomplete")
        forced_offset = {"mask-minus-3": -3, "mask-plus-3": 3}.get(cohort_name)
        for stratum_index, stratum in enumerate(RELEASE_STRATA):
            selected = [record for record in records if record.get("stratum") == stratum]
            if sorted(record.get("sample_index") for record in selected) != list(range(count)):
                raise ValueError("evaluation sample indices differ from the locked cohort")
            for record in selected:
                expected_seed = training.evaluation_sample_seed(
                    seed, stratum_index, record["sample_index"]
                )
                if (
                    record.get("seed") != expected_seed
                    or record.get("moving_appearance_mode") not in ("template", "label")
                    or not isinstance(record.get("manifest_sha256"), str)
                    or len(record["manifest_sha256"]) != 64
                    or (
                        forced_offset is not None
                        and record.get("mask_offset_px") != forced_offset
                    )
                ):
                    raise ValueError("evaluation sample descriptor differs from the locked cohort")
    regeneration = partial(synthetic.make_registration_manifest, generator_contract)
    expected = _cohorts(
        regeneration, split, seed, main_count, MASK_STRESS_SAMPLES_PER_STRATUM
    )
    expected_descriptors = {
        name: [_record_descriptor(record) for record in records]
        for name, records in expected.items()
    }
    if cohorts != expected_descriptors:
        raise ValueError("evaluation manifest does not exactly regenerate")


def _verify_sample_evidence(report: dict, descriptors: list[dict]) -> None:
    evidence = report.get("sample_evidence", [])
    if report.get("sample_evidence_payload_sha256") != canonical_json_sha256(evidence):
        raise ValueError("per-sample evidence hash mismatch")
    descriptor_keys = (
        "stratum", "sample_index", "seed", "manifest_sha256",
        "moving_appearance_mode", "mask_offset_px",
    )
    observed = [
        {key: sample[key] for key in descriptor_keys}
        for sample in evidence
    ]
    if observed != descriptors:
        raise ValueError("per-sample evidence differs from the locked cohort")


def run_sealed_benchmark(
    candidate_path: str | Path,
    *,
    confirmation: str,
    atlas: str | Path = training.DEFAULT_ATLAS,
    device: str = "cuda",
    batch_size: int = 2,
) -> dict:
    """Consume the cooperative local v2 sealed benchmark, including failed attempts."""
    if confirmation != SEALED_CONFIRMATION:
        raise PermissionError(f"sealed evaluation requires {SEALED_CONFIRMATION}")
    candidate, checkpoint_path = _verified_candidate(candidate_path)
    root = Path(_LOCAL_SEALED_STATE_ROOT).resolve()
    claim_path, result_path, receipt_path = (
        root / "claim.json", root / "result.json", root / "receipt.json"
    )
    seed = secrets.randbelow(2**31)
    claim = {
        "format_version": FORMAT_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
        "mechanism": "cooperative-local-one-shot; deleting local state can reset it",
        "claimed_utc": _utc_now(),
        "seed": seed,
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "checkpoint": candidate["identity"],
    }
    try:
        _claim_once(claim_path, claim)
    except FileExistsError as error:
        raise RuntimeError("the global v2 sealed benchmark has already been consumed") from error
    environment = {"device": str(device), "deterministic_setup": False}
    try:
        environment = deterministic_environment(device)
        environment["deterministic_setup"] = True
        identity, _, model, generator = _load_release_components(
            checkpoint_path, atlas, device
        )
        cohorts = _cohorts(
            generator.make_manifest, "sealed-test", seed,
            SEALED_SAMPLES_PER_STRATUM, MASK_STRESS_SAMPLES_PER_STRATUM,
        )
        manifest = _locked_manifest(
            identity,
            generator,
            "sealed-test",
            seed,
            cohorts,
            claim_file_sha256=sha256_file(claim_path),
        )
        main, stress = _evaluate_cohorts(model, generator, cohorts, batch_size, final=True)
        result = {
            "format_version": FORMAT_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
            "scope": EVIDENCE_SCOPE,
            "seed": seed,
            "candidate_payload_sha256": candidate["candidate_payload_sha256"],
            "claim_file_sha256": sha256_file(claim_path),
            "checkpoint": identity,
            "environment": environment,
            "manifest": manifest,
            "main": main,
            "mask_stress": stress,
            "release_gate": release_gate_report(main, stress),
        }
        _write_once(result_path, result)
        receipt = {
            "format_version": FORMAT_VERSION,
            "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
            "status": "passed" if result["release_gate"]["passed"] else "rejected",
            "completed_utc": _utc_now(),
            "seed": seed,
            "candidate_payload_sha256": candidate["candidate_payload_sha256"],
            "claim_file_sha256": sha256_file(claim_path),
            "result_file_sha256": sha256_file(result_path),
            "environment": environment,
        }
        _write_once(receipt_path, receipt)
        return result
    except BaseException as error:
        if not receipt_path.exists():
            _write_once(
                receipt_path,
                {
                    "format_version": FORMAT_VERSION,
                    "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
                    "status": "failed",
                    "completed_utc": _utc_now(),
                    "seed": seed,
                    "candidate_payload_sha256": candidate["candidate_payload_sha256"],
                    "claim_file_sha256": sha256_file(claim_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "environment": environment,
                },
            )
        raise


def verify_passing_sealed_receipt(candidate_path: str | Path) -> tuple[dict, dict, dict]:
    candidate, _ = _verified_candidate(candidate_path)
    root = Path(_LOCAL_SEALED_STATE_ROOT).resolve()
    claim = json.loads((root / "claim.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    commitment = candidate["candidate_payload_sha256"]
    if receipt.get("status") != "passed":
        raise RuntimeError("a passing sealed-v2 receipt is required")
    if any(item.get("candidate_payload_sha256") != commitment for item in (claim, result, receipt)):
        raise ValueError("sealed evidence belongs to another candidate")
    if claim.get("seed") != result.get("seed") or claim.get("seed") != receipt.get("seed"):
        raise ValueError("sealed seed commitment mismatch")
    if receipt.get("result_file_sha256") != sha256_file(root / "result.json"):
        raise ValueError("sealed result hash mismatch")
    claim_file_sha256 = sha256_file(root / "claim.json")
    if (
        claim.get("benchmark_id") != BENCHMARK_ID
        or any(
            item.get("release_protocol_sha256") != V2_RELEASE_PROTOCOL_SHA256
            for item in (claim, result, receipt)
        )
        or result.get("claim_file_sha256") != claim_file_sha256
        or receipt.get("claim_file_sha256") != claim_file_sha256
        or claim.get("checkpoint") != candidate["identity"]
        or result.get("benchmark_id") != BENCHMARK_ID
        or result.get("checkpoint") != candidate["identity"]
    ):
        raise ValueError("sealed evidence differs from the frozen v2 protocol")
    manifest = result.get("manifest", {})
    if manifest.get("claim_file_sha256") != claim_file_sha256:
        raise ValueError("sealed manifest is detached from its claim")
    _verify_manifest(
        manifest,
        candidate["identity"],
        candidate["generator_contract"],
        "sealed-test",
        claim["seed"],
    )
    _verify_sample_evidence(result["main"], manifest["cohorts"]["main"])
    stress_descriptors = (
        manifest["cohorts"]["mask-minus-3"]
        + manifest["cohorts"]["mask-plus-3"]
    )
    _verify_sample_evidence(result["mask_stress"], stress_descriptors)
    expected = release_gate_report(result["main"], result["mask_stress"])
    if result.get("release_gate") != expected or not expected["passed"]:
        raise RuntimeError("sealed-v2 evidence does not pass")
    return candidate, result, {
        "claim": claim,
        "claim_file_sha256": claim_file_sha256,
        "result_file_sha256": sha256_file(root / "result.json"),
        "receipt_file_sha256": sha256_file(root / "receipt.json"),
        "receipt": receipt,
    }


def _onnx_parity_inputs(
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, MODEL_SHAPE[0], device=device),
        torch.linspace(-1.0, 1.0, MODEL_SHAPE[1], device=device),
        indexing="ij",
    )
    fixed_mask = ((x / 0.88).square() + (y / 0.82).square() <= 1.0).float()
    moving_mask = (
        ((x - 0.04) / 0.84).square() + ((y + 0.03) / 0.78).square() <= 1.0
    ).float()
    fixed_image = (
        (0.50 + 0.24 * x + 0.18 * y + 0.08 * x * y).clamp(0.0, 1.0)
        * fixed_mask
    )
    moving_image = (
        (0.46 - 0.17 * x + 0.22 * y + 0.10 * x.square()).clamp(0.0, 1.0)
        * moving_mask
    )
    return (
        torch.stack((fixed_image, fixed_mask), dim=0)[None],
        torch.stack((moving_image, moving_mask), dim=0)[None],
    )


def _parity_manifests(generator) -> list[dict]:
    parity = V2_RELEASE_PROTOCOL["cohorts"]["onnx_parity"]
    cases = []
    case_index = 0
    for stratum_index, stratum in enumerate(V2_RELEASE_PROTOCOL["strata"]):
        for appearance in parity["appearances"]:
            mode = int(appearance == "label")
            for mask_offset in parity["mask_offsets"]:
                seed = training.evaluation_sample_seed(parity["seed"], stratum_index, case_index)
                manifest = generator.make_manifest(1, "validation", seed, stratum)
                manifest["moving_appearance_mode"] = np.asarray([mode], dtype=np.uint8)
                manifest["mask_offset_px"] = np.asarray([mask_offset], dtype=np.int8)
                unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
                manifest["manifest_sha256"] = synthetic._payload_sha256(unhashed)
                cases.append({
                    "stratum": stratum, "appearance": "label" if mode else "template",
                    "mask_offset_px": mask_offset, "manifest": manifest,
                })
                case_index += 1
    return cases


def run_onnx_parity(
    onnx_path: str | Path,
    model,
    generator,
) -> dict:
    """Check DML production and explicit CPU diagnostic parity on all v2 cases."""
    import onnxruntime as ort

    _enable_strict_determinism()
    protocol = V2_RELEASE_PROTOCOL["cohorts"]["onnx_parity"]
    directml_version = importlib.metadata.version("onnxruntime-directml")
    if directml_version != protocol["onnxruntime_directml_version"]:
        raise RuntimeError(
            "onnxruntime-directml differs from the release-pinned version"
        )
    production_provider = protocol["production_provider"]
    diagnostic_provider = protocol["diagnostic_provider"]
    required = [diagnostic_provider, production_provider]
    missing = set(required) - set(ort.get_available_providers())
    if missing:
        raise RuntimeError(f"required ONNX Runtime providers unavailable: {sorted(missing)}")
    sessions = {
        provider: ort.InferenceSession(str(onnx_path), providers=[provider])
        for provider in required
    }
    evidence = []
    for case in _parity_manifests(generator):
        pair = generator.batch(case["manifest"])
        moving_mask = pair.get("moving_model_mask", pair["moving_tissue_mask"])
        fixed = torch.cat((pair["fixed"], pair["fixed_mask"].float()), dim=1)
        moving = torch.cat((pair["moving"], moving_mask.float()), dim=1)
        with torch.inference_mode():
            expected = [value.detach().float().cpu() for value in model(fixed, moving)]
        expected_device = [value.to(pair["fixed"].device) for value in expected]
        pytorch_metrics = training.summarize_metrics(
            training._sample_metrics(pair, expected_device[0], expected_device[1])
        )
        inputs = {
            "fixed_atlas_and_mask": fixed.detach().cpu().numpy(),
            "moving_slice_and_mask": moving.detach().cpu().numpy(),
        }
        case_evidence = {
            **{key: case[key] for key in ("stratum", "appearance", "mask_offset_px")},
            "pytorch_metrics": pytorch_metrics,
            "providers": {},
        }
        for provider, session in sessions.items():
            observed = [torch.from_numpy(value).float() for value in session.run(None, inputs)]
            maximum = max(float((a - b).abs().max()) for a, b in zip(expected, observed))
            mean = float(np.mean([float((a - b).abs().mean()) for a, b in zip(expected, observed)]))
            metric_maps = [value.to(pair["fixed"].device) for value in observed]
            metrics = training.summarize_metrics(
                training._sample_metrics(pair, metric_maps[0], metric_maps[1])
            )
            topology_ok = (
                metrics["fold_fraction"] == 0.0
                and metrics["inverse_fold_fraction"] == 0.0
                and metrics["jacobian_min"] >= protocol["minimum_jacobian"]
                and metrics["inverse_jacobian_min"] >= protocol["minimum_jacobian"]
            )
            delta_bounds = protocol["metric_absolute_delta_bounds"]
            metric_deltas = {
                name: abs(float(metrics[name]) - float(pytorch_metrics[name]))
                for name in delta_bounds
            }
            primary = session.get_providers()[0]
            primary_check = primary == provider
            delta_check = all(
                metric_deltas[name] <= bound for name, bound in delta_bounds.items()
            )
            provider_passed = (
                maximum <= protocol["maximum_absolute_px"]
                and topology_ok
                and primary_check
                and delta_check
            )
            case_evidence["providers"][provider] = {
                "passed": provider_passed,
                "primary_provider": primary,
                "primary_check": primary_check,
                "maximum_absolute_px": maximum,
                "mean_absolute_px": mean,
                "metrics": metrics,
                "metric_absolute_deltas": metric_deltas,
                "metric_delta_check": delta_check,
            }
        evidence.append(case_evidence)
    aggregates = {}
    for provider in required:
        reports = [case["providers"][provider] for case in evidence]
        metric_names = set().union(
            *(report["metric_absolute_deltas"] for report in reports)
        )
        aggregates[provider] = {
            "passed": all(report["passed"] for report in reports),
            "primary_check": all(report["primary_check"] for report in reports),
            "case_count": len(reports),
            "maximum_absolute_px": max(report["maximum_absolute_px"] for report in reports),
            "mean_absolute_px": float(np.mean([report["mean_absolute_px"] for report in reports])),
            "maximum_metric_absolute_deltas": {
                name: max(report["metric_absolute_deltas"][name] for report in reports)
                for name in sorted(metric_names)
            },
        }
    return {
        "passed": all(report["passed"] for report in aggregates.values()),
        "production_provider": production_provider,
        "diagnostic_provider": diagnostic_provider,
        "onnxruntime_directml_version": directml_version,
        "map_maximum_absolute_px_threshold": protocol["maximum_absolute_px"],
        "metric_absolute_delta_bounds": protocol["metric_absolute_delta_bounds"],
        "provider_aggregates": aggregates,
        "cases": evidence,
        "cases_payload_sha256": canonical_json_sha256(evidence),
    }


def build_v2_release_metadata(
    *,
    onnx_model_file_sha256: str,
    candidate: dict,
    checkpoint: dict,
    sealed_result: dict,
    sealed_evidence: dict,
    parity: dict,
    environment: dict,
    candidate_file_sha256: str,
) -> dict:
    """Build the single format-v2 contract consumed by the runtime verifier."""
    generator = checkpoint["generator_contract"]
    identity = candidate["identity"]
    directml_version = environment.get("onnxruntime_directml")
    production_provider = V2_RELEASE_PROTOCOL["cohorts"]["onnx_parity"]["production_provider"]
    pinned_version = V2_RELEASE_PROTOCOL["cohorts"]["onnx_parity"][
        "onnxruntime_directml_version"
    ]
    if directml_version != pinned_version:
        raise RuntimeError("onnxruntime-directml differs from the release protocol")
    if parity.get("onnxruntime_directml_version") != directml_version:
        raise RuntimeError("parity and release environments use different DirectML versions")
    return {
        "format_version": FORMAT_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "release_protocol": V2_RELEASE_PROTOCOL,
        "release_protocol_sha256": V2_RELEASE_PROTOCOL_SHA256,
        "scope": EVIDENCE_SCOPE,
        "candidate": candidate,
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_file_sha256": candidate_file_sha256,
        "candidate_checkpoint_file_sha256": identity["checkpoint_file_sha256"],
        "checkpoint": identity,
        "onnx_model_file_sha256": onnx_model_file_sha256,
        "model_shape": list(MODEL_SHAPE),
        "model_config": checkpoint["model_config"],
        "preprocessing_contract": PREPROCESSING_CONTRACT_V2,
        "mask_contract_payload_sha256": MASK_CONTRACT_SHA256,
        "appearance_contract_sha256": generator["appearance_contract_sha256"],
        "query_sha256": generator["query_sha256"],
        "generator_contract_payload_sha256": identity[
            "generator_contract_payload_sha256"
        ],
        "generator_contract": generator,
        "input_contract": "two grayscale/mask tensors; channels=[image,brain-outline/tissue-mask]",
        "output_contract": "absolute x,y pixel maps fixed->moving and moving->fixed",
        "sealed_test": {
            "claim": sealed_evidence["claim"],
            "claim_file_sha256": sealed_evidence["claim_file_sha256"],
            "manifest": sealed_result["manifest"],
            "manifest_payload_sha256": sealed_result["manifest"]["manifest_payload_sha256"],
            "result": sealed_result,
            "result_file_sha256": sealed_evidence["result_file_sha256"],
            "result_payload_sha256": canonical_json_sha256(sealed_result),
            "receipt_file_sha256": sealed_evidence["receipt_file_sha256"],
            "receipt": sealed_evidence["receipt"],
        },
        "production_provider": production_provider,
        "onnxruntime_directml_version": directml_version,
        "onnxruntime_parity": parity,
        "environment": environment,
        "source_file_sha256": source_hashes(),
    }


def export_onnx_release(
    candidate_path: str | Path,
    output_folder: str | Path,
    *,
    atlas: str | Path = training.DEFAULT_ATLAS,
    device: str = "cuda",
) -> tuple[Path, Path]:
    """Atomically publish a sealed-passing model plus verified format-v2 metadata."""
    environment = deterministic_environment(device)
    candidate, sealed_result, sealed_evidence = verify_passing_sealed_receipt(candidate_path)
    destination = Path(output_folder).resolve()
    final_model = destination / "dense_registration.onnx"
    final_metadata = destination / "dense_registration.metadata.json"
    if destination.exists():
        verified = verify_dense_registration_v2_bundle(final_model, final_metadata)
        if verified.get("candidate_payload_sha256") != candidate[
            "candidate_payload_sha256"
        ]:
            raise ValueError("published bundle belongs to another candidate")
        return final_model, final_metadata
    _, checkpoint_path = _verified_candidate(candidate_path)
    identity, checkpoint, model, generator = _load_release_components(
        checkpoint_path, atlas, device
    )
    export_model = model
    if torch.device(device).type != "cpu":
        export_model, _ = training.model_from_checkpoint(
            checkpoint_path, "cpu", use_ema=True
        )
        export_model.eval()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    staged_model = staging / "dense_registration.onnx"
    staged_metadata = staging / "dense_registration.metadata.json"
    try:
        fixed, moving = _onnx_parity_inputs("cpu")
        torch.onnx.export(
            export_model, (fixed, moving), staged_model,
            input_names=["fixed_atlas_and_mask", "moving_slice_and_mask"],
            output_names=["fixed_to_moving_map", "moving_to_fixed_map"],
            opset_version=17, dynamo=False,
        )
        parity = run_onnx_parity(
            staged_model, model, generator
        )
        if not parity["passed"]:
            raise RuntimeError("ONNX v2 parity failed")
        metadata = build_v2_release_metadata(
            onnx_model_file_sha256=sha256_file(staged_model),
            candidate=candidate,
            checkpoint=checkpoint,
            sealed_result=sealed_result,
            sealed_evidence=sealed_evidence,
            parity=parity,
            environment=environment,
            candidate_file_sha256=sha256_file(candidate_path),
        )
        _write_once(staged_metadata, metadata)
        verify_dense_registration_v2_bundle(staged_model, staged_metadata)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_model, final_metadata
