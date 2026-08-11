"""One-shot animal-disjoint release gate for a frozen nonlinear ONNX model."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from training.real_histology_registration import (
    REAL_HISTOLOGY_LOCKED_SEED,
    RegisteredHistologySource,
    evaluate_real_histology,
    file_sha256,
)
from training.train_diffeomorphic_registration import (
    ATLAS,
    OnnxRegistrationModel,
    make_synthetic_pair,
)


def run_locked_evaluation(
    candidate_path: str | Path,
    registered_root: str | Path,
    output_folder: str | Path,
    atlas_folder: str | Path = ATLAS,
) -> dict:
    candidate_path = Path(candidate_path)
    candidate_manifest_path = candidate_path.with_suffix(".manifest.json")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    model_sha256 = file_sha256(candidate_path)
    if candidate_manifest.get("model_sha256") != model_sha256:
        raise ValueError("Frozen nonlinear candidate differs from its manifest")
    if candidate_manifest.get("real_histology_gate_passed") or candidate_manifest.get("promotion_ready"):
        raise ValueError("Candidate already claims locked real-histology evidence")
    evidence_path = candidate_path.with_suffix(".prelocked.json")
    if candidate_manifest.get("prelocked_evidence_file") != evidence_path.name:
        raise ValueError("Candidate manifest does not name its canonical prelocked evidence")
    if not evidence_path.is_file() or candidate_manifest.get("prelocked_evidence_sha256") != file_sha256(evidence_path):
        raise ValueError("Candidate prelocked evidence is missing or fails its SHA-256 commitment")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("model_sha256") != model_sha256:
        raise ValueError("Candidate prelocked evidence belongs to a different model")
    synthetic_passed = evidence.get("synthetic_gate", {}).get("passed") is True
    onnx_passed = evidence.get("onnx_gate", {}).get("passed") is True
    if (
        not synthetic_passed
        or not onnx_passed
        or candidate_manifest.get("synthetic_gate_passed") is not synthetic_passed
        or candidate_manifest.get("onnx_gate_passed") is not onnx_passed
    ):
        raise ValueError("Locked real-histology evaluation requires hashed synthetic and ONNX evidence")
    commitment = evidence.get("locked_real_histology_commitment")
    if not isinstance(commitment, dict) or candidate_manifest.get(
        "locked_real_histology_commitment"
    ) != commitment:
        raise ValueError("Candidate has no consistent locked real-histology commitment")
    output_folder = Path(output_folder)
    if output_folder.exists():
        raise FileExistsError(f"Locked evaluation output already exists: {output_folder}")
    consumption_root = Path(registered_root) / ".nonlinear_locked_test_consumption"
    consumption_root.mkdir(parents=True, exist_ok=True)
    claim_path = consumption_root / f"{model_sha256}.claim"
    consumption_path = consumption_root / f"{model_sha256}.json"
    consumption = {
        "model_sha256": model_sha256,
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "prelocked_evidence_sha256": file_sha256(evidence_path),
        "registered_root": str(Path(registered_root).resolve()),
        "atlas_folder": str(Path(atlas_folder).resolve()),
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_consumed_before_source_access": True,
        "sealed_data_used": False,
    }
    try:
        with claim_path.open("x", encoding="ascii") as stream:
            stream.write(model_sha256 + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"Locked test split was already consumed for candidate {model_sha256}"
        ) from error
    temporary_receipt = consumption_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary_receipt.open("w", encoding="utf-8") as stream:
        json.dump(consumption, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_receipt, consumption_path)

    source = RegisteredHistologySource(registered_root, atlas_folder)
    evaluation_manifest = source.evaluation_manifest("test", REAL_HISTOLOGY_LOCKED_SEED)
    if (
        commitment.get("source") != source.contract
        or commitment.get("evaluation_manifest_sha256") != evaluation_manifest["manifest_sha256"]
    ):
        raise ValueError("Locked real-histology source differs from the candidate commitment")
    model = OnnxRegistrationModel(candidate_path)
    output_folder.mkdir(parents=True, exist_ok=False)
    manifest_path = output_folder / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(evaluation_manifest, indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = evaluate_real_histology(
        model,
        source,
        evaluation_manifest,
        make_synthetic_pair,
        device,
        model_sha256,
    )
    report_path = output_folder / "locked_test_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    proposed_manifest = {
        **candidate_manifest,
        "real_histology_gate_passed": bool(report["passed"]),
        "real_histology_gate_report_sha256": report["report_sha256"],
        "real_histology_evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
        "real_histology_source": source.contract,
        "real_histology_benchmark_role": "locked_promotion_gate",
        "promotion_ready": bool(report["passed"]),
    }
    proposed_path = output_folder / "proposed_model_manifest.json"
    proposed_path.write_text(json.dumps(proposed_manifest, indent=2, sort_keys=True), encoding="utf-8")
    release = {
        "model_sha256": model_sha256,
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "prelocked_evidence_sha256": file_sha256(evidence_path),
        "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
        "evaluation_manifest_artifact_sha256": file_sha256(manifest_path),
        "locked_test_report_sha256": report["report_sha256"],
        "locked_test_report_artifact_sha256": file_sha256(report_path),
        "proposed_model_manifest_sha256": file_sha256(proposed_path),
        "consumption_receipt_sha256": file_sha256(consumption_path),
        "test_split_consumed": True,
        "sealed_data_used": False,
        "promotion_approved": bool(report["passed"]),
    }
    release_path = output_folder / "release_receipt.json"
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True), encoding="utf-8")
    return release


if __name__ == "__main__":
    print(json.dumps(run_locked_evaluation(
        os.environ["DIFFEO_LOCKED_CANDIDATE"],
        os.environ["DIFFEO_REGISTERED_ROOT"],
        os.environ["DIFFEO_LOCKED_OUTPUT"],
    ), indent=2))
