"""One-shot animal-disjoint release gate for a frozen nonlinear ONNX model."""

from __future__ import annotations

import json
import os
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
    if not candidate_manifest.get("synthetic_gate_passed") or not candidate_manifest.get("onnx_gate_passed"):
        raise ValueError("Locked real-histology evaluation requires a synthetic- and ONNX-gated candidate")
    if candidate_manifest.get("real_histology_gate_passed") or candidate_manifest.get("promotion_ready"):
        raise ValueError("Candidate already claims locked real-histology evidence")

    source = RegisteredHistologySource(registered_root, atlas_folder)
    evaluation_manifest = source.evaluation_manifest("test", REAL_HISTOLOGY_LOCKED_SEED)
    model = OnnxRegistrationModel(candidate_path)
    output_folder = Path(output_folder)
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
        "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
        "evaluation_manifest_artifact_sha256": file_sha256(manifest_path),
        "locked_test_report_sha256": report["report_sha256"],
        "locked_test_report_artifact_sha256": file_sha256(report_path),
        "proposed_model_manifest_sha256": file_sha256(proposed_path),
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
