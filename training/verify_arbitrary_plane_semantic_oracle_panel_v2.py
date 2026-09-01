"""Independent read-only verifier for a frozen v2 semantic-oracle panel."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import nrrd

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_semantic_oracle_null_gate_v2 as null_gate_v2
from training.arbitrary_plane_semantic_oracle_panel_v2 import (
    SEMANTIC_ORACLE_PANEL_V2_SCHEMA,
    adapt_development_panel_records_to_semantic_gate_v2,
    arbitrary_plane_semantic_oracle_development_panel_v2,
    verify_arbitrary_plane_semantic_oracle_case_record_v2,
    verify_arbitrary_plane_semantic_oracle_development_panel_v2,
)


_RUN_SCHEMA = "anatomy-tracker.arbitrary-plane-semantic-oracle-development-run/v2"
_QUALIFICATION_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-live-qualification/v2"
)
_LIVE_VERIFICATION_MODES = {"newly-evaluated", "strict-replayed"}
_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_SOURCE = Path(__file__).with_name(
    "run_arbitrary_plane_semantic_oracle_panel_v2.py"
)
_OUTPUT_ENV = "ANATOMY_TRACKER_SEMANTIC_ORACLE_PANEL_V2_OUTPUT"
_ATLAS_ENV = "ANATOMY_TRACKER_ATLAS_FOLDER"
_TEMPLATE_SOURCE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
_ANNOTATION_SOURCE_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
_PINNED_PYNRRD_VERSION = "1.1.3"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_pinned_allen_assets_v2(
    atlas_folder: str | Path,
) -> dict[str, object]:
    requested = Path(atlas_folder)
    if not requested.is_absolute():
        raise ValueError("atlas_folder must be an explicit absolute path")
    atlas_folder = requested.resolve()
    if str(nrrd.__version__) != _PINNED_PYNRRD_VERSION:
        raise RuntimeError(
            f"pynrrd version {nrrd.__version__!s} does not match pinned version "
            f"{_PINNED_PYNRRD_VERSION}."
        )
    template_path = atlas_folder / "average_template_25.nrrd"
    annotation_path = atlas_folder / "annotation_25.nrrd"
    template_sha256 = _file_sha256(template_path)
    annotation_sha256 = _file_sha256(annotation_path)
    if template_sha256 != _TEMPLATE_SOURCE_SHA256:
        raise ValueError("pinned Allen average-template source hash mismatch")
    if annotation_sha256 != _ANNOTATION_SOURCE_SHA256:
        raise ValueError("pinned Allen annotation source hash mismatch")
    return {
        "atlas_folder": str(atlas_folder),
        "template_file": template_path.name,
        "template_source_sha256": template_sha256,
        "annotation_file": annotation_path.name,
        "annotation_source_sha256": annotation_sha256,
        "decoder": "pynrrd",
        "decoder_version": _PINNED_PYNRRD_VERSION,
        "index_order": "F",
    }


def _complete_record_v2_context_sha256(
    records: list[Mapping[str, object]],
) -> str | None:
    context_ids = set()
    for record in records:
        if record["status"] != "complete":
            continue
        primary_context_id = record["semantic_oracle_result"].get(
            "upstream_reference", {}
        ).get("v2_context_sha256")
        null_context_id = record["semantic_null_result"].get(
            "upstream_reference", {}
        ).get("v2_context_sha256")
        if (
            not isinstance(primary_context_id, str)
            or len(primary_context_id) != 64
            or any(character not in "0123456789abcdef" for character in primary_context_id)
            or null_context_id != primary_context_id
        ):
            raise ValueError(
                "complete primary/null records do not share a valid v2 context ID"
            )
        context_ids.add(primary_context_id)
    if len(context_ids) > 1:
        raise ValueError("complete records were generated from multiple v2 context IDs")
    return next(iter(context_ids)) if context_ids else None


def _load_canonical_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_bytes = (
        acquisition._canonical_json(acquisition._json_value(value)) + "\n"
    ).encode("utf-8")
    if path.read_bytes() != expected_bytes:
        raise ValueError(f"frozen JSON is not canonical: {path.name}")
    return value


def _expected_live_qualification(
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
    modes: Mapping[str, str],
) -> dict[str, object]:
    case_ids = [case["case_id"] for case in panel["cases"]]
    record_by_id = {record["case_spec"]["case_id"]: record for record in records}
    if (
        len(case_ids) != 24
        or len(set(case_ids)) != 24
        or set(record_by_id) != set(case_ids)
        or set(modes) != set(case_ids)
        or any(mode not in _LIVE_VERIFICATION_MODES for mode in modes.values())
    ):
        raise ValueError("live qualification does not cover the exact 24-case panel")
    cases = []
    for case_id in case_ids:
        record = record_by_id[case_id]
        complete = record["status"] == "complete"
        cases.append(
            {
                "case_id": case_id,
                "case_status": record["status"],
                "case_receipt_sha256": record["case_receipt_sha256"],
                "live_verification_mode": modes[case_id],
                "primary_result_id": record["semantic_oracle_result"][
                    "semantic_oracle_result_id"
                ]
                if complete
                else None,
                "primary_result_receipt_sha256": record["semantic_oracle_result"][
                    "receipt_sha256"
                ]
                if complete
                else None,
                "null_result_id": record["semantic_null_result"][
                    "semantic_null_result_id"
                ]
                if complete
                else None,
                "null_result_receipt_sha256": record["semantic_null_result"][
                    "receipt_sha256"
                ]
                if complete
                else None,
            }
        )
    payload = {
        "schema_version": _QUALIFICATION_SCHEMA,
        "claim_scope": (
            "runner-level live replay qualification for freezing the small "
            "development engineering gate; not a benchmark or self-receipt-only audit"
        ),
        "panel_receipt_sha256": panel["panel_receipt_sha256"],
        "planned_case_count": 24,
        "live_verified_case_count": len(cases),
        "all_planned_cases_live_verified_in_qualifying_execution": True,
        "all_complete_cases_primary_and_null_verified_from_live_upstream_inputs": True,
        "new_cases_qualify_after_in_memory_make_and_verify": True,
        "preexisting_cases_qualify_only_after_strict_live_replay": True,
        "self_receipted_preexisting_case_alone_qualifies": False,
        "live_verification_mode_counts": {
            mode: sum(item["live_verification_mode"] == mode for item in cases)
            for mode in sorted(_LIVE_VERIFICATION_MODES)
        },
        "case_evidence": cases,
        "implementation_source_sha256": acquisition._normalized_text_sha256(
            _RUNNER_SOURCE
        ),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }
    payload["qualification_evidence_receipt_sha256"] = acquisition._payload_sha256(
        payload
    )
    return payload


def _expected_run_summary(
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
    gate_summary: Mapping[str, object],
    qualification: Mapping[str, object],
) -> dict[str, object]:
    completed = sum(record["status"] == "complete" for record in records)
    failed = len(records) - completed
    unevaluable = sum(
        record["later_gate_policy"]["evaluable"] is False for record in records
    )
    payload = {
        "schema_version": _RUN_SCHEMA,
        "panel_schema_version": SEMANTIC_ORACLE_PANEL_V2_SCHEMA,
        "panel_receipt_sha256": panel["panel_receipt_sha256"],
        "claim_scope": (
            "development engineering premise and failure-adverse gates only; no "
            "benchmark, learned-model, posterior, or uncertainty-calibration claim"
        ),
        "scheduled_case_count": len(panel["cases"]),
        "recorded_case_count": len(records),
        "completed_case_count": completed,
        "failed_case_count": failed,
        "unevaluable_or_failed_case_count": unevaluable,
        "all_scheduled_cases_recorded": len(records) == len(panel["cases"]),
        "failure_adverse_gate_summary_computed": True,
        "gate_summary_file": "gate-summary.json",
        "semantic_gate_summary_id": gate_summary["semantic_gate_summary_id"],
        "semantic_gate_summary_receipt_sha256": gate_summary["receipt_sha256"],
        "semantic_gate_summary_passed": bool(gate_summary["passed"]),
        "gate_evaluated_here": True,
        "live_qualification_evidence_file": "live-qualification.json",
        "live_qualification_evidence": acquisition._json_value(qualification),
        "all_cases_live_verified_when_gate_frozen": True,
        "case_receipts": [record["case_receipt_sha256"] for record in records],
        "failure_and_zero_support_denominator_policy": (
            "all 24 scheduled cases remain in later denominators; failures and "
            "zero-support truth cannot be redrawn"
        ),
    }
    payload["run_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def verify_frozen_arbitrary_plane_semantic_oracle_panel_v2(
    output_folder: str | Path,
) -> dict[str, object]:
    output = Path(output_folder).resolve()
    expected_files = {
        "panel.json",
        "live-qualification.json",
        "gate-summary.json",
        "result.json",
        *{f"cases/case-{case_index:03d}.json" for case_index in range(24)},
    }
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != {"cases"}:
        raise ValueError("frozen output tree is incomplete or contains unexpected entries")

    panel = _load_canonical_json(output / "panel.json")
    expected_panel = arbitrary_plane_semantic_oracle_development_panel_v2()
    verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)
    if acquisition._canonical_json(panel) != acquisition._canonical_json(expected_panel):
        raise ValueError("frozen panel does not match the source-bound 24-case panel")

    records = []
    for case_spec in panel["cases"]:
        record = _load_canonical_json(
            output / "cases" / f"case-{case_spec['case_index']:03d}.json"
        )
        verify_arbitrary_plane_semantic_oracle_case_record_v2(
            record, panel, case_spec
        )
        records.append(record)
    complete_record_context_id = _complete_record_v2_context_sha256(records)

    qualification = _load_canonical_json(output / "live-qualification.json")
    modes = {
        item["case_id"]: item["live_verification_mode"]
        for item in qualification.get("case_evidence", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("case_id"), str)
        and isinstance(item.get("live_verification_mode"), str)
    }
    expected_qualification = _expected_live_qualification(panel, records, modes)
    if acquisition._canonical_json(qualification) != acquisition._canonical_json(
        expected_qualification
    ):
        raise ValueError("frozen live qualification evidence changed")

    planned_cases, gate_records = adapt_development_panel_records_to_semantic_gate_v2(
        panel, records
    )
    gate_summary = _load_canonical_json(output / "gate-summary.json")
    null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
        gate_summary,
        planned_cases,
        gate_records,
        expected_panel_id=panel["panel_receipt_sha256"],
    )

    result = _load_canonical_json(output / "result.json")
    expected_result = _expected_run_summary(
        panel, records, gate_summary, qualification
    )
    if acquisition._canonical_json(result) != acquisition._canonical_json(
        expected_result
    ):
        raise ValueError("frozen panel run summary changed")

    experiment_ids = sorted({case["experiment_id"] for case in panel["cases"]})
    return {
        "event": "arbitrary-plane-semantic-oracle-panel-v2-verified",
        "output_folder": str(output),
        "panel_receipt_sha256": panel["panel_receipt_sha256"],
        "run_receipt_sha256": result["run_receipt_sha256"],
        "semantic_gate_summary_passed": result["semantic_gate_summary_passed"],
        "verified_case_count": len(records),
        "complete_record_v2_context_sha256": complete_record_context_id,
        "animal_ids": [animal["animal_id"] for animal in panel["animals"]],
        "specimen_ids": [animal["specimen_id"] for animal in panel["animals"]],
        "experiment_ids": experiment_ids,
        "verifier_source_sha256": acquisition._normalized_text_sha256(Path(__file__)),
        "verifier_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }


def main() -> None:
    atlas_raw = os.environ.get(_ATLAS_ENV)
    if not atlas_raw:
        raise ValueError(f"{_ATLAS_ENV} must name an explicit absolute atlas directory.")
    atlas_authentication = authenticate_pinned_allen_assets_v2(atlas_raw)
    raw = os.environ.get(_OUTPUT_ENV)
    if not raw:
        raise ValueError(f"{_OUTPUT_ENV} must name an explicit absolute output directory.")
    requested = Path(raw)
    if not requested.is_absolute():
        raise ValueError(f"{_OUTPUT_ENV} must be an absolute path.")
    output = requested.resolve()
    if output == _ROOT or _ROOT in output.parents:
        raise ValueError(f"{_OUTPUT_ENV} must be outside the repository.")
    report = verify_frozen_arbitrary_plane_semantic_oracle_panel_v2(output)
    report["allen_asset_authentication"] = atlas_authentication
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
