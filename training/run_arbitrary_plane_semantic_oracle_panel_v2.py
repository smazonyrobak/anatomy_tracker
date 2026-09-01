"""Atomic resumable runner for the development-only v2 semantic-oracle panel."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import nrrd

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_support import build_annotation_support_index
import training.arbitrary_plane_semantic_oracle_null_gate_v2 as null_gate_v2
from training.arbitrary_plane_semantic_oracle_panel_v2 import (
    SEMANTIC_ORACLE_PANEL_V2_SCHEMA,
    _evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2,
    _make_development_panel_subject_plan_with_mapper_v2,
    arbitrary_plane_semantic_oracle_development_panel_v2,
    adapt_development_panel_records_to_semantic_gate_v2,
    make_arbitrary_plane_semantic_oracle_failure_record_v2,
    verify_arbitrary_plane_semantic_oracle_case_record_v2,
    verify_arbitrary_plane_semantic_oracle_development_panel_v2,
)


SEMANTIC_ORACLE_PANEL_RUN_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-development-run/v2"
)
SEMANTIC_ORACLE_PANEL_LIVE_QUALIFICATION_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-live-qualification/v2"
)
_LIVE_VERIFICATION_MODES = {"newly-evaluated", "strict-replayed"}
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ATLAS_FOLDER = _ROOT / "data" / "Allen Brain Atlas 25um"
_TEMPLATE_SOURCE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
_ANNOTATION_SOURCE_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
_PINNED_PYNRRD_VERSION = "1.1.3"
_RUN_ENV = "ANATOMY_TRACKER_RUN_SEMANTIC_ORACLE_PANEL_V2"
_OUTPUT_ENV = "ANATOMY_TRACKER_SEMANTIC_ORACLE_PANEL_V2_OUTPUT"
_STRICT_REPLAY_ENV = "ANATOMY_TRACKER_SEMANTIC_ORACLE_PANEL_V2_STRICT_REPLAY"
_BATCH_SIZE_ENV = "ANATOMY_TRACKER_SEMANTIC_ORACLE_PANEL_V2_BATCH_SIZE"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_allen_context_v2(atlas_folder: str | Path) -> Mapping[str, object]:
    atlas_folder = Path(atlas_folder)
    template_path = atlas_folder / "average_template_25.nrrd"
    annotation_path = atlas_folder / "annotation_25.nrrd"
    if str(nrrd.__version__) != _PINNED_PYNRRD_VERSION:
        raise RuntimeError(
            f"pynrrd version {nrrd.__version__!s} does not match pinned version "
            f"{_PINNED_PYNRRD_VERSION}."
        )
    if _file_sha256(template_path) != _TEMPLATE_SOURCE_SHA256:
        raise RuntimeError("Pinned Allen average-template source hash mismatch.")
    if _file_sha256(annotation_path) != _ANNOTATION_SOURCE_SHA256:
        raise RuntimeError("Pinned Allen annotation source hash mismatch.")

    template = nrrd.read(str(template_path), index_order="F")[0]
    annotation = nrrd.read(str(annotation_path), index_order="F")[0]
    support = build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri="data/Allen Brain Atlas 25um/annotation_25.nrrd",
        source_sha256=_ANNOTATION_SOURCE_SHA256,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        template,
        annotation,
        support,
        scalar_source_uri="data/Allen Brain Atlas 25um/average_template_25.nrrd",
        scalar_source_sha256=_TEMPLATE_SOURCE_SHA256,
        scalar_source_entity_type="atlas-template",
        template_decoder=f"pynrrd {_PINNED_PYNRRD_VERSION}",
        template_index_order="F",
        annotation_decoder=f"pynrrd {_PINNED_PYNRRD_VERSION}",
        annotation_index_order="F",
    )


def _external_output_folder_from_environment() -> Path:
    raw = os.environ.get(_OUTPUT_ENV)
    if not raw:
        raise ValueError(f"{_OUTPUT_ENV} must name an explicit absolute output directory.")
    requested = Path(raw)
    if not requested.is_absolute():
        raise ValueError(f"{_OUTPUT_ENV} must be an absolute path.")
    output_folder = requested.resolve()
    if output_folder == _ROOT or _ROOT in output_folder.parents:
        raise ValueError(f"{_OUTPUT_ENV} must be outside the repository.")
    return output_folder


def _canonical_bytes(value: object) -> bytes:
    return (acquisition._canonical_json(acquisition._json_value(value)) + "\n").encode(
        "utf-8"
    )


def _atomic_immutable_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace changed frozen output: {path.name}")
        return digest
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic temporary requires audit: {temporary.name}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if path.read_bytes() != payload:
        raise RuntimeError("atomic semantic-oracle panel output changed after replace")
    return digest


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit_milestone(payload: Mapping[str, object]) -> None:
    print(
        json.dumps(
            acquisition._json_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _emit_case_milestone(
    event: str,
    case_spec: Mapping[str, object],
    case_count: int,
    execution_mode: str,
    *,
    record: Mapping[str, object] | None = None,
    caught_error: Exception | None = None,
) -> None:
    payload = {
        "event": event,
        "case_index": int(case_spec["case_index"]),
        "case_number": int(case_spec["case_index"]) + 1,
        "case_count": int(case_count),
        "case_id": case_spec["case_id"],
        "plane_stratum": case_spec["plane_stratum"],
        "execution_mode": execution_mode,
    }
    if record is None:
        payload.update(
            {
                "verification_status": "running",
                "record_status": None,
                "scientific_outcome": "pending",
                "caught_exception_type": None,
                "caught_exception_status": "none",
                "recorded_failure_exception_type": None,
            }
        )
    else:
        record_status = str(record["status"])
        payload.update(
            {
                "verification_status": "verified",
                "record_status": record_status,
                "scientific_outcome": (
                    "failure-adverse-pipeline-failure"
                    if record_status == "failed"
                    else "complete-evaluable"
                    if record["later_gate_policy"]["evaluable"]
                    else "complete-unevaluable"
                ),
                "caught_exception_type": (
                    None if caught_error is None else type(caught_error).__name__
                ),
                "caught_exception_status": (
                    "none"
                    if caught_error is None
                    else "converted-to-failure-adverse-record"
                ),
                "recorded_failure_exception_type": (
                    record.get("failure", {}).get("exception_type")
                    if record_status == "failed"
                    else None
                ),
            }
        )
    _emit_milestone(payload)


def _live_qualification_evidence(
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
    live_verification_modes: Mapping[str, str],
) -> dict[str, object]:
    case_ids = [case["case_id"] for case in panel["cases"]]
    record_by_id = {record["case_spec"]["case_id"]: record for record in records}
    if (
        len(case_ids) != 24
        or len(set(case_ids)) != 24
        or set(record_by_id) != set(case_ids)
        or set(live_verification_modes) != set(case_ids)
        or any(mode not in _LIVE_VERIFICATION_MODES for mode in live_verification_modes.values())
    ):
        raise ValueError("live qualification requires all 24 planned cases in one execution")
    cases = []
    for case_id in case_ids:
        record = record_by_id[case_id]
        complete = record["status"] == "complete"
        cases.append(
            {
                "case_id": case_id,
                "case_status": record["status"],
                "case_receipt_sha256": record["case_receipt_sha256"],
                "live_verification_mode": live_verification_modes[case_id],
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
    mode_counts = {
        mode: sum(item["live_verification_mode"] == mode for item in cases)
        for mode in sorted(_LIVE_VERIFICATION_MODES)
    }
    payload = {
        "schema_version": SEMANTIC_ORACLE_PANEL_LIVE_QUALIFICATION_V2_SCHEMA,
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
        "live_verification_mode_counts": mode_counts,
        "case_evidence": cases,
        "implementation_source_sha256": acquisition._normalized_text_sha256(
            Path(__file__)
        ),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }
    payload["qualification_evidence_receipt_sha256"] = acquisition._payload_sha256(
        payload
    )
    return payload


def verify_semantic_oracle_panel_live_qualification_v2(
    evidence: Mapping[str, object],
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
) -> None:
    modes = {
        item["case_id"]: item["live_verification_mode"]
        for item in evidence.get("case_evidence", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("case_id"), str)
        and isinstance(item.get("live_verification_mode"), str)
    }
    expected = _live_qualification_evidence(panel, records, modes)
    if acquisition._canonical_json(acquisition._json_value(evidence)) != acquisition._canonical_json(
        expected
    ):
        raise ValueError("semantic-oracle live qualification evidence changed")


def _run_summary(
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
    gate_summary: Mapping[str, object],
    qualification_evidence: Mapping[str, object],
) -> dict[str, object]:
    completed = sum(record["status"] == "complete" for record in records)
    failed = len(records) - completed
    unevaluable = sum(
        record["later_gate_policy"]["evaluable"] is False for record in records
    )
    payload = {
        "schema_version": SEMANTIC_ORACLE_PANEL_RUN_V2_SCHEMA,
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
        "live_qualification_evidence": acquisition._json_value(
            qualification_evidence
        ),
        "all_cases_live_verified_when_gate_frozen": True,
        "case_receipts": [record["case_receipt_sha256"] for record in records],
        "failure_and_zero_support_denominator_policy": (
            "all 24 scheduled cases remain in later denominators; failures and "
            "zero-support truth cannot be redrawn"
        ),
    }
    payload["run_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def verify_arbitrary_plane_semantic_oracle_panel_run_v2(
    result: Mapping[str, object],
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
    gate_summary: Mapping[str, object],
    qualification_evidence: Mapping[str, object],
) -> None:
    planned_cases, gate_records = adapt_development_panel_records_to_semantic_gate_v2(
        panel, records
    )
    null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
        gate_summary,
        planned_cases,
        gate_records,
        expected_panel_id=panel["panel_receipt_sha256"],
    )
    verify_semantic_oracle_panel_live_qualification_v2(
        qualification_evidence, panel, records
    )
    expected = _run_summary(panel, records, gate_summary, qualification_evidence)
    if acquisition._canonical_json(acquisition._json_value(result)) != acquisition._canonical_json(
        expected
    ):
        raise ValueError("semantic-oracle panel run summary changed")


def run_arbitrary_plane_semantic_oracle_development_panel_v2(
    prepared_context: Mapping[str, object],
    output_folder: str | Path,
    *,
    batch_size: int | None = None,
    strict_replay_existing: bool = False,
) -> dict[str, object]:
    """Run pending cases; optionally regenerate existing records before reuse."""
    acquisition._validate_v2_context(prepared_context)
    output = Path(output_folder)
    panel = arbitrary_plane_semantic_oracle_development_panel_v2()
    verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)
    _atomic_immutable_json(output / "panel.json", panel)
    qualification_path = output / "live-qualification.json"
    gate_path = output / "gate-summary.json"
    result_path = output / "result.json"
    frozen_presence = tuple(
        path.exists() for path in (qualification_path, gate_path, result_path)
    )
    if any(frozen_presence) and not all(frozen_presence):
        raise RuntimeError(
            "incomplete frozen live-qualification/gate/result outputs require audit"
        )
    frozen_qualification_exists = all(frozen_presence)
    case_count = len(panel["cases"])
    _emit_milestone(
        {
            "event": "arbitrary-plane-semantic-oracle-panel-v2-started",
            "output_folder": str(output.resolve()),
            "panel_receipt_sha256": panel["panel_receipt_sha256"],
            "case_count": case_count,
            "batch_size": batch_size,
            "strict_replay_existing": bool(strict_replay_existing),
            "frozen_summary_present": frozen_qualification_exists,
        }
    )
    animal_specs = {
        animal["animal_index"]: animal for animal in panel["animals"]
    }
    subject_plans: dict[int, Mapping[str, object]] = {}
    subject_to_ccf_mappers: dict[int, object] = {}
    records = []
    live_verification_modes: dict[str, str] = {}
    for case_spec in panel["cases"]:
        case_path = output / "cases" / f"case-{case_spec['case_index']:03d}.json"
        execution_mode = (
            "strict-replayed"
            if case_path.exists() and strict_replay_existing
            else "verified-existing"
            if case_path.exists()
            else "newly-evaluated"
        )
        _emit_case_milestone(
            "arbitrary-plane-semantic-oracle-panel-v2-case-started",
            case_spec,
            case_count,
            execution_mode,
        )
        caught_error = None
        if case_path.exists():
            record = _load_json(case_path)
            verify_arbitrary_plane_semantic_oracle_case_record_v2(
                record, panel, case_spec
            )
            _atomic_immutable_json(case_path, record)
            if strict_replay_existing:
                try:
                    animal_index = int(case_spec["animal_index"])
                    if animal_index not in subject_plans:
                        (
                            subject_plans[animal_index],
                            subject_to_ccf_mappers[animal_index],
                        ) = _make_development_panel_subject_plan_with_mapper_v2(
                            prepared_context,
                            animal_specs[animal_index],
                        )
                    replay = _evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2(
                        prepared_context,
                        panel,
                        case_spec,
                        subject_plans[animal_index],
                        batch_size=batch_size,
                        subject_to_ccf_mapper=subject_to_ccf_mappers[animal_index],
                    )
                except Exception as error:
                    caught_error = error
                    replay = make_arbitrary_plane_semantic_oracle_failure_record_v2(
                        panel, case_spec, error
                    )
                verify_arbitrary_plane_semantic_oracle_case_record_v2(
                    replay, panel, case_spec
                )
                if acquisition._canonical_json(record) != acquisition._canonical_json(
                    replay
                ):
                    raise ValueError(
                        f"strict replay changed case {case_spec['case_index']:03d}"
                    )
                live_verification_modes[case_spec["case_id"]] = "strict-replayed"
        else:
            try:
                animal_index = int(case_spec["animal_index"])
                if animal_index not in subject_plans:
                    (
                        subject_plans[animal_index],
                        subject_to_ccf_mappers[animal_index],
                    ) = _make_development_panel_subject_plan_with_mapper_v2(
                        prepared_context,
                        animal_specs[animal_index],
                    )
                record = _evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2(
                    prepared_context,
                    panel,
                    case_spec,
                    subject_plans[animal_index],
                    batch_size=batch_size,
                    subject_to_ccf_mapper=subject_to_ccf_mappers[animal_index],
                )
            except Exception as error:
                caught_error = error
                record = make_arbitrary_plane_semantic_oracle_failure_record_v2(
                    panel, case_spec, error
                )
            verify_arbitrary_plane_semantic_oracle_case_record_v2(
                record, panel, case_spec
            )
            _atomic_immutable_json(case_path, record)
            live_verification_modes[case_spec["case_id"]] = "newly-evaluated"
        verify_arbitrary_plane_semantic_oracle_case_record_v2(record, panel, case_spec)
        records.append(record)
        _emit_case_milestone(
            "arbitrary-plane-semantic-oracle-panel-v2-case-verified",
            case_spec,
            case_count,
            execution_mode,
            record=record,
            caught_error=caught_error,
        )
    planned_cases, gate_records = adapt_development_panel_records_to_semantic_gate_v2(
        panel, records
    )
    if frozen_qualification_exists:
        qualification_evidence = _load_json(qualification_path)
        verify_semantic_oracle_panel_live_qualification_v2(
            qualification_evidence, panel, records
        )
        gate_summary = _load_json(gate_path)
        null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
            gate_summary,
            planned_cases,
            gate_records,
            expected_panel_id=panel["panel_receipt_sha256"],
        )
        result = _load_json(result_path)
        verify_arbitrary_plane_semantic_oracle_panel_run_v2(
            result,
            panel,
            records,
            gate_summary,
            qualification_evidence,
        )
        _atomic_immutable_json(qualification_path, qualification_evidence)
        _atomic_immutable_json(gate_path, gate_summary)
        _atomic_immutable_json(result_path, result)
        return result
    if set(live_verification_modes) != {
        case["case_id"] for case in panel["cases"]
    }:
        raise RuntimeError(
            "case files were retained, but gate-summary.json and result.json were not "
            "written because every case must be live verified in the qualifying "
            "execution; rerun with strict_replay_existing=True"
        )
    qualification_evidence = _live_qualification_evidence(
        panel, records, live_verification_modes
    )
    verify_semantic_oracle_panel_live_qualification_v2(
        qualification_evidence, panel, records
    )
    gate_summary = null_gate_v2.make_arbitrary_plane_semantic_gate_summary_v2(
        planned_cases,
        gate_records,
        panel_id=panel["panel_receipt_sha256"],
    )
    null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
        gate_summary,
        planned_cases,
        gate_records,
        expected_panel_id=panel["panel_receipt_sha256"],
    )
    result = _run_summary(panel, records, gate_summary, qualification_evidence)
    verify_arbitrary_plane_semantic_oracle_panel_run_v2(
        result, panel, records, gate_summary, qualification_evidence
    )
    _atomic_immutable_json(qualification_path, qualification_evidence)
    _atomic_immutable_json(gate_path, gate_summary)
    _atomic_immutable_json(result_path, result)
    return result


def main() -> None:
    if os.environ.get(_RUN_ENV) != "1":
        raise PermissionError(f"Set {_RUN_ENV}=1 to authorize panel execution.")
    output_folder = _external_output_folder_from_environment()
    strict_replay_value = os.environ.get(_STRICT_REPLAY_ENV, "0")
    if strict_replay_value not in {"0", "1"}:
        raise ValueError(f"{_STRICT_REPLAY_ENV} must be 0 or 1.")
    batch_size_value = os.environ.get(_BATCH_SIZE_ENV)
    batch_size = None if batch_size_value is None else int(batch_size_value)
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"{_BATCH_SIZE_ENV} must be a positive integer.")
    atlas_folder = Path(
        os.environ.get("ANATOMY_TRACKER_ATLAS_FOLDER", str(_DEFAULT_ATLAS_FOLDER))
    )
    context = load_pinned_allen_context_v2(atlas_folder)
    result = run_arbitrary_plane_semantic_oracle_development_panel_v2(
        context,
        output_folder,
        batch_size=batch_size,
        strict_replay_existing=strict_replay_value == "1",
    )
    _emit_milestone(
        {
            "event": "arbitrary-plane-semantic-oracle-panel-v2-frozen",
            "output_folder": str(output_folder),
            "panel_receipt_sha256": result["panel_receipt_sha256"],
            "scheduled_case_count": result["scheduled_case_count"],
            "completed_case_count": result["completed_case_count"],
            "failed_case_count": result["failed_case_count"],
            "unevaluable_or_failed_case_count": result[
                "unevaluable_or_failed_case_count"
            ],
            "semantic_gate_summary_passed": result[
                "semantic_gate_summary_passed"
            ],
            "all_cases_live_verified_when_gate_frozen": result[
                "all_cases_live_verified_when_gate_frozen"
            ],
            "live_verification_mode_counts": result[
                "live_qualification_evidence"
            ]["live_verification_mode_counts"],
            "strict_replay_existing": strict_replay_value == "1",
        }
    )


if __name__ == "__main__":
    main()
