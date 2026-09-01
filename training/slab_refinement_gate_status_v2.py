"""Authenticated decision for the retired two-level slab-refinement gate."""

from pathlib import Path

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_synthetic_generator_v2 as slab


SCHEMA = "anatomy-tracker.v2-slab-refinement-gate-decision/v1"


def legacy_gate_contract() -> dict[str, object]:
    return {
        "decision": "reject_legacy_universal_gate",
        "qualification_eligible": False,
        "reason": (
            "one universal MAE/p99 envelope mixes smooth scalar quadrature with "
            "support and label boundaries, categorical flips, and the threshold-amplified "
            "dense correspondence transform"
        ),
        "legacy_comparison": {
            "axial_steps_um_max": [12.5, 6.25],
            "mean_absolute_error_max": 0.02,
            "absolute_error_p99_max": 0.10,
            "thresholds_changed": False,
        },
        "replacement_experiment": {
            "status": "pending",
            "case_selection": (
                "first deterministic failing pose and animal; retain the accepted support "
                "attempt and do not redraw"
            ),
            "axial_steps_um_max": [12.5, 6.25, 3.125, 1.5625],
            "controls": [
                "same nonidentity subject deformation at every resolution",
                "repeat the identical pose with identity deformation",
            ],
            "provisional_reference_axial_step_um_max": 1.5625,
            "retain_raw_arrays": True,
            "metric_families": [
                "smooth normalized scalar",
                "support occupancy and label mass",
                "threshold-amplified dense correspondence weight",
                "categorical label, support, and abstention flips",
            ],
            "spatial_strata": [
                "stable interior",
                "axial or tissue-support boundary",
                "atlas-label boundary",
            ],
            "strict_invariants": [
                "same finite pose and support attempt",
                "byte-identical centre-plane coordinates, labels, and support",
            ],
            "threshold_policy": (
                "predeclare metric-specific and boundary-aware convergence rules only "
                "after inspecting the fixed multiresolution experiment; do not loosen "
                "the legacy thresholds or drop difficult planes"
            ),
        },
    }


def assess_rejected_legacy_gate(
    report: dict[str, object], prepared_context: dict[str, object]
) -> dict[str, object]:
    if (
        report.get("v2_context_sha256") != prepared_context.get("v2_context_sha256")
        or report.get("qualification_receipt_sha256")
        != acquisition._payload_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "qualification_receipt_sha256"
            }
        )
    ):
        raise ValueError("legacy slab assessment context or receipt does not match")
    expected = slab.evaluate_v2_slab_refinement_smoke(prepared_context)
    if acquisition._canonical_json(report) != acquisition._canonical_json(expected):
        raise ValueError("legacy slab assessment does not replay exactly")
    if report.get("all_cases_passed") is not False:
        raise ValueError("retired legacy gate assessment did not reproduce its rejection")
    cases = report["cases"]
    payload = {
        "schema_version": SCHEMA,
        "gate_contract": legacy_gate_contract(),
        "legacy_report_receipt_sha256": report["qualification_receipt_sha256"],
        "legacy_report_schema_version": report["schema_version"],
        "legacy_numerical_outcome": {
            "case_count": report["case_count"],
            "passing_case_indices": [
                case["sample_index"] for case in cases if case["passed"]
            ],
            "failing_case_indices": [
                case["sample_index"] for case in cases if not case["passed"]
            ],
        },
        "decision_source_sha256": acquisition._normalized_text_sha256(Path(__file__)),
    }
    payload["decision_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def save_gate_decision(path: str | Path, decision: dict[str, object]) -> None:
    payload = {
        key: value for key, value in decision.items() if key != "decision_receipt_sha256"
    }
    if decision.get("decision_receipt_sha256") != acquisition._payload_sha256(payload):
        raise ValueError("slab gate decision receipt does not match")
    Path(path).write_text(
        acquisition._canonical_json(decision) + "\n", encoding="utf-8", newline="\n"
    )
