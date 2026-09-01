from copy import deepcopy
from inspect import signature
import json

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_realization_v2 as realization_v2
import training.arbitrary_plane_semantic_oracle_null_gate_v2 as null_gate_v2
import training.arbitrary_plane_semantic_oracle_panel_v2 as panel_v2
import training.run_arbitrary_plane_semantic_oracle_panel_v2 as runner_v2
import training.verify_arbitrary_plane_semantic_oracle_panel_v2 as verifier_v2


def test_panel_is_four_development_animals_by_all_six_frozen_strata():
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    panel_v2.verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)

    assert panel["configuration"]["plane_strata"] == list(
        acquisition.V2_GENERIC_PLANE_STRATA
    )
    assert len(panel["animals"]) == 4
    assert len(panel["cases"]) == 24
    assert all(animal["animal_id"].startswith("development-") for animal in panel["animals"])
    assert panel["cohort_policy"] == {
        "development_only": True,
        "synthetic_animals_only": True,
        "authenticated_atlas_reference_allowed": True,
        "public_benchmark_or_deepslice_cases_used": False,
        "benchmark_animals_used": False,
        "final_test_animals_used": False,
        "real_lab_histology_used": False,
        "animal_is_statistical_unit_for_future_validation": True,
    }
    assert all(not values for values in panel["asset_dependencies"].values())
    for animal in panel["animals"]:
        cases = [
            case
            for case in panel["cases"]
            if case["animal_index"] == animal["animal_index"]
        ]
        assert [case["plane_stratum"] for case in cases] == list(
            acquisition.V2_GENERIC_PLANE_STRATA
        )
        assert {
            mode: sum(case["selected_trainable_input_mode"] == mode for case in cases)
            for mode in realization_v2.TRAINABLE_INPUT_MODES
        } == {mode: 2 for mode in realization_v2.TRAINABLE_INPUT_MODES}


def test_mode_modality_and_seed_schedule_is_balanced_and_numeric_only():
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    cases = panel["cases"]

    assert {
        mode: sum(case["selected_trainable_input_mode"] == mode for case in cases)
        for mode in realization_v2.TRAINABLE_INPUT_MODES
    } == {mode: 8 for mode in realization_v2.TRAINABLE_INPUT_MODES}
    assert {
        modality: sum(case["modality"] == modality for case in cases)
        for modality in panel_v2.MODALITIES
    } == {modality: 12 for modality in panel_v2.MODALITIES}
    assert all(case["selected_trainable_input_mode"] != "raw" for case in cases)
    assert "animal_id" not in signature(
        panel_v2._realization_index_for_mode
    ).parameters
    for case in cases:
        observation_bundle = {
            "provenance": {
                "root_seed_uint64": case["root_seeds_uint64"]["observation"],
                "split": case["split"],
                "split_index": case["split_index"],
                "animal_index": case["animal_index"],
                "animal_id": case["animal_id"],
                "section_index": case["section_index"],
                "observation_index": case["observation_index"],
            }
        }
        choice = realization_v2.sample_synthetic_realization_choice_v2(
            observation_bundle, case["realization_index"]
        )
        assert choice["selected_mode"] == case["selected_trainable_input_mode"]
        assert all(
            seed.startswith("0x") and len(seed) == 18
            for seed in case["root_seeds_uint64"].values()
        )


def test_failure_record_is_receipt_bound_retained_and_never_redrawn():
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    case = panel["cases"][0]
    record = panel_v2.make_arbitrary_plane_semantic_oracle_failure_record_v2(
        panel, case, RuntimeError("scheduled failure")
    )

    panel_v2.verify_arbitrary_plane_semantic_oracle_case_record_v2(
        record, panel, case
    )
    assert record["later_gate_policy"]["included_in_all_scheduled_denominators"]
    assert record["later_gate_policy"]["top1_and_top3_success"] is False
    assert record["failure"]["redraw_or_replacement_case_created"] is False

    changed = deepcopy(record)
    changed["failure"]["redraw_or_replacement_case_created"] = True
    with pytest.raises(ValueError, match="receipt or lineage"):
        panel_v2.verify_arbitrary_plane_semantic_oracle_case_record_v2(
            changed, panel, case
        )


def test_case_evaluator_uses_existing_generator_pose_candidate_and_oracle_apis(monkeypatch):
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    case = panel["cases"][0]
    calls = []
    context = {"v2_context_sha256": "c" * 64}
    subject_plan = {
        "resolved_config": {"deformation_stratum": "standard"},
        "subject_deformation_plan_id": "1" * 64,
        "subject_deformation_realization_id": "2" * 64,
        "synthetic_animal_id": "3" * 64,
        "receipt_sha256": "4" * 64,
    }
    precursor = {
        "v2_plane_realization_id": "5" * 64,
        "slab_render_id": "6" * 64,
        "receipt_sha256": "7" * 64,
    }
    resolution = {
        "status": "accepted",
        "configuration": {"plane_stratum": case["plane_stratum"]},
        "support_resolution_plan_id": "8" * 64,
        "subject_support_resolution_id": "9" * 64,
        "receipt_sha256": "a" * 64,
    }
    support_bundle = {"resolution": resolution, "accepted_precursor": precursor}
    physical = np.zeros((2, 2, 3), dtype=np.float64)
    subject_slab = {
        "coordinate_map": {
            "kernel": {"centre_index": 0},
            "arrays": {
                "subject_physical_coordinates_ap_dv_ml_um_float64": physical[None]
            },
        },
        "subject_coordinate_map_id": "b" * 64,
        "subject_slab_render_id": "c" * 64,
        "receipt_sha256": "d" * 64,
    }
    section_plan = {
        "resolved_config": {"deformation_mode": "standard"},
        "provenance": {"animal_id": case["animal_id"]},
        "section_processing_plan_id": "e" * 64,
        "section_processing_realization_id": "f" * 64,
        "receipt_sha256": "0" * 64,
    }
    section_render = {
        "section_processing_render_id": "1" * 64,
        "receipt_sha256": "2" * 64,
    }
    observation = {
        "provenance": {"split_index": case["split_index"]},
        "modality": case["modality"],
        "observation_plan_id": "3" * 64,
        "observation_bundle_id": "4" * 64,
        "receipt_sha256": "5" * 64,
    }
    final = {
        "provenance": {"realization_index": case["realization_index"]},
        "mode_selection": {"selected_mode": case["selected_trainable_input_mode"]},
        "synthetic_realization_id": "6" * 64,
        "training_row_id": "7" * 64,
        "receipt_sha256": "8" * 64,
    }
    pose = {
        "finite_plane_pose_truth_id": "9" * 64,
        "receipt_sha256": "a" * 64,
    }
    bank = {
        "rng_contract": {"candidate_root_seed_uint64": case["root_seeds_uint64"]["candidate_bank"]},
        "schedule": {"total": 40},
        "candidate_bank_id": "b" * 64,
        "receipt_sha256": "c" * 64,
    }
    source_lineage = {
        "support_resolution_plan_id": resolution["support_resolution_plan_id"],
        "support_resolution_receipt_sha256": resolution["receipt_sha256"],
        **{
            name: case[name]
            for name in (
                "split",
                "split_index",
                "animal_id",
                "animal_index",
                "specimen_id",
                "experiment_id",
                "section_index",
                "plane_stratum",
            )
        },
    }
    oracle = {
        "provenance": {"source_lineage": source_lineage},
        "scope": {"model_free": True},
        "scorer_input_contract": {"allowed_inputs": ["target_labels"]},
        "coverage": {"evaluable": True},
        "semantic_oracle_result_id": "d" * 64,
        "receipt_sha256": "e" * 64,
    }
    null = {
        "provenance": oracle["provenance"],
        "scope": {
            "model_training_or_benchmark_claim": False,
            "posterior_or_probability_claim": False,
        },
        "rng_contract": {
            "null_root_seed_uint64": case["root_seeds_uint64"]["semantic_null"]
        },
        "semantic_null_result_id": "f" * 64,
        "receipt_sha256": "0" * 64,
    }

    def resolve(*args, **kwargs):
        calls.append(("support", kwargs))
        return support_bundle

    def make_subject(*args, **kwargs):
        calls.append(("subject", kwargs))
        return subject_slab

    def make_section_plan(*args, **kwargs):
        calls.append(("section-plan", kwargs))
        return section_plan

    def make_section_render(*args, **kwargs):
        calls.append(("section-render", kwargs))
        return section_render

    def make_observation(*args, **kwargs):
        calls.append(("observation", kwargs))
        return observation

    def make_final(*args, **kwargs):
        calls.append(("final", kwargs))
        return final

    def make_pose(*args, **kwargs):
        calls.append(("pose", kwargs))
        return pose

    def make_bank(*args, **kwargs):
        calls.append(("candidate-bank", kwargs))
        return bank

    def make_oracle(*args, **kwargs):
        calls.append(("oracle", kwargs))
        return oracle

    def make_null(*args, **kwargs):
        calls.append(("null", kwargs))
        return null

    monkeypatch.setattr(
        panel_v2.support_resolution_v2,
        "_resolve_subject_support_with_mapper_v2",
        resolve,
    )
    monkeypatch.setattr(
        panel_v2.subject_slab_v2,
        "_make_subject_slab_render_with_mapper_v2",
        make_subject,
    )
    monkeypatch.setattr(
        panel_v2.section_processing_v2,
        "_orthogonal_section_pixel_metric",
        lambda values: ((10.0, 10.0), None, None),
    )
    monkeypatch.setattr(
        panel_v2.section_processing_v2,
        "sample_section_processing_plan_v2",
        make_section_plan,
    )
    monkeypatch.setattr(
        panel_v2.section_processing_v2,
        "_make_section_processing_render_with_mapper_v2",
        make_section_render,
    )
    monkeypatch.setattr(
        panel_v2.observation_v2,
        "_make_arbitrary_plane_observation_with_mapper_v2",
        make_observation,
    )
    monkeypatch.setattr(
        panel_v2.realization_v2,
        "_make_arbitrary_plane_realization_with_mapper_v2",
        make_final,
    )
    monkeypatch.setattr(panel_v2.pose_v2, "make_arbitrary_plane_pose_truth_v2", make_pose)
    monkeypatch.setattr(
        panel_v2.candidate_bank_v2, "make_arbitrary_plane_candidate_bank_v2", make_bank
    )
    monkeypatch.setattr(
        panel_v2.semantic_oracle_v2,
        "make_arbitrary_plane_semantic_oracle_result_v2",
        make_oracle,
    )
    monkeypatch.setattr(
        panel_v2.semantic_oracle_v2,
        "verify_arbitrary_plane_semantic_oracle_result_v2",
        lambda *args, **kwargs: calls.append(("oracle-verify", kwargs)),
    )
    monkeypatch.setattr(
        panel_v2.null_gate_v2,
        "make_arbitrary_plane_semantic_null_result_v2",
        make_null,
    )
    monkeypatch.setattr(
        panel_v2.null_gate_v2,
        "verify_arbitrary_plane_semantic_null_result_v2",
        lambda *args, **kwargs: calls.append(("null-verify", kwargs)),
    )

    mapper = object()
    record = panel_v2._evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2(
        context,
        panel,
        case,
        subject_plan,
        subject_to_ccf_mapper=mapper,
    )

    assert [name for name, _ in calls] == [
        "support",
        "subject",
        "section-plan",
        "section-render",
        "observation",
        "final",
        "pose",
        "candidate-bank",
        "oracle",
        "oracle-verify",
        "null",
        "null-verify",
    ]
    support_kwargs = calls[0][1]
    assert support_kwargs["animal_id"] == case["animal_id"]
    assert support_kwargs["specimen_id"] == case["specimen_id"]
    assert support_kwargs["experiment_id"] == case["experiment_id"]
    assert support_kwargs["plane_stratum"] == case["plane_stratum"]
    assert support_kwargs["subject_to_ccf_mapper"] is mapper
    assert calls[1][1]["subject_to_ccf_mapper"] is mapper
    assert calls[3][1]["subject_to_ccf_mapper"] is mapper
    assert calls[4][1]["subject_to_ccf_mapper"] is mapper
    assert calls[5][1]["subject_to_ccf_mapper"] is mapper
    assert calls[7][1]["candidate_root_seed"] == case["root_seeds_uint64"][
        "candidate_bank"
    ]
    assert calls[10][1]["null_root_seed"] == case["root_seeds_uint64"][
        "semantic_null"
    ]
    assert record["status"] == "complete"
    assert record["selected_trainable_input_mode"] == case[
        "selected_trainable_input_mode"
    ]
    assert record["raw_mode_trainable"] is False
    assert record["semantic_null_result"] == null


def test_all_failure_adapter_is_exact_and_gate_summary_is_failure_adverse():
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    records = [
        panel_v2.make_arbitrary_plane_semantic_oracle_failure_record_v2(
            panel, case, RuntimeError(f"failure-{case['case_index']}")
        )
        for case in panel["cases"]
    ]
    planned, gate_records = panel_v2.adapt_development_panel_records_to_semantic_gate_v2(
        panel, records
    )

    assert len(planned) == len(gate_records) == 24
    assert all(
        set(case) == {"case_id", "plane_stratum", "animal_index", "animal_id"}
        for case in planned
    )
    assert all(
        set(record)
        == {"case_id", "status", "primary_result", "null_result", "failure"}
        and record["status"] == "execution_failure"
        and record["primary_result"] is None
        and record["null_result"] is None
        for record in gate_records
    )
    summary = null_gate_v2.make_arbitrary_plane_semantic_gate_summary_v2(
        planned, gate_records, panel_id=panel["panel_receipt_sha256"]
    )
    null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
        summary,
        planned,
        gate_records,
        expected_panel_id=panel["panel_receipt_sha256"],
    )
    assert summary["passed"] is False
    assert summary["gates"]["execution_receipt_control_completeness"] is False
    assert summary["gates"]["shape_preserving_null"] is False
    assert summary["metrics"]["overall"]["execution_complete_rate"] == 0.0
    assert summary["metrics"]["overall"][
        "null_original_truth_mean_reciprocal_rank"
    ] == 1.0

    changed = acquisition._json_value(summary)
    changed["passed"] = True
    with pytest.raises(ValueError, match="structure or receipt"):
        null_gate_v2.verify_arbitrary_plane_semantic_gate_summary_v2(
            changed,
            planned,
            gate_records,
            expected_panel_id=panel["panel_receipt_sha256"],
        )


def test_case_milestone_reports_completed_evaluable_success(capsys):
    case = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()["cases"][0]
    runner_v2._emit_case_milestone(
        "arbitrary-plane-semantic-oracle-panel-v2-case-verified",
        case,
        24,
        "newly-evaluated",
        record={
            "status": "complete",
            "later_gate_policy": {"evaluable": True},
        },
    )

    event = json.loads(capsys.readouterr().out)
    assert event == {
        "event": "arbitrary-plane-semantic-oracle-panel-v2-case-verified",
        "case_index": 0,
        "case_number": 1,
        "case_count": 24,
        "case_id": case["case_id"],
        "plane_stratum": case["plane_stratum"],
        "execution_mode": "newly-evaluated",
        "verification_status": "verified",
        "record_status": "complete",
        "scientific_outcome": "complete-evaluable",
        "caught_exception_type": None,
        "caught_exception_status": "none",
        "recorded_failure_exception_type": None,
    }


def test_atomic_runner_records_failures_and_resumes_without_reexecution(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner_v2.acquisition, "_validate_v2_context", lambda value: None)
    plan_calls = []
    case_calls = []
    subject_mappers = {}

    def subject_plan(context, animal):
        plan_calls.append(animal["animal_index"])
        subject_mappers[animal["animal_index"]] = object()
        return (
            {"animal_index": animal["animal_index"]},
            subject_mappers[animal["animal_index"]],
        )

    def fail_case(context, panel, case, plan, **kwargs):
        case_calls.append(case["case_index"])
        assert kwargs["subject_to_ccf_mapper"] is subject_mappers[
            case["animal_index"]
        ]
        raise RuntimeError(f"scheduled-{case['case_index']}")

    monkeypatch.setattr(
        runner_v2,
        "_make_development_panel_subject_plan_with_mapper_v2",
        subject_plan,
    )
    monkeypatch.setattr(
        runner_v2,
        "_evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2",
        fail_case,
    )
    output = tmp_path / "panel"
    result = runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
        {}, output
    )
    first_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]

    assert result["scheduled_case_count"] == 24
    assert result["recorded_case_count"] == 24
    assert result["failed_case_count"] == 24
    assert result["all_scheduled_cases_recorded"] is True
    assert result["failure_adverse_gate_summary_computed"] is True
    assert result["semantic_gate_summary_passed"] is False
    assert result["all_cases_live_verified_when_gate_frozen"] is True
    assert result["live_qualification_evidence"]["live_verification_mode_counts"] == {
        "newly-evaluated": 24,
        "strict-replayed": 0,
    }
    assert plan_calls == [2101, 2102, 2103, 2104]
    assert case_calls == list(range(24))
    assert len(list((output / "cases").glob("case-*.json"))) == 24
    assert first_events[0] == {
        "event": "arbitrary-plane-semantic-oracle-panel-v2-started",
        "output_folder": str(output.resolve()),
        "panel_receipt_sha256": result["panel_receipt_sha256"],
        "case_count": 24,
        "batch_size": None,
        "strict_replay_existing": False,
        "frozen_summary_present": False,
    }
    first_started = [
        event
        for event in first_events
        if event["event"]
        == "arbitrary-plane-semantic-oracle-panel-v2-case-started"
    ]
    first_verified = [
        event
        for event in first_events
        if event["event"]
        == "arbitrary-plane-semantic-oracle-panel-v2-case-verified"
    ]
    assert len(first_started) == len(first_verified) == 24
    assert [event["case_index"] for event in first_started] == list(range(24))
    assert all(event["case_count"] == 24 for event in first_started)
    assert all(event["execution_mode"] == "newly-evaluated" for event in first_started)
    assert all(
        event["record_status"] == "failed"
        and event["scientific_outcome"] == "failure-adverse-pipeline-failure"
        and event["caught_exception_type"] == "RuntimeError"
        and event["caught_exception_status"]
        == "converted-to-failure-adverse-record"
        and event["recorded_failure_exception_type"] == "RuntimeError"
        and event["verification_status"] == "verified"
        for event in first_verified
    )
    assert [event["plane_stratum"] for event in first_started] == [
        case["plane_stratum"]
        for case in panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()[
            "cases"
        ]
    ]
    gate_summary_bytes = (output / "gate-summary.json").read_bytes()
    qualification_bytes = (output / "live-qualification.json").read_bytes()
    result_bytes = (output / "result.json").read_bytes()
    assert not list(output.rglob("*.tmp"))

    monkeypatch.setattr(
        runner_v2,
        "_make_development_panel_subject_plan_with_mapper_v2",
        lambda *args, **kwargs: pytest.fail("resume remade an animal plan"),
    )
    monkeypatch.setattr(
        runner_v2,
        "_evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2",
        lambda *args, **kwargs: pytest.fail("resume re-executed a frozen case"),
    )
    replay = runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
        {}, output
    )
    existing_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert replay == result
    assert (output / "gate-summary.json").read_bytes() == gate_summary_bytes
    assert (output / "live-qualification.json").read_bytes() == qualification_bytes
    assert (output / "result.json").read_bytes() == result_bytes
    existing_verified = [
        event
        for event in existing_events
        if event["event"]
        == "arbitrary-plane-semantic-oracle-panel-v2-case-verified"
    ]
    assert len(existing_verified) == 24
    assert all(
        event["execution_mode"] == "verified-existing"
        and event["caught_exception_type"] is None
        and event["caught_exception_status"] == "none"
        and event["recorded_failure_exception_type"] == "RuntimeError"
        for event in existing_verified
    )

    strict_calls = []
    monkeypatch.setattr(
        runner_v2,
        "_make_development_panel_subject_plan_with_mapper_v2",
        subject_plan,
    )

    def replay_failure(context, panel, case, plan, **kwargs):
        strict_calls.append(case["case_index"])
        assert kwargs["subject_to_ccf_mapper"] is subject_mappers[
            case["animal_index"]
        ]
        raise RuntimeError(f"scheduled-{case['case_index']}")

    monkeypatch.setattr(
        runner_v2,
        "_evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2",
        replay_failure,
    )
    strict = runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
        {}, output, strict_replay_existing=True
    )
    strict_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert strict == result
    assert strict_calls == list(range(24))
    assert (output / "gate-summary.json").read_bytes() == gate_summary_bytes
    strict_verified = [
        event
        for event in strict_events
        if event["event"]
        == "arbitrary-plane-semantic-oracle-panel-v2-case-verified"
    ]
    assert len(strict_verified) == 24
    assert all(
        event["execution_mode"] == "strict-replayed"
        and event["caught_exception_type"] == "RuntimeError"
        and event["caught_exception_status"]
        == "converted-to-failure-adverse-record"
        for event in strict_verified
    )

    evidence = deepcopy(result["live_qualification_evidence"])
    evidence["case_evidence"][0]["case_receipt_sha256"] = "0" * 64
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    records = [
        runner_v2._load_json(output / "cases" / f"case-{index:03d}.json")
        for index in range(24)
    ]
    with pytest.raises(ValueError, match="qualification evidence changed"):
        runner_v2.verify_semantic_oracle_panel_live_qualification_v2(
            evidence, panel, records
        )

    changed_result = deepcopy(result)
    changed_result["all_cases_live_verified_when_gate_frozen"] = False
    (output / "result.json").write_bytes(runner_v2._canonical_bytes(changed_result))
    with pytest.raises(ValueError, match="run summary changed"):
        runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
            {}, output
        )


def test_partial_default_resume_refuses_gate_then_strict_replay_qualifies(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner_v2.acquisition, "_validate_v2_context", lambda value: None)
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    output = tmp_path / "partial"
    runner_v2._atomic_immutable_json(output / "panel.json", panel)
    first = panel_v2.make_arbitrary_plane_semantic_oracle_failure_record_v2(
        panel, panel["cases"][0], RuntimeError("scheduled-0")
    )
    runner_v2._atomic_immutable_json(output / "cases" / "case-000.json", first)
    calls = []

    monkeypatch.setattr(
        runner_v2,
        "_make_development_panel_subject_plan_with_mapper_v2",
        lambda context, animal: (
            {"animal_index": animal["animal_index"]},
            object(),
        ),
    )

    def fail_case(context, panel, case, plan, **kwargs):
        calls.append(case["case_index"])
        raise RuntimeError(f"scheduled-{case['case_index']}")

    monkeypatch.setattr(
        runner_v2,
        "_evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2",
        fail_case,
    )
    with pytest.raises(RuntimeError, match="rerun with strict_replay_existing=True"):
        runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
            {}, output
        )
    assert calls == list(range(1, 24))
    assert len(list((output / "cases").glob("case-*.json"))) == 24
    assert not (output / "live-qualification.json").exists()
    assert not (output / "gate-summary.json").exists()
    assert not (output / "result.json").exists()

    calls.clear()
    result = runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2(
        {}, output, strict_replay_existing=True
    )
    assert calls == list(range(24))
    assert result["semantic_gate_summary_passed"] is False
    assert result["live_qualification_evidence"]["live_verification_mode_counts"] == {
        "newly-evaluated": 0,
        "strict-replayed": 24,
    }
    assert (output / "live-qualification.json").exists()
    assert (output / "gate-summary.json").exists()
    assert (output / "result.json").exists()


def test_atomic_writer_is_idempotent_and_refuses_changed_frozen_output(tmp_path):
    path = tmp_path / "record.json"
    first = runner_v2._atomic_immutable_json(path, {"case": 1})
    second = runner_v2._atomic_immutable_json(path, {"case": np.int64(1)})
    assert first == second
    with pytest.raises(FileExistsError, match="refusing to replace"):
        runner_v2._atomic_immutable_json(path, {"case": 2})


def test_pinned_allen_loader_binds_hashes_decoder_and_provenance(
    tmp_path, monkeypatch
):
    template = np.zeros((2, 3, 4), dtype=np.float32)
    annotation = np.ones((2, 3, 4), dtype=np.uint32)
    calls = []
    support = object()
    context = {"prepared": True}

    monkeypatch.setattr(runner_v2.nrrd, "__version__", "1.1.3")
    monkeypatch.setattr(
        runner_v2,
        "_file_sha256",
        lambda path: runner_v2._TEMPLATE_SOURCE_SHA256
        if path.name == "average_template_25.nrrd"
        else runner_v2._ANNOTATION_SOURCE_SHA256,
    )

    def read(path, **kwargs):
        calls.append(("read", path, kwargs))
        return (template if path.endswith("average_template_25.nrrd") else annotation, {})

    def build_support(values, **kwargs):
        calls.append(("support", values, kwargs))
        return support

    def prepare(values, annotation_values, support_index, **kwargs):
        calls.append(("prepare", values, annotation_values, support_index, kwargs))
        return context

    monkeypatch.setattr(runner_v2.nrrd, "read", read)
    monkeypatch.setattr(runner_v2, "build_annotation_support_index", build_support)
    monkeypatch.setattr(
        runner_v2.acquisition,
        "prepare_arbitrary_plane_acquisition_context_v2",
        prepare,
    )

    assert runner_v2.load_pinned_allen_context_v2(tmp_path) is context
    assert calls[0][2] == {"index_order": "F"}
    assert calls[1][2] == {"index_order": "F"}
    assert calls[2][2] == {
        "atlas_id": "Allen CCFv3",
        "atlas_version": "2017 25um",
        "source_uri": "data/Allen Brain Atlas 25um/annotation_25.nrrd",
        "source_sha256": runner_v2._ANNOTATION_SOURCE_SHA256,
        "source_entity_type": "atlas-annotation",
        "voxel_size_um": (25.0, 25.0, 25.0),
        "origin_um": (0.0, 0.0, 0.0),
        "coordinate_axis_directions": ("posterior", "inferior", "right"),
    }
    assert calls[3][2] is annotation
    assert calls[3][3] is support
    assert calls[3][4] == {
        "scalar_source_uri": "data/Allen Brain Atlas 25um/average_template_25.nrrd",
        "scalar_source_sha256": runner_v2._TEMPLATE_SOURCE_SHA256,
        "scalar_source_entity_type": "atlas-template",
        "template_decoder": "pynrrd 1.1.3",
        "template_index_order": "F",
        "annotation_decoder": "pynrrd 1.1.3",
        "annotation_index_order": "F",
    }


def test_independent_verifier_authenticates_explicit_pinned_allen_assets(
    tmp_path, monkeypatch
):
    atlas = (tmp_path / "allen").resolve()
    hashed = []
    monkeypatch.setattr(verifier_v2.nrrd, "__version__", "1.1.3")

    def pinned_hash(path):
        hashed.append(path)
        return (
            verifier_v2._TEMPLATE_SOURCE_SHA256
            if path.name == "average_template_25.nrrd"
            else verifier_v2._ANNOTATION_SOURCE_SHA256
        )

    monkeypatch.setattr(verifier_v2, "_file_sha256", pinned_hash)
    receipt = verifier_v2.authenticate_pinned_allen_assets_v2(atlas)

    assert [path.name for path in hashed] == [
        "average_template_25.nrrd",
        "annotation_25.nrrd",
    ]
    assert receipt == {
        "atlas_folder": str(atlas),
        "template_file": "average_template_25.nrrd",
        "template_source_sha256": verifier_v2._TEMPLATE_SOURCE_SHA256,
        "annotation_file": "annotation_25.nrrd",
        "annotation_source_sha256": verifier_v2._ANNOTATION_SOURCE_SHA256,
        "decoder": "pynrrd",
        "decoder_version": "1.1.3",
        "index_order": "F",
    }

    monkeypatch.setattr(verifier_v2, "_file_sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="average-template source hash mismatch"):
        verifier_v2.authenticate_pinned_allen_assets_v2(atlas)
    monkeypatch.setattr(verifier_v2.nrrd, "__version__", "1.1.2")
    with pytest.raises(RuntimeError, match="does not match pinned version 1.1.3"):
        verifier_v2.authenticate_pinned_allen_assets_v2(atlas)
    with pytest.raises(ValueError, match="explicit absolute path"):
        verifier_v2.authenticate_pinned_allen_assets_v2("relative-atlas")


def test_independent_verifier_requires_one_complete_record_context_id():
    context_id = "a" * 64
    record = {
        "status": "complete",
        "semantic_oracle_result": {
            "upstream_reference": {"v2_context_sha256": context_id}
        },
        "semantic_null_result": {
            "upstream_reference": {"v2_context_sha256": context_id}
        },
    }
    assert verifier_v2._complete_record_v2_context_sha256(
        [record, deepcopy(record)]
    ) == context_id
    assert verifier_v2._complete_record_v2_context_sha256(
        [{"status": "execution_failure"}]
    ) is None

    primary_null_mismatch = deepcopy(record)
    primary_null_mismatch["semantic_null_result"]["upstream_reference"][
        "v2_context_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="primary/null records"):
        verifier_v2._complete_record_v2_context_sha256([primary_null_mismatch])

    second_context = deepcopy(record)
    second_context["semantic_oracle_result"]["upstream_reference"][
        "v2_context_sha256"
    ] = "b" * 64
    second_context["semantic_null_result"]["upstream_reference"][
        "v2_context_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="multiple v2 context IDs"):
        verifier_v2._complete_record_v2_context_sha256([record, second_context])


def test_environment_runner_requires_external_output_and_forwards_strict_replay(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "semantic-panel"
    atlas = tmp_path / "allen"
    calls = []
    context = {"prepared": True}
    result = {
        "panel_receipt_sha256": "a" * 64,
        "scheduled_case_count": 24,
        "completed_case_count": 0,
        "failed_case_count": 24,
        "unevaluable_or_failed_case_count": 24,
        "semantic_gate_summary_passed": False,
        "all_cases_live_verified_when_gate_frozen": True,
        "live_qualification_evidence": {
            "live_verification_mode_counts": {
                "newly-evaluated": 0,
                "strict-replayed": 24,
            }
        },
    }
    monkeypatch.setenv(runner_v2._RUN_ENV, "1")
    monkeypatch.setenv(runner_v2._OUTPUT_ENV, str(output.resolve()))
    monkeypatch.setenv("ANATOMY_TRACKER_ATLAS_FOLDER", str(atlas))
    monkeypatch.setenv(runner_v2._STRICT_REPLAY_ENV, "1")
    monkeypatch.setenv(runner_v2._BATCH_SIZE_ENV, "7")
    monkeypatch.setattr(
        runner_v2,
        "load_pinned_allen_context_v2",
        lambda path: calls.append(("load", path)) or context,
    )
    monkeypatch.setattr(
        runner_v2,
        "run_arbitrary_plane_semantic_oracle_development_panel_v2",
        lambda prepared, folder, **kwargs: calls.append(
            ("run", prepared, folder, kwargs)
        )
        or result,
    )

    runner_v2.main()

    assert calls == [
        ("load", atlas),
        (
            "run",
            context,
            output.resolve(),
            {"batch_size": 7, "strict_replay_existing": True},
        ),
    ]
    report = json.loads(capsys.readouterr().out)
    assert report["event"] == "arbitrary-plane-semantic-oracle-panel-v2-frozen"
    assert report["panel_receipt_sha256"] == "a" * 64
    assert report["scheduled_case_count"] == 24
    assert report["failed_case_count"] == 24
    assert report["all_cases_live_verified_when_gate_frozen"] is True
    assert report["live_verification_mode_counts"] == {
        "newly-evaluated": 0,
        "strict-replayed": 24,
    }
    assert report["strict_replay_existing"] is True

    monkeypatch.delenv(runner_v2._OUTPUT_ENV)
    with pytest.raises(ValueError, match="explicit absolute output"):
        runner_v2.main()

    monkeypatch.setenv(
        runner_v2._OUTPUT_ENV,
        str((runner_v2._ROOT / "build" / "forbidden-panel-output").resolve()),
    )
    with pytest.raises(ValueError, match="outside the repository"):
        runner_v2.main()


def test_independent_verifier_checks_exact_frozen_tree_and_provenance(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner_v2.acquisition, "_validate_v2_context", lambda value: None)
    monkeypatch.setattr(
        runner_v2,
        "_make_development_panel_subject_plan_with_mapper_v2",
        lambda context, animal: (
            {"animal_index": animal["animal_index"]},
            object(),
        ),
    )

    def fail_case(context, panel, case, plan, **kwargs):
        raise RuntimeError(f"scheduled-{case['case_index']}")

    monkeypatch.setattr(
        runner_v2,
        "_evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2",
        fail_case,
    )
    output = tmp_path / "frozen-panel"
    runner_v2.run_arbitrary_plane_semantic_oracle_development_panel_v2({}, output)
    capsys.readouterr()

    report = verifier_v2.verify_frozen_arbitrary_plane_semantic_oracle_panel_v2(
        output
    )
    panel = panel_v2.arbitrary_plane_semantic_oracle_development_panel_v2()
    assert report["verified_case_count"] == 24
    assert report["complete_record_v2_context_sha256"] is None
    assert report["animal_ids"] == [animal["animal_id"] for animal in panel["animals"]]
    assert report["specimen_ids"] == [
        animal["specimen_id"] for animal in panel["animals"]
    ]
    assert report["experiment_ids"] == sorted(
        {case["experiment_id"] for case in panel["cases"]}
    )

    atlas = (tmp_path / "allen").resolve()
    atlas_authentication = {
        "atlas_folder": str(atlas),
        "template_source_sha256": verifier_v2._TEMPLATE_SOURCE_SHA256,
        "annotation_source_sha256": verifier_v2._ANNOTATION_SOURCE_SHA256,
        "decoder_version": "1.1.3",
    }
    monkeypatch.setenv(verifier_v2._ATLAS_ENV, str(atlas))
    monkeypatch.setenv(verifier_v2._OUTPUT_ENV, str(output.resolve()))
    monkeypatch.setattr(
        verifier_v2,
        "authenticate_pinned_allen_assets_v2",
        lambda path: atlas_authentication,
    )
    verifier_v2.main()
    main_report = json.loads(capsys.readouterr().out)
    assert main_report["verified_case_count"] == 24
    assert main_report["allen_asset_authentication"] == atlas_authentication

    monkeypatch.delenv(verifier_v2._ATLAS_ENV)
    with pytest.raises(ValueError, match="explicit absolute atlas directory"):
        verifier_v2.main()

    case_path = output / "cases" / "case-000.json"
    case_path.write_bytes(case_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        verifier_v2.verify_frozen_arbitrary_plane_semantic_oracle_panel_v2(output)
