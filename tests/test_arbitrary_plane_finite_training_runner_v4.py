import copy
import hashlib

import numpy as np
import pytest

import arbitrary_plane_production_v3_fixtures as fixture
import test_arbitrary_plane_row_cache_v4 as cache_fixture
import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_deformation_gauge_v4 as gauge_v4
import training.arbitrary_plane_finite_training_runner_v4 as runner_v4
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4


def _catalogue():
    return catalogue_v3.make_arbitrary_plane_catalogue_v3(
        np.ones((10, 10, 10), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=2,
        offset_count=3,
        roll_count=1,
        raster_shape_h_w=(6, 6),
        raster_physical_span_y_x_um=(6.0, 6.0),
    )


def _binding(row_count, render_mode="finite_boxcar"):
    algorithm = "finite-runner-v4-test-generator/v4"
    source = {"finite_runner_fixture.py": hashlib.sha256(b"fixture").hexdigest()}
    config = {
        "schema_version": "anatomy-tracker.finite-runner-v4-test-generator/v4",
        "algorithm": algorithm,
        "row_count": row_count,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    return cache_v4.make_generator_binding_v4(
        generator_ids=[algorithm],
        source_sha256=source,
        geometry_gauge_contract=gauge_v4.direct_deformation_target_contract_v4(),
        generation_config=config,
        seed_record={"root_seed_uint64": "0x0000000000004321"},
        generation_lineage={
            "schema_version": cache_v4.GENERATION_LINEAGE_V4_SCHEMA,
            "generation_run_id": "finite-runner-v4-test",
            "source_commit": "4" * 40,
            "split": "development-finite-runner-v4",
        },
        finite_psf_run_contract=(
            cache_v4.make_finite_psf_cache_run_contract_v4(render_mode)
        ),
    )


def _row(index, binding, thickness=25.0, render_mode="finite_boxcar", zero=False):
    row = cache_fixture._row(index, thickness, render_mode)
    row["lineage"]["split"] = binding["generation_lineage"]["split"]
    row["upstream_reference"]["schema_version"] = binding[
        "generation_config"
    ]["schema_version"]
    row["upstream_reference"]["algorithm"] = binding["generator_ids"][0]
    row["upstream_reference"]["implementation_source_sha256"] = copy.deepcopy(
        binding["source_sha256"]
    )
    row["deformation_pose_gauge_reference"] = {
        **gauge_v4.direct_deformation_target_contract_v4(),
        "direct_deformation_target_id": acquisition_v2._payload_sha256(
            {"runner-gauge": index}
        ),
        "receipt_sha256": acquisition_v2._payload_sha256(
            {"runner-gauge-receipt": index}
        ),
    }
    row["upstream_reference"]["support_supervision_contract"] = {
        "point_pose_supervision_weight": 0.0 if zero else 1.0,
        "dense_deformation_supervision_weight": 0.0 if zero else 1.0,
    }
    if zero:
        row["arrays"]["target_correspondence_weight_float32"].fill(0.0)
        row["array_receipts"] = {
            name: acquisition_v2._array_receipt(value)
            for name, value in row["arrays"].items()
        }
        row["training_row_id"] = acquisition_v2._payload_sha256(
            {
                "domain": psf_v4.TRAINING_ROW_V4_SCHEMA,
                "synthetic_realization_id": row["synthetic_realization_id"],
                "array_receipts": row["array_receipts"],
                "finite_psf_sha256": row["finite_psf_contract"][
                    "finite_psf_sha256"
                ],
                "slab_observation_v4_receipt_sha256": row[
                    "finite_psf_contract"
                ]["slab_observation_v4_receipt_sha256"],
            }
        )
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    return row


def _prepared(tmp_path, *, row_count=2, target=1, zero=False, render_mode="finite_boxcar"):
    binding = _binding(row_count, render_mode)
    cache = tmp_path / "cache"
    cache_v4.initialize_training_row_cache_v4(cache, generator_binding=binding)
    thickness = 0.0 if render_mode == "centre_plane_ablation" else 25.0
    rows = [
        _row(index, binding, thickness + (0.0 if render_mode != "finite_boxcar" else index), render_mode, zero)
        for index in range(row_count)
    ]
    cache_v4.append_training_rows_v4(cache, rows)
    frozen = cache_v4.freeze_training_row_cache_v4(cache)
    source = tmp_path / "allen-source.bin"
    source.write_bytes(b"authenticated Allen finite runner fixture")
    config = fixture.runner_config(target)
    config["axial_offsets_um"] = ()
    config["axial_weights"] = ()
    manifest, state = runner_v4.initialize_finite_training_run_v4(
        tmp_path / "run",
        cache_directory=cache,
        expected_generator_binding=binding,
        catalogue=_catalogue(),
        atlas_volume=fixture.atlas(),
        atlas_source_assets=(
            {
                "path": str(source),
                "role": "Allen fixture",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ),
        atlas_preprocessing={"normalization": "deterministic fixture"},
        model_kwargs=fixture.model_kwargs(),
        training_config=fixture.training_config(),
        runner_config=config,
        device="cpu",
    )
    return tmp_path / "run", binding, frozen, manifest, state


def test_finite_runner_trains_resumes_and_exports_exact_s9_contract(tmp_path):
    run, binding, frozen, manifest, initial = _prepared(tmp_path)
    assert manifest["schema_version"] == runner_v4.FINITE_TRAINING_RUN_V4_SCHEMA
    assert manifest["cache"]["manifest_receipt_sha256"] == frozen["receipt_sha256"]
    assert manifest["finite_psf_training_schedule_source"][
        "finite_psf_cache_run_contract"
    ]["axial_sample_count"] == 9
    assert manifest["runner_config"]["axial_offsets_um"] == []
    assert manifest["staged_training_binding"]["generator_ids"] == binding[
        "generator_ids"
    ]
    assert initial["applied_step_count"] == 0

    reports = runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    assert len(reports) == 1
    assert reports[0]["global_step_after"] == 1
    assert reports[0]["supervision_weight_summary"][
        "zero_weight_rows_retained"
    ]
    loaded = runner_v4.load_finite_training_run_v4(run)
    assert loaded["training_state"]["global_step"] == 1
    receipt = runner_v4.make_finite_training_run_export_receipt_v4(run)
    assert receipt["finite_psf_cache_run_contract"]["axial_sample_count"] == 9
    assert receipt["checkpoint"][
        "finite_psf_cache_run_contract_receipt_sha256"
    ] == frozen["finite_psf_run_contract"]["receipt_sha256"]
    staged_receipt = receipt["staged_training_export_receipt"]
    assert runner_v4.verify_finite_staged_training_export_receipt_v4(
        staged_receipt,
        model_kwargs=manifest["model_kwargs"],
        catalogue_id=loaded["catalogue"]["catalogue_id"],
        catalogue_receipt_sha256=loaded["catalogue"]["receipt_sha256"],
        catalogue_cell_count=loaded["catalogue"]["counts"]["cell_count"],
        model_state_sha256=staged_receipt["model_state_sha256"],
        require_source_file=True,
    )
    changed = copy.deepcopy(staged_receipt)
    changed["training_report_ledger_evidence"]["applied_step_count"] = 0
    with pytest.raises(ValueError, match="invalid"):
        runner_v4.verify_finite_staged_training_export_receipt_v4(
            changed,
            model_kwargs=manifest["model_kwargs"],
            catalogue_id=loaded["catalogue"]["catalogue_id"],
            catalogue_receipt_sha256=loaded["catalogue"]["receipt_sha256"],
            catalogue_cell_count=loaded["catalogue"]["counts"]["cell_count"],
            model_state_sha256=staged_receipt["model_state_sha256"],
        )
    assert runner_v4.verify_finite_training_run_export_receipt_v4(receipt, run)


def test_zero_weight_marginal_row_is_sampled_and_contributes_exact_zero_loss(tmp_path):
    run, _, _, _, _ = _prepared(tmp_path, row_count=1, zero=True)
    report = runner_v4.run_finite_training_attempts_v4(
        run, max_attempts=1
    )[0]
    summary = report["supervision_weight_summary"]
    assert summary["batch_row_count"] == 1
    assert summary["pose_positive_row_count"] == 0
    assert summary["dense_positive_row_count"] == 0
    assert summary["pose_weight_sum"] == 0.0
    assert summary["dense_weight_sum"] == 0.0
    assert report["training_report"]["objective"] == 0.0
    assert report["training_report"]["optimizer_step_applied"]


def test_centre_plane_s1_is_a_separate_authenticated_run(tmp_path):
    run, _, frozen, manifest, _ = _prepared(
        tmp_path,
        row_count=1,
        render_mode="centre_plane_ablation",
    )
    contract = manifest["finite_psf_training_schedule_source"][
        "finite_psf_cache_run_contract"
    ]
    assert contract["render_mode"] == "centre_plane_ablation"
    assert contract["axial_sample_count"] == 1
    assert contract["receipt_sha256"] == frozen["finite_psf_run_contract"][
        "receipt_sha256"
    ]
    assert runner_v4.load_finite_training_run_v4(run)["manifest"] == manifest


def test_atomic_fault_before_run_state_replays_only_uncommitted_attempt(tmp_path, monkeypatch):
    run, _, _, _, _ = _prepared(tmp_path, row_count=1)

    def fail(point):
        if point == "after_reports":
            raise RuntimeError("simulated finite v4 commit interruption")

    monkeypatch.setattr(runner_v4, "_commit_fault_injection_point", fail)
    with pytest.raises(RuntimeError, match="simulated finite v4"):
        runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    recovered = runner_v4.load_finite_training_run_v4(run)
    assert recovered["run_state"]["attempt_count"] == 0
    assert recovered["training_state"]["global_step"] == 0

    monkeypatch.setattr(
        runner_v4, "_commit_fault_injection_point", lambda point: None
    )
    report = runner_v4.run_finite_training_attempts_v4(
        run, max_attempts=1
    )[0]
    assert report["attempt_index"] == 0
    assert report["global_step_after"] == 1


def test_three_bounded_commits_preserve_history_across_resume_slot_reuse(tmp_path):
    run, _, _, _, _ = _prepared(tmp_path, row_count=2, target=3)

    for expected_step in (1, 2, 3):
        report = runner_v4.run_finite_training_attempts_v4(
            run, max_attempts=1
        )[0]
        assert report["global_step_after"] == expected_step

    loaded = runner_v4.load_finite_training_run_v4(run)
    assert loaded["run_state"]["attempt_count"] == 3
    assert loaded["run_state"]["applied_step_count"] == 3
    assert len(loaded["training_reports"]) == 3


def test_rehashed_report_cannot_change_authenticated_per_row_psf_identity(tmp_path):
    run, _, _, _, _ = _prepared(tmp_path, row_count=2)
    runner_v4.run_finite_training_attempts_v4(run, max_attempts=1)
    report_path = run / "reports" / "attempt_00000000.json"
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    report["ordered_row_finite_psf_identity_sha256"] = "0" * 64
    report_payload = {
        key: value for key, value in report.items() if key != "receipt_sha256"
    }
    report["receipt_sha256"] = runner_v4._hash_json(report_payload)
    runner_v4.runner_primitives._atomic_json(report_path, report)

    state_path = run / "run_state.json"
    state = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    state["committed_reports"][0]["file_sha256"] = runner_v4._file_sha256(
        report_path
    )
    state["committed_reports"][0]["report_receipt_sha256"] = report[
        "receipt_sha256"
    ]
    state_payload = {
        key: value for key, value in state.items() if key != "receipt_sha256"
    }
    state["receipt_sha256"] = runner_v4._hash_json(state_payload)
    runner_v4.runner_primitives._atomic_json(state_path, state)

    with pytest.raises(ValueError, match="failed authentication"):
        runner_v4.load_finite_training_run_v4(run)


def test_runner_rejects_global_psf_and_rehashed_schedule_tampering(tmp_path):
    binding = _binding(1)
    cache = tmp_path / "cache"
    cache_v4.initialize_training_row_cache_v4(cache, generator_binding=binding)
    cache_v4.append_training_rows_v4(cache, [_row(0, binding)])
    cache_v4.freeze_training_row_cache_v4(cache)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    config = fixture.runner_config(1)
    with pytest.raises(ValueError, match="PSF"):
        runner_v4.initialize_finite_training_run_v4(
            tmp_path / "bad-run",
            cache_directory=cache,
            expected_generator_binding=binding,
            catalogue=_catalogue(),
            atlas_volume=fixture.atlas(),
            atlas_source_assets=(
                {
                    "path": str(source),
                    "role": "fixture",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            ),
            atlas_preprocessing={"normalization": "fixture"},
            model_kwargs=fixture.model_kwargs(),
            training_config=fixture.training_config(),
            runner_config=config,
            device="cpu",
        )

    run, _, _, _, _ = _prepared(tmp_path / "clean")
    manifest_path = run / "run_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finite_psf_training_schedule_source"][
        "finite_psf_cache_run_contract"
    ]["axial_sample_count"] = 1
    payload = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
    identity = dict(payload)
    identity.pop("run_id")
    payload["run_id"] = runner_v4._hash_json(
        {
            "domain": runner_v4.FINITE_TRAINING_RUN_V4_SCHEMA,
            "scientific_core": runner_v4._scientific_run_id_payload(identity),
        }
    )
    manifest["run_id"] = payload["run_id"]
    manifest["receipt_sha256"] = runner_v4._hash_json(payload)
    runner_v4.runner_primitives._atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="binding differs"):
        runner_v4.load_finite_training_run_v4(run)
