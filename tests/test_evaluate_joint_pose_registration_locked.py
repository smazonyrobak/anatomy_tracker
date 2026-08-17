from __future__ import annotations

import json
import inspect
from pathlib import Path

import nrrd
import numpy as np
import pytest
import torch

from source.atlas_pose_runtime import POSE_IMAGE_SIZE
from training import joint_pose_registration_locked_data as locked
from training import evaluate_joint_pose_registration_locked as evaluator
from training import joint_pose_registration_release as release
from training import train_joint_pose_registration as trainer


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    folder = tmp_path_factory.mktemp("joint-evaluator-ccf")
    ap, dv, ml = 420, 18, 22
    ap_axis = np.arange(ap, dtype=np.float32)[:, None, None]
    dv_axis = np.arange(dv, dtype=np.float32)[None, :, None]
    ml_axis = np.arange(ml, dtype=np.float32)[None, None, :]
    average = ap_axis * 0.2 + dv_axis * 2.0 + ml_axis
    labels = np.broadcast_to(
        (np.arange(ap, dtype=np.int64) + 1)[:, None, None], (ap, dv, ml)
    ).copy()
    nrrd.write(str(folder / "average_template_25.nrrd"), average.astype(np.float32))
    nrrd.write(str(folder / "annotation_25.nrrd"), labels.astype(np.int64))
    return locked.LockedJointSyntheticBenchmark(folder, "cpu")


def _receipt(pose):
    return {
        "map_space": evaluator.MAP_SPACE,
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        "map_pose": torch.as_tensor(pose).clone(),
        "source_shape": locked.MODEL_SHAPE,
        "composition": "candidate-aligned maps composed to source model canvas",
    }


def _map_record(pose, forward, inverse):
    return {
        "map_pose": torch.as_tensor(pose).clone(),
        "map_space": evaluator.MAP_SPACE,
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        "map_domain_receipt": _receipt(pose),
        "fixed_to_source_model": forward.clone(),
        "source_model_to_fixed": inverse.clone(),
    }


def _exact_prediction(batch, item):
    pose = batch["pose"][item].clone()
    forward = batch["fixed_to_moving"][item].clone()
    inverse = batch["moving_to_fixed"][item].clone()
    iteration = _map_record(pose, forward, inverse)
    return {
        "status": "success",
        "initial_pose": pose.clone(),
        "recurrent_poses": pose[None].clone(),
        "iteration_predictions": [iteration, _map_record(pose, forward, inverse)],
        "final_pose": pose.clone(),
        **_map_record(pose, forward, inverse),
        "provider": "CPUExecutionProvider",
        "wall_time_seconds": 0.25 + 0.05 * item,
        "peak_memory_bytes": 1024 * (item + 1),
    }


def _exact_predictor(batch, *, overrides=None, exact_override=None):
    overrides = overrides or {}
    pose_images = None

    def predictor(payload):
        nonlocal pose_images
        if payload["task"] == "exact_plane_registration":
            height, width = payload["moving_raw_uint8"].shape[-2:]
            identity = locked._identity_grid(
                1, height, width, payload["moving_raw_uint8"].device
            )
            output = {
                "fixed_to_source_model": identity,
                "source_model_to_fixed": identity.clone(),
                "map_domain_receipt": _receipt(payload["candidate_pose"]),
            }
            return exact_override(output) if exact_override is not None else output
        if payload["task"] == "compatibility":
            matches = [
                torch.equal(payload["pose_image"], image[None])
                for image in pose_images
            ]
            item = matches.index(True)
            exact = torch.isclose(
                payload["candidate_pose"][0], batch["pose"][item],
                rtol=0.0, atol=1e-5,
            ).all()
            return torch.tensor(3.0 if bool(exact) else 0.0)
        pose_images = payload["pose_image"].clone()
        predictions = [
            _exact_prediction(batch, item)
            for item in range(len(batch["pose"]))
        ]
        for item, callback in overrides.items():
            predictions[item] = callback(predictions[item])
        return predictions

    return predictor


def test_public_evaluator_reports_complete_exact_metrics_and_atomic_outputs(
    benchmark, tmp_path
):
    batch = benchmark.generate(2, "locked-validation", 4811, "clean", 3)
    result = evaluator.evaluate_public_split(
        benchmark, batch, _exact_predictor(batch), tmp_path / "evaluation"
    )

    assert result["attempted_count"] == result["success_count"] == 2
    assert result["failure_count"] == result["nonfinite_output_count"] == 0
    assert result["ap_mae_um"] == result["lr_mae_deg"] == result["dv_mae_deg"] == 0.0
    assert result["plane_anchor_tre_mean_um"] == 0.0
    assert result["five_anchor_plane_distance_mean_um"] == 0.0
    assert "physical_corresponding_plane_distance_mean_um" not in result
    assert result["end_to_end_visible_region_correspondence"] > 0.98
    assert result["end_to_end_interior_region_correspondence"] > 0.98
    assert result["end_to_end_macro_region_dice"] > 0.95
    assert result["end_to_end_bottom_30_region_dice"] > 0.90
    assert result["end_to_end_boundary_f1_2px"] > 0.99
    assert result["end_to_end_forward_negative_jacobian_fraction"] == 0.0
    assert result["end_to_end_inverse_negative_jacobian_fraction"] == 0.0
    assert result["true_plane_rank_1_rate"] == result["true_plane_mrr"] == 1.0
    assert result["pose_monotonic_case_rate"] == 1.0
    assert result["correspondence_monotonic_case_rate"] == 1.0
    assert result["warp_only_receipt_count"] == 2
    assert result["warp_only_forward_endpoint_mean_px"] >= 0.0
    assert result["warp_only_inverse_endpoint_mean_px"] >= 0.0
    assert 0.0 <= result["warp_only_visible_region_correspondence"] <= 1.0
    assert 0.0 <= result["warp_only_macro_region_dice"] <= 1.0
    assert 0.0 <= result["warp_only_bottom_30_region_dice"] <= 1.0
    assert 0.0 <= result["warp_only_boundary_f1_2px"] <= 1.0
    assert result["warp_only_forward_negative_jacobian_fraction"] == 0.0
    assert result["attested_runtime"]["peak_memory_bytes_maximum"] == 2048
    assert result["measured_inference_runtime"]["wall_time_seconds"] >= 0.0

    output = tmp_path / "evaluation"
    assert (output / "raw_predictions.pt").is_file()
    assert (output / "aggregate_metrics.json").is_file()
    assert (output / "evaluation_receipt.json").is_file()
    saved = json.loads((output / "aggregate_metrics.json").read_text())
    assert saved["attempted_count"] == 2
    assert saved["raw_predictions_sha256"] == result["raw_predictions_sha256"]
    assert not list(output.glob(".*.tmp"))


def test_failures_and_nonfinite_predictions_remain_in_every_denominator(
    benchmark, tmp_path
):
    batch = benchmark.generate(3, "development", 5912, "clean", 2)
    failed = {
        "status": "failed",
        "failure_reason": "runtime error",
        "provider": "DirectML",
        "wall_time_seconds": 1.5,
        "peak_memory_bytes": 4096,
    }
    def fail(_prediction):
        return failed

    def nonfinite(prediction):
        prediction["fixed_to_source_model"][0, 0, 0] = torch.nan
        return prediction

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        _exact_predictor(batch, overrides={1: fail, 2: nonfinite}),
        tmp_path / "failures",
        error_threshold_um=150.0,
    )

    assert result["attempted_count"] == 3
    assert result["success_count"] == 1
    assert result["failure_count"] == 2
    assert result["failure_rate"] == pytest.approx(2 / 3)
    assert result["nonfinite_output_count"] == 1
    assert result["ap_mae_um"] == pytest.approx(10000.0 / 3.0)
    assert result["ap_bias_um"] == 0.0
    assert 0.32 < result["end_to_end_visible_region_correspondence"] < 0.34
    assert result["true_plane_rank_1_rate"] == pytest.approx(1 / 3)
    assert result["true_plane_mrr"] == pytest.approx(1 / 3)
    assert result["warp_only_failure_count"] == 0
    assert result["warp_only_receipt_count"] == 3
    assert 0.0 <= result["risk_ranking"]["error_detection_auroc"] <= 1.0
    assert 0.0 <= result["risk_ranking"]["error_detection_auprc"] <= 1.0
    assert "false_safe_rate_among_errors" not in result["risk_ranking"]
    assert "calibrated" in result["risk_ranking"]["score_contract"]
    assert result["attested_runtime"]["providers"] == [
        "CPUExecutionProvider", "DirectML"
    ]


@pytest.mark.parametrize("stale_location", ("final", "iteration"))
def test_stale_or_mismatched_map_receipts_are_rejected(
    benchmark, tmp_path, stale_location
):
    batch = benchmark.generate(1, "locked-validation", 6123, "clean", 2)
    def stale(prediction):
        if stale_location == "final":
            prediction["map_domain_receipt"]["map_pose"][0] += 25.0
        else:
            prediction["iteration_predictions"][0]["map_pose"][0] += 25.0
        return prediction

    result = evaluator.evaluate_public_split(
        benchmark, batch, _exact_predictor(batch, overrides={0: stale}),
        tmp_path / stale_location
    )
    assert result["failure_count"] == 1


def test_final_maps_must_equal_the_last_recurrent_map_receipt(benchmark, tmp_path):
    batch = benchmark.generate(1, "locked-validation", 6323, "clean", 2)
    def different(prediction):
        prediction["fixed_to_source_model"] = prediction["fixed_to_source_model"].clone()
        prediction["fixed_to_source_model"][0, 0, 0] += 1.0
        return prediction

    result = evaluator.evaluate_public_split(
        benchmark, batch, _exact_predictor(batch, overrides={0: different}),
        tmp_path / "different-final-map"
    )
    assert result["failure_count"] == 1


def test_warp_only_metrics_require_a_complete_exact_plane_receipt(benchmark, tmp_path):
    batch = benchmark.generate(1, "development", 6523, "clean", 2)
    def missing(record):
        del record["map_domain_receipt"]
        return record

    result = evaluator.evaluate_public_split(
        benchmark, batch, _exact_predictor(batch, exact_override=missing),
        tmp_path / "partial-exact"
    )
    assert result["failure_count"] == 0
    assert result["warp_only_failure_count"] == 1
    assert result["any_mode_failure_count"] == 1
    assert result["warp_only_forward_endpoint_mean_px"] == pytest.approx(
        evaluator.FAILURE_PIXEL_DISTANCE
    )


def test_end_to_end_failure_does_not_erase_independent_exact_plane_track(
    benchmark, tmp_path
):
    batch = benchmark.generate(1, "development", 6533, "clean", 2)

    def fail(_prediction):
        return {
            "status": "failed",
            "failure_reason": "initializer failed",
            "provider": "CPUExecutionProvider",
            "wall_time_seconds": 0.0,
            "peak_memory_bytes": 0,
        }

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        _exact_predictor(batch, overrides={0: fail}),
        tmp_path / "pose-failed-warp-valid",
    )
    assert result["failure_count"] == 1
    assert result["warp_only_failure_count"] == 0
    assert result["warp_only_receipt_count"] == 1


def test_end_to_end_exception_does_not_prevent_exact_plane_track(
    benchmark, tmp_path
):
    batch = benchmark.generate(1, "development", 6534, "clean", 2)
    base = _exact_predictor(batch)

    def predictor(payload):
        if payload["task"] == "end_to_end":
            raise RuntimeError("initializer process failed")
        return base(payload)

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        predictor,
        tmp_path / "pose-exception-warp-valid",
    )
    assert result["failure_count"] == 1
    assert result["warp_only_failure_count"] == 0
    assert result["warp_only_receipt_count"] == 1


@pytest.mark.parametrize("malformed_mode", ("end_to_end", "exact_plane_registration"))
def test_malformed_mode_output_is_counted_without_erasing_the_other_track(
    benchmark, tmp_path, malformed_mode
):
    batch = benchmark.generate(1, "development", 6538, "clean", 2)
    base = _exact_predictor(batch)

    def predictor(payload):
        if payload["task"] == malformed_mode:
            return None
        return base(payload)

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        predictor,
        tmp_path / f"malformed-{malformed_mode}",
    )
    if malformed_mode == "end_to_end":
        assert result["failure_count"] == 1
        assert result["warp_only_failure_count"] == 0
    else:
        assert result["failure_count"] == 0
        assert result["warp_only_failure_count"] == 1


def test_exact_plane_truth_uses_distinct_late_predictor_phase(benchmark):
    batch = benchmark.generate(2, "development", 6543, "clean", 2)
    payload, expected = evaluator.build_predictor_payload(
        benchmark, batch, shuffle_secret=b"phase-isolation"
    )
    base = _exact_predictor(batch)
    events = []

    def truth_free(payload):
        events.append(("truth-free", payload["task"]))
        assert payload["task"] != "exact_plane_registration"
        return base(payload)

    def exact_only(payload):
        events.append(("exact", payload["task"]))
        assert payload["task"] == "exact_plane_registration"
        height, width = payload["moving_raw_uint8"].shape[-2:]
        identity = locked._identity_grid(1, height, width, torch.device("cpu"))
        return {
            "fixed_to_source_model": identity,
            "source_model_to_fixed": identity.clone(),
            "map_domain_receipt": _receipt(payload["candidate_pose"]),
        }

    predictions, _ = evaluator._run_bound_predictor(
        benchmark,
        truth_free,
        payload,
        expected,
        exact_predictor=exact_only,
    )
    first_exact = next(index for index, event in enumerate(events) if event[0] == "exact")
    assert all(event[0] == "truth-free" for event in events[:first_exact])
    assert all(event[0] == "exact" for event in events[first_exact:])
    assert sum(event == ("exact", "exact_plane_registration") for event in events) == 2
    assert all("exact_plane_fixed_to_source_model" in value for value in predictions)


def test_frozen_receipt_loads_hashed_factory_and_locked_entry_is_capability_gated(
    benchmark, tmp_path
):
    checkpoint = tmp_path / "best-validation.pt"
    trainer.save_checkpoint(
        checkpoint,
        {
            "format_version": trainer.FORMAT_VERSION,
            "ema": {"shadow": {"weight": torch.ones(1)}},
            "release_selection": {
                "state": "ema.shadow",
                "criterion": "validation_selection_score",
                "validation_score": -1.25,
                "completed_views": 120,
            },
            "completed_views": 120,
            "best_validation_score": -1.25,
            "latest_validation": {"selection_score": -1.25},
            "generator_contract": {"contract": "test"},
        },
    )
    _, release_state_receipt = release.load_joint_release_state(checkpoint)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"release_state_receipt": release_state_receipt})
    )
    predictor_source = tmp_path / "predictor.py"
    predictor_source.write_text(
        "import torch\n"
        "def create_frozen_predictor(bundle):\n"
        "    assert any(f['role'] == 'checkpoint' for f in bundle['files'])\n"
        "    assert bundle['release_state_receipt']['selected_state'] == 'ema.shadow'\n"
        "    def predict(payload):\n"
        "        if payload['task'] == 'exact_plane_registration':\n"
        "            h, w = payload['moving_raw_uint8'].shape[-2:]\n"
        "            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')\n"
        "            identity = torch.stack((x, y), 0).float()[None]\n"
        "            pose = payload['candidate_pose'].clone()\n"
        "            receipt = {'map_space': 'source-model-canvas', "
        "'refiner_preprocessing': payload['refiner_preprocessing'], "
        "'map_pose': pose, 'source_shape': (h, w)}\n"
        "            return {'fixed_to_source_model': identity, "
        "'source_model_to_fixed': identity.clone(), 'map_domain_receipt': receipt}\n"
        "        if payload['task'] == 'compatibility':\n"
        "            return 0.0\n"
        "        return []\n"
        "    return predict\n"
    )
    files = [
        {"role": "checkpoint", "path": str(checkpoint), "sha256": evaluator._file_sha256(checkpoint)},
        {"role": "metadata", "path": str(metadata), "sha256": evaluator._file_sha256(metadata)},
        {"role": "predictor_source", "path": str(predictor_source), "sha256": evaluator._file_sha256(predictor_source)},
    ]
    bundle = {
        "files": files,
        "preprocessing_contract": locked.PREPROCESSING_CONTRACT_V2,
        "mask_contract_sha256": locked.MASK_CONTRACT_SHA256,
        "pose_preprocessing_contract_sha256": evaluator.atlas_pose_preprocessing_contract_sha256(),
        "risk_score_contract_sha256": evaluator.RISK_SCORE_CONTRACT_SHA256,
        "evaluator_dependency_tree_sha256": evaluator.evaluator_dependency_tree_sha256(),
        "recurrence_count": 4,
        "configuration": {"model": "joint"},
        "provider_policy": ["CPUExecutionProvider"],
        "release_state_receipt": release_state_receipt,
    }
    receipt = {
        "frozen": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": evaluator._file_sha256(checkpoint),
        "release_state_receipt": release_state_receipt,
        "evaluator_source_sha256": evaluator.evaluator_source_sha256(),
        "benchmark_contract_sha256": benchmark.contract["contract_sha256"],
        "generator_source_sha256": locked._source_sha256(),
        "inference_bundle": bundle,
        "inference_bundle_sha256": locked._payload_sha256(bundle),
    }
    evaluator.validate_frozen_locked_receipt(benchmark, receipt)
    loaded = evaluator._load_frozen_predictor(receipt)
    exact_loaded = evaluator._load_frozen_predictor(receipt)
    assert loaded is not exact_loaded
    assert loaded({"task": "end_to_end"}) == []
    assert loaded({"task": "compatibility"}) == 0.0
    exact = loaded(
        {
            "task": "exact_plane_registration",
            "moving_raw_uint8": torch.zeros(1, 1, 5, 7, dtype=torch.uint8),
            "candidate_pose": torch.zeros(1, 3),
            "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        }
    )
    assert set(exact) == {
        "fixed_to_source_model", "source_model_to_fixed", "map_domain_receipt"
    }
    assert exact["fixed_to_source_model"].shape == (1, 2, 5, 7)
    changed = {**receipt, "evaluator_source_sha256": "0" * 64}
    with pytest.raises(ValueError, match="evaluator source"):
        evaluator.validate_frozen_locked_receipt(benchmark, changed)
    with pytest.raises(PermissionError, match="private capability"):
        evaluator.run_locally_locked_once(
            benchmark,
            tmp_path / "sealed-not-run",
            receipt,
            _capability=object(),
        )
    assert "predictor" not in inspect.signature(evaluator.run_locally_locked_once).parameters
    assert not (tmp_path / "sealed-not-run" / "LOCALLY_LOCKED_CLAIM.json").exists()

    without_source_files = [
        value for value in files if value["role"] != "predictor_source"
    ]
    without_source_bundle = {**bundle, "files": without_source_files}
    without_source = {
        **receipt,
        "inference_bundle": without_source_bundle,
        "inference_bundle_sha256": locked._payload_sha256(without_source_bundle),
    }
    with pytest.raises(ValueError, match="predictor source adapter"):
        evaluator.validate_frozen_locked_receipt(benchmark, without_source)

    for name in ("latest.pt", "checkpoint-000120.pt"):
        noncanonical = tmp_path / name
        noncanonical.write_bytes(checkpoint.read_bytes())
        with pytest.raises(ValueError, match="best-validation.pt"):
            evaluator.validate_frozen_locked_receipt(
                benchmark,
                {
                    **receipt,
                    "checkpoint_path": str(noncanonical),
                    "checkpoint_sha256": evaluator._file_sha256(noncanonical),
                },
            )

    for folder_name, selection, latest_score in (
        ("raw-model", "model", -1.25),
        ("non-best", "ema.shadow", -2.0),
    ):
        invalid_path = tmp_path / folder_name / "best-validation.pt"
        invalid_path.parent.mkdir()
        trainer.save_checkpoint(
            invalid_path,
            {
                "format_version": trainer.FORMAT_VERSION,
                "ema": {"shadow": {"weight": torch.ones(1)}},
                "release_selection": {
                    "state": selection,
                    "criterion": "validation_selection_score",
                    "validation_score": -1.25,
                    "completed_views": 120,
                },
                "completed_views": 120,
                "best_validation_score": -1.25,
                "latest_validation": {"selection_score": latest_score},
                "generator_contract": {"contract": "test"},
            },
        )
        with pytest.raises(
            ValueError,
            match="canonical EMA|best validation state|best-validation release",
        ):
            evaluator.validate_frozen_locked_receipt(
                benchmark,
                {
                    **receipt,
                    "checkpoint_path": str(invalid_path),
                    "checkpoint_sha256": evaluator._file_sha256(invalid_path),
                },
            )


def test_predictor_payload_contains_no_truth_maps_labels_or_pose(benchmark, tmp_path):
    batch = benchmark.generate(1, "development", 7023, "clean", 3)
    forbidden = {
        "pose", "fixed", "fixed_mask", "fixed_labels", "moving_labels",
        "fixed_to_moving", "moving_to_fixed", "moving_visible_mask",
        "negative_pose", "negative_pose_offset",
        "candidate_poses", "candidate_fixed", "candidate_fixed_mask",
        "candidate_set_sha256", "case_sha256", "pair_id", "view", "severity",
        "moving_damage_mask", "moving_optical_artifact_mask",
    }
    exact = _exact_predictor(batch)
    seen = {"end_to_end": 0, "compatibility": 0, "exact_plane_registration": 0}

    def predictor(payload):
        seen[payload["task"]] += 1
        if payload["task"] == "end_to_end":
            assert forbidden.isdisjoint(payload)
        elif payload["task"] == "compatibility":
            assert "candidate_poses" not in payload
            assert "candidate_set_sha256" not in payload
            assert "case_sha256" not in payload
            assert payload["candidate_pose"].shape == (1, 3)
            assert "pose" not in payload
            assert "fixed_labels" not in payload
        else:
            assert payload["task"] == "exact_plane_registration"
            assert payload["candidate_pose"].shape == (1, 3)
            assert payload["candidate_fixed"].shape[0] == 1
            assert payload["candidate_fixed_mask"].shape[0] == 1
            assert {
                "pose", "fixed_labels", "moving_labels", "fixed_to_moving",
                "moving_to_fixed", "candidate_set_sha256", "case_sha256",
            }.isdisjoint(payload)
        return exact(payload)

    result = evaluator.evaluate_public_split(
        benchmark, batch, predictor, tmp_path / "sanitized"
    )
    assert result["success_count"] == 1
    assert seen == {
        "end_to_end": 1,
        "compatibility": 11,
        "exact_plane_registration": 1,
    }


def test_pose_initializer_payload_is_exact_runtime_preprocessing(benchmark):
    batch = benchmark.generate(2, "development", 7073, "moderate", 3)
    payload, _ = evaluator.build_predictor_payload(benchmark, batch)
    expected = np.stack(
        [
            evaluator.preprocess_atlas_pose_image(
                image[0].numpy(), mask[0].numpy()
            )
            for image, mask in zip(
                batch["pose_view_raw_uint8"], batch["pose_view_mask"]
            )
        ]
    )
    assert payload["pose_view_raw_uint8"].dtype == torch.uint8
    assert payload["pose_image"].shape == (
        2, 3, POSE_IMAGE_SIZE, POSE_IMAGE_SIZE
    )
    np.testing.assert_array_equal(payload["pose_image"].numpy(), expected)
    assert (
        payload["pose_preprocessing_contract_sha256"]
        == evaluator.atlas_pose_preprocessing_contract_sha256()
    )


def test_reference_and_challenged_payloads_share_geometry_but_not_appearance(
    benchmark,
):
    batch = benchmark.generate(2, "development", 7083, "severe", 3)
    secret = b"reference-challenge-pair"
    challenged, challenged_expected = evaluator.build_predictor_payload(
        benchmark, batch, shuffle_secret=secret, view="challenged"
    )
    reference, reference_expected = evaluator.build_predictor_payload(
        benchmark, batch, shuffle_secret=secret, view="reference"
    )
    assert "view" not in challenged and "view" not in reference
    assert not torch.equal(
        challenged["moving_raw_uint8"], reference["moving_raw_uint8"]
    )
    assert torch.equal(
        reference["moving_raw_uint8"], batch["reference_moving_raw_uint8"]
    )
    assert "pair_id" not in challenged and "pair_id" not in reference
    for first, second in zip(challenged_expected, reference_expected):
        assert first["candidate_set_sha256"] == second["candidate_set_sha256"]
        assert first["true_index"] == second["true_index"]
        assert torch.equal(first["candidate_poses"], second["candidate_poses"])


def test_paired_support_reports_damage_separately_from_common_metric_domain(
    benchmark,
):
    batch = benchmark.generate(1, "development", 7088, "clean", 3)
    tissue = batch["moving_tissue_mask"].clone()
    visible = tissue.clone()
    visible[:, :, :, : visible.shape[-1] // 2] = False
    batch["moving_visible_mask"] = visible
    record = evaluator._paired_support_records(batch)[0]
    expected_visible = float(visible.sum() / tissue.sum())
    assert record["reference_full_tissue_pixels"] == int(tissue.sum())
    assert record["challenged_visible_pixels"] == int(visible.sum())
    assert record["reference_full_tissue_support"] == 1.0
    assert record["challenged_visible_support"] == pytest.approx(expected_visible)
    assert record["coverage_loss"] == pytest.approx(1.0 - expected_visible)
    records = [record] * len(locked.SEVERITIES)
    summary = evaluator._paired_support_summary(records, list(locked.SEVERITIES))
    assert summary["support_denominator"] == "reference_full_tissue_pixels"
    assert summary["overall"]["reference_full_tissue_pixels"] == (
        int(tissue.sum()) * len(locked.SEVERITIES)
    )
    assert summary["overall"]["challenged_visible_pixels"] == (
        int(visible.sum()) * len(locked.SEVERITIES)
    )
    assert summary["overall"]["aggregate_coverage_loss"] == pytest.approx(
        1.0 - expected_visible
    )


def test_paired_degradation_declares_common_challenged_visible_domain():
    reference = [{"status": "failed", "risk_score": 1e9}] * 4
    challenged = [{"status": "failed", "risk_score": 1e9}] * 4
    result = evaluator._paired_degradation(
        reference, challenged, list(locked.SEVERITIES)
    )
    assert result["paired_metric_domain"] == (
        "challenged_visible_common_support"
    )


def test_private_candidate_lattice_does_not_expose_shuffle_or_make_truth_the_hub(benchmark):
    batch = benchmark.generate(1, "development", 7093, "clean", 6)
    observed_true_indices = set()
    truth_is_distance_hub = []
    for value in range(16):
        payload, expected = evaluator.build_predictor_payload(
            benchmark, batch, shuffle_secret=value.to_bytes(4, "big")
        )
        assert "case_sha256" not in payload
        assert "candidate_set_sha256" not in payload
        assert "candidate_poses" not in payload
        candidates = expected[0]["candidate_poses"]
        true_index = expected[0]["true_index"]
        observed_true_indices.add(true_index)
        distances = torch.cdist(candidates, candidates).sum(1)
        truth_is_distance_hub.append(true_index == int(distances.argmin()))
    assert len(observed_true_indices) > 1
    assert not all(truth_is_distance_hub)


def test_risk_score_uses_center_margin_and_boundary_candidates_are_distinct():
    confident_center = torch.tensor((8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    confident_neighbor = torch.tensor((0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert evaluator._compatibility_risk_score(confident_center) < 0.01
    assert evaluator._compatibility_risk_score(confident_neighbor) > 0.99
    for pose in (
        (-4500.0, -35.0, -35.0),
        (500.0, 35.0, 35.0),
        (-2000.0, 0.0, 0.0),
    ):
        candidates = evaluator._risk_candidate_poses(torch.tensor(pose))
        assert candidates.shape == (7, 3)
        assert len(torch.unique(candidates, dim=0)) == 7
        assert candidates[:, 0].min() >= locked.AP_RANGE_UM[0]
        assert candidates[:, 0].max() <= locked.AP_RANGE_UM[1]
        assert candidates[:, 1:].abs().max() <= 35.0


@pytest.mark.parametrize(
    "attack", ("singleton", "reordered", "injected_logits", "injected_exact")
)
def test_evaluator_bound_candidates_cannot_be_changed_or_omitted(
    benchmark, tmp_path, attack
):
    batch = benchmark.generate(1, "development", 7123, "clean", 3)

    def corrupt(prediction):
        if attack == "singleton":
            prediction["candidate_poses"] = batch["pose"][:1]
        elif attack == "reordered":
            prediction["candidate_set_sha256"] = "0" * 64
        elif attack == "injected_logits":
            prediction["compatibility_logits"] = torch.tensor((100.0,))
        else:
            prediction["exact_plane_fixed_to_source_model"] = torch.zeros(
                1, 2, *locked.MODEL_SHAPE
            )
        return prediction

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        _exact_predictor(batch, overrides={0: corrupt}),
        tmp_path / attack,
    )
    assert result["success_count"] == 0
    assert result["failure_count"] == 1
    assert result["warp_only_failure_count"] == 0
    assert result["true_plane_rank_1_rate"] == 0.0


def test_derived_nonfinite_spatial_metrics_become_failure(benchmark, tmp_path):
    batch = benchmark.generate(1, "development", 7223, "clean", 3)

    def outside(prediction):
        prediction["fixed_to_source_model"] = torch.full_like(
            prediction["fixed_to_source_model"], -10000.0
        )
        prediction["iteration_predictions"][-1]["fixed_to_source_model"] = prediction[
            "fixed_to_source_model"
        ].clone()
        return prediction

    result = evaluator.evaluate_public_split(
        benchmark,
        batch,
        _exact_predictor(batch, overrides={0: outside}),
        tmp_path / "empty-spatial",
    )
    assert result["failure_count"] == 1
    assert result["warp_only_failure_count"] == 0
