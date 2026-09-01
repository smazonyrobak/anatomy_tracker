import json

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_semantic_oracle_null_gate_v2 as null_gate
import training.arbitrary_plane_semantic_oracle_v2 as primary_v2
from training.arbitrary_plane_semantic_oracle import rank_candidate_ids


def _control(name):
    payload = {"control": name, "passed": True, "evidence": {"control": name}}
    return {**payload, "evidence_receipt_sha256": acquisition._payload_sha256(payload)}


def _null_inputs(monkeypatch, homogeneous=False):
    shape = (8, 10)
    y, x = np.indices(shape)
    target = (
        np.full(shape, 17, dtype=np.int64)
        if homogeneous
        else np.where((x + 2 * y) % 3 == 0, 17, 31).astype(np.int64)
    )
    valid = np.ones(shape, dtype=bool)
    valid[0, :3] = False
    labels = np.zeros((40, *shape), dtype=np.int64)
    truth_index = 7
    labels[truth_index] = target
    candidate_ids = [f"candidate-{index:02d}" for index in range(40)]
    bank = {
        "candidate_bank_id": "1" * 64,
        "receipt_sha256": "2" * 64,
        "ordered_candidate_ids": candidate_ids,
        "candidates": [
            {
                "candidate_id": candidate_ids[index],
                "arrays": {"rendered_annotation_int64": labels[index]},
            }
            for index in range(40)
        ],
    }
    primary_score = primary_v2._score_semantic_arrays_v2(
        target, labels, valid, 10.0
    )
    primary = {
        "semantic_oracle_result_id": "3" * 64,
        "receipt_sha256": "4" * 64,
        "provenance": {
            "source_lineage": {
                "animal_id": "animal-five",
                "animal_index": 5,
                "specimen_id": "specimen-five",
                "experiment_id": "experiment-five",
                "plane_stratum": "general_oblique",
            },
            "observation_and_realization": {
                "split_index": 3,
                "animal_index": 5,
                "section_index": 41,
                "observation_index": 7,
                "realization_index": 11,
            },
        },
        "target_reference": {
            "labels_receipt": acquisition._array_receipt(target),
            "fixed_valid_mask_receipt": acquisition._array_receipt(valid),
            "pixel_pitch_reference": {"selected_pixel_pitch_um": 10.0},
            "channel_reference": {
                "large_region_ids": primary_score["target_large_region_ids"].tolist(),
                "small_pooled_region_ids": primary_score["target_small_region_ids"].tolist(),
                "channel_count": primary_score["channel_count"],
            },
        },
        "candidate_reference": {
            "ordered_candidate_ids": candidate_ids,
            "truth_candidate_id": candidate_ids[truth_index],
            "candidate_label_stack_receipt": acquisition._array_receipt(labels),
        },
        "scores": {"smoothing_sigma_px": primary_score["smoothing_sigma_px"]},
    }
    final = {
        "targets": {
            "source_label_ground_truth_crop_int64": target,
            "valid_correspondence_mask": valid,
        },
        "provenance": {
            "split_index": 3,
            "animal_index": 5,
            "section_index": 41,
            "observation_index": 7,
            "realization_index": 11,
            "animal_id": "animal-five",
        },
        "synthetic_realization_id": "5" * 64,
        "receipt_sha256": "6" * 64,
    }
    pose = {"finite_plane_pose_truth_id": "7" * 64}
    context = {"v2_context_sha256": "8" * 64}
    monkeypatch.setattr(
        null_gate.primary_v2,
        "verify_arbitrary_plane_semantic_oracle_result_v2",
        lambda *args: None,
    )
    return primary, bank, pose, final, context, target, valid, labels


def test_shape_preserving_null_replay_receipt_and_numeric_seed(monkeypatch):
    primary, bank, pose, final, context, target, valid, labels = _null_inputs(
        monkeypatch
    )
    result = null_gate.make_arbitrary_plane_semantic_null_result_v2(
        primary, bank, pose, final, context
    )
    replayed = null_gate.replay_arbitrary_plane_semantic_null_result_v2(
        result, primary, bank, pose, final, context
    )
    null_gate.verify_arbitrary_plane_semantic_null_result_v2(
        result, primary, bank, pose, final, context
    )
    json_result = json.loads(json.dumps(acquisition._json_value(result)))
    null_gate.verify_arbitrary_plane_semantic_null_result_v2(
        json_result, primary, bank, pose, final, context
    )

    assert result["receipt_sha256"] == replayed["receipt_sha256"]
    assert result["target_reference"]["shape_h_w"] == tuple(target.shape)
    assert acquisition._json_value(
        result["target_reference"]["fixed_valid_mask_receipt"]
    ) == acquisition._array_receipt(valid)
    assert acquisition._json_value(
        result["candidate_reference"]["candidate_label_stack_receipt"]
    ) == acquisition._array_receipt(labels)
    assert result["target_reference"]["label_histogram_preserved"] is True
    assert result["target_reference"]["channel_definition_preserved"] is True
    assert result["target_reference"]["scorer_denominator_preserved"] is True
    assert result["rng_contract"]["redraw_count"] == 0
    assert all(record["passed"] for record in result["exact_controls"].values())
    assert result["scope"]["weighted_correspondence_sensitivity_included"] is False
    assert result["scope"]["posterior_or_probability_claim"] is False

    seed = null_gate.derive_semantic_null_seed_v2(
        result["rng_contract"]["null_root_seed_uint64"], 3, 5, 41, 7, 11
    )
    assert result["rng_contract"]["seed_uint64"] == f"0x{seed:016x}"
    renamed = dict(final)
    renamed["provenance"] = dict(final["provenance"], animal_id="renamed-animal")
    renamed_result = null_gate.make_arbitrary_plane_semantic_null_result_v2(
        primary, bank, pose, renamed, context
    )
    assert renamed_result["rng_contract"]["seed_uint64"] == result["rng_contract"][
        "seed_uint64"
    ]
    assert renamed_result["target_reference"]["permutation_receipt"] == result[
        "target_reference"
    ]["permutation_receipt"]

    changed_target = np.roll(target, 1, axis=1)
    changed_labels = np.roll(labels, 1, axis=2)
    changed_final = dict(final)
    changed_final["targets"] = dict(
        final["targets"], source_label_ground_truth_crop_int64=changed_target
    )
    changed_bank = dict(bank)
    changed_bank["candidates"] = [
        {
            **candidate,
            "arrays": {"rendered_annotation_int64": changed_labels[index]},
        }
        for index, candidate in enumerate(bank["candidates"])
    ]
    changed_primary = dict(primary)
    changed_primary["target_reference"] = dict(
        primary["target_reference"],
        labels_receipt=acquisition._array_receipt(changed_target),
    )
    changed_primary["candidate_reference"] = dict(
        primary["candidate_reference"],
        candidate_label_stack_receipt=acquisition._array_receipt(changed_labels),
    )
    content_changed_result = null_gate.make_arbitrary_plane_semantic_null_result_v2(
        changed_primary, changed_bank, pose, changed_final, context
    )
    assert content_changed_result["rng_contract"]["seed_uint64"] == result[
        "rng_contract"
    ]["seed_uint64"]
    assert content_changed_result["target_reference"]["permutation_receipt"] == result[
        "target_reference"
    ]["permutation_receipt"]


def test_degenerate_null_is_retained_without_redraw(monkeypatch):
    primary, bank, pose, final, context, *_ = _null_inputs(
        monkeypatch, homogeneous=True
    )
    result = null_gate.make_arbitrary_plane_semantic_null_result_v2(
        primary, bank, pose, final, context
    )
    assert result["target_reference"]["degenerate"] is True
    assert result["target_reference"]["changed_valid_pixel_count"] == 0
    assert result["target_reference"]["degenerate_policy"] == (
        "record and retain; never redraw"
    )
    assert result["rng_contract"]["redraw_count"] == 0


def test_null_raw_score_tamper_is_rejected(monkeypatch):
    primary, bank, pose, final, context, *_ = _null_inputs(monkeypatch)
    result = null_gate.make_arbitrary_plane_semantic_null_result_v2(
        primary, bank, pose, final, context
    )
    changed = dict(result)
    changed_scores = dict(result["scores"])
    changed_arrays = dict(result["scores"]["arrays"])
    semantic = np.array(changed_arrays["semantic_score_float64"], copy=True)
    semantic[0] = 0.75
    changed_arrays["semantic_score_float64"] = semantic
    changed_scores["arrays"] = changed_arrays
    changed["scores"] = changed_scores
    with pytest.raises(ValueError, match="structure, receipt, control, or ranking"):
        null_gate.verify_arbitrary_plane_semantic_null_result_v2(
            changed, primary, bank, pose, final, context
        )


def _sha(label):
    return acquisition._payload_sha256({"fixture": label})


PANEL_ID = _sha("semantic-panel")
WRONG_PANEL_ID = _sha("wrong-semantic-panel")


def _primary_ranking(scores, candidate_ids, evaluable):
    ranking = rank_candidate_ids(scores, candidate_ids, candidate_ids[0])
    ranking["top3"] = bool(ranking["true_rank"] <= 3)
    ranking["raw_top1_before_coverage_policy"] = bool(ranking["top1"])
    ranking["raw_top3_before_coverage_policy"] = bool(ranking["top3"])
    ranking["top1"] = bool(evaluable and ranking["raw_top1_before_coverage_policy"])
    ranking["top3"] = bool(evaluable and ranking["raw_top3_before_coverage_policy"])
    ranking["coverage_adjusted_top1"] = ranking["top1"]
    ranking["coverage_adjusted_top3"] = ranking["top3"]
    return ranking


def _reseal_primary(result):
    result["semantic_oracle_result_id"] = acquisition._payload_sha256(
        primary_v2._identity_payload(result)
    )
    result["receipt_sha256"] = acquisition._payload_sha256(
        primary_v2.arbitrary_plane_semantic_oracle_result_receipt_v2(result)
    )
    return acquisition._freeze_value(result)


def _reseal_null(result):
    result["semantic_null_result_id"] = acquisition._payload_sha256(
        null_gate._null_identity(result)
    )
    result["receipt_sha256"] = acquisition._payload_sha256(
        null_gate.arbitrary_plane_semantic_null_result_receipt_v2(result)
    )
    return acquisition._freeze_value(result)


def _primary_summary_result(plan, case_index, evaluable=True):
    shape = (2, 3)
    target = np.array([[17, 31, 17], [31, 17, 31]], dtype=np.int64)
    valid = np.ones(shape, dtype=bool)
    candidate_stack = np.zeros((40, *shape), dtype=np.int64)
    candidate_ids = [f"candidate-{index:02d}" for index in range(40)]
    scores = np.zeros(40, dtype=np.float64)
    scores[0] = 1.0
    score_arrays = {
        "semantic_score_float64": scores,
        "raw_id_agreement_float64": scores.copy(),
        "mask_dice_float64": scores.copy(),
    }
    large_ids = np.empty(0, dtype=np.int64)
    small_ids = np.array([17, 31], dtype=np.int64)
    channel_reference = {
        "minimum_individual_region_pixels": primary_v2.MINIMUM_INDIVIDUAL_REGION_PIXELS,
        "large_region_ids": large_ids.tolist(),
        "small_pooled_region_ids": small_ids.tolist(),
        "large_region_ids_receipt": acquisition._array_receipt(large_ids),
        "small_pooled_region_ids_receipt": acquisition._array_receipt(small_ids),
        "channel_count": 1,
    }
    pixel_pitch_reference = {
        "pixel_pitch_y_x_um": [10.0, 10.0],
        "selected_pixel_pitch_um": 10.0,
        "isotropy_relative_tolerance": primary_v2.PITCH_ISOTROPY_RTOL,
        "crop_window_id": _sha(f"crop-{case_index}"),
        "observation_bundle_receipt_sha256": _sha(f"observation-{case_index}"),
        "source": "authenticated observation crop processed_pixel_pitch_y_x_um",
        "model_ouv_pitch_used": False,
    }
    candidate_bank_id = _sha(f"bank-{case_index}")
    candidate_bank_receipt = _sha(f"bank-receipt-{case_index}")
    realization_id = _sha(f"realization-{case_index}")
    realization_receipt = _sha(f"realization-receipt-{case_index}")
    pose_id = _sha(f"pose-{case_index}")
    context_id = _sha(f"context-{case_index}")
    brain_pixels = 6 if evaluable else 0
    provenance = {
        "source_lineage": {
            "support_resolution_plan_id": _sha(f"plan-{case_index}"),
            "support_resolution_receipt_sha256": _sha(f"plan-receipt-{case_index}"),
            "split": "development",
            "split_index": 1,
            "animal_id": plan["animal_id"],
            "animal_index": plan["animal_index"],
            "specimen_id": f"specimen-{plan['animal_index']}",
            "experiment_id": f"experiment-{plan['animal_index']}",
            "section_index": case_index,
            "plane_stratum": plan["plane_stratum"],
        },
        "observation_and_realization": {
            "split": "development",
            "split_index": 1,
            "animal_id": plan["animal_id"],
            "animal_index": plan["animal_index"],
            "section_index": case_index,
            "observation_index": case_index,
            "realization_index": 0,
        },
    }
    candidate_summaries = [
        {
            "candidate_id": candidate_id,
            "candidate_class": "truth" if index == 0 else "global_decoy",
            "slot": index,
            "brain_pixel_count": brain_pixels if index == 0 else 1,
            "finite_raster_support": {"fixture": True},
            "plane_intersects_support_envelope": True,
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    labels_receipt = acquisition._array_receipt(target)
    valid_receipt = acquisition._array_receipt(valid)
    candidate_receipt = acquisition._array_receipt(candidate_stack)
    result = {
        "schema_version": primary_v2.SEMANTIC_ORACLE_RESULT_V2_SCHEMA,
        "algorithm": primary_v2.SEMANTIC_ORACLE_RESULT_V2_ALGORITHM,
        "implementation_source_sha256": primary_v2._source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "asset_dependencies": acquisition._json_value(
            null_gate._NO_LEARNED_ASSET_DEPENDENCIES
        ),
        "scope": acquisition._json_value(null_gate._PRIMARY_SCOPE_CONTRACT),
        "upstream_reference": {
            "candidate_bank_id": candidate_bank_id,
            "candidate_bank_receipt_sha256": candidate_bank_receipt,
            "finite_plane_pose_truth_id": pose_id,
            "finite_plane_pose_truth_receipt_sha256": _sha(f"pose-receipt-{case_index}"),
            "synthetic_realization_id": realization_id,
            "synthetic_realization_receipt_sha256": realization_receipt,
            "training_row_id": _sha(f"row-{case_index}"),
            "frame_transform_id": _sha(f"frame-{case_index}"),
            "v2_context_sha256": context_id,
            "prepared_context_receipt_sha256": _sha(f"context-receipt-{case_index}"),
            "support_index_sha256": _sha("support"),
            "annotation_array_sha256": _sha("annotation"),
        },
        "provenance": provenance,
        "scorer_input_contract": {
            "allowed_inputs": list(primary_v2.NUMERICAL_SCORER_INPUTS),
            "forbidden_inputs": list(primary_v2.FORBIDDEN_NUMERICAL_SCORER_INPUTS),
            "target_labels_receipt": labels_receipt,
            "fixed_valid_mask_receipt": valid_receipt,
            "candidate_label_stack_receipt": candidate_receipt,
            "pixel_pitch_reference": pixel_pitch_reference,
            "candidate_ids_and_metadata_joined_after_scoring_only": True,
        },
        "target_reference": {
            "labels_receipt": labels_receipt,
            "fixed_valid_mask_receipt": valid_receipt,
            "shape_h_w": list(shape),
            "fixed_valid_pixel_count": 6,
            "fixed_valid_nonzero_region_id_count": 2,
            "channel_reference": channel_reference,
            "pixel_pitch_reference": pixel_pitch_reference,
        },
        "candidate_reference": {
            "candidate_count": 40,
            "ordered_candidate_ids": candidate_ids,
            "truth_candidate_id": candidate_ids[0],
            "truth_candidate_index": 0,
            "candidate_label_stack_receipt": candidate_receipt,
            "model_grid_reference": {"output_shape_h_w": list(shape)},
            "candidate_summaries": candidate_summaries,
        },
        "scores": {
            "arrays": score_arrays,
            "array_receipts": {
                name: acquisition._array_receipt(array)
                for name, array in score_arrays.items()
            },
            "channel_count": 1,
            "smoothing_sigma_px": 7.5,
        },
        "ranking": _primary_ranking(scores, candidate_ids, evaluable),
        "selected_pose_error": {"available": True},
        "coverage": {
            "target_fixed_valid_pixel_count": 6,
            "independent_truth_brain_pixel_count": brain_pixels,
            "evaluable": bool(evaluable),
            "failure_reason": None
            if evaluable
            else "independent truth atlas render has zero finite-crop brain support",
            "zero_support_truth_policy": "save raw ranking, count top1/top3 false, never redraw",
        },
        "exact_controls": {
            name: _control(name) for name in primary_v2.EXACT_CONTROL_NAMES
        },
    }
    return _reseal_primary(result)


def _null_summary_result(primary):
    candidate_ids = list(primary["candidate_reference"]["ordered_candidate_ids"])
    scores = np.ones(40, dtype=np.float64)
    scores[0] = 0.0
    score_arrays = {
        "semantic_score_float64": scores,
        "raw_id_agreement_float64": np.zeros(40, dtype=np.float64),
        "mask_dice_float64": np.zeros(40, dtype=np.float64),
    }
    ranking = rank_candidate_ids(scores, candidate_ids, candidate_ids[0])
    ranking["top3"] = bool(ranking["true_rank"] <= 3)
    observation = primary["provenance"]["observation_and_realization"]
    numeric_coordinates = {
        name: observation[name] for name in null_gate._NULL_RNG_COORDINATES[:-1]
    } | {"null_index": 0}
    root = null_gate.DEFAULT_SEMANTIC_NULL_ROOT_SEED
    seed = null_gate.derive_semantic_null_seed_v2(
        root, *(numeric_coordinates[name] for name in null_gate._NULL_RNG_COORDINATES)
    )
    rng_contract = {
        "domain": null_gate.SEMANTIC_NULL_RNG_DOMAIN,
        "null_root_seed_uint64": f"0x{root:016x}",
        "numeric_coordinates": numeric_coordinates,
        "excluded_coordinates": list(null_gate._NULL_RNG_EXCLUSIONS),
        "seed_uint64": f"0x{seed:016x}",
        "generator": "numpy.random.PCG64DXSM",
        "redraw_count": 0,
    }
    target = primary["target_reference"]
    candidates = primary["candidate_reference"]
    valid_count = target["fixed_valid_pixel_count"]
    valid_indices = np.arange(valid_count, dtype=np.int64)
    permutation = valid_indices[::-1].copy()
    null_labels = np.array([[31, 17, 31], [17, 31, 17]], dtype=np.int64)
    null_target_reference = {
        "shape_h_w": list(target["shape_h_w"]),
        "original_target_labels_receipt": acquisition._json_value(
            target["labels_receipt"]
        ),
        "null_target_labels_receipt": acquisition._array_receipt(null_labels),
        "fixed_valid_mask_receipt": acquisition._json_value(
            target["fixed_valid_mask_receipt"]
        ),
        "valid_index_count": valid_count,
        "valid_indices_receipt": acquisition._array_receipt(valid_indices),
        "permutation_receipt": acquisition._array_receipt(permutation),
        "changed_valid_pixel_count": valid_count,
        "changed_valid_pixel_fraction": 1.0,
        "degenerate": False,
        "degeneracy_reason": None,
        "degenerate_policy": "record and retain; never redraw",
        "outside_fixed_valid_unchanged": True,
        "label_histogram_preserved": True,
        "channel_definition_preserved": True,
        "scorer_denominator_preserved": True,
        "primary_channel_reference": acquisition._json_value(
            target["channel_reference"]
        ),
    }
    candidate_reference = {
        "candidate_count": 40,
        "ordered_candidate_ids": candidate_ids,
        "truth_candidate_id": candidate_ids[0],
        "candidate_label_stack_receipt": acquisition._json_value(
            candidates["candidate_label_stack_receipt"]
        ),
    }
    upstream = primary["upstream_reference"]
    null_upstream = {
        "semantic_oracle_result_id": primary["semantic_oracle_result_id"],
        "semantic_oracle_result_receipt_sha256": primary["receipt_sha256"],
        "candidate_bank_id": upstream["candidate_bank_id"],
        "candidate_bank_receipt_sha256": upstream["candidate_bank_receipt_sha256"],
        "synthetic_realization_id": upstream["synthetic_realization_id"],
        "synthetic_realization_receipt_sha256": upstream[
            "synthetic_realization_receipt_sha256"
        ],
        "finite_plane_pose_truth_id": upstream["finite_plane_pose_truth_id"],
        "v2_context_sha256": upstream["v2_context_sha256"],
    }
    permutation_evidence = {
        "valid_index_count": valid_count,
        "valid_indices_receipt": acquisition._array_receipt(valid_indices),
        "permutation_receipt": acquisition._array_receipt(permutation),
        "outside_fixed_valid_unchanged": True,
        "redraw_count": 0,
    }
    histogram_evidence = {
        "original_label_ids": [17, 31],
        "original_label_counts": [3, 3],
        "null_label_ids": [17, 31],
        "null_label_counts": [3, 3],
        "primary_channel_reference": acquisition._json_value(
            target["channel_reference"]
        ),
        "null_large_region_ids": [],
        "null_small_region_ids": [17, 31],
        "null_channel_count": 1,
    }
    denominator_evidence = {
        "fixed_valid_mask_receipt": acquisition._json_value(
            target["fixed_valid_mask_receipt"]
        ),
        "candidate_label_stack_receipt": acquisition._json_value(
            candidates["candidate_label_stack_receipt"]
        ),
        "candidate_bank_id": upstream["candidate_bank_id"],
        "ordered_candidate_ids": candidate_ids,
        "truth_candidate_id": candidate_ids[0],
    }
    result = {
        "schema_version": null_gate.SEMANTIC_NULL_RESULT_V2_SCHEMA,
        "algorithm": null_gate.SEMANTIC_NULL_RESULT_V2_ALGORITHM,
        "implementation_source_sha256": null_gate._source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "asset_dependencies": acquisition._json_value(
            null_gate._NO_LEARNED_ASSET_DEPENDENCIES
        ),
        "scope": acquisition._json_value(null_gate._NULL_SCOPE_CONTRACT),
        "upstream_reference": null_upstream,
        "provenance": acquisition._json_value(primary["provenance"]),
        "rng_contract": rng_contract,
        "target_reference": null_target_reference,
        "candidate_reference": candidate_reference,
        "scores": {
            "arrays": score_arrays,
            "array_receipts": {
                name: acquisition._array_receipt(array)
                for name, array in score_arrays.items()
            },
            "channel_count": 1,
            "smoothing_sigma_px": primary["scores"]["smoothing_sigma_px"],
        },
        "ranking": ranking,
        "exact_controls": {
            "numeric_lineage_seed_exclusion": null_gate._control_record(
                "numeric_lineage_seed_exclusion", True, rng_contract
            ),
            "within_fixed_valid_permutation_bijection": null_gate._control_record(
                "within_fixed_valid_permutation_bijection",
                True,
                permutation_evidence,
            ),
            "label_histogram_and_channel_preservation": null_gate._control_record(
                "label_histogram_and_channel_preservation",
                True,
                histogram_evidence,
            ),
            "fixed_denominator_and_candidate_bank_identity": null_gate._control_record(
                "fixed_denominator_and_candidate_bank_identity",
                True,
                denominator_evidence,
            ),
        },
    }
    return _reseal_null(result)


def _rebind_null_to_primary(null, primary):
    changed = dict(null)
    upstream = dict(null["upstream_reference"])
    upstream["semantic_oracle_result_id"] = primary["semantic_oracle_result_id"]
    upstream["semantic_oracle_result_receipt_sha256"] = primary["receipt_sha256"]
    changed["upstream_reference"] = upstream
    return _reseal_null(changed)


def _panel(zero_support_case=None):
    plan = []
    records = []
    case_index = 0
    for animal_index in range(4):
        for stratum in null_gate.PLANE_STRATA:
            item = {
                "case_id": f"case-{case_index:02d}",
                "plane_stratum": stratum,
                "animal_index": animal_index,
                "animal_id": f"synthetic-animal-{animal_index}",
            }
            plan.append(item)
            primary = _primary_summary_result(
                item, case_index, evaluable=case_index != zero_support_case
            )
            records.append(
                {
                    "case_id": item["case_id"],
                    "status": "completed",
                    "primary_result": primary,
                    "null_result": _null_summary_result(primary),
                    "failure": None,
                }
            )
            case_index += 1
    return plan, records


def _replace_record(records, index, primary=None, null=None):
    changed = list(records)
    record = dict(changed[index])
    if primary is not None:
        record["primary_result"] = primary
    if null is not None:
        record["null_result"] = null
    changed[index] = record
    return changed


def test_frozen_panel_layout_passes_and_reports_each_stratum_and_animal():
    plan, records = _panel()
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, records, panel_id=PANEL_ID
    )

    assert summary["metrics"]["overall"]["scheduled_count"] == 24
    assert summary["metrics"]["gate_groups"]["reference"]["scheduled_count"] == 4
    assert summary["metrics"]["gate_groups"]["pooled_cardinal"]["scheduled_count"] == 12
    assert all(
        metrics["scheduled_count"] == 4
        for metrics in summary["metrics"]["by_stratum"].values()
    )
    assert len(summary["metrics"]["by_synthetic_animal"]) == 4
    assert all(
        item["metrics"]["scheduled_count"] == 6
        for item in summary["metrics"]["by_synthetic_animal"]
    )
    assert all(summary["gates"].values())
    assert summary["passed"] is True
    assert summary["scope"]["benchmark_or_final_test_claim"] is False
    assert summary["scope"]["weighted_correspondence_sensitivity_included"] is False
    assert summary["scope"]["aggregate_artifact_checks_are_self_consistency_only"] is True
    assert summary["scope"][
        "scientific_qualification_requires_live_deterministic_primary_and_null_verification"
    ] is True
    assert summary["scope"][
        "artifact_hashes_are_not_authentication_against_hostile_coherent_resigning"
    ] is True


def test_zero_support_truth_is_failure_adverse_and_retained_in_denominator():
    plan, records = _panel(zero_support_case=0)
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, records, panel_id=PANEL_ID
    )
    reference = summary["metrics"]["gate_groups"]["reference"]
    zero_support = summary["case_contributions"][0]

    assert reference["scheduled_count"] == 4
    assert reference["primary_top1_rate"] == 0.75
    assert zero_support["artifact_contract_self_consistent"] is True
    assert zero_support["evaluable"] is False
    assert zero_support["primary_top1"] is False
    assert zero_support["primary_top3"] is False
    assert zero_support["primary_rank"] == 40
    assert zero_support["primary_win_fraction"] == 0.0
    assert summary["gates"]["reference"] is False
    assert summary["passed"] is False


def test_missing_scheduled_case_is_failure_adverse_and_never_dropped():
    plan, records = _panel()
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, records[:-1], panel_id=PANEL_ID
    )
    missing = summary["case_contributions"][-1]

    assert summary["metrics"]["overall"]["scheduled_count"] == 24
    assert missing["status"] == "execution_failure"
    assert missing["primary_top1"] is False
    assert missing["primary_top3"] is False
    assert missing["primary_rank"] == 40
    assert missing["primary_win_fraction"] == 0.0
    assert missing["null_top1"] is True
    assert missing["null_reciprocal_rank"] == 1.0
    assert summary["gates"]["execution_receipt_control_completeness"] is False
    assert summary["gates"]["edge_or_partial"] is False
    assert summary["passed"] is False


def test_gate_summary_external_panel_id_replay_json_and_tamper_rejection():
    plan, records = _panel()
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, records, panel_id=PANEL_ID
    )
    replayed = null_gate.replay_arbitrary_plane_semantic_gate_summary_v2(
        summary, plan, records, expected_panel_id=PANEL_ID
    )
    assert replayed["receipt_sha256"] == summary["receipt_sha256"]
    null_gate.verify_arbitrary_plane_semantic_gate_summary_v2(
        summary, plan, records, expected_panel_id=PANEL_ID
    )
    json_summary = json.loads(json.dumps(acquisition._json_value(summary)))
    null_gate.verify_arbitrary_plane_semantic_gate_summary_v2(
        json_summary, plan, records, expected_panel_id=PANEL_ID
    )

    with pytest.raises(ValueError, match="canonical lowercase SHA-256"):
        null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
            plan, records, panel_id="not-a-panel-hash"
        )
    with pytest.raises(ValueError, match="external expectation"):
        null_gate.replay_arbitrary_plane_semantic_gate_summary_v2(
            summary, plan, records, expected_panel_id=WRONG_PANEL_ID
        )
    with pytest.raises(ValueError, match="structure or receipt"):
        null_gate.verify_arbitrary_plane_semantic_gate_summary_v2(
            summary, plan, records, expected_panel_id=WRONG_PANEL_ID
        )

    changed = dict(summary)
    changed_gates = dict(summary["gates"])
    changed_gates["reference"] = False
    changed["gates"] = changed_gates
    with pytest.raises(ValueError, match="structure or receipt"):
        null_gate.verify_arbitrary_plane_semantic_gate_summary_v2(
            changed, plan, records, expected_panel_id=PANEL_ID
        )


def test_frozen_panel_rejects_every_nonexact_24_case_layout():
    plan, records = _panel()
    with pytest.raises(ValueError, match="exactly 24"):
        null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
            plan[:-1], records[:-1], panel_id=PANEL_ID
        )
    with pytest.raises(ValueError, match="exactly 24"):
        null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
            plan + [dict(plan[0], case_id="case-extra")], records, panel_id=PANEL_ID
        )

    wrong_identity = [dict(item) for item in plan]
    wrong_identity[1]["animal_id"] = "different-animal"
    with pytest.raises(ValueError, match="one-to-one"):
        null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
            wrong_identity, records, panel_id=PANEL_ID
        )

    wrong_strata = [dict(item) for item in plan]
    wrong_strata[0]["plane_stratum"] = "near_AP"
    with pytest.raises(ValueError, match="every stratum exactly once"):
        null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
            wrong_strata, records, panel_id=PANEL_ID
        )


@pytest.mark.parametrize("invalid_field", ["primary_scope", "null_rng", "null_target"])
def test_empty_contract_fields_fail_self_consistency_completeness(invalid_field):
    plan, records = _panel()
    primary = records[0]["primary_result"]
    null = records[0]["null_result"]
    if invalid_field == "primary_scope":
        changed_primary = dict(primary)
        changed_primary["scope"] = {}
        changed_primary = _reseal_primary(changed_primary)
        changed_null = _rebind_null_to_primary(null, changed_primary)
    else:
        changed_primary = primary
        changed_null = dict(null)
        changed_null["rng_contract" if invalid_field == "null_rng" else "target_reference"] = {}
        changed_null = _reseal_null(changed_null)
    changed_records = _replace_record(
        records, 0, primary=changed_primary, null=changed_null
    )
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, changed_records, panel_id=PANEL_ID
    )

    contribution = summary["case_contributions"][0]
    assert contribution["artifact_contract_self_consistent"] is False
    assert contribution["combined_complete"] is False
    assert contribution["primary_top1"] is False
    assert contribution["null_top1"] is True
    assert summary["gates"]["execution_receipt_control_completeness"] is False


def test_out_of_range_score_is_failure_adverse_after_coherent_rereceipt():
    plan, records = _panel()
    primary = records[0]["primary_result"]
    changed_primary = dict(primary)
    changed_scores = dict(primary["scores"])
    changed_arrays = dict(primary["scores"]["arrays"])
    semantic = np.array(changed_arrays["semantic_score_float64"], copy=True)
    semantic[0] = 1.25
    changed_arrays["semantic_score_float64"] = semantic
    changed_receipts = dict(primary["scores"]["array_receipts"])
    changed_receipts["semantic_score_float64"] = acquisition._array_receipt(semantic)
    changed_scores["arrays"] = changed_arrays
    changed_scores["array_receipts"] = changed_receipts
    changed_primary["scores"] = changed_scores
    changed_primary["ranking"] = _primary_ranking(
        semantic,
        list(primary["candidate_reference"]["ordered_candidate_ids"]),
        True,
    )
    changed_primary = _reseal_primary(changed_primary)
    changed_null = _rebind_null_to_primary(records[0]["null_result"], changed_primary)
    changed_records = _replace_record(
        records, 0, primary=changed_primary, null=changed_null
    )
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, changed_records, panel_id=PANEL_ID
    )

    assert summary["case_contributions"][0][
        "artifact_contract_self_consistent"
    ] is False
    assert summary["gates"]["execution_receipt_control_completeness"] is False


def test_coherently_rereceipted_null_scope_tamper_fails_contract():
    plan, records = _panel()
    changed_null = dict(records[0]["null_result"])
    changed_scope = dict(changed_null["scope"])
    changed_scope["model_training_or_benchmark_claim"] = True
    changed_null["scope"] = changed_scope
    changed_null = _reseal_null(changed_null)
    assert null_gate._null_self_audit(changed_null)["receipt"] is True
    assert null_gate._null_self_audit(changed_null)["structure"] is False

    changed_records = _replace_record(records, 0, null=changed_null)
    summary = null_gate.make_arbitrary_plane_semantic_gate_summary_v2(
        plan, changed_records, panel_id=PANEL_ID
    )
    assert summary["case_contributions"][0][
        "artifact_contract_self_consistent"
    ] is False
    assert summary["gates"]["execution_receipt_control_completeness"] is False
