import hashlib

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from training.arbitrary_plane_semantic_oracle import (
    EXACT_CONTROL_NAMES,
    build_oracle_target,
    canonical_payload_sha256,
    finite_point_error,
    fixed_oracle_target,
    frozen_orientation_family,
    rank_candidate_ids,
    rank_truth,
    rp2_plane_error,
    score_semantic_candidates,
    semantic_gate_summary,
    shuffled_target_index,
    wilson_interval,
)


def test_fixed_oracle_target_uses_exact_fixed_to_source_pullback_and_validity():
    source = np.arange(20, dtype=np.int64).reshape(4, 5)
    yy, xx = np.indices(source.shape, dtype=np.float32)
    fixed_to_source = np.stack((xx + 1.0, yy))
    valid = np.ones(source.shape, dtype=bool)
    valid[:, -1] = False
    target, returned_valid = fixed_oracle_target(
        {
            "arrays": {
                "source_annotation": source,
                "fixed_to_source_map": fixed_to_source,
                "fixed_valid_correspondence_mask": valid,
            }
        }
    )
    assert np.array_equal(target[:, :-1], source[:, 1:])
    assert np.all(target[:, -1] == 0)
    assert np.array_equal(returned_valid, valid)


def test_oracle_target_pitch_and_semantics_ignore_paired_outline_observation():
    source = np.arange(1, 21, dtype=np.int64).reshape(4, 5)
    yy, xx = np.indices(source.shape, dtype=np.float32)
    artifact = {
        "arrays": {
            "source_annotation": source,
            "fixed_to_source_map": np.stack((xx, yy)),
            "fixed_valid_correspondence_mask": np.ones(source.shape, dtype=bool),
            "model_input_image": np.zeros(source.shape, dtype=np.float32),
        },
        "finite_parent": {
            "geometry": {
                "reference_aspect_policy": {
                    "pixel_pitch_u_um": 25.0,
                    "pixel_pitch_v_um": 25.0,
                }
            }
        },
    }
    first = build_oracle_target(artifact)
    artifact["arrays"]["model_input_image"][:] = 1.0
    second = build_oracle_target(artifact)
    assert first["pixel_pitch_um"] == 25.0
    assert np.array_equal(first["labels"], second["labels"])
    assert np.array_equal(first["fixed_valid_mask"], second["fixed_valid_mask"])


def test_semantic_truth_ranks_first_and_ablation_scores_share_fixed_denominator():
    target = np.zeros((12, 14), dtype=np.int64)
    target[1:11, 1:7] = 11
    target[1:11, 7:13] = 29
    target[5, 6:8] = 101
    valid = target != 0
    shifted = np.roll(target, 3, axis=1)
    background = np.zeros_like(target)
    result = score_semantic_candidates(
        target, np.stack((shifted, target, background)), valid, pixel_pitch_um=150.0
    )
    rank = rank_truth(result["semantic_score"], 1)
    assert rank["top1"]
    assert rank["true_rank"] == 1
    assert result["channel_count"] == 3
    assert result["raw_id_agreement"][1] == 1.0
    assert result["mask_dice"][1] == 1.0
    assert result["semantic_score"][1] == pytest.approx(1.0)
    assert result["semantic_score"][1] > result["semantic_score"][0] > result["semantic_score"][2]
    for key in ("semantic_score", "raw_id_agreement", "mask_dice"):
        assert np.all((result[key] >= 0.0) & (result[key] <= 1.0))


def test_invalid_labels_are_zeroed_before_smoothing_and_cannot_leak_into_score():
    target = np.zeros((15, 15), dtype=np.int64)
    target[5:10, 5:10] = 7
    valid = np.zeros_like(target, dtype=bool)
    valid[5:10, 5:10] = True
    first = target.copy()
    second = target.copy()
    second[~valid] = 7
    scores = score_semantic_candidates(
        target, np.stack((first, second)), valid, pixel_pitch_um=25.0
    )["semantic_score"]
    assert np.array_equal(scores, np.ones(2))


def test_gaussian_dice_matches_the_frozen_float64_formula():
    target = np.zeros((9, 11), dtype=np.int64)
    target[2:7, 3:8] = 17
    candidate = np.roll(target, 1, axis=1)
    valid = target != 0
    result = score_semantic_candidates(target, np.stack((candidate, target)), valid, 25.0)
    target_channel = gaussian_filter(
        (valid & (target == 17)).astype(np.float64),
        sigma=3.0,
        truncate=3.0,
        mode="constant",
        cval=0.0,
    )
    candidate_channel = gaussian_filter(
        (valid & (candidate == 17)).astype(np.float64),
        sigma=3.0,
        truncate=3.0,
        mode="constant",
        cval=0.0,
    )
    expected = (2 * np.sum(target_channel[valid] * candidate_channel[valid]) + 1e-12) / (
        np.sum(target_channel[valid] ** 2) + np.sum(candidate_channel[valid] ** 2) + 1e-12
    )
    assert result["smoothing_sigma_px"] == 3.0
    assert result["semantic_score"][0] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert result["semantic_score"][1] == 1.0


def test_candidate_order_is_equivariant_and_ties_fail_top1_conservatively():
    target = np.zeros((8, 10), dtype=np.int64)
    target[:, :5] = 3
    target[:, 5:] = 9
    valid = np.ones_like(target, dtype=bool)
    candidates = np.stack((target, np.roll(target, 1, 1), np.zeros_like(target)))
    first = score_semantic_candidates(target, candidates, valid, 100.0)["semantic_score"]
    order = np.array([2, 0, 1])
    second = score_semantic_candidates(target, candidates[order], valid, 100.0)["semantic_score"]
    assert np.array_equal(second, first[order])
    tied = rank_truth(np.array([0.8, 0.8 - 0.5e-12, 0.2]), 0)
    assert not tied["top1"]
    assert tied["tied_maximum_indices"] == [0, 1]
    assert tied["selected_index"] is None
    unique = rank_truth(np.array([0.8, 0.8 - 2.0e-12, 0.2]), 0)
    assert unique["top1"]
    assert unique["selected_index"] == 0


def test_tied_candidate_ids_and_chunked_scores_are_permutation_equivariant():
    target = np.full((8, 10), 3, dtype=np.int64)
    target[:, 5:] = 9
    candidates = np.stack(
        [target if index in (0, 9) else np.roll(target, index % 5 + 1, axis=1) for index in range(40)]
    )
    candidate_ids = [f"candidate-{index:02d}" for index in range(40)]
    scores = score_semantic_candidates(
        target, candidates, np.ones_like(target, dtype=bool), 100.0
    )["semantic_score"]
    first = rank_candidate_ids(scores, candidate_ids, candidate_ids[0])
    order = np.roll(np.arange(40), 13)
    permuted_scores = score_semantic_candidates(
        target, candidates[order], np.ones_like(target, dtype=bool), 100.0
    )["semantic_score"]
    second = rank_candidate_ids(
        permuted_scores, [candidate_ids[index] for index in order], candidate_ids[0]
    )
    assert np.array_equal(permuted_scores, scores[order])
    assert first["tied_maximum_candidate_ids"] == ["candidate-00", "candidate-09"]
    assert second["tied_maximum_candidate_ids"] == first["tied_maximum_candidate_ids"]
    assert first["selected_candidate_id"] is None
    assert second["selected_candidate_id"] is None


def test_rp2_sign_equivalence_aligns_offset_and_rejects_zero_normals():
    same = rp2_plane_error([1, 0, 0], 125.0, [-1, 0, 0], -125.0)
    assert same["normal_geodesic_angle_deg"] == 0.0
    assert same["sign_aligned_offset_error_um"] == 0.0
    tilted = rp2_plane_error([1, 0, 0], 0.0, [np.sqrt(0.5), np.sqrt(0.5), 0], 30.0)
    assert tilted["normal_geodesic_angle_deg"] == pytest.approx(45.0)
    assert tilted["sign_aligned_offset_error_um"] == 30.0
    with pytest.raises(ValueError, match="unit"):
        rp2_plane_error([0, 0, 0], 0.0, [1, 0, 0], 0.0)
    with pytest.raises(ValueError, match="finite"):
        rp2_plane_error([1, 0, 0], np.nan, [1, 0, 0], 0.0)


def test_region_threshold_and_small_pool_are_target_defined_only():
    target = np.full((8, 8), 3, dtype=np.int64)
    target[:2, :] = 1
    target[2:5, :5] = 2
    valid = np.ones_like(target, dtype=bool)
    candidate_only_id = target.copy()
    candidate_only_id[2, 0] = 999
    result = score_semantic_candidates(
        target, np.stack((target, candidate_only_id)), valid, pixel_pitch_um=100.0
    )
    assert result["target_large_region_ids"].tolist() == [1, 3]
    assert result["target_small_region_ids"].tolist() == [2]
    assert result["channel_count"] == 3
    assert result["semantic_score"][1] < result["semantic_score"][0]


def test_negative_allen_ids_are_rejected():
    target = np.ones((4, 4), dtype=np.int64)
    candidates = np.stack((target, target.copy()))
    candidates[1, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        score_semantic_candidates(target, candidates, np.ones_like(target, bool), 25.0)


def test_finite_point_error_uses_x_over_width_without_half_pixel_shift():
    truth = np.array([0, 0, 0, 4, 0, 0, 0, 2, 0], dtype=np.float64)
    selected = np.array([0, 0, 0, 8, 0, 0, 0, 2, 0], dtype=np.float64)
    result = finite_point_error(truth, selected, np.ones((1, 4), dtype=bool))
    assert result["corresponding_point_rms_um"] == pytest.approx(np.sqrt(3.5))
    assert result["corresponding_point_p95_um"] == pytest.approx(2.85)


def test_wilson_interval_and_input_contracts_are_finite_and_deterministic():
    lower, upper = wilson_interval(52, 64)
    assert lower == pytest.approx(0.7002563944)
    assert upper == pytest.approx(0.8893535683)
    with pytest.raises(ValueError, match="integer"):
        score_semantic_candidates(
            np.ones((3, 3), dtype=np.float32),
            np.ones((2, 3, 3), dtype=np.int64),
            np.ones((3, 3), dtype=bool),
            25.0,
        )


def test_frozen_shuffled_mapping_and_semantic_gate_summary():
    assert shuffled_target_index(0) == 17
    assert shuffled_target_index(63) == 16
    case_results, shuffled_results, exact_controls = _gate_fixture()
    summary = semantic_gate_summary(
        case_results, shuffled_results, exact_controls=exact_controls
    )
    assert summary["metrics"]["top1_successes"] == 52
    assert summary["metrics"]["shuffled_original_truth_top1_rate"] == 6 / 64
    assert summary["passed"]
    exact_controls["exact_replay"]["passed"] = False
    exact_controls["exact_replay"]["evidence_receipt_sha256"] = canonical_payload_sha256(
        {
            "control": "exact_replay",
            "passed": False,
            "evidence": exact_controls["exact_replay"]["evidence"],
        }
    )
    assert not semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)[
        "passed"
    ]


def test_semantic_gate_rejects_duplicate_statistical_units_and_shuffled_target_tamper():
    case_results, shuffled_results, exact_controls = _gate_fixture()
    case_results[1]["paired_view_group_id"] = case_results[0]["paired_view_group_id"]
    with pytest.raises(ValueError, match="unique"):
        semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)
    case_results, shuffled_results, exact_controls = _gate_fixture()
    shuffled_results[0]["target"] = case_results[18]["target"]
    with pytest.raises(ValueError, match="source case"):
        semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)


def test_semantic_gate_derives_metrics_from_raw_scores_and_checks_boundaries():
    case_results, shuffled_results, exact_controls = _gate_fixture()
    case_results[0]["scores"]["semantic"] = _scores_for_rank(2)
    summary = semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)
    assert summary["metrics"]["top1_successes"] == 51
    assert not summary["gates"]["top1_rate"]
    case_results, shuffled_results, exact_controls = _gate_fixture()
    shuffled_results[6]["scores"]["semantic"] = _scores_for_rank(1)
    summary = semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)
    assert summary["metrics"]["shuffled_original_truth_top1_rate"] == 7 / 64
    assert not summary["gates"]["shuffled_top1"]


def test_exact_control_receipts_bind_name_and_passed_value():
    case_results, shuffled_results, exact_controls = _gate_fixture()
    exact_controls["exact_replay"]["passed"] = False
    with pytest.raises(ValueError, match="does not match"):
        semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)
    case_results, shuffled_results, exact_controls = _gate_fixture()
    exact_controls["exact_replay"]["evidence"] = exact_controls[
        "rp2_sign_equivalence"
    ]["evidence"]
    with pytest.raises(ValueError, match="cross-bound"):
        semantic_gate_summary(case_results, shuffled_results, exact_controls=exact_controls)


def test_frozen_orientation_is_derived_from_case_index_and_truth_normal():
    assert frozen_orientation_family(0, [1, 0, 0]) == "near_AP"
    assert frozen_orientation_family(12, [0, -1, 0]) == "near_DV"
    assert frozen_orientation_family(24, [0, 0, 1]) == "near_ML"
    assert frozen_orientation_family(36, np.ones(3) / np.sqrt(3)) == "general_oblique"
    with pytest.raises(ValueError, match="stratum"):
        frozen_orientation_family(36, [1, 0, 0])


def _digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _scores_for_rank(rank):
    scores = np.linspace(0.79, 0.01, 40)
    truth = 0
    scores[truth] = 1.0 if rank == 1 else 0.8 if rank < 40 else 0.0
    if rank > 1:
        scores[1:rank] = np.linspace(0.99, scores[truth] + 0.01, rank - 1)
    return scores.tolist()


def _gate_fixture():
    orientation_rows = (
        [("near_AP", np.array([1.0, 0.0, 0.0]), 10)] * 12
        + [("near_DV", np.array([0.0, 1.0, 0.0]), 10)] * 12
        + [("near_ML", np.array([0.0, 0.0, 1.0]), 10)] * 12
        + [("general_oblique", np.ones(3) / np.sqrt(3), 22)] * 28
    )
    local_counts = {name: 0 for name in ("near_AP", "near_DV", "near_ML", "general_oblique")}
    primary = []
    for index, (orientation, normal, successful_count) in enumerate(orientation_rows):
        local_index = local_counts[orientation]
        local_counts[orientation] += 1
        candidate_ids = [_digest(f"candidate-{index}-{candidate}") for candidate in range(40)]
        paired_id = _digest(f"paired-{index}")
        channel_receipt = {"large_ids": [1, 2], "small_ids": [], "channel_count": 2}
        target = {
            "source_case_index": index,
            "paired_view_group_id": paired_id,
            "labels_receipt": {"shape": [192, 256], "array_sha256": _digest(f"labels-{index}")},
            "mask_receipt": {"shape": [192, 256], "array_sha256": _digest(f"mask-{index}")},
            "channel_receipt": channel_receipt,
            "channel_receipt_sha256": canonical_payload_sha256(channel_receipt),
            "pixel_pitch_um": 25.0,
        }
        target["target_receipt_sha256"] = canonical_payload_sha256(target)
        primary.append(
            {
                "case_index": index,
                "parent_plane_realization_id": _digest(f"parent-{index}"),
                "paired_view_group_id": paired_id,
                "outline_descendant_ids": [_digest(f"outline-{index}-{mode}") for mode in range(3)],
                "candidate_bank_id": _digest(f"bank-{index}"),
                "ordered_candidate_ids": candidate_ids,
                "truth_candidate_id": candidate_ids[0],
                "truth_normal_ap_dv_ml": normal.tolist(),
                "orientation": orientation,
                "target": target,
                "scores": {
                    "semantic": _scores_for_rank(1 if local_index < successful_count else 2),
                    "raw_ID_agreement": [0.5] * 40,
                    "mask_only_Dice": [0.75] * 40,
                    "channel_count": 2,
                    "smoothing_sigma_px": 3.0,
                },
            }
        )
    shuffled = []
    for index, base in enumerate(primary):
        shuffled.append(
            {
                "case_index": index,
                "paired_view_group_id": base["paired_view_group_id"],
                "candidate_bank_id": base["candidate_bank_id"],
                "ordered_candidate_ids": base["ordered_candidate_ids"],
                "truth_candidate_id": base["truth_candidate_id"],
                "target": primary[shuffled_target_index(index)]["target"],
                "scores": {
                    "semantic": _scores_for_rank(1 if index < 6 else 40),
                    "raw_ID_agreement": [0.25] * 40,
                    "mask_only_Dice": [0.5] * 40,
                    "channel_count": 2,
                    "smoothing_sigma_px": 3.0,
                },
            }
        )
    exact_controls = {}
    for name in EXACT_CONTROL_NAMES:
        evidence = {"control": name, "checked_cases": 64}
        exact_controls[name] = {
            "passed": True,
            "evidence": evidence,
            "evidence_receipt_sha256": canonical_payload_sha256(
                {"control": name, "passed": True, "evidence": evidence}
            ),
        }
    return primary, shuffled, exact_controls
