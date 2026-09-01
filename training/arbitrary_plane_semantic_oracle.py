"""Model-free semantic scoring for the arbitrary-plane pose oracle."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from training.arbitrary_plane_synthetic_ops import nearest_sample_labels


SEMANTIC_ORACLE_VERSION = "arbitrary-plane-semantic-oracle/v1"
MINIMUM_REGION_PIXELS = 16
SMOOTHING_SIGMA_UM = 75.0
SMOOTHING_TRUNCATE = 3.0
SCORE_EPSILON = 1e-12
TIE_TOLERANCE = 1e-12
CANDIDATE_CHUNK_SIZE = 8
ORIENTATION_COUNTS = {"near_AP": 12, "near_DV": 12, "near_ML": 12, "general_oblique": 28}
EXACT_CONTROL_NAMES = {
    "exact_replay",
    "candidate_order_permutation_equivariance",
    "rp2_sign_equivalence",
    "truth_metadata_coordinate_channel_exclusion",
    "xy_over_wh_coordinate_contract",
}


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_payload_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_scalar,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fixed_oracle_target(synthetic_artifact: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    arrays = synthetic_artifact["arrays"]
    source = np.asarray(arrays["source_annotation"])
    fixed_to_source = np.asarray(arrays["fixed_to_source_map"])
    valid = np.asarray(arrays["fixed_valid_correspondence_mask"], dtype=bool)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("source_annotation must be one integer raster")
    if np.any(source < 0):
        raise ValueError("source_annotation must contain nonnegative Allen IDs")
    if fixed_to_source.shape != (2, *valid.shape) or source.shape != valid.shape:
        raise ValueError("oracle source, map and fixed-valid mask must share one raster shape")
    target = nearest_sample_labels(source, fixed_to_source)
    if not valid.any() or np.any(target[valid] == 0):
        raise ValueError("fixed-valid oracle pixels must all have nonzero anatomical labels")
    target = np.where(valid, target, np.zeros((), dtype=target.dtype))
    return target, valid


def build_oracle_target(synthetic_artifact: dict[str, object]) -> dict[str, np.ndarray | float]:
    labels, valid = fixed_oracle_target(synthetic_artifact)
    aspect = synthetic_artifact["finite_parent"]["geometry"]["reference_aspect_policy"]
    pitch_u = float(aspect["pixel_pitch_u_um"])
    pitch_v = float(aspect["pixel_pitch_v_um"])
    if not np.isfinite(pitch_u) or pitch_u <= 0.0 or not np.isclose(
        pitch_u, pitch_v, rtol=1e-12, atol=0.0
    ):
        raise ValueError("oracle target requires one finite positive isotropic pixel pitch")
    return {"labels": labels, "fixed_valid_mask": valid, "pixel_pitch_um": pitch_u}


def _target_channel_ids(
    target_labels: np.ndarray, fixed_valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(target_labels[fixed_valid_mask & (target_labels != 0)], return_counts=True)
    large = labels[counts >= MINIMUM_REGION_PIXELS]
    small = labels[counts < MINIMUM_REGION_PIXELS]
    if not len(large) and not len(small):
        raise ValueError("semantic target has no nonzero fixed-valid labels")
    return large, small


def _binary_channel(
    labels: np.ndarray, channel_ids: int | np.ndarray, fixed_valid_mask: np.ndarray
) -> np.ndarray:
    channel = labels == channel_ids if np.isscalar(channel_ids) else np.isin(labels, channel_ids)
    return np.where(fixed_valid_mask, channel, False).astype(np.float64)


def score_semantic_candidates(
    target_labels: np.ndarray,
    candidate_labels: np.ndarray,
    fixed_valid_mask: np.ndarray,
    pixel_pitch_um: float,
) -> dict[str, np.ndarray | float | int]:
    target = np.asarray(target_labels)
    candidates = np.asarray(candidate_labels)
    valid = np.asarray(fixed_valid_mask, dtype=bool)
    pitch = float(pixel_pitch_um)
    if target.ndim != 2 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("target labels must be one integer raster")
    if candidates.ndim != 3 or candidates.shape[1:] != target.shape:
        raise ValueError("candidate labels must have shape (K,H,W) matching the target")
    if not np.issubdtype(candidates.dtype, np.integer) or valid.shape != target.shape:
        raise ValueError("candidate labels must be integer and the fixed-valid mask must match")
    if np.any(target < 0) or np.any(candidates < 0):
        raise ValueError("semantic rasters must contain nonnegative Allen IDs")
    if not np.isfinite(pitch) or pitch <= 0.0 or not valid.any():
        raise ValueError("pixel pitch must be positive and the fixed-valid mask nonempty")
    if np.any(target[valid] == 0):
        raise ValueError("fixed-valid target pixels must have nonzero anatomical labels")

    large_ids, small_ids = _target_channel_ids(target, valid)
    sigma_px = SMOOTHING_SIGMA_UM / pitch
    channel_ids = [int(label) for label in large_ids]
    if len(small_ids):
        channel_ids.append(small_ids)
    semantic_sums = np.zeros(len(candidates), dtype=np.float64)
    for ids in channel_ids:
        target_channel = gaussian_filter(
            _binary_channel(target, ids, valid),
            sigma=sigma_px,
            truncate=SMOOTHING_TRUNCATE,
            mode="constant",
            cval=0.0,
        )
        target_squared = float(np.sum(target_channel[valid] * target_channel[valid]))
        for start in range(0, len(candidates), CANDIDATE_CHUNK_SIZE):
            stop = min(start + CANDIDATE_CHUNK_SIZE, len(candidates))
            candidate_channels = gaussian_filter(
                np.where(
                    valid[None],
                    candidates[start:stop] == ids
                    if np.isscalar(ids)
                    else np.isin(candidates[start:stop], ids),
                    False,
                ).astype(np.float64),
                sigma=(0.0, sigma_px, sigma_px),
                truncate=SMOOTHING_TRUNCATE,
                mode="constant",
                cval=0.0,
            )
            fixed_candidate_channels = candidate_channels[:, valid]
            intersection = np.sum(fixed_candidate_channels * target_channel[valid], axis=1)
            candidate_squared = np.sum(fixed_candidate_channels * fixed_candidate_channels, axis=1)
            semantic_sums[start:stop] += (2.0 * intersection + SCORE_EPSILON) / (
                target_squared + candidate_squared + SCORE_EPSILON
            )

    raw_id_agreement = []
    mask_dice = []
    target_tissue = valid & (target != 0)
    for candidate in candidates:
        raw_id_agreement.append(float(np.mean(candidate[valid] == target[valid])))
        candidate_tissue = valid & (candidate != 0)
        mask_dice.append(
            float(
                (2.0 * np.count_nonzero(target_tissue & candidate_tissue) + SCORE_EPSILON)
                / (np.count_nonzero(target_tissue) + np.count_nonzero(candidate_tissue) + SCORE_EPSILON)
            )
        )
    return {
        "semantic_score": np.clip(semantic_sums / len(channel_ids), 0.0, 1.0),
        "raw_id_agreement": np.clip(
            np.asarray(raw_id_agreement, dtype=np.float64), 0.0, 1.0
        ),
        "mask_dice": np.clip(np.asarray(mask_dice, dtype=np.float64), 0.0, 1.0),
        "target_large_region_ids": large_ids.astype(np.int64),
        "target_small_region_ids": small_ids.astype(np.int64),
        "channel_count": int(len(large_ids) + bool(len(small_ids))),
        "smoothing_sigma_px": float(sigma_px),
    }


def rank_truth(scores: np.ndarray, truth_index: int) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    truth_index = int(truth_index)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector with at least two candidates")
    if not 0 <= truth_index < len(values):
        raise ValueError("truth_index is outside the candidate bank")
    truth = float(values[truth_index])
    decoys = np.delete(values, truth_index)
    maximum = float(values.max())
    tied_maximum_indices = np.flatnonzero(np.abs(values - maximum) <= TIE_TOLERANCE).tolist()
    conservative_rank = 1 + int(np.count_nonzero(decoys >= truth - TIE_TOLERANCE))
    return {
        "truth_index": truth_index,
        "truth_score": truth,
        "top1": tied_maximum_indices == [truth_index],
        "true_rank": conservative_rank,
        "reciprocal_rank": 1.0 / conservative_rank,
        "true_versus_decoy_win_fraction": float(np.mean(decoys < truth - TIE_TOLERANCE)),
        "true_score_margin": truth - float(decoys.max()),
        "tied_maximum_indices": tied_maximum_indices,
        "selected_index": tied_maximum_indices[0] if len(tied_maximum_indices) == 1 else None,
    }


def rank_candidate_ids(
    scores: np.ndarray, ordered_candidate_ids: list[str], truth_candidate_id: str
) -> dict[str, object]:
    candidate_ids = [str(value) for value in ordered_candidate_ids]
    truth_id = str(truth_candidate_id)
    if len(candidate_ids) != len(scores) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("ordered candidate IDs must be unique and match the score count")
    if candidate_ids.count(truth_id) != 1:
        raise ValueError("the truth candidate ID must occur exactly once")
    ranking = rank_truth(scores, candidate_ids.index(truth_id))
    ranking["tied_maximum_candidate_ids"] = sorted(
        candidate_ids[index] for index in ranking["tied_maximum_indices"]
    )
    ranking["selected_candidate_id"] = (
        None if ranking["selected_index"] is None else candidate_ids[ranking["selected_index"]]
    )
    return ranking


def rp2_plane_error(
    truth_normal: np.ndarray,
    truth_offset_um: float,
    selected_normal: np.ndarray,
    selected_offset_um: float,
) -> dict[str, float]:
    truth = np.asarray(truth_normal, dtype=np.float64)
    selected = np.asarray(selected_normal, dtype=np.float64)
    if truth.shape != (3,) or selected.shape != (3,) or not np.isfinite(truth).all() or not np.isfinite(selected).all():
        raise ValueError("plane normals must be finite 3-vectors")
    if not np.isfinite(truth_offset_um) or not np.isfinite(selected_offset_um):
        raise ValueError("plane offsets must be finite")
    truth_norm = float(np.linalg.norm(truth))
    selected_norm = float(np.linalg.norm(selected))
    if not np.isclose(truth_norm, 1.0, rtol=0.0, atol=1e-9) or not np.isclose(
        selected_norm, 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("plane normals must be unit vectors")
    truth /= truth_norm
    selected /= selected_norm
    sign = -1.0 if float(np.dot(truth, selected)) < 0.0 else 1.0
    aligned = sign * selected
    angle = math.degrees(math.acos(float(np.clip(np.dot(truth, aligned), 0.0, 1.0))))
    return {
        "normal_geodesic_angle_deg": angle,
        "sign_aligned_offset_error_um": abs(float(truth_offset_um) - sign * float(selected_offset_um)),
    }


def finite_point_error(
    truth_effective_physical_ouv_um: np.ndarray,
    selected_effective_physical_ouv_um: np.ndarray,
    fixed_valid_mask: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth_effective_physical_ouv_um, dtype=np.float64)
    selected = np.asarray(selected_effective_physical_ouv_um, dtype=np.float64)
    valid = np.asarray(fixed_valid_mask, dtype=bool)
    if truth.shape != (9,) or selected.shape != (9,) or not np.isfinite(truth).all() or not np.isfinite(selected).all():
        raise ValueError("effective physical O/U/V must be finite 9-vectors")
    if valid.ndim != 2 or not valid.any():
        raise ValueError("fixed-valid mask must be one nonempty raster")
    height, width = valid.shape
    s = np.arange(width, dtype=np.float64) / width
    t = np.arange(height, dtype=np.float64) / height
    tt, ss = np.meshgrid(t, s, indexing="ij")
    truth_points = truth[:3] + ss[..., None] * truth[3:6] + tt[..., None] * truth[6:9]
    selected_points = selected[:3] + ss[..., None] * selected[3:6] + tt[..., None] * selected[6:9]
    error = np.linalg.norm(truth_points - selected_points, axis=-1)[valid]
    return {
        "corresponding_point_rms_um": float(np.sqrt(np.mean(error * error))),
        "corresponding_point_p95_um": float(np.quantile(error, 0.95, method="linear")),
    }


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    successes, count, z = int(successes), int(count), float(z)
    if count <= 0 or not 0 <= successes <= count or not np.isfinite(z) or z <= 0.0:
        raise ValueError("Wilson inputs must be a valid count and positive finite z")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    half = z / denominator * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
    return centre - half, centre + half


def shuffled_target_index(case_index: int, count: int = 64) -> int:
    case_index, count = int(case_index), int(count)
    if count != 64 or not 0 <= case_index < count:
        raise ValueError("the frozen shuffled control requires case indices 0 through 63")
    return (case_index + 17) % count


def frozen_orientation_family(case_index: int, truth_normal_ap_dv_ml: np.ndarray) -> str:
    case_index = int(case_index)
    if not 0 <= case_index < 64:
        raise ValueError("the frozen design requires case indices 0 through 63")
    normal = np.asarray(truth_normal_ap_dv_ml, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(
        np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("truth normal must be one finite unit AP-DV-ML vector")
    family, axis = (
        ("near_AP", 0)
        if case_index < 12
        else ("near_DV", 1)
        if case_index < 24
        else ("near_ML", 2)
        if case_index < 36
        else ("general_oblique", None)
    )
    if (axis is not None and abs(normal[axis]) < 0.90) or (
        axis is None and np.max(np.abs(normal)) >= 0.90
    ):
        raise ValueError("truth normal does not satisfy its frozen orientation stratum")
    return family


def semantic_gate_summary(
    case_results: list[dict[str, object]],
    shuffled_results: list[dict[str, object]],
    *,
    exact_controls: dict[str, dict[str, object]],
) -> dict[str, object]:
    if len(case_results) != 64 or len(shuffled_results) != 64:
        raise ValueError("the frozen semantic gate requires 64 base and 64 shuffled results")
    if set(exact_controls) != EXACT_CONTROL_NAMES:
        raise ValueError("exact controls must match the five frozen named controls")
    for name, record in exact_controls.items():
        if type(record.get("passed")) is not bool:
            raise ValueError(f"exact control {name} must have a strict Boolean result")
        evidence_payload = record.get("evidence")
        if not isinstance(evidence_payload, dict) or not evidence_payload:
            raise ValueError(f"exact control {name} lacks its evidence payload")
        if evidence_payload.get("control") != name:
            raise ValueError(f"exact control {name} evidence is cross-bound to another control")
        evidence = str(record.get("evidence_receipt_sha256", ""))
        if len(evidence) != 64 or any(character not in "0123456789abcdef" for character in evidence):
            raise ValueError(f"exact control {name} lacks a canonical evidence receipt")
        control_payload = {
            "control": name,
            "passed": record["passed"],
            "evidence": evidence_payload,
        }
        if evidence != canonical_payload_sha256(control_payload):
            raise ValueError(f"exact control {name} evidence receipt does not match its payload")

    primary = sorted(case_results, key=lambda result: int(result["case_index"]))
    shuffled = sorted(shuffled_results, key=lambda result: int(result["case_index"]))
    if [int(result["case_index"]) for result in primary] != list(range(64)) or [
        int(result["case_index"]) for result in shuffled
    ] != list(range(64)):
        raise ValueError("primary and shuffled case indices must each be exactly 0 through 63")
    paired_ids = [str(result["paired_view_group_id"]) for result in primary]
    parent_ids = [str(result["parent_plane_realization_id"]) for result in primary]
    bank_ids = [str(result["candidate_bank_id"]) for result in primary]
    outline_ids = [
        str(value) for result in primary for value in result["outline_descendant_ids"]
    ]
    if (
        len(set(paired_ids)) != 64
        or len(set(parent_ids)) != 64
        or len(set(bank_ids)) != 64
        or len(outline_ids) != 192
        or len(set(outline_ids)) != 192
        or any(len(result["outline_descendant_ids"]) != 3 for result in primary)
    ):
        raise ValueError("parent, paired-view, bank and three-per-base outline identities must be unique")
    orientations = np.asarray(
        [
            frozen_orientation_family(index, result["truth_normal_ap_dv_ml"])
            for index, result in enumerate(primary)
        ]
    )
    if any(str(result["orientation"]) != orientations[index] for index, result in enumerate(primary)):
        raise ValueError("orientation labels must be derived from case index and truth normal")

    def validate_target(target: dict[str, object], source_index: int, paired_id: str) -> None:
        required = {
            "source_case_index",
            "paired_view_group_id",
            "labels_receipt",
            "mask_receipt",
            "channel_receipt",
            "channel_receipt_sha256",
            "pixel_pitch_um",
            "target_receipt_sha256",
        }
        if not required <= target.keys() or int(target["source_case_index"]) != source_index:
            raise ValueError("target receipt has the wrong source case")
        if str(target["paired_view_group_id"]) != paired_id:
            raise ValueError("target receipt has the wrong paired-view group")
        if target["labels_receipt"].get("shape") != [192, 256] or target["mask_receipt"].get(
            "shape"
        ) != [192, 256]:
            raise ValueError("the frozen semantic target raster must be 192 by 256")
        if not np.isfinite(float(target["pixel_pitch_um"])) or float(target["pixel_pitch_um"]) <= 0:
            raise ValueError("target pixel pitch must be finite and positive")
        for value in (
            target["labels_receipt"].get("array_sha256", ""),
            target["mask_receipt"].get("array_sha256", ""),
            target["channel_receipt_sha256"],
            target["target_receipt_sha256"],
        ):
            digest = str(value)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("target receipts must contain canonical SHA-256 digests")
        if target["channel_receipt_sha256"] != canonical_payload_sha256(
            target["channel_receipt"]
        ):
            raise ValueError("target channel receipt does not match its payload")
        target_payload = {
            key: value for key, value in target.items() if key != "target_receipt_sha256"
        }
        if target["target_receipt_sha256"] != canonical_payload_sha256(target_payload):
            raise ValueError("target receipt does not match its payload")

    def derived_rank(result: dict[str, object]) -> dict[str, object]:
        candidate_ids = [str(value) for value in result["ordered_candidate_ids"]]
        if len(candidate_ids) != 40 or len(set(candidate_ids)) != 40:
            raise ValueError("each result must bind exactly 40 unique ordered candidate IDs")
        truth_id = str(result["truth_candidate_id"])
        score_payload = result["scores"]
        required_score_keys = {
            "semantic",
            "raw_ID_agreement",
            "mask_only_Dice",
            "channel_count",
            "smoothing_sigma_px",
        }
        if set(score_payload) != required_score_keys:
            raise ValueError("raw score payload does not match the frozen five-field schema")
        arrays = {
            name: np.asarray(score_payload[name], dtype=np.float64)
            for name in ("semantic", "raw_ID_agreement", "mask_only_Dice")
        }
        if any(
            values.shape != (40,)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            for values in arrays.values()
        ):
            raise ValueError("each raw semantic and ablation landscape must contain 40 finite [0,1] values")
        channel_count = score_payload["channel_count"]
        expected_channel_count = result["target"]["channel_receipt"]["channel_count"]
        if type(channel_count) is not int or channel_count != expected_channel_count or channel_count <= 0:
            raise ValueError("score channel count does not match the target-defined channel receipt")
        expected_sigma = SMOOTHING_SIGMA_UM / float(result["target"]["pixel_pitch_um"])
        if not np.isclose(
            float(score_payload["smoothing_sigma_px"]), expected_sigma, rtol=0.0, atol=0.0
        ):
            raise ValueError("score smoothing sigma does not equal 75 micrometres over target pitch")
        scores = arrays["semantic"]
        return rank_candidate_ids(scores, candidate_ids, truth_id)

    primary_rankings = []
    shuffled_rankings = []
    for index, (base, control) in enumerate(zip(primary, shuffled, strict=True)):
        validate_target(base["target"], index, paired_ids[index])
        target_index = shuffled_target_index(index)
        if (
            str(control["paired_view_group_id"]) != paired_ids[index]
            or control["candidate_bank_id"] != base["candidate_bank_id"]
            or control["ordered_candidate_ids"] != base["ordered_candidate_ids"]
            or control["truth_candidate_id"] != base["truth_candidate_id"]
        ):
            raise ValueError("the shuffled control must retain the exact original candidate bank and order")
        validate_target(control["target"], target_index, paired_ids[target_index])
        if int(control["target"]["source_case_index"]) != target_index or control["target"] != primary[
            target_index
        ]["target"]:
            raise ValueError("the shuffled target must be the exact (i+17) modulo 64 target receipt")
        primary_rankings.append(derived_rank(base))
        shuffled_rankings.append(derived_rank(control))

    top1 = np.asarray([bool(result["top1"]) for result in primary_rankings])
    ranks = np.asarray([int(result["true_rank"]) for result in primary_rankings])
    wins = np.asarray(
        [float(result["true_versus_decoy_win_fraction"]) for result in primary_rankings]
    )
    successes = int(top1.sum())
    wilson_low, wilson_high = wilson_interval(successes, 64)
    family_rates = {
        name: float(np.mean(top1[orientations == name])) for name in ORIENTATION_COUNTS
    }
    shuffled_top1 = np.asarray([bool(result["top1"]) for result in shuffled_rankings])
    shuffled_rr = np.asarray([float(result["reciprocal_rank"]) for result in shuffled_rankings])
    metrics = {
        "base_count": 64,
        "top1_successes": successes,
        "top1_rate": float(successes / 64),
        "top1_wilson_95": [float(wilson_low), float(wilson_high)],
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_true_rank": float(np.median(ranks)),
        "median_true_versus_decoy_win_fraction": float(np.median(wins)),
        "orientation_top1_rate": family_rates,
        "shuffled_original_truth_top1_rate": float(np.mean(shuffled_top1)),
        "shuffled_original_truth_mean_reciprocal_rank": float(np.mean(shuffled_rr)),
    }
    gates = {
        "exact_controls": all(record.get("passed") is True for record in exact_controls.values()),
        "top1_rate": metrics["top1_rate"] >= 0.80,
        "top1_wilson_lower": wilson_low >= 0.65,
        "median_true_rank": metrics["median_true_rank"] <= 1.0,
        "median_win_fraction": metrics["median_true_versus_decoy_win_fraction"] >= 0.95,
        "orientation_rates": all(rate >= 0.60 for rate in family_rates.values()),
        "shuffled_top1": metrics["shuffled_original_truth_top1_rate"] <= 0.10,
        "shuffled_mrr": metrics["shuffled_original_truth_mean_reciprocal_rank"] <= 0.15,
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "primary_rankings": primary_rankings,
        "shuffled_rankings": shuffled_rankings,
        "passed": all(gates.values()),
    }
